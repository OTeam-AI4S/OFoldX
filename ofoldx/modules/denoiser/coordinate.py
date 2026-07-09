r"""
Coordinate denoising networks.

The coordinate denoiser is the diffusion-model score network: given noisy atom coords and an
EDM ``noise_level``, it conditions the token single/pair representations, refines them through an
atom-encoder/token-refiner/atom-decoder stack, and returns denoised coords. This module holds the
single-head base variant; multi-head forks live in the model families.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from oops import LayerNorm
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from ..atom import (
    AtomContext,
    AtomToTokenOutput,
    AttentionLayout,
    CoordinateAtomToTokenEncoder,
    LocalAtomGeometry,
    LocalAtomWindowInfo,
    TokenToAtomDecoder,
)
from ..attention import DenoisingTokenEncoder
from ..diffusion.conditioning import condition_pair, condition_single
from ..diffusion.edm import EdmPreconditioner
from ..diffusion.types import CoordinateDenoisingInput
from ..embedding.fourier import FrozenFourierEmbedding
from ..embedding.position import RelativePositionBuckets, RelativePositionEmbedder
from ..transition import Transition


@dataclass(slots=True)
class DenoisingConditioning:
    r"""
    Single and pair conditioning tensors produced for one coordinate-denoising step.

    Bundles the two representations the token refiner consumes: the noise-conditioned single
    (carrying a per-sample axis) and the relative-position-augmented pair.

    Attributes:
        denoising_single: Conditioned token single representation, shape
            `[..., num_samples, num_tokens, single_size]`.
        denoising_pair: Conditioned token pair representation, shape
            `[..., num_tokens, num_tokens, pair_size]`.

    Examples:
        >>> import torch
        >>> from ofoldx.modules.denoiser.coordinate import DenoisingConditioning
        >>> cond = DenoisingConditioning(
        ...     denoising_single=torch.zeros(1, 5, 384),
        ...     denoising_pair=torch.zeros(5, 5, 128),
        ... )
        >>> cond.denoising_single.shape, cond.denoising_pair.shape
        (torch.Size([1, 5, 384]), torch.Size([5, 5, 128]))
    """

    denoising_single: Tensor
    denoising_pair: Tensor


@dataclass(slots=True)
class _TokenRefinementInput:
    r"""
    Internal atom-to-token refinement bundle.

    Attributes:
        atom_features: Per-atom input features.
        atom_to_token: Token index for each atom.
        single: Token single representation.
        denoising_single: Conditioned token single representation.
        denoising_pair: Conditioned token pair representation.
        reference_positions: Centered reference-conformer atom coords.
        reference_charges: Per-atom reference formal charges.
        atom_mask: Boolean atom validity mask.
        noisy_coords: Noisy atom coords to refine.
        local_geometry: Optional local atom-pair geometry.
    """

    atom_features: Tensor
    atom_to_token: Tensor
    single: Tensor
    denoising_single: Tensor
    denoising_pair: Tensor
    reference_positions: Tensor
    reference_charges: Tensor
    atom_mask: Tensor
    noisy_coords: Tensor
    local_geometry: LocalAtomGeometry | None


class CoordinateDenoiser(nn.Module):
    r"""
    EDM coordinate denoiser with atom-level encoder/decoder blocks.

    Runs the full diffusion score network for one denoising step: an
    [`EdmPreconditioner`][ofoldx.modules.diffusion.edm.EdmPreconditioner] scales the noisy
    coords, a [`DenoisingConditioner`][ofoldx.modules.denoiser.coordinate.DenoisingConditioner]
    conditions the token single/pair on the noise level, a
    [`CoordinateAtomToTokenEncoder`][ofoldx.modules.atom.CoordinateAtomToTokenEncoder] lifts
    atoms to tokens, a [`DenoisingTokenRefiner`][ofoldx.modules.denoiser.coordinate.DenoisingTokenRefiner]
    refines the token states, and a
    [`TokenToAtomDecoder`][ofoldx.modules.atom.TokenToAtomDecoder] projects back to a
    per-atom coordinate update that the preconditioner combines into the denoised coords.

    This is the single-head base variant; the ``_build_*`` / ``_decode_head_type`` hooks let
    multi-head forks (e.g. in the model families) swap the refiner and decoder without
    reimplementing ``forward``.

    Args:
        num_encoder_blocks: Number of token-refiner attention blocks.
        num_encoder_attention_heads: Number of attention heads per token-refiner block.
        sigma_data: EDM data-scale hyperparameter shared by the preconditioner and conditioner.
        atom_hidden_size: Width of the per-atom hidden channel.
        atom_pair_hidden_size: Width of the per-atom-pair hidden channel.
        token_hidden_size: Width of the refined token hidden channel.
        single_size: Width of the token single representation.
        pair_size: Width of the token pair representation.
        token_feature_size: Width of the raw per-token conditioning features.
        atom_feature_size: Width of the raw per-atom input features.
        num_atom_encoder_blocks: Number of atom-encoder attention blocks.
        num_atom_encoder_attention_heads: Number of attention heads per atom-encoder block.
        num_atom_decoder_blocks: Number of atom-decoder attention blocks.
        num_atom_decoder_attention_heads: Number of attention heads per atom-decoder block.
        num_query_atoms: Query window size for local atom attention.
        num_key_atoms: Key window size for local atom attention.
        blocks_per_checkpoint: Blocks per gradient-checkpoint segment; ``None`` disables
            checkpointing.
        operator_backend: Backend selector passed to the attention/transition operators.

    Examples:
        >>> denoiser = CoordinateDenoiser(
        ...     num_encoder_blocks=1,
        ...     num_atom_encoder_blocks=1,
        ...     num_atom_decoder_blocks=1,
        ... )
        >>> isinstance(denoiser.preconditioner, EdmPreconditioner)
        True
    """

    def __init__(
        self,
        num_encoder_blocks: int = 24,
        num_encoder_attention_heads: int = 16,
        *,
        sigma_data: float = 16.0,
        atom_hidden_size: int = 128,
        atom_pair_hidden_size: int = 16,
        token_hidden_size: int = 768,
        single_size: int = 384,
        pair_size: int = 128,
        token_feature_size: int = 449,
        atom_feature_size: int = 385,
        embed_relative_position: bool = True,
        num_atom_encoder_blocks: int = 3,
        num_atom_encoder_attention_heads: int = 4,
        num_atom_decoder_blocks: int = 3,
        num_atom_decoder_attention_heads: int = 4,
        num_query_atoms: int = 32,
        num_key_atoms: int = 128,
        blocks_per_checkpoint: int | None = None,
        operator_backend: str = "auto",
    ) -> None:
        super().__init__()
        self.blocks_per_checkpoint = blocks_per_checkpoint
        self.preconditioner = EdmPreconditioner(sigma_data=sigma_data)
        self.conditioner = DenoisingConditioner(
            pair_input_size=2 * pair_size if embed_relative_position else pair_size + 139,
            pair_size=pair_size,
            single_conditioning_size=single_size + token_feature_size,
            single_size=single_size,
            sigma_data=sigma_data,
            embed_relative_position=embed_relative_position,
        )
        self.atom_encoder = CoordinateAtomToTokenEncoder(
            token_hidden_size=token_hidden_size,
            atom_hidden_size=atom_hidden_size,
            atom_pair_hidden_size=atom_pair_hidden_size,
            single_size=single_size,
            pair_size=pair_size,
            num_blocks=num_atom_encoder_blocks,
            num_attention_heads=num_atom_encoder_attention_heads,
            num_query_atoms=num_query_atoms,
            num_key_atoms=num_key_atoms,
            atom_feature_size=atom_feature_size,
            operator_backend=operator_backend,
        )
        self.token_refiner = self._build_token_refiner(
            num_blocks=num_encoder_blocks,
            num_attention_heads=num_encoder_attention_heads,
            token_hidden_size=token_hidden_size,
            single_size=single_size,
            pair_size=pair_size,
            blocks_per_checkpoint=blocks_per_checkpoint,
            operator_backend=operator_backend,
        )
        self._build_atom_decoder(
            token_hidden_size=token_hidden_size,
            atom_hidden_size=atom_hidden_size,
            atom_pair_hidden_size=atom_pair_hidden_size,
            num_blocks=num_atom_decoder_blocks,
            num_attention_heads=num_atom_decoder_attention_heads,
            num_query_atoms=num_query_atoms,
            num_key_atoms=num_key_atoms,
            operator_backend=operator_backend,
        )

    def _build_token_refiner(self, **kwargs: Any) -> DenoisingTokenRefiner:
        r"""
        Build the single-head token refiner; the multi-head fork overrides this.

        Args:
            **kwargs: Keyword arguments forwarded to
                [`DenoisingTokenRefiner`][ofoldx.modules.denoiser.coordinate.DenoisingTokenRefiner].

        Returns:
            The constructed token refiner.
        """

        return DenoisingTokenRefiner(**kwargs)

    @staticmethod
    def _new_atom_decoder(
        *,
        token_hidden_size: int,
        atom_hidden_size: int,
        atom_pair_hidden_size: int,
        num_blocks: int,
        num_attention_heads: int,
        num_query_atoms: int,
        num_key_atoms: int,
        operator_backend: str,
    ) -> TokenToAtomDecoder:
        r"""
        Construct a token-to-atom decoder from explicit width/head/window arguments.

        Args:
            token_hidden_size: Width of the refined token hidden channel.
            atom_hidden_size: Width of the per-atom hidden channel.
            atom_pair_hidden_size: Width of the per-atom-pair hidden channel.
            num_blocks: Number of atom-decoder attention blocks.
            num_attention_heads: Number of attention heads per block.
            num_query_atoms: Query window size for local atom attention.
            num_key_atoms: Key window size for local atom attention.
            operator_backend: Backend selector passed to the operators.

        Returns:
            The constructed [`TokenToAtomDecoder`][ofoldx.modules.atom.TokenToAtomDecoder].
        """

        return TokenToAtomDecoder(
            token_hidden_size=token_hidden_size,
            atom_hidden_size=atom_hidden_size,
            atom_pair_hidden_size=atom_pair_hidden_size,
            num_blocks=num_blocks,
            num_attention_heads=num_attention_heads,
            num_query_atoms=num_query_atoms,
            num_key_atoms=num_key_atoms,
            operator_backend=operator_backend,
        )

    def _build_atom_decoder(self, **kwargs: Any) -> None:
        r"""
        Build the single-head atom decoder and store it as ``self.atom_decoder``.

        The multi-head fork overrides this to register several head-specific decoders.

        Args:
            **kwargs: Keyword arguments forwarded to `_new_atom_decoder`.
        """

        self.atom_decoder = self._new_atom_decoder(**kwargs)

    def _decode_head_type(self) -> str | None:
        r"""
        Return the head key passed to the token refiner / decoder.

        Returns:
            ``None`` for the single-head base; a head-name string in multi-head forks.
        """

        return None

    def _select_atom_decoder(self) -> TokenToAtomDecoder:
        r"""
        Return the atom decoder to run for the current head.

        Returns:
            The single [`TokenToAtomDecoder`][ofoldx.modules.atom.TokenToAtomDecoder] of
            the base variant.
        """

        return self.atom_decoder

    def build_local_geometry(self, reference_positions: Tensor, reference_space_id: Tensor) -> LocalAtomGeometry:
        r"""
        Build local atom-pair geometry using the denoising atom-encoder window layout.

        Delegates to the atom encoder so the geometry (window offsets, pair features) matches the
        layout the encoder will attend over. Precompute it once and pass it through the input
        bundle's ``local_geometry`` to enable windowed local attention.

        Args:
            reference_positions: Centered reference-conformer atom coords, shape `[..., num_atoms, 3]`.
            reference_space_id: Per-atom conformer/space identifier used to gate cross-window pairs,
                shape `[..., num_atoms]`.

        Returns:
            A [`LocalAtomGeometry`][ofoldx.modules.atom.LocalAtomGeometry] describing the local
            atom-pair layout.

        Examples:
            >>> import torch  # doctest: +SKIP
            >>> denoiser = CoordinateDenoiser(num_encoder_blocks=1)  # doctest: +SKIP
            >>> geom = denoiser.build_local_geometry(  # doctest: +SKIP
            ...     torch.zeros(1, 8, 3), torch.zeros(1, 8, dtype=torch.long)
            ... )
        """

        return self.atom_encoder.build_local_geometry(reference_positions, reference_space_id)

    def predict_coordinate_update(self, inputs: CoordinateDenoisingInput) -> Tensor:
        r"""
        Predict the raw per-atom coordinate update, before EDM output preconditioning.

        Conditions the token single/pair on the noise level, refines the atom-encoded token states,
        and decodes them back to a per-atom update. `forward` scales the inputs and combines
        this update through the preconditioner; call this directly only when you need the
        un-preconditioned update.

        Args:
            inputs: Typed coordinate-denoising bundle. ``noisy_coords`` must carry a leading sample
                axis whose length matches the last dimension of ``noise_level``.

        Returns:
            Per-atom coordinate update, shape `[..., num_samples, num_atoms, 3]`.

        Raises:
            ValueError: If the last dimension of ``inputs.noise_level`` does not match the sample
                axis of ``inputs.noisy_coords``.

        Examples:
            >>> denoiser = CoordinateDenoiser(num_encoder_blocks=1)  # doctest: +SKIP
            >>> update = denoiser.predict_coordinate_update(inputs)  # doctest: +SKIP
        """

        num_samples = inputs.noisy_coords.shape[-3]
        if inputs.noise_level.shape[-1] != num_samples:
            raise ValueError(
                "noise_level's last dimension must match the sample axis, got "
                f"{inputs.noise_level.shape[-1]} and {num_samples}"
            )
        conditioning = self._condition(inputs)
        denoising_single = conditioning.denoising_single
        denoising_pair = conditioning.denoising_pair

        target_prefix = tuple(denoising_single.shape[:-2])
        single = _broadcast_prefix(inputs.single, target_prefix, tail_dims=2)
        denoising_pair = _broadcast_prefix(denoising_pair, target_prefix, tail_dims=3)
        refinement_input = _TokenRefinementInput(
            atom_features=inputs.atom_features,
            atom_to_token=inputs.atom_to_token,
            single=single,
            denoising_single=denoising_single,
            denoising_pair=denoising_pair,
            reference_positions=inputs.reference_positions,
            reference_charges=inputs.reference_charges,
            atom_mask=inputs.atom_mask,
            noisy_coords=inputs.noisy_coords,
            local_geometry=inputs.local_geometry,
        )
        atom_encoder_output, token_hidden, atom_mask = self.token_refiner(
            self.atom_encoder,
            refinement_input,
            decode_head_type=self._decode_head_type(),
        )
        return self._decode_atoms(
            self._select_atom_decoder(),
            atom_to_token=inputs.atom_to_token,
            token_hidden=token_hidden,
            atom_encoder_output=atom_encoder_output,
            atom_mask=atom_mask,
            attention_layout="dense" if inputs.local_geometry is None else "local",
        )

    def _checkpoint_enabled(self) -> bool:
        r"""
        Report whether gradient checkpointing should wrap the sub-blocks.

        Returns:
            ``True`` when ``blocks_per_checkpoint`` is set and grad is enabled.
        """

        return self.blocks_per_checkpoint is not None and torch.is_grad_enabled()

    def _condition(self, inputs: CoordinateDenoisingInput) -> DenoisingConditioning:
        r"""
        Run the conditioner, optionally under a gradient checkpoint.

        Args:
            inputs: The coordinate-denoising input bundle.

        Returns:
            The [`DenoisingConditioning`][ofoldx.modules.denoiser.coordinate.DenoisingConditioning]
            produced by the conditioner (rebuilt from tensors when checkpointed).
        """

        if not self._checkpoint_enabled():
            return self.conditioner(
                inputs.noise_level,
                token_embedding=inputs.token_embedding,
                single=inputs.single,
                pair=inputs.pair,
                relative_features=inputs.relative_features,
            )

        def run(
            noise_level: Tensor,
            token_embedding: Tensor,
            single: Tensor,
            pair: Tensor,
            relative_features: RelativePositionBuckets,
        ) -> tuple[Tensor, Tensor]:
            r"""
            Checkpoint wrapper running ``self.conditioner`` and returning its tensor fields as a
            ``(denoising_single, denoising_pair)`` tuple (checkpoint cannot save the dataclass output).

            Args:
                noise_level: Per-sample noise level fed to the conditioner.
                token_embedding: Trunk token embedding.
                single: Token single representation.
                pair: Token pair representation.
                relative_features: Relative-position bucket features for pair conditioning.

            Returns:
                ``(denoising_single, denoising_pair)`` tensors from the conditioner output.
            """

            output = self.conditioner(
                noise_level,
                token_embedding=token_embedding,
                single=single,
                pair=pair,
                relative_features=relative_features,
            )
            return output.denoising_single, output.denoising_pair

        denoising_single, denoising_pair = checkpoint(
            run,
            inputs.noise_level,
            inputs.token_embedding,
            inputs.single,
            inputs.pair,
            inputs.relative_features,
            use_reentrant=False,
        )
        return DenoisingConditioning(denoising_single=denoising_single, denoising_pair=denoising_pair)

    def _decode_atoms(
        self,
        atom_decoder: TokenToAtomDecoder,
        *,
        atom_to_token: Tensor,
        token_hidden: Tensor,
        atom_encoder_output: AtomToTokenOutput,
        atom_mask: Tensor | None,
        attention_layout: AttentionLayout,
    ) -> Tensor:
        r"""
        Decode refined token states back to a per-atom coordinate update.

        Runs ``atom_decoder`` directly, or under a gradient checkpoint (flattening/rebuilding the
        [`AtomContext`][ofoldx.modules.atom.AtomContext] and encoding a ``None`` mask as an
        empty tensor, since checkpoint cannot save ``None`` or dataclasses).

        Args:
            atom_decoder: The token-to-atom decoder to run.
            atom_to_token: Token index for each atom.
            token_hidden: Refined per-token hidden states.
            atom_encoder_output: Output of the atom encoder providing atom queries and context.
            atom_mask: Boolean atom validity mask, or ``None`` when the decoder rebuilds its own.
            attention_layout: ``"dense"`` for full attention or ``"local"`` for windowed attention.

        Returns:
            Per-atom coordinate update, shape `[..., num_atoms, 3]`.
        """

        if not self._checkpoint_enabled():
            return atom_decoder(
                atom_to_token=atom_to_token,
                token_hidden=token_hidden,
                atom_query=atom_encoder_output.atom_query,
                atom_context=atom_encoder_output.atom_context,
                atom_mask=atom_mask,
                attention_layout=attention_layout,
            )

        context = atom_encoder_output.atom_context
        if atom_mask is None:
            atom_mask = context.conditioning.new_empty(0)

        def run(
            token_hidden: Tensor,
            atom_query: Tensor,
            context_pair: Tensor,
            context_conditioning: Tensor,
            atom_mask: Tensor,
        ) -> Tensor:
            r"""
            Checkpoint wrapper running ``atom_decoder`` on plain tensors, rebuilding the [`AtomContext`][]
            from ``context_pair``/``context_conditioning`` and treating an empty ``atom_mask`` as ``None``.

            Args:
                token_hidden: Refined per-token hidden states.
                atom_query: Per-atom query features from the atom encoder.
                context_pair: Atom-context pair tensor to rebuild the [`AtomContext`][].
                context_conditioning: Atom-context conditioning tensor to rebuild the [`AtomContext`][].
                atom_mask: Boolean atom validity mask, empty tensor standing in for ``None``.

            Returns:
                Per-atom coordinate update, shape `[..., num_atoms, 3]`.
            """

            return atom_decoder(
                atom_to_token=atom_to_token,
                token_hidden=token_hidden,
                atom_query=atom_query,
                atom_context=AtomContext(pair=context_pair, conditioning=context_conditioning),
                atom_mask=None if atom_mask.numel() == 0 else atom_mask,
                attention_layout=attention_layout,
            )

        return checkpoint(
            run,
            token_hidden,
            atom_encoder_output.atom_query,
            context.pair,
            context.conditioning,
            atom_mask,
            use_reentrant=False,
        )

    def forward(self, inputs: CoordinateDenoisingInput) -> Tensor:
        r"""
        Denoise atom coords from a typed coordinate-denoising input bundle.

        Applies EDM input scaling to the noisy coords, predicts the raw coordinate update, and
        combines it back through the preconditioner to yield the denoised coords. A missing sample
        axis (``noisy_coords.dim() == atom_features.dim()``) is inserted for the forward pass and
        squeezed back out on return.

        Args:
            inputs: Typed coordinate-denoising bundle carrying atom features, token
                single/pair representations, the EDM ``noise_level``, and the ``noisy_coords`` to
                denoise.

        Returns:
            Denoised atom coords, shape `[..., num_atoms, 3]` (or with a
            `[..., num_samples, num_atoms, 3]` sample axis when the input carried one).

        Examples:
            >>> denoiser = CoordinateDenoiser(num_encoder_blocks=1)  # doctest: +SKIP
            >>> denoised = denoiser(inputs)  # doctest: +SKIP
        """

        dtype = inputs.atom_features.dtype

        noisy_coords = inputs.noisy_coords
        squeeze_sample_axis = noisy_coords.dim() == inputs.atom_features.dim()
        if squeeze_sample_axis:
            noisy_coords = noisy_coords.unsqueeze(-3)

        scaled_noisy_coords = self.preconditioner.scale_noisy_coords(noisy_coords, inputs.noise_level)

        scaled_inputs = CoordinateDenoisingInput(
            atom_features=inputs.atom_features,
            atom_to_token=inputs.atom_to_token,
            token_embedding=inputs.token_embedding,
            single=inputs.single,
            pair=inputs.pair,
            relative_features=inputs.relative_features,
            noise_level=inputs.noise_level,
            reference_positions=inputs.reference_positions,
            reference_charges=inputs.reference_charges,
            atom_mask=inputs.atom_mask.to(dtype=dtype),
            noisy_coords=scaled_noisy_coords,
            local_geometry=inputs.local_geometry,
        )
        coordinate_update = self.predict_coordinate_update(scaled_inputs)

        denoised_coords = self.preconditioner.combine_update(noisy_coords, inputs.noise_level, coordinate_update)
        if squeeze_sample_axis and denoised_coords.shape[-3] == 1:
            denoised_coords = denoised_coords.squeeze(-3)
        return denoised_coords


class DenoisingConditioner(nn.Module):
    r"""
    Condition the token single and pair representations for one coordinate-denoising step.

    Augments the pair representation with relative-position embeddings and fuses the
    Fourier-embedded EDM log-noise into the single representation, so downstream refinement is
    aware of both sequence geometry and the current noise level. Each branch is a
    norm/projection followed by two [`Transition`][ofoldx.modules.transition.Transition]
    feed-forward blocks.

    Args:
        relative_position_size: Width of the relative-position feature bucket embedding input.
        pair_input_size: Width of the incoming (concatenated) pair representation.
        pair_size: Width of the conditioned pair representation.
        single_conditioning_size: Width of the concatenated single conditioning input.
        single_size: Width of the conditioned single representation.
        noise_embedding_size: Width of the Fourier noise embedding.
        sigma_data: EDM data-scale hyperparameter used to normalise the log-noise.

    Examples:
        >>> conditioner = DenoisingConditioner(pair_size=128, single_size=384)
        >>> conditioner.sigma_data
        16.0
    """

    def __init__(
        self,
        relative_position_size: int = 139,
        pair_input_size: int = 256,
        pair_size: int = 128,
        single_conditioning_size: int = 833,
        single_size: int = 384,
        noise_embedding_size: int = 256,
        sigma_data: float = 16.0,
        embed_relative_position: bool = True,
    ) -> None:
        super().__init__()
        self.sigma_data = sigma_data
        self.embed_relative_position = embed_relative_position

        self.relative_position_embedder = (
            RelativePositionEmbedder(relative_position_size, pair_size) if embed_relative_position else None
        )
        self.pair_input_norm = LayerNorm(pair_input_size, bias=False)
        self.pair_projection = nn.Linear(pair_input_size, pair_size, bias=False)
        self.pair_feed_forward_1 = Transition(pair_size, expansion=2)
        self.pair_feed_forward_2 = Transition(pair_size, expansion=2)

        self.single_conditioning_norm = LayerNorm(single_conditioning_size, bias=False)
        self.single_conditioning_projection = nn.Linear(single_conditioning_size, single_size, bias=False)
        self.noise_embedding = FrozenFourierEmbedding(noise_embedding_size)
        self.noise_norm = LayerNorm(noise_embedding_size, bias=False)
        self.noise_projection = nn.Linear(noise_embedding_size, single_size, bias=False)
        self.single_feed_forward_1 = Transition(single_size, expansion=2)
        self.single_feed_forward_2 = Transition(single_size, expansion=2)

    def forward(
        self,
        noise_level: Tensor,
        *,
        token_embedding: Tensor,
        single: Tensor,
        pair: Tensor,
        relative_features: RelativePositionBuckets,
    ) -> DenoisingConditioning:
        r"""
        Condition the token single and pair on the noise level and token embedding.

        The pair branch adds relative-position embeddings to the incoming pair; the single branch
        fuses the Fourier-embedded log-noise (``log(noise_level / sigma_data) / 4``) into the single
        update and broadcasts a per-sample axis.

        Args:
            noise_level: Per-sample EDM noise level, shape `[..., num_samples]`.
            token_embedding: Token embedding before trunk updates, shape `[..., num_tokens, *]`.
            single: Token single representation, shape `[..., num_tokens, single_size]`.
            pair: Token pair representation, shape `[..., num_tokens, num_tokens, pair_input_size]`.
            relative_features: Relative-position feature bundle indexing the pair embedding.

        Returns:
            A [`DenoisingConditioning`][ofoldx.modules.denoiser.coordinate.DenoisingConditioning]
            whose ``denoising_single`` is `[..., num_samples, num_tokens, single_size]` and
            ``denoising_pair`` is `[..., num_tokens, num_tokens, pair_size]`.

        Examples:
            >>> conditioner = DenoisingConditioner()  # doctest: +SKIP
            >>> cond = conditioner(  # doctest: +SKIP
            ...     noise_level, token_embedding=token_embedding, single=single,
            ...     pair=pair, relative_features=relative_features,
            ... )
        """

        relative_position_conditioning = (
            self.relative_position_embedder(relative_features)
            if self.relative_position_embedder is not None
            else _relative_position_one_hot(relative_features)
        )
        denoising_pair = condition_pair(
            pair,
            relative_position_conditioning,
            norm=self.pair_input_norm,
            projection=self.pair_projection,
            transitions=[self.pair_feed_forward_1, self.pair_feed_forward_2],
        )
        denoising_single, _ = condition_single(
            single,
            token_embedding,
            torch.log(noise_level / self.sigma_data) / 4,
            norm=self.single_conditioning_norm,
            projection=self.single_conditioning_projection,
            noise_fourier=self.noise_embedding,
            noise_norm=self.noise_norm,
            noise_projection=self.noise_projection,
            transitions=[self.single_feed_forward_1, self.single_feed_forward_2],
            sample_axis=True,
        )
        return DenoisingConditioning(denoising_single=denoising_single, denoising_pair=denoising_pair)


def _relative_position_one_hot(buckets: RelativePositionBuckets) -> Tensor:
    r"""
    Materialise AlphaFold 3's raw relative-position feature vector.

    The official DeepMind diffusion head concatenates ``pair_embedding`` with the
    raw relative features (residue offset, token offset, same entity, chain
    offset), whereas Protenix-style checkpoints first project those features to
    ``pair_size``. This helper is used only for the official-AF3 conditioner.
    """

    residue = F.one_hot(buckets.residue_bucket.long(), num_classes=buckets.residue_size)
    token = F.one_hot(buckets.token_bucket.long(), num_classes=buckets.residue_size)
    same_entity = buckets.same_entity.unsqueeze(-1)
    chain = F.one_hot(buckets.chain_bucket.long(), num_classes=buckets.chain_size)
    return torch.cat([residue, token, same_entity, chain], dim=-1).to(dtype=buckets.dtype)


class DenoisingTokenRefiner(nn.Module):
    r"""
    Refine atom-encoded token states with denoising single and pair conditioning.

    Adds the projected, normed single conditioning into the atom-encoder token hidden states, runs
    them through a [`DenoisingTokenEncoder`][ofoldx.modules.attention.DenoisingTokenEncoder]
    stack biased by the pair conditioning, and returns the refined states together with the atom
    encoder output and the decoder atom mask. Handles both the dense (samples flattened into the
    batch) and local-windowed (samples carried structurally) atom regimes.

    This is the single-head base variant; the ``_build_token_encoder`` hook lets multi-head forks
    swap the encoder.

    Args:
        num_blocks: Number of token-encoder attention blocks.
        token_hidden_size: Width of the token hidden channel.
        single_size: Width of the single conditioning representation.
        pair_size: Width of the pair conditioning representation.
        num_attention_heads: Number of attention heads per token-encoder block.
        blocks_per_checkpoint: Blocks per gradient-checkpoint segment; ``None`` disables
            checkpointing.
        operator_backend: Backend selector passed to the attention operators.

    Examples:
        >>> refiner = DenoisingTokenRefiner(num_blocks=1)
        >>> from ofoldx.modules.attention import DenoisingTokenEncoder
        >>> isinstance(refiner.token_encoder, DenoisingTokenEncoder)
        True
    """

    def __init__(
        self,
        num_blocks: int = 24,
        token_hidden_size: int = 768,
        single_size: int = 384,
        pair_size: int = 128,
        num_attention_heads: int = 16,
        blocks_per_checkpoint: int | None = None,
        operator_backend: str = "auto",
    ) -> None:
        super().__init__()
        self.blocks_per_checkpoint = blocks_per_checkpoint
        self.single_norm = LayerNorm(single_size, bias=False)
        self.single_to_token_projection = nn.Linear(single_size, token_hidden_size, bias=False)
        self._build_token_encoder(
            num_blocks=num_blocks,
            num_attention_heads=num_attention_heads,
            token_hidden_size=token_hidden_size,
            single_size=single_size,
            pair_size=pair_size,
            blocks_per_checkpoint=blocks_per_checkpoint,
            operator_backend=operator_backend,
        )
        self.token_norm = LayerNorm(token_hidden_size, bias=False)

    @staticmethod
    def _new_token_encoder(
        *,
        num_blocks: int,
        num_attention_heads: int,
        token_hidden_size: int,
        single_size: int,
        pair_size: int,
        blocks_per_checkpoint: int | None,
        operator_backend: str,
    ) -> DenoisingTokenEncoder:
        r"""
        Construct a denoising token encoder from explicit width/head/depth arguments.

        Args:
            num_blocks: Number of token-encoder attention blocks.
            num_attention_heads: Number of attention heads per block.
            token_hidden_size: Width of the token hidden channel.
            single_size: Width of the single conditioning (passed as ``conditioning_size``).
            pair_size: Width of the pair conditioning representation.
            blocks_per_checkpoint: Blocks per gradient-checkpoint segment; ``None`` disables it.
            operator_backend: Backend selector passed to the attention operators.

        Returns:
            The constructed
            [`DenoisingTokenEncoder`][ofoldx.modules.attention.DenoisingTokenEncoder].
        """

        return DenoisingTokenEncoder(
            num_blocks=num_blocks,
            num_attention_heads=num_attention_heads,
            token_hidden_size=token_hidden_size,
            conditioning_size=single_size,
            pair_size=pair_size,
            blocks_per_checkpoint=blocks_per_checkpoint,
            operator_backend=operator_backend,
        )

    def _build_token_encoder(self, *, num_blocks: int, **kwargs: Any) -> None:
        r"""
        Build the single-head token encoder and store it as ``self.token_encoder``.

        The multi-head fork overrides this to register several head-specific encoders.

        Args:
            num_blocks: Number of token-encoder attention blocks.
            **kwargs: Remaining keyword arguments forwarded to `_new_token_encoder`.
        """

        self.token_encoder = self._new_token_encoder(num_blocks=num_blocks, **kwargs)

    def forward(
        self,
        atom_encoder: CoordinateAtomToTokenEncoder,
        inputs: _TokenRefinementInput,
        *,
        decode_head_type: str | None = None,
    ) -> tuple[AtomToTokenOutput, Tensor, Tensor | None]:
        r"""
        Encode atoms, refine the token states, and return the decoder inputs.

        Two atom regimes share one token-refinement tail. With dense full attention
        (``local_geometry is None``) the sample axis is broadcast and flattened into the
        batch so every sample is an independent atom graph, and the flattened reference
        mask is handed back for the dense decoder. With local windowed attention the
        atoms stay structured and shared across samples -- the windowed encoder carries
        the sample axis through ``noisy_coords`` -- so nothing is broadcast, the refined
        states are reshaped back to `[..., num_tokens, hidden]`, and the decoder rebuilds
        its own mask (``None`` here).

        Args:
            atom_encoder: The atom-to-token encoder to run.
            inputs: Internal refinement bundle of atom features and token conditioning.
            decode_head_type: Head key forwarded to the token encoder; ``None`` for the base variant.

        Returns:
            A tuple of the atom-encoder output, the refined token hidden states of shape
            `[..., num_tokens, token_hidden_size]`, and the decoder atom mask (a flattened mask in
            the dense regime, ``None`` in the local regime).

        Examples:
            >>> refiner = DenoisingTokenRefiner(num_blocks=1)  # doctest: +SKIP
            >>> atom_output, token_hidden, atom_mask = refiner(  # doctest: +SKIP
            ...     atom_encoder, inputs, decode_head_type=None
            ... )
        """
        target_prefix = tuple(inputs.denoising_single.shape[:-2])
        num_tokens = inputs.denoising_single.shape[-2]
        reference_charges = _reference_charges_with_channel(inputs.reference_charges, inputs.reference_positions)
        denoising_single = _flatten_prefix(inputs.denoising_single, tail_dims=2)
        denoising_pair = _flatten_prefix(
            _broadcast_prefix(inputs.denoising_pair, target_prefix, tail_dims=3), tail_dims=3
        )

        decoder_atom_mask: Tensor | None
        restore_prefix: tuple[int, ...] | None
        if inputs.local_geometry is None:
            atom_mask = _flatten_broadcast(inputs.atom_mask, target_prefix, tail_dims=1)
            atom_encoder_output = self._encode_atoms(
                atom_encoder,
                atom_features=_flatten_broadcast(inputs.atom_features, target_prefix, tail_dims=2),
                reference_positions=_flatten_broadcast(inputs.reference_positions, target_prefix, tail_dims=2),
                reference_charges=torch.arcsinh(_flatten_broadcast(reference_charges, target_prefix, tail_dims=2)),
                reference_mask=atom_mask,
                single=_flatten_broadcast(inputs.single, target_prefix, tail_dims=2),
                pair=denoising_pair,
                noisy_coords=_flatten_broadcast(inputs.noisy_coords, target_prefix, tail_dims=2),
                atom_mask=atom_mask,
                atom_to_token=_flatten_broadcast(inputs.atom_to_token, target_prefix, tail_dims=1),
                num_tokens=num_tokens,
            )
            decoder_atom_mask = atom_mask
            restore_prefix = None
        else:
            local_geometry = _broadcast_local_geometry(inputs.local_geometry, target_prefix)
            atom_encoder_output = self._encode_atoms(
                atom_encoder,
                atom_features=_broadcast_prefix(inputs.atom_features, target_prefix, tail_dims=2),
                reference_positions=_broadcast_prefix(inputs.reference_positions, target_prefix, tail_dims=2),
                reference_charges=torch.arcsinh(_broadcast_prefix(reference_charges, target_prefix, tail_dims=2)),
                reference_mask=_broadcast_prefix(inputs.atom_mask, target_prefix, tail_dims=1),
                single=_broadcast_prefix(inputs.single, target_prefix, tail_dims=2),
                pair=_broadcast_prefix(inputs.denoising_pair, target_prefix, tail_dims=3),
                noisy_coords=inputs.noisy_coords,
                atom_mask=_broadcast_prefix(inputs.atom_mask, target_prefix, tail_dims=1),
                atom_to_token=_broadcast_prefix(inputs.atom_to_token, target_prefix, tail_dims=1),
                num_tokens=num_tokens,
                local_geometry=local_geometry,
            )
            decoder_atom_mask = None
            restore_prefix = target_prefix

        token_hidden = _flatten_prefix(atom_encoder_output.token_hidden, tail_dims=2)
        token_hidden = token_hidden + self.single_to_token_projection(self.single_norm(denoising_single))
        token_hidden = self._encode_tokens(
            token_hidden, denoising_single, denoising_pair, decode_head_type=decode_head_type
        )
        token_hidden = self.token_norm(token_hidden)
        if restore_prefix is not None:
            token_hidden = token_hidden.reshape(*restore_prefix, *token_hidden.shape[-2:])
        return atom_encoder_output, token_hidden, decoder_atom_mask

    def _checkpoint_enabled(self) -> bool:
        r"""
        Report whether gradient checkpointing should wrap the atom encoder.

        Returns:
            ``True`` when ``blocks_per_checkpoint`` is set and grad is enabled.
        """

        return self.blocks_per_checkpoint is not None and torch.is_grad_enabled()

    def _encode_atoms(
        self,
        atom_encoder: CoordinateAtomToTokenEncoder,
        *,
        atom_features: Tensor,
        reference_positions: Tensor,
        reference_charges: Tensor,
        reference_mask: Tensor,
        single: Tensor,
        pair: Tensor,
        noisy_coords: Tensor,
        atom_mask: Tensor,
        atom_to_token: Tensor,
        num_tokens: int,
        local_geometry: LocalAtomGeometry | None = None,
    ) -> AtomToTokenOutput:
        r"""
        Run the atom encoder, optionally under a gradient checkpoint.

        When checkpointed, the [`AtomToTokenOutput`][ofoldx.modules.atom.AtomToTokenOutput] is
        flattened into saveable tensors and rebuilt after, since checkpoint cannot save dataclasses.

        Args:
            atom_encoder: The atom-to-token encoder to run.
            atom_features: Per-atom input features.
            reference_positions: Centered reference-conformer atom coords.
            reference_charges: Per-atom reference charges (already ``arcsinh``-compressed).
            reference_mask: Boolean reference atom validity mask.
            single: Token single representation.
            pair: Conditioned token pair representation.
            noisy_coords: Noisy atom coords to encode.
            atom_mask: Boolean atom validity mask.
            atom_to_token: Token index for each atom.
            num_tokens: Number of tokens the atoms map onto.
            local_geometry: Optional local atom-pair geometry enabling windowed attention.

        Returns:
            The [`AtomToTokenOutput`][ofoldx.modules.atom.AtomToTokenOutput] of the encoder.
        """

        if not self._checkpoint_enabled():
            return atom_encoder(
                atom_features,
                reference_positions=reference_positions,
                reference_charges=reference_charges,
                reference_mask=reference_mask,
                single=single,
                pair=pair,
                noisy_coords=noisy_coords,
                atom_mask=atom_mask,
                atom_to_token=atom_to_token,
                num_tokens=num_tokens,
                local_geometry=local_geometry,
            )

        def run(
            atom_features: Tensor,
            reference_positions: Tensor,
            reference_charges: Tensor,
            reference_mask: Tensor,
            single: Tensor,
            pair: Tensor,
            noisy_coords: Tensor,
            atom_mask: Tensor,
            atom_to_token: Tensor,
        ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
            r"""
            Checkpoint wrapper running ``atom_encoder`` and flattening its [`AtomToTokenOutput`][] into a
            ``(token_hidden, atom_query, context_pair, context_conditioning)`` tuple of saveable tensors.

            Args:
                atom_features: Per-atom input features.
                reference_positions: Centered reference-conformer atom coords.
                reference_charges: Per-atom reference charges (already ``arcsinh``-compressed).
                reference_mask: Boolean reference atom validity mask.
                single: Token single representation.
                pair: Conditioned token pair representation.
                noisy_coords: Noisy atom coords to encode.
                atom_mask: Boolean atom validity mask.
                atom_to_token: Token index for each atom.

            Returns:
                ``(token_hidden, atom_query, context_pair, context_conditioning)`` tensors from the encoder
                output.
            """

            output = atom_encoder(
                atom_features,
                reference_positions=reference_positions,
                reference_charges=reference_charges,
                reference_mask=reference_mask,
                single=single,
                pair=pair,
                noisy_coords=noisy_coords,
                atom_mask=atom_mask,
                atom_to_token=atom_to_token,
                num_tokens=num_tokens,
                local_geometry=local_geometry,
            )
            return (
                output.token_hidden,
                output.atom_query,
                output.atom_context.pair,
                output.atom_context.conditioning,
            )

        token_hidden, atom_query, context_pair, context_conditioning = checkpoint(
            run,
            atom_features,
            reference_positions,
            reference_charges,
            reference_mask,
            single,
            pair,
            noisy_coords,
            atom_mask,
            atom_to_token,
            use_reentrant=False,
        )
        return AtomToTokenOutput(
            token_hidden=token_hidden,
            atom_query=atom_query,
            atom_context=AtomContext(pair=context_pair, conditioning=context_conditioning),
        )

    def _encode_tokens(
        self,
        token_hidden: Tensor,
        conditioning: Tensor,
        pair: Tensor,
        *,
        decode_head_type: str | None = None,
    ) -> Tensor:
        r"""
        Run the token encoder on the conditioned token states.

        The base variant ignores ``decode_head_type`` and runs its single encoder; multi-head forks
        override this to route to a head-specific encoder.

        Args:
            token_hidden: Per-token hidden states to refine.
            conditioning: Single conditioning representation used as attention conditioning.
            pair: Pair conditioning representation used as attention bias.
            decode_head_type: Head key; unused in the base variant.

        Returns:
            Refined token hidden states with the same shape as ``token_hidden``.
        """

        return self.token_encoder(token_hidden, conditioning, pair)


def _reference_charges_with_channel(reference_charges: Tensor, reference_positions: Tensor) -> Tensor:
    r"""
    Normalize scalar per-atom charges to `[..., N_atom, 1]`.

    Some family inputs store formal charges as `[..., N_atom]`; the atom conditioner projects
    a one-channel charge feature, so add the channel before sample-axis broadcasting.
    """

    if reference_charges.shape == reference_positions.shape[:-1]:
        return reference_charges.unsqueeze(-1)
    expected = (*reference_positions.shape[:-1], 1)
    if tuple(reference_charges.shape) == expected:
        return reference_charges
    if reference_charges.ndim == reference_positions.ndim - 1:
        return reference_charges.unsqueeze(-1)
    return reference_charges


def _broadcast_prefix(tensor: Tensor, target_prefix: tuple[int, ...], tail_dims: int) -> Tensor:
    r"""
    Broadcast a tensor's leading (prefix) dims to ``target_prefix``, keeping the last ``tail_dims``.

    The trailing ``tail_dims`` dimensions are treated as the payload shape and left untouched; the
    remaining leading dimensions are expanded to ``target_prefix`` (inserting size-1 axes as needed).

    Args:
        tensor: The tensor to broadcast.
        target_prefix: Desired leading-dimension shape.
        tail_dims: Number of trailing payload dimensions to preserve.

    Returns:
        A view of ``tensor`` with shape ``(*target_prefix, *tail_shape)``.

    Examples:
        >>> import torch
        >>> from ofoldx.modules.denoiser.coordinate import _broadcast_prefix
        >>> _broadcast_prefix(torch.zeros(3, 4), (5, 2), tail_dims=2).shape
        torch.Size([5, 2, 3, 4])
    """

    tail_shape = tuple(tensor.shape[-tail_dims:])
    target_shape = (*target_prefix, *tail_shape)
    if tuple(tensor.shape) == target_shape:
        return tensor

    source_prefix = tuple(tensor.shape[:-tail_dims])
    if len(source_prefix) <= len(target_prefix) and source_prefix == target_prefix[: len(source_prefix)]:
        view_shape = (*source_prefix, *((1,) * (len(target_prefix) - len(source_prefix))), *tail_shape)
        tensor = tensor.reshape(view_shape)
    return torch.broadcast_to(tensor, target_shape)


def _flatten_prefix(tensor: Tensor, tail_dims: int) -> Tensor:
    r"""
    Collapse all leading dims into one batch axis, keeping the last ``tail_dims``.

    Args:
        tensor: The tensor to flatten.
        tail_dims: Number of trailing payload dimensions to preserve.

    Returns:
        A view of shape ``(-1, *tail_shape)``.

    Examples:
        >>> import torch
        >>> from ofoldx.modules.denoiser.coordinate import _flatten_prefix
        >>> _flatten_prefix(torch.zeros(2, 3, 4, 5), tail_dims=2).shape
        torch.Size([6, 4, 5])
    """

    return tensor.reshape(-1, *tensor.shape[-tail_dims:])


def _flatten_broadcast(tensor: Tensor, target_prefix: tuple[int, ...], tail_dims: int) -> Tensor:
    r"""
    Broadcast the prefix to ``target_prefix`` then flatten it into a single batch axis.

    Composes `_broadcast_prefix` and `_flatten_prefix`; used to fold a broadcast sample
    axis into the batch so every sample becomes an independent atom graph.

    Args:
        tensor: The tensor to broadcast and flatten.
        target_prefix: Desired leading-dimension shape before flattening.
        tail_dims: Number of trailing payload dimensions to preserve.

    Returns:
        A view of shape ``(prod(target_prefix), *tail_shape)``.

    Examples:
        >>> import torch
        >>> from ofoldx.modules.denoiser.coordinate import _flatten_broadcast
        >>> _flatten_broadcast(torch.zeros(3, 4), (5, 2), tail_dims=2).shape
        torch.Size([10, 3, 4])
    """

    return _flatten_prefix(_broadcast_prefix(tensor, target_prefix, tail_dims), tail_dims)


def _broadcast_local_geometry(geometry: LocalAtomGeometry, target_prefix: tuple[int, ...]) -> LocalAtomGeometry:
    r"""
    Broadcast precomputed local atom geometry over denoising sample prefixes.

    Local geometry is built once from reference coordinates and therefore often carries only the batch prefix,
    while diffusion inputs may add a sample axis before the atom/window payload. Keep the window payload fixed and
    expand only the leading prefix so the geometry, attention mask, and token-pair bias share one shape contract.
    """

    info = geometry.window_info
    return LocalAtomGeometry(
        distance_vectors=_broadcast_prefix(geometry.distance_vectors, target_prefix, tail_dims=4),
        distance_mask=_broadcast_prefix(geometry.distance_mask, target_prefix, tail_dims=4),
        window_info=LocalAtomWindowInfo(
            attention_mask=_broadcast_prefix(info.attention_mask, target_prefix, tail_dims=3),
            query_padding=info.query_padding,
            key_padding_left=info.key_padding_left,
            key_padding_right=info.key_padding_right,
        ),
    )
