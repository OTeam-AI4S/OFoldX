r"""
Coordinate denoising networks.

The coordinate denoiser is the diffusion-model score network: given noisy atom coords and an
EDM ``noise_level``, it conditions the token single/pair representations, refines them through an
atom-encoder/token-refiner/atom-decoder stack, and returns denoised coords.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..atom import (
    AtomToTokenOutput,
    CoordinateAtomToTokenEncoder,
    LocalAtomGeometry,
    LocalAtomWindowInfo,
    TokenToAtomDecoder,
)
from ..attention import DenoisingTokenEncoder
from ..backend import LayerNorm
from ..diffusion.conditioning import condition_pair, condition_single
from ..diffusion.edm import EdmPreconditioner
from ..diffusion.types import CoordinateDenoisingInput
from ..embedding.fourier import FrozenFourierEmbedding
from ..embedding.position import RelativePositionBuckets, RelativePositionEmbedder
from ..transformer import CrossAttentionKeySource
from ..transition import Transition


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
        embed_relative_position: Whether to project relative-position buckets before pair conditioning.
        num_atom_encoder_blocks: Number of atom-encoder attention blocks.
        num_atom_encoder_attention_heads: Number of attention heads per atom-encoder block.
        num_atom_decoder_blocks: Number of atom-decoder attention blocks.
        num_atom_decoder_attention_heads: Number of attention heads per atom-decoder block.
        num_query_atoms: Query window size for local atom attention.
        num_key_atoms: Key window size for local atom attention.
        operator_backend: Backend selector passed to the attention/transition operators.
        cross_attention_key_source: Whether atom key/value normalization starts from the raw hidden state
            or the query-normalized state.

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
        operator_backend: str = "auto",
        cross_attention_key_source: CrossAttentionKeySource = "raw",
    ) -> None:
        super().__init__()
        self.preconditioner = EdmPreconditioner(sigma_data=sigma_data)
        self.conditioner = DenoisingConditioner(
            pair_input_size=2 * pair_size if embed_relative_position else pair_size + 139,
            pair_size=pair_size,
            single_conditioning_size=single_size + token_feature_size,
            single_size=single_size,
            sigma_data=sigma_data,
            embed_relative_position=embed_relative_position,
            operator_backend=operator_backend,
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
            cross_attention_key_source=cross_attention_key_source,
        )
        self.token_refiner = DenoisingTokenRefiner(
            num_blocks=num_encoder_blocks,
            num_attention_heads=num_encoder_attention_heads,
            token_hidden_size=token_hidden_size,
            single_size=single_size,
            pair_size=pair_size,
            operator_backend=operator_backend,
        )
        self.atom_decoder = TokenToAtomDecoder(
            token_hidden_size=token_hidden_size,
            atom_hidden_size=atom_hidden_size,
            atom_pair_hidden_size=atom_pair_hidden_size,
            num_blocks=num_atom_decoder_blocks,
            num_attention_heads=num_atom_decoder_attention_heads,
            num_query_atoms=num_query_atoms,
            num_key_atoms=num_key_atoms,
            operator_backend=operator_backend,
            cross_attention_key_source=cross_attention_key_source,
        )

    def build_local_geometry(
        self, reference_positions: Tensor, reference_space_id: Tensor, num_valid_atoms: int | None = None
    ) -> LocalAtomGeometry:
        r"""
        Build local atom-pair geometry using the denoising atom-encoder window layout.

        Delegates to the atom encoder so the geometry (window offsets, pair features) matches the
        layout the encoder will attend over. Precompute it once and pass it through the input
        bundle's ``local_geometry`` to enable windowed local attention.

        Args:
            reference_positions: Centered reference-conformer atom coords, shape `[..., num_atoms, 3]`.
            reference_space_id: Per-atom conformer/space identifier used to gate cross-window pairs,
                shape `[..., num_atoms]`.
            num_valid_atoms: Number of real atoms; enables clamped key windows containing only real atoms.

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

        return self.atom_encoder.build_local_geometry(reference_positions, reference_space_id, num_valid_atoms)

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

        Examples:
            >>> denoiser = CoordinateDenoiser(num_encoder_blocks=1)  # doctest: +SKIP
            >>> update = denoiser.predict_coordinate_update(inputs)  # doctest: +SKIP
        """

        denoising_single, denoising_pair = self.conditioner(
            inputs.noise_level,
            token_embedding=inputs.token_embedding,
            single=inputs.single,
            pair=inputs.pair,
            relative_features=inputs.relative_features,
        )

        target_prefix = tuple(denoising_single.shape[:-2])
        single = _broadcast_prefix(inputs.single, target_prefix, tail_dims=2)
        denoising_pair = _broadcast_prefix(denoising_pair, target_prefix, tail_dims=3)
        atom_encoder_output, token_hidden, atom_mask = self.token_refiner(
            self.atom_encoder,
            atom_features=inputs.atom_features,
            atom_to_token=inputs.atom_to_token,
            single=single,
            denoising_single=denoising_single,
            denoising_pair=denoising_pair,
            reference_positions=inputs.reference_positions,
            reference_charges=inputs.reference_charges,
            reference_mask=inputs.reference_mask,
            atom_mask=inputs.atom_mask,
            noisy_coords=inputs.noisy_coords,
            local_geometry=inputs.local_geometry,
        )
        num_valid_atoms = None if inputs.local_geometry is None else inputs.local_geometry.window_info.num_valid_atoms
        return self.atom_decoder(
            atom_to_token=inputs.atom_to_token,
            token_hidden=token_hidden,
            atom_query=atom_encoder_output.atom_query,
            atom_context=atom_encoder_output.atom_context,
            atom_mask=atom_mask,
            attention_layout="dense" if inputs.local_geometry is None else "local",
            num_valid_atoms=num_valid_atoms,
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
            reference_mask=inputs.reference_mask,
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
    updates.

    Args:
        relative_position_size: Width of the relative-position feature bucket embedding input.
        pair_input_size: Width of the incoming (concatenated) pair representation.
        pair_size: Width of the conditioned pair representation.
        single_conditioning_size: Width of the concatenated single conditioning input.
        single_size: Width of the conditioned single representation.
        noise_embedding_size: Width of the Fourier noise embedding.
        sigma_data: EDM data-scale hyperparameter used to normalise the log-noise.
        embed_relative_position: Whether to project relative-position buckets before pair conditioning.
        operator_backend: Public backend request forwarded to the pair and single transitions.

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
        operator_backend: str = "auto",
    ) -> None:
        super().__init__()
        self.sigma_data = sigma_data
        self.embed_relative_position = embed_relative_position

        self.relative_position_embedder = (
            RelativePositionEmbedder(relative_position_size, pair_size) if embed_relative_position else None
        )
        self.pair_input_norm = LayerNorm(pair_input_size, bias=False, operator_backend=operator_backend)
        self.pair_projection = nn.Linear(pair_input_size, pair_size, bias=False)
        self.pair_transitions = nn.ModuleList(
            Transition(pair_size, expansion=2, operator_backend=operator_backend) for _ in range(2)
        )

        self.single_conditioning_norm = LayerNorm(
            single_conditioning_size, bias=False, operator_backend=operator_backend
        )
        self.single_conditioning_projection = nn.Linear(single_conditioning_size, single_size, bias=False)
        self.noise_embedding = FrozenFourierEmbedding(noise_embedding_size)
        self.noise_norm = LayerNorm(noise_embedding_size, bias=False, operator_backend=operator_backend)
        self.noise_projection = nn.Linear(noise_embedding_size, single_size, bias=False)
        self.single_transitions = nn.ModuleList(
            Transition(single_size, expansion=2, operator_backend=operator_backend) for _ in range(2)
        )

    def forward(
        self,
        noise_level: Tensor,
        *,
        token_embedding: Tensor,
        single: Tensor,
        pair: Tensor,
        relative_features: RelativePositionBuckets,
    ) -> tuple[Tensor, Tensor]:
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
            ``(denoising_single, denoising_pair)`` with shapes
            `[..., num_samples, num_tokens, single_size]` and
            `[..., num_tokens, num_tokens, pair_size]`.

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
            transitions=self.pair_transitions,
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
            transitions=self.single_transitions,
            sample_axis=True,
        )
        return denoising_single, denoising_pair


def _relative_position_one_hot(buckets: RelativePositionBuckets) -> Tensor:
    r"""
    Materialize an unprojected relative-position feature vector.

    Concatenates residue offsets, token offsets, same-entity indicators, and chain
    offsets for conditioners that consume the raw categorical features alongside
    pair embeddings.
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

    Args:
        num_blocks: Number of token-encoder attention blocks.
        token_hidden_size: Width of the token hidden channel.
        single_size: Width of the single conditioning representation.
        pair_size: Width of the pair conditioning representation.
        num_attention_heads: Number of attention heads per token-encoder block.
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
        operator_backend: str = "auto",
    ) -> None:
        super().__init__()
        self.single_norm = LayerNorm(single_size, bias=False, operator_backend=operator_backend)
        self.single_to_token_projection = nn.Linear(single_size, token_hidden_size, bias=False)
        self.token_encoder = DenoisingTokenEncoder(
            num_blocks=num_blocks,
            num_attention_heads=num_attention_heads,
            token_hidden_size=token_hidden_size,
            conditioning_size=single_size,
            pair_size=pair_size,
            operator_backend=operator_backend,
        )
        self.token_norm = LayerNorm(token_hidden_size, bias=False, operator_backend=operator_backend)

    def forward(
        self,
        atom_encoder: CoordinateAtomToTokenEncoder,
        *,
        atom_features: Tensor,
        atom_to_token: Tensor,
        single: Tensor,
        denoising_single: Tensor,
        denoising_pair: Tensor,
        reference_positions: Tensor,
        reference_charges: Tensor,
        reference_mask: Tensor,
        atom_mask: Tensor,
        noisy_coords: Tensor,
        local_geometry: LocalAtomGeometry | None,
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
            atom_features: Per-atom input features.
            atom_to_token: Token index for each atom.
            single: Token single representation.
            denoising_single: Conditioned token single representation.
            denoising_pair: Conditioned token pair representation.
            reference_positions: Centered reference-conformer atom coordinates.
            reference_charges: Per-atom reference formal charges.
            reference_mask: Boolean mask for atoms with valid reference conformers.
            atom_mask: Boolean atom validity mask.
            noisy_coords: Noisy atom coordinates to refine.
            local_geometry: Optional local atom-pair geometry.
        Returns:
            A tuple of the atom-encoder output, the refined token hidden states of shape
            `[..., num_tokens, token_hidden_size]`, and the decoder atom mask (a flattened mask in
            the dense regime, ``None`` in the local regime).

        Examples:
            >>> refiner = DenoisingTokenRefiner(num_blocks=1)  # doctest: +SKIP
            >>> atom_output, token_hidden, atom_mask = refiner(  # doctest: +SKIP
            ...     atom_encoder, atom_features=atom_features, atom_to_token=atom_to_token,
            ...     single=single, denoising_single=denoising_single,
            ...     denoising_pair=denoising_pair, reference_positions=reference_positions,
            ...     reference_charges=reference_charges, reference_mask=reference_mask,
            ...     atom_mask=atom_mask,
            ...     noisy_coords=noisy_coords, local_geometry=None,
            ... )
        """
        target_prefix = tuple(denoising_single.shape[:-2])
        num_tokens = denoising_single.shape[-2]
        flat_denoising_single = _flatten_prefix(denoising_single, tail_dims=2)
        flat_denoising_pair = _flatten_prefix(
            _broadcast_prefix(denoising_pair, target_prefix, tail_dims=3), tail_dims=3
        )

        decoder_atom_mask: Tensor | None
        restore_prefix: tuple[int, ...] | None
        if local_geometry is None:
            atom_mask = _flatten_broadcast(atom_mask, target_prefix, tail_dims=1)
            atom_encoder_output = atom_encoder(
                atom_features=_flatten_broadcast(atom_features, target_prefix, tail_dims=2),
                reference_positions=_flatten_broadcast(reference_positions, target_prefix, tail_dims=2),
                reference_charges=torch.arcsinh(_flatten_broadcast(reference_charges, target_prefix, tail_dims=2)),
                reference_mask=_flatten_broadcast(reference_mask, target_prefix, tail_dims=1),
                single=_flatten_broadcast(single, target_prefix, tail_dims=2),
                pair=flat_denoising_pair,
                noisy_coords=_flatten_broadcast(noisy_coords, target_prefix, tail_dims=2),
                atom_mask=atom_mask,
                atom_to_token=_flatten_broadcast(atom_to_token, target_prefix, tail_dims=1),
                num_tokens=num_tokens,
            )
            decoder_atom_mask = atom_mask
            restore_prefix = None
        else:
            local_geometry = _broadcast_local_geometry(local_geometry, target_prefix)
            atom_encoder_output = atom_encoder(
                atom_features=_broadcast_prefix(atom_features, target_prefix, tail_dims=2),
                reference_positions=_broadcast_prefix(reference_positions, target_prefix, tail_dims=2),
                reference_charges=torch.arcsinh(_broadcast_prefix(reference_charges, target_prefix, tail_dims=2)),
                reference_mask=_broadcast_prefix(reference_mask, target_prefix, tail_dims=1),
                single=_broadcast_prefix(single, target_prefix, tail_dims=2),
                pair=_broadcast_prefix(denoising_pair, target_prefix, tail_dims=3),
                noisy_coords=noisy_coords,
                atom_mask=_broadcast_prefix(atom_mask, target_prefix, tail_dims=1),
                atom_to_token=_broadcast_prefix(atom_to_token, target_prefix, tail_dims=1),
                num_tokens=num_tokens,
                local_geometry=local_geometry,
            )
            decoder_atom_mask = None
            restore_prefix = target_prefix

        token_hidden = _flatten_prefix(atom_encoder_output.token_hidden, tail_dims=2)
        token_hidden = token_hidden + self.single_to_token_projection(self.single_norm(flat_denoising_single))
        token_hidden = self.token_encoder(token_hidden, flat_denoising_single, flat_denoising_pair)
        token_hidden = self.token_norm(token_hidden)
        if restore_prefix is not None:
            token_hidden = token_hidden.reshape(*restore_prefix, *token_hidden.shape[-2:])
        return atom_encoder_output, token_hidden, decoder_atom_mask


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
            num_valid_atoms=info.num_valid_atoms,
        ),
    )
