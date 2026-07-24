r"""
Coordinate-denoising networks.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..atom import (
    AtomToTokenOutput,
    CoordinateAtomToTokenEncoder,
    LocalAtomGeometry,
    LocalAtomWindowInfo,
    NormalizedKeyCoordinateAtomToTokenEncoder,
    NormalizedKeyTokenToAtomDecoder,
)
from ..attention import DenoisingTokenEncoder
from ..backend import LayerNorm
from ..diffusion.conditioning import condition_pair, condition_single
from ..diffusion.edm import EdmPreconditioner
from ..diffusion.types import CoordinateDenoisingInput
from ..embedding.fourier import FrozenFourierEmbedding
from ..embedding.position import RelativePositionBuckets, RelativePositionEmbedder
from ..transition import Transition


class CoordinateDenoiser(nn.Module):
    r"""
    Predict denoised atom coordinates with EDM preconditioning.

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
        operator_backend: Backend selector passed to the attention/transition operators.
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
        num_atom_encoder_blocks: int = 3,
        num_atom_encoder_attention_heads: int = 4,
        num_atom_decoder_blocks: int = 3,
        num_atom_decoder_attention_heads: int = 4,
        num_query_atoms: int = 32,
        num_key_atoms: int = 128,
        operator_backend: str = "auto",
    ) -> None:
        super().__init__()
        self.preconditioner = EdmPreconditioner(sigma_data=sigma_data)
        self.conditioner = DenoisingConditioner(
            pair_input_size=2 * pair_size,
            pair_size=pair_size,
            single_conditioning_size=single_size + token_feature_size,
            single_size=single_size,
            sigma_data=sigma_data,
            operator_backend=operator_backend,
        )
        self.atom_encoder = NormalizedKeyCoordinateAtomToTokenEncoder(
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
        self.token_refiner = DenoisingTokenRefiner(
            num_blocks=num_encoder_blocks,
            num_attention_heads=num_encoder_attention_heads,
            token_hidden_size=token_hidden_size,
            single_size=single_size,
            pair_size=pair_size,
            operator_backend=operator_backend,
        )
        self.atom_decoder = NormalizedKeyTokenToAtomDecoder(
            token_hidden_size=token_hidden_size,
            atom_hidden_size=atom_hidden_size,
            atom_pair_hidden_size=atom_pair_hidden_size,
            num_blocks=num_atom_decoder_blocks,
            num_attention_heads=num_atom_decoder_attention_heads,
            num_query_atoms=num_query_atoms,
            num_key_atoms=num_key_atoms,
            operator_backend=operator_backend,
        )

    def build_local_geometry(
        self, reference_positions: Tensor, reference_space_id: Tensor, num_valid_atoms: int | None = None
    ) -> LocalAtomGeometry:
        r"""
        Build local atom-pair geometry using the denoising atom-encoder window layout.

        Args:
            reference_positions: Centered reference coordinates with shape
                ``[..., N_atom, 3]``.
            reference_space_id: Reference-space index with shape ``[..., N_atom]``.
            num_valid_atoms: Number of real atoms; enables clamped key windows containing only real atoms.

        Returns:
            Geometry and masks aligned to the atom encoder's local windows.
        """

        return self.atom_encoder.build_local_geometry(reference_positions, reference_space_id, num_valid_atoms)

    def predict_coordinate_update(self, inputs: CoordinateDenoisingInput) -> Tensor:
        r"""
        Predict the raw per-atom coordinate update, before EDM output preconditioning.

        Args:
            inputs: Denoising input whose coordinate and noise tensors share a
                sample axis.

        Returns:
            Coordinate update with shape ``[..., N_sample, N_atom, 3]``.
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
        Denoise atom coordinates.

        Args:
            inputs: Atom, token, noise-level, and noisy-coordinate inputs.

        Returns:
            Denoised coordinates with shape ``[..., N_atom, 3]`` or
            ``[..., N_sample, N_atom, 3]``, matching the input's sample-axis layout.
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

    Args:
        relative_position_size: Width of the relative-position feature bucket embedding input.
        pair_input_size: Width of the incoming (concatenated) pair representation.
        pair_size: Width of the conditioned pair representation.
        single_conditioning_size: Width of the concatenated single conditioning input.
        single_size: Width of the conditioned single representation.
        noise_embedding_size: Width of the Fourier noise embedding.
        sigma_data: EDM data-scale hyperparameter used to normalise the log-noise.
        operator_backend: Public backend request forwarded to the pair and single transitions.
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
        operator_backend: str = "auto",
    ) -> None:
        super().__init__()
        self.sigma_data = sigma_data
        self.relative_position_embedder = RelativePositionEmbedder(relative_position_size, pair_size)
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

        Args:
            noise_level: Noise levels with shape ``[..., N_sample]``.
            token_embedding: Token embeddings with shape ``[..., N_token, C_token]``.
            single: Single representation with shape ``[..., N_token, single_size]``.
            pair: Pair representation with shape
                ``[..., N_token, N_token, pair_input_size]``.
            relative_features: Relative-position feature bundle indexing the pair embedding.

        Returns:
            Conditioned single and pair representations with shapes
            ``[..., N_sample, N_token, single_size]`` and
            ``[..., N_token, N_token, pair_size]``.
        """

        relative_position_conditioning = self.relative_position_embedder(relative_features)
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


class DenoisingTokenRefiner(nn.Module):
    r"""
    Refine atom-encoded token states with denoising single and pair conditioning.

    Args:
        num_blocks: Number of token-encoder attention blocks.
        token_hidden_size: Width of the token hidden channel.
        single_size: Width of the single conditioning representation.
        pair_size: Width of the pair conditioning representation.
        num_attention_heads: Number of attention heads per token-encoder block.
        operator_backend: Backend selector passed to the attention operators.
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
        atom_encoder: CoordinateAtomToTokenEncoder | NormalizedKeyCoordinateAtomToTokenEncoder,
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
            The atom-encoder output, refined token states with trailing shape
            ``(N_token, token_hidden_size)``, and the decoder atom mask. The mask
            is ``None`` for local-window attention.
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
