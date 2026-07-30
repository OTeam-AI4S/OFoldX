r"""
Coordinate-denoising components.
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
    Predict denoised coordinates with the normalized-key diffusion-module equation.

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
        backend: Backend selector passed to the attention and transition operators.
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
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.preconditioner = EdmPreconditioner(sigma_data=sigma_data)
        self.conditioner = DenoisingConditioner(
            pair_input_size=2 * pair_size,
            pair_size=pair_size,
            single_conditioning_size=single_size + token_feature_size,
            single_size=single_size,
            sigma_data=sigma_data,
            backend=backend,
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
            backend=backend,
        )
        self.token_refiner = DenoisingTokenRefiner(
            num_blocks=num_encoder_blocks,
            num_attention_heads=num_encoder_attention_heads,
            token_hidden_size=token_hidden_size,
            single_size=single_size,
            pair_size=pair_size,
            backend=backend,
        )
        self.atom_decoder = NormalizedKeyTokenToAtomDecoder(
            token_hidden_size=token_hidden_size,
            atom_hidden_size=atom_hidden_size,
            atom_pair_hidden_size=atom_pair_hidden_size,
            num_blocks=num_atom_decoder_blocks,
            num_attention_heads=num_atom_decoder_attention_heads,
            num_query_atoms=num_query_atoms,
            num_key_atoms=num_key_atoms,
            backend=backend,
        )

    def build_local_geometry(
        self,
        reference_positions: Tensor,
        reference_space_id: Tensor,
        atom_mask: Tensor | None = None,
    ) -> LocalAtomGeometry:
        r"""
        Build local atom-pair geometry using the denoising atom-encoder windows.
        """

        return self.atom_encoder.build_local_geometry(
            reference_positions,
            reference_space_id,
            atom_mask,
        )

    def predict_coordinate_update(self, inputs: CoordinateDenoisingInput) -> Tensor:
        r"""
        Predict the raw coordinate update from preconditioned noisy coordinates.
        """

        denoising_single, denoising_pair = self.conditioner(
            inputs.noise_level,
            token_embedding=inputs.token_embedding,
            single=inputs.conditioning_single,
            pair=inputs.conditioning_pair,
            relative_features=inputs.relative_features,
        )
        atom_output, token_hidden = self.token_refiner(
            self.atom_encoder,
            atom_features=inputs.atom_features,
            atom_to_token=inputs.atom_to_token,
            single=inputs.single,
            denoising_single=denoising_single,
            denoising_pair=denoising_pair,
            reference_positions=inputs.reference_positions,
            reference_charges=inputs.reference_charges,
            reference_mask=inputs.reference_mask,
            atom_mask=inputs.atom_mask,
            noisy_coords=inputs.noisy_coords,
            local_geometry=inputs.local_geometry,
        )
        return self.atom_decoder(
            atom_to_token=inputs.atom_to_token,
            token_hidden=token_hidden,
            atom_query=atom_output.atom_query,
            atom_context=atom_output.atom_context,
            atom_mask=inputs.atom_mask,
            attention_layout="local",
            key_index=inputs.local_geometry.window_info.key_index,
        )

    def forward(self, inputs: CoordinateDenoisingInput) -> Tensor:
        r"""
        Denoise coordinates with explicit batch, sample, atom, and xyz axes.
        """

        scaled_inputs = CoordinateDenoisingInput(
            atom_features=inputs.atom_features,
            atom_to_token=inputs.atom_to_token,
            token_embedding=inputs.token_embedding,
            single=inputs.single,
            conditioning_single=inputs.conditioning_single,
            conditioning_pair=inputs.conditioning_pair,
            relative_features=inputs.relative_features,
            noise_level=inputs.noise_level,
            reference_positions=inputs.reference_positions,
            reference_charges=inputs.reference_charges,
            reference_mask=inputs.reference_mask,
            atom_mask=inputs.atom_mask.to(dtype=inputs.atom_features.dtype),
            noisy_coords=self.preconditioner.scale_noisy_coords(
                inputs.noisy_coords,
                inputs.noise_level,
            ),
            local_geometry=inputs.local_geometry,
        )
        coordinate_update = self.predict_coordinate_update(scaled_inputs)
        return self.preconditioner.combine_update(
            inputs.noisy_coords,
            inputs.noise_level,
            coordinate_update,
        )


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
        backend: Public backend request forwarded to the pair and single transitions.
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
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.sigma_data = sigma_data
        self.relative_position_embedder = RelativePositionEmbedder(relative_position_size, pair_size)
        self.pair_input_norm = LayerNorm(pair_input_size, bias=False, backend=backend)
        self.pair_projection = nn.Linear(pair_input_size, pair_size, bias=False)
        self.pair_transitions = nn.ModuleList(Transition(pair_size, expansion=2, backend=backend) for _ in range(2))

        self.single_conditioning_norm = LayerNorm(single_conditioning_size, bias=False, backend=backend)
        self.single_conditioning_projection = nn.Linear(single_conditioning_size, single_size, bias=False)
        self.noise_embedding = FrozenFourierEmbedding(noise_embedding_size)
        self.noise_norm = LayerNorm(noise_embedding_size, bias=False, backend=backend)
        self.noise_projection = nn.Linear(noise_embedding_size, single_size, bias=False)
        self.single_transitions = nn.ModuleList(Transition(single_size, expansion=2, backend=backend) for _ in range(2))

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
        backend: Backend selector passed to the attention operators.
    """

    def __init__(
        self,
        num_blocks: int = 24,
        token_hidden_size: int = 768,
        single_size: int = 384,
        pair_size: int = 128,
        num_attention_heads: int = 16,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.single_norm = LayerNorm(single_size, bias=False, backend=backend)
        self.single_to_token_projection = nn.Linear(single_size, token_hidden_size, bias=False)
        self.token_encoder = DenoisingTokenEncoder(
            num_blocks=num_blocks,
            num_attention_heads=num_attention_heads,
            token_hidden_size=token_hidden_size,
            conditioning_size=single_size,
            pair_size=pair_size,
            backend=backend,
        )
        self.token_norm = LayerNorm(token_hidden_size, bias=False, backend=backend)

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
        local_geometry: LocalAtomGeometry,
    ) -> tuple[AtomToTokenOutput, Tensor]:
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
            local_geometry: Local atom-window geometry.

        Returns:
            The atom-encoder output and refined token states with trailing shape
            ``(N_token, token_hidden_size)``.
        """

        target_prefix = tuple(denoising_single.shape[:-2])
        num_tokens = denoising_single.shape[-2]
        flat_denoising_single = _flatten_prefix(denoising_single, tail_dims=2)
        flat_denoising_pair = _flatten_prefix(
            _broadcast_prefix(denoising_pair, target_prefix, tail_dims=3), tail_dims=3
        )

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

        token_hidden = _flatten_prefix(atom_encoder_output.token_hidden, tail_dims=2)
        token_hidden = token_hidden + self.single_to_token_projection(self.single_norm(flat_denoising_single))
        token_hidden = self.token_encoder(token_hidden, flat_denoising_single, flat_denoising_pair)
        token_hidden = self.token_norm(token_hidden)
        token_hidden = token_hidden.reshape(*target_prefix, *token_hidden.shape[-2:])
        return atom_encoder_output, token_hidden


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
            key_index=(
                None if info.key_index is None else _broadcast_prefix(info.key_index, target_prefix, tail_dims=2)
            ),
        ),
    )
