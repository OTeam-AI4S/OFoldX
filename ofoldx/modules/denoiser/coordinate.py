r"""
Normalized-key coordinate denoising.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, NamedTuple

import torch
from torch import Tensor, nn

from ..atom import (
    AtomConditioner,
    AtomPairEncoder,
    AtomToTokenMean,
    LocalAtomGeometry,
    TokenPairToLocalAtomPairBroadcast,
    TokenToAtomBroadcast,
)
from ..atom.encoders import NormalizedKeyAtomSelfAttention
from ..attention import DenoisingTokenEncoder
from ..backend import LayerNorm
from ..diffusion.conditioning import condition_pair
from ..diffusion.edm import EdmPreconditioner
from ..diffusion.types import CoordinateDenoiseFn, CoordinateDenoisingInput
from ..embedding.fourier import FrozenFourierEmbedding
from ..embedding.position import RelativePositionBuckets, RelativePositionEmbedder
from ..transition import Transition

if TYPE_CHECKING:
    from ..diffusion.sampling import CoordinateDenoiseContext


class _PreparedCoordinateDenoisingContext(NamedTuple):
    single: Tensor
    pair: Tensor
    atom_conditioning: Tensor
    atom_encoder_packed_bias: Tensor
    atom_decoder_packed_bias: Tensor


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
        self.gradient_checkpointing = False
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

    _gradient_checkpointing_func: Callable[..., Any]

    def build_local_geometry(
        self,
        reference_positions: Tensor,
        reference_space_id: Tensor,
        atom_mask: Tensor | None = None,
    ) -> LocalAtomGeometry:
        r"""Build the local atom windows consumed by this denoiser."""

        return self.atom_encoder.build_local_geometry(
            reference_positions,
            reference_space_id,
            atom_mask,
        )

    def _prepare_context(
        self,
        inputs: CoordinateDenoiseContext | CoordinateDenoisingInput,
    ) -> _PreparedCoordinateDenoisingContext:
        single, pair = self.conditioner.prepare_context(
            token_embedding=inputs.token_embedding,
            single=inputs.conditioning_single,
            pair=inputs.conditioning_pair,
            relative_features=inputs.relative_features,
        )
        atom_conditioning, atom_pair = self.atom_encoder.prepare_context(
            inputs.atom_features,
            reference_positions=inputs.reference_positions,
            reference_charges=inputs.reference_charges,
            reference_mask=inputs.reference_mask,
            single=inputs.single,
            pair=pair,
            atom_to_token=inputs.atom_to_token,
            local_geometry=inputs.local_geometry,
        )
        atom_encoder_packed_bias = self.atom_encoder.atom_self_attention.project_pair_bias(atom_pair)
        atom_decoder_packed_bias = self.atom_decoder.atom_self_attention.project_pair_bias(atom_pair)
        return _PreparedCoordinateDenoisingContext(
            single,
            pair,
            atom_conditioning,
            atom_encoder_packed_bias,
            atom_decoder_packed_bias,
        )

    def _checkpointed_context(
        self,
        inputs: CoordinateDenoiseContext | CoordinateDenoisingInput,
    ) -> _PreparedCoordinateDenoisingContext:
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            return self._gradient_checkpointing_func(self._prepare_context, inputs)
        return self._prepare_context(inputs)

    def prepare(
        self,
        inputs: CoordinateDenoiseContext | CoordinateDenoisingInput,
    ) -> CoordinateDenoiseFn:
        r"""Prepare one differentiable denoising invocation shared across samples."""

        prepared = self._checkpointed_context(inputs)

        def denoise(step_inputs: CoordinateDenoisingInput) -> Tensor:
            return self._forward_prepared(step_inputs, prepared)

        return denoise

    def _predict_coordinate_update_prepared(
        self,
        inputs: CoordinateDenoisingInput,
        prepared: _PreparedCoordinateDenoisingContext,
    ) -> Tensor:
        scaled_noisy_coords = self.preconditioner.scale_noisy_coords(
            inputs.noisy_coords,
            inputs.noise_level,
        )
        denoising_single = self.conditioner(inputs.noise_level, prepared.single)
        atom_tensors = self.atom_encoder(
            scaled_noisy_coords,
            prepared.atom_conditioning,
            prepared.atom_encoder_packed_bias,
            inputs.atom_mask,
            inputs.atom_to_token,
            denoising_single.shape[-2],
            inputs.local_geometry.window_info.key_index,
        )
        token_hidden = self.token_refiner(
            atom_tensors[0],
            denoising_single,
            prepared.pair,
        )
        decoder_args = (
            atom_tensors[1],
            prepared.atom_conditioning,
            prepared.atom_decoder_packed_bias,
            token_hidden,
            inputs.atom_to_token,
            inputs.atom_mask,
            inputs.local_geometry.window_info.key_index,
        )
        return self.atom_decoder(*decoder_args)

    def predict_coordinate_update(
        self,
        inputs: CoordinateDenoisingInput,
    ) -> Tensor:
        r"""Predict the raw coordinate update from ordinary noisy-coordinate inputs."""

        return self._predict_coordinate_update_prepared(inputs, self._checkpointed_context(inputs))

    def _forward_prepared(
        self,
        inputs: CoordinateDenoisingInput,
        prepared: _PreparedCoordinateDenoisingContext,
    ) -> Tensor:
        if inputs.noisy_coords.shape[-1] != 3:
            raise ValueError("noisy coordinates must have three spatial channels")
        coordinate_update = self._predict_coordinate_update_prepared(inputs, prepared)
        return self.preconditioner.combine_update(
            inputs.noisy_coords,
            inputs.noise_level,
            coordinate_update,
        )

    def forward(self, inputs: CoordinateDenoisingInput) -> Tensor:
        r"""Denoise coordinates with explicit batch, sample, atom, and xyz axes."""

        return self._forward_prepared(inputs, self._checkpointed_context(inputs))


class DenoisingConditioner(nn.Module):
    r"""Condition token single and pair representations for coordinate denoising."""

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
        single: Tensor,
    ) -> Tensor:
        r"""Add sample-specific noise conditioning to a prepared single representation."""

        noise = self.noise_norm(self.noise_embedding(torch.log(noise_level / self.sigma_data) / 4).to(single.dtype))
        single = single.unsqueeze(-3) + self.noise_projection(noise).unsqueeze(-2)
        for transition in self.single_transitions:
            single = single + transition(single)
        return single

    def prepare_context(
        self,
        *,
        token_embedding: Tensor,
        single: Tensor,
        pair: Tensor,
        relative_features: RelativePositionBuckets,
    ) -> tuple[Tensor, Tensor]:
        r"""Prepare the single base and pair conditioning shared across samples."""

        relative_position_conditioning = self.relative_position_embedder(relative_features)
        pair = condition_pair(
            pair,
            relative_position_conditioning,
            norm=self.pair_input_norm,
            projection=self.pair_projection,
            transitions=self.pair_transitions,
        )
        single = self.single_conditioning_projection(
            self.single_conditioning_norm(torch.cat([single, token_embedding], dim=-1))
        )
        return single, pair


class NormalizedKeyCoordinateAtomToTokenEncoder(nn.Module):
    r"""Encode local atom geometry and noisy coordinates with normalized gathered keys."""

    def __init__(
        self,
        token_hidden_size: int,
        atom_hidden_size: int = 128,
        atom_pair_hidden_size: int = 16,
        single_size: int = 384,
        pair_size: int = 128,
        num_blocks: int = 3,
        num_attention_heads: int = 4,
        num_query_atoms: int = 32,
        num_key_atoms: int = 128,
        atom_feature_size: int = 385,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.atom_conditioner = AtomConditioner(
            atom_feature_size=atom_feature_size,
            atom_hidden_size=atom_hidden_size,
        )
        self.single_norm = LayerNorm(single_size, bias=False, backend=backend)
        self.single_to_atom_projection = nn.Linear(single_size, atom_hidden_size, bias=False)
        self.token_to_atom_broadcast = TokenToAtomBroadcast(backend=backend)
        self.pair_norm = LayerNorm(pair_size, bias=False, backend=backend)
        self.pair_projection = nn.Linear(pair_size, atom_pair_hidden_size, bias=False)
        self.token_pair_to_local_atom_pair_broadcast = TokenPairToLocalAtomPairBroadcast(
            num_query_atoms=num_query_atoms,
            num_key_atoms=num_key_atoms,
            backend=backend,
        )
        self.atom_pair_encoder = AtomPairEncoder(
            atom_hidden_size=atom_hidden_size,
            atom_pair_hidden_size=atom_pair_hidden_size,
            num_query_atoms=num_query_atoms,
            num_key_atoms=num_key_atoms,
        )
        self.noisy_position_projection = nn.Linear(3, atom_hidden_size, bias=False)
        self.atom_self_attention = NormalizedKeyAtomSelfAttention(
            atom_hidden_size=atom_hidden_size,
            conditioning_size=atom_hidden_size,
            pair_size=atom_pair_hidden_size,
            num_blocks=num_blocks,
            num_attention_heads=num_attention_heads,
            num_query_atoms=num_query_atoms,
            num_key_atoms=num_key_atoms,
            backend=backend,
        )
        self.token_projection = nn.Linear(atom_hidden_size, token_hidden_size, bias=False)
        self.relu = nn.ReLU()
        self.atom_to_token_mean = AtomToTokenMean(backend=backend)

    def build_local_geometry(
        self,
        reference_positions: Tensor,
        reference_space_id: Tensor,
        atom_mask: Tensor | None = None,
    ) -> LocalAtomGeometry:
        return self.atom_pair_encoder.build_local_geometry(reference_positions, reference_space_id, atom_mask)

    def forward(
        self,
        noisy_coords: Tensor,
        atom_conditioning: Tensor,
        atom_packed_bias: Tensor,
        atom_mask: Tensor,
        atom_to_token: Tensor,
        num_tokens: int,
        key_index: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        r"""Encode one or more coordinate samples from prepared atom tensors."""

        atom_query = _broadcast_prefix(
            atom_conditioning,
            tuple(noisy_coords.shape[:-2]),
            tail_dims=2,
        )
        atom_query = atom_query + self.noisy_position_projection(noisy_coords)
        atom_query = self.atom_self_attention.forward_with_packed_bias(
            atom_query,
            atom_conditioning,
            atom_packed_bias,
            atom_mask=atom_mask,
            key_index=key_index,
        )
        token_hidden = self.relu(self.token_projection(atom_query))
        token_hidden = self.atom_to_token_mean(
            token_hidden,
            atom_to_token,
            num_tokens,
            atom_mask,
        )
        return token_hidden, atom_query

    def prepare_context(
        self,
        atom_features: Tensor,
        *,
        reference_positions: Tensor,
        reference_charges: Tensor,
        reference_mask: Tensor,
        single: Tensor,
        pair: Tensor,
        atom_to_token: Tensor,
        local_geometry: LocalAtomGeometry,
    ) -> tuple[Tensor, Tensor]:
        r"""Prepare atom conditioning and local pair features shared across samples."""

        atom_conditioning = self.atom_conditioner(
            atom_features,
            reference_positions=reference_positions,
            reference_charges=torch.arcsinh(reference_charges),
            reference_mask=reference_mask,
        )
        single_conditioning = self.single_to_atom_projection(self.single_norm(single))
        atom_conditioning = atom_conditioning + self.token_to_atom_broadcast(single_conditioning, atom_to_token)

        pair = self.pair_projection(self.pair_norm(pair))
        local_pair = self.token_pair_to_local_atom_pair_broadcast(
            pair,
            atom_to_token,
            key_index=local_geometry.window_info.key_index,
        )
        atom_pair = self.atom_pair_encoder(
            atom_conditioning,
            local_geometry=local_geometry,
            pair=local_pair,
        )
        return atom_conditioning, atom_pair


class DenoisingTokenRefiner(nn.Module):
    r"""Refine atom-encoded token states with denoising single and pair conditioning."""

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
        token_hidden: Tensor,
        single: Tensor,
        pair: Tensor,
    ) -> Tensor:
        token_hidden = token_hidden + self.single_to_token_projection(self.single_norm(single))
        token_hidden = self.token_encoder(token_hidden, single, pair)
        return self.token_norm(token_hidden)


class NormalizedKeyTokenToAtomDecoder(nn.Module):
    r"""Decode refined token states into local normalized-key atom updates."""

    def __init__(
        self,
        token_hidden_size: int = 768,
        atom_hidden_size: int = 128,
        atom_pair_hidden_size: int = 16,
        num_blocks: int = 3,
        num_attention_heads: int = 4,
        num_query_atoms: int = 32,
        num_key_atoms: int = 128,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.token_projection = nn.Linear(token_hidden_size, atom_hidden_size, bias=False)
        self.token_to_atom_broadcast = TokenToAtomBroadcast(backend=backend)
        self.atom_self_attention = NormalizedKeyAtomSelfAttention(
            atom_hidden_size=atom_hidden_size,
            conditioning_size=atom_hidden_size,
            pair_size=atom_pair_hidden_size,
            num_blocks=num_blocks,
            num_attention_heads=num_attention_heads,
            num_query_atoms=num_query_atoms,
            num_key_atoms=num_key_atoms,
            backend=backend,
        )
        self.query_norm = LayerNorm(atom_hidden_size, bias=False, backend=backend)
        self.output_projection = nn.Linear(atom_hidden_size, 3, bias=False)

    def forward(
        self,
        atom_query: Tensor,
        atom_conditioning: Tensor,
        atom_packed_bias: Tensor,
        token_hidden: Tensor,
        atom_to_token: Tensor,
        atom_mask: Tensor,
        key_index: Tensor | None,
    ) -> Tensor:
        atom_query = atom_query + self.token_to_atom_broadcast(
            self.token_projection(token_hidden),
            atom_to_token,
        )
        atom_query = self.atom_self_attention.forward_with_packed_bias(
            atom_query,
            conditioning=atom_conditioning,
            packed_bias=atom_packed_bias,
            atom_mask=atom_mask,
            key_index=key_index,
        )
        return self.output_projection(self.query_norm(atom_query))


def _broadcast_prefix(tensor: Tensor, target_prefix: tuple[int, ...], tail_dims: int) -> Tensor:
    tail_shape = tuple(tensor.shape[-tail_dims:])
    target_shape = (*target_prefix, *tail_shape)
    if tuple(tensor.shape) == target_shape:
        return tensor

    source_prefix = tuple(tensor.shape[:-tail_dims])
    if len(source_prefix) <= len(target_prefix) and source_prefix == target_prefix[: len(source_prefix)]:
        tensor = tensor.reshape(
            *source_prefix,
            *((1,) * (len(target_prefix) - len(source_prefix))),
            *tail_shape,
        )
    return torch.broadcast_to(tensor, target_shape)
