# OFoldX
# Copyright (C) 2026-Present  Valhalla Technology

# This file is part of OFoldX.

# OFoldX is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# OFoldX is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# For additional terms and clarifications, please refer to our License FAQ at:
# <https://ofoldx.valtech.ai/about/license-faq/>.

r"""
Normalized-key coordinate denoising.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING, Any, NamedTuple

import torch
from torch import Tensor, nn

from ..atom.coordinates import project_coordinates
from ..atom.encoders import (
    AtomConditioner,
    AtomPairEncoder,
    CoordinateAtomToTokenEncoder,
    TokenToAtomDecoder,
    TrunkToAtomConditioning,
)
from ..atom.window import LocalAtomGeometry, LocalAtomWindowing
from ..attention.pair_bias import PerBlockPairBiasProjection
from ..backend import LayerNorm
from ..backend.normalization import layer_norm_projection
from ..diffusion.conditioning import SingleConditioning, condition_pair
from ..diffusion.edm import EdmPreconditioner
from ..diffusion.types import CoordinateDenoiseFn, CoordinateDenoisingInput
from ..embedding.position import RelativePositionBuckets, RelativePositionEmbedder
from ..transformer import (
    NormalizedKeyAtomCrossAttentionTransformerEncoderBlock,
    PairBiasedConditionedTransformerEncoder,
)
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

    The centred [`windowing`][ofoldx.modules.atom.window.LocalAtomWindowing] builds the local atom
    geometry a denoising context carries. The atom conditioning and the encoder and decoder pair biases
    depend only on the reference structure and trunk, so they are prepared once and reused for every
    noisy-coordinate sample.

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
            backend=backend,
        )
        self.windowing = LocalAtomWindowing(num_query_atoms, num_key_atoms)
        self.atom_conditioner = AtomConditioner(atom_feature_size, atom_hidden_size)
        self.trunk_conditioning = TrunkToAtomConditioning(
            single_size,
            pair_size,
            atom_hidden_size,
            atom_pair_hidden_size,
            bias=False,
            backend=backend,
        )
        self.atom_pair_encoder = AtomPairEncoder(atom_hidden_size, atom_pair_hidden_size)
        self.atom_encoder_pair_bias_projection = PerBlockPairBiasProjection(
            pair_size=atom_pair_hidden_size,
            num_blocks=num_atom_encoder_blocks,
            num_attention_heads=num_atom_encoder_attention_heads,
            bias=False,
            backend=backend,
        )
        self.atom_decoder_pair_bias_projection = PerBlockPairBiasProjection(
            pair_size=atom_pair_hidden_size,
            num_blocks=num_atom_decoder_blocks,
            num_attention_heads=num_atom_decoder_attention_heads,
            bias=False,
            backend=backend,
        )
        self.atom_encoder = CoordinateAtomToTokenEncoder(
            atom_hidden_size,
            token_hidden_size,
            num_atom_encoder_blocks,
            num_atom_encoder_attention_heads,
            backend,
            block=NormalizedKeyAtomCrossAttentionTransformerEncoderBlock,
        )
        self.token_refiner = DenoisingTokenRefiner(
            num_blocks=num_encoder_blocks,
            num_attention_heads=num_encoder_attention_heads,
            token_hidden_size=token_hidden_size,
            single_size=single_size,
            pair_size=pair_size,
            backend=backend,
        )
        self.atom_decoder = TokenToAtomDecoder(
            token_hidden_size,
            atom_hidden_size,
            num_atom_decoder_blocks,
            num_atom_decoder_attention_heads,
            backend,
            block=NormalizedKeyAtomCrossAttentionTransformerEncoderBlock,
            bias=False,
        )

    _gradient_checkpointing_func: Callable[..., Any]

    def _prepare_context(
        self,
        inputs: CoordinateDenoiseContext | CoordinateDenoisingInput,
    ) -> _PreparedCoordinateDenoisingContext:
        checkpoint = (
            self._gradient_checkpointing_func
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled()
            else None
        )
        single, pair = self.conditioner.prepare_context(
            token_embedding=inputs.token_embedding,
            single=inputs.conditioning_single,
            pair=inputs.conditioning_pair,
            relative_features=inputs.relative_features,
            checkpoint=checkpoint,
        )
        atom_args = (
            inputs.atom_features,
            inputs.reference_positions,
            inputs.reference_charges,
            inputs.reference_mask,
            inputs.single,
            pair,
            inputs.atom_to_token,
            inputs.local_geometry,
        )
        atom_context = (
            self._prepare_atom_context(*atom_args)
            if checkpoint is None
            else checkpoint(self._prepare_atom_context, *atom_args)
        )
        return _PreparedCoordinateDenoisingContext(single, pair, *atom_context)

    def _prepare_atom_context(
        self,
        atom_features: Tensor,
        reference_positions: Tensor,
        reference_charges: Tensor,
        reference_mask: Tensor,
        single: Tensor,
        pair: Tensor,
        atom_to_token: Tensor,
        local_geometry: LocalAtomGeometry,
    ) -> tuple[Tensor, Tensor, Tensor]:
        atom_conditioning = self.atom_conditioner(
            atom_features,
            reference_positions=reference_positions,
            reference_charges=reference_charges,
            reference_mask=reference_mask,
        )
        atom_conditioning, local_pair = self.trunk_conditioning(
            atom_conditioning,
            single,
            pair,
            atom_to_token,
            local_geometry.window_info,
        )
        atom_pair = self.atom_pair_encoder(atom_conditioning, local_geometry, local_pair)
        return (
            atom_conditioning,
            self.atom_encoder_pair_bias_projection(atom_pair),
            self.atom_decoder_pair_bias_projection(atom_pair),
        )

    def prepare(
        self,
        inputs: CoordinateDenoiseContext | CoordinateDenoisingInput,
    ) -> CoordinateDenoiseFn:
        r"""Prepare denoising shared across one sampling invocation.

        The callable reuses this invocation's conditioning and returns coordinates
        that own their storage.
        """

        prepared = self._prepare_context(inputs)

        def denoise(step_inputs: CoordinateDenoisingInput) -> Tensor:
            return self(step_inputs, prepared=prepared).clone()

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
        denoising_single = self.conditioner.condition_noise(
            self.preconditioner.c_noise(inputs.noise_level), prepared.single
        )
        window_info = inputs.local_geometry.window_info
        token_hidden, atom_query = self.atom_encoder(
            scaled_noisy_coords,
            prepared.atom_conditioning,
            prepared.atom_conditioning,
            prepared.atom_encoder_packed_bias,
            inputs.atom_mask,
            inputs.atom_to_token,
            prepared.single,
            window_info,
        )
        token_hidden = self.token_refiner(token_hidden, denoising_single, prepared.pair)
        return self.atom_decoder(
            token_hidden,
            atom_query,
            prepared.atom_conditioning,
            prepared.atom_decoder_packed_bias,
            inputs.atom_mask,
            inputs.atom_to_token,
            window_info,
        )

    def predict_coordinate_update(
        self,
        inputs: CoordinateDenoisingInput,
    ) -> Tensor:
        r"""Predict the raw coordinate update from ordinary noisy-coordinate inputs."""

        inputs = self._project_noisy_input(inputs)
        return self._predict_coordinate_update_prepared(inputs, self._prepare_context(inputs))

    def _project_noisy_input(self, inputs: CoordinateDenoisingInput) -> CoordinateDenoisingInput:
        observed = inputs.atom_mask.to(torch.bool)
        while observed.dim() < inputs.noisy_coords.dim() - 1:
            observed = observed.unsqueeze(-2)
        coordinates = project_coordinates(
            inputs.noisy_coords,
            observed & torch.ones_like(inputs.noisy_coords[..., 0], dtype=torch.bool),
            dtype=self.atom_encoder.noisy_position_projection.weight.dtype,
        )
        return replace(inputs, noisy_coords=coordinates)

    def _forward_prepared(
        self,
        inputs: CoordinateDenoisingInput,
        prepared: _PreparedCoordinateDenoisingContext,
    ) -> Tensor:
        inputs = self._project_noisy_input(inputs)
        coordinate_update = self._predict_coordinate_update_prepared(inputs, prepared)
        return self.preconditioner.combine_update(
            inputs.noisy_coords,
            inputs.noise_level,
            coordinate_update,
        )

    def forward(
        self,
        inputs: CoordinateDenoisingInput,
        *,
        prepared: _PreparedCoordinateDenoisingContext | None = None,
    ) -> Tensor:
        r"""Denoise coordinates with explicit batch, sample, atom, and xyz axes."""

        if prepared is None:
            prepared = self._prepare_context(inputs)
        return self._forward_prepared(inputs, prepared)


class DenoisingConditioner(SingleConditioning):
    r"""Condition token single and pair representations for coordinate denoising."""

    def __init__(
        self,
        relative_position_size: int = 139,
        pair_input_size: int = 256,
        pair_size: int = 128,
        single_conditioning_size: int = 833,
        single_size: int = 384,
        noise_embedding_size: int = 256,
        backend: str = "auto",
        *,
        relative_position_embedder: Callable[[RelativePositionBuckets], Tensor] | None = None,
    ) -> None:
        super().__init__(
            single_conditioning_size,
            single_size,
            fourier_embedding_size=noise_embedding_size,
            backend=backend,
            norm_bias=False,
            single_projection_bias=False,
        )
        self.relative_position_embedder = (
            RelativePositionEmbedder(relative_position_size, pair_size)
            if relative_position_embedder is None
            else relative_position_embedder
        )
        self.pair_input_norm = LayerNorm(pair_input_size, bias=False, backend=backend)
        self.pair_projection = nn.Linear(pair_input_size, pair_size, bias=False)
        self.pair_transitions = nn.ModuleList(Transition(pair_size, expansion=2, backend=backend) for _ in range(2))

    def prepare_context(
        self,
        *,
        token_embedding: Tensor,
        single: Tensor,
        pair: Tensor,
        relative_features: RelativePositionBuckets,
        checkpoint: Callable[..., Tensor] | None = None,
    ) -> tuple[Tensor, Tensor]:
        r"""
        Prepare differentiable single and pair conditioning shared across samples.

        Args:
            token_embedding: Raw token features ``[..., N_token, C_token]``.
            single: Trunk single representation ``[..., N_token, C_single]``.
            pair: Trunk pair representation ``[..., N_token, N_token, C_pair]``.
            relative_features: Relative-position buckets on the same token axes.
            checkpoint: Optional recomputation callable for the pair projection
                and each pair transition. These boundaries retain their inputs;
                the single path is evaluated directly.

        Returns:
            single: Single conditioning ``[..., N_token, single_size]``.
            pair: Pair conditioning ``[..., N_token, N_token, pair_size]``.
        """

        relative_position_conditioning = self.relative_position_embedder(relative_features)
        pair = condition_pair(
            pair,
            relative_position_conditioning,
            norm=self.pair_input_norm,
            projection=self.pair_projection,
            transitions=self.pair_transitions,
            checkpoint=checkpoint,
        )
        single = self.prepare(single, token_embedding)
        return single, pair


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
        self.token_encoder = PairBiasedConditionedTransformerEncoder(
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
        token_hidden = token_hidden + layer_norm_projection(
            single, norm=self.single_norm, projection=self.single_to_token_projection
        )
        token_hidden = self.token_encoder(token_hidden, single, pair)
        return self.token_norm(token_hidden)
