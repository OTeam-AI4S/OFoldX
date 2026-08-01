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
Sample-independent atom and token conditioning for coordinate denoisers.
"""

from __future__ import annotations

from typing import NamedTuple

from torch import Tensor, nn

from ..atom.encoders import AtomPairEncoder, TrunkToAtomConditioning
from ..atom.window import LocalAtomWindowInfo, LocalAtomWindowing
from ..attention import PerBlockPairBiasProjection
from ..diffusion.conditioning import PairwiseConditioning


class AtomDenoiserContext(NamedTuple):
    r"""Sample-independent atom states, attention biases, and local atom layout."""

    atom_query: Tensor
    atom_conditioning: Tensor
    atom_encoder_bias: Tensor
    atom_decoder_bias: Tensor
    token_encoder_bias: Tensor
    window_info: LocalAtomWindowInfo


class AtomDenoiserConditioning(nn.Module):
    r"""Prepare reference-atom and trunk conditioning for coordinate denoising.

    Refines trunk pairs with relative-position features, builds local reference
    geometry, conditions reference atom states with trunk single/pair features,
    and projects attention biases for each atom/token encoder and atom decoder
    block. Noise levels and sample axes belong to the downstream denoiser.
    """

    def __init__(
        self,
        single_size: int = 384,
        pair_size: int = 128,
        atom_hidden_size: int = 128,
        atom_pair_hidden_size: int = 16,
        atom_feature_size: int = 128,
        num_query_atoms: int = 32,
        num_key_atoms: int = 128,
        num_atom_encoder_blocks: int = 3,
        num_atom_encoder_attention_heads: int = 4,
        num_atom_decoder_blocks: int = 3,
        num_atom_decoder_attention_heads: int = 4,
        num_token_encoder_blocks: int = 24,
        num_token_encoder_attention_heads: int = 16,
        num_conditioning_transitions: int = 2,
        backend: str = "auto",
        *,
        atom_feature_bias: bool = True,
    ) -> None:
        super().__init__()
        self.pairwise_conditioner = PairwiseConditioning(
            pair_size=pair_size,
            relative_position_size=pair_size,
            num_transitions=num_conditioning_transitions,
            backend=backend,
        )
        self.windowing = LocalAtomWindowing(num_query_atoms, num_key_atoms)
        self.atom_pair_encoder = AtomPairEncoder(atom_hidden_size, atom_pair_hidden_size)
        self.atom_feature_projection = nn.Linear(atom_feature_size, atom_hidden_size, bias=atom_feature_bias)
        self.trunk_conditioning = TrunkToAtomConditioning(
            single_size,
            pair_size,
            atom_hidden_size,
            atom_pair_hidden_size,
            backend=backend,
        )
        self.atom_encoder_pair_bias_projection = PerBlockPairBiasProjection(
            pair_size=atom_pair_hidden_size,
            num_blocks=num_atom_encoder_blocks,
            num_attention_heads=num_atom_encoder_attention_heads,
            backend=backend,
        )
        self.atom_decoder_pair_bias_projection = PerBlockPairBiasProjection(
            pair_size=atom_pair_hidden_size,
            num_blocks=num_atom_decoder_blocks,
            num_attention_heads=num_atom_decoder_attention_heads,
            backend=backend,
        )
        self.token_encoder_pair_bias_projection = PerBlockPairBiasProjection(
            pair_size=pair_size,
            num_blocks=num_token_encoder_blocks,
            num_attention_heads=num_token_encoder_attention_heads,
            backend=backend,
        )

    def forward(
        self,
        single_trunk: Tensor,
        pair_trunk: Tensor,
        relative_position_encoding: Tensor,
        *,
        atom_features: Tensor,
        reference_positions: Tensor,
        atom_mask: Tensor,
        reference_space_id: Tensor,
        atom_to_token: Tensor,
    ) -> AtomDenoiserContext:
        r"""Return reference atom states, per-block biases, and their local layout."""

        pair = self.pairwise_conditioner(pair_trunk, relative_position_encoding)
        geometry = self.windowing(reference_positions, reference_space_id, atom_mask)
        atom_query = self.atom_feature_projection(atom_features)
        atom_conditioning, local_pair = self.trunk_conditioning(
            atom_query,
            single_trunk,
            pair,
            atom_to_token,
            geometry.window_info,
        )
        atom_pair = self.atom_pair_encoder(atom_conditioning, geometry, local_pair)
        return AtomDenoiserContext(
            atom_query=atom_query,
            atom_conditioning=atom_conditioning,
            atom_encoder_bias=self.atom_encoder_pair_bias_projection(atom_pair),
            atom_decoder_bias=self.atom_decoder_pair_bias_projection(atom_pair),
            token_encoder_bias=self.token_encoder_pair_bias_projection(pair),
            window_info=geometry.window_info,
        )
