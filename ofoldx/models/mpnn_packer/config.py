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
Configuration for MPNN Packer.
"""

from __future__ import annotations

from chanfig import Config


class MpnnPackerConfig(Config):
    r"""
    Configure the MPNN Packer architecture.

    Attributes:
        hidden_size: Width of graph hidden states.
        edge_size: Width of projected edge features.
        num_encoder_blocks: Number of graph encoder blocks.
        num_decoder_blocks: Number of autoregressive decoder blocks.
        num_rbf: Number of radial-basis bins for distance features.
        num_neighbors: Number of nearest neighbors in each residue graph.
        num_context_atoms: Number of nearby context atoms represented per residue.
        num_mix: Number of mixture components predicted for each torsion.
        positional_embedding_size: Width of the relative-position embedding.
        dropout: Dropout probability in encoder and decoder blocks.
        coordinate_noise: Training-time Gaussian coordinate-noise standard deviation.
        lower_bound: Lower distance bound for context radial-basis expansion.
        upper_bound: Upper distance bound for context radial-basis expansion.

    Examples:
        >>> from ofoldx.models.mpnn_packer import MpnnPackerConfig
        >>> config = MpnnPackerConfig()
        >>> config.num_mix
        3
    """

    hidden_size: int = 128
    edge_size: int = 128
    num_encoder_blocks: int = 3
    num_decoder_blocks: int = 3
    num_rbf: int = 16
    num_neighbors: int = 32
    num_context_atoms: int = 16
    num_mix: int = 3
    positional_embedding_size: int = 16
    dropout: float = 0.0
    coordinate_noise: float = 0.0
    lower_bound: float = 0.0
    upper_bound: float = 20.0


class MpnnPackerProcessorConfig(Config):
    r"""
    Configure MPNN Packer input processing.

    Attributes:
        num_context_atoms: Number of nearby context atoms emitted per residue.
    """

    num_context_atoms: int = 16
