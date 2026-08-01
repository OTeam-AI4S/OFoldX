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
Configuration for the released OInvFold model graphs.
"""

from __future__ import annotations

from typing import Literal

from chanfig import Config


class OInvFoldPolymerConfig(Config):
    r"""
    Configure the shared protein, DNA, and RNA OInvFold graph.

    Attributes:
        hidden_size: Width of node, edge, and decoder representations.
        num_encoder_blocks: Number of graph encoder blocks.
        num_virtual_atoms: Number of learned geometric vectors per graph node.
        num_attention_heads: Number of graph-attention heads.
        num_geometric_layers: Number of layers in each geometric edge update.
        num_edge_layers: Number of layers in each learned edge update.
        dropout: Dropout probability in the graph encoder and autoregressive decoder.
        decoder_hidden_size: Width of the polymer decoder transition.
        modality: The checkpoint's polymer: which nucleic classes stand for the shared A, C and G output columns.

    Examples:
        >>> from ofoldx.models.oinvfold import OInvFoldPolymerConfig
        >>> config = OInvFoldPolymerConfig()
        >>> config.num_encoder_blocks
        12
    """

    hidden_size: int = 128
    num_encoder_blocks: int = 12
    num_virtual_atoms: int = 32
    num_attention_heads: int = 4
    num_geometric_layers: int = 3
    num_edge_layers: int = 3
    dropout: float = 0.0
    decoder_hidden_size: int = 256
    modality: Literal["protein", "dna", "rna"] = "protein"


class OInvFoldLigandConfig(Config):
    r"""
    Configure the released ligand atom-element OInvFold graph.

    Attributes:
        hidden_size: Width of node and edge representations.
        num_encoder_blocks: Number of graph encoder blocks.
        num_virtual_atoms: Number of learned geometric vectors per graph node.
        num_attention_heads: Number of graph-attention heads.
        num_geometric_layers: Number of layers in each geometric edge update.
        num_edge_layers: Number of layers in each learned edge update.
        dropout: Dropout probability in the graph encoder.

    Examples:
        >>> from ofoldx.models.oinvfold import OInvFoldLigandConfig
        >>> config = OInvFoldLigandConfig()
        >>> config.num_encoder_blocks
        12
    """

    hidden_size: int = 128
    num_encoder_blocks: int = 12
    num_virtual_atoms: int = 32
    num_attention_heads: int = 4
    num_geometric_layers: int = 3
    num_edge_layers: int = 3
    dropout: float = 0.0


class OInvFoldPolymerProcessorConfig(Config):
    r"""
    Select the polymer chemistry represented by one OInvFold repository.

    Attributes:
        modality: Protein, DNA, or RNA coordinate and vocabulary convention.
    """

    modality: Literal["protein", "dna", "rna"] = "protein"
