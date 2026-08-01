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
Configuration values for the ProteinMPNN model and processor.
"""

from __future__ import annotations

from chanfig import Config


class ProteinMpnnConfig(Config):
    r"""
    Configure architecture values shared by the ProteinMPNN graphs.

    Attributes:
        hidden_size: Width of node, edge, and decoder hidden states.
        num_encoder_blocks: Number of graph encoder blocks.
        num_decoder_blocks: Number of autoregressive decoder blocks.
        num_neighbors: Number of nearest neighbors in each residue graph.
        dropout: Dropout probability in encoder and decoder blocks.
        ca_only: Whether to build the CA-only graph instead of the backbone graph.

    Examples:
        >>> from ofoldx.models.proteinmpnn import ProteinMpnnConfig
        >>> config = ProteinMpnnConfig(ca_only=True)
        >>> config.ca_only
        True
    """

    hidden_size: int = 128
    num_encoder_blocks: int = 3
    num_decoder_blocks: int = 3
    num_neighbors: int = 48
    dropout: float = 0.1
    ca_only: bool = False


class ProteinMpnnProcessorConfig(Config):
    r"""
    Configure ProteinMPNN input processing.

    Attributes:
        ca_only: Whether inputs contain only C-alpha coordinates instead of the
            N, C-alpha, C, and O backbone atoms.
    """

    ca_only: bool = False
