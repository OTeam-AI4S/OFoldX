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
Configuration values for the NA-MPNN model and processor.
"""

from __future__ import annotations

from chanfig import Config


class NaMpnnConfig(Config):
    r"""
    Configure the NA-MPNN architecture.

    Attributes:
        hidden_size: Width of the message-passing hidden states.
        feature_size: Shared width of the polymer node and geometric edge features.
        num_encoder_blocks: Number of graph encoder blocks.
        num_decoder_blocks: Number of autoregressive decoder blocks.
        num_neighbors: Number of nearest neighbors in the residue graph.
        dropout: Dropout probability in encoder and decoder blocks.
        positional_embedding_size: Width of the relative-position embedding.
        num_rbf: Number of radial-basis bins for atom-pair distances.
        include_predicted_na_n: Whether to append the idealized nucleobase N feature.

    Examples:
        >>> from ofoldx.models.nampnn import NaMpnnConfig
        >>> config = NaMpnnConfig()
        >>> config.num_neighbors
        32
    """

    hidden_size: int = 128
    feature_size: int = 128
    num_encoder_blocks: int = 3
    num_decoder_blocks: int = 3
    num_neighbors: int = 32
    dropout: float = 0.1
    positional_embedding_size: int = 16
    num_rbf: int = 16
    include_predicted_na_n: bool = True


class NaMpnnProcessorConfig(Config):
    r"""
    Configure NA-MPNN input processing.

    Attributes:
        shared_nucleic_tokens: Whether RNA and DNA share nucleic-acid token ids.
    """

    shared_nucleic_tokens: bool = True
