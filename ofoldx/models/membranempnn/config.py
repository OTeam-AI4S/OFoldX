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

r"""MembraneMPNN architecture configuration."""

from __future__ import annotations

from typing import Literal

from chanfig import Config

from ..proteinmpnn.config import ProteinMpnnProcessorConfig

MembraneLabelMode = Literal["global", "per_residue"]


class MembraneMpnnConfig(Config):
    r"""Configure the shared graph of the two membrane-label checkpoints.

    Attributes:
        label_mode: The checkpoint's global or per-residue label meaning. None is unknown and cannot be used.
        hidden_size: Width of node, edge and decoder hidden states.
        num_encoder_blocks: Number of residue graph encoder blocks.
        num_decoder_blocks: Number of autoregressive decoder blocks.
        num_neighbors: Number of nearest C-alpha neighbours.
        dropout: Probability of encoder and decoder dropout.
    """

    label_mode: MembraneLabelMode | None = None
    hidden_size: int = 128
    num_encoder_blocks: int = 3
    num_decoder_blocks: int = 3
    num_neighbors: int = 48
    dropout: float = 0.0


class MembraneMpnnProcessorConfig(ProteinMpnnProcessorConfig):
    r"""Persistent membrane-label meaning for a Processor and its paired Model."""

    label_mode: MembraneLabelMode | None = None
