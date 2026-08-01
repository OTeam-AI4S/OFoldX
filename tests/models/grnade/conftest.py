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

import numpy as np
import pytest
import torch

from ofoldx.chemistry.construction import PolymerSequence, Recipe, build_system
from ofoldx.data.structure import CoordinateEvidence, replace_coordinates
from ofoldx.models.grnade import GrnadeConfig, GrnadeModel


@pytest.fixture
def rna(library, placed):
    """``rna(3)``: chain ``A`` of that many adenosines, placed."""

    def build(length: int = 3):
        entities = {"R": PolymerSequence("polyribonucleotide", ("A",) * length)}
        system, _, _ = build_system(Recipe(entities, {"A": "R"}), library)
        return placed(system)

    return build


@pytest.fixture
def unobserve():
    """``unobserve(structure, atom)``: the structure with that atom's coordinates ``NONE``."""

    def apply(structure, atom: int):
        scope = np.zeros(len(structure.coordinates), dtype=bool)
        scope[atom] = True
        empty = np.zeros(0, dtype=np.int64)
        return replace_coordinates(
            structure, empty, np.zeros((0, 3)), scope=scope, evidence=CoordinateEvidence.OBSERVED
        )[0]

    return apply


@pytest.fixture
def make_model():
    def make(edge_scalar_size: int) -> GrnadeModel:
        torch.manual_seed(61)
        config = GrnadeConfig(
            node_hidden_scalar_size=8,
            node_hidden_vector_size=2,
            edge_scalar_size=edge_scalar_size,
            edge_hidden_scalar_size=8,
            edge_hidden_vector_size=2,
            num_blocks=1,
            dropout=0.0,
        )
        return GrnadeModel(config).double().eval()

    return make
