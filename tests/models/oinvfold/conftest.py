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

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
import torch

from ofoldx.chemistry.construction import PolymerSequence, Recipe, build_system
from ofoldx.data.structure import CoordinateEvidence, Structure, replace_coordinates
from ofoldx.models.oinvfold import OInvFoldPolymerConfig, OInvFoldPolymerModel

SMALL = {
    "hidden_size": 8,
    "num_encoder_blocks": 1,
    "num_virtual_atoms": 2,
    "num_attention_heads": 1,
    "num_geometric_layers": 1,
    "num_edge_layers": 1,
    "decoder_hidden_size": 8,
}


@pytest.fixture
def polymer_model():
    torch.manual_seed(83)
    return OInvFoldPolymerModel(OInvFoldPolymerConfig(SMALL)).double().eval()


@pytest.fixture
def chains(library) -> Callable[..., Structure]:
    """``chains("polydeoxyribonucleotide", "DA", "DA", instances=("A",))``: one chain of those residues per instance,
    placed as ``peptide`` is."""

    def build(polymer_type: str, *components: str, instances: tuple[str, ...] = ("A",)) -> Structure:
        recipe = Recipe({"P": PolymerSequence(polymer_type, components)}, dict.fromkeys(instances, "P"))
        system, _, _ = build_system(recipe, library)
        reference, _ = system.reference_coordinates()
        owner = np.repeat(np.arange(len(system.component_symbol)), np.diff(system.component_atom_ptr))
        coordinates = reference + 4.0 * owner[:, None] * np.array([1.0, 0.0, 0.0])
        every = np.ones(len(owner), dtype=bool)
        structure, _ = replace_coordinates(
            Structure.unobserved(system),
            np.arange(len(owner)),
            coordinates,
            scope=every,
            evidence=CoordinateEvidence.OBSERVED,
        )
        return structure

    return build
