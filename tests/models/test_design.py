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

"""Behavior of ofoldx.models.design."""

from __future__ import annotations

import numpy as np
import pytest

from ofoldx.data.molecular import ComponentRef, MonomerType
from ofoldx.data.structure import CoordinateEvidence
from ofoldx.data.tokenization import TOKEN_INDEX
from ofoldx.errors import InvalidRequestError
from ofoldx.models.design import PositionSelection, produce_design, resolve_positions
from ofoldx.models.layout import LayoutTable, slot_map, token_layout
from ofoldx.models.preparation import PreparationContext, PreparationRecord, PreparedInput, RegenerationScope
from ofoldx.provenance import Records

PROTEIN = MonomerType.PROTEIN
TABLE = LayoutTable(
    family="test",
    monomer_types={PROTEIN: 0},
    residue_slots=None,
    monomer_slots={PROTEIN: ("N", "CA", "C", "O")},
    hydrogens="exclude",
    per_atom_components="none",
    exclude_designated_leaving=False,
    centre_atom={PROTEIN: "CA"},
    frame_atoms={},
    representative={},
    representative_default={},
)


@pytest.fixture
def task(peptide):
    structure = peptide("ALA", "MSE", "GLY")
    slots = slot_map(structure.system, token_layout(structure.system, TABLE), TABLE)
    record = PreparationRecord(Records.empty(), np.zeros(3, dtype=np.uint8))
    return PreparedInput(None, PreparationContext(structure, slots, RegenerationScope.none(structure.system), record))


class TestResolvePositions:
    @pytest.mark.parametrize(
        "selection,expected",
        [
            (PositionSelection(), [0, 1, 2]),
            (PositionSelection(components=(ComponentRef("A", 1),)), [1]),
            (PositionSelection(instances=("A",), exclude=(ComponentRef("A", 0),)), [1, 2]),
            (PositionSelection(indices=np.array([2])), [2]),
        ],
    )
    def test_forms(self, task, selection, expected):
        allowed = np.ones(3, dtype=bool)
        assert resolve_positions(task.context.structure, selection, allowed=allowed).tolist() == expected

    @pytest.mark.parametrize(
        "selection",
        [
            PositionSelection(components=(ComponentRef("A", 0),), indices=np.array([0])),
            PositionSelection(components=(ComponentRef("Z", 0),)),
            PositionSelection(indices=np.array([1])),
        ],
        ids=["several_forms", "unknown", "not_allowed"],
    )
    def test_refused(self, task, selection):
        allowed = np.array([True, False, True])
        with pytest.raises(InvalidRequestError, match="request_invalid_positions"):
            resolve_positions(task.context.structure, selection, allowed=allowed)


class TestProduceDesign:
    def test_encoded_class_retained(self, task, library):
        classes = np.array([TOKEN_INDEX["M"]])
        designed, record = produce_design(task, classes, positions=np.array([1]), table=TABLE, library=library)
        assert record.positions[0].basis == "retained_encoded"
        assert designed.system.text("component", designed.system.component_symbol).tolist() == ["ALA", "MSE", "GLY"]
        assert record.replaced().tolist() == []

    def test_replacement_guides(self, task, library):
        classes = np.array([TOKEN_INDEX["G"]])
        designed, record = produce_design(task, classes, positions=np.array([0]), table=TABLE, library=library)
        system = designed.system
        assert system.text("component", system.component_symbol).tolist()[0] == "GLY"
        assert record.replaced().tolist() == [0]
        start, stop = system.component_atom_ptr[0], system.component_atom_ptr[1]
        guided = system.text("atom", system.atom_symbol[start:stop]) == "CA"
        assert designed.coordinate_evidence[start:stop][guided].tolist() == [CoordinateEvidence.OBSERVED]

    def test_unmappable_class(self, task, library):
        with pytest.raises(InvalidRequestError, match="design_token_unmappable"):
            produce_design(task, np.array([TOKEN_INDEX["X"]]), positions=np.array([0]), table=TABLE, library=library)
