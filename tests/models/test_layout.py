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

"""Behavior of ofoldx.models.layout."""

from __future__ import annotations

import numpy as np
import pytest

from ofoldx.data.molecular import MonomerType
from ofoldx.data.tokenization import TOKEN_INDEX, Encoding
from ofoldx.models.layout import LayoutTable, model_tokens, slot_map, token_frames, token_layout, token_representatives

PROTEIN = MonomerType.PROTEIN


def table(**changes) -> LayoutTable:
    values = {
        "family": "test",
        "monomer_types": {PROTEIN: 0},
        "residue_slots": None,
        "monomer_slots": None,
        "hydrogens": "exclude",
        "per_atom_components": "nonstandard",
        "exclude_designated_leaving": True,
        "centre_atom": {PROTEIN: "CA"},
        "frame_atoms": {PROTEIN: ("N", "CA", "C")},
        "representative": {TOKEN_INDEX["G"]: "CA"},
        "representative_default": {PROTEIN: "CB"},
    }
    return LayoutTable(**(values | changes))


def names(structure, atoms) -> list[str]:
    return structure.system.text("atom", structure.system.atom_symbol[atoms]).tolist()


class TestTokenLayout:
    def test_nonstandard_per_atom(self, peptide):
        structure = peptide("ALA", "MSE", "GLY")
        layout = token_layout(structure.system, table())
        counts = np.bincount(layout.token_component, minlength=3)
        assert counts[0] == counts[2] == 1
        assert counts[1] > 1

    def test_unaddressed_atoms(self, peptide):
        structure = peptide("ALA", "GLY")
        layout = token_layout(structure.system, table())
        addressed = names(structure, layout.token_atoms)
        assert "OXT" not in addressed
        assert not any(name.startswith("H") for name in addressed)

    def test_monomer_slots(self, peptide):
        structure = peptide("ALA", "GLY", ligands=("ACT",))
        layout = token_layout(structure.system, table(monomer_slots={PROTEIN: ("N", "CA", "C", "O")}))
        assert layout.token_component.tolist() == [0, 1]


class TestSlotMap:
    def test_named_slots(self, peptide):
        structure = peptide("ALA", "GLY")
        layout_table = table(monomer_slots={PROTEIN: ("N", "CA", "CB")}, per_atom_components="none")
        slots = slot_map(structure.system, token_layout(structure.system, layout_table), layout_table)
        assert slots.atom.reshape(2, 3)[1, 2] == -1
        assert names(structure, slots.atom[:3]) == ["N", "CA", "CB"]
        assert "O" in names(structure, slots.unaddressed)


class TestModelTokens:
    def test_parent_encoding_loss(self, peptide):
        structure = peptide("ALA", "MSE")
        layout = token_layout(structure.system, table(per_atom_components="none"))
        tokens, loss = model_tokens(structure.system, layout)
        assert tokens.tolist() == [TOKEN_INDEX["A"], TOKEN_INDEX["M"]]
        assert loss.component.tolist() == [1]
        assert loss.encoding.tolist() == [Encoding.PARENT]


@pytest.mark.parametrize("residue,expected", [("GLY", "CA"), ("ALA", "CB")])
def test_representative_atom(peptide, residue, expected):
    structure = peptide(residue)
    layout = token_layout(structure.system, table())
    assert names(structure, token_representatives(structure.system, layout, table())) == [expected]


def test_frame_atoms(peptide):
    structure = peptide("ALA")
    layout = token_layout(structure.system, table())
    assert names(structure, token_frames(structure.system, layout, table())[0]) == ["N", "CA", "C"]
