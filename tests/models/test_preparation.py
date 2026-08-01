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

"""Behavior of ofoldx.models.preparation."""

from __future__ import annotations


import numpy as np
import pytest

from ofoldx.data.molecular import EndpointState, MolecularSystem, MonomerType
from ofoldx.data.structure import CoordinateEvidence, Structure
from ofoldx.errors import ContradictionError, InsufficientInformationError, InvalidRequestError
from ofoldx.models.layout import LayoutTable, slot_map, token_layout
from ofoldx.models.preparation import (
    ConformerSource,
    GatingPolicy,
    PreparationContext,
    PreparationRecord,
    PreparedInput,
    RegenerationScope,
    decode_coordinates,
    gate,
    missing_reference,
    reference_conformers,
)

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


def prepared(structure: Structure, scope: RegenerationScope) -> PreparedInput[None]:
    slots = slot_map(structure.system, token_layout(structure.system, TABLE), TABLE)
    record = PreparationRecord(gate(structure, slots, GatingPolicy()), np.zeros(0, dtype=np.uint8))
    return PreparedInput(None, PreparationContext(structure, slots, scope, record))


def rebuilt(structure: Structure, **cells) -> Structure:
    arrays = structure.system.build()
    for column, (index, value) in cells.items():
        arrays.columns[column][index] = value
    return Structure.from_arrays(MolecularSystem.from_arrays(arrays), structure.build())


class TestRegenerationScope:
    @pytest.mark.parametrize(
        "make,count,frame",
        [(RegenerationScope.none, 0, "source"), (RegenerationScope.all, None, "new")],
    )
    def test_whole_rules(self, peptide, make, count, frame):
        system = peptide("ALA").system
        scope = make(system)
        assert int(scope.atoms.sum()) == (len(system.atom_symbol) if count is None else count)
        assert scope.frame == frame

    def test_components_keep_atoms(self, peptide):
        system = peptide("ALA", "GLY").system
        keep = np.zeros(len(system.atom_symbol), dtype=bool)
        keep[system.component_atom_ptr[1]] = True
        scope = RegenerationScope.components(system, np.array([1]), keep_atoms=keep)
        assert int(scope.atoms.sum()) == int(np.diff(system.component_atom_ptr)[1]) - 1


class TestDecode:
    def test_unpredicted_atoms_invalid(self, peptide):
        structure = peptide("ALA")
        task = prepared(structure, RegenerationScope.all(structure.system))
        decoded, report = decode_coordinates(task, np.ones((len(task.context.slots.atom), 3)))
        assert (decoded.coordinate_evidence[task.context.slots.atom] == CoordinateEvidence.PREDICTED).all()
        assert len(report.invalidated_unwritten) == len(structure.coordinates) - len(task.context.slots.atom)
        assert report.frame_withdrawn

    def test_input_coordinates_ignored(self, peptide):
        structure = peptide("ALA", "GLY")
        moved = structure.build()
        moved.columns["coordinates"] += 10.0
        output = np.arange(8 * 3, dtype=np.float64).reshape(8, 3)
        first = decode_coordinates(prepared(structure, RegenerationScope.all(structure.system)), output)[0]
        other = Structure.from_arrays(structure.system, moved)
        second = decode_coordinates(prepared(other, RegenerationScope.all(other.system)), output)[0]
        assert first.digest() == second.digest()

    def test_outside_scope_refused(self, peptide):
        structure = peptide("ALA")
        task = prepared(structure, RegenerationScope.none(structure.system))
        with pytest.raises(InvalidRequestError, match="scope_violation"):
            decode_coordinates(task, np.zeros((len(task.context.slots.atom), 3)))


class TestGate:
    def test_conflict_outside_slots(self, peptide):
        structure = peptide("ALA", ligands=("ACT",))
        ligand_atom = int(structure.system.component_atom_ptr[1])
        conflicted = rebuilt(
            structure, atom_endpoint_state=(ligand_atom, EndpointState.CONFLICT), component_topology_complete=(1, False)
        )
        task = prepared(conflicted, RegenerationScope.none(conflicted.system))
        assert task.context.record.records.counts() == {"prepare_accepted_incomplete": 1}

    def test_conflict_bonded_to_slot(self, peptide):
        structure = peptide("ALA")
        beta = int(np.flatnonzero(structure.system.text("atom", structure.system.atom_symbol) == "CB")[0])
        conflicted = rebuilt(
            structure, atom_endpoint_state=(beta, EndpointState.CONFLICT), component_topology_complete=(0, False)
        )
        with pytest.raises(ContradictionError, match="prepare_source_conflict"):
            prepared(conflicted, RegenerationScope.none(conflicted.system))

    def test_incomplete_outside_slots(self, peptide):
        structure = peptide("ALA", ligands=("ACT",))
        open_ligand = rebuilt(structure, component_topology_complete=(1, False))
        task = prepared(open_ligand, RegenerationScope.none(open_ligand.system))
        assert task.context.record.records.counts() == {"prepare_accepted_incomplete": 1}

    def test_unknown_required_state(self, peptide):
        structure = peptide("ALA")
        unknown = rebuilt(structure, formal_charge_state=(1, 8), formal_charge=(1, 0))
        slots = slot_map(unknown.system, token_layout(unknown.system, TABLE), TABLE)
        with pytest.raises(InsufficientInformationError, match="prepare_unknown_feature"):
            gate(unknown, slots, GatingPolicy(required_states=frozenset({"formal_charge"})))


class TestReferenceConformers:
    def test_pool_coordinates(self, peptide):
        structure = peptide("ALA")
        slots = prepared(structure, RegenerationScope.none(structure.system)).context.slots
        coordinates, source, _ = reference_conformers(structure, slots, embed=np.zeros(1, dtype=bool), seed=0)
        pool, _ = structure.system.reference_coordinates()
        np.testing.assert_array_equal(coordinates, pool[slots.atom])
        assert (source == ConformerSource.REFERENCE).all()

    def test_unconsumed_atom_ignored(self, peptide):
        structure = peptide("ALA")
        beta = int(np.flatnonzero(structure.system.text("atom", structure.system.atom_symbol) == "CB")[0])
        extra = rebuilt(structure, atom_reference_row=(beta, -1))
        slots = prepared(extra, RegenerationScope.none(extra.system)).context.slots
        _, source, _ = reference_conformers(extra, slots, embed=np.zeros(1, dtype=bool), seed=0)
        assert (source == ConformerSource.REFERENCE).all()

    def test_missing_not_embedded(self, peptide):
        structure = peptide("ALA")
        missing = rebuilt(structure, atom_reference_row=(slice(None), -1))
        assert missing_reference(missing.system).tolist() == [True]
        slots = prepared(missing, RegenerationScope.none(missing.system)).context.slots
        with pytest.raises(InsufficientInformationError, match="prepare_reference_missing"):
            reference_conformers(missing, slots, embed=np.zeros(1, dtype=bool), seed=0)
