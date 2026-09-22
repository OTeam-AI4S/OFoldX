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

"""The retention rule of the declared chemical edit (ofoldx.data.editing)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from ofoldx.data.assembly import batch
from ofoldx.data.editing import ChemicalEdit, ComponentRows, EdgeRows, apply_edit
from ofoldx.data.molecular import MolecularSystem
from ofoldx.data.stereo import StereoColumns
from ofoldx.data.structure import CoordinateEvidence, Structure
from ofoldx.errors import CODE_NAMES, UnsupportedError
from ofoldx.provenance import Basis, Resolution, state

KD, IA = state(Resolution.KNOWN, Basis.DECLARED), state(Resolution.INAPPLICABLE, Basis.DECLARED)
UN, UDER = state(Resolution.UNKNOWN, Basis.NONE), state(Resolution.UNKNOWN, Basis.DERIVED)
ID = state(Resolution.INAPPLICABLE, Basis.DEFINED)
ATOM_COLUMNS = [f.name for f in MolecularSystem.SCHEMA.fields if f.axis == "atom" and f.name != "atom_symbol"]
BOND_COLUMNS = [
    f.name
    for f in MolecularSystem.SCHEMA.fields
    if f.axis == "edge" and f.name not in ("edge_index", "bond_connection_type")
]


def atoms_like(system, atoms):
    return {name: system.arrays()[name][atoms].copy() for name in ATOM_COLUMNS}


def bonds(count, order=1):
    values = {"bond_order": order, "bond_order_state": KD, "bond_aromatic": False, "bond_aromatic_state": KD}
    values |= {"bond_kind": 1, "bond_kind_state": KD, "bond_direction": 0, "bond_direction_state": UN, "bond_origin": 5}
    return {name: np.full(count, values[name]) for name in BOND_COLUMNS}


def empty_stereo():
    zeros = lambda dtype, *shape: np.zeros((0, *shape), dtype=dtype)  # noqa: E731
    return StereoColumns(
        zeros(np.int32), zeros(np.int8), zeros(np.int32, 4), zeros(np.uint8), zeros(np.int32), zeros(np.int8),
        zeros(np.int32, 2), zeros(np.uint8), zeros(np.uint8), np.zeros(1, np.int64), zeros(np.int32),
    )  # fmt: skip


@pytest.fixture
def edit():
    """An empty edit; tests replace the fields they exercise."""
    return ChemicalEdit(
        remove_atoms=np.zeros(0, dtype=np.int64),
        replace_components={},
        retained_atoms={},
        remove_edges=np.zeros(0, dtype=np.int64),
        add_edges=EdgeRows(np.zeros((2, 0), dtype=np.int64), bonds(0), (), ()),
        set_atoms={},
        claims={},
        keep_coordinates=None,
        identity="inherit",
    )


ALA = {0: (False, False)}
LIG = {2: (False, False)}
CA_CB = 2  # edge (1, 4)
CA, HA = 1, 5
HEAVY = CODE_NAMES.index("parity_cleared_heavy_neighbour_removed")
NEW = CODE_NAMES.index("parity_cleared_new_neighbour_without_vacated_slot")


@pytest.fixture
def unconnected(system_arrays, structure_arrays) -> Structure:
    """The fixture structure without external connections, so that an edit may remove every atom."""
    arrays = system_arrays()
    columns = arrays.columns
    for field in MolecularSystem.SCHEMA.fields:
        if field.axis in ("external", "external_declared_leaving"):
            columns[field.name] = columns[field.name][:0]
    columns["external_declared_leaving_ptr"] = np.zeros(1, dtype=np.int64)
    return Structure.from_arrays(MolecularSystem.from_arrays(arrays), structure_arrays())


@pytest.fixture
def emptying(edit):
    """The edit removing every atom of ``unconnected`` (the junction listed)."""
    claims = {component: (False, False) for component in range(4)}
    return replace(edit, remove_atoms=np.arange(18), remove_edges=np.array([5]), claims=claims)


class TestApplyEdit:
    def test_set_atoms_writes_states(self, system, edit):
        change = {"formal_charge": (np.array([0]), np.array([1]), np.array([KD], dtype=np.uint8))}
        value, atom_map, report = apply_edit(system, replace(edit, set_atoms=change, claims=ALA))
        assert value.formal_charge[0] == 1 and value.formal_charge_state[0] == KD
        assert np.array_equal(value.formal_charge[1:], system.formal_charge[1:])
        assert atom_map.old_to_new.tolist() == list(range(18)) and report.claims_set == ALA

    def test_set_atoms_writes_isotope(self, system, edit):  # a stateless fact: no states
        change = {"isotope": (np.array([0]), np.array([15]), None)}
        value, _, _ = apply_edit(system, replace(edit, set_atoms=change, claims=ALA))
        assert value.isotope[[0, 5]].tolist() == [15, 2]

    # a changed hydrogen count is Unknown unless the edit restates it; an INAPPLICABLE count is decided too
    @pytest.mark.parametrize(
        ("restated", "complete"), [(KD, True), (IA, True), (0, False)], ids=["restated", "inapplicable", "withdrawn"]
    )
    def test_changed_count_needs_restating(self, system, edit, restated, complete):
        rows = (np.array([CA]), np.array([int(restated == KD)]), np.array([restated], np.uint8))
        count = {"implicit_hydrogens": rows} if restated else {}
        change = replace(edit, remove_atoms=np.array([HA]), set_atoms=count, claims=ALA)
        value, atom_map, _ = apply_edit(system, change)
        assert value.implicit_hydrogens_state[atom_map.old_to_new[CA]] == (restated or UN)
        assert value.component_hydrogens_complete.tolist() == [complete, False, True, True]

    def test_touched_components_need_claims(self, system, edit, invalid):
        invalid(apply_edit, system, replace(edit, remove_atoms=np.array([4])), code="edit_invalid")

    def test_removed_neighbour_clears(self, system, edit):
        value, atom_map, report = apply_edit(system, replace(edit, remove_atoms=np.array([4]), claims=ALA))
        assert atom_map.old_to_new[4] == -1 and value.stereocentre_state[0] == UDER
        assert report.parity_clears.reason.tolist() == [HEAVY]
        assert not value.component_topology_complete[0] and value.component_topology_complete[1]

    def test_unvacated_new_neighbour_clears(self, system, edit):
        new_edge = EdgeRows(np.array([[1], [3]]), bonds(1), (None,), ((),))
        value, _, report = apply_edit(system, replace(edit, add_edges=new_edge, claims=ALA))
        assert value.stereocentre_state[0] == UDER and report.parity_clears.reason.tolist() == [NEW]

    def test_connection_never_dropped_silently(self, system, edit, invalid):
        both = {0: (False, False), 1: (False, False)}
        invalid(apply_edit, system, replace(edit, remove_atoms=np.array([6]), claims=both), code="edit_invalid")
        value, _, _ = apply_edit(
            system, replace(edit, remove_atoms=np.array([6]), remove_edges=np.array([5]), claims=both)
        )
        assert len(value.edge_index[0]) == len(system.edge_index[0]) - 2

    def test_removed_edge_clears(self, system, edit):
        value, atom_map, report = apply_edit(system, replace(edit, remove_edges=np.array([CA_CB]), claims=ALA))
        assert (atom_map.old_to_new >= 0).all() and value.stereocentre_state[0] == UDER
        assert report.parity_clears.reason.tolist() == [HEAVY]

    def test_edit_of_atomless_components(self, unconnected, emptying, edit):
        emptied, _, _ = apply_edit(unconnected.system, emptying)
        value, _, _ = apply_edit(emptied, edit)
        assert len(value.atom_symbol) == 0 and (value.component_port == -1).all()

    @pytest.mark.parametrize(
        "change",
        [
            {"remove_atoms": np.array([-1])},
            {"remove_edges": np.array([-1])},
            {"set_atoms": {"formal_charge": (np.array([-1]), np.array([1]), np.array([KD], dtype=np.uint8))}},
            {"claims": {-1: (True, True)}},
        ],
        ids=["remove_atoms", "remove_edges", "set_atoms", "claims"],
    )
    def test_negative_index_refused(self, system, edit, invalid, change):  # never the last row
        every = {component: (False, False) for component in range(4)}
        invalid(apply_edit, system, replace(edit, **({"claims": every} | change)), code="edit_invalid")

    @pytest.mark.parametrize("dtype", [np.int64, np.uint8])
    def test_unrepresentable_charge_refused(self, system, edit, dtype):  # 128 is no int8 charge, never -128
        change = {"formal_charge": (np.array([0]), np.array([128], dtype=dtype), np.array([KD], dtype=np.uint8))}
        with pytest.raises(UnsupportedError) as raised:
            apply_edit(system, replace(edit, set_atoms=change, claims=ALA))
        assert raised.value.code == "precision_loss"

    def test_column_shape_checked_whole(self, system, edit, invalid):  # [2, 1] is no one-dimensional column
        change = {"formal_charge": (np.array([0, 1]), np.zeros((2, 1), dtype=np.int8), np.full(2, KD, dtype=np.uint8))}
        invalid(apply_edit, system, replace(edit, set_atoms=change, claims=ALA), code="edit_invalid")

    def test_externally_connected_atom_refused(self, system, edit, invalid):
        invalid(
            apply_edit,
            system,
            replace(edit, remove_atoms=np.array([10]), claims={1: (False,) * 2}),
            code="edit_invalid",
        )


@pytest.fixture
def replacement(system):
    """LIG replaced by a three-atom LIG2 keeping C1, N1 and O1 (O1 keeps its external connection)."""
    atoms = atoms_like(system, np.array([11, 15, 16])) | {"atom_reference_row": np.array([0, 1, 2])}
    return ComponentRows(
        symbol="LIG2", names=("C1", "N1", "O1"), atoms=atoms,
        edges=np.array([[0, 0], [1, 2]]), bonds=bonds(2), stereo=empty_stereo(), claims=(True, True),
        parents=(), definition_parents=(), monomer_type=0, monomer_type_state=UN,
        definition_scope=4, definition_digest=999, port=(-1, -1),
        reference=np.zeros((3, 3), dtype=np.float32), reference_source=np.full(3, 2, dtype=np.uint8),
    )  # fmt: skip


class TestReplaceComponent:
    def test_component_takes_final_rows(self, structure, edit, replacement):
        keep = np.zeros(18, dtype=bool)
        keep[[11, 15]] = True
        change = replace(
            edit, replace_components={2: replacement}, retained_atoms={2: np.array([11, 15, 16])},
            keep_coordinates=keep, identity="withdraw",
        )  # fmt: skip
        value, atom_map, report = apply_edit(structure, change)
        system = value.system
        assert system.text("component", system.component_symbol).tolist()[2] == "LIG2"
        assert len(system.atom_symbol) == 15 and atom_map.old_to_new[[12, 13, 14]].tolist() == [-1, -1, -1]
        assert len(system.stereocentre_atom) == 1 and len(system.stereobond_edge) == 0
        new_o1 = atom_map.old_to_new[16]
        assert value.coordinate_evidence[new_o1] == CoordinateEvidence.NONE and value.occupancy_state[new_o1] == ID
        assert value.coordinate_evidence[atom_map.old_to_new[11]] == CoordinateEvidence.OBSERVED
        assert report.identity_changed.tolist() == [2] and report.coordinates_invalidated.tolist() == [new_o1]
        assert value.component_author_number_state[2] == UN

    def test_atomless_replacement_has_no_port(self, system, edit, replacement):
        atoms = {name: column[:0] for name, column in replacement.atoms.items()}
        void = replace(
            replacement, symbol="UNL", names=(), atoms=atoms, edges=np.zeros((2, 0), dtype=np.int64), bonds=bonds(0),
            reference=np.zeros((0, 3), dtype=np.float32), reference_source=np.zeros(0, dtype=np.uint8),
        )  # fmt: skip
        value, _, _ = apply_edit(system, replace(edit, replace_components={3: void}))
        assert len(value.atom_symbol) == 17 and value.component_port[3].tolist() == [-1, -1]


@pytest.fixture
def ligand_observed_only(system, structure_arrays) -> Structure:
    """The fixture structure in which only LIG's coordinates are OBSERVED (the others PREDICTED)."""
    arrays = structure_arrays()
    others = np.ones(18, dtype=bool)
    others[11:17] = False
    columns = arrays.columns
    columns["coordinate_evidence"][others & (columns["coordinate_evidence"] != 0)] = CoordinateEvidence.PREDICTED
    for name in ("occupancy", "occupancy_uncertainty", "b_factor"):
        columns[name][others], columns[f"{name}_state"][others] = 0, ID
    return Structure.from_arrays(system, arrays)


class TestEditStructure:
    def test_inherit_keeps_author_numbering(self, structure, edit, replacement):
        change = replace(edit, replace_components={2: replacement}, retained_atoms={2: np.array([11, 15, 16])})
        value, _, report = apply_edit(structure, change)
        assert report.identity_changed.tolist() == [2]
        assert value.component_author_number[2] == structure.component_author_number[2]

    def test_last_observation_withdraws_method(self, ligand_observed_only, edit):
        change = {"formal_charge": (np.array([11]), np.array([0]), np.array([KD], dtype=np.uint8))}
        value, _, _ = apply_edit(ligand_observed_only, replace(edit, set_atoms=change, claims=LIG))
        assert len(value.method) == 0 and len(value.resolution) == 0
        assert value.frame_state.tolist() == ligand_observed_only.frame_state.tolist()
        assert len(value.operation_id) == len(ligand_observed_only.operation_id)

    def test_removing_every_atom(self, unconnected, emptying):
        value, _, _ = apply_edit(unconnected, emptying)
        assert len(value.coordinates) == 0

    def test_withdrawal_is_per_system(self, ligand_observed_only, structure, edit):
        change = {"formal_charge": (np.array([11]), np.array([0]), np.array([KD], dtype=np.uint8))}
        value, _, _ = apply_edit(batch([ligand_observed_only, structure]), replace(edit, set_atoms=change, claims=LIG))
        assert value.system_method_ptr.tolist() == [0, 0, 1] and value.system_resolution_ptr.tolist() == [0, 0, 1]
