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

"""Synthetic canonical values for the data-layer tests.

One system of three instances: ``A`` (entity 1, a protein dipeptide ALA–GLY with a junction edge declaring ALA OXT
leaving, a materialised deuterium HA and a KNOWN stereocentre at ALA CA), ``L`` (entity 2, ligand LIG with an
implicit-slot stereocentre, a KNOWN stereobond, a symmetry external connection to A GLY OXT and a declared leaving id)
and ``W`` (entity 3, one water). GLY OXT carries an identity external connection to an atom absent from the value (a
projection cut). Author chain ``A`` spans A and L; instance ``L`` is asym ``B``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from ofoldx.data.molecular import MolecularSystem, SystemArrays
from ofoldx.data.structure import Structure, StructureArrays
from ofoldx.provenance import Basis, Resolution, state

KR, KD, KDEF = (state(Resolution.KNOWN, b) for b in (Basis.REFERENCE, Basis.DECLARED, Basis.DEFINED))
KDER, KDFT = state(Resolution.KNOWN, Basis.DERIVED), state(Resolution.KNOWN, Basis.DEFAULT)
UN, UD = state(Resolution.UNKNOWN, Basis.NONE), state(Resolution.UNKNOWN, Basis.DECLARED)
ID = state(Resolution.INAPPLICABLE, Basis.DEFINED)

#: atom names per component: ALA (0-5), GLY (6-10), LIG (11-16), HOH (17)
ATOMS = ("N", "CA", "C", "O", "CB", "HA", "N", "CA", "C", "O", "OXT", "C1", "C2", "C3", "C4", "N1", "O1", "O")
ATOM_FAMILY = ("N", "CA", "C", "O", "CB", "HA", "OXT", "C1", "C2", "C3", "C4", "N1", "O1", "HXT", "ZN")
ELEMENT = (7, 6, 6, 8, 6, 1, 7, 6, 6, 8, 8, 6, 6, 6, 6, 7, 8, 8)
EDGES = ((0, 1), (1, 2), (1, 4), (1, 5), (2, 3), (2, 6), (6, 7), (7, 8), (8, 9), (8, 10))
EDGES += ((11, 12), (11, 15), (11, 16), (12, 13), (13, 14))
DOUBLE = {(2, 3), (8, 9), (12, 13)}
JUNCTION = 5  # edge (2, 6)
STEREOBOND_EDGE = 13  # edge (12, 13)
N_ATOMS = len(ATOMS)


def _a(values, dtype) -> np.ndarray:
    return np.array(values, dtype=dtype)


def system_columns() -> dict[str, np.ndarray]:
    n, m = N_ATOMS, len(EDGES)
    implicit = (2, 0, 0, 0, 3, 0, 1, 2, 0, 0, 0, 1, 1, 1, 3, 2, 0, 2)
    valence = (3, 4, 4, 2, 4, 1, 3, 4, 4, 2, 2, 4, 4, 4, 4, 3, 2, 2)
    endpoint = [0] * n
    endpoint[2], endpoint[6], endpoint[16] = 2, 2, 1
    designation = [0] * n
    designation[10] = 2
    implicit_state = [KR] * n
    implicit_state[10] = UN
    isotope = [0] * n
    isotope[5] = 2
    charge_state = [KR] * n
    charge_state[0], charge_state[17] = KD, KDFT
    order = [2 if edge in DOUBLE else 1 for edge in EDGES]
    order_state = [KR] * m
    order_state[JUNCTION] = KDEF
    origin = [0] * m
    origin[JUNCTION] = 1
    rotation = np.stack((np.eye(3), np.diag([-1, -1, 1]))).astype(np.int8)
    reference = np.arange(17 * 3, dtype=np.float32).reshape(17, 3) / 10
    return {
        "system_entity_ptr": _a([0, 3], np.int64),
        "system_instance_ptr": _a([0, 3], np.int64),
        "entity_symbol": _a([0, 1, 2], np.int32),
        "entity_type": _a([1, 2, 5], np.uint8),
        "entity_type_state": _a([KD, KD, KD], np.uint8),
        "entity_polymer_type": _a([0, -1, -1], np.int32),
        "entity_polymer_type_state": _a([KD, ID, ID], np.uint8),
        "entity_nstd_linkage": _a([1, 0, 0], np.uint8),
        "entity_nstd_linkage_state": _a([KD, ID, ID], np.uint8),
        "instance_symbol": _a([0, 1, 2], np.int32),
        "instance_entity": _a([0, 1, 2], np.int32),
        "instance_connections_complete": _a([True, True, True], bool),
        "instance_component_ptr": _a([0, 2, 3, 4], np.int64),
        "component_symbol": _a([0, 1, 2, 3], np.int32),
        "component_definition_scope": _a([1, 1, 3, 1], np.uint8),
        "component_definition_digest": _a([101, 102, 103, 104], np.int64),
        "component_monomer_type": _a([1, 1, 0, 0], np.uint8),
        "component_monomer_type_state": _a([KDEF, KDEF, UN, UN], np.uint8),
        "component_port": _a([[0, 2], [6, 8], [-1, -1], [-1, -1]], np.int32),
        "component_topology_complete": _a([True, True, True, True], bool),
        "component_heavy_atoms_complete": _a([True, True, True, True], bool),
        "component_atom_ptr": _a([0, 6, 11, 17, 18], np.int64),
        "component_parent_ptr": _a([0, 0, 0, 1, 1], np.int64),
        "component_parent": _a([0], np.int32),
        "component_definition_parent_ptr": _a([0, 0, 0, 1, 1], np.int64),
        "component_definition_parent": _a([0], np.int32),
        "atom_symbol": _a([ATOM_FAMILY.index(name) for name in ATOMS], np.int32),
        "atomic_number": _a(ELEMENT, np.uint8),
        "atomic_number_state": _a([KR] * n, np.uint8),
        "isotope": _a(isotope, np.uint16),
        "formal_charge": _a([0] * n, np.int8),
        "formal_charge_state": _a(charge_state, np.uint8),
        "implicit_hydrogens": _a([0 if s == UN else v for v, s in zip(implicit, implicit_state)], np.int8),
        "implicit_hydrogens_state": _a(implicit_state, np.uint8),
        "atom_valence": _a(valence, np.int8),
        "atom_valence_state": _a([KR] * n, np.uint8),
        "atom_aromatic": _a([False] * n, bool),
        "atom_aromatic_state": _a([KR] * n, np.uint8),
        "atom_fate": _a([0] * n, np.uint8),
        "atom_endpoint_state": _a(endpoint, np.uint8),
        "atom_designation": _a(designation, np.uint8),
        "atom_reference_row": _a([*range(17), -1], np.int32),
        "reference_pool_coordinates": reference,
        "reference_pool_source": _a([1] * 11 + [2] * 6, np.uint8),
        "edge_index": _a(EDGES, np.int32).T.copy(),
        "bond_order": _a(order, np.uint8),
        "bond_order_state": _a(order_state, np.uint8),
        "bond_aromatic": _a([False] * m, bool),
        "bond_aromatic_state": _a([KR] * m, np.uint8),
        "bond_kind": _a([1] * m, np.uint8),
        "bond_kind_state": _a(order_state, np.uint8),
        "bond_direction": _a([0] * m, np.int8),
        "bond_direction_state": _a([ID] * m, np.uint8),
        "bond_origin": _a(origin, np.uint8),
        "bond_connection_type": _a([-1] * m, np.int32),
        "bond_declared_leaving_ptr": _a([0] * (JUNCTION + 1) + [1] * (m - JUNCTION), np.int64),
        "bond_declared_leaving": _a([ATOM_FAMILY.index("OXT")], np.int32),
        "bond_declared_leaving_side": _a([0], np.uint8),
        "external_atom": _a([16, 10], np.int32),
        "external_partner_instance": _a([0, 3], np.int32),
        "external_partner_component": _a([1, 0], np.int32),
        "external_partner_atom": _a([ATOM_FAMILY.index("OXT"), ATOM_FAMILY.index("ZN")], np.int32),
        "external_transform": _a([1, 0], np.int32),
        "external_order": _a([1, 0], np.uint8),
        "external_order_state": _a([KD, ID], np.uint8),
        "external_aromatic": _a([False, False], bool),
        "external_aromatic_state": _a([UN, UN], np.uint8),
        "external_kind": _a([1, 2], np.uint8),
        "external_kind_state": _a([KD, KD], np.uint8),
        "external_direction": _a([0, 0], np.int8),
        "external_direction_state": _a([UN, UN], np.uint8),
        "external_origin": _a([3, 3], np.uint8),
        "external_connection_type": _a([0, 1], np.int32),
        "external_declared_leaving_ptr": _a([0, 1, 1], np.int64),
        "external_declared_leaving": _a([ATOM_FAMILY.index("HXT")], np.int32),
        "external_declared_leaving_side": _a([1], np.uint8),
        "transform_rotation": rotation,
        "transform_translation": _a([[0, 0, 0], [0, 0, 12]], np.int16),
        "transform_code": _a([-1, 2555], np.int32),
        "transform_state": _a([KDEF, KDER], np.uint8),
        "stereocentre_atom": _a([1, 11], np.int32),
        "stereocentre_parity": _a([1, 2], np.int8),
        "stereocentre_neighbors": _a([[0, 2, 4, 5], [12, 15, 16, -1]], np.int32),
        "stereocentre_state": _a([KR, KR], np.uint8),
        "stereobond_edge": _a([STEREOBOND_EDGE], np.int32),
        "stereobond_value": _a([2], np.int8),
        "stereobond_reference": _a([[11, 14]], np.int32),
        "stereobond_state": _a([KR], np.uint8),
        "stereogroup_kind": _a([0], np.uint8),
        "stereogroup_ptr": _a([0, 1], np.int64),
        "stereogroup_member": _a([1], np.int32),
    }


SYSTEM_SYMBOLS = {
    "entity": ("1", "2", "3"),
    "instance": ("A", "L", "W", "B"),
    "component": ("ALA", "GLY", "LIG", "HOH"),
    "atom": ATOM_FAMILY,
    "connection_type": ("covale", "metalc"),
    "parent": ("GLY",),
    "polymer_type": ("polypeptide(L)",),
}


def structure_columns() -> dict[str, np.ndarray]:
    n = N_ATOMS
    observed = np.ones(n, dtype=bool)
    observed[5] = False  # HA: materialised from the definition, no atom_site row
    coordinates = np.where(observed[:, None], np.arange(n * 3, dtype=np.float64).reshape(n, 3) / 7, 0.0)
    return {
        "coordinates": coordinates,
        "coordinate_evidence": observed.astype(np.uint8),
        "coordinate_quantum": np.where(observed, 0.001, 0.0),
        "occupancy": np.where(observed, 1.0, 0.0),
        "occupancy_state": np.where(observed, KD, ID).astype(np.uint8),
        "occupancy_uncertainty": np.zeros(n),
        "occupancy_uncertainty_state": np.where(observed, UN, ID).astype(np.uint8),
        "b_factor": np.where(observed, 20.0, 0.0),
        "b_factor_state": np.where(observed, KD, ID).astype(np.uint8),
        "atom_site_charge": _a([0] * n, np.int8),
        "atom_site_charge_state": np.where(observed, UD, UN).astype(np.uint8),
        "atom_altloc": _a([-1] * n, np.int32),
        "component_sequence_number": _a([1, 2, 0, 0], np.int32),
        "component_sequence_number_state": _a([KD, KD, ID, ID], np.uint8),
        "component_author_text": _a([0, 1, 2, 3], np.int32),
        "component_author_number": _a([1, 2, 101, 201], np.int64),
        "component_author_number_state": _a([KD] * 4, np.uint8),
        "component_insertion": _a([-1] * 4, np.int32),
        "instance_copy_kind": _a([0, 0, 0], np.uint8),
        "instance_copy_operators": _a([-1, -1, -1], np.int32),
        "instance_source_asym": _a([0, 1, 2], np.int32),
        "instance_author_chain": _a([0, 0, 1], np.int32),
        "instance_author_chain_state": _a([KD] * 3, np.uint8),
        "source_sha256": np.arange(1, 33, dtype=np.uint8)[None],
        "system_model_number": _a([1], np.int32),
        "system_model_number_state": _a([KD], np.uint8),
        "cell": _a([[50, 60, 70, 90, 90, 90]], np.float64),
        "cell_state": _a([KD], np.uint8),
        "cell_quantum": np.full((1, 6), 0.01),
        "frame": (np.diag([1 / 50, 1 / 60, 1 / 70]) @ np.eye(3, 4))[None],  # [A | u]
        "frame_state": _a([KDER], np.uint8),
        "frame_quantum": np.zeros((1, 3, 4)),
        "space_group": _a([0], np.int32),
        "space_group_state": _a([KD], np.uint8),
        "operation_numbering": _a([1], np.uint8),
        "system_method_ptr": _a([0, 1], np.int64),
        "system_resolution_ptr": _a([0, 1], np.int64),
        "operation_ptr": _a([0, 2], np.int64),
        "method": _a([0], np.int32),
        "resolution": _a([1.8], np.float64),
        "resolution_kind": _a([1], np.uint8),
        "resolution_state": _a([KD], np.uint8),
        "operation_id": _a([0, 1], np.int32),
        "operation_rotation": np.stack((np.eye(3), np.diag([-1, 1, -1]))).astype(np.int8),
        "operation_translation": _a([[0, 0, 0], [0, 12, 0]], np.int16),
    }


STRUCTURE_SYMBOLS = {
    "asym": ("A", "B", "C"),
    "chain": ("A", "B"),
    "author_number": ("1", "2", "101", "201"),
    "insertion": (),
    "altloc": (),
    "operator": (),
    "method": ("X-RAY DIFFRACTION",),
    "space_group": ("P 1 21 1",),
    "operation": ("1", "2"),
}


@pytest.fixture
def system_arrays() -> Callable[..., SystemArrays]:
    """A fresh valid build buffer; ``symbols`` replaces the families."""

    def make(symbols=None) -> SystemArrays:
        return SystemArrays(dict(SYSTEM_SYMBOLS if symbols is None else symbols), system_columns())

    return make


@pytest.fixture
def system(system_arrays) -> MolecularSystem:
    return MolecularSystem.from_arrays(system_arrays())


@pytest.fixture
def structure_arrays() -> Callable[[], StructureArrays]:
    """A fresh valid structure build buffer."""
    return lambda: StructureArrays(dict(STRUCTURE_SYMBOLS), structure_columns())


@pytest.fixture
def structure(system, structure_arrays) -> Structure:
    return Structure.from_arrays(system, structure_arrays())


@pytest.fixture
def add_edge() -> Callable[[dict, int, int], None]:
    """Insert a covalent single edge (a, b) of origin CONNECTION in canonical order, keeping stereobond edges."""

    def insert(columns: dict, a: int, b: int) -> None:
        first, second = columns["edge_index"].astype(np.int64)
        position = int(np.searchsorted(first * 1000 + second, a * 1000 + b))
        for name, value in (
            ("bond_order", 1),
            ("bond_order_state", KD),
            ("bond_aromatic", False),
            ("bond_aromatic_state", KD),
            ("bond_kind", 1),
            ("bond_kind_state", KD),
            ("bond_direction", 0),
            ("bond_direction_state", UN),
            ("bond_origin", 3),
            ("bond_connection_type", 0),
        ):
            columns[name] = np.insert(columns[name], position, value)
        columns["edge_index"] = np.insert(columns["edge_index"], position, [a, b], axis=1)
        columns["bond_declared_leaving_ptr"] = np.insert(columns["bond_declared_leaving_ptr"], position, 0)
        edge = columns["stereobond_edge"]
        columns["stereobond_edge"] = np.where(edge >= position, edge + 1, edge).astype(np.int32)

    return insert


@pytest.fixture
def add_external() -> Callable[..., None]:
    """Append a covalent single external connection of ``atom`` to (instance code, component ordinal, atom name)."""

    def append(columns: dict, atom: int, instance: int, component: int, name: str, transform: int) -> None:
        row = {
            "external_atom": atom,
            "external_partner_instance": instance,
            "external_partner_component": component,
            "external_partner_atom": ATOM_FAMILY.index(name),
            "external_transform": transform,
            "external_order": 1,
            "external_order_state": KD,
            "external_aromatic": False,
            "external_aromatic_state": KD,
            "external_kind": 1,
            "external_kind_state": KD,
            "external_direction": 0,
            "external_direction_state": UN,
            "external_origin": 3,
            "external_connection_type": 0,
        }
        for key, value in row.items():
            columns[key] = np.append(columns[key], np.array([value], dtype=columns[key].dtype))
        pointer = columns["external_declared_leaving_ptr"]
        columns["external_declared_leaving_ptr"] = np.append(pointer, pointer[-1])

    return append


@pytest.fixture
def cross_linked(system_arrays, structure_arrays, add_edge) -> Structure:
    """The fixture with ALA N – LIG C1 linked, LIG C1's implicit slot taken by ALA N."""
    arrays = system_arrays()
    add_edge(arrays.columns, 0, 11)
    arrays.columns["stereocentre_neighbors"][1] = [12, 15, 16, 0]
    return Structure.from_arrays(MolecularSystem.from_arrays(arrays), structure_arrays())
