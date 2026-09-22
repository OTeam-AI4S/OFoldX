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

"""Behaviour of ofoldx.data.structure: digest, accessors, unobserved values and coordinate replacement."""

from __future__ import annotations

import numpy as np
import pytest

from ofoldx.data.molecular import MolecularSystem
from ofoldx.data.structure import CoordinateEvidence, CopyId, CopyKind, Structure, replace_coordinates
from ofoldx.provenance import Basis, Resolution, known, state

ID, UN = state(Resolution.INAPPLICABLE, Basis.DEFINED), state(Resolution.UNKNOWN, Basis.NONE)
KDER = state(Resolution.KNOWN, Basis.DERIVED)


class TestDigest:
    def test_build_round_trip_keeps_it(self, structure):
        assert Structure.from_arrays(structure.system, structure.build()).digest() == structure.digest()

    def test_structure_byte_changes_it(self, structure):
        arrays = structure.build()
        arrays.columns["b_factor"][0] += 1
        assert Structure.from_arrays(structure.system, arrays).digest() != structure.digest()

    def test_system_byte_changes_it(self, structure):
        arrays = structure.system.build()
        arrays.columns["component_definition_digest"][0] += 1
        changed = Structure.from_arrays(MolecularSystem.from_arrays(arrays), structure.build())
        assert changed.digest() != structure.digest()

    def test_symbol_text_changes_it(self, structure):
        arrays = structure.build()
        arrays.symbols["method"] = ("SOLUTION NMR",)
        assert Structure.from_arrays(structure.system, arrays).digest() != structure.digest()


class TestUnobserved:
    def test_nothing_is_observed(self, system):
        value = Structure.unobserved(system)
        assert (value.coordinate_evidence == CoordinateEvidence.NONE).all()
        assert (value.occupancy_state == ID).all() and (value.b_factor_state == ID).all()
        assert not value.source_sha256.any()
        assert (value.instance_copy_kind == CopyKind.RECIPE).all()

    def test_polymers_are_numbered(self, system):
        value = Structure.unobserved(system)
        assert value.component_sequence_number.tolist() == [1, 2, 0, 0]
        assert value.component_sequence_number_state.tolist() == [KDER, KDER, ID, ID]


class TestAccessors:
    @pytest.mark.parametrize(
        ("kind", "operators"),
        [(CopyKind.DEPOSITED, ()), (CopyKind.ASSEMBLY, ("1", "X0")), (CopyKind.CRYSTAL, ("3_655",))],
    )
    def test_copy_id_is_stated_value(self, kind, operators, system, structure_arrays):
        arrays = structure_arrays()
        arrays.symbols["operator"] = ("\x1f".join(operators),) if operators else ()
        arrays.columns["instance_copy_kind"][0] = kind
        arrays.columns["instance_copy_operators"][0] = 0 if operators else -1
        assert Structure.from_arrays(system, arrays).copy_id(0) == CopyId(kind, operators, "A")

    def test_copy_id_of_no_instance(self, structure, invalid):  # -1 is no instance, never the last
        invalid(structure.copy_id, -1)


@pytest.fixture
def ligand(structure):
    """Scope = the ligand instance L (atoms 11-16), two of them written."""
    scope = np.zeros(len(structure.coordinates), dtype=bool)
    scope[11:17] = True
    return scope, np.array([11, 12]), np.ones((2, 3))


def replace_all(value, **kwargs):
    n = len(value.coordinates)
    return replace_coordinates(value, np.arange(n), np.zeros((n, 3)), scope=np.ones(n, dtype=bool), **kwargs)


class TestReplaceCoordinates:
    def test_scope_is_regenerated(self, structure, ligand):
        scope, atoms, coordinates = ligand
        value, report = replace_coordinates(
            structure, atoms, coordinates, scope=scope, evidence=CoordinateEvidence.PREDICTED
        )
        assert (value.coordinates[atoms] == 1).all()
        assert (value.coordinate_evidence[atoms] == CoordinateEvidence.PREDICTED).all()
        assert (value.coordinate_evidence[13:17] == CoordinateEvidence.NONE).all()
        assert report.invalidated_unwritten.tolist() == [13, 14, 15, 16]
        assert (value.occupancy_state[scope] == ID).all() and report.observations_withdrawn == 6
        assert not value.coordinate_quantum[scope].any()
        kept = ("coordinates", "coordinate_evidence", "coordinate_quantum", "occupancy", "occupancy_state", "b_factor")
        for name in kept:
            assert np.array_equal(value.arrays()[name][~scope], structure.arrays()[name][~scope])

    def test_old_coordinates_never_refill(self, structure, ligand):
        scope, atoms, coordinates = ligand
        moved = structure.build()
        moved.columns["coordinates"][13] += 3.0
        other = Structure.from_arrays(structure.system, moved)
        results = [
            replace_coordinates(value, atoms, coordinates, scope=scope, evidence=CoordinateEvidence.PREDICTED)[0]
            for value in (structure, other)
        ]
        assert results[0].digest() == results[1].digest()

    @pytest.mark.parametrize(
        ("scope_atoms", "evidence", "resolution", "method"),
        [
            pytest.param([3], CoordinateEvidence.GENERATED, True, False, id="observed-replaced"),
            pytest.param([5], CoordinateEvidence.GENERATED, False, False, id="unobserved-written"),
            pytest.param("all", CoordinateEvidence.PREDICTED, True, True, id="prediction"),
        ],
    )
    def test_resolution_and_method_withdrawal(self, scope_atoms, evidence, resolution, method, structure):
        n = len(structure.coordinates)
        atoms = np.arange(n) if scope_atoms == "all" else np.array(scope_atoms)
        scope = np.isin(np.arange(n), atoms)
        value, report = replace_coordinates(structure, atoms, np.zeros((len(atoms), 3)), scope=scope, evidence=evidence)
        assert (report.resolution_withdrawn, report.method_withdrawn) == (resolution, method)
        assert (len(value.resolution) == 0) == resolution and (len(value.method) == 0) == method

    def test_unstated_metadata_not_withdrawn(self, system):
        _, report = replace_all(Structure.unobserved(system), evidence=CoordinateEvidence.PREDICTED)
        assert not report.method_withdrawn and not report.resolution_withdrawn

    def test_new_frame_withdraws_frame(self, structure):
        value, report = replace_all(structure, evidence=CoordinateEvidence.PREDICTED, frame="new")
        assert report.frame_withdrawn and not known(value.cell_state).any() and not known(value.frame_state).any()
        assert value.space_group_state.tolist() == [UN] and len(value.operation_rotation) == 0
        assert np.array_equal(value.system.external_transform, structure.system.external_transform)

    def test_new_frame_needs_full_scope(self, structure, ligand, invalid):
        scope, atoms, coordinates = ligand
        evidence = CoordinateEvidence.PREDICTED
        call = replace_coordinates
        invalid(
            call, structure, atoms, coordinates, scope=scope, evidence=evidence, frame="new", code="scope_violation"
        )

    def test_written_atom_outside_scope(self, structure, ligand, invalid):
        scope, _, _ = ligand
        args = (structure, np.array([0]), np.zeros((1, 3)))
        invalid(replace_coordinates, *args, scope=scope, evidence=CoordinateEvidence.PREDICTED, code="scope_violation")

    @pytest.mark.parametrize(
        ("atoms", "points"), [([-1], [[1.0, 2.0, 3.0]]), ([11, 12], [1.0, 2.0, 3.0])], ids=["negative", "one_point"]
    )
    def test_request_checked_before_writing(self, structure, ligand, invalid, atoms, points):  # never the last atom
        scope, _, _ = ligand
        scope[-1] = True
        invalid(
            replace_coordinates, structure, np.array(atoms), np.array(points), scope=scope,
            evidence=CoordinateEvidence.PREDICTED,
        )  # fmt: skip

    def test_evidence_none_is_invalid(self, structure, ligand, invalid):
        scope, atoms, coordinates = ligand
        invalid(replace_coordinates, structure, atoms, coordinates, scope=scope, evidence=CoordinateEvidence.NONE)

    def test_deposited_observation_is_invalid(self, structure, ligand, invalid):  # no source printed it
        scope, atoms, coordinates = ligand
        invalid(replace_coordinates, structure, atoms, coordinates, scope=scope, evidence=CoordinateEvidence.OBSERVED)
