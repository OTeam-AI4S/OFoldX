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

"""The retention rules of union, replication and batching (ofoldx.data.assembly)."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from ofoldx.data.assembly import Copy, batch, replicate, unbatch, union
from ofoldx.data.molecular import MolecularSystem
from ofoldx.data.projection import project_instances
from ofoldx.data.structure import CopyId, CopyKind, Structure
from ofoldx.errors import UnsupportedError
from ofoldx.provenance import Basis, Resolution, state

UN = state(Resolution.UNKNOWN, Basis.NONE)
SHIFT = np.eye(4)
SHIFT[:3, 3] = [10.0, 0.0, 0.0]
TWOFOLD = np.diag([-1.0, -1.0, 1.0, 1.0])  # transform row 1 (code 2555) in the fixture frame diag(50, 60, 70)
TWOFOLD[2, 3] = 35.0


@pytest.fixture
def read_like(system_arrays, structure_arrays, add_edge) -> Structure:
    """The fixture named as a read names it (instance symbol = asym id: A, B, C), with ALA CB – LIG C4 linked."""
    arrays = system_arrays()
    arrays.symbols["instance"] = ("A", "B", "C", "Z")
    add_edge(arrays.columns, 4, 14)
    return Structure.from_arrays(MolecularSystem.from_arrays(arrays), structure_arrays())


def instance_symbols(value):
    return value.system.text("instance", value.system.instance_symbol).tolist()


def externals(system):
    rows = zip(
        system.external_atom.tolist(),
        system.text("instance", system.external_partner_instance),
        system.external_partner_component.tolist(),
        system.text("atom", system.external_partner_atom),
        system.external_transform.tolist(),
    )
    return Counter(rows)


def same_value(left, right):
    """Equal columns, external rows compared as a multiset (their order is not semantic)."""
    for record_left, record_right in ((left.system, right.system), (left, right)):
        assert record_left.symbols() == record_right.symbols()
        for name, array in record_left.arrays().items():
            if not name.startswith("external_"):
                assert np.array_equal(array, record_right.arrays()[name]), name
    assert externals(left.system) == externals(right.system)


class TestUnion:
    def test_complementary_projections_reunite(self, cross_linked):
        keep = np.array([True, False, False])
        parts = [project_instances(cross_linked, mask)[0] for mask in (keep, ~keep)]
        value, maps = union(parts)
        same_value(value, cross_linked)
        assert maps[1].atom.tolist()[:2] == [len(parts[0].coordinates), len(parts[0].coordinates) + 1]

    def test_identity_stub_rejoins_beside_symmetry(self, system_arrays, structure_arrays, add_edge, add_external):
        arrays = system_arrays()
        add_edge(arrays.columns, 4, 14)  # ALA CB – LIG C4, and LIG C4 also bonded to CB of A's 2_555 image
        add_external(arrays.columns, 14, 0, 0, "CB", 1)
        whole = Structure.from_arrays(MolecularSystem.from_arrays(arrays), structure_arrays())
        keep = np.array([True, False, False])
        value, _ = union([project_instances(whole, mask)[0] for mask in (keep, ~keep)])
        same_value(value, whole)

    def test_entity_sequences_must_agree(self, system_arrays, invalid):
        first = MolecularSystem.from_arrays(system_arrays())
        arrays = system_arrays()
        arrays.symbols["instance"] = ("X", "Y", "Z", "B")
        arrays.columns["component_symbol"][:2] = [1, 0]  # GLY–ALA under entity 1
        second = MolecularSystem.from_arrays(arrays)
        keep = np.array([True, False, False])
        parts = [project_instances(value, keep)[0] for value in (first, second)]
        invalid(union, parts)

    def test_instance_collision(self, structure, invalid):
        invalid(union, [structure, structure], code="instance_collision")

    def test_nothing_to_union_invalid(self, invalid):
        invalid(union, [])

    def test_differing_stubs_mismatch(self, cross_linked, invalid):
        keep = np.array([True, False, False])
        first, second = (project_instances(cross_linked, mask)[0] for mask in (keep, ~keep))
        system = second.system.build()
        row = int(np.flatnonzero(system.columns["external_partner_atom"] == cross_linked.system.atom_symbol[0])[0])
        system.columns["external_order"][row] = 2
        second = Structure.from_arrays(MolecularSystem.from_arrays(system), second.build())
        invalid(union, [first, second], code="stub_mismatch")


class TestBatch:
    def test_nothing_to_batch_invalid(self, invalid):
        invalid(batch, [])

    def test_unbatch_inverts_batch(self, structure):
        batched = batch([structure, structure])
        assert len(batched.system_model_number) == 2
        assert [value.digest() for value in unbatch(batched)] == [structure.digest()] * 2


class TestReplicate:
    def test_copy_repeats_and_places(self, structure):
        value, replica = replicate(
            structure, [Copy(np.array([1]), CopyId(CopyKind.ASSEMBLY, ("2",), None), SHIFT)], keep_source=True
        )
        copied = replica.atom_copy == 0
        assert replica.atom_source[copied].tolist() == list(range(11, 17))
        assert np.allclose(value.coordinates[copied], structure.coordinates[11:17] + [10.0, 0, 0])
        assert not value.coordinate_quantum[copied].any()
        assert value.copy_id(3) == CopyId(CopyKind.ASSEMBLY, ("2",), "B")
        assert instance_symbols(value) == ["A", "L", "W", "B-2"]  # the kept source is copy 1 of asym B
        assert len(value.system.stereocentre_atom) == 3

    def test_copies_are_named_in_order(self, structure):
        copies = [Copy(np.array([1]), CopyId(CopyKind.ASSEMBLY, (k,), None), np.eye(4)) for k in ("1", "2")]
        value, _ = replicate(structure, copies, keep_source=False)
        assert instance_symbols(value) == ["B", "B-2"]

    def test_kept_source_is_first_copy(self, read_like):
        copy = Copy(np.array([0]), CopyId(CopyKind.ASSEMBLY, ("2",), None), SHIFT)
        value, _ = replicate(read_like, [copy], keep_source=True)
        assert instance_symbols(value) == ["A", "B", "C", "A-2"]

    def test_unplaced_partner_transform_unknown(self, read_like):
        copies = [
            Copy(np.array([0, 1, 2]), CopyId(CopyKind.ASSEMBLY, ("1",), None), np.eye(4)),
            Copy(np.array([0]), CopyId(CopyKind.ASSEMBLY, ("2",), None), SHIFT),
        ]
        value, replica = replicate(read_like, copies, keep_source=False)
        system = value.system
        assert instance_symbols(value) == ["A", "B", "C", "A-2"]
        cb = int(np.flatnonzero((replica.atom_copy == 1) & (replica.atom_source == 4))[0])
        row = int(np.flatnonzero(system.external_atom == cb)[0])
        assert system.text("instance", system.external_partner_instance[row : row + 1]).tolist() == ["B"]
        assert system.transform_state[system.external_transform[row]] == UN

    def test_doubly_cut_atom_keeps_slot(self, system_arrays, structure_arrays, add_edge):
        arrays = system_arrays()
        add_edge(arrays.columns, 0, 13)  # LIG C3 bonded to ALA N and to ALA CA, which lists it in a slot
        add_edge(arrays.columns, 1, 13)
        arrays.columns["stereocentre_neighbors"][0] = [0, 2, 4, 13]
        whole = Structure.from_arrays(MolecularSystem.from_arrays(arrays), structure_arrays())
        value, _ = replicate(
            whole, [Copy(np.array([0]), CopyId(CopyKind.ASSEMBLY, ("2",), None), SHIFT)], keep_source=False
        )
        system = value.system
        slot = int(system.stereocentre_neighbors[0][3])
        assert slot <= -2 and system.external_atom[-slot - 2] == system.stereocentre_atom[0]
        assert system.stereocentre_parity[0] == 1

    @pytest.mark.parametrize(
        ("instances", "matrix"), [([-1], np.eye(4)), ([1], np.eye(3))], ids=["negative", "matrix_3x3"]
    )
    def test_copy_checked_before_indexing(self, structure, invalid, instances, matrix):  # never the last instance
        copy = Copy(np.array(instances), CopyId(CopyKind.ASSEMBLY, ("2",), None), matrix)
        invalid(replicate, structure, [copy], keep_source=False)

    def test_improper_copy_is_unsupported(self, structure):
        with pytest.raises(UnsupportedError) as caught:
            replicate(
                structure,
                [Copy(np.array([1]), CopyId(CopyKind.ASSEMBLY, ("2",), None), np.diag([-1.0, 1, 1, 1]))],
                keep_source=True,
            )
        assert caught.value.code == "improper_copy"

    @pytest.mark.parametrize("origin", [0.0, 0.25])
    def test_copy_join_makes_edge(self, system, structure_arrays, origin):  # x ↦ O·[R·(A·x + u) + τ − u]
        arrays = structure_arrays()
        arrays.columns["frame"][0, 0, 3] = origin
        structure = Structure.from_arrays(system, arrays)
        matrix = TWOFOLD.copy()
        matrix[0, 3] = -100.0 * origin  # O·(R·u − u) along a, O = diag(50, 60, 70)
        copy = Copy(np.array([0]), CopyId(CopyKind.CRYSTAL, ("2_555",), None), matrix)
        value, replica = replicate(structure, [copy], keep_source=True)
        oxt = int(np.flatnonzero((replica.atom_copy == 0) & (replica.atom_source == 10))[0])
        edges = set(zip(*value.system.edge_index.tolist()))
        assert (16, oxt) in edges
        assert len(value.system.external_atom) == len(structure.system.external_atom)
