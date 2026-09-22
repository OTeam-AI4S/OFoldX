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

"""Behaviour of ofoldx.data.stereo: parity primitives, coordinate parity and the one remap rule."""

from __future__ import annotations

import numpy as np
import pytest

from ofoldx.data.stereo import (
    StereoColumns,
    bond_configuration,
    parity_from_coordinates,
    permutation_parity,
    remap_parity,
)
from ofoldx.errors import CODE_NAMES
from ofoldx.provenance import Basis, Resolution, state

KR, KDER = state(Resolution.KNOWN, Basis.REFERENCE), state(Resolution.KNOWN, Basis.DERIVED)
UR, UDER = state(Resolution.UNKNOWN, Basis.REFERENCE), state(Resolution.UNKNOWN, Basis.DERIVED)
TETRAHEDRON = np.array([[0, 0, 0], [0, 0, 1], [1, 0, -0.33], [-0.5, 0.87, -0.33], [-0.5, -0.87, -0.33]], dtype=float)
QUANTUM = np.full(5, 0.001)


def stereo(rows, bonds=(), groups=()) -> StereoColumns:
    """Rows ``(centre, parity, slots, state)``, bonds ``(edge, value, refs, state)``, groups ``(kind, rows)``."""
    rows, bonds = list(rows), list(bonds)
    sizes = [len(members) for _, members in groups]
    return StereoColumns(
        centre=np.array([r[0] for r in rows], dtype=np.int32),
        parity=np.array([r[1] for r in rows], dtype=np.int8),
        neighbors=np.array([r[2] for r in rows], dtype=np.int32).reshape(-1, 4),
        centre_state=np.array([r[3] for r in rows], dtype=np.uint8),
        bond_edge=np.array([b[0] for b in bonds], dtype=np.int32),
        bond_value=np.array([b[1] for b in bonds], dtype=np.int8),
        bond_reference=np.array([b[2] for b in bonds], dtype=np.int32).reshape(-1, 2),
        bond_state=np.array([b[3] for b in bonds], dtype=np.uint8),
        group_kind=np.array([kind for kind, _ in groups], dtype=np.uint8),
        group_ptr=np.concatenate(([0], np.cumsum(sizes))).astype(np.int64),
        group_member=np.array([m for _, members in groups for m in members], dtype=np.int32),
    )


def remap(columns, atom_map, *, hydrogen=(), substitutions=(), externalised=None, **kwargs):
    atom_map = np.array(atom_map, dtype=np.int64)
    removed = np.zeros(len(atom_map), dtype=bool)
    removed[list(hydrogen)] = True
    external = np.full(len(atom_map), -1, dtype=np.int64)
    for atom, row in (externalised or {}).items():
        external[atom] = row
    return remap_parity(
        columns,
        atom_map,
        removed_hydrogen=removed,
        substitutions=np.array(substitutions, dtype=np.int64).reshape(-1, 3),
        externalised=external,
        **kwargs,
    )


class TestPermutationParity:
    @pytest.mark.parametrize(
        ("order", "parity"),
        [([0, 1, 2, 3], 0), ([1, 0, 2, 3], 1), ([1, 2, 0, 3], 0), ([3, -1, 1, 2], 1), ([2, -1, 1, 0], 0)],
    )
    def test_inversion_count_modulo_two(self, order, parity):
        assert permutation_parity(np.array([order])).tolist() == [parity]


def parity(points, slots, quantum=QUANTUM, centre=0):
    return int(parity_from_coordinates(points, np.array([centre]), np.array([slots]), quantum)[0])


class TestParityFromCoordinates:
    def test_tetrahedron_and_mirror(self):
        mirror = TETRAHEDRON * np.array([-1, 1, 1])
        assert (parity(TETRAHEDRON, [1, 2, 3, 4]), parity(mirror, [1, 2, 3, 4])) == (1, 2)

    def test_neighbour_swap_flips(self):
        assert parity(TETRAHEDRON, [2, 1, 3, 4]) == 2

    @pytest.mark.parametrize("slot", [0, 1, 2, 3])
    def test_implicit_slot_reads_centre(self, slot):
        slots = [1, 2, 3, 4]
        explicit = parity(TETRAHEDRON, slots)
        slots[slot] = -1
        assert parity(TETRAHEDRON, slots) == explicit

    def test_lone_pair_centre_decides(self):
        points = np.array([[0, 0, 0.4], [1, 0, 0], [-0.5, 0.87, 0], [-0.5, -0.87, 0]])
        assert parity(points, [1, 2, 3, -1], np.full(4, 0.001)) in (1, 2)

    def test_planar_centre_is_undetermined(self):
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]], dtype=float)
        assert parity(points, [1, 2, 3, 4]) == 0

    def test_volume_within_bound_undetermined(self):
        assert parity(TETRAHEDRON * 0.01, [1, 2, 3, 4], np.full(5, 0.01)) == 0

    def test_centre_index_checked(self, invalid):  # -1 is no centre, never the last atom
        invalid(parity_from_coordinates, TETRAHEDRON, np.array([-1]), np.array([[1, 2, 3, 4]]), QUANTUM)

    def test_external_slot_is_undetermined(self):
        assert parity(TETRAHEDRON, [1, 2, 3, -2]) == 0

    def test_bound_is_sound(self):
        rng = np.random.default_rng(7)
        for _ in range(200):
            points = rng.normal(size=(5, 3)) * 0.3
            quantum = np.full(5, 10.0 ** rng.uniform(-3, -1))
            decided = parity(points, [1, 2, 3, 4], quantum)
            if not decided:
                continue
            for _ in range(20):
                shaken = points + rng.uniform(-0.5, 0.5, size=points.shape) * quantum[:, None]
                assert parity(shaken, [1, 2, 3, 4], np.zeros(5)) == decided


BOND = np.array([[0, 0, 0], [1.3, 0, 0], [-0.7, 1.2, 0], [2.0, 1.2, 0], [2.0, -1.2, 0], [2.6, 0, 0]], dtype=float)


_DEFAULT_QUANTUM = np.full(6, 0.001)


def configuration(points, references, quantum=_DEFAULT_QUANTUM):
    return int(bond_configuration(points, np.array([[0, 1]]), np.array([references]), quantum)[0])


class TestBondConfiguration:
    @pytest.mark.parametrize(("references", "value"), [([2, 3], 1), ([2, 4], 2), ([2, 5], 0)])
    def test_cis_trans_collinear(self, references, value):
        assert configuration(BOND, references) == value

    def test_bond_index_checked(self, invalid):  # -1 is no atom, never the last
        invalid(bond_configuration, BOND, np.array([[-1, 1]]), np.array([[2, 3]]), np.full(6, 0.001))

    def test_bound_is_sound(self):
        rng = np.random.default_rng(11)
        for _ in range(200):
            points = rng.normal(size=(6, 3)) * 0.5
            quantum = np.full(6, 10.0 ** rng.uniform(-3, -1))
            decided = configuration(points, [2, 3], quantum)
            if not decided:
                continue
            for _ in range(20):
                shaken = points + rng.uniform(-0.5, 0.5, size=points.shape) * quantum[:, None]
                assert configuration(shaken, [2, 3], np.zeros(6)) == decided


CENTRE = (0, 1, [1, 2, 3, 4], KR)
HEAVY = CODE_NAMES.index("parity_cleared_heavy_neighbour_removed")
TWO = CODE_NAMES.index("parity_cleared_two_virtual_slots")
NEW = CODE_NAMES.index("parity_cleared_new_neighbour_without_vacated_slot")


def row(columns):
    return (int(columns.parity[0]), columns.neighbors[0].tolist(), int(columns.centre_state[0]))


class TestRemapParity:
    def test_kept_slots_are_renumbered(self):
        out, clears = remap(stereo([CENTRE]), [1, 2, 3, 4, 5, 6])
        assert row(out) == (1, [2, 3, 4, 5], KR) and out.centre.tolist() == [1] and not len(clears.centre)

    def test_externalised_slot_keeps_parity(self):
        out, _ = remap(stereo([CENTRE]), [0, 1, 2, 3, -1, 4], externalised={4: 0})
        assert row(out) == (1, [1, 2, 3, -2], KR)

    @pytest.mark.parametrize("also_externalised", [False, True])
    def test_transport_is_derived(self, also_externalised):
        external = {4: 3} if also_externalised else None
        out, _ = remap(stereo([CENTRE]), [0, 1, 2, 3, -1, 4], substitutions=[(0, 4, -2)], externalised=external)
        assert row(out) == (1, [1, 2, 3, -2], KDER)

    def test_removed_hydrogen_becomes_implicit(self):
        out, _ = remap(stereo([CENTRE]), [0, 1, 2, 3, -1, 4], hydrogen=[4])
        assert row(out) == (1, [1, 2, 3, -1], KR)

    def test_second_implicit_slot_clears(self):
        columns = stereo([(0, 1, [1, 2, 3, -1], KR)])
        out, clears = remap(columns, [0, 1, 2, -1, 3, 4], hydrogen=[3])
        assert row(out) == (0, [-1, -1, -1, -1], UDER)
        assert clears.centre.tolist() == [0] and clears.reason.tolist() == [TWO]

    def test_substituted_neighbour_keeps_parity(self):
        out, _ = remap(stereo([CENTRE]), [0, 1, 2, 3, -1, 4], substitutions=[(0, 4, 4)])
        assert row(out) == (1, [1, 2, 3, 4], KDER)

    def test_partner_takes_departing_hydrogen(self):
        out, _ = remap(stereo([CENTRE]), [0, 1, 2, 3, -1, 4], hydrogen=[4], substitutions=[(0, 4, 4)])
        assert row(out) == (1, [1, 2, 3, 4], KDER)

    def test_removed_heavy_neighbour_clears(self):
        out, clears = remap(stereo([CENTRE]), [0, 1, 2, 3, -1, 4])
        assert row(out) == (0, [-1, -1, -1, -1], UDER)
        assert clears.reason.tolist() == [HEAVY]

    def test_unvacated_new_neighbour_clears(self):
        out, clears = remap(stereo([CENTRE]), [0, 1, 2, 3, 4, 5], substitutions=[(0, -2, 5)])
        assert row(out)[2] == UDER and clears.reason.tolist() == [NEW]

    def test_removed_centre_drops_row(self):
        columns = stereo([CENTRE, (5, 2, [1, 2, 3, -1], KR)], groups=[(1, (0, 1))])
        out, _ = remap(columns, [-1, 0, 1, 2, 3, 4])
        assert out.centre.tolist() == [4] and out.group_member.tolist() == [0] and out.group_ptr.tolist() == [0, 1]

    def test_emptied_group_is_dropped(self):
        columns = stereo([CENTRE, (5, 2, [1, 2, 3, -1], KR)], groups=[(1, (0,)), (2, (1,))])
        out, _ = remap(columns, [-1, 0, 1, 2, 3, 4])
        assert out.group_kind.tolist() == [2] and out.group_member.tolist() == [0]

    def test_undetermined_row_passes_through(self):
        out, clears = remap(stereo([(0, 0, [-1, -1, -1, -1], UR)]), [1, -1, -1, -1, -1, 0])
        assert row(out) == (0, [-1, -1, -1, -1], UR) and out.centre.tolist() == [1] and not len(clears.centre)

    def test_bond_rows_follow_edges(self):
        columns = stereo([], bonds=[(0, 2, [2, 3], KR), (1, 1, [4, 5], KR)])
        out, _ = remap(columns, [0, 1, 2, 3, 4, 5], edge_map=np.array([-1, 0]))
        assert out.bond_edge.tolist() == [0] and out.bond_reference.tolist() == [[4, 5]]

    def test_removed_reference_clears_bond(self):  # the cleared row is named with its reason
        columns = stereo([], bonds=[(0, 2, [2, 3], KR), (1, 1, [4, 5], KR)])
        out, clears = remap(columns, [0, 1, 2, -1, 3, 4], edge_map=np.array([0, 1]))
        assert out.bond_state.tolist() == [UDER, KR]
        assert (clears.bond.tolist(), clears.bond_reason.tolist()) == ([0], [HEAVY])
