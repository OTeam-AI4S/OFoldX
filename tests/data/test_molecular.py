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

"""Behaviour of ofoldx.data.molecular: construction, lookups and the endpoint projection."""

from __future__ import annotations

import numpy as np
import pytest

from ofoldx.data.assembly import batch
from ofoldx.data.molecular import (
    ELEMENTS,
    GROUP_BY_ATOMIC_NUMBER,
    PERIOD_BY_ATOMIC_NUMBER,
    AtomSite,
    ComponentRef,
    MolecularSystem,
    ReferenceSource,
    endpoint_resolution,
)
from ofoldx.data.projection import project_atoms
from ofoldx.provenance import Resolution


def test_build_round_trips(system):
    rebuilt = MolecularSystem.from_arrays(system.build())
    assert rebuilt.symbols() == system.symbols()
    assert all(np.array_equal(rebuilt.arrays()[k], v) for k, v in system.arrays().items())


class TestLookups:
    def test_component_refs_invert_index(self, system):
        components = np.arange(len(system.component_symbol))
        refs = system.component_refs(components)
        assert refs[1] == ComponentRef("A", 1)
        assert system.component_index(refs).tolist() == components.tolist()

    @pytest.mark.parametrize(
        ("site", "atom"),
        [
            (AtomSite("A", 0, "CA"), 1),
            (AtomSite("A", 1, "OXT"), 10),
            (AtomSite("L", 0, "O1"), 16),
            (AtomSite("Q", 0, "CA"), -1),
            (AtomSite("A", 2, "CA"), -1),
            (AtomSite("A", 0, "OXT"), -1),
            (AtomSite("B", 0, "ZN"), -1),
        ],
    )
    def test_atom_index(self, site, atom, system):
        assert system.atom_index([site]).tolist() == [atom]

    def test_unknown_instance_is_missing(self, system):
        assert system.instance_index(["L", "Q"]).tolist() == [1, -1]

    def test_batched_symbol_is_ambiguous(self, structure, invalid):
        invalid(batch([structure, structure]).system.instance_index, ["A"])

    def test_lookup_in_empty_system(self, system):
        empty, _ = project_atoms(system, np.zeros(len(system.atom_symbol), dtype=bool))
        assert empty.atom_index([AtomSite("A", 0, "CA")]).tolist() == [-1]

    def test_text_renders_codes(self, system):
        assert system.text("instance", np.array([1, -1])).tolist() == ["L", ""]

    @pytest.mark.parametrize(
        ("method", "args"),
        [("component_refs", (np.array([-1]),)), ("text", ("atom", np.array([-2])))],
        ids=["component_refs", "text"],
    )
    def test_negative_index_refused(self, system, invalid, method, args):  # never the last row
        invalid(getattr(system, method), *args)


def test_reference_coordinates_gather_pool(system):
    coordinates, source = system.reference_coordinates()
    assert np.allclose(coordinates[:17], system.reference_pool_coordinates)
    assert (coordinates[17] == 0).all() and source[17] == ReferenceSource.NONE
    assert source[0] == ReferenceSource.IDEAL and source[11] == ReferenceSource.MODEL


def test_endpoint_resolution():
    expected = [Resolution.INAPPLICABLE, *[Resolution.KNOWN] * 3, Resolution.INAPPLICABLE]
    expected += [Resolution.UNKNOWN, Resolution.CONFLICT]
    assert endpoint_resolution(np.arange(7, dtype=np.uint8)).tolist() == expected


def test_endpoint_state_outside_range(invalid):
    invalid(endpoint_resolution, np.array([-1]))


class TestElements:
    def test_index_is_atomic_number(self):
        assert (len(ELEMENTS), ELEMENTS[0], ELEMENTS[6], ELEMENTS[118]) == (119, "X", "C", "Og")

    @pytest.mark.parametrize(
        ("element", "group", "period"),
        [("X", 0, 0), ("He", 18, 1), ("C", 14, 2), ("Fe", 8, 4), ("Ce", 3, 6), ("Og", 18, 7)],
    )
    def test_group_and_period(self, element, group, period):
        number = ELEMENTS.index(element)
        assert (GROUP_BY_ATOMIC_NUMBER[number], PERIOD_BY_ATOMIC_NUMBER[number]) == (group, period)
