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

"""The retention rules of the projections and ``select`` (ofoldx.data.projection)."""

from __future__ import annotations

import numpy as np
import pytest

from ofoldx.data.molecular import MolecularSystem
from ofoldx.data.projection import fold_hydrogens, project_arrays, project_atoms, project_instances, select
from ofoldx.data.structure import Structure
from ofoldx.errors import CODE_NAMES, UnsupportedError
from ofoldx.provenance import Basis, Resolution, known, state

KR, KDER = state(Resolution.KNOWN, Basis.REFERENCE), state(Resolution.KNOWN, Basis.DERIVED)
UN, UDER = state(Resolution.UNKNOWN, Basis.NONE), state(Resolution.UNKNOWN, Basis.DERIVED)
HEAVY = CODE_NAMES.index("parity_cleared_heavy_neighbour_removed")
HA, CA = 5, 1


@pytest.fixture
def protium(system_arrays, structure_arrays) -> Structure:
    """The fixture with HA's isotope not stated (an ordinary hydrogen)."""
    arrays = system_arrays()
    arrays.columns["isotope"][HA] = 0
    return Structure.from_arrays(MolecularSystem.from_arrays(arrays), structure_arrays())


def without(value, *atoms):
    keep = np.ones(len(value.system.atom_symbol if isinstance(value, Structure) else value.atom_symbol), dtype=bool)
    keep[list(atoms)] = False
    return project_atoms(value, keep)


def claims(system, component):
    names = ("topology", "heavy_atoms", "hydrogens")
    return tuple(bool(getattr(system, f"component_{name}_complete")[component]) for name in names)


def partners(system):
    texts = system.text("instance", system.external_partner_instance)
    atoms = system.text("atom", system.external_partner_atom)
    return list(zip(system.external_atom.tolist(), texts, system.external_partner_component.tolist(), atoms))


class TestProjectAtoms:
    def test_removed_heavy_neighbour_clears(self, system):
        value, projection = without(system, 4)
        assert value.stereocentre_state[0] == UDER and value.stereocentre_parity[0] == 0
        assert projection.parity_clears.centre.tolist() == [0] and projection.parity_clears.reason.tolist() == [HEAVY]
        assert claims(value, 0) == (False, False, True) and claims(value, 1) == claims(system, 1)

    def test_removed_hydrogen_becomes_implicit(self, system):
        value, _ = without(system, 5)
        assert value.stereocentre_neighbors[0].tolist() == [0, 2, 4, -1] and value.stereocentre_state[0] == KR

    @pytest.mark.parametrize(
        ("value", "count", "count_state"), [("protium", 1, KDER), ("structure", 0, UDER)], ids=["protium", "deuterium"]
    )
    def test_removed_hydrogen_is_counted(self, value, count, count_state, request):
        projected, projection = without(request.getfixturevalue(value), HA)
        system, ca = projected.system, projection.atom[CA]
        assert system.implicit_hydrogens[ca] == count and system.implicit_hydrogens_state[ca] == count_state

    def test_cut_becomes_external(self, system):
        value, projection = without(system, 6)
        assert (2, "A", 1, "N") in partners(value)
        assert len(value.edge_index[0]) == len(system.edge_index[0]) - 2
        assert claims(value, 0) == claims(system, 0) and claims(value, 1) == (False, False, False)
        assert projection.component.tolist() == [0, 1, 2, 3]

    def test_structure_columns_are_gathered(self, structure):
        value, projection = without(structure, 13)
        kept = projection.atom >= 0
        assert np.array_equal(value.coordinates, structure.coordinates[kept])

    def test_indices_are_no_mask(self, system, invalid):  # np.arange is no keep mask: it never drops atom 0
        invalid(project_atoms, system, np.arange(len(system.atom_symbol)), code="projection_invalid")

    def test_emptied_instance_is_removed(self, system):
        value, projection = without(system, 17)
        assert projection.instance.tolist() == [0, 1, -1]


class TestProjectInstances:
    def test_cut_edges_become_externals(self, cross_linked):
        keep = np.array([False, True, True])
        value, projection = project_instances(cross_linked, keep)
        system = value.system
        assert (projection.atom[11], "A", 0, "N") in partners(system)
        assert claims(system, 0) == claims(cross_linked.system, 2)
        assert system.instance_connections_complete.all()

    @pytest.mark.parametrize("keep", [np.array([0, 2]), np.array([False, True])], ids=["indices", "short"])
    def test_keep_must_be_mask(self, system, invalid, keep):  # never read as another selection
        invalid(project_instances, system, keep, code="projection_invalid")

    def test_cut_slot_becomes_external(self, cross_linked):
        value, _ = project_instances(cross_linked, np.array([False, True, True]))
        system = value.system
        slots = system.stereocentre_neighbors[0]
        assert slots[3] <= -2 and system.external_atom[-slots[3] - 2] == system.stereocentre_atom[0]
        assert system.stereocentre_parity[0] == 2 and system.stereocentre_state[0] == KR

    def test_symmetry_external_is_kept(self, structure):
        value, projection = project_instances(structure, np.array([False, True, True]))
        assert projection.external.tolist() == [0, -1]
        assert partners(value.system)[0][1:] == ("A", 1, "OXT")

    def test_atomless_components_kept(self, system_arrays):  # a sequence-only read: components, no atom
        arrays = system_arrays()
        every = np.arange(len(arrays.columns["instance_symbol"]))
        atomless, _, _ = project_arrays(arrays, None, every, np.zeros(len(arrays.columns["atom_symbol"]), bool))
        value, projection = project_instances(MolecularSystem.from_arrays(atomless), np.ones(len(every), bool))
        assert len(value.atom_symbol) == 0 and (projection.component >= 0).all()


def stereobond_on_ca_c(columns):
    """A KNOWN trans configuration on ALA CA=C referenced by HA and O (edge 1 is CA–C)."""
    columns["stereobond_edge"][0], columns["stereobond_reference"][0] = 1, [HA, 3]


class TestFoldHydrogens:
    def test_hydrogen_becomes_a_count(self, protium):
        value, projection = fold_hydrogens(protium)
        system = value.system
        ca = projection.atom[CA]
        assert len(system.atom_symbol) == len(protium.system.atom_symbol) - 1
        assert system.implicit_hydrogens[ca] == 1 and system.implicit_hydrogens_state[ca] == KDER
        assert system.stereocentre_neighbors[0].tolist() == [0, 2, 4, -1]
        assert claims(system, 0) == (True, True, True)

    def test_stated_isotope_is_inexact(self, structure):
        value, projection = fold_hydrogens(structure)
        system = value.system
        assert system.implicit_hydrogens_state[projection.atom[CA]] == UDER
        assert claims(system, 0) == (False, True, False)

    def test_reference_hydrogen_is_repointed(self, system_arrays):
        arrays = system_arrays()
        stereobond_on_ca_c(arrays.columns)
        value, projection = fold_hydrogens(MolecularSystem.from_arrays(arrays))
        assert value.stereobond_reference.tolist() == [[projection.atom[0], projection.atom[3]]]
        assert value.stereobond_value.tolist() == [1] and value.stereobond_state.tolist() == [KR]

    def test_lone_reference_hydrogen_refused(self, system_arrays):
        arrays = system_arrays()
        stereobond_on_ca_c(arrays.columns)
        arrays.columns["atomic_number"][[0, 4]] = 1  # CA's other neighbours, N and CB, become hydrogens
        with pytest.raises(UnsupportedError) as caught:
            fold_hydrogens(MolecularSystem.from_arrays(arrays))
        assert caught.value.code == "hydrogen_view_loses_stereo"

    def test_second_implicit_slot_is_refused(self, system_arrays):
        arrays = system_arrays()
        arrays.columns["stereocentre_neighbors"][0] = [0, 2, 5, -1]
        with pytest.raises(UnsupportedError) as caught:
            fold_hydrogens(MolecularSystem.from_arrays(arrays))
        assert caught.value.code == "hydrogen_view_loses_stereo"

    def test_unknown_count_stays_unknown(self, system_arrays):
        arrays = system_arrays()
        arrays.columns["implicit_hydrogens_state"][1] = state(Resolution.UNKNOWN, Basis.NONE)
        value, projection = fold_hydrogens(MolecularSystem.from_arrays(arrays))
        assert value.implicit_hydrogens_state[projection.atom[1]] == UDER
        assert not known(value.implicit_hydrogens_state[projection.atom[1]])


class TestSelect:
    @pytest.mark.parametrize(
        ("chains", "namespace", "instances"),
        [
            (["A"], "author", ["A", "L"]),
            (["B"], "author", ["W"]),
            (["B"], "label", ["L"]),
            (["A", "C"], "label", ["A", "W"]),
        ],
    )
    def test_chains_select_instances(self, chains, namespace, instances, structure):
        value = select(structure, chains=chains, namespace=namespace)
        assert value.system.text("instance", value.system.instance_symbol).tolist() == instances

    @pytest.mark.parametrize(("chains", "namespace"), [(["Z"], "author"), (["W"], "label"), (["C"], "author")])
    def test_absent_chain_is_invalid(self, chains, namespace, structure, invalid):
        invalid(select, structure, chains=chains, namespace=namespace, code="projection_invalid")

    def test_unknown_author_chain_matches_nothing(self, system, structure_arrays, invalid):
        arrays = structure_arrays()
        arrays.columns["instance_author_chain"][2], arrays.columns["instance_author_chain_state"][2] = -1, UN
        invalid(select, Structure.from_arrays(system, arrays), chains=["B"], code="projection_invalid")

    def test_water_dropped_by_entity(self, structure):
        value = structure.select(water=False)
        assert value.system.text("instance", value.system.instance_symbol).tolist() == ["A", "L"]

    def test_hydrogens_are_folded(self, structure):
        value = structure.select(hydrogens=False)
        assert not (value.system.atomic_number == 1).any()
