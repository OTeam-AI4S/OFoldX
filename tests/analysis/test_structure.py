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

"""Behavior of ofoldx.analysis.structure."""

from math import pi

import numpy as np
import pytest
from biotite.structure.info import vdw_radius_single

from ofoldx.analysis.structure import (
    assign_secondary_structure,
    structure_coordinates,
    structure_design_lddt,
    structure_rmsd_metrics,
    target_delta_sasa,
)
from ofoldx.chemistry.construction import Molecule, PolymerSequence, Recipe, build_system
from ofoldx.chemistry.smiles import read_smiles
from ofoldx.data.molecular import AtomSite, ComponentRef
from ofoldx.data.structure import CoordinateEvidence, Structure, replace_coordinates


@pytest.fixture
def place(library):
    """``place(recipe, sites, points, components)``: the recipe's system with only ``sites`` placed at ``points``."""

    def make(recipe: Recipe, sites: list[AtomSite], points: np.ndarray, components=None) -> Structure:
        system, _, _ = build_system(recipe, library, components or {})
        structure, _ = replace_coordinates(
            Structure.unobserved(system),
            system.atom_index(sites),
            np.asarray(points, dtype=np.float64).reshape(-1, 3),
            scope=np.ones(len(system.atom_symbol), dtype=bool),
            evidence=CoordinateEvidence.OBSERVED,
        )
        return structure

    return make


@pytest.fixture
def waters(place):
    """Two SMILES water ligands ``LIG`` (Biotite drops residues named ``HOH`` as solvent)."""
    recipe = Recipe({"W": Molecule(("LIG",))}, {"A": "W", "B": "W"})
    sites, points = [AtomSite("A", 0, "O1"), AtomSite("B", 0, "O1")], [[0.0, 0.0, 0.0], [6.0, 0.0, 0.0]]
    return place(recipe, sites, points, {"LIG": read_smiles("O", component_id="LIG").component})


@pytest.fixture
def backbones(place):
    """Protein chains of ALA (PRO where named) with only N, CA, C, O placed from ``[residues, 4, 3]`` arrays."""

    def make(*chains, prolines=(), linked=None) -> Structure:
        entities, instances, sites = {}, {}, []
        for k, points in enumerate(chains):
            ids = tuple("PRO" if (k, p) in prolines else "ALA" for p in range(len(points)))
            entities[str(k)] = PolymerSequence("polypeptide(L)", ids, None if linked is None else linked[k])
            instances[f"chain{k}"] = str(k)
            sites += [AtomSite(f"chain{k}", p, name) for p in range(len(points)) for name in ("N", "CA", "C", "O")]
        return place(Recipe(entities, instances), sites, np.concatenate(chains))

    return make


class TestSiteMatching:
    def test_identical_metrics(self, peptide):
        structure = peptide("ALA", "GLY", "ALA")
        design, target = {ComponentRef("A", 1)}, {ComponentRef("A", 0), ComponentRef("A", 2)}

        metrics = structure_rmsd_metrics(structure, structure, designed_positions=design, target=target)

        assert all(value == pytest.approx(0.0, abs=1e-6) for value in metrics.values())
        assert structure_design_lddt(structure, structure, designed_positions=design) == 1.0

    def test_backbone_sites(self, peptide):
        structure = peptide("ALA", "GLY", "ALA")

        reference, predicted, sites = structure_coordinates(structure, structure, backbone_only=True)

        assert {site.atom for site in sites} == {"N", "CA", "C", "O"}
        assert len(sites) == 12
        np.testing.assert_array_equal(reference, predicted)

    def test_unplaced_sites_excluded(self, backbones, secondary_backbones):
        structure = backbones(secondary_backbones["helix"][:2])

        _, _, sites = structure_coordinates(structure, structure)

        assert {site.atom for site in sites} == {"N", "CA", "C", "O"}


class TestSasa:
    def test_isolated_water(self, waters):
        result = target_delta_sasa(waters, target={ComponentRef("A", 0)}, designed_positions={ComponentRef("B", 0)})

        assert result.delta == pytest.approx(0.0)
        assert result.target_unbound == pytest.approx(4 * pi * (vdw_radius_single("O") + 1.4) ** 2, rel=1e-6)

    def test_requires_heavy_coordinates(self, library):
        system, _, _ = build_system(Recipe({"W": Molecule(("HOH",))}, {"A": "W"}), library)
        unplaced = Structure.unobserved(system)

        with pytest.raises(ValueError, match="heavy"):
            target_delta_sasa(unplaced, target={ComponentRef("A", 0)}, designed_positions=())


class TestSecondaryStructure:
    def test_cross_chain_beta(self, backbones, secondary_backbones):
        backbone = secondary_backbones["beta"]

        assert assign_secondary_structure(backbones(backbone[:2], backbone[2:])) == {
            ComponentRef("chain0", 0): "L",
            ComponentRef("chain0", 1): "E",
            ComponentRef("chain1", 0): "L",
            ComponentRef("chain1", 1): "E",
        }
        proline = backbones(backbone[:2], backbone[2:], prolines=((0, 1),))
        assert set(assign_secondary_structure(proline).values()) == {"L"}

    def test_helix_context(self, backbones, secondary_backbones):
        backbone = secondary_backbones["helix"]
        selected = (ComponentRef("chain0", 1), ComponentRef("chain0", 2))

        assert assign_secondary_structure(backbones(backbone), selected) == dict.fromkeys(selected, "H")
        broken = backbones(backbone, linked=((True, True, True, False, True, True),))
        assert set(assign_secondary_structure(broken).values()) == {"L"}

    def test_requires_coordinates(self, place, secondary_backbones):
        recipe = Recipe({"0": PolymerSequence("polypeptide(L)", ("ALA", "ALA"))}, {"chain0": "0"})
        backbone = secondary_backbones["helix"][:2]
        sites = [AtomSite("chain0", p, name) for p in range(2) for name in ("N", "CA", "C", "O")]
        structure = place(recipe, sites[:-1], backbone.reshape(-1, 3)[:-1])

        with pytest.raises(ValueError, match="secondary structure requires"):
            assign_secondary_structure(structure, components={ComponentRef("chain0", 1)})
        assert set(assign_secondary_structure(structure, components={ComponentRef("chain0", 0)})) == {
            ComponentRef("chain0", 0)
        }


def _without_coordinates(structure, *sites):
    indices = structure.system.atom_index(sites)
    assert (indices >= 0).all()
    arrays = structure.build()
    arrays.columns["coordinate_evidence"][indices] = CoordinateEvidence.NONE
    arrays.columns["coordinates"][indices] = 0
    return Structure.from_arrays(structure.system, arrays)


def _upstream_atoms(structure):
    import biotite.structure as struc

    from ofoldx.data.molecular import ELEMENTS

    system = structure.system
    owner = np.repeat(np.arange(len(system.component_symbol)), np.diff(system.component_atom_ptr))
    instance = np.repeat(np.arange(len(system.instance_symbol)), np.diff(system.instance_component_ptr))
    indices = np.flatnonzero(structure.coordinate_mask & (system.atomic_number != 1))
    arr = struc.AtomArray(len(indices))
    arr.coord = structure.coordinates[indices]
    arr.chain_id = instance[owner[indices]].astype(str)
    arr.res_id = owner[indices] - system.instance_component_ptr[instance[owner[indices]]] + 1
    arr.ins_code = np.full(len(indices), "")
    arr.atom_name = system.text("atom", system.atom_symbol[indices])
    arr.res_name = system.text("component", system.component_symbol[owner[indices]])
    arr.element = np.asarray(ELEMENTS)[system.atomic_number[indices]]
    arr.hetero = np.zeros(len(indices), dtype=bool)
    return arr, indices


def _upstream_sasa(array):
    import biotite.structure as struc
    from biotite.structure.info import vdw_radius_protor, vdw_radius_single

    radii = []
    for residue, name, element in zip(array.res_name, array.atom_name, array.element, strict=True):
        try:
            value = vdw_radius_protor(str(residue), str(name))
        except KeyError:
            value = None
        if value is None:
            value = vdw_radius_single(str(element))
        radii.append(1.8 if value is None else value)
    return struc.sasa(array, probe_radius=1.4, point_number=960, vdw_radii=np.asarray(radii))


def test_present_atom_sasa_uses_each_structure_mask(peptide):
    from ofoldx.analysis.structure import present_atom_target_delta_sasa

    full = peptide("ALA", "SER", copies=2)
    original = _without_coordinates(full, AtomSite("A", 1, "OG"), AtomSite("A", 1, "CB"))
    refolded = _without_coordinates(full, AtomSite("B", 1, "OXT"))
    target = {ComponentRef("B", 0), ComponentRef("B", 1)}
    design = {ComponentRef("A", 0), ComponentRef("A", 1)}
    for structure in (original, refolded):
        got = present_atom_target_delta_sasa(structure, target=target, designed_positions=design)
        array, indices = _upstream_atoms(structure)
        target_mask = array.chain_id == "1"
        bound = float(np.sum(_upstream_sasa(array)[target_mask]))
        unbound = float(np.sum(_upstream_sasa(array[target_mask])))
        assert got.sasa.target_bound == pytest.approx(bound, abs=1e-8)
        assert got.sasa.target_unbound == pytest.approx(unbound, abs=1e-8)
        assert got.sasa.delta == pytest.approx(unbound - bound, abs=1e-8)
        assert got.coverage.atom_indices == tuple(indices.tolist())
        assert set(got.coverage.missing_atom_indices) == set(
            np.flatnonzero(~structure.coordinate_mask & (structure.system.atomic_number != 1))
        )
    assert (
        present_atom_target_delta_sasa(original, target=target, designed_positions=design).coverage
        != present_atom_target_delta_sasa(refolded, target=target, designed_positions=design).coverage
    )
    with pytest.raises(ValueError, match="complete coordinates"):
        target_delta_sasa(original, target=target, designed_positions=design)


@pytest.mark.parametrize("design_state", ["empty", "unobserved"])
def test_present_contacts_and_sasa_require_observed_design(peptide, design_state):
    from ofoldx.analysis.structure import (
        PresentAtomUnavailableError,
        present_atom_noncovalent_contacts,
        present_atom_target_delta_sasa,
    )

    source = peptide("ALA", copies=2)
    design = {ComponentRef("A", 0)} if design_state == "unobserved" else set()
    if design:
        arrays = source.build()
        start, stop = source.system.component_atom_ptr[:2]
        arrays.columns["coordinate_evidence"][start:stop] = CoordinateEvidence.NONE
        arrays.columns["coordinates"][start:stop] = 0
        source = Structure.from_arrays(source.system, arrays)
    with pytest.raises(PresentAtomUnavailableError, match="present design heavy atom"):
        present_atom_noncovalent_contacts(source, designed_positions=design, connected_designed_positions=design)
    with pytest.raises(PresentAtomUnavailableError, match="present design heavy atom"):
        present_atom_target_delta_sasa(source, target={ComponentRef("B", 0)}, designed_positions=design)


@pytest.mark.parametrize("target_state", ["connected", "unobserved"])
def test_present_contacts_require_observed_target(peptide, target_state):
    from ofoldx.analysis.structure import PresentAtomUnavailableError, present_atom_noncovalent_contacts

    source = peptide("ALA", copies=2)
    design = {ComponentRef("A", 0)}
    connected = design | {ComponentRef("B", 0)} if target_state == "connected" else design
    if target_state == "unobserved":
        arrays = source.build()
        start, stop = source.system.component_atom_ptr[1:3]
        arrays.columns["coordinate_evidence"][start:stop] = CoordinateEvidence.NONE
        arrays.columns["coordinates"][start:stop] = 0
        source = Structure.from_arrays(source.system, arrays)
    with pytest.raises(PresentAtomUnavailableError, match="present target heavy atom"):
        present_atom_noncovalent_contacts(source, designed_positions=design, connected_designed_positions=connected)


def test_present_contacts_and_sasa_reject_invalid_scopes(peptide):
    from ofoldx.analysis.structure import (
        AnalysisUnavailableError,
        present_atom_noncovalent_contacts,
        present_atom_target_delta_sasa,
    )

    structure = peptide("ALA", copies=2)
    design = {ComponentRef("A", 0)}
    with pytest.raises(ValueError, match="contained in connected_designed_positions") as contacts_error:
        present_atom_noncovalent_contacts(structure, designed_positions=design, connected_designed_positions=())
    assert not isinstance(contacts_error.value, AnalysisUnavailableError)
    with pytest.raises(ValueError, match="must be disjoint") as sasa_error:
        present_atom_target_delta_sasa(structure, target=design, designed_positions=design)
    assert not isinstance(sasa_error.value, AnalysisUnavailableError)


@pytest.mark.parametrize("residue,atom", [("ASP", "OD2"), ("GLU", "OE2"), ("LYS", "NZ"), ("ARG", "NH2")])
def test_present_contacts_reject_wrong_element_for_protein_charge(peptide, residue, atom):
    from ofoldx.analysis.structure import PresentAtomUnavailableError, present_atom_noncovalent_contacts
    from ofoldx.data.molecular import MolecularSystem

    source = peptide("LYS", copies=2)
    index = source.system.atom_index([AtomSite("A", 0, "NZ")])[0]
    arrays = source.system.build()
    arrays.symbols["component"] = tuple(residue if name == "LYS" else name for name in arrays.symbols["component"])
    arrays.symbols["atom"] = tuple(atom if name == "NZ" else name for name in arrays.symbols["atom"])
    arrays.columns["atomic_number"][index] = 6
    altered = Structure.from_arrays(MolecularSystem.from_arrays(arrays), source.build())
    design = {ComponentRef("A", 0)}
    with pytest.raises(PresentAtomUnavailableError, match="protein charge element does not match"):
        present_atom_noncovalent_contacts(altered, designed_positions=design, connected_designed_positions=design)


def test_present_atom_contacts_use_upstream_charge_and_hydride(peptide):
    import biotite.structure as struc
    import hydride

    from ofoldx.analysis.structure import count_noncovalent_contacts, present_atom_noncovalent_contacts

    source = peptide("LYS", "SER", copies=2)
    partial = _without_coordinates(source, AtomSite("A", 0, "NZ"), AtomSite("B", 1, "OXT"))
    array, indices = _upstream_atoms(partial)
    design = array.chain_id == "0"
    array.set_annotation("is_design", design)
    array.set_annotation("is_chain_design", design)
    charge = np.zeros(len(array), dtype=int)
    charge[(array.res_name == "LYS") & (array.atom_name == "NZ")] = 1
    charge[(array.res_name == "ASP") & (array.atom_name == "OD2")] = -1
    array.set_annotation("charge", charge)
    array.bonds = struc.connect_via_residue_names(array)
    hydrated, _ = hydride.add_hydrogen(array)
    hbonds = struc.hbond(hydrated)
    d, a = hbonds[:, 0], hbonds[:, 2]
    is_design, chain_design = hydrated.is_design, hydrated.is_chain_design
    expected_hbonds = int(np.sum(is_design[d] & ~chain_design[a]) + np.sum(~chain_design[d] & is_design[a]))
    positive, negative = np.flatnonzero(hydrated.charge > 0), np.flatnonzero(hydrated.charge < 0)
    distances = np.linalg.norm(hydrated.coord[positive, None] - hydrated.coord[None, negative], axis=-1)
    left, right = np.where((distances > 0.5) & (distances < 5.5))
    expected_pairs = int(
        np.sum(is_design[positive[left]] & ~chain_design[negative[right]])
        + np.sum(~chain_design[positive[left]] & is_design[negative[right]])
    )
    refs = {ComponentRef("A", 0), ComponentRef("A", 1)}
    got = present_atom_noncovalent_contacts(partial, designed_positions=refs, connected_designed_positions=refs)
    assert got.contacts.num_hydrogen_bonds == expected_hbonds
    assert got.contacts.num_charged_pairs == expected_pairs
    assert got.coverage.atom_indices == tuple(indices.tolist())
    assert got.charge_model == "protein_table_and_reference_definitions"
    assert all(not partial.coordinate_mask[i] for i in got.coverage.missing_atom_indices)
    with pytest.raises(ValueError):
        count_noncovalent_contacts(partial, designed_positions=refs, connected_designed_positions=refs)


def test_present_contacts_with_distant_chains(peptide):
    from ofoldx.analysis.structure import (
        count_noncovalent_contacts,
        present_atom_noncovalent_contacts,
        present_atom_target_delta_sasa,
    )

    source = peptide(*(["LYS"] * 8), copies=2)
    arrays = source.build()
    start = source.system.component_atom_ptr[8]
    arrays.columns["coordinates"][start:] += 10000
    separated = Structure.from_arrays(source.system, arrays)
    design = {ComponentRef("A", i) for i in range(8)}
    result = present_atom_noncovalent_contacts(
        separated, designed_positions=design, connected_designed_positions=design
    )
    assert result.contacts.num_hydrogen_bonds == 0
    assert result.contacts.num_charged_pairs == 0
    strict = count_noncovalent_contacts(separated, designed_positions=design, connected_designed_positions=design)
    assert strict.num_hydrogen_bonds == 0
    assert strict.num_charged_pairs == 0
    target = {ComponentRef("B", i) for i in range(8)}
    assert present_atom_target_delta_sasa(separated, target=target, designed_positions=design).sasa.delta == 0
    assert target_delta_sasa(separated, target=target, designed_positions=design).delta == 0


@pytest.mark.parametrize("separation", [0.2, 5.8399, 5.8401])
def test_sasa_spatial_groups_match_biotite_at_boundaries(waters, separation):
    import biotite.structure as struc

    from ofoldx.analysis.structure import present_atom_target_delta_sasa

    arrays = waters.build()
    oxygen = waters.system.atom_index([AtomSite("A", 0, "O1"), AtomSite("B", 0, "O1")])
    arrays.columns["coordinates"][oxygen] = [[15.9, 0, 0], [15.9 + separation, 0, 0]]
    structure = Structure.from_arrays(waters.system, arrays)
    array, _ = _upstream_atoms(structure)
    target, design = {ComponentRef("A", 0)}, {ComponentRef("B", 0)}
    present = present_atom_target_delta_sasa(structure, target=target, designed_positions=design).sasa
    assert present.target_bound == pytest.approx(float(_upstream_sasa(array)[0]), abs=1e-6)
    assert present.target_unbound == pytest.approx(float(_upstream_sasa(array[:1])[0]), abs=1e-6)
    strict = target_delta_sasa(structure, target=target, designed_positions=design)
    radii = np.full(2, vdw_radius_single("O"))
    bound = struc.sasa(array, probe_radius=1.4, point_number=960, vdw_radii=radii)
    unbound = struc.sasa(array[:1], probe_radius=1.4, point_number=960, vdw_radii=radii[:1])
    assert strict.target_bound == pytest.approx(float(bound[0]), abs=1e-6)
    assert strict.target_unbound == pytest.approx(float(unbound[0]), abs=1e-6)


def test_present_contacts_with_distant_generated_hydrogen(peptide, monkeypatch):
    import biotite.structure as struc
    import hydride

    from ofoldx.analysis.structure import present_atom_noncovalent_contacts

    source = peptide("LYS", "SER", copies=2)
    add_hydrogen = hydride.add_hydrogen
    expected = []

    def distant_hydrogen(array):
        hydrated, original_mask = add_hydrogen(array)
        moved = int(np.flatnonzero(~original_mask)[0])
        remaining = hydrated[np.arange(len(hydrated)) != moved]
        bonds = struc.hbond(remaining)
        donor, acceptor = bonds[:, 0], bonds[:, 2]
        expected.append(
            int(
                np.sum(remaining.is_design[donor] & ~remaining.is_chain_design[acceptor])
                + np.sum(~remaining.is_chain_design[donor] & remaining.is_design[acceptor])
            )
        )
        hydrated.coord[moved] += 10000
        return hydrated, original_mask

    monkeypatch.setattr(hydride, "add_hydrogen", distant_hydrogen)
    design = {ComponentRef("A", 0), ComponentRef("A", 1)}
    result = present_atom_noncovalent_contacts(source, designed_positions=design, connected_designed_positions=design)
    assert result.contacts.num_hydrogen_bonds == expected[0]


def test_present_contacts_with_shared_generated_hydrogen(peptide, monkeypatch):
    import biotite.structure as struc
    import hydride

    from ofoldx.analysis.structure import present_atom_noncovalent_contacts

    source = peptide("LYS", "SER", copies=2)
    add_hydrogen = hydride.add_hydrogen
    expected = []

    def shared_hydrogen(array):
        hydrated, original_mask = add_hydrogen(array)
        hydrogen = int(np.flatnonzero(~original_mask)[0])
        donor = int(np.flatnonzero(~hydrated.is_chain_design & (hydrated.element == "N"))[0])
        acceptor = int(np.flatnonzero(hydrated.is_design & (hydrated.element == "O"))[0])
        hydrated.bonds.add_bond(donor, hydrogen, struc.BondType.SINGLE)
        hydrated.coord[donor], hydrated.coord[hydrogen], hydrated.coord[acceptor] = (
            (0, 0, 0),
            (1, 0, 0),
            (2.5, 0, 0),
        )
        bonds = struc.hbond(hydrated)
        assert any(np.array_equal(row, (donor, hydrogen, acceptor)) for row in bonds)
        d, a = bonds[:, 0], bonds[:, 2]
        expected.append(
            int(
                np.sum(hydrated.is_design[d] & ~hydrated.is_chain_design[a])
                + np.sum(~hydrated.is_chain_design[d] & hydrated.is_design[a])
            )
        )
        return hydrated, original_mask

    monkeypatch.setattr(hydride, "add_hydrogen", shared_hydrogen)
    design = {ComponentRef("A", 0), ComponentRef("A", 1)}
    result = present_atom_noncovalent_contacts(source, designed_positions=design, connected_designed_positions=design)
    assert result.contacts.num_hydrogen_bonds == expected[0]


def test_present_hydrophobic_patch_matches_upstream_on_missing_oxt(peptide):
    from scipy.sparse.csgraph import connected_components

    from ofoldx.analysis.structure import largest_hydrophobic_patch_area, present_atom_largest_hydrophobic_patch_area

    structure = _without_coordinates(peptide("ALA", "ALA"), AtomSite("A", 1, "OXT"))
    array, _ = _upstream_atoms(structure)
    sasa = _upstream_sasa(array)
    mask = (
        np.isin(array.res_name, ("ALA", "VAL", "LEU", "ILE", "MET", "PHE", "PRO", "TRP"))
        & np.char.startswith(array.atom_name, "C")
        & (sasa > 0)
    )
    distances = np.linalg.norm(array.coord[mask, None] - array.coord[None, mask], axis=-1)
    _, labels = connected_components(distances <= 6.0, directed=False)
    expected = max(float(sasa[mask][labels == label].sum()) for label in np.unique(labels))
    got = present_atom_largest_hydrophobic_patch_area(structure)
    assert got.area == pytest.approx(expected, abs=1e-8)
    assert got.coverage.missing_atom_indices == tuple(structure.system.atom_index([AtomSite("A", 1, "OXT")]))
    with pytest.raises(ValueError, match="complete coordinates"):
        largest_hydrophobic_patch_area(structure)


def test_present_nonprotein_charge_uses_matching_reference_definition(peptide, library, monkeypatch):
    from ofoldx.analysis.structure import present_atom_noncovalent_contacts
    from ofoldx.data.molecular import MolecularSystem

    source = peptide("LYS", ligands=("ACT",))
    monkeypatch.setattr("ofoldx.analysis.structure.configured_library", lambda: library)
    default = present_atom_noncovalent_contacts(
        source, designed_positions={ComponentRef("A", 0)}, connected_designed_positions={ComponentRef("A", 0)}
    )
    result = present_atom_noncovalent_contacts(
        source,
        designed_positions={ComponentRef("A", 0)},
        connected_designed_positions={ComponentRef("A", 0)},
        library=library,
    )
    assert default == result
    atom = int(source.system.atom_index([AtomSite("B", 0, "O1")])[0])
    arrays = source.system.build()
    arrays.columns["formal_charge"][atom] = 1
    altered = Structure.from_arrays(MolecularSystem.from_arrays(arrays), source.build())
    changed = present_atom_noncovalent_contacts(
        altered,
        designed_positions={ComponentRef("A", 0)},
        connected_designed_positions={ComponentRef("A", 0)},
        library=library,
    )
    assert changed.contacts == result.contacts
    arrays = source.system.build()
    arrays.columns["atomic_number"][atom] = 6
    wrong_element = Structure.from_arrays(MolecularSystem.from_arrays(arrays), source.build())
    with pytest.raises(ValueError, match="reference element does not match"):
        present_atom_noncovalent_contacts(
            wrong_element,
            designed_positions={ComponentRef("A", 0)},
            connected_designed_positions={ComponentRef("A", 0)},
            library=library,
        )
    arrays = source.system.build()
    arrays.columns["component_definition_digest"][1] ^= 1
    mismatched = Structure.from_arrays(MolecularSystem.from_arrays(arrays), source.build())
    with pytest.raises(ValueError, match="reference definition does not match"):
        present_atom_noncovalent_contacts(
            mismatched,
            designed_positions={ComponentRef("A", 0)},
            connected_designed_positions={ComponentRef("A", 0)},
            library=library,
        )


def test_analysis_unavailable_is_distinct_from_unexpected_error(peptide, monkeypatch):
    import hydride

    from ofoldx.analysis.structure import (
        AnalysisUnavailableError,
        PresentAtomUnavailableError,
        count_noncovalent_contacts,
        present_atom_noncovalent_contacts,
        present_atom_target_delta_sasa,
    )

    source = peptide("ALA", "SER", copies=2)
    partial = _without_coordinates(source, AtomSite("A", 1, "OXT"))
    refs = {ComponentRef("A", 0), ComponentRef("A", 1)}
    with pytest.raises(AnalysisUnavailableError):
        count_noncovalent_contacts(partial, designed_positions=refs, connected_designed_positions=refs)
    empty_target = _without_coordinates(
        source,
        *[
            AtomSite("B", i, name)
            for i in range(2)
            for name in source.system.text(
                "atom",
                source.system.atom_symbol[
                    source.system.component_atom_ptr[2 + i] : source.system.component_atom_ptr[3 + i]
                ],
            )
        ],
    )
    with pytest.raises(PresentAtomUnavailableError, match="at least one present target atom"):
        present_atom_target_delta_sasa(
            empty_target, target={ComponentRef("B", 0), ComponentRef("B", 1)}, designed_positions=refs
        )

    def fail_hydride(_array):
        raise ValueError("unexpected hydride failure")

    monkeypatch.setattr(hydride, "add_hydrogen", fail_hydride)
    with pytest.raises(ValueError, match="unexpected hydride failure") as caught:
        present_atom_noncovalent_contacts(source, designed_positions=refs, connected_designed_positions=refs)
    assert not isinstance(caught.value, AnalysisUnavailableError)


def test_present_hydrophobic_patch_without_exposed_carbon_is_unavailable(peptide):
    from ofoldx.analysis.structure import PresentAtomUnavailableError, present_atom_largest_hydrophobic_patch_area

    with pytest.raises(PresentAtomUnavailableError, match="no exposed hydrophobic carbon"):
        present_atom_largest_hydrophobic_patch_area(peptide("SER"))


@pytest.mark.parametrize("argument", ["design", "connected", "target"])
def test_present_analysis_rejects_missing_component_references(peptide, argument):
    from ofoldx.analysis.structure import (
        PresentAtomUnavailableError,
        present_atom_noncovalent_contacts,
        present_atom_target_delta_sasa,
    )

    structure = peptide("ALA", copies=2)
    missing = ComponentRef("missing", 0)
    if argument == "target":
        with pytest.raises(PresentAtomUnavailableError, match="components"):
            present_atom_target_delta_sasa(structure, target={missing}, designed_positions={ComponentRef("A", 0)})
    else:
        design = {missing} if argument == "design" else {ComponentRef("A", 0)}
        connected = design | {missing} if argument == "connected" else design
        with pytest.raises(PresentAtomUnavailableError, match="positions"):
            present_atom_noncovalent_contacts(
                structure, designed_positions=design, connected_designed_positions=connected
            )
