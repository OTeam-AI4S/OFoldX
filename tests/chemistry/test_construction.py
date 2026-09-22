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

from __future__ import annotations

import pytest

from ofoldx.chemistry.construction import (
    Admissibility,
    Molecule,
    PolymerSequence,
    Recipe,
    RecipeBond,
    build_system,
    instantiate_component,
    polymer_sequence,
    replacement_admissibility,
)
from ofoldx.chemistry.smiles import read_smiles
from ofoldx.data.molecular import AtomSite, BondKind, BondOrigin, MonomerType
from ofoldx.errors import CODE_NAMES, ContradictionError, InvalidRequestError


def names(system, component: int) -> set[str]:
    start, stop = system.component_atom_ptr[component : component + 2]
    return set(system.text("atom", system.atom_symbol[start:stop]).tolist())


@pytest.fixture
def dipeptide(archive, library):
    r"""``G-A`` built from the CCD subset definitions."""
    recipe = Recipe({"1": polymer_sequence("GA", MonomerType.PROTEIN)}, {"A": "1"})
    system, _, _ = build_system(recipe, library([archive("GLY"), archive("ALA")]))
    return system


class TestPolymerSequence:
    @pytest.mark.parametrize(
        ("text", "monomer", "components", "polymer_type"),
        [
            ("GUOX", MonomerType.PROTEIN, ("GLY", "SEC", "PYL", "UNK"), "polypeptide(L)"),
            ("ACTN", MonomerType.RNA, ("A", "C", "U", "N"), "polyribonucleotide"),
            ("ACUI", MonomerType.DNA, ("DA", "DC", "DT", "DI"), "polydeoxyribonucleotide"),
        ],
        ids=["protein", "rna", "dna"],
    )
    def test_symbols_to_components(self, text, monomer, components, polymer_type):
        sequence = polymer_sequence(text, monomer)
        assert (sequence.components, sequence.polymer_type) == (components, polymer_type)

    @pytest.mark.parametrize("text", ["AB", "A-", "Ag"])
    def test_ambiguous_symbol_refused(self, text):
        with pytest.raises(InvalidRequestError) as caught:
            polymer_sequence(text, MonomerType.PROTEIN)
        assert caught.value.code == "request_invalid_sequence"
        assert caught.value.detail["position"] == 1


class TestBuildSystem:
    def test_junction_removes_leaving_atoms(self, dipeptide):
        assert not {"OXT", "HXT"} & names(dipeptide, 0)
        assert "H2" not in names(dipeptide, 1)
        junction = dipeptide.bond_origin == BondOrigin.JUNCTION
        assert junction.sum() == 1
        assert dipeptide.component_topology_complete.all()

    def test_smiles_ligand_component(self, ligand):
        system = ligand("OCC")
        assert len(system.atom_symbol) == 9
        assert system.text("component", system.component_symbol).tolist() == ["LIG"]

    def test_recipe_bond_unknown_atom(self, library):
        component = read_smiles("CC", component_id="LIG").component
        recipe = Recipe(
            {"L": Molecule(("LIG", "LIG"))},
            {"L": "L"},
            (RecipeBond(AtomSite("L", 0, "C1"), AtomSite("L", 1, "X9"), 1, BondKind.COVALENT),),
        )
        with pytest.raises(InvalidRequestError) as caught:
            build_system(recipe, library([]), {"LIG": component})
        assert caught.value.code == "value_invalid"

    def test_conflicting_caller_refused(self, archive, library):  # F2: no component is left without its atoms
        component = read_smiles("N[C@@H](C)C(=O)O", component_id="ALA").component
        with pytest.raises(ContradictionError) as caught:
            build_system(Recipe({"L": Molecule(("ALA",))}, {"L": "L"}), library([archive("ALA")]), {"ALA": component})
        assert caught.value.code == "read_conflicts"


class TestInstantiateComponent:
    @pytest.mark.parametrize(
        ("linked", "absent"),
        [((False, False), set()), ((True, False), {"H2"}), ((True, True), {"H2", "OXT", "HXT"})],
        ids=["free", "previous", "both"],
    )
    def test_linked_ports_lose_leaving(self, archive, linked, absent):
        rows = instantiate_component(archive("ALA"), linked=linked)
        assert set(archive("ALA").atom_names) - set(rows.names) == absent
        assert rows.claims == (True, True)

    def test_unverified_port_refused(self):
        component = read_smiles("CC", component_id="LIG").component
        with pytest.raises(InvalidRequestError) as caught:
            instantiate_component(component, linked=(True, False))
        assert caught.value.code == "value_invalid"


class TestReplacementAdmissibility:
    def test_shared_port_admissible(self, dipeptide, archive):
        check = replacement_admissibility(dipeptide, 0, archive("ALA"))
        assert check.outcome.tolist() == [Admissibility.ADMISSIBLE]

    def test_absent_endpoint_inadmissible(self, dipeptide):
        check = replacement_admissibility(dipeptide, 0, read_smiles("CC", component_id="LIG").component)
        assert check.outcome.tolist() == [Admissibility.INADMISSIBLE]
        assert CODE_NAMES[check.reason[0]] == "replacement_atom_absent"


@pytest.mark.parametrize("broken_side", ["previous", "next"])
def test_recipe_rejects_unverified_junction_and_allows_unlinked(items, glycine, define, library, broken_side):
    columns = glycine()
    atoms = columns["atoms"]
    names = atoms["atom_id"]
    leaving = atoms["pdbx_leaving_atom_flag"]
    leaving[names.index("H2" if broken_side == "previous" else "OXT")] = "N"
    component = define(items(**columns))
    definitions = library([component])
    recipe = Recipe({"P": PolymerSequence("polypeptide(L)", ("GLY", "GLY"))}, {"A": "P"})
    with pytest.raises(InvalidRequestError) as caught:
        build_system(recipe, definitions)
    assert caught.value.code == "value_invalid"
    assert "UNVERIFIED" in str(caught.value)
    assert caught.value.detail["condition"] == "LEAVING_GROUP"

    separate = Recipe({"P": PolymerSequence("polypeptide(L)", ("GLY", "GLY"), (False,))}, {"A": "P"})
    system, _, _ = build_system(separate, definitions)
    assert not (system.bond_origin == BondOrigin.JUNCTION).any()


def test_recipe_rejects_cap_facing_wrong_way(archive, library):
    recipe = Recipe({"P": PolymerSequence("polypeptide(L)", ("GLY", "ACE"))}, {"A": "P"})
    with pytest.raises(InvalidRequestError) as caught:
        build_system(recipe, library([archive("GLY"), archive("ACE")]))
    assert caught.value.code == "value_invalid"


@pytest.mark.parametrize(
    ("polymer", "components"),
    [
        ("polypeptide(L)", ("GLY", "DA")),
        ("polypeptide(L)", ("DA", "DA")),
        ("polyribonucleotide", ("GLY", "GLY")),
    ],
)
def test_recipe_rejects_junction_outside_polymer_family(archive, library, polymer, components):
    definitions = library([archive(name) for name in set(components)])
    recipe = Recipe({"P": PolymerSequence(polymer, components)}, {"A": "P"})
    with pytest.raises(InvalidRequestError) as caught:
        build_system(recipe, definitions)
    assert caught.value.code == "value_invalid"
    assert caught.value.detail["reason"] == "junction_family"

    separated = Recipe({"P": PolymerSequence(polymer, components, (False,))}, {"A": "P"})
    system, _, _ = build_system(separated, definitions)
    assert not (system.bond_origin == BondOrigin.JUNCTION).any()


def test_recipe_accepts_rna_dna_nucleic_family_junction(archive, library):
    recipe = Recipe({"P": PolymerSequence("polyribonucleotide", ("A", "DA"))}, {"A": "P"})
    system, _, _ = build_system(recipe, library([archive("A"), archive("DA")]))
    assert (system.bond_origin == BondOrigin.JUNCTION).sum() == 1
