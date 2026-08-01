from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from ofoldx.data import (
    AtomSite,
    ChemicalAtom,
    ChemicalBond,
    ChemicalComponent,
    MolecularSystem,
    Molecule,
    SystemBond,
    chemistry,
)


@pytest.mark.parametrize(
    "smiles",
    [
        "c1ccccc1",
        "C[C@H](O)C(=O)O",
        "F/C=C/F",
        "F/C=C\\F",
        "C1CCCCC1",
        "[Na+]",
        "[13CH3][C@@H](F)Cl",
        "c1cc[nH]c1",
        "[O-][N+](=O)c1ccccc1",
        "C[C@H](F)[C@H](O)[C@H](F)C",
        "C[C@H](F)[C@@H](O)[C@H](F)C",
    ],
)
def test_connected_validation_preserves_graph_and_acceptance(smiles, monkeypatch):
    from rdkit import Chem

    molecule = Molecule.from_smiles(smiles)
    system = MolecularSystem.from_entities({"molecule": molecule}, {"A": "molecule", "B": "molecule"})
    before = deepcopy(system.state_dict())
    reference = chemistry.to_rdkit_system(system)
    observed = []
    original = chemistry._to_rdkit_graph

    def projection(*args, **kwargs):
        molecule = original(*args, **kwargs)
        observed.append(Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True))
        return molecule

    monkeypatch.setattr(chemistry, "_to_rdkit_graph", projection)
    chemistry.validate_rdkit_system(system)
    expected = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), canonical=True, isomericSmiles=True)
    assert sorted(observed) == [expected, expected]
    assert sorted(Chem.MolToSmiles(reference, canonical=True, isomericSmiles=True).split(".")) == [expected, expected]
    torch.testing.assert_close(system.state_dict(), before, rtol=0, atol=0, equal_nan=True)


def test_cross_instance_ring_is_validated_whole(monkeypatch):
    component = ChemicalComponent(
        "SEG", (ChemicalAtom("C1", "C"), ChemicalAtom("C2", "C")), frozenset({ChemicalBond("C1", "C2")})
    )
    system = MolecularSystem.from_entities(
        {"segment": Molecule((component,))},
        {"A": "segment", "B": "segment", "C": "segment"},
        (
            SystemBond(AtomSite("A", 0, "C2"), AtomSite("B", 0, "C1")),
            SystemBond(AtomSite("B", 0, "C2"), AtomSite("A", 0, "C1")),
        ),
    )
    seen = []
    original = chemistry._to_rdkit_graph

    def projection(system, atoms, edges, description, **kwargs):
        value = original(system, atoms, edges, description, **kwargs)
        seen.append((len(atoms), value.GetRingInfo().NumRings()))
        return value

    monkeypatch.setattr(chemistry, "_to_rdkit_graph", projection)
    chemistry.validate_rdkit_system(system)
    assert sorted(seen) == [(2, 0), (4, 1)]


@pytest.mark.parametrize("smiles", ["C/C(CC)=C(/Cl)Br", "C/C(CC)=C(\\Cl)Br"])
def test_declared_double_bond_stereo_without_defining_atoms_uses_cip_priorities(smiles):
    from rdkit import Chem

    reference = Chem.MolFromSmiles(smiles)
    stereo = str(reference.GetBondBetweenAtoms(1, 4).GetStereo()).removeprefix("STEREO")
    component = ChemicalComponent(
        "LIG",
        tuple(ChemicalAtom(f"C{i}", "C") for i in range(1, 6)) + (ChemicalAtom("Cl", "Cl"), ChemicalAtom("Br", "Br")),
        frozenset(
            {
                ChemicalBond("C1", "C2"),
                ChemicalBond("C2", "C3"),
                ChemicalBond("C3", "C4"),
                ChemicalBond("C2", "C5", order=2.0, stereo=stereo),
                ChemicalBond("C5", "Cl"),
                ChemicalBond("C5", "Br"),
            }
        ),
    )
    system = MolecularSystem.from_entities({"ligand": Molecule((component,))}, {"A": "ligand", "B": "ligand"})
    # CCD E/Z declarations need not supply defining atom indices. Methyl and
    # ethyl both start with carbon, so atomic number alone cannot rank them.
    system.bond_stereo_index.fill_(-1)
    projected = chemistry.to_rdkit_system(system)
    expected = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    assert sorted(Chem.MolToSmiles(projected).split(".")) == [expected, expected]


def test_dative_cross_instance_bond_is_not_cut(monkeypatch):
    donor = ChemicalComponent("DON", (ChemicalAtom("N", "N", formal_charge=0),), frozenset())
    metal = ChemicalComponent("ZN", (ChemicalAtom("ZN", "Zn", formal_charge=2),), frozenset())
    system = MolecularSystem.from_entities(
        {"donor": Molecule((donor,)), "metal": Molecule((metal,))},
        {"A": "donor", "B": "metal"},
        (SystemBond(AtomSite("A", 0, "N"), AtomSite("B", 0, "ZN"), kind="dative", direction="a_to_b"),),
    )
    seen = []
    original = chemistry._to_rdkit_graph

    def projection(system, atoms, edges, description, **kwargs):
        seen.append((len(atoms), len(edges)))
        return original(system, atoms, edges, description, **kwargs)

    monkeypatch.setattr(chemistry, "_to_rdkit_graph", projection)
    chemistry.validate_rdkit_system(system)
    assert seen == [(2, 1)]


def test_nonunique_declared_ez_is_rejected():
    component = ChemicalComponent(
        "LIG",
        tuple(
            ChemicalAtom(name, element)
            for name, element in (("C1", "C"), ("C2", "C"), ("C3", "C"), ("C4", "C"), ("F", "F"), ("Cl", "Cl"))
        ),
        frozenset(
            {
                ChemicalBond("C1", "C2"),
                ChemicalBond("C3", "C2"),
                ChemicalBond("C2", "C4", order=2.0, stereo="E"),
                ChemicalBond("C4", "F"),
                ChemicalBond("C4", "Cl"),
            }
        ),
    )
    system = MolecularSystem.from_entities({"ligand": Molecule((component,))}, {"A": "ligand"})
    with pytest.raises(ValueError, match="no unique E/Z assignment"):
        chemistry.to_rdkit_system(system)


@pytest.mark.parametrize("unknown", [False, True])
def test_invalid_or_unknown_bonds_still_raise(unknown):
    atoms = (ChemicalAtom("C", "C"),) + tuple(ChemicalAtom(f"F{i}", "F") for i in range(5))
    bonds = frozenset(ChemicalBond("C", f"F{i}", order=None if unknown else 1.0) for i in range(5))
    molecule = Molecule((ChemicalComponent("BAD", atoms, bonds),))
    system = MolecularSystem.from_entities({"bad": molecule}, {"A": "bad"})
    for validate in (chemistry.to_rdkit_system, chemistry.validate_rdkit_system):
        with pytest.raises(ValueError):
            validate(system)


def test_nonstereogenic_atom_declaration_is_rejected():
    component = ChemicalComponent("LIG", (ChemicalAtom("C", "C", stereo="R"),), frozenset())
    system = MolecularSystem.from_entities({"ligand": Molecule((component,))}, {"A": "ligand"})
    with pytest.raises(ValueError, match="exact CIP assignment"):
        chemistry.to_rdkit_system(system)
