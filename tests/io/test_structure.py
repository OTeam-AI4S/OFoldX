from __future__ import annotations

from pathlib import Path

import pytest

from ofoldx.data import ChemicalAtom, ChemicalBond, ChemicalComponent
from ofoldx.io import read_structure, read_structures

_ALTLOC_PDB = """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 10.00           N
ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00 10.00           C
ATOM      3  C   ALA A   1       2.000   0.000   0.000  1.00 10.00           C
ATOM      4  O   ALA A   1       3.000   0.000   0.000  1.00 10.00           O
ATOM      5  CB AALA A   1       1.000   1.000   0.000  0.30 10.00           C
ATOM      6  CB BALA A   1       1.000  -1.000   0.000  0.70 10.00           C
END
"""


def test_pdb_altloc_policy_and_plural_expansion_are_residue_consistent(tmp_path: Path) -> None:
    path = tmp_path / "altloc.pdb"
    path.write_text(_ALTLOC_PDB, encoding="ascii")

    # One altloc realization is one Structure; plural selection keeps each site axis consistent.
    highest = read_structure(path)
    first = read_structure(path, altloc="first")
    expanded = read_structures(path)

    assert len(expanded) == 2
    assert len(highest.sites) == len(first.sites)


def test_complete_heavy_materializes_declared_sites_with_missing_masks(tmp_path: Path) -> None:
    path = tmp_path / "alanine.pdb"
    path.write_text(_ALTLOC_PDB, encoding="ascii")

    structure = read_structure(path, atom_layout="complete_heavy")

    # The complete-heavy layout materializes declared heavy sites and marks unobserved sites false.
    assert int(structure.coordinate_mask.sum()) == 5
    assert len(structure.sites) > 5


def test_unknown_component_reads_in_observed_layout_but_completion_needs_explicit_graph(tmp_path: Path) -> None:
    path = tmp_path / "ligand.pdb"
    path.write_text(
        "HETATM    1  C1  LIG L   1       0.000   0.000   0.000  1.00 10.00           C  \n"
        "HETATM    2  O1  LIG L   1       1.200   0.000   0.000  1.00 10.00           O  \n"
        "CONECT    1    2\nEND\n",
        encoding="ascii",
    )
    # The observed layout keeps source atoms, so an unknown ligand is declared from what the file shows.
    observed = read_structure(path)
    assert len(observed.system.entities) == 1
    assert observed.system.atom(observed.sites[0]).element == "C"

    # Completing the graph needs the full declared chemistry, so an unknown component must be supplied.
    with pytest.raises(ValueError, match="unknown component 'LIG'"):
        read_structure(path, atom_layout="complete_heavy")

    ligand = ChemicalComponent(
        "LIG",
        atoms=(ChemicalAtom("C1", "C"), ChemicalAtom("O1", "O")),
        bonds=frozenset({ChemicalBond("C1", "O1")}),
    )
    completed = read_structure(path, atom_layout="complete_heavy", components={"LIG": ligand})
    assert len(completed.system.entities) == 1
    assert completed.system.atom(completed.sites[0]).element == "C"


def test_complete_all_materializes_hydrogen_counts_as_an_explicit_graph_transform(tmp_path: Path) -> None:
    path = tmp_path / "methane.pdb"
    path.write_text(
        "HETATM    1  C1  MET L   1       0.000   0.000   0.000  1.00 10.00           C  \nEND\n",
        encoding="ascii",
    )
    component = ChemicalComponent(
        "MET",
        atoms=(
            ChemicalAtom("C1", "C", explicit_hydrogens=0, implicit_hydrogens=3),
            ChemicalAtom("O1", "O", explicit_hydrogens=0, implicit_hydrogens=1),
        ),
        bonds=frozenset({ChemicalBond("C1", "O1")}),
        reference_coordinates=((0.0, 0.0, 0.0), (1.4, 0.0, 0.0)),
    )

    structure = read_structure(path, atom_layout="complete_all", components={"MET": component})

    assert len(structure.sites) == 6
    assert int(structure.coordinate_mask.sum()) == 1
    materialized = next(iter(structure.system.entities.values())).components[0]
    assert all(atom.explicit_hydrogens == atom.implicit_hydrogens == 0 for atom in materialized.atoms)
    reference_by_atom = dict(zip((atom.atom_id for atom in materialized.atoms), materialized.reference_coordinates))
    assert reference_by_atom["C1"] == (0.0, 0.0, 0.0)
    assert reference_by_atom["O1"] == (1.4, 0.0, 0.0)
    assert all(reference_by_atom[atom.atom_id] is None for atom in materialized.atoms if atom.element == "H")
