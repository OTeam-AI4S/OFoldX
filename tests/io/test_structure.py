from __future__ import annotations

from pathlib import Path

import pytest

from ofoldx.data import ChemicalAtom, ChemicalBond, ChemicalComponent, Molecule
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


@pytest.mark.xfail(
    reason="reopened §3.5: altloc selection policy and multi-altloc plural expansion; "
    "§3.5.1 removed the altloc_ids carrier field (one altloc realization is one Structure)",
    strict=False,
)
def test_pdb_altloc_policy_and_plural_expansion_are_residue_consistent(tmp_path: Path) -> None:
    path = tmp_path / "altloc.pdb"
    path.write_text(_ALTLOC_PDB, encoding="ascii")

    # §3.5.1 removed the altloc_ids carrier field: one altloc realization is one Structure and
    # altloc identity stays reopened §3.5. Only the residue-consistent plural expansion count and
    # a consistent per-selection site axis remain observable on the frozen carrier.
    highest = read_structure(path)
    first = read_structure(path, altloc="first")
    expanded = read_structures(path)

    assert len(expanded) == 2
    assert len(highest.sites) == len(first.sites)


@pytest.mark.xfail(
    reason="reopened §3.5: the atom_layout site-materialization knob (complete_heavy); "
    "§3.5.1 removed the atom_layout carrier field and the system.sites() accessor",
    strict=False,
)
def test_complete_heavy_materializes_declared_sites_with_missing_masks(tmp_path: Path) -> None:
    path = tmp_path / "alanine.pdb"
    path.write_text(_ALTLOC_PDB, encoding="ascii")

    structure = read_structure(path, atom_layout="complete_heavy")

    # §3.5.1/§3.5.2 removed the atom_layout carrier field and the system.sites() accessor;
    # heavy-atom realization is a reopened §3.5 read-time element filter. Only the materialized
    # site axis and its partially-false mask remain observable on the frozen carrier.
    assert int(structure.coordinate_mask.sum()) == 5
    assert len(structure.sites) > 5


def test_unknown_component_requires_an_explicit_complete_graph(tmp_path: Path) -> None:
    path = tmp_path / "ligand.pdb"
    path.write_text(
        "HETATM    1  C1  LIG L   1       0.000   0.000   0.000  1.00 10.00           C  \n"
        "HETATM    2  O1  LIG L   1       1.200   0.000   0.000  1.00 10.00           O  \n"
        "CONECT    1    2\nEND\n",
        encoding="ascii",
    )
    with pytest.raises(ValueError, match="unknown component 'LIG'"):
        read_structure(path)

    ligand = ChemicalComponent(
        "LIG",
        atoms=(ChemicalAtom("C1", "C"), ChemicalAtom("O1", "O")),
        bonds=frozenset({ChemicalBond("C1", "O1")}),
    )
    structure = read_structure(path, components={"LIG": ligand})
    assert len(structure.system.entities) == 1
    assert structure.system.atom(structure.sites[0]).element == "C"


@pytest.mark.xfail(
    reason="reopened §3.5: complete_all hydrogen-count materialization is a reopened §3.5 I/O "
    "graph transform; §3.5.1/§3.3 also removed component provenance",
    strict=False,
)
def test_complete_all_materializes_hydrogen_counts_as_an_explicit_graph_transform(tmp_path: Path) -> None:
    path = tmp_path / "methane.pdb"
    path.write_text(
        "HETATM    1  C1  MET L   1       0.000   0.000   0.000  1.00 10.00           C  \nEND\n",
        encoding="ascii",
    )
    component = ChemicalComponent(
        "MET",
        atoms=(
            ChemicalAtom("C1", "C", implicit_hydrogens=3),
            ChemicalAtom("O1", "O", implicit_hydrogens=1),
        ),
        bonds=frozenset({ChemicalBond("C1", "O1")}),
        reference_coordinates=((0.0, 0.0, 0.0), (1.4, 0.0, 0.0)),
    )

    structure = read_structure(path, atom_layout="complete_all", components={"MET": component})

    assert len(structure.sites) == 6
    assert int(structure.coordinate_mask.sum()) == 1
    molecule = next(iter(structure.system.entities.values()))
    assert type(molecule) is Molecule
    materialized = molecule.components[0]
    assert all(atom.explicit_hydrogens == atom.implicit_hydrogens == 0 for atom in materialized.atoms)
    # §3.3/§3.5.1 removed ChemicalComponent.provenance; the materialization transform record is
    # reopened, so only the resulting hydrogen graph and reference coordinates remain observable.
    reference_by_atom = dict(zip((atom.atom_id for atom in materialized.atoms), materialized.reference_coordinates))
    assert reference_by_atom["C1"] == (0.0, 0.0, 0.0)
    assert reference_by_atom["O1"] == (1.4, 0.0, 0.0)
    assert all(reference_by_atom[atom.atom_id] is None for atom in materialized.atoms if atom.element == "H")
