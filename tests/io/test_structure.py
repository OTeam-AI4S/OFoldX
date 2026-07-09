from __future__ import annotations

import pytest

from ofoldx.data.structure import AtomRecord, ResidueKey, Structure
from ofoldx.io import read_mmcif, read_pdb, write_mmcif, write_structure

ATOM_SITE_COLUMNS = (
    "group_PDB",
    "id",
    "type_symbol",
    "label_atom_id",
    "label_alt_id",
    "label_comp_id",
    "label_asym_id",
    "label_entity_id",
    "label_seq_id",
    "pdbx_PDB_ins_code",
    "Cartn_x",
    "Cartn_y",
    "Cartn_z",
    "occupancy",
    "B_iso_or_equiv",
    "pdbx_formal_charge",
    "auth_seq_id",
    "auth_comp_id",
    "auth_asym_id",
    "auth_atom_id",
    "pdbx_PDB_model_num",
)
STRUCT_CONN_COLUMNS = (
    "id",
    "conn_type_id",
    "ptnr1_label_asym_id",
    "ptnr1_label_comp_id",
    "ptnr1_label_seq_id",
    "ptnr1_label_atom_id",
    "pdbx_ptnr1_PDB_ins_code",
    "ptnr2_label_asym_id",
    "ptnr2_label_comp_id",
    "ptnr2_label_seq_id",
    "ptnr2_label_atom_id",
    "pdbx_ptnr2_PDB_ins_code",
)


@pytest.fixture
def write(tmp_path):
    r"""Write ``text`` to ``name`` under the test tmp dir and return the path."""

    def _write(name: str, text: str):
        path = tmp_path / name
        path.write_text(text)
        return path

    return _write


def cif(*, atoms, extra=()):
    r"""Assemble a minimal mmCIF: an ``_atom_site`` loop plus optional extra blocks."""

    lines = (
        ["data_t", "#", "loop_"]
        + [f"_atom_site.{column}" for column in ATOM_SITE_COLUMNS]
        + list(atoms)
        + ["#"]
        + list(extra)
    )
    return "\n".join(lines) + "\n"


def test_read_pdb(write):
    structure = read_pdb(
        write(
            "tiny.pdb",
            "\n".join(
                [
                    "ATOM      1  N   ALA A   1      11.104  13.207   8.111  1.00 20.00           N",
                    "ATOM      2  CA  ALA A   1      12.560  13.228   8.211  1.00 20.00           C",
                    "ATOM      3  N   GLY A   2      13.104  14.207   9.111  1.00 20.00           N",
                    "END",
                ]
            ),
        )
    )
    assert len(structure.atoms) == 3
    assert len(structure.residues) == 2
    assert structure.atoms[0].name == "N"
    assert structure.atoms[0].element == "N"
    assert structure.atoms[0].residue.chain_id == "A"
    assert structure.atoms[1].x == 12.56

    # Only the first MODEL is kept.
    multi = read_pdb(
        write(
            "multi.pdb",
            "\n".join(
                [
                    "MODEL        1",
                    "ATOM      1  CA  ALA A   1       1.000   0.000   0.000  1.00 20.00           C",
                    "ENDMDL",
                    "MODEL        2",
                    "ATOM      2  CA  ALA A   1       9.000   0.000   0.000  1.00 20.00           C",
                    "ENDMDL",
                    "END",
                ]
            ),
        )
    )
    assert len(multi.atoms) == 1
    assert multi.atoms[0].x == 1.0


def test_read_mmcif(write):
    # Hydrogens are dropped, only the first altloc of CA survives, HETATM sets hetero.
    parsed = read_mmcif(
        write(
            "tiny.cif",
            cif(
                atoms=[
                    "ATOM 1 N N . ALA A 1 1 ? 1.000 2.000 3.000 1.00 20.00 ? 10 ALA H N 1",
                    "ATOM 2 H H . ALA A 1 1 ? 9.000 9.000 9.000 1.00 20.00 ? 10 ALA H H 1",
                    "ATOM 3 C CA B ALA A 1 1 ? 4.000 5.000 6.000 1.00 20.00 ? 10 ALA H CA 1",
                    "ATOM 4 C CA A ALA A 1 1 ? 7.000 8.000 9.000 1.00 20.00 ? 10 ALA H CA 1",
                    "HETATM 5 C C1 . LIG B 2 . ? 7.000 8.000 9.000 1.00 20.00 ? 1 LIG L C1 1",
                    "ATOM 6 C C . ALA A 1 1 ? 0.000 0.000 0.000 1.00 20.00 ? 10 ALA H C 2",
                ]
            ),
        )
    )
    assert [atom.name for atom in parsed.atoms] == ["N", "CA", "C1"]
    assert parsed.atoms[1].x == 4.0
    assert parsed.atoms[2].hetero

    # Inter-residue covalent bond CA(1) <-> ligand C1(3) is kept; metal coordination is excluded.
    bonded = read_mmcif(
        write(
            "covalent.cif",
            cif(
                atoms=[
                    "ATOM   1 N  N  . ALA A 1 1 ? 0.000 0.000 0.000 1.00 20.00 ? 1 ALA A N  1",
                    "ATOM   2 C  CA . ALA A 1 1 ? 1.500 0.000 0.000 1.00 20.00 ? 1 ALA A CA 1",
                    "ATOM   3 C  C  . ALA A 1 1 ? 2.400 0.900 0.000 1.00 20.00 ? 1 ALA A C  1",
                    "HETATM 4 C  C1 . LIG B 2 . ? 2.800 0.000 0.000 1.00 20.00 ? 2 LIG B C1 1",
                    "HETATM 5 ZN ZN . ZN  C 3 . ? 0.500 0.500 0.500 1.00 20.00 ? 3 ZN  C ZN 1",
                ],
                extra=[
                    "loop_",
                    *[f"_struct_conn.{column}" for column in STRUCT_CONN_COLUMNS],
                    "1 covale A ALA 1 CA ? B LIG . C1 ?",
                    "2 metalc A ALA 1 N  ? C ZN  . ZN ?",
                    "#",
                ],
            ),
        )
    )
    assert bonded.bonds == ((1, 3),)

    # A modified residue (MSE) inherits the polypeptide entity type, not "ligand".
    typed = read_mmcif(
        write(
            "entity.cif",
            cif(
                atoms=[
                    "ATOM   1 C CA . ALA A 1 1 ? 0 0 0 1 20 ? 1 ALA A CA 1",
                    "HETATM 2 C CA . MSE A 1 2 ? 4 0 0 1 20 ? 2 MSE A CA 1",
                    "HETATM 3 C C1 . LIG B 2 . ? 9 0 0 1 20 ? 1 LIG B C1 1",
                ],
                extra=["loop_", "_entity_poly.entity_id", "_entity_poly.type", "1 'polypeptide(L)'", "#"],
            ),
        )
    )
    assert [atom.mol_type for atom in typed.atoms] == ["protein", "protein", "ligand"]


@pytest.mark.parametrize(
    ("name", "reader", "text"),
    [
        ("empty.pdb", read_pdb, "END\n"),
        (
            "empty.cif",
            read_mmcif,
            cif(atoms=["ATOM 1 H H . ALA A 1 1 ? 1.0 2.0 3.0 1.00 20.00 ? 10 ALA H H 1"]),
        ),
    ],
)
def test_empty_structure_raises(write, name, reader, text):
    # A hydrogen-only input counts as empty once H is filtered.
    with pytest.raises(ValueError, match="at least one atom"):
        reader(write(name, text))


def _protein_structure() -> Structure:
    r"""A tiny two-residue, two-chain protein structure with one inter-residue bond."""

    atoms = (
        AtomRecord("N", "N", "ALA", ResidueKey("A", 1), 1.0, 2.0, 3.0),
        AtomRecord("CA", "C", "ALA", ResidueKey("A", 1), 1.5, 2.5, 3.5),
        AtomRecord("C", "C", "ALA", ResidueKey("A", 1), 2.0, 3.0, 4.0),
        AtomRecord("N", "N", "GLY", ResidueKey("B", 2), 5.0, 6.0, 7.0),
    )
    return Structure(atoms, bonds=((2, 3),))


@pytest.mark.parametrize("suffix", ["pdb", "cif"])
def test_write_read_round_trip(tmp_path, suffix):
    # write_structure -> read_* recovers atom identity and coordinates for both formats.
    structure = _protein_structure()
    path = tmp_path / f"structure.{suffix}"
    write_structure(structure, path)
    loaded = read_mmcif(path) if suffix == "cif" else read_pdb(path)
    assert [atom.name for atom in loaded.atoms] == [atom.name for atom in structure.atoms]
    assert [atom.element for atom in loaded.atoms] == [atom.element for atom in structure.atoms]
    assert [atom.residue for atom in loaded.atoms] == [atom.residue for atom in structure.atoms]
    for loaded_atom, original in zip(loaded.atoms, structure.atoms):
        assert (loaded_atom.x, loaded_atom.y, loaded_atom.z) == pytest.approx(
            (original.x, original.y, original.z), abs=1e-3
        )


def test_write_mmcif_currently_drops_inter_residue_bonds(tmp_path):
    # Known limitation: the installed biotite does not emit struct_conn, so inter-residue bonds are not
    # written; atoms still round-trip. If a future biotite writes them, this test flags the revisit.
    path = tmp_path / "bonded.cif"
    write_mmcif(_protein_structure(), path)
    reloaded = read_mmcif(path)
    assert len(reloaded.atoms) == 4
    assert reloaded.bonds == ()


def test_write_structure_rejects_unknown_format(tmp_path):
    with pytest.raises(ValueError, match="unsupported structure format"):
        write_structure(_protein_structure(), tmp_path / "structure.xyz")
