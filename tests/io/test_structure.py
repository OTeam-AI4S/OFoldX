from __future__ import annotations

from pathlib import Path

from ofoldx.data import AtomSite
from ofoldx.io import read_structure

_ALTLOC_PDB = """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 10.00           N
ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00 10.00           C
ATOM      3  C   ALA A   1       2.000   0.000   0.000  1.00 10.00           C
ATOM      4  O   ALA A   1       3.000   0.000   0.000  1.00 10.00           O
ATOM      5  CB AALA A   1       1.000   1.000   0.000  0.30 10.00           C
ATOM      6  CB BALA A   1       1.000  -1.000   0.000  0.70 10.00           C
ATOM      7  H   ALA A   1      -1.000   0.000   0.000  1.00 10.00           H
HETATM    8  O   HOH W   2       5.000   0.000   0.000  1.00 10.00           O
END
"""


class TestStructureIO:
    def test_altloc_selects_one_coordinate_on_the_canonical_atom_axis(
        self,
        tmp_path: Path,
    ) -> None:
        path = tmp_path / "altloc.pdb"
        path.write_text(_ALTLOC_PDB, encoding="ascii")

        highest = read_structure(path)
        first = read_structure(path, altloc="first")
        cb = highest.system.atom_index(AtomSite("A", 0, "CB"))

        assert highest.coordinates[cb, 1].item() == -1.0
        assert first.coordinates[cb, 1].item() == 1.0
        assert highest.coordinate_mask.shape == highest.system.atom_symbol.shape

    def test_water_and_hydrogen_filters_change_the_canonical_graph(
        self,
        tmp_path: Path,
    ) -> None:
        path = tmp_path / "filtered.pdb"
        path.write_text(_ALTLOC_PDB, encoding="ascii")

        complete = read_structure(path)
        filtered = read_structure(
            path,
            include_water=False,
            include_hydrogens=False,
        )

        assert complete.system.entity_symbol.numel() == 2
        assert filtered.system.entity_symbol.numel() == 1
        assert 1 in complete.system.atomic_number.tolist()
        assert 1 not in filtered.system.atomic_number.tolist()
