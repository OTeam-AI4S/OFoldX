from __future__ import annotations

from pathlib import Path

import pytest

from ofoldx.data import AtomSite, ChemicalAtom, ChemicalBond, ChemicalComponent
from ofoldx.io import read_structure
from ofoldx.io.structure import NoRetainedStructureError

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

        complete = read_structure(path, include_water=True, include_hydrogens=True)
        filtered = read_structure(path)

        assert complete.system.entity_symbol.numel() == 2
        assert filtered.system.entity_symbol.numel() == 1
        assert 1 in complete.system.atomic_number.tolist()
        assert 1 not in filtered.system.atomic_number.tolist()

    def test_water_only_structure_requires_explicit_opt_in(self, tmp_path: Path) -> None:
        path = tmp_path / "water-only.pdb"
        path.write_text(
            """\
HETATM    1  O   HOH W   1       0.000   0.000   0.000  1.00 10.00           O
HETATM    2  H1  HOH W   1       0.900   0.000   0.000  1.00 10.00           H
HETATM    3  H2  HOH W   1      -0.300   0.800   0.000  1.00 10.00           H
END
""",
            encoding="ascii",
        )

        with pytest.raises(NoRetainedStructureError, match="only water excluded"):
            read_structure(path)
        retained = read_structure(path, include_water=True)

        assert retained.system.entity_symbol.numel() == 1
        assert retained.system.atomic_number.tolist() == [8]

    def test_default_filters_hydrogen_isotopes_before_explicit_component_matching(
        self,
        tmp_path: Path,
    ) -> None:
        path = tmp_path / "isotopes.pdb"
        path.write_text(
            """\
HETATM    1  C1  LIG A   1       0.000   0.000   0.000  1.00 10.00           C
HETATM    2  H1  LIG A   1       1.000   0.000   0.000  1.00 10.00           H
HETATM    3  D1  LIG A   1       2.000   0.000   0.000  1.00 10.00           D
HETATM    4  T1  LIG A   1       3.000   0.000   0.000  1.00 10.00           T
END
""",
            encoding="ascii",
        )
        component = ChemicalComponent(
            "LIG",
            (
                ChemicalAtom("C1", "C"),
                ChemicalAtom("H1", "H"),
                ChemicalAtom("D1", "D"),
                ChemicalAtom("T1", "T"),
            ),
            frozenset(
                {
                    ChemicalBond("C1", "H1"),
                    ChemicalBond("C1", "D1"),
                    ChemicalBond("C1", "T1"),
                }
            ),
        )

        heavy = read_structure(path, components={"LIG": component})
        complete = read_structure(path, components={"LIG": component}, include_hydrogens=True)

        assert heavy.system.atom_symbol.numel() == 1
        assert heavy.system.atomic_number.tolist() == [6]
        assert heavy.system.component_topology_complete.tolist() == [True]
        assert heavy.system.edge_index.shape[1] == 0
        assert complete.system.atom_symbol.numel() == 4
        assert complete.system.atomic_number.tolist() == [6, 1, 1, 1]
        assert complete.system.edge_index.shape[1] == 3

    def test_pdb_conect_does_not_restore_same_component_topology(self, tmp_path: Path) -> None:
        path = tmp_path / "same-component-conect.pdb"
        path.write_text(
            """\
HETATM    1  C1  LIG A   1       0.000   0.000   0.000  1.00 10.00           C
HETATM    2  O1  LIG A   1       1.000   0.000   0.000  1.00 10.00           O
CONECT    1    2
CONECT    2    1
END
""",
            encoding="ascii",
        )

        restored = read_structure(path)

        assert restored.system.component_topology_complete.tolist() == [False]
        assert restored.system.edge_index.shape[1] == 0

    def test_all_hydrogen_isotopes_require_explicit_opt_in(self, tmp_path: Path) -> None:
        path = tmp_path / "all-hydrogen.pdb"
        path.write_text(
            """\
HETATM    1  H1  ISO A   1       0.000   0.000   0.000  1.00 10.00           H
HETATM    2  D1  ISO A   1       1.000   0.000   0.000  1.00 10.00           D
HETATM    3  T1  ISO A   1       2.000   0.000   0.000  1.00 10.00           T
END
""",
            encoding="ascii",
        )
        component = ChemicalComponent(
            "ISO",
            (
                ChemicalAtom("H1", "H"),
                ChemicalAtom("D1", "H"),
                ChemicalAtom("T1", "H"),
            ),
            frozenset(),
        )

        with pytest.raises(NoRetainedStructureError, match="no heavy atoms"):
            read_structure(path, components={"ISO": component})
        retained = read_structure(
            path,
            components={"ISO": component},
            include_hydrogens=True,
        )

        assert retained.system.atomic_number.tolist() == [1, 1, 1]
        assert retained.system.isotope.tolist() == [0, 2, 3]

    def test_pdb_conect_preserves_cross_component_connectivity(self, tmp_path: Path) -> None:
        path = tmp_path / "cross-component-conect.pdb"
        path.write_text(
            """\
HETATM    1  C1  LFT A   1       0.000   0.000   0.000  1.00 10.00           C
HETATM    2  N1  RGT A   2       1.000   0.000   0.000  1.00 10.00           N
CONECT    1    2
CONECT    2    1
END
""",
            encoding="ascii",
        )

        restored = read_structure(path)

        assert restored.system.edge_index.shape[1] == 1
