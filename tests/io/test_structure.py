from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ofoldx.data import AtomSite, ChemicalAtom, ChemicalComponent, MolecularSystem, Molecule, Structure
from ofoldx.io import read_structure, write_structure

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
    def test_component_resolver(self, tmp_path: Path) -> None:
        path = tmp_path / "ligand.pdb"
        path.write_text(
            "HETATM    1  N1  LIG A   1       0.000   0.000   0.000  1.00 10.00           N  \n"
            "HETATM    2  C1  LIG A   1       1.000   0.000   0.000  1.00 10.00           C  \n"
            "END\n",
            encoding="ascii",
        )
        component = ChemicalComponent(
            "LIG",
            (ChemicalAtom("N1", "N", formal_charge=1), ChemicalAtom("C1", "C")),
            frozenset(),
        )

        resolved = read_structure(path, component_resolver=lambda component_id: component)

        assert resolved.system.formal_charge.tolist() == [1, 0]
        assert not resolved.source_formal_charge_mask.any()

        charged = path.read_text(encoding="ascii").replace("           N  \n", "           N1+\n")
        path.write_text(charged, encoding="ascii")
        explicit = read_structure(path, component_resolver=lambda component_id: component)
        assert explicit.source_formal_charge_mask.tolist() == [True, False]

    def test_unknown_element_mmcif(self, tmp_path: Path) -> None:
        component = ChemicalComponent("UNX", (ChemicalAtom("X1", "X"),), frozenset())
        system = MolecularSystem.from_entities({"unknown": Molecule((component,))}, {"A": "unknown"})
        structure = Structure(
            system=system,
            coordinates=torch.tensor([[1.0, 2.0, 3.0]]),
            coordinate_mask=torch.ones(1, dtype=torch.bool),
            occupancy=torch.ones(1),
            occupancy_mask=torch.ones(1, dtype=torch.bool),
            b_factor=torch.zeros(1),
            b_factor_mask=torch.ones(1, dtype=torch.bool),
            source_symbols=("1",),
            residue_id=torch.tensor([0]),
            insertion_code=torch.tensor([-1]),
            standard_component_id=torch.tensor([-1]),
            batch_size=[],
        )

        mmcif_path = tmp_path / "unknown.cif"
        write_structure(structure, mmcif_path)
        restored = read_structure(mmcif_path)

        assert restored.system.atomic_number.tolist() == [0]
        assert restored.system.component_topology_complete.tolist() == [True]

    def test_unknown_element_pdb(self, tmp_path: Path) -> None:
        component = ChemicalComponent("UNX", (ChemicalAtom("X1", "X"),), frozenset())
        system = MolecularSystem.from_entities({"unknown": Molecule((component,))}, {"A": "unknown"})
        structure = Structure(
            system=system,
            coordinates=torch.tensor([[1.0, 2.0, 3.0]]),
            coordinate_mask=torch.ones(1, dtype=torch.bool),
            occupancy=torch.ones(1),
            occupancy_mask=torch.ones(1, dtype=torch.bool),
            b_factor=torch.zeros(1),
            b_factor_mask=torch.ones(1, dtype=torch.bool),
            source_symbols=("1",),
            residue_id=torch.tensor([0]),
            insertion_code=torch.tensor([-1]),
            standard_component_id=torch.tensor([-1]),
            batch_size=[],
        )

        with pytest.raises(ValueError, match="write mmCIF to keep its chemistry"):
            write_structure(structure, tmp_path / "complete-unknown.pdb")

        incomplete = structure.replace(
            system=structure.system.replace(
                component_topology_complete=torch.zeros_like(structure.system.component_topology_complete)
            )
        )
        pdb_path = tmp_path / "incomplete-unknown.pdb"
        write_structure(incomplete, pdb_path)
        restored = read_structure(pdb_path)

        assert restored.system.atomic_number.tolist() == [0]
        assert restored.system.component_topology_complete.tolist() == [False]

    def test_altloc(
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

    def test_filters(
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
