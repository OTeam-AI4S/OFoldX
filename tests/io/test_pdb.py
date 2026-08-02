from __future__ import annotations

from pathlib import Path

import torch

from ofoldx.data import (
    ATOMIC_NUMBER_BY_ELEMENT,
    ChemicalAtom,
    ChemicalComponent,
    MolecularSystem,
    PolymerSequence,
    Structure,
)
from ofoldx.io import read_structure, write_structure


class TestPdbIO:
    def test_reader_normalizes_elements_and_keeps_residue_instances_separate(
        self,
        tmp_path: Path,
    ) -> None:
        path = tmp_path / "elements.pdb"
        path.write_text(
            "HETATM    1 ZN   ZN  Z   1       0.000   0.000   0.000  1.00 10.00          ZN  \n"
            "HETATM    2 CL   CL  C   2       2.000   0.000   0.000  1.00 10.00          CL  \n"
            "END\n",
            encoding="ascii",
        )
        components = {
            "ZN": ChemicalComponent(
                "ZN",
                (ChemicalAtom("ZN", "Zn"),),
                frozenset(),
            ),
            "CL": ChemicalComponent(
                "CL",
                (ChemicalAtom("CL", "Cl"),),
                frozenset(),
            ),
        }

        structure = read_structure(path, components=components)
        assert structure.system.atomic_number.tolist() == [
            ATOMIC_NUMBER_BY_ELEMENT["Zn"],
            ATOMIC_NUMBER_BY_ELEMENT["Cl"],
        ]
        assert structure.system.entity_symbol.numel() == 2
        assert structure.system.instance_symbol.numel() == 2

    def test_assembly_materializes_two_instances_of_one_entity(
        self,
        tmp_path: Path,
    ) -> None:
        path = tmp_path / "assembly.pdb"
        path.write_text(
            """REMARK 350
REMARK 350 BIOMOLECULE: 1
REMARK 350 APPLY THE FOLLOWING TO CHAINS: A
REMARK 350   BIOMT1   1  1.000000  0.000000  0.000000        0.00000
REMARK 350   BIOMT2   1  0.000000  1.000000  0.000000        0.00000
REMARK 350   BIOMT3   1  0.000000  0.000000  1.000000        0.00000
REMARK 350   BIOMT1   2  1.000000  0.000000  0.000000       10.00000
REMARK 350   BIOMT2   2  0.000000  1.000000  0.000000        0.00000
REMARK 350   BIOMT3   2  0.000000  0.000000  1.000000        0.00000
ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 10.00           C
END
""",
            encoding="ascii",
        )

        structure = read_structure(path, assembly_id="1")
        ca_rows = [
            index
            for index in range(structure.system.atom_symbol.numel())
            if structure.system.atom_site(index).atom_id == "CA"
        ]

        assert structure.system.entity_symbol.numel() == 1
        assert [structure.system.instance_id(index) for index in range(2)] == ["A#0", "A#1"]
        assert structure.coordinates[ca_rows, 0].tolist() == [0.0, 10.0]

    def test_declared_missing_residues_remain_on_the_component_axis(
        self,
        tmp_path: Path,
    ) -> None:
        path = tmp_path / "missing.pdb"
        missing = "".join(f"REMARK 465     ALA A {number:>5}\n" for number in range(3, 10))
        path.write_text(
            missing
            + "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 10.00           C  \n"
            + "ATOM      2  CA  ALA A   2       1.000   0.000   0.000  1.00 10.00           C  \n"
            + "ATOM      3  CA  ALA A  10       2.000   0.000   0.000  1.00 10.00           C  \n"
            + "END\n",
            encoding="ascii",
        )

        structure = read_structure(path)
        assert structure.system.component_symbol.numel() == 10
        assert structure.coordinate_mask.sum().item() == 3
        assert structure.residue_id.shape == (10,)

    def test_writer_projects_observed_polymer_atoms(self, tmp_path: Path) -> None:
        system = MolecularSystem.from_entities(
            {"protein": PolymerSequence.from_one_letter("A")},
            {"A": "protein"},
        )
        atom_count = system.atom_symbol.numel()
        structure = Structure(
            system=system,
            coordinates=torch.arange(
                atom_count * 3,
                dtype=torch.float32,
            ).reshape(atom_count, 3),
            coordinate_mask=torch.ones(atom_count, dtype=torch.bool),
            occupancy=torch.zeros(atom_count),
            occupancy_mask=torch.zeros(atom_count, dtype=torch.bool),
            b_factor=torch.zeros(atom_count),
            b_factor_mask=torch.zeros(atom_count, dtype=torch.bool),
            source_symbols=("1",),
            residue_id=torch.tensor([0]),
            insertion_code=torch.tensor([-1]),
            batch_size=[],
        )
        path = tmp_path / "alanine.pdb"

        write_structure(structure, path)
        restored = read_structure(path, include_hydrogens=True)

        assert restored.system.instance_id(0) == "A"
        assert torch.equal(restored.coordinates, structure.coordinates)
        assert "ATOM" in path.read_text(encoding="ascii")
