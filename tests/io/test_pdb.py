from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from ofoldx.data import (
    ATOMIC_NUMBER_BY_ELEMENT,
    BOND_DIRECTION_ENDPOINT_0_DONOR,
    BOND_DIRECTION_ENDPOINT_1_DONOR,
    BOND_KIND_DATIVE,
    BOND_ROLE_ASSEMBLY,
    BOND_ROLE_COMPONENT,
    BOND_ROLE_POLYMER,
    ENTITY_MOLECULE,
    ENTITY_PROTEIN,
    ChemicalAtom,
    ChemicalBond,
    ChemicalComponent,
    MolecularSystem,
    Molecule,
    PolymerSequence,
    Structure,
)
from ofoldx.data.components import get_component
from ofoldx.io import read_source, read_structure, write_structure

_POLYMER = (
    "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 10.00           N\n"
    "ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00 11.00           C\n"
    "ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00 12.00           C\n"
    "ATOM      4  O   ALA A   1       1.251   2.390   0.000  1.00 13.00           O\n"
)

#: PDB sources whose round trip has to reproduce the complete public structure exactly.
_ROUND_TRIP = {
    "blank_numeric_fields": _POLYMER.replace("  1.00 10.00", "            ", 1) + "END\n",
    "compound_entity": "COMPND    MOL_ID: 7; MOLECULE: TEST; CHAIN: A;\n" + _POLYMER + "END\n",
    "polymer": _POLYMER + "END\n",
    "ligand_and_water": (
        _POLYMER
        + "HETATM    5 ZN    ZN A 301       8.000   0.000   0.000  1.00 14.00          ZN\n"
        + "HETATM    6  O   HOH A 401      12.000   0.000   0.000  1.00 15.00           O\n"
        + "END\n"
    ),
    "declared_missing": (
        "SEQRES   1 A    4  ALA ALA ALA ALA\n"
        "REMARK 465     ALA A     2\n"
        "REMARK 465     ALA A     3\n"
        + _POLYMER
        + "ATOM      5  CA  ALA A   4       9.000   0.000   0.000  1.00 16.00           C\n"
        + "END\n"
    ),
    "resolution": "REMARK   2 RESOLUTION.    2.30 ANGSTROMS.\n" + _POLYMER + "END\n",
}


class TestPdbReader:
    @pytest.mark.parametrize(("occupancy", "b_factor"), [(0.33, 13.12), (0.0, 0.0), (None, None)])
    def test_observation_precision(self, tmp_path: Path, occupancy: float | None, b_factor: float | None) -> None:
        path = tmp_path / "observations.pdb"
        occupancy_field = " " * 6 if occupancy is None else f"{occupancy:6.2f}"
        b_factor_field = " " * 6 if b_factor is None else f"{b_factor:6.2f}"
        atom = f"ATOM      1  N   ALA A   1       1.234  -2.345   3.456{occupancy_field}{b_factor_field}           N"
        path.write_text(atom + "\nEND\n", encoding="ascii")
        source = read_source(path)
        rows = source.atoms.by_index(1)
        assert rows.coordinates.dtype == np.float64
        assert rows.coordinates.tolist() == [[1.234, -2.345, 3.456]]
        assert rows.occupancy.tolist() == [occupancy]
        assert rows.b_factor.tolist() == [b_factor]

        structure = read_structure(source.view())
        for field in ("coordinates", "occupancy", "b_factor"):
            assert getattr(structure, field).dtype == torch.float64
        assert structure.coordinates[structure.coordinate_mask].tolist() == [[1.234, -2.345, 3.456]]
        assert structure.occupancy_mask[structure.coordinate_mask].tolist() == [occupancy is not None]
        assert structure.b_factor_mask[structure.coordinate_mask].tolist() == [b_factor is not None]
        assert structure.occupancy[structure.coordinate_mask].tolist() == [0.0 if occupancy is None else occupancy]
        assert structure.b_factor[structure.coordinate_mask].tolist() == [0.0 if b_factor is None else b_factor]
        target = tmp_path / "observations-out.pdb"
        write_structure(structure, target)
        written = next(line for line in target.read_text().splitlines() if line.startswith("ATOM  "))
        assert written[30:66] == atom[30:66]

    def test_resolution(self, tmp_path: Path) -> None:
        path = tmp_path / "resolution.pdb"
        path.write_text(
            "REMARK   2 RESOLUTION.    2.30 ANGSTROMS.\n" + _POLYMER + "END\n",
            encoding="ascii",
        )

        structure = read_structure(path)

        assert structure.resolution.tolist() == pytest.approx([2.3])
        assert structure.system_resolution_ptr.tolist() == [0, 1]

    @pytest.mark.parametrize("value", ["2.345", "1e100", "1e-100"])
    def test_resolution_precision(self, tmp_path: Path, value: str) -> None:
        path = tmp_path / "resolution.pdb"
        path.write_text(
            f"REMARK   2 RESOLUTION. {value} ANGSTROMS.\n" + _POLYMER + "END\n",
            encoding="ascii",
        )

        with pytest.raises(ValueError, match="not representable as F7.2"):
            read_structure(path)

    @pytest.mark.parametrize("declaration", ["2.30 NANOMETERS.", "2.30", "2.30 ANGSTROMS. EXTRA"])
    def test_resolution_syntax(self, tmp_path: Path, declaration: str) -> None:
        path = tmp_path / "resolution.pdb"
        path.write_text(
            f"REMARK   2 RESOLUTION. {declaration}\n" + _POLYMER + "END\n",
            encoding="ascii",
        )

        with pytest.raises(ValueError, match="invalid PDB REMARK 2 resolution declaration"):
            read_structure(path)

    def test_duplicate_resolution(self, tmp_path: Path) -> None:
        path = tmp_path / "resolution.pdb"
        path.write_text(
            "REMARK   2 RESOLUTION. NOT APPLICABLE.\n"
            "REMARK   2 RESOLUTION.    2.30 ANGSTROMS.\n" + _POLYMER + "END\n",
            encoding="ascii",
        )
        with pytest.raises(ValueError, match="multiple REMARK 2 resolution declarations"):
            read_structure(path)

    def test_elements(
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


class TestPdbAssembly:
    def test_instances(
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

        output = tmp_path / "assembly-output.pdb"
        write_structure(structure, output)
        restored = read_structure(output)
        assert restored.system.entity_symbol.numel() == 1
        assert restored.system.instance_symbol.numel() == 2
        assert restored.system.instance_entity.tolist() == [0, 0]
        assert restored.coordinates[restored.coordinate_mask, 0].tolist() == [0.0, 10.0]

    def test_disulfide_placement(self, tmp_path: Path) -> None:
        path = tmp_path / "assembly-global-operators.pdb"
        path.write_text(
            "REMARK 350 BIOMOLECULE: 1\n"
            "REMARK 350 APPLY THE FOLLOWING TO CHAINS: A\n"
            "REMARK 350   BIOMT1   1  1.000000  0.000000  0.000000        0.00000\n"
            "REMARK 350   BIOMT2   1  0.000000  1.000000  0.000000        0.00000\n"
            "REMARK 350   BIOMT3   1  0.000000  0.000000  1.000000        0.00000\n"
            "REMARK 350   BIOMT1   2  1.000000  0.000000  0.000000       10.00000\n"
            "REMARK 350   BIOMT2   2  0.000000  1.000000  0.000000        0.00000\n"
            "REMARK 350   BIOMT3   2  0.000000  0.000000  1.000000        0.00000\n"
            "REMARK 350 APPLY THE FOLLOWING TO CHAINS: B\n"
            "REMARK 350   BIOMT1   1  1.000000  0.000000  0.000000        0.00000\n"
            "REMARK 350   BIOMT2   1  0.000000  1.000000  0.000000        0.00000\n"
            "REMARK 350   BIOMT3   1  0.000000  0.000000  1.000000        0.00000\n"
            "REMARK 350   BIOMT1   3  1.000000  0.000000  0.000000       20.00000\n"
            "REMARK 350   BIOMT2   3  0.000000  1.000000  0.000000        0.00000\n"
            "REMARK 350   BIOMT3   3  0.000000  0.000000  1.000000        0.00000\n"
            "SSBOND   1 CYS A    1    CYS B    2                          1555   1555  2.03\n"
            "ATOM      1  SG  CYS A   1       0.000   0.000   0.000  1.00 10.00           S\n"
            "TER       2      CYS A   1\n"
            "ATOM      3  SG  CYS B   2       2.000   0.000   0.000  1.00 10.00           S\n"
            "END\n",
            encoding="ascii",
        )

        structure = read_structure(path, assembly_id="1")

        assert structure.system.instance_symbol.numel() == 4
        assert int((structure.system.bond_role == BOND_ROLE_ASSEMBLY).sum()) == 1
        assert structure.coordinates[structure.coordinate_mask, 0].tolist() == [0.0, 2.0, 10.0, 22.0]

    def test_crystal_connection(self, tmp_path: Path) -> None:
        link = "LINK         C1  LIG A   1                 X1  OTH B   1".ljust(78)
        link = link[:66] + f"{2555:>6}" + link[72:]
        body = (
            "REMARK 290   SMTRY1   2  1.000000  0.000000  0.000000       10.00000\n"
            "REMARK 290   SMTRY2   2  0.000000  1.000000  0.000000        0.00000\n"
            "REMARK 290   SMTRY3   2  0.000000  0.000000  1.000000        0.00000\n"
            "REMARK 350 BIOMOLECULE: 1\n"
            "REMARK 350 APPLY THE FOLLOWING TO CHAINS: A\n"
            "REMARK 350   BIOMT1   1  1.000000  0.000000  0.000000      100.00000\n"
            "REMARK 350   BIOMT2   1  0.000000  1.000000  0.000000        0.00000\n"
            "REMARK 350   BIOMT3   1  0.000000  0.000000  1.000000        0.00000\n"
            "REMARK 350 APPLY THE FOLLOWING TO CHAINS: B\n"
            "REMARK 350   BIOMT1   2  1.000000  0.000000  0.000000      110.00000\n"
            "REMARK 350   BIOMT2   2  0.000000  1.000000  0.000000        0.00000\n"
            "REMARK 350   BIOMT3   2  0.000000  0.000000  1.000000        0.00000\n"
            "HETATM    1  C1  LIG A   1       0.000   0.000   0.000  1.00 10.00           C\n"
            "HETATM    2  X1  OTH B   1       2.000   0.000   0.000  1.00 10.00           C\n"
            "END\n"
        )
        path = tmp_path / "assembly-crystallographic-link.pdb"
        path.write_text(link + "\n" + body, encoding="ascii")

        structure = read_structure(path, assembly_id="1")

        assert structure.system.instance_symbol.numel() == 2
        assert int((structure.system.bond_role == BOND_ROLE_ASSEMBLY).sum()) == 1
        assert structure.coordinates[structure.coordinate_mask, 0].tolist() == [100.0, 112.0]

        packing = tmp_path / "assembly-excludes-crystal-contact.pdb"
        packing.write_text(link + "\n" + body.replace("110.00000", "120.00000"), encoding="ascii")
        without_contact = read_structure(packing, assembly_id="1")
        assert not bool((without_contact.system.bond_role == BOND_ROLE_ASSEMBLY).any())


class TestPdbPolymer:
    def test_missing_residues(
        self,
        tmp_path: Path,
    ) -> None:
        path = tmp_path / "missing.pdb"
        missing = "".join(f"REMARK 465     ALA A {number:>5}\n" for number in range(3, 10))
        path.write_text(
            "SEQRES   1 A   10  ALA ALA ALA ALA ALA ALA ALA ALA ALA ALA\n"
            + missing
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

        invalid = tmp_path / "missing-without-seqres.pdb"
        invalid.write_text(
            "\n".join(line for line in path.read_text(encoding="ascii").splitlines() if not line.startswith("SEQRES"))
            + "\n",
            encoding="ascii",
        )
        with pytest.raises(ValueError, match="requires a SEQRES declaration"):
            read_structure(invalid)

    def test_free_polymer_residue(self, tmp_path: Path) -> None:
        source = tmp_path / "sequence-and-free-alanine.pdb"
        lines = (
            "SEQRES   1 A    1  ALA",
            "MODRES 1ABC MSE A   7   MET  SELENOMETHIONINE",
            "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 10.00           C",
            "TER       2      ALA A   1",
            "HETATM    3  CA  ALA B   1       4.000   0.000   0.000  1.00 10.00           C",
            "END",
        )
        source.write_text("\n".join(lines) + "\n", encoding="ascii")

        structure = read_structure(source)
        instance_types = [
            int(structure.system.entity_type[int(structure.system.instance_entity[index])])
            for index in range(structure.system.instance_symbol.numel())
        ]
        assert instance_types.count(ENTITY_PROTEIN) == 1
        assert instance_types.count(ENTITY_MOLECULE) == 1

    def test_modified_polymer(self, tmp_path: Path) -> None:
        modified = tmp_path / "modified-polymer.pdb"
        modified.write_text(
            "SEQRES   1 A    3  ALA MSE ALA\n"
            "MODRES 1ABC MSE A   2   MET  SELENOMETHIONINE\n"
            "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 10.00           C\n"
            "HETATM    2  N   MSE A   2       1.000   0.000   0.000  1.00 10.00           N\n"
            "HETATM    3  CA  MSE A   2       2.000   0.000   0.000  1.00 10.00           C\n"
            "HETATM    4  C   MSE A   2       3.000   0.000   0.000  1.00 10.00           C\n"
            "ATOM      5  CA  ALA A   3       4.000   0.000   0.000  1.00 10.00           C\n"
            "END\n",
            encoding="ascii",
        )
        modified_structure = read_structure(modified)
        assert (modified_structure.system.bond_role == BOND_ROLE_POLYMER).sum().item() == 2

        source_defined_mse = ChemicalComponent(
            "MSE",
            (
                ChemicalAtom("N", "N"),
                ChemicalAtom("CA", "C"),
                ChemicalAtom("C", "C"),
                ChemicalAtom("H2", "H"),
            ),
            frozenset(
                {
                    ChemicalBond("N", "CA"),
                    ChemicalBond("CA", "C"),
                    ChemicalBond("N", "H2"),
                }
            ),
            non_leaving_atom_ids=frozenset({"H2"}),
        )
        source_defined = read_structure(modified, components={"MSE": source_defined_mse})
        assert (source_defined.system.bond_role == BOND_ROLE_POLYMER).sum().item() == 2

    def test_modified_residue_parents(self, tmp_path: Path) -> None:
        modified = ChemicalComponent(
            "MSE",
            (
                ChemicalAtom("N", "N"),
                ChemicalAtom("CA", "C"),
                ChemicalAtom("C", "C"),
            ),
            None,
        )
        system = MolecularSystem.from_entities(
            {"modified": PolymerSequence("protein", (modified,))},
            {"A": "modified", "B": "modified"},
        )
        atom_count = int(system.atom_symbol.numel())
        structure = Structure(
            system=system,
            coordinates=torch.arange(atom_count * 3, dtype=torch.float32).reshape(atom_count, 3),
            coordinate_mask=torch.ones(atom_count, dtype=torch.bool),
            occupancy=torch.ones(atom_count),
            occupancy_mask=torch.ones(atom_count, dtype=torch.bool),
            b_factor=torch.zeros(atom_count),
            b_factor_mask=torch.ones(atom_count, dtype=torch.bool),
            source_symbols=("MET", "ALA", "1"),
            residue_id=torch.full((2,), 2),
            insertion_code=torch.full((2,), -1),
            component_parent_ptr=torch.tensor([0, 1, 2], dtype=torch.int64),
            parent_component_id=torch.tensor([0, 1], dtype=torch.int64),
            batch_size=[],
        )
        path = tmp_path / "modified-residues.pdb"

        write_structure(structure, path)

        lines = path.read_text(encoding="ascii").splitlines()
        assert all(line.startswith("HETATM") for line in lines if line.startswith(("ATOM", "HETATM")))
        restored = read_structure(path)
        assert [restored.component_parents(i) for i in range(2)] == [("MET",), ("ALA",)]

        with pytest.raises(ValueError, match="multiple parent components"):
            write_structure(
                structure.replace(
                    component_parent_ptr=torch.tensor([0, 2, 3]),
                    parent_component_id=torch.tensor([0, 1, 1]),
                ),
                tmp_path / "multiple-parents.pdb",
            )

        canonical = Structure.from_sequence("N")
        canonical = canonical.replace(
            coordinates=torch.zeros_like(canonical.coordinates),
            coordinate_mask=torch.ones_like(canonical.coordinate_mask),
            source_symbols=("ASN", "1"),
            residue_id=torch.tensor([1]),
            component_parent_ptr=torch.tensor([0, 1], dtype=torch.int64),
            parent_component_id=torch.tensor([0], dtype=torch.int64),
        )
        canonical_path = tmp_path / "canonical-modres.pdb"
        write_structure(canonical, canonical_path)
        canonical_lines = canonical_path.read_text(encoding="ascii").splitlines()
        assert all(line.startswith("ATOM") for line in canonical_lines if line.startswith(("ATOM", "HETATM")))
        assert not any(line.startswith("CONECT") for line in canonical_lines)

    def test_declared_sequence(
        self,
        tmp_path: Path,
    ) -> None:
        structure = Structure.from_sequence("AA")
        first_component_stop = int(structure.system.component_atom_ptr[1])
        structure = structure.replace(
            coordinates=torch.zeros_like(structure.coordinates),
            coordinate_mask=torch.arange(structure.coordinates.shape[0]) < first_component_stop,
            source_symbols=("1", "2"),
            residue_id=torch.tensor([0, 1]),
        )
        target = tmp_path / "declared-sequence.pdb"

        write_structure(structure, target)
        lines = target.read_text(encoding="ascii").splitlines()

        assert [line for line in lines if line.startswith("SEQRES")] == ["SEQRES   1 A    2  ALA ALA"]
        assert [line for line in lines if line.startswith("REMARK 465")] == ["REMARK 465     ALA A     2"]

        dna_system = MolecularSystem.from_entities(
            {"dna": PolymerSequence("dna", (get_component("DN"),))},
            {"D": "dna"},
        )
        dna_atom_count = int(dna_system.atom_symbol.numel())
        dna = Structure(
            system=dna_system,
            coordinates=torch.zeros((dna_atom_count, 3)),
            coordinate_mask=torch.ones(dna_atom_count, dtype=torch.bool),
            occupancy=torch.zeros(dna_atom_count),
            occupancy_mask=torch.zeros(dna_atom_count, dtype=torch.bool),
            b_factor=torch.zeros(dna_atom_count),
            b_factor_mask=torch.zeros(dna_atom_count, dtype=torch.bool),
            source_symbols=("1",),
            residue_id=torch.tensor([0]),
            insertion_code=torch.full((1,), -1),
            batch_size=[],
        )
        dna_target = tmp_path / "unknown-dna.pdb"
        write_structure(dna, dna_target)
        dna_lines = dna_target.read_text(encoding="ascii").splitlines()
        assert [line for line in dna_lines if line.startswith("SEQRES")] == ["SEQRES   1 D    1    N"]
        assert {line[17:20].strip() for line in dna_lines if line.startswith("ATOM")} == {"N"}

    def test_unknown_nucleotide(self, tmp_path: Path) -> None:
        source = tmp_path / "dna-with-n.pdb"
        source.write_text(
            "SEQRES   1 A    3   DA   N  DG\n"
            "ATOM      1  P    DA A   1       0.000   0.000   0.000  1.00 10.00           P\n"
            "ATOM      2  P     N A   2       1.000   0.000   0.000  1.00 10.00           P\n"
            "ATOM      3  P    DG A   3       2.000   0.000   0.000  1.00 10.00           P\n"
            "TER       4       DG A   3\n"
            "END\n",
            encoding="ascii",
        )

        structure = read_structure(source)
        component_ids = [structure.system.symbols[int(symbol)] for symbol in structure.system.component_symbol.tolist()]
        assert component_ids == ["DA", "DN", "DG"]

        target = tmp_path / "dna-with-n-output.pdb"
        write_structure(structure, target)
        lines = target.read_text(encoding="ascii").splitlines()
        assert [line for line in lines if line.startswith("SEQRES")] == ["SEQRES   1 A    3   DA   N  DG"]
        assert {line[17:20].strip() for line in lines if line.startswith("ATOM") and line[22:26].strip() == "2"} == {
            "N"
        }

        ambiguous = tmp_path / "ambiguous-n.pdb"
        ambiguous.write_text(
            "SEQRES   1 A    1    N\n"
            "ATOM      1  P     N A   1       0.000   0.000   0.000  1.00 10.00           P\n"
            "END\n",
            encoding="ascii",
        )
        with pytest.raises(ValueError, match="only ambiguous N nucleotides"):
            read_structure(ambiguous)


class TestPdbRoundTrip:
    @pytest.mark.parametrize("case", sorted(_ROUND_TRIP))
    def test_round_trip(
        self,
        case: str,
        tmp_path: Path,
    ) -> None:
        source = tmp_path / f"{case}-in.pdb"
        source.write_text(_ROUND_TRIP[case], encoding="ascii")
        structure = read_structure(source)

        target = tmp_path / f"{case}-out.pdb"
        write_structure(structure, target)
        restored = read_structure(target)

        assert restored.system.symbols == structure.system.symbols
        for field in MolecularSystem.__annotations__:
            if field != "symbols":
                assert torch.equal(getattr(restored.system, field), getattr(structure.system, field))
        assert restored.source_symbols == structure.source_symbols
        for field in Structure.__annotations__:
            if field not in {"system", "source_symbols"}:
                assert torch.equal(getattr(restored, field), getattr(structure, field))


class TestPdbWriter:
    def test_charge_provenance(self, tmp_path: Path) -> None:
        source = tmp_path / "charge-source.pdb"
        source.write_text(_POLYMER + "END\n", encoding="ascii")
        structure = read_structure(source)
        structure.source_formal_charge_mask[0] = True

        with pytest.raises(ValueError, match="cannot preserve which formal charges were explicit"):
            write_structure(structure, tmp_path / "charge.pdb")

    def test_numeric_precision(self, tmp_path: Path) -> None:
        source = tmp_path / "precision-source.pdb"
        source.write_text(_POLYMER + "END\n", encoding="ascii")
        structure = read_structure(source)

        coordinates = structure.coordinates.clone()
        coordinates[0, 0] += 0.0004
        with pytest.raises(ValueError, match="coordinates exactly at 0.001"):
            write_structure(structure.replace(coordinates=coordinates), tmp_path / "coordinates.pdb")

        occupancy = structure.occupancy.clone()
        occupancy[0] = 0.123
        with pytest.raises(ValueError, match="occupancy exactly at 0.01"):
            write_structure(structure.replace(occupancy=occupancy), tmp_path / "occupancy.pdb")

        b_factor = structure.b_factor.clone()
        b_factor[0] = 10.123
        with pytest.raises(ValueError, match="B-factor exactly at 0.01"):
            write_structure(structure.replace(b_factor=b_factor), tmp_path / "b-factor.pdb")

    def test_unknown_charge(self, tmp_path: Path) -> None:
        component = ChemicalComponent("X0", (ChemicalAtom("C", "C", formal_charge=0),), None)
        system = MolecularSystem.from_entities({"ligand": Molecule((component,))}, {"A": "ligand"})
        structure = Structure(
            system=system,
            coordinates=torch.zeros(1, 3),
            coordinate_mask=torch.ones(1, dtype=torch.bool),
            occupancy=torch.ones(1),
            occupancy_mask=torch.ones(1, dtype=torch.bool),
            b_factor=torch.zeros(1),
            b_factor_mask=torch.ones(1, dtype=torch.bool),
            source_symbols=("1",),
            residue_id=torch.tensor([0]),
            insertion_code=torch.tensor([-1]),
            batch_size=[],
        )

        with pytest.raises(ValueError, match="cannot reconstruct unknown component 'X0'"):
            write_structure(structure, tmp_path / "known-zero-charge.pdb")

    def test_no_observed_atoms(self, tmp_path: Path) -> None:
        structure = Structure.from_sequence("ACDEF").replace(
            source_symbols=("1", "2", "3", "4", "5"),
            residue_id=torch.arange(5),
        )

        with pytest.raises(ValueError, match="observes none"):
            write_structure(structure, tmp_path / "topology.pdb")

    def test_author_residue_keys(self, tmp_path: Path) -> None:
        system = MolecularSystem.from_entities(
            {"protein": PolymerSequence.from_one_letter("AG")},
            {"A": "protein"},
        )
        atom_count = system.atom_symbol.numel()
        observed = torch.zeros(atom_count, dtype=torch.bool)
        observed[int(system.component_atom_ptr[1]) :] = True
        structure = Structure(
            system=system,
            coordinates=torch.arange(atom_count * 3, dtype=torch.float32).reshape(atom_count, 3),
            coordinate_mask=observed,
            occupancy=torch.zeros(atom_count),
            occupancy_mask=torch.zeros(atom_count, dtype=torch.bool),
            b_factor=torch.zeros(atom_count),
            b_factor_mask=torch.zeros(atom_count, dtype=torch.bool),
            source_symbols=("7",),
            residue_id=torch.tensor([0, 0]),
            insertion_code=torch.tensor([-1, -1]),
            batch_size=[],
        )

        with pytest.raises(ValueError, match="repeats author residue key 7"):
            write_structure(structure, tmp_path / "collision.pdb")

        with pytest.raises(ValueError, match="cannot preserve an unknown author residue ID"):
            write_structure(
                structure.replace(
                    source_symbols=(),
                    residue_id=torch.full((2,), -1),
                ),
                tmp_path / "unknown-residue-id.pdb",
            )

        large_number = structure.replace(
            source_symbols=("10000", "10001"),
            residue_id=torch.tensor([0, 1]),
        )
        with pytest.raises(ValueError, match="official decimal range"):
            write_structure(large_number, tmp_path / "large-missing-number.pdb")

    def test_unresolved_polymer(self, tmp_path: Path) -> None:
        polymer = PolymerSequence.from_one_letter("A")
        polymer_system = MolecularSystem.from_entities(
            {"protein": polymer},
            {"A": "protein", "B": "protein"},
        )
        atom_count = polymer_system.atom_symbol.numel()
        first_stop = int(polymer_system.instance_component_ptr[1])
        first_atom_stop = int(polymer_system.component_atom_ptr[first_stop])
        polymer_structure = Structure(
            system=polymer_system,
            coordinates=torch.zeros((atom_count, 3)),
            coordinate_mask=torch.arange(atom_count) < first_atom_stop,
            occupancy=torch.zeros(atom_count),
            occupancy_mask=torch.zeros(atom_count, dtype=torch.bool),
            b_factor=torch.zeros(atom_count),
            b_factor_mask=torch.zeros(atom_count, dtype=torch.bool),
            source_symbols=("1",),
            residue_id=torch.tensor([0, 0]),
            insertion_code=torch.full((2,), -1),
            batch_size=[],
        )
        polymer_path = tmp_path / "unresolved-instance.pdb"
        write_structure(polymer_structure, polymer_path)
        polymer_lines = polymer_path.read_text(encoding="ascii").splitlines()
        assert "SEQRES   1 B    1  ALA" in polymer_lines
        assert "REMARK 465     ALA B     1" in polymer_lines
        restored_polymer = read_structure(polymer_path)
        unresolved_instance = restored_polymer.system.instance_index("B")
        unresolved_component = int(restored_polymer.system.instance_component_ptr[unresolved_instance])
        unresolved_atom_start = int(restored_polymer.system.component_atom_ptr[unresolved_component])
        unresolved_atom_stop = int(restored_polymer.system.component_atom_ptr[unresolved_component + 1])
        assert not bool(restored_polymer.coordinate_mask[unresolved_atom_start:unresolved_atom_stop].any())

    def test_unresolved_molecule(self, tmp_path: Path) -> None:
        components = (
            ChemicalComponent("ONE", (ChemicalAtom("C1", "C"),), frozenset()),
            ChemicalComponent("TWO", (ChemicalAtom("C2", "C"),), frozenset()),
        )
        molecule_system = MolecularSystem.from_entities(
            {"ligand": Molecule(components)},
            {"L": "ligand"},
        )
        molecule_structure = Structure(
            system=molecule_system,
            coordinates=torch.zeros((2, 3)),
            coordinate_mask=torch.tensor([True, False]),
            occupancy=torch.zeros(2),
            occupancy_mask=torch.zeros(2, dtype=torch.bool),
            b_factor=torch.zeros(2),
            b_factor_mask=torch.zeros(2, dtype=torch.bool),
            source_symbols=("1", "2"),
            residue_id=torch.tensor([0, 1]),
            insertion_code=torch.full((2,), -1),
            batch_size=[],
        )
        with pytest.raises(ValueError, match=r"unresolved non-polymer component 'L'\[1\]"):
            write_structure(molecule_structure, tmp_path / "unresolved-component.pdb")

    def test_molecule_metadata(self, tmp_path: Path) -> None:
        component = ChemicalComponent("LIG", (ChemicalAtom("C", "C"),), bonds=None)
        water = MolecularSystem.from_entities(
            {"ligand": Molecule((component,), molecule_type="water")},
            {"L": "ligand"},
        )
        atom_count = water.atom_symbol.numel()
        structure = Structure(
            system=water,
            coordinates=torch.zeros((atom_count, 3)),
            coordinate_mask=torch.ones(atom_count, dtype=torch.bool),
            occupancy=torch.zeros(atom_count),
            occupancy_mask=torch.zeros(atom_count, dtype=torch.bool),
            b_factor=torch.zeros(atom_count),
            b_factor_mask=torch.zeros(atom_count, dtype=torch.bool),
            source_symbols=("1",),
            residue_id=torch.tensor([0]),
            insertion_code=torch.full((1,), -1),
            batch_size=[],
        )
        with pytest.raises(ValueError, match="keep its grouping and type"):
            write_structure(structure, tmp_path / "wrong-water-type.pdb")

        shared = MolecularSystem.from_entities(
            {"ligand": Molecule((component,))},
            {"L1": "ligand", "L2": "ligand"},
        )
        shared_atom_count = shared.atom_symbol.numel()
        repeated = structure.replace(
            system=shared,
            coordinates=torch.zeros((shared_atom_count, 3)),
            coordinate_mask=torch.ones(shared_atom_count, dtype=torch.bool),
            occupancy=torch.zeros(shared_atom_count),
            occupancy_mask=torch.zeros(shared_atom_count, dtype=torch.bool),
            b_factor=torch.zeros(shared_atom_count),
            b_factor_mask=torch.zeros(shared_atom_count, dtype=torch.bool),
            source_formal_charge_mask=torch.zeros(shared_atom_count, dtype=torch.bool),
            residue_id=torch.tensor([0, 0]),
            insertion_code=torch.full((2,), -1),
            component_parent_ptr=torch.zeros(3, dtype=torch.int64),
            parent_component_id=torch.empty(0, dtype=torch.int64),
        )
        shared_path = tmp_path / "shared-ligand.pdb"
        write_structure(repeated, shared_path)
        restored = read_structure(shared_path)
        assert restored.system.entity_symbol.numel() == 1
        assert restored.system.instance_entity.tolist() == [0, 0]

        distinct = MolecularSystem.from_entities(
            {"first": Molecule((component,)), "second": Molecule((component,))},
            {"L1": "first", "L2": "second"},
        )
        with pytest.raises(ValueError, match="map one-to-one with component IDs"):
            write_structure(repeated.replace(system=distinct), tmp_path / "ambiguous-ligands.pdb")

    def test_shared_entities(self, tmp_path: Path) -> None:
        alanine = PolymerSequence.from_one_letter("A")
        glycine = PolymerSequence.from_one_letter("G")
        system = MolecularSystem.from_entities(
            {"protein": alanine},
            {"A": "protein", "B": "protein"},
            instance_values={"B": glycine},
        )
        atom_count = system.atom_symbol.numel()
        structure = Structure(
            system=system,
            coordinates=torch.zeros((atom_count, 3)),
            coordinate_mask=torch.ones(atom_count, dtype=torch.bool),
            occupancy=torch.zeros(atom_count),
            occupancy_mask=torch.zeros(atom_count, dtype=torch.bool),
            b_factor=torch.zeros(atom_count),
            b_factor_mask=torch.zeros(atom_count, dtype=torch.bool),
            source_symbols=("1",),
            residue_id=torch.tensor([0, 0]),
            insertion_code=torch.full((2,), -1),
            batch_size=[],
        )

        path = tmp_path / "shared-entity.pdb"
        write_structure(structure, path)
        restored = read_structure(path)

        assert restored.system.entity_symbol.numel() == 1
        assert restored.system.instance_entity.tolist() == [0, 0]
        assert [restored.system.symbols[int(symbol)] for symbol in restored.system.component_symbol] == ["ALA", "GLY"]

    @pytest.mark.parametrize(
        ("component_id", "observed", "message"),
        (
            ("XYZ", False, "nonstandard polymer component"),
            ("XYZ", True, "nonstandard polymer component"),
            ("ALA", False, "does not match its authoritative definition"),
            ("ALA", True, "does not match its authoritative definition"),
        ),
    )
    def test_non_authoritative_component(
        self,
        component_id: str,
        observed: bool,
        message: str,
        tmp_path: Path,
    ) -> None:
        standard = PolymerSequence.from_one_letter("A").components[0]
        custom = ChemicalComponent(component_id, (ChemicalAtom("X", "C"),), frozenset())
        system = MolecularSystem.from_entities(
            {"protein": PolymerSequence("protein", (standard, custom), linked=(False,))},
            {"A": "protein"},
        )
        atom_count = system.atom_symbol.numel()
        first_stop = int(system.component_atom_ptr[1])
        structure = Structure(
            system=system,
            coordinates=torch.zeros((atom_count, 3)),
            coordinate_mask=(
                torch.ones(atom_count, dtype=torch.bool) if observed else torch.arange(atom_count) < first_stop
            ),
            occupancy=torch.zeros(atom_count),
            occupancy_mask=torch.zeros(atom_count, dtype=torch.bool),
            b_factor=torch.zeros(atom_count),
            b_factor_mask=torch.zeros(atom_count, dtype=torch.bool),
            source_symbols=("1", "2"),
            residue_id=torch.tensor([0, 1]),
            insertion_code=torch.full((2,), -1),
            batch_size=[],
        )

        with pytest.raises(ValueError, match=message):
            write_structure(structure, tmp_path / f"{component_id}-{observed}.pdb")

    def test_polymer_bond_chemistry(self, tmp_path: Path) -> None:
        linked = Structure.from_sequence("AA")
        linked = linked.replace(
            coordinates=torch.zeros_like(linked.coordinates),
            coordinate_mask=torch.ones_like(linked.coordinate_mask),
            source_symbols=("1", "2"),
            residue_id=torch.tensor([0, 1]),
        )
        polymer_edge = int(torch.nonzero(linked.system.bond_role == BOND_ROLE_POLYMER)[0].item())
        bond_order = linked.system.bond_order.clone()
        bond_order[polymer_edge] = 2.0
        linked = linked.replace(system=linked.system.replace(bond_order=bond_order))
        with pytest.raises(ValueError, match="noncanonical polymer edge chemistry"):
            write_structure(linked, tmp_path / "noncanonical-edge.pdb")

    def test_polymer_topology(self, tmp_path: Path) -> None:
        components = PolymerSequence.from_one_letter("AA").components
        nick_system = MolecularSystem.from_entities(
            {"protein": PolymerSequence("protein", components, linked=(False,))},
            {"A": "protein"},
        )
        atom_count = nick_system.atom_symbol.numel()
        visible = torch.tensor([nick_system.symbols[symbol] == "CA" for symbol in nick_system.atom_symbol.tolist()])
        nick = Structure(
            system=nick_system,
            coordinates=torch.zeros((atom_count, 3)),
            coordinate_mask=visible,
            occupancy=torch.zeros(atom_count),
            occupancy_mask=torch.zeros(atom_count, dtype=torch.bool),
            b_factor=torch.zeros(atom_count),
            b_factor_mask=torch.zeros(atom_count, dtype=torch.bool),
            source_symbols=("1", "2"),
            residue_id=torch.tensor([0, 1]),
            insertion_code=torch.full((2,), -1),
            batch_size=[],
        )
        with pytest.raises(ValueError, match="invisible polymer nick"):
            write_structure(nick, tmp_path / "invisible-nick.pdb")

        heavy_component = components[0].without_hydrogens()
        heavy_system = MolecularSystem.from_entities(
            {"protein": PolymerSequence("protein", (heavy_component,))},
            {"A": "protein"},
        )
        heavy_atom_count = heavy_system.atom_symbol.numel()
        heavy = nick.replace(
            system=heavy_system,
            coordinates=torch.zeros((heavy_atom_count, 3)),
            coordinate_mask=torch.ones(heavy_atom_count, dtype=torch.bool),
            occupancy=torch.zeros(heavy_atom_count),
            occupancy_mask=torch.zeros(heavy_atom_count, dtype=torch.bool),
            b_factor=torch.zeros(heavy_atom_count),
            b_factor_mask=torch.zeros(heavy_atom_count, dtype=torch.bool),
            source_formal_charge_mask=torch.zeros(heavy_atom_count, dtype=torch.bool),
            residue_id=torch.tensor([0]),
            insertion_code=torch.full((1,), -1),
            component_parent_ptr=torch.zeros(2, dtype=torch.int64),
            parent_component_id=torch.empty(0, dtype=torch.int64),
        )
        with pytest.raises(ValueError, match="does not match its authoritative definition"):
            write_structure(heavy, tmp_path / "heavy-only.pdb")


class TestPdbConnections:
    def test_het_conect(self, tmp_path: Path) -> None:
        alanine = PolymerSequence.from_one_letter("A").components[0]
        system = MolecularSystem.from_entities(
            {"alanine": Molecule((alanine,))},
            {"L": "alanine"},
        )
        atom_count = system.atom_symbol.numel()
        structure = Structure(
            system=system,
            coordinates=torch.zeros((atom_count, 3)),
            coordinate_mask=torch.ones(atom_count, dtype=torch.bool),
            occupancy=torch.zeros(atom_count),
            occupancy_mask=torch.zeros(atom_count, dtype=torch.bool),
            b_factor=torch.zeros(atom_count),
            b_factor_mask=torch.zeros(atom_count, dtype=torch.bool),
            source_symbols=("1",),
            residue_id=torch.tensor([0]),
            insertion_code=torch.full((1,), -1),
            batch_size=[],
        )
        target = tmp_path / "free-alanine.pdb"

        write_structure(structure, target)
        lines = target.read_text(encoding="ascii").splitlines()
        assert all(line.startswith("HETATM") for line in lines if line.startswith(("ATOM", "HETATM")))
        directed = {
            (int(line[6:11]), int(line[start : start + 5]))
            for line in lines
            if line.startswith("CONECT")
            for start in (11, 16, 21, 26)
            if line[start : start + 5].strip()
        }
        component_edge_count = int((system.bond_role == BOND_ROLE_COMPONENT).sum())
        assert len(directed) == 2 * component_edge_count
        assert all((target_serial, source_serial) in directed for source_serial, target_serial in directed)

    def test_disulfide(self, tmp_path: Path) -> None:
        source = tmp_path / "disulfide.pdb"
        source.write_text(
            "SSBOND   1 CYS A    1    CYS B    2                          1555   1555  2.03\n"
            "ATOM      1  SG  CYS A   1       0.000   0.000   0.000  1.00 10.00           S\n"
            "ATOM      2  SG  CYS B   2       2.030   0.000   0.000  1.00 10.00           S\n"
            "END\n",
            encoding="ascii",
        )

        structure = read_structure(source)
        edge = int(torch.nonzero(structure.system.bond_role == BOND_ROLE_ASSEMBLY).item())
        assert bool(structure.system.bond_order_known[edge])
        assert structure.system.bond_order[edge].item() == 1.0
        assert bool(structure.system.bond_aromatic_known[edge])
        assert not bool(structure.system.bond_aromatic[edge])
        assert "HG" not in {structure.system.symbols[int(symbol)] for symbol in structure.system.atom_symbol.tolist()}

        target = tmp_path / "disulfide-output.pdb"
        write_structure(structure, target)
        lines = target.read_text(encoding="ascii").splitlines()
        assert [line for line in lines if line.startswith("SSBOND")] == [
            "SSBOND   1 CYS A    1    CYS B    2                          1555   1555  2.03"
        ]
        assert [line for line in lines if line.startswith("CONECT")] == [
            "CONECT    1    3",
            "CONECT    3    1",
        ]

        restored = read_structure(target)
        restored_edge = int(torch.nonzero(restored.system.bond_role == BOND_ROLE_ASSEMBLY).item())
        assert bool(restored.system.bond_order_known[restored_edge])
        assert restored.system.bond_order[restored_edge].item() == 1.0
        assert bool(restored.system.bond_aromatic_known[restored_edge])
        assert not bool(restored.system.bond_aromatic[restored_edge])
        assert "HG" not in {restored.system.symbols[int(symbol)] for symbol in restored.system.atom_symbol.tolist()}

    def test_missing_disulfide_endpoint(self, tmp_path: Path) -> None:
        source = tmp_path / "missing-disulfide.pdb"
        source.write_text(
            "SSBOND   1 CYS A    9    CYS B   10                          1555   1555  2.03\n"
            "ATOM      1  SG  CYS A   1       0.000   0.000   0.000  1.00 10.00           S\n"
            "ATOM      2  SG  CYS B   2       2.030   0.000   0.000  1.00 10.00           S\n"
            "END\n",
            encoding="ascii",
        )

        with pytest.raises(ValueError, match="SSBOND endpoint identity is missing"):
            read_structure(source)

    def test_crystallographic_disulfide(self, tmp_path: Path) -> None:
        ssbond = "SSBOND   1 CYS A    1    CYS B    2                          1555   1555"
        body = (
            "CRYST1   10.000   20.000   30.000  90.00  90.00  90.00 P 1           1\n"
            "SCALE1      0.100000  0.020000  0.000000        0.00000\n"
            "SCALE2      0.000000  0.050000  0.000000        0.00000\n"
            "SCALE3      0.000000  0.000000  0.033333        0.00000\n"
            "REMARK 290   SMTRY1   2  1.000000  0.000000  0.000000       10.00000\n"
            "REMARK 290   SMTRY2   2  0.000000  1.000000  0.000000        0.00000\n"
            "REMARK 290   SMTRY3   2  0.000000  0.000000  1.000000        0.00000\n"
            "SEQRES   1 A    1  CYS\n"
            "SEQRES   1 B    1  CYS\n"
            "ATOM      1  CA  CYS A   1       0.000   0.000   0.000  1.00 10.00           C\n"
            "TER       2      CYS A   1\n"
            "ATOM      3  CA  CYS B   2       4.000   0.000   0.000  1.00 10.00           C\n"
            "TER       4      CYS B   2\n"
            "END\n"
        )
        source = tmp_path / "unobserved-disulfide.pdb"
        source.write_text(ssbond + "\n" + body, encoding="ascii")

        structure = read_structure(source)
        edge = int(torch.nonzero(structure.system.bond_role == BOND_ROLE_ASSEMBLY).item())
        endpoint_atoms = {
            structure.system.symbols[int(structure.system.atom_symbol[int(structure.system.edge_index[row, edge])])]
            for row in (0, 1)
        }
        assert endpoint_atoms == {"SG"}

        target = tmp_path / "unobserved-disulfide-output.pdb"
        write_structure(structure, target)
        lines = target.read_text(encoding="ascii").splitlines()
        written_ssbond = next(line for line in lines if line.startswith("SSBOND"))
        assert written_ssbond[73:78] == " " * 5
        assert not any(line.startswith("CONECT") for line in lines)

        nonidentity = ssbond.ljust(80)
        nonidentity = nonidentity[:59] + f"{2565:>6}" + nonidentity[65:]
        crystal = tmp_path / "nonidentity-disulfide.pdb"
        crystal.write_text(nonidentity + "\n" + body, encoding="ascii")
        materialized = read_structure(crystal)
        assert materialized.system.instance_symbol.numel() == 3
        assert int((materialized.system.bond_role == BOND_ROLE_ASSEMBLY).sum()) == 1
        assert materialized.coordinates[materialized.coordinate_mask, 0].tolist() == [0.0, 4.0, 6.0]
        assert materialized.coordinates[materialized.coordinate_mask, 1].tolist() == [0.0, 0.0, 20.0]
        for instance in ("A", "B", "A#2565"):
            names = {
                materialized.system.atom_site(i).atom_id
                for i in range(materialized.system.atom_symbol.numel())
                if materialized.system.atom_site(i).instance_id == instance
            }
            assert "SG" in names
            assert "HG" not in names

    def test_nmr_conect(self, tmp_path: Path) -> None:
        source = tmp_path / "nmr-conect.pdb"
        source.write_text(
            "MODEL        1\n"
            "HETATM    1  C1  LIG A   1       0.000   0.000   0.000  1.00 10.00           C\n"
            "HETATM    2  C2  LIG A   1       1.000   0.000   0.000  1.00 10.00           C\n"
            "HETATM    3  X1  OTH A   2       2.000   0.000   0.000  1.00 10.00           C\n"
            "ENDMDL\n"
            "MODEL        2\n"
            "HETATM   11  C1  LIG A   1      10.000   0.000   0.000  1.00 10.00           C\n"
            "HETATM   12  C2  LIG A   1      11.000   0.000   0.000  1.00 10.00           C\n"
            "HETATM   13  X1  OTH A   2      12.000   0.000   0.000  1.00 10.00           C\n"
            "ENDMDL\n"
            "LINK         C2  LIG A   1                 X1  OTH A   2\n"
            "CONECT    1    2\n"
            "CONECT    2    3\n"
            "END\n",
            encoding="ascii",
        )

        structure = read_structure(source, model_index=2)

        assert int((structure.system.bond_role == BOND_ROLE_COMPONENT).sum()) == 1
        assert int((structure.system.bond_role == BOND_ROLE_ASSEMBLY).sum()) == 1
        assert structure.coordinates[structure.coordinate_mask, 0].tolist() == [10.0, 11.0, 12.0]

    def test_conect_columns(self, tmp_path: Path) -> None:
        source = tmp_path / "conect.pdb"
        link = "LINK         C2  LIG A   1                 X1  OTH A   2"
        source.write_text(
            link + "\nHETATM    1  C1  LIG A   1       0.000   0.000   0.000  1.00 10.00           C\n"
            "HETATM    2  C2  LIG A   1       1.000   0.000   0.000  1.00 10.00           C\n"
            "HETATM    3  X1  OTH A   2       2.000   0.000   0.000  1.00 10.00           C\n"
            "HETATM    4  X2  IGN A   3       3.000   0.000   0.000  1.00 10.00           C\n"
            f"CONECT{1:>5}{2:>5}{'':>5}{'':>5}{'':>5}{4:>5}\n"
            "CONECT    2    3\n"
            "END\n",
            encoding="ascii",
        )

        structure = read_structure(source)
        component_edges = torch.nonzero(structure.system.bond_role == BOND_ROLE_COMPONENT).flatten()
        assembly_edges = torch.nonzero(structure.system.bond_role == BOND_ROLE_ASSEMBLY).flatten()
        assert component_edges.numel() == 1
        assert assembly_edges.numel() == 1
        ligand_component = next(
            index
            for index in range(structure.system.component_symbol.numel())
            if structure.system.symbols[int(structure.system.component_symbol[index])] == "LIG"
        )
        assert not bool(structure.system.component_topology_complete[ligand_component])

        for edge in (int(component_edges[0]), int(assembly_edges[0])):
            assert not bool(structure.system.bond_order_known[edge])
            assert not bool(structure.system.bond_aromatic_known[edge])
        edge = int(assembly_edges[0])

        target = tmp_path / "conect-output.pdb"
        write_structure(structure, target)
        target_lines = target.read_text(encoding="ascii").splitlines()
        written_link = next(line for line in target_lines if line.startswith("LINK"))
        assert (written_link[12:16].strip(), written_link[42:46].strip()) == ("C2", "X1")
        assert [line for line in target_lines if line.startswith("CONECT")] == [
            "CONECT    1    2",
            "CONECT    2    1    3",
            "CONECT    3    2",
        ]
        restored = read_structure(target)
        assert int((restored.system.bond_role == BOND_ROLE_ASSEMBLY).sum()) == 1
        assert int((restored.system.bond_role == BOND_ROLE_COMPONENT).sum()) == 1
        assert not bool(restored.system.component_topology_complete[ligand_component])

    def test_ambiguous_conect(self, tmp_path: Path) -> None:
        source = tmp_path / "conect.pdb"
        source.write_text(
            "HETATM    1  C1  LIG A   1       0.000   0.000   0.000  1.00 10.00           C\n"
            "HETATM    2  C2  LIG A   1       1.000   0.000   0.000  1.00 10.00           C\n"
            "HETATM    3  X1  OTH A   2       2.000   0.000   0.000  1.00 10.00           C\n"
            "CONECT    1    2\n"
            "CONECT    2    3\n"
            "END\n",
            encoding="ascii",
        )

        with pytest.raises(ValueError, match="does not distinguish"):
            read_structure(source)

    def test_known_conect_order(self, tmp_path: Path) -> None:
        source = tmp_path / "conect.pdb"
        source.write_text(
            "LINK         C2  LIG A   1                 X1  OTH A   2\n"
            "HETATM    1  C1  LIG A   1       0.000   0.000   0.000  1.00 10.00           C\n"
            "HETATM    2  C2  LIG A   1       1.000   0.000   0.000  1.00 10.00           C\n"
            "HETATM    3  X1  OTH A   2       2.000   0.000   0.000  1.00 10.00           C\n"
            "CONECT    1    2\n"
            "CONECT    2    3\n"
            "END\n",
            encoding="ascii",
        )
        structure = read_structure(source)
        edge = int(torch.nonzero(structure.system.bond_role == BOND_ROLE_ASSEMBLY)[0])

        bond_order_known = structure.system.bond_order_known.clone()
        bond_order_known[edge] = True
        with pytest.raises(ValueError, match="chemically unspecified nonmetal covalent links"):
            write_structure(
                structure.replace(system=structure.system.replace(bond_order_known=bond_order_known)),
                tmp_path / "known-order.pdb",
            )

    def test_source_bonds(self, tmp_path: Path) -> None:
        source = tmp_path / "completed-conect.pdb"
        source.write_text(
            "HETATM    1  C1  LIG A   1       0.000   0.000   0.000  1.00 10.00           C\n"
            "HETATM    2  C2  LIG A   1       1.000   0.000   0.000  1.00 10.00           C\n"
            "HETATM    3  O1  LIG A   1       2.000   0.000   0.000  1.00 10.00           O\n"
            "CONECT    2    3\n"
            "END\n",
            encoding="ascii",
        )
        component = ChemicalComponent(
            "LIG",
            (ChemicalAtom("C1", "C"), ChemicalAtom("C2", "C"), ChemicalAtom("O1", "O")),
            frozenset({ChemicalBond("C1", "C2", order=1.0, aromatic=False)}),
        )

        structure = read_structure(source, components={"LIG": component})
        edges = {
            frozenset(
                structure.system.symbols[
                    int(structure.system.atom_symbol[int(structure.system.edge_index[side, edge])])
                ]
                for side in (0, 1)
            ): bool(structure.system.bond_order_known[edge])
            for edge in torch.nonzero(structure.system.bond_role == BOND_ROLE_COMPONENT).flatten().tolist()
        }

        assert edges == {frozenset({"C1", "C2"}): True, frozenset({"C2", "O1"}): False}

    def test_crystallographic_link(self, tmp_path: Path) -> None:
        link = "LINK         SG  CYS A   1                 C1  LIG B   1"
        body = (
            "REMARK 290   SMTRY1   2  1.000000  0.000000  0.000000       10.00000\n"
            "REMARK 290   SMTRY2   2  0.000000  1.000000  0.000000        0.00000\n"
            "REMARK 290   SMTRY3   2  0.000000  0.000000  1.000000        0.00000\n"
            "SEQRES   1 A    1  CYS\n"
            "ATOM      1  CA  CYS A   1       0.000   0.000   0.000  1.00 10.00           C\n"
            "TER       2      CYS A   1\n"
            "HETATM    3  C1  LIG B   1       2.000   0.000   0.000  1.00 10.00           C\n"
            "HETATM    4  X1  OTH C   1       4.000   0.000   0.000  1.00 10.00           C\n"
            "HETATM    5  Y1  EXT D   1       6.000   0.000   0.000  1.00 10.00           C\n"
            "HETATM    6  Z1  IGN E   1       8.000   0.000   0.000  1.00 10.00           C\n"
            "END\n"
        )
        source = tmp_path / "unobserved-link.pdb"
        source.write_text(link + "\n" + body, encoding="ascii")

        structure = read_structure(source)
        edge = int(torch.nonzero(structure.system.bond_role == BOND_ROLE_ASSEMBLY).item())
        endpoint_atoms = {
            structure.system.symbols[int(structure.system.atom_symbol[int(structure.system.edge_index[row, edge])])]
            for row in (0, 1)
        }
        assert endpoint_atoms == {"SG", "C1"}

        target = tmp_path / "unobserved-link-output.pdb"
        write_structure(structure, target)
        written_link = next(line for line in target.read_text(encoding="ascii").splitlines() if line.startswith("LINK"))
        assert written_link[73:78] == " " * 5
        assert not any(line.startswith("CONECT") for line in target.read_text(encoding="ascii").splitlines())

        nonidentity = link.ljust(78)
        nonidentity = nonidentity[:59] + f"{2555:>6}" + nonidentity[65:]
        attached = "LINK         CA  CYS A   1                 X1  OTH C   1"
        transitive = "LINK         X1  OTH C   1                 Y1  EXT D   1"
        crystal = tmp_path / "nonidentity-link.pdb"
        crystal.write_text("\n".join((nonidentity, attached, transitive)) + "\n" + body, encoding="ascii")
        materialized = read_structure(crystal)
        assert materialized.system.instance_symbol.numel() == 8
        assert int((materialized.system.bond_role == BOND_ROLE_ASSEMBLY).sum()) == 5
        assert materialized.coordinates[materialized.coordinate_mask, 0].tolist() == [
            0.0,
            2.0,
            4.0,
            6.0,
            8.0,
            10.0,
            14.0,
            16.0,
        ]
        native = read_source(crystal)
        for chain, expected_x in (("A", [0.0, 10.0]), ("B", [2.0])):
            selected = read_structure(native.view(asym_ids=frozenset({chain})))
            assert selected.system.instance_symbol.numel() == len(expected_x)
            assert not bool((selected.system.bond_role == BOND_ROLE_ASSEMBLY).any())
            assert selected.coordinates[selected.coordinate_mask, 0].tolist() == expected_x
            if chain == "A":
                assert (
                    sum(
                        selected.system.atom_site(i).atom_id == "HG" for i in range(selected.system.atom_symbol.numel())
                    )
                    == 2
                )

        excluded_altloc = nonidentity[:16] + "B" + nonidentity[17:]
        altloc_body = (
            "REMARK 290   SMTRY1   2  1.000000  0.000000  0.000000       10.00000\n"
            "REMARK 290   SMTRY2   2  0.000000  1.000000  0.000000        0.00000\n"
            "REMARK 290   SMTRY3   2  0.000000  0.000000  1.000000        0.00000\n"
            "SEQRES   1 A    1  CYS\n"
            "ATOM      1  CA ACYS A   1       0.000   0.000   0.000  0.80 10.00           C\n"
            "ATOM      2  SG BCYS A   1       1.000   0.000   0.000  0.20 10.00           S\n"
            "TER       3      CYS A   1\n"
            "HETATM    4  C1  LIG B   1       2.000   0.000   0.000  1.00 10.00           C\n"
            "END\n"
        )
        excluded = tmp_path / "excluded-altloc-crystallographic-link.pdb"
        excluded.write_text(excluded_altloc + "\n" + altloc_body, encoding="ascii")
        without_copy = read_structure(excluded)
        assert without_copy.system.instance_symbol.numel() == 2
        assert not bool((without_copy.system.bond_role == BOND_ROLE_ASSEMBLY).any())
        assert without_copy.coordinates[without_copy.coordinate_mask, 0].tolist() == [0.0, 2.0]

        hydrogen_link = "LINK         HG  CYS A   1                 C1  LIG B   1".ljust(78)
        hydrogen_link = hydrogen_link[:59] + f"{2555:>6}" + hydrogen_link[65:]
        excluded_hydrogen = tmp_path / "excluded-hydrogen-crystallographic-link.pdb"
        excluded_hydrogen.write_text(hydrogen_link + "\n" + body, encoding="ascii")
        with pytest.raises(ValueError, match="cannot suppress a hydrogen connected by a non-single bond"):
            read_structure(excluded_hydrogen, include_hydrogens=False)

    def test_metal_link(self, tmp_path: Path) -> None:
        link = "LINK        ZN    ZN A   1                 O   HOH W   1                  2.00"
        zinc = "HETATM    1 ZN    ZN A   1       0.000   0.000   0.000  1.00 10.00          ZN\n"
        water = "HETATM    2  O   HOH W   1       2.000   0.000   0.000  1.00 10.00           O\n"
        source = tmp_path / "metal-water-link.pdb"
        source.write_text(link + "\n" + zinc + water + "END\n", encoding="ascii")

        structure = read_structure(source)
        edge = int(torch.nonzero(structure.system.bond_role == BOND_ROLE_ASSEMBLY).item())
        assert int(structure.system.bond_kind[edge]) == BOND_KIND_DATIVE
        atom_a = int(structure.system.edge_index[0, edge])
        atom_b = int(structure.system.edge_index[1, edge])
        direction = int(structure.system.bond_direction[edge])
        donor, metal = (atom_a, atom_b) if direction == BOND_DIRECTION_ENDPOINT_0_DONOR else (atom_b, atom_a)
        assert direction in {BOND_DIRECTION_ENDPOINT_0_DONOR, BOND_DIRECTION_ENDPOINT_1_DONOR}
        assert int(structure.system.atomic_number[donor]) == ATOMIC_NUMBER_BY_ELEMENT["O"]
        assert int(structure.system.atomic_number[metal]) == ATOMIC_NUMBER_BY_ELEMENT["Zn"]

        target = tmp_path / "metal-water-link-output.pdb"
        write_structure(structure, target)
        restored = read_structure(target)
        restored_edge = int(torch.nonzero(restored.system.bond_role == BOND_ROLE_ASSEMBLY).item())
        assert int(restored.system.bond_kind[restored_edge]) == BOND_KIND_DATIVE
        assert int(restored.system.bond_direction[restored_edge]) == direction

        without_water = read_structure(source, include_water=False)
        assert not bool((without_water.system.bond_role == BOND_ROLE_ASSEMBLY).any())

        nonidentity_water = link.ljust(80)
        nonidentity_water = nonidentity_water[:66] + f"{2555:>6}" + nonidentity_water[72:]
        water_to_ligand = "LINK         O   HOH W   1                 C1  LIG C   1"
        crystal = tmp_path / "excluded-water-crystallographic-link.pdb"
        crystal.write_text(
            nonidentity_water
            + "\n"
            + water_to_ligand
            + "\nREMARK 290   SMTRY1   2  1.000000  0.000000  0.000000       10.00000\n"
            + "REMARK 290   SMTRY2   2  0.000000  1.000000  0.000000        0.00000\n"
            + "REMARK 290   SMTRY3   2  0.000000  0.000000  1.000000        0.00000\n"
            + zinc
            + water
            + "HETATM    3  C1  LIG C   1       4.000   0.000   0.000  1.00 10.00           C\n"
            + "END\n",
            encoding="ascii",
        )
        without_water_copy = read_structure(crystal, include_water=False)
        assert without_water_copy.system.instance_symbol.numel() == 2
        assert not bool((without_water_copy.system.bond_role == BOND_ROLE_ASSEMBLY).any())
        assert without_water_copy.coordinates[without_water_copy.coordinate_mask, 0].tolist() == [0.0, 4.0]

        missing = tmp_path / "missing-water-link.pdb"
        missing.write_text(link + "\n" + zinc + "END\n", encoding="ascii")
        with pytest.raises(ValueError, match="endpoint identity is missing"):
            read_structure(missing, include_water=False)

        metal_pair = tmp_path / "indeterminate-metal-link.pdb"
        metal_pair.write_text(
            "LINK        ZN    ZN A   1                FE    FE B   1                  2.00\n"
            + zinc
            + "HETATM    2 FE    FE B   1       2.000   0.000   0.000  1.00 10.00          FE\n"
            + "END\n",
            encoding="ascii",
        )
        with pytest.raises(ValueError, match="indeterminate between two metal endpoints"):
            read_structure(metal_pair)

    def test_altloc(self, tmp_path: Path) -> None:
        source = tmp_path / "conect-altloc.pdb"
        source.write_text(
            "HETATM    1  C1 ALIG A   1       0.000   0.000   0.000  0.30 10.00           C\n"
            "HETATM    2  C1 BLIG A   1       0.000   1.000   0.000  0.70 10.00           C\n"
            "HETATM    3  C2 ALIG A   1       1.000   0.000   0.000  0.30 10.00           C\n"
            "HETATM    4  C2 BLIG A   1       1.000   1.000   0.000  0.70 10.00           C\n"
            "CONECT    2    4\n"
            "END\n",
            encoding="ascii",
        )

        selected_b = read_structure(source)
        selected_a = read_structure(source, altloc="first")

        assert int((selected_b.system.bond_role == BOND_ROLE_COMPONENT).sum()) == 1
        assert not bool((selected_a.system.bond_role == BOND_ROLE_COMPONENT).any())


def test_observed_polymer_atoms(tmp_path: Path) -> None:
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
    restored = read_structure(path)

    assert restored.system.instance_id(0) == "A"
    torch.testing.assert_close(restored.coordinates, structure.coordinates.double(), rtol=0, atol=0)
    assert "ATOM" in path.read_text(encoding="ascii")
