from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from ofoldx.data import (
    ChemicalAtom,
    ChemicalBond,
    ChemicalComponent,
    MolecularSystem,
    Molecule,
    MoleculeBond,
    Structure,
    form_bond,
    load_structure,
)
from ofoldx.data.ccd import prepare_structure
from ofoldx.io import read_structure, write_structure

_LIGAND_PDB = """\
HETATM    1  C1  LIG A   1       0.000   0.000   0.000  1.00 10.00           C
HETATM    2  O1  LIG A   1       1.000   0.000   0.000  1.00 10.00           O
END
"""


def _snapshot(path: Path, component_id: str = "LIG") -> Path:
    path.mkdir()
    row = {
        "component_id": component_id,
        "chem_type": "non-polymer",
        "atom_ids": ["C1", "O1"],
        "elements": ["C", "O"],
        "formal_charges": [0, 0],
        "aromatic": [False, False],
        "stereo": [None, None],
        "bond_atom_a": ["C1"],
        "bond_atom_b": ["O1"],
        "bond_orders": [1.0],
        "bond_aromatic": [False],
        "bond_stereo": [None],
        "reference_coordinates": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        "leaving_atom_ids": ["O1"],
        "non_leaving_atom_ids": ["C1"],
        "previous_attachment_atom": None,
        "previous_leaving_atoms": [],
        "next_attachment_atom": None,
        "next_leaving_atoms": [],
    }
    pq.write_table(pa.Table.from_pylist([row]), path / "components.parquet")
    return path


def _component_row(
    component_id: str,
    atoms: list[tuple[str, str]],
    bonds: list[tuple[str, str]],
) -> dict[str, object]:
    atom_ids = [atom_id for atom_id, _ in atoms]
    return {
        "component_id": component_id,
        "chem_type": "non-polymer",
        "atom_ids": atom_ids,
        "elements": [element for _, element in atoms],
        "formal_charges": [0] * len(atoms),
        "aromatic": [False] * len(atoms),
        "stereo": [None] * len(atoms),
        "bond_atom_a": [first for first, _ in bonds],
        "bond_atom_b": [second for _, second in bonds],
        "bond_orders": [1.0] * len(bonds),
        "bond_aromatic": [False] * len(bonds),
        "bond_stereo": [None] * len(bonds),
        "reference_coordinates": [None] * len(atoms),
        "leaving_atom_ids": [],
        "non_leaving_atom_ids": atom_ids,
        "previous_attachment_atom": None,
        "previous_leaving_atoms": [],
        "next_attachment_atom": None,
        "next_leaving_atoms": [],
    }


def _chemistry_snapshot(path: Path, *rows: dict[str, object]) -> Path:
    path.mkdir()
    pq.write_table(pa.Table.from_pylist(rows), path / "components.parquet")
    return path


def _crosslinked_source(path: Path) -> None:
    ligand = ChemicalComponent(
        "LIG",
        (ChemicalAtom("C1", "C"), ChemicalAtom("O1", "O")),
        frozenset({ChemicalBond("C1", "O1")}),
    )
    partner = ChemicalComponent("OTH", (ChemicalAtom("X1", "C"),), frozenset())
    system = MolecularSystem.from_entities(
        {"ligand": Molecule((ligand,)), "partner": Molecule((partner,))},
        {"A": "ligand", "B": "partner"},
    )
    system, _ = form_bond(
        system,
        1,
        2,
        torch.empty(0, dtype=torch.int64),
        torch.empty(0, dtype=torch.int64),
    )
    write_structure(
        Structure(
            system=system,
            coordinates=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            coordinate_mask=torch.ones(3, dtype=torch.bool),
            occupancy=torch.ones(3),
            occupancy_mask=torch.ones(3, dtype=torch.bool),
            b_factor=torch.zeros(3),
            b_factor_mask=torch.ones(3, dtype=torch.bool),
            source_symbols=("1", "2"),
            residue_id=torch.tensor([0, 1]),
            insertion_code=torch.full((2,), -1),
            standard_component_id=torch.full((2,), -1),
            batch_size=[],
        ),
        path,
    )


class TestPrepareStructure:
    def test_local_snapshot(self, tmp_path: Path) -> None:
        structure_path = tmp_path / "ligand.pdb"
        structure_path.write_text(_LIGAND_PDB.replace("           O\n", "           O1-\n"), encoding="ascii")
        snapshot = _snapshot(tmp_path / "ccd")
        destination = tmp_path / "ligand.structure.safetensors"

        resolved = prepare_structure(
            structure_path,
            destination,
            snapshot_path=snapshot / "components.parquet",
        )
        restored = load_structure(destination)

        assert resolved.system.reference_coordinate_mask.tolist() == [True, True]
        assert resolved.system.edge_index.shape[1] == 1
        assert resolved.system.formal_charge.tolist() == [0, -1]
        assert restored.system.symbols == resolved.system.symbols
        for key, value in resolved.state_dict().items():  # type: ignore[attr-defined]
            assert value.equal(restored.state_dict()[key])  # type: ignore[attr-defined]

    def test_missing_component(self, tmp_path: Path) -> None:
        structure_path = tmp_path / "ligand.pdb"
        structure_path.write_text(_LIGAND_PDB, encoding="ascii")
        snapshot = _snapshot(tmp_path / "ccd", component_id="OTHER")

        structure = read_structure(structure_path)

        assert structure.system.component_topology_complete.tolist() == [False]
        with pytest.raises(ValueError, match="unknown connectivity"):
            prepare_structure(
                structure_path,
                tmp_path / "ligand.structure.safetensors",
                snapshot_path=snapshot,
            )

    def test_observed_leaving_branch(self, tmp_path: Path) -> None:
        from biotite.structure.io import pdbx

        donor = ChemicalComponent(
            "DNR",
            (ChemicalAtom("C", "C"), ChemicalAtom("O", "O")),
            frozenset({ChemicalBond("C", "O")}),
        )
        acceptor = ChemicalComponent("ACC", (ChemicalAtom("C", "C"),), frozenset())
        branch = Molecule(
            (donor, acceptor),
            frozenset({MoleculeBond(0, "C", 1, "C")}),
            molecule_type="branched",
        )
        system = MolecularSystem.from_entities({"branch": branch}, {"B": "branch"})
        atom_count = int(system.atom_symbol.numel())
        source = tmp_path / "observed-leaving.cif"
        write_structure(
            Structure(
                system=system,
                coordinates=torch.zeros(atom_count, 3),
                coordinate_mask=torch.ones(atom_count, dtype=torch.bool),
                occupancy=torch.ones(atom_count),
                occupancy_mask=torch.ones(atom_count, dtype=torch.bool),
                b_factor=torch.zeros(atom_count),
                b_factor_mask=torch.ones(atom_count, dtype=torch.bool),
                source_symbols=("1", "2"),
                residue_id=torch.tensor([0, 1]),
                insertion_code=torch.full((2,), -1),
                standard_component_id=torch.full((2,), -1),
                batch_size=[],
            ),
            source,
        )
        cif_file = pdbx.CIFFile.read(str(source))
        cif_file.block["pdbx_entity_branch_link"]["leaving_atom_id_1"] = pdbx.CIFColumn(["O"])
        cif_file.write(str(source))
        snapshot = _chemistry_snapshot(
            tmp_path / "ccd",
            _component_row("DNR", [("C", "C"), ("O", "O")], [("C", "O")]),
            _component_row("ACC", [("C", "C")], []),
        )
        destination = tmp_path / "observed-leaving.structure.safetensors"

        with pytest.raises(ValueError, match="declares observed leaving atom 'O'"):
            prepare_structure(source, destination, snapshot_path=snapshot)

        assert not destination.exists()

    def test_source_proton(self, tmp_path: Path) -> None:
        source = tmp_path / "protonated.pdb"
        source.write_text(
            "HETATM    1  N1  LIG A   1       0.000   0.000   0.000  1.00 10.00           N\n"
            "HETATM    2  C1  LIG A   1      -1.500   0.000   0.000  1.00 10.00           C\n"
            "HETATM    3  C2  LIG A   1       0.000  -1.500   0.000  1.00 10.00           C\n"
            "HETATM    4  C3  LIG A   1       0.000   0.000  -1.500  1.00 10.00           C\n"
            "HETATM    5  H1  LIG A   1       1.000   0.000   0.000  1.00 10.00           H\n"
            "END\n",
            encoding="ascii",
        )
        snapshot = _chemistry_snapshot(
            tmp_path / "ccd",
            _component_row(
                "LIG",
                [("N1", "N"), ("C1", "C"), ("C2", "C"), ("C3", "C")],
                [("N1", "C1"), ("N1", "C2"), ("N1", "C3")],
            ),
        )

        structure = prepare_structure(
            source,
            tmp_path / "protonated.structure.safetensors",
            snapshot_path=snapshot,
        )
        atoms = {
            structure.system.atom_site(index).atom_id: index
            for index in range(int(structure.system.atom_symbol.numel()))
        }
        edges = {
            frozenset(
                structure.system.atom_site(int(structure.system.edge_index[side, edge])).atom_id for side in (0, 1)
            )
            for edge in range(int(structure.system.edge_index.shape[1]))
        }

        assert int(structure.system.formal_charge[atoms["N1"]]) == 1
        assert frozenset({"N1", "H1"}) in edges
        assert structure.system.component_topology_complete.tolist() == [True]

    def test_ambiguous_proton(self, tmp_path: Path) -> None:
        source = tmp_path / "ambiguous.pdb"
        source.write_text(
            "HETATM    1  N1  LIG A   1      -0.700   0.000   0.000  1.00 10.00           N\n"
            "HETATM    2  N2  LIG A   1       0.700   0.000   0.000  1.00 10.00           N\n"
            "HETATM    3  H1  LIG A   1       0.000   0.000   0.000  1.00 10.00           H\n"
            "END\n",
            encoding="ascii",
        )
        snapshot = _chemistry_snapshot(
            tmp_path / "ccd",
            _component_row("LIG", [("N1", "N"), ("N2", "N")], []),
        )

        with pytest.raises(ValueError, match="unknown connectivity"):
            prepare_structure(
                source,
                tmp_path / "ambiguous.structure.safetensors",
                snapshot_path=snapshot,
            )

    def test_implied_proton(self, tmp_path: Path) -> None:
        source = tmp_path / "crosslinked.cif"
        _crosslinked_source(source)
        snapshot = _chemistry_snapshot(
            tmp_path / "ccd",
            _component_row("LIG", [("C1", "C"), ("O1", "O"), ("H1", "H")], [("C1", "O1"), ("O1", "H1")]),
            _component_row("OTH", [("X1", "C")], []),
        )

        structure = prepare_structure(
            source,
            tmp_path / "crosslinked.structure.safetensors",
            snapshot_path=snapshot,
        )

        assert "H1" not in {
            structure.system.atom_site(index).atom_id for index in range(int(structure.system.atom_symbol.numel()))
        }

    def test_deuterium(self, tmp_path: Path) -> None:
        source = tmp_path / "deuterated.cif"
        _crosslinked_source(source)
        snapshot = _chemistry_snapshot(
            tmp_path / "ccd",
            _component_row("LIG", [("C1", "C"), ("O1", "O"), ("D1", "D")], [("C1", "O1"), ("O1", "D1")]),
            _component_row("OTH", [("X1", "C")], []),
        )

        with pytest.raises(ValueError, match="prepared molecular graph is chemically invalid"):
            prepare_structure(
                source,
                tmp_path / "deuterated.structure.safetensors",
                snapshot_path=snapshot,
            )
