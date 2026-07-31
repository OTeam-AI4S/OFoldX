from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ofoldx.data import (
    MOLECULE_BRANCHED,
    ChemicalAtom,
    ChemicalBond,
    ChemicalComponent,
    MolecularSystem,
    Molecule,
    Structure,
    encode_system,
)
from ofoldx.io import read_structure, write_structure


def _structure() -> Structure:
    component = ChemicalComponent(
        "LIG",
        (
            ChemicalAtom("C1", "C", formal_charge=0),
            ChemicalAtom("O1", "O", formal_charge=-1),
        ),
        frozenset({ChemicalBond("C1", "O1")}),
        reference_coordinates=((0.0, 0.0, 0.0), (1.2, 0.0, 0.0)),
    )
    system = MolecularSystem.from_entities(
        {"ligand": Molecule((component,))},
        {"L": "ligand"},
    )
    return Structure(
        system=system,
        coordinates=torch.tensor([[1.0, 2.0, 3.0], [2.2, 2.0, 3.0]]),
        coordinate_mask=torch.ones(2, dtype=torch.bool),
        occupancy=torch.tensor([1.0, 0.5]),
        occupancy_mask=torch.ones(2, dtype=torch.bool),
        b_factor=torch.tensor([10.0, 20.0]),
        b_factor_mask=torch.ones(2, dtype=torch.bool),
        source_symbols=("11",),
        residue_id=torch.tensor([0]),
        insertion_code=torch.tensor([-1]),
        batch_size=[],
    )


class TestMmcifIO:
    def test_round_trip_preserves_the_complete_public_graph_and_coordinates(
        self,
        tmp_path: Path,
    ) -> None:
        structure = _structure()
        path = tmp_path / "ligand.cif"
        write_structure(structure, path)
        restored = read_structure(path)

        expected_metadata, expected_arrays = encode_system(structure.system)
        actual_metadata, actual_arrays = encode_system(restored.system)
        assert actual_metadata == expected_metadata
        for name, expected in expected_arrays.items():
            assert torch.equal(actual_arrays[name], expected)
        assert torch.equal(restored.coordinates, structure.coordinates)
        assert torch.equal(restored.coordinate_mask, structure.coordinate_mask)
        assert torch.equal(restored.occupancy, structure.occupancy)
        assert restored.source_symbols == structure.source_symbols

    def test_atom_site_is_the_coordinate_source_of_truth(
        self,
        tmp_path: Path,
    ) -> None:
        from biotite.structure.io import pdbx

        path = tmp_path / "coordinates.cif"
        write_structure(_structure(), path)
        cif_file = pdbx.CIFFile.read(str(path))
        values = cif_file.block["atom_site"]["Cartn_x"].as_array()
        values[0] = 123.0
        cif_file.block["atom_site"]["Cartn_x"] = pdbx.CIFColumn(values)
        cif_file.write(str(path))

        assert read_structure(path).coordinates[0, 0].item() == 123.0

    def test_masked_coordinates_round_trip_without_numeric_evidence(
        self,
        tmp_path: Path,
    ) -> None:
        structure = _structure()
        masked = structure.replace(
            coordinates=structure.coordinates.clone(),
            coordinate_mask=torch.tensor([False, True]),
            occupancy=structure.occupancy.clone(),
            occupancy_mask=torch.tensor([False, True]),
            b_factor=structure.b_factor.clone(),
            b_factor_mask=torch.tensor([False, True]),
        )
        masked.coordinates[0] = torch.nan
        path = tmp_path / "masked.cif"

        write_structure(masked, path)
        restored = read_structure(path)

        assert not restored.coordinate_mask[0]
        assert restored.coordinates[0].tolist() == [0.0, 0.0, 0.0]

    def test_aromaticity_masks_are_not_replaced_by_filler_values(
        self,
        tmp_path: Path,
    ) -> None:
        structure = _structure()
        unknown_atoms = structure.replace(
            system=structure.system.replace(
                atom_aromatic=torch.ones_like(structure.system.atom_aromatic),
            )
        )
        path = tmp_path / "unknown-aromaticity.cif"
        write_structure(unknown_atoms, path)
        assert not read_structure(path).system.atom_aromatic_known.any()

        partial_bond = structure.replace(
            system=structure.system.replace(
                bond_aromatic_known=torch.zeros_like(structure.system.bond_aromatic_known),
            )
        )
        with pytest.raises(ValueError, match="partially known bond order"):
            write_structure(partial_bond, tmp_path / "partial-bond.cif")

    def test_branch_definition_cannot_override_the_entity_category(
        self,
        tmp_path: Path,
    ) -> None:
        from biotite.structure.io import pdbx

        structure = _structure()
        structure = structure.replace(
            system=structure.system.replace(
                entity_molecule_type=structure.system.entity_molecule_type.new_tensor([MOLECULE_BRANCHED])
            )
        )
        path = tmp_path / "conflicting-category.cif"
        write_structure(structure, path)
        cif_file = pdbx.CIFFile.read(str(path))
        cif_file.block["entity"]["type"] = pdbx.CIFColumn(["non-polymer"])
        cif_file.write(str(path))

        with pytest.raises(ValueError, match="branched-entity definition"):
            read_structure(path)
