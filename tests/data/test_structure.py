from __future__ import annotations

import torch

import ofoldx.data as public_data
import ofoldx.data.structure as structure_module
from ofoldx.data import ChemicalAtom, ChemicalComponent, MolecularSystem, Molecule
from ofoldx.data.structure import Structure


def _structure(instance_id: str, offset: float = 0.0) -> Structure:
    component = ChemicalComponent(
        "LIG",
        (ChemicalAtom("C1", "C"), ChemicalAtom("O1", "O")),
        frozenset(),
    )
    system = MolecularSystem.from_entities(
        {f"entity_{instance_id}": Molecule((component,))},
        {instance_id: f"entity_{instance_id}"},
    )
    return Structure(
        system=system,
        coordinates=torch.tensor(
            [[offset, 0.0, 0.0], [offset + 1.0, 0.0, 0.0]],
        ),
        coordinate_mask=torch.tensor([True, False]),
        occupancy=torch.tensor([1.0, 0.0]),
        occupancy_mask=torch.tensor([True, False]),
        b_factor=torch.tensor([10.0, 0.0]),
        b_factor_mask=torch.tensor([True, False]),
        source_symbols=("7", "A"),
        residue_id=torch.tensor([0]),
        insertion_code=torch.tensor([1]),
        batch_size=[],
    )


class TestStructure:
    def test_public_class_and_atom_aligned_values(self) -> None:
        structure = _structure("A")

        assert public_data.Structure is structure_module.Structure
        assert structure.coordinates.shape == (
            structure.system.atom_symbol.numel(),
            3,
        )
        assert structure.coordinate_mask.tolist() == [True, False]
        assert structure.residue_id.tolist() == [0]
        assert structure.source_symbols == ("7", "A")

        double = structure.double()
        assert double.coordinates.dtype == torch.float64
        assert double.occupancy.dtype == torch.float64
        assert double.coordinate_mask.dtype == torch.bool
        assert double.residue_id.dtype == torch.int64
        assert double.system.edge_index.dtype == torch.int64

    def test_batch_and_unbatch_remap_source_symbols(self) -> None:
        first = _structure("A")
        second = _structure("B", 10.0)

        batched = Structure.batch((first, second))
        assert batched.coordinates[:, 0].tolist() == [0.0, 1.0, 10.0, 11.0]
        assert batched.source_symbols == ("7", "A", "7", "A")
        assert batched.residue_id.tolist() == [0, 2]
        assert batched.insertion_code.tolist() == [1, 3]

        restored_first, restored_second = batched.unbatch()
        assert (
            restored_first.source_symbols
            == restored_second.source_symbols
            == (
                "7",
                "A",
            )
        )
        assert restored_first.residue_id.tolist() == [0]
        assert restored_second.residue_id.tolist() == [0]
        assert restored_second.coordinates[:, 0].tolist() == [10.0, 11.0]
