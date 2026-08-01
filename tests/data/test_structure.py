from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors import safe_open

from ofoldx.data import ChemicalAtom, ChemicalComponent, MolecularSystem, Molecule
from ofoldx.data.structure import (
    RESOLUTION_KIND_REFINEMENT,
    Structure,
    load_structure,
    save_structure,
)


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
        component_parent_ptr=torch.tensor([0, 1], dtype=torch.int64),
        parent_component_id=torch.tensor([1], dtype=torch.int64),
        batch_size=[],
    )


class TestStructure:
    def test_defaults(self) -> None:
        structure = _structure("A")

        assert structure.coordinates.shape == (structure.system.atom_symbol.numel(), 3)
        assert structure.coordinate_mask.tolist() == [True, False]
        assert structure.residue_id.tolist() == [0]
        assert structure.source_symbols == ("7", "A")
        assert structure.system_resolution_ptr.tolist() == [0, 0]
        assert structure.resolution.numel() == 0

    def test_dtype_conversion(self) -> None:
        structure = _structure("A")

        double = structure.double()

        assert double.coordinates.dtype == torch.float64
        assert double.occupancy.dtype == torch.float64
        assert double.coordinate_mask.dtype == torch.bool
        assert double.residue_id.dtype == torch.int64
        assert double.system.edge_index.dtype == torch.int64

    def test_atom_subgraph(self) -> None:
        structure = _structure("A")

        selected, old_to_new = structure.atom_subgraph(torch.tensor([False, True]))

        assert selected.system.atom_symbol.numel() == 1
        assert selected.coordinates[:, 0].tolist() == [1.0]
        assert selected.coordinate_mask.tolist() == [False]
        assert old_to_new.tolist() == [-1, 0]
        with pytest.raises(TypeError, match="bool dtype"):
            structure.atom_subgraph(torch.ones(2))
        with pytest.raises(ValueError, match="one value per atom"):
            structure.atom_subgraph(torch.ones(1, dtype=torch.bool))

    def test_batch_round_trip(self) -> None:
        first = _structure("A").replace(
            system_resolution_ptr=torch.tensor([0, 1]),
            resolution=torch.tensor([1.5]),
            resolution_kind=torch.tensor([RESOLUTION_KIND_REFINEMENT], dtype=torch.uint8),
        )
        second = _structure("B", 10.0)

        batched = Structure.batch((first, second))
        assert batched.coordinates[:, 0].tolist() == [0.0, 1.0, 10.0, 11.0]
        assert batched.source_symbols == ("7", "A", "7", "A")
        assert batched.residue_id.tolist() == [0, 2]
        assert batched.insertion_code.tolist() == [1, 3]
        assert [batched.component_parents(i) for i in range(2)] == [("A",), ("A",)]
        assert batched.system_resolution_ptr.tolist() == [0, 1, 1]
        assert batched.resolution.tolist() == [1.5]
        assert batched.resolution_kind.tolist() == [RESOLUTION_KIND_REFINEMENT]

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
        assert restored_first.component_parents(0) == ("A",)
        assert restored_second.component_parents(0) == ("A",)
        assert restored_second.coordinates[:, 0].tolist() == [10.0, 11.0]
        assert restored_first.resolution.tolist() == [1.5]
        assert restored_second.system_resolution_ptr.tolist() == [0, 0]

    def test_safetensors_roundtrip(self, tmp_path: Path) -> None:
        structure = _structure("A").replace(
            system_resolution_ptr=torch.tensor([0, 1]),
            resolution=torch.tensor([2.1]),
            resolution_kind=torch.tensor([RESOLUTION_KIND_REFINEMENT], dtype=torch.uint8),
        )
        path = tmp_path / "example.structure.safetensors"

        save_structure(structure, path)
        restored = load_structure(path)
        with safe_open(str(path), framework="pt") as artifact:
            assert artifact.metadata().keys() == {"format", "system_symbols", "source_symbols"}
            assert artifact.metadata()["format"] == "ofoldx.structure"

        assert restored.system.symbols == structure.system.symbols
        assert restored.source_symbols == structure.source_symbols
        assert restored.state_dict().keys() == structure.state_dict().keys()  # type: ignore[attr-defined]
        for key, value in structure.state_dict().items():  # type: ignore[attr-defined]
            assert torch.equal(restored.state_dict()[key], value)  # type: ignore[attr-defined]

    def test_safetensors_suffix_is_explicit(self, tmp_path: Path) -> None:
        path = tmp_path / "example.safetensors"

        with pytest.raises(ValueError, match=r"\.structure\.safetensors"):
            save_structure(_structure("A"), path)
        with pytest.raises(ValueError, match=r"\.structure\.safetensors"):
            load_structure(path)


@pytest.fixture
def parented_structure() -> Structure:
    return Structure.from_sequence("AA").replace(
        source_symbols=("ALA", "GLY"),
        component_parent_ptr=torch.tensor([0, 0, 2]),
        parent_component_id=torch.tensor([0, 1]),
    )


class TestComponentParents:
    @pytest.mark.parametrize(
        "indices, pointers, parents",
        [
            ([1, -1, 1, 0], [0, 2, 2, 4, 4], [0, 1, 0, 1]),
            ([0], [0, 0], []),
            ([], [0], []),
        ],
        ids=["reorder-copy-new", "no-declaration", "empty"],
    )
    def test_selection(self, parented_structure, indices, pointers, parents):
        pointer, identifiers = parented_structure.select_component_parents(indices)
        assert pointer.tolist() == pointers
        assert identifiers.tolist() == parents
        assert parented_structure.component_parents(1) == ("ALA", "GLY")

    def test_preservation(self, parented_structure, tmp_path):
        path = tmp_path / "parents.structure.safetensors"
        save_structure(parented_structure, path)
        restored = load_structure(path)
        batch = Structure.batch([Structure.from_sequence("G"), restored]).double()
        first, second = batch.unbatch()
        assert first.component_parents(0) == ()
        assert [second.component_parents(i) for i in range(2)] == [(), ("ALA", "GLY")]
        selected, _ = second.atom_subgraph(second.system.atomic_number != 1)
        assert selected.component_parents(1) == ("ALA", "GLY")
        assert selected.coordinates.dtype == torch.float64
