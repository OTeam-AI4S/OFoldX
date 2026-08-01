from __future__ import annotations

import torch

from ofoldx.data import MolecularSystem, PolymerSequence, Structure
from ofoldx.models.pxdesign.pipeline import _full_complex


def _structure(chains: dict[str, str]) -> Structure:
    system = MolecularSystem.from_entities(
        {f"entity_{instance}": PolymerSequence.from_one_letter(sequence) for instance, sequence in chains.items()},
        {instance: f"entity_{instance}" for instance in chains},
    )
    atom_count = system.atom_symbol.numel()
    component_count = system.component_symbol.numel()
    return Structure(  # type: ignore[call-arg]
        system=system,
        coordinates=torch.zeros(atom_count, 3),
        coordinate_mask=torch.zeros(atom_count, dtype=torch.bool),
        occupancy=torch.zeros(atom_count),
        occupancy_mask=torch.zeros(atom_count, dtype=torch.bool),
        b_factor=torch.zeros(atom_count),
        b_factor_mask=torch.zeros(atom_count, dtype=torch.bool),
        source_symbols=(),
        residue_id=torch.full((component_count,), -1, dtype=torch.int64),
        insertion_code=torch.full((component_count,), -1, dtype=torch.int64),
        batch_size=[],
    )


def test_parent_offsets() -> None:
    target = _structure({"target": "AA"}).replace(  # type: ignore[attr-defined]
        source_symbols=("31", "I", "GLY", "SER", "unused"),
        residue_id=torch.tensor([0, -1]),
        insertion_code=torch.tensor([1, -1]),
        component_parent_ptr=torch.tensor([0, 2, 2]),
        parent_component_id=torch.tensor([2, 3]),
    )
    design = _structure({"cropped": "A", "binder": "AA"}).replace(  # type: ignore[attr-defined]
        source_symbols=("78", "Q", "ASP", "GLU", "discarded"),
        residue_id=torch.tensor([0, -1, 0]),
        insertion_code=torch.tensor([-1, -1, 1]),
        component_parent_ptr=torch.tensor([0, 1, 1, 3]),
        parent_component_id=torch.tensor([4, 2, 3]),
    )

    result = _full_complex(target, design)

    assert tuple(result.system.instance_id(i) for i in range(2)) == ("target", "binder")
    assert result.source_symbols == target.source_symbols + design.source_symbols
    assert result.component_parent_ptr.tolist() == [0, 2, 2, 2, 4]
    assert result.parent_component_id.tolist() == [2, 3, 7, 8]
    assert [result.component_parents(i) for i in range(4)] == [("GLY", "SER"), (), (), ("ASP", "GLU")]
    assert result.residue_id.tolist() == [0, -1, -1, 5]
    assert result.insertion_code.tolist() == [1, -1, -1, 6]
    assert target.component_parent_ptr.tolist() == [0, 2, 2]
    assert target.parent_component_id.tolist() == [2, 3]
    assert design.component_parent_ptr.tolist() == [0, 1, 1, 3]
    assert design.parent_component_id.tolist() == [4, 2, 3]
