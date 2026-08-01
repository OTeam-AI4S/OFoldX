from __future__ import annotations

import torch

from ofoldx.data import AtomSite, MolecularSystem, PolymerSequence, Structure
from ofoldx.data.structure import RESOLUTION_KIND_REFINEMENT, RESOLUTION_KIND_REFLECTIONS
from ofoldx.models.boltzgen.pipeline import _extract_instances, _write_designed_structure
from ofoldx.models.boltzgen.processor import _transfer_structure


def _structure(instances: tuple[str, ...]) -> Structure:
    polymer = PolymerSequence.from_one_letter("A")
    system = MolecularSystem.from_entities(
        {f"entity_{instance}": polymer for instance in instances},
        {instance: f"entity_{instance}" for instance in instances},
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


def test_design_parents() -> None:
    source = _structure(("A", "B", "C")).replace(  # type: ignore[attr-defined]
        source_symbols=("GLY", "SER", "ASP", "GLU", "ASN"),
        component_parent_ptr=torch.tensor([0, 2, 4, 5]),
        parent_component_id=torch.tensor([0, 1, 2, 3, 4]),
    )

    result = _write_designed_structure(source, {("A", 0): "ALA", ("B", 0): "GLY"})

    assert [result.system.symbols[int(symbol)] for symbol in result.system.component_symbol] == ["ALA", "GLY", "ALA"]
    assert result.source_symbols == source.source_symbols
    assert result.component_parent_ptr.tolist() == [0, 2, 2, 3]
    assert result.parent_component_id.tolist() == [0, 1, 4]
    assert [result.component_parents(i) for i in range(3)] == [("GLY", "SER"), (), ("ASN",)]
    assert source.component_parent_ptr.tolist() == [0, 2, 4, 5]
    assert source.parent_component_id.tolist() == [0, 1, 2, 3, 4]


def test_parent_transfer() -> None:
    source = _structure(("A", "B", "C"))
    charge_atoms = {
        source.system.atom_index(AtomSite("A", 0, "N")),
        source.system.atom_index(AtomSite("B", 0, "CA")),
        source.system.atom_index(AtomSite("C", 0, "O")),
    }
    source = source.replace(  # type: ignore[attr-defined]
        source_symbols=("GLY", "SER", "discarded", "ASP", "GLU", "11", "22", "33"),
        residue_id=torch.tensor([5, 6, 7]),
        component_parent_ptr=torch.tensor([0, 2, 3, 5]),
        parent_component_id=torch.tensor([0, 1, 2, 3, 4]),
        source_formal_charge_mask=torch.tensor(
            [atom in charge_atoms for atom in range(source.system.atom_symbol.numel())],
            dtype=torch.bool,
        ),
        system_resolution_ptr=torch.tensor([0, 2]),
        resolution=torch.tensor([1.5, 2.25], dtype=torch.float64),
        resolution_kind=torch.tensor([RESOLUTION_KIND_REFINEMENT, RESOLUTION_KIND_REFLECTIONS], dtype=torch.uint8),
    )

    selected = _extract_instances(source, {"A", "C"})

    assert selected.source_symbols == source.source_symbols
    assert selected.residue_id.tolist() == [5, 7]
    assert selected.component_parent_ptr.tolist() == [0, 2, 4]
    assert selected.parent_component_id.tolist() == [0, 1, 3, 4]
    assert {
        selected.system.atom_site(atom)
        for atom in torch.nonzero(selected.source_formal_charge_mask, as_tuple=True)[0].tolist()
    } == {AtomSite("A", 0, "N"), AtomSite("C", 0, "O")}
    assert selected.system_resolution_ptr.tolist() == [0, 2]
    assert selected.resolution.tolist() == [1.5, 2.25]
    assert selected.resolution_kind.tolist() == [RESOLUTION_KIND_REFINEMENT, RESOLUTION_KIND_REFLECTIONS]

    reordered_system = _structure(("C", "A")).system
    atom_to_new = torch.tensor(
        [
            reordered_system.atom_index(selected.system.atom_site(atom))
            for atom in range(selected.system.atom_symbol.numel())
        ],
        dtype=torch.int64,
    )
    result = _transfer_structure(selected, reordered_system, atom_to_new, torch.tensor([1, 0]))

    assert tuple(result.system.instance_id(i) for i in range(2)) == ("C", "A")
    assert result.source_symbols == source.source_symbols
    assert result.residue_id.tolist() == [7, 5]
    assert result.component_parent_ptr.tolist() == [0, 2, 4]
    assert result.parent_component_id.tolist() == [3, 4, 0, 1]
    assert [result.component_parents(i) for i in range(2)] == [("ASP", "GLU"), ("GLY", "SER")]
    assert {
        result.system.atom_site(atom)
        for atom in torch.nonzero(result.source_formal_charge_mask, as_tuple=True)[0].tolist()
    } == {AtomSite("A", 0, "N"), AtomSite("C", 0, "O")}
    assert result.system_resolution_ptr.tolist() == [0, 2]
    assert result.resolution.tolist() == [1.5, 2.25]
    assert result.resolution_kind.tolist() == [RESOLUTION_KIND_REFINEMENT, RESOLUTION_KIND_REFLECTIONS]
    assert selected.parent_component_id.tolist() == [0, 1, 3, 4]
    assert source.parent_component_id.tolist() == [0, 1, 2, 3, 4]
    assert set(torch.nonzero(source.source_formal_charge_mask, as_tuple=True)[0].tolist()) == charge_atoms

    cleared = _transfer_structure(
        selected,
        reordered_system,
        atom_to_new,
        torch.tensor([1, 0]),
        preserve_observation_metadata=False,
    )
    assert not bool(cleared.source_formal_charge_mask.any())
    assert cleared.system_resolution_ptr.tolist() == [0, 0]
    assert cleared.resolution.numel() == cleared.resolution_kind.numel() == 0
