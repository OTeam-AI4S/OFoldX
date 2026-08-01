from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from ofoldx.data import Structure
from ofoldx.models.proteinmpnn import ProteinMpnnInput
from ofoldx.runners.collate import (
    collate_boltzgen_denoising,
    collate_boltzgen_ifold,
    collate_odesign,
    collate_sequence,
)


def _protein_input(length: int) -> ProteinMpnnInput:
    return ProteinMpnnInput(  # type: ignore[call-arg]
        backbone_coordinates=torch.randn(1, length, 4, 3),
        sequence_tokens=torch.arange(length).remainder(20).unsqueeze(0),
        residue_mask=torch.ones(1, length, dtype=torch.bool),
        residue_index=torch.arange(length).unsqueeze(0),
        chain_index=torch.zeros(1, length, dtype=torch.int64),
        design_positions=torch.ones(1, length, dtype=torch.bool),
        structure=Structure.from_sequence("A" * length),
        component_index=torch.arange(length).unsqueeze(0),
        tied_group_ids=torch.arange(length).unsqueeze(0),
        batch_size=[],
    )


def test_collate_sequence_ragged() -> None:
    batch = collate_sequence([_protein_input(2), _protein_input(3)])

    input = batch["input"]
    assert input.sequence_tokens[0].shape == (2,)
    assert input.sequence_tokens[1].shape == (3,)
    assert batch["target"].sequence_tokens.tolist() == input.sequence_tokens.tolist()
    assert batch["loss_normalizer"] == 5


def test_collate_boltzgen_denoising_mask() -> None:
    input = SimpleNamespace(
        atom_mask=torch.tensor([[True, True, True, True]]),
        atom_index=torch.full((1, 4), -1),
    )
    target = SimpleNamespace(
        coordinate_mask=torch.tensor([[True, False, True, True]]),
        coordinate_atom_index=torch.tensor([[2, 0, -1, 1]]),
    )

    batch = collate_boltzgen_denoising([(input, target)])

    assert torch.equal(batch["target"], torch.tensor([[True, True, False, False]]))
    assert batch["loss_normalizer"] == 2


def test_collate_boltzgen_ifold_mask() -> None:
    input = SimpleNamespace(
        token_mask=torch.tensor([[True, True, True, False]]),
        token_resolved_mask=torch.tensor([[True, False, True, True]]),
        design_mask=torch.tensor([[True, True, False, True]]),
    )
    residue_type = torch.zeros(1, 4, 33)

    batch = collate_boltzgen_ifold([(input, residue_type)])

    assert torch.equal(batch["target"]["design_mask"], torch.tensor([[True, False, False, False]]))
    assert batch["loss_normalizer"] == 1


def test_collate_odesign_mask() -> None:
    input = SimpleNamespace(
        atom_mask=torch.tensor([[True, True, True, False]]),
        fixed_coordinate_mask=torch.tensor([[True, False, False, False]]),
    )
    target = SimpleNamespace(coordinate_mask=torch.tensor([[True, True, False, True]]))

    batch = collate_odesign([(input, target)])

    assert batch["loss_normalizer"] == 1


def test_collate_odesign_empty() -> None:
    input = SimpleNamespace(
        atom_mask=torch.tensor([[True, True]]),
        fixed_coordinate_mask=torch.tensor([[True, False]]),
    )
    target = SimpleNamespace(coordinate_mask=torch.tensor([[True, False]]))

    with pytest.raises(ValueError, match="no resolved generated atom"):
        collate_odesign([(input, target)])
