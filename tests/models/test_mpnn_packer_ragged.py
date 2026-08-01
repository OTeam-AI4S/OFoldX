from __future__ import annotations

import pytest
import torch
from danling.tensors import NestedTensor, nested_execution_guard
from torch._dynamo.testing import CompileCounter
from typing_extensions import NamedTuple

from ofoldx.data import Structure
from ofoldx.models.mpnn_packer import MpnnPackerConfig, MpnnPackerInput, MpnnPackerModel
from ofoldx.runners.collate import collate_mpnn


def _strict_guard():
    return nested_execution_guard(
        forbid_iteration=True,
        forbid_storage_map=True,
        forbid_eager_fallback=True,
        forbid_padded_materialization=True,
        forbid_dense_repack=True,
    )


def _model(device: torch.device) -> MpnnPackerModel:
    return (
        MpnnPackerModel(
            MpnnPackerConfig(
                {
                    "hidden_size": 8,
                    "edge_size": 8,
                    "num_encoder_blocks": 1,
                    "num_decoder_blocks": 1,
                    "num_rbf": 2,
                    "num_neighbors": 4,
                    "num_context_atoms": 2,
                    "num_mix": 2,
                    "num_positional_embeddings": 4,
                    "dropout": 0.0,
                    "coordinate_noise": 0.0,
                }
            )
        )
        .to(device)
        .eval()
    )


def _input(
    length: int,
    device: torch.device,
    *,
    structure: Structure | None = None,
) -> tuple[MpnnPackerInput, torch.Tensor, torch.Tensor]:
    residue = torch.arange(length, device=device, dtype=torch.float32)
    backbone = torch.stack(
        (
            torch.stack((4 * residue, 0.1 * residue, 0.2 * residue), dim=-1),
            torch.stack((4 * residue + 1.2, 0.4 + 0.1 * residue, 0.1 * residue), dim=-1),
            torch.stack((4 * residue + 2.1, 0.2 * residue, 0.8 + 0.1 * residue), dim=-1),
            torch.stack((4 * residue + 2.8, 0.9 + 0.1 * residue, 0.5 + 0.2 * residue), dim=-1),
        ),
        dim=1,
    ).unsqueeze(0)
    context = torch.stack(
        (
            backbone[0, :, 1] + backbone.new_tensor((0.3, 1.1, -0.2)),
            backbone[0, :, 1] + backbone.new_tensor((-0.7, 0.4, 1.3)),
        ),
        dim=1,
    ).unsqueeze(0)
    atom14 = torch.zeros(1, length, 14, 3, device=device)
    atom14[:, :, :4] = backbone
    atom14[:, :, 4:] = backbone[:, :, 1:2] + torch.arange(10, device=device).view(1, 1, 10, 1) * 0.03
    atom14_mask = torch.ones(1, length, 14, device=device, dtype=torch.bool)
    atom37 = torch.zeros(1, length, 37, 3, device=device)
    atom37[:, :, :3] = backbone[:, :, :3]
    atom37[:, :, 4] = backbone[:, :, 3]
    atom37_mask = torch.zeros(1, length, 37, device=device, dtype=torch.bool)
    atom37_mask[:, :, (0, 1, 2, 4)] = True
    structure = Structure.from_sequence("A" * length) if structure is None else structure
    input = MpnnPackerInput(  # type: ignore[call-arg]
        backbone_coordinates=backbone,
        sequence_tokens=torch.zeros(1, length, device=device, dtype=torch.long),
        residue_mask=torch.ones(1, length, device=device, dtype=torch.bool),
        residue_index=torch.arange(length, device=device).unsqueeze(0),
        chain_index=torch.zeros(1, length, device=device, dtype=torch.long),
        context_coordinates=context,
        context_atomic_numbers=torch.tensor((6, 8), device=device).view(1, 1, 2).expand(1, length, 2),
        context_mask=torch.ones(1, length, 2, device=device, dtype=torch.bool),
        atom37_coordinates=atom37,
        atom37_mask=atom37_mask,
        pack_positions=torch.ones(1, length, device=device, dtype=torch.bool),
        structure=structure,
        component_index=torch.arange(length, device=device).unsqueeze(0),
        batch_size=[],
    )
    return input, atom14, atom14_mask


def _case(
    lengths: tuple[int, int],
    device: torch.device,
    *,
    structure: Structure | None = None,
):
    dense = tuple(_input(length, device, structure=structure) for length in lengths)
    nested_input = collate_mpnn([item[0] for item in dense])
    coordinates = NestedTensor(tuple(item[1].squeeze(0) for item in dense), ragged_dims=(0,))
    atom_mask = NestedTensor(tuple(item[2].squeeze(0) for item in dense), ragged_dims=(0,))
    return dense, nested_input, coordinates, atom_mask


def _dense_reference(model: MpnnPackerModel, dense) -> tuple[torch.Tensor, ...]:
    outputs = tuple(
        model(input, atom14_coordinates=coordinates, atom14_mask=atom_mask) for input, coordinates, atom_mask in dense
    )
    return tuple(
        torch.cat(tuple(getattr(output, name).squeeze(0) for output in outputs))
        for name in ("mean", "concentration", "mix_logits")
    )


def _assert_outputs(actual, expected) -> None:
    for value, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, reference, rtol=2e-4, atol=2e-4)


_TENSOR_INPUT_FIELDS = (
    "backbone_coordinates",
    "sequence_tokens",
    "residue_mask",
    "residue_index",
    "chain_index",
    "context_coordinates",
    "context_atomic_numbers",
    "context_mask",
    "atom37_coordinates",
    "atom37_mask",
    "pack_positions",
    "component_index",
)


class _TensorMpnnPackerInput(NamedTuple):
    backbone_coordinates: torch.Tensor | NestedTensor
    sequence_tokens: torch.Tensor | NestedTensor
    residue_mask: torch.Tensor | NestedTensor
    residue_index: torch.Tensor | NestedTensor
    chain_index: torch.Tensor | NestedTensor
    context_coordinates: torch.Tensor | NestedTensor
    context_atomic_numbers: torch.Tensor | NestedTensor
    context_mask: torch.Tensor | NestedTensor
    atom37_coordinates: torch.Tensor | NestedTensor
    atom37_mask: torch.Tensor | NestedTensor
    pack_positions: torch.Tensor | NestedTensor
    component_index: torch.Tensor | NestedTensor


@pytest.mark.skipif(not torch.cuda.is_available(), reason="OOps masked_knn requires CUDA")
def test_mpnn_packer_ragged_raw_forward_strict_dense_parity() -> None:
    torch.manual_seed(20260830)
    device = torch.device("cuda")
    model = _model(device)
    dense, nested_input, coordinates, atom_mask = _case((3, 5), device)
    expected = _dense_reference(model, dense)

    with _strict_guard():
        output = model(nested_input, atom14_coordinates=coordinates, atom14_mask=atom_mask)
        loss = output.mean.concat.square().mean() + output.concentration.concat.mean() + output.mix_logits.concat.mean()
        loss.backward()

    actual = (output.mean.concat, output.concentration.concat, output.mix_logits.concat)
    _assert_outputs(actual, expected)
    assert model.torsion_projection.weight.grad is not None
    assert torch.isfinite(model.torsion_projection.weight.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="OOps masked_knn requires CUDA")
def test_mpnn_packer_ragged_raw_forward_reuses_dynamic_fullgraph() -> None:
    torch.manual_seed(20260830)
    device = torch.device("cuda")
    model = _model(device)
    counter = CompileCounter()
    cases = tuple(_case(lengths, device) for lengths in ((3, 5), (4, 6)))

    def run(*args):
        fields = args[: len(_TENSOR_INPUT_FIELDS)]
        coordinates, atom_mask = args[len(_TENSOR_INPUT_FIELDS) :]
        # Tensordict specializes each Tensorclass non-tensor metadata checker by
        # identity. This tensor-only adapter isolates the real model graph from that
        # upstream wrapper boundary; every changing field remains a graph input.
        input = _TensorMpnnPackerInput(*fields)
        output = model(  # type: ignore[arg-type]
            input,
            atom14_coordinates=coordinates,
            atom14_mask=atom_mask,
        )
        return output.mean.concat, output.concentration.concat, output.mix_logits.concat

    compiled = torch.compile(run, backend=counter, fullgraph=True, dynamic=True)
    for dense, nested_input, coordinates, atom_mask in cases:
        expected = _dense_reference(model, dense)
        tensor_fields = tuple(getattr(nested_input, name) for name in _TENSOR_INPUT_FIELDS)
        with _strict_guard():
            actual = compiled(*tensor_fields, coordinates, atom_mask)
        _assert_outputs(actual, expected)
    assert counter.frame_count == 1
