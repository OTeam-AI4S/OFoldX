# OFoldX
# Copyright (C) 2026-Present  Valhalla Technology

# This file is part of OFoldX.

# OFoldX is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# OFoldX is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# For additional terms and clarifications, please refer to our License FAQ at:
# <https://ofoldx.valtech.ai/about/license-faq/>.

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from danling.tensors import NestedTensor

from ofoldx.data.tokenization import TOKENS
from ofoldx.models.alphafold3 import AlphaFold3Processor, AlphaFold3TemplateFeatures
from ofoldx.models.proteinmpnn import ProteinMpnnConfig, ProteinMpnnInput, ProteinMpnnModel
from ofoldx.models.protenix import ProtenixProcessor, ProtenixTemplateFeatures
from ofoldx.models.targets import SequenceTarget
from ofoldx.modules.mpnn import MpnnSequenceLoss
from ofoldx.runners.collate import (
    collate_alphafold3_inputs,
    collate_boltzgen_denoising,
    collate_boltzgen_ifold,
    collate_odesign,
    collate_protenix_inputs,
    collate_sequence,
)

# Label smoothing would put mass on the classes outside ProteinMPNN's trained vocabulary (-inf logits).
sequence_loss = MpnnSequenceLoss(label_smoothing=0.0)


def mpnn_criterion(output, input):
    return sequence_loss(output.logits, input.sequence_tokens, input.residue_mask & input.design_mask)


def _assert_causal_unit_ranks(ranks: NestedTensor, group_ids: NestedTensor) -> None:
    for item_ranks, item_groups in zip(ranks, group_ids, strict=True):
        for group in item_groups.unique():
            assert item_ranks[item_groups == group].unique().numel() == 1
        assert item_ranks.unique().numel() == item_groups.unique().numel()


def _protein_input(
    length: int,
    *,
    ca_only: bool = False,
    tied_group_ids: tuple[int, ...] | None = None,
    design_mask: tuple[bool, ...] | None = None,
) -> ProteinMpnnInput:
    if tied_group_ids is None:
        tied_group_ids = tuple(range(length))
    if design_mask is None:
        design_mask = (True,) * length
    return ProteinMpnnInput(  # type: ignore[call-arg]
        backbone_coordinates=torch.randn(1, length, 3) if ca_only else torch.randn(1, length, 4, 3),
        sequence_tokens=(torch.arange(length).remainder(20) + TOKENS.index("A")).unsqueeze(0),
        residue_mask=torch.ones(1, length, dtype=torch.bool),
        residue_index=torch.arange(length).unsqueeze(0),
        chain_index=torch.zeros(1, length, dtype=torch.int64),
        design_mask=torch.tensor(design_mask, dtype=torch.bool).unsqueeze(0),
        component_index=torch.arange(length).unsqueeze(0),
        tied_group_ids=torch.tensor(tied_group_ids).unsqueeze(0),
        batch_size=[],
    )


def _sequence_sample(input, *, tokens=None, mask=None):
    return input, SequenceTarget(
        input.sequence_tokens.clone() if tokens is None else tokens,
        torch.ones_like(input.sequence_tokens, dtype=torch.bool) if mask is None else mask,
    )


def test_collate_sequence_ragged() -> None:
    batch = collate_sequence([_sequence_sample(_protein_input(2)), _sequence_sample(_protein_input(3))])

    input = batch["input"]
    assert all(isinstance(value, NestedTensor) for value in input.values() if value is not None)
    assert input.sequence_tokens[0].shape == (2,)
    assert input.sequence_tokens[1].shape == (3,)
    assert torch.equal(input.sequence_tokens.element_sizes(), torch.tensor([[2], [3]]))
    assert torch.equal(input.backbone_coordinates.element_sizes(), torch.tensor([[2, 4, 3], [3, 4, 3]]))
    assert input.sequence_tokens.ragged_dims == (0,)
    assert input.backbone_coordinates.ragged_dims == (0,)
    assert input.component_index[1].tolist() == [0, 1, 2]
    assert batch["target"]["target"].tokens.tolist() == input.sequence_tokens.tolist()
    assert batch["loss_normalizer"] == 5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="OOps masked_knn requires CUDA")
@pytest.mark.parametrize("ca_only", [False, True])
def test_protein_mpnn_ragged_training_without_materialization(ca_only: bool) -> None:
    inputs = (
        _protein_input(
            3,
            ca_only=ca_only,
            tied_group_ids=(0, 1, 0),
            design_mask=(True, True, False),
        ).to("cuda"),
        _protein_input(
            5,
            ca_only=ca_only,
            tied_group_ids=(0, 1, 2, 1, 3),
            design_mask=(True, True, True, False, True),
        ).to("cuda"),
    )
    model = ProteinMpnnModel(
        ProteinMpnnConfig(
            {
                "hidden_size": 8,
                "num_encoder_blocks": 1,
                "num_decoder_blocks": 1,
                "num_neighbors": 4,
                "dropout": 0.0,
                "ca_only": ca_only,
            }
        )
    ).to("cuda")
    dense_outputs = tuple(model(input, decoding_ranks=input.tied_group_ids) for input in inputs)
    dense_loss = sum(
        mpnn_criterion(output, input) * (input.residue_mask & input.design_mask).sum()
        for output, input in zip(dense_outputs, inputs, strict=True)
    ) / sum((input.residue_mask & input.design_mask).sum() for input in inputs)

    nested_input = collate_sequence([_sequence_sample(input) for input in inputs])["input"]
    nested_output = model(nested_input, decoding_ranks=nested_input.tied_group_ids)
    nested_loss = mpnn_criterion(nested_output, nested_input)
    sampled_order_output = model(
        nested_input,
        coordinate_noise=0.05,
        generator=torch.Generator(device="cuda").manual_seed(20260830),
    )
    sampled_order_loss = mpnn_criterion(sampled_order_output, nested_input)
    sampled_order_loss.backward()

    assert isinstance(nested_output.logits, NestedTensor)
    assert isinstance(nested_output.decoding_ranks, NestedTensor)
    assert torch.equal(nested_output.logits.element_sizes(), torch.tensor([[3, len(TOKENS)], [5, len(TOKENS)]]))
    for index, dense_output in enumerate(dense_outputs):
        torch.testing.assert_close(nested_output.logits[index], dense_output.logits[0])
        assert torch.equal(nested_output.decoding_ranks[index], dense_output.decoding_ranks[0])
    torch.testing.assert_close(nested_loss, dense_loss)
    assert isinstance(sampled_order_output.decoding_ranks, NestedTensor)
    _assert_causal_unit_ranks(sampled_order_output.decoding_ranks, nested_input.tied_group_ids)
    assert torch.isfinite(sampled_order_loss)


def test_collate_sequence_preserves_single_input() -> None:
    input = _protein_input(2)

    batch = collate_sequence([_sequence_sample(input)])

    assert batch["input"] is input


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
    assert batch["loss_normalizer"] == 1


def test_collate_boltzgen_ifold_mask() -> None:
    input = SimpleNamespace(
        token_mask=torch.tensor([[True, True, True, False]]),
        token_resolved_mask=torch.tensor([[True, False, True, True]]),
        design_mask=torch.tensor([[True, True, False, True]]),
    )
    target = SequenceTarget(torch.zeros(1, 4, dtype=torch.long), torch.ones(1, 4, dtype=torch.bool))

    batch = collate_boltzgen_ifold([(input, target)])

    assert batch["target"]["target"] is target
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


@pytest.mark.parametrize(
    "processor, collate, features",
    [
        (AlphaFold3Processor, collate_alphafold3_inputs, AlphaFold3TemplateFeatures),
        (ProtenixProcessor, collate_protenix_inputs, ProtenixTemplateFeatures),
    ],
    ids=["alphafold3", "protenix"],
)
@pytest.mark.parametrize("with_templates", [False, True])
def test_structure_inputs_ragged(peptide, processor, collate, features, with_templates) -> None:
    inputs = [processor().prepare_input(peptide(*components)).features for components in (("ALA", "GLY"), ("GLY",) * 3)]
    if with_templates:
        for input in inputs:
            length = input.token_mask.shape[-1]
            input.template_features = features(
                distogram=torch.randn(1, 4, length, length, 39),
                pseudo_beta_mask=torch.rand(1, 4, length, length) > 0.5,
                residue_type_ids=torch.randint(32, (1, 4, length)),
                unit_vector=torch.randn(1, 4, length, length, 3),
                backbone_frame_mask=torch.rand(1, 4, length, length) > 0.5,
                batch_size=[1],
            )
    batch = collate(inputs)
    assert [len(tokens) for tokens in batch.token_mask] == [2, 3]
    assert [len(atoms) for atoms in batch.atom_index] == [len(input.atom_index[0]) for input in inputs]
    for index, input in enumerate(inputs):
        torch.testing.assert_close(batch.reference_positions[index], input.reference_positions[0])
        if with_templates:
            for field in ("distogram", "pseudo_beta_mask", "residue_type_ids", "unit_vector", "backbone_frame_mask"):
                torch.testing.assert_close(
                    getattr(batch.template_features, field)[index], getattr(input.template_features, field)[0]
                )


def test_collate_sequence_keeps_truth_separate():
    input = _protein_input(3)
    original = input.sequence_tokens.clone()
    tokens = original.flip(-1)
    mask = torch.tensor([[True, False, True]])
    batch = collate_sequence([_sequence_sample(input, tokens=tokens, mask=mask)])

    assert batch["input"] is input
    torch.testing.assert_close(input.sequence_tokens, original)
    torch.testing.assert_close(batch["target"]["target"].tokens, tokens)
    assert batch["loss_normalizer"] == 2


def test_compact_target_device_transfer():
    from danling.data import to_device

    target = SequenceTarget(torch.tensor([[1, 2]]), torch.tensor([[True, False]]))
    moved = to_device({"target": target}, torch.device("meta"))["target"]
    assert moved.tokens.device.type == moved.mask.device.type == "meta"
    assert moved.tokens.dtype == torch.int64 and moved.mask.dtype == torch.bool
    assert target.tokens.device.type == target.mask.device.type == "cpu"
