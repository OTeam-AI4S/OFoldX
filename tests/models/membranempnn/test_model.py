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

"""Behavior of ofoldx.models.membranempnn.model."""

from __future__ import annotations

import pytest
import torch

from ofoldx.data.molecular import ComponentRef
from ofoldx.errors import InvalidRequestError
from ofoldx.models.conditions import TiedResidueIdentities

RANKS = torch.arange(3).unsqueeze(0)


@pytest.fixture
def case(designer, peptide):
    input = designer.processor.prepare_input(peptide("ALA", "GLY", "LYS"), membrane_labels=1).features
    input.residue_mask[0, -1] = False
    return designer.model, input


@pytest.mark.parametrize("payload", [-1, 2**40])
def test_membrane_categories(case, payload):
    model, input = case
    expected = model(input, decoding_ranks=RANKS).logits[input.residue_mask]
    input.membrane_labels[~input.residue_mask] = payload
    actual = model(input, decoding_ranks=RANKS).logits[input.residue_mask]
    torch.testing.assert_close(actual, expected)
    input.membrane_labels[0, 0] = payload
    with pytest.raises(InvalidRequestError, match="membrane labels"):
        model(input, decoding_ranks=RANKS)


def test_labels_required(case):
    model, input = case
    input.membrane_labels = None
    with pytest.raises(InvalidRequestError, match="membrane_labels"):
        model(input, decoding_ranks=RANKS)


@pytest.mark.parametrize("weight", [0.0, -1.0])
def test_degenerate_tied_weight(designer, peptide, weight):
    pair = (ComponentRef("A", 0), ComponentRef("A", 1))
    input = designer.processor.prepare_input(
        peptide("ALA", "GLY"), conditions=(TiedResidueIdentities((pair,)),), tied_weights={pair[0]: weight}
    ).features
    output = designer.model.sample(input, decoding_ranks=torch.zeros(1, 2, dtype=torch.long))
    assert torch.isfinite(output.sampling_probabilities).all()


def test_model_global_rejects_wrong_or_nonuniform_categories(case):
    model, input = case
    model.config.label_mode = "global"
    model(input, decoding_ranks=RANKS)
    input.membrane_labels[0, 0] = 2
    with pytest.raises(InvalidRequestError, match="membrane labels"):
        model(input, decoding_ranks=RANKS)
    input.membrane_labels[0, 0] = 0
    with pytest.raises(InvalidRequestError, match="global labels must be uniform"):
        model(input, decoding_ranks=RANKS)


def test_global_ragged_batch_accepts_different_per_structure_labels(peptide):
    from ofoldx.models.membranempnn import (
        MembraneMpnnConfig,
        MembraneMpnnModel,
        MembraneMpnnProcessor,
        MembraneMpnnProcessorConfig,
    )
    from ofoldx.runners.collate import collate_mpnn

    processor = MembraneMpnnProcessor(MembraneMpnnProcessorConfig(label_mode="global"))
    inputs = [
        processor.prepare_input(peptide("ALA", "GLY"), membrane_labels=0).features,
        processor.prepare_input(peptide("ALA", "GLY", "LYS"), membrane_labels=1).features,
    ]
    batch = collate_mpnn(inputs)
    config = MembraneMpnnConfig(
        label_mode="global", hidden_size=8, num_encoder_blocks=1, num_decoder_blocks=1, num_neighbors=2
    )
    model = MembraneMpnnModel(config).eval().set_backend("eager")
    output = model(batch, decoding_ranks=batch.tied_group_ids)
    assert len(output.logits) == 2


def test_global_invalid_ragged_batch_rejected(peptide):
    from danling.tensors import NestedTensor

    from ofoldx.models.membranempnn import (
        MembraneMpnnConfig,
        MembraneMpnnModel,
        MembraneMpnnProcessor,
        MembraneMpnnProcessorConfig,
    )
    from ofoldx.runners.collate import collate_mpnn

    processor = MembraneMpnnProcessor(MembraneMpnnProcessorConfig(label_mode="global"))
    inputs = [
        processor.prepare_input(peptide("ALA", "GLY"), membrane_labels=0).features,
        processor.prepare_input(peptide("ALA", "GLY", "LYS"), membrane_labels=1).features,
    ]
    batch = collate_mpnn(inputs)
    batch.membrane_labels = NestedTensor([torch.tensor([0, 0]), torch.tensor([1, 0, 1])], ragged_dims=(0,))
    model = (
        MembraneMpnnModel(
            MembraneMpnnConfig(
                label_mode="global", hidden_size=8, num_encoder_blocks=1, num_decoder_blocks=1, num_neighbors=2
            )
        )
        .eval()
        .set_backend("eager")
    )
    with pytest.raises(InvalidRequestError, match="global labels must be uniform"):
        model(batch, decoding_ranks=batch.tied_group_ids)


def test_global_model_compiles_fullgraph(case):
    model, input = case
    model.config.label_mode = "global"
    model.set_backend("eager")
    expected = model.encode(input)
    compiled = torch.compile(model.encode, backend="eager", fullgraph=True)
    actual = compiled(input)
    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, want)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_global_ragged_cuda_matches_per_structure_and_gradients(peptide):
    from ofoldx.data.tokenization import TOKEN_INDEX
    from ofoldx.models.membranempnn import (
        MembraneMpnnConfig,
        MembraneMpnnModel,
        MembraneMpnnProcessor,
        MembraneMpnnProcessorConfig,
    )
    from ofoldx.runners.collate import collate_mpnn

    processor = MembraneMpnnProcessor(MembraneMpnnProcessorConfig(label_mode="global"), device="cuda")
    inputs = [
        processor.prepare_input(peptide("ALA", "GLY"), membrane_labels=0).features,
        processor.prepare_input(peptide("ALA", "GLY", "LYS"), membrane_labels=1).features,
    ]
    model = (
        MembraneMpnnModel(
            MembraneMpnnConfig(
                label_mode="global", hidden_size=8, num_encoder_blocks=1, num_decoder_blocks=1, num_neighbors=2
            )
        )
        .to("cuda")
        .eval()
        .set_backend("eager")
    )
    dense = [model(item, decoding_ranks=item.tied_group_ids).logits for item in inputs]
    batch = collate_mpnn(inputs)
    logits = model(batch, decoding_ranks=batch.tied_group_ids).logits
    for index, expected in enumerate(dense):
        torch.testing.assert_close(logits[index], expected[0])
    objective = sum(row[:, TOKEN_INDEX["A"]].sum() for row in logits)
    (gradient,) = torch.autograd.grad(objective, model.node_embedding.weight)
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
