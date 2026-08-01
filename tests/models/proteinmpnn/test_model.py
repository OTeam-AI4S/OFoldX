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

"""Behavior of ofoldx.models.proteinmpnn.model."""

import pytest
import torch

from ofoldx.data.tokenization import TOKEN_INDEX
from ofoldx.errors import InvalidRequestError
from ofoldx.models.proteinmpnn import (
    ProteinMpnnConfig,
    ProteinMpnnModel,
    ProteinMpnnProcessor,
    ProteinMpnnProcessorConfig,
)


@pytest.fixture(params=[False, True], ids=["backbone", "alpha-carbon"])
def coordinate_case(request, peptide):
    processor = ProteinMpnnProcessor(ProteinMpnnProcessorConfig(ca_only=request.param))
    input = processor.prepare_input(peptide("ALA", "GLY", "LYS")).features
    model = ProteinMpnnModel(
        ProteinMpnnConfig(
            hidden_size=8,
            num_encoder_blocks=1,
            num_decoder_blocks=1,
            num_neighbors=3,
            dropout=0.0,
            ca_only=request.param,
        )
    ).eval()
    input.residue_mask[0, -1] = False
    return model, input


def test_unavailable_backbone(coordinate_case):
    model, input = coordinate_case
    ranks = torch.arange(3).unsqueeze(0)
    expected = model(input, decoding_ranks=ranks).logits
    values = input.backbone_coordinates.detach().clone()
    values[0, -1] = float("nan")
    values.requires_grad_()
    input.backbone_coordinates = values
    actual = model(input, decoding_ranks=ranks).logits
    torch.testing.assert_close(actual, expected)
    finite = torch.where(torch.isfinite(actual), actual, 0.0)
    gradient = torch.autograd.grad(finite.square().sum(), values)[0]
    assert torch.isfinite(gradient).all()
    assert not gradient[0, -1].any()


def test_untrained_classes_excluded(coordinate_case):
    model, input = coordinate_case
    logits = model(input, decoding_ranks=torch.arange(3).unsqueeze(0)).logits
    assert torch.isneginf(logits[..., TOKEN_INDEX["OTHER"]]).all()
    assert torch.isfinite(logits[..., TOKEN_INDEX["A"]]).all()


@pytest.mark.parametrize("payload", [-1, 2**40])
def test_sequence_categories(coordinate_case, payload):
    model, input = coordinate_case
    ranks = torch.arange(3).unsqueeze(0)
    expected = model(input, decoding_ranks=ranks).logits[input.residue_mask]
    input.sequence_tokens[~input.residue_mask] = payload
    actual = model(input, decoding_ranks=ranks).logits[input.residue_mask]
    torch.testing.assert_close(actual, expected)
    input.sequence_tokens[0, 0] = payload
    with pytest.raises(InvalidRequestError, match="sequence tokens"):
        model(input, decoding_ranks=ranks)
