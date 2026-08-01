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

"""Behavior of ofoldx.models.nampnn.model."""

import pytest
import torch

from ofoldx.data.tokenization import TOKEN_INDEX
from ofoldx.models.nampnn import NaMpnnProcessor


@pytest.fixture
def case(model, complex_structure):
    input = NaMpnnProcessor().prepare_input(complex_structure).features
    input.residue_mask[0, -1] = False
    return model, input


RANKS = torch.arange(4).unsqueeze(0)


def test_unavailable_backbone(case):
    model, input = case
    expected = model(input, decoding_ranks=RANKS).logits
    values = input.atom_coordinates.detach().clone()
    values[0, -1] = float("nan")
    values.requires_grad_()
    input.atom_coordinates = values
    actual = model(input, decoding_ranks=RANKS).logits
    torch.testing.assert_close(actual, expected)
    finite = torch.where(torch.isfinite(actual), actual, 0.0)
    gradient = torch.autograd.grad(finite.square().sum(), values)[0]
    assert torch.isfinite(gradient).all()
    assert not gradient[0, -1].any()


def test_untrained_classes_excluded(case):
    model, input = case
    logits = model(input, decoding_ranks=RANKS).logits
    assert torch.isneginf(logits[..., TOKEN_INDEX["OTHER"]]).all()
    assert torch.isfinite(logits[..., TOKEN_INDEX["RNA_U"]]).all()


@pytest.mark.parametrize("payload", [-1, 2**40])
def test_unavailable_categories(case, payload):
    model, input = case
    expected = model(input, decoding_ranks=RANKS).logits[input.residue_mask]
    input.sequence_tokens[~input.residue_mask] = payload
    input.polymer_type_index[~input.residue_mask] = payload
    actual = model(input, decoding_ranks=RANKS).logits[input.residue_mask]
    torch.testing.assert_close(actual, expected)


def test_sample_respects_support(case):
    model, input = case
    sampled = model.sample(input, num_samples=3, generator=torch.Generator().manual_seed(5)).sequence_tokens
    rows = input.residue_mask[0]
    allowed = input.allowed_token_mask[0, rows].expand(3, -1, -1)
    assert allowed.gather(-1, sampled[:, rows].unsqueeze(-1)).all()
