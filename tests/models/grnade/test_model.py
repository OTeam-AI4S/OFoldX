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

"""Behavior of ofoldx.models.grnade.model."""

import pytest
import torch

from ofoldx.data.tokenization import TOKEN_INDEX
from ofoldx.models.grnade import GrnadeProcessor


@pytest.fixture
def case(rna, make_model):
    source = rna(2)
    input = GrnadeProcessor(dtype=torch.float64).prepare_input((source, source)).features
    return make_model(input.edge_scalar_features.shape[-1]), input


def test_unavailable_conformation(case):
    model, input = case
    input.conformation_mask = input.conformation_mask.clone()
    input.conformation_mask[:, -1] = False
    expected = model(input).logits
    for field in ("node_scalar_features", "node_vector_features", "edge_scalar_features", "edge_vector_features"):
        values = getattr(input, field).clone()
        values[:, -1] = float("nan")
        setattr(input, field, values)
    input.node_scalar_features.requires_grad_()
    actual = model(input).logits
    torch.testing.assert_close(actual, expected)
    finite = torch.where(torch.isfinite(actual), actual, 0.0)
    gradient = torch.autograd.grad(finite.square().sum(), input.node_scalar_features)[0]
    assert torch.isfinite(gradient).all()
    assert not gradient[:, -1].any()


def test_untrained_classes_excluded(case):
    model, input = case
    logits = model(input).logits
    assert torch.isneginf(logits[..., TOKEN_INDEX["DNA_A"]]).all()
    assert torch.isfinite(logits[..., TOKEN_INDEX["RNA_U"]]).all()


def test_fixed_rows_keep_identity(case):
    model, input = case
    input.design_mask = torch.tensor([False, True])
    sampled = model.sample(input, num_samples=4, generator=torch.Generator().manual_seed(2)).sequence_tokens
    assert (sampled[:, 0] == TOKEN_INDEX["RNA_A"]).all()
