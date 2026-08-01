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

import pytest
import torch
from danling import NestedTensor

from ofoldx.metrics import sequence_isoelectric_point, sequence_nll


def test_sequence_nll_broadcasts_source_over_candidate_axes() -> None:
    values = torch.tensor([[-0.2, -1.7], [-2.0, -0.1]], dtype=torch.float64, requires_grad=True)
    tokens = torch.tensor([[0, 1], [1, 0]])
    observed = sequence_nll(values, tokens, torch.ones(2, dtype=torch.bool))
    torch.testing.assert_close(observed, values.new_tensor([0.15, 1.85]))
    torch.testing.assert_close(torch.autograd.grad(observed.sum(), values)[0], values.new_full((2, 2), -0.5))


def test_sequence_nll_ragged_mean_and_vjp() -> None:
    first = torch.tensor([[-0.2, -1.7], [-2.0, -0.1]], dtype=torch.float64, requires_grad=True)
    second = torch.tensor([[-0.3, -1.7], [-2.0, -0.1], [-0.6, -1.7]], dtype=torch.float64, requires_grad=True)
    values = NestedTensor([first, second])
    tokens = NestedTensor([torch.tensor([0, 1]), torch.tensor([0, 1, 0])])
    mask = NestedTensor([torch.tensor([True, True]), torch.tensor([True, False, True])])
    observed = sequence_nll(values, tokens, mask)
    torch.testing.assert_close(observed, first.new_tensor([0.15, 0.45]))
    first_gradient, second_gradient = torch.autograd.grad(observed.sum(), (first, second))
    torch.testing.assert_close(first_gradient, first.new_tensor([[-0.5, 0.0], [0.0, -0.5]]))
    torch.testing.assert_close(second_gradient, second.new_tensor([[-0.5, 0.0], [0.0, 0.0], [-0.5, 0.0]]))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_sequence_nll_representable_mean_does_not_overflow(dtype) -> None:
    logits = torch.tensor([-80.0, 0.0], dtype=dtype).expand(1000, 2).clone().requires_grad_()
    result = sequence_nll(
        logits.log_softmax(-1), torch.zeros(1000, dtype=torch.long), torch.ones(1000, dtype=torch.bool)
    )
    assert result.dtype == dtype
    torch.testing.assert_close(result, logits.new_tensor(80.0))
    torch.testing.assert_close(
        torch.autograd.grad(result, logits)[0], logits.new_tensor([-1 / 1000, 1 / 1000]).expand_as(logits)
    )


@pytest.mark.parametrize("ignored_token", [-1, 2])
def test_sequence_nll_excludes_nonfinite_unobserved_payloads(ignored_token) -> None:
    values = torch.tensor([[-0.2, -1.7], [float("nan"), -0.1]], dtype=torch.float64, requires_grad=True)
    result = sequence_nll(values, torch.tensor([0, ignored_token]), torch.tensor([True, False]))
    torch.testing.assert_close(result, values.new_tensor(0.2))
    torch.testing.assert_close(torch.autograd.grad(result, values)[0], values.new_tensor([[-1.0, 0.0], [0.0, 0.0]]))


def test_sequence_nll_all_unobserved_scope_has_zero_log_probability_gradient() -> None:
    values = torch.tensor([[-0.2, -1.7], [float("nan"), -0.1]], dtype=torch.float64, requires_grad=True)
    result = sequence_nll(values, torch.tensor([0, 0]), torch.tensor([False, False]))
    assert result.isnan()
    torch.testing.assert_close(torch.autograd.grad(result, values)[0], torch.zeros_like(values))


@pytest.mark.parametrize("sequence", ["D" * 100, "R" * 100, "ACDEFGHIKLMNPQRSTVWY"])
def test_isoelectric_point_solves_charge_beyond_default_bracket(sequence: str) -> None:
    from Bio.SeqUtils.IsoelectricPoint import IsoelectricPoint

    charge_model = IsoelectricPoint(sequence)
    predicted_pi = sequence_isoelectric_point(sequence)
    assert abs(charge_model.charge_at_pH(predicted_pi)) < 1e-4
    if sequence == "D" * 100:
        assert predicted_pi < 4.05
    elif sequence == "R" * 100:
        assert predicted_pi > 12.0
