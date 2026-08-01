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

import math

import pytest
import torch

from ofoldx.metrics import (
    categorical_entropy_from_logits,
    categorical_expectation,
    categorical_mean,
    categorical_probability_below,
    interface_confidence_weights,
    minimum_confidence_error,
    plddt_from_logits,
    resolved_probability_from_logits,
)
from ofoldx.modules.functional import weighted_mean


def test_categorical_probability_strict_cutoff_and_vjp() -> None:
    logits = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64, requires_grad=True)
    probability = categorical_probability_below(logits, logits.new_tensor([7.0, 8.0, 9.0]), 8.0)
    torch.testing.assert_close(probability, logits.new_tensor(1 / 3))
    torch.testing.assert_close(torch.autograd.grad(probability, logits)[0], logits.new_tensor([2 / 9, -1 / 9, -1 / 9]))


@pytest.mark.parametrize("weights", [(2.0, 1.0, 0.0), (20.0, 10.0, 1.0)])
def test_invalid_partner_cannot_create_interface(weights) -> None:
    chain = torch.tensor([[1, 1, 0]])
    valid = torch.tensor([[True, True, False]])
    ligand = torch.zeros_like(valid)
    scores = torch.tensor([[20.0, 80.0, float("nan")]], dtype=torch.float64, requires_grad=True)
    for invalid_distance in (0.0, 100.0):
        distances = scores.new_tensor([[[0, 20, invalid_distance], [20, 0, 100], [invalid_distance, 100, 0]]])
        observed = interface_confidence_weights(
            distances,
            chain,
            ligand,
            valid,
            ligand_weight=weights[0],
            interface_weight=weights[1],
            background_weight=weights[2],
        )
        torch.testing.assert_close(observed, scores.new_tensor([[weights[2], weights[2], 0.0]]))
        result = weighted_mean(scores, observed)
        if weights[2] == 0:
            assert torch.isnan(result).all()
            expected_gradient = torch.zeros_like(scores)
        else:
            torch.testing.assert_close(result, scores.new_tensor([50.0]))
            expected_gradient = scores.new_tensor([[0.5, 0.5, 0.0]])
        torch.testing.assert_close(torch.autograd.grad(result.sum(), scores, retain_graph=True)[0], expected_gradient)


def test_interface_ligand_and_contact_weights_keep_their_scientific_scopes() -> None:
    distances = torch.tensor([[[0, 7, 8], [7, 0, 2], [8, 2, 0]]], dtype=torch.float64)
    weights = interface_confidence_weights(
        distances,
        torch.tensor([[0, 1, 2]]),
        torch.tensor([[False, True, False]]),
        torch.tensor([[True, True, False]]),
        ligand_weight=2.5,
        interface_weight=1.1,
    )
    torch.testing.assert_close(weights, distances.new_tensor([[1.1, 2.5, 0.0]]), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_minimum_error_has_no_finite_absence_sentinel(dtype) -> None:
    values = torch.tensor([[[4.0, 1.0], [2.0, 3.0]], [[2.0, 8.0], [6.0, 1.0]]], dtype=dtype, requires_grad=True)
    mask = torch.tensor([[[False, True], [False, False]], [[False, False], [False, False]]])
    result = minimum_confidence_error(values, mask)
    assert result.dtype == dtype and float(result[0]) == 1.0 and torch.isnan(result[1])
    expected = torch.zeros_like(values)
    expected[0, 0, 1] = 1.0
    torch.testing.assert_close(torch.autograd.grad(result.sum(), values)[0], expected)
    empty = torch.empty(2, 0, 0, dtype=dtype, requires_grad=True)
    result = minimum_confidence_error(empty, torch.empty(2, 0, 0, dtype=torch.bool))
    assert result.shape == (2,) and torch.isnan(result).all()
    torch.testing.assert_close(torch.autograd.grad(result.sum(), empty)[0], torch.zeros_like(empty))


def test_resolved_probability_uses_canonical_positive_class() -> None:
    logits = torch.tensor([[0.0, torch.log(torch.tensor(3.0)).item()]], dtype=torch.float64)
    torch.testing.assert_close(resolved_probability_from_logits(logits), logits.new_tensor([0.75]))


def test_categorical_expectation_explicit_values_and_probability_vjp() -> None:
    probabilities = torch.tensor([[0.2, 0.8], [0.7, 0.3]], dtype=torch.float64, requires_grad=True)
    centers = torch.tensor([1.0, 4.0], dtype=torch.float64, requires_grad=True)
    mean = categorical_expectation(probabilities, centers)
    torch.testing.assert_close(mean, probabilities.new_tensor([3.4, 1.9]))
    probability_gradient, center_gradient = torch.autograd.grad(mean.sum(), (probabilities, centers))
    torch.testing.assert_close(probability_gradient, centers.expand(2, 2))
    torch.testing.assert_close(center_gradient, centers.new_tensor([0.9, 1.1]))


@pytest.mark.parametrize(("start", "end"), [(0.0, 1.0), (0.0, 32.0), (0.0, 100.0), (-8.0, 8.0)])
def test_categorical_mean_units_and_gradient(start: float, end: float) -> None:
    logits = torch.tensor([0.0, 1.0, -1.0], dtype=torch.float64, requires_grad=True)
    probability = torch.tensor([1.0, torch.e, 1.0 / torch.e], dtype=torch.float64)
    probability = probability / probability.sum()
    midpoints = torch.tensor(
        [start + (end - start) * fraction for fraction in (1 / 6, 1 / 2, 5 / 6)], dtype=torch.float64
    )
    expected = (probability * midpoints).sum()
    actual = categorical_mean(logits, start=start, end=end)
    gradient = torch.autograd.grad(actual, logits)[0]

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(gradient, probability * (midpoints - expected))


def test_plddt_from_logits_equation() -> None:
    logits = torch.tensor([[0.0, 1.0, -1.0]])
    probability = logits.softmax(dim=-1)
    expected = (
        probability[..., 0] * (0.5 * 100.0 / 3.0)
        + probability[..., 1] * (1.5 * 100.0 / 3.0)
        + probability[..., 2] * (2.5 * 100.0 / 3.0)
    )

    torch.testing.assert_close(plddt_from_logits(logits), expected)


def test_categorical_entropy_missing_categories_have_zero_terms_and_gradients() -> None:
    logits = torch.tensor([0.0, -torch.inf, math.log(3.0)], dtype=torch.float64, requires_grad=True)
    actual = categorical_entropy_from_logits(logits)
    probability = logits.new_tensor([0.25, 0.75])
    expected = -(probability * probability.log()).sum()
    expected_gradient = logits.new_zeros(3)
    expected_gradient[[0, 2]] = -probability * (probability.log() + expected)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual, logits)[0], expected_gradient)


@pytest.mark.parametrize("num_bins", [0, 3])
def test_categorical_entropy_absent_support_is_undefined_with_zero_gradients(num_bins) -> None:
    logits = torch.full((2, num_bins), -torch.inf, dtype=torch.float64, requires_grad=True)
    actual = categorical_entropy_from_logits(logits)
    assert actual.shape == (2,) and torch.isnan(actual).all()
    torch.testing.assert_close(torch.autograd.grad(actual.sum(), logits)[0], torch.zeros_like(logits))


@pytest.mark.parametrize("invalid", [torch.nan, torch.inf])
def test_categorical_entropy_does_not_repair_invalid_selected_logits(invalid) -> None:
    logits = torch.tensor([0.0, invalid, -torch.inf], dtype=torch.float64)
    assert torch.isnan(categorical_entropy_from_logits(logits))
