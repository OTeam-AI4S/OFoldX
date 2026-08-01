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
from danling import NestedTensor

from ofoldx.metrics import (
    chain_pair_mean_tm_scores,
    chain_pair_tm_scores,
    directional_ipsae,
    mean_partner_tm_scores,
    predicted_tm_score,
    tm_bin_weights,
    tm_length_scale,
    tm_weight_from_pae,
)


def test_chain_pair_mean() -> None:
    values = torch.tensor([[0.1, 0.8, 0.4], [0.5, 0.2, 0.9], [0.3, 0.7, 0.6]], dtype=torch.float64, requires_grad=True)
    weights = torch.ones_like(values, requires_grad=True)
    chains = torch.tensor([2, 2, 0])
    actual = chain_pair_mean_tm_scores(values, chains, [2, 0, 5], pair_weights=weights)
    zero = values.sum() * 0 + weights.sum() * 0

    def mean(rows, columns):
        observations, coefficients = values[rows, columns], weights[rows, columns]
        return (observations * coefficients).sum() / coefficients.sum()

    expected = torch.stack(
        [
            torch.stack([mean(slice(2), slice(2)), mean(slice(2), slice(2, 3)), zero]),
            torch.stack([mean(slice(2, 3), slice(2)), mean(slice(2, 3), slice(2, 3)), zero]),
            torch.stack([zero, zero, zero]),
        ]
    )
    torch.testing.assert_close(actual, values.new_tensor([[0.4, 0.65, 0], [0.5, 0.6, 0], [0, 0, 0]]))
    cotangent = values.new_tensor([[0.2, 0.7, -0.4], [-0.3, 0.4, 0.1], [0.8, -0.2, 0.9]])
    gradients = torch.autograd.grad((actual * cotangent).sum(), (values, weights), retain_graph=True)
    reference = torch.autograd.grad((expected * cotangent).sum(), (values, weights))
    for gradient, wanted in zip(gradients, reference, strict=True):
        torch.testing.assert_close(gradient, wanted)
    best = chain_pair_tm_scores(values, chains, [2, 0])
    torch.testing.assert_close(best[0, 1], values.new_tensor(0.9))
    assert not torch.isclose(actual[0, 1], best[0, 1])


def test_partner_mean_counts_each_eligible_partner_once_and_has_correct_vjp() -> None:
    scores = torch.tensor([[0.9, 0.2, 0.8], [0.2, 0.7, 0.4], [0.8, 0.4, 0.1]], dtype=torch.float64, requires_grad=True)
    frames = torch.tensor([True, True, False])
    actual = mean_partner_tm_scores(scores, pair_mask=frames[:, None] | frames[None, :])
    torch.testing.assert_close(actual, scores.new_tensor([0.5, 0.3, 0.6]))
    gradient = torch.autograd.grad(actual[0], scores)[0]
    expected_gradient = torch.zeros_like(scores)
    expected_gradient[0, 1:] = 0.5
    torch.testing.assert_close(gradient, expected_gradient)


def test_partner_mean_excludes_nonfinite_scores_and_empty_domains() -> None:
    scores = torch.tensor([[0.9, float("nan")], [0.4, 0.7]], dtype=torch.float64, requires_grad=True)
    actual = mean_partner_tm_scores(scores)
    assert torch.isnan(actual[0])
    torch.testing.assert_close(actual[1], scores.new_tensor(0.4))
    gradient = torch.autograd.grad(actual.sum(), scores, retain_graph=True)[0]
    torch.testing.assert_close(gradient, scores.new_tensor([[0.0, 0.0], [1.0, 0.0]]))
    absent = mean_partner_tm_scores(scores, pair_mask=torch.zeros(2, 2, dtype=torch.bool))
    assert torch.isnan(absent).all()
    torch.testing.assert_close(torch.autograd.grad(absent.sum(), scores)[0], torch.zeros_like(scores))
    assert mean_partner_tm_scores(torch.empty(0, 0)).shape == (0,)


def test_chain_pair_scores_select_source_frames_and_gradients() -> None:
    values = torch.tensor(
        [[0.8, 0.8, 0.3, 0.3], [0.4, 0.4, 0.03, 0.03], [0.15, 0.15, 0.6, 0.6], [0.7, 0.7, 0.9, 0.9]],
        dtype=torch.float64,
        requires_grad=True,
    )
    chains = torch.tensor([0, 0, 1, 1])
    actual = chain_pair_tm_scores(values, chains, range(3))
    expected = values.new_tensor([[0.8, 0.3, 0.0], [0.7, 0.9, 0.0], [0.0, 0.0, 0.0]])
    torch.testing.assert_close(actual, expected)
    reference = torch.stack(
        [
            values[:2, :2].mean(-1).max(),
            values[:2, 2:].mean(-1).max(),
            values[2:, :2].mean(-1).max(),
            values[2:, 2:].mean(-1).max(),
        ]
    ).sum()
    actual_gradient = torch.autograd.grad(actual.sum(), values, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(reference, values)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient)
    # Averaging both pair axes would incorrectly report 0.165 for source0->target1.
    assert not torch.isclose(actual[0, 1], values[:2, 2:].mean())


def test_chain_pair_scores_retain_owner_weights_and_chain_identity_order() -> None:
    values = torch.tensor(
        [[[0.2, 0.8, 0.5], [0.9, 0.3, 0.7], [0.4, 0.6, 0.1]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    chain = torch.tensor([[4, 1, 4]])
    weights = torch.tensor(
        [[[0.2, 0.8, 0.0], [0.5, 0.1, 0.5], [0.4, 0.3, 0.2]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    actual = chain_pair_tm_scores(values, chain, [4, 1], pair_weights=weights)
    expected_parts = []
    for source in (4, 1):
        targets = []
        for target in (4, 1):
            rows = values[0, chain[0] == source][:, chain[0] == target]
            coefficients = weights[0, chain[0] == source][:, chain[0] == target]
            targets.append(((rows * coefficients).sum(-1) / coefficients.sum(-1)).max())
        expected_parts.append(torch.stack(targets))
    expected = torch.stack(expected_parts).unsqueeze(0)
    torch.testing.assert_close(actual, expected)
    actual_gradients = torch.autograd.grad(actual.sum(), (values, weights), retain_graph=True)
    expected_gradients = torch.autograd.grad(expected.sum(), (values, weights))
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual_gradient, expected_gradient)


@pytest.mark.parametrize("weight_scale", [1.0, 1e-8])
def test_predicted_tm_score_weighted_mean_and_vjp(weight_scale: float) -> None:
    values = torch.tensor([[0.5, 0.8, 0.2], [0.1, 0.3, 0.9]], dtype=torch.float64, requires_grad=True)
    weights = torch.tensor([[0.2, 0.8, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float64) * weight_scale
    weights.requires_grad_()
    actual = predicted_tm_score(values, pair_weights=weights)
    torch.testing.assert_close(actual, values.new_tensor(0.74))
    value_gradient, weight_gradient = torch.autograd.grad(actual, (values, weights))
    torch.testing.assert_close(value_gradient, values.new_tensor([[0.2, 0.8, 0.0], [0.0, 0.0, 0.0]]))
    torch.testing.assert_close(
        weight_gradient,
        values.new_tensor([[-0.24, 0.06, -0.54], [0.0, 0.0, 0.0]]) / weight_scale,
    )


def test_predicted_tm_score_reference_weights_only_choose_alignment() -> None:
    values = torch.tensor([[0.9, 0.9], [0.4, 0.4]], dtype=torch.float64, requires_grad=True)
    reference = torch.tensor([0.5, 1.0], dtype=torch.float64)
    score = predicted_tm_score(values, reference_weights=reference)
    torch.testing.assert_close(score, values.new_tensor(0.9))
    torch.testing.assert_close(torch.autograd.grad(score, values)[0], values.new_tensor([[0.5, 0.5], [0.0, 0.0]]))


def test_predicted_tm_score_broadcast_batch_sample_axes() -> None:
    values = torch.tensor([[[[0.2, 0.4], [0.7, 0.9]], [[0.1, 0.5], [0.2, 0.6]]]], dtype=torch.float64)
    pair_weights = torch.tensor([[1.0, 1.0], [0.0, 0.0]], dtype=torch.float64)
    torch.testing.assert_close(predicted_tm_score(values, pair_weights=pair_weights), values.new_tensor([[0.3, 0.3]]))


@pytest.mark.parametrize("weights", [torch.tensor(0.2), torch.tensor([[0.2], [0.7]])])
def test_predicted_tm_score_broadcast_weights_before_normalization(weights: torch.Tensor) -> None:
    values = torch.tensor([[0.2, 0.4], [0.7, 0.9]], dtype=torch.float64, requires_grad=True)
    score = predicted_tm_score(values, pair_weights=weights, reference_weights=torch.tensor(0.5))
    torch.testing.assert_close(score, values.new_tensor(0.8))
    torch.testing.assert_close(torch.autograd.grad(score, values)[0], values.new_tensor([[0.0, 0.0], [0.5, 0.5]]))


def test_predicted_tm_score_reference_batch_broadcast() -> None:
    values = torch.tensor([[0.2, 0.4], [0.7, 0.9]], dtype=torch.float64, requires_grad=True)
    references = torch.tensor([[1.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
    score = predicted_tm_score(values, reference_weights=references)
    torch.testing.assert_close(score, values.new_tensor([0.8, 0.3, 0.0]))
    torch.testing.assert_close(torch.autograd.grad(score.sum(), values)[0], values.new_full((2, 2), 0.5))


def test_predicted_tm_score_ragged_target_lengths_and_vjp() -> None:
    first = torch.tensor([[0.2, 0.4], [0.7, 0.9]], dtype=torch.float64, requires_grad=True)
    second = torch.tensor([[0.1, 0.3, 0.5], [0.5, 0.8, 0.8], [0.9, 0.9, 0.9]], dtype=torch.float64, requires_grad=True)
    values = NestedTensor([first, second])
    score = predicted_tm_score(values, pair_weights=values.new_tensor(0.2))
    torch.testing.assert_close(score, first.new_tensor([0.8, 0.9]))
    first_gradient, second_gradient = torch.autograd.grad(score.sum(), (first, second))
    torch.testing.assert_close(first_gradient, first.new_tensor([[0.0, 0.0], [0.5, 0.5]]))
    torch.testing.assert_close(second_gradient, second.new_tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1 / 3] * 3]))


def test_ragged_padding_is_not_candidate():
    first = torch.tensor([[0.2]], dtype=torch.float64, requires_grad=True)
    second = torch.full((2, 2), 0.6, dtype=torch.float64, requires_grad=True)
    values = NestedTensor([first, second], ragged_dims=(0, 1), padding_value=1e6)
    actual = predicted_tm_score(values)
    torch.testing.assert_close(actual, first.new_tensor([0.2, 0.6]))
    first_gradient, second_gradient = torch.autograd.grad(actual.sum(), (first, second))
    torch.testing.assert_close(first_gradient, torch.ones_like(first))
    torch.testing.assert_close(second_gradient, second.new_tensor([[0.5, 0.5], [0.0, 0.0]]))


def test_predicted_tm_score_empty_and_explicit_undefined_domains() -> None:
    values = torch.tensor([[0.2, 0.4], [0.7, 0.9]], dtype=torch.float64, requires_grad=True)
    empty = torch.zeros_like(values)
    score = predicted_tm_score(values, pair_weights=empty)
    torch.testing.assert_close(score, values.new_tensor(0.0))
    torch.testing.assert_close(torch.autograd.grad(score, values)[0], torch.zeros_like(values))
    undefined = predicted_tm_score(values, pair_weights=empty, reference_weights=torch.ones(2))
    assert undefined.isnan()
    no_frames = predicted_tm_score(values, pair_weights=empty, reference_weights=torch.zeros(2))
    torch.testing.assert_close(no_frames, values.new_tensor(0.0))
    torch.testing.assert_close(predicted_tm_score(torch.empty(3, 0, 0)), torch.zeros(3))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_predicted_tm_score_relative_coefficients_do_not_overflow(dtype) -> None:
    values = torch.full((1, 1000), 0.75, dtype=dtype, requires_grad=True)
    result = predicted_tm_score(values, pair_weights=values.new_tensor(1000.0))
    assert result.dtype == dtype
    torch.testing.assert_close(result, values.new_tensor(0.75))
    torch.testing.assert_close(torch.autograd.grad(result, values)[0], values.new_full(values.shape, 1 / 1000))


def test_predicted_tm_score_excludes_nonfinite_unobserved_pairs() -> None:
    values = torch.tensor([[0.2, float("nan")], [0.8, 0.6]], dtype=torch.float64, requires_grad=True)
    result = predicted_tm_score(values, pair_weights=values.new_tensor([[1.0, 0.0], [0.0, 0.0]]))
    torch.testing.assert_close(result, values.new_tensor(0.2))
    torch.testing.assert_close(torch.autograd.grad(result, values)[0], values.new_tensor([[1.0, 0.0], [0.0, 0.0]]))


def test_predicted_tm_score_ragged_full_reduction_and_finite_zero_coefficient_vjps() -> None:
    parts = [
        torch.tensor([[0.2, 0.9], [0.8, 0.6]], dtype=torch.float64, requires_grad=True),
        torch.tensor([[0.1, 0.3, 0.5], [0.8, 0.8, 0.8], [0.9, 0.9, 0.9]], dtype=torch.float64, requires_grad=True),
    ]
    coefficients = [
        torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float64, requires_grad=True),
        torch.tensor([[0.2, 0.3, 0.5], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float64, requires_grad=True),
    ]
    values, weights = NestedTensor(parts, ragged_dims=(0, 1)), NestedTensor(coefficients, ragged_dims=(0, 1))
    references = NestedTensor([torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0, 0.0])], ragged_dims=(0,))
    result = predicted_tm_score(values, pair_weights=weights, reference_weights=references)
    torch.testing.assert_close(result, parts[0].new_tensor([0.2, 0.36]))
    gradients = torch.autograd.grad(result.sum(), (*parts, *coefficients))
    expected = (
        [[1.0, 0.0], [0.0, 0.0]],
        [[0.2, 0.3, 0.5], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [[0.0, 0.7], [0.0, 0.0]],
        [[-0.26, -0.06, 0.14], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    )
    for gradient, oracle in zip(gradients, expected, strict=True):
        torch.testing.assert_close(gradient, gradient.new_tensor(oracle))


@pytest.mark.parametrize("reference_mode", ["default", "weighted", "none_eligible"])
def test_predicted_tm_score_mixed_empty_ragged_equation_and_vjps(reference_mode) -> None:
    lengths = (0, 2, 4)
    parts = [torch.linspace(0.1, 0.9, n * n, dtype=torch.float64).reshape(n, n).requires_grad_() for n in lengths]
    weight_parts = [
        torch.linspace(0.2, 1.0, n * n, dtype=torch.float64).reshape(n, n).requires_grad_() for n in lengths
    ]
    ref_parts = [
        (
            torch.zeros(n, dtype=torch.float64)
            if reference_mode == "none_eligible"
            else torch.linspace(1.0, 0.3, n, dtype=torch.float64)
        ).requires_grad_()
        for n in lengths
    ]
    values = NestedTensor(parts, ragged_dims=(0, 1))
    weights = NestedTensor(weight_parts, ragged_dims=(0, 1))
    references = NestedTensor(ref_parts, ragged_dims=(0,))
    actual = predicted_tm_score(
        values,
        pair_weights=weights,
        reference_weights=None if reference_mode == "default" else references,
    )

    oracle_parts = [part.detach().clone().requires_grad_() for part in parts]
    oracle_weights = [part.detach().clone().requires_grad_() for part in weight_parts]
    expected = []
    for part, weight, reference in zip(oracle_parts, oracle_weights, ref_parts, strict=True):
        if part.shape[0] == 0 or reference_mode == "none_eligible":
            expected.append(part.sum() * 0 + weight.sum() * 0)
            continue
        means = [(row * row_weight).sum() / row_weight.sum() for row, row_weight in zip(part, weight, strict=True)]
        source = max(
            range(len(means)),
            key=lambda i: float(means[i].detach() * (1 if reference_mode == "default" else reference[i].detach())),
        )
        expected.append(means[source])
    expected = torch.stack(expected)
    torch.testing.assert_close(actual, expected)
    cotangent = torch.tensor([0.4, -0.6, 1.3], dtype=torch.float64)
    actual_gradients = torch.autograd.grad(actual, (*parts, *weight_parts), cotangent)
    expected_gradients = torch.autograd.grad(expected, (*oracle_parts, *oracle_weights), cotangent)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual_gradient, expected_gradient)


def test_predicted_tm_score_mixed_empty_ragged_identity_and_undefined_target() -> None:
    values = NestedTensor([torch.empty(0, 0), torch.ones(2, 2), torch.ones(4, 4)], ragged_dims=(0, 1))
    torch.testing.assert_close(predicted_tm_score(values), torch.tensor([0.0, 1.0, 1.0]))
    zeros = torch.zeros_like(values)
    refs = NestedTensor([torch.empty(0), torch.ones(2), torch.ones(4)], ragged_dims=(0,))
    output = predicted_tm_score(values, pair_weights=zeros, reference_weights=refs)
    assert output[0] == 0
    assert torch.isnan(output[1:]).all()


def test_predicted_tm_score_empty_target_domain() -> None:
    values = torch.empty(2, 0, dtype=torch.float64, requires_grad=True)
    default = predicted_tm_score(values)
    torch.testing.assert_close(default, torch.tensor(0.0, dtype=torch.float64))
    torch.testing.assert_close(torch.autograd.grad(default, values)[0], torch.empty_like(values))
    assert predicted_tm_score(values, reference_weights=torch.ones(2)).isnan()

    nested = NestedTensor([values, torch.full((3, 2), 0.7, dtype=torch.float64)], ragged_dims=(0, 1))
    torch.testing.assert_close(predicted_tm_score(nested), torch.tensor([0.0, 0.7], dtype=torch.float64))


def test_predicted_tm_score_tie_selects_first_eligible_reference() -> None:
    values = torch.tensor([[0.0, 0.0], [0.0, 0.0]], dtype=torch.float64, requires_grad=True)
    score = predicted_tm_score(values, reference_weights=torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(torch.autograd.grad(score, values)[0], values.new_tensor([[0.0, 0.0], [0.5, 0.5]]))


def test_tm_equations() -> None:
    clamped = 1.24 * (19.0 - 15.0) ** (1.0 / 3.0) - 1.8
    assert tm_length_scale(10.0) == clamped

    counts = torch.tensor([10.0, 64.0])
    scales = 1.24 * (counts.clamp(min=19.0) - 15.0) ** (1.0 / 3.0) - 1.8
    torch.testing.assert_close(tm_length_scale(counts), scales)

    centers = torch.tensor([0.0, 1.0, 4.0])
    expected = 1.0 / (1.0 + (centers / scales[:, None]) ** 2)
    torch.testing.assert_close(tm_bin_weights(centers, counts), expected)

    logits = torch.tensor([[0.0, 1.0, -1.0], [2.0, -1.0, 0.5]])
    expected_from_pae = (logits.softmax(dim=-1) * expected).sum(dim=-1)
    torch.testing.assert_close(
        tm_weight_from_pae(logits, counts, bin_centers=centers),
        expected_from_pae,
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_directional_ipsae_mask_count_and_gradient(reverse: bool) -> None:
    generator = torch.Generator().manual_seed(71)
    pae = (torch.rand(40, 40, generator=generator, dtype=torch.float64) * 12).requires_grad_()
    source = torch.arange(40) >= 38
    target = torch.arange(40) < 37
    if reverse:
        source, target = target, source
    frame = torch.arange(40) != 38
    token = torch.arange(40) != 36
    cutoff = 10.0
    values = pae.detach().tolist()
    rows = []
    row_targets = []
    scales = []
    for i in range(40):
        selected = [
            j
            for j in range(40)
            if source[i] and frame[i] and token[i] and target[j] and token[j] and values[i][j] < cutoff
        ]
        scale = max(1.0, 1.24 * math.pow(max(len(selected), 19) - 15, 1 / 3) - 1.8)
        rows.append(sum(1 / (1 + (values[i][j] / scale) ** 2) for j in selected) / len(selected) if selected else 0.0)
        row_targets.append(selected)
        scales.append(scale)
    winner = max(range(40), key=rows.__getitem__)
    expected_gradient = torch.zeros_like(pae)
    for j in row_targets[winner]:
        value, scale = values[winner][j], scales[winner]
        expected_gradient[winner, j] = (
            -2 * value / scale**2 / (1 + (value / scale) ** 2) ** 2 / len(row_targets[winner])
        )

    score = directional_ipsae(
        pae, source_mask=source, target_mask=target, frame_mask=frame, token_mask=token, cutoff=cutoff
    )
    torch.testing.assert_close(score, pae.new_tensor(rows[winner]))
    torch.testing.assert_close(torch.autograd.grad(score, pae)[0], expected_gradient)


def test_directional_ipsae_unobserved_source() -> None:
    pae = torch.ones(2, 3, 3)
    selected = torch.ones(2, 3, dtype=torch.bool)
    result = directional_ipsae(
        pae,
        source_mask=selected,
        target_mask=selected,
        frame_mask=torch.zeros_like(selected),
        token_mask=selected,
    )
    torch.testing.assert_close(result, torch.zeros(2))


def test_empty_source_retains_broadcasted_score_batches_and_dtype() -> None:
    terms = torch.empty(0, 3, dtype=torch.float32, requires_grad=True)
    weights = torch.ones(4, 1, 3, dtype=torch.float64, requires_grad=True)
    result = predicted_tm_score(terms, pair_weights=weights)
    torch.testing.assert_close(result, torch.zeros(4, dtype=torch.float64))
    term_gradient, weight_gradient = torch.autograd.grad(result.sum(), (terms, weights))
    torch.testing.assert_close(term_gradient, torch.zeros_like(terms))
    torch.testing.assert_close(weight_gradient, torch.zeros_like(weights))

    references = torch.empty(4, 0)
    result = predicted_tm_score(terms, reference_weights=references)
    torch.testing.assert_close(result, torch.zeros(4))
