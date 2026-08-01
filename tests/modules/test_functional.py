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

import pytest
import torch
from danling.tensors import NestedTensor
from torch._dynamo.testing import CompileCounter
from torch._subclasses.fake_tensor import FakeTensorMode, is_fake
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from ofoldx.modules.functional import (
    banded_pair_mask,
    grouped_softmax,
    indexed_sum,
    pair_mask_from_msa_mask,
    pair_mask_from_token_mask,
    pairwise_distance,
    project_categories,
    project_counts,
    weighted_mean,
)


def _indexed_sum_reference(values, index, num_groups, dim):
    leading = values.movedim(dim, 0)
    return torch.stack([leading[index == group].sum(dim=0) for group in range(num_groups)]).movedim(0, dim)


def test_mixed_nested_weight_gradients():
    parts = [torch.tensor([2.0, 5.0], requires_grad=True), torch.tensor([7.0, 11.0, 13.0], requires_grad=True)]
    weights = torch.tensor([[1.0, 0.0, 42.0], [0.0, 1.0, 1.0]], requires_grad=True)
    result = weighted_mean(NestedTensor(parts, ragged_dims=(0,)), weights)
    first_gradient, second_gradient, weight_gradient = torch.autograd.grad(result.sum(), (*parts, weights))
    torch.testing.assert_close(result, torch.tensor([2.0, 12.0]))
    torch.testing.assert_close(first_gradient, torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(second_gradient, torch.tensor([0.0, 0.5, 0.5]))
    torch.testing.assert_close(weight_gradient, torch.tensor([[0.0, 3.0, 0.0], [-2.5, -0.5, 0.5]]))


def _grouped_softmax_reference(values, index, num_groups, dim):
    leading = values.movedim(dim, 0)
    output = leading * 0
    for group in range(num_groups):
        selected = index == group
        output[selected] = leading[selected].softmax(dim=0)
    return output.movedim(0, dim)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_queries, num_keys", [(4, 4), (4, 3), (30, 32), (0, 3), (4, 0), (0, 0)])
def test_pairwise_distance_values_and_both_coordinate_vjps(dtype, num_queries, num_keys):
    query_values = torch.arange(2 * num_queries * 3, dtype=torch.float64).reshape(2, num_queries, 3) / 4
    key_values = torch.arange(num_keys * 3, dtype=torch.float64).reshape(1, num_keys, 3) / 2 + 1
    query = query_values.to(dtype).requires_grad_()
    key = key_values.to(dtype).requires_grad_()
    expected_query = query.detach().double().requires_grad_()
    expected_key = key.detach().double().requires_grad_()
    actual = pairwise_distance(query, key)
    expected = torch.linalg.vector_norm(expected_query.unsqueeze(-2) - expected_key.unsqueeze(-3), dim=-1)
    assert actual.shape == (2, num_queries, num_keys)
    assert actual.dtype == dtype
    tolerance = {torch.float64: 1e-12, torch.float32: 2e-5, torch.float16: 5e-3, torch.bfloat16: 4e-2}[dtype]
    torch.testing.assert_close(actual.double(), expected, rtol=tolerance, atol=tolerance)
    cotangent = torch.linspace(-0.7, 0.9, actual.numel(), dtype=torch.float64).reshape(actual.shape)
    actual_gradients = torch.autograd.grad(actual, (query, key), cotangent.to(dtype))
    expected_gradients = torch.autograd.grad(expected, (expected_query, expected_key), cotangent.to(dtype).double())
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual_gradient.double(), expected_gradient, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
def test_pairwise_distance_coincident_points_have_zero_coordinate_subgradient(dtype):
    coordinates = torch.zeros(2, 30, 3, dtype=dtype, requires_grad=True)
    distances = pairwise_distance(coordinates)
    assert torch.count_nonzero(distances) == 0
    gradient = torch.autograd.grad(distances.sum(), coordinates)[0]
    assert torch.count_nonzero(gradient) == 0
    assert torch.isfinite(gradient).all()


@pytest.mark.parametrize("dtype, delta", [(torch.float32, 0.25), (torch.float64, 1e-8)])
@pytest.mark.parametrize("offset", [-1, 0, 1])
def test_pairwise_distance_preserves_translated_physical_boundary_and_vjps(dtype, delta, offset):
    left = torch.zeros(30, 3, dtype=dtype)
    left[:, 0] = torch.arange(30, dtype=dtype) * 1000
    right = left + torch.tensor([15.0 + offset * delta, 0.0, 0.0], dtype=dtype)
    translation = torch.full((3,), 1e6, dtype=dtype)
    query = (left + translation).requires_grad_()
    key = (right + translation).requires_grad_()
    expected_query = query.detach().clone().requires_grad_()
    expected_key = key.detach().clone().requires_grad_()
    actual = pairwise_distance(query, key)
    expected = torch.linalg.vector_norm(expected_query[:, None] - expected_key[None, :], dim=-1)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual.diagonal() <= 15.0, torch.full((30,), offset <= 0))
    weights = torch.eye(30, dtype=dtype)
    actual_gradients = torch.autograd.grad(actual, (query, key), weights)
    expected_gradients = torch.autograd.grad(expected, (expected_query, expected_key), weights)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual_gradient, expected_gradient)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for caller AMP controls")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("amp_dtype", [torch.float16, torch.bfloat16])
def test_pairwise_distance_cuda_amp_values_and_coordinate_vjps(dtype, amp_dtype):
    query_values = torch.arange(12, device="cuda", dtype=dtype).reshape(4, 3) / 4
    key_values = torch.arange(9, device="cuda", dtype=dtype).reshape(3, 3) / 2 + 1
    query = query_values.requires_grad_()
    key = key_values.requires_grad_()
    expected_query = query.detach().double().requires_grad_()
    expected_key = key.detach().double().requires_grad_()
    with torch.autocast("cuda", dtype=amp_dtype):
        actual = pairwise_distance(query, key)
    expected = torch.linalg.vector_norm(expected_query[:, None] - expected_key[None, :], dim=-1)
    tolerance = {torch.float64: 1e-12, torch.float32: 2e-5, torch.float16: 5e-3, torch.bfloat16: 4e-2}[dtype]
    torch.testing.assert_close(actual.double(), expected, rtol=tolerance, atol=tolerance)
    weights = torch.linspace(-0.7, 0.9, actual.numel(), device="cuda", dtype=actual.dtype).reshape(actual.shape)
    actual_gradients = torch.autograd.grad(actual, (query, key), weights)
    expected_gradients = torch.autograd.grad(expected, (expected_query, expected_key), weights.double())
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual_gradient.double(), expected_gradient, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize(
    ("shape", "dim", "groups"),
    [
        ((4, 2, 3), 0, [2, 0, 2, 1]),
        ((2, 4, 3), 1, [2, 0, 2, 1]),
        ((2, 3, 4), -1, [2, 0, 2, 1]),
        # Equal extents must not let an index attach to the wrong axis.
        ((2, 2, 3), 1, [1, 0]),
    ],
)
@pytest.mark.parametrize(
    ("operation", "reference"),
    [(indexed_sum, _indexed_sum_reference), (grouped_softmax, _grouped_softmax_reference)],
)
def test_indexed_reductions_follow_the_declared_axis(shape, dim, groups, operation, reference):
    index = torch.tensor(groups)
    values = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float64).reshape(shape).sin()
    actual_input = values.clone().requires_grad_()
    reference_input = values.clone().requires_grad_()
    actual = operation(actual_input, index, num_groups=4, dim=dim)
    expected = reference(reference_input, index, 4, dim)
    torch.testing.assert_close(actual, expected)
    cotangent = torch.linspace(-0.7, 0.9, actual.numel(), dtype=actual.dtype).reshape(actual.shape)
    actual_gradient = torch.autograd.grad((actual * cotangent).sum(), actual_input)[0]
    expected_gradient = torch.autograd.grad((expected * cotangent).sum(), reference_input)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient)


@pytest.mark.parametrize(
    ("operation", "reference"),
    [(indexed_sum, _indexed_sum_reference), (grouped_softmax, _grouped_softmax_reference)],
)
def test_indexed_reductions_accept_noncontiguous_slices(operation, reference):
    values = torch.randn(4, 3, 2, dtype=torch.float64).transpose(0, 1).requires_grad_()
    index = torch.tensor([3, 0, 3, 1])
    actual = operation(values, index, num_groups=5, dim=1)
    expected = reference(values, index, 5, 1)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual.square().sum(), values, retain_graph=True)[0],
        torch.autograd.grad(expected.square().sum(), values)[0],
    )


@pytest.mark.parametrize("dim", [0, 1, -1])
def test_indexed_reductions_preserve_empty_input_and_groups(dim):
    shape = [2, 3, 4]
    shape[dim] = 0
    values = torch.empty(shape, dtype=torch.float64, requires_grad=True)
    index = torch.empty(0, dtype=torch.long)
    sums = indexed_sum(values, index, num_groups=5, dim=dim)
    expected_shape = shape.copy()
    expected_shape[dim] = 5
    assert sums.shape == tuple(expected_shape)
    assert torch.equal(sums, torch.zeros_like(sums))
    probabilities = grouped_softmax(values, index, num_groups=5, dim=dim)
    assert probabilities.shape == values.shape
    gradient = torch.autograd.grad(sums.sum() + probabilities.sum(), values)[0]
    assert gradient.shape == values.shape
    assert indexed_sum(values, index, num_groups=0, dim=dim).shape == values.shape
    assert grouped_softmax(values, index, num_groups=0, dim=dim).shape == values.shape


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_grouped_softmax_normalizes_stable_logits_and_caller_dropout(dtype):
    index = torch.tensor([2, 0, 2, 0, 3])
    logits = torch.tensor([[1000, -1000], [-900, 900], [1001, -999], [-899, 899], [7, -4]], dtype=dtype)
    # The caller keeps one finite incoming edge per group and head.
    dropped = torch.tensor([[True, False], [False, True], [False, False], [False, False], [False, False]])
    logits = logits.masked_fill(dropped, -torch.inf)
    actual = grouped_softmax(logits, index, num_groups=5)
    torch.testing.assert_close(actual, _grouped_softmax_reference(logits, index, 5, 0))
    assert torch.isfinite(actual).all()
    assert torch.equal(actual[dropped], torch.zeros_like(actual[dropped]))
    sums = indexed_sum(actual, index, num_groups=5)
    torch.testing.assert_close(sums[[0, 2, 3]], torch.ones(3, 2, dtype=dtype))
    assert torch.equal(sums[[1, 4]], torch.zeros(2, 2, dtype=dtype))


def test_indexed_sum_preserves_integer_counts():
    values = torch.tensor([[1, 0], [0, 1], [1, 1]], dtype=torch.int64)
    result = indexed_sum(values, torch.tensor([1, 0, 1]), num_groups=3)
    assert result.dtype == values.dtype
    assert torch.equal(result, torch.tensor([[0, 1], [2, 1], [0, 0]], dtype=torch.int64))


def test_token_pair_mask() -> None:
    token_mask = torch.tensor([[True, False, True]])
    expected = torch.tensor([[[True, False, True], [False, False, False], [True, False, True]]])
    assert torch.equal(pair_mask_from_token_mask(token_mask), expected)


def test_msa_pair_mask() -> None:
    # an MSA mask collapses its rows (logical-or) before the outer mask, so two rows == token [F, T, T]
    msa_mask = torch.tensor([[[False, True, False], [False, False, True]]])
    pair_mask = pair_mask_from_msa_mask(msa_mask)
    assert pair_mask.shape == (1, 3, 3)
    assert torch.equal(pair_mask, pair_mask_from_token_mask(torch.tensor([[False, True, True]])))


def test_banded_pair_mask() -> None:
    band = banded_pair_mask(torch.zeros(1, 5), 1)

    assert band.shape == (5, 5)
    assert torch.equal(band, torch.ones(5, 5, dtype=torch.bool).tril(1).triu(-1))
    assert torch.equal(banded_pair_mask(torch.zeros(1, 4), 0), torch.eye(4, dtype=torch.bool))


def test_banded_pair_mask_rejects_negative_window() -> None:
    with pytest.raises(ValueError, match="half_window"):
        banded_pair_mask(torch.zeros(1, 3), -1)


def test_nested_banded_pair_mask_stays_packed() -> None:
    lengths = (2, 5)
    axis = NestedTensor(tuple(torch.zeros(length) for length in lengths), ragged_dims=(0,))

    band = banded_pair_mask(axis, 1)

    assert isinstance(band, NestedTensor)
    assert band.ragged_dims == (0, 1)
    assert band.packed_offsets().tolist() == [0, 4, 29]
    # Each element's band is bounded by its own length, so no pair crosses a batch element.
    expected = torch.cat(
        tuple(torch.ones(length, length, dtype=torch.bool).tril(1).triu(-1).reshape(-1) for length in lengths)
    )
    assert torch.equal(band.concat, expected)


def test_nested_token_pair_mask_stays_packed() -> None:
    parts = (
        torch.tensor([True, False]),
        torch.tensor([False, True, True]),
    )
    token_mask = NestedTensor(parts, ragged_dims=(0,))

    pair_mask = pair_mask_from_token_mask(token_mask)

    assert isinstance(pair_mask, NestedTensor)
    assert pair_mask.ragged_dims == (0, 1)
    assert pair_mask.packed_dim_order == (0, 1)
    assert pair_mask.packed_offsets().tolist() == [0, 4, 13]
    expected = torch.cat(tuple((part[:, None] & part[None, :]).reshape(-1) for part in parts))
    assert torch.equal(pair_mask.concat, expected)


def test_nested_token_pair_mask_external_fake() -> None:
    template = NestedTensor([torch.empty(2, dtype=torch.bool), torch.empty(3, dtype=torch.bool)], ragged_dims=(0,))
    shape_env = ShapeEnv(allow_dynamic_output_shape_ops=True)
    mode = FakeTensorMode(shape_env=shape_env)
    with mode:
        token_mask = template.packed_like(torch.empty(5, device="cuda", dtype=torch.bool))
        pair_mask = pair_mask_from_token_mask(token_mask)

    assert isinstance(pair_mask, NestedTensor)
    assert is_fake(pair_mask.concat)
    assert pair_mask.concat.dim() == 1
    assert pair_mask.ragged_dims == (0, 1)
    assert pair_mask.packed_dim_order == (0, 1)


def test_nested_token_pair_mask_rejects_noncanonical_layout() -> None:
    token_mask = NestedTensor(
        [torch.ones(2, 3, dtype=torch.bool), torch.ones(2, 4, dtype=torch.bool)],
        ragged_dims=(1,),
    )
    with pytest.raises(ValueError, match="explicit canonical"):
        pair_mask_from_token_mask(token_mask)


def test_nested_token_pair_mask_preserves_empty_elements() -> None:
    token_mask = NestedTensor(
        [torch.empty(0, dtype=torch.bool), torch.tensor([True, False])],
        ragged_dims=(0,),
    )
    pair_mask = pair_mask_from_token_mask(token_mask)
    assert pair_mask.packed_offsets().tolist() == [0, 0, 4]
    assert torch.equal(pair_mask.concat, torch.tensor([True, False, False, False]))


def test_nested_token_pair_mask_reuses_one_dynamic_fullgraph() -> None:
    counter = CompileCounter()

    def run(token_mask):
        pair_mask = pair_mask_from_token_mask(token_mask)
        return pair_mask.concat, pair_mask.packed_offsets()

    compiled = torch.compile(run, backend=counter, fullgraph=True, dynamic=True)
    for parts in (
        (torch.tensor([True, False]), torch.tensor([False, True, True])),
        (torch.tensor([True, False, True]), torch.tensor([True, True, False, True, False])),
    ):
        token_mask = NestedTensor(parts, ragged_dims=(0,))
        values, offsets = compiled(token_mask)
        expected = torch.cat(tuple((part[:, None] & part[None, :]).reshape(-1) for part in parts))
        assert torch.equal(values, expected)
        assert offsets.tolist() == [0, *(sum(part.numel() ** 2 for part in parts[:index]) for index in range(1, 3))]
    assert counter.frame_count == 1


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize(("count", "weight"), [(1000, 1.0), (100, 20.0), (100, 1000.0)])
def test_weighted_mean_representable_mean_avoids_overflowing_sum(dtype, count, weight) -> None:
    scores = torch.full((count,), 80.0, dtype=dtype, requires_grad=True)
    weights = torch.full((count,), weight, dtype=dtype)
    result = weighted_mean(scores, weights)
    assert result.dtype == dtype
    torch.testing.assert_close(result, scores.new_tensor(80.0))
    torch.testing.assert_close(torch.autograd.grad(result, scores)[0], scores.new_full((count,), 1 / count))


@pytest.mark.parametrize("weight_dtype", [torch.bool, torch.int64])
def test_weighted_mean_integer_mask_keeps_value_dtype(weight_dtype) -> None:
    scores = torch.tensor([20.0, 80.0, float("nan")], dtype=torch.float16, requires_grad=True)
    weights = torch.tensor([1, 1, 0], dtype=weight_dtype)
    result = weighted_mean(scores, weights)
    assert result.dtype == scores.dtype
    torch.testing.assert_close(result, scores.new_tensor(50.0))
    torch.testing.assert_close(torch.autograd.grad(result, scores)[0], scores.new_tensor([0.5, 0.5, 0.0]))


def test_weighted_mean_finite_zero_coefficient_keeps_its_one_sided_derivative() -> None:
    scores = torch.tensor([20.0, 80.0], dtype=torch.float64, requires_grad=True)
    weights = torch.tensor([1.0, 0.0], dtype=torch.float64, requires_grad=True)
    result = weighted_mean(scores, weights)
    score_vjp, coefficient_vjp = torch.autograd.grad(result, (scores, weights))
    torch.testing.assert_close(result, scores.new_tensor(20.0))
    torch.testing.assert_close(score_vjp, scores.new_tensor([1.0, 0.0]))
    torch.testing.assert_close(coefficient_vjp, scores.new_tensor([0.0, 60.0]))
    epsilon = 1e-7
    perturbed = weighted_mean(scores.detach(), weights.detach() + weights.new_tensor([0.0, epsilon]))
    torch.testing.assert_close((perturbed - result.detach()) / epsilon, scores.new_tensor(60.0), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("second_weight", [20, 1000])
def test_weighted_mean_ragged_true_length_mean_and_empty_scope(dtype, second_weight) -> None:
    parts = [torch.full((count,), 80.0, dtype=dtype, requires_grad=True) for count in (1000, 100, 0)]
    weight_parts = [
        torch.full((count,), weight, dtype=dtype) for count, weight in ((1000, 1), (100, second_weight), (0, 1))
    ]
    scores, weights = NestedTensor(parts), NestedTensor(weight_parts)
    result = weighted_mean(scores, weights)
    assert result.dtype == dtype
    torch.testing.assert_close(result, parts[0].new_tensor([80.0, 80.0, float("nan")]), equal_nan=True)
    gradients = torch.autograd.grad(result.sum(), parts)
    for count, gradient in zip((1000, 100, 0), gradients, strict=True):
        torch.testing.assert_close(gradient, gradient.new_full((count,), 1 / count if count else 0))


def test_weighted_mean_ragged_values_and_finite_coefficient_vjps() -> None:
    values = [
        torch.tensor([20.0, 80.0], dtype=torch.float64, requires_grad=True),
        torch.tensor([10.0, 20.0, 80.0], dtype=torch.float64, requires_grad=True),
    ]
    coefficients = [
        torch.tensor([1.0, 0.0], dtype=torch.float64, requires_grad=True),
        torch.tensor([0.1, 0.2, 0.7], dtype=torch.float64, requires_grad=True),
    ]
    packed_values, packed_coefficients = NestedTensor(values), NestedTensor(coefficients)
    result = weighted_mean(packed_values, packed_coefficients)
    torch.testing.assert_close(result, values[0].new_tensor([20.0, 61.0]))
    gradients = torch.autograd.grad(result.sum(), (*values, *coefficients))
    expected = ([1.0, 0.0], [0.1, 0.2, 0.7], [0.0, 60.0], [-51.0, -41.0, 19.0])
    for gradient, oracle in zip(gradients, expected, strict=True):
        torch.testing.assert_close(gradient, gradient.new_tensor(oracle))


@pytest.mark.parametrize("excluded", [float("nan"), float("inf"), -float("inf")])
def test_weighted_mean_ragged_nonobservation_payload_has_zero_value_gradient(excluded) -> None:
    values = [
        torch.tensor([20.0, excluded], dtype=torch.float64, requires_grad=True),
        torch.tensor([80.0], dtype=torch.float64, requires_grad=True),
    ]
    selected = NestedTensor([torch.tensor([True, False]), torch.tensor([True])])
    scores = NestedTensor(values)
    result = weighted_mean(scores, selected)
    torch.testing.assert_close(result, values[0].new_tensor([20.0, 80.0]))
    gradients = torch.autograd.grad(result.sum(), values)
    torch.testing.assert_close(gradients[0], values[0].new_tensor([1.0, 0.0]))
    torch.testing.assert_close(gradients[1], values[1].new_tensor([1.0]))


@pytest.mark.parametrize("layout_kind", ["sampled_atoms", "pair_targets", "sampled_pairs"])
def test_packed_mean_fullgraph_gradients(layout_kind):
    generator = torch.Generator().manual_seed(903)
    if layout_kind == "sampled_atoms":
        shapes, ragged = ((2, 3, 2), (2, 3, 4)), (2,)
    elif layout_kind == "sampled_pairs":
        shapes, ragged = ((2, 2, 2), (2, 4, 4)), (1, 2)
    else:
        shapes, ragged = ((2, 2), (4, 4)), (0, 1)
    values = [torch.randn(shape, dtype=torch.float64, generator=generator).requires_grad_() for shape in shapes]
    coefficients = [torch.rand(shape, dtype=torch.float64, generator=generator) + 0.1 for shape in shapes]
    for coefficient in coefficients:
        coefficient[..., 0] = 0
        coefficient.requires_grad_()
    value_layout = NestedTensor(values, ragged_dims=ragged)
    coefficient_layout = NestedTensor(coefficients, ragged_dims=ragged)

    def run(value_buffer, coefficient_buffer, value_layout, coefficient_layout):
        result = weighted_mean(
            value_layout.packed_like(value_buffer), coefficient_layout.packed_like(coefficient_buffer)
        )
        return result.concat if isinstance(result, NestedTensor) else result

    compiled = torch.compile(run, backend="aot_eager", fullgraph=True, dynamic=True)
    actual = compiled(value_layout.concat, coefficient_layout.concat, value_layout, coefficient_layout)
    dense_means = [
        (value * weight).sum(-1) / weight.sum(-1) for value, weight in zip(values, coefficients, strict=True)
    ]
    if layout_kind == "sampled_atoms":
        expected = torch.stack(dense_means)
    elif layout_kind == "sampled_pairs":
        expected = torch.cat([mean.movedim(-1, 0) for mean in dense_means])
    else:
        expected = torch.cat(dense_means)
    torch.testing.assert_close(actual, expected)
    actual_gradients = torch.autograd.grad(actual.sum(), (*values, *coefficients))
    expected_gradients = torch.autograd.grad(expected.sum(), (*values, *coefficients))
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual_gradient, expected_gradient)


@pytest.mark.parametrize("payload", [-1.0, 0.5, 10000.0, float("nan"), float("inf")])
def test_categorical_availability(payload):
    values = torch.tensor([[2.0, payload]], dtype=torch.float64)
    result = project_categories(values, torch.tensor([[True, False]]), num_classes=3)
    assert result.dtype == torch.int64
    assert result.tolist() == [[2, 0]]
    with pytest.raises(ValueError):
        project_categories(values, torch.ones_like(values, dtype=torch.bool), num_classes=3)


def test_count_width_and_availability():
    counts = torch.tensor([[100000.0, float("nan")]], dtype=torch.float64, requires_grad=True)
    selected = project_counts(counts, torch.tensor([[True, False]]))
    assert selected.dtype == torch.float64
    value = torch.atan(selected / 3).to(torch.float16)
    assert torch.isfinite(value).all()
    gradient = torch.autograd.grad(value.sum(), counts)[0]
    assert torch.isfinite(gradient).all()
    assert gradient[0, 1] == 0
    for invalid in (-1.0, 0.5, float("inf")):
        with pytest.raises(ValueError):
            project_counts(torch.tensor([invalid]), torch.tensor([True]))


def test_ragged_msa_projection_fullgraph():
    def project(layout, values, columns):
        raw = layout.packed_like(values)
        available = columns.unsqueeze(1)
        return project_categories(raw, available, num_classes=3).concat, project_counts(raw, available).concat

    compiled = torch.compile(project, backend="aot_eager", fullgraph=True, dynamic=True)
    for rows, widths in (((2, 3), (4, 2)), ((4, 5), (3, 6))):
        masks = [torch.arange(width) != width - 1 for width in widths]
        values = [
            torch.where(mask[None].expand(row, -1), 1.0, torch.nan).double().requires_grad_()
            for row, mask in zip(rows, masks, strict=True)
        ]
        layout = NestedTensor(values, ragged_dims=(0, 1))
        columns = NestedTensor(masks, ragged_dims=(0,))
        categories, counts = compiled(layout, layout.concat, columns)
        expected = torch.cat(
            [
                torch.where(mask[None].expand(row, -1), value, 0).reshape(-1)
                for row, mask, value in zip(rows, masks, values, strict=True)
            ]
        )
        torch.testing.assert_close(categories, expected.long())
        torch.testing.assert_close(counts, expected)
        gradients = torch.autograd.grad(counts.sum(), values)
        for gradient, row, mask in zip(gradients, rows, masks, strict=True):
            torch.testing.assert_close(gradient, mask[None].expand(row, -1).double())
