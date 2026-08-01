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
from danling.tensors import NestedTensor

from ofoldx.modules.backend import masked_knn


def test_masked_knn_dense_contract() -> None:
    coordinates = torch.tensor([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [5.0, 0.0, 0.0]]])
    mask = torch.tensor([[True, False, True]])

    distances, indices, valid = masked_knn(coordinates, mask, 3)

    assert indices.tolist() == [[[0, 2, 0], [0, 0, 0], [2, 0, 0]]]
    assert valid.tolist() == [[[True, True, False], [False, False, False], [True, True, False]]]
    torch.testing.assert_close(
        distances,
        torch.tensor([[[0.001, 5.0, 6.0], [1.0, 1.0, 1.0], [0.001, 5.0, 6.0]]]),
        atol=1e-6,
        rtol=1e-6,
    )


def test_masked_knn_preserves_float64_geometry_and_zero_norm_subgradient() -> None:
    coordinates = torch.tensor(
        [[[1e8, 0, 0], [1e8 + 0.25, 0, 0], [1e8 + 1, 0, 0]]], dtype=torch.float64, requires_grad=True
    )
    distances, indices, valid = masked_knn(coordinates, torch.ones(1, 3, dtype=torch.bool), 2, eps=0)
    assert indices.tolist() == [[[0, 1], [1, 0], [2, 1]]]
    assert valid.all()
    torch.testing.assert_close(distances, coordinates.new_tensor([[[0, 0.25], [0, 0.25], [0, 0.75]]]))
    torch.testing.assert_close(
        torch.autograd.grad(distances.sum(), coordinates)[0],
        coordinates.new_tensor([[[-2, 0, 0], [1, 0, 0], [1, 0, 0]]]),
    )


def test_masked_knn_empty_domain_keeps_zero_coordinate_gradient() -> None:
    coordinates = torch.empty(2, 0, 3, dtype=torch.float64, requires_grad=True)
    distances, indices, valid = masked_knn(coordinates, torch.empty(2, 0, dtype=torch.bool), 3, eps=0)
    assert distances.shape == indices.shape == valid.shape == (2, 0, 3)
    torch.testing.assert_close(torch.autograd.grad(distances.sum(), coordinates)[0], torch.zeros_like(coordinates))


def test_masked_knn_dense_counts_follow_requested_width() -> None:
    coordinates = torch.zeros(1, 2, 3, requires_grad=True)
    mask = torch.ones(1, 2, dtype=torch.bool)
    empty, _, _ = masked_knn(coordinates, mask, 0, eps=0)
    assert empty.shape == (1, 2, 0)
    torch.testing.assert_close(torch.autograd.grad(empty.sum(), coordinates)[0], torch.zeros_like(coordinates))
    distances, indices, valid = masked_knn(coordinates, mask, 129, eps=0)
    assert distances.shape == indices.shape == valid.shape == (1, 2, 129)
    assert valid[..., :2].all() and not valid[..., 2:].any()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_masked_knn_orders_physical_distances_before_storage_rounding(dtype: torch.dtype) -> None:
    coordinates = torch.tensor(
        [[[0, 0, 0], [1, 0.02, 0], [1, 0, 0], [-1, 0, 0], [float("nan"), 0, 0]]],
        dtype=dtype,
        requires_grad=True,
    )
    mask = torch.tensor([[True, True, True, True, False]])
    distances, indices, valid = masked_knn(coordinates, mask, 6, eps=1e-6, backend="eager")

    reference = coordinates.detach().double().requires_grad_()
    physical = reference.masked_fill(~mask.unsqueeze(-1), 0)
    squared = (physical.unsqueeze(-2) - physical.unsqueeze(-3)).square().sum(dim=-1)
    pair_valid = mask.unsqueeze(-1) & mask.unsqueeze(-2)
    ordered = squared.masked_fill(~pair_valid, torch.inf).argsort(dim=-1, stable=True)
    expected_valid = pair_valid.gather(-1, ordered)
    expected_indices = ordered.masked_fill(~expected_valid, 0)
    torch.testing.assert_close(indices[..., :5], expected_indices)
    torch.testing.assert_close(valid[..., :5], expected_valid)
    assert indices[0, 0].tolist() == [0, 2, 3, 1, 0, 0]
    assert not valid[..., -1].any()

    lengths = (squared + 1e-6).sqrt()
    filler = (lengths.masked_fill(~pair_valid, 0).amax(dim=-1) + 1).detach().unsqueeze(-1)
    expected_distances = torch.where(expected_valid, lengths.gather(-1, ordered), filler)
    torch.testing.assert_close(distances[..., :5], expected_distances.to(dtype))
    torch.testing.assert_close(distances[..., 5:], filler.to(dtype))
    weights = torch.linspace(-0.5, 0.5, 25, dtype=dtype).reshape(1, 5, 5)
    expected_gradient = torch.autograd.grad((expected_distances * weights.double()).sum(), reference)[0]
    gradient = torch.autograd.grad((distances[..., :5] * weights).sum(), coordinates)[0]
    torch.testing.assert_close(gradient, expected_gradient.to(dtype))


@pytest.mark.parametrize("scale", [1e-200, 1e200])
def test_masked_knn_float64_selection_retains_finite_norm_range(scale: float) -> None:
    coordinates = torch.tensor([[[0, 0, 0], [2 * scale, 0, 0], [scale, 0, 0]]], dtype=torch.float64)
    distances, indices, valid = masked_knn(coordinates, torch.ones(1, 3, dtype=torch.bool), 2, eps=0, backend="eager")
    assert indices[0, 0].tolist() == [0, 2]
    assert valid.all()
    torch.testing.assert_close(distances[0, 0], coordinates.new_tensor([0, scale]), atol=0, rtol=1e-15)


@pytest.mark.parametrize("backend", ["auto", "eager"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("num_neighbors", [0, 2, 129])
def test_masked_knn_nested_cpu_uses_complete_native_contract(
    backend: str, dtype: torch.dtype, num_neighbors: int
) -> None:
    samples = [
        torch.tensor([[0, 0, 0], [1, 0.02, 0], [1, 0, 0]], dtype=dtype, requires_grad=True),
        torch.empty(0, 3, dtype=dtype, requires_grad=True),
        torch.tensor([[3, 4, 0]], dtype=dtype, requires_grad=True),
    ]
    masks = [torch.ones(sample.shape[0], dtype=torch.bool) for sample in samples]
    nested = NestedTensor(samples, ragged_dims=(0,))
    distances, indices, valid = masked_knn(
        nested, NestedTensor(masks, ragged_dims=(0,)), num_neighbors, eps=0, backend=backend
    )
    for sample, mask, distance, index, validity in zip(samples, masks, distances, indices, valid, strict=True):
        expected = masked_knn(sample, mask, num_neighbors, eps=0, backend="eager")
        for observed, reference in zip((distance, index, validity), expected, strict=True):
            torch.testing.assert_close(observed, reference)
    gradients = torch.autograd.grad(distances.concat.sum(), samples)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
