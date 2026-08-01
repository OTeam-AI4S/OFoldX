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

from typing import NamedTuple, cast

import pytest
import torch
from danling.tensors import NestedTensor

from ofoldx.modules.diffusion.loss import bond_loss, edm_denoising_loss


class _Prediction(NamedTuple):
    coordinates: torch.Tensor
    augmented_target_coordinates: torch.Tensor
    noise_level: torch.Tensor


def _nested_prediction(
    atom_lengths: tuple[int, ...],
) -> tuple[_Prediction, NestedTensor]:
    samples = 2
    generator = torch.Generator().manual_seed(7)
    targets = NestedTensor(
        [torch.randn(samples, atoms, 3, generator=generator) for atoms in atom_lengths],
        ragged_dims=(1,),
    )
    coordinates = NestedTensor(
        [target + 0.1 * torch.randn(target.shape, generator=generator) for target in targets],
        ragged_dims=(1,),
    )
    resolved = NestedTensor(
        [torch.arange(atoms).remainder(3) != 2 for atoms in atom_lengths],
        ragged_dims=(0,),
    )
    return (
        _Prediction(
            coordinates=coordinates,
            augmented_target_coordinates=targets,
            noise_level=torch.tensor([[0.5, 1.0], [0.75, 1.25]]),
        ),
        resolved,
    )


def test_edm_denoising_loss_ragged_matches_per_structure_dense() -> None:
    prediction, resolved = _nested_prediction((4, 7))

    actual = edm_denoising_loss(prediction, resolved, sigma_data=16.0)

    expected = torch.stack(
        [
            edm_denoising_loss(
                _Prediction(
                    coordinates=coordinates.unsqueeze(0),
                    augmented_target_coordinates=target.unsqueeze(0),
                    noise_level=prediction.noise_level[index : index + 1],
                ),
                mask.unsqueeze(0),
                sigma_data=16.0,
            )
            for index, (coordinates, target, mask) in enumerate(
                zip(prediction.coordinates, prediction.augmented_target_coordinates, resolved, strict=True)
            )
        ]
    ).mean()

    torch.testing.assert_close(actual, expected)


def test_edm_denoising_loss_ragged_fullgraph_backward() -> None:
    template, resolved = _nested_prediction((4, 7))
    coordinate_layout = cast(NestedTensor, template.coordinates)
    eager_values = coordinate_layout.concat.detach().requires_grad_()
    compiled_values = eager_values.detach().clone().requires_grad_()

    def loss(
        coordinate_values: torch.Tensor,
        coordinate_layout: NestedTensor,
        targets: NestedTensor,
        noise_level: torch.Tensor,
        mask: NestedTensor,
    ) -> torch.Tensor:
        prediction = _Prediction(
            coordinates=coordinate_layout.packed_like(coordinate_values),
            augmented_target_coordinates=targets,
            noise_level=noise_level,
        )
        return edm_denoising_loss(prediction, mask, sigma_data=16.0)

    arguments = (
        coordinate_layout,
        template.augmented_target_coordinates,
        template.noise_level,
        resolved,
    )
    eager_loss = loss(eager_values, *arguments)
    compiled = torch.compile(loss, backend="aot_eager", fullgraph=True, dynamic=True)
    compiled_loss = compiled(compiled_values, *arguments)
    compiled_loss.backward()
    eager_loss.backward()

    torch.testing.assert_close(compiled_loss, eager_loss)
    torch.testing.assert_close(compiled_values.grad, eager_values.grad)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("target_samples", [1, 2])
def test_bond_loss_native_dtype_equation_and_gradients(dtype: torch.dtype, target_samples: int) -> None:
    prediction = torch.tensor(
        [[[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 3.0, 0.0]], [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [3.0, 4.0, 0.0]]]],
        dtype=dtype,
        requires_grad=True,
    )
    target = (
        torch.tensor([[[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 2.0, 0.0]]]], dtype=dtype)
        .repeat(1, target_samples, 1, 1)
        .requires_grad_()
    )
    endpoints = torch.tensor([[[0, 1], [1, 2]]])
    mask = torch.ones(1, 2, dtype=torch.bool)
    actual = bond_loss(prediction, target, endpoints, mask)
    assert actual.dtype == dtype
    expected = prediction.new_tensor([[1.0, 4.0]])
    torch.testing.assert_close(actual, expected)
    prediction_gradient, target_gradient = torch.autograd.grad(actual.sum(), (prediction, target))
    expected_prediction_gradient = torch.tensor(
        [
            [
                [[-1.0, 0.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]],
                [[-2.0, 0.0, 0.0], [2.0, -2.0, 0.0], [0.0, 2.0, 0.0]],
            ]
        ],
        dtype=dtype,
    )
    expected_target_gradient = -expected_prediction_gradient
    if target_samples == 1:
        expected_target_gradient = expected_target_gradient.sum(dim=1, keepdim=True)
    torch.testing.assert_close(prediction_gradient, expected_prediction_gradient)
    torch.testing.assert_close(target_gradient, expected_target_gradient)


@pytest.mark.parametrize("empty", [False, True])
def test_bond_loss_excludes_unknown_geometry_and_negative_endpoints(empty: bool) -> None:
    prediction = torch.tensor(
        [[[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [float("nan"), float("nan"), float("nan")]]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    target = prediction.detach().clone()
    target[:, :, 1, 0] = 1.0
    target.requires_grad_()
    endpoints = torch.tensor([[[0, 1], [0, 2], [-1, -1]]])
    mask = torch.tensor([[not empty, False, False]])
    actual = bond_loss(prediction, target, endpoints, mask)
    if empty:
        assert torch.isnan(actual).all()
    else:
        torch.testing.assert_close(actual, torch.ones_like(actual))
    gradients = torch.autograd.grad(actual.sum(), (prediction, target))
    for gradient in gradients:
        assert torch.isfinite(gradient).all()
        torch.testing.assert_close(gradient[:, :, 2], torch.zeros_like(gradient[:, :, 2]))
        if empty:
            torch.testing.assert_close(gradient, torch.zeros_like(gradient))


def test_edm_loss_excluded_coordinates_have_zero_gradients() -> None:
    generator = torch.Generator().manual_seed(31)
    target = torch.randn(1, 1, 5, 3, generator=generator)
    coordinates = (target + 0.2 * torch.randn(1, 1, 5, 3, generator=generator)).requires_grad_()
    resolved = torch.tensor([[True, True, True, True, False]])
    with torch.no_grad():
        target[:, :, 4] = torch.nan
        coordinates[:, :, 4] = torch.nan
    actual = edm_denoising_loss(_Prediction(coordinates, target, torch.ones(1, 1)), resolved, sigma_data=16.0)
    assert torch.isfinite(actual)
    (gradient,) = torch.autograd.grad(actual, (coordinates,))
    assert torch.isfinite(gradient).all()
    torch.testing.assert_close(gradient[:, :, 4], torch.zeros_like(gradient[:, :, 4]))


def test_edm_loss_empty_scope_has_zero_gradients() -> None:
    coordinates = torch.full((1, 1, 4, 3), torch.nan, requires_grad=True)
    target = torch.full_like(coordinates, torch.nan)
    resolved = torch.zeros(1, 4, dtype=torch.bool)
    actual = edm_denoising_loss(_Prediction(coordinates, target, torch.ones(1, 1)), resolved, sigma_data=16.0)
    assert torch.isnan(actual)
    (gradient,) = torch.autograd.grad(actual, (coordinates,))
    torch.testing.assert_close(gradient, torch.zeros_like(gradient))
