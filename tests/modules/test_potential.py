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

import math

import pytest
import torch

from ofoldx.modules.potential import (
    chain_center_separation,
    flat_bottom_radial_potential,
    indexed_distance,
    indexed_distance_with_gradient,
    indexed_signed_dihedral,
    indexed_signed_dihedral_with_gradient,
    linear_flat_bottom_derivative,
    linear_flat_bottom_energy,
    smooth_coordination_potential,
    substrate_interaction_potential,
)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("prefix", [(), (3,), (2, 3)])
@pytest.mark.parametrize("translation", [0.0, 12.0])
@pytest.mark.parametrize("reverse_pairs", [False, True])
def test_chain_center_separation_equation_and_vjps(dtype, prefix, translation, reverse_pairs) -> None:
    atom_chain = torch.tensor([3, 3, 0, 0, 0, 5, 5, 5, 5, 9])
    atom_mask = torch.tensor([True] * 9 + [False])
    chain_pairs = torch.tensor([[3, 0, 3], [0, 5, 0]])
    if reverse_pairs:
        chain_pairs = chain_pairs.flip(0)
    coordinates = torch.zeros(*prefix, 10, 3, dtype=dtype)
    coordinates[..., :9, 0] = torch.tensor([0, 2, 3, 5, 7, 8, 10, 12, 14], dtype=dtype)
    coordinate_set = torch.arange(math.prod(prefix), dtype=dtype).reshape(prefix)
    coordinates[..., 2:5, 0] += coordinate_set[..., None] * 0.5
    coordinates[..., :9, :] += translation
    coordinates[..., 9, :] = torch.tensor([torch.nan, torch.inf, -torch.inf], dtype=dtype)
    coordinates.requires_grad_()
    lower_bound = torch.tensor([6.0, 7.0, 6.0], dtype=dtype, requires_grad=True)
    slope = torch.tensor([1.0, 2.0, 0.5], dtype=dtype, requires_grad=True)

    expected_terms = []
    for pair, bound, coefficient in zip(chain_pairs.T, lower_bound, slope):
        first = coordinates[..., atom_mask & (atom_chain == pair[0]), :].mean(dim=-2)
        second = coordinates[..., atom_mask & (atom_chain == pair[1]), :].mean(dim=-2)
        distance = torch.linalg.vector_norm(first - second, dim=-1)
        expected_terms.append(torch.where(distance < bound, coefficient * (bound - distance), 0.0))
    expected = torch.stack(expected_terms, dim=-1).sum(dim=-1)
    actual = chain_center_separation(
        coordinates, atom_chain, atom_mask, 10, chain_pairs, lower_bound, slope, compute_gradient=False
    )
    with_gradient, manual_gradient = chain_center_separation(
        coordinates, atom_chain, atom_mask, 10, chain_pairs, lower_bound, slope, compute_gradient=True
    )
    upstream = 1.0 + coordinate_set * 0.25
    operands = (coordinates, lower_bound, slope)
    expected_vjps = torch.autograd.grad((expected * upstream).sum(), operands, retain_graph=True)
    actual_vjps = torch.autograd.grad((actual * upstream).sum(), operands)
    tolerance = {"rtol": 2e-11, "atol": 2e-12} if dtype == torch.float64 else {}

    assert actual.shape == prefix
    assert manual_gradient.shape == coordinates.shape
    assert actual.dtype == dtype and manual_gradient.dtype == dtype
    torch.testing.assert_close(actual, expected, **tolerance)
    torch.testing.assert_close(with_gradient, expected, **tolerance)
    for actual_vjp, expected_vjp in zip(actual_vjps, expected_vjps):
        torch.testing.assert_close(actual_vjp, expected_vjp, **tolerance)
    torch.testing.assert_close(manual_gradient * upstream[..., None, None], expected_vjps[0], **tolerance)
    torch.testing.assert_close(manual_gradient[..., 9, :], torch.zeros_like(manual_gradient[..., 9, :]))


def test_chain_center_separation_coincidence_and_closed_boundary() -> None:
    distances = torch.tensor([0.0, 2.0, 4.0, 6.0], dtype=torch.float64)
    coordinates = torch.zeros(4, 4, 3, dtype=torch.float64)
    coordinates[:, 2:, 0] = distances[:, None]
    coordinates.requires_grad_()
    lower_bound = torch.tensor(4.0, dtype=torch.float64, requires_grad=True)
    slope = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    energy, gradient = chain_center_separation(
        coordinates,
        torch.tensor([0, 0, 1, 1]),
        torch.ones(4, dtype=torch.bool),
        2,
        torch.tensor([[0], [1]]),
        lower_bound,
        slope,
        compute_gradient=True,
    )
    expected_gradient = torch.zeros_like(coordinates)
    expected_gradient[1, :2, 0] = 1.0
    expected_gradient[1, 2:, 0] = -1.0
    coordinate_vjp, bound_vjp, slope_vjp = torch.autograd.grad(energy.sum(), (coordinates, lower_bound, slope))
    torch.testing.assert_close(energy, torch.tensor([8.0, 4.0, 0.0, 0.0], dtype=torch.float64))
    torch.testing.assert_close(gradient, expected_gradient)
    torch.testing.assert_close(coordinate_vjp, expected_gradient)
    torch.testing.assert_close(bound_vjp, torch.tensor(4.0, dtype=torch.float64))
    torch.testing.assert_close(slope_vjp, torch.tensor(6.0, dtype=torch.float64))


@pytest.mark.parametrize("prefix", [(), (3,), (2, 3)])
@pytest.mark.parametrize("retained", [False, True])
@pytest.mark.parametrize("compute_gradient", [False, True])
def test_chain_center_separation_empty_domain(prefix, retained, compute_gradient) -> None:
    coordinates = torch.zeros(*prefix, 4, 3, dtype=torch.float64)
    coordinates[..., :, :] = torch.tensor([torch.nan, torch.inf, -torch.inf], dtype=torch.float64)
    if retained:
        coordinates[..., :2, :] = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float64)
    coordinates.requires_grad_()
    lower_bound = torch.tensor(4.0, dtype=torch.float64, requires_grad=True)
    slope = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    arguments = (
        coordinates,
        torch.tensor([0, 0, 9, 9] if retained else [9, 9, 9, 9]),
        torch.tensor([retained, retained, False, False]),
        1 if retained else 0,
        torch.empty(2, 0, dtype=torch.long),
        lower_bound,
        slope,
    )
    result = chain_center_separation(*arguments, compute_gradient=compute_gradient)
    if compute_gradient:
        energy, gradient = result
        torch.testing.assert_close(gradient, torch.zeros_like(coordinates))
    else:
        energy = result
    coordinate_vjp, bound_vjp, slope_vjp = torch.autograd.grad(energy.sum(), (coordinates, lower_bound, slope))
    torch.testing.assert_close(energy, torch.zeros(prefix, dtype=torch.float64))
    torch.testing.assert_close(coordinate_vjp, torch.zeros_like(coordinates))
    torch.testing.assert_close(bound_vjp, torch.zeros_like(lower_bound))
    torch.testing.assert_close(slope_vjp, torch.zeros_like(slope))


def test_indexed_distance() -> None:
    coordinates = torch.tensor([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]], dtype=torch.float64)
    index = torch.tensor([[0], [1]])

    distance = indexed_distance(coordinates, index)

    torch.testing.assert_close(distance, torch.tensor([5.0], dtype=torch.float64))


def test_distance_gradient() -> None:
    coordinates = torch.tensor(
        [[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    index = torch.tensor([[0], [1]])

    distance, gradient = indexed_distance_with_gradient(coordinates, index)
    distance.sum().backward()

    torch.testing.assert_close(gradient[:, 0], coordinates.grad)


def _dihedral_coordinates(*, requires_grad: bool = False) -> torch.Tensor:
    return torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=torch.float64,
        requires_grad=requires_grad,
    )


def test_indexed_dihedral() -> None:
    coordinates = _dihedral_coordinates()
    index = torch.tensor([[0], [1], [2], [3]])

    dihedral = indexed_signed_dihedral(coordinates, index)

    torch.testing.assert_close(dihedral, torch.tensor([-math.pi / 2], dtype=torch.float64))


def test_dihedral_gradient() -> None:
    coordinates = _dihedral_coordinates(requires_grad=True)
    index = torch.tensor([[0], [1], [2], [3]])

    dihedral, gradient = indexed_signed_dihedral_with_gradient(coordinates, index)
    dihedral.sum().backward()

    torch.testing.assert_close(gradient[:, 0], coordinates.grad)


@pytest.mark.parametrize("central_length", [1.0, 1e-2, 1e-4, 1e-6])
@pytest.mark.parametrize("eps", [0.0, 1e-8, 1e-4])
def test_stabilized_dihedral_all_endpoint_gradients(central_length: float, eps: float) -> None:
    coordinates = torch.tensor(
        [[0.6, 0.3, 0.2], [0.0, 0.0, 0.0], [0.0, central_length, 0.0], [0.5, central_length + 0.1, 0.7]],
        dtype=torch.float64,
        requires_grad=True,
    )
    index = torch.arange(4).reshape(4, 1)
    actual, gradient = indexed_signed_dihedral_with_gradient(coordinates, index, eps=eps)
    expected = indexed_signed_dihedral(coordinates, index, eps=eps)
    expected_gradient = torch.autograd.grad(expected.sum(), coordinates)[0]
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(gradient[:, 0], expected_gradient, rtol=2e-10, atol=2e-8)


@pytest.mark.parametrize("central_length", [1e-4, 1e-6])
def test_stabilized_dihedral_finite_difference(central_length: float) -> None:
    coordinates = torch.tensor(
        [[0.6, 0.3, 0.2], [0.0, 0.0, 0.0], [0.0, central_length, 0.0], [0.5, central_length + 0.1, 0.7]],
        dtype=torch.float64,
    )
    index = torch.arange(4).reshape(4, 1)
    _, gradient = indexed_signed_dihedral_with_gradient(coordinates, index)
    step = central_length * 1e-4
    finite_difference = torch.zeros_like(coordinates)
    for atom in range(4):
        for dimension in range(3):
            perturbation = torch.zeros_like(coordinates)
            perturbation[atom, dimension] = step
            above = indexed_signed_dihedral(coordinates + perturbation, index)
            below = indexed_signed_dihedral(coordinates - perturbation, index)
            finite_difference[atom, dimension] = (above - below)[0] / (2 * step)
    torch.testing.assert_close(gradient[:, 0], finite_difference, rtol=1e-5, atol=2e-6)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.float16, torch.bfloat16])
def test_dihedral_dtype_and_inference_gradient(dtype: torch.dtype) -> None:
    coordinates = _dihedral_coordinates().to(dtype=dtype).requires_grad_()
    index = torch.arange(4).reshape(4, 1)
    expected = indexed_signed_dihedral(coordinates, index)
    expected_gradient = torch.autograd.grad(expected.sum(), coordinates)[0]
    with torch.inference_mode():
        actual, gradient = indexed_signed_dihedral_with_gradient(coordinates, index)
    assert actual.dtype == dtype and gradient.dtype == dtype
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(gradient[:, 0], expected_gradient)


def test_undefined_dihedral_retains_undefined_derivative() -> None:
    coordinates = torch.zeros(4, 3, dtype=torch.float64)
    index = torch.arange(4).reshape(4, 1)
    actual, gradient = indexed_signed_dihedral_with_gradient(coordinates, index)
    torch.testing.assert_close(actual, indexed_signed_dihedral(coordinates, index))
    assert torch.isnan(gradient).all()


def test_flat_bottom_energy() -> None:
    value = torch.tensor([-1.0, 0.0, 1.0, 2.0, 3.0])
    slope = torch.tensor(2.0)
    lower_bound = torch.tensor(0.0)
    upper_bound = torch.tensor(2.0)

    energy = linear_flat_bottom_energy(value, slope, lower_bound, upper_bound)

    torch.testing.assert_close(energy, torch.tensor([2.0, 0.0, 0.0, 0.0, 2.0]))


def test_flat_bottom_derivative() -> None:
    value = torch.tensor([-1.0, 0.0, 1.0, 2.0, 3.0], requires_grad=True)
    slope = torch.tensor(2.0)
    lower_bound = torch.tensor(0.0)
    upper_bound = torch.tensor(2.0)

    energy = linear_flat_bottom_energy(value, slope, lower_bound, upper_bound)
    derivative = linear_flat_bottom_derivative(value, slope, lower_bound, upper_bound)
    energy.sum().backward()

    torch.testing.assert_close(derivative, value.grad)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize(
    "values, lower, upper, expected_energy, expected_value_vjp, expected_parameter_vjps",
    [
        ([-1, 0, 1, 2, 3], 0, 2, [2, 0, 0, 0, 2], [-2, 0, 0, 0, 2], [2, 2, -2]),
        ([0, 1, 2], 1, math.inf, [2, 0, 0], [-2, 0, 0], [1, 2, 0]),
        ([2, 3, 4], -math.inf, 3, [0, 0, 2], [0, 0, 2], [1, 0, -2]),
        ([-1, 0, 1], -math.inf, math.inf, [0, 0, 0], [0, 0, 0], [0, 0, 0]),
        ([0.5, 1.5], 0, 2, [0, 0], [0, 0], [0, 0, 0]),
        ([0, 2], 0, 2, [0, 0], [0, 0], [0, 0, 0]),
    ],
    ids=["two-sided", "lower-only", "upper-only", "unbounded", "inside", "boundary"],
)
def test_flat_bottom_selected_arithmetic_all_operand_vjps(
    dtype, values, lower, upper, expected_energy, expected_value_vjp, expected_parameter_vjps
) -> None:
    value = torch.tensor(values, dtype=dtype, requires_grad=True)
    slope = torch.tensor(2.0, dtype=dtype, requires_grad=True)
    lower_bound = torch.tensor(lower, dtype=dtype, requires_grad=True)
    upper_bound = torch.tensor(upper, dtype=dtype, requires_grad=True)
    energy = linear_flat_bottom_energy(value, slope, lower_bound, upper_bound)
    derivative = linear_flat_bottom_derivative(value, slope, lower_bound, upper_bound)
    value_vjp, slope_vjp, lower_vjp, upper_vjp = torch.autograd.grad(
        energy.sum(), (value, slope, lower_bound, upper_bound)
    )
    assert energy.dtype == dtype and derivative.dtype == dtype
    torch.testing.assert_close(energy, value.new_tensor(expected_energy))
    torch.testing.assert_close(value_vjp, value.new_tensor(expected_value_vjp))
    torch.testing.assert_close(derivative, value_vjp)
    torch.testing.assert_close(
        torch.stack((slope_vjp, lower_vjp, upper_vjp)), value.new_tensor(expected_parameter_vjps)
    )


def test_flat_bottom_native_operand_broadcast_and_vjps() -> None:
    value = torch.tensor([-1.0, 0.0, 3.0], dtype=torch.float64, requires_grad=True)
    slope = torch.tensor([[2.0], [3.0]], dtype=torch.float64, requires_grad=True)
    lower_bound = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    upper_bound = torch.full((3,), 2.0, dtype=torch.float64, requires_grad=True)
    energy = linear_flat_bottom_energy(value, slope, lower_bound, upper_bound)
    derivative = linear_flat_bottom_derivative(value, slope, lower_bound, upper_bound)
    value_vjp, slope_vjp, lower_vjp, upper_vjp = torch.autograd.grad(
        energy.sum(), (value, slope, lower_bound, upper_bound)
    )
    assert energy.dtype == value.dtype and derivative.dtype == value.dtype
    torch.testing.assert_close(energy, value.new_tensor([[2.0, 0.0, 2.0], [3.0, 0.0, 3.0]]))
    torch.testing.assert_close(derivative, value.new_tensor([[-2.0, 0.0, 2.0], [-3.0, 0.0, 3.0]]))
    torch.testing.assert_close(value_vjp, value.new_tensor([-5.0, 0.0, 5.0]))
    torch.testing.assert_close(slope_vjp, slope.new_tensor([[2.0], [2.0]]))
    torch.testing.assert_close(lower_vjp, lower_bound.new_tensor([5.0, 0.0, 0.0]))
    torch.testing.assert_close(upper_vjp, upper_bound.new_tensor([0.0, 0.0, -5.0]))


def test_coordination() -> None:
    distances = torch.tensor([2.0, 10.0], dtype=torch.float64)

    coordination = smooth_coordination_potential(distances)

    torch.testing.assert_close(coordination, torch.tensor(1.5, dtype=torch.float64))


def test_flat_bottom_radial_potential() -> None:
    coordinates = torch.tensor([[-1.0, 0.0], [1.0, 0.0]], dtype=torch.float64)

    torch.testing.assert_close(flat_bottom_radial_potential(coordinates), torch.tensor(15.0, dtype=torch.float64))
    torch.testing.assert_close(
        flat_bottom_radial_potential(coordinates, radius_threshold=0.5),
        torch.tensor(1.0, dtype=torch.float64),
    )


def test_substrate_interaction() -> None:
    residue = torch.tensor([[0.0, 0.0]], dtype=torch.float64, requires_grad=True)
    substrate = torch.tensor([[2.0, 0.0], [4.0, 0.0]], dtype=torch.float64)

    nearest = substrate_interaction_potential(residue, substrate)
    all_pairs = substrate_interaction_potential(residue, substrate, nearest_only_repulsion=False)
    extra_repulsion = 2.0**2 / (1.5 * 5.0**0.5)

    torch.testing.assert_close(nearest - all_pairs, torch.tensor(extra_repulsion, dtype=torch.float64))
    nearest.backward()
    assert residue.grad is not None and torch.isfinite(residue.grad).all()
