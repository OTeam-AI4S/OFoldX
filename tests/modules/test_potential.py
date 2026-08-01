from __future__ import annotations

import math

import torch

from ofoldx.modules.potential import (
    indexed_distance,
    indexed_distance_with_gradient,
    indexed_signed_dihedral,
    indexed_signed_dihedral_with_gradient,
    linear_flat_bottom_derivative,
    linear_flat_bottom_energy,
    smooth_coordination_potential,
    substrate_interaction_potential,
    upper_bounded_radius_of_gyration,
)


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


def test_coordination() -> None:
    distances = torch.tensor([2.0, 10.0], dtype=torch.float64)

    coordination = smooth_coordination_potential(distances)

    torch.testing.assert_close(coordination, torch.tensor(1.5, dtype=torch.float64))


def test_radius_of_gyration() -> None:
    coordinates = torch.tensor([[-1.0, 0.0], [1.0, 0.0]], dtype=torch.float64)

    torch.testing.assert_close(upper_bounded_radius_of_gyration(coordinates), torch.tensor(15.0, dtype=torch.float64))
    torch.testing.assert_close(
        upper_bounded_radius_of_gyration(coordinates, upper=0.5),
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
