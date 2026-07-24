r"""
Differentiable geometric potentials.
"""

from __future__ import annotations

import torch
from torch import Tensor


def smooth_coordination_potential(distances: Tensor, center: float, scale: float) -> Tensor:
    r"""
    Sum a sixth-power soft coordination count over pair distances.
    """

    scaled = (distances - center) / scale
    return (1 / (1 + scaled.pow(6))).sum()


def interval_distance_penalty(distance: Tensor, lower: float, upper: float) -> Tensor:
    r"""
    Squared distance to a closed scalar interval.
    """

    return (
        torch.relu(distance.new_tensor(lower) - distance).square()
        + torch.relu(distance - distance.new_tensor(upper)).square()
    )


def upper_bounded_radius_of_gyration(coordinates: Tensor, upper: float) -> Tensor:
    r"""
    Root-mean-square radius with every radial distance bounded below by ``upper``.
    """

    distances = torch.linalg.vector_norm(coordinates - coordinates.mean(dim=-2, keepdim=True), dim=-1)
    return torch.sqrt(torch.clamp_min(distances, upper).square().mean())


def substrate_interaction_potential(
    residue_coordinates: Tensor,
    substrate_coordinates: Tensor,
    *,
    center: float,
    scale: float,
    attraction_strength: float,
    repulsion_slope: float,
    repulsion_cutoff: float,
) -> Tensor:
    r"""
    Soft attraction plus short-range repulsion against the nearest substrate atom.
    """

    nearest = torch.cdist(residue_coordinates, substrate_coordinates).amin(dim=-1)
    scaled = (nearest - center) / scale
    attraction = attraction_strength / (1 + scaled.pow(6))
    exponent = 1.5
    repulsion = (
        repulsion_slope**2
        * torch.clamp_min(repulsion_cutoff - nearest, 0).pow(exponent)
        / (exponent * repulsion_cutoff ** (exponent - 1))
    )
    return (attraction - repulsion).sum()


def harmonic_distance_potential(distance: Tensor, target: float) -> Tensor:
    r"""
    Negative squared displacement from a target distance.
    """

    return -(distance - target).square()
