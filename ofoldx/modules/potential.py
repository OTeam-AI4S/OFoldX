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

r"""
Differentiable geometric potentials.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .functional import indexed_sum
from .geometry import dihedral_angle, smoothed_vector_norm


def indexed_distance(coordinates: Tensor, index: Tensor, *, minimum: float = 0.0) -> Tensor:
    r"""
    Compute indexed Euclidean distances with an optional lower bound.
    """

    displacement = coordinates.index_select(-2, index[0]) - coordinates.index_select(-2, index[1])
    return torch.linalg.vector_norm(displacement, dim=-1).clamp_min(minimum)


def indexed_distance_with_gradient(
    coordinates: Tensor, index: Tensor, *, minimum: float = 0.0
) -> tuple[Tensor, Tensor]:
    r"""
    Compute indexed distances and their derivatives with respect to both endpoints.
    """

    displacement = coordinates.index_select(-2, index[0]) - coordinates.index_select(-2, index[1])
    raw_distance = torch.linalg.vector_norm(displacement, dim=-1)
    distance = raw_distance.clamp_min(minimum)
    positive = raw_distance > 0
    denominator = torch.where(positive, distance, 1.0)
    direction = torch.where(
        (positive & (raw_distance >= minimum)).unsqueeze(-1), displacement / denominator.unsqueeze(-1), 0.0
    )
    return distance, torch.stack((direction, -direction), dim=-3)


def indexed_signed_dihedral(coordinates: Tensor, index: Tensor, *, eps: float = 1e-8, backend: str = "auto") -> Tensor:
    r"""
    Compute signed dihedral angles selected by a four-row coordinate index.
    """

    return dihedral_angle(*(coordinates.index_select(-2, row) for row in index), eps=eps, backend=backend)


def indexed_signed_dihedral_with_gradient(
    coordinates: Tensor,
    index: Tensor,
    *,
    eps: float = 1e-8,
    backend: str = "auto",
) -> tuple[Tensor, Tensor]:
    r"""
    Compute indexed signed dihedrals and their derivatives with respect to all four atoms.

    Differentiate the same stabilized central-axis projection as ``dihedral_angle``.
    Its axis has length below one when ``eps`` is positive, so the usual unit-axis
    dihedral derivative does not apply. An undefined ``atan2(0, 0)`` retains its
    ordinary undefined derivative.
    """

    points = [coordinates.index_select(-2, row) for row in index]
    first = points[0] - points[1]
    middle = points[2] - points[1]
    last = points[3] - points[2]
    middle_norm = smoothed_vector_norm(middle, eps=eps, keepdim=True, backend=backend)
    axis = middle / middle_norm
    first_dot = (first * axis).sum(dim=-1, keepdim=True)
    last_dot = (last * axis).sum(dim=-1, keepdim=True)
    first_plane = first - first_dot * axis
    last_plane = last - last_dot * axis
    cosine = (first_plane * last_plane).sum(dim=-1, keepdim=True)
    sine = (torch.linalg.cross(axis, first_plane) * last_plane).sum(dim=-1, keepdim=True)
    denominator = cosine.square() + sine.square()
    cosine_gradient = -sine / denominator
    sine_gradient = cosine / denominator

    first_plane_gradient = cosine_gradient * last_plane + sine_gradient * torch.linalg.cross(last_plane, axis)
    last_plane_gradient = cosine_gradient * first_plane + sine_gradient * torch.linalg.cross(axis, first_plane)
    axis_gradient = sine_gradient * torch.linalg.cross(first_plane, last_plane)
    first_axis_gradient = (first_plane_gradient * axis).sum(dim=-1, keepdim=True)
    last_axis_gradient = (last_plane_gradient * axis).sum(dim=-1, keepdim=True)
    axis_gradient = (
        axis_gradient
        - first_dot * first_plane_gradient
        - first_axis_gradient * first
        - last_dot * last_plane_gradient
        - last_axis_gradient * last
    )
    gradient_first = first_plane_gradient - first_axis_gradient * axis
    gradient_last = last_plane_gradient - last_axis_gradient * axis
    gradient_middle = (axis_gradient - (axis_gradient * axis).sum(dim=-1, keepdim=True) * axis) / middle_norm
    gradient_second = -gradient_first - gradient_middle
    gradient_third = gradient_middle - gradient_last
    return dihedral_angle(*points, eps=eps, backend=backend), torch.stack(
        (gradient_first, gradient_second, gradient_third, gradient_last), dim=-3
    )


def linear_flat_bottom_energy(
    value: Tensor,
    slope: Tensor,
    lower_bound: Tensor,
    upper_bound: Tensor,
) -> Tensor:
    r"""
    Evaluate a linear penalty outside a closed interval.
    """

    value, slope, lower_bound, upper_bound = torch.broadcast_tensors(value, slope, lower_bound, upper_bound)
    below = value < lower_bound
    above = value > upper_bound
    energy = torch.zeros_like(value)
    energy[below] = slope[below] * (lower_bound[below] - value[below])
    energy[above] = slope[above] * (value[above] - upper_bound[above])
    return energy


def linear_flat_bottom_derivative(
    value: Tensor,
    slope: Tensor,
    lower_bound: Tensor,
    upper_bound: Tensor,
) -> Tensor:
    r"""
    Differentiate a linear flat-bottom energy with respect to its value.
    """

    value, slope, lower_bound, upper_bound = torch.broadcast_tensors(value, slope, lower_bound, upper_bound)
    below = value < lower_bound
    above = value > upper_bound
    derivative = torch.zeros_like(value)
    derivative[below] = -slope[below]
    derivative[above] = slope[above]
    return derivative


def chain_center_separation(
    coordinates: Tensor,
    atom_chain: Tensor,
    atom_mask: Tensor,
    num_chains: int,
    chain_pair_index: Tensor,
    lower_bound: Tensor,
    slope: Tensor,
    *,
    compute_gradient: bool,
) -> Tensor | tuple[Tensor, Tensor]:
    r"""Penalize chain-center distances below ``lower_bound``.

    Coordinates have shape ``[..., A, 3]``; every leading axis describes
    independent coordinates for the same prepared atom/chain correspondence.
    Centers are arithmetic means of retained atoms, with no mass weighting.
    The energy sums strict lower flat-bottom penalties over the selected chain
    pairs. When requested, the analytic coordinate gradient divides each chain
    force by its retained count and is zero at excluded atoms.
    """

    retained_chain = atom_chain[atom_mask]
    retained_coordinates = coordinates[..., atom_mask, :]
    counts = torch.bincount(retained_chain, minlength=num_chains)
    centers = indexed_sum(retained_coordinates, retained_chain, num_chains, dim=-2)
    centers = centers / counts.clamp_min(1).unsqueeze(-1)
    if compute_gradient:
        distance, distance_gradient = indexed_distance_with_gradient(centers, chain_pair_index)
    else:
        distance = indexed_distance(centers, chain_pair_index)
    upper_bound = distance.new_tensor(float("inf"))
    energy = linear_flat_bottom_energy(distance, slope, lower_bound, upper_bound).sum(dim=-1)
    if not compute_gradient:
        return energy

    derivative = linear_flat_bottom_derivative(distance, slope, lower_bound, upper_bound)
    contributions = derivative[..., None, :, None] * distance_gradient
    center_gradient = indexed_sum(contributions.flatten(-3, -2), chain_pair_index.flatten(), num_chains, dim=-2)
    atom_gradient = torch.zeros_like(coordinates)
    atom_gradient[..., atom_mask, :] = center_gradient.index_select(-2, retained_chain) / counts[
        retained_chain
    ].unsqueeze(-1)
    return energy, atom_gradient


def smooth_coordination_potential(distances: Tensor, center: float = 2.0, scale: float = 8.0) -> Tensor:
    r"""
    Sum a sixth-power soft coordination count over pair distances.
    """

    scaled = (distances - center) / scale
    return (1 / (1 + scaled.pow(6))).sum()


def flat_bottom_radial_potential(coordinates: Tensor, radius_threshold: float = 15.0) -> Tensor:
    r"""
    Return the root-mean-square centered radii, each bounded below by ``radius_threshold``.

    The threshold and coordinates use the same distance unit. The value is constant
    when every centered radius lies below the threshold, and grows outside it.
    """

    distances = torch.linalg.vector_norm(coordinates - coordinates.mean(dim=-2, keepdim=True), dim=-1)
    return torch.sqrt(torch.clamp_min(distances, radius_threshold).square().mean())


def substrate_interaction_potential(
    residue_coordinates: Tensor,
    substrate_coordinates: Tensor,
    *,
    center: float = 2.0,
    scale: float = 8.0,
    attraction_strength: float = 1.0,
    repulsion_slope: float = 2.0,
    repulsion_cutoff: float = 5.0,
    nearest_only_repulsion: bool = True,
) -> Tensor:
    r"""
    Attract each residue to its nearest substrate atom and apply short-range repulsion.

    Repulsion uses nearest distances when ``nearest_only_repulsion`` is true and
    every residue--substrate atom distance otherwise.
    """

    distances = torch.cdist(residue_coordinates, substrate_coordinates)
    nearest = distances.amin(dim=-1)
    attraction = attraction_strength / (1 + ((nearest - center) / scale).pow(6))
    exponent = 1.5
    repulsion_distances = nearest if nearest_only_repulsion else distances
    repulsion = (
        repulsion_slope**2
        * torch.clamp_min(repulsion_cutoff - repulsion_distances, 0).pow(exponent)
        / (exponent * repulsion_cutoff ** (exponent - 1))
    )
    return attraction.sum() - repulsion.sum()
