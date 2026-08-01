r"""
Differentiable geometric potentials.
"""

from __future__ import annotations

import torch
from torch import Tensor


def indexed_distance(coordinates: Tensor, index: Tensor) -> Tensor:
    r"""
    Compute distances between coordinate pairs selected by a two-row index.
    """

    displacement = coordinates.index_select(-2, index[0]) - coordinates.index_select(-2, index[1])
    return torch.linalg.vector_norm(displacement, dim=-1)


def indexed_distance_with_gradient(coordinates: Tensor, index: Tensor) -> tuple[Tensor, Tensor]:
    r"""
    Compute indexed distances and their derivatives with respect to both endpoints.
    """

    displacement = coordinates.index_select(-2, index[0]) - coordinates.index_select(-2, index[1])
    distance = torch.linalg.vector_norm(displacement, dim=-1)
    direction = displacement / distance.unsqueeze(-1)
    return distance, torch.stack((direction, -direction), dim=-3)


def indexed_signed_dihedral(coordinates: Tensor, index: Tensor) -> Tensor:
    r"""
    Compute signed dihedral angles selected by a four-row coordinate index.
    """

    first = coordinates.index_select(-2, index[0]) - coordinates.index_select(-2, index[1])
    middle = coordinates.index_select(-2, index[2]) - coordinates.index_select(-2, index[1])
    last = coordinates.index_select(-2, index[2]) - coordinates.index_select(-2, index[3])
    normal_first = torch.linalg.cross(first, middle)
    normal_last = torch.linalg.cross(middle, last)
    sign = torch.sign(
        (middle.unsqueeze(-2) @ torch.linalg.cross(normal_first, normal_last).unsqueeze(-1)).squeeze(-1, -2)
    )
    cosine = (normal_first.unsqueeze(-2) @ normal_last.unsqueeze(-1)).squeeze(-1, -2) / (
        torch.linalg.vector_norm(normal_first, dim=-1) * torch.linalg.vector_norm(normal_last, dim=-1)
    )
    return sign * torch.arccos(cosine.clamp(-1 + 1e-8, 1 - 1e-8))


def indexed_signed_dihedral_with_gradient(
    coordinates: Tensor,
    index: Tensor,
) -> tuple[Tensor, Tensor]:
    r"""
    Compute indexed signed dihedrals and their derivatives with respect to all four atoms.
    """

    first = coordinates.index_select(-2, index[0]) - coordinates.index_select(-2, index[1])
    middle = coordinates.index_select(-2, index[2]) - coordinates.index_select(-2, index[1])
    last = coordinates.index_select(-2, index[2]) - coordinates.index_select(-2, index[3])
    normal_first = torch.linalg.cross(first, middle)
    normal_last = torch.linalg.cross(middle, last)
    middle_norm = torch.linalg.vector_norm(middle, dim=-1)
    normal_first_norm = torch.linalg.vector_norm(normal_first, dim=-1)
    normal_last_norm = torch.linalg.vector_norm(normal_last, dim=-1)
    sign = torch.sign(
        (middle.unsqueeze(-2) @ torch.linalg.cross(normal_first, normal_last).unsqueeze(-1)).squeeze(-1, -2)
    )
    cosine = (normal_first.unsqueeze(-2) @ normal_last.unsqueeze(-1)).squeeze(-1, -2) / (
        normal_first_norm * normal_last_norm
    )
    dihedral = sign * torch.arccos(cosine.clamp(-1 + 1e-8, 1 - 1e-8))
    first_projection = ((first.unsqueeze(-2) @ middle.unsqueeze(-1)).squeeze(-1, -2) / middle_norm.square()).unsqueeze(
        -1
    )
    last_projection = ((last.unsqueeze(-2) @ middle.unsqueeze(-1)).squeeze(-1, -2) / middle_norm.square()).unsqueeze(-1)
    gradient_first = normal_first * (middle_norm / normal_first_norm.square()).unsqueeze(-1)
    gradient_last = -normal_last * (middle_norm / normal_last_norm.square()).unsqueeze(-1)
    gradient_second = (first_projection - 1) * gradient_first - last_projection * gradient_last
    gradient_third = (last_projection - 1) * gradient_last - first_projection * gradient_first
    return dihedral, torch.stack(
        (gradient_first, gradient_second, gradient_third, gradient_last),
        dim=-3,
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

    below = value < lower_bound
    above = value > upper_bound
    energy = torch.zeros_like(value)
    energy[below] = (slope * (lower_bound - value))[below]
    energy[above] = (slope * (value - upper_bound))[above]
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

    below = value < lower_bound
    above = value > upper_bound
    derivative = torch.zeros_like(value)
    derivative[below] = -slope.expand_as(value)[below]
    derivative[above] = slope.expand_as(value)[above]
    return derivative


def smooth_coordination_potential(distances: Tensor, center: float = 2.0, scale: float = 8.0) -> Tensor:
    r"""
    Sum a sixth-power soft coordination count over pair distances.
    """

    scaled = (distances - center) / scale
    return (1 / (1 + scaled.pow(6))).sum()


def upper_bounded_radius_of_gyration(coordinates: Tensor, upper: float = 15.0) -> Tensor:
    r"""
    Root-mean-square radius with every radial distance bounded below by ``upper``.
    """

    distances = torch.linalg.vector_norm(coordinates - coordinates.mean(dim=-2, keepdim=True), dim=-1)
    return torch.sqrt(torch.clamp_min(distances, upper).square().mean())


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
