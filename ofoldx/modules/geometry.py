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
Geometry utilities shared by model modules and losses.
"""

from __future__ import annotations

import math
from numbers import Real
from typing import Any, cast

import torch
from danling.tensors import NestedTensor
from torch import Tensor
from torch.autograd.function import once_differentiable

from .._oops import oops_kernel
from .backend.knn import gather_nodes
from .backend.resolve import resolve_tensor_backend
from .batch import flatten_batch, unflatten_batch
from .functional import _scaled_power, weighted_mean

_oops_smoothed_vector_norm = oops_kernel("smoothed_vector_norm")


class _SmoothedVectorNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, vectors: Tensor, eps: float) -> Tensor:
        work = vectors.double()
        identity = torch.zeros_like(work.sum(-1, keepdim=True))
        maximum = torch.cat((work.abs(), identity), -1).amax(-1, keepdim=True)
        epsilon_m, epsilon_e = math.frexp(math.sqrt(eps))
        scale = torch.maximum(maximum, maximum.new_tensor(math.sqrt(eps)))
        mantissa, exponent = torch.frexp(scale)
        mantissa = torch.where(mantissa == 0, 1.0, mantissa)
        normalized = _scaled_power(work, -exponent) / mantissa
        epsilon = _scaled_power(torch.full_like(mantissa, epsilon_m), epsilon_e - exponent) / mantissa
        factor = (normalized.square().sum(-1, keepdim=True) + epsilon.square()).sqrt()
        ctx.save_for_backward(vectors, mantissa, exponent, factor)
        result = _scaled_power(mantissa * factor, exponent)
        result = torch.where(maximum.isinf(), float("inf"), result)
        return result.squeeze(-1).to(vectors.dtype)

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, gradient: Tensor) -> tuple[Tensor, None]:
        vectors, mantissa, exponent, factor = ctx.saved_tensors
        xm, xe = torch.frexp(vectors.double())
        gm, ge = torch.frexp(gradient.double().unsqueeze(-1))
        denominator = torch.where(factor == 0, 1.0, mantissa * factor)
        output = _scaled_power(gm * xm / denominator, ge + xe - exponent)
        return torch.where(factor == 0, 0.0, output).to(vectors.dtype), None


def smoothed_vector_norm(vectors: Tensor, *, eps: float = 1e-8, keepdim: bool = False, backend: str = "auto") -> Tensor:
    r"""Compute a smoothed Euclidean norm over the final vector axis.

    Args:
        vectors: Input tensor with a static final vector axis. Nested inputs admit
            leading ragged dimensions or a static sample axis before one ragged atom axis.
        eps: Finite nonnegative smoothing term added to the sum of squares.
        keepdim: Whether to retain the final axis with length one.
        backend: Operator backend request, such as eager, auto, or oops.

    Returns:
        `sqrt(sum(vectors**2, dim=-1) + eps)` in the input dtype. Logical axes,
        nested cardinality, and empty batches are retained.

    Raises:
        TypeError: `eps` is not a real scalar or `keepdim` is not boolean.
        ValueError: `eps` is negative or nonfinite, or the input lacks a vector axis.
        NotImplementedError: A nested layout is outside the supported axis arrangements.

    Notes:
        Scaled arithmetic and coupled first-order gradients preserve representable
        finite components when separately materialized products or ratios would
        overflow or underflow. Higher-order gradients through the native implementation
        are unavailable.

    Examples:
        >>> import torch
        >>> vectors = torch.tensor([[3.0, 4.0]])
        >>> smoothed_vector_norm(vectors, eps=0.0, backend="eager").tolist()
        [5.0]
    """
    if isinstance(eps, bool) or not isinstance(eps, Real):
        raise TypeError("smoothed_vector_norm eps must be a real scalar")
    if not math.isfinite(eps) or eps < 0:
        raise ValueError("smoothed_vector_norm eps must be finite and non-negative")
    if type(keepdim) is not bool:
        raise TypeError("smoothed_vector_norm keepdim must be a bool")
    if vectors.dim() == 0:
        raise ValueError("smoothed_vector_norm vectors must have at least one axis")
    if isinstance(vectors, NestedTensor):
        rank = vectors.dim() - 1
        ragged = vectors.ragged_dims
        if (
            ragged
            and ragged == tuple(range(len(ragged)))
            and vectors.packed_dim_order == tuple(range(rank))
            and rank - 1 not in ragged
        ):
            values = smoothed_vector_norm(vectors.concat, eps=eps, keepdim=keepdim, backend=backend)
            return vectors.packed_with_static_tail(values)
        if vectors.batch_first and ragged == (1,) and rank > 2:
            batch_size, num_samples = vectors.shape[:2]
            norms = smoothed_vector_norm(flatten_batch(vectors), eps=eps, keepdim=keepdim, backend=backend)
            return unflatten_batch(norms, (batch_size, num_samples))
        raise NotImplementedError("smoothed_vector_norm requires leading ragged atoms and a static final vector axis")
    resolved = resolve_tensor_backend(
        backend,
        tensors=(vectors,),
        supported=("oops", "eager"),
        public_default="eager",
        oops=_oops_smoothed_vector_norm,
    )
    if resolved == "oops":
        return cast(Tensor, _oops_smoothed_vector_norm(vectors, eps=eps, keepdim=keepdim))
    output = cast(Tensor, _SmoothedVectorNorm.apply(vectors, eps))
    return output.unsqueeze(-1) if keepdim else output


def rotate_vectors(vectors: Tensor, rotation: Tensor) -> Tensor:
    r"""Apply row-vector rotation ``vectors @ rotation`` without materializing a matmul."""

    return (vectors.unsqueeze(-1) * rotation.unsqueeze(-3)).sum(dim=-2)


def apply_rigid_transform(points: Tensor, rotation: Tensor, translation: Tensor) -> Tensor:
    r"""Apply a column-vector rigid transform to points ``[..., P, 3]``.

    ``rotation[..., 3, 3]`` and ``translation[..., 3]`` broadcast over the
    leading frame axes. The result is ``points @ rotation.mT + translation``;
    single-vector callers insert their own singleton point axis.
    """

    return rotate_vectors(points, rotation.mT) + translation.unsqueeze(-2)


def apply_inverse_rigid_transform(points: Tensor, rotation: Tensor, translation: Tensor) -> Tensor:
    r"""Invert a rigid transform on points ``[..., P, 3]`` using ``rotation.mT``.

    The rotation is orthogonal and uses the same column-vector convention as
    ``apply_rigid_transform``. The result is ``(points - translation) @ rotation``.
    Leading frame axes broadcast; callers supply the explicit point axis.
    """

    return rotate_vectors(points - translation.unsqueeze(-2), rotation)


def normalize_vectors(vectors: Tensor) -> Tensor:
    r"""Normalize each nonzero vector along its last axis in its input dtype."""

    scale = vectors.detach().abs().amax(dim=-1, keepdim=True)
    vectors = vectors / scale
    return vectors / torch.linalg.vector_norm(vectors, dim=-1, keepdim=True)


def quaternion_to_rotation(quaternion: Tensor) -> Tensor:
    r"""Normalize scalar-first ``(..., 4)`` quaternions and return column-vector rotations.

    Nonzero ``(w, x, y, z)`` quaternions represent the same rotation at every
    nonzero scale, including the opposite gauge. The result has shape
    ``(..., 3, 3)`` and maps a vector as ``rotation @ vector``.
    """

    return unit_quaternion_to_rotation(normalize_vectors(quaternion))


def unit_quaternion_to_rotation(quaternion: Tensor) -> Tensor:
    r"""Convert scalar-first unit quaternions ``(..., 4)`` to column-vector rotations.

    The caller supplies unit ``(w, x, y, z)`` coefficients. This operation does
    not normalize them; nonunit coefficients are outside its input contract.
    """

    w, x, y, z = quaternion.unbind(dim=-1)
    return torch.stack(
        (
            torch.stack((1 - 2 * (y.square() + z.square()), 2 * (x * y - w * z), 2 * (x * z + w * y)), dim=-1),
            torch.stack((2 * (x * y + w * z), 1 - 2 * (x.square() + z.square()), 2 * (y * z - w * x)), dim=-1),
            torch.stack((2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x.square() + y.square())), dim=-1),
        ),
        dim=-2,
    )


def rotation_to_quaternion(rotation: Tensor) -> Tensor:
    r"""Return scalar-first unit quaternions from proper ``(..., 3, 3)`` rotations.

    Select the strongest quaternion component and normalize its bilinear row.
    This keeps exact half turns and avoids square roots of vanishing unselected
    components. The scalar is nonnegative; at an exact half turn, the first
    strongest vector component is positive.
    """

    row0, row1, row2 = rotation.unbind(dim=-2)
    xx, xy, xz = row0.unbind(dim=-1)
    yx, yy, yz = row1.unbind(dim=-1)
    zx, zy, zz = row2.unbind(dim=-1)
    ww = 1 + xx + yy + zz
    xx_component = 1 + xx - yy - zz
    yy_component = 1 - xx + yy - zz
    zz_component = 1 - xx - yy + zz
    candidates = torch.stack(
        (
            torch.stack((ww, zy - yz, xz - zx, yx - xy), dim=-1),
            torch.stack((zy - yz, xx_component, xy + yx, xz + zx), dim=-1),
            torch.stack((xz - zx, xy + yx, yy_component, yz + zy), dim=-1),
            torch.stack((yx - xy, xz + zx, yz + zy, zz_component), dim=-1),
        ),
        dim=-2,
    )
    strongest = torch.stack((ww, xx_component, yy_component, zz_component), dim=-1).argmax(dim=-1)
    index = strongest.unsqueeze(-1).unsqueeze(-1).expand(*([-1] * strongest.dim()), 1, 4)
    quaternion = candidates.gather(-2, index).squeeze(-2)
    quaternion = normalize_vectors(quaternion)
    return torch.where(quaternion[..., :1] < 0, -quaternion, quaternion)


def neighbor_atom_distances(
    coordinates: Tensor, neighbor_indices: Tensor, *, eps: float = 1e-6, backend: str = "auto"
) -> Tensor:
    r"""Return ordered atom distances along a sample-local residue neighbor graph.

    Args:
        coordinates: Atom coordinates ``[B, N, A, 3]``.
        neighbor_indices: Target residue indexes ``[B, N, K]`` for each source.
        eps: Added to the squared Cartesian distance before the square root.

    Returns:
        Distances ``[B, N, K, A, A]``, with source atom before target atom.
        A packed residue axis retains its original topology.
    """

    neighbors = gather_nodes(coordinates.flatten(start_dim=-2), neighbor_indices)
    neighbors = neighbors.unflatten(-1, (coordinates.shape[-2], coordinates.shape[-1]))
    displacement = coordinates.unsqueeze(-3).unsqueeze(-2) - neighbors.unsqueeze(-3)
    return smoothed_vector_norm(displacement, eps=eps, backend=backend)


def rotation_from_two_vectors(
    x_vector: Tensor, xy_vector: Tensor, *, eps: float = 1e-8, backend: str = "auto"
) -> Tensor:
    r"""
    Build the right-handed rotation whose x-axis follows ``x_vector`` and whose xy-plane contains ``xy_vector``.

    Gram-Schmidt: the x-axis is ``x_vector`` normalized, the y-axis is the component of ``xy_vector``
    orthogonal to it, normalized, and the z-axis is their cross product. Every normalization divides by
    ``sqrt(|v|^2 + eps)``, so a zero vector gives a zero axis rather than NaN. The frame convention is
    the caller's choice of vectors: a backbone frame with the x-axis along ``C - CA`` passes
    ``(C - CA, N - CA)``; AlphaFold's negative-x-axis frames pass ``(origin - negative_x, xy - origin)``.

    Args:
        x_vector: `[..., 3]` vector defining the x-axis.
        xy_vector: `[..., 3]` vector spanning the xy-plane with ``x_vector``.
        eps: Added to every squared norm before the square root.

    Returns:
        `[..., 3, 3]` rotations whose columns are the frame axes, so ``rotation @ local + origin`` maps
        frame coordinates to global coordinates.
    """

    x_axis = x_vector / smoothed_vector_norm(x_vector, eps=eps, keepdim=True, backend=backend)
    y_vector = xy_vector - (xy_vector * x_axis).sum(dim=-1, keepdim=True) * x_axis
    y_axis = y_vector / smoothed_vector_norm(y_vector, eps=eps, keepdim=True, backend=backend)
    z_axis = torch.cross(x_axis, y_axis, dim=-1)
    return torch.stack((x_axis, y_axis, z_axis), dim=-1)


def dihedral_angle(
    first: Tensor, second: Tensor, third: Tensor, fourth: Tensor, *, eps: float = 1e-8, backend: str = "auto"
) -> Tensor:
    r"""
    Return the signed dihedral angle of four points about the ``second``-``third`` bond.

    The angle follows the IUPAC convention: positive when, looking along ``second -> third``, the
    ``first`` bond must turn clockwise to eclipse the ``fourth`` bond. Both outer bonds are projected
    onto the plane normal to the central bond and the angle is ``atan2`` of their sine and cosine, which
    keeps full precision near ``0`` and ``pi``.

    Args:
        first: `[..., 3]` first point.
        second: `[..., 3]` start of the central bond.
        third: `[..., 3]` end of the central bond.
        fourth: `[..., 3]` fourth point.
        eps: Added to the squared central-bond length before the square root.

    Returns:
        `[...]` dihedral angles in radians, in ``[-pi, pi]``.
    """

    axis = third - second
    axis = axis / smoothed_vector_norm(axis, eps=eps, keepdim=True, backend=backend)
    first_bond = first - second
    fourth_bond = fourth - third
    first_bond = first_bond - (first_bond * axis).sum(dim=-1, keepdim=True) * axis
    fourth_bond = fourth_bond - (fourth_bond * axis).sum(dim=-1, keepdim=True) * axis
    cosine = (first_bond * fourth_bond).sum(dim=-1)
    sine = (torch.cross(axis, first_bond, dim=-1) * fourth_bond).sum(dim=-1)
    return torch.atan2(sine, cosine)


def atom37_to_torsion_angles(
    positions: Tensor,
    atom_mask: Tensor,
    *,
    chi_atom_indices: Tensor,
    chi_angles_mask: Tensor,
    chi_pi_periodic: Tensor,
    eps: float = 1e-8,
    backend: str = "auto",
) -> tuple[Tensor, Tensor, Tensor]:
    r"""
    Compute the seven AlphaFold torsion angles of every residue from atom37 coordinates.

    The torsions are pre-omega, phi, psi, and chi1 through chi4, each as a normalized ``(sin, cos)``
    pair. Each torsion's frame has its x-axis from the second to the third atom and the first atom in
    its xy-plane; the fourth atom's position in that frame gives the angle. Psi is negated to follow
    AlphaFold's rigid-group convention. The alternative angles add ``pi`` to the chi angles that
    ``chi_pi_periodic`` marks, whose two placements are chemically equivalent.

    Args:
        positions: `[..., L, 37, 3]` atom37 coordinates of consecutive residues.
        atom_mask: `[..., L, 37]` boolean atom existence mask.
        chi_atom_indices: `[..., L, 4, 4]` atom37 indices of the four atoms of each chi angle.
        chi_angles_mask: `[..., L, 4]` boolean mask of the chi angles each residue defines.
        chi_pi_periodic: `[..., L, 4]` boolean mask of the pi-periodic chi angles.
        eps: Added to every squared norm before the square root.

    Returns:
        ``(torsion_sin_cos, alternative_torsion_sin_cos, torsion_mask)`` with shapes `[..., L, 7, 2]`,
        `[..., L, 7, 2]`, and `[..., L, 7]`; the mask is boolean.
    """

    previous_positions = torch.cat([torch.zeros_like(positions[..., :1, :, :]), positions[..., :-1, :, :]], dim=-3)
    previous_mask = torch.cat([torch.zeros_like(atom_mask[..., :1, :]), atom_mask[..., :-1, :]], dim=-2)
    pre_omega = torch.cat([previous_positions[..., 1:3, :], positions[..., :2, :]], dim=-2)
    phi = torch.cat([previous_positions[..., 2:3, :], positions[..., :3, :]], dim=-2)
    psi = torch.cat([positions[..., :3, :], positions[..., 4:5, :]], dim=-2)
    pre_omega_mask = previous_mask[..., 1:3].all(dim=-1) & atom_mask[..., :2].all(dim=-1)
    phi_mask = previous_mask[..., 2] & atom_mask[..., :3].all(dim=-1)
    psi_mask = atom_mask[..., :3].all(dim=-1) & atom_mask[..., 4]

    chi_positions = torch.take_along_dim(positions.unsqueeze(-3), chi_atom_indices.unsqueeze(-1), dim=-2)
    chi_atom_mask = torch.take_along_dim(atom_mask.unsqueeze(-2), chi_atom_indices, dim=-1)
    chi_mask = chi_angles_mask & chi_atom_mask.all(dim=-1)

    torsion_atoms = torch.cat([pre_omega.unsqueeze(-3), phi.unsqueeze(-3), psi.unsqueeze(-3), chi_positions], dim=-3)
    torsion_mask = torch.cat(
        [pre_omega_mask.unsqueeze(-1), phi_mask.unsqueeze(-1), psi_mask.unsqueeze(-1), chi_mask], dim=-1
    )
    first, second, third, fourth = torsion_atoms.unbind(dim=-2)
    rotation = rotation_from_two_vectors(third - second, first - third, eps=eps, backend=backend)
    fourth_bond = fourth - third
    torsion_sin_cos = torch.stack(
        (
            (fourth_bond * rotation[..., 2]).sum(dim=-1),
            (fourth_bond * rotation[..., 1]).sum(dim=-1),
        ),
        dim=-1,
    )
    torsion_sin_cos = torsion_sin_cos / smoothed_vector_norm(torsion_sin_cos, eps=eps, keepdim=True, backend=backend)
    torsion_sin_cos = torsion_sin_cos * torsion_sin_cos.new_tensor((1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0))[..., None]
    mirror = torch.cat(
        [
            torch.ones_like(torsion_sin_cos[..., :3, 0]),
            1.0 - 2.0 * chi_pi_periodic.to(torsion_sin_cos.dtype),
        ],
        dim=-1,
    )
    return torsion_sin_cos, torsion_sin_cos * mirror.unsqueeze(-1), torsion_mask


def torsion_angles_to_frames(
    backbone_rotation: Tensor,
    backbone_translation: Tensor,
    torsion_sin_cos: Tensor,
    *,
    default_frames: Tensor,
) -> tuple[Tensor, Tensor]:
    r"""
    Build the eight global AlphaFold rigid-group frames of every residue from its backbone frame and torsions.

    The groups are backbone, pre-omega, phi, psi, and chi1 through chi4. Each group's default frame is
    rotated about its x-axis by the group's torsion; chi2 through chi4 chain onto the preceding chi
    frame, and every group is then mapped into global coordinates through the backbone frame.

    Args:
        backbone_rotation: `[..., 3, 3]` backbone rotations.
        backbone_translation: `[..., 3]` backbone translations.
        torsion_sin_cos: `[..., 7, 2]` ``(sin, cos)`` torsions in the order of
            [`atom37_to_torsion_angles`][ofoldx.modules.geometry.atom37_to_torsion_angles].
        default_frames: `[..., 8, 4, 4]` default rigid-group transforms of each residue.

    Returns:
        ``(rotation, translation)`` of the global group frames with shapes `[..., 8, 3, 3]` and `[..., 8, 3]`.
    """

    sine = torch.cat([torch.zeros_like(torsion_sin_cos[..., :1, 0]), torsion_sin_cos[..., 0]], dim=-1)
    cosine = torch.cat([torch.ones_like(torsion_sin_cos[..., :1, 1]), torsion_sin_cos[..., 1]], dim=-1)
    zeros = torch.zeros_like(sine)
    ones = torch.ones_like(sine)
    torsion_rotation = torch.stack(
        [
            torch.stack([ones, zeros, zeros], dim=-1),
            torch.stack([zeros, cosine, -sine], dim=-1),
            torch.stack([zeros, sine, cosine], dim=-1),
        ],
        dim=-2,
    )
    rotations = list((default_frames[..., :3, :3] @ torsion_rotation).unbind(dim=-3))
    translations = list(default_frames[..., :3, 3].unbind(dim=-2))
    for group in (5, 6, 7):
        translations[group] = (rotations[group - 1] @ translations[group].unsqueeze(-1)).squeeze(-1) + translations[
            group - 1
        ]
        rotations[group] = rotations[group - 1] @ rotations[group]
    rotation = torch.stack(rotations, dim=-3)
    translation = torch.stack(translations, dim=-2)
    global_translation = (backbone_rotation.unsqueeze(-3) @ translation.unsqueeze(-1)).squeeze(
        -1
    ) + backbone_translation.unsqueeze(-2)
    return backbone_rotation.unsqueeze(-3) @ rotation, global_translation


def frames_to_atom14_positions(
    rotation: Tensor,
    translation: Tensor,
    *,
    atom14_to_rigid_group: Tensor,
    atom14_rigid_group_positions: Tensor,
    atom14_mask: Tensor,
) -> Tensor:
    r"""
    Place each residue's ideal atom14 positions with its global rigid-group frames.

    Args:
        rotation: `[..., 8, 3, 3]` global rigid-group rotations.
        translation: `[..., 8, 3]` global rigid-group translations.
        atom14_to_rigid_group: `[..., 14]` rigid group of each atom14 slot.
        atom14_rigid_group_positions: `[..., 14, 3]` ideal atom positions in their group frames.
        atom14_mask: `[..., 14]` boolean mask of the atoms each residue defines.

    Returns:
        `[..., 14, 3]` global atom14 positions; undefined atoms are zero.
    """

    atom_rotation = torch.take_along_dim(rotation, atom14_to_rigid_group[..., None, None], dim=-3)
    atom_translation = torch.take_along_dim(translation, atom14_to_rigid_group[..., None], dim=-2)
    positions = (atom_rotation @ atom14_rigid_group_positions.unsqueeze(-1)).squeeze(-1) + atom_translation
    return positions * atom14_mask.unsqueeze(-1)


def virtual_beta_carbon(nitrogen: Tensor, alpha_carbon: Tensor, carbon: Tensor) -> Tensor:
    r"""Reconstruct the idealized protein Cβ coordinate from N, Cα, and C coordinates."""

    b = alpha_carbon - nitrogen
    c = carbon - alpha_carbon
    return -0.58273431 * torch.cross(b, c, dim=-1) + 0.56802827 * b - 0.54067466 * c + alpha_carbon


def kabsch_transform(
    *,
    source: Tensor,
    target: Tensor,
    weight: Tensor,
    eps: float = 1e-8,
    backend: str = "auto",
) -> tuple[Tensor, Tensor, Tensor]:
    r"""
    Fit a differentiable weighted rigid transform from ``source`` onto ``target``.

    Computes weighted centroids and the proper SO(3) orthogonal Procrustes
    rotation. Autograd follows PyTorch's SVD derivative; rank degeneracy and
    repeated singular values have its ordinary derivative limitations. Inputs use
    their common promoted dtype.

    Args:
        source: Coordinates ``[..., N, 3]`` to align.
        target: Target coordinates ``[..., N, 3]``.
        weight: Non-negative per-point weights ``[..., N]``.
        eps: Denominator used when total weight is zero.

    Returns:
        Column-vector rotation and source/target centroids. Apply to row-vector
        coordinates as ``(source - source_centroid) @ rotation.mT + target_centroid``.
    """

    dtype = torch.promote_types(torch.promote_types(source.dtype, target.dtype), weight.dtype)
    source = source.to(dtype=dtype)
    target = target.to(dtype=dtype)
    weight_f = weight.to(dtype=dtype)
    source = source.masked_fill((weight_f == 0).unsqueeze(-1) & ~torch.isfinite(source), 0.0)
    target = target.masked_fill((weight_f == 0).unsqueeze(-1) & ~torch.isfinite(target), 0.0)
    present = (weight_f > 0).any(dim=-1)
    empty_mean = torch.zeros((), device=source.device, dtype=source.dtype) / eps
    source_centroid = weighted_mean(source.transpose(-2, -1), weight_f.unsqueeze(-2), backend=backend)
    target_centroid = weighted_mean(target.transpose(-2, -1), weight_f.unsqueeze(-2), backend=backend)
    source_centroid = torch.where(present.unsqueeze(-1), source_centroid, empty_mean).unsqueeze(-2)
    target_centroid = torch.where(present.unsqueeze(-1), target_centroid, empty_mean).unsqueeze(-2)
    source_centered = source - source_centroid
    target_centered = target - target_centroid
    products = target_centered.unsqueeze(-1) * source_centered.unsqueeze(-2)
    covariance = weighted_mean(products.movedim(-3, -1), weight_f.unsqueeze(-2).unsqueeze(-2), backend=backend)
    covariance = torch.where(present.unsqueeze(-1).unsqueeze(-1), covariance, empty_mean)
    u, _, v = torch.linalg.svd(covariance, driver="gesvd" if covariance.is_cuda else None)
    v = v.mH
    correction = torch.eye(source.shape[-1], dtype=covariance.dtype, device=covariance.device)
    correction = correction.expand(*covariance.shape[:-2], source.shape[-1], source.shape[-1]).clone()
    correction[..., -1, -1] = (torch.det(u) * torch.det(v)).sign()
    rotation = torch.einsum("...ij,...jk,...lk->...il", u, correction, v)
    return rotation, source_centroid, target_centroid


def rigid_alignment_transform(
    *,
    source: Tensor,
    target: Tensor,
    weight: Tensor,
    eps: float = 1e-8,
    backend: str = "auto",
) -> tuple[Tensor, Tensor, Tensor]:
    r"""
    Fit the weighted Kabsch rotation and centroids that align ``source`` onto ``target``.

    Solves the weighted orthogonal Procrustes problem: the rotation (no translation, scale, or
    reflection) that minimizes the weighted squared distance between centered ``source`` and
    ``target``. Uses native tensor dtype promotion.
    Dtype and device support follows `torch.linalg.svd`. Every returned tensor is detached
    from autograd, so this is meant for building alignment targets rather than as a
    differentiable layer.

    Returned separately from [`weighted_rigid_align`][ofoldx.modules.geometry.weighted_rigid_align]
    so a caller can reuse one fitted transform against coordinates other than ``source`` itself
    (for example, scoring several candidate correspondences under the same rigid fit).

    Args:
        source: `[..., N, 3]` coordinates to align.
        target: `[..., N, 3]` target coordinates.
        weight: `[..., N]` non-negative per-point weights; points with weight ``0`` are ignored.
        eps: Denominator used for all-zero weights; positive weights normalize exactly.

    Returns:
        ``(rotation, source_centroid, target_centroid)``: a `[..., 3, 3]` rotation mapping
        centered ``source`` onto centered ``target``, and the `[..., 1, 3]` weighted centroids
        used to center each side, with native input promotion. Apply as
        ``(source - source_centroid) @ rotation.mT + target_centroid`` for row-vector
        coordinates: the returned rotation maps column vectors onto the target.
    """

    with torch.no_grad():
        rotation, source_centroid, target_centroid = kabsch_transform(
            source=source, target=target, weight=weight, eps=eps, backend=backend
        )
    return rotation.detach(), source_centroid.detach(), target_centroid.detach()


def weighted_rigid_align(
    *, source: Tensor, target: Tensor, weight: Tensor, eps: float = 1e-8, backend: str = "auto"
) -> Tensor:
    r"""
    Rigidly align ``source`` onto ``target`` with a weighted Kabsch fit.

    Solves the weighted orthogonal Procrustes problem: the rigid transform (rotation plus
    translation, no scale or reflection) that minimizes the weighted squared distance between
    ``source`` and ``target``. See
    [`rigid_alignment_transform`][ofoldx.modules.geometry.rigid_alignment_transform] for the
    dtype and autograd behavior of the underlying fit; the result here uses native input promotion
    and is detached from autograd, so this is meant for building alignment
    targets rather than as a differentiable layer.

    Args:
        source: `[..., N, 3]` coordinates to align.
        target: `[..., N, 3]` target coordinates.
        weight: `[..., N]` non-negative per-point weights; points with weight ``0`` are ignored.
        eps: Denominator used for all-zero weights; positive weights normalize exactly.

    Returns:
        Aligned ``source`` of shape `[..., N, 3]`, detached from autograd.
    """

    rotation, source_centroid, target_centroid = rigid_alignment_transform(
        source=source, target=target, weight=weight, eps=eps, backend=backend
    )
    source_centered = source - source_centroid
    aligned = apply_rigid_transform(source_centered, rotation, target_centroid.squeeze(-2))
    return aligned.detach()


def uniform_random_rotation(
    num_samples: int = 1,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    r"""
    Sample Haar-uniform ``SO(3)`` rotation matrices.

    Uses Shoemake's method (a uniform unit quaternion drawn from three uniform deviates, then
    converted to a matrix): closed-form, branch-free, and half-precision-safe, unlike
    QR-of-Gaussian. Intended for coordinate augmentation, so only the uniformity of the
    distribution matters, not the specific matrices produced.

    Args:
        num_samples: Number of rotation matrices to draw.
        device: Device for the sampled tensor; defaults to the current default device.
        dtype: Floating dtype for the sampled tensor; defaults to the current default dtype.
        generator: Optional random generator for reproducible sampling.

    Returns:
        Rotation matrices of shape `[num_samples, 3, 3]`, each orthonormal with determinant ``+1``.
    """

    device = torch.get_default_device() if device is None else device
    random_device = device if generator is None else generator.device
    random = torch.empty((num_samples, 3), device=random_device, dtype=dtype).uniform_(generator=generator).to(device)
    u1, u2, u3 = torch.unbind(random, dim=-1)
    q1 = torch.sqrt(1 - u1) * torch.sin(2 * torch.pi * u2)
    q2 = torch.sqrt(1 - u1) * torch.cos(2 * torch.pi * u2)
    q3 = torch.sqrt(u1) * torch.sin(2 * torch.pi * u3)
    q4 = torch.sqrt(u1) * torch.cos(2 * torch.pi * u3)

    return quaternion_to_rotation(torch.stack((q1, q2, q3, q4), dim=-1))


def local_backbone_frame_features(
    nitrogen: Tensor,
    alpha_carbon: Tensor,
    carbon: Tensor,
    points: Tensor,
    *,
    backend: str = "auto",
) -> Tensor:
    r"""
    Express context-point directions in each residue's local backbone frame.

    Builds a right-handed frame at each alpha carbon with its x-axis along ``nitrogen - alpha_carbon``
    and ``carbon - alpha_carbon`` in its xy-plane, then encodes every context point as four scale-free
    numbers: the in-plane direction cosine pair, the planar-to-spatial radius ratio, and the out-of-plane
    direction cosine.

    Args:
        nitrogen: Backbone nitrogen coordinates `[B, L, 3]`.
        alpha_carbon: Alpha-carbon coordinates `[B, L, 3]`, the frame origin.
        carbon: Backbone carbon coordinates `[B, L, 3]`.
        points: Context points to encode `[B, L, P, 3]`.

    Returns:
        Direction features `[B, L, P, 4]`.
    """

    frame = rotation_from_two_vectors(nitrogen - alpha_carbon, carbon - alpha_carbon, backend=backend)
    local = torch.einsum("blqp,blyq->blyp", frame, points - alpha_carbon[:, :, None])
    planar_radius = smoothed_vector_norm(local[..., :2], eps=1e-8, backend=backend)
    spatial_radius = smoothed_vector_norm(local, eps=0, backend=backend) + 1e-8
    return torch.stack(
        (
            local[..., 0] / planar_radius,
            local[..., 1] / planar_radius,
            planar_radius / spatial_radius,
            local[..., 2] / spatial_radius,
        ),
        dim=-1,
    )
