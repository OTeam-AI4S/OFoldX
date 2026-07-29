r"""
Isotropic Gaussian diffusion mathematics on :math:`SO(3)`.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor

DEFAULT_NUM_TERMS = 2000


def so3_exp_map(rotation_vector: Tensor) -> Tensor:
    r"""
    Map axis-angle rotation vectors to rotation matrices.

    Args:
        rotation_vector: Rotation vectors `[..., 3]` represented as axis times angle.

    Returns:
        Rotation matrices `[..., 3, 3]`.
    """

    angle = torch.linalg.vector_norm(rotation_vector, dim=-1, keepdim=True)
    scale = torch.sinc(angle / math.pi)
    second_order_scale = 0.5 * torch.sinc(angle / (2 * math.pi)).square()
    x, y, z = rotation_vector.unbind(dim=-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1).reshape(rotation_vector.shape[:-1] + (3, 3))
    identity = torch.eye(3, dtype=rotation_vector.dtype, device=rotation_vector.device)
    return identity + scale[..., None] * skew + second_order_scale[..., None] * (skew @ skew)


def so3_log_map(rotation_matrix: Tensor) -> Tensor:
    r"""
    Map rotation matrices to axis-angle rotation vectors.

    Inverse of [`so3_exp_map`][ofoldx.modules.diffusion.igso3.so3_exp_map]: recovers the
    rotation vector (axis times angle in :math:`[0, \pi]`) from a rotation matrix. This matches
    ``scipy.spatial.transform.Rotation.from_matrix(...).as_rotvec()`` for an orthonormal input;
    feed a near-orthonormal matrix through
    [`so3_project_to_rotation`][ofoldx.modules.diffusion.igso3.so3_project_to_rotation] first to
    reproduce SciPy's implicit orthonormalization. Like any SO(3) logarithm the map is continuous
    but not differentiable at the axis-angle singularities :math:`\theta \in \{0, \pi\}`.

    Args:
        rotation_matrix: Rotation matrices `[..., 3, 3]`.

    Returns:
        Rotation vectors `[..., 3]` represented as axis times angle.
    """

    trace = rotation_matrix[..., 0, 0] + rotation_matrix[..., 1, 1] + rotation_matrix[..., 2, 2]
    cos_angle = ((trace - 1) * 0.5).clamp(-1.0, 1.0)
    angle = torch.arccos(cos_angle)
    # ``antisymmetric`` holds ``2 sin(angle)`` times the unit rotation axis.
    antisymmetric = torch.stack(
        (
            rotation_matrix[..., 2, 1] - rotation_matrix[..., 1, 2],
            rotation_matrix[..., 0, 2] - rotation_matrix[..., 2, 0],
            rotation_matrix[..., 1, 0] - rotation_matrix[..., 0, 1],
        ),
        dim=-1,
    )
    # Generic branch: the rotation vector is ``angle / (2 sin(angle))`` times ``antisymmetric``.
    # The small-angle branch is the Taylor series of that scale, avoiding the ``0 / 0`` at
    # ``angle == 0``.
    small_angle = angle < 1e-3
    scale_small = 0.5 * (1 + angle.square() / 6.0 + 7.0 * angle.square().square() / 360.0)
    scale = torch.where(small_angle, scale_small, angle / (2 * torch.sin(angle)).clamp_min(1e-30))
    rotation_vector = scale[..., None] * antisymmetric
    # Near ``angle == pi`` the antisymmetric part vanishes (``sin(angle) -> 0``), so the axis above
    # is ill-conditioned there. Recover it from the symmetric part instead:
    # ``S = sym - cos(angle) I`` equals ``(1 - cos(angle)) axis (x) axis``, a rank-1 matrix whose
    # every column is parallel to the axis; take the largest column for stability and fix the sign
    # from the antisymmetric part (which still carries the correct sign for ``angle < pi``). This
    # matches SciPy's quaternion ``as_rotvec`` except at the exact ``angle == pi`` singularity,
    # where the axis sign is genuinely ambiguous (measure zero).
    identity = torch.eye(3, dtype=rotation_matrix.dtype, device=rotation_matrix.device)
    symmetric = 0.5 * (rotation_matrix + rotation_matrix.transpose(-1, -2)) - cos_angle[..., None, None] * identity
    strongest = torch.linalg.vector_norm(symmetric, dim=-2).argmax(dim=-1)
    column = symmetric.gather(-1, strongest[..., None, None].expand(symmetric.shape[:-1] + (1,))).squeeze(-1)
    axis = column / torch.linalg.vector_norm(column, dim=-1, keepdim=True).clamp_min(1e-30)
    sign = torch.sign((axis * antisymmetric).sum(dim=-1, keepdim=True))
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    rotation_vector_near_pi = angle[..., None] * axis * sign
    return torch.where((angle > math.pi - 1e-1)[..., None], rotation_vector_near_pi, rotation_vector)


def so3_project_to_rotation(matrix: Tensor) -> Tensor:
    r"""
    Project square matrices onto the nearest proper rotation.

    Computes the orthogonal polar factor with a sign correction so the determinant is ``+1``,
    matching ``scipy.spatial.transform.Rotation.from_matrix(...).as_matrix()``. Used to clean up
    the near-orthonormal frames produced by
    [`rigid_from_three_points`][ofoldx.modules.geometry.rigid_from_three_points] before taking a
    rotation logarithm.

    Args:
        matrix: Near-orthonormal matrices `[..., 3, 3]`.

    Returns:
        Proper rotation matrices `[..., 3, 3]`.
    """

    u, _s, vh = torch.linalg.svd(matrix)
    determinant = torch.linalg.det(torch.matmul(u, vh))
    correction = torch.eye(matrix.shape[-1], dtype=matrix.dtype, device=matrix.device)
    correction = correction.expand(matrix.shape[:-2] + correction.shape).clone()
    correction[..., -1, -1] = torch.sign(determinant)
    return torch.matmul(torch.matmul(u, correction), vh)


def igso3_expansion(omega: Tensor, variance: float | Tensor, num_terms: int = DEFAULT_NUM_TERMS) -> Tensor:
    r"""
    Evaluate the truncated IGSO(3) density power series.

    Args:
        omega: Rotation angles `[...]`.
        variance: Diffusion variance.
        num_terms: Number of series terms.

    Returns:
        Density values with the shape of ``omega``.
    """

    omega = omega.float()
    variance = torch.as_tensor(variance, dtype=omega.dtype, device=omega.device)
    degree = torch.arange(num_terms, dtype=omega.dtype, device=omega.device)[None]
    angles = omega[..., None]
    denominator = torch.sin(angles / 2).clamp_min(1e-12)
    terms = (2 * degree + 1) * torch.exp(-degree * (degree + 1) * variance / 2)
    return (terms * torch.sin(angles * (degree + 0.5)) / denominator).sum(dim=-1)


def igso3_log_derivative(
    omega: np.ndarray,
    variance: float,
    num_terms: int = DEFAULT_NUM_TERMS,
) -> np.ndarray:
    r"""
    Evaluate the IGSO(3) score with respect to rotation angle.
    """

    with torch.inference_mode(False), torch.enable_grad():
        omega_tensor = torch.tensor(omega, dtype=torch.float32, requires_grad=True)
        log_density = torch.log(igso3_expansion(omega_tensor, variance, num_terms=num_terms).clamp_min(1e-30))
        return torch.autograd.grad(log_density.sum(), omega_tensor)[0].detach().cpu().numpy()


def igso3_angle_density(
    omega: np.ndarray,
    variance: float,
    num_terms: int = DEFAULT_NUM_TERMS,
) -> np.ndarray:
    r"""
    Evaluate the IGSO(3) density including the angular Haar factor.
    """

    omega_tensor = torch.tensor(omega, dtype=torch.float32)
    density = igso3_expansion(omega_tensor, variance, num_terms=num_terms).detach().cpu().numpy()
    return density * (1 - np.cos(omega)) / np.pi


def calculate_igso3(
    *,
    num_sigma: int,
    num_omega: int,
    min_sigma: float,
    max_sigma: float,
    num_terms: int = DEFAULT_NUM_TERMS,
) -> dict[str, np.ndarray]:
    r"""
    Precompute inverse-CDF and score tables over a sigma/angle grid.
    """

    discrete_omega = np.linspace(0, np.pi, num_omega + 1, dtype=np.float64)[1:]
    discrete_sigma = 10 ** np.linspace(np.log10(min_sigma), np.log10(max_sigma), num_sigma + 1, dtype=np.float64)[1:]
    densities = np.asarray(
        [igso3_angle_density(discrete_omega, float(sigma**2), num_terms=num_terms) for sigma in discrete_sigma]
    )
    cdf = np.asarray([density.cumsum() / num_omega * np.pi for density in densities])
    score = np.asarray(
        [igso3_log_derivative(discrete_omega, float(sigma**2), num_terms=num_terms) for sigma in discrete_sigma]
    )
    score_moment = np.sum(score**2 * densities, axis=1) / np.sum(densities, axis=1)
    return {
        "cdf": cdf,
        "score_norm": score,
        "exp_score_norms": np.sqrt(np.clip(score_moment, a_min=0.0, a_max=None)),
        "discrete_omega": discrete_omega,
        "discrete_sigma": discrete_sigma,
    }
