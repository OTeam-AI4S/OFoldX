r"""
Isotropic Gaussian diffusion mathematics on :math:`SO(3)`.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor

_DEFAULT_NUM_TERMS = 2000


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

    Args:
        rotation_matrix: Rotation matrices `[..., 3, 3]`.

    Returns:
        Rotation vectors `[..., 3]` represented as axis times angle.
    """

    trace = rotation_matrix[..., 0, 0] + rotation_matrix[..., 1, 1] + rotation_matrix[..., 2, 2]
    cosine = ((trace - 1) * 0.5).clamp(-1.0, 1.0)
    one_minus_cosine = 1 - cosine
    small_angle = one_minus_cosine < 5e-7
    angle = torch.arccos(torch.where(small_angle, torch.zeros_like(cosine), cosine))
    antisymmetric = torch.stack(
        (
            rotation_matrix[..., 2, 1] - rotation_matrix[..., 1, 2],
            rotation_matrix[..., 0, 2] - rotation_matrix[..., 2, 0],
            rotation_matrix[..., 1, 0] - rotation_matrix[..., 0, 1],
        ),
        dim=-1,
    )
    scale_small = 0.5 + one_minus_cosine / 6 + one_minus_cosine.square() / 15
    scale = torch.where(small_angle, scale_small, angle / (2 * torch.sin(angle)).clamp_min(1e-30))
    rotation_vector = scale[..., None] * antisymmetric

    identity = torch.eye(3, dtype=rotation_matrix.dtype, device=rotation_matrix.device)
    symmetric = 0.5 * (rotation_matrix + rotation_matrix.transpose(-1, -2)) - cosine[..., None, None] * identity
    strongest_column = torch.linalg.vector_norm(symmetric, dim=-2).argmax(dim=-1)
    column = symmetric.gather(
        -1,
        strongest_column[..., None, None].expand(symmetric.shape[:-1] + (1,)),
    ).squeeze(-1)
    axis = column / torch.linalg.vector_norm(column, dim=-1, keepdim=True).clamp_min(1e-30)
    sign = torch.sign((axis * antisymmetric).sum(dim=-1, keepdim=True))
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    near_pi = angle[..., None] * axis * sign
    return torch.where((cosine < math.cos(math.pi - 0.1))[..., None], near_pi, rotation_vector)


def so3_project_to_rotation(matrix: Tensor) -> Tensor:
    r"""
    Project square matrices onto the nearest proper rotation.

    Args:
        matrix: Near-orthonormal matrices `[..., 3, 3]`.

    Returns:
        Proper rotation matrices `[..., 3, 3]`.
    """

    u, _singular_values, vh = torch.linalg.svd(matrix)
    determinant = torch.linalg.det(u @ vh)
    correction = torch.eye(matrix.shape[-1], dtype=matrix.dtype, device=matrix.device)
    correction = correction.expand(matrix.shape[:-2] + correction.shape).clone()
    correction[..., -1, -1] = torch.sign(determinant)
    return u @ correction @ vh


def igso3_expansion(
    omega: Tensor,
    variance: float | Tensor,
    num_terms: int = _DEFAULT_NUM_TERMS,
) -> Tensor:
    r"""
    Evaluate the truncated IGSO(3) density power series.

    Args:
        omega: Rotation angles `[...]`.
        variance: Diffusion variance.
        num_terms: Number of series terms.

    Returns:
        Density values with the shape of ``omega``.
    """

    variance = torch.as_tensor(variance, dtype=omega.dtype, device=omega.device)
    degree = torch.arange(num_terms, device=omega.device)
    angles = omega[..., None]
    terms = (2 * degree + 1) * torch.exp(-degree * (degree + 1) * variance / 2)
    zero = angles == 0
    denominator = torch.where(zero, torch.ones_like(angles), torch.sin(angles / 2))
    ratio = torch.sin(angles * (degree + 0.5)) / denominator
    ratio = torch.where(zero, 2 * degree + 1, ratio)
    return (terms * ratio).sum(dim=-1)


def igso3_log_derivative(
    omega: np.ndarray,
    variance: float,
    num_terms: int = _DEFAULT_NUM_TERMS,
) -> np.ndarray:
    r"""
    Evaluate the IGSO(3) log-density derivative with respect to rotation angle.
    """

    with torch.inference_mode(False), torch.enable_grad():
        omega_tensor = torch.tensor(omega, requires_grad=True)
        log_density = torch.log(igso3_expansion(omega_tensor, variance, num_terms=num_terms))
        return torch.autograd.grad(log_density.sum(), omega_tensor)[0].detach().numpy()


def igso3_angle_density(
    omega: np.ndarray,
    variance: float,
    num_terms: int = _DEFAULT_NUM_TERMS,
) -> np.ndarray:
    r"""
    Evaluate the IGSO(3) density including the angular Haar factor.
    """

    omega_tensor = torch.tensor(omega)
    density = igso3_expansion(omega_tensor, variance, num_terms=num_terms).detach().numpy()
    return density * (1 - np.cos(omega)) / np.pi


def calculate_igso3(
    *,
    num_sigma: int,
    num_omega: int,
    min_sigma: float,
    max_sigma: float,
    num_terms: int = _DEFAULT_NUM_TERMS,
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
    # The second moment is nonnegative; truncated-series roundoff can cross zero.
    return {
        "cdf": cdf,
        "score_norm": score,
        "exp_score_norms": np.sqrt(np.clip(score_moment, a_min=0.0, a_max=None)),
        "discrete_omega": discrete_omega,
        "discrete_sigma": discrete_sigma,
    }
