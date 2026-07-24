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
