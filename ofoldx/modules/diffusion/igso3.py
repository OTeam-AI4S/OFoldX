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
Isotropic Gaussian diffusion mathematics on :math:`SO(3)`.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor
from torch.autograd.function import FunctionCtx

from ..geometry import normalize_vectors, rotation_to_quaternion, smoothed_vector_norm

NUM_IGSO3_TERMS = 2000


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


def so3_log_map(rotation_matrix: Tensor, *, backend: str = "auto") -> Tensor:
    r"""
    Map rotation matrices to axis-angle rotation vectors.

    Args:
        rotation_matrix: Rotation matrices `[..., 3, 3]`.

    Returns:
        Rotation vectors `[..., 3]` represented as axis times angle.
    """

    trace = rotation_matrix[..., 0, 0] + rotation_matrix[..., 1, 1] + rotation_matrix[..., 2, 2]
    trace_cosine = (trace - 1) * 0.5
    cosine = trace_cosine.clamp(-1.0, 1.0)
    one_minus_cosine = 1 - cosine
    small_angle = one_minus_cosine < 5e-7
    antisymmetric = torch.stack(
        (
            rotation_matrix[..., 2, 1] - rotation_matrix[..., 1, 2],
            rotation_matrix[..., 0, 2] - rotation_matrix[..., 2, 0],
            rotation_matrix[..., 1, 0] - rotation_matrix[..., 0, 1],
        ),
        dim=-1,
    )
    quaternion = rotation_to_quaternion(rotation_matrix)
    vector = torch.where(small_angle[..., None], rotation_matrix.new_tensor((1, 0, 0)), quaternion[..., 1:])
    scalar = torch.where(small_angle, 1.0, quaternion[..., 0])
    angle = 2 * torch.atan2(smoothed_vector_norm(vector, eps=0, backend=backend), scalar)
    scale_small = 0.5 + one_minus_cosine / 6 + one_minus_cosine.square() / 15
    return torch.where(
        small_angle[..., None],
        scale_small[..., None] * antisymmetric,
        angle[..., None] * normalize_vectors(vector),
    )


def so3_project_to_rotation(matrix: Tensor) -> Tensor:
    r"""
    Project matrices onto the nearest proper rotation.

    Derivatives require a unique nearest rotation.

    Args:
        matrix: Matrices `[..., 3, 3]`.

    Returns:
        Proper rotation matrices `[..., 3, 3]`.
    """

    return _So3ProjectToRotation.apply(matrix)


class _So3ProjectToRotation(torch.autograd.Function):
    generate_vmap_rule = True

    @staticmethod
    def forward(matrix: Tensor) -> Tensor:
        u, _singular_values, vh = torch.linalg.svd(matrix)
        determinant = torch.linalg.det(u @ vh)
        one = torch.ones_like(determinant)
        correction = torch.diag_embed(torch.stack((one, one, torch.sign(determinant)), dim=-1))
        return u @ correction @ vh

    @staticmethod
    def setup_context(ctx: FunctionCtx, inputs: tuple[Tensor], output: Tensor) -> None:
        ctx.save_for_backward(inputs[0], output)
        ctx.save_for_forward(inputs[0], output)

    @staticmethod
    def _differential(matrix: Tensor, rotation: Tensor, tangent: Tensor) -> Tensor:
        # Differentiating Q^T M = M^T Q gives (tr(S) I - S) v = vee(Q^T dM - dM^T Q).
        symmetric = rotation.transpose(-1, -2) @ matrix
        trace = symmetric.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        identity = torch.eye(3, dtype=matrix.dtype, device=matrix.device)
        coefficient = trace[..., None, None] * identity - symmetric
        projected = rotation.transpose(-1, -2) @ tangent
        rhs = torch.stack(
            (
                projected[..., 2, 1] - projected[..., 1, 2],
                projected[..., 0, 2] - projected[..., 2, 0],
                projected[..., 1, 0] - projected[..., 0, 1],
            ),
            dim=-1,
        )
        x, y, z = torch.linalg.solve(coefficient, rhs.unsqueeze(-1)).squeeze(-1).unbind(dim=-1)
        zero = torch.zeros_like(x)
        skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1).reshape(matrix.shape)
        return rotation @ skew

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: Tensor) -> Tensor:
        matrix, rotation = ctx.saved_tensors
        return _So3ProjectToRotation._differential(matrix, rotation, grad_output)

    @staticmethod
    def jvp(ctx: FunctionCtx, matrix_tangent: Tensor) -> Tensor:
        matrix, rotation = ctx.saved_tensors
        return _So3ProjectToRotation._differential(matrix, rotation, matrix_tangent)


def igso3_expansion(
    omega: Tensor,
    variance: float | Tensor,
    num_terms: int = NUM_IGSO3_TERMS,
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
    num_terms: int = NUM_IGSO3_TERMS,
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
    num_terms: int = NUM_IGSO3_TERMS,
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
    num_terms: int = NUM_IGSO3_TERMS,
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
