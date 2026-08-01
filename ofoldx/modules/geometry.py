r"""
Geometry utilities shared by model modules and losses.
"""

from __future__ import annotations

import torch
from torch import Tensor


def rigid_from_three_points(
    first: Tensor,
    center: Tensor,
    third: Tensor,
    *,
    eps: float = 1e-8,
) -> tuple[Tensor, Tensor]:
    r"""
    Build a right-handed orthonormal frame from three points.

    The first frame axis points from ``center`` to ``third``; the second is the
    component from ``center`` to ``first`` orthogonal to that axis.
    """

    first_axis = third - center
    second_direction = first - center
    first_axis = first_axis / (torch.linalg.vector_norm(first_axis, dim=-1, keepdim=True) + eps)
    second_axis = second_direction - (
        torch.einsum("...i,...i->...", first_axis, second_direction)[..., None] * first_axis
    )
    second_axis = second_axis / (torch.linalg.vector_norm(second_axis, dim=-1, keepdim=True) + eps)
    third_axis = torch.cross(first_axis, second_axis, dim=-1)
    return torch.stack((first_axis, second_axis, third_axis), dim=-1), center


def virtual_beta_carbon(nitrogen: Tensor, alpha_carbon: Tensor, carbon: Tensor) -> Tensor:
    r"""Reconstruct the idealized protein Cβ coordinate from N, Cα, and C coordinates."""

    b = alpha_carbon - nitrogen
    c = carbon - alpha_carbon
    return -0.58273431 * torch.cross(b, c, dim=-1) + 0.56802827 * b - 0.54067466 * c + alpha_carbon


def weighted_rigid_align(*, source: Tensor, target: Tensor, weight: Tensor, eps: float = 1e-8) -> Tensor:
    r"""
    Rigidly align ``source`` onto ``target`` with a weighted Kabsch fit.

    Solves the weighted orthogonal Procrustes problem: the rigid transform (rotation plus
    translation, no scale or reflection) that minimizes the weighted squared distance between
    ``source`` and ``target``. The covariance factorization runs in fp32 with autocast disabled,
    and the result is detached from autograd, so this is meant for building alignment targets
    rather than as a differentiable layer.

    Args:
        source: `[..., N, 3]` coordinates to align.
        target: `[..., N, 3]` target coordinates.
        weight: `[..., N]` non-negative per-point weights; points with weight ``0`` are ignored.
        eps: Floor for the weighted-centroid denominator, guarding against all-zero weights.

    Returns:
        Aligned ``source`` of shape `[..., N, 3]`, detached from autograd.
    """

    with torch.no_grad(), torch.autocast(device_type=source.device.type, enabled=False):
        work_dtype = torch.float32 if source.dtype in (torch.float16, torch.bfloat16) else source.dtype
        source_f = source.to(dtype=work_dtype)
        target_f = target.to(dtype=work_dtype)
        weight_f = weight.to(dtype=work_dtype).unsqueeze(-1)
        weight_sum = weight_f.sum(dim=-2, keepdim=True).clamp_min(eps)
        source_centroid = (source_f * weight_f).sum(dim=-2, keepdim=True) / weight_sum
        target_centroid = (target_f * weight_f).sum(dim=-2, keepdim=True) / weight_sum
        source_centered = source_f - source_centroid
        target_centered = target_f - target_centroid
        covariance = torch.einsum("...ni,...nj->...ij", weight_f * target_centered, source_centered)
        u, _, v = torch.linalg.svd(covariance, driver="gesvd" if covariance.is_cuda else None)
        v = v.mH
        rotation = torch.einsum("...ij,...kj->...ik", u, v)

        correction = torch.eye(source.shape[-1], dtype=covariance.dtype, device=covariance.device)
        correction = correction.expand(*rotation.shape[:-2], source.shape[-1], source.shape[-1]).clone()
        correction[..., -1, -1] = torch.det(rotation)
        rotation = torch.einsum("...ij,...jk,...lk->...il", u, correction, v)
        aligned = torch.einsum("...ni,...ji->...nj", source_centered, rotation) + target_centroid
    return aligned.to(dtype=source.dtype).detach()


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

    random = torch.empty((num_samples, 3), device=device, dtype=dtype).uniform_(generator=generator)
    u1, u2, u3 = torch.unbind(random, dim=-1)
    q1 = torch.sqrt(1 - u1) * torch.sin(2 * torch.pi * u2)
    q2 = torch.sqrt(1 - u1) * torch.cos(2 * torch.pi * u2)
    q3 = torch.sqrt(u1) * torch.sin(2 * torch.pi * u3)
    q4 = torch.sqrt(u1) * torch.cos(2 * torch.pi * u3)

    return torch.stack(
        [
            torch.stack([1 - 2 * (q3**2 + q4**2), 2 * (q2 * q3 - q1 * q4), 2 * (q2 * q4 + q1 * q3)], dim=-1),
            torch.stack([2 * (q2 * q3 + q1 * q4), 1 - 2 * (q2**2 + q4**2), 2 * (q3 * q4 - q1 * q2)], dim=-1),
            torch.stack([2 * (q2 * q4 - q1 * q3), 2 * (q3 * q4 + q1 * q2), 1 - 2 * (q2**2 + q3**2)], dim=-1),
        ],
        dim=-2,
    )
