r"""
Shared EDM coordinate-denoising training loss for the diffusion structure families.

The internally-noising diffusion models (Boltz-1, Boltz-2, Protenix) expose their denoised coordinates
and the augmented training target on the same augmented frame, plus the per-sample noise level, so the
canonical EDM denoising objective -- the ``1/c_out(sigma)**2``-weighted squared error between the two --
grounds directly against the model output. Families with a richer published objective (e.g. BoltzGen's
alignment/lDDT/bond loss) wrap their own loss instead; this is the common core the others share.
"""

from __future__ import annotations

from typing import Any

from torch import Tensor


def edm_loss_weight(noise_level: Tensor, sigma_data: float) -> Tensor:
    r"""
    Return the EDM per-sample loss weight ``(sigma**2 + sigma_data**2) / (sigma * sigma_data)**2``.

    This is the standard ``1 / c_out(sigma)**2`` weighting from Karras et al.; it upweights low-noise
    samples where the denoiser must be precise.

    Args:
        noise_level: Per-sample noise standard deviations ``sigma``.
        sigma_data: EDM data scale used by the model's preconditioner.
    """

    return (noise_level**2 + sigma_data**2) / ((noise_level * sigma_data) ** 2 + 1e-8)


def edm_denoising_loss(pred: Any, model_input: Any, *, sigma_data: float = 16.0) -> Tensor:
    r"""
    Compute the EDM-weighted coordinate denoising loss over the valid atoms.

    Args:
        pred: A diffusion model output carrying ``coordinates`` and ``augmented_target_coordinates`` of
            shape ``(B, S, A, 3)`` and ``noise_level`` of shape ``(B, S)``, all on the augmented frame.
        model_input: The batched model Input, masked by ``atom_mask`` of shape ``(B, A)``.
        sigma_data: EDM data scale (matches the model's denoiser preconditioner).

    An empty atom scope gives a NaN mean -- surfaced, not hidden by a fallback.
    """

    squared_error = (pred.coordinates - pred.augmented_target_coordinates).pow(2).sum(dim=-1)  # (B, S, A)
    weight = edm_loss_weight(pred.noise_level, sigma_data).unsqueeze(-1)  # (B, S, 1)
    weighted_error = weight * squared_error  # (B, S, A)
    mask = model_input.atom_mask.to(weighted_error.dtype).unsqueeze(1).expand_as(weighted_error)  # (B, S, A)
    return (weighted_error * mask).sum() / mask.sum()


def denoising_criterion(pred: Any, target: Any, *, sigma_data: float = 16.0) -> Tensor:
    r"""
    DanLing-shaped EDM denoising loss shared by the internally-noising diffusion families.

    Args:
        pred: The diffusion model output (see [`edm_denoising_loss`][ofoldx.modules.diffusion.loss]).
        target: The ``{"input", "target"}`` mapping from
            [`collate_denoising`][ofoldx.modules.collate.collate_denoising]; the loss masks by the model
            input's ``atom_mask``.
        sigma_data: EDM data scale (matches the model's denoiser preconditioner default).
    """

    return edm_denoising_loss(pred, target["input"], sigma_data=sigma_data)
