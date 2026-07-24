r"""
EDM diffusion training objective.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from ..registry import OBJECTIVES


def edm_weight(sigma: Tensor, sigma_data: float) -> Tensor:
    r"""
    Compute the EDM denoising loss weight.

    Args:
        sigma: Per-sample noise levels.
        sigma_data: Data standard deviation used by EDM preconditioning.

    Returns:
        Weights with the same shape as ``sigma``.
    """

    return (sigma**2 + sigma_data**2) / ((sigma * sigma_data) ** 2 + 1e-8)


@OBJECTIVES.register("diffusion")
class DiffusionObjective:
    r"""
    Compute an EDM-weighted coordinate denoising loss.

    Args:
        sigma_data: Data standard deviation used by EDM preconditioning.
        num_samples: Number of noisy coordinate samples requested from the model.
    """

    def __init__(self, *, sigma_data: float = 16.0, num_samples: int = 1) -> None:
        self.sigma_data = sigma_data
        self.num_samples = num_samples

    def __call__(self, model: nn.Module, batch: Any) -> tuple[Any, Tensor]:
        output = model(batch, num_samples=self.num_samples)

        denoised: Tensor = output.denoised_coords
        target: Tensor = output.augmented_target
        batch_size, num_samples = denoised.shape[0], denoised.shape[1]

        squared_error = (denoised - target).pow(2).sum(dim=-1)

        weight = edm_weight(output.noise_level, self.sigma_data).reshape(batch_size, num_samples, 1)
        weighted_error = weight * squared_error

        target_mask: Tensor | None = getattr(batch, "target_mask", None)
        loss_mask = batch.atom_mask if target_mask is None else target_mask
        mask = loss_mask.unsqueeze(1).to(weighted_error.dtype).expand_as(weighted_error)
        denominator = mask.sum().clamp_min(torch.finfo(weighted_error.dtype).tiny)
        loss = (weighted_error * mask).sum() / denominator

        return output, loss
