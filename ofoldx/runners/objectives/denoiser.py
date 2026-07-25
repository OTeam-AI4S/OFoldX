r"""
EDM coordinate-denoising objective for externally-noised denoiser models.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from ..registry import OBJECTIVES
from .diffusion import edm_weight


@OBJECTIVES.register("denoiser")
class DenoiserObjective:
    r"""
    Compute an EDM-weighted denoising loss for models that denoise externally
    supplied noisy coordinates.

    Unlike [`DiffusionObjective`][ofoldx.runners.objectives.diffusion.DiffusionObjective]
    (which relies on a model that noises its target internally), this objective owns
    the corruption: the model forward is
    ``model(input, noisy_coordinates=(B, S, A, 3), noise_level=(B, S)) ->
    output.coordinates (B, S, A, 3)``. Mirroring the shared EDM training path, it draws
    per-sample noise levels ``sigma_data * exp(noise_log_mean + noise_log_std * N(0, 1))``,
    corrupts the observed coordinates, and takes the EDM-weighted loss against them over
    atoms that are both observed and valid.

    Args:
        sigma_data: EDM data scale; matches the denoiser preconditioner.
        noise_log_mean: Mean of the log-noise training distribution.
        noise_log_std: Standard deviation of the log-noise training distribution.
        num_samples: Number of noisy coordinate samples drawn per structure.
    """

    def __init__(
        self,
        *,
        sigma_data: float = 16.0,
        noise_log_mean: float = -1.2,
        noise_log_std: float = 1.5,
        num_samples: int = 1,
    ) -> None:
        self.sigma_data = sigma_data
        self.noise_log_mean = noise_log_mean
        self.noise_log_std = noise_log_std
        self.num_samples = num_samples

    def __call__(self, model: nn.Module, batch: Any) -> tuple[Any, Tensor]:
        target: Tensor = batch.observed_positions  # (B, A, 3)
        mask = (batch.observed_mask & batch.atom_mask).to(target.dtype)  # (B, A)
        batch_size, atoms = mask.shape
        samples = self.num_samples

        clean = target.unsqueeze(1).expand(batch_size, samples, atoms, 3)  # (B, S, A, 3)
        normal = torch.randn(batch_size, samples, device=target.device, dtype=target.dtype)
        noise_level = self.sigma_data * (self.noise_log_mean + self.noise_log_std * normal).exp()  # (B, S)
        noisy = clean + noise_level[..., None, None] * torch.randn_like(clean)

        output = model(batch, noisy_coordinates=noisy, noise_level=noise_level)
        denoised: Tensor = output.coordinates  # (B, S, A, 3)

        squared_error = (denoised - clean).pow(2).sum(dim=-1)  # (B, S, A)
        weight = edm_weight(noise_level, self.sigma_data).unsqueeze(-1)  # (B, S, 1)
        weighted_error = weight * squared_error  # (B, S, A)

        loss_mask = mask.unsqueeze(1).expand(batch_size, samples, atoms)  # (B, S, A)
        denominator = loss_mask.sum().clamp_min(torch.finfo(weighted_error.dtype).tiny)
        loss = (weighted_error * loss_mask).sum() / denominator
        return output, loss
