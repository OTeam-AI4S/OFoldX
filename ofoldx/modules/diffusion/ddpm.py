r"""
Discrete denoising-diffusion probability-model equations.
"""

from __future__ import annotations

import torch
from torch import Tensor


def ddpm_forward_step(
    values: Tensor,
    beta: Tensor,
    *,
    variance_scale: float = 1.0,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    r"""
    Sample one Gaussian forward-diffusion step.

    Returns the sampled values and their displacement from the input.
    """

    mean = torch.sqrt(1 - beta) * values
    std = torch.sqrt(torch.ones_like(values) * beta * variance_scale)
    sampled = mean + torch.randn(values.shape, dtype=values.dtype, device=values.device, generator=generator) * std
    return sampled, sampled - values


def ddpm_posterior(
    noisy: Tensor,
    predicted_clean: Tensor,
    timestep: int,
    beta: Tensor,
    alpha_bar: Tensor,
    *,
    eps: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    r"""
    Compute ``q(x_(t-1) | x_t, x_0)`` mean and scalar variance.

    ``timestep`` is one-indexed and must be greater than one.
    """

    index = timestep - 1
    beta = beta.to(device=noisy.device, dtype=noisy.dtype)
    alpha_bar = alpha_bar.to(device=noisy.device, dtype=noisy.dtype)
    variance = ((1 - alpha_bar[index - 1]) / (1 - alpha_bar[index])) * beta[index]
    clean_term = torch.sqrt(alpha_bar[index - 1] + eps) * beta[index] / (1 - alpha_bar[index]) * predicted_clean
    noisy_term = torch.sqrt(1 - beta[index] + eps) * (1 - alpha_bar[index - 1]) / (1 - alpha_bar[index]) * noisy
    return clean_term + noisy_term, variance
