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
Discrete denoising-diffusion probability-model equations.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .sampling import draw_normal


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
    standard_deviation = torch.sqrt(torch.ones_like(values) * beta * variance_scale)
    noise = draw_normal(values.shape, dtype=values.dtype, device=values.device, generator=generator)
    sampled = mean + standard_deviation * noise
    return sampled, sampled - values


def ddpm_posterior(
    noisy: Tensor,
    predicted_clean: Tensor,
    timestep: int,
    beta: Tensor,
    alpha_bar: Tensor,
) -> tuple[Tensor, Tensor]:
    r"""
    Compute the mean and variance of :math:`q(x_{t-1} \mid x_t, x_0)`.

    ``timestep`` is one-indexed and greater than one in the reverse process.
    """

    index = timestep - 1
    beta = beta.to(device=noisy.device, dtype=noisy.dtype)
    alpha_bar = alpha_bar.to(device=noisy.device, dtype=noisy.dtype)
    variance = ((1 - alpha_bar[index - 1]) / (1 - alpha_bar[index])) * beta[index]
    clean_term = torch.sqrt(alpha_bar[index - 1]) * beta[index] / (1 - alpha_bar[index]) * predicted_clean
    noisy_term = torch.sqrt(1 - beta[index]) * (1 - alpha_bar[index - 1]) / (1 - alpha_bar[index]) * noisy
    return clean_term + noisy_term, variance
