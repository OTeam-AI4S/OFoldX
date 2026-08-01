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

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from ofoldx.modules.diffusion.ddpm import ddpm_forward_step, ddpm_posterior


def test_ddpm_forward_step() -> None:
    values = torch.tensor([[1.0, -1.0]], dtype=torch.float64, requires_grad=True)
    sampled, displacement = ddpm_forward_step(
        values,
        torch.tensor(0.0, dtype=torch.float64),
        generator=torch.Generator().manual_seed(0),
    )

    torch.testing.assert_close(sampled, values)
    torch.testing.assert_close(displacement, torch.zeros_like(values))
    assert sampled.dtype == values.dtype
    assert sampled.device == values.device
    sampled.sum().backward()
    torch.testing.assert_close(values.grad, torch.ones_like(values))


def test_ddpm_posterior() -> None:
    noisy = torch.tensor([[1.0, -1.0]], dtype=torch.float64, requires_grad=True)
    predicted_clean = torch.zeros_like(noisy, requires_grad=True)
    beta = torch.tensor([0.1, 0.2], dtype=torch.float32)
    alpha_bar = torch.cumprod(1 - beta, dim=0)

    mean, variance = ddpm_posterior(noisy, predicted_clean, 2, beta, alpha_bar)

    expected_variance = (1 - alpha_bar[0].double()) / (1 - alpha_bar[1].double()) * beta[1].double()
    torch.testing.assert_close(variance, expected_variance)
    assert mean.shape == noisy.shape
    assert mean.dtype == noisy.dtype
    mean.sum().backward()
    assert noisy.grad is not None
    assert predicted_clean.grad is not None


def _gaussian_precision_product(
    noisy: Tensor, predicted_clean: Tensor, timestep: int, beta: Tensor
) -> tuple[Tensor, Tensor]:
    previous_alpha_bar = (1 - beta[: timestep - 1]).prod()
    prior_mean = previous_alpha_bar.sqrt() * predicted_clean
    prior_variance = 1 - previous_alpha_bar
    likelihood_scale = (1 - beta[timestep - 1]).sqrt()
    likelihood_variance = beta[timestep - 1]
    posterior_precision = 1 / prior_variance + likelihood_scale.square() / likelihood_variance
    variance = 1 / posterior_precision
    mean = variance * (prior_mean / prior_variance + likelihood_scale * noisy / likelihood_variance)
    return mean, variance


@pytest.mark.parametrize("schedule", [(0.1, 0.2, 0.3), (0.001, 0.002, 0.003), (0.9, 0.9999, 0.999999)])
@pytest.mark.parametrize("timestep", [2, 3])
def test_complete_posterior_matches_gaussian_product_and_all_schedule_input_vjps(schedule, timestep) -> None:
    generator = torch.Generator().manual_seed(418)
    noisy = torch.randn(2, 4, 3, generator=generator, dtype=torch.float64, requires_grad=True)
    predicted_clean = torch.randn(2, 4, 3, generator=generator, dtype=torch.float64, requires_grad=True)
    beta = torch.tensor(schedule, dtype=torch.float64, requires_grad=True)
    cotangent = torch.randn(noisy.shape, generator=generator, dtype=torch.float64)

    actual_mean, actual_variance = ddpm_posterior(noisy, predicted_clean, timestep, beta, (1 - beta).cumprod(0))
    expected_mean, expected_variance = _gaussian_precision_product(noisy, predicted_clean, timestep, beta)
    torch.testing.assert_close(actual_mean, expected_mean, rtol=2e-11, atol=2e-11)
    torch.testing.assert_close(actual_variance, expected_variance, rtol=2e-11, atol=2e-11)
    operands = (noisy, predicted_clean, beta)
    actual_vjps = torch.autograd.grad(
        (actual_mean * cotangent).sum() + 0.37 * actual_variance, operands, retain_graph=True
    )
    expected_vjps = torch.autograd.grad((expected_mean * cotangent).sum() + 0.37 * expected_variance, operands)
    for actual, expected in zip(actual_vjps, expected_vjps, strict=True):
        torch.testing.assert_close(actual, expected, rtol=2e-11, atol=2e-11)
