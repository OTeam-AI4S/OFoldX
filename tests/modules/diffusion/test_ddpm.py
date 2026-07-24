from __future__ import annotations

import torch

from ofoldx.modules.diffusion.ddpm import ddpm_forward_step, ddpm_posterior


def test_ddpm_equations() -> None:
    values = torch.tensor([[1.0, -1.0]])
    sampled, displacement = ddpm_forward_step(
        values,
        torch.tensor(0.0),
        generator=torch.Generator().manual_seed(0),
    )
    torch.testing.assert_close(sampled, values)
    torch.testing.assert_close(displacement, torch.zeros_like(values))

    beta = torch.tensor([0.1, 0.2])
    alpha_bar = torch.cumprod(1 - beta, dim=0)
    mean, variance = ddpm_posterior(values, torch.zeros_like(values), 2, beta, alpha_bar)
    assert mean.shape == values.shape
    torch.testing.assert_close(variance, torch.tensor(0.0714286), atol=1e-6, rtol=0)
