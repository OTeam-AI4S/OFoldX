from __future__ import annotations

import torch

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
