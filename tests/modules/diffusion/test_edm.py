from __future__ import annotations

import math

import pytest
import torch

from ofoldx.modules.diffusion import EdmPreconditioner


@pytest.fixture
def preconditioner() -> EdmPreconditioner:
    return EdmPreconditioner(sigma_data=16.0)


@pytest.mark.parametrize("update", [0.0, 5.0])
def test_edm_coefficients(preconditioner: EdmPreconditioner, update: float) -> None:
    coords = torch.ones(1, 2, 5, 3)
    noise_level = torch.full((1, 2), 2.0)
    norm = math.sqrt(16.0**2 + 2.0**2)

    scaled = preconditioner.scale_noisy_coords(coords, noise_level)
    combined = preconditioner.combine_update(coords, noise_level, torch.full_like(coords, update))

    assert scaled.shape == coords.shape
    assert combined.shape == coords.shape
    torch.testing.assert_close(scaled, torch.full_like(coords, 1.0 / norm))
    c_skip = 16.0**2 / norm**2
    c_out = 2.0 * 16.0 / norm
    torch.testing.assert_close(combined, torch.full_like(coords, c_skip + c_out * update))
