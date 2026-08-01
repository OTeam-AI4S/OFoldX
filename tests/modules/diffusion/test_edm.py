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

import math

import pytest
import torch

from ofoldx.modules.diffusion import EdmPreconditioner, edm_loss_weight


@pytest.mark.parametrize("update", [0.0, 5.0])
def test_edm_coefficients(update: float) -> None:
    preconditioner = EdmPreconditioner(sigma_data=16.0)
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


def test_edm_loss_weight() -> None:
    noise_level = torch.tensor([0.5, 2.0, 8.0])
    sigma_data = 16.0

    expected = (noise_level.square() + sigma_data**2) / (noise_level * sigma_data).square()

    torch.testing.assert_close(edm_loss_weight(noise_level, sigma_data), expected)


def test_native_edm_coefficients_avoid_intermediate_half_overflow() -> None:
    for dtype, tolerance in (
        (torch.float64, 1e-12),
        (torch.float32, 2e-6),
        (torch.float16, 2e-3),
        (torch.bfloat16, 2e-2),
    ):
        sigma = torch.tensor([0.5, 16.0, 256.0, 1024.0], dtype=dtype, requires_grad=True)
        reference = sigma.detach().double().requires_grad_()
        preconditioner = EdmPreconditioner(16.0)
        norm = (reference.square() + 256.0).sqrt()
        actual = (
            preconditioner.c_skip(sigma),
            preconditioner.c_out(sigma),
            preconditioner.c_in(sigma),
            edm_loss_weight(sigma, 16.0),
        )
        expected = (
            256.0 / norm.square(),
            reference * 16.0 / norm,
            1 / norm,
            (reference.square() + 256.0) / (reference * 16.0).square(),
        )
        for observed, wanted in zip(actual, expected):
            assert observed.dtype == dtype
            torch.testing.assert_close(observed.double(), wanted, rtol=tolerance, atol=torch.finfo(dtype).tiny)
        derivative = torch.autograd.grad(sum(x.sum() for x in actual), sigma)[0]
        wanted_derivative = torch.autograd.grad(sum(x.sum() for x in expected), reference)[0]
        # Bound the Jacobian by its elementary terms before cancellation. At
        # large sigma the two c_out terms nearly cancel, although each is much
        # larger than the combined coefficient derivative.
        derivative_terms = torch.stack(
            (
                -2 * 256.0 * reference / norm.pow(4),
                16.0 / norm,
                -16.0 * reference.square() / norm.pow(3),
                -reference / norm.pow(3),
                2.0 / (256.0 * reference),
                -2.0 * (reference.square() + 256.0) / (256.0 * reference.pow(3)),
            )
        )
        torch.testing.assert_close(derivative_terms.sum(0), wanted_derivative, rtol=1e-10, atol=1e-14)
        epsilon = torch.finfo(dtype).eps
        gamma = 8 * epsilon / (1 - 8 * epsilon)
        bound = gamma * derivative_terms.abs().sum(0) + torch.finfo(dtype).tiny * epsilon
        assert torch.isfinite(derivative).all()
        assert ((derivative.double() - wanted_derivative).abs() <= bound).all(), (dtype, derivative, wanted_derivative)
