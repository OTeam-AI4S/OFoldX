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
Elucidated-diffusion-model preconditioning.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .sampling import draw_normal


def lognormal_training_noise(
    batch_size: int,
    *,
    sigma_data: float,
    log_mean: float,
    log_std: float,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None = None,
) -> Tensor:
    r"""Draw EDM training noise levels from the configured log-normal law.

    Draw on the supplied generator's device, then place the normal samples on
    the requested output device before applying the physical noise scale.
    """

    normal = draw_normal((batch_size,), device=device, dtype=dtype, generator=generator)
    return sigma_data * (log_mean + log_std * normal).exp()


def edm_loss_weight(noise_level: Tensor, sigma_data: float) -> Tensor:
    r"""
    Return the EDM loss weight ``1 / c_out(sigma)**2``.

    Args:
        noise_level: Per-sample noise standard deviations.
        sigma_data: Data noise scale used by EDM preconditioning.

    Returns:
        ``(sigma**2 + sigma_data**2) / (sigma * sigma_data)**2``.
    """

    return noise_level.reciprocal().square() + noise_level.new_tensor(sigma_data).reciprocal().square()


class EdmPreconditioner(nn.Module):
    r"""
    Compute EDM input, output, skip, and noise-conditioning coefficients.

    Args:
        sigma_data: Data noise scale used by EDM preconditioning.
    """

    def __init__(self, sigma_data: float = 16.0) -> None:
        super().__init__()
        self.sigma_data = sigma_data

    def c_skip(self, sigma: Tensor) -> Tensor:
        r"""
        Compute ``sigma_data**2 / (sigma**2 + sigma_data**2)``.

        Args:
            sigma: Noise level tensor.

        Returns:
            Skip coefficient with the same shape as ``sigma``.
        """

        return (sigma.new_full((), self.sigma_data) / torch.hypot(sigma, sigma.new_full((), self.sigma_data))).square()

    def c_out(self, sigma: Tensor) -> Tensor:
        r"""
        Compute ``sigma * sigma_data / sqrt(sigma**2 + sigma_data**2)``.

        Args:
            sigma: Noise level tensor.

        Returns:
            Output coefficient with the same shape as ``sigma``.
        """

        return sigma / torch.hypot(sigma, sigma.new_full((), self.sigma_data)) * self.sigma_data

    def c_in(self, sigma: Tensor) -> Tensor:
        r"""
        Compute ``1 / sqrt(sigma**2 + sigma_data**2)``.

        Args:
            sigma: Noise level tensor.

        Returns:
            Input coefficient with the same shape as ``sigma``.
        """

        return torch.hypot(sigma, sigma.new_full((), self.sigma_data)).reciprocal()

    def c_noise(self, sigma: Tensor) -> Tensor:
        r"""
        Compute the noise-conditioning value ``0.25 * log(sigma / sigma_data)``.

        Args:
            sigma: Noise level tensor.

        Returns:
            Log-scaled noise coordinate with the same shape as ``sigma``.
        """

        return (torch.log(sigma) - sigma.new_full((), self.sigma_data).log()) * 0.25

    def scale_noisy_coords(self, noisy_coords: Tensor, noise_level: Tensor) -> Tensor:
        r"""
        Scale noisy coordinates by the EDM input coefficient.

        Args:
            noisy_coords: Noisy coordinates with shape ``[..., N_atom, 3]``.
            noise_level: Noise level broadcastable to ``noisy_coords``.

        Returns:
            Scaled coordinates with the same shape as ``noisy_coords`` and native promoted dtype.
        """

        noise_level = self.broadcast_noise_level(noise_level, noisy_coords)
        return noisy_coords * self.c_in(noise_level)

    def combine_update(self, noisy_coords: Tensor, noise_level: Tensor, coordinate_update: Tensor) -> Tensor:
        r"""
        Combine coordinates and a network update with EDM output coefficients.

        Args:
            noisy_coords: Noisy coordinates with shape ``[..., N_atom, 3]``.
            noise_level: Noise level broadcastable to ``coordinate_update``.
            coordinate_update: Predicted coordinate update with shape
                ``[..., N_atom, 3]``.

        Returns:
            Denoised coordinates with the update shape and native promoted dtype.
        """

        noise_level = self.broadcast_noise_level(noise_level, coordinate_update)
        return self.c_skip(noise_level) * noisy_coords + self.c_out(noise_level) * coordinate_update

    @staticmethod
    def broadcast_noise_level(noise_level: Tensor, target: Tensor) -> Tensor:
        r"""
        Append singleton dimensions so noise levels broadcast over ``target``.

        Args:
            noise_level: Noise level tensor.
            target: Target tensor whose rank should be matched.

        Returns:
            Noise levels with their original dtype and the same rank as ``target``.
        """

        return noise_level.reshape(
            *noise_level.shape,
            *((1,) * (target.dim() - noise_level.dim())),
        )
