r"""
Elucidated-diffusion-model preconditioning.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


def edm_loss_weight(noise_level: Tensor, sigma_data: float) -> Tensor:
    r"""
    Return the EDM loss weight ``1 / c_out(sigma)**2``.

    Args:
        noise_level: Per-sample noise standard deviations.
        sigma_data: Data noise scale used by EDM preconditioning.

    Returns:
        ``(sigma**2 + sigma_data**2) / (sigma * sigma_data)**2``.
    """

    return (noise_level**2 + sigma_data**2) / (noise_level * sigma_data) ** 2


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

        return self.sigma_data**2 / (sigma**2 + self.sigma_data**2)

    def c_out(self, sigma: Tensor) -> Tensor:
        r"""
        Compute ``sigma * sigma_data / sqrt(sigma**2 + sigma_data**2)``.

        Args:
            sigma: Noise level tensor.

        Returns:
            Output coefficient with the same shape as ``sigma``.
        """

        return sigma * self.sigma_data / torch.sqrt(self.sigma_data**2 + sigma**2)

    def c_in(self, sigma: Tensor) -> Tensor:
        r"""
        Compute ``1 / sqrt(sigma**2 + sigma_data**2)``.

        Args:
            sigma: Noise level tensor.

        Returns:
            Input coefficient with the same shape as ``sigma``.
        """

        return 1 / torch.sqrt(sigma**2 + self.sigma_data**2)

    def c_noise(self, sigma: Tensor) -> Tensor:
        r"""
        Compute the noise-conditioning value ``0.25 * log(sigma / sigma_data)``.

        Args:
            sigma: Noise level tensor.

        Returns:
            Log-scaled noise coordinate with the same shape as ``sigma``.
        """

        return torch.log(sigma / self.sigma_data) * 0.25

    def scale_noisy_coords(self, noisy_coords: Tensor, noise_level: Tensor) -> Tensor:
        r"""
        Scale noisy coordinates by the EDM input coefficient.

        Args:
            noisy_coords: Noisy coordinates with shape ``[..., N_atom, 3]``.
            noise_level: Noise level broadcastable to ``noisy_coords``.

        Returns:
            Scaled coordinates with the same shape and dtype as ``noisy_coords``.
        """

        noise_level = self.broadcast_noise_level(noise_level, noisy_coords)
        scaled_coords = noisy_coords / torch.sqrt(self.sigma_data**2 + noise_level**2)
        return scaled_coords.to(dtype=noisy_coords.dtype)

    def combine_update(self, noisy_coords: Tensor, noise_level: Tensor, coordinate_update: Tensor) -> Tensor:
        r"""
        Combine coordinates and a network update with EDM output coefficients.

        Args:
            noisy_coords: Noisy coordinates with shape ``[..., N_atom, 3]``.
            noise_level: Noise level broadcastable to ``coordinate_update``.
            coordinate_update: Predicted coordinate update with shape
                ``[..., N_atom, 3]``.

        Returns:
            Denoised coordinates with the shape and dtype of ``coordinate_update``.
        """

        noise_level = self.broadcast_noise_level(noise_level, coordinate_update)
        noisy_coords = noisy_coords.to(dtype=coordinate_update.dtype)
        denoised_coords = self.c_skip(noise_level) * noisy_coords + self.c_out(noise_level) * coordinate_update
        return denoised_coords.to(dtype=coordinate_update.dtype)

    @staticmethod
    def broadcast_noise_level(noise_level: Tensor, target: Tensor) -> Tensor:
        r"""
        Append singleton dimensions so noise levels broadcast over ``target``.

        Args:
            noise_level: Noise level tensor.
            target: Target tensor whose rank should be matched.

        Returns:
            Noise levels in ``target.dtype`` with the same rank as ``target``.
        """

        noise_level = noise_level.to(dtype=target.dtype)
        return noise_level.reshape(
            *noise_level.shape,
            *((1,) * (target.dim() - noise_level.dim())),
        )
