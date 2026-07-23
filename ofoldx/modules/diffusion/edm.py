r"""
EDM preconditioning (Karras et al.).

The ``c_skip`` / ``c_out`` / ``c_in`` / ``c_noise`` reweighting that keeps a diffusion denoiser's input and target
unit-variance across noise levels. Pure functions of the noise level ``sigma`` and ``sigma_data`` — no learned
parameters or persistent state.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class EdmPreconditioner(nn.Module):
    r"""
    EDM input/output preconditioning coefficients ``c_skip`` / ``c_out`` / ``c_in`` / ``c_noise``.

    Implements the ``c_skip`` / ``c_out`` / ``c_in`` / ``c_noise`` reweighting of Karras et al. that keeps the
    denoiser's input and target unit-variance across noise levels. The module holds no learned parameters and no
    persistent state; every method is a closed-form function of the noise level ``sigma`` and ``sigma_data``.

    Args:
        sigma_data: Data noise scale used by EDM preconditioning.

    Examples:
        >>> import torch
        >>> pre = EdmPreconditioner(sigma_data=16.0)
        >>> pre.c_skip(torch.tensor([0.0, 16.0])).tolist()
        [1.0, 0.5]
    """

    def __init__(self, sigma_data: float = 16.0) -> None:
        super().__init__()
        self.sigma_data = sigma_data

    def c_skip(self, sigma: Tensor) -> Tensor:
        r"""
        Return the EDM skip coefficient that weights the noisy-input passthrough.

        The skip weight ``sigma_data**2 / (sigma**2 + sigma_data**2)`` is ``1`` at zero noise (trust the input) and
        decays toward ``0`` as ``sigma`` grows (trust the network).

        Args:
            sigma: Noise level tensor.

        Returns:
            Skip coefficient with the same shape as ``sigma``.

        Examples:
            >>> import torch
            >>> pre = EdmPreconditioner(sigma_data=16.0)
            >>> pre.c_skip(torch.tensor([0.0, 16.0])).tolist()
            [1.0, 0.5]
        """

        return self.sigma_data**2 / (sigma**2 + self.sigma_data**2)

    def c_out(self, sigma: Tensor) -> Tensor:
        r"""
        Return the EDM output coefficient that scales the network prediction.

        Complements [`c_skip`][ofoldx.modules.diffusion.edm.EdmPreconditioner.c_skip]: it is ``0`` at zero noise
        and rises with ``sigma`` so the denoiser's residual carries the correction the skip term cannot.

        Args:
            sigma: Noise level tensor.

        Returns:
            Output coefficient with the same shape as ``sigma``.

        Examples:
            >>> import torch
            >>> pre = EdmPreconditioner(sigma_data=16.0)
            >>> [round(float(x), 4) for x in pre.c_out(torch.tensor([0.0, 16.0]))]
            [0.0, 11.3137]
        """

        return sigma * self.sigma_data / torch.sqrt(self.sigma_data**2 + sigma**2)

    def c_in(self, sigma: Tensor) -> Tensor:
        r"""
        Return the EDM input coefficient that normalizes the noisy coords before the network.

        Scaling the noisy input by ``1 / sqrt(sigma**2 + sigma_data**2)`` gives the denoiser a unit-variance input at
        every noise level.

        Args:
            sigma: Noise level tensor.

        Returns:
            Input coefficient with the same shape as ``sigma``.

        Examples:
            >>> import torch
            >>> pre = EdmPreconditioner(sigma_data=16.0)
            >>> [round(float(x), 4) for x in pre.c_in(torch.tensor([0.0, 16.0]))]
            [0.0625, 0.0442]
        """

        return 1 / torch.sqrt(sigma**2 + self.sigma_data**2)

    def c_noise(self, sigma: Tensor) -> Tensor:
        r"""
        Return the EDM noise embedding coordinate fed to the noise conditioner.

        The denoiser is conditioned on ``0.25 * log(sigma / sigma_data)`` rather than the raw ``sigma`` so the
        conditioning input stays in a compact range across the whole schedule.

        Args:
            sigma: Noise level tensor.

        Returns:
            Log-scaled noise coordinate with the same shape as ``sigma``.

        Examples:
            >>> import torch
            >>> pre = EdmPreconditioner(sigma_data=16.0)
            >>> pre.c_noise(torch.tensor([16.0])).round(decimals=4).tolist()
            [0.0]
        """

        return torch.log(sigma / self.sigma_data) * 0.25

    def scale_noisy_coords(self, noisy_coords: Tensor, noise_level: Tensor) -> Tensor:
        r"""
        Scale the noisy input coords by ``c_in`` before handing them to the denoiser.

        Convenience wrapper that broadcasts ``noise_level`` to the coord rank, divides by
        ``sqrt(sigma_data**2 + noise_level**2)``, and restores the input dtype.

        Args:
            noisy_coords: Noisy coords of shape `[..., N_atom, 3]`.
            noise_level: Noise level broadcastable to ``noisy_coords``.

        Returns:
            Scaled noisy coords of shape `[..., N_atom, 3]` in the ``noisy_coords`` dtype.

        Examples:
            >>> import torch
            >>> pre = EdmPreconditioner(sigma_data=16.0)
            >>> pre.scale_noisy_coords(torch.ones(2, 3), torch.tensor(0.0)).round(decimals=4).tolist()
            [[0.0625, 0.0625, 0.0625], [0.0625, 0.0625, 0.0625]]
        """

        noise_level = self.broadcast_noise_level(noise_level, noisy_coords)
        scaled_coords = noisy_coords / torch.sqrt(self.sigma_data**2 + noise_level**2)
        return scaled_coords.to(dtype=noisy_coords.dtype)

    def combine_update(self, noisy_coords: Tensor, noise_level: Tensor, coordinate_update: Tensor) -> Tensor:
        r"""
        Combine the skip connection and the network output into denoised coords.

        Applies the EDM output parametrization ``c_skip * noisy_coords + c_out * coordinate_update``: at zero noise
        this returns the input unchanged, while at high noise it trusts the network's predicted update.

        Args:
            noisy_coords: Original noisy coords of shape `[..., N_atom, 3]`.
            noise_level: Noise level broadcastable to ``coordinate_update``.
            coordinate_update: Network-predicted coordinate update of shape `[..., N_atom, 3]`.

        Returns:
            Denoised coords of shape `[..., N_atom, 3]` in the ``coordinate_update`` dtype.

        Examples:
            >>> import torch
            >>> pre = EdmPreconditioner(sigma_data=16.0)
            >>> pre.combine_update(torch.ones(1, 3), torch.tensor(0.0), torch.zeros(1, 3)).tolist()
            [[1.0, 1.0, 1.0]]
        """

        noise_level = self.broadcast_noise_level(noise_level, coordinate_update)
        noisy_coords = noisy_coords.to(dtype=coordinate_update.dtype)
        denoised_coords = self.c_skip(noise_level) * noisy_coords + self.c_out(noise_level) * coordinate_update
        return denoised_coords.to(dtype=coordinate_update.dtype)

    @staticmethod
    def broadcast_noise_level(noise_level: Tensor, target: Tensor) -> Tensor:
        r"""
        Broadcast a per-sample noise-level tensor to a target coordinate rank.

        Casts ``noise_level`` to ``target``'s dtype and appends trailing singleton dimensions so it
        aligns with the leading axes of ``target`` and broadcasts over the trailing coordinate axes.
        Both tensors follow the module's ordinary same-device contract.

        Args:
            noise_level: Noise level tensor.
            target: Target tensor whose rank should be matched.

        Returns:
            ``noise_level`` reshaped with trailing singleton dimensions to match ``target``'s rank.

        Examples:
            >>> import torch
            >>> pre = EdmPreconditioner()
            >>> pre.broadcast_noise_level(torch.tensor([1.0, 2.0]), torch.zeros(2, 4, 3)).shape
            torch.Size([2, 1, 1])
        """

        noise_level = noise_level.to(dtype=target.dtype)
        return noise_level.reshape(
            *noise_level.shape,
            *((1,) * (target.dim() - noise_level.dim())),
        )
