r"""
EDM noise schedules.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class EdmScheduler(nn.Module):
    r"""
    Construct a rho-spaced EDM noise schedule.

    The configured endpoints are expressed before multiplication by ``sigma_data``.
    The lower endpoint controls grid spacing; its final grid entry is replaced by zero.

    Args:
        max_noise_level: Upper interpolation endpoint before ``sigma_data`` scaling.
        min_noise_level: Lower interpolation endpoint used to shape the grid before the
            terminal entry is replaced by zero.
        rho: Exponent controlling the concentration of steps at low noise.
        sigma_data: Scale applied to every nonzero schedule entry.
    """

    def __init__(
        self,
        *,
        max_noise_level: float = 160.0,
        min_noise_level: float = 4e-4,
        rho: float = 7.0,
        sigma_data: float = 16.0,
    ) -> None:
        super().__init__()
        if not math.isfinite(max_noise_level) or max_noise_level <= 0:
            raise ValueError("max_noise_level must be a finite positive number")
        self.max_noise_level = max_noise_level
        self.min_noise_level = min_noise_level
        self.rho = rho
        self.sigma_data = sigma_data

    def forward(
        self,
        num_sampling_steps: int = 200,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype,
        max_noise_level: float | None = None,
    ) -> Tensor:
        r"""
        Return descending EDM noise levels followed by a terminal zero.

        Args:
            num_sampling_steps: Number of consecutive sampler transitions.
            device: Device of the returned schedule.
            dtype: Floating-point dtype of the returned schedule.
            max_noise_level: Optional upper interpolation endpoint before ``sigma_data``
                scaling.

        Returns:
            A tensor with shape ``(num_sampling_steps + 1,)``. The first entry is
            ``sigma_data * max_noise_level`` and the final entry is zero.

        Raises:
            ValueError: If the step count or maximum noise level is invalid.
        """

        if num_sampling_steps < 1:
            raise ValueError(f"num_sampling_steps must be positive, got {num_sampling_steps}")
        step_size = 1 / num_sampling_steps
        step_indices = torch.arange(num_sampling_steps + 1, device=device, dtype=dtype)
        maximum = self.max_noise_level if max_noise_level is None else max_noise_level
        if max_noise_level is not None:
            if torch.compiler.is_compiling():
                torch._check(maximum > 0)
                torch._check(maximum < math.inf)
            elif not math.isfinite(maximum) or maximum <= 0:
                raise ValueError("max_noise_level must be a finite positive number")
        max_root = maximum ** (1 / self.rho)
        min_root = self.min_noise_level ** (1 / self.rho)
        schedule_base = max_root + step_indices * step_size * (min_root - max_root)
        schedule = self.sigma_data * schedule_base**self.rho
        schedule[..., -1] = 0
        return schedule
