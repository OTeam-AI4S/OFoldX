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
EDM noise schedules.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from scipy.special import betainc
from torch import Tensor, nn


def beta_scale_schedule(
    num_sampling_steps: int,
    *,
    min_scale: float,
    max_scale: float,
    alpha: float,
    beta: float,
    device: torch.device | str | None = None,
    dtype: torch.dtype,
) -> Tensor:
    r"""Return ``min_scale + (max_scale-min_scale) * I_t(alpha,beta)``.

    The time grid spans zero through one. A single transition uses time zero
    and therefore the minimum scale. The output has one coefficient per transition.
    """

    times = np.linspace(0, 1, num_sampling_steps)
    scales = min_scale + (max_scale - min_scale) * betainc(alpha, beta, times)
    return torch.as_tensor(scales, device=device, dtype=dtype)


def karras_noise_schedule(
    num_sampling_steps: int,
    *,
    max_noise_level: float,
    min_noise_level: float,
    rho: float,
    sigma_data: float,
    device: torch.device | str | None = None,
    dtype: torch.dtype,
) -> Tensor:
    r"""
    Return the Karras rho-spaced noise levels with an appended terminal zero.

    The ramp reaches its lower endpoint exactly, so the last nonzero entry is
    ``sigma_data * min_noise_level`` and the terminal zero is appended after it. This is the grid
    used by the released Boltz-lineage samplers, and it differs from
    [`EdmScheduler`][ofoldx.modules.diffusion.schedule.EdmScheduler], whose ramp spans
    ``num_sampling_steps + 1`` points and replaces its final entry with zero. The two grids place
    every intermediate noise level differently and are not interchangeable.

    A single step yields the lone entry ``sigma_data * max_noise_level`` followed by zero.

    The first entry is evaluated through the same ``(max_noise_level ** (1 / rho)) ** rho`` ramp as
    every other entry, so for a ``rho`` whose round trip is inexact in the requested dtype it differs
    from ``sigma_data * max_noise_level`` by a rounding step, uniformly at every step count. The
    released Boltz-lineage schedules carry that same rounding step, so it is reproduced rather than
    corrected here. A caller whose own released schedule takes the endpoint directly for some step
    count owns that case before calling this function.

    Args:
        num_sampling_steps: Number of consecutive sampler transitions.
        max_noise_level: Upper interpolation endpoint before ``sigma_data`` scaling.
        min_noise_level: Lower interpolation endpoint before ``sigma_data`` scaling.
        rho: Exponent controlling the concentration of steps at low noise.
        sigma_data: Scale applied to every nonzero schedule entry.
        device: Device of the returned schedule.
        dtype: Floating-point dtype of the returned schedule.

    Returns:
        A tensor with shape ``(num_sampling_steps + 1,)`` whose final entry is zero.

    Raises:
        ValueError: If ``num_sampling_steps`` is not positive.

    Examples:
        >>> schedule = karras_noise_schedule(
        ...     1, max_noise_level=160.0, min_noise_level=4e-4, rho=7.0, sigma_data=16.0,
        ...     dtype=torch.float32,
        ... )
        >>> schedule.tolist()
        [2560.0, 0.0]
    """

    if num_sampling_steps < 1:
        raise ValueError(f"num_sampling_steps must be positive, got {num_sampling_steps}")
    inverse_rho = 1 / rho
    # Clamping the divisor makes a single step yield the lone maximum noise level instead of
    # dividing by zero; for two or more steps the divisor is exactly ``num_sampling_steps - 1``.
    ramp = torch.arange(num_sampling_steps, device=device, dtype=dtype) / max(num_sampling_steps - 1, 1)
    schedule = (
        max_noise_level**inverse_rho + ramp * (min_noise_level**inverse_rho - max_noise_level**inverse_rho)
    ) ** rho
    return torch.nn.functional.pad(schedule * sigma_data, (0, 1), value=0.0)


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
