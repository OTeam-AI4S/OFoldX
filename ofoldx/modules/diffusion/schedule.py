r"""
EDM noise schedules.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class EdmScheduler(nn.Module):
    r"""
    EDM noise schedule that produces the descending ``sigma`` levels a coordinate sampler walks.

    Builds the Karras et al. ``rho``-warped schedule: noise levels are spaced linearly in ``sigma ** (1 / rho)``
    between ``max_noise_level`` and ``min_noise_level`` (scaled by ``sigma_data``), then the final level is pinned to
    zero so the last denoising step lands on clean coords.

    Args:
        max_noise_level: Largest noise level, used for the first (fully noised) step.
        min_noise_level: Smallest non-zero noise level before the schedule is pinned to zero.
        rho: Warping exponent controlling how steps concentrate at low noise; larger ``rho`` packs more steps there.
        sigma_data: Data noise scale the warped schedule is multiplied by.

    Examples:
        >>> import torch
        >>> scheduler = EdmScheduler()
        >>> schedule = scheduler(num_sampling_steps=4, dtype=torch.float32)
        >>> schedule.shape
        torch.Size([5])
        >>> float(schedule[-1])
        0.0
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
    ) -> Tensor:
        r"""
        Return ``num_sampling_steps + 1`` descending EDM noise levels, ending at zero.

        The extra entry over ``num_sampling_steps`` is the terminal zero level; each consecutive
        ``(sigma_prev, sigma_next)`` pair defines one sampler step.

        Args:
            num_sampling_steps: Number of denoising steps; the schedule has one more entry than this.
            device: Device to place the schedule on.
            dtype: Floating dtype of the returned schedule.

        Returns:
            Descending noise levels of shape `[num_sampling_steps + 1]`, with the last entry equal to ``0``.

        Raises:
            ValueError: If ``num_sampling_steps`` is not positive.

        Examples:
            >>> import torch
            >>> scheduler = EdmScheduler()
            >>> [round(float(x), 2) for x in scheduler(num_sampling_steps=4, dtype=torch.float32)]
            [2560.0, 489.86, 55.97, 2.38, 0.0]
        """

        if num_sampling_steps < 1:
            raise ValueError(f"num_sampling_steps must be positive, got {num_sampling_steps}")
        step_size = 1 / num_sampling_steps
        step_indices = torch.arange(num_sampling_steps + 1, device=device, dtype=dtype)
        max_root = self.max_noise_level ** (1 / self.rho)
        min_root = self.min_noise_level ** (1 / self.rho)
        schedule_base = max_root + step_indices * step_size * (min_root - max_root)
        schedule = self.sigma_data * schedule_base**self.rho
        schedule[..., -1] = 0
        return schedule
