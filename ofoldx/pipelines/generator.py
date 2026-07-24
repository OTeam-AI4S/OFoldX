r"""
Random-generator handling shared by task pipelines.
"""

from __future__ import annotations

from typing import Protocol

import torch


class PipelineProcessor(Protocol):
    r"""
    Runtime placement required from processors used by task pipelines.
    """

    @property
    def device(self) -> torch.device:
        r"""
        Device used for tensors produced by this processor.

        Returns:
            The processor's current tensor device.
        """

        ...

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> object:
        r"""
        Update the processor's tensor device and dtype.

        Args:
            device: New tensor device. ``None`` preserves the current device.
            dtype: New floating-point dtype. ``None`` preserves the current dtype.

        Returns:
            The updated processor.
        """

        ...


def resolve_generator(
    seed: int | None,
    generator: torch.Generator | None,
    *,
    device: torch.device | str | None = None,
) -> torch.Generator | None:
    r"""
    Resolve mutually exclusive seed and generator arguments.

    Args:
        seed: Optional integer seed; mutually exclusive with ``generator``.
        generator: Optional caller-supplied RNG; mutually exclusive with ``seed``.
        device: Device the seeded generator is bound to when ``seed`` is given.

    Returns:
        ``generator`` when supplied; a fresh ``Generator(device).manual_seed(seed)`` when only
        ``seed`` is given; ``None`` (ambient RNG) when neither is given.

    Raises:
        ValueError: If both ``seed`` and ``generator`` are supplied.
    """

    if seed is not None and generator is not None:
        raise ValueError("seed and generator are mutually exclusive")
    if generator is not None:
        return generator
    if seed is not None:
        return torch.Generator(device=device).manual_seed(seed)
    return None
