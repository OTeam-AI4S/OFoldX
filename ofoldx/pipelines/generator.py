r"""The single seed|generator resolution point for every Pipeline.

The public seed is resolved before processing on the Processor's declared output
device. Generic task Pipelines have deterministic Processor contracts and pass the
resulting stream only to the Model operation. A concrete Pipeline whose
preprocessing is stochastic explicitly passes that same stream to its Processor.
"""

from __future__ import annotations

from typing import Protocol

import torch


class PipelineProcessor(Protocol):
    r"""Placement capabilities shared by concrete Pipeline Processors."""

    @property
    def device(self) -> torch.device:
        r"""Return the device used for lowered tensors."""

        ...

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> object:
        r"""Move the Processor to a device and floating-point dtype."""

        ...


def resolve_generator(
    seed: int | None,
    generator: torch.Generator | None,
    *,
    device: torch.device | str | None = None,
) -> torch.Generator | None:
    r"""Resolve ``seed``/``generator`` to a single generator object once, at the boundary.

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
