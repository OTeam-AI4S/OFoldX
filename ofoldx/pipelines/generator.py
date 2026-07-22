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
    def device(self) -> torch.device | str | None: ...

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> object: ...


def processor_device(processor: PipelineProcessor) -> torch.device | None:
    r"""Return the output device declared by one concrete Processor.

    Processors own output placement and expose it before processing, which lets a
    Pipeline construct the correct device-specific generator.

    Args:
        processor: A concrete Processor following the placement contract.

    Returns:
        The Processor's output device, or ``None`` for the default device.
    """

    device = processor.device
    return None if device is None else torch.device(device)


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
