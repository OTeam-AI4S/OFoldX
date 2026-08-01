r"""
Random-generator handling shared by task pipelines.
"""

from __future__ import annotations

import torch


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
