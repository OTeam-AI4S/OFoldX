r"""The single seed|generator resolution point for every Pipeline.

Per OWNER principle P1 the model graph is a pure function of ``(tensors, generator)``:
it accepts a :class:`torch.Generator` object only and never resolves an integer seed,
creates a generator, or touches the global default RNG. The seed to generator resolution
(and the ``seed``/``generator`` mutual exclusion) lives exactly once, here, at the pipeline
boundary. This generalizes the reference implementation in ``boltzgen/pipeline.py``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass

import torch


def inputs_device(inputs: object) -> torch.device | None:
    r"""Read the device the Processor placed its output tensors on.

    The generic task Pipelines bind an arbitrary Model/Processor pair, so the processed
    ``inputs`` are a model-specific carrier with no common field. Rather than scanning the
    Model's parameters (which infers placement from the model graph and ``StopIteration``s on
    a param-less model), this reads the device off the data the Processor produced — the
    device the model will sample on — by returning the device of the first tensor found on
    ``inputs``. This generalizes what every concrete Pipeline does explicitly (e.g.
    ``batch.backbone_coordinates.device`` / ``fold_batch.atom_mask.device``).

    Args:
        inputs: A processed-inputs carrier: a tensor, a mapping, a dataclass, or any object
            whose attributes hold the compute tensors.

    Returns:
        The device of the first tensor discovered on ``inputs``, or ``None`` when none is
        found (``resolve_generator`` then binds the seeded generator to the default device).
    """

    if isinstance(inputs, torch.Tensor):
        return inputs.device
    if isinstance(inputs, Mapping):
        values: object = inputs.values()
    elif is_dataclass(inputs) and not isinstance(inputs, type):
        values = [getattr(inputs, field.name) for field in fields(inputs)]
    elif hasattr(inputs, "__dict__"):
        values = vars(inputs).values()
    else:
        return None
    for value in values:  # type: ignore[attr-defined]
        if isinstance(value, torch.Tensor):
            return value.device
    return None


def _device_mismatch(source: torch.device, target: torch.device) -> bool:
    r"""Report whether a caller generator on ``source`` cannot sample inputs on ``target``.

    A ``cuda`` generator with no explicit index is compatible with any ``cuda`` input (and
    vice versa) — only differing device *types*, or two explicit indices that differ, are a
    genuine mismatch that would crash deep in sampling.
    """

    if source.type != target.type:
        return True
    return source.index is not None and target.index is not None and source.index != target.index


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
        device: Device the seeded generator is bound to when ``seed`` is given, and the
            device a caller-supplied ``generator`` is validated against.

    Returns:
        ``generator`` when supplied; a fresh ``Generator(device).manual_seed(seed)`` when only
        ``seed`` is given; ``None`` (ambient RNG) when neither is given.

    Raises:
        ValueError: If both ``seed`` and ``generator`` are supplied, or if a caller-supplied
            ``generator`` lives on a different device than the pipeline inputs (which would
            otherwise crash deep in sampling with an opaque device-mismatch error).
    """

    if seed is not None and generator is not None:
        raise ValueError("seed and generator are mutually exclusive")
    if generator is not None:
        if device is not None and _device_mismatch(generator.device, torch.device(device)):
            raise ValueError(
                f"generator is on device {generator.device} but the pipeline inputs are on "
                f"{torch.device(device)}; create the generator on the inputs device, or pass "
                "seed= instead to let the pipeline bind one to that device"
            )
        return generator
    if seed is not None:
        return torch.Generator(device=device).manual_seed(seed)
    return None
