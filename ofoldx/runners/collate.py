r"""
Collation for typed model-input dataclasses.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import torch
from torch import Tensor


def collate(batch: list[Any]) -> Any:
    r"""
    Collate model-input dataclasses along their leading batch axes.

    Tensor fields are right-padded on non-batch axes and concatenated on axis zero.
    Sequence fields are flattened in input order. Optional fields must be present on
    all items or absent on all items.

    Args:
        batch: Non-empty list of dataclass instances of one type.

    Returns:
        A dataclass whose leading tensor-axis length is the sum of input leading-axis
        lengths.

    Raises:
        TypeError: If inputs are not dataclass instances of one type.
        ValueError: If the batch is empty or aligned fields cannot be collated.
    """

    if not batch:
        raise ValueError("collate requires a non-empty batch")

    first = batch[0]
    if not dataclasses.is_dataclass(first) or isinstance(first, type):
        raise TypeError("collate expects family Input dataclass instances")
    input_type = type(first)
    for item in batch:
        if type(item) is not input_type:
            raise TypeError(
                "collate requires every batch element to be the same family Input type; got "
                f"{input_type.__name__} and {type(item).__name__}"
            )

    merged = {
        field.name: _collate_field(field.name, [getattr(item, field.name) for item in batch])
        for field in dataclasses.fields(first)
    }
    return dataclasses.replace(first, **merged)


def _collate_field(name: str, values: list[Any]) -> Any:
    r"""
    Collate one aligned dataclass field.
    """

    present = [value is not None for value in values]
    if not any(present):
        return None
    if not all(present):
        raise ValueError(
            f"field {name!r} is present on some batch elements and None on others; an optional "
            "field must be None on every element or on none of them"
        )

    reference = values[0]
    if isinstance(reference, Tensor):
        return _collate_tensors(name, values)
    if isinstance(reference, (tuple, list)):
        return _collate_sequences(values)
    return _collate_identity(name, values)


def _collate_tensors(name: str, tensors: list[Tensor]) -> Tensor:
    r"""
    Right-pad non-batch axes and concatenate tensors on axis zero.

    All tensors must have the same rank. Their leading-axis lengths may differ.
    """

    reference = tensors[0]
    for tensor in tensors:
        if tensor.ndim != reference.ndim:
            raise ValueError(f"field {name!r} tensors must share their rank to collate")
        if tensor.ndim == 0:
            raise ValueError(f"field {name!r} tensors must carry a leading batch axis")

    non_batch_maxima = [max(tensor.shape[axis] for tensor in tensors) for axis in range(1, reference.ndim)]
    padded: list[Tensor] = []
    for tensor in tensors:
        if list(tensor.shape[1:]) == non_batch_maxima:
            padded.append(tensor)
            continue
        rectangular = torch.zeros(
            (tensor.shape[0], *non_batch_maxima),
            dtype=tensor.dtype,
            device=tensor.device,
        )
        region = (slice(None), *(slice(0, extent) for extent in tensor.shape[1:]))
        rectangular[region] = tensor
        padded.append(rectangular)
    return torch.cat(padded, dim=0)


def _collate_sequences(sequences: list[Any]) -> tuple[Any, ...]:
    r"""
    Flatten aligned sequence fields in input order.
    """

    flattened: list[Any] = []
    for sequence in sequences:
        flattened.extend(sequence)
    return tuple(flattened)


def _collate_identity(name: str, values: list[Any]) -> Any:
    r"""
    Return a non-batched field shared by every input.
    """

    reference = values[0]
    for value in values[1:]:
        if value != reference:
            raise ValueError(
                f"field {name!r} is a non-tensor identity field and must be equal across the "
                f"batch; got {reference!r} and {value!r}"
            )
    return reference
