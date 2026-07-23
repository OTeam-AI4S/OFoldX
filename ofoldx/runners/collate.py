r"""Collate a list of per-structure family Inputs into one batched family Input.

Each dataset item is a family Input dataclass whose tensor fields carry a leading
singleton batch axis (as the Processors emit). The generic [`collate`][]
concatenates matching fields along that axis, padding variable-length axes so a
heterogeneous batch (different sequence / atom counts) becomes one rectangular
Input.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import torch
from torch import Tensor


def collate(batch: list[Any]) -> Any:
    r"""Merge a list of single-structure family Inputs into one batched Input.

    Contract:

    - Every element is the same family Input dataclass with a leading singleton
      batch axis on its tensor fields.
    - Tensor fields are concatenated along dim 0, right-padding the ragged axes
      (sequence length, atom count, ...) to the batch maximum; the pad value is 0
      except for boolean/validity masks, which pad ``False``.
    - Tuple/sequence identity fields (``structures``, ``component_references``)
      are concatenated into a length-``B`` tuple.
    - ``None`` optional fields collate to ``None`` iff every element is ``None``;
      a mixed presence is an error.

    Returns one family Input dataclass whose tensor fields have leading dim ``B``.
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
    r"""Merge one field's per-element values by kind (tensor / sequence / identity).

    Optional fields are resolved first: a field collates to ``None`` only when it
    is ``None`` on every element, and a mixed presence (some ``None``, some a
    tensor) is rejected rather than silently zero-filled.
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
    r"""Right-pad the ragged non-batch axes to the batch maximum, then concatenate.

    Every tensor shares the leading batch axis and the trailing rank; only the
    intermediate extents (sequence length, atom count, ...) may differ. Each
    tensor is copied into a zero-filled tensor sized to the per-axis batch
    maximum, so the pad value is 0 and a boolean mask pads ``False`` (the boolean
    zero). Concatenation along dim 0 then yields the leading batch dimension.
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
    r"""Flatten each element's per-example tuple into one length-``B`` tuple.

    A Processor emits an identity sequence (``structures``, ``component_references``)
    with a leading batch axis of length one, so flattening the singleton tuples
    across the batch reproduces the ``B`` per-example identities in order.
    """

    flattened: list[Any] = []
    for sequence in sequences:
        flattened.extend(sequence)
    return tuple(flattened)


def _collate_identity(name: str, values: list[Any]) -> Any:
    r"""Keep a scalar / configuration field shared by the whole batch.

    A field that is neither a tensor nor a sequence carries no batch axis, so it
    must be identical across the batch; the shared value is asserted with ``==``
    and returned unchanged.
    """

    reference = values[0]
    for value in values[1:]:
        if value != reference:
            raise ValueError(
                f"field {name!r} is a non-tensor identity field and must be equal across the "
                f"batch; got {reference!r} and {value!r}"
            )
    return reference
