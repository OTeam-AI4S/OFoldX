r"""
Collators for the MPNN inverse-folding family Inputs.

These know each field's meaning and pad it correctly:

* Most tensors are residue-aligned on axis 1 only and pad to the batch's longest sequence; padded rows
  are gated out downstream by ``residue_mask``.
* ``pair_logit_bias`` is residue-aligned on **two** axes -- ``(B, L, V, L, V)`` -- so it is padded on
  axis 1 *and* axis 3. Fields with more than one residue axis must be listed in ``_PAIR_BIAS_FIELDS``;
  that explicit declaration is what keeps this a true per-field collator instead of a "pad axis 1"
  default that would silently mis-shape (and fail to concatenate) ragged pair biases.
* ``tied_group_ids`` indexes an autoregressive-unit tensor via ``scatter_reduce_``, so a padded row
  must become its **own** throwaway unit (its collated row index) rather than merging into real unit 0.
* Non-tensor per-item carriers (``structures``, ``component_references``, ``polymer_types``,
  ``conditions``) are only used for output decoding, not the forward, so they are concatenated in order.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass, replace
from typing import Any

import torch
from torch import Tensor

from ..models.nampnn.model import NA_MPNN_PADDING_TOKEN_ID

#: Fields whose residue axis appears more than once (both endpoints of a token pair).
_PAIR_BIAS_FIELDS = frozenset({"pair_logit_bias"})


def _pad_axis(tensor: Tensor, axis: int, length: int, value: float | bool) -> Tensor:
    current = tensor.shape[axis]
    if current == length:
        return tensor
    shape = list(tensor.shape)
    shape[axis] = length - current
    return torch.cat([tensor, tensor.new_full(shape, value)], dim=axis)


def _pad_tied_group_ids(tensor: Tensor, length: int) -> Tensor:
    # Padded rows become singleton throwaway units: their own collated row index (>= real unit ids,
    # which are < the item's own length, and < the padded length so they stay valid scatter indices).
    current = tensor.shape[1]
    if current == length:
        return tensor
    extra = torch.arange(current, length, device=tensor.device, dtype=tensor.dtype).expand(tensor.shape[0], -1)
    return torch.cat([tensor, extra], dim=1)


def _residue_axes(field_name: str) -> tuple[int, ...]:
    return (1, 3) if field_name in _PAIR_BIAS_FIELDS else (1,)


def _pad_value(field_name: str, token_pad_id: int) -> float | bool:
    if field_name == "sequence_tokens":
        return token_pad_id
    if field_name == "allowed_token_mask":
        return True  # padded rows are gated out; keep unrestricted rather than forbid every token
    return 0


def _collate_mpnn(batch: list[Any], *, token_pad_id: int) -> Any:
    if not batch:
        raise ValueError("collate requires a non-empty batch")
    first = batch[0]
    if not is_dataclass(first) or isinstance(first, type):
        raise TypeError("MPNN collate expects family Input dataclass instances")
    input_type = type(first)
    for item in batch:
        if type(item) is not input_type:
            raise TypeError(
                f"collate requires one Input type per batch; got {input_type.__name__} and {type(item).__name__}"
            )

    length = max(item.sequence_tokens.shape[1] for item in batch)
    merged: dict[str, Any] = {}
    for field in fields(first):
        values = [getattr(item, field.name) for item in batch]
        present = [value is not None for value in values]
        if not any(present):
            merged[field.name] = None
            continue
        if not all(present):
            raise ValueError(f"optional field {field.name!r} must be present on every item or on none")
        if isinstance(values[0], Tensor):
            if field.name == "tied_group_ids":
                merged[field.name] = torch.cat([_pad_tied_group_ids(value, length) for value in values], dim=0)
                continue
            pad = _pad_value(field.name, token_pad_id)
            axes = _residue_axes(field.name)
            padded = []
            for value in values:
                for axis in axes:
                    value = _pad_axis(value, axis, length, pad)
                padded.append(value)
            merged[field.name] = torch.cat(padded, dim=0)
        else:
            merged[field.name] = tuple(element for value in values for element in value)
    return replace(first, **merged)


def collate_proteinmpnn(batch: list[Any]) -> Any:
    r"""Collate a batch of ``ProteinMpnnInput`` (sequence pad id ``0``, an ordinary residue token)."""

    return _collate_mpnn(batch, token_pad_id=0)


def collate_nampnn(batch: list[Any]) -> Any:
    r"""Collate a batch of ``NaMpnnInput`` (sequence pad id ``NA_MPNN_PADDING_TOKEN_ID``)."""

    return _collate_mpnn(batch, token_pad_id=NA_MPNN_PADDING_TOKEN_ID)
