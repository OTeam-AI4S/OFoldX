r"""
Shared batching helpers for the training Inputs.

``pad_and_collate``/``collate_single`` batch the MPNN-family sequence Inputs; ``collate_denoising`` batches
the diffusion families' ``(input, target)`` pairs into DanLing's training batch. The first is pure padding
logic parameterised by the family's pad token and its multi-residue-axis fields;
each family wraps it into its own ``training_collate``. Most tensors are residue-aligned on axis 1;
fields listed in ``pair_bias_fields`` (e.g. NA-MPNN ``pair_logit_bias`` of shape ``(B, L, V, L, V)``) are
residue-aligned on axes 1 and 3. ``tied_group_ids`` indexes an autoregressive-unit tensor via
``scatter_reduce_``, so padded rows become their own collated index rather than merging into real unit 0.
Non-tensor per-item carriers are concatenated in order (they feed output decoding, not the forward).
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import fields, is_dataclass, replace
from typing import Any

import torch
from torch import Tensor


def _pad_axis(tensor: Tensor, axis: int, length: int, value: float | bool) -> Tensor:
    current = tensor.shape[axis]
    if current == length:
        return tensor
    shape = list(tensor.shape)
    shape[axis] = length - current
    return torch.cat([tensor, tensor.new_full(shape, value)], dim=axis)


def _pad_tied_group_ids(tensor: Tensor, length: int) -> Tensor:
    current = tensor.shape[1]
    if current == length:
        return tensor
    extra = torch.arange(current, length, device=tensor.device, dtype=tensor.dtype).expand(tensor.shape[0], -1)
    return torch.cat([tensor, extra], dim=1)


def pad_and_collate(batch: list[Any], *, token_pad_id: int, pair_bias_fields: Collection[str] = ()) -> Any:
    r"""
    Collate a batch of one MPNN-family Input dataclass to the longest sequence.

    Args:
        batch: Non-empty list of the same Input dataclass type.
        token_pad_id: Pad value for ``sequence_tokens``.
        pair_bias_fields: Field names that are residue-aligned on axes 1 and 3.
    """

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

    pair_fields = frozenset(pair_bias_fields)
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
            if field.name == "sequence_tokens":
                pad: float | bool = token_pad_id
            elif field.name == "allowed_token_mask":
                pad = True
            else:
                pad = 0
            axes = (1, 3) if field.name in pair_fields else (1,)
            padded = []
            for value in values:
                for axis in axes:
                    value = _pad_axis(value, axis, length, pad)
                padded.append(value)
            merged[field.name] = torch.cat(padded, dim=0)
        else:
            merged[field.name] = tuple(element for value in values for element in value)
    return replace(first, **merged)


def collate_single(batch: list[Any]) -> Any:
    r"""
    Batch-size-1 collate for large structure models: return the sole already-batched Input.

    Multi-structure batching for these families comes from gradient accumulation weighted by the
    token-count normalizer, not from padding heterogeneous atom/token tensors here.
    """

    if len(batch) != 1:
        raise ValueError(f"this family trains at batch_size=1; got a batch of {len(batch)}")
    return batch[0]


def collate_denoising(batch: list[Any]) -> dict[str, Any]:
    r"""
    Batch-size-1 collate for the diffusion structure families to DanLing's training batch.

    Each item is a ``(model_input, model_target)`` pair from the Processor's ``process_training``. The
    diffusion forward needs both (it noises the target internally), so ``input`` is the mapping
    ``{"input", "target"}`` that DanLing splats as ``model(input=..., target=...)``; the same mapping is
    the criterion's ``target`` (its ``input`` carries ``atom_mask``, its ``target`` the truth a fuller
    per-family loss may need). Multi-structure training comes from gradient accumulation weighted by the
    atom-count normalizer, not from padding heterogeneous atom tensors here.
    """

    if len(batch) != 1:
        raise ValueError(f"the diffusion families train at batch_size=1; got a batch of {len(batch)}")
    model_input, model_target = batch[0]
    pair = {"input": model_input, "target": model_target}
    return {"input": pair, "target": pair, "loss_normalizer": int(model_input.atom_mask.sum())}
