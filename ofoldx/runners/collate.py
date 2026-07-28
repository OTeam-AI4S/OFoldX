r"""
Batching of family training Inputs into DanLing's ``{"input", "target", "loss_normalizer"}`` batch.

This is Runner plumbing, not model math: it turns a list of per-structure processed items into one batch
and reports the ``loss_normalizer`` -- the exact count each family's criterion averages over, so DanLing's
accumulation/DDP reduction (which scales the backward by ``normalizer / Σnormalizer``) forms a true global
per-unit mean. The sequence families average over designable residues; the diffusion families average over
resolved atoms. ``pad_and_collate`` is the multi-residue-axis padding for batch_size>1 sequence families;
``collate_single`` and ``collate_denoising`` are batch_size=1 (large single-structure models -- multi
structure training comes from gradient accumulation weighted by the normalizer).
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import fields, is_dataclass
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
            # A list (not a tuple) becomes a batched NonTensorStack, so batch indexing ``input[i]`` selects
            # the i-th item's metadata; a tuple would be one shared blob that does not track the batch axis.
            merged[field.name] = [element for value in values for element in value]
    # Rebuild with an explicit batch_size: the tensors are now ``(len(batch), ...)``. ``dataclasses.replace``
    # would reset a tensorclass batch_size to ``[]``, leaving the batch axis untracked.
    return type(first)(**merged, batch_size=[len(batch)])


def collate_single(batch: list[Any]) -> Any:
    r"""
    Batch-size-1 collate for large structure models: return the sole already-batched Input.
    """

    if len(batch) != 1:
        raise ValueError(f"this family trains at batch_size=1; got a batch of {len(batch)}")
    return batch[0]


def _design_count(batched: Any) -> int:
    r"""Count designable residues -- the unit the sequence-recovery criteria average over."""

    return int((batched.residue_mask & batched.design_positions).sum())


def collate_sequence(batch: list[Any], *, token_pad_id: int, pair_bias_fields: Collection[str] = ()) -> dict[str, Any]:
    r"""
    Collate a batch_size>1 sequence family to DanLing's batch, normalized by designable residues.
    """

    batched = pad_and_collate(batch, token_pad_id=token_pad_id, pair_bias_fields=pair_bias_fields)
    return {"input": batched, "target": batched, "loss_normalizer": _design_count(batched)}


def collate_sequence_single(batch: list[Any]) -> dict[str, Any]:
    r"""
    Collate a batch_size=1 sequence family (e.g. LigandMPNN's single-structure Input) to DanLing's batch.
    """

    batched = collate_single(batch)
    return {"input": batched, "target": batched, "loss_normalizer": _design_count(batched)}


def collate_denoising(batch: list[Any]) -> dict[str, Any]:
    r"""
    Batch-size-1 collate for the diffusion families to DanLing's batch, normalized by resolved atoms.

    A training item is a ``(model_input, model_target)`` pair from the Processor's ``process_training``. The
    diffusion forward needs both, so ``input`` is the ``{"input", "target"}`` mapping DanLing splats as
    ``model(input=..., target=...)``; the same mapping is the criterion's ``target`` (carrying both the
    ``atom_mask`` and the resolved-atom Target). The normalizer is the resolved-atom count the denoising
    loss averages over.

    An inference item is a bare model input from the plain Processor call (the ``infer`` split lowers that
    way, so inference-only kwargs like ``tfg_guidance`` / ``placement`` flow through). There is no target;
    ``infer_step`` samples from the input alone, so the batch is just ``{"input": model_input}``.
    """

    from ..modules.diffusion.loss import resolved_atom_mask

    if len(batch) != 1:
        raise ValueError(f"the diffusion families train at batch_size=1; got a batch of {len(batch)}")
    item = batch[0]
    if not isinstance(item, tuple):
        return {"input": item}
    model_input, model_target = item
    pair = {"input": model_input, "target": model_target}
    resolved = resolved_atom_mask(model_target, model_input)
    return {"input": pair, "target": pair, "loss_normalizer": int(resolved.sum())}
