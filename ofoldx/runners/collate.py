r"""
Batching of family training Inputs into DanLing's ``{"input", "target", "loss_normalizer"}`` batch.

This is Runner plumbing, not model math: it turns a list of per-structure processed items into one batch
and reports the ``loss_normalizer`` -- the exact count each family's criterion averages over, so DanLing's
accumulation/DDP reduction (which scales the backward by ``normalizer / Σnormalizer``) forms a true global
per-unit mean. The sequence families average over scored residues; the diffusion families average over
scored atoms. Large structure models remain batch_size=1; multi-structure training comes from gradient
accumulation weighted by the normalizer.
"""

from __future__ import annotations

from dataclasses import fields
from typing import Any

import torch
from torch import Tensor

from ..data.structure import Structure


def nested_collate(batch: list[Any]) -> Any:
    r"""
    Collate a batch of one MPNN-family Input dataclass into a ragged ``NestedTensor`` batch (no padding).

    Every residue-aligned tensor field becomes a ``danling.NestedTensor`` carrying each structure at its own
    length; the model runs on the packed values, so a prediction is independent of what shared the batch
    (topology-invariant) with no padded rows to contaminate the k-NN or attention. ``NestedTensor`` handles
    multi-axis raggedness natively, so pair fields (ragged on two residue axes) need no special handling and
    ``tied_group_ids`` need no padded-unit extension -- each structure's causal units stay local to it.
    A tensor-native ``structure`` is one disjoint union, and its aligned nonnegative
    ``component_index`` values are offset into that union.
    """

    from danling import NestedTensor

    first = batch[0]
    input_type = type(first)
    input_fields = fields(first)
    graph_input = {"structure", "component_index"} <= {field.name for field in input_fields}

    merged: dict[str, Any] = {}
    for field in input_fields:
        values = [getattr(item, field.name) for item in batch]
        present = [value is not None for value in values]
        if not any(present):
            merged[field.name] = None
            continue
        if not all(present):
            raise ValueError(f"optional field {field.name!r} must be present on every item or on none")
        if graph_input and field.name == "structure":
            merged[field.name] = Structure.batch(values)
        elif graph_input and field.name == "component_index":
            component_offset = 0
            component_indices = []
            for value, item in zip(values, batch, strict=True):
                value = value.squeeze(0)
                component_indices.append(torch.where(value < 0, value, value + component_offset))
                component_offset += item.structure.system.component_symbol.numel()
            merged[field.name] = NestedTensor(component_indices)
        elif isinstance(values[0], Tensor):
            # Each item is ``(1, N_i, ...)`` from the Processor; drop the singleton batch axis so the
            # NestedTensor's leading axis concatenates the structures, ragged on the residue axis.
            merged[field.name] = NestedTensor([value.squeeze(0) for value in values])
        else:
            # A list (not a tuple) becomes a batched NonTensorStack so ``input[i]`` selects item ``i``'s
            # metadata; a tuple would be one shared blob that does not track the batch axis.
            merged[field.name] = [element for value in values for element in value]
    return input_type(
        **merged,
        batch_size=[] if graph_input else [len(batch)],
    )


def collate_single(batch: list[Any]) -> Any:
    r"""
    Batch-size-1 collate for large structure models: return the sole already-batched Input.
    """

    if len(batch) != 1:
        raise ValueError(f"this family trains at batch_size=1; got a batch of {len(batch)}")
    return batch[0]


def _design_count(batched: Any) -> int:
    r"""Count designable residues -- the unit the sequence-recovery criteria average over."""

    count = int((batched.residue_mask & batched.design_positions).sum())
    if count == 0:
        raise ValueError("the batch has no designable residue to score")
    return count


def collate_sequence(batch: list[Any]) -> dict[str, Any]:
    r"""
    Collate a sequence family to a ragged ``NestedTensor`` batch, normalized by designable residues. Each
    structure runs at its own length, so BOTH the training loss and inference are topology-invariant
    (independent of what shared the batch, no padded rows to contaminate the k-NN/attention); inference
    un-nests per structure (``prediction[i]``) with no tail-padding to trim.
    """

    batched = nested_collate(batch)
    return {"input": batched, "target": batched, "loss_normalizer": _design_count(batched)}


def collate_sequence_single(batch: list[Any]) -> dict[str, Any]:
    r"""
    Collate a batch_size=1 sequence family (e.g. LigandMPNN's single-structure Input) to DanLing's batch. A
    single structure has no batch to be topology-invariant to and no padding, so it stays dense.
    """

    batched = collate_single(batch)
    return {"input": batched, "target": batched, "loss_normalizer": _design_count(batched)}


def collate_alphafold2(batch: list[Any]) -> dict[str, Any]:
    r"""Batch one AlphaFold 2 example; its masked-MSA loss is normalized per example."""

    batched = collate_single(batch)
    return {"input": batched, "target": batched, "loss_normalizer": 1}


def collate_grnade(batch: list[Any]) -> dict[str, Any]:
    r"""Batch one gRNAde graph, normalized by designable canonical-RNA targets."""

    from ..models.grnade.model import GRNADE_UNKNOWN_TOKEN

    batched = collate_single(batch)
    known_design = batched.design_mask & (batched.sequence_tokens != GRNADE_UNKNOWN_TOKEN)
    return {"input": batched, "target": batched, "loss_normalizer": int(known_design.sum())}


def collate_oinvfold(batch: list[Any]) -> dict[str, Any]:
    r"""Batch one OInvFold polymer graph, normalized by designable nodes."""

    batched = collate_single(batch)
    return {"input": batched, "target": batched, "loss_normalizer": int(batched.design_mask.sum())}


def collate_denoising(batch: list[Any]) -> dict[str, Any]:
    r"""
    Batch-size-1 collate for the diffusion families to DanLing's batch, normalized by resolved atoms.

    A training item is a ``(model_input, model_target)`` pair from the Processor's ``process_training``. The
    diffusion forward needs both, so ``input`` is the ``{"input", "target"}`` mapping DanLing splats as
    ``model(input=..., target=...)``; the same mapping is the criterion's ``target`` (carrying both the
    ``atom_mask`` and the resolved-atom Target). The normalizer is the resolved-atom count the denoising
    loss averages over.
    """

    from ..modules.diffusion.loss import resolved_atom_mask

    model_input, model_target = collate_single(batch)
    resolved = resolved_atom_mask(model_target, model_input)
    count = int(resolved.sum())
    if count == 0:
        raise ValueError("the batch has no resolved atom to score")
    pair = {"input": model_input, "target": model_target}
    return {"input": pair, "target": pair, "loss_normalizer": count}


def collate_odesign(batch: list[Any]) -> dict[str, Any]:
    r"""Batch one ODesign target, normalized by resolved non-conditioning atoms."""

    from ..modules.diffusion.loss import resolved_atom_mask

    model_input, model_target = collate_single(batch)
    pair = {"input": model_input, "target": model_target}
    scored = resolved_atom_mask(model_target, model_input) & ~model_input.fixed_coordinate_mask.bool()
    return {"input": pair, "target": pair, "loss_normalizer": int(scored.sum())}
