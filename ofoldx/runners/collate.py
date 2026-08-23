r"""Collation for the exact Input mathematics used by built-in objectives."""

from __future__ import annotations

from dataclasses import fields
from typing import Any

import torch

from ..data.structure import Structure
from ..modules.diffusion.loss import resolved_atom_mask


def collate_mpnn(batch: list[Any]) -> Any:
    r"""Form one topology-invariant ragged ProteinMPNN or NA-MPNN batch."""

    from danling import NestedTensor

    first = batch[0]
    merged: dict[str, Any] = {}
    for field in fields(first):
        values = [getattr(item, field.name) for item in batch]
        present = [value is not None for value in values]
        if not any(present):
            merged[field.name] = None
        elif not all(present):
            raise ValueError(f"optional field {field.name!r} must be present on every item or none")
        elif field.name == "structure":
            merged[field.name] = Structure.batch(values)
        elif field.name == "component_index":
            component_offset = 0
            component_indices = []
            for value, item in zip(values, batch, strict=True):
                value = value.squeeze(0)
                component_indices.append(torch.where(value < 0, value, value + component_offset))
                component_offset += item.structure.system.component_symbol.numel()
            merged[field.name] = NestedTensor(component_indices)
        else:
            merged[field.name] = NestedTensor([value.squeeze(0) for value in values])
    return type(first)(**merged, batch_size=[])


def collate_single(batch: list[Any]) -> Any:
    r"""
    Return the one already formed family Input, refusing to drop the others.

    The Input a family Processor emits carries that family's own padded layout, which
    no general rule concatenates, so its objective trains one example per step. See
    [`ofoldx.runners.Runner`][] for the resulting ``dataloader.batch_size`` contract.
    """

    if len(batch) != 1:
        raise ValueError(f"this architecture requires batch_size=1; got {len(batch)}")
    return batch[0]


def _design_count(input: Any) -> int:
    count = int((input.residue_mask & input.design_mask).sum())
    if count == 0:
        raise ValueError("the batch has no designable residue to score")
    return count


def collate_sequence(batch: list[Any]) -> dict[str, Any]:
    input = collate_mpnn(batch)
    return {"input": input, "target": input, "loss_normalizer": _design_count(input)}


def collate_sequence_single(batch: list[Any]) -> dict[str, Any]:
    input = collate_single(batch)
    return {"input": input, "target": input, "loss_normalizer": _design_count(input)}


def collate_grnade(batch: list[Any]) -> dict[str, Any]:
    from ..models.grnade.model import GRNADE_UNKNOWN_TOKEN

    input = collate_single(batch)
    known_design = input.design_mask & (input.sequence_tokens != GRNADE_UNKNOWN_TOKEN)
    count = int(known_design.sum())
    if count == 0:
        raise ValueError("the batch has no canonical designed RNA residue to score")
    return {"input": input, "target": input, "loss_normalizer": count}


def collate_oinvfold(batch: list[Any]) -> dict[str, Any]:
    input = collate_single(batch)
    count = int(input.design_mask.sum())
    if count == 0:
        raise ValueError("the batch has no designed polymer node to score")
    return {"input": input, "target": input, "loss_normalizer": count}


def collate_alphafold2(batch: list[Any]) -> dict[str, Any]:
    return {"input": collate_single(batch), "target": None, "loss_normalizer": 1}


def collate_denoising(batch: list[Any]) -> dict[str, Any]:
    input, target = collate_single(batch)
    resolved = resolved_atom_mask(
        target.coordinate_mask,
        input.atom_mask,
        input.atom_index,
    )
    count = int(resolved.sum())
    if count == 0:
        raise ValueError("the batch has no resolved atom to score")
    return {
        "input": {"input": input, "target": target},
        "target": resolved,
        "loss_normalizer": count,
    }


def collate_boltzgen_denoising(batch: list[Any]) -> dict[str, Any]:
    input, target = collate_single(batch)
    resolved = resolved_atom_mask(
        target.coordinate_mask,
        input.atom_mask,
        target.coordinate_atom_index,
    )
    count = int(resolved.sum())
    if count == 0:
        raise ValueError("the batch has no resolved atom to score")
    return {
        "input": {"input": input, "target": target},
        "target": resolved,
        "loss_normalizer": count,
    }


def collate_boltzgen_ifold(batch: list[Any]) -> dict[str, Any]:
    input, residue_type = collate_single(batch)
    design_mask = input.token_mask.bool() & input.token_resolved_mask.bool() & input.design_mask.bool()
    count = int(design_mask.sum())
    if count == 0:
        raise ValueError("the batch has no designed BoltzGen token to score")
    return {
        "input": {"input": input, "target_residue_type": residue_type},
        "target": {"residue_type": residue_type, "design_mask": design_mask},
        "loss_normalizer": count,
    }


def collate_odesign(batch: list[Any]) -> dict[str, Any]:
    input, target = collate_single(batch)
    scored = target.coordinate_mask.bool() & input.atom_mask.bool() & ~input.fixed_coordinate_mask.bool()
    if not bool(scored.any()):
        raise ValueError("the batch has no resolved generated atom to score")
    pair = {"input": input, "target": target}
    return {"input": pair, "target": pair, "loss_normalizer": 1}
