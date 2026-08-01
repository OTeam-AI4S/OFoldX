r"""Collation for the exact Input mathematics used by built-in objectives."""

from __future__ import annotations

from dataclasses import fields
from typing import Any

import torch

from ..data.molecular import MolecularSystem
from ..data.structure import Structure
from ..modules.diffusion.loss import resolved_atom_mask


def _nested(values: list[torch.Tensor], ragged_dims: tuple[int, ...]) -> Any:
    from danling.tensors import NestedTensor

    return NestedTensor([value.squeeze(0) for value in values], ragged_dims=ragged_dims)


def _offset_nested_indices(
    values: list[torch.Tensor],
    offsets: list[int],
    *,
    ragged_dims: tuple[int, ...] = (0,),
) -> Any:
    return _nested(
        [torch.where(value < 0, value, value + offset) for value, offset in zip(values, offsets, strict=True)],
        ragged_dims,
    )


def collate_mpnn(batch: list[Any]) -> Any:
    r"""Form one topology-invariant ragged ProteinMPNN or NA-MPNN batch."""

    if len(batch) == 1:
        return batch[0]

    from danling.tensors import NestedTensor

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
            merged[field.name] = NestedTensor(component_indices, ragged_dims=(0,))
        else:
            merged[field.name] = NestedTensor([value.squeeze(0) for value in values], ragged_dims=(0,))
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


def collate_alphafold3_inputs(inputs: list[Any]) -> Any:
    r"""Form one explicit ragged AlphaFold 3 model-input batch."""

    if len(inputs) == 1:
        return inputs[0]

    from ..models.alphafold3.model import (
        AlphaFold3Input,
        AlphaFold3TemplateFeatures,
    )

    systems = [input.system for input in inputs]
    atom_offsets: list[int] = []
    component_offsets: list[int] = []
    atom_offset = 0
    component_offset = 0
    for system in systems:
        atom_offsets.append(atom_offset)
        component_offsets.append(component_offset)
        atom_offset += system.atom_symbol.numel()
        component_offset += system.component_symbol.numel()

    def nested_fields(names: tuple[str, ...], ragged_dims: tuple[int, ...]) -> dict[str, Any]:
        return {name: _nested([getattr(input, name) for input in inputs], ragged_dims) for name in names}

    token = nested_fields(
        (
            "token_mask",
            "residue_index",
            "chain_index",
            "entity_index",
            "symmetry_index",
            "token_index",
            "token_to_rep_atom",
            "token_features",
        ),
        (0,),
    )
    atom = nested_fields(
        (
            "atom_features",
            "atom_to_token",
            "atom_mask",
            "reference_positions",
            "reference_space_id",
            "atom_token_index",
            "reference_mask",
            "reference_charges",
        ),
        (0,),
    )
    pair = nested_fields(("token_bonds",), (0, 1))
    msa = nested_fields(("msa_tokens", "msa_deletion_counts", "msa_mask"), (0, 1))

    template_values = [input.template_features for input in inputs]
    if all(value is None for value in template_values):
        template_features = None
    elif any(value is None for value in template_values):
        raise ValueError("AlphaFold 3 batches must group inputs with the same template count")
    else:
        templates = [value for value in template_values if value is not None]
        if len({value.distogram.shape[1] for value in templates}) != 1:
            raise ValueError("AlphaFold 3 batches must group inputs with the same template count")
        template_masks = []
        for template in templates:
            mask = (
                template.template_mask
                if template.template_mask is not None
                else template.distogram.new_ones(template.distogram.shape[:2])
            )
            template_masks.append(mask)
        template_features = AlphaFold3TemplateFeatures(
            distogram=_nested([value.distogram for value in templates], (1, 2)),
            pseudo_beta_mask=_nested([value.pseudo_beta_mask for value in templates], (1, 2)),
            residue_type_ids=_nested([value.residue_type_ids for value in templates], (1,)),
            unit_vector=_nested([value.unit_vector for value in templates], (1, 2)),
            backbone_frame_mask=_nested([value.backbone_frame_mask for value in templates], (1, 2)),
            template_mask=torch.cat(template_masks),
            batch_size=[len(inputs)],
        )

    return AlphaFold3Input(  # type: ignore[call-arg]
        **token,
        **atom,
        **pair,
        **msa,
        template_features=template_features,
        system=MolecularSystem.batch(systems),
        atom_index=_nested([input.atom_index for input in inputs], (0,)),
        token_component_index=_offset_nested_indices(
            [input.token_component_index for input in inputs], component_offsets
        ),
        token_atom_index=_offset_nested_indices([input.token_atom_index for input in inputs], atom_offsets),
        batch_size=[],
    )


def collate_alphafold3_denoising(batch: list[Any]) -> dict[str, Any]:
    r"""Form one explicit ragged AlphaFold 3 training batch."""

    if len(batch) == 1:
        return collate_denoising(batch)

    from ..models.alphafold3.model import AlphaFold3StructureTarget

    inputs, targets = map(list, zip(*batch, strict=True))
    nested_input = collate_alphafold3_inputs(inputs)
    nested_target = AlphaFold3StructureTarget(  # type: ignore[call-arg]
        coordinates=_nested([target.coordinates for target in targets], (0,)),
        coordinate_mask=_nested([target.coordinate_mask for target in targets], (0,)),
        batch_size=[],
    )
    resolved = resolved_atom_mask(
        nested_target.coordinate_mask,
        nested_input.atom_mask,
        nested_input.atom_index,
    )
    count = _resolved_structure_count(resolved)
    pair = {"input": nested_input, "target": nested_target}
    return {"input": pair, "target": resolved, "loss_normalizer": count}


def collate_protenix_inputs(inputs: list[Any]) -> Any:
    r"""Form one explicit ragged Protenix model-input batch."""

    if len(inputs) == 1:
        return inputs[0]

    from ..models.protenix.model import (
        ProtenixInput,
        ProtenixTemplateFeatures,
    )

    systems = [input.system for input in inputs]
    atom_offsets: list[int] = []
    component_offsets: list[int] = []
    atom_offset = 0
    component_offset = 0
    for system in systems:
        atom_offsets.append(atom_offset)
        component_offsets.append(component_offset)
        atom_offset += system.atom_symbol.numel()
        component_offset += system.component_symbol.numel()

    def fields(names: tuple[str, ...], ragged_dims: tuple[int, ...]) -> dict[str, Any]:
        return {name: _nested([getattr(input, name) for input in inputs], ragged_dims) for name in names}

    token = fields(
        (
            "token_mask",
            "residue_index",
            "chain_index",
            "entity_index",
            "symmetry_index",
            "token_index",
            "token_to_rep_atom",
            "token_features",
            "token_to_esm_residue",
            "token_has_esm",
            "structure_rep_positions",
            "structure_rep_mask",
        ),
        (0,),
    )
    atom = fields(
        (
            "atom_features",
            "atom_to_token",
            "atom_mask",
            "reference_positions",
            "reference_space_id",
            "atom_token_index",
            "reference_mask",
            "reference_charges",
        ),
        (0,),
    )
    pair = fields(("token_bonds", "pocket", "token_contact", "atom_contact", "substructure"), (0, 1))
    msa = fields(("msa_tokens", "msa_deletion_counts", "esm_tokens", "esm_mask"), (0, 1))

    template_values = [input.template_features for input in inputs]
    if all(value is None for value in template_values):
        template_features = None
    elif any(value is None for value in template_values):
        raise ValueError("Protenix template features must be present for every batch element or none")
    else:
        templates = [value for value in template_values if value is not None]
        template_features = ProtenixTemplateFeatures(  # type: ignore[call-arg]
            distogram=_nested([value.distogram for value in templates], (1, 2)),
            pseudo_beta_mask=_nested([value.pseudo_beta_mask for value in templates], (1, 2)),
            residue_type_ids=_nested([value.residue_type_ids for value in templates], (1,)),
            unit_vector=_nested([value.unit_vector for value in templates], (1, 2)),
            backbone_frame_mask=_nested([value.backbone_frame_mask for value in templates], (1, 2)),
            batch_size=[],
        )
    if any(input.tfg_features is not None for input in inputs):
        raise ValueError("training-free-guidance geometry is request-local and cannot enter a training batch")

    nested_input = ProtenixInput(  # type: ignore[call-arg]
        **token,
        **atom,
        **pair,
        **msa,
        template_features=template_features,
        tfg_features=None,
        system=MolecularSystem.batch(systems),
        atom_index=_nested([input.atom_index for input in inputs], (0,)),
        token_component_index=_offset_nested_indices(
            [input.token_component_index for input in inputs], component_offsets
        ),
        token_atom_index=_offset_nested_indices([input.token_atom_index for input in inputs], atom_offsets),
        batch_size=[],
    )
    return nested_input


def collate_protenix_denoising(batch: list[Any]) -> dict[str, Any]:
    r"""Form one explicit ragged Protenix training batch."""

    if len(batch) == 1:
        return collate_denoising(batch)

    from ..models.protenix.model import ProtenixStructureTarget

    inputs, targets = map(list, zip(*batch, strict=True))
    nested_input = collate_protenix_inputs(inputs)
    nested_target = ProtenixStructureTarget(  # type: ignore[call-arg]
        coordinates=_nested([target.coordinates for target in targets], (0,)),
        coordinate_mask=_nested([target.coordinate_mask for target in targets], (0,)),
        resolution=torch.cat([target.resolution for target in targets]),
        resolution_mask=torch.cat([target.resolution_mask for target in targets]),
        batch_size=[],
    )
    resolved = resolved_atom_mask(
        nested_target.coordinate_mask,
        nested_input.atom_mask,
        nested_input.atom_index,
    )
    count = _resolved_structure_count(resolved)
    pair = {"input": nested_input, "target": nested_target}
    return {"input": pair, "target": resolved, "loss_normalizer": count}


def _design_count(input: Any) -> int:
    count = int((input.residue_mask & input.design_positions).sum())
    if count == 0:
        raise ValueError("the batch has no designable residue to score")
    return count


def collate_sequence(batch: list[Any]) -> dict[str, Any]:
    input = collate_mpnn(batch)
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


def _resolved_structure_count(resolved: torch.Tensor) -> int:
    resolved_per_structure = resolved.sum(dim=-1)
    if bool((resolved_per_structure == 0).any()):
        raise ValueError("every structure in the batch must have a resolved atom to score")
    return resolved_per_structure.numel()


def collate_denoising(batch: list[Any]) -> dict[str, Any]:
    input, target = collate_single(batch)
    resolved = resolved_atom_mask(
        target.coordinate_mask,
        input.atom_mask,
        input.atom_index,
    )
    count = _resolved_structure_count(resolved)
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
    count = _resolved_structure_count(resolved)
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
