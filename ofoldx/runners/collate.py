# OFoldX
# Copyright (C) 2026-Present  Valhalla Technology

# This file is part of OFoldX.

# OFoldX is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# OFoldX is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# For additional terms and clarifications, please refer to our License FAQ at:
# <https://ofoldx.valtech.ai/about/license-faq/>.

r"""Collation for the exact Input mathematics used by built-in objectives."""

from __future__ import annotations

from dataclasses import fields
from typing import Any

import torch
from danling.tensors import NestedTensor

from ..models.alphafold3.model import AlphaFold3Input, AlphaFold3StructureTarget, AlphaFold3TemplateFeatures
from ..models.grnade.model import GRNADE_CLASSES
from ..models.protenix.model import ProtenixInput, ProtenixStructureTarget, ProtenixTemplateFeatures
from ..models.targets import SequenceTarget
from ..modules.diffusion.loss import resolved_atom_mask


def _nested(values: list[torch.Tensor], ragged_dims: tuple[int, ...]) -> Any:
    return NestedTensor([value.squeeze(0) for value in values], ragged_dims=ragged_dims)


def collate_mpnn(batch: list[Any]) -> Any:
    r"""
    Form one topology-invariant ragged ProteinMPNN or NA-MPNN batch.

    Every field is nested along its residue axis; index fields (``component_index``) keep indexing their own
    example's target, since a batch carries no joined structure.
    """

    if len(batch) == 1:
        return batch[0]

    first = batch[0]
    merged: dict[str, Any] = {}
    for field in fields(first):
        values = [getattr(item, field.name) for item in batch]
        present = [value is not None for value in values]
        if not any(present):
            merged[field.name] = None
        elif not all(present):
            raise ValueError(f"optional field {field.name!r} must be present on every item or none")
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

    def nested_fields(names: tuple[str, ...], ragged_dims: tuple[int, ...]) -> dict[str, Any]:
        return {name: _nested([getattr(input, name) for input in inputs], ragged_dims) for name in names}

    token = nested_fields(
        (
            "token_mask",
            "frame_mask",
            "residue_index",
            "chain_index",
            "entity_index",
            "symmetry_index",
            "token_index",
            "token_representative_atom_index",
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
        atom_index=_nested([input.atom_index for input in inputs], (0,)),
        batch_size=[],
    )


def collate_alphafold3_denoising(batch: list[Any]) -> dict[str, Any]:
    r"""Form one explicit ragged AlphaFold 3 training batch."""

    if len(batch) == 1:
        return collate_denoising(batch)

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
    num_structures = _resolved_structure_count(resolved)
    pair = {"input": nested_input, "target": nested_target}
    return {"input": pair, "target": resolved, "loss_normalizer": num_structures}


def collate_protenix_inputs(inputs: list[Any]) -> Any:
    r"""Form one explicit ragged Protenix model-input batch."""

    if len(inputs) == 1:
        return inputs[0]

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
            "token_representative_atom_index",
            "token_features",
            "token_to_esm_residue",
            "token_has_esm",
            "structure_rep_positions",
            "structure_rep_mask",
            "frame_atom_index",
            "frame_mask",
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
            "atom_is_polymer",
        ),
        (0,),
    )
    pair = nested_fields(("token_bonds", "pocket", "token_contact", "atom_contact", "substructure"), (0, 1))
    msa = nested_fields(("msa_tokens", "msa_deletion_counts", "esm_tokens", "esm_mask"), (0, 1))

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
        atom_index=_nested([input.atom_index for input in inputs], (0,)),
        batch_size=[],
    )
    return nested_input


def collate_protenix_denoising(batch: list[Any]) -> dict[str, Any]:
    r"""Form one explicit ragged Protenix training batch."""

    if len(batch) == 1:
        return collate_denoising(batch)

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
    num_structures = _resolved_structure_count(resolved)
    pair = {"input": nested_input, "target": nested_target}
    return {"input": pair, "target": resolved, "loss_normalizer": num_structures}


def _sequence_targets(targets: list[SequenceTarget]) -> SequenceTarget:
    if len(targets) == 1:
        return targets[0]
    return SequenceTarget(
        tokens=NestedTensor([value.tokens.squeeze(0) for value in targets], ragged_dims=(0,)),
        mask=NestedTensor([value.mask.squeeze(0) for value in targets], ragged_dims=(0,)),
    )


def _design_count(features: Any, target: SequenceTarget) -> int:
    mask = features.residue_mask.bool() & features.design_mask.bool() & target.mask
    count = int(mask.sum())
    if not count:
        raise ValueError("the batch has no known designable residue to score")
    return count


def collate_sequence(batch: list[Any]) -> dict[str, Any]:
    inputs, targets = map(list, zip(*batch, strict=True))
    features, target = collate_mpnn(inputs), _sequence_targets(targets)
    return {
        "input": features,
        "target": {"input": features, "target": target},
        "loss_normalizer": _design_count(features, target),
    }


def collate_grnade(batch: list[Any]) -> dict[str, Any]:
    features, target = collate_single(batch)
    classes = torch.tensor(GRNADE_CLASSES, device=target.tokens.device)
    known_design = features.design_mask.bool() & target.mask & torch.isin(target.tokens, classes)
    count = int(known_design.sum())
    if not count:
        raise ValueError("the batch has no canonical designed RNA residue to score")
    return {"input": features, "target": {"input": features, "target": target}, "loss_normalizer": count}


def collate_oinvfold(batch: list[Any]) -> dict[str, Any]:
    features, target = collate_single(batch)
    count = int((features.design_mask.bool() & target.mask).sum())
    if not count:
        raise ValueError("the batch has no known designed polymer node to score")
    return {"input": features, "target": {"input": features, "target": target}, "loss_normalizer": count}


def collate_alphafold2(batch: list[Any]) -> dict[str, Any]:
    features, target = collate_single(batch)
    if target is not None:
        raise ValueError("masked-MSA supervision is constructed by the model from its input alignment")
    return {"input": features, "target": None, "loss_normalizer": 1}


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
    num_structures = _resolved_structure_count(resolved)
    return {
        "input": {"input": input, "target": target},
        "target": resolved,
        "loss_normalizer": num_structures,
    }


def collate_boltzgen_denoising(batch: list[Any]) -> dict[str, Any]:
    input, target = collate_single(batch)
    resolved = resolved_atom_mask(
        target.coordinate_mask,
        input.atom_mask,
        target.coordinate_atom_index,
    )
    num_structures = _resolved_structure_count(resolved)
    return {
        "input": {"input": input, "target": target},
        "target": resolved,
        "loss_normalizer": num_structures,
    }


def collate_boltzgen_ifold(batch: list[Any]) -> dict[str, Any]:
    features, target = collate_single(batch)
    mask = features.token_mask.bool() & features.token_resolved_mask.bool() & features.design_mask.bool() & target.mask
    count = int(mask.sum())
    if not count:
        raise ValueError("the batch has no known designed BoltzGen token to score")
    return {"input": features, "target": {"input": features, "target": target}, "loss_normalizer": count}


def collate_odesign(batch: list[Any]) -> dict[str, Any]:
    input, target = collate_single(batch)
    scored = target.coordinate_mask.bool() & input.atom_mask.bool() & ~input.fixed_coordinate_mask.bool()
    if not bool(scored.any()):
        raise ValueError("the batch has no resolved generated atom to score")
    pair = {"input": input, "target": target}
    return {"input": pair, "target": pair, "loss_normalizer": 1}
