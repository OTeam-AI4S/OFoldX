r"""
ODesign coordinate-denoising objective.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from ...data.molecular import ENTITY_MOLECULE, ENTITY_PROTEIN
from ...modules.diffusion.loss import edm_loss_weight
from ...modules.geometry import weighted_rigid_align


def weighted_denoising_loss(
    pred: Any,
    model_input: Any,
    target: Any,
    *,
    sigma_data: float,
) -> Tensor:
    r"""
    Compute ODesign's molecule-weighted, aligned EDM coordinate objective.

    Fixed conditioning atoms define context but are not scored. The target is
    independently aligned for every sampled denoising output with weights 1 for
    protein, 6 for DNA/RNA, and 11 for ligand atoms.
    """

    resolved = target.coordinate_mask.bool() & model_input.atom_mask.bool()
    scored = resolved & ~model_input.fixed_coordinate_mask.bool()
    target_coordinates = target.coordinates.masked_fill(
        ~resolved.unsqueeze(-1),
        0,
    )

    system = model_input.system
    component_instance = torch.repeat_interleave(
        torch.arange(
            system.instance_symbol.numel(),
            dtype=torch.int64,
            device=system.atom_symbol.device,
        ),
        system.instance_component_ptr[1:] - system.instance_component_ptr[:-1],
    )
    atom_component = torch.repeat_interleave(
        torch.arange(
            system.component_symbol.numel(),
            dtype=torch.int64,
            device=system.atom_symbol.device,
        ),
        system.component_atom_ptr[1:] - system.component_atom_ptr[:-1],
    )
    atom_entity_type = system.entity_type[system.instance_entity[component_instance[atom_component]]]
    canonical_weight = torch.where(
        atom_entity_type == ENTITY_PROTEIN,
        1.0,
        torch.where(atom_entity_type == ENTITY_MOLECULE, 11.0, 6.0),
    ).to(model_input.reference_positions)
    atom_index = model_input.atom_index.long()
    canonical = atom_index >= 0
    atom_weight = torch.full_like(
        atom_index,
        11.0,
        dtype=model_input.reference_positions.dtype,
    )
    atom_weight[canonical] = canonical_weight[atom_index[canonical].to(canonical_weight.device)].to(atom_weight.device)
    alignment_weight = (atom_weight * scored).unsqueeze(1).expand_as(pred.coordinates[..., 0])
    aligned_target = weighted_rigid_align(
        source=target_coordinates.unsqueeze(1).expand_as(pred.coordinates),
        target=pred.coordinates,
        weight=alignment_weight,
    )
    squared_error = (pred.coordinates - aligned_target).square().sum(dim=-1)
    noise_weight = edm_loss_weight(pred.noise_level, sigma_data).unsqueeze(-1)
    score_mask = scored.unsqueeze(1).expand_as(squared_error)
    weighted_error = squared_error * noise_weight * atom_weight.unsqueeze(1)
    return (weighted_error * score_mask).sum() / (3 * (score_mask.sum() + 1e-6))


def criterion_factory(model: Any, processor: Any) -> Any:
    r"""Bind ODesign's denoising objective to the Model's EDM data scale."""

    del processor
    sigma_data = float(model.config.denoiser.sigma_data)

    def criterion(pred: Any, target: Any) -> Tensor:
        return weighted_denoising_loss(
            pred,
            target["input"],
            target["target"],
            sigma_data=sigma_data,
        )

    return criterion
