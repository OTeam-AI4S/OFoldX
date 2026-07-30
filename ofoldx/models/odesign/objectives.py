r"""
ODesign coordinate-denoising objective.
"""

from __future__ import annotations

from typing import Any

from torch import Tensor

from ...data.molecular import Molecule
from ...modules.diffusion.loss import edm_loss_weight, resolved_atom_mask
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

    resolved = resolved_atom_mask(target, model_input)
    scored = resolved & ~model_input.fixed_coordinate_mask.bool()
    atom_index = target.atom_index.long()
    target_coordinates = target.coordinates.gather(
        1,
        atom_index.clamp_min(0).unsqueeze(-1).expand(-1, -1, 3),
    )
    target_coordinates = target_coordinates.masked_fill(~resolved.unsqueeze(-1), 0)

    system = model_input.layout.system
    weights = []
    for site in model_input.layout.atom_sites:
        entity = system.entities[system.instances[site.instance_id]]
        if isinstance(entity, Molecule):
            weights.append(11.0)
        elif entity.polymer_type != "protein":
            weights.append(6.0)
        else:
            weights.append(1.0)
    atom_weight = model_input.reference_positions.new_tensor(weights).unsqueeze(0).expand_as(scored)
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
