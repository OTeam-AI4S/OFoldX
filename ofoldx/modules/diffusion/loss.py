r"""Shared coordinate-denoising objectives for diffusion structure models."""

from __future__ import annotations

from typing import Any

from torch import Tensor

from ..geometry import weighted_rigid_align
from .edm import edm_loss_weight


def resolved_atom_mask(
    coordinate_mask: Tensor,
    atom_mask: Tensor,
    coordinate_atom_index: Tensor,
) -> Tensor:
    r"""Gather resolved coordinates onto an explicit model atom axis.

    Args:
        coordinate_mask: Resolved-coordinate mask on the target coordinate axis.
        atom_mask: Valid rows on the model atom axis.
        coordinate_atom_index: Model-row to target-coordinate-row relation; ``-1``
            marks a row without coordinate truth.

    Returns:
        Boolean mask aligned with ``coordinate_atom_index``.
    """

    coordinate_atom_index = coordinate_atom_index.long()
    valid = coordinate_atom_index >= 0
    resolved = coordinate_mask.gather(1, coordinate_atom_index.clamp_min(0)).bool()
    return resolved & valid & atom_mask.bool()


def edm_denoising_loss(
    prediction: Any,
    resolved: Tensor,
    *,
    sigma_data: float,
) -> Tensor:
    r"""Return the rigidly aligned, EDM-weighted mean coordinate error.

    Atom counts vary between structures while the diffusion sample axis is an
    independent Monte Carlo axis.  The objective therefore averages resolved
    atoms within each ``(structure, sample)`` pair before averaging structures
    and samples.  This gives every structure and every corruption equal weight,
    independent of molecular size.
    """

    resolved = resolved.bool()
    sample_resolved = resolved.unsqueeze(-2)
    aligned_target = weighted_rigid_align(
        source=prediction.augmented_target_coordinates.where(sample_resolved.unsqueeze(-1), 0.0),
        target=prediction.coordinates,
        weight=sample_resolved.to(prediction.coordinates),
    )
    squared_error = (prediction.coordinates - aligned_target).square().sum(dim=-1)
    weighted_error = squared_error * edm_loss_weight(
        prediction.noise_level,
        sigma_data,
    ).unsqueeze(-1)
    mask = sample_resolved.to(weighted_error)
    per_sample = (weighted_error * mask).sum(dim=-1) / mask.sum(dim=-1)
    return per_sample.mean()
