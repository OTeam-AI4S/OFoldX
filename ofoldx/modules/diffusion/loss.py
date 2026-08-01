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
    r"""Return the rigidly aligned, EDM-weighted mean coordinate error."""

    resolved = resolved.bool()
    aligned_target = weighted_rigid_align(
        source=prediction.augmented_target_coordinates.masked_fill(
            ~resolved[:, None, :, None],
            0,
        ),
        target=prediction.coordinates,
        weight=resolved.to(prediction.coordinates)[:, None, :],
    )
    squared_error = (prediction.coordinates - aligned_target).square().sum(dim=-1)
    weighted_error = squared_error * edm_loss_weight(
        prediction.noise_level,
        sigma_data,
    ).unsqueeze(-1)
    mask = resolved.to(weighted_error)[:, None, :].expand_as(weighted_error)
    return (weighted_error * mask).sum() / mask.sum()
