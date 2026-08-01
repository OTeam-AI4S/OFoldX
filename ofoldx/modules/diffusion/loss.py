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

r"""Shared coordinate-denoising objectives for diffusion structure models."""

from __future__ import annotations

import operator
from typing import Any, cast

import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from ..functional import weighted_mean
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
    backend: str = "auto",
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
    sample_mask = sample_resolved.unsqueeze(-1)
    coordinates = prediction.coordinates.where(sample_mask, 0.0)
    aligned_target = weighted_rigid_align(
        source=prediction.augmented_target_coordinates.where(sample_mask, 0.0),
        target=coordinates,
        weight=sample_resolved.to(coordinates),
        backend=backend,
    )
    squared_error = (coordinates - aligned_target.where(sample_mask, 0.0)).square().sum(dim=-1)
    weighted_error = squared_error * edm_loss_weight(
        prediction.noise_level,
        sigma_data,
    ).unsqueeze(-1)
    per_sample = weighted_mean(weighted_error, sample_resolved, backend=backend)
    return per_sample.mean()


def bond_loss(
    predicted_coordinates: Tensor,
    target_coordinates: Tensor,
    bond_atom_index: Tensor,
    bond_mask: Tensor,
) -> Tensor:
    r"""
    Return mean squared selected bond-length error per diffusion sample.

    Gather endpoints directly from the atom axis, so backward never creates an
    atom-by-bond tensor. Masked bonds may have unavailable coordinates or ``-1``
    endpoints; their geometry is zeroed before subtraction and norms. Both
    coordinate arguments retain gradients on scored endpoints. Arithmetic follows
    the ordinary promoted coordinate dtype.

    Args:
        predicted_coordinates: Predicted coordinates ``(B, S, A, 3)`` in angstroms.
        target_coordinates: Truth ``(B, 1, A, 3)`` shared across samples or
            sample-specific truth ``(B, S, A, 3)``, in angstroms.
        bond_atom_index: Model-row endpoint indices ``(B, E, 2)``. Every scored
            endpoint must index the atom axis.
        bond_mask: Scored-bond mask ``(B, E)``, shared across samples. Both
            endpoints of a scored bond must have finite coordinates.

    Returns:
        Mean squared error ``(B, S)`` in squared angstroms. Empty bond scopes
        report NaN with zero coordinate gradients.

    Raises:
        ValueError: If the target coordinate shape disagrees with predictions.
    """

    batch_size, num_samples, num_atoms, _ = predicted_coordinates.shape
    if target_coordinates.shape not in ((batch_size, 1, num_atoms, 3), predicted_coordinates.shape):
        raise ValueError("target_coordinates must have shape (B, 1, A, 3) or (B, S, A, 3)")
    dtype = torch.promote_types(predicted_coordinates.dtype, target_coordinates.dtype)
    bond_mask = bond_mask.bool()
    num_bonds = bond_atom_index.shape[1]
    endpoint_index = bond_atom_index.long().masked_fill(~bond_mask[..., None], 0)
    gather_index = endpoint_index.reshape(batch_size, 1, 2 * num_bonds, 1)
    predicted_bond_coordinates = (
        predicted_coordinates.gather(2, gather_index.expand(-1, num_samples, -1, 3))
        .reshape(batch_size, num_samples, num_bonds, 2, 3)
        .to(dtype)
        .masked_fill(~bond_mask[:, None, :, None, None], 0)
    )
    target_samples = target_coordinates.shape[1]
    target_bond_coordinates = (
        target_coordinates.gather(2, gather_index.expand(-1, target_samples, -1, 3))
        .reshape(batch_size, target_samples, num_bonds, 2, 3)
        .to(dtype)
        .masked_fill(~bond_mask[:, None, :, None, None], 0)
    )
    predicted_length = torch.linalg.vector_norm(
        predicted_bond_coordinates[..., 0, :] - predicted_bond_coordinates[..., 1, :],
        dim=-1,
    )
    target_length = torch.linalg.vector_norm(
        target_bond_coordinates[..., 0, :] - target_bond_coordinates[..., 1, :],
        dim=-1,
    )
    errors = (predicted_length - target_length).square().masked_fill(~bond_mask[:, None, :], 0)
    return weighted_mean(errors, bond_mask[:, None, :].to(errors))


def smooth_lddt_loss(
    predicted_coordinates: Tensor,
    target_coordinates: Tensor,
    target_mask: Tensor,
    atom_is_nucleotide: Tensor,
    *,
    nucleic_acid_cutoff: float = 30.0,
    other_cutoff: float = 15.0,
    sample_chunk_size: int = 1,
    pair_chunk_size: int = 1048576,
) -> Tensor:
    r"""
    Return smooth atom lDDT loss for each structure and diffusion sample.

    Score directed pairs of resolved model rows with distinct row indices and
    target distance strictly below the row atom's cutoff. Each pair contributes
    the mean of four sigmoids at distance-error thresholds 0.5, 1, 2 and 4
    angstroms. Normalize over all eligible pairs once per ``(B, S)``.
    Synthetic rows remain distinct even when they share coordinate truth.

    Coordinate subtraction and Euclidean norms avoid GEMM distance cancellation.
    Pair geometry, masks and scores are checkpointed together when gradients are
    enabled. Unresolved coordinates may contain NaNs and receive zero gradients;
    resolved coordinates must be finite. Gradients flow through both coordinate
    arguments, with pair selection treated as fixed. Arithmetic and reductions
    retain the promoted coordinate dtype.

    Args:
        predicted_coordinates: Predicted coordinates ``(B, S, A, 3)`` in angstroms.
        target_coordinates: Truth ``(B, 1, A, 3)`` shared across samples or
            sample-specific truth ``(B, S, A, 3)``, in angstroms.
        target_mask: Resolved model-row mask ``(B, A)``, shared across samples.
        atom_is_nucleotide: Nucleotide indicator ``(B, A)`` selecting row cutoffs.
        nucleic_acid_cutoff: Nucleotide neighborhood radius in angstroms.
        other_cutoff: Neighborhood radius for other atoms in angstroms.
        sample_chunk_size: Positive maximum samples evaluated in one tile.
        pair_chunk_size: Positive maximum row-count times column-count per tile,
            excluding batch and sample multipliers. Columns are bounded by the
            atom count; the remaining budget determines the row count.

    Returns:
        Loss ``(B, S)``. Empty pair scopes report NaN with zero coordinate
        gradients. Chunk sizes change the evaluation schedule, not the formula.

    Raises:
        ValueError: If a chunk size is not positive or input shapes disagree.

    Examples:
        >>> coordinates = torch.zeros(1, 2, 2, 3)
        >>> resolved = torch.ones(1, 2, dtype=torch.bool)
        >>> loss = smooth_lddt_loss(coordinates, coordinates[:, :1], resolved, ~resolved)
        >>> loss.round(decimals=4)
        tensor([[0.1959, 0.1959]])
    """

    loss, _ = smooth_lddt_loss_with_counts(
        predicted_coordinates,
        target_coordinates,
        target_mask,
        atom_is_nucleotide,
        nucleic_acid_cutoff=nucleic_acid_cutoff,
        other_cutoff=other_cutoff,
        sample_chunk_size=sample_chunk_size,
        pair_chunk_size=pair_chunk_size,
    )
    return loss


def smooth_lddt_loss_with_counts(
    predicted_coordinates: Tensor,
    target_coordinates: Tensor,
    target_mask: Tensor,
    atom_is_nucleotide: Tensor,
    *,
    nucleic_acid_cutoff: float = 30.0,
    other_cutoff: float = 15.0,
    sample_chunk_size: int = 1,
    pair_chunk_size: int = 1048576,
) -> tuple[Tensor, Tensor]:
    r"""Return per-sample losses and directed-pair counts without retaining pair tensors."""

    sample_chunk_size = operator.index(sample_chunk_size)
    pair_chunk_size = operator.index(pair_chunk_size)
    if sample_chunk_size < 1 or pair_chunk_size < 1:
        raise ValueError("sample_chunk_size and pair_chunk_size must be positive")
    if predicted_coordinates.ndim != 4 or predicted_coordinates.shape[-1] != 3:
        raise ValueError("predicted_coordinates must have shape (B, S, A, 3)")
    batch_size, num_samples, num_atoms, _ = predicted_coordinates.shape
    if target_coordinates.shape not in ((batch_size, 1, num_atoms, 3), predicted_coordinates.shape):
        raise ValueError("target_coordinates must have shape (B, 1, A, 3) or (B, S, A, 3)")
    if target_mask.shape != (batch_size, num_atoms) or atom_is_nucleotide.shape != target_mask.shape:
        raise ValueError("target_mask and atom_is_nucleotide must have shape (B, A)")

    dtype = torch.promote_types(predicted_coordinates.dtype, target_coordinates.dtype)
    # Empty slices retain a zero-gradient connection without reading masked NaNs.
    zero = predicted_coordinates[..., :0, :].sum(dim=(-2, -1)).to(dtype)
    zero = zero + target_coordinates[..., :0, :].sum(dim=(-2, -1)).to(dtype)
    if num_atoms == 0 or num_samples == 0:
        return zero + zero.new_full((), float("nan")), torch.zeros_like(zero, dtype=torch.long)

    column_chunk_size = min(num_atoms, pair_chunk_size)
    row_chunk_size = max(1, pair_chunk_size // column_chunk_size)
    checkpoint_tiles = torch.is_grad_enabled() and (
        predicted_coordinates.requires_grad or target_coordinates.requires_grad
    )
    losses = []
    counts = []
    for sample_start in range(0, num_samples, sample_chunk_size):
        sample_stop = min(sample_start + sample_chunk_size, num_samples)
        predicted_samples = predicted_coordinates[:, sample_start:sample_stop]
        target_samples = (
            target_coordinates if target_coordinates.shape[1] == 1 else target_coordinates[:, sample_start:sample_stop]
        )
        score_sum = zero[:, sample_start:sample_stop]
        count = torch.zeros_like(score_sum, dtype=torch.long)
        for row_start in range(0, num_atoms, row_chunk_size):
            row_stop = min(row_start + row_chunk_size, num_atoms)
            for column_start in range(0, num_atoms, column_chunk_size):
                column_stop = min(column_start + column_chunk_size, num_atoms)
                arguments = (
                    predicted_samples[:, :, row_start:row_stop],
                    predicted_samples[:, :, column_start:column_stop],
                    target_samples[:, :, row_start:row_stop],
                    target_samples[:, :, column_start:column_stop],
                    target_mask[:, row_start:row_stop],
                    target_mask[:, column_start:column_stop],
                    atom_is_nucleotide[:, row_start:row_stop],
                    row_start,
                    column_start,
                    nucleic_acid_cutoff,
                    other_cutoff,
                )
                if checkpoint_tiles:
                    tile_score, tile_count = cast(
                        tuple[Tensor, Tensor],
                        checkpoint(
                            _smooth_lddt_tile,
                            *arguments,
                            use_reentrant=False,
                            preserve_rng_state=False,
                        ),
                    )
                else:
                    tile_score, tile_count = _smooth_lddt_tile(*arguments)
                score_sum = score_sum + tile_score
                count = count + tile_count
        value = 1.0 - score_sum / count.clamp_min(1)
        losses.append(torch.where(count > 0, value, value.new_full((), float("nan"))))
        counts.append(count)
    return torch.cat(losses, dim=1), torch.cat(counts, dim=1)


def _smooth_lddt_tile(
    predicted_rows: Tensor,
    predicted_columns: Tensor,
    target_rows: Tensor,
    target_columns: Tensor,
    row_mask: Tensor,
    column_mask: Tensor,
    row_is_nucleotide: Tensor,
    row_start: int,
    column_start: int,
    nucleic_acid_cutoff: float,
    other_cutoff: float,
) -> tuple[Tensor, Tensor]:
    r"""Own all pair geometry and activations for one checkpointed rectangular tile."""

    dtype = torch.promote_types(predicted_rows.dtype, target_rows.dtype)
    row_mask = row_mask[:, None, :].bool()
    column_mask = column_mask[:, None, :].bool()
    predicted_rows = predicted_rows.to(dtype).masked_fill(~row_mask[..., None], 0)
    predicted_columns = predicted_columns.to(dtype).masked_fill(~column_mask[..., None], 0)
    target_rows = target_rows.to(dtype).masked_fill(~row_mask[..., None], 0)
    target_columns = target_columns.to(dtype).masked_fill(~column_mask[..., None], 0)
    row_index = torch.arange(row_start, row_start + predicted_rows.shape[-2], device=predicted_rows.device)
    column_index = torch.arange(
        column_start, column_start + predicted_columns.shape[-2], device=predicted_columns.device
    )
    pair_mask = row_mask.unsqueeze(-1) & column_mask.unsqueeze(-2)
    pair_mask = pair_mask & (row_index[:, None] != column_index[None, :])
    target_displacement = target_rows.unsqueeze(-2) - target_columns.unsqueeze(-3)
    target_displacement = target_displacement.masked_fill(~pair_mask[..., None], 0)
    target_distance = torch.linalg.vector_norm(target_displacement, dim=-1)

    cutoff = torch.where(
        row_is_nucleotide.bool(),
        target_distance.new_tensor(nucleic_acid_cutoff),
        target_distance.new_tensor(other_cutoff),
    )[:, None, :, None]
    pair_mask = pair_mask & (target_distance < cutoff)
    predicted_displacement = predicted_rows.unsqueeze(-2) - predicted_columns.unsqueeze(-3)
    predicted_displacement = predicted_displacement.masked_fill(~pair_mask[..., None], 0)
    predicted_distance = torch.linalg.vector_norm(predicted_displacement, dim=-1)
    distance_error = (target_distance.masked_fill(~pair_mask, 0) - predicted_distance).abs()
    score = 0.25 * (
        torch.sigmoid(0.5 - distance_error)
        + torch.sigmoid(1.0 - distance_error)
        + torch.sigmoid(2.0 - distance_error)
        + torch.sigmoid(4.0 - distance_error)
    )
    score_sum = score.masked_fill(~pair_mask, 0).sum(dim=(-2, -1))
    count = pair_mask.sum(dim=(-2, -1)).expand_as(score_sum)
    return score_sum, count
