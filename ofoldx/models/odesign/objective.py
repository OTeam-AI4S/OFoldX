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

r"""
Released ODesign training objectives.
"""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ...data.molecular import MonomerType
from ...modules.atom.coordinates import gather_observed_coordinates
from ...modules.diffusion.edm import edm_loss_weight
from ...modules.diffusion.loss import bond_loss
from ...modules.diffusion.loss import smooth_lddt_loss as smooth_coordinate_lddt_loss
from ...modules.geometry import weighted_rigid_align
from ..processing import project_model_coordinates
from .model import (
    ODesignInput,
    ODesignLigandInput,
    ODesignOutput,
    ODesignStructureTarget,
)


def weighted_denoising_loss(
    pred: ODesignOutput,
    input: ODesignInput | ODesignLigandInput,
    target: ODesignStructureTarget,
    *,
    sigma_data: float = 16.0,
) -> Tensor:
    r"""
    Compute the molecule-weighted, independently aligned EDM coordinate loss.

    Fixed atoms provide context but are not scored. Protein atoms have weight 1,
    nucleic-acid atoms weight 6, and ligand or provisional atoms weight 11.
    """

    resolved = target.coordinate_mask.bool() & input.atom_mask.bool()
    scored = resolved & ~input.fixed_coordinate_mask.bool()
    low_precision = pred.coordinates.dtype in (torch.float16, torch.bfloat16)
    compute_dtype = torch.float32 if low_precision else pred.coordinates.dtype
    target_coordinates = project_model_coordinates(
        target.coordinates, scored, dtype=compute_dtype, feature="ODesign loss truth"
    )
    predicted_coordinates = project_model_coordinates(
        pred.coordinates,
        scored.unsqueeze(1).expand(pred.coordinates.shape[:-1]),
        dtype=compute_dtype,
        feature="ODesign loss prediction",
    )
    monomer = input.atom_monomer
    atom_weight = torch.where(
        monomer == MonomerType.PROTEIN,
        1,
        torch.where(monomer == MonomerType.NONE, 11, 6),
    ).to(device=pred.coordinates.device, dtype=compute_dtype)
    alignment_weight = (atom_weight * scored).unsqueeze(1).expand_as(pred.coordinates[..., 0])
    context = (
        torch.autocast(device_type=pred.coordinates.device.type, enabled=False) if low_precision else nullcontext()
    )
    with context:
        aligned_target = weighted_rigid_align(
            source=target_coordinates.unsqueeze(1).expand_as(pred.coordinates),
            target=predicted_coordinates,
            weight=alignment_weight,
        ).detach()
    squared_error = (predicted_coordinates - aligned_target).square().sum(dim=-1)
    score_mask = scored.unsqueeze(1).expand_as(squared_error)
    # A structure whose atoms are all fixed has nothing to score, so its average is undefined and it
    # reports NaN. Dividing by a safe count and selecting the NaN afterwards keeps a 0/0 out of the
    # gradient, and weighting before the selection keeps the NaN off the EDM co-factor as well.
    scored_atoms = score_mask.sum(dim=-1)
    per_sample = (squared_error * atom_weight.unsqueeze(1) * score_mask).sum(dim=-1) / (3 * scored_atoms.clamp_min(1))
    noise_level = pred.noise_level.float() if low_precision else pred.noise_level
    weighted = per_sample * edm_loss_weight(noise_level, sigma_data)
    return torch.where(scored_atoms > 0, weighted, weighted.new_full((), float("nan"))).nanmean()


def covalent_bond_distance_loss(
    pred: ODesignOutput,
    input: ODesignInput | ODesignLigandInput,
    target: ODesignStructureTarget,
    *,
    sigma_data: float = 16.0,
) -> Tensor:
    r"""Score resolved covalent distances whose endpoints are ligand or atom tokens."""

    edge = target.covalent_bond_index.long()
    left = edge[..., 0]
    right = edge[..., 1]
    resolved = target.coordinate_mask.bool() & input.atom_mask.bool()
    atom_token = input.atom_token_mask.bool().gather(1, input.atom_to_token.long())
    atomized_or_ligand = atom_token | input.atom_monomer.eq(MonomerType.NONE)
    valid = (
        resolved.gather(-1, left)
        & resolved.gather(-1, right)
        & atomized_or_ligand.gather(-1, left)
        & atomized_or_ligand.gather(-1, right)
    )
    # An absent bond mean remains NaN in the result. Exclude it before EDM weighting
    # so its cofactor has zero gradient; nanmean then retains the structures with bonds.
    has_bonds = valid.any(dim=-1, keepdim=True)
    truth = project_model_coordinates(
        target.coordinates, resolved, dtype=pred.coordinates.dtype, feature="ODesign bond truth"
    )
    per_sample = bond_loss(pred.coordinates, truth[:, None], edge, valid)
    per_sample = torch.where(has_bonds, per_sample, 0.0)
    weighted = per_sample * edm_loss_weight(pred.noise_level, sigma_data)
    return torch.where(has_bonds, weighted, weighted.new_full((), float("nan"))).nanmean()


def smooth_lddt_loss(
    pred: ODesignOutput,
    input: ODesignInput | ODesignLigandInput,
    target: ODesignStructureTarget,
) -> Tensor:
    r"""Compute the released smooth-lDDT coordinate objective."""

    resolved = target.coordinate_mask.bool() & input.atom_mask.bool()
    is_nucleotide = input.atom_monomer.eq(MonomerType.RNA) | input.atom_monomer.eq(MonomerType.DNA)
    return smooth_coordinate_lddt_loss(
        pred.coordinates,
        project_model_coordinates(
            target.coordinates, resolved, dtype=pred.coordinates.dtype, feature="ODesign lDDT truth"
        )[:, None],
        resolved,
        is_nucleotide,
    ).nanmean()


def distogram_loss(
    pred: ODesignOutput,
    input: ODesignInput | ODesignLigandInput,
    target: ODesignStructureTarget,
    *,
    num_bins: int = 64,
    min_distance: float = 2.3125,
    max_distance: float = 21.6875,
) -> Tensor:
    r"""Score representative-token distance bins."""

    representative = input.token_representative_atom_index.long()
    coordinates, representative_mask = gather_observed_coordinates(
        target.coordinates,
        target.coordinate_mask.bool() & input.atom_mask.bool(),
        representative,
        input.token_mask.bool(),
    )
    representative_coordinates = project_model_coordinates(
        coordinates, representative_mask, dtype=pred.distogram_logits.dtype, feature="ODesign distogram truth"
    )
    pair_mask = representative_mask.unsqueeze(-1) & representative_mask.unsqueeze(-2)
    distance = torch.cdist(representative_coordinates, representative_coordinates)
    boundaries = torch.linspace(
        min_distance,
        max_distance,
        num_bins - 1,
        dtype=distance.dtype,
        device=distance.device,
    )
    label = (distance.unsqueeze(-1) > boundaries).sum(dim=-1)
    error = (
        -F.log_softmax(pred.distogram_logits.masked_fill(~pair_mask.unsqueeze(-1), 0), dim=-1)
        .gather(
            -1,
            label.unsqueeze(-1),
        )
        .squeeze(-1)
    )
    # Reduced per structure like the other terms, so a structure without representative atoms reports
    # NaN for itself instead of being absorbed into its neighbours' denominator, and ``nanmean`` keeps
    # the term alive for the structures that did score.
    count = pair_mask.sum(dim=(-1, -2))
    per_item = (error * pair_mask).sum(dim=(-1, -2)) / count.clamp_min(1)
    per_item = torch.where(count > 0, per_item, per_item.new_full((), float("nan")))
    return per_item.nanmean()


def token_bond_loss(
    pred: ODesignOutput,
    input: ODesignInput | ODesignLigandInput,
    target: ODesignStructureTarget,
) -> Tensor:
    r"""Score binary covalent bonds on eligible pairs with an unknown endpoint."""

    num_tokens = input.token_mask.shape[-1]
    labels = torch.zeros(
        input.token_mask.shape[0],
        num_tokens * num_tokens,
        dtype=torch.int64,
        device=pred.token_bond_logits.device,
    )
    edge = target.covalent_bond_index.long()
    left_token = input.atom_to_token.gather(-1, edge[..., 0])
    right_token = input.atom_to_token.gather(-1, edge[..., 1])
    labels.scatter_(1, left_token * num_tokens + right_token, 1)
    labels.scatter_(1, right_token * num_tokens + left_token, 1)
    labels = labels.reshape(-1, num_tokens, num_tokens)

    atom_token = input.atom_token_mask.bool()
    generated_count = torch.zeros_like(input.token_mask, dtype=torch.int64)
    generated_count.scatter_add_(1, input.atom_to_token.long(), input.generated_atom_mask.to(torch.int64))
    canonical_atom_token = atom_token & ~generated_count.bool()
    fixed_count = torch.zeros_like(input.token_mask, dtype=torch.int64)
    fixed_count.scatter_add_(
        1,
        input.atom_to_token.long(),
        input.fixed_coordinate_mask.to(torch.int64),
    )
    known_token = fixed_count.bool() | canonical_atom_token
    candidate = input.token_mask.bool() & atom_token
    pair_mask = (
        candidate.unsqueeze(-1) & candidate.unsqueeze(-2) & ~(known_token.unsqueeze(-1) & known_token.unsqueeze(-2))
    )
    token_pairs = input.atom_to_token.unsqueeze(-1) * num_tokens + input.atom_to_token.unsqueeze(-2)
    atom_pairs = input.atom_mask.bool().unsqueeze(-1) & input.atom_mask.bool().unsqueeze(-2)
    unknown_count = labels.new_zeros((labels.shape[0], num_tokens * num_tokens))
    pair_count = torch.zeros_like(unknown_count)
    unknown_count.scatter_add_(
        1, token_pairs.flatten(1).long(), (atom_pairs & ~target.covalent_pair_mask.bool()).flatten(1).long()
    )
    pair_count.scatter_add_(1, token_pairs.flatten(1).long(), atom_pairs.flatten(1).long())
    known_pair = ((pair_count > 0) & (unknown_count == 0)).reshape_as(labels) | labels.bool()
    pair_mask &= known_pair
    pair_mask = torch.triu(pair_mask, diagonal=1)
    error = (
        -F.log_softmax(pred.token_bond_logits.masked_fill(~pair_mask.unsqueeze(-1), 0), dim=-1)
        .gather(
            -1,
            labels.unsqueeze(-1),
        )
        .squeeze(-1)
    )
    count = pair_mask.sum(dim=(-1, -2))
    per_item = (error * pair_mask).sum(dim=(-1, -2)) / count.clamp_min(1)
    per_item = torch.where(count > 0, per_item, per_item.new_full((), float("nan")))
    # ``nanmean`` keeps an empty scope confined to its own structure; a plain mean would turn the whole
    # batch NaN and the caller's ``nansum`` would drop the term along with every scored structure.
    return per_item.nanmean()


def odesign_loss(
    pred: ODesignOutput,
    input: ODesignInput | ODesignLigandInput,
    target: ODesignStructureTarget,
    *,
    sigma_data: float = 16.0,
    num_distogram_bins: int = 64,
    min_distogram_distance: float = 2.3125,
    max_distogram_distance: float = 21.6875,
) -> tuple[Tensor, dict[str, Tensor]]:
    r"""Return the released weighted objective and its five unreweighted terms."""

    losses = {
        "weighted_denoising_loss": weighted_denoising_loss(
            pred,
            input,
            target,
            sigma_data=sigma_data,
        ),
        "covalent_bond_distance_loss": covalent_bond_distance_loss(
            pred,
            input,
            target,
            sigma_data=sigma_data,
        ),
        "smooth_lddt_loss": smooth_lddt_loss(pred, input, target),
        "distogram_loss": distogram_loss(
            pred,
            input,
            target,
            num_bins=num_distogram_bins,
            min_distance=min_distogram_distance,
            max_distance=max_distogram_distance,
        ),
        "token_bond_loss": token_bond_loss(pred, input, target),
    }
    # Every term returns NaN when its own scope is empty (a plain polymer, for instance, has no
    # atomized or ligand bond to score). ``nansum`` is where that NaN is consumed: an inapplicable
    # term drops out of the total instead of poisoning it, while the term itself still reports NaN.
    total = torch.stack(
        (
            4 * losses["weighted_denoising_loss"],
            4 * losses["covalent_bond_distance_loss"],
            4 * losses["smooth_lddt_loss"],
            0.03 * losses["distogram_loss"],
            losses["token_bond_loss"],
        )
    ).nansum(dim=0)
    return total, losses


class ODesignLoss(nn.Module):
    r"""
    The released weighted objective's total, discarding its five unreweighted terms.

    Args:
        sigma_data: Positive EDM data scale used by the producing model.
        num_distogram_bins: Number of representative-token distance bins.
        min_distogram_distance: Lower edge of the distogram distance grid, in angstroms.
        max_distogram_distance: Upper edge of the distogram distance grid, in angstroms.
    """

    def __init__(
        self,
        *,
        sigma_data: float = 16.0,
        num_distogram_bins: int = 64,
        min_distogram_distance: float = 2.3125,
        max_distogram_distance: float = 21.6875,
    ) -> None:
        super().__init__()
        self.sigma_data = sigma_data
        self.num_distogram_bins = num_distogram_bins
        self.min_distogram_distance = min_distogram_distance
        self.max_distogram_distance = max_distogram_distance

    def forward(
        self,
        pred: ODesignOutput,
        input: ODesignInput | ODesignLigandInput,
        target: ODesignStructureTarget,
    ) -> Tensor:
        total, _ = odesign_loss(
            pred,
            input,
            target,
            sigma_data=self.sigma_data,
            num_distogram_bins=self.num_distogram_bins,
            min_distogram_distance=self.min_distogram_distance,
            max_distogram_distance=self.max_distogram_distance,
        )
        return total
