r"""
Released ODesign training objectives.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import Tensor

from ...data.molecular import ENTITY_MOLECULE, ENTITY_PROTEIN
from ...modules.diffusion.edm import edm_loss_weight
from ...modules.geometry import weighted_rigid_align
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
    target_coordinates = target.coordinates.to(pred.coordinates).masked_fill(
        ~resolved.unsqueeze(-1),
        0,
    )
    entity_type = _atom_entity_type(input)
    atom_weight = torch.where(
        entity_type == ENTITY_PROTEIN,
        1,
        torch.where(entity_type == ENTITY_MOLECULE, 11, 6),
    ).to(pred.coordinates)
    alignment_weight = (atom_weight * scored).unsqueeze(1).expand_as(pred.coordinates[..., 0])
    aligned_target = weighted_rigid_align(
        source=target_coordinates.unsqueeze(1).expand_as(pred.coordinates),
        target=pred.coordinates,
        weight=alignment_weight,
    ).detach()
    squared_error = (pred.coordinates - aligned_target).square().sum(dim=-1)
    score_mask = scored.unsqueeze(1).expand_as(squared_error)
    # A structure whose atoms are all fixed has nothing to score, so its average is undefined and it
    # reports NaN. Dividing by a safe count and selecting the NaN afterwards keeps a 0/0 out of the
    # gradient, and weighting before the selection keeps the NaN off the EDM co-factor as well.
    scored_atoms = score_mask.sum(dim=-1)
    per_sample = (squared_error * atom_weight.unsqueeze(1) * score_mask).sum(dim=-1) / (3 * scored_atoms.clamp_min(1))
    weighted = per_sample * edm_loss_weight(pred.noise_level, sigma_data)
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
    num_samples = pred.coordinates.shape[1]
    left_pred = pred.coordinates.gather(
        -2,
        left[:, None, :, None].expand(-1, num_samples, -1, 3),
    )
    right_pred = pred.coordinates.gather(
        -2,
        right[:, None, :, None].expand(-1, num_samples, -1, 3),
    )
    target_coordinates = target.coordinates.to(pred.coordinates)
    left_true = target_coordinates.gather(
        -2,
        left[..., None].expand(-1, -1, 3),
    )
    right_true = target_coordinates.gather(
        -2,
        right[..., None].expand(-1, -1, 3),
    )
    squared_error = (
        (left_pred - right_pred).norm(dim=-1) - (left_true - right_true).norm(dim=-1).unsqueeze(1)
    ).square()

    resolved = target.coordinate_mask.bool() & input.atom_mask.bool()
    entity_type = _atom_entity_type(input)
    atom_token = input.token_atom_index.gather(1, input.atom_to_token.long()).ge(0)
    atomized_or_ligand = atom_token | entity_type.eq(ENTITY_MOLECULE) | input.atom_index.lt(0)
    valid = (
        resolved.gather(-1, left)
        & resolved.gather(-1, right)
        & atomized_or_ligand.gather(-1, left)
        & atomized_or_ligand.gather(-1, right)
    )
    # A plain polymer has no atomized or ligand bond to score, so this scope is routinely empty and the
    # average is undefined: report NaN. Dividing by a safe count and selecting the NaN afterwards keeps
    # a 0/0 out of the gradient -- ``torch.where`` routes zero gradient into the empty branch -- and
    # weighting before the selection keeps the NaN off the EDM co-factor as well. ``nanmean`` then
    # reduces over the structures that did have a bond to score: a plain mean would return NaN for the
    # whole batch as soon as one structure's scope was empty, and the caller's ``nansum`` would drop
    # this term entirely, discarding the gradient of every structure that did score.
    count = valid.sum(dim=-1, keepdim=True)
    per_sample = (squared_error * valid.unsqueeze(1)).sum(dim=-1) / count.clamp_min(1)
    weighted = per_sample * edm_loss_weight(pred.noise_level, sigma_data)
    return torch.where(count > 0, weighted, weighted.new_full((), float("nan"))).nanmean()


def smooth_lddt_loss(
    pred: ODesignOutput,
    input: ODesignInput | ODesignLigandInput,
    target: ODesignStructureTarget,
) -> Tensor:
    r"""Compute the released smooth-lDDT coordinate objective."""

    target_coordinates = target.coordinates.to(pred.coordinates)
    true_distance = torch.cdist(target_coordinates, target_coordinates)
    resolved = target.coordinate_mask.bool() & input.atom_mask.bool()
    distance_mask = resolved.unsqueeze(-1) & resolved.unsqueeze(-2)
    entity_type = _atom_entity_type(input)
    is_nucleotide = entity_type.ne(ENTITY_PROTEIN) & entity_type.ne(ENTITY_MOLECULE)
    locality = torch.where(
        is_nucleotide.unsqueeze(-1),
        true_distance < 30,
        true_distance < 15,
    )
    diagonal = torch.eye(
        target_coordinates.shape[-2],
        dtype=torch.bool,
        device=target_coordinates.device,
    )
    pair_mask = distance_mask & locality & ~diagonal
    pair_count = pair_mask.sum(dim=(-1, -2))

    def score_chunk(
        sample_coordinates: Tensor,
        target_distance: Tensor,
        mask: Tensor,
        count: Tensor,
    ) -> Tensor:
        pred_distance = torch.cdist(sample_coordinates, sample_coordinates)
        difference = (pred_distance - target_distance).abs()
        score = sum(torch.sigmoid(threshold - difference) for threshold in (0.5, 1.0, 2.0, 4.0)) / 4
        per_sample = (score * mask).sum(dim=(-1, -2)) / count.clamp_min(1)
        return torch.where(count > 0, per_sample, per_sample.new_full((), float("nan")))

    sample_scores: list[Tensor] = []
    for sample_index in range(pred.coordinates.shape[1]):
        sample_coordinates = pred.coordinates[:, sample_index]
        if torch.is_grad_enabled() and sample_coordinates.requires_grad:
            sample_lddt = torch.utils.checkpoint.checkpoint(
                score_chunk,
                sample_coordinates,
                true_distance,
                pair_mask,
                pair_count,
                use_reentrant=False,
            )
        else:
            sample_lddt = score_chunk(
                sample_coordinates,
                true_distance,
                pair_mask,
                pair_count,
            )
        sample_scores.append(sample_lddt)
    if not sample_scores:
        return pred.coordinates.new_full((), float("nan"))
    # Each structure keeps its own row so an empty scope stays confined to it: ``nanmean`` averages the
    # scored (structure, sample) entries and only reports NaN when every one is empty. Summing the rows
    # first would let a single empty structure turn the whole term NaN, and the caller's ``nansum``
    # would then drop it -- discarding the gradient of the structures that did score.
    return 1 - torch.stack(sample_scores, dim=-1).nanmean()


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

    representative = input.representative_atom_index.long()
    coordinates = target.coordinates.to(pred.distogram_logits)
    representative_coordinates = coordinates.gather(
        -2,
        representative.clamp_min(0)[..., None].expand(-1, -1, 3),
    )
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
        -F.log_softmax(pred.distogram_logits, dim=-1)
        .gather(
            -1,
            label.unsqueeze(-1),
        )
        .squeeze(-1)
    )
    representative_mask = (
        input.token_mask.bool()
        & (representative >= 0)
        & target.coordinate_mask.gather(-1, representative.clamp_min(0)).bool()
    )
    pair_mask = representative_mask.unsqueeze(-1) & representative_mask.unsqueeze(-2)
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

    atom_token = input.token_atom_index.ge(0)
    provisional_count = torch.zeros_like(input.token_mask, dtype=torch.int64)
    provisional_count.scatter_add_(
        1,
        input.atom_to_token.long(),
        (input.atom_index < 0).to(torch.int64),
    )
    atom_token = atom_token | provisional_count.bool()
    fixed_count = torch.zeros_like(input.token_mask, dtype=torch.int64)
    fixed_count.scatter_add_(
        1,
        input.atom_to_token.long(),
        input.fixed_coordinate_mask.to(torch.int64),
    )
    known_token = fixed_count.bool() | input.token_atom_index.ge(0)
    candidate = input.token_mask.bool() & atom_token
    pair_mask = (
        candidate.unsqueeze(-1) & candidate.unsqueeze(-2) & ~(known_token.unsqueeze(-1) & known_token.unsqueeze(-2))
    )
    pair_mask = torch.triu(pair_mask, diagonal=1)
    error = (
        -F.log_softmax(pred.token_bond_logits, dim=-1)
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


def _atom_entity_type(input: ODesignInput | ODesignLigandInput) -> Tensor:
    system = input.system
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
    canonical_entity_type = system.entity_type[system.instance_entity[component_instance[atom_component]]].to(
        device=input.atom_index.device, dtype=input.atom_index.dtype
    )
    atom_index = input.atom_index.long()
    canonical = atom_index >= 0
    entity_type = torch.full_like(atom_index, ENTITY_MOLECULE)
    entity_type[canonical] = canonical_entity_type[atom_index[canonical]]
    return entity_type
