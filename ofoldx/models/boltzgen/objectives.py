r"""
BoltzGen training objectives.

The Runner trains BoltzGen with the shared EDM coordinate-denoising objective (see the registered
contract below). ``diffusion_loss`` here is BoltzGen's own fuller published objective (rigid-aligned,
type-weighted MSE plus smooth lDDT and bond terms), kept as a family API for callers who want it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor

from ...modules.diffusion.loss import denoising_criterion_factory
from ...modules.geometry import weighted_rigid_align
from ...modules.head.tm import tm_length_scale
from ..training import TRAINING_CONTRACTS, TrainingContract
from .model import (
    BoltzGenConfidenceOutput,
    BoltzGenInput,
    BoltzGenOutput,
    BoltzGenStructureTarget,
    _boltzgen_coordinate_frames,
)

TRAINING_CONTRACTS.register(
    TrainingContract(
        architectures=("boltzgen",),
        criterion=denoising_criterion_factory,
        strategy="denoising",
        freeze=("distogram_head", "bfactor_head", "template_embedder"),
    )
)

_RESIDUE_TYPE_WEIGHTS = (
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    2.0,
    1.0,
    2.0,
    1.0,
    1.0,
    2.0,
    4.0,
    2.0,
    2.0,
    2.0,
    2.0,
    4.0,
    4.0,
    4.0,
    4.0,
    4.0,
    4.0,
    10.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
)


@dataclass(eq=False)
class BoltzGenConfidenceLoss:
    r"""
    Unreduced BoltzGen confidence loss and its categorical components.

    Every field has shape ``(B, S)``.
    """

    loss: Tensor
    plddt_loss: Tensor
    pde_loss: Tensor
    resolved_loss: Tensor
    pae_loss: Tensor


@dataclass(eq=False)
class BoltzGenDiffusionLoss:
    r"""
    Unreduced BoltzGen coordinate loss and its components.

    Every field has shape ``(B, S)``.
    """

    loss: Tensor
    mse_loss: Tensor
    smooth_lddt_loss: Tensor
    bond_loss: Tensor


def confidence_loss(
    output: BoltzGenConfidenceOutput,
    predicted_coordinates: Tensor,
    target_coordinates: Tensor,
    target_mask: Tensor,
    input: BoltzGenInput,
    *,
    token_level_confidence: bool,
    pae_weight: float = 0.0,
) -> BoltzGenConfidenceLoss:
    r"""
    Compose the released BoltzGen categorical confidence objectives.

    Coordinates and the target mask have shapes ``(B, S, A, 3)`` and
    ``(B, S, A)``. Targets must already have undergone truth-symmetry matching.
    """

    plddt = plddt_loss(
        output.plddt_logits,
        predicted_coordinates,
        target_coordinates,
        target_mask,
        input,
        token_level_confidence=token_level_confidence,
    )
    pde = pde_loss(
        output.pde_logits,
        predicted_coordinates,
        target_coordinates,
        target_mask,
        input,
    )
    resolved = resolved_loss(
        output.resolved_logits,
        target_mask,
        input,
        token_level_confidence=token_level_confidence,
    )
    if pae_weight == 0.0:
        pae = output.pae_logits.new_zeros(output.pae_logits.shape[:2])
    else:
        pae = pae_loss(
            output.pae_logits,
            predicted_coordinates,
            target_coordinates,
            target_mask,
            input,
        )
    return BoltzGenConfidenceLoss(
        loss=plddt + pde + resolved + pae_weight * pae,
        plddt_loss=plddt,
        pde_loss=pde,
        resolved_loss=resolved,
        pae_loss=pae,
    )


def resolved_loss(
    resolved_logits: Tensor,
    target_mask: Tensor,
    input: BoltzGenInput,
    *,
    token_level_confidence: bool,
) -> Tensor:
    r"""Return unreduced resolved/unresolved cross entropy with shape ``(B, S)``."""

    if token_level_confidence:
        resolved_target = _representative_mask(target_mask, input)
        spatial_mask = input.token_mask[:, None, :]
    else:
        resolved_target = target_mask
        spatial_mask = input.atom_mask[:, None, :]

    resolved_target = resolved_target.to(resolved_logits)
    log_probabilities = F.log_softmax(resolved_logits, dim=-1)
    errors = -resolved_target * log_probabilities[..., 0] - (1.0 - resolved_target) * log_probabilities[..., 1]
    return _spatial_mean(errors, spatial_mask, dim=-1)


def plddt_loss(
    plddt_logits: Tensor,
    predicted_coordinates: Tensor,
    target_coordinates: Tensor,
    target_mask: Tensor,
    input: BoltzGenInput,
    *,
    token_level_confidence: bool,
) -> Tensor:
    r"""Return unreduced categorical pLDDT loss with shape ``(B, S)``."""

    target, spatial_mask = _plddt_target(
        predicted_coordinates,
        target_coordinates,
        target_mask,
        input,
        token_level_confidence=token_level_confidence,
    )
    errors = _categorical_errors(plddt_logits, target, max_value=1.0)
    return _spatial_mean(errors, spatial_mask, dim=-1)


def pde_loss(
    pde_logits: Tensor,
    predicted_coordinates: Tensor,
    target_coordinates: Tensor,
    target_mask: Tensor,
    input: BoltzGenInput,
    *,
    max_distance: float = 32.0,
) -> Tensor:
    r"""Return unreduced categorical predicted-distance-error loss with shape ``(B, S)``."""

    target, pair_mask = _pde_target(
        predicted_coordinates,
        target_coordinates,
        target_mask,
        input,
    )
    errors = _categorical_errors(pde_logits, target, max_value=max_distance)
    return _spatial_mean(errors, pair_mask, dim=(-2, -1))


def pae_loss(
    pae_logits: Tensor,
    predicted_coordinates: Tensor,
    target_coordinates: Tensor,
    target_mask: Tensor,
    input: BoltzGenInput,
    *,
    max_distance: float = 32.0,
) -> Tensor:
    r"""Return unreduced categorical predicted-aligned-error loss with shape ``(B, S)``."""

    target, pair_mask = _pae_target(
        predicted_coordinates,
        target_coordinates,
        target_mask,
        input,
    )
    errors = _categorical_errors(pae_logits, target, max_value=max_distance)
    return _spatial_mean(errors, pair_mask, dim=(-2, -1))


def plddt_regression_loss(
    predicted_plddt: Tensor,
    predicted_coordinates: Tensor,
    target_coordinates: Tensor,
    target_mask: Tensor,
    input: BoltzGenInput,
    *,
    token_level_confidence: bool,
) -> Tensor:
    r"""Return the unreduced Huber pLDDT alternative with shape ``(B, S)``."""

    target, spatial_mask = _plddt_target(
        predicted_coordinates,
        target_coordinates,
        target_mask,
        input,
        token_level_confidence=token_level_confidence,
    )
    errors = F.huber_loss(
        predicted_plddt,
        target.to(predicted_plddt),
        reduction="none",
        delta=0.03,
    )
    return _spatial_mean(errors, spatial_mask, dim=-1)


def pde_regression_loss(
    predicted_pde: Tensor,
    predicted_coordinates: Tensor,
    target_coordinates: Tensor,
    target_mask: Tensor,
    input: BoltzGenInput,
    *,
    max_distance: float = 32.0,
) -> Tensor:
    r"""Return the unreduced Huber PDE alternative with shape ``(B, S)``."""

    target, pair_mask = _pde_target(
        predicted_coordinates,
        target_coordinates,
        target_mask,
        input,
    )
    prediction = predicted_pde / max_distance
    target = target.to(predicted_pde).clamp(max=max_distance) / max_distance
    errors = F.huber_loss(
        prediction,
        target,
        reduction="none",
        delta=0.03,
    )
    return _spatial_mean(errors, pair_mask, dim=(-2, -1))


def pae_regression_loss(
    predicted_pae: Tensor,
    predicted_coordinates: Tensor,
    target_coordinates: Tensor,
    target_mask: Tensor,
    input: BoltzGenInput,
    *,
    max_distance: float = 32.0,
) -> Tensor:
    r"""Return the unreduced Huber TM-weighted PAE alternative with shape ``(B, S)``."""

    target, pair_mask = _pae_target(
        predicted_coordinates,
        target_coordinates,
        target_mask,
        input,
    )
    num_tokens = input.token_mask.to(predicted_pae).sum(dim=-1)
    length_scale = cast(Tensor, tm_length_scale(num_tokens))[:, None, None, None]
    prediction = predicted_pae
    target = target.to(predicted_pae).clamp(max=max_distance)
    prediction_tm = 1.0 / (1.0 + (prediction / length_scale).square())
    target_tm = 1.0 / (1.0 + (target / length_scale).square())
    errors = F.huber_loss(
        prediction_tm,
        target_tm,
        reduction="none",
        delta=0.03,
    )
    return _spatial_mean(errors, pair_mask, dim=(-2, -1))


def distogram_loss(
    logits: Tensor,
    target_coordinates: Tensor,
    target_mask: Tensor,
    *,
    min_distance: float = 2.0,
    max_distance: float = 22.0,
    aggregate_distogram: bool = True,
) -> Tensor:
    r"""
    Return the unreduced BoltzGen distogram loss with shape ``(B,)``.

    ``logits`` has shape ``(B, N, N, D, C)`` and target conformers have
    shape ``(B, K, N, 3)``.
    """

    target_coordinates = target_coordinates.masked_fill(
        ~target_mask[:, None, :, None].bool(),
        0,
    )
    num_bins = logits.shape[-1]
    boundaries = torch.linspace(
        min_distance,
        max_distance,
        num_bins - 1,
        device=target_coordinates.device,
        dtype=target_coordinates.dtype,
    )
    target_distance = torch.cdist(target_coordinates, target_coordinates)
    target_bin = (target_distance.unsqueeze(-1) > boundaries).sum(dim=-1)
    target_histogram = F.one_hot(target_bin, num_classes=num_bins).to(logits)

    num_tokens = target_mask.shape[-1]
    pair_mask = target_mask.bool().unsqueeze(-1) & target_mask.bool().unsqueeze(-2)
    pair_mask = pair_mask & ~torch.eye(
        num_tokens,
        dtype=torch.bool,
        device=target_mask.device,
    ).unsqueeze(0)
    pair_weight = pair_mask.to(logits)
    denominator = pair_weight.sum(dim=(-2, -1)) + 1e-5

    if aggregate_distogram:
        (prediction,) = logits.unbind(dim=3)
        target_probability = target_histogram.sum(dim=1)
        target_probability = target_probability / target_probability.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1)
        error = -(target_probability * F.log_softmax(prediction, dim=-1)).sum(dim=-1)
        return (error * pair_weight).sum(dim=(-2, -1)) / denominator

    log_probability = F.log_softmax(logits, dim=-1)
    error = -(target_histogram.unsqueeze(-2) * log_probability.unsqueeze(1)).sum(dim=-1)
    conformer_prediction_loss = (error * pair_weight[:, None, :, :, None]).sum(dim=(-3, -2)) / denominator[
        :, None, None
    ]
    return conformer_prediction_loss.min(dim=-1).values.mean(dim=-1)


def b_factor_loss(
    logits: Tensor,
    target: Tensor,
    token_mask: Tensor,
) -> Tensor:
    r"""Return the unreduced categorical B-factor loss with shape ``(B,)``."""

    num_bins = logits.shape[-1]
    boundaries = torch.linspace(
        0,
        100,
        num_bins - 1,
        device=target.device,
        dtype=target.dtype,
    )
    target_bin = (target.unsqueeze(-1) > boundaries).sum(dim=-1)
    error = (
        -F.log_softmax(logits, dim=-1)
        .gather(
            -1,
            target_bin.unsqueeze(-1),
        )
        .squeeze(-1)
    )
    weight = (token_mask.bool() & ~torch.isnan(target)).to(error)
    return (error * weight).sum(dim=-1) / (weight.sum(dim=-1) + 1e-5)


def residue_type_loss(
    logits: Tensor,
    target: Tensor,
    mask: Tensor,
) -> Tensor:
    r"""
    Return soft-target residue cross entropy over the explicit token mask.

    The result retains the leading batch and optional sample axes.
    """

    sample_dimensions = logits.ndim - target.ndim
    target = target.reshape(target.shape[:1] + (1,) * sample_dimensions + target.shape[1:]).to(logits)
    mask = mask.reshape(mask.shape[:1] + (1,) * sample_dimensions + mask.shape[1:])
    error = -(target * F.log_softmax(logits, dim=-1)).sum(dim=-1)
    weight = mask.to(error)
    return (error * weight).sum(dim=-1) / (weight.sum(dim=-1) + 1e-5)


def smooth_lddt_loss(
    predicted_coordinates: Tensor,
    target_coordinates: Tensor,
    target_mask: Tensor,
    atom_is_nucleotide: Tensor,
    *,
    nucleic_acid_cutoff: float = 30.0,
    other_cutoff: float = 15.0,
) -> Tensor:
    r"""Return the unreduced smooth atom lDDT loss with shape ``(B, S)``."""

    target_coordinates = target_coordinates.masked_fill(
        ~target_mask[:, None, :, None].bool(),
        0,
    )
    target_distance = torch.cdist(target_coordinates, target_coordinates)
    predicted_distance = torch.cdist(predicted_coordinates, predicted_coordinates)
    distance_error = (target_distance - predicted_distance).abs()

    cutoff = torch.where(
        atom_is_nucleotide.bool(),
        target_distance.new_tensor(nucleic_acid_cutoff),
        target_distance.new_tensor(other_cutoff),
    )[:, None, :, None]
    num_atoms = target_mask.shape[-1]
    resolved = target_mask.bool()[:, None, :]
    pair_mask = resolved.unsqueeze(-1) & resolved.unsqueeze(-2)
    pair_mask = (
        pair_mask
        & ~torch.eye(
            num_atoms,
            dtype=torch.bool,
            device=target_mask.device,
        )[None, None, :, :]
    )
    pair_mask = pair_mask & (target_distance < cutoff)

    score = 0.25 * (
        torch.sigmoid(0.5 - distance_error)
        + torch.sigmoid(1.0 - distance_error)
        + torch.sigmoid(2.0 - distance_error)
        + torch.sigmoid(4.0 - distance_error)
    )
    weight = pair_mask.to(score)
    count = weight.sum(dim=(-2, -1))
    loss = 1.0 - (score * weight).sum(dim=(-2, -1)) / (count + 1e-5)
    return torch.where(count != 0, loss, torch.zeros_like(loss))


def bond_loss(
    predicted_coordinates: Tensor,
    target_coordinates: Tensor,
    bond_atom_index: Tensor,
    bond_mask: Tensor,
) -> Tensor:
    r"""Return the unreduced inter-residue bond-length loss with shape ``(B, S)``."""

    batch_size, multiplicity, _, _ = predicted_coordinates.shape
    num_bonds = bond_atom_index.shape[1]
    gather_index = (
        bond_atom_index[:, None, :, :, None]
        .expand(
            batch_size,
            multiplicity,
            num_bonds,
            2,
            3,
        )
        .long()
    )
    predicted_bond_coordinates = (
        predicted_coordinates[:, :, None]
        .expand(
            -1,
            -1,
            num_bonds,
            -1,
            -1,
        )
        .gather(3, gather_index)
    )
    target_bond_coordinates = (
        target_coordinates[:, :, None]
        .expand(
            -1,
            -1,
            num_bonds,
            -1,
            -1,
        )
        .gather(3, gather_index)
    )
    target_bond_coordinates = target_bond_coordinates.masked_fill(
        ~bond_mask[:, None, :, None, None].bool(),
        0,
    )
    predicted_length = torch.linalg.vector_norm(
        predicted_bond_coordinates[..., 0, :] - predicted_bond_coordinates[..., 1, :],
        dim=-1,
    )
    target_length = torch.linalg.vector_norm(
        target_bond_coordinates[..., 0, :] - target_bond_coordinates[..., 1, :],
        dim=-1,
    )
    weight = bond_mask[:, None, :].to(predicted_length)
    return ((predicted_length - target_length).square() * weight).sum(dim=-1) / (weight.sum(dim=-1) + 1e-5)


def diffusion_loss(
    output: BoltzGenOutput,
    input: BoltzGenInput,
    target: BoltzGenStructureTarget,
    *,
    sigma_data: float = 16.0,
    rotational_alignment: bool = True,
    add_smooth_lddt_loss: bool = True,
    add_bond_loss: bool = False,
    nucleotide_loss_weight: float = 5.0,
    ligand_loss_weight: float = 10.0,
    fake_atom_weight: float = 1.0,
    residue_type_weight: float = 0.0,
) -> BoltzGenDiffusionLoss:
    r"""Compose the unreduced BoltzGen coordinate objectives."""

    atom_to_token = input.atom_to_token.long()
    atom_molecule_type = input.molecule_type.gather(1, atom_to_token)
    atom_is_nucleotide = (atom_molecule_type == 1) | (atom_molecule_type == 2)
    atom_is_ligand = atom_molecule_type == 3
    type_weight = (
        1.0
        + nucleotide_loss_weight * atom_is_nucleotide.to(output.coordinates)
        + ligand_loss_weight * atom_is_ligand.to(output.coordinates)
    )
    target_atom_index = target.atom_index.long()
    valid_target_atom = target_atom_index >= 0
    resolved = target.coordinate_mask.gather(
        1,
        target_atom_index.clamp_min(0),
    ).bool()
    resolved = resolved & valid_target_atom & input.atom_mask.bool()
    alignment_weight = type_weight * resolved.to(type_weight)

    augmented_target_coordinates = output.augmented_target_coordinates.masked_fill(
        ~resolved[:, None, :, None],
        0,
    )
    if rotational_alignment:
        aligned_target = weighted_rigid_align(
            source=augmented_target_coordinates,
            target=output.coordinates,
            weight=alignment_weight[:, None, :],
        )
    else:
        aligned_target = _weighted_center(
            augmented_target_coordinates,
            output.coordinates,
            alignment_weight[:, None, :],
        )

    fake_weight = (
        1.0
        - target.fake_atom_mask.to(output.coordinates)
        + target.fake_atom_mask.to(output.coordinates) * fake_atom_weight
    )
    residue_class_weight = output.coordinates.new_tensor(_RESIDUE_TYPE_WEIGHTS)
    token_residue_weight = target.residue_type.to(output.coordinates) @ residue_class_weight
    atom_residue_weight = token_residue_weight.gather(1, atom_to_token)
    atom_design_mask = cast(Tensor, input.design_mask).bool().gather(1, atom_to_token)
    atom_residue_weight = torch.where(
        atom_design_mask,
        atom_residue_weight,
        torch.ones_like(atom_residue_weight),
    ).pow(residue_type_weight)

    mse_weight = type_weight * fake_weight * atom_residue_weight * resolved.to(type_weight)
    squared_error = (output.coordinates - aligned_target).square().sum(dim=-1)
    mse = (squared_error * mse_weight[:, None, :]).sum(dim=-1) / (3.0 * mse_weight.sum(dim=-1, keepdim=True) + 1e-5)
    sigma = output.noise_level
    edm_weight = (sigma.square() + sigma_data**2) / (sigma * sigma_data).square()
    mse = mse * edm_weight

    if add_smooth_lddt_loss:
        smooth_lddt = smooth_lddt_loss(
            output.coordinates,
            aligned_target,
            resolved,
            atom_is_nucleotide,
        )
    else:
        smooth_lddt = torch.zeros_like(mse)

    if add_bond_loss and target.bond_atom_index is not None and target.bond_mask is not None:
        bonds = bond_loss(
            output.coordinates,
            aligned_target,
            target.bond_atom_index,
            target.bond_mask,
        )
    else:
        bonds = torch.zeros_like(mse)

    return BoltzGenDiffusionLoss(
        loss=mse + smooth_lddt + bonds,
        mse_loss=mse,
        smooth_lddt_loss=smooth_lddt,
        bond_loss=bonds,
    )


def _weighted_center(
    source: Tensor,
    target: Tensor,
    weight: Tensor,
) -> Tensor:
    weight = weight.unsqueeze(-1)
    denominator = weight.sum(dim=-2, keepdim=True) + 1e-8
    source_center = (source * weight).sum(dim=-2, keepdim=True) / denominator
    target_center = (target * weight).sum(dim=-2, keepdim=True) / denominator
    return (source - source_center + target_center).detach()


def _representative_coordinates(coordinates: Tensor, input: BoltzGenInput) -> Tensor:
    multiplicity = coordinates.shape[1]
    index = (
        input.token_center_atom_index[:, None, :, None]
        .expand(
            -1,
            multiplicity,
            -1,
            3,
        )
        .long()
    )
    return coordinates.gather(2, index)


def _representative_mask(target_mask: Tensor, input: BoltzGenInput) -> Tensor:
    multiplicity = target_mask.shape[1]
    index = (
        input.token_center_atom_index[:, None, :]
        .expand(
            -1,
            multiplicity,
            -1,
        )
        .long()
    )
    return target_mask.gather(2, index)


def _plddt_target(
    predicted_coordinates: Tensor,
    target_coordinates: Tensor,
    target_mask: Tensor,
    input: BoltzGenInput,
    *,
    token_level_confidence: bool,
) -> tuple[Tensor, Tensor]:
    target_coordinates = target_coordinates.masked_fill(
        ~target_mask[..., None].bool(),
        0,
    )
    predicted_representatives = _representative_coordinates(predicted_coordinates, input)
    target_representatives = _representative_coordinates(target_coordinates, input)
    representative_mask = _representative_mask(target_mask, input).bool()
    token_mask = input.token_mask[:, None, :].bool()
    polymer_reference_mask = (input.molecule_type != 3)[:, None, :]
    reference_mask = representative_mask & token_mask & polymer_reference_mask

    if token_level_confidence:
        predicted_query = predicted_representatives
        target_query = target_representatives
        query_mask = representative_mask & token_mask
        query_atom_index = input.token_center_atom_index[:, None, :].expand(
            -1,
            target_mask.shape[1],
            -1,
        )
    else:
        predicted_query = predicted_coordinates
        target_query = target_coordinates
        query_mask = target_mask.bool() & input.atom_mask[:, None, :].bool()
        query_atom_index = torch.arange(
            target_mask.shape[-1],
            device=target_mask.device,
        )[
            None, None, :
        ].expand_as(target_mask)

    true_distance = torch.cdist(target_query, target_representatives)
    predicted_distance = torch.cdist(predicted_query, predicted_representatives)
    reference_atom_index = input.token_center_atom_index[:, None, None, :]
    different_atom = query_atom_index.unsqueeze(-1) != reference_atom_index
    pair_mask = query_mask.unsqueeze(-1) & reference_mask.unsqueeze(-2) & different_atom
    nucleotide_reference = ((input.molecule_type == 1) | (input.molecule_type == 2)).to(true_distance)
    cutoff = 15.0 + 15.0 * nucleotide_reference[:, None, None, :]
    distance_mask = pair_mask & (true_distance < cutoff)
    distance_error = (true_distance - predicted_distance).abs()
    score = 0.25 * (
        (distance_error < 0.5).to(distance_error)
        + (distance_error < 1.0).to(distance_error)
        + (distance_error < 2.0).to(distance_error)
        + (distance_error < 4.0).to(distance_error)
    )
    weight = distance_mask.to(score)
    count = weight.sum(dim=-1)
    target = (1e-10 + (score * weight).sum(dim=-1)) / (1e-10 + count)
    return target, query_mask & (count != 0)


def _pde_target(
    predicted_coordinates: Tensor,
    target_coordinates: Tensor,
    target_mask: Tensor,
    input: BoltzGenInput,
) -> tuple[Tensor, Tensor]:
    target_coordinates = target_coordinates.masked_fill(
        ~target_mask[..., None].bool(),
        0,
    )
    predicted_representatives = _representative_coordinates(predicted_coordinates, input)
    target_representatives = _representative_coordinates(target_coordinates, input)
    representative_mask = _representative_mask(target_mask, input).bool()
    representative_mask = representative_mask & input.token_mask[:, None, :].bool()
    target_distance = torch.cdist(target_representatives, target_representatives)
    predicted_distance = torch.cdist(predicted_representatives, predicted_representatives)
    pair_mask = representative_mask.unsqueeze(-1) & representative_mask.unsqueeze(-2)
    return (target_distance - predicted_distance).abs(), pair_mask


def _pae_target(
    predicted_coordinates: Tensor,
    target_coordinates: Tensor,
    target_mask: Tensor,
    input: BoltzGenInput,
) -> tuple[Tensor, Tensor]:
    target_coordinates = target_coordinates.masked_fill(
        ~target_mask[..., None].bool(),
        0,
    )
    batch_size, multiplicity, num_atoms, _ = predicted_coordinates.shape
    predicted_frame_index, predicted_frame_mask = _boltzgen_coordinate_frames(
        predicted_coordinates.reshape(batch_size * multiplicity, num_atoms, 3),
        input=input,
        multiplicity=multiplicity,
    )
    target_frame_index, target_frame_mask = _boltzgen_coordinate_frames(
        target_coordinates.reshape(batch_size * multiplicity, num_atoms, 3),
        input=input,
        multiplicity=multiplicity,
        atom_mask=target_mask.reshape(batch_size * multiplicity, num_atoms),
    )
    num_tokens = input.token_mask.shape[-1]
    predicted_frame_index = predicted_frame_index.reshape(batch_size, multiplicity, num_tokens, 3)
    target_frame_index = target_frame_index.reshape(batch_size, multiplicity, num_tokens, 3)
    predicted_frame_mask = predicted_frame_mask.reshape(batch_size, multiplicity, num_tokens)
    target_frame_mask = target_frame_mask.reshape(batch_size, multiplicity, num_tokens)
    predicted_in_frame = _express_token_centers_in_frames(
        predicted_coordinates,
        predicted_frame_index,
    )
    target_in_frame = _express_token_centers_in_frames(
        target_coordinates,
        target_frame_index,
    )
    target_pae = ((target_in_frame - predicted_in_frame).square().sum(dim=-1) + 1e-8).sqrt()

    target_center_mask = target_mask.gather(2, target_frame_index[..., 1])
    token_mask = input.token_mask[:, None, :].bool()
    pair_mask = (
        target_frame_mask.unsqueeze(-1)
        & predicted_frame_mask.unsqueeze(-1)
        & target_center_mask.bool().unsqueeze(-2)
        & token_mask.unsqueeze(-1)
        & token_mask.unsqueeze(-2)
    )
    return target_pae, pair_mask


def _express_token_centers_in_frames(
    coordinates: Tensor,
    frame_index: Tensor,
) -> Tensor:
    batch_size, multiplicity, num_tokens, _ = frame_index.shape
    points = coordinates.gather(
        2,
        frame_index.reshape(batch_size, multiplicity, num_tokens * 3).unsqueeze(-1).expand(-1, -1, -1, 3),
    ).reshape(batch_size, multiplicity, num_tokens, 3, 3)
    first, center, third = points.unbind(dim=-2)
    first_direction = (first - center) / (torch.linalg.norm(first - center, dim=-1, keepdim=True) + 1e-5)
    third_direction = (third - center) / (torch.linalg.norm(third - center, dim=-1, keepdim=True) + 1e-5)
    axis_one = (first_direction + third_direction) / (
        torch.linalg.norm(first_direction + third_direction, dim=-1, keepdim=True) + 1e-5
    )
    axis_two = (third_direction - first_direction) / (
        torch.linalg.norm(third_direction - first_direction, dim=-1, keepdim=True) + 1e-5
    )
    axis_three = torch.linalg.cross(axis_one, axis_two, dim=-1)
    displacement = center.unsqueeze(-3) - center.unsqueeze(-2)
    return torch.stack(
        (
            (displacement * axis_one.unsqueeze(-2)).sum(dim=-1),
            (displacement * axis_two.unsqueeze(-2)).sum(dim=-1),
            (displacement * axis_three.unsqueeze(-2)).sum(dim=-1),
        ),
        dim=-1,
    )


def _categorical_errors(
    logits: Tensor,
    target: Tensor,
    *,
    max_value: float,
) -> Tensor:
    num_bins = logits.shape[-1]
    target_bin = torch.floor(target * num_bins / max_value).long().clamp(max=num_bins - 1)
    return -F.log_softmax(logits, dim=-1).gather(-1, target_bin.unsqueeze(-1)).squeeze(-1)


def _spatial_mean(
    errors: Tensor,
    mask: Tensor,
    *,
    dim: int | tuple[int, ...],
) -> Tensor:
    weight = mask.to(errors)
    return (errors * weight).sum(dim=dim) / (weight.sum(dim=dim) + 1e-7)
