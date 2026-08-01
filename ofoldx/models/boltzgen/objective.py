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
BoltzGen training objectives.

OFoldX exposes the shared EDM coordinate-denoising objective as BoltzGen's built-in training contract.
``diffusion_loss`` here is BoltzGen's own fuller published objective (rigid-aligned, type-weighted MSE
plus smooth lDDT and bond terms), kept as a family API for callers who want it.
"""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ...data.tokenization import TOKENS
from ...errors import RequestRef, fail
from ...metrics import tm_length_scale
from ...metrics.structure import lddt_from_distances
from ...modules.atom.coordinates import gather_observed_coordinates
from ...modules.diffusion.edm import edm_loss_weight
from ...modules.diffusion.loss import bond_loss, smooth_lddt_loss_with_counts
from ...modules.functional import weighted_mean
from ...modules.geometry import weighted_rigid_align
from ..processing import project_model_coordinates, project_model_features
from .model import (
    BoltzGenConfidenceOutput,
    BoltzGenInput,
    BoltzGenInverseFoldingInput,
    BoltzGenInverseFoldingOutput,
    BoltzGenOutput,
    BoltzGenStructureTarget,
    boltzgen_coordinate_frames,
)

#: The released per-residue coordinate weights, by canonical class; every class the recipe does not name is 1.
_RESIDUE_TYPE_WEIGHTS = {
    "D": 2.0,
    "F": 2.0,
    "H": 2.0,
    "I": 4.0,
    "K": 2.0,
    "L": 2.0,
    "M": 2.0,
    "P": 4.0,
    "Q": 2.0,
    "S": 4.0,
    "T": 4.0,
    "V": 4.0,
    "W": 4.0,
    "Y": 4.0,
    "X": 10.0,
}


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
    target: BoltzGenStructureTarget,
    input: BoltzGenInput,
    *,
    token_level_confidence: bool,
    pae_weight: float = 0.0,
    max_chain_assignments: int | None = 100,
    generator: torch.Generator | None = None,
) -> BoltzGenConfidenceLoss:
    r"""
    Compose the released BoltzGen categorical confidence objectives.

    Prediction-dependent chain and local-atom truth assignments are selected
    independently for every sample before the confidence targets are computed.
    """

    if target.coordinates.shape[1] != 1:
        fail("value_invalid", RequestRef("target.coordinates", str(target.coordinates.shape[1])))
    target_coordinates, target_mask = match_symmetric_truth(
        predicted_coordinates,
        target.coordinates[:, 0],
        target.coordinate_mask,
        target.coordinate_atom_index,
        target.chain_span,
        target.chain_equivalence_index,
        target.symmetry_atom_index,
        target.symmetry_permutation,
        max_chain_assignments=max_chain_assignments,
        generator=generator,
    )
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

    return _categorical_loss(resolved_logits, resolved_target.long(), spatial_mask)


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
    target = target.masked_fill(~spatial_mask, 0)
    num_bins = plddt_logits.shape[-1]
    target_bin = torch.floor(target * num_bins).long().clamp(max=num_bins - 1)
    return _categorical_loss(plddt_logits, target_bin, spatial_mask)


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
    target = target.masked_fill(~pair_mask, 0)
    num_bins = pde_logits.shape[-1]
    target_bin = torch.floor(target * num_bins / max_distance).long().clamp(max=num_bins - 1)
    return _categorical_loss(pde_logits, target_bin, pair_mask)


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
    target = target.masked_fill(~pair_mask, 0)
    num_bins = pae_logits.shape[-1]
    target_bin = torch.floor(target * num_bins / max_distance).long().clamp(max=num_bins - 1)
    return _categorical_loss(pae_logits, target_bin, pair_mask)


def plddt_regression_loss(
    predicted_plddt: Tensor,
    predicted_coordinates: Tensor,
    target_coordinates: Tensor,
    target_mask: Tensor,
    input: BoltzGenInput,
    *,
    token_level_confidence: bool,
) -> Tensor:
    r"""
    Return the unreduced Huber pLDDT alternative with shape ``(B, S)``.

    ``predicted_plddt`` is a fraction in ``[0, 1]``, which is the space the lDDT target
    is computed in and what the ``delta`` here is scaled for. The family reports pLDDT on
    the ``0-100`` scale, so a prediction taken from a reported score must be divided by
    one hundred before it reaches this loss.
    """

    target, spatial_mask = _plddt_target(
        predicted_coordinates,
        target_coordinates,
        target_mask,
        input,
        token_level_confidence=token_level_confidence,
    )
    errors = F.huber_loss(
        predicted_plddt.masked_fill(~spatial_mask, 0),
        target.to(predicted_plddt).masked_fill(~spatial_mask, 0),
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
    prediction = predicted_pde.masked_fill(~pair_mask, 0) / max_distance
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
    prediction = predicted_pae.masked_fill(~pair_mask, 0)
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

    target_coordinates = project_model_coordinates(
        target_coordinates,
        target_mask.bool()[:, None, :].expand(target_coordinates.shape[:-1]),
        feature="BoltzGen distogram truth",
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
    logits = logits.masked_fill(~pair_mask[..., None, None], 0)
    pair_weight = pair_mask.to(logits)
    # An empty scope reports NaN, but the NaN is selected in rather than divided into: dividing by a
    # safe count and applying ``torch.where`` afterwards keeps a 0/0 out of the autograd graph, which
    # would otherwise backpropagate NaN into every parameter regardless of how the caller reduces.
    scored_pairs = pair_weight.sum(dim=(-2, -1))
    denominator = scored_pairs.clamp_min(1)

    if aggregate_distogram:
        (prediction,) = logits.unbind(dim=3)
        target_probability = target_histogram.sum(dim=1)
        target_probability = target_probability / target_probability.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1)
        error = -(target_probability * F.log_softmax(prediction, dim=-1)).sum(dim=-1)
        aggregate = (error * pair_weight).sum(dim=(-2, -1)) / denominator
        return torch.where(scored_pairs > 0, aggregate, aggregate.new_full((), float("nan")))

    log_probability = F.log_softmax(logits, dim=-1)
    error = -(target_histogram.unsqueeze(-2) * log_probability.unsqueeze(1)).sum(dim=-1)
    conformer_prediction_loss = (error * pair_weight[:, None, :, :, None]).sum(dim=(-3, -2)) / denominator[
        :, None, None
    ]
    conformer_prediction_loss = torch.where(
        (scored_pairs > 0)[:, None, None],
        conformer_prediction_loss,
        conformer_prediction_loss.new_full((), float("nan")),
    )
    return conformer_prediction_loss.min(dim=-1).values.mean(dim=-1)


def b_factor_loss(
    logits: Tensor,
    target: Tensor,
    b_factor_mask: Tensor,
    token_mask: Tensor,
) -> Tensor:
    r"""
    Return the unreduced categorical B-factor loss with shape ``(B,)``.

    BoltzGen's B-factor head is a linear read of the trunk single representation, so
    ``logits`` are per token with shape ``(B, N, D)``. ``target`` carries the deposited
    B factor of each token's representative atom with shape ``(B, N)``, which is what
    [`BoltzGenProcessor`][ofoldx.models.boltzgen.processor.BoltzGenProcessor] prepares as
    [`BoltzGenStructureTarget`][ofoldx.models.boltzgen.model.BoltzGenStructureTarget]. A
    token whose representative atom has no deposited B factor is unknown rather than zero,
    and ``b_factor_mask`` keeps it out of the average.

    Args:
        logits: Per-token B-factor logits with shape ``(B, N, D)``.
        target: Per-token representative-atom B factors with shape ``(B, N)``.
        b_factor_mask: Deposited-B-factor mask with shape ``(B, N)``.
        token_mask: Token validity mask with shape ``(B, N)``.

    Examples:
        >>> import torch
        >>> from ofoldx.models.boltzgen import b_factor_loss
        >>> logits = torch.tensor([[[0.0, 0.0, 0.0, 0.0], [4.0, 0.0, 0.0, 0.0]]])
        >>> target = torch.tensor([[10.0, 60.0]])
        >>> token_mask = torch.ones(1, 2, dtype=torch.bool)
        >>> measured = torch.ones(1, 2, dtype=torch.bool)
        >>> b_factor_loss(logits, target, measured, token_mask).round(decimals=4)
        tensor([2.7199])
        >>> measured[0, 1] = False
        >>> b_factor_loss(logits, target, measured, token_mask).round(decimals=4)
        tensor([1.3863])
        >>> b_factor_loss(logits, target, torch.zeros_like(measured), token_mask)
        tensor([nan])
    """

    measured = token_mask.bool() & b_factor_mask.bool()
    target = project_model_features(target, measured, feature="BoltzGen B-factor truth")
    logits = logits.masked_fill(~measured[..., None], 0)
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
    weight = (token_mask.bool() & b_factor_mask.bool()).to(error)
    scored = weight.sum(dim=-1)
    value = (error * weight).sum(dim=-1) / scored.clamp_min(1)
    return torch.where(scored > 0, value, value.new_full((), float("nan")))


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
    target = target.masked_fill(~mask.bool()[..., None], 0)
    logits = logits.masked_fill(~mask.bool()[..., None], 0)
    error = -(target * F.log_softmax(logits, dim=-1)).sum(dim=-1)
    weight = mask.to(error)
    scored = weight.sum(dim=-1)
    value = (error * weight).sum(dim=-1) / scored.clamp_min(1)
    return torch.where(scored > 0, value, value.new_full((), float("nan")))


class BoltzGenLoss(nn.Module):
    r"""
    Reduce the released residue-type reconstruction loss to a scalar over the designable positions.

    Scores exactly the redesigned tokens that are structurally valid: present (``input.token_mask``),
    coordinate-resolved (``input.token_resolved_mask``), and selected for design
    (``input.design_mask``). Resolution is an input-side property here because inverse folding
    conditions on a fixed structure; it is unrelated to any target-side coordinate resolution.
    """

    def forward(
        self,
        pred: BoltzGenInverseFoldingOutput,
        input: BoltzGenInverseFoldingInput,
        target: Tensor,
    ) -> Tensor:
        r"""
        Args:
            pred: Predicted residue-type logits.
            input: The batched inverse-folding input carrying the validity and design masks.
            target: Unmasked residue distributions with shape ``(B, N, V)``.
        """

        design_mask = input.token_mask.bool() & input.token_resolved_mask.bool() & input.design_mask.bool()
        return residue_type_loss(pred.logits, target, design_mask).mean()


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
    smooth_lddt_sample_chunk_size: int = 1,
    smooth_lddt_pair_chunk_size: int = 1048576,
) -> BoltzGenDiffusionLoss:
    r"""
    Compose the BoltzGen coordinate objectives for each structure and sample.

    Align detached augmented truth to predictions for type-weighted coordinate
    MSE, normalized by three times the scored atom weight and weighted by EDM
    noise level. Smooth lDDT and optional inter-component bond error use the
    augmented truth before alignment: these distances are rigid invariant, and
    their target neighborhoods must not depend on alignment roundoff.

    Resolution is gathered through ``target.coordinate_atom_index`` and
    intersected with ``input.atom_mask``. Synthetic model rows with coordinate
    truth remain eligible. All resolved rows, including fixed context, enter
    alignment and geometry; ``input.design_mask`` selects residue-class MSE
    reweighting only. Alignment uses type weights, while MSE additionally uses
    fake-atom and residue-class weights. Unresolved coordinates may contain NaNs
    and are masked locally before geometry without changing the supplied tensors.

    Args:
        output: Denoised coordinates ``(B, S, A, 3)``, augmented truth
            ``(B, 1, A, 3)`` or ``(B, S, A, 3)`` in angstroms, and positive
            noise levels ``(B, S)``. Coordinate truth is detached in this objective.
        input: Molecular features, masks and model-row bond endpoints for the
            same atom and token axes as ``output``.
        target: Complete-assembly resolution and model-row coordinate-source
            mapping, plus unmasked residue distributions on the input token axis.
        sigma_data: Positive EDM data scale used by the producing model.
        rotational_alignment: Fit a proper rotation and translation when true;
            otherwise fit translation only. Alignment targets are detached.
        add_smooth_lddt_loss: Include smooth lDDT with directed 30/15 angstrom
            nucleotide/other cutoffs and unit coefficient.
        add_bond_loss: Include mean squared inter-component bond-length error
            with unit coefficient, scoring bonds whose endpoints are resolved.
        nucleotide_loss_weight: Additive nucleotide type weight; 5 gives weight 6.
        ligand_loss_weight: Additive ligand type weight; 10 gives weight 11.
        fake_atom_weight: Nonnegative MSE multiplier for synthetic atom14 rows.
        residue_type_weight: Exponent of the residue-class MSE weight on designed
            rows; zero disables this reweighting. Released recipe weights follow
            canonical residue identities; recipe-undefined B/Z/J/U/O use neutral
            weight 1.
        smooth_lddt_sample_chunk_size: Positive sample-block size forwarded to
            smooth lDDT when enabled; independent of denoiser chunking.
        smooth_lddt_pair_chunk_size: Positive pair budget forwarded to smooth
            lDDT when enabled. It bounds rectangular row-by-column positions
            before resolved-atom and distance-cutoff filtering, excluding batch
            and sample multipliers. Pair storage also scales with the batch size
            and the samples in each block. Changing either chunk size preserves
            the objective weights, cutoffs and diffusion-sample count.

    Returns:
        Components and their sum, each ``(B, S)``. Enabled components report NaN
        for empty scopes; disabled components report zero. Only explicitly empty
        or disabled components contribute zero to the sum, so an entirely empty
        example has zero total loss and zero coordinate gradients. Nonfinite
        applicable losses propagate.

    Raises:
        ValueError: From ``ofoldx.modules`` if enabled smooth-lDDT chunk sizes or coordinate shapes are
            invalid, or if enabled bond truth has an incompatible shape.
    """

    dtype = torch.promote_types(output.coordinates.dtype, output.augmented_target_coordinates.dtype)
    dtype = torch.promote_types(dtype, output.noise_level.dtype)
    predicted_coordinates = output.coordinates.to(dtype)

    atom_to_token = input.atom_to_token.long()
    atom_molecule_type = input.molecule_type.gather(1, atom_to_token)
    atom_is_nucleotide = (atom_molecule_type == 1) | (atom_molecule_type == 2)
    atom_is_ligand = atom_molecule_type == 3
    type_weight = (
        1.0
        + nucleotide_loss_weight * atom_is_nucleotide.to(predicted_coordinates)
        + ligand_loss_weight * atom_is_ligand.to(predicted_coordinates)
    )
    _, resolved = gather_observed_coordinates(
        target.coordinates[:, 0],
        target.coordinate_mask.bool(),
        target.coordinate_atom_index,
        input.atom_mask.bool(),
    )
    alignment_weight = type_weight * resolved.to(type_weight)

    predicted_coordinates = project_model_coordinates(
        predicted_coordinates,
        resolved[:, None, :].expand(predicted_coordinates.shape[:-1]),
        feature="BoltzGen diffusion prediction",
    )
    augmented_target_coordinates = project_model_coordinates(
        output.augmented_target_coordinates.detach(),
        resolved[:, None, :].expand(output.augmented_target_coordinates.shape[:-1]),
        dtype=dtype,
        feature="BoltzGen diffusion truth",
    )
    if rotational_alignment:
        aligned_target = weighted_rigid_align(
            source=augmented_target_coordinates,
            target=predicted_coordinates,
            weight=alignment_weight[:, None, :],
        )
    else:
        aligned_target = _weighted_center(
            augmented_target_coordinates,
            predicted_coordinates,
            alignment_weight[:, None, :],
        )

    synthetic_atom = input.atom_mask.bool() & (input.atom_index < 0) & (target.coordinate_atom_index >= 0)
    fake_weight = (
        1.0 - synthetic_atom.to(predicted_coordinates) + synthetic_atom.to(predicted_coordinates) * fake_atom_weight
    )
    residue_class_weight = predicted_coordinates.new_tensor([_RESIDUE_TYPE_WEIGHTS.get(token, 1.0) for token in TOKENS])
    token_residue_weight = target.residue_type.to(predicted_coordinates) @ residue_class_weight
    atom_residue_weight = token_residue_weight.gather(1, atom_to_token)
    atom_design_mask = cast(Tensor, input.design_mask).bool().gather(1, atom_to_token)
    atom_residue_weight = torch.where(
        atom_design_mask,
        atom_residue_weight,
        torch.ones_like(atom_residue_weight),
    ).pow(residue_type_weight)

    mse_weight = type_weight * fake_weight * atom_residue_weight * resolved.to(type_weight)
    mse_selected = mse_weight > 0
    mse_coordinates = predicted_coordinates.masked_fill(~mse_selected[:, None, :, None], 0)
    aligned_target = aligned_target.masked_fill(~mse_selected[:, None, :, None], 0)
    squared_error = (mse_coordinates - aligned_target).square().sum(dim=-1)
    has_mse_atoms = mse_selected.any(dim=-1, keepdim=True)
    mse = weighted_mean(squared_error, mse_weight[:, None, :]) / 3.0
    mse = torch.where(has_mse_atoms, mse, 0.0)
    sigma = output.noise_level.to(dtype).masked_fill(~has_mse_atoms, 1)
    edm_weight = edm_loss_weight(sigma, sigma_data)
    mse = mse * edm_weight
    mse = torch.where(has_mse_atoms, mse, mse.new_full((), float("nan")))
    loss = torch.where(has_mse_atoms, mse, 0)

    if add_smooth_lddt_loss:
        smooth_lddt, scored_pairs = smooth_lddt_loss_with_counts(
            predicted_coordinates,
            augmented_target_coordinates,
            resolved,
            atom_is_nucleotide,
            sample_chunk_size=smooth_lddt_sample_chunk_size,
            pair_chunk_size=smooth_lddt_pair_chunk_size,
        )
        loss = loss + torch.where(scored_pairs > 0, smooth_lddt, 0)
    else:
        smooth_lddt = torch.zeros_like(mse)

    if add_bond_loss:
        bond_atom_index = input.bond_atom_index.long()
        valid_bond_atom = bond_atom_index >= 0
        safe_bond_atom_index = bond_atom_index.clamp_min(0)
        bond_mask = (
            input.bond_mask.bool()
            & valid_bond_atom.all(dim=-1)
            & resolved.gather(1, safe_bond_atom_index[..., 0])
            & resolved.gather(1, safe_bond_atom_index[..., 1])
        )
        bonds = bond_loss(
            predicted_coordinates,
            augmented_target_coordinates,
            safe_bond_atom_index,
            bond_mask,
        )
        loss = loss + torch.where(bond_mask.any(dim=-1, keepdim=True), bonds, 0)
    else:
        bonds = torch.zeros_like(mse)

    return BoltzGenDiffusionLoss(
        loss=loss,
        mse_loss=mse,
        smooth_lddt_loss=smooth_lddt,
        bond_loss=bonds,
    )


def _weighted_center(
    source: Tensor,
    target: Tensor,
    weight: Tensor,
) -> Tensor:
    with torch.no_grad():
        weight = weight.unsqueeze(-2)
        source_center = weighted_mean(source.transpose(-2, -1), weight).unsqueeze(-2)
        target_center = weighted_mean(target.transpose(-2, -1), weight).unsqueeze(-2)
        return source - source_center + target_center


def _representative_coordinates(coordinates: Tensor, input: BoltzGenInput) -> Tensor:
    samples = coordinates.shape[1]
    index = input.token_center_atom_index[:, None, :].expand(-1, samples, -1)
    source = input.atom_mask.bool()[:, None, :].expand(coordinates.shape[:-1])
    tokens = input.token_mask.bool()[:, None, :].expand_as(index)
    return gather_observed_coordinates(coordinates, source, index, tokens)[0]


def _representative_mask(target_mask: Tensor, input: BoltzGenInput) -> Tensor:
    # Coordinate gathering validates this same map; -1 indexes the explicit
    # unavailable sentinel, including an empty source atom axis.
    index = (input.token_center_atom_index[:, None, :] + 1).expand(-1, target_mask.shape[1], -1).long()
    return F.pad(target_mask.bool(), (1, 0), value=False).gather(2, index) & input.token_mask.bool()[:, None, :]


def _plddt_target(
    predicted_coordinates: Tensor,
    target_coordinates: Tensor,
    target_mask: Tensor,
    input: BoltzGenInput,
    *,
    token_level_confidence: bool,
) -> tuple[Tensor, Tensor]:
    scored = target_mask.bool() & input.atom_mask.bool()[:, None, :]
    target_coordinates = project_model_coordinates(
        target_coordinates, scored.expand(target_coordinates.shape[:-1]), feature="BoltzGen confidence truth"
    )
    predicted_coordinates = project_model_coordinates(
        predicted_coordinates, scored.expand(predicted_coordinates.shape[:-1]), feature="BoltzGen scored prediction"
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
    target = lddt_from_distances(predicted_distance, true_distance, distance_mask, dim=-1)
    observed = distance_mask.any(dim=-1)
    target = torch.where(observed, target, 1.0)
    return target, query_mask & observed


def _pde_target(
    predicted_coordinates: Tensor,
    target_coordinates: Tensor,
    target_mask: Tensor,
    input: BoltzGenInput,
) -> tuple[Tensor, Tensor]:
    scored = target_mask.bool() & input.atom_mask.bool()[:, None, :]
    target_coordinates = project_model_coordinates(
        target_coordinates, scored.expand(target_coordinates.shape[:-1]), feature="BoltzGen confidence truth"
    )
    predicted_coordinates = project_model_coordinates(
        predicted_coordinates, scored.expand(predicted_coordinates.shape[:-1]), feature="BoltzGen scored prediction"
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
    scored = target_mask.bool() & input.atom_mask.bool()[:, None, :]
    target_coordinates = project_model_coordinates(
        target_coordinates, scored.expand(target_coordinates.shape[:-1]), feature="BoltzGen confidence truth"
    )
    predicted_coordinates = project_model_coordinates(
        predicted_coordinates, scored.expand(predicted_coordinates.shape[:-1]), feature="BoltzGen scored prediction"
    )
    batch_size, num_samples, num_atoms, _ = predicted_coordinates.shape
    predicted_frame_index, predicted_frame_mask = boltzgen_coordinate_frames(
        predicted_coordinates.reshape(batch_size * num_samples, num_atoms, 3),
        input=input,
        num_samples=num_samples,
    )
    target_frame_index, target_frame_mask = boltzgen_coordinate_frames(
        target_coordinates.reshape(batch_size * num_samples, num_atoms, 3),
        input=input,
        num_samples=num_samples,
        atom_mask=target_mask.reshape(batch_size * num_samples, num_atoms),
    )
    num_tokens = input.token_mask.shape[-1]
    predicted_frame_index = predicted_frame_index.reshape(batch_size, num_samples, num_tokens, 3)
    target_frame_index = target_frame_index.reshape(batch_size, num_samples, num_tokens, 3)
    predicted_frame_mask = predicted_frame_mask.reshape(batch_size, num_samples, num_tokens)
    target_frame_mask = target_frame_mask.reshape(batch_size, num_samples, num_tokens)
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
    batch_size, num_samples, num_tokens, _ = frame_index.shape
    points = coordinates.gather(
        2,
        frame_index.reshape(batch_size, num_samples, num_tokens * 3).unsqueeze(-1).expand(-1, -1, -1, 3),
    ).reshape(batch_size, num_samples, num_tokens, 3, 3)
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


def _categorical_loss(
    logits: Tensor,
    labels: Tensor,
    mask: Tensor,
) -> Tensor:
    mask = mask.bool().expand(logits.shape[:-1])
    labels = labels.expand(logits.shape[:-1])
    logits = logits.masked_fill(~mask.unsqueeze(-1), 0)
    errors = -F.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    spatial_mask = mask.flatten(start_dim=2)
    value = weighted_mean(errors.flatten(start_dim=2), spatial_mask)
    return torch.where(spatial_mask.any(dim=-1), value, 0.0)


def _spatial_mean(
    errors: Tensor,
    mask: Tensor,
    *,
    dim: int | tuple[int, ...],
) -> Tensor:
    errors = errors.masked_fill(~mask.bool(), 0)
    weight = mask.to(errors)
    scored = weight.sum(dim=dim)
    value = (errors * weight).sum(dim=dim) / scored.clamp_min(1)
    return torch.where(scored > 0, value, value.new_full((), float("nan")))


@torch.no_grad()
def match_symmetric_truth(
    predicted_coordinates: Tensor,
    target_coordinates: Tensor,
    target_mask: Tensor,
    coordinate_atom_index: Tensor,
    target_chain_span: Tensor,
    target_chain_equivalence_index: Tensor,
    target_symmetry_atom_index: Tensor,
    target_symmetry_permutation: Tensor,
    *,
    max_chain_assignments: int | None = 100,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    r"""
    Select chain and local-atom truth assignments by the released staged lDDT rule.

    Chain assignments are injective within each equivalence class. Complete
    enumeration returns the global optimum of the chain stage; a finite budget
    keeps identity and samples the remaining assignment ordinals uniformly
    without replacement. Local atom groups are then optimized greedily in their
    carried order.
    """

    if max_chain_assignments is not None:
        max_chain_assignments = operator.index(max_chain_assignments)
        if max_chain_assignments < 1:
            fail("value_invalid", RequestRef("max_chain_assignments", str(max_chain_assignments)))

    device = predicted_coordinates.device
    target_coordinates = target_coordinates.to(device=device)
    target_coordinates = project_model_coordinates(
        target_coordinates,
        target_mask.to(device=device).bool(),
        dtype=predicted_coordinates.dtype,
        feature="BoltzGen matching truth",
    )
    target_mask = target_mask.to(device=device, dtype=torch.bool)
    model_atom_index = coordinate_atom_index.to(
        device=device,
        dtype=torch.long,
    )
    target_chain_span = target_chain_span.to(device=device, dtype=torch.long)
    target_chain_equivalence_index = target_chain_equivalence_index.to(
        device=device,
        dtype=torch.long,
    )
    target_symmetry_atom_index = target_symmetry_atom_index.to(
        device=device,
        dtype=torch.long,
    )
    target_symmetry_permutation = target_symmetry_permutation.to(
        device=device,
        dtype=torch.long,
    )

    matched_coordinates = torch.empty_like(predicted_coordinates)
    matched_mask = torch.empty(
        predicted_coordinates.shape[:-1],
        device=device,
        dtype=torch.bool,
    )
    batch_size, num_samples = predicted_coordinates.shape[:2]
    for batch_index in range(batch_size):
        spans = target_chain_span[batch_index]
        valid_chain = spans[:, 1] > spans[:, 0]
        spans = spans[valid_chain]
        equivalence = target_chain_equivalence_index[batch_index][valid_chain]
        current_atom_index = model_atom_index[batch_index]
        assignment_groups = _chain_assignment_groups(
            spans,
            equivalence,
            current_atom_index,
        )
        assignment_counts = [math.perm(len(destinations), len(sources)) for sources, destinations in assignment_groups]
        total_assignments = math.prod(assignment_counts)
        identity_ordinal = _chain_assignment_ordinal(
            assignment_groups,
            assignment_counts,
        )
        for sample_index in range(num_samples):
            assignment_ordinals = _assignment_ordinals(
                total_assignments,
                identity_ordinal,
                max_chain_assignments,
                device=device,
                generator=generator,
            )
            predicted = predicted_coordinates[batch_index, sample_index]
            best_coordinates: Tensor | None = None
            best_mask: Tensor | None = None
            best_score: Tensor | None = None
            for ordinal in assignment_ordinals:
                assignment = _decode_chain_assignment(
                    ordinal,
                    assignment_groups,
                    assignment_counts,
                )
                candidate_atom_index = _assigned_atom_index(
                    current_atom_index,
                    spans,
                    assignment,
                )
                candidate_coordinates, candidate_mask = gather_observed_coordinates(
                    target_coordinates[batch_index],
                    target_mask[batch_index],
                    candidate_atom_index,
                    torch.ones_like(candidate_atom_index, dtype=torch.bool),
                )
                score = _global_lddt_score(
                    predicted,
                    candidate_coordinates,
                    candidate_mask,
                )
                if best_score is None or score > best_score:
                    best_coordinates = candidate_coordinates
                    best_mask = candidate_mask
                    best_score = score

            current_coordinates = best_coordinates
            current_mask = best_mask
            assert current_coordinates is not None and current_mask is not None
            for group_index in range(target_symmetry_atom_index.shape[1]):
                atom_group = target_symmetry_atom_index[
                    batch_index,
                    group_index,
                ]
                atom_group = atom_group[atom_group >= 0]
                if atom_group.numel() == 0:
                    continue
                group_score = _local_lddt_score(
                    predicted,
                    current_coordinates,
                    current_mask,
                    atom_group,
                )
                group_coordinates = current_coordinates
                group_mask = current_mask
                for permutation in target_symmetry_permutation[
                    batch_index,
                    group_index,
                ]:
                    permutation = permutation[permutation >= 0]
                    if permutation.numel() == 0:
                        continue
                    candidate_coordinates = current_coordinates.clone()
                    candidate_mask = current_mask.clone()
                    candidate_coordinates[atom_group] = current_coordinates[permutation]
                    candidate_mask[atom_group] = current_mask[permutation]
                    score = _local_lddt_score(
                        predicted,
                        candidate_coordinates,
                        candidate_mask,
                        atom_group,
                    )
                    if score > group_score:
                        group_coordinates = candidate_coordinates
                        group_mask = candidate_mask
                        group_score = score
                current_coordinates = group_coordinates
                current_mask = group_mask

            matched_coordinates[batch_index, sample_index] = current_coordinates
            matched_mask[batch_index, sample_index] = current_mask
    return matched_coordinates, matched_mask


def _chain_assignment_groups(
    chain_span: Tensor,
    chain_equivalence_index: Tensor,
    atom_index: Tensor,
) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    represented = []
    for chain_index, (start, end) in enumerate(chain_span.tolist()):
        if bool(((atom_index >= start) & (atom_index < end)).any()):
            represented.append(chain_index)

    groups = []
    for equivalence_index in torch.unique(chain_equivalence_index).tolist():
        destinations = tuple(
            index for index, value in enumerate(chain_equivalence_index.tolist()) if value == equivalence_index
        )
        sources = tuple(index for index in represented if index in destinations)
        if sources:
            groups.append((sources, destinations))
    return groups


def _chain_assignment_ordinal(
    groups: list[tuple[tuple[int, ...], tuple[int, ...]]],
    counts: list[int],
) -> int:
    ordinal = 0
    for (sources, destinations), count in zip(groups, counts):
        available = list(destinations)
        group_ordinal = 0
        for position, source in enumerate(sources):
            block_size = math.perm(
                len(destinations) - position - 1,
                len(sources) - position - 1,
            )
            digit = available.index(source)
            group_ordinal += digit * block_size
            available.pop(digit)
        ordinal = ordinal * count + group_ordinal
    return ordinal


def _assignment_ordinals(
    num_assignments: int,
    identity: int,
    budget: int | None,
    *,
    device: torch.device,
    generator: torch.Generator | None,
) -> tuple[int, ...]:
    if budget is None or num_assignments <= budget:
        return (identity, *(ordinal for ordinal in range(num_assignments) if ordinal != identity))

    random_device = device if generator is None else generator.device
    population = num_assignments - 1
    sample_count = budget - 1
    remaining = population
    moved: dict[int, int] = {}
    sampled = []
    for _ in range(sample_count):
        draw = _random_integer_below(
            remaining,
            device=random_device,
            generator=generator,
        )
        value = moved.get(draw, draw)
        remaining -= 1
        moved[draw] = moved.get(remaining, remaining)
        sampled.append(value if value < identity else value + 1)
    return (identity, *sampled)


def _random_integer_below(
    upper: int,
    *,
    device: torch.device,
    generator: torch.Generator | None,
) -> int:
    r"""Draw uniformly from ``range(upper)``, composing signed-int64-safe chunks when necessary."""

    if upper <= (1 << 63) - 1:
        return int(torch.randint(upper, (), device=device, generator=generator).item())

    num_bits = (upper - 1).bit_length()
    while True:
        value = 0
        remaining_bits = num_bits
        while remaining_bits:
            chunk_bits = min(remaining_bits, 62)
            chunk = int(
                torch.randint(
                    1 << chunk_bits,
                    (),
                    device=device,
                    generator=generator,
                ).item()
            )
            value = (value << chunk_bits) | chunk
            remaining_bits -= chunk_bits
        if value < upper:
            return value


def _decode_chain_assignment(
    ordinal: int,
    groups: list[tuple[tuple[int, ...], tuple[int, ...]]],
    counts: list[int],
) -> dict[int, int]:
    group_ordinals = [0] * len(groups)
    for index in range(len(groups) - 1, -1, -1):
        ordinal, group_ordinals[index] = divmod(ordinal, counts[index])

    assignment: dict[int, int] = {}
    for group_ordinal, (sources, destinations) in zip(group_ordinals, groups):
        available = list(destinations)
        for position, source in enumerate(sources):
            block_size = math.perm(
                len(destinations) - position - 1,
                len(sources) - position - 1,
            )
            digit, group_ordinal = divmod(group_ordinal, block_size)
            assignment[source] = available.pop(digit)
    return assignment


def _assigned_atom_index(
    atom_index: Tensor,
    chain_span: Tensor,
    assignment: dict[int, int],
) -> Tensor:
    assigned = atom_index.clone()
    for source, destination in assignment.items():
        source_start, source_end = chain_span[source]
        destination_start = chain_span[destination, 0]
        included = (atom_index >= source_start) & (atom_index < source_end)
        assigned[included] = destination_start + atom_index[included] - source_start
    return assigned


def _global_lddt_score(
    predicted_coordinates: Tensor,
    target_coordinates: Tensor,
    target_mask: Tensor,
) -> Tensor:
    predicted_coordinates = project_model_coordinates(
        predicted_coordinates, target_mask.bool(), feature="BoltzGen assignment prediction"
    )
    target_coordinates = project_model_coordinates(
        target_coordinates, target_mask.bool(), feature="BoltzGen assignment truth"
    )
    predicted_distance = torch.cdist(
        predicted_coordinates,
        predicted_coordinates,
    )
    target_distance = torch.cdist(target_coordinates, target_coordinates)
    num_atoms = target_coordinates.shape[0]
    pair_mask = target_mask.unsqueeze(-1) & target_mask.unsqueeze(-2)
    pair_mask = pair_mask & ~torch.eye(
        num_atoms,
        device=target_coordinates.device,
        dtype=torch.bool,
    )
    pair_mask = pair_mask & (target_distance < 15.0)
    return _lddt_score(predicted_distance, target_distance, pair_mask)


def _local_lddt_score(
    predicted_coordinates: Tensor,
    target_coordinates: Tensor,
    target_mask: Tensor,
    atom_group: Tensor,
) -> Tensor:
    predicted_coordinates = project_model_coordinates(
        predicted_coordinates, target_mask.bool(), feature="BoltzGen assignment prediction"
    )
    target_coordinates = project_model_coordinates(
        target_coordinates, target_mask.bool(), feature="BoltzGen assignment truth"
    )
    predicted_distance = torch.cdist(
        predicted_coordinates,
        predicted_coordinates[atom_group],
    )
    target_distance = torch.cdist(
        target_coordinates,
        target_coordinates[atom_group],
    )
    row_index = torch.arange(
        target_coordinates.shape[0],
        device=target_coordinates.device,
    ).unsqueeze(-1)
    pair_mask = target_mask.unsqueeze(-1) & target_mask[atom_group].unsqueeze(0)
    pair_mask = pair_mask & (row_index != atom_group.unsqueeze(0))
    pair_mask = pair_mask & (target_distance < 15.0)
    return _lddt_score(predicted_distance, target_distance, pair_mask)


def _lddt_score(
    predicted_distance: Tensor,
    target_distance: Tensor,
    pair_mask: Tensor,
) -> Tensor:
    score = lddt_from_distances(predicted_distance, target_distance, pair_mask)
    return torch.where(pair_mask.any(), score, 0.0)
