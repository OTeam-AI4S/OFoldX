r"""
Family-local Protenix training objective.

This module implements the released Protenix initial-training loss recipe over
the explicit training-output contract.  It deliberately does not import the
vendored baseline at runtime and does not change the narrower shared diffusion
objective used by other OFoldX model families.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any, Literal, cast

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from ...data.molecular import ENTITY_DNA, ENTITY_DNA_RNA_HYBRID, ENTITY_MOLECULE, ENTITY_PROTEIN, ENTITY_RNA
from ...modules.geometry import weighted_rigid_align

_TERM_NAMES = ("mse", "smooth_lddt", "bond", "distogram", "plddt", "pde", "resolved", "pae")


@dataclass(frozen=True)
class ProtenixObjectiveConfig:
    r"""Configuration for the baseline-style Protenix objective.

    The ``alpha_*`` fields are the raw stage factors from the released
    configuration.  ``term_weights`` expands them into the eight final loss
    weights.  The optional ``*_weight`` fields are backward-compatible final
    weight overrides for callers of the earlier OFoldX API; new code should
    configure the raw stage factors instead.
    """

    alpha_confidence: float = 1e-4
    alpha_pae: float = 0.0
    alpha_except_pae: float = 1.0
    alpha_diffusion: float = 4.0
    alpha_distogram: float = 0.03
    alpha_bond: float = 0.0
    smooth_lddt: float = 1.0
    train_confidence_only: bool = False
    mse_weight: float | None = None
    smooth_lddt_weight: float | None = None
    bond_weight: float | None = None
    distogram_weight: float | None = None
    plddt_weight: float | None = None
    pde_weight: float | None = None
    resolved_weight: float | None = None
    pae_weight: float | None = None
    mse_internal_weight: float = 1.0 / 3.0
    protein_atom_weight: float = 1.0
    nucleotide_atom_weight: float = 6.0
    ligand_atom_weight: float = 11.0
    resolution_min: float = 0.1
    resolution_max: float = 4.0
    distogram_min_bin: float = 2.3125
    distogram_max_bin: float = 21.6875
    confidence_min_bin: float = 0.0
    confidence_max_bin: float = 32.0
    nucleotide_lddt_cutoff: float = 30.0
    other_lddt_cutoff: float = 15.0
    eps: float = 1e-6
    smooth_lddt_chunk_size: int | None = 1
    compute_zero_weight_terms: bool = True

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if value is None:
                continue
            if field.name == "smooth_lddt_chunk_size":
                if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                    raise ValueError("smooth_lddt_chunk_size must be a positive integer or None")
                continue
            if isinstance(value, bool):
                continue
            if not math.isfinite(value):
                raise ValueError(f"{field.name} must be finite")
        for name in (
            "alpha_confidence",
            "alpha_pae",
            "alpha_except_pae",
            "alpha_diffusion",
            "alpha_distogram",
            "alpha_bond",
            "smooth_lddt",
            "mse_weight",
            "smooth_lddt_weight",
            "bond_weight",
            "distogram_weight",
            "plddt_weight",
            "pde_weight",
            "resolved_weight",
            "pae_weight",
            "mse_internal_weight",
            "protein_atom_weight",
            "nucleotide_atom_weight",
            "ligand_atom_weight",
            "eps",
        ):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.eps == 0:
            raise ValueError("eps must be positive")
        if self.resolution_min > self.resolution_max:
            raise ValueError("resolution_min must not exceed resolution_max")
        if self.distogram_min_bin >= self.distogram_max_bin:
            raise ValueError("distogram_min_bin must be smaller than distogram_max_bin")
        if self.confidence_min_bin >= self.confidence_max_bin:
            raise ValueError("confidence_min_bin must be smaller than confidence_max_bin")

    @property
    def term_weights(self) -> dict[str, float]:
        r"""Return the expanded aggregate weight for every reported term."""

        expanded = {
            "mse": self.alpha_diffusion,
            "smooth_lddt": self.alpha_diffusion * self.smooth_lddt,
            "bond": self.alpha_diffusion * self.alpha_bond,
            "distogram": self.alpha_distogram,
            "plddt": self.alpha_confidence * self.alpha_except_pae,
            "pde": self.alpha_confidence * self.alpha_except_pae,
            "resolved": self.alpha_confidence * self.alpha_except_pae,
            "pae": self.alpha_confidence * self.alpha_pae,
        }
        overrides = {
            "mse": self.mse_weight,
            "smooth_lddt": self.smooth_lddt_weight,
            "bond": self.bond_weight,
            "distogram": self.distogram_weight,
            "plddt": self.plddt_weight,
            "pde": self.pde_weight,
            "resolved": self.resolved_weight,
            "pae": self.pae_weight,
        }
        expanded.update({name: value for name, value in overrides.items() if value is not None})
        return expanded


@dataclass(frozen=True)
class ProtenixLossTerms:
    r"""Detached scalar values for all eight Protenix objective terms."""

    mse: Tensor
    smooth_lddt: Tensor
    bond: Tensor
    distogram: Tensor
    plddt: Tensor
    pde: Tensor
    resolved: Tensor
    pae: Tensor

    def as_dict(self) -> dict[str, Tensor]:
        r"""Return the named terms without changing their detached values."""

        return {name: cast(Tensor, getattr(self, name)) for name in _TERM_NAMES}


@dataclass(frozen=True)
class ProtenixLossResult:
    r"""Scalar objective plus detached raw, weighted, and gating metrics."""

    loss: Tensor
    raw: ProtenixLossTerms
    weighted: ProtenixLossTerms
    confidence_resolution_valid: Tensor
    confidence_raw_metrics: Mapping[str, Tensor] | None = None
    confidence_weighted_metrics: Mapping[str, Tensor] | None = None
    mode: Literal["train", "eval"] = "train"
    available_terms: tuple[str, ...] = _TERM_NAMES

    @property
    def metrics(self) -> dict[str, Tensor]:
        r"""Return detached Runner-facing metrics."""

        confidence_names = ("plddt", "pde", "resolved", "pae")
        metrics: dict[str, Tensor] = {
            f"objective/{name}": value
            for name, value in self.raw.as_dict().items()
            if name in self.available_terms and name not in confidence_names
        }
        for name, value in self.weighted.as_dict().items():
            if name in self.available_terms and name not in confidence_names:
                metrics[f"objective/weighted_{name}"] = value
        if self.confidence_raw_metrics is not None:
            metrics.update(
                {
                    f"objective/{name}": value
                    for name, value in self.confidence_raw_metrics.items()
                    if name in self.available_terms
                }
            )
        if self.confidence_weighted_metrics is not None:
            metrics.update(
                {
                    f"objective/weighted_{name}": value
                    for name, value in self.confidence_weighted_metrics.items()
                    if name in self.available_terms
                }
            )
        metrics["objective/loss"] = self.loss.detach()
        metrics["objective/confidence_resolution_valid"] = self.confidence_resolution_valid
        return metrics


class ProtenixCriterion:
    r"""DanLing-compatible scalar criterion retaining the latest loss breakdown."""

    def __init__(self, *, sigma_data: float, config: ProtenixObjectiveConfig | None = None) -> None:
        if not math.isfinite(sigma_data) or sigma_data <= 0:
            raise ValueError("sigma_data must be finite and positive")
        self.sigma_data = float(sigma_data)
        self.config = ProtenixObjectiveConfig() if config is None else config
        self.latest_result: ProtenixLossResult | None = None

    @property
    def latest_metrics(self) -> Mapping[str, Tensor]:
        r"""Return detached metrics from the latest call, or an empty mapping."""

        return {} if self.latest_result is None else self.latest_result.metrics

    def __call__(self, prediction: Any, target: Mapping[str, Any]) -> Tensor:
        self.latest_result = protenix_training_objective(
            prediction,
            target["input"],
            target["target"],
            sigma_data=self.sigma_data,
            config=self.config,
        )
        return self.latest_result.loss

    def evaluate(self, prediction: Any, target: Mapping[str, Any]) -> Tensor:
        r"""Evaluate reverse-diffusion predictions without training noise scaling."""

        self.latest_result = protenix_evaluation_objective(
            prediction,
            target["input"],
            target["target"],
            config=self.config,
        )
        return self.latest_result.loss


def protenix_training_objective(
    output: Any,
    model_input: Any,
    target: Any,
    *,
    sigma_data: float,
    config: ProtenixObjectiveConfig | None = None,
) -> ProtenixLossResult:
    r"""Compute the structure-reduced baseline-style Protenix training loss."""

    resolved_config = ProtenixObjectiveConfig() if config is None else config
    if not math.isfinite(sigma_data) or sigma_data <= 0:
        raise ValueError("sigma_data must be finite and positive")

    matched_coordinates = _required_tensor(output, "matched_target_coordinates").float()
    matched_mask = _required_tensor(output, "matched_target_mask").bool() & model_input.atom_mask.bool()
    if matched_coordinates.shape[:-1] != matched_mask.shape or matched_coordinates.shape[-1] != 3:
        raise ValueError("matched target coordinates/mask must have shapes (B, A, 3)/(B, A)")
    molecule_type = _current_atom_molecule_type(target, model_input, matched_mask)
    atom_weight = _atom_chemistry_weight(molecule_type, matched_mask, matched_coordinates, resolved_config)
    is_nucleotide = (
        (molecule_type == ENTITY_RNA)
        | (molecule_type == ENTITY_DNA)
        | (molecule_type == ENTITY_DNA_RNA_HYBRID)
    )

    with torch.autocast(device_type=matched_coordinates.device.type, enabled=False):
        raw_per_structure: dict[str, Tensor] = {}
        batch_zero = matched_coordinates.sum(dim=(-2, -1)) * 0.0
        if resolved_config.train_confidence_only:
            raw_per_structure.update(
                {
                    "mse": batch_zero,
                    "smooth_lddt": batch_zero,
                    "bond": batch_zero,
                    "distogram": batch_zero,
                }
            )
        elif getattr(output, "coordinates", None) is None:
            unavailable = [
                name
                for name in ("mse", "smooth_lddt", "bond")
                if resolved_config.term_weights[name] != 0
            ]
            if unavailable:
                raise ValueError(
                    "nonzero coordinate-loss weights require training coordinates for: " + ", ".join(unavailable)
                )
            raw_per_structure["mse"] = batch_zero
            raw_per_structure["smooth_lddt"] = batch_zero
        else:
            raw_per_structure["mse"] = _coordinate_mse(
                output,
                matched_mask,
                atom_weight,
                sigma_data=float(sigma_data),
                config=resolved_config,
            )
            raw_per_structure["smooth_lddt"] = _smooth_lddt(
                _required_training_coordinates(output),
                matched_coordinates,
                matched_mask,
                is_nucleotide,
                resolved_config,
            )
        if not resolved_config.train_confidence_only:
            raw_per_structure["bond"] = _optional_bond_loss(
                output,
                target,
                matched_coordinates,
                matched_mask,
                sigma_data=float(sigma_data),
                config=resolved_config,
            )
            raw_per_structure["distogram"] = _optional_distogram_loss(
                output,
                model_input,
                matched_coordinates,
                matched_mask,
                config=resolved_config,
            )

        resolution_valid = _resolution_gate(target, matched_coordinates, resolved_config)
        confidence_terms = _confidence_losses(
            output,
            model_input,
            target,
            matched_coordinates,
            matched_mask,
            is_nucleotide,
            resolution_valid,
            resolved_config,
        )
        raw_per_structure.update(confidence_terms)
    available_terms = (
        ("plddt", "pde", "resolved", "pae") if resolved_config.train_confidence_only else _TERM_NAMES
    )
    return _finalize_loss_result(
        raw_per_structure,
        resolution_valid,
        matched_coordinates,
        config=resolved_config,
        mode="train",
        available_terms=available_terms,
    )


def protenix_evaluation_objective(
    output: Any,
    model_input: Any,
    target: Any,
    *,
    config: ProtenixObjectiveConfig | None = None,
) -> ProtenixLossResult:
    r"""Evaluate canonicalized reverse-diffusion predictions without EDM scaling.

    ``output`` must already have been canonicalized by the prediction-side
    permutation API.  This function gathers deterministic full-axis truth onto
    the current model axis but never performs stochastic truth-side matching.
    Distogram is intentionally unavailable in evaluation, regardless of its
    configured training-stage weight.
    """

    resolved_config = ProtenixObjectiveConfig() if config is None else config
    coordinates = _required_training_coordinates(output)
    target_coordinates, target_mask = _current_target_coordinates(target, model_input, coordinates)
    molecule_type = _current_atom_molecule_type(target, model_input, target_mask)
    atom_weight = _atom_chemistry_weight(molecule_type, target_mask, target_coordinates, resolved_config)
    is_nucleotide = (
        (molecule_type == ENTITY_RNA)
        | (molecule_type == ENTITY_DNA)
        | (molecule_type == ENTITY_DNA_RNA_HYBRID)
    )

    with torch.autocast(device_type=coordinates.device.type, enabled=False):
        batch_zero = coordinates.sum(dim=(1, 2, 3)) * 0.0
        raw_per_structure = {
            "mse": _evaluation_coordinate_mse(
                coordinates,
                target_coordinates,
                target_mask,
                atom_weight,
                resolved_config,
            ),
            "smooth_lddt": _smooth_lddt(
                coordinates,
                target_coordinates,
                target_mask,
                is_nucleotide,
                resolved_config,
            ),
            "bond": _optional_bond_loss(
                output,
                target,
                target_coordinates,
                target_mask,
                sigma_data=None,
                config=resolved_config,
            ),
            "distogram": batch_zero,
        }
        resolution_valid = _resolution_gate(target, target_coordinates, resolved_config)
        raw_per_structure.update(
            _confidence_losses(
                output,
                model_input,
                target,
                target_coordinates,
                target_mask,
                is_nucleotide,
                resolution_valid,
                resolved_config,
                confidence_coordinates=coordinates,
            )
        )
    available_terms = tuple(name for name in _TERM_NAMES if name != "distogram")
    return _finalize_loss_result(
        raw_per_structure,
        resolution_valid,
        target_coordinates,
        config=resolved_config,
        mode="eval",
        available_terms=available_terms,
    )


def _finalize_loss_result(
    raw_per_structure: Mapping[str, Tensor],
    resolution_valid: Tensor,
    like: Tensor,
    *,
    config: ProtenixObjectiveConfig,
    mode: Literal["train", "eval"],
    available_terms: tuple[str, ...],
) -> ProtenixLossResult:
    r"""Reduce per-structure terms while keeping loss and logging semantics separate."""

    scalar_raw = {name: raw_per_structure[name].float().mean() for name in _TERM_NAMES}
    weighted = {
        name: scalar_raw[name]
        * scalar_raw[name].new_tensor(config.term_weights[name] if name in available_terms else 0.0)
        for name in _TERM_NAMES
    }
    total = sum((weighted[name] for name in available_terms), like.new_zeros((), dtype=torch.float32))

    confidence_names = ("plddt", "pde", "resolved", "pae")
    if resolution_valid.any():
        confidence_raw_metrics = {
            name: raw_per_structure[name][resolution_valid].float().mean().detach() for name in confidence_names
        }
        confidence_weighted_metrics = {
            name: value * value.new_tensor(config.term_weights[name])
            for name, value in confidence_raw_metrics.items()
        }
    else:
        confidence_raw_metrics = None
        confidence_weighted_metrics = None

    raw_result = ProtenixLossTerms(**{name: scalar_raw[name].detach() for name in _TERM_NAMES})
    weighted_result = ProtenixLossTerms(**{name: weighted[name].detach() for name in _TERM_NAMES})
    return ProtenixLossResult(
        loss=total,
        raw=raw_result,
        weighted=weighted_result,
        confidence_resolution_valid=resolution_valid.float().mean().detach(),
        confidence_raw_metrics=confidence_raw_metrics,
        confidence_weighted_metrics=confidence_weighted_metrics,
        mode=mode,
        available_terms=available_terms,
    )


def _coordinate_mse(
    output: Any,
    resolved: Tensor,
    atom_weight: Tensor,
    *,
    sigma_data: float,
    config: ProtenixObjectiveConfig,
) -> Tensor:
    coordinates = _required_training_coordinates(output)
    augmented = _required_tensor(output, "augmented_target_coordinates").float()
    if augmented.shape != coordinates.shape:
        raise ValueError("augmented_target_coordinates must have the same shape as coordinates")
    augmented_mask = getattr(output, "augmented_target_mask", None)
    if augmented_mask is None:
        sample_resolved = resolved[:, None, :].expand(coordinates.shape[:-1])
    else:
        sample_resolved = augmented_mask.bool() & resolved[:, None, :]
        if sample_resolved.shape != coordinates.shape[:-1]:
            raise ValueError("augmented_target_mask must have shape (B, S, A)")
    alignment_weight = atom_weight[:, None, :] * sample_resolved.float()
    aligned_target = weighted_rigid_align(
        source=augmented.masked_fill(~sample_resolved[..., None], 0),
        target=coordinates,
        weight=alignment_weight,
    ).float()
    squared_error = (coordinates - aligned_target).square().sum(dim=-1)
    numerator = (squared_error * alignment_weight).sum(dim=-1)
    denominator = sample_resolved.float().sum(dim=-1) + config.eps
    noise_weight = _edm_weight(_required_tensor(output, "noise_level").float(), sigma_data)
    if noise_weight.shape != numerator.shape:
        raise ValueError("noise_level must have shape (B, S)")
    return config.mse_internal_weight * (noise_weight * numerator / denominator).mean(dim=-1)


def _evaluation_coordinate_mse(
    coordinates: Tensor,
    target_coordinates: Tensor,
    resolved: Tensor,
    atom_weight: Tensor,
    config: ProtenixObjectiveConfig,
) -> Tensor:
    r"""Baseline evaluation MSE: Kabsch and chemistry weights, without EDM scale."""

    sample_resolved = resolved[:, None, :].expand(coordinates.shape[:-1])
    alignment_weight = atom_weight[:, None, :] * sample_resolved.float()
    source = target_coordinates[:, None].expand_as(coordinates).masked_fill(~sample_resolved[..., None], 0)
    aligned_target = weighted_rigid_align(
        source=source,
        target=coordinates,
        weight=alignment_weight,
    ).float()
    squared_error = (coordinates - aligned_target).square().sum(dim=-1)
    numerator = (squared_error * alignment_weight).sum(dim=-1)
    denominator = sample_resolved.float().sum(dim=-1) + config.eps
    return config.mse_internal_weight * (numerator / denominator).mean(dim=-1)


def _smooth_lddt(
    coordinates: Tensor,
    target_coordinates: Tensor,
    resolved: Tensor,
    is_nucleotide: Tensor,
    config: ProtenixObjectiveConfig,
) -> Tensor:
    target = target_coordinates.masked_fill(~resolved[..., None], 0)
    target_distance = torch.cdist(target, target)
    cutoff = torch.where(
        is_nucleotide,
        target_distance.new_tensor(config.nucleotide_lddt_cutoff),
        target_distance.new_tensor(config.other_lddt_cutoff),
    )[:, :, None]
    atom_count = resolved.shape[-1]
    pair_mask = resolved[:, :, None] & resolved[:, None, :]
    pair_mask &= ~torch.eye(atom_count, device=resolved.device, dtype=torch.bool)[None, :, :]
    pair_mask &= target_distance < cutoff
    weight = pair_mask[:, None, :, :].float()
    chunk_size = config.smooth_lddt_chunk_size
    if chunk_size is None:
        per_sample = _smooth_lddt_chunk(coordinates, target_distance, weight, config.eps)
    else:
        chunks: list[Tensor] = []
        for start in range(0, coordinates.shape[1], chunk_size):
            coordinate_chunk = coordinates[:, start : start + chunk_size]
            if torch.is_grad_enabled() and coordinate_chunk.requires_grad:
                chunk = checkpoint(
                    _smooth_lddt_chunk,
                    coordinate_chunk,
                    target_distance,
                    weight,
                    config.eps,
                    use_reentrant=False,
                )
            else:
                chunk = _smooth_lddt_chunk(coordinate_chunk, target_distance, weight, config.eps)
            chunks.append(chunk)
        per_sample = torch.cat(chunks, dim=-1)
    return per_sample.mean(dim=-1)


def _smooth_lddt_chunk(
    coordinates: Tensor,
    target_distance: Tensor,
    weight: Tensor,
    eps: float,
) -> Tensor:
    r"""Compute smooth-LDDT for one contiguous chunk of diffusion samples."""

    prediction_distance = torch.cdist(coordinates, coordinates)
    distance_error = (prediction_distance - target_distance[:, None, :, :]).abs()
    score = 0.25 * (
        torch.sigmoid(0.5 - distance_error)
        + torch.sigmoid(1.0 - distance_error)
        + torch.sigmoid(2.0 - distance_error)
        + torch.sigmoid(4.0 - distance_error)
    )
    return 1.0 - (score * weight).sum(dim=(-2, -1)) / (weight.sum(dim=(-2, -1)) + eps)


def _optional_bond_loss(
    output: Any,
    target: Any,
    target_coordinates: Tensor,
    resolved: Tensor,
    *,
    sigma_data: float | None,
    config: ProtenixObjectiveConfig,
) -> Tensor:
    bond_index = getattr(target, "ligand_polymer_bond_index", None)
    bond_mask = getattr(target, "ligand_polymer_bond_mask", None)
    bond_weight = config.term_weights["bond"]
    if getattr(output, "coordinates", None) is None:
        if bond_weight != 0:
            raise ValueError("nonzero bond_weight requires training coordinates")
        return target_coordinates.sum(dim=(-2, -1)) * 0.0
    if bond_index is None or bond_mask is None:
        if bond_weight != 0:
            raise ValueError("nonzero bond_weight requires ligand_polymer_bond_index and ligand_polymer_bond_mask")
        return target_coordinates.sum(dim=(-2, -1)) * 0.0
    if bond_weight == 0 and not config.compute_zero_weight_terms:
        return target_coordinates.sum(dim=(-2, -1)) * 0.0

    coordinates = _required_training_coordinates(output)
    index = bond_index.long()
    if index.ndim != 3 or index.shape[0] != coordinates.shape[0] or index.shape[-1] != 2:
        raise ValueError("ligand_polymer_bond_index must have shape (B, E, 2)")
    if bond_mask.shape != index.shape[:-1]:
        raise ValueError("ligand_polymer_bond_mask must have shape (B, E)")
    valid_index = (index >= 0).all(dim=-1) & (index < coordinates.shape[-2]).all(dim=-1)
    safe_index = index.clamp(0, coordinates.shape[-2] - 1)
    endpoint_resolved = resolved.gather(1, safe_index[..., 0]) & resolved.gather(1, safe_index[..., 1])
    scored = bond_mask.bool() & valid_index & endpoint_resolved
    pred_first = _gather_atoms(coordinates, safe_index[..., 0])
    pred_second = _gather_atoms(coordinates, safe_index[..., 1])
    true_first = _gather_atoms(target_coordinates[:, None], safe_index[..., 0])[:, 0]
    true_second = _gather_atoms(target_coordinates[:, None], safe_index[..., 1])[:, 0]
    pred_distance = torch.linalg.vector_norm(pred_first - pred_second, dim=-1)
    true_distance = torch.linalg.vector_norm(true_first - true_second, dim=-1)
    error = (pred_distance - true_distance[:, None, :]).square()
    weight = scored[:, None, :].float()
    numerator = (error * weight).sum(dim=-1)
    count = weight.sum(dim=-1)
    denominator = torch.where(count > 0, count, torch.ones_like(count))
    graph_zero = coordinates.sum(dim=(-2, -1)) * 0.0
    per_sample = torch.where(count > 0, numerator / denominator, graph_zero)
    if sigma_data is not None:
        per_sample = per_sample * _edm_weight(_required_tensor(output, "noise_level").float(), sigma_data)
    return per_sample.mean(dim=-1)


def _optional_distogram_loss(
    output: Any,
    model_input: Any,
    target_coordinates: Tensor,
    resolved: Tensor,
    *,
    config: ProtenixObjectiveConfig,
) -> Tensor:
    logits = getattr(output, "distogram_logits", None)
    if logits is None:
        if config.term_weights["distogram"] != 0:
            raise ValueError("nonzero distogram_weight requires distogram_logits")
        return target_coordinates.sum(dim=(-2, -1)) * 0.0
    logits = logits.float()
    representative = model_input.token_to_rep_atom.long()
    if representative.shape != logits.shape[:1] + logits.shape[1:2]:
        raise ValueError("token_to_rep_atom must have shape (B, N) matching distogram_logits")
    valid = (representative >= 0) & (representative < target_coordinates.shape[-2]) & model_input.token_mask.bool()
    safe = representative.clamp(0, target_coordinates.shape[-2] - 1)
    rep_coordinates = _gather_atoms(target_coordinates[:, None], safe)[:, 0]
    rep_resolved = resolved.gather(1, safe) & valid
    distance = torch.cdist(rep_coordinates, rep_coordinates)
    boundaries = torch.linspace(
        config.distogram_min_bin,
        config.distogram_max_bin,
        logits.shape[-1] - 1,
        device=logits.device,
        dtype=torch.float32,
    )
    label = (distance[..., None] > boundaries).sum(dim=-1)
    error = _categorical_error(logits, label)
    pair_mask = rep_resolved[:, :, None] & rep_resolved[:, None, :]
    weight = pair_mask.float()
    return (error * weight).sum(dim=(-2, -1)) / (weight.sum(dim=(-2, -1)) + config.eps)


def _confidence_losses(
    output: Any,
    model_input: Any,
    target: Any,
    target_coordinates: Tensor,
    resolved: Tensor,
    is_nucleotide: Tensor,
    resolution_valid: Tensor,
    config: ProtenixObjectiveConfig,
    *,
    confidence_coordinates: Tensor | None = None,
) -> dict[str, Tensor]:
    batch_zero = target_coordinates.sum(dim=(-2, -1)) * 0.0
    term_weights = config.term_weights
    logits_by_name = {
        "plddt": getattr(output, "plddt_logits", None),
        "pde": getattr(output, "pde_logits", None),
        "resolved": getattr(output, "resolved_logits", None),
        "pae": getattr(output, "pae_logits", None),
    }
    required = [name for name, logits in logits_by_name.items() if logits is None and term_weights[name] != 0]
    if required:
        raise ValueError(f"nonzero confidence weights require logits for: {', '.join(required)}")
    if all(logits is None for logits in logits_by_name.values()):
        return {name: batch_zero for name in ("plddt", "pde", "resolved", "pae")}

    if confidence_coordinates is None:
        confidence_coordinates = _required_tensor(output, "mini_rollout_coordinates").detach().float()
        if confidence_coordinates.ndim != 4 or confidence_coordinates.shape[0] != target_coordinates.shape[0]:
            raise ValueError("mini_rollout_coordinates must have shape (B, 1, A, 3)")
        if confidence_coordinates.shape[1] != 1 or confidence_coordinates.shape[2:] != target_coordinates.shape[1:]:
            raise ValueError("confidence losses require detached mini_rollout_coordinates with sample axis 1")
    else:
        confidence_coordinates = confidence_coordinates.detach().float()
        if confidence_coordinates.ndim != 4 or confidence_coordinates.shape[0] != target_coordinates.shape[0]:
            raise ValueError("evaluation confidence coordinates must have shape (B, S, A, 3)")
        if confidence_coordinates.shape[1] < 1 or confidence_coordinates.shape[2:] != target_coordinates.shape[1:]:
            raise ValueError("evaluation confidence coordinates must have a nonempty sample axis")
    gate = resolution_valid.float()
    losses: dict[str, Tensor] = {}

    plddt_logits = logits_by_name["plddt"]
    if plddt_logits is None:
        losses["plddt"] = batch_zero
    else:
        losses["plddt"] = _plddt_loss(
            plddt_logits.float(),
            confidence_coordinates,
            target_coordinates,
            resolved,
            is_nucleotide,
            target.plddt_rep_atom_mask.bool(),
            config,
        ) * gate

    pde_logits = logits_by_name["pde"]
    if pde_logits is None:
        losses["pde"] = batch_zero
    else:
        losses["pde"] = _pde_loss(
            pde_logits.float(),
            confidence_coordinates,
            target_coordinates,
            resolved,
            model_input,
            config,
        ) * gate

    resolved_logits = logits_by_name["resolved"]
    if resolved_logits is None:
        losses["resolved"] = batch_zero
    else:
        losses["resolved"] = _resolved_loss(
            resolved_logits.float(),
            resolved,
            model_input.atom_mask.bool(),
            config,
        ) * gate

    pae_logits = logits_by_name["pae"]
    frame_fields_present = all(
        getattr(target, name, None) is not None
        for name in ("token_center_atom_index", "token_frame_atom_index", "token_has_frame")
    )
    if pae_logits is None or not frame_fields_present:
        if term_weights["pae"] != 0:
            raise ValueError(
                "nonzero pae_weight requires pae_logits, token_center_atom_index, token_frame_atom_index, "
                "and token_has_frame"
            )
        losses["pae"] = batch_zero
    elif term_weights["pae"] == 0 and not config.compute_zero_weight_terms:
        losses["pae"] = batch_zero
    else:
        losses["pae"] = _pae_loss(
            pae_logits.float(),
            confidence_coordinates,
            target_coordinates,
            resolved,
            model_input.token_mask.bool(),
            target.token_center_atom_index.long(),
            target.token_frame_atom_index.long(),
            target.token_has_frame.bool(),
            config,
        ) * gate
    return losses


def _plddt_loss(
    logits: Tensor,
    prediction: Tensor,
    target: Tensor,
    resolved: Tensor,
    is_nucleotide: Tensor,
    representative_mask: Tensor,
    config: ProtenixObjectiveConfig,
) -> Tensor:
    if logits.shape[:3] != prediction.shape[:3] or logits.shape[1] < 1:
        raise ValueError("plddt_logits must have shape (B, S, A, C) matching coordinates")
    reference = resolved & representative_mask.bool()
    prediction_distance = torch.cdist(prediction, prediction)
    target_distance = torch.cdist(target, target)
    distance_error = (prediction_distance - target_distance[:, None]).abs()
    cutoff = torch.where(
        is_nucleotide,
        target_distance.new_tensor(config.nucleotide_lddt_cutoff),
        target_distance.new_tensor(config.other_lddt_cutoff),
    )[:, None, :]
    atom_count = target.shape[-2]
    pair_mask = resolved[:, :, None] & reference[:, None, :]
    pair_mask &= ~torch.eye(atom_count, device=target.device, dtype=torch.bool)[None]
    pair_mask &= target_distance < cutoff
    hard_score = 0.25 * (
        (distance_error < 0.5).float()
        + (distance_error < 1.0).float()
        + (distance_error < 2.0).float()
        + (distance_error < 4.0).float()
    )
    weight = pair_mask[:, None].float()
    per_atom_weight = weight.sum(dim=-1)
    per_atom_lddt = (hard_score * weight).sum(dim=-1) / (per_atom_weight + config.eps)
    boundaries = torch.linspace(0.0, 1.0, logits.shape[-1] + 1, device=logits.device, dtype=torch.float32)
    label = (per_atom_lddt[..., None] > boundaries).sum(dim=-1).clamp(1, logits.shape[-1]) - 1
    error = _categorical_error(logits, label)
    atom_weight = resolved[:, None].float()
    return _exact_masked_atom_mean(error, atom_weight)


def _pde_loss(
    logits: Tensor,
    prediction: Tensor,
    target: Tensor,
    resolved: Tensor,
    model_input: Any,
    config: ProtenixObjectiveConfig,
) -> Tensor:
    representative = model_input.token_to_rep_atom.long()
    if logits.ndim != 5 or logits.shape[:2] != prediction.shape[:2]:
        raise ValueError("pde_logits must have shape (B, S, N, N, C) matching coordinates")
    if logits.shape[2] != representative.shape[1] or logits.shape[3] != representative.shape[1]:
        raise ValueError("pde_logits token axes must match token_to_rep_atom")
    valid = (representative >= 0) & (representative < target.shape[-2]) & model_input.token_mask.bool()
    safe = representative.clamp(0, target.shape[-2] - 1)
    pred_rep = _gather_atoms(prediction, safe)
    true_rep = _gather_atoms(target[:, None], safe)[:, 0]
    pred_distance = torch.cdist(pred_rep, pred_rep)
    true_distance = torch.cdist(true_rep, true_rep)
    error_distance = (pred_distance - true_distance[:, None]).abs()
    boundaries = torch.linspace(
        config.confidence_min_bin,
        config.confidence_max_bin,
        logits.shape[-1] + 1,
        device=logits.device,
        dtype=torch.float32,
    )
    label = (error_distance[..., None] > boundaries).sum(dim=-1).clamp(1, logits.shape[-1]) - 1
    error = _categorical_error(logits, label)
    rep_resolved = resolved.gather(1, safe) & valid
    pair_weight = (rep_resolved[:, :, None] & rep_resolved[:, None, :])[:, None].float()
    return ((error * pair_weight).sum(dim=(-2, -1)) / (pair_weight.sum(dim=(-2, -1)) + config.eps)).mean(
        dim=-1
    )


def _resolved_loss(logits: Tensor, resolved: Tensor, atom_mask: Tensor, config: ProtenixObjectiveConfig) -> Tensor:
    if logits.shape[:1] + logits.shape[2:3] != resolved.shape or logits.shape[1] < 1 or logits.shape[-1] != 2:
        raise ValueError("resolved_logits must have shape (B, S, A, 2) with S >= 1")
    label = resolved.long()[:, None, :].expand(logits.shape[:-1])
    error = _categorical_error(logits, label)
    weight = atom_mask[:, None].float()
    return _exact_masked_atom_mean(error, weight)


def _exact_masked_atom_mean(error: Tensor, weight: Tensor) -> Tensor:
    r"""Return an exact per-structure masked atom mean with a differentiable empty zero."""

    numerator = (error * weight).sum(dim=-1)
    count = weight.sum(dim=-1)
    denominator = torch.where(count > 0, count, torch.ones_like(count))
    graph_zero = error.sum(dim=-1) * 0.0
    per_sample = torch.where(count > 0, numerator / denominator, graph_zero)
    return per_sample.mean(dim=-1)


def _pae_loss(
    logits: Tensor,
    prediction: Tensor,
    target: Tensor,
    resolved: Tensor,
    token_mask: Tensor,
    center_index: Tensor,
    frame_index: Tensor,
    has_frame: Tensor,
    config: ProtenixObjectiveConfig,
) -> Tensor:
    atom_count = target.shape[-2]
    if logits.ndim != 5 or logits.shape[:2] != prediction.shape[:2]:
        raise ValueError("pae_logits must have shape (B, S, N, N, C) matching coordinates")
    if logits.shape[2:4] != token_mask.shape[1:] + token_mask.shape[1:]:
        raise ValueError("pae_logits token axes must match token_mask")
    if center_index.shape != token_mask.shape or frame_index.shape != token_mask.shape + (3,):
        raise ValueError("PAE center/frame indices must have shapes (B, N)/(B, N, 3)")
    valid_center = (center_index >= 0) & (center_index < atom_count) & token_mask
    valid_frame_index = (frame_index >= 0).all(dim=-1) & (frame_index < atom_count).all(dim=-1)
    safe_center = center_index.clamp(0, atom_count - 1)
    safe_frame = frame_index.clamp(0, atom_count - 1)
    center_resolved = resolved.gather(1, safe_center) & valid_center
    frame_resolved = resolved.gather(1, safe_frame[..., 0])
    frame_resolved &= resolved.gather(1, safe_frame[..., 1])
    frame_resolved &= resolved.gather(1, safe_frame[..., 2])
    frame_resolved &= has_frame & valid_frame_index & token_mask

    pred_centers = _gather_atoms(prediction, safe_center)
    true_centers = _gather_atoms(target[:, None], safe_center)[:, 0]
    pred_frames = _gather_frames(prediction, safe_frame)
    true_frames = _gather_frames(target[:, None], safe_frame)[:, 0]
    pred_local = _express_in_frames(pred_centers, pred_frames)
    true_local = _express_in_frames(true_centers[:, None], true_frames[:, None])
    squared_error = (pred_local - true_local).square().sum(dim=-1)
    boundaries = torch.linspace(
        config.confidence_min_bin,
        config.confidence_max_bin,
        logits.shape[-1] + 1,
        device=logits.device,
        dtype=torch.float32,
    ).square()
    label = (squared_error[..., None] > boundaries).sum(dim=-1).clamp(1, logits.shape[-1]) - 1
    error = _categorical_error(logits, label)
    pair_weight = (frame_resolved[:, :, None] & center_resolved[:, None, :])[:, None].float()
    return ((error * pair_weight).sum(dim=(-2, -1)) / (pair_weight.sum(dim=(-2, -1)) + config.eps)).mean(
        dim=-1
    )


def _express_in_frames(coordinates: Tensor, frames: Tensor) -> Tensor:
    first, center, third = frames.unbind(dim=-2)
    first_direction = F.normalize(first - center, dim=-1, eps=1e-8)
    third_direction = F.normalize(third - center, dim=-1, eps=1e-8)
    first_axis = F.normalize(first_direction + third_direction, dim=-1, eps=1e-8)
    second_axis = F.normalize(third_direction - first_direction, dim=-1, eps=1e-8)
    third_axis = torch.cross(first_axis, second_axis, dim=-1)
    displacement = coordinates[:, :, None, :, :] - center[:, :, :, None, :]
    return torch.stack(
        (
            (displacement * first_axis[:, :, :, None, :]).sum(dim=-1),
            (displacement * second_axis[:, :, :, None, :]).sum(dim=-1),
            (displacement * third_axis[:, :, :, None, :]).sum(dim=-1),
        ),
        dim=-1,
    )


def _current_target_coordinates(target: Any, model_input: Any, prediction: Tensor) -> tuple[Tensor, Tensor]:
    r"""Gather canonical full-axis truth onto the current prediction atom axis."""

    canonical_coordinates = _required_tensor(target, "coordinates").to(
        device=prediction.device,
        dtype=torch.float32,
    )
    canonical_mask = _required_tensor(target, "coordinate_mask").to(device=prediction.device, dtype=torch.bool)
    atom_index = model_input.atom_index.to(device=prediction.device, dtype=torch.long)
    if canonical_coordinates.ndim != 3 or canonical_coordinates.shape[-1] != 3:
        raise ValueError("target.coordinates must have shape (B, T, 3)")
    if canonical_mask.shape != canonical_coordinates.shape[:-1]:
        raise ValueError("target.coordinate_mask must have shape (B, T)")
    if atom_index.shape != prediction.shape[:1] + prediction.shape[2:3]:
        raise ValueError("input.atom_index must have shape (B, A) matching prediction coordinates")
    if canonical_coordinates.shape[0] != prediction.shape[0] or canonical_coordinates.shape[1] < 1:
        raise ValueError("canonical target and prediction batch axes must match and target must contain atoms")
    if atom_index.shape[1] != canonical_coordinates.shape[1]:
        raise ValueError("evaluation requires a full, uncropped atom_index bijection")
    valid_index = (atom_index >= 0) & (atom_index < canonical_coordinates.shape[1])
    expected_index = torch.arange(canonical_coordinates.shape[1], device=prediction.device).expand_as(atom_index)
    if not bool(valid_index.all()) or not torch.equal(atom_index.sort(dim=-1).values, expected_index):
        raise ValueError("evaluation requires atom_index to be an in-bounds canonical bijection")
    safe_index = atom_index.clamp(0, canonical_coordinates.shape[1] - 1)
    current_coordinates = canonical_coordinates.gather(1, safe_index[..., None].expand(-1, -1, 3))
    current_mask = canonical_mask.gather(1, safe_index)
    current_mask &= valid_index & model_input.atom_mask.to(device=prediction.device, dtype=torch.bool)
    return current_coordinates.masked_fill(~current_mask[..., None], 0), current_mask


def _current_atom_molecule_type(target: Any, model_input: Any, resolved: Tensor) -> Tensor:
    canonical_type = _required_tensor(target, "atom_molecule_type").to(device=resolved.device, dtype=torch.long)
    atom_index = model_input.atom_index.to(device=resolved.device, dtype=torch.long)
    if atom_index.shape != resolved.shape or canonical_type.ndim != 2 or canonical_type.shape[0] != resolved.shape[0]:
        raise ValueError("atom_molecule_type and atom_index must have shapes (B, T)/(B, A)")
    valid = (atom_index >= 0) & (atom_index < canonical_type.shape[1])
    if bool((resolved & ~valid).any()):
        raise ValueError("a resolved model atom has no valid canonical atom_molecule_type")
    return canonical_type.gather(1, atom_index.clamp(0, canonical_type.shape[1] - 1))


def _atom_chemistry_weight(
    molecule_type: Tensor,
    resolved: Tensor,
    like: Tensor,
    config: ProtenixObjectiveConfig,
) -> Tensor:
    supported = (
        (molecule_type == ENTITY_MOLECULE)
        | (molecule_type == ENTITY_PROTEIN)
        | (molecule_type == ENTITY_RNA)
        | (molecule_type == ENTITY_DNA)
        | (molecule_type == ENTITY_DNA_RNA_HYBRID)
    )
    if bool((resolved & ~supported).any()):
        values = torch.unique(molecule_type[resolved & ~supported]).detach().cpu().tolist()
        raise ValueError(f"no baseline chemistry weight is defined for resolved atom molecule types {values}")
    nucleotide = (
        (molecule_type == ENTITY_RNA)
        | (molecule_type == ENTITY_DNA)
        | (molecule_type == ENTITY_DNA_RNA_HYBRID)
    )
    return torch.where(
        molecule_type == ENTITY_PROTEIN,
        like.new_tensor(config.protein_atom_weight),
        torch.where(
            nucleotide,
            like.new_tensor(config.nucleotide_atom_weight),
            like.new_tensor(config.ligand_atom_weight),
        ),
    ).float()


def _resolution_gate(target: Any, like: Tensor, config: ProtenixObjectiveConfig) -> Tensor:
    resolution = _required_tensor(target, "resolution").float()
    known = _required_tensor(target, "resolution_is_known").bool()
    if resolution.shape != (like.shape[0],) or known.shape != resolution.shape:
        raise ValueError("resolution and resolution_is_known must have shape (B,)")
    return known & torch.isfinite(resolution) & (resolution >= config.resolution_min) & (
        resolution <= config.resolution_max
    )


def _edm_weight(noise_level: Tensor, sigma_data: float) -> Tensor:
    sigma_data_tensor = noise_level.new_tensor(sigma_data)
    return (noise_level.square() + sigma_data_tensor.square()) / (noise_level * sigma_data_tensor).square()


def _categorical_error(logits: Tensor, label: Tensor) -> Tensor:
    if logits.shape[:-1] != label.shape:
        raise ValueError(f"categorical logits/label shapes do not match: {logits.shape[:-1]} != {label.shape}")
    return -F.log_softmax(logits.float(), dim=-1).gather(-1, label.long().unsqueeze(-1)).squeeze(-1)


def _gather_atoms(coordinates: Tensor, index: Tensor) -> Tensor:
    # coordinates: (B, S, A, 3), index: (B, K)
    gather_index = index[:, None, :, None].expand(coordinates.shape[0], coordinates.shape[1], index.shape[1], 3)
    return coordinates.gather(2, gather_index)


def _gather_frames(coordinates: Tensor, index: Tensor) -> Tensor:
    # coordinates: (B, S, A, 3), index: (B, N, 3)
    batch_size, sample_count = coordinates.shape[:2]
    atom_count = index.shape[1] * 3
    flat_index = index.reshape(index.shape[0], atom_count)
    gathered = _gather_atoms(coordinates, flat_index)
    return gathered.reshape(batch_size, sample_count, index.shape[1], 3, 3)


def _required_training_coordinates(output: Any) -> Tensor:
    coordinates = _required_tensor(output, "coordinates")
    if coordinates.ndim != 4 or coordinates.shape[-1] != 3:
        raise ValueError("coordinates must have shape (B, S, A, 3)")
    return coordinates.float()


def _required_tensor(owner: Any, name: str) -> Tensor:
    value = getattr(owner, name, None)
    if not isinstance(value, Tensor):
        raise ValueError(f"Protenix objective requires tensor field {name!r}")
    return value


def criterion_factory(
    model: Any,
    processor: Any,
    objective_config: ProtenixObjectiveConfig | None = None,
) -> ProtenixCriterion:
    r"""Bind the Protenix objective to the model's EDM data scale for Runner."""

    del processor
    return ProtenixCriterion(
        sigma_data=float(model.config.denoiser.sigma_data),
        config=objective_config,
    )
