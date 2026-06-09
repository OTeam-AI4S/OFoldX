r"""
Composite model loss.

Assembles the individual coordinate, distogram, and confidence loss terms into a single weighted
training objective via [`CompositeLoss`][ofoldx.modules.loss.composite.CompositeLoss], with
structural [Protocol][typing.Protocol]s describing the model output and input fields the loss reads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor, nn

from ..head import ConfidenceLoss, DistogramLoss
from .coordinate import CoordinateMseLoss, DiffusionDenoisingLoss


class StructurePredictionOutputLike(Protocol):
    r"""
    Structural contract for the model output fields the composite loss reads.

    A [Protocol][typing.Protocol]: any structure-prediction output exposing these attributes
    satisfies it, without an explicit subclass relationship. The diffusion-training fields
    (``denoised_coords``, ``augmented_target``, ``noise_level``) are ``None`` at eval/inference time,
    which selects the fallback MSE coordinate term.

    Attributes:
        pred_coords: `[..., num_samples, N_atom, 3]` sampled predicted coordinates.
        distogram_logits: `[..., N_token, N_token, num_bins]` distogram head logits.
        plddt_logits: Optional pLDDT confidence logits; when ``None`` the confidence term is skipped.
        pae_logits: Optional predicted-aligned-error logits.
        pde_logits: Optional predicted-distance-error logits.
        resolved_logits: Optional experimentally-resolved head logits.
        denoised_coords: Optional `[..., num_samples, N_atom, 3]` single-step denoiser predictions;
            present only in diffusion-training mode.
        augmented_target: Optional per-sample augmented ground-truth coordinates.
        noise_level: Optional `[..., num_samples]` per-sample diffusion sigma.
    """

    pred_coords: Tensor
    distogram_logits: Tensor
    plddt_logits: Tensor | None
    pae_logits: Tensor | None
    pde_logits: Tensor | None
    resolved_logits: Tensor | None
    denoised_coords: Tensor | None
    augmented_target: Tensor | None
    noise_level: Tensor | None


class ModelInputLike(Protocol):
    r"""
    Structural contract for the model input (batch) fields the composite loss reads.

    A [Protocol][typing.Protocol]: any batch object exposing these attributes satisfies it. The
    loss-only inputs are ``Optional`` on the model contract but must be present to compute the
    training losses; missing ones raise via
    [`_require_target`][ofoldx.modules.loss.composite._require_target].

    Attributes:
        target_positions: Optional `[..., N_atom, 3]` ground-truth atom coordinates.
        target_mask: Optional `[..., N_atom]` valid-target mask.
        representative_atom_mask: `[..., N_atom]` mask selecting each token's representative atom.
        token_to_rep_atom: Index map from tokens to their representative atom.
        atom_mask: `[..., N_atom]` valid-atom mask.
        atom_to_token: `[..., N_atom]` index map from atoms to their owning token.
        molecule_type: Optional per-token molecule type (0=protein, 1=rna, 2=dna, 3=ligand).
        frame_atom_index: Optional per-frame atom indices used by the confidence loss.
        frame_mask: Optional valid-frame mask used by the confidence loss.
    """

    target_positions: Tensor | None
    target_mask: Tensor | None
    representative_atom_mask: Tensor
    token_to_rep_atom: Tensor
    atom_mask: Tensor
    atom_to_token: Tensor
    # Loss-only inputs: optional on the contract, but required to compute the training losses.
    molecule_type: Tensor | None
    frame_atom_index: Tensor | None
    frame_mask: Tensor | None


@dataclass(slots=True)
class LossOutput:
    r"""
    Structured loss output for training steps.

    Bundles the single scalar objective that autograd backpropagates through together with the
    unweighted per-term components, so they can be logged individually without recomputing them.

    Attributes:
        total: Weighted scalar loss.
        components: Named unweighted or component-weighted loss terms.

    Examples:
        >>> import torch
        >>> out = LossOutput(total=torch.tensor(1.5), components={"coordinate": torch.tensor(0.5)})
        >>> round(float(out.total), 4)
        1.5
        >>> sorted(out.components)
        ['coordinate']
    """

    total: Tensor
    components: dict[str, Tensor]


class CompositeLoss(nn.Module):
    r"""
    Combine the coordinate, distogram, and confidence loss terms into one scalar objective.

    Each sub-loss defaults to its standard module and may be overridden for testing or ablations.
    The forward pass selects the diffusion denoising term in training mode and the plain coordinate
    MSE otherwise, sums the enabled components with per-name scalar weights, and returns both the
    weighted total and the raw components.

    Args:
        coordinate: Optional coordinate loss override.
        diffusion: Optional diffusion denoising loss override.
        distogram: Optional distogram loss override.
        confidence: Optional confidence loss override.
        weights: Optional scalar weights keyed by loss component name; merged over the defaults
            ``{"coordinate": 1.0, "distogram": 1.0, "confidence": 1.0}``.

    Examples:
        >>> loss = CompositeLoss(weights={"distogram": 2.0})
        >>> loss.weights == {"coordinate": 1.0, "distogram": 2.0, "confidence": 1.0}
        True
    """

    def __init__(
        self,
        *,
        coordinate: CoordinateMseLoss | None = None,
        diffusion: DiffusionDenoisingLoss | None = None,
        distogram: DistogramLoss | None = None,
        confidence: ConfidenceLoss | None = None,
        weights: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.coordinate = CoordinateMseLoss() if coordinate is None else coordinate
        self.diffusion = DiffusionDenoisingLoss() if diffusion is None else diffusion
        self.distogram = DistogramLoss() if distogram is None else distogram
        self.confidence = ConfidenceLoss() if confidence is None else confidence
        self.weights = {"coordinate": 1.0, "distogram": 1.0, "confidence": 1.0}
        if weights is not None:
            self.weights.update(weights)

    def forward(self, output: StructurePredictionOutputLike, batch: ModelInputLike) -> LossOutput:
        r"""
        Return weighted total loss and named components.

        The coordinate term is the single-step EDM denoising loss when the model ran in
        training mode (``output.denoised_coords`` present); otherwise it falls back to MSE
        against the sampled ``pred_coords`` (eval/inference scoring). The confidence term is
        included only when ``output.plddt_logits`` is present.

        Args:
            output: Structure-prediction output fields required by the loss.
            batch: Model input fields carrying targets and masks.

        Returns:
            Structured total loss and component dictionary.

        Raises:
            ValueError: If a required target or mask field on ``batch`` (or a diffusion-mode field
                on ``output``) is ``None``.

        Examples:
            >>> loss = CompositeLoss()
            >>> result = loss(model_output, batch)  # doctest: +SKIP
            >>> result.total.shape  # doctest: +SKIP
            torch.Size([])
        """

        target_positions = _require_target(batch.target_positions, "target_positions")
        target_mask = _require_target(batch.target_mask, "target_mask")
        molecule_type = _require_target(batch.molecule_type, "molecule_type")
        frame_atom_index = _require_target(batch.frame_atom_index, "frame_atom_index")
        frame_mask = _require_target(batch.frame_mask, "frame_mask")
        if output.denoised_coords is not None:
            coordinate_loss = self.diffusion(
                output.denoised_coords,
                _require_target(output.augmented_target, "augmented_target"),
                _require_target(output.noise_level, "noise_level"),
                # Resolved mask: materialized missing atoms have no true coordinate to denoise toward.
                target_mask,
                _atom_loss_weight(molecule_type, batch.atom_to_token),
            )
        else:
            coordinate_loss = self.coordinate(output.pred_coords, target_positions, target_mask)
        components = {
            "coordinate": coordinate_loss,
            "distogram": self.distogram(
                output.distogram_logits,
                target_positions,
                target_mask,
                batch.token_to_rep_atom,
            ),
        }
        if output.plddt_logits is not None:
            components["confidence"] = self.confidence(
                pred_coords=output.pred_coords,
                target_positions=target_positions,
                target_mask=target_mask,
                representative_atom_mask=batch.representative_atom_mask,
                token_to_rep_atom=batch.token_to_rep_atom,
                frame_atom_index=frame_atom_index,
                frame_mask=frame_mask,
                molecule_type=molecule_type,
                atom_to_token=batch.atom_to_token,
                atom_mask=batch.atom_mask,
                plddt_logits=output.plddt_logits,
                resolved_logits=output.resolved_logits,
                pae_logits=output.pae_logits,
                pde_logits=output.pde_logits,
            )
        total = output.pred_coords.new_zeros(())
        for name, loss in components.items():
            total = total + loss * self.weights.get(name, 1.0)
        return LossOutput(total=total, components=components)


def _require_target(target: Tensor | None, name: str) -> Tensor:
    r"""
    Return ``target`` unchanged, raising if it is ``None``.

    Args:
        target: An optional loss input that must be present.
        name: Field name used in the error message.

    Returns:
        The non-``None`` ``target``.

    Raises:
        ValueError: If ``target`` is ``None``.
    """
    if target is None:
        raise ValueError(f"{name} is required to compute training losses")
    return target


def _atom_loss_weight(molecule_type: Tensor, atom_to_token: Tensor) -> Tensor:
    r"""
    Per-atom diffusion-loss weight ``1 + 5*is_dna + 5*is_rna + 10*is_ligand``.

    ``molecule_type`` is per-token (0=protein, 1=rna, 2=dna, 3=ligand); it is broadcast to atoms
    through ``atom_to_token``, up-weighting nucleic-acid and ligand atoms relative to protein atoms.

    Args:
        molecule_type: `[..., N_token]` per-token molecule-type codes.
        atom_to_token: `[..., N_atom]` index map from atoms to their owning token.

    Returns:
        `[..., N_atom]` per-atom loss weight.

    Examples:
        >>> import torch
        >>> molecule_type = torch.tensor([[0, 1, 2, 3]])
        >>> atom_to_token = torch.tensor([[0, 0, 1, 2, 3]])
        >>> _atom_loss_weight(molecule_type, atom_to_token).tolist()
        [[1.0, 1.0, 6.0, 6.0, 11.0]]
    """

    atom_molecule_type = torch.gather(molecule_type, -1, atom_to_token)  # [..., N_atom]
    is_rna = atom_molecule_type == 1
    is_dna = atom_molecule_type == 2
    is_ligand = atom_molecule_type == 3
    return 1.0 + 5.0 * is_dna + 5.0 * is_rna + 10.0 * is_ligand
