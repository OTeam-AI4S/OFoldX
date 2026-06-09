r"""
Coordinate losses.

Loss terms that compare predicted atom coordinates against a ground-truth structure:
[`CoordinateMseLoss`][ofoldx.modules.loss.coordinate.CoordinateMseLoss] for direct masked MSE
and [`DiffusionDenoisingLoss`][ofoldx.modules.loss.coordinate.DiffusionDenoisingLoss] for the
EDM-weighted single-step diffusion training objective.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..geometry import weighted_rigid_align


class CoordinateMseLoss(nn.Module):
    r"""
    Masked mean-squared error between predicted and target atom coordinates.

    Compares one or more predicted coordinate samples against a single ground-truth structure,
    averaging the squared error only over valid (unmasked) atoms and the three spatial axes. Used
    as the eval/inference coordinate score when no diffusion denoising target is available.

    Examples:
        >>> import torch
        >>> loss = CoordinateMseLoss()
        >>> pred_coords = torch.zeros(1, 2, 3, 3)
        >>> target_positions = torch.ones(1, 3, 3)
        >>> target_mask = torch.ones(1, 3)
        >>> round(float(loss(pred_coords, target_positions, target_mask)), 4)
        1.0
    """

    def forward(self, pred_coords: Tensor, target_positions: Tensor, target_mask: Tensor) -> Tensor:
        r"""
        Return coordinate MSE averaged over valid target coordinates.

        The predicted tensor carries an extra leading sample axis; the target is broadcast across
        it. Squared error is summed over valid atoms and the three spatial axes, then divided by the
        number of contributing scalars (valid atoms times samples times ``3``, floored at ``1``).

        Args:
            pred_coords: `[..., num_samples, N_atom, 3]` predicted coordinates.
            target_positions: `[..., N_atom, 3]` target coordinates.
            target_mask: `[..., N_atom]` valid-target mask.

        Returns:
            Scalar masked coordinate MSE.

        Raises:
            ValueError: If ``pred_coords`` does not include a sample axis before the atom axis.

        Examples:
            >>> import torch
            >>> loss = CoordinateMseLoss()
            >>> pred_coords = torch.full((1, 2, 3, 3), 2.0)
            >>> target_positions = torch.zeros(1, 3, 3)
            >>> target_mask = torch.ones(1, 3)
            >>> out = loss(pred_coords, target_positions, target_mask)
            >>> out.shape
            torch.Size([])
            >>> round(float(out), 4)
            4.0
        """

        target_positions = target_positions.to(device=pred_coords.device, dtype=pred_coords.dtype)
        target_mask = target_mask.to(device=pred_coords.device).bool()
        if pred_coords.dim() != target_positions.dim() + 1:
            raise ValueError(
                "pred_coords must have shape [..., S, N_atom, 3], "
                f"got {tuple(pred_coords.shape)} for target {tuple(target_positions.shape)}"
            )
        squared_error = (pred_coords - target_positions.unsqueeze(-3)) ** 2
        coordinate_mask = target_mask.unsqueeze(-2).unsqueeze(-1).to(dtype=pred_coords.dtype)
        normalizer = (coordinate_mask.sum() * pred_coords.shape[-3] * 3).clamp_min(1)
        return (squared_error * coordinate_mask).sum() / normalizer


class DiffusionDenoisingLoss(nn.Module):
    r"""
    EDM-weighted single-step denoising loss for diffusion training.

    For each diffusion sample the per-sample augmented ground truth is rigidly aligned onto the
    denoiser's prediction (weighted Kabsch, computed in fp32 with stop-gradient) so the loss measures
    structure error rather than the global pose the diffusion model cannot control. The aligned
    per-atom MSE is then scaled by ``weight_mse`` and the EDM per-sample weight
    ``(sigma^2 + sigma_data^2) / (sigma_data * sigma)^2``. This is the diffusion training objective,
    NOT a loss on reverse-sampled coordinates.

    The per-atom ``atom_weight`` argument to ``forward`` accepts an optional weighting when
    molecule-type features are available; it defaults to the valid-atom mask (uniform weight ``1``
    per real atom).

    Args:
        sigma_data: EDM data-scale constant used to form the per-sample weight.
        weight_mse: Constant multiplier on the aligned per-sample MSE.
        eps: Floor for the weighted-normalizer and Kabsch denominators.

    Examples:
        >>> import torch
        >>> loss = DiffusionDenoisingLoss()
        >>> loss.sigma_data
        16.0
        >>> round(loss.weight_mse, 4)
        0.3333
    """

    def __init__(self, *, sigma_data: float = 16.0, weight_mse: float = 1.0 / 3.0, eps: float = 1e-8) -> None:
        super().__init__()
        self.sigma_data = sigma_data
        self.weight_mse = weight_mse
        self.eps = eps

    def forward(
        self,
        denoised_coords: Tensor,
        augmented_target: Tensor,
        noise_level: Tensor,
        atom_mask: Tensor,
        atom_weight: Tensor | None = None,
    ) -> Tensor:
        r"""
        Return the EDM-weighted, alignment-corrected denoising MSE.

        Each augmented target is rigidly aligned (weighted Kabsch) onto the corresponding denoised
        prediction before the squared error is taken, so only structure error contributes. The
        per-sample MSE is normalized by the summed atom weights, scaled by ``weight_mse`` and the EDM
        weight, and averaged across samples and any leading batch axes.

        Args:
            denoised_coords: `[..., num_samples, N_atom, 3]` single-step predictions.
            augmented_target: `[..., num_samples, N_atom, 3]` per-sample ground truth.
            noise_level: `[..., num_samples]` per-sample sigma.
            atom_mask: `[..., N_atom]` valid-atom mask.
            atom_weight: Optional `[..., N_atom]` per-atom loss weight; defaults to ``atom_mask``.

        Returns:
            Scalar EDM-weighted denoising loss.

        Examples:
            >>> import torch
            >>> _ = torch.manual_seed(0)
            >>> loss = DiffusionDenoisingLoss()
            >>> denoised_coords = torch.randn(1, 2, 4, 3)
            >>> augmented_target = torch.randn(1, 2, 4, 3)
            >>> noise_level = torch.tensor([[1.0, 2.0]])
            >>> atom_mask = torch.ones(1, 4)
            >>> out = loss(denoised_coords, augmented_target, noise_level, atom_mask)
            >>> out.shape
            torch.Size([])
            >>> round(float(out), 4)
            0.457
        """

        augmented_target = augmented_target.to(device=denoised_coords.device, dtype=denoised_coords.dtype)
        atom_mask = atom_mask.to(device=denoised_coords.device, dtype=denoised_coords.dtype)
        if atom_weight is None:
            atom_weight = atom_mask
        else:
            atom_weight = atom_weight.to(device=denoised_coords.device, dtype=denoised_coords.dtype) * atom_mask
        sample_weight = atom_weight.unsqueeze(-2)  # broadcast over the sample axis: [..., 1, N_atom]

        aligned_target = weighted_rigid_align(
            source=augmented_target,
            target=denoised_coords,
            weight=sample_weight,
            eps=self.eps,
        ).to(dtype=denoised_coords.dtype)

        per_atom_se = ((denoised_coords - aligned_target) ** 2).sum(dim=-1)  # [..., num_samples, N_atom]
        per_sample_mse = (per_atom_se * sample_weight).sum(dim=-1) / sample_weight.sum(dim=-1).clamp_min(self.eps)

        sigma_data = torch.as_tensor(self.sigma_data, device=denoised_coords.device, dtype=denoised_coords.dtype)
        edm_weight = (noise_level**2 + sigma_data**2) / (sigma_data * noise_level).clamp_min(1e-12) ** 2
        return (self.weight_mse * per_sample_mse * edm_weight).mean()
