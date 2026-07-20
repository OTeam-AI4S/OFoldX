r"""The EDM diffusion denoising objective.

Wires the diffusion families' training forward — Boltz1 / Boltz2 / BoltzGen /
Protenix run one EDM denoising step and return the loss-ready tensors
(``denoised_coords``, ``augmented_target``, ``noise_level``) plus the per-sample
EDM loss weight — into one scalar training loss. The model emits the tensors; the
Runner-side objective owns the weighting and reduction (DESIGN §10.3).
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from ..registry import register_objective


def edm_weight(sigma: Tensor, sigma_data: float) -> Tensor:
    r"""EDM per-sample loss weight ``(sigma^2 + sigma_data^2) / (sigma * sigma_data)^2``.

    This is the standard Karras EDM training weight. It reproduces the diffusion
    families' own helper verbatim — it is numerically identical to Boltz-2's
    :meth:`ofoldx.models.boltz2.model.AtomDiffusion.loss_weight`, including its
    ``+ 1e-8`` denominator guard against a zero ``sigma`` — while staying
    family-agnostic so the same reduction serves Boltz1 / Boltz2 / BoltzGen /
    Protenix. It is applied inline (rather than reaching into a Model attribute)
    because ``sigma_data`` is a construction-time hyperparameter on this objective,
    not a Model config field the objective may read.

    Args:
        sigma: Per-sample EDM noise levels, any shape.
        sigma_data: EDM data standard deviation (the family's preconditioning scale).

    Returns:
        Loss weights with the same shape as ``sigma``.
    """

    return (sigma**2 + sigma_data**2) / ((sigma * sigma_data) ** 2 + 1e-8)


@register_objective("diffusion")
class DiffusionObjective:
    r"""EDM-weighted mean-squared denoising error over valid atoms.

    Contract (``__call__(model, batch) -> (output, loss)``):

    - ``batch`` is a diffusion-family Input carrying ``target_positions`` and the
      atom validity mask; ``model(batch)`` runs the single EDM denoising step and
      returns an Output with ``denoised_coords``, ``augmented_target``, and
      ``noise_level`` (the per-sample noise sigma).
    - ``loss`` = mean over valid atoms of
      ``edm_weight(noise_level) * ||denoised_coords - augmented_target||^2``,
      where ``edm_weight(sigma) = (sigma^2 + sigma_data^2) / (sigma * sigma_data)^2``
      is the family's standard EDM loss weight (use the family's own weight helper
      when present, e.g. ``boltz2/model.py``'s EDM weighting).
    - Reduce over the sample axis (``num_samples``) and the batch, masking padded
      atoms; return ``(output, loss)``.

    ``sigma_data`` and ``num_samples`` are construction-time hyperparameters on
    this objective (defaults match the family config), not Model config fields.
    """

    def __init__(self, *, sigma_data: float = 16.0, num_samples: int = 1) -> None:
        self.sigma_data = sigma_data
        self.num_samples = num_samples

    def __call__(self, model: nn.Module, batch: Any) -> tuple[Any, Tensor]:
        output = model(batch, num_samples=self.num_samples)

        # ``denoised_coords`` / ``augmented_target`` carry the sample axis:
        # ``(B, num_samples, M, 3)``; ``noise_level`` is the flat per-sample sigma
        # ``(B*num_samples,)`` (boltz2/model.py: unflattened coords, flat sigmas).
        denoised: Tensor = output.denoised_coords
        target: Tensor = output.augmented_target
        batch_size, num_samples = denoised.shape[0], denoised.shape[1]

        # Squared error per atom: sum over the xyz axis first -> ``(B, num_samples, M)``.
        squared_error = (denoised - target).pow(2).sum(dim=-1)

        # Per-sample EDM weight, reshaped to broadcast over the atom axis.
        weight = edm_weight(output.noise_level, self.sigma_data).reshape(batch_size, num_samples, 1)
        weighted_error = weight * squared_error

        # Masked mean over valid atoms, reducing the sample and batch axes. The
        # atom mask ``(B, M)`` is shared across samples, so broadcast it in.
        mask = batch.atom_mask.unsqueeze(1).to(weighted_error.dtype).expand_as(weighted_error)
        denominator = mask.sum().clamp_min(torch.finfo(weighted_error.dtype).tiny)
        loss = (weighted_error * mask).sum() / denominator

        return output, loss
