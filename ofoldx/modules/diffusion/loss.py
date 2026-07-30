r"""
Shared EDM coordinate-denoising training loss for the diffusion structure families.

The internally-noising diffusion models (Boltz-1, Boltz-2, BoltzGen, Protenix) expose their denoised
coordinates, the augmented training target, and the per-sample noise level, so the canonical EDM
denoising objective -- the ``1/c_out(sigma)**2``-weighted squared error -- grounds directly against the
model output. The target is rigidly aligned onto the prediction first (detached weighted Kabsch, as in
AF3/Boltz/Protenix): the denoiser output is not exactly equivariant, so its global pose is not anchored
to the augmented frame and the arbitrary augmentation rotation would otherwise be penalized.

The loss is reduced over the *resolved* atoms only. ``input.atom_mask`` marks every atom slot in the
padded layout, but unresolved/missing atoms carry no ground truth, so their augmented target coordinates
are meaningless; including them corrupts the loss. The resolved set is reconstructed from the structure
Target (``coordinate_mask`` gathered through ``atom_index``) intersected with ``input.atom_mask`` -- the
same mask the models' own ``_gather_*_structure_target`` helpers compute.
"""

from __future__ import annotations

from typing import Any

from torch import Tensor

from ..geometry import weighted_rigid_align


def resolved_atom_mask(target: Any, model_input: Any) -> Tensor:
    r"""
    Return the ``(B, A)`` boolean mask of atoms that carry ground-truth coordinates and are present.

    Args:
        target: The structure Target, carrying ``coordinate_mask`` ``(B, T)`` over the complete atom pool
            and ``atom_index`` ``(B, A)`` mapping each padded-layout atom to a pool index (``-1`` for
            padding).
        model_input: The batched model Input, carrying ``atom_mask`` ``(B, A)``.
    """

    atom_index = target.atom_index.long()
    valid = atom_index >= 0
    resolved = target.coordinate_mask.gather(1, atom_index.clamp_min(0)).bool()
    return resolved & valid & model_input.atom_mask.bool()


def edm_loss_weight(noise_level: Tensor, sigma_data: float) -> Tensor:
    r"""
    Return the EDM per-sample loss weight ``(sigma**2 + sigma_data**2) / (sigma * sigma_data)**2``.

    This is the standard ``1 / c_out(sigma)**2`` weighting from Karras et al.; it upweights low-noise
    samples where the denoiser must be precise. ``sigma`` is strictly positive during training (drawn
    log-normal), so no epsilon guard is needed or wanted -- an epsilon perturbs exactly this low-noise
    regime the weight is meant to emphasize.

    Args:
        noise_level: Per-sample noise standard deviations ``sigma``.
        sigma_data: EDM data scale used by the model's preconditioner.
    """

    return (noise_level**2 + sigma_data**2) / (noise_level * sigma_data) ** 2


def edm_denoising_loss(pred: Any, resolved: Tensor, *, sigma_data: float) -> Tensor:
    r"""
    Compute the EDM-weighted coordinate denoising loss, averaged over resolved atoms and drawn samples.

    Args:
        pred: A diffusion model output carrying ``coordinates`` and ``augmented_target_coordinates`` of
            shape ``(B, S, A, 3)`` and ``noise_level`` of shape ``(B, S)``. The target is rigidly aligned
            onto the prediction before the squared error.
        resolved: The ``(B, A)`` resolved-atom mask from
            [`resolved_atom_mask`][ofoldx.modules.diffusion.loss.resolved_atom_mask].
        sigma_data: EDM data scale (from the model's ``config.denoiser.sigma_data``).

    An empty resolved scope gives a NaN mean -- surfaced, not hidden by a fallback. The denominator is the
    same resolved-atom count the training batch reports as its ``loss_normalizer``, so DanLing's
    accumulation/DDP reduction forms the true global per-atom mean.
    """

    # Rigidly align the augmented target onto the prediction (per sample, over resolved atoms) before the
    # MSE, as AF3/Boltz/Protenix -- and this repo's own BoltzGen loss -- do. The EDM output
    # ``denoised = c_skip*noisy + c_out*update`` is dominated at high sigma by the non-equivariant network
    # ``update``, whose global pose is NOT anchored to the augmented frame, so the arbitrary augmentation
    # rotation must be aligned out rather than penalized. ``weighted_rigid_align`` is detached, so the
    # gradient flows only through the prediction; unresolved atoms are zeroed and given zero weight so they
    # neither bias nor NaN the fit.
    resolved_atoms = resolved.bool()
    aligned_target = weighted_rigid_align(
        source=pred.augmented_target_coordinates.masked_fill(~resolved_atoms[:, None, :, None], 0),
        target=pred.coordinates,
        weight=resolved_atoms.to(pred.coordinates.dtype)[:, None, :],
    )
    squared_error = (pred.coordinates - aligned_target).pow(2).sum(dim=-1)  # (B, S, A)
    weight = edm_loss_weight(pred.noise_level, sigma_data).unsqueeze(-1)  # (B, S, 1)
    weighted_error = weight * squared_error  # (B, S, A)
    mask = resolved.to(weighted_error.dtype).unsqueeze(1).expand_as(weighted_error)  # (B, S, A)
    return (weighted_error * mask).sum() / mask.sum()


def denoising_criterion_factory(model: Any, processor: Any) -> Any:
    r"""
    Build the coordinate-denoising criterion for a diffusion family, closed over the model's ``sigma_data``.

    The diffusion families share this criterion (they all expose the same augmented-frame
    ``coordinates``/``augmented_target_coordinates``/``noise_level`` output); the Runner selects it
    for the supported architectures. The resolved-atom mask is rebuilt from the collated
    ``(input, target)`` pair at each call, and ``sigma_data`` is read once from
    ``model.config.denoiser.sigma_data``.
    """

    sigma_data = float(model.config.denoiser.sigma_data)

    def criterion(pred: Any, target: Any) -> Tensor:
        resolved = resolved_atom_mask(target["target"], target["input"])
        return edm_denoising_loss(pred, resolved, sigma_data=sigma_data)

    return criterion
