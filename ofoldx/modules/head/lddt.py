r"""
Per-atom LDDT for the pLDDT confidence target.

The pLDDT head is trained to predict a per-atom local-distance-difference-test score: for each atom
``l`` it averages, over neighbour atoms ``m``, the fraction of the {0.5, 1, 2, 4} A thresholds that
the predicted-vs-true distance difference falls under. Neighbours are restricted to polymer
representative atoms within an inclusion radius (30 A if ``m`` is a nucleotide, else 15 A), excluding
the atom itself. This is a relative-distance-preservation score in [0, 1], not an absolute
positional error.
"""

from __future__ import annotations

import torch
from torch import Tensor


def compute_per_atom_lddt(
    pred_coords: Tensor,
    true_coords: Tensor,
    *,
    neighbor_mask: Tensor,
    is_nucleotide: Tensor,
    nucleotide_radius: float = 30.0,
    other_radius: float = 15.0,
) -> tuple[Tensor, Tensor]:
    r"""
    Return the per-atom LDDT numerator and weight (denominator).

    The numerator sums, over eligible neighbour atoms within the inclusion radius, the fraction of
    the {0.5, 1, 2, 4} A thresholds that the predicted-vs-true distance difference falls under; the
    weight counts those neighbours. Dividing the two yields the [0, 1] LDDT score.

    Args:
        pred_coords: `[..., N_sample, N_atom, 3]` predicted coords.
        true_coords: `[..., N_atom, 3]` ground-truth coords.
        neighbor_mask: `[..., N_atom]` atoms eligible as neighbours ``m`` (polymer rep atoms).
        is_nucleotide: `[..., N_atom]` per-atom nucleotide flag (controls the inclusion radius).
        nucleotide_radius: Neighbour inclusion radius in angstrom for nucleotide neighbours.
        other_radius: Neighbour inclusion radius in angstrom for non-nucleotide neighbours.

    Returns:
        ``(per_atom_lddt, per_atom_weight)``; divide to obtain the [0, 1] score. ``per_atom_lddt``
        is `[..., N_sample, N_atom]` and ``per_atom_weight`` broadcasts against it.

    Examples:
        >>> import torch
        >>> torch.manual_seed(0)  # doctest: +ELLIPSIS
        <torch._C.Generator object at ...>
        >>> true = torch.randn(1, 3, 3)
        >>> pred = true.unsqueeze(-3)
        >>> neighbor_mask = torch.ones(1, 3, dtype=torch.bool)
        >>> is_nucleotide = torch.zeros(1, 3, dtype=torch.bool)
        >>> num, weight = compute_per_atom_lddt(pred, true, neighbor_mask=neighbor_mask,
        ...                                     is_nucleotide=is_nucleotide)
        >>> num.shape
        torch.Size([1, 1, 3])
        >>> (num / weight.clamp_min(1)).round(decimals=4).tolist()
        [[[1.0, 1.0, 1.0]]]
    """

    n_atom = true_coords.shape[-2]
    pred = pred_coords.float()
    true = true_coords.float()

    pred_distance = torch.cdist(pred, pred)  # [..., N_sample, N_atom, N_atom]
    true_distance = torch.cdist(true, true)  # [..., N_atom, N_atom]
    delta = (pred_distance - true_distance.unsqueeze(dim=-3)).abs()
    lddt_lm = 0.25 * (
        (delta < 0.5).float() + (delta < 1.0).float() + (delta < 2.0).float() + (delta < 4.0).float()
    )  # [..., N_sample, N_atom, N_atom]

    radius = torch.where(is_nucleotide.bool(), float(nucleotide_radius), float(other_radius))  # [..., N_atom]
    locality = (true_distance < radius.unsqueeze(dim=-2)).float()  # [..., N_atom(l), N_atom(m)]
    off_diagonal = 1.0 - torch.eye(n_atom, device=true.device, dtype=true.dtype)
    neighbor = neighbor_mask.float().unsqueeze(dim=-2)  # [..., 1, N_atom(m)]
    pair_mask = (locality * off_diagonal * neighbor).unsqueeze(dim=-3)  # [..., 1, N_atom, N_atom(m)]

    per_atom_lddt = (lddt_lm * pair_mask).sum(dim=-1)  # [..., N_sample, N_atom]
    per_atom_weight = pair_mask.sum(dim=-1)  # [..., 1, N_atom]
    return per_atom_lddt, per_atom_weight


def lddt_bin_labels(
    per_atom_lddt: Tensor,
    per_atom_weight: Tensor,
    *,
    num_bins: int,
    min_bin: float = 0.0,
    max_bin: float = 1.0,
    eps: float = 1e-6,
) -> Tensor:
    r"""
    Normalize the per-atom LDDT to [0, 1] and return integer bin labels in ``[0, num_bins)``.

    Args:
        per_atom_lddt: Per-atom LDDT numerator, e.g. from [compute_per_atom_lddt][].
        per_atom_weight: Per-atom LDDT denominator (neighbour count).
        num_bins: Number of output bins.
        min_bin: Lower normalized score edge.
        max_bin: Upper normalized score edge.
        eps: Floor added to ``per_atom_weight`` to avoid division by zero.

    Returns:
        Integer bin labels in ``[0, num_bins)`` with shape matching ``per_atom_lddt``.

    Examples:
        >>> import torch
        >>> per_atom_lddt = torch.tensor([[0.0, 0.5, 1.0]])
        >>> per_atom_weight = torch.tensor([[1.0, 1.0, 1.0]])
        >>> lddt_bin_labels(per_atom_lddt, per_atom_weight, num_bins=4).tolist()
        [[0, 1, 3]]
    """

    normalized = per_atom_lddt / (per_atom_weight + eps)
    boundaries = torch.linspace(min_bin, max_bin, steps=num_bins + 1, device=normalized.device, dtype=normalized.dtype)
    bins = (normalized.unsqueeze(dim=-1) > boundaries).sum(dim=-1)  # [..., N_sample, N_atom], range [0, num_bins+1]
    return bins.clamp(min=1, max=num_bins) - 1
