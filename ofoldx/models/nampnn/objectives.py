r"""
NA-MPNN training objectives.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from .model import NaMpnnInput, NaMpnnOutput

_PROTEIN_RESIDUE_TYPE_SPAN = (0, 21)
_DNA_RESIDUE_TYPE_SPAN = (21, 26)
_RNA_RESIDUE_TYPE_SPAN = (26, 31)
_POLYMER_PROTEIN = 0
_POLYMER_DNA = 1
_POLYMER_RNA = 2


def smoothed_loss(
    sequence_tokens: Tensor,
    log_probabilities: Tensor,
    polymer_type_index: Tensor,
    *,
    weight: float = 0.1,
    shared_nucleic_tokens: bool = True,
    target_probabilities: Tensor | None = None,
    target_probability_mask: Tensor | None = None,
) -> Tensor:
    r"""
    Compute NA-MPNN's polymer-specific label-smoothed cross entropy.

    Args:
        sequence_tokens: Target token IDs with shape ``(..., L)``.
        log_probabilities: Vocabulary log probabilities with shape ``(..., L, V)``.
        polymer_type_index: Protein, DNA, or RNA category IDs with shape ``(..., L)``.
        weight: Smoothing mass distributed over tokens of the same polymer type.
        shared_nucleic_tokens: Whether RNA targets use the shared DNA token span.
        target_probabilities: Optional aligned target distributions replacing
            one-hot rows before smoothing.
        target_probability_mask: Rows replaced by ``target_probabilities``.

    Returns:
        Per-residue losses with shape ``(..., L)``.
    """

    num_classes = log_probabilities.shape[-1]
    device = log_probabilities.device
    dtype = log_probabilities.dtype
    target = F.one_hot(sequence_tokens, num_classes).to(dtype)
    if target_probabilities is not None and target_probability_mask is not None:
        target = torch.where(
            target_probability_mask.bool().unsqueeze(-1),
            target_probabilities,
            target,
        )
    elif target_probabilities is not None or target_probability_mask is not None:
        raise ValueError("target_probabilities and target_probability_mask must be provided together")

    def span_mask(span: tuple[int, int]) -> Tensor:
        mask = torch.zeros(num_classes, device=device, dtype=dtype)
        mask[span[0] : span[1]] = 1.0
        return mask

    protein_set = span_mask(_PROTEIN_RESIDUE_TYPE_SPAN)
    dna_set = span_mask(_DNA_RESIDUE_TYPE_SPAN)
    rna_set = dna_set if shared_nucleic_tokens else span_mask(_RNA_RESIDUE_TYPE_SPAN)
    residue_type_mask = protein_set + dna_set + rna_set

    target = target * torch.where(
        residue_type_mask.bool(),
        1.0 - weight,
        torch.ones((), device=device, dtype=dtype),
    )

    protein_rows = (polymer_type_index == _POLYMER_PROTEIN).to(dtype).unsqueeze(-1)
    dna_rows = (polymer_type_index == _POLYMER_DNA).to(dtype).unsqueeze(-1)
    rna_rows = (polymer_type_index == _POLYMER_RNA).to(dtype).unsqueeze(-1)
    smoothing = (
        protein_rows * protein_set * (weight / (_PROTEIN_RESIDUE_TYPE_SPAN[1] - _PROTEIN_RESIDUE_TYPE_SPAN[0]))
        + dna_rows * dna_set * (weight / (_DNA_RESIDUE_TYPE_SPAN[1] - _DNA_RESIDUE_TYPE_SPAN[0]))
        + rna_rows * rna_set * (weight / (_RNA_RESIDUE_TYPE_SPAN[1] - _RNA_RESIDUE_TYPE_SPAN[0]))
    )
    return -((target + smoothing) * log_probabilities).sum(dim=-1)


def criterion(
    pred: NaMpnnOutput,
    target: NaMpnnInput,
    *,
    label_smoothing: float = 0.1,
    shared_nucleic_tokens: bool = True,
) -> Tensor:
    r"""
    DanLing-shaped NA-MPNN loss: polymer-specific label-smoothed cross entropy over designable residues.

    ``shared_nucleic_tokens`` must match how the paired Processor tokenized RNA (default ``True``).
    """

    log_probabilities = F.log_softmax(pred.logits, dim=-1)
    per_residue = smoothed_loss(
        target.sequence_tokens,
        log_probabilities,
        target.polymer_type_index,
        weight=label_smoothing,
        shared_nucleic_tokens=shared_nucleic_tokens,
    )
    mask = (target.residue_mask & target.design_mask).to(per_residue.dtype)
    scope = mask.sum()
    return (per_residue * mask).sum() / scope
