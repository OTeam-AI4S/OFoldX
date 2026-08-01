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
NA-MPNN training objectives.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from danling.tensors import NestedTensor
from torch import Tensor, nn

from ...data.tokenization import TOKEN_INDEX
from ...errors import RequestRef, fail
from ...modules.functional import weighted_mean
from ..processing import model_log_probabilities
from ..targets import SequenceTarget
from .model import NaMpnnOutput

#: Per polymer code, the classes its label smoothing spreads over (upstream's residue-type spans, ``X`` included).
_PROTEIN_CLASSES = tuple(TOKEN_INDEX[letter] for letter in "ARNDCQEGHILKMFPSTWYVX")
_DNA_CLASSES = tuple(TOKEN_INDEX[f"DNA_{base}"] for base in "ACGTX")
_RNA_CLASSES = tuple(TOKEN_INDEX[f"RNA_{base}"] for base in "ACGUX")
_POLYMER_PROTEIN = 0
_POLYMER_DNA = 1
_POLYMER_RNA = 2


def _require(condition: Tensor, message: str) -> None:
    if torch.compiler.is_compiling():
        torch._assert_async(condition, message)
    elif not bool(condition):
        fail("value_invalid", RequestRef("NaMpnnLoss", message))


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
        sequence_tokens: Target canonical classes with shape ``(..., L)``.
        log_probabilities: Vocabulary log probabilities with shape ``(..., L, V)``.
        polymer_type_index: Protein, DNA, or RNA category IDs with shape ``(..., L)``.
        weight: Smoothing mass distributed over tokens of the same polymer type.
        shared_nucleic_tokens: Whether RNA targets use the shared DNA classes.
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
        fail("value_invalid", RequestRef("target_probabilities", "without target_probability_mask"))

    def class_mask(classes: tuple[int, ...]) -> Tensor:
        mask = torch.zeros(num_classes, device=device, dtype=dtype)
        mask[list(classes)] = 1.0
        return mask

    protein_set = class_mask(_PROTEIN_CLASSES)
    dna_set = class_mask(_DNA_CLASSES)
    rna_set = dna_set if shared_nucleic_tokens else class_mask(_RNA_CLASSES)
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
        protein_rows * protein_set * (weight / len(_PROTEIN_CLASSES))
        + dna_rows * dna_set * (weight / len(_DNA_CLASSES))
        + rna_rows * rna_set * (weight / len(_RNA_CLASSES))
    )
    probabilities = target + smoothing
    required = probabilities != 0
    finite = torch.isfinite(log_probabilities)
    _require(
        (finite | torch.isneginf(log_probabilities)).all(),
        "NA-MPNN log probabilities require finite values or explicit class exclusions",
    )
    _require(
        (finite | ~required).all(),
        "NA-MPNN target mass requires finite log probabilities at its supported classes",
    )
    # A zero target mass carries no observation of an excluded class. Select
    # those operands before multiplication, including deliberate -Inf logits.
    # Keep finite zero-coefficient terms so derivatives of an optional soft
    # target at zero retain their usual cross-entropy value.
    selected = torch.where(finite, log_probabilities, 0.0)
    return -(probabilities * selected).sum(dim=-1)


class NaMpnnLoss(nn.Module):
    r"""
    Polymer-specific label-smoothed cross entropy over designable residues.

    Args:
        label_smoothing: Uniform label-smoothing mass distributed over the vocabulary.
        shared_nucleic_tokens: Whether RNA targets use the shared DNA classes. Must match how the paired
            Processor tokenized RNA (default ``True``).
    """

    def __init__(self, *, label_smoothing: float = 0.1, shared_nucleic_tokens: bool = True) -> None:
        super().__init__()
        self.label_smoothing = label_smoothing
        self.shared_nucleic_tokens = shared_nucleic_tokens

    def forward(
        self,
        pred: NaMpnnOutput,
        target: SequenceTarget,
        *,
        polymer_type_index: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        r"""Score known sequence truth in the requested polymer-specific domain.

        Args:
            pred: Predicted vocabulary logits.
            target: Aligned classes and their knownness mask.
            polymer_type_index: Input-side protein, DNA, or RNA category for each row.
            mask: Optional scoring domain, intersected with target knownness.

        Returns:
            Mean label-smoothed cross entropy over the selected rows.
        """
        mask = target.mask if mask is None else target.mask & mask.bool()
        # Unscored rows have no categorical observation. Their payload must not
        # enter log-softmax, one-hot construction, or their backward operations.
        tokens = torch.where(mask, target.tokens, 0)
        polymer_type = torch.where(mask, polymer_type_index, 0)
        _require(
            ((tokens >= 0) & (tokens < pred.logits.shape[-1])).all(),
            "NA-MPNN scored sequence tokens are outside the model vocabulary",
        )
        _require(
            ((polymer_type >= _POLYMER_PROTEIN) & (polymer_type <= _POLYMER_RNA)).all(),
            "NA-MPNN scored polymer types are outside the model vocabulary",
        )
        log_probabilities = model_log_probabilities(pred.logits, mask, feature="NA-MPNN scored logits")
        per_residue = smoothed_loss(
            tokens,
            log_probabilities,
            polymer_type,
            weight=self.label_smoothing,
            shared_nucleic_tokens=self.shared_nucleic_tokens,
        )
        mask = mask.to(per_residue.dtype)
        values = per_residue.concat if isinstance(per_residue, NestedTensor) else per_residue
        weights = mask.concat if isinstance(mask, NestedTensor) else mask
        return weighted_mean(values.reshape(-1), weights.reshape(-1))
