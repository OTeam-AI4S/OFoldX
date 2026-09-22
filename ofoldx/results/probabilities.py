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
Autoregressive probability result carrier.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from ..data.molecular import ComponentRef, MolecularSystem
from ..data.structure import Structure


@dataclass(eq=False)
class AutoregressiveProbabilityResult:
    r"""
    Probability evaluations from a structure-conditioned autoregressive model.

    Attributes:
        target: Structure used for conditioning.
        component_references: Component occurrence for each sequence row; ``None`` marks
            padding or an unresolved row.
        sequence_tokens: Input token IDs with shape ``(num_rows,)``.
        residue_mask: Valid-row mask with shape ``(num_rows,)``.
        design_mask: Designed-position mask with shape ``(num_rows,)``.
        logits: Vocabulary logits with shape
            ``(num_evaluations, num_rows, vocab_size)``.
        decoding_ranks: Autoregressive ranks with shape
            ``(num_evaluations, num_rows)`` for whole-sequence evaluations.
            Per-row conditional and unconditional evaluations retain the
            effective order for every target and have shape
            ``(num_evaluations, num_target_rows, num_rows)``.
    """

    target: Structure
    component_references: tuple[ComponentRef | None, ...]
    sequence_tokens: Tensor
    residue_mask: Tensor
    design_mask: Tensor
    logits: Tensor
    decoding_ranks: Tensor

    @property
    def log_probabilities(self) -> Tensor:
        r"""Return log probabilities for every evaluation."""

        return self.logits.log_softmax(dim=-1)

    @property
    def probabilities(self) -> Tensor:
        r"""Return probabilities for every evaluation."""

        return self.logits.softmax(dim=-1)

    @property
    def mean_probabilities(self) -> Tensor:
        r"""Return probabilities averaged over evaluations."""

        return self.probabilities.mean(dim=0)

    @property
    def probability_standard_deviation(self) -> Tensor:
        r"""Return population standard deviations over evaluations."""

        return self.probabilities.std(dim=0, correction=0)


def component_references(system: MolecularSystem, component_index: Tensor) -> tuple[ComponentRef | None, ...]:
    r"""Bind ordered canonical component indices to occurrences, preserving unresolved rows (``-1``) as ``None``."""

    index = component_index.detach().cpu().numpy()
    refs = iter(system.component_refs(index[index >= 0]))
    return tuple(None if component < 0 else next(refs) for component in index.tolist())


def bind_autoregressive_probabilities(
    target: Structure,
    component_index: Tensor,
    sequence_tokens: Tensor,
    residue_mask: Tensor,
    design_mask: Tensor,
    logits: Tensor,
    decoding_ranks: Tensor,
) -> AutoregressiveProbabilityResult:
    r"""Bind one prepared sequence axis to its unmodified autoregressive evaluations."""

    return AutoregressiveProbabilityResult(
        target=target,
        component_references=component_references(target.system, component_index[0]),
        sequence_tokens=sequence_tokens[0],
        residue_mask=residue_mask[0],
        design_mask=design_mask[0],
        logits=logits,
        decoding_ranks=decoding_ranks,
    )
