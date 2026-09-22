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

from __future__ import annotations

import pytest
import torch

from ofoldx.data.molecular import ComponentRef
from ofoldx.results.probabilities import bind_autoregressive_probabilities


@pytest.mark.parametrize("rank_ndim", [2, 3])
def test_binding_and_derivatives(peptide, rank_ndim: int) -> None:
    target = peptide("ALA", "GLY", "ALA")
    components = torch.tensor([[2, -1, 0, 1, 2]])
    tokens = torch.tensor([[4, 3, 2, 1, 0]])
    residue_mask = torch.tensor([[True, False, True, True, True]])
    design_mask = torch.tensor([[False, False, True, True, False]])
    logits = torch.linspace(-2, 3, 2 * 5 * 7, dtype=torch.float64).reshape(2, 5, 7).requires_grad_()
    ranks = torch.arange(2 * 5).reshape(2, 5) if rank_ndim == 2 else torch.arange(2 * 5 * 5).reshape(2, 5, 5)
    result = bind_autoregressive_probabilities(target, components, tokens, residue_mask, design_mask, logits, ranks)
    assert result.target is target
    torch.testing.assert_close(result.logits, logits)
    torch.testing.assert_close(result.decoding_ranks, ranks)
    assert result.component_references == (
        ComponentRef("A", 2),
        None,
        ComponentRef("A", 0),
        ComponentRef("A", 1),
        ComponentRef("A", 2),
    )
    for output, operand in (
        (result.sequence_tokens, tokens),
        (result.residue_mask, residue_mask),
        (result.design_mask, design_mask),
    ):
        torch.testing.assert_close(output, operand[0])
    reference_log = logits - torch.logsumexp(logits, -1, keepdim=True)
    reference = reference_log.exp()
    torch.testing.assert_close(result.log_probabilities, reference_log)
    torch.testing.assert_close(result.probabilities, reference)
    torch.testing.assert_close(result.mean_probabilities, reference.mean(0))
    torch.testing.assert_close(
        result.probability_standard_deviation, ((reference - reference.mean(0)).square().mean(0)).sqrt()
    )
    coefficients = torch.linspace(0.3, 1.7, logits.numel(), dtype=logits.dtype).reshape_as(logits)
    derivative = torch.autograd.grad((result.probabilities * coefficients).sum(), logits)[0]
    torch.testing.assert_close(
        derivative, reference * (coefficients - (coefficients * reference).sum(-1, keepdim=True))
    )
