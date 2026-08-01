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

"""Behavior of ofoldx.models.nampnn.objective."""

import pytest
import torch

from ofoldx.data.tokenization import TOKEN_INDEX, TOKENS
from ofoldx.errors import InvalidRequestError
from ofoldx.models.nampnn import NaMpnnLoss, NaMpnnOutput, NaMpnnProcessor, smoothed_loss

PROTEIN = [TOKEN_INDEX[letter] for letter in "ARNDCQEGHILKMFPSTWYVX"]


@pytest.fixture
def target(peptide):
    structure = peptide("ALA")
    prepared, truth = NaMpnnProcessor().prepare(structure, target=structure)
    return prepared.features, truth


def protein_logits() -> torch.Tensor:
    logits = torch.full((1, 1, len(TOKENS)), -torch.inf, dtype=torch.float64)
    logits[..., PROTEIN] = 0.0
    return logits


class TestLoss:
    def test_exclusions_have_no_mass(self, target):
        input, truth = target
        logits = protein_logits().requires_grad_()
        loss = NaMpnnLoss()(
            NaMpnnOutput(logits=logits, decoding_ranks=input.tied_group_ids),
            truth,
            polymer_type_index=input.polymer_type_index,
            mask=input.residue_mask & input.design_mask,
        )
        torch.testing.assert_close(loss, logits.new_tensor(len(PROTEIN)).log())
        gradient = torch.autograd.grad(loss, logits)[0]
        assert torch.isfinite(gradient).all()
        assert not bool(gradient[..., TOKEN_INDEX["DNA_A"]].any())

    def test_excluded_target_refused(self, target):
        input, truth = target
        logits = protein_logits()
        logits[..., TOKEN_INDEX["A"]] = -torch.inf
        with pytest.raises(InvalidRequestError, match="target mass"):
            NaMpnnLoss()(
                NaMpnnOutput(logits=logits, decoding_ranks=input.tied_group_ids),
                truth,
                polymer_type_index=input.polymer_type_index,
                mask=input.residue_mask & input.design_mask,
            )


class TestSharedRna:
    @pytest.fixture
    def arguments(self):
        log_probabilities = torch.full((1, len(TOKENS)), -torch.inf, dtype=torch.float64)
        dna = [TOKEN_INDEX[f"DNA_{base}"] for base in "ACGTX"]
        log_probabilities[0, dna] = log_probabilities.new_tensor(1 / 5).log()
        return torch.tensor([TOKEN_INDEX["DNA_A"]]), log_probabilities, torch.tensor([2])

    def test_shared_smoothing(self, arguments):
        loss = smoothed_loss(*arguments, shared_nucleic_tokens=True)
        torch.testing.assert_close(loss, arguments[1].new_tensor([5.0]).log())

    def test_separate_columns_refused(self, arguments):
        with pytest.raises(InvalidRequestError, match="target mass"):
            smoothed_loss(*arguments, shared_nucleic_tokens=False)


def test_zero_soft_target_derivative():
    log_probabilities = torch.arange(len(TOKENS), dtype=torch.float64).neg().log_softmax(-1).unsqueeze(0)
    probabilities = torch.zeros_like(log_probabilities)
    probabilities[0, 0] = 1
    probabilities.requires_grad_()
    loss = smoothed_loss(
        torch.tensor([0]),
        log_probabilities,
        torch.tensor([0]),
        weight=0,
        target_probabilities=probabilities,
        target_probability_mask=torch.tensor([True]),
    )
    gradient = torch.autograd.grad(loss.sum(), probabilities)[0]
    torch.testing.assert_close(gradient, -log_probabilities)
