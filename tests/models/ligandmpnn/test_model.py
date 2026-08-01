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

"""Behavior of ofoldx.models.ligandmpnn.model."""

from __future__ import annotations

import pytest
import torch

from ofoldx.data.tokenization import TOKEN_INDEX
from ofoldx.errors import InvalidRequestError
from ofoldx.models.ligandmpnn import LigandMpnnFeatureEncoder
from ofoldx.models.ligandmpnn.model import LIGANDMPNN_GENERATABLE


@pytest.fixture
def case(designer, peptide):
    input = designer.processor.prepare_input(peptide("ALA", "GLY", "LYS", ligands=("ACT",))).features
    input.residue_mask[0, -1] = False
    return designer.model, input


RANKS = torch.arange(3).unsqueeze(0)


class TestLogits:
    def test_untrained_classes_excluded(self, case):
        model, input = case
        logits = model(input, decoding_ranks=RANKS).logits
        assert torch.isneginf(logits[..., TOKEN_INDEX["OTHER"]]).all()
        assert torch.isfinite(logits[..., TOKEN_INDEX["X"]]).all()

    def test_sampling_omits_unknown(self, case):
        model, input = case
        sampled = model.sample(input, num_samples=8, temperature=10.0, generator=torch.Generator().manual_seed(3))
        generatable = [TOKEN_INDEX[token] for token in LIGANDMPNN_GENERATABLE]
        assert torch.isin(sampled.sequence_tokens[:, :2], torch.tensor(generatable)).all()

    @pytest.mark.parametrize("payload", [-1, 2**40])
    def test_sequence_categories(self, case, payload):
        model, input = case
        expected = model(input, decoding_ranks=RANKS).logits[input.residue_mask]
        input.sequence_tokens[~input.residue_mask] = payload
        actual = model(input, decoding_ranks=RANKS).logits[input.residue_mask]
        torch.testing.assert_close(actual, expected)
        input.sequence_tokens[0, 0] = payload
        with pytest.raises(InvalidRequestError, match="sequence tokens"):
            model(input, decoding_ranks=RANKS)

    def test_unavailable_context(self, case):
        model, input = case
        expected = model(input, decoding_ranks=RANKS).logits
        values = input.context_coordinates.detach().clone()
        values[~input.context_mask] = float("nan")
        values.requires_grad_()
        input.context_coordinates = values
        actual = model(input, decoding_ranks=RANKS).logits
        torch.testing.assert_close(actual, expected)
        finite = torch.where(torch.isfinite(actual), actual, 0.0)
        gradient = torch.autograd.grad(finite.square().sum(), values)[0]
        assert torch.isfinite(gradient).all()
        assert not gradient[~input.context_mask].any()


def test_invalid_element_refused(case):
    _, input = case
    encoder = LigandMpnnFeatureEncoder(hidden_size=8, num_neighbors=2).eval()
    input.context_mask[0, 0, 0] = True
    input.context_atomic_numbers[0, 0, 0] = 120
    with pytest.raises(InvalidRequestError, match="element IDs"):
        encoder(
            input.backbone_coordinates,
            input.residue_mask,
            input.residue_index,
            input.chain_index,
            input.context_coordinates,
            input.context_atomic_numbers,
            input.context_mask,
            input.residue_context_coordinates,
            input.residue_context_atomic_numbers,
            input.residue_context_mask,
            input.design_mask,
            use_side_chain_context=True,
        )
