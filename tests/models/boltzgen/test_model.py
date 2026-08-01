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

"""Behavior of ofoldx.models.boltzgen.model."""

from __future__ import annotations

import pytest
import torch

from ofoldx.data.tokenization import TOKENS
from ofoldx.errors import InvalidRequestError
from ofoldx.models.boltzgen.model import BoltzGenInverseFoldingModel, normalize_boltzgen_input
from ofoldx.models.boltzgen.processor import BoltzGenInverseFoldingProcessor, BoltzGenProcessor


class TestNeighborGraph:
    @pytest.fixture
    def graph(self, inverse_folding_config, peptide):
        model = BoltzGenInverseFoldingModel(inverse_folding_config).double().eval()
        input = (
            BoltzGenInverseFoldingProcessor(dtype=torch.float64).prepare_input(peptide("ALA", "GLY", "LYS")).features
        )
        return model.inverse_folding_encoder.build_knn_graph(input)

    def test_neighbors_of_every_token(self, graph):
        edge_index, valid_mask, _ = graph
        assert valid_mask[0].tolist() == [True, True, True]
        assert edge_index.shape == (2, 6)

    def test_self_loops_kept(self, graph):
        edge_index, _, _ = graph
        pairs = set(zip(*edge_index.tolist()))
        assert {(0, 0), (1, 1), (2, 2)} <= pairs

    def test_bonded_neighbors(self, graph):
        edge_index, _, bond = graph
        pairs = list(zip(*edge_index.tolist()))
        assert bond.shape == (len(pairs), 8)
        bonded = {pair for pair, value in zip(pairs, bond[:, 0].tolist()) if value}
        assert bonded and all(abs(source - destination) == 1 for source, destination in bonded)


def test_msa_input_availability(peptide):
    value = BoltzGenProcessor().prepare_input(peptide("ALA", "GLY")).features
    value.msa_tokens = torch.tensor([[[1, -10000]]])
    value.msa_mask = torch.tensor([[[True, False]]])
    value.token_mask = torch.tensor([[True, False]])
    value.has_deletion = torch.tensor([[[1.0, torch.nan]]])
    value.deletion_value = torch.tensor([[[0.5, torch.nan]]])
    value.msa_paired = torch.tensor([[[0.0, torch.nan]]])
    normalized = normalize_boltzgen_input(value, dtype=torch.float16)
    assert normalized.msa_tokens.tolist() == [[[1, 0]]]
    assert normalized.msa_profile.shape[-1] == len(TOKENS)
    assert normalized.deletion_value.dtype == torch.float16
    assert normalized.deletion_value.tolist() == [[[0.5, 0.0]]]
    value.msa_tokens[0, 0, 0] = -1
    with pytest.raises(InvalidRequestError):
        normalize_boltzgen_input(value)
