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

"""Behavior of ofoldx.models.odesign.model."""

from __future__ import annotations

import pytest
import torch

from ofoldx.data.tokenization import TOKEN_INDEX
from ofoldx.errors import InvalidRequestError
from ofoldx.models.odesign.model import ODESIGN_SOURCE_TOKEN_INDEX, normalize_odesign_input, odesign_source_map
from ofoldx.models.odesign.processor import ODesignProcessor


class TestSourceMap:
    @pytest.mark.parametrize(
        "token,expected",
        [
            ("A", {"A": 1.0}),
            ("OTHER", {"X": 1.0}),
            ("RNA_X", {"RNA_N": 1.0}),
            ("B", {"D": 0.5, "N": 0.5}),
            ("<mask>", {}),
        ],
    )
    def test_canonical_columns(self, token, expected):
        column = odesign_source_map()[:, TOKEN_INDEX[token]]
        rows = {name: float(column[row]) for name, row in ODESIGN_SOURCE_TOKEN_INDEX.items() if column[row]}
        assert rows == expected


class TestNormalize:
    def test_masked_msa_projected(self, peptide):
        prepared = ODesignProcessor().prepare_input(peptide("ALA", "GLY"))
        input = prepared.features
        input.msa_mask = torch.zeros_like(input.msa_mask)
        input.msa_tokens = torch.full_like(input.msa_tokens, -10000)
        normalized = normalize_odesign_input(input)
        assert int(normalized.msa_tokens.max()) == 0
        assert not bool(normalized.msa_mask.any())

    def test_fixed_atom_needs_observation(self, peptide):
        prepared = ODesignProcessor().prepare_input(peptide("ALA", "GLY"))
        input = prepared.features
        input.fixed_coordinate_mask = torch.ones_like(input.fixed_coordinate_mask)
        input.observed_mask = torch.zeros_like(input.observed_mask)
        with pytest.raises(InvalidRequestError) as caught:
            normalize_odesign_input(input)
        assert caught.value.refs[0].argument == "fixed_atoms"
