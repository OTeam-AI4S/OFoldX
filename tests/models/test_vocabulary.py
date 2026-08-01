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

"""Behavior of ofoldx.models.vocabulary."""

from __future__ import annotations

import pytest
import torch

from ofoldx.data.tokenization import TOKEN_INDEX, TOKENS
from ofoldx.models.vocabulary import (
    FOLDING_SOURCE_TOKEN_INDEX,
    canonical_word_embeddings,
    folding_source_map,
    folding_word_embeddings,
    map_token_feature_planes,
)

SOURCE = tuple("ACDEFGHIKLMNPQRSTVWYX")


class TestCanonicalWordEmbeddings:
    def test_known_rows_copied(self):
        rows = torch.randn(len(SOURCE), 4, dtype=torch.float64)
        (converted,) = canonical_word_embeddings(rows, vocabulary=SOURCE)
        assert converted.shape == (len(TOKENS), 4)
        torch.testing.assert_close(converted[[TOKEN_INDEX[token] for token in SOURCE]], rows)

    def test_ambiguity_code_mean(self):
        rows = torch.randn(len(SOURCE), 4)
        (converted,) = canonical_word_embeddings(rows, vocabulary=SOURCE)
        expected = (rows[SOURCE.index("D")] + rows[SOURCE.index("N")]) / 2
        torch.testing.assert_close(converted[TOKEN_INDEX["B"]], expected)

    def test_vocabulary_axis(self):
        weight = torch.randn(3, len(SOURCE))
        (converted,) = canonical_word_embeddings(weight, vocabulary=SOURCE, dim=-1)
        torch.testing.assert_close(converted[:, TOKEN_INDEX["W"]], weight[:, SOURCE.index("W")])


class TestFoldingWordEmbeddings:
    def test_other_reads_unknown(self):
        rows = torch.randn(len(FOLDING_SOURCE_TOKEN_INDEX), 4, dtype=torch.float64)
        (converted,) = folding_word_embeddings(rows)
        torch.testing.assert_close(converted[TOKEN_INDEX["OTHER"]], rows[FOLDING_SOURCE_TOKEN_INDEX["X"]])


class TestFoldingSourceMap:
    @pytest.mark.parametrize(
        "token,expected",
        [("A", {"A": 1.0}), ("B", {"D": 0.5, "N": 0.5}), ("OTHER", {"X": 1.0}), ("<null>", {}), ("DNA_X", None)],
    )
    def test_columns(self, token, expected):
        column = folding_source_map()[:, TOKEN_INDEX[token]]
        if expected is None:
            expected = dict.fromkeys(("DNA_A", "DNA_C", "DNA_G", "DNA_T"), 0.25)
        reference = torch.zeros(len(FOLDING_SOURCE_TOKEN_INDEX))
        for member, value in expected.items():
            reference[FOLDING_SOURCE_TOKEN_INDEX[member]] = value
        torch.testing.assert_close(column, reference)


def test_feature_planes(native_token_features, source_feature_planes):
    generator = torch.Generator().manual_seed(7)
    features = native_token_features(8, 5, 6, generator)
    mapped = map_token_feature_planes(features, folding_source_map().double(), 8)
    torch.testing.assert_close(mapped, source_feature_planes(features, 8))
