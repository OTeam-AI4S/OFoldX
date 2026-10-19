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

"""Behaviour of ofoldx.msa.alignment: the row identity and the token spreading of a plan."""

from __future__ import annotations

import numpy as np
import pytest

from ofoldx.data.tokenization import TOKENS
from ofoldx.errors import ContradictionError, InvalidRequestError
from ofoldx.msa import MSA, MsaCollection, msa_rows, unique_rows


class TestUniqueRows:
    def test_first_occurrence_order(self):
        rows = np.array(["MK", "M-", "MK", "QK", "M-"], dtype="S")
        assert unique_rows(rows).tolist() == [0, 1, 3]

    def test_exclude_removes_rows(self):
        rows = np.array(["MK", "M-", "QK"], dtype="S")
        assert unique_rows(rows, exclude=np.array(["QK", "MK"], dtype="S")).tolist() == [1]


def names(rows):
    return [[TOKENS[token] for token in row] for row in rows]


class TestMsaRows:
    @pytest.fixture
    def system(self, polymers):
        return polymers(("A", "1", ("MET", "LYS")), ("L", "2", ("HEM",)))

    @pytest.fixture
    def plan(self):
        deletions = np.array([[0, 0], [0, 0], [0, 3]], dtype=np.uint32)
        msa = MSA("1", ("MK", "M-", "QK"), deletions, ("MK",), np.zeros((1, 2), dtype=np.uint32))
        return MsaCollection((msa,), np.array([[0], [1], [2]]), np.array([[True], [False], [False]]))

    def test_spreads_plan(self, system, plan, layout):
        rows = msa_rows(plan, system, layout(system))
        assert names(rows.token) == [["M", "K", "OTHER", "OTHER"], ["M", "-", "-", "-"], ["Q", "K", "-", "-"]]
        assert rows.deletions.tolist() == [[0, 0, 0, 0], [0, 0, 0, 0], [0, 3, 0, 0]]
        assert rows.paired.tolist() == [[True] * 4, [False] * 4, [False] * 4]

    def test_per_atom_tokens(self, system, plan, layout):
        rows = msa_rows(plan, system, layout(system, per_atom=[1]))
        assert names(rows.token)[2] == ["Q", "K", "K", "-", "-"]

    def test_hybrid_letters(self, polymers, pool, layout):
        system = polymers(("H", "1", ("A", "DA")))
        plan = MsaCollection((pool("1", "AA", "UU"),), np.array([[0], [1]]), np.array([[True], [False]]))
        assert names(msa_rows(plan, system, layout(system)).token) == [["RNA_A", "DNA_A"], ["RNA_U", "DNA_T"]]

    @pytest.mark.parametrize(
        ("entity", "query", "error", "code"),
        [
            ("1", "MKV", ContradictionError, "msa_length_mismatch"),
            ("1", "MR", ContradictionError, "msa_query_mismatch"),
            ("9", "MK", InvalidRequestError, "value_invalid"),
        ],
    )
    def test_binding_errors(self, system, pool, layout, entity, query, error, code):
        plan = MsaCollection((pool(entity, query),), np.array([[0]]), np.array([[True]]))
        with pytest.raises(error) as caught:
            msa_rows(plan, system, layout(system))
        assert caught.value.code == code
