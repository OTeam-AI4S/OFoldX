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

"""Behaviour of ofoldx.io.cif: CIF text as typed columns with lexical states."""

import gzip

import numpy as np
import pytest

from ofoldx.errors import ContradictionError, InvalidRequestError, UnsupportedError
from ofoldx.io.cif import TokenState, integer_tokens, parse_cif, read_cif

LOOP = """data_x
loop_
_a.item
_a.number
1 '1.25(3)'
? 1.2e3(4)
. -0.5
'?' 12
"""


@pytest.fixture
def category():
    return parse_cif(LOOP).block("x").categories["a"]


class TestColumns:
    def test_lexical_states(self, category):
        values, states = category.text("item")
        assert states.tolist() == [TokenState.VALUE, TokenState.UNKNOWN, TokenState.INAPPLICABLE, TokenState.VALUE]
        assert values.tolist() == [b"1", b"", b"", b"?"]

    def test_absent_item(self, category):
        assert (category.text("missing")[1] == TokenState.ABSENT).all()

    def test_names_case_insensitive(self, category):
        assert category.text("ITEM")[0][0] == b"1"

    def test_codes_intern_values(self, category):
        codes, _, symbols = category.codes("item")
        assert [symbols[c] if c >= 0 else None for c in codes] == ["1", None, None, "?"]

    def test_numbers_uncertainty_quantum(self):
        value, state, uncertainty, quantum = (
            parse_cif("data_x\nloop_\n_a.v\n1.25(3)\n1.2e3(4)\n-0.5\n").blocks[0].categories["a"].floats("v")
        )
        np.testing.assert_allclose(value, [1.25, 1200.0, -0.5])
        np.testing.assert_allclose(uncertainty, [0.03, 400.0, np.nan])
        np.testing.assert_allclose(quantum, [0.01, 100.0, 0.1])

    @pytest.mark.parametrize(
        ("token", "number", "su"),
        [
            ("1.0(1)e-308", "1.0e-308", "1e-309"),  # a subnormal su
            ("1(1)e-323", "1e-323", "1e-323"),
            ("1.79(1)e308", "1.79e308", "1e306"),
            ("0.57(57)", "0.57", "0.57"),  # 57 * 0.01 is not the double nearest 0.57
            ("1.2(123456789012345678)", "1.2", "12345678901234567.8"),  # su digits beyond 2**53
        ],
    )
    def test_numbers_nearest_decimal(self, token, number, su):
        value, _, uncertainty, _ = parse_cif(f"data_x\n_a.v {token}\n").blocks[0].categories["a"].floats("v")
        assert (value.tolist(), uncertainty.tolist()) == ([float(number)], [float(su)])

    @pytest.mark.parametrize("token", ["1e309", "2.5e-400", "1(2)e308"])
    def test_numbers_beyond_float64_refused(self, token):
        category = parse_cif(f"data_x\n_a.v {token}\n").blocks[0].categories["a"]
        with pytest.raises(UnsupportedError, match="numeric_range"):
            category.floats("v")

    @pytest.mark.parametrize("token", ["1.2.3", "nan", "1(2)e3(4)", "''"])
    def test_number_grammar(self, token):
        category = parse_cif(f"data_x\n_a.v {token}\n").blocks[0].categories["a"]
        with pytest.raises(ContradictionError, match="number_grammar"):
            category.floats("v")

    @pytest.mark.parametrize("token", ["9223372036854775807", "-9223372036854775808", "+0001000000000000000000"])
    def test_int64_bounds_read(self, token):
        category = parse_cif(f"data_x\n_a.v {token}\n").blocks[0].categories["a"]
        assert category.ints("v")[0].tolist() == [int(token)]

    @pytest.mark.parametrize("token", ["9223372036854775808", "-9223372036854775809"])
    def test_beyond_int64_refused(self, token):
        category = parse_cif(f"data_x\n_a.v {token}\n").blocks[0].categories["a"]
        with pytest.raises(UnsupportedError, match="numeric_range"):
            category.ints("v")

    def test_integer_readings(self):  # the grammar ints() and an author number read by, without refusing
        value, grammatical, readable = integer_tokens(np.array([b"+7", b"-0012", b"1a", b"9223372036854775808"]))
        assert value.tolist() == [7, -12, 0, 0]
        assert grammatical.tolist() == [True, True, False, True] and readable.tolist() == [True, True, False, False]


class TestFile:
    def test_gzip_by_content(self, tmp_path):
        path = tmp_path / "plain.cif"
        path.write_bytes(gzip.compress(LOOP.encode()))
        assert read_cif(path).blocks[0].name == "x"

    def test_block_lookup(self):
        cif = parse_cif("data_one\n_a.v 1\ndata_Two\n_a.v 2\n")
        assert cif.block("two").categories["a"].text("v")[0][0] == b"2"
        with pytest.raises(InvalidRequestError, match="block_unknown"):
            cif.block("three")

    def test_save_frame_refused(self):
        with pytest.raises(UnsupportedError, match="cif_frame_unsupported"):
            parse_cif("data_x\nsave_f\n_a.v 1\nsave_\n")

    def test_category_split_refused(self):
        with pytest.raises(ContradictionError, match="cif_loop_count"):
            parse_cif("data_x\n_a.v 1\nloop_\n_a.w\n1\n2\n")

    @pytest.mark.parametrize(
        "text",
        [
            "global_\n_a.v 1\n",
            "data_x\n_a_v 1\n",
            "data_x\nloop_\n_a_v\n_a_w\n1 2\n",
            "data_x\nloop_\n_a.v\n_b.w\n1 2\n",
        ],
    )
    def test_non_mmcif_form_refused(self, text):
        with pytest.raises(UnsupportedError, match="cif_form_unsupported"):
            parse_cif(text)
