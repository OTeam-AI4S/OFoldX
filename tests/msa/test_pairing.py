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

"""Behaviour of ofoldx.msa.pairing: the one pairing rule."""

from __future__ import annotations

import numpy as np
import pytest

from ofoldx.data.tokenization import select_tokens
from ofoldx.msa import pair_msas

HUMAN, MOUSE, RAT = "tr|A0A000|X_HUMAN", "tr|A0A001|X_MOUSE", "tr|A0A002|X_RAT"


@pytest.fixture
def complex_(polymers):
    """Proteins A (MK), B (GS), C (a copy of A), D (PW), RNA R (AC) and a ligand L."""
    return polymers(
        ("A", "1", ("MET", "LYS")),
        ("B", "2", ("GLY", "SER")),
        ("C", "1", ("MET", "LYS")),
        ("D", "5", ("PRO", "TRP")),
        ("R", "3", ("A", "C")),
        ("L", "4", ("HEM",)),
    )


class TestPairMsas:
    @pytest.mark.parametrize(
        ("first", "second", "pairs"),
        [
            pytest.param("UniRef100_P04166_10116/", "UniRef100_Q9XYZ1_10116/", True, id="corpus-taxid"),
            pytest.param("UniRef100_P04166_10116/", "UniRef100_Q9XYZ1_9606/", False, id="other-taxon"),
            pytest.param("UniRef100_P04166_0/", "UniRef100_Q9XYZ1_0/", False, id="taxid-zero"),
            pytest.param("tr|A0A000|FOO_HUMAN", "sp|P12345|BAR_HUMAN/3-40 text", True, id="mnemonic"),
            pytest.param("tr|A0A000|FOO_12345", "tr|B0B000|BAR_12345", False, id="mnemonic-without-letter"),
            pytest.param("UniRef100_P04166_9606", "tr|B0B000|BAR_HUMAN", False, id="never-cross"),
            pytest.param("uniref100_p04166_9606", "uniref100_q9xyz1_9606", False, id="case-kept"),
            pytest.param("UniRef100_P04166_HUMAN", "UniRef100_Q9XYZ1_HUMAN", False, id="named-taxon"),
        ],
    )
    def test_species_key(self, complex_, pool, layout, first, second, pairs):
        pools = {"A": pool("1", "MK", paired=[(first, "MR")]), "B": pool("2", "GS", paired=[(second, "GT")])}
        collection = pair_msas(pools, complex_, layout(complex_))
        assert int(collection.row_is_paired.all(axis=1).sum()) == (2 if pairs else 1)

    def test_copies_do_not_pair(self, complex_, pool, layout):
        pools = {name: pool("1", "MK", paired=[(HUMAN, "MR")]) for name in ("A", "C")}
        collection = pair_msas(pools, complex_, layout(complex_))
        assert collection.row_is_paired[1:].sum() == 0

    def test_one_row_per_species(self, complex_, pool, layout):
        pools = {
            "A": pool("1", "MK", paired=[(HUMAN, "MR"), (HUMAN, "MQ")]),
            "B": pool("2", "GS", paired=[(HUMAN, "GT")]),
        }
        collection = pair_msas(pools, complex_, layout(complex_))
        assert collection.row_pairing[collection.row_is_paired.all(axis=1)].tolist() == [[0, 0], [1, 1]]

    def test_block_order(self, complex_, pool, layout):
        species = [(HUMAN, "A"), (MOUSE, "C"), (RAT, "D")]
        pools = {
            "A": pool("1", "MK", paired=[(h, "M" + r) for h, r in species]),
            "B": pool("2", "GS", paired=[(h, "G" + r) for h, r in species]),
            "D": pool("5", "PW", paired=[(MOUSE, "PA")]),
        }
        collection = pair_msas(pools, complex_, layout(complex_))
        assert collection.target_instance_ids == ("A", "B", "D")
        assert collection.row_pairing[:4].tolist() == [[0, 0, 0], [2, 2, 1], [1, 1, -1], [3, 3, -1]]

    def test_protein_only(self, complex_, pool, layout):
        pools = {
            "A": pool("1", "MK", paired=[(HUMAN, "MR")]),
            "B": pool("2", "GS", paired=[(HUMAN, "GT")]),
            "R": pool("3", "AC", paired=[(HUMAN, "AG")]),
        }
        collection = pair_msas(pools, complex_, layout(complex_))
        assert collection.row_pairing.tolist() == [[0, 0, 0], [1, 1, -1]]

    def test_crop_selects_participants(self, complex_, pool, layout):
        pools = {"A": pool("1", "MK"), "B": pool("2", "GS")}
        tokens = layout(complex_)
        cropped = select_tokens(tokens, ~np.isin(tokens.token_component, [2, 3]))  # B's components
        assert pair_msas(pools, complex_, cropped).target_instance_ids == ("A",)

    def test_depth_stack(self, complex_, pool, layout):
        pools = {
            "A": pool("1", "MK", "MA", "MA", "MR", paired=[(HUMAN, "MA")]),
            "B": pool("2", "GS", "GT", "GV", paired=[(HUMAN, "GA")]),
        }
        collection = pair_msas(pools, complex_, layout(complex_))
        assert collection.row_pairing.tolist() == [[0, 0], [1, 1], [3, 1], [-1, 2]]
        assert collection.row_is_paired.tolist() == [[True, True], [True, True], [False, False], [False, False]]
        assert collection.msas == (pools["A"], pools["B"])

    @pytest.mark.parametrize(("max_rows", "paired"), [(None, True), (2, False)])
    def test_max_rows_halves_block(self, complex_, pool, layout, max_rows, paired):
        pools = {"A": pool("1", "MK", "MA", paired=[(HUMAN, "MA")]), "B": pool("2", "GS", "GA", paired=[(HUMAN, "GA")])}
        collection = pair_msas(pools, complex_, layout(complex_), max_rows=max_rows)
        assert collection.row_pairing.tolist() == [[0, 0], [1, 1]]
        assert collection.row_is_paired[1].tolist() == [paired, paired]

    def test_max_rows_cuts_plan(self, complex_, pool, layout):
        collection = pair_msas({"A": pool("1", "MK", "MA", "MC", "MD", "ME")}, complex_, layout(complex_), max_rows=3)
        assert collection.row_pairing.tolist() == [[0], [1], [2]]

    def test_no_participant(self, complex_, pool, layout):
        assert pair_msas({"Z": pool("9", "MK")}, complex_, layout(complex_)) is None
