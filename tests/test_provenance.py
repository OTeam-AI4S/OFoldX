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

import itertools

import numpy as np
import pytest

from ofoldx.errors import CODE_NAMES, CODES, Cause, InvalidRequestError, RecordKind, SourceRef
from ofoldx.provenance import (
    ALLOWED_STATES,
    Basis,
    Record,
    Records,
    Resolution,
    SourceRefs,
    SubjectKind,
    basis,
    determined,
    known,
    resolution,
    state,
    states,
)

R, B = Resolution, Basis
# the legal (resolution, basis) pairs
LEGAL = {
    (R.KNOWN, B.DECLARED),
    (R.KNOWN, B.DEFAULT),
    (R.KNOWN, B.DEFINED),
    (R.KNOWN, B.DERIVED),
    (R.KNOWN, B.REFERENCE),
    (R.KNOWN, B.ASSUMED),
    (R.UNKNOWN, B.NONE),
    (R.UNKNOWN, B.DECLARED),
    (R.UNKNOWN, B.DEFINED),
    (R.UNKNOWN, B.DERIVED),
    (R.UNKNOWN, B.REFERENCE),
    (R.INAPPLICABLE, B.DECLARED),
    (R.INAPPLICABLE, B.DEFINED),
    (R.CONFLICT, B.NONE),
}


def source_refs(rows, block="1abc"):
    """``rows`` of ``(category, row, item)`` as one table."""
    categories = tuple(dict.fromkeys(r[0] for r in rows))
    items = tuple(dict.fromkeys(r[2] for r in rows if r[2] is not None))
    return SourceRefs(
        np.array([categories.index(r[0]) for r in rows], dtype=np.int16),
        np.array([r[1] for r in rows], dtype=np.int64),
        np.array([-1 if r[2] is None else items.index(r[2]) for r in rows], dtype=np.int16),
        categories,
        items,
        block,
    )


def records(rows, refs, texts=()):
    """``rows`` of ``(code, subject kind, subject, ref rows, payload slots)`` as columns."""
    payload = np.zeros((len(rows), 4), dtype=np.int64)
    for index, row in enumerate(rows):
        payload[index, : len(row[4])] = row[4]
    return Records(
        np.array([CODE_NAMES.index(r[0]) for r in rows], dtype=np.uint16),
        np.array([r[1] for r in rows], dtype=np.uint8),
        np.array([r[2] for r in rows], dtype=np.int64),
        np.cumsum([0, *(len(r[3]) for r in rows)]).astype(np.int64),
        np.array([i for r in rows for i in r[3]], dtype=np.int64),
        refs,
        payload,
        np.zeros(len(rows) + 1, dtype=np.int64),
        np.zeros(0, dtype=np.int64),
        tuple(texts),
    )


@pytest.fixture
def refs():
    return source_refs([("struct_conn", 3, "ptnr1_label_atom_id"), ("atom_site", 17, None)])


@pytest.fixture
def table(refs):
    return records(
        [
            ("observed_leaving_atom", SubjectKind.ATOM, 5, (0, 1), (5, 2, 17, 1)),
            ("port_unverified", SubjectKind.COMPONENT, 1, (), (1, 0, 1)),
            ("order_unknown", SubjectKind.EDGE, 0, (0,), ()),
            ("observed_leaving_atom", SubjectKind.ATOM, 7, (1,), (7, 0, 17, 0)),
        ],
        refs,
        texts=("next", "7WC"),
    )


class TestState:
    def test_allowed_states(self):
        assert ALLOWED_STATES == {r << 3 | b for r, b in LEGAL}

    def test_state_packs_legal_pairs(self):
        for r, b in itertools.product(Resolution, Basis):
            if (r, b) in LEGAL:
                assert state(r, b) == r << 3 | b
            else:
                with pytest.raises(AssertionError):
                    state(r, b)

    def test_states_round_trip(self):
        pairs = sorted(LEGAL)
        packed = states(np.array([r for r, _ in pairs], np.uint8), np.array([b for _, b in pairs], np.uint8))
        assert packed.tolist() == [state(r, b) for r, b in pairs]
        assert resolution(packed).tolist() == [r for r, _ in pairs]
        assert basis(packed).tolist() == [b for _, b in pairs]

    @pytest.mark.parametrize(("r", "b"), [(R.KNOWN, B.NONE), (R.CONFLICT, B.DECLARED), (R.UNKNOWN, 9)])
    def test_states_rejects_illegal(self, r, b):
        with pytest.raises(AssertionError):
            states(np.array([r], np.uint8), np.array([b], np.uint8))

    def test_known_and_determined(self):
        pairs = [(R.KNOWN, B.DECLARED), (R.KNOWN, B.ASSUMED), (R.UNKNOWN, B.DECLARED), (R.INAPPLICABLE, B.DEFINED)]
        packed = np.array([state(r, b) for r, b in pairs], dtype=np.uint8)
        assert known(packed).tolist() == [True, True, False, False]
        assert determined(packed).tolist() == [True, False, False, False]


class TestSourceRefs:
    def test_ref(self, refs):
        assert len(refs) == 2
        assert refs.ref(0) == SourceRef("struct_conn", 3, "ptnr1_label_atom_id", "1abc")
        assert refs.ref(1) == SourceRef("atom_site", 17, None, "1abc")

    def test_concat_maps_every_row(self, refs):
        other = source_refs([("atom_site", 2, "Cartn_x"), ("pdbx_struct_oper_list", 0, None)])
        joined, maps = SourceRefs.concat([refs, other])
        assert [m.tolist() for m in maps] == [[0, 1], [2, 3]]
        for part, rows in zip((refs, other), maps):
            assert [joined.ref(row) for row in rows] == [part.ref(i) for i in range(len(part))]

    def test_concat_refuses_two_blocks(self, refs):
        with pytest.raises(InvalidRequestError):
            SourceRefs.concat([refs, source_refs([("atom_site", 0, None)], block="2xyz")])


class TestRecords:
    def test_empty(self):
        assert len(Records.empty()) == 0 and dict(Records.empty().counts()) == {} and len(Records.concat([])) == 0

    def test_getitem_renders_detail(self, table, refs):
        assert table[0] == Record(
            "observed_leaving_atom",
            RecordKind.CONFLICT,
            Cause.NONE,
            (SubjectKind.ATOM, 5),
            (refs.ref(0), refs.ref(1)),
            {"atom": 5, "declaration": 2, "atom_site_row": 17, "carbonyl_o_observed": 1},
        )
        assert table[1].detail == {"id": "7WC", "side": "next", "condition": 1}
        assert table[-2].detail == {} and table[-2].sources == (refs.ref(0),)
        with pytest.raises(IndexError):
            table[4]

    def test_code_slot_renders_name(self, refs):
        cause = CODE_NAMES.index("order_unknown")
        table = records([("prepare_accepted_incomplete", SubjectKind.COMPONENT, 0, (), (cause,))], refs)
        assert table[0].detail == {"cause": "order_unknown"}

    def test_kind_and_cause(self, table):
        names = [CODE_NAMES[c] for c in table.code]
        assert table.kind().tolist() == [CODES[n].kind for n in names]
        assert table.cause().tolist() == [CODES[n].cause for n in names]

    @pytest.mark.parametrize(
        ("filters", "subjects"),
        [
            ({"kind": RecordKind.CONFLICT}, [5, 7]),
            ({"code": "order_unknown"}, [0]),
            ({"subject_kind": SubjectKind.COMPONENT}, [1]),
            ({"kind": RecordKind.UNKNOWN, "subject_kind": SubjectKind.ATOM}, []),
        ],
    )
    def test_where(self, table, filters, subjects):
        selected = table.where(**filters)
        assert list(selected) == [record for record in table if record.subject[1] in subjects]

    @pytest.mark.parametrize("code", ["format_unknown", "no_such_code"])
    def test_where_refuses_unrecorded_code(self, table, code):
        with pytest.raises(InvalidRequestError):
            table.where(code=code)

    def test_counts(self, table):
        assert dict(table.counts()) == {"observed_leaving_atom": 2, "order_unknown": 1, "port_unverified": 1}

    def test_concat_keeps_every_record(self, table):
        other = records(
            [
                ("port_unverified", SubjectKind.COMPONENT, 0, (0,), (0, 1, 2)),
                ("contact_row", SubjectKind.NONE, -1, (), ()),
            ],
            source_refs([("struct_conn", 9, None)]),
            texts=("unused", "previous"),
        )
        assert list(Records.concat([table, Records.empty(), other])) == [*table, *other]

    @pytest.mark.parametrize(
        ("axis", "old_to_new", "subjects", "atoms"),
        [
            # a removed subject keeps its record with -1; the atom slot follows the atom axis
            (SubjectKind.ATOM, [0, 0, 0, 0, 0, 2, 0, -1], [2, 1, 0, -1], [2, None, None, -1]),
            (SubjectKind.COMPONENT, [3, -1], [5, -1, 0, 7], [5, None, None, 7]),
        ],
    )
    def test_remap(self, table, axis, old_to_new, subjects, atoms):
        remapped = table.remap(axis, np.array(old_to_new, dtype=np.int64))
        assert remapped.subject.tolist() == subjects
        assert [record.detail.get("atom") for record in remapped] == atoms
