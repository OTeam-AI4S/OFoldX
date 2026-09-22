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

import ast
from pathlib import Path

import pytest

from ofoldx.errors import (
    CODE_NAMES,
    CODES,
    Cause,
    ContradictionError,
    Disposition,
    InsufficientInformationError,
    InvalidRequestError,
    OFoldXError,
    PayloadKind,
    RecordKind,
    RequestRef,
    ResourceLimitError,
    SourceRef,
    StructureRef,
    UnsupportedError,
    fail,
)
from ofoldx.provenance import Record, SubjectKind

PACKAGE = Path(__file__).resolve().parents[1] / "ofoldx"


class TestCodes:
    def test_names_are_sorted(self):
        assert CODE_NAMES == tuple(sorted(CODES))

    def test_rows_are_consistent(self):
        for name, code in CODES.items():
            assert code.name == name and code.basis and ".py" not in code.basis  # a source path goes stale
            if code.disposition is Disposition.RAISE:
                assert issubclass(code.error, OFoldXError) and code.kind is None and not code.payload
            else:
                assert code.error is None and code.kind is not None
            assert code.cause is Cause.NONE or code.kind is RecordKind.UNKNOWN
            assert len(code.payload) <= 4

    @pytest.mark.parametrize(
        ("name", "error"),
        [
            ("read_conflicts", ContradictionError),
            ("multiple_data_blocks", InsufficientInformationError),
            ("format_unsupported", UnsupportedError),
            ("format_unknown", InvalidRequestError),
            ("embedding_failed", ResourceLimitError),
        ],
    )
    def test_raise_class(self, name, error):
        assert CODES[name].error is error

    @pytest.mark.parametrize(
        ("name", "kind", "cause"),
        [
            ("seqres_correspondence_ambiguous", RecordKind.UNKNOWN, Cause.AMBIGUOUS),
            ("reference_incomplete_absent", RecordKind.UNKNOWN, Cause.INSUFFICIENT),
            ("reference_incomplete_not_in_library", RecordKind.UNKNOWN, Cause.UNSUPPORTED),
            ("observed_leaving_atom", RecordKind.CONFLICT, Cause.NONE),
            ("absent_in_model", RecordKind.UNREALISED, Cause.NONE),
        ],
    )
    def test_record_kind_and_cause(self, name, kind, cause):
        assert (CODES[name].kind, CODES[name].cause) == (kind, cause)

    @pytest.mark.parametrize(
        ("name", "payload"),
        [
            ("seqres_correspondence_ambiguous", (("embeddings", PayloadKind.INT),)),
            ("prepare_accepted_incomplete", (("cause", PayloadKind.CODE),)),
            ("assumption_falsified", (("rule", PayloadKind.TEXT),)),
        ],
    )
    def test_payload(self, name, payload):
        assert CODES[name].payload == payload

    def test_fail_sites_name_raise_codes(self):
        raise_codes = {name for name, code in CODES.items() if code.disposition is Disposition.RAISE}
        wrong = []
        for path in PACKAGE.rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "fail":
                    code = node.args[0]
                    if isinstance(code, ast.Constant) and code.value not in raise_codes:
                        wrong.append(f"{path}:{node.lineno} {code.value}")
        assert not wrong


class TestFail:
    def test_raises_code_class(self):
        ref = SourceRef("struct_conn", 3, "ptnr1_label_atom_id", "1abc")
        with pytest.raises(ContradictionError) as raised:
            fail("endpoint_dangling", ref, side=1)
        assert (raised.value.code, raised.value.refs, dict(raised.value.detail)) == (
            "endpoint_dangling",
            (ref,),
            {"side": 1},
        )
        assert "endpoint_dangling" in str(raised.value) and "struct_conn" in str(raised.value)

    def test_message_names_records(self):  # a refusal's records: code, subject, payload, source rows
        subject, refs = (SubjectKind.ATOM, 4), (SourceRef("atom_site", 9),)
        record = Record("placed_leaving_atom", RecordKind.CONFLICT, Cause.NONE, subject, refs, {"atom": 5})
        message = str(ContradictionError("read_conflicts", conflicts=(record,)))
        assert "placed_leaving_atom(subject=(atom, 4), atom=5, SourceRef(category='atom_site', row=9))" in message

    def test_refuses_record_code(self):
        with pytest.raises(AssertionError):
            fail("contact_row")

    def test_class_follows_code(self):
        with pytest.raises(AssertionError):
            UnsupportedError("format_unknown")


class TestCandidates:
    def test_none_means_uninterpretable(self):
        with pytest.raises(InsufficientInformationError) as raised:
            fail("component_not_in_library", RequestRef("component", "XYZ"))
        assert raised.value.candidates == ()

    def test_several_mean_ambiguity(self):
        candidates = (StructureRef(atom=1), StructureRef(atom=2))
        with pytest.raises(InsufficientInformationError) as raised:
            fail("endpoint_ambiguous", candidates=candidates)
        assert raised.value.candidates == candidates

    def test_one_is_not_ambiguity(self):
        with pytest.raises(AssertionError):
            fail("endpoint_ambiguous", candidates=(SourceRef("atom_site", 1),))
