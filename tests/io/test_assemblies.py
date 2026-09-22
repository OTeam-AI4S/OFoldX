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

"""Declared operators and assembly copies (ofoldx.io.assemblies)."""

from __future__ import annotations

import numpy as np
import pytest

from ofoldx.data.structure import CopyId, CopyKind
from ofoldx.errors import OFoldXError, SourceRef
from ofoldx.io.assemblies import assembly_copies, declared_operators, parse_operator_expression
from ofoldx.io.symmetry import CrystalFrame, Symmetry, SymOp
from ofoldx.io.tables import AssemblyTable, OperatorTable
from ofoldx.provenance import Basis, Records, Resolution, SourceRefs, state

GEN = SourceRef("pdbx_struct_assembly_gen", 0)
IDENTITY = np.eye(4)[:3]
SHIFT_X = np.array([[1.0, 0, 0, 10], [0, 1, 0, 0], [0, 0, 1, 0]])
QUARTER_Z = np.array([[0.0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0]])  # 90° about z
FIFTH_TURN = np.array([[0.5, -0.31, 0.81, 0], [-0.31, 0.81, 0.5, 0], [-0.81, -0.5, 0.31, 0]])  # 72°, 7ns2's 2 decimals
SCREW = SymOp((-1, 0, 0, 0, 1, 0, 0, 0, -1), (0, 12, 0))  # -x,y+1/2,-z
CUBE = CrystalFrame(
    np.eye(3, 4) / 10, 10 * np.eye(3), state(Resolution.KNOWN, Basis.DECLARED), np.zeros((3, 4)), (), np.zeros((3, 3))
)
CRYSTAL, POINT = "crystal symmetry operation", "point symmetry operation"
INAPPLICABLE = state(Resolution.INAPPLICABLE, Basis.DEFINED)
INSTANCES = {"A": np.array([0]), "B": np.array([1])}  # _struct_asym order


def p1211(frame: CrystalFrame | None = CUBE, disagreement: frozenset[str] = frozenset()) -> Symmetry:
    identity = SymOp((1, 0, 0, 0, 1, 0, 0, 0, 1), (0, 0, 0))
    unknown = state(Resolution.UNKNOWN, Basis.NONE)
    operations, records = (identity, SCREW), Records.empty()
    return Symmetry(operations, ("1", "2"), "file", None, frame, disagreement, records, (1, 2), -1, unknown, -1, False)


class Source:
    """One read's operator and assembly tables, built row by row with their symbol families."""

    def __init__(self) -> None:
        self.families: dict[str, list[str]] = {"operator": [], "text": [], "assembly": []}
        self.operators: list[tuple] = []
        self.generators: list[tuple] = []

    def code(self, family: str, text: str | None) -> int:
        if text is None:
            return -1
        table = self.families[family]
        if text not in table:
            table.append(text)
        return table.index(text)

    def operator(self, id_, matrix=IDENTITY, type_=POINT, name=None) -> Source:
        codes = self.code("operator", id_), self.code("text", type_), self.code("text", name)
        self.operators.append((*codes, np.asarray(matrix, dtype=np.float64)))
        return self

    def generator(self, expression: str, asyms: str, assembly: str = "1") -> Source:
        codes = self.code("assembly", assembly), self.code("text", expression), self.code("text", asyms)
        self.generators.append(codes)
        return self

    @property
    def symbols(self):
        return {family: tuple(table) for family, table in self.families.items()}

    def refs(self) -> SourceRefs:
        o, g = len(self.operators), len(self.generators)
        return SourceRefs(
            np.array([0] * o + [1] * g, np.int16), np.concatenate([np.arange(o), np.arange(g)]).astype(np.int64),
            np.full(o + g, -1, np.int16),
            ("pdbx_struct_oper_list", "pdbx_struct_assembly_gen"), (), "TEST",
        )  # fmt: skip

    def table(self, quantum: float = 5e-7) -> OperatorTable:
        ids, types, names, matrices = zip(*self.operators)
        return OperatorTable(
            np.array(ids, np.int32), np.array(types, np.int32), np.array(names, np.int32), np.stack(matrices),
            np.full((len(ids), 3, 4), quantum), np.arange(len(ids), dtype=np.int64),
        )  # fmt: skip

    def declared(self, symmetry: Symmetry | None = None, quantum: float = 5e-7):
        ids = self.families["operator"]
        return declared_operators(symmetry or p1211(), self.table(quantum), self.symbols, self.refs(), ids)

    def assemble(self, assembly: str = "1", instances=INSTANCES):
        assemblies, expressions, asyms = (np.array(column, np.int32) for column in zip(*self.generators))
        table = AssemblyTable(
            np.unique(assemblies), assemblies, expressions, asyms,
            np.arange(len(self.operators), len(self.operators) + len(assemblies), dtype=np.int64),
        )  # fmt: skip
        return assembly_copies(p1211(), self.table(), table, self.symbols, self.refs(), assembly, instances)

    def copies(self, assembly: str = "1", instances=INSTANCES):
        return self.assemble(assembly, instances)[0]


@pytest.fixture
def source() -> Source:
    return Source()


class TestParseOperatorExpression:
    @pytest.mark.parametrize(
        ("text", "products"),
        [
            ("1", [("1",)]),
            ("1-3", [("1",), ("2",), ("3",)]),
            ("(1,2)(3,4)", [("1", "3"), ("1", "4"), ("2", "3"), ("2", "4")]),
            (" (X0) (1-2) ", [("X0", "1"), ("X0", "2")]),
        ],
    )
    def test_products_left_to_right(self, text, products):
        assert parse_operator_expression(text, GEN) == tuple(products)

    @pytest.mark.parametrize("text", ["", "()", "(1,2", "1)(2", "(3-1)", "1,,2"])
    def test_grammar_violation_refused(self, text):
        with pytest.raises(OFoldXError) as error:
            parse_operator_expression(text, GEN)
        assert error.value.code == "operator_expression_grammar"


class TestDeclaredOperators:
    def test_crystal_row_computed(self, source):
        expected = np.array([[-1.0, 0, 0, 10], [0, 1, 0, 5], [0, 0, -1, 0], [0, 0, 0, 1]])
        deposited = expected[:3] + 1e-7
        (operator,), records = source.operator("1", deposited, CRYSTAL, "2_655").declared()
        assert np.allclose(operator.matrix, expected, rtol=0, atol=1e-12)
        assert (operator.status, operator.defect_origin, len(records)) == ("rigid", "crystal_frame", 0)

    def test_deviating_deposit_recorded(self, source):
        deposited = np.array([[-1.0, 0, 0, 10], [0, 1, 0, 5.1], [0, 0, -1, 0]])
        (operator,), records = source.operator("1", deposited, CRYSTAL, "2_655").declared()
        assert operator.matrix[1, 3] == 5.0
        assert records.counts() == {"operator_matrix_deviates": 1}

    @pytest.mark.parametrize(("name", "disagreement"), [("7_555", frozenset()), ("2_555", frozenset({"2"}))])
    def test_unresolved_code_applies_deposit(self, source, name, disagreement):
        (operator,), _ = source.operator("1", SHIFT_X, CRYSTAL, name).declared(p1211(disagreement=disagreement))
        assert np.array_equal(operator.matrix[:3], SHIFT_X)

    def test_inapplicable_frame_applies_deposit(self, source):  # no crystal operation applies
        nan = np.full((3, 4), np.nan)
        frame = CrystalFrame(nan, nan[:, :3], INAPPLICABLE, np.zeros((3, 4)), (), np.zeros((3, 3)))
        (operator,), _ = source.operator("1", SHIFT_X, CRYSTAL, "2_555").declared(p1211(frame=frame))
        assert np.array_equal(operator.matrix[:3], SHIFT_X)

    def test_unknown_frame_leaves_undetermined(self, source):
        (operator,), _ = source.operator("1", SHIFT_X, CRYSTAL, "2_555").declared(p1211(frame=None))
        assert operator.status == "undetermined" and np.isnan(operator.matrix).all()

    @pytest.mark.parametrize(
        ("scale", "status", "counts"),
        [
            (1.0, "rigid", {}),
            (1 + 5e-4, "defect_recorded", {"operator_defect_recorded": 1}),
            (1.01, "defect_recorded", {"non_isometric_operator": 1}),
        ],
    )
    def test_rigidity_defect_classified(self, source, scale, status, counts):
        (operator,), records = source.operator("1", IDENTITY * scale).declared()
        assert operator.status == status and records.counts() == counts

    @pytest.mark.parametrize(
        ("quantum", "code"), [(5e-3, "operator_defect_recorded"), (5e-7, "non_isometric_operator")]
    )
    def test_conflict_beyond_printed_rounding(self, source, quantum, code):  # defect 1.6e-3, as printed at 2 or 6 dp
        (operator,), records = source.operator("1", FIFTH_TURN).declared(quantum=quantum)
        assert operator.status == "defect_recorded" and records.counts() == {code: 1}

    def test_2d3d_operation_unsupported(self, source):
        (operator,), records = source.operator("1", type_="2D crystal symmetry operation").declared()
        assert operator.status == "unsupported" and records.counts() == {"operator_2d3d_present": 1}


class TestAssemblyCopies:
    def test_copy_order_and_ids(self, source):
        copies = source.operator("1").operator("2", SHIFT_X).generator("(1,2)", "B,A").generator("1", "A").copies()
        assert [copy.copy_id for copy in copies] == [
            CopyId(CopyKind.ASSEMBLY, operators, asym) for operators in (("1",), ("2",)) for asym in "AB"
        ]
        assert np.array_equal(copies[0].matrix, np.eye(4)) and copies[2].matrix[0, 3] == 10

    def test_chain_copy_names_no_asym(self, source):
        (copy,) = source.operator("1").generator("1", "A").copies(instances={"A": np.array([0, 1])})
        assert copy.copy_id == CopyId(CopyKind.ASSEMBLY, ("1",), None)

    @pytest.mark.parametrize(("expression", "counts"), [("1", {}), ("1,2", {"operator_matrix_deviates": 1})])
    def test_named_operators_recorded(self, source, expression, counts):  # 2zex: assembly 1 never applies its 1_556
        deviating = np.array([[-1.0, 0, 0, 10], [0, 1, 0, 5.1], [0, 0, -1, 0]])
        source.operator("1").operator("2", deviating, CRYSTAL, "2_655").generator(expression, "A")
        assert source.assemble()[1].counts() == counts

    def test_rightmost_operator_applied_first(self, source):
        (copy,) = source.operator("1", QUARTER_Z).operator("2", SHIFT_X).generator("(1)(2)", "A").copies()
        assert np.allclose(copy.matrix @ [0, 0, 0, 1], [0, 10, 0, 1])

    @pytest.mark.parametrize(
        ("operators", "expression", "asyms", "assembly", "code"),
        [
            ([("1", IDENTITY, POINT)], "1", "A", "9", "assembly_unknown"),
            ([("1", IDENTITY, POINT)], "(1,3)", "A", "1", "operator_unresolved"),
            ([("1", IDENTITY, POINT)], "1", "A,Z", "1", "asym_undeclared"),
            ([("1", IDENTITY, POINT), ("2", IDENTITY + 1e-7, POINT)], "1,2", "A", "1", "copy_identity_conflict"),
            ([("1", IDENTITY, "3D crystal symmetry operation")], "1", "A", "1", "operator_2d3d"),
        ],
    )
    def test_request_refused(self, source, operators, expression, asyms, assembly, code):
        for id_, matrix, type_ in operators:
            source.operator(id_, matrix, type_)
        with pytest.raises(OFoldXError) as error:
            source.generator(expression, asyms).copies(assembly)
        assert error.value.code == code
