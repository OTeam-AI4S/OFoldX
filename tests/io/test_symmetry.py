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

import json
import re
from fractions import Fraction
from importlib import resources

import numpy as np
import pytest

from ofoldx.data.molecular import BondKind, Transforms
from ofoldx.errors import InsufficientInformationError
from ofoldx.io.binding import QuerySource
from ofoldx.io.symmetry import (
    Symmetry,
    SymOp,
    cartesian,
    edge_geometry,
    parse_symmetry_code,
    resolve_transforms,
    stabilisers,
    symmetry,
)
from ofoldx.io.tables import CrystalTable, ExperimentTable
from ofoldx.provenance import Basis, Resolution, SourceRefs, known, state

KNOWN_DECLARED = state(Resolution.KNOWN, Basis.DECLARED)
KNOWN_DEFINED = state(Resolution.KNOWN, Basis.DEFINED)
KNOWN_DERIVED = state(Resolution.KNOWN, Basis.DERIVED)
UNKNOWN_DECLARED = state(Resolution.UNKNOWN, Basis.DECLARED)
UNKNOWN = state(Resolution.UNKNOWN, Basis.NONE)
INAPPLICABLE = state(Resolution.INAPPLICABLE, Basis.DEFINED)
CONFLICT = state(Resolution.CONFLICT, Basis.NONE)
IDENTITY = SymOp((1, 0, 0, 0, 1, 0, 0, 0, 1), (0, 0, 0))
TRIVIAL = {(0, (0, 0, 0))}
NOWHERE = [np.nan] * 3
TORSION_FREE = {1, 4, 7, 9, 19, 29, 33, 76, 78, 144, 145, 169, 170}  # the 13 Bieberbach space-group types
ABSENT, DOT, ASK = None, ".", "?"
COV, COORD = BondKind.COVALENT, BondKind.COORDINATION

CUBIC = (10.0, 10.0, 10.0, 90.0, 90.0, 90.0)
ORTHORHOMBIC = (10.0, 20.0, 30.0, 90.0, 90.0, 90.0)
PARTIAL_CELL = (10.0, None, 10.0, 90.0, 90.0, 90.0)  # frame UNKNOWN
IMPOSSIBLE_CELL = (10.0, 10.0, 10.0, 90.0, 90.0, 200.0)  # frame CONFLICT
PLACEHOLDER = (1.0, 1.0, 1.0, 90.0, 90.0, 90.0)
EM = ("ELECTRON MICROSCOPY",)

P1211 = ("x,y,z", "-x,y+1/2,-z")
P121 = ("x,y,z", "-x,y,-z")
C121 = ("x,y,z", "-x,y,-z", "x+1/2,y+1/2,z", "-x+1/2,y+1/2,-z")
P1M1 = ("x,y,z", "x,-y,z")
P1C1 = ("x,y,z", "x,-y,z+1/2")
P222 = ("x,y,z", "-x,-y,z", "-x,y,-z", "x,-y,-z")
P212121 = ("x,y,z", "-x+1/2,-y,z+1/2", "-x,y+1/2,-z+1/2", "x+1/2,-y+1/2,-z")
CCTBX = ("x,y,z", "x+1/2,-y+1/2,-z", "-x,y+1/2,-z+1/2", "-x+1/2,-y,z+1/2")  # P 21 21 21 as PHENIX lists it
P4 = ("x,y,z", "-y,x,z", "-x,-y,z", "y,-x,z")
P422 = ("x,y,z", "-x,-y,z", "-y,x,z", "y,-x,z", "-x,y,-z", "x,-y,-z", "y,x,-z", "-y,-x,-z")
P4322 = ("x,y,z", "-x,-y,z+1/2", "-y,x,z+3/4", "y,-x,z+1/4", "-x,y,-z", "x,-y,-z+1/2", "y,x,-z+1/4", "-y,-x,-z+3/4")
P61 = ("x,y,z", "-y,x-y,z+1/3", "-x+y,-x,z+2/3", "-x,-y,z+1/2", "y,-x+y,z+5/6", "x-y,x,z+1/6")
C2221 = (
    "x,y,z", "-x,-y,z+1/2", "-x,y,-z+1/2", "x,-y,-z",
    "x+1/2,y+1/2,z", "-x+1/2,-y+1/2,z+1/2", "-x+1/2,y+1/2,-z+1/2", "x+1/2,-y+1/2,-z",
)  # fmt: skip


def op(triplet: str) -> SymOp:
    """A test operation from its triplet (signed axes and fractions only)."""
    rotation, translation = [], []
    for component in triplet.split(","):
        row, constant = [0, 0, 0], Fraction(0)
        for sign, term in re.findall(r"([+-]?)([xyz]|[\d/]+)", component):
            value = -1 if sign == "-" else 1
            if term in "xyz":
                row["xyz".index(term)] = value
            else:
                constant += value * Fraction(term)
        rotation += row
        translation.append(int(constant * 24))
    return SymOp(tuple(rotation), tuple(translation))


def statements(values):
    """One fact's statements in category order (rows 0, 8): a value or a tuple; None absent, "?" stated unknown."""
    values = values if isinstance(values, tuple) else (values,)
    states = [UNKNOWN if v is None else UNKNOWN_DECLARED if v == "?" else KNOWN_DECLARED for v in values]
    refs = [-1 if v is None else 8 * k for k, v in enumerate(values)]
    return [v if s == KNOWN_DECLARED else None for v, s in zip(values, states)], np.array(states, np.uint8), refs


def crystal(
    name=None, number=None, table=(), cell=None, cell_quantum=None, deposited=None, axes=False, methods=(), wwpdb=False
):
    """A CrystalTable: ``name``, ``number`` one or several statements; ``table`` of (id, text) rows, text None
    stating none; ``cell`` six values (None entries absent); ``deposited`` maps 0 (fract) / 1 (Cartn) to twelve
    (value, quantum) pairs, None absent, "?" stated unknown; ``axes`` a stated Cartn_transform_axes; ``methods`` the
    texts of the experiment's methods; ``wwpdb`` a stated process_site."""
    names, name_states, name_refs = statements(name)
    numbers, number_states, number_refs = statements(number)
    groups = tuple(dict.fromkeys(n for n in names if n is not None))
    ids, texts = tuple(dict.fromkeys(i for i, _ in table)), tuple(dict.fromkeys(t for _, t in table if t is not None))
    cell = [None] * 6 if cell is None else list(cell)
    frame, quanta = np.full((2, 3, 4), np.nan), np.zeros((2, 3, 4))
    frame_state = np.full((2, 3, 4), UNKNOWN, dtype=np.uint8)
    for group, values in (deposited or {}).items():
        for index, value in enumerate(values):
            if value == "?":
                frame_state[group].flat[index] = UNKNOWN_DECLARED
            elif value is not None:
                frame[group].flat[index], quanta[group].flat[index] = value
                frame_state[group].flat[index] = KNOWN_DECLARED
    return CrystalTable(
        cell=np.array([np.nan if v is None else v for v in cell], dtype=np.float64),
        cell_quantum=np.array(cell_quantum or [0.0] * 6, dtype=np.float64),
        cell_state=np.array([UNKNOWN if v is None else KNOWN_DECLARED for v in cell], dtype=np.uint8),
        space_group=np.array([-1 if n is None else groups.index(n) for n in names], np.int32),
        space_group_state=name_states,
        number=np.array([0 if n is None else n for n in numbers], np.int64),
        number_state=number_states,
        operation_id=np.array([ids.index(i) for i, _ in table], dtype=np.int32),
        operation_text=np.array([-1 if t is None else texts.index(t) for _, t in table], dtype=np.int32),
        operation_ref=np.arange(1, len(table) + 1, dtype=np.int64),  # symop rows 1…n of the read's references
        deposited_frame=frame,
        deposited_quantum=quanta,
        deposited_state=frame_state,
        symbols={"space_group": groups, "operation_id": ids, "operation_text": texts, "text": tuple(methods)},
        space_group_ref=np.array(name_refs, np.int64),
        number_ref=np.array(number_refs, np.int64),
        cell_ref=7,
        deposited_ref=np.array([6, 6], dtype=np.int64),
        axes_ref=9 if axes else -1,
        wwpdb_ref=15 if wwpdb else -1,
    )


def read(**items) -> Symmetry:
    n, m = 16, len(items.get("methods", ()))
    refs = SourceRefs(
        np.zeros(n, np.int16), np.arange(n, dtype=np.int64), np.full(n, -1, np.int16),
        ("struct_conn",), (), None,
    )  # fmt: skip
    rows = np.zeros(m, np.uint8), np.full(m, UNKNOWN, np.uint8)  # method rows: ResolutionKind.NONE
    experiment = ExperimentTable(np.arange(m, dtype=np.int32), np.zeros(m), *rows, np.arange(10, 10 + m))
    return symmetry(crystal(**items), experiment, refs)


def listed(triplets, **items) -> Symmetry:
    """A read of a file table of the triplets, ids "1"…"N"."""
    return read(table=[(str(k), t) for k, t in enumerate(triplets, 1)], **items)


def stabilise(symmetry_, sites, placed=True, quantum=0.0005, endpoints=None):
    """stabilisers over the sites; a site of NaN is never placed."""
    sites = np.array(sites, dtype=np.float64).reshape(-1, 3)
    n = len(sites)
    endpoints = np.arange(n, dtype=np.int32) if endpoints is None else np.array(endpoints, dtype=np.int32)
    placed = np.broadcast_to(np.array(placed), n) & np.isfinite(sites).all(axis=1)
    return stabilisers(symmetry_, endpoints, sites, placed, np.full(n, quantum))


def members(table, row=0) -> set[tuple[int, tuple[int, ...]]]:
    span = slice(table.ptr[row], table.ptr[row + 1])
    return {(int(k), tuple(map(int, s))) for k, s in zip(table.operation[span], table.shift[span])}


def undecided(table, row=0) -> set[int]:
    return set(map(int, table.undecided[table.undecided_ptr[row] : table.undecided_ptr[row + 1]]))


def stated(sides):
    """The code texts, the codes into them and the code states of declaration sides."""
    texts = tuple(dict.fromkeys(s for pair in sides for s in pair if s not in (ABSENT, DOT, ASK)))
    by = {ABSENT: UNKNOWN, DOT: KNOWN_DEFINED, ASK: UNKNOWN_DECLARED}
    codes = np.array([[texts.index(s) if s in texts else -1 for s in pair] for pair in sides], np.int32).reshape(-1, 2)
    states = np.array([[by.get(s, KNOWN_DECLARED) for s in pair] for pair in sides], np.uint8).reshape(-1, 2)
    return texts, codes, states


def resolve(symmetry_, sides, sites=(), origin=QuerySource.STRUCT_CONN, rows=None):
    """resolve_transforms over declarations whose sides are code texts, DOT, ASK or ABSENT; ``rows`` gives each
    declaration's (first, second) atom rows of ``sites``."""
    texts, codes, states = stated(sides)
    d = len(sides)
    pairs = np.array(rows if rows is not None else [(-1, -1)] * d, dtype=np.int64).reshape(-1, 2)
    return resolve_transforms(
        symmetry_, codes, states, texts, np.full(d, origin, np.uint8), np.arange(d, dtype=np.int64),
        pairs[:, 0].copy(), pairs[:, 1].copy(), np.array(sites, dtype=np.float64).reshape(-1, 3),
    )  # fmt: skip


def relative(table, i=0):
    """The operation (None unless KNOWN) and state of declaration i's relative transform."""
    t, row = table.transforms, table.relative[i]
    resolved = int(t.state[row])
    if not known(resolved):
        return None, resolved
    return SymOp(tuple(t.rotation[row].ravel().tolist()), tuple(t.translation[row].tolist())), resolved


def side(table, i, s):
    return int(table.transforms.state[table.per_side[i, s]])


def geometry(symmetry_, declarations, sites, statement=None):
    """resolve_transforms → stabilisers → edge_geometry: endpoints are atom rows of ``sites`` (None = unplaced);
    declarations are (first, second, code1, code2, kind, topology_fixed); ``statement`` what each states (alike)."""
    coordinates = np.array([NOWHERE if s is None else s for s in sites], dtype=np.float64)
    pairs, sides = [(f, s) for f, s, *_ in declarations], [(a, b) for _, _, a, b, *_ in declarations]
    transforms = resolve(symmetry_, sides, coordinates, rows=pairs)
    table = stabilise(symmetry_, coordinates)
    statement = np.zeros(len(declarations)) if statement is None else np.array(statement)
    edges = edge_geometry(
        symmetry_, transforms, *stated(sides)[1:], table,
        np.array([f for f, *_ in declarations], np.int32), np.array([s for _, s, *_ in declarations], np.int32),
        np.array([k for *_, k, _ in declarations], np.uint8), np.array([t for *_, t in declarations], bool),
        statement.astype(np.int64).reshape(len(declarations), -1),
    )  # fmt: skip
    return transforms, table, edges


class TestSymOp:
    @pytest.mark.parametrize("triplet", P4322 + P61)
    def test_identity_is_neutral(self, triplet):
        assert IDENTITY.compose(op(triplet)) == op(triplet) == op(triplet).compose(IDENTITY)

    @pytest.mark.parametrize(
        ("a", "b", "c"),
        [(P4322[2], P4322[5], P4322[7]), (P4322[3], P4322[3], P4322[1]), (P61[1], P61[5], P61[4])],
    )
    def test_composition_is_associative(self, a, b, c):
        a, b, c = op(a), op(b), op(c)
        assert a.compose(b).compose(c) == a.compose(b.compose(c))

    @pytest.mark.parametrize("triplet", P4322 + P61)
    def test_inverse_is_two_sided(self, triplet):
        assert op(triplet).compose(op(triplet).inverse()) == IDENTITY == op(triplet).inverse().compose(op(triplet))

    def test_screw_power_is_translation(self):
        screw = power = op("x-y,x,z+1/6")
        for _ in range(5):
            power = power.compose(screw)
        assert power == SymOp(IDENTITY.rotation, (0, 0, 24))


class TestParseSymmetryCode:
    @pytest.mark.parametrize(
        ("text", "parsed"),
        [("3_655", ("3", (1, 0, 0))), ("12", ("12", (0, 0, 0))), (" 2_545 ", ("2", (0, -1, 0))),
         ("192_919", ("192", (4, -4, 4)))],
    )  # fmt: skip
    def test_grammatical_code(self, text, parsed):
        assert parse_symmetry_code(text) == parsed

    @pytest.mark.parametrize("text", ["4_8105", "1_055", "0_555", "193", "1555", "x,y,z"])
    def test_grammar_violation(self, text):
        assert parse_symmetry_code(text) is None


class TestSetting:
    @pytest.mark.parametrize(
        ("name", "number", "index", "triplet"),
        [("C 2 2 21", 20, 3, "-x,y,-z+1/2"), ("I 2 2 2", 23, 7, "-x+1/2,y+1/2,-z+1/2"),
         ("P 43 2 2", 95, 8, "-y,-x,-z+3/4")],
    )  # fmt: skip
    def test_ita_order(self, name, number, index, triplet):
        symmetry_ = read(name=name)
        assert (symmetry_.setting.name, symmetry_.setting.number, symmetry_.numbering) == (name, number, "ita")
        assert symmetry_.operations[index - 1] == op(triplet) and symmetry_.operation_ids[index - 1] == str(index)

    def test_alias_resolves_full_setting(self):
        setting = read(name="P b c a", number=61).setting
        assert (setting.name, setting.number, len(setting.operations)) == ("P 21/b 21/c 21/a", 61, 8)

    @pytest.mark.parametrize(
        ("name", "setting"),
        [("R 3 2 :H", "H 3 2"), ("R 3 2 :R", "R 3 2"), ("R 3 2", "R 3 2"), ("R -3 m :H", "H -3 2/m")],
    )
    def test_rhombohedral_axes(self, name, setting):  # ITA :H is wwPDB H; :R and a bare R are rhombohedral axes
        assert read(name=name).setting.name == setting

    @pytest.mark.parametrize(("name", "number"), [("P 2 21 21", 18), ("R 3 2", 155)])
    def test_alternative_setting_number(self, name, number):
        symmetry_ = read(name=name, number=number)
        assert symmetry_.setting.number == number and len(symmetry_.records) == 0

    @pytest.mark.parametrize("name", [None, "C 4 21 2", "P 21 21 2 A"])
    def test_unresolved_setting(self, name):  # never from the number, never outside the table
        symmetry_ = read(name=name, number=19)
        assert (symmetry_.setting, symmetry_.numbering, symmetry_.operations) == (None, "none", ())
        assert symmetry_.records.counts() == {"space_group_setting_unresolved": 1}

    def test_number_contradicts_name(self):
        symmetry_ = read(name="P 21 21 21", number=4)
        assert symmetry_.setting.name == "P 21 21 21"
        assert symmetry_.records.counts() == {"space_group_name_vs_number": 1}
        assert stabilise(symmetry_, [NOWHERE]).state.tolist() == [CONFLICT]
        assert relative(resolve(symmetry_, [("2_555", DOT)]))[1] == CONFLICT

    @pytest.mark.parametrize(
        ("name", "number", "records", "resolved"),
        [
            pytest.param(("?", "P 1 21 1"), None, {}, KNOWN_DECLARED, id="unstated-name-agrees"),
            pytest.param(("P 21/b 21/c 21/a", "P b c a"), (61, 61), {}, KNOWN_DECLARED, id="aliases-agree"),
            pytest.param(("H 3 2", "R 3 2 :H"), None, {}, KNOWN_DECLARED, id="6m3g-hexagonal-axes"),
            pytest.param(("H 3 2", "R 3 2 :R"), None, {"space_group_restatement_conflict": 1}, CONFLICT, id="two-axes"),
            pytest.param(
                ("P 1 21 1", "P 1 2 1"), None, {"space_group_restatement_conflict": 1}, CONFLICT, id="two-settings"
            ),
            pytest.param("P 1 21 1", (4, 3), {"space_group_restatement_conflict": 1}, KNOWN_DECLARED, id="two-numbers"),
        ],
    )
    def test_restated_space_group(self, name, number, records, resolved):  # _symmetry and _space_group
        symmetry_ = read(name=name, number=number)
        assert (symmetry_.records.counts(), symmetry_.space_group_state) == (records, resolved)

    @pytest.mark.parametrize("items", [{"name": ("P 1 21 1", "P 1 2 1")}, {"name": "P 1 21 1", "number": (4, 3)}])
    def test_contradicted_setting_conflicts(self, items):  # no file table governs: through the setting is CONFLICT
        symmetry_ = read(**items)
        assert relative(resolve(symmetry_, [("2_555", DOT)]))[1] == CONFLICT
        assert stabilise(symmetry_, [NOWHERE]).state.tolist() == [CONFLICT]

    def test_contested_number_keeps_table(self):
        symmetry_ = listed(P121, name="P 1 2 1", number=4, cell=CUBIC)
        assert symmetry_.records.counts() == {"space_group_name_vs_number": 1}
        assert (symmetry_.numbering, stabilise(symmetry_, [1.0, 2.0, 3.0]).state[0]) == ("file", KNOWN_DERIVED)


class TestNumbering:
    def test_file_ids_govern(self):
        symmetry_ = read(table=[("2", "-x,y,-z"), ("1", "x,y,z")])
        assert (symmetry_.numbering, symmetry_.operation_ids) == ("file", ("2", "1"))
        assert relative(resolve(symmetry_, [("2_555", DOT)]))[0] == op("-x,y,-z")

    def test_ita_completion(self):
        symmetry_ = read(name="P 1 21 1")
        assert (symmetry_.numbering, symmetry_.operation_ids, symmetry_.operation_refs) == ("ita", ("1", "2"), (-1, -1))

    @pytest.mark.parametrize(
        ("items", "operations", "resolved"),
        [
            pytest.param({}, (IDENTITY,), KNOWN_DERIVED, id="nothing-declared"),
            pytest.param({"name": "C 4 21 2"}, (), UNKNOWN, id="space-group-unresolved"),
            pytest.param({"cell": ORTHORHOMBIC}, (), UNKNOWN, id="cell-without-space-group"),
        ],
    )
    def test_no_list_in_force(self, items, operations, resolved):
        symmetry_ = read(**items)
        assert (symmetry_.numbering, symmetry_.operation_ids, symmetry_.operations) == ("none", (), operations)
        assert symmetry_.records.counts() == ({} if operations else {"space_group_setting_unresolved": 1})
        assert stabilise(symmetry_, [NOWHERE]).state.tolist() == [resolved]

    def test_unparsed_table_numbers_nothing(self):  # 1l6j: a trailing comma after every operator
        symmetry_ = listed(("+X,+Y,+Z,", "-X,Y+1/2,-Z,"), name="P 1 21 1")
        assert (symmetry_.numbering, symmetry_.operation_ids, symmetry_.operations) == ("none", (), ())
        details = [{"id": "1", "text": "+X,+Y,+Z,"}, {"id": "2", "text": "-X,Y+1/2,-Z,"}]
        assert [dict(record.detail) for record in symmetry_.records] == details
        transforms = resolve(symmetry_, [("2_555", DOT)])
        assert (relative(transforms)[1], len(transforms.records)) == (UNKNOWN, 0)  # the setting is not the cause

    def test_unstated_operation_agrees(self):  # _space_group_symop states ?, _symmetry_equiv the operation
        symmetry_ = read(table=[("1", "x,y,z"), ("2", None), ("1", "x,y,z"), ("2", "-x,y,-z")])
        assert (symmetry_.numbering, symmetry_.operation_ids, len(symmetry_.records)) == ("file", ("1", "2"), 0)

    def test_partly_unparsed_table_numbers_nothing(self):  # one statement: a bad row puts none of it in force
        symmetry_ = listed(("x,y,z", "-x,y,-z,"), name="P 1 2 1")
        assert (symmetry_.numbering, symmetry_.operations) == ("none", ())
        transforms = resolve(symmetry_, [("1_555", DOT)])  # a decoded row's id is Unknown too
        assert (relative(transforms)[1], len(transforms.records)) == (UNKNOWN, 0)

    def test_conflicting_table_numbers_nothing(self):  # its one id names two operations: no row is decoded
        symmetry_ = read(table=[("2", "x,y,z"), ("2", "-x,y,-z")])
        assert (symmetry_.numbering, symmetry_.operations) == ("none", ())

    @pytest.mark.parametrize(
        ("triplets", "wwpdb", "numbering"),
        [
            pytest.param(CCTBX, True, "ita", id="wwpdb"),
            pytest.param((*CCTBX[:3], "-x+1/2,-y,z-1/2"), True, "ita", id="unreduced-row"),  # 6efz: -x+1,y+1/2,-z+1/2
            pytest.param(CCTBX, False, "file", id="user-file"),
            pytest.param(CCTBX[:2], True, "file", id="other-operations"),
        ],
    )
    def test_wwpdb_numbers_by_ita(self, triplets, wwpdb, numbering):  # 3,749 wwPDB tables list this order
        symmetry_ = listed(triplets, name="P 21 21 21", wwpdb=wwpdb)
        assert symmetry_.numbering == numbering
        operation = P212121[1] if numbering == "ita" else triplets[1]
        assert relative(resolve(symmetry_, [(DOT, "2_555")]))[0] == op(operation)

    @pytest.mark.parametrize("triplets", [P4322, C2221], ids=["6kpq-P4322", "C2221"])
    def test_renumbering_changes_nothing(self, triplets):  # acceptance property X4 (unit level)
        n = len(triplets)
        sites = [[2.0, 3.0, 4.0], [3.5, 3.0, 4.0], [0.0, 0.0, 0.0], None]
        declarations = [(0, 1, "1_555", "1_555", COV, False), (0, 2, "1_555", "3_655", COV, False),
                        (1, 2, "2_555", "5_555", COV, False), (2, 3, "1_555", "4_545", COV, True),
                        (0, 0, "1_555", f"{n}_555", COV, False)]  # fmt: skip
        outcomes = []
        for rename in (lambda k: k, lambda k: n + 1 - k):
            table = [(str(rename(k)), t) for k, t in enumerate(triplets, 1)]
            renamed = [(f, s, *(f"{rename(int(c[:-4]))}{c[-4:]}" for c in codes), k, t)
                       for f, s, *codes, k, t in declarations]  # fmt: skip
            symmetry_ = read(table=table, cell=(30.0, 30.0, 40.0, 90.0, 90.0, 90.0))
            transforms, stab, edges = geometry(symmetry_, renamed, sites)
            outcomes.append((
                [relative(transforms, i) for i in range(len(declarations))], [members(stab, e) for e in range(4)],
                stab.state.tolist(), edges.count.tolist(), edges.physical_edge.tolist(), edges.self_bond.tolist(),
            ))  # fmt: skip
        assert outcomes[0] == outcomes[1]


class TestCrossCheck:
    def test_permuted_table_is_compatible(self):
        symmetry_ = read(name="P 1 21 1", table=[("7", "-x,y+1/2,-z"), ("3", "x,y,z")])
        assert (len(symmetry_.records), symmetry_.disagreement, symmetry_.numbering) == (0, frozenset(), "file")

    def test_differing_operation_conflicts(self):  # whether or not a declaration references it
        symmetry_ = read(name="P 1 21 1", table=[("1", "x,y,z"), ("2", "-x,y,-z")])
        assert symmetry_.disagreement == {"2"}
        records = symmetry_.records.where(code="operation_table_vs_setting")
        assert len(records) == 2  # the differing row, and the setting's screw the table lacks
        assert {ref.row for ref in records[0].sources} == {2, 0}  # the symop row and the H-M item
        assert stabilise(symmetry_, [NOWHERE]).state.tolist() == [CONFLICT]

    def test_subset_table_conflicts(self):
        symmetry_ = read(name="P 1 21 1", table=[("1", "x,y,z")])
        assert (symmetry_.records.counts(), symmetry_.disagreement) == ({"operation_table_vs_setting": 1}, frozenset())

    def test_unresolved_setting_keeps_table(self):
        symmetry_ = read(name="C 4 21 2", table=[("1", "x,y,z"), ("2", "-x,y,-z")])
        assert (symmetry_.numbering, symmetry_.disagreement) == ("file", frozenset())
        assert symmetry_.records.counts() == {"space_group_setting_unresolved": 1}

    def test_unparsable_row_is_unknown(self):
        symmetry_ = read(name="P 1 21 1", table=[("1", "x,y,z"), ("2", "-x,y+1/2,q")])
        assert symmetry_.records.counts() == {"operation_table_unparsable": 1}
        assert stabilise(symmetry_, [NOWHERE]).state.tolist() == [UNKNOWN]

    def test_table_not_a_group(self):
        symmetry_ = read(table=[("1", "x,y,z"), ("2", "-y,x,z+1/2")])
        assert symmetry_.records.counts() == {"operation_table_not_group": 1}
        assert stabilise(symmetry_, [NOWHERE]).state.tolist() == [CONFLICT]

    def test_id_names_two_operations(self):
        symmetry_ = read(table=[("1", "x,y,z"), ("2", "-x,y,-z"), ("2", "x,-y,-z")])
        assert symmetry_.records.counts() == {"operation_id_conflict": 1}
        assert relative(resolve(symmetry_, [("2_555", DOT)]))[1] == CONFLICT
        assert stabilise(symmetry_, [NOWHERE]).state.tolist() == [CONFLICT]

    @pytest.mark.parametrize(
        ("records", "code", "detail"),
        [
            pytest.param(
                lambda: read(name="P 21 21 21", number=4).records,
                "space_group_name_vs_number", {"name": "P 21 21 21", "number": 4, "setting_number": 19},
                id="name-vs-number",
            ),
            pytest.param(
                lambda: read(table=[("1", "x,y,z"), ("2", "-y,x,z+1/2")]).records,
                "operation_table_not_group", {"listed": 2, "generated": 4},
                id="not-a-group",
            ),
            pytest.param(
                lambda: resolve(read(name="C 4 21 2", number=90), [(DOT, "2_555")]).records,
                "space_group_setting_unresolved", {"side": 1, "name": "C 4 21 2", "number": 90, "code": "2_555"},
                id="setting-unresolved",
            ),
        ],
    )  # fmt: skip
    def test_record_payload(self, records, code, detail):
        assert [dict(record.detail) for record in records().where(code=code)] == [detail]


def diagonal(values, quantum, vector=(0.0, 0.0, 0.0)):
    """Twelve (value, quantum) coefficients of a diagonal deposited frame."""
    cells = [0.0] * 12
    cells[0], cells[5], cells[10] = values
    cells[3], cells[7], cells[11] = vector
    return [(v, quantum) for v in cells]


def fract(rows):
    """Twelve (value, quantum) coefficients of [A | 0], A's rows printed to six decimals."""
    return [(v, 5e-7) for row in rows for v in (*row, 0.0)]


PRINTED = [0.0005] * 3 + [0.005] * 3  # a cell printed to three and two decimals
UNRESOLVED = {"space_group_setting_unresolved": 1}  # a lattice stated without its operations


class TestFrame:
    def test_cell_derives_frame(self):  # no transform stated: [A | 0] by the PDB convention
        frame = read(cell=ORTHORHOMBIC).frame
        assert frame.state == KNOWN_DERIVED
        np.testing.assert_allclose(frame.fractionalisation, np.diag([0.1, 0.05, 1 / 30]) @ np.eye(3, 4), atol=1e-15)
        np.testing.assert_allclose(frame.orthogonalisation, np.diag([10.0, 20.0, 30.0]), atol=1e-12)

    def test_origin_kept(self):  # 1gdi: u = (-1/4, 0, 0) is no conflict with the cell, which fixes the metric only
        frame = {0: diagonal((0.1, 0.05, 1 / 30), 0.0, (-0.25, 0.0, 0.0))}
        symmetry_ = read(name="P 1", cell=ORTHORHOMBIC, deposited=frame)
        assert (symmetry_.frame.state, len(symmetry_.records)) == (KNOWN_DECLARED, 0)
        assert symmetry_.frame.fractionalisation[:, 3].tolist() == [-0.25, 0.0, 0.0]

    @pytest.mark.parametrize(
        ("cell", "printed", "rows", "conflict"),
        [
            pytest.param(
                (40.593, 46.128, 38.65, 90.0, 112.26, 90.0), [0.0005] * 6,
                ((0.024635, 0, 0.010085), (0, 0.021679, 0), (0, 0, 0.027958)), None, id="3a0x",
            ),  # β 112.260 beside the matrix's 112.2631: no conflict provable within 112.26's range
            pytest.param(
                (57.34, 62.523, 83.354, 90.0, 90.0, 90.0), PRINTED,
                ((0.017431, 0, 0), (0, 0.015994, 0), (0, 0, 0.011997)), "length_a", id="3vm9",
            ),  # a 57.340 beside 57.369: beyond even 57.34's range
            pytest.param(
                (100.0, 50.0, 60.0, 90.0, 90.0, 90.0), PRINTED,
                ((0.009941, 0, 0), (0, 0.02, 0), (0, 0, 0.016667)), "length_a", id="integer",
            ),  # a 100.000 beside 100.594: the integer part is never stripped
            pytest.param(
                (54.184, 87.725, 133.45, 90.0, 90.0, 90.0), PRINTED,
                ((0.01846, 0, 0), (0, 0.0114, 0), (0, 0, 0.00749)), None, id="1fhj",
            ),  # the matrix relaxed too: 0.018460 bounds a as 0.01846 does
            pytest.param(
                (1.0, 1.0, 1.0, 90.0, 90.01, 90.0), PRINTED, np.eye(3), "angle_beta", id="3gzt",
            ),  # a printed 0.000000 keeps its half-unit: β 90.01 beside 90
        ],
    )  # fmt: skip
    def test_metric_checked(self, cell, printed, rows, conflict):  # the matrix's metric against the cell
        symmetry_ = read(name="P 1", cell=cell, cell_quantum=printed, deposited={0: fract(rows)})
        assert symmetry_.frame.state == (KNOWN_DECLARED if conflict is None else CONFLICT)
        assert [record.detail["item"] for record in symmetry_.records] == ([] if conflict is None else [conflict])

    @pytest.mark.parametrize("printed", [[0.05] * 3 + [0.005] * 3, PRINTED], ids=["mmcif", "cryst1"])
    def test_cryst1_reread_alike(self, printed):  # 1hy0: a cell of 100.6 Å written as CRYST1 100.600
        rows = ((0.009939, 0.005738, 0), (0, 0.011477, 0), (0, 0, 0.005952))
        cell = (100.6, 100.6, 168.0, 90.0, 90.0, 120.0)
        symmetry_ = read(name="P 1", cell=cell, cell_quantum=printed, deposited={0: fract(rows)})
        assert (symmetry_.frame.state, len(symmetry_.records)) == (KNOWN_DECLARED, 0)

    @pytest.mark.parametrize(("beta", "resolved"), [(90.0, KNOWN_DECLARED), (90.02, CONFLICT)])
    def test_angle_checked(self, beta, resolved):  # the metric fixes the angles too: 0.02° beyond 0.005°
        cell = (*ORTHORHOMBIC[:4], beta, 90.0)
        frame = {0: diagonal((0.1, 0.05, 1 / 30), 5e-7)}
        assert read(name="P 1", cell=cell, cell_quantum=PRINTED, deposited=frame).frame.state == resolved

    def test_printed_zero_bounds(self):  # 2zxi: A13 printed 0.000000 has its half-unit, which allows β = 90.002°
        cell = (119.414, 50.0, 60.0, 90.0, 90.002, 90.0)
        frame = {0: diagonal((0.008374, 0.02, 0.016667), 5e-7)}
        assert read(name="P 1", cell=cell, cell_quantum=[0.0005] * 6, deposited=frame).frame.state == KNOWN_DECLARED

    def test_payload_shows_excess(self):  # rounded outward, the difference never reads as the bound
        frame = {0: diagonal((0.1, 0.05, 1 / 30), 5e-7)}
        symmetry_ = read(name="P 1", cell=(*ORTHORHOMBIC[:4], 90.0064, 90.0), cell_quantum=PRINTED, deposited=frame)
        record = symmetry_.records.where(code="frame_conflict")[0]
        assert record.detail["difference"] > record.detail["bound"]

    @pytest.mark.parametrize(
        ("c_star", "quantum", "resolved"), [(0.00657, 5e-6, KNOWN_DECLARED), (0.006568, 5e-7, CONFLICT)]
    )
    def test_printed_decimals_decide(self, c_star, quantum, resolved):  # 2ou3: five decimals state 0.12 Å
        cell = (38.49, 56.19, 152.3, 90.0, 90.0, 90.0)
        frame = read(cell=cell, deposited={0: diagonal((0.02598, 0.0178, c_star), quantum)}).frame
        assert frame.state == resolved

    def test_cartesian_group_inverted(self):  # a lone Cartn_transf states the frame as its inverse
        frame = read(cell=ORTHORHOMBIC, deposited={1: diagonal((10.0, 20.0, 30.0), 5e-5)}).frame
        assert frame.state == KNOWN_DECLARED
        np.testing.assert_allclose(frame.fractionalisation, np.diag([0.1, 0.05, 1 / 30]) @ np.eye(3, 4))

    @pytest.mark.parametrize(("length", "resolved"), [(30.3, KNOWN_DECLARED), (30.6, CONFLICT)])
    def test_cartesian_digits_stripped(self, length, resolved):  # a lone Cartn_transf 30.00 bounds c as 30 does
        cell = (10.0, 20.0, length, 90.0, 90.0, 90.0)
        assert read(cell=cell, deposited={1: diagonal((10.0, 20.0, 30.0), 5e-3)}).frame.state == resolved

    @pytest.mark.parametrize(("length", "resolved"), [(30.0, KNOWN_DECLARED), (30.5, CONFLICT)])
    def test_cartesian_group_checked(self, length, resolved):  # against [A | u], at the unit cell's corners
        deposited = {0: diagonal((0.1, 0.05, 0.033333), 5e-7), 1: diagonal((10.0, 20.0, length), 5e-5)}
        assert read(deposited=deposited).frame.state == resolved

    def test_partial_group_incomplete(self):
        symmetry_ = read(name="P 1", cell=ORTHORHOMBIC, deposited={0: [(0.1, 5e-7)] + [None] * 11})
        assert (symmetry_.frame.state, symmetry_.records.counts()) == (UNKNOWN, {"frame_incomplete": 1})

    def test_null_group_states_none(self):  # every coefficient ?: no transform is stated, the cell's is derived
        symmetry_ = read(name="P 1", cell=ORTHORHOMBIC, deposited={0: ["?"] * 12, 1: ["?"] * 12})
        assert (symmetry_.frame.state, len(symmetry_.records)) == (KNOWN_DERIVED, 0)

    def test_singular_fract_conflicts(self):
        symmetry_ = read(name="P 1", cell=ORTHORHOMBIC, deposited={0: diagonal((0.1, 0.0, 0.03), 5e-7)})
        assert (symmetry_.frame.state, symmetry_.records.counts()) == (CONFLICT, {"frame_conflict": 1})

    @pytest.mark.parametrize(
        ("cell", "axes", "frame", "code"),
        [
            pytest.param((0.0, 0.0, 0.0, 90.0, 90.0, 90.0), False, None, None, id="zero-lengths-declare-no-crystal"),
            pytest.param(None, False, None, None, id="no-cell"),
            pytest.param(PARTIAL_CELL, False, UNKNOWN, "frame_incomplete", id="partial-cell"),
            pytest.param((10.0, 0.0, 30.0, 90.0, 90.0, 90.0), False, CONFLICT, "frame_conflict", id="partly-zero"),
            pytest.param(IMPOSSIBLE_CELL, False, CONFLICT, "frame_conflict", id="impossible-angle"),
            pytest.param(ORTHORHOMBIC, True, UNKNOWN, "frame_incomplete", id="axes-not-interpreted"),
        ],
    )
    def test_cell(self, cell, axes, frame, code):
        symmetry_ = read(name="P 1", cell=cell, axes=axes)
        assert (None if symmetry_.frame is None else symmetry_.frame.state) == frame
        assert symmetry_.records.counts() == ({} if code is None else {code: 1})

    @pytest.mark.parametrize(
        ("cell", "methods", "resolved"),
        [
            pytest.param(PLACEHOLDER, EM, INAPPLICABLE, id="em"),
            pytest.param(PLACEHOLDER, ("SOLUTION NMR", "X-RAY DIFFRACTION"), KNOWN_DERIVED, id="nmr-beside-x-ray"),
            pytest.param(PLACEHOLDER, (), KNOWN_DERIVED, id="no-method"),
            pytest.param((2.0, 2.0, 2.0, 90.0, 90.0, 90.0), EM, KNOWN_DERIVED, id="em-cell"),
        ],
    )
    def test_placeholder(self, cell, methods, resolved):  # the placeholder cell in a non-crystal context
        assert read(name="P 2 3", cell=cell, methods=methods).frame.state == resolved

    def test_placeholder_ignores_transform(self):  # 3j9b: fract_transf of the 1/box map is no crystal frame
        symmetry_ = read(cell=PLACEHOLDER, methods=EM, deposited={0: diagonal((0.005435,) * 3, 5e-7)})
        assert (symmetry_.frame.state, len(symmetry_.records)) == (INAPPLICABLE, 0)

    @pytest.mark.parametrize(
        ("scale", "methods", "resolved", "records"),
        [
            pytest.param(1.0, ("SOLUTION NMR",), INAPPLICABLE, {}, id="nmr"),  # no lattice asks for a setting
            pytest.param(1.0, ("X-RAY DIFFRACTION",), KNOWN_DECLARED, UNRESOLVED, id="x-ray"),
            pytest.param(0.001, ("SOLUTION NMR",), KNOWN_DECLARED, UNRESOLVED, id="nmr-1000-angstrom"),
        ],
    )
    def test_placeholder_transform(self, scale, methods, resolved, records):  # 1ho7: no _cell, fract_transf = I
        symmetry_ = read(methods=methods, deposited={0: diagonal((scale,) * 3, 5e-7)})
        assert (symmetry_.frame.state, symmetry_.records.counts()) == (resolved, records)
        assert symmetry_.frame.fractionalisation.tolist() == (scale * np.eye(3, 4)).tolist()  # kept

    @pytest.mark.parametrize("cell", [(0.0, 0.0, 0.0, 90.0, 90.0, 90.0), PARTIAL_CELL], ids=["2z33", "partial"])
    def test_unknown_cell_placeholder_transform(self, cell):  # the cell the value stores is none
        frame = read(cell=cell, methods=("SOLUTION NMR",), deposited={0: diagonal((1.0,) * 3, 5e-7)}).frame
        assert frame.state == INAPPLICABLE


class TestCoordinateFreeStabiliser:
    @pytest.mark.parametrize("triplets", [P1211, P212121, P1C1], ids=["P1211", "P212121", "P1c1"])
    def test_fixed_point_free_is_trivial(self, triplets):
        table = stabilise(listed(triplets), [NOWHERE])
        assert (members(table), undecided(table), table.state[0], len(table.records)) == (
            TRIVIAL, set(), KNOWN_DERIVED, 0
        )  # fmt: skip

    @pytest.mark.parametrize(
        "triplets",
        [
            pytest.param(C121, id="centring-and-screw-decided"),
            pytest.param(P1M1, id="mirror"),
            pytest.param(("x,y,z", "-x,-y,z+1", "-y,x,z+1/2", "y,-x,z+1/2"), id="P42-twofold-lattice-translate"),
        ],
    )
    def test_unplaced_endpoint_undecided(self, triplets):
        table = stabilise(listed(triplets), [NOWHERE])
        assert (members(table), undecided(table), table.state[0]) == (TRIVIAL, {1}, UNKNOWN)
        assert table.records.counts() == {"special_position_unplaced": 1}

    @pytest.mark.parametrize("cell", [None, PARTIAL_CELL], ids=["none", "unknown"])
    def test_unknown_frame_is_unplaced(self, cell):
        table = stabilise(listed(P121, cell=cell), [1.0, 2.0, 3.0])
        assert (undecided(table), table.state[0]) == ({1}, UNKNOWN)
        assert table.records.counts() == {"special_position_unplaced": 1}

    def test_conflict_frame_is_inherited(self):
        table = stabilise(listed(P121, cell=IMPOSSIBLE_CELL), [[0.0, 3.0, 0.0], NOWHERE])
        assert (undecided(table, 0), undecided(table, 1)) == ({1}, {1})
        assert table.state.tolist() == [CONFLICT, UNKNOWN]
        assert [(record.code, record.subject[1]) for record in table.records] == [("special_position_unplaced", 1)]

    def test_inapplicable_frame_is_trivial(self):  # 6hbg: EM in P 2 3 on the 1 Å placeholder cell
        table = stabilise(read(name="P 2 3", cell=PLACEHOLDER, methods=EM), [[0.0, 0.0, 0.3], NOWHERE])
        assert (members(table), members(table, 1), table.state.tolist(), len(table.records)) == (
            TRIVIAL, TRIVIAL, [KNOWN_DERIVED] * 2, 0
        )  # fmt: skip

    def test_rows_sorted_by_endpoint(self):
        table = stabilise(listed(P1M1), [NOWHERE, [1.0, 2.0, 3.0]], endpoints=[7, 3])
        assert list(table.endpoint) == [3, 7]
        assert list(table.records.subject) == [3, 7]


class TestNumericStabiliser:
    @pytest.fixture
    def p121(self):
        return listed(P121, cell=CUBIC)

    def test_member_carries_lattice_shift(self, p121):
        table = stabilise(p121, [5.0, 3.0, 0.0])  # on the 2-fold axis at x = 1/2
        assert (members(table), table.state[0], len(table.records)) == (
            {(0, (0, 0, 0)), (1, (1, 0, 0))}, KNOWN_DERIVED, 0
        )  # fmt: skip

    @pytest.mark.parametrize(
        ("offset", "member", "pending", "resolved"),
        [
            pytest.param(0.0, True, set(), KNOWN_DERIVED, id="on-axis"),
            pytest.param(0.25, False, {1}, UNKNOWN, id="image-0.5A-away"),
            pytest.param(1.0, False, set(), KNOWN_DERIVED, id="image-2A-away"),
        ],
    )
    def test_image_distance_decides(self, p121, offset, member, pending, resolved):
        table = stabilise(p121, [offset, 3.0, 0.0])
        assert ((1, (0, 0, 0)) in members(table), undecided(table), table.state[0]) == (member, pending, resolved)
        assert table.records.counts() == ({"special_position_undecided": 1} if pending else {})

    @pytest.mark.parametrize(("quantum", "member"), [(0.0005, True), (0.00005, False)])
    def test_coordinate_quantum_bounds(self, p121, quantum, member):
        table = stabilise(p121, [0.0005, 3.0, 0.0], quantum=quantum)  # image 0.001 Å away
        assert ((1, (0, 0, 0)) in members(table), undecided(table)) == (member, set() if member else {1})

    @pytest.mark.parametrize(("length_quantum", "member"), [(0.0, False), (0.001, True)])
    def test_cell_quantum_widens_bound(self, length_quantum, member):
        symmetry_ = listed(P121, cell=CUBIC, cell_quantum=[length_quantum] * 3 + [0.0] * 3)
        table = stabilise(symmetry_, [5.0005, 3.0, 0.0], quantum=0.00005)
        assert ((1, (1, 0, 0)) in members(table)) == member

    @pytest.mark.parametrize(("deposited", "member"), [(None, False), (0, True)], ids=["cell", "matrix"])
    def test_frame_quantum_bounds(self, deposited, member):  # 9qm6: the matrix in use, not the cell, bounds r
        cell, quanta = (100.0, 100.0, 100.0, 90.0, 90.0, 90.0), [0.0005] * 3 + [0.5] * 3  # angles printed whole
        stated = None if deposited is None else {deposited: diagonal((0.01, 0.01, 0.01), 5e-7)}
        table = stabilise(listed(P121, cell=cell, cell_quantum=quanta, deposited=stated), [50.0, 60.0, 0.0])
        assert ((1, (1, 0, 0)) in members(table), undecided(table)) == (member, set() if member else {1})

    @pytest.mark.parametrize(("x", "member"), [(2.5, True), (0.0, False)])
    def test_origin_moves_axis(self, x, member):  # f = A·x + u: with u = (-1/4, 0, 0) the axis f_x = 0 is x = 2.5 Å
        frame = {0: diagonal((0.1, 0.1, 0.1), 5e-7, (-0.25, 0.0, 0.0))}
        table = stabilise(listed(P121, cell=CUBIC, deposited=frame), [x, 3.0, 0.0])
        assert ((1, (0, 0, 0)) in members(table), undecided(table)) == (member, set())

    def test_group_property_failure(self):
        table = stabilise(listed(P222, cell=CUBIC), [0.3, 0.3, 0.0], quantum=0.24)  # 2x, 2y inside; 2z beyond
        assert (members(table), undecided(table), table.state[0]) == (TRIVIAL, {1, 2, 3}, UNKNOWN)
        assert [record.detail["group_property_failed"] for record in table.records] == [1]

    def test_wide_bound_decides_nothing(self, p121):  # the inner bound reaches the outer one
        table = stabilise(p121, [0.0, 3.0, 0.0], quantum=0.3)
        assert (members(table), undecided(table), table.state[0]) == (TRIVIAL, {1}, UNKNOWN)
        assert [record.detail["group_property_failed"] for record in table.records] == [0]


class TestResolveTransforms:
    @pytest.mark.parametrize(
        ("sides", "origin", "resolved", "records"),
        [
            pytest.param((DOT, DOT), QuerySource.STRUCT_CONN, KNOWN_DEFINED, {}, id="dot"),
            pytest.param((ASK, DOT), QuerySource.STRUCT_CONN, UNKNOWN_DECLARED, {}, id="ask"),
            pytest.param((ABSENT, DOT), QuerySource.STRUCT_CONN, UNKNOWN, {"symmetry_absent": 1}, id="absent"),
            pytest.param((ABSENT, DOT), QuerySource.SSBOND, UNKNOWN, {"ssbond_symmetry_absent": 1}, id="ssbond"),
        ],
    )
    def test_unstated_side(self, sides, origin, resolved, records):
        table = resolve(listed(P1211), [sides], origin=origin)
        assert relative(table)[1] == resolved
        assert table.records.counts() == records

    @pytest.mark.parametrize(
        ("text", "resolved", "code"),
        [("4_8105", CONFLICT, "symmetry_code_grammar"), ("2_5512", UNKNOWN, "symmetry_code_ambiguous")],
    )
    def test_ungrammatical_code(self, text, resolved, code):
        table = resolve(listed(P1211), [(text, DOT)])
        assert (side(table, 0, 0), relative(table)[1]) == (resolved, resolved)
        assert table.records.counts() == {code: 1}

    def test_ambiguous_code_names_reading(self):
        assert resolve(listed(P1211), [("1555", DOT)]).records[0].detail["reading"] == "1_555"

    def test_relative_transform_is_exact(self):
        table = resolve(listed(P1211), [("1_555", "2_655")])
        assert relative(table) == (op("-x+1,y+1/2,-z"), KNOWN_DERIVED)
        assert table.transforms.code[table.relative[0]] == 2655

    @pytest.mark.parametrize(
        ("sides", "operation", "code"),
        [
            pytest.param(("2_555", "2_655"), "x-1,y,z", 1455, id="identity-coset-one-cell-back"),
            pytest.param(("2_155", "2_955"), "x-8,y,z", -1, id="shift-beyond-four-cells"),  # no code, still KNOWN
        ],
    )
    def test_derived_code(self, sides, operation, code):
        table = resolve(listed(P121), [sides])
        assert relative(table) == (op(operation), KNOWN_DERIVED)
        assert table.transforms.code[table.relative[0]] == code

    @pytest.mark.parametrize(
        ("symmetry_", "sides"),
        [
            pytest.param(listed(P422), ("7_555", "7_555"), id="5gsk-same-code"),
            pytest.param(read(name="C 4 21 2"), ("1_555", "1"), id="equal-parsed-codes-without-table"),
            pytest.param(listed(P1211), ("1_555", "1_555"), id="identity-through-the-table"),
        ],
    )
    def test_equal_codes_are_identity(self, symmetry_, sides):
        table = resolve(symmetry_, [sides])
        assert relative(table) == (IDENTITY, KNOWN_DERIVED) and table.relative[0] != 0

    @pytest.mark.parametrize(
        "table",
        [
            pytest.param([("1", "x,y,z"), ("2", "-x,y,-z")], id="id-disagreeing-with-the-setting"),
            pytest.param([("1", "x,y,z"), ("2", "-x,y,-z"), ("2", "x,-y,-z")], id="id-naming-two-operations"),
        ],
    )
    def test_contested_equal_codes(self, table):  # still the identity
        transforms = resolve(read(name="P 1 21 1", table=table), [("2_555", "2_555")])
        assert relative(transforms) == (IDENTITY, KNOWN_DERIVED)
        assert (side(transforms, 0, 0), side(transforms, 0, 1)) == (CONFLICT, CONFLICT)

    def test_codes_without_table_unknown(self):
        table = resolve(read(name="C 4 21 2"), [("2_555", "2_655")])
        assert relative(table)[1] == UNKNOWN
        assert table.records.counts() == {"space_group_setting_unresolved": 2}

    def test_absent_id_never_resolves(self):
        table = resolve(listed(P121), [("3_555", DOT)])
        assert side(table, 0, 0) == UNKNOWN
        assert dict(table.records[0].detail) == {"side": 0, "name": "", "number": -1, "code": "3_555"}

    def test_restated_unparsed_id_unknown(self):  # one of its two rows states no operation
        symmetry_ = read(table=[("1", "x,y,z"), ("2", "-x,y,-z"), ("2", "-x,y,-z,")])
        assert side(resolve(symmetry_, [("2_555", DOT)]), 0, 0) == UNKNOWN

    def test_operation_one_is_not_identity(self):  # it is whatever the file says
        table = resolve(listed(("-x,-y,z", "x,y,z")), [("1_555", DOT)])
        assert relative(table) == (op("-x,-y,z"), KNOWN_DERIVED)

    def test_disagreeing_operation_conflicts(self):
        symmetry_ = read(name="P 1 21 1", table=[("1", "x,y,z"), ("2", "-x,y,-z")])
        table = resolve(symmetry_, [("2_555", DOT), ("1_555", DOT)])
        assert (relative(table, 0)[1], relative(table, 1)) == (CONFLICT, (IDENTITY, KNOWN_DERIVED))
        record = table.records.where(code="operation_table_vs_setting")[0]
        assert (record.detail["file_operation"], record.detail["setting_operation"]) == ("-x,y,-z", "-x,y+1/2,-z")
        assert {ref.row for ref in record.sources} == {0, 2}  # the declaration and the symop row

    @pytest.mark.parametrize(
        ("partner", "resolved", "records"),
        [
            pytest.param([2.0, 0.0, 2.0], KNOWN_DERIVED, [], id="plausible"),
            pytest.param([4.0, 0.0, 4.0], CONFLICT, [{"distance_mA": 7071}], id="implausible"),
            pytest.param(NOWHERE, KNOWN_DERIVED, [], id="unplaced"),
        ],
    )
    def test_contact_plausibility(self, partner, resolved, records):
        table = resolve(listed(P121, cell=CUBIC), [("1_555", "2_555")], [[1.0, 0.0, 1.0], partner], rows=[(0, 1)])
        assert relative(table)[1] == resolved
        assert [dict(record.detail) for record in table.records] == records

    @pytest.mark.parametrize(("code", "resolved"), [("2_555", CONFLICT), ("1_555", KNOWN_DERIVED)])
    def test_inapplicable_frame_applies_no_operation(self, code, resolved):  # 4bog: 1_655 on a 1 Å placeholder
        table = resolve(read(name="P 2 3", cell=PLACEHOLDER, methods=EM), [("1_555", code)])
        assert relative(table)[1] == resolved
        sources = [[ref.row for ref in record.sources] for record in table.records]
        assert sources == ([[0, 7, 10]] if resolved == CONFLICT else [])  # the declaration, the cell, the method

    @pytest.mark.parametrize(
        "items",
        [
            pytest.param({"cell": PLACEHOLDER, "methods": EM}, id="cell"),
            pytest.param({"methods": EM, "deposited": {0: diagonal((1.0,) * 3, 5e-7)}}, id="transform"),
        ],
    )
    @pytest.mark.parametrize("code", ["1_655", "2_555"])
    def test_unlisted_codes_stay_unknown(self, items, code):  # C.5 under an INAPPLICABLE frame too: 1 is no identity
        table = resolve(read(**items), [("1_555", code)])
        assert (relative(table)[1], table.records.counts()) == (UNKNOWN, {"space_group_setting_unresolved": 2})

    @pytest.mark.parametrize(
        ("sides", "sites", "rows"),
        [
            pytest.param([(DOT, DOT)], (), None, id="no-unknown-side"),
            pytest.param([("1_555", "2_555")], [[1.0, 0.0, 1.0], [4.0, 0.0, 4.0]], [(0, 1)], id="contact-implausible"),
        ],
    )
    def test_every_row_is_used(self, sides, sites, rows):
        table = resolve(listed(P121, cell=CUBIC), sides, sites, rows=rows)
        used = {0, *table.per_side.ravel().tolist(), *table.relative.tolist()}
        assert used == set(range(len(table.transforms.state)))

    def test_identity_row_and_zero_payloads(self):
        t = resolve(listed(P1211), [("4_8105", ASK)]).transforms
        assert (t.state[0], t.code[0]) == (KNOWN_DEFINED, -1) and (t.rotation[0] == np.eye(3)).all()
        unresolved = t.state != KNOWN_DEFINED
        assert not t.rotation[unresolved].any() and not t.translation[unresolved].any()
        assert (t.code[unresolved] == -1).all()


class TestEdgeGeometry:
    @pytest.fixture
    def p222(self):
        return listed(P222, cell=CUBIC)

    def test_general_positions(self, p222):
        declarations = [(0, 1, DOT, DOT, COV, False), (0, 1, DOT, DOT, COORD, False)]
        *_, edges = geometry(p222, declarations, [[2, 3, 4], [3, 3, 4]])
        assert edges.count.tolist() == [[1, 1], [0, 0]] and len(set(edges.physical_edge)) == 1

    def test_special_position_counts_images(self, p222):
        *_, edges = geometry(p222, [(0, 1, DOT, DOT, COV, False)], [[0, 0, 3], [1.5, 0, 3]])
        assert edges.count.tolist() == [[2, 1]]

    @pytest.mark.parametrize(
        ("partner", "edges_", "counts"),
        [
            pytest.param([0, 2, 0], 1, [[1, 2], [1, 2]], id="on-the-op3-axis"),
            pytest.param([2, 2, 0], 2, [[1, 1], [1, 1]], id="two-angstrom-off"),
            pytest.param(None, 2, [[-1, -1], [-1, -1]], id="unplaced"),
        ],
    )
    def test_partner_stabiliser_merges_rows(self, p222, partner, edges_, counts):  # 1ca1 metalc16/18
        declarations = [(0, 1, "1_555", "1_555", COV, False), (0, 1, "1_555", "3_555", COV, False)]
        *_, edges = geometry(p222, declarations, [[1.5, 2, 1], partner])
        assert len(set(edges.physical_edge.tolist())) == edges_ and edges.count.tolist() == counts

    @pytest.mark.parametrize(("root", "edges_"), [([0, 0, 3], 1), ([1, 1, 3], 2)])
    def test_root_stabiliser_merges_rows(self, p222, root, edges_):  # double coset
        declarations = [(0, 1, "1_555", "1_555", COV, False), (0, 1, "1_555", "2_555", COV, False)]
        *_, edges = geometry(p222, declarations, [root, [1.5, 0, 3]])
        assert len(set(edges.physical_edge.tolist())) == edges_
        assert edges.count[:, 0].tolist() == ([2, 2] if edges_ == 1 else [1, 1])

    def test_unlike_rows_stay_undecided(self, p222):  # E on the axis by tolerance sets no orders apart
        declarations = [(0, 1, "1_555", "1_555", COV, False), (0, 1, "1_555", "2_555", COV, False)]
        *_, edges = geometry(p222, declarations, [[0, 0, 3], [1.5, 0, 3]], statement=[1, 2])  # sing, doub
        assert len(set(edges.physical_edge.tolist())) == 2 and edges.reading.tolist() == [UNKNOWN] * 2
        assert edges.count.tolist() == [[-1, -1], [-1, -1]]

    def test_reversed_declaration_same_edge(self, p222):
        declarations = [(0, 1, "1_555", "2_555", COV, False), (1, 0, "2_555", "1_555", COV, False)]
        *_, edges = geometry(p222, declarations, [[1, 1, 1], [-1.5, -1, 1]])
        assert edges.physical_edge.tolist() == [0, 0]

    @pytest.mark.parametrize(("code", "edges_", "reading"), [("3_555", 1, KNOWN_DERIVED), (ABSENT, 2, UNKNOWN)])
    def test_restated_code_reading(self, code, edges_, reading):  # no table resolves the code
        *_, edges = geometry(read(), [(0, 1, "1_555", code, COV, False)] * 2, [[1, 2, 3], [2, 3, 4]])
        assert len(set(edges.physical_edge.tolist())) == edges_ and edges.reading.tolist() == [reading] * 2

    @pytest.mark.parametrize(
        ("triplets", "code", "count"),
        [
            pytest.param(P222, "2_555", 1, id="involution"),
            pytest.param(P4, "2_555", 2, id="four-fold"),
            pytest.param(P222, ASK, -1, id="unknown"),
            pytest.param(P222, "4_8105", -1, id="conflict"),
        ],
    )
    def test_self_link(self, triplets, code, count):
        *_, edges = geometry(listed(triplets, cell=CUBIC), [(0, 0, "1_555", code, COV, False)], [[1, 1, 2]])
        assert edges.count.tolist() == [[count, count]] and not edges.self_bond[0]

    @pytest.mark.parametrize("code", ["4_555", "7_555"])
    def test_self_links_merge(self, code):  # by inversion or by the stabiliser
        declarations = [(0, 0, "1_555", "3_555", COV, False), (0, 0, "1_555", code, COV, False)]
        *_, edges = geometry(listed(P422, cell=CUBIC), declarations, [[3, 0, 0]])  # on the 2x axis
        assert edges.physical_edge.tolist() == [0, 0]

    @pytest.mark.parametrize(("code", "self_bond"), [("1_555", True), ("2_555", False)])
    def test_self_bond(self, p222, code, self_bond):  # E^g within tolerance of E is not E
        *_, edges = geometry(p222, [(0, 0, "1_555", code, COV, False)], [[0, 0, 3]])
        assert edges.self_bond[0] == self_bond and edges.count.tolist() == [[1, 1]]

    @pytest.mark.parametrize(("code", "edges_", "reading"), [(ASK, 1, KNOWN_DERIVED), ("2_555", 2, CONFLICT)])
    def test_inapplicable_frame_one_edge(self, code, edges_, reading):  # 6oes: no lattice, only the identity
        symmetry_ = read(name="P 2 3", cell=PLACEHOLDER, methods=EM)
        declarations = [(0, 1, "1_555", "1_555", COORD, False), (0, 1, "1_555", code, COORD, False)]
        *_, edges = geometry(symmetry_, declarations, [[0.1, 0.2, 0.3], [0.2, 0.3, 2.3]])
        assert len(set(edges.physical_edge.tolist())) == edges_ and edges.reading.tolist() == [reading] * 2

    def test_inapplicable_frame_without_group_counts_once(self):  # 9h03 with H-M '?': Stab(E) is {I}, no row holds I
        symmetry_ = read(name=ASK, cell=PLACEHOLDER, methods=EM)
        *_, edges = geometry(symmetry_, [(0, 1, "1_555", "1_555", COV, False)], [[0.1, 0.2, 0.3], [2.1, 0.2, 0.3]])
        assert symmetry_.operations == () and edges.count.tolist() == [[1, 1]]

    @pytest.mark.parametrize(("partner", "count"), [([0, 0, 3], -1), (None, -1), ([2, 3, 4], 1)])
    def test_unknown_transform_count(self, p222, partner, count):  # 1 only at a decided trivial stabiliser
        *_, edges = geometry(p222, [(0, 1, DOT, ASK, COV, False)], [[1, 2, 3], partner])
        assert edges.count.tolist() == [[1, count]]

    def test_unplaced_end_is_local(self, p222):  # 1akn ASN 361 ND2 / NAG C1
        *_, edges = geometry(p222, [(0, 1, DOT, DOT, COV, False)], [None, [1.5, 2.5, 3.5]])
        assert edges.count.tolist() == [[-1, 1]]

    @pytest.mark.parametrize(
        ("codes", "sites"),
        [
            pytest.param((DOT, DOT), [None, [1.5, 2.5, 3.5]], id="1akn-unplaced"),
            pytest.param((DOT, ASK), [None, [1.5, 2.5, 3.5]], id="unknown-transform"),
            pytest.param((DOT, DOT), [[0, 0, 3], [1.5, 0, 3]], id="on-an-axis"),  # tolerance makes no Conflict
        ],
    )
    def test_topology_fixed_counts_once(self, p222, codes, sites):  # 1akn ALA 548 N
        *_, edges = geometry(p222, [(0, 1, *codes, COV, True)], sites)
        assert edges.count.tolist() == [[1, 1]]


def transforms(*triplets, states=None) -> Transforms:
    operations = [op(t) for t in triplets]
    return Transforms(
        np.array([o.rotation for o in operations], np.int8).reshape(-1, 3, 3),
        np.array([o.translation for o in operations], np.int16).reshape(-1, 3),
        np.full(len(operations), -1, np.int32),
        np.array(states or [KNOWN_DERIVED] * len(operations), np.uint8),
    )


class TestCartesian:
    def test_identity_matrix(self):
        matrices = cartesian(read(cell=CUBIC).frame, transforms("x,y,z"))
        np.testing.assert_allclose(matrices, np.eye(4)[None], atol=1e-12)

    def test_twofold_is_orthogonal(self):
        frame = read(cell=(10.0, 12.0, 14.0, 90.0, 110.0, 90.0)).frame
        matrix = cartesian(frame, transforms("-x,y+1/2,-z"))[0]
        np.testing.assert_allclose(matrix[:3, :3] @ matrix[:3, :3].T, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(matrix[:3, 3], [0.0, 6.0, 0.0], atol=1e-12)

    def test_unknown_row_is_nan(self):
        assert np.isnan(cartesian(read(cell=CUBIC).frame, transforms("x,y,z", states=[UNKNOWN]))).all()

    def test_origin_kept(self):  # x ↦ O·[R·(A·x + u) + τ − u]: u moves the two-fold's axis
        frame = read(cell=CUBIC, deposited={0: diagonal((0.1, 0.1, 0.1), 5e-7, (-0.25, 0.0, 0.0))}).frame
        np.testing.assert_allclose(cartesian(frame, transforms("-x,y,-z"))[0, :3, 3], [5.0, 0.0, 0.0], atol=1e-12)

    @pytest.mark.parametrize(
        "items",
        [{}, {"cell": IMPOSSIBLE_CELL}, {"cell": PLACEHOLDER, "methods": EM}],
        ids=["none", "conflict", "inapplicable"],
    )
    def test_unknown_frame_refused(self, items):
        with pytest.raises(InsufficientInformationError) as caught:
            cartesian(read(**items).frame, transforms("x,y,z"))
        assert caught.value.code == "frame_unknown"


@pytest.fixture(scope="module")
def shipped() -> list[tuple[str, dict]]:
    """Every shipped setting name and alias with its entry."""
    text = (
        resources.files("ofoldx.reference.space_groups")
        .joinpath("space_group_operators.json")
        .read_text(encoding="utf-8")
    )
    return [(name, entry) for entry in json.loads(text)["settings"] for name in (entry["name"], *entry["aliases"])]


class TestShippedTable:
    def test_every_setting_is_a_group(self, shipped):
        for name, entry in shipped:
            setting = read(name=name).setting
            assert setting.name == entry["name"], name
            if name != entry["name"]:  # an alias names its entry's setting: checked once, by its own name
                continue
            operations = setting.operations
            cosets = {(o.rotation, tuple(t % 24 for t in o.translation)) for o in operations}
            products = {(p.rotation, tuple(t % 24 for t in p.translation))
                        for a in operations for b in operations for p in (a.compose(b),)}  # fmt: skip
            assert operations[0] == IDENTITY, name
            assert len(operations) == len(entry["operators"]), name
            assert products == cosets and len(cosets) == len(operations), name

    def test_trivial_iff_torsion_free(self, shipped):  # every non-identity element fixed-point-free
        for name, entry in shipped:
            trivial = stabilise(read(name=name), [NOWHERE]).state[0] == KNOWN_DERIVED
            assert trivial == (entry["number"] % 1000 in TORSION_FREE), name
