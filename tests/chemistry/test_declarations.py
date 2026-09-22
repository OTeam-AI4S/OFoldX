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

from dataclasses import replace

import numpy as np
import pytest

from ofoldx.chemistry.declarations import (
    Assumption,
    AssumptionRule,
    InUse,
    LinkKind,
    OrderToken,
    Restatement,
    Strength,
    declaration_table,
    endpoint_table,
    in_use,
    join_declarations,
    merge_edge_fields,
    polymer_link_declarations,
    recipe_declarations,
)
from ofoldx.data.molecular import BondKind, BondOrigin
from ofoldx.io import parse_source, read_structure
from ofoldx.provenance import Basis, Resolution, SourceRefs, resolution, state

from .builders import ABSENT, DECLARED, KNOWN_ASSUMED, KNOWN_DEFINED, UNKNOWN_DECLARED, UNKNOWN_DEFINED, occurrences

REFS = SourceRefs(*(np.zeros(0, t) for t in (np.int16, np.int64, np.int16)), (), (), None)
# a file-scope S-S ligand (no CCD entry) whose bond an intra-residue struct_conn row restates
LIGAND = """data_QQQ
_chem_comp.id QQQ
loop_
_chem_comp_atom.comp_id
_chem_comp_atom.atom_id
_chem_comp_atom.type_symbol
QQQ S1 S
QQQ S2 S
_chem_comp_bond.comp_id QQQ
_chem_comp_bond.atom_id_1 S1
_chem_comp_bond.atom_id_2 S2
_chem_comp_bond.value_order {order}
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.auth_seq_id
_atom_site.auth_asym_id
HETATM 1 S S1 QQQ A 1 . 0.000 0.000 0.000 1 A
HETATM 2 S S2 QQQ A 1 . 2.050 0.000 0.000 1 A
_struct_conn.id c1
_struct_conn.conn_type_id {type}
_struct_conn.ptnr1_label_asym_id A
_struct_conn.ptnr1_label_comp_id QQQ
_struct_conn.ptnr1_label_seq_id .
_struct_conn.ptnr1_label_atom_id S1
_struct_conn.ptnr2_label_asym_id A
_struct_conn.ptnr2_label_comp_id QQQ
_struct_conn.ptnr2_label_seq_id .
_struct_conn.ptnr2_label_atom_id S2
_struct_conn.ptnr1_symmetry .
_struct_conn.ptnr2_symmetry .
"""
TYPED = state(Resolution.UNKNOWN, Basis.REFERENCE)  # an AROM, DELO, PI or POLY reference bond
CONFLICT = state(Resolution.CONFLICT, Basis.NONE)
SING = "covale\n_struct_conn.pdbx_value_order sing"  # a covale row stating a single bond


def rows(n: int = 2, **columns):
    r"""``n`` covalent declarations of one physical edge between endpoints 0 and 1."""
    base = {
        "physical_edge": np.zeros(n),
        "transform": np.zeros(n),
        "kind": np.full(n, BondKind.COVALENT),
        "kind_state": np.full(n, KNOWN_DEFINED),
        "restatement": np.full(n, Restatement.NEW_INTER),
    }
    return declaration_table(np.zeros(n), np.ones(n), **(base | columns))


def codes(records) -> list[str]:
    return [records[k].code for k in range(len(records))]


class TestMergeEdgeFields:
    def test_declared_against_intrinsic(self):
        table = rows(
            order_token=[OrderToken.DOUB, 0],
            order_declared=[2, 0],
            order_declared_state=[DECLARED, UNKNOWN_DECLARED],
            order_defined=[0, 1],
            order_defined_strength=[0, Strength.INTRINSIC],
        )
        edges, records = merge_edge_fields(table, REFS)
        assert len(edges.first) == 1 and resolution(edges.order_state[0]) == Resolution.CONFLICT
        assert codes(records) == ["type_vs_value"]

    def test_declared_overrides_implied(self):
        table = rows(
            order_token=[OrderToken.DOUB, 0],
            order_declared=[2, 0],
            order_declared_state=[DECLARED, ABSENT],
            order_defined=[0, 1],
            order_defined_strength=[0, Strength.IMPLIED],
        )
        edges, records = merge_edge_fields(table, REFS)
        assert (edges.order[0], edges.order_state[0]) == (2, DECLARED)
        assert codes(records) == ["implied_value_overridden"]

    def test_two_declared_values(self):
        table = rows(order_token=[1, 2], order_declared=[1, 2], order_declared_state=[DECLARED, DECLARED])
        _, records = merge_edge_fields(table, REFS)
        assert codes(records) == ["declaration_vs_declaration"]

    def test_null_stays_unknown(self):
        edges, records = merge_edge_fields(rows(1, order_declared_state=[UNKNOWN_DECLARED]), REFS)
        assert (edges.order_state[0], len(records)) == (UNKNOWN_DECLARED, 0)

    def test_coordination_restatement(self):
        table = rows(
            1,
            kind=[BondKind.COORDINATION],
            restatement=[Restatement.RESTATEMENT],
            reference_order=[2],
        )
        edges, records = merge_edge_fields(table, REFS)
        assert (edges.kind[0], edges.order[0]) == (BondKind.COORDINATION, 2)
        assert codes(records) == ["kind_order_mismatch"]

    @pytest.mark.parametrize(
        ("order", "row", "token", "edge", "conflicts"),
        [
            ("?", "disulf", OrderToken.NONE, (1, KNOWN_DEFINED), []),
            ("AROM", "covale", OrderToken.AROM, (0, TYPED), []),  # the dictionary default sing fills no AROM
            ("AROM", SING, OrderToken.AROM, (0, CONFLICT), ["declaration_vs_reference"]),
        ],
    )
    def test_reference_order_restated(self, library, order, row, token, edge, conflicts):  # a ? is no AROM type
        source = parse_source(LIGAND.format(order=order, type=row), format="mmcif")
        structure, report = read_structure(source, library=library(()), on_conflict="record")
        system = structure.system
        assert report.definitions.by_id["QQQ"].bond_order_token.tolist() == [token]
        assert ((system.bond_order[0], system.bond_order_state[0]), codes(report.records)) == (edge, conflicts)

    def test_self_bond_is_no_edge(self):
        edges, records = merge_edge_fields(rows(1, self_bond=[True]), REFS)
        assert (len(edges.first), codes(records)) == (0, ["self_bond"])

    def test_assumed_order(self):
        table = rows(1, order_declared_state=[UNKNOWN_DECLARED])
        assumption = Assumption(AssumptionRule.UNSTATED_COVALENT_ORDER_SINGLE, "order", 0, 1)
        edges, _ = merge_edge_fields(table, REFS, [assumption])
        assert (edges.order[0], edges.order_state[0]) == (1, KNOWN_ASSUMED)

    def test_candidate_is_no_edge(self):
        edges, _ = merge_edge_fields(rows(1, existence=[UNKNOWN_DEFINED]), REFS)
        assert len(edges.first) == 0


class TestInUse:
    def test_bits(self, archive):
        gly = archive("GLY")
        occ = occurrences(gly, observed={"CA"})
        endpoints = endpoint_table([0], [gly.atom_names.index("C")], [-1], [0])
        table = declaration_table([0], [-1])
        bits = in_use(occ, endpoints, table)
        assert bits[gly.atom_names.index("CA")] == InUse.OBSERVED | InUse.PLACED
        assert bits[gly.atom_names.index("C")] == InUse.BONDED


class TestPolymerLinkDeclarations:
    @pytest.fixture
    def gly(self, archive):
        return archive("GLY")

    def test_junction_row(self, gly):
        names = {name: k for k, name in enumerate(gly.atom_names)}
        endpoints, table = polymer_link_declarations(
            np.array([[0, 1]]),
            np.array([LinkKind.JUNCTION]),
            np.array([KNOWN_DEFINED]),
            np.zeros(1),
            occurrences(gly, gly),
            (gly, gly),
            names,
        )
        atoms = [gly.atom_names[endpoints.atom[e]] for e in (table.first[0], table.second[0])]
        assert atoms == ["C", "N"]
        assert (table.order_defined[0], table.order_defined_strength[0]) == (1, Strength.IMPLIED)
        assert table.topology_fixed[0] and table.origin[0] == BondOrigin.JUNCTION
        assert table.count[0].tolist() == [1, 1]

    def test_unlocated_row(self, gly):
        names = {name: k for k, name in enumerate(gly.atom_names)}
        endpoints, table = polymer_link_declarations(
            np.array([[-1, 0]]),
            np.array([LinkKind.UNLOCATED]),
            np.array([UNKNOWN_DEFINED]),
            np.zeros(1),
            occurrences(gly),
            (gly,),
            names,
        )
        assert gly.atom_names[endpoints.atom[table.first[0]]] == "N" and table.second[0] == -1
        assert not table.topology_fixed[0]

    def test_undefined_side_refused(self, gly, invalid):
        undefined = replace(occurrences(gly, gly), definition=np.array([0, -1], dtype=np.int32))
        names = {name: k for k, name in enumerate(gly.atom_names)}
        args = (np.array([LinkKind.JUNCTION]), np.array([KNOWN_DEFINED]), np.zeros(1), undefined, (gly,), names)
        invalid(polymer_link_declarations, np.array([[0, 1]]), *args)


class TestJoinDeclarations:
    def test_equal_endpoints_merge(self):
        a = endpoint_table([0, 1], [3, 4], [-1, -1], [7, 8])
        b = endpoint_table([1, 2], [4, 5], [-1, -1], [8, 9])
        endpoints, table = join_declarations([(a, declaration_table([0], [1])), (b, declaration_table([0], [1]))])
        assert len(endpoints.occurrence) == 3
        assert table.second[0] == table.first[1]


class TestRecipeDeclarations:
    @pytest.fixture
    def recipe(self, archive):
        gly = archive("GLY")

        def build(first: str, second: str, order: int):
            a, b = gly.atom_names.index(first), gly.atom_names.index(second)
            ends = (np.array([[0, a, 0]]), np.array([[0, b, 1]]))
            return recipe_declarations(
                *ends, np.array([BondKind.COVALENT]), np.array([order]), [()], occurrences(gly), (gly,)
            )

        return build

    @pytest.mark.parametrize(
        ("first", "second", "restatement"),
        [("CA", "C", Restatement.RESTATEMENT), ("N", "C", Restatement.NEW_INTRA)],
        ids=["restates", "new"],
    )
    def test_restatement_class(self, recipe, first, second, restatement):
        _, table = recipe(first, second, 1)
        assert table.restatement[0] == restatement

    @pytest.mark.parametrize(("order", "state"), [(2, DECLARED), (0, ABSENT)], ids=["stated", "unstated"])
    def test_caller_order(self, recipe, order, state):
        _, table = recipe("N", "C", order)
        assert (table.order_token[0], table.order_declared_state[0]) == (order, state)
        assert table.origin[0] == BondOrigin.RECIPE
