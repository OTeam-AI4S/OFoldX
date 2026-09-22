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

import numpy as np
import pytest

from ofoldx.chemistry.attachments import Hydrogens, resolve_attachments, rule_l
from ofoldx.chemistry.declarations import (
    AttachmentInputs,
    LinkKind,
    OrderToken,
    Restatement,
    declaration_columns,
    declaration_table,
    endpoint_table,
    in_use,
    polymer_link_declarations,
)
from ofoldx.chemistry.smiles import read_smiles
from ofoldx.data.molecular import AtomFate, BondKind, EndpointState
from ofoldx.errors import CODE_NAMES
from ofoldx.provenance import SourceRefs

from .builders import (
    ABSENT,
    DECLARED,
    KNOWN_ASSUMED,
    KNOWN_DEFINED,
    NULLS,
    UNKNOWN_DECLARED,
    edge,
    identity,
    occurrences,
    problem,
)

S = EndpointState


def names(component, atoms) -> set[str]:
    return {component.atom_names[a] for a in atoms}


def reason(outcome) -> str:
    return CODE_NAMES[outcome.reason]


def resolve(components, occ, ends, **columns):
    r"""``resolve_attachments`` over one new connection between ``ends`` ((occurrence, atom name) pairs); occurrence
    atom i is ``atom_site`` row i."""
    atoms = [components[o].atom_names.index(name) for o, name in ends]
    endpoints = endpoint_table([o for o, _ in ends], atoms, [-1, -1], [0, 0])
    edge_columns = {"physical_edge": [0], "transform": [0], "restatement": [Restatement.NEW_INTER], "count": [[1, 1]]}
    table = declaration_table([0], [1], **edge_columns | columns)
    n = len(occ.atom_ref)
    occ.atom_ref[:] = np.arange(n)
    refs = SourceRefs(
        np.zeros(n, np.int16),
        np.arange(n),
        np.full(n, -1, np.int16),
        ("atom_site",),
        (),
        None,
    )
    unsourced = (np.zeros(len(components) + 1, np.int64), np.zeros(0, np.int64))
    inputs = AttachmentInputs(
        occ, components, endpoints, table, identity(), in_use(occ, endpoints, table), refs, (), *unsourced
    )
    return resolve_attachments(inputs)


class TestRuleL:
    def test_peptide_carbon_port(self, archive):
        gly = archive("GLY")
        outcome = rule_l(problem(gly, "C", [edge(1, KNOWN_DEFINED, retention=True)], observed={"O", "CA"}), gly)
        assert (outcome.state, names(gly, outcome.atoms)) == (S.DERIVED_FROM_REFERENCE, {"OXT", "HXT"})

    def test_peptide_nitrogen_port(self, archive):
        gly = archive("GLY")
        outcome = rule_l(problem(gly, "N", [edge(1, KNOWN_DEFINED, retention=True)], observed={"CA"}), gly)
        assert (outcome.state, names(gly, outcome.atoms)) == (S.DERIVED_FROM_REFERENCE, {"H2"})

    def test_isopeptide_lysine(self, archive):
        lys = archive("LYS")
        outcome = rule_l(problem(lys, "NZ", [edge(1)], observed={"CE"}), lys)
        assert outcome.state == S.DERIVED_FROM_VALENCE
        assert names(lys, outcome.atoms) == {"HZ1"}
        assert names(lys, outcome.alternatives) == {"HZ2", "HZ3"}

    def test_hosted_hydrogens_never_leave(self, archive):  # NZ hosts an observed H no reference atom names
        lys = archive("LYS")
        outcome = rule_l(problem(lys, "NZ", [edge(1)], observed={"CE"}, observed_only=True), lys)
        assert (outcome.state, reason(outcome)) == (S.UNKNOWN, "no_balancing_fragment_absent")

    def test_neutral_lysine_charge_site(self, archive):
        lys = archive("LYS")
        outcome = rule_l(problem(lys, "NZ", observed={"CE"}, charge=0), lys)
        assert (outcome.state, names(lys, outcome.atoms)) == (S.DERIVED_FROM_VALENCE, {"HZ1"})

    def test_modelled_thiol_blocks_disulfide(self, archive):
        cys = archive("CYS")
        outcome = rule_l(problem(cys, "SG", [edge(1, KNOWN_DEFINED)], observed={"CB", "HG"}), cys)
        assert (outcome.state, reason(outcome)) == (S.UNKNOWN, "no_balancing_fragment_blocked")
        assert [(CODE_NAMES[code], names(cys, atoms)) for code, atoms in outcome.conflicts] == [
            ("placed_leaving_atom", {"HG"})
        ]

    @pytest.mark.parametrize(
        ("component", "atom", "edges", "observed"),
        [
            # b = 2, OXT/HXT designated: r = 1, and only the CA side weighs 1 (3rqd BB9 C)
            ("GLY", "C", [edge(1), edge(1)], {"O", "CA"}),
            # the methyl is in use, but {O2, C3, H4–H6} is free and weighs 1
            ("CC(=O)OC", "C2", [edge(1, retention=True)], {"C1", "O1"}),
        ],
        ids=["residue-body", "free-ester"],
    )
    def test_no_balancing_fragment(self, archive, component, atom, edges, observed):
        c = archive(component) if component.isalpha() else read_smiles(component, component_id="MAC").component
        outcome = rule_l(problem(c, atom, edges, observed=observed), c)
        assert (outcome.state, reason(outcome), outcome.conflicts) == (S.UNKNOWN, "no_balancing_fragment_absent", ())

    def test_disulfide_thiol_leaves(self, archive):
        cys = archive("CYS")
        outcome = rule_l(problem(cys, "SG", [edge(1, KNOWN_DEFINED)], observed={"CB"}), cys)
        assert (outcome.state, names(cys, outcome.atoms)) == (S.DERIVED_FROM_VALENCE, {"HG"})

    def test_amide_cap_keeps_hydrogens(self, archive):
        nh2 = archive("NH2")
        outcome = rule_l(problem(nh2, "N", [edge(1, KNOWN_DEFINED, retention=True)], observed={"HN1", "HN2"}), nh2)
        assert (outcome.state, outcome.atoms, outcome.conflicts) == (S.DERIVED_FROM_REFERENCE, (), ())

    def test_unknown_order(self, archive):
        lys = archive("LYS")
        outcome = rule_l(problem(lys, "NZ", [edge(1, UNKNOWN_DECLARED)], observed={"CE"}), lys)
        assert (outcome.state, reason(outcome)) == (S.UNKNOWN, "order_unknown")

    def test_observed_leaving_atom(self, archive):
        gly = archive("GLY")
        outcome = rule_l(problem(gly, "C", [edge(1, KNOWN_DEFINED)], observed={"O", "OXT", "CA"}), gly)
        assert outcome.state == S.CONFLICT
        assert ("observed_leaving_atom", {"OXT"}) in [(CODE_NAMES[c], names(gly, a)) for c, a in outcome.conflicts]

    def test_declared_endpoint_is_conflict(self, archive):
        gly = archive("GLY")
        outcome = rule_l(problem(gly, "C", [edge(1)], observed={"CA"}, declared=("C",)), gly)
        assert (outcome.state, reason(outcome)) == (S.CONFLICT, "endpoint_named_as_leaving")

    def test_neutral_carboxylate_gains_hydrogen(self, archive):
        act = archive("ACT")  # OXT -1 stated 0, no new edge: rho = 0 + 1 - 2 = -1 (4km4 PO4 O3)
        outcome = rule_l(problem(act, "OXT", observed={"C"}, charge=0), act)
        assert (outcome.state, outcome.atoms, outcome.added_hydrogens) == (S.DERIVED_FROM_VALENCE, (), 1)
        assert outcome.conflicts == ()

    def test_aromatic_charge_change(self, archive):
        his = archive("HIS")
        outcome = rule_l(problem(his, "ND1", observed={"CG", "CE1"}, charge=0), his)
        assert (outcome.state, reason(outcome), outcome.payload) == (S.UNKNOWN, "charge_differs", (("aromatic",),))

    @pytest.mark.parametrize(
        ("fields", "expected"),
        [
            (
                {"existence_unknown": CODE_NAMES.index("junction_unresolved_nstd_linkage_yes")},
                "junction_unresolved_nstd_linkage_yes",
            ),
            ({"symmetry_unknown": True}, "symmetry_unknown"),
        ],
    )
    def test_undetermined_inputs(self, archive, fields, expected):
        gly = archive("GLY")
        outcome = rule_l(problem(gly, "C", [edge(1, KNOWN_DEFINED)], observed={"O", "CA"}, **fields), gly)
        assert (outcome.state, reason(outcome)) == (S.UNKNOWN, expected)

    def test_coordination_is_inapplicable(self, archive):
        his = archive("HIS")
        outcome = rule_l(problem(his, "NE2", [edge(0, ABSENT, kind=2)], observed={"CD2", "CE1"}), his)
        assert outcome.state == S.INAPPLICABLE

    def test_unresolved_coordination_inapplicable(self, archive):
        his = archive("HIS")
        unusable = CODE_NAMES.index("reference_incomplete_unusable")
        p = problem(his, "NE2", [edge(0, ABSENT, kind=2)], reference_incomplete=unusable)
        assert rule_l(p, None).state == S.INAPPLICABLE

    def test_designated_group_leaves_whole(self, items, glycine, define):
        columns = glycine()
        columns["atoms"]["type_symbol"][9] = "C"  # HXT as a carbon: a two-heavy-atom group designated at C
        ester = define(items(**columns))
        outcome = rule_l(problem(ester, "C", [edge(1)], observed={"O", "CA"}), ester)
        assert (outcome.state, names(ester, outcome.atoms)) == (S.DERIVED_FROM_REFERENCE, {"OXT", "HXT"})
        assert outcome.phantom == ()

    def test_unknown_element_has_no_class(self, items, glycine, define):
        columns = glycine()
        columns["atoms"]["type_symbol"][0] = "?"  # N of unknown element: neither class A nor class B
        gly = define(items(**columns))
        outcome = rule_l(problem(gly, "N", [edge(1)], observed={"CA"}), gly)
        assert (outcome.state, reason(outcome)) == (S.UNKNOWN, "reference_incomplete_absent")

    def test_assumed_order_falsified(self, archive):
        gly = archive("GLY")  # a declared H and the designated H2 against an assumed single bond
        assumed = (("unstated_covalent_order_single", 1),)
        outcome = rule_l(problem(gly, "N", [edge(1, KNOWN_ASSUMED)], declared=("H",), assumed=assumed), gly)
        assert (outcome.state, reason(outcome)) == (S.UNKNOWN, "assumption_falsified")


class TestResolveAttachments:
    @pytest.fixture
    def dipeptide(self, archive):
        gly = archive("GLY")
        occ = occurrences(gly, gly, observed={"N", "CA", "C", "O"})
        names_ = {name: k for k, name in enumerate(gly.atom_names)}
        endpoints, table = polymer_link_declarations(
            np.array([[0, 1]], dtype=np.int32),
            np.array([LinkKind.JUNCTION], dtype=np.uint8),
            np.array([KNOWN_DEFINED], dtype=np.uint8),
            np.zeros(1, dtype=np.uint16),
            occ,
            (gly, gly),
            names_,
        )
        columns = declaration_columns(table) | {
            "physical_edge": np.zeros(1, np.int32),
            "transform": np.zeros(1, np.int32),
            "restatement": np.ones(1, np.uint8),
        }
        table = declaration_table(**columns)
        refs = SourceRefs(*(np.zeros(0, t) for t in (np.int16, np.int64, np.int16)), (), (), None)
        unsourced = (np.zeros(3, np.int64), np.zeros(0, np.int64))
        inputs = AttachmentInputs(
            occ, (gly, gly), endpoints, table, identity(), in_use(occ, endpoints, table), refs, (), *unsourced
        )
        return gly, resolve_attachments(inputs)

    def test_standard_junction(self, dipeptide):
        _, result = dipeptide
        assert result.leaving.state.tolist() == [S.DERIVED_FROM_REFERENCE] * 2
        assert result.leaving.standard.all()

    def test_leaving_atoms_left(self, dipeptide):
        gly, result = dipeptide
        n = len(gly.atom_names)
        left = {(i // n, gly.atom_names[i % n]) for i in np.flatnonzero(result.fate == AtomFate.LEFT)}
        assert left == {(0, "OXT"), (0, "HXT"), (1, "H2")}

    def test_hydrogens_materialised(self, dipeptide):
        gly, result = dipeptide
        heavy = np.array([z != 1 for z in gly.element] * 2)
        assert (result.hydrogens[heavy & (result.fate == AtomFate.RETAINED)] == Hydrogens.MATERIALISE).all()
        assert len(result.records) == 0

    def test_metal_coordination_keeps_charge(self, archive):
        zn, act = archive("ZN"), archive("ACT")  # ZN (CCD +2) and OXT (CCD -1) of a metalc row, both stated 0
        occ = occurrences(zn, act, observed={"ZN", "C", "O", "OXT", "CH3"})
        occ.charge_state[:] = DECLARED
        oxt = act.atom_names.index("OXT")
        coordination = {"kind": [BondKind.COORDINATION], "kind_state": [KNOWN_DEFINED]}
        result = resolve((zn, act), occ, [(0, "ZN"), (1, "OXT")], **coordination)
        assert result.leaving.state.tolist() == [S.INAPPLICABLE, S.DERIVED_FROM_VALENCE]
        assert (result.charge[0], result.charge[1 + oxt], result.implicit_added[1 + oxt]) == (0, 0, 1)
        assert len(result.records) == 0

    @pytest.mark.parametrize(
        ("link", "expected"),
        [
            ({"kind_state": [UNKNOWN_DECLARED]}, S.UNKNOWN),
            ({"kind": [BondKind.COVALENT], "kind_state": [KNOWN_DEFINED]}, S.CONFLICT),
        ],
        ids=["kind-unknown", "observed-leaving"],
    )
    def test_observed_oxt_stays_retained(self, archive, link, expected):
        gly = archive("GLY")  # an in-use designated fragment is no candidate: OXT keeps its observed fate
        occ = occurrences(gly, gly, observed={"N", "CA", "C", "O", "OXT"})
        result = resolve((gly, gly), occ, [(0, "C"), (1, "N")], **link)
        oxt = gly.atom_names.index("OXT")
        assert result.leaving.state[0] == expected
        assert (result.fate[oxt], result.hydrogens[oxt]) == (AtomFate.RETAINED, Hydrogens.OBSERVED_ONLY)

    @pytest.mark.parametrize(("charge_state", "named"), [(DECLARED, True), (NULLS["?"], False)], ids=["stated", "null"])
    def test_blocked_thiol_names_charge(self, archive, charge_state, named):
        cys = archive("CYS")  # a modelled HG on each disulfide SG: only a stated charge at SG is a named fact
        occ = occurrences(cys, cys, observed={"CB", "SG", "HG"})
        occ.charge_state[:] = charge_state
        single = {"order_token": [OrderToken.SING], "order_declared": [1], "order_declared_state": [DECLARED]}
        result = resolve(
            (cys, cys), occ, [(0, "SG"), (1, "SG")], kind=[BondKind.COVALENT], kind_state=[KNOWN_DEFINED], **single
        )
        blocked = result.records.where(code="placed_leaving_atom")
        assert len(blocked) == 2
        assert (cys.atom_names.index("SG") in [ref.row for ref in blocked[0].sources]) == named
