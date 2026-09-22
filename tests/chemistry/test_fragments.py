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

from types import MappingProxyType

import numpy as np
import pytest

from ofoldx.chemistry.definitions import ComponentItems
from ofoldx.chemistry.fragments import (
    TERMINAL_CAPS,
    LinkingFamily,
    PortCondition,
    PortStatus,
    ValenceClass,
    anomeric_carbons,
    cap_link,
    findings,
    fragment_classes,
    junction_ports,
    ports,
    sigma_lone_pair,
    terminal_fragments,
    valence_class,
    valence_target,
)

from .builders import edit_atom

S, C = PortStatus, PortCondition


def atom(component, name: str) -> int:
    return component.atom_names.index(name)


def names(component, atoms) -> set[str]:
    return {component.atom_names[a] for a in atoms}


def side(component, record, which: int) -> tuple:
    atom_index = (record.previous, record.next)[which]
    designated = (record.previous_designated, record.next_designated)[which]
    name = component.atom_names[atom_index] if atom_index >= 0 else None
    return name, record.status[which], record.condition[which], names(component, designated)


class TestTerminalCaps:
    def test_four_approved_directions(self):
        directions = {(cap.component_id, cap.side, cap.atom, cap.partner_port) for cap in TERMINAL_CAPS.values()}
        assert directions == {
            ("ACE", "next", "C", "previous"),
            ("FOR", "next", "C", "previous"),
            ("NH2", "previous", "N", "next"),
            ("NME", "previous", "N", "next"),
        }


class TestPorts:
    @pytest.mark.parametrize(
        ("component_id", "previous", "next_"),
        [
            ("ALA", ("N", S.VERIFIED, C.NONE, {"H2"}), ("C", S.VERIFIED, C.NONE, {"OXT", "HXT"})),
            # witnesses: CRO has no atom named N or C; 7WC has no CD, and CG (Y-flagged OD2/HD2) is never taken
            ("CRO", ("N1", S.VERIFIED, C.NONE, {"H2"}), ("C3", S.VERIFIED, C.NONE, {"OXT", "HXT"})),
            ("7WC", ("N", S.VERIFIED, C.NONE, {"H"}), (None, S.UNVERIFIED, C.LEAVING_GROUP, set())),
            ("DA", ("P", S.VERIFIED, C.NONE, {"OP3", "HOP3"}), ("O3'", S.VERIFIED, C.NONE, {"HO3'"})),
            ("0NC", (None, S.NO_SIDE, C.NONE, set()), (None, S.UNVERIFIED, C.LEAVING_GROUP, set())),
            ("NAG", (None, S.NO_RULE, C.NONE, set()), (None, S.NO_RULE, C.NONE, set())),
        ],
    )
    def test_ccd_ports(self, archive, component_id, previous, next_):
        component = archive(component_id)
        record = ports(component)
        assert (side(component, record, 0), side(component, record, 1)) == (previous, next_)

    def test_atomless_sides_undefined(self, items, define):
        record = ports(define(items({"id": "NEW", "type": "L-peptide linking"})))
        assert (record.status, record.previous, record.next) == ((S.UNDEFINED, S.UNDEFINED), -1, -1)
        assert (record.previous_name, record.next_name) == ("N", "C")

    @pytest.mark.parametrize(("token", "status"), [("?", S.TYPE_UNKNOWN), ("FOO LINKING", S.NO_RULE)])
    def test_type_without_rule(self, items, glycine, define, token, status):
        columns = glycine()
        columns["component"]["type"] = token
        assert ports(define(items(**columns))).status == (status, status)

    @pytest.mark.parametrize(
        ("category", "item", "row", "token", "condition"),
        [("atoms", "type_symbol", 0, "C", C.ELEMENT), ("bonds", "value_order", 0, "DOUB", C.BOND_GRAPH)],
        ids=["element", "bond-graph"],  # N as carbon; N=CA
    )
    def test_failed_condition(self, items, glycine, define, category, item, row, token, condition):
        columns = glycine()
        columns[category][item][row] = token
        record = ports(define(items(**columns)))
        assert (record.status[0], record.condition[0]) == (S.UNVERIFIED, condition)

    def test_candidates_disagree(self, items, glycine, define):
        columns = glycine()
        atoms, bonds = columns["atoms"], columns["bonds"]
        rows = [("NZ", "N", "N", "Y", "N"), ("HZ1", "H", "Y", "Y", "N"), ("HZ2", "H", "N", "Y", "N")]
        atoms["pdbx_n_terminal_atom_flag"] = ["N"] * 10
        for name, symbol, leaving, n_terminal, c_terminal in rows:
            for item, value in zip(
                atoms, [name, symbol, 0.0, "N", leaving, "N", n_terminal, c_terminal, 0.0, 0.0, 0.0]
            ):
                atoms[item].append(value)
        for pair in (("CA", "NZ"), ("NZ", "HZ1"), ("NZ", "HZ2")):
            for item, value in zip(bonds, [*pair, "SING", "N", "N"]):
                bonds[item].append(value)
        record = ports(define(items(**columns)))
        assert (record.status[0], record.condition[0]) == (S.UNVERIFIED, C.CANDIDATES_DISAGREE)

    def test_two_designated_fragments(self, items, glycine, define):
        columns = glycine()
        columns["atoms"]["pdbx_leaving_atom_flag"][5] = "Y"  # H as well as H2
        component = define(items(**columns))
        record = ports(component)
        assert (record.status[0], names(component, record.previous_designated)) == (S.VERIFIED, {"H", "H2"})
        assert "two_fragment_port" in dict(findings(component))


class TestJunctionPorts:
    def test_peptide_pair(self, archive):
        alanine = archive("ALA")
        assert junction_ports(alanine, alanine, LinkingFamily.PEPTIDE) == (atom(alanine, "C"), atom(alanine, "N"))

    def test_undefined_side_atomless(self, archive, items, define):
        undefined = define(items({"id": "NEW", "type": "L-PEPTIDE LINKING"}))
        assert junction_ports(archive("ALA"), undefined, LinkingFamily.PEPTIDE) == (atom(archive("ALA"), "C"), -1)

    @pytest.mark.parametrize(
        ("left", "right", "family"),
        [
            ("ALA", "DA", LinkingFamily.PEPTIDE),
            ("ALA", "0NC", LinkingFamily.PEPTIDE),
            ("7WC", "ALA", LinkingFamily.PEPTIDE),
            ("ALA", "ALA", LinkingFamily.NUCLEIC),
        ],
        ids=["family-mismatch", "no-side", "unverified", "other-family"],
    )
    def test_no_junction(self, archive, left, right, family):
        assert junction_ports(archive(left), archive(right), family) is None


class TestCapLink:
    def test_leading_cap(self, archive):
        cap, alanine = archive("ACE"), archive("ALA")
        assert cap_link(cap, alanine) == (atom(cap, "C"), atom(alanine, "N"))

    def test_trailing_cap(self, archive):
        cap, alanine = archive("NH2"), archive("ALA")
        assert cap_link(alanine, cap) == (atom(cap, "N"), atom(alanine, "C"))

    def test_missing_cap_atom(self, items, define, archive):
        cap = define(items({"id": "ACE", "type": "NON-POLYMER"}, atoms={"atom_id": ["CH3"], "type_symbol": ["C"]}))
        assert cap_link(cap, archive("ALA")) is None and "port_atom_absent" in dict(findings(cap))

    def test_unverified_partner(self, items, glycine, define, archive):
        columns = glycine()
        columns["bonds"]["value_order"][0] = "DOUB"
        assert cap_link(archive("ACE"), define(items(**columns))) is None
        assert cap_link(archive("ACE"), archive("NAG")) is None


class TestTerminalFragments:
    def test_carbonyl_port(self, archive):
        alanine = archive("ALA")
        found = {
            frozenset(names(alanine, f.atoms)): (f.weight, f.designated)
            for f in terminal_fragments(alanine, atom(alanine, "C"))
        }
        assert found[frozenset({"O"})] == (2, None)
        assert found[frozenset({"OXT", "HXT"})] == (1, "next")
        body = next(f for f in found if "CA" in f)
        assert "H2" in body and found[body] == (1, None)  # holds a Y-flagged atom of the other port: not designated

    def test_amine_port(self, archive):
        alanine = archive("ALA")
        designated = [names(alanine, f.atoms) for f in terminal_fragments(alanine, atom(alanine, "N")) if f.designated]
        assert designated == [{"H2"}]

    def test_ring_neighbours_excluded(self, archive):
        proline = archive("PRO")
        roots = {proline.atom_names[f.root] for f in terminal_fragments(proline, atom(proline, "CD"))}
        assert roots == {"HD2", "HD3"}


class TestFindings:
    def test_flagged_ring_atom_unusable(self, ccd, define):
        component = define(edit_atom(ccd["PRO"], "pdbx_leaving_atom_flag", "CG", b"Y"))
        assert ("leaving_flag_unusable", {"atom": "CG"}) in [(code, dict(p)) for code, p in findings(component)]

    def test_standard_residue_clean(self, archive):
        assert findings(archive("ALA")) == ()

    def test_unlisted_type(self, items, glycine, define):
        columns = glycine()
        columns["component"]["type"] = "FOO LINKING"
        assert "chem_comp_type_unlisted" in dict(findings(define(items(**columns))))


class TestFragmentClasses:
    def test_equivalent_hydrogens_shared(self, archive):
        lysine = archive("LYS")
        found = terminal_fragments(lysine, atom(lysine, "NZ"))
        classes = dict(zip((lysine.atom_names[f.root] for f in found), fragment_classes(lysine, atom(lysine, "NZ"))))
        assert classes["HZ1"] == classes["HZ2"] == classes["HZ3"] != classes["CE"]

    def test_boundary_orders_separate(self, archive):
        alanine = archive("ALA")
        assert len(set(fragment_classes(alanine, atom(alanine, "C")).tolist())) == 3

    def test_parity_neighbours_separate(self, archive):
        alanine = archive("ALA")
        assert len(set(fragment_classes(alanine, atom(alanine, "CA")).tolist())) == len(
            terminal_fragments(alanine, atom(alanine, "CA"))
        )

    @pytest.mark.parametrize(("symbol", "classes"), [("H", 1), ("D", 2)])
    def test_isotopes_separate(self, items, glycine, define, symbol, classes):
        columns = glycine()
        columns["atoms"]["type_symbol"][7] = symbol  # HA2
        component = define(items(**columns))
        hydrogens = [i for i, f in enumerate(terminal_fragments(component, 1)) if len(f.atoms) == 1]
        assert len({int(fragment_classes(component, 1)[i]) for i in hydrogens}) == classes

    @pytest.mark.parametrize(("chlorine", "shared"), [((-1.5, 2.4), True), ((-3.6, 1.2), False)], ids=["same", "other"])
    def test_cis_trans_separates(self, items, define, chlorine, shared):
        # CX carries two CH=CHCl fragments; where side B's chlorine sits makes its configuration side A's or not
        hydrogen = (-3.6, 1.2) if chlorine == (-1.5, 2.4) else (-1.5, 2.4)
        rows = [("CX", "C", 0, 0, 0), ("C1A", "C", 1.5, 0, 0), ("C2A", "C", 2.2, 1.2, 0), ("CLA", "Cl", 1.5, 2.4, 0)]
        rows += [("H2A", "H", 3.6, 1.2, 0), ("H1A", "H", 1.9, -1, 0), ("C1B", "C", -1.5, 0, 0)]
        rows += [("C2B", "C", -2.2, 1.2, 0), ("CLB", "Cl", *chlorine, 0), ("H2B", "H", *hydrogen, 0)]
        rows += [("H1B", "H", -1.9, -1, 0), ("HX1", "H", 0, -0.5, 0.9), ("HX2", "H", 0, -0.5, -0.9)]
        atoms = dict(zip(("atom_id", "type_symbol", *(f"pdbx_model_Cartn_{a}_ideal" for a in "xyz")), zip(*rows)))
        pairs = "CX C1A, C1A C2A, C2A CLA, C2A H2A, C1A H1A, CX C1B, C1B C2B, C2B CLB, C2B H2B, C1B H1B, CX HX1, CX HX2"
        ends = list(zip(*(pair.split() for pair in pairs.split(", "))))
        orders = ["DOUB" if {x, y} in ({"C1A", "C2A"}, {"C1B", "C2B"}) else "SING" for x, y in zip(*ends)]
        labels = ["Z" if order == "DOUB" else "N" for order in orders]
        bonds = {"atom_id_1": ends[0], "atom_id_2": ends[1], "value_order": orders, "pdbx_stereo_config": labels}
        component = define(
            items({"id": "EZX", "type": "NON-POLYMER"}, atoms={k: list(v) for k, v in atoms.items()}, bonds=bonds)
        )
        roots = [component.atom_names[f.root] for f in terminal_fragments(component, 0)]
        classes = fragment_classes(component, 0).tolist()
        assert (classes[roots.index("C1A")] == classes[roots.index("C1B")]) is shared


class TestValence:
    def test_valence_class(self):
        element = np.array([1, 6, 7, 8, 16, 30, 0], dtype=np.uint8)
        assert valence_class(element).tolist() == [ValenceClass.A] * 4 + [ValenceClass.B] * 2 + [0]

    @pytest.mark.parametrize(
        ("element", "charge", "target"),
        [(6, -1, 3), (7, 1, 4), (8, -1, 1), (17, 0, 1), (1, 1, 0), (17, -2, None), (16, 0, None)],
    )
    def test_valence_target(self, element, charge, target):
        assert valence_target(element, charge) == target


class TestSigmaLonePair:
    @pytest.mark.parametrize(
        ("component_id", "atom_name", "available"),
        [("HIS", "NE2", False), ("HIS", "ND1", False), ("LYS", "NZ", False), ("HOH", "O", True), ("ALA", "O", True)]
        + [("ACT", "OXT", True), ("CYS", "SG", True), ("ZN", "ZN", False)],
    )
    def test_pd_o5_cases(self, archive, component_id, atom_name, available):
        component = archive(component_id)
        assert bool(sigma_lone_pair(component)[atom(component, atom_name)]) is available


class TestAnomericCarbons:
    @pytest.mark.parametrize(("component_id", "carbons"), [("NAG", {"C1"}), ("FRU", {"C2"}), ("ALA", set())])
    def test_anomeric_centres(self, archive, component_id, carbons):
        component = archive(component_id)
        assert names(component, np.flatnonzero(anomeric_carbons(component))) == carbons

    def test_only_saccharides(self, ccd, define):
        nag = ccd["NAG"]
        component = MappingProxyType({**nag.component, "type": ("NON-POLYMER", nag.component["type"][1])})
        renamed = ComponentItems(component, nag.atoms, nag.numbers, nag.element, nag.bonds, nag.descriptors, nag.refs)
        assert not anomeric_carbons(define(renamed)).any()
