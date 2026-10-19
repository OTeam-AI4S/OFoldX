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

r"""
Pure reference-graph primitives over one component definition: linking families and the one port rule, terminal caps,
terminal fragments and their static equivalence, valence classes and targets, σ-lone-pair availability and anomeric
carbons.

Every function reads a ``ChemicalComponent`` and nothing else. Results are memoised per (component id, digest), so a
definition is traversed once however many occurrences consult it; memoised arrays are read-only.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import IntEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
from multimolecule.graph import UndirectedGraph, is_isomorphic  # the one graph toolkit
from multimolecule.graph.topology import torch as tensors  # its tensor type (no direct torch import)
from numpy.typing import NDArray

from ..data.molecular import ELEMENTS, MonomerType
from ..data.stereo import permutation_parity
from ..provenance import known

if TYPE_CHECKING:
    from .definitions import ChemicalComponent

_Z = {symbol.upper(): number for number, symbol in enumerate(ELEMENTS)}
_H, _C, _N, _O, _P = _Z["H"], _Z["C"], _Z["N"], _Z["O"], _Z["P"]
_HALOGENS = frozenset(_Z[symbol] for symbol in ("F", "CL", "BR", "I", "AT"))
_Y, _SIDES = 2, ("previous", "next")
#: Metalloids and metals by atomic number (valence classes, ``kind_element_mismatch``, ``link_metal_coordination``):
#: every known element outside the non-metals and the metalloids is a metal.
METALLOIDS: frozenset[int] = frozenset(_Z[symbol] for symbol in ("B", "SI", "GE", "AS", "SB", "TE"))
METALS: frozenset[int] = (
    frozenset(range(1, len(ELEMENTS)))
    - METALLOIDS
    - frozenset(
        _Z[symbol]
        for symbol in (
            "H",
            "HE",
            "C",
            "N",
            "O",
            "F",
            "NE",
            "P",
            "S",
            "CL",
            "AR",
            "SE",
            "BR",
            "KR",
            "I",
            "XE",
            "AT",
            "RN",
        )
    )
)


class LinkingFamily(IntEnum):
    PEPTIDE = 1
    NUCLEIC = 2


def _rows(types: tuple[str, ...], family: LinkingFamily, monomer: MonomerType, templates: tuple) -> dict:
    return {type_: (family, monomer, templates) for type_ in types}


_PEPTIDE, _NUCLEIC = LinkingFamily.PEPTIDE, LinkingFamily.NUCLEIC
_PROTEIN, _DNA, _RNA = MonomerType.PROTEIN, MonomerType.DNA, MonomerType.RNA
# The `_chem_comp.type` table: which linking rule may apply and its port templates, never port atom ids.
FAMILY_TABLE: Mapping[str, tuple[LinkingFamily, MonomerType, tuple[str | None, str | None]]] = MappingProxyType(
    {
        **_rows(("L-PEPTIDE LINKING", "D-PEPTIDE LINKING", "PEPTIDE LINKING"), _PEPTIDE, _PROTEIN, ("N", "C")),
        **_rows(("L-PEPTIDE NH3 AMINO TERMINUS", "D-PEPTIDE NH3 AMINO TERMINUS"), _PEPTIDE, _PROTEIN, (None, "C")),
        **_rows(
            ("L-PEPTIDE COOH CARBOXY TERMINUS", "D-PEPTIDE COOH CARBOXY TERMINUS"), _PEPTIDE, _PROTEIN, ("N", None)
        ),
        **_rows(
            ("L-BETA-PEPTIDE, C-GAMMA LINKING", "D-BETA-PEPTIDE, C-GAMMA LINKING"), _PEPTIDE, _PROTEIN, ("N", "CG")
        ),
        **_rows(
            ("L-GAMMA-PEPTIDE, C-DELTA LINKING", "D-GAMMA-PEPTIDE, C-DELTA LINKING"), _PEPTIDE, _PROTEIN, ("N", "CD")
        ),
        **_rows(("DNA LINKING", "L-DNA LINKING"), _NUCLEIC, _DNA, ("P", "O3'")),
        **_rows(("RNA LINKING", "L-RNA LINKING"), _NUCLEIC, _RNA, ("P", "O3'")),
        **_rows(("DNA OH 5 PRIME TERMINUS",), _NUCLEIC, _DNA, (None, "O3'")),
        **_rows(("DNA OH 3 PRIME TERMINUS",), _NUCLEIC, _DNA, ("P", None)),
        **_rows(("RNA OH 5 PRIME TERMINUS",), _NUCLEIC, _RNA, (None, "O3'")),
        **_rows(("RNA OH 3 PRIME TERMINUS",), _NUCLEIC, _RNA, ("P", None)),
    }
)
_SACCHARIDE_TYPES = frozenset(
    ("SACCHARIDE", "D-SACCHARIDE", "L-SACCHARIDE")
    + tuple(f"{hand}-SACCHARIDE, {anomer} LINKING" for hand in "DL" for anomer in ("ALPHA", "BETA"))
)
# The listed "none" row of the table: types that have no automatic junction rule. Any other type outside
# FAMILY_TABLE is unlisted (artifact finding chem_comp_type_unlisted).
_NO_RULE_TYPES = frozenset(("PEPTIDE-LIKE", "NON-POLYMER", "OTHER")) | _SACCHARIDE_TYPES

# The `_entity_poly.type` dispatch table; None = no automatic junction rule.
POLYMER_FAMILIES: Mapping[str, LinkingFamily | None] = MappingProxyType(
    {
        "polypeptide(L)": _PEPTIDE,
        "polypeptide(D)": _PEPTIDE,
        "cyclic-pseudo-peptide": _PEPTIDE,
        "polydeoxyribonucleotide": _NUCLEIC,
        "polyribonucleotide": _NUCLEIC,
        "polydeoxyribonucleotide/polyribonucleotide hybrid": _NUCLEIC,
        "peptide nucleic acid": None,
        "other": None,
        "polysaccharide(D)": None,
        "polysaccharide(L)": None,
    }
)


class PortStatus(IntEnum):
    VERIFIED = 0  # the candidate passed every applicability condition of its template
    NO_SIDE = 1  # "—": the linking class has no automatic junction on this side (terminus classes)
    NO_RULE = 2  # "none": a known type without an automatic junction rule (listed or unlisted)
    UNVERIFIED = 3  # the template's conditions failed: no port, finding port_unverified (never a plausible atom)
    UNDEFINED = 4  # linking class known, no atom definition: the junction exists, its endpoint has no atom
    TYPE_UNKNOWN = 5  # _chem_comp.type is not KNOWN (`?`, absent or in conflict): no rule applies


class PortCondition(IntEnum):
    NONE = 0
    ATOM_ABSENT = 1
    ELEMENT = 2
    BOND_GRAPH = 3
    LEAVING_GROUP = 4
    CANDIDATES_DISAGREE = 5


@dataclass(frozen=True)
class Ports:
    r"""The port record of one component, per side (previous, next); see ``ports``."""

    previous_name: str | None
    next_name: str | None
    previous: int
    next: int
    previous_designated: tuple[int, ...]
    next_designated: tuple[int, ...]
    status: tuple[PortStatus, PortStatus]
    condition: tuple[PortCondition, PortCondition]


@dataclass(frozen=True)
class TerminalCap:
    component_id: str
    side: Literal["previous", "next"]
    atom: str
    partner_port: Literal["previous", "next"]


# wwPDB cap conventions, exactly these four directions.
TERMINAL_CAPS: Mapping[str, TerminalCap] = MappingProxyType(
    {
        "ACE": TerminalCap("ACE", "next", "C", "previous"),
        "FOR": TerminalCap("FOR", "next", "C", "previous"),
        "NH2": TerminalCap("NH2", "previous", "N", "next"),
        "NME": TerminalCap("NME", "previous", "N", "next"),
    }
)


@dataclass(frozen=True)
class Fragment:
    root: int
    atoms: tuple[int, ...]
    weight: int
    designated: Literal["previous", "next"] | None


class ValenceClass(IntEnum):
    A = 1
    B = 2


_CACHE: dict[tuple, object] = {}


def _cached(key: object, component: ChemicalComponent, compute: Callable[[], object]):
    slot = (key, component.component_id, component.digest)
    if slot not in _CACHE:
        _CACHE[slot] = compute()
    return _CACHE[slot]


def _read_only(array: NDArray) -> NDArray:
    array.flags.writeable = False
    return array


def _neighbours(component: ChemicalComponent) -> tuple[tuple[tuple[int, int], ...], ...]:
    r"""Per atom, its ``(neighbour, bond row)`` pairs in ascending neighbour order."""

    def compute():
        lists: list[list[tuple[int, int]]] = [[] for _ in component.atom_names]
        for bond, (a, b) in enumerate(component.bond_atoms.tolist()):
            lists[a].append((b, bond))
            lists[b].append((a, bond))
        return tuple(tuple(sorted(pairs)) for pairs in lists)

    return _cached("neighbours", component, compute)


def _order(component: ChemicalComponent, bond: int) -> int:
    r"""The bond's order where KNOWN, else ``-1``."""
    return int(component.bond_order[bond]) if known(component.bond_order_state[bond]) else -1


def _fragments(component: ChemicalComponent, atom: int) -> tuple[tuple[int, tuple[int, ...], int], ...]:
    r"""``(root, atoms, boundary bond)`` of every terminal fragment at ``atom``, by root index."""

    def compute():
        graph, found = _neighbours(component), []
        for root, bond in graph[atom]:
            seen, stack, closed = {root}, [root], True
            while stack and closed:
                for other, edge in graph[stack.pop()]:
                    if edge == bond:
                        continue
                    if other == atom:
                        closed = False
                        break
                    if other not in seen:
                        seen.add(other)
                        stack.append(other)
            if closed:
                found.append((root, tuple(sorted(seen)), bond))
        return tuple(found)

    return _cached(("fragments", atom), component, compute)


# template id -> (element, kind); the kind names the bond-graph and leaving-group conditions of the port rule
_TEMPLATES = {"N": (_N, "amine"), "C": (_C, "carbonyl"), "CG": (_C, "carbonyl"), "CD": (_C, "carbonyl")}
_TEMPLATES |= {"P": (_P, "phosphate"), "O3'": (_O, "hydroxyl")}


def _bond_graph(component: ChemicalComponent, atom: int, kind: str) -> bool:
    element = component.element
    bonds = [(element[other], _order(component, bond), bond) for other, bond in _neighbours(component)[atom]]
    single = [z for z, order, _ in bonds if order == 1]
    if kind == "amine":
        return len(single) == len(bonds) and _C in single and _H in single
    if kind == "carbonyl":
        flat = [
            bond
            for z, order, bond in bonds
            if z == _O
            and order == 2
            and known(component.bond_aromatic_state[bond])
            and not component.bond_aromatic[bond]
        ]
        return len(flat) == 1 and _C in single
    if kind == "phosphate":
        return sum(z == _O for z, _, _ in bonds) >= 3
    return single.count(_C) == 1 and single.count(_H) == 1


def _leaving_kind(component: ChemicalComponent, root: int, atoms: tuple[int, ...], bond: int, kind: str) -> bool:
    r"""The fragment is the canonical leaving group of the template's kind: {H} or a single-bonded {O, H}."""
    element = component.element
    if kind in ("amine", "hydroxyl"):
        return atoms == (root,) and element[root] == _H
    others = [atom for atom in atoms if atom != root]
    return element[root] == _O and _order(component, bond) == 1 and len(others) == 1 and element[others[0]] == _H


def _test(component: ChemicalComponent, atom: int, element: int, kind: str) -> tuple[PortCondition, tuple[int, ...]]:
    r"""The first failed condition of one candidate (NONE when it passes) and the atoms of its designated
    fragments: those whose root is Y-flagged, i.e. whose Y-flagged group is attached to the candidate."""

    if component.element[atom] != element:
        return PortCondition.ELEMENT, ()
    if not _bond_graph(component, atom, kind):
        return PortCondition.BOND_GRAPH, ()
    designated = [found for found in _fragments(component, atom) if component.leaving_flag[found[0]] == _Y]
    if not any(_leaving_kind(component, *found, kind) for found in designated):
        return PortCondition.LEAVING_GROUP, ()
    return PortCondition.NONE, tuple(sorted(a for _, atoms, _ in designated for a in atoms))


def _side(component: ChemicalComponent, side: int, template: str | None) -> tuple:
    r"""``(status, condition, atom, designated, name)`` of one side."""

    if template is None:
        return PortStatus.NO_SIDE, PortCondition.NONE, -1, (), None
    if not component.atom_names:
        return PortStatus.UNDEFINED, PortCondition.NONE, -1, (), template
    element, kind = _TEMPLATES[template]
    flagged = np.flatnonzero((component.terminal_flag[:, side] == _Y) & (component.element == element))
    candidates = [int(flagged[0])] if len(flagged) == 1 else []
    if template in component.atom_names and component.atom_names.index(template) not in candidates:
        candidates.append(component.atom_names.index(template))
    if not candidates:
        return PortStatus.UNVERIFIED, PortCondition.ATOM_ABSENT, -1, (), None
    tested = [(atom, *_test(component, atom, element, kind)) for atom in candidates]
    passing = [(atom, designated) for atom, condition, designated in tested if condition == PortCondition.NONE]
    if len(passing) == 1:
        atom, designated = passing[0]
        return PortStatus.VERIFIED, PortCondition.NONE, atom, designated, component.atom_names[atom]
    if passing:
        return PortStatus.UNVERIFIED, PortCondition.CANDIDATES_DISAGREE, -1, (), None
    # the highest condition any candidate reached (the candidate nearest to passing)
    return PortStatus.UNVERIFIED, max(condition for _, condition, _ in tested), -1, (), None


def ports(component: ChemicalComponent) -> Ports:
    r"""
    The one port rule: per side, the template of the type's linking rule, candidates from the
    definition's own terminal group and the template id, verified by element, local bond graph and canonical leaving
    group. No other atom is ever taken.
    """

    def compute():
        if not known(component.chem_comp_type_state):
            sides = [(PortStatus.TYPE_UNKNOWN, PortCondition.NONE, -1, (), None)] * 2
        elif (rule := FAMILY_TABLE.get(cast(str, component.chem_comp_type).strip().upper())) is None:
            sides = [(PortStatus.NO_RULE, PortCondition.NONE, -1, (), None)] * 2
        else:
            sides = [_side(component, side, template) for side, template in enumerate(rule[2])]
        (s0, c0, a0, d0, n0), (s1, c1, a1, d1, n1) = sides
        return Ports(n0, n1, a0, a1, d0, d1, (s0, s1), (c0, c1))

    return _cached("ports", component, compute)


def _family(component: ChemicalComponent) -> LinkingFamily | None:
    if not known(component.chem_comp_type_state):
        return None
    rule = FAMILY_TABLE.get(cast(str, component.chem_comp_type).strip().upper())
    return None if rule is None else rule[0]


_LINKABLE = (PortStatus.VERIFIED, PortStatus.UNDEFINED)


def junction_ports(left: ChemicalComponent, right: ChemicalComponent, family: LinkingFamily) -> tuple[int, int] | None:
    r"""The facing ports (left next, right previous) when both components are in ``family`` and both facing sides
    are VERIFIED or UNDEFINED (``-1`` on an UNDEFINED side); otherwise None."""

    if _family(left) != family or _family(right) != family:
        return None
    facing, back = ports(left), ports(right)
    if facing.status[1] not in _LINKABLE or back.status[0] not in _LINKABLE:
        return None
    return facing.next, back.previous


def cap_link(left: ChemicalComponent, right: ChemicalComponent) -> tuple[int, int] | None:
    r"""``(cap atom, partner port)`` when exactly one of the two is a terminal cap facing the other, the partner is a
    PEPTIDE-family component whose matching port is VERIFIED, and the cap definition has the exact cap atom id.
    """

    facing = [
        (cap, partner)
        for cap, partner, side in ((left, right, "next"), (right, left, "previous"))
        if (rule := TERMINAL_CAPS.get(cap.component_id)) is not None and rule.side == side
    ]
    if len(facing) != 1:
        return None
    cap, partner = facing[0]
    rule = TERMINAL_CAPS[cap.component_id]
    port, record = _SIDES.index(rule.partner_port), ports(partner)
    if _family(partner) != _PEPTIDE or record.status[port] != PortStatus.VERIFIED:
        return None
    if rule.atom not in cap.atom_names:
        return None
    return cap.atom_names.index(rule.atom), (record.previous, record.next)[port]


def terminal_fragments(component: ChemicalComponent, atom: int) -> tuple[Fragment, ...]:
    r"""
    The terminal fragments at ``atom``: maximal, pairwise disjoint connected sets with exactly one
    boundary bond, to ``atom``; weight = the boundary bond's order (``-1`` when not KNOWN). A fragment is designated
    for the side whose port record names ``atom`` when its root is Y-flagged (its Y-flagged group is attached to
    ``atom``); a fragment that merely contains a Y-flagged atom of another port is not.
    """

    def compute():
        record = ports(component)
        sides = [
            name
            for name, port, status in zip(_SIDES, (record.previous, record.next), record.status)
            if status == PortStatus.VERIFIED and port == atom
        ]
        return tuple(
            Fragment(
                root, atoms, _order(component, bond), sides[0] if sides and component.leaving_flag[root] == _Y else None
            )
            for root, atoms, bond in _fragments(component, atom)
        )

    return _cached(("terminal", atom), component, compute)


def findings(component: ChemicalComponent) -> tuple[tuple[str, Mapping[str, object]], ...]:
    r"""
    The reference-graph findings of a definition with their payloads: ``leaving_flag_unusable`` per connected group of
    Y-flagged atoms rooting no terminal fragment at any non-terminal neighbour, ``two_fragment_port`` per verified side
    with two designated fragments, ``port_unverified`` per UNVERIFIED side, ``port_atom_absent`` for a terminal cap
    without its cap atom, ``chem_comp_type_unlisted`` for a type outside the ``_chem_comp.type`` table.
    """

    def compute():
        out, graph, record = [], _neighbours(component), ports(component)
        flagged, seen = component.leaving_flag == _Y, set()
        for start in np.flatnonzero(flagged).tolist():
            if start in seen:
                continue
            group, stack = {start}, [start]
            while stack:
                for other, _ in graph[stack.pop()]:
                    if flagged[other] and other not in group:
                        group.add(other)
                        stack.append(other)
            seen |= group
            # a terminal neighbour (degree 1) is never where a group leaves: beyond it lies the whole body
            border = {
                other for atom in group for other, _ in graph[atom] if not flagged[other] and len(graph[other]) > 1
            }
            if not any(root in group for e in border for root, _, _ in _fragments(component, e)):
                out.append(("leaving_flag_unusable", {"atom": component.atom_names[min(group)]}))
        for side, status, atom, condition in zip(
            _SIDES, record.status, (record.previous, record.next), record.condition
        ):
            if (
                status == PortStatus.VERIFIED
                and sum(component.leaving_flag[root] == _Y for root, _, _ in _fragments(component, atom)) > 1
            ):
                out.append(("two_fragment_port", {"side": side}))
            if status == PortStatus.UNVERIFIED:
                out.append(("port_unverified", {"side": side, "condition": int(condition)}))
        if (cap := TERMINAL_CAPS.get(component.component_id)) is not None and cap.atom not in component.atom_names:
            out.append(("port_atom_absent", {"atom": cap.atom}))
        kind = component.chem_comp_type.strip().upper() if known(component.chem_comp_type_state) else None
        if kind is not None and kind not in FAMILY_TABLE and kind not in _NO_RULE_TYPES:
            out.append(("chem_comp_type_unlisted", {"type": component.chem_comp_type}))
        return tuple((code, MappingProxyType(payload)) for code, payload in out)

    return _cached("findings", component, compute)


def _labels(component: ChemicalComponent) -> tuple[list, list]:
    r"""Per atom and per bond, the facts an equivalence must preserve, each with its state."""

    def compute():
        c = component
        atoms = list(zip(*(column.tolist() for column in (c.element, c.isotope, c.charge, c.charge_state))))
        atoms = [
            label + (aromatic, state)
            for label, aromatic, state in zip(atoms, c.aromatic.tolist(), c.aromatic_state.tolist())
        ]
        columns = (c.bond_order, c.bond_order_state, c.bond_aromatic, c.bond_aromatic_state)
        return atoms, list(zip(*(column.tolist() for column in columns)))

    return _cached("labels", component, compute)


def _isomorphic(component: ChemicalComponent, centre: int, first: Fragment, second: Fragment) -> bool:
    r"""A root-to-root isomorphism between two fragments at ``centre`` preserving atom and bond facts, every parity
    row inside them (compared after mapping the neighbour order) and every cis/trans row whose bond lies inside them
    (compared after mapping the reference pair: one reference end mapped to the other neighbour flips cis and trans).
    The search is ``multimolecule.graph.is_isomorphic``; the stereo rows filter its complete mappings."""

    graph, (atom_label, bond_label) = _neighbours(component), _labels(component)
    inside_a, inside_b = set(first.atoms), set(second.atoms)
    rows = {int(c): r for r, c in enumerate(component.stereo_centre.tolist())}
    ends = [frozenset(component.bond_atoms[b].tolist()) for b in component.bond_stereo_bond.tolist()]
    stereobonds = {pair: k for k, pair in enumerate(ends)}
    if sorted(atom_label[a] for a in first.atoms) != sorted(atom_label[b] for b in second.atoms):
        return False

    def cis_trans_preserved(mapping: dict[int, int]) -> bool:
        inside = [k for k, pair in enumerate(ends) if pair <= inside_a]
        for k in inside:
            if (t := stereobonds.get(frozenset(mapping[x] for x in ends[k]))) is None:
                return False
            (i, _), (ri, rj) = component.bond_atoms[component.bond_stereo_bond[k]], component.bond_stereo_reference[k]
            (ti, _), (si, sj) = component.bond_atoms[component.bond_stereo_bond[t]], component.bond_stereo_reference[t]
            mri, mrj = mapping.get(int(ri), int(ri)), mapping.get(int(rj), int(rj))
            mri, mrj = (mri, mrj) if mapping[int(i)] == ti else (mrj, mri)
            va, vb = int(component.bond_stereo_value[k]), int(component.bond_stereo_value[t])
            if (va == 0) != (vb == 0) or (va and (3 - va if (mri != si) != (mrj != sj) else va) != vb):
                return False
        return len(inside) == sum(pair <= inside_b for pair in ends)

    def parity_preserved(mapping: dict[int, int]) -> bool:
        for u in first.atoms:
            if (ra := rows.get(u)) is None:
                continue
            if (rb := rows.get(mapping[u])) is None:
                return False
            image = [-1 if n < 0 else mapping.get(n, n) for n in component.stereo_neighbors[ra].tolist()]
            target = component.stereo_neighbors[rb].tolist()
            pa, pb = int(component.stereo_parity[ra]), int(component.stereo_parity[rb])
            if sorted(image) != sorted(target) or (pa == 0) != (pb == 0):
                return False
            odd = int(permutation_parity(np.array([target.index(n) for n in image], dtype=np.int64)))
            if pa and (pa == pb) == bool(odd):
                return False
        return sum(rows.get(u) is not None for u in first.atoms) == sum(rows.get(v) is not None for v in second.atoms)

    def preserved(target) -> bool:
        mapping = dict(zip(first.atoms, target.tolist()))
        return parity_preserved(mapping) and cis_trans_preserved(mapping)

    def bond_graph(fragment: Fragment) -> tuple:
        inside = set(fragment.atoms)
        bonds = [(a, b, e) for a in fragment.atoms for b, e in graph[a] if b in inside and a < b]
        edges = tensors.tensor([(a, b) for a, b, _ in bonds], dtype=tensors.long).reshape(-1, 2)
        nodes = tensors.tensor(fragment.atoms, dtype=tensors.long)
        labels = [atom_label[a] for a in fragment.atoms], [bond_label[e] for *_, e in bonds]
        return UndirectedGraph(edges, nodes=nodes), *labels

    (a, atoms_a, bonds_a), (b, atoms_b, bonds_b) = bond_graph(first), bond_graph(second)
    return is_isomorphic(
        a,
        b,
        node_labels=atoms_a,
        other_node_labels=atoms_b,
        edge_labels=bonds_a,
        other_edge_labels=bonds_b,
        fixed_mapping={first.root: second.root},
        mapping_filter=preserved,
    )


def fragment_classes(component: ChemicalComponent, atom: int) -> NDArray[np.int32]:
    r"""
    The static equivalence class of each terminal fragment at ``atom``, as the smallest fragment index of
    its class. Two fragments share a class iff their boundary orders are equal and KNOWN, neither root is a parity
    neighbour slot of a stereocentre row at ``atom``, and an isomorphism maps one onto the other (``_isomorphic``).
    """

    def compute():
        found = terminal_fragments(component, atom)
        slots = set(component.stereo_neighbors[component.stereo_centre == atom].ravel().tolist())
        classes = list(range(len(found)))
        for j, second in enumerate(found):
            for i, first in enumerate(found[:j]):
                if (
                    classes[i] == i
                    and first.weight == second.weight >= 0
                    and first.root not in slots
                    and second.root not in slots
                    and _isomorphic(component, atom, first, second)
                ):
                    classes[j] = i
                    break
        return _read_only(np.array(classes, dtype=np.int32))

    return _cached(("classes", atom), component, compute)


_CLASS = np.full(len(ELEMENTS), ValenceClass.B, dtype=np.uint8)
_CLASS[0] = 0
_CLASS[[_H, _C, _N, _O]] = ValenceClass.A
_CLASS.flags.writeable = False


def valence_class(element: NDArray[np.uint8]) -> NDArray[np.uint8]:
    r"""Valence classes of elements: A for H, C, N, O; B for every other known element; 0 for the unknown element."""
    return _CLASS[np.asarray(element)]


def valence_target(element: int, charge: int) -> int | None:
    r"""The closed-shell valence: C 4−|q|, N 3+q, O 2+q, halogen 1+q, H 1−|q|; None for another element or a negative
    result."""

    if element == _C:
        target = 4 - abs(charge)
    elif element == _N:
        target = 3 + charge
    elif element == _O:
        target = 2 + charge
    elif element in _HALOGENS:
        target = 1 + charge
    elif element == _H:
        target = 1 - abs(charge)
    else:
        return None
    return target if target >= 0 else None


_VALENCE_ELECTRONS = {_C: 4, _N: 5, _O: 6, _Z["S"]: 6, _Z["SE"]: 6, _P: 5} | dict.fromkeys(_HALOGENS, 7)


def sigma_lone_pair(component: ChemicalComponent) -> NDArray[np.bool_]:
    r"""
    Per atom, whether it has a σ lone pair to donate: n = V_e − q − Σ reference bond orders (hydrogens
    included) ≥ 2, unless the atom is aromatic with n = 2 and three neighbours. Elements outside the table (metals,
    metalloids, H) and atoms with a charge or bond order not KNOWN are False.
    """

    def compute():
        graph, available = _neighbours(component), np.zeros(len(component.atom_names), dtype=bool)
        for atom, valence in enumerate(component.element.tolist()):
            orders = [_order(component, bond) for _, bond in graph[atom]]
            if valence not in _VALENCE_ELECTRONS or not known(component.charge_state[atom]) or -1 in orders:
                continue
            n = _VALENCE_ELECTRONS[valence] - int(component.charge[atom]) - sum(orders)
            aromatic = bool(known(component.aromatic_state[atom]) and component.aromatic[atom])
            available[atom] = n >= 2 and not (aromatic and n == 2 and len(orders) == 3)
        return _read_only(available)

    return _cached("sigma", component, compute)


def _rings(component: ChemicalComponent) -> tuple[frozenset[int], ...]:
    r"""Every simple cycle of at most 7 heavy atoms, each found from its smallest atom."""

    heavy = component.element != _H
    graph = [[other for other, _ in pairs if heavy[other]] for pairs in _neighbours(component)]
    rings: set[frozenset[int]] = set()

    def walk(path: list[int]) -> None:
        for other in graph[path[-1]]:
            if other == path[0] and len(path) >= 3:
                rings.add(frozenset(path))
            elif other > path[0] and other not in path and len(path) < 7:
                walk([*path, other])

    for start in np.flatnonzero(heavy).tolist():
        walk([start])
    return tuple(sorted(rings, key=sorted))


def anomeric_carbons(component: ChemicalComponent) -> NDArray[np.bool_]:
    r"""Per atom, whether it is an anomeric carbon: in a saccharide-class component, a ring carbon (ring size
    ≤ 7) bonded to an oxygen of the same ring and to exactly one exocyclic oxygen."""

    def compute():
        found = np.zeros(len(component.atom_names), dtype=bool)
        saccharide = known(component.chem_comp_type_state) and (
            component.chem_comp_type.strip().upper() in _SACCHARIDE_TYPES
        )
        graph, element = _neighbours(component), component.element
        for ring in _rings(component) if saccharide else ():
            for atom in ring:
                oxygens = [other for other, _ in graph[atom] if element[other] == _O]
                if element[atom] == _C and any(o in ring for o in oxygens) and sum(o not in ring for o in oxygens) == 1:
                    found[atom] = True
        return _read_only(found)

    return _cached("anomeric", component, compute)
