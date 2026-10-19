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
Rule L and its consequences: one pure derivation per native endpoint and chemical state, then atom fates, the
hydrogen plan, effective charges and valences, and the stereo plan.

``rule_l`` is a function of its ``EndpointProblem`` and the endpoint's definition alone, so ``resolve_attachments``
solves each distinct problem once and broadcasts the outcome. A problem never chooses: its outcome is a unique
derivation, or ``Unknown`` with the accounting reason, or ``Conflict`` with the records that name the contradiction.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import IntEnum, IntFlag
from itertools import combinations
from typing import NamedTuple, cast

import numpy as np
from numpy.typing import NDArray

from ..data.molecular import ELEMENTS, AtomFate, BondKind, BondOrigin, EndpointState
from ..errors import CODE_NAMES, RecordKind
from ..provenance import Basis, Records, Resolution, SubjectKind, basis, known, state
from .declarations import (
    AttachmentInputs,
    HydrogenHost,
    InUse,
    MergedEdges,
    OrderToken,
    Restatement,
    merge_edge_fields,
)
from .definitions import ChemicalComponent
from .fragments import (
    METALLOIDS,
    METALS,
    ValenceClass,
    fragment_classes,
    sigma_lone_pair,
    terminal_fragments,
    valence_class,
    valence_target,
)

_CODE = {name: index for index, name in enumerate(CODE_NAMES)}
_H = ELEMENTS.index("H")
_ONIUM = frozenset(ELEMENTS.index(symbol) for symbol in ("P", "S", "Se"))
_Y = 2
_KNOWN_DECLARED = state(Resolution.KNOWN, Basis.DECLARED)
_KNOWN_DEFINED = state(Resolution.KNOWN, Basis.DEFINED)
_KNOWN_DERIVED = state(Resolution.KNOWN, Basis.DERIVED)
_KNOWN_REFERENCE = state(Resolution.KNOWN, Basis.REFERENCE)
_UNKNOWN = state(Resolution.UNKNOWN, Basis.NONE)
_UNKNOWN_DECLARED = state(Resolution.UNKNOWN, Basis.DECLARED)
_UNKNOWN_DERIVED = state(Resolution.UNKNOWN, Basis.DERIVED)
_UNKNOWN_REFERENCE = state(Resolution.UNKNOWN, Basis.REFERENCE)
_INAPPLICABLE_DECLARED = state(Resolution.INAPPLICABLE, Basis.DECLARED)  # a `.` type: no kind, never filled
_CONFLICT = state(Resolution.CONFLICT, Basis.NONE)
_DECLARED, _DERIVED_REF, _DERIVED_VAL = (
    EndpointState.DECLARED_BY_SOURCE,
    EndpointState.DERIVED_FROM_REFERENCE,
    EndpointState.DERIVED_FROM_VALENCE,
)
_INAPPLICABLE, _UNKNOWN_STATE, _CONFLICT_STATE = (
    EndpointState.INAPPLICABLE,
    EndpointState.UNKNOWN,
    EndpointState.CONFLICT,
)
# one new edge of a problem: (kind, kind_state, order, order_state, order_token, count, retention_basis)
_KIND, _KIND_STATE, _ORDER, _ORDER_STATE, _TOKEN, _COUNT, _RETENTION = range(7)


class Hydrogens(IntEnum):
    r"""The hydrogen decision of a heavy atom: its reference hydrogens are materialised, only its observed hydrogens
    are kept, or it creates none (an undetermined atom; hydrogens and definition-less atoms)."""

    NONE = 0
    MATERIALISE = 1
    OBSERVED_ONLY = 2


class StereoAction(IntEnum):
    INHERIT = 0
    TRANSPORT = 1
    UNDETERMINED_NONPORT = 2
    CLEAR = 3


class Dependency(IntFlag):
    r"""What an endpoint outcome depends on that a later step reads, as bits of ``LeavingTable.dependencies``: an
    open-valence reference atom (its hydrogen count) and a consumer assumption."""

    OPEN_VALENCE_REFERENCE = 1 << 0
    ASSUMPTION = 1 << 1


@dataclass(frozen=True, eq=False)
class LeavingTable:
    r"""
    One row per endpoint that has a state (sorted by ``endpoint``): every endpoint problem, and the endpoints whose only
    edges are restatements (``Inapplicable``, or ``Conflict`` with the edge's conflict code as ``reason``). ``atoms``
    left, ``alternatives`` equivalent naming alternatives, ``phantom`` unobserved heavy neighbours or, at a
    coordination endpoint, its undetermined designated ligands; ``standard`` a bulk junction derivation, counted
    only.
    """

    endpoint: NDArray[np.int32]
    state: NDArray[np.uint8]
    reason: NDArray[np.uint16]
    atoms_ptr: NDArray[np.int64]
    atoms: NDArray[np.int32]
    alternatives_ptr: NDArray[np.int64]
    alternatives: NDArray[np.int32]
    phantom_ptr: NDArray[np.int64]
    phantom: NDArray[np.int32]
    added_hydrogens: NDArray[np.int8]
    dependencies: NDArray[np.uint32]
    standard: NDArray[np.bool_]


@dataclass(frozen=True, eq=False)
class StereoPlan:
    r"""Per (occurrence, definition stereocentre) whose neighbourhood changes (an unchanged centre inherits): the
    departed definition atom, the partner endpoint and merged edge of a TRANSPORT row, the ``parity_cleared_*`` code
    of an UNDETERMINED_NONPORT or CLEAR row."""

    occurrence: NDArray[np.int32]
    centre: NDArray[np.int16]
    action: NDArray[np.uint8]
    departed: NDArray[np.int32]
    partner_endpoint: NDArray[np.int32]
    partner_edge: NDArray[np.int32]
    reason: NDArray[np.uint16]


@dataclass(frozen=True, eq=False)
class AttachmentResult:
    r"""
    The attachment chemistry of one chemical state. Per occurrence atom: ``fate`` (``AtomFate``, LEFT included), the
    hydrogen decision and its record code, hydrogens added without ids, the effective formal charge and allowed
    covalent valence with states. ``records`` subjects: ATOM = occurrence atom, EDGE = declaration row, OCCURRENCE =
    native ordinal.
    """

    edges: MergedEdges
    leaving: LeavingTable
    fate: NDArray[np.uint8]
    hydrogens: NDArray[np.uint8]
    hydrogen_reason: NDArray[np.uint16]
    implicit_added: NDArray[np.int8]
    charge: NDArray[np.int8]
    charge_state: NDArray[np.uint8]
    valence: NDArray[np.int8]
    valence_state: NDArray[np.uint8]
    stereo: StereoPlan
    records: Records


@dataclass(frozen=True)
class EndpointProblem:
    r"""
    The complete argument of ``rule_l`` besides the definition.

    ``edges``: the new edges at E, sorted; a count is the physical neighbours at E's side, -1 undeterminable.
    ``fragment_atoms``: the atoms of E's terminal fragments in definition order; ``declared``, ``designated`` (the
    Y-flagged atoms of fragments rooted at a Y-flagged atom) and ``in_use`` (one ``InUse`` value per atom) are over
    them. ``declared_outside``: declared ids not among them (E itself, another reference atom, -1 for an id not in the
    reference). ``reference_incomplete``: the code index of E's ``reference_incomplete_*`` sub-reason, 0 when the
    reference is usable at E. ``body_unobserved``: E's heavy reference neighbours outside its terminal fragments that
    are not observed. ``observed_only``: E hosts an observed hydrogen with no reference counterpart, so none of
    E's reference hydrogens is a candidate. ``observable``: False for a value built from a recipe, which has no source:
    none of its atoms is unobserved, so E has no phantom neighbour.
    """

    definition: int
    atom: int
    edges: tuple[tuple[int, int, int, int, int, int, bool], ...]
    fragment_atoms: tuple[int, ...]
    declared: int
    in_use: tuple[int, ...]
    designated: int
    charge: int
    charge_state: int
    observed_only: bool
    existence_unknown: int
    symmetry_unknown: bool
    assumed: tuple[tuple[str, int], ...]
    declared_outside: tuple[int, ...] = ()
    reference_incomplete: int = 0
    body_unobserved: tuple[int, ...] = ()
    observable: bool = True


@dataclass(frozen=True)
class EndpointOutcome:
    r"""``rule_l``'s result. ``conflicts``: (code, atoms) per Conflict record; ``payload``: the payload values (and
    the list) of the state's record."""

    state: EndpointState
    reason: int
    atoms: tuple[int, ...]
    alternatives: tuple[int, ...]
    phantom: tuple[int, ...]
    added_hydrogens: int
    dependencies: int
    conflicts: tuple[tuple[int, tuple[int, ...]], ...]
    payload: tuple = field(default=())


class _Fragment(NamedTuple):
    position: int  # position in terminal_fragments
    root: int
    atoms: tuple[int, ...]
    weight: int
    in_use: bool
    declared: bool
    designated: bool
    hydrogen: bool  # a one-atom hydrogen fragment (the only kind class B admits)
    selectable: bool  # a hydrogen, or a terminal heavy atom with its hydrogens: what the derivation may choose

    @property
    def may_leave(self) -> bool:
        r"""Rule L may remove it at E: a fragment of C_E, M_E or D_E with no atom in use."""
        return not self.in_use and (self.selectable or self.declared or self.designated)


def _fragments(problem: EndpointProblem, definition: ChemicalComponent) -> list[_Fragment]:
    position = {atom: k for k, atom in enumerate(problem.fragment_atoms)}
    found = []
    for index, f in enumerate(terminal_fragments(definition, problem.atom)):
        bits = [position[a] for a in f.atoms]
        hydrogen = f.atoms == (f.root,) and definition.element[f.root] == _H
        found.append(
            _Fragment(
                index,
                f.root,
                f.atoms,
                f.weight,
                any(problem.in_use[k] for k in bits),
                any(problem.declared >> k & 1 for k in bits),
                bool(problem.designated >> position[f.root] & 1),
                hydrogen,
                not (hydrogen and problem.observed_only)  # an E hosting an unnamed H chooses none of its own
                and all(a == f.root or definition.element[a] == _H for a in f.atoms),
            )
        )
    return found


def _bonds(definition: ChemicalComponent, atom: int) -> list[tuple[int, int, bool]]:
    r"""(neighbour, order or -1 when not KNOWN, aromatic) of every reference bond at ``atom``."""
    rows = np.flatnonzero((definition.bond_atoms == atom).any(axis=1))
    return [
        (
            int(definition.bond_atoms[r].sum() - atom),
            int(definition.bond_order[r]) if known(definition.bond_order_state[r]) else -1,
            bool(known(definition.bond_aromatic_state[r]) and definition.bond_aromatic[r]),
        )
        for r in rows.tolist()
    ]


def _atoms(fragments: list[_Fragment]) -> tuple[int, ...]:
    return tuple(sorted(a for f in fragments for a in f.atoms))


class _Result(NamedTuple):
    state: EndpointState
    reason: int = 0
    atoms: tuple[int, ...] = ()
    alternatives: tuple[int, ...] = ()
    added: int = 0
    dependencies: int = 0
    conflicts: tuple[tuple[int, tuple[int, ...]], ...] = ()
    payload: tuple = ()
    phantom: tuple[int, ...] | None = None


def _unknown(
    code: str,
    atoms: tuple[int, ...] = (),
    values: tuple = (),
    listed: Iterable[int] | None = None,
    dependencies: int = 0,
) -> _Result:
    payload = (values,) if listed is None else (values, tuple(listed))
    return _Result(_UNKNOWN_STATE, _CODE[code], atoms, payload=payload, dependencies=dependencies)


def _conflict(code: str, atoms: tuple[int, ...]) -> _Result:
    return _Result(_CONFLICT_STATE, _CODE[code], conflicts=((_CODE[code], atoms),))


def _choices(groups: list[tuple[int, int]], remainder: int) -> list[tuple[int, ...]]:
    r"""Every way to take k_i of the n_i fragments of weight w_i of each class (``groups`` = (n_i, w_i)) that sums to
    ``remainder``: the class-equivalent balancing sets."""
    if not groups:
        return [()] if remainder == 0 else []
    (n, w), rest = groups[0], groups[1:]
    return [(k, *tail) for k in range(min(n, remainder // w) + 1) for tail in _choices(rest, remainder - k * w)]


def rule_l(problem: EndpointProblem, definition: ChemicalComponent | None) -> EndpointOutcome:
    r"""
    Rule L at one native endpoint: stage 1 evaluates every constraint that needs no derivation (the declarations,
    in-use leaving atoms, conflicted edges and the undetermined-input rows) and records every source contradiction;
    stage 2 derives the leaving set. A stage-1 outcome is the state; stage 2 then runs for the phantom neighbours only
    and records nothing.
    """

    p, c = problem, definition
    resolved = c is not None and p.atom >= 0 and p.reference_incomplete == 0
    fragments = _fragments(p, cast(ChemicalComponent, c)) if resolved else []
    covalent = [e for e in p.edges if e[_KIND] == BondKind.COVALENT and known(e[_KIND_STATE])]
    undetermined_kind = [e for e in p.edges if e[_KIND_STATE] in (_UNKNOWN_DECLARED, _UNKNOWN, _INAPPLICABLE_DECLARED)]
    declared = bool(p.declared) or bool(p.declared_outside)
    incomplete = p.reference_incomplete or _CODE["reference_incomplete_absent"]
    in_use = dict(zip(p.fragment_atoms, p.in_use))

    conflicts: list[tuple[int, tuple[int, ...]]] = []
    if p.atom >= 0 and p.atom in p.declared_outside:  # E itself declared leaving
        conflicts.append((_CODE["endpoint_named_as_leaving"], (p.atom,)))
    if resolved:  # a declared id outside the terminal fragments (an unresolved reference is the Unknown below)
        if -1 in p.declared_outside:
            conflicts.append((_CODE["declaration_not_in_reference"], ()))
        conflicts += [(_CODE["declaration_not_terminal"], (a,)) for a in p.declared_outside if a >= 0 and a != p.atom]
    checked = {a for f in fragments if f.declared for a in f.atoms}  # declared leaving atoms must not be in use
    if covalent:  # a designation is activated only by a new covalent bond at E
        checked |= {a for k, a in enumerate(p.fragment_atoms) if p.designated >> k & 1}
    for atom in sorted(checked):
        bits = in_use[atom]
        if bits & InUse.OBSERVED:
            conflicts.append((_CODE["observed_leaving_atom"], (atom,)))
        elif bits & InUse.PLACED:
            conflicts.append((_CODE["placed_leaving_atom"], (atom,)))
        if bits & InUse.BONDED:
            conflicts.append((_CODE["bonded_leaving_atom"], (atom,)))

    stage1: list[tuple[EndpointState, int]] = []
    if conflicts:
        stage1.append((_CONFLICT_STATE, conflicts[0][0]))
    # a conflicted kind or order: the edge's record
    if any(_CONFLICT in (e[_KIND_STATE], e[_ORDER_STATE]) for e in p.edges):
        stage1.append((_CONFLICT_STATE, 0))
    conflicting = p.existence_unknown == _CODE["conflicting_connection"]  # its record is the atom pair's
    if conflicting:
        stage1.append((_CONFLICT_STATE, p.existence_unknown))
    if declared and not resolved:  # a declaration with the reference unresolved
        stage1.append((_UNKNOWN_STATE, incomplete))
    if p.symmetry_unknown or any(e[_COUNT] == -1 for e in p.edges):  # unknown symmetry or neighbour count
        stage1.append((_UNKNOWN_STATE, _CODE["symmetry_unknown"]))
    if undetermined_kind and not covalent:
        stage1.append((_UNKNOWN_STATE, _CODE["kind_unknown"]))
    if p.existence_unknown and not conflicting:
        stage1.append((_UNKNOWN_STATE, p.existence_unknown))
    if declared and not covalent and not undetermined_kind and not conflicts:
        stage1.append((_UNKNOWN_STATE, _CODE["unsupported_coordination_substitution"]))

    derived = _derive(p, c, fragments, covalent, undetermined_kind, resolved, incomplete, bool(conflicts))
    phantom = (
        derived.phantom
        if derived.phantom is not None
        else _phantom(p, cast(ChemicalComponent, c), fragments) if resolved else ()
    )
    phantom = phantom if p.observable else ()
    if not stage1:
        return EndpointOutcome(
            derived.state,
            derived.reason,
            derived.atoms,
            derived.alternatives,
            phantom,
            derived.added,
            derived.dependencies,
            derived.conflicts,
            derived.payload,
        )
    state_, reason = stage1[0]
    left: tuple[int, ...] = ()
    if state_ == _UNKNOWN_STATE and reason != _CODE["unsupported_coordination_substitution"]:
        left = _atoms([f for f in fragments if f.declared])  # M_E leaves at an Unknown endpoint
    return EndpointOutcome(state_, reason, left, (), phantom, 0, 0, tuple(conflicts))


def _phantom(p: EndpointProblem, c: ChemicalComponent, fragments: list[_Fragment]) -> tuple[int, ...]:
    r"""E's unobserved heavy reference neighbours that stay: outside the fragments that may leave."""
    roots = [
        f.root
        for f in fragments
        if not f.may_leave and c.element[f.root] != _H and not p.in_use[p.fragment_atoms.index(f.root)] & InUse.OBSERVED
    ]
    return tuple(sorted({*p.body_unobserved, *roots}))


def _derive(
    p: EndpointProblem,
    c: ChemicalComponent | None,
    fragments: list[_Fragment],
    covalent: list[tuple],
    undetermined_kind: list[tuple],
    resolved: bool,
    incomplete: int,
    recorded: bool,
) -> _Result:
    r"""Stage 2: the leaving set from the reference valence, the new bond orders and the charge."""

    c = cast(ChemicalComponent, c)  # every reference lookup is guarded by resolved
    e = p.atom
    if not covalent and not undetermined_kind and not p.existence_unknown:
        changed = resolved and known(p.charge_state) and known(c.charge_state[e]) and p.charge != int(c.charge[e])
        metal = resolved and bool(p.edges) and c.element[e] in METALS  # a coordination endpoint, before its charge
        if metal or not changed:  # coordination only: never a removal, whatever the reference
            ligands: tuple[int, ...] = ()  # a metal's unobserved designated ligands are undetermined
            if resolved and c.element[e] in METALS:
                ligands = _atoms([f for f in fragments if f.designated and not _observed(p, f)])
            return _Result(_INAPPLICABLE, phantom=ligands)
    if not resolved:  # no usable reference
        return _Result(_UNKNOWN_STATE, incomplete)
    bonds = _bonds(c, e)
    if any(order < 0 for _, order, _ in bonds) or not known(c.charge_state[e]) or not c.element[e]:
        return _unknown("reference_incomplete_absent")  # an unknown element has no valence class
    declared = [f for f in fragments if f.declared]
    m_atoms = _atoms(declared)

    reference_charge = int(c.charge[e])
    occurrence_known = bool(known(p.charge_state))
    charge = p.charge if occurrence_known else reference_charge
    dependencies = 0

    budget, order_unknown, no_order = 0, bool(undetermined_kind), False
    for edge in covalent:  # the order budget of the new covalent edges
        if edge[_TOKEN] > OrderToken.QUAD:
            no_order = True
        elif not known(edge[_ORDER_STATE]):
            order_unknown = True
        else:
            budget += edge[_ORDER] * max(edge[_COUNT], 1)
    if no_order:  # AROM/DELO/PI/POLY change retained reference orders at E
        return _unknown("valence_model_inapplicable", m_atoms, (0,), dependencies=dependencies)
    if order_unknown:  # M_E leaves, the rest is undetermined
        return _unknown("order_unknown", m_atoms, dependencies=dependencies)

    element = int(c.element[e])
    a_class = valence_class(np.array([element]))[0] == ValenceClass.A
    v_ref = sum(order for _, order, _ in bonds)
    aromatic = any(flag for _, _, flag in bonds)
    target_ref, target = valence_target(element, reference_charge), valence_target(element, charge)
    onium = element in _ONIUM and charge == reference_charge + 1 and budget >= 1
    if charge != reference_charge and not onium:  # a charge change the valence model must cover
        if not a_class or target is None:
            return _unknown("charge_differs", m_atoms, ("class_b",), dependencies=dependencies)
        if aromatic:
            return _unknown("charge_differs", m_atoms, ("aromatic",), dependencies=dependencies)
    if a_class and target_ref is not None and v_ref < target_ref:  # an open-valence reference atom
        if aromatic or target_ref - v_ref > budget:
            return _unknown("valence_model_inapplicable", m_atoms, (1,), dependencies=dependencies)
        dependencies |= Dependency.OPEN_VALENCE_REFERENCE
    if a_class:
        if target is None:
            return _unknown("valence_model_inapplicable", m_atoms, (0,), dependencies=dependencies)
        removed_order = budget + v_ref - target
    else:
        removed_order = budget - 1 if onium else budget

    assumed = any(basis(edge[_ORDER_STATE]) == Basis.ASSUMED for edge in covalent)
    rule = next((name for name, _ in p.assumed), "")  # the rule of an assumed-order edge at E
    weight = sum(f.weight for f in declared)
    # the declared fragments exceed the removable order; a negative removed order alone contradicts no source
    if weight and weight > removed_order:
        if assumed:
            return _unknown("assumption_falsified", m_atoms, (rule,), dependencies=dependencies | Dependency.ASSUMPTION)
        return _conflict("declaration_exceeds_order", m_atoms)
    designated = [f for f in fragments if f.designated and not f.in_use] if covalent else []
    extra = [f for f in designated if not f.declared]
    removed = declared + extra
    weight += sum(f.weight for f in extra)
    if weight and weight > removed_order:  # with the designated fragments added
        if assumed:
            return _unknown("assumption_falsified", m_atoms, (rule,), dependencies=dependencies | Dependency.ASSUMPTION)
        if declared and extra:
            return _conflict("declaration_vs_designation", _atoms(removed))
        if not declared and any(basis(edge[_ORDER_STATE]) == Basis.DECLARED for edge in covalent):
            return _conflict("declaration_vs_designation", _atoms(extra))
        return _unknown("designation_exceeds_order", m_atoms, dependencies=dependencies)
    if assumed:
        dependencies |= Dependency.ASSUMPTION
    remainder = removed_order - weight
    rest = [f for f in fragments if f.selectable and not f.in_use and f not in removed]  # C_E outside M_E ∪ D_E
    admissible = [f for f in rest if a_class or f.hydrogen]

    def internal_edit(retained_out: list[_Fragment]) -> bool:
        r"""A retained non-aromatic multiple bond at E without a retention basis."""
        gone = {a for f in retained_out for a in f.atoms}
        multiple = [other for other, order, flag in bonds if order >= 2 and not flag and other not in gone]
        if not multiple or budget == 0:
            return False
        return not (designated or all(edge[_RETENTION] for edge in covalent))

    def settled(result: _Result, out: list[_Fragment]) -> _Result:
        if internal_edit(out):
            return _unknown("internal_edit_unresolved", m_atoms, dependencies=dependencies)
        return result

    if remainder > 0 and (element in METALS or element in METALLOIDS):  # a metal or metalloid balances no remainder
        return _unknown("valence_model_inapplicable", m_atoms, (0,), dependencies=dependencies)
    if remainder < 0:  # a charge change needs hydrogens the reference does not have
        result = _Result(_DERIVED_VAL, 0, _atoms(removed), (), -remainder, dependencies)
        return settled(result, removed)
    if remainder == 0:
        final = _DECLARED if removed and all(f.declared for f in removed) else _DERIVED_REF
        return settled(_Result(final, 0, _atoms(removed), dependencies=dependencies), removed)

    classes = fragment_classes(c, e)
    groups: dict[int, list[_Fragment]] = {}
    for f in admissible:
        groups.setdefault(int(classes[f.position]), []).append(f)
    keys = sorted(groups)
    solutions = _choices([(len(groups[k]), groups[k][0].weight) for k in keys], remainder)
    if len(solutions) == 1:
        chosen, alternatives = [], []
        for k, take in zip(keys, solutions[0]):
            members = sorted(groups[k], key=lambda f: c.atom_names[f.root])
            chosen += members[:take]
            if 0 < take < len(members):
                alternatives += members[take:]
        result = _Result(_DERIVED_VAL, 0, _atoms(removed + chosen), _atoms(alternatives), 0, dependencies)
        return settled(result, removed + chosen)
    if solutions:
        roots = sorted({f.root for k, counts in zip(keys, zip(*solutions)) if any(counts) for f in groups[k]})
        return settled(_unknown("ambiguous_candidates", m_atoms, (), roots, dependencies), removed)
    if not a_class and _choices([(1, f.weight) for f in rest], remainder):
        return _unknown(
            "valence_model_inapplicable", m_atoms, (0,), dependencies=dependencies
        )  # only a heavy fragment could
    # no balancing set: blocked iff some admissible set of the right weight exists and every legal set of that weight,
    # selectable or not, removes an in-use atom
    outside = [f for f in fragments if f.selectable and f not in removed and (a_class or f.hydrogen)]
    sets = [
        chosen
        for size in range(1, remainder + 1)
        for chosen in combinations(outside, size)
        if sum(f.weight for f in chosen) == remainder
    ]
    if internal_edit(removed):
        return _unknown("internal_edit_unresolved", m_atoms, dependencies=dependencies)
    # a legal set outside the selectable fragments (whole terminal fragments, none in use, beside M_E ∪ D_E) refutes
    # the Conflict and is never applied; dropping more fragments never adds an internal edit (checked above)
    free = [(1, f.weight) for f in fragments if f not in removed and not f.in_use]
    if recorded or not sets or _choices(free, remainder):
        return _unknown("no_balancing_fragment_absent", m_atoms, dependencies=dependencies)
    blocking = []
    for f in sorted({f for s in sets for f in s if f.in_use}, key=lambda f: f.position):
        atom = next(a for a in (f.root, *f.atoms) if p.in_use[p.fragment_atoms.index(a)])
        bits = p.in_use[p.fragment_atoms.index(atom)]
        code = "placed_leaving_atom" if bits & InUse.PLACED else "bonded_leaving_atom"
        blocking.append((_CODE[code], (atom,)))
    result = _unknown("no_balancing_fragment_blocked", m_atoms, dependencies=dependencies)
    return result._replace(conflicts=tuple(blocking))


def _observed(p: EndpointProblem, f: _Fragment) -> bool:
    return bool(p.in_use[p.fragment_atoms.index(f.root)] & InUse.OBSERVED)


def _per_atom(definition: ChemicalComponent) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.bool_]]:
    r"""Per definition atom: its reference hydrogens, its reference valence V_ref and whether every order at it is
    KNOWN."""
    n = len(definition.atom_names)
    hydrogens, valence, complete = np.zeros(n, np.int64), np.zeros(n, np.int64), np.ones(n, bool)
    pairs = definition.bond_atoms.astype(np.int64)
    if len(pairs):
        is_h = definition.element == _H
        np.add.at(hydrogens, pairs[:, 0], is_h[pairs[:, 1]])
        np.add.at(hydrogens, pairs[:, 1], is_h[pairs[:, 0]])
        np.add.at(valence, pairs.ravel(), np.repeat(definition.bond_order.astype(np.int64), 2))
        complete[pairs[~known(definition.bond_order_state)].ravel()] = False
    return hydrogens, valence, complete


def _csr(lists: list[tuple[int, ...]], dtype: type = np.int32) -> tuple[NDArray[np.int64], NDArray]:
    ptr = np.zeros(len(lists) + 1, dtype=np.int64)
    ptr[1:] = np.cumsum([len(values) for values in lists])
    return ptr, np.array([v for values in lists for v in values], dtype=dtype)


class _Site(NamedTuple):
    fragment_atoms: tuple[int, ...]
    designated: int
    body: tuple[int, ...]  # heavy reference neighbours outside the terminal fragments
    position: dict[int, int]  # fragment atom -> its index in fragment_atoms


_NO_SITE = _Site((), 0, (), {})


def _site(definition: ChemicalComponent, atom: int) -> _Site:
    fragments = terminal_fragments(definition, atom)
    atoms = tuple(sorted(a for f in fragments for a in f.atoms))
    position = {a: k for k, a in enumerate(atoms)}
    designated = 0
    for f in fragments:
        if definition.leaving_flag[f.root] == _Y:
            for a in f.atoms:
                if definition.leaving_flag[a] == _Y:
                    designated |= 1 << position[a]
    body = tuple(
        other for other, _, _ in _bonds(definition, atom) if other not in position and definition.element[other] != _H
    )
    return _Site(atoms, designated, body, position)


def _members(base: NDArray[np.int64], ptr: NDArray[np.int64], values: NDArray, rows: NDArray[np.bool_]) -> NDArray:
    r"""``base[k] + values`` of every CSR row ``k`` that ``rows`` selects, in row order."""
    sizes = np.diff(ptr)
    return (np.repeat(base, sizes) + values)[np.repeat(rows, sizes)]


def _rank(rows: list[tuple]) -> tuple[NDArray[np.int64], list[tuple]]:
    r"""Per row, its index among the distinct rows; and the distinct rows."""
    index: dict[tuple, int] = {}
    return np.array([index.setdefault(row, len(index)) for row in rows], dtype=np.int64), list(index)


def resolve_attachments(inputs: AttachmentInputs) -> AttachmentResult:
    r"""
    Attachment chemistry of one chemical state: merge the edge fields, solve one Rule L problem per distinct endpoint
    problem, then derive atom fates, the hydrogen plan, effective charges and allowed valences, the stereo plan and
    the records.
    """

    occ, ends, decl, defs = inputs.occurrences, inputs.endpoints, inputs.declarations, inputs.definitions
    read = bool(inputs.refs.categories)  # a value built from a recipe has no source, so no atom of it is unobserved
    edges, merge_records = merge_edge_fields(decl, inputs.refs, inputs.assumptions)
    n_end, n_edge, n_atom = len(ends.occurrence), len(edges.first), len(occ.placed)

    # ---- per merged edge: the count and retention basis at each side, an assumed order
    sizes = np.diff(edges.declaration_ptr)
    group = np.repeat(np.arange(n_edge), sizes)
    rows = edges.declaration
    flip = (decl.first[rows] != edges.first[group])[:, None]
    count_rows = np.where(flip, decl.count[rows][:, ::-1], decl.count[rows])
    retention = np.zeros((n_edge, 2), dtype=bool)
    np.logical_or.at(retention, group, np.where(flip, decl.retention_basis[rows][:, ::-1], decl.retention_basis[rows]))
    count = count_rows[edges.declaration_ptr[:-1]] if n_edge else np.zeros((0, 2), np.int8)
    fixed = np.flatnonzero(decl.topology_fixed[rows])
    fixed_groups, first_fixed = np.unique(group[fixed], return_index=True)
    count[fixed_groups] = count_rows[fixed[first_fixed]]  # a sequence or branch link's multiplicity
    count = count.astype(np.int64)
    edge_dependency = np.where(basis(edges.order_state) == Basis.ASSUMED, int(Dependency.ASSUMPTION), 0)
    restated = edges.edge_class == Restatement.RESTATEMENT
    # a kept CONECT connection; a struct_conn row with no type still declares its endpoints
    unclassified = (edges.kind_state == _UNKNOWN) & ~restated & decl.connectivity_only[rows[edges.declaration_ptr[:-1]]]
    symmetry_restated = edges.edge_class == Restatement.RESTATEMENT_SYMMETRY_UNKNOWN
    new = ~unclassified & ~restated & ~symmetry_restated
    in_conflict = (edges.order_state == _CONFLICT) | (edges.kind_state == _CONFLICT)
    in_conflict |= (edges.aromatic_state == _CONFLICT) | (edges.direction_state == _CONFLICT)
    edge_code = np.zeros(n_edge, dtype=np.int64)
    representative = {int(r): g for g, r in enumerate(rows[edges.declaration_ptr[:-1]].tolist())} if n_edge else {}
    for k in np.flatnonzero(merge_records.kind() == RecordKind.CONFLICT).tolist():
        if (g := representative.get(int(merge_records.subject[k]))) is not None:
            edge_code[g] = merge_records.code[k]

    # ---- incidences (endpoint, edge, side); a self-link is incident once
    loops = edges.first == edges.second
    inc_edge = np.concatenate([np.arange(n_edge), np.flatnonzero(~loops)])
    inc_side = np.concatenate([np.zeros(n_edge, np.int64), np.ones(int((~loops).sum()), np.int64)])
    inc_end = np.where(inc_side == 0, edges.first[inc_edge], edges.second[inc_edge]).astype(np.int64)
    incident: list[list[tuple[int, int]]] = [[] for _ in range(n_end)]
    for e, g, s in zip(inc_end.tolist(), inc_edge.tolist(), inc_side.tolist()):
        incident[e].append((g, s))

    existence = np.zeros(n_end, dtype=np.int64)
    candidate_rows: dict[int, list[int]] = {}
    applied = {a.declaration for a in inputs.assumptions if a.item == "existence"}
    for row in np.flatnonzero(~known(decl.existence)).tolist():
        if row in applied:
            continue
        for e in (int(decl.first[row]), int(decl.second[row])):
            if e >= 0:
                reason = int(decl.existence_reason[row])
                existence[e] = reason if not existence[e] else min(existence[e], reason)
                candidate_rows.setdefault(e, []).append(row)

    # ---- per occurrence atom: definition facts
    atom_occ = np.repeat(np.arange(len(occ.definition)), np.diff(occ.atom_ptr))
    atom_local = np.arange(n_atom) - occ.atom_ptr[atom_occ]
    used_definitions = [definition for definition in defs if definition is not None]
    offset = np.zeros(len(defs) + 1, dtype=np.int64)
    for d in range(len(defs)):
        definition = defs[d]
        offset[d + 1] = offset[d] + (len(definition.atom_names) if definition is not None else 0)

    source = offset[occ.definition[atom_occ]] + atom_local  # each atom's row in the definitions' concatenation

    def gather(columns: list[NDArray], dtype: type) -> NDArray:
        values = np.concatenate(columns).astype(dtype) if columns else np.zeros(0, dtype)
        return values[source] if n_atom else np.zeros(0, dtype)

    element = gather([definition.element for definition in used_definitions], np.uint8)
    reference_charge = gather([definition.charge for definition in used_definitions], np.int8)
    reference_charge_state = gather([definition.charge_state for definition in used_definitions], np.uint8)
    leaving_flag = gather([definition.leaving_flag for definition in used_definitions], np.uint8)
    per_atom = [_per_atom(definition) for definition in used_definitions]
    hydrogen_count = gather([facts[0] for facts in per_atom], np.int64)
    valence_ref = gather([facts[1] for facts in per_atom], np.int64)
    valence_known = gather([facts[2] for facts in per_atom], bool)
    occurrence_known = known(occ.charge_state)
    end_atom = np.where(ends.atom >= 0, occ.atom_ptr[ends.occurrence] + ends.atom, -1)

    def atom_index(e: int) -> int:
        return int(end_atom[e])

    # ---- the endpoints that get a state
    has_new = np.zeros(n_end, dtype=bool)
    np.logical_or.at(has_new, inc_end, (new | symmetry_restated)[inc_edge])
    has_restated = np.zeros(n_end, dtype=bool)
    np.logical_or.at(has_restated, inc_end, restated[inc_edge])
    changed = occurrence_known & known(reference_charge_state) & (occ.charge != reference_charge)
    charge_site = np.append(changed, False)[end_atom]  # an endpoint with no atom (-1) reads the last, False
    restated_conflict = np.zeros(n_end, dtype=bool)  # a Conflict state whose candidates are undetermined
    np.logical_or.at(restated_conflict, inc_end, (restated & in_conflict)[inc_edge])
    # an endpoint outside the value (a built connection to an atom it does not hold) has no problem of its own
    inside = ends.occurrence >= 0
    problem_end = np.flatnonzero((has_new | (existence > 0) | charge_site | restated_conflict) & inside)

    # ---- one problem per endpoint (its fields in ``EndpointProblem`` order); solved once per distinct problem
    sites: dict[tuple[int, int], _Site] = {}
    assumed_by_row: dict[int, list[tuple[str, int]]] = {}
    for a in inputs.assumptions:
        assumed_by_row.setdefault(a.declaration, []).append((str(a.rule), a.value))
    # the columns the per-endpoint loop reads, as Python values
    end_occurrence, end_atom_id, end_existence = ends.occurrence.tolist(), ends.atom.tolist(), existence.tolist()
    end_charge = np.append(occ.charge, 0)[end_atom].tolist()  # an endpoint with no atom (-1) reads the default
    end_charge_state = np.append(occ.charge_state, _UNKNOWN)[end_atom].tolist()
    end_hosts = np.append(occ.observed_only == HydrogenHost.HOSTS, False)[end_atom].tolist()
    occ_definition, occ_unusable, occ_base = occ.definition.tolist(), occ.unusable.tolist(), occ.atom_ptr.tolist()
    atom_in_use, atom_observed = inputs.in_use.tolist(), occ.observed.tolist()
    edge_assumed = (basis(edges.order_state) == Basis.ASSUMED).tolist()
    edge_symmetry, edge_new, edge_loop = symmetry_restated.tolist(), new.tolist(), loops.tolist()
    edge_fields = list(
        zip(
            edges.kind.tolist(),
            edges.kind_state.tolist(),
            np.where(edges.order <= OrderToken.QUAD, edges.order, 0).tolist(),
            edges.order_state.tolist(),
            np.where(edges.order_state == _KNOWN_DECLARED, edges.order, 0).tolist(),
        )
    )
    edge_count, edge_retention = count.tolist(), retention.tolist()
    declaration_ptr, rows_list = edges.declaration_ptr.tolist(), rows.tolist()
    leaving_ptr = edges.declared_leaving_ptr.tolist()
    leaving_atom, leaving_side = edges.declared_leaving.tolist(), edges.declared_leaving_side.tolist()
    keys = []
    for e in problem_end.tolist():
        o, atom = end_occurrence[e], end_atom_id[e]
        d = occ_definition[o]
        definition = defs[d] if d >= 0 else None
        if occ_unusable[o]:
            incomplete = _CODE["reference_incomplete_unusable"]
        elif definition is None:
            incomplete = _CODE["reference_incomplete_not_in_library"]
        elif atom < 0 or not definition.atom_names:
            incomplete = _CODE["reference_incomplete_absent"]
        else:
            incomplete = 0
        edge_rows, declared_ids, symmetry_flag, assumed = [], set(), False, []
        for g, side in incident[e]:
            if edge_assumed[g]:  # an assumed order or an assumed existence
                for row in rows_list[declaration_ptr[g] : declaration_ptr[g + 1]]:
                    assumed += assumed_by_row.get(row, [])
            if edge_symmetry[g]:
                symmetry_flag = True
                continue
            if not edge_new[g]:
                continue
            edge_rows.append((*edge_fields[g], edge_count[g][side], edge_retention[g][side]))
            for k in range(leaving_ptr[g], leaving_ptr[g + 1]):
                if leaving_side[k] == side or edge_loop[g]:
                    declared_ids.add(leaving_atom[k])
        site = _NO_SITE
        if not incomplete:
            if (d, atom) not in sites:
                sites[(d, atom)] = _site(cast(ChemicalComponent, definition), atom)
            site = sites[(d, atom)]
        base = occ_base[o]
        position = site.position
        keys.append(
            (
                d if not incomplete else -1,
                atom if not incomplete else -1,
                tuple(sorted(edge_rows)),
                site.fragment_atoms,
                sum(1 << position[a] for a in declared_ids if a in position) if declared_ids else 0,
                tuple([atom_in_use[base + a] for a in site.fragment_atoms]),
                site.designated,
                end_charge[e],
                end_charge_state[e],
                end_hosts[e],
                end_existence[e],
                symmetry_flag,
                tuple(sorted(set(assumed))) if assumed else (),
                tuple(sorted(a for a in declared_ids if a not in position)) if declared_ids else (),
                incomplete,
                tuple(a for a in site.body if not atom_observed[base + a]),
                read,
            )
        )
    which, distinct = _rank(keys)
    problem_u = [EndpointProblem(*key) for key in distinct]  # equal keys are equal problems
    definition_u = [defs[p.definition] if p.definition >= 0 else None for p in problem_u]
    outcomes = [rule_l(p, c) for p, c in zip(problem_u, definition_u)]
    unit_end = np.full(n_end, -1, dtype=np.int64)  # per endpoint, its distinct problem (-1: none)
    unit_end[problem_end] = which
    unit_of = dict(zip(problem_end.tolist(), which.tolist()))
    outcome_of = {e: outcomes[u] for e, u in unit_of.items()}

    # ---- edge-referenced conflicts, restatement conflicts and the propagated assumption failure
    conflicted_edge = in_conflict & (new | restated)
    has_conflicted = np.zeros(n_end, dtype=bool)
    np.logical_or.at(has_conflicted, inc_end, conflicted_edge[inc_edge])
    for e in problem_end[has_conflicted[problem_end]].tolist():  # a conflicted new edge or restatement at E: the
        outcome = outcome_of[e]  # state refers to the edge's record
        conflicted = [g for g, _ in incident[e] if conflicted_edge[g]]
        if conflicted and (outcome.state != _CONFLICT_STATE or outcome.reason == 0):
            decided = outcome.state == _CONFLICT_STATE
            outcome_of[e] = EndpointOutcome(
                _CONFLICT_STATE,
                int(edge_code[conflicted[0]]),
                (),
                (),
                outcome.phantom,
                0,
                0,
                outcome.conflicts if decided else (),
            )
    falsified = [e for e in problem_end.tolist() if outcome_of[e].reason == _CODE["assumption_falsified"]]
    for e in falsified:
        for g, _ in incident[e]:
            if not edge_assumed[g]:
                continue
            for other in (int(edges.first[g]), int(edges.second[g])):
                current = outcome_of.get(other)
                if current is not None and current.state != _CONFLICT_STATE and current.reason != outcome_of[e].reason:
                    outcome_of[other] = EndpointOutcome(
                        _UNKNOWN_STATE,
                        _CODE["assumption_falsified"],
                        _declared_atoms(problem_u[unit_of[other]], definition_u[unit_of[other]]),
                        (),
                        current.phantom,
                        0,
                        int(Dependency.ASSUMPTION),
                        current.conflicts,
                        outcome_of[e].payload,
                    )

    # the other restatement-only endpoints are Inapplicable (a conflicted one is a problem, Conflict above)
    inapplicable = EndpointOutcome(_INAPPLICABLE, 0, (), (), (), 0, 0, ())
    for e in np.flatnonzero(has_restated).tolist():
        outcome_of.setdefault(e, inapplicable)

    # ---- the leaving table
    stated = sorted(outcome_of)
    table_rows = [outcome_of[e] for e in stated]
    stated_end = np.array(stated, dtype=np.int64)
    row_unit = unit_end[stated_end]  # -1: a restatement-only endpoint, no problem
    row_atom = end_atom[stated_end]
    row_base = occ.atom_ptr[ends.occurrence[stated_end]]
    row_state = np.array([o.state for o in table_rows], dtype=np.uint8)
    row_reason = np.array([o.reason for o in table_rows], dtype=np.uint16)
    row_conflicts = np.array([bool(o.conflicts) for o in table_rows], dtype=bool)
    edge_bits = np.zeros(n_end, dtype=np.int64)
    np.bitwise_or.at(edge_bits, inc_end, edge_dependency[inc_edge])
    junction_edge = ~new | ((edges.origin == BondOrigin.JUNCTION) & (edges.order_state == _KNOWN_DEFINED))
    junction_only = np.bincount(inc_end, minlength=n_end) > 0
    np.logical_and.at(junction_only, inc_end, junction_edge[inc_edge])
    declared_u = np.array([bool(p.declared or p.declared_outside) for p in problem_u] + [False], dtype=bool)
    standard = (
        junction_only[stated_end]
        & (row_state == _DERIVED_REF)
        & ~row_conflicts
        & ~declared_u[row_unit]  # -1 reads the appended False
    )
    atoms_ptr, atoms_values = _csr([o.atoms for o in table_rows])
    alternatives_ptr, alternatives_values = _csr([o.alternatives for o in table_rows])
    phantom_ptr, phantom_values = _csr([o.phantom for o in table_rows])
    leaving = LeavingTable(
        endpoint=stated_end.astype(np.int32),
        state=row_state,
        reason=row_reason,
        atoms_ptr=atoms_ptr,
        atoms=atoms_values,
        alternatives_ptr=alternatives_ptr,
        alternatives=alternatives_values,
        phantom_ptr=phantom_ptr,
        phantom=phantom_values,
        added_hydrogens=np.array([o.added_hydrogens for o in table_rows], dtype=np.int8),
        dependencies=(np.array([o.dependencies for o in table_rows], dtype=np.int64) | edge_bits[stated_end]).astype(
            np.uint32
        ),
        standard=standard,
    )

    # ---- atom fates
    fate = np.full(n_atom, AtomFate.RETAINED, dtype=np.uint8)
    derived = np.isin(row_state, (_DECLARED, _DERIVED_REF, _DERIVED_VAL))
    in_problem = row_unit >= 0
    unknown_rows = (row_state == _UNKNOWN_STATE) & in_problem
    unsupported_reason = _CODE["unsupported_coordination_substitution"]
    left_rows = derived | (unknown_rows & (row_reason != unsupported_reason))
    fate[_members(row_base, atoms_ptr, atoms_values, left_rows)] = AtomFate.LEFT
    undetermined = [_members(row_base, phantom_ptr, phantom_values, row_state == _INAPPLICABLE)]  # a metal's
    # unobserved designated ligands
    candidates: dict[int, tuple[int, ...]] = {}
    for k in np.flatnonzero(unknown_rows | ((row_state == _CONFLICT_STATE) & in_problem)).tolist():
        outcome, u = table_rows[k], int(row_unit[k])
        if outcome.state == _UNKNOWN_STATE and outcome.reason == unsupported_reason:
            atoms = _declared_atoms(problem_u[u], definition_u[u])
        else:
            if u not in candidates:
                candidates[u] = _candidate_atoms(problem_u[u], definition_u[u])
            atoms = candidates[u]
            if outcome.state == _UNKNOWN_STATE:
                atoms = tuple(sorted(set(atoms) - set(outcome.atoms)))
        undetermined.append(np.array([int(row_base[k]) + a for a in atoms], dtype=np.int64))
    fate[np.concatenate(undetermined)] = AtomFate.UNDETERMINED
    if read:
        endpoint_atoms = row_atom[row_atom >= 0]
        unplaced = endpoint_atoms[~occ.placed[endpoint_atoms] & (fate[endpoint_atoms] == AtomFate.RETAINED)]
        fate[unplaced] = AtomFate.UNOBSERVED_ENDPOINT
    unsupported = occ.unsupported[atom_occ] & (leaving_flag == _Y) & (fate == AtomFate.RETAINED)
    fate[unsupported] = AtomFate.UNDETERMINED

    # ---- records
    log: list[tuple] = []

    def sources(e: int) -> list[int]:
        found = set()
        for g, _ in incident[e]:
            for row in rows[edges.declaration_ptr[g] : edges.declaration_ptr[g + 1]].tolist():
                found.update(decl.ref[decl.ref_ptr[row] : decl.ref_ptr[row + 1]].tolist())
        for row in candidate_rows.get(e, ()):
            found.update(decl.ref[decl.ref_ptr[row] : decl.ref_ptr[row + 1]].tolist())
        return sorted(found)

    first_atom = end_atom[decl.first]
    second_atom = np.where(decl.second >= 0, end_atom[decl.second], -1)

    def bonding(i: int) -> list[int]:
        r"""The source rows of the declarations with an endpoint at occurrence atom ``i``."""
        found = np.flatnonzero((first_atom == i) | (second_atom == i)).tolist()
        return [r for row in found for r in decl.ref[decl.ref_ptr[row] : decl.ref_ptr[row + 1]].tolist()]

    def subject(e: int) -> tuple[SubjectKind, int]:
        i = atom_index(e)
        return (SubjectKind.ATOM, i) if i >= 0 else (SubjectKind.OCCURRENCE, int(occ.ordinal[ends.occurrence[e]]))

    # the rows that may record: the loop below appends nothing for any other
    phantom_recorded = (np.diff(phantom_ptr) > 0) & ((row_state == _INAPPLICABLE) | (in_problem & ~standard))
    unobserved_endpoint = np.append(fate == AtomFate.UNOBSERVED_ENDPOINT, False)[row_atom]
    recording = row_conflicts | (row_state == _UNKNOWN_STATE) | phantom_recorded | unobserved_endpoint
    for k in np.flatnonzero(recording).tolist():
        e, outcome, bulk, u = stated[k], table_rows[k], standard[k], int(row_unit[k])
        base, refs_e, (kind_, at) = int(row_base[k]), sources(e), subject(e)
        first_row = next((int(rows[edges.declaration_ptr[g]]) for g, _ in incident[e]), -1)
        basis_rows = []  # a stated charge at E is one of the incompatible statements of a blocked endpoint, and the
        # definition its ports and fragments derive from is the derivation basis
        if outcome.state == _UNKNOWN_STATE and outcome.reason == _CODE["no_balancing_fragment_blocked"]:
            i, d = atom_index(e), int(occ.definition[ends.occurrence[e]])
            if occ.charge_state[i] == _KNOWN_DECLARED and occ.atom_ref[i] >= 0:
                basis_rows = [int(occ.atom_ref[i])]
            ptr = inputs.definition_ref_ptr
            basis_rows += inputs.definition_ref[ptr[d] : ptr[d + 1]].tolist()
        for code, atoms in outcome.conflicts:
            name = CODE_NAMES[code]
            i = base + atoms[0] if atoms else -1
            extra = basis_rows + ([int(occ.atom_ref[i])] if i >= 0 else [])
            if name == "bonded_leaving_atom" and i >= 0:
                extra = basis_rows + bonding(i)
            if name == "observed_leaving_atom":
                row_number = int(inputs.refs.row[occ.atom_ref[i]]) if occ.atom_ref[i] >= 0 else -1
                values: tuple[int, ...] = (
                    i,
                    first_row,
                    row_number,
                    _carbonyl_observed(
                        problem_u[u] if u >= 0 else None, definition_u[u] if u >= 0 else None, occ, base
                    ),
                )
            elif name in ("placed_leaving_atom", "bonded_leaving_atom"):
                values = (i,)
            else:
                values = ()
            log.append((name, kind_, at, refs_e + extra, values))
        reason_name = CODE_NAMES[outcome.reason] if outcome.state == _UNKNOWN_STATE else ""
        # recorded once by their source: per occurrence, per ambiguous atom pair
        if reason_name and not reason_name.startswith("reference_incomplete") and reason_name != "ambiguous_connection":
            payload: tuple = outcome.payload or ((),)
            listed = tuple(base + a for a in payload[1]) if len(payload) > 1 else None
            row = (reason_name, kind_, at, refs_e, payload[0]) + ((listed,) if listed is not None else ())
            log.append(row)
        if outcome.state == _INAPPLICABLE and outcome.phantom:
            log.append(
                (
                    "coordination_substitution_unresolved",
                    kind_,
                    at,
                    refs_e,
                    (),
                    tuple(base + a for a in outcome.phantom),
                )
            )
        elif outcome.phantom and u >= 0 and not bulk:
            log.append(("phantom_heavy_neighbours", kind_, at, refs_e, (), tuple(base + a for a in outcome.phantom)))
        i = atom_index(e)
        if i >= 0 and fate[i] == AtomFate.UNOBSERVED_ENDPOINT and refs_e:
            log.append(("endpoint_unobserved", kind_, at, refs_e, ()))

    # ---- hydrogen plan: rows in priority order, an atom takes the first that matches and only an applying row
    # records
    heavy = (element != _H) & (element != 0)
    hydrogens = np.where(heavy, Hydrogens.MATERIALISE, Hydrogens.NONE).astype(np.uint8)
    hydrogen_reason = np.zeros(n_atom, dtype=np.uint16)
    unknown_h, observed_only_h = _CODE["hydrogen_count_unknown"], _CODE["hydrogen_count_observed_only"]
    decided = fate == AtomFate.UNDETERMINED  # row 1: none created, the endpoint's record covers it
    hydrogens[decided] = Hydrogens.NONE

    def observed_only(atoms: list[int] | NDArray, code: int) -> NDArray:
        r"""Apply a row to the undecided heavy atoms of ``atoms``; returns them."""
        atoms = np.asarray(atoms, dtype=np.int64)
        atoms = atoms[heavy[atoms] & ~decided[atoms]]
        hydrogens[atoms], hydrogen_reason[atoms], decided[atoms] = Hydrogens.OBSERVED_ONLY, code, True
        return atoms

    for o in np.flatnonzero(occ.unusable | (occ.definition < 0)).tolist():  # row 2
        observed_only(np.arange(occ.atom_ptr[o], occ.atom_ptr[o + 1]), unknown_h)
        log.append(
            ("hydrogen_count_unknown", SubjectKind.OCCURRENCE, int(occ.ordinal[o]), (), ("reference_unresolved",))
        )
    for i in observed_only(np.flatnonzero(occ.observed_only == HydrogenHost.HOSTS), observed_only_h).tolist():  # row 3
        log.append(("hydrogen_count_observed_only", SubjectKind.ATOM, i, (int(occ.atom_ref[i]),), ()))
    # row 3b: a candidate host of an observed hydrogen whose parent io could not decide (io records it, citing the H)
    observed_only(np.flatnonzero(occ.observed_only == HydrogenHost.CANDIDATE), unknown_h)
    for k in np.flatnonzero(np.isin(row_state, (_UNKNOWN_STATE, _CONFLICT_STATE)) & in_problem).tolist():  # row 4
        e, outcome, u = stated[k], table_rows[k], int(row_unit[k])
        definition = definition_u[u]
        if definition is None:
            continue
        base, p = int(row_base[k]), problem_u[u]
        # what may leave, and the declared and designated fragments in use (HXT on an observed OXT at a Conflict C)
        considered = [
            a for f in _fragments(p, definition) if f.may_leave or f.declared or f.designated for a in f.atoms
        ]
        affected = [base + a for a in (p.atom, *considered) if heavy[base + a] and not decided[base + a]]
        if any(hydrogen_count[i] for i in affected):
            observed_only(affected, unknown_h)
            cause = "endpoint_unknown" if outcome.state == _UNKNOWN_STATE else "endpoint_conflict"
            log.append(("hydrogen_count_unknown", SubjectKind.ATOM, base + p.atom, sources(e), (cause,)))
    coordination = np.zeros(n_end, dtype=bool)
    np.logical_or.at(coordination, inc_end, (new & (edges.kind == BondKind.COORDINATION))[inc_edge])  # row 5
    for k in np.flatnonzero(coordination[stated_end] & (row_state == _INAPPLICABLE) & (row_atom >= 0)).tolist():
        e, i = stated[k], int(row_atom[k])
        o = int(ends.occurrence[e])
        definition = defs[occ.definition[o]] if occ.definition[o] >= 0 and not occ.unusable[o] else None
        if (
            not decided[i]
            and hydrogen_count[i]
            and definition is not None
            and not sigma_lone_pair(definition)[ends.atom[e]]
        ):
            observed_only([i], unknown_h)
            log.append(("hydrogen_count_unknown", SubjectKind.ATOM, i, sources(e), ("coordination_incompatible",)))
            log.append(("reference_charge_unverified_at_coordination", SubjectKind.ATOM, i, sources(e), ()))
    unclassified_ends = {int(e) for g in np.flatnonzero(unclassified) for e in (edges.first[g], edges.second[g])}
    for e in sorted(unclassified_ends):  # row 6
        i = atom_index(e)
        if i >= 0 and not decided[i] and hydrogen_count[i] and heavy[i]:
            observed_only([i], unknown_h)
            log.append(("hydrogen_count_unknown", SubjectKind.ATOM, i, sources(e), ("unclassified_connection",)))

    # ---- implicit hydrogens, effective charge and allowed valence
    implicit_added = np.zeros(n_atom, dtype=np.int8)
    charge = np.where(occurrence_known, occ.charge, reference_charge).astype(np.int8)
    charge_state = np.where(occurrence_known, occ.charge_state, reference_charge_state).astype(np.uint8)
    valence = valence_ref.astype(np.int64)
    valence_state = np.where(valence_known, _KNOWN_REFERENCE, _UNKNOWN_REFERENCE).astype(np.uint8)
    shifts: dict[tuple[int, int, int], int | None] = {}  # _valence_shift reads the outcome's state and dependencies
    for k in np.flatnonzero((row_atom >= 0) & in_problem).tolist():
        outcome, u, i = table_rows[k], int(row_unit[k]), int(row_atom[k])
        definition = definition_u[u]
        if definition is None:
            continue
        implicit_added[i] = outcome.added_hydrogens
        key = (u, outcome.state, outcome.dependencies)
        if key not in shifts:
            shifts[key] = _valence_shift(problem_u[u], definition, outcome)
        shift = shifts[key]
        if shift is None:
            valence_state[i] = _UNKNOWN_DERIVED
        elif shift:
            valence[i] += shift
            valence_state[i] = _KNOWN_DERIVED
    valence[~known(valence_state)] = 0

    stereo = _stereo_plan(stated, table_rows, row_unit, problem_u, definition_u, incident, edges, new, ends, occ, fate)
    records = Records.concat([merge_records, Records.from_rows(log, inputs.refs)])
    return AttachmentResult(
        edges,
        leaving,
        fate,
        hydrogens,
        hydrogen_reason,
        implicit_added,
        charge,
        charge_state,
        valence.astype(np.int8),
        valence_state,
        stereo,
        records,
    )


def _declared_atoms(problem: EndpointProblem, definition: ChemicalComponent | None) -> tuple[int, ...]:
    if definition is None or problem.atom < 0:
        return ()
    return _atoms([f for f in _fragments(problem, definition) if f.declared])


def _candidate_atoms(problem: EndpointProblem, definition: ChemicalComponent | None) -> tuple[int, ...]:
    r"""The atoms of E's fragments that may leave: the candidates of an Unknown or Conflict endpoint's fates."""
    if definition is None or problem.atom < 0:
        return ()
    return _atoms([f for f in _fragments(problem, definition) if f.may_leave])


def _carbonyl_observed(problem: EndpointProblem | None, definition: ChemicalComponent | None, occ, base: int) -> int:
    r"""On a linked carbon, 1 when a doubly bonded oxygen of it is observed (a pentavalent carbon), else 0 (legacy
    carbonyl naming)."""
    if problem is None or definition is None or definition.element[problem.atom] != ELEMENTS.index("C"):
        return 0
    oxygen = ELEMENTS.index("O")
    return int(
        any(
            order == 2 and definition.element[other] == oxygen and occ.observed[base + other]
            for other, order, _ in _bonds(definition, problem.atom)
        )
    )


def _valence_shift(problem: EndpointProblem, definition: ChemicalComponent, outcome: EndpointOutcome) -> int | None:
    r"""The change of E's allowed covalent valence from its reference valence (``atom_valence``): the charge's
    shift (class A by ``valence_target``, the onium rule) and the open valence the new bonds filled; None when the
    outcome leaves it undetermined."""
    if outcome.state not in (_DECLARED, _DERIVED_REF, _DERIVED_VAL, _INAPPLICABLE):
        changed = known(problem.charge_state) and problem.charge != int(definition.charge[problem.atom])
        return None if changed else 0
    element, reference_charge = int(definition.element[problem.atom]), int(definition.charge[problem.atom])
    charge = problem.charge if known(problem.charge_state) else reference_charge
    target_ref, target = valence_target(element, reference_charge), valence_target(element, charge)
    v_ref = sum(order for _, order, _ in _bonds(definition, problem.atom))
    if charge != reference_charge:
        if element in _ONIUM:
            return 1
        return None if target is None or target_ref is None else target - target_ref + max(target_ref - v_ref, 0)
    if outcome.dependencies & Dependency.OPEN_VALENCE_REFERENCE and target_ref is not None:
        return target_ref - v_ref
    return 0


def _stereo_plan(
    stated, table_rows, row_unit, problem_u, definition_u, incident, edges, new, ends, occ, fate
) -> StereoPlan:
    r"""Per definition stereocentre whose neighbourhood changes, TRANSPORT at a port (a designation at E or a
    junction/cap port) with a determined outcome, one departed neighbour and one new partner, UNDETERMINED_NONPORT at a
    new covalent edge of a non-port centre, CLEAR otherwise."""
    centre_u = [  # per distinct problem with a definition, the first stereocentre row at its atom (-1: none)
        -1 if c is None else next(iter(np.flatnonzero(c.stereo_centre == p.atom).tolist()), -1)
        for p, c in zip(problem_u, definition_u)
    ]
    centre_row = np.array(centre_u + [-1], dtype=np.int64)[row_unit]  # a row without a problem (-1) reads -1
    rows: list[tuple[int, int, int, int, int, int, int]] = []
    for k in np.flatnonzero(centre_row >= 0).tolist():
        e, outcome, u = stated[k], table_rows[k], int(row_unit[k])
        definition, atom, o = definition_u[u], int(ends.atom[e]), int(ends.occurrence[e])
        base = int(occ.atom_ptr[o])
        slots = [int(n) for n in definition.stereo_neighbors[centre_u[u]].tolist() if n >= 0]
        departed = [n for n in slots if fate[base + n] == AtomFate.LEFT]
        partners = [
            (g, int(edges.second[g] if side == 0 else edges.first[g]))
            for g, side in incident[e]
            if new[g] and edges.kind[g] == BondKind.COVALENT
        ]
        # a port: E's reference designates a terminal fragment at E (D_E, activated by the new covalent bond), or a
        # junction/cap edge — whatever the outcome; an undetermined departure at a port is a non-unique map (CLEAR)
        port = bool(problem_u[u].designated) or any(
            edges.origin[g] in (BondOrigin.JUNCTION, BondOrigin.CAP) for g, _ in partners
        )
        if not partners and not departed:
            continue
        if partners and not port:
            rows.append((o, atom, StereoAction.UNDETERMINED_NONPORT, -1, -1, -1, _CODE["parity_cleared_nonport_link"]))
        elif len(partners) == 1 and len(departed) == 1 and outcome.state in (_DECLARED, _DERIVED_REF, _DERIVED_VAL):
            g, partner = partners[0]
            rows.append((o, atom, StereoAction.TRANSPORT, departed[0], partner, g, 0))
        elif not departed:
            rows.append(
                (o, atom, StereoAction.CLEAR, -1, -1, -1, _CODE["parity_cleared_new_neighbour_without_vacated_slot"])
            )
        else:
            rows.append(
                (o, atom, StereoAction.CLEAR, departed[0], -1, -1, _CODE["parity_cleared_heavy_neighbour_removed"])
            )
    columns = list(zip(*rows)) if rows else [()] * 7
    dtypes = (np.int32, np.int16, np.uint8, np.int32, np.int32, np.int32, np.uint16)
    return StereoPlan(*(np.array(column, dtype=dtype) for column, dtype in zip(columns, dtypes)))
