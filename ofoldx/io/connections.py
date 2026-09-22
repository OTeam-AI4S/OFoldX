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
Declared connections of a read → attachment inputs.

One owner per source semantics: ``struct_conn``/SSBOND/LINK rows (``connection_declarations``), branch links
(``branch_declarations``), PDB CONECT pairs (``conect_declarations``) and the per-pair junction rule
(``polymer_junctions``). ``attachment_inputs`` builds the occurrences of the chemical state, binds every endpoint once,
resolves transforms, stabilisers and physical edges through ``io.symmetry``, classifies restatements and evaluates a
consumer's assumption rules into rows. Every value io states here is a decoded, default or defined fact of the source;
nothing is chosen.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import TYPE_CHECKING, NamedTuple, cast

import numpy as np
from numpy.typing import NDArray

from ..chemistry.declarations import (
    Assumption,
    AssumptionRule,
    AssumptionSet,
    AttachmentInputs,
    DeclarationTable,
    EndpointTable,
    HydrogenHost,
    LinkKind,
    OccurrenceTable,
    OrderToken,
    Restatement,
    Strength,
    declaration_columns,
    declaration_table,
    endpoint_table,
    in_use,
    join_declarations,
    polymer_link_declarations,
)
from ..chemistry.definitions import ChemicalComponent, Definitions
from ..chemistry.fragments import (
    FAMILY_TABLE,
    METALLOIDS,
    METALS,
    POLYMER_FAMILIES,
    TERMINAL_CAPS,
    PortStatus,
    anomeric_carbons,
    cap_link,
    junction_ports,
    ports,
)
from ..chemistry.realize import OccurrenceLayout, atom_elements
from ..data.molecular import ELEMENTS, BondKind, BondOrigin, EntityType, NstdLinkage, Transforms
from ..errors import CODE_NAMES, StructureRef, fail
from ..provenance import Basis, Records, Resolution, SourceRefs, SubjectKind, known, state
from .binding import (
    NO_CODE,
    POLYMER_TYPES,
    BindingOutcome,
    ChemicalState,
    EndpointQuery,
    EndpointResolution,
    QuerySource,
    ResidueTable,
    author_key,
    definition_atoms,
    endpoint_queries,
    pdb_polymer_types,
    resolve_endpoints,
    stated_identity,
)
from .symmetry import EdgeGeometry, Symmetry, edge_geometry, resolve_transforms, stabilisers

if TYPE_CHECKING:
    from .tables import SourceTables

_KNOWN_DECLARED = state(Resolution.KNOWN, Basis.DECLARED)
_KNOWN_DEFINED = state(Resolution.KNOWN, Basis.DEFINED)
_KNOWN_DEFAULT = state(Resolution.KNOWN, Basis.DEFAULT)
_KNOWN_DERIVED = state(Resolution.KNOWN, Basis.DERIVED)
_UNKNOWN = state(Resolution.UNKNOWN, Basis.NONE)
_UNKNOWN_DECLARED = state(Resolution.UNKNOWN, Basis.DECLARED)
_UNKNOWN_DEFINED = state(Resolution.UNKNOWN, Basis.DEFINED)
_UNKNOWN_DERIVED = state(Resolution.UNKNOWN, Basis.DERIVED)
_INAPPLICABLE_DECLARED = state(Resolution.INAPPLICABLE, Basis.DECLARED)
_INAPPLICABLE_DEFINED = state(Resolution.INAPPLICABLE, Basis.DEFINED)
_CONFLICT = state(Resolution.CONFLICT, Basis.NONE)
_Z = {symbol.upper(): number for number, symbol in enumerate(ELEMENTS)}
_S, _SE, _H = _Z["S"], _Z["SE"], _Z["H"]
_COVALENT = frozenset({"covale", "covale_base", "covale_phosphate", "covale_sugar", "modres"})
_CONTACT = frozenset({"hydrog", "saltbr", "mismat"})
# the stated glycosylation roles and the element of their non-saccharide endpoint
_ROLES = {"N-Glycosylation": _Z["N"], "O-Glycosylation": _Z["O"], "S-Glycosylation": _S, "C-Mannosylation": _Z["C"]}
_TEMPLATES = ("N", "C", "CG", "CD", "P", "O3'")


# a non-resolved endpoint → the record of its row, in order of precedence between the two sides
_UNBOUND = {
    BindingOutcome.ATOM_NOT_IN_REFERENCE: "endpoint_atom_not_in_reference",
    BindingOutcome.UNRESOLVED_ASYM: "unresolved_asym",
    BindingOutcome.ATOM_UNBOUND: "endpoint_atom_unbound",
    BindingOutcome.UNSELECTED_ALTERNATIVE: "unselected_alternative",
    BindingOutcome.ABSENT_IN_MODEL: "absent_in_model",
}
_RAISED: dict[int, str] = {
    BindingOutcome.AMBIGUOUS: "endpoint_ambiguous",
    BindingOutcome.DANGLING: "endpoint_dangling",
    BindingOutcome.LABEL_AUTH_CONFLICT: "endpoint_label_auth_conflict",
}


@dataclass(frozen=True, eq=False)
class DeclarationDraft:
    r"""Declarations of one source before the symmetry results, over their own endpoint rows; ``codes`` and
    ``code_state`` [d, 2] are the stated symmetry codes per side (``io.symmetry`` resolves them)."""

    endpoints: EndpointTable
    table: DeclarationTable
    codes: NDArray[np.int32]
    code_state: NDArray[np.uint8]


def _symbols(tables: SourceTables, family: str) -> tuple[str, ...]:
    return tables.atoms.symbols.get(family, ())


def _rows_of(occurrences: OccurrenceTable, ordinals: NDArray | int) -> NDArray[np.int64]:
    r"""Ordinal → occurrence row (-1 outside the chemical state), sized for ``ordinals`` (or a count of them)."""
    size = ordinals if isinstance(ordinals, int) else int(np.max(ordinals, initial=-1)) + 1
    rows = np.full(max(size, int(occurrences.ordinal.max(initial=-1)) + 1), -1, dtype=np.int64)
    rows[occurrences.ordinal] = np.arange(len(occurrences.ordinal))
    return rows


def _pairs(query: EndpointQuery, sources: Sequence[int]) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    r"""The two query rows of every source row: ``endpoint_queries`` emits a row's endpoints consecutively."""
    picked = np.flatnonzero(np.isin(query.source, list(sources)))
    return picked[0::2], picked[1::2]


class _Rows:
    r"""Accumulates declaration rows of one draft: endpoints per side, columns, leaving ids, references."""

    def __init__(self) -> None:
        self.sides: list[tuple[tuple[int, int, int, int], tuple[int, int, int, int] | None]] = []
        self.columns: dict[str, list] = {}
        self.leaving: list[tuple[int, int, int]] = []  # (row, atom, side)
        self.refs: list[list[int]] = []
        self.codes: list[tuple[int, int]] = []
        self.code_state: list[tuple[int, int]] = []

    def add(self, first, second, refs, codes=(-1, -1), code_state=(_KNOWN_DEFINED, _KNOWN_DEFINED), **columns) -> int:
        row = len(self.sides)
        self.sides.append((first, second))
        for name, value in columns.items():
            self.columns.setdefault(name, [None] * row).append(value)
        for values in self.columns.values():
            if len(values) == row:
                values.append(None)
        self.refs.append([int(r) for r in refs if r >= 0])
        self.codes.append(tuple(codes))
        self.code_state.append(tuple(code_state))
        return row

    def draft(self) -> DeclarationDraft:
        d = len(self.sides)
        ends = [s for s, _ in self.sides] + [t for _, t in self.sides if t is not None]
        second = np.full(d, -1, dtype=np.int64)
        second[[k for k, (_, t) in enumerate(self.sides) if t is not None]] = np.arange(d, len(ends))
        columns = {}
        defaults = declaration_columns(declaration_table(np.zeros(d), np.zeros(d)))
        for name, values in self.columns.items():
            column = defaults[name].copy()
            for k, value in enumerate(values):
                if value is not None:
                    column[k] = value
            columns[name] = column
        leaving = sorted(self.leaving)
        columns["leaving_ptr"] = np.searchsorted([r for r, _, _ in leaving], np.arange(d + 1)).astype(np.int64)
        columns["leaving_atom"] = np.array([a for _, a, _ in leaving], dtype=np.int32)
        columns["leaving_side"] = np.array([s for _, _, s in leaving], dtype=np.uint8)
        columns["ref_ptr"] = np.r_[0, np.cumsum([len(r) for r in self.refs])].astype(np.int64)
        columns["ref"] = np.array([r for refs in self.refs for r in refs], dtype=np.int64)
        endpoints = endpoint_table(*(np.array([e[k] for e in ends], dtype=np.int32).reshape(-1) for k in range(4)))
        table = declaration_table(np.arange(d), second, **columns)
        return DeclarationDraft(
            endpoints,
            table,
            np.array(self.codes, dtype=np.int32).reshape(d, 2),
            np.array(self.code_state, dtype=np.uint8).reshape(d, 2),
        )


def _definition(occurrences: OccurrenceTable, definitions, occurrence: int) -> ChemicalComponent | None:
    d = int(occurrences.definition[occurrence])
    return definitions[d] if d >= 0 and not occurrences.unusable[occurrence] else None


def _unbound(
    tables: SourceTables, query: EndpointQuery, resolution: EndpointResolution, pair: tuple[int, int]
) -> str | None:
    r"""The record of a row with a non-resolved endpoint, or None when both resolved; raises the binding failures
    with the row (and an ambiguous endpoint's candidate residues)."""
    outcomes = [int(resolution.outcome[q]) for q in pair]
    for q, outcome in zip(pair, outcomes):
        if outcome in _RAISED:
            ref = tables.refs.ref(int(query.ref[q]))
            if outcome == BindingOutcome.AMBIGUOUS:
                span = resolution.candidates[resolution.candidates_ptr[q] : resolution.candidates_ptr[q + 1]]
                fail(_RAISED[outcome], ref, candidates=tuple(StructureRef(component=int(o)) for o in span))
            fail(_RAISED[outcome], ref)
    for found, mapped in _UNBOUND.items():
        if found in outcomes:
            return mapped
    return None


def glycosylation_order(
    roles: Sequence[str | None],
    components: Sequence[tuple[ChemicalComponent | None, ChemicalComponent | None]],
    atoms: NDArray[np.int32],
    elements: NDArray[np.uint8],
) -> tuple[NDArray[np.bool_], tuple[str | None, ...]]:
    r"""
    The glycosylation-role rule: per row, whether its stated ``pdbx_role`` defines order 1 — the
    non-saccharide endpoint has the role's element and the other is an anomeric carbon of a saccharide-class
    component — and otherwise the finding: ``role_environment_unknown`` when the saccharide side cannot be tested,
    ``role_atoms_incompatible`` when it fails (None for a row without a glycosylation role).
    """

    defined = np.zeros(len(roles), dtype=bool)
    findings: list[str | None] = []
    for k, role in enumerate(roles):
        if role not in _ROLES:
            findings.append(None)
            continue
        untestable = False
        for a, b in ((0, 1), (1, 0)):
            if elements[k, a] != _ROLES[role]:
                continue
            component, atom = components[k][b], int(atoms[k, b])
            if component is None or atom < 0 or not component.atom_names:
                untestable = True
            elif anomeric_carbons(component)[atom]:
                defined[k] = True
        findings.append(None if defined[k] else "role_environment_unknown" if untestable else "role_atoms_incompatible")
    return defined, tuple(findings)


def _cap_row(left: ChemicalComponent | None, right: ChemicalComponent | None, atoms: tuple[int, int]) -> bool:
    r"""A row naming exactly a terminal cap's atom and its partner's port, in either orientation."""
    if left is None or right is None:
        return False
    for (a, b), (x, y) in (((left, right), atoms), ((right, left), atoms[::-1])):
        found = cap_link(a, b)
        if found is None:
            continue
        cap = TERMINAL_CAPS.get(a.component_id)
        expected = found if cap is not None and cap.side == "next" else found[::-1]
        if (x, y) == expected:
            return True
    return False


def connection_declarations(
    tables: SourceTables,
    residues: ResidueTable,
    query: EndpointQuery,
    resolution: EndpointResolution,
    occurrences: OccurrenceTable,
    definitions: Sequence[ChemicalComponent | None],
    element: NDArray[np.uint8],
) -> tuple[DeclarationDraft, NDArray[np.int64], Records]:
    r"""
    ``struct_conn``, SSBOND and LINK rows → declarations, with the declaration row of each
    source row (-1 none). ``element`` is each query's endpoint element (0 when unbound or unknown). Two endpoints
    stating different altlocs at one bound residue position are mutually exclusive.

    ``covale*``/``modres``: covalent, the stated order declared, ``?``/``.`` kept, an absent item's verified default;
    a glycosylation role or a row naming exactly a terminal cap's atom and partner port defines order 1 (implied).
    ``disulf`` and SSBOND (which states no order), one rule on the bound endpoints' elements: covalent, order 1
    (intrinsic) between two S; otherwise no order from the type nor from an absent item's default (a stated one kept):
    S–Se or Se–Se is ``disulf_selenium_endpoint``, an endpoint of another known element ``Conflict(type_vs_element)``
    with the order Unknown (its ``order_declared_state`` CONFLICT, outside every assumption); an endpoint of unknown
    element proves neither. ``metalc``: coordination, a stated order kept for the merge's ``order_on_coordination``,
    direction from ``ptnr*_role``. LINK (UNKNOWN×DECLARED) and a ``?``, absent or ``.`` ``conn_type_id`` (the type
    field's own state; a stated order kept): kind undetermined, so no leaving budget counts the row until another basis
    fixes its kind. Contacts, rows with an unlisted type (the decoder records it) and rows with an unbound endpoint make
    no declaration.
    """

    c = tables.connections
    types, names, texts = (_symbols(tables, f) for f in ("conn_type", "atom", "text"))
    row_of = _rows_of(occurrences, resolution.ordinal)
    q0, q1 = _pairs(query, (QuerySource.STRUCT_CONN, QuerySource.SSBOND, QuerySource.LINK))
    by_ref = {int(query.ref[a]): (int(a), int(b)) for a, b in zip(q0, q1)}
    rows, log, declaration = _Rows(), [], np.full(len(c.ref), -1, dtype=np.int64)
    glycosylation: list[
        tuple[int, str, tuple[ChemicalComponent | None, ChemicalComponent | None], tuple[int, int], tuple[int, int]]
    ] = []
    for k in range(len(c.ref)):
        ref, source = int(c.ref[k]), int(c.origin[k])
        a, b = by_ref[ref]
        if source == QuerySource.LINK:
            kind_text = "link"
        elif source == QuerySource.SSBOND:
            kind_text = "disulf"
        elif known(c.conn_type_state[k]):
            kind_text = types[c.conn_type[k]].lower()
        elif c.conn_type_state[k] == _CONFLICT:  # an unlisted token: the decoder's conn_type_enumeration record
            continue
        else:  # `?` or absent: the row stays, its kind undetermined
            kind_text = ""
        if kind_text in _CONTACT:
            log.append(("contact_row", SubjectKind.EDGE, -1, [ref], ()))
            continue
        if kind_text not in _COVALENT | {"disulf", "metalc", "link", ""}:
            log.append(("conn_type_enumeration", SubjectKind.EDGE, -1, [ref], ()))
            continue
        site = [int(residues.position[o]) if o >= 0 else -1 for o in resolution.ordinal[[a, b]].tolist()]
        same_site = site[0] == site[1] >= 0  # the binding's position, not label fields a PDB source lacks
        if same_site and query.altloc[a] >= 0 and query.altloc[b] >= 0 and query.altloc[a] != query.altloc[b]:
            log.append(("altloc_exclusive_endpoints", SubjectKind.EDGE, -1, [ref], ()))  # no declaration
            continue
        unbound = _unbound(tables, query, resolution, (int(a), int(b)))
        if unbound is not None:
            log.append((unbound, SubjectKind.EDGE, -1, [ref], ()))
            continue
        occ = (int(row_of[resolution.ordinal[a]]), int(row_of[resolution.ordinal[b]]))
        atoms = (int(resolution.atom[a]), int(resolution.atom[b]))
        ends = tuple(
            (o, int(resolution.atom[q]), int(resolution.extra[q]), int(resolution.name[q])) for o, q in zip(occ, (a, b))
        )
        found = (_definition(occurrences, definitions, occ[0]), _definition(occurrences, definitions, occ[1]))
        columns: dict[str, object] = {"origin": BondOrigin.CONNECTION}
        covalent = kind_text not in ("metalc", "link", "")  # only a covalent type gives a default or defined order
        if kind_text == "link":
            columns.update(kind_state=_UNKNOWN_DECLARED)
        elif kind_text == "":  # the type field's own basis: `?` UNKNOWN×DECLARED, absent UNKNOWN×NONE, `.` inapplicable
            columns.update(kind_state=int(c.conn_type_state[k]))
        else:
            columns.update(kind=BondKind.COVALENT if covalent else BondKind.COORDINATION, kind_state=_KNOWN_DEFINED)
        if kind_text != "link":
            token, token_state = int(c.order[k]), int(c.order_state[k])
            if token_state == _KNOWN_DECLARED:
                columns.update(
                    order_token=token,
                    order_declared=token if token <= OrderToken.QUAD else 0,
                    order_declared_state=token_state,
                )
            elif token_state == _KNOWN_DEFAULT and covalent:
                columns.update(order_default=token)
            elif token_state == _KNOWN_DEFINED and covalent and source == QuerySource.STRUCT_CONN:
                # an _ofoldx_fact_basis row restored a defined order; the row does not say what defined it (a role, or
                # a disulfide stated after another type), so it gives no retention basis
                columns.update(order_defined=token, order_defined_strength=Strength.IMPLIED)
            elif token_state in (_UNKNOWN_DECLARED, _INAPPLICABLE_DECLARED):
                columns.update(order_declared_state=token_state)
        pair = (int(element[a]), int(element[b]))
        if kind_text == "disulf":  # struct_conn disulf and SSBOND alike, on the bound endpoints' elements
            if pair == (_S, _S):
                columns.update(order_defined=1, order_defined_strength=Strength.INTRINSIC)
            elif any(z not in (0, _S, _SE) for z in pair):  # an endpoint of another known element
                for name in ("order_token", "order_declared", "order_default"):  # the stated token stays in its row
                    columns.pop(name, None)
                columns["order_declared_state"] = _CONFLICT  # contradicted by its type: no order, never assumed
                log.append(("type_vs_element", SubjectKind.EDGE, len(rows.sides), [ref], ()))
            else:  # an absent item's default is no stated order: only two S give one, as to an SSBOND
                columns.pop("order_default", None)
                if 0 not in pair:  # a known Se endpoint
                    log.append(("disulf_selenium_endpoint", SubjectKind.EDGE, len(rows.sides), [ref], ()))
            # an unknown element (0) proves no incompatibility, nor the S–S order
        if kind_text == "metalc":
            if not any(z in METALS or z in METALLOIDS for z in pair):
                log.append(("kind_element_mismatch", SubjectKind.EDGE, len(rows.sides), [ref], ()))
            first_role, second_role = (
                texts[c.role[k, s]].casefold() if known(c.role_state[k, s]) else "" for s in (0, 1)
            )
            stated = (first_role, second_role)
            direction = {("donor", ""): 1, ("", "acceptor"): 1, ("donor", "acceptor"): 1}.get(stated)
            direction = direction or {("acceptor", ""): -1, ("", "donor"): -1, ("acceptor", "donor"): -1}.get(stated)
            if direction:
                columns.update(direction_declared=direction, direction_declared_state=_KNOWN_DECLARED)
        if kind_text in _COVALENT:
            if _cap_row(found[0], found[1], atoms):
                columns.update(order_defined=1, order_defined_strength=Strength.IMPLIED, retention_basis=(True, True))
            elif known(c.pdbx_role_state[k]):
                glycosylation.append((len(rows.sides), texts[c.pdbx_role[k]], found, atoms, pair))
        row = rows.add(
            ends[0],
            ends[1],
            [ref],
            codes=(int(c.symmetry[k, 0]), int(c.symmetry[k, 1])),
            # a code an OFoldX basis row marks derived is stated; resolving it derives it again
            code_state=tuple(_KNOWN_DECLARED if int(s) == _KNOWN_DERIVED else int(s) for s in c.symmetry_state[k]),
            **columns,
        )
        for side in (0, 1):
            if known(c.leaving_state[k, side]) and c.leaving[k, side] >= 0:
                definition = found[side]
                text = names[c.leaving[k, side]]
                atom = definition.atom_names.index(text) if definition and text in definition.atom_names else -1
                rows.leaving.append((row, atom, side))
        declaration[k] = row
    if glycosylation:
        defined, findings = glycosylation_order(
            [role for _, role, _, _, _ in glycosylation],
            [pair for _, _, pair, _, _ in glycosylation],
            np.array([atoms for _, _, _, atoms, _ in glycosylation], dtype=np.int32),
            np.array([elements for _, _, _, _, elements in glycosylation], dtype=np.uint8),
        )
        for (row, _, _, _, _), ok, finding in zip(glycosylation, defined, findings):
            if ok:
                for name, value in (
                    ("order_defined", 1),
                    ("order_defined_strength", Strength.IMPLIED),
                    ("retention_basis", (True, True)),
                ):
                    rows.columns.setdefault(name, [None] * len(rows.sides))[row] = value
            elif finding:
                log.append((finding, SubjectKind.EDGE, row, rows.refs[row], ()))
    return rows.draft(), declaration, Records.from_rows(log, tables.refs)


def branch_declarations(
    tables: SourceTables,
    query: EndpointQuery,
    resolution: EndpointResolution,
    occurrences: OccurrenceTable,
    definitions: Sequence[ChemicalComponent | None],
) -> tuple[DeclarationDraft, Records]:
    r"""
    ``_pdbx_entity_branch_link`` rows, replicated to every asym of the branched entity by ``endpoint_queries`` →
    covalent declarations (Defined) with the stated order and leaving ids; an explicit adjacency table, so each counts
    one physical neighbour at each endpoint (``topology_fixed``) and a declared leaving id establishes the linked
    product form at its own endpoint (retention basis).
    """

    b, names = tables.branch_links, _symbols(tables, "atom")
    row_of = _rows_of(occurrences, resolution.ordinal)
    position = {int(r): k for k, r in enumerate(b.ref.tolist())}
    rows, log = _Rows(), []
    for a, q in zip(*_pairs(query, (QuerySource.BRANCH_LINK,))):
        ref = int(query.ref[a])
        k = position[ref]
        unbound = _unbound(tables, query, resolution, (int(a), int(q)))
        if unbound is not None:
            log.append((unbound, SubjectKind.EDGE, -1, [ref], ()))
            continue
        occ = (int(row_of[resolution.ordinal[a]]), int(row_of[resolution.ordinal[q]]))
        ends = tuple(
            (o, int(resolution.atom[x]), int(resolution.extra[x]), int(resolution.name[x])) for o, x in zip(occ, (a, q))
        )
        stated = [int(b.leaving[k, side]) for side in (0, 1)]
        columns = {
            "origin": BondOrigin.BRANCH_LINK,
            "topology_fixed": True,
            "kind": BondKind.COVALENT,
            "kind_state": _KNOWN_DEFINED,
            "retention_basis": tuple(s >= 0 for s in stated),  # a side's own leaving id, never the other's
        }
        token, token_state = int(b.order[k]), int(b.order_state[k])
        if token_state == _KNOWN_DECLARED:
            columns.update(order_token=token, order_declared=token if token <= OrderToken.QUAD else 0)
        columns["order_declared_state"] = token_state if token_state != _KNOWN_DEFAULT else _UNKNOWN
        if token_state == _KNOWN_DEFAULT:
            columns["order_default"] = token
        row = rows.add(ends[0], ends[1], [ref], **columns)
        for side, code in enumerate(stated):
            if code >= 0:
                definition = _definition(occurrences, definitions, occ[side])
                text = names[code]
                atom = definition.atom_names.index(text) if definition and text in definition.atom_names else -1
                rows.leaving.append((row, atom, side))
    return rows.draft(), Records.from_rows(log, tables.refs)


def _has_bond(definition: ChemicalComponent | None, a: int, b: int) -> bool:
    if definition is None or a < 0 or b < 0:
        return False
    pairs = definition.bond_atoms
    return bool((((pairs[:, 0] == a) & (pairs[:, 1] == b)) | ((pairs[:, 0] == b) & (pairs[:, 1] == a))).any())


def conect_declarations(
    tables: SourceTables,
    query: EndpointQuery,
    resolution: EndpointResolution,
    occurrences: OccurrenceTable,
    specific: DeclarationDraft,
) -> tuple[DeclarationDraft, Records]:
    r"""
    PDB CONECT pairs: connectivity only, never a kind or an order. A reverse
    duplicate adds nothing; inside one residue a definition-less residue's pair or a pair its definition lacks is kept
    unclassified, a pair its definition has restates the reference bond; across residues a pair an SSBOND or LINK
    names merges into that row (taking its symmetry codes), anything else is ``conect_unclassified`` with no edge.
    MODRES is not atom-pair evidence.
    """

    conect = tables.conect
    row_of = _rows_of(occurrences, resolution.ordinal)
    position = {int(r): k for k, r in enumerate(conect.ref.tolist())}
    named: dict[frozenset[tuple[int, ...]], int] = {}  # endpoint pair of a specific SSBOND/LINK row → its row
    sp = specific.endpoints
    for row in range(len(specific.table.first)):
        f, s = int(specific.table.first[row]), int(specific.table.second[row])
        if s >= 0:
            key = frozenset(
                (tuple(int(x) for x in (sp.occurrence[f], sp.atom[f], sp.extra[f])),)
                + (tuple(int(x) for x in (sp.occurrence[s], sp.atom[s], sp.extra[s])),)
            )
            named.setdefault(key, row)
    rows, log, seen = _Rows(), [], set()
    for a, q in zip(*_pairs(query, (QuerySource.CONECT,))):
        ref = int(query.ref[a])
        k = position[ref]
        if bool(np.asarray(conect.contact[k]).any()):
            continue
        pair = frozenset(int(s) for s in conect.serial[k])
        if pair in seen:  # A→B and B→A: format redundancy
            continue
        seen.add(pair)
        unbound = _unbound(tables, query, resolution, (int(a), int(q)))
        if unbound is not None:
            log.append((unbound, SubjectKind.EDGE, -1, [ref], ()))
            continue
        occ = (int(row_of[resolution.ordinal[a]]), int(row_of[resolution.ordinal[q]]))
        ends = tuple(
            (o, int(resolution.atom[x]), int(resolution.extra[x]), int(resolution.name[x])) for o, x in zip(occ, (a, q))
        )
        if occ[0] == occ[1]:  # a reference bond's restatement or kept unclassified (classify_restatements)
            rows.add(ends[0], ends[1], [ref], connectivity_only=True)
            continue
        key = frozenset((ends[0][:3], ends[1][:3]))
        if key in named:
            target = named[key]
            rows.add(
                ends[0],
                ends[1],
                [ref],
                codes=tuple(specific.codes[target]),
                code_state=tuple(specific.code_state[target]),
                connectivity_only=True,
            )
            continue
        log.append(("conect_unclassified", SubjectKind.EDGE, -1, [ref], ()))
    return rows.draft(), Records.from_rows(log, tables.refs)


@dataclass(frozen=True, eq=False)
class PolymerLinks:
    r"""What ``polymer_junctions`` decides: the junction, cap and unlocated rows; the rows of ``specific`` that realise
    a consecutive pair (sequence links: topology-fixed); the unlinked pairs (occurrence rows); the occurrences
    with no port rule facing an unlinked pair (``unsupported_polymer_type``)."""

    draft: DeclarationDraft
    sequence_rows: NDArray[np.int64]
    unlinked_pairs: NDArray[np.int32]
    unsupported: NDArray[np.bool_]
    records: Records


def _entity_polymers(tables: SourceTables) -> dict[int, tuple[str | None, int, int, int]]:
    r"""mmCIF entity code → (``_entity_poly.type`` text or None, nstd_linkage, its state, the polymer row's ref)."""
    texts, p = _symbols(tables, "polymer_type"), tables.polymers
    found = {}
    for k, entity in enumerate(p.entity.tolist()):
        text = texts[p.type[k]] if known(p.type_state[k]) and p.type[k] >= 0 else None
        found[entity] = (text, int(p.nstd_linkage[k]), int(p.nstd_linkage_state[k]), int(p.ref[k]))
    return found


def _family(definition: ChemicalComponent | None):
    r"""The linking family a component's ``_chem_comp.type`` names (``FAMILY_TABLE``), or None."""
    if definition is None or not known(definition.chem_comp_type_state):
        return None
    rule = FAMILY_TABLE.get(cast(str, definition.chem_comp_type).strip().upper())
    return None if rule is None else rule[0]


def polymer_junctions(
    tables: SourceTables,
    residues: ResidueTable,
    state: ChemicalState,
    occurrences: OccurrenceTable,
    definitions: Sequence[ChemicalComponent | None],
    names: Mapping[str, int],
    specific: DeclarationDraft,
) -> PolymerLinks:
    r"""
    The per-pair junction rule over consecutive selected positions of each polymer instance: the positions of mmCIF
    entities with an ``_entity_poly`` row, or of PDB polymer chains declared by SEQRES (a source without a block is
    PDB, which has no ``PolymerTable`` rows).

    (1) The default link: JUNCTION when both components are in the applicable family (``POLYMER_FAMILIES`` of the
    entity's type; for PDB the pair's common family) with linkable facing ports; CAP by the terminal-cap rule. (2) A
    ``specific`` row joining exactly its atoms restates it; a row joining a facing port to a non-port atom of the
    partner withholds it (``junction_realised_otherwise``, under ``nstd_linkage no`` also
    ``nstd_linkage_contradicted``); a pair without a default link joined by a row is linked by that row. (3) An
    un-rowed default link exists under ``nstd_linkage no``, stated or derived (the ``no`` OFoldX's writer derives,
    with its basis, for a value whose every consecutive pair has its default link: a PDB source's; PDB:
    SEQRES adjacency with a unique correspondence); under ``yes``, ``?``, absent or ``.`` it is a candidate. (4) A
    pair with neither is unlocated (``consecutive_pair_unlinked``): one unlocated row from each facing port that
    exists, ``no_family`` when a component is outside the applicable family (or none applies), else ``no_port``.
    """

    pdb = tables.refs.block is None
    polymers = _entity_polymers(tables)
    if pdb:
        polymer = residues.kind == EntityType.POLYMER
    else:
        polymer = np.isin(residues.entity, list(polymers))
    row_of = _rows_of(occurrences, len(residues.entity))
    ordinals = np.flatnonzero(state.selected & residues.declared & polymer)
    ordinals = ordinals[np.lexsort((residues.position[ordinals], residues.instance[ordinals]))]
    same = residues.instance[ordinals[1:]] == residues.instance[ordinals[:-1]]
    left_ord, right_ord = ordinals[:-1][same], ordinals[1:][same]
    left, right = row_of[left_ord], row_of[right_ord]

    # rows of `specific` between two occurrences, by unordered occurrence pair
    sp, st = specific.endpoints, specific.table
    joined: dict[tuple[int, int], list[int]] = {}
    for row in np.flatnonzero((st.second >= 0) & known(st.existence)).tolist():
        o1, o2 = int(sp.occurrence[st.first[row]]), int(sp.occurrence[st.second[row]])
        if o1 != o2:
            joined.setdefault((min(o1, o2), max(o1, o2)), []).append(row)

    def end_at(row: int, occurrence: int) -> tuple[int, int]:
        r"""(definition atom, extra) of the row's endpoint in ``occurrence``."""
        e = int(st.first[row]) if sp.occurrence[st.first[row]] == occurrence else int(st.second[row])
        return int(sp.atom[e]), int(sp.extra[e])

    def default_link(dl: ChemicalComponent | None, dr: ChemicalComponent | None, family) -> tuple:
        r"""(link kind or None, (left atom, right atom)) of the pair's default link."""
        if dl is not None and dr is not None:
            if family is not None and (found := junction_ports(dl, dr, family)) is not None:
                return LinkKind.JUNCTION, found
            if (found := cap_link(dl, dr)) is not None:
                cap = TERMINAL_CAPS.get(dl.component_id)
                return LinkKind.CAP, found if cap is not None and cap.side == "next" else found[::-1]
        return None, (-1, -1)

    pairs, kinds, existence, sequence_rows, unlinked = [], [], [], [], []
    reasons: list[str | None] = []
    unsupported = np.zeros(len(occurrences.ordinal), dtype=bool)
    log: list[tuple] = []
    entity_findings = set()
    texts = _symbols(tables, "entity")
    usable = np.where(occurrences.unusable, -1, occurrences.definition).tolist()  # what ``_definition`` reads
    links: dict[tuple, tuple] = {}  # the default link is the two definitions' and the family's
    entity_of = residues.entity.tolist()
    for lo, ro, l_ord, r_ord in zip(left.tolist(), right.tolist(), left_ord.tolist(), right_ord.tolist()):
        il, ir = usable[lo], usable[ro]
        dl, dr = definitions[il] if il >= 0 else None, definitions[ir] if ir >= 0 else None
        if pdb:  # the SEQRES rows of the pair; the family is the pair's own
            nstd, nstd_state = 0, _INAPPLICABLE_DEFINED
            sources = [int(residues.declaration[l_ord]), int(residues.declaration[r_ord])]
            family = _family(dl) if _family(dl) == _family(dr) else None
            entity = entity_of[l_ord]
            label = texts[entity] if 0 <= entity < len(texts) else residues.instance_symbol[residues.instance[l_ord]]
        else:
            type_text, nstd, nstd_state, polymer_ref = polymers[entity_of[l_ord]]
            sources = [polymer_ref]
            family = POLYMER_FAMILIES.get(type_text) if type_text is not None else None
            label = texts[entity_of[l_ord]]
        if (key := (il, ir, family)) not in links:
            links[key] = default_link(dl, dr, family)
        link, atoms = links[key]
        rows_here = joined.get((min(lo, ro), max(lo, ro)), [])
        realised = [(row, end_at(row, lo), end_at(row, ro)) for row in rows_here]
        if link is None:
            if rows_here:
                sequence_rows += rows_here
                continue
            unlinked.append((lo, ro))
            payload = (label, int(residues.number[l_ord]), int(residues.number[r_ord]))
            log.append(("consecutive_pair_unlinked", SubjectKind.OCCURRENCE, l_ord, sources, payload))
            outside = family is None or _family(dl) != family or _family(dr) != family
            reason = "junction_unresolved_no_family" if outside else "junction_unresolved_no_port"
            for occurrence, definition, side, partner in ((lo, dl, 1, (lo, -1)), (ro, dr, 0, (-1, ro))):
                if definition is None:
                    continue
                status = ports(definition).status[side]
                if status in (PortStatus.VERIFIED, PortStatus.UNDEFINED):
                    pairs.append(partner)
                    kinds.append(LinkKind.UNLOCATED)
                    existence.append(_UNKNOWN_DEFINED)
                    reasons.append(reason)
                elif status in (PortStatus.NO_RULE, PortStatus.TYPE_UNKNOWN) and not unsupported[occurrence]:
                    unsupported[occurrence] = True
                    ordinal = int(occurrences.ordinal[occurrence])
                    log.append(("unsupported_polymer_type", SubjectKind.OCCURRENCE, ordinal, sources, ()))
            continue
        default = ((atoms[0], -1), (atoms[1], -1))  # an undefined side's port has no atom: never matched by name
        if any((x, y) == default for _, x, y in realised):  # (2)(i) restated: the link exists
            pairs.append((lo, ro))
            kinds.append(link)
            existence.append(_KNOWN_DEFINED)
            reasons.append(None)
            continue
        otherwise = [row for row, x, y in realised if (x == default[0]) != (y == default[1])]
        if otherwise:  # (2)(ii) withheld: the row realises the pair, never doubled
            sequence_rows += otherwise
            for row in otherwise:
                refs = specific.table.ref[specific.table.ref_ptr[row] : specific.table.ref_ptr[row + 1]].tolist()
                log.append(("junction_realised_otherwise", SubjectKind.EDGE, row, refs, ()))
                if nstd_state == _KNOWN_DECLARED and nstd == NstdLinkage.NO:
                    log.append(("nstd_linkage_contradicted", SubjectKind.EDGE, row, refs, ()))
            continue
        pairs.append((lo, ro))  # (3) determinacy of an un-rowed default link
        kinds.append(link)
        if pdb:
            ambiguous = not (known(residues.number_state[l_ord]) and known(residues.number_state[r_ord]))
            existence.append(_UNKNOWN_DEFINED if ambiguous else _KNOWN_DEFINED)
            reasons.append("junction_unresolved_sequence_ambiguous" if ambiguous else None)
        elif nstd_state in (_KNOWN_DECLARED, _KNOWN_DERIVED) and nstd == NstdLinkage.NO:
            existence.append(_KNOWN_DEFINED)
            reasons.append(None)
        else:
            stated_yes = nstd_state == _KNOWN_DECLARED and nstd == NstdLinkage.YES
            existence.append(_UNKNOWN_DEFINED)
            reasons.append(
                "junction_unresolved_nstd_linkage_yes" if stated_yes else "junction_unresolved_nstd_linkage_unstated"
            )
            finding = "nonstandard_linkage_declared" if stated_yes else "nstd_linkage_unstated"
            if (finding, sources[0]) not in entity_findings:
                entity_findings.add((finding, sources[0]))
                log.append((finding, SubjectKind.NONE, -1, sources, ()))

    code = {name: index for index, name in enumerate(CODE_NAMES)}
    endpoints, table = polymer_link_declarations(
        np.array(pairs, dtype=np.int32).reshape(-1, 2),
        np.array(kinds, dtype=np.uint8),
        np.array(existence, dtype=np.uint8),
        np.array([code[r] if r else 0 for r in reasons], dtype=np.uint16),
        occurrences,
        definitions,
        names,
    )
    d = len(table.first)
    draft = DeclarationDraft(
        endpoints, table, np.full((d, 2), -1, dtype=np.int32), np.full((d, 2), _KNOWN_DEFINED, dtype=np.uint8)
    )
    return PolymerLinks(
        draft,
        np.array(sorted(set(sequence_rows)), dtype=np.int64),
        np.array(unlinked, dtype=np.int32).reshape(-1, 2),
        unsupported,
        Records.from_rows(log, tables.refs),
    )


def classify_restatements(
    draft: DeclarationDraft,
    geometry: EdgeGeometry,
    transforms: Transforms,
    occurrences: OccurrenceTable,
    definitions: Sequence[ChemicalComponent | None],
) -> tuple[NDArray[np.uint8], NDArray[np.uint8], NDArray[np.uint8], NDArray[np.uint8]]:
    r"""
    Per declaration: an atom pair of one occurrence that is a reference bond, under the identity →
    RESTATEMENT (with the reference order, its token and the aromaticity for the merge); under an UNKNOWN transform →
    RESTATEMENT_SYMMETRY_UNKNOWN; one native atom to its own image → NEW_INTER (a self-link); a pair of one occurrence
    that is no reference bond, under the identity → NEW_INTRA; anything else → NEW_INTER.
    """

    t, e = draft.table, draft.endpoints
    d = len(t.first)
    restatement = np.full(d, Restatement.NEW_INTER, dtype=np.uint8)
    reference_order = np.zeros(d, dtype=np.uint8)
    reference_order_token = np.zeros(d, dtype=np.uint8)
    reference_aromatic = np.zeros(d, dtype=np.uint8)
    relative = geometry.transform
    identity = (transforms.rotation[relative] == np.eye(3, dtype=np.int8)).all(axis=(1, 2))
    identity &= (transforms.translation[relative] == 0).all(axis=1) & known(transforms.state[relative])
    unknown = transforms.state[relative] >> 3 == Resolution.UNKNOWN
    for row in np.flatnonzero(t.second >= 0).tolist():
        f, s = int(t.first[row]), int(t.second[row])
        if e.occurrence[f] != e.occurrence[s] or f == s:
            continue
        definition = _definition(occurrences, definitions, int(e.occurrence[f]))
        a, b = int(e.atom[f]), int(e.atom[s])
        if _has_bond(definition, a, b):
            definition = cast(ChemicalComponent, definition)
            pairs = definition.bond_atoms
            k = int(
                np.flatnonzero(((pairs[:, 0] == a) & (pairs[:, 1] == b)) | ((pairs[:, 0] == b) & (pairs[:, 1] == a)))[0]
            )
            if identity[row]:
                restatement[row] = Restatement.RESTATEMENT
                reference_order[row] = definition.bond_order[k] if known(definition.bond_order_state[k]) else 0
                reference_order_token[row] = definition.bond_order_token[k]
                stated = known(definition.bond_aromatic_state[k])
                reference_aromatic[row] = definition.bond_aromatic[k] + 1 if stated else 0  # 2 aromatic, 1 not
            elif unknown[row]:
                restatement[row] = Restatement.RESTATEMENT_SYMMETRY_UNKNOWN
        elif identity[row]:
            restatement[row] = Restatement.NEW_INTRA
    return restatement, reference_order, reference_order_token, reference_aromatic


class _Atoms(NamedTuple):
    r"""Per occurrence atom its AtomTable row (-1 unplaced) and its element (``chemistry.realize.atom_elements``, 0
    unknown); the extras (CSR over occurrences, AtomTable rows in row order); the read's atom-name family and its
    codes; (extra, occurrence atom) per candidate host of an extra hydrogen whose parent is undecided."""

    atom_row: NDArray[np.int64]
    element: NDArray[np.uint8]
    extra_ptr: NDArray[np.int64]
    extra_row: NDArray[np.int64]
    family: tuple[str, ...]
    names: dict[str, int]
    unattributed: tuple[tuple[int, int], ...]


def _with(occurrences: OccurrenceTable, **changes: NDArray) -> OccurrenceTable:
    return OccurrenceTable(**({f.name: getattr(occurrences, f.name) for f in fields(occurrences)} | changes))


def _ranges(lengths: NDArray[np.int64]) -> NDArray[np.int64]:
    return np.arange(int(lengths.sum())) - np.repeat(np.cumsum(lengths) - lengths, lengths)


def _occurrences(
    tables: SourceTables, residues: ResidueTable, state: ChemicalState, definitions: Definitions
) -> tuple[OccurrenceTable, tuple[ChemicalComponent, ...], _Atoms, list[tuple]]:
    r"""
    The occurrences of the chemical state: one row per selected ordinal in instance order with its
    definition, and per definition atom the placed/observed/charge facts of its selected row, bound by binding's one
    atom-name rule (``definition_atoms``: the isotope name ``D*`` as ``H*`` only where the definition lacks ``D*`` and
    no row of the realisation is ``H*``, ``isotope_name_mapped``). An observed atom
    without a definition atom is an extra of its occurrence, extras in AtomTable row order; an observed element that
    differs from the one the definition states makes the definition unusable for the occurrence
    (``element_differs_from_reference``; an unknown on either side differs from nothing), and so does a definition
    withheld by a same-scope disagreement; a stated mass number (D/T) that differs from the definition's
    stated one is such an element difference. Each occurrence atom's element is then decided once
    (``chemistry.realize.atom_elements``), the one its endpoints are classified by. An observed hydrogen with no
    reference counterpart, located, HOSTS on the nearest located heavy atom within 1.3 Å; with no such atom,
    every unlocated heavy atom of its occurrence, and every heavy atom when the hydrogen is unlocated, may host it
    (CANDIDATE, never a guess); located = coordinates stated. One ``reference_incomplete_*`` record per occurrence
    without a usable atomic definition.
    """

    atoms = tables.atoms
    comps, file_names = _symbols(tables, "comp"), _symbols(tables, "atom")
    ordinals = np.flatnonzero(state.selected)
    ordinals = ordinals[np.lexsort((ordinals, residues.position[ordinals], residues.instance[ordinals]))]
    ids = [comps[c] for c in residues.component[ordinals].tolist()]
    withheld = {r.detail.get("id") for r in definitions.records.where(code="definition_conflict")}
    used = sorted({i for i in ids if i in definitions.by_id})
    found = tuple(definitions.by_id[i] for i in used)
    index_of = {i: k for k, i in enumerate(used)}
    definition = np.array([index_of.get(i, -1) for i in ids], dtype=np.int32)
    unusable = np.array([i in withheld and i not in definitions.by_id for i in ids], dtype=bool)
    sizes = np.array([len(found[d].atom_names) if d >= 0 else 0 for d in definition.tolist()], dtype=np.int64)
    atom_ptr = np.r_[0, np.cumsum(sizes)].astype(np.int64)
    n = int(atom_ptr[-1])

    names = {name: k for k, name in enumerate(file_names)}  # the file's names keep their codes
    for component in found:
        for name in (*component.atom_names, *_TEMPLATES):
            names.setdefault(name, len(names))

    counts = np.diff(residues.row_ptr)[ordinals]
    rows = residues.rows[np.repeat(residues.row_ptr[:-1][ordinals], counts) + _ranges(counts)]
    owner = np.repeat(np.arange(len(ordinals)), counts)
    keep = state.row_selected[rows]
    rows, owner = rows[keep], owner[keep]
    atom_of_row, by_isotope = definition_atoms(residues, atoms, state, definitions)
    local = atom_of_row[rows]
    log: list[tuple] = [
        ("isotope_name_mapped", SubjectKind.OCCURRENCE, int(ordinals[owner[k]]), [], ())
        for k in np.flatnonzero(by_isotope[rows]).tolist()
    ]
    matched = local >= 0
    index = atom_ptr[owner[matched]] + local[matched]  # one row per atom: binding's one atom-name rule
    row_of_atom = rows[matched]
    placed = np.zeros(n, dtype=bool)
    placed[index] = True
    observed, zero_occupancy = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    unknown_occupancy = atoms.occupancy_state[row_of_atom] >> 3 == Resolution.UNKNOWN
    observed[index] = (atoms.occupancy[row_of_atom] > 0) | unknown_occupancy
    zero_occupancy[index] = known(atoms.occupancy_state[row_of_atom]) & (atoms.occupancy[row_of_atom] == 0)
    atom_row = np.full(n, -1, dtype=np.int64)
    atom_row[index] = row_of_atom
    charge = np.zeros(n, dtype=np.int8)
    charge_state = np.full(n, _UNKNOWN, dtype=np.uint8)
    charge[index], charge_state[index] = atoms.charge[row_of_atom], atoms.charge_state[row_of_atom]
    element = np.concatenate([found[d].element for d in definition.tolist() if d >= 0] or [np.zeros(0, np.uint8)])
    isotope = np.concatenate([found[d].isotope for d in definition.tolist() if d >= 0] or [np.zeros(0, np.uint16)])
    seen, mass, defined = atoms.element[row_of_atom], atoms.isotope[row_of_atom], isotope[index]
    stated = (seen > 0) & (element[index] > 0)  # an unknown on either side differs from nothing
    differs = np.zeros(n, dtype=bool)
    differs[index] = ((seen != element[index]) & stated) | ((mass != defined) & (mass > 0) & (defined > 0))
    for o in np.unique(np.searchsorted(atom_ptr, np.flatnonzero(differs), side="right") - 1).tolist():
        unusable[o] = True
        log.append(("element_differs_from_reference", SubjectKind.OCCURRENCE, int(ordinals[o]), [], ()))
    observed_element = np.zeros(n, dtype=np.uint8)
    observed_element[index] = seen
    decided, _ = atom_elements(element, observed_element, np.repeat(~unusable, sizes))

    order = np.lexsort((rows[~matched], owner[~matched]))
    extra_row, extra_owner = rows[~matched][order], owner[~matched][order]
    extra_ptr = np.searchsorted(extra_owner, np.arange(len(ordinals) + 1)).astype(np.int64)
    located = np.zeros(n, dtype=bool)
    located[index] = known(atoms.coordinate_state[row_of_atom])
    observed_only = np.zeros(n, dtype=np.uint8)
    unattributed: list[tuple[int, int]] = []
    for k in np.flatnonzero(atoms.element[extra_row] == _H).tolist():
        span = np.arange(atom_ptr[extra_owner[k]], atom_ptr[extra_owner[k] + 1])
        heavy = span[element[span] != _H]
        hosts = heavy
        if known(atoms.coordinate_state[extra_row[k]]):
            near = heavy[located[heavy]]
            distance = np.linalg.norm(atoms.coordinates[atom_row[near]] - atoms.coordinates[extra_row[k]], axis=1)
            if len(near) and distance.min() <= 1.3:
                observed_only[near[int(np.argmin(distance))]] = HydrogenHost.HOSTS
                continue
            hosts = heavy[~located[heavy]]
        unattributed += [(k, int(i)) for i in hosts]
    candidates = np.array([i for _, i in unattributed], dtype=np.int64)
    observed_only[candidates[observed_only[candidates] == HydrogenHost.NONE]] = HydrogenHost.CANDIDATE

    version = definitions.library.content_version if definitions.library is not None else ""
    for o, (d, component_id) in enumerate(zip(definition.tolist(), ids)):  # one record per occurrence
        ordinal = int(ordinals[o])
        if unusable[o]:
            log.append(("reference_incomplete_unusable", SubjectKind.OCCURRENCE, ordinal, [], ()))
        elif d < 0:
            payload = (component_id, version)
            log.append(("reference_incomplete_not_in_library", SubjectKind.OCCURRENCE, ordinal, [], payload))
        elif not found[d].atom_names:
            log.append(("reference_incomplete_absent", SubjectKind.OCCURRENCE, ordinal, [], ()))

    occurrences = OccurrenceTable(
        ordinal=ordinals.astype(np.int32),
        definition=definition,
        unusable=unusable,
        unsupported=np.zeros(len(ordinals), dtype=bool),
        atom_ptr=atom_ptr,
        placed=placed,
        observed=observed,
        zero_occupancy=zero_occupancy,
        atom_ref=np.full(n, -1, dtype=np.int64),
        charge=charge,
        charge_state=charge_state,
        observed_only=observed_only,
    )
    per_atom = _Atoms(atom_row, decided, extra_ptr, extra_row, tuple(names), names, tuple(unattributed))
    return occurrences, found, per_atom, log


def _atom_refs(tables: SourceTables, rows: NDArray[np.int64]) -> tuple[SourceRefs, NDArray[np.int64]]:
    r"""``tables.refs`` followed by one reference per given AtomTable row (its ``atom_site`` row, or the PDB line of
    its ATOM/HETATM record), and per given row the index of its reference (-1 for a row -1)."""
    present = np.flatnonzero(rows >= 0)
    picked = rows[present]
    categories: tuple[str, ...]
    if tables.refs.block is not None:
        category, categories = np.zeros(len(picked), dtype=np.int16), ("atom_site",)
    else:
        category, categories = (tables.atoms.group[picked] == 2).astype(np.int16), ("ATOM", "HETATM")
    extra = SourceRefs(
        category,
        tables.atoms.row[picked].astype(np.int64),
        np.full(len(picked), -1, dtype=np.int16),
        categories,
        (),
        tables.refs.block,
    )
    joined, (_, index) = SourceRefs.concat([tables.refs, extra])
    ref = np.full(len(rows), -1, dtype=np.int64)
    ref[present] = index
    return joined, ref


def _definition_refs(
    refs: SourceRefs, definitions: Definitions, found: tuple[ChemicalComponent, ...]
) -> tuple[SourceRefs, NDArray[np.int64], NDArray[np.int64]]:
    r"""``refs`` followed by the statements each definition of ``found`` was resolved from (``Definitions.refs``, in
    the read's block as its records are), and per definition (CSR) their rows."""
    listed = [definitions.refs.get(d.component_id, ()) for d in found]
    flat = [ref for refs_of in listed for ref in refs_of]
    categories = tuple(dict.fromkeys(ref.category for ref in flat))
    items = tuple(dict.fromkeys(ref.item for ref in flat if ref.item is not None))
    extra = SourceRefs(
        np.array([categories.index(ref.category) for ref in flat], dtype=np.int16),
        np.array([ref.row for ref in flat], dtype=np.int64),
        np.array([-1 if ref.item is None else items.index(ref.item) for ref in flat], dtype=np.int16),
        categories,
        items,
        refs.block,
    )
    joined, (_, index) = SourceRefs.concat([refs, extra])
    return joined, np.cumsum([0, *map(len, listed)]).astype(np.int64), index


def _assumptions(
    assumptions: AssumptionSet,
    table: DeclarationTable,
    element: NDArray[np.uint8],
    nstd_state: NDArray[np.uint8],
    link: NDArray[np.bool_],
) -> tuple[Assumption, ...]:
    r"""
    The consumer's rules evaluated over the declarations, each only where it fills an undetermined input item.
    ``element`` is per endpoint; ``nstd_state`` per declaration the ``nstd_linkage`` state of its
    entity (``?`` and absent are in ``unstated_junction_standard``'s scope, ``.`` is not); ``link`` per declaration
    whether a PDB LINK row states it, the only source ``link_metal_coordination`` reads (never a ``struct_conn`` row,
    whatever its type field states).
    """

    rules, out = assumptions.rules, []
    unstated = CODE_NAMES.index("junction_unresolved_nstd_linkage_unstated")
    second = np.maximum(table.second, 0)
    if AssumptionRule.UNSTATED_JUNCTION_STANDARD in rules:
        hit = ~known(table.existence) & (table.existence_reason == unstated) & (table.second >= 0)
        hit &= np.isin(nstd_state, [_UNKNOWN_DECLARED, _UNKNOWN])
        out += [
            Assumption(AssumptionRule.UNSTATED_JUNCTION_STANDARD, "existence", int(r), 1) for r in np.flatnonzero(hit)
        ]
    if AssumptionRule.LINK_METAL_COORDINATION in rules:
        z = np.stack([element[table.first], np.where(table.second >= 0, element[second], 0)], axis=1)
        metal = np.isin(z, list(METALS))
        nonmetal = (z > 0) & ~metal & ~np.isin(z, list(METALLOIDS))
        hit = link & (table.kind_state == _UNKNOWN_DECLARED) & (table.second >= 0)
        hit &= ~np.isin(table.restatement, [Restatement.RESTATEMENT, Restatement.RESTATEMENT_SYMMETRY_UNKNOWN])
        hit &= (metal.sum(axis=1) == 1) & (nonmetal.sum(axis=1) == 1)
        out += [
            Assumption(AssumptionRule.LINK_METAL_COORDINATION, "kind", int(r), int(BondKind.COORDINATION))
            for r in np.flatnonzero(hit)
        ]
    if AssumptionRule.UNSTATED_COVALENT_ORDER_SINGLE in rules:
        hit = (table.kind == BondKind.COVALENT) & known(table.kind_state)
        hit &= np.isin(table.origin, [BondOrigin.CONNECTION, BondOrigin.BRANCH_LINK])
        hit &= np.isin(table.order_declared_state, [_UNKNOWN_DECLARED, _UNKNOWN])
        hit &= (table.order_defined == 0) & (table.order_default == 0)
        out += [
            Assumption(AssumptionRule.UNSTATED_COVALENT_ORDER_SINGLE, "order", int(r), 1) for r in np.flatnonzero(hit)
        ]
    return tuple(out)


def _layout(
    tables: SourceTables,
    residues: ResidueTable,
    occurrences: OccurrenceTable,
    atoms: _Atoms,
    extra_ref: NDArray[np.int64],
    links: PolymerLinks,
    endpoints: EndpointTable,
    table: DeclarationTable,
    pdb_types: Sequence[NDArray] | None,
    offsets: NDArray[np.int64],
    declaration: NDArray[np.int64],
    query: EndpointQuery,
    resolution: EndpointResolution,
    log: list[tuple],
) -> OccurrenceLayout:
    r"""
    How the native occurrences form entities, instances and components. mmCIF: the ``_entity`` rows; an
    instance whose asym names no declared entity gets an entity of its own (type unknown). PDB: a ``COMPND`` molecule is
    the entity of its polymer chains (type POLYMER, derived from SEQRES), every other instance — a heterogen, a chain's
    waters, a chain without ``COMPND`` — is its own entity named by the instance (its kind derived from the format). An
    entity's polymer type codes the ``polymer_type`` symbol family (mmCIF) or ``binding.POLYMER_TYPES`` (PDB: the type
    its chains derive, UNKNOWN×DERIVED when they differ). An instance's connections are open when an existence-unknown
    link or an unlinked pair touches it, or a connection of it names an atom or asym nothing resolves. A
    modified-residue row naming no occurrence is appended to ``log``.
    """

    pdb, a = pdb_types is not None, tables.atoms
    entities, polymers = tables.entities, tables.polymers
    entity_texts = _symbols(tables, "entity")
    polymer_row = {int(e): k for k, e in enumerate(polymers.entity.tolist())}
    symbol = [entity_texts[e] for e in entities.id.tolist()]
    kind, kind_state = entities.type.astype(np.uint8).tolist(), entities.type_state.astype(np.uint8).tolist()
    # no _entity_poly row: the items do not apply to a known non-polymer (nor to any PDB entity, the format has no
    # nstd_linkage); for a polymer or an entity of unknown type they are unstated
    unrowed = [
        _INAPPLICABLE_DEFINED if pdb or (known(np.uint8(s)) and t != EntityType.POLYMER) else _UNKNOWN
        for t, s in zip(kind, kind_state)
    ]
    polymer_type = [int(polymers.type[polymer_row[e]]) if e in polymer_row else -1 for e in entities.id.tolist()]
    polymer_state = [
        int(polymers.type_state[polymer_row[e]]) if e in polymer_row else unrowed[k]
        for k, e in enumerate(entities.id.tolist())
    ]
    nstd = [int(polymers.nstd_linkage[polymer_row[e]]) if e in polymer_row else 0 for e in entities.id.tolist()]
    nstd_state = [
        int(polymers.nstd_linkage_state[polymer_row[e]]) if e in polymer_row else unrowed[k]
        for k, e in enumerate(entities.id.tolist())
    ]
    index = {int(e): k for k, e in enumerate(entities.id.tolist())}
    n_instances = len(residues.instance_symbol)
    first = np.full(n_instances, -1, dtype=np.int64)  # the first ordinal of each instance
    for ordinal in range(len(residues.instance) - 1, -1, -1):
        first[residues.instance[ordinal]] = ordinal

    def add(name: str, entity_kind: int, entity_state: int, type_code: int, type_state: int) -> int:
        while name in symbol:  # an entity of its own never takes a declared entity's id
            name += "'"
        symbol.append(name)
        kind.append(entity_kind)
        kind_state.append(entity_state)
        polymer_type.append(type_code)
        polymer_state.append(type_state)
        nstd.append(0)
        nstd_state.append(_INAPPLICABLE_DEFINED if pdb else _UNKNOWN)  # PDB: the format has no such item
        return len(symbol) - 1

    instance_entity = np.full(n_instances, -1, dtype=np.int32)
    derived: dict[int, set[tuple[int, int]]] = {}  # PDB: the polymer types a COMPND molecule's chains derive
    for i, name in enumerate(residues.instance_symbol):
        o, entity = int(first[i]), int(residues.instance_entity[i])  # binding's: _struct_asym's, else the rows'
        if pdb:
            instance_kind = int(residues.kind[o]) if o >= 0 else 0
            pdb_type_columns = cast(Sequence[NDArray], pdb_types)
            code, code_state = int(pdb_type_columns[0][i]), int(pdb_type_columns[1][i])
            if instance_kind == EntityType.POLYMER and entity in index:
                instance_entity[i] = index[entity]
                derived.setdefault(index[entity], set()).add((code, code_state))
            else:
                state_of = _KNOWN_DERIVED if instance_kind else _UNKNOWN
                instance_entity[i] = add(name, instance_kind, state_of, code, code_state)
        else:  # an mmCIF instance is its asym
            if entity not in index:
                text = entity_texts[entity] if entity >= 0 else name
                index[entity] = add(text, 0, _UNKNOWN, -1, _UNKNOWN)
            instance_entity[i] = index[entity]
            if entity < 0:  # no statement names its entity: an entity of its own
                del index[entity]
    for k, types in derived.items():
        kind[k], kind_state[k] = EntityType.POLYMER, _KNOWN_DERIVED
        polymer_type[k], polymer_state[k] = types.pop() if len(types) == 1 else (-1, _UNKNOWN_DERIVED)

    instance = residues.instance[occurrences.ordinal]
    open_ = np.zeros(n_instances, dtype=bool)
    undetermined = np.flatnonzero(~known(table.existence))
    touched = np.concatenate([table.first[undetermined], table.second[undetermined][table.second[undetermined] >= 0]])
    open_[instance[endpoints.occurrence[touched]]] = True
    open_[instance[links.unlinked_pairs.ravel()]] = True
    unresolved = (BindingOutcome.ATOM_UNBOUND, BindingOutcome.UNRESOLVED_ASYM, BindingOutcome.ABSENT_IN_MODEL)
    for q in np.flatnonzero(np.isin(resolution.outcome, unresolved)).tolist():
        for side in (q, q ^ 1):  # a row's two queries are consecutive
            if resolution.ordinal[side] >= 0:
                open_[residues.instance[resolution.ordinal[side]]] = True

    modified = tables.modified
    parents: list[list[tuple[int, int]]] = [[] for _ in occurrences.ordinal]
    # a row annotates the occurrence at its label address when it states one (its asym and label_seq_id, `.` for a
    # residue without a sequence position) with its component: the copies of one author chain differ only there;
    # else at its author address, the one the endpoint resolver matches (a blank chain the stated none, the number by
    # its author_key) with its author component; a part the occurrence or the row leaves unstated matches nothing, an
    # address naming several occurrences is ambiguous, one naming none leaves the annotation unrealised (recorded); a
    # label part the row states and the occurrence's differ in is label_auth_conflict
    r, o = residues, occurrences.ordinal
    number = np.where(known(r.number_state) & (r.kind != EntityType.BRANCHED), r.number, -1)
    labels: dict[tuple[int, ...], list[int]] = {}
    address: dict[tuple[int, ...], list[int]] = {}
    if (modified.parent >= 0).any():  # the addresses only a row with a parent looks up
        k = np.flatnonzero((r.asym >= 0)[o])
        for key, at in zip(zip(*(c[o[k]].tolist() for c in (r.asym, number, r.component))), k.tolist()):
            labels.setdefault(key, []).append(at)
        parts = (stated_identity(r.author_chain, r.author_chain_state), r.author_key, r.insertion, r.author_component)
        k = np.flatnonzero(((parts[0] != NO_CODE) & (r.author_key != NO_CODE) & (r.author_component >= 0))[o])
        for key, at in zip(zip(*(part[o[k]].tolist() for part in parts)), k.tolist()):
            address.setdefault(key, []).append(at)
    labelled = (modified.label_asym_state == _KNOWN_DECLARED) & np.isin(
        modified.label_seq_state, (_KNOWN_DECLARED, _INAPPLICABLE_DECLARED)
    )
    row_position = np.where(modified.label_seq_state == _KNOWN_DECLARED, modified.label_seq, -1)
    row_chain = stated_identity(modified.asym, modified.asym_state)
    row_number = author_key(modified.seq_text, modified.seq_state, modified.seq, pdb)
    for j in np.flatnonzero(modified.parent >= 0).tolist():  # a row whose parent is unstated annotates none
        ref = int(modified.ref[j])
        if labelled[j]:
            found = labels.get((int(modified.label_asym[j]), int(row_position[j]), int(modified.comp[j])), [])
        else:
            found = address.get(
                (int(row_chain[j]), int(row_number[j]), int(modified.ins[j]), int(modified.comp[j])), []
            )
        if len(found) > 1:
            candidates = tuple(StructureRef(component=k) for k in found)
            fail("modified_residue_ambiguous", tables.refs.ref(ref), candidates=candidates)
        if not found:
            log.append(("modified_residue_unbound", SubjectKind.NONE, -1, [ref], ()))
            continue
        ordinal = int(o[found[0]])
        seq_stated = known(modified.label_seq_state[j]) & known(r.number_state[ordinal])
        if (
            (known(modified.label_asym_state[j]) and modified.label_asym[j] != r.asym[ordinal])
            or 0 <= modified.label_comp[j] != r.component[ordinal]
            or (seq_stated and modified.label_seq[j] != r.number[ordinal])
        ):
            fail("label_auth_conflict", tables.refs.ref(ref))
        parents[found[0]].append((int(modified.parent[j]), ref))

    # each declaration's connection type: its struct_conn/SSBOND/LINK row's, CONECT for a CONECT pair
    connection_types = _symbols(tables, "conn_type")
    connection_types += ("CONECT",) * ("CONECT" not in connection_types)
    connection_type = np.full(len(table.first), -1, dtype=np.int32)
    c = tables.connections
    rows = np.flatnonzero(declaration >= 0)
    stated = known(c.conn_type_state[rows])
    connection_type[declaration[rows[stated]]] = c.conn_type[rows[stated]]
    connection_type[offsets[2] : offsets[3]] = connection_types.index("CONECT")

    extra = atoms.extra_row
    placed = atoms.atom_row >= 0
    observed_element = np.zeros(len(atoms.atom_row), dtype=np.uint8)
    observed_element[placed] = a.element[atoms.atom_row[placed]]
    observed_isotope = np.zeros(len(atoms.atom_row), dtype=np.uint16)
    observed_isotope[placed] = a.isotope[atoms.atom_row[placed]]
    return OccurrenceLayout(
        entity_symbol=tuple(symbol),
        entity_type=np.array(kind, dtype=np.uint8),
        entity_type_state=np.array(kind_state, dtype=np.uint8),
        entity_polymer_type=np.array(polymer_type, dtype=np.int32),
        entity_polymer_type_state=np.array(polymer_state, dtype=np.uint8),
        entity_nstd_linkage=np.array(nstd, dtype=np.uint8),
        entity_nstd_linkage_state=np.array(nstd_state, dtype=np.uint8),
        instance_symbol=residues.instance_symbol,
        instance_entity=instance_entity,
        instance_occurrence_ptr=np.searchsorted(instance, np.arange(n_instances + 1)).astype(np.int64),
        occurrence_component=residues.component[occurrences.ordinal].astype(np.int32),
        observed_element=observed_element,
        observed_isotope=observed_isotope,
        extra_ptr=atoms.extra_ptr,
        extra_name=residues.name_of_row[extra],
        extra_element=a.element[extra].astype(np.uint8),
        extra_isotope=a.isotope[extra].astype(np.uint16),
        extra_charge=a.charge[extra].astype(np.int8),
        extra_charge_state=a.charge_state[extra].astype(np.uint8),
        extra_ref=extra_ref,
        declared_parent_ptr=np.r_[0, np.cumsum([len(p) for p in parents])].astype(np.int64),
        declared_parent=np.array([p for ps in parents for p, _ in ps], dtype=np.int32),
        declared_parent_ref=np.array([r for ps in parents for _, r in ps], dtype=np.int64),
        declaration_connection_type=connection_type,
        unlinked_pairs=links.unlinked_pairs,
        connections_open=open_,
        symbols=MappingProxyType(
            {
                "atom": atoms.family,
                "component": _symbols(tables, "comp"),
                "polymer_type": POLYMER_TYPES if pdb else _symbols(tables, "polymer_type"),
                "parent": _symbols(tables, "comp"),
                "connection_type": connection_types,
            }
        ),
    )


def attachment_inputs(
    tables: SourceTables,
    residues: ResidueTable,
    state: ChemicalState,
    symmetry: Symmetry,
    definitions: Definitions,
    assumptions: AssumptionSet,
) -> tuple[AttachmentInputs, OccurrenceLayout, NDArray[np.int64], Records]:
    r"""
    The attachment inputs of one chemical state: the occurrences, every endpoint bound once, the
    declarations of every source with their symmetry results and restatement classes, the charge sites (an atom whose
    KNOWN occurrence charge differs from its reference charge) as endpoints, the in-use set and the consumer's
    assumption rules evaluated into rows (none by default). Rows naming one atom pair that are not provably one physical
    edge and not that edge under a KNOWN transform or a fixed topology make no edge: several readings leave
    their existence Unknown(``ambiguous_connection``), none Conflict(``conflicting_connection``). Also returns every
    record of the stage.
    """

    occurrences, found, atoms, log = _occurrences(tables, residues, state, definitions)
    query = endpoint_queries(tables, residues)
    resolution = resolve_endpoints(residues, tables.atoms, state, definitions, query)
    row_of = _rows_of(occurrences, resolution.ordinal)

    element = np.zeros(len(query.ref), dtype=np.uint8)  # each query's endpoint element (0 unbound or unknown)
    for q in np.flatnonzero(resolution.outcome == BindingOutcome.RESOLVED).tolist():
        if resolution.atom[q] >= 0:
            element[q] = atoms.element[occurrences.atom_ptr[row_of[resolution.ordinal[q]]] + resolution.atom[q]]
        elif resolution.extra[q] >= 0:
            element[q] = tables.atoms.element[resolution.extra[q]]

    connections, declaration, connection_records = connection_declarations(
        tables, residues, query, resolution, occurrences, found, element
    )
    branches, branch_records = branch_declarations(tables, query, resolution, occurrences, found)
    conect_draft, conect_records = conect_declarations(tables, query, resolution, occurrences, connections)
    pdb_types, polymer_records = None, Records.empty()
    if tables.refs.block is None:  # PDB: polymer types per instance, derived from SEQRES
        *pdb_types, polymer_records = pdb_polymer_types(tables, residues, definitions)
    links = polymer_junctions(tables, residues, state, occurrences, found, atoms.names, connections)
    occurrences = _with(occurrences, unsupported=links.unsupported)

    columns = declaration_columns(connections.table)
    columns["topology_fixed"] = columns["topology_fixed"].copy()
    columns["topology_fixed"][links.sequence_rows] = True  # the rows that realise a consecutive pair
    connections = DeclarationDraft(
        connections.endpoints, declaration_table(**columns), connections.codes, connections.code_state
    )
    drafts = [connections, branches, conect_draft, links.draft]
    offsets = np.cumsum([0, *(len(d.table.first) for d in drafts)])
    endpoints, table = join_declarations([(d.endpoints, d.table) for d in drafts])
    extra_index = np.full(len(tables.atoms.element), -1, dtype=np.int32)  # AtomTable row -> its occurrence's extra
    owner_start = np.repeat(atoms.extra_ptr[:-1], np.diff(atoms.extra_ptr))
    extra_index[atoms.extra_row] = np.arange(len(atoms.extra_row)) - owner_start
    extra = np.append(extra_index, -1)[endpoints.extra]  # an endpoint with no extra (-1) reads the last, -1
    endpoints = endpoint_table(endpoints.occurrence, endpoints.atom, extra, endpoints.name)  # extras occurrence-local
    codes = np.concatenate([d.codes for d in drafts])
    code_state = np.concatenate([d.code_state for d in drafts])

    # the charge sites are endpoints too
    reference_charge = np.zeros(len(occurrences.placed), dtype=np.int16)
    reference_known = np.zeros(len(occurrences.placed), dtype=bool)
    atom_occurrence = np.repeat(np.arange(len(occurrences.definition)), np.diff(occurrences.atom_ptr))
    atom_definition = occurrences.definition[atom_occurrence]
    defined = atom_definition >= 0
    if defined.any():  # an occurrence's atoms are its definition's: one table over the definitions in use
        used = np.unique(atom_definition[defined]).tolist()
        offset = np.zeros(used[-1] + 1, dtype=np.int64)
        offset[used] = np.cumsum([0, *(len(found[d].charge) for d in used[:-1])])
        local = np.arange(len(atom_occurrence)) - occurrences.atom_ptr[atom_occurrence]
        index = offset[atom_definition[defined]] + local[defined]
        reference_charge[defined] = np.concatenate([found[d].charge for d in used])[index]
        reference_known[defined] = known(np.concatenate([found[d].charge_state for d in used]))[index]
    sites = np.flatnonzero(known(occurrences.charge_state) & reference_known & (occurrences.charge != reference_charge))
    if len(sites):
        site_occurrence = np.searchsorted(occurrences.atom_ptr, sites, side="right") - 1
        site_atom = sites - occurrences.atom_ptr[site_occurrence]
        site_name = [
            atoms.names[found[occurrences.definition[o]].atom_names[a]]
            for o, a in zip(site_occurrence.tolist(), site_atom.tolist())
        ]
        charge_sites = endpoint_table(site_occurrence, site_atom, np.full(len(sites), -1), np.array(site_name))
        empty = declaration_table(np.zeros(0), np.zeros(0))
        endpoints, table = join_declarations([(endpoints, table), (charge_sites, empty)])

    refs, row_ref = _atom_refs(tables, np.concatenate([atoms.atom_row, atoms.extra_row]))
    refs, definition_ref_ptr, definition_ref = _definition_refs(refs, definitions, found)
    atom_ref, extra_ref = row_ref[: len(atoms.atom_row)], row_ref[len(atoms.atom_row) :]
    for k, i in atoms.unattributed:  # the hydrogen's row is the source; the hydrogen plan applies it
        log.append(("hydrogen_count_unknown", SubjectKind.ATOM, i, [int(extra_ref[k])], ("hydrogen_parent_unknown",)))
    occurrences = _with(occurrences, atom_ref=atom_ref)
    base = occurrences.atom_ptr[endpoints.occurrence]
    endpoint_row = np.full(len(endpoints.occurrence), -1, dtype=np.int64)
    at = endpoints.atom >= 0
    endpoint_row[at] = atoms.atom_row[base[at] + endpoints.atom[at]]
    ex = endpoints.extra >= 0
    endpoint_row[ex] = atoms.extra_row[atoms.extra_ptr[endpoints.occurrence[ex]] + endpoints.extra[ex]]
    endpoint_element = np.zeros(len(endpoints.occurrence), dtype=np.uint8)
    endpoint_element[at] = atoms.element[base[at] + endpoints.atom[at]]
    endpoint_element[ex] = tables.atoms.element[endpoint_row[ex]]

    # symmetry: relative transforms, stabilisers, physical edges and counts (io.symmetry); a row locates its endpoint
    # only where it states the coordinates (a `?` reads 0: no position, never the origin)
    located = endpoint_row >= 0
    located[located] = known(tables.atoms.coordinate_state[endpoint_row[located]])
    site_row = np.where(located, endpoint_row, -1)
    d = len(table.first)
    has_ref = np.diff(table.ref_ptr) > 0
    first_ref = np.full(d, -1, dtype=np.int64)
    first_ref[has_ref] = table.ref[table.ref_ptr[:-1][has_ref]]
    origin = np.zeros(d, dtype=np.uint8)
    origin[offsets[0] : offsets[1]] = tables.connections.origin[declaration >= 0]
    origin[offsets[1] : offsets[2]] = QuerySource.BRANCH_LINK
    origin[offsets[2] : offsets[3]] = QuerySource.CONECT
    second_row = np.where(table.second >= 0, site_row[np.maximum(table.second, 0)], -1)
    transforms = resolve_transforms(
        symmetry,
        codes,
        code_state,
        _symbols(tables, "symmetry"),
        origin,
        first_ref,
        site_row[table.first],
        second_row,
        tables.atoms.coordinates,
    )
    used = np.unique(np.concatenate([table.first, table.second[table.second >= 0]])).astype(np.int32)
    used_rows = site_row[used]
    sites_xyz = np.full((len(used), 3), np.nan)
    sites_xyz[used_rows >= 0] = tables.atoms.coordinates[used_rows[used_rows >= 0]]
    quantum = np.zeros(len(used))
    # the printed half-unit is stored as float32: widened by its representation error, at most half a float32 ulp,
    # so the stabiliser's bound never falls below the exact half-unit
    printed = np.asarray(tables.atoms.coordinate_quantum)[used_rows[used_rows >= 0]].astype(np.float64)
    quantum[used_rows >= 0] = (printed.max(axis=1) if printed.ndim > 1 else printed) * (1 + np.finfo(np.float32).eps)
    stabiliser = stabilisers(symmetry, used, sites_xyz, used_rows >= 0, quantum)
    # what each row states of its edge (its kind, order and direction, oriented (lo, hi)): a join resting on a
    # stabiliser member decided within a tolerance merges only rows that state alike
    direction = np.where(table.first > table.second, -table.direction_declared, table.direction_declared)
    stated: tuple[str, ...] = (
        "kind",
        "kind_state",
        "order_declared_state",
        "order_token",
        "order_defined",
        "order_defined_strength",
    )
    stated += ("direction_declared_state",)
    statement = np.column_stack([getattr(table, name) for name in stated] + [direction]).astype(np.int64)
    geometry = edge_geometry(
        symmetry,
        transforms,
        codes,
        code_state,
        stabiliser,
        table.first,
        table.second,
        table.kind,
        table.topology_fixed,
        statement,
    )
    restatement, reference_order, reference_order_token, reference_aromatic = classify_restatements(
        DeclarationDraft(endpoints, table, codes, code_state), geometry, transforms.transforms, occurrences, found
    )
    for row in np.flatnonzero(restatement == Restatement.NEW_INTRA).tolist():
        if _definition(occurrences, found, int(endpoints.occurrence[table.first[row]])) is not None:
            log.append(
                (
                    "edge_not_in_reference",
                    SubjectKind.EDGE,
                    row,
                    table.ref[table.ref_ptr[row] : table.ref_ptr[row + 1]].tolist(),
                    (),
                )
            )
    # of the rows of one atom pair without one reading, those one physical edge with a KNOWN transform or a
    # fixed topology are that edge in every reading; the others make no edge, existence Unknown (several readings) or
    # Conflict (none), one record per pair naming every row and both atoms; an ambiguous pair states whether a row of
    # it can be covalent (covalent or of undetermined kind): only then can its readings differ in the covalent graph
    # or a budget (coordination counts in none)
    resolved = transforms.transforms.state[geometry.transform]
    determined = np.zeros(d, dtype=bool)
    determined[geometry.physical_edge[known(resolved) | table.topology_fixed]] = True
    coordination = (table.kind == BondKind.COORDINATION) & known(table.kind_state)
    existence, reason = table.existence.copy(), table.existence_reason.copy()
    for reading, code, outcome in (
        (_UNKNOWN, "ambiguous_connection", _UNKNOWN_DERIVED),
        (_CONFLICT, "conflicting_connection", _CONFLICT),
    ):
        pairs: dict[frozenset[int], list[int]] = {}
        for row in np.flatnonzero(geometry.reading == reading).tolist():
            pairs.setdefault(frozenset((int(table.first[row]), int(table.second[row]))), []).append(row)
        for pair, members in pairs.items():
            sources = sorted({int(r) for m in members for r in table.ref[table.ref_ptr[m] : table.ref_ptr[m + 1]]})
            named = [int(base[e] + endpoints.atom[e]) if endpoints.atom[e] >= 0 else -1 for e in sorted(pair)]
            payload = (int(not coordination[members].all()),) if reading == _UNKNOWN else ()
            log.append((code, SubjectKind.EDGE, members[0], sources, payload, named))
        rows = (geometry.reading == reading) & ~determined[geometry.physical_edge]
        existence[rows], reason[rows] = outcome, CODE_NAMES.index(code)
    table = declaration_table(
        **(
            declaration_columns(table)
            | {
                "physical_edge": geometry.physical_edge,
                "count": geometry.count,
                "transform": geometry.transform,
                "self_bond": geometry.self_bond,
                "restatement": restatement,
                "reference_order": reference_order,
                "reference_order_token": reference_order_token,
                "reference_aromatic": reference_aromatic,
                "existence": existence,
                "existence_reason": reason,
            }
        )
    )

    polymers = _entity_polymers(tables)
    entity = residues.entity[occurrences.ordinal[endpoints.occurrence[table.first]]]
    nstd_state = np.array([polymers.get(int(e), (None, 0, _UNKNOWN))[2] for e in entity.tolist()], dtype=np.uint8)
    evaluated = _assumptions(assumptions, table, endpoint_element, nstd_state, origin == QuerySource.LINK)
    endpoint_atom = np.where(endpoints.atom >= 0, base + endpoints.atom, -1).astype(np.int64)
    declaration_records = [connection_records, branch_records, conect_records]
    source_records = Records.concat(
        [
            p.remap(SubjectKind.EDGE, np.arange(stop - start) + start)
            for p, start, stop in zip(declaration_records, offsets, offsets[1:])
        ]
    )
    layout = _layout(
        tables,
        residues,
        occurrences,
        atoms,
        extra_ref,
        links,
        endpoints,
        table,
        pdb_types,
        offsets,
        declaration,
        query,
        resolution,
        log,
    )
    records = Records.concat(
        [
            Records.from_rows(log, refs),
            source_records,
            links.records,
            polymer_records,
            transforms.records,
            stabiliser.records.remap(SubjectKind.ATOM, endpoint_atom),
        ]
    )
    inputs = AttachmentInputs(
        occurrences,
        found,
        endpoints,
        table,
        transforms.transforms,
        in_use(occurrences, endpoints, table),
        refs,
        evaluated,
        definition_ref_ptr,
        definition_ref,
    )
    return inputs, layout, declaration, records
