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
The attachment input tables and the invariants every producer shares.

A read (``io.connections``) and a recipe (``chemistry.construction``) both describe the declared connections of a
chemical state in the same columns: native occurrences, native endpoints, and one declaration row per source row,
defined junction, cap link or unlocated link, carrying the evidence of each edge field separately (declared, defined
with its strength, dictionary default, restated reference). The junction rows, the in-use set and the edge-field merge
are each built by one function here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from enum import IntEnum, IntFlag, StrEnum
from typing import ClassVar, Literal, cast

import numpy as np
from numpy.typing import NDArray

from ..data.molecular import BondKind, BondOrigin, Transforms
from ..errors import RequestRef, fail
from ..provenance import Basis, Records, Resolution, SourceRefs, SubjectKind, known, state
from .definitions import ChemicalComponent, OrderToken
from .fragments import TERMINAL_CAPS, PortStatus, cap_link, ports

_KNOWN_DECLARED = state(Resolution.KNOWN, Basis.DECLARED)
_KNOWN_DEFINED = state(Resolution.KNOWN, Basis.DEFINED)
_KNOWN_DEFAULT = state(Resolution.KNOWN, Basis.DEFAULT)
_KNOWN_REFERENCE = state(Resolution.KNOWN, Basis.REFERENCE)
_KNOWN_ASSUMED = state(Resolution.KNOWN, Basis.ASSUMED)
_UNKNOWN = state(Resolution.UNKNOWN, Basis.NONE)
_UNKNOWN_REFERENCE = state(Resolution.UNKNOWN, Basis.REFERENCE)  # a reference bond stating no integer order
_UNKNOWN_DECLARED = state(Resolution.UNKNOWN, Basis.DECLARED)
_UNKNOWN_DEFINED = state(Resolution.UNKNOWN, Basis.DEFINED)
_INAPPLICABLE_DECLARED = state(Resolution.INAPPLICABLE, Basis.DECLARED)
_INAPPLICABLE_DEFINED = state(Resolution.INAPPLICABLE, Basis.DEFINED)
_CONFLICT = state(Resolution.CONFLICT, Basis.NONE)


class Strength(IntEnum):
    r"""The strength of a defined value: the meaning of the declaring row's own token, or a value a definition
    implies absent a more specific statement."""

    NONE = 0
    INTRINSIC = 1
    IMPLIED = 2


class Restatement(IntEnum):
    r"""How a declaration relates to the reference of its occurrence; 0 until io classifies it."""

    UNSET = 0
    NEW_INTER = 1
    NEW_INTRA = 2
    RESTATEMENT = 3
    RESTATEMENT_SYMMETRY_UNKNOWN = 4


class LinkKind(IntEnum):
    JUNCTION = 1
    CAP = 2
    UNLOCATED = 3


class InUse(IntFlag):
    OBSERVED = 1
    PLACED = 2
    BONDED = 4


class HydrogenHost(IntEnum):
    r"""The observed-only marker of a definition atom: it hosts an observed hydrogen with no reference counterpart
    (HOSTS), or it may host one whose parent is undecided (CANDIDATE,
    ``hydrogen_count_unknown(hydrogen_parent_unknown)``)."""

    NONE = 0
    HOSTS = 1
    CANDIDATE = 2


class AssumptionRule(StrEnum):
    r"""Consumer assumption rules: never in a default read."""

    UNSTATED_JUNCTION_STANDARD = "unstated_junction_standard"
    LINK_METAL_COORDINATION = "link_metal_coordination"
    UNSTATED_COVALENT_ORDER_SINGLE = "unstated_covalent_order_single"


@dataclass(frozen=True)
class AssumptionSet:
    rules: frozenset[AssumptionRule] = frozenset()
    EMPTY: ClassVar[AssumptionSet]
    VERSION: ClassVar[str] = "assumptions/1"


AssumptionSet.EMPTY = AssumptionSet()


@dataclass(frozen=True)
class Assumption:
    r"""One evaluated rule application: io evaluates the rules, chemistry consumes the rows."""

    rule: AssumptionRule
    item: Literal["existence", "order", "kind"]
    declaration: int
    value: int


@dataclass(frozen=True, eq=False)
class OccurrenceTable:
    r"""
    One row per native occurrence of the chemical state, in instance order (instance, position); ``ordinal`` is its row
    of the read's residue table. ``definition`` indexes ``AttachmentInputs.definitions`` (-1: no usable definition);
    per-atom columns are over the occurrence's definition atoms (CSR ``atom_ptr``): ``placed`` = any row, ``observed`` =
    occupancy > 0 or ``?``, ``zero_occupancy`` = a row stating occupancy exactly 0, ``atom_ref`` the ``atom_site`` row
    in ``AttachmentInputs.refs`` (-1 unplaced), the occurrence charge fact (an absent ``pdbx_formal_charge`` reads its
    schema default 0 over the reference charge), ``observed_only`` the observed-only marker (a ``HydrogenHost``).
    ``unusable``: an observed element differs from the definition's. ``unsupported``: a polymer component with no port
    rule facing an unlinked pair (its Y-flagged atoms are undetermined).
    """

    ordinal: NDArray[np.int32]
    definition: NDArray[np.int32]
    unusable: NDArray[np.bool_]
    unsupported: NDArray[np.bool_]
    atom_ptr: NDArray[np.int64]
    placed: NDArray[np.bool_]
    observed: NDArray[np.bool_]
    zero_occupancy: NDArray[np.bool_]
    atom_ref: NDArray[np.int64]
    charge: NDArray[np.int8]
    charge_state: NDArray[np.uint8]
    observed_only: NDArray[np.uint8]


@dataclass(frozen=True, eq=False)
class EndpointTable:
    r"""
    Native endpoints: (occurrence, definition atom) or (occurrence, extra observed atom); ``name`` is the
    atom-name code in the read's name family, always set. ``atom = extra = -1`` is a port of a definition-less
    component (``PortStatus.UNDEFINED``; ``name`` the template id): never bound to an observed atom by that name.
    ``occurrence = -1`` is an atom outside the value, named by ``outside_instance`` (a code of the layout's instance
    family), ``outside_component`` (its ordinal there) and ``name``: a built value's connection to an atom it does not
    hold (-1 for every endpoint of the value).
    """

    occurrence: NDArray[np.int32]
    atom: NDArray[np.int32]
    extra: NDArray[np.int32]
    name: NDArray[np.int32]
    outside_instance: NDArray[np.int32]
    outside_component: NDArray[np.int32]


@dataclass(frozen=True, eq=False)
class DeclarationTable:
    r"""
    One row per source row, defined junction, cap link or unlocated link.
    ``first``/``second`` are endpoint rows (-1: no endpoint on that side). The symmetry results (``physical_edge``,
    ``count`` per side, -1 undeterminable; ``transform``; ``self_bond``) and ``restatement`` come from io. Per edge
    field the evidence is kept apart: ``order_declared`` is the numeric order of ``order_token`` (0 for AROM–POLY),
    ``order_defined`` with its ``Strength``, ``order_default`` (an absent item with a verified
    dictionary default only), the restated reference fields (``reference_order`` 0 unless known,
    ``reference_order_token`` the reference bond's own token; ``reference_aromatic`` 0 unknown, 1 not aromatic,
    2 aromatic).
    ``direction_declared`` +1 when ``first`` is the donor.
    ``leaving_atom`` (CSR ``leaving_ptr``) are definition atoms of the side's occurrence (-1: not in its reference),
    ``leaving_side`` 0 for ``first``; ``ref`` (CSR ``ref_ptr``) rows of ``AttachmentInputs.refs``.
    ``connectivity_only`` marks a row stating only that two atoms connect (a CONECT pair): with no kind from
    another row or a reference bond it is an unclassified connection, whereas a row whose type alone is unstated
    (absent ``conn_type_id``) still declares its connection and endpoints.
    """

    first: NDArray[np.int32]
    second: NDArray[np.int32]
    physical_edge: NDArray[np.int32]
    count: NDArray[np.int8]
    topology_fixed: NDArray[np.bool_]
    connectivity_only: NDArray[np.bool_]
    transform: NDArray[np.int32]
    self_bond: NDArray[np.bool_]
    restatement: NDArray[np.uint8]
    existence: NDArray[np.uint8]
    existence_reason: NDArray[np.uint16]
    origin: NDArray[np.uint8]
    kind: NDArray[np.uint8]
    kind_state: NDArray[np.uint8]
    order_declared: NDArray[np.uint8]
    order_declared_state: NDArray[np.uint8]
    order_token: NDArray[np.uint8]
    order_defined: NDArray[np.uint8]
    order_defined_strength: NDArray[np.uint8]
    order_default: NDArray[np.uint8]
    direction_declared: NDArray[np.int8]
    direction_declared_state: NDArray[np.uint8]
    reference_order: NDArray[np.uint8]
    reference_order_token: NDArray[np.uint8]
    reference_aromatic: NDArray[np.uint8]
    retention_basis: NDArray[np.bool_]
    leaving_ptr: NDArray[np.int64]
    leaving_atom: NDArray[np.int32]
    leaving_side: NDArray[np.uint8]
    ref_ptr: NDArray[np.int64]
    ref: NDArray[np.int64]


@dataclass(frozen=True, eq=False)
class AttachmentInputs:
    r"""What one chemical state's attachment chemistry reads; ``definition_ref`` (CSR ``definition_ref_ptr``) per
    definition the rows of ``refs`` naming the statements it was resolved from (``Definitions.refs``)."""

    occurrences: OccurrenceTable
    definitions: tuple[ChemicalComponent | None, ...]
    endpoints: EndpointTable
    declarations: DeclarationTable
    transforms: Transforms
    in_use: NDArray[np.uint8]  # [occurrence atoms] InUse bits, from in_use() only
    refs: SourceRefs
    assumptions: tuple[Assumption, ...]
    definition_ref_ptr: NDArray[np.int64]
    definition_ref: NDArray[np.int64]


@dataclass(frozen=True, eq=False)
class MergedEdges:
    r"""
    One row per physical edge (the only merge site). ``order`` holds the ``OrderToken`` of a declared value
    (SING–QUAD equal the order) or the order itself; ``edge_class`` is a ``Restatement``; ``declaration`` (CSR) the
    contributing rows; ``declared_leaving`` (CSR) the union of declared leaving ids, side 0 = ``first``.
    """

    first: NDArray[np.int32]
    second: NDArray[np.int32]
    transform: NDArray[np.int32]
    origin: NDArray[np.uint8]
    order: NDArray[np.uint8]
    order_state: NDArray[np.uint8]
    aromatic: NDArray[np.bool_]
    aromatic_state: NDArray[np.uint8]
    kind: NDArray[np.uint8]
    kind_state: NDArray[np.uint8]
    direction: NDArray[np.int8]
    direction_state: NDArray[np.uint8]
    edge_class: NDArray[np.uint8]
    declaration_ptr: NDArray[np.int64]
    declaration: NDArray[np.int64]
    declared_leaving_ptr: NDArray[np.int64]
    declared_leaving: NDArray[np.int32]
    declared_leaving_side: NDArray[np.uint8]


def endpoint_table(
    occurrence: NDArray[np.int32],
    atom: NDArray[np.int32],
    extra: NDArray[np.int32],
    name: NDArray[np.int32],
    outside_instance: NDArray[np.int32] | None = None,
    outside_component: NDArray[np.int32] | None = None,
) -> EndpointTable:
    r"""The endpoint rows, one per given position (the outside columns -1 unless given)."""
    absent = np.full(len(occurrence), -1)
    instance = absent if outside_instance is None else outside_instance
    component = absent if outside_component is None else outside_component
    columns = (occurrence, atom, extra, name, instance, component)
    return EndpointTable(*(np.asarray(column, dtype=np.int32) for column in columns))


def _defaults(d: int) -> dict[str, NDArray]:
    r"""The canonical zero of every declaration column but the endpoints: nothing stated, nothing resolved."""

    def full(value: int, dtype: type, *shape: int) -> NDArray:
        return np.full((d, *shape), value, dtype=dtype)

    return {
        "physical_edge": full(-1, np.int32),
        "count": full(-1, np.int8, 2),
        "topology_fixed": full(False, np.bool_),
        "connectivity_only": full(False, np.bool_),
        "transform": full(-1, np.int32),
        "self_bond": full(False, np.bool_),
        "restatement": full(Restatement.UNSET, np.uint8),
        "existence": full(_KNOWN_DEFINED, np.uint8),
        "existence_reason": full(0, np.uint16),
        "origin": full(BondOrigin.CONNECTION, np.uint8),
        "kind": full(0, np.uint8),
        "kind_state": full(_UNKNOWN, np.uint8),
        "order_declared": full(0, np.uint8),
        "order_declared_state": full(_UNKNOWN, np.uint8),
        "order_token": full(OrderToken.NONE, np.uint8),
        "order_defined": full(0, np.uint8),
        "order_defined_strength": full(Strength.NONE, np.uint8),
        "order_default": full(0, np.uint8),
        "direction_declared": full(0, np.int8),
        "direction_declared_state": full(_UNKNOWN, np.uint8),
        "reference_order": full(0, np.uint8),
        "reference_order_token": full(OrderToken.NONE, np.uint8),
        "reference_aromatic": full(0, np.uint8),
        "retention_basis": full(False, np.bool_, 2),
        "leaving_ptr": np.zeros(d + 1, dtype=np.int64),
        "leaving_atom": np.zeros(0, dtype=np.int32),
        "leaving_side": np.zeros(0, dtype=np.uint8),
        "ref_ptr": np.zeros(d + 1, dtype=np.int64),
        "ref": np.zeros(0, dtype=np.int64),
    }


def declaration_table(first: NDArray[np.int32], second: NDArray[np.int32], **columns: NDArray) -> DeclarationTable:
    r"""Declaration rows from the given columns; every column not given holds its canonical zero (``_defaults``)."""

    defaults = _defaults(len(first))
    values = {name: np.asarray(columns.get(name, default), dtype=default.dtype) for name, default in defaults.items()}
    unknown = set(columns) - set(defaults)
    assert not unknown, f"not declaration columns: {sorted(unknown)}"
    return DeclarationTable(np.asarray(first, dtype=np.int32), np.asarray(second, dtype=np.int32), **values)


def declaration_columns(table: DeclarationTable) -> dict[str, NDArray]:
    r"""Every column of ``table`` by name: ``declaration_table(**declaration_columns(t) | changes)`` rebuilds it."""
    return {f.name: getattr(table, f.name) for f in fields(table)}


_CSR = {"leaving_ptr": ("leaving_atom", "leaving_side"), "ref_ptr": ("ref",)}


def join_declarations(
    parts: Sequence[tuple[EndpointTable, DeclarationTable]],
) -> tuple[EndpointTable, DeclarationTable]:
    r"""
    The declarations of every part in order over one endpoint table: equal endpoints (occurrence, atom, extra, name
    and the outside naming) become one row, in ascending key order.
    """

    columns_of = ("occurrence", "atom", "extra", "name", "outside_instance", "outside_component")
    keys = np.concatenate(
        [np.stack([getattr(e, name) for name in columns_of], axis=1) for e, _ in parts]
        or [np.zeros((0, len(columns_of)), np.int32)]
    )
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = np.append(inverse.ravel(), -1).astype(np.int32)  # -1 stays "no endpoint"
    starts = np.cumsum([0, *(len(e.occurrence) for e, _ in parts)])
    columns: dict[str, NDArray] = {}
    for name in (f.name for f in fields(DeclarationTable)):
        values = [getattr(t, name) for _, t in parts]
        if name in ("first", "second"):
            values = [inverse[np.where(v >= 0, v + start, -1)] for v, start in zip(values, starts)]
        if name in _CSR:
            offsets = np.cumsum([0, *(v[-1] for v in values)])
            columns[name] = np.concatenate([[0]] + [v[1:] + o for v, o in zip(values, offsets)]).astype(np.int64)
        else:
            columns[name] = np.concatenate(values) if values else _defaults(0).get(name, np.zeros(0, np.int32))
    first, second = columns.pop("first"), columns.pop("second")
    endpoints = endpoint_table(*(unique.T if len(unique) else np.zeros((len(columns_of), 0), np.int32)))
    return endpoints, declaration_table(first, second, **columns)


def in_use(occurrences: OccurrenceTable, endpoints: EndpointTable, declarations: DeclarationTable) -> NDArray[np.uint8]:
    r"""
    The one in-use set: per occurrence atom OBSERVED (an observed row), PLACED (any other row, but one stating
    occupancy exactly 0: that row is no evidence the atom stays) and BONDED (an endpoint of any declaration of the
    chemical state, candidates and symmetry partners included).
    """

    placed = occurrences.placed & ~occurrences.zero_occupancy
    bits = occurrences.observed * np.uint8(InUse.OBSERVED) | placed * np.uint8(InUse.PLACED)
    ends = np.concatenate([declarations.first, declarations.second])
    ends = ends[ends >= 0]
    atom, occurrence = endpoints.atom[ends], endpoints.occurrence[ends]
    hit = atom >= 0
    bits[occurrences.atom_ptr[occurrence[hit]] + atom[hit]] |= np.uint8(InUse.BONDED)
    return bits.astype(np.uint8)


def polymer_link_declarations(
    pairs: NDArray[np.int32],
    kind: NDArray[np.uint8],
    existence: NDArray[np.uint8],
    existence_reason: NDArray[np.uint16],
    occurrences: OccurrenceTable,
    definitions: Sequence[ChemicalComponent | None],
    names: Mapping[str, int],
) -> tuple[EndpointTable, DeclarationTable]:
    r"""
    The rows of junctions, caps and unlocated links.

    ``pairs`` are occurrence rows (previous, next) of consecutive positions. A JUNCTION joins the left definition's
    verified next port to the right one's previous port, a CAP the cap atom to the partner's port (``cap_link``); both
    have order ``Defined(1, implied)``, a retention basis at both ports and multiplicity fixed by the sequence
    (``topology_fixed``, count 1 at both sides). An UNLOCATED row has one side -1 and runs from the other side's facing
    port to no partner. A port of an UNDEFINED side has no atom; its endpoint carries the template id. ``names`` maps
    every atom name and template id the rows need to its code.

    Raises:
        InvalidRequestError: ``value_invalid`` for a linked side without a definition or a verified facing port,
            or for a cap without a valid partner.
    """

    pairs, link = np.asarray(pairs).reshape(-1, 2), np.asarray(kind, dtype=np.uint8)
    present = pairs >= 0  # a side of the pair; column 0 faces with its next port, column 1 with its previous
    definition = np.where(present, occurrences.definition[np.where(present, pairs, 0)], 0)
    if len(bad := np.flatnonzero((present & (definition < 0)).any(axis=1))):
        left, right = pairs[bad[0]].tolist()
        fail("value_invalid", RequestRef("pairs", f"occurrences {left} {right}: a linked side has no definition"))

    def end(component: ChemicalComponent, s: int, a: int | None = None) -> tuple[int, int]:
        r"""(atom, name code) of the endpoint at atom ``a`` (default: the facing port) of the side ``s``; a port of an
        UNDEFINED side (-1) carries the template id."""
        record = ports(component)
        if a is None and record.status[s] not in (PortStatus.VERIFIED, PortStatus.UNDEFINED):
            fail(
                "value_invalid",
                RequestRef("linked", f"{component.component_id} side {s}: port {record.status[s].name}"),
                condition=record.condition[s].name,
            )
        a = (record.previous, record.next)[s] if a is None else a
        return a, names[component.atom_names[a] if a >= 0 else cast(str, (record.previous_name, record.next_name)[s])]

    # a junction or unlocated endpoint is the facing port of its definition and side: one lookup per distinct pair
    facing = present & (link != LinkKind.CAP)[:, None]
    keys, inverse = np.unique((definition * 2 + (1 - np.arange(2)))[facing], return_inverse=True)
    found_ports = [end(definitions[key // 2], key % 2) for key in keys.tolist()]
    atom, name = np.zeros((2, *pairs.shape), dtype=np.int64)
    atom[facing], name[facing] = np.array(found_ports, dtype=np.int64).reshape(-1, 2)[inverse.ravel()].T
    for row in np.flatnonzero(link == LinkKind.CAP).tolist():  # the cap atom and the partner's port (cap_link)
        found = [definitions[d] for d in definition[row].tolist()]
        pairing = cap_link(*found)
        if pairing is None:
            fail("value_invalid", RequestRef("linked", f"invalid cap {found[0].component_id} {found[1].component_id}"))
        cap_atom, port = pairing
        cap = TERMINAL_CAPS.get(found[0].component_id)
        atoms = [cap_atom, port] if cap is not None and cap.side == "next" else [port, cap_atom]
        atom[row], name[row] = zip(*(end(c, s, a) for c, s, a in zip(found, (1, 0), atoms)))
    sides = present.sum(axis=1)
    first = np.cumsum(sides) - sides  # the endpoints of a row are consecutive, its present sides in order
    second = np.where(sides == 2, first + 1, -1)
    d = len(first)
    linked = link != LinkKind.UNLOCATED
    endpoints = endpoint_table(pairs[present], atom[present], np.full(int(sides.sum()), -1), name[present])
    table = declaration_table(
        first,
        second,
        count=np.where(linked, 1, -1)[:, None].repeat(2, axis=1),  # a sequence link counts one neighbour
        topology_fixed=linked,
        existence=existence,
        existence_reason=existence_reason,
        origin=np.where(link == LinkKind.CAP, BondOrigin.CAP, BondOrigin.JUNCTION),
        kind=np.full(d, BondKind.COVALENT),
        kind_state=np.full(d, _KNOWN_DEFINED),
        order_defined=linked.astype(np.uint8),
        order_defined_strength=np.where(linked, Strength.IMPLIED, Strength.NONE),
        retention_basis=np.repeat(linked[:, None], 2, axis=1),
    )
    return endpoints, table


def recipe_declarations(
    first: NDArray[np.int64],
    second: NDArray[np.int64],
    kind: NDArray[np.uint8],
    order: NDArray[np.uint8],
    leaving: Sequence[Sequence[tuple[int, int]]],
    occurrences: OccurrenceTable,
    definitions: Sequence[ChemicalComponent | None],
) -> tuple[EndpointTable, DeclarationTable]:
    r"""
    The rows of recipe bonds. ``first``/``second`` are ``[m, 3]`` endpoints (occurrence, definition atom, name code);
    ``leaving`` per row the caller's declared ids as (definition atom, side 0 | 1), a dictionary-class override. A
    recipe states the kind (KNOWN×DECLARED) and the order where given (0: not stated, UNKNOWN×NONE); one neighbour at
    each side, the identity transform, origin RECIPE; a bond inside one occurrence that its definition has restates
    that bond.
    """

    m = len(first)
    ends = np.concatenate((first, second)).reshape(-1, 3)
    endpoints = endpoint_table(ends[:, 0], ends[:, 1], np.full(2 * m, -1), ends[:, 2])
    restatement = np.full(m, Restatement.NEW_INTER, dtype=np.uint8)
    reference_order = np.zeros(m, dtype=np.uint8)
    reference_order_token = np.zeros(m, dtype=np.uint8)
    reference_aromatic = np.zeros(m, dtype=np.uint8)
    for k, ((o1, a1, _), (o2, a2, _)) in enumerate(zip(first.tolist(), second.tolist())):
        if o1 != o2:
            continue
        definition = definitions[occurrences.definition[o1]]
        bonds = [b for b, pair in enumerate(definition.bond_atoms.tolist()) if set(pair) == {a1, a2}]
        restatement[k] = Restatement.RESTATEMENT if bonds else Restatement.NEW_INTRA
        if bonds:
            b = bonds[0]
            reference_order[k] = definition.bond_order[b] if known(definition.bond_order_state[b]) else 0
            reference_order_token[k] = definition.bond_order_token[b]
            reference_aromatic[k] = definition.bond_aromatic[b] + 1 if known(definition.bond_aromatic_state[b]) else 0
    order = np.asarray(order, dtype=np.uint8)
    table = declaration_table(
        np.arange(m),
        np.arange(m, 2 * m),
        count=np.ones((m, 2), dtype=np.int8),
        transform=np.zeros(m, dtype=np.int32),
        restatement=restatement,
        origin=np.full(m, BondOrigin.RECIPE),
        kind=kind,
        kind_state=np.full(m, _KNOWN_DECLARED),
        order_token=order,
        order_declared=order,
        order_declared_state=np.where(order > 0, _KNOWN_DECLARED, _UNKNOWN),
        reference_order=reference_order,
        reference_order_token=reference_order_token,
        reference_aromatic=reference_aromatic,
        leaving_ptr=np.cumsum([0, *map(len, leaving)]),
        leaving_atom=np.array([atom for row in leaving for atom, _ in row], dtype=np.int32),
        leaving_side=np.array([side for row in leaving for _, side in row], dtype=np.uint8),
    )
    return endpoints, table


def inherited_declarations(
    first: NDArray[np.int64],
    second: NDArray[np.int64],
    edges: Mapping[str, NDArray],
    leaving: Sequence[Sequence[tuple[int, int]]],
    outside: NDArray[np.int64] | None = None,
) -> tuple[EndpointTable, DeclarationTable]:
    r"""
    The rows that restate resolved edges between two occurrences as they are. ``first``/``second`` are ``[m, 3]``
    endpoints (occurrence, definition atom, name code); ``edges`` the resolved ``bond_*`` columns with their states
    (``bond_direction`` relative to ``first``); ``leaving`` per row its declared ids as (definition atom, side);
    ``outside`` per row ``[m, 2]`` the outside naming of its second end (instance code, ordinal; -1 for an end of the
    value): such a second end is an endpoint outside the value, ``second`` holding only its name code.

    Each row states the evidence whose merge gives back the edge's kind, order, aromaticity and direction with their
    states: the kind with its state (an undetermined one as a connection only), a declared order as its token, a
    defined order as intrinsic (a declaration that contradicts it conflicts), a declared order of a defined linkage
    with that linkage, a default order as the default, a stated null as stated, a declared direction; the edge's
    origin, the identity transform and one neighbour at each side.

    Raises:
        InvalidRequestError: ``value_invalid`` for an edge whose states no row restates (a basis only a reference,
            an assumption or a conflict gives).
    """

    m = len(first)
    ends = np.concatenate((first, second)).reshape(-1, 3)
    named = np.full((2 * m, 2), -1, dtype=np.int64)
    if outside is not None:
        named[m:] = outside
    away = named[:, 0] >= 0
    occurrence, atom = np.where(away, -1, ends[:, 0]), np.where(away, -1, ends[:, 1])
    endpoints = endpoint_table(occurrence, atom, np.full(2 * m, -1), ends[:, 2], named[:, 0], named[:, 1])
    order, order_state = (np.asarray(edges[name], dtype=np.uint8) for name in ("bond_order", "bond_order_state"))
    declared, defined = order_state == _KNOWN_DECLARED, order_state == _KNOWN_DEFINED
    linkage = defined | (declared & (np.asarray(edges["bond_aromatic_state"]) == _KNOWN_DEFINED))
    directed = np.asarray(edges["bond_direction_state"]) == _KNOWN_DECLARED
    kind_state = np.asarray(edges["bond_kind_state"], dtype=np.uint8)
    table = declaration_table(
        np.arange(m),
        np.arange(m, 2 * m),
        count=np.ones((m, 2), dtype=np.int8),
        connectivity_only=kind_state == _UNKNOWN,
        transform=np.zeros(m, dtype=np.int32),
        restatement=np.full(m, Restatement.NEW_INTER, dtype=np.uint8),
        origin=edges["bond_origin"],
        kind=edges["bond_kind"],
        kind_state=kind_state,
        order_token=np.where(declared, order, OrderToken.NONE),
        order_declared=np.where(declared, order, 0),
        order_declared_state=np.where(declared | (order_state == _UNKNOWN_DECLARED), order_state, _UNKNOWN),
        order_defined=np.where(linkage, order, 0),
        order_defined_strength=np.where(
            defined, Strength.INTRINSIC, np.where(linkage, Strength.IMPLIED, Strength.NONE)
        ),
        order_default=np.where(order_state == _KNOWN_DEFAULT, order, 0),
        direction_declared=np.where(directed, edges["bond_direction"], 0),
        direction_declared_state=np.where(directed, _KNOWN_DECLARED, _UNKNOWN),
        leaving_ptr=np.cumsum([0, *map(len, leaving)]),
        leaving_atom=np.array([atom for row in leaving for atom, _ in row], dtype=np.int32),
        leaving_side=np.array([side for row in leaving for _, side in row], dtype=np.uint8),
    )
    stated = np.stack(
        [np.asarray(edges[f"bond_{name}{suffix}"], dtype=np.int64) for name in _MERGED for suffix in ("", "_state")],
        axis=1,
    )
    evidence = np.stack([getattr(table, name).astype(np.int64) for name in _EVIDENCE], axis=1)
    for row, (values, expected) in enumerate(zip(evidence.tolist(), stated.tolist())):
        merged = _merge(((*values, 0, 0),))
        if list(merged[:8]) != expected or merged[8]:
            fail("value_invalid", RequestRef("edges", str(row)), reason="no declaration restates its states")
    return endpoints, table


# The fields ``_merge`` returns first, in its order, each with its state.
_MERGED = ("order", "aromatic", "kind", "direction")
# Per-row evidence the merge reads; the signature of a group is the sorted tuple of its rows' evidence.
_EVIDENCE = (
    "kind",
    "kind_state",
    "order_declared_state",
    "order_token",
    "order_defined",
    "order_defined_strength",
    "order_default",
    "direction_declared",
    "direction_declared_state",
    "restatement",
    "reference_order",
    "reference_order_token",
    "reference_aromatic",
)


def _merge(rows: tuple[tuple[int, ...], ...]) -> tuple:
    r"""
    The effective fields of one physical edge from its rows' evidence; each row is ``_EVIDENCE`` plus (existence
    assumed, order assumed value). Returns ``(order, order_state, aromatic, aromatic_state, kind, kind_state,
    direction, direction_state, conflicts, findings)``.
    """

    ev = [dict(zip((*_EVIDENCE, "existence_assumed", "order_assumed"), row)) for row in rows]
    conflicts: list[str] = []
    findings: list[str] = []

    restated = [r for r in ev if r["restatement"] == Restatement.RESTATEMENT]
    kinds = [(r["kind"], r["kind_state"]) for r in ev if known(r["kind_state"])]
    if len({k for k, _ in kinds}) > 1:
        kind, kind_state = 0, _CONFLICT
        conflicts.append("definition_vs_definition")
    elif kinds:
        kind = kinds[0][0]
        kind_state = _KNOWN_ASSUMED if all(s == _KNOWN_ASSUMED for _, s in kinds) else _KNOWN_DEFINED
    elif restated:
        kind, kind_state = BondKind.COVALENT, _KNOWN_REFERENCE  # rows leaving the kind open keep the reference's
    else:  # a stated `?` before a stated `.`, before none
        kind = 0
        stated = {r["kind_state"] for r in ev}
        kind_state = next((s for s in (_UNKNOWN_DECLARED, _INAPPLICABLE_DECLARED) if s in stated), _UNKNOWN)

    reference = {r["reference_order"] for r in restated} - {0}  # 0: the reference bond states no integer order
    typed = any(r["reference_order_token"] > OrderToken.QUAD for r in restated)  # a type without one (AROM–POLY)
    declared = {r["order_token"] for r in ev if r["order_declared_state"] == _KNOWN_DECLARED}
    defined = {
        (r["order_defined"], r["order_defined_strength"], r["existence_assumed"]) for r in ev if r["order_defined"]
    }
    if kind == BondKind.COORDINATION and known(kind_state):
        if declared:
            findings.append("order_on_coordination")
        order, order_state = 0, _INAPPLICABLE_DEFINED
        if reference:  # a metalc restatement keeps the reference order on the one edge (1e61)
            order, order_state = max(reference), _KNOWN_REFERENCE
            if order > 1:
                findings.append("kind_order_mismatch")
    else:
        if kind == BondKind.COVALENT and any(r["order_declared_state"] == _INAPPLICABLE_DECLARED for r in ev):
            findings.append("inapplicable_on_covalent")
        intrinsic = {v for v, s, _ in defined if s == Strength.INTRINSIC}
        implied = {v for v, s, _ in defined if s == Strength.IMPLIED}
        stated_null = any(r["order_declared_state"] in (_UNKNOWN_DECLARED, _INAPPLICABLE_DECLARED) for r in ev)
        if len(declared) > 1:
            order, order_state = 0, _CONFLICT
            conflicts.append("declaration_vs_declaration")
        elif declared:
            (order,) = declared
            numeric = order if order <= OrderToken.QUAD else 0
            if intrinsic - {numeric}:
                order, order_state = 0, _CONFLICT
                conflicts.append("type_vs_value")
            elif (reference and reference != {numeric}) or (typed and numeric):  # an integer against such a type
                order, order_state = 0, _CONFLICT
                conflicts.append("declaration_vs_reference")
            else:
                order_state = _KNOWN_DECLARED
                if implied - {numeric}:
                    findings.append("implied_value_overridden")
        elif reference:
            order, order_state = max(reference), _KNOWN_REFERENCE
        elif typed:  # the reference's AROM/DELO/PI/POLY type stays, never filled by a default or an assumption
            order, order_state = 0, _UNKNOWN_REFERENCE
            if defined:  # a defined integer order against it
                order_state = _CONFLICT
                conflicts.append("type_vs_value")
        elif defined:
            values = {v for v, _, _ in defined}
            if len(values) > 1:
                order, order_state = 0, _CONFLICT
                conflicts.append("definition_vs_definition")
            else:
                (order,) = values
                order_state = _KNOWN_ASSUMED if all(a for _, _, a in defined) else _KNOWN_DEFINED
        elif defaults := {r["order_default"] for r in ev if r["order_default"]}:
            order, order_state = min(defaults), _KNOWN_DEFAULT
        elif not restated and (assumed := {r["order_assumed"] for r in ev if r["order_assumed"]}):
            order, order_state = min(assumed), _KNOWN_ASSUMED  # never for a reference bond whose order is unknown
        else:  # a restatement that settles nothing keeps its reference bond's own unknown
            order, order_state = 0, _UNKNOWN_DECLARED if stated_null else _UNKNOWN_REFERENCE if restated else _UNKNOWN

    aromatic, aromatic_state = 0, _UNKNOWN  # no source row states an edge's aromaticity
    if codes := {r["reference_aromatic"] for r in restated} - {0}:  # the reference bond's: 2 aromatic, 1 not
        aromatic, aromatic_state = int(2 in codes), _KNOWN_REFERENCE
    elif defined:  # a defined linkage (junction, cap, disulfide, role) is not aromatic
        aromatic_state = _KNOWN_ASSUMED if all(a for _, _, a in defined) else _KNOWN_DEFINED

    directions = {r["direction_declared"] for r in ev if r["direction_declared_state"] == _KNOWN_DECLARED}
    direction, direction_state = 0, _UNKNOWN
    if len(directions) > 1:
        direction_state = _CONFLICT
        conflicts.append("declaration_vs_declaration")
    elif directions:
        (direction,) = directions
        direction_state = _KNOWN_DECLARED

    return (
        order,
        order_state,
        aromatic,
        aromatic_state,
        kind,
        kind_state,
        direction,
        direction_state,
        tuple(dict.fromkeys(conflicts)),
        tuple(dict.fromkeys(findings)),
    )


def merge_edge_fields(
    declarations: DeclarationTable, refs: SourceRefs, assumptions: Sequence[Assumption] = ()
) -> tuple[MergedEdges, Records]:
    r"""
    Merge the declarations of each physical edge: the only merge site.

    Rows whose existence is not KNOWN (standard candidates, unlocated links), with no second endpoint or ``self_bond``
    make no edge; a self-bond records ``Conflict(self_bond)``. ``assumptions`` fill only undetermined
    items: an assumed existence turns a candidate into an edge whose order basis is ASSUMED, an assumed kind
    makes a kind-undetermined row coordination, an assumed order fills an UNKNOWN covalent order. One record per
    conflict code per edge; record subjects are (EDGE, the edge's representative declaration row). The
    representative is the row with the smallest transform row (the identity is row 0), then the smallest row.
    """

    d = len(declarations.first)
    t = declarations
    assumed_existence = np.zeros(d, dtype=bool)
    assumed_kind = np.zeros(d, dtype=bool)
    assumed_order = np.zeros(d, dtype=np.uint8)
    for a in assumptions:
        if a.item == "existence":
            assumed_existence[a.declaration] = True
        elif a.item == "kind":
            assumed_kind[a.declaration] = True
        else:
            assumed_order[a.declaration] = a.value
    kind = np.where(assumed_kind, BondKind.COORDINATION, t.kind).astype(np.uint8)
    kind_state = np.where(assumed_kind, _KNOWN_ASSUMED, t.kind_state).astype(np.uint8)

    log: list[tuple] = []
    for row in np.flatnonzero(t.self_bond).tolist():
        log.append(("self_bond", SubjectKind.EDGE, row, t.ref[t.ref_ptr[row] : t.ref_ptr[row + 1]].tolist(), ()))
    edge_rows = (known(t.existence) | assumed_existence) & (t.second >= 0) & ~t.self_bond
    rows = np.flatnonzero(edge_rows)
    group_key = np.where(t.physical_edge[rows] >= 0, t.physical_edge[rows], d + rows)
    order = np.lexsort((rows, t.transform[rows], group_key))
    rows, group_key = rows[order], group_key[order]
    starts = np.flatnonzero(np.r_[True, group_key[1:] != group_key[:-1]]) if len(rows) else np.zeros(0, np.int64)
    sizes = np.diff(np.r_[starts, len(rows)])
    rep = rows[starts]  # the representative: smallest transform, then smallest row
    first, second = t.first[rep], t.second[rep]

    # orientation of every row against its representative (a flipped row reads its direction negated)
    group = np.repeat(np.arange(len(rep)), sizes)
    flipped = (t.first[rows] != first[group]) & (first[group] != second[group])
    direction = np.where(flipped, -t.direction_declared[rows], t.direction_declared[rows])
    columns = {name: getattr(t, name)[rows] for name in _EVIDENCE}
    columns.update(kind=kind[rows], kind_state=kind_state[rows], direction_declared=direction)
    evidence = np.stack(
        [columns[name].astype(np.int64) for name in _EVIDENCE]
        + [assumed_existence[rows].astype(np.int64), assumed_order[rows].astype(np.int64)],
        axis=1,
    )

    results: list[tuple] = []
    cache: dict[tuple, tuple] = {}
    single = sizes == 1
    singles = [(tuple(row),) for row in evidence[starts[single]].tolist()]  # the key of a one-row group
    for single_key in singles:
        if single_key not in cache:
            cache[single_key] = _merge(single_key)
    single_result = iter(singles)
    for start, size in zip(starts.tolist(), sizes.tolist()):
        if size == 1:
            results.append(cache[next(single_result)])
            continue
        key = tuple(sorted(map(tuple, evidence[start : start + size].tolist())))
        if key not in cache:
            cache[key] = _merge(key)
        results.append(cache[key])

    for g, result in enumerate(results):
        if result[8] or result[9]:
            members = rows[starts[g] : starts[g] + sizes[g]]
            sources = sorted({int(r) for m in members.tolist() for r in t.ref[t.ref_ptr[m] : t.ref_ptr[m + 1]]})
            for code in (*result[8], *result[9]):
                log.append((code, SubjectKind.EDGE, int(rep[g]), sources, ()))

    def column(index: int, dtype: type) -> NDArray:
        return np.array([r[index] for r in results], dtype=dtype)

    # the origin of an edge: a junction, cap or branch-link row names it before a connection row
    priority = np.where(t.origin[rows] == BondOrigin.CONNECTION, 1, 0)
    origin_row = rows[np.lexsort((priority, group))][starts] if len(rows) else rows

    # declared leaving ids, oriented to the edge's sides
    counts = np.diff(t.leaving_ptr)[rows]
    position = np.repeat(np.arange(len(rows)), counts)
    entries = t.leaving_ptr[:-1][rows][position] + _ranges(counts)
    owner, owned = group[position], rows[position]
    endpoint = np.where(t.leaving_side[entries] == 0, t.first[owned], t.second[owned])
    side = (endpoint != first[owner]).astype(np.int64)
    leaving = np.unique(np.stack([owner, side, t.leaving_atom[entries].astype(np.int64)], axis=1), axis=0)
    leaving_ptr = np.searchsorted(leaving[:, 0], np.arange(len(rep) + 1))

    edges = MergedEdges(
        first=first.astype(np.int32),
        second=second.astype(np.int32),
        transform=t.transform[rep].astype(np.int32),
        origin=t.origin[origin_row].astype(np.uint8),
        order=column(0, np.uint8),
        order_state=column(1, np.uint8),
        aromatic=column(2, np.bool_),
        aromatic_state=column(3, np.uint8),
        kind=column(4, np.uint8),
        kind_state=column(5, np.uint8),
        direction=column(6, np.int8),
        direction_state=column(7, np.uint8),
        edge_class=t.restatement[rep].astype(np.uint8),
        declaration_ptr=np.r_[starts, len(rows)].astype(np.int64),
        declaration=rows.astype(np.int64),
        declared_leaving_ptr=np.asarray(leaving_ptr, dtype=np.int64),
        declared_leaving=leaving[:, 2].astype(np.int32),
        declared_leaving_side=leaving[:, 1].astype(np.uint8),
    )
    return edges, Records.from_rows(log, refs)


def _ranges(lengths: NDArray[np.int64]) -> NDArray[np.int64]:
    r"""``0 .. n-1`` for each length n, concatenated."""
    total = int(lengths.sum())
    return np.arange(total) - np.repeat(np.cumsum(lengths) - lengths, lengths)
