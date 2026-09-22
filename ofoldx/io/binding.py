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
Binding: the one residue table per (source, model), the PDB polymer-type derivation, the one
chemical state (model, altloc, microheterogeneity), the one atom-name rule and the one endpoint resolver with the one
query producer. Every identity fact of a residue is decided here once; later stages read these columns.

A residue is what the source declares to exist, observed or not; which keys identify it follows from its instance's
kind, as the dictionary defines it. mmCIF polymer rows bind by ``(label_asym_id, label_seq_id, label_comp_id)`` to a
candidate of the declared sequence; branched, non-polymer and water rows bind by the author address ``(auth_seq_id,
insertion code, component)`` to their scheme; a kind whose scheme is absent binds by the atom table's own keys. A PDB
chain's polymer residues are those of its SEQRES correspondence; every other residue of the format is a heterogen of
its own instance (the waters of a chain share one). Ordinals follow instance order, then declaration order, then the
first deposited row, so permuting coordinate rows changes no ordinal.

Symbol families read (the codes of every table index ``AtomTable.symbols``, as ``io.tables`` names them): ``chain``
(label asym and author chain ids), ``comp``, ``atom``, ``alt`` and ``ins``.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
from numpy.typing import NDArray

from ..chemistry.fragments import FAMILY_TABLE, POLYMER_FAMILIES, LinkingFamily
from ..data.molecular import EntityType, MonomerType
from ..errors import RequestRef, SourceRef, fail
from ..provenance import Basis, Records, Resolution, SourceRefs, SubjectKind, known, resolution
from ..provenance import state as fact_state
from .tables import TableBuilder

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ..chemistry.definitions import ChemicalComponent, Definitions
    from .tables import AtomTable, SourceTables

AltlocPolicy = Literal["highest_occupancy"] | str  # or one altloc label

_KNOWN_DECLARED = fact_state(Resolution.KNOWN, Basis.DECLARED)
_KNOWN_DERIVED = fact_state(Resolution.KNOWN, Basis.DERIVED)
_UNKNOWN_DERIVED = fact_state(Resolution.UNKNOWN, Basis.DERIVED)
_INAPPLICABLE = fact_state(Resolution.INAPPLICABLE, Basis.DEFINED)
_BLANK = fact_state(Resolution.INAPPLICABLE, Basis.DECLARED)  # a blank PDB chain: the stated none
_UNKNOWN = fact_state(Resolution.UNKNOWN, Basis.NONE)
_POLYMER, _BRANCHED, _WATER = EntityType.POLYMER, EntityType.BRANCHED, EntityType.WATER
_ONE_MOLECULE = (EntityType.NON_POLYMER, EntityType.MACROLIDE)
_SEQ, _ADDRESS, _SINGLE = 0, 1, 2  # how the rows of an instance bind
_ATOM, _HETATM = 1, 2  # AtomTable.group (0 not stated)
_PDB_WATER = frozenset(("HOH", "DOD"))  # the format's water residue names
POLYMER_TYPES: tuple[str, ...] = tuple(POLYMER_FAMILIES)  # the codes pdb_polymer_types returns
# EndpointQuery.field_state columns (the item order of ConnectionTable.field_state)
_LABEL_ASYM, _LABEL_SEQ, _LABEL_COMP, _LABEL_ATOM, _ALTLOC = range(5)
_AUTH_ASYM, _AUTH_SEQ, _AUTH_INS, _AUTH_COMP, _AUTH_ATOM = range(5, 10)
_FIELDS = 10
NO_CODE = -2  # an unknown code in a key: never equal to a code (>= 0) or to "none" (-1)
_PENDING = 255


@dataclass(frozen=True)
class ResidueTable:
    r"""
    The residues of one (source, model): every declared residue and microheterogeneity candidate, observed or not.

    ``kind`` is a ``data.molecular.EntityType`` value, 0 when the instance's type is undetermined (a PDB heterogen, an
    mmCIF entity whose type is not KNOWN). ``asym`` is the label asym code (-1 for PDB); ``instance`` indexes
    ``instance_symbol`` (mmCIF: ``_struct_asym`` order, the source's symbols; PDB: polymer chains, then heterogens in
    first-row order, symbols OFoldX generates). An instance without residues is an asym whose residues
    nothing determines. ``number`` is the sequence position (``label_seq_id``, branch ``num``, SEQRES ``num``);
    ``position`` groups the candidates of one sequence position (a PDB site's residues). ``component`` is stated for
    every residue, the component id of its occurrence (an unstated one is ``residue_unbound``). A PDB ``entity`` is
    the chain's ``COMPND`` entity, -1 where none. ``declaration`` is the row of ``records.refs`` that
    declares the residue (scheme, sequence or SEQRES row), -1 when only coordinates state it. ``row_ptr``/``rows``
    are each ordinal's atom rows (selected model, row order); ``ordinal_of_row`` and ``name_of_row`` (the atom-name
    code a row binds by, ``label_atom_id`` else the author name) cover every row of the atom table, -1 outside the
    model.

    The author address is decided here once (``_addresses``): chain, number (``author_number`` the integer reading of
    ``author_text``, with its state), insertion and ``author_component`` (-1 not stated), and ``author_key`` the
    number as the format compares it (``author_key``). Per instance: its label asym (-1 for PDB), its entity (-1
    none) and its author chain, the one its residues state (``asym_two_chains`` otherwise).
    """

    kind: NDArray[np.uint8]
    asym: NDArray[np.int32]
    entity: NDArray[np.int32]
    instance: NDArray[np.int32]
    number: NDArray[np.int32]
    number_state: NDArray[np.uint8]
    position: NDArray[np.int32]
    component: NDArray[np.int32]
    author_component: NDArray[np.int32]
    author_chain: NDArray[np.int32]
    author_chain_state: NDArray[np.uint8]
    author_number: NDArray[np.int64]
    author_state: NDArray[np.uint8]
    author_text: NDArray[np.int32]
    author_key: NDArray[np.int64]
    insertion: NDArray[np.int32]
    declared: NDArray[np.bool_]
    observed: NDArray[np.bool_]
    declaration: NDArray[np.int64]
    row_ptr: NDArray[np.int64]
    rows: NDArray[np.int64]
    ordinal_of_row: NDArray[np.int32]
    name_of_row: NDArray[np.int32]
    instance_symbol: tuple[str, ...]
    instance_asym: NDArray[np.int32]
    instance_entity: NDArray[np.int32]
    instance_author_chain: NDArray[np.int32]
    instance_author_chain_state: NDArray[np.uint8]
    records: Records


@dataclass(frozen=True)
class ChemicalState:
    r"""The one model, altloc and microheterogeneity choice; later stages read only these masks."""

    selected: NDArray[np.bool_]
    row_selected: NDArray[np.bool_]
    altloc: NDArray[np.int32]
    records: Records


class BindingOutcome(IntEnum):
    RESOLVED = 0
    ABSENT_IN_MODEL = 1
    UNSELECTED_ALTERNATIVE = 2
    AMBIGUOUS = 3
    LABEL_AUTH_CONFLICT = 4
    DANGLING = 5
    UNRESOLVED_ASYM = 6
    ATOM_UNBOUND = 7
    ATOM_NOT_IN_REFERENCE = 8


class QuerySource(IntEnum):
    STRUCT_CONN = 1
    SSBOND = 2
    LINK = 3
    BRANCH_LINK = 4
    CONECT = 5


@dataclass(frozen=True)
class EndpointQuery:
    r"""
    One query per endpoint, in pairs (side 0, side 1) of one source row: every ``struct_conn``/``SSBOND``/``LINK`` row,
    then every branch link once per asym of its branched entity, then every CONECT pair. ``field_state`` [q, 10] holds
    the states of label asym, seq, comp, atom, altloc, author asym, seq, insertion, comp, atom; ``auth_key`` is the
    author number as ``ResidueTable.author_key`` compares it. ``scheme_num`` is a branch link's ``num`` (-1 none);
    ``row`` the atom row a CONECT serial names in the selected model (-1 none: the author namespace of the serial's
    first-model row then re-binds it).
    """

    source: NDArray[np.uint8]
    label_asym: NDArray[np.int32]
    label_seq: NDArray[np.int64]
    label_comp: NDArray[np.int32]
    label_atom: NDArray[np.int32]
    auth_asym: NDArray[np.int32]
    auth_key: NDArray[np.int64]
    auth_ins: NDArray[np.int32]
    auth_comp: NDArray[np.int32]
    auth_atom: NDArray[np.int32]
    altloc: NDArray[np.int32]
    field_state: NDArray[np.uint8]
    scheme_num: NDArray[np.int32]
    row: NDArray[np.int64]
    ref: NDArray[np.int64]


@dataclass(frozen=True)
class EndpointResolution:
    r"""
    Per query: the ``BindingOutcome``, the residue ordinal (-1 none), the definition atom (-1 none), the selected atom
    row of an observed atom the definition lacks or of a definition-less residue (``extra``, -1 none), the atom-name
    code (-1 when the query names none) and, for AMBIGUOUS and LABEL_AUTH_CONFLICT, the candidate ordinals.
    """

    outcome: NDArray[np.uint8]
    ordinal: NDArray[np.int32]
    atom: NDArray[np.int32]
    extra: NDArray[np.int32]
    name: NDArray[np.int32]
    candidates_ptr: NDArray[np.int64]
    candidates: NDArray[np.int32]


# --- helpers ------------------------------------------------------------------------------------------------------


def _code(values: NDArray, states: NDArray[np.uint8]) -> NDArray[np.int64]:
    r"""The code where the state is KNOWN, else -1 (a code column alone never stands for a null)."""
    return np.where(known(states), values, -1).astype(np.int64)


def stated_identity(values: NDArray, states: NDArray[np.uint8]) -> NDArray[np.int64]:
    r"""An identity as the source states it: the code where KNOWN, -1 where it states none (a blank PDB chain, ``.``),
    ``NO_CODE`` where it states nothing."""
    r = resolution(states)
    return np.select([r == Resolution.KNOWN, r == Resolution.INAPPLICABLE], [values, -1], NO_CODE).astype(np.int64)


def _token(text: NDArray, states: NDArray[np.uint8]) -> NDArray[np.int64]:
    r"""
    An author number as the source states it: its token (``auth_seq_id`` and ``pdb_seq_num`` are codes; the text code
    is -1 exactly where no token is stated), -1 where it states none (``.``), ``NO_CODE`` where it states nothing. The
    state is the integer reading's (UNKNOWN×DERIVED for a token that is no integer), so the token decides first.
    """
    none = resolution(states) == Resolution.INAPPLICABLE
    return np.where(text >= 0, text, np.where(none, -1, NO_CODE)).astype(np.int64)


def author_key(text: NDArray, states: NDArray[np.uint8], integer: NDArray, pdb: bool) -> NDArray[np.int64]:
    r"""
    An author number as the format compares it, the one key every author address is matched by: an mmCIF
    number by its token (``auth_seq_id`` is a code), a PDB ``resSeq`` by its integer reading (the format defines one;
    the integers ``0, -1, 1, -2, …`` as the keys ``0, 1, 2, 3, …``, so a key is never -1 or ``NO_CODE``); -1 where the
    source states none (``.``), ``NO_CODE`` where it states nothing.
    """
    if not pdb:
        return _token(text, states)
    value = np.asarray(integer, dtype=np.int64)
    return np.where(known(states), np.where(value >= 0, 2 * value, -2 * value - 1), NO_CODE).astype(np.int64)


def _floor(values: NDArray, valid: NDArray[np.bool_]) -> int:
    r"""A value at or below 0 and below every valid value: ``floor - k`` are keys no stated number takes."""
    return min(int(values[valid].min()) if valid.any() else 0, 0)


def _pack(*columns: NDArray) -> NDArray[np.int64]:
    r"""
    One int64 key per item from integer columns (sorted-key joins): equal exactly where the items' columns are equal,
    and ordered as the items are, column by column. The columns' offsets packed into one integer when their ranges fit
    63 bits; otherwise each item's rank among the distinct items (a lexsort, then adjacent rows compared), which groups
    and orders the same.
    """

    arrays = [np.asarray(column, dtype=np.int64) for column in columns]
    if not len(arrays[0]):
        return np.zeros(0, dtype=np.int64)
    lows = [int(column.min()) for column in arrays]
    widths = [max(1, (int(column.max()) - low).bit_length()) for column, low in zip(arrays, lows)]
    if sum(widths) <= 63:
        keys = np.zeros(len(arrays[0]), dtype=np.int64)
        for column, low, width in zip(arrays, lows, widths):
            keys = (keys << width) | (column - low)
        return keys
    order = np.lexsort(arrays[::-1])
    rows = np.stack(arrays, axis=1)[order]
    new = np.ones(len(rows), dtype=bool)
    new[1:] = (rows[1:] != rows[:-1]).any(axis=1)
    keys = np.empty(len(rows), dtype=np.int64)
    keys[order] = np.cumsum(new) - 1
    return keys


def _first_groups(keys: NDArray[np.int64]) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    r"""Per item its group, groups numbered in first-occurrence order, and each group's first item."""

    if not len(keys):
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    _, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
    rank = np.empty(len(first), dtype=np.int64)
    rank[np.argsort(first, kind="stable")] = np.arange(len(first))
    return rank[inverse.ravel()], np.sort(first)


def _pairs(keys: NDArray[np.int64], probes: NDArray[np.int64]) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    r"""Every (probe, key) index pair of equal values."""

    order = np.argsort(keys, kind="stable")
    low = np.searchsorted(keys, probes, "left", sorter=order)
    high = np.searchsorted(keys, probes, "right", sorter=order)
    owner, index = _expand(low, high)
    return owner, order[index]


def _lookup(keys: NDArray[np.int64], probes: NDArray[np.int64]) -> NDArray[np.int64]:
    r"""Per probe the index of an equal key (the first in key order), -1 none."""

    if not len(keys):
        return np.full(len(probes), -1, dtype=np.int64)
    order = np.argsort(keys, kind="stable")
    hit = order[np.searchsorted(keys, probes, sorter=order).clip(0, len(keys) - 1)]
    return np.where(keys[hit] == probes, hit, -1)


def _expand(low: NDArray[np.int64], high: NDArray[np.int64]) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    r"""The ranges ``[low, high)`` as (owner, index) pairs."""

    counts = high - low
    owner = np.repeat(np.arange(len(low)), counts)
    return owner, np.repeat(low, counts) + np.arange(counts.sum()) - np.repeat(_ptr(counts)[:-1], counts)


def _ptr(counts: NDArray) -> NDArray[np.int64]:
    ptr = np.zeros(len(counts) + 1, dtype=np.int64)
    np.cumsum(counts, out=ptr[1:])
    return ptr


def _split(keys: NDArray) -> dict[int, NDArray[np.int64]]:
    r"""Item indices of each key, in item order."""

    if not len(keys):
        return {}
    order = np.argsort(keys, kind="stable")
    unique, starts = np.unique(keys[order], return_index=True)
    return dict(zip(unique.tolist(), np.split(order, starts[1:])))


def _disagreeing(
    owner: NDArray[np.int64], value: NDArray[np.int64], order: NDArray[np.int64] | None = None
) -> tuple[int, int] | None:
    r"""Two items of one owner stating different values (``NO_CODE`` states nothing and agrees with any), None when
    every owner's stated values agree: of the lowest such owner, its first stated item and its first stating another
    value. ``order`` is the owners' stable argsort, when the caller has it."""

    order = np.argsort(owner, kind="stable") if order is None else order
    told = order[value[order] != NO_CODE]
    if not len(told):
        return None
    head = np.ones(len(told), dtype=bool)
    head[1:] = owner[told[1:]] != owner[told[:-1]]
    lead = told[np.flatnonzero(head)][np.cumsum(head) - 1]  # per stated item, its owner's first stated item
    differ = np.flatnonzero(value[told] != value[lead])
    return (int(lead[differ[0]]), int(told[differ[0]])) if len(differ) else None


def _told_first(
    owner: NDArray[np.int64], value: NDArray[np.int64], order: NDArray[np.int64] | None = None
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    r"""The owners (ascending) and per owner its first item stating a value (not ``NO_CODE``), else its first item.
    ``order`` is the owners' stable argsort, when the caller has it."""

    order = np.argsort(owner, kind="stable") if order is None else order
    ordered = owner[order]
    head = np.ones(len(order), dtype=bool)
    head[1:] = ordered[1:] != ordered[:-1]
    starts, group = np.flatnonzero(head), np.cumsum(head) - 1
    told = np.flatnonzero(value[order] != NO_CODE)
    first = np.ones(len(told), dtype=bool)
    first[1:] = group[told[1:]] != group[told[:-1]]
    at = starts.copy()
    at[group[told[first]]] = told[first]
    return ordered[starts], order[at]


def _row_ref(atoms: AtomTable, row: int, block: str | None) -> SourceRef:
    r"""The coordinate statement of one atom row: an ``atom_site`` row (mmCIF, which has a block) or a PDB record."""

    if block is not None:
        category = "atom_site"
    else:
        category = "HETATM" if atoms.group[row] == _HETATM else "ATOM"
    return SourceRef(category, int(atoms.row[row]), None, block, int(atoms.model[row]))


def _refs(refs: SourceRefs, rows: NDArray[np.int64]) -> tuple[SourceRef, ...]:
    return tuple(refs.ref(int(row)) for row in rows if row >= 0)


def _atom_name(atoms: AtomTable, rows: NDArray[np.int64]) -> NDArray[np.int64]:
    r"""The atom-name code of rows: ``label_atom_id``, or the author name a PDB row carries."""
    label = _code(atoms.label_atom[rows], atoms.label_atom_state[rows])
    return np.where(label >= 0, label, _code(atoms.auth_atom[rows], atoms.auth_atom_state[rows]))


# --- residue table --------------------------------------------------------------------------------------------------

_COLUMNS = {
    "kind": (np.uint8, 0),
    "asym": (np.int64, -1),
    "entity": (np.int64, -1),
    "instance": (np.int64, -1),
    "number": (np.int64, -1),
    "number_state": (np.uint8, _UNKNOWN),
    "component": (np.int64, -1),
    "author_component": (np.int64, -1),
    "author_chain": (np.int64, -1),
    "author_chain_state": (np.uint8, _UNKNOWN),
    "author_number": (np.int64, 0),
    "author_state": (np.uint8, _UNKNOWN),
    "author_text": (np.int64, -1),
    "insertion": (np.int64, -1),
    "declared": (np.bool_, False),
    "declaration": (np.int64, -1),
    "addressed": (np.bool_, False),  # the declaration states the author address (a scheme row, REMARK 465)
    "within": (np.int64, 0),
    "within_row": (np.int64, 0),
    "grouped": (np.bool_, False),  # the residue shares its sequence position with other candidates
}


class _Draft:
    r"""Residues before ordering: blocks of columns, concatenated once."""

    def __init__(self) -> None:
        self.blocks: list[dict[str, NDArray]] = []
        self.size = 0

    def add(self, count: int, **columns: object) -> None:
        self.blocks.append(
            {
                name: np.broadcast_to(np.asarray(columns.get(name, default), dtype=dtype), (count,))
                for name, (dtype, default) in _COLUMNS.items()
            }
        )
        self.size += count

    def columns(self) -> dict[str, NDArray]:
        return {
            name: np.concatenate([block[name] for block in self.blocks]) if self.blocks else np.zeros(0, dtype)
            for name, (dtype, _) in _COLUMNS.items()
        }


@dataclass
class _Binding:
    r"""What a format's half hands the shared tail."""

    residues: dict[str, NDArray]
    rows: NDArray[np.int64]  # the selected model's atom rows
    index_of_row: NDArray[np.int64]  # per selected row its residue
    symbols: tuple[str, ...]  # per instance
    instance_ref: NDArray[np.int64]
    instance_asym: NDArray[np.int64]
    instance_entity: NDArray[np.int64]
    log: list[tuple]  # records; an OCCURRENCE subject is a residue index until the tail orders them
    others: tuple[list[tuple], SourceRefs] | None = None  # the other models' records, sources into their own refs


def residue_table(tables: SourceTables, model: int) -> ResidueTable:
    r"""
    The one residue table of ``model``: every declared residue (``observed`` False when no row binds to it), the
    binding of every row of the model, and the instances. A source without a block is a PDB source.

    Raises:
        ContradictionError: ``residue_unbound``, ``label_auth_conflict``, ``asym_two_chains``, ``duplicate_atom``,
            ``asym_entity_mismatch`` (two statements of an instance's entity — its ``_struct_asym`` row, its scheme
            rows, its atom rows — disagree; the one entity check) or ``scheme_mismatch`` in the selected model. Another
            model's atom row stating another entity than its instance's is recorded as ``entry_level_violation`` (its
            model, check ``asym_entity_mismatch`` and asym; the row and the statement of the instance's entity as
            sources), as every check is for the other models.
        InsufficientInformationError: ``nonpolymer_address_ambiguous``; ``molecule_absent`` for a source with no atom
            row and no declared instance.
    """

    rows = np.flatnonzero(tables.atoms.model == model)
    binding = _pdb(tables, rows, model) if tables.refs.block is None else _mmcif(tables, rows)
    if not binding.symbols:
        fail("molecule_absent")
    return _finish(tables, binding)


def _scheme_block(draft: _Draft, schemes, picked: NDArray[np.int64], kind: int, instance: int, asym: int) -> None:
    sequenced = kind in (_POLYMER, _BRANCHED)
    number = schemes.number[picked] if sequenced else -1
    draft.add(
        len(picked),
        kind=kind,
        asym=asym,
        instance=instance,
        number=number,
        number_state=_KNOWN_DECLARED if sequenced else _INAPPLICABLE,
        component=schemes.mon_id[picked],
        author_chain=schemes.pdb_strand_id[picked],
        author_chain_state=schemes.pdb_strand_state[picked],
        author_number=schemes.pdb_seq_num[picked],
        author_state=schemes.pdb_seq_state[picked],
        author_text=schemes.pdb_seq_text[picked],
        insertion=_code(schemes.pdb_ins_code[picked], schemes.pdb_ins_state[picked]),
        declared=True,
        declaration=schemes.ref[picked],
        addressed=True,
        within=number if sequenced else np.arange(len(picked)),
        within_row=np.arange(len(picked)),
        grouped=sequenced,
    )


def _sequence_block(draft: _Draft, tables: SourceTables, picked, scheme, instance: int, asym: int) -> None:
    r"""The candidates ``_entity_poly_seq`` declares, with the author address of the scheme row naming each."""

    sequences, schemes = tables.sequences, tables.schemes
    count = len(picked)
    address = {
        "author_chain": np.full(count, -1, np.int64),
        "author_chain_state": np.full(count, _UNKNOWN, np.uint8),
        "author_number": np.zeros(count, np.int64),
        "author_state": np.full(count, _UNKNOWN, np.uint8),
        "author_text": np.full(count, -1, np.int64),
        "insertion": np.full(count, -1, np.int64),
        "declaration": sequences.ref[picked].astype(np.int64),
        "addressed": np.zeros(count, bool),
    }
    if len(scheme):
        keys = _pack(
            np.concatenate([sequences.num[picked], schemes.number[scheme]]),
            np.concatenate([sequences.mon_id[picked], schemes.mon_id[scheme]]),
        )
        target = _lookup(keys[:count], keys[count:])
        if (target < 0).any():
            fail("scheme_mismatch", *_refs(tables.refs, schemes.ref[scheme[target < 0]]))
        for name, values in (
            ("author_chain", schemes.pdb_strand_id),
            ("author_chain_state", schemes.pdb_strand_state),
            ("author_number", schemes.pdb_seq_num),
            ("author_state", schemes.pdb_seq_state),
            ("author_text", schemes.pdb_seq_text),
            ("declaration", schemes.ref),
        ):
            address[name][target] = values[scheme]
        address["insertion"][target] = _code(schemes.pdb_ins_code[scheme], schemes.pdb_ins_state[scheme])
        address["addressed"][target] = True
    draft.add(
        count,
        kind=_POLYMER,
        asym=asym,
        instance=instance,
        number=sequences.num[picked],
        number_state=_KNOWN_DECLARED,
        component=sequences.mon_id[picked],
        declared=True,
        within=sequences.num[picked],
        within_row=np.arange(count),
        grouped=True,
        **address,
    )


def _mmcif(tables: SourceTables, rows: NDArray[np.int64]) -> _Binding:
    atoms, refs = tables.atoms, tables.refs
    block = refs.block
    asym = _code(atoms.label_asym[rows], atoms.label_asym_state[rows])
    if (asym < 0).any():
        fail("residue_unbound", _row_ref(atoms, int(rows[np.argmax(asym < 0)]), block))

    # instances: _struct_asym order, then the asyms only coordinates state, in first-row order; an instance's entity
    # merges every statement of it, its _struct_asym row's, its scheme rows' and its atom rows': two different are
    # asym_entity_mismatch, otherwise the stated one (none known where nothing states it)
    order = np.argsort(tables.asyms.order, kind="stable")
    _, first = _first_groups(asym)
    extra = first[~np.isin(asym[first], tables.asyms.id)]
    inst_asym = np.concatenate([tables.asyms.id[order], asym[extra]]).astype(np.int64)
    inst_ref = np.concatenate([tables.asyms.ref[order], np.full(len(extra), -1)]).astype(np.int64)
    instance_of_row = _lookup(inst_asym, asym)
    schemes = tables.schemes
    instance_of_scheme = _lookup(inst_asym, schemes.asym)
    named = np.flatnonzero(instance_of_scheme >= 0)
    declared_entity = np.concatenate([tables.asyms.entity[order], np.full(len(extra), -1), schemes.entity[named]])
    owner = np.concatenate([np.arange(len(inst_asym)), instance_of_scheme[named], instance_of_row])
    told = np.concatenate(
        [
            np.where(declared_entity >= 0, declared_entity, NO_CODE),  # a code column: -1 states nothing
            stated_identity(atoms.label_entity[rows], atoms.label_entity_state[rows]),
        ]
    ).astype(np.int64)
    statements = np.concatenate([inst_ref, schemes.ref[named]])

    def stated_at(k: int) -> SourceRef:
        if k < len(statements):
            return refs.ref(int(statements[k]))
        return _row_ref(atoms, int(rows[k - len(statements)]), block)

    by_owner = np.argsort(owner, kind="stable")
    pair = _disagreeing(owner, told, by_owner)
    if pair is not None:
        fail("asym_entity_mismatch", *(stated_at(k) for k in pair))
    _, picked = _told_first(owner, told, by_owner)  # every instance owns its first item
    inst_entity = np.maximum(told[picked], -1)  # nothing stated: none known (-1)
    others = _other_models(tables, rows, inst_asym, told[picked], [stated_at(int(k)) for k in picked])

    entities = tables.entities
    kind_of_entity = dict(zip(entities.id.tolist(), np.where(known(entities.type_state), entities.type, 0).tolist()))
    scheme_of: dict[tuple[int, int], list[int]] = {}
    for index, key in enumerate(zip(schemes.kind.tolist(), schemes.asym.tolist())):
        scheme_of.setdefault(key, []).append(index)
    sequence_of = {
        entity: picked[np.argsort(tables.sequences.num[picked], kind="stable")]
        for entity, picked in _split(tables.sequences.entity).items()
    }
    observed_instances = set(np.unique(instance_of_row).tolist())

    draft, log = _Draft(), []
    kind = np.array([kind_of_entity.get(e, 0) for e in inst_entity.tolist()], dtype=np.uint8)
    mode = np.where(kind == _POLYMER, _SEQ, np.where(np.isin(kind, _ONE_MOLECULE), _SINGLE, _ADDRESS))
    from_atoms = np.zeros(len(inst_asym), dtype=bool)
    for i, (code, entity, k) in enumerate(zip(inst_asym.tolist(), inst_entity.tolist(), kind.tolist())):
        scheme_kind = k if k in (_POLYMER, _BRANCHED) else EntityType.NON_POLYMER  # one nonpoly scheme
        scheme = np.array(scheme_of.get((scheme_kind, code), ()), dtype=np.int64)
        if k == _POLYMER and entity in sequence_of:
            _sequence_block(draft, tables, sequence_of[entity], scheme, i, code)
        elif k in _ONE_MOLECULE and len(scheme) > 1:
            fail("scheme_mismatch", *_refs(refs, schemes.ref[scheme]))
        elif k and len(scheme):
            _scheme_block(draft, schemes, scheme, k, i, code)
            mode[i] = _SEQ if k == _POLYMER else _ADDRESS
        else:
            from_atoms[i] = True
            if k and i in observed_instances:
                log.append(("scheme_absent", SubjectKind.INSTANCE, i, (inst_ref[i],), ()))

    declared = draft.columns()
    count = len(declared["kind"])
    declared_mode, row_mode = mode[declared["instance"]], mode[instance_of_row]
    comp = _code(atoms.label_comp[rows], atoms.label_comp_state[rows])
    insertion = _code(atoms.insertion[rows], atoms.insertion_state[rows])
    # the second key: the sequence position (SEQ) or the author number token (ADDRESS); an unknown one matches nothing
    declared_token = _token(declared["author_text"], declared["author_state"])
    row_token = _token(atoms.auth_seq_text[rows], atoms.auth_seq_state[rows])
    second = np.concatenate(
        [
            np.select([declared_mode == _SEQ, declared_mode == _ADDRESS], [declared["number"], declared_token], 0),
            np.select([row_mode == _SEQ, row_mode == _ADDRESS], [atoms.label_seq[rows], row_token], 0),
        ]
    )
    valid = np.concatenate(
        [
            (declared_mode != _ADDRESS) | (declared_token != NO_CODE),
            np.select(
                [row_mode == _SEQ, row_mode == _ADDRESS],
                [known(atoms.label_seq_state[rows]), row_token != NO_CODE],
                True,
            ),
        ]
    )
    floor = _floor(second, valid)
    side = np.concatenate([np.full(count, 2), np.ones(len(rows), np.int64)])
    keys = _pack(
        np.concatenate([declared["instance"], instance_of_row]),
        np.where(valid, second, floor - side),
        np.concatenate(
            [np.where(declared_mode == _ADDRESS, declared["insertion"], 0), insertion * (row_mode == _ADDRESS)]
        ),
        np.concatenate(
            [
                np.where(declared_mode == _SINGLE, 0, declared["component"]),
                np.where(row_mode == _SINGLE, 0, np.where(comp >= 0, comp, NO_CODE)),
            ]
        ),
    )
    declared_keys, row_keys = keys[:count], keys[count:]
    unique, counts = np.unique(declared_keys, return_counts=True)
    if (counts > 1).any():
        fail("scheme_mismatch", *_refs(refs, declared["declaration"][declared_keys == unique[np.argmax(counts > 1)]]))
    index_of_row = _lookup(declared_keys, row_keys)
    unbound = index_of_row < 0
    if (unbound & ~from_atoms[instance_of_row]).any():
        fail("residue_unbound", _row_ref(atoms, int(rows[np.argmax(unbound & ~from_atoms[instance_of_row])]), block))
    if unbound.any():
        members = np.flatnonzero(unbound)
        _from_atoms(draft, tables, rows, members, row_keys, instance_of_row, kind, mode, index_of_row)
    residues = draft.columns()
    residues["entity"] = inst_entity[residues["instance"]]  # every residue is of its instance's entity
    _addresses(tables, rows, residues, index_of_row)
    symbols = tuple(atoms.symbols["chain"][code] for code in inst_asym.tolist())
    return _Binding(residues, rows, index_of_row, symbols, inst_ref, inst_asym, inst_entity, log, others)


def _other_models(
    tables: SourceTables, rows: NDArray[np.int64], asym: NDArray[np.int64], told: NDArray[np.int64], at: list[SourceRef]
) -> tuple[list[tuple], SourceRefs]:
    r"""The entity check over the atom rows of every model but the selected one (``rows``): a row stating another
    entity than its instance's (``told`` per instance, stated where ``at`` names) is an ``entry_level_violation`` of
    check ``asym_entity_mismatch``, naming the row's model and asym. Returns the records and the references their
    sources index."""

    atoms, builder, log = tables.atoms, TableBuilder(tables.refs.block), []
    others = np.setdiff1d(np.arange(len(atoms.row)), rows)
    instance = _lookup(asym, _code(atoms.label_asym[others], atoms.label_asym_state[others]))
    entity = stated_identity(atoms.label_entity[others], atoms.label_entity_state[others])
    expected = np.where(instance >= 0, told[instance], NO_CODE)
    for k in np.flatnonzero((entity != NO_CODE) & (expected != NO_CODE) & (entity != expected)).tolist():
        row, statement = int(others[k]), at[int(instance[k])]
        sources = [builder.ref("atom_site", int(atoms.row[row]), "label_entity_id")]
        sources.append(builder.ref(statement.category, statement.row, statement.item))
        values = (int(atoms.model[row]), "asym_entity_mismatch", atoms.symbols["chain"][asym[instance[k]]])
        log.append(("entry_level_violation", SubjectKind.NONE, -1, sources, values))
    return log, builder.source_refs()


def _from_atoms(
    draft: _Draft, tables: SourceTables, rows, members, row_keys, instance_of_row, kind, mode, index_of_row
) -> None:
    r"""The residues of instances no declaration enumerates: one per distinct row key, in first-row order."""

    atoms, block = tables.atoms, tables.refs.block
    member_rows = rows[members]
    instance = instance_of_row[members]
    if ((mode[instance] == _SEQ) & ~known(atoms.label_seq_state[member_rows])).any():
        bad = member_rows[np.argmax((mode[instance] == _SEQ) & ~known(atoms.label_seq_state[member_rows]))]
        fail("residue_unbound", _row_ref(atoms, int(bad), block))
    token = _token(atoms.auth_seq_text[member_rows], atoms.auth_seq_state[member_rows])
    nameless = (kind[instance] == _WATER) & (token < 0)
    if nameless.any():
        fail("nonpolymer_address_ambiguous", _row_ref(atoms, int(member_rows[np.argmax(nameless)]), block))
    single = np.flatnonzero(mode[instance] == _SINGLE)
    for value in (  # an instance that holds one molecule states one address
        token,
        _code(atoms.insertion[member_rows], atoms.insertion_state[member_rows]),
        stated_identity(atoms.label_comp[member_rows], atoms.label_comp_state[member_rows]),
    ):
        pair = _disagreeing(instance[single], value[single])
        if pair is not None:
            two = member_rows[single[list(pair)]]
            fail("nonpolymer_address_ambiguous", candidates=tuple(_row_ref(atoms, int(row), block) for row in two))
    group, first = _first_groups(row_keys[members])
    heads, head_instance = member_rows[first], instance[first]
    number = np.where(known(atoms.label_seq_state[heads]), atoms.label_seq[heads], -1)
    sequenced = mode[head_instance] == _SEQ
    index_of_row[members] = draft.size + group
    draft.add(
        len(first),
        kind=kind[head_instance],
        asym=_code(atoms.label_asym[heads], atoms.label_asym_state[heads]),
        instance=head_instance,
        number=number,
        number_state=atoms.label_seq_state[heads],
        component=_code(atoms.label_comp[heads], atoms.label_comp_state[heads]),
        within=np.where(sequenced, number, members[first]),
        within_row=members[first],
        grouped=sequenced,
    )


def _addresses(tables: SourceTables, rows, residues: dict[str, NDArray], index_of_row) -> None:
    r"""
    The author address of every mmCIF residue, one merge per part (chain, number, insertion, author component) over
    the value its declaration states (a scheme row; it states no component) and the value each of its rows states:
    two that state different values are ``label_auth_conflict`` (a part left unknown agrees with any), and the residue
    takes the first stated one, the declaration's before its rows', else the first unknown one.
    """

    atoms, refs, block = tables.atoms, tables.refs, tables.refs.block
    declared = np.flatnonzero(residues["addressed"])
    owner = np.concatenate([declared, index_of_row])
    by_owner = np.argsort(owner, kind="stable")

    def joined(name: str, column: NDArray) -> NDArray:
        return np.concatenate([residues[name][declared], column[rows]])

    def where(k: int) -> SourceRef:
        if k < len(declared):
            return refs.ref(int(residues["declaration"][declared[k]]))
        return _row_ref(atoms, int(rows[k - len(declared)]), block)

    chain, chain_state = joined("author_chain", atoms.auth_asym), joined("author_chain_state", atoms.auth_asym_state)
    number, number_state = joined("author_number", atoms.auth_seq), joined("author_state", atoms.auth_seq_state)
    text = joined("author_text", atoms.auth_seq_text)
    insertion = joined("insertion", _code(atoms.insertion, atoms.insertion_state))
    comp = np.concatenate([np.full(len(declared), -1), atoms.auth_comp[rows]])
    comp_state = np.concatenate([np.full(len(declared), _UNKNOWN, np.uint8), atoms.auth_comp_state[rows]])
    for value, columns in (
        (stated_identity(chain, chain_state), {"author_chain": chain, "author_chain_state": chain_state}),
        (_token(text, number_state), {"author_number": number, "author_state": number_state, "author_text": text}),
        (insertion, {"insertion": insertion}),
        (stated_identity(comp, comp_state), {"author_component": _code(comp, comp_state)}),
    ):
        pair = _disagreeing(owner, value, by_owner)
        if pair is not None:
            fail("label_auth_conflict", *(where(k) for k in pair))
        targets, picked = _told_first(owner, value, by_owner)
        for name, values in columns.items():
            residues[name][targets] = values[picked]


def _pdb(tables: SourceTables, rows: NDArray[np.int64], model: int) -> _Binding:
    r"""
    PDB residues ``(chain, resSeq, iCode, resName)`` in first-row order, ``resSeq`` by its integer reading
    (``author_key``); the residues of one numbered ``(chain, resSeq, iCode)`` are the candidates of one site
    (microheterogeneity, PDB v3.3). A chain with SEQRES (``SequenceTable.entity`` is the chain) holds one polymer: the
    leading run of its sites (record order, before the first ``TER`` after its first site) that embeds in SEQRES,
    placed by the SEQRES correspondence; a chain without SEQRES holds the polymer of its ATOM sites. Every other
    residue is a heterogen of its own instance; the waters of a chain share one. The entity is the chain's ``COMPND``
    entity, -1 where ``COMPND`` names none. A blank chain is the stated none (INAPPLICABLE×DECLARED), its rows' and its
    SEQRES's alike.
    """

    atoms, block = tables.atoms, tables.refs.block
    names = atoms.symbols
    chain = _code(atoms.auth_asym[rows], atoms.auth_asym_state[rows])
    number = author_key(atoms.auth_seq_text[rows], atoms.auth_seq_state[rows], atoms.auth_seq[rows], pdb=True)
    insertion = _code(atoms.insertion[rows], atoms.insertion_state[rows])
    comp = _code(atoms.auth_comp[rows], atoms.auth_comp_state[rows])
    residue_of_row, first = _first_groups(_pack(chain, number, insertion, comp))
    heads, count = rows[first], len(first)
    standard = np.bincount(residue_of_row, weights=atoms.group[rows] == _ATOM, minlength=count) > 0
    res_chain, res_comp, res_insertion, res_number = chain[first], comp[first], insertion[first], number[first]
    numbered = res_number != NO_CODE
    site_of_residue, _ = _first_groups(
        np.where(numbered, _pack(res_chain, res_number, res_insertion), -1 - np.arange(count))
    )

    sequences = tables.sequences
    sequence_of = {
        c: picked[np.argsort(sequences.num[picked], kind="stable")] for c, picked in _split(sequences.entity).items()
    }
    entity_of_chain = dict(zip(tables.asyms.id.tolist(), tables.asyms.entity.tolist()))
    unobserved = tables.unobserved
    in_model = (unobserved.model == model) | (unobserved.model == 0)  # 0: the row states no model
    missing_of = _split(np.where(in_model, unobserved.chain, NO_CODE))
    mods = tables.modified  # MODRES states every address part: a blank chain is the stated none, seqNum is required
    modified = set(
        zip(
            stated_identity(mods.asym, mods.asym_state).tolist(),
            author_key(mods.seq_text, mods.seq_state, mods.seq, pdb=True).tolist(),
            mods.ins.tolist(),
            mods.comp.tolist(),
        )
    )
    residues_of_chain = _split(res_chain)
    chains = list(dict.fromkeys(res_chain.tolist()))
    chains += [c for c in dict.fromkeys(sequences.entity.tolist()) if c not in residues_of_chain]

    draft, instance_ref, instance_entity = _Draft(), [], []
    log: list[tuple] = []
    symbols: list[str] = []
    index_of_residue = np.full(count, -1, dtype=np.int64)
    polymer = np.zeros(count, dtype=bool)
    for c in chains:
        local = residues_of_chain.get(c, np.zeros(0, np.int64))
        site = site_of_residue[local]
        sites = [local[site == s] for s in dict.fromkeys(site.tolist())]
        sequence = sequence_of.get(c)
        instance = len(symbols)
        base = {
            "kind": _POLYMER,
            "entity": entity_of_chain.get(c, -1),
            "instance": instance,
            "author_chain": c,
            "author_chain_state": _KNOWN_DECLARED if c >= 0 else _BLANK,
            "grouped": True,
        }
        vacant_block = None
        if sequence is None:
            sites = [s for s in sites if standard[s].any()]
            if not sites:
                continue
            log.append(("seqres_absent", SubjectKind.INSTANCE, instance, (), ()))
            positions = np.arange(len(sites))
            numbers, number_state, declaration, chain_ref = -1, _UNKNOWN, -1, -1
        else:
            chain_ref = int(sequences.ref[sequence[0]])
            seq_comps = sequences.mon_id[sequence].tolist()
            candidates = [frozenset(res_comp[s].tolist()) for s in sites]
            # TER ends the chain's ATOM/HETATM list (PDB v3.3): a site after the chain's first TER is in no polymer
            starts = atoms.row[heads[[s[0] for s in sites]]] if sites else np.zeros(0, np.int64)
            ends = tables.terminators[tables.terminators > starts[0]] if sites else tables.terminators[:0]
            listed = int(np.searchsorted(starts, ends[0])) if len(ends) else len(sites)
            run = _leading_run(candidates[:listed], seq_comps)
            outside = [s[standard[s]] for s in sites[run:] if standard[s].any()]
            if outside:
                fail("residue_unbound", _row_ref(atoms, int(heads[outside[0][0]]), block))
            sites, candidates = sites[:run], candidates[:run]
            missing = missing_of.get(c, np.zeros(0, np.int64))
            observed_addresses = [
                (
                    (int(atoms.auth_seq[heads[s[0]]]), _insertion_text(names, res_insertion[s[0]]))
                    if numbered[s[0]]
                    else None
                )
                for s in sites
            ]
            missing_addresses = [
                (
                    (int(unobserved.seq[j]), _insertion_text(names, unobserved.ins[j])),
                    frozenset((int(unobserved.comp[j]),)),
                )
                for j in missing.tolist()
            ]
            positions, fixed, vacant, embeddings = _correspondence(
                candidates, observed_addresses, seq_comps, missing_addresses
            )
            if embeddings != 1:
                log.append(
                    ("seqres_correspondence_ambiguous", SubjectKind.INSTANCE, instance, (chain_ref,), (embeddings,))
                )
            numbers = np.where(fixed, sequences.num[sequence][positions], -1)
            number_state = np.where(fixed, _KNOWN_DERIVED, _UNKNOWN)
            declaration = sequences.ref[sequence][positions]
            if vacant:
                p = np.array([p for p, _, _ in vacant], dtype=np.int64)
                j = np.array([missing[j] if j >= 0 else -1 for _, j, _ in vacant], dtype=np.int64)
                settled = np.array([f for _, _, f in vacant], dtype=bool)
                stated = j >= 0
                author_number, author_insertion = np.zeros(len(p), np.int64), np.full(len(p), -1, np.int64)
                author_text, author_component = np.full(len(p), -1, np.int64), np.full(len(p), -1, np.int64)
                author_number[stated], author_insertion[stated] = unobserved.seq[j[stated]], unobserved.ins[j[stated]]
                author_text[stated], author_component[stated] = (
                    unobserved.seq_text[j[stated]],
                    unobserved.comp[j[stated]],
                )
                vacant_block = {
                    "number": np.where(settled, sequences.num[sequence][p], -1),
                    "number_state": np.where(settled, _KNOWN_DERIVED, _UNKNOWN),
                    "component": sequences.mon_id[sequence][p],
                    "author_component": author_component,
                    "author_number": author_number,
                    "author_state": np.where(stated, _KNOWN_DECLARED, _UNKNOWN),
                    "author_text": author_text,
                    "insertion": author_insertion,
                    "declared": True,
                    "declaration": sequences.ref[sequence][p],
                    "addressed": stated,
                    "within": p,
                    "within_row": -1,
                }
        sizes = [len(s) for s in sites]
        members = np.concatenate(sites) if sites else np.zeros(0, np.int64)
        site_of = np.repeat(np.arange(len(sites)), sizes)
        if sequence is not None:
            # a HETATM residue in SEQRES that no MODRES names (one without a resSeq has no address)
            for k in np.flatnonzero(~standard[members]).tolist():
                m = members[k]
                address = (c, int(res_number[m]), int(res_insertion[m]), int(res_comp[m]))
                if not numbered[m] or address not in modified:
                    log.append(("group_vs_seqres", SubjectKind.OCCURRENCE, draft.size + k, (), ()))
        index_of_residue[members] = draft.size + np.arange(len(members))
        polymer[members] = True
        draft.add(
            len(members),
            **base,
            number=np.broadcast_to(numbers, len(sites))[site_of],
            number_state=np.broadcast_to(number_state, len(sites))[site_of],
            component=res_comp[members],
            author_component=res_comp[members],
            author_number=atoms.auth_seq[heads[members]],
            author_state=atoms.auth_seq_state[heads[members]],
            author_text=atoms.auth_seq_text[heads[members]],
            insertion=res_insertion[members],
            declared=sequence is not None,
            declaration=np.broadcast_to(declaration, len(sites))[site_of],
            addressed=True,
            within=positions[site_of],
            within_row=members,
        )
        if vacant_block is not None:
            draft.add(len(vacant_block["within"]), **base, **vacant_block)
        symbols.append(_chain_text(names, c))
        instance_ref.append(chain_ref)
        instance_entity.append(base["entity"])

    hetero = np.flatnonzero(~polymer)
    water_of_chain: dict[int, int] = {}
    instance = np.empty(len(hetero), dtype=np.int64)
    water = np.zeros(len(hetero), dtype=bool)
    for k, r in enumerate(hetero.tolist()):
        name = names["comp"][res_comp[r]] if res_comp[r] >= 0 else ""
        water[k] = name in _PDB_WATER
        if water[k] and res_chain[r] in water_of_chain:
            instance[k] = water_of_chain[res_chain[r]]
            continue
        instance[k] = len(symbols)
        if water[k]:
            water_of_chain[int(res_chain[r])] = instance[k]
            symbols.append(f"{_chain_text(names, res_chain[r])}_water")
        else:
            number = str(int(atoms.auth_seq[heads[r]])) if numbered[r] else "?"
            symbols.append(
                f"{_chain_text(names, res_chain[r])}_{name}_{number}{_insertion_text(names, res_insertion[r])}"
            )
        instance_ref.append(-1)
        instance_entity.append(-1)
    index_of_residue[hetero] = draft.size + np.arange(len(hetero))
    draft.add(
        len(hetero),
        kind=np.where(water, _WATER, 0),
        instance=instance,
        number_state=np.where(water, _INAPPLICABLE, _UNKNOWN),
        component=res_comp[hetero],
        author_component=res_comp[hetero],
        author_chain=res_chain[hetero],
        author_chain_state=atoms.auth_asym_state[heads[hetero]],
        author_number=atoms.auth_seq[heads[hetero]],
        author_state=atoms.auth_seq_state[heads[hetero]],
        author_text=atoms.auth_seq_text[heads[hetero]],
        insertion=res_insertion[hetero],
        addressed=True,
        within=hetero,
        within_row=hetero,
    )
    return _Binding(
        draft.columns(),
        rows,
        index_of_residue[residue_of_row],
        tuple(symbols),
        np.array(instance_ref, dtype=np.int64),
        np.full(len(symbols), -1, dtype=np.int64),
        np.array(instance_entity, dtype=np.int64),
        log,
    )


def _chain_text(names, code: int) -> str:
    return names["chain"][code] if code >= 0 else ""


def _insertion_text(names, code: int) -> str:
    return names["ins"][code] if code >= 0 else ""


_Site = frozenset[int]  # the candidate components of one observed site


def _leading_run(sites: list[_Site], sequence: list[int]) -> int:
    r"""The length of the longest leading run of a chain's sites (record order) that embeds in order in SEQRES."""

    at: dict[int, list[int]] = {}
    for position, component in enumerate(sequence):
        at.setdefault(component, []).append(position)
    next_free = 0
    for run, site in enumerate(sites):
        found = []
        for component in site:
            positions = at.get(component, [])
            k = bisect_left(positions, next_free)
            if k < len(positions):
                found.append(positions[k])
        if not found:
            return run
        next_free = min(found) + 1
    return len(sites)


def _earliest(sites: list[_Site], sequence: list[int]) -> list[int]:
    position, earliest = 0, []
    for site in sites:
        while sequence[position] not in site:
            position += 1
        earliest.append(position)
        position += 1
    return earliest


def _correspondence(
    sites: list[_Site], addresses: list[tuple | None], sequence: list[int], missing: list[tuple[tuple, _Site]]
) -> tuple[NDArray[np.int64], NDArray[np.bool_], list[tuple[int, int, bool]], int]:
    r"""
    The SEQRES correspondence of a chain's observed polymer sites (which embed in order).

    REMARK 465 places unobserved residues only when the observed and REMARK 465 residues, ordered by author address,
    reproduce SEQRES exactly. Otherwise the record order alone: a site is placed where every order-preserving embedding
    puts it; elsewhere the earliest embedding lays the chain out, so the components in layout order are SEQRES's.

    Returns:
        Per observed site its SEQRES position and whether every embedding puts it there; per other position its index
        of a REMARK 465 residue (-1 none) and whether every embedding leaves it unobserved; the number of embeddings.
    """

    if missing and None not in addresses:
        merged = sorted(
            [(a, 0, i, c) for i, (a, c) in enumerate(zip(addresses, sites))]
            + [(a, 1, j, c) for j, (a, c) in enumerate(missing)]
        )
        if len({a for a, *_ in merged}) == len(merged) == len(sequence) and all(
            component in c for component, (*_, c) in zip(sequence, merged)
        ):
            place = np.zeros(len(sites), dtype=np.int64)
            vacant = []
            for position, (_, kind, index, _) in enumerate(merged):
                if kind:
                    vacant.append((position, index, True))
                else:
                    place[index] = position
            return place, np.ones(len(sites), dtype=bool), vacant, 1
    earliest = _earliest(sites, sequence)
    latest = [len(sequence) - 1 - p for p in _earliest(sites[::-1], sequence[::-1])[::-1]]
    fixed = np.array([e == last for e, last in zip(earliest, latest)], dtype=bool)
    # a position some site takes under some embedding: its component, between the site's earliest and latest
    reachable = {p for site, e, last in zip(sites, earliest, latest) for p in range(e, last + 1) if sequence[p] in site}
    taken = set(earliest)
    vacant = [(p, -1, p not in reachable) for p in range(len(sequence)) if p not in taken]
    embeddings = 1 if fixed.all() else _embeddings(sites, sequence)
    return np.array(earliest, dtype=np.int64), fixed, vacant, embeddings


def _embeddings(sites: list[_Site], sequence: list[int]) -> int:
    r"""The number of order-preserving embeddings of ``sites`` in ``sequence`` (clipped to the int64 range)."""

    sequence = np.array(sequence)
    ways = np.ones(len(sequence) + 1)
    for site in sites:
        ways = np.concatenate([[0.0], np.cumsum(np.where(np.isin(sequence, list(site)), ways[:-1], 0.0))])
    return int(min(ways[-1], 2.0**62))


def _finish(tables: SourceTables, binding: _Binding) -> ResidueTable:
    r"""
    Order the residues into ordinals, bind rows and check what every format shares: one author chain per instance,
    the one its residues state (``asym_two_chains``); each atom stated once in a realisation (``duplicate_atom``: two
    rows of one name and altloc, or an altloc-less row beside another row of its name, which every realisation holds).
    """

    atoms, refs, block = tables.atoms, tables.refs, tables.refs.block
    d = binding.residues
    order = np.lexsort((d["within_row"], d["within"], d["instance"]))
    count = len(order)
    ordinal_of_index = np.empty(count, dtype=np.int64)
    ordinal_of_index[order] = np.arange(count)
    d = {name: values[order] for name, values in d.items()}
    ordinals = ordinal_of_index[binding.index_of_row]
    ordinal_of_row = np.full(len(atoms.row), -1, dtype=np.int32)
    ordinal_of_row[binding.rows] = ordinals
    name_of_row = np.full(len(atoms.row), -1, dtype=np.int32)
    name_of_row[binding.rows] = _atom_name(atoms, binding.rows)
    rows = binding.rows[np.argsort(ordinals, kind="stable")]
    counts = np.bincount(ordinals, minlength=count)
    row_ptr = _ptr(counts)
    observed = counts > 0
    owner = np.repeat(np.arange(count), counts)

    def where(o: int) -> SourceRef:  # the statement of a residue: its first row, else its declaration
        return _row_ref(atoms, int(rows[row_ptr[o]]), block) if observed[o] else refs.ref(int(d["declaration"][o]))

    grouped = np.flatnonzero(d["grouped"])
    group_key = -1 - np.arange(count, dtype=np.int64)
    if len(grouped):
        group_key[grouped] = _pack(d["instance"][grouped], d["within"][grouped])
    position, _ = _first_groups(group_key)

    told = stated_identity(d["author_chain"], d["author_chain_state"])
    pair = _disagreeing(d["instance"], told)
    if pair is not None:
        fail("asym_two_chains", *(where(o) for o in pair))
    instance_chain = np.full(len(binding.symbols), -1, dtype=np.int32)
    instance_chain_state = np.full(len(binding.symbols), _UNKNOWN, dtype=np.uint8)
    holders, picked = _told_first(d["instance"], told)
    instance_chain[holders], instance_chain_state[holders] = d["author_chain"][picked], d["author_chain_state"][picked]

    name, alt = name_of_row[rows], _code(atoms.label_alt[rows], atoms.label_alt_state[rows])
    _, inverse, counted = np.unique(_pack(owner, name, alt), return_inverse=True, return_counts=True)
    atom, _ = _first_groups(_pack(owner, name))
    # altloc-less: the stated none; an unknown altloc (``?``) is no statement, chemical_state's ``altloc_unknown``
    altless = resolution(atoms.label_alt_state[rows]) == Resolution.INAPPLICABLE
    beside = (np.bincount(atom, weights=altless)[atom] > 0) & (np.bincount(atom)[atom] > 1)
    twice = (counted[inverse.ravel()] > 1) | beside
    if twice.any():
        fail("duplicate_atom", *(_row_ref(atoms, int(row), block) for row in rows[twice & (atom == atom[twice][0])]))
    unnamed = np.flatnonzero(d["component"] < 0)  # a residue is an occurrence of a named component
    if len(unnamed):  # an unstated label_comp_id or scheme mon_id names none
        fail("residue_unbound", where(int(unnamed[0])))

    log = [
        (code, kind, int(ordinal_of_index[subject]) if kind == SubjectKind.OCCURRENCE else subject, sources, payload)
        for code, kind, subject, sources, payload in binding.log
    ]
    author_component = d["author_component"]
    differs = observed & (author_component >= 0) & (author_component != d["component"])
    log += [
        ("author_component_differs", SubjectKind.OCCURRENCE, o, (d["declaration"][o],), ())
        for o in np.flatnonzero(differs)
    ]
    both = known(atoms.label_atom_state[rows]) & known(atoms.auth_atom_state[rows])
    renamed = np.unique(owner[both & (atoms.label_atom[rows] != atoms.auth_atom[rows])])
    log += [("author_atom_differs", SubjectKind.OCCURRENCE, int(o), (d["declaration"][o],), ()) for o in renamed]
    if binding.others is not None and binding.others[0]:  # their references follow the source's, which keep theirs
        refs, (_, at) = SourceRefs.concat([refs, binding.others[1]])
        log += [(*row[:3], tuple(int(at[k]) for k in row[3]), row[4]) for row in binding.others[0]]
    seen = np.bincount(d["instance"], weights=observed, minlength=len(binding.symbols))
    for i in np.flatnonzero((seen == 0) & (binding.instance_ref >= 0)).tolist():  # a declared instance
        log.append(("instance_unobserved", SubjectKind.INSTANCE, i, (binding.instance_ref[i],), ()))

    return ResidueTable(
        kind=d["kind"].astype(np.uint8),
        asym=d["asym"].astype(np.int32),
        entity=d["entity"].astype(np.int32),
        instance=d["instance"].astype(np.int32),
        number=d["number"].astype(np.int32),
        number_state=d["number_state"].astype(np.uint8),
        position=position.astype(np.int32),
        component=d["component"].astype(np.int32),
        author_component=author_component.astype(np.int32),
        author_chain=d["author_chain"].astype(np.int32),
        author_chain_state=d["author_chain_state"].astype(np.uint8),
        author_number=d["author_number"].astype(np.int64),
        author_state=d["author_state"].astype(np.uint8),
        author_text=d["author_text"].astype(np.int32),
        author_key=author_key(d["author_text"], d["author_state"], d["author_number"], block is None),
        insertion=d["insertion"].astype(np.int32),
        declared=d["declared"].astype(bool),
        observed=observed,
        declaration=d["declaration"].astype(np.int64),
        row_ptr=row_ptr,
        rows=rows,
        ordinal_of_row=ordinal_of_row,
        name_of_row=name_of_row,
        instance_symbol=binding.symbols,
        instance_asym=binding.instance_asym.astype(np.int32),
        instance_entity=binding.instance_entity.astype(np.int32),
        instance_author_chain=instance_chain,
        instance_author_chain_state=instance_chain_state,
        records=Records.from_rows(log, refs),
    )


# --- PDB polymer types ------------------------------------------------------------------------------------------------

_NUCLEIC_TYPES = {
    frozenset({MonomerType.DNA}): "polydeoxyribonucleotide",
    frozenset({MonomerType.RNA}): "polyribonucleotide",
    frozenset({MonomerType.DNA, MonomerType.RNA}): "polydeoxyribonucleotide/polyribonucleotide hybrid",
}


def pdb_polymer_types(
    tables: SourceTables, residues: ResidueTable, definitions: Definitions
) -> tuple[NDArray[np.int32], NDArray[np.uint8], Records]:
    r"""
    Per instance of ``residues`` the ``_entity_poly.type`` its chain's SEQRES components derive: an
    index into ``POLYMER_TYPES`` with state KNOWN×DERIVED, or -1, UNKNOWN×DERIVED and record ``polymer_type_unstated``
    when a component's ``_chem_comp.type`` is not KNOWN, or the family-bearing components mix families or hands, or
    none bears a family or a hand. A polymer chain without SEQRES derives nothing (-1, UNKNOWN×NONE; ``seqres_absent``
    records it); every other instance is -1, INAPPLICABLE×DEFINED.
    """

    sequences, names = tables.sequences, tables.atoms.symbols["comp"]
    codes = np.full(len(residues.instance_symbol), -1, dtype=np.int32)
    states = np.full(len(residues.instance_symbol), _INAPPLICABLE, dtype=np.uint8)
    log = []
    for i in np.unique(residues.instance[residues.kind == _POLYMER]).tolist():
        c = int(residues.instance_author_chain[i])  # the chain's SEQRES (SequenceTable.entity, -1 a blank chain)
        states[i] = _UNKNOWN
        picked = np.flatnonzero(sequences.entity == c)
        if not len(picked):
            continue
        derived = derived_polymer_type({names[m] for m in sequences.mon_id[picked].tolist()}, definitions.by_id)
        if derived is None:
            states[i] = _UNKNOWN_DERIVED
            log.append(("polymer_type_unstated", SubjectKind.INSTANCE, i, (sequences.ref[picked[0]],), ()))
        else:
            codes[i], states[i] = POLYMER_TYPES.index(derived), _KNOWN_DERIVED
    return codes, states, Records.from_rows(log, tables.refs)


def derived_polymer_type(component_ids: set[str], definitions: Mapping[str, ChemicalComponent]) -> str | None:
    r"""The ``_entity_poly.type`` a SEQRES of the components ``component_ids`` derives (``pdb_polymer_types``), None
    when none is determined; a component ``definitions`` does not hold determines none."""
    families, monomers, hands = set(), set(), set()
    for cid in component_ids:
        definition = definitions.get(cid)
        if definition is None or not known(np.uint8(definition.chem_comp_type_state)):
            return None
        linking = cast(str, definition.chem_comp_type).upper()
        if linking not in FAMILY_TABLE:
            continue
        family, monomer, _ = FAMILY_TABLE[linking]
        families.add(family)
        monomers.add(monomer)
        if family == LinkingFamily.PEPTIDE and linking[:2] in ("L-", "D-"):
            hands.add(linking[0])
    if families == {LinkingFamily.PEPTIDE} and len(hands) == 1:
        return f"polypeptide({hands.pop()})"
    if families == {LinkingFamily.NUCLEIC}:
        return _NUCLEIC_TYPES[frozenset(monomers)]
    return None


# --- chemical state ---------------------------------------------------------------------------------------------------


def chemical_state(residues: ResidueTable, atoms: AtomTable, *, altloc: AltlocPolicy) -> ChemicalState:
    r"""
    The one chemical state: per sequence position one complete realisation — a candidate component with one
    altloc label, the altloc-less rows belonging to every realisation — by the highest mean occupancy of its rows (a
    labelled candidate's lettered rows, else all its rows); a tie goes to the lower label (recorded). A label policy
    selects that label where a position has it and the occupancy rule elsewhere.

    Raises:
        InvalidRequestError: ``altloc_policy_invalid``: the label names no altloc of the source.
        InsufficientInformationError: ``altloc_unknown`` (``label_alt_id ?`` beside lettered rows of one position),
            ``altloc_occupancy_unknown`` (the occupancy rule meets an UNKNOWN occupancy),
            ``microheterogeneity_unobserved`` (several candidates, none observed).
    """

    labels = tuple(atoms.symbols["alt"])
    label = -1
    if altloc != "highest_occupancy":
        if altloc not in labels:
            fail("altloc_policy_invalid", RequestRef("altloc", altloc))
        label = labels.index(altloc)
    refs, block = residues.records.refs, residues.records.refs.block
    count = len(residues.kind)
    rows = residues.rows
    owner = np.repeat(np.arange(count), np.diff(residues.row_ptr))
    alt = _code(atoms.label_alt[rows], atoms.label_alt_state[rows])
    positions = int(residues.position.max()) + 1 if count else 0
    row_position = residues.position[owner].astype(np.int64)

    asked = resolution(atoms.label_alt_state[rows]) == Resolution.UNKNOWN
    lettered_at = np.bincount(row_position, weights=alt >= 0, minlength=positions) > 0
    if (asked & lettered_at[row_position]).any():
        fail("altloc_unknown", _row_ref(atoms, int(rows[np.argmax(asked & lettered_at[row_position])]), block))

    # realisations: (ordinal, label) over the lettered rows; (ordinal, none) for a candidate without labels
    lettered = np.bincount(owner, weights=alt >= 0, minlength=count) > 0
    used = np.flatnonzero((alt >= 0) | ~lettered[owner])
    group, first = _first_groups(_pack(owner[used], alt[used] + 1)) if len(used) else (used, used)
    g_ordinal, g_alt = owner[used][first], alt[used][first]
    g_position = residues.position[g_ordinal].astype(np.int64)
    stated = known(atoms.occupancy_state[rows[used]])
    total = np.bincount(group, weights=np.where(stated, atoms.occupancy[rows[used]], 0.0), minlength=len(first))
    mean = total / np.maximum(np.bincount(group, minlength=len(first)), 1)
    blind = np.bincount(group, weights=~stated, minlength=len(first)) > 0

    competing = np.bincount(g_position, minlength=positions)
    candidates = np.bincount(residues.position, minlength=positions)
    if ((competing == 0) & (candidates > 1)).any():
        p = int(np.argmax((competing == 0) & (candidates > 1)))
        fail("microheterogeneity_unobserved", candidates=_refs(refs, residues.declaration[residues.position == p]))

    by_label = np.zeros(positions, dtype=bool)
    if label >= 0:
        by_label[g_position[g_alt == label]] = True
    eligible = ~by_label[g_position] | (g_alt == label)
    contested = np.bincount(g_position[eligible], minlength=positions) > 1
    if (eligible & contested[g_position] & blind).any():
        g = int(np.argmax(eligible & contested[g_position] & blind))
        unknown_rows = rows[used][(group == g) & ~stated]
        fail("altloc_occupancy_unknown", _row_ref(atoms, int(unknown_rows[0]), block))

    label_rank = np.argsort(np.argsort(np.array(labels, dtype=object))) if labels else np.zeros(0, np.int64)
    rank = np.where(g_alt >= 0, label_rank[g_alt.clip(0)] + 1 if labels else 0, 0)
    order = np.lexsort((g_ordinal, rank, -mean, ~eligible, g_position))
    s_position = g_position[order]
    head = np.ones(len(order), dtype=bool)
    head[1:] = s_position[1:] != s_position[:-1]
    winners = order[head]
    tie = np.zeros(len(order), dtype=bool)
    tie[:-1] = head[:-1] & ~head[1:] & eligible[order][1:] & (mean[order][1:] == mean[order][:-1])
    tie &= ~by_label[s_position]

    selected = np.ones(count, dtype=bool)
    selected[competing[residues.position] > 0] = False
    selected[g_ordinal[winners]] = True
    chosen = np.full(count, -1, dtype=np.int32)
    chosen[g_ordinal[winners]] = g_alt[winners]
    row_selected = np.zeros(len(atoms.row), dtype=bool)
    row_selected[rows] = selected[owner] & ((alt < 0) | (alt == chosen[owner]))

    contested_heads = head & (competing[s_position] > 1)
    log = [
        ("altloc_tie_by_label", SubjectKind.OCCURRENCE, int(o), (residues.declaration[o],), ())
        for o in g_ordinal[order[contested_heads & tie]]
    ]
    return ChemicalState(selected, row_selected, chosen, Records.from_rows(log, refs))


# --- endpoint queries -------------------------------------------------------------------------------------------------

_QUERY = {
    "source": (np.uint8, 0),
    "label_asym": (np.int32, -1),
    "label_seq": (np.int64, 0),
    "label_comp": (np.int32, -1),
    "label_atom": (np.int32, -1),
    "auth_asym": (np.int32, -1),
    "auth_key": (np.int64, NO_CODE),
    "auth_ins": (np.int32, -1),
    "auth_comp": (np.int32, -1),
    "auth_atom": (np.int32, -1),
    "altloc": (np.int32, -1),
    "scheme_num": (np.int32, -1),
    "row": (np.int64, -1),
    "ref": (np.int64, -1),
}


def _query_block(count: int, field_state: NDArray[np.uint8], **columns: NDArray) -> dict[str, NDArray]:
    block = {
        name: np.broadcast_to(np.asarray(columns.get(name, default), dtype), (count,))
        for name, (dtype, default) in _QUERY.items()
    }
    block["field_state"] = field_state
    return block


def endpoint_queries(tables: SourceTables, residues: ResidueTable) -> EndpointQuery:
    r"""
    The one query producer: both endpoints of every ``struct_conn``/``SSBOND``/``LINK`` row, of every branch
    link once per asym of its branched entity, and of every CONECT pair (the serial's first-model row; its author
    namespace re-binds it in another model).

    Raises:
        ContradictionError: ``key_invalid`` for a CONECT serial that two first-model rows carry.
    """

    atoms, pdb = tables.atoms, tables.refs.block is None
    c = tables.connections
    field_state = c.field_state.reshape(-1, _FIELDS)
    blocks = [
        _query_block(
            2 * len(c.ref),
            field_state,
            source=np.repeat(c.origin, 2),
            **{
                name: getattr(c, name).reshape(-1)
                for name in (
                    "label_asym",
                    "label_seq",
                    "label_comp",
                    "label_atom",
                    "auth_asym",
                    "auth_ins",
                    "auth_comp",
                    "auth_atom",
                )
            },
            auth_key=author_key(c.auth_seq_text.reshape(-1), field_state[:, _AUTH_SEQ], c.auth_seq.reshape(-1), pdb),
            altloc=c.label_alt.reshape(-1),
            ref=np.repeat(c.ref, 2),
        )
    ]

    # branch links, once per asym of the branched entity (instance order)
    links = tables.branch_links
    branched = residues.kind == _BRANCHED
    _, first = np.unique(residues.instance[branched], return_index=True)
    asym_entity = list(zip(residues.asym[branched][first].tolist(), residues.entity[branched][first].tolist()))
    pairs = [(j, a) for j, e in enumerate(links.entity.tolist()) for a, entity in asym_entity if entity == e]
    link = np.array([j for j, _ in pairs], dtype=np.int64)
    field_state = np.full((2 * len(pairs), _FIELDS), _UNKNOWN, dtype=np.uint8)
    field_state[:, _LABEL_ASYM] = _KNOWN_DERIVED
    field_state[:, [_LABEL_COMP, _LABEL_ATOM]] = _KNOWN_DECLARED
    blocks.append(
        _query_block(
            2 * len(pairs),
            field_state,
            source=QuerySource.BRANCH_LINK,
            label_asym=np.repeat([a for _, a in pairs], 2).astype(np.int32),
            label_comp=links.comp[link].reshape(-1),
            label_atom=links.atom[link].reshape(-1),
            scheme_num=links.num[link].reshape(-1),
            ref=np.repeat(links.ref[link], 2),
        )
    )

    # CONECT: serial -> the first model's row; a serial two of its rows carry names no one atom
    serial = tables.conect.serial.reshape(-1)
    first_model = np.flatnonzero(atoms.model == atoms.model[0]) if len(atoms.model) else np.zeros(0, np.int64)
    if len(serial):
        keys, counts = np.unique(atoms.serial[first_model], return_counts=True)
        if (doubled := np.isin(serial, keys[counts > 1])).any():
            k = int(np.argmax(doubled))
            rows = first_model[atoms.serial[first_model] == serial[k]]
            conect = tables.refs.ref(int(tables.conect.ref[k // 2]))
            fail("key_invalid", conect, *(_row_ref(atoms, int(row), tables.refs.block) for row in rows))
    hit = _lookup(atoms.serial[first_model], serial)
    named = hit >= 0
    row = np.where(named, first_model[hit.clip(0)] if len(first_model) else -1, -1)
    r = row.clip(0)
    take = (lambda column: np.where(named, column[r], -1)) if len(atoms.row) else (lambda column: np.full(len(row), -1))
    field_state = np.full((len(row), _FIELDS), _UNKNOWN, dtype=np.uint8)
    if len(atoms.row):
        for column, states in (
            (_AUTH_ASYM, atoms.auth_asym_state),
            (_AUTH_SEQ, atoms.auth_seq_state),
            (_AUTH_INS, atoms.insertion_state),
            (_AUTH_COMP, atoms.auth_comp_state),
            (_AUTH_ATOM, atoms.auth_atom_state),
            (_ALTLOC, atoms.label_alt_state),
        ):
            field_state[named, column] = states[r[named]]
    blocks.append(
        _query_block(
            len(row),
            field_state,
            source=QuerySource.CONECT,
            auth_asym=take(atoms.auth_asym),
            auth_key=author_key(take(atoms.auth_seq_text), field_state[:, _AUTH_SEQ], take(atoms.auth_seq), pdb),
            auth_ins=take(atoms.insertion),
            auth_comp=take(atoms.auth_comp),
            auth_atom=take(atoms.auth_atom),
            altloc=take(atoms.label_alt),
            row=np.where(take(residues.ordinal_of_row) >= 0, row, -1),  # the row lies in the residues' model
            ref=np.repeat(tables.conect.ref, 2),
        )
    )
    return EndpointQuery(
        **{name: np.concatenate([b[name] for b in blocks]).astype(dtype) for name, (dtype, _) in _QUERY.items()},
        field_state=np.concatenate([b["field_state"] for b in blocks]).astype(np.uint8),
    )


# --- atom names -------------------------------------------------------------------------------------------------------


def _bind_names(
    residues: ResidueTable,
    atoms: AtomTable,
    state: ChemicalState,
    definitions: Definitions,
    ordinal: NDArray[np.int64],
    name: NDArray[np.int64],
) -> tuple[NDArray[np.bool_], NDArray[np.int64], NDArray[np.bool_]]:
    r"""
    The one atom-name rule per (ordinal, atom-name code): whether the ordinal's component has an
    atomic definition in force; the definition atom the name names, -1 none — its own, else, for an isotope name
    ``D*`` the definition lacks, ``H*`` where the definition names ``H*`` and no row of the ordinal's realisation
    (``state``) is named ``H*``; and whether it names that atom by the isotope name.
    """

    ordinal, name = np.asarray(ordinal, dtype=np.int64), np.asarray(name, dtype=np.int64)
    comps, names = atoms.symbols["comp"], atoms.symbols["atom"]
    width = len(names) + 1
    keys, inverse = np.unique(residues.component[ordinal].astype(np.int64) * width + name + 1, return_inverse=True)
    code = {text: k for k, text in enumerate(names)}
    defined = np.zeros(len(keys), dtype=bool)
    own, isotope, hydrogen = (np.full(len(keys), -1, dtype=np.int64) for _ in range(3))
    for k, key in enumerate(keys.tolist()):
        c, a = divmod(key, width)
        cid = comps[c]
        definition = definitions.by_id.get(cid)
        if definition is None or not definition.atom_names:
            continue
        defined[k] = True
        text = names[a - 1] if a else ""
        if text in definition.atom_names:
            own[k] = definition.atom_names.index(text)
        elif text[:1] == "D" and (swapped := "H" + text[1:]) in definition.atom_names:
            isotope[k], hydrogen[k] = definition.atom_names.index(swapped), code.get(swapped, -1)
    inverse = inverse.ravel()
    swap = np.flatnonzero(hydrogen[inverse] >= 0)
    taken = np.zeros(len(ordinal), dtype=bool)
    if len(swap):  # a D* name the definition lacks: H* is taken when a row of the realisation is named H*
        rows = residues.rows[state.row_selected[residues.rows]]
        present = residues.ordinal_of_row[rows].astype(np.int64) * width + residues.name_of_row[rows] + 1
        taken[swap] = np.isin(ordinal[swap] * width + hydrogen[inverse[swap]] + 1, present)
    mapped = (isotope[inverse] >= 0) & ~taken
    return defined[inverse], np.where(mapped, isotope[inverse], own[inverse]), mapped


def definition_atoms(
    residues: ResidueTable, atoms: AtomTable, state: ChemicalState, definitions: Definitions
) -> tuple[NDArray[np.int64], NDArray[np.bool_]]:
    r"""
    Per atom row the definition atom it observes by the one atom-name rule (``_bind_names``; the resolver binds a row's
    endpoint by the same rule): -1 for a row of the state its definition lacks — an extra atom of its occurrence, as
    every row of a residue without an atomic definition is — and for a row outside the state; and whether it observes
    that atom by its isotope name (``isotope_name_mapped``). Two rows of a realisation never observe one atom: a name
    is stated once there (``duplicate_atom``) and ``D*`` reads as ``H*`` only where no row is ``H*``.
    """

    rows = residues.rows[state.row_selected[residues.rows]]
    atom, mapped = np.full(len(atoms.row), -1, dtype=np.int64), np.zeros(len(atoms.row), dtype=bool)
    _, atom[rows], mapped[rows] = _bind_names(
        residues, atoms, state, definitions, residues.ordinal_of_row[rows], residues.name_of_row[rows]
    )
    return atom, mapped


# --- endpoint resolution ----------------------------------------------------------------------------------------------


def _candidates(residue_keys, query_keys, stated, active, residue_comp, query_comp, comp_stated):
    r"""
    The residues one namespace admits, as (query, residue) pairs: per known-field pattern one sorted key of the stated
    fields; a stated component the residue leaves unstated contradicts nothing; a namespace stating only a component
    names the residues of that component.
    """

    queries, found = [np.zeros(0, np.int64)], [np.zeros(0, np.int64)]
    members = np.flatnonzero(active)
    if len(members):
        patterns, inverse = np.unique(stated[members], axis=0, return_inverse=True)
        for p, pattern in enumerate(patterns):
            these = members[inverse.ravel() == p]
            columns = np.flatnonzero(pattern)
            if len(columns):
                keys = _pack(*(np.concatenate([residue_keys[:, k], query_keys[these, k]]) for k in columns))
            else:
                keys = np.concatenate([residue_comp, query_comp[these]]).astype(np.int64)
            owner, index = _pairs(keys[: len(residue_keys)], keys[len(residue_keys) :])
            queries.append(these[owner])
            found.append(index)
    queries, found = np.concatenate(queries), np.concatenate(found)
    keep = ~comp_stated[queries] | (residue_comp[found] < 0) | (residue_comp[found] == query_comp[queries])
    return queries[keep], found[keep]


def resolve_endpoints(
    residues: ResidueTable, atoms: AtomTable, state: ChemicalState, definitions: Definitions, query: EndpointQuery
) -> EndpointResolution:
    r"""
    The one endpoint resolver: the label and author namespaces are evaluated independently (an all-unknown
    namespace is no constraint) and intersected; an author address compares a blank chain as the stated none and a
    number by its ``author_key``. Then the chemical state, the atom (the rows of the label name, else of the author
    name; a query stating both names a row stating both) and the altloc (rows of the named altloc plus altloc-less
    rows); a branch-link query matches the scheme ``num`` of its asym, a CONECT query its row. The definition atom is
    the found row's, else the query name's, by the one atom-name rule (``definition_atoms``): a row it binds no
    definition atom is the occurrence's extra atom; an atom with no row in the state stays RESOLVED when the
    definition has it. ``name`` is the found row's name, else the query's.
    """

    n = len(query.source)
    nres = len(residues.kind)
    fs = query.field_state
    stated = known(fs)
    outcome = np.full(n, _PENDING, dtype=np.uint8)
    ordinal = np.full(n, -1, dtype=np.int64)
    found_row = np.full(n, -1, dtype=np.int64)
    name = np.where(stated[:, _LABEL_ATOM], query.label_atom, np.where(stated[:, _AUTH_ATOM], query.auth_atom, -1))
    name = name.astype(np.int64)
    alt = np.where(stated[:, _ALTLOC], query.altloc, -1).astype(np.int64)
    alt_of_row = _code(atoms.label_alt, atoms.label_alt_state)

    # a CONECT serial naming a row of the selected model names its atom directly
    direct = query.row >= 0
    r = query.row[direct]
    ordinal[direct] = residues.ordinal_of_row[r]
    name[direct] = residues.name_of_row[r]
    alt[direct] = alt_of_row[r]
    found_row[direct] = np.where(state.row_selected[r], r, -1)
    outcome[direct] = np.where(state.row_selected[r], _PENDING, BindingOutcome.UNSELECTED_ALTERNATIVE)

    # the residues each namespace admits
    branch = query.scheme_num >= 0
    seq_stated = stated[:, _LABEL_SEQ] | (resolution(fs[:, _LABEL_SEQ]) == Resolution.INAPPLICABLE) | branch
    label_on = ~direct & (stated[:, _LABEL_ASYM] | seq_stated | stated[:, _LABEL_COMP])
    chain = stated_identity(query.auth_asym, fs[:, _AUTH_ASYM])
    number_stated = query.auth_key != NO_CODE
    author_on = ~direct & ((chain != NO_CODE) | number_stated | stated[:, _AUTH_COMP])
    # label_seq_id: a polymer residue's position; every other residue (a branched one too) has `.`; a branch-link
    # query names the branch scheme's num instead
    query_number = np.where(branch, query.scheme_num, query.label_seq)
    numbered, branched = known(residues.number_state), residues.kind == _BRANCHED
    floor = min(_floor(residues.number, numbered), _floor(query_number, stated[:, _LABEL_SEQ] | branch))
    query_number = np.where(stated[:, _LABEL_SEQ] | branch, query_number, floor - 1)
    label = [
        _candidates(
            np.column_stack([residues.asym, number_key]),
            np.column_stack([query.label_asym, query_number]),
            np.column_stack([stated[:, _LABEL_ASYM], seq_stated]),
            label_on & selects,
            residues.component,
            query.label_comp,
            stated[:, _LABEL_COMP],
        )
        for number_key, selects in (
            (np.where(numbered & ~branched, residues.number, floor - 1), ~branch),
            (np.where(branched, residues.number, floor - 1), branch),
        )
    ]
    lq, lc = np.concatenate([label[0][0], label[1][0]]), np.concatenate([label[0][1], label[1][1]])
    aq, ac = _candidates(
        np.column_stack(
            [
                stated_identity(residues.author_chain, residues.author_chain_state),
                residues.author_key,
                residues.insertion,
            ]
        ),
        np.column_stack([chain, query.auth_key, np.where(stated[:, _AUTH_INS], query.auth_ins, -1)]),
        np.column_stack([chain != NO_CODE, number_stated, number_stated]),
        author_on,
        residues.author_component,
        query.auth_comp,
        stated[:, _AUTH_COMP],
    )

    pending = outcome == _PENDING
    with_residues = np.zeros(len(residues.instance_symbol), dtype=bool)
    with_residues[residues.instance] = True
    unresolved = residues.instance_asym[~with_residues & (residues.instance_asym >= 0)]
    no_label = pending & label_on & (np.bincount(lq, minlength=n) == 0)
    no_author = pending & author_on & (np.bincount(aq, minlength=n) == 0)
    outcome[no_label & stated[:, _LABEL_ASYM] & np.isin(query.label_asym, unresolved)] = BindingOutcome.UNRESOLVED_ASYM
    nothing = pending & ~direct & ~label_on & ~author_on
    outcome[(outcome == _PENDING) & (no_label | no_author | nothing)] = BindingOutcome.DANGLING

    width = nres + 1
    both = label_on & author_on
    label_keys, author_keys = lq * width + lc, aq * width + ac
    shared = np.intersect1d(label_keys[both[lq]], author_keys[both[aq]])
    split = (outcome == _PENDING) & both & (np.bincount(shared // width, minlength=n) == 0)
    outcome[split] = BindingOutcome.LABEL_AUTH_CONFLICT
    conflict_pairs = np.concatenate([label_keys[split[lq]], author_keys[split[aq]]])
    keys = np.unique(
        np.concatenate([label_keys[(label_on & ~author_on)[lq]], author_keys[(author_on & ~label_on)[aq]], shared])
    )
    pq, pc = keys // width, keys % width
    live = outcome[pq] == _PENDING
    pq, pc = pq[live], pc[live]

    # one position, or AMBIGUOUS; within it the state's candidate
    places = np.unique(pq * (nres + 1) + residues.position[pc])
    distinct = np.bincount(places // (nres + 1), minlength=n)
    ambiguous = (outcome == _PENDING) & (distinct > 1)
    outcome[ambiguous] = BindingOutcome.AMBIGUOUS
    ambiguous_pairs = (pq * width + pc)[ambiguous[pq]]
    single = (outcome == _PENDING) & (distinct == 1)
    chosen = state.selected[pc] & single[pq]
    ordinal[pq[chosen]] = pc[chosen]
    first = np.full(n, -1, dtype=np.int64)
    named, at = np.unique(pq, return_index=True)
    first[named] = pc[at]
    outside = single & (ordinal < 0)
    ordinal[outside] = first[outside]
    outcome[outside] = BindingOutcome.UNSELECTED_ALTERNATIVE
    outcome[(outcome == _PENDING) & ~direct & ~residues.observed[ordinal.clip(0)]] = BindingOutcome.ABSENT_IN_MODEL

    # the atom: the rows of the label name, else of the author name (a row without one name answers to its other)
    need = np.flatnonzero((outcome == _PENDING) & (found_row < 0) & (name >= 0))
    owner = np.repeat(np.arange(nres), np.diff(residues.row_ptr))
    names = len(atoms.symbols["atom"]) + 1
    label_name = residues.name_of_row[residues.rows]
    author_name = _code(atoms.auth_atom[residues.rows], atoms.auth_atom_state[residues.rows])
    author_name = np.where(author_name >= 0, author_name, label_name)
    by_author = ~stated[need, _LABEL_ATOM]
    which, index = [], []
    for row_name, these in ((label_name, need[~by_author]), (author_name, need[by_author])):
        w, i = _pairs(owner * names + row_name + 1, ordinal[these] * names + name[these] + 1)
        which.append(these[w])
        index.append(i)
    which, index = np.concatenate(which), np.concatenate(index)
    # a query stating both names: a row stating both answers it; a row answering to one name alone contradicts it
    both_names = stated[:, _LABEL_ATOM] & stated[:, _AUTH_ATOM]
    answered = np.zeros(n, dtype=bool)
    answered[which] = True
    w, _ = _pairs(owner * names + author_name + 1, ordinal[need] * names + query.auth_atom[need] + 1)
    answered[need[w]] = True
    agree = ~both_names[which] | (author_name[index] == query.auth_atom[which])
    which, index = which[agree], index[agree]
    exists = np.zeros(n, dtype=bool)
    exists[which] = True
    renamed = np.flatnonzero(both_names & answered & ~exists)
    outcome[renamed] = BindingOutcome.LABEL_AUTH_CONFLICT
    conflict_pairs = np.concatenate([conflict_pairs, renamed * width + ordinal[renamed]])
    # within the state: rows of the named altloc and altloc-less rows
    candidate_rows = residues.rows[index]
    asked = alt[which]
    fits = state.row_selected[candidate_rows] & (
        (asked < 0) | (alt_of_row[candidate_rows] < 0) | (alt_of_row[candidate_rows] == asked)
    )
    hit, at = np.unique(which[fits], return_index=True)
    found_row[hit] = candidate_rows[fits][at]
    name[hit] = label_name[index[fits][at]]
    elsewhere = (alt >= 0) & (state.altloc[ordinal.clip(0)] != alt)
    unselected = (outcome == _PENDING) & (found_row < 0) & (name >= 0) & (exists | elsewhere)
    outcome[unselected] = BindingOutcome.UNSELECTED_ALTERNATIVE

    # the definition atom: the found row's, else the query name's, by the one atom-name rule
    atom = np.full(n, -1, dtype=np.int64)
    defined = np.zeros(n, dtype=bool)
    rest = np.flatnonzero(outcome == _PENDING)
    defined[rest], atom[rest], _ = _bind_names(residues, atoms, state, definitions, ordinal[rest], name[rest])
    seen = (outcome == _PENDING) & (found_row >= 0)
    extra = np.where(seen & (atom < 0), found_row, -1)
    unseen = (outcome == _PENDING) & (found_row < 0)
    outcome[unseen & (~defined | (name < 0))] = BindingOutcome.ATOM_UNBOUND
    outcome[unseen & defined & (name >= 0) & (atom < 0)] = BindingOutcome.ATOM_NOT_IN_REFERENCE
    outcome[outcome == _PENDING] = BindingOutcome.RESOLVED

    pairs = np.unique(np.concatenate([conflict_pairs, ambiguous_pairs]))
    return EndpointResolution(
        outcome=outcome,
        ordinal=ordinal.astype(np.int32),
        atom=np.where(outcome == BindingOutcome.RESOLVED, atom, -1).astype(np.int32),
        extra=extra.astype(np.int32),
        name=name.astype(np.int32),
        candidates_ptr=_ptr(np.bincount(pairs // width, minlength=n)),
        candidates=(pairs % width).astype(np.int32),
    )
