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
The packed fact state, its projections, and the columnar records and source references every layer reports in.

A fact state byte packs two dimensions: ``state = (resolution << 3) | basis``. ``known`` is the only "is known"
projection. A record is one row of ``Records``: a code of ``ofoldx.errors.CODES``, a subject in the value the operation
returned, its source references and the typed payload the code's row declares; kind and cause are read from the code,
never stored.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import IntEnum
from typing import cast

import numpy as np
from numpy.typing import NDArray

from .errors import CODE_NAMES, CODES, Cause, Disposition, PayloadKind, RecordKind, RequestRef, SourceRef, fail


class Resolution(IntEnum):
    KNOWN = 0
    UNKNOWN = 1
    INAPPLICABLE = 2
    CONFLICT = 3


class Basis(IntEnum):
    NONE = 0  # no basis: absent item without a default, nothing determines the fact
    DECLARED = 1  # a stated token of the source at occurrence scope (or a caller-stated recipe value)
    DEFAULT = 2  # DDL2 _item_default.value applied because the item is ABSENT
    DEFINED = 3  # dictionary/type semantics of a stated token or structure
    DERIVED = 4  # deterministic derivation with recorded dependencies
    REFERENCE = 5  # retained from the component definition (file, caller or archive scope chem_comp*)
    ASSUMED = 6  # consumer assumption set only; never produced by a default read


_BASIS_BITS = 3
_BASIS_MASK = (1 << _BASIS_BITS) - 1
# The legal (resolution, basis) pairs.
_LEGAL = {
    Resolution.KNOWN: (Basis.DECLARED, Basis.DEFAULT, Basis.DEFINED, Basis.DERIVED, Basis.REFERENCE, Basis.ASSUMED),
    Resolution.UNKNOWN: (Basis.NONE, Basis.DECLARED, Basis.DEFINED, Basis.DERIVED, Basis.REFERENCE),
    Resolution.INAPPLICABLE: (Basis.DECLARED, Basis.DEFINED),
    Resolution.CONFLICT: (Basis.NONE,),
}
ALLOWED_STATES: frozenset[int] = frozenset(r << _BASIS_BITS | b for r, bases in _LEGAL.items() for b in bases)


def state(resolution: Resolution, basis: Basis) -> int:
    r"""The state byte of one legal (resolution, basis) pair."""

    packed = resolution << _BASIS_BITS | basis
    assert packed in ALLOWED_STATES, f"({resolution!r}, {basis!r}) is not a state"
    return packed


def states(resolution: NDArray[np.uint8], basis: NDArray[np.uint8]) -> NDArray[np.uint8]:
    r"""The state bytes of legal (resolution, basis) pairs, elementwise."""

    packed = resolution << _BASIS_BITS | basis
    # a basis above the mask would carry into the resolution bits and could read as another legal state
    legal = (basis <= _BASIS_MASK) & np.isin(packed, list(ALLOWED_STATES))
    assert legal.all(), f"not states: {np.unique(packed[~legal]).tolist()}"
    return packed


def resolution(states: NDArray[np.uint8]) -> NDArray[np.uint8]:
    return states >> _BASIS_BITS


def basis(states: NDArray[np.uint8]) -> NDArray[np.uint8]:
    return states & _BASIS_MASK


def known(states: NDArray[np.uint8]) -> NDArray[np.bool_]:
    r"""``resolution == KNOWN``: the only "is known" projection."""
    return resolution(states) == Resolution.KNOWN


def determined(states: NDArray[np.uint8]) -> NDArray[np.bool_]:
    r"""Known on a basis other than a consumer assumption."""
    return known(states) & (basis(states) != Basis.ASSUMED)


class SubjectKind(IntEnum):
    NONE = 0
    INSTANCE = 1
    COMPONENT = 2
    ATOM = 3
    EDGE = 4
    EXTERNAL = 5
    STEREOCENTRE = 6
    PAIR = 7
    OCCURRENCE = 8  # native ordinal of the read model's residue table: keys native chemistry across reads


def _freeze(*columns: NDArray) -> None:
    for column in columns:
        column.flags.writeable = False


def _union(tables: Sequence[tuple[str, ...]]) -> tuple[tuple[str, ...], tuple[NDArray[np.int64], ...]]:
    r"""One symbol table for several, and per table the new index of each of its symbols."""

    index: dict[str, int] = {}
    maps = tuple(np.array([index.setdefault(s, len(index)) for s in table], dtype=np.int64) for table in tables)
    return tuple(index), maps


@dataclass(frozen=True, eq=False)
class SourceRefs:
    r"""
    One table of source references per read or operation, stored as columns.

    ``category`` indexes ``categories``; ``item`` indexes ``items``, ``-1`` when the reference names none. The block is
    shared by every row. Construction makes the columns read-only.
    """

    category: NDArray[np.int16]
    row: NDArray[np.int64]
    item: NDArray[np.int16]
    categories: tuple[str, ...]
    items: tuple[str, ...]
    block: str | None

    def __post_init__(self) -> None:
        _freeze(self.category, self.row, self.item)

    def ref(self, index: int) -> SourceRef:
        item = int(self.item[index])
        return SourceRef(
            self.categories[self.category[index]],
            int(self.row[index]),
            self.items[item] if item >= 0 else None,
            self.block,
        )

    def __len__(self) -> int:
        return len(self.row)

    @classmethod
    def concat(cls, parts: Sequence[SourceRefs]) -> tuple[SourceRefs, tuple[NDArray[np.int64], ...]]:
        r"""
        Join reference tables of one block.

        Returns:
            The joined table, and per part the row in it of each of the part's rows.
        """

        blocks = {part.block for part in parts if len(part)}
        if len(blocks) > 1:
            fail("value_invalid", RequestRef("SourceRefs.block", ", ".join(sorted(map(repr, blocks)))))
        if not parts:
            return _no_refs(), ()
        categories, category_maps = _union([part.categories for part in parts])
        items, item_maps = _union([part.items for part in parts])
        starts = np.cumsum([0, *map(len, parts)])
        joined = cls(
            np.concatenate([m[part.category] for m, part in zip(category_maps, parts)]).astype(np.int16),
            np.concatenate([part.row for part in parts]),
            # the appended -1 keeps "no item" (-1 indexes the last element)
            np.concatenate([np.append(m, -1)[part.item] for m, part in zip(item_maps, parts)]).astype(np.int16),
            categories,
            items,
            blocks.pop() if blocks else None,
        )
        return joined, tuple(np.arange(start, start + len(part)) for start, part in zip(starts, parts))


def _no_refs() -> SourceRefs:
    return SourceRefs(
        np.zeros(0, dtype=np.int16),
        np.zeros(0, dtype=np.int64),
        np.zeros(0, dtype=np.int16),
        (),
        (),
        None,
    )


@dataclass(frozen=True, slots=True)
class Record:
    r"""One record, materialised only by ``Records.__getitem__`` (diagnostics, tests)."""

    code: str
    kind: RecordKind
    cause: Cause
    subject: tuple[SubjectKind, int]  # index into the returned value; -1 = subject not realised in it
    sources: tuple[SourceRef, ...]
    detail: Mapping[str, object]  # rendered from the typed payload columns by CODES[code].payload


# Per code (indexed like CODE_NAMES): its record kind (255 for raise-only codes), its cause, the payload kind of each
# of its four slots and of its list field (-1 when unused).
_SLOTS = 4
_CODE_INDEX = {name: index for index, name in enumerate(CODE_NAMES)}
_KIND = np.array([255 if CODES[name].kind is None else CODES[name].kind for name in CODE_NAMES], dtype=np.uint8)
_CAUSE = np.array([CODES[name].cause for name in CODE_NAMES], dtype=np.uint8)
_SLOT_KIND = np.full((len(CODE_NAMES), _SLOTS), -1, dtype=np.int8)
_LIST_KIND = np.full(len(CODE_NAMES), -1, dtype=np.int8)
for _index, _name in enumerate(CODE_NAMES):
    _SLOT_KIND[_index, : len(CODES[_name].payload)] = [kind for _, kind in CODES[_name].payload]
    _payload_list = CODES[_name].payload_list
    if _payload_list is not None:
        _LIST_KIND[_index] = _payload_list[1]
del _index, _name, _payload_list
# a remapped subject axis also re-indexes the payload slots and list values that point into it
_PAYLOAD_AXIS = {SubjectKind.ATOM: PayloadKind.ATOM, SubjectKind.COMPONENT: PayloadKind.COMPONENT}


def _take_csr(ptr: NDArray[np.int64], values: NDArray, rows: NDArray[np.int64]) -> tuple[NDArray, NDArray]:
    lengths = np.diff(ptr)[rows]
    taken = np.zeros(len(rows) + 1, dtype=np.int64)
    np.cumsum(lengths, out=taken[1:])
    return taken, values[np.repeat(ptr[:-1][rows] - taken[:-1], lengths) + np.arange(taken[-1])]


def _join_ptr(ptrs: Sequence[NDArray[np.int64]]) -> NDArray[np.int64]:
    return np.concatenate([[0], np.cumsum(np.concatenate([np.diff(ptr) for ptr in ptrs]))]).astype(np.int64)


def _translate(values: NDArray, selected: NDArray[np.bool_], mapping: NDArray[np.int64]) -> NDArray[np.int64]:
    r"""``values`` with each selected non-negative value replaced by ``mapping[value]``."""

    hit = selected & (values >= 0)
    translated = values.copy()
    translated[hit] = mapping[values[hit]]
    return translated


@dataclass(frozen=True, eq=False)
class Records:
    r"""
    Subject-keyed records of one read or operation, stored as columns; any number of records may share a subject.

    ``code`` indexes ``CODE_NAMES`` (record codes only). ``ref_ptr``/``ref`` are the CSR of each record's rows of
    ``refs``. ``payload`` holds the code's fixed slots (0 when unused), ``list_ptr``/``list_value`` the CSR of its
    list field; a TEXT value indexes ``texts``, a CODE value ``CODE_NAMES``, and an ATOM, COMPONENT or DECLARATION
    value is an index or ``-1``. Construction makes the columns read-only.
    """

    code: NDArray[np.uint16]
    subject_kind: NDArray[np.uint8]
    subject: NDArray[np.int64]
    ref_ptr: NDArray[np.int64]
    ref: NDArray[np.int64]
    refs: SourceRefs
    payload: NDArray[np.int64]
    list_ptr: NDArray[np.int64]
    list_value: NDArray[np.int64]
    texts: tuple[str, ...]

    def __post_init__(self) -> None:
        _freeze(
            self.code,
            self.subject_kind,
            self.subject,
            self.ref_ptr,
            self.ref,
            self.payload,
            self.list_ptr,
            self.list_value,
        )

    @classmethod
    def empty(cls) -> Records:
        return cls(
            np.zeros(0, dtype=np.uint16),
            np.zeros(0, dtype=np.uint8),
            np.zeros(0, dtype=np.int64),
            np.zeros(1, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            _no_refs(),
            np.zeros((0, _SLOTS), dtype=np.int64),
            np.zeros(1, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            (),
        )

    @classmethod
    def from_rows(cls, rows: Sequence[tuple], refs: SourceRefs) -> Records:
        r"""
        Records from rows ``(code, subject kind, subject, source rows, payload values[, list values])``.

        A TEXT payload value is a ``str``, a CODE value a code name; a source row ``-1`` is dropped.
        """

        texts: dict[str, int] = {}
        payload = np.zeros((len(rows), _SLOTS), dtype=np.int64)
        lists: list[int] = []
        list_ptr = np.zeros(len(rows) + 1, dtype=np.int64)
        for i, row in enumerate(rows):
            spec = CODES[row[0]]
            for j, ((_, kind), value) in enumerate(zip(spec.payload, row[4], strict=True)):
                payload[i, j] = cls._encode(kind, value, texts)
            if len(row) > 5:
                # A list value has the kind declared by its code's list field.
                lists += [
                    cls._encode(cast(tuple[str, PayloadKind], spec.payload_list)[1], value, texts) for value in row[5]
                ]
            list_ptr[i + 1] = len(lists)
        sources = [[int(ref) for ref in row[3] if ref >= 0] for row in rows]
        ref_ptr = np.zeros(len(rows) + 1, dtype=np.int64)
        ref_ptr[1:] = np.cumsum([len(row) for row in sources])
        return cls(
            np.array([_CODE_INDEX[row[0]] for row in rows], dtype=np.uint16),
            np.array([row[1] for row in rows], dtype=np.uint8),
            np.array([row[2] for row in rows], dtype=np.int64),
            ref_ptr,
            np.array([ref for row in sources for ref in row], dtype=np.int64),
            refs,
            payload,
            list_ptr,
            np.array(lists, dtype=np.int64),
            tuple(texts),
        )

    @staticmethod
    def _encode(kind: PayloadKind, value: str | int | np.integer, texts: dict[str, int]) -> int:
        if kind == PayloadKind.TEXT:
            return texts.setdefault(str(value), len(texts))
        if kind == PayloadKind.CODE:
            return _CODE_INDEX[cast(str, value)]  # A CODE payload is a code name.
        return int(value)

    @classmethod
    def concat(cls, parts: Sequence[Records]) -> Records:
        r"""The records of every part in order, with one reference table and one text table."""

        if not parts:
            return cls.empty()
        # parts filtered from one table share it: join each referenced table once
        tables = {id(part.refs): part.refs for part in parts if len(part.ref)}
        refs, ref_maps = SourceRefs.concat(list(tables.values()))
        rows = dict(zip(tables, ref_maps))
        texts, text_maps = _union([part.texts for part in parts])
        payload, list_value = [], []
        for text_map, part in zip(text_maps, parts):
            slot_kinds, list_kinds = part.payload_kinds()
            payload.append(_translate(part.payload, slot_kinds == PayloadKind.TEXT, text_map))
            list_value.append(_translate(part.list_value, list_kinds == PayloadKind.TEXT, text_map))
        return cls(
            np.concatenate([part.code for part in parts]),
            np.concatenate([part.subject_kind for part in parts]),
            np.concatenate([part.subject for part in parts]),
            _join_ptr([part.ref_ptr for part in parts]),
            np.concatenate([rows[id(part.refs)][part.ref] if len(part.ref) else part.ref for part in parts]),
            refs,
            np.concatenate(payload),
            _join_ptr([part.list_ptr for part in parts]),
            np.concatenate(list_value),
            texts,
        )

    def kind(self) -> NDArray[np.uint8]:
        return _KIND[self.code]

    def cause(self) -> NDArray[np.uint8]:
        return _CAUSE[self.code]

    def where(
        self,
        *,
        kind: RecordKind | None = None,
        code: str | None = None,
        subject_kind: SubjectKind | None = None,
    ) -> Records:
        r"""The records matching every given filter, in order."""

        keep = np.ones(len(self), dtype=bool)
        if kind is not None:
            keep &= self.kind() == kind
        if code is not None:
            # a code that is never recorded would silently match nothing
            if code not in CODES or CODES[code].disposition is not Disposition.RECORD:
                fail("value_invalid", RequestRef("code", code))
            keep &= self.code == CODE_NAMES.index(code)
        if subject_kind is not None:
            keep &= self.subject_kind == subject_kind
        rows = np.flatnonzero(keep)
        ref_ptr, ref = _take_csr(self.ref_ptr, self.ref, rows)
        list_ptr, list_value = _take_csr(self.list_ptr, self.list_value, rows)
        return Records(
            self.code[rows],
            self.subject_kind[rows],
            self.subject[rows],
            ref_ptr,
            ref,
            self.refs,
            self.payload[rows],
            list_ptr,
            list_value,
            self.texts,
        )

    def remap(self, subject_kind: SubjectKind, old_to_new: NDArray[np.int64]) -> Records:
        r"""
        Re-index the subjects of one kind (and the payload values pointing into the same axis) through a projection.

        Args:
            subject_kind: The axis the projection changed.
            old_to_new: Per old index its new index, ``-1`` when it was removed. A removed subject keeps its record
                with subject ``-1``.
        """

        hit = (self.subject_kind == subject_kind) & (self.subject >= 0)
        subject = self.subject.copy()
        subject[hit] = old_to_new[self.subject[hit]]
        payload, list_value = self.payload, self.list_value
        if subject_kind in _PAYLOAD_AXIS:
            axis = _PAYLOAD_AXIS[subject_kind]
            slot_kinds, list_kinds = self.payload_kinds()
            payload = _translate(payload, slot_kinds == axis, old_to_new)
            list_value = _translate(list_value, list_kinds == axis, old_to_new)
        return Records(
            self.code,
            self.subject_kind,
            subject,
            self.ref_ptr,
            self.ref,
            self.refs,
            payload,
            self.list_ptr,
            list_value,
            self.texts,
        )

    def counts(self) -> Mapping[str, int]:
        r"""Records per code name, for the codes present."""
        codes, counts = np.unique(self.code, return_counts=True)
        return {CODE_NAMES[c]: int(n) for c, n in zip(codes, counts)}

    def __len__(self) -> int:
        return len(self.code)

    def __iter__(self) -> Iterator[Record]:
        for index in range(len(self)):
            yield self[index]

    def __getitem__(self, index: int) -> Record:
        index = range(len(self))[index]
        row = CODES[CODE_NAMES[self.code[index]]]
        sources = tuple(self.refs.ref(i) for i in self.ref[self.ref_ptr[index] : self.ref_ptr[index + 1]])
        detail = {name: self._render(kind, value) for (name, kind), value in zip(row.payload, self.payload[index])}
        if row.payload_list is not None:
            name, kind = row.payload_list
            values = self.list_value[self.list_ptr[index] : self.list_ptr[index + 1]]
            detail[name] = tuple(self._render(kind, value) for value in values)
        subject = (SubjectKind(self.subject_kind[index]), int(self.subject[index]))
        return Record(row.name, cast(RecordKind, row.kind), row.cause, subject, sources, detail)

    def _render(self, kind: PayloadKind, value: np.int64) -> object:
        if kind == PayloadKind.TEXT:
            return self.texts[value]
        if kind == PayloadKind.CODE:
            return CODE_NAMES[value]
        return int(value)

    def payload_kinds(self) -> tuple[NDArray[np.int8], NDArray[np.int8]]:
        r"""The ``PayloadKind`` of each fixed slot (``-1`` when unused) and of each list value, by record code."""
        return _SLOT_KIND[self.code], np.repeat(_LIST_KIND[self.code], np.diff(self.list_ptr))
