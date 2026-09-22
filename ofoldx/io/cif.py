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
CIF files as typed columns: a semantic adapter over gemmi's CIF syntax layer.

gemmi parses; this module turns each block into categories whose items are fixed-width byte columns with one
of four lexical states per cell (a value, ``?``, ``.``, or the item absent from the category), and decodes them on
request into interned codes, integers or numbers with their standard uncertainty and printed quantum. The interface
carries no Python string per cell. The tokens come from gemmi's Python API, one loop at a
time, and the gemmi ``Document`` is released before a ``CifFile`` is returned.

Category and item names are case-insensitive (CIF 1.1): categories are keyed by their lowercase name without the
leading underscore (``"atom_site"``), and items are looked up case-insensitively.
"""

from __future__ import annotations

import gzip
import os
import zlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import IntEnum
from types import MappingProxyType

import gemmi
import numpy as np
from numpy.typing import NDArray

from ..errors import RequestRef, SourceRef, fail


class TokenState(IntEnum):
    VALUE = 0
    UNKNOWN = 1  # ?
    INAPPLICABLE = 2  # .
    ABSENT = 3  # the item is not in the category


@dataclass(frozen=True, eq=False)
class CifCategory:
    r"""
    One category of a block: ``nrows`` rows of the ``items`` it states, in file spelling. An item not in ``items``
    is ``ABSENT`` for every row. Decoded columns are computed once per item and kept.
    """

    name: str
    nrows: int
    items: tuple[str, ...]
    _columns: Mapping[str, tuple[NDArray[np.bytes_], NDArray[np.uint8]]] = field(repr=False)
    _block: str | None = field(default=None, repr=False)
    _decoded: dict = field(default_factory=dict, repr=False)

    def text(self, item: str) -> tuple[NDArray[np.bytes_], NDArray[np.uint8]]:
        r"""The unquoted values (``b""`` where not a value) and the ``TokenState`` of each row."""
        column = self._columns.get(item.lower())
        if column is None:
            return np.zeros(self.nrows, dtype="S1"), np.full(self.nrows, TokenState.ABSENT, dtype=np.uint8)
        return column

    def codes(self, item: str) -> tuple[NDArray[np.int32], NDArray[np.uint8], tuple[str, ...]]:
        r"""The values interned: per row an index into the symbols (−1 where not a value), the states, the symbols."""
        key = ("codes", item.lower())
        if key not in self._decoded:
            values, states = self.text(item)
            self._decoded[key] = (*intern(values, states == TokenState.VALUE), states)
        codes, symbols, states = self._decoded[key]
        return codes, states, symbols

    def floats(
        self, item: str
    ) -> tuple[NDArray[np.float64], NDArray[np.uint8], NDArray[np.float64], NDArray[np.float64]]:
        r"""
        The values as numbers: value (0 where not a value), state, standard uncertainty (``nan`` where none is
        printed) and printed quantum (one unit of the last printed digit; 0 where not a value).

        Raises:
            ContradictionError: ``number_grammar`` for a value outside the CIF numeric grammar.
            UnsupportedError: ``numeric_range`` for a number outside finite float64.
        """
        key = ("floats", item.lower())
        if key not in self._decoded:
            values, states = self.text(item)
            self._decoded[key] = parse_numbers(values, states == TokenState.VALUE, self._where(item))
        value, uncertainty, quantum = self._decoded[key]
        return value, self.text(item)[1], uncertainty, quantum

    def ints(self, item: str) -> tuple[NDArray[np.int64], NDArray[np.uint8]]:
        r"""
        The values as integers (0 where not a value) and the states.

        Raises:
            ContradictionError: ``number_grammar`` for a value that is not an optionally signed digit string.
            UnsupportedError: ``numeric_range`` for an integer outside int64.
        """
        key = ("ints", item.lower())
        if key not in self._decoded:
            values, states = self.text(item)
            self._decoded[key] = parse_integers(values, states == TokenState.VALUE, self._where(item))
        return self._decoded[key], self.text(item)[1]

    def token(self, item: str, row: int) -> str:
        r"""One cell for diagnostics: the value quoted as needed, ``?``, ``.``, or ``""`` for an absent item."""
        values, states = self.text(item)
        state = TokenState(int(states[row]))
        if state is TokenState.VALUE:
            return gemmi.cif.quote(values[row].decode())
        return {TokenState.UNKNOWN: "?", TokenState.INAPPLICABLE: ".", TokenState.ABSENT: ""}[state]

    def _where(self, item: str) -> Callable[[int], SourceRef]:
        return lambda row: SourceRef(self.name, row, item, self._block)


@dataclass(frozen=True)
class CifBlock:
    name: str
    categories: Mapping[str, CifCategory]


@dataclass(frozen=True)
class CifFile:
    blocks: tuple[CifBlock, ...]

    def block(self, key: str | int) -> CifBlock:
        r"""A block by position or by name (case-insensitive); ``InvalidRequestError(block_unknown)`` when none."""
        if isinstance(key, int):
            if not -len(self.blocks) <= key < len(self.blocks):
                fail("block_unknown", RequestRef("block", str(key)))
            return self.blocks[key]
        for block in self.blocks:
            if block.name.lower() == key.lower():
                return block
        fail("block_unknown", RequestRef("block", key))


def read_cif(path: str | os.PathLike[str]) -> CifFile:
    r"""Read a CIF file, gzip-compressed or not (detected by content)."""
    with open(path, "rb") as stream:
        return parse_cif(stream.read())


def parse_cif(data: str | bytes) -> CifFile:
    r"""
    Parse CIF text.

    Raises:
        InvalidRequestError: ``format_unknown`` for a broken gzip stream.
        ContradictionError: ``cif_syntax``, ``cif_loop_count``, ``cif_duplicate_item`` or ``non_ascii_source`` as
            gemmi reports the text; ``cif_loop_count`` also for a category stated in more than one loop, or in a loop
            and as pairs.
        UnsupportedError: ``cif_frame_unsupported`` for a save frame; ``cif_form_unsupported`` for a ``global_``
            block, a tag that is not ``_category.item``, or a loop over more than one category.
    """
    data = decompress(data.encode() if isinstance(data, str) else data)
    try:
        document = gemmi.cif.read_string(data)
    except (RuntimeError, ValueError) as error:
        message = str(error)
        if "duplicate" in message:
            fail("cif_duplicate_item", detail=message)
        if "Wrong number of values" in message:
            fail("cif_loop_count", detail=message)
        if not data.isascii():
            fail("non_ascii_source", detail=message)
        fail("cif_syntax", detail=message)
    blocks = tuple(_block(block) for block in document)
    del document
    return CifFile(blocks)


def decompress(data: bytes) -> bytes:
    r"""``data`` itself, or its gzip decompression when it starts with the gzip magic bytes."""
    if not data.startswith(b"\x1f\x8b"):
        return data
    try:
        return gzip.decompress(data)
    except (OSError, EOFError, zlib.error) as error:
        fail("format_unknown", detail=f"gzip: {error}")


def _block(block: gemmi.cif.Block) -> CifBlock:
    r"""Categories of one block in file order: each loop one category, single-valued pairs grouped by category."""
    if not block.name:
        fail("cif_form_unsupported", SourceRef("global_", 0, None, None))
    order: dict[str, None] = {}
    loops: dict[str, CifCategory] = {}
    pairs: dict[str, list[tuple[str, str]]] = {}
    for item in block:
        if item.frame is not None:
            fail("cif_frame_unsupported", SourceRef(f"save_{item.frame.name}", item.line_number, None, block.name))
        if item.pair is not None:
            category, name = _split(item.pair[0], item.line_number, block.name)
            if category in loops:
                fail("cif_loop_count", SourceRef(category, 0, name, block.name))
            pairs.setdefault(category, []).append((name, item.pair[1]))
            order.setdefault(category)
        elif item.loop is not None:
            tags = item.loop.tags
            split = [_split(tag, item.line_number, block.name) for tag in tags]
            category = split[0][0]
            for other, name in split:
                if other != category:
                    fail("cif_form_unsupported", SourceRef(category, item.line_number, f"_{other}.{name}", block.name))
            if category in order:
                fail("cif_loop_count", SourceRef(category, 0, None, block.name))
            names = [name for _, name in split]
            loops[category] = _category(category, item.loop.length(), names, _columns(item.loop), block.name)
            order.setdefault(category)
    for category, items in pairs.items():
        loops[category] = _category(
            category, 1, [name for name, _ in items], [_column([token]) for _, token in items], block.name
        )
    return CifBlock(block.name, MappingProxyType({category: loops[category] for category in order}))


def _split(tag: str, line: int, block: str) -> tuple[str, str]:
    r"""Category (lowercase, no leading underscore) and item of a ``_category.item`` tag; a dotless tag is refused."""
    category, dot, item = tag.partition(".")
    if not dot:
        fail("cif_form_unsupported", SourceRef(tag, line, None, block))
    return category.lstrip("_").lower(), item


def _category(name: str, nrows: int, items: list[str], columns: list, block: str) -> CifCategory:
    keys = [item.lower() for item in items]  # gemmi has refused duplicate tags already
    return CifCategory(name, nrows, tuple(items), MappingProxyType(dict(zip(keys, columns))), block)


def _columns(loop: gemmi.cif.Loop) -> list[tuple[NDArray[np.bytes_], NDArray[np.uint8]]]:
    r"""
    The columns of a loop (``_column`` each), cut from one buffer of its row-major gemmi tokens joined by NUL. The
    token list is released once joined and the joined text once padded: neither is held beside the columns.
    """
    tokens, width = loop.values, loop.width()
    joined = "\0".join(tokens)
    if not tokens or joined.count("\0") != len(tokens) - 1:  # a token holding a NUL: one column at a time
        return [_column(tokens[j::width]) for j in range(width)]
    del tokens
    try:
        text = joined.encode("ascii")
    except UnicodeEncodeError:
        fail("non_ascii_source", detail="a CIF token outside ASCII")
    del joined
    buffer = np.frombuffer(text, dtype=np.uint8)
    # token k spans bounds[k] + 1 .. bounds[k + 1]: the NUL before it (-1 for the first) to the one after it
    bounds = np.concatenate(([-1], np.flatnonzero(buffer == 0), [len(buffer)]))
    padded = np.zeros(len(buffer) + max(int(np.diff(bounds).max()) - 1, 1), dtype=np.uint8)
    padded[: len(buffer)] = buffer
    del buffer, text
    columns = []
    for j in range(width):
        start, end = bounds[j:-1:width] + 1, bounds[j + 1 :: width]
        length = end - start
        size = max(int(length.max()), 1)
        matrix = np.lib.stride_tricks.sliding_window_view(padded, size)[start]
        matrix *= np.arange(size) < length[:, None]

        def token(row: int, starts: NDArray = start, ends: NDArray = end) -> str:
            return padded[starts[row] : ends[row]].tobytes().decode()

        columns.append(_unquote(matrix.view(f"S{size}").ravel(), token))
    return columns


def _column(tokens: list[str]) -> tuple[NDArray[np.bytes_], NDArray[np.uint8]]:
    r"""Raw gemmi tokens as unquoted fixed-width bytes and token states (an unquoted ``?``/``.`` is a null)."""
    try:
        values = np.array(tokens, dtype=np.bytes_)
    except UnicodeEncodeError:
        fail("non_ascii_source", detail="a CIF token outside ASCII")
    return _unquote(values.astype(f"S{max(values.dtype.itemsize, 1)}"), tokens.__getitem__)


def _unquote(values: NDArray[np.bytes_], token: Callable[[int], str]) -> tuple[NDArray[np.bytes_], NDArray[np.uint8]]:
    r"""Raw tokens as fixed-width bytes (written in place) unquoted, and their token states; ``token(row)`` is one."""
    n, width = len(values), values.dtype.itemsize
    states = np.zeros(n, dtype=np.uint8)
    states[values == b"?"] = TokenState.UNKNOWN
    states[values == b"."] = TokenState.INAPPLICABLE
    matrix = values.view(np.uint8).reshape(n, width)
    matrix[states > 0] = 0
    first = matrix[:, 0]
    quoted = np.flatnonzero((first == ord("'")) | (first == ord('"')))
    if len(quoted):
        lengths = np.strings.str_len(values[quoted])
        inner = matrix[quoted]
        inner[:, :-1] = inner[:, 1:]
        inner[:, -1] = 0
        inner[np.arange(len(quoted)), lengths - 2] = 0
        matrix[quoted] = inner
    for row in np.flatnonzero(first == ord(";")):  # text fields are rare: one at a time
        text = gemmi.cif.as_string(token(row)).encode()
        if len(text) > width:
            values = values.astype(f"S{len(text)}")
            width = len(text)
        values[row] = text
    return values, states


def intern(values: NDArray[np.bytes_], mask: NDArray[np.bool_]) -> tuple[NDArray[np.int32], tuple[str, ...]]:
    r"""
    The distinct values of the masked rows and per row its index among them (−1 outside the mask). Values of at most
    8 bytes are interned as ``uint64``, wider ones as fixed-width ``void`` rows: never a per-cell string.
    """
    n, width = len(values), values.dtype.itemsize
    codes = np.full(n, -1, dtype=np.int32)
    if not mask.any():
        return codes, ()
    selected = np.ascontiguousarray(values[mask])
    matrix = selected.view(np.uint8).reshape(len(selected), width)
    head = np.ones(len(selected), dtype=bool)  # columns repeat in runs: intern the first value of each run
    if width <= 8:
        padded = np.zeros((len(selected), 8), dtype=np.uint8)
        padded[:, :width] = matrix
        keys = padded.view("<u8").ravel()
        head[1:] = keys[1:] != keys[:-1]
    else:
        keys = selected.view(np.dtype((np.void, width)))
        head[1:] = (matrix[1:] != matrix[:-1]).any(axis=1)
    unique, inverse = np.unique(keys[head], return_inverse=True)
    codes[mask] = inverse.reshape(-1)[np.cumsum(head) - 1]
    return codes, tuple(key.tobytes().rstrip(b"\0").decode() for key in unique)


# The CIF numeric grammar: mantissa [+-]?(d+(.d*)?|.d+), standard uncertainty (d+) before or after an optional
# exponent [eE][+-]?d+, never both. A table-driven automaton runs over the byte columns.
_DIGIT, _SIGN, _DOT, _EXP, _OPEN, _CLOSE, _OTHER, _END = range(8)
_CLASS = np.full(256, _OTHER, dtype=np.uint8)
_CLASS[ord("0") : ord("9") + 1] = _DIGIT
_CLASS[[ord("+"), ord("-")]] = _SIGN
_CLASS[ord(".")] = _DOT
_CLASS[[ord("e"), ord("E")]] = _EXP
_CLASS[ord("(")] = _OPEN
_CLASS[ord(")")] = _CLOSE
_CLASS[0] = _END
_DONE, _ERROR = 15, 16
_TRANSITIONS = {
    0: {_DIGIT: 2, _SIGN: 1, _DOT: 5},  # start
    1: {_DIGIT: 2, _DOT: 5},  # sign
    2: {_DIGIT: 2, _DOT: 3, _EXP: 6, _OPEN: 9, _END: _DONE},  # integer digits
    3: {_DIGIT: 4, _EXP: 6, _OPEN: 9, _END: _DONE},  # "d."
    4: {_DIGIT: 4, _EXP: 6, _OPEN: 9, _END: _DONE},  # fraction digits
    5: {_DIGIT: 4},  # leading "."
    6: {_DIGIT: 8, _SIGN: 7},  # exponent mark
    7: {_DIGIT: 8},  # exponent sign
    8: {_DIGIT: 8, _OPEN: 12, _END: _DONE},  # exponent digits
    9: {_DIGIT: 10},  # "(" before an exponent
    10: {_DIGIT: 10, _CLOSE: 11},
    11: {_EXP: 17, _END: _DONE},  # ")" before an exponent
    12: {_DIGIT: 13},  # "(" after the exponent
    13: {_DIGIT: 13, _CLOSE: 14},
    14: {_END: _DONE},
    _DONE: {_END: _DONE},
    17: {_DIGIT: 19, _SIGN: 18},  # exponent after a standard uncertainty: no second one
    18: {_DIGIT: 19},
    19: {_DIGIT: 19, _END: _DONE},
}
_TABLE = np.full((20, 8), _ERROR, dtype=np.uint8)
for _state, _moves in _TRANSITIONS.items():
    for _class, _next in _moves.items():
        _TABLE[_state, _class] = _next
del _state, _moves, _class, _next


def _bracket(classes: NDArray[np.uint8]) -> NDArray[np.bool_]:
    r"""The standard uncertainty's bytes, its parentheses included."""
    return (np.cumsum(classes == _OPEN, axis=1) > np.cumsum(classes == _CLOSE, axis=1)) | (classes == _CLOSE)


def _horner(digits: NDArray[np.uint8], mask: NDArray[np.bool_]) -> NDArray[np.float64]:
    r"""The decimal integer of the masked digit bytes of each row, as float64 (exact up to 2**53)."""
    value = np.zeros(len(digits))
    for j in range(digits.shape[1]):
        value = np.where(mask[:, j], value * 10 + (digits[:, j].astype(np.float64) - ord("0")), value)
    return value


def parse_numbers(
    values: NDArray[np.bytes_], mask: NDArray[np.bool_], where: Callable[[int], SourceRef | RequestRef]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    r"""
    The CIF numbers of the masked rows: value, standard uncertainty (``nan`` none), each the float nearest its decimal
    (subnormals included), and printed quantum (10 to the power of the last printed digit's place); outside the mask
    0, ``nan`` and 0. ``where(row)`` names a refused row.

    Raises:
        ContradictionError: ``number_grammar`` for a token outside the grammar.
        UnsupportedError: ``numeric_range`` for a value or uncertainty outside finite float64 (an underflow to zero
            of a non-zero mantissa included).
    """
    n = len(values)
    value, uncertainty, quantum = np.zeros(n), np.full(n, np.nan), np.zeros(n)
    rows = np.flatnonzero(mask)
    if not len(rows):
        return value, uncertainty, quantum
    selected = np.ascontiguousarray(values[rows])
    matrix = selected.view(np.uint8).reshape(len(rows), selected.dtype.itemsize)
    # one pass over the byte columns: the automaton, and per row its mantissa's decimals and non-zero digits
    count = len(rows)
    state, decimals = np.zeros(count, dtype=np.uint8), np.zeros(count, dtype=np.int64)
    dotted, ended, nonzero, exponent_mark, opened = (np.zeros(count, dtype=bool) for _ in range(5))
    for byte in np.ascontiguousarray(matrix.T):
        classes = _CLASS[byte]
        state = _TABLE[state, classes]
        exponent_mark |= classes == _EXP
        opened |= classes == _OPEN
        ended |= (classes == _EXP) | (classes == _OPEN)  # the mantissa ends at an exponent or an uncertainty
        dotted |= classes == _DOT
        decimals += ~ended & dotted & (classes == _DIGIT)
        nonzero |= ~ended & (byte >= ord("1")) & (byte <= ord("9"))
    accepted = _TABLE[state, _END] == _DONE
    if not accepted.all():
        fail("number_grammar", where(int(rows[np.flatnonzero(~accepted)[0]])))
    exponent = np.zeros(len(rows))
    marked = np.flatnonzero(exponent_mark)
    if len(marked):
        digits, classes = matrix[marked], _CLASS[matrix[marked]]
        after = np.cumsum(classes == _EXP, axis=1) > 0
        part = after & ~_bracket(classes) & (classes == _DIGIT)
        magnitude = np.minimum(_horner(digits, part), 10_000)
        negative = (after & (digits == ord("-"))).any(axis=1)
        exponent[marked] = np.where(negative, -magnitude, magnitude)
    with np.errstate(over="ignore", under="ignore"):
        place = 10.0 ** (exponent - decimals)
        braced = np.flatnonzero(opened)  # the grammar closes every "(" it opens
        if len(braced):
            classes = _CLASS[matrix[braced]]
            bracket = _bracket(classes)
            keep = ~bracket & (matrix[braced] != 0)
            order = np.argsort(~keep, axis=1, kind="stable")
            compact = np.take_along_axis(np.where(keep, matrix[braced], 0), order, axis=1).astype(np.uint8)
            selected = selected.copy()
            selected[braced] = np.ascontiguousarray(compact).view(selected.dtype).ravel()
            digits = bracket & (classes == _DIGIT)
            order = np.argsort(~digits, axis=1, kind="stable")
            su = np.take_along_axis(np.where(digits, matrix[braced], 0), order, axis=1).astype(np.uint8)
            su = np.strings.add(np.ascontiguousarray(su).view(selected.dtype).ravel(), b"e")
            # its digits at the value's last place, read as the value is: the float nearest the decimal
            power = (exponent - decimals)[braced].astype(np.int64).astype("S")
            uncertainty[rows[braced]] = np.strings.add(su, power).astype(np.float64)
        number = selected.astype(np.float64)
    bad = ~np.isfinite(number) | (nonzero & (number == 0)) | ~np.isfinite(place) | (place == 0)
    bad |= np.isinf(uncertainty[rows])
    if bad.any():
        fail("numeric_range", where(int(rows[np.flatnonzero(bad)[0]])))
    value[rows], quantum[rows] = number, place
    return value, uncertainty, quantum


def integer_tokens(values: NDArray[np.bytes_]) -> tuple[NDArray[np.int64], NDArray[np.bool_], NDArray[np.bool_]]:
    r"""
    The CIF integer grammar, the one definition every integer reading uses (``parse_integers``, an mmCIF author
    number): per token its integer (0 where it has none), whether it is an optionally signed digit string, and whether
    it is one within int64.
    """
    n = len(values)
    value, grammatical, readable = np.zeros(n, dtype=np.int64), np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    if not n:
        return value, grammatical, readable
    selected = np.ascontiguousarray(values)
    matrix = selected.view(np.uint8).reshape(n, selected.dtype.itemsize)
    classes = _CLASS[matrix]
    lead = (classes[:, 0] == _SIGN).astype(np.int64)
    body = np.arange(classes.shape[1])[None, :] >= lead[:, None]
    digit, end = body & (classes == _DIGIT), classes == _END
    grammatical[:] = digit.any(axis=1) & ~(body & ~digit & ~end).any(axis=1)
    grammatical &= ~((np.cumsum(end, axis=1) > 0) & ~end).any(axis=1)
    significant = (digit & (np.cumsum(digit & (matrix != ord("0")), axis=1) > 0)).sum(axis=1)
    readable[:] = grammatical & (significant <= 19)
    for k in np.flatnonzero(readable & (significant == 19)).tolist():  # 19 significant digits may still lie in int64
        readable[k] = -(2**63) <= int(selected[k]) < 2**63
    value[readable] = selected[readable].astype(np.int64)
    return value, grammatical, readable


def parse_integers(
    values: NDArray[np.bytes_], mask: NDArray[np.bool_], where: Callable[[int], SourceRef]
) -> NDArray[np.int64]:
    r"""
    The integers of the masked rows (``integer_tokens``), 0 outside the mask.

    Raises:
        ContradictionError: ``number_grammar`` for a token that is no optionally signed digit string.
        UnsupportedError: ``numeric_range`` for an integer outside int64.
    """
    out = np.zeros(len(values), dtype=np.int64)
    rows = np.flatnonzero(mask)
    value, grammatical, readable = integer_tokens(values[rows])
    for code, ok in (("number_grammar", grammatical), ("numeric_range", readable)):
        if not ok.all():
            fail(code, where(int(rows[np.flatnonzero(~ok)[0]])))
    out[rows] = value
    return out
