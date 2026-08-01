r"""Read-only CIF source columns. Syntax is parsed once by Gemmi.

Missing items, unknown values, inapplicable values and quoted literals remain
separate. No structural selection, chemistry, atom-name inference or training
policy belongs in this module.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping, Sequence
from functools import cached_property
from typing import Any, cast

import gemmi
import numpy as np

from .._arrays import immutable_array
from ._numbers import integer, number_with_uncertainty

PRESENT = 0
INAPPLICABLE = 1
UNKNOWN = 2
_UNSPECIFIED = object()
_NULL_STATES = {"?": UNKNOWN, ".": INAPPLICABLE}
_CACHE_DTYPES = frozenset(np.dtype(dtype) for dtype in (np.float32, np.float64, np.int32, np.int64))


def _raw_tokens(values: Sequence[str | None | bool], *, raw: bool) -> tuple[str, ...]:
    """Normalize external values once; all columns store owned raw tokens."""
    if raw:
        raw_values = tuple(values)
        if any(not isinstance(token, str) or not token for token in raw_values):
            raise ValueError("raw CIF tokens must be non-empty strings")
        return cast(tuple[str, ...], raw_values)
    tokens: list[str] = []
    for value in values:
        if value is None:
            tokens.append("?")
        elif value is False:
            tokens.append(".")
        else:
            tokens.append(gemmi.cif.quote(str(value)))
    return tuple(tokens)


class CifColumn:
    """An owned raw column with lazy decoded, null-state and numeric views.

    Strings preserve whitespace and case. Raw tokens additionally retain quotes
    and the distinction between unknown, inapplicable and literal '?' / '.'.
    Numeric conversion requires unquoted numeric syntax and an explicit null fill.

    Args:
        values: Decoded strings, with None for unknown and False for inapplicable.
        raw: Whether values are original CIF tokens, including quotes and nulls.
        location: Source item description used in conversion errors.

    Examples:
        >>> column = CifColumn(["?", ".", "'?'", "' 001 '"], raw=True)
        >>> column.values
        (None, None, '?', ' 001 ')
        >>> column.mask.tolist()
        [2, 1, 0, 0]
        >>> numbers = CifColumn(["1.25", None])
        >>> numbers.as_array(float, -1).tolist()
        [1.25, -1.0]
        >>> CifColumn(["1.23e2(4)"]).uncertainty.tolist()
        [4.0]
    """

    def __init__(
        self,
        values: Sequence[str | None | bool],
        *,
        raw: bool = False,
        location: str = "<constructed column>",
    ) -> None:
        self._raw_values = _raw_tokens(values, raw=raw)
        self._location = location
        self._arrays: dict[str, np.ndarray] = {}

    @property
    def raw_values(self) -> tuple[str, ...]:
        """Original lexical tokens, retaining quotes and null markers."""
        return self._raw_values

    @cached_property
    def _decoded(self) -> tuple[str | None, ...]:
        values: list[str | None] = []
        for token in self.raw_values:
            if token in {"?", "."}:
                values.append(None)
            elif token[0] in "\"';":
                values.append(gemmi.cif.as_string(token))
            else:
                values.append(token)
        return tuple(values)

    @property
    def values(self) -> tuple[str | None, ...]:
        """Decoded text; use mask to distinguish the two null states."""
        return self._decoded

    @property
    def strings(self) -> tuple[str | None, ...]:
        return self._decoded

    @cached_property
    def _states(self) -> np.ndarray:
        return immutable_array(
            np.fromiter(
                (_NULL_STATES.get(token, PRESENT) for token in self.raw_values),
                dtype=np.uint8,
                count=len(self),
            )
        )

    @property
    def mask(self) -> np.ndarray:
        """Read-only states: PRESENT=0, INAPPLICABLE=1, UNKNOWN=2."""
        return self._states

    @cached_property
    def present(self) -> np.ndarray:
        return immutable_array(self.mask == PRESENT)

    def __len__(self) -> int:
        return len(self.raw_values)

    def _error(self, row: int | np.integer, error: Exception | str) -> ValueError:
        return ValueError(f"{self._location}, row {row + 1}: {error}")

    @cached_property
    def _floats(self) -> np.ndarray:
        # Native conversion handles the common case, including lexical validation.
        # Gemmi accepts an empty SU and only understands SU after the exponent;
        # the shared parser validates/decodes all bracketed forms explicitly.
        array = np.fromiter(map(gemmi.cif.as_number, self.raw_values), dtype=np.float64, count=len(self))
        for row, token in enumerate(self.raw_values):
            if "(" in token:
                try:
                    array[row] = number_with_uncertainty(token)[0]
                except ValueError as error:
                    raise self._error(row, error) from error
        invalid = self.present & ~np.isfinite(array)
        if invalid.any():
            row = int(np.flatnonzero(invalid)[0])
            raise self._error(row, f"invalid or unrepresentable CIF number {self.raw_values[row]!r}")
        return immutable_array(array)

    @cached_property
    def _integers(self) -> np.ndarray:
        array = np.zeros(len(self), dtype=np.int64)
        for row in np.flatnonzero(self.present):
            try:
                array[row] = integer(self.raw_values[row])
            except (ValueError, OverflowError) as error:
                raise self._error(row, error) from error
        return immutable_array(array)

    @cached_property
    def uncertainty(self) -> np.ndarray:
        """Standard uncertainties in the value's units, or NaN when undeclared."""
        # Access validates nominal values as well, even for a column without SU.
        self._floats
        array: np.ndarray = np.full(len(self), np.nan, dtype=np.float64)
        for row, token in enumerate(self.raw_values):
            if "(" in token:
                try:
                    array[row] = number_with_uncertainty(token)[1]
                except ValueError as error:
                    raise self._error(row, error) from error
        return immutable_array(array)

    @cached_property
    def uncertainty_present(self) -> np.ndarray:
        return immutable_array(np.isfinite(self.uncertainty))

    def as_array(self, dtype: Any = str, masked_value: Any = _UNSPECIFIED) -> np.ndarray:
        """Export decoded strings or validated numbers as a read-only array.

        Args:
            dtype: NumPy string, object, floating or integer dtype.
            masked_value: Explicit fill for null cells. Numeric conversion requires
                it when nulls occur; string conversion otherwise retains '?' or '.'.

        Returns:
            An array in source row order. Object exports are independent copies.

        Raises:
            ValueError: Nulls lack a numeric fill, or a present token cannot be
                converted to a finite number or an integer within the dtype range.
            TypeError: The requested dtype is unsupported.
        """
        target = np.dtype(dtype)
        if target.kind in {"U", "S", "O"}:
            return self._string_array(target, masked_value)
        return self._numeric_array(target, masked_value)

    def _string_array(self, target: np.dtype, masked_value: Any) -> np.ndarray:
        values = list(self.values)
        for row in np.flatnonzero(~self.present):
            if masked_value is _UNSPECIFIED:
                values[row] = self.raw_values[row]
            else:
                values[row] = masked_value
        array = np.asarray(values, dtype=target)
        if target.hasobject:
            # Object exports are independent, since object storage cannot have
            # an immutable byte owner. They are never retained by this column.
            array.setflags(write=False)
            return array
        return immutable_array(array)

    def _numeric_array(self, target: np.dtype, masked_value: Any) -> np.ndarray:
        if target.kind not in {"f", "i", "u"}:
            raise TypeError(f"unsupported CIF numeric dtype {target}")
        missing = not self.present.all()
        if missing and masked_value is _UNSPECIFIED:
            raise ValueError("numeric CIF conversion requires an explicit missing-value policy")
        canonical = self._floats if target.kind == "f" else self._integers
        canonical_fill = not missing or (
            masked_value is not _UNSPECIFIED
            and np.isscalar(masked_value)
            and ((target.kind == "f" and np.isnan(masked_value)) or (target.kind in {"i", "u"} and masked_value == 0))
        )
        # Only a fixed set of canonical projections is cached. Arbitrary fills
        # cannot accumulate a new full-column allocation on every call.
        cacheable = canonical_fill and target in _CACHE_DTYPES
        if cacheable and target.str in self._arrays:
            return self._arrays[target.str]
        if target.kind in {"i", "u"}:
            bounds = np.iinfo(target)
            if np.any(canonical[self.present] < bounds.min) or np.any(canonical[self.present] > bounds.max):
                raise ValueError(f"{self._location}: CIF integer outside {target} range")
        with np.errstate(over="ignore", under="ignore"):
            array = canonical.astype(target, copy=True)
        if target.kind == "f" and not np.isfinite(array[self.present]).all():
            raise ValueError(f"{self._location}: CIF number outside finite {target} range")
        if missing and not canonical_fill:
            array[~self.present] = masked_value
        result = immutable_array(array)
        if cacheable:
            self._arrays[target.str] = result
        return result

    def as_item(self) -> str:
        if len(self) != 1:
            raise ValueError("CIF item requires exactly one row")
        value = self.values[0]
        return value if value is not None else "." if self.mask[0] == INAPPLICABLE else "?"


class CifCategory(Mapping[str, CifColumn]):
    """One table; raw tokens are owned and columns are instantiated on demand."""

    def __init__(
        self,
        values: Mapping[str, Sequence[str | None | bool]],
        *,
        raw: bool = False,
        location: str = "<constructed category>",
    ) -> None:
        self._tokens = {name.casefold(): _raw_tokens(column, raw=raw) for name, column in values.items()}
        if len(self._tokens) != len(values):
            raise ValueError("duplicate case-insensitive CIF item")
        lengths = {len(column) for column in self._tokens.values()}
        if len(lengths) > 1:
            raise ValueError("CIF category columns have different lengths")
        self.row_count = next(iter(lengths), 0)
        self._location = location
        self._columns: dict[str, CifColumn] = {}

    def __getitem__(self, name: str) -> CifColumn:
        key = name.casefold()
        if key not in self._columns:
            self._columns[key] = CifColumn(self._tokens[key], raw=True, location=f"{self._location}.{key}")
        return self._columns[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._tokens)

    def __len__(self) -> int:
        return len(self._tokens)

    def strings(self, name: str) -> tuple[str | None, ...]:
        column = self.get(name)
        return (None,) * self.row_count if column is None else column.strings

    def value(self, name: str, row: int) -> str | None:
        column = self.get(name)
        return None if column is None else column.strings[row]


class CifBlock(Mapping[str, CifCategory]):
    """Lazy category access to an owned native block, with one decoded table per category."""

    def __init__(self, block: gemmi.cif.Block, *, source: str = "<memory>") -> None:
        self._source = source
        self._native = block
        self.name = block.name
        self._prefixes = {prefix[1:-1].casefold(): prefix for prefix in block.get_mmcif_category_names()}
        self._categories: dict[str, CifCategory] = {}

    def __getitem__(self, name: str) -> CifCategory:
        key = name.casefold()
        if key not in self._categories:
            self._categories[key] = CifCategory(
                self._native.get_mmcif_category(self._prefixes[key], raw=True),
                raw=True,
                location=f"{self._source}:data_{self.name}:_{key}",
            )
        return self._categories[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._prefixes)

    def __len__(self) -> int:
        return len(self._prefixes)


class CifDocument(Mapping[str, CifBlock]):
    """Owned source document. Unused categories remain in native storage.

    This is a source representation, not a chemically completed Structure. The
    document can be shared by consumers during one call without reparsing files.
    """

    def __init__(self, document: gemmi.cif.Document, *, name: str = "<memory>") -> None:
        self._native = document
        self.name = name
        self._blocks = {block.name: CifBlock(block, source=name) for block in document}

    def __getitem__(self, name: str) -> CifBlock:
        return self._blocks[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._blocks)

    def __len__(self) -> int:
        return len(self._blocks)

    @property
    def block(self) -> CifBlock:
        if len(self) != 1:
            raise ValueError(f"expected one CIF data block, got {len(self)} in {self.name}")
        return next(iter(self._blocks.values()))


def read_cif(path: str | os.PathLike[str]) -> CifDocument:
    """Read CIF/CIF.gz syntax and preserve every source category without preprocessing."""
    name = os.fspath(path)
    return CifDocument(gemmi.cif.read(name), name=name)


def parse_cif(text: str) -> CifDocument:
    r"""Parse in-memory CIF using the same native parser as [read_cif][ofoldx.io.cif.read_cif].

    Examples:
        >>> document = parse_cif("data_x\n_a.id '001'\n")
        >>> document.block["A"].value("ID", 0)
        '001'
    """
    return CifDocument(gemmi.cif.read_string(text))


def column_strings(category: CifCategory, name: str) -> tuple[str | None, ...]:
    """Read an exact optional string column, preserving quoted literals."""
    return category.strings(name)


def column_integers(category: CifCategory, name: str) -> tuple[int | None, ...]:
    """Read a schema integer item without converting opaque identifier columns."""
    column = category.get(name)
    if column is None:
        return (None,) * category.row_count
    values: np.ndarray = column.as_array(np.int64, 0).astype(object)
    values[~column.present] = None
    return tuple(values)


def column_points(
    category: CifCategory,
    x_name: str,
    y_name: str,
    z_name: str,
) -> tuple[tuple[float, float, float] | None, ...]:
    r"""One Cartesian column triple, with `None` where a row states no complete point."""

    columns = [category.get(name) for name in (x_name, y_name, z_name)]
    values = [
        np.full(category.row_count, np.nan) if column is None else column.as_array(np.float64, np.nan)
        for column in columns
    ]
    points = np.column_stack(values)
    return tuple(
        (float(point[0]), float(point[1]), float(point[2])) if np.isfinite(point).all() else None for point in points
    )
