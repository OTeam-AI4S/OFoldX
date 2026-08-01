"""Indexed atom occurrences; physical copies do not duplicate source records."""

from __future__ import annotations

import operator
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, SupportsIndex, cast, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .._arrays import immutable_array


@dataclass(frozen=True, eq=False)
class PlacementTransform:
    """One source-numbered placement and its owned Cartesian transform."""

    symmetry_id: str | None
    matrix: NDArray[np.float64]

    def __post_init__(self) -> None:
        matrix = immutable_array(self.matrix, np.float64)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError("placement transform must be a finite 4 by 4 matrix")
        if not np.array_equal(matrix[3], [0, 0, 0, 1]):
            raise ValueError("placement transform must be affine")
        object.__setattr__(self, "matrix", matrix)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PlacementTransform):
            return NotImplemented
        return self.symmetry_id == other.symmetry_id and np.array_equal(self.matrix, other.matrix)


class PlacedAtoms:
    """A gather plan over one atom table with deferred Cartesian transforms.

    Identity columns are gathered once on demand. Coordinates are transformed
    only when the coordinate consumer requests them. ``source_positions`` keeps
    every physical copy traceable to its original source row. Index and transform
    inputs are snapshotted; the source atom table is borrowed and must remain
    unchanged for the lifetime of this plan and its views.
    """

    def __init__(
        self,
        base: np.recarray,
        positions: ArrayLike,
        copies: ArrayLike,
        transforms: Sequence[PlacementTransform | tuple[str | None, np.ndarray]],
        row_ids: ArrayLike | None = None,
    ) -> None:
        self.base = base
        self.source_positions = _indices(positions)
        self.copy_indices = _indices(copies)
        self.transforms = tuple(
            transform if isinstance(transform, PlacementTransform) else PlacementTransform(*transform)
            for transform in transforms
        )
        self.row_ids = _indices(np.arange(len(self.source_positions), dtype=np.int64) if row_ids is None else row_ids)
        if len(self.copy_indices) != len(self.source_positions) or len(self.row_ids) != len(self.source_positions):
            raise ValueError("placement index columns have different lengths")
        if np.any(self.source_positions >= len(base)) or np.any(self.source_positions < 0):
            raise ValueError("placement source index is out of range")
        if np.any(self.copy_indices >= len(self.transforms)) or np.any(self.copy_indices < 0):
            raise ValueError("placement transform index is out of range")
        self._columns: dict[str, np.ndarray] = {}

    @classmethod
    def assembly(cls, rows: np.recarray, placements: Mapping[tuple[str | None, str], np.ndarray]) -> PlacedAtoms:
        by_chain: dict[str, list[int]] = {}
        for i, chain in enumerate(rows.chain_id):
            by_chain.setdefault(chain, []).append(i)
        positions: list[int] = []
        copies: list[int] = []
        transforms: list[tuple[str | None, np.ndarray]] = []
        for copy_index, ((symmetry, chain), transform) in enumerate(placements.items()):
            indices = by_chain.get(chain, ())
            positions.extend(indices)
            copies.extend([copy_index] * len(indices))
            transforms.append((symmetry, transform))
        return cls(rows, positions, copies, transforms)

    def select_source_rows(self, row_ids: ArrayLike) -> PlacedAtoms:
        return self[np.isin(self.base.row_index[self.source_positions], row_ids)]

    def __len__(self) -> int:
        return len(self.source_positions)

    @property
    def dtype(self) -> np.dtype:
        return self.base.dtype

    @overload
    def __getitem__(self, key: str) -> NDArray[Any]: ...

    @overload
    def __getitem__(self, key: int | np.integer) -> _AtomRow: ...

    @overload
    def __getitem__(self, key: slice | Sequence[int] | NDArray[Any]) -> PlacedAtoms: ...

    def __getitem__(
        self, key: str | int | np.integer | slice | Sequence[int] | NDArray[Any]
    ) -> NDArray[Any] | _AtomRow | PlacedAtoms:
        if isinstance(key, str):
            if key not in self._columns:
                values: NDArray[Any]
                if key == "row_index":
                    values = self.row_ids
                elif key == "symmetry_id":
                    names = np.asarray([transform.symmetry_id for transform in self.transforms], dtype=object)
                    values = names[self.copy_indices]
                elif key == "coordinates":
                    values = self._coordinates()
                else:
                    values = self.base[key][self.source_positions]
                if values.dtype.hasobject:
                    values.setflags(write=False)
                else:
                    values = immutable_array(values)
                self._columns[key] = values
            return self._columns[key]
        if np.isscalar(key):
            position = operator.index(cast(SupportsIndex, key))
            if not -len(self) <= position < len(self):
                raise IndexError(position)
            return _AtomRow(self, position)
        return type(self)(
            self.base, self.source_positions[key], self.copy_indices[key], self.transforms, self.row_ids[key]
        )

    def __getattr__(self, name: str) -> NDArray[Any]:
        if name in (self.base.dtype.names or ()):
            return self[name]
        raise AttributeError(name)

    def __iter__(self) -> Iterator[_AtomRow]:
        return (_AtomRow(self, i) for i in range(len(self)))

    def _coordinates(self) -> NDArray[np.float64]:
        values = self.base.coordinates[self.source_positions].copy()
        observed = self.base.coordinate_present[self.source_positions]
        positions = np.flatnonzero(observed)
        copies = self.copy_indices[positions]
        order = np.argsort(copies, kind="stable")
        positions = positions[order]
        boundaries = np.r_[0, np.cumsum(np.bincount(copies, minlength=len(self.transforms)))]
        for copy, placement in enumerate(self.transforms):
            transform = placement.matrix
            group = positions[boundaries[copy] : boundaries[copy + 1]]
            if len(group):
                values[group] = values[group] @ transform[:3, :3].T + transform[:3, 3]
        return values


def _indices(values: ArrayLike) -> NDArray[np.int64]:
    array = np.asarray(values)
    if array.ndim != 1 or (array.size and array.dtype.kind not in {"i", "u"}):
        raise ValueError("placement indices must be a one-dimensional integer array")
    if array.dtype.kind == "u" and np.any(array > np.iinfo(np.int64).max):
        raise ValueError("placement index is outside int64 range")
    return immutable_array(array, np.int64)


class _AtomRow:
    __slots__ = ("table", "position")

    def __init__(self, table: PlacedAtoms, position: int) -> None:
        self.table, self.position = table, position

    def __getattr__(self, name: str) -> Any:
        return self.table[name][self.position]

    def __getitem__(self, name: str) -> Any:
        return self.table[name][self.position]
