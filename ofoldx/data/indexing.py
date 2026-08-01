"""Maps between occurrence rows, molecular atom addresses and graph indices.

AtomSiteTable owns row-to-address facts. MolecularAtomIndex consumes these facts
when a materializer needs graph positions. Neither is cached on a mutable graph.
"""

from __future__ import annotations

import operator
from collections.abc import Iterable, Iterator, Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .._arrays import immutable_array
from .references import AtomSite, ComponentReference

if TYPE_CHECKING:
    from .molecular import MolecularSystem


class MolecularAtomIndex:
    """Snapshot of atom addresses in one CPU graph, indexed by component and name."""

    def __init__(self, system: MolecularSystem) -> None:
        if system.atom_symbol.device.type != "cpu":
            raise ValueError("molecular atom indexing requires a CPU graph")
        symbols = tuple(system.symbols)
        self._symbol_ids = {symbol: index for index, symbol in enumerate(symbols)}
        references: list[ComponentReference] = []
        for symbol, start, stop in zip(
            system.instance_symbol.tolist(),
            system.instance_component_ptr[:-1].tolist(),
            system.instance_component_ptr[1:].tolist(),
            strict=True,
        ):
            references.extend((symbols[symbol], index - start) for index in range(start, stop))
        self._component_references = tuple(references)
        self._component_indices = MappingProxyType({reference: index for index, reference in enumerate(references)})
        if len(self._component_indices) != len(references):
            raise ValueError("molecular graph has duplicate component identities")
        self._symbol_count = max(len(symbols), 1)
        if len(references) > np.iinfo(np.int64).max // self._symbol_count:
            raise OverflowError("molecular atom identity key exceeds int64")
        component_offsets = system.component_atom_ptr.numpy()
        components: NDArray[np.int64] = np.repeat(
            np.arange(len(references), dtype=np.int64), np.diff(component_offsets)
        )
        keys = components * self._symbol_count + system.atom_symbol.numpy()
        order = np.argsort(keys, kind="stable")
        self._sorted_keys = immutable_array(keys[order])
        self._atom_indices = immutable_array(order)
        if len(keys) > 1 and np.any(self._sorted_keys[1:] == self._sorted_keys[:-1]):
            raise ValueError("molecular component has duplicate atom names")

    @property
    def component_references(self) -> tuple[ComponentReference, ...]:
        return self._component_references

    @property
    def component_indices(self) -> Mapping[ComponentReference, int]:
        return self._component_indices

    def atom_indices(
        self, component_indices: ArrayLike, atom_ids: Iterable[str], *, allow_missing: bool = False
    ) -> NDArray[np.int64]:
        """Resolve graph component indices and atom names; optional misses are -1."""
        components = _integer_vector(component_indices, name="component indices")
        symbols: NDArray[np.int64] = np.fromiter((self._symbol_ids.get(name, -1) for name in atom_ids), dtype=np.int64)
        if components.shape != symbols.shape:
            raise ValueError("component and atom-name lookup columns differ")
        valid = (components >= 0) & (components < len(self._component_references)) & (symbols >= 0)
        keys = np.where(valid, components, 0) * self._symbol_count + np.where(valid, symbols, 0)
        positions = np.searchsorted(self._sorted_keys, keys)
        bounded = np.minimum(positions, max(len(self._sorted_keys) - 1, 0))
        if len(self._sorted_keys):
            valid &= (positions < len(self._sorted_keys)) & (self._sorted_keys[bounded] == keys)
        else:
            valid[:] = False
        result: NDArray[np.int64] = np.full(len(keys), -1, dtype=np.int64)
        result[valid] = self._atom_indices[positions[valid]]
        if not allow_missing and not valid.all():
            query = int(np.flatnonzero(~valid)[0])
            raise KeyError(f"atom query {query} references an unrealized molecular atom")
        return result

    def find_atom(self, site: AtomSite) -> int | None:
        component = self._component_indices.get((site.instance_id, site.component_index), -1)
        index = int(self.atom_indices([component], [site.atom_id], allow_missing=True)[0])
        return None if index < 0 else index

    def map_rows(self, sites: AtomSiteTable, row_ids: ArrayLike) -> NDArray[np.int64]:
        """Map occurrence-row identifiers to graph atoms, requiring every match."""
        positions = sites.positions(row_ids)
        components: NDArray[np.int64] = np.fromiter(
            (self._component_indices[reference] for reference in sites.components), dtype=np.int64
        )
        names = np.asarray(sites.atom_ids, dtype=object)[positions]
        return self.atom_indices(components[sites.component_codes[positions]], names)


class AtomSiteTable(Mapping[int, AtomSite]):
    """Owned occurrence-row → atom-address table, independent of graph storage.

    Row identifiers refer to the supplied occurrence table. After assembly they
    identify physical copies, so they must not be confused with deposited CIF rows.
    """

    def __init__(
        self,
        row_ids: ArrayLike,
        components: Sequence[ComponentReference],
        component_codes: ArrayLike,
        atom_ids: Sequence[str],
    ) -> None:
        self._row_ids = immutable_array(_integer_vector(row_ids, name="occurrence rows"))
        self._components = tuple(components)
        self._component_codes = immutable_array(_integer_vector(component_codes, name="component codes"))
        self._atom_ids = tuple(atom_ids)
        if not (len(self._row_ids) == len(self._component_codes) == len(self._atom_ids)):
            raise ValueError("atom-site columns have different lengths")
        if len(self._component_codes) and (
            self._component_codes.min() < 0 or self._component_codes.max() >= len(self._components)
        ):
            raise ValueError("atom-site component code is out of range")
        order = np.argsort(self._row_ids, kind="stable")
        self._sorted_rows = immutable_array(self._row_ids[order])
        self._row_positions = immutable_array(order)
        if len(order) > 1 and np.any(self._sorted_rows[1:] == self._sorted_rows[:-1]):
            raise ValueError("an occurrence row has multiple molecular identities")

    @property
    def row_ids(self) -> NDArray[np.int64]:
        return self._row_ids

    @property
    def components(self) -> tuple[ComponentReference, ...]:
        return self._components

    @property
    def component_codes(self) -> NDArray[np.int64]:
        return self._component_codes

    @property
    def atom_ids(self) -> tuple[str, ...]:
        return self._atom_ids

    def __len__(self) -> int:
        return len(self._row_ids)

    def __iter__(self) -> Iterator[int]:
        return map(int, self._row_ids)

    def positions(self, row_ids: ArrayLike) -> NDArray[np.int64]:
        """Resolve external occurrence identifiers to positions in these columns."""
        rows = _integer_vector(row_ids, name="occurrence rows")
        indices = np.searchsorted(self._sorted_rows, rows)
        if np.any(indices >= len(self)) or np.any(self._sorted_rows[indices] != rows):
            raise KeyError("unknown occurrence row")
        return self._row_positions[indices]

    def __getitem__(self, row: int) -> AtomSite:
        identifier = operator.index(row)
        sorted_position = int(np.searchsorted(self._sorted_rows, identifier))
        if sorted_position == len(self) or self._sorted_rows[sorted_position] != identifier:
            raise KeyError(identifier)
        position = int(self._row_positions[sorted_position])
        instance_id, component_index = self._components[int(self._component_codes[position])]
        return AtomSite(instance_id, component_index, self._atom_ids[position])


class AtomSiteTableBuilder:
    """Collect addresses by component and publish an independently owned table."""

    def __init__(self) -> None:
        self._row_ids: list[int] = []
        self._components: list[ComponentReference] = []
        self._component_codes: list[int] = []
        self._atom_ids: list[str] = []

    def add_component(self, reference: ComponentReference, row_ids: Iterable[int], atom_ids: Iterable[str]) -> None:
        rows, atoms = tuple(row_ids), tuple(atom_ids)
        if len(rows) != len(atoms):
            raise ValueError("occurrence row and atom-name columns differ")
        self._component_codes.extend([len(self._components)] * len(rows))
        self._components.append(reference)
        self._row_ids.extend(rows)
        self._atom_ids.extend(atoms)

    def build(self) -> AtomSiteTable:
        return AtomSiteTable(self._row_ids, self._components, self._component_codes, self._atom_ids)


def _integer_vector(values: ArrayLike, *, name: str) -> NDArray[np.int64]:
    array = np.asarray(values)
    if array.ndim != 1 or (array.size and array.dtype.kind not in {"i", "u"}):
        raise ValueError(f"{name} must be a one-dimensional integer array")
    if array.dtype.kind == "u" and np.any(array > np.iinfo(np.int64).max):
        raise ValueError(f"{name} exceed int64 range")
    return np.asarray(array, dtype=np.int64)
