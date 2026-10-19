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
Template coordinates: polymer-chain coordinates projected from a read ``Structure``.

``chain_coordinates`` gathers, per requested polymer instance, every position of the instance (declared positions with
no atom that has coordinates included), its atoms with the coordinate mask, the component facts and, per position,
the canonical class with its encoding from ``token_classes``, so no consumer re-derives the class mapping.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from types import MappingProxyType
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ..data.molecular import EntityType
from ..data.structure import Structure
from ..data.tokenization import token_classes
from ..errors import RequestRef, StructureRef, fail
from ..provenance import known

#: The array fields of ``ChainCoordinates`` and their dtypes; the builder passes owned arrays of these dtypes.
_ARRAY_DTYPES = {
    "token_class": np.uint8,
    "encoding": np.uint8,
    "atom_name": np.int32,
    "component_index": np.int64,
    "coordinates": np.float64,
    "coordinate_mask": np.bool_,
}


@dataclass(frozen=True, eq=False)
class ChainCoordinates:
    r"""
    One polymer instance in sequence order, including declared positions with no atom that has coordinates.

    ``component_index`` addresses zero-based sequence positions, never author numbers. ``atom_name`` codes index
    ``atom_symbols``; ``coordinate_mask`` marks the rows that have coordinates, whatever their evidence (a row without
    coordinates is not a chemical absence).
    ``polymer_type`` is the entity's stated polymer type (``None`` when not stated); ``parent_component_ids`` are the
    definition's parents (those ``token_class`` maps by), empty where none is known.
    ``token_class``/``encoding`` are the positions' ``token_classes``. Arrays are owned read-only snapshots.
    """

    polymer_type: str | None
    component_ids: tuple[str, ...]
    token_class: NDArray[np.uint8]
    encoding: NDArray[np.uint8]
    atom_name: NDArray[np.int32]
    atom_symbols: tuple[str, ...]
    component_index: NDArray[np.int64]
    coordinates: NDArray[np.float64]
    coordinate_mask: NDArray[np.bool_]
    parent_component_ids: tuple[tuple[str, ...], ...]

    def __post_init__(self) -> None:
        for name, dtype in _ARRAY_DTYPES.items():
            array = getattr(self, name)
            if array.dtype != dtype:
                fail("value_invalid", RequestRef(name, str(array.dtype)), invariant=f"dtype_{np.dtype(dtype).name}")
            array.flags.writeable = False

    @cached_property
    def _rows(self) -> Mapping[tuple[int, str], int]:
        names = self.atom_symbols
        return MappingProxyType(
            {(int(p), names[int(a)]): row for row, (p, a) in enumerate(zip(self.component_index, self.atom_name))}
        )

    def gather(
        self, positions: Sequence[int], atom_names: Sequence[str | None]
    ) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
        r"""
        The named atoms' coordinates at ``positions`` (in order, repeats allowed) as independent arrays
        ``(len(positions), len(atom_names), 3)`` and the presence mask; a missing atom or one without coordinates, and
        a ``None`` slot, is zero with a false mask.
        """

        rows = np.array(
            [
                [self._rows.get((int(p), name), -1) if name is not None else -1 for name in atom_names]
                for p in positions
            ],
            dtype=np.int64,
        ).reshape(len(positions), len(atom_names))
        present = rows >= 0
        present[present] = self.coordinate_mask[rows[present]]
        coordinates = np.zeros((*rows.shape, 3), dtype=np.float64)
        coordinates[present] = self.coordinates[rows[present]]
        return coordinates, present


@dataclass(frozen=True, eq=False)
class PolymerCoordinates:
    r"""Polymer chain coordinates in one shared frame, by chain name (a read-only mapping); no bond graph is implied."""

    chains: Mapping[str, ChainCoordinates]


def template_atoms(
    chain: ChainCoordinates, position: int, atom_names: Sequence[str | None]
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    r"""The named atoms' coordinates at one position: coordinates ``(len(atom_names), 3)`` and the presence mask."""
    coordinates, present = chain.gather((position,), atom_names)
    return coordinates[0], present[0]


def chain_coordinates(
    structure: Structure, chains: Sequence[str], *, namespace: Literal["author", "label"] = "author"
) -> PolymerCoordinates:
    r"""
    The coordinates of the polymer instance of each chain, keyed by the chain name.

    In the author namespace a chain names the polymer instances with that author chain; in the label namespace, those
    of that label asym. A chain naming no polymer instance is ``InvalidRequestError(projection_invalid)``; one naming
    several (copies, or several polymer asyms under one author chain) is
    ``InsufficientInformationError(template_chain_ambiguous)``.
    """

    system, s = structure.system, structure.system.arrays()
    symbols = system.symbols()
    if namespace == "author":
        codes, present, family = structure.instance_author_chain, known(structure.instance_author_chain_state), "chain"
    elif namespace == "label":
        codes, family = structure.instance_source_asym, "asym"
        present = codes >= 0
    else:
        fail("value_invalid", RequestRef("namespace", str(namespace)))
    entity = s["instance_entity"]
    polymer = known(s["entity_type_state"][entity]) & (s["entity_type"][entity] == EntityType.POLYMER)
    lookup = {text: code for code, text in enumerate(structure.symbols()[family])}
    classes = token_classes(system)
    mask = structure.coordinate_mask
    component_ptr, atom_ptr = s["instance_component_ptr"], s["component_atom_ptr"]
    parent_ptr = s["component_definition_parent_ptr"]
    out = {}
    for chain in chains:
        instances = np.flatnonzero(present & polymer & (codes == lookup.get(chain, -2)))
        if not len(instances):
            fail("projection_invalid", RequestRef("chains", chain), namespace=namespace)
        if len(instances) > 1:
            fail(
                "template_chain_ambiguous",
                RequestRef("chains", chain),
                candidates=tuple(StructureRef(instance=int(i)) for i in instances),
            )
        instance = int(instances[0])
        start, stop = int(component_ptr[instance]), int(component_ptr[instance + 1])
        first, last = int(atom_ptr[start]), int(atom_ptr[stop])
        components = range(start, stop)
        polymer_type = int(s["entity_polymer_type"][entity[instance]])
        out[chain] = ChainCoordinates(
            polymer_type=symbols["polymer_type"][polymer_type] if polymer_type >= 0 else None,
            component_ids=tuple(symbols["component"][int(s["component_symbol"][c])] for c in components),
            token_class=classes.token[start:stop].astype(np.uint8),
            encoding=classes.encoding[start:stop].astype(np.uint8),
            atom_name=s["atom_symbol"][first:last].astype(np.int32),
            atom_symbols=symbols["atom"],
            component_index=np.repeat(np.arange(stop - start, dtype=np.int64), np.diff(atom_ptr[start : stop + 1])),
            coordinates=structure.coordinates[first:last].astype(np.float64),
            coordinate_mask=mask[first:last].astype(np.bool_),
            parent_component_ids=tuple(
                tuple(
                    symbols["parent"][int(p)]
                    for p in s["component_definition_parent"][parent_ptr[c] : parent_ptr[c + 1]]
                )
                for c in components
            ),
        )
    return PolymerCoordinates(MappingProxyType(out))
