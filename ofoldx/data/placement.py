r"""Placement declarations and symbolic references.

A :data:`PlacementSpec` maps declaration groups to globally unique placement IDs.
References target placement IDs, while groups determine output grouping and order.
A :data:`PlacementMap` maps each placement to ordered realized component references;
symbolic offsets use standard Python indexing, including negative offsets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from .molecular import ComponentReference, PolymerType


@dataclass(frozen=True)
class GeneratedSegment:
    r"""One de novo polymer interval to be realized.

    ``min_length == max_length`` fixes the length, and ``min_length == 0`` is valid.
    The containing :data:`PlacementSpec` key supplies its identity.
    """

    polymer_type: PolymerType
    min_length: int
    max_length: int


@dataclass(frozen=True)
class ExistingSpan:
    r"""A half-open ``[start, stop)`` interval of an existing input instance.

    ``instance_id`` names the source instance.
    """

    instance_id: str
    start: int
    stop: int


#: A generated segment or copied source span.
PlacementValue: TypeAlias = GeneratedSegment | ExistingSpan

#: Ordered ``declaration_group -> placement_id -> value`` mappings. Placement IDs are
#: globally unique and are the symbolic reference targets.
PlacementSpec: TypeAlias = dict[str, dict[str, PlacementValue]]

#: Realized ``declaration_group -> placement_id -> component references`` mappings.
#: Every placement is present; a zero-length placement maps to ``()``.
PlacementMap: TypeAlias = dict[str, dict[str, tuple[ComponentReference, ...]]]


@dataclass(frozen=True)
class SymbolicResidueSite:
    r"""A pre-realization reference to one placement residue.

    ``placement_id`` is globally unique. ``offset`` uses Python indexing over the
    realized occurrence tuple, so negative values count from the end.
    """

    placement_id: str
    offset: int


@dataclass(frozen=True)
class SymbolicAtomSite:
    r"""One named atom of a pre-realization placement residue."""

    residue: SymbolicResidueSite
    atom_id: str


def realized_instance_id(index: int) -> str:
    r"""Return the 0-based realized instance ID ``chain_{index}``."""

    if index < 0:
        raise ValueError(f"realized instance index must be >= 0, got {index}")
    return f"chain_{index}"


class InstanceIdAllocator:
    r"""Allocate realized instance IDs in declaration order."""

    __slots__ = ("_next",)

    def __init__(self) -> None:
        self._next = 0

    def allocate(self) -> str:
        r"""Return the next ``chain_{n}`` and advance the counter."""

        instance_id = realized_instance_id(self._next)
        self._next += 1
        return instance_id


def instance_of(placement: PlacementSpec) -> dict[str, str]:
    r"""Build a ``placement_id -> declaration_group`` index.

    Duplicate placement IDs raise :class:`ValueError`.
    """

    index: dict[str, str] = {}
    for group, placements in placement.items():
        for placement_id in placements:
            existing = index.get(placement_id)
            if existing is not None:
                raise ValueError(
                    f"duplicate placement_id {placement_id!r}: declared under both "
                    f"{existing!r} and {group!r}; a placement_id must be globally unique "
                    f"across the whole PlacementSpec"
                )
            index[placement_id] = group
    return index


def validate_placement(placement: PlacementSpec) -> None:
    r"""Raise when a :data:`PlacementSpec` contains duplicate placement IDs."""

    instance_of(placement)


def resolve(
    site: SymbolicResidueSite,
    placement_map: PlacementMap,
    instance_of: dict[str, str],
) -> ComponentReference:
    r"""Resolve a symbolic site to its realized :data:`ComponentReference`.

    ``site.offset`` is applied directly to the placement's occurrence tuple.
    """

    occurrences = placement_map[instance_of[site.placement_id]][site.placement_id]
    return occurrences[site.offset]
