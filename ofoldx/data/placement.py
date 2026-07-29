r"""
Declarations and symbolic references for generated molecular placements.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from .molecular import ComponentReference, PolymerType


@dataclass(frozen=True)
class GeneratedSegment:
    r"""
    One de novo polymer interval to be realized.

    ``min_length == max_length`` fixes the length, and zero-length segments are valid.

    Args:
        polymer_type: Chemistry of the generated polymer.
        min_length: Minimum number of generated components.
        max_length: Maximum number of generated components.

    Raises:
        ValueError: If the lengths do not satisfy ``0 <= min_length <= max_length``.
    """

    polymer_type: PolymerType
    min_length: int
    max_length: int

    def __post_init__(self) -> None:
        if self.min_length < 0:
            raise ValueError(f"min_length must be >= 0, got {self.min_length}")
        if self.max_length < self.min_length:
            raise ValueError(
                f"max_length must be >= min_length, got min_length={self.min_length}, "
                f"max_length={self.max_length}"
            )


@dataclass(frozen=True)
class ExistingSpan:
    r"""
    A half-open ``[start, stop)`` interval of an existing input instance.

    Args:
        instance_id: Source molecular instance.
        start: Zero-based index of the first included component.
        stop: Zero-based index after the last included component.

    Raises:
        ValueError: If the bounds do not satisfy ``0 <= start <= stop``.
    """

    instance_id: str
    start: int
    stop: int

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError(f"start must be >= 0, got {self.start}")
        if self.stop < self.start:
            raise ValueError(f"stop must be >= start, got start={self.start}, stop={self.stop}")


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
    r"""
    A pre-realization reference to one placement residue.

    Args:
        placement_id: Globally unique placement identifier.
        offset: Python-style index into the realized component occurrences. Negative
            values count from the end.
    """

    placement_id: str
    offset: int


@dataclass(frozen=True)
class SymbolicAtomSite:
    r"""
    One named atom of a pre-realization placement residue.

    Args:
        residue: Symbolic residue containing the atom.
        atom_id: Atom identifier within the residue.
    """

    residue: SymbolicResidueSite
    atom_id: str


def symbolic_residue_aliases(
    references: tuple[ComponentReference | SymbolicResidueSite, ...],
) -> dict[SymbolicResidueSite, SymbolicResidueSite]:
    r"""
    Map negative placement offsets to equivalent non-negative references.

    Args:
        references: Existing and symbolic residue references in realization order.

    Returns:
        A mapping from each valid negative symbolic offset to its non-negative form.
    """

    placements: dict[str, list[SymbolicResidueSite]] = {}
    for reference in references:
        if isinstance(reference, SymbolicResidueSite):
            placements.setdefault(reference.placement_id, []).append(reference)
    return {
        SymbolicResidueSite(placement_id, index - len(rows)): reference
        for placement_id, rows in placements.items()
        for index, reference in enumerate(rows)
    }


def realized_instance_id(index: int) -> str:
    r"""
    Return a realized instance ID for a zero-based index.

    Args:
        index: Non-negative instance index.

    Returns:
        The identifier `chain_{index}`.

    Raises:
        ValueError: If `index` is negative.
    """

    if index < 0:
        raise ValueError(f"realized instance index must be >= 0, got {index}")
    return f"chain_{index}"


class InstanceIdAllocator:
    r"""
    Allocate ``chain_{n}`` instance IDs in declaration order.
    """

    __slots__ = ("_next",)

    def __init__(self) -> None:
        self._next = 0

    def allocate(self) -> str:
        r"""
        Return the next realized instance ID and advance the allocator.

        Returns:
            The next `chain_{n}` identifier.
        """

        instance_id = realized_instance_id(self._next)
        self._next += 1
        return instance_id


def instance_of(placement: PlacementSpec) -> dict[str, str]:
    r"""
    Build an index from placement IDs to declaration groups.

    Args:
        placement: Placement declaration to index.

    Returns:
        The declaration group for every placement ID.

    Raises:
        ValueError: If a placement ID occurs in more than one declaration group.
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
    r"""
    Validate that placement IDs are globally unique.

    Args:
        placement: Placement declaration to validate.

    Raises:
        ValueError: If a placement ID occurs in more than one declaration group.
    """

    instance_of(placement)


def resolve(
    site: SymbolicResidueSite,
    placement_map: PlacementMap,
    instance_of: dict[str, str],
) -> ComponentReference:
    r"""
    Resolve a symbolic residue against a realized placement map.

    Args:
        site: Symbolic placement residue to resolve.
        placement_map: Realized component occurrences grouped by declaration and
            placement ID.
        instance_of: Mapping from placement IDs to declaration groups.

    Returns:
        The selected realized component occurrence.
    """

    occurrences = placement_map[instance_of[site.placement_id]][site.placement_id]
    return occurrences[site.offset]
