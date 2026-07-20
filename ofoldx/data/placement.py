r"""Frozen placement core — the shared generation-topology identity model.

This module is the single canonical home of the §3.7 placement core, frozen
by the owner verdicts ratified 2026-07-18 (see ``DESIGN.md`` §3.7 "placement
core" and the rationale, four-model comparison, and worked examples A-C in
``CONTIG_IDENTITY.md``). The four generative families (RFdiffusion, ODesign, PXDesign,
BoltzGen) previously carried family-local placeholder copies of these records; they
now migrate onto these shared records.

Identity is **placement-keyed**:

* A PlacementSpec is an ordered ``declaration_group -> placement_id -> PlacementValue`` nested
  mapping. The outer key is a declaration-grouping and ordering label ONLY and is never
  named by any reference. The inner ``placement_id`` is **globally unique across the
  whole PlacementSpec** and is the SOLE pre-realization reference target.
* Identity lives in the mapping KEY, not in a value field, so the same ordinary
  ``GeneratedSegment`` / ``ExistingSpan`` value may occur more than once without
  acquiring a hidden id. ``GeneratedSegment`` therefore carries NO embedded id (today's
  ``segment_id`` is dropped).
* A pre-realization reference is exactly the two fields ``(placement_id, offset)`` — a
  RENAME of today's ``(segment_id, offset)``, not a widening. Because ``placement_id`` is
  globally unique it resolves without the outer group.
* The realized instance id is 100% family-owned and surfaces ONLY inside
  ``ComponentReference`` values in a ``placement_map``.

There is exactly one canonical way to express identity (the ``placement_id`` inner key)
and one to express a reference (``SymbolicResidueSite(placement_id, offset)``); there are
no aliases and no parallel positive/negative or ``instance_binding`` tables — those are
deleted, not migrated, because ``placement_map`` values are offset-indexed ordered tuples
with native negative indexing.

Length is re-homed per family: there is NO shared ``length_group`` / ``chain_lengths`` /
``total_len`` field on this core. Each family carries its own length argument
(RFdiffusion combined inpaint length, ODesign per-chain exact lengths, BoltzGen
whole-assembly ``total_len``) and resolves its own segment lengths at realization.

``@dataclass(frozen=True)`` denotes runtime immutability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from .molecular import ComponentReference, PolymerType


@dataclass(frozen=True)
class GeneratedSegment:
    r"""One de novo polymer interval to be realized.

    ``min_length == max_length`` fixes the length; ``min_length == 0`` is meaningful.
    Identity is the declaration-mapping KEY (the ``placement_id`` inner key of a
    :data:`PlacementSpec`), so there is deliberately NO embedded id field.
    """

    polymer_type: PolymerType
    min_length: int
    max_length: int


@dataclass(frozen=True)
class ExistingSpan:
    r"""A half-open ``[start, stop)`` interval of an existing SOURCE input-system instance.

    ``instance_id`` names the primary input system and the interval is copied verbatim
    from it.
    """

    instance_id: str
    start: int
    stop: int


#: One placement value carried under a ``placement_id`` inner key.
PlacementValue: TypeAlias = GeneratedSegment | ExistingSpan

#: An ordered ``declaration_group -> placement_id -> value`` nested mapping. The outer
#: key groups placements into a declared output instance and fixes declared order and is
#: NEVER a reference target; the inner ``placement_id`` is globally unique across the
#: whole PlacementSpec (validated by :func:`instance_of` / :func:`validate_placement`).
PlacementSpec: TypeAlias = dict[str, dict[str, PlacementValue]]

#: The realized output surface: ``declaration_group -> placement_id -> ordered tuple of
#: realized OUTPUT occurrences``. Outer/inner order is the declared order; every declared
#: placement appears (a zero-length placement maps to ``()``); each ``ComponentReference``
#: names a realized output occurrence (its family-owned realized id rides in
#: ``ComponentReference[0]``), so one declared placement may span more than one realized
#: instance id via symmetry/copy expansion.
PlacementMap: TypeAlias = dict[str, dict[str, tuple[ComponentReference, ...]]]


@dataclass(frozen=True)
class SymbolicResidueSite:
    r"""A pre-realization reference to one placement residue — two fields, one target.

    ``placement_id`` is the globally-unique inner :data:`PlacementSpec` key and resolves without
    the outer declaration group. ``offset >= 0`` counts from the placement start; ``offset
    < 0`` counts from its realized end (native Python from-end indexing into the realized
    occurrence tuple). A whole-placement reference is spelled as the bare ``placement_id``
    (families handle that directly); reusing a :class:`GeneratedSegment` value as an id is
    forbidden.
    """

    placement_id: str
    offset: int


@dataclass(frozen=True)
class SymbolicAtomSite:
    r"""One named atom of a pre-realization placement residue."""

    residue: SymbolicResidueSite
    atom_id: str


def realized_instance_id(index: int) -> str:
    r"""The canonical realized-instance id ``chain_{index}`` (uniform, 0-based, no family prefix).

    ``index`` is a 0-based integer assigned in DECLARATION order over every realized output
    chain that has at least one occurrence. This string surfaces ONLY inside
    ``ComponentReference[0]`` in a :data:`PlacementMap`; it is never a reference target
    (references use :class:`SymbolicResidueSite`), so its spelling is safe by contract.
    """

    if index < 0:
        raise ValueError(f"realized instance index must be >= 0, got {index}")
    return f"chain_{index}"


class InstanceIdAllocator:
    r"""Mints realized instance ids in declaration order per the canonical convention.

    Each call to :meth:`allocate` returns the next ``chain_{n}`` (``n`` starting at 0), so
    a family walks its realized output chains in declaration order and calls once per chain
    that has >=1 occurrence. A zero-length placement mints no chain and consumes no index:
    simply do not call :meth:`allocate` for it (map it to ``()`` in the :data:`PlacementMap`).
    A symmetry/copy-expanded placement calls once per realized copy, so each copy gets the
    next distinct index.
    """

    __slots__ = ("_next",)

    def __init__(self) -> None:
        self._next = 0

    def allocate(self) -> str:
        r"""Return the next ``chain_{n}`` and advance the counter."""

        instance_id = realized_instance_id(self._next)
        self._next += 1
        return instance_id


def instance_of(placement: PlacementSpec) -> dict[str, str]:
    r"""Derive the ``placement_id -> declaration_group`` index, enforcing global uniqueness.

    This is the construction-time uniqueness check: a ``placement_id`` that appears under
    more than one declaration group (or twice anywhere) raises :class:`ValueError`. Build
    it once from a :data:`PlacementSpec`; it is the trivial index a :class:`SymbolicResidueSite`
    uses to recover its group during :func:`resolve`.
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
    r"""Validate a :data:`PlacementSpec` at construction: raise on any duplicate ``placement_id``."""

    instance_of(placement)


def resolve(
    site: SymbolicResidueSite,
    placement_map: PlacementMap,
    instance_of: dict[str, str],
) -> ComponentReference:
    r"""Resolve a :class:`SymbolicResidueSite` to its realized ``ComponentReference``.

    ``placement_map[instance_of[site.placement_id]][site.placement_id][site.offset]`` with
    plain Python negative indexing for from-the-end offsets. ``instance_of`` is the derived
    index from :func:`instance_of`.
    """

    occurrences = placement_map[instance_of[site.placement_id]][site.placement_id]
    return occurrences[site.offset]
