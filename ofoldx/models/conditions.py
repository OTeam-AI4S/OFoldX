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
Symbolic design conditions: generated placements, residue-identity and hotspot conditions, and covalent links between
existing or generated atoms (moved from ``data/placement.py``, ``data/conditions.py`` and ``data/bonds.py``; used by
the rfdiffusion, pxdesign, odesign and boltzgen families).

Existing positions are named by ``ofoldx.data.ComponentRef`` and ``AtomSite``; generated ones by a placement id and an
offset into its realised components. A ``SymbolicBond`` resolves its generated endpoints through a ``PlacementMap``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypeAlias

from ..data import AtomSite, ComponentRef
from ..errors import RequestRef, fail


@dataclass(frozen=True)
class GeneratedSegment:
    r"""
    One de novo polymer interval to be realised; ``min_length == max_length`` fixes the length (0 allowed).

    Attributes:
        polymer_type: Chemistry of the generated polymer.
        min_length: Minimum number of generated components.
        max_length: Maximum number of generated components.
    """

    polymer_type: Literal["protein", "rna", "dna", "dna_rna_hybrid", "pna", "other"]
    min_length: int
    max_length: int

    def __post_init__(self) -> None:
        if not 0 <= self.min_length <= self.max_length:
            fail("value_invalid", RequestRef("GeneratedSegment", f"{self.min_length}..{self.max_length}"))


@dataclass(frozen=True)
class ExistingSpan:
    r"""
    A half-open ``[start, stop)`` interval of an existing input instance.

    Attributes:
        instance_id: Source molecular instance.
        start: Zero-based index of the first included component.
        stop: Zero-based index after the last included component.
    """

    instance_id: str
    start: int
    stop: int

    def __post_init__(self) -> None:
        if not 0 <= self.start <= self.stop:
            fail("value_invalid", RequestRef("ExistingSpan", f"{self.start}..{self.stop}"))


#: A generated segment or copied source span.
PlacementValue: TypeAlias = GeneratedSegment | ExistingSpan

#: Ordered ``declaration_group -> placement_id -> value`` mappings; placement ids are globally unique.
PlacementSpec: TypeAlias = dict[str, dict[str, PlacementValue]]

#: Realised ``declaration_group -> placement_id -> component references``; a zero-length placement maps to ``()``.
PlacementMap: TypeAlias = dict[str, dict[str, tuple[ComponentRef, ...]]]


@dataclass(frozen=True)
class SymbolicResidueSite:
    r"""One residue of a placement before realisation: a placement id and a Python-style offset into it."""

    placement_id: str
    offset: int


@dataclass(frozen=True)
class SymbolicAtomSite:
    r"""One named atom of a symbolic residue."""

    residue: SymbolicResidueSite
    atom_id: str


def placement_instance_of(placement: PlacementSpec) -> dict[str, str]:
    r"""The declaration group of every placement id; an id in two groups is ``value_invalid``."""

    index: dict[str, str] = {}
    for group, placements in placement.items():
        for placement_id in placements:
            if placement_id in index:
                fail("value_invalid", RequestRef("placement_id", placement_id), groups=(index[placement_id], group))
            index[placement_id] = group
    return index


def validate_placement(placement: PlacementSpec) -> None:
    r"""Placement ids must be globally unique."""
    placement_instance_of(placement)


def resolve_placement(
    site: SymbolicResidueSite, placement_map: PlacementMap, instance_of: dict[str, str]
) -> ComponentRef:
    r"""The realised component a symbolic residue names."""
    occurrences = placement_map[instance_of[site.placement_id]][site.placement_id]
    if not -len(occurrences) <= site.offset < len(occurrences):
        fail("value_invalid", RequestRef(site.placement_id, str(site.offset)), length=len(occurrences))
    return occurrences[site.offset]


@dataclass(frozen=True)
class AllowedResidueIdentities:
    r"""Restrict residue positions to a set of chemical-component ids."""

    positions: tuple[ComponentRef, ...]
    component_ids: frozenset[str]


@dataclass(frozen=True)
class TiedResidueIdentities:
    r"""Tie groups of residue positions to the same chemical identity."""

    groups: tuple[tuple[ComponentRef, ...], ...]


#: An existing residue, a whole generated placement id, or a generated residue.
ResidueEndpoint: TypeAlias = ComponentRef | str | SymbolicResidueSite

#: An existing or generated atom.
AtomEndpoint: TypeAlias = AtomSite | SymbolicAtomSite

#: Any residue- or atom-granularity conditioning endpoint.
Endpoint: TypeAlias = ResidueEndpoint | AtomEndpoint


@dataclass(frozen=True)
class Hotspot:
    r"""A one-sided per-residue conditioning label: the residues to label as interaction hotspots."""

    residues: tuple[Endpoint, ...]


#: An existing atom or a generated atom with a symbolic residue reference.
SymbolicBondSite: TypeAlias = AtomSite | SymbolicAtomSite


def resolve_bond_site(site: SymbolicBondSite, placement_map: PlacementMap, instance_of: dict[str, str]) -> AtomSite:
    r"""Resolve a symbolic bond site; an existing atom site is returned unchanged."""
    if isinstance(site, AtomSite):
        return site
    reference = resolve_placement(site.residue, placement_map, instance_of)
    return AtomSite(reference.instance, reference.component, site.atom_id)


@dataclass(frozen=True, eq=False)
class SymbolicBond:
    r"""
    A covalent link between existing or symbolic endpoints, each with the atom ids that leave its component.

    Equality and hashing are unordered over ``(site, leaving)`` pairs for undirected bonds; directed bonds compare donor
    and acceptor in order. ``order`` and ``aromatic`` are the caller's statements (``None`` = not stated); their
    defaults state a single, non-aromatic bond (the frozen public signature; whether they become ``None`` is open).

    Examples:
        >>> a, b = AtomSite("A", 0, "C"), AtomSite("B", 0, "N")
        >>> SymbolicBond(a, b, leaving_a=frozenset({"OXT"})) == SymbolicBond(b, a, leaving_b=frozenset({"OXT"}))
        True
    """

    site_a: SymbolicBondSite
    site_b: SymbolicBondSite
    order: float | None = 1.0
    aromatic: bool | None = False
    stereo: str | None = None
    leaving_a: frozenset[str] = field(default_factory=frozenset)
    leaving_b: frozenset[str] = field(default_factory=frozenset)
    kind: Literal["covalent", "dative"] = "covalent"
    direction: Literal["undirected", "a_to_b", "b_to_a"] = "undirected"

    def __post_init__(self) -> None:
        if self.site_a == self.site_b:
            fail("value_invalid", RequestRef("SymbolicBond", repr(self.site_a)), invariant="two_atoms")

    @property
    def endpoints(self) -> frozenset[SymbolicBondSite]:
        r"""Both endpoints as an unordered set."""
        return frozenset((self.site_a, self.site_b))

    def _identity(self) -> tuple[object, ...]:
        endpoint_a, endpoint_b = (self.site_a, self.leaving_a), (self.site_b, self.leaving_b)
        if self.direction == "undirected":
            endpoints: object = frozenset((endpoint_a, endpoint_b))
        elif self.direction == "a_to_b":
            endpoints = (endpoint_a, endpoint_b)
        else:
            endpoints = (endpoint_b, endpoint_a)
        return endpoints, self.order, self.aromatic, self.kind, self.direction == "undirected", self.stereo

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SymbolicBond):
            return NotImplemented
        return self._identity() == other._identity()

    def __hash__(self) -> int:
        return hash(self._identity())

    def resolve(self, placement_map: PlacementMap, instance_of: dict[str, str]) -> SymbolicBond:
        r"""The same bond with both endpoints resolved to existing ``AtomSite``\ s."""
        return SymbolicBond(
            resolve_bond_site(self.site_a, placement_map, instance_of),
            resolve_bond_site(self.site_b, placement_map, instance_of),
            self.order,
            self.aromatic,
            self.stereo,
            self.leaving_a,
            self.leaving_b,
            self.kind,
            self.direction,
        )
