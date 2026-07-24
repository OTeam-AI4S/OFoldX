r"""
Covalent links between existing or generated atoms.

Symbolic bonds resolve generated endpoints through a `PlacementMap`. Endpoint
identity is unordered, while each leaving-atom set remains associated with its site.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias

from .molecular import AtomSite, SystemBond
from .placement import (
    PlacementMap,
    SymbolicAtomSite,
    resolve,
)

#: An existing atom or a generated atom with a symbolic residue reference.
SymbolicBondSite: TypeAlias = AtomSite | SymbolicAtomSite


def resolve_bond_site(
    site: SymbolicBondSite,
    placement_map: PlacementMap,
    instance_of: dict[str, str],
) -> AtomSite:
    r"""
    Resolve a symbolic bond site, or return an existing atom site unchanged.

    Args:
        site: Existing or symbolic atom site to resolve.
        placement_map: Realized component occurrences grouped by declaration and
            placement ID.
        instance_of: Mapping from placement IDs to declaration groups.

    Returns:
        The corresponding realized atom site.
    """

    if isinstance(site, AtomSite):
        return site
    instance_id, component_index = resolve(site.residue, placement_map, instance_of)
    return AtomSite(instance_id, component_index, site.atom_id)


@dataclass(frozen=True, eq=False)
class SymbolicBond:
    r"""
    A covalent bond with existing or symbolic endpoints.

    Equality and hashing are unordered over `(site, leaving_atoms)` pairs. Bond
    order, aromaticity, and stereochemistry are preserved by [`resolve`][].

    Args:
        site_a: First existing or symbolic bond endpoint.
        site_b: Second existing or symbolic bond endpoint.
        order: Bond order.
        aromatic: Whether the bond is aromatic.
        stereo: Bond stereochemistry, when specified.
        leaving_a: Atom IDs removed from the first endpoint's component.
        leaving_b: Atom IDs removed from the second endpoint's component.

    Raises:
        ValueError: If both endpoints identify the same atom.
    """

    site_a: SymbolicBondSite
    site_b: SymbolicBondSite
    order: float = 1.0
    aromatic: bool = False
    stereo: str | None = None
    leaving_a: frozenset[str] = field(default_factory=frozenset)
    leaving_b: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if self.site_a == self.site_b:
            raise ValueError("a symbolic bond cannot connect an atom to itself")

    @property
    def endpoints(self) -> frozenset[SymbolicBondSite]:
        r"""
        Return both endpoints as an unordered set.

        Returns:
            The two existing or symbolic atom sites.
        """

        return frozenset((self.site_a, self.site_b))

    def _identity(self) -> tuple[object, ...]:
        r"""
        Return the unordered equality and hash key.
        """

        return (
            frozenset(((self.site_a, self.leaving_a), (self.site_b, self.leaving_b))),
            self.order,
            self.aromatic,
            self.stereo,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SymbolicBond):
            return NotImplemented
        return self._identity() == other._identity()

    def __hash__(self) -> int:
        return hash(self._identity())

    def resolve(
        self,
        placement_map: PlacementMap,
        instance_of: dict[str, str],
    ) -> SystemBond:
        r"""
        Resolve both endpoints to a user-origin [`SystemBond`][].

        Args:
            placement_map: Realized component occurrences grouped by declaration and
                placement ID.
            instance_of: Mapping from placement IDs to declaration groups.

        Returns:
            The resolved covalent bond.
        """

        return SystemBond(
            site_a=resolve_bond_site(self.site_a, placement_map, instance_of),
            site_b=resolve_bond_site(self.site_b, placement_map, instance_of),
            order=self.order,
            aromatic=self.aromatic,
            stereo=self.stereo,
            origin="user",
        )
