r"""
Residue-identity, hotspot, and contact conditions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from .molecular import AtomSite, ComponentReference
from .placement import SymbolicAtomSite, SymbolicResidueSite


@dataclass(frozen=True)
class AllowedResidueIdentities:
    r"""
    Restrict residue positions to a set of chemical-component identities.

    Args:
        positions: Residue occurrences to constrain.
        component_ids: Component IDs allowed at every selected position.
    """

    positions: tuple[ComponentReference, ...]
    component_ids: frozenset[str]


@dataclass(frozen=True)
class TiedResidueIdentities:
    r"""
    Tie groups of residue positions to the same chemical identity.

    Args:
        groups: Position groups whose members must receive the same component ID.
    """

    groups: tuple[tuple[ComponentReference, ...], ...]


#: An existing residue, a whole generated placement ID, or a generated residue.
ResidueEndpoint: TypeAlias = ComponentReference | str | SymbolicResidueSite

#: An existing or generated atom.
AtomEndpoint: TypeAlias = AtomSite | SymbolicAtomSite

#: Any residue- or atom-granularity target-conditioning endpoint.
Endpoint: TypeAlias = ResidueEndpoint | AtomEndpoint


@dataclass(frozen=True)
class Hotspot:
    r"""
    A one-sided per-residue conditioning label.

    Args:
        residues: Target endpoints to label as interaction hotspots. The generated
            partner is the whole design.
    """

    residues: tuple[Endpoint, ...]


@dataclass(frozen=True, eq=False)
class Contact:
    r"""
    A pairwise proximity condition between two endpoint sets.

    Equality and hashing treat ``side_a`` and ``side_b`` as unordered.

    Args:
        side_a: Endpoints on one side of the requested contact.
        side_b: Endpoints on the other side of the requested contact.
        cutoff_angstrom: Maximum contact distance in angstroms.
        tolerance_angstrom: Additional accepted distance in angstroms.
    """

    side_a: tuple[Endpoint, ...]
    side_b: tuple[Endpoint, ...]
    cutoff_angstrom: float
    tolerance_angstrom: float = 0.0

    def _identity(self) -> tuple[object, ...]:
        return (
            frozenset((self.side_a, self.side_b)),
            self.cutoff_angstrom,
            self.tolerance_angstrom,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Contact):
            return NotImplemented
        return self._identity() == other._identity()

    def __hash__(self) -> int:
        return hash(self._identity())
