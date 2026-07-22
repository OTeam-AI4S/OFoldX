r"""Shared residue identity, hotspot, and contact conditions.

A :class:`Hotspot` is a one-sided label without a distance. A :class:`Contact`
is a symmetric two-sided proximity condition with a distance cutoff. Model-specific
weights and enforcement settings remain in model inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from .molecular import AtomSite, ComponentReference
from .placement import SymbolicAtomSite, SymbolicResidueSite


@dataclass(frozen=True)
class AllowedResidueIdentities:
    r"""Chemical-component identities allowed at exact residue positions."""

    positions: tuple[ComponentReference, ...]
    component_ids: frozenset[str]


@dataclass(frozen=True)
class TiedResidueIdentities:
    r"""Groups of exact residue positions that must share chemical identity."""

    groups: tuple[tuple[ComponentReference, ...], ...]


#: An existing residue, a whole generated placement ID, or a generated residue.
ResidueEndpoint: TypeAlias = ComponentReference | str | SymbolicResidueSite

#: An existing or generated atom.
AtomEndpoint: TypeAlias = AtomSite | SymbolicAtomSite

#: Any residue- or atom-granularity target-conditioning endpoint.
Endpoint: TypeAlias = ResidueEndpoint | AtomEndpoint


@dataclass(frozen=True)
class Hotspot:
    r"""A one-sided per-residue conditioning label.

    ``residues`` names target endpoints; the generated partner is the whole design.
    """

    residues: tuple[Endpoint, ...]


@dataclass(frozen=True, eq=False)
class Contact:
    r"""A pairwise proximity condition between two endpoint sets.

    Equality and hashing treat ``side_a`` and ``side_b`` as unordered. The cutoff and
    tolerance are part of identity.
    """

    side_a: tuple[Endpoint, ...]
    side_b: tuple[Endpoint, ...]
    cutoff_angstrom: float
    tolerance_angstrom: float = 0.0

    def _identity(self) -> tuple[object, ...]:
        r"""Return the unordered sides and distance values used for identity."""

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
