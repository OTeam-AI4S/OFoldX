r"""
Residue-identity and hotspot conditions, with shared endpoint references.
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

    Attributes:
        positions: Residue occurrences to constrain.
        component_ids: Component IDs allowed at every selected position.
    """

    positions: tuple[ComponentReference, ...]
    component_ids: frozenset[str]


@dataclass(frozen=True)
class TiedResidueIdentities:
    r"""
    Tie groups of residue positions to the same chemical identity.

    Attributes:
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

    Attributes:
        residues: Target endpoints to label as interaction hotspots. The generated
            partner is the whole design.
    """

    residues: tuple[Endpoint, ...]
