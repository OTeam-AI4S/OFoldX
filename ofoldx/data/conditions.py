r"""Small biological condition values shared by concrete model families.

Alongside the frozen §3.7.1 identity records (:class:`AllowedResidueIdentities` /
:class:`TiedResidueIdentities`) this module hosts the frozen target-conditioning layer
(§3.7.4, owner verdict D3 ratified 2026-07-19): the distance-free per-residue
:class:`Hotspot` label and the strictly-larger two-sided geometric :class:`Contact`.

The two are orthogonal, not one record with a toggle:

* :class:`Hotspot` carries NO distance — it is a single-sided per-residue label (the
  generated partner is implicit: the whole design). All four families already own the
  concept (RFdiffusion ``hotspot_mask``, PXDesign per-token bool, ODesign
  ``token_features[..., 452]``, BoltzGen ``binding``).
* :class:`Contact` carries a real ``cutoff_angstrom`` — a two-sided geometric proximity
  constraint. A ``Contact`` degenerated to one side would still carry a cutoff every label
  consumer ignores, so a hotspot is NOT a degenerate ``Contact``: ``Hotspot`` is the
  primitive and ``Contact`` the strictly larger record.

Both records reuse the placement-freeze reference vocabulary via :data:`Endpoint`; neither
invents a new way to name a residue/atom. Family-owned extensions stay OFF the frozen
core: ``weight`` (px/od), directional ``kind`` (BoltzGen ``pocket>binder`` /
``binder>pocket``), the negative ``not_binding`` label (BoltzGen), the allowed
cutoff/threshold *values*, and hard/soft ``intent`` all ride on the family envelope.
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


#: One residue-granularity endpoint: an EXISTING realized residue
#: (:data:`~ofoldx.data.molecular.ComponentReference` == ``(instance_id, component_index)``),
#: a WHOLE generated placement (the bare globally-unique ``placement_id`` ``str``), or one
#: GENERATED residue (:class:`~ofoldx.data.placement.SymbolicResidueSite`).
ResidueEndpoint: TypeAlias = ComponentReference | str | SymbolicResidueSite

#: One atom-granularity endpoint — the superset for families that reach individual atoms
#: (identical to :data:`~ofoldx.data.bonds.SymbolicBondSite`).
AtomEndpoint: TypeAlias = AtomSite | SymbolicAtomSite

#: Any residue- or atom-granularity target-conditioning endpoint.
Endpoint: TypeAlias = ResidueEndpoint | AtomEndpoint


@dataclass(frozen=True)
class Hotspot:
    r"""A per-residue TARGET-CONDITIONING label — no distance, single-sided.

    ``residues`` name the target residues to condition on; the generated partner is implicit
    (the whole design). Endpoints reference the frozen placement model via :data:`Endpoint`
    and MAY be non-protein (e.g. an ODesign ligand ``ComponentReference`` such as ``E/505``).
    Lowered per-family to a per-token signal (RFdiffusion ``hotspot_mask`` / PXDesign
    per-token bool / ODesign ``token_features[..., 452]`` / BoltzGen ``binding``). Carries no
    ``cutoff`` — that is what distinguishes it from :class:`Contact`.
    """

    residues: tuple[Endpoint, ...]


@dataclass(frozen=True, eq=False)
class Contact:
    r"""A GEOMETRIC pairwise proximity constraint between two endpoint sets.

    Identity is the distance ``cutoff_angstrom`` — a ``Contact`` without a real cutoff is a
    :class:`Hotspot`, not a ``Contact``. ``side_a`` / ``side_b`` name the two contacting
    endpoint sets; the constraint is symmetric (swapping the sides names the same physical
    contact), so equality and hashing use the UNORDERED ``{side_a, side_b}`` pair. Endpoints
    reference the placement model via :data:`Endpoint` exactly as the frozen core dictates.
    Family-owned extensions (``weight``, directional ``kind``, the allowed cutoff/threshold
    *values*, hard/soft ``intent``) stay OFF this frozen core.
    """

    side_a: tuple[Endpoint, ...]
    side_b: tuple[Endpoint, ...]
    cutoff_angstrom: float
    tolerance_angstrom: float = 0.0

    def _identity(self) -> tuple[object, ...]:
        r"""The UNORDERED equality/hash key: ``{side_a, side_b}`` plus the distances.

        Swapping the two sides names the same physical contact, so identity is the
        ``frozenset`` of both sides (each an ordered ``tuple`` of endpoints) together with
        the geometry (``cutoff_angstrom`` / ``tolerance_angstrom``).
        """

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
