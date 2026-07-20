r"""Frozen pre-realization covalent-link core — ``SymbolicBond`` (§3.7.4).

This is the constraint-layer counterpart that rides on top of the now-frozen placement
core (:mod:`ofoldx.data.placement`) and the exact covalent surface
(:mod:`ofoldx.data.molecular`). A :class:`SymbolicBond` is a covalent link declared
BEFORE realization: unlike :class:`~ofoldx.data.molecular.SystemBond` (whose endpoints are
already-realized :class:`~ofoldx.data.molecular.AtomSite` occurrences), a
``SymbolicBond`` endpoint may be a *generated* atom named through the frozen placement
reference vocabulary (:class:`~ofoldx.data.placement.SymbolicAtomSite` over the two-field
:class:`~ofoldx.data.placement.SymbolicResidueSite`). Once realized, it lowers to a
``SystemBond`` via :meth:`SymbolicBond.resolve`.

Owner verdicts ratified 2026-07-19:

* **D1 — explicit leaving atoms.** A generated-atom endpoint has no finalized
  ``ChemicalComponent`` to carry :attr:`PolymerAttachment.leaving_atom_ids`, so the
  leaving atoms deleted on bond formation ride EXPLICITLY on the bond as
  ``leaving_a`` / ``leaving_b`` (``frozenset[str]`` of ``atom_id``). This is lossless and
  can always be re-derived by a family helper later; inference-only could not be added back
  without breaking identity.
* **D2 — unordered physical identity.** Identity is the UNORDERED endpoint set (a
  ``frozenset`` of the two endpoints), mirroring ``SystemBond`` (whose identity is enforced
  as a frozenset on ``MolecularSystem``). ``site_a`` / ``site_b`` positional order is
  retained ONLY to anchor per-endpoint leaving-atom attribution (``leaving_a`` pairs with
  ``site_a``, ``leaving_b`` with ``site_b``); it carries no directional chemistry.

There is deliberately **no** ``intent`` field: hard/soft is family-owned (it rides on the
family condition envelope, exactly as the placement freeze re-homed ``intent`` /
``chain_lengths``). Chemistry (``order`` / ``aromatic`` / ``stereo``) is kept so the
lowering to ``SystemBond`` is lossless — ODesign's hardcoded ``1.0`` / ``False`` becomes
explicit rather than lost.

``@dataclass(frozen=True)`` denotes runtime immutability.
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

#: One atom-granularity bond endpoint, promoted from the per-family ``AtomReference`` alias
#: (``rfdiffusion``/``odesign`` ``conditions.py``). Either an EXISTING realized atom
#: (:class:`~ofoldx.data.molecular.AtomSite`, whose ``(instance_id, component_index)`` IS a
#: :data:`~ofoldx.data.molecular.ComponentReference`) or a GENERATED atom
#: (:class:`~ofoldx.data.placement.SymbolicAtomSite` over the frozen two-field
#: :class:`~ofoldx.data.placement.SymbolicResidueSite`). The whole resolution chain is
#: closed by the placement freeze:
#: ``SymbolicAtomSite -> SymbolicResidueSite -> resolve() -> ComponentReference -> AtomSite``.
SymbolicBondSite: TypeAlias = AtomSite | SymbolicAtomSite


def resolve_bond_site(
    site: SymbolicBondSite,
    placement_map: PlacementMap,
    instance_of: dict[str, str],
) -> AtomSite:
    r"""Lower one :data:`SymbolicBondSite` endpoint to a realized :class:`AtomSite`.

    An :class:`AtomSite` endpoint is already realized and returned unchanged. A
    :class:`~ofoldx.data.placement.SymbolicAtomSite` endpoint resolves its
    :class:`~ofoldx.data.placement.SymbolicResidueSite` to a
    :data:`~ofoldx.data.molecular.ComponentReference` through the frozen placement
    :func:`~ofoldx.data.placement.resolve`, then re-attaches the named ``atom_id``.
    """

    if isinstance(site, AtomSite):
        return site
    instance_id, component_index = resolve(site.residue, placement_map, instance_of)
    return AtomSite(instance_id, component_index, site.atom_id)


@dataclass(frozen=True, eq=False)
class SymbolicBond:
    r"""A pre-realization covalent bond; endpoints reference the frozen placement model.

    Physical identity is the UNORDERED endpoint set (:attr:`endpoints`), mirroring
    :class:`~ofoldx.data.molecular.SystemBond`. ``site_a`` / ``site_b`` order is retained
    ONLY to anchor per-endpoint leaving-atom attribution: ``leaving_a`` names ``atom_id``
    values deleted from ``site_a`` on bond formation and ``leaving_b`` from ``site_b``.
    Each :class:`~ofoldx.data.placement.SymbolicAtomSite` endpoint resolves to a
    :class:`~ofoldx.data.molecular.AtomSite`; an ``AtomSite`` endpoint is already realized.
    The pair lowers to
    ``SystemBond(site_a, site_b, order, aromatic, stereo, origin="user")`` via
    :meth:`resolve` (leaving atoms are family-deleted from the realized components, not
    carried on ``SystemBond``, which has no leaving-atom concept).

    There is deliberately no ``intent`` field: hard/soft is family-owned.
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
        r"""The UNORDERED physical identity: the ``frozenset`` of both endpoints."""

        return frozenset((self.site_a, self.site_b))

    def _identity(self) -> tuple[object, ...]:
        r"""The UNORDERED equality/hash key (§3.7.4 D2).

        Endpoints are physically UNORDERED, so identity is the ``frozenset`` of the two
        ``(site, leaving)`` pairs — the leaving-atom set rides WITH its site, so swapping
        ``site_a`` / ``site_b`` (and correspondingly ``leaving_a`` / ``leaving_b``) names
        the same bond. Directionless chemistry (``order`` / ``aromatic`` / ``stereo``)
        completes the key.
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
        r"""Lower to a realized :class:`~ofoldx.data.molecular.SystemBond`.

        Each endpoint is resolved by :func:`resolve_bond_site`; the chemistry
        (``order`` / ``aromatic`` / ``stereo``) is carried through unchanged and ``origin``
        is fixed to ``"user"``. Leaving atoms (``leaving_a`` / ``leaving_b``) are NOT placed
        on the ``SystemBond`` (which has no leaving-atom concept); a family strips them from
        the realized components separately.
        """

        return SystemBond(
            site_a=resolve_bond_site(self.site_a, placement_map, instance_of),
            site_b=resolve_bond_site(self.site_b, placement_map, instance_of),
            order=self.order,
            aromatic=self.aromatic,
            stereo=self.stereo,
            origin="user",
        )
