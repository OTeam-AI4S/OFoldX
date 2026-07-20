from __future__ import annotations

from ofoldx.data.bonds import SymbolicBond
from ofoldx.data.conditions import Contact
from ofoldx.data.molecular import AtomSite

# These tests lock the frozen §3.7.4 UNORDERED physical identity of the two
# pre-realization constraint records: a SymbolicBond and a Contact each name a
# symmetric relation, so swapping the two endpoints/sides (carrying each
# endpoint's per-site data along) is the SAME record for both == and hash().


def test_symbolic_bond_swap_is_equal_with_leaving_atoms_following_sites() -> None:
    a = AtomSite("chain_0", 0, "C")
    b = AtomSite("chain_1", 0, "N")
    # The leaving-atom set is per-site, so a valid swap moves leaving_a<->leaving_b
    # together with site_a<->site_b: the same physical bond.
    forward = SymbolicBond(a, b, leaving_a=frozenset({"OXT"}), leaving_b=frozenset())
    swapped = SymbolicBond(b, a, leaving_a=frozenset(), leaving_b=frozenset({"OXT"}))
    assert forward == swapped
    assert hash(forward) == hash(swapped)
    assert len({forward, swapped}) == 1


def test_symbolic_bond_stays_unequal_when_genuinely_different() -> None:
    a = AtomSite("chain_0", 0, "C")
    b = AtomSite("chain_1", 0, "N")
    base = SymbolicBond(a, b, leaving_a=frozenset({"OXT"}), leaving_b=frozenset())
    # Swapping the sites WITHOUT swapping the per-site leaving atoms is a
    # different bond (OXT now leaves from N, not C).
    assert base != SymbolicBond(b, a, leaving_a=frozenset({"OXT"}), leaving_b=frozenset())
    # Directionless chemistry is part of identity.
    assert SymbolicBond(a, b, order=2.0) != SymbolicBond(a, b, order=1.0)
    assert SymbolicBond(a, b, aromatic=True) != SymbolicBond(a, b, aromatic=False)
    assert SymbolicBond(a, b, stereo="E") != SymbolicBond(a, b, stereo=None)
    # A different endpoint is a different bond.
    assert base != SymbolicBond(a, AtomSite("chain_2", 0, "N"), leaving_a=frozenset({"OXT"}))


def test_contact_side_swap_is_equal_and_hashes_equal() -> None:
    side_a = (("chain_0", 0),)
    side_b = (("chain_1", 0),)
    forward = Contact(side_a, side_b, cutoff_angstrom=8.0)
    swapped = Contact(side_b, side_a, cutoff_angstrom=8.0)
    assert forward == swapped
    assert hash(forward) == hash(swapped)
    assert len({forward, swapped}) == 1


def test_contact_stays_unequal_when_geometry_differs() -> None:
    side_a = (("chain_0", 0),)
    side_b = (("chain_1", 0),)
    base = Contact(side_a, side_b, cutoff_angstrom=8.0)
    assert base != Contact(side_a, side_b, cutoff_angstrom=6.0)
    assert base != Contact(side_a, side_b, cutoff_angstrom=8.0, tolerance_angstrom=1.0)
    # A different endpoint set is a different contact.
    assert base != Contact(side_a, (("chain_2", 0),), cutoff_angstrom=8.0)
