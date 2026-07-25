from __future__ import annotations

from ofoldx.data.bonds import SymbolicBond
from ofoldx.data.molecular import AtomSite


def test_symbolic_bond_endpoint_swap_preserves_identity() -> None:
    site_a = AtomSite("chain_0", 0, "C")
    site_b = AtomSite("chain_1", 0, "N")
    bond = SymbolicBond(
        site_a,
        site_b,
        leaving_a=frozenset({"OXT"}),
        leaving_b=frozenset(),
    )
    swapped = SymbolicBond(
        site_b,
        site_a,
        leaving_a=frozenset(),
        leaving_b=frozenset({"OXT"}),
    )

    assert bond == swapped
    assert hash(bond) == hash(swapped)


def test_symbolic_bond_distinguishes_site_data() -> None:
    site_a = AtomSite("chain_0", 0, "C")
    site_b = AtomSite("chain_1", 0, "N")
    bond = SymbolicBond(
        site_a,
        site_b,
        leaving_a=frozenset({"OXT"}),
        leaving_b=frozenset(),
    )

    assert bond != SymbolicBond(
        site_b,
        site_a,
        leaving_a=frozenset({"OXT"}),
        leaving_b=frozenset(),
    )
