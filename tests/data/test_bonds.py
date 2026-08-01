from __future__ import annotations

from ofoldx.data.bonds import SymbolicBond
from ofoldx.data.molecular import AtomSite


class TestSymbolicBond:
    def test_swap(self) -> None:
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

    def test_site_data(self) -> None:
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
