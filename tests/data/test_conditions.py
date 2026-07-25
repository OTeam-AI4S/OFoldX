from __future__ import annotations

from ofoldx.data.conditions import Contact


def test_contact_side_swap_preserves_identity() -> None:
    side_a = (("chain_0", 0),)
    side_b = (("chain_1", 0),)
    contact = Contact(side_a, side_b, cutoff_angstrom=8.0)
    swapped = Contact(side_b, side_a, cutoff_angstrom=8.0)

    assert contact == swapped
    assert hash(contact) == hash(swapped)


def test_contact_distinguishes_geometry() -> None:
    side_a = (("chain_0", 0),)
    side_b = (("chain_1", 0),)

    assert Contact(side_a, side_b, cutoff_angstrom=8.0) != Contact(
        side_a,
        side_b,
        cutoff_angstrom=6.0,
    )
