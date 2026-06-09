from __future__ import annotations

import pytest

from ofoldx.data import AtomRecord, ResidueKey, Structure, index_to_chain_id


@pytest.fixture
def structure() -> Structure:
    first = ResidueKey("A", 1)
    second = ResidueKey("A", 2)
    return Structure(
        (
            AtomRecord("N", "N", "ALA", first, 0.0, 0.0, 0.0),
            AtomRecord("CA", "C", "ALA", first, 1.0, 0.0, 0.0),
            AtomRecord("N", "N", "GLY", second, 2.0, 0.0, 0.0),
            AtomRecord("C", "C", "ALA", first, 3.0, 0.0, 0.0),
        )
    )


def test_residues(structure: Structure) -> None:
    # first-atom order with later repeats of an already-seen key dropped
    assert structure.residues == (ResidueKey("A", 1), ResidueKey("A", 2))


@pytest.mark.parametrize(
    ("atoms", "match"),
    [((), "at least one atom")],
)
def test_invalid_inputs(atoms: tuple[AtomRecord, ...], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        Structure(atoms)


def test_index_to_chain_id_matches_upstream() -> None:
    assert [index_to_chain_id(n) for n in (1, 2, 26, 27, 28)] == ["A", "B", "Z", "AA", "AB"]
