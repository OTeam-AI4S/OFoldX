from __future__ import annotations

from ofoldx.data.bonds import (
    MAX_COVALENT_BOND_LENGTH,
    bond_length,
    inter_chain_bonds,
    validate_covalent_bonds,
)
from ofoldx.data.structure import AtomRecord, ResidueKey, Structure


def _atom(name: str, chain: str, number: int, x: float, y: float = 0.0, z: float = 0.0) -> AtomRecord:
    return AtomRecord(
        name=name,
        element=name[0],
        residue_name="CYS",
        residue=ResidueKey(chain_id=chain, residue_number=number),
        x=x,
        y=y,
        z=z,
    )


def test_bond_length_is_euclidean() -> None:
    a = _atom("SG", "A", 1, 0.0)
    b = _atom("SG", "B", 2, 3.0, 4.0, 0.0)
    assert bond_length(a, b) == 5.0


def test_short_bond_kept_long_bond_dropped() -> None:
    # One 2.0 A bond (kept) and one 5.0 A bond (dropped by the length gate).
    structure = Structure(
        (
            _atom("SG", "A", 1, 0.0),
            _atom("SG", "A", 2, 2.0),
            _atom("SG", "A", 3, 7.0),
        ),
        bonds=((0, 1), (0, 2)),
    )
    validated = validate_covalent_bonds(structure)
    assert len(validated) == 1
    assert validated[0].first_atom_index == 0 and validated[0].second_atom_index == 1
    assert validated[0].length == 2.0


def test_inter_chain_flag_set_by_chain_id() -> None:
    structure = Structure(
        (
            _atom("SG", "A", 1, 0.0),
            _atom("SG", "B", 1, 2.0),  # different chain, within cutoff
            _atom("SG", "A", 2, 1.0),  # same chain
        ),
        bonds=((0, 1), (0, 2)),
    )
    validated = validate_covalent_bonds(structure)
    flags = {(bond.first_atom_index, bond.second_atom_index): bond.inter_chain for bond in validated}
    assert flags[(0, 1)] is True
    assert flags[(0, 2)] is False


def test_inter_chain_bonds_returns_only_cross_chain() -> None:
    structure = Structure(
        (
            _atom("SG", "A", 1, 0.0),
            _atom("SG", "B", 1, 2.0),
            _atom("SG", "A", 2, 1.0),
        ),
        bonds=((0, 1), (0, 2)),
    )
    cross = inter_chain_bonds(structure)
    assert len(cross) == 1
    assert cross[0].inter_chain is True
    assert (cross[0].first_atom_index, cross[0].second_atom_index) == (0, 1)


def test_custom_cutoff_respected() -> None:
    structure = Structure(
        (_atom("SG", "A", 1, 0.0), _atom("SG", "A", 2, MAX_COVALENT_BOND_LENGTH + 0.5)), bonds=((0, 1),)
    )
    assert validate_covalent_bonds(structure) == []
    assert len(validate_covalent_bonds(structure, max_bond_length=10.0)) == 1
