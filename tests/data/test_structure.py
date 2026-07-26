from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import ofoldx.data as public_data
import ofoldx.data.structure as structure_module
from ofoldx.data.molecular import (
    AtomSite,
    ChemicalAtom,
    ChemicalBond,
    ChemicalComponent,
    MolecularSystem,
    Molecule,
)
from ofoldx.data.structure import ResidueNumber, Structure


def test_public_structure_is_the_canonical_class() -> None:
    assert public_data.Structure is structure_module.Structure


def _molecule_system(*, implicit_hydrogens: int = 0) -> MolecularSystem:
    atoms = [
        ChemicalAtom("C1", "C", implicit_hydrogens=implicit_hydrogens),
        ChemicalAtom("O1", "O"),
    ]
    bonds = [ChemicalBond("C1", "O1")]
    if implicit_hydrogens == 0:
        atoms.append(ChemicalAtom("H1", "H"))
        bonds.append(ChemicalBond("C1", "H1"))
    component = ChemicalComponent("LIG", atoms=tuple(atoms), bonds=frozenset(bonds))
    molecule = Molecule((component,))
    return MolecularSystem(entities={"ligand": molecule}, instances={"ligand_1": "ligand"})


def _system_sites(system: MolecularSystem) -> tuple[AtomSite, ...]:
    return tuple(site for site, _ in system.iter_atoms())


def _observed_structure(**overrides: Any) -> Structure:
    system = _molecule_system()
    sites = _system_sites(system)
    values: dict[str, Any] = {
        "system": system,
        "sites": (sites[1], sites[0]),
        "coordinates": np.array(
            [[1.0, 2.0, 3.0], [float("nan"), float("inf"), 9.0]],
            dtype=np.float32,
        ),
        "coordinate_mask": np.array([True, False], dtype=np.bool_),
    }
    values.update(overrides)
    return Structure(**values)


def test_observed_structure_retains_arrays_by_reference() -> None:
    # §3.5.1: the carrier retains arrays by reference and performs no defensive
    # copy, finiteness scan, or range validation of occupancy.
    coordinates = np.array([[1.0, 2.0, 3.0], [7.0, 8.0, 9.0]], dtype=np.float32)
    mask = np.array([True, False], dtype=np.bool_)
    occupancy = np.array([1.0, np.nan], dtype=np.float32)
    b_factor = np.array([0.0, np.nan], dtype=np.float32)
    system = _molecule_system()
    sites = _system_sites(system)

    structure = Structure(
        system=system,
        sites=(sites[1], sites[0]),
        coordinates=coordinates,
        coordinate_mask=mask,
        occupancy=occupancy,
        b_factor=b_factor,
    )

    assert structure.coordinates is coordinates
    assert structure.coordinate_mask is mask
    assert structure.occupancy is occupancy
    assert structure.b_factor is b_factor
    coordinates[0, 0] = -1.0
    assert structure.coordinates[0, 0] == -1.0
    assert np.isnan(structure.occupancy[1])  # type: ignore[index]
    assert structure.b_factor[0] == 0.0  # type: ignore[index]


def test_structure_obeys_explicit_dtypes_without_coercion() -> None:
    # §3.5.1: explicit user dtypes are obeyed; the carrier does not coerce dtype or
    # make arrays contiguous/read-only. A real integer occupancy kind is accepted.
    system = _molecule_system()
    sites = _system_sites(system)
    coordinates = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    occupancy = np.array([1, 0], dtype=np.int64)

    structure = Structure(
        system=system,
        sites=(sites[1], sites[0]),
        coordinates=coordinates,
        coordinate_mask=np.array([True, False], dtype=np.bool_),
        occupancy=occupancy,
    )

    assert structure.coordinates.dtype == np.float64
    assert structure.occupancy is occupancy
    assert structure.occupancy.dtype == np.int64  # type: ignore[union-attr]


def test_structure_has_object_identity_equality() -> None:
    # §3.5.1: Structure has object-identity equality and hashing; it has no NumPy-aware
    # custom equality (canonical value equality and deliberate unhashability were removed).
    first = _observed_structure()
    second = _observed_structure()

    assert first == first
    assert first != second
    assert hash(first) == hash(first)
    assert isinstance(hash(first), int)


def test_residue_numbers_pass_through_without_validation() -> None:
    # §3.5.1: residue_numbers is retained by reference and construction does not scan,
    # complete, copy, normalize, or validate it (producer/consumer contracts own coverage).
    system = _molecule_system()
    sites = _system_sites(system)
    residue_numbers = {("ligand_1", 0): ResidueNumber(7, "A")}

    structure = Structure(
        system=system,
        sites=(sites[1], sites[0]),
        coordinates=np.zeros((2, 3), dtype=np.float32),
        coordinate_mask=np.array([True, False], dtype=np.bool_),
        residue_numbers=residue_numbers,
    )

    assert structure.residue_numbers is residue_numbers
    assert _observed_structure().residue_numbers is None
    # An unresolved reference is retained verbatim; residue_numbers is never validated.
    unchecked = {("does_not_exist", 99): ResidueNumber(1)}
    passthrough = Structure(
        system=system,
        sites=(sites[0],),
        coordinates=np.zeros((1, 3), dtype=np.float32),
        coordinate_mask=np.ones(1, dtype=np.bool_),
        residue_numbers=unchecked,
    )
    assert passthrough.residue_numbers is unchecked


def test_unjustified_carrier_fields_are_absent() -> None:
    # §3.5.1 retains B factor for BoltzGen training, while the other generic
    # source-identity carriers remain unsupported.
    structure = _observed_structure()
    for removed in ("atom_layout", "altloc_ids", "source_identities"):
        assert not hasattr(structure, removed)
    assert not hasattr(structure_module, "SourceAtomIdentity")
    with pytest.raises(TypeError):
        _observed_structure(atom_layout="observed")


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    [
        ({"coordinates": np.zeros((1, 3), dtype=np.float32)}, ValueError, "coordinates must have shape"),
        ({"coordinate_mask": np.ones(1, dtype=np.bool_)}, ValueError, "coordinate_mask must have shape"),
        ({"coordinate_mask": np.zeros(2, dtype=np.bool_)}, ValueError, "at least one observed"),
        ({"occupancy": np.ones(1, dtype=np.float32)}, ValueError, "occupancy must have shape"),
        ({"b_factor": np.ones(1, dtype=np.float32)}, ValueError, "b_factor must have shape"),
        ({"coordinates": np.zeros((2, 3), dtype=np.bool_)}, TypeError, "coordinates must have a real numerical dtype"),
        ({"coordinate_mask": np.zeros(2, dtype=np.int64)}, TypeError, "coordinate_mask must have boolean dtype"),
        ({"occupancy": np.zeros(2, dtype=np.bool_)}, TypeError, "occupancy must have a real numerical dtype"),
        ({"b_factor": np.zeros(2, dtype=np.bool_)}, TypeError, "b_factor must have a real numerical dtype"),
    ],
)
def test_structure_rejects_invalid_arrays(overrides: dict[str, Any], error: type[Exception], message: str) -> None:
    with pytest.raises(error, match=message):
        _observed_structure(**overrides)


def test_structure_rejects_duplicate_and_unknown_sites() -> None:
    system = _molecule_system()
    site = _system_sites(system)[0]

    with pytest.raises(ValueError, match="sites must be unique"):
        Structure(
            system,
            (site, site),
            np.zeros((2, 3), dtype=np.float32),
            np.ones(2, dtype=np.bool_),
        )

    unknown = AtomSite("ligand_1", 0, "missing")
    with pytest.raises(KeyError):
        Structure(
            system,
            (unknown,),
            np.zeros((1, 3), dtype=np.float32),
            np.ones(1, dtype=np.bool_),
        )
