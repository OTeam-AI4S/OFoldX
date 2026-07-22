from __future__ import annotations

import json

import numpy as np
import pytest

from ofoldx.data.codec import decode_structure, decode_system, encode_structure, encode_system
from ofoldx.data.components import get_component
from ofoldx.data.molecular import (
    AtomSite,
    ChemicalAtom,
    ChemicalBond,
    ChemicalComponent,
    MolecularSystem,
    Molecule,
    PolymerSequence,
    SystemBond,
)
from ofoldx.data.structure import ResidueNumber, Structure


def _heterogeneous_structure() -> Structure:
    mse = ChemicalComponent(
        "MSE",
        atoms=(
            ChemicalAtom("N", "N"),
            ChemicalAtom("CA", "C"),
            ChemicalAtom("SE", "Se"),
        ),
        bonds=frozenset({ChemicalBond("N", "CA"), ChemicalBond("CA", "SE")}),
        reference_coordinates=((0.0, 0.0, 0.0), (1.4, 0.0, 0.0), (2.0, 1.0, 0.0)),
    )
    polymer = PolymerSequence("protein", (mse,))
    ligand_left = ChemicalComponent(
        "LFT",
        atoms=(ChemicalAtom("C1", "C"), ChemicalAtom("N1", "N")),
        bonds=frozenset({ChemicalBond("C1", "N1")}),
        reference_coordinates=((0.0, 0.0, 0.0), (1.3, 0.0, 0.0)),
    )
    ligand_right = ChemicalComponent(
        "RGT",
        atoms=(ChemicalAtom("N1", "N"), ChemicalAtom("O1", "O")),
        bonds=frozenset({ChemicalBond("N1", "O1")}),
        reference_coordinates=((2.5, 0.0, 0.0), (3.7, 0.0, 0.0)),
    )
    ligand = Molecule((ligand_left, ligand_right))
    covalent = SystemBond(
        AtomSite("L", 0, "N1"),
        AtomSite("L", 1, "N1"),
        origin="source",
    )
    system = MolecularSystem(
        entities={"target": polymer, "ligand": ligand},
        instances={"A": "target", "L": "ligand"},
        bonds=frozenset({covalent}),
    )
    sites = tuple(site for site, _ in system.iter_atoms())
    coordinates = np.arange(len(sites) * 3, dtype=np.float32).reshape(len(sites), 3)
    return Structure(
        system=system,
        sites=sites,
        coordinates=coordinates,
        coordinate_mask=np.ones(len(sites), dtype=np.bool_),
        occupancy=np.linspace(0.5, 1.0, len(sites), dtype=np.float32),
        residue_numbers={("A", 0): ResidueNumber(7, "A")},
    )


def test_system_codec_is_deterministic_known_type_json() -> None:
    structure = _heterogeneous_structure()
    first = encode_system(structure.system)
    second = encode_system(structure.system)

    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert set(first) == {"kind", "entities", "instances", "bonds"}
    assert set(first["entities"]) == {"target", "ligand"}  # type: ignore[arg-type]
    molecule = first["entities"]["ligand"]  # type: ignore[index]
    assert molecule["kind"] == "molecule"
    assert set(molecule) == {"kind", "components", "bonds"}
    assert [component["component_id"] for component in molecule["components"]] == ["LFT", "RGT"]
    # instances encode as an ORDERED [instance_id, entity_id] array so sort_keys cannot
    # scramble declaration order on the mmCIF write path.
    assert first["instances"] == [["A", "target"], ["L", "ligand"]]
    for bond in first["bonds"]:  # type: ignore[union-attr]
        for endpoint in (bond["site_a"], bond["site_b"]):
            assert set(endpoint) == {"instance_id", "component_index", "atom_id"}
            assert type(endpoint["component_index"]) is int

    restored = decode_system(json.loads(json.dumps(first)))
    # MolecularSystem has no value equality and the codec canonicalizes undirected bond
    # endpoint order, so round-trip is proven at the deterministic serialized level.
    assert encode_system(restored) == first
    assert restored.instances == structure.system.instances
    restored_mse = restored.component(("A", 0))
    assert restored_mse.reference_coordinates == structure.system.component(("A", 0)).reference_coordinates
    restored_ligand = restored.entities["ligand"]
    assert type(restored_ligand) is Molecule
    assert tuple(component.component_id for component in restored_ligand.components) == ("LFT", "RGT")
    assert restored_ligand.components[1].reference_coordinates == ((2.5, 0.0, 0.0), (3.7, 0.0, 0.0))
    restored_bond = next(iter(restored.bonds))
    assert {restored_bond.site_a.component_index, restored_bond.site_b.component_index} == {0, 1}
    assert restored_bond.origin == "source"


def test_released_nucleic_components_round_trip() -> None:
    rna = PolymerSequence("rna", tuple(get_component(component_id) for component_id in ("A", "C", "G", "U", "N")))
    dna = PolymerSequence("dna", tuple(get_component(component_id) for component_id in ("DA", "DC", "DG", "DT", "DN")))
    system = MolecularSystem(
        entities={"rna": rna, "dna": dna},
        instances={"R": "rna", "D": "dna"},
    )

    encoded = encode_system(system)
    restored = decode_system(json.loads(json.dumps(encoded)))

    # The codec round-trip is idempotent at the serialized level even though undirected
    # ChemicalBond declaration direction is not part of frozen value equality.
    assert encode_system(restored) == encoded
    assert restored.instances == system.instances
    restored_components = {
        component.component_id: component for entity in restored.entities.values() for component in entity.components
    }
    for component_id in ("A", "C", "G", "U", "N", "DA", "DC", "DG", "DT", "DN"):
        component = restored_components[component_id]
        audited = get_component(component_id)
        assert component.reference_coordinates == audited.reference_coordinates


def test_structure_codec_round_trips_arrays_and_residue_numbers() -> None:
    structure = _heterogeneous_structure()
    payload, arrays = encode_structure(structure, array_prefix="structure_0")
    payload = json.loads(json.dumps(payload))
    restored = decode_structure(payload, arrays)

    assert restored.sites == structure.sites
    np.testing.assert_array_equal(restored.coordinates, structure.coordinates)
    np.testing.assert_array_equal(restored.coordinate_mask, structure.coordinate_mask)
    assert restored.occupancy is not None
    np.testing.assert_array_equal(restored.occupancy, structure.occupancy)
    assert restored.residue_numbers == structure.residue_numbers
    assert encode_system(restored.system) == encode_system(structure.system)
    assert restored.system.instances == structure.system.instances


def test_structure_codec_rejects_unknown_fields_and_missing_arrays() -> None:
    structure = _heterogeneous_structure()
    payload, arrays = encode_structure(structure, array_prefix="structure_0")
    invalid = dict(payload)
    invalid["surprise"] = True
    with pytest.raises(ValueError, match="unknown=.*surprise"):
        decode_structure(invalid, arrays)

    with pytest.raises(ValueError, match="missing array"):
        decode_structure(payload, {})
