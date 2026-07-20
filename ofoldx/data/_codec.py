r"""Private explicit codec for canonical MolecularSystem and Structure values.

This module deliberately contains ordinary known-type encoders and decoders.  It
does not register classes, import Python paths, or accept arbitrary objects.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from .evidence import MSA, MSACollection, Template, TemplateAlignment
from .molecular import (
    AtomSite,
    ChemicalAtom,
    ChemicalBond,
    ChemicalComponent,
    MolecularSystem,
    Molecule,
    MoleculeBond,
    PolymerAttachment,
    PolymerSequence,
    SystemBond,
)
from .structure import ResidueNumber, Structure


def encode_system(system: MolecularSystem) -> dict[str, object]:
    if type(system) is not MolecularSystem:
        raise TypeError("system must be a MolecularSystem")
    return {
        "kind": "molecular_system",
        "entities": {
            entity_id: _encode_entity(entity)
            for entity_id, entity in sorted(system.entities.items(), key=lambda item: item[0])
        },
        # Instance DECLARATION ORDER is observable (MolecularSystem.iter_atoms and
        # iter_bonds walk ``instances`` in insertion order).  A JSON object would be
        # reordered by ``json.dumps(sort_keys=True)`` on the mmCIF write path, so encode
        # ``instances`` as an ORDERED array of [instance_id, entity_id] pairs that
        # sort_keys cannot scramble.
        "instances": [[instance_id, entity_id] for instance_id, entity_id in system.instances.items()],
        "bonds": [_encode_system_bond(bond) for bond in sorted(system.bonds, key=_system_bond_key)],
    }


def decode_system(value: object) -> MolecularSystem:
    data = _expect_object(value, "MolecularSystem")
    _expect_keys(
        data,
        required={"kind", "entities", "instances", "bonds"},
        owner="MolecularSystem",
    )
    if data["kind"] != "molecular_system":
        raise ValueError("MolecularSystem kind must be 'molecular_system'")
    entities = _expect_object(data["entities"], "MolecularSystem.entities")
    instances = _decode_instances(data["instances"])
    bonds = _expect_list(data["bonds"], "MolecularSystem.bonds")
    return MolecularSystem(
        entities={entity_id: _decode_entity(body) for entity_id, body in entities.items()},
        instances=instances,
        bonds=frozenset(_decode_system_bond(item) for item in bonds),
    )


def _decode_instances(value: object) -> dict[str, str]:
    # ``instances`` is an ORDERED array of [instance_id, entity_id] pairs; rebuild the
    # mapping preserving declaration order (dict keeps insertion order in py3.7+).
    instances: dict[str, str] = {}
    for item in _expect_list(value, "MolecularSystem.instances"):
        pair = _expect_list(item, "MolecularSystem.instances entry")
        if len(pair) != 2:
            raise ValueError("MolecularSystem.instances entry must have two elements")
        instance_id, entity_id = pair
        if type(instance_id) is not str or type(entity_id) is not str:
            raise TypeError("MolecularSystem.instances entry must be [str, str]")
        if instance_id in instances:
            raise ValueError(f"MolecularSystem.instances has a duplicate instance {instance_id!r}")
        instances[instance_id] = entity_id
    return instances


def encode_structure(
    structure: Structure,
    *,
    array_prefix: str,
) -> tuple[dict[str, object], dict[str, NDArray]]:
    if type(structure) is not Structure:
        raise TypeError("structure must be a Structure")
    if type(array_prefix) is not str or not array_prefix:
        raise ValueError("array_prefix must be a non-empty string")
    coordinates_key = f"{array_prefix}.coordinates"
    coordinate_mask_key = f"{array_prefix}.coordinate_mask"
    arrays: dict[str, NDArray] = {
        coordinates_key: structure.coordinates,
        coordinate_mask_key: structure.coordinate_mask,
    }
    occupancy_key: str | None = None
    if structure.occupancy is not None:
        occupancy_key = f"{array_prefix}.occupancy"
        arrays[occupancy_key] = structure.occupancy
    return (
        {
            "kind": "structure",
            "system": encode_system(structure.system),
            "sites": [_encode_site(site) for site in structure.sites],
            "coordinates": coordinates_key,
            "coordinate_mask": coordinate_mask_key,
            "occupancy": occupancy_key,
            "residue_numbers": _encode_residue_numbers(structure.residue_numbers),
        },
        arrays,
    )


def decode_structure(value: object, arrays: Mapping[str, NDArray]) -> Structure:
    data = _expect_object(value, "Structure")
    _expect_keys(
        data,
        required={
            "kind",
            "system",
            "sites",
            "coordinates",
            "coordinate_mask",
            "occupancy",
            "residue_numbers",
        },
        owner="Structure",
    )
    if data["kind"] != "structure":
        raise ValueError("Structure kind must be 'structure'")
    sites = _expect_list(data["sites"], "Structure.sites")
    return Structure(
        system=decode_system(data["system"]),
        sites=tuple(_decode_site(item) for item in sites),
        coordinates=_read_array(arrays, data["coordinates"], "Structure.coordinates"),
        coordinate_mask=_read_array(arrays, data["coordinate_mask"], "Structure.coordinate_mask"),
        occupancy=_read_optional_array(arrays, data["occupancy"], "Structure.occupancy"),
        residue_numbers=_decode_residue_numbers(data["residue_numbers"]),
    )


def _encode_residue_numbers(
    residue_numbers: dict[tuple[str, int], ResidueNumber] | None,
) -> list[dict[str, object]] | None:
    if residue_numbers is None:
        return None
    entries: list[dict[str, object]] = []
    for (instance_id, component_index), residue_number in sorted(residue_numbers.items(), key=lambda item: item[0]):
        entries.append(
            {
                "instance_id": instance_id,
                "component_index": component_index,
                "number": residue_number.number,
                "insertion_code": residue_number.insertion_code,
            }
        )
    return entries


def _decode_residue_numbers(value: object) -> dict[tuple[str, int], ResidueNumber] | None:
    if value is None:
        return None
    residue_numbers: dict[tuple[str, int], ResidueNumber] = {}
    for item in _expect_list(value, "Structure.residue_numbers"):
        data = _expect_object(item, "ResidueNumber entry")
        _expect_keys(
            data,
            required={"instance_id", "component_index", "number", "insertion_code"},
            owner="ResidueNumber entry",
        )
        reference = (data["instance_id"], data["component_index"])
        if reference in residue_numbers:
            raise ValueError(f"Structure.residue_numbers has a duplicate reference {reference!r}")
        residue_numbers[reference] = ResidueNumber(  # type: ignore[index]
            number=data["number"],  # type: ignore[arg-type]
            insertion_code=data["insertion_code"],  # type: ignore[arg-type]
        )
    return residue_numbers


def _encode_atom(atom: ChemicalAtom) -> dict[str, object]:
    return {
        "atom_id": atom.atom_id,
        "element": atom.element,
        "isotope": atom.isotope,
        "formal_charge": atom.formal_charge,
        "radical_electrons": atom.radical_electrons,
        "explicit_hydrogens": atom.explicit_hydrogens,
        "implicit_hydrogens": atom.implicit_hydrogens,
        "implicit_hydrogens_allowed": atom.implicit_hydrogens_allowed,
        "aromatic": atom.aromatic,
        "stereo": atom.stereo,
    }


def _decode_atom(value: object) -> ChemicalAtom:
    data = _expect_object(value, "ChemicalAtom")
    _expect_keys(
        data,
        required={
            "atom_id",
            "element",
            "isotope",
            "formal_charge",
            "radical_electrons",
            "explicit_hydrogens",
            "implicit_hydrogens",
            "implicit_hydrogens_allowed",
            "aromatic",
            "stereo",
        },
        owner="ChemicalAtom",
    )
    return ChemicalAtom(**data)  # type: ignore[arg-type]


def _encode_bond(bond: ChemicalBond) -> dict[str, object]:
    atom_a, atom_b = bond.atom_a, bond.atom_b
    stereo_atoms = bond.stereo_atoms
    if atom_b < atom_a:
        atom_a, atom_b = atom_b, atom_a
        if stereo_atoms is not None:
            stereo_atoms = stereo_atoms[1], stereo_atoms[0]
    return {
        "atom_a": atom_a,
        "atom_b": atom_b,
        "order": bond.order,
        "aromatic": bond.aromatic,
        "stereo": bond.stereo,
        "stereo_atoms": None if stereo_atoms is None else list(stereo_atoms),
    }


def _decode_bond(value: object) -> ChemicalBond:
    data = _expect_object(value, "ChemicalBond")
    _expect_keys(
        data,
        required={"atom_a", "atom_b", "order", "aromatic", "stereo", "stereo_atoms"},
        owner="ChemicalBond",
    )
    stereo_atoms_value = data["stereo_atoms"]
    stereo_atoms = None if stereo_atoms_value is None else tuple(_expect_list(stereo_atoms_value, "stereo_atoms"))
    return ChemicalBond(
        atom_a=data["atom_a"],  # type: ignore[arg-type]
        atom_b=data["atom_b"],  # type: ignore[arg-type]
        order=data["order"],  # type: ignore[arg-type]
        aromatic=data["aromatic"],  # type: ignore[arg-type]
        stereo=data["stereo"],  # type: ignore[arg-type]
        stereo_atoms=stereo_atoms,  # type: ignore[arg-type]
    )


def _encode_attachment(attachment: PolymerAttachment | None) -> dict[str, object] | None:
    if attachment is None:
        return None
    return {
        "atom_id": attachment.atom_id,
        "leaving_atom_ids": sorted(attachment.leaving_atom_ids),
    }


def _decode_attachment(value: object, name: str) -> PolymerAttachment | None:
    if value is None:
        return None
    data = _expect_object(value, name)
    _expect_keys(data, required={"atom_id", "leaving_atom_ids"}, owner=name)
    return PolymerAttachment(
        atom_id=data["atom_id"],  # type: ignore[arg-type]
        leaving_atom_ids=frozenset(_expect_list(data["leaving_atom_ids"], f"{name}.leaving_atom_ids")),
    )


def _encode_component(component: ChemicalComponent) -> dict[str, object]:
    bonds = component.bonds
    return {
        "component_id": component.component_id,
        "atoms": [_encode_atom(atom) for atom in component.atoms],
        # TODO(§3.2.1 reopened): partial components carry bonds=None; the complete
        # None-vs-frozenset() codec facet is reopened, so round-trip None as JSON null.
        "bonds": None if bonds is None else [_encode_bond(bond) for bond in sorted(bonds, key=_chemical_bond_key)],
        "previous_attachment": _encode_attachment(component.previous_attachment),
        "next_attachment": _encode_attachment(component.next_attachment),
        "reference_coordinates": [
            None if coordinate is None else list(coordinate) for coordinate in component.reference_coordinates
        ],
    }


def _decode_component(value: object) -> ChemicalComponent:
    data = _expect_object(value, "ChemicalComponent")
    _expect_keys(
        data,
        required={
            "component_id",
            "atoms",
            "bonds",
            "previous_attachment",
            "next_attachment",
            "reference_coordinates",
        },
        owner="ChemicalComponent",
    )
    bonds_value = data["bonds"]
    # TODO(§3.2.1 reopened): partial components decode bonds=None; complete facet reopened.
    bonds = (
        None
        if bonds_value is None
        else frozenset(_decode_bond(item) for item in _expect_list(bonds_value, "ChemicalComponent.bonds"))
    )
    return ChemicalComponent(
        component_id=data["component_id"],  # type: ignore[arg-type]
        atoms=tuple(_decode_atom(item) for item in _expect_list(data["atoms"], "ChemicalComponent.atoms")),
        bonds=bonds,
        previous_attachment=_decode_attachment(data["previous_attachment"], "previous_attachment"),
        next_attachment=_decode_attachment(data["next_attachment"], "next_attachment"),
        reference_coordinates=_decode_reference_coordinates(
            data["reference_coordinates"], "ChemicalComponent.reference_coordinates"
        ),
    )


def _encode_molecule_bond(bond: MoleculeBond) -> dict[str, object]:
    component_a, atom_a = bond.component_a, bond.atom_a
    component_b, atom_b = bond.component_b, bond.atom_b
    stereo_atoms = bond.stereo_atoms
    if (component_b, atom_b) < (component_a, atom_a):
        component_a, atom_a, component_b, atom_b = component_b, atom_b, component_a, atom_a
        if stereo_atoms is not None:
            stereo_atoms = stereo_atoms[1], stereo_atoms[0]
    return {
        "component_a": component_a,
        "atom_a": atom_a,
        "component_b": component_b,
        "atom_b": atom_b,
        "order": bond.order,
        "aromatic": bond.aromatic,
        "stereo": bond.stereo,
        "stereo_atoms": (
            None if stereo_atoms is None else [[component_index, atom_id] for component_index, atom_id in stereo_atoms]
        ),
    }


def _decode_molecule_bond(value: object) -> MoleculeBond:
    data = _expect_object(value, "MoleculeBond")
    _expect_keys(
        data,
        required={
            "component_a",
            "atom_a",
            "component_b",
            "atom_b",
            "order",
            "aromatic",
            "stereo",
            "stereo_atoms",
        },
        owner="MoleculeBond",
    )
    stereo_value = data["stereo_atoms"]
    stereo_atoms = (
        None
        if stereo_value is None
        else tuple(
            tuple(_expect_list(item, "MoleculeBond.stereo_atoms entry"))
            for item in _expect_list(stereo_value, "MoleculeBond.stereo_atoms")
        )
    )
    return MoleculeBond(
        component_a=data["component_a"],  # type: ignore[arg-type]
        atom_a=data["atom_a"],  # type: ignore[arg-type]
        component_b=data["component_b"],  # type: ignore[arg-type]
        atom_b=data["atom_b"],  # type: ignore[arg-type]
        order=data["order"],  # type: ignore[arg-type]
        aromatic=data["aromatic"],  # type: ignore[arg-type]
        stereo=data["stereo"],  # type: ignore[arg-type]
        stereo_atoms=stereo_atoms,  # type: ignore[arg-type]
    )


def _encode_entity(entity: PolymerSequence | Molecule) -> dict[str, object]:
    if type(entity) is PolymerSequence:
        return {
            "kind": "polymer_sequence",
            "polymer_type": entity.polymer_type,
            "components": [_encode_component(component) for component in entity.components],
        }
    if type(entity) is Molecule:
        return {
            "kind": "molecule",
            "components": [_encode_component(component) for component in entity.components],
            "bonds": [_encode_molecule_bond(bond) for bond in sorted(entity.bonds, key=_molecule_bond_key)],
        }
    raise TypeError("entity must be a PolymerSequence or Molecule")


def _decode_entity(value: object) -> PolymerSequence | Molecule:
    data = _expect_object(value, "entity")
    kind = data.get("kind")
    if kind == "polymer_sequence":
        _expect_keys(data, required={"kind", "polymer_type", "components"}, owner="PolymerSequence")
        return PolymerSequence(
            polymer_type=data["polymer_type"],  # type: ignore[arg-type]
            components=tuple(
                _decode_component(item) for item in _expect_list(data["components"], "PolymerSequence.components")
            ),
        )
    if kind == "molecule":
        _expect_keys(data, required={"kind", "components", "bonds"}, owner="Molecule")
        return Molecule(
            components=tuple(
                _decode_component(item) for item in _expect_list(data["components"], "Molecule.components")
            ),
            bonds=frozenset(_decode_molecule_bond(item) for item in _expect_list(data["bonds"], "Molecule.bonds")),
        )
    raise ValueError(f"unsupported entity kind: {kind!r}")


def _decode_reference_coordinates(
    value: object,
    owner: str,
) -> tuple[tuple[float, float, float] | None, ...]:
    return tuple(
        None if coordinate is None else tuple(_expect_list(coordinate, owner))  # type: ignore[misc]
        for coordinate in _expect_list(value, owner)
    )


def _encode_site(site: AtomSite) -> dict[str, object]:
    return {
        "instance_id": site.instance_id,
        "component_index": site.component_index,
        "atom_id": site.atom_id,
    }


def _decode_site(value: object) -> AtomSite:
    data = _expect_object(value, "AtomSite")
    _expect_keys(data, required={"instance_id", "component_index", "atom_id"}, owner="AtomSite")
    return AtomSite(data["instance_id"], data["component_index"], data["atom_id"])  # type: ignore[arg-type]


def _encode_system_bond(bond: SystemBond) -> dict[str, object]:
    site_a, site_b = bond.site_a, bond.site_b
    stereo_sites = bond.stereo_sites
    if _site_key(site_b) < _site_key(site_a):
        site_a, site_b = site_b, site_a
        if stereo_sites is not None:
            stereo_sites = stereo_sites[1], stereo_sites[0]
    return {
        "site_a": _encode_site(site_a),
        "site_b": _encode_site(site_b),
        "order": bond.order,
        "aromatic": bond.aromatic,
        "stereo": bond.stereo,
        "stereo_sites": None if stereo_sites is None else [_encode_site(site) for site in stereo_sites],
        "origin": bond.origin,
    }


def _decode_system_bond(value: object) -> SystemBond:
    data = _expect_object(value, "SystemBond")
    _expect_keys(
        data,
        required={"site_a", "site_b", "order", "aromatic", "stereo", "stereo_sites", "origin"},
        owner="SystemBond",
    )
    stereo_value = data["stereo_sites"]
    stereo_sites = (
        None
        if stereo_value is None
        else tuple(_decode_site(item) for item in _expect_list(stereo_value, "SystemBond.stereo_sites"))
    )
    return SystemBond(
        site_a=_decode_site(data["site_a"]),
        site_b=_decode_site(data["site_b"]),
        order=data["order"],  # type: ignore[arg-type]
        aromatic=data["aromatic"],  # type: ignore[arg-type]
        stereo=data["stereo"],  # type: ignore[arg-type]
        stereo_sites=stereo_sites,  # type: ignore[arg-type]
        origin=data["origin"],  # type: ignore[arg-type]
    )


def encode_msa(msa: MSA) -> dict[str, object]:
    if type(msa) is not MSA:
        raise TypeError("msa must be an MSA")
    # Cells are UPPERCASE one-letter MSA symbols (or "-"), stored verbatim as strings; the codec does
    # not resolve symbols to tokens or ChemicalComponent IDs (that is the Processor's job). Deletion
    # arrays are CPU uint32; ``tolist`` widens each row to plain ints for the known-type JSON form.
    return {
        "entity_id": msa.entity_id,
        "unpaired_sequences": list(msa.unpaired_sequences),
        "unpaired_deletion_counts": msa.unpaired_deletion_counts.tolist(),
        "paired_sequences": list(msa.paired_sequences),
        "paired_deletion_counts": msa.paired_deletion_counts.tolist(),
    }


def decode_msa(value: object) -> MSA:
    data = _expect_object(value, "MSA")
    _expect_keys(
        data,
        required={
            "entity_id",
            "unpaired_sequences",
            "unpaired_deletion_counts",
            "paired_sequences",
            "paired_deletion_counts",
        },
        owner="MSA",
    )
    return MSA(
        entity_id=data["entity_id"],  # type: ignore[arg-type]
        unpaired_sequences=tuple(  # type: ignore[arg-type]
            _expect_list(data["unpaired_sequences"], "MSA.unpaired_sequences")
        ),
        unpaired_deletion_counts=_decode_uint32_matrix(
            data["unpaired_deletion_counts"], "MSA.unpaired_deletion_counts"
        ),
        paired_sequences=tuple(  # type: ignore[arg-type]
            _expect_list(data["paired_sequences"], "MSA.paired_sequences")
        ),
        paired_deletion_counts=_decode_uint32_matrix(data["paired_deletion_counts"], "MSA.paired_deletion_counts"),
    )


def encode_msa_collection(collection: MSACollection) -> dict[str, object]:
    if type(collection) is not MSACollection:
        raise TypeError("collection must be an MSACollection")
    # ``row_pairing`` (int64, >=0 source row / -1 gap) and ``row_is_paired`` (bool, paired vs unpaired
    # pool) are the dense (plan_rows, len(msas)) plan; ``tolist`` renders each as nested plain lists.
    return {
        "msas": [encode_msa(msa) for msa in collection.msas],
        "row_pairing": collection.row_pairing.tolist(),
        "row_is_paired": collection.row_is_paired.tolist(),
        "target_instance_ids": (
            None if collection.target_instance_ids is None else list(collection.target_instance_ids)
        ),
    }


def decode_msa_collection(value: object) -> MSACollection:
    data = _expect_object(value, "MSACollection")
    _expect_keys(
        data,
        required={
            "msas",
            "row_pairing",
            "row_is_paired",
            "target_instance_ids",
        },
        owner="MSACollection",
    )
    target_value = data["target_instance_ids"]
    target_instance_ids = (
        None if target_value is None else tuple(_expect_list(target_value, "MSACollection.target_instance_ids"))
    )
    return MSACollection(
        msas=tuple(decode_msa(item) for item in _expect_list(data["msas"], "MSACollection.msas")),
        row_pairing=_decode_int64_matrix(data["row_pairing"], "MSACollection.row_pairing"),
        row_is_paired=_decode_bool_matrix(data["row_is_paired"], "MSACollection.row_is_paired"),
        target_instance_ids=target_instance_ids,  # type: ignore[arg-type]
    )


def encode_template(
    template: Template,
    *,
    array_prefix: str,
) -> tuple[dict[str, object], dict[str, NDArray]]:
    if type(template) is not Template:
        raise TypeError("template must be a Template")
    structure, arrays = encode_structure(template.structure, array_prefix=array_prefix)
    return (
        {
            "structure": structure,
            "alignments": [_encode_template_alignment(alignment) for alignment in template.alignments],
        },
        arrays,
    )


def decode_template(value: object, arrays: Mapping[str, NDArray]) -> Template:
    data = _expect_object(value, "Template")
    _expect_keys(
        data,
        required={"structure", "alignments"},
        owner="Template",
    )
    return Template(
        structure=decode_structure(data["structure"], arrays),
        alignments=tuple(
            _decode_template_alignment(item) for item in _expect_list(data["alignments"], "Template.alignments")
        ),
    )


def _encode_template_alignment(alignment: TemplateAlignment) -> dict[str, object]:
    if type(alignment) is not TemplateAlignment:
        raise TypeError("alignment must be a TemplateAlignment")
    return {
        "target_entity_id": alignment.target_entity_id,
        "target_instance_id": alignment.target_instance_id,
        "template_instance_id": alignment.template_instance_id,
        "residue_pairs": [[target_index, template_index] for target_index, template_index in alignment.residue_pairs],
    }


def _decode_template_alignment(value: object) -> TemplateAlignment:
    data = _expect_object(value, "TemplateAlignment")
    _expect_keys(
        data,
        required={"target_entity_id", "target_instance_id", "template_instance_id", "residue_pairs"},
        owner="TemplateAlignment",
    )
    return TemplateAlignment(
        target_entity_id=data["target_entity_id"],  # type: ignore[arg-type]
        target_instance_id=data["target_instance_id"],  # type: ignore[arg-type]
        template_instance_id=data["template_instance_id"],  # type: ignore[arg-type]
        residue_pairs=tuple(
            _decode_residue_pair(pair)
            for pair in _expect_list(data["residue_pairs"], "TemplateAlignment.residue_pairs")
        ),
    )


def _decode_residue_pair(value: object) -> tuple[int, int]:
    pair = _expect_list(value, "TemplateAlignment residue pair")
    if len(pair) != 2:
        raise ValueError("TemplateAlignment residue pair must have two indices")
    return pair[0], pair[1]  # type: ignore[return-value]


def _chemical_bond_key(bond: ChemicalBond) -> tuple[object, ...]:
    endpoint_a, endpoint_b = sorted((bond.atom_a, bond.atom_b))
    return endpoint_a, endpoint_b, bond.order, bond.aromatic, bond.stereo or ""


def _molecule_bond_key(bond: MoleculeBond) -> tuple[object, ...]:
    endpoint_a, endpoint_b = sorted(((bond.component_a, bond.atom_a), (bond.component_b, bond.atom_b)))
    return endpoint_a, endpoint_b, bond.order, bond.aromatic, bond.stereo or ""


def _site_key(site: AtomSite) -> tuple[object, ...]:
    return (site.instance_id, site.component_index, site.atom_id)


def _system_bond_key(bond: SystemBond) -> tuple[object, ...]:
    endpoint_a, endpoint_b = sorted((_site_key(bond.site_a), _site_key(bond.site_b)))
    return (
        endpoint_a,
        endpoint_b,
        bond.order,
        bond.aromatic,
        bond.stereo or "",
        bond.origin,
    )


def _expect_object(value: object, owner: str) -> dict[str, object]:
    if type(value) is not dict:
        raise TypeError(f"{owner} must be a JSON object")
    return value  # type: ignore[return-value]


def _expect_list(value: object, owner: str) -> list[object]:
    if type(value) is not list:
        raise TypeError(f"{owner} must be a JSON array")
    return value


def _expect_keys(data: dict[str, object], *, required: set[str], owner: str) -> None:
    keys = set(data)
    missing = required - keys
    unknown = keys - required
    if missing or unknown:
        raise ValueError(f"{owner} fields mismatch: missing={sorted(missing)!r}, unknown={sorted(unknown)!r}")


def _read_array(arrays: Mapping[str, NDArray], key_value: object, owner: str) -> NDArray:
    if type(key_value) is not str or not key_value:
        raise TypeError(f"{owner} array reference must be a non-empty string")
    try:
        array = arrays[key_value]
    except KeyError:
        raise ValueError(f"{owner} references missing array {key_value!r}") from None
    if type(array) is not np.ndarray:
        raise TypeError(f"{owner} storage must be a numpy.ndarray")
    return array


def _read_optional_array(arrays: Mapping[str, NDArray], key_value: object, owner: str) -> NDArray | None:
    if key_value is None:
        return None
    return _read_array(arrays, key_value, owner)


def _decode_uint32_matrix(value: object, owner: str) -> NDArray[np.uint32]:
    rows = [list(_expect_list(row, f"{owner} row")) for row in _expect_list(value, owner)]
    return np.asarray(rows, dtype=np.uint32)


def _decode_int64_matrix(value: object, owner: str) -> NDArray[np.int64]:
    rows = [list(_expect_list(row, f"{owner} row")) for row in _expect_list(value, owner)]
    return np.asarray(rows, dtype=np.int64)


def _decode_bool_matrix(value: object, owner: str) -> NDArray[np.bool_]:
    rows = [list(_expect_list(row, f"{owner} row")) for row in _expect_list(value, owner)]
    return np.asarray(rows, dtype=np.bool_)
