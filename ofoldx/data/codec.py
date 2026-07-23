r"""Explicit codec for canonical molecular, structure, and evidence values.

This module deliberately contains ordinary known-type encoders and decoders.  It
does not register classes, import Python paths, or accept arbitrary objects.
"""

from __future__ import annotations

from typing import Mapping, cast

import numpy as np
from numpy.typing import NDArray

from .evidence import MSA, MSACollection, Template, TemplateAlignment
from .molecular import (
    AtomSite,
    BondOrigin,
    ChemicalAtom,
    ChemicalBond,
    ChemicalComponent,
    MolecularSystem,
    Molecule,
    MoleculeBond,
    PolymerAttachment,
    PolymerSequence,
    PolymerType,
    SystemBond,
)
from .structure import ResidueNumber, Structure


def encode_system(system: MolecularSystem) -> dict[str, object]:
    r"""Serialize a :class:`MolecularSystem` to a JSON-safe object (inverse of :func:`decode_system`).

    Entities are emitted in sorted key order and instance declaration order is preserved so the
    round trip ``decode_system(encode_system(system)) == system`` holds.
    """
    if not isinstance(system, MolecularSystem):
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
    r"""Rebuild a :class:`MolecularSystem` from the object produced by :func:`encode_system` (its inverse)."""
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
        if not isinstance(instance_id, str) or not isinstance(entity_id, str):
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
    r"""Serialize a :class:`Structure` (inverse of :func:`decode_structure`).

    Returns a ``(metadata, arrays)`` pair: bulk numeric fields (coordinates, mask, optional occupancy)
    are moved into ``arrays`` under ``array_prefix``-keyed names and referenced by key from the metadata,
    so ``decode_structure(*encode_structure(structure, array_prefix=p))`` reconstructs the original.
    """
    if not isinstance(structure, Structure):
        raise TypeError("structure must be a Structure")
    if not isinstance(array_prefix, str):
        raise TypeError("array_prefix must be a string")
    if not array_prefix:
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
            "coordinates_dtype": structure.coordinates.dtype.str,
            "coordinate_mask": coordinate_mask_key,
            "occupancy": occupancy_key,
            "occupancy_dtype": None if structure.occupancy is None else structure.occupancy.dtype.str,
            "residue_numbers": _encode_residue_numbers(structure.residue_numbers),
        },
        arrays,
    )


def decode_structure(value: object, arrays: Mapping[str, NDArray]) -> Structure:
    r"""Rebuild a :class:`Structure` from :func:`encode_structure`'s metadata and array map (its inverse).

    Array references in ``value`` are resolved against ``arrays`` to restore coordinates, mask, and
    optional occupancy.
    """
    data = _expect_object(value, "Structure")
    _expect_keys(
        data,
        required={
            "kind",
            "system",
            "sites",
            "coordinates",
            "coordinates_dtype",
            "coordinate_mask",
            "occupancy",
            "occupancy_dtype",
            "residue_numbers",
        },
        owner="Structure",
    )
    if data["kind"] != "structure":
        raise ValueError("Structure kind must be 'structure'")
    sites = _expect_list(data["sites"], "Structure.sites")
    coordinates = _read_array(arrays, data["coordinates"], "Structure.coordinates")
    occupancy = _read_optional_array(arrays, data["occupancy"], "Structure.occupancy")
    if coordinates.dtype != np.dtype(_expect_str(data["coordinates_dtype"], "Structure.coordinates_dtype")):
        raise ValueError("Structure.coordinates dtype disagrees with its metadata")
    occupancy_dtype = data["occupancy_dtype"]
    if occupancy is None:
        if occupancy_dtype is not None:
            raise ValueError("Structure.occupancy_dtype must be null when occupancy is null")
    elif occupancy.dtype != np.dtype(_expect_str(occupancy_dtype, "Structure.occupancy_dtype")):
        raise ValueError("Structure.occupancy dtype disagrees with its metadata")
    return Structure(
        system=decode_system(data["system"]),
        sites=tuple(_decode_site(item) for item in sites),
        coordinates=coordinates,
        coordinate_mask=_read_array(arrays, data["coordinate_mask"], "Structure.coordinate_mask"),
        occupancy=occupancy,
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
        reference = (
            _expect_str(data["instance_id"], "ResidueNumber.instance_id"),
            _expect_int(data["component_index"], "ResidueNumber.component_index"),
        )
        if reference in residue_numbers:
            raise ValueError(f"Structure.residue_numbers has a duplicate reference {reference!r}")
        residue_numbers[reference] = ResidueNumber(
            number=_expect_int(data["number"], "ResidueNumber.number"),
            insertion_code=_expect_str(data["insertion_code"], "ResidueNumber.insertion_code"),
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
    return ChemicalAtom(
        atom_id=_expect_str(data["atom_id"], "ChemicalAtom.atom_id"),
        element=_expect_str(data["element"], "ChemicalAtom.element"),
        isotope=_expect_optional_int(data["isotope"], "ChemicalAtom.isotope"),
        formal_charge=_expect_optional_int(data["formal_charge"], "ChemicalAtom.formal_charge"),
        radical_electrons=_expect_optional_int(data["radical_electrons"], "ChemicalAtom.radical_electrons"),
        explicit_hydrogens=_expect_optional_int(data["explicit_hydrogens"], "ChemicalAtom.explicit_hydrogens"),
        implicit_hydrogens=_expect_optional_int(data["implicit_hydrogens"], "ChemicalAtom.implicit_hydrogens"),
        implicit_hydrogens_allowed=_expect_optional_bool(
            data["implicit_hydrogens_allowed"], "ChemicalAtom.implicit_hydrogens_allowed"
        ),
        aromatic=_expect_optional_bool(data["aromatic"], "ChemicalAtom.aromatic"),
        stereo=_expect_optional_str(data["stereo"], "ChemicalAtom.stereo"),
    )


def _encode_bond(bond: ChemicalBond) -> dict[str, object]:
    return {
        "atom_a": bond.atom_a,
        "atom_b": bond.atom_b,
        "order": bond.order,
        "aromatic": bond.aromatic,
        "stereo": bond.stereo,
        "stereo_atoms": None if bond.stereo_atoms is None else list(bond.stereo_atoms),
    }


def _decode_bond(value: object) -> ChemicalBond:
    data = _expect_object(value, "ChemicalBond")
    _expect_keys(
        data,
        required={"atom_a", "atom_b", "order", "aromatic", "stereo", "stereo_atoms"},
        owner="ChemicalBond",
    )
    stereo_atoms_value = data["stereo_atoms"]
    stereo_atoms = (
        None if stereo_atoms_value is None else _expect_string_pair(stereo_atoms_value, "ChemicalBond.stereo_atoms")
    )
    return ChemicalBond(
        atom_a=_expect_str(data["atom_a"], "ChemicalBond.atom_a"),
        atom_b=_expect_str(data["atom_b"], "ChemicalBond.atom_b"),
        order=_expect_float(data["order"], "ChemicalBond.order"),
        aromatic=_expect_bool(data["aromatic"], "ChemicalBond.aromatic"),
        stereo=_expect_optional_str(data["stereo"], "ChemicalBond.stereo"),
        stereo_atoms=stereo_atoms,
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
        atom_id=_expect_str(data["atom_id"], f"{name}.atom_id"),
        leaving_atom_ids=frozenset(_expect_strings(data["leaving_atom_ids"], f"{name}.leaving_atom_ids")),
    )


def _encode_component(component: ChemicalComponent) -> dict[str, object]:
    bonds = component.bonds
    return {
        "component_id": component.component_id,
        "atoms": [_encode_atom(atom) for atom in component.atoms],
        # JSON null preserves an unknown bond graph; an empty array means the graph is
        # known to contain no bonds.
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
    # Preserve the distinction between an unknown bond graph and a known empty graph.
    bonds = (
        None
        if bonds_value is None
        else frozenset(_decode_bond(item) for item in _expect_list(bonds_value, "ChemicalComponent.bonds"))
    )
    return ChemicalComponent(
        component_id=_expect_str(data["component_id"], "ChemicalComponent.component_id"),
        atoms=tuple(_decode_atom(item) for item in _expect_list(data["atoms"], "ChemicalComponent.atoms")),
        bonds=bonds,
        previous_attachment=_decode_attachment(data["previous_attachment"], "previous_attachment"),
        next_attachment=_decode_attachment(data["next_attachment"], "next_attachment"),
        reference_coordinates=_decode_reference_coordinates(
            data["reference_coordinates"], "ChemicalComponent.reference_coordinates"
        ),
    )


def _encode_molecule_bond(bond: MoleculeBond) -> dict[str, object]:
    return {
        "component_a": bond.component_a,
        "atom_a": bond.atom_a,
        "component_b": bond.component_b,
        "atom_b": bond.atom_b,
        "order": bond.order,
        "aromatic": bond.aromatic,
        "stereo": bond.stereo,
        "stereo_atoms": (
            None
            if bond.stereo_atoms is None
            else [[component_index, atom_id] for component_index, atom_id in bond.stereo_atoms]
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
    stereo_atoms = None if stereo_value is None else _decode_molecule_stereo_atoms(stereo_value)
    return MoleculeBond(
        component_a=_expect_int(data["component_a"], "MoleculeBond.component_a"),
        atom_a=_expect_str(data["atom_a"], "MoleculeBond.atom_a"),
        component_b=_expect_int(data["component_b"], "MoleculeBond.component_b"),
        atom_b=_expect_str(data["atom_b"], "MoleculeBond.atom_b"),
        order=_expect_float(data["order"], "MoleculeBond.order"),
        aromatic=_expect_bool(data["aromatic"], "MoleculeBond.aromatic"),
        stereo=_expect_optional_str(data["stereo"], "MoleculeBond.stereo"),
        stereo_atoms=stereo_atoms,
    )


def _encode_entity(entity: PolymerSequence | Molecule) -> dict[str, object]:
    if isinstance(entity, PolymerSequence):
        return {
            "kind": "polymer_sequence",
            "polymer_type": entity.polymer_type,
            "components": [_encode_component(component) for component in entity.components],
        }
    if isinstance(entity, Molecule):
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
            polymer_type=_expect_polymer_type(data["polymer_type"]),
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
        None if coordinate is None else _expect_coordinate(coordinate, owner)
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
    return AtomSite(
        _expect_str(data["instance_id"], "AtomSite.instance_id"),
        _expect_int(data["component_index"], "AtomSite.component_index"),
        _expect_str(data["atom_id"], "AtomSite.atom_id"),
    )


def _encode_system_bond(bond: SystemBond) -> dict[str, object]:
    return {
        "site_a": _encode_site(bond.site_a),
        "site_b": _encode_site(bond.site_b),
        "order": bond.order,
        "aromatic": bond.aromatic,
        "stereo": bond.stereo,
        "stereo_sites": None if bond.stereo_sites is None else [_encode_site(site) for site in bond.stereo_sites],
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
        order=_expect_float(data["order"], "SystemBond.order"),
        aromatic=_expect_bool(data["aromatic"], "SystemBond.aromatic"),
        stereo=_expect_optional_str(data["stereo"], "SystemBond.stereo"),
        stereo_sites=_expect_site_pair(stereo_sites, "SystemBond.stereo_sites"),
        origin=_expect_bond_origin(data["origin"]),
    )


def encode_msa(msa: MSA) -> dict[str, object]:
    r"""Serialize an :class:`MSA` to a JSON-safe object (inverse of :func:`decode_msa`).

    Sequence rows are stored verbatim as strings and deletion-count arrays are widened to nested int
    lists, so ``decode_msa(encode_msa(msa)) == msa`` round-trips.
    """
    if not isinstance(msa, MSA):
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
    r"""Rebuild an :class:`MSA` from :func:`encode_msa`'s object (its inverse), restoring uint32 deletion arrays."""
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
        entity_id=_expect_str(data["entity_id"], "MSA.entity_id"),
        unpaired_sequences=tuple(_expect_strings(data["unpaired_sequences"], "MSA.unpaired_sequences")),
        unpaired_deletion_counts=_decode_uint32_matrix(
            data["unpaired_deletion_counts"], "MSA.unpaired_deletion_counts"
        ),
        paired_sequences=tuple(_expect_strings(data["paired_sequences"], "MSA.paired_sequences")),
        paired_deletion_counts=_decode_uint32_matrix(data["paired_deletion_counts"], "MSA.paired_deletion_counts"),
    )


def encode_msa_collection(collection: MSACollection) -> dict[str, object]:
    r"""Serialize an :class:`MSACollection` to a JSON-safe object (inverse of :func:`decode_msa_collection`).

    Each member MSA is encoded via :func:`encode_msa` and the dense pairing plan (``row_pairing``,
    ``row_is_paired``) is rendered as nested lists, so the collection round-trips through
    :func:`decode_msa_collection`.
    """
    if not isinstance(collection, MSACollection):
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
    r"""Rebuild an :class:`MSACollection` from :func:`encode_msa_collection`'s object (its inverse).

    Restores the member MSAs and the int64/bool pairing-plan matrices.
    """
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
        None if target_value is None else tuple(_expect_strings(target_value, "MSACollection.target_instance_ids"))
    )
    return MSACollection(
        msas=tuple(decode_msa(item) for item in _expect_list(data["msas"], "MSACollection.msas")),
        row_pairing=_decode_int64_matrix(data["row_pairing"], "MSACollection.row_pairing"),
        row_is_paired=_decode_bool_matrix(data["row_is_paired"], "MSACollection.row_is_paired"),
        target_instance_ids=target_instance_ids,
    )


def encode_template(
    template: Template,
    *,
    array_prefix: str,
) -> tuple[dict[str, object], dict[str, NDArray]]:
    r"""Serialize a :class:`Template` (inverse of :func:`decode_template`).

    Returns a ``(metadata, arrays)`` pair: the template :class:`Structure` is encoded via
    :func:`encode_structure` (its arrays keyed by ``array_prefix``) alongside the alignments, so
    ``decode_template(*encode_template(template, array_prefix=p))`` reconstructs the original.
    """
    if not isinstance(template, Template):
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
    r"""Rebuild a :class:`Template` from :func:`encode_template`'s metadata and array map (its inverse)."""
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
    if not isinstance(alignment, TemplateAlignment):
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
        target_entity_id=_expect_str(data["target_entity_id"], "TemplateAlignment.target_entity_id"),
        target_instance_id=_expect_optional_str(data["target_instance_id"], "TemplateAlignment.target_instance_id"),
        template_instance_id=_expect_str(data["template_instance_id"], "TemplateAlignment.template_instance_id"),
        residue_pairs=tuple(
            _decode_residue_pair(pair)
            for pair in _expect_list(data["residue_pairs"], "TemplateAlignment.residue_pairs")
        ),
    )


def _decode_residue_pair(value: object) -> tuple[int, int]:
    pair = _expect_list(value, "TemplateAlignment residue pair")
    if len(pair) != 2:
        raise ValueError("TemplateAlignment residue pair must have two indices")
    return (
        _expect_int(pair[0], "TemplateAlignment target residue index"),
        _expect_int(pair[1], "TemplateAlignment template residue index"),
    )


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
    if not isinstance(value, dict):
        raise TypeError(f"{owner} must be a JSON object")
    if not all(isinstance(key, str) for key in value):
        raise TypeError(f"{owner} keys must be strings")
    return cast(dict[str, object], value)


def _expect_list(value: object, owner: str) -> list[object]:
    if not isinstance(value, list):
        raise TypeError(f"{owner} must be a JSON array")
    return value


def _expect_strings(value: object, owner: str) -> list[str]:
    values = _expect_list(value, owner)
    strings: list[str] = []
    for item in values:
        if not isinstance(item, str):
            raise TypeError(f"{owner} must contain only strings")
        strings.append(item)
    return strings


def _expect_str(value: object, owner: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{owner} must be a string")
    return value


def _expect_optional_str(value: object, owner: str) -> str | None:
    return None if value is None else _expect_str(value, owner)


def _expect_int(value: object, owner: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{owner} must be an integer")
    return value


def _expect_optional_int(value: object, owner: str) -> int | None:
    return None if value is None else _expect_int(value, owner)


def _expect_float(value: object, owner: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise TypeError(f"{owner} must be a number")
    return float(value)


def _expect_bool(value: object, owner: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{owner} must be a boolean")
    return value


def _expect_optional_bool(value: object, owner: str) -> bool | None:
    return None if value is None else _expect_bool(value, owner)


def _expect_string_pair(value: object, owner: str) -> tuple[str, str]:
    pair = _expect_strings(value, owner)
    if len(pair) != 2:
        raise ValueError(f"{owner} must contain two strings")
    return pair[0], pair[1]


def _expect_coordinate(value: object, owner: str) -> tuple[float, float, float]:
    coordinate = _expect_list(value, owner)
    if len(coordinate) != 3:
        raise ValueError(f"{owner} coordinate must contain three numbers")
    return (
        _expect_float(coordinate[0], owner),
        _expect_float(coordinate[1], owner),
        _expect_float(coordinate[2], owner),
    )


def _decode_molecule_stereo_atoms(value: object) -> tuple[tuple[int, str], tuple[int, str]]:
    raw_pairs = _expect_list(value, "MoleculeBond.stereo_atoms")
    if len(raw_pairs) != 2:
        raise ValueError("MoleculeBond.stereo_atoms must contain two sites")
    pairs: list[tuple[int, str]] = []
    for raw_pair in raw_pairs:
        pair = _expect_list(raw_pair, "MoleculeBond.stereo_atoms entry")
        if len(pair) != 2:
            raise ValueError("MoleculeBond.stereo_atoms entry must contain a component index and atom ID")
        pairs.append(
            (
                _expect_int(pair[0], "MoleculeBond.stereo_atoms component index"),
                _expect_str(pair[1], "MoleculeBond.stereo_atoms atom ID"),
            )
        )
    return pairs[0], pairs[1]


def _expect_site_pair(
    value: tuple[AtomSite, ...] | None,
    owner: str,
) -> tuple[AtomSite, AtomSite] | None:
    if value is None:
        return None
    if len(value) != 2:
        raise ValueError(f"{owner} must contain two atom sites")
    return value[0], value[1]


def _expect_polymer_type(value: object) -> PolymerType:
    polymer_type = _expect_str(value, "PolymerSequence.polymer_type")
    if polymer_type not in {"protein", "rna", "dna", "dna_rna_hybrid"}:
        raise ValueError(f"unsupported polymer type: {polymer_type!r}")
    return cast(PolymerType, polymer_type)


def _expect_bond_origin(value: object) -> BondOrigin:
    origin = _expect_str(value, "SystemBond.origin")
    if origin not in {"component", "molecule", "polymer_rule", "source", "user"}:
        raise ValueError(f"unsupported SystemBond origin: {origin!r}")
    return cast(BondOrigin, origin)


def _expect_keys(data: dict[str, object], *, required: set[str], owner: str) -> None:
    keys = set(data)
    missing = required - keys
    unknown = keys - required
    if missing or unknown:
        raise ValueError(f"{owner} fields mismatch: missing={sorted(missing)!r}, unknown={sorted(unknown)!r}")


def _read_array(arrays: Mapping[str, NDArray], key_value: object, owner: str) -> NDArray:
    if not isinstance(key_value, str) or not key_value:
        raise TypeError(f"{owner} array reference must be a non-empty string")
    try:
        array = arrays[key_value]
    except KeyError:
        raise ValueError(f"{owner} references missing array {key_value!r}") from None
    if not isinstance(array, np.ndarray):
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
