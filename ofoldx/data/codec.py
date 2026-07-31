r"""
JSON-safe metadata and tensor storage for molecular graphs and evidence.
"""

from __future__ import annotations

from typing import Mapping, cast

import numpy as np
from torch import Tensor

from .evidence import MSA, MSACollection, Template, TemplateAlignment
from .molecular import MolecularSystem
from .structure import Structure

_SYSTEM_TENSOR_FIELDS = (
    "system_entity_ptr",
    "system_instance_ptr",
    "entity_symbol",
    "entity_type",
    "entity_molecule_type",
    "instance_symbol",
    "instance_entity",
    "instance_component_ptr",
    "component_symbol",
    "component_topology_complete",
    "component_atom_ptr",
    "atom_symbol",
    "atomic_number",
    "isotope",
    "isotope_known",
    "formal_charge",
    "formal_charge_known",
    "radical_electrons",
    "radical_electrons_known",
    "explicit_hydrogens",
    "explicit_hydrogens_known",
    "implicit_hydrogens",
    "implicit_hydrogens_known",
    "implicit_hydrogens_allowed",
    "implicit_hydrogens_allowed_known",
    "atom_aromatic",
    "atom_aromatic_known",
    "atom_stereo",
    "reference_coordinates",
    "reference_coordinate_mask",
    "edge_index",
    "bond_order",
    "bond_order_known",
    "bond_aromatic",
    "bond_aromatic_known",
    "bond_kind",
    "bond_direction",
    "bond_role",
    "bond_stereo",
    "bond_stereo_index",
)

_STRUCTURE_TENSOR_FIELDS = (
    "coordinates",
    "coordinate_mask",
    "occupancy",
    "occupancy_mask",
    "b_factor",
    "b_factor_mask",
    "residue_id",
    "insertion_code",
)


def _encode_tensor_fields(
    value: object,
    fields: tuple[str, ...],
    array_prefix: str,
) -> tuple[dict[str, str], dict[str, Tensor]]:
    references = {field: f"{array_prefix}.{field}" for field in fields}
    arrays = {references[field]: cast(Tensor, getattr(value, field)).detach().cpu() for field in fields}
    return references, arrays


def _decode_tensor_fields(
    references: object,
    fields: tuple[str, ...],
    arrays: Mapping[str, Tensor],
) -> dict[str, Tensor]:
    keys = cast(dict[str, str], references)
    return {field: arrays[keys[field]] for field in fields}


def encode_system(
    system: MolecularSystem,
    *,
    array_prefix: str = "system",
) -> tuple[dict[str, object], dict[str, Tensor]]:
    r"""
    Encode a tensor-native molecular system.

    Args:
        system: Molecular graph to encode.
        array_prefix: Prefix for the named tensor storage.

    Returns:
        JSON-safe metadata and CPU tensors containing the complete realized graph.
    """

    references, arrays = _encode_tensor_fields(system, _SYSTEM_TENSOR_FIELDS, array_prefix)
    return (
        {
            "kind": "molecular_system",
            "symbols": list(system.symbols),
            "tensors": references,
        },
        arrays,
    )


def decode_system(
    value: object,
    arrays: Mapping[str, Tensor],
) -> MolecularSystem:
    r"""
    Decode a tensor-native molecular system.

    Args:
        value: Metadata produced by [`encode_system`][ofoldx.data.codec.encode_system].
        arrays: Named tensors referenced by the metadata.

    Returns:
        The decoded molecular graph on the tensors' device.
    """

    data = cast(dict[str, object], value)
    if data["kind"] != "molecular_system":
        raise ValueError("MolecularSystem kind must be 'molecular_system'")
    fields = _decode_tensor_fields(data["tensors"], _SYSTEM_TENSOR_FIELDS, arrays)
    return MolecularSystem(  # type: ignore[call-arg]
        symbols=tuple(cast(list[str], data["symbols"])),
        **fields,
        batch_size=[],
    )


def encode_structure(
    structure: Structure,
    *,
    array_prefix: str,
) -> tuple[dict[str, object], dict[str, Tensor]]:
    r"""
    Encode one coordinate realization over its tensor-native molecular graph.

    Args:
        structure: Structure to encode.
        array_prefix: Prefix for the named tensor storage.

    Returns:
        JSON-safe metadata and CPU tensors containing the graph and aligned evidence.
    """

    system, arrays = encode_system(
        structure.system,
        array_prefix=f"{array_prefix}.system",
    )
    references, structure_arrays = _encode_tensor_fields(
        structure,
        _STRUCTURE_TENSOR_FIELDS,
        array_prefix,
    )
    arrays.update(structure_arrays)
    return (
        {
            "kind": "structure",
            "system": system,
            "source_symbols": list(structure.source_symbols),
            "tensors": references,
        },
        arrays,
    )


def decode_structure(
    value: object,
    arrays: Mapping[str, Tensor],
) -> Structure:
    r"""
    Decode one coordinate realization over a tensor-native molecular graph.

    Args:
        value: Metadata produced by [`encode_structure`][ofoldx.data.codec.encode_structure].
        arrays: Named tensors referenced by the metadata.

    Returns:
        The decoded structure on the tensors' device.
    """

    data = cast(dict[str, object], value)
    if data["kind"] != "structure":
        raise ValueError("Structure kind must be 'structure'")
    fields = _decode_tensor_fields(data["tensors"], _STRUCTURE_TENSOR_FIELDS, arrays)
    return Structure(  # type: ignore[call-arg]
        system=decode_system(data["system"], arrays),
        source_symbols=tuple(cast(list[str], data["source_symbols"])),
        **fields,
        batch_size=[],
    )


def encode_msa(msa: MSA) -> dict[str, object]:
    r"""
    Encode an MSA as a JSON-safe object.

    Args:
        msa: MSA to encode.

    Returns:
        Sequence rows and deletion counts represented with JSON-safe values.
    """

    return {
        "entity_id": msa.entity_id,
        "unpaired_sequences": list(msa.unpaired_sequences),
        "unpaired_deletion_counts": msa.unpaired_deletion_counts.tolist(),
        "paired_sequences": list(msa.paired_sequences),
        "paired_deletion_counts": msa.paired_deletion_counts.tolist(),
    }


def decode_msa(value: object) -> MSA:
    r"""
    Decode an MSA from a JSON-safe object.

    Args:
        value: Object produced by [`encode_msa`][ofoldx.data.codec.encode_msa].

    Returns:
        The decoded MSA with unsigned deletion-count arrays.
    """

    data = cast(dict[str, object], value)
    return MSA(
        entity_id=cast(str, data["entity_id"]),
        unpaired_sequences=tuple(cast(list[str], data["unpaired_sequences"])),
        unpaired_deletion_counts=np.asarray(
            data["unpaired_deletion_counts"],
            dtype=np.uint32,
        ),
        paired_sequences=tuple(cast(list[str], data["paired_sequences"])),
        paired_deletion_counts=np.asarray(
            data["paired_deletion_counts"],
            dtype=np.uint32,
        ),
    )


def encode_msa_collection(collection: MSACollection) -> dict[str, object]:
    r"""
    Encode an MSA collection as a JSON-safe object.

    Args:
        collection: MSA collection to encode.

    Returns:
        Encoded MSA columns, row-selection arrays, and optional instance targets.
    """

    return {
        "msas": [encode_msa(msa) for msa in collection.msas],
        "row_pairing": collection.row_pairing.tolist(),
        "row_is_paired": collection.row_is_paired.tolist(),
        "target_instance_ids": (
            None if collection.target_instance_ids is None else list(collection.target_instance_ids)
        ),
    }


def decode_msa_collection(value: object) -> MSACollection:
    r"""
    Decode an MSA collection from a JSON-safe object.

    Args:
        value: Object produced by
            [`encode_msa_collection`][ofoldx.data.codec.encode_msa_collection].

    Returns:
        The decoded MSA collection with signed row indices and Boolean pool selectors.
    """

    data = cast(dict[str, object], value)
    target_value = data["target_instance_ids"]
    target_instance_ids = None if target_value is None else tuple(cast(list[str], target_value))
    return MSACollection(
        msas=tuple(decode_msa(item) for item in cast(list[object], data["msas"])),
        row_pairing=np.asarray(data["row_pairing"], dtype=np.int64),
        row_is_paired=np.asarray(data["row_is_paired"], dtype=np.bool_),
        target_instance_ids=target_instance_ids,
    )


def encode_template(
    template: Template,
    *,
    array_prefix: str,
) -> tuple[dict[str, object], dict[str, Tensor]]:
    r"""
    Encode a structural template as metadata and tensors.

    Args:
        template: Template to encode.
        array_prefix: Prefix for the embedded structure tensors.

    Returns:
        A ``(metadata, tensors)`` pair containing the structure and alignments.
    """

    structure, arrays = encode_structure(template.structure, array_prefix=array_prefix)
    return (
        {
            "structure": structure,
            "alignments": [_encode_template_alignment(alignment) for alignment in template.alignments],
        },
        arrays,
    )


def decode_template(value: object, arrays: Mapping[str, Tensor]) -> Template:
    r"""
    Decode a structural template from metadata and tensors.

    Args:
        value: Metadata produced by [`encode_template`][ofoldx.data.codec.encode_template].
        arrays: Tensors keyed by the embedded structure metadata.

    Returns:
        The decoded template.
    """

    data = cast(dict[str, object], value)
    return Template(
        structure=decode_structure(data["structure"], arrays),
        alignments=tuple(_decode_template_alignment(item) for item in cast(list[object], data["alignments"])),
    )


def _encode_template_alignment(alignment: TemplateAlignment) -> dict[str, object]:
    return {
        "target_entity_id": alignment.target_entity_id,
        "target_instance_id": alignment.target_instance_id,
        "template_instance_id": alignment.template_instance_id,
        "residue_pairs": [[target_index, template_index] for target_index, template_index in alignment.residue_pairs],
    }


def _decode_template_alignment(value: object) -> TemplateAlignment:
    data = cast(dict[str, object], value)
    target_instance_id = cast(str | None, data["target_instance_id"])
    return TemplateAlignment(
        target_entity_id=cast(str, data["target_entity_id"]),
        target_instance_id=target_instance_id,
        template_instance_id=cast(str, data["template_instance_id"]),
        residue_pairs=tuple((pair[0], pair[1]) for pair in cast(list[list[int]], data["residue_pairs"])),
    )
