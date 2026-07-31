r"""
PDB visualization writer for MpnnPacker predictions.

[`write_pdb`][] serializes one complete [`PackPrediction`][ofoldx.results.PackPrediction]
using the packed-structure conventions produced by MpnnPacker. The output is
intended for visualization rather than general ``Structure`` exchange.
"""

from __future__ import annotations

import math
import string
from os import PathLike
from pathlib import Path

import torch
from biotite.rust.structure.io.pdb import encode_hybrid36

from ...data import ATOMIC_NUMBER_BY_ELEMENT, ENTITY_PROTEIN, water_component_mask
from ...data.molecular import ComponentReference
from ...modules import residue_constants as rc
from ...results import PackPrediction


def write_pdb(
    prediction: PackPrediction,
    path: str | PathLike[str],
) -> None:
    r"""
    Write a packed structure prediction in PDB format.

    Args:
        prediction: Packed structure and optional per-residue chi metrics.
        path: Destination file path.

    Raises:
        ValueError: If the prediction contains identifiers or coordinates that
            cannot be represented in PDB format.
    """

    structure = prediction.structure
    system = structure.system
    instance_component_ptr = system.instance_component_ptr.detach().cpu()
    component_atom_ptr = system.component_atom_ptr.detach().cpu()
    component_symbol = system.component_symbol.detach().cpu()
    atom_symbol = system.atom_symbol.detach().cpu()
    atomic_number = system.atomic_number.detach().cpu()
    instance_symbol = system.instance_symbol.detach().cpu()
    instance_entity = system.instance_entity.detach().cpu()
    entity_type = system.entity_type.detach().cpu()
    coordinates = structure.coordinates.detach().cpu()
    coordinate_mask = structure.coordinate_mask.detach().cpu()
    occupancy = structure.occupancy.detach().cpu()
    occupancy_mask = structure.occupancy_mask.detach().cpu()
    residue_id = structure.residue_id.detach().cpu()
    insertion_code = structure.insertion_code.detach().cpu()

    component_instance = torch.repeat_interleave(
        torch.arange(system.instance_symbol.numel()),
        instance_component_ptr[1:] - instance_component_ptr[:-1],
    )
    atom_component = torch.repeat_interleave(
        torch.arange(system.component_symbol.numel()),
        component_atom_ptr[1:] - component_atom_ptr[:-1],
    )
    retained_atom = coordinate_mask & ~water_component_mask(system).detach().cpu()[atom_component]
    record_indices = (
        torch.nonzero(
            retained_atom,
            as_tuple=False,
        )
        .flatten()
        .tolist()
    )
    retained_instances = {int(component_instance[int(atom_component[index])]) for index in record_indices}
    instance_indices = [index for index in range(system.instance_symbol.numel()) if index in retained_instances]
    instance_ids = {index: system.symbols[int(instance_symbol[index])] for index in instance_indices}
    element_by_number = {number: element for element, number in ATOMIC_NUMBER_BY_ELEMENT.items()}
    chain_alphabet = string.ascii_uppercase + string.ascii_lowercase + string.digits + string.punctuation + " "
    chain_by_instance: dict[int, str] = {}
    retained_chain_ids = {
        instance_id
        for instance_id in instance_ids.values()
        if len(instance_id) == 1 and instance_id.isascii() and instance_id.isprintable() and not instance_id.isspace()
    }
    used_chains = set(retained_chain_ids)
    empty_instance = next(
        (index for index, instance_id in instance_ids.items() if instance_id == ""),
        None,
    )
    if empty_instance is not None:
        chain_by_instance[empty_instance] = " "
        used_chains.add(" ")
    available_chains = (chain for chain in chain_alphabet if chain not in used_chains)
    for instance_index in instance_indices:
        instance_id = instance_ids[instance_index]
        if instance_id == "":
            continue
        if instance_id in retained_chain_ids:
            chain_by_instance[instance_index] = instance_id
            continue
        try:
            chain_by_instance[instance_index] = next(available_chains)
        except StopIteration:
            raise ValueError("PDB cannot encode all MolecularSystem instance IDs as unique chains") from None

    density_by_group = {}
    for chi_index in range(4):
        metric = prediction.metrics.get(f"chi{chi_index + 1}_log_density")
        if metric is not None:
            density_by_group[chi_index + 4] = dict(zip(metric.axis, metric.values, strict=True))
    amino_acid_index_by_component = {
        component_id: index
        for index, component_id in enumerate(
            rc.AMINO_ACID_ONE_TO_THREE[code] for code in rc.AMINO_ACID_ONE_LETTER_CODES
        )
    }

    lines: list[str] = []
    serialized_reference_by_key: dict[tuple[str, int, str], ComponentReference] = {}
    for serial, atom_index in enumerate(record_indices, start=1):
        component_index = int(atom_component[atom_index])
        instance_index = int(component_instance[component_index])
        local_component = component_index - int(instance_component_ptr[instance_index])
        instance_id = instance_ids[instance_index]
        reference = (instance_id, local_component)

        serial_field = f"{encode_hybrid36(serial, 5):>5}"
        entity_index = int(instance_entity[instance_index])
        is_protein = int(entity_type[entity_index]) == ENTITY_PROTEIN
        record = "ATOM" if is_protein else "HETATM"
        component_id = system.symbols[int(component_symbol[component_index])]
        atom_id = system.symbols[int(atom_symbol[atom_index])]
        element = element_by_number[int(atomic_number[atom_index])]
        if not component_id or len(component_id) > 3:
            raise ValueError(f"PDB cannot encode component ID {component_id!r}")
        if not atom_id or len(atom_id) > 4:
            raise ValueError(f"PDB cannot encode atom ID {atom_id!r}")
        if not element or len(element) > 2:
            raise ValueError(f"PDB cannot encode element {element!r}")
        if any(character.isspace() for value in (component_id, atom_id) for character in value):
            raise ValueError("PDB component and atom IDs cannot contain whitespace")
        if any(not value.isascii() or not value.isprintable() for value in (component_id, atom_id, element)):
            raise ValueError("PDB identity fields must contain printable ASCII characters")
        if len(atom_id) < 4 and len(element) == 1 and not atom_id[0].isdigit():
            atom_field = f" {atom_id:<3}"
        else:
            atom_field = f"{atom_id:<4}"

        source_residue = int(residue_id[component_index])
        if source_residue < 0:
            number = local_component + 1
        else:
            value = structure.source_symbols[source_residue]
            try:
                number = int(value)
            except ValueError:
                raise ValueError(f"PDB cannot encode residue number {value!r}") from None
        source_insertion = int(insertion_code[component_index])
        code = "" if source_insertion < 0 else structure.source_symbols[source_insertion]
        chain = chain_by_instance[instance_index]
        if number < 0:
            residue_field = f"{number:>4}"
            if len(residue_field) != 4:
                raise ValueError(f"PDB cannot encode residue number {number}")
        else:
            residue_field = f"{encode_hybrid36(number, 4):>4}"
        if code and (len(code) != 1 or not code.isascii() or not code.isalpha()):
            raise ValueError(f"PDB cannot encode insertion code {code!r}")
        occurrence_key = (chain, number, code)
        serialized_reference = serialized_reference_by_key.setdefault(occurrence_key, reference)
        if serialized_reference != reference:
            raise ValueError(
                f"PDB component occurrences {serialized_reference!r} and {reference!r} "
                f"collide at {occurrence_key!r}"
            )

        coordinate_fields = tuple(f"{float(value):8.3f}" for value in coordinates[atom_index])
        if any(len(value) != 8 for value in coordinate_fields) or not all(
            math.isfinite(float(value)) for value in coordinates[atom_index]
        ):
            raise ValueError(f"PDB cannot encode coordinates for " f"{(instance_id, local_component, atom_id)!r}")
        if not bool(occupancy_mask[atom_index]):
            occupancy_field = " " * 6
        else:
            occupancy_value = float(occupancy[atom_index])
            occupancy_field = f"{occupancy_value:6.2f}"
            if len(occupancy_field) != 6 or not math.isfinite(occupancy_value):
                raise ValueError(f"PDB cannot encode occupancy {occupancy_value!r}")

        b_value = 0.0
        amino_acid_index = amino_acid_index_by_component.get(component_id)
        if is_protein and amino_acid_index is not None:
            atom14_names = rc.AMINO_ACID_ATOM14_NAMES[component_id]
            if atom_id in atom14_names:
                atom14_index = atom14_names.index(atom_id)
                rigid_group = int(rc.AMINO_ACID_ATOM14_TO_RIGID_GROUP[amino_acid_index, atom14_index])
                density = density_by_group.get(rigid_group, {}).get(reference)
                if density is not None:
                    b_value = float(density)
        b_field = f"{b_value:6.2f}"
        if len(b_field) != 6 or not math.isfinite(b_value):
            raise ValueError(f"PDB cannot encode chi log density {b_value!r}")

        lines.append(
            f"{record:<6}{serial_field} {atom_field} {component_id:>3} {chain}{residue_field}"
            f"{code or ' '}   {''.join(coordinate_fields)}{occupancy_field}{b_field}"
            f"          {element:>2}  "
        )

    lines.append("END")
    Path(path).expanduser().write_text("\n".join(lines) + "\n", encoding="ascii")
