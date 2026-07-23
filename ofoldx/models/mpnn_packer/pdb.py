r"""PDB visualization writer for MpnnPacker predictions.

:func:`write_pdb` serializes one complete :class:`~ofoldx.results.PackPrediction`
using the packed-structure conventions produced by MpnnPacker. The output is
intended for visualization rather than general ``Structure`` exchange.
"""

from __future__ import annotations

import string
from os import PathLike
from pathlib import Path

import numpy as np
from biotite.rust.structure.io.pdb import encode_hybrid36

from ...data import is_water_component
from ...data.molecular import ComponentReference, PolymerSequence
from ...modules import residue_constants as rc
from ...results import PackPrediction


def write_pdb(
    prediction: PackPrediction,
    path: str | PathLike[str],
) -> None:
    r"""Write one packed prediction as its canonical PDB visualization."""

    structure = prediction.structure
    system = structure.system
    record_indices = tuple(
        index
        for index, site in enumerate(structure.sites)
        if bool(structure.coordinate_mask[index])
        and not is_water_component(system.component((site.instance_id, site.component_index)))
    )
    retained_instances = {structure.sites[index].instance_id for index in record_indices}
    instance_ids = tuple(instance_id for instance_id in system.instances if instance_id in retained_instances)
    chain_alphabet = string.ascii_uppercase + string.ascii_lowercase + string.digits + string.punctuation + " "
    chain_by_instance: dict[str, str] = {}
    retained_chain_ids = {
        instance_id
        for instance_id in instance_ids
        if len(instance_id) == 1 and instance_id.isascii() and instance_id.isprintable() and not instance_id.isspace()
    }
    used_chains = set(retained_chain_ids)
    if "" in retained_instances:
        chain_by_instance[""] = " "
        used_chains.add(" ")
    available_chains = (chain for chain in chain_alphabet if chain not in used_chains)
    for instance_id in instance_ids:
        if instance_id == "":
            continue
        if instance_id in retained_chain_ids:
            chain_by_instance[instance_id] = instance_id
            continue
        try:
            chain_by_instance[instance_id] = next(available_chains)
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
    for serial, site_index in enumerate(record_indices, start=1):
        site = structure.sites[site_index]
        reference = (site.instance_id, site.component_index)
        component = system.component(reference)

        serial_field = f"{encode_hybrid36(serial, 5):>5}"
        entity = system.entities[system.instances[site.instance_id]]
        record = "ATOM" if isinstance(entity, PolymerSequence) and entity.polymer_type == "protein" else "HETATM"
        component_id = component.component_id
        atom_id = site.atom_id
        atom = system.atom(site)
        element = atom.element
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

        residue_number = None if structure.residue_numbers is None else structure.residue_numbers.get(reference)
        number = site.component_index + 1 if residue_number is None else residue_number.number
        insertion_code = "" if residue_number is None else residue_number.insertion_code
        chain = chain_by_instance[site.instance_id]
        if number < 0:
            residue_field = f"{number:>4}"
            if len(residue_field) != 4:
                raise ValueError(f"PDB cannot encode residue number {number}")
        else:
            residue_field = f"{encode_hybrid36(number, 4):>4}"
        if insertion_code and (
            len(insertion_code) != 1 or not insertion_code.isascii() or not insertion_code.isalpha()
        ):
            raise ValueError(f"PDB cannot encode insertion code {insertion_code!r}")
        occurrence_key = (chain, number, insertion_code)
        serialized_reference = serialized_reference_by_key.setdefault(occurrence_key, reference)
        if serialized_reference != reference:
            raise ValueError(
                f"PDB component occurrences {serialized_reference!r} and {reference!r} "
                f"collide at {occurrence_key!r}"
            )

        coordinate_fields = tuple(f"{float(value):8.3f}" for value in structure.coordinates[site_index])
        if (
            any(len(value) != 8 for value in coordinate_fields)
            or not np.isfinite(structure.coordinates[site_index]).all()
        ):
            raise ValueError(f"PDB cannot encode coordinates for {site!r}")
        if structure.occupancy is None or np.isnan(structure.occupancy[site_index]):
            occupancy_field = " " * 6
        else:
            occupancy = float(structure.occupancy[site_index])
            occupancy_field = f"{occupancy:6.2f}"
            if len(occupancy_field) != 6 or not np.isfinite(occupancy):
                raise ValueError(f"PDB cannot encode occupancy {occupancy!r}")

        b_value = 0.0
        amino_acid_index = amino_acid_index_by_component.get(component_id)
        if isinstance(entity, PolymerSequence) and entity.polymer_type == "protein" and amino_acid_index is not None:
            atom14_names = rc.AMINO_ACID_ATOM14_NAMES[component_id]
            if atom_id in atom14_names:
                atom14_index = atom14_names.index(atom_id)
                rigid_group = int(rc.AMINO_ACID_ATOM14_TO_RIGID_GROUP[amino_acid_index, atom14_index])
                density = density_by_group.get(rigid_group, {}).get(reference)
                if density is not None:
                    b_value = float(density)
        b_field = f"{b_value:6.2f}"
        if len(b_field) != 6 or not np.isfinite(b_value):
            raise ValueError(f"PDB cannot encode chi log density {b_value!r}")

        lines.append(
            f"{record:<6}{serial_field} {atom_field} {component_id:>3} {chain}{residue_field}"
            f"{insertion_code or ' '}   {''.join(coordinate_fields)}{occupancy_field}{b_field}"
            f"          {element:>2}  "
        )

    lines.append("END")
    Path(path).expanduser().write_text("\n".join(lines) + "\n", encoding="ascii")
