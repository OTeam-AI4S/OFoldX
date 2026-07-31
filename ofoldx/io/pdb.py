r"""
Generic PDB codec for canonical [`ofoldx.data.Structure`][] values.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import torch

from ..data.elements import ATOMIC_NUMBER_BY_ELEMENT
from ..data.molecular import BOND_ROLE_ENTITY, ENTITY_MOLECULE, AtomSite, SystemBond
from ..data.structure import Structure
from .records import (
    atom_row_residue_key,
    atom_rows_from_array,
    reject_duplicate_atom_rows,
    selected_source_sites,
    source_row_key,
)


def parse_pdb(
    path: Path,
    *,
    model: int,
    assembly_id: str | None,
) -> tuple[np.recarray, tuple[str, ...], tuple[tuple[str, str, int, str], ...]]:
    r"""
    Parse one PDB model or assembly into format-neutral atom records.

    Args:
        path: Source PDB path.
        model: One-based model number to select.
        assembly_id: Biological assembly identifier, or `None` to read the stored model.

    Returns:
        Parsed atom records and original PDB lines.

    Raises:
        ValueError: If the requested model or assembly is unavailable, atom records are
            duplicated, or the file is not ASCII text.
    """

    from biotite.structure.io.pdb import PDBFile

    if path.suffix.lower() == ".gz":
        handle = gzip.open(path, mode="rt", encoding="ascii")
    else:
        handle = path.open(encoding="ascii")
    try:
        with handle:
            pdb_file = PDBFile.read(handle)
    except UnicodeDecodeError as error:
        raise ValueError("PDB input must be ASCII text") from error
    try:
        if assembly_id is None:
            array = pdb_file.get_structure(
                model=model,
                altloc="all",
                extra_fields=["atom_id", "b_factor", "occupancy", "charge"],
                include_bonds=False,
            )
        else:
            array = pdb_file.get_assembly(
                assembly_id=assembly_id,
                model=model,
                altloc="all",
                extra_fields=["atom_id", "b_factor", "occupancy", "charge"],
                include_bonds=False,
            )
    except IndexError as error:
        raise ValueError(f"PDB does not contain requested model {model}") from error
    except KeyError as error:
        raise ValueError(f"PDB does not contain requested assembly {assembly_id!r}") from error
    rows = atom_rows_from_array(array, format_name="pdb")
    reject_duplicate_atom_rows(rows)
    lines = tuple(pdb_file.lines)
    return rows, lines, _missing_residues(lines, model)


def _missing_residues(
    pdb_lines: tuple[str, ...],
    model: int,
) -> tuple[tuple[str, str, int, str], ...]:
    r"""
    Read unmodeled polymer residues explicitly declared by ``REMARK 465``.

    PDB author numbering is not a sequence declaration: absent numbers alone do not
    imply absent residues. This reader therefore considers only fixed-column
    ``REMARK 465`` data rows and filters model-specific declarations to the selected
    model.
    """

    missing: list[tuple[str, str, int, str]] = []
    for line in pdb_lines:
        if not line.startswith("REMARK 465"):
            continue
        component_id = line[15:18].strip()
        residue_number = line[21:26].strip()
        if not component_id or not residue_number:
            continue
        try:
            number = int(residue_number)
        except ValueError:
            continue
        model_text = line[11:14].strip()
        if model_text:
            try:
                declared_model = int(model_text)
            except ValueError:
                continue
            if declared_model != model:
                continue
        declaration = (
            line[19:20].strip(),
            component_id,
            number,
            line[26:27].strip(),
        )
        if declaration not in missing:
            missing.append(declaration)
    return tuple(missing)


def pdb_segment_by_serial(pdb_lines: tuple[str, ...]) -> dict[str, int]:
    r"""
    Map each PDB atom serial to a zero-based polymer segment index.

    A PDB lacks an ``_entity.type`` declaration, so ``TER`` records mark polymer
    boundaries. The segment counter advances at every ``TER``; because a PDB lists each
    chain's atoms contiguously, an internal ``TER`` splits one chain id into segments
    while inter-chain ``TER`` records leave each chain a single segment.

    Args:
        pdb_lines: Original PDB source lines.

    Returns:
        Segment index keyed by atom serial.
    """

    mapping: dict[str, int] = {}
    segment = 0
    for line in pdb_lines:
        record = line[:6].strip()
        if record in ("ATOM", "HETATM"):
            serial = line[6:11].strip()
            if serial:
                mapping[serial] = segment
        elif record == "TER":
            segment += 1
    return mapping


def read_pdb_bonds(
    pdb_lines: tuple[str, ...],
    source_rows: np.recarray,
    selected_rows: np.recarray,
    site_by_row_index: dict[int, AtomSite],
) -> tuple[SystemBond, ...]:
    r"""
    Decode PDB `CONECT` declarations against selected canonical sites.

    Args:
        pdb_lines: Original PDB source lines.
        source_rows: All source atom rows before alternate-location selection.
        selected_rows: Atom rows retained for the selected structure.
        site_by_row_index: Canonical atom sites indexed by source row.

    Returns:
        Resolved covalent system bonds.

    Raises:
        ValueError: If atom serials are ambiguous or missing, or a declared bond
            identity is ambiguous.
    """

    conect_lines = [line for line in pdb_lines if line.startswith("CONECT")]
    directed_edges: set[tuple[str, str]] = set()
    for line in conect_lines:
        source_serial = line[6:11].strip()
        if not source_serial:
            continue
        targets = [line[start : start + 5].strip() for start in range(11, len(line), 5)]
        targets = [target for target in targets if target]
        directed_edges.update((source_serial, target_serial) for target_serial in targets)
    source_row_by_serial: dict[tuple[str | None, str], np.record] = {}
    for row in source_rows:
        if row.serial is None:
            continue
        key = (row.symmetry_id, row.serial)
        previous = source_row_by_serial.get(key)
        if previous is not None and atom_row_residue_key(previous) != atom_row_residue_key(row):
            raise ValueError(f"PDB atom serial {row.serial!r} is ambiguous")
        source_row_by_serial[key] = row
    declared_serials = {serial for pair in directed_edges for serial in pair}
    available_serials = {serial for _, serial in source_row_by_serial}
    missing_serials = declared_serials - available_serials
    if missing_serials:
        raise ValueError(f"PDB CONECT references unknown atom serials: {sorted(missing_serials)!r}")
    selected_sites, _ = selected_source_sites(selected_rows, site_by_row_index)
    symmetry_ids = {row.symmetry_id for row in selected_rows}
    bonds: dict[tuple[str | None, frozenset[str]], SystemBond] = {}
    undirected_serials = {frozenset(pair) for pair in directed_edges if pair[0] != pair[1]}
    for symmetry_id in symmetry_ids:
        for serials in undirected_serials:
            if len(serials) != 2:
                continue
            serial_a, serial_b = sorted(serials)
            row_a = source_row_by_serial.get((symmetry_id, serial_a))
            row_b = source_row_by_serial.get((symmetry_id, serial_b))
            if row_a is None or row_b is None:
                continue
            site_a = selected_sites.get(source_row_key(row_a))
            site_b = selected_sites.get(source_row_key(row_b))
            if site_a is None or site_b is None or site_a == site_b:
                continue
            bond = SystemBond(
                site_a,
                site_b,
                order=None,
                aromatic=None,
                role="assembly",
            )
            bonds[(symmetry_id, serials)] = bond
    return tuple(bonds.values())


def write_pdb(structure: Structure, path: Path) -> None:
    r"""Write the observed projection of a Structure as legacy PDB."""

    system = structure.system
    coordinates = structure.coordinates.detach().cpu()
    coordinate_mask = structure.coordinate_mask.detach().cpu()
    occupancy = structure.occupancy.detach().cpu()
    occupancy_mask = structure.occupancy_mask.detach().cpu()
    b_factor = structure.b_factor.detach().cpu()
    b_factor_mask = structure.b_factor_mask.detach().cpu()
    residue_id = structure.residue_id.detach().cpu()
    insertion_code = structure.insertion_code.detach().cpu()
    instance_entity = system.instance_entity.detach().cpu()
    entity_type = system.entity_type.detach().cpu()
    instance_component_ptr = system.instance_component_ptr.detach().cpu()
    component_symbol = system.component_symbol.detach().cpu()
    component_atom_ptr = system.component_atom_ptr.detach().cpu()
    atom_symbol = system.atom_symbol.detach().cpu()
    atomic_numbers = system.atomic_number.detach().cpu()
    isotopes = system.isotope.detach().cpu()
    isotopes_known = system.isotope_known.detach().cpu()
    formal_charges = system.formal_charge.detach().cpu()
    formal_charges_known = system.formal_charge_known.detach().cpu()
    edge_index = system.edge_index.detach().cpu()
    bond_role = system.bond_role.detach().cpu()
    element_by_number = {number: element for element, number in ATOMIC_NUMBER_BY_ELEMENT.items()}
    component_count = int(component_symbol.shape[0])
    atom_count = int(atom_symbol.shape[0])
    component_instance = [0] * component_count
    atom_component = [0] * atom_count
    for instance_index in range(int(system.instance_symbol.shape[0])):
        component_start = int(instance_component_ptr[instance_index])
        component_stop = int(instance_component_ptr[instance_index + 1])
        for component_index in range(component_start, component_stop):
            component_instance[component_index] = instance_index
            atom_start = int(component_atom_ptr[component_index])
            atom_stop = int(component_atom_ptr[component_index + 1])
            atom_component[atom_start:atom_stop] = [component_index] * (atom_stop - atom_start)

    observed_atom_indices = (
        torch.nonzero(
            coordinate_mask,
            as_tuple=False,
        )
        .flatten()
        .tolist()
    )
    observed_instances: list[int] = []
    seen_instances: set[int] = set()
    for atom_index in observed_atom_indices:
        instance_index = component_instance[atom_component[atom_index]]
        if instance_index not in seen_instances:
            observed_instances.append(instance_index)
            seen_instances.add(instance_index)
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    if len(observed_instances) > len(alphabet):
        raise ValueError("PDB supports at most 62 observed chains")
    chain_by_instance: dict[int, str] = {}
    used_chains: set[str] = set()
    for instance_index in observed_instances:
        preferred = system.instance_id(instance_index)
        if len(preferred) == 1 and preferred.isalnum() and preferred not in used_chains:
            chain = preferred
        else:
            chain = next(candidate for candidate in alphabet if candidate not in used_chains)
        chain_by_instance[instance_index] = chain
        used_chains.add(chain)

    lines: list[str] = []
    serial_by_atom: dict[int, int] = {}
    used_names: dict[tuple[str, int, str], set[str]] = {}
    for atom_index in observed_atom_indices:
        if len(serial_by_atom) >= 99999:
            raise ValueError("PDB atom serial limit of 99999 was exceeded")
        component_index = atom_component[atom_index]
        instance_index = component_instance[component_index]
        entity_index = int(instance_entity[instance_index])
        local_component = component_index - int(instance_component_ptr[instance_index])
        component_id = system.symbols[int(component_symbol[component_index])][:3]
        atom_id = system.symbols[int(atom_symbol[atom_index])][:4]
        source_residue = int(residue_id[component_index])
        if source_residue >= 0:
            try:
                number = int(structure.source_symbols[source_residue])
            except ValueError:
                number = local_component + 1
        else:
            number = local_component + 1
        if number < -999 or number > 9999:
            number = local_component + 1
        source_insertion = int(insertion_code[component_index])
        code = "" if source_insertion < 0 else structure.source_symbols[source_insertion][:1]
        residue_key = (chain_by_instance[instance_index], number, code)
        names = used_names.setdefault(residue_key, set())
        if atom_id in names:
            raise ValueError(f"PDB atom-name truncation collides within residue {residue_key!r}: {atom_id!r}")
        names.add(atom_id)

        xyz = tuple(float(value) for value in coordinates[atom_index])
        if any(not np.isfinite(value) or value < -999.999 or value > 9999.999 for value in xyz):
            raise ValueError("PDB coordinate field range was exceeded")
        atom_occupancy = float(occupancy[atom_index]) if bool(occupancy_mask[atom_index]) else 1.0
        atom_b_factor = float(b_factor[atom_index]) if bool(b_factor_mask[atom_index]) else 0.0
        if not np.isfinite(atom_occupancy) or len(f"{atom_occupancy:6.2f}") > 6:
            raise ValueError("PDB occupancy field range was exceeded")
        if not np.isfinite(atom_b_factor) or len(f"{atom_b_factor:6.2f}") > 6:
            raise ValueError("PDB B-factor field range was exceeded")
        atomic_number = int(atomic_numbers[atom_index])
        element = element_by_number[atomic_number]
        if atomic_number == 1 and bool(isotopes_known[atom_index]):
            if int(isotopes[atom_index]) == 2:
                element = "D"
            elif int(isotopes[atom_index]) == 3:
                element = "T"
        charge_value = int(formal_charges[atom_index]) if bool(formal_charges_known[atom_index]) else 0
        if abs(charge_value) > 9:
            charge_value = 0
        charge = "  " if charge_value == 0 else f"{abs(charge_value)}{'+' if charge_value > 0 else '-'}"
        serial = len(serial_by_atom) + 1
        serial_by_atom[atom_index] = serial
        group = "HETATM" if int(entity_type[entity_index]) == ENTITY_MOLECULE else "ATOM"
        atom_name = _format_pdb_atom_name(atom_id, element)
        x, y, z = xyz
        lines.append(
            f"{group:<6}{serial:>5} {atom_name} {component_id:>3} "
            f"{chain_by_instance[instance_index]}{number:>4}{code or ' '}   "
            f"{x:>8.3f}{y:>8.3f}{z:>8.3f}{atom_occupancy:>6.2f}{atom_b_factor:>6.2f}          "
            f"{element:>2}{charge:>2}"
        )

    for bond_index in range(edge_index.shape[1]):
        if int(bond_role[bond_index]) < BOND_ROLE_ENTITY:
            continue
        source_atom = int(edge_index[0, bond_index])
        target_atom = int(edge_index[1, bond_index])
        source_serial = serial_by_atom.get(source_atom)
        target_serial = serial_by_atom.get(target_atom)
        if source_serial is not None and target_serial is not None:
            lines.append(f"CONECT{source_serial:>5}{target_serial:>5}")
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _format_pdb_atom_name(atom_name: str, element: str) -> str:
    if len(atom_name) >= 4:
        return atom_name[:4]
    if len(element) == 1 and not atom_name[0].isdigit():
        return f" {atom_name:<3}"
    return f"{atom_name:<4}"
