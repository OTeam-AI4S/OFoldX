r"""
Generic PDB codec for canonical [`ofoldx.data.Structure`][] values.
"""

from __future__ import annotations

import gzip
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from ..data.components import get_component
from ..data.molecular import AtomSite, Molecule, PolymerSequence, SystemBond
from ..data.structure import Structure
from .records import (
    AtomRecord,
    ParsedStructure,
    atom_record_residue_key,
    atom_records_from_array,
    ordered_output_bonds,
    reject_duplicate_atom_records,
    selected_source_sites,
    source_record_key,
)


@dataclass(frozen=True, slots=True)
class PdbAtomRecord:
    r"""
    One fully materialized PDB output row.

    Args:
        site: Canonical atom site represented by the row.
        serial: One-based PDB atom serial.
        group: `ATOM` or `HETATM` record category.
        atom_name: PDB atom name.
        altloc_id: Alternate-location identifier.
        component_id: PDB residue name.
        chain_id: One-character PDB chain identifier.
        residue_id: PDB residue number.
        insertion_code: PDB insertion code.
        coordinates: Cartesian coordinates in angstroms.
        occupancy: Atom occupancy.
        b_factor: Isotropic B factor.
        element: Chemical element symbol.
        charge: Formal charge.
    """

    site: AtomSite
    serial: int
    group: Literal["ATOM", "HETATM"]
    atom_name: str
    altloc_id: str
    component_id: str
    chain_id: str
    residue_id: int
    insertion_code: str
    coordinates: tuple[float, float, float]
    occupancy: float
    b_factor: float
    element: str
    charge: int


def parse_pdb(path: Path, *, model: int, assembly_id: str | None) -> ParsedStructure:
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
    rows = atom_records_from_array(array, format_name="pdb")
    reject_duplicate_atom_records(rows)
    return ParsedStructure(
        format="pdb",
        rows=rows,
        canonical_payload=None,
        pdb_lines=tuple(pdb_file.lines),
    )


def read_pdb_bonds(
    parsed: ParsedStructure,
    selected_rows: tuple[AtomRecord, ...],
    site_by_row_index: dict[int, AtomSite],
) -> tuple[SystemBond, ...]:
    r"""
    Decode PDB `CONECT` declarations against selected canonical sites.

    Args:
        parsed: Parsed PDB records and original source lines.
        selected_rows: Atom rows retained for the selected structure.
        site_by_row_index: Canonical atom sites indexed by source row.

    Returns:
        Resolved covalent system bonds.

    Raises:
        ValueError: If atom serials are ambiguous or missing, or a declared bond
            multiplicity is unsupported.
    """

    conect_lines = [line for line in parsed.pdb_lines if line.startswith("CONECT")]
    directed_counts: dict[tuple[str, str], int] = {}
    for line in conect_lines:
        source_serial = line[6:11].strip()
        if not source_serial:
            continue
        targets = [line[start : start + 5].strip() for start in range(11, len(line), 5)]
        targets = [target for target in targets if target]
        for target_serial in set(targets):
            directed_counts[(source_serial, target_serial)] = max(
                directed_counts.get((source_serial, target_serial), 0), targets.count(target_serial)
            )
    source_row_by_serial: dict[tuple[str | None, str], AtomRecord] = {}
    for row in parsed.rows:
        if row.serial is None:
            continue
        key = (row.symmetry_id, row.serial)
        previous = source_row_by_serial.get(key)
        if previous is not None and atom_record_residue_key(previous) != atom_record_residue_key(row):
            raise ValueError(f"PDB atom serial {row.serial!r} is ambiguous")
        source_row_by_serial[key] = row
    declared_serials = {serial for pair in directed_counts for serial in pair}
    available_serials = {serial for _, serial in source_row_by_serial}
    missing_serials = declared_serials - available_serials
    if missing_serials:
        raise ValueError(f"PDB CONECT references unknown atom serials: {sorted(missing_serials)!r}")
    selected_sites = selected_source_sites(selected_rows, site_by_row_index)
    symmetry_ids = {row.symmetry_id for row in selected_rows}
    bonds: dict[tuple[str | None, frozenset[str]], SystemBond] = {}
    undirected_serials = {frozenset(pair) for pair in directed_counts if pair[0] != pair[1]}
    for symmetry_id in symmetry_ids:
        for serials in undirected_serials:
            if len(serials) != 2:
                continue
            serial_a, serial_b = sorted(serials)
            row_a = source_row_by_serial.get((symmetry_id, serial_a))
            row_b = source_row_by_serial.get((symmetry_id, serial_b))
            if row_a is None or row_b is None:
                continue
            site_a = selected_sites.get(source_record_key(row_a))
            site_b = selected_sites.get(source_record_key(row_b))
            if site_a is None or site_b is None or site_a == site_b:
                continue
            order = max(
                directed_counts.get((serial_a, serial_b), 0),
                directed_counts.get((serial_b, serial_a), 0),
                1,
            )
            if order > 4:
                raise ValueError(f"unsupported PDB CONECT bond multiplicity {order}")
            bond = SystemBond(
                site_a,
                site_b,
                order=float(order),
                origin="source",
            )
            bonds[(symmetry_id, serials)] = bond
    return tuple(bonds.values())


def pdb_loss_fields(
    structure: Structure,
    atoms: tuple[PdbAtomRecord, ...],
    dropped: set[str],
    rewritten: set[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    r"""
    Return canonical fields dropped or rewritten by PDB output.

    Args:
        structure: Source structure being serialized.
        atoms: Prepared PDB atom rows.
        dropped: Field tokens already known to be omitted.
        rewritten: Field tokens already known to be changed.

    Returns:
        Sorted tuples of dropped and rewritten field tokens.
    """

    # LossReport covers information represented by Structure and MolecularSystem;
    # unrepresented source annotations are outside its scope.
    system = structure.system
    written_sites = {atom.site for atom in atoms}
    atoms_by_instance: dict[str, list[PdbAtomRecord]] = {}
    for atom in atoms:
        atoms_by_instance.setdefault(atom.site.instance_id, []).append(atom)
    for instance_id, entity_id in system.instances.items():
        instance_atoms = atoms_by_instance.get(instance_id, [])
        if not instance_atoms:
            dropped.add("system.instances")
            continue
        entity = system.entities[entity_id]
        if isinstance(entity, PolymerSequence):
            observed_components = {atom.site.component_index for atom in instance_atoms}
            if observed_components != set(range(len(entity))):
                dropped.add("system.polymer_sequence")
    if any(bond.site_a not in written_sites or bond.site_b not in written_sites for bond in system.bonds):
        dropped.add("system.bonds")
    if any(bond.origin != "source" for bond in system.bonds):
        rewritten.add("system.bond_origins")
    if any(isinstance(entity, Molecule) for entity in system.entities.values()):
        dropped.add("system.nonpolymer_graph")
    polymer_components = {
        component.component_id: component
        for entity in system.entities.values()
        if isinstance(entity, PolymerSequence)
        for component in entity.components
    }
    for component in polymer_components.values():
        try:
            builtin = get_component(component.component_id)
        except KeyError:
            dropped.add("system.nonstandard_components")
            break
        if builtin != component:
            dropped.add("system.nonstandard_components")
            break
    if any(bond.stereo is not None or bond.aromatic or bond.order not in {1.0, 2.0, 3.0, 4.0} for bond in system.bonds):
        dropped.add("system.bond_semantics")
    return tuple(sorted(dropped)), tuple(sorted(rewritten))


def prepare_pdb_atoms(structure: Structure) -> tuple[tuple[PdbAtomRecord, ...], set[str], set[str]]:
    r"""
    Materialize PDB rows and collect dropped or rewritten field tokens.

    Args:
        structure: Structure to lower into PDB atom rows.

    Returns:
        Prepared atom rows, dropped field tokens, and rewritten field tokens.

    Raises:
        ValueError: If PDB chain, serial, atom-name, coordinate, occupancy, or B-factor
            limits cannot represent the structure.
    """

    # The writer derives source-identity columns and emits one conformer.
    residue_numbers = structure.residue_numbers or {}
    dropped: set[str] = set()
    rewritten: set[str] = set()
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    instance_sites: dict[str, list[AtomSite]] = {}
    for site in structure.sites:
        instance_sites.setdefault(site.instance_id, []).append(site)
    if len(instance_sites) > len(alphabet):
        raise ValueError("PDB supports at most 62 distinct one-character chain IDs")
    chain_by_instance: dict[str, str] = {}
    used_chains: set[str] = set()
    for instance_id in instance_sites:
        preferred = instance_id
        if len(preferred) == 1 and preferred not in used_chains and preferred.isalnum():
            chain = preferred
        else:
            chain = next(value for value in alphabet if value not in used_chains)
            rewritten.add("system.instance_ids")
        chain_by_instance[instance_id] = chain
        used_chains.add(chain)

    atoms: list[PdbAtomRecord] = []
    names_by_residue: dict[tuple[str, int, str], set[str]] = {}
    for site_index, site in enumerate(structure.sites):
        if not bool(structure.coordinate_mask[site_index]):
            dropped.add("structure.coordinate_mask")
            if structure.b_factor is not None and not np.isnan(structure.b_factor[site_index]):
                dropped.add("structure.b_factor")
            continue
        if len(atoms) >= 99999:
            raise ValueError("PDB atom serial limit of 99999 was exceeded")
        entity = structure.system.entities[structure.system.instances[site.instance_id]]
        component = structure.system.component((site.instance_id, site.component_index))
        canonical_component_id = component.component_id
        canonical_residue_id = site.component_index + 1
        residue_number = residue_numbers.get((site.instance_id, site.component_index))
        if isinstance(entity, PolymerSequence):
            try:
                get_component(canonical_component_id)
            except KeyError:
                group: Literal["ATOM", "HETATM"] = "HETATM"
            else:
                group = "ATOM"
        else:
            group = "HETATM"
        component_id = str(canonical_component_id)
        if len(component_id) > 3:
            component_id = component_id[:3]
            rewritten.add("system.component_ids")
        residue_id = residue_number.number if residue_number is not None else canonical_residue_id
        residue_id = int(residue_id)
        if residue_id < -999 or residue_id > 9999:
            residue_id = canonical_residue_id
            rewritten.add("structure.author_residue_ids")
        insertion_code = residue_number.insertion_code if residue_number is not None else ""
        if len(insertion_code) > 1:
            insertion_code = insertion_code[0]
            rewritten.add("structure.insertion_codes")
        atom_name = site.atom_id
        if len(atom_name) > 4:
            atom_name = atom_name[:4]
            rewritten.add("system.atom_ids")
        residue_key = (chain_by_instance[site.instance_id], residue_id, insertion_code)
        used_names = names_by_residue.setdefault(residue_key, set())
        if atom_name in used_names:
            raise ValueError(f"PDB atom-name rewrite collides within residue {residue_key!r}: {atom_name!r}")
        used_names.add(atom_name)
        altloc_id = ""
        x, y, z = structure.coordinates[site_index]
        coordinate = (float(x), float(y), float(z))
        if any(not np.isfinite(value) or value < -999.999 or value > 9999.999 for value in coordinate):
            raise ValueError("PDB coordinate field range was exceeded")
        rounded_coordinate = (
            round(coordinate[0], 3),
            round(coordinate[1], 3),
            round(coordinate[2], 3),
        )
        if not np.array_equal(np.asarray(rounded_coordinate, dtype=np.float32), structure.coordinates[site_index]):
            rewritten.add("structure.coordinates")
        if structure.occupancy is None or np.isnan(structure.occupancy[site_index]):
            occupancy = 1.0
            rewritten.add("structure.occupancy")
        else:
            occupancy = float(structure.occupancy[site_index])
            if not np.isfinite(occupancy) or len(f"{occupancy:6.2f}") > 6:
                raise ValueError("PDB occupancy field range was exceeded")
            if np.float32(round(occupancy, 2)) != structure.occupancy[site_index]:
                rewritten.add("structure.occupancy")
        if structure.b_factor is None or np.isnan(structure.b_factor[site_index]):
            b_factor = 0.0
            rewritten.add("structure.b_factor")
        else:
            b_factor = float(structure.b_factor[site_index])
            if not np.isfinite(b_factor) or len(f"{b_factor:6.2f}") > 6:
                raise ValueError("PDB B-factor field range was exceeded")
            if np.float32(round(b_factor, 2)) != structure.b_factor[site_index]:
                rewritten.add("structure.b_factor")
        chemical_atom = structure.system.atom(site)
        charge = chemical_atom.formal_charge or 0
        if abs(charge) > 9:
            dropped.add("system.formal_charge")
            charge = 0
        atoms.append(
            PdbAtomRecord(
                site=site,
                serial=len(atoms) + 1,
                group=group,
                atom_name=atom_name,
                altloc_id=altloc_id,
                component_id=component_id,
                chain_id=chain_by_instance[site.instance_id],
                residue_id=residue_id,
                insertion_code=insertion_code,
                coordinates=rounded_coordinate,
                occupancy=round(occupancy, 2),
                b_factor=round(b_factor, 2),
                element=chemical_atom.element,
                charge=charge,
            )
        )
    return tuple(atoms), dropped, rewritten


def write_pdb(structure: Structure, path: Path, atoms: tuple[PdbAtomRecord, ...]) -> None:
    r"""
    Write prepared PDB atom records and representable system bonds.

    Args:
        structure: Source structure providing the system bond graph.
        path: Destination PDB path.
        atoms: Prepared PDB atom rows.
    """

    lines: list[str] = []
    serial_by_site = {atom.site: atom.serial for atom in atoms}
    for atom in atoms:
        x, y, z = atom.coordinates
        atom_name = _format_pdb_atom_name(atom.atom_name, atom.element)
        charge = "  " if atom.charge == 0 else f"{abs(atom.charge)}{'+' if atom.charge > 0 else '-'}"
        lines.append(
            f"{atom.group:<6}{atom.serial:>5} {atom_name}{atom.altloc_id or ' '}{atom.component_id:>3} "
            f"{atom.chain_id}{atom.residue_id:>4}{atom.insertion_code or ' '}   "
            f"{x:>8.3f}{y:>8.3f}{z:>8.3f}{atom.occupancy:>6.2f}{atom.b_factor:>6.2f}          "
            f"{atom.element:>2}{charge:>2}"
        )
    for bond in ordered_output_bonds(structure.system):
        source_serial = serial_by_site.get(bond.site_a)
        target_serial = serial_by_site.get(bond.site_b)
        if source_serial is None or target_serial is None:
            continue
        multiplicity = int(bond.order) if bond.order in {1.0, 2.0, 3.0, 4.0} else 1
        targets = [target_serial] * multiplicity
        for start in range(0, len(targets), 4):
            lines.append(f"CONECT{source_serial:>5}" + "".join(f"{serial:>5}" for serial in targets[start : start + 4]))
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _format_pdb_atom_name(atom_name: str, element: str) -> str:
    if len(atom_name) >= 4:
        return atom_name[:4]
    if len(element) == 1 and not atom_name[0].isdigit():
        return f" {atom_name:<3}"
    return f"{atom_name:<4}"
