r"""
Generic PDB codec for canonical [`ofoldx.data.Structure`][] values.
"""

from __future__ import annotations

import gzip
import io
import math
import textwrap
from collections.abc import Mapping
from collections.abc import Set as AbstractSet
from contextlib import suppress
from dataclasses import replace
from pathlib import Path
from typing import TypeAlias, cast

import numpy as np
import torch

from ..data.components import get_builtin_component_polymer_type, get_component
from ..data.elements import ATOMIC_NUMBER_BY_ELEMENT, NON_METAL_ELEMENTS
from ..data.molecular import (
    BOND_DIRECTION_ENDPOINT_0_DONOR,
    BOND_DIRECTION_ENDPOINT_1_DONOR,
    BOND_DIRECTION_UNDIRECTED,
    BOND_KIND_COVALENT,
    BOND_KIND_DATIVE,
    BOND_ROLE_ASSEMBLY,
    BOND_ROLE_COMPONENT,
    BOND_ROLE_ENTITY,
    BOND_ROLE_POLYMER,
    ENTITY_DNA,
    ENTITY_DNA_RNA_HYBRID,
    ENTITY_MOLECULE,
    ENTITY_PNA,
    ENTITY_PROTEIN,
    ENTITY_RNA,
    MOLECULE_NON_POLYMER,
    MOLECULE_WATER,
    STANDARD_AMINO_ACID_IDS,
    AtomSite,
    ChemicalAtom,
    ChemicalBond,
    ChemicalComponent,
    PolymerType,
    SystemBond,
    is_water_component,
)
from ..data.structure import RESOLUTION_KIND_PDB_REMARK_2, Structure
from .records import (
    atom_row_residue_key,
    atom_rows_from_array,
    reject_duplicate_atom_rows,
    selected_source_sites,
)
from .structure import (
    _graph_component_signatures,
    _pdb_component_polymer_type,
    _realized_component_signature,
)

_STANDARD_PDB_POLYMER_COMPONENT_IDS = STANDARD_AMINO_ACID_IDS | {
    "UNK",
    "A",
    "C",
    "G",
    "I",
    "U",
    "N",
    "DA",
    "DC",
    "DG",
    "DI",
    "DT",
    "DU",
}

_PdbAssemblyPlacements: TypeAlias = dict[str, tuple[np.ndarray, frozenset[str]]]
_PDB_RESOLUTION_PREFIX = "REMARK   2 RESOLUTION."


def parse_pdb(
    path: Path,
    *,
    model: int,
    assembly_id: str | None,
) -> tuple[
    np.recarray,
    tuple[str, ...],
    tuple[tuple[str, str, int, str], ...],
    _PdbAssemblyPlacements | None,
]:
    r"""
    Parse one PDB model or assembly into format-neutral atom records.

    Args:
        path: Source PDB path.
        model: One-based model number to select.
        assembly_id: Biological assembly identifier, or `None` to read the stored model.

    Returns:
        Parsed atom records, original PDB lines, missing residues, and biological-
        assembly placements. The final value is `None` for the stored model.

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
            text = handle.read()
    except UnicodeDecodeError as error:
        raise ValueError("PDB input must be ASCII text") from error
    pdb_file = PDBFile.read(io.StringIO(text))
    lines = tuple(pdb_file.lines)
    parser_lines = list(lines)
    for index, line in enumerate(parser_lines):
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        if not line[54:60].strip():
            line = line[:54] + "  0.00" + line[60:]
        if not line[60:66].strip():
            line = line[:60] + "  0.00" + line[66:]
        parser_lines[index] = line
    if parser_lines != list(lines):
        pdb_file = PDBFile.read(io.StringIO("\n".join(parser_lines) + "\n"))
    try:
        array = pdb_file.get_structure(
            model=model,
            altloc="all",
            extra_fields=["atom_id", "b_factor", "occupancy", "charge"],
            include_bonds=False,
        )
    except IndexError as error:
        raise ValueError(f"PDB does not contain requested model {model}") from error
    rows = atom_rows_from_array(array, format_name="pdb")
    source_atom_lines = _pdb_model_atom_lines(lines, model)
    if len(source_atom_lines) != len(rows):
        raise ValueError("PDB atom records do not form a consistent model")
    for row, line in zip(rows, source_atom_lines, strict=True):
        if not line[54:60].strip():
            row.occupancy = None
        if not line[60:66].strip():
            row.b_factor = None
    placements: _PdbAssemblyPlacements | None = None
    if assembly_id is not None:
        placements = _pdb_assembly_placements(lines, assembly_id)
        copies: list[np.recarray] = []
        for symmetry_id, (transform, chains) in placements.items():
            selected = np.asarray([str(row.auth_chain_id or "") in chains for row in rows], dtype=np.bool_)
            if not bool(selected.any()):
                continue
            copied = rows[selected].copy().view(np.recarray)
            present = np.asarray(copied.coordinate_present, dtype=np.bool_)
            copied.coordinates[present] = copied.coordinates[present] @ transform[:3, :3].T + transform[:3, 3]
            copied.symmetry_id[:] = symmetry_id
            copies.append(copied)
        rows = np.concatenate(copies).view(np.recarray) if copies else np.empty(0, dtype=rows.dtype).view(np.recarray)
        rows.row_index[:] = np.arange(len(rows))
    reject_duplicate_atom_rows(rows)
    return rows, lines, _missing_residues(lines, model), placements


def _pdb_model_atom_lines(pdb_lines: tuple[str, ...], model: int) -> tuple[str, ...]:
    r"""Return the atom records belonging to one one-based PDB model."""

    has_models = any(line.startswith("MODEL ") for line in pdb_lines)
    if not has_models:
        return tuple(line for line in pdb_lines if line.startswith(("ATOM  ", "HETATM"))) if model == 1 else ()

    current_model = 0
    selected = False
    atom_lines: list[str] = []
    for line in pdb_lines:
        if line.startswith("MODEL "):
            current_model += 1
            selected = current_model == model
        elif line.startswith("ENDMDL"):
            selected = False
        elif selected and line.startswith(("ATOM  ", "HETATM")):
            atom_lines.append(line)
    return tuple(atom_lines)


def read_pdb_resolutions(pdb_lines: tuple[str, ...]) -> tuple[tuple[float, int], ...]:
    r"""Read the legacy diffraction resolution from ``REMARK   2``."""

    resolutions: list[float] = []
    record_count = 0
    for line in pdb_lines:
        if not line.startswith(_PDB_RESOLUTION_PREFIX):
            continue
        record_count += 1
        if record_count > 1:
            raise ValueError("PDB contains multiple REMARK 2 resolution declarations")
        content = line[len(_PDB_RESOLUTION_PREFIX) :].strip()
        if content == "NOT APPLICABLE.":
            continue
        fields = content.split()
        if len(fields) != 2 or fields[1] != "ANGSTROMS.":
            raise ValueError(f"invalid PDB REMARK 2 resolution declaration {content!r}")
        try:
            resolution = float(fields[0])
        except ValueError:
            raise ValueError(f"invalid PDB REMARK 2 resolution {fields[0]!r}") from None
        if not math.isfinite(resolution) or resolution <= 0.0:
            raise ValueError(f"invalid PDB REMARK 2 resolution {fields[0]!r}")
        encoded = f"{resolution:7.2f}"
        if len(encoded) > 7 or float(encoded) != resolution:
            raise ValueError(f"PDB REMARK 2 resolution is not representable as F7.2: {fields[0]!r}")
        resolutions.append(resolution)
    return tuple((resolution, RESOLUTION_KIND_PDB_REMARK_2) for resolution in resolutions)


def _pdb_assembly_placements(pdb_lines: tuple[str, ...], assembly_id: str) -> _PdbAssemblyPlacements:
    r"""Read the globally identified BIOMT placements for one biological assembly."""

    selected: list[str] | None = None
    for line in pdb_lines:
        if not line.startswith("REMARK 350"):
            continue
        content = line[10:].strip()
        if content.startswith("BIOMOLECULE:"):
            identifiers = {identifier.strip() for identifier in content.partition(":")[2].split(",")}
            if selected is not None:
                break
            if assembly_id in identifiers:
                selected = []
        elif selected is not None:
            selected.append(content)
    if selected is None:
        raise ValueError(f"PDB does not contain requested assembly {assembly_id!r}")

    operator_rows: dict[str, dict[int, tuple[float, float, float, float]]] = {}
    operator_chains: dict[str, set[str]] = {}
    operator_order: list[str] = []
    chains: set[str] | None = None
    group_has_biomt = False
    for content in selected:
        if content.startswith("APPLY THE FOLLOWING TO CHAINS:"):
            declared = {chain.strip() for chain in content.partition(":")[2].split(",") if chain.strip()}
            if chains is not None and not group_has_biomt:
                chains.update(declared)
            else:
                chains = declared
                group_has_biomt = False
            continue
        if content.startswith("AND CHAINS:"):
            if chains is None:
                raise ValueError("PDB REMARK 350 continues a missing chain declaration")
            chains.update(chain.strip() for chain in content.partition(":")[2].split(",") if chain.strip())
            continue
        fields = content.split()
        if not fields or not fields[0].startswith("BIOMT"):
            continue
        if chains is None or len(fields) != 6:
            raise ValueError("PDB REMARK 350 contains an invalid BIOMT record")
        try:
            row = int(fields[0][-1])
            operator = fields[1]
            values = (float(fields[2]), float(fields[3]), float(fields[4]), float(fields[5]))
        except ValueError:
            raise ValueError("PDB REMARK 350 contains an invalid BIOMT record") from None
        if row not in {1, 2, 3}:
            raise ValueError("PDB REMARK 350 contains an invalid BIOMT record")
        group_has_biomt = True
        if operator not in operator_rows:
            operator_rows[operator] = {}
            operator_order.append(operator)
        previous = operator_rows[operator].setdefault(row, values)
        if previous != values:
            raise ValueError(f"PDB REMARK 350 gives BIOMT operator {operator!r} conflicting rows")
        operator_chains.setdefault(operator, set()).update(chains)

    placements: _PdbAssemblyPlacements = {}
    for symmetry_index, operator in enumerate(operator_order):
        rows = operator_rows[operator]
        if set(rows) != {1, 2, 3}:
            raise ValueError(f"PDB REMARK 350 gives incomplete BIOMT operator {operator!r}")
        transform = np.eye(4, dtype=np.float64)
        transform[:3] = np.asarray([rows[row] for row in (1, 2, 3)], dtype=np.float64)
        placements[str(symmetry_index)] = transform, frozenset(operator_chains[operator])
    if not placements:
        raise ValueError(f"PDB assembly {assembly_id!r} contains no BIOMT placements")
    return placements


def _pdb_symmetry_operation(field: str) -> tuple[str | None, int, tuple[int, int, int]]:
    r"""Decode one legacy PDB crystallographic symmetry operation."""

    text = field.strip() or "1555"
    if len(text) < 4 or not text.isdecimal():
        raise ValueError(f"invalid PDB symmetry operation {field!r}")
    operator = int(text[:-3])
    if operator < 1:
        raise ValueError(f"invalid PDB symmetry operation {field!r}")
    translation = (int(text[-3]) - 5, int(text[-2]) - 5, int(text[-1]) - 5)
    symmetry_id = None if operator == 1 and translation == (0, 0, 0) else f"{operator}{text[-3:]}"
    return symmetry_id, operator, translation


def _pdb_crystallographic_transform_data(
    pdb_lines: tuple[str, ...],
) -> tuple[
    dict[int, dict[int, tuple[float, float, float, float]]],
    dict[int, tuple[float, float, float, float]],
]:
    r"""Read REMARK 290 SMTRY and SCALE matrices used by connection symmetry fields."""

    transformations: dict[int, dict[int, tuple[float, float, float, float]]] = {}
    for line in pdb_lines:
        fields = line.split()
        if len(fields) != 8 or fields[:2] != ["REMARK", "290"] or not fields[2].startswith("SMTRY"):
            continue
        try:
            row = int(fields[2][-1])
            operator = int(fields[3])
            values = (float(fields[4]), float(fields[5]), float(fields[6]), float(fields[7]))
        except ValueError:
            raise ValueError("PDB REMARK 290 contains an invalid SMTRY record") from None
        if row not in {1, 2, 3}:
            raise ValueError("PDB REMARK 290 contains an invalid SMTRY record")
        previous = transformations.setdefault(operator, {}).setdefault(row, values)
        if previous != values:
            raise ValueError(f"PDB REMARK 290 gives symmetry operator {operator} conflicting rows")

    scale_rows: dict[int, tuple[float, float, float, float]] = {}
    for line in pdb_lines:
        fields = line.split()
        if len(fields) != 5 or not fields[0].startswith("SCALE"):
            continue
        try:
            row = int(fields[0][-1])
            values = (float(fields[1]), float(fields[2]), float(fields[3]), float(fields[4]))
        except ValueError:
            raise ValueError("PDB contains an invalid SCALE record") from None
        if row not in {1, 2, 3}:
            raise ValueError("PDB contains an invalid SCALE record")
        previous = scale_rows.setdefault(row, values)
        if previous != values:
            raise ValueError(f"PDB SCALE{row} records conflict")
    return transformations, scale_rows


def _pdb_crystallographic_transform(
    field: str,
    transformations: Mapping[int, Mapping[int, tuple[float, float, float, float]]],
    scale_rows: Mapping[int, tuple[float, float, float, float]],
) -> np.ndarray:
    r"""Return one connection endpoint's Cartesian crystallographic placement."""

    symmetry_id, operator, lattice_translation = _pdb_symmetry_operation(field)
    if symmetry_id is None:
        return np.eye(4, dtype=np.float64)
    transform = transformations.get(operator)
    if transform is None or set(transform) != {1, 2, 3}:
        raise ValueError(f"PDB REMARK 290 does not define symmetry operator {operator}")
    result = np.eye(4, dtype=np.float64)
    result[:3] = np.asarray([transform[row] for row in (1, 2, 3)], dtype=np.float64)
    if lattice_translation != (0, 0, 0):
        if set(scale_rows) != {1, 2, 3}:
            raise ValueError("PDB crystallographic cell translation requires complete SCALE records")
        scale = np.asarray([scale_rows[row][:3] for row in (1, 2, 3)], dtype=np.float64)
        try:
            result[:3, 3] += np.linalg.solve(scale, np.asarray(lattice_translation, dtype=np.float64))
        except np.linalg.LinAlgError:
            raise ValueError("PDB SCALE matrix is singular") from None
    return result


def _pdb_connection_occurrences(
    line: str,
) -> (
    tuple[
        tuple[str | None, int, str | None, str],
        tuple[str | None, int, str | None, str],
    ]
    | None
):
    r"""Return the two component occurrences named by one LINK or SSBOND record."""

    record = line[:6].strip()
    if record == "SSBOND":
        return (
            (line[15:16].strip() or None, decode_hybrid36(line[17:21]), line[21:22].strip() or None, "CYS"),
            (line[29:30].strip() or None, decode_hybrid36(line[31:35]), line[35:36].strip() or None, "CYS"),
        )
    if record == "LINK":
        return (
            (
                line[21:22].strip() or None,
                decode_hybrid36(line[22:26]),
                line[26:27].strip() or None,
                line[17:20].strip(),
            ),
            (
                line[51:52].strip() or None,
                decode_hybrid36(line[52:56]),
                line[56:57].strip() or None,
                line[47:50].strip(),
            ),
        )
    return None


def _pdb_row_occurrence(row: np.record) -> tuple[str | None, int, str | None, str]:
    r"""Return one parsed row's author-space component occurrence."""

    residue_number = row.auth_residue_id if row.auth_residue_id is not None else row.residue_id
    return row.auth_chain_id, int(residue_number), row.insertion_code, str(row.component_id)


def _pdb_link_applies_to_selection(
    line: str,
    selected_rows: np.recarray,
    *,
    include_water: bool,
    components: Mapping[str, ChemicalComponent] | None,
) -> bool:
    r"""Return whether one LINK survives the requested molecular projection."""

    identities = _pdb_connection_occurrences(line)
    if identities is None or line[:6].strip() != "LINK":
        return True
    altlocs = (line[16:17].strip(), line[46:47].strip())
    for endpoint, altloc in zip(identities, altlocs, strict=True):
        if altloc and not any(
            _pdb_row_occurrence(row) == endpoint and row.altloc_id == altloc for row in selected_rows
        ):
            return False
        component = None if components is None else components.get(endpoint[-1])
        if component is None:
            with suppress(KeyError):
                component = get_component(endpoint[-1])
        if not include_water and component is not None and is_water_component(component):
            return False
    return True


def _pdb_connection_symmetry_endpoints(
    pdb_lines: tuple[str, ...],
    selected_rows: np.recarray,
    *,
    include_water: bool,
    components: Mapping[str, ChemicalComponent] | None,
) -> dict[str, tuple[tuple[str | None, int, str | None, str], ...]]:
    r"""Return each nonidentity crystallographic copy and its referenced occurrences."""

    endpoints: dict[str, list[tuple[str | None, int, str | None, str]]] = {}
    for line in pdb_lines:
        identities = _pdb_connection_occurrences(line)
        if identities is None or not _pdb_link_applies_to_selection(
            line,
            selected_rows,
            include_water=include_water,
            components=components,
        ):
            continue
        for field, identity in zip((line[59:65], line[66:72]), identities, strict=True):
            symmetry_id, _, _ = _pdb_symmetry_operation(field)
            if symmetry_id is not None:
                values = endpoints.setdefault(symmetry_id, [])
                if identity not in values:
                    values.append(identity)
    return {symmetry_id: tuple(values) for symmetry_id, values in endpoints.items()}


def _materialize_pdb_crystallographic_copies(
    rows: np.recarray,
    selected_rows: np.recarray,
    pdb_lines: tuple[str, ...],
    endpoints_by_symmetry: Mapping[str, tuple[tuple[str | None, int, str | None, str], ...]],
    *,
    include_water: bool,
    components: Mapping[str, ChemicalComponent] | None,
) -> np.recarray:
    r"""Apply the exact REMARK 290 operators and unit-cell shifts referenced by bonds."""

    transformations, scale_rows = _pdb_crystallographic_transform_data(pdb_lines)

    sequences = pdb_sequences(pdb_lines)
    modified = pdb_modified_residues(pdb_lines)
    segment_by_serial = pdb_segment_by_serial(pdb_lines)

    def is_polymer_row(row: np.record) -> bool:
        chain, residue_number, insertion_code, component_id = _pdb_row_occurrence(row)
        return (
            str(row.group) == "ATOM"
            or (
                chain or "",
                residue_number,
                insertion_code or "",
                component_id,
            )
            in modified
        )

    polymer = np.asarray(
        [is_polymer_row(row) for row in selected_rows],
        dtype=np.bool_,
    )

    def rows_for_occurrence(endpoint: tuple[str | None, int, str | None, str]) -> np.ndarray:
        endpoint_rows = np.asarray([_pdb_row_occurrence(row) == endpoint for row in selected_rows], dtype=np.bool_)
        chain, _, _, _ = endpoint
        endpoint_is_polymer = bool((endpoint_rows & polymer).any()) or (
            not bool(endpoint_rows.any()) and (chain or "") in sequences
        )
        if not endpoint_is_polymer:
            return endpoint_rows
        chain_rows = np.asarray([row.auth_chain_id == chain for row in selected_rows], dtype=np.bool_)
        if (chain or "") in sequences:
            return chain_rows & polymer
        segments = {
            segment_by_serial[str(row.serial)]
            for row in selected_rows[endpoint_rows & polymer]
            if row.serial is not None and str(row.serial) in segment_by_serial
        }
        if len(segments) != 1:
            raise ValueError(f"PDB symmetry endpoint {endpoint!r} does not identify one polymer segment")
        return np.asarray(
            [
                bool(polymer[index])
                and row.auth_chain_id == chain
                and row.serial is not None
                and segment_by_serial.get(str(row.serial)) in segments
                for index, row in enumerate(selected_rows)
            ],
            dtype=np.bool_,
        )

    identity_connections = []
    for line in pdb_lines:
        identities = _pdb_connection_occurrences(line)
        if identities is None or not _pdb_link_applies_to_selection(
            line,
            selected_rows,
            include_water=include_water,
            components=components,
        ):
            continue
        symmetry_a, _, _ = _pdb_symmetry_operation(line[59:65])
        symmetry_b, _, _ = _pdb_symmetry_operation(line[66:72])
        if symmetry_a is None and symmetry_b is None:
            identity_connections.append(identities)

    copies: list[np.recarray] = [rows]
    for symmetry_id, endpoints in endpoints_by_symmetry.items():
        transform = _pdb_crystallographic_transform(symmetry_id, transformations, scale_rows)

        selected = np.zeros(len(selected_rows), dtype=np.bool_)
        for endpoint in endpoints:
            endpoint_rows = rows_for_occurrence(endpoint)
            if not bool(endpoint_rows.any()):
                raise ValueError(f"PDB symmetry endpoint {endpoint!r} is absent from the selected model")
            selected |= endpoint_rows

        changed = True
        while changed:
            changed = False
            for endpoint_a, endpoint_b in identity_connections:
                rows_a = rows_for_occurrence(endpoint_a)
                rows_b = rows_for_occurrence(endpoint_b)
                if not bool((selected & (rows_a | rows_b)).any()):
                    continue
                if not bool(rows_a.any()) or not bool(rows_b.any()):
                    raise ValueError("PDB identity LINK/SSBOND connected to a symmetry copy has a missing endpoint")
                expanded = selected | rows_a | rows_b
                if not np.array_equal(expanded, selected):
                    selected = expanded
                    changed = True
        if not bool(selected.any()):
            raise ValueError(f"PDB symmetry copy {symmetry_id} has no observed atoms to materialize")
        copied = selected_rows[selected].copy().view(np.recarray)
        present = np.asarray(copied.coordinate_present, dtype=np.bool_)
        coordinates = copied.coordinates[present] @ transform[:3, :3].T + transform[:3, 3]
        if not np.isfinite(coordinates).all():
            raise ValueError(f"PDB symmetry operator {symmetry_id} produces non-finite coordinates")
        copied.coordinates[present] = coordinates
        copied.symmetry_id[:] = symmetry_id
        copies.append(copied)

    materialized = np.concatenate(copies).view(np.recarray)
    materialized.row_index[:] = np.arange(len(materialized))
    return materialized


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


def decode_pdb_serial(serial: str) -> str:
    r"""
    Return one PDB atom serial field in its decoded decimal form.

    Some legacy producers use hybrid-36 after exhausting the official decimal range,
    which the atom-record parser accepts for compatibility. Every line-level reader must
    decode the same way, or a ``CONECT`` reference and the atom it names resolve to
    different keys.

    Args:
        serial: Raw five-column serial field, already stripped.

    Returns:
        The decoded serial as a decimal string, or ``serial`` unchanged when it is not
        valid hybrid-36. An undecodable serial is reported by its caller as unknown.
    """

    try:
        return str(decode_hybrid36(serial))
    except ValueError:
        return serial


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
        Segment index keyed by decoded atom serial.
    """

    mapping: dict[str, int] = {}
    segment = 0
    for line in pdb_lines:
        record = line[:6].strip()
        if record in ("ATOM", "HETATM"):
            serial = line[6:11].strip()
            if serial:
                mapping[decode_pdb_serial(serial)] = segment
        elif record == "TER":
            segment += 1
    return mapping


def pdb_entity_by_chain(pdb_lines: tuple[str, ...]) -> dict[str, str]:
    r"""Map PDB chains to explicitly declared COMPND molecular entities."""

    specification = " ".join(line[10:80].strip() for line in pdb_lines if line.startswith("COMPND"))
    mapping: dict[str, str] = {}
    molecule_id: str | None = None
    for statement in specification.split(";"):
        name, separator, value = statement.partition(":")
        if not separator:
            continue
        name = name.strip()
        value = value.strip()
        if name == "MOL_ID":
            if not value.isdecimal() or int(value) < 1:
                raise ValueError(f"PDB COMPND has invalid MOL_ID {value!r}")
            molecule_id = value
            continue
        if name != "CHAIN" or molecule_id is None:
            continue
        entity_id = f"pdb_molecule_{molecule_id}"
        for item in (item.strip() for item in value.split(",")):
            if not item:
                continue
            chain = "" if item == "NULL" else item
            previous = mapping.setdefault(chain, entity_id)
            if previous != entity_id:
                raise ValueError(f"PDB chain {chain!r} belongs to conflicting COMPND molecules")
    return mapping


def pdb_sequences(pdb_lines: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    r"""Read each fixed-column ``SEQRES`` chain declaration in source order."""

    sequences: dict[str, list[str]] = {}
    declared_lengths: dict[str, int] = {}
    for line in pdb_lines:
        if line[:6].strip() != "SEQRES":
            continue
        chain = line[11:12].strip()
        try:
            declared_length = int(line[13:17])
        except ValueError:
            raise ValueError("PDB SEQRES has an invalid residue count") from None
        previous_length = declared_lengths.setdefault(chain, declared_length)
        if previous_length != declared_length:
            raise ValueError(f"PDB SEQRES records for chain {chain!r} declare conflicting residue counts")
        components = [line[start : start + 3].strip() for start in range(19, 70, 4)]
        sequences.setdefault(chain, []).extend(component for component in components if component)
    for chain, sequence in sequences.items():
        declared_length = declared_lengths[chain]
        if len(sequence) != declared_length:
            raise ValueError(
                f"PDB SEQRES chain {chain!r} declares {declared_length} residues but lists {len(sequence)}"
            )
    return {chain: tuple(sequence) for chain, sequence in sequences.items()}


def pdb_modified_residues(pdb_lines: tuple[str, ...]) -> dict[tuple[str, int, str, str], str]:
    r"""Map fixed-column ``MODRES`` occurrence identities to their standard parent IDs."""

    modified: dict[tuple[str, int, str, str], str] = {}
    for line in pdb_lines:
        if line[:6].strip() != "MODRES":
            continue
        component_id = line[12:15].strip()
        standard_id = line[24:27].strip()
        try:
            residue_number = decode_hybrid36(line[18:22])
        except ValueError:
            raise ValueError("PDB MODRES has an invalid residue number") from None
        if not component_id or not standard_id:
            raise ValueError("PDB MODRES must name both modified and standard components")
        key = (line[16:17].strip(), residue_number, line[22:23].strip(), component_id)
        previous = modified.setdefault(key, standard_id)
        if previous != standard_id:
            raise ValueError(f"PDB MODRES occurrence {key!r} has conflicting standard parents")
    return modified


def _pdb_assembly_connection_pairs(
    line: str,
    endpoint_a: tuple[object, ...],
    endpoint_b: tuple[object, ...],
    placements: _PdbAssemblyPlacements,
    transformations: Mapping[int, Mapping[int, tuple[float, float, float, float]]],
    scale_rows: Mapping[int, tuple[float, float, float, float]],
) -> tuple[tuple[str, str], ...]:
    r"""Match assembly copies that realize one crystallographic connection placement."""

    crystal_a = _pdb_crystallographic_transform(line[59:65], transformations, scale_rows)
    crystal_b = _pdb_crystallographic_transform(line[66:72], transformations, scale_rows)
    try:
        inverse_a = np.linalg.inv(crystal_a)
        inverse_b = np.linalg.inv(crystal_b)
    except np.linalg.LinAlgError:
        raise ValueError("PDB connection uses a singular crystallographic symmetry operator") from None
    chain_a = str(endpoint_a[0] or "")
    chain_b = str(endpoint_b[0] or "")
    candidates_a = [
        (symmetry_id, transform @ inverse_a)
        for symmetry_id, (transform, chains) in placements.items()
        if chain_a in chains
    ]
    candidates_b = [
        (symmetry_id, transform @ inverse_b)
        for symmetry_id, (transform, chains) in placements.items()
        if chain_b in chains
    ]
    if np.allclose(crystal_a, crystal_b, rtol=0.0, atol=1e-4):
        candidate_ids_a = {symmetry_id for symmetry_id, _ in candidates_a}
        return tuple((symmetry_id, symmetry_id) for symmetry_id, _ in candidates_b if symmetry_id in candidate_ids_a)
    return tuple(
        (symmetry_a, symmetry_b)
        for symmetry_a, placement_a in candidates_a
        for symmetry_b, placement_b in candidates_b
        if np.allclose(placement_a, placement_b, rtol=0.0, atol=1e-4)
    )


def read_pdb_bonds(
    pdb_lines: tuple[str, ...],
    source_rows: np.recarray,
    selected_rows: np.recarray,
    site_by_row_index: dict[int, AtomSite],
    source_sites: Mapping[tuple[object, ...], AtomSite],
    ambiguous_source_sites: AbstractSet[tuple[object, ...]],
    component_table: Mapping[str, ChemicalComponent],
    *,
    include_water: bool,
    assembly_placements: _PdbAssemblyPlacements | None,
) -> tuple[SystemBond, ...]:
    r"""
    Decode PDB `SSBOND`, `LINK`, and cross-component `CONECT` declarations.

    Args:
        pdb_lines: Original PDB source lines.
        source_rows: All source atom rows before alternate-location selection.
        selected_rows: Atom rows retained for the selected structure.
        site_by_row_index: Canonical atom sites indexed by source row.
        source_sites: Additional canonical sites, including unobserved atoms, keyed by
            source identity.
        ambiguous_source_sites: Source identities that do not resolve uniquely.
        component_table: Resolved component chemistry.
        include_water: Whether water entities are retained.
        assembly_placements: Global biological-assembly placements, or `None` for
            the stored model and explicitly materialized crystallographic copies.

    Returns:
        Resolved covalent and metal-coordination system bonds.

    Raises:
        ValueError: If atom serials are ambiguous or missing, or a declared bond
            identity is ambiguous.
    """

    bonds: dict[frozenset[AtomSite], SystemBond] = {}
    selected_sites, selected_ambiguous = selected_source_sites(selected_rows, site_by_row_index)
    all_sites = dict(source_sites)
    ambiguous_sites = set(ambiguous_source_sites) | selected_ambiguous
    for key, site in selected_sites.items():
        if key in ambiguous_sites:
            continue
        previous = all_sites.get(key)
        if previous is not None and previous != site:
            all_sites.pop(key)
            ambiguous_sites.add(key)
        else:
            all_sites[key] = site
    selected_altlocs: dict[tuple[object, ...], set[str]] = {}
    source_altlocs: dict[tuple[object, ...], set[str]] = {}
    selected_occurrence_altlocs: dict[tuple[object, ...], set[str]] = {}
    source_occurrence_altlocs: dict[tuple[object, ...], set[str]] = {}
    source_occurrences: set[tuple[object, ...]] = set()
    selected_occurrences: set[tuple[object, ...]] = set()
    pdb_occurrences = {
        (
            line[21:22].strip() or None,
            decode_hybrid36(line[22:26]),
            line[26:27].strip() or None,
            line[17:20].strip(),
        )
        for line in pdb_lines
        if line[:6].strip() in {"ATOM", "HETATM"}
    }
    for row in source_rows:
        key = (
            row.symmetry_id,
            row.auth_chain_id,
            row.auth_residue_id,
            row.insertion_code,
            row.auth_component_id,
            row.auth_atom_name,
        )
        altloc = str(row.altloc_id or "")
        source_altlocs.setdefault(key, set()).add(altloc)
        source_occurrence_altlocs.setdefault(key[1:-1], set()).add(altloc)
        source_occurrences.add(key[:-1])
    for row in selected_rows:
        key = (
            row.symmetry_id,
            row.auth_chain_id,
            row.auth_residue_id,
            row.insertion_code,
            row.auth_component_id,
            row.auth_atom_name,
        )
        altloc = str(row.altloc_id or "")
        selected_altlocs.setdefault(key, set()).add(altloc)
        selected_occurrence_altlocs.setdefault(key[1:-1], set()).add(altloc)
        selected_occurrences.add(key[:-1])
    element_by_site = {site_by_row_index[int(row.row_index)]: str(row.element) for row in selected_rows}
    symmetry_ids = tuple(dict.fromkeys(row.symmetry_id for row in selected_rows)) or (None,)
    needs_crystallographic_transform = assembly_placements is not None and any(
        line[:6].strip() in {"LINK", "SSBOND"}
        and any(_pdb_symmetry_operation(field)[0] is not None for field in (line[59:65], line[66:72]))
        for line in pdb_lines
    )
    transformations, scale_rows = (
        _pdb_crystallographic_transform_data(pdb_lines) if needs_crystallographic_transform else ({}, {})
    )
    for line in pdb_lines:
        if line[:6].strip() != "SSBOND":
            continue
        if line[11:14].strip() != "CYS" or line[25:28].strip() != "CYS":
            raise ValueError("PDB SSBOND must name canonical CYS SG endpoints")
        symmetry_a, _, _ = _pdb_symmetry_operation(line[59:65])
        symmetry_b, _, _ = _pdb_symmetry_operation(line[66:72])
        endpoint_a = (
            line[15:16].strip() or None,
            decode_hybrid36(line[17:21]),
            line[21:22].strip() or None,
            "CYS",
            "SG",
        )
        endpoint_b = (
            line[29:30].strip() or None,
            decode_hybrid36(line[31:35]),
            line[35:36].strip() or None,
            "CYS",
            "SG",
        )
        symmetry_pairs = (
            _pdb_assembly_connection_pairs(
                line,
                endpoint_a,
                endpoint_b,
                assembly_placements,
                transformations,
                scale_rows,
            )
            if assembly_placements is not None
            else (
                tuple((symmetry_id, symmetry_id) for symmetry_id in symmetry_ids)
                if symmetry_a is None and symmetry_b is None
                else ((symmetry_a, symmetry_b),)
            )
        )
        for endpoint_symmetry_a, endpoint_symmetry_b in symmetry_pairs:
            key_a = (endpoint_symmetry_a, *endpoint_a)
            key_b = (endpoint_symmetry_b, *endpoint_b)
            outside_a = (key_a[:-1] in source_occurrences or key_a[1:-1] in pdb_occurrences) and key_a[
                :-1
            ] not in selected_occurrences
            outside_b = (key_b[:-1] in source_occurrences or key_b[1:-1] in pdb_occurrences) and key_b[
                :-1
            ] not in selected_occurrences
            if outside_a or outside_b:
                missing_copy_a = endpoint_symmetry_a is not None and key_a[:-1] not in source_occurrences
                missing_copy_b = endpoint_symmetry_b is not None and key_b[:-1] not in source_occurrences
                if missing_copy_a != missing_copy_b:
                    raise ValueError("PDB identity SSBOND connected to a symmetry copy has a missing endpoint")
                continue
            if key_a in ambiguous_sites or key_b in ambiguous_sites:
                raise ValueError("PDB SSBOND endpoint identity is ambiguous")
            site_a = all_sites.get(key_a)
            site_b = all_sites.get(key_b)
            if site_a is None or site_b is None:
                raise ValueError("PDB SSBOND endpoint identity is missing")
            if site_a == site_b:
                raise ValueError("PDB SSBOND endpoints resolve to the same atom")
            leaving: list[frozenset[str]] = []
            for endpoint_symmetry, endpoint, site in (
                (endpoint_symmetry_a, endpoint_a, site_a),
                (endpoint_symmetry_b, endpoint_b, site_b),
            ):
                hydrogen_key = (endpoint_symmetry, *endpoint[:-1], "HG")
                if hydrogen_key in ambiguous_sites:
                    raise ValueError("PDB SSBOND thiol-hydrogen identity is ambiguous")
                hydrogen_site = all_sites.get(hydrogen_key)
                leaving.append(
                    frozenset({"HG"})
                    if hydrogen_site is not None
                    and hydrogen_site.instance_id == site.instance_id
                    and hydrogen_site.component_index == site.component_index
                    else frozenset()
                )
            sites = frozenset((site_a, site_b))
            bonds[sites] = SystemBond(
                site_a,
                site_b,
                order=1.0,
                aromatic=False,
                role="assembly",
                leaving_a=leaving[0],
                leaving_b=leaving[1],
            )

    for line in pdb_lines:
        if line[:6].strip() != "LINK":
            continue
        symmetry_a, _, _ = _pdb_symmetry_operation(line[59:65])
        symmetry_b, _, _ = _pdb_symmetry_operation(line[66:72])
        endpoint_a = (
            line[21:22].strip() or None,
            decode_hybrid36(line[22:26]),
            line[26:27].strip() or None,
            line[17:20].strip(),
            line[12:16].strip(),
        )
        endpoint_b = (
            line[51:52].strip() or None,
            decode_hybrid36(line[52:56]),
            line[56:57].strip() or None,
            line[47:50].strip(),
            line[42:46].strip(),
        )
        if not all((*endpoint_a[-2:], *endpoint_b[-2:])):
            raise ValueError("PDB LINK must name both component and atom endpoints")
        altloc_a = line[16:17].strip()
        altloc_b = line[46:47].strip()
        endpoint_components = tuple(component_table.get(str(endpoint[-2])) for endpoint in (endpoint_a, endpoint_b))
        if not include_water and any(
            component is not None and is_water_component(component) for component in endpoint_components
        ):
            continue
        excluded_altloc = any(
            altloc
            and altloc in source_occurrence_altlocs.get(endpoint[:-1], set())
            and altloc not in selected_occurrence_altlocs.get(endpoint[:-1], set())
            for endpoint, altloc in ((endpoint_a, altloc_a), (endpoint_b, altloc_b))
        )
        if excluded_altloc:
            continue
        symmetry_pairs = (
            _pdb_assembly_connection_pairs(
                line,
                endpoint_a,
                endpoint_b,
                assembly_placements,
                transformations,
                scale_rows,
            )
            if assembly_placements is not None
            else (
                tuple((symmetry_id, symmetry_id) for symmetry_id in symmetry_ids)
                if symmetry_a is None and symmetry_b is None
                else ((symmetry_a, symmetry_b),)
            )
        )
        for endpoint_symmetry_a, endpoint_symmetry_b in symmetry_pairs:
            key_a = (endpoint_symmetry_a, *endpoint_a)
            key_b = (endpoint_symmetry_b, *endpoint_b)
            outside_a = (key_a[:-1] in source_occurrences or key_a[1:-1] in pdb_occurrences) and key_a[
                :-1
            ] not in selected_occurrences
            outside_b = (key_b[:-1] in source_occurrences or key_b[1:-1] in pdb_occurrences) and key_b[
                :-1
            ] not in selected_occurrences
            if outside_a or outside_b:
                missing_copy_a = endpoint_symmetry_a is not None and key_a[:-1] not in source_occurrences
                missing_copy_b = endpoint_symmetry_b is not None and key_b[:-1] not in source_occurrences
                if missing_copy_a != missing_copy_b:
                    raise ValueError("PDB identity LINK connected to a symmetry copy has a missing endpoint")
                continue
            if key_a in ambiguous_sites or key_b in ambiguous_sites:
                raise ValueError("PDB LINK endpoint identity is ambiguous")
            if altloc_a and altloc_a not in selected_altlocs.get(key_a, set()):
                if altloc_a in source_altlocs.get(key_a, set()):
                    continue
                raise ValueError("PDB LINK endpoint alternate location is missing")
            if altloc_b and altloc_b not in selected_altlocs.get(key_b, set()):
                if altloc_b in source_altlocs.get(key_b, set()):
                    continue
                raise ValueError("PDB LINK endpoint alternate location is missing")
            site_a = all_sites.get(key_a)
            site_b = all_sites.get(key_b)
            if site_a is None or site_b is None:
                raise ValueError("PDB LINK endpoint identity is missing")
            if site_a == site_b:
                raise ValueError("PDB LINK endpoints resolve to the same atom")
            if site_a.instance_id == site_b.instance_id and site_a.component_index == site_b.component_index:
                raise ValueError("PDB LINK must connect distinct component occurrences")
            sites = frozenset((site_a, site_b))
            if sites in bonds:
                continue
            elements: list[str | None] = [element_by_site.get(site_a), element_by_site.get(site_b)]
            for endpoint_index, endpoint in enumerate((endpoint_a, endpoint_b)):
                if elements[endpoint_index] is not None:
                    continue
                try:
                    component = get_component(str(endpoint[-2]))
                except KeyError:
                    continue
                elements[endpoint_index] = next(
                    (atom.element for atom in component.atoms if atom.atom_id == endpoint[-1]),
                    None,
                )
            if elements[0] is None or elements[1] is None:
                raise ValueError("PDB LINK chemistry is indeterminate because an endpoint element is unknown")
            metal_a = elements[0] not in NON_METAL_ELEMENTS
            metal_b = elements[1] not in NON_METAL_ELEMENTS
            if metal_a and metal_b:
                raise ValueError("PDB LINK chemistry is indeterminate between two metal endpoints")
            bonds[sites] = SystemBond(
                site_a,
                site_b,
                order=None,
                aromatic=None,
                kind="dative" if metal_a or metal_b else "covalent",
                direction="b_to_a" if metal_a else "a_to_b" if metal_b else "undirected",
                role="assembly",
            )

    for row_a, row_b in _pdb_conect_row_pairs(pdb_lines, selected_rows):
        same_component = atom_row_residue_key(row_a) == atom_row_residue_key(row_b)
        if same_component and row_a.altloc_id is None and row_b.altloc_id is None:
            continue
        site_a = site_by_row_index[int(row_a.row_index)]
        site_b = site_by_row_index[int(row_b.row_index)]
        sites = frozenset((site_a, site_b))
        if not same_component:
            if sites not in bonds:
                raise ValueError(
                    "PDB cross-residue CONECT does not distinguish covalent bonds, hydrogen bonds, and salt "
                    "bridges; a LINK or SSBOND record is required"
                )
            continue
        bonds.setdefault(
            sites,
            SystemBond(
                site_a,
                site_b,
                order=None,
                aromatic=None,
                kind="covalent",
                direction="undirected",
                role="component",
            ),
        )
    return tuple(bonds.values())


def read_pdb_component_bonds(
    pdb_lines: tuple[str, ...],
    selected_rows: np.recarray,
    components: dict[str, ChemicalComponent],
) -> None:
    r"""Merge positive same-component ``CONECT`` adjacency into component chemistry."""

    selected_indices = {int(row.row_index) for row in selected_rows}
    unresolved_component_ids = {
        component_id for component_id, component in components.items() if component.bonds is None
    }
    occurrences_by_component: dict[str, dict[tuple[object, ...], set[str]]] = {}
    for row in selected_rows:
        component_id = str(row.component_id)
        if component_id not in unresolved_component_ids:
            continue
        occurrence = atom_row_residue_key(row)
        occurrences_by_component.setdefault(component_id, {}).setdefault(occurrence, set()).add(str(row.atom_name))
    bonds_by_component: dict[str, set[ChemicalBond]] = {}
    bonds_by_occurrence: dict[tuple[object, ...], set[ChemicalBond]] = {}
    for row_a, row_b in _pdb_conect_row_pairs(pdb_lines, selected_rows):
        occurrence = atom_row_residue_key(row_a)
        if (
            int(row_a.row_index) not in selected_indices
            or int(row_b.row_index) not in selected_indices
            or occurrence != atom_row_residue_key(row_b)
        ):
            continue
        component_id = str(row_a.component_id)
        atom_a, atom_b = sorted((str(row_a.atom_name), str(row_b.atom_name)))
        bond = ChemicalBond(
            atom_a,
            atom_b,
            order=None,
            aromatic=None,
        )
        bonds_by_component.setdefault(component_id, set()).add(bond)
        if component_id in unresolved_component_ids:
            bonds_by_occurrence.setdefault(occurrence, set()).add(bond)
    for component_id, bonds in bonds_by_component.items():
        component = components[component_id]
        if component_id in unresolved_component_ids:
            for occurrence, atoms in occurrences_by_component[component_id].items():
                induced = {bond for bond in bonds if {bond.atom_a, bond.atom_b} <= atoms}
                if bonds_by_occurrence.get(occurrence, set()) != induced:
                    raise ValueError(f"PDB component ID {component_id!r} has conflicting CONECT definitions")
            components[component_id] = replace(component, bonds=frozenset(bonds))
            continue

        existing = {frozenset((bond.atom_a, bond.atom_b)): bond for bond in component.bonds or ()}
        for bond in bonds:
            existing.setdefault(frozenset((bond.atom_a, bond.atom_b)), bond)
        components[component_id] = replace(component, bonds=frozenset(existing.values()))


def _pdb_conect_row_pairs(
    pdb_lines: tuple[str, ...],
    selected_rows: np.recarray,
) -> tuple[tuple[np.record, np.record], ...]:
    r"""Project reference-model ``CONECT`` identities onto retained-model rows."""

    directed_edges: set[tuple[str, str]] = set()
    for line in pdb_lines:
        record = line[:6].strip()
        if record != "CONECT":
            continue
        source_serial = line[6:11].strip()
        if not source_serial:
            continue
        source_serial = decode_pdb_serial(source_serial)
        targets = [line[start : start + 5].strip() for start in (11, 16, 21, 26)]
        targets = [decode_pdb_serial(target) for target in targets if target]
        directed_edges.update((source_serial, target_serial) for target_serial in targets)

    has_models = any(line[:6].strip() == "MODEL" for line in pdb_lines)
    in_reference_model = not has_models
    reference_model_seen = False
    identity_by_serial: dict[str, tuple[tuple[object, ...], str]] = {}
    for line in pdb_lines:
        record = line[:6].strip()
        if record == "MODEL":
            in_reference_model = not reference_model_seen
            reference_model_seen = True
            continue
        if record == "ENDMDL":
            in_reference_model = False
            continue
        if not in_reference_model or record not in {"ATOM", "HETATM"}:
            continue
        serial = line[6:11].strip()
        if not serial:
            continue
        serial = decode_pdb_serial(serial)
        reference_identity = (
            (
                line[21:22].strip() or None,
                decode_hybrid36(line[22:26]),
                line[26:27].strip() or None,
                line[17:20].strip(),
                line[12:16].strip(),
            ),
            line[16:17].strip(),
        )
        previous = identity_by_serial.setdefault(serial, reference_identity)
        if previous != reference_identity:
            raise ValueError(f"PDB atom serial {serial!r} is ambiguous in the reference model")

    declared_serials = {serial for pair in directed_edges for serial in pair}
    missing_serials = declared_serials - identity_by_serial.keys()
    if missing_serials:
        raise ValueError(f"PDB CONECT references unknown atom serials: {sorted(missing_serials)!r}")

    row_by_identity: dict[tuple[object, ...], np.record] = {}
    selected_altlocs: dict[tuple[object, ...], set[str]] = {}
    for row in selected_rows:
        selected_identity = (
            row.symmetry_id,
            row.auth_chain_id,
            row.auth_residue_id,
            row.insertion_code,
            row.auth_component_id,
            row.auth_atom_name,
        )
        altloc = str(row.altloc_id or "")
        row_by_identity[(*selected_identity, altloc)] = row
        selected_altlocs.setdefault(selected_identity, set()).add(altloc)

    symmetry_ids = tuple(dict.fromkeys(row.symmetry_id for row in selected_rows)) or (None,)
    pairs: list[tuple[np.record, np.record]] = []
    identity_pairs: dict[
        frozenset[tuple[tuple[object, ...], str]],
        tuple[tuple[tuple[object, ...], str], tuple[tuple[object, ...], str]],
    ] = {}
    for serial_a, serial_b in sorted(directed_edges):
        identity_a = identity_by_serial[serial_a]
        identity_b = identity_by_serial[serial_b]
        if identity_a != identity_b:
            identity_pairs.setdefault(frozenset((identity_a, identity_b)), (identity_a, identity_b))
    for symmetry_id in symmetry_ids:
        for identity_a, identity_b in identity_pairs.values():
            source_identity_a, altloc_a = identity_a
            source_identity_b, altloc_b = identity_b
            base_a = (symmetry_id, *source_identity_a)
            base_b = (symmetry_id, *source_identity_b)
            key_a = (*base_a, altloc_a)
            key_b = (*base_b, altloc_b)
            row_a = row_by_identity.get(key_a)
            row_b = row_by_identity.get(key_b)
            if (altloc_a and altloc_a not in selected_altlocs.get(base_a, set())) or (
                altloc_b and altloc_b not in selected_altlocs.get(base_b, set())
            ):
                continue
            if row_a is None and row_b is None:
                continue
            if row_a is None or row_b is None:
                if symmetry_id is not None:
                    continue
                raise ValueError("PDB CONECT identity is absent from the selected model")
            pairs.append((row_a, row_b))
    return tuple(pairs)


def write_pdb(structure: Structure, path: Path) -> None:
    r"""Write the observed projection of a Structure as legacy PDB."""

    system = structure.system.cpu()  # type: ignore[attr-defined]
    coordinates = structure.coordinates.detach().cpu()
    coordinate_mask = structure.coordinate_mask.detach().cpu()
    occupancy = structure.occupancy.detach().cpu()
    occupancy_mask = structure.occupancy_mask.detach().cpu()
    b_factor = structure.b_factor.detach().cpu()
    b_factor_mask = structure.b_factor_mask.detach().cpu()
    residue_id = structure.residue_id.detach().cpu()
    insertion_code = structure.insertion_code.detach().cpu()
    standard_component_id = structure.standard_component_id.detach().cpu()
    resolution = structure.resolution.detach().cpu()
    resolution_kind = structure.resolution_kind.detach().cpu()
    instance_entity = system.instance_entity.detach().cpu()
    entity_type = system.entity_type.detach().cpu()
    entity_molecule_type = system.entity_molecule_type.detach().cpu()
    instance_component_ptr = system.instance_component_ptr.detach().cpu()
    component_symbol = system.component_symbol.detach().cpu()
    component_atom_ptr = system.component_atom_ptr.detach().cpu()
    atom_symbol = system.atom_symbol.detach().cpu()
    atomic_numbers = system.atomic_number.detach().cpu()
    isotopes = system.isotope.detach().cpu()
    isotopes_known = system.isotope_known.detach().cpu()
    formal_charges = system.formal_charge.detach().cpu()
    formal_charges_known = system.formal_charge_known.detach().cpu()
    source_formal_charge_mask = structure.source_formal_charge_mask.detach().cpu()
    edge_index = system.edge_index.detach().cpu()
    bond_order = system.bond_order.detach().cpu()
    bond_order_known = system.bond_order_known.detach().cpu()
    bond_aromatic = system.bond_aromatic.detach().cpu()
    bond_aromatic_known = system.bond_aromatic_known.detach().cpu()
    bond_kind = system.bond_kind.detach().cpu()
    bond_direction = system.bond_direction.detach().cpu()
    bond_role = system.bond_role.detach().cpu()
    bond_stereo = system.bond_stereo.detach().cpu()
    bond_stereo_index = system.bond_stereo_index.detach().cpu()
    element_by_number = {number: element for element, number in ATOMIC_NUMBER_BY_ELEMENT.items()}
    component_count = int(component_symbol.shape[0])
    atom_count = int(atom_symbol.shape[0])
    encoded_formal_charge_mask = coordinate_mask & formal_charges_known & formal_charges.ne(0)
    if not torch.equal(source_formal_charge_mask, encoded_formal_charge_mask):
        raise ValueError("PDB cannot preserve which formal charges were explicit in the source; write mmCIF to keep it")
    component_signatures = _graph_component_signatures(system)
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
    polymer_types: dict[int, PolymerType] = {
        ENTITY_PROTEIN: "protein",
        ENTITY_RNA: "rna",
        ENTITY_DNA: "dna",
        ENTITY_DNA_RNA_HYBRID: "dna_rna_hybrid",
        ENTITY_PNA: "pna",
    }
    pdb_component_ids: list[str] = []
    standard_parent_ids: list[str | None] = []
    for component_index in range(component_count):
        instance_index = component_instance[component_index]
        entity_index = int(instance_entity[instance_index])
        type_code = int(entity_type[entity_index])
        component_id = system.symbols[int(component_symbol[component_index])]
        pdb_component_id = "N" if type_code != ENTITY_MOLECULE and component_id == "DN" else component_id
        pdb_component_ids.append(pdb_component_id)
        source_parent = int(standard_component_id[component_index])
        parent_id = None if source_parent < 0 else structure.source_symbols[source_parent]
        standard_parent_ids.append(parent_id)
        if type_code == ENTITY_MOLECULE:
            if parent_id is not None:
                raise ValueError("PDB cannot preserve a standard-parent declaration on a non-polymer component")
            continue
        if parent_id is not None:
            if len(parent_id) > 3 or parent_id not in _STANDARD_PDB_POLYMER_COMPONENT_IDS:
                raise ValueError(f"PDB MODRES parent {parent_id!r} is not a standard polymer component")
            canonical_parent_id = "DN" if parent_id == "N" and type_code == ENTITY_DNA else parent_id
            parent_type = get_builtin_component_polymer_type(canonical_parent_id)
            expected_type = polymer_types[type_code]
            if parent_type != expected_type and not (
                expected_type == "dna_rna_hybrid" and parent_type in {"dna", "rna"}
            ):
                raise ValueError(
                    f"PDB MODRES parent {parent_id!r} is incompatible with {expected_type} component "
                    f"{component_id!r}"
                )
        elif pdb_component_id not in _STANDARD_PDB_POLYMER_COMPONENT_IDS:
            raise ValueError(
                f"PDB cannot encode nonstandard polymer component {component_id!r} without an authoritative "
                "MODRES parent; write mmCIF to keep it"
            )
    polymer_edges_by_pair: dict[tuple[int, int], list[int]] = {}
    for edge in range(edge_index.shape[1]):
        component_a = atom_component[int(edge_index[0, edge])]
        component_b = atom_component[int(edge_index[1, edge])]
        role = int(bond_role[edge])
        if role == BOND_ROLE_COMPONENT:
            if component_a != component_b:
                raise ValueError("a component-owned bond cannot cross component occurrences")
            continue
        if role != BOND_ROLE_POLYMER:
            continue
        pair = (min(component_a, component_b), max(component_a, component_b))
        polymer_edges_by_pair.setdefault(pair, []).append(edge)
    observed_atom_indices = torch.nonzero(coordinate_mask, as_tuple=False).flatten().tolist()
    if not observed_atom_indices:
        raise ValueError(
            "PDB states a structure through its observed atoms and this one observes none, "
            "so the file would carry no chemistry at all; write mmCIF to keep a structure "
            "whose atoms are all unresolved"
        )
    instance_count = int(system.instance_symbol.shape[0])
    instances_by_entity: dict[int, list[int]] = {}
    for instance_index in range(instance_count):
        instances_by_entity.setdefault(int(instance_entity[instance_index]), []).append(instance_index)
    observed_by_component = [
        bool(coordinate_mask[int(component_atom_ptr[index]) : int(component_atom_ptr[index + 1])].any())
        for index in range(component_count)
    ]
    for instance_index in range(instance_count):
        component_start = int(instance_component_ptr[instance_index])
        component_stop = int(instance_component_ptr[instance_index + 1])
        instance_observed = observed_by_component[component_start:component_stop]
        # ``REMARK 465`` can retain missing positions beside an observed polymer segment,
        # but a non-polymer occurrence has no corresponding declaration record.
        if int(entity_type[int(instance_entity[instance_index])]) == ENTITY_MOLECULE:
            for local_index, observed in enumerate(instance_observed):
                if not observed:
                    raise ValueError(
                        "PDB cannot represent unresolved non-polymer component "
                        f"{system.instance_id(instance_index)!r}[{local_index}]; write mmCIF to keep it"
                    )

    position_by_component: dict[int, tuple[int, str]] = {}
    positions_by_instance: dict[int, set[tuple[int, str]]] = {}
    for component_index in range(component_count):
        instance_index = component_instance[component_index]
        source_residue = int(residue_id[component_index])
        if source_residue >= 0:
            author_number = structure.source_symbols[source_residue]
            try:
                number = int(author_number)
            except ValueError:
                raise ValueError(f"PDB cannot encode author residue number {author_number!r}") from None
        else:
            raise ValueError(
                f"PDB cannot preserve an unknown author residue ID for component {component_index}; "
                "write mmCIF to keep it unknown"
            )
        _encode_pdb_residue_number(number)
        source_insertion = int(insertion_code[component_index])
        code = "" if source_insertion < 0 else structure.source_symbols[source_insertion]
        if len(code) > 1:
            raise ValueError(f"PDB cannot encode insertion code {code!r} in one column")
        position = (number, code)
        instance_positions = positions_by_instance.setdefault(instance_index, set())
        if position in instance_positions:
            raise ValueError(
                f"PDB instance {system.instance_id(instance_index)!r} repeats author residue key "
                f"{number}{code}; write mmCIF to keep both components"
            )
        instance_positions.add(position)
        position_by_component[component_index] = position

    for (component_a, component_b), edges in polymer_edges_by_pair.items():
        instance_a = component_instance[component_a]
        instance_b = component_instance[component_b]
        entity_index = int(instance_entity[instance_a])
        if (
            instance_a != instance_b
            or component_b != component_a + 1
            or int(entity_type[entity_index]) == ENTITY_MOLECULE
        ):
            raise ValueError("PDB cannot reconstruct a noncanonical polymer-owned edge; write mmCIF to keep it")
        if len(edges) != 1:
            raise ValueError("PDB cannot reconstruct multiple edges at one polymer junction; write mmCIF to keep them")

    def polymer_component(component_index: int) -> ChemicalComponent | None:
        component_id = system.symbols[int(component_symbol[component_index])]
        try:
            return get_component(component_id)
        except KeyError:
            parent_id = standard_parent_ids[component_index]
            if parent_id is None:
                return None
            instance_index = component_instance[component_index]
            entity_index = int(instance_entity[instance_index])
            if parent_id == "N" and int(entity_type[entity_index]) == ENTITY_DNA:
                parent_id = "DN"
            return get_component(parent_id)

    for instance_index in range(instance_count):
        entity_index = int(instance_entity[instance_index])
        if int(entity_type[entity_index]) == ENTITY_MOLECULE:
            continue
        component_start = int(instance_component_ptr[instance_index])
        component_stop = int(instance_component_ptr[instance_index + 1])
        has_missing_records = not all(observed_by_component[component_start:component_stop])
        ordered_positions = [position_by_component[index] for index in range(component_start, component_stop)]
        if has_missing_records and any(left >= right for left, right in zip(ordered_positions, ordered_positions[1:])):
            raise ValueError(
                f"PDB cannot preserve component order for instance {system.instance_id(instance_index)!r}; "
                "write mmCIF to keep it"
            )

        for left_index in range(component_start, component_stop - 1):
            right_index = left_index + 1
            junction_edges = polymer_edges_by_pair.get((left_index, right_index))
            left_start = int(component_atom_ptr[left_index])
            left_stop = int(component_atom_ptr[left_index + 1])
            right_start = int(component_atom_ptr[right_index])
            right_stop = int(component_atom_ptr[right_index + 1])
            left_component = polymer_component(left_index)
            right_component = polymer_component(right_index)
            left_attachment = None if left_component is None else left_component.next_attachment
            right_attachment = None if right_component is None else right_component.previous_attachment

            observed_left = {
                system.symbols[int(atom_symbol[atom])]
                for atom in range(left_start, left_stop)
                if bool(coordinate_mask[atom])
            }
            observed_right = {
                system.symbols[int(atom_symbol[atom])]
                for atom in range(right_start, right_stop)
                if bool(coordinate_mask[atom])
            }
            visible_terminus = bool(
                (frozenset() if left_attachment is None else left_attachment.leaving_atom_ids) & observed_left
                or (frozenset() if right_attachment is None else right_attachment.leaving_atom_ids) & observed_right
            )
            (number, code), (next_number, next_code) = (
                position_by_component[left_index],
                position_by_component[right_index],
            )
            reader_links = not visible_terminus and (
                has_missing_records or bool(code or next_code) or next_number - number == 1
            )
            reader_forms_link = left_attachment is not None and right_attachment is not None and reader_links

            if junction_edges:
                edge = junction_edges[0]
                if left_attachment is None or right_attachment is None:
                    raise ValueError(
                        "PDB cannot reconstruct a polymer edge whose component attachment is not authoritative; "
                        "write mmCIF to keep it"
                    )
                atom_a = int(edge_index[0, edge])
                atom_b = int(edge_index[1, edge])
                endpoint_by_component = {
                    atom_component[atom_a]: system.symbols[int(atom_symbol[atom_a])],
                    atom_component[atom_b]: system.symbols[int(atom_symbol[atom_b])],
                }
                canonical = (
                    endpoint_by_component.get(left_index) == left_attachment.atom_id
                    and endpoint_by_component.get(right_index) == right_attachment.atom_id
                    and bool(bond_order_known[edge])
                    and float(bond_order[edge]) == 1.0
                    and bool(bond_aromatic_known[edge])
                    and not bool(bond_aromatic[edge])
                    and int(bond_kind[edge]) == BOND_KIND_COVALENT
                    and int(bond_direction[edge]) == BOND_DIRECTION_UNDIRECTED
                    and int(bond_stereo[edge]) < 0
                    and not bool((bond_stereo_index[:, edge] >= 0).any())
                )
                if not canonical:
                    raise ValueError(
                        "PDB cannot reconstruct noncanonical polymer edge chemistry; write mmCIF to keep it"
                    )
                if not reader_forms_link:
                    raise ValueError(
                        "PDB author numbering or observed termini would erase a polymer edge; write mmCIF to keep it"
                    )
                continue

            if left_attachment is None or right_attachment is None:
                continue
            if reader_forms_link:
                raise ValueError("PDB cannot represent an invisible polymer nick; write mmCIF to keep it")

    ssbond_edges: list[int] = []
    link_edges: list[int] = []
    for edge in range(edge_index.shape[1]):
        role = int(bond_role[edge])
        if role < BOND_ROLE_ENTITY:
            continue
        atom_a = int(edge_index[0, edge])
        atom_b = int(edge_index[1, edge])
        component_a = atom_component[atom_a]
        component_b = atom_component[atom_b]
        entity_a = int(instance_entity[component_instance[component_a]])
        entity_b = int(instance_entity[component_instance[component_b]])
        recordable_assembly_bond = (
            role == BOND_ROLE_ASSEMBLY
            and component_a != component_b
            and int(bond_stereo[edge]) < 0
            and not bool((bond_stereo_index[:, edge] >= 0).any())
        )
        metal_a = element_by_number[int(atomic_numbers[atom_a])] not in NON_METAL_ELEMENTS
        metal_b = element_by_number[int(atomic_numbers[atom_b])] not in NON_METAL_ELEMENTS
        covalent = int(bond_kind[edge]) == BOND_KIND_COVALENT and int(bond_direction[edge]) == BOND_DIRECTION_UNDIRECTED
        disulfide = (
            recordable_assembly_bond
            and covalent
            and int(entity_type[entity_a]) == ENTITY_PROTEIN
            and int(entity_type[entity_b]) == ENTITY_PROTEIN
            and system.symbols[int(component_symbol[component_a])] == "CYS"
            and system.symbols[int(component_symbol[component_b])] == "CYS"
            and system.symbols[int(atom_symbol[atom_a])] == "SG"
            and system.symbols[int(atom_symbol[atom_b])] == "SG"
            and bool(bond_order_known[edge])
            and float(bond_order[edge]) == 1.0
            and bool(bond_aromatic_known[edge])
            and not bool(bond_aromatic[edge])
        )
        dative = (
            int(bond_kind[edge]) == BOND_KIND_DATIVE
            and metal_a != metal_b
            and int(bond_direction[edge])
            == (BOND_DIRECTION_ENDPOINT_1_DONOR if metal_a else BOND_DIRECTION_ENDPOINT_0_DONOR)
        )
        link = (
            recordable_assembly_bond
            and not bool(bond_order_known[edge])
            and not bool(bond_aromatic_known[edge])
            and ((covalent and not metal_a and not metal_b) or dative)
        )
        if disulfide:
            ssbond_edges.append(edge)
        elif link:
            link_edges.append(edge)
        else:
            raise ValueError(
                "PDB can encode only canonical CYS disulfides, chemically unspecified nonmetal covalent "
                "links, or directed single-metal coordination; write mmCIF to keep this bond"
            )

    disulfide_components = {
        atom_component[int(edge_index[endpoint, edge])] for edge in ssbond_edges for endpoint in range(2)
    }
    expected_component_signatures: dict[tuple[str, int, bool, bool, bool], tuple[object, ...]] = {}
    for component_index in range(component_count):
        component_id = system.symbols[int(component_symbol[component_index])]
        try:
            authoritative = get_component(component_id)
        except KeyError:
            continue
        instance_index = component_instance[component_index]
        component_start = int(instance_component_ptr[instance_index])
        component_stop = int(instance_component_ptr[instance_index + 1])
        has_previous = (
            component_index > component_start and (component_index - 1, component_index) in polymer_edges_by_pair
        )
        has_next = (
            component_index + 1 < component_stop and (component_index, component_index + 1) in polymer_edges_by_pair
        )
        atom_start = int(component_atom_ptr[component_index])
        entity_index = int(instance_entity[instance_index])
        type_code = int(entity_type[entity_index])
        loses_thiol_hydrogen = component_index in disulfide_components
        expected_key = (component_id, type_code, has_previous, has_next, loses_thiol_hydrogen)
        expected = expected_component_signatures.get(expected_key)
        if expected is None:
            if loses_thiol_hydrogen:
                keep = tuple(atom.atom_id != "HG" for atom in authoritative.atoms)
                retained_atoms = tuple(atom for atom, retained in zip(authoritative.atoms, keep) if retained)
                retained_atom_ids = frozenset(atom.atom_id for atom in retained_atoms)
                authoritative = replace(
                    authoritative,
                    atoms=retained_atoms,
                    bonds=(
                        None
                        if authoritative.bonds is None
                        else frozenset(
                            bond for bond in authoritative.bonds if bond.atom_a != "HG" and bond.atom_b != "HG"
                        )
                    ),
                    reference_coordinates=(
                        tuple(
                            coordinate
                            for coordinate, retained in zip(authoritative.reference_coordinates, keep)
                            if retained
                        )
                        if authoritative.reference_coordinates
                        else ()
                    ),
                    leaving_atom_ids=authoritative.leaving_atom_ids & retained_atom_ids,
                    non_leaving_atom_ids=authoritative.non_leaving_atom_ids & retained_atom_ids,
                )
            expected = _realized_component_signature(
                authoritative,
                None if type_code == ENTITY_MOLECULE else polymer_types[type_code],
                has_previous=has_previous,
                has_next=has_next,
            )
            expected_component_signatures[expected_key] = expected
        actual = component_signatures[component_index]
        actual_atoms = cast(tuple[tuple[object, ...], ...], actual[1])
        expected_atoms = cast(tuple[tuple[object, ...], ...], expected[1])
        normalized_atoms: list[tuple[object, ...]] = []
        if len(actual_atoms) == len(expected_atoms):
            for offset, (actual_atom, expected_atom) in enumerate(zip(actual_atoms, expected_atoms, strict=True)):
                normalized = list(expected_atom)
                if actual_atom[:2] == expected_atom[:2] and bool(coordinate_mask[atom_start + offset]):
                    if actual_atom[2] != expected_atom[2] and actual_atom[1] == 1 and actual_atom[2] in {2, 3}:
                        normalized[2] = actual_atom[2]
                    if actual_atom[3] != expected_atom[3] and (
                        actual_atom[3] not in {None, 0} or (actual_atom[3] == 0 and expected_atom[3] is None)
                    ):
                        normalized[3] = actual_atom[3]
                normalized_atoms.append(tuple(normalized))
        normalized_expected = (expected[0], tuple(normalized_atoms), expected[2])
        if actual != normalized_expected:
            raise ValueError(
                f"PDB component ID {component_id!r} does not match its authoritative definition; "
                "write mmCIF to keep its chemistry"
            )

    unknown_component_indices: list[int] = []
    projected_unknown_atoms: dict[str, dict[str, ChemicalAtom]] = {}
    for component_index in range(component_count):
        component_id = system.symbols[int(component_symbol[component_index])]
        try:
            get_component(component_id)
        except KeyError:
            pass
        else:
            continue
        unknown_component_indices.append(component_index)
        projected_atoms = projected_unknown_atoms.setdefault(component_id, {})
        atom_start = int(component_atom_ptr[component_index])
        atom_stop = int(component_atom_ptr[component_index + 1])
        for atom_index in range(atom_start, atom_stop):
            if not bool(coordinate_mask[atom_index]):
                continue
            atomic_number = int(atomic_numbers[atom_index])
            isotope = (
                int(isotopes[atom_index])
                if atomic_number == 1 and bool(isotopes_known[atom_index]) and int(isotopes[atom_index]) in {2, 3}
                else None
            )
            atom_charge = (
                int(formal_charges[atom_index])
                if bool(formal_charges_known[atom_index]) and int(formal_charges[atom_index]) != 0
                else None
            )
            atom = ChemicalAtom(
                system.symbols[int(atom_symbol[atom_index])],
                element_by_number[atomic_number],
                isotope=isotope,
                formal_charge=atom_charge,
            )
            previous = projected_atoms.setdefault(atom.atom_id, atom)
            if previous != atom:
                raise ValueError(f"PDB cannot encode conflicting definitions for component ID {component_id!r}")

    projected_unknown_bonds: dict[str, set[ChemicalBond]] = {
        component_id: set() for component_id in projected_unknown_atoms
    }
    for edge in range(edge_index.shape[1]):
        if int(bond_role[edge]) != BOND_ROLE_COMPONENT:
            continue
        atom_a = int(edge_index[0, edge])
        atom_b = int(edge_index[1, edge])
        component_id = system.symbols[int(component_symbol[atom_component[atom_a]])]
        component_bonds = projected_unknown_bonds.get(component_id)
        if component_bonds is None or not bool(coordinate_mask[atom_a]) or not bool(coordinate_mask[atom_b]):
            continue
        if (
            bool(bond_order_known[edge])
            or bool(bond_aromatic_known[edge])
            or int(bond_kind[edge]) != BOND_KIND_COVALENT
            or int(bond_direction[edge]) != BOND_DIRECTION_UNDIRECTED
            or int(bond_stereo[edge]) >= 0
            or bool((bond_stereo_index[:, edge] >= 0).any())
        ):
            continue
        component_bonds.add(
            ChemicalBond(
                system.symbols[int(atom_symbol[atom_a])],
                system.symbols[int(atom_symbol[atom_b])],
                order=None,
                aromatic=None,
            )
        )

    projected_unknown_components: dict[str, ChemicalComponent] = {}
    for component_id, atoms in projected_unknown_atoms.items():
        bonds = projected_unknown_bonds[component_id]
        projected_unknown_components[component_id] = ChemicalComponent(
            component_id,
            tuple(atoms.values()),
            bonds=frozenset(bonds) if bonds else None,
        )
    projected_unknown_signatures: dict[str, tuple[object, ...]] = {}
    for component_id, component in projected_unknown_components.items():
        _, signature_atoms, signature_edges = _realized_component_signature(component, None)
        projected_unknown_signatures[component_id] = (False, signature_atoms, signature_edges)
    for component_index in unknown_component_indices:
        component_id = system.symbols[int(component_symbol[component_index])]
        actual = component_signatures[component_index]
        actual_atoms = cast(tuple[tuple[object, ...], ...], actual[1])
        normalized_actual = (actual[0], actual_atoms, actual[2])
        expected = projected_unknown_signatures[component_id]
        if normalized_actual != expected:
            raise ValueError(
                f"PDB cannot reconstruct unknown component {component_id!r} from its atom records; "
                "write mmCIF to keep its chemistry"
            )

    molecule_entity_by_component: dict[str, int] = {}
    molecule_component_by_entity: dict[int, str] = {}
    for instance_index in range(instance_count):
        entity_index = int(instance_entity[instance_index])
        type_code = int(entity_type[entity_index])
        component_start = int(instance_component_ptr[instance_index])
        component_stop = int(instance_component_ptr[instance_index + 1])
        classifications: set[PolymerType] = set()
        molecule_component: ChemicalComponent | None = None
        for component_index in range(component_start, component_stop):
            component_id = system.symbols[int(component_symbol[component_index])]
            try:
                component = get_component(component_id)
            except KeyError:
                component = projected_unknown_components[component_id]
            if component_index == component_start:
                molecule_component = component
            if component_id == "I":
                classification: PolymerType | None = "rna"
            elif component_id in {"DI", "DU"}:
                classification = "dna"
            else:
                classification = _pdb_component_polymer_type(component)
            if classification is not None:
                classifications.add(classification)

        if type_code == ENTITY_MOLECULE:
            recovered_molecule_type = (
                MOLECULE_WATER
                if molecule_component is not None and is_water_component(molecule_component)
                else MOLECULE_NON_POLYMER
            )
            if (
                component_stop - component_start != 1
                or int(entity_molecule_type[entity_index]) != recovered_molecule_type
            ):
                raise ValueError(
                    f"PDB cannot preserve molecule instance {system.instance_id(instance_index)!r}; "
                    "write mmCIF to keep its grouping and type"
                )
            component_id = system.symbols[int(component_symbol[component_start])]
            previous_entity = molecule_entity_by_component.setdefault(component_id, entity_index)
            previous_component = molecule_component_by_entity.setdefault(entity_index, component_id)
            if previous_entity != entity_index or previous_component != component_id:
                raise ValueError(
                    "PDB cannot distinguish molecule entities that do not map one-to-one "
                    "with component IDs; write mmCIF to keep their ownership"
                )
            continue
        recovered_type: PolymerType | None
        if classifications == {"dna", "rna"}:
            recovered_type = "dna_rna_hybrid"
        elif len(classifications) == 1:
            recovered_type = next(iter(classifications))
        else:
            recovered_type = None
        if recovered_type != polymer_types[type_code]:
            raise ValueError(
                f"PDB cannot recover polymer type for instance {system.instance_id(instance_index)!r}; "
                "write mmCIF to keep it"
            )

    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    # A ligand or water instance is named after the chain it was read from
    # (``A:301:ZN``), and a symmetry copy after its operator (``A#0``). Instances from one
    # author chain continue sharing its letter when they are non-polymers. Polymer instances
    # receive distinct chains so COMPND can state their entity ownership without ambiguity.
    base_by_instance = {index: _author_chain(system.instance_id(index)) for index in range(instance_count)}
    distinct_bases = list(dict.fromkeys(base_by_instance.values()))
    if len(distinct_bases) > len(alphabet):
        raise ValueError(
            f"PDB represents at most {len(alphabet)} chains in its one-character chain field; "
            "write mmCIF to keep every instance"
        )
    reserved_chains = {base for base in distinct_bases if base in alphabet}
    primary_chain_by_base: dict[str, str] = {}
    used_chains: set[str] = set()
    for base in distinct_bases:
        if base in alphabet and base not in used_chains:
            chain = base
        else:
            chain = next(
                candidate for candidate in alphabet if candidate not in used_chains and candidate not in reserved_chains
            )
        primary_chain_by_base[base] = chain
        used_chains.add(chain)

    chains_by_base = {base: [chain] for base, chain in primary_chain_by_base.items()}
    occupied_positions_by_chain: dict[str, set[tuple[int, str]]] = {}
    polymer_chains: set[str] = set()
    chain_by_instance: dict[int, str] = {}
    for instance_index in range(instance_count):
        base = base_by_instance[instance_index]
        positions = positions_by_instance[instance_index]
        polymer = int(entity_type[int(instance_entity[instance_index])]) != ENTITY_MOLECULE
        selected = next(
            (
                candidate
                for candidate in chains_by_base[base]
                if positions.isdisjoint(occupied_positions_by_chain.get(candidate, set()))
                and not (polymer and candidate in polymer_chains)
            ),
            None,
        )
        if selected is None:
            selected = next((candidate for candidate in alphabet if candidate not in used_chains), None)
            if selected is None:
                raise ValueError(
                    f"PDB needs more than {len(alphabet)} chains to distinguish repeated author residue keys; "
                    "write mmCIF to keep every instance"
                )
            chains_by_base[base].append(selected)
            used_chains.add(selected)
        chain_by_instance[instance_index] = selected
        occupied_positions_by_chain.setdefault(selected, set()).update(positions)
        if polymer:
            polymer_chains.add(selected)

    # Chain, residue number, and insertion code are the whole identity of a PDB residue, so
    # every component the file mentions -- one carrying ATOM records and one declared through
    # REMARK 465 alike -- has to claim its own key. Two components sharing one key produce a
    # file whose reader cannot tell which of them a record names.
    residue_key_by_component: dict[int, tuple[str, int, str]] = {}
    for component_index in range(component_count):
        instance_index = component_instance[component_index]
        residue_chain = chain_by_instance[instance_index]
        number, code = position_by_component[component_index]
        key = (residue_chain, number, code)
        residue_key_by_component[component_index] = key

    source_molecule_ids: dict[int, str] = {}
    for entity_index in instances_by_entity:
        entity_id = system.entity_id(entity_index)
        if not entity_id.startswith("pdb_molecule_"):
            continue
        molecule_id = entity_id.removeprefix("pdb_molecule_")
        if not molecule_id.isdecimal() or int(molecule_id) < 1:
            raise ValueError(f"PDB cannot preserve molecular entity ID {entity_id!r}")
        source_molecule_ids[entity_index] = molecule_id
    if len(set(source_molecule_ids.values())) != len(source_molecule_ids):
        raise ValueError("PDB molecular entities contain duplicate COMPND MOL_ID values")

    compound_parts: list[str] = []
    used_molecule_ids = set(source_molecule_ids.values())
    next_molecule_id = 1
    for entity_index, instance_indices in instances_by_entity.items():
        source_molecule_id = source_molecule_ids.get(entity_index)
        if int(entity_type[entity_index]) == ENTITY_MOLECULE or (
            len(instance_indices) < 2 and source_molecule_id is None
        ):
            continue
        if source_molecule_id is None:
            while str(next_molecule_id) in used_molecule_ids:
                next_molecule_id += 1
            source_molecule_id = str(next_molecule_id)
            used_molecule_ids.add(source_molecule_id)
            next_molecule_id += 1
        chains = list(dict.fromkeys(chain_by_instance[index] for index in instance_indices))
        compound_parts.extend(
            (
                f"MOL_ID: {source_molecule_id};",
                f"MOLECULE: OFOLDX ENTITY {source_molecule_id};",
                f"CHAIN: {', '.join(chains)};",
            )
        )
    lines: list[str] = []
    if resolution.numel():
        if resolution.numel() != 1 or int(resolution_kind[0]) != RESOLUTION_KIND_PDB_REMARK_2:
            raise ValueError("PDB can encode only one legacy REMARK 2 resolution record")
        value = float(resolution[0])
        encoded_field = f"{value:7.2f}"
        encoded = float(encoded_field)
        if len(encoded_field) > 7 or not bool(torch.equal(resolution[:1], resolution.new_tensor([encoded]))):
            raise ValueError("PDB REMARK 2 cannot preserve this experimental resolution exactly")
        lines.append(f"REMARK   2 RESOLUTION. {encoded_field} ANGSTROMS.")
    if compound_parts:
        for continuation, text in enumerate(
            textwrap.wrap(
                " ".join(compound_parts),
                width=70,
                break_long_words=False,
                break_on_hyphens=False,
            ),
            1,
        ):
            continuation_field = "" if continuation == 1 else str(continuation)
            lines.append(f"COMPND{continuation_field:>4}{text}")
    # A polymer position with no observed atom is still part of the molecule, so it is
    # declared through REMARK 465 rather than dropped -- the reader reads the same record
    # back onto the component axis.
    for component_index, (declared_chain, number, code) in residue_key_by_component.items():
        atom_start = int(component_atom_ptr[component_index])
        atom_stop = int(component_atom_ptr[component_index + 1])
        if bool(coordinate_mask[atom_start:atom_stop].any()):
            continue
        component_id = pdb_component_ids[component_index]
        if len(component_id) > 3:
            raise ValueError(f"PDB cannot encode component ID {component_id!r} in three columns")
        lines.append(f"REMARK 465     {component_id:>3} {declared_chain} {number:>5}{code}")

    for instance_index in range(instance_count):
        entity_index = int(instance_entity[instance_index])
        if int(entity_type[entity_index]) == ENTITY_MOLECULE:
            continue
        component_start = int(instance_component_ptr[instance_index])
        component_stop = int(instance_component_ptr[instance_index + 1])
        component_ids = pdb_component_ids[component_start:component_stop]
        if len(component_ids) > 9999:
            raise ValueError("PDB SEQRES cannot encode more than 9999 residues in one polymer chain")
        chain = chain_by_instance[instance_index]
        for serial, start in enumerate(range(0, len(component_ids), 13), 1):
            residues = " ".join(f"{component_id:>3}" for component_id in component_ids[start : start + 13])
            lines.append(f"SEQRES{serial:>4} {chain}{len(component_ids):>5}  {residues}")

    for component_index, parent_id in enumerate(standard_parent_ids):
        if parent_id is None:
            continue
        component_id = pdb_component_ids[component_index]
        if len(component_id) > 3:
            raise ValueError(f"PDB cannot encode component ID {component_id!r} in three columns")
        chain, number, code = residue_key_by_component[component_index]
        lines.append(
            f"MODRES {'':4} {component_id:>3} {chain} "
            f"{_encode_pdb_residue_number(number)}{code or ' '} {parent_id:>3}  MODIFIED RESIDUE"
        )

    if len(ssbond_edges) > 999:
        raise ValueError("PDB cannot encode more than 999 disulfide bonds")
    for serial, edge in enumerate(ssbond_edges, 1):
        atom_a = int(edge_index[0, edge])
        atom_b = int(edge_index[1, edge])
        chain_a, number_a, code_a = residue_key_by_component[atom_component[atom_a]]
        chain_b, number_b, code_b = residue_key_by_component[atom_component[atom_b]]
        distance_field = " " * 5
        if bool(coordinate_mask[atom_a]) and bool(coordinate_mask[atom_b]):
            distance = float(torch.linalg.vector_norm(coordinates[atom_a] - coordinates[atom_b]))
            distance_field = f"{distance:5.2f}"
            if not np.isfinite(distance) or len(distance_field) > 5:
                raise ValueError("PDB SSBOND distance field range was exceeded")
        lines.append(
            f"SSBOND {serial:>3} CYS {chain_a} {_encode_pdb_residue_number(number_a)}{code_a or ' '}   "
            f"CYS {chain_b} {_encode_pdb_residue_number(number_b)}{code_b or ' '}"
            f"{'':23}{'1555':>6} {'1555':>6} {distance_field}"
        )

    for edge in link_edges:
        atom_a = int(edge_index[0, edge])
        atom_b = int(edge_index[1, edge])
        component_a = atom_component[atom_a]
        component_b = atom_component[atom_b]
        chain_a, number_a, code_a = residue_key_by_component[component_a]
        chain_b, number_b, code_b = residue_key_by_component[component_b]
        component_id_a = pdb_component_ids[component_a]
        component_id_b = pdb_component_ids[component_b]
        atom_id_a = system.symbols[int(atom_symbol[atom_a])]
        atom_id_b = system.symbols[int(atom_symbol[atom_b])]
        if len(component_id_a) > 3 or len(component_id_b) > 3:
            raise ValueError("PDB LINK cannot encode a component ID longer than three characters")
        if len(atom_id_a) > 4 or len(atom_id_b) > 4:
            raise ValueError("PDB LINK cannot encode an atom ID longer than four characters")
        atom_name_a = _format_pdb_atom_name(atom_id_a, element_by_number[int(atomic_numbers[atom_a])])
        atom_name_b = _format_pdb_atom_name(atom_id_b, element_by_number[int(atomic_numbers[atom_b])])
        distance_field = " " * 5
        if bool(coordinate_mask[atom_a]) and bool(coordinate_mask[atom_b]):
            distance = float(torch.linalg.vector_norm(coordinates[atom_a] - coordinates[atom_b]))
            distance_field = f"{distance:5.2f}"
            if not np.isfinite(distance) or len(distance_field) > 5:
                raise ValueError("PDB LINK distance field range was exceeded")
        lines.append(
            f"LINK  {'':6}{atom_name_a} {component_id_a:>3} {chain_a}"
            f"{_encode_pdb_residue_number(number_a)}{code_a or ' '}{'':15}"
            f"{atom_name_b} {component_id_b:>3} {chain_b}"
            f"{_encode_pdb_residue_number(number_b)}{code_b or ' '}{'':16}{distance_field}"
        )

    serial_by_atom: dict[int, int] = {}
    # A polymer instance is closed by TER, which is how the reader recovers the segment
    # boundaries that separate two polymers sharing one author chain.
    open_polymer: tuple[int, str, str, str] | None = None
    next_serial = 1  # noqa: SIM113  # TER consumes a serial too, so this is not the loop index
    for atom_index in observed_atom_indices:
        component_index = atom_component[atom_index]
        instance_index = component_instance[component_index]
        if open_polymer is not None and instance_index != open_polymer[0]:
            lines.append(_terminator(next_serial, open_polymer))
            next_serial += 1
            open_polymer = None
        entity_index = int(instance_entity[instance_index])
        component_id = pdb_component_ids[component_index]
        atom_id = system.symbols[int(atom_symbol[atom_index])]
        if len(component_id) > 3:
            raise ValueError(f"PDB cannot encode component ID {component_id!r} in three columns")
        if len(atom_id) > 4:
            raise ValueError(f"PDB cannot encode atom ID {atom_id!r} in four columns")
        _, number, code = residue_key_by_component[component_index]

        xyz = tuple(float(value) for value in coordinates[atom_index])
        if any(not np.isfinite(value) or value < -999.999 or value > 9999.999 for value in xyz):
            raise ValueError("PDB coordinate field range was exceeded")
        encoded_xyz = tuple(float(f"{value:.3f}") for value in xyz)
        if not torch.equal(coordinates[atom_index], coordinates.new_tensor(encoded_xyz)):
            raise ValueError("PDB cannot preserve atom coordinates exactly at 0.001 angstrom precision")
        if bool(occupancy_mask[atom_index]):
            atom_occupancy = float(occupancy[atom_index])
            occupancy_field = f"{atom_occupancy:6.2f}"
            if (
                not np.isfinite(atom_occupancy)
                or len(occupancy_field) > 6
                or not torch.equal(occupancy.new_tensor(float(occupancy_field)), occupancy[atom_index])
            ):
                raise ValueError("PDB cannot preserve atom occupancy exactly at 0.01 precision")
        else:
            occupancy_field = " " * 6
        if bool(b_factor_mask[atom_index]):
            atom_b_factor = float(b_factor[atom_index])
            b_factor_field = f"{atom_b_factor:6.2f}"
            if (
                not np.isfinite(atom_b_factor)
                or len(b_factor_field) > 6
                or not torch.equal(b_factor.new_tensor(float(b_factor_field)), b_factor[atom_index])
            ):
                raise ValueError("PDB cannot preserve atom B-factor exactly at 0.01 precision")
        else:
            b_factor_field = " " * 6
        atomic_number = int(atomic_numbers[atom_index])
        element = element_by_number[atomic_number]
        if atomic_number == 1 and bool(isotopes_known[atom_index]):
            if int(isotopes[atom_index]) == 2:
                element = "D"
            elif int(isotopes[atom_index]) == 3:
                element = "T"
        charge_value = int(formal_charges[atom_index]) if bool(formal_charges_known[atom_index]) else 0
        if abs(charge_value) > 9:
            raise ValueError(f"PDB cannot encode formal charge {charge_value} in two columns")
        charge = "  " if charge_value == 0 else f"{abs(charge_value)}{'+' if charge_value > 0 else '-'}"
        serial = next_serial
        next_serial += 1
        serial_by_atom[atom_index] = serial
        polymer = int(entity_type[entity_index]) != ENTITY_MOLECULE
        standard_polymer = polymer and component_id in _STANDARD_PDB_POLYMER_COMPONENT_IDS
        group = "ATOM" if standard_polymer else "HETATM"
        atom_name = _format_pdb_atom_name(atom_id, element)
        x, y, z = xyz
        lines.append(
            f"{group:<6}{_encode_pdb_serial(serial)} {atom_name} {component_id:>3} "
            f"{chain_by_instance[instance_index]}{_encode_pdb_residue_number(number)}{code or ' '}   "
            f"{x:>8.3f}{y:>8.3f}{z:>8.3f}{occupancy_field}{b_factor_field}          "
            f"{element:>2}{charge:>2}"
        )
        if polymer:
            open_polymer = (
                instance_index,
                component_id,
                chain_by_instance[instance_index],
                f"{_encode_pdb_residue_number(number)}{code or ' '}",
            )
    if open_polymer is not None:
        lines.append(_terminator(next_serial, open_polymer))
        next_serial += 1

    conect_targets: dict[int, set[int]] = {}
    for bond_index in range(edge_index.shape[1]):
        role = int(bond_role[bond_index])
        source_atom = int(edge_index[0, bond_index])
        target_atom = int(edge_index[1, bond_index])
        if role == BOND_ROLE_COMPONENT:
            component_index = atom_component[source_atom]
            instance_index = component_instance[component_index]
            entity_index = int(instance_entity[instance_index])
            molecule = int(entity_type[entity_index]) == ENTITY_MOLECULE
            standard_polymer = (
                not molecule and pdb_component_ids[component_index] in _STANDARD_PDB_POLYMER_COMPONENT_IDS
            )
            if (
                standard_polymer
                or (molecule and int(entity_molecule_type[entity_index]) == MOLECULE_WATER)
                or not bool(coordinate_mask[source_atom])
                or not bool(coordinate_mask[target_atom])
            ):
                continue
        elif (
            role < BOND_ROLE_ENTITY or not bool(coordinate_mask[source_atom]) or not bool(coordinate_mask[target_atom])
        ):
            continue
        source_serial = serial_by_atom[source_atom]
        target_serial = serial_by_atom[target_atom]
        conect_targets.setdefault(source_serial, set()).add(target_serial)
        conect_targets.setdefault(target_serial, set()).add(source_serial)
    for source_serial, targets in sorted(conect_targets.items()):
        ordered_targets = sorted(targets)
        for start in range(0, len(ordered_targets), 4):
            lines.append(
                f"CONECT{_encode_pdb_serial(source_serial)}"
                + "".join(_encode_pdb_serial(target) for target in ordered_targets[start : start + 4])
            )
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


_HYBRID36_UPPER = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_HYBRID36_LOWER = "0123456789abcdefghijklmnopqrstuvwxyz"


def decode_hybrid36(field: str) -> int:
    r"""
    Decode one hybrid-36 field, which may also be plain decimal.

    Args:
        field: Raw field text; surrounding spaces are ignored.

    Returns:
        The decoded integer.

    Raises:
        ValueError: If the field is empty or is not decimal or base-36 text.
    """

    text = field.strip()
    if not text:
        raise ValueError("hybrid-36 field is empty")
    width = len(text)
    if text[0].isdigit() or text[0] == "-":
        return int(text)
    for index, digits in enumerate((_HYBRID36_UPPER, _HYBRID36_LOWER)):
        if all(character in digits for character in text):
            offset = 10**width - 10 * 36 ** (width - 1) + index * 26 * 36 ** (width - 1)
            return int(text, 36) + offset
    raise ValueError(f"field {field!r} is not hybrid-36")


def _encode_pdb_serial(serial: int) -> str:
    r"""Encode one atom serial within the official five-column decimal field."""

    if not 1 <= serial <= 99999:
        raise ValueError(f"PDB atom serial {serial} is outside the official decimal range 1..99999")
    return f"{serial:>5}"


def _encode_pdb_residue_number(number: int) -> str:
    r"""Encode one author residue number within the official four-column decimal field."""

    if not -999 <= number <= 9999:
        raise ValueError(f"PDB residue number {number} is outside the official decimal range -999..9999")
    return f"{number:>4}"


def _terminator(serial: int, polymer: tuple[int, str, str, str]) -> str:
    r"""Format the ``TER`` record that closes one polymer instance."""

    _, component_id, chain, residue_field = polymer
    return f"TER   {_encode_pdb_serial(serial)}      {component_id:>3} {chain}{residue_field}"


def _author_chain(instance_id: str) -> str:
    r"""
    Return the author chain an instance was realized from.

    ``ofoldx.io`` names a non-polymer instance after its chain and occurrence
    (``A:301:ZN``) and a symmetry copy after its operator (``A#0``); both belong to the
    author chain named first.
    """

    return instance_id.split(":", 1)[0].split("#", 1)[0]


def _format_pdb_atom_name(atom_name: str, element: str) -> str:
    if len(atom_name) == 4:
        return atom_name
    if len(element) == 1 and not atom_name[0].isdigit():
        return f" {atom_name:<3}"
    return f"{atom_name:<4}"
