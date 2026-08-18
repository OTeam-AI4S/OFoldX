r"""Standard mmCIF projection of the canonical tensor molecular graph."""

from __future__ import annotations

import math
import re
from collections.abc import Collection, Mapping, Sequence
from fractions import Fraction
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import torch

from ..data.components import (
    get_builtin_component_polymer_type,
    get_component,
    polymer_attachment_anchors,
    terminal_leaving_groups,
)
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
    HYDROGEN_ELEMENTS,
    MOLECULE_BRANCHED,
    MOLECULE_MACROLIDE,
    MOLECULE_NON_POLYMER,
    MOLECULE_WATER,
    AtomSite,
    BondDirection,
    ChemicalAtom,
    ChemicalBond,
    ChemicalComponent,
    Molecule,
    MoleculeBond,
    PolymerAttachment,
    PolymerType,
    SystemBond,
)
from ..data.structure import (
    RESOLUTION_KIND_EM_RECONSTRUCTION,
    RESOLUTION_KIND_REFINEMENT,
    RESOLUTION_KIND_REFLECTIONS,
    Structure,
)
from .records import (
    atom_rows_from_array,
    optional_integer,
    present_string,
    reject_duplicate_atom_rows,
    unique_identifier,
)
from .structure import _graph_component_signatures, _realized_component_signature

if TYPE_CHECKING:
    from biotite.structure.io.pdbx import CIFBlock, CIFCategory, CIFFile


#: ``value_order`` to ``(order, aromatic)``. A Kekulé order carries no aromaticity claim:
#: the wwPDB writes ring bonds as ``sing``/``doub`` alongside ``pdbx_aromatic_flag Y``,
#: so only the explicit ``arom`` order determines the flag.
_CIF_BOND_ORDERS: dict[str, tuple[float, bool | None]] = {
    "sing": (1.0, None),
    "doub": (2.0, None),
    "trip": (3.0, None),
    "quad": (4.0, None),
    "arom": (1.5, True),
}

_MMCIF_RESOLUTION_FIELDS = (
    ("refine", "ls_d_res_high", RESOLUTION_KIND_REFINEMENT),
    ("reflns", "d_resolution_high", RESOLUTION_KIND_REFLECTIONS),
    ("em_3d_reconstruction", "resolution", RESOLUTION_KIND_EM_RECONSTRUCTION),
)
_MMCIF_NUMBER = re.compile(r"(?P<value>-?(?:\d+\.?|\d*\.\d+))(?:\(\d+\))?(?P<exponent>[eE][+-]?\d+)?")


def _is_graph_connection_type(value: str | None) -> bool:
    if value is None:
        return False
    normalized = value.casefold()
    return normalized.startswith(("covale", "disulf", "metalc")) or normalized == "modres"


def _read_mmcif_file(path: Path) -> CIFFile:
    r"""Parse CIF syntax with Gemmi and adapt the document to Biotite's data model."""

    import gemmi
    from biotite.structure.io import pdbx

    document = gemmi.cif.read(str(path))
    return pdbx.CIFFile.deserialize(document.as_string())


def _affine_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def _assembly_operation_steps(expression: str) -> tuple[tuple[str, ...], ...]:
    r"""Expand one PDBx assembly operation expression in application order."""

    groups = [group.strip() for group in expression.replace(")", "").split("(") if group.strip()]
    groups.reverse()
    choices: list[tuple[str, ...]] = []
    for group in groups:
        operation_ids: list[str] = []
        for item in group.split(","):
            item = item.strip()
            endpoints = item.split("-", maxsplit=1)
            if len(endpoints) == 2 and all(endpoint.isdecimal() for endpoint in endpoints):
                first, last = (int(endpoint) for endpoint in endpoints)
                if last < first:
                    raise ValueError(f"invalid mmCIF assembly operation range {item!r}")
                operation_ids.extend(str(operation_id) for operation_id in range(first, last + 1))
            elif item:
                operation_ids.append(item)
        if not operation_ids:
            raise ValueError(f"invalid mmCIF assembly operation expression {expression!r}")
        choices.append(tuple(operation_ids))
    if not choices:
        raise ValueError(f"invalid mmCIF assembly operation expression {expression!r}")
    return tuple(product(*choices))


def _mmcif_assembly_placements(block: object, assembly_id: str) -> dict[tuple[str | None, str], np.ndarray]:
    r"""Assign one global copy ID to each biological assembly operation tuple."""

    cif_block = cast("CIFBlock", block)
    try:
        assembly = cif_block["pdbx_struct_assembly_gen"]
        operators = cif_block["pdbx_struct_oper_list"]
    except KeyError as error:
        raise ValueError("mmCIF biological assembly lacks its operation categories") from error

    transformations: dict[str, np.ndarray] = {}
    operator_ids = _cif_column(operators, "id")
    try:
        matrices = {
            (row, column): operators[f"matrix[{row}][{column}]"].as_array(float)
            for row in (1, 2, 3)
            for column in (1, 2, 3)
        }
        vectors = {row: operators[f"vector[{row}]"].as_array(float) for row in (1, 2, 3)}
    except KeyError as error:
        raise ValueError("mmCIF biological assembly has an incomplete operation") from error
    for index, operator_id in enumerate(operator_ids):
        if operator_id is None:
            raise ValueError("mmCIF biological assembly operation lacks an id")
        transform = _affine_transform(
            np.asarray(
                [[matrices[row, column][index] for column in (1, 2, 3)] for row in (1, 2, 3)],
                dtype=np.float64,
            ),
            np.asarray([vectors[row][index] for row in (1, 2, 3)], dtype=np.float64),
        )
        previous = transformations.setdefault(operator_id, transform)
        if not np.array_equal(previous, transform):
            raise ValueError(f"mmCIF biological assembly operation {operator_id!r} is defined more than once")

    placements: dict[tuple[str | None, str], np.ndarray] = {}
    symmetry_id_by_operation: dict[tuple[str, ...], str] = {}
    for declared_assembly, expression, asym_list in zip(
        _cif_column(assembly, "assembly_id"),
        _cif_column(assembly, "oper_expression"),
        _cif_column(assembly, "asym_id_list"),
    ):
        if declared_assembly != assembly_id:
            continue
        if expression is None or asym_list is None:
            raise ValueError(f"mmCIF assembly {assembly_id!r} has an incomplete generation rule")
        operations = _assembly_operation_steps(expression)
        composed: list[tuple[tuple[str, ...], np.ndarray]] = []
        for operation in operations:
            transform = np.eye(4, dtype=np.float64)
            for operator_id in operation:
                try:
                    transform = transformations[operator_id] @ transform
                except KeyError:
                    raise ValueError(
                        f"mmCIF assembly {assembly_id!r} references unknown operation {operator_id!r}"
                    ) from None
            composed.append((operation, transform))
        asym_ids = tuple(item.strip() for item in asym_list.split(","))
        for asym_id in asym_ids:
            if not asym_id:
                raise ValueError(f"mmCIF assembly {assembly_id!r} has an empty asym id")
        for operation, transform in composed:
            symmetry_id = symmetry_id_by_operation.setdefault(operation, str(len(symmetry_id_by_operation)))
            for asym_id in asym_ids:
                key = symmetry_id, asym_id
                previous = placements.setdefault(key, transform)
                if not np.allclose(previous, transform, rtol=0.0, atol=1e-12):
                    raise ValueError(f"mmCIF assembly placement {key!r} has conflicting transforms")
    if not placements:
        raise ValueError(f"mmCIF assembly {assembly_id!r} has no generation rules")
    return placements


def parse_mmcif(
    path: Path,
    *,
    model: int,
    assembly_id: str | None,
) -> tuple[np.recarray, object, dict[tuple[str | None, str], np.ndarray]]:
    r"""
    Parse one mmCIF model or assembly into transient source columns.

    Args:
        path: Source mmCIF path.
        model: One-based model number to select.
        assembly_id: Biological assembly identifier, or `None` to read the asymmetric unit.

    Returns:
        Selected source atom rows, the mmCIF document, and the Cartesian biological
        placement of every selected label-asym copy.

    Raises:
        ValueError: If the requested model or assembly is unavailable or atom rows are
            duplicated.
    """

    from biotite.structure.io import pdbx

    cif_file = _read_mmcif_file(path)
    block = cif_file.block
    placements: dict[tuple[str | None, str], np.ndarray]
    declared_asym_ids = _cif_column(block["struct_asym"], "id") if "struct_asym" in block else ()
    if any(asym_id is None for asym_id in declared_asym_ids):
        raise ValueError("mmCIF _struct_asym row is missing id")
    atom_site = block.get("atom_site")
    if atom_site is None or atom_site.row_count == 0:
        if model != 1:
            raise ValueError(f"mmCIF does not contain requested model {model}")
        from biotite.structure import AtomArray

        if assembly_id is not None and (
            "pdbx_struct_assembly_gen" not in block or "pdbx_struct_oper_list" not in block
        ):
            raise ValueError("topology-only mmCIF cannot materialize an assembly without operation categories")
        placements = (
            {(None, str(asym_id)): np.eye(4, dtype=np.float64) for asym_id in declared_asym_ids}
            if assembly_id is None
            else _mmcif_assembly_placements(block, assembly_id)
        )
        return atom_rows_from_array(AtomArray(0), format_name="mmcif"), cif_file, placements
    available = set(atom_site.keys())
    requested: list[str] = [
        field
        for field in (
            "auth_asym_id",
            "auth_seq_id",
            "auth_comp_id",
            "auth_atom_id",
            "label_entity_id",
            "label_asym_id",
            "label_seq_id",
            "label_comp_id",
            "label_atom_id",
        )
        if field in available
    ]
    for special, source_column in (
        ("atom_id", "id"),
        ("b_factor", "B_iso_or_equiv"),
        ("occupancy", "occupancy"),
        ("pdbx_formal_charge", "pdbx_formal_charge"),
    ):
        if source_column in available:
            requested.append(special)
    try:
        array = pdbx.get_structure(
            cif_file,
            model=model,
            altloc="all",
            extra_fields=requested,
            use_author_fields=False,
            include_bonds=False,
        )
    except IndexError as error:
        raise ValueError(f"mmCIF does not contain requested model {model}") from error
    rows = atom_rows_from_array(array, format_name="mmcif")
    source_ids = _cif_column(atom_site, "id")
    if not all(source_id is not None for source_id in source_ids):
        raise ValueError("mmCIF _atom_site.id is required")
    source_occupancies = (
        _cif_column(atom_site, "occupancy") if "occupancy" in atom_site else ("1.0",) * atom_site.row_count
    )
    source_values = {
        source_id: (occupancy, b_factor, charge, author_number)
        for source_id, occupancy, b_factor, charge, author_number in zip(
            source_ids,
            source_occupancies,
            _cif_column(atom_site, "B_iso_or_equiv"),
            _cif_column(atom_site, "pdbx_formal_charge"),
            _cif_column(atom_site, "auth_seq_id"),
        )
    }
    if len(source_values) != len(source_ids):
        raise ValueError("mmCIF _atom_site.id values must be unique")
    row_serials = rows["serial"]
    row_occupancies = rows["occupancy"]
    row_b_factors = rows["b_factor"]
    row_charges = rows["charge"]
    row_auth_residue_ids = rows["auth_residue_id"]
    for row_index in range(len(rows)):
        source_id = str(row_serials[row_index])
        try:
            occupancy, b_factor, charge, author_number = source_values[source_id]
        except KeyError:
            raise ValueError(f"mmCIF assembly atom {source_id!r} has no source _atom_site row") from None
        row_occupancies[row_index] = None if occupancy is None else float(occupancy)
        row_b_factors[row_index] = None if b_factor is None else float(b_factor)
        row_charges[row_index] = optional_integer(charge)
        row_auth_residue_ids[row_index] = author_number
    if assembly_id is None:
        asym_ids = declared_asym_ids or tuple(dict.fromkeys(str(chain_id) for chain_id in rows["chain_id"]))
        placements = {(None, str(asym_id)): np.eye(4, dtype=np.float64) for asym_id in asym_ids}
    else:
        placements = _mmcif_assembly_placements(block, assembly_id)
        copies: list[np.recarray] = []
        for (symmetry_id, asym_id), transform in placements.items():
            copied = rows[np.asarray(rows["chain_id"] == asym_id, dtype=np.bool_)].copy().view(np.recarray)
            if not len(copied):
                continue
            present = np.asarray(copied.coordinate_present, dtype=np.bool_)
            copied.coordinates[present] = copied.coordinates[present] @ transform[:3, :3].T + transform[:3, 3]
            copied.symmetry_id[:] = symmetry_id
            copies.append(copied)
        rows = np.concatenate(copies).view(np.recarray) if copies else np.empty(0, dtype=rows.dtype).view(np.recarray)
        rows.row_index[:] = np.arange(len(rows))
    reject_duplicate_atom_rows(rows)
    return rows, cif_file, placements


def _cif_column(category: CIFCategory, name: str) -> tuple[str | None, ...]:
    if name not in category:
        return (None,) * category.row_count
    return tuple(present_string(value) for value in category[name].as_array())


def read_mmcif_resolutions(cif_file: object) -> tuple[tuple[float, int], ...]:
    r"""Read every typed experimental-resolution value, in angstroms."""

    block = cast("CIFFile", cif_file).block
    resolutions: list[tuple[float, int]] = []
    for category_name, column_name, kind in _MMCIF_RESOLUTION_FIELDS:
        if category_name not in block or column_name not in block[category_name]:
            continue
        for value in _cif_column(block[category_name], column_name):
            if value is None:
                continue
            match = _MMCIF_NUMBER.fullmatch(value)
            if match is None:
                raise ValueError(f"invalid mmCIF experimental resolution {value!r}")
            resolution = float(match["value"] + (match["exponent"] or ""))
            if not math.isfinite(resolution) or resolution <= 0.0:
                raise ValueError(f"invalid mmCIF experimental resolution {value!r}")
            resolutions.append((resolution, kind))
    return tuple(resolutions)


def _read_mmcif_atom_provenance(cif_file: object) -> frozenset[AtomSite] | None:
    r"""Read the OFoldX atom provenance needed for lossless Structure round trips."""

    block = cast("CIFFile", cif_file).block
    marker = block.get("ofoldx_provenance")
    if marker is None:
        return None
    if _cif_column(marker, "formal_charge_source_complete") != ("Y",):
        raise ValueError("mmCIF _ofoldx_provenance.formal_charge_source_complete must be one Y value")
    category = block.get("ofoldx_source_formal_charge")
    if category is None:
        return frozenset()
    required = {"label_asym_id", "component_index", "label_atom_id"}
    if not required.issubset(category):
        raise ValueError("mmCIF _ofoldx_source_formal_charge is incomplete")

    provenance: set[AtomSite] = set()
    for asym_id, component_index, atom_id in zip(
        _cif_column(category, "label_asym_id"),
        _cif_column(category, "component_index"),
        _cif_column(category, "label_atom_id"),
        strict=True,
    ):
        index = optional_integer(component_index)
        if asym_id is None or index is None or index < 0 or atom_id is None:
            raise ValueError("mmCIF _ofoldx_source_formal_charge has an invalid atom identity")
        site = AtomSite(asym_id, index, atom_id)
        if site in provenance:
            raise ValueError(f"mmCIF _ofoldx_source_formal_charge repeats {site!r}")
        provenance.add(site)
    return frozenset(provenance)


def _validate_mmcif_atom_provenance(
    cif_file: object,
    provenance: frozenset[AtomSite],
    components: Mapping[str, ChemicalComponent],
) -> None:
    r"""Require every OFoldX provenance row to name one declared source atom."""

    if not provenance:
        return

    block = cast("CIFFile", cif_file).block
    struct_asym = block.get("struct_asym")
    declared_asym_ids = frozenset() if struct_asym is None else frozenset(_cif_column(struct_asym, "id"))
    components_by_position: dict[tuple[str, int], set[str]] = {}

    def add_positions(
        category_name: str,
        position_name: str | None,
    ) -> None:
        category = block.get(category_name)
        if category is None:
            return
        asym_ids = _cif_column(category, "asym_id")
        positions = ("1",) * len(asym_ids) if position_name is None else _cif_column(category, position_name)
        for asym_id, position, component_id in zip(
            asym_ids,
            positions,
            _cif_column(category, "mon_id"),
            strict=True,
        ):
            component_index = optional_integer(position)
            if asym_id is None or component_index is None or component_index < 1 or component_id is None:
                raise ValueError(f"mmCIF _{category_name} cannot identify an OFoldX provenance component")
            key = asym_id, component_index - 1
            components_by_position.setdefault(key, set()).add(component_id)

    add_positions("pdbx_poly_seq_scheme", "seq_id")
    add_positions("pdbx_branch_scheme", "num")
    add_positions("pdbx_nonpoly_scheme", None)

    atom_ids_by_component = {
        component_id: frozenset(atom.atom_id for atom in component.atoms)
        for component_id, component in components.items()
    }
    for site in provenance:
        component_ids = components_by_position.get((site.instance_id, site.component_index), ())
        if (
            site.instance_id not in declared_asym_ids
            or not component_ids
            or not any(site.atom_id in atom_ids_by_component.get(component_id, ()) for component_id in component_ids)
        ):
            raise ValueError(f"mmCIF _ofoldx_source_formal_charge references unknown atom {site!r}")


def _cif_points(
    category: CIFCategory,
    x_name: str,
    y_name: str,
    z_name: str,
) -> tuple[tuple[float, float, float] | None, ...]:
    r"""One Cartesian column triple, with `None` where a row states no complete point."""

    return tuple(
        None if x is None or y is None or z is None else (float(x), float(y), float(z))
        for x, y, z in zip(
            _cif_column(category, x_name),
            _cif_column(category, y_name),
            _cif_column(category, z_name),
        )
    )


def _fractional_symmetry_operation(expression: str) -> np.ndarray:
    r"""Parse one official ``operation_xyz`` affine transform in fractional space."""

    coordinates = expression.lower().replace(" ", "").split(",")
    if len(coordinates) != 3:
        raise ValueError(f"invalid mmCIF space-group operation {expression!r}")
    rotation = np.zeros((3, 3), dtype=np.float64)
    translation = np.zeros(3, dtype=np.float64)
    for output_axis, coordinate in enumerate(coordinates):
        terms = coordinate.replace("-", "+-").split("+")
        for term in (term for term in terms if term):
            variables = [axis for axis in "xyz" if axis in term]
            if variables:
                if len(variables) != 1 or term not in {variables[0], f"-{variables[0]}"}:
                    raise ValueError(f"invalid mmCIF space-group operation {expression!r}")
                input_axis = "xyz".index(variables[0])
                rotation[output_axis, input_axis] += -1.0 if term.startswith("-") else 1.0
            else:
                try:
                    translation[output_axis] += float(Fraction(term))
                except (ValueError, ZeroDivisionError):
                    raise ValueError(f"invalid mmCIF space-group operation {expression!r}") from None
    determinant = np.linalg.det(rotation)
    if not np.isclose(abs(determinant), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError(f"invalid mmCIF space-group operation {expression!r}")
    return _affine_transform(rotation, translation)


def _mmcif_symmetry_operators(block: CIFBlock) -> dict[str, np.ndarray]:
    r"""Read current and legacy official fractional space-group operation tables."""

    operators: dict[str, np.ndarray] = {}
    for category_name, expression_name in (
        ("space_group_symop", "operation_xyz"),
        ("symmetry_equiv", "pos_as_xyz"),
    ):
        if category_name not in block:
            continue
        category = block[category_name]
        for operator_id, expression in zip(
            _cif_column(category, "id"),
            _cif_column(category, expression_name),
        ):
            if operator_id is None or expression is None:
                raise ValueError(f"mmCIF {category_name} row lacks its id or operation")
            transform = _fractional_symmetry_operation(expression)
            previous = operators.setdefault(operator_id, transform)
            if not np.allclose(previous, transform, rtol=0.0, atol=1e-12):
                raise ValueError(f"mmCIF space-group operation {operator_id!r} has conflicting definitions")
    return operators


def _mmcif_fractionalization(block: CIFBlock) -> np.ndarray:
    r"""Return the homogeneous Cartesian-to-fractional transform declared by mmCIF."""

    if "atom_sites" not in block:
        raise ValueError("mmCIF crystallographic symmetry requires _atom_sites coordinate transforms")
    category = block["atom_sites"]

    def declared_transform(prefix: str) -> np.ndarray | None:
        matrix_names = [f"{prefix}_transf_matrix[{row}][{column}]" for row in (1, 2, 3) for column in (1, 2, 3)]
        vector_names = [f"{prefix}_transf_vector[{row}]" for row in (1, 2, 3)]
        names = matrix_names + vector_names
        values = [(_cif_column(category, name) or (None,))[0] for name in names]
        if all(value is None for value in values):
            return None
        if any(value is None for value in values):
            raise ValueError(f"mmCIF _atom_sites has an incomplete {prefix}_transf transform")
        try:
            numeric = np.asarray(values, dtype=np.float64)
        except ValueError:
            raise ValueError(f"mmCIF _atom_sites has an invalid {prefix}_transf transform") from None
        matrix = numeric[:9].reshape(3, 3)
        vector = numeric[9:]
        return _affine_transform(matrix, vector)

    fractional = declared_transform("fract")
    cartesian = declared_transform("Cartn")
    if fractional is None and cartesian is None:
        raise ValueError("mmCIF crystallographic symmetry requires _atom_sites coordinate transforms")
    if fractional is None:
        assert cartesian is not None
        try:
            fractional = np.linalg.inv(cartesian)
        except np.linalg.LinAlgError:
            raise ValueError("mmCIF _atom_sites Cartesian transform is singular") from None
    elif cartesian is not None and not np.allclose(cartesian @ fractional, np.eye(4), rtol=0.0, atol=1e-5):
        raise ValueError("mmCIF _atom_sites Cartesian and fractional transforms disagree")
    try:
        np.linalg.inv(fractional)
    except np.linalg.LinAlgError:
        raise ValueError("mmCIF _atom_sites fractional transform is singular") from None
    return fractional


def _mmcif_crystallographic_code(symmetry: str | None) -> tuple[str | None, str]:
    r"""Parse one official ``n_klm`` partner-operation code."""

    if symmetry is None:
        return None, "555"
    operator_id, separator, translation_code = symmetry.partition("_")
    if (
        not operator_id.isascii()
        or not operator_id.isdecimal()
        or operator_id != str(int(operator_id))
        or not 1 <= int(operator_id) <= 192
        or (separator and (len(translation_code) != 3 or any(value not in "123456789" for value in translation_code)))
    ):
        raise ValueError(f"invalid mmCIF crystallographic symmetry {symmetry!r}")
    if not separator:
        translation_code = "555"
    return operator_id, translation_code


def _mmcif_crystallographic_transform(
    block: CIFBlock,
    symmetry: str | None,
) -> tuple[str | None, np.ndarray]:
    r"""Resolve one ``n_klm`` partner operation to a Cartesian affine transform."""

    operator_id, translation_code = _mmcif_crystallographic_code(symmetry)
    if operator_id is None:
        return None, np.eye(4, dtype=np.float64)
    translation = np.asarray([int(value) - 5 for value in translation_code], dtype=np.float64)
    operators = _mmcif_symmetry_operators(block)
    try:
        fractional_operation = operators[operator_id].copy()
    except KeyError:
        raise ValueError(f"mmCIF crystallographic symmetry references unknown operation {operator_id!r}") from None
    fractional_operation[:3, 3] += translation
    if np.allclose(fractional_operation, np.eye(4), rtol=0.0, atol=1e-12):
        return None, np.eye(4, dtype=np.float64)
    fractionalization = _mmcif_fractionalization(block)
    try:
        transform = np.linalg.inv(fractionalization) @ fractional_operation @ fractionalization
    except np.linalg.LinAlgError:
        raise ValueError("mmCIF _atom_sites fractional transform is singular") from None
    if not np.isfinite(transform).all():
        raise ValueError(f"mmCIF crystallographic symmetry {symmetry!r} produces a non-finite transform")
    return f"{operator_id}_{translation_code}", transform


def _mmcif_crystallographic_pair(
    block: CIFBlock,
    symmetry_a: str | None,
    symmetry_b: str | None,
) -> tuple[tuple[str | None, np.ndarray], tuple[str | None, np.ndarray]]:
    r"""Resolve the relative crystallographic placement of two connection endpoints."""

    code_a = _mmcif_crystallographic_code(symmetry_a)
    code_b = _mmcif_crystallographic_code(symmetry_b)
    if code_a == code_b:
        operator_id = code_a[0]
        if (
            operator_id is not None
            and ("space_group_symop" in block or "symmetry_equiv" in block)
            and operator_id not in _mmcif_symmetry_operators(block)
        ):
            raise ValueError(f"mmCIF crystallographic symmetry references unknown operation {operator_id!r}")
        identity = np.eye(4, dtype=np.float64)
        return (None, identity), (None, identity.copy())
    return (
        _mmcif_crystallographic_transform(block, symmetry_a),
        _mmcif_crystallographic_transform(block, symmetry_b),
    )


def _cif_row_value(category: CIFCategory, name: str, index: int) -> str | None:
    return None if name not in category else present_string(category[name].as_array()[index])


def _mmcif_connection_endpoint(
    category: CIFCategory,
    index: int,
    endpoint: int,
) -> tuple[tuple[object, ...], tuple[object, ...], str | None, str | None]:
    label = (
        _cif_row_value(category, f"ptnr{endpoint}_label_asym_id", index),
        optional_integer(_cif_row_value(category, f"ptnr{endpoint}_label_seq_id", index)),
        _cif_row_value(category, f"pdbx_ptnr{endpoint}_PDB_ins_code", index),
        _cif_row_value(category, f"ptnr{endpoint}_label_comp_id", index),
        _cif_row_value(category, f"ptnr{endpoint}_label_atom_id", index),
    )
    auth = (
        _cif_row_value(category, f"ptnr{endpoint}_auth_asym_id", index) or label[0],
        _cif_row_value(category, f"ptnr{endpoint}_auth_seq_id", index) or label[1],
        label[2],
        _cif_row_value(category, f"ptnr{endpoint}_auth_comp_id", index) or label[3],
        _cif_row_value(category, f"ptnr{endpoint}_auth_atom_id", index) or label[4],
    )
    altlocs = {
        value
        for name in (
            f"pdbx_ptnr{endpoint}_label_alt_id",
            f"ptnr{endpoint}_label_alt_id",
            f"pdbx_ptnr{endpoint}_auth_alt_id",
        )
        if (value := _cif_row_value(category, name, index)) is not None
    }
    if len(altlocs) > 1:
        raise ValueError(f"mmCIF struct_conn row {index} has conflicting alternate-location identities")
    return (
        label,
        auth,
        next(iter(altlocs)) if altlocs else None,
        _cif_row_value(category, f"ptnr{endpoint}_symmetry", index),
    )


type _MmcifEndpoint = tuple[tuple[object, ...], tuple[object, ...], str | None, str | None]
type _MmcifEndpointIndex = tuple[
    dict[tuple[object, ...], list[int]],
    dict[tuple[object, ...], list[int]],
    tuple[str | None, ...],
]
type _MmcifEntityDeclarations = tuple[
    dict[str, str],
    dict[str, tuple[tuple[int, tuple[str, ...]], ...]] | None,
    dict[str, str],
    dict[tuple[str, int] | tuple[str, int, str], tuple[str | None, str | None]],
    dict[str, str],
    dict[str, str],
    dict[str, tuple[tuple[str, str, str, str | None, str | None, str | None], ...]],
]


def _mmcif_endpoint_index(rows: np.recarray) -> _MmcifEndpointIndex:
    label_rows: dict[tuple[object, ...], list[int]] = {}
    auth_rows: dict[tuple[object, ...], list[int]] = {}
    altlocs: list[str | None] = []
    chain_ids = rows["chain_id"]
    label_seq_ids = rows["label_seq_id"]
    insertion_codes = rows["insertion_code"]
    component_ids = rows["component_id"]
    atom_names = rows["atom_name"]
    auth_chain_ids = rows["auth_chain_id"]
    auth_residue_ids = rows["auth_residue_id"]
    auth_component_ids = rows["auth_component_id"]
    auth_atom_names = rows["auth_atom_name"]
    altloc_ids = rows["altloc_id"]
    for index in range(len(rows)):
        label_identity = (
            chain_ids[index],
            label_seq_ids[index],
            insertion_codes[index],
            component_ids[index],
            atom_names[index],
        )
        auth_identity = (
            auth_chain_ids[index],
            auth_residue_ids[index],
            insertion_codes[index],
            auth_component_ids[index],
            auth_atom_names[index],
        )
        label_rows.setdefault(label_identity, []).append(index)
        auth_rows.setdefault(auth_identity, []).append(index)
        altlocs.append(altloc_ids[index])
    return label_rows, auth_rows, tuple(altlocs)


def _mmcif_endpoint_rows(
    endpoint_index: _MmcifEndpointIndex,
    endpoint: _MmcifEndpoint,
    *,
    match_altloc: bool = True,
    included_rows: Collection[int] | None = None,
) -> tuple[int, ...]:
    r"""Return source-row positions matching one ``struct_conn`` endpoint."""

    label, auth, required_altloc, _ = endpoint
    label_rows, auth_rows, altlocs = endpoint_index
    identities = [
        (label, label_rows),
        (auth, auth_rows),
    ]
    identities = [
        (source, observed)
        for source, observed in identities
        if source[0] is not None and source[3] is not None and source[4] is not None
    ]
    specificity = max((sum(value is not None for value in source) for source, _ in identities), default=0)
    matches = {
        row
        for source, observed in identities
        if sum(value is not None for value in source) == specificity
        for row in observed.get(source, ())
        if (included_rows is None or row in included_rows)
        and (not match_altloc or required_altloc is None or altlocs[row] == required_altloc)
    }
    return tuple(sorted(matches))


def _mmcif_endpoint_row(
    endpoint_index: _MmcifEndpointIndex,
    endpoint: _MmcifEndpoint,
) -> int | None:
    candidates = _mmcif_endpoint_rows(endpoint_index, endpoint)
    if len(candidates) > 1:
        raise ValueError("mmCIF struct_conn endpoint is ambiguous in the selected atom rows")
    return candidates[0] if candidates else None


def _materialize_mmcif_crystallographic_copies(
    cif_file: object,
    rows: np.recarray,
    selected_rows: np.recarray,
    placements: Mapping[tuple[str | None, str], np.ndarray],
    *,
    include_water: bool,
) -> tuple[
    np.recarray,
    dict[tuple[str | None, str], np.ndarray],
    set[tuple[str, str, str | None]],
]:
    r"""Materialize the molecular units required by ASU crystallographic connections."""

    if any(symmetry_id is not None for symmetry_id, _ in placements):
        return rows, dict(placements), set()
    block = cast("CIFFile", cif_file).block
    if "struct_conn" not in block:
        return rows, dict(placements), set()
    category = block["struct_conn"]
    relevant_connections = tuple(
        index
        for index, connection_type in enumerate(_cif_column(category, "conn_type_id"))
        if _is_graph_connection_type(connection_type)
    )
    if all(
        all(
            operation[0] is None
            for operation in _mmcif_crystallographic_pair(
                block,
                _cif_row_value(category, "ptnr1_symmetry", index),
                _cif_row_value(category, "ptnr2_symmetry", index),
            )
        )
        for index in relevant_connections
    ):
        return rows, dict(placements), set()
    entity_types, _, _, _, asym_entities, _, nonpoly_occurrences = _read_mmcif_entity_declarations(cif_file)
    branched_entities = (
        {entity_id for entity_id in _cif_column(block["pdbx_entity_branch_list"], "entity_id") if entity_id is not None}
        if "pdbx_entity_branch_list" in block
        else set()
    )

    def unit_mask(
        endpoint: tuple[tuple[object, ...], tuple[object, ...], str | None, str | None],
        endpoint_row: int | None,
    ) -> np.ndarray:
        label, auth, _, _ = endpoint
        chain_unit = False
        if endpoint_row is None:
            label_occurrence = label[:-1]
            auth_occurrence = auth[:-1]
            occurrence = np.asarray(
                [
                    (
                        label_occurrence[0] is not None
                        and label_occurrence[3] is not None
                        and (
                            present_string(row.chain_id),
                            optional_integer(row.label_seq_id),
                            present_string(row.insertion_code),
                            present_string(row.component_id),
                        )
                        == label_occurrence
                    )
                    or (
                        auth_occurrence[0] is not None
                        and auth_occurrence[3] is not None
                        and (
                            present_string(row.auth_chain_id),
                            present_string(row.auth_residue_id),
                            present_string(row.insertion_code),
                            present_string(row.auth_component_id),
                        )
                        == auth_occurrence
                    )
                    for row in selected_rows
                ],
                dtype=np.bool_,
            )
            occurrence_indices = np.flatnonzero(occurrence)
            if not len(occurrence_indices):
                asym_id = label[0]
                entity_id = None if asym_id is None else asym_entities.get(str(asym_id))
                if (
                    entity_types.get(entity_id or "") not in {"polymer", "branched"}
                    and entity_id not in branched_entities
                ):
                    return occurrence
                chain_unit = True
                occurrence = np.asarray(
                    [asym_id is not None and present_string(row.chain_id) == asym_id for row in selected_rows],
                    dtype=np.bool_,
                )
                occurrence_indices = np.flatnonzero(occurrence)
            if not len(occurrence_indices):
                return occurrence
            endpoint_row = int(occurrence_indices[0])
        else:
            occurrence = np.zeros(len(selected_rows), dtype=np.bool_)
            occurrence[endpoint_row] = True

        if not chain_unit and int(occurrence.sum()) > 1:
            occurrence_identities = {
                (
                    selected_rows[index].chain_id,
                    selected_rows[index].label_seq_id,
                    selected_rows[index].auth_chain_id,
                    selected_rows[index].auth_residue_id,
                    selected_rows[index].insertion_code,
                    selected_rows[index].component_id,
                )
                for index in np.flatnonzero(occurrence)
            }
            if len(occurrence_identities) > 1:
                raise ValueError("mmCIF struct_conn endpoint occurrence is ambiguous")
        row = selected_rows[endpoint_row]
        entity_type = entity_types.get(str(row.entity_id))
        if entity_type in {"polymer", "branched"} or str(row.entity_id) in branched_entities:
            return np.asarray([other.chain_id == row.chain_id for other in selected_rows], dtype=np.bool_)
        residue_identity = (
            row.chain_id,
            row.label_seq_id,
            row.auth_chain_id,
            row.auth_residue_id,
            row.insertion_code,
            row.component_id,
        )
        return np.asarray(
            [
                (
                    other.chain_id,
                    other.label_seq_id,
                    other.auth_chain_id,
                    other.auth_residue_id,
                    other.insertion_code,
                    other.component_id,
                )
                == residue_identity
                for other in selected_rows
            ],
            dtype=np.bool_,
        )

    Unit = tuple[np.ndarray, tuple[str, str | None, str] | None]
    seeds: dict[str, tuple[np.ndarray, np.ndarray, set[tuple[str, str | None, str]]]] = {}
    identity_connections: list[tuple[Unit, Unit]] = []
    materialized_placements = dict(placements)
    topology_placements: set[tuple[str, str, str | None]] = set()
    source_endpoint_index = _mmcif_endpoint_index(rows)
    selected_endpoint_index = _mmcif_endpoint_index(selected_rows)

    def declared_nonpoly_occurrence(
        endpoint: tuple[tuple[object, ...], tuple[object, ...], str | None, str | None],
    ) -> tuple[str, str, str] | None:
        label, auth, _, _ = endpoint
        asym_id = label[0]
        if asym_id is None:
            return None
        source_entity_id = asym_entities.get(str(asym_id))
        component_ids = {str(value) for value in (label[3], auth[3]) if value is not None}
        author_number = None if auth[1] is None else str(auth[1])
        insertion = None if label[2] is None else str(label[2])
        author_chain = None if auth[0] is None else str(auth[0])
        candidates = [
            (ndb_code, occurrence_entity)
            for ndb_code, occurrence_entity, monomer, occurrence_number, occurrence_insertion, pdb_chain in (
                nonpoly_occurrences.get(str(asym_id), ())
            )
            if occurrence_entity == source_entity_id
            and monomer in component_ids
            and (occurrence_number is None or author_number is None or occurrence_number == author_number)
            and (occurrence_insertion is None or insertion is None or occurrence_insertion == insertion)
            and (pdb_chain is None or author_chain is None or pdb_chain == author_chain)
        ]
        if len(candidates) > 1:
            raise ValueError("mmCIF struct_conn endpoint occurrence is ambiguous")
        if not candidates:
            return None
        ndb_code, occurrence_entity = candidates[0]
        return str(asym_id), ndb_code, occurrence_entity

    def declared_unit(
        endpoint: tuple[tuple[object, ...], tuple[object, ...], str | None, str | None],
    ) -> tuple[str, str | None, str] | None:
        asym_id = endpoint[0][0]
        if asym_id is None:
            return None
        source_entity_id = asym_entities.get(str(asym_id))
        if source_entity_id is None:
            return None
        if entity_types.get(source_entity_id) in {"polymer", "branched"} or source_entity_id in branched_entities:
            return str(asym_id), None, source_entity_id
        occurrence = declared_nonpoly_occurrence(endpoint)
        if occurrence is None:
            return None
        occurrence_asym, ndb_code, occurrence_entity = occurrence
        return occurrence_asym, ndb_code, occurrence_entity

    def add_placement(symmetry_id: str, asym_id: str, transform: np.ndarray) -> None:
        previous = materialized_placements.setdefault((symmetry_id, asym_id), transform)
        if not np.allclose(previous, transform, rtol=0.0, atol=1e-8):
            raise ValueError(f"mmCIF crystallographic placement {(symmetry_id, asym_id)!r} has conflicting transforms")

    for index, connection_type in enumerate(_cif_column(category, "conn_type_id")):
        if not _is_graph_connection_type(connection_type):
            continue
        endpoints = (
            _mmcif_connection_endpoint(category, index, 1),
            _mmcif_connection_endpoint(category, index, 2),
        )
        endpoint_row_a = _mmcif_endpoint_row(selected_endpoint_index, endpoints[0])
        endpoint_row_b = _mmcif_endpoint_row(selected_endpoint_index, endpoints[1])
        endpoint_rows = (endpoint_row_a, endpoint_row_b)
        # An atom present in the source but absent here was removed by the requested
        # altloc/water projection. A genuinely unmodeled atom may instead use
        # the rest of its component occurrence to identify the molecular copy.
        if any(
            endpoint_row is None and bool(_mmcif_endpoint_rows(source_endpoint_index, endpoint, match_altloc=False))
            for endpoint, endpoint_row in zip(endpoints, endpoint_rows, strict=True)
        ):
            continue
        masks = (
            unit_mask(endpoints[0], endpoint_rows[0]),
            unit_mask(endpoints[1], endpoint_rows[1]),
        )
        operations = _mmcif_crystallographic_pair(block, endpoints[0][3], endpoints[1][3])
        declarations = tuple(
            None if bool(mask.any()) else declared_unit(endpoint)
            for endpoint, mask in zip(endpoints, masks, strict=True)
        )
        if any(
            declaration is None for declaration, mask in zip(declarations, masks, strict=True) if not bool(mask.any())
        ):
            continue
        if any(
            not include_water and entity_types.get(declaration[2]) == "water"
            for declaration in declarations
            if declaration is not None
        ):
            continue
        units: tuple[Unit, Unit] = (
            (masks[0], declarations[0]),
            (masks[1], declarations[1]),
        )
        if operations[0][0] is None and operations[1][0] is None:
            identity_connections.append(units)
            continue
        for (symmetry_id, transform), (mask, declaration) in zip(operations, units, strict=True):
            if symmetry_id is None:
                continue
            previous = seeds.setdefault(symmetry_id, (transform, mask.copy(), set()))
            if not np.allclose(previous[0], transform, rtol=0.0, atol=1e-8):
                raise ValueError(f"mmCIF crystallographic symmetry {symmetry_id!r} has conflicting transforms")
            np.logical_or(previous[1], mask, out=previous[1])
            if declaration is not None:
                previous[2].add(declaration)

    copies: list[np.recarray] = [rows]
    for symmetry_id, (transform, seed, seed_declarations) in seeds.items():
        selected = seed.copy()
        selected_declarations = set(seed_declarations)
        changed = True
        while changed:
            changed = False
            for unit_a, unit_b in identity_connections:
                rows_a, declaration_a = unit_a
                rows_b, declaration_b = unit_b
                connected = bool((selected & (rows_a | rows_b)).any()) or any(
                    declaration is not None and declaration in selected_declarations
                    for declaration in (declaration_a, declaration_b)
                )
                if not connected:
                    continue
                expanded = selected | rows_a | rows_b
                if not np.array_equal(expanded, selected):
                    selected = expanded
                    changed = True
                for declaration in (declaration_a, declaration_b):
                    if declaration is not None and declaration not in selected_declarations:
                        selected_declarations.add(declaration)
                        changed = True
        copied = selected_rows[selected].copy().view(np.recarray)
        present = np.asarray(copied.coordinate_present, dtype=np.bool_)
        copied.coordinates[present] = copied.coordinates[present] @ transform[:3, :3].T + transform[:3, 3]
        copied.symmetry_id[:] = symmetry_id
        copies.append(copied)
        for asym_id in sorted({str(row.chain_id) for row in copied}):
            add_placement(symmetry_id, asym_id, transform)
        for asym_id, ndb_code, _ in sorted(
            selected_declarations,
            key=lambda declaration: (
                declaration[0],
                "" if declaration[1] is None else declaration[1],
                declaration[2],
            ),
        ):
            add_placement(symmetry_id, asym_id, transform)
            topology_placements.add((symmetry_id, asym_id, ndb_code))

    materialized = np.concatenate(copies).view(np.recarray)
    materialized.row_index[:] = np.arange(len(materialized))
    reject_duplicate_atom_rows(materialized)
    return materialized, materialized_placements, topology_placements


def read_mmcif_components(
    cif_file: object,
    *,
    owned: Collection[str] = (),
) -> tuple[
    dict[str, ChemicalComponent],
    set[str],
    dict[str, str],
    dict[str, tuple[frozenset[str], frozenset[frozenset[str]]]],
]:
    r"""
    Read embedded component definitions and every declared component ID.

    ``chem_comp_atom`` and ``chem_comp_bond`` carry the component graph, ``_chem_comp.type``
    its polymer chemistry, and ``pdbx_leaving_atom_flag`` the atoms a realized polymer link
    removes; together the last two give the component its polymer attachments, exactly as
    [`ofoldx.data.ccd.build`][] reads the same items out of the wwPDB dictionary. Reference
    coordinates come from the idealized conformer where the source declares one and from the
    representative model coordinates otherwise.

    An embedded definition explicitly supplied through ``owned`` is still declared but
    never built. All other source chemistry is retained; a later dictionary completion may
    fill facts it leaves unstated but cannot replace facts it states.

    Args:
        cif_file: Parsed mmCIF document.
        owned: Component IDs whose chemistry the caller supplies itself.

    Returns:
        Embedded component definitions by component ID, every declared component ID,
        nonstandard-to-parent component IDs from ``_chem_comp``, and explicit atom/bond
        stereo fields (including explicit ``N`` values) by component ID.

    Raises:
        ValueError: If a ``chem_comp_atom`` or ``chem_comp_bond`` row omits an identifier,
            an atom states an invalid leaving flag, or a bond states an unsupported order
            or an aromaticity disagreeing with it.
    """

    block = cast("CIFFile", cif_file).block
    declared_components: set[str] = set()
    component_parents: dict[str, str] = {}
    if "chem_comp" in block:
        category = block["chem_comp"]
        for component_id, parent_id in zip(
            _cif_column(category, "id"),
            _cif_column(category, "mon_nstd_parent_comp_id"),
        ):
            if component_id is None or parent_id is None:
                continue
            parents = tuple(value.strip() for value in parent_id.split(",") if value.strip())
            if len(parents) != 1:
                raise ValueError(
                    f"mmCIF component {component_id!r} has multiple standard parents, which Structure cannot represent"
                )
            component_parents[component_id] = parents[0]
    for category_name, column_name in (
        ("entity_poly_seq", "mon_id"),
        ("pdbx_entity_branch_list", "comp_id"),
        ("pdbx_entity_nonpoly", "comp_id"),
    ):
        if category_name in block:
            declared_components.update(
                value for value in _cif_column(block[category_name], column_name) if value is not None
            )
    if "chem_comp_atom" not in block:
        return {}, declared_components, component_parents, {}
    component_types = (
        dict(
            zip(
                _cif_column(block["chem_comp"], "id"),
                _cif_column(block["chem_comp"], "type"),
            )
        )
        if "chem_comp" in block
        else {}
    )
    atom_category = block["chem_comp_atom"]
    component_ids = _cif_column(atom_category, "comp_id")
    declared_components.update(component_id for component_id in component_ids if component_id is not None)
    ignored_components = set(owned)
    atom_ids = _cif_column(atom_category, "atom_id")
    elements = _cif_column(atom_category, "type_symbol")
    charges = _cif_column(atom_category, "charge")
    aromatic_flags = _cif_column(atom_category, "pdbx_aromatic_flag")
    stereo_values = _cif_column(atom_category, "pdbx_stereo_config")
    leaving_flags = _cif_column(atom_category, "pdbx_leaving_atom_flag")

    atoms: dict[str, list[ChemicalAtom]] = {}
    leaving_atoms: dict[str, set[str]] = {}
    non_leaving_atoms: dict[str, set[str]] = {}
    model_coordinates: dict[str, list[tuple[float, float, float] | None]] = {}
    ideal_coordinates: dict[str, list[tuple[float, float, float] | None]] = {}
    atom_stereo_known: dict[str, set[str]] = {}
    for values in zip(
        component_ids,
        atom_ids,
        elements,
        charges,
        aromatic_flags,
        stereo_values,
        leaving_flags,
        _cif_points(atom_category, "model_Cartn_x", "model_Cartn_y", "model_Cartn_z"),
        _cif_points(
            atom_category,
            "pdbx_model_Cartn_x_ideal",
            "pdbx_model_Cartn_y_ideal",
            "pdbx_model_Cartn_z_ideal",
        ),
    ):
        component_id, atom_id, element, charge, aromatic, stereo, leaving, model_point, ideal_point = values
        if component_id is None:
            raise ValueError("mmCIF chem_comp_atom row is missing comp_id, atom_id, or type_symbol")
        if component_id in ignored_components:
            continue
        if atom_id is None or element is None:
            raise ValueError("mmCIF chem_comp_atom row is missing comp_id, atom_id, or type_symbol")
        atoms.setdefault(component_id, []).append(
            ChemicalAtom(
                atom_id=atom_id,
                element=element.capitalize(),
                formal_charge=optional_integer(charge),
                aromatic=None if aromatic is None else aromatic.upper() == "Y",
                stereo=None if stereo in {None, "N"} else stereo,
            )
        )
        if stereo is not None:
            atom_stereo_known.setdefault(component_id, set()).add(atom_id)
        if leaving is not None:
            normalized_leaving = leaving.upper()
            if normalized_leaving not in {"Y", "N"}:
                raise ValueError("mmCIF chem_comp_atom pdbx_leaving_atom_flag must be Y or N")
            if normalized_leaving == "Y":
                leaving_atoms.setdefault(component_id, set()).add(atom_id)
            else:
                non_leaving_atoms.setdefault(component_id, set()).add(atom_id)
        model_coordinates.setdefault(component_id, []).append(model_point)
        ideal_coordinates.setdefault(component_id, []).append(ideal_point)

    bonds_by_component: dict[str, set[ChemicalBond]] = {component_id: set() for component_id in atoms}
    components_with_bonds: set[str] = set()
    bond_stereo_known: dict[str, set[frozenset[str]]] = {}
    if "chem_comp_bond" in block:
        bond_category = block["chem_comp_bond"]
        order_names = (
            _cif_column(bond_category, "value_order")
            if "value_order" in bond_category
            else ("sing",) * bond_category.row_count
        )
        for component_id, atom_a, atom_b, order_name, aromatic, stereo in zip(
            _cif_column(bond_category, "comp_id"),
            _cif_column(bond_category, "atom_id_1"),
            _cif_column(bond_category, "atom_id_2"),
            order_names,
            _cif_column(bond_category, "pdbx_aromatic_flag"),
            _cif_column(bond_category, "pdbx_stereo_config"),
        ):
            if component_id is None:
                raise ValueError("mmCIF chem_comp_bond row is missing comp_id or atom IDs")
            if component_id in ignored_components:
                continue
            if atom_a is None or atom_b is None:
                raise ValueError("mmCIF chem_comp_bond row is missing comp_id or atom IDs")
            components_with_bonds.add(component_id)
            order, order_aromatic = _cif_bond_order(order_name)
            if aromatic is None:
                aromatic_value = order_aromatic
            else:
                if aromatic.upper() not in {"Y", "N"}:
                    raise ValueError("mmCIF chem_comp_bond pdbx_aromatic_flag must be Y or N")
                aromatic_value = aromatic.upper() == "Y"
                if order_aromatic is not None and aromatic_value != order_aromatic:
                    raise ValueError("mmCIF chem_comp_bond value_order and pdbx_aromatic_flag disagree")
            bonds_by_component.setdefault(component_id, set()).add(
                ChemicalBond(
                    atom_a,
                    atom_b,
                    order=order,
                    aromatic=aromatic_value,
                    stereo=None if stereo in {None, "N"} else stereo,
                )
            )
            if stereo is not None:
                bond_stereo_known.setdefault(component_id, set()).add(frozenset((atom_a, atom_b)))
    components: dict[str, ChemicalComponent] = {}
    for component_id, component_atoms in atoms.items():
        component_leaving_atoms = leaving_atoms.get(component_id, set())
        component_non_leaving_atoms = non_leaving_atoms.get(component_id, set())
        component_atom_ids = {atom.atom_id for atom in component_atoms}
        leaving_flags_complete = component_leaving_atoms | component_non_leaving_atoms == component_atom_ids
        component_bonds = (
            frozenset(bonds_by_component[component_id])
            if len(component_atoms) == 1 or component_id in components_with_bonds
            else None
        )
        previous_attachment, next_attachment = _polymer_attachments(
            component_types.get(component_id),
            component_atoms,
            component_leaving_atoms if leaving_flags_complete else None,
            component_bonds,
        )
        ideal_points = ideal_coordinates[component_id]
        components[component_id] = ChemicalComponent(
            component_id=component_id,
            atoms=tuple(component_atoms),
            bonds=component_bonds,
            previous_attachment=previous_attachment,
            next_attachment=next_attachment,
            # Each column triple is one whole conformer, so a component takes every
            # reference coordinate from the same one: the idealized conformer this project
            # reads as the reference geometry, or the representative model coordinates for a
            # component that declares no idealized one.
            reference_coordinates=tuple(
                ideal_points if any(point is not None for point in ideal_points) else model_coordinates[component_id]
            ),
            leaving_atom_ids=frozenset(component_leaving_atoms),
            non_leaving_atom_ids=frozenset(component_non_leaving_atoms),
        )
    stereo_known = {
        component_id: (
            frozenset(atom_stereo_known.get(component_id, ())),
            frozenset(bond_stereo_known.get(component_id, ())),
        )
        for component_id in components
    }
    return components, declared_components, component_parents, stereo_known


def _polymer_attachments(
    chem_comp_type: str | None,
    atoms: Sequence[ChemicalAtom],
    leaving_atom_ids: set[str] | None,
    bonds: frozenset[ChemicalBond] | None,
) -> tuple[PolymerAttachment | None, PolymerAttachment | None]:
    r"""
    Derive the linear-polymer attachments an embedded component declares.

    ``_chem_comp.type`` names the polymer chemistry while ``pdbx_leaving_atom_flag``
    names the atoms a realized link removes. Each connected leaving group belongs to
    its unique non-leaving graph boundary. This matters for beta- and gamma-peptides,
    whose outgoing carbonyl is not the conventional atom ``C``.
    """

    if leaving_atom_ids is None or bonds is None:
        return None, None
    previous_anchor, next_anchor = polymer_attachment_anchors(
        chem_comp_type,
        tuple(atom.atom_id for atom in atoms),
    )
    if previous_anchor is None and next_anchor is None:
        return None, None

    atom_ids = tuple(atom.atom_id for atom in atoms)
    present = set(atom_ids)
    bond_pairs = tuple((bond.atom_a, bond.atom_b) for bond in bonds)

    def attachment(anchor_id: str | None) -> PolymerAttachment | None:
        if anchor_id is None or anchor_id not in present:
            return None
        groups = terminal_leaving_groups(atom_ids, bond_pairs, leaving_atom_ids, anchor_id)
        if len(groups) > 1:
            return None
        return PolymerAttachment(anchor_id, groups[0] if groups else frozenset())

    return attachment(previous_anchor), attachment(next_anchor)


def _read_mmcif_entity_declarations(
    cif_file: object,
) -> _MmcifEntityDeclarations:
    r"""
    Decode mmCIF entity-type, polymer-sequence, and author-numbering declarations.

    The ``_entity.type`` declaration is the authoritative entity category, and
    ``_entity_poly_seq`` is the ordered canonical monomer sequence per polymer entity,
    including residues unmodeled in the coordinates.

    Args:
        cif_file: Parsed mmCIF document.

    Returns:
        Entity declarations gathered from the source. Absent categories yield empty
        mappings. Non-polymer occurrence tuples contain NDB identity, entity, monomer,
        author number, insertion code, and author strand in declaration order.
    """

    block = cast("CIFFile", cif_file).block

    entity_types: dict[str, str] = {}
    has_entity_declarations = "entity" in block
    if has_entity_declarations:
        category = block["entity"]
        for entity_id, entity_type in zip(_cif_column(category, "id"), _cif_column(category, "type")):
            if entity_id is None or entity_type is None:
                raise ValueError("mmCIF _entity row is missing id or type")
            if entity_id in entity_types:
                raise ValueError(f"mmCIF _entity contains duplicate id {entity_id!r}")
            normalized_type = entity_type.lower()
            if normalized_type not in {"branched", "macrolide", "non-polymer", "polymer", "water"}:
                raise ValueError(f"unsupported mmCIF entity type {entity_type!r}")
            entity_types[entity_id] = normalized_type

    polymer_monomers: dict[str, dict[int, list[str]]] = {}
    polymer_monomer_hetero: dict[tuple[str, int, str], list[str | None]] = {}
    has_polymer_sequences = "entity_poly_seq" in block
    if has_polymer_sequences:
        category = block["entity_poly_seq"]
        for entity_id, number, monomer, hetero in zip(
            _cif_column(category, "entity_id"),
            _cif_column(category, "num"),
            _cif_column(category, "mon_id"),
            _cif_column(category, "hetero"),
        ):
            index = optional_integer(number)
            if entity_id is None or index is None or monomer is None:
                raise ValueError("mmCIF _entity_poly_seq row is missing entity_id, num, or mon_id")
            if index < 1:
                raise ValueError(f"mmCIF _entity_poly_seq.num must be positive, got {index}")
            candidates = polymer_monomers.setdefault(entity_id, {}).setdefault(index, [])
            if monomer not in candidates:
                candidates.append(monomer)
            polymer_monomer_hetero.setdefault((entity_id, index, monomer), []).append(
                None if hetero is None else hetero.casefold()
            )
        for declared_entity_id, positions in polymer_monomers.items():
            numbers = sorted(positions)
            if numbers != list(range(1, len(numbers) + 1)):
                raise ValueError(f"mmCIF polymer entity {declared_entity_id!r} has a non-contiguous declared sequence")
            for position_number, candidates in positions.items():
                if len(candidates) > 1 and any(
                    flag not in {"y", "yes"}
                    for monomer in candidates
                    for flag in polymer_monomer_hetero[(declared_entity_id, position_number, monomer)]
                ):
                    raise ValueError(
                        f"mmCIF polymer entity {declared_entity_id!r} position {position_number} has conflicting "
                        f"non-heterogeneous monomer declarations {candidates!r}"
                    )

    entity_poly_types: dict[str, str] = {}
    if "entity_poly" in block:
        category = block["entity_poly"]
        for entity_id, poly_type in zip(_cif_column(category, "entity_id"), _cif_column(category, "type")):
            if entity_id is None or poly_type is None:
                raise ValueError("mmCIF _entity_poly row is missing entity_id or type")
            if entity_id in entity_poly_types:
                raise ValueError(f"mmCIF _entity_poly contains duplicate entity_id {entity_id!r}")
            entity_poly_types[entity_id] = poly_type.lower()

    author_numbering: dict[
        tuple[str, int] | tuple[str, int, str],
        tuple[str | None, str | None],
    ] = {}
    if "pdbx_poly_seq_scheme" in block:
        category = block["pdbx_poly_seq_scheme"]
        polymer_numbering: dict[tuple[str, int, str | None], tuple[str | None, str | None]] = {}
        for asym_id, entity_id, seq_id, monomer, auth_number, pdb_number, ins_code in zip(
            _cif_column(category, "asym_id"),
            _cif_column(category, "entity_id"),
            _cif_column(category, "seq_id"),
            _cif_column(category, "mon_id"),
            _cif_column(category, "auth_seq_num"),
            _cif_column(category, "pdb_seq_num"),
            _cif_column(category, "pdb_ins_code"),
        ):
            seq = optional_integer(seq_id)
            if asym_id is None or seq is None:
                raise ValueError("mmCIF _pdbx_poly_seq_scheme row is missing asym_id or seq_id")
            # ``pdb_seq_num`` is the namespace ``_atom_site.auth_seq_id`` numbers the observed
            # rows in, so it is the one that identifies a residue. ``auth_seq_num`` names the
            # nearest observed residue for an unmodeled position, which repeats its neighbour.
            author_residue_number = present_string(pdb_number)
            if author_residue_number is None:
                author_residue_number = present_string(auth_number)
            if entity_id is not None and monomer is not None:
                declared_candidates = polymer_monomers.get(entity_id, {}).get(seq)
                if declared_candidates is None or monomer not in declared_candidates:
                    raise ValueError(
                        f"mmCIF _pdbx_poly_seq_scheme monomer {monomer!r} is not declared "
                        f"at entity {entity_id!r} position {seq}"
                    )
            scheme_key = (asym_id, seq, monomer)
            numbering = (author_residue_number, ins_code)
            if scheme_key in polymer_numbering and polymer_numbering[scheme_key] != numbering:
                raise ValueError(f"mmCIF _pdbx_poly_seq_scheme has conflicting numbering for {scheme_key!r}")
            polymer_numbering[scheme_key] = numbering

        for (asym_id, seq, monomer), numbering in polymer_numbering.items():
            if monomer is None:
                author_numbering[(asym_id, seq)] = numbering
            else:
                author_numbering[(asym_id, seq, monomer)] = numbering

    if "pdbx_branch_scheme" in block:
        category = block["pdbx_branch_scheme"]
        for asym_id, number, auth_number, pdb_number, ins_code in zip(
            _cif_column(category, "asym_id"),
            _cif_column(category, "num"),
            _cif_column(category, "auth_seq_num"),
            _cif_column(category, "pdb_seq_num"),
            _cif_column(category, "pdb_ins_code"),
        ):
            component_number = optional_integer(number)
            if asym_id is None or component_number is None:
                raise ValueError("mmCIF _pdbx_branch_scheme row is missing asym_id or num")
            author_residue_number = pdb_number or auth_number
            numbering_key = (asym_id, component_number)
            numbering = (author_residue_number, ins_code)
            if numbering_key in author_numbering and author_numbering[numbering_key] != numbering:
                raise ValueError(f"mmCIF _pdbx_branch_scheme has conflicting numbering for {numbering_key!r}")
            author_numbering[numbering_key] = numbering

    nonpoly_occurrences: dict[
        str,
        list[tuple[str, str, str, str | None, str | None, str | None]],
    ] = {}
    if "pdbx_nonpoly_scheme" in block:
        category = block["pdbx_nonpoly_scheme"]
        ordinal_by_asym: dict[str, int] = {}
        occurrence_keys: set[tuple[str, str]] = set()
        for asym_id, entity_id, monomer, ndb_number, auth_number, pdb_number, ins_code, strand_id in zip(
            _cif_column(category, "asym_id"),
            _cif_column(category, "entity_id"),
            _cif_column(category, "mon_id"),
            _cif_column(category, "ndb_seq_num"),
            _cif_column(category, "auth_seq_num"),
            _cif_column(category, "pdb_seq_num"),
            _cif_column(category, "pdb_ins_code"),
            _cif_column(category, "pdb_strand_id"),
        ):
            if asym_id is None or entity_id is None or monomer is None or ndb_number is None:
                raise ValueError("mmCIF _pdbx_nonpoly_scheme row is missing asym_id, entity_id, mon_id, or ndb_seq_num")
            occurrence_key = asym_id, ndb_number
            if occurrence_key in occurrence_keys:
                raise ValueError(f"mmCIF _pdbx_nonpoly_scheme contains duplicate key {occurrence_key!r}")
            occurrence_keys.add(occurrence_key)
            author_residue_number = present_string(pdb_number) or present_string(auth_number)
            # One non-polymer asym holds many occurrences (every water of a chain), each
            # with its own opaque NDB identity and author number. The canonical component
            # axis follows declaration order; the NDB code is not an integer component index.
            ordinal = ordinal_by_asym[asym_id] = ordinal_by_asym.get(asym_id, 0) + 1
            numbering_key = asym_id, ordinal
            numbering = (author_residue_number, ins_code)
            if numbering_key in author_numbering and author_numbering[numbering_key] != numbering:
                raise ValueError(f"mmCIF _pdbx_nonpoly_scheme has conflicting numbering for {numbering_key!r}")
            author_numbering[numbering_key] = numbering
            nonpoly_occurrences.setdefault(asym_id, []).append(
                (ndb_number, entity_id, monomer, author_residue_number, ins_code, strand_id)
            )

    asym_entities: dict[str, str] = {}
    if "struct_asym" in block:
        category = block["struct_asym"]
        for asym_id, entity_id in zip(
            _cif_column(category, "id"),
            _cif_column(category, "entity_id"),
        ):
            if asym_id is None or entity_id is None:
                raise ValueError("mmCIF _struct_asym row is missing id or entity_id")
            previous = asym_entities.get(asym_id)
            if previous is not None and previous != entity_id:
                raise ValueError(f"mmCIF _struct_asym has conflicting entity for {asym_id!r}")
            asym_entities[asym_id] = entity_id

        for category_name in ("pdbx_poly_seq_scheme", "pdbx_branch_scheme"):
            if category_name not in block:
                continue
            scheme = block[category_name]
            for asym_id, entity_id in zip(
                _cif_column(scheme, "asym_id"),
                _cif_column(scheme, "entity_id"),
            ):
                if asym_id is None or asym_id not in asym_entities:
                    raise ValueError(f"mmCIF {category_name} references undeclared asym {asym_id!r}")
                if entity_id is not None and asym_entities[asym_id] != entity_id:
                    raise ValueError(f"mmCIF {category_name} disagrees with _struct_asym for {asym_id!r}")

    nonpoly_components: dict[str, str] = {}
    if "pdbx_entity_nonpoly" in block:
        category = block["pdbx_entity_nonpoly"]
        for entity_id, component_id in zip(
            _cif_column(category, "entity_id"),
            _cif_column(category, "comp_id"),
        ):
            if entity_id is None or component_id is None:
                raise ValueError("mmCIF _pdbx_entity_nonpoly row is missing entity_id or comp_id")
            previous = nonpoly_components.get(entity_id)
            if previous is not None and previous != component_id:
                raise ValueError(f"mmCIF non-polymer entity {entity_id!r} has conflicting components")
            nonpoly_components[entity_id] = component_id

    if has_entity_declarations:
        for asym_id, entity_id in asym_entities.items():
            if entity_id not in entity_types:
                raise ValueError(f"mmCIF _struct_asym {asym_id!r} references undeclared entity {entity_id!r}")

    for asym_id, occurrences in nonpoly_occurrences.items():
        declared_entity = asym_entities.get(asym_id)
        if "struct_asym" in block and declared_entity is None:
            raise ValueError(f"mmCIF non-polymer scheme references undeclared asym {asym_id!r}")
        for _, entity_id, component_id, *_ in occurrences:
            if declared_entity is not None and declared_entity != entity_id:
                raise ValueError(f"mmCIF non-polymer scheme for asym {asym_id!r} disagrees with _struct_asym")
            declared_component = nonpoly_components.get(entity_id)
            if "pdbx_entity_nonpoly" in block and declared_component is None:
                raise ValueError(f"mmCIF non-polymer scheme references undeclared entity {entity_id!r}")
            if declared_component is not None and declared_component != component_id:
                raise ValueError(f"mmCIF non-polymer scheme for entity {entity_id!r} disagrees with its component")

    if has_entity_declarations:
        for entity_id in {*polymer_monomers, *entity_poly_types}:
            if entity_types.get(entity_id) != "polymer":
                raise ValueError(f"mmCIF polymer declaration belongs to non-polymer entity {entity_id!r}")
        molecule_types = {"macrolide", "non-polymer", "water"}
        for entity_id in nonpoly_components:
            if entity_types.get(entity_id) not in molecule_types:
                raise ValueError(f"mmCIF non-polymer declaration belongs to entity {entity_id!r} of another type")
        for occurrences in nonpoly_occurrences.values():
            for _, entity_id, *_ in occurrences:
                if entity_types.get(entity_id) not in molecule_types:
                    raise ValueError(f"mmCIF non-polymer occurrence belongs to entity {entity_id!r} of another type")
        if "pdbx_entity_branch_list" in block:
            for entity_id in _cif_column(block["pdbx_entity_branch_list"], "entity_id"):
                if entity_id is None:
                    raise ValueError("mmCIF branch declaration is missing entity_id")
                entity_type = entity_types.get(entity_id)
                if entity_type != "branched":
                    raise ValueError(
                        f"mmCIF entity {entity_id!r} is declared {entity_type!r} "
                        "but also has a branched-entity definition"
                    )

    if "atom_site" in block:
        atom_site = block["atom_site"]
        if has_entity_declarations and "label_entity_id" in atom_site:
            for entity_id in _cif_column(atom_site, "label_entity_id"):
                if entity_id is not None and entity_id not in entity_types:
                    raise ValueError(f"mmCIF atom_site references undeclared entity {entity_id!r}")
        if "struct_asym" in block and "label_asym_id" in atom_site:
            for asym_id in _cif_column(atom_site, "label_asym_id"):
                if asym_id is not None and asym_id not in asym_entities:
                    raise ValueError(f"mmCIF atom_site references undeclared asym {asym_id!r}")

    return (
        entity_types,
        (
            {
                entity_id: tuple((number, tuple(positions[number])) for number in sorted(positions))
                for entity_id, positions in polymer_monomers.items()
            }
            if has_polymer_sequences
            else None
        ),
        entity_poly_types,
        author_numbering,
        asym_entities,
        nonpoly_components,
        {asym_id: tuple(occurrences) for asym_id, occurrences in nonpoly_occurrences.items()},
    )


def _cif_bond_order(value: str | None) -> tuple[float | None, bool | None]:
    if value is None:
        return None, None
    try:
        return _CIF_BOND_ORDERS[value.lower()]
    except KeyError:
        raise ValueError(f"unsupported mmCIF covalent bond order: {value!r}") from None


def _remove_component_atoms(component: ChemicalComponent, atom_ids: set[str]) -> ChemicalComponent:
    atom_by_id = {atom.atom_id: atom for atom in component.atoms}
    removed = set(atom_by_id).intersection(atom_ids)
    if not removed:
        return component
    bonds = component.bonds
    if bonds is not None:
        for bond in bonds:
            if bond.atom_a in removed and atom_by_id[bond.atom_b].element in HYDROGEN_ELEMENTS:
                removed.add(bond.atom_b)
            if bond.atom_b in removed and atom_by_id[bond.atom_a].element in HYDROGEN_ELEMENTS:
                removed.add(bond.atom_a)
        for bond in bonds:
            if (
                bond.stereo_atoms is not None
                and removed.intersection(bond.stereo_atoms)
                and not removed.intersection((bond.atom_a, bond.atom_b))
            ):
                raise ValueError("a declared leaving atom is required to define retained bond stereochemistry")
    keep = tuple(atom.atom_id not in removed for atom in component.atoms)
    retained = set(atom_by_id) - removed
    previous = component.previous_attachment
    if previous is not None:
        previous = (
            PolymerAttachment(
                previous.atom_id,
                previous.leaving_atom_ids & retained,
            )
            if previous.atom_id in retained
            else None
        )
    following = component.next_attachment
    if following is not None:
        following = (
            PolymerAttachment(
                following.atom_id,
                following.leaving_atom_ids & retained,
            )
            if following.atom_id in retained
            else None
        )
    return ChemicalComponent(
        component_id=component.component_id,
        atoms=tuple(atom for atom, present in zip(component.atoms, keep) if present),
        bonds=(
            None
            if bonds is None
            else frozenset(bond for bond in bonds if bond.atom_a not in removed and bond.atom_b not in removed)
        ),
        previous_attachment=previous,
        next_attachment=following,
        reference_coordinates=(
            tuple(coordinate for coordinate, present in zip(component.reference_coordinates, keep) if present)
            if component.reference_coordinates
            else ()
        ),
        leaving_atom_ids=component.leaving_atom_ids & retained,
        non_leaving_atom_ids=component.non_leaving_atom_ids & retained,
    )


def _component_atom_groups(
    component: ChemicalComponent,
    atom_ids: Collection[str],
) -> tuple[tuple[frozenset[str], frozenset[str]], ...]:
    r"""Connected selected atom groups and their boundaries in a component graph."""

    selected = set(atom_ids)
    if not selected:
        return ()
    component_atoms = {atom.atom_id for atom in component.atoms}
    if not selected <= component_atoms:
        raise ValueError(f"component {component.component_id!r} leaving group references an unknown atom")
    if component.bonds is None:
        raise ValueError(f"component {component.component_id!r} has no known bond graph")
    adjacency: dict[str, set[str]] = {atom_id: set() for atom_id in component_atoms}
    for bond in component.bonds:
        adjacency[bond.atom_a].add(bond.atom_b)
        adjacency[bond.atom_b].add(bond.atom_a)

    groups: list[tuple[frozenset[str], frozenset[str]]] = []
    unseen = set(selected)
    for atom in component.atoms:
        if atom.atom_id not in unseen:
            continue
        unseen.remove(atom.atom_id)
        group = {atom.atom_id}
        pending = [atom.atom_id]
        while pending:
            for neighbor in adjacency[pending.pop()] & unseen:
                unseen.remove(neighbor)
                group.add(neighbor)
                pending.append(neighbor)
        boundary = {neighbor for member in group for neighbor in adjacency[member] if neighbor not in group}
        groups.append((frozenset(group), frozenset(boundary)))
    return tuple(groups)


def _component_leaving_atom(
    component: ChemicalComponent,
    group: Collection[str],
    endpoint: str,
) -> str:
    r"""Return the unique atom through which a leaving group is bonded to its endpoint."""

    if component.bonds is None:
        raise ValueError(f"component {component.component_id!r} has no known bond graph")
    leaving_atoms = set(group)
    candidates = {
        bond.atom_b if bond.atom_a == endpoint else bond.atom_a
        for bond in component.bonds
        if (bond.atom_a == endpoint and bond.atom_b in leaving_atoms)
        or (bond.atom_b == endpoint and bond.atom_a in leaving_atoms)
    }
    if len(candidates) != 1:
        raise ValueError(f"mmCIF cannot infer a unique leaving atom for {component.component_id!r}:{endpoint}")
    return next(iter(candidates))


def read_mmcif_branch_entities(
    cif_file: object,
    component_table: Mapping[str, ChemicalComponent],
    selected_rows: np.recarray,
) -> tuple[
    dict[str, dict[str | None, Molecule]],
    dict[tuple[object, ...], int],
    dict[tuple[str, str | None], tuple[tuple[MoleculeBond, bool, bool], ...]],
]:
    r"""
    Decode selected mmCIF branched entities and source component indices.

    Args:
        cif_file: Parsed mmCIF document.
        component_table: Chemical components available by component ID.
        selected_rows: Atom rows after alternate-location selection.

    Returns:
        Branched-entity realizations by entity and label asym ID, selected source
        residue keys mapped to component indices, and the corresponding source link
        declarations before their leaving atoms are removed. ``None`` identifies an
        entity-wide realization whose declaration has no microheterogeneous positions.

    Raises:
        ValueError: If branch declarations, links, bond orders, component references,
            or source numbering are invalid.
    """

    block = cast("CIFFile", cif_file).block
    if "pdbx_entity_branch_list" not in block:
        return {}, {}, {}

    category = block["pdbx_entity_branch_list"]
    entity_ids = _cif_column(category, "entity_id")
    component_ids = _cif_column(category, "comp_id")
    numbers = _cif_column(category, "num")
    declarations: dict[str, dict[int, list[str]]] = {}
    for entity_id, component_id, number in zip(entity_ids, component_ids, numbers):
        if entity_id is None:
            raise ValueError("mmCIF branched-entity component is missing entity_id")
        component_index = optional_integer(number)
        if component_id is None or component_index is None:
            raise ValueError("mmCIF branched-entity components require entity_id, comp_id, and num")
        candidates = declarations.setdefault(entity_id, {}).setdefault(component_index, [])
        if component_id in candidates:
            raise ValueError(
                f"duplicate branched-entity component {component_id!r} at number "
                f"{component_index} for entity {entity_id!r}"
            )
        candidates.append(component_id)

    number_to_index = {
        entity_id: {number: index for index, number in enumerate(sorted(components))}
        for entity_id, components in declarations.items()
    }
    source_choices: dict[tuple[object, ...], set[tuple[str, str, int, str]]] = {}
    scheme_records: list[
        tuple[
            str,
            str,
            int,
            str,
            tuple[tuple[object, ...], ...],
        ]
    ] = []
    scheme_asym_ids: dict[str, list[str]] = {}
    if "pdbx_branch_scheme" in block:
        scheme = block["pdbx_branch_scheme"]
        label_asym_ids = _cif_column(scheme, "asym_id")
        scheme_entities = _cif_column(scheme, "entity_id")
        monomer_ids = _cif_column(scheme, "mon_id")
        scheme_numbers = _cif_column(scheme, "num")
        pdb_asym_ids = _cif_column(scheme, "pdb_asym_id")
        pdb_monomer_ids = _cif_column(scheme, "pdb_mon_id")
        pdb_sequence_numbers = _cif_column(scheme, "pdb_seq_num")
        pdb_insertion_codes = _cif_column(scheme, "pdb_ins_code")
        for values in zip(
            label_asym_ids,
            scheme_entities,
            monomer_ids,
            scheme_numbers,
            pdb_asym_ids,
            pdb_monomer_ids,
            pdb_sequence_numbers,
            pdb_insertion_codes,
        ):
            (
                label_asym_id,
                entity_id,
                monomer_id,
                number,
                pdb_asym_id,
                pdb_monomer_id,
                pdb_sequence_number,
                pdb_insertion_code,
            ) = values
            if label_asym_id is None or entity_id is None or entity_id not in declarations:
                raise ValueError(f"mmCIF branch scheme references undeclared entity {entity_id!r}")
            number_int = optional_integer(number)
            if number_int is None:
                raise ValueError("mmCIF branch scheme requires an integer component number")
            declared_candidates = declarations[entity_id].get(number_int)
            if declared_candidates is None or monomer_id is None or monomer_id not in declared_candidates:
                raise ValueError(f"mmCIF branch scheme disagrees with component declaration for entity {entity_id!r}")
            pdb_sequence_number = present_string(pdb_sequence_number)
            pdb_insertion_code = present_string(pdb_insertion_code)
            source_keys: set[tuple[object, ...]] = {
                (label_asym_id, entity_id, monomer_id, label_asym_id, None, pdb_insertion_code),
            }
            if pdb_asym_id is not None and pdb_monomer_id is not None and pdb_sequence_number is not None:
                source_keys.add(
                    (
                        label_asym_id,
                        entity_id,
                        pdb_monomer_id,
                        pdb_asym_id,
                        pdb_sequence_number,
                        pdb_insertion_code,
                    )
                )
            choice = (label_asym_id, entity_id, number_int, monomer_id)
            for source_key in source_keys:
                source_choices.setdefault(source_key, set()).add(choice)
            scheme_records.append((label_asym_id, entity_id, number_int, monomer_id, tuple(source_keys)))
            asym_ids = scheme_asym_ids.setdefault(entity_id, [])
            if label_asym_id not in asym_ids:
                asym_ids.append(label_asym_id)

    selected_components: dict[tuple[str, str, int], str] = {}
    selected_atom_ids: dict[tuple[str, int, str], set[str]] = {}
    for row in selected_rows:
        entity_id = present_string(row.entity_id)
        if entity_id not in declarations:
            continue
        choices: set[tuple[str, str, int, str]] = set()
        source_keys = {
            (row.chain_id, row.entity_id, row.component_id, row.chain_id, None, row.insertion_code),
            (
                row.chain_id,
                row.entity_id,
                row.auth_component_id,
                row.auth_chain_id,
                row.auth_residue_id,
                row.insertion_code,
            ),
        }
        for source_key in source_keys:
            source_candidate_choices = source_choices.get(source_key, ())
            if len(source_candidate_choices) == 1:
                choices.update(source_candidate_choices)
        if len(choices) > 1:
            raise ValueError("selected atom rows ambiguously identify a branched-entity component")
        if not choices:
            continue
        selected_asym_id, selected_entity_id, selected_number, selected_component_id = next(iter(choices))
        selected_key = selected_entity_id, selected_asym_id, selected_number
        previous_selection = selected_components.get(selected_key)
        if previous_selection is not None and previous_selection != selected_component_id:
            raise ValueError(
                f"selected atom rows contain multiple branched-entity candidates at "
                f"number {selected_number} for asym {selected_asym_id!r}"
            )
        selected_components[selected_key] = selected_component_id
        if bool(row.coordinate_present):
            selected_atom_ids.setdefault(
                (selected_entity_id, selected_number, selected_component_id),
                set(),
            ).add(str(row.atom_name))

    realized_components: dict[tuple[str, str | None], dict[int, str]] = {}
    for declared_entity_id, numbered_candidates in declarations.items():
        if all(len(candidates) == 1 for candidates in numbered_candidates.values()):
            realized_components[(declared_entity_id, None)] = {
                component_number: candidates[0] for component_number, candidates in numbered_candidates.items()
            }
            continue
        for realization_asym_id in scheme_asym_ids.get(declared_entity_id, ()):
            realization: dict[int, str] = {}
            for component_number, candidates in numbered_candidates.items():
                if len(candidates) == 1:
                    realization[component_number] = candidates[0]
                    continue
                selected = selected_components.get((declared_entity_id, realization_asym_id, component_number))
                if selected is None:
                    break
                realization[component_number] = selected
            else:
                realized_components[(declared_entity_id, realization_asym_id)] = realization

    branch_links: list[
        tuple[
            str,
            int,
            str,
            str,
            str | None,
            int,
            str,
            str,
            str | None,
            float | None,
            bool | None,
        ]
    ] = []
    if "pdbx_entity_branch_link" in block:
        links = block["pdbx_entity_branch_link"]
        link_entities = _cif_column(links, "entity_id")
        number_1 = _cif_column(links, "entity_branch_list_num_1")
        number_2 = _cif_column(links, "entity_branch_list_num_2")
        component_1 = _cif_column(links, "comp_id_1")
        component_2 = _cif_column(links, "comp_id_2")
        atom_1 = _cif_column(links, "atom_id_1")
        atom_2 = _cif_column(links, "atom_id_2")
        leaving_1 = _cif_column(links, "leaving_atom_id_1")
        leaving_2 = _cif_column(links, "leaving_atom_id_2")
        orders = _cif_column(links, "value_order") if "value_order" in links else ("sing",) * links.row_count
        for values in zip(
            link_entities,
            number_1,
            component_1,
            atom_1,
            leaving_1,
            number_2,
            component_2,
            atom_2,
            leaving_2,
            orders,
        ):
            (
                entity_id,
                first_number,
                first_component,
                first_atom,
                first_leaving,
                second_number,
                second_component,
                second_atom,
                second_leaving,
                order_name,
            ) = values
            if entity_id not in declarations:
                raise ValueError(f"mmCIF branched-entity link references undeclared entity {entity_id!r}")
            first_number_int = optional_integer(first_number)
            second_number_int = optional_integer(second_number)
            if first_number_int is None or second_number_int is None or first_atom is None or second_atom is None:
                raise ValueError("mmCIF branched-entity links require two component numbers and atom IDs")
            declared = declarations[entity_id]
            if (
                first_component is None
                or second_component is None
                or first_component not in declared.get(first_number_int, ())
                or second_component not in declared.get(second_number_int, ())
            ):
                raise ValueError(f"branched-entity link disagrees with component declaration for entity {entity_id!r}")
            order, aromatic = _cif_bond_order(order_name)
            if order is not None and aromatic is None:
                aromatic = False
            branch_links.append(
                (
                    entity_id,
                    first_number_int,
                    first_component,
                    first_atom,
                    first_leaving,
                    second_number_int,
                    second_component,
                    second_atom,
                    second_leaving,
                    order,
                    aromatic,
                )
            )

    entities: dict[str, dict[str | None, Molecule]] = {entity_id: {} for entity_id in declarations}
    source_bonds: dict[tuple[str, str | None], tuple[tuple[MoleculeBond, bool, bool], ...]] = {}
    for (entity_id, label_asym_id), numbered_components in realized_components.items():
        removed_atoms: dict[int, set[str]] = {}
        bonds: set[MoleculeBond] = set()
        declared_bonds: list[tuple[MoleculeBond, bool, bool]] = []

        def branch_leaving_group(
            component_id: str,
            endpoint: str,
            leaving_atom: str | None,
        ) -> frozenset[str]:
            try:
                component = component_table[component_id]
            except KeyError:
                raise ValueError(
                    f"unknown component {component_id!r}; pass components={{...}} with its complete declared graph"
                ) from None
            atom_ids = tuple(atom.atom_id for atom in component.atoms)
            if endpoint not in atom_ids:
                raise ValueError(
                    f"mmCIF branched-entity link endpoint {component_id!r}:{endpoint} is not in its component"
                )
            if leaving_atom is None:
                return frozenset()
            if leaving_atom not in atom_ids or leaving_atom == endpoint:
                raise ValueError(
                    f"mmCIF branched-entity link {component_id!r}:{endpoint} has invalid leaving atom "
                    f"{leaving_atom!r}"
                )
            if component.bonds is None:
                raise ValueError(f"component {component_id!r} has no known bond graph for its leaving atom")
            groups = terminal_leaving_groups(
                atom_ids,
                ((bond.atom_a, bond.atom_b) for bond in component.bonds),
                {leaving_atom},
                endpoint,
            )
            if len(groups) != 1:
                raise ValueError(
                    f"mmCIF branched-entity link {component_id!r}:{endpoint} has invalid leaving atom "
                    f"{leaving_atom!r}"
                )
            return groups[0]

        for (
            link_entity_id,
            first_number,
            first_component,
            first_atom,
            first_leaving,
            second_number,
            second_component,
            second_atom,
            second_leaving,
            order,
            aromatic,
        ) in branch_links:
            if link_entity_id != entity_id or (
                numbered_components[first_number] != first_component
                or numbered_components[second_number] != second_component
            ):
                continue
            first_index = number_to_index[entity_id][first_number]
            second_index = number_to_index[entity_id][second_number]
            first_leaving_group = branch_leaving_group(first_component, first_atom, first_leaving)
            second_leaving_group = branch_leaving_group(second_component, second_atom, second_leaving)
            for component_number, component_id, leaving_group in (
                (first_number, first_component, first_leaving_group),
                (second_number, second_component, second_leaving_group),
            ):
                observed_leaving = leaving_group & selected_atom_ids.get(
                    (entity_id, component_number, component_id),
                    set(),
                )
                if observed_leaving:
                    raise ValueError(
                        f"mmCIF branched-entity link for entity {entity_id!r} component "
                        f"{component_number} {component_id!r} declares observed leaving "
                        f"atom {sorted(observed_leaving)[0]!r}"
                    )
            bonds.add(MoleculeBond(first_index, first_atom, second_index, second_atom, order=order, aromatic=aromatic))
            declared_bonds.append(
                (
                    MoleculeBond(
                        first_index,
                        first_atom,
                        second_index,
                        second_atom,
                        order=order,
                        aromatic=aromatic,
                        leaving_a=first_leaving_group,
                        leaving_b=second_leaving_group,
                    ),
                    first_leaving is not None,
                    second_leaving is not None,
                )
            )
            removed_atoms.setdefault(first_index, set()).update(first_leaving_group)
            removed_atoms.setdefault(second_index, set()).update(second_leaving_group)

        components: list[ChemicalComponent] = []
        for component_index, component_number in enumerate(sorted(numbered_components)):
            component_id = numbered_components[component_number]
            try:
                component = component_table[component_id]
            except KeyError:
                raise ValueError(
                    f"unknown component {component_id!r}; pass components={{...}} with its complete declared graph"
                ) from None
            components.append(_remove_component_atoms(component, removed_atoms.get(component_index, set())))
        entities[entity_id][label_asym_id] = Molecule(
            components=tuple(components),
            bonds=frozenset(bonds),
            molecule_type="branched",
        )
        source_bonds[(entity_id, label_asym_id)] = tuple(declared_bonds)

    component_indices: dict[tuple[object, ...], int] = {}
    ambiguous_component_indices: set[tuple[object, ...]] = set()
    for scheme_asym_id, scheme_entity_id, scheme_number, scheme_component_id, scheme_source_keys in scheme_records:
        selected_realization = realized_components.get((scheme_entity_id, scheme_asym_id))
        if selected_realization is None:
            selected_realization = realized_components.get((scheme_entity_id, None))
        if selected_realization is None or selected_realization[scheme_number] != scheme_component_id:
            continue
        selected_component_index = number_to_index[scheme_entity_id][scheme_number]
        for selected_source_key in scheme_source_keys:
            if selected_source_key in ambiguous_component_indices:
                continue
            previous_component_index = component_indices.get(selected_source_key)
            if previous_component_index is not None and previous_component_index != selected_component_index:
                component_indices.pop(selected_source_key)
                ambiguous_component_indices.add(selected_source_key)
                continue
            component_indices[selected_source_key] = selected_component_index
    return entities, component_indices, source_bonds


def _is_metal_site(site: AtomSite, element_by_site: dict[AtomSite, str]) -> bool:
    r"""Whether one atom occurrence is a metal center, from source or component chemistry."""

    element = element_by_site.get(site)
    return element is not None and element not in NON_METAL_ELEMENTS


def read_mmcif_bonds(
    cif_file: object,
    source_rows: np.recarray,
    selected_rows: np.recarray,
    site_by_row_index: dict[int, AtomSite],
    unmodeled_source_sites: dict[tuple[object, ...], AtomSite],
    ambiguous_unmodeled_sites: set[tuple[object, ...]],
    component_table: Mapping[str, ChemicalComponent] | None = None,
    *,
    placements: Mapping[tuple[str | None, str], np.ndarray],
    component_topology_complete: Mapping[str, bool],
    source_non_leaving_atom_ids: Mapping[str, frozenset[str]],
    declared_branch_bonds: Mapping[frozenset[AtomSite], tuple[SystemBond, bool, bool]] | None = None,
) -> tuple[SystemBond, ...]:
    r"""
    Decode selected mmCIF covalent connections into system bonds.

    Args:
        cif_file: Parsed mmCIF document.
        source_rows: Atom rows before user-requested alternate-location and water
            projection.
        selected_rows: Atom rows retained for the selected structure.
        site_by_row_index: Canonical atom sites indexed by source row.
        unmodeled_source_sites: Declared but unobserved canonical atoms indexed
            by their label identities.
        ambiguous_unmodeled_sites: Source identities that name multiple unobserved
            canonical atoms.
        component_table: Component definitions used to validate or uniquely infer
            connection leaving atoms retained in the canonical graph.
        placements: Cartesian placement of each selected label-asym copy relative to
            the source asymmetric unit.
        component_topology_complete: Whether each component definition closes its
            atom and bond graph.
        source_non_leaving_atom_ids: Component atoms the source itself explicitly
            marks as non-leaving.
        declared_branch_bonds: Occurrence-local bonds already represented by
            ``pdbx_entity_branch_link`` in a branched entity, keyed by physical edge.

    Returns:
        Resolved covalent system bonds. Connections whose declared partner placement is
        absent from the selected model or assembly are skipped.

    Raises:
        ValueError: If a selected covalent connection declares an unsupported bond order,
            unresolved crystallographic operation, or conflicting or ambiguous atom identities.
    """

    block = cast("CIFFile", cif_file).block
    if declared_branch_bonds is None:
        declared_branch_bonds = {}
    if "struct_conn" not in block:
        return ()
    category = block["struct_conn"]
    source_endpoint_index = _mmcif_endpoint_index(source_rows)
    source_positions = {int(row_index): index for index, row_index in enumerate(source_rows["row_index"])}
    try:
        selected_source_positions = frozenset(
            source_positions[int(row_index)] for row_index in selected_rows["row_index"]
        )
    except KeyError as error:
        raise ValueError(f"selected mmCIF atom row {error.args[0]} is absent from the source rows") from None

    def alternate_ids(*column_names: str) -> tuple[str | None, ...]:
        columns = [_cif_column(category, name) for name in column_names if name in category]
        if not columns:
            return (None,) * category.row_count
        merged: list[str | None] = []
        for row_index, values in enumerate(zip(*columns)):
            present = {value for value in values if value is not None}
            if len(present) > 1:
                raise ValueError(f"mmCIF struct_conn row {row_index} has conflicting alternate-location identities")
            merged.append(next(iter(present)) if present else None)
        return tuple(merged)

    conn_types = _cif_column(category, "conn_type_id")
    label_asym_1 = _cif_column(category, "ptnr1_label_asym_id")
    label_asym_2 = _cif_column(category, "ptnr2_label_asym_id")
    label_seq_1 = _cif_column(category, "ptnr1_label_seq_id")
    label_seq_2 = _cif_column(category, "ptnr2_label_seq_id")
    label_comp_1 = _cif_column(category, "ptnr1_label_comp_id")
    label_comp_2 = _cif_column(category, "ptnr2_label_comp_id")
    label_atom_1 = _cif_column(category, "ptnr1_label_atom_id")
    label_atom_2 = _cif_column(category, "ptnr2_label_atom_id")
    auth_asym_1 = _cif_column(category, "ptnr1_auth_asym_id")
    auth_asym_2 = _cif_column(category, "ptnr2_auth_asym_id")
    auth_seq_1 = _cif_column(category, "ptnr1_auth_seq_id")
    auth_seq_2 = _cif_column(category, "ptnr2_auth_seq_id")
    auth_comp_1 = _cif_column(category, "ptnr1_auth_comp_id")
    auth_comp_2 = _cif_column(category, "ptnr2_auth_comp_id")
    auth_atom_1 = _cif_column(category, "ptnr1_auth_atom_id")
    auth_atom_2 = _cif_column(category, "ptnr2_auth_atom_id")
    alt_1 = alternate_ids(
        "pdbx_ptnr1_label_alt_id",
        "ptnr1_label_alt_id",
        "pdbx_ptnr1_auth_alt_id",
    )
    alt_2 = alternate_ids(
        "pdbx_ptnr2_label_alt_id",
        "ptnr2_label_alt_id",
        "pdbx_ptnr2_auth_alt_id",
    )
    ins_1 = _cif_column(category, "pdbx_ptnr1_PDB_ins_code")
    ins_2 = _cif_column(category, "pdbx_ptnr2_PDB_ins_code")
    leaving_flags = _cif_column(category, "pdbx_leaving_atom_flag")
    leaving_ids_1 = _cif_column(category, "pdbx_ptnr1_leaving_atom_id")
    leaving_ids_2 = _cif_column(category, "pdbx_ptnr2_leaving_atom_id")
    roles_1 = _cif_column(category, "ptnr1_role")
    roles_2 = _cif_column(category, "ptnr2_role")
    symmetry_1 = _cif_column(category, "ptnr1_symmetry")
    symmetry_2 = _cif_column(category, "ptnr2_symmetry")
    value_orders = (
        _cif_column(category, "pdbx_value_order") if "pdbx_value_order" in category else (None,) * category.row_count
    )
    selected_sites: dict[tuple[object, ...], AtomSite] = {}
    ambiguous_source_sites: set[tuple[object, ...]] = set()
    element_by_site: dict[AtomSite, str] = {}
    altloc_by_site: dict[AtomSite, str | None] = {}
    observed_atom_id_sets: dict[tuple[str, int], set[str]] = {}
    selected_row_indices = selected_rows["row_index"]
    selected_symmetry_ids = selected_rows["symmetry_id"]
    selected_chain_ids = selected_rows["chain_id"]
    selected_label_seq_ids = selected_rows["label_seq_id"]
    selected_insertion_codes = selected_rows["insertion_code"]
    selected_component_ids = selected_rows["component_id"]
    selected_atom_names = selected_rows["atom_name"]
    selected_auth_chain_ids = selected_rows["auth_chain_id"]
    selected_auth_residue_ids = selected_rows["auth_residue_id"]
    selected_auth_component_ids = selected_rows["auth_component_id"]
    selected_auth_atom_names = selected_rows["auth_atom_name"]
    selected_altloc_ids = selected_rows["altloc_id"]
    selected_coordinate_present = selected_rows["coordinate_present"]
    selected_elements = selected_rows["element"]
    for row_position in range(len(selected_rows)):
        row_index = int(selected_row_indices[row_position])
        try:
            site = site_by_row_index[row_index]
        except KeyError:
            raise ValueError(f"selected mmCIF atom row {row_index} has no canonical atom site") from None
        label_key = (
            selected_symmetry_ids[row_position],
            selected_chain_ids[row_position],
            selected_label_seq_ids[row_position],
            selected_insertion_codes[row_position],
            selected_component_ids[row_position],
            selected_atom_names[row_position],
        )
        auth_key = (
            selected_symmetry_ids[row_position],
            selected_auth_chain_ids[row_position],
            selected_auth_residue_ids[row_position],
            selected_insertion_codes[row_position],
            selected_auth_component_ids[row_position],
            selected_auth_atom_names[row_position],
        )
        for selected_key in (label_key, auth_key):
            if selected_key in ambiguous_source_sites:
                continue
            previous = selected_sites.get(selected_key)
            if previous is not None and previous != site:
                selected_sites.pop(selected_key)
                ambiguous_source_sites.add(selected_key)
            else:
                selected_sites[selected_key] = site
        altloc_by_site[site] = selected_altloc_ids[row_position]
        if bool(selected_coordinate_present[row_position]):
            observed_atom_id_sets.setdefault((site.instance_id, site.component_index), set()).add(site.atom_id)
        element = selected_elements[row_position]
        if element:
            element_by_site[site] = str(element)

    ambiguous_source_sites.update(ambiguous_unmodeled_sites)
    for source_key, site in unmodeled_source_sites.items():
        if source_key not in ambiguous_source_sites:
            selected_sites[source_key] = site
    observed_atom_ids_by_occurrence = {
        occurrence: frozenset(atom_ids) for occurrence, atom_ids in observed_atom_id_sets.items()
    }
    component_elements = (
        {}
        if component_table is None
        else {
            component_id: {atom.atom_id: atom.element for atom in component.atoms}
            for component_id, component in component_table.items()
        }
    )
    for source_key, site in unmodeled_source_sites.items():
        component_id, atom_id = source_key[-2:]
        if component_id is None or atom_id is None:
            continue
        element = component_elements.get(str(component_id), {}).get(str(atom_id))
        if element is not None:
            element_by_site.setdefault(site, element)

    def candidate_leaving_groups(
        component: ChemicalComponent,
        endpoint: str,
        known_non_leaving: frozenset[str],
        *,
        include_unmarked: bool,
    ) -> set[frozenset[str]]:
        assert component.bonds is not None
        atom_ids = tuple(atom.atom_id for atom in component.atoms)
        bond_pairs = tuple((bond.atom_a, bond.atom_b) for bond in component.bonds)
        possible_leaving = component.leaving_atom_ids
        if not possible_leaving and include_unmarked:
            possible_leaving = frozenset(atom_ids) - known_non_leaving - {endpoint}
        candidates = set(terminal_leaving_groups(atom_ids, bond_pairs, possible_leaving, endpoint))
        for attachment in (component.previous_attachment, component.next_attachment):
            if (
                attachment is None
                or attachment.atom_id != endpoint
                or not attachment.leaving_atom_ids
                or attachment.leaving_atom_ids & known_non_leaving
            ):
                continue
            groups = terminal_leaving_groups(atom_ids, bond_pairs, attachment.leaving_atom_ids, endpoint)
            if len(groups) != 1:
                raise ValueError(f"component {component.component_id!r} has an ambiguous attachment leaving group")
            candidates.add(groups[0])
        return candidates

    def explicit_leaving_group(
        component: ChemicalComponent,
        endpoint: str,
        leaving_atom: str,
        candidates: set[frozenset[str]],
        known_non_leaving: frozenset[str],
        *,
        chemistry_complete: bool,
    ) -> frozenset[str]:
        atom_by_id = {atom.atom_id: atom for atom in component.atoms}
        if leaving_atom not in atom_by_id or leaving_atom == endpoint:
            raise ValueError(
                f"mmCIF struct_conn endpoint {component.component_id!r}:{endpoint} "
                f"has invalid leaving atom {leaving_atom!r}"
            )
        if leaving_atom in known_non_leaving:
            raise ValueError(
                f"mmCIF struct_conn endpoint {component.component_id!r}:{endpoint} "
                f"declares known non-leaving atom {leaving_atom!r} as leaving"
            )
        if not chemistry_complete or component.bonds is None:
            raise ValueError(
                f"mmCIF struct_conn endpoint {component.component_id!r}:{endpoint} "
                "has incomplete chemistry for its leaving atom"
            )
        matches = {group for group in candidates if leaving_atom in group}
        if len(matches) > 1:
            raise ValueError(
                f"mmCIF struct_conn endpoint {component.component_id!r}:{endpoint} "
                "has an ambiguous component leaving group"
            )
        if matches:
            return next(iter(matches))

        assert component.bonds is not None
        groups = terminal_leaving_groups(
            tuple(atom_by_id),
            ((bond.atom_a, bond.atom_b) for bond in component.bonds),
            {leaving_atom},
            endpoint,
        )
        if len(groups) != 1:
            raise ValueError(
                f"mmCIF struct_conn endpoint {component.component_id!r}:{endpoint} "
                f"has invalid leaving atom {leaving_atom!r}"
            )
        return groups[0]

    def unobserved_leaving_group(
        component: ChemicalComponent,
        endpoint: str,
        group: frozenset[str],
        observed_atom_ids: frozenset[str],
    ) -> frozenset[str]:
        r"""Relabel only graph-equivalent terminal hydrogens to avoid deleting observations."""

        observed_leaving = group & observed_atom_ids
        if not observed_leaving:
            return group
        if component.bonds is None:
            raise ValueError(f"component {component.component_id!r} has no known bond graph")
        atom_by_id = {atom.atom_id: atom for atom in component.atoms}
        incident: dict[str, list[ChemicalBond]] = {atom.atom_id: [] for atom in component.atoms}
        stereo_atoms = {
            atom_id for bond in component.bonds if bond.stereo_atoms is not None for atom_id in bond.stereo_atoms
        }
        for bond in component.bonds:
            incident[bond.atom_a].append(bond)
            incident[bond.atom_b].append(bond)

        def signature(atom_id: str) -> tuple[object, ...] | None:
            atom = atom_by_id[atom_id]
            bonds = incident[atom_id]
            if (
                atom.element not in HYDROGEN_ELEMENTS
                or atom.stereo is not None
                or atom_id in stereo_atoms
                or len(bonds) != 1
            ):
                return None
            bond = bonds[0]
            neighbor = bond.atom_b if bond.atom_a == atom_id else bond.atom_a
            if neighbor != endpoint or bond.stereo is not None or bond.stereo_atoms is not None:
                return None
            return (
                atom.element,
                atom.isotope,
                atom.formal_charge,
                atom.radical_electrons,
                atom.explicit_hydrogens,
                atom.implicit_hydrogens,
                atom.implicit_hydrogens_allowed,
                atom.aromatic,
                bond.order,
                bond.aromatic,
                bond.kind,
                bond.direction,
            )

        replacement = set(group)
        available = set(atom_by_id) - set(group) - set(observed_atom_ids)
        for atom_id in sorted(observed_leaving):
            atom_signature = signature(atom_id)
            alternatives = sorted(candidate for candidate in available if signature(candidate) == atom_signature)
            if atom_signature is None or not alternatives:
                raise ValueError(
                    f"component {component.component_id!r} leaving group contains observed atom {atom_id!r}"
                )
            replacement.remove(atom_id)
            replacement.add(alternatives[0])
            available.remove(alternatives[0])
        return frozenset(replacement)

    bonds: list[SystemBond] = []
    for index, raw_conn_type in enumerate(conn_types):
        if raw_conn_type is None:
            raise ValueError(f"mmCIF struct_conn row {index} has no connection type")
        conn_type = raw_conn_type.casefold()
        if conn_type in {"hydrog", "mismat", "saltbr"}:
            continue
        if not (
            conn_type.startswith("covale")
            or conn_type.startswith("disulf")
            or conn_type.startswith("metalc")
            or conn_type == "modres"
        ):
            raise ValueError(f"mmCIF struct_conn row {index} has unsupported connection type {raw_conn_type!r}")
        value_order = value_orders[index]
        if value_order is not None and value_order.lower() == "arom":
            raise ValueError("mmCIF struct_conn pdbx_value_order does not support aromatic bonds")
        order, aromatic = _cif_bond_order(value_order)
        # `struct_conn` has no aromatic-flag column, so unlike `chem_comp_bond` its order
        # is the only channel: an explicit non-aromatic order states non-aromaticity.
        if order is not None and aromatic is None:
            aromatic = False
        if conn_type.startswith("disulf") and order is None:
            order, aromatic = 1.0, False
        direction: BondDirection | None = "undirected"
        if conn_type.startswith("metalc"):
            direction_evidence: set[BondDirection] = set()
            for role_endpoint, role in enumerate(
                (roles_1[index], roles_2[index]),
                start=1,
            ):
                normalized = None if role is None else role.casefold()
                if normalized in {"donor", "metal coordination"}:
                    direction_evidence.add("a_to_b" if role_endpoint == 1 else "b_to_a")
                elif normalized in {"acceptor", "metal"}:
                    direction_evidence.add("b_to_a" if role_endpoint == 1 else "a_to_b")
            direction = direction_evidence.pop() if len(direction_evidence) == 1 else None
        endpoint_1 = (
            label_asym_1[index],
            optional_integer(label_seq_1[index]),
            ins_1[index],
            label_comp_1[index],
            label_atom_1[index],
        )
        endpoint_2 = (
            label_asym_2[index],
            optional_integer(label_seq_2[index]),
            ins_2[index],
            label_comp_2[index],
            label_atom_2[index],
        )
        auth_endpoint_1 = (
            auth_asym_1[index] or label_asym_1[index],
            present_string(auth_seq_1[index]) or label_seq_1[index],
            ins_1[index],
            auth_comp_1[index] or label_comp_1[index],
            auth_atom_1[index] or label_atom_1[index],
        )
        auth_endpoint_2 = (
            auth_asym_2[index] or label_asym_2[index],
            present_string(auth_seq_2[index]) or label_seq_2[index],
            ins_2[index],
            auth_comp_2[index] or label_comp_2[index],
            auth_atom_2[index] or label_atom_2[index],
        )

        def endpoint_is_declared(
            label_endpoint: tuple[object, ...],
            auth_endpoint: tuple[object, ...],
            required_altloc: str | None,
        ) -> bool:
            endpoint_identity = (label_endpoint, auth_endpoint, required_altloc, None)
            if _mmcif_endpoint_rows(source_endpoint_index, endpoint_identity):
                return True
            tails = {label_endpoint, auth_endpoint}
            return any(key[1:] in tails for key in unmodeled_source_sites)

        declared_endpoints = (
            endpoint_is_declared(endpoint_1, auth_endpoint_1, alt_1[index]),
            endpoint_is_declared(endpoint_2, auth_endpoint_2, alt_2[index]),
        )
        if not all(declared_endpoints):
            missing = ", ".join(
                str(endpoint + 1) for endpoint, declared in enumerate(declared_endpoints) if not declared
            )
            raise ValueError(
                f"mmCIF struct_conn row {index} cannot resolve endpoint {missing} "
                "in the source molecular declaration"
            )

        raw_leaving_flag = leaving_flags[index]
        leaving_flag = None if raw_leaving_flag is None else raw_leaving_flag.casefold()
        if leaving_flag not in {None, "none", "one", "both"}:
            raise ValueError(f"unsupported mmCIF struct_conn leaving-atom flag: {leaving_flags[index]!r}")
        crystallographic_pair = _mmcif_crystallographic_pair(block, symmetry_1[index], symmetry_2[index])
        crystallographic_a = crystallographic_pair[0][1]
        crystallographic_b = crystallographic_pair[1][1]
        try:
            inverse_a = np.linalg.inv(crystallographic_a)
            inverse_b = np.linalg.inv(crystallographic_b)
        except np.linalg.LinAlgError:
            raise ValueError(f"mmCIF struct_conn row {index} uses a singular crystallographic operation") from None
        asym_a = label_asym_1[index] or auth_asym_1[index]
        asym_b = label_asym_2[index] or auth_asym_2[index]
        candidates_a = (
            ()
            if asym_a is None
            else tuple(
                (symmetry_id, placement @ inverse_a)
                for (symmetry_id, asym_id), placement in placements.items()
                if asym_id == asym_a
            )
        )
        candidates_b = (
            ()
            if asym_b is None
            else tuple(
                (symmetry_id, placement @ inverse_b)
                for (symmetry_id, asym_id), placement in placements.items()
                if asym_id == asym_b
            )
        )
        same_crystallographic_operation = np.allclose(
            crystallographic_a,
            crystallographic_b,
            rtol=0.0,
            atol=1e-8,
        )
        placement_pairs = tuple(
            (symmetry_a, symmetry_b)
            for symmetry_a, global_a in candidates_a
            for symmetry_b, global_b in candidates_b
            if (
                (not same_crystallographic_operation or symmetry_a == symmetry_b)
                and np.allclose(global_a, global_b, rtol=0.0, atol=1e-5)
            )
        )
        for endpoint_symmetry_a, endpoint_symmetry_b in placement_pairs:
            resolved_sites: list[AtomSite | None] = []
            projected_endpoints: list[bool] = []
            for endpoint_number, (endpoint_symmetry, label_endpoint, auth_endpoint, required_altloc) in enumerate(
                (
                    (endpoint_symmetry_a, endpoint_1, auth_endpoint_1, alt_1[index]),
                    (endpoint_symmetry_b, endpoint_2, auth_endpoint_2, alt_2[index]),
                ),
                start=1,
            ):
                endpoint = (label_endpoint, auth_endpoint, required_altloc, None)
                source_match = bool(_mmcif_endpoint_rows(source_endpoint_index, endpoint))
                selected_match = bool(
                    _mmcif_endpoint_rows(
                        source_endpoint_index,
                        endpoint,
                        included_rows=selected_source_positions,
                    )
                )
                projected_endpoints.append(source_match and not selected_match)
                if required_altloc is not None and not selected_match:
                    resolved_sites.append(None)
                    continue
                keys = ((endpoint_symmetry, *label_endpoint), (endpoint_symmetry, *auth_endpoint))
                site_candidates = {selected_sites[key] for key in keys if key in selected_sites}
                if len(site_candidates) > 1 or (
                    not site_candidates and any(key in ambiguous_source_sites for key in keys)
                ):
                    raise ValueError(f"mmCIF struct_conn row {index} endpoint {endpoint_number} is ambiguous")
                resolved_site: AtomSite | None = next(iter(site_candidates)) if site_candidates else None
                if required_altloc is not None and (
                    resolved_site is None or altloc_by_site.get(resolved_site) != required_altloc
                ):
                    resolved_site = None
                resolved_sites.append(resolved_site)
            site_a, site_b = resolved_sites
            if site_a is None or site_b is None:
                unresolved = [
                    endpoint + 1
                    for endpoint, (site, projected) in enumerate(zip(resolved_sites, projected_endpoints))
                    if site is None and not projected
                ]
                if unresolved:
                    endpoints = ", ".join(str(endpoint) for endpoint in unresolved)
                    raise ValueError(
                        f"mmCIF struct_conn row {index} cannot resolve endpoint {endpoints} "
                        "in the selected molecular declaration"
                    )
                # Every absent endpoint existed before a caller-requested projection
                # (currently alternate-location or water selection), so omitting this
                # connection is part of that explicit projection rather than data loss.
                continue
            declared_branch = declared_branch_bonds.get(frozenset((site_a, site_b)))
            if direction is None:
                # The archive rarely populates the partner-role columns, so fall back to
                # chemistry: a coordination bond runs from the donor ligand atom to the
                # metal center, and the elements name which endpoint is which. Neither or
                # both being a metal leaves the direction genuinely unknown.
                metal_a = _is_metal_site(site_a, element_by_site)
                metal_b = _is_metal_site(site_b, element_by_site)
                if metal_b and not metal_a:
                    direction = "a_to_b"
                elif metal_a and not metal_b:
                    direction = "b_to_a"
                else:
                    raise ValueError(
                        f"mmCIF struct_conn row {index} declares a metal coordination "
                        "whose donor direction cannot be identified"
                    )

            if declared_branch is not None:
                declared_branch_bond, leaving_a_known, leaving_b_known = declared_branch
                declared_direction = declared_branch_bond.direction
                declared_leaving = (declared_branch_bond.leaving_a, declared_branch_bond.leaving_b)
                declared_leaving_known = (leaving_a_known, leaving_b_known)
                if declared_branch_bond.site_a == site_b and declared_branch_bond.site_b == site_a:
                    declared_leaving = declared_leaving[::-1]
                    declared_leaving_known = declared_leaving_known[::-1]
                    if declared_direction == "a_to_b":
                        declared_direction = "b_to_a"
                    elif declared_direction == "b_to_a":
                        declared_direction = "a_to_b"
                elif declared_branch_bond.site_a != site_a or declared_branch_bond.site_b != site_b:
                    raise ValueError(f"mmCIF struct_conn row {index} duplicates an ambiguous branch link")
                declared_semantics = (
                    declared_branch_bond.order,
                    declared_branch_bond.aromatic,
                    declared_branch_bond.kind,
                    declared_direction,
                )
                decoded_semantics = (
                    order,
                    aromatic,
                    "dative" if conn_type.startswith("metalc") else "covalent",
                    direction,
                )
                semantic_matches = (
                    (declared_semantics[0] is None or order is None or declared_semantics[0] == order)
                    and (declared_semantics[1] is None or aromatic is None or declared_semantics[1] == aromatic)
                    and declared_semantics[2:] == decoded_semantics[2:]
                )
                explicit_leaving = (leaving_ids_1[index], leaving_ids_2[index])
                explicit_matches = all(
                    leaving_id is None or not known or leaving_id in group
                    for leaving_id, group, known in zip(
                        explicit_leaving,
                        declared_leaving,
                        declared_leaving_known,
                        strict=True,
                    )
                )
                known_leaving_count = sum(
                    bool(group) for group, known in zip(declared_leaving, declared_leaving_known) if known
                )
                flag_matches = (
                    leaving_flag is None
                    or (all(declared_leaving_known) and leaving_flag == ("none", "one", "both")[known_leaving_count])
                    or (
                        not all(declared_leaving_known)
                        and not (known_leaving_count and leaving_flag == "none")
                        and not (known_leaving_count == 2 and leaving_flag == "one")
                    )
                )
                if not semantic_matches or not flag_matches or not explicit_matches:
                    raise ValueError(
                        f"mmCIF struct_conn row {index} conflicts with its branched-entity link: "
                        f"declared semantics {declared_semantics!r}, decoded semantics {decoded_semantics!r}, "
                        f"declared leaving {declared_leaving!r} known {declared_leaving_known!r}, "
                        f"flag {leaving_flag!r}, "
                        f"explicit leaving {explicit_leaving!r}"
                    )
                bonds.append(
                    SystemBond(
                        site_a,
                        site_b,
                        order=declared_branch_bond.order if order is None else order,
                        aromatic=declared_branch_bond.aromatic if aromatic is None else aromatic,
                        role="entity",
                        kind=declared_branch_bond.kind,
                        direction=declared_direction,
                    )
                )
                continue

            explicit_leaving = (leaving_ids_1[index], leaving_ids_2[index])
            component_ids = (
                label_comp_1[index] or auth_comp_1[index],
                label_comp_2[index] or auth_comp_2[index],
            )
            endpoint_sites = (site_a, site_b)
            leaving_sets: list[frozenset[str]] = []
            candidate_sets: list[set[frozenset[str]]] = []
            retained_atom_ids: list[frozenset[str]] = []
            endpoint_components: list[ChemicalComponent | None] = []
            observed_atom_ids: list[frozenset[str]] = []
            for endpoint_number, (site, component_id, leaving_id) in enumerate(
                zip(endpoint_sites, component_ids, explicit_leaving),
                start=1,
            ):
                component = (
                    None if component_table is None or component_id is None else component_table.get(component_id)
                )
                endpoint_components.append(component)
                chemistry_complete = (
                    component is not None
                    and component_topology_complete.get(component.component_id, False)
                    and component.bonds is not None
                    and all(atom.element != "X" for atom in component.atoms)
                )
                observed_atom_ids.append(
                    observed_atom_ids_by_occurrence.get((site.instance_id, site.component_index), frozenset())
                )
                retained_atom_ids.append(
                    frozenset() if component is None else frozenset(atom.atom_id for atom in component.atoms)
                )
                known_non_leaving = (
                    frozenset() if component_id is None else source_non_leaving_atom_ids.get(component_id, frozenset())
                )
                leaving_candidates = (
                    set()
                    if component is None or not chemistry_complete
                    else candidate_leaving_groups(
                        component,
                        site.atom_id,
                        known_non_leaving,
                        include_unmarked=leaving_id is not None or leaving_flag in {"one", "both"},
                    )
                )
                if conn_type.startswith("disulf") and leaving_id is None and chemistry_complete:
                    assert component is not None and component.bonds is not None
                    hydrogen_ids = {atom.atom_id for atom in component.atoms if atom.element in HYDROGEN_ELEMENTS}
                    leaving_candidates.update(
                        terminal_leaving_groups(
                            tuple(atom.atom_id for atom in component.atoms),
                            ((bond.atom_a, bond.atom_b) for bond in component.bonds),
                            hydrogen_ids,
                            site.atom_id,
                        )
                    )
                if leaving_id is None and component is not None:
                    viable_candidates: set[frozenset[str]] = set()
                    for candidate in leaving_candidates:
                        try:
                            viable_candidates.add(
                                unobserved_leaving_group(
                                    component,
                                    site.atom_id,
                                    candidate,
                                    observed_atom_ids[-1],
                                )
                            )
                        except ValueError:
                            if conn_type.startswith("disulf"):
                                raise
                            continue
                    leaving_candidates = viable_candidates
                candidate_sets.append(leaving_candidates)
                if leaving_id is None:
                    leaving_sets.append(frozenset())
                elif component is None:
                    raise ValueError(
                        f"mmCIF struct_conn row {index} endpoint {endpoint_number} "
                        "has a leaving atom without a component definition"
                    )
                else:
                    leaving_sets.append(
                        explicit_leaving_group(
                            component,
                            site.atom_id,
                            leaving_id,
                            leaving_candidates,
                            known_non_leaving,
                            chemistry_complete=chemistry_complete,
                        )
                    )

            explicit_count = sum(bool(values) for values in leaving_sets)
            if leaving_flag == "none" and explicit_count:
                raise ValueError(f"mmCIF struct_conn row {index} contradicts its leaving-atom flag")
            if leaving_flag is None:
                for endpoint_index, candidates in enumerate(candidate_sets):
                    projected = {values & retained_atom_ids[endpoint_index] for values in candidates}
                    if len(projected) > 1:
                        raise ValueError(
                            f"mmCIF struct_conn row {index} cannot identify endpoint "
                            f"{endpoint_index + 1}'s leaving group"
                        )
                    if projected and not leaving_sets[endpoint_index]:
                        leaving_sets[endpoint_index] = next(iter(projected))
            elif leaving_flag == "both":
                for endpoint_index in range(2):
                    if leaving_sets[endpoint_index]:
                        continue
                    candidates = candidate_sets[endpoint_index]
                    projected = {values & retained_atom_ids[endpoint_index] for values in candidates}
                    if not candidates or len(projected) != 1:
                        raise ValueError(
                            f"mmCIF struct_conn row {index} cannot identify endpoint "
                            f"{endpoint_index + 1}'s leaving group"
                        )
                    leaving_sets[endpoint_index] = next(iter(projected))
            elif leaving_flag == "one":
                if explicit_count > 1:
                    raise ValueError(f"mmCIF struct_conn row {index} contradicts its leaving-atom flag")
                if explicit_count == 0:
                    inferred = {
                        (
                            group & retained_atom_ids[0] if endpoint_index == 0 else frozenset(),
                            group & retained_atom_ids[1] if endpoint_index == 1 else frozenset(),
                        )
                        for endpoint_index, candidates in enumerate(candidate_sets)
                        for group in candidates
                    }
                    if not any(candidate_sets) or len(inferred) != 1:
                        raise ValueError(f"mmCIF struct_conn row {index} cannot identify its leaving group")
                    leaving_sets = list(next(iter(inferred)))
            leaving_sets = [values & retained for values, retained in zip(leaving_sets, retained_atom_ids)]
            for endpoint_index, component in enumerate(endpoint_components):
                observed_leaving = leaving_sets[endpoint_index] & observed_atom_ids[endpoint_index]
                if not observed_leaving:
                    continue
                if explicit_leaving[endpoint_index] is not None:
                    atom_id = sorted(observed_leaving)[0]
                    raise ValueError(
                        f"mmCIF struct_conn row {index} endpoint {endpoint_index + 1} "
                        f"declares observed leaving atom {atom_id!r}"
                    )
                assert component is not None
                leaving_sets[endpoint_index] = unobserved_leaving_group(
                    component,
                    endpoint_sites[endpoint_index].atom_id,
                    leaving_sets[endpoint_index],
                    observed_atom_ids[endpoint_index],
                )
            decoded_bond = SystemBond(
                site_a,
                site_b,
                order=order,
                aromatic=aromatic,
                kind="dative" if conn_type.startswith("metalc") else "covalent",
                direction=direction,
                role="assembly",
                leaving_a=leaving_sets[0],
                leaving_b=leaving_sets[1],
            )
            bonds.append(decoded_bond)
    return tuple(bonds)


def write_mmcif(structure: Structure, path: Path) -> None:
    r"""Write the observed projection and molecular declarations as standard mmCIF."""

    from biotite.structure.io.pdbx import CIFBlock, CIFCategory, CIFFile

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
    source_formal_charge_mask = structure.source_formal_charge_mask.detach().cpu()
    resolution = structure.resolution.detach().cpu()
    resolution_kind = structure.resolution_kind.detach().cpu()
    instance_entity = system.instance_entity.detach()
    entity_type = system.entity_type.detach()
    entity_molecule_type = system.entity_molecule_type.detach()
    instance_component_ptr = system.instance_component_ptr.detach()
    component_symbol = system.component_symbol.detach()
    component_topology_complete = system.component_topology_complete.detach()
    component_atom_ptr = system.component_atom_ptr.detach()
    atom_symbol = system.atom_symbol.detach()
    atom_stereo = system.atom_stereo.detach()
    atomic_number = system.atomic_number.detach()
    isotope = system.isotope.detach()
    isotope_known = system.isotope_known.detach()
    formal_charge = system.formal_charge.detach()
    formal_charge_known = system.formal_charge_known.detach()
    atom_aromatic = system.atom_aromatic.detach()
    atom_aromatic_known = system.atom_aromatic_known.detach()
    reference_coordinates = system.reference_coordinates.detach()
    reference_coordinate_mask = system.reference_coordinate_mask.detach()
    edge_index = system.edge_index.detach()
    bond_order = system.bond_order.detach()
    bond_order_known = system.bond_order_known.detach()
    bond_aromatic = system.bond_aromatic.detach()
    bond_aromatic_known = system.bond_aromatic_known.detach()
    bond_kind = system.bond_kind.detach()
    bond_direction = system.bond_direction.detach()
    bond_role = system.bond_role.detach()
    bond_stereo = system.bond_stereo.detach()
    bond_stereo_index = system.bond_stereo_index.detach()
    element_by_number = {number: element for element, number in ATOMIC_NUMBER_BY_ELEMENT.items()}

    instance_count = int(system.instance_symbol.shape[0])
    component_count = int(component_symbol.shape[0])
    atom_count = int(atom_symbol.shape[0])
    component_instance = [0] * component_count
    atom_component = [0] * atom_count
    for instance_index in range(instance_count):
        start = int(instance_component_ptr[instance_index])
        stop = int(instance_component_ptr[instance_index + 1])
        for component_index in range(start, stop):
            component_instance[component_index] = instance_index
            atom_start = int(component_atom_ptr[component_index])
            atom_stop = int(component_atom_ptr[component_index + 1])
            atom_component[atom_start:atom_stop] = [component_index] * (atom_stop - atom_start)

    label_asym_by_instance: dict[int, str] = {}
    used_asym_ids: set[str] = set()
    for instance_index in range(instance_count):
        label_asym_by_instance[instance_index] = unique_identifier(
            system.instance_id(instance_index),
            used_asym_ids,
        )
    entity_id_by_index: dict[int, str] = {}
    used_entity_ids: set[str] = set()
    for entity_index in range(int(system.entity_symbol.shape[0])):
        entity_id_by_index[entity_index] = unique_identifier(
            system.entity_id(entity_index),
            used_entity_ids,
        )

    def component_id(component_index: int) -> str:
        return system.symbols[int(component_symbol[component_index])]

    def atom_id(atom_index: int) -> str:
        return system.symbols[int(atom_symbol[atom_index])]

    def element(atom_index: int) -> str:
        number = int(atomic_number[atom_index])
        if number == 1 and bool(isotope_known[atom_index]):
            if int(isotope[atom_index]) == 2:
                return "D"
            if int(isotope[atom_index]) == 3:
                return "T"
        return element_by_number[number]

    def source_value(values: torch.Tensor, index: int) -> str | None:
        symbol_index = int(values[index])
        return None if symbol_index < 0 else structure.source_symbols[symbol_index]

    def order_name(index: int) -> str:
        if not bool(bond_order_known[index]):
            return "?"
        if bool(bond_aromatic_known[index]) and bool(bond_aromatic[index]) and float(bond_order[index]) == 1.5:
            return "arom"
        try:
            return {
                1.0: "sing",
                2.0: "doub",
                3.0: "trip",
                4.0: "quad",
            }[float(bond_order[index])]
        except KeyError:
            raise ValueError(f"mmCIF cannot encode known bond order {float(bond_order[index])!r}") from None

    def connection_order_name(index: int) -> str:
        value = order_name(index)
        decoded_order, decoded_aromatic = _cif_bond_order(None if value == "?" else value)
        # Mirror the reader: in `struct_conn` an explicit order also states non-aromaticity.
        if decoded_order is not None and decoded_aromatic is None:
            decoded_aromatic = False
        expected_order = float(bond_order[index]) if bool(bond_order_known[index]) else None
        expected_aromatic = bool(bond_aromatic[index]) if bool(bond_aromatic_known[index]) else None
        if (decoded_order, decoded_aromatic) != (
            expected_order,
            expected_aromatic,
        ):
            raise ValueError(
                "mmCIF struct_conn cannot encode partially known or inconsistent bond order and aromaticity"
            )
        return value

    def component_bond_values(index: int) -> tuple[str, str]:
        order_value = order_name(index)
        expected_order = float(bond_order[index]) if bool(bond_order_known[index]) else None
        expected_aromatic = bool(bond_aromatic[index]) if bool(bond_aromatic_known[index]) else None
        decoded_order, order_aromatic = _cif_bond_order(None if order_value == "?" else order_value)
        if expected_aromatic is None:
            decoded_aromatic = order_aromatic
            aromatic_value = "?"
        else:
            if order_aromatic is not None and order_aromatic != expected_aromatic:
                raise ValueError("mmCIF chem_comp_bond cannot encode inconsistent bond order and aromaticity")
            decoded_aromatic = expected_aromatic
            aromatic_value = "Y" if expected_aromatic else "N"
        if (decoded_order, decoded_aromatic) != (
            expected_order,
            expected_aromatic,
        ):
            raise ValueError("mmCIF chem_comp_bond cannot encode partially known bond order and aromaticity")
        return order_value, aromatic_value

    entity_columns: dict[str, list[object]] = {"id": [], "type": []}
    struct_asym_columns: dict[str, list[object]] = {"id": [], "entity_id": []}
    entity_poly_columns: dict[str, list[object]] = {"entity_id": [], "type": []}
    entity_poly_seq_columns: dict[str, list[object]] = {
        "entity_id": [],
        "num": [],
        "mon_id": [],
        "hetero": [],
    }
    poly_scheme_columns: dict[str, list[object]] = {
        "asym_id": [],
        "entity_id": [],
        "seq_id": [],
        "mon_id": [],
        "ndb_seq_num": [],
        "auth_seq_num": [],
        "pdb_seq_num": [],
        "pdb_mon_id": [],
        "auth_mon_id": [],
        "pdb_strand_id": [],
        "pdb_ins_code": [],
        "hetero": [],
    }
    branch_columns: dict[str, list[object]] = {"entity_id": [], "type": []}
    nonpoly_columns: dict[str, list[object]] = {
        "entity_id": [],
        "name": [],
        "comp_id": [],
    }
    nonpoly_scheme_columns: dict[str, list[object]] = {
        "asym_id": [],
        "entity_id": [],
        "mon_id": [],
        "ndb_seq_num": [],
        "pdb_seq_num": [],
        "auth_seq_num": [],
        "pdb_mon_id": [],
        "auth_mon_id": [],
        "pdb_strand_id": [],
        "pdb_ins_code": [],
    }
    branch_list_columns: dict[str, list[object]] = {
        "entity_id": [],
        "num": [],
        "comp_id": [],
        "hetero": [],
    }
    branch_link_columns: dict[str, list[object]] = {
        "entity_id": [],
        "link_id": [],
        "entity_branch_list_num_1": [],
        "comp_id_1": [],
        "atom_id_1": [],
        "leaving_atom_id_1": [],
        "entity_branch_list_num_2": [],
        "comp_id_2": [],
        "atom_id_2": [],
        "leaving_atom_id_2": [],
        "value_order": [],
    }
    branch_scheme_columns: dict[str, list[object]] = {
        "asym_id": [],
        "entity_id": [],
        "mon_id": [],
        "num": [],
        "pdb_asym_id": [],
        "pdb_mon_id": [],
        "pdb_seq_num": [],
        "pdb_ins_code": [],
        "auth_asym_id": [],
        "auth_mon_id": [],
        "auth_seq_num": [],
    }

    entity_type_name = {
        ENTITY_PROTEIN: "polypeptide(L)",
        ENTITY_RNA: "polyribonucleotide",
        ENTITY_DNA: "polydeoxyribonucleotide",
        ENTITY_DNA_RNA_HYBRID: "polydeoxyribonucleotide/polyribonucleotide hybrid",
        ENTITY_PNA: "peptide nucleic acid",
    }
    first_instance_by_entity: dict[int, int] = {}
    instances_by_entity: dict[int, list[int]] = {}
    branch_leaving_by_endpoint: dict[tuple[int, int], str] = {}
    branch_leaving_by_component: dict[int, frozenset[str]] = {}
    assembly_leaving_by_endpoint: dict[tuple[int, int], tuple[str, frozenset[str]]] = {}
    assembly_leaving_by_component: dict[int, frozenset[str]] = {}
    for instance_index in range(instance_count):
        entity_index = int(instance_entity[instance_index])
        first_instance_by_entity.setdefault(entity_index, instance_index)
        instances_by_entity.setdefault(entity_index, []).append(instance_index)
        struct_asym_columns["id"].append(label_asym_by_instance[instance_index])
        struct_asym_columns["entity_id"].append(entity_id_by_index[entity_index])

    def instance_components(instance_index: int) -> tuple[int, ...]:
        return tuple(
            range(
                int(instance_component_ptr[instance_index]),
                int(instance_component_ptr[instance_index + 1]),
            )
        )

    polymer_pairs = {
        tuple(
            sorted(
                (
                    atom_component[int(edge_index[0, edge])],
                    atom_component[int(edge_index[1, edge])],
                )
            )
        )
        for edge in range(edge_index.shape[1])
        if int(bond_role[edge]) == BOND_ROLE_POLYMER
    }

    def local_atom(instance_index: int, atom_index: int) -> tuple[int, str]:
        component_index = atom_component[atom_index]
        if component_instance[component_index] != instance_index:
            raise ValueError("an entity-owned bond crosses molecular instances")
        return (
            component_index - int(instance_component_ptr[instance_index]),
            atom_id(atom_index),
        )

    def owned_edge_signature(
        instance_index: int,
        role: int,
    ) -> tuple[tuple[object, ...], ...]:
        rows = []
        for edge in range(edge_index.shape[1]):
            if int(bond_role[edge]) != role:
                continue
            atom_a = int(edge_index[0, edge])
            atom_b = int(edge_index[1, edge])
            if component_instance[atom_component[atom_a]] != instance_index:
                continue
            endpoint_a = local_atom(instance_index, atom_a)
            endpoint_b = local_atom(instance_index, atom_b)
            stereo_atoms = tuple(
                (None if int(atom) < 0 else local_atom(instance_index, int(atom)))
                for atom in bond_stereo_index[:, edge]
            )
            direction = int(bond_direction[edge])
            leaving_atoms: tuple[str | None, str | None] | None = None
            if role == BOND_ROLE_ENTITY:
                leaving_atoms = (
                    branch_leaving_by_endpoint.get((edge, 0)),
                    branch_leaving_by_endpoint.get((edge, 1)),
                )
            if endpoint_b < endpoint_a:
                endpoint_a, endpoint_b = endpoint_b, endpoint_a
                stereo_atoms = stereo_atoms[::-1]
                direction = -direction
                if leaving_atoms is not None:
                    leaving_atoms = leaving_atoms[::-1]
            stereo = int(bond_stereo[edge])
            row = (
                endpoint_a,
                endpoint_b,
                (
                    bool(bond_order_known[edge]),
                    float(bond_order[edge]),
                ),
                (
                    bool(bond_aromatic_known[edge]),
                    bool(bond_aromatic[edge]),
                ),
                int(bond_kind[edge]),
                direction,
                None if stereo < 0 else system.symbols[stereo],
                stereo_atoms,
            )
            rows.append(row if leaving_atoms is None else (*row, leaving_atoms))
        return tuple(sorted(rows))

    for edge in range(edge_index.shape[1]):
        role = int(bond_role[edge])
        atom_a = int(edge_index[0, edge])
        atom_b = int(edge_index[1, edge])
        component_a = atom_component[atom_a]
        component_b = atom_component[atom_b]
        instance_a = component_instance[component_a]
        instance_b = component_instance[component_b]
        if role == BOND_ROLE_COMPONENT and component_a != component_b:
            raise ValueError("a component-owned bond crosses component occurrences")
        if role in {BOND_ROLE_POLYMER, BOND_ROLE_ENTITY} and instance_a != instance_b:
            raise ValueError("an entity-owned bond crosses molecular instances")
        if role == BOND_ROLE_POLYMER and component_a == component_b:
            raise ValueError("a polymer-owned bond must join component occurrences")
        if role == BOND_ROLE_ENTITY and component_a == component_b:
            raise ValueError("an entity-owned bond must join component occurrences")
        if role == BOND_ROLE_ENTITY:
            entity_index = int(instance_entity[instance_a])
            if (
                int(entity_type[entity_index]) != ENTITY_MOLECULE
                or int(entity_molecule_type[entity_index]) != MOLECULE_BRANCHED
            ):
                raise ValueError("only a branched molecule can carry entity-owned bonds")

    branch_endpoints_by_component: dict[int, list[tuple[int, int, str]]] = {}
    for edge in range(edge_index.shape[1]):
        if int(bond_role[edge]) != BOND_ROLE_ENTITY:
            continue
        for side in range(2):
            atom = int(edge_index[side, edge])
            branch_endpoints_by_component.setdefault(
                atom_component[atom],
                [],
            ).append((edge, side, atom_id(atom)))

    for component_index, endpoints in branch_endpoints_by_component.items():
        try:
            component = get_component(component_id(component_index))
        except KeyError:
            continue
        if component.bonds is None:
            raise ValueError(f"component {component.component_id!r} has no known bond graph")
        start = int(component_atom_ptr[component_index])
        stop = int(component_atom_ptr[component_index + 1])
        present = {atom_id(atom) for atom in range(start, stop)}
        missing = {atom.atom_id for atom in component.atoms} - present
        selected: set[str] = set()
        groups = _component_atom_groups(component, missing)
        for edge, side, endpoint in endpoints:
            candidates = [
                group
                for group, boundary in groups
                if boundary == frozenset({endpoint}) and not group.intersection(selected)
            ]
            if len(candidates) > 1:
                raise ValueError(f"mmCIF cannot infer a unique leaving group for {component.component_id!r}:{endpoint}")
            if candidates:
                group = candidates[0]
                selected.update(group)
                branch_leaving_by_endpoint[(edge, side)] = _component_leaving_atom(component, group, endpoint)
        branch_leaving_by_component[component_index] = frozenset(selected)

    assembly_endpoints_by_component: dict[int, list[tuple[int, int, str]]] = {}
    for edge in range(edge_index.shape[1]):
        if int(bond_role[edge]) != BOND_ROLE_ASSEMBLY:
            continue
        for side in range(2):
            atom = int(edge_index[side, edge])
            assembly_endpoints_by_component.setdefault(atom_component[atom], []).append((edge, side, atom_id(atom)))

    for component_index, endpoints in assembly_endpoints_by_component.items():
        try:
            component = get_component(component_id(component_index))
        except KeyError:
            continue
        instance_index = component_instance[component_index]
        component_start = int(instance_component_ptr[instance_index])
        component_stop = int(instance_component_ptr[instance_index + 1])
        ordinary_leaving = set(branch_leaving_by_component.get(component_index, ()))
        if (
            component_index > component_start
            and (component_index - 1, component_index) in polymer_pairs
            and component.previous_attachment is not None
        ):
            ordinary_leaving.update(component.previous_attachment.leaving_atom_ids)
        if (
            component_index + 1 < component_stop
            and (component_index, component_index + 1) in polymer_pairs
            and component.next_attachment is not None
        ):
            ordinary_leaving.update(component.next_attachment.leaving_atom_ids)
        ordinary_product = _remove_component_atoms(component, ordinary_leaving)
        start = int(component_atom_ptr[component_index])
        stop = int(component_atom_ptr[component_index + 1])
        present = {atom_id(atom) for atom in range(start, stop)}
        missing = {atom.atom_id for atom in ordinary_product.atoms} - present
        groups = _component_atom_groups(ordinary_product, missing)
        assembly_selected: set[str] = set()
        for edge, side, endpoint in endpoints:
            candidates = [
                group
                for group, boundary in groups
                if boundary == frozenset({endpoint}) and not group.intersection(assembly_selected)
            ]
            if len(candidates) > 1:
                raise ValueError(f"mmCIF cannot infer a unique leaving group for {component.component_id!r}:{endpoint}")
            if candidates:
                group = candidates[0]
                assembly_selected.update(group)
                assembly_leaving_by_endpoint[(edge, side)] = (
                    _component_leaving_atom(ordinary_product, group, endpoint),
                    group,
                )
        assembly_leaving_by_component[component_index] = frozenset(assembly_selected)

    for entity_index in range(int(system.entity_symbol.shape[0])):
        entity_id = entity_id_by_index[entity_index]
        type_code = int(entity_type[entity_index])
        first_instance = first_instance_by_entity[entity_index]
        components = list(instance_components(first_instance))
        component_start = components[0] if components else int(instance_component_ptr[first_instance])
        component_ids_by_position: list[tuple[str, ...]] = []
        for instance_index in instances_by_entity[entity_index]:
            if len(instance_components(instance_index)) != len(components):
                raise ValueError(f"mmCIF entity {entity_id!r} has inconsistent component-axis lengths")
        for local_index in range(len(components)):
            component_ids = tuple(
                dict.fromkeys(
                    component_id(instance_components(instance_index)[local_index])
                    for instance_index in instances_by_entity[entity_index]
                )
            )
            component_ids_by_position.append(component_ids)
        if type_code == ENTITY_MOLECULE:
            if any(len(component_ids) != 1 for component_ids in component_ids_by_position):
                raise ValueError(f"mmCIF cannot encode divergent molecule instances for entity {entity_id!r}")
            first_entity_edges = owned_edge_signature(
                first_instance,
                BOND_ROLE_ENTITY,
            )
            if any(
                owned_edge_signature(instance_index, BOND_ROLE_ENTITY) != first_entity_edges
                for instance_index in instances_by_entity[entity_index][1:]
            ):
                raise ValueError(f"mmCIF cannot encode divergent entity bonds for entity {entity_id!r}")
            if any(
                owned_edge_signature(instance_index, BOND_ROLE_POLYMER)
                for instance_index in instances_by_entity[entity_index]
            ):
                raise ValueError("a molecule cannot carry polymer-owned bonds")
            molecule_type = int(entity_molecule_type[entity_index])
            if molecule_type == MOLECULE_BRANCHED:
                entity_category = "branched"
            elif molecule_type in {
                MOLECULE_MACROLIDE,
                MOLECULE_NON_POLYMER,
                MOLECULE_WATER,
            }:
                if molecule_type == MOLECULE_MACROLIDE:
                    entity_category = "macrolide"
                elif molecule_type == MOLECULE_WATER:
                    entity_category = "water"
                else:
                    entity_category = "non-polymer"
                if len(components) != 1 or first_entity_edges:
                    raise ValueError(
                        f"mmCIF {entity_category} entities require one component and no entity-owned bonds"
                    )
            else:
                raise ValueError(f"unsupported molecule type code {molecule_type}")
        else:
            if any(
                owned_edge_signature(instance_index, BOND_ROLE_ENTITY)
                for instance_index in instances_by_entity[entity_index]
            ):
                raise ValueError("a polymer cannot carry entity-owned bonds")
            for instance_index in instances_by_entity[entity_index]:
                expected_polymer_edges = []
                occurrence_components = instance_components(instance_index)
                for local_index, (left_index, right_index) in enumerate(
                    zip(occurrence_components, occurrence_components[1:])
                ):
                    try:
                        left = get_component(component_id(left_index))
                        right = get_component(component_id(right_index))
                    except KeyError:
                        continue
                    left_start = int(component_atom_ptr[left_index])
                    left_stop = int(component_atom_ptr[left_index + 1])
                    right_start = int(component_atom_ptr[right_index])
                    right_stop = int(component_atom_ptr[right_index + 1])
                    left_attachment = left.next_attachment
                    right_attachment = right.previous_attachment
                    if left_attachment is None or right_attachment is None:
                        continue
                    observed_left = {
                        atom_id(atom_index)
                        for atom_index in range(left_start, left_stop)
                        if bool(coordinate_mask[atom_index])
                    }
                    observed_right = {
                        atom_id(atom_index)
                        for atom_index in range(right_start, right_stop)
                        if bool(coordinate_mask[atom_index])
                    }
                    if (
                        left_attachment.leaving_atom_ids & observed_left
                        or right_attachment.leaving_atom_ids & observed_right
                    ):
                        continue
                    endpoint_a = (local_index, left_attachment.atom_id)
                    endpoint_b = (
                        local_index + 1,
                        right_attachment.atom_id,
                    )
                    expected_polymer_edges.append(
                        (
                            endpoint_a,
                            endpoint_b,
                            (True, 1.0),
                            (True, False),
                            BOND_KIND_COVALENT,
                            BOND_DIRECTION_UNDIRECTED,
                            None,
                            (None, None),
                        )
                    )
                if owned_edge_signature(
                    instance_index,
                    BOND_ROLE_POLYMER,
                ) != tuple(expected_polymer_edges):
                    raise ValueError(f"mmCIF cannot reconstruct polymer bonds for entity {entity_id!r}")
            entity_category = "polymer"
        entity_columns["id"].append(entity_id)
        entity_columns["type"].append(entity_category)

        if type_code != ENTITY_MOLECULE:
            entity_poly_columns["entity_id"].append(entity_id)
            entity_poly_columns["type"].append(entity_type_name[type_code])
            for local_index, candidate_ids in enumerate(
                component_ids_by_position,
                1,
            ):
                if len(candidate_ids) > 1:
                    for instance_index in instances_by_entity[entity_index]:
                        occurrence = instance_components(instance_index)[local_index - 1]
                        atom_start = int(component_atom_ptr[occurrence])
                        atom_stop = int(component_atom_ptr[occurrence + 1])
                        if not bool(coordinate_mask[atom_start:atom_stop].any()):
                            raise ValueError("mmCIF cannot select an unobserved microheterogeneous polymer component")
                for candidate in candidate_ids:
                    entity_poly_seq_columns["entity_id"].append(entity_id)
                    entity_poly_seq_columns["num"].append(local_index)
                    entity_poly_seq_columns["mon_id"].append(candidate)
                    entity_poly_seq_columns["hetero"].append("y" if len(candidate_ids) > 1 else "n")
            for instance_index in range(instance_count):
                if int(instance_entity[instance_index]) != entity_index:
                    continue
                instance_component_start = int(instance_component_ptr[instance_index])
                for local_index in range(len(components)):
                    component_index = instance_component_start + local_index
                    author_number = source_value(residue_id, component_index)
                    monomer = component_id(component_index)
                    asym_id = label_asym_by_instance[instance_index]
                    poly_scheme_values = (
                        asym_id,
                        entity_id,
                        local_index + 1,
                        monomer,
                        local_index + 1,
                        author_number or "?",
                        author_number or "?",
                        monomer,
                        monomer,
                        asym_id,
                        source_value(insertion_code, component_index) or "?",
                        ("y" if len(component_ids_by_position[local_index]) > 1 else "n"),
                    )
                    for name, value in zip(poly_scheme_columns, poly_scheme_values):
                        poly_scheme_columns[name].append(value)
        elif int(entity_molecule_type[entity_index]) == MOLECULE_BRANCHED:
            branch_columns["entity_id"].append(entity_id)
            branch_columns["type"].append("oligosaccharide")
            for local_index, component_index in enumerate(components, 1):
                branch_list_columns["entity_id"].append(entity_id)
                branch_list_columns["num"].append(local_index)
                branch_list_columns["comp_id"].append(component_id(component_index))
                branch_list_columns["hetero"].append("n")

            link_id = 1
            for bond_index in range(edge_index.shape[1]):
                if int(bond_role[bond_index]) != BOND_ROLE_ENTITY:
                    continue
                atom_a = int(edge_index[0, bond_index])
                atom_b = int(edge_index[1, bond_index])
                component_a = atom_component[atom_a]
                component_b = atom_component[atom_b]
                if (
                    component_instance[component_a] != first_instance
                    or component_instance[component_b] != first_instance
                ):
                    continue
                local_a = component_a - component_start
                local_b = component_b - component_start
                if local_a == local_b:
                    continue
                branch_link_values = (
                    entity_id,
                    link_id,
                    local_a + 1,
                    component_id(component_a),
                    atom_id(atom_a),
                    branch_leaving_by_endpoint.get((bond_index, 0), "?"),
                    local_b + 1,
                    component_id(component_b),
                    atom_id(atom_b),
                    branch_leaving_by_endpoint.get((bond_index, 1), "?"),
                    connection_order_name(bond_index),
                )
                for name, value in zip(branch_link_columns, branch_link_values):
                    branch_link_columns[name].append(value)
                link_id += 1

            for instance_index in range(instance_count):
                if int(instance_entity[instance_index]) != entity_index:
                    continue
                instance_component_start = int(instance_component_ptr[instance_index])
                asym_id = label_asym_by_instance[instance_index]
                monomer_counts = {
                    component_id(index): sum(component_id(other) == component_id(index) for other in components)
                    for index in components
                }
                for local_index in range(len(components)):
                    component_index = instance_component_start + local_index
                    source_author_number = source_value(
                        residue_id,
                        component_index,
                    )
                    monomer = component_id(component_index)
                    atom_start = int(component_atom_ptr[component_index])
                    atom_stop = int(component_atom_ptr[component_index + 1])
                    if (
                        source_author_number is None
                        and monomer_counts[monomer] > 1
                        and bool(coordinate_mask[atom_start:atom_stop].any())
                    ):
                        raise ValueError(
                            "mmCIF cannot identify an observed repeated "
                            "branched-entity monomer without an author "
                            "sequence number"
                        )
                    auth_number = source_author_number or "?"
                    branch_scheme_values = (
                        asym_id,
                        entity_id,
                        monomer,
                        local_index + 1,
                        asym_id,
                        monomer,
                        auth_number,
                        source_value(insertion_code, component_index) or "?",
                        asym_id,
                        monomer,
                        auth_number,
                    )
                    for name, value in zip(branch_scheme_columns, branch_scheme_values):
                        branch_scheme_columns[name].append(value)
        else:
            monomer = component_id(components[0])
            nonpoly_columns["entity_id"].append(entity_id)
            nonpoly_columns["name"].append(monomer)
            nonpoly_columns["comp_id"].append(monomer)
            for instance_index in range(instance_count):
                if int(instance_entity[instance_index]) != entity_index:
                    continue
                component_index = int(instance_component_ptr[instance_index])
                asym_id = label_asym_by_instance[instance_index]
                author_number = source_value(residue_id, component_index) or "?"
                nonpoly_scheme_values = (
                    asym_id,
                    entity_id,
                    monomer,
                    1,
                    author_number,
                    author_number,
                    monomer,
                    monomer,
                    asym_id,
                    source_value(insertion_code, component_index) or "?",
                )
                for name, value in zip(nonpoly_scheme_columns, nonpoly_scheme_values):
                    nonpoly_scheme_columns[name].append(value)

    chem_comp_columns: dict[str, list[object]] = {
        "id": [],
        "type": [],
        "mon_nstd_parent_comp_id": [],
    }
    chem_atom_columns: dict[str, list[object]] = {
        "comp_id": [],
        "atom_id": [],
        "type_symbol": [],
        "charge": [],
        "pdbx_aromatic_flag": [],
        "pdbx_stereo_config": [],
        "model_Cartn_x": [],
        "model_Cartn_y": [],
        "model_Cartn_z": [],
    }
    chem_bond_columns: dict[str, list[object]] = {
        "comp_id": [],
        "atom_id_1": [],
        "atom_id_2": [],
        "value_order": [],
        "pdbx_aromatic_flag": [],
        "pdbx_stereo_config": [],
    }
    component_bond_indices: dict[int, list[int]] = {}
    for bond_index in range(edge_index.shape[1]):
        if int(bond_role[bond_index]) != BOND_ROLE_COMPONENT:
            continue
        atom_a = int(edge_index[0, bond_index])
        atom_b = int(edge_index[1, bond_index])
        component_index = atom_component[atom_a]
        if atom_component[atom_b] == component_index:
            component_bond_indices.setdefault(component_index, []).append(bond_index)

    def component_definition(index: int) -> tuple[object, ...]:
        start = int(component_atom_ptr[index])
        stop = int(component_atom_ptr[index + 1])
        atoms = tuple(
            (
                atom_id(atom_index),
                element(atom_index),
                (int(formal_charge[atom_index]) if bool(formal_charge_known[atom_index]) else None),
                (bool(atom_aromatic[atom_index]) if bool(atom_aromatic_known[atom_index]) else None),
                (system.symbols[int(atom_stereo[atom_index])] if int(atom_stereo[atom_index]) >= 0 else None),
                (
                    tuple(float(value) for value in reference_coordinates[atom_index])
                    if bool(reference_coordinate_mask[atom_index])
                    else None
                ),
            )
            for atom_index in range(start, stop)
        )
        bonds = tuple(
            (
                atom_id(int(edge_index[0, bond_index])),
                atom_id(int(edge_index[1, bond_index])),
                *component_bond_values(bond_index),
                (system.symbols[int(bond_stereo[bond_index])] if int(bond_stereo[bond_index]) >= 0 else None),
            )
            for bond_index in component_bond_indices.get(index, ())
        )
        return bool(component_topology_complete[index]), atoms, bonds

    def known_component_signatures(
        component_index: int,
        expected: tuple[object, ...],
    ) -> tuple[tuple[object, ...], tuple[object, ...]]:
        actual = component_signatures[component_index]
        actual_atoms = cast(tuple[tuple[object, ...], ...], actual[1])
        expected_atoms = cast(tuple[tuple[object, ...], ...], expected[1])
        if len(actual_atoms) != len(expected_atoms):
            return actual, expected

        normalized_atoms = []
        start = int(component_atom_ptr[component_index])
        for offset, (actual_atom, expected_atom) in enumerate(zip(actual_atoms, expected_atoms)):
            normalized = list(expected_atom)
            if actual_atom[:2] == expected_atom[:2]:
                observed = bool(coordinate_mask[start + offset])
                if actual_atom[2] != expected_atom[2]:
                    if not observed or actual_atom[1] != 1 or actual_atom[2] not in {2, 3}:
                        raise ValueError("mmCIF cannot encode this isotope difference for a known component")
                    normalized[2] = actual_atom[2]
                if actual_atom[3] != expected_atom[3]:
                    if not observed or actual_atom[3] is None:
                        raise ValueError("mmCIF cannot encode this formal-charge difference for a known component")
                    normalized[3] = actual_atom[3]
            normalized_atoms.append(tuple(normalized))
        return actual, (expected[0], tuple(normalized_atoms), expected[2])

    components_by_id: dict[str, list[int]] = {}
    for component_index in range(component_count):
        components_by_id.setdefault(component_id(component_index), []).append(component_index)
    chem_comp_type_by_id: dict[str, str] = {}
    for name, component_indices in components_by_id.items():
        polymer_type = get_builtin_component_polymer_type(name)
        component_types: set[str]
        if polymer_type == "protein":
            component_types = {"L-peptide linking"}
        elif polymer_type == "rna":
            component_types = {"RNA linking"}
        elif polymer_type == "dna":
            component_types = {"DNA linking"}
        else:
            component_types = set()
            for component_index in component_indices:
                instance_index = component_instance[component_index]
                entity_index = int(instance_entity[instance_index])
                type_code = int(entity_type[entity_index])
                if type_code == ENTITY_PROTEIN:
                    component_types.add("L-peptide linking")
                elif type_code == ENTITY_RNA:
                    component_types.add("RNA linking")
                elif type_code == ENTITY_DNA:
                    component_types.add("DNA linking")
                elif type_code == ENTITY_MOLECULE:
                    molecule_type = int(entity_molecule_type[entity_index])
                    component_types.add("saccharide" if molecule_type == MOLECULE_BRANCHED else "non-polymer")
                else:
                    component_types.add("other")
        chem_comp_type = next(iter(component_types)) if len(component_types) == 1 else "other"
        chem_comp_type_by_id[name] = chem_comp_type
        chem_comp_columns["id"].append(name)
        chem_comp_columns["type"].append(chem_comp_type)
        parent_ids = {
            structure.source_symbols[int(standard_component_id[index])]
            for index in component_indices
            if int(standard_component_id[index]) >= 0
        }
        if parent_ids and any(int(standard_component_id[index]) < 0 for index in component_indices):
            raise ValueError(f"component {name!r} has standard-parent metadata on only some occurrences")
        if len(parent_ids) > 1:
            raise ValueError(f"component {name!r} has conflicting standard parent IDs")
        chem_comp_columns["mon_nstd_parent_comp_id"].append(next(iter(parent_ids), "?"))
    embedded_component_by_id: dict[str, int] = {}
    component_signatures = _graph_component_signatures(system)
    polymer_types: dict[int, PolymerType] = {
        ENTITY_PROTEIN: "protein",
        ENTITY_RNA: "rna",
        ENTITY_DNA: "dna",
        ENTITY_DNA_RNA_HYBRID: "dna_rna_hybrid",
        ENTITY_PNA: "pna",
    }

    expected_definitions: dict[tuple[str, int, bool, bool, tuple[str, ...]], tuple[object, ...]] = {}
    for name, component_indices in components_by_id.items():
        try:
            authoritative = get_component(name)
        except KeyError:
            first = component_indices[0]
            definition = component_definition(first)
            if any(component_definition(component_index) != definition for component_index in component_indices[1:]):
                raise ValueError(f"mmCIF cannot encode conflicting realized definitions for component ID {name!r}")
            embedded_component_by_id[name] = first
            continue

        for component_index in component_indices:
            instance_index = component_instance[component_index]
            component_start = int(instance_component_ptr[instance_index])
            component_stop = int(instance_component_ptr[instance_index + 1])
            has_previous = component_index > component_start and (component_index - 1, component_index) in polymer_pairs
            has_next = component_index + 1 < component_stop and (component_index, component_index + 1) in polymer_pairs
            entity_index = int(instance_entity[instance_index])
            type_code = int(entity_type[entity_index])
            leaving_atoms = tuple(
                sorted(
                    branch_leaving_by_component.get(component_index, frozenset())
                    | assembly_leaving_by_component.get(component_index, frozenset())
                )
            )
            key = (
                name,
                type_code,
                has_previous,
                has_next,
                leaving_atoms,
            )
            expected = expected_definitions.get(key)
            if expected is None:
                expected_component = authoritative
                if leaving_atoms:
                    expected_component = _remove_component_atoms(
                        expected_component,
                        set(leaving_atoms),
                    )
                expected = _realized_component_signature(
                    expected_component,
                    None if type_code == ENTITY_MOLECULE else polymer_types[type_code],
                    has_previous=has_previous,
                    has_next=has_next,
                )
                expected_definitions[key] = expected
            actual, normalized_expected = known_component_signatures(
                component_index,
                expected,
            )
            if actual != normalized_expected:
                raise ValueError(f"component ID {name!r} does not match its authoritative definition")

    for name in embedded_component_by_id:
        if chem_comp_type_by_id[name] == "other" and any(
            name_component in pair for name_component in components_by_id[name] for pair in polymer_pairs
        ):
            raise ValueError(
                f"mmCIF cannot reconstruct polymer attachments for component {name!r} with chem_comp.type other"
            )
        for component_index in components_by_id[name]:
            start = int(component_atom_ptr[component_index])
            stop = int(component_atom_ptr[component_index + 1])
            decoded_topology_complete = stop - start == 1 or bool(component_bond_indices.get(component_index))
            if bool(component_topology_complete[component_index]) != decoded_topology_complete:
                raise ValueError("mmCIF cannot encode this unknown component's topology completeness")
            isotope_representable = (atomic_number[start:stop] == 1) & torch.isin(
                isotope[start:stop],
                isotope.new_tensor((2, 3)),
            )
            if bool((isotope_known[start:stop] & ~isotope_representable).any()):
                raise ValueError(
                    "mmCIF cannot encode explicit isotopes other than deuterium and tritium for an unknown component"
                )
            for field in (
                "radical_electrons_known",
                "explicit_hydrogens_known",
                "implicit_hydrogens_known",
                "implicit_hydrogens_allowed_known",
            ):
                if bool(getattr(system, field)[start:stop].any()):
                    raise ValueError(
                        f"mmCIF cannot encode known {field.removesuffix('_known')} for an unknown component"
                    )
            for edge in component_bond_indices.get(component_index, ()):
                if (
                    int(bond_kind[edge]) != BOND_KIND_COVALENT
                    or int(bond_direction[edge]) != BOND_DIRECTION_UNDIRECTED
                    or bool((system.bond_stereo_index[:, edge] >= 0).any())
                ):
                    raise ValueError(
                        "mmCIF chem_comp_bond cannot encode directed, dative, or stereo-defining component bonds"
                    )

    for edge in range(edge_index.shape[1]):
        role = int(bond_role[edge])
        if role == BOND_ROLE_ENTITY and (
            int(bond_kind[edge]) != BOND_KIND_COVALENT
            or int(bond_direction[edge]) != BOND_DIRECTION_UNDIRECTED
            or int(bond_stereo[edge]) >= 0
            or bool((system.bond_stereo_index[:, edge] >= 0).any())
        ):
            raise ValueError("mmCIF branched-entity links cannot encode directed, dative, or stereochemical bonds")
        if role >= BOND_ROLE_POLYMER and (
            int(bond_stereo[edge]) >= 0 or bool((system.bond_stereo_index[:, edge] >= 0).any())
        ):
            raise ValueError("mmCIF struct_conn cannot encode bond stereochemistry")
    for name, component_index in embedded_component_by_id.items():
        atom_start = int(component_atom_ptr[component_index])
        atom_stop = int(component_atom_ptr[component_index + 1])
        for atom_index in range(atom_start, atom_stop):
            reference = reference_coordinates[atom_index]
            chem_atom_values = (
                name,
                atom_id(atom_index),
                element(atom_index),
                int(formal_charge[atom_index]) if bool(formal_charge_known[atom_index]) else "?",
                ("?" if not bool(atom_aromatic_known[atom_index]) else "Y" if bool(atom_aromatic[atom_index]) else "N"),
                (system.symbols[int(atom_stereo[atom_index])] if int(atom_stereo[atom_index]) >= 0 else "N"),
                float(reference[0]) if bool(reference_coordinate_mask[atom_index]) else "?",
                float(reference[1]) if bool(reference_coordinate_mask[atom_index]) else "?",
                float(reference[2]) if bool(reference_coordinate_mask[atom_index]) else "?",
            )
            for column, value in zip(chem_atom_columns, chem_atom_values):
                chem_atom_columns[column].append(value)

    emitted_component_bonds: set[tuple[str, str, str]] = set()
    for bond_index in range(edge_index.shape[1]):
        if int(bond_role[bond_index]) != BOND_ROLE_COMPONENT:
            continue
        atom_a = int(edge_index[0, bond_index])
        atom_b = int(edge_index[1, bond_index])
        component_index = atom_component[atom_a]
        if atom_component[atom_b] != component_index:
            continue
        name = component_id(component_index)
        if embedded_component_by_id.get(name) != component_index:
            continue
        component_bond_key = (name, atom_id(atom_a), atom_id(atom_b))
        if component_bond_key in emitted_component_bonds:
            continue
        emitted_component_bonds.add(component_bond_key)
        order_value, aromatic_value = component_bond_values(bond_index)
        chem_bond_values = (
            name,
            atom_id(atom_a),
            atom_id(atom_b),
            order_value,
            aromatic_value,
            (system.symbols[int(bond_stereo[bond_index])] if int(bond_stereo[bond_index]) >= 0 else "N"),
        )
        for column, value in zip(chem_bond_columns, chem_bond_values):
            chem_bond_columns[column].append(value)

    atom_site_columns: dict[str, list[object]] = {
        "group_PDB": [],
        "id": [],
        "type_symbol": [],
        "label_atom_id": [],
        "label_alt_id": [],
        "label_comp_id": [],
        "label_asym_id": [],
        "label_entity_id": [],
        "label_seq_id": [],
        "pdbx_PDB_ins_code": [],
        "Cartn_x": [],
        "Cartn_y": [],
        "Cartn_z": [],
        "occupancy": [],
        "B_iso_or_equiv": [],
        "pdbx_formal_charge": [],
        "auth_seq_id": [],
        "auth_comp_id": [],
        "auth_asym_id": [],
        "auth_atom_id": [],
        "pdbx_PDB_model_num": [],
    }
    provenance_columns: dict[str, list[object]] = {
        "label_asym_id": [],
        "component_index": [],
        "label_atom_id": [],
    }
    endpoint_identity: dict[int, tuple[object, ...]] = {}
    serial = 1
    for atom_index in range(atom_count):
        component_index = atom_component[atom_index]
        instance_index = component_instance[component_index]
        entity_index = int(instance_entity[instance_index])
        local_component = component_index - int(instance_component_ptr[instance_index])
        polymer = int(entity_type[entity_index]) != ENTITY_MOLECULE
        asym_id = label_asym_by_instance[instance_index]
        monomer = component_id(component_index)
        source_author_number = source_value(residue_id, component_index)
        author_number = source_author_number or "?"
        code = source_value(insertion_code, component_index)
        label_sequence: object = local_component + 1 if polymer else "."
        endpoint_identity[atom_index] = (
            asym_id,
            label_sequence,
            monomer,
            atom_id(atom_index),
            asym_id,
            author_number,
            monomer,
            atom_id(atom_index),
            code or "?",
        )
        if bool(source_formal_charge_mask[atom_index]):
            provenance_values = (asym_id, local_component, atom_id(atom_index))
            for column, value in zip(provenance_columns, provenance_values, strict=True):
                provenance_columns[column].append(value)
        if not bool(coordinate_mask[atom_index]):
            continue
        xyz = coordinates[atom_index]
        if not bool(torch.isfinite(xyz).all()):
            raise ValueError("mmCIF cannot represent non-finite observed coordinates")
        if bool(occupancy_mask[atom_index]):
            atom_occupancy: float | str = float(occupancy[atom_index])
            if not np.isfinite(atom_occupancy):
                raise ValueError("mmCIF cannot represent non-finite occupancy")
        else:
            atom_occupancy = "."
        if bool(b_factor_mask[atom_index]):
            atom_b_factor: float | str = float(b_factor[atom_index])
            if not np.isfinite(atom_b_factor):
                raise ValueError("mmCIF cannot represent non-finite B factor")
        else:
            atom_b_factor = "."
        atom_site_values = (
            "ATOM" if polymer else "HETATM",
            serial,
            element(atom_index),
            atom_id(atom_index),
            ".",
            monomer,
            asym_id,
            entity_id_by_index[entity_index],
            label_sequence,
            code or "?",
            float(xyz[0]),
            float(xyz[1]),
            float(xyz[2]),
            atom_occupancy,
            atom_b_factor,
            int(formal_charge[atom_index]) if bool(formal_charge_known[atom_index]) else "?",
            author_number,
            monomer,
            asym_id,
            atom_id(atom_index),
            1,
        )
        for column, value in zip(atom_site_columns, atom_site_values):
            atom_site_columns[column].append(value)
        serial += 1

    conn_columns: dict[str, list[object]] = {
        "id": [],
        "conn_type_id": [],
        "pdbx_leaving_atom_flag": [],
        "pdbx_ptnr1_leaving_atom_id": [],
        "pdbx_ptnr2_leaving_atom_id": [],
        "ptnr1_label_asym_id": [],
        "ptnr1_label_seq_id": [],
        "ptnr1_label_comp_id": [],
        "ptnr1_label_atom_id": [],
        "ptnr2_label_asym_id": [],
        "ptnr2_label_seq_id": [],
        "ptnr2_label_comp_id": [],
        "ptnr2_label_atom_id": [],
        "ptnr1_auth_asym_id": [],
        "ptnr1_auth_seq_id": [],
        "ptnr1_auth_comp_id": [],
        "ptnr1_auth_atom_id": [],
        "ptnr2_auth_asym_id": [],
        "ptnr2_auth_seq_id": [],
        "ptnr2_auth_comp_id": [],
        "ptnr2_auth_atom_id": [],
        "pdbx_ptnr1_PDB_ins_code": [],
        "pdbx_ptnr2_PDB_ins_code": [],
        "ptnr1_role": [],
        "ptnr2_role": [],
        "ptnr1_symmetry": [],
        "ptnr2_symmetry": [],
        "pdbx_value_order": [],
    }
    connection_id = 1
    for bond_index in range(edge_index.shape[1]):
        if int(bond_role[bond_index]) != BOND_ROLE_ASSEMBLY:
            continue
        atom_a = int(edge_index[0, bond_index])
        atom_b = int(edge_index[1, bond_index])
        kind = int(bond_kind[bond_index])
        direction = int(bond_direction[bond_index])
        if kind == BOND_KIND_DATIVE:
            if direction == BOND_DIRECTION_ENDPOINT_0_DONOR:
                roles = ("donor", "acceptor")
            elif direction == BOND_DIRECTION_ENDPOINT_1_DONOR:
                roles = ("acceptor", "donor")
            else:
                raise ValueError("mmCIF struct_conn requires a donor direction for a dative bond")
            conn_type = "metalc"
        elif kind == BOND_KIND_COVALENT:
            if direction != BOND_DIRECTION_UNDIRECTED:
                raise ValueError("mmCIF struct_conn cannot encode a directed covalent bond")
            roles = ("?", "?")
            conn_type = "covale"
        else:
            raise ValueError(f"unknown canonical bond kind {kind}")
        value_order = connection_order_name(bond_index)
        if value_order == "arom":
            raise ValueError("mmCIF struct_conn pdbx_value_order does not support aromatic bonds")
        first_endpoint = endpoint_identity[atom_a]
        second_endpoint = endpoint_identity[atom_b]
        first_leaving = assembly_leaving_by_endpoint.get((bond_index, 0))
        second_leaving = assembly_leaving_by_endpoint.get((bond_index, 1))
        leaving_count = int(first_leaving is not None) + int(second_leaving is not None)
        leaving_flag = ("none", "one", "both")[leaving_count]
        connection_values = (
            connection_id,
            conn_type,
            leaving_flag,
            "?" if first_leaving is None else first_leaving[0],
            "?" if second_leaving is None else second_leaving[0],
            *first_endpoint[:4],
            *second_endpoint[:4],
            *first_endpoint[4:8],
            *second_endpoint[4:8],
            first_endpoint[8],
            second_endpoint[8],
            *roles,
            "1_555",
            "1_555",
            value_order,
        )
        for column, value in zip(conn_columns, connection_values):
            conn_columns[column].append(value)
        connection_id += 1

    categories: dict[str, CIFCategory | str] = {
        "entity": CIFCategory(entity_columns),
        "struct_asym": CIFCategory(struct_asym_columns),
        "ofoldx_provenance": CIFCategory({"formal_charge_source_complete": ["Y"]}),
    }
    if provenance_columns["label_asym_id"]:
        categories["ofoldx_source_formal_charge"] = CIFCategory(provenance_columns)
    resolution_rows: dict[int, list[float]] = {}
    for value, kind in zip(resolution.tolist(), resolution_kind.tolist(), strict=True):
        if kind not in {
            RESOLUTION_KIND_REFINEMENT,
            RESOLUTION_KIND_REFLECTIONS,
            RESOLUTION_KIND_EM_RECONSTRUCTION,
        }:
            raise ValueError(f"mmCIF cannot encode experimental-resolution kind {kind}")
        resolution_rows.setdefault(kind, []).append(value)
    if resolution_rows:
        categories["entry"] = CIFCategory({"id": ["ofoldx"]})
    refinement = resolution_rows.get(RESOLUTION_KIND_REFINEMENT, [])
    if refinement:
        categories["refine"] = CIFCategory(
            {
                "entry_id": ["ofoldx"] * len(refinement),
                "pdbx_refine_id": [f"OFOLDX {index}" for index in range(1, len(refinement) + 1)],
                "ls_d_res_high": refinement,
            }
        )
    reflections = resolution_rows.get(RESOLUTION_KIND_REFLECTIONS, [])
    if reflections:
        categories["reflns"] = CIFCategory(
            {
                "pdbx_ordinal": list(range(1, len(reflections) + 1)),
                "entry_id": ["ofoldx"] * len(reflections),
                "d_resolution_high": reflections,
            }
        )
    reconstructions = resolution_rows.get(RESOLUTION_KIND_EM_RECONSTRUCTION, [])
    if reconstructions:
        categories["em_3d_reconstruction"] = CIFCategory(
            {
                "id": list(range(1, len(reconstructions) + 1)),
                "entry_id": ["ofoldx"] * len(reconstructions),
                "resolution": reconstructions,
            }
        )
    # A structure that resolves no atom still states its chemistry through the declaration
    # categories below, and mmCIF writes an absent loop rather than an empty one.
    if tuple(atom_site_columns.values())[0]:
        categories["atom_site"] = CIFCategory(atom_site_columns)
    for name, columns in (
        ("entity_poly", entity_poly_columns),
        ("entity_poly_seq", entity_poly_seq_columns),
        ("pdbx_poly_seq_scheme", poly_scheme_columns),
        ("pdbx_entity_branch", branch_columns),
        ("pdbx_entity_branch_list", branch_list_columns),
        ("pdbx_entity_branch_link", branch_link_columns),
        ("pdbx_branch_scheme", branch_scheme_columns),
        ("pdbx_entity_nonpoly", nonpoly_columns),
        ("pdbx_nonpoly_scheme", nonpoly_scheme_columns),
        ("chem_comp", chem_comp_columns),
        ("chem_comp_atom", chem_atom_columns),
        ("chem_comp_bond", chem_bond_columns),
        ("struct_conn", conn_columns),
    ):
        if columns and tuple(columns.values())[0]:
            categories[name] = CIFCategory(columns)
    CIFFile({"ofoldx": CIFBlock(categories)}).write(str(path))
