r"""
Transient source columns shared by the PDB and mmCIF readers.

These arrays exist only while text identities are resolved into the canonical
tensor molecular graph.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

from ..data.molecular import AtomSite

if TYPE_CHECKING:
    from biotite.structure import AtomArray


_ATOM_ROW_DTYPE = np.dtype(
    [
        ("row_index", np.int64),
        ("group", object),
        ("serial", object),
        ("element", object),
        ("isotope", object),
        ("charge", object),
        ("atom_name", object),
        ("component_id", object),
        ("chain_id", object),
        ("entity_id", object),
        ("label_seq_id", object),
        ("residue_id", object),
        ("auth_chain_id", object),
        ("auth_residue_id", object),
        ("insertion_code", object),
        ("auth_component_id", object),
        ("auth_atom_name", object),
        ("altloc_id", object),
        ("occupancy", object),
        ("b_factor", object),
        ("coordinates", np.float32, (3,)),
        ("coordinate_present", np.bool_),
        ("symmetry_id", object),
    ]
)


def atom_rows_from_array(
    array: AtomArray,
    *,
    format_name: Literal["pdb", "mmcif"],
) -> np.recarray:
    r"""
    Normalize a selected Biotite atom array into transient source columns.
    """

    categories = set(array.get_annotation_categories())
    length = int(array.array_length())

    def annotation(name: str) -> NDArray | None:
        return array.get_annotation(name) if name in categories else None

    atom_serials = annotation("atom_id")
    charges = annotation("charge" if format_name == "pdb" else "pdbx_formal_charge")
    occupancies = annotation("occupancy")
    b_factors = annotation("b_factor")
    altloc_ids = annotation("altloc_id")
    symmetry_ids = annotation("sym_id")
    auth_asym_ids = annotation("auth_asym_id")
    auth_seq_ids = annotation("auth_seq_id")
    auth_comp_ids = annotation("auth_comp_id")
    auth_atom_ids = annotation("auth_atom_id")
    label_entity_ids = annotation("label_entity_id")
    label_asym_ids = annotation("label_asym_id")
    label_seq_ids = annotation("label_seq_id")

    rows = np.empty(length, dtype=_ATOM_ROW_DTYPE).view(np.recarray)
    for index in range(length):
        chain_id = str(array.chain_id[index]).strip()
        if format_name == "mmcif":
            chain_id = present_string(_at(label_asym_ids, index)) or chain_id or "_"
        label_seq_id = optional_integer(_at(label_seq_ids, index))
        residue_id = label_seq_id if label_seq_id is not None else int(array.res_id[index])
        coordinate = np.asarray(array.coord[index], dtype=np.float32)
        coordinate_present = bool(np.isfinite(coordinate).all())
        if not coordinate_present:
            coordinate = np.zeros(3, dtype=np.float32)
        charge = optional_integer(_at(charges, index))
        if format_name == "pdb" and charge == 0:
            charge = None
        element = str(array.element[index]).strip().capitalize()
        isotope = None
        if element == "D":
            element = "H"
            isotope = 2
        elif element == "T":
            element = "H"
            isotope = 3
        rows[index] = (
            index,
            "HETATM" if bool(array.hetero[index]) else "ATOM",
            present_string(_at(atom_serials, index)),
            element,
            isotope,
            charge,
            str(array.atom_name[index]),
            str(array.res_name[index]),
            chain_id,
            present_string(_at(label_entity_ids, index)),
            label_seq_id,
            residue_id,
            (
                present_string(_at(auth_asym_ids, index))
                if format_name == "mmcif"
                else present_string(array.chain_id[index])
            ),
            (present_string(_at(auth_seq_ids, index)) if format_name == "mmcif" else int(array.res_id[index])),
            present_string(array.ins_code[index]),
            (present_string(_at(auth_comp_ids, index)) if format_name == "mmcif" else str(array.res_name[index])),
            (present_string(_at(auth_atom_ids, index)) if format_name == "mmcif" else str(array.atom_name[index])),
            present_string(_at(altloc_ids, index)),
            _optional_float(_at(occupancies, index)),
            _optional_float(_at(b_factors, index)),
            coordinate,
            coordinate_present,
            present_string(_at(symmetry_ids, index)),
        )
    return rows


def _at(values: NDArray | None, index: int) -> object | None:
    return None if values is None else values[index]


def present_string(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return None if text in {"", ".", "?"} else text


def optional_integer(value: object | None) -> int | None:
    text = present_string(value)
    if text is None:
        return None
    try:
        return int(text)
    except ValueError:
        raise ValueError(f"expected an integer identity, got {text!r}") from None


def _optional_float(value: object | None) -> float | None:
    text = present_string(value)
    return None if text is None else float(text)


def reject_duplicate_atom_rows(rows: np.recarray) -> None:
    seen: set[tuple[object, ...]] = set()
    for row in rows:
        key = (
            row.symmetry_id,
            row.chain_id,
            row.residue_id,
            row.auth_residue_id,
            row.insertion_code,
            row.component_id,
            row.atom_name,
            row.altloc_id,
        )
        if key in seen:
            raise ValueError(f"duplicate source atom declaration: {key!r}")
        seen.add(key)


def atom_row_residue_key(row: np.record) -> tuple[object, ...]:
    return (
        row.symmetry_id,
        row.chain_id,
        row.residue_id,
        row.auth_residue_id,
        row.insertion_code,
        row.component_id,
    )


def selected_source_sites(
    selected_rows: np.recarray,
    site_by_row_index: dict[int, AtomSite],
) -> tuple[dict[tuple[object, ...], AtomSite], set[tuple[object, ...]]]:
    mapping: dict[tuple[object, ...], AtomSite] = {}
    ambiguous: set[tuple[object, ...]] = set()
    for row in selected_rows:
        site = site_by_row_index[int(row.row_index)]
        label_key = (
            row.symmetry_id,
            row.chain_id,
            row.label_seq_id,
            row.insertion_code,
            row.component_id,
            row.atom_name,
        )
        auth_key = (
            row.symmetry_id,
            row.auth_chain_id,
            row.auth_residue_id,
            row.insertion_code,
            row.auth_component_id,
            row.auth_atom_name,
        )
        for key in (label_key, auth_key):
            if key in ambiguous:
                continue
            previous = mapping.get(key)
            if previous is not None and previous != site:
                mapping.pop(key)
                ambiguous.add(key)
            else:
                mapping[key] = site
    return mapping, ambiguous


def source_row_key(row: np.record) -> tuple[object, ...]:
    return (
        row.symmetry_id,
        row.chain_id,
        row.residue_id,
        row.insertion_code,
        row.component_id,
        row.atom_name,
    )


def unique_identifier(seed: str, used: set[str]) -> str:
    candidate = seed
    suffix = 2
    while candidate in used:
        candidate = f"{seed}_{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate
