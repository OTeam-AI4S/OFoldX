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

    source_chain_ids = array.chain_id
    source_residue_ids = array.res_id
    source_atom_names = array.atom_name
    source_component_ids = array.res_name
    is_mmcif = format_name == "mmcif"

    chain_ids = [str(value).strip() for value in source_chain_ids]
    if is_mmcif:
        chain_ids = [
            present_string(_at(label_asym_ids, index)) or chain_id or "_" for index, chain_id in enumerate(chain_ids)
        ]
    normalized_label_seq_ids = [optional_integer(_at(label_seq_ids, index)) for index in range(length)]
    residue_ids = [
        int(source_residue_id) if label_seq_id is None else label_seq_id
        for source_residue_id, label_seq_id in zip(
            source_residue_ids,
            normalized_label_seq_ids,
            strict=True,
        )
    ]
    normalized_charges = [optional_integer(_at(charges, index)) for index in range(length)]
    if not is_mmcif:
        normalized_charges = [None if charge == 0 else charge for charge in normalized_charges]
    normalized_elements: list[str] = []
    isotopes: list[int | None] = []
    for value in array.element:
        element = str(value).strip().capitalize()
        isotope = {"D": 2, "T": 3}.get(element)
        normalized_elements.append("H" if isotope is not None else element)
        isotopes.append(isotope)
    coordinates = np.array(array.coord, dtype=np.float32, copy=True, order="C")
    coordinate_present = np.isfinite(coordinates).all(axis=1)
    coordinates[~coordinate_present] = 0.0

    rows = np.empty(length, dtype=_ATOM_ROW_DTYPE).view(np.recarray)
    rows["row_index"] = np.arange(length, dtype=np.int64)
    rows["group"] = np.where(array.hetero, "HETATM", "ATOM")
    rows["serial"] = [present_string(_at(atom_serials, index)) for index in range(length)]
    rows["element"] = normalized_elements
    rows["isotope"] = isotopes
    rows["charge"] = normalized_charges
    rows["atom_name"] = [str(value) for value in source_atom_names]
    rows["component_id"] = [str(value) for value in source_component_ids]
    rows["chain_id"] = chain_ids
    rows["entity_id"] = [present_string(_at(label_entity_ids, index)) for index in range(length)]
    rows["label_seq_id"] = normalized_label_seq_ids
    rows["residue_id"] = residue_ids
    rows["auth_chain_id"] = [
        present_string(_at(auth_asym_ids, index)) if is_mmcif else present_string(source_chain_ids[index])
        for index in range(length)
    ]
    rows["auth_residue_id"] = [
        present_string(_at(auth_seq_ids, index)) if is_mmcif else int(source_residue_ids[index])
        for index in range(length)
    ]
    rows["insertion_code"] = [present_string(value) for value in array.ins_code]
    rows["auth_component_id"] = [
        present_string(_at(auth_comp_ids, index)) if is_mmcif else str(source_component_ids[index])
        for index in range(length)
    ]
    rows["auth_atom_name"] = [
        present_string(_at(auth_atom_ids, index)) if is_mmcif else str(source_atom_names[index])
        for index in range(length)
    ]
    rows["altloc_id"] = [present_string(_at(altloc_ids, index)) for index in range(length)]
    rows["occupancy"] = [_optional_float(_at(occupancies, index)) for index in range(length)]
    rows["b_factor"] = [_optional_float(_at(b_factors, index)) for index in range(length)]
    rows["coordinates"] = coordinates
    rows["coordinate_present"] = coordinate_present
    rows["symmetry_id"] = [present_string(_at(symmetry_ids, index)) for index in range(length)]
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
    symmetry_ids = rows["symmetry_id"]
    chain_ids = rows["chain_id"]
    residue_ids = rows["residue_id"]
    auth_residue_ids = rows["auth_residue_id"]
    insertion_codes = rows["insertion_code"]
    component_ids = rows["component_id"]
    atom_names = rows["atom_name"]
    altloc_ids = rows["altloc_id"]
    for index in range(len(rows)):
        key = (
            symmetry_ids[index],
            chain_ids[index],
            residue_ids[index],
            auth_residue_ids[index],
            insertion_codes[index],
            component_ids[index],
            atom_names[index],
            altloc_ids[index],
        )
        if key in seen:
            raise ValueError(f"duplicate source atom declaration: {key!r}")
        seen.add(key)


def atom_row_residue_key(row: np.record) -> tuple[object, ...]:
    return (
        row["symmetry_id"],
        row["chain_id"],
        row["residue_id"],
        row["auth_residue_id"],
        row["insertion_code"],
        row["component_id"],
    )


def selected_source_sites(
    selected_rows: np.recarray,
    site_by_row_index: dict[int, AtomSite],
) -> tuple[dict[tuple[object, ...], AtomSite], set[tuple[object, ...]]]:
    mapping: dict[tuple[object, ...], AtomSite] = {}
    ambiguous: set[tuple[object, ...]] = set()
    row_indices = selected_rows["row_index"]
    symmetry_ids = selected_rows["symmetry_id"]
    chain_ids = selected_rows["chain_id"]
    label_seq_ids = selected_rows["label_seq_id"]
    insertion_codes = selected_rows["insertion_code"]
    component_ids = selected_rows["component_id"]
    atom_names = selected_rows["atom_name"]
    auth_chain_ids = selected_rows["auth_chain_id"]
    auth_residue_ids = selected_rows["auth_residue_id"]
    auth_component_ids = selected_rows["auth_component_id"]
    auth_atom_names = selected_rows["auth_atom_name"]
    for index in range(len(selected_rows)):
        site = site_by_row_index[int(row_indices[index])]
        label_key = (
            symmetry_ids[index],
            chain_ids[index],
            label_seq_ids[index],
            insertion_codes[index],
            component_ids[index],
            atom_names[index],
        )
        auth_key = (
            symmetry_ids[index],
            auth_chain_ids[index],
            auth_residue_ids[index],
            insertion_codes[index],
            auth_component_ids[index],
            auth_atom_names[index],
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


def unique_identifier(seed: str, used: set[str]) -> str:
    candidate = seed
    suffix = 2
    while candidate in used:
        candidate = f"{seed}_{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate
