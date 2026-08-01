r"""Resolve chemical components from a Hub-hosted or local CCD snapshot."""

from __future__ import annotations

import mmap
import os
import re
from collections.abc import Mapping
from functools import cache
from pathlib import Path

from ..components import component_from_rows
from ..molecular import (
    ChemicalComponent,
    PolymerAttachment,
)

#: Hub dataset repository containing CCD component rows.
CCD_DATASET_REPOSITORY = "oteam/ccd"

_POLYMER_CHEM_TYPES = frozenset({"protein", "rna", "dna", "dna_rna_hybrid"})


def resolve_ccd_component(
    component_id: str,
    *,
    revision: str | None = None,
    snapshot_path: str | os.PathLike[str] | None = None,
) -> ChemicalComponent:
    r"""
    Load one wwPDB Chemical Component Dictionary entry.

    Args:
        component_id: Exact CCD component code.
        revision: Hub dataset revision containing the desired CCD snapshot.
        snapshot_path: Local uncompressed CCD CIF, Parquet shard, or shard directory.

    Returns:
        The resolved chemical component.

    Raises:
        KeyError: If the component does not exist in the selected snapshot.
        FileNotFoundError: If ``snapshot_path`` does not exist.
        TypeError: If the stored row has an invalid type.
        ValueError: If exactly one snapshot locator is not supplied, the local
            snapshot format is unsupported, or a stored row is inconsistent.
    """

    if (revision is None) == (snapshot_path is None):
        raise ValueError("exactly one of revision or snapshot_path is required")
    if revision is not None:
        component = _resolve_cached(component_id, revision, None)
    else:
        assert snapshot_path is not None
        snapshot = _resolve_local_snapshot(os.fspath(snapshot_path))
        component = _resolve_cached(component_id, None, str(snapshot))
    if component is None:
        raise KeyError(component_id)
    return component


@cache
def _resolve_cached(
    component_id: str,
    revision: str | None,
    snapshot_path: str | None,
) -> ChemicalComponent | None:
    if revision is not None:
        snapshot = _resolve_ccd_snapshot(revision)
    else:
        assert snapshot_path is not None
        snapshot = Path(snapshot_path)
    row = _read_component_row(snapshot, component_id)
    return None if row is None else _lower_row(row)


@cache
def _resolve_ccd_snapshot(revision: str) -> Path:
    r"""
    Download the pinned CCD dataset's Parquet shards.
    """

    from huggingface_hub import snapshot_download

    return _validate_snapshot(
        Path(
            snapshot_download(
                repo_id=CCD_DATASET_REPOSITORY,
                repo_type="dataset",
                revision=revision,
                allow_patterns=["*.parquet"],
            )
        )
    )


def _resolve_local_snapshot(snapshot_path: str) -> Path:
    path = Path(snapshot_path).expanduser()
    try:
        return _validate_snapshot(path.resolve(strict=True))
    except FileNotFoundError:
        raise FileNotFoundError(f"CCD snapshot does not exist: {path}") from None


def _validate_snapshot(snapshot: Path) -> Path:
    if snapshot.is_file() and snapshot.suffix.casefold() == ".cif":
        return snapshot
    if not _parquet_shards(snapshot):
        raise ValueError(f"CCD snapshot contains no Parquet shards: {snapshot}")
    return snapshot


def _parquet_shards(snapshot: Path) -> tuple[Path, ...]:
    if snapshot.is_file():
        return (snapshot,) if snapshot.suffix.casefold() == ".parquet" else ()
    return tuple(sorted(snapshot.glob("*.parquet")))


def _read_component_row(snapshot: Path, component_id: str) -> Mapping[str, object] | None:
    r"""
    Return the unique component row for ``component_id``, or ``None``.
    """

    if snapshot.suffix.casefold() == ".cif":
        return _read_cif_component_row(snapshot, component_id)

    import pyarrow.dataset as ds

    shards = _parquet_shards(snapshot)
    selected = ds.dataset(shards, format="parquet").to_table(filter=ds.field("component_id") == component_id)
    if selected.num_rows == 0:
        return None
    if selected.num_rows != 1:
        raise ValueError(f"CCD component {component_id!r} occurs in {selected.num_rows} rows")
    return {name: column[0].as_py() for name, column in zip(selected.column_names, selected.columns)}


def _lower_row(row: Mapping[str, object]) -> ChemicalComponent:
    r"""
    Lower one CCD Parquet row to a [`ChemicalComponent`][].
    """

    component_id = _require_str(row, "component_id")
    atom_ids = _require_str_list(row, "atom_ids")
    # CCD symbols are often uppercase (CA, CL); source readers use Ca, Cl.
    elements = [element.strip().capitalize() for element in _require_str_list(row, "elements")]
    formal_charges = _require_optional_int_list(row, "formal_charges")
    aromatic = _require_optional_bool_list(row, "aromatic")
    stereo = _require_optional_str_list(row, "stereo")

    count = len(atom_ids)
    for name, values in (
        ("elements", elements),
        ("formal_charges", formal_charges),
        ("aromatic", aromatic),
        ("stereo", stereo),
    ):
        if len(values) != count:
            raise ValueError(f"CCD row {component_id!r} field {name!r} misaligns with atom_ids")

    atom_rows = list(zip(atom_ids, elements, formal_charges, aromatic, stereo))

    bond_rows = _lower_bond_rows(row, component_id)
    reference_coordinates = _lower_reference_coordinates(row, component_id, count)
    leaving_atom_ids = frozenset(_require_str_list(row, "leaving_atom_ids"))
    non_leaving_atom_ids = frozenset(_require_str_list(row, "non_leaving_atom_ids"))
    previous_attachment = _lower_attachment(row, "previous")
    next_attachment = _lower_attachment(row, "next")

    chem_type = _optional_str(row.get("chem_type"))
    if chem_type is not None and chem_type not in _POLYMER_CHEM_TYPES:
        # A non-polymer component never carries linear polymer endpoints.
        previous_attachment = None
        next_attachment = None

    return component_from_rows(
        component_id,
        atom_rows,
        bond_rows,
        reference_coordinates=reference_coordinates,
        previous_attachment=previous_attachment,
        next_attachment=next_attachment,
        leaving_atom_ids=leaving_atom_ids,
        non_leaving_atom_ids=non_leaving_atom_ids,
    )


def _lower_bond_rows(
    row: Mapping[str, object],
    component_id: str,
) -> list[tuple[str, str, float, bool, str | None]]:
    r"""
    Lower the parallel bond arrays to raw ``(a, b, order, aromatic, stereo)`` rows.
    """

    atom_a = _require_str_list(row, "bond_atom_a")
    atom_b = _require_str_list(row, "bond_atom_b")
    orders = _require_optional_float_list(row, "bond_orders")
    aromatic = _require_optional_bool_list(row, "bond_aromatic")
    stereo = _require_optional_str_list(row, "bond_stereo")

    count = len(atom_a)
    for name, values in (
        ("bond_atom_b", atom_b),
        ("bond_orders", orders),
        ("bond_aromatic", aromatic),
        ("bond_stereo", stereo),
    ):
        if len(values) != count:
            raise ValueError(f"CCD row {component_id!r} bond field {name!r} misaligns with bond_atom_a")

    result: list[tuple[str, str, float, bool, str | None]] = []
    for index, (first, second, order, is_aromatic, bond_stereo) in enumerate(
        zip(atom_a, atom_b, orders, aromatic, stereo)
    ):
        if order is None:
            raise ValueError(f"CCD row {component_id!r} bond {index} has no bond order")
        if is_aromatic is None:
            raise ValueError(f"CCD row {component_id!r} bond {index} has no aromaticity flag")
        result.append((first, second, order, is_aromatic, bond_stereo))
    return result


def _lower_reference_coordinates(
    row: Mapping[str, object],
    component_id: str,
    count: int,
) -> tuple[tuple[float, float, float] | None, ...]:
    r"""
    Lower reference coordinates to a per-atom coordinate tuple aligned to the atoms.
    """

    raw = row.get("reference_coordinates")
    if raw is None:
        return ()
    if not isinstance(raw, (list, tuple)):
        raise TypeError(f"CCD row {component_id!r} reference_coordinates must be a list")
    if len(raw) != count:
        raise ValueError(f"CCD row {component_id!r} reference_coordinates misaligns with atom_ids")

    coordinates: list[tuple[float, float, float] | None] = []
    for entry in raw:
        if entry is None:
            coordinates.append(None)
            continue
        if not isinstance(entry, (list, tuple)) or len(entry) != 3:
            raise TypeError(f"CCD row {component_id!r} reference_coordinates entry must be an (x, y, z) triple")
        coordinates.append((float(entry[0]), float(entry[1]), float(entry[2])))
    return tuple(coordinates)


def _lower_attachment(row: Mapping[str, object], side: str) -> PolymerAttachment | None:
    r"""
    Lower one polymer attachment endpoint, or ``None`` when absent.
    """

    atom_id = _optional_str(row.get(f"{side}_attachment_atom"))
    if atom_id is None:
        return None
    leaving = row.get(f"{side}_leaving_atoms")
    if leaving is None:
        leaving_ids: frozenset[str] = frozenset()
    else:
        if not isinstance(leaving, (list, tuple)):
            raise TypeError(f"CCD row {side}_leaving_atoms must be a list")
        leaving_ids = frozenset(_as_str(value) for value in leaving)
    return PolymerAttachment(atom_id, leaving_ids)


def _as_str(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"expected a string, got {type(value).__name__}")
    return value


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    if value == "" or value == "?":
        return None
    return _as_str(value)


def _require_str(row: Mapping[str, object], name: str) -> str:
    if name not in row:
        raise ValueError(f"CCD row is missing required field {name!r}")
    return _as_str(row[name])


def _require_list(row: Mapping[str, object], name: str) -> list[object]:
    if name not in row:
        raise ValueError(f"CCD row is missing required field {name!r}")
    value = row[name]
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"CCD row field {name!r} must be a list")
    return list(value)


def _require_str_list(row: Mapping[str, object], name: str) -> list[str]:
    return [_as_str(value) for value in _require_list(row, name)]


def _require_optional_int_list(row: Mapping[str, object], name: str) -> list[int | None]:
    result: list[int | None] = []
    for value in _require_list(row, name):
        if value is None:
            result.append(None)
            continue
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"CCD row field {name!r} must contain integers or None")
        result.append(value)
    return result


def _require_optional_float_list(row: Mapping[str, object], name: str) -> list[float | None]:
    result: list[float | None] = []
    for value in _require_list(row, name):
        if value is None:
            result.append(None)
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"CCD row field {name!r} must contain numbers or None")
        result.append(float(value))
    return result


def _require_optional_bool_list(row: Mapping[str, object], name: str) -> list[bool | None]:
    result: list[bool | None] = []
    for value in _require_list(row, name):
        if value is None:
            result.append(None)
            continue
        if not isinstance(value, bool):
            raise TypeError(f"CCD row field {name!r} must contain booleans or None")
        result.append(value)
    return result


def _require_optional_str_list(row: Mapping[str, object], name: str) -> list[str | None]:
    return [_optional_str(value) for value in _require_list(row, name)]


@cache
def _cif_block_offsets(path: str) -> dict[str, tuple[int, int]]:
    r"""Index a CCD using shared mmap pages without retaining its full text."""

    blocks: dict[str, tuple[int, int]] = {}
    with open(path, "rb") as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as data:
        previous: tuple[str, int] | None = None
        for match in re.finditer(rb"(?m)^data_([^\s]+)[ \t]*\r?$", data):
            if previous is not None:
                blocks[previous[0]] = (previous[1], match.start())
            identifier = match.group(1).decode("utf-8")
            if identifier in blocks:
                raise ValueError(f"duplicate CCD block {identifier!r} in {path}")
            previous = identifier, match.start()
        if previous is not None:
            blocks[previous[0]] = (previous[1], len(data))
    if not blocks:
        raise ValueError(f"CCD CIF contains no data blocks: {path}")
    return blocks


def _read_cif_component_row(snapshot: Path, component_id: str) -> Mapping[str, object] | None:
    from ...io.cif import parse_cif
    from .reader import decode_component

    offsets = _cif_block_offsets(str(snapshot)).get(component_id)
    if offsets is None:
        return None
    start, stop = offsets
    with snapshot.open("rb") as handle:
        handle.seek(start)
        text = handle.read(stop - start).decode("utf-8")
    return decode_component(parse_cif(text).block)
