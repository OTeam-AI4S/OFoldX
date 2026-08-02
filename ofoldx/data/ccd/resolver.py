r"""
Resolve chemical components from a Hub-hosted or local CCD dataset.

The dataset stores one component per Parquet row. A Hub ``revision`` or a local
``snapshot_path`` pins the wwPDB snapshot used for resolution.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

from ..components import component_from_rows
from ..elements import ATOMIC_NUMBER_BY_ELEMENT
from ..molecular import (
    ChemicalComponent,
    HYDROGEN_ELEMENTS,
    PolymerAttachment,
)

#: Hub dataset repository containing CCD component rows.
CCD_DATASET_REPOSITORY = "ofoldx/ccd"

_POLYMER_CHEM_TYPES = frozenset({"protein", "rna", "dna", "dna_rna_hybrid"})


def resolve_ccd_component(
    component_id: str,
    *,
    revision: str | None = None,
    snapshot_path: str | None = None,
) -> ChemicalComponent:
    r"""
    Load one wwPDB Chemical Component Dictionary entry.

    Args:
        component_id: Exact CCD component code.
        revision: Hub dataset revision containing the desired CCD snapshot.
        snapshot_path: Existing local directory containing resolver-schema
            Parquet shards. Local resolution performs no Hub access.

    Returns:
        The resolved chemical component.

    Raises:
        KeyError: If the component does not exist in the selected snapshot.
        FileNotFoundError: If ``snapshot_path`` is not an existing directory.
        TypeError: If the stored row has an invalid type.
        ValueError: If exactly one locator is not supplied, the local directory
            has no Parquet shards, or a stored row is internally inconsistent.
    """

    if (revision is None) == (snapshot_path is None):
        raise ValueError("exactly one of revision or snapshot_path is required")
    if revision is not None:
        return _resolve_cached(component_id, revision, None)

    assert snapshot_path is not None
    canonical = validate_ccd_snapshot_path(snapshot_path)
    return _resolve_cached(component_id, None, str(canonical))


def validate_ccd_snapshot_path(snapshot_path: str) -> Path:
    r"""Return the canonical path of a readable local resolver snapshot."""

    path = Path(snapshot_path).expanduser()
    try:
        canonical = path.resolve(strict=True)
    except FileNotFoundError:
        raise FileNotFoundError(f"CCD snapshot_path does not exist: {path}") from None
    if not canonical.is_dir():
        raise FileNotFoundError(f"CCD snapshot_path is not a directory: {canonical}")
    if not any(canonical.glob("*.parquet")):
        raise ValueError(f"CCD snapshot_path contains no Parquet shards: {canonical}")
    return canonical


@cache
def _resolve_cached(
    component_id: str,
    revision: str | None,
    snapshot_path: str | None,
) -> ChemicalComponent:
    snapshot = Path(snapshot_path) if snapshot_path is not None else _resolve_ccd_snapshot(revision=revision)
    row = _read_component_row(snapshot, component_id)
    if row is None:
        raise KeyError(component_id)
    return _lower_row(row)


@cache
def _resolve_ccd_snapshot(*, revision: str | None) -> Path:
    r"""
    Download the pinned CCD dataset's Parquet shards.
    """

    if revision is None:
        raise ValueError("revision is required for Hub CCD resolution")

    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=CCD_DATASET_REPOSITORY,
            repo_type="dataset",
            revision=revision,
            allow_patterns=["*.parquet"],
        )
    )


def _read_component_row(snapshot: Path, component_id: str) -> dict[str, object] | None:
    r"""
    Return the unique Parquet row for ``component_id``, or ``None``.
    """

    import pyarrow.dataset as ds

    shards = sorted(snapshot.glob("*.parquet"))
    if not shards:
        return None
    selected = ds.dataset(shards, format="parquet").to_table(filter=ds.field("component_id") == component_id)
    if selected.num_rows == 0:
        return None
    if selected.num_rows != 1:
        raise ValueError(f"CCD component {component_id!r} occurs in {selected.num_rows} rows")
    return {name: column[0].as_py() for name, column in zip(selected.column_names, selected.columns)}


def _lower_row(row: dict[str, object]) -> ChemicalComponent:
    r"""
    Lower one CCD Parquet row to a [`ChemicalComponent`][].
    """

    component_id = _require_str(row, "component_id")
    atom_ids = _require_str_list(row, "atom_ids")
    elements = [_canonical_element(value) for value in _require_str_list(row, "elements")]
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
    )


def _lower_bond_rows(
    row: dict[str, object],
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
    row: dict[str, object],
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


def _lower_attachment(row: dict[str, object], side: str) -> PolymerAttachment | None:
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


def _canonical_element(value: str) -> str:
    canonical = value[:1].upper() + value[1:].lower()
    if canonical not in ATOMIC_NUMBER_BY_ELEMENT and canonical not in HYDROGEN_ELEMENTS:
        raise ValueError(f"CCD row contains invalid element {value!r}")
    return canonical


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    if value == "" or value == "?":
        return None
    return _as_str(value)


def _require_str(row: dict[str, object], name: str) -> str:
    if name not in row:
        raise ValueError(f"CCD row is missing required field {name!r}")
    return _as_str(row[name])


def _require_list(row: dict[str, object], name: str) -> list[object]:
    if name not in row:
        raise ValueError(f"CCD row is missing required field {name!r}")
    value = row[name]
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"CCD row field {name!r} must be a list")
    return list(value)


def _require_str_list(row: dict[str, object], name: str) -> list[str]:
    return [_as_str(value) for value in _require_list(row, name)]


def _require_optional_int_list(row: dict[str, object], name: str) -> list[int | None]:
    result: list[int | None] = []
    for value in _require_list(row, name):
        if value is None:
            result.append(None)
            continue
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"CCD row field {name!r} must contain integers or None")
        result.append(value)
    return result


def _require_optional_float_list(row: dict[str, object], name: str) -> list[float | None]:
    result: list[float | None] = []
    for value in _require_list(row, name):
        if value is None:
            result.append(None)
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"CCD row field {name!r} must contain numbers or None")
        result.append(float(value))
    return result


def _require_optional_bool_list(row: dict[str, object], name: str) -> list[bool | None]:
    result: list[bool | None] = []
    for value in _require_list(row, name):
        if value is None:
            result.append(None)
            continue
        if not isinstance(value, bool):
            raise TypeError(f"CCD row field {name!r} must contain booleans or None")
        result.append(value)
    return result


def _require_optional_str_list(row: dict[str, object], name: str) -> list[str | None]:
    return [_optional_str(value) for value in _require_list(row, name)]
