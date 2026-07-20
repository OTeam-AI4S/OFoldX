r"""Import-time resolver from a Hub CCD dataset row to a `ChemicalComponent`.

This resolver is the boundary-time counterpart to
`ofoldx.data.components.get_component`. `get_component` is audited-only and does
zero I/O (Principle 7): it only ever returns one of the package-audited standard
components and never guesses a graph from a residue name. `resolve_ccd_component`
is the opposite tool for a different job — a family that folds arbitrary CCD-code
ligands calls it once, at its importer / Processor boundary, to fetch the wwPDB
CCD snapshot and lower one component row into a canonical `ChemicalComponent`
VALUE. The family then embeds the returned values in its `MolecularSystem`;
nothing here runs during tensor compute, and there is no live registry lookup.

The lowering here mirrors `ofoldx.data.components._build_protein_component` and
`_build_nucleic_component`: the same `ChemicalAtom` / `ChemicalBond` /
`PolymerAttachment` value shape, per-atom `reference_coordinates` aligned to the
atom tuple, and polymer attachments only for polymer components.

The CCD is a Hub `dataset` repository (`ofoldx/ccd`), Parquet, one component per
row, with `revision` pinned to the wwPDB snapshot date. See `README.md` for the
exact row schema and `build.py` for the wwPDB `components.cif` converter.
"""

from __future__ import annotations

from pathlib import Path

from ..components import _component_from_raw
from ..molecular import (
    ChemicalComponent,
    PolymerAttachment,
)

CCD_DATASET_REPOSITORY = "ofoldx/ccd"
r"""The Hub dataset repository that distributes the CCD as Parquet rows."""

_POLYMER_CHEM_TYPES = frozenset({"protein", "rna", "dna", "dna_rna_hybrid"})


def resolve_ccd_component(component_id: str, *, revision: str) -> ChemicalComponent:
    r"""Fetch and lower one wwPDB CCD component to a canonical value.

    ``component_id`` is an exact wwPDB CCD code (for example ``"HEM"`` or
    ``"ATP"``). ``revision`` is the wwPDB snapshot date that pins the Hub dataset
    commit, so the returned value is reproducible and provenance-tagged.

    This performs network / disk I/O and is therefore an import-time (boundary)
    operation only. Call it once while building a `MolecularSystem` and embed the
    returned value; never call it during tensor compute. Unknown ``component_id``
    raises `KeyError`; a malformed row raises `ValueError` or `TypeError`.
    """

    snapshot = _resolve_ccd_snapshot(revision=revision)
    row = _read_component_row(snapshot, component_id)
    if row is None:
        raise KeyError(component_id)
    return _lower_row(row)


def _resolve_ccd_snapshot(*, revision: str) -> Path:
    r"""Download the pinned CCD dataset snapshot, reusing the hub mechanism.

    This reuses the same `huggingface_hub.snapshot_download` mechanism that
    `ofoldx.hub` uses for model repositories, but targets a ``dataset`` repo and
    restricts the download to Parquet shards. ``revision`` pins the exact wwPDB
    snapshot commit.
    """

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
    r"""Return the single Parquet row for ``component_id`` as a plain dict.

    One component per row: the first shard that contains the code wins, and a code
    absent from every shard yields ``None``. Reading uses ``pyarrow`` lazily so no
    tensor or array dependency is imported.
    """

    import pyarrow.parquet as pq
    from pyarrow import compute

    for shard in sorted(snapshot.glob("*.parquet")):
        table = pq.read_table(shard)
        mask = compute.equal(table.column("component_id"), component_id)
        selected = table.filter(mask)
        if selected.num_rows == 0:
            continue
        if selected.num_rows != 1:
            raise ValueError(f"CCD component {component_id!r} occurs in {selected.num_rows} rows")
        return {name: column[0].as_py() for name, column in zip(selected.column_names, selected.columns)}
    return None


def _lower_row(row: dict[str, object]) -> ChemicalComponent:
    r"""Lower one CCD Parquet row to a `ChemicalComponent` value.

    Mirrors the value shape produced by
    `ofoldx.data.components._build_protein_component` /
    `_build_nucleic_component`.
    """

    component_id = _require_str(row, "component_id")
    atom_ids = _require_str_list(row, "atom_ids")
    elements = _require_str_list(row, "elements")
    formal_charges = _require_int_list(row, "formal_charges")
    aromatic = _require_bool_list(row, "aromatic")
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

    return _component_from_raw(
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
    r"""Lower the parallel bond arrays to raw ``(a, b, order, aromatic, stereo)`` rows."""

    atom_a = _require_str_list(row, "bond_atom_a")
    atom_b = _require_str_list(row, "bond_atom_b")
    orders = _require_float_list(row, "bond_orders")
    aromatic = _require_bool_list(row, "bond_aromatic")
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

    return list(zip(atom_a, atom_b, orders, aromatic, stereo))


def _lower_reference_coordinates(
    row: dict[str, object],
    component_id: str,
    count: int,
) -> tuple[tuple[float, float, float] | None, ...]:
    r"""Lower ``ref_coords`` to a per-atom coordinate tuple aligned to the atoms."""

    raw = row.get("ref_coords")
    if raw is None:
        return ()
    if not isinstance(raw, (list, tuple)):
        raise TypeError(f"CCD row {component_id!r} ref_coords must be a list")
    if len(raw) != count:
        raise ValueError(f"CCD row {component_id!r} ref_coords misaligns with atom_ids")

    coordinates: list[tuple[float, float, float] | None] = []
    for entry in raw:
        if entry is None:
            coordinates.append(None)
            continue
        if not isinstance(entry, (list, tuple)) or len(entry) != 3:
            raise TypeError(f"CCD row {component_id!r} ref_coords entry must be an (x, y, z) triple")
        coordinates.append((float(entry[0]), float(entry[1]), float(entry[2])))
    return tuple(coordinates)


def _lower_attachment(row: dict[str, object], side: str) -> PolymerAttachment | None:
    r"""Lower one polymer attachment endpoint, or ``None`` when absent."""

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


def _require_int_list(row: dict[str, object], name: str) -> list[int]:
    result: list[int] = []
    for value in _require_list(row, name):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"CCD row field {name!r} must contain integers")
        result.append(value)
    return result


def _require_float_list(row: dict[str, object], name: str) -> list[float]:
    result: list[float] = []
    for value in _require_list(row, name):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"CCD row field {name!r} must contain numbers")
        result.append(float(value))
    return result


def _require_bool_list(row: dict[str, object], name: str) -> list[bool]:
    result: list[bool] = []
    for value in _require_list(row, name):
        if not isinstance(value, bool):
            raise TypeError(f"CCD row field {name!r} must contain booleans")
        result.append(value)
    return result


def _require_optional_str_list(row: dict[str, object], name: str) -> list[str | None]:
    return [_optional_str(value) for value in _require_list(row, name)]
