r"""
Convert wwPDB ``components.cif`` to CCD Parquet rows.

This is the offline producer for the ``oteam/ccd`` Hub dataset. It parses the
wwPDB Chemical Component Dictionary mmCIF (``components.cif`` / ``components.cif.gz``)
and emits one row per component in the schema that
`ofoldx.data.ccd.resolver.resolve_ccd_component` consumes on demand at an input
boundary. It is a build-time tool only: it is never imported during modeling and
never runs during tensor compute.

Parsing follows the reference generator
``alphafold3/constants/converters/ccd_pickle_gen.py``: read the (optionally
gzipped) multi-data mmCIF and split it into one dict per ``data_`` block. The
field extraction below mirrors ``alphafold3/structure/chemical_components.py``
(`parse_atom_data` / `parse_bond_data`):

- ``_chem_comp.type``                         -> ``chem_type``
- ``_chem_comp.mon_nstd_parent_comp_id``      -> ``parent_component_ids``
- ``_chem_comp_atom.atom_id``                 -> ``atom_ids``
- ``_chem_comp_atom.type_symbol``             -> ``elements``
- ``_chem_comp_atom.charge``                  -> ``formal_charges``
- ``_chem_comp_atom.pdbx_stereo_config``      -> ``stereo`` (per atom; ``N`` -> None)
- ``_chem_comp_atom.pdbx_model_Cartn_{x,y,z}_ideal`` -> ``reference_coordinates``
- ``_chem_comp_atom.pdbx_leaving_atom_flag``  -> component leaving atoms
- ``_chem_comp_bond.atom_id_1`` / ``atom_id_2`` -> ``bond_atom_a`` / ``bond_atom_b``
- ``_chem_comp_bond.value_order``             -> ``bond_orders``
- ``_chem_comp_bond.pdbx_aromatic_flag``      -> ``bond_aromatic`` and per-atom ``aromatic``
- ``_chem_comp_bond.pdbx_stereo_config``      -> ``bond_stereo``

The dataset revision is set (out of band, at upload time) to the wwPDB snapshot
date so that `resolve_ccd_component(..., revision=...)` pins exact provenance.

The CIF and Parquet dependencies are imported only by the two build-time entry
points. Importing this module during modeling does not load either dependency.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from ..components import polymer_attachment_anchors, terminal_leaving_groups

# CCD ``_chem_comp_bond.value_order`` tokens -> canonical bond order.
_VALUE_ORDER = {"SING": 1.0, "DOUB": 2.0, "TRIP": 3.0, "QUAD": 4.0, "AROM": 1.5}


def build_rows(components_cif_path: str) -> list[dict[str, object]]:
    r"""
    Parse a wwPDB `components.cif` into resolver-schema rows.

    The input may be the plain or gzip-compressed multi-data CCD file.

    Args:
        components_cif_path: Path to the plain or gzip-compressed CCD mmCIF file.

    Returns:
        One resolver-schema mapping per component data block.

    Raises:
        ValueError: If a component identifier occurs in more than one data block.
    """

    import gemmi
    from biotite.structure.io.pdbx import CIFFile

    path = Path(components_cif_path)
    cif = CIFFile.deserialize(gemmi.cif.read(str(path)).as_string())

    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for block in cif.values():
        flat = {
            f"_{category_name}.{item_name}": column.as_array(str).tolist()
            for category_name, category in block.items()
            for item_name, column in category.items()
        }
        row = row_from_block(flat)
        component_id = str(row["component_id"])
        if component_id in seen:
            raise ValueError(f"CCD component {component_id!r} occurs in multiple data blocks")
        seen.add(component_id)
        rows.append(row)
    return rows


def write_parquet(rows: Iterable[dict[str, object]], output_path: str) -> None:
    r"""
    Write resolver-schema rows to a single Parquet shard.

    Args:
        rows: Component rows to serialize.
        output_path: Destination Parquet path.
    """

    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.Table.from_pylist(list(rows)), output_path)


def row_from_block(block: Mapping[str, Sequence[str]]) -> dict[str, object]:
    r"""
    Lower one parsed CCD mmCIF data block into a resolver-schema row.

    Args:
        block: Flat `category.item` mapping produced by a multi-data mmCIF parser.

    Returns:
        The normalized component row consumed by the CCD resolver.

    Raises:
        ValueError: If required CCD fields are missing, inconsistent, or invalid.
    """

    component_id = _one(block, "_chem_comp.id")
    raw_type = _one(block, "_chem_comp.type", default="")
    chem_type = _classify_chem_type(raw_type)
    raw_parents = _one(block, "_chem_comp.mon_nstd_parent_comp_id", default="")
    parent_component_ids = [parent.strip() for parent in raw_parents.split(",") if parent.strip() not in {"", ".", "?"}]

    atom_ids = list(block.get("_chem_comp_atom.atom_id", []))
    n_atoms = len(atom_ids)
    elements = _required_column(block, "_chem_comp_atom.type_symbol", n_atoms)
    formal_charges = [_to_optional_int(value) for value in _column(block, "_chem_comp_atom.charge", n_atoms)]
    atom_stereo = [_stereo(value) for value in _column(block, "_chem_comp_atom.pdbx_stereo_config", n_atoms)]
    leaving_flags = _column(block, "_chem_comp_atom.pdbx_leaving_atom_flag", n_atoms)
    normalized_leaving_flags = [_optional_flag(value) for value in leaving_flags]
    leaving_atom_ids = [atom_id for atom_id, flag in zip(atom_ids, normalized_leaving_flags) if flag]
    non_leaving_atom_ids = [atom_id for atom_id, flag in zip(atom_ids, normalized_leaving_flags) if flag is False]
    reference_coordinates = _reference_coordinates(block, n_atoms)

    bond_atom_a = list(block.get("_chem_comp_bond.atom_id_1", []))
    bond_atom_b = _required_column(block, "_chem_comp_bond.atom_id_2", len(bond_atom_a))
    n_bonds = len(bond_atom_a)
    bond_orders = [_bond_order(value) for value in _column(block, "_chem_comp_bond.value_order", n_bonds)]
    bond_aromatic = [_optional_flag(value) for value in _column(block, "_chem_comp_bond.pdbx_aromatic_flag", n_bonds)]
    bond_stereo = [_stereo(value) for value in _column(block, "_chem_comp_bond.pdbx_stereo_config", n_bonds)]

    # Per-atom aromaticity is derived: an atom is aromatic when it participates in
    # any aromatic bond (CCD marks aromaticity on bonds, not atoms).
    aromatic_by_atom: dict[str, bool | None] = dict.fromkeys(atom_ids, False)
    for atom_a, atom_b, is_aromatic in zip(bond_atom_a, bond_atom_b, bond_aromatic):
        for atom_id in (atom_a, atom_b):
            if is_aromatic is True:
                aromatic_by_atom[atom_id] = True
            elif is_aromatic is None and aromatic_by_atom.get(atom_id) is not True:
                aromatic_by_atom[atom_id] = None
    aromatic = [aromatic_by_atom[atom_id] for atom_id in atom_ids]

    previous_atom, previous_leaving, next_atom, next_leaving = _attachments(
        raw_type,
        atom_ids,
        leaving_atom_ids if len(leaving_atom_ids) + len(non_leaving_atom_ids) == len(atom_ids) else None,
        bond_atom_a,
        bond_atom_b,
    )

    return {
        "component_id": component_id,
        "chem_type": chem_type,
        "parent_component_ids": parent_component_ids,
        "atom_ids": atom_ids,
        "elements": elements,
        "formal_charges": formal_charges,
        "aromatic": aromatic,
        "stereo": atom_stereo,
        "bond_atom_a": bond_atom_a,
        "bond_atom_b": bond_atom_b,
        "bond_orders": bond_orders,
        "bond_aromatic": bond_aromatic,
        "bond_stereo": bond_stereo,
        "reference_coordinates": reference_coordinates,
        "leaving_atom_ids": leaving_atom_ids,
        "non_leaving_atom_ids": non_leaving_atom_ids,
        "previous_attachment_atom": previous_atom,
        "previous_leaving_atoms": previous_leaving,
        "next_attachment_atom": next_atom,
        "next_leaving_atoms": next_leaving,
    }


def _classify_chem_type(raw_type: str) -> str:
    upper = raw_type.upper()
    if ("PEPTIDE" in upper and upper != "PEPTIDE-LIKE") or upper in {
        "C-GAMMA LINKING",
        "C-DELTA LINKING",
    }:
        return "protein"
    if "DNA" in upper and "RNA" in upper:
        return "dna_rna_hybrid"
    if "DNA" in upper:
        return "dna"
    if "RNA" in upper:
        return "rna"
    return "non-polymer"


def _attachments(
    chem_comp_type: str,
    atom_ids: Sequence[str],
    leaving_atom_ids: Sequence[str] | None,
    bond_atom_a: Sequence[str] = (),
    bond_atom_b: Sequence[str] = (),
) -> tuple[str | None, list[str], str | None, list[str]]:
    r"""
    Derive default linear-polymer attachments for a polymer component.

    Non-polymer components carry no linear attachments but retain their generic
    leaving-atom facts in the row. For polymers each connected leaving group is
    assigned to its unique non-leaving graph boundary; this recovers nonstandard
    peptide backbones without hard-coding every CCD component ID.
    """

    previous_anchor, next_anchor = polymer_attachment_anchors(chem_comp_type, atom_ids)
    if (previous_anchor is None and next_anchor is None) or leaving_atom_ids is None:
        return None, [], None, []

    order = {atom_id: index for index, atom_id in enumerate(atom_ids)}

    def attachment(anchor: str | None) -> tuple[str | None, list[str]]:
        if anchor is None:
            return None, []
        matching = terminal_leaving_groups(
            atom_ids,
            zip(bond_atom_a, bond_atom_b),
            leaving_atom_ids,
            anchor,
        )
        if len(matching) > 1:
            return None, []
        if not matching:
            return anchor, []
        return anchor, sorted(matching[0], key=order.__getitem__)

    previous_anchor, previous_leaving = attachment(previous_anchor)
    next_anchor, next_leaving = attachment(next_anchor)
    return (
        previous_anchor,
        previous_leaving,
        next_anchor,
        next_leaving,
    )


def _reference_coordinates(
    block: Mapping[str, Sequence[str]],
    n_atoms: int,
) -> list[list[float] | None]:
    xs = _column(block, "_chem_comp_atom.pdbx_model_Cartn_x_ideal", n_atoms)
    ys = _column(block, "_chem_comp_atom.pdbx_model_Cartn_y_ideal", n_atoms)
    zs = _column(block, "_chem_comp_atom.pdbx_model_Cartn_z_ideal", n_atoms)
    coordinates: list[list[float] | None] = []
    for x, y, z in zip(xs, ys, zs):
        x_value, y_value, z_value = _to_float(x), _to_float(y), _to_float(z)
        if x_value is None or y_value is None or z_value is None:
            coordinates.append(None)
        else:
            coordinates.append([x_value, y_value, z_value])
    return coordinates


def _column(block: Mapping[str, Sequence[str]], key: str, length: int) -> list[str]:
    values = block.get(key)
    if values is None:
        return ["?"] * length
    result = list(values)
    if len(result) != length:
        raise ValueError(f"CCD item {key!r} has {len(result)} values; expected {length}")
    return result


def _required_column(block: Mapping[str, Sequence[str]], key: str, length: int) -> list[str]:
    if key not in block and length:
        raise ValueError(f"CCD block is missing required item {key!r}")
    return _column(block, key, length)


def _one(block: Mapping[str, Sequence[str]], key: str, *, default: str | None = None) -> str:
    values = block.get(key)
    if not values:
        if default is not None:
            return default
        raise ValueError(f"CCD block is missing required item {key!r}")
    return values[0]


def _optional_flag(value: str) -> bool | None:
    token = value.strip().upper()
    if token in {"", "?", "."}:
        return None
    if token == "Y":
        return True
    if token == "N":
        return False
    raise ValueError(f"unknown CCD flag {value!r}")


def _stereo(value: str) -> str | None:
    token = value.strip().upper()
    if token in {"", "?", ".", "N"}:
        return None
    return token


def _to_optional_int(value: str) -> int | None:
    token = value.strip()
    if token in {"", "?", "."}:
        return None
    return int(token)


def _bond_order(value: str) -> float | None:
    token = value.strip().upper()
    if token in {"", "?", "."}:
        return None
    try:
        return _VALUE_ORDER[token]
    except KeyError as error:
        raise ValueError(f"unknown CCD bond order {value!r}") from error


def _to_float(value: str) -> float | None:
    token = value.strip()
    if token in {"", "?", "."}:
        return None
    return float(token)
