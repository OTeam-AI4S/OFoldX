r"""Converter stub: wwPDB ``components.cif`` -> CCD Parquet rows.

This is the offline producer for the ``ofoldx/ccd`` Hub dataset. It parses the
wwPDB Chemical Component Dictionary mmCIF (``components.cif`` / ``components.cif.gz``)
and emits one row per component in the schema that
`ofoldx.data.ccd.resolver.resolve_ccd_component` consumes at import time. It is a
build-time tool only: it is never imported during modeling and never runs during
tensor compute.

Parsing follows the reference generator
``alphafold3/constants/converters/ccd_pickle_gen.py``: read the (optionally
gzipped) multi-data mmCIF and split it into one dict per ``data_`` block. The
field extraction below mirrors ``alphafold3/structure/chemical_components.py``
(`parse_atom_data` / `parse_bond_data`):

- ``_chem_comp.type``                         -> ``chem_type``
- ``_chem_comp_atom.atom_id``                 -> ``atom_ids``
- ``_chem_comp_atom.type_symbol``             -> ``elements``
- ``_chem_comp_atom.charge``                  -> ``formal_charges``
- ``_chem_comp_atom.pdbx_stereo_config``      -> ``stereo`` (per atom; ``N`` -> None)
- ``_chem_comp_atom.pdbx_model_Cartn_{x,y,z}_ideal`` -> ``ref_coords``
- ``_chem_comp_atom.pdbx_leaving_atom_flag``  -> polymer attachment leaving atoms
- ``_chem_comp_bond.atom_id_1`` / ``atom_id_2`` -> ``bond_atom_a`` / ``bond_atom_b``
- ``_chem_comp_bond.value_order``             -> ``bond_orders``
- ``_chem_comp_bond.pdbx_aromatic_flag``      -> ``bond_aromatic`` and per-atom ``aromatic``
- ``_chem_comp_bond.pdbx_stereo_config``      -> ``bond_stereo``

The dataset revision is set (out of band, at upload time) to the wwPDB snapshot
date so that `resolve_ccd_component(..., revision=...)` pins exact provenance.

This module is a STUB: the field-lowering helpers below are complete, but reading
mmCIF blocks and writing Parquet are left as ``NotImplementedError`` boundaries so
this file carries no heavy parser/writer dependency and imports cleanly.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

# CCD ``_chem_comp_bond.value_order`` tokens -> canonical bond order.
_VALUE_ORDER = {"SING": 1.0, "DOUB": 2.0, "TRIP": 3.0, "QUAD": 4.0, "AROM": 1.5}

# wwPDB ``_chem_comp.type`` is descriptive free text; these substrings classify the
# polymer family that determines default linear-polymer attachments. Anything else
# is treated as non-polymer (no attachments).
_PROTEIN_TYPE_HINT = "PEPTIDE"
_RNA_TYPE_HINT = "RNA LINKING"
_DNA_TYPE_HINT = "DNA LINKING"


def build_rows(components_cif_path: str) -> list[dict[str, object]]:
    r"""Parse a wwPDB ``components.cif`` into resolver-schema rows.

    STUB: wire this to the mmCIF reader (see ``ccd_pickle_gen.py``) and pass each
    parsed component block to `row_from_block`.
    """

    raise NotImplementedError(
        "wire build_rows to a multi-data mmCIF reader (see the reference "
        "ccd_pickle_gen.py) and map each data_ block through row_from_block"
    )


def write_parquet(rows: Iterable[dict[str, object]], output_path: str) -> None:
    r"""Write resolver-schema rows to a single Parquet shard.

    STUB: serialize ``rows`` with ``pyarrow.parquet`` (imported lazily at the real
    call site) preserving the list-valued columns the resolver expects.
    """

    raise NotImplementedError("wire write_parquet to pyarrow.parquet.write_table")


def row_from_block(block: Mapping[str, Sequence[str]]) -> dict[str, object]:
    r"""Lower one parsed CCD mmCIF ``data_`` block into a resolver-schema row.

    ``block`` is the flat ``{category.item: [values...]}`` mapping produced by a
    multi-data mmCIF parser, exactly the shape ``ccd_pickle_gen.py`` pickles.
    """

    component_id = _one(block, "_chem_comp.id")
    chem_type = _classify_chem_type(_one(block, "_chem_comp.type", default=""))

    atom_ids = list(block.get("_chem_comp_atom.atom_id", []))
    n_atoms = len(atom_ids)
    elements = list(block.get("_chem_comp_atom.type_symbol", []))
    formal_charges = [_to_int(value) for value in _column(block, "_chem_comp_atom.charge", n_atoms)]
    atom_stereo = [_stereo(value) for value in _column(block, "_chem_comp_atom.pdbx_stereo_config", n_atoms)]
    leaving_flags = _column(block, "_chem_comp_atom.pdbx_leaving_atom_flag", n_atoms)
    ref_coords = _reference_coordinates(block, n_atoms)

    bond_atom_a = list(block.get("_chem_comp_bond.atom_id_1", []))
    bond_atom_b = list(block.get("_chem_comp_bond.atom_id_2", []))
    n_bonds = len(bond_atom_a)
    bond_orders = [
        _VALUE_ORDER.get(token.upper(), 1.0) for token in _column(block, "_chem_comp_bond.value_order", n_bonds)
    ]
    bond_aromatic = [_flag(value) for value in _column(block, "_chem_comp_bond.pdbx_aromatic_flag", n_bonds)]
    bond_stereo = [_stereo(value) for value in _column(block, "_chem_comp_bond.pdbx_stereo_config", n_bonds)]

    # Per-atom aromaticity is derived: an atom is aromatic when it participates in
    # any aromatic bond (CCD marks aromaticity on bonds, not atoms).
    aromatic_atoms = set()
    for atom_a, atom_b, is_aromatic in zip(bond_atom_a, bond_atom_b, bond_aromatic):
        if is_aromatic:
            aromatic_atoms.add(atom_a)
            aromatic_atoms.add(atom_b)
    aromatic = [atom_id in aromatic_atoms for atom_id in atom_ids]

    previous_atom, previous_leaving, next_atom, next_leaving = _attachments(chem_type, atom_ids, leaving_flags)

    return {
        "component_id": component_id,
        "chem_type": chem_type,
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
        "ref_coords": ref_coords,
        "previous_attachment_atom": previous_atom,
        "previous_leaving_atoms": previous_leaving,
        "next_attachment_atom": next_atom,
        "next_leaving_atoms": next_leaving,
    }


def _classify_chem_type(raw_type: str) -> str:
    upper = raw_type.upper()
    if _PROTEIN_TYPE_HINT in upper:
        return "protein"
    if _RNA_TYPE_HINT in upper:
        return "rna"
    if _DNA_TYPE_HINT in upper:
        return "dna"
    return "non-polymer"


def _attachments(
    chem_type: str,
    atom_ids: Sequence[str],
    leaving_flags: Sequence[str],
) -> tuple[str | None, list[str], str | None, list[str]]:
    r"""Derive default linear-polymer attachments for a polymer component.

    Non-polymer components carry no attachments. For polymers the backbone anchor
    atoms follow the standard convention (protein ``N``/``C``, nucleic ``P``/``O3'``)
    and the leaving atoms are the CCD ``pdbx_leaving_atom_flag == 'Y'`` atoms bonded
    at each terminus. This stub records the leaving set as declared by the flag and
    leaves precise terminus partitioning to the real build.
    """

    if chem_type == "protein":
        previous_anchor, next_anchor = "N", "C"
    elif chem_type in {"rna", "dna"}:
        previous_anchor, next_anchor = "P", "O3'"
    else:
        return None, [], None, []

    present = set(atom_ids)
    leaving = [atom_id for atom_id, flag in zip(atom_ids, leaving_flags) if _flag(flag)]
    previous_anchor = previous_anchor if previous_anchor in present else None
    next_anchor = next_anchor if next_anchor in present else None
    return previous_anchor, leaving, next_anchor, leaving


def _reference_coordinates(
    block: Mapping[str, Sequence[str]],
    n_atoms: int,
) -> list[tuple[float, float, float] | None]:
    xs = _column(block, "_chem_comp_atom.pdbx_model_Cartn_x_ideal", n_atoms)
    ys = _column(block, "_chem_comp_atom.pdbx_model_Cartn_y_ideal", n_atoms)
    zs = _column(block, "_chem_comp_atom.pdbx_model_Cartn_z_ideal", n_atoms)
    coordinates: list[tuple[float, float, float] | None] = []
    for x, y, z in zip(xs, ys, zs):
        triple = (_to_float(x), _to_float(y), _to_float(z))
        coordinates.append(None if None in triple else triple)  # type: ignore[arg-type]
    return coordinates


def _column(block: Mapping[str, Sequence[str]], key: str, length: int) -> list[str]:
    values = block.get(key)
    if values is None:
        return ["?"] * length
    return list(values)


def _one(block: Mapping[str, Sequence[str]], key: str, *, default: str | None = None) -> str:
    values = block.get(key)
    if not values:
        if default is not None:
            return default
        raise ValueError(f"CCD block is missing required item {key!r}")
    return values[0]


def _flag(value: str) -> bool:
    return value.strip().upper() == "Y"


def _stereo(value: str) -> str | None:
    token = value.strip().upper()
    if token in {"", "?", ".", "N"}:
        return None
    return token


def _to_int(value: str) -> int:
    token = value.strip()
    if token in {"", "?", "."}:
        return 0
    return int(token)


def _to_float(value: str) -> float | None:
    token = value.strip()
    if token in {"", "?", "."}:
        return None
    return float(token)
