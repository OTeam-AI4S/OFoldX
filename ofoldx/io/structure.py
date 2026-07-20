r"""Canonical PDB/mmCIF I/O over :mod:`ofoldx.data` domain values."""

from __future__ import annotations

import contextlib
import itertools
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np

from ..data._codec import decode_structure, encode_structure
from ..data._nucleic_components import _AUDITED_NUCLEIC_COMPONENT_DATA
from ..data._protein_components import _AUDITED_PROTEIN_COMPONENT_DATA
from ..data.components import get_component
from ..data.molecular import (
    AtomSite,
    ChemicalAtom,
    ChemicalBond,
    ChemicalComponent,
    ComponentReference,
    MolecularSystem,
    Molecule,
    MoleculeBond,
    PolymerSequence,
    PolymerType,
    SystemBond,
)
from ..data.structure import ResidueNumber, Structure

_FORMATS = {".pdb": "pdb", ".cif": "mmcif", ".mmcif": "mmcif"}
_ATOM_LAYOUTS = frozenset({"observed", "complete_heavy", "complete_all"})
_ALTLOC_POLICIES = frozenset({"highest_occupancy", "first"})
_CIF_BOND_ORDERS = {
    None: (1.0, False),
    "sing": (1.0, False),
    "doub": (2.0, False),
    "trip": (3.0, False),
    "quad": (4.0, False),
    "arom": (1.5, True),
}
_AUDITED_PROTEIN_COMPONENT_IDS = frozenset(_AUDITED_PROTEIN_COMPONENT_DATA)
_AUDITED_NUCLEIC_COMPONENT_IDS = frozenset(_AUDITED_NUCLEIC_COMPONENT_DATA)


def _component_polymer_type(component_id: str) -> PolymerType | None:
    r"""Classify an audited residue's polymer type, or ``None`` for a non-polymer."""

    # TODO(§3.5 reopened): source-driven polymer/non-polymer classification is reopened;
    # this importer-internal fallback classifies audited residues by their audited set,
    # reproducing the removed ``ChemicalComponent.polymer_types`` behavior until §3.5 lands.
    if component_id in _AUDITED_PROTEIN_COMPONENT_IDS:
        return "protein"
    if component_id in _AUDITED_NUCLEIC_COMPONENT_IDS:
        return "dna" if component_id.startswith("D") else "rna"
    return None


@dataclass(frozen=True)
class LossReport:
    r"""Machine-readable fields lost or rewritten by an exchange writer."""

    dropped_fields: tuple[str, ...] = ()
    rewritten_fields: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _AtomRow:
    row_index: int
    group: Literal["ATOM", "HETATM"]
    serial: str | None
    element: str
    charge: int | None
    atom_name: str
    component_id: str
    chain_id: str
    entity_id: str | None
    label_seq_id: int | None
    residue_id: int | None
    auth_chain_id: str | None
    auth_residue_id: int | None
    insertion_code: str | None
    auth_component_id: str | None
    auth_atom_name: str | None
    altloc_id: str | None
    occupancy: float | None
    b_factor: float | None
    coordinates: tuple[float, float, float]
    coordinate_present: bool
    symmetry_id: str | None = None
    canonical_site: AtomSite | None = None


@dataclass(frozen=True)
class _ParsedInput:
    format: Literal["pdb", "mmcif"]
    rows: tuple[_AtomRow, ...]
    canonical_payload: dict[str, object] | None
    mmcif_file: object | None = None
    pdb_lines: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _PdbAtom:
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


def read_structure(
    path: str | os.PathLike[str],
    *,
    model: int = 1,
    assembly_id: str | None = None,
    altloc: Literal["highest_occupancy", "first"] = "highest_occupancy",
    atom_layout: Literal["observed", "complete_heavy", "complete_all"] = "observed",
    components: Mapping[str, ChemicalComponent] | None = None,
) -> Structure:
    r"""Read one model/assembly with one residue-consistent altloc choice."""

    parsed, component_table = _prepare_read(path, model, assembly_id, atom_layout, components)
    if altloc not in _ALTLOC_POLICIES:
        raise ValueError(f"unsupported altloc policy: {altloc!r}")
    if parsed.canonical_payload is not None:
        return _decode_canonical_mmcif(parsed)
    selected = _select_altloc(parsed.rows, altloc)
    return _build_structure(parsed, selected, atom_layout, component_table)


def read_structures(
    path: str | os.PathLike[str],
    *,
    model: int = 1,
    assembly_id: str | None = None,
    atom_layout: Literal["observed", "complete_heavy", "complete_all"] = "observed",
    components: Mapping[str, ChemicalComponent] | None = None,
) -> tuple[Structure, ...]:
    r"""Expand every residue-consistent alternate-location combination."""

    parsed, component_table = _prepare_read(path, model, assembly_id, atom_layout, components)
    if parsed.canonical_payload is not None:
        return (_decode_canonical_mmcif(parsed),)
    selections = _expand_altlocs(parsed.rows)
    return tuple(_build_structure(parsed, selected, atom_layout, component_table) for selected in selections)


def write_structure(
    structure: Structure,
    path: str | os.PathLike[str],
    *,
    allow_lossy: bool = False,
) -> LossReport | None:
    r"""Write one canonical Structure, refusing unacknowledged format loss."""

    target, format_name = _output_path(path)
    if format_name == "mmcif":
        _write_mmcif(structure, target)
        return LossReport() if allow_lossy else None

    report = _pdb_loss_report(structure)
    if (report.dropped_fields or report.rewritten_fields) and not allow_lossy:
        fields = sorted({*report.dropped_fields, *report.rewritten_fields})
        raise ValueError(
            "PDB cannot represent this Structure losslessly; pass allow_lossy=True " f"to acknowledge: {fields!r}"
        )
    _write_pdb(structure, target)
    return report if allow_lossy else None


def _prepare_read(
    path: str | os.PathLike[str],
    model: int,
    assembly_id: str | None,
    atom_layout: str,
    components: Mapping[str, ChemicalComponent] | None,
) -> tuple[_ParsedInput, dict[str, ChemicalComponent]]:
    source, format_name = _input_path(path)
    if model <= 0:
        raise ValueError("model must be a positive one-based integer")
    if atom_layout not in _ATOM_LAYOUTS:
        raise ValueError(f"unsupported atom_layout: {atom_layout!r}")
    if format_name == "mmcif":
        parsed = _parse_mmcif(source, model=model, assembly_id=assembly_id)
    else:
        parsed = _parse_pdb(source, model=model, assembly_id=assembly_id)
    if not parsed.rows:
        raise ValueError("structure source contains no atoms for the requested model/assembly")

    component_table: dict[str, ChemicalComponent] = {}
    for component_id in {row.component_id for row in parsed.rows}:
        with contextlib.suppress(KeyError):
            component_table[component_id] = get_component(component_id)
    if components is not None:
        supplied = dict(components)
        for component_id, component in supplied.items():
            if component.component_id != component_id:
                raise ValueError("component mapping keys must equal ChemicalComponent.component_id")
        component_table.update(supplied)
    if atom_layout == "complete_all":
        component_table = {
            component_id: _materialize_component_hydrogens(component)
            for component_id, component in component_table.items()
        }
    return parsed, component_table


def _materialize_component_hydrogens(component: ChemicalComponent) -> ChemicalComponent:
    counts = tuple(atom.explicit_hydrogens + atom.implicit_hydrogens for atom in component.atoms)
    if not any(counts):
        return component
    used_atom_ids = {atom.atom_id for atom in component.atoms}
    atoms: list[ChemicalAtom] = []
    bonds = set(component.bonds)
    reference_coordinates: list[tuple[float, float, float] | None] = []
    for atom_index, (atom, count) in enumerate(zip(component.atoms, counts)):
        atoms.append(
            ChemicalAtom(
                atom_id=atom.atom_id,
                element=atom.element,
                isotope=atom.isotope,
                formal_charge=atom.formal_charge,
                radical_electrons=atom.radical_electrons,
                explicit_hydrogens=0,
                implicit_hydrogens=0,
                implicit_hydrogens_allowed=False,
                aromatic=atom.aromatic,
                stereo=atom.stereo,
            )
        )
        if component.reference_coordinates:
            reference_coordinates.append(component.reference_coordinates[atom_index])
        for hydrogen_index in range(1, count + 1):
            seed = f"{atom.atom_id}:H{hydrogen_index}"
            hydrogen_id = seed
            suffix = 2
            while hydrogen_id in used_atom_ids:
                hydrogen_id = f"{seed}_{suffix}"
                suffix += 1
            used_atom_ids.add(hydrogen_id)
            atoms.append(
                ChemicalAtom(
                    atom_id=hydrogen_id,
                    element="H",
                    implicit_hydrogens_allowed=False,
                )
            )
            if component.reference_coordinates:
                reference_coordinates.append(None)
            bonds.add(ChemicalBond(atom.atom_id, hydrogen_id))
    return ChemicalComponent(
        component_id=component.component_id,
        atoms=tuple(atoms),
        bonds=frozenset(bonds),
        previous_attachment=component.previous_attachment,
        next_attachment=component.next_attachment,
        reference_coordinates=tuple(reference_coordinates),
    )


def _input_path(path: str | os.PathLike[str]) -> tuple[Path, Literal["pdb", "mmcif"]]:
    source = Path(os.path.expanduser(os.fspath(path)))
    if not source.exists():
        raise FileNotFoundError(source)
    if not source.is_file():
        raise ValueError(f"structure path is not a file: {source}")
    try:
        return source, _FORMATS[source.suffix.lower()]  # type: ignore[return-value]
    except KeyError:
        raise ValueError("structure path suffix must be .pdb, .cif, or .mmcif") from None


def _output_path(path: str | os.PathLike[str]) -> tuple[Path, Literal["pdb", "mmcif"]]:
    target = Path(os.path.expanduser(os.fspath(path)))
    try:
        format_name = _FORMATS[target.suffix.lower()]
    except KeyError:
        raise ValueError("structure path suffix must be .pdb, .cif, or .mmcif") from None
    if target.exists() and not target.is_file():
        raise ValueError(f"structure output path is not a file: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    return target, format_name  # type: ignore[return-value]


def _parse_pdb(path: Path, *, model: int, assembly_id: str | None) -> _ParsedInput:
    from biotite import InvalidFileError
    from biotite.structure.io.pdb import PDBFile

    pdb_file = PDBFile.read(str(path))
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
    except (InvalidFileError, KeyError) as error:
        raise ValueError(f"PDB does not contain requested assembly {assembly_id!r}") from error
    rows = _rows_from_atom_array(array, format_name="pdb")
    _reject_duplicate_atom_rows(rows)
    try:
        lines = tuple(path.read_text(encoding="ascii").splitlines())
    except UnicodeDecodeError as error:
        raise ValueError("PDB input must be ASCII text") from error
    return _ParsedInput(
        format="pdb",
        rows=rows,
        canonical_payload=None,
        pdb_lines=lines,
    )


def _parse_mmcif(path: Path, *, model: int, assembly_id: str | None) -> _ParsedInput:
    from biotite import InvalidFileError
    from biotite.structure.io import pdbx

    cif_file = pdbx.CIFFile.read(str(path))
    block = cif_file.block
    canonical_payload: dict[str, object] | None = None
    if "ofoldx_structure" in block:
        category = block["ofoldx_structure"]
        if "json" not in category or category.row_count != 1:
            raise ValueError("_ofoldx_structure must contain exactly one json value")
        raw_payload = str(category["json"].as_array()[0])
        decoded_payload = json.loads(raw_payload)
        if type(decoded_payload) is not dict:
            raise ValueError("_ofoldx_structure.json must decode to an object")
        canonical_payload = decoded_payload
        if model != 1:
            raise ValueError("an OFoldX canonical mmCIF exchange view contains model 1 only")
        if assembly_id is not None:
            raise ValueError("an OFoldX canonical mmCIF exchange view contains one materialized assembly")

    atom_site = block.get("atom_site")
    if atom_site is None:
        raise ValueError("mmCIF input has no atom_site category")
    available = set(atom_site.keys())
    requested = [
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
            "ofoldx_instance_id",
            "ofoldx_component_index",
            "ofoldx_atom_id",
            "ofoldx_coordinate_mask",
        )
        if field in available
    ]
    for special, source_column in (
        ("atom_id", "id"),
        ("b_factor", "B_iso_or_equiv"),
        ("occupancy", "occupancy"),
        ("charge", "pdbx_formal_charge"),
    ):
        if source_column in available:
            requested.append(special)
    try:
        if assembly_id is None:
            array = pdbx.get_structure(
                cif_file,
                model=model,
                altloc="all",
                extra_fields=requested,
                use_author_fields=False,
                include_bonds=False,
            )
        else:
            assemblies = pdbx.list_assemblies(cif_file)
            if assembly_id not in assemblies:
                raise ValueError(f"unknown mmCIF assembly_id {assembly_id!r}; available={sorted(assemblies)!r}")
            array = pdbx.get_assembly(
                cif_file,
                assembly_id=assembly_id,
                model=model,
                altloc="all",
                extra_fields=requested,
                use_author_fields=False,
                include_bonds=False,
            )
    except IndexError as error:
        raise ValueError(f"mmCIF does not contain requested model {model}") from error
    except InvalidFileError as error:
        raise ValueError(f"mmCIF cannot materialize requested assembly {assembly_id!r}") from error
    rows = _rows_from_atom_array(array, format_name="mmcif")
    _reject_duplicate_atom_rows(rows)
    return _ParsedInput(
        format="mmcif",
        rows=rows,
        canonical_payload=canonical_payload,
        mmcif_file=cif_file,
    )


def _rows_from_atom_array(array: object, *, format_name: Literal["pdb", "mmcif"]) -> tuple[_AtomRow, ...]:
    categories = set(array.get_annotation_categories())  # type: ignore[attr-defined]
    length = int(array.array_length())  # type: ignore[attr-defined]

    def annotation(name: str) -> object | None:
        if name not in categories:
            return None
        return array.get_annotation(name)  # type: ignore[attr-defined]

    atom_serials = annotation("atom_id")
    charges = annotation("charge")
    occupancies = annotation("occupancy")
    b_factors = annotation("b_factor")
    altloc_ids = annotation("altloc_id")
    sym_ids = annotation("sym_id")
    auth_asym_ids = annotation("auth_asym_id")
    auth_seq_ids = annotation("auth_seq_id")
    auth_comp_ids = annotation("auth_comp_id")
    auth_atom_ids = annotation("auth_atom_id")
    label_entity_ids = annotation("label_entity_id")
    label_asym_ids = annotation("label_asym_id")
    label_seq_ids = annotation("label_seq_id")
    canonical_instances = annotation("ofoldx_instance_id")
    canonical_components = annotation("ofoldx_component_index")
    canonical_atoms = annotation("ofoldx_atom_id")
    coordinate_masks = annotation("ofoldx_coordinate_mask")

    rows: list[_AtomRow] = []
    for index in range(length):
        chain_id = _present_string(_at(label_asym_ids, index)) or _present_string(array.chain_id[index]) or "_"
        label_seq_id = _optional_int(_at(label_seq_ids, index))
        residue_id = label_seq_id
        if residue_id is None:
            residue_id = int(array.res_id[index])
        canonical_site = None
        canonical_instance = _present_string(_at(canonical_instances, index))
        canonical_component = _present_string(_at(canonical_components, index))
        canonical_atom = _present_string(_at(canonical_atoms, index))
        if canonical_instance is not None or canonical_component is not None or canonical_atom is not None:
            if canonical_instance is None or canonical_component is None or canonical_atom is None:
                raise ValueError("OFoldX canonical atom identity fields must be supplied together")
            canonical_site = AtomSite(
                canonical_instance,
                _optional_int(canonical_component),
                canonical_atom,
            )
        coordinate_present = True
        mask_text = _present_string(_at(coordinate_masks, index))
        if mask_text is not None:
            if mask_text not in {"0", "1"}:
                raise ValueError("ofoldx_coordinate_mask values must be 0 or 1")
            coordinate_present = mask_text == "1"
        coordinate = tuple(float(value) for value in array.coord[index])  # type: ignore[attr-defined]
        charge = _optional_int(_at(charges, index))
        if charge == 0:
            charge = None
        rows.append(
            _AtomRow(
                row_index=index,
                group="HETATM" if bool(array.hetero[index]) else "ATOM",  # type: ignore[attr-defined]
                serial=_present_string(_at(atom_serials, index)),
                element=str(array.element[index]).strip().capitalize(),  # type: ignore[attr-defined]
                charge=charge,
                atom_name=str(array.atom_name[index]),  # type: ignore[attr-defined]
                component_id=str(array.res_name[index]),  # type: ignore[attr-defined]
                chain_id=chain_id,
                entity_id=_present_string(_at(label_entity_ids, index)),
                label_seq_id=label_seq_id,
                residue_id=residue_id,
                auth_chain_id=(
                    _present_string(_at(auth_asym_ids, index))
                    if format_name == "mmcif"
                    else _present_string(array.chain_id[index])  # type: ignore[attr-defined]
                ),
                auth_residue_id=(
                    _optional_int(_at(auth_seq_ids, index))
                    if format_name == "mmcif"
                    else int(array.res_id[index])  # type: ignore[attr-defined]
                ),
                insertion_code=_present_string(array.ins_code[index]),  # type: ignore[attr-defined]
                auth_component_id=(
                    _present_string(_at(auth_comp_ids, index))
                    if format_name == "mmcif"
                    else str(array.res_name[index])  # type: ignore[attr-defined]
                ),
                auth_atom_name=(
                    _present_string(_at(auth_atom_ids, index))
                    if format_name == "mmcif"
                    else str(array.atom_name[index])  # type: ignore[attr-defined]
                ),
                altloc_id=_present_string(_at(altloc_ids, index)),
                occupancy=_optional_float(_at(occupancies, index)),
                b_factor=_optional_float(_at(b_factors, index)),
                coordinates=coordinate,  # type: ignore[arg-type]
                coordinate_present=coordinate_present,
                symmetry_id=_present_string(_at(sym_ids, index)),
                canonical_site=canonical_site,
            )
        )
    return tuple(rows)


def _at(values: object | None, index: int) -> object | None:
    return None if values is None else values[index]  # type: ignore[index]


def _present_string(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return None if text in {"", ".", "?"} else text


def _optional_int(value: object | None) -> int | None:
    text = _present_string(value)
    if text is None:
        return None
    try:
        return int(text)
    except ValueError:
        raise ValueError(f"expected an integer identity, got {text!r}") from None


def _optional_float(value: object | None) -> float | None:
    text = _present_string(value)
    if text is None:
        return None
    number = float(text)
    return number if np.isfinite(number) else None


def _reject_duplicate_atom_rows(rows: tuple[_AtomRow, ...]) -> None:
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


def _residue_key(row: _AtomRow) -> tuple[object, ...]:
    return (
        row.symmetry_id,
        row.chain_id,
        row.residue_id,
        row.auth_residue_id,
        row.insertion_code,
        row.component_id,
    )


def _select_altloc(
    rows: tuple[_AtomRow, ...],
    policy: Literal["highest_occupancy", "first"],
) -> tuple[_AtomRow, ...]:
    choices = _altloc_choices(rows)
    selected_labels: dict[tuple[object, ...], str] = {}
    for residue_key, labels in choices:
        if policy == "first":
            selected_labels[residue_key] = labels[0]
            continue
        residue_rows = [row for row in rows if _residue_key(row) == residue_key]
        scores = {
            label: sum((row.occupancy or 0.0) for row in residue_rows if row.altloc_id == label) for label in labels
        }
        selected_labels[residue_key] = max(labels, key=lambda label: scores[label])
    return tuple(
        row
        for row in rows
        if row.altloc_id is None or selected_labels.get(_residue_key(row), row.altloc_id) == row.altloc_id
    )


def _expand_altlocs(rows: tuple[_AtomRow, ...]) -> tuple[tuple[_AtomRow, ...], ...]:
    choices = _altloc_choices(rows)
    if not choices:
        return (rows,)
    expanded: list[tuple[_AtomRow, ...]] = []
    for labels in itertools.product(*(labels for _, labels in choices)):
        selection = {residue_key: label for (residue_key, _), label in zip(choices, labels)}
        expanded.append(
            tuple(
                row
                for row in rows
                if row.altloc_id is None or selection.get(_residue_key(row), row.altloc_id) == row.altloc_id
            )
        )
    return tuple(expanded)


def _altloc_choices(
    rows: tuple[_AtomRow, ...],
) -> tuple[tuple[tuple[object, ...], tuple[str, ...]], ...]:
    labels_by_residue: dict[tuple[object, ...], list[str]] = {}
    for row in rows:
        if row.altloc_id is None:
            continue
        labels = labels_by_residue.setdefault(_residue_key(row), [])
        if row.altloc_id not in labels:
            labels.append(row.altloc_id)
    return tuple((key, tuple(labels)) for key, labels in labels_by_residue.items() if labels)


def _decode_canonical_mmcif(parsed: _ParsedInput) -> Structure:
    payload = cast(dict[str, object], parsed.canonical_payload)
    rows = tuple(sorted(parsed.rows, key=lambda row: row.row_index))
    coordinates_key = payload.get("coordinates")
    coordinate_mask_key = payload.get("coordinate_mask")
    occupancy_key = payload.get("occupancy")
    for key, name in (
        (coordinates_key, "coordinates"),
        (coordinate_mask_key, "coordinate_mask"),
    ):
        if type(key) is not str or not key:
            raise ValueError(f"canonical mmCIF payload has invalid {name} array reference")
    arrays: dict[str, np.ndarray] = {
        coordinates_key: np.asarray([row.coordinates for row in rows], dtype=np.float32),  # type: ignore[dict-item]
        coordinate_mask_key: np.asarray(
            [row.coordinate_present for row in rows], dtype=np.bool_
        ),  # type: ignore[dict-item]
    }
    if occupancy_key is not None:
        if type(occupancy_key) is not str or not occupancy_key:
            raise ValueError("canonical mmCIF payload has invalid occupancy array reference")
        arrays[occupancy_key] = np.asarray(
            [np.nan if row.occupancy is None else row.occupancy for row in rows], dtype=np.float32
        )
    # TODO(§3.5 reopened): §3.5.1 drops the b_factor carrier array; the exchange payload no
    # longer emits a b_factor reference, so no b_factor sidecar is injected here.
    structure = decode_structure(payload, arrays)
    row_sites = tuple(row.canonical_site for row in rows)
    if any(site is None for site in row_sites) or row_sites != structure.sites:
        raise ValueError("canonical mmCIF atom_site identity does not match _ofoldx_structure payload")
    for row, site in zip(rows, structure.sites):
        atom = structure.system.atom(site)
        if row.element != atom.element:
            raise ValueError(f"canonical mmCIF element disagrees with System atom at {site!r}")
    return structure


def _cif_column(category: object, name: str) -> tuple[str | None, ...]:
    if name not in category:  # type: ignore[operator]
        return (None,) * category.row_count  # type: ignore[attr-defined]
    return tuple(_present_string(value) for value in category[name].as_array())  # type: ignore[index]


def _cif_bond_order(value: str | None) -> tuple[float, bool]:
    key = None if value is None else value.lower()
    try:
        return _CIF_BOND_ORDERS[key]
    except KeyError:
        raise ValueError(f"unsupported mmCIF covalent bond order: {value!r}") from None


def _remove_component_atoms(component: ChemicalComponent, atom_ids: set[str]) -> ChemicalComponent:
    atom_by_id = {atom.atom_id: atom for atom in component.atoms}
    removed = set(atom_by_id).intersection(atom_ids)
    if not removed:
        return component
    for bond in component.bonds:
        if bond.atom_a in removed and atom_by_id[bond.atom_b].element == "H":
            removed.add(bond.atom_b)
        if bond.atom_b in removed and atom_by_id[bond.atom_a].element == "H":
            removed.add(bond.atom_a)
    for bond in component.bonds:
        if (
            bond.stereo_atoms is not None
            and removed.intersection(bond.stereo_atoms)
            and not removed.intersection((bond.atom_a, bond.atom_b))
        ):
            raise ValueError("a declared leaving atom is required to define retained bond stereochemistry")
    keep = tuple(atom.atom_id not in removed for atom in component.atoms)
    return ChemicalComponent(
        component_id=component.component_id,
        atoms=tuple(atom for atom, present in zip(component.atoms, keep) if present),
        bonds=frozenset(bond for bond in component.bonds if bond.atom_a not in removed and bond.atom_b not in removed),
        previous_attachment=component.previous_attachment,
        next_attachment=component.next_attachment,
        reference_coordinates=(
            tuple(coordinate for coordinate, present in zip(component.reference_coordinates, keep) if present)
            if component.reference_coordinates
            else ()
        ),
    )


def _read_mmcif_branch_entities(
    parsed: _ParsedInput,
    component_table: Mapping[str, ChemicalComponent],
    selected_rows: tuple[_AtomRow, ...],
) -> tuple[dict[str, Molecule], dict[tuple[object, ...], int]]:
    cif_file = parsed.mmcif_file
    if cif_file is None:
        return {}, {}
    block = cif_file.block  # type: ignore[attr-defined]
    if "pdbx_entity_branch_list" not in block:
        return {}, {}

    selected_entity_ids = {row.entity_id for row in selected_rows if row.entity_id is not None}
    category = block["pdbx_entity_branch_list"]
    entity_ids = _cif_column(category, "entity_id")
    component_ids = _cif_column(category, "comp_id")
    numbers = _cif_column(category, "num")
    declarations: dict[str, dict[int, str]] = {}
    for entity_id, component_id, number in zip(entity_ids, component_ids, numbers):
        if entity_id is None or entity_id not in selected_entity_ids:
            continue
        component_index = _optional_int(number)
        if component_id is None or component_index is None:
            raise ValueError("mmCIF branched-entity components require entity_id, comp_id, and num")
        entity_components = declarations.setdefault(entity_id, {})
        if component_index in entity_components:
            raise ValueError(f"duplicate branched-entity component number {component_index} for entity {entity_id!r}")
        entity_components[component_index] = component_id

    number_to_index = {
        entity_id: {number: index for index, number in enumerate(sorted(components))}
        for entity_id, components in declarations.items()
    }
    removed_atoms: dict[tuple[str, int], set[str]] = {}
    bonds: dict[str, set[MoleculeBond]] = {entity_id: set() for entity_id in declarations}
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
        orders = _cif_column(links, "value_order")
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
                continue
            first_number_int = _optional_int(first_number)
            second_number_int = _optional_int(second_number)
            if first_number_int is None or second_number_int is None or first_atom is None or second_atom is None:
                raise ValueError("mmCIF branched-entity links require two component numbers and atom IDs")
            declared = declarations[entity_id]
            if declared.get(first_number_int) != first_component or declared.get(second_number_int) != second_component:
                raise ValueError(f"branched-entity link disagrees with component declaration for entity {entity_id!r}")
            first_index = number_to_index[entity_id][first_number_int]
            second_index = number_to_index[entity_id][second_number_int]
            order, aromatic = _cif_bond_order(order_name)
            bonds[entity_id].add(
                MoleculeBond(first_index, first_atom, second_index, second_atom, order=order, aromatic=aromatic)
            )
            if first_leaving is not None:
                removed_atoms.setdefault((entity_id, first_index), set()).add(first_leaving)
            if second_leaving is not None:
                removed_atoms.setdefault((entity_id, second_index), set()).add(second_leaving)

    entities: dict[str, Molecule] = {}
    for entity_id, numbered_components in declarations.items():
        components: list[ChemicalComponent] = []
        for component_index, number in enumerate(sorted(numbered_components)):
            component_id = numbered_components[number]
            try:
                component = component_table[component_id]
            except KeyError:
                raise ValueError(
                    f"unknown component {component_id!r}; pass components={{...}} with its complete declared graph"
                ) from None
            components.append(
                _remove_component_atoms(component, removed_atoms.get((entity_id, component_index), set()))
            )
        entities[entity_id] = Molecule(
            components=tuple(components),
            bonds=frozenset(bonds[entity_id]),
        )

    component_indices: dict[tuple[object, ...], int] = {}
    if "pdbx_branch_scheme" in block:
        scheme = block["pdbx_branch_scheme"]
        asym_ids = _cif_column(scheme, "asym_id")
        scheme_entities = _cif_column(scheme, "entity_id")
        monomer_ids = _cif_column(scheme, "mon_id")
        scheme_numbers = _cif_column(scheme, "num")
        pdb_asym_ids = _cif_column(scheme, "pdb_asym_id")
        pdb_monomer_ids = _cif_column(scheme, "pdb_mon_id")
        pdb_sequence_numbers = _cif_column(scheme, "pdb_seq_num")
        auth_asym_ids = _cif_column(scheme, "auth_asym_id")
        auth_monomer_ids = _cif_column(scheme, "auth_mon_id")
        auth_sequence_numbers = _cif_column(scheme, "auth_seq_num")
        for values in zip(
            asym_ids,
            scheme_entities,
            monomer_ids,
            scheme_numbers,
            pdb_asym_ids,
            pdb_monomer_ids,
            pdb_sequence_numbers,
            auth_asym_ids,
            auth_monomer_ids,
            auth_sequence_numbers,
        ):
            (
                asym_id,
                entity_id,
                monomer_id,
                number,
                pdb_asym_id,
                pdb_monomer_id,
                pdb_sequence_number,
                auth_asym_id,
                auth_monomer_id,
                auth_sequence_number,
            ) = values
            if entity_id not in entities:
                continue
            number_int = _optional_int(number)
            if number_int is None:
                raise ValueError("mmCIF branch scheme requires an integer component number")
            component_index = number_to_index[entity_id][number_int]
            source_monomer_ids = {monomer_id, pdb_monomer_id, auth_monomer_id} - {None}
            source_positions = {
                (source_asym_id, source_sequence_number)
                for source_asym_id, source_sequence_number in (
                    (pdb_asym_id, _optional_int(pdb_sequence_number)),
                    (auth_asym_id, _optional_int(auth_sequence_number)),
                )
                if source_asym_id is not None and source_sequence_number is not None
            }
            for source_monomer_id in source_monomer_ids:
                for source_asym_id, source_sequence_number in source_positions:
                    key = (
                        asym_id,
                        entity_id,
                        source_monomer_id,
                        source_asym_id,
                        source_sequence_number,
                    )
                    previous = component_indices.get(key)
                    if previous is not None and previous != component_index:
                        raise ValueError(f"ambiguous mmCIF branch-scheme identity: {key!r}")
                    component_indices[key] = component_index
    return entities, component_indices


def _build_structure(
    parsed: _ParsedInput,
    rows: tuple[_AtomRow, ...],
    atom_layout: str,
    component_table: dict[str, ChemicalComponent],
) -> Structure:
    if not rows:
        raise ValueError("alternate-location selection contains no atoms")
    residue_rows: dict[tuple[object, ...], list[_AtomRow]] = {}
    chain_residue_order: dict[tuple[str | None, str], list[tuple[object, ...]]] = {}
    for row in rows:
        residue_key = _residue_key(row)
        residue_rows.setdefault(residue_key, []).append(row)
        chain_key = (row.symmetry_id, row.chain_id)
        order = chain_residue_order.setdefault(chain_key, [])
        if residue_key not in order:
            order.append(residue_key)

    for residue_key, residue in residue_rows.items():
        component_id = residue[0].component_id
        if component_id not in component_table:
            raise ValueError(
                f"unknown component {component_id!r}; pass components={{...}} with its complete declared graph"
            )
        source_atom_names = {row.atom_name for row in residue}
        component_atom_names = {atom.atom_id for atom in component_table[component_id].atoms}
        unknown_atoms = source_atom_names - component_atom_names
        if unknown_atoms:
            raise ValueError(
                f"source residue {residue_key!r} contains atoms absent from component {component_id!r}: "
                f"{sorted(unknown_atoms)!r}"
            )

    branch_entities, branch_component_indices = _read_mmcif_branch_entities(parsed, component_table, rows)
    # TODO(§3.5 reopened): the polymer/non-polymer partition and single polymer-type
    # derivation are source-driven and reopened; this reproduces the prior per-component
    # behavior from the importer-internal audited classification (`_component_polymer_type`).
    groups: list[
        tuple[
            int,
            str,
            str,
            PolymerSequence | Molecule,
            tuple[tuple[int, tuple[object, ...]], ...],
        ]
    ] = []
    used_instance_ids: set[str] = set()
    entity_by_id: dict[str, PolymerSequence | Molecule] = {}

    for (symmetry_id, chain_id), residue_order in chain_residue_order.items():
        base_instance_id = chain_id if symmetry_id is None else f"{chain_id}#{symmetry_id}"
        polymer_residues = [
            key for key in residue_order if _component_polymer_type(residue_rows[key][0].component_id) is not None
        ]
        if polymer_residues:
            polymer_types = {_component_polymer_type(residue_rows[key][0].component_id) for key in polymer_residues}
            if len(polymer_types) != 1:
                raise ValueError(f"polymer chain {base_instance_id!r} has ambiguous polymer type declarations")
            polymer_type = next(iter(polymer_types))
            first = residue_rows[polymer_residues[0]][0]
            source_entity_id = first.entity_id
            entity_id = (
                f"polymer_{source_entity_id}"
                if parsed.format == "mmcif" and source_entity_id is not None
                else f"polymer_{chain_id}"
            )
            entity = _reuse_entity(
                entity_by_id,
                entity_id,
                PolymerSequence(
                    polymer_type,  # type: ignore[arg-type]
                    tuple(component_table[residue_rows[key][0].component_id] for key in polymer_residues),
                ),
            )
            instance_id = _unique_id(base_instance_id, used_instance_ids)
            groups.append(
                (
                    min(row.row_index for key in polymer_residues for row in residue_rows[key]),
                    instance_id,
                    entity_id,
                    entity,
                    tuple(enumerate(polymer_residues)),
                )
            )

        nonpolymer_residues = [
            key for key in residue_order if _component_polymer_type(residue_rows[key][0].component_id) is None
        ]
        molecule_groups: list[tuple[tuple[object, ...], ...]] = []
        if parsed.format == "mmcif":
            grouped_by_entity: dict[str, list[tuple[object, ...]]] = {}
            for residue_key in nonpolymer_residues:
                source_entity_id = residue_rows[residue_key][0].entity_id
                if source_entity_id in branch_entities:
                    grouped_by_entity.setdefault(source_entity_id, []).append(residue_key)
                else:
                    molecule_groups.append((residue_key,))
            molecule_groups.extend(tuple(group) for group in grouped_by_entity.values())
            molecule_groups.sort(key=lambda group: min(row.row_index for key in group for row in residue_rows[key]))
        else:
            molecule_groups.extend((residue_key,) for residue_key in nonpolymer_residues)

        for group_residues in molecule_groups:
            first = residue_rows[group_residues[0]][0]
            first_component = component_table[first.component_id]
            residue_label = first.auth_residue_id if first.auth_residue_id is not None else first.residue_id
            residue_identity = f"{residue_label}{first.insertion_code or ''}"
            source_entity_id = first.entity_id
            if source_entity_id in branch_entities:
                entity = branch_entities[source_entity_id]
                entity_id = f"molecule_{source_entity_id}"
                instance_seed = base_instance_id
                indexed_residues: list[tuple[int, tuple[object, ...]]] = []
                for residue_key in group_residues:
                    row = residue_rows[residue_key][0]
                    scheme_key = (
                        row.chain_id,
                        row.entity_id,
                        row.component_id,
                        row.auth_chain_id,
                        row.auth_residue_id,
                    )
                    component_index = branch_component_indices.get(scheme_key)
                    if component_index is None and row.auth_component_id is not None:
                        component_index = branch_component_indices.get(
                            (
                                row.chain_id,
                                row.entity_id,
                                row.auth_component_id,
                                row.auth_chain_id,
                                row.auth_residue_id,
                            )
                        )
                    if component_index is None:
                        raise ValueError(f"cannot map branched-entity residue {residue_key!r} to its declaration")
                    indexed_residues.append((component_index, residue_key))
                if len({index for index, _ in indexed_residues}) != len(indexed_residues):
                    raise ValueError(f"branched entity {source_entity_id!r} maps multiple residues to one component")
                indexed_residues.sort()
            elif parsed.format == "mmcif" and source_entity_id is not None:
                entity = Molecule((first_component,))
                entity_id = f"molecule_{source_entity_id}"
                instance_seed = f"{base_instance_id}:{residue_identity}:{first_component.component_id}"
                indexed_residues = [(0, group_residues[0])]
            else:
                entity = Molecule((first_component,))
                entity_id = f"molecule_{chain_id}:{residue_identity}:{first_component.component_id}"
                instance_seed = f"{base_instance_id}:{residue_identity}:{first_component.component_id}"
                indexed_residues = [(0, group_residues[0])]
            entity = _reuse_entity(entity_by_id, entity_id, entity)
            instance_id = _unique_id(instance_seed, used_instance_ids)
            groups.append(
                (
                    min(row.row_index for key in group_residues for row in residue_rows[key]),
                    instance_id,
                    entity_id,
                    entity,
                    tuple(indexed_residues),
                )
            )

    groups.sort(key=lambda item: item[0])
    instances = {instance_id: entity_id for _, instance_id, entity_id, _, _ in groups}
    provisional = MolecularSystem(entity_by_id, instances)

    site_by_row_index: dict[int, AtomSite] = {}
    # TODO(§3.5 reopened): the residue_numbers population scope and label-vs-author source
    # category are reopened; populate polymer occurrences with the author-else-label number.
    residue_numbers: dict[ComponentReference, ResidueNumber] = {}
    for _, instance_id, _, group_entity, group_indexed in groups:
        is_polymer = type(group_entity) is PolymerSequence
        for component_index, residue_key in group_indexed:
            residue = residue_rows[residue_key]
            if is_polymer:
                first = residue[0]
                number = first.auth_residue_id if first.auth_residue_id is not None else first.residue_id
                if number is not None:
                    residue_numbers[(instance_id, component_index)] = ResidueNumber(number, first.insertion_code or "")
            for row in residue:
                site_by_row_index[row.row_index] = AtomSite(instance_id, component_index, row.atom_name)

    for row in rows:
        site = site_by_row_index[row.row_index]
        atom = provisional.atom(site)
        if atom.element != row.element:
            raise ValueError(
                f"source element {row.element!r} disagrees with component atom {atom.element!r} at {site!r}"
            )
        if row.charge is not None and row.charge != atom.formal_charge:
            raise ValueError(
                f"source formal charge {row.charge!r} disagrees with component atom "
                f"{atom.formal_charge!r} at {site!r}"
            )

    source_bonds = _read_source_bonds(parsed, rows, site_by_row_index, provisional)
    system = MolecularSystem(entity_by_id, instances, frozenset(source_bonds))
    observed_sites = tuple(site_by_row_index[row.row_index] for row in rows)
    if len(observed_sites) != len(set(observed_sites)):
        raise ValueError("selected source rows map more than once to one canonical AtomSite")
    # TODO(§3.5 reopened): the atom_layout site-materialization knob is reopened; heavy
    # realization is an inline element filter over iter_atoms(), not a system-level API.
    if atom_layout == "observed":
        sites = observed_sites
    elif atom_layout == "complete_all":
        sites = tuple(site for site, _ in system.iter_atoms())
    else:
        sites = tuple(site for site, atom in system.iter_atoms() if atom.element != "H")
    site_indices = {site: index for index, site in enumerate(sites)}
    coordinates = np.zeros((len(sites), 3), dtype=np.float32)
    coordinate_mask = np.zeros(len(sites), dtype=np.bool_)
    occupancy = (
        np.full(len(sites), np.nan, dtype=np.float32) if any(row.occupancy is not None for row in rows) else None
    )
    for row in rows:
        site = site_by_row_index[row.row_index]
        index = site_indices[site]
        coordinates[index] = row.coordinates
        coordinate_mask[index] = row.coordinate_present
        if occupancy is not None and row.occupancy is not None:
            occupancy[index] = row.occupancy
    return Structure(
        system=system,
        sites=sites,
        coordinates=coordinates,
        coordinate_mask=coordinate_mask,
        occupancy=occupancy,
        residue_numbers=residue_numbers or None,
    )


def _reuse_entity(
    entities: dict[str, PolymerSequence | Molecule],
    entity_id: str,
    entity: PolymerSequence | Molecule,
) -> PolymerSequence | Molecule:
    previous = entities.get(entity_id)
    if previous is None:
        entities[entity_id] = entity
        return entity
    if type(previous) is not type(entity) or previous != entity:
        raise ValueError(f"source entity ID {entity_id!r} has conflicting molecular definitions")
    return previous


def _unique_id(seed: str, used: set[str]) -> str:
    candidate = seed or "_"
    suffix = 2
    while candidate in used:
        candidate = f"{seed}_{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def _read_source_bonds(
    parsed: _ParsedInput,
    selected_rows: tuple[_AtomRow, ...],
    site_by_row_index: dict[int, AtomSite],
    system: MolecularSystem,
) -> tuple[SystemBond, ...]:
    if parsed.format == "pdb":
        declared = _read_pdb_bonds(parsed, selected_rows, site_by_row_index)
    else:
        declared = _read_mmcif_bonds(parsed, selected_rows, site_by_row_index)
    base_bonds = {frozenset((bond.site_a, bond.site_b)): bond for bond in system.iter_bonds()}
    overlay: dict[frozenset[AtomSite], SystemBond] = {}
    for bond in declared:
        endpoints = frozenset((bond.site_a, bond.site_b))
        base = base_bonds.get(endpoints)
        if base is not None:
            if (bond.order, bond.aromatic) != (base.order, base.aromatic):
                raise ValueError(
                    f"source bond conflicts with declared intrinsic topology: " f"{bond.site_a!r}, {bond.site_b!r}"
                )
            continue
        previous = overlay.get(endpoints)
        if previous is not None:
            if (previous.order, previous.aromatic, previous.stereo) != (bond.order, bond.aromatic, bond.stereo):
                raise ValueError(f"conflicting source bond declarations: {bond.site_a!r}, {bond.site_b!r}")
            continue
        overlay[endpoints] = bond
    return tuple(overlay.values())


def _selected_source_sites(
    selected_rows: tuple[_AtomRow, ...], site_by_row_index: dict[int, AtomSite]
) -> dict[tuple[object, ...], AtomSite]:
    mapping: dict[tuple[object, ...], AtomSite] = {}
    ambiguous: set[tuple[object, ...]] = set()
    for row in selected_rows:
        site = site_by_row_index[row.row_index]
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
                continue
            mapping[key] = site
    return mapping


def _read_pdb_bonds(
    parsed: _ParsedInput,
    selected_rows: tuple[_AtomRow, ...],
    site_by_row_index: dict[int, AtomSite],
) -> tuple[SystemBond, ...]:
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
    source_row_by_serial: dict[tuple[str | None, str], _AtomRow] = {}
    for row in parsed.rows:
        if row.serial is None:
            continue
        key = (row.symmetry_id, row.serial)
        previous = source_row_by_serial.get(key)
        if previous is not None and _residue_key(previous) != _residue_key(row):
            raise ValueError(f"PDB atom serial {row.serial!r} is ambiguous")
        source_row_by_serial[key] = row
    selected_sites = _selected_source_sites(selected_rows, site_by_row_index)
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
            site_a = selected_sites.get(_row_source_key(row_a))
            site_b = selected_sites.get(_row_source_key(row_b))
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


def _row_source_key(row: _AtomRow) -> tuple[object, ...]:
    return (
        row.symmetry_id,
        row.chain_id,
        row.residue_id,
        row.insertion_code,
        row.component_id,
        row.atom_name,
    )


def _read_mmcif_bonds(
    parsed: _ParsedInput,
    selected_rows: tuple[_AtomRow, ...],
    site_by_row_index: dict[int, AtomSite],
) -> tuple[SystemBond, ...]:
    cif_file = parsed.mmcif_file
    if cif_file is None:
        return ()
    block = cif_file.block  # type: ignore[attr-defined]
    if "struct_conn" not in block:
        return ()
    category = block["struct_conn"]
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
    ins_1 = _cif_column(category, "pdbx_ptnr1_PDB_ins_code")
    ins_2 = _cif_column(category, "pdbx_ptnr2_PDB_ins_code")
    symmetry_1 = _cif_column(category, "ptnr1_symmetry")
    symmetry_2 = _cif_column(category, "ptnr2_symmetry")
    value_orders = _cif_column(category, "pdbx_value_order")
    selected_sites = _selected_source_sites(selected_rows, site_by_row_index)
    symmetry_ids = {row.symmetry_id for row in selected_rows}
    bonds: list[SystemBond] = []
    for index, conn_type in enumerate(conn_types):
        if conn_type is None or not (conn_type.startswith("covale") or conn_type.startswith("disulf")):
            continue
        if symmetry_1[index] not in {None, "1_555"} or symmetry_2[index] not in {None, "1_555"}:
            continue
        order, aromatic = _cif_bond_order(value_orders[index])
        endpoint_1 = (
            label_asym_1[index],
            _optional_int(label_seq_1[index]),
            ins_1[index],
            label_comp_1[index],
            label_atom_1[index],
        )
        endpoint_2 = (
            label_asym_2[index],
            _optional_int(label_seq_2[index]),
            ins_2[index],
            label_comp_2[index],
            label_atom_2[index],
        )
        auth_endpoint_1 = (
            auth_asym_1[index] or label_asym_1[index],
            _optional_int(auth_seq_1[index] or label_seq_1[index]),
            ins_1[index],
            auth_comp_1[index] or label_comp_1[index],
            auth_atom_1[index] or label_atom_1[index],
        )
        auth_endpoint_2 = (
            auth_asym_2[index] or label_asym_2[index],
            _optional_int(auth_seq_2[index] or label_seq_2[index]),
            ins_2[index],
            auth_comp_2[index] or label_comp_2[index],
            auth_atom_2[index] or label_atom_2[index],
        )
        for symmetry_id in symmetry_ids:
            site_a = selected_sites.get((symmetry_id, *endpoint_1)) or selected_sites.get(
                (symmetry_id, *auth_endpoint_1)
            )
            site_b = selected_sites.get((symmetry_id, *endpoint_2)) or selected_sites.get(
                (symmetry_id, *auth_endpoint_2)
            )
            if site_a is None or site_b is None:
                raise ValueError(
                    "mmCIF struct_conn references an atom absent from the selected assembly/model: "
                    f"label={endpoint_1!r}, {endpoint_2!r}; auth={auth_endpoint_1!r}, {auth_endpoint_2!r}"
                )
            bonds.append(
                SystemBond(
                    site_a,
                    site_b,
                    order=order,
                    aromatic=aromatic,
                    origin="source",
                )
            )
    return tuple(bonds)


def _write_mmcif(structure: Structure, path: Path) -> None:
    from biotite.structure.io.pdbx import CIFBlock, CIFCategory, CIFFile

    payload, _ = encode_structure(structure, array_prefix="atom_site")
    # TODO(§3.5 reopened): §3.5.1 removes source-identity provenance; the writer derives every
    # identity column from the canonical system and residue_numbers, emitting a single conformer
    # and a placeholder B column.
    residue_numbers = structure.residue_numbers or {}
    label_asym_by_instance: dict[str, str] = {}
    used_label_asym_ids: set[str] = set()
    for instance_id in structure.system.instances:
        label_asym_by_instance[instance_id] = _unique_id(instance_id, used_label_asym_ids)
    columns: dict[str, list[object]] = {
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
        "ofoldx_instance_id": [],
        "ofoldx_component_index": [],
        "ofoldx_atom_id": [],
        "ofoldx_coordinate_mask": [],
    }
    endpoint_identity: dict[AtomSite, tuple[str, str, str, str, str, str, str, str]] = {}
    for index, site in enumerate(structure.sites):
        atom = structure.system.atom(site)
        entity_id = structure.system.instances[site.instance_id]
        entity = structure.system.entities[entity_id]
        component = structure.system.component((site.instance_id, site.component_index))
        component_id = component.component_id
        residue_number = residue_numbers.get((site.instance_id, site.component_index))
        if type(entity) is PolymerSequence:
            group = "ATOM"
            label_seq_id: object = site.component_index + 1
        else:
            group = "HETATM"
            label_seq_id = "."
        label_entity_id = entity_id
        label_asym_id = label_asym_by_instance[site.instance_id]
        label_component_id = component_id
        label_atom_id = site.atom_id
        auth_asym_id = label_asym_id
        auth_seq_id: object = residue_number.number if residue_number is not None else site.component_index + 1
        auth_component_id = label_component_id
        auth_atom_id = label_atom_id
        insertion_code = (
            residue_number.insertion_code if residue_number is not None and residue_number.insertion_code else "?"
        )
        coordinate = structure.coordinates[index]
        occupancy = (
            "."
            if structure.occupancy is None or np.isnan(structure.occupancy[index])
            else float(structure.occupancy[index])
        )
        b_factor = "."
        altloc = "."
        values = {
            "group_PDB": group,
            "id": index + 1,
            "type_symbol": atom.element,
            "label_atom_id": label_atom_id,
            "label_alt_id": altloc,
            "label_comp_id": label_component_id,
            "label_asym_id": label_asym_id,
            "label_entity_id": label_entity_id,
            "label_seq_id": label_seq_id,
            "pdbx_PDB_ins_code": insertion_code,
            "Cartn_x": float(coordinate[0]),
            "Cartn_y": float(coordinate[1]),
            "Cartn_z": float(coordinate[2]),
            "occupancy": occupancy,
            "B_iso_or_equiv": b_factor,
            "pdbx_formal_charge": atom.formal_charge,
            "auth_seq_id": auth_seq_id,
            "auth_comp_id": auth_component_id,
            "auth_asym_id": auth_asym_id,
            "auth_atom_id": auth_atom_id,
            "pdbx_PDB_model_num": 1,
            "ofoldx_instance_id": site.instance_id,
            "ofoldx_component_index": site.component_index,
            "ofoldx_atom_id": site.atom_id,
            "ofoldx_coordinate_mask": 1 if bool(structure.coordinate_mask[index]) else 0,
        }
        for name, value in values.items():
            columns[name].append(value)
        endpoint_identity[site] = (
            str(label_asym_id),
            str(label_seq_id),
            str(label_component_id),
            str(label_atom_id),
            str(auth_asym_id),
            str(auth_seq_id),
            str(auth_component_id),
            str(auth_atom_id),
        )

    categories: dict[str, object] = {
        "atom_site": CIFCategory(columns),
        "ofoldx_structure": CIFCategory(
            {"json": [json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)]}
        ),
    }
    all_bonds = _output_bonds(structure.system)
    if all_bonds:
        conn_columns: dict[str, list[object]] = {
            "id": [],
            "conn_type_id": [],
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
            "pdbx_value_order": [],
        }
        order_names = {1.0: "sing", 1.5: "arom", 2.0: "doub", 3.0: "trip", 4.0: "quad"}
        for index, bond in enumerate(all_bonds, 1):
            if bond.site_a not in endpoint_identity or bond.site_b not in endpoint_identity:
                continue
            first = endpoint_identity[bond.site_a]
            second = endpoint_identity[bond.site_b]
            values = (
                index,
                "covale",
                *first[:4],
                *second[:4],
                *first[4:],
                *second[4:],
                order_names[bond.order],
            )
            for name, value in zip(conn_columns, values):
                conn_columns[name].append(value)
        if conn_columns["id"]:
            categories["struct_conn"] = CIFCategory(conn_columns)
    cif_file = CIFFile({"ofoldx": CIFBlock(categories)})
    cif_file.write(str(path))


def _output_bonds(system: MolecularSystem) -> tuple[SystemBond, ...]:
    bonds: list[SystemBond] = []
    for bond in system.iter_bonds():
        key_a = (bond.site_a.instance_id, bond.site_a.component_index, bond.site_a.atom_id)
        key_b = (bond.site_b.instance_id, bond.site_b.component_index, bond.site_b.atom_id)
        if key_b < key_a:
            bond = SystemBond(
                bond.site_b,
                bond.site_a,
                order=bond.order,
                aromatic=bond.aromatic,
                stereo=bond.stereo,
                stereo_sites=None if bond.stereo_sites is None else bond.stereo_sites[::-1],
                origin=bond.origin,
            )
        bonds.append(bond)
    return tuple(
        sorted(
            bonds,
            key=lambda bond: (
                bond.site_a.instance_id,
                bond.site_a.component_index,
                bond.site_a.atom_id,
                bond.site_b.instance_id,
                bond.site_b.component_index,
                bond.site_b.atom_id,
            ),
        )
    )


def _pdb_loss_report(structure: Structure) -> LossReport:
    # TODO(§3.5 reopened): the full LossReport contract (source-identity round-trip
    # accounting) is reopened under §3.5; this reports the losses still derivable from
    # the committed six-field carrier and dict-based MolecularSystem.
    atoms, dropped, rewritten = _pdb_atoms(structure)
    system = structure.system
    written_sites = {atom.site for atom in atoms}
    atoms_by_instance: dict[str, list[_PdbAtom]] = {}
    for atom in atoms:
        atoms_by_instance.setdefault(atom.site.instance_id, []).append(atom)
    for instance_id, entity_id in system.instances.items():
        instance_atoms = atoms_by_instance.get(instance_id, [])
        if not instance_atoms:
            dropped.add("system.instances")
            continue
        entity = system.entities[entity_id]
        if type(entity) is PolymerSequence:
            observed_components = {atom.site.component_index for atom in instance_atoms}
            if observed_components != set(range(len(entity))):
                dropped.add("system.polymer_sequence")
    if any(bond.site_a not in written_sites or bond.site_b not in written_sites for bond in system.bonds):
        dropped.add("system.bonds")
    if any(bond.origin != "source" for bond in system.bonds):
        rewritten.add("system.bond_origins")
    if any(type(entity) is Molecule for entity in system.entities.values()):
        dropped.add("system.nonpolymer_graph")
    polymer_components = {
        component.component_id: component
        for entity in system.entities.values()
        if type(entity) is PolymerSequence
        for component in entity.components
    }
    for component in polymer_components.values():
        try:
            audited = get_component(component.component_id)
        except KeyError:
            dropped.add("system.nonstandard_components")
            break
        if audited != component:
            dropped.add("system.nonstandard_components")
            break
    if any(bond.stereo is not None or bond.aromatic or bond.order not in {1.0, 2.0, 3.0, 4.0} for bond in system.bonds):
        dropped.add("system.bond_semantics")
    return LossReport(tuple(sorted(dropped)), tuple(sorted(rewritten)))


def _pdb_atoms(structure: Structure) -> tuple[tuple[_PdbAtom, ...], set[str], set[str]]:
    # TODO(§3.5 reopened): §3.5.1 removes source-identity provenance; PDB columns are derived
    # from the committed system + residue_numbers, with a single conformer and a placeholder B.
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

    atoms: list[_PdbAtom] = []
    names_by_residue: dict[tuple[str, int, str], set[str]] = {}
    for site_index, site in enumerate(structure.sites):
        if not bool(structure.coordinate_mask[site_index]):
            dropped.add("structure.coordinate_mask")
            continue
        if len(atoms) >= 99999:
            raise ValueError("PDB atom serial limit of 99999 was exceeded")
        entity = structure.system.entities[structure.system.instances[site.instance_id]]
        component = structure.system.component((site.instance_id, site.component_index))
        canonical_component_id = component.component_id
        canonical_residue_id = site.component_index + 1
        residue_number = residue_numbers.get((site.instance_id, site.component_index))
        if type(entity) is PolymerSequence:
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
        coordinate = tuple(float(value) for value in structure.coordinates[site_index])
        if any(value < -999.999 or value > 9999.999 for value in coordinate):
            raise ValueError("PDB coordinate field range was exceeded")
        rounded_coordinate = tuple(round(value, 3) for value in coordinate)
        if not np.array_equal(np.asarray(rounded_coordinate, dtype=np.float32), structure.coordinates[site_index]):
            rewritten.add("structure.coordinates")
        if structure.occupancy is None or np.isnan(structure.occupancy[site_index]):
            occupancy = 1.0
            rewritten.add("structure.occupancy")
        else:
            occupancy = float(structure.occupancy[site_index])
            if np.float32(round(occupancy, 2)) != structure.occupancy[site_index]:
                rewritten.add("structure.occupancy")
        b_factor = 0.0
        chemical_atom = structure.system.atom(site)
        charge = chemical_atom.formal_charge or 0
        if abs(charge) > 9:
            dropped.add("system.formal_charge")
            charge = 0
        atoms.append(
            _PdbAtom(
                site=site,
                serial=len(atoms) + 1,
                group=group,
                atom_name=atom_name,
                altloc_id=altloc_id,
                component_id=component_id,
                chain_id=chain_by_instance[site.instance_id],
                residue_id=residue_id,
                insertion_code=insertion_code,
                coordinates=rounded_coordinate,  # type: ignore[arg-type]
                occupancy=round(occupancy, 2),
                b_factor=round(b_factor, 2),
                element=chemical_atom.element,
                charge=charge,
            )
        )
    return tuple(atoms), dropped, rewritten


def _write_pdb(structure: Structure, path: Path) -> None:
    atoms, _, _ = _pdb_atoms(structure)
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
    for bond in _output_bonds(structure.system):
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
