r"""PDB and mmCIF I/O for the canonical tensor molecular graph."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from pathlib import Path
from typing import AbstractSet, Literal

import numpy as np
import torch

from ..data.ccd import resolve_ccd_component
from ..data.components import get_builtin_component_polymer_type, get_component
from ..data.elements import ATOMIC_NUMBER_BY_ELEMENT
from ..data.molecular import (
    HYDROGEN_ELEMENTS,
    AtomSite,
    ChemicalAtom,
    ChemicalComponent,
    ComponentReference,
    MolecularSystem,
    Molecule,
    PolymerSequence,
    PolymerType,
    is_water_component,
)
from ..data.structure import Structure
from .records import atom_row_residue_key, unique_identifier

_FORMATS: dict[str, Literal["pdb", "mmcif"]] = {".pdb": "pdb", ".cif": "mmcif", ".mmcif": "mmcif"}
_ALTLOC_POLICIES = frozenset({"highest_occupancy", "first"})


class NoRetainedStructureError(ValueError):
    """The ingestion policy intentionally removed every source occurrence."""


def read_structure(
    path: str | os.PathLike[str],
    *,
    model: int = 1,
    assembly_id: str | None = None,
    altloc: Literal["highest_occupancy", "first"] = "highest_occupancy",
    include_water: bool = False,
    include_hydrogens: bool = False,
    components: Mapping[str, ChemicalComponent] | None = None,
    ccd_revision: str | None = None,
    ccd_snapshot_path: str | None = None,
    generator: torch.Generator | None = None,
) -> Structure:
    r"""
    Read one model and alternate-location realization from PDB or mmCIF.

    Args:
        path: Input PDB or mmCIF path, optionally gzip-compressed.
        model: One-based model number.
        assembly_id: Biological assembly identifier, or `None` for the stored
            asymmetric unit.
        altloc: Residue-level alternate-location choice.
        include_water: Whether to retain water entities.
        include_hydrogens: Whether to retain hydrogen atoms in the molecular graph.
        components: Optional component definitions overriding CCD chemistry.
        ccd_revision: Optional pinned Hub CCD revision.
        ccd_snapshot_path: Optional local resolver-schema CCD snapshot directory.
        generator: Optional CPU random generator used only to realize legal mmCIF
            polymer microheterogeneity. By default the ambient Torch CPU RNG is used.

    Returns:
        A CPU tensor Structure whose atom axis is the complete retained system graph.
    """

    source, format_name = _structure_path(path, allow_compressed=True)
    if model <= 0:
        raise ValueError("model must be a positive one-based integer")
    if altloc not in _ALTLOC_POLICIES:
        raise ValueError(f"unsupported altloc policy: {altloc!r}")
    if generator is not None and torch.device(generator.device).type != "cpu":
        raise ValueError("read_structure generator must be a CPU torch.Generator")
    if ccd_revision is not None and ccd_snapshot_path is not None:
        raise ValueError("configure at most one of ccd_revision or ccd_snapshot_path")

    cif_file: object | None = None
    declared_component_ids: set[str] = set()
    pdb_lines: tuple[str, ...] = ()
    pdb_missing_residues: tuple[tuple[str, str, int, str], ...] = ()
    if format_name == "mmcif":
        from .mmcif import parse_mmcif

        source_rows, cif_file = parse_mmcif(source, model=model, assembly_id=assembly_id)
        from .mmcif import read_mmcif_declared_component_ids

        declared_component_ids = read_mmcif_declared_component_ids(cif_file)
    else:
        from .pdb import parse_pdb

        source_rows, pdb_lines, pdb_missing_residues = parse_pdb(
            source,
            model=model,
            assembly_id=assembly_id,
        )
    if not len(source_rows):
        raise ValueError("structure source contains no atoms for the requested model/assembly")

    explicit_components = {} if components is None else dict(components)
    for component_id, component in explicit_components.items():
        if component.component_id != component_id:
            raise ValueError("component mapping keys must equal ChemicalComponent.component_id")
    rows = _normalize_source_rows(source_rows, explicit_component_ids=set(explicit_components))
    if not len(rows):
        raise NoRetainedStructureError("structure contains only excluded UNX/UNL occurrences")
    source_asym_ids = (
        frozenset(str(chain_id) for chain_id in rows.chain_id)
        if format_name == "mmcif"
        else frozenset()
    )
    if not include_hydrogens:
        rows = rows[
            np.isin(
                rows.element,
                tuple(HYDROGEN_ELEMENTS),
                invert=True,
            )
        ].view(np.recarray)
        explicit_components = {
            component_id: component.without_hydrogens()
            for component_id, component in explicit_components.items()
        }
    if not len(rows):
        raise NoRetainedStructureError("structure contains no heavy atoms after hydrogen filtering")
    component_table: dict[str, ChemicalComponent] = dict(explicit_components)
    component_ids = {str(value) for value in rows.component_id} | declared_component_ids
    component_ids -= {component_id for component_id in {"UNX", "UNL"} if component_id not in explicit_components}
    for component_id in component_ids - set(component_table):
        try:
            if ccd_revision is not None or ccd_snapshot_path is not None:
                component_table[component_id] = resolve_ccd_component(
                    component_id,
                    revision=ccd_revision,
                    snapshot_path=ccd_snapshot_path,
                )
            else:
                component_table[component_id] = get_component(component_id)
            if not include_hydrogens:
                component_table[component_id] = component_table[component_id].without_hydrogens()
        except KeyError:
            pass
    observed_only_component_ids = _add_observed_components(rows, component_table)
    rows = _filter_source_component_atoms(rows, component_table, observed_only_component_ids)
    if format_name == "mmcif":
        assert cif_file is not None
        row_count_before_unknown_filter = len(rows)
        rows = _remove_all_unknown_protein_chains(rows, cif_file)
        if row_count_before_unknown_filter and not len(rows):
            raise NoRetainedStructureError("structure contains only excluded all-UNK protein chains")
    if not include_water:
        rows = _exclude_water_rows(format_name, rows, cif_file, component_table)
        if not len(rows):
            raise NoRetainedStructureError("structure contains only water excluded by the ingestion policy")
    component_table = {
        component_id: (
            component
            if component_id in explicit_components
            else _without_component_topology(component)
        )
        for component_id, component in component_table.items()
    }
    if format_name == "mmcif":
        assert cif_file is not None
        from .mmcif import _read_mmcif_entity_declarations

        _, polymer_monomers, _, _, _, _ = _read_mmcif_entity_declarations(cif_file)
        rows = _select_mmcif_polymer_microheterogeneity(rows, polymer_monomers, generator=generator)
    rows = _select_altloc(rows, altloc)
    policy_removed_asym_ids = source_asym_ids - frozenset(str(chain_id) for chain_id in rows.chain_id)
    return _build_structure(
        format_name,
        rows,
        source_rows,
        component_table,
        include_water=include_water,
        include_hydrogens=include_hydrogens,
        include_declared_instances=assembly_id is None,
        policy_removed_asym_ids=policy_removed_asym_ids,
        cif_file=cif_file,
        pdb_lines=pdb_lines,
        pdb_missing_residues=pdb_missing_residues,
    )


def write_structure(structure: Structure, path: str | os.PathLike[str]) -> None:
    r"""Write a Structure through the standard mmCIF or legacy PDB projection."""

    if structure.system.system_entity_ptr.numel() != 2 or structure.system.system_instance_ptr.numel() != 2:
        raise ValueError("one structure file cannot represent multiple molecular systems")
    omitted = ~structure.coordinate_mask
    if bool((omitted & (structure.occupancy_mask | structure.b_factor_mask)).any()):
        raise ValueError(
            "a structure file cannot preserve occupancy or B-factor evidence " "for atoms without coordinates"
        )
    target, format_name = _structure_path(path)
    if format_name == "mmcif":
        from .mmcif import write_mmcif

        write_mmcif(structure, target)
    else:
        from .pdb import write_pdb

        write_pdb(structure, target)


def _structure_path(
    path: str | os.PathLike[str],
    *,
    allow_compressed: bool = False,
) -> tuple[Path, Literal["pdb", "mmcif"]]:
    resolved = Path(os.path.expanduser(os.fspath(path)))
    suffix = resolved.suffix.lower()
    if allow_compressed and suffix == ".gz":
        suffix = resolved.with_suffix("").suffix.lower()
    try:
        return resolved, _FORMATS[suffix]
    except KeyError:
        expected = (
            ".pdb, .pdb.gz, .cif, .cif.gz, .mmcif, or .mmcif.gz"
            if allow_compressed
            else ".pdb, .cif, or .mmcif"
        )
        raise ValueError(f"structure path suffix must be {expected}") from None


def _exclude_water_rows(
    format_name: Literal["pdb", "mmcif"],
    rows: np.recarray,
    cif_file: object | None,
    component_table: Mapping[str, ChemicalComponent],
) -> np.recarray:
    entity_types: Mapping[str, str] = {}
    if format_name == "mmcif":
        assert cif_file is not None
        from .mmcif import _read_mmcif_entity_declarations

        entity_types, _, _, _, _, _ = _read_mmcif_entity_declarations(cif_file)

    fallback_rows = rows[np.asarray([row.entity_id is None or row.entity_id not in entity_types for row in rows])].view(
        np.recarray
    )
    groups: dict[tuple[object, ...], list[np.record]] = {}
    for row in fallback_rows:
        key = (
            ("entity", row.entity_id)
            if format_name == "mmcif" and row.entity_id is not None
            else ("residue", *atom_row_residue_key(row))
        )
        groups.setdefault(key, []).append(row)
    fallback_water_rows = {
        int(row.row_index)
        for group_rows in groups.values()
        if all(
            str(component_id) in component_table and is_water_component(component_table[str(component_id)])
            for component_id in {row.component_id for row in group_rows}
        )
        for row in group_rows
    }
    water_entities = {entity_id for entity_id, entity_type in entity_types.items() if entity_type == "water"}
    keep = np.asarray(
        [row.entity_id not in water_entities and int(row.row_index) not in fallback_water_rows for row in rows],
        dtype=np.bool_,
    )
    return rows[keep].view(np.recarray)


def _add_observed_components(
    rows: np.recarray,
    component_table: dict[str, ChemicalComponent],
) -> set[str]:
    r"""Create atom-site-only placeholders for component IDs absent from the configured CCD."""

    atoms_by_component: dict[str, dict[str, ChemicalAtom]] = {}
    for row in rows:
        component_id = str(row.component_id)
        if component_id in component_table:
            continue
        element = str(row.element)
        if element not in ATOMIC_NUMBER_BY_ELEMENT:
            raise ValueError(
                f"source atom {row.atom_name!r} in component {component_id!r} has invalid element {element!r}"
            )
        atom = ChemicalAtom(
            atom_id=str(row.atom_name),
            element=element,
        )
        component_atoms = atoms_by_component.setdefault(component_id, {})
        previous = component_atoms.get(atom.atom_id)
        if previous is None:
            component_atoms[atom.atom_id] = atom
            continue
        if previous.element != atom.element:
            raise ValueError(
                f"component {component_id!r} has conflicting observed elements for atom {atom.atom_id!r}"
            )
    for component_id, atoms_by_id in atoms_by_component.items():
        component_table[component_id] = ChemicalComponent(
            component_id=component_id,
            atoms=tuple(atoms_by_id.values()),
            bonds=None,
        )
    return set(atoms_by_component)


def _normalize_source_rows(
    rows: np.recarray,
    *,
    explicit_component_ids: set[str],
) -> np.recarray:
    """Apply the Protenix raw-occurrence normalization before CCD matching."""

    normalized = rows.copy().view(np.recarray)
    keep = np.ones(len(normalized), dtype=np.bool_)
    for index, row in enumerate(normalized):
        component_id = str(row.component_id)
        if component_id in {"UNX", "UNL"} and component_id not in explicit_component_ids:
            keep[index] = False
            continue
        if component_id == "ASX":
            row.component_id = "ASP"
            row.auth_component_id = "ASP"
            if row.atom_name in {"XD1", "XD2"}:
                row.atom_name = "O" + str(row.atom_name)[1:]
            if row.auth_atom_name in {"XD1", "XD2"}:
                row.auth_atom_name = "O" + str(row.auth_atom_name)[1:]
            if row.element == "X":
                row.element = "O"
        elif component_id == "GLX":
            row.component_id = "GLU"
            row.auth_component_id = "GLU"
            if row.atom_name in {"XE1", "XE2"}:
                row.atom_name = "O" + str(row.atom_name)[1:]
            if row.auth_atom_name in {"XE1", "XE2"}:
                row.auth_atom_name = "O" + str(row.auth_atom_name)[1:]
            if row.element == "X":
                row.element = "O"
    return normalized[keep].view(np.recarray)


def _filter_source_component_atoms(
    rows: np.recarray,
    component_table: Mapping[str, ChemicalComponent],
    observed_only_component_ids: set[str],
) -> np.recarray:
    """Keep exact CCD atom IDs and apply Protenix's matched>unmatched occurrence gate."""

    keep = np.zeros(len(rows), dtype=np.bool_)
    occurrences: dict[tuple[object, ...], list[int]] = {}
    for index, row in enumerate(rows):
        occurrences.setdefault(atom_row_residue_key(row), []).append(index)
    for indices in occurrences.values():
        component_id = str(rows[indices[0]].component_id)
        component = component_table[component_id]
        if component_id in observed_only_component_ids:
            keep[indices] = True
            continue
        atoms_by_id = {atom.atom_id: atom for atom in component.atoms}
        atom_ids = set(atoms_by_id)
        matched = [index for index in indices if str(rows[index].atom_name) in atom_ids]
        if len(matched) > len(indices) - len(matched):
            for index in matched:
                row = rows[index]
                expected = atoms_by_id[str(row.atom_name)].element
                source_element = str(row.element)
                comparable_source = "H" if source_element in HYDROGEN_ELEMENTS else source_element
                comparable_expected = "H" if expected in HYDROGEN_ELEMENTS else expected
                if comparable_source != comparable_expected:
                    raise ValueError(
                        f"source atom {row.atom_name!r} in component {component_id!r} has element "
                        f"{row.element!r}, expected CCD element {expected!r}"
                    )
            keep[matched] = True
    return rows[keep].view(np.recarray)


def _remove_all_unknown_protein_chains(rows: np.recarray, cif_file: object) -> np.recarray:
    from .mmcif import _read_mmcif_entity_declarations

    _, polymer_monomers, entity_poly_types, _, _, _ = _read_mmcif_entity_declarations(cif_file)
    keep = np.ones(len(rows), dtype=np.bool_)
    chains: dict[tuple[object, object], list[int]] = {}
    for index, row in enumerate(rows):
        chains.setdefault((row.symmetry_id, row.chain_id), []).append(index)
    for indices in chains.values():
        entity_id = str(rows[indices[0]].entity_id)
        if entity_poly_types.get(entity_id, "").casefold() != "polypeptide(l)":
            continue
        declared = None if polymer_monomers is None else polymer_monomers.get(entity_id)
        if declared and all(set(candidates) == {"UNK"} for _, candidates in declared):
            keep[indices] = False
    return rows[keep].view(np.recarray)


def _induce_component_atoms(component: ChemicalComponent, retained_atom_ids: set[str]) -> ChemicalComponent:
    keep = tuple(atom.atom_id in retained_atom_ids for atom in component.atoms)
    atoms = tuple(atom for atom, retained in zip(component.atoms, keep) if retained)
    bonds = component.bonds
    if bonds is not None:
        bonds = frozenset(
            bond
            for bond in bonds
            if bond.atom_a in retained_atom_ids
            and bond.atom_b in retained_atom_ids
            and (bond.stereo_atoms is None or set(bond.stereo_atoms) <= retained_atom_ids)
        )
    return ChemicalComponent(
        component.component_id,
        atoms,
        bonds,
        reference_coordinates=(
            tuple(value for value, retained in zip(component.reference_coordinates, keep) if retained)
            if component.reference_coordinates
            else ()
        ),
    )


def _without_component_topology(component: ChemicalComponent) -> ChemicalComponent:
    """Retain component atoms, attachments, and references but mark internal topology unknown."""

    if component.bonds is None:
        return component
    return ChemicalComponent(
        component_id=component.component_id,
        atoms=component.atoms,
        bonds=None,
        previous_attachment=component.previous_attachment,
        next_attachment=component.next_attachment,
        reference_coordinates=component.reference_coordinates,
    )


def _altloc_position_key(row: np.record) -> tuple[object, ...]:
    # Component identity is deliberately absent: microheterogeneous components at one
    # structural position compete as alternate realizations.
    return (
        row.symmetry_id,
        row.chain_id,
        row.residue_id,
        row.auth_residue_id,
        row.insertion_code,
    )


def _select_mmcif_polymer_microheterogeneity(
    rows: np.recarray,
    polymer_monomers: Mapping[str, tuple[tuple[int, tuple[str, ...]], ...]] | None,
    *,
    generator: torch.Generator | None,
) -> np.recarray:
    """Randomly retain one complete deposited polymer identity at each alternate block."""

    if polymer_monomers is None:
        return rows
    declared_positions = {
        (entity_id, number): frozenset(candidates)
        for entity_id, positions in polymer_monomers.items()
        for number, candidates in positions
    }
    microheterogeneous = {
        (entity_id, number): frozenset(candidates)
        for entity_id, positions in polymer_monomers.items()
        for number, candidates in positions
        if len(candidates) > 1
    }
    if not microheterogeneous:
        return rows

    position_rows: dict[tuple[str, str, int], list[np.record]] = {}
    for row in rows:
        if row.entity_id is None or row.label_seq_id is None:
            continue
        entity_id = str(row.entity_id)
        number = int(row.label_seq_id)
        if (entity_id, number) not in declared_positions:
            continue
        position_rows.setdefault((str(row.chain_id), entity_id, number), []).append(row)

    # A selection is (altloc label, component ID). A None label denotes an
    # uncommon microheterogeneous position whose candidates are wholly unlabelled;
    # it is selected independently and intentionally breaks an altloc block.
    selected: dict[tuple[str, str, int], tuple[str | None, str]] = {}
    alt_positions: dict[tuple[str, str], list[tuple[int, dict[str, str], bool]]] = {}
    for position, source_position_rows in position_rows.items():
        _, entity_id, number = position
        candidates = declared_positions[(entity_id, number)]
        is_microheterogeneous = (entity_id, number) in microheterogeneous
        label_components: dict[str, set[str]] = {}
        observed_components: set[str] = set()
        for row in source_position_rows:
            component_id = str(row.component_id)
            if is_microheterogeneous and component_id not in candidates:
                raise ValueError(
                    f"observed mmCIF component {component_id!r} is not declared at "
                    f"entity {entity_id!r} position {number}"
                )
            observed_components.add(component_id)
            if row.altloc_id is not None:
                label_components.setdefault(str(row.altloc_id), set()).add(component_id)
        ambiguous_labels = {
            label: components for label, components in label_components.items() if len(components) != 1
        }
        if ambiguous_labels:
            if is_microheterogeneous:
                raise ValueError(
                    f"mmCIF polymer microheterogeneity at {position!r} maps alternate labels "
                    f"to multiple monomers: {ambiguous_labels!r}"
                )
            # This ordinary position cannot safely participate in a correlated
            # microheterogeneous block; leave it to the generic altloc policy.
            continue
        labels = {label: next(iter(components)) for label, components in label_components.items()}
        if labels:
            labelled_components = set(labels.values())
            if is_microheterogeneous and observed_components - labelled_components:
                raise ValueError(
                    f"mmCIF polymer microheterogeneity at {position!r} mixes labelled and "
                    f"unlabelled monomer candidates"
                )
            alt_positions.setdefault(position[:2], []).append((number, labels, is_microheterogeneous))
            continue
        if not is_microheterogeneous:
            continue
        if not observed_components:
            continue
        choices = sorted(observed_components)
        choice_index = int(torch.randint(len(choices), (), generator=generator, device="cpu").item())
        selected[position] = (None, choices[choice_index])

    def choose_block(
        chain_entity: tuple[str, str],
        block: list[tuple[int, dict[str, str], bool]],
        common_labels: set[str],
    ) -> None:
        if not any(is_microheterogeneous for _, _, is_microheterogeneous in block):
            return
        choices = sorted(common_labels)
        choice_index = int(torch.randint(len(choices), (), generator=generator, device="cpu").item())
        label = choices[choice_index]
        chain_id, entity_id = chain_entity
        for number, labels, _ in block:
            selected[(chain_id, entity_id, number)] = (label, labels[label])

    for chain_entity, positions in alt_positions.items():
        positions.sort(key=lambda item: item[0])
        block: list[tuple[int, dict[str, str], bool]] = []
        common_labels: set[str] = set()
        previous_number: int | None = None
        for number, labels, is_microheterogeneous in positions:
            label_set = set(labels)
            intersection = common_labels & label_set
            if block and (previous_number is None or number != previous_number + 1 or not intersection):
                choose_block(chain_entity, block, common_labels)
                block = []
                common_labels = set()
            if not block:
                common_labels = label_set
            else:
                common_labels &= label_set
            block.append((number, labels, is_microheterogeneous))
            previous_number = number
        if block:
            choose_block(chain_entity, block, common_labels)

    if not selected:
        return rows
    keep: list[bool] = []
    for row in rows:
        if row.entity_id is None or row.label_seq_id is None:
            keep.append(True)
            continue
        position = (str(row.chain_id), str(row.entity_id), int(row.label_seq_id))
        realization = selected.get(position)
        if realization is None:
            keep.append(True)
            continue
        label, component_id = realization
        row_label = None if row.altloc_id is None else str(row.altloc_id)
        keep.append(
            str(row.component_id) == component_id
            and (label is None or row_label is None or row_label == label)
        )
    retained = rows[np.asarray(keep, dtype=np.bool_)].view(np.recarray)
    if not len(retained):
        raise ValueError("polymer microheterogeneity selection contains no atoms")
    return retained


def _select_altloc(
    rows: np.recarray,
    policy: Literal["highest_occupancy", "first"],
) -> np.recarray:
    labels_by_residue: dict[tuple[object, ...], list[str]] = {}
    occupancy_by_label: dict[tuple[tuple[object, ...], str], float] = {}
    for row in rows:
        if row.altloc_id is None:
            continue
        residue_key = _altloc_position_key(row)
        label = str(row.altloc_id)
        labels = labels_by_residue.setdefault(residue_key, [])
        if label not in labels:
            labels.append(label)
        key = residue_key, label
        occupancy_by_label[key] = occupancy_by_label.get(key, 0.0) + float(row.occupancy or 0.0)
    selected: dict[tuple[object, ...], str] = {}
    for residue_key, labels in labels_by_residue.items():
        if policy == "first":
            selected[residue_key] = labels[0]
        else:
            selected[residue_key] = max(
                labels,
                key=lambda label: occupancy_by_label[(residue_key, label)],
            )
    keep = np.asarray(
        [
            row.altloc_id is None or selected.get(_altloc_position_key(row), row.altloc_id) == row.altloc_id
            for row in rows
        ],
        dtype=np.bool_,
    )
    return rows[keep].view(np.recarray)


def _resolve_branch_component_index(
    row: np.record,
    component_indices: Mapping[tuple[object, ...], AbstractSet[int]],
    residue_key: tuple[object, ...],
) -> int:
    identity_candidates: list[set[int]] = []
    label_candidates = component_indices.get(
        ("label", row.chain_id, row.entity_id, row.component_id, row.chain_id, None)
    )
    if label_candidates is not None:
        identity_candidates.append(set(label_candidates))
    if row.auth_component_id is not None and row.auth_chain_id is not None:
        pdb_identity_candidates = component_indices.get(
            (
                "pdb",
                row.chain_id,
                row.entity_id,
                row.auth_component_id,
                row.auth_chain_id,
                row.auth_residue_id,
            )
        )
        if pdb_identity_candidates is not None:
            identity_candidates.append(set(pdb_identity_candidates))
    candidates = set.intersection(*identity_candidates) if identity_candidates else set()
    if not candidates:
        raise ValueError(f"cannot map branched-entity residue {residue_key!r} to its declaration")
    if len(candidates) != 1:
        raise ValueError(f"branched-entity residue {residue_key!r} maps ambiguously to its declaration")
    return next(iter(candidates))


_ENTITY_POLY_TYPE_TO_POLYMER: dict[str, PolymerType] = {
    "polypeptide(l)": "protein",
    "polypeptide(d)": "protein",
    "cyclic-pseudo-peptide": "protein",
    "polyribonucleotide": "rna",
    "polydeoxyribonucleotide": "dna",
    "polydeoxyribonucleotide/polyribonucleotide hybrid": "dna_rna_hybrid",
    "peptide nucleic acid": "pna",
}


def _mmcif_residue_is_polymer(row: np.record, entity_types: Mapping[str, str]) -> bool:
    if row.entity_id is not None:
        entity_type = entity_types.get(str(row.entity_id))
        if entity_type is not None:
            return entity_type == "polymer"
    return get_builtin_component_polymer_type(str(row.component_id)) is not None


def _declared_polymer_type(
    declared_monomers: tuple[tuple[int, tuple[str, ...]], ...],
    entity_poly_type: str | None,
) -> PolymerType | None:
    resolved = {
        get_builtin_component_polymer_type(monomer) for _, candidates in declared_monomers for monomer in candidates
    } - {None}
    if resolved == {"dna", "rna"}:
        return "dna_rna_hybrid"
    if len(resolved) == 1:
        return next(iter(resolved))
    return _ENTITY_POLY_TYPE_TO_POLYMER.get(entity_poly_type) if entity_poly_type is not None else None


def _resolve_declared_component(
    component_id: str,
    component_table: Mapping[str, ChemicalComponent],
    *,
    include_hydrogens: bool,
) -> ChemicalComponent:
    component = component_table.get(component_id)
    if component is None:
        component = ChemicalComponent(component_id=component_id, atoms=(), bonds=None)
    return component if include_hydrogens else component.without_hydrogens()


def _declared_polymer(
    entity_id: str,
    polymer_monomers: Mapping[str, tuple[tuple[int, tuple[str, ...]], ...]] | None,
    entity_poly_types: Mapping[str, str],
    component_table: Mapping[str, ChemicalComponent],
    polymer_residues: list[tuple[object, ...]],
    residue_rows: Mapping[tuple[object, ...], list[np.record]],
    *,
    include_hydrogens: bool,
) -> tuple[PolymerSequence, tuple[tuple[int, tuple[object, ...]], ...]] | None:
    if polymer_monomers is None:
        return None
    declared_monomers = polymer_monomers.get(entity_id)
    if not declared_monomers:
        raise ValueError(f"mmCIF polymer entity {entity_id!r} has no _entity_poly_seq declaration")

    observed: dict[int, str] = {}
    indexed: list[tuple[int, tuple[object, ...]]] = []
    for key in polymer_residues:
        row = residue_rows[key][0]
        if row.label_seq_id is None:
            raise ValueError("observed mmCIF polymer residue lacks label_seq_id")
        index = int(row.label_seq_id) - 1
        if index < 0 or index >= len(declared_monomers):
            raise ValueError(
                f"observed mmCIF label_seq_id {row.label_seq_id} " "is outside the declared polymer sequence"
            )
        if index in observed:
            raise ValueError(f"multiple observed mmCIF residues map to label_seq_id {row.label_seq_id}")
        component_id = str(row.component_id)
        candidates = declared_monomers[index][1]
        if component_id not in candidates:
            raise ValueError(
                f"observed mmCIF component {component_id!r} is not declared at "
                f"entity {entity_id!r} position {row.label_seq_id}"
            )
        observed[index] = component_id
        indexed.append((index, key))

    polymer_type = _declared_polymer_type(declared_monomers, entity_poly_types.get(entity_id))
    if polymer_type is None:
        raise ValueError(f"mmCIF polymer entity {entity_id!r} has ambiguous polymer type declarations")
    components: list[ChemicalComponent] = []
    for index, (_, candidates) in enumerate(declared_monomers):
        selected_component_id = observed.get(index)
        if selected_component_id is None:
            if len(candidates) != 1:
                raise ValueError(
                    f"unmodeled mmCIF polymer position {index + 1} of entity {entity_id!r} "
                    f"has ambiguous monomers {candidates!r}"
                )
            selected_component_id = candidates[0]
        components.append(
            _resolve_declared_component(
                selected_component_id,
                component_table,
                include_hydrogens=include_hydrogens,
            )
        )
    return PolymerSequence(polymer_type, tuple(components)), tuple(indexed)


def _source_author_numbering(
    author_numbering: Mapping[
        tuple[str, int] | tuple[str, int, str],
        tuple[str | None, str | None],
    ],
    asym_id: str,
    component_index: int,
    component_id: str,
) -> tuple[str | None, str | None] | None:
    """Resolve candidate-specific polymer numbering before a generic fallback."""

    return author_numbering.get(
        (asym_id, component_index + 1, component_id),
        author_numbering.get((asym_id, component_index + 1)),
    )


def _register_realized_polymer_entity(
    entities: dict[str, PolymerSequence | Molecule],
    variants: dict[str, list[tuple[PolymerSequence, str]]],
    reserved_entity_ids: set[str],
    source_entity_id: str,
    entity: PolymerSequence,
) -> tuple[str, PolymerSequence]:
    """Reuse equal source-entity sequences and split divergent realized sequences."""

    source_variants = variants.setdefault(source_entity_id, [])
    for previous, entity_id in source_variants:
        if previous == entity:
            return entity_id, previous
    if not source_variants:
        previous = entities.get(source_entity_id)
        if previous is not None and previous != entity:
            raise ValueError(f"source entity ID {source_entity_id!r} has conflicting molecular definitions")
        entities[source_entity_id] = entity
        source_variants.append((entity, source_entity_id))
        return source_entity_id, entity

    signature = "\0".join(
        (entity.polymer_type, *(component.component_id for component in entity.components))
    ).encode("utf-8")
    digest = hashlib.sha256(signature).hexdigest()[:16]
    base_id = f"{source_entity_id}#variant-{digest}"
    entity_id = base_id
    ordinal = 2
    while entity_id in reserved_entity_ids or entity_id in entities:
        entity_id = f"{base_id}-{ordinal}"
        ordinal += 1
    entities[entity_id] = entity
    reserved_entity_ids.add(entity_id)
    source_variants.append((entity, entity_id))
    return entity_id, entity


def _pdb_polymer_segments(
    polymer_residues: list[tuple[object, ...]],
    residue_rows: Mapping[tuple[object, ...], list[np.record]],
    segment_by_serial: Mapping[str, int],
) -> list[list[tuple[object, ...]]]:
    segments: list[list[tuple[object, ...]]] = []
    current: list[tuple[object, ...]] = []
    previous_segment: int | None = None
    for key in polymer_residues:
        row = residue_rows[key][0]
        segment = segment_by_serial.get(str(row.serial)) if row.serial is not None else None
        if current and previous_segment is not None and segment is not None and segment != previous_segment:
            segments.append(current)
            current = []
        current.append(key)
        previous_segment = segment
    if current:
        segments.append(current)
    return segments


def _pdb_polymer_with_declared_missing_residues(
    polymer_type: PolymerType,
    chain_id: str,
    segment_residues: list[tuple[object, ...]],
    residue_rows: Mapping[tuple[object, ...], list[np.record]],
    component_table: Mapping[str, ChemicalComponent],
    declarations: tuple[tuple[str, str, int, str], ...],
    *,
    include_terminal: bool,
    include_hydrogens: bool,
) -> tuple[
    PolymerSequence,
    tuple[tuple[int, tuple[object, ...]], ...],
    tuple[tuple[int, str, str], ...],
]:
    observed: list[tuple[tuple[int, str], tuple[object, ...], str]] = []
    for residue_key in segment_residues:
        row = residue_rows[residue_key][0]
        number = row.auth_residue_id if row.auth_residue_id is not None else row.residue_id
        if number is None:
            raise ValueError("PDB polymer residue lacks author numbering")
        observed.append(((int(number), str(row.insertion_code or "")), residue_key, str(row.component_id)))

    observed_positions = [position for position, _, _ in observed]
    declared = [
        ((number, insertion_code), component_id)
        for declared_chain, component_id, number, insertion_code in declarations
        if declared_chain == chain_id
    ]
    if not include_terminal:
        first, last = observed_positions[0], observed_positions[-1]
        declared = [(position, component_id) for position, component_id in declared if first < position < last]
    observed_position_set = set(observed_positions)
    declared = [
        (position, component_id) for position, component_id in declared if position not in observed_position_set
    ]
    if not declared:
        return (
            PolymerSequence(polymer_type, tuple(component_table[component_id] for _, _, component_id in observed)),
            tuple((index, residue_key) for index, (_, residue_key, _) in enumerate(observed)),
            (),
        )
    if any(left >= right for left, right in zip(observed_positions, observed_positions[1:])):
        raise ValueError("cannot place declared missing PDB residues in non-monotonic author numbering")

    entries: list[tuple[tuple[int, str], tuple[object, ...] | None, str]] = [
        (position, residue_key, component_id) for position, residue_key, component_id in observed
    ]
    entries.extend((position, None, component_id) for position, component_id in declared)
    entries.sort(key=lambda entry: entry[0])
    components: list[ChemicalComponent] = []
    indexed: list[tuple[int, tuple[object, ...]]] = []
    unmodeled: list[tuple[int, str, str]] = []
    for component_index, (position, entry_residue_key, component_id) in enumerate(entries):
        components.append(
            _resolve_declared_component(
                component_id,
                component_table,
                include_hydrogens=include_hydrogens,
            )
        )
        if entry_residue_key is None:
            unmodeled.append((component_index, str(position[0]), position[1]))
        else:
            indexed.append((component_index, entry_residue_key))
    return PolymerSequence(polymer_type, tuple(components)), tuple(indexed), tuple(unmodeled)


def _build_structure(
    format_name: Literal["pdb", "mmcif"],
    rows: np.recarray,
    source_rows: np.recarray,
    component_table: dict[str, ChemicalComponent],
    *,
    include_water: bool,
    include_hydrogens: bool,
    include_declared_instances: bool,
    policy_removed_asym_ids: AbstractSet[str],
    cif_file: object | None,
    pdb_lines: tuple[str, ...],
    pdb_missing_residues: tuple[tuple[str, str, int, str], ...],
) -> Structure:
    if not len(rows):
        raise ValueError("alternate-location selection contains no atoms")
    residue_rows: dict[tuple[object, ...], list[np.record]] = {}
    chain_residue_order: dict[tuple[str | None, str], list[tuple[object, ...]]] = {}
    for row in rows:
        residue_key = atom_row_residue_key(row)
        bucket = residue_rows.get(residue_key)
        if bucket is None:
            residue_rows[residue_key] = [row]
            chain_key = (row.symmetry_id, str(row.chain_id))
            chain_residue_order.setdefault(chain_key, []).append(residue_key)
        else:
            bucket.append(row)

    entity_types: dict[str, str] = {}
    polymer_monomers: dict[str, tuple[tuple[int, tuple[str, ...]], ...]] | None = None
    entity_poly_types: dict[str, str] = {}
    author_numbering: dict[
        tuple[str, int] | tuple[str, int, str],
        tuple[str | None, str | None],
    ] = {}
    asym_entities: dict[str, str] = {}
    nonpoly_components: dict[str, str] = {}
    segment_by_serial: dict[str, int] = {}
    molecule_component_table: dict[str, ChemicalComponent] = dict(component_table)
    if format_name == "mmcif":
        assert cif_file is not None
        from .mmcif import (
            _mmcif_branch_entities_with_components,
            _read_mmcif_entity_declarations,
            read_mmcif_branch_entities,
        )

        (
            entity_types,
            polymer_monomers,
            entity_poly_types,
            author_numbering,
            asym_entities,
            nonpoly_components,
        ) = _read_mmcif_entity_declarations(cif_file)
        retained_nonpoly_atoms: dict[str, set[str]] = {}
        for row in rows:
            if entity_types.get(str(row.entity_id)) == "polymer":
                continue
            retained_nonpoly_atoms.setdefault(str(row.component_id), set()).add(str(row.atom_name))
        for component_id, atom_ids in retained_nonpoly_atoms.items():
            molecule_component_table[component_id] = _induce_component_atoms(
                component_table[component_id],
                atom_ids,
            )
        missing_raw_component_ids = {"UNX", "UNL"} - set(molecule_component_table)
        excluded_branch_entities = _mmcif_branch_entities_with_components(cif_file, missing_raw_component_ids)
        branch_entities, branch_component_indices = read_mmcif_branch_entities(
            cif_file,
            molecule_component_table,
            selected_rows=rows,
            excluded_entity_ids=frozenset(excluded_branch_entities),
        )
        for entity_id in branch_entities:
            entity_type = entity_types.get(entity_id)
            if entity_type not in {None, "branched"}:
                raise ValueError(
                    f"mmCIF entity {entity_id!r} is declared {entity_type!r} "
                    "but also has a branched-entity definition"
                )
            entity_types[entity_id] = "branched"
        missing_branch_definition = {
            entity_id
            for entity_id, entity_type in entity_types.items()
            if entity_type == "branched"
            and entity_id not in branch_entities
            and entity_id not in excluded_branch_entities
        }
        if missing_branch_definition:
            raise ValueError(
                "mmCIF branched entities require branch declarations: " f"{sorted(missing_branch_definition)!r}"
            )
    else:
        from .pdb import pdb_segment_by_serial

        branch_entities, branch_component_indices = {}, {}
        segment_by_serial = pdb_segment_by_serial(pdb_lines)

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
    entities: dict[str, PolymerSequence | Molecule] = {}
    realized_polymer_entities: dict[str, list[tuple[PolymerSequence, str]]] = {}
    reserved_entity_ids = set(entity_types) | set(asym_entities.values())
    entity: PolymerSequence | Molecule
    unmodeled_numbers: dict[ComponentReference, tuple[str, str]] = {}
    unmodeled_source_sites: dict[tuple[object, ...], AtomSite] = {}
    ambiguous_unmodeled_sites: set[tuple[object, ...]] = set()

    def add_unmodeled_site(key: tuple[object, ...], site: AtomSite) -> None:
        if key in ambiguous_unmodeled_sites:
            return
        previous = unmodeled_source_sites.get(key)
        if previous is not None and previous != site:
            unmodeled_source_sites.pop(key)
            ambiguous_unmodeled_sites.add(key)
        else:
            unmodeled_source_sites[key] = site

    for (symmetry_id, chain_id), residue_order in chain_residue_order.items():
        base_instance_id = chain_id if symmetry_id is None else f"{chain_id}#{symmetry_id}"
        if format_name == "mmcif":
            polymer_residues = [
                key for key in residue_order if _mmcif_residue_is_polymer(residue_rows[key][0], entity_types)
            ]
            polymer_segments = [polymer_residues] if polymer_residues else []
        else:
            polymer_residues = [
                key
                for key in residue_order
                if any(row.group == "ATOM" for row in residue_rows[key])
                or _pdb_component_polymer_type(component_table[str(residue_rows[key][0].component_id)]) is not None
            ]
            polymer_segments = _pdb_polymer_segments(polymer_residues, residue_rows, segment_by_serial)

        segment_declarations = [pdb_missing_residues for _ in polymer_segments]
        if format_name == "pdb" and len(polymer_segments) > 1:
            assigned: list[list[tuple[str, str, int, str]]] = [[] for _ in polymer_segments]
            segment_positions: list[list[tuple[int, str]]] = []
            for segment_residues in polymer_segments:
                positions: list[tuple[int, str]] = []
                for residue_key in segment_residues:
                    row = residue_rows[residue_key][0]
                    number = row.auth_residue_id if row.auth_residue_id is not None else row.residue_id
                    if number is None:
                        raise ValueError("PDB polymer residue lacks author numbering")
                    positions.append((int(number), str(row.insertion_code or "")))
                segment_positions.append(positions)
            for declaration in pdb_missing_residues:
                declared_chain, _, number, declared_insertion_code = declaration
                if declared_chain != chain_id:
                    continue
                position = (number, declared_insertion_code)
                candidates = [
                    index
                    for index, positions in enumerate(segment_positions)
                    if position in positions or positions[0] < position < positions[-1]
                ]
                if len(candidates) == 1:
                    assigned[candidates[0]].append(declaration)
            segment_declarations = [tuple(values) for values in assigned]

        for segment_ordinal, segment_residues in enumerate(polymer_segments):
            first = residue_rows[segment_residues[0]][0]
            source_entity_id = None if first.entity_id is None else str(first.entity_id)
            declared = None
            pdb_unmodeled: tuple[tuple[int, str, str], ...] = ()
            if format_name == "mmcif" and source_entity_id is not None:
                declared = _declared_polymer(
                    source_entity_id,
                    polymer_monomers,
                    entity_poly_types,
                    component_table,
                    segment_residues,
                    residue_rows,
                    include_hydrogens=include_hydrogens,
                )
            if declared is not None:
                declared_entity, indexed = declared
                assert source_entity_id is not None
                entity_id, declared_entity = _register_realized_polymer_entity(
                    entities,
                    realized_polymer_entities,
                    reserved_entity_ids,
                    source_entity_id,
                    declared_entity,
                )
                entity = declared_entity
            else:
                classified_values = {
                    (
                        get_builtin_component_polymer_type(str(residue_rows[key][0].component_id))
                        if format_name == "mmcif"
                        else _pdb_component_polymer_type(component_table[str(residue_rows[key][0].component_id)])
                    )
                    for key in segment_residues
                }
                classified = {polymer_type for polymer_type in classified_values if polymer_type is not None}
                if classified == {"dna", "rna"}:
                    chain_polymer_type: PolymerType = "dna_rna_hybrid"
                elif len(classified) == 1:
                    chain_polymer_type = next(iter(classified))
                elif not classified and format_name == "mmcif" and source_entity_id is not None:
                    mapped = _ENTITY_POLY_TYPE_TO_POLYMER.get(entity_poly_types.get(source_entity_id, ""))
                    if mapped is None:
                        raise ValueError(f"polymer chain {base_instance_id!r} has ambiguous polymer type declarations")
                    chain_polymer_type = mapped
                else:
                    raise ValueError(f"polymer chain {base_instance_id!r} has ambiguous polymer type declarations")
                if format_name == "mmcif" and source_entity_id is not None:
                    entity_id = source_entity_id
                elif segment_ordinal == 0:
                    entity_id = f"polymer_{chain_id}"
                else:
                    entity_id = f"polymer_{chain_id}#{segment_ordinal}"
                if format_name == "pdb":
                    resolved, indexed, pdb_unmodeled = _pdb_polymer_with_declared_missing_residues(
                        chain_polymer_type,
                        chain_id,
                        segment_residues,
                        residue_rows,
                        component_table,
                        segment_declarations[segment_ordinal],
                        include_terminal=len(polymer_segments) == 1,
                        include_hydrogens=include_hydrogens,
                    )
                else:
                    resolved = PolymerSequence(
                        chain_polymer_type,
                        tuple(component_table[str(residue_rows[key][0].component_id)] for key in segment_residues),
                    )
                    indexed = tuple(enumerate(segment_residues))
                entity = _reuse_entity(entities, entity_id, resolved)

            instance_id = unique_identifier(base_instance_id, used_instance_ids)
            if declared is not None:
                observed_indices = {index for index, _ in indexed}
                for component_index, component in enumerate(declared_entity.components):
                    if component_index in observed_indices:
                        continue
                    author = _source_author_numbering(
                        author_numbering,
                        chain_id,
                        component_index,
                        component.component_id,
                    )
                    missing_insertion_code = None if author is None else author[1]
                    for atom in component.atoms:
                        add_unmodeled_site(
                            (
                                symmetry_id,
                                chain_id,
                                component_index + 1,
                                missing_insertion_code,
                                component.component_id,
                                atom.atom_id,
                            ),
                            AtomSite(instance_id, component_index, atom.atom_id),
                        )
                    if author is not None and author[0] is not None:
                        unmodeled_numbers[(instance_id, component_index)] = (
                            str(author[0]),
                            author[1] or "",
                        )
            for component_index, residue_id, insertion_code in pdb_unmodeled:
                unmodeled_numbers[(instance_id, component_index)] = (residue_id, insertion_code)
            groups.append(
                (
                    min(int(row.row_index) for key in segment_residues for row in residue_rows[key]),
                    instance_id,
                    entity_id,
                    entity,
                    indexed,
                )
            )

        polymer_residue_set = set(polymer_residues)
        nonpolymer_residues = [key for key in residue_order if key not in polymer_residue_set]
        molecule_groups: list[tuple[tuple[object, ...], ...]] = []
        if format_name == "mmcif":
            grouped_by_entity: dict[str, list[tuple[object, ...]]] = {}
            for residue_key in nonpolymer_residues:
                source_entity_id = residue_rows[residue_key][0].entity_id
                if source_entity_id in branch_entities:
                    grouped_by_entity.setdefault(str(source_entity_id), []).append(residue_key)
                else:
                    molecule_groups.append((residue_key,))
            molecule_groups.extend(tuple(group) for group in grouped_by_entity.values())
            molecule_groups.sort(
                key=lambda group: min(int(row.row_index) for key in group for row in residue_rows[key])
            )
        else:
            molecule_groups.extend((residue_key,) for residue_key in nonpolymer_residues)

        for group_residues in molecule_groups:
            first = residue_rows[group_residues[0]][0]
            first_component = molecule_component_table[str(first.component_id)]
            residue_label = first.auth_residue_id if first.auth_residue_id is not None else first.residue_id
            residue_identity = f"{residue_label}{first.insertion_code or ''}"
            source_entity_id = None if first.entity_id is None else str(first.entity_id)
            if source_entity_id in branch_entities:
                entity = branch_entities[source_entity_id]
                entity_id = source_entity_id
                instance_seed = base_instance_id
                indexed_residues: list[tuple[int, tuple[object, ...]]] = []
                for residue_key in group_residues:
                    row = residue_rows[residue_key][0]
                    branch_component_index = _resolve_branch_component_index(
                        row,
                        branch_component_indices,
                        residue_key,
                    )
                    indexed_residues.append((branch_component_index, residue_key))
                if len({index for index, _ in indexed_residues}) != len(indexed_residues):
                    raise ValueError(f"branched entity {source_entity_id!r} maps multiple residues to one component")
                indexed_residues.sort()
            elif format_name == "mmcif" and source_entity_id is not None:
                source_entity_type = entity_types.get(source_entity_id)
                if source_entity_type == "branched":
                    raise ValueError(f"branched entity {source_entity_id!r} has no branch declaration")
                entity = Molecule(
                    (first_component,),
                    molecule_type=("water" if source_entity_type == "water" else "non-polymer"),
                )
                entity_id = source_entity_id
                instance_seed = base_instance_id
                indexed_residues = [(0, group_residues[0])]
            else:
                entity = Molecule(
                    (first_component,),
                    molecule_type=("water" if is_water_component(first_component) else "non-polymer"),
                )
                entity_id = f"molecule_{chain_id}:{residue_identity}:{first_component.component_id}"
                instance_seed = f"{base_instance_id}:{residue_identity}:{first_component.component_id}"
                indexed_residues = [(0, group_residues[0])]
            entity = _reuse_entity(entities, entity_id, entity)
            instance_id = unique_identifier(instance_seed, used_instance_ids)
            groups.append(
                (
                    min(int(row.row_index) for key in group_residues for row in residue_rows[key]),
                    instance_id,
                    entity_id,
                    entity,
                    tuple(indexed_residues),
                )
            )

    observed_asym_ids = {chain_id for _, chain_id in chain_residue_order}
    declared_instances = asym_entities.items() if include_declared_instances else ()
    for declaration_index, (asym_id, source_entity_id) in enumerate(declared_instances):
        if asym_id in observed_asym_ids or asym_id in policy_removed_asym_ids:
            continue
        source_entity_type = entity_types.get(source_entity_id)
        if not include_water and source_entity_type == "water":
            continue
        if source_entity_type == "polymer":
            declared_monomers = None if polymer_monomers is None else polymer_monomers.get(source_entity_id)
            if (
                entity_poly_types.get(source_entity_id, "").casefold() == "polypeptide(l)"
                and declared_monomers
                and all(set(candidates) == {"UNK"} for _, candidates in declared_monomers)
            ):
                continue
            declared = _declared_polymer(
                source_entity_id,
                polymer_monomers,
                entity_poly_types,
                component_table,
                [],
                residue_rows,
                include_hydrogens=include_hydrogens,
            )
            if declared is None:
                raise ValueError(f"unobserved mmCIF polymer entity {source_entity_id!r} " "has no sequence declaration")
            resolved, _ = declared
            entity_id, entity = _register_realized_polymer_entity(
                entities,
                realized_polymer_entities,
                reserved_entity_ids,
                source_entity_id,
                resolved,
            )
        elif source_entity_id in branch_entities:
            entity_id = source_entity_id
            entity = _reuse_entity(entities, entity_id, branch_entities[source_entity_id])
        else:
            component_id = nonpoly_components.get(source_entity_id)
            if component_id is None:
                continue
            if component_id in {"UNX", "UNL"} and component_id not in molecule_component_table:
                continue
            component = _resolve_declared_component(
                component_id,
                molecule_component_table,
                include_hydrogens=include_hydrogens,
            )
            entity_id = source_entity_id
            entity = _reuse_entity(
                entities,
                entity_id,
                Molecule(
                    (component,),
                    molecule_type=("water" if source_entity_type == "water" else "non-polymer"),
                ),
            )

        instance_id = unique_identifier(asym_id, used_instance_ids)
        for component_index, component in enumerate(entity.components):
            author = _source_author_numbering(
                author_numbering,
                asym_id,
                component_index,
                component.component_id,
            )
            source_author_number = None if author is None else author[0]
            insertion = None if author is None else author[1]
            if source_author_number is not None:
                unmodeled_numbers[(instance_id, component_index)] = (
                    source_author_number,
                    insertion or "",
                )
            if source_entity_type == "polymer":
                label_sequence: int | None = component_index + 1
            else:
                label_sequence = None
            for atom in component.atoms:
                site = AtomSite(instance_id, component_index, atom.atom_id)
                add_unmodeled_site(
                    (
                        None,
                        asym_id,
                        label_sequence,
                        insertion,
                        component.component_id,
                        atom.atom_id,
                    ),
                    site,
                )
                for source_key, branch_indices in branch_component_indices.items():
                    (
                        namespace,
                        label_asym,
                        source_branch_entity,
                        monomer,
                        auth_asym,
                        auth_sequence,
                    ) = source_key
                    if (
                        component_index not in branch_indices
                        or label_asym != asym_id
                        or source_branch_entity != source_entity_id
                    ):
                        continue
                    if namespace == "label":
                        add_unmodeled_site(
                            (None, label_asym, None, None, monomer, atom.atom_id),
                            site,
                        )
                    elif namespace == "pdb":
                        add_unmodeled_site(
                            (
                                None,
                                auth_asym,
                                auth_sequence,
                                None,
                                monomer,
                                atom.atom_id,
                            ),
                            site,
                        )
                if source_entity_id not in branch_entities:
                    add_unmodeled_site(
                        (
                            None,
                            asym_id,
                            source_author_number,
                            insertion,
                            component.component_id,
                            atom.atom_id,
                        ),
                        site,
                    )
        groups.append(
            (
                len(source_rows) + declaration_index,
                instance_id,
                entity_id,
                entity,
                (),
            )
        )

    groups.sort(key=lambda item: item[0])
    instances = {instance_id: entity_id for _, instance_id, entity_id, _, _ in groups}
    site_by_row_index: dict[int, AtomSite] = {}
    residue_numbers: dict[ComponentReference, tuple[str, str]] = dict(unmodeled_numbers)
    for _, instance_id, _, entity, indexed in groups:
        for component_index, residue_key in indexed:
            residue = residue_rows[residue_key]
            first = residue[0]
            source_number = first.auth_residue_id
            if format_name == "pdb" and source_number is None:
                source_number = first.residue_id
            source_insertion = str(first.insertion_code or "")
            if format_name == "mmcif":
                component = entity.components[component_index]
                author = _source_author_numbering(
                    author_numbering,
                    str(first.chain_id),
                    component_index,
                    component.component_id,
                )
                if author is not None:
                    if source_number is None:
                        source_number = author[0]
                    if first.insertion_code is None:
                        source_insertion = author[1] or ""
            if source_number is not None:
                residue_numbers[(instance_id, component_index)] = (
                    str(source_number),
                    source_insertion,
                )
            if format_name == "mmcif":
                component = entity.components[component_index]
                for atom in component.atoms:
                    site = AtomSite(instance_id, component_index, atom.atom_id)
                    add_unmodeled_site(
                        (
                            first.symmetry_id,
                            first.chain_id,
                            first.label_seq_id,
                            first.insertion_code,
                            component.component_id,
                            atom.atom_id,
                        ),
                        site,
                    )
                    add_unmodeled_site(
                        (
                            first.symmetry_id,
                            first.auth_chain_id,
                            first.auth_residue_id,
                            first.insertion_code,
                            first.auth_component_id or component.component_id,
                            atom.atom_id,
                        ),
                        site,
                    )
            for row in residue:
                site_by_row_index[int(row.row_index)] = AtomSite(
                    instance_id,
                    component_index,
                    str(row.atom_name),
                )
        if format_name == "mmcif" and isinstance(entity, Molecule) and indexed:
            first = residue_rows[indexed[0][1]][0]
            source_entity_id = first.entity_id
            observed_components = {component_index for component_index, _ in indexed}
            for source_key, component_indices in branch_component_indices.items():
                (
                    namespace,
                    source_label_asym_id,
                    source_branch_entity_id,
                    source_monomer_id,
                    source_auth_asym_id,
                    source_auth_number,
                ) = source_key
                if (
                    namespace not in {"label", "pdb"}
                    or source_branch_entity_id != source_entity_id
                    or source_label_asym_id != first.chain_id
                ):
                    continue
                if len(component_indices) != 1:
                    continue
                component_index = next(iter(component_indices))
                if component_index in observed_components:
                    continue
                component = entity.components[component_index]
                author = _source_author_numbering(
                    author_numbering,
                    str(source_label_asym_id),
                    component_index,
                    component.component_id,
                )
                insertion = None if author is None else author[1]
                if author is not None and author[0] is not None:
                    residue_numbers[(instance_id, component_index)] = (
                        author[0],
                        insertion or "",
                    )
                component = entity.components[component_index]
                for atom in component.atoms:
                    site = AtomSite(instance_id, component_index, atom.atom_id)
                    if namespace == "label":
                        add_unmodeled_site(
                            (
                                first.symmetry_id,
                                source_label_asym_id,
                                None,
                                insertion,
                                source_monomer_id,
                                atom.atom_id,
                            ),
                            site,
                        )
                    else:
                        add_unmodeled_site(
                            (
                                first.symmetry_id,
                                source_auth_asym_id,
                                source_auth_number,
                                insertion,
                                source_monomer_id,
                                atom.atom_id,
                            ),
                            site,
                        )

    if format_name == "pdb":
        from .pdb import read_pdb_bonds

        source_bonds = read_pdb_bonds(pdb_lines, source_rows, rows, site_by_row_index)
    else:
        assert cif_file is not None
        from .mmcif import read_mmcif_bonds

        source_bonds = read_mmcif_bonds(
            cif_file,
            rows,
            site_by_row_index,
            unmodeled_source_sites,
            ambiguous_unmodeled_sites,
        )
    source_bonds = tuple(
        bond
        for bond in source_bonds
        if (
            bond.site_a.instance_id,
            bond.site_a.component_index,
        )
        != (
            bond.site_b.instance_id,
            bond.site_b.component_index,
        )
    )
    instance_values = {instance_id: entity for _, instance_id, _, entity, _ in groups}
    polymer_leaving_sites = _canonical_polymer_leaving_sites(
        entities,
        instances,
        instance_values=instance_values,
    )
    source_bonds = tuple(
        bond
        for bond in source_bonds
        if bond.site_a not in polymer_leaving_sites and bond.site_b not in polymer_leaving_sites
    )
    system = MolecularSystem.from_entities(
        entities,
        instances,
        source_bonds,
        instance_values=instance_values,
    )

    atom_count = int(system.atomic_number.shape[0])
    component_count = int(system.component_symbol.shape[0])
    component_index_by_reference: dict[ComponentReference, int] = {}
    atom_index_by_site: dict[AtomSite, int] = {}
    for instance_index in range(int(system.instance_symbol.shape[0])):
        instance_id = system.instance_id(instance_index)
        component_start = int(system.instance_component_ptr[instance_index])
        component_stop = int(system.instance_component_ptr[instance_index + 1])
        for component_index in range(component_start, component_stop):
            local_component = component_index - component_start
            component_index_by_reference[(instance_id, local_component)] = component_index
            atom_start = int(system.component_atom_ptr[component_index])
            atom_stop = int(system.component_atom_ptr[component_index + 1])
            for atom_index in range(atom_start, atom_stop):
                atom_index_by_site[
                    AtomSite(
                        instance_id,
                        local_component,
                        system.symbols[int(system.atom_symbol[atom_index])],
                    )
                ] = atom_index
    coordinates = torch.zeros((atom_count, 3), dtype=torch.float32)
    coordinate_mask = torch.zeros(atom_count, dtype=torch.bool)
    occupancy = torch.zeros(atom_count, dtype=torch.float32)
    occupancy_mask = torch.zeros(atom_count, dtype=torch.bool)
    b_factor = torch.zeros(atom_count, dtype=torch.float32)
    b_factor_mask = torch.zeros(atom_count, dtype=torch.bool)
    atom_indices = []
    resolved_source_rows: list[int] = []
    for source_row_index, row in enumerate(rows):
        site = site_by_row_index[int(row.row_index)]
        try:
            atom_indices.append(atom_index_by_site[site])
        except KeyError:
            # CCD leaving atoms (notably observed OXT) are intentionally absent
            # after canonical polymer bonds are formed.
            continue
        resolved_source_rows.append(source_row_index)
    rows = rows[np.asarray(resolved_source_rows, dtype=np.int64)].view(np.recarray)
    seen_atom_indices: set[int] = set()
    for atom_index in atom_indices:
        if atom_index in seen_atom_indices:
            raise ValueError(f"selected source rows map more than once to {system.atom_site(atom_index)!r}")
        seen_atom_indices.add(atom_index)
    source_atom_index = torch.tensor(atom_indices, dtype=torch.int64)
    source_atomic_number = torch.tensor(
        [ATOMIC_NUMBER_BY_ELEMENT[str(element)] for element in rows.element],
        dtype=system.atomic_number.dtype,
    )
    element_mismatch = system.atomic_number[source_atom_index].cpu() != source_atomic_number
    if bool(element_mismatch.any()):
        source_index = int(torch.nonzero(element_mismatch, as_tuple=False)[0])
        site = system.atom_site(atom_indices[source_index])
        raise ValueError(f"source element {rows.element[source_index]!r} disagrees with component atom at {site!r}")
    source_isotope_mask = torch.tensor(
        [value is not None for value in rows.isotope],
        dtype=torch.bool,
    )
    system.isotope[source_atom_index[source_isotope_mask]] = torch.tensor(
        [int(value) for value in rows.isotope if value is not None],
        dtype=system.isotope.dtype,
    )
    system.isotope_known[source_atom_index[source_isotope_mask]] = True
    source_charge_mask = torch.tensor(
        [value is not None for value in rows.charge],
        dtype=torch.bool,
    )
    system.formal_charge[source_atom_index[source_charge_mask]] = torch.tensor(
        [int(value) for value in rows.charge if value is not None],
        dtype=system.formal_charge.dtype,
    )
    system.formal_charge_known[source_atom_index[source_charge_mask]] = True

    source_coordinate_mask = torch.from_numpy(np.asarray(rows.coordinate_present, dtype=np.bool_))
    coordinate_mask[source_atom_index] = source_coordinate_mask
    coordinates[source_atom_index] = torch.from_numpy(
        np.array(rows.coordinates, dtype=np.float32, copy=True, order="C")
    )
    source_occupancy_mask = torch.tensor(
        [value is not None for value in rows.occupancy],
        dtype=torch.bool,
    )
    occupancy_mask[source_atom_index] = source_occupancy_mask
    occupancy[source_atom_index] = torch.tensor(
        [0.0 if value is None else float(value) for value in rows.occupancy],
        dtype=torch.float32,
    )
    source_b_factor_mask = torch.tensor(
        [value is not None for value in rows.b_factor],
        dtype=torch.bool,
    )
    b_factor_mask[source_atom_index] = source_b_factor_mask
    b_factor[source_atom_index] = torch.tensor(
        [0.0 if value is None else float(value) for value in rows.b_factor],
        dtype=torch.float32,
    )
    if not bool(coordinate_mask.any()):
        raise ValueError("selected structure contains no observed coordinates")

    source_symbols: list[str] = []
    source_symbol_index: dict[str, int] = {}

    def intern(value: str) -> int:
        index = source_symbol_index.get(value)
        if index is None:
            index = len(source_symbols)
            source_symbol_index[value] = index
            source_symbols.append(value)
        return index

    residue_id = torch.full((component_count,), -1, dtype=torch.int64)
    insertion_code = torch.full((component_count,), -1, dtype=torch.int64)
    for reference, (number, code) in residue_numbers.items():
        component_index = component_index_by_reference[reference]
        residue_id[component_index] = intern(number)
        if code:
            insertion_code[component_index] = intern(code)
    resolution: torch.Tensor | None = None
    resolution_is_known: torch.Tensor | None = None
    if format_name == "mmcif":
        assert cif_file is not None
        from .mmcif import read_mmcif_resolution

        source_resolution = read_mmcif_resolution(cif_file)
        resolution = torch.tensor(-1.0 if source_resolution is None else source_resolution, dtype=torch.float32)
        resolution_is_known = torch.tensor(source_resolution is not None, dtype=torch.bool)
    return Structure(  # type: ignore[call-arg]
        system=system,
        coordinates=coordinates,
        coordinate_mask=coordinate_mask,
        occupancy=occupancy,
        occupancy_mask=occupancy_mask,
        b_factor=b_factor,
        b_factor_mask=b_factor_mask,
        source_symbols=tuple(source_symbols),
        residue_id=residue_id,
        insertion_code=insertion_code,
        resolution=resolution,
        resolution_is_known=resolution_is_known,
        batch_size=[],
    )


def _canonical_polymer_leaving_sites(
    entities: Mapping[str, PolymerSequence | Molecule],
    instances: Mapping[str, str],
    *,
    instance_values: Mapping[str, PolymerSequence | Molecule] | None = None,
) -> frozenset[AtomSite]:
    """Return instance-local atoms removed by realized canonical polymer links."""

    sites: set[AtomSite] = set()
    for instance_id, entity_id in instances.items():
        declared_entity = entities[entity_id]
        entity = declared_entity if instance_values is None else instance_values.get(instance_id, declared_entity)
        if not isinstance(entity, PolymerSequence):
            continue
        for component_index, (left, right) in enumerate(zip(entity.components, entity.components[1:])):
            next_attachment = left.next_attachment
            previous_attachment = right.previous_attachment
            if next_attachment is None or previous_attachment is None:
                continue
            sites.update(
                AtomSite(instance_id, component_index, atom_id)
                for atom_id in next_attachment.leaving_atom_ids
            )
            sites.update(
                AtomSite(instance_id, component_index + 1, atom_id)
                for atom_id in previous_attachment.leaving_atom_ids
            )
    return frozenset(sites)


def _reuse_entity(
    entities: dict[str, PolymerSequence | Molecule],
    entity_id: str,
    entity: PolymerSequence | Molecule,
) -> PolymerSequence | Molecule:
    previous = entities.get(entity_id)
    if previous is None:
        entities[entity_id] = entity
        return entity
    if previous != entity:
        raise ValueError(f"source entity ID {entity_id!r} has conflicting molecular definitions")
    return previous


def _pdb_component_polymer_type(component: ChemicalComponent) -> PolymerType | None:
    polymer_type = get_builtin_component_polymer_type(component.component_id)
    if polymer_type is not None:
        return polymer_type
    if (
        component.previous_attachment is not None
        and component.next_attachment is not None
        and component.previous_attachment.atom_id == "N"
        and component.next_attachment.atom_id == "C"
    ):
        return "protein"
    elements = {atom.atom_id: atom.element for atom in component.atoms}
    if all(elements.get(atom_id) == element for atom_id, element in (("N", "N"), ("CA", "C"), ("C", "C"))):
        return "protein"
    return None
