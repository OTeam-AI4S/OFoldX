r"""MPNN-specific PDB reading for ProteinMPNN, LigandMPNN, and MpnnPacker.

``read_mpnn_pdb`` reads the first physical model, retains blank and ``A`` alternate
locations in source order after optional chain filtering, and decodes hybrid-36
residue numbers. Standard amino acids reconstruct their non-hydrogen atom14
vocabulary (excluding ``OXT``) plus retained resolvable atoms from the built-in
component definitions; ``UNK`` keeps only retained resolvable atoms, while other
protein and nonprotein occurrences use open components built from retained source
atoms.
"""

from __future__ import annotations

import gzip
from collections.abc import Sequence
from os import PathLike
from pathlib import Path

import numpy as np
from biotite.rust.structure.io.pdb import decode_hybrid36

from ..data import ATOMIC_NUMBER_BY_ELEMENT, STANDARD_AMINO_ACID_IDS
from ..data.components import get_component
from ..data.molecular import (
    AtomSite,
    ChemicalAtom,
    ChemicalComponent,
    ComponentReference,
    MolecularSystem,
    Molecule,
    PolymerSequence,
)
from ..data.structure import ResidueNumber, Structure

_PROTEIN_COMPONENT_IDS = frozenset(
    {
        *STANDARD_AMINO_ACID_IDS,
        "ASX",
        "GLX",
        "CME",
        "CSB",
        "CSO",
        "GLH",
        "HIP",
        "HSE",
        "LYN",
        "MEN",
        "MSE",
        "PHD",
        "PTR",
        "SEC",
        "SEP",
        "TPO",
        "UNK",
    }
)
_ATOM_RECORD_PROTEIN_COMPONENT_IDS = frozenset({"ASH", "ARN", "CYX", "HID", "HIE", "HSD", "HSP", "TYM", "XAA", "XLE"})


def read_mpnn_pdb(
    path: str | PathLike[str],
    *,
    chains: Sequence[str] | None = None,
) -> Structure:
    r"""Read one PDB through the input policy shared by OFoldX MPNN models."""

    source = Path(path).expanduser()
    if source.suffix == ".gz":
        handle = gzip.open(source, mode="rt", encoding="ascii")
    else:
        handle = source.open(encoding="ascii")
    component_ids: dict[tuple[str, int, str], str] = {}
    occurrence_order: list[tuple[str, int, str]] = []
    atom_occurrences: set[tuple[str, int, str]] = set()
    atom_rows: dict[
        tuple[str, int, str],
        list[tuple[str, str, int | None]],
    ] = {}
    rows: list[
        tuple[
            tuple[str, int, str],
            str,
            tuple[float, float, float],
            float,
        ]
    ] = []
    explicit_model = False
    implicit_model = False
    with handle:
        for line in handle:
            record = line[:6].strip()
            if record == "MODEL":
                if explicit_model or implicit_model:
                    break
                explicit_model = True
                continue
            if record == "ENDMDL":
                if explicit_model:
                    break
                continue
            if record not in {"ATOM", "HETATM"}:
                continue
            implicit_model = not explicit_model

            chain_id = line[21:22].strip()
            if chains is not None and chain_id not in chains:
                continue
            if line[16:17].strip() not in {"", "A"}:
                continue

            number = int(decode_hybrid36(line[22:26]))
            insertion_code = line[26:27].strip()
            occurrence = (chain_id, number, insertion_code)
            component_id = line[17:20].strip()
            previous_component_id = component_ids.get(occurrence)
            if previous_component_id is None:
                component_ids[occurrence] = component_id
            elif previous_component_id != component_id:
                raise ValueError(
                    f"PDB occurrence {occurrence!r} has conflicting component IDs "
                    f"{previous_component_id!r} and {component_id!r}"
                )
            if record == "ATOM":
                atom_occurrences.add(occurrence)
            if occurrence not in atom_rows:
                occurrence_order.append(occurrence)
                atom_rows[occurrence] = []

            atom_id = line[12:16].strip()
            element = line[76:78].strip().title()
            isotope = None
            if element == "D":
                element = "H"
                isotope = 2
            elif element == "T":
                element = "H"
                isotope = 3
            coordinate = (
                float(line[30:38]),
                float(line[38:46]),
                float(line[46:54]),
            )
            occupancy_text = line[54:60].strip()
            occupancy = float(occupancy_text) if occupancy_text else np.nan
            atom_rows[occurrence].append((atom_id, element, isotope))
            rows.append((occurrence, atom_id, coordinate, occupancy))

    protein_occurrences: dict[str, list[tuple[str, int, str]]] = {}
    is_protein: dict[tuple[str, int, str], bool] = {}
    components: dict[tuple[str, int, str], ChemicalComponent] = {}
    for occurrence in occurrence_order:
        component_id = component_ids[occurrence]
        protein = component_id in _PROTEIN_COMPONENT_IDS or (
            component_id in _ATOM_RECORD_PROTEIN_COMPONENT_IDS and occurrence in atom_occurrences
        )
        is_protein[occurrence] = protein
        if protein:
            protein_occurrences.setdefault(occurrence[0], []).append(occurrence)
        if protein and (component_id in STANDARD_AMINO_ACID_IDS or component_id == "UNK"):
            definition = get_component(component_id)
            definition_atoms = {atom.atom_id: atom for atom in definition.atoms}
            retained_atoms: dict[str, tuple[str, int | None]] = {}
            for atom_id, element, isotope in atom_rows[occurrence]:
                definition_atom = definition_atoms.get(atom_id)
                if definition_atom is None:
                    continue
                if atom_id in retained_atoms:
                    raise ValueError(f"PDB occurrence {occurrence!r} contains duplicate atom {atom_id!r}")
                if not element:
                    element = definition_atom.element
                    isotope = definition_atom.isotope
                elif element not in ATOMIC_NUMBER_BY_ELEMENT:
                    raise ValueError(
                        f"PDB atom {atom_id!r} in occurrence {occurrence!r} has invalid element {element!r}"
                    )
                elif element != definition_atom.element:
                    raise ValueError(
                        f"PDB atom {atom_id!r} in occurrence {occurrence!r} has element {element!r}, "
                        f"expected {definition_atom.element!r}"
                    )
                retained_atoms[atom_id] = (element, isotope)
            declared_atoms = {
                atom.atom_id
                for atom in definition.atoms
                if component_id in STANDARD_AMINO_ACID_IDS and atom.element != "H" and atom.atom_id != "OXT"
            }
            declared_atoms.update(retained_atoms)
            components[occurrence] = ChemicalComponent(
                component_id=component_id,
                atoms=tuple(
                    ChemicalAtom(
                        atom_id=atom.atom_id,
                        element=retained_atoms.get(atom.atom_id, (atom.element, atom.isotope))[0],
                        isotope=retained_atoms.get(atom.atom_id, (atom.element, atom.isotope))[1],
                    )
                    for atom in definition.atoms
                    if atom.atom_id in declared_atoms
                ),
                bonds=None,
            )
        else:
            atoms: list[ChemicalAtom] = []
            for atom_id, element, isotope in atom_rows[occurrence]:
                if not element or element not in ATOMIC_NUMBER_BY_ELEMENT:
                    raise ValueError(
                        f"PDB atom {atom_id!r} in occurrence {occurrence!r} has invalid element {element!r}"
                    )
                atoms.append(ChemicalAtom(atom_id=atom_id, element=element, isotope=isotope))
            components[occurrence] = ChemicalComponent(
                component_id=component_id,
                atoms=tuple(atoms),
                bonds=None,
            )

    component_references: dict[tuple[str, int, str], ComponentReference] = {}
    entities: dict[str, PolymerSequence | Molecule] = {}
    instances: dict[str, str] = {}
    protein_component_indices: dict[tuple[str, int, str], int] = {}
    for occurrences in protein_occurrences.values():
        for component_index, occurrence in enumerate(occurrences):
            protein_component_indices[occurrence] = component_index

    seen_protein_chains: set[str] = set()
    molecule_count = 0
    for occurrence in occurrence_order:
        if is_protein[occurrence]:
            chain_id = occurrence[0]
            component_references[occurrence] = (
                chain_id,
                protein_component_indices[occurrence],
            )
            if chain_id in seen_protein_chains:
                continue
            seen_protein_chains.add(chain_id)
            entity = PolymerSequence(
                "protein",
                tuple(components[item] for item in protein_occurrences[chain_id]),
            )
            entities[chain_id] = entity
            instances[chain_id] = chain_id
        else:
            molecule_count += 1
            instance_id = f"molecule_{molecule_count}"
            component_references[occurrence] = (instance_id, 0)
            entities[instance_id] = Molecule((components[occurrence],))
            instances[instance_id] = instance_id

    system = MolecularSystem(entities, instances)
    realized_sites = {site for site, _atom in system.iter_atoms()}
    sites: list[AtomSite] = []
    coordinates: list[tuple[float, float, float]] = []
    occupancies: list[float] = []
    for occurrence, atom_id, coordinate, occupancy in rows:
        instance_id, component_index = component_references[occurrence]
        site = AtomSite(instance_id, component_index, atom_id)
        if site not in realized_sites:
            continue
        sites.append(site)
        coordinates.append(coordinate)
        occupancies.append(occupancy)

    residue_numbers = {
        component_references[occurrence]: ResidueNumber(occurrence[1], occurrence[2])
        for occurrence in occurrence_order
        if is_protein[occurrence]
    }
    return Structure(
        system,
        tuple(sites),
        np.asarray(coordinates).reshape((-1, 3)),
        np.ones(len(sites), dtype=np.bool_),
        np.asarray(occupancies),
        residue_numbers,
    )
