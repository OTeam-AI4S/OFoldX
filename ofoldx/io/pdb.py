r"""Canonical PDB reader shared by the released MPNN models.

This module owns :func:`read_pdb` (including its hybrid-36 residue-number decoding
with the biotite layout fallback) and the element/protein-component classification
tables it depends on.
"""

from __future__ import annotations

import gzip
import string
from collections.abc import Sequence
from os import PathLike
from pathlib import Path

import numpy as np

try:
    # biotite >= 1.7 exposes hybrid-36 codecs from the Rust extension.
    from biotite.rust.structure.io.pdb import decode_hybrid36, encode_hybrid36
except ImportError:  # pragma: no cover - older biotite layout
    from biotite.structure.io.pdb.hybrid36 import decode_hybrid36, encode_hybrid36

from ..data import STANDARD_AMINO_ACID_IDS, is_water_component
from ..data import residue_constants as rc
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
from ..results import PackPrediction

_ELEMENTS = (
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
)
ATOMIC_NUMBER_BY_ELEMENT = {element: index for index, element in enumerate(_ELEMENTS, 1)}
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


def read_pdb(
    path: str | PathLike[str],
    *,
    chains: Sequence[str] | None = None,
) -> Structure:
    r"""Read the PDB representation shared by the released MPNN models."""

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

            number = decode_hybrid36(line[22:26])
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
            coordinates = tuple(float(line[start : start + 8]) for start in (30, 38, 46))
            occupancy_text = line[54:60].strip()
            occupancy = float(occupancy_text) if occupancy_text else np.nan
            atom_rows[occurrence].append((atom_id, element, isotope))
            rows.append((occurrence, atom_id, coordinates, occupancy))

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
            audited = get_component(component_id)
            audited_atoms = {atom.atom_id: atom for atom in audited.atoms}
            retained_atoms: dict[str, tuple[str, int | None]] = {}
            for atom_id, element, isotope in atom_rows[occurrence]:
                audited_atom = audited_atoms.get(atom_id)
                if audited_atom is None:
                    continue
                if atom_id in retained_atoms:
                    raise ValueError(f"PDB occurrence {occurrence!r} contains duplicate atom {atom_id!r}")
                if not element:
                    element = audited_atom.element
                    isotope = audited_atom.isotope
                elif element not in ATOMIC_NUMBER_BY_ELEMENT:
                    raise ValueError(
                        f"PDB atom {atom_id!r} in occurrence {occurrence!r} has invalid element {element!r}"
                    )
                elif element != audited_atom.element:
                    raise ValueError(
                        f"PDB atom {atom_id!r} in occurrence {occurrence!r} has element {element!r}, "
                        f"expected {audited_atom.element!r}"
                    )
                retained_atoms[atom_id] = (element, isotope)
            declared_atoms = {
                atom.atom_id
                for atom in audited.atoms
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
                    for atom in audited.atoms
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
    for chain_id, occurrences in protein_occurrences.items():
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


def write_pdb(
    prediction: PackPrediction,
    path: str | PathLike[str],
) -> None:
    r"""Write one packed prediction as its canonical PDB visualization."""

    structure = prediction.structure
    system = structure.system
    record_indices = tuple(
        index
        for index, site in enumerate(structure.sites)
        if bool(structure.coordinate_mask[index])
        and not is_water_component(system.component((site.instance_id, site.component_index)))
    )
    retained_instances = {structure.sites[index].instance_id for index in record_indices}
    instance_ids = tuple(instance_id for instance_id in system.instances if instance_id in retained_instances)
    chain_alphabet = string.ascii_uppercase + string.ascii_lowercase + string.digits + string.punctuation + " "
    chain_by_instance: dict[str, str] = {}
    retained_chain_ids = {
        instance_id
        for instance_id in instance_ids
        if len(instance_id) == 1 and instance_id.isascii() and instance_id.isprintable() and not instance_id.isspace()
    }
    used_chains = set(retained_chain_ids)
    if "" in retained_instances:
        chain_by_instance[""] = " "
        used_chains.add(" ")
    available_chains = (chain for chain in chain_alphabet if chain not in used_chains)
    for instance_id in instance_ids:
        if instance_id == "":
            continue
        if instance_id in retained_chain_ids:
            chain_by_instance[instance_id] = instance_id
            continue
        try:
            chain_by_instance[instance_id] = next(available_chains)
        except StopIteration:
            raise ValueError("PDB cannot encode all MolecularSystem instance IDs as unique chains") from None

    density_by_group = {}
    for chi_index in range(4):
        metric = prediction.metrics.get(f"chi{chi_index + 1}_log_density")
        if metric is not None:
            density_by_group[chi_index + 4] = dict(zip(metric.axis, metric.values, strict=True))
    restype_index = {
        component_id: index for index, component_id in enumerate(rc.restype_1to3[restype] for restype in rc.restypes)
    }

    lines: list[str] = []
    serialized_reference_by_key: dict[tuple[str, int, str], ComponentReference] = {}
    serial = 0
    for site_index in record_indices:
        site = structure.sites[site_index]
        reference = (site.instance_id, site.component_index)
        component = system.component(reference)

        serial += 1
        serial_field = f"{encode_hybrid36(serial, 5):>5}"
        entity = system.entities[system.instances[site.instance_id]]
        record = "ATOM" if isinstance(entity, PolymerSequence) and entity.polymer_type == "protein" else "HETATM"
        component_id = component.component_id
        atom_id = site.atom_id
        atom = system.atom(site)
        element = atom.element
        if not component_id or len(component_id) > 3:
            raise ValueError(f"PDB cannot encode component ID {component_id!r}")
        if not atom_id or len(atom_id) > 4:
            raise ValueError(f"PDB cannot encode atom ID {atom_id!r}")
        if not element or len(element) > 2:
            raise ValueError(f"PDB cannot encode element {element!r}")
        if any(character.isspace() for value in (component_id, atom_id) for character in value):
            raise ValueError("PDB component and atom IDs cannot contain whitespace")
        if any(not value.isascii() or not value.isprintable() for value in (component_id, atom_id, element)):
            raise ValueError("PDB identity fields must contain printable ASCII characters")
        if len(atom_id) < 4 and len(element) == 1 and not atom_id[0].isdigit():
            atom_field = f" {atom_id:<3}"
        else:
            atom_field = f"{atom_id:<4}"

        residue_number = None if structure.residue_numbers is None else structure.residue_numbers.get(reference)
        number = site.component_index + 1 if residue_number is None else residue_number.number
        insertion_code = "" if residue_number is None else residue_number.insertion_code
        chain = chain_by_instance[site.instance_id]
        if number < 0:
            residue_field = f"{number:>4}"
            if len(residue_field) != 4:
                raise ValueError(f"PDB cannot encode residue number {number}")
        else:
            residue_field = f"{encode_hybrid36(number, 4):>4}"
        if insertion_code and (
            len(insertion_code) != 1 or not insertion_code.isascii() or not insertion_code.isalpha()
        ):
            raise ValueError(f"PDB cannot encode insertion code {insertion_code!r}")
        occurrence_key = (chain, number, insertion_code)
        serialized_reference = serialized_reference_by_key.setdefault(occurrence_key, reference)
        if serialized_reference != reference:
            raise ValueError(
                f"PDB component occurrences {serialized_reference!r} and {reference!r} "
                f"collide at {occurrence_key!r}"
            )

        coordinate_fields = tuple(f"{float(value):8.3f}" for value in structure.coordinates[site_index])
        if (
            any(len(value) != 8 for value in coordinate_fields)
            or not np.isfinite(structure.coordinates[site_index]).all()
        ):
            raise ValueError(f"PDB cannot encode coordinates for {site!r}")
        if structure.occupancy is None or np.isnan(structure.occupancy[site_index]):
            occupancy_field = " " * 6
        else:
            occupancy = float(structure.occupancy[site_index])
            occupancy_field = f"{occupancy:6.2f}"
            if len(occupancy_field) != 6 or not np.isfinite(occupancy):
                raise ValueError(f"PDB cannot encode occupancy {occupancy!r}")

        b_value = 0.0
        restype = restype_index.get(component_id)
        if isinstance(entity, PolymerSequence) and entity.polymer_type == "protein" and restype is not None:
            atom14_names = rc.restype_name_to_atom14_names[component_id]
            if atom_id in atom14_names:
                atom14_index = atom14_names.index(atom_id)
                rigid_group = int(rc.restype_atom14_to_rigid_group[restype, atom14_index])
                density = density_by_group.get(rigid_group, {}).get(reference)
                if density is not None:
                    b_value = float(density)
        b_field = f"{b_value:6.2f}"
        if len(b_field) != 6 or not np.isfinite(b_value):
            raise ValueError(f"PDB cannot encode chi log density {b_value!r}")

        lines.append(
            f"{record:<6}{serial_field} {atom_field} {component_id:>3} {chain}{residue_field}"
            f"{insertion_code or ' '}   {''.join(coordinate_fields)}{occupancy_field}{b_field}"
            f"          {element:>2}  "
        )

    lines.append("END")
    Path(path).expanduser().write_text("\n".join(lines) + "\n", encoding="ascii")
