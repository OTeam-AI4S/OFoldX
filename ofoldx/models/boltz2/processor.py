r"""
Input processing for Boltz-2 folding and affinity scoring.
"""

from __future__ import annotations

import math
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
from multimolecule.io import read_fasta_records
from torch import Tensor

from ...data.evidence import MSA, MSACollection, msa_bindings, resolve_msa_bindings
from ...data.molecular import AtomSite, MolecularSystem, Molecule, PolymerSequence
from ...data.structure import Structure
from ...io.structure import read_structure
from ...processor import Processor
from .model import Boltz2Input, Boltz2Layout

if TYPE_CHECKING:
    from rdkit.Chem.rdchem import Mol

_PROTEIN_RESIDUE_TYPES = {
    component_id: index
    for index, component_id in enumerate(
        (
            "ALA",
            "ARG",
            "ASN",
            "ASP",
            "CYS",
            "GLN",
            "GLU",
            "GLY",
            "HIS",
            "ILE",
            "LEU",
            "LYS",
            "MET",
            "PHE",
            "PRO",
            "SER",
            "THR",
            "TRP",
            "TYR",
            "VAL",
        ),
        start=2,
    )
}
_RNA_RESIDUE_TYPES = {"A": 23, "G": 24, "C": 25, "U": 26, "N": 27}
_DNA_RESIDUE_TYPES = {"DA": 28, "DG": 29, "DC": 30, "DT": 31, "DN": 32}
_STANDARD_COMPONENTS = {
    "protein": frozenset(_PROTEIN_RESIDUE_TYPES),
    "rna": frozenset(_RNA_RESIDUE_TYPES),
    "dna": frozenset(_DNA_RESIDUE_TYPES),
}
_MOLECULE_TYPES = {"protein": 0, "dna": 1, "rna": 2}
_UNKNOWN_RESIDUE_TYPES = {"protein": 22, "rna": 27, "dna": 32}
_MSA_GAP = 1
# MSA cells are UPPERCASE one-letter MSA symbols (or the ``-`` gap), not ChemicalComponent IDs. The
# Processor owns the symbol->token mapping keyed by the entity's resolved polymer type. Standard
# residues map to their model token and every other component (ambiguity codes and unknowns) maps to
# the model's UNKNOWN token. ``_PROTEIN_MSA_RESTYPE``/``_DNA_MSA_RESTYPE`` project one-letter symbols onto the
# three-letter tokens Boltz-2 already uses; RNA one-letter symbols equal their component IDs directly.
_PROTEIN_MSA_RESTYPE = {
    symbol: _PROTEIN_RESIDUE_TYPES[component_id]
    for symbol, component_id in {
        "A": "ALA",
        "R": "ARG",
        "N": "ASN",
        "D": "ASP",
        "C": "CYS",
        "Q": "GLN",
        "E": "GLU",
        "G": "GLY",
        "H": "HIS",
        "I": "ILE",
        "L": "LEU",
        "K": "LYS",
        "M": "MET",
        "F": "PHE",
        "P": "PRO",
        "S": "SER",
        "T": "THR",
        "W": "TRP",
        "Y": "TYR",
        "V": "VAL",
    }.items()
}
_DNA_MSA_RESTYPE = {
    symbol: _DNA_RESIDUE_TYPES[component_id]
    for symbol, component_id in {"A": "DA", "G": "DG", "C": "DC", "T": "DT", "N": "DN"}.items()
}


def _msa_symbol_restype(symbol: str, polymer_type: str) -> int:
    r"""
    Map one UPPERCASE one-letter MSA symbol (or ``-``) to a Boltz-2 token.
    """

    if symbol == "-":
        return _MSA_GAP
    if polymer_type == "protein":
        return _PROTEIN_MSA_RESTYPE.get(symbol, _UNKNOWN_RESIDUE_TYPES["protein"])
    if polymer_type == "rna":
        return _RNA_RESIDUE_TYPES.get(symbol, _UNKNOWN_RESIDUE_TYPES["rna"])
    return _DNA_MSA_RESTYPE.get(symbol, _UNKNOWN_RESIDUE_TYPES["dna"])


_MAX_TOKENS = 256
_MAX_POLYMER_TOKENS = 200
_MAX_ATOMS = 2048
_CROP_NEIGHBORHOOD = 10


class Boltz2AffinityProcessor(Processor):
    r"""
    Convert protein-ligand systems into `Boltz2Input` for affinity scoring.

    Args:
        device: Device for produced tensors.
        dtype: Floating-point dtype for produced tensors.
    """

    def __init__(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(device=device, dtype=dtype)

    def __call__(
        self,
        input: MolecularSystem | Structure | str | PathLike[str],
        *,
        ligand: str | None = None,
        msa: MSA | MSACollection | str | PathLike[str] | None = None,
        generator: torch.Generator | None = None,
    ) -> Boltz2Input:
        r"""
        Prepare affinity features for one protein-ligand system.

        Args:
            input: Molecular system, structure, or local structure path.
            ligand: Molecule instance scored by the affinity head. May be omitted for a system
                containing exactly one molecule instance.
            msa: Optional MSA value, collection, or alignment path.
            generator: Optional generator for independent rigid transforms of reference
                conformers.

        Returns:
            Ligand-centered token, atom, MSA, and affinity tensors. Reference
            coordinates are in angstroms; tensor axes align to ``layout``.
        """

        if isinstance(input, str | PathLike):
            input = read_structure(Path(input).expanduser())
        if not isinstance(input, MolecularSystem | Structure):
            raise TypeError("Boltz2AffinityProcessor input must be a MolecularSystem, Structure, or structure path")

        if isinstance(input, Structure):
            structure = input
            system = input.system
        else:
            structure = None
            system = input
        protein_instances = [
            instance_id
            for instance_id, entity_id in system.instances.items()
            if isinstance((entity := system.entities[entity_id]), PolymerSequence) and entity.polymer_type == "protein"
        ]
        if not protein_instances:
            raise ValueError("Boltz-2 affinity requires a protein receptor")
        molecule_instances = [
            instance_id
            for instance_id, entity_id in system.instances.items()
            if isinstance(system.entities[entity_id], Molecule)
        ]
        if ligand is None:
            if len(molecule_instances) != 1:
                raise ValueError("ligand is required unless the system contains exactly one Molecule instance")
            ligand = molecule_instances[0]
        ligand_entity_id = system.instances[ligand]
        ligand_entity = system.entities[ligand_entity_id]
        if not isinstance(ligand_entity, Molecule):
            raise ValueError(f"ligand instance {ligand!r} does not reference a Molecule")
        if sum(entity_id == ligand_entity_id for entity_id in system.instances.values()) != 1:
            raise ValueError("affinity scoring does not support copied ligand entities")
        ligand_atom_count, affinity_mw = _affinity_ligand_properties(ligand_entity)
        if ligand_atom_count > 128:
            raise ValueError("the affinity head supports at most 128 ligand atoms after removing hydrogens")

        normalized_msa = _normalize_msa(msa, system)
        return _prepare_boltz2_affinity_features(
            system,
            structure=structure,
            ligand=ligand,
            affinity_mw=affinity_mw,
            msa=normalized_msa,
            device=self.device,
            dtype=self.dtype,
            generator=generator,
        )


class Boltz2Processor(Processor):
    r"""
    Convert supported biomolecular inputs into `Boltz2Input`.

    Args:
        device: Device for produced tensors.
        dtype: Floating-point dtype for produced tensors.
    """

    def __init__(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(device=device, dtype=dtype)

    def __call__(
        self,
        input: MolecularSystem | Structure | str | PathLike[str],
        *,
        msa: MSA | MSACollection | str | PathLike[str] | None = None,
        generator: torch.Generator | None = None,
    ) -> Boltz2Input:
        r"""
        Prepare folding features for one molecular system.

        Args:
            input: Molecular system, structure, or local structure path.
            msa: Optional MSA value, collection, or alignment path.
            generator: Optional generator for independent rigid transforms of reference
                conformers.

        Returns:
            Batched token, atom, and MSA tensors. Reference coordinates are in
            angstroms; tensor axes align to ``layout``.
        """

        if isinstance(input, str | PathLike):
            input = read_structure(Path(input).expanduser())
        if not isinstance(input, MolecularSystem | Structure):
            raise TypeError("Boltz2Processor input must be a MolecularSystem, Structure, or structure path")

        if isinstance(input, Structure):
            structure = input
            system = input.system
        else:
            structure = None
            system = input
        if not system.instances:
            raise ValueError("Boltz-2 requires at least one molecular instance")

        normalized_msa = _normalize_msa(msa, system)
        return _prepare_boltz2_structure_features(
            system,
            structure=structure,
            msa=normalized_msa,
            device=self.device,
            dtype=self.dtype,
            generator=generator,
        )


def _prepare_boltz2_affinity_features(
    system: MolecularSystem,
    *,
    structure: Structure | None,
    ligand: str,
    affinity_mw: float,
    msa: MSA | MSACollection | None,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None = None,
) -> Boltz2Input:
    r"""
    Build the affinity token, atom, MSA, and identity layout.
    """

    available_sites = {site for site, atom in system.iter_atoms() if atom.element != "H"}
    entity_index_by_id: dict[str, int] = {}
    instance_count_by_entity: dict[str, int] = {}
    tokens: list[dict[str, object]] = []

    for chain_index, (instance_id, entity_id) in enumerate(system.instances.items()):
        entity = system.entities[entity_id]
        entity_index = entity_index_by_id.setdefault(entity_id, len(entity_index_by_id))
        symmetry_index = instance_count_by_entity.get(entity_id, 0)
        instance_count_by_entity[entity_id] = symmetry_index + 1
        if isinstance(entity, PolymerSequence):
            molecule_type = _MOLECULE_TYPES[entity.polymer_type]
            standards = _STANDARD_COMPONENTS[entity.polymer_type]
            for monomer_index, component in enumerate(entity.components):
                component_id = component.component_id
                atom_sites = tuple(
                    site
                    for atom in component.atoms
                    if atom.element != "H"
                    and atom.atom_id not in {"OXT", "OP3"}
                    and (site := AtomSite(instance_id, monomer_index, atom.atom_id)) in available_sites
                )
                if not atom_sites:
                    raise ValueError(f"component {component_id!r} has no model-visible heavy atoms")
                is_standard = component_id in standards
                residue_type = (
                    _residue_type(component_id, entity.polymer_type)
                    if is_standard
                    else _UNKNOWN_RESIDUE_TYPES[entity.polymer_type]
                )
                by_name = {site.atom_id: site for site in atom_sites}
                center_name = "CA" if entity.polymer_type == "protein" else "C1'"
                if entity.polymer_type == "protein":
                    distogram_name = "CA" if component_id == "GLY" else "CB"
                    frame_names = ("N", "CA", "C") if is_standard else ("CA", "N", "C")
                else:
                    distogram_name = (
                        "C4"
                        if component_id in {"A", "G", "DA", "DG"}
                        else "C2" if component_id in {"C", "U", "DC", "DT"} else "C1'"
                    )
                    frame_names = ("C1'", "C3'", "C4'")
                center_site = by_name.get(center_name, atom_sites[0])
                distogram_site = by_name.get(distogram_name, center_site)
                tokens.append(
                    {
                        "atom_sites": atom_sites,
                        "token_site": (instance_id, monomer_index),
                        "instance_id": instance_id,
                        "entity_id": entity_id,
                        "monomer_index": monomer_index,
                        "residue_index": monomer_index,
                        "chain_index": chain_index,
                        "entity_index": entity_index,
                        "symmetry_index": symmetry_index,
                        "molecule_type": molecule_type,
                        "residue_type": residue_type,
                        "modified": not is_standard,
                        "center_site": center_site,
                        "distogram_site": distogram_site,
                        "frame_sites": tuple(by_name.get(name, center_site) for name in frame_names),
                        "affinity": False,
                    }
                )
        else:
            atom_sites = tuple(
                site
                for component_index, component in enumerate(entity.components)
                for atom in component.atoms
                if atom.element != "H"
                and (site := AtomSite(instance_id, component_index, atom.atom_id)) in available_sites
            )
            if not atom_sites:
                raise ValueError(f"molecule {instance_id!r} has no model-visible heavy atoms")
            for site in atom_sites:
                tokens.append(
                    {
                        "atom_sites": (site,),
                        "token_site": site,
                        "instance_id": instance_id,
                        "entity_id": entity_id,
                        "monomer_index": None,
                        "residue_index": 0,
                        "chain_index": chain_index,
                        "entity_index": entity_index,
                        "symmetry_index": symmetry_index,
                        "molecule_type": 3,
                        "residue_type": 22,
                        "modified": False,
                        "center_site": site,
                        "distogram_site": site,
                        "frame_sites": (site, site, site),
                        "affinity": instance_id == ligand,
                    }
                )

    selected = _affinity_crop(tokens, structure=structure)
    tokens = [tokens[index] for index in selected]
    if not any(cast(bool, token["affinity"]) for token in tokens):
        raise ValueError("the affinity crop does not contain the selected ligand")

    atom_sites = tuple(site for token in tokens for site in cast(tuple[AtomSite, ...], token["atom_sites"]))
    site_to_atom = {site: index for index, site in enumerate(atom_sites)}
    site_to_token = {
        site: token_index
        for token_index, token in enumerate(tokens)
        for site in cast(tuple[AtomSite, ...], token["atom_sites"])
    }
    atom_count = len(atom_sites)
    padded_atom_count = max(32, math.ceil(atom_count / 32) * 32)
    reference_coordinates = _reference_coordinates(system, atom_sites)
    reference_space_by_residue: dict[tuple[str, int], int] = {}
    reference_space_ids = [
        reference_space_by_residue.setdefault(
            (site.instance_id, site.component_index),
            len(reference_space_by_residue),
        )
        for site in atom_sites
    ]
    resolved = torch.zeros(atom_count, dtype=torch.bool)
    target_positions = torch.zeros(padded_atom_count, 3)
    if structure is not None:
        coordinate_index = {site: index for index, site in enumerate(structure.sites)}
        for atom_index, site in enumerate(atom_sites):
            source_index = coordinate_index.get(site)
            if source_index is not None and bool(structure.coordinate_mask[source_index]):
                resolved[atom_index] = True
                target_positions[atom_index] = torch.from_numpy(structure.coordinates[source_index].copy())

    reference_positions = torch.zeros(padded_atom_count, 3)
    reference_positions[:atom_count] = torch.tensor(reference_coordinates)
    reference_space_id = torch.zeros(padded_atom_count, dtype=torch.long)
    reference_space_id[:atom_count] = torch.tensor(reference_space_ids)
    _augment_reference_positions(
        reference_positions[:atom_count],
        reference_space_id[:atom_count],
        resolved,
        generator=generator,
    )

    atomic_number = torch.zeros(padded_atom_count, dtype=torch.long)
    reference_charges = torch.zeros(padded_atom_count)
    atom_name_characters = torch.zeros(padded_atom_count, 4, dtype=torch.long)
    atom_mask = torch.zeros(padded_atom_count, dtype=torch.bool)
    atom_to_token = torch.zeros(padded_atom_count, dtype=torch.long)
    atom_token_index = torch.zeros(padded_atom_count, dtype=torch.long)
    atom_names = _model_atom_names(system, atom_sites)
    from rdkit.Chem import GetPeriodicTable

    periodic_table = GetPeriodicTable()
    for atom_index, (site, atom_name) in enumerate(zip(atom_sites, atom_names)):
        atom = system.atom(site)
        atomic_number[atom_index] = periodic_table.GetAtomicNumber(atom.element)
        reference_charges[atom_index] = atom.formal_charge
        atom_name_characters[atom_index] = torch.tensor(
            [ord(character) - 32 if character != " " else 0 for character in atom_name.ljust(4)]
        )
        atom_mask[atom_index] = True
        token_index = site_to_token[site]
        atom_to_token[atom_index] = token_index
        atom_token_index[atom_index] = cast(tuple[AtomSite, ...], tokens[token_index]["atom_sites"]).index(site)

    atom_features = torch.cat(
        (
            reference_positions,
            reference_charges.unsqueeze(-1),
            torch.nn.functional.one_hot(atomic_number, 128).float(),
            torch.nn.functional.one_hot(atom_name_characters, 64).reshape(padded_atom_count, 256).float(),
        ),
        dim=-1,
    )
    token_count = len(tokens)
    token_to_rep_atom = torch.tensor(
        [site_to_atom[cast(AtomSite, token["distogram_site"])] for token in tokens],
        dtype=torch.long,
    )
    representative_atom_mask = torch.zeros(padded_atom_count, dtype=torch.bool)
    representative_atom_mask[token_to_rep_atom] = True
    token_bonds, type_bonds = _token_bond_features(system, site_to_token, token_count)
    msa_values = _msa_features(msa, tokens, system)
    molecule_types = torch.tensor([cast(int, token["molecule_type"]) for token in tokens])
    residue_types = torch.tensor([cast(int, token["residue_type"]) for token in tokens])
    frame_atom_index = torch.tensor(
        [
            [site_to_atom[site] for site in cast(tuple[AtomSite, AtomSite, AtomSite], token["frame_sites"])]
            for token in tokens
        ],
        dtype=torch.long,
    )
    frame_mask = torch.tensor(
        [
            all(resolved[site_to_atom[site]] for site in cast(tuple[AtomSite, ...], token["frame_sites"]))
            for token in tokens
        ]
    )
    cyclic_period = torch.tensor(
        [_cyclic_period(system, cast(str, token["instance_id"])) for token in tokens],
        dtype=torch.float32,
    )

    tensor_values: dict[str, Tensor] = {
        "token_mask": torch.ones(token_count, dtype=torch.bool),
        "residue_index": torch.tensor([cast(int, token["residue_index"]) for token in tokens]),
        "chain_index": torch.tensor([cast(int, token["chain_index"]) for token in tokens]),
        "entity_index": torch.tensor([cast(int, token["entity_index"]) for token in tokens]),
        "symmetry_index": torch.tensor([cast(int, token["symmetry_index"]) for token in tokens]),
        "token_index": torch.arange(token_count),
        "atom_features": atom_features,
        "atom_to_token": atom_to_token,
        "atom_mask": atom_mask,
        "reference_positions": reference_positions,
        "reference_space_id": reference_space_id,
        "representative_atom_mask": representative_atom_mask,
        "atom_token_index": atom_token_index,
        "token_to_rep_atom": token_to_rep_atom,
        "residue_type": torch.nn.functional.one_hot(residue_types, 33).float(),
        "msa_profile": msa_values.pop("msa_profile"),
        "deletion_mean": msa_values.pop("deletion_mean"),
        "token_bonds": token_bonds,
        "type_bonds": type_bonds,
        "contact_conditioning": torch.nn.functional.one_hot(
            torch.zeros(token_count, token_count, dtype=torch.long),
            5,
        ),
        "contact_threshold": torch.zeros(token_count, token_count),
        "molecule_type": molecule_types,
        "frame_atom_index": frame_atom_index,
        "frame_mask": frame_mask,
        "method_feature": torch.full((token_count,), 4, dtype=torch.long),
        "modified": torch.tensor(
            [cast(bool, token["modified"]) for token in tokens],
            dtype=torch.long,
        ),
        "cyclic_period": cyclic_period,
        "affinity_token_mask": torch.tensor([cast(bool, token["affinity"]) for token in tokens]),
        "affinity_mw": torch.tensor([affinity_mw]),
        "affinity_msa_profile": torch.nn.functional.one_hot(residue_types, 33).float(),
        "affinity_deletion_mean": torch.zeros(token_count),
        **msa_values,
    }
    if structure is not None:
        tensor_values["target_positions"] = target_positions
        tensor_values["target_mask"] = torch.nn.functional.pad(
            resolved,
            (0, padded_atom_count - atom_count),
        )
    placed = {
        name: value.to(device=device, dtype=dtype) if value.is_floating_point() else value.to(device=device)
        for name, value in tensor_values.items()
    }
    return Boltz2Input(
        **{name: value.unsqueeze(0) for name, value in placed.items()},
        layout=Boltz2Layout(
            system=system,
            atom_sites=atom_sites,
            token_sites=tuple(cast(AtomSite | tuple[str, int], token["token_site"]) for token in tokens),
            affinity_instance_id=ligand,
        ),
    )


def _prepare_boltz2_structure_features(
    system: MolecularSystem,
    *,
    structure: Structure | None,
    msa: MSA | MSACollection | None,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None = None,
) -> Boltz2Input:
    r"""
    Build the whole-system folding token, atom, MSA, and identity layout.

    Shares the token/atom/bond/MSA construction with the affinity Processor for the *whole* system:
    no ligand-centered crop and no affinity tensors. Reuses the module-level feature helpers directly.
    """

    available_sites = {site for site, atom in system.iter_atoms() if atom.element != "H"}
    entity_index_by_id: dict[str, int] = {}
    instance_count_by_entity: dict[str, int] = {}
    tokens: list[dict[str, object]] = []

    for chain_index, (instance_id, entity_id) in enumerate(system.instances.items()):
        entity = system.entities[entity_id]
        entity_index = entity_index_by_id.setdefault(entity_id, len(entity_index_by_id))
        symmetry_index = instance_count_by_entity.get(entity_id, 0)
        instance_count_by_entity[entity_id] = symmetry_index + 1
        if isinstance(entity, PolymerSequence):
            molecule_type = _MOLECULE_TYPES[entity.polymer_type]
            standards = _STANDARD_COMPONENTS[entity.polymer_type]
            for monomer_index, component in enumerate(entity.components):
                component_id = component.component_id
                atom_sites = tuple(
                    site
                    for atom in component.atoms
                    if atom.element != "H"
                    and atom.atom_id not in {"OXT", "OP3"}
                    and (site := AtomSite(instance_id, monomer_index, atom.atom_id)) in available_sites
                )
                if not atom_sites:
                    raise ValueError(f"component {component_id!r} has no model-visible heavy atoms")
                is_standard = component_id in standards
                residue_type = (
                    _residue_type(component_id, entity.polymer_type)
                    if is_standard
                    else _UNKNOWN_RESIDUE_TYPES[entity.polymer_type]
                )
                by_name = {site.atom_id: site for site in atom_sites}
                center_name = "CA" if entity.polymer_type == "protein" else "C1'"
                if entity.polymer_type == "protein":
                    distogram_name = "CA" if component_id == "GLY" else "CB"
                    frame_names = ("N", "CA", "C") if is_standard else ("CA", "N", "C")
                else:
                    distogram_name = (
                        "C4"
                        if component_id in {"A", "G", "DA", "DG"}
                        else "C2" if component_id in {"C", "U", "DC", "DT"} else "C1'"
                    )
                    frame_names = ("C1'", "C3'", "C4'")
                center_site = by_name.get(center_name, atom_sites[0])
                distogram_site = by_name.get(distogram_name, center_site)
                tokens.append(
                    {
                        "atom_sites": atom_sites,
                        "token_site": (instance_id, monomer_index),
                        "instance_id": instance_id,
                        "entity_id": entity_id,
                        "monomer_index": monomer_index,
                        "residue_index": monomer_index,
                        "chain_index": chain_index,
                        "entity_index": entity_index,
                        "symmetry_index": symmetry_index,
                        "molecule_type": molecule_type,
                        "residue_type": residue_type,
                        "modified": not is_standard,
                        "center_site": center_site,
                        "distogram_site": distogram_site,
                        "frame_sites": tuple(by_name.get(name, center_site) for name in frame_names),
                    }
                )
        else:
            atom_sites = tuple(
                site
                for component_index, component in enumerate(entity.components)
                for atom in component.atoms
                if atom.element != "H"
                and (site := AtomSite(instance_id, component_index, atom.atom_id)) in available_sites
            )
            if not atom_sites:
                raise ValueError(f"molecule {instance_id!r} has no model-visible heavy atoms")
            for site in atom_sites:
                tokens.append(
                    {
                        "atom_sites": (site,),
                        "token_site": site,
                        "instance_id": instance_id,
                        "entity_id": entity_id,
                        "monomer_index": None,
                        "residue_index": 0,
                        "chain_index": chain_index,
                        "entity_index": entity_index,
                        "symmetry_index": symmetry_index,
                        "molecule_type": 3,
                        "residue_type": 22,
                        "modified": False,
                        "center_site": site,
                        "distogram_site": site,
                        "frame_sites": (site, site, site),
                    }
                )

    atom_sites = tuple(site for token in tokens for site in cast(tuple[AtomSite, ...], token["atom_sites"]))
    site_to_atom = {site: index for index, site in enumerate(atom_sites)}
    site_to_token = {
        site: token_index
        for token_index, token in enumerate(tokens)
        for site in cast(tuple[AtomSite, ...], token["atom_sites"])
    }
    atom_count = len(atom_sites)
    padded_atom_count = max(32, math.ceil(atom_count / 32) * 32)
    reference_coordinates = _reference_coordinates(system, atom_sites)
    reference_space_by_residue: dict[tuple[str, int], int] = {}
    reference_space_ids = [
        reference_space_by_residue.setdefault(
            (site.instance_id, site.component_index),
            len(reference_space_by_residue),
        )
        for site in atom_sites
    ]
    resolved = torch.zeros(atom_count, dtype=torch.bool)
    target_positions = torch.zeros(padded_atom_count, 3)
    if structure is not None:
        coordinate_index = {site: index for index, site in enumerate(structure.sites)}
        for atom_index, site in enumerate(atom_sites):
            source_index = coordinate_index.get(site)
            if source_index is not None and bool(structure.coordinate_mask[source_index]):
                resolved[atom_index] = True
                target_positions[atom_index] = torch.from_numpy(structure.coordinates[source_index].copy())

    reference_positions = torch.zeros(padded_atom_count, 3)
    reference_positions[:atom_count] = torch.tensor(reference_coordinates)
    reference_space_id = torch.zeros(padded_atom_count, dtype=torch.long)
    reference_space_id[:atom_count] = torch.tensor(reference_space_ids)
    _augment_reference_positions(
        reference_positions[:atom_count],
        reference_space_id[:atom_count],
        resolved,
        generator=generator,
    )

    atomic_number = torch.zeros(padded_atom_count, dtype=torch.long)
    reference_charges = torch.zeros(padded_atom_count)
    atom_name_characters = torch.zeros(padded_atom_count, 4, dtype=torch.long)
    atom_mask = torch.zeros(padded_atom_count, dtype=torch.bool)
    atom_to_token = torch.zeros(padded_atom_count, dtype=torch.long)
    atom_token_index = torch.zeros(padded_atom_count, dtype=torch.long)
    atom_names = _model_atom_names(system, atom_sites)
    from rdkit.Chem import GetPeriodicTable

    periodic_table = GetPeriodicTable()
    for atom_index, (site, atom_name) in enumerate(zip(atom_sites, atom_names)):
        atom = system.atom(site)
        atomic_number[atom_index] = periodic_table.GetAtomicNumber(atom.element)
        reference_charges[atom_index] = atom.formal_charge
        atom_name_characters[atom_index] = torch.tensor(
            [ord(character) - 32 if character != " " else 0 for character in atom_name.ljust(4)]
        )
        atom_mask[atom_index] = True
        token_index = site_to_token[site]
        atom_to_token[atom_index] = token_index
        atom_token_index[atom_index] = cast(tuple[AtomSite, ...], tokens[token_index]["atom_sites"]).index(site)

    atom_features = torch.cat(
        (
            reference_positions,
            reference_charges.unsqueeze(-1),
            torch.nn.functional.one_hot(atomic_number, 128).float(),
            torch.nn.functional.one_hot(atom_name_characters, 64).reshape(padded_atom_count, 256).float(),
        ),
        dim=-1,
    )
    token_count = len(tokens)
    token_to_rep_atom = torch.tensor(
        [site_to_atom[cast(AtomSite, token["distogram_site"])] for token in tokens],
        dtype=torch.long,
    )
    representative_atom_mask = torch.zeros(padded_atom_count, dtype=torch.bool)
    representative_atom_mask[token_to_rep_atom] = True
    token_bonds, type_bonds = _token_bond_features(system, site_to_token, token_count)
    msa_values = _msa_features(msa, tokens, system)
    molecule_types = torch.tensor([cast(int, token["molecule_type"]) for token in tokens])
    residue_types = torch.tensor([cast(int, token["residue_type"]) for token in tokens])
    frame_atom_index = torch.tensor(
        [
            [site_to_atom[site] for site in cast(tuple[AtomSite, AtomSite, AtomSite], token["frame_sites"])]
            for token in tokens
        ],
        dtype=torch.long,
    )
    frame_mask = torch.tensor(
        [
            all(resolved[site_to_atom[site]] for site in cast(tuple[AtomSite, ...], token["frame_sites"]))
            for token in tokens
        ]
    )
    cyclic_period = torch.tensor(
        [_cyclic_period(system, cast(str, token["instance_id"])) for token in tokens],
        dtype=torch.float32,
    )

    tensor_values: dict[str, Tensor] = {
        "token_mask": torch.ones(token_count, dtype=torch.bool),
        "residue_index": torch.tensor([cast(int, token["residue_index"]) for token in tokens]),
        "chain_index": torch.tensor([cast(int, token["chain_index"]) for token in tokens]),
        "entity_index": torch.tensor([cast(int, token["entity_index"]) for token in tokens]),
        "symmetry_index": torch.tensor([cast(int, token["symmetry_index"]) for token in tokens]),
        "token_index": torch.arange(token_count),
        "atom_features": atom_features,
        "atom_to_token": atom_to_token,
        "atom_mask": atom_mask,
        "reference_positions": reference_positions,
        "reference_space_id": reference_space_id,
        "representative_atom_mask": representative_atom_mask,
        "atom_token_index": atom_token_index,
        "token_to_rep_atom": token_to_rep_atom,
        "residue_type": torch.nn.functional.one_hot(residue_types, 33).float(),
        "msa_profile": msa_values.pop("msa_profile"),
        "deletion_mean": msa_values.pop("deletion_mean"),
        "token_bonds": token_bonds,
        "type_bonds": type_bonds,
        "contact_conditioning": torch.nn.functional.one_hot(
            torch.zeros(token_count, token_count, dtype=torch.long),
            5,
        ),
        "contact_threshold": torch.zeros(token_count, token_count),
        "molecule_type": molecule_types,
        "frame_atom_index": frame_atom_index,
        "frame_mask": frame_mask,
        # Whole-system folding uses the x-ray-diffraction method id; the affinity
        # pass uses the ``other`` method id.
        "method_feature": torch.full((token_count,), 1, dtype=torch.long),
        "modified": torch.tensor(
            [cast(bool, token["modified"]) for token in tokens],
            dtype=torch.long,
        ),
        "cyclic_period": cyclic_period,
        **msa_values,
    }
    if structure is not None:
        tensor_values["target_positions"] = target_positions
        tensor_values["target_mask"] = torch.nn.functional.pad(
            resolved,
            (0, padded_atom_count - atom_count),
        )
    placed = {
        name: value.to(device=device, dtype=dtype) if value.is_floating_point() else value.to(device=device)
        for name, value in tensor_values.items()
    }
    return Boltz2Input(
        **{name: value.unsqueeze(0) for name, value in placed.items()},
        layout=Boltz2Layout(
            system=system,
            atom_sites=atom_sites,
            token_sites=tuple(cast(AtomSite | tuple[str, int], token["token_site"]) for token in tokens),
            affinity_instance_id=None,
        ),
    )


def _affinity_crop(tokens: list[dict[str, object]], *, structure: Structure | None) -> list[int]:
    if structure is None:
        atom_count = sum(len(cast(tuple[AtomSite, ...], token["atom_sites"])) for token in tokens)
        polymer_count = sum(cast(int, token["molecule_type"]) != 3 for token in tokens)
        if len(tokens) > _MAX_TOKENS or atom_count > _MAX_ATOMS or polymer_count > _MAX_POLYMER_TOKENS:
            raise ValueError("large affinity systems require a Structure for ligand-centered cropping")
        return list(range(len(tokens)))

    coordinate_index = {site: index for index, site in enumerate(structure.sites)}
    valid: list[int] = []
    centers: list[Tensor] = []
    for token_index, token in enumerate(tokens):
        site = cast(AtomSite, token["center_site"])
        atom_index = coordinate_index.get(site)
        if atom_index is not None and bool(structure.coordinate_mask[atom_index]):
            valid.append(token_index)
            centers.append(torch.from_numpy(structure.coordinates[atom_index].copy()))
    ligand_centers = [center for index, center in zip(valid, centers) if cast(bool, tokens[index]["affinity"])]
    if not ligand_centers:
        raise ValueError("the selected ligand has no resolved coordinates for affinity cropping")
    center_tensor = torch.stack(centers)
    ligand_tensor = torch.stack(ligand_centers)
    distances = torch.cdist(center_tensor, ligand_tensor).min(dim=1).values
    order = [valid[index] for index in distances.argsort().tolist()]
    chain_tokens: dict[int, list[int]] = {}
    for index, token in enumerate(tokens):
        chain_tokens.setdefault(cast(int, token["chain_index"]), []).append(index)
    ligand_ids = {index for index in valid if cast(int, tokens[index]["molecule_type"]) == 3}
    cropped: set[int] = set()
    cropped_polymer: set[int] = set()
    total_atoms = 0
    for query_index in order:
        query = tokens[query_index]
        chain = chain_tokens[cast(int, query["chain_index"])]
        if len(chain) <= _CROP_NEIGHBORHOOD:
            neighborhood = chain
        else:
            query_residue = cast(int, query["residue_index"])
            maximum = [
                index
                for index in chain
                if abs(cast(int, tokens[index]["residue_index"]) - query_residue) <= _CROP_NEIGHBORHOOD
            ]
            radius = 0
            neighborhood = [index for index in maximum if cast(int, tokens[index]["residue_index"]) == query_residue]
            while len(neighborhood) < _CROP_NEIGHBORHOOD:
                radius += 1
                neighborhood = [
                    index
                    for index in maximum
                    if abs(cast(int, tokens[index]["residue_index"]) - query_residue) <= radius
                ]
        new = set(neighborhood) - cropped
        new_atoms = sum(len(cast(tuple[AtomSite, ...], tokens[index]["atom_sites"])) for index in new)
        if (
            len(new) > _MAX_TOKENS - len(cropped)
            or total_atoms + new_atoms > _MAX_ATOMS
            or len(cropped_polymer | (new - ligand_ids)) > _MAX_POLYMER_TOKENS
        ):
            break
        cropped.update(new)
        cropped_polymer.update(new - ligand_ids)
        total_atoms += new_atoms
    return sorted(cropped)


def _token_bond_features(
    system: MolecularSystem,
    site_to_token: dict[AtomSite, int],
    token_count: int,
) -> tuple[Tensor, Tensor]:
    token_bonds = torch.zeros(token_count, token_count)
    type_bonds = torch.zeros(token_count, token_count, dtype=torch.long)

    def add(site_a: AtomSite, site_b: AtomSite, bond_type: int) -> None:
        token_a = site_to_token.get(site_a)
        token_b = site_to_token.get(site_b)
        if token_a is None or token_b is None or token_a == token_b:
            return
        token_bonds[token_a, token_b] = token_bonds[token_b, token_a] = 1
        type_bonds[token_a, token_b] = type_bonds[token_b, token_a] = bond_type

    def bond_type_of(order: float, aromatic: bool) -> int:
        return 5 if aromatic else {1.0: 2, 2.0: 3, 3.0: 4}.get(order, 1)

    for instance_id, entity_id in system.instances.items():
        entity = system.entities[entity_id]
        if not isinstance(entity, Molecule):
            continue
        for component_index, component in enumerate(entity.components):
            for component_bond in component.bonds or ():
                add(
                    AtomSite(instance_id, component_index, component_bond.atom_a),
                    AtomSite(instance_id, component_index, component_bond.atom_b),
                    bond_type_of(component_bond.order, component_bond.aromatic),
                )
        for molecule_bond in entity.bonds:
            add(
                AtomSite(instance_id, molecule_bond.component_a, molecule_bond.atom_a),
                AtomSite(instance_id, molecule_bond.component_b, molecule_bond.atom_b),
                bond_type_of(molecule_bond.order, molecule_bond.aromatic),
            )
    for system_bond in system.bonds:
        add(system_bond.site_a, system_bond.site_b, 6)
    return token_bonds, type_bonds


def _msa_features(
    msa: MSA | MSACollection | None,
    tokens: list[dict[str, object]],
    system: MolecularSystem,
) -> dict[str, Tensor]:
    residue_types = [cast(int, token["residue_type"]) for token in tokens]
    if msa is None:
        rows = torch.tensor([residue_types])
        deletion_counts = torch.zeros(1, len(tokens))
        paired = torch.ones(1, len(tokens))
    else:
        # The MSA holds no PolymerSequence: column ``c`` binds ``columns[c].entity_id`` to a system
        # entity, ``query_to_entity`` is now the identity map (column i binds ``entity.components[i]``,
        # so a token's monomer index is directly the aligned-row column), and cells are UPPERCASE
        # one-letter symbols (or ``-``) rather than ChemicalComponent IDs. A bare MSA is a single-chain
        # plan over its unpaired pool; a collection follows its dense ``row_pairing`` and
        # ``row_is_paired`` plan across the paired and unpaired pools.
        bindings = msa_bindings(msa)
        columns = tuple(value for value, _target_instance_id in bindings)
        if isinstance(msa, MSA):
            plan_rows: list[tuple[tuple[int, int, bool], ...]] = [
                ((0, row_index, False),) for row_index in range(len(msa.unpaired_sequences))
            ]
        else:
            plan_rows = [
                tuple(
                    (column_index, int(msa.row_pairing[plan_index, column_index]), bool(is_paired))
                    for column_index, is_paired in enumerate(msa.row_is_paired[plan_index])
                    if msa.row_pairing[plan_index, column_index] != -1
                )
                for plan_index in range(msa.row_pairing.shape[0])
            ]
        polymer_types = tuple(cast(PolymerSequence, system.entities[value.entity_id]).polymer_type for value in columns)
        row_values: list[Tensor] = []
        deletion_values: list[Tensor] = []
        paired_values: list[Tensor] = []
        for selections in plan_rows:
            row = torch.full((len(tokens),), _MSA_GAP, dtype=torch.long)
            deletion_row = torch.zeros(len(tokens))
            paired_row = torch.zeros(len(tokens))
            present = len(selections)
            # A taxonomy-paired plan row has more than one populated column and draws every
            # populated cell from the paired pool.
            taxonomic_pair = (
                isinstance(msa, MSACollection) and present > 1 and all(is_paired for _, _, is_paired in selections)
            )
            for column_index, source_row, is_paired in selections:
                value = columns[column_index]
                target_instance_id = bindings[column_index][1]
                polymer_type = polymer_types[column_index]
                aligned = value.paired_sequences[source_row] if is_paired else value.unpaired_sequences[source_row]
                deletion_source = (value.paired_deletion_counts if is_paired else value.unpaired_deletion_counts)[
                    source_row
                ]
                for token_index, token in enumerate(tokens):
                    if (
                        token["entity_id"] != value.entity_id
                        or (target_instance_id is not None and token["instance_id"] != target_instance_id)
                        or token["monomer_index"] is None
                    ):
                        continue
                    monomer_index = cast(int, token["monomer_index"])
                    row[token_index] = _msa_symbol_restype(aligned[monomer_index], polymer_type)
                    deletion_row[token_index] = float(deletion_source[monomer_index])
                    paired_row[token_index] = float(taxonomic_pair)
            row_values.append(row)
            deletion_values.append(deletion_row)
            paired_values.append(paired_row)
        rows = torch.stack(row_values)
        deletion_counts = torch.stack(deletion_values)
        paired = torch.stack(paired_values)
        rows[0] = torch.tensor(residue_types)
        paired[0] = 1
    has_deletion = deletion_counts > 0
    deletion_value = (math.pi / 2) * torch.atan(deletion_counts / 3)
    return {
        "msa": rows,
        "has_deletion": has_deletion,
        "deletion_value": deletion_value,
        "msa_paired": paired,
        "msa_mask": torch.ones_like(rows, dtype=torch.bool),
        "msa_profile": torch.nn.functional.one_hot(rows, 33).float().mean(dim=0),
        "deletion_mean": deletion_value.mean(dim=0),
    }


def _reference_coordinates(system: MolecularSystem, sites: tuple[AtomSite, ...]) -> list[tuple[float, float, float]]:
    generated: dict[str, dict[tuple[int, str], tuple[float, float, float]]] = {}
    coordinates: list[tuple[float, float, float]] = []
    for site in sites:
        entity_id = system.instances[site.instance_id]
        entity = system.entities[entity_id]
        component = entity.components[site.component_index]
        atom_index = next(index for index, atom in enumerate(component.atoms) if atom.atom_id == site.atom_id)
        coordinate = component.reference_coordinates[atom_index] if component.reference_coordinates else None
        if coordinate is None and isinstance(entity, Molecule):
            if entity_id not in generated:
                generated[entity_id] = _molecule_reference_coordinates(entity)
            coordinate = generated[entity_id][(site.component_index, site.atom_id)]
        if coordinate is None:
            raise ValueError(f"Boltz-2 requires reference coordinates for {site!r}")
        coordinates.append(coordinate)
    return coordinates


def _molecule_reference_coordinates(
    molecule: Molecule,
) -> dict[tuple[int, str], tuple[float, float, float]]:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    rdkit_molecule = Chem.AddHs(_canonical_rdkit_molecule(molecule))
    options = AllChem.ETKDGv3()
    options.clearConfs = False
    options.randomSeed = 42
    conformer_id = AllChem.EmbedMolecule(rdkit_molecule, options)
    if conformer_id == -1:
        options.useRandomCoords = True
        conformer_id = AllChem.EmbedMolecule(rdkit_molecule, options)
    AllChem.UFFOptimizeMolecule(rdkit_molecule, confId=conformer_id, maxIters=1000)
    conformer = rdkit_molecule.GetConformer(conformer_id)
    coordinates: dict[tuple[int, str], tuple[float, float, float]] = {}
    flat_index = 0
    for component_index, component in enumerate(molecule.components):
        for atom in component.atoms:
            position = conformer.GetAtomPosition(flat_index)
            coordinates[(component_index, atom.atom_id)] = (position.x, position.y, position.z)
            flat_index += 1
    return coordinates


def _augment_reference_positions(
    positions: Tensor,
    space_ids: Tensor,
    resolved: Tensor,
    *,
    generator: torch.Generator | None = None,
) -> None:
    r"""
    Randomly rotate and translate each resolved reference conformer.

    Random values are drawn from ``generator``; ``None`` uses the ambient PyTorch
    RNG.
    """

    if not len(space_ids):
        return
    for space_id in range(int(space_ids.max().item())):
        included = space_ids == space_id
        mask = resolved[included]
        if not bool(mask.any()):
            continue
        values = positions[included]
        center = (values * mask[:, None]).sum(dim=0, keepdim=True) / mask.sum()
        quaternion = torch.randn(
            4,
            dtype=values.dtype,
            device=values.device,
            generator=generator,
        )
        norm = torch.sqrt((quaternion * quaternion).sum())
        quaternion = quaternion / torch.copysign(norm, quaternion[0])
        real, i, j, k = quaternion
        two = 2.0 / (quaternion * quaternion).sum()
        rotation = torch.stack(
            (
                1 - two * (j * j + k * k),
                two * (i * j - k * real),
                two * (i * k + j * real),
                two * (i * j + k * real),
                1 - two * (i * i + k * k),
                two * (j * k - i * real),
                two * (i * k - j * real),
                two * (j * k + i * real),
                1 - two * (i * i + j * j),
            )
        ).reshape(3, 3)
        positions[included] = (values - center) @ rotation + torch.randn(
            1,
            3,
            dtype=values.dtype,
            device=values.device,
            generator=generator,
        )


def _model_atom_names(system: MolecularSystem, sites: tuple[AtomSite, ...]) -> list[str]:
    names: list[str] = []
    counts: dict[tuple[str, str], int] = {}
    for site in sites:
        name = site.atom_id
        if len(name) > 4 or any(not 32 <= ord(character) < 96 for character in name):
            element = system.atom(site).element.upper()
            key = (site.instance_id, element)
            counts[key] = counts.get(key, 0) + 1
            name = f"{element}{counts[key]}"
        if len(name) > 4:
            raise ValueError(f"Boltz-2 atom name cannot be encoded in four characters: {name!r}")
        names.append(name)
    return names


def _affinity_ligand_properties(molecule: Molecule) -> tuple[int, float]:
    r"""
    Return graph atom count and molecular weight.
    """

    from rdkit.Chem import Descriptors

    value = _canonical_rdkit_molecule(molecule)
    return value.GetNumAtoms(), float(Descriptors.MolWt(value))


def _canonical_rdkit_molecule(molecule: Molecule) -> Mol:
    r"""
    Materialize molecule data as an RDKit graph.
    """

    from rdkit import Chem

    editable = Chem.RWMol()
    atom_index: dict[tuple[int, str], int] = {}
    for component_index, component in enumerate(molecule.components):
        for atom in component.atoms:
            rdkit_atom = Chem.Atom(atom.element)
            if atom.isotope is not None:
                rdkit_atom.SetIsotope(atom.isotope)
            rdkit_atom.SetFormalCharge(atom.formal_charge)
            rdkit_atom.SetNumRadicalElectrons(atom.radical_electrons)
            rdkit_atom.SetNumExplicitHs(atom.explicit_hydrogens)
            rdkit_atom.SetNoImplicit(not atom.implicit_hydrogens_allowed)
            rdkit_atom.SetIsAromatic(atom.aromatic)
            atom_index[(component_index, atom.atom_id)] = editable.AddAtom(rdkit_atom)

    bond_types = {
        1.0: Chem.BondType.SINGLE,
        1.5: Chem.BondType.AROMATIC,
        2.0: Chem.BondType.DOUBLE,
        3.0: Chem.BondType.TRIPLE,
        4.0: Chem.BondType.QUADRUPLE,
    }
    for component_index, component in enumerate(molecule.components):
        for component_bond in component.bonds or ():
            editable.AddBond(
                atom_index[(component_index, component_bond.atom_a)],
                atom_index[(component_index, component_bond.atom_b)],
                bond_types[component_bond.order],
            )
            editable.GetBondWithIdx(editable.GetNumBonds() - 1).SetIsAromatic(component_bond.aromatic)
    for molecule_bond in molecule.bonds:
        editable.AddBond(
            atom_index[(molecule_bond.component_a, molecule_bond.atom_a)],
            atom_index[(molecule_bond.component_b, molecule_bond.atom_b)],
            bond_types[molecule_bond.order],
        )
        editable.GetBondWithIdx(editable.GetNumBonds() - 1).SetIsAromatic(molecule_bond.aromatic)

    value = editable.GetMol()
    Chem.SanitizeMol(value)
    return value


def _cyclic_period(system: MolecularSystem, instance_id: str) -> int:
    entity = system.entities[system.instances[instance_id]]
    if not isinstance(entity, PolymerSequence) or len(entity) <= 2:
        return 0
    for bond in system.bonds:
        if bond.site_a.instance_id != instance_id or bond.site_b.instance_id != instance_id:
            continue
        if {bond.site_a.component_index, bond.site_b.component_index} == {0, len(entity) - 1}:
            return len(entity)
    return 0


def _residue_type(component_id: str, polymer_type: str) -> int:
    if polymer_type == "protein":
        return _PROTEIN_RESIDUE_TYPES.get(component_id, 22)
    if polymer_type == "rna":
        return _RNA_RESIDUE_TYPES.get(component_id, 27)
    return _DNA_RESIDUE_TYPES.get(component_id, 32)


def _normalize_msa(
    msa: MSA | MSACollection | str | PathLike[str] | None,
    system: MolecularSystem,
) -> MSA | MSACollection | None:
    if msa is None:
        return None
    if isinstance(msa, MSA | MSACollection):
        resolve_msa_bindings(msa, system)
        return msa
    path = Path(msa).expanduser()
    proteins = [
        entity_id
        for entity_id, entity in system.entities.items()
        if isinstance(entity, PolymerSequence) and entity.polymer_type == "protein"
    ]
    if len(proteins) != 1:
        raise ValueError("an aligned FASTA is unambiguous only for one protein entity; pass MSACollection")
    records = read_fasta_records(path)
    if not records:
        raise ValueError("MSA FASTA contains no rows")
    # Project each row onto non-gap query columns, normalize ``.`` to ``-``, and use zero deletion
    # counts. Row 0 is the ungapped query.
    query = records[0].sequence.upper()
    keep = [index for index, symbol in enumerate(query) if symbol not in "-."]
    width = len(keep)
    projected: list[str] = []
    for record in records:
        sequence = record.sequence.upper()
        projected.append("".join("-" if sequence[index] in "-." else sequence[index] for index in keep))
    return MSA(
        entity_id=proteins[0],
        unpaired_sequences=tuple(projected),
        unpaired_deletion_counts=np.zeros((len(projected), width), dtype=np.uint32),
        paired_sequences=(projected[0],),
        paired_deletion_counts=np.zeros((1, width), dtype=np.uint32),
    )
