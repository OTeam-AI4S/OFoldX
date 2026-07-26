r"""
Input processing for Boltz-2 folding and affinity scoring.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor

from ...data.evidence import MSA, MSACollection, Template, msa_bindings, resolve_msa_bindings
from ...data.molecular import (
    AtomSite,
    ChemicalComponent,
    ComponentReference,
    MolecularSystem,
    Molecule,
    PolymerSequence,
)
from ...data.structure import Structure
from ...processor import Processor
from .model import (
    Boltz2Contact,
    Boltz2Input,
    Boltz2Layout,
    Boltz2Pocket,
    Boltz2StructureTarget,
    Boltz2Template,
)

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
_CONSTRAINT_FREE_COMPONENTS = {
    "protein": frozenset((*_PROTEIN_RESIDUE_TYPES, "UNK", "MSE")),
    "rna": frozenset(_RNA_RESIDUE_TYPES),
    "dna": frozenset(_DNA_RESIDUE_TYPES),
}
_MOLECULE_TYPES = {"protein": 0, "dna": 1, "rna": 2}
_UNKNOWN_RESIDUE_TYPES = {"protein": 22, "rna": 27, "dna": 32}
_MSA_GAP = 1
_METHOD_TYPES = {
    "md": 0,
    "x-ray diffraction": 1,
    "electron microscopy": 2,
    "solution nmr": 3,
    "solid-state nmr": 4,
    "neutron diffraction": 4,
    "electron crystallography": 4,
    "fiber diffraction": 4,
    "powder diffraction": 4,
    "infrared spectroscopy": 4,
    "fluorescence transfer": 4,
    "epr": 4,
    "theoretical model": 4,
    "solution scattering": 4,
    "other": 4,
    "afdb": 5,
    "boltz-1": 6,
    "future1": 7,
    "future2": 8,
    "future3": 9,
    "future4": 10,
    "future5": 11,
}
_STANDARD_LOCAL_SYMMETRY = {
    "ASP": (("OD1", "OD2"), ("OD2", "OD1")),
    "GLU": (("OE1", "OE2"), ("OE2", "OE1")),
    "PHE": (
        ("CD1", "CD2", "CE1", "CE2"),
        ("CD2", "CD1", "CE2", "CE1"),
    ),
    "TYR": (
        ("CD1", "CD2", "CE1", "CE2"),
        ("CD2", "CD1", "CE2", "CE1"),
    ),
    "A": (("OP1", "OP2"), ("OP2", "OP1")),
    "G": (("OP1", "OP2"), ("OP2", "OP1")),
    "C": (("OP1", "OP2"), ("OP2", "OP1")),
    "U": (("OP1", "OP2"), ("OP2", "OP1")),
    "DA": (("OP1", "OP2"), ("OP2", "OP1")),
    "DG": (("OP1", "OP2"), ("OP2", "OP1")),
    "DC": (("OP1", "OP2"), ("OP2", "OP1")),
    "DT": (("OP1", "OP2"), ("OP2", "OP1")),
}
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
_CONTACT_UNSPECIFIED = 0
_CONTACT_UNSELECTED = 1
_CONTACT_POCKET_TO_BINDER = 2
_CONTACT_BINDER_TO_POCKET = 3
_CONTACT_PAIR = 4


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
        input: MolecularSystem | Structure,
        *,
        ligand: str | None = None,
        msa: MSA | MSACollection | None = None,
        contacts: Sequence[Boltz2Contact] = (),
        pockets: Sequence[Boltz2Pocket] = (),
        method: str = "other",
        generator: torch.Generator | None = None,
    ) -> Boltz2Input:
        r"""
        Prepare affinity features for one protein-ligand system.

        Args:
            input: Molecular system or structure.
            ligand: Molecule instance scored by the affinity head. May be omitted for a system
                containing exactly one molecule instance.
            msa: Optional MSA value or collection.
            contacts: Learned pairwise contact conditions.
            pockets: Learned binder-to-pocket conditions.
            method: Experimental method used by the learned method embedding.
            generator: Optional generator for independent rigid transforms of reference
                conformers.

        Returns:
            Ligand-centered token, atom, MSA, and affinity tensors. Reference
            coordinates are in angstroms; tensor axes align to ``layout``.
        """

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

        if msa is not None:
            resolve_msa_bindings(msa, system)
        return _prepare_boltz2_affinity_features(
            system,
            structure=structure,
            ligand=ligand,
            affinity_mw=affinity_mw,
            msa=msa,
            contacts=contacts,
            pockets=pockets,
            method=method,
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
        input: MolecularSystem | Structure,
        *,
        msa: MSA | MSACollection | None = None,
        templates: Sequence[Template | Boltz2Template] = (),
        contacts: Sequence[Boltz2Contact] = (),
        pockets: Sequence[Boltz2Pocket] = (),
        method: str = "x-ray diffraction",
        generator: torch.Generator | None = None,
    ) -> Boltz2Input:
        r"""
        Prepare folding features for one molecular system.

        Args:
            input: Molecular system or structure.
            msa: Optional MSA value or collection.
            templates: Structural protein templates with canonical residue alignments.
            contacts: Learned pairwise contact conditions.
            pockets: Learned binder-to-pocket conditions.
            method: Experimental method used by the learned method embedding.
            generator: Optional generator for independent rigid transforms of reference
                conformers.

        Returns:
            Batched token, atom, and MSA tensors. Reference coordinates are in
            angstroms; tensor axes align to ``layout``.
        """

        if isinstance(input, Structure):
            structure = input
            system = input.system
        else:
            structure = None
            system = input
        if not system.instances:
            raise ValueError("Boltz-2 requires at least one molecular instance")

        if msa is not None:
            resolve_msa_bindings(msa, system)
        return _prepare_boltz2_structure_features(
            system,
            structure=structure,
            msa=msa,
            templates=templates,
            contacts=contacts,
            pockets=pockets,
            method=method,
            device=self.device,
            dtype=self.dtype,
            generator=generator,
        )

    def process_training(
        self,
        input: Structure,
        *,
        msa: MSA | MSACollection | None = None,
        templates: Sequence[Template | Boltz2Template] = (),
        contacts: Sequence[Boltz2Contact] = (),
        pockets: Sequence[Boltz2Pocket] = (),
        method: str = "x-ray diffraction",
        generator: torch.Generator | None = None,
    ) -> tuple[Boltz2Input, Boltz2StructureTarget]:
        r"""
        Lower one complete structure into model input and separate truth.
        """

        processed = self(
            input,
            msa=msa,
            templates=templates,
            contacts=contacts,
            pockets=pockets,
            method=method,
            generator=generator,
        )
        return processed, _prepare_boltz2_target(
            input,
            cast(Boltz2Layout, processed.layout),
            padded_atom_count=processed.atom_mask.shape[-1],
            device=self.device,
            dtype=self.dtype,
        )


def _prepare_boltz2_affinity_features(
    system: MolecularSystem,
    *,
    structure: Structure | None,
    ligand: str,
    affinity_mw: float,
    msa: MSA | MSACollection | None,
    contacts: Sequence[Boltz2Contact],
    pockets: Sequence[Boltz2Pocket],
    method: str,
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
                    frame_names = ("N", "CA", "C")
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
    if structure is not None:
        coordinate_index = {site: index for index, site in enumerate(structure.sites)}
        for atom_index, site in enumerate(atom_sites):
            source_index = coordinate_index.get(site)
            if source_index is not None and bool(structure.coordinate_mask[source_index]):
                resolved[atom_index] = True

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
    token_bonds, type_bonds = _token_bond_features(system, site_to_token, token_count)
    physical_potential_values = _physical_potential_features(
        system,
        site_to_atom,
    )
    contact_values = _contact_features(
        system,
        tokens,
        contacts=contacts,
        pockets=pockets,
    )
    contact_conditioning = contact_values.pop("contact_conditioning")
    contact_threshold = contact_values.pop("contact_threshold")
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
        "token_to_rep_atom": token_to_rep_atom,
        "residue_type_features": torch.nn.functional.one_hot(residue_types, 33).float(),
        "msa_profile": msa_values.pop("msa_profile"),
        "deletion_mean": msa_values.pop("deletion_mean"),
        "token_bonds": token_bonds,
        "type_bonds": type_bonds,
        "contact_conditioning": contact_conditioning,
        "contact_threshold": contact_threshold,
        **contact_values,
        "molecule_type": molecule_types,
        **physical_potential_values,
        "frame_atom_index": frame_atom_index,
        "method_feature": torch.full((token_count,), _METHOD_TYPES[method.lower()], dtype=torch.long),
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
    templates: Sequence[Template | Boltz2Template],
    contacts: Sequence[Boltz2Contact],
    pockets: Sequence[Boltz2Pocket],
    method: str,
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
                    frame_names = ("N", "CA", "C")
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
    if structure is not None:
        coordinate_index = {site: index for index, site in enumerate(structure.sites)}
        for atom_index, site in enumerate(atom_sites):
            source_index = coordinate_index.get(site)
            if source_index is not None and bool(structure.coordinate_mask[source_index]):
                resolved[atom_index] = True

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
    token_bonds, type_bonds = _token_bond_features(system, site_to_token, token_count)
    physical_potential_values = _physical_potential_features(
        system,
        site_to_atom,
    )
    contact_values = _contact_features(
        system,
        tokens,
        contacts=contacts,
        pockets=pockets,
    )
    contact_conditioning = contact_values.pop("contact_conditioning")
    contact_threshold = contact_values.pop("contact_threshold")
    template_values = _template_features(system, tokens, templates)
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
        "token_to_rep_atom": token_to_rep_atom,
        "residue_type_features": torch.nn.functional.one_hot(residue_types, 33).float(),
        "msa_profile": msa_values.pop("msa_profile"),
        "deletion_mean": msa_values.pop("deletion_mean"),
        "token_bonds": token_bonds,
        "type_bonds": type_bonds,
        "contact_conditioning": contact_conditioning,
        "contact_threshold": contact_threshold,
        **contact_values,
        "molecule_type": molecule_types,
        **physical_potential_values,
        "frame_atom_index": frame_atom_index,
        "method_feature": torch.full((token_count,), _METHOD_TYPES[method.lower()], dtype=torch.long),
        "modified": torch.tensor(
            [cast(bool, token["modified"]) for token in tokens],
            dtype=torch.long,
        ),
        "cyclic_period": cyclic_period,
        **msa_values,
        **template_values,
    }
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


def _prepare_boltz2_target(
    structure: Structure,
    layout: Boltz2Layout,
    *,
    padded_atom_count: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Boltz2StructureTarget:
    r"""
    Lower complete pre-crop structure truth and its discrete symmetry candidates.
    """

    atom_sites = layout.atom_sites
    coordinate_index = {site: index for index, site in enumerate(structure.sites)}
    source_coordinates = torch.from_numpy(structure.coordinates)
    target_positions = torch.zeros(
        len(atom_sites),
        3,
        dtype=source_coordinates.dtype,
    )
    target_mask = torch.zeros(len(atom_sites), dtype=torch.bool)
    for atom_index, site in enumerate(atom_sites):
        source_index = coordinate_index.get(site)
        if source_index is not None and bool(structure.coordinate_mask[source_index]):
            target_positions[atom_index] = source_coordinates[source_index]
            target_mask[atom_index] = True

    target_atom_index = torch.full(
        (padded_atom_count,),
        -1,
        dtype=torch.long,
    )
    target_atom_index[: len(atom_sites)] = torch.arange(len(atom_sites))

    chain_span = []
    chain_equivalence = []
    equivalence_by_layout: dict[tuple[str, tuple[tuple[int, str], ...]], int] = {}
    chain_start = 0
    for instance_id, entity_id in structure.system.instances.items():
        instance_sites = tuple(site for site in atom_sites if site.instance_id == instance_id)
        chain_end = chain_start + len(instance_sites)
        chain_span.append((chain_start, chain_end))
        layout_key = (
            entity_id,
            tuple((site.component_index, site.atom_id) for site in instance_sites),
        )
        chain_equivalence.append(
            equivalence_by_layout.setdefault(
                layout_key,
                len(equivalence_by_layout),
            )
        )
        chain_start = chain_end

    symmetry_atom_index, symmetry_permutation = _truth_symmetry_features(
        structure.system,
        atom_sites,
    )
    return Boltz2StructureTarget(
        coordinates=target_positions.to(device=device, dtype=dtype)[None, None],
        coordinate_mask=target_mask.to(device=device).unsqueeze(0),
        atom_index=target_atom_index.to(device=device).unsqueeze(0),
        chain_span=torch.tensor(
            chain_span,
            dtype=torch.long,
            device=device,
        ).unsqueeze(0),
        chain_equivalence_index=torch.tensor(
            chain_equivalence,
            dtype=torch.long,
            device=device,
        ).unsqueeze(0),
        symmetry_atom_index=symmetry_atom_index.to(device=device).unsqueeze(0),
        symmetry_permutation=symmetry_permutation.to(device=device).unsqueeze(0),
    )


def _truth_symmetry_features(
    system: MolecularSystem,
    atom_sites: tuple[AtomSite, ...],
) -> tuple[Tensor, Tensor]:
    r"""
    Build complete local permutations in stable model-occurrence order.
    """

    site_to_atom = {site: index for index, site in enumerate(atom_sites)}
    groups: list[tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]] = []
    for instance_id, entity_id in system.instances.items():
        entity = system.entities[entity_id]
        for component_index, component in enumerate(entity.components):
            standard = _STANDARD_LOCAL_SYMMETRY.get(component.component_id)
            if (
                isinstance(entity, PolymerSequence)
                and component.component_id in _STANDARD_COMPONENTS[entity.polymer_type]
            ):
                if standard is not None:
                    permutations = tuple(
                        tuple(site_to_atom[AtomSite(instance_id, component_index, atom_id)] for atom_id in names)
                        for names in standard
                        if all(AtomSite(instance_id, component_index, atom_id) in site_to_atom for atom_id in names)
                    )
                    if len(permutations) > 1:
                        groups.append((permutations[0], permutations))
                continue

            rdkit_group = _rdkit_symmetry_group(
                system,
                instance_id,
                component_index,
                component,
                site_to_atom,
            )
            if rdkit_group is not None:
                groups.append(rdkit_group)

    if not groups:
        return (
            torch.empty(0, 0, dtype=torch.long),
            torch.empty(0, 0, 0, dtype=torch.long),
        )

    max_width = max(len(group) for group, _permutations in groups)
    max_permutations = max(len(permutations) for _group, permutations in groups)
    atom_index = torch.full(
        (len(groups), max_width),
        -1,
        dtype=torch.long,
    )
    permutation = torch.full(
        (len(groups), max_permutations, max_width),
        -1,
        dtype=torch.long,
    )
    for group_index, (atom_group, permutations) in enumerate(groups):
        atom_index[group_index, : len(atom_group)] = torch.tensor(atom_group)
        for permutation_index, value in enumerate(permutations):
            permutation[group_index, permutation_index, : len(value)] = torch.tensor(value)
    return atom_index, permutation


def _rdkit_symmetry_group(
    system: MolecularSystem,
    instance_id: str,
    component_index: int,
    component: ChemicalComponent,
    site_to_atom: dict[AtomSite, int],
) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...]] | None:
    from rdkit import Chem

    atoms = component.atoms
    visible_local_index = [
        local_index
        for local_index, atom in enumerate(atoms)
        if AtomSite(instance_id, component_index, atom.atom_id) in site_to_atom
    ]
    if len(visible_local_index) < 2:
        return None

    molecule = _canonical_rdkit_molecule(Molecule((component,)))
    boundary_atom_ids = _component_boundary_atom_ids(
        system,
        instance_id,
        component_index,
    )
    for boundary_index, atom_id in enumerate(sorted(boundary_atom_ids), start=1):
        local_index = next(index for index, atom in enumerate(atoms) if atom.atom_id == atom_id)
        molecule.GetAtomWithIdx(local_index).SetIsotope(1000 + boundary_index)
    Chem.AssignStereochemistry(molecule, force=True, cleanIt=True)

    identity = tuple(
        site_to_atom[AtomSite(instance_id, component_index, atoms[local_index].atom_id)]
        for local_index in visible_local_index
    )
    permutations = {identity}
    for match in molecule.GetSubstructMatches(
        molecule,
        uniquify=False,
        useChirality=True,
        maxMatches=1000,
    ):
        mapped_local_index = tuple(match[index] for index in visible_local_index)
        if set(mapped_local_index) != set(visible_local_index):
            continue
        permutations.add(
            tuple(
                site_to_atom[AtomSite(instance_id, component_index, atoms[index].atom_id)]
                for index in mapped_local_index
            )
        )
    if len(permutations) == 1:
        return None
    ordered = (identity, *sorted(permutations - {identity}))
    return identity, ordered


def _component_boundary_atom_ids(
    system: MolecularSystem,
    instance_id: str,
    component_index: int,
) -> set[str]:
    boundary = set()
    for bond in system.iter_bonds():
        for site, other in (
            (bond.site_a, bond.site_b),
            (bond.site_b, bond.site_a),
        ):
            if (
                site.instance_id == instance_id
                and site.component_index == component_index
                and (other.instance_id != instance_id or other.component_index != component_index)
            ):
                boundary.add(site.atom_id)
    return boundary


def _physical_potential_features(
    system: MolecularSystem,
    site_to_atom: dict[AtomSite, int],
) -> dict[str, Tensor]:
    r"""
    Lower molecular geometry, assembly connections, and entity copies.
    """

    geometry_pairs: list[tuple[int, int]] = []
    geometry_bond_mask: list[bool] = []
    geometry_angle_mask: list[bool] = []
    geometry_upper_bounds: list[float] = []
    geometry_lower_bounds: list[float] = []
    chiral_atoms: list[tuple[int, int, int, int]] = []
    chiral_is_r: list[bool] = []
    stereo_bonds: list[tuple[int, int, int, int]] = []
    stereo_is_e: list[bool] = []
    planar_bonds: list[tuple[int, int, int, int, int, int]] = []
    constraint_scopes: list[tuple[Mol, dict[int, int]]] = []

    for instance_id, entity_id in system.instances.items():
        entity = system.entities[entity_id]
        if isinstance(entity, PolymerSequence):
            for component_index, component in enumerate(entity.components):
                if component.component_id in _CONSTRAINT_FREE_COMPONENTS[entity.polymer_type]:
                    continue
                rdkit_molecule = _canonical_rdkit_molecule(Molecule((component,)))
                component_atom_index = {
                    local_index: site_to_atom[site]
                    for local_index, atom in enumerate(component.atoms)
                    if (site := AtomSite(instance_id, component_index, atom.atom_id)) in site_to_atom
                }
                constraint_scopes.append((rdkit_molecule, component_atom_index))
        else:
            rdkit_molecule = _canonical_rdkit_molecule(entity)
            molecule_atom_index: dict[int, int] = {}
            local_index = 0
            for component_index, component in enumerate(entity.components):
                for atom in component.atoms:
                    site = AtomSite(instance_id, component_index, atom.atom_id)
                    if site in site_to_atom:
                        molecule_atom_index[local_index] = site_to_atom[site]
                    local_index += 1
            constraint_scopes.append((rdkit_molecule, molecule_atom_index))

    for rdkit_molecule, atom_index in constraint_scopes:
        (
            scope_geometry_pairs,
            scope_bond_mask,
            scope_angle_mask,
            scope_upper_bounds,
            scope_lower_bounds,
            scope_chiral_atoms,
            scope_chiral_is_r,
            scope_stereo_bonds,
            scope_stereo_is_e,
            scope_planar_bonds,
        ) = _rdkit_constraint_features(rdkit_molecule, atom_index)
        geometry_pairs.extend(scope_geometry_pairs)
        geometry_bond_mask.extend(scope_bond_mask)
        geometry_angle_mask.extend(scope_angle_mask)
        geometry_upper_bounds.extend(scope_upper_bounds)
        geometry_lower_bounds.extend(scope_lower_bounds)
        chiral_atoms.extend(scope_chiral_atoms)
        chiral_is_r.extend(scope_chiral_is_r)
        stereo_bonds.extend(scope_stereo_bonds)
        stereo_is_e.extend(scope_stereo_is_e)
        planar_bonds.extend(scope_planar_bonds)

    chain_index = {instance_id: index for index, instance_id in enumerate(system.instances)}
    connections = [
        bond
        for bond in system.bonds
        if bond.site_a in site_to_atom
        and bond.site_b in site_to_atom
        and bond.site_a.instance_id != bond.site_b.instance_id
    ]
    connections.sort(
        key=lambda bond: (
            min(site_to_atom[bond.site_a], site_to_atom[bond.site_b]),
            max(site_to_atom[bond.site_a], site_to_atom[bond.site_b]),
            bond.order,
            bond.aromatic,
            bond.stereo or "",
            bond.origin,
            repr(bond.stereo_sites),
        )
    )
    connected_chains = [
        (chain_index[bond.site_a.instance_id], chain_index[bond.site_b.instance_id]) for bond in connections
    ]
    connected_atoms = [(site_to_atom[bond.site_a], site_to_atom[bond.site_b]) for bond in connections]
    instances = tuple(system.instances.items())
    symmetric_chains = [
        (first_index, second_index)
        for first_index, (_first_instance, first_entity) in enumerate(instances)
        for second_index, (_second_instance, second_entity) in enumerate(instances)
        if first_index < second_index and first_entity == second_entity
    ]

    return {
        "geometry_pair_index": _index_columns(geometry_pairs, 2),
        "geometry_bond_mask": torch.tensor(geometry_bond_mask, dtype=torch.bool),
        "geometry_angle_mask": torch.tensor(geometry_angle_mask, dtype=torch.bool),
        "geometry_upper_bounds": torch.tensor(geometry_upper_bounds, dtype=torch.float32),
        "geometry_lower_bounds": torch.tensor(geometry_lower_bounds, dtype=torch.float32),
        "chiral_atom_index": _index_columns(chiral_atoms, 4),
        "chiral_atom_is_r": torch.tensor(chiral_is_r, dtype=torch.bool),
        "stereo_bond_index": _index_columns(stereo_bonds, 4),
        "stereo_bond_is_e": torch.tensor(stereo_is_e, dtype=torch.bool),
        "planar_bond_index": _index_columns(planar_bonds, 6),
        "connected_chain_index": _index_columns(connected_chains, 2),
        "connected_atom_index": _index_columns(connected_atoms, 2),
        "symmetric_chain_index": _index_columns(symmetric_chains, 2),
    }


def _index_columns(values: Sequence[tuple[int, ...]], width: int) -> Tensor:
    if not values:
        return torch.empty(width, 0, dtype=torch.long)
    return torch.tensor(values, dtype=torch.long).T.contiguous()


def _rdkit_constraint_features(
    molecule: Mol,
    atom_index: dict[int, int],
) -> tuple[
    list[tuple[int, int]],
    list[bool],
    list[bool],
    list[float],
    list[float],
    list[tuple[int, int, int, int]],
    list[bool],
    list[tuple[int, int, int, int]],
    list[bool],
    list[tuple[int, int, int, int, int, int]],
]:
    from rdkit import Chem
    from rdkit.Chem import HybridizationType
    from rdkit.Chem.rdchem import BondStereo
    from rdkit.Chem.rdDistGeom import GetMoleculeBoundsMatrix

    molecule = Chem.AddHs(molecule)
    molecule.UpdatePropertyCache(strict=False)
    Chem.GetSymmSSSR(molecule)
    Chem.AssignStereochemistry(molecule, force=True, cleanIt=True)

    bounds = GetMoleculeBoundsMatrix(
        molecule,
        set15bounds=True,
        scaleVDW=True,
        doTriangleSmoothing=True,
        useMacrocycle14config=False,
    )
    bond_pattern = Chem.MolFromSmarts("*~*")
    angle_pattern = Chem.MolFromSmarts("*~*~*")
    bond_pairs = {tuple(sorted(match)) for match in molecule.GetSubstructMatches(bond_pattern)}
    angle_pairs = {tuple(sorted((match[0], match[2]))) for match in molecule.GetSubstructMatches(angle_pattern)}

    geometry_pairs: list[tuple[int, int]] = []
    geometry_bond_mask: list[bool] = []
    geometry_angle_mask: list[bool] = []
    geometry_upper_bounds: list[float] = []
    geometry_lower_bounds: list[float] = []
    for first in range(molecule.GetNumAtoms()):
        for second in range(first + 1, molecule.GetNumAtoms()):
            if first not in atom_index or second not in atom_index:
                continue
            geometry_pairs.append((atom_index[first], atom_index[second]))
            geometry_bond_mask.append((first, second) in bond_pairs)
            geometry_angle_mask.append((first, second) in angle_pairs)
            geometry_upper_bounds.append(float(bounds[first, second]))
            geometry_lower_bounds.append(float(bounds[second, first]))

    chiral_atoms: list[tuple[int, int, int, int]] = []
    chiral_is_r: list[bool] = []
    if all(atom.HasProp("_CIPRank") for atom in molecule.GetAtoms()):
        for center_index, orientation in Chem.FindMolChiralCenters(
            molecule,
            includeUnassigned=False,
        ):
            center = molecule.GetAtomWithIdx(center_index)
            neighbors = tuple(
                index
                for index, _rank in sorted(
                    ((neighbor.GetIdx(), int(neighbor.GetProp("_CIPRank"))) for neighbor in center.GetNeighbors()),
                    key=lambda value: value[1],
                    reverse=True,
                )
            )
            if len(neighbors) > 4 or center.GetHybridization() != HybridizationType.SP3:
                continue
            indices = (*neighbors[:3], center_index)
            if all(index in atom_index for index in indices):
                chiral_atoms.append(
                    cast(
                        tuple[int, int, int, int],
                        tuple(atom_index[index] for index in indices),
                    )
                )
                chiral_is_r.append(orientation == "R")
            if len(neighbors) == 4:
                for skipped in range(3):
                    neighbor_set = neighbors[:skipped] + neighbors[skipped + 1 :]
                    indices = (
                        neighbor_set[::-1] + (center_index,) if skipped % 2 == 0 else neighbor_set + (center_index,)
                    )
                    if all(index in atom_index for index in indices):
                        chiral_atoms.append(
                            cast(
                                tuple[int, int, int, int],
                                tuple(atom_index[index] for index in indices),
                            )
                        )
                        chiral_is_r.append(orientation == "R")

    stereo_bonds: list[tuple[int, int, int, int]] = []
    stereo_is_e: list[bool] = []
    if all(atom.HasProp("_CIPRank") for atom in molecule.GetAtoms()):
        for bond in molecule.GetBonds():
            stereo = bond.GetStereo()
            if stereo not in {BondStereo.STEREOE, BondStereo.STEREOZ}:
                continue
            start_index = bond.GetBeginAtomIdx()
            end_index = bond.GetEndAtomIdx()
            start_neighbors = [
                index
                for index, _rank in sorted(
                    (
                        (neighbor.GetIdx(), int(neighbor.GetProp("_CIPRank")))
                        for neighbor in molecule.GetAtomWithIdx(start_index).GetNeighbors()
                        if neighbor.GetIdx() != end_index
                    ),
                    key=lambda value: value[1],
                    reverse=True,
                )
            ]
            end_neighbors = [
                index
                for index, _rank in sorted(
                    (
                        (neighbor.GetIdx(), int(neighbor.GetProp("_CIPRank")))
                        for neighbor in molecule.GetAtomWithIdx(end_index).GetNeighbors()
                        if neighbor.GetIdx() != start_index
                    ),
                    key=lambda value: value[1],
                    reverse=True,
                )
            ]
            indices = (start_neighbors[0], start_index, end_index, end_neighbors[0])
            if all(index in atom_index for index in indices):
                stereo_bonds.append(
                    cast(
                        tuple[int, int, int, int],
                        tuple(atom_index[index] for index in indices),
                    )
                )
                stereo_is_e.append(stereo == BondStereo.STEREOE)
            if len(start_neighbors) == 2 and len(end_neighbors) == 2:
                indices = (
                    start_neighbors[1],
                    start_index,
                    end_index,
                    end_neighbors[1],
                )
                if all(index in atom_index for index in indices):
                    stereo_bonds.append(
                        cast(
                            tuple[int, int, int, int],
                            tuple(atom_index[index] for index in indices),
                        )
                    )
                    stereo_is_e.append(stereo == BondStereo.STEREOE)

    planar_pattern = Chem.MolFromSmarts("[C;X3;^2](*)(*)=[C;X3;^2](*)(*)")
    planar_bonds = [
        cast(
            tuple[int, int, int, int, int, int],
            tuple(atom_index[index] for index in match),
        )
        for match in molecule.GetSubstructMatches(planar_pattern)
        if all(index in atom_index for index in match)
    ]
    return (
        geometry_pairs,
        geometry_bond_mask,
        geometry_angle_mask,
        geometry_upper_bounds,
        geometry_lower_bounds,
        chiral_atoms,
        chiral_is_r,
        stereo_bonds,
        stereo_is_e,
        planar_bonds,
    )


def _contact_features(
    system: MolecularSystem,
    tokens: list[dict[str, object]],
    *,
    contacts: Sequence[Boltz2Contact],
    pockets: Sequence[Boltz2Pocket],
) -> dict[str, Tensor]:
    token_count = len(tokens)
    token_by_site = {
        cast(ComponentReference | AtomSite, token["token_site"]): token_index
        for token_index, token in enumerate(tokens)
    }
    contact_class = torch.full(
        (token_count, token_count),
        _CONTACT_UNSELECTED,
        dtype=torch.long,
    )
    threshold = torch.zeros(token_count, token_count)
    atom_sites = tuple(site for token in tokens for site in cast(tuple[AtomSite, ...], token["atom_sites"]))
    site_to_atom = {site: index for index, site in enumerate(atom_sites)}
    token_atoms = [
        torch.tensor(
            [site_to_atom[site] for site in cast(tuple[AtomSite, ...], token["atom_sites"])],
            dtype=torch.long,
        )
        for token in tokens
    ]
    potential_pairs: list[Tensor] = []
    potential_union: list[Tensor] = []
    potential_thresholds: list[Tensor] = []
    union_index = 0

    # Pocket conditioning is directional and precedes exact contacts. A later
    # contact therefore replaces both the class and threshold for an overlap.
    for pocket in pockets:
        binder_entity_id = system.instances[pocket.binder]
        binder_tokens = [
            token_index
            for token_index, token in enumerate(tokens)
            if token["instance_id"] == pocket.binder and token["entity_id"] == binder_entity_id
        ]
        for endpoint in pocket.contacts:
            pocket_token = _condition_endpoint_token(system, token_by_site, endpoint)
            if pocket_token is None:
                continue
            contact_class[binder_tokens, pocket_token] = _CONTACT_BINDER_TO_POCKET
            contact_class[pocket_token, binder_tokens] = _CONTACT_POCKET_TO_BINDER
            threshold[binder_tokens, pocket_token] = pocket.max_distance
            threshold[pocket_token, binder_tokens] = pocket.max_distance
            if pocket.force:
                binder_atoms = torch.cat([token_atoms[index] for index in binder_tokens])
                pairs = torch.cartesian_prod(binder_atoms, token_atoms[pocket_token]).T
                potential_pairs.append(pairs)
                potential_union.append(torch.full((pairs.shape[1],), union_index))
                potential_thresholds.append(torch.full((pairs.shape[1],), pocket.max_distance))
                union_index += 1

    for contact in contacts:
        token_a = _condition_endpoint_token(system, token_by_site, contact.endpoint_a)
        token_b = _condition_endpoint_token(system, token_by_site, contact.endpoint_b)
        if token_a is None or token_b is None:
            continue
        contact_class[token_a, token_b] = _CONTACT_PAIR
        contact_class[token_b, token_a] = _CONTACT_PAIR
        threshold[token_a, token_b] = contact.max_distance
        threshold[token_b, token_a] = contact.max_distance
        if contact.force:
            pairs = torch.cartesian_prod(token_atoms[token_a], token_atoms[token_b]).T
            potential_pairs.append(pairs)
            potential_union.append(torch.full((pairs.shape[1],), union_index))
            potential_thresholds.append(torch.full((pairs.shape[1],), contact.max_distance))
            union_index += 1

    if bool((contact_class == _CONTACT_UNSELECTED).all()):
        contact_class.fill_(_CONTACT_UNSPECIFIED)
    if potential_pairs:
        pair_index = torch.cat(potential_pairs, dim=1)
        union = torch.cat(potential_union)
    else:
        pair_index = torch.empty(2, 0, dtype=torch.long)
        union = torch.empty(0, dtype=torch.long)
    return {
        "contact_conditioning": torch.nn.functional.one_hot(contact_class, 5),
        "contact_threshold": threshold,
        "contact_pair_index": pair_index,
        "contact_union_index": union,
        "contact_thresholds": torch.cat(potential_thresholds) if potential_thresholds else torch.empty(0),
    }


def _condition_endpoint_token(
    system: MolecularSystem,
    token_by_site: dict[ComponentReference | AtomSite, int],
    endpoint: ComponentReference | AtomSite,
) -> int | None:
    if isinstance(endpoint, AtomSite):
        entity = system.entities[system.instances[endpoint.instance_id]]
        system.atom(endpoint)
        if isinstance(entity, PolymerSequence):
            raise ValueError("Boltz-2 polymer contact endpoints must identify residues, not exact atoms")
        return token_by_site.get(endpoint)

    instance_id, component_index = endpoint
    entity = system.entities[system.instances[instance_id]]
    if not isinstance(entity, PolymerSequence):
        raise ValueError("Boltz-2 non-polymer contact endpoints must identify exact atoms")
    system.component(endpoint)
    return token_by_site.get(endpoint)


def _template_features(
    system: MolecularSystem,
    tokens: list[dict[str, object]],
    templates: Sequence[Template | Boltz2Template],
) -> dict[str, Tensor]:
    if not templates:
        return {}

    token_count = len(tokens)
    token_by_residue = {
        (cast(str, token["instance_id"]), cast(int, token["monomer_index"])): token_index
        for token_index, token in enumerate(tokens)
        if token["monomer_index"] is not None
    }
    tokens_by_instance: dict[str, list[int]] = {}
    for token_index, token in enumerate(tokens):
        tokens_by_instance.setdefault(cast(str, token["instance_id"]), []).append(token_index)

    residue_type_rows: list[Tensor] = []
    frame_rotation_rows: list[Tensor] = []
    frame_translation_rows: list[Tensor] = []
    frame_mask_rows: list[Tensor] = []
    beta_carbon_position_rows: list[Tensor] = []
    alpha_carbon_position_rows: list[Tensor] = []
    beta_carbon_mask_rows: list[Tensor] = []
    template_mask_rows: list[Tensor] = []
    visibility_rows: list[Tensor] = []
    force_rows: list[bool] = []
    force_threshold_rows: list[float] = []
    identity = torch.eye(3)

    for template_index, value in enumerate(templates):
        if isinstance(value, Boltz2Template):
            template = value.template
            force_rows.append(value.force)
            if value.force and value.threshold is None:
                raise ValueError("a forced Boltz-2 template requires threshold")
            force_threshold_rows.append(float("inf") if value.threshold is None else value.threshold)
        else:
            template = value
            force_rows.append(False)
            force_threshold_rows.append(float("inf"))
        residue_type = torch.zeros(token_count, dtype=torch.long)
        frame_rotations = torch.zeros(token_count, 3, 3)
        frame_translations = torch.zeros(token_count, 3)
        frame_mask = torch.zeros(token_count)
        beta_carbon_positions = torch.zeros(token_count, 3)
        alpha_carbon_positions = torch.zeros(token_count, 3)
        beta_carbon_mask = torch.zeros(token_count)
        template_mask = torch.zeros(token_count)
        visibility = torch.tensor(
            [-1.0 - cast(int, token["chain_index"]) for token in tokens],
        )
        covered_instances: set[str] = set()
        coordinate_index = {site: atom_index for atom_index, site in enumerate(template.structure.sites)}

        for alignment in template.alignments:
            target_entity = system.entities[alignment.target_entity_id]
            if not isinstance(target_entity, PolymerSequence) or target_entity.polymer_type != "protein":
                raise ValueError("Boltz-2 templates support only protein target alignments")

            template_entity_id = template.structure.system.instances[alignment.template_instance_id]
            template_entity = template.structure.system.entities[template_entity_id]
            if not isinstance(template_entity, PolymerSequence) or template_entity.polymer_type != "protein":
                raise ValueError("Boltz-2 templates support only protein template alignments")

            if alignment.target_instance_id is None:
                target_instances = tuple(
                    instance_id
                    for instance_id, entity_id in system.instances.items()
                    if entity_id == alignment.target_entity_id
                )
            else:
                if system.instances[alignment.target_instance_id] != alignment.target_entity_id:
                    raise ValueError(
                        f"template target instance {alignment.target_instance_id!r} does not belong to "
                        f"entity {alignment.target_entity_id!r}"
                    )
                target_instances = (alignment.target_instance_id,)

            for target_component_index, template_component_index in alignment.residue_pairs:
                if target_component_index >= len(target_entity):
                    raise IndexError(
                        f"template target component index {target_component_index} is outside "
                        f"entity {alignment.target_entity_id!r}"
                    )
                template_component = template_entity.components[template_component_index]
                template_residue_type = _residue_type(template_component.component_id, "protein")
                atom_values = {
                    atom_id: _template_atom(
                        template.structure,
                        coordinate_index,
                        AtomSite(
                            alignment.template_instance_id,
                            template_component_index,
                            atom_id,
                        ),
                    )
                    for atom_id in ("N", "CA", "C", "CB")
                }
                beta_atom = "CA" if template_component.component_id == "GLY" else "CB"

                for target_instance in target_instances:
                    target_token = token_by_residue.get((target_instance, target_component_index))
                    if target_token is None:
                        continue
                    covered_instances.add(target_instance)
                    residue_type[target_token] = template_residue_type
                    template_mask[target_token] = 1
                    frame_rotations[target_token] = identity

                    alpha_carbon, alpha_carbon_present = atom_values["CA"]
                    if alpha_carbon_present:
                        alpha_carbon_positions[target_token] = alpha_carbon
                    beta_carbon, beta_carbon_present = atom_values[beta_atom]
                    if beta_carbon_present:
                        beta_carbon_positions[target_token] = beta_carbon
                        beta_carbon_mask[target_token] = 1

                    nitrogen, nitrogen_present = atom_values["N"]
                    carbon, carbon_present = atom_values["C"]
                    if nitrogen_present and alpha_carbon_present and carbon_present:
                        first = carbon - alpha_carbon
                        first_axis = first / (torch.linalg.vector_norm(first) + 1e-10)
                        second = nitrogen - alpha_carbon
                        second = second - first_axis * torch.dot(first_axis, second)
                        second_axis = second / (torch.linalg.vector_norm(second) + 1e-10)
                        third_axis = torch.linalg.cross(first_axis, second_axis)
                        frame_rotations[target_token] = torch.stack(
                            (first_axis, second_axis, third_axis),
                            dim=-1,
                        )
                        frame_translations[target_token] = alpha_carbon
                        frame_mask[target_token] = 1

        for instance_id in covered_instances:
            visibility[tokens_by_instance[instance_id]] = float(template_index)

        residue_type_rows.append(torch.nn.functional.one_hot(residue_type, 33))
        frame_rotation_rows.append(frame_rotations)
        frame_translation_rows.append(frame_translations)
        frame_mask_rows.append(frame_mask)
        beta_carbon_position_rows.append(beta_carbon_positions)
        alpha_carbon_position_rows.append(alpha_carbon_positions)
        beta_carbon_mask_rows.append(beta_carbon_mask)
        template_mask_rows.append(template_mask)
        visibility_rows.append(visibility)

    return {
        "template_residue_type_features": torch.stack(residue_type_rows),
        "template_frame_rotations": torch.stack(frame_rotation_rows),
        "template_frame_translations": torch.stack(frame_translation_rows),
        "template_frame_mask": torch.stack(frame_mask_rows),
        "template_beta_carbon_positions": torch.stack(beta_carbon_position_rows),
        "template_alpha_carbon_positions": torch.stack(alpha_carbon_position_rows),
        "template_beta_carbon_mask": torch.stack(beta_carbon_mask_rows),
        "template_mask": torch.stack(template_mask_rows),
        "template_visibility_index": torch.stack(visibility_rows),
        "template_force": torch.tensor(force_rows, dtype=torch.bool),
        "template_force_threshold": torch.tensor(force_threshold_rows),
    }


def _template_atom(
    structure: Structure,
    coordinate_index: dict[AtomSite, int],
    site: AtomSite,
) -> tuple[Tensor, bool]:
    index = coordinate_index.get(site)
    if index is None or not bool(structure.coordinate_mask[index]):
        return torch.zeros(3), False
    return torch.from_numpy(structure.coordinates[index].copy()).to(dtype=torch.float32), True


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
        polymer_types = tuple(
            cast(
                PolymerSequence,
                system.entities[value.entity_id],
            ).polymer_type
            for value in columns
        )
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
        "msa_tokens": rows,
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
    for space_id in range(int(space_ids.max().item()) + 1):
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
    atom_stereo: dict[int, str] = {}
    for component_index, component in enumerate(molecule.components):
        for atom in component.atoms:
            rdkit_atom = Chem.Atom(atom.element)
            if atom.isotope is not None:
                rdkit_atom.SetIsotope(atom.isotope)
            if atom.formal_charge is not None:
                rdkit_atom.SetFormalCharge(atom.formal_charge)
            if atom.radical_electrons is not None:
                rdkit_atom.SetNumRadicalElectrons(atom.radical_electrons)
            if atom.explicit_hydrogens is not None:
                rdkit_atom.SetNumExplicitHs(atom.explicit_hydrogens)
            if atom.implicit_hydrogens_allowed is not None:
                rdkit_atom.SetNoImplicit(not atom.implicit_hydrogens_allowed)
            if atom.aromatic is not None:
                rdkit_atom.SetIsAromatic(atom.aromatic)
            index = editable.AddAtom(rdkit_atom)
            atom_index[(component_index, atom.atom_id)] = index
            if atom.stereo is not None:
                atom_stereo[index] = atom.stereo.upper()

    bond_types = {
        1.0: Chem.BondType.SINGLE,
        1.5: Chem.BondType.AROMATIC,
        2.0: Chem.BondType.DOUBLE,
        3.0: Chem.BondType.TRIPLE,
        4.0: Chem.BondType.QUADRUPLE,
    }
    bond_stereo: list[tuple[int, str, tuple[int, int] | None]] = []
    for component_index, component in enumerate(molecule.components):
        component_bonds = sorted(
            component.bonds or (),
            key=lambda bond: (
                atom_index[(component_index, bond.atom_a)],
                atom_index[(component_index, bond.atom_b)],
            ),
        )
        for component_bond in component_bonds:
            editable.AddBond(
                atom_index[(component_index, component_bond.atom_a)],
                atom_index[(component_index, component_bond.atom_b)],
                bond_types[component_bond.order],
            )
            bond_index = editable.GetNumBonds() - 1
            editable.GetBondWithIdx(bond_index).SetIsAromatic(component_bond.aromatic)
            if component_bond.stereo is not None:
                stereo_atoms = (
                    tuple(atom_index[(component_index, atom_id)] for atom_id in component_bond.stereo_atoms)
                    if component_bond.stereo_atoms is not None
                    else None
                )
                bond_stereo.append(
                    (
                        bond_index,
                        component_bond.stereo,
                        cast(tuple[int, int] | None, stereo_atoms),
                    )
                )
    molecule_bonds = sorted(
        molecule.bonds,
        key=lambda bond: (
            atom_index[(bond.component_a, bond.atom_a)],
            atom_index[(bond.component_b, bond.atom_b)],
        ),
    )
    for molecule_bond in molecule_bonds:
        editable.AddBond(
            atom_index[(molecule_bond.component_a, molecule_bond.atom_a)],
            atom_index[(molecule_bond.component_b, molecule_bond.atom_b)],
            bond_types[molecule_bond.order],
        )
        bond_index = editable.GetNumBonds() - 1
        editable.GetBondWithIdx(bond_index).SetIsAromatic(molecule_bond.aromatic)
        if molecule_bond.stereo is not None:
            stereo_atoms = (
                tuple(atom_index[site] for site in molecule_bond.stereo_atoms)
                if molecule_bond.stereo_atoms is not None
                else None
            )
            bond_stereo.append(
                (
                    bond_index,
                    molecule_bond.stereo,
                    cast(tuple[int, int] | None, stereo_atoms),
                )
            )

    value = editable.GetMol()
    Chem.SanitizeMol(value)
    for index in atom_stereo:
        value.GetAtomWithIdx(index).SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
    Chem.AssignStereochemistry(value, force=True, cleanIt=True)
    for index, desired in atom_stereo.items():
        atom = value.GetAtomWithIdx(index)
        assigned = atom.GetProp("_CIPCode").upper() if atom.HasProp("_CIPCode") else ""
        if assigned != desired:
            atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
    Chem.AssignStereochemistry(value, force=True, cleanIt=True)

    stereo_types = {
        "E": Chem.BondStereo.STEREOE,
        "TRANS": Chem.BondStereo.STEREOE,
        "Z": Chem.BondStereo.STEREOZ,
        "CIS": Chem.BondStereo.STEREOZ,
    }
    for index, stereo, stereo_atoms in bond_stereo:
        bond = value.GetBondWithIdx(index)
        if stereo_atoms is None:
            begin = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()
            begin_neighbors = [
                neighbor for neighbor in value.GetAtomWithIdx(begin).GetNeighbors() if neighbor.GetIdx() != end
            ]
            end_neighbors = [
                neighbor for neighbor in value.GetAtomWithIdx(end).GetNeighbors() if neighbor.GetIdx() != begin
            ]
            stereo_atoms = (
                max(begin_neighbors, key=lambda atom: int(atom.GetProp("_CIPRank"))).GetIdx(),
                max(end_neighbors, key=lambda atom: int(atom.GetProp("_CIPRank"))).GetIdx(),
            )
        bond.SetStereoAtoms(*stereo_atoms)
        bond.SetStereo(stereo_types[stereo.upper()])
    Chem.AssignStereochemistry(value, force=True, cleanIt=True)
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
