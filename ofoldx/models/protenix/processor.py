r"""
Prepare biomolecular features for Protenix.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from ...data import ATOMIC_NUMBER_BY_ELEMENT
from ...data.components import get_component
from ...data.evidence import (
    MSA,
    MSACollection,
    Template,
    TemplateAlignment,
    msa_binding_map,
    msa_bindings,
    resolve_msa_bindings,
    select_msa_binding,
)
from ...data.molecular import (
    AtomSite,
    ChemicalComponent,
    ComponentReference,
    MolecularSystem,
    Molecule,
    PolymerSequence,
)
from ...data.structure import Structure
from ...modules.geometry import uniform_random_rotation
from ...processor import Processor
from .config import ProtenixConstraintProcessorConfig, ProtenixProcessorConfig
from .model import (
    ProtenixAtomContact,
    ProtenixInput,
    ProtenixPocket,
    ProtenixStructureTarget,
    ProtenixTemplateFeatures,
    ProtenixTfgFeatures,
    ProtenixTfgGuidance,
    ProtenixTokenContact,
)

if TYPE_CHECKING:
    from rdkit.Chem.rdchem import Mol

_RESIDUE_TYPE = {
    name: index
    for index, name in enumerate(
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
            "UNK",
            "A",
            "G",
            "C",
            "U",
            "N",
            "DA",
            "DG",
            "DC",
            "DT",
            "DN",
            "-",
        )
    )
}
_STANDARD = {
    "protein": frozenset(tuple(_RESIDUE_TYPE)[:21]),
    "rna": frozenset(("A", "G", "C", "U", "N")),
    "dna": frozenset(("DA", "DG", "DC", "DT", "DN")),
}
_METAL_ATOMIC_NUMBERS = frozenset(
    {
        3,
        4,
        *range(11, 14),
        *range(19, 32),
        *range(37, 52),
        *range(55, 85),
        *range(87, 119),
    }
)
_UNKNOWN = _RESIDUE_TYPE["UNK"]
_GAP = _RESIDUE_TYPE["-"]
# MSA cells are UPPERCASE one-letter symbols (or the ``-`` gap), not ChemicalComponent IDs. This
# projects the twenty standard protein one-letter codes onto their residue-type indices; ``-`` becomes
# the gap class and every other symbol (ambiguity codes B Z U J O X, etc.) becomes UNK.
_MSA_RESIDUE_TYPE = {
    symbol: _RESIDUE_TYPE[three]
    for symbol, three in {
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
_MSA_RESIDUE_TYPE["-"] = _GAP
_RNA_MSA_RESIDUE_TYPE = {
    **{chr(index): _RESIDUE_TYPE["N"] for index in range(ord("A"), ord("Z") + 1)},
    "-": _GAP,
    "A": _RESIDUE_TYPE["A"],
    "G": _RESIDUE_TYPE["G"],
    "C": _RESIDUE_TYPE["C"],
    "U": _RESIDUE_TYPE["U"],
}
_ESM_TOKEN_BY_COMPONENT = {
    "ALA": 5,
    "ARG": 10,
    "ASN": 17,
    "ASP": 13,
    "CYS": 23,
    "GLN": 16,
    "GLU": 9,
    "GLY": 6,
    "HIS": 21,
    "ILE": 12,
    "LEU": 4,
    "LYS": 15,
    "MET": 20,
    "PHE": 18,
    "PRO": 14,
    "SER": 8,
    "THR": 11,
    "TRP": 22,
    "TYR": 19,
    "VAL": 7,
    "UNK": 24,
    "ASX": 25,
    "SEC": 26,
    "GLX": 27,
    "PYL": 28,
}
_ESM_CLS_TOKEN = 0
_ESM_PAD_TOKEN = 1
_ESM_EOS_TOKEN = 2
_ESM_UNKNOWN_RESIDUE_TOKEN = 24


class ProtenixProcessor(Processor):
    r"""
    Convert supported biomolecular inputs into `ProtenixInput`.

    Args:
        config: Feature limits and reference-position options. Defaults to
            ``ProtenixProcessorConfig()``.
        device: Device for produced tensors.
        dtype: Floating-point dtype for produced tensors.
    """

    config_class = ProtenixProcessorConfig
    config: ProtenixProcessorConfig

    def __init__(
        self,
        config: ProtenixProcessorConfig | None = None,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.config = ProtenixProcessorConfig() if config is None else config
        if not 1 <= self.config.max_msa_rows <= 16384:
            raise ValueError("max_msa_rows must be between 1 and 16384")
        super().__init__(device=device, dtype=dtype)

    def __call__(
        self,
        input: PolymerSequence | Sequence[PolymerSequence] | MolecularSystem | Structure,
        *,
        msa: MSA | MSACollection | None = None,
        tfg_guidance: ProtenixTfgGuidance | None = None,
        generator: torch.Generator | None = None,
    ) -> ProtenixInput:
        r"""
        Prepare model features for one molecular assembly.

        Args:
            input: One or more ``PolymerSequence`` values, ``MolecularSystem``,
                or ``Structure``. A structure contributes topology, not observed
                coordinates.
            msa: Optional ``MSA`` or ``MSACollection``.
            tfg_guidance: Training-free guidance whose static molecular geometry
                should be prepared. ``None`` performs no TFG geometry work.
            generator: Optional PyTorch generator for reference-position augmentation.

        Returns:
            Batched token, atom, and MSA tensors. Reference coordinates are in
            angstroms. Atom-aligned tensors follow ``atom_sites`` and token-aligned
            tensors follow ``token_sites``.
        """

        return self._lower(
            input,
            msa=msa,
            tfg_guidance=tfg_guidance,
            generator=generator,
        )

    def process_training(
        self,
        input: Structure,
        *,
        msa: MSA | MSACollection | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[ProtenixInput, ProtenixStructureTarget]:
        r"""
        Lower one complete structure into model input and separate coordinate truth.
        """

        processed = self(
            input,
            msa=msa,
            generator=generator,
        )
        return processed, _structure_target(
            input,
            processed,
            device=self.device,
            dtype=self.dtype,
        )

    def _normalize_msa(
        self,
        msa: MSA | MSACollection | None,
        system: MolecularSystem,
    ) -> MSA | MSACollection | None:
        return _normalize_msa(msa, system)

    def _lower(
        self,
        input: PolymerSequence | Sequence[PolymerSequence] | MolecularSystem | Structure,
        *,
        msa: MSA | MSACollection | None,
        templates: tuple[Template, ...] | None = None,
        atom_contacts: Sequence[ProtenixAtomContact] = (),
        token_contacts: Sequence[ProtenixTokenContact] = (),
        pocket: ProtenixPocket | None = None,
        substructure: Structure | None = None,
        substructure_coordinate_noise_scale: float = 0.0,
        structure_condition: Structure | None = None,
        tfg_guidance: ProtenixTfgGuidance | None = None,
        generator: torch.Generator | None = None,
    ) -> ProtenixInput:
        atom_contact_values = tuple(atom_contacts)
        token_contact_values = tuple(token_contacts)
        system = molecular_system_from_input(input)
        normalized_msa = self._normalize_msa(msa, system)
        # Atom contacts require atom-level tokens. Token contacts and pockets use
        # the tokenization already selected for each residue.
        expanded_residues = _atom_constraint_residues(atom_contact_values, system)
        tokens = _tokenize(system, expanded_residues)
        return _features(
            system,
            tokens=tokens,
            msa=normalized_msa,
            templates=templates,
            atom_contacts=atom_contact_values,
            token_contacts=token_contact_values,
            pocket=pocket,
            substructure=substructure,
            substructure_coordinate_noise_scale=substructure_coordinate_noise_scale,
            structure_condition=structure_condition,
            tfg_guidance=tfg_guidance,
            generator=generator,
            max_msa_rows=self.config.max_msa_rows,
            augment_reference_positions=self.config.augment_reference_positions,
            device=self.device,
            dtype=self.dtype,
        )


class ProtenixTemplateProcessor(ProtenixProcessor):
    r"""
    Convert template-capable Protenix v1 and v2 inputs into `ProtenixInput`.

    Args:
        config: Feature limits and reference-position options.
        device: Device for produced tensors.
        dtype: Floating-point dtype for produced tensors.
    """

    def __call__(
        self,
        input: PolymerSequence | Sequence[PolymerSequence] | MolecularSystem | Structure,
        *,
        msa: MSA | MSACollection | None = None,
        templates: Sequence[Template] = (),
        tfg_guidance: ProtenixTfgGuidance | None = None,
        generator: torch.Generator | None = None,
    ) -> ProtenixInput:
        r"""
        Prepare protein/RNA MSA and fixed-four protein-template features.

        Args:
            input: One or more ``PolymerSequence`` values, ``MolecularSystem``,
                or ``Structure``.
            msa: Optional protein and unpaired-RNA MSA evidence.
            templates: Protein templates aligned to target chains.
            tfg_guidance: Training-free guidance whose static molecular geometry
                should be prepared. ``None`` performs no TFG geometry work.
            generator: Optional generator for reference-position augmentation.

        Returns:
            Batched token, atom, MSA, and fixed-four template tensors.
        """

        return self._lower(
            input,
            msa=msa,
            templates=tuple(templates),
            tfg_guidance=tfg_guidance,
            generator=generator,
        )

    def process_training(
        self,
        input: Structure,
        *,
        msa: MSA | MSACollection | None = None,
        templates: Sequence[Template] = (),
        generator: torch.Generator | None = None,
    ) -> tuple[ProtenixInput, ProtenixStructureTarget]:
        r"""
        Lower template-capable input and separate complete coordinate truth.
        """

        processed = self(
            input,
            msa=msa,
            templates=templates,
            generator=generator,
        )
        return processed, _structure_target(
            input,
            processed,
            device=self.device,
            dtype=self.dtype,
        )

    def _normalize_msa(
        self,
        msa: MSA | MSACollection | None,
        system: MolecularSystem,
    ) -> MSA | MSACollection | None:
        return _normalize_template_msa(msa, system)


class ProtenixStructureConditionProcessor(ProtenixProcessor):
    r"""
    Convert a molecular assembly and an explicit coordinate condition into
    structure-conditioned Protenix inputs.

    Args:
        config: Feature limits and reference-position options.
        device: Device for produced tensors.
        dtype: Floating-point dtype for produced tensors.
    """

    def __call__(
        self,
        input: PolymerSequence | Sequence[PolymerSequence] | MolecularSystem | Structure,
        *,
        structure_condition: Structure | None = None,
        msa: MSA | MSACollection | None = None,
        tfg_guidance: ProtenixTfgGuidance | None = None,
        generator: torch.Generator | None = None,
    ) -> ProtenixInput:
        r"""
        Prepare the released representative-atom structure condition.

        Args:
            input: One or more ``PolymerSequence`` values, ``MolecularSystem``,
                or ``Structure``. A primary Structure still contributes topology
                only.
            structure_condition: Optional in-memory coordinates matched to the
                input's representative token atoms.
            msa: Optional protein MSA evidence.
            tfg_guidance: Optional training-free guidance whose static molecular
                geometry should be prepared.
            generator: Optional generator for reference-position augmentation.

        Returns:
            Batched Protenix tensors with representative coordinates and their
            validity mask aligned to the token axis.
        """

        return self._lower(
            input,
            msa=msa,
            structure_condition=structure_condition,
            tfg_guidance=tfg_guidance,
            generator=generator,
        )

    def process_training(
        self,
        input: Structure,
        *,
        structure_condition: Structure | None = None,
        msa: MSA | MSACollection | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[ProtenixInput, ProtenixStructureTarget]:
        r"""
        Lower structure-conditioned input and separate complete coordinate truth.
        """

        processed = self(
            input,
            structure_condition=structure_condition,
            msa=msa,
            generator=generator,
        )
        return processed, _structure_target(
            input,
            processed,
            device=self.device,
            dtype=self.dtype,
        )


class ProtenixConstraintProcessor(ProtenixProcessor):
    r"""
    Convert biomolecular inputs and spatial constraints into `ProtenixInput`.

    Args:
        config: Constraint feature limits and reference-position options. Defaults to
            ``ProtenixConstraintProcessorConfig()``.
        device: Device for produced tensors.
        dtype: Floating-point dtype for produced tensors.
    """

    config_class = ProtenixConstraintProcessorConfig
    config: ProtenixConstraintProcessorConfig

    def __init__(
        self,
        config: ProtenixConstraintProcessorConfig | None = None,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            ProtenixConstraintProcessorConfig() if config is None else config,
            device=device,
            dtype=dtype,
        )

    def __call__(
        self,
        input: PolymerSequence | Sequence[PolymerSequence] | MolecularSystem | Structure,
        *,
        msa: MSA | MSACollection | None = None,
        atom_contacts: Sequence[ProtenixAtomContact] = (),
        token_contacts: Sequence[ProtenixTokenContact] = (),
        pocket: ProtenixPocket | None = None,
        substructure: Structure | None = None,
        substructure_coordinate_noise_scale: float = 0.0,
        tfg_guidance: ProtenixTfgGuidance | None = None,
        generator: torch.Generator | None = None,
    ) -> ProtenixInput:
        r"""
        Prepare model features and spatial constraints for one assembly.

        Args:
            input: One or more ``PolymerSequence`` values, ``MolecularSystem``,
                or ``Structure``. A structure contributes topology, not observed
                coordinates.
            msa: Optional ``MSA`` or ``MSACollection``.
            atom_contacts: Upper-bound constraints between exact atom pairs.
            token_contacts: Upper-bound constraints between token-level endpoints.
            pocket: Optional directed binder-to-residue pocket constraint.
            substructure: Optional resolved token-center coordinates used for the
                released four-channel substructure condition.
            substructure_coordinate_noise_scale: Standard deviation of Gaussian
                noise added to selected center coordinates before distance binning.
            tfg_guidance: Training-free guidance whose static molecular geometry
                should be prepared. ``None`` performs no TFG geometry work.
            generator: Optional generator for reference-position augmentation
                and random selection among multi-token constraint endpoints.

        Returns:
            Batched token, atom, MSA, and constraint tensors. Reference
            coordinates and distance bounds are in angstroms. Atom-aligned tensors
            follow ``atom_sites``; token-aligned tensors follow ``token_sites``.
        """

        return self._lower(
            input,
            msa=msa,
            atom_contacts=atom_contacts,
            token_contacts=token_contacts,
            pocket=pocket,
            substructure=substructure,
            substructure_coordinate_noise_scale=substructure_coordinate_noise_scale,
            tfg_guidance=tfg_guidance,
            generator=generator,
        )

    def process_training(
        self,
        input: Structure,
        *,
        msa: MSA | MSACollection | None = None,
        atom_contacts: Sequence[ProtenixAtomContact] = (),
        token_contacts: Sequence[ProtenixTokenContact] = (),
        pocket: ProtenixPocket | None = None,
        substructure: Structure | None = None,
        substructure_coordinate_noise_scale: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> tuple[ProtenixInput, ProtenixStructureTarget]:
        r"""
        Lower constrained input and separate complete coordinate truth.
        """

        processed = self(
            input,
            msa=msa,
            atom_contacts=atom_contacts,
            token_contacts=token_contacts,
            pocket=pocket,
            substructure=substructure,
            substructure_coordinate_noise_scale=substructure_coordinate_noise_scale,
            generator=generator,
        )
        return processed, _structure_target(
            input,
            processed,
            device=self.device,
            dtype=self.dtype,
        )


def _structure_target(
    structure: Structure,
    input: ProtenixInput,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> ProtenixStructureTarget:
    r"""
    Lower the complete Structure coordinate pool and current atom-axis mapping.
    """

    coordinate_index = {site: index for index, site in enumerate(structure.sites)}
    atom_index = torch.tensor(
        [coordinate_index.get(site, -1) for site in input.atom_sites],
        dtype=torch.int64,
    )
    return ProtenixStructureTarget(
        coordinates=torch.from_numpy(structure.coordinates.copy()).to(device=device, dtype=dtype).unsqueeze(0),
        coordinate_mask=torch.from_numpy(structure.coordinate_mask.copy()).to(device=device).unsqueeze(0),
        atom_index=atom_index.to(device=device).unsqueeze(0),
        batch_size=[1],
    )


def molecular_system_from_input(
    input: PolymerSequence | Sequence[PolymerSequence] | MolecularSystem | Structure,
) -> MolecularSystem:
    r"""
    Normalize one Protenix biological input to its molecular system.

    Args:
        input: One polymer, a polymer collection, a molecular system, or a
            structure whose topology supplies the system.

    Returns:
        The existing molecular system or one constructed from the supplied
        polymer sequence values.
    """

    # A Structure supplied as the primary input contributes only topology. The
    # constraint Processor lowers coordinates only from its explicit substructure argument.
    if isinstance(input, Structure):
        return input.system
    if isinstance(input, MolecularSystem):
        return input
    if isinstance(input, PolymerSequence):
        return molecular_system_from_named_sequences((("A", input),))
    if isinstance(input, Sequence):
        sequence_values = tuple(cast(Sequence[object], input))
        if not sequence_values or not all(isinstance(sequence, PolymerSequence) for sequence in sequence_values):
            raise TypeError("Protenix sequence collections must contain PolymerSequence values")
        polymer_sequences = cast(tuple[PolymerSequence, ...], sequence_values)
        return molecular_system_from_named_sequences(
            tuple((f"chain_{index + 1}", sequence) for index, sequence in enumerate(polymer_sequences))
        )
    raise TypeError(
        "Protenix input must be a PolymerSequence, sequence of PolymerSequence " "values, MolecularSystem, or Structure"
    )


def molecular_system_from_named_sequences(
    named: tuple[tuple[str, PolymerSequence], ...],
) -> MolecularSystem:
    r"""
    Construct a Protenix molecular system from named polymer instances.

    Repeated names receive deterministic numeric suffixes while preserving
    declaration order.

    Args:
        named: Ordered ``(base_id, sequence)`` pairs.

    Returns:
        A molecular system with one entity and instance per sequence.
    """

    used: set[str] = set()
    entities: dict[str, PolymerSequence | Molecule] = {}
    instances: dict[str, str] = {}
    for base_id, sequence in named:
        entity_id = base_id
        suffix = 1
        while entity_id in used:
            suffix += 1
            entity_id = f"{base_id}_{suffix}"
        used.add(entity_id)
        entities[entity_id] = sequence
        instances[entity_id] = entity_id
    return MolecularSystem(entities, instances)


def _normalize_msa(
    msa: MSA | MSACollection | None,
    system: MolecularSystem,
) -> MSA | MSACollection | None:
    if msa is None:
        return None
    for _value, _target_instance_id, entity in resolve_msa_bindings(msa, system):
        if entity.polymer_type != "protein":
            raise ValueError("Protenix accepts supplied MSA evidence only for proteins")
    return msa


def _normalize_template_msa(
    msa: MSA | MSACollection | None,
    system: MolecularSystem,
) -> MSA | MSACollection | None:
    if msa is None:
        return None
    for _value, _target_instance_id, entity in resolve_msa_bindings(msa, system):
        if entity.polymer_type not in {"protein", "rna"}:
            raise ValueError("template-capable Protenix accepts supplied MSA evidence only for proteins and RNA")
    return msa


def _atom_constraint_residues(
    atom_contacts: tuple[ProtenixAtomContact, ...],
    system: MolecularSystem,
) -> frozenset[ComponentReference]:
    # Every polymer residue named by an atom contact must lower to per-atom tokens so the two
    # endpoints occupy distinct atom-level tokens. Non-polymer atoms are already per-atom tokens.
    residues: set[ComponentReference] = set()
    for contact in atom_contacts:
        for site in (contact.atom_a, contact.atom_b):
            if not isinstance(system.entities[system.instances[site.instance_id]], Molecule):
                residues.add((site.instance_id, site.component_index))
    return frozenset(residues)


def _is_standard(component: ChemicalComponent, polymer_type: str) -> bool:
    if component.component_id not in _STANDARD[polymer_type]:
        return False
    return component == get_component(component.component_id)


def _tokenize(system: MolecularSystem, expanded: frozenset[tuple[str, int]]) -> list[dict[str, object]]:
    available = {site for site, atom in system.iter_atoms() if atom.element != "H"}
    tokens: list[dict[str, object]] = []
    entity_indices: dict[str, int] = {}
    entity_copies: dict[str, int] = {}
    reference_space = 0

    for chain_index, (instance_id, entity_id) in enumerate(system.instances.items()):
        entity = system.entities[entity_id]
        entity_index = entity_indices.setdefault(entity_id, len(entity_indices))
        symmetry_index = entity_copies.get(entity_id, 0)
        entity_copies[entity_id] = symmetry_index + 1
        if isinstance(entity, PolymerSequence):
            for monomer_index, component in enumerate(entity.components):
                component_id = component.component_id
                atom_sites = tuple(
                    site
                    for atom in component.atoms
                    if atom.element != "H"
                    and not (component_id in {"N", "DN"} and atom.atom_id == "N")
                    and (site := AtomSite(instance_id, monomer_index, atom.atom_id)) in available
                )
                if not atom_sites:
                    raise ValueError(
                        f"Protenix component {component_id!r} at {instance_id}:{monomer_index} "
                        "has no model-visible heavy atoms"
                    )
                standard = _is_standard(component, entity.polymer_type)
                groups = (
                    tuple((site,) for site in atom_sites)
                    if not standard or (instance_id, monomer_index) in expanded
                    else (atom_sites,)
                )
                by_name = {site.atom_id: site for site in atom_sites}
                center_name = "CA" if entity.polymer_type == "protein" else "C1'"
                if entity.polymer_type == "protein":
                    representative_name = "CA" if component_id == "GLY" else "CB"
                elif component_id in {"A", "G", "DA", "DG"}:
                    representative_name = "C4"
                elif component_id in {"C", "U", "DC", "DT"}:
                    representative_name = "C2"
                else:
                    representative_name = "C1'"
                for group in groups:
                    atom_token = len(group) == 1 and (not standard or len(groups) > 1)
                    center_site = group[0] if atom_token else by_name[center_name]
                    representative_site = group[0] if atom_token else by_name.get(representative_name, center_site)
                    tokens.append(
                        {
                            "atoms": group,
                            "site": group[0] if atom_token else (instance_id, monomer_index),
                            "instance_id": instance_id,
                            "entity_id": entity_id,
                            "monomer_index": monomer_index,
                            "residue_index": monomer_index,
                            "chain_index": chain_index,
                            "entity_index": entity_index,
                            "symmetry_index": symmetry_index,
                            "residue_type": _RESIDUE_TYPE.get(component_id, _UNKNOWN) if standard else _UNKNOWN,
                            "center": center_site,
                            "representative": representative_site,
                            "standard": standard,
                            "expanded": standard and len(groups) > 1,
                            "reference_space": reference_space,
                        }
                    )
                reference_space += 1
        else:
            atom_sites = tuple(
                site
                for component_index, component in enumerate(entity.components)
                for atom in component.atoms
                if atom.element != "H" and (site := AtomSite(instance_id, component_index, atom.atom_id)) in available
            )
            if not atom_sites:
                raise ValueError(f"Protenix molecule {instance_id!r} has no model-visible heavy atoms")
            for site in atom_sites:
                tokens.append(
                    {
                        "atoms": (site,),
                        "site": site,
                        "instance_id": instance_id,
                        "entity_id": entity_id,
                        "monomer_index": None,
                        "residue_index": 0,
                        "chain_index": chain_index,
                        "entity_index": entity_index,
                        "symmetry_index": symmetry_index,
                        "residue_type": _UNKNOWN,
                        "center": site,
                        "representative": site,
                        "standard": False,
                        "expanded": False,
                        "reference_space": reference_space,
                    }
                )
            reference_space += 1
    return tokens


def _features(
    system: MolecularSystem,
    *,
    tokens: list[dict[str, object]],
    msa: MSA | MSACollection | None,
    templates: tuple[Template, ...] | None,
    atom_contacts: tuple[ProtenixAtomContact, ...],
    token_contacts: tuple[ProtenixTokenContact, ...],
    pocket: ProtenixPocket | None,
    substructure: Structure | None,
    substructure_coordinate_noise_scale: float,
    structure_condition: Structure | None,
    tfg_guidance: ProtenixTfgGuidance | None,
    generator: torch.Generator | None,
    max_msa_rows: int,
    augment_reference_positions: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> ProtenixInput:
    atom_sites = tuple(site for token in tokens for site in cast(tuple[AtomSite, ...], token["atoms"]))
    site_to_atom = {site: index for index, site in enumerate(atom_sites)}
    site_to_token = {
        site: token_index
        for token_index, token in enumerate(tokens)
        for site in cast(tuple[AtomSite, ...], token["atoms"])
    }
    atom_to_token = torch.tensor([site_to_token[site] for site in atom_sites], dtype=torch.long)
    reference_positions, reference_mask, reference_charges = _reference_features(
        system,
        atom_sites,
        site_to_atom,
        augment=augment_reference_positions,
        generator=generator,
    )
    atom_names = _model_atom_names(atom_sites)
    element_indices = torch.tensor([ATOMIC_NUMBER_BY_ELEMENT[system.atom(site).element] - 1 for site in atom_sites])
    name_indices = torch.tensor(
        [[min(63, max(0, ord(character) - 32)) for character in name.ljust(4)[:4]] for name in atom_names],
        dtype=torch.long,
    )
    atom_features = torch.cat(
        (
            reference_mask[:, None].float(),
            F.one_hot(element_indices, num_classes=128).float(),
            F.one_hot(name_indices, num_classes=64).reshape(len(atom_sites), 256).float(),
        ),
        dim=-1,
    )
    atom_token_index = torch.empty(len(atom_sites), dtype=torch.long)
    for token in tokens:
        group = cast(tuple[AtomSite, ...], token["atoms"])
        if cast(bool, token["expanded"]) or not cast(bool, token["standard"]):
            for site in group:
                atom_token_index[site_to_atom[site]] = 0
        else:
            instance_id = cast(str, token["instance_id"])
            entity = cast(PolymerSequence, system.entities[system.instances[instance_id]])
            component = entity.components[cast(int, token["monomer_index"])]
            if component.component_id == "N":
                heavy_order = {
                    name: index
                    for index, name in enumerate(
                        ("OP3", "P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'")
                    )
                }
            elif component.component_id == "DN":
                heavy_order = {
                    name: index
                    for index, name in enumerate(
                        ("OP3", "P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'")
                    )
                }
            else:
                heavy_order = {
                    atom.atom_id: index
                    for index, atom in enumerate(atom for atom in component.atoms if atom.element != "H")
                }
            for site in group:
                atom_token_index[site_to_atom[site]] = heavy_order[site.atom_id]

    _msa_rows, _msa_mask, profile, deletion_mean = _msa_features(msa, tokens, system, max_rows=max_msa_rows)
    residue_type = torch.tensor([cast(int, token["residue_type"]) for token in tokens])
    token_features = torch.cat(
        (
            torch.zeros(len(tokens), 384),
            F.one_hot(residue_type, num_classes=32).float(),
            profile,
            deletion_mean[:, None],
        ),
        dim=-1,
    )
    deletion_counts = _assembled_deletions(msa, tokens, system, max_rows=max_msa_rows)
    has_deletion = deletion_counts.clamp(0, 1)
    deletion_value = torch.atan(deletion_counts / 3.0) * (2.0 / math.pi)
    esm_tokens, esm_mask, token_to_esm_residue, token_has_esm = _esm_features(system, tokens)

    pocket_feature, token_contact_feature, atom_contact_feature, substructure_feature = _constraint_features(
        tokens=tokens,
        atom_contacts=atom_contacts,
        token_contacts=token_contacts,
        pocket=pocket,
        substructure=substructure,
        substructure_coordinate_noise_scale=substructure_coordinate_noise_scale,
        generator=generator,
        site_to_token=site_to_token,
    )
    token_bonds = _token_bonds(system, site_to_token, len(tokens))
    token_to_rep_atom = torch.tensor(
        [site_to_atom[cast(AtomSite, token["representative"])] for token in tokens], dtype=torch.long
    )
    structure_rep_positions, structure_rep_mask = _structure_condition_features(
        structure_condition,
        tokens,
    )
    template_features = (
        None
        if templates is None
        else _template_to(
            _template_features(templates, tokens, system),
            device=device,
            dtype=dtype,
        )
    )
    tfg_features = (
        None
        if tfg_guidance is None
        else _tfg_features(
            system,
            site_to_atom=site_to_atom,
            atom_chain=torch.tensor(
                [cast(int, tokens[site_to_token[site]]["chain_index"]) for site in atom_sites],
                dtype=torch.long,
            ),
            element_index=element_indices,
            device=device,
            dtype=dtype,
        )
    )

    def floating(value: Tensor) -> Tensor:
        return value.to(device=device, dtype=dtype).unsqueeze(0)

    def integer(value: Tensor) -> Tensor:
        return value.to(device=device).unsqueeze(0)

    return ProtenixInput(
        token_mask=integer(torch.ones(len(tokens), dtype=torch.bool)),
        residue_index=integer(torch.tensor([cast(int, token["residue_index"]) for token in tokens])),
        chain_index=integer(torch.tensor([cast(int, token["chain_index"]) for token in tokens])),
        entity_index=integer(torch.tensor([cast(int, token["entity_index"]) for token in tokens])),
        symmetry_index=integer(torch.tensor([cast(int, token["symmetry_index"]) for token in tokens])),
        token_index=integer(torch.arange(len(tokens))),
        atom_features=floating(atom_features),
        atom_to_token=integer(atom_to_token),
        atom_mask=integer(torch.ones(len(atom_sites), dtype=torch.bool)),
        reference_positions=floating(reference_positions),
        reference_space_id=integer(
            torch.tensor(
                [cast(int, tokens[site_to_token[site]]["reference_space"]) for site in atom_sites],
                dtype=torch.long,
            )
        ),
        atom_token_index=integer(atom_token_index),
        token_to_rep_atom=integer(token_to_rep_atom),
        token_features=floating(token_features),
        reference_mask=integer(reference_mask),
        reference_charges=floating(reference_charges[:, None]),
        token_bonds=floating(token_bonds),
        msa_tokens=integer(_msa_rows),
        has_deletion=floating(has_deletion),
        deletion_value=floating(deletion_value),
        esm_tokens=integer(esm_tokens),
        esm_mask=integer(esm_mask),
        token_to_esm_residue=integer(token_to_esm_residue),
        token_has_esm=integer(token_has_esm),
        pocket=floating(pocket_feature),
        token_contact=floating(token_contact_feature),
        atom_contact=floating(atom_contact_feature),
        substructure=floating(substructure_feature),
        structure_rep_positions=floating(structure_rep_positions),
        structure_rep_mask=integer(structure_rep_mask),
        template_features=template_features,
        tfg_features=tfg_features,
        tfg_guidance=tfg_guidance,
        system=system,
        atom_sites=atom_sites,
        token_sites=tuple(cast(AtomSite | ComponentReference, token["site"]) for token in tokens),
        batch_size=[1],
    )


@dataclass(frozen=True)
class _ProtenixComponentGeometry:
    pair_index: tuple[tuple[int, int], ...]
    pair_lower_bound: tuple[float, ...]
    pair_upper_bound: tuple[float, ...]
    pair_is_bond: tuple[bool, ...]
    pair_is_angle: tuple[bool, ...]
    experimental_torsion_index: tuple[tuple[int, int, int, int], ...]
    experimental_torsion_force_constant: tuple[tuple[float, ...], ...]
    experimental_torsion_sign: tuple[tuple[float, ...], ...]
    linear_bond_index: tuple[tuple[int, int, int], ...]
    chiral_index: tuple[tuple[int, int, int, int], ...]
    chiral_orientation: tuple[float, ...]
    stereo_bond_index: tuple[tuple[int, int, int, int], ...]
    stereo_bond_orientation: tuple[float, ...]
    planar_improper_index: tuple[tuple[int, int, int, int], ...]


def _tfg_features(
    system: MolecularSystem,
    *,
    site_to_atom: dict[AtomSite, int],
    atom_chain: Tensor,
    element_index: Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> ProtenixTfgFeatures:
    r"""
    Prepare the static molecular geometry used by Protenix TFG.
    """

    interchain_bonds = [
        (site_to_atom[bond.site_a], site_to_atom[bond.site_b])
        for bond in system.iter_bonds()
        if bond.site_a.instance_id != bond.site_b.instance_id
        and bond.site_a in site_to_atom
        and bond.site_b in site_to_atom
    ]
    pair_index: list[tuple[int, int]] = []
    pair_lower_bound: list[float] = []
    pair_upper_bound: list[float] = []
    pair_is_bond: list[bool] = []
    pair_is_angle: list[bool] = []
    experimental_torsion_index: list[tuple[int, int, int, int]] = []
    experimental_torsion_force_constant: list[tuple[float, ...]] = []
    experimental_torsion_sign: list[tuple[float, ...]] = []
    linear_bond_index: list[tuple[int, int, int]] = []
    chiral_index: list[tuple[int, int, int, int]] = []
    chiral_orientation: list[float] = []
    stereo_bond_index: list[tuple[int, int, int, int]] = []
    stereo_bond_orientation: list[float] = []
    planar_improper_index: list[tuple[int, int, int, int]] = []

    component_geometry: dict[ChemicalComponent, _ProtenixComponentGeometry] = {}
    for instance_id, entity_id in system.instances.items():
        entity = system.entities[entity_id]
        for component_index, component in enumerate(entity.components):
            if isinstance(entity, PolymerSequence) and component.component_id in _STANDARD[entity.polymer_type]:
                continue
            if any(ATOMIC_NUMBER_BY_ELEMENT[atom.element] in _METAL_ATOMIC_NUMBERS for atom in component.atoms):
                continue
            local_to_global: dict[int, int] = {}
            for local_index, atom in enumerate(component.atoms):
                site = AtomSite(instance_id, component_index, atom.atom_id)
                if site in site_to_atom:
                    local_to_global[local_index] = site_to_atom[site]
            if len(component.reference_coordinates) != len(component.atoms) or any(
                component.reference_coordinates[index] is None for index in local_to_global
            ):
                raise ValueError(
                    f"TFG requires reference coordinates for modeled atoms in component {component.component_id!r}"
                )
            geometry = component_geometry.get(component)
            if geometry is None:
                geometry = _component_geometry(component)
                component_geometry[component] = geometry

            for pair_local_index, lower, upper, is_bond, is_angle in zip(
                geometry.pair_index,
                geometry.pair_lower_bound,
                geometry.pair_upper_bound,
                geometry.pair_is_bond,
                geometry.pair_is_angle,
            ):
                pair_global_index = _mapped_index(pair_local_index, local_to_global)
                if pair_global_index is not None:
                    pair_index.append(cast(tuple[int, int], pair_global_index))
                    pair_lower_bound.append(lower)
                    pair_upper_bound.append(upper)
                    pair_is_bond.append(is_bond)
                    pair_is_angle.append(is_angle)
            for torsion_local_index, force_constant, sign in zip(
                geometry.experimental_torsion_index,
                geometry.experimental_torsion_force_constant,
                geometry.experimental_torsion_sign,
            ):
                torsion_global_index = _mapped_index(torsion_local_index, local_to_global)
                if torsion_global_index is not None:
                    experimental_torsion_index.append(cast(tuple[int, int, int, int], torsion_global_index))
                    experimental_torsion_force_constant.append(force_constant)
                    experimental_torsion_sign.append(sign)
            for linear_local_index in geometry.linear_bond_index:
                linear_global_index = _mapped_index(linear_local_index, local_to_global)
                if linear_global_index is not None:
                    linear_bond_index.append(cast(tuple[int, int, int], linear_global_index))
            for chiral_local_index, orientation in zip(geometry.chiral_index, geometry.chiral_orientation):
                chiral_global_index = _mapped_index(chiral_local_index, local_to_global)
                if chiral_global_index is not None:
                    chiral_index.append(cast(tuple[int, int, int, int], chiral_global_index))
                    chiral_orientation.append(orientation)
            for stereo_local_index, orientation in zip(
                geometry.stereo_bond_index,
                geometry.stereo_bond_orientation,
            ):
                stereo_global_index = _mapped_index(stereo_local_index, local_to_global)
                if stereo_global_index is not None:
                    stereo_bond_index.append(cast(tuple[int, int, int, int], stereo_global_index))
                    stereo_bond_orientation.append(orientation)
            for planar_local_index in geometry.planar_improper_index:
                planar_global_index = _mapped_index(planar_local_index, local_to_global)
                if planar_global_index is not None:
                    planar_improper_index.append(cast(tuple[int, int, int, int], planar_global_index))

    return ProtenixTfgFeatures(
        atom_chain=atom_chain.to(device=device),
        atom_vdw_radius=_vdw_radii(device=device, dtype=dtype)[element_index.to(device=device)],
        interchain_bond_index=_index_tensor(interchain_bonds, 2, device),
        pairwise_distance_index=_index_tensor(pair_index, 2, device),
        pairwise_distance_lower_bound=torch.tensor(pair_lower_bound, device=device, dtype=dtype),
        pairwise_distance_upper_bound=torch.tensor(pair_upper_bound, device=device, dtype=dtype),
        pairwise_distance_is_bond=torch.tensor(pair_is_bond, device=device, dtype=torch.bool),
        pairwise_distance_is_angle=torch.tensor(pair_is_angle, device=device, dtype=torch.bool),
        experimental_torsion_index=_index_tensor(experimental_torsion_index, 4, device),
        experimental_torsion_force_constant=torch.tensor(
            experimental_torsion_force_constant,
            device=device,
            dtype=dtype,
        ).reshape(-1, 6),
        experimental_torsion_sign=torch.tensor(
            experimental_torsion_sign,
            device=device,
            dtype=dtype,
        ).reshape(-1, 6),
        linear_bond_index=_index_tensor(linear_bond_index, 3, device),
        chiral_index=_index_tensor(chiral_index, 4, device),
        chiral_orientation=torch.tensor(chiral_orientation, device=device, dtype=dtype),
        stereo_bond_index=_index_tensor(stereo_bond_index, 4, device),
        stereo_bond_orientation=torch.tensor(stereo_bond_orientation, device=device, dtype=dtype),
        planar_improper_index=_index_tensor(planar_improper_index, 4, device),
    )


def _index_tensor(
    rows: Sequence[tuple[int, ...]],
    width: int,
    device: torch.device,
) -> Tensor:
    if not rows:
        return torch.empty((width, 0), device=device, dtype=torch.long)
    return torch.tensor(rows, device=device, dtype=torch.long).T.contiguous()


def _mapped_index(
    local_index: tuple[int, ...],
    local_to_global: dict[int, int],
) -> tuple[int, ...] | None:
    if not all(index in local_to_global for index in local_index):
        return None
    return tuple(local_to_global[index] for index in local_index)


def _component_geometry(component: ChemicalComponent) -> _ProtenixComponentGeometry:
    r"""
    Extract the released TFG constraints for one non-standard component.
    """

    from itertools import combinations

    from rdkit import Chem
    from rdkit.Chem.rdchem import BondStereo
    from rdkit.Chem.rdDistGeom import GetExperimentalTorsions, GetMoleculeBoundsMatrix
    from rdkit.Chem.rdMolTransforms import GetDihedralRad

    molecule = _rdkit_component(component)
    bounds = GetMoleculeBoundsMatrix(molecule)
    bonds = {tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))) for bond in molecule.GetBonds()}
    neighbors = {
        atom.GetIdx(): tuple(neighbor.GetIdx() for neighbor in atom.GetNeighbors()) for atom in molecule.GetAtoms()
    }
    angle_pairs = {
        tuple(sorted((first, second)))
        for center_neighbors in neighbors.values()
        for first, second in combinations(sorted(center_neighbors), 2)
    }
    pair_index: list[tuple[int, int]] = []
    pair_lower_bound: list[float] = []
    pair_upper_bound: list[float] = []
    pair_is_bond: list[bool] = []
    pair_is_angle: list[bool] = []
    for first in range(molecule.GetNumAtoms() - 1):
        for second in range(first + 1, molecule.GetNumAtoms()):
            pair_index.append((first, second))
            pair_upper_bound.append(float(bounds[first, second]))
            pair_lower_bound.append(float(bounds[second, first]))
            pair_is_bond.append((first, second) in bonds)
            pair_is_angle.append((first, second) in angle_pairs)

    experimental_torsion_index: list[tuple[int, int, int, int]] = []
    experimental_torsion_force_constant: list[tuple[float, ...]] = []
    experimental_torsion_sign: list[tuple[float, ...]] = []
    marked_bonds: set[tuple[int, int]] = set()
    for torsion in GetExperimentalTorsions(molecule, useSmallRingTorsions=True):
        torsion_index = cast(tuple[int, int, int, int], tuple(torsion["atomIndices"]))
        experimental_torsion_index.append(torsion_index)
        experimental_torsion_force_constant.append(tuple(float(value) for value in torsion["V"]))
        experimental_torsion_sign.append(tuple(float(value) for value in torsion["signs"]))
        marked_bonds.add(
            (
                min(torsion_index[1], torsion_index[2]),
                max(torsion_index[1], torsion_index[2]),
            )
        )
    for ring in molecule.GetRingInfo().AtomRings():
        if not 3 < len(ring) < 7:
            continue
        for start in range(len(ring)):
            ring_index = cast(
                tuple[int, int, int, int],
                tuple(ring[(start + offset) % len(ring)] for offset in range(4)),
            )
            central_bond = (
                min(ring_index[1], ring_index[2]),
                max(ring_index[1], ring_index[2]),
            )
            if central_bond in marked_bonds:
                continue
            if all(
                molecule.GetAtomWithIdx(atom_index).GetHybridization() == Chem.HybridizationType.SP2
                for atom_index in ring_index
            ):
                experimental_torsion_index.append(ring_index)
                experimental_torsion_force_constant.append((0.0, 100.0, 0.0, 0.0, 0.0, 0.0))
                experimental_torsion_sign.append((1.0, -1.0, 1.0, 1.0, 1.0, 1.0))
                marked_bonds.add(central_bond)

    conformer = molecule.GetConformer(0)
    chiral_index: list[tuple[int, int, int, int]] = []
    chiral_orientation: list[float] = []
    for atom in molecule.GetAtoms():
        if atom.GetChiralTag() not in {
            Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
            Chem.ChiralType.CHI_TETRAHEDRAL_CW,
        }:
            continue
        atom_neighbors = sorted(neighbor.GetIdx() for neighbor in atom.GetNeighbors())
        if not 3 <= len(atom_neighbors) <= 4:
            continue
        for selected in combinations(atom_neighbors, 3):
            index = (*selected, atom.GetIdx())
            chiral_index.append(index)
            chiral_orientation.append(1.0 if GetDihedralRad(conformer, *index) >= 0.0 else -1.0)

    linear_bond_index: list[tuple[int, int, int]] = []
    for bond in molecule.GetBonds():
        if (
            bond.GetBondType() != Chem.BondType.TRIPLE
            or bond.GetIsAromatic()
            or bond.GetBeginAtom().GetIsAromatic()
            or bond.GetEndAtom().GetIsAromatic()
            or bond.GetBeginAtom().GetHybridization() != Chem.HybridizationType.SP
            or bond.GetEndAtom().GetHybridization() != Chem.HybridizationType.SP
        ):
            continue
        first = bond.GetBeginAtomIdx()
        second = bond.GetEndAtomIdx()
        linear_bond_index.extend(
            (neighbor.GetIdx(), first, second)
            for neighbor in bond.GetBeginAtom().GetNeighbors()
            if neighbor.GetIdx() != second
        )
        linear_bond_index.extend(
            (first, second, neighbor.GetIdx())
            for neighbor in bond.GetEndAtom().GetNeighbors()
            if neighbor.GetIdx() != first
        )

    stereo_bond_index: list[tuple[int, int, int, int]] = []
    stereo_bond_orientation: list[float] = []
    for bond in molecule.GetBonds():
        if bond.GetStereo() not in {BondStereo.STEREOE, BondStereo.STEREOZ}:
            continue
        first = bond.GetBeginAtomIdx()
        second = bond.GetEndAtomIdx()
        first_neighbors = sorted(
            neighbor.GetIdx() for neighbor in bond.GetBeginAtom().GetNeighbors() if neighbor.GetIdx() != second
        )
        second_neighbors = sorted(
            neighbor.GetIdx() for neighbor in bond.GetEndAtom().GetNeighbors() if neighbor.GetIdx() != first
        )
        if not first_neighbors or not second_neighbors:
            continue
        indices = [(first_neighbors[0], first, second, second_neighbors[0])]
        if len(first_neighbors) == 2 and len(second_neighbors) == 2:
            indices.append((first_neighbors[1], first, second, second_neighbors[1]))
        for index in indices:
            stereo_bond_index.append(index)
            stereo_bond_orientation.append(0.0 if abs(GetDihedralRad(conformer, *index)) < math.pi / 2 else 1.0)

    planar_improper_index: list[tuple[int, int, int, int]] = []
    for atom in molecule.GetAtoms():
        if atom.GetSymbol() not in {"C", "N", "O"} or atom.GetHybridization() != Chem.HybridizationType.SP2:
            continue
        atom_neighbors = sorted(neighbor.GetIdx() for neighbor in atom.GetNeighbors())
        if len(atom_neighbors) != 3:
            continue
        first, second, third = atom_neighbors
        center = atom.GetIdx()
        planar_improper_index.extend(
            (
                (first, second, center, third),
                (third, first, center, second),
                (second, third, center, first),
            )
        )

    return _ProtenixComponentGeometry(
        pair_index=tuple(pair_index),
        pair_lower_bound=tuple(pair_lower_bound),
        pair_upper_bound=tuple(pair_upper_bound),
        pair_is_bond=tuple(pair_is_bond),
        pair_is_angle=tuple(pair_is_angle),
        experimental_torsion_index=tuple(experimental_torsion_index),
        experimental_torsion_force_constant=tuple(experimental_torsion_force_constant),
        experimental_torsion_sign=tuple(experimental_torsion_sign),
        linear_bond_index=tuple(linear_bond_index),
        chiral_index=tuple(chiral_index),
        chiral_orientation=tuple(chiral_orientation),
        stereo_bond_index=tuple(stereo_bond_index),
        stereo_bond_orientation=tuple(stereo_bond_orientation),
        planar_improper_index=tuple(planar_improper_index),
    )


def _rdkit_component(component: ChemicalComponent) -> Mol:
    r"""
    Materialize one canonical component as an RDKit graph with its reference conformer.
    """

    from rdkit import Chem
    from rdkit.Geometry import Point3D

    if component.bonds is None:
        raise ValueError(f"TFG requires known connectivity for component {component.component_id!r}")
    if len(component.reference_coordinates) != len(component.atoms):
        raise ValueError(f"TFG requires aligned reference coordinates for component {component.component_id!r}")

    editable = Chem.RWMol()
    atom_index: dict[str, int] = {}
    atom_stereo: dict[int, str] = {}
    for atom in component.atoms:
        value = Chem.Atom(atom.element)
        if atom.isotope is not None:
            value.SetIsotope(atom.isotope)
        if atom.formal_charge is not None:
            value.SetFormalCharge(atom.formal_charge)
        if atom.radical_electrons is not None:
            value.SetNumRadicalElectrons(atom.radical_electrons)
        if atom.explicit_hydrogens is not None:
            value.SetNumExplicitHs(atom.explicit_hydrogens)
        if atom.implicit_hydrogens_allowed is not None:
            value.SetNoImplicit(not atom.implicit_hydrogens_allowed)
        if atom.aromatic is not None:
            value.SetIsAromatic(atom.aromatic)
        index = editable.AddAtom(value)
        atom_index[atom.atom_id] = index
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
    for bond in sorted(component.bonds, key=lambda value: (value.atom_a, value.atom_b)):
        editable.AddBond(atom_index[bond.atom_a], atom_index[bond.atom_b], bond_types[bond.order])
        rdkit_bond = editable.GetBondWithIdx(editable.GetNumBonds() - 1)
        rdkit_bond.SetIsAromatic(bond.aromatic)
        if bond.stereo is not None:
            stereo_atoms = (
                None
                if bond.stereo_atoms is None
                else (atom_index[bond.stereo_atoms[0]], atom_index[bond.stereo_atoms[1]])
            )
            bond_stereo.append((rdkit_bond.GetIdx(), bond.stereo, stereo_atoms))

    molecule = editable.GetMol()
    Chem.SanitizeMol(molecule)
    conformer = Chem.Conformer(len(component.atoms))
    for index, coordinate in enumerate(component.reference_coordinates):
        x, y, z = (0.0, 0.0, 0.0) if coordinate is None else coordinate
        conformer.SetAtomPosition(index, Point3D(x, y, z))
    molecule.AddConformer(conformer, assignId=True)

    for index in atom_stereo:
        molecule.GetAtomWithIdx(index).SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
    Chem.AssignStereochemistry(molecule, force=True, cleanIt=True)
    for index, desired in atom_stereo.items():
        atom = molecule.GetAtomWithIdx(index)
        assigned = atom.GetProp("_CIPCode").upper() if atom.HasProp("_CIPCode") else ""
        if assigned != desired:
            atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
    Chem.AssignStereochemistry(molecule, force=True, cleanIt=True)

    stereo_types = {
        "E": Chem.BondStereo.STEREOE,
        "TRANS": Chem.BondStereo.STEREOE,
        "Z": Chem.BondStereo.STEREOZ,
        "CIS": Chem.BondStereo.STEREOZ,
    }
    for bond_index, stereo, stereo_atoms in bond_stereo:
        bond = molecule.GetBondWithIdx(bond_index)
        if stereo_atoms is None:
            first = bond.GetBeginAtomIdx()
            second = bond.GetEndAtomIdx()
            first_neighbors = [
                atom for atom in molecule.GetAtomWithIdx(first).GetNeighbors() if atom.GetIdx() != second
            ]
            second_neighbors = [
                atom for atom in molecule.GetAtomWithIdx(second).GetNeighbors() if atom.GetIdx() != first
            ]
            stereo_atoms = (
                max(first_neighbors, key=lambda atom: int(atom.GetProp("_CIPRank"))).GetIdx(),
                max(second_neighbors, key=lambda atom: int(atom.GetProp("_CIPRank"))).GetIdx(),
            )
        bond.SetStereoAtoms(*stereo_atoms)
        bond.SetStereo(stereo_types[stereo.upper()])
    Chem.AssignStereochemistry(molecule, force=True, cleanIt=True)
    return molecule


def _vdw_radii(
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    from rdkit.Chem import GetPeriodicTable

    periodic_table = GetPeriodicTable()
    return torch.tensor(
        [periodic_table.GetRvdw(atomic_number) for atomic_number in range(1, 119)],
        device=device,
        dtype=dtype,
    )


def _esm_features(
    system: MolecularSystem,
    tokens: list[dict[str, object]],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    r"""
    Encode protein entities with the released ESM2 vocabulary and map model tokens to residues.
    """

    protein_entity_ids: list[str] = []
    seen_entities: set[str] = set()
    for entity_id in system.instances.values():
        if entity_id in seen_entities:
            continue
        seen_entities.add(entity_id)
        entity = system.entities[entity_id]
        if isinstance(entity, PolymerSequence) and entity.polymer_type == "protein":
            protein_entity_ids.append(entity_id)
    protein_entities = tuple(cast(PolymerSequence, system.entities[entity_id]) for entity_id in protein_entity_ids)
    sequence_length = max((len(entity.components) + 2 for entity in protein_entities), default=0)
    esm_tokens = torch.full(
        (len(protein_entities), sequence_length),
        _ESM_PAD_TOKEN,
        dtype=torch.long,
    )
    esm_mask = torch.zeros_like(esm_tokens, dtype=torch.bool)
    for sequence_index, entity in enumerate(protein_entities):
        residue_tokens = torch.tensor(
            [
                _ESM_TOKEN_BY_COMPONENT.get(component.component_id, _ESM_UNKNOWN_RESIDUE_TOKEN)
                for component in entity.components
            ],
            dtype=torch.long,
        )
        end = len(residue_tokens) + 1
        esm_tokens[sequence_index, 0] = _ESM_CLS_TOKEN
        esm_tokens[sequence_index, 1:end] = residue_tokens
        esm_tokens[sequence_index, end] = _ESM_EOS_TOKEN
        esm_mask[sequence_index, : end + 1] = True

    entity_to_sequence = {entity_id: index for index, entity_id in enumerate(protein_entity_ids)}
    token_to_esm_residue = torch.zeros(len(tokens), dtype=torch.long)
    token_has_esm = torch.zeros(len(tokens), dtype=torch.bool)
    for token_index, token in enumerate(tokens):
        esm_sequence_index = entity_to_sequence.get(cast(str, token["entity_id"]))
        monomer_index = cast(int | None, token["monomer_index"])
        if esm_sequence_index is None or monomer_index is None:
            continue
        token_to_esm_residue[token_index] = esm_sequence_index * sequence_length + monomer_index + 1
        token_has_esm[token_index] = True
    return esm_tokens, esm_mask, token_to_esm_residue, token_has_esm


def _model_atom_names(atom_sites: tuple[AtomSite, ...]) -> tuple[str, ...]:
    # Preserve canonical atom IDs; SMILES imports assign element-number IDs during construction.
    return tuple(site.atom_id for site in atom_sites)


def _structure_condition_features(
    structure: Structure | None,
    tokens: list[dict[str, object]],
) -> tuple[Tensor, Tensor]:
    r"""
    Align resolved structure coordinates to the Protenix representative-token axis.
    """

    positions = torch.zeros(len(tokens), 3)
    mask = torch.zeros(len(tokens), dtype=torch.bool)
    if structure is None:
        return positions, mask

    coordinate_index = {
        site: index for index, site in enumerate(structure.sites) if bool(structure.coordinate_mask[index])
    }
    for token_index, token in enumerate(tokens):
        source_index = coordinate_index.get(cast(AtomSite, token["representative"]))
        if source_index is not None:
            positions[token_index] = torch.from_numpy(structure.coordinates[source_index].copy())
            mask[token_index] = True
    return positions, mask


def _reference_features(
    system: MolecularSystem,
    atom_sites: tuple[AtomSite, ...],
    site_to_atom: dict[AtomSite, int],
    *,
    augment: bool,
    generator: torch.Generator | None,
) -> tuple[Tensor, Tensor, Tensor]:
    positions = torch.zeros(len(atom_sites), 3)
    mask = torch.zeros(len(atom_sites), dtype=torch.bool)
    charges = torch.tensor([system.atom(site).formal_charge for site in atom_sites], dtype=torch.float32)
    # One reference space per polymer residue and per whole non-polymer instance, matching the
    # reference_space assignment in _tokenize so centring stays consistent with reference_space_id.
    grouped: dict[tuple[str, int | None], list[AtomSite]] = {}
    for site in atom_sites:
        entity = system.entities[system.instances[site.instance_id]]
        space = None if isinstance(entity, Molecule) else site.component_index
        grouped.setdefault((site.instance_id, space), []).append(site)

    for (instance_id, _space), sites in grouped.items():
        entity = system.entities[system.instances[instance_id]]
        checked: set[int] = set()
        for site in sites:
            component = entity.components[site.component_index]
            if site.component_index not in checked:
                checked.add(site.component_index)
                if not any(coordinate is not None for coordinate in component.reference_coordinates):
                    raise ValueError(
                        f"Protenix requires explicit reference_coordinates for {component.component_id!r}; "
                        "the Processor does not generate conformers implicitly"
                    )
            atom_by_name = {atom.atom_id: index for index, atom in enumerate(component.atoms)}
            global_index = site_to_atom[site]
            coordinate = component.reference_coordinates[atom_by_name[site.atom_id]]
            if coordinate is not None:
                positions[global_index] = torch.tensor(coordinate)
                mask[global_index] = True
        indices = torch.tensor([site_to_atom[site] for site in sites])
        positions[indices] -= positions[indices].mean(dim=0)
        if augment:
            translation = torch.empty(3).uniform_(-1.0, 1.0, generator=generator)
            rotation = uniform_random_rotation(dtype=positions.dtype, generator=generator)[0]
            positions[indices] = torch.matmul(
                positions[indices] + translation,
                rotation.transpose(-1, -2),
            )
    return positions, mask, charges


def _msa_polymer_type(system: MolecularSystem, entity_id: str) -> str | None:
    entity = system.entities.get(entity_id)
    return entity.polymer_type if isinstance(entity, PolymerSequence) else None


def _msa_symbol_residue_type(symbol: str, polymer_type: str | None) -> int:
    # Cells are UPPERCASE one-letter MSA symbols or ``-``; the query axis binds column i to
    # ``entity.components[i]`` so ``monomer_index`` is the column directly (query_to_entity is now the
    # identity map).
    if polymer_type == "rna":
        return _RNA_MSA_RESIDUE_TYPE.get(symbol, _RESIDUE_TYPE["N"])
    return _MSA_RESIDUE_TYPE.get(symbol, _UNKNOWN)


def _msa_descriptors(
    msa: MSA | MSACollection | None,
    system: MolecularSystem,
    *,
    max_rows: int,
) -> list[tuple[dict[tuple[str, str | None], tuple[int, bool]], bool]]:
    # Each descriptor maps an entity-broadcast or exact-instance column binding to
    # ``(source_row, is_paired)``. The bool is the query flag.
    if msa is None:
        return [({}, True)]
    if isinstance(msa, MSA):
        # A bare MSA is the entity's own deep alignment: enumerate its unpaired pool (row 0 is the
        # query), which the single-file reader packs as the full alignment.
        key = (msa.entity_id, None)
        return [({key: (row, False)}, row == 0) for row in range(min(len(msa.unpaired_sequences), max_rows))]

    row_pairing = msa.row_pairing
    row_is_paired = msa.row_is_paired
    bindings = msa_bindings(msa)
    plan_rows = int(row_pairing.shape[0])

    # RNA contributes only its unpaired pool. Protein paired rows keep the RNA query
    # sequence instead of treating RNA columns as another paired alignment.
    is_query = np.all(row_pairing == 0, axis=1)
    protein_columns = np.asarray(
        [_msa_polymer_type(system, value.entity_id) == "protein" for value, _target in bindings],
        dtype=np.bool_,
    )
    protein_paired = np.count_nonzero(
        (row_pairing != -1) & row_is_paired & protein_columns[None, :],
        axis=1,
    )
    paired = [row for row in range(plan_rows) if not is_query[row] and protein_paired[row] >= 2]
    paired_rows = set(paired)
    unpaired = [row for row in range(plan_rows) if not is_query[row] and row not in paired_rows]

    pair_limit = max_rows // 2
    ordered: list[int | None] = [None]
    ordered.extend(paired[: max(0, pair_limit - 1)])
    ordered.extend(unpaired[: max_rows - len(ordered)])
    ordered = ordered[:max_rows]

    descriptors: list[tuple[dict[tuple[str, str | None], tuple[int, bool]], bool]] = []
    for row in ordered:
        if row is None:
            descriptors.append(({}, True))
            continue
        mapping: dict[tuple[str, str | None], tuple[int, bool]] = {}
        for column, (value, target_instance_id) in enumerate(bindings):
            source = int(row_pairing[row, column])
            is_paired = bool(row_is_paired[row, column])
            if source == -1 or (is_paired and _msa_polymer_type(system, value.entity_id) == "rna"):
                continue
            mapping[(value.entity_id, target_instance_id)] = (source, is_paired)
        descriptors.append((mapping, False))
    return descriptors


def _msa_features(
    msa: MSA | MSACollection | None,
    tokens: list[dict[str, object]],
    system: MolecularSystem,
    *,
    max_rows: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    bindings = {} if msa is None else msa_binding_map(msa)
    descriptors = _msa_descriptors(msa, system, max_rows=max_rows)
    rows = torch.empty(len(descriptors), len(tokens), dtype=torch.long)

    for row_index, (descriptor, query) in enumerate(descriptors):
        for token_index, token in enumerate(tokens):
            entity_id = cast(str, token["entity_id"])
            instance_id = cast(str, token["instance_id"])
            monomer_index = cast(int | None, token["monomer_index"])
            polymer_type = _msa_polymer_type(system, entity_id)
            selected = select_msa_binding(bindings, entity_id, instance_id)
            if query or selected is None or polymer_type not in {"protein", "rna"}:
                rows[row_index, token_index] = cast(int, token["residue_type"])
            elif selected[0] not in descriptor:
                rows[row_index, token_index] = cast(int, token["residue_type"]) if polymer_type == "rna" else _GAP
            else:
                key, value = selected
                source_row, is_paired = descriptor[key]
                pool = value.paired_sequences if is_paired else value.unpaired_sequences
                rows[row_index, token_index] = _msa_symbol_residue_type(
                    pool[source_row][cast(int, monomer_index)],
                    polymer_type,
                )

    profile = torch.zeros(len(tokens), 32)
    deletion_mean = torch.zeros(len(tokens))
    for token_index, token in enumerate(tokens):
        entity_id = cast(str, token["entity_id"])
        instance_id = cast(str, token["instance_id"])
        monomer_index = cast(int | None, token["monomer_index"])
        polymer_type = _msa_polymer_type(system, entity_id)
        selected = select_msa_binding(bindings, entity_id, instance_id)
        if selected is None or polymer_type not in {"protein", "rna"}:
            profile[token_index, cast(int, token["residue_type"])] = 1
            continue
        _key, value = selected
        # The profile and deletion mean are the entity's own alignment statistics, taken over the full
        # unpaired pool, including the query row.
        column = cast(int, monomer_index)
        encoded = [_msa_symbol_residue_type(sequence[column], polymer_type) for sequence in value.unpaired_sequences]
        profile[token_index] = F.one_hot(torch.tensor(encoded), num_classes=32).float().mean(dim=0)
        deletion_mean[token_index] = torch.tensor(value.unpaired_deletion_counts[:, column], dtype=torch.float32).mean()
    return rows, torch.ones_like(rows, dtype=torch.float32), profile, deletion_mean


def _assembled_deletions(
    msa: MSA | MSACollection | None,
    tokens: list[dict[str, object]],
    system: MolecularSystem,
    *,
    max_rows: int,
) -> Tensor:
    bindings = {} if msa is None else msa_binding_map(msa)
    descriptors = _msa_descriptors(msa, system, max_rows=max_rows)
    deletions = torch.zeros(len(descriptors), len(tokens))
    for row_index, (descriptor, query) in enumerate(descriptors):
        if query:
            continue
        for token_index, token in enumerate(tokens):
            entity_id = cast(str, token["entity_id"])
            instance_id = cast(str, token["instance_id"])
            monomer_index = cast(int | None, token["monomer_index"])
            selected = select_msa_binding(bindings, entity_id, instance_id)
            if (
                selected is None
                or selected[0] not in descriptor
                or _msa_polymer_type(system, entity_id) not in {"protein", "rna"}
            ):
                continue
            key, value = selected
            source_row, is_paired = descriptor[key]
            pool = value.paired_deletion_counts if is_paired else value.unpaired_deletion_counts
            deletions[row_index, token_index] = float(pool[source_row, cast(int, monomer_index)])
    return deletions


def _template_to(
    features: ProtenixTemplateFeatures,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> ProtenixTemplateFeatures:
    return ProtenixTemplateFeatures(
        distogram=features.distogram.to(device=device, dtype=dtype).unsqueeze(0),
        pseudo_beta_mask=features.pseudo_beta_mask.to(device=device, dtype=dtype).unsqueeze(0),
        residue_type_ids=features.residue_type_ids.to(device=device).unsqueeze(0),
        unit_vector=features.unit_vector.to(device=device, dtype=dtype).unsqueeze(0),
        backbone_frame_mask=features.backbone_frame_mask.to(device=device, dtype=dtype).unsqueeze(0),
        batch_size=[1],
    )


def _template_features(
    templates: tuple[Template, ...],
    tokens: list[dict[str, object]],
    system: MolecularSystem,
    *,
    num_distogram_bins: int = 39,
) -> ProtenixTemplateFeatures:
    token_count = len(tokens)
    residue_tokens: dict[tuple[str, int], list[int]] = {}
    chain_tokens: dict[str, list[int]] = {}
    for token_index, token in enumerate(tokens):
        instance_id = cast(str, token["instance_id"])
        chain_tokens.setdefault(instance_id, []).append(token_index)
        monomer_index = cast(int | None, token["monomer_index"])
        if monomer_index is not None:
            residue_tokens.setdefault((instance_id, monomer_index), []).append(token_index)

    chain_templates: dict[str, list[tuple[Template, tuple[TemplateAlignment, ...]]]] = {}
    for instance_id, entity_id in system.instances.items():
        entity = system.entities[entity_id]
        if isinstance(entity, PolymerSequence) and entity.polymer_type == "protein":
            chain_templates[instance_id] = []
    for template in templates:
        alignments_by_chain: dict[str, list[TemplateAlignment]] = {}
        for alignment in template.alignments:
            for instance_id in _template_target_instances(alignment, system):
                alignments_by_chain.setdefault(instance_id, []).append(alignment)
        for instance_id, alignments in alignments_by_chain.items():
            if len(chain_templates[instance_id]) < 4:
                chain_templates[instance_id].append((template, tuple(alignments)))

    positions = torch.zeros(4, token_count, 4, 3)
    atom_mask = torch.zeros(4, token_count, 4)
    residue_type_ids = torch.zeros(4, token_count, dtype=torch.long)
    # An empty chain contributes one GAP slot; fixed-four padding leaves the remaining slots at residue ID zero.
    residue_type_ids[0] = _GAP

    template_coordinates: dict[int, dict[tuple[str, int, str], tuple[np.ndarray, bool]]] = {}
    for instance_id, selected_templates in chain_templates.items():
        target_tokens = chain_tokens[instance_id]
        for template_index, (template, template_alignments) in enumerate(selected_templates):
            residue_type_ids[template_index, target_tokens] = _GAP
            template_key = id(template)
            coordinates = template_coordinates.get(template_key)
            if coordinates is None:
                coordinates = _structure_atom_lookup(template.structure)
                template_coordinates[template_key] = coordinates
            for alignment in template_alignments:
                template_entity_id = template.structure.system.instances[alignment.template_instance_id]
                template_entity = cast(
                    PolymerSequence,
                    template.structure.system.entities[template_entity_id],
                )
                if template_entity.polymer_type != "protein":
                    raise ValueError("Protenix templates must align protein template instances")
                for target_component, template_component in alignment.residue_pairs:
                    targets = residue_tokens.get((instance_id, target_component))
                    if targets is None:
                        raise ValueError(
                            f"template alignment target {instance_id!r}:{target_component} "
                            "does not resolve to a token"
                        )
                    component = template_entity.components[template_component]
                    backbone, backbone_mask = _template_residue_backbone(
                        coordinates,
                        alignment.template_instance_id,
                        template_component,
                    )
                    residue_type = (
                        _RESIDUE_TYPE.get(component.component_id, _UNKNOWN)
                        if _is_standard(component, template_entity.polymer_type)
                        else _UNKNOWN
                    )
                    for token_index in targets:
                        positions[template_index, token_index] = backbone
                        atom_mask[template_index, token_index] = backbone_mask
                        residue_type_ids[template_index, token_index] = residue_type

    pseudo_beta, pseudo_beta_mask, unit_vector, backbone_mask = _template_backbone_features(
        positions,
        atom_mask,
        residue_type_ids,
    )
    pseudo_beta_mask_2d = pseudo_beta_mask[..., :, None] * pseudo_beta_mask[..., None, :]
    backbone_mask_2d = backbone_mask[..., :, None] * backbone_mask[..., None, :]
    return ProtenixTemplateFeatures(
        distogram=_template_distogram(
            pseudo_beta,
            pseudo_beta_mask_2d,
            num_bins=num_distogram_bins,
        ),
        pseudo_beta_mask=pseudo_beta_mask_2d,
        residue_type_ids=residue_type_ids,
        unit_vector=unit_vector * backbone_mask_2d[..., None],
        backbone_frame_mask=backbone_mask_2d,
        batch_size=[residue_type_ids.shape[0]],
    )


def _template_target_instances(
    alignment: TemplateAlignment,
    system: MolecularSystem,
) -> tuple[str, ...]:
    entity = system.entities.get(alignment.target_entity_id)
    if not isinstance(entity, PolymerSequence) or entity.polymer_type != "protein":
        raise ValueError(f"template target {alignment.target_entity_id!r} is not a protein entity")
    if alignment.target_instance_id is None:
        return tuple(
            instance_id
            for instance_id, entity_id in system.instances.items()
            if entity_id == alignment.target_entity_id
        )
    actual_entity_id = system.instances.get(alignment.target_instance_id)
    if actual_entity_id is None:
        raise ValueError(f"template alignment targets unknown instance {alignment.target_instance_id!r}")
    if actual_entity_id != alignment.target_entity_id:
        raise ValueError(
            f"template alignment instance {alignment.target_instance_id!r} belongs to entity "
            f"{actual_entity_id!r}, not {alignment.target_entity_id!r}"
        )
    return (alignment.target_instance_id,)


def _structure_atom_lookup(
    structure: Structure,
) -> dict[tuple[str, int, str], tuple[np.ndarray, bool]]:
    return {
        (site.instance_id, site.component_index, site.atom_id): (
            structure.coordinates[index],
            bool(structure.coordinate_mask[index]),
        )
        for index, site in enumerate(structure.sites)
    }


def _template_residue_backbone(
    coordinates: dict[tuple[str, int, str], tuple[np.ndarray, bool]],
    instance_id: str,
    component_index: int,
) -> tuple[Tensor, Tensor]:
    positions = torch.zeros(4, 3)
    mask = torch.zeros(4)
    for atom_index, atom_id in enumerate(("N", "CA", "C", "CB")):
        entry = coordinates.get((instance_id, component_index, atom_id))
        if entry is None:
            continue
        coordinate, present = entry
        if present:
            positions[atom_index] = torch.from_numpy(np.asarray(coordinate, dtype=np.float32))
            mask[atom_index] = 1
    return positions, mask


def _template_backbone_features(
    positions: Tensor,
    mask: Tensor,
    residue_type_ids: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    nitrogen, alpha, carbon, beta = positions.unbind(dim=-2)
    nitrogen_mask, alpha_mask, carbon_mask, beta_mask = mask.unbind(dim=-1)

    is_glycine = residue_type_ids == _RESIDUE_TYPE["GLY"]
    pseudo_beta = torch.where(is_glycine[..., None], alpha, beta)
    pseudo_beta_mask = torch.where(is_glycine, alpha_mask, beta_mask)

    x_axis = carbon - alpha
    x_axis = x_axis / (torch.sqrt((x_axis * x_axis).sum(dim=-1, keepdim=True)) + 1e-6)
    reference = nitrogen - alpha
    reference = reference - (reference * x_axis).sum(dim=-1, keepdim=True) * x_axis
    y_axis = reference / (torch.sqrt((reference * reference).sum(dim=-1, keepdim=True)) + 1e-6)
    z_axis = torch.cross(x_axis, y_axis, dim=-1)
    rotation = torch.stack((x_axis, y_axis, z_axis), dim=-1)

    difference = alpha[..., None, :, :] - alpha[..., :, None, :]
    local = torch.einsum("...ilk,...ijl->...ijk", rotation, difference)
    unit_vector = local / (torch.sqrt((local * local).sum(dim=-1, keepdim=True)) + 1e-6)
    backbone_mask = nitrogen_mask * alpha_mask * carbon_mask
    return pseudo_beta, pseudo_beta_mask, unit_vector, backbone_mask


def _template_distogram(
    pseudo_beta: Tensor,
    pseudo_beta_mask: Tensor,
    *,
    num_bins: int,
) -> Tensor:
    lower = torch.linspace(
        3.25,
        50.75,
        num_bins,
        device=pseudo_beta.device,
        dtype=pseudo_beta.dtype,
    ).square()
    upper = torch.cat((lower[1:], lower.new_tensor([1e8])))
    difference = pseudo_beta[..., :, None, :] - pseudo_beta[..., None, :, :]
    squared_distance = (difference * difference).sum(dim=-1, keepdim=True)
    distogram = ((squared_distance > lower) & (squared_distance < upper)).to(dtype=pseudo_beta.dtype)
    return distogram * pseudo_beta_mask[..., None]


def _constraint_features(
    *,
    tokens: list[dict[str, object]],
    atom_contacts: tuple[ProtenixAtomContact, ...],
    token_contacts: tuple[ProtenixTokenContact, ...],
    pocket: ProtenixPocket | None,
    substructure: Structure | None,
    substructure_coordinate_noise_scale: float,
    generator: torch.Generator | None,
    site_to_token: dict[AtomSite, int],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    r"""
    Build the four Protenix constraint pair-bias channels.

    The architecture maps atom contacts to channel 0 and token contacts to channel 1:

    * ``atom_contact`` -- an atom contact writes ``[max, 0]`` (channel 0 was the only
      trained atom column); symmetric.
    * ``token_contact`` -- a token contact writes ``[0, max]`` (channel 1); symmetric.
    * ``pocket`` -- a directed binder->residue fill of ``max`` on channel 0.
    * ``substructure`` -- distances among resolved token-center coordinates are
      digitized into the released four-channel representation.

    Token- and pocket-residue endpoints that lower to several candidate tokens draw a
    single token from the shared advancing ``generator`` (this selection is open
    Processor policy, not a record field). Endpoints that lower to one token draw nothing.
    """

    count = len(tokens)
    pocket_feature = torch.zeros(count, count, 1)
    token_contact_feature = torch.zeros(count, count, 2)
    atom_contact_feature = torch.zeros(count, count, 2)
    substructure_feature = torch.zeros(count, count, 4)
    residue_tokens: dict[ComponentReference, list[int]] = {}
    instance_tokens: dict[str, list[int]] = {}
    for index, token in enumerate(tokens):
        instance_id = cast(str, token["instance_id"])
        instance_tokens.setdefault(instance_id, []).append(index)
        monomer_index = cast(int | None, token["monomer_index"])
        if monomer_index is not None:
            residue_tokens.setdefault((instance_id, monomer_index), []).append(index)

    def atom_token(site: AtomSite) -> int:
        try:
            return site_to_token[site]
        except KeyError:
            raise ValueError(
                f"Protenix constraint references atom {site!r}, which is not a model-visible token"
            ) from None

    def endpoint_token(endpoint: ComponentReference | AtomSite) -> int:
        if isinstance(endpoint, AtomSite):
            return atom_token(endpoint)
        candidates = residue_tokens.get(endpoint)
        if not candidates:
            raise ValueError(
                f"Protenix token contact references residue {endpoint!r}, which has no model-visible token"
            )
        return _draw_token(candidates, generator)

    for atom_contact in atom_contacts:
        first, second = atom_token(atom_contact.atom_a), atom_token(atom_contact.atom_b)
        _write_symmetric(atom_contact_feature, first, second, (atom_contact.max_distance, 0.0))

    for token_contact in token_contacts:
        first = endpoint_token(token_contact.endpoint_a)
        second = endpoint_token(token_contact.endpoint_b)
        _write_symmetric(token_contact_feature, first, second, (0.0, token_contact.max_distance))

    if pocket is not None:
        binder_tokens = instance_tokens.get(pocket.binder)
        if not binder_tokens:
            raise ValueError(f"Protenix pocket binder chain {pocket.binder!r} has no model-visible token")
        for residue in pocket.contact_residues:
            candidates = residue_tokens.get(residue)
            if not candidates:
                raise ValueError(f"Protenix pocket residue {residue!r} has no model-visible token")
            pocket_token = _draw_token(candidates, generator)
            for binder_token in binder_tokens:
                pocket_feature[binder_token, pocket_token, 0] = pocket.max_distance

    if substructure is not None:
        coordinate_index = {
            site: index for index, site in enumerate(substructure.sites) if bool(substructure.coordinate_mask[index])
        }
        selected_token: list[int] = []
        selected_coordinates: list[Tensor] = []
        for token_index, token in enumerate(tokens):
            center = cast(AtomSite, token["center"])
            source_index = coordinate_index.get(center)
            if source_index is not None:
                selected_token.append(token_index)
                selected_coordinates.append(torch.from_numpy(substructure.coordinates[source_index].copy()))
        if selected_token:
            coordinates = torch.stack(selected_coordinates)
            if substructure_coordinate_noise_scale != 0.0:
                coordinates = coordinates + substructure_coordinate_noise_scale * torch.empty_like(coordinates).normal_(
                    generator=generator
                )
            distance = torch.cdist(coordinates, coordinates)
            boundaries = distance.new_tensor((0.0, 4.0, 8.0, 16.0, float("inf")))
            distance_bin = torch.bucketize(distance, boundaries, right=True) - 1
            selected_index = torch.tensor(selected_token, dtype=torch.long)
            row = selected_index[:, None].expand_as(distance_bin)
            column = selected_index[None, :].expand_as(distance_bin)
            present = distance_bin > 0
            substructure_feature[
                row[present],
                column[present],
                distance_bin[present],
            ] = 1

    return pocket_feature, token_contact_feature, atom_contact_feature, substructure_feature


def _draw_token(candidates: list[int], generator: torch.Generator | None) -> int:
    if len(candidates) == 1:
        return candidates[0]
    choice = int(torch.randint(0, len(candidates), (1,), generator=generator).item())
    return candidates[choice]


def _write_symmetric(feature: Tensor, first: int, second: int, value: tuple[float, float]) -> None:
    channels = torch.tensor(value, dtype=feature.dtype)
    previous = feature[first, second]
    if bool((previous != 0).any()) and not torch.equal(previous, channels):
        raise ValueError("conflicting Protenix constraints target the same token pair")
    feature[first, second] = channels
    feature[second, first] = channels


def _token_bonds(system: MolecularSystem, site_to_token: dict[AtomSite, int], token_count: int) -> Tensor:
    feature = torch.zeros(token_count, token_count)
    for bond in system.iter_bonds():
        first, second = bond.site_a, bond.site_b
        if first not in site_to_token or second not in site_to_token:
            continue
        overlay = bond.origin not in ("component", "molecule", "polymer_rule")
        first_entity = system.entities[system.instances[first.instance_id]]
        second_entity = system.entities[system.instances[second.instance_id]]
        keep = isinstance(first_entity, Molecule) or isinstance(second_entity, Molecule)
        if isinstance(first_entity, PolymerSequence) and isinstance(second_entity, PolymerSequence):
            if first.instance_id == second.instance_id and first.component_index == second.component_index:
                component = first_entity.components[first.component_index]
                keep = not _is_standard(component, first_entity.polymer_type)
            elif overlay and (
                first.instance_id != second.instance_id or abs(first.component_index - second.component_index) > 1
            ):
                keep = True
            else:
                keep = False
        if keep:
            first_token, second_token = site_to_token[first], site_to_token[second]
            feature[first_token, second_token] = 1
            feature[second_token, first_token] = 1
    return feature
