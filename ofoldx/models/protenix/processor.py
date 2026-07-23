r"""Lower canonical biomolecular inputs to Protenix tensors."""

from __future__ import annotations

import math
from collections.abc import Sequence
from os import PathLike
from pathlib import Path
from typing import cast

import numpy as np
import torch
import torch.nn.functional as F
from multimolecule.io import read_fasta_records
from torch import Tensor

from ...data import ATOMIC_NUMBER_BY_ELEMENT
from ...data.components import get_component
from ...data.evidence import (
    MSA,
    MSACollection,
    Template,
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
from ...io.msa import read_msa
from ...io.structure import read_structure
from ...modules.geometry import uniform_random_rotation
from ...processor import Processor
from ...references import classify_reference
from .config import ProtenixConstraintProcessorConfig, ProtenixProcessorConfig
from .model import ProtenixAtomContact, ProtenixInput, ProtenixPocket, ProtenixTokenContact

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
_FASTA_SUFFIXES = frozenset((".fa", ".faa", ".fasta", ".fna", ".seq"))


class ProtenixProcessor(Processor):
    r"""Build atom, token, MSA, and guidance tensors for Protenix folding.

    Args:
        config: Feature limits and reference-position options. Defaults to
            ``ProtenixProcessorConfig()``.
        device: Device for produced tensors.
        dtype: Floating-point dtype for produced tensors.
    """

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
        input: str | PathLike[str] | PolymerSequence | Sequence[PolymerSequence] | MolecularSystem | Structure,
        *,
        msa: MSA | MSACollection | str | PathLike[str] | None = None,
        templates: Sequence[Template | str | PathLike[str]] = (),
        generator: torch.Generator | None = None,
    ) -> ProtenixInput:
        r"""Lower an unconstrained assembly and its evidence to ``ProtenixInput``.

        Args:
            input: Inline protein sequence, FASTA or structure path, one or more
                ``PolymerSequence`` values, ``MolecularSystem``, or ``Structure``. A structure
                contributes topology, not observed coordinates.
            msa: Optional ``MSA``, ``MSACollection``, or alignment path.
            templates: Reserved template evidence; this architecture requires an empty sequence.
            generator: Optional PyTorch generator for reference-position augmentation.

        Returns:
            A default-graph ``ProtenixInput`` containing model tensors and canonical system, atom,
            and token identities.

        Notes:
            Atom-aligned tensors follow ``atom_sites``; token-aligned tensors follow
            ``token_sites``.
        """

        return self._lower(input, msa=msa, templates=templates, generator=generator)

    def _lower(
        self,
        input: str | PathLike[str] | PolymerSequence | Sequence[PolymerSequence] | MolecularSystem | Structure,
        *,
        msa: MSA | MSACollection | str | PathLike[str] | None,
        templates: Sequence[Template | str | PathLike[str]],
        atom_contacts: Sequence[ProtenixAtomContact] = (),
        token_contacts: Sequence[ProtenixTokenContact] = (),
        pocket: ProtenixPocket | None = None,
        generator: torch.Generator | None = None,
    ) -> ProtenixInput:
        r"""Shared lowering body. Constraint channels are empty unless a subclass supplies them."""

        template_values = tuple(templates)
        if template_values:
            raise ValueError("Protenix does not enable template blocks")
        atom_contact_values = tuple(atom_contacts)
        token_contact_values = tuple(token_contacts)
        system = _normalize_input(input)
        normalized_msa = _normalize_msa(msa, system)
        # Only atom contacts force per-atom token expansion (their endpoints must become atom-level
        # tokens); token contacts and pockets consume whichever tokens the residues already lower to.
        expanded_residues = _atom_constraint_residues(atom_contact_values, system)
        tokens = _tokenize(system, expanded_residues)
        return _features(
            system,
            tokens=tokens,
            msa=normalized_msa,
            atom_contacts=atom_contact_values,
            token_contacts=token_contact_values,
            pocket=pocket,
            generator=generator,
            max_msa_rows=self.config.max_msa_rows,
            augment_reference_positions=self.config.augment_reference_positions,
            device=self.device,
            dtype=self.dtype,
        )


class ProtenixConstraintProcessor(ProtenixProcessor):
    r"""Lower atom contacts, token contacts, and one pocket into Protenix constraints.

    Its public call adds constraint arguments that :class:`ProtenixProcessor` deliberately
    omits.

    Args:
        config: Constraint feature limits and reference-position options. Defaults to
            ``ProtenixConstraintProcessorConfig()``.
        device: Device for produced tensors.
        dtype: Floating-point dtype for produced tensors.
    """

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
        input: str | PathLike[str] | PolymerSequence | Sequence[PolymerSequence] | MolecularSystem | Structure,
        *,
        msa: MSA | MSACollection | str | PathLike[str] | None = None,
        templates: Sequence[Template | str | PathLike[str]] = (),
        atom_contacts: Sequence[ProtenixAtomContact] = (),
        token_contacts: Sequence[ProtenixTokenContact] = (),
        pocket: ProtenixPocket | None = None,
        generator: torch.Generator | None = None,
    ) -> ProtenixInput:
        r"""Lower an assembly, evidence, and constraints to ``ProtenixInput``.

        Args:
            input: Inline protein sequence, FASTA or structure path, one or more
                ``PolymerSequence`` values, ``MolecularSystem``, or ``Structure``. A structure
                contributes topology, not observed coordinates.
            msa: Optional ``MSA``, ``MSACollection``, or alignment path.
            templates: Reserved template evidence; this architecture requires an empty sequence.
            atom_contacts: Upper-bound constraints between exact atom pairs.
            token_contacts: Upper-bound constraints between token-level endpoints.
            pocket: Optional directed binder-to-residue pocket constraint.
            generator: Optional PyTorch generator for reference-position augmentation and
                multi-token constraint selection.

        Returns:
            A constraint-graph ``ProtenixInput`` containing model tensors and canonical system,
            atom, and token identities.

        Notes:
            Atom-aligned tensors follow ``atom_sites``; token-aligned and constraint tensors follow
            ``token_sites``.
        """

        return self._lower(
            input,
            msa=msa,
            templates=templates,
            atom_contacts=atom_contacts,
            token_contacts=token_contacts,
            pocket=pocket,
            generator=generator,
        )


def _normalize_input(
    input: str | PathLike[str] | PolymerSequence | Sequence[PolymerSequence] | MolecularSystem | Structure,
) -> MolecularSystem:
    # A Structure input contributes only its topology: v0.5.3 substructure conditioning is always the
    # all-zero grid, so observed coordinates are not lowered into the constraint channels.
    if isinstance(input, Structure):
        return input.system
    if isinstance(input, MolecularSystem):
        return input
    if isinstance(input, str | PathLike):
        classified = classify_reference(input)
        if isinstance(classified, Path):
            if any(suffix.lower() in _FASTA_SUFFIXES for suffix in classified.suffixes):
                fasta_sequences = tuple(
                    (
                        record.id or f"chain_{index + 1}",
                        PolymerSequence.from_one_letter(record.sequence.strip().upper()),
                    )
                    for index, record in enumerate(read_fasta_records(classified))
                )
                return _sequence_system(fasta_sequences)
            return read_structure(classified).system
        sequence = PolymerSequence.from_one_letter(classified.strip().upper())
        return _sequence_system((("protein", sequence),))
    if isinstance(input, PolymerSequence):
        return _sequence_system((("A", input),))
    if isinstance(input, Sequence):
        sequence_values = tuple(cast(Sequence[object], input))
        if not sequence_values or not all(isinstance(sequence, PolymerSequence) for sequence in sequence_values):
            raise TypeError("Protenix sequence collections must contain PolymerSequence values")
        polymer_sequences = cast(tuple[PolymerSequence, ...], sequence_values)
        return _sequence_system(
            tuple((f"chain_{index + 1}", sequence) for index, sequence in enumerate(polymer_sequences))
        )
    raise TypeError(
        "Protenix input must be an inline sequence, FASTA/structure path, PolymerSequence, "
        "sequence of PolymerSequence values, MolecularSystem, or Structure"
    )


def _sequence_system(named: tuple[tuple[str, PolymerSequence], ...]) -> MolecularSystem:
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
    msa: MSA | MSACollection | str | PathLike[str] | None,
    system: MolecularSystem,
) -> MSA | MSACollection | None:
    if msa is None:
        return None
    if isinstance(msa, MSA | MSACollection):
        for _value, _target_instance_id, entity in resolve_msa_bindings(msa, system):
            if entity.polymer_type != "protein":
                raise ValueError("Protenix accepts supplied MSA evidence only for proteins")
        return msa

    path = Path(msa).expanduser()
    proteins = [
        entity_id
        for entity_id, entity in system.entities.items()
        if isinstance(entity, PolymerSequence) and entity.polymer_type == "protein"
    ]
    if len(proteins) != 1:
        raise ValueError("an aligned FASTA MSA is unambiguous only for one protein entity")
    return read_msa(path, entity_id=proteins[0])


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
    atom_contacts: tuple[ProtenixAtomContact, ...],
    token_contacts: tuple[ProtenixTokenContact, ...],
    pocket: ProtenixPocket | None,
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

    pocket_feature, token_contact_feature, atom_contact_feature, substructure_feature = _constraint_features(
        tokens=tokens,
        atom_contacts=atom_contacts,
        token_contacts=token_contacts,
        pocket=pocket,
        generator=generator,
        site_to_token=site_to_token,
    )
    token_bonds = _token_bonds(system, site_to_token, len(tokens))
    token_to_rep_atom = torch.tensor(
        [site_to_atom[cast(AtomSite, token["representative"])] for token in tokens], dtype=torch.long
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
        msa=integer(_msa_rows),
        has_deletion=floating(has_deletion),
        deletion_value=floating(deletion_value),
        pocket=floating(pocket_feature),
        token_contact=floating(token_contact_feature),
        atom_contact=floating(atom_contact_feature),
        substructure=floating(substructure_feature),
        system=system,
        atom_sites=atom_sites,
        token_sites=tuple(cast(AtomSite | ComponentReference, token["site"]) for token in tokens),
    )


def _model_atom_names(atom_sites: tuple[AtomSite, ...]) -> tuple[str, ...]:
    # Preserve canonical atom IDs; SMILES imports assign element-number IDs during construction.
    return tuple(site.atom_id for site in atom_sites)


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


def _msa_symbol_residue_type(symbol: str) -> int:
    # Cells are UPPERCASE one-letter MSA symbols or ``-``; the query axis binds column i to
    # ``entity.components[i]`` so ``monomer_index`` is the column directly (query_to_entity is now the
    # identity map). Standard residues resolve to their residue-type index, ``-`` to the gap class, and
    # every other symbol (ambiguity codes) to UNK.
    return _MSA_RESIDUE_TYPE.get(symbol, _UNKNOWN)


def _msa_descriptors(
    msa: MSA | MSACollection | None,
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

    # A plan row is the query when every column selects source row 0 (no gaps); paired when >= 2
    # columns are populated and unpaired otherwise.
    is_query = np.all(row_pairing == 0, axis=1)
    populated = np.count_nonzero(row_pairing != -1, axis=1)
    paired = [row for row in range(plan_rows) if not is_query[row] and populated[row] >= 2]
    unpaired = [row for row in range(plan_rows) if not is_query[row] and populated[row] < 2]

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
            if source == -1:
                continue
            mapping[(value.entity_id, target_instance_id)] = (source, bool(row_is_paired[row, column]))
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
    descriptors = _msa_descriptors(msa, max_rows=max_rows)
    rows = torch.empty(len(descriptors), len(tokens), dtype=torch.long)

    for row_index, (descriptor, query) in enumerate(descriptors):
        for token_index, token in enumerate(tokens):
            entity_id = cast(str, token["entity_id"])
            instance_id = cast(str, token["instance_id"])
            monomer_index = cast(int | None, token["monomer_index"])
            selected = select_msa_binding(bindings, entity_id, instance_id)
            if query or selected is None or _msa_polymer_type(system, entity_id) != "protein":
                rows[row_index, token_index] = cast(int, token["residue_type"])
            elif selected[0] not in descriptor:
                rows[row_index, token_index] = _GAP
            else:
                key, value = selected
                source_row, is_paired = descriptor[key]
                pool = value.paired_sequences if is_paired else value.unpaired_sequences
                rows[row_index, token_index] = _msa_symbol_residue_type(pool[source_row][cast(int, monomer_index)])

    profile = torch.zeros(len(tokens), 32)
    deletion_mean = torch.zeros(len(tokens))
    for token_index, token in enumerate(tokens):
        entity_id = cast(str, token["entity_id"])
        instance_id = cast(str, token["instance_id"])
        monomer_index = cast(int | None, token["monomer_index"])
        selected = select_msa_binding(bindings, entity_id, instance_id)
        if selected is None or _msa_polymer_type(system, entity_id) != "protein":
            profile[token_index, cast(int, token["residue_type"])] = 1
            continue
        _key, value = selected
        # The profile and deletion mean are the entity's own alignment statistics, taken over the full
        # unpaired pool, including the query row.
        column = cast(int, monomer_index)
        encoded = [_msa_symbol_residue_type(sequence[column]) for sequence in value.unpaired_sequences]
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
    descriptors = _msa_descriptors(msa, max_rows=max_rows)
    deletions = torch.zeros(len(descriptors), len(tokens))
    for row_index, (descriptor, query) in enumerate(descriptors):
        if query:
            continue
        for token_index, token in enumerate(tokens):
            entity_id = cast(str, token["entity_id"])
            instance_id = cast(str, token["instance_id"])
            monomer_index = cast(int | None, token["monomer_index"])
            selected = select_msa_binding(bindings, entity_id, instance_id)
            if selected is None or selected[0] not in descriptor or _msa_polymer_type(system, entity_id) != "protein":
                continue
            key, value = selected
            source_row, is_paired = descriptor[key]
            pool = value.paired_deletion_counts if is_paired else value.unpaired_deletion_counts
            deletions[row_index, token_index] = float(pool[source_row, cast(int, monomer_index)])
    return deletions


def _constraint_features(
    *,
    tokens: list[dict[str, object]],
    atom_contacts: tuple[ProtenixAtomContact, ...],
    token_contacts: tuple[ProtenixTokenContact, ...],
    pocket: ProtenixPocket | None,
    generator: torch.Generator | None,
    site_to_token: dict[AtomSite, int],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    r"""Build the four Protenix constraint pair-bias channels.

    The architecture maps atom contacts to channel 0 and token contacts to channel 1:

    * ``atom_contact`` -- an atom contact writes ``[max, 0]`` (channel 0 was the only
      trained atom column); symmetric.
    * ``token_contact`` -- a token contact writes ``[0, max]`` (channel 1); symmetric.
    * ``pocket`` -- a directed binder->residue fill of ``max`` on channel 0.
    * ``substructure`` -- always the all-zero grid; the embedder still runs so the pair
      baseline stays nonzero.

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
