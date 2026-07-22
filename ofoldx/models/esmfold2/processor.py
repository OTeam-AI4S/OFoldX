r"""Lower canonical molecular systems and MSA evidence to ESMFold2 tensors."""

from __future__ import annotations

import math
from collections.abc import Sequence
from os import PathLike
from pathlib import Path
from typing import cast

import numpy as np
import torch
from multimolecule import ProteinTokenizer
from multimolecule.io import read_fasta_records
from typing_extensions import Self

from ...data.evidence import MSA, MSACollection, msa_bindings, resolve_msa_bindings
from ...data.molecular import (
    AtomSite,
    MolecularSystem,
    Molecule,
    PolymerSequence,
    one_letter_to_component,
    restype_3to1,
)
from ...data.structure import Structure
from ...references import classify_reference
from .config import EsmFold2ExperimentalProcessorConfig, EsmFold2ProcessorConfig
from .model import EsmFold2Input, EsmFold2Layout

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
_PROTEIN_UNKNOWN = 22
_MSA_GAP = 1

# Binder-design prompts use amino-acid letters for fixed positions and ``#`` for
# mutable positions. Antibody CDRs use Chothia numbering.
_MUTABLE_TOKEN = "#"
_ANTIBODY_CDR_SCHEME = "chothia"


class EsmFold2Processor:
    r"""Lower complete canonical ESMFold2 inputs without changing their chemistry."""

    def __init__(
        self,
        config: EsmFold2ProcessorConfig | None = None,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.config = EsmFold2ProcessorConfig() if config is None else config
        self.device = torch.device("cpu" if device is None else device)
        self.dtype = torch.float32 if dtype is None else dtype
        self.tokenizer = ProteinTokenizer()

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Self:
        r"""Update output placement in place and return this Processor."""

        if device is not None:
            self.device = torch.device(device)
        if dtype is not None:
            self.dtype = dtype
        return self

    def __call__(
        self,
        input: str | PathLike[str] | PolymerSequence | Sequence[PolymerSequence] | MolecularSystem | Structure,
        *,
        msa: MSA | MSACollection | str | PathLike[str] | None = None,
    ) -> EsmFold2Input:
        r"""Lower one molecular system and optional MSA evidence."""

        if msa is not None and not self.config.msa_conditioning:
            raise ValueError("this ESMFold2 checkpoint is the single-sequence Fast variant")
        if isinstance(input, Structure):
            input = input.system
        system = input if isinstance(input, MolecularSystem) else molecular_system(input)
        normalized_msa = normalize_msa(msa, system)
        return prepare_esmfold2_features(
            system,
            msa=normalized_msa,
            tokenizer=self.tokenizer,
            device=self.device,
            dtype=self.dtype,
        )


class EsmFold2ExperimentalProcessor(EsmFold2Processor):
    r"""Lower target|binder complexes for the experimental recurrent ESMFold2 design graph.

    Paired with [`EsmFold2ExperimentalModel`][ofoldx.models.esmfold2.EsmFold2ExperimentalModel].
    It uses the same ESMFold2 token/atom lowering, while its independent config
    records whether the paired checkpoint accepts MSA evidence. The soft residue
    probabilities consumed by the differentiable trunk are assembled by the
    binder-design Pipeline because they depend on the per-step optimisation logits.
    """

    def __init__(
        self,
        config: EsmFold2ExperimentalProcessorConfig | None = None,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        config = EsmFold2ExperimentalProcessorConfig() if config is None else config
        super().__init__(
            EsmFold2ProcessorConfig({"msa_conditioning": config.msa_conditioning}),
            device=device,
            dtype=dtype,
        )
        self.config = config


def molecular_system(
    input: str | PathLike[str] | PolymerSequence | Sequence[PolymerSequence],
) -> MolecularSystem:
    r"""Construct the polymer-only convenience input accepted by the fold boundary."""

    named: list[tuple[str, PolymerSequence]]
    if isinstance(input, str | PathLike):
        classified = classify_reference(input)
        if isinstance(classified, Path):
            records = read_fasta_records(classified)
            named = [
                (
                    record.id or f"chain_{index + 1}",
                    PolymerSequence.from_one_letter(record.sequence.strip().upper()),
                )
                for index, record in enumerate(records)
            ]
        else:
            named = [("protein", PolymerSequence.from_one_letter(classified.strip().upper()))]
    elif isinstance(input, PolymerSequence):
        named = [("A", input)]
    elif isinstance(input, Sequence):
        sequences = tuple(input)
        if not sequences or not all(isinstance(sequence, PolymerSequence) for sequence in sequences):
            raise TypeError("ESMFold2 sequence collections must contain PolymerSequence values")
        named = [
            (_default_chain_id(index), cast(PolymerSequence, sequence)) for index, sequence in enumerate(sequences)
        ]
    else:
        raise TypeError(
            "ESMFold2 input must be a sequence string, FASTA path, PolymerSequence, "
            "sequence of PolymerSequence values, MolecularSystem, or Structure"
        )

    entities: dict[str, PolymerSequence | Molecule] = {}
    instances: dict[str, str] = {}
    for base_id, sequence in named:
        key = base_id
        suffix = 1
        while key in entities:
            suffix += 1
            key = f"{base_id}_{suffix}"
        entities[key] = sequence
        instances[key] = key
    return MolecularSystem(entities=entities, instances=instances)


def _default_chain_id(index: int) -> str:
    r"""Assign a default entity and instance key."""

    return chr(ord("A") + index) if index < 26 else f"chain_{index + 1}"


def normalize_msa(
    msa: MSA | MSACollection | str | PathLike[str] | None,
    system: MolecularSystem,
) -> MSA | MSACollection | None:
    r"""Bind MSA evidence to the exact polymer definitions in ``system``."""

    if msa is None:
        return None
    if isinstance(msa, MSA | MSACollection):
        for _value, _target_instance_id, entity in resolve_msa_bindings(msa, system):
            if entity.polymer_type != "protein":
                raise ValueError("released ESMFold2 consumes MSA evidence only for protein entities")
        return msa

    path = Path(msa).expanduser()
    proteins = [
        (entity_id, entity)
        for entity_id, entity in system.entities.items()
        if isinstance(entity, PolymerSequence) and entity.polymer_type == "protein"
    ]
    if len(proteins) != 1:
        raise ValueError("an aligned FASTA MSA is unambiguous only for one protein entity; pass MSACollection")
    records = read_fasta_records(path)
    if not records:
        raise ValueError("MSA FASTA contains no rows")
    entity_id, _ = proteins[0]
    a3m = ".a3m" in {suffix.lower() for suffix in path.suffixes}
    parsed = tuple(aligned_protein_row(record.sequence, a3m=a3m) for record in records)
    aligned_rows = tuple(row for row, _ in parsed)
    deletion_rows = tuple(deletions for _, deletions in parsed)
    width = len(aligned_rows[0])
    # The FASTA is a plain single-chain alignment: every row goes to the unpaired pool (row 0 is the
    # ungapped query), and the paired pool carries only the shared query row so both pools validate.
    return MSA(
        entity_id=entity_id,
        unpaired_sequences=aligned_rows,
        unpaired_deletion_counts=np.array(deletion_rows, dtype=np.uint32).reshape(len(aligned_rows), width),
        paired_sequences=(aligned_rows[0],),
        paired_deletion_counts=np.zeros((1, width), dtype=np.uint32),
    )


def aligned_protein_row(
    sequence: str,
    *,
    a3m: bool = False,
) -> tuple[str, tuple[int, ...]]:
    r"""Convert one aligned FASTA/A3M row to a query-width symbol string plus insertion counts."""

    symbols: list[str] = []
    deletion_counts: list[int] = []
    insertion_count = 0
    for symbol in sequence.strip():
        if a3m and (symbol.islower() or symbol == "."):
            insertion_count += 1
            continue
        symbols.append("-" if symbol in "-." else symbol.upper())
        deletion_counts.append(insertion_count)
        insertion_count = 0
    return "".join(symbols), tuple(deletion_counts)


def _msa_residue_type(symbol: str) -> int:
    r"""Map one uppercase MSA symbol or gap to a residue-type index."""

    if symbol == "-":
        return _MSA_GAP
    component_id = one_letter_to_component(symbol, "protein", policy="permissive")
    return _PROTEIN_RESIDUE_TYPES.get(component_id, _PROTEIN_UNKNOWN)


def prepare_esmfold2_features(
    system: MolecularSystem,
    *,
    msa: MSA | MSACollection | None,
    tokenizer: ProteinTokenizer,
    device: torch.device,
    dtype: torch.dtype,
) -> EsmFold2Input:
    r"""Build the token and atom layout from canonical identity and topology."""

    token_groups: list[tuple[AtomSite, ...]] = []
    token_sites: list[AtomSite | tuple[str, int]] = []
    token_instance_ids: list[str] = []
    token_entity_ids: list[str] = []
    token_monomer_indices: list[int | None] = []
    residue_indices: list[int] = []
    asym_ids: list[int] = []
    entity_ids: list[int] = []
    sym_ids: list[int] = []
    molecule_types: list[int] = []
    residue_types: list[int] = []
    reference_space_ids: list[int] = []

    available_sites = {site for site, atom in system.iter_atoms() if atom.element != "H"}
    entity_index_by_id: dict[str, int] = {}
    instance_count_by_entity: dict[str, int] = {}
    reference_space = 0

    for asym_id, (instance_id, entity_key) in enumerate(system.instances.items()):
        entity = system.entities[entity_key]
        entity_id = entity_index_by_id.setdefault(entity_key, len(entity_index_by_id) + 1)
        sym_id = instance_count_by_entity.get(entity_key, 0)
        instance_count_by_entity[entity_key] = sym_id + 1

        if isinstance(entity, PolymerSequence):
            molecule_type = _MOLECULE_TYPES[entity.polymer_type]
            standards = _STANDARD_COMPONENTS[entity.polymer_type]
            for component_index, component in enumerate(entity.components):
                component_id = component.component_id
                sites = tuple(
                    site
                    for atom in component.atoms
                    if atom.element != "H"
                    and atom.atom_id not in {"OXT", "OP3"}
                    and (site := AtomSite(instance_id, component_index, atom.atom_id)) in available_sites
                )
                if not sites:
                    raise ValueError(
                        f"ESMFold2 component {component_id!r} at {instance_id}:{component_index} "
                        "has no model-visible heavy atoms"
                    )
                is_standard = component_id in standards
                groups = (sites,) if is_standard else tuple((site,) for site in sites)
                if entity.polymer_type == "protein":
                    residue_type = _PROTEIN_RESIDUE_TYPES.get(component_id, _PROTEIN_UNKNOWN)
                elif entity.polymer_type == "rna":
                    residue_type = _RNA_RESIDUE_TYPES.get(component_id, _PROTEIN_UNKNOWN)
                else:
                    residue_type = _DNA_RESIDUE_TYPES.get(component_id, _PROTEIN_UNKNOWN)
                for group in groups:
                    token_groups.append(group)
                    token_sites.append((instance_id, component_index) if is_standard else group[0])
                    token_instance_ids.append(instance_id)
                    token_entity_ids.append(entity_key)
                    token_monomer_indices.append(component_index)
                    residue_indices.append(component_index)
                    asym_ids.append(asym_id)
                    entity_ids.append(entity_id)
                    sym_ids.append(sym_id)
                    molecule_types.append(molecule_type)
                    residue_types.append(residue_type)
                    reference_space_ids.extend([reference_space] * len(group))
                reference_space += 1
        else:
            sites = tuple(
                site
                for component_index, component in enumerate(entity.components)
                for atom in component.atoms
                if atom.element != "H"
                and (site := AtomSite(instance_id, component_index, atom.atom_id)) in available_sites
            )
            if not sites:
                raise ValueError(f"ESMFold2 molecule {instance_id!r} has no heavy atoms")
            for site in sites:
                token_groups.append((site,))
                token_sites.append(site)
                token_instance_ids.append(instance_id)
                token_entity_ids.append(entity_key)
                token_monomer_indices.append(None)
                residue_indices.append(0)
                asym_ids.append(asym_id)
                entity_ids.append(entity_id)
                sym_ids.append(sym_id)
                molecule_types.append(3)
                residue_types.append(_PROTEIN_UNKNOWN)
                reference_space_ids.append(reference_space)
            reference_space += 1

    atom_sites = tuple(site for group in token_groups for site in group)
    site_to_atom = {site: index for index, site in enumerate(atom_sites)}
    site_to_token = {site: token_index for token_index, group in enumerate(token_groups) for site in group}
    real_atom_count = len(atom_sites)
    padded_atom_count = max(32, math.ceil(real_atom_count / 32) * 32)

    optional_reference_coordinates = [reference_coordinate(system, site) for site in atom_sites]
    for space_id in set(reference_space_ids):
        indices = [index for index, value in enumerate(reference_space_ids) if value == space_id]
        if len(indices) > 1 and any(optional_reference_coordinates[index] is None for index in indices):
            raise ValueError(f"ESMFold2 reference conformer is incomplete for {atom_sites[indices[0]].instance_id!r}")
    reference_coordinates: list[tuple[float, float, float]] = []
    atomic_numbers: list[int] = []
    formal_charges: list[int] = []
    encoded_atom_names: list[list[int]] = []
    molecule_atom_counts: dict[tuple[str, str], int] = {}
    from rdkit.Chem import GetPeriodicTable

    periodic_table = GetPeriodicTable()
    atom_cursor = 0
    for group in token_groups:
        for site in group:
            coordinate = optional_reference_coordinates[atom_cursor]
            atom_cursor += 1
            atom = system.atom(site)
            reference_coordinates.append((0.0, 0.0, 0.0) if coordinate is None else coordinate)
            atomic_numbers.append(int(periodic_table.GetAtomicNumber(atom.element)))
            formal_charges.append(0 if atom.formal_charge is None else atom.formal_charge)
            entity = system.entities[system.instances[site.instance_id]]
            atom_name = site.atom_id
            if isinstance(entity, Molecule) and not valid_atom_name(atom_name):
                key = (site.instance_id, atom.element.upper())
                molecule_atom_counts[key] = molecule_atom_counts.get(key, 0) + 1
                atom_name = f"{atom.element.upper()}{molecule_atom_counts[key]}"
            if not valid_atom_name(atom_name):
                raise ValueError(f"ESMFold2 atom name cannot be encoded in four 6-bit characters: {atom_name!r}")
            encoded_atom_names.append(
                [ord(character) - 32 if character != " " else 0 for character in atom_name.ljust(4)]
            )

    reference_positions = torch.zeros(padded_atom_count, 3, dtype=torch.float32)
    ref_element = torch.zeros(padded_atom_count, dtype=torch.long)
    ref_charge = torch.zeros(padded_atom_count, dtype=torch.int8)
    ref_atom_name_chars = torch.zeros(padded_atom_count, 4, dtype=torch.long)
    ref_space_uid = torch.zeros(padded_atom_count, dtype=torch.long)
    atom_mask = torch.zeros(padded_atom_count, dtype=torch.bool)
    atom_to_token = torch.zeros(padded_atom_count, dtype=torch.long)
    reference_positions[:real_atom_count] = torch.tensor(reference_coordinates, dtype=torch.float32)
    ref_element[:real_atom_count] = torch.tensor(atomic_numbers, dtype=torch.long)
    ref_charge[:real_atom_count] = torch.tensor(formal_charges, dtype=torch.int8)
    ref_atom_name_chars[:real_atom_count] = torch.tensor(encoded_atom_names, dtype=torch.long)
    ref_space_uid[:real_atom_count] = torch.tensor(reference_space_ids, dtype=torch.long)
    atom_mask[:real_atom_count] = True
    atom_to_token[:real_atom_count] = torch.tensor(
        [token_index for token_index, group in enumerate(token_groups) for _ in group],
        dtype=torch.long,
    )

    token_count = len(token_groups)
    representative_atoms = [
        representative_atom(group, molecule_types[index], residue_types[index], site_to_atom)
        for index, group in enumerate(token_groups)
    ]
    token_bonds = token_bond_matrix(system, site_to_token, token_count)
    msa_features = msa_tensors(
        msa,
        token_entity_ids=token_entity_ids,
        token_instance_ids=token_instance_ids,
        token_monomer_indices=token_monomer_indices,
        residue_types=residue_types,
    )
    esmc_input_ids, esmc_attention_mask, token_to_esmc, token_has_esmc = esmc_tensors(
        system,
        token_instance_ids=token_instance_ids,
        token_monomer_indices=token_monomer_indices,
        molecule_types=molecule_types,
        tokenizer=tokenizer,
    )

    features = {
        "token_index": torch.arange(token_count, dtype=torch.long),
        "residue_index": torch.tensor(residue_indices, dtype=torch.long),
        "chain_index": torch.tensor(asym_ids, dtype=torch.long),
        "symmetry_index": torch.tensor(sym_ids, dtype=torch.long),
        "entity_index": torch.tensor(entity_ids, dtype=torch.long),
        "molecule_type": torch.tensor(molecule_types, dtype=torch.long),
        "residue_type": torch.tensor(residue_types, dtype=torch.long),
        "token_bonds": token_bonds,
        "token_mask": torch.ones(token_count, dtype=torch.bool),
        "reference_positions": reference_positions,
        "ref_element": ref_element,
        "ref_charge": ref_charge,
        "ref_atom_name_chars": ref_atom_name_chars,
        "ref_space_uid": ref_space_uid,
        "atom_mask": atom_mask,
        "atom_to_token": atom_to_token,
        "distogram_atom_idx": torch.tensor(representative_atoms, dtype=torch.long),
        "token_has_esmc": token_has_esmc,
        **msa_features,
    }
    placed = {
        name: tensor.to(device=device, dtype=dtype) if tensor.is_floating_point() else tensor.to(device=device)
        for name, tensor in features.items()
    }
    return EsmFold2Input(
        **{name: value.unsqueeze(0) for name, value in placed.items()},
        esmc_input_ids=esmc_input_ids.to(device=device),
        esmc_attention_mask=esmc_attention_mask.to(device=device),
        token_to_esmc=token_to_esmc.unsqueeze(0).to(device=device),
        layout=EsmFold2Layout(
            system=system,
            atom_sites=atom_sites,
            token_sites=tuple(token_sites),
            msa=msa,
        ),
    )


def reference_coordinate(
    system: MolecularSystem,
    site: AtomSite,
) -> tuple[float, float, float] | None:
    r"""Return the canonical atom-aligned reference coordinate for one site."""

    entity = system.entities[system.instances[site.instance_id]]
    component = entity.components[site.component_index]
    if not component.reference_coordinates:
        return None
    index = next(index for index, atom in enumerate(component.atoms) if atom.atom_id == site.atom_id)
    return component.reference_coordinates[index]


def valid_atom_name(atom_name: str) -> bool:
    r"""Return whether the four-character atom encoding is lossless."""

    return len(atom_name) <= 4 and all(32 <= ord(character) < 96 for character in atom_name)


def representative_atom(
    group: tuple[AtomSite, ...],
    molecule_type: int,
    residue_type: int,
    site_to_atom: dict[AtomSite, int],
) -> int:
    r"""Select the representative atom for one structure token."""

    by_name = {site.atom_id: site_to_atom[site] for site in group}
    fallback = site_to_atom[group[0]]
    if molecule_type == 0:
        return by_name.get("CB", by_name.get("CA", fallback))
    if molecule_type in (1, 2):
        if residue_type in (27, 32):
            return by_name.get("C1'", fallback)
        if residue_type in (23, 24, 28, 29):
            return by_name.get("C4", by_name.get("C1'", fallback))
        return by_name.get("C2", by_name.get("C1'", fallback))
    return fallback


def token_bond_matrix(
    system: MolecularSystem,
    site_to_token: dict[AtomSite, int],
    token_count: int,
) -> torch.Tensor:
    r"""Project canonical intrinsic, polymer, and assembly bonds onto token pairs."""

    edges: set[tuple[int, int]] = set()

    def add(site_a: AtomSite, site_b: AtomSite) -> None:
        token_a = site_to_token.get(site_a)
        token_b = site_to_token.get(site_b)
        if token_a is not None and token_b is not None and token_a != token_b:
            edges.add((min(token_a, token_b), max(token_a, token_b)))

    protein_standards = _STANDARD_COMPONENTS["protein"]
    for bond in system.iter_bonds():
        if bond.origin == "polymer_rule":
            # Encode an explicit token bond only for a protein backbone
            # link at a non-standard junction; standard peptide adjacency and every nucleic
            # backbone link stay implicit in the token/residue index and are skipped here.
            entity = system.entities[system.instances[bond.site_a.instance_id]]
            if isinstance(entity, PolymerSequence):
                left_id = entity.components[bond.site_a.component_index].component_id
                right_id = entity.components[bond.site_b.component_index].component_id
                if entity.polymer_type != "protein" or (left_id in protein_standards and right_id in protein_standards):
                    continue
        add(bond.site_a, bond.site_b)

    values = torch.zeros(token_count, token_count, 1, dtype=torch.float32)
    for left, right in edges:
        values[left, right, 0] = 1.0
        values[right, left, 0] = 1.0
    return values


def msa_tensors(
    msa: MSA | MSACollection | None,
    *,
    token_entity_ids: list[str],
    token_instance_ids: list[str],
    token_monomer_indices: list[int | None],
    residue_types: list[int],
) -> dict[str, torch.Tensor]:
    r"""Construct paired and unpaired MSA rows over structure tokens."""

    token_count = len(residue_types)
    if msa is None:
        tokens = torch.tensor([residue_types], dtype=torch.long)
        deletion_counts = torch.zeros(1, token_count, dtype=torch.float32)
    else:
        # The MSA holds no PolymerSequence: column ``c`` binds ``columns[c].entity_id`` to a system
        # entity, and query_to_entity is the identity map, so a token's monomer index is directly the
        # aligned-row column. A bare MSA is a single-chain plan over its unpaired pool; a collection
        # replays its dense row_pairing/row_is_paired plan across the paired and unpaired pools.
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
        rows: list[torch.Tensor] = []
        deletions: list[torch.Tensor] = []
        for selections in plan_rows:
            row = torch.full((token_count,), _MSA_GAP, dtype=torch.long)
            deletion_row = torch.zeros(token_count, dtype=torch.float32)
            for column_index, source_row, is_paired in selections:
                value = columns[column_index]
                target_instance_id = bindings[column_index][1]
                aligned = value.paired_sequences[source_row] if is_paired else value.unpaired_sequences[source_row]
                deletion_source = (value.paired_deletion_counts if is_paired else value.unpaired_deletion_counts)[
                    source_row
                ]
                for token_index, (entity_id, instance_id, monomer_index) in enumerate(
                    zip(token_entity_ids, token_instance_ids, token_monomer_indices)
                ):
                    if (
                        entity_id != value.entity_id
                        or (target_instance_id is not None and instance_id != target_instance_id)
                        or monomer_index is None
                    ):
                        continue
                    row[token_index] = _msa_residue_type(aligned[monomer_index])
                    deletion_row[token_index] = float(deletion_source[monomer_index])
            rows.append(row)
            deletions.append(deletion_row)
        tokens = torch.stack(rows)
        deletion_counts = torch.stack(deletions)
        tokens[0] = torch.tensor(residue_types, dtype=torch.long)

    has_deletion = deletion_counts > 0
    deletion_value = (math.pi / 2) * torch.atan(deletion_counts / 3)
    return {
        "msa": tokens,
        "has_deletion": has_deletion,
        "deletion_value": deletion_value,
        "deletion_mean": deletion_value.mean(dim=0),
        "msa_attention_mask": torch.ones_like(tokens, dtype=torch.bool),
    }


def esmc_tensors(
    system: MolecularSystem,
    *,
    token_instance_ids: list[str],
    token_monomer_indices: list[int | None],
    molecule_types: list[int],
    tokenizer: ProteinTokenizer,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Build one isolated MultiMolecule ESMC row per protein chain."""

    proteins: list[tuple[str, PolymerSequence]] = []
    for instance_id, entity_key in system.instances.items():
        entity = system.entities[entity_key]
        if isinstance(entity, PolymerSequence) and entity.polymer_type == "protein":
            proteins.append((instance_id, entity))

    if proteins:
        max_length = max(len(entity) for _, entity in proteins) + 2
        input_ids = torch.full(
            (len(proteins), max_length),
            tokenizer.pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros(len(proteins), max_length, dtype=torch.bool)
        row_by_instance: dict[str, int] = {}
        for row, (instance_id, entity) in enumerate(proteins):
            row_by_instance[instance_id] = row
            letters = [restype_3to1.get(component.component_id, "X") for component in entity.components]
            residue_ids = tokenizer.convert_tokens_to_ids(letters)
            chain_ids = torch.tensor(
                [tokenizer.bos_token_id, *residue_ids, tokenizer.eos_token_id],
                dtype=torch.long,
            )
            input_ids[row, : len(chain_ids)] = chain_ids
            attention_mask[row, : len(chain_ids)] = True
    else:
        max_length = 2
        row_by_instance = {}
        input_ids = torch.tensor([[tokenizer.bos_token_id, tokenizer.eos_token_id]], dtype=torch.long)
        attention_mask = torch.ones(1, 2, dtype=torch.bool)

    token_to_esmc: list[int] = []
    token_has_esmc: list[bool] = []
    for instance_id, monomer_index, molecule_type in zip(
        token_instance_ids,
        token_monomer_indices,
        molecule_types,
    ):
        has_esmc = molecule_type == 0
        token_has_esmc.append(has_esmc)
        token_to_esmc.append(
            row_by_instance[instance_id] * max_length + cast(int, monomer_index) + 1 if has_esmc else 0
        )
    return (
        input_ids,
        attention_mask,
        torch.tensor(token_to_esmc, dtype=torch.long),
        torch.tensor(token_has_esmc, dtype=torch.bool),
    )


# Antibody Fv prompts use ANARCI Chothia numbering. CDR-H1/H2/H3 and
# CDR-L1/L2/L3 are mutable; framework positions remain fixed.


def is_antibody_sequence(sequence: str) -> bool:
    r"""Return ``True`` when ANARCI recognises ``sequence`` as antibody variable domain(s).

    Mutable ``#`` marks are substituted with alanine before numbering, and every
    detected domain must be a heavy
    (``H``) or light (``K``/``L``) variable domain.
    """

    from abnumber import Chain, ChainParseError

    resolved = sequence.replace(_MUTABLE_TOKEN, "A")
    try:
        chains = Chain.multiple_domains(
            resolved,
            scheme=_ANTIBODY_CDR_SCHEME,
            cdr_definition=_ANTIBODY_CDR_SCHEME,
            allowed_species=None,
            use_anarcii=False,
        )
    except ChainParseError:
        return False
    valid_chain_types = {"H", "K", "L"}
    return all(chain.chain_type in valid_chain_types for chain in chains)


def antibody_cdr_indices(sequence: str) -> list[int]:
    r"""0-based indices of every Chothia CDR residue (H1/H2/H3, L1/L2/L3) in an Fv ``sequence``.

    ANARCI splits the Fv into variable domains, orders the heavy domain first, and locates
    each domain's ``cdr1/cdr2/cdr3`` loops in the input sequence. These indices are the
    CDR-only design positions (framework fixed) and the CDR ipTM-proxy scoring rows.
    """

    from abnumber import Chain

    chains = Chain.multiple_domains(
        sequence,
        scheme=_ANTIBODY_CDR_SCHEME,
        cdr_definition=_ANTIBODY_CDR_SCHEME,
        allowed_species=None,
        use_anarcii=False,
    )
    if len(chains) == 2 and not chains[0].is_heavy_chain():
        chains.reverse()
    indices: list[int] = []
    for chain in chains:
        for cdr in (chain.cdr1_seq, chain.cdr2_seq, chain.cdr3_seq):
            start = sequence.find(cdr)
            if start < 0:
                raise ValueError(f"Chothia CDR {cdr!r} was numbered but not found in the Fv sequence")
            indices.extend(range(start, start + len(cdr)))
    return indices


def antibody_cdr_prompt(fv_sequence: str) -> str:
    r"""Build a CDR-only design mask (as a ``binder_prompt``) from a full antibody Fv sequence.

    The antibody input-preparation step: ANARCI annotates the Chothia CDRs and the returned
    prompt fixes every framework residue to its input amino acid while marking every CDR
    position with the mutable ``#`` token. The prompt is the per-input design mask;
    [`EsmFold2DesignPipeline.design`][ofoldx.models.esmfold2.pipeline.EsmFold2DesignPipeline.design] then
    optimises exactly the marked CDR positions with no antibody-specific branch.
    Antibody framework design uses ``language_model_weight=0.05``; general binders
    use the ``0.15`` default.
    """

    cdr_indices = set(antibody_cdr_indices(fv_sequence))
    return "".join(_MUTABLE_TOKEN if i in cdr_indices else residue for i, residue in enumerate(fv_sequence))
