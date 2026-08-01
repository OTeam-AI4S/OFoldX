r"""
Input processing for ESMFold2 structure prediction and binder design.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

import torch
from multimolecule import ProteinTokenizer

from ...data.evidence import (
    MSA,
    MSA_GAP_INDEX,
    MsaCollection,
    msa_residue_index,
    resolve_msa_bindings,
)
from ...data.molecular import (
    AMINO_ACID_THREE_TO_ONE,
    BOND_ROLE_POLYMER,
    ENTITY_DNA,
    ENTITY_MOLECULE,
    ENTITY_PROTEIN,
    ENTITY_RNA,
    STANDARD_AMINO_ACID_IDS,
    MolecularSystem,
    PolymerSequence,
)
from ...data.structure import Structure
from ...data.tokenization import TOKEN_INDEX, tokenize_molecular_system
from ...processor import Processor
from .config import EsmFold2ExperimentalProcessorConfig, EsmFold2ProcessorConfig
from .model import EsmFold2Input

if TYPE_CHECKING:
    from ...model import Model
    from .model import EsmFold2ExperimentalModel, EsmFold2Model

_MOLECULE_TYPES = {"protein": 0, "dna": 1, "rna": 2}

# Binder-design prompts use amino-acid letters for fixed positions and ``#`` for
# mutable positions. Antibody CDRs use Chothia numbering.
_MUTABLE_TOKEN = "#"
_ANTIBODY_CDR_SCHEME = "chothia"


class EsmFold2Processor(Processor):
    r"""
    Convert supported protein inputs into `EsmFold2Input`.

    Args:
        config: MSA-processing options. Defaults to ``EsmFold2ProcessorConfig()``.
        device: Device for produced tensors.
        dtype: Floating-point dtype for produced tensors.
    """

    config_class = EsmFold2ProcessorConfig
    config: EsmFold2ProcessorConfig

    def __init__(
        self,
        config: EsmFold2ProcessorConfig | None = None,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.config = EsmFold2ProcessorConfig() if config is None else config
        super().__init__(device=device, dtype=dtype)
        self.tokenizer = ProteinTokenizer()

    def validate_model_input_layout(self, model: Model) -> None:
        r"""Require the Processor and folding graph to agree on MSA conditioning."""

        model = cast("EsmFold2Model", model)
        if self.config.msa_conditioning != model.config.msa_encoder.enabled:
            raise ValueError("processor.msa_conditioning must match model.msa_encoder.enabled")

    def __call__(
        self,
        input: PolymerSequence | Sequence[PolymerSequence] | MolecularSystem | Structure,
        *,
        msa: MSA | MsaCollection | None = None,
    ) -> EsmFold2Input:
        r"""
        Build model inputs for one molecular system and optional MSA evidence.

        Args:
            input: One or more polymer sequences, molecular system, or structure topology.
                Coordinates in a ``Structure`` are ignored.
            msa: Optional canonical MSA evidence.

        Returns:
            An [`EsmFold2Input`][] with batch size one and aligned language-model, MSA,
            token, atom, and canonical molecular index axes.
        """

        if msa is not None and not self.config.msa_conditioning:
            raise ValueError("this ESMFold2 checkpoint is the single-sequence Fast variant")
        if isinstance(input, Structure):
            input = input.system
        system = input if isinstance(input, MolecularSystem) else esmfold2_molecular_system_from_sequences(input)
        resolved_msa: tuple[tuple[MSA, int | None, int], ...] = ()
        if msa is not None:
            resolved_msa = resolve_msa_bindings(msa, system)
            for _value, _target_instance_index, entity_index in resolved_msa:
                if int(system.entity_type[entity_index]) != ENTITY_PROTEIN:
                    raise ValueError("ESMFold2 accepts MSA evidence only for protein entities")
        return _prepare_esmfold2_features(
            system,
            msa=msa,
            resolved_msa=resolved_msa,
            tokenizer=self.tokenizer,
            device=self.device,
            dtype=self.dtype,
        )


class EsmFold2ExperimentalProcessor(EsmFold2Processor):
    r"""
    Convert supported binder-design inputs into `EsmFold2Input`.

    Args:
        config: Design-time MSA-processing options. Defaults to
            ``EsmFold2ExperimentalProcessorConfig()``.
        device: Device for produced tensors.
        dtype: Floating-point dtype for produced tensors.
    """

    config_class = EsmFold2ExperimentalProcessorConfig
    config: EsmFold2ExperimentalProcessorConfig

    def __init__(
        self,
        config: EsmFold2ExperimentalProcessorConfig | None = None,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        config = EsmFold2ExperimentalProcessorConfig() if config is None else config
        super().__init__(config, device=device, dtype=dtype)

    def validate_model_input_layout(self, model: Model) -> None:
        r"""Require the Processor and experimental graph to agree on MSA inputs."""

        model = cast("EsmFold2ExperimentalModel", model)
        if self.config.msa_conditioning != model.config.use_msa_features:
            raise ValueError("processor.msa_conditioning must match model.use_msa_features")


def esmfold2_molecular_system_from_sequences(
    input: PolymerSequence | Sequence[PolymerSequence],
) -> MolecularSystem:
    r"""
    Construct the polymer-only convenience input accepted by the fold boundary.
    """

    if isinstance(input, PolymerSequence):
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
            "ESMFold2 input must be a PolymerSequence, sequence of PolymerSequence "
            "values, MolecularSystem, or Structure"
        )

    return esmfold2_molecular_system_from_named_sequences(named)


def esmfold2_molecular_system_from_named_sequences(
    named: Sequence[tuple[str, PolymerSequence]],
) -> MolecularSystem:
    r"""
    Construct one system while preserving caller-supplied entity names.
    """

    entities: dict[str, PolymerSequence] = {}
    instances: dict[str, str] = {}
    for base_id, sequence in named:
        key = base_id
        suffix = 1
        while key in entities:
            suffix += 1
            key = f"{base_id}_{suffix}"
        entities[key] = sequence
        instances[key] = key
    return MolecularSystem.from_entities(entities, instances)


def _default_chain_id(index: int) -> str:
    r"""
    Assign a default entity and instance key.
    """

    return chr(ord("A") + index) if index < 26 else f"chain_{index + 1}"


def _prepare_esmfold2_features(
    system: MolecularSystem,
    *,
    msa: MSA | MsaCollection | None,
    resolved_msa: tuple[tuple[MSA, int | None, int], ...],
    tokenizer: ProteinTokenizer,
    device: torch.device,
    dtype: torch.dtype,
) -> EsmFold2Input:
    r"""
    Build ESMFold2 tensors directly from canonical hierarchy and chemistry tensors.
    """

    instance_count = system.instance_symbol.numel()
    component_count = system.component_symbol.numel()
    component_instance = torch.repeat_interleave(
        torch.arange(instance_count, device=system.instance_component_ptr.device),
        system.instance_component_ptr[1:] - system.instance_component_ptr[:-1],
    )
    atom_component = torch.repeat_interleave(
        torch.arange(component_count, device=system.component_atom_ptr.device),
        system.component_atom_ptr[1:] - system.component_atom_ptr[:-1],
    )

    atom_names = [system.symbols[int(symbol)] for symbol in system.atom_symbol.detach().cpu().tolist()]
    atom_entity = system.instance_entity[component_instance[atom_component]].detach().cpu().tolist()
    entity_types = system.entity_type.detach().cpu().tolist()
    atom_mask = torch.tensor(
        [
            entity_types[atom_entity[atom_index]] == ENTITY_MOLECULE or atom_name not in {"OXT", "OP3"}
            for atom_index, atom_name in enumerate(atom_names)
        ],
        dtype=torch.bool,
        device=system.atom_symbol.device,
    )
    layout = tokenize_molecular_system(system, atom_mask=atom_mask)
    token_atom_ptr = layout.token_atom_ptr.detach().cpu().tolist()
    layout_atoms = layout.atom_index.detach().cpu().tolist()
    token_groups = [
        tuple(layout_atoms[token_atom_ptr[token_index] : token_atom_ptr[token_index + 1]])
        for token_index in range(layout.token_component_index.numel())
    ]
    token_component_indices = layout.token_component_index.detach().cpu().tolist()
    token_atom_indices = layout.token_atom_index.detach().cpu().tolist()
    token_instance_indices = layout.token_instance_index.detach().cpu().tolist()
    token_entity_indices = layout.token_entity_index.detach().cpu().tolist()
    token_monomer_indices: list[int] = []
    residue_indices: list[int] = []
    chain_indices: list[int] = []
    entity_indices: list[int] = []
    symmetry_indices: list[int] = []
    molecule_types: list[int] = []
    residue_type_ids = layout.token_type_index.detach().cpu().tolist()
    msa_query_ids = residue_type_ids.copy()
    reference_space_indices = [atom_component[atom_index].item() for atom_index in layout_atoms]

    polymer_type = {
        ENTITY_PROTEIN: "protein",
        ENTITY_DNA: "dna",
        ENTITY_RNA: "rna",
    }
    for token_index, _component_index in enumerate(token_component_indices):
        instance_index = token_instance_indices[token_index]
        entity_index = token_entity_indices[token_index]
        entity_type = entity_types[entity_index]
        local_component_index = int(layout.token_residue_index[token_index])
        if entity_type != ENTITY_MOLECULE:
            family_polymer_type = polymer_type[entity_type]
            molecule_type = _MOLECULE_TYPES[family_polymer_type]
            token_monomer_indices.append(local_component_index)
            molecule_types.append(molecule_type)
        else:
            token_monomer_indices.append(-1)
            molecule_types.append(3)
        residue_indices.append(local_component_index)
        chain_indices.append(instance_index)
        entity_indices.append(entity_index + 1)
        symmetry_indices.append(int(layout.token_symmetry_index[token_index]))

    atom_indices = tuple(atom_index for group in token_groups for atom_index in group)
    canonical_to_model_atom = {atom_index: model_index for model_index, atom_index in enumerate(atom_indices)}
    real_atom_count = len(atom_indices)
    padded_atom_count = max(32, math.ceil(real_atom_count / 32) * 32)

    selected_atoms = torch.tensor(atom_indices, dtype=torch.long, device=system.atom_symbol.device)
    reference_coordinate_mask = system.reference_coordinate_mask[selected_atoms].detach().cpu()
    reference_space_tensor = torch.tensor(reference_space_indices, dtype=torch.long)
    for reference_space in reference_space_tensor.unique():
        space_mask = reference_space_tensor == reference_space
        if int(space_mask.sum()) > 1 and not bool(reference_coordinate_mask[space_mask].all()):
            first_atom = atom_indices[int(space_mask.nonzero(as_tuple=False)[0, 0])]
            component_index = int(atom_component[first_atom])
            instance_index = int(component_instance[component_index])
            raise ValueError(f"ESMFold2 reference conformer is incomplete for {system.instance_id(instance_index)!r}")

    encoded_atom_names: list[list[int]] = []
    molecule_atom_counts: dict[tuple[int, str], int] = {}
    from rdkit.Chem import GetPeriodicTable

    periodic_table = GetPeriodicTable()
    for atom_index in atom_indices:
        atom_name = system.symbols[int(system.atom_symbol[atom_index])]
        component_index = int(atom_component[atom_index])
        instance_index = int(component_instance[component_index])
        entity_index = int(system.instance_entity[instance_index])
        if int(system.entity_type[entity_index]) == ENTITY_MOLECULE and not _valid_atom_name(atom_name):
            element = periodic_table.GetElementSymbol(int(system.atomic_number[atom_index])).upper()
            key = (instance_index, element)
            molecule_atom_counts[key] = molecule_atom_counts.get(key, 0) + 1
            atom_name = f"{element}{molecule_atom_counts[key]}"
        if not _valid_atom_name(atom_name):
            raise ValueError(f"ESMFold2 atom name cannot be encoded in four 6-bit characters: {atom_name!r}")
        encoded_atom_names.append([ord(character) - 32 if character != " " else 0 for character in atom_name.ljust(4)])

    reference_positions = torch.zeros(padded_atom_count, 3, dtype=torch.float32)
    reference_elements = torch.zeros(padded_atom_count, dtype=torch.long)
    reference_charges = torch.zeros(padded_atom_count, dtype=torch.int8)
    reference_atom_name_characters = torch.zeros(padded_atom_count, 4, dtype=torch.long)
    reference_space_id = torch.zeros(padded_atom_count, dtype=torch.long)
    atom_mask = torch.zeros(padded_atom_count, dtype=torch.bool)
    atom_to_token = torch.zeros(padded_atom_count, dtype=torch.long)
    atom_index = torch.full((padded_atom_count,), -1, dtype=torch.long)
    reference_positions[:real_atom_count] = system.reference_coordinates[selected_atoms].detach().cpu()
    reference_elements[:real_atom_count] = system.atomic_number[selected_atoms].detach().cpu().long()
    reference_charges[:real_atom_count] = (
        torch.where(
            system.formal_charge_known[selected_atoms],
            system.formal_charge[selected_atoms],
            0,
        )
        .detach()
        .cpu()
        .to(torch.int8)
    )
    reference_atom_name_characters[:real_atom_count] = torch.tensor(encoded_atom_names, dtype=torch.long)
    reference_space_id[:real_atom_count] = reference_space_tensor
    atom_mask[:real_atom_count] = True
    atom_index[:real_atom_count] = torch.tensor(atom_indices, dtype=torch.long)
    atom_to_token_values = [token_index for token_index, group in enumerate(token_groups) for _ in group]
    atom_to_token[:real_atom_count] = torch.tensor(atom_to_token_values, dtype=torch.long)

    token_count = len(token_groups)
    representative_atoms = [
        _representative_atom(
            system,
            group,
            molecule_types[index],
            residue_type_ids[index],
            canonical_to_model_atom,
        )
        for index, group in enumerate(token_groups)
    ]
    token_bonds = _token_bond_matrix(
        system,
        atom_component=atom_component,
        component_instance=component_instance,
        atom_indices=atom_indices,
        atom_to_token=atom_to_token_values,
        token_count=token_count,
    )
    msa_features = _msa_tensors(
        msa,
        resolved_msa=resolved_msa,
        token_entity_indices=token_entity_indices,
        token_instance_indices=token_instance_indices,
        token_monomer_indices=token_monomer_indices,
        query_msa_ids=msa_query_ids,
    )
    language_model_tokens, language_model_mask, token_to_esmc, token_has_esmc = _esmc_tensors(
        system,
        token_instance_indices=token_instance_indices,
        token_monomer_indices=token_monomer_indices,
        molecule_types=molecule_types,
        tokenizer=tokenizer,
    )

    features = {
        "token_index": torch.arange(token_count, dtype=torch.long),
        "residue_index": torch.tensor(residue_indices, dtype=torch.long),
        "chain_index": torch.tensor(chain_indices, dtype=torch.long),
        "symmetry_index": torch.tensor(symmetry_indices, dtype=torch.long),
        "entity_index": torch.tensor(entity_indices, dtype=torch.long),
        "molecule_type": torch.tensor(molecule_types, dtype=torch.long),
        "residue_type_ids": torch.tensor(residue_type_ids, dtype=torch.long),
        "token_bonds": token_bonds,
        "token_mask": torch.ones(token_count, dtype=torch.bool),
        "reference_positions": reference_positions,
        "reference_elements": reference_elements,
        "reference_charges": reference_charges,
        "reference_atom_name_characters": reference_atom_name_characters,
        "reference_space_id": reference_space_id,
        "atom_mask": atom_mask,
        "atom_to_token": atom_to_token,
        "atom_index": atom_index,
        "token_component_index": torch.tensor(token_component_indices, dtype=torch.long),
        "token_atom_index": torch.tensor(token_atom_indices, dtype=torch.long),
        "distogram_atom_index": torch.tensor(representative_atoms, dtype=torch.long),
        "token_has_esmc": token_has_esmc,
        **msa_features,
    }
    placed = {
        name: tensor.to(device=device, dtype=dtype) if tensor.is_floating_point() else tensor.to(device=device)
        for name, tensor in features.items()
    }
    return EsmFold2Input(  # type: ignore[call-arg]
        **{name: value.unsqueeze(0) for name, value in placed.items()},
        language_model_tokens=language_model_tokens.to(device=device),
        language_model_mask=language_model_mask.to(device=device),
        token_to_esmc=token_to_esmc.unsqueeze(0).to(device=device),
        system=system,
    )


def _valid_atom_name(atom_name: str) -> bool:
    r"""
    Return whether the four-character atom encoding is lossless.
    """

    return len(atom_name) <= 4 and all(32 <= ord(character) < 96 for character in atom_name)


def _representative_atom(
    system: MolecularSystem,
    group: tuple[int, ...],
    molecule_type: int,
    residue_type: int,
    canonical_to_model_atom: dict[int, int],
) -> int:
    r"""
    Select the representative model-atom index for one structure token.
    """

    by_name = {
        system.symbols[int(system.atom_symbol[atom_index])]: canonical_to_model_atom[atom_index] for atom_index in group
    }
    fallback = canonical_to_model_atom[group[0]]
    if molecule_type == 0:
        return by_name.get("CB", by_name.get("CA", fallback))
    if molecule_type in (1, 2):
        if residue_type in (TOKEN_INDEX["RNA_N"], TOKEN_INDEX["DNA_N"]):
            return by_name.get("C1'", fallback)
        if residue_type in (
            TOKEN_INDEX["RNA_A"],
            TOKEN_INDEX["RNA_G"],
            TOKEN_INDEX["DNA_A"],
            TOKEN_INDEX["DNA_G"],
        ):
            return by_name.get("C4", by_name.get("C1'", fallback))
        return by_name.get("C2", by_name.get("C1'", fallback))
    return fallback


def _token_bond_matrix(
    system: MolecularSystem,
    *,
    atom_component: torch.Tensor,
    component_instance: torch.Tensor,
    atom_indices: tuple[int, ...],
    atom_to_token: list[int],
    token_count: int,
) -> torch.Tensor:
    r"""
    Project canonical intrinsic, polymer, and assembly bonds onto token pairs.
    """

    canonical_atom_to_token = torch.full(
        (system.atom_symbol.numel(),),
        -1,
        dtype=torch.long,
        device=system.atom_symbol.device,
    )
    canonical_atom_to_token[torch.tensor(atom_indices, dtype=torch.long, device=system.atom_symbol.device)] = (
        torch.tensor(atom_to_token, dtype=torch.long, device=system.atom_symbol.device)
    )
    edge_index = system.edge_index
    left_token = canonical_atom_to_token[edge_index[0]]
    right_token = canonical_atom_to_token[edge_index[1]]
    keep = (left_token >= 0) & (right_token >= 0) & (left_token != right_token)

    left_component = atom_component[edge_index[0]]
    right_component = atom_component[edge_index[1]]
    left_instance = component_instance[left_component]
    left_entity = system.instance_entity[left_instance]
    protein_standard = torch.tensor(
        [system.symbols[int(symbol)] in STANDARD_AMINO_ACID_IDS for symbol in system.component_symbol.detach().cpu()],
        dtype=torch.bool,
        device=system.component_symbol.device,
    )
    polymer_edge = system.bond_role == BOND_ROLE_POLYMER
    implicit_polymer_edge = polymer_edge & (
        (system.entity_type[left_entity] != ENTITY_PROTEIN)
        | (protein_standard[left_component] & protein_standard[right_component])
    )
    keep &= ~implicit_polymer_edge

    values = torch.zeros(token_count, token_count, 1, dtype=torch.float32)
    left = left_token[keep].detach().cpu()
    right = right_token[keep].detach().cpu()
    values[left, right, 0] = 1.0
    values[right, left, 0] = 1.0
    return values


def _msa_tensors(
    msa: MSA | MsaCollection | None,
    *,
    resolved_msa: tuple[tuple[MSA, int | None, int], ...],
    token_entity_indices: list[int],
    token_instance_indices: list[int],
    token_monomer_indices: list[int],
    query_msa_ids: list[int],
) -> dict[str, torch.Tensor]:
    r"""
    Construct paired and unpaired MSA rows over structure tokens.
    """

    token_count = len(query_msa_ids)
    if msa is None:
        tokens = torch.tensor([query_msa_ids], dtype=torch.long)
        deletion_counts = torch.zeros(1, token_count, dtype=torch.float32)
    else:
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

        token_entity = torch.tensor(token_entity_indices, dtype=torch.long)
        token_instance = torch.tensor(token_instance_indices, dtype=torch.long)
        token_monomer = torch.tensor(token_monomer_indices, dtype=torch.long)
        rows: list[torch.Tensor] = []
        deletions: list[torch.Tensor] = []
        for selections in plan_rows:
            row = torch.full((token_count,), MSA_GAP_INDEX, dtype=torch.long)
            deletion_row = torch.zeros(token_count, dtype=torch.float32)
            for column_index, source_row, is_paired in selections:
                value, target_instance_index, entity_index = resolved_msa[column_index]
                aligned = value.paired_sequences[source_row] if is_paired else value.unpaired_sequences[source_row]
                deletion_source = (value.paired_deletion_counts if is_paired else value.unpaired_deletion_counts)[
                    source_row
                ]
                selected = (token_entity == entity_index) & (token_monomer >= 0)
                if target_instance_index is not None:
                    selected &= token_instance == target_instance_index
                monomer = token_monomer[selected]
                aligned_types = torch.tensor(
                    [msa_residue_index(symbol, "protein") for symbol in aligned],
                    dtype=torch.long,
                )
                row[selected] = aligned_types[monomer]
                deletion_row[selected] = torch.from_numpy(deletion_source.astype("float32", copy=False))[monomer]
            rows.append(row)
            deletions.append(deletion_row)
        tokens = torch.stack(rows)
        deletion_counts = torch.stack(deletions)
        tokens[0] = torch.tensor(query_msa_ids, dtype=torch.long)

    has_deletion = deletion_counts > 0
    deletion_value = (math.pi / 2) * torch.atan(deletion_counts / 3)
    return {
        "msa_tokens": tokens,
        "has_deletion": has_deletion,
        "deletion_value": deletion_value,
        "deletion_mean": deletion_value.mean(dim=0),
        "msa_mask": torch.ones_like(tokens, dtype=torch.bool),
    }


def _esmc_tensors(
    system: MolecularSystem,
    *,
    token_instance_indices: list[int],
    token_monomer_indices: list[int],
    molecule_types: list[int],
    tokenizer: ProteinTokenizer,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    Build one isolated MultiMolecule ESMC row per protein chain.
    """

    proteins = [
        instance_index
        for instance_index, entity_index in enumerate(system.instance_entity.detach().cpu().tolist())
        if int(system.entity_type[entity_index]) == ENTITY_PROTEIN
    ]
    if proteins:
        max_length = (
            max(
                int(system.instance_component_ptr[index + 1] - system.instance_component_ptr[index])
                for index in proteins
            )
            + 2
        )
        language_model_tokens = torch.full(
            (len(proteins), max_length),
            tokenizer.pad_token_id,
            dtype=torch.long,
        )
        language_model_mask = torch.zeros(len(proteins), max_length, dtype=torch.bool)
        row_by_instance: dict[int, int] = {}
        for row, instance_index in enumerate(proteins):
            row_by_instance[instance_index] = row
            start = int(system.instance_component_ptr[instance_index])
            stop = int(system.instance_component_ptr[instance_index + 1])
            letters = [
                AMINO_ACID_THREE_TO_ONE.get(system.symbols[int(symbol)], "X")
                for symbol in system.component_symbol[start:stop].detach().cpu()
            ]
            residue_tokens = tokenizer.convert_tokens_to_ids(letters)
            sequence_tokens = torch.tensor(
                [tokenizer.bos_token_id, *residue_tokens, tokenizer.eos_token_id],
                dtype=torch.long,
            )
            language_model_tokens[row, : len(sequence_tokens)] = sequence_tokens
            language_model_mask[row, : len(sequence_tokens)] = True
    else:
        max_length = 2
        row_by_instance = {}
        language_model_tokens = torch.tensor([[tokenizer.bos_token_id, tokenizer.eos_token_id]], dtype=torch.long)
        language_model_mask = torch.ones(1, 2, dtype=torch.bool)

    token_to_esmc: list[int] = []
    token_has_esmc: list[bool] = []
    for instance_index, monomer_index, molecule_type in zip(
        token_instance_indices,
        token_monomer_indices,
        molecule_types,
    ):
        has_esmc = molecule_type == 0
        token_has_esmc.append(has_esmc)
        token_to_esmc.append(row_by_instance[instance_index] * max_length + monomer_index + 1 if has_esmc else 0)
    return (
        language_model_tokens,
        language_model_mask,
        torch.tensor(token_to_esmc, dtype=torch.long),
        torch.tensor(token_has_esmc, dtype=torch.bool),
    )


# Antibody Fv prompts use ANARCI Chothia numbering. CDR-H1/H2/H3 and
# CDR-L1/L2/L3 are mutable; framework positions remain fixed.


def is_antibody_sequence(sequence: str) -> bool:
    r"""
    Check whether a sequence contains only antibody variable domains.

    Args:
        sequence: Amino-acid sequence. Mutable ``#`` positions are treated as alanine
            during numbering.

    Returns:
        ``True`` when numbering succeeds and every detected domain is a heavy or light
            antibody variable domain; otherwise ``False``.
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
    r"""
    Return the Chothia CDR positions in an antibody Fv sequence.

    Args:
        sequence: Antibody Fv amino-acid sequence.

    Returns:
        Zero-based sequence indices for CDR1, CDR2, and CDR3 of each variable domain,
            with the heavy domain listed before the light domain when both are present.
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
    r"""
    Mark the CDR residues of an antibody Fv sequence as designable.

    Args:
        fv_sequence: Antibody Fv amino-acid sequence.

    Returns:
        A sequence-length prompt with each Chothia CDR residue replaced by ``#`` and
            each framework residue preserved.
    """

    cdr_indices = set(antibody_cdr_indices(fv_sequence))
    return "".join(_MUTABLE_TOKEN if i in cdr_indices else residue for i, residue in enumerate(fv_sequence))
