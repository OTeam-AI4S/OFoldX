r"""
Input processing for Boltz-1 structure prediction.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor

from ...data.evidence import MSA, MSACollection, resolve_msa_bindings
from ...data.molecular import (
    BOND_ROLE_ASSEMBLY,
    BOND_ROLE_COMPONENT,
    BOND_ROLE_ENTITY,
    BOND_ROLE_POLYMER,
    ENTITY_DNA,
    ENTITY_DNA_RNA_HYBRID,
    ENTITY_MOLECULE,
    ENTITY_PNA,
    ENTITY_PROTEIN,
    ENTITY_RNA,
    ComponentReference,
    MolecularSystem,
)
from ...data.structure import Structure
from ...processor import Processor
from .model import (
    Boltz1Input,
    Boltz1Pocket,
    Boltz1StructureTarget,
)

if TYPE_CHECKING:
    from rdkit.Chem.rdchem import Mol

# Boltz-1 pocket one-hot classes: UNSPECIFIED, UNSELECTED, POCKET, and BINDER.
_POCKET_UNSPECIFIED = 0
_POCKET_UNSELECTED = 1
_POCKET_POCKET = 2
_POCKET_BINDER = 3
_POCKET_SIZE = 4

# The 33-token residue vocabulary places proteins at 2..21, UNKNOWN tokens at
# 22/27/32, RNA at 23..27, DNA at 28..32, and the MSA gap at 1.
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
_STANDARD_BY_ENTITY_TYPE = {
    ENTITY_PROTEIN: _STANDARD_COMPONENTS["protein"],
    ENTITY_RNA: _STANDARD_COMPONENTS["rna"],
    ENTITY_DNA: _STANDARD_COMPONENTS["dna"],
    ENTITY_DNA_RNA_HYBRID: _STANDARD_COMPONENTS["rna"] | _STANDARD_COMPONENTS["dna"],
    ENTITY_PNA: frozenset(),
}
# Boltz parses these polymer components through its canonical residue path and
# therefore does not emit physical-potential constraints for them. MSE is
# normalized to MET before that decision.
_CONSTRAINT_FREE_COMPONENTS = {
    "protein": frozenset((*_PROTEIN_RESIDUE_TYPES, "UNK", "MSE")),
    "rna": frozenset(_RNA_RESIDUE_TYPES),
    "dna": frozenset(_DNA_RESIDUE_TYPES),
}
_MOLECULE_TYPES = {"protein": 0, "dna": 1, "rna": 2}
_UNKNOWN_RESIDUE_TYPES = {"protein": 22, "rna": 27, "dna": 32}
_POLYMER_TYPE = {
    ENTITY_PROTEIN: "protein",
    ENTITY_RNA: "rna",
    ENTITY_DNA: "dna",
}
_MSA_GAP = 1
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
# Processor owns the symbol->token mapping keyed by the entity's resolved polymer_type, reproducing the
# Boltz-1 ``_residue_type`` path: standard residues map to their model token and everything else to the
# model's UNKNOWN token. RNA one-letter symbols equal their component IDs directly.
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
    Map one UPPERCASE one-letter MSA symbol (or ``-``) to a Boltz-1 token.
    """

    if symbol == "-":
        return _MSA_GAP
    if polymer_type == "protein":
        return _PROTEIN_MSA_RESTYPE.get(symbol, _UNKNOWN_RESIDUE_TYPES["protein"])
    if polymer_type == "rna":
        return _RNA_RESIDUE_TYPES.get(symbol, _UNKNOWN_RESIDUE_TYPES["rna"])
    return _DNA_MSA_RESTYPE.get(symbol, _UNKNOWN_RESIDUE_TYPES["dna"])


class Boltz1Processor(Processor):
    r"""
    Convert supported biomolecular inputs into `Boltz1Input`.

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
        pocket: Boltz1Pocket | None = None,
        generator: torch.Generator | None = None,
    ) -> Boltz1Input:
        r"""
        Prepare folding features for one molecular system.

        Args:
            input: Molecular system or structure.
            msa: Optional MSA value or collection.
            pocket: Optional binder and pocket-residue specification.
            generator: Optional generator for independent rigid transforms of reference
                conformers.

        Returns:
            Batched token, atom, MSA, and pocket tensors. Reference coordinates
            are in angstroms; canonical integer indices bind the family atom
            and token axes to ``system``.
        """

        if isinstance(input, Structure):
            structure = input
            system = input.system
        else:
            structure = None
            system = input
        if system.system_entity_ptr.numel() != 2:
            raise ValueError("Boltz-1 Processor accepts one canonical molecular system")
        if not system.instance_symbol.numel():
            raise ValueError("Boltz-1 requires at least one molecular instance")

        resolved_msa = () if msa is None else resolve_msa_bindings(msa, system)
        return _prepare_boltz1_features(
            system,
            structure=structure,
            msa=msa,
            msa_bindings=resolved_msa,
            pocket=pocket,
            device=self.device,
            dtype=self.dtype,
            generator=generator,
        )

    def process_training(
        self,
        input: Structure,
        *,
        msa: MSA | MSACollection | None = None,
        pocket: Boltz1Pocket | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[Boltz1Input, Boltz1StructureTarget]:
        r"""
        Lower one complete structure into model input and separate truth.
        """

        processed = self(
            input,
            msa=msa,
            pocket=pocket,
            generator=generator,
        )
        return processed, _prepare_boltz1_target(
            input,
            processed.atom_index,
            device=self.device,
            dtype=self.dtype,
        )


def _prepare_boltz1_features(
    system: MolecularSystem,
    *,
    structure: Structure | None,
    msa: MSA | MSACollection | None,
    msa_bindings: tuple[tuple[MSA, int | None, int], ...],
    pocket: Boltz1Pocket | None = None,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None = None,
) -> Boltz1Input:
    r"""
    Build Boltz-1 tensors directly from the canonical molecular graph.
    """

    instance_components = system.instance_component_ptr.detach().cpu().tolist()
    component_atoms = system.component_atom_ptr.detach().cpu().tolist()
    instance_entities = system.instance_entity.detach().cpu().tolist()
    entity_types = system.entity_type.detach().cpu().tolist()
    component_symbols = system.component_symbol.detach().cpu().tolist()
    atom_symbols = system.atom_symbol.detach().cpu().tolist()
    atomic_numbers = system.atomic_number.detach().cpu().tolist()
    atom_component_index = _atom_component_index(system)
    atom_components = atom_component_index.detach().cpu().tolist()
    instance_count_by_entity: dict[int, int] = {}
    tokens: list[dict[str, object]] = []

    for instance_index, entity_index in enumerate(instance_entities):
        entity_type = entity_types[entity_index]
        symmetry_index = instance_count_by_entity.get(entity_index, 0)
        instance_count_by_entity[entity_index] = symmetry_index + 1
        component_start = instance_components[instance_index]
        component_stop = instance_components[instance_index + 1]
        if entity_type != ENTITY_MOLECULE:
            standards = _STANDARD_BY_ENTITY_TYPE.get(entity_type, frozenset())
            for component_index in range(component_start, component_stop):
                monomer_index = component_index - component_start
                component_id = system.symbols[component_symbols[component_index]]
                atom_indices = tuple(
                    atom_index
                    for atom_index in range(
                        component_atoms[component_index],
                        component_atoms[component_index + 1],
                    )
                    if atomic_numbers[atom_index] != 1
                    and system.symbols[atom_symbols[atom_index]] not in {"OXT", "OP3"}
                )
                if not atom_indices:
                    raise ValueError(f"component {component_id!r} has no model-visible heavy atoms")
                is_standard = component_id in standards
                polymer_type = _component_polymer_type(
                    entity_type,
                    component_id,
                )
                common = {
                    "component_index": component_index,
                    "instance_index": instance_index,
                    "entity_index": entity_index,
                    "monomer_index": monomer_index,
                    "residue_index": monomer_index,
                    "chain_index": instance_index,
                    "symmetry_index": symmetry_index,
                    "molecule_type": (3 if polymer_type is None else _MOLECULE_TYPES[polymer_type]),
                }
                if not is_standard:
                    residue_type = 22 if polymer_type is None else _UNKNOWN_RESIDUE_TYPES[polymer_type]
                    for atom_index in atom_indices:
                        tokens.append(
                            {
                                **common,
                                "atoms": (atom_index,),
                                "atom_index": atom_index,
                                "residue_type": residue_type,
                                "center": atom_index,
                                "distogram": atom_index,
                                "frame": (atom_index, atom_index, atom_index),
                            }
                        )
                    continue

                by_name = {system.symbols[atom_symbols[atom_index]]: atom_index for atom_index in atom_indices}
                center_name = "CA" if polymer_type == "protein" else "C1'"
                if polymer_type == "protein":
                    distogram_name = "CA" if component_id == "GLY" else "CB"
                    frame_names = ("N", "CA", "C")
                else:
                    distogram_name = "C4" if component_id in {"A", "G", "DA", "DG"} else "C2"
                    frame_names = ("C1'", "C3'", "C4'")
                center = by_name.get(center_name, atom_indices[0])
                distogram = by_name.get(distogram_name, center)
                tokens.append(
                    {
                        **common,
                        "atoms": atom_indices,
                        "atom_index": None,
                        "residue_type": _residue_type(
                            component_id,
                            cast(str, polymer_type),
                        ),
                        "center": center,
                        "distogram": distogram,
                        "frame": tuple(by_name.get(name, center) for name in frame_names),
                    }
                )
        else:
            atom_indices = tuple(
                atom_index
                for component_index in range(component_start, component_stop)
                for atom_index in range(
                    component_atoms[component_index],
                    component_atoms[component_index + 1],
                )
                if atomic_numbers[atom_index] != 1
            )
            if not atom_indices:
                raise ValueError(f"molecule {system.instance_id(instance_index)!r} " "has no model-visible heavy atoms")
            for atom_index in atom_indices:
                tokens.append(
                    {
                        "atoms": (atom_index,),
                        "atom_index": atom_index,
                        "component_index": atom_components[atom_index],
                        "instance_index": instance_index,
                        "entity_index": entity_index,
                        "monomer_index": None,
                        "residue_index": 0,
                        "chain_index": instance_index,
                        "symmetry_index": symmetry_index,
                        "molecule_type": 3,
                        "residue_type": 22,
                        "center": atom_index,
                        "distogram": atom_index,
                        "frame": (atom_index, atom_index, atom_index),
                    }
                )

    atom_indices = tuple(atom_index for token in tokens for atom_index in cast(tuple[int, ...], token["atoms"]))
    system_to_model_atom = {atom_index: model_atom_index for model_atom_index, atom_index in enumerate(atom_indices)}
    system_atom_to_token = {
        atom_index: token_index
        for token_index, token in enumerate(tokens)
        for atom_index in cast(tuple[int, ...], token["atoms"])
    }
    atom_count = len(atom_indices)
    padded_atom_count = max(32, math.ceil(atom_count / 32) * 32)
    reference_coordinates = _reference_coordinates(system, atom_indices)
    reference_space_ids = (
        atom_component_index[torch.tensor(atom_indices, device=atom_component_index.device)].detach().cpu()
    )
    resolved = torch.zeros(atom_count, dtype=torch.bool)
    if structure is not None:
        resolved = (
            structure.coordinate_mask[
                torch.tensor(
                    atom_indices,
                    device=structure.coordinate_mask.device,
                )
            ]
            .detach()
            .cpu()
        )

    reference_positions = torch.zeros(padded_atom_count, 3)
    reference_positions[:atom_count] = reference_coordinates
    reference_space_id = torch.zeros(padded_atom_count, dtype=torch.long)
    reference_space_id[:atom_count] = reference_space_ids
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
    atom_names = _model_atom_names(system, atom_indices)
    canonical_atom_index = torch.tensor(
        atom_indices,
        device=system.atomic_number.device,
    )
    atomic_number[:atom_count] = system.atomic_number[canonical_atom_index].detach().cpu().long()
    reference_charges[:atom_count] = system.formal_charge[canonical_atom_index].detach().cpu().float()
    for model_atom_index, (system_atom_index, atom_name) in enumerate(zip(atom_indices, atom_names)):
        atom_name_characters[model_atom_index] = torch.tensor(
            [ord(character) - 32 if character != " " else 0 for character in atom_name.ljust(4)]
        )
        atom_mask[model_atom_index] = True
        token_index = system_atom_to_token[system_atom_index]
        atom_to_token[model_atom_index] = token_index
        atom_token_index[model_atom_index] = cast(
            tuple[int, ...],
            tokens[token_index]["atoms"],
        ).index(system_atom_index)

    # Boltz-1 reference-atom features contain positions (3), charge (1), mask (1),
    # element (128), and atom-name characters (4x64=256). The mask channel at index 4
    # is the sole difference from Boltz-2's 388-wide vector.
    atom_features = torch.cat(
        (
            reference_positions,
            reference_charges.unsqueeze(-1),
            atom_mask.unsqueeze(-1).float(),
            torch.nn.functional.one_hot(atomic_number, 128).float(),
            torch.nn.functional.one_hot(atom_name_characters, 64).reshape(padded_atom_count, 256).float(),
        ),
        dim=-1,
    )
    token_count = len(tokens)
    token_center_atom_index = torch.tensor(
        [system_to_model_atom[cast(int, token["center"])] for token in tokens],
        dtype=torch.long,
    )
    token_distogram_atom_index = torch.tensor(
        [system_to_model_atom[cast(int, token["distogram"])] for token in tokens],
        dtype=torch.long,
    )
    token_bonds = _token_bond_indicator(
        system,
        system_atom_to_token,
        token_count,
    )
    msa_values = _msa_features(msa, msa_bindings, tokens, system)
    molecule_types = torch.tensor([cast(int, token["molecule_type"]) for token in tokens])
    residue_types = torch.tensor([cast(int, token["residue_type"]) for token in tokens])
    frame_atom_index = torch.tensor(
        [
            [system_to_model_atom[atom_index] for atom_index in cast(tuple[int, int, int], token["frame"])]
            for token in tokens
        ],
        dtype=torch.long,
    )
    cyclic_period_by_instance = _cyclic_periods(system)
    cyclic_period = torch.tensor(
        [cyclic_period_by_instance[cast(int, token["instance_index"])] for token in tokens],
        dtype=torch.float32,
    )
    pocket_feature = _pocket_feature(pocket, tokens, system)
    steering_features = _steering_features(system, system_to_model_atom)

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
        "atom_token_index": atom_token_index,
        "token_center_atom_index": token_center_atom_index,
        "token_distogram_atom_index": token_distogram_atom_index,
        "residue_type_features": torch.nn.functional.one_hot(residue_types, 33).float(),
        "msa_profile": msa_values.pop("msa_profile"),
        "deletion_mean": msa_values.pop("deletion_mean"),
        "token_bonds": token_bonds,
        "molecule_type": molecule_types,
        "frame_atom_index": frame_atom_index,
        "cyclic_period": cyclic_period,
        "pocket_feature": pocket_feature,
        **steering_features,
        **msa_values,
    }
    placed = {
        name: value.to(device=device, dtype=dtype) if value.is_floating_point() else value.to(device=device)
        for name, value in tensor_values.items()
    }
    return Boltz1Input(  # type: ignore[call-arg]
        **{name: value.unsqueeze(0) for name, value in placed.items()},
        system=system,
        atom_index=torch.cat(
            (
                torch.tensor(atom_indices, device=device),
                torch.full(
                    (padded_atom_count - atom_count,),
                    -1,
                    dtype=torch.long,
                    device=device,
                ),
            )
        ).unsqueeze(0),
        token_component_index=torch.tensor(
            [cast(int, token["component_index"]) for token in tokens],
            dtype=torch.long,
            device=device,
        ).unsqueeze(0),
        token_atom_index=torch.tensor(
            [-1 if token["atom_index"] is None else cast(int, token["atom_index"]) for token in tokens],
            dtype=torch.long,
            device=device,
        ).unsqueeze(0),
        batch_size=[],
    )


def _atom_component_index(system: MolecularSystem) -> Tensor:
    return torch.repeat_interleave(
        torch.arange(
            system.component_symbol.numel(),
            device=system.component_atom_ptr.device,
        ),
        system.component_atom_ptr[1:] - system.component_atom_ptr[:-1],
    )


def _prepare_boltz1_target(
    structure: Structure,
    atom_index: Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Boltz1StructureTarget:
    r"""
    Lower full-axis canonical truth and its discrete symmetry candidates.
    """

    system = structure.system
    instance_components = system.instance_component_ptr.detach().cpu().tolist()
    component_atoms = system.component_atom_ptr.detach().cpu().tolist()
    instance_entities = system.instance_entity.detach().cpu().tolist()
    atom_symbols = system.atom_symbol.detach().cpu().tolist()
    chain_span: list[tuple[int, int]] = []
    chain_equivalence: list[int] = []
    equivalence_by_layout: dict[tuple[int, tuple[tuple[int, int], ...]], int] = {}
    for instance_index, entity_index in enumerate(instance_entities):
        component_start = instance_components[instance_index]
        component_stop = instance_components[instance_index + 1]
        atom_start = component_atoms[component_start]
        atom_stop = component_atoms[component_stop]
        chain_span.append((atom_start, atom_stop))
        layout_key = (
            entity_index,
            tuple(
                (
                    component_index - component_start,
                    atom_symbols[system_atom_index],
                )
                for component_index in range(component_start, component_stop)
                for system_atom_index in range(
                    component_atoms[component_index],
                    component_atoms[component_index + 1],
                )
            ),
        )
        chain_equivalence.append(
            equivalence_by_layout.setdefault(
                layout_key,
                len(equivalence_by_layout),
            )
        )

    symmetry_atom_index, symmetry_permutation = _truth_symmetry_features(
        system,
        tuple(int(value) for value in atom_index[0].detach().cpu().tolist() if value >= 0),
    )
    return Boltz1StructureTarget(  # type: ignore[call-arg]
        coordinates=structure.coordinates.to(
            device=device,
            dtype=dtype,
        ).unsqueeze(0),
        coordinate_mask=structure.coordinate_mask.to(device=device).unsqueeze(0),
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
        batch_size=[1],
    )


def _truth_symmetry_features(
    system: MolecularSystem,
    atom_indices: tuple[int, ...],
) -> tuple[Tensor, Tensor]:
    r"""
    Build complete local permutations in stable model-occurrence order.
    """

    system_to_model_atom = {atom_index: model_atom_index for model_atom_index, atom_index in enumerate(atom_indices)}
    component_instance = (
        torch.repeat_interleave(
            torch.arange(
                system.instance_symbol.numel(),
                device=system.instance_component_ptr.device,
            ),
            system.instance_component_ptr[1:] - system.instance_component_ptr[:-1],
        )
        .detach()
        .cpu()
        .tolist()
    )
    instance_entities = system.instance_entity.detach().cpu().tolist()
    entity_types = system.entity_type.detach().cpu().tolist()
    component_symbols = system.component_symbol.detach().cpu().tolist()
    component_atoms = system.component_atom_ptr.detach().cpu().tolist()
    atom_symbols = system.atom_symbol.detach().cpu().tolist()
    groups: list[tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]] = []
    for component_index, instance_index in enumerate(component_instance):
        entity_index = instance_entities[instance_index]
        entity_type = entity_types[entity_index]
        component_id = system.symbols[component_symbols[component_index]]
        standard = _STANDARD_LOCAL_SYMMETRY.get(component_id)
        if component_id in _STANDARD_BY_ENTITY_TYPE.get(
            entity_type,
            frozenset(),
        ):
            if standard is not None:
                by_name = {
                    system.symbols[atom_symbols[atom_index]]: atom_index
                    for atom_index in range(
                        component_atoms[component_index],
                        component_atoms[component_index + 1],
                    )
                    if atom_index in system_to_model_atom
                }
                permutations = tuple(
                    tuple(system_to_model_atom[by_name[atom_id]] for atom_id in names)
                    for names in standard
                    if all(atom_id in by_name for atom_id in names)
                )
                if len(permutations) > 1:
                    groups.append((permutations[0], permutations))
            continue

        rdkit_group = _rdkit_symmetry_group(
            system,
            component_index,
            system_to_model_atom,
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
    component_index: int,
    system_to_model_atom: dict[int, int],
) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...]] | None:
    from rdkit import Chem

    if not bool(system.component_topology_complete[component_index]):
        return None
    atom_start = int(system.component_atom_ptr[component_index])
    atom_stop = int(system.component_atom_ptr[component_index + 1])
    visible_local_index = [
        atom_index - atom_start for atom_index in range(atom_start, atom_stop) if atom_index in system_to_model_atom
    ]
    if len(visible_local_index) < 2:
        return None

    edge_indices = (
        (
            (system.edge_index[0] >= atom_start)
            & (system.edge_index[1] < atom_stop)
            & (system.bond_role == BOND_ROLE_COMPONENT)
        )
        .nonzero(as_tuple=False)
        .flatten()
        .detach()
        .cpu()
        .tolist()
    )
    molecule = _canonical_rdkit_molecule(
        system,
        tuple(range(atom_start, atom_stop)),
        edge_indices,
    )
    boundary_atoms = _component_boundary_atoms(
        system,
        component_index,
    )
    for boundary_index, atom_index in enumerate(
        sorted(boundary_atoms),
        start=1,
    ):
        local_index = atom_index - atom_start
        molecule.GetAtomWithIdx(local_index).SetIsotope(1000 + boundary_index)
    Chem.AssignStereochemistry(molecule, force=True, cleanIt=True)

    identity = tuple(system_to_model_atom[atom_start + local_index] for local_index in visible_local_index)
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
        permutations.add(tuple(system_to_model_atom[atom_start + index] for index in mapped_local_index))
    if len(permutations) == 1:
        return None
    ordered = (identity, *sorted(permutations - {identity}))
    return identity, ordered


def _component_boundary_atoms(
    system: MolecularSystem,
    component_index: int,
) -> set[int]:
    atom_start = int(system.component_atom_ptr[component_index])
    atom_stop = int(system.component_atom_ptr[component_index + 1])
    first_inside = (system.edge_index[0] >= atom_start) & (system.edge_index[0] < atom_stop)
    second_inside = (system.edge_index[1] >= atom_start) & (system.edge_index[1] < atom_stop)
    boundary: set[int] = set()
    for first, second, first_is_inside, second_is_inside in zip(
        system.edge_index[0].detach().cpu().tolist(),
        system.edge_index[1].detach().cpu().tolist(),
        first_inside.detach().cpu().tolist(),
        second_inside.detach().cpu().tolist(),
    ):
        if first_is_inside and not second_is_inside:
            boundary.add(first)
        elif second_is_inside and not first_is_inside:
            boundary.add(second)
    return boundary


def _pocket_feature(
    pocket: Boltz1Pocket | None,
    tokens: list[dict[str, object]],
    system: MolecularSystem,
) -> Tensor:
    r"""
    Encode an optional ``Boltz1Pocket`` as a four-class ``(N, 4)`` one-hot.

    Every token of the ``binder`` instance becomes ``BINDER``; named pocket
    residues become ``POCKET``; all remaining tokens become ``UNSELECTED``.
    ``pocket=None`` yields all ``UNSPECIFIED`` tokens.
    """

    token_count = len(tokens)
    if pocket is None:
        return torch.nn.functional.one_hot(
            torch.full((token_count,), _POCKET_UNSPECIFIED, dtype=torch.long),
            _POCKET_SIZE,
        ).float()

    try:
        binder_instance = system.instance_index(pocket.binder)
    except KeyError:
        raise ValueError(f"pocket binder instance {pocket.binder!r} is not in the input system")

    pocket_components: set[int] = set()
    unresolved: set[ComponentReference] = set()
    resolved_references: list[tuple[ComponentReference, int]] = []
    for reference in pocket.pocket:
        try:
            component_index = system.component_index(reference)
        except (KeyError, IndexError):
            unresolved.add(reference)
        else:
            pocket_components.add(component_index)
            resolved_references.append((reference, component_index))
    available = {cast(int, token["component_index"]) for token in tokens}
    unresolved.update(
        reference for reference, component_index in resolved_references if component_index not in available
    )
    if unresolved:
        raise ValueError(f"pocket references do not resolve to any token: {sorted(unresolved)}")

    labels = []
    for token in tokens:
        if token["instance_index"] == binder_instance:
            labels.append(_POCKET_BINDER)
        elif token["component_index"] in pocket_components:
            labels.append(_POCKET_POCKET)
        else:
            labels.append(_POCKET_UNSELECTED)
    return torch.nn.functional.one_hot(torch.tensor(labels, dtype=torch.long), _POCKET_SIZE).float()


def _token_bond_indicator(
    system: MolecularSystem,
    system_atom_to_token: dict[int, int],
    token_count: int,
) -> Tensor:
    r"""
    Build the symmetric inter-token bond indicator ``[N, N]`` consumed by ``token_bond_projection``.
    """

    token_bonds = torch.zeros(token_count, token_count)

    def add(atom_a: int, atom_b: int) -> None:
        token_a = system_atom_to_token.get(atom_a)
        token_b = system_atom_to_token.get(atom_b)
        if token_a is None or token_b is None or token_a == token_b:
            return
        token_bonds[token_a, token_b] = token_bonds[token_b, token_a] = 1

    edge_rows = system.edge_index.detach().cpu().T.tolist()
    bond_roles = system.bond_role.detach().cpu().tolist()
    for (atom_a, atom_b), bond_role in zip(edge_rows, bond_roles):
        if bond_role == BOND_ROLE_POLYMER:
            continue
        add(atom_a, atom_b)
    return token_bonds


def _steering_features(
    system: MolecularSystem,
    system_to_model_atom: dict[int, int],
) -> dict[str, Tensor]:
    r"""
    Lower molecular geometry, assembly connections, and entity copies for Boltz-1x.
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

    instance_components = system.instance_component_ptr.detach().cpu().tolist()
    component_atoms = system.component_atom_ptr.detach().cpu().tolist()
    instance_entities = system.instance_entity.detach().cpu().tolist()
    entity_types = system.entity_type.detach().cpu().tolist()
    component_symbols = system.component_symbol.detach().cpu().tolist()
    for instance_index, entity_index in enumerate(instance_entities):
        component_start = instance_components[instance_index]
        component_stop = instance_components[instance_index + 1]
        entity_type = entity_types[entity_index]
        if entity_type != ENTITY_MOLECULE:
            for component_index in range(component_start, component_stop):
                component_id = system.symbols[component_symbols[component_index]]
                polymer_type = _component_polymer_type(
                    entity_type,
                    component_id,
                )
                if polymer_type is not None and component_id in _CONSTRAINT_FREE_COMPONENTS[polymer_type]:
                    continue
                if not bool(system.component_topology_complete[component_index]):
                    continue
                atom_start = component_atoms[component_index]
                atom_stop = component_atoms[component_index + 1]
                edge_indices = (
                    (
                        (system.edge_index[0] >= atom_start)
                        & (system.edge_index[1] < atom_stop)
                        & (system.bond_role == BOND_ROLE_COMPONENT)
                    )
                    .nonzero(as_tuple=False)
                    .flatten()
                    .detach()
                    .cpu()
                    .tolist()
                )
                rdkit_molecule = _canonical_rdkit_molecule(
                    system,
                    tuple(range(atom_start, atom_stop)),
                    edge_indices,
                )
                component_atom_index = {
                    atom_index - atom_start: system_to_model_atom[atom_index]
                    for atom_index in range(atom_start, atom_stop)
                    if atom_index in system_to_model_atom
                }
                constraint_scopes.append((rdkit_molecule, component_atom_index))
        else:
            if not bool(system.component_topology_complete[component_start:component_stop].all()):
                continue
            atom_start = component_atoms[component_start]
            atom_stop = component_atoms[component_stop]
            edge_indices = (
                (
                    (system.edge_index[0] >= atom_start)
                    & (system.edge_index[1] < atom_stop)
                    & (system.bond_role <= BOND_ROLE_ENTITY)
                )
                .nonzero(as_tuple=False)
                .flatten()
                .detach()
                .cpu()
                .tolist()
            )
            rdkit_molecule = _canonical_rdkit_molecule(
                system,
                tuple(range(atom_start, atom_stop)),
                edge_indices,
            )
            molecule_atom_index = {
                atom_index - atom_start: system_to_model_atom[atom_index]
                for atom_index in range(atom_start, atom_stop)
                if atom_index in system_to_model_atom
            }
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

    atom_component = _atom_component_index(system)
    component_instance = torch.repeat_interleave(
        torch.arange(
            system.instance_symbol.numel(),
            device=system.instance_component_ptr.device,
        ),
        system.instance_component_ptr[1:] - system.instance_component_ptr[:-1],
    )
    atom_components = atom_component.detach().cpu().tolist()
    component_instances = component_instance.detach().cpu().tolist()
    connections = (system.bond_role == BOND_ROLE_ASSEMBLY).nonzero(as_tuple=False).flatten().detach().cpu().tolist()
    connection_edges = system.edge_index[:, connections].detach().cpu().T.tolist() if connections else []
    connected_chains = []
    connected_atoms = []
    for first, second in connection_edges:
        if first not in system_to_model_atom or second not in system_to_model_atom:
            continue
        connected_chains.append(
            (
                component_instances[atom_components[first]],
                component_instances[atom_components[second]],
            )
        )
        connected_atoms.append(
            (
                system_to_model_atom[first],
                system_to_model_atom[second],
            )
        )

    symmetric_chains = [
        (first_index, second_index)
        for first_index, first_entity in enumerate(instance_entities)
        for second_index, second_entity in enumerate(instance_entities)
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
                indices = (start_neighbors[1], start_index, end_index, end_neighbors[1])
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


def _msa_features(
    msa: MSA | MSACollection | None,
    resolved_bindings: tuple[tuple[MSA, int | None, int], ...],
    tokens: list[dict[str, object]],
    system: MolecularSystem,
) -> dict[str, Tensor]:
    residue_types = [cast(int, token["residue_type"]) for token in tokens]
    if msa is None:
        rows = torch.tensor([residue_types])
        deletion_counts = torch.zeros(1, len(tokens))
        paired = torch.ones(1, len(tokens))
    else:
        # The MSA holds no PolymerSequence: column ``c`` binds ``columns[c].entity_id`` to a system entity,
        # a token's monomer index is directly the aligned-row column, and cells are UPPERCASE one-letter
        # symbols (or ``-``). A bare MSA is a single-chain plan over its unpaired pool; a collection replays
        # its dense ``row_pairing`` and ``row_is_paired`` plan across the paired and unpaired pools.
        columns = tuple(value for value, _target_instance_index, _entity_index in resolved_bindings)
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
            _POLYMER_TYPE.get(
                int(system.entity_type[entity_index]),
                "dna",
            )
            for _value, _target_instance_index, entity_index in resolved_bindings
        )
        row_values: list[Tensor] = []
        deletion_values: list[Tensor] = []
        paired_values: list[Tensor] = []
        for selections in plan_rows:
            row = torch.full((len(tokens),), _MSA_GAP, dtype=torch.long)
            deletion_row = torch.zeros(len(tokens))
            paired_row = torch.zeros(len(tokens))
            present = len(selections)
            taxonomic_pair = (
                isinstance(msa, MSACollection) and present > 1 and all(is_paired for _, _, is_paired in selections)
            )
            for column_index, source_row, is_paired in selections:
                value = columns[column_index]
                _value, target_instance_index, entity_index = resolved_bindings[column_index]
                polymer_type = polymer_types[column_index]
                aligned = value.paired_sequences[source_row] if is_paired else value.unpaired_sequences[source_row]
                deletion_source = (value.paired_deletion_counts if is_paired else value.unpaired_deletion_counts)[
                    source_row
                ]
                for token_index, token in enumerate(tokens):
                    if (
                        token["entity_index"] != entity_index
                        or (target_instance_index is not None and token["instance_index"] != target_instance_index)
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


def _reference_coordinates(
    system: MolecularSystem,
    atom_indices: tuple[int, ...],
) -> Tensor:
    index = torch.tensor(atom_indices, device=system.atom_symbol.device)
    coordinates = system.reference_coordinates[index].detach().cpu().clone()
    reference_mask = system.reference_coordinate_mask[index].detach().cpu()
    if bool(reference_mask.all()):
        return coordinates

    atom_component = _atom_component_index(system)
    component_instance = torch.repeat_interleave(
        torch.arange(
            system.instance_symbol.numel(),
            device=system.instance_component_ptr.device,
        ),
        system.instance_component_ptr[1:] - system.instance_component_ptr[:-1],
    )
    atom_instance = component_instance[atom_component].detach().cpu().tolist()
    instance_entities = system.instance_entity.detach().cpu().tolist()
    entity_types = system.entity_type.detach().cpu().tolist()
    generated: dict[int, dict[int, tuple[float, float, float]]] = {}
    for model_atom_index, canonical_atom_index in enumerate(atom_indices):
        if bool(reference_mask[model_atom_index]):
            continue
        instance_index = atom_instance[canonical_atom_index]
        entity_index = instance_entities[instance_index]
        if entity_types[entity_index] != ENTITY_MOLECULE:
            raise ValueError("Boltz-1 requires reference coordinates for canonical atom " f"{canonical_atom_index}")
        if instance_index not in generated:
            generated[instance_index] = _molecule_reference_coordinates(
                system,
                instance_index,
            )
        coordinates[model_atom_index] = coordinates.new_tensor(generated[instance_index][canonical_atom_index])
    return coordinates


def _molecule_reference_coordinates(
    system: MolecularSystem,
    instance_index: int,
) -> dict[int, tuple[float, float, float]]:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    component_start = int(system.instance_component_ptr[instance_index])
    component_stop = int(system.instance_component_ptr[instance_index + 1])
    atom_start = int(system.component_atom_ptr[component_start])
    atom_stop = int(system.component_atom_ptr[component_stop])
    atom_indices = tuple(range(atom_start, atom_stop))
    if not bool(system.component_topology_complete[component_start:component_stop].all()):
        raise ValueError("Boltz-1 cannot generate reference coordinates from unknown " "molecular connectivity")
    edge_indices = (
        (
            (system.edge_index[0] >= atom_start)
            & (system.edge_index[1] < atom_stop)
            & (system.bond_role <= BOND_ROLE_ENTITY)
        )
        .nonzero(as_tuple=False)
        .flatten()
        .detach()
        .cpu()
        .tolist()
    )
    rdkit_molecule = Chem.AddHs(
        _canonical_rdkit_molecule(
            system,
            atom_indices,
            edge_indices,
        )
    )
    options = AllChem.ETKDGv3()
    options.clearConfs = False
    options.randomSeed = 42
    conformer_id = AllChem.EmbedMolecule(rdkit_molecule, options)
    if conformer_id == -1:
        options.useRandomCoords = True
        conformer_id = AllChem.EmbedMolecule(rdkit_molecule, options)
    AllChem.UFFOptimizeMolecule(rdkit_molecule, confId=conformer_id, maxIters=1000)
    conformer = rdkit_molecule.GetConformer(conformer_id)
    coordinates: dict[int, tuple[float, float, float]] = {}
    for local_index, atom_index in enumerate(atom_indices):
        position = conformer.GetAtomPosition(local_index)
        coordinates[atom_index] = (position.x, position.y, position.z)
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
    active_space_ids = torch.unique(space_ids[resolved & (space_ids >= 0)], sorted=True)
    num_active_spaces = active_space_ids.numel()
    if not num_active_spaces:
        return

    random_device = positions.device if generator is None else generator.device
    random_values = torch.randn(
        num_active_spaces,
        7,
        dtype=positions.dtype,
        device=random_device,
        generator=generator,
    ).to(positions.device)
    quaternions = random_values[:, :4]
    norm = torch.sqrt((quaternions * quaternions).sum(dim=-1, keepdim=True))
    quaternions = quaternions / torch.copysign(norm, quaternions[:, :1])
    real, i, j, k = quaternions.unbind(dim=-1)
    two = 2.0 / (quaternions * quaternions).sum(dim=-1)
    rotations = torch.stack(
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
        ),
        dim=-1,
    ).reshape(-1, 3, 3)
    translations = random_values[:, 4:]

    for space_id, rotation, translation in zip(active_space_ids, rotations, translations):
        included = space_ids == space_id
        mask = resolved[included]
        values = positions[included]
        center = (values * mask[:, None]).sum(dim=0, keepdim=True) / mask.sum()
        positions[included] = (values - center) @ rotation + translation


def _model_atom_names(
    system: MolecularSystem,
    atom_indices: tuple[int, ...],
) -> list[str]:
    from rdkit.Chem import GetPeriodicTable

    atom_component = _atom_component_index(system)
    component_instance = torch.repeat_interleave(
        torch.arange(
            system.instance_symbol.numel(),
            device=system.instance_component_ptr.device,
        ),
        system.instance_component_ptr[1:] - system.instance_component_ptr[:-1],
    )
    atom_instance = component_instance[atom_component].detach().cpu().tolist()
    atom_symbols = system.atom_symbol.detach().cpu().tolist()
    atomic_numbers = system.atomic_number.detach().cpu().tolist()
    periodic_table = GetPeriodicTable()
    names: list[str] = []
    counts: dict[tuple[int, str], int] = {}
    for atom_index in atom_indices:
        name = system.symbols[atom_symbols[atom_index]]
        if len(name) > 4 or any(not 32 <= ord(character) < 96 for character in name):
            element = periodic_table.GetElementSymbol(atomic_numbers[atom_index]).upper()
            key = (atom_instance[atom_index], element)
            counts[key] = counts.get(key, 0) + 1
            name = f"{element}{counts[key]}"
        if len(name) > 4:
            raise ValueError(f"Boltz-1 atom name cannot be encoded in four characters: {name!r}")
        names.append(name)
    return names


def _canonical_rdkit_molecule(
    system: MolecularSystem,
    atom_indices: tuple[int, ...],
    edge_indices: Sequence[int],
) -> Mol:
    r"""
    Project one family-owned canonical graph scope into RDKit.
    """

    from rdkit import Chem

    editable = Chem.RWMol()
    selected_atoms = torch.tensor(
        atom_indices,
        device=system.atom_symbol.device,
    )
    atomic_numbers = system.atomic_number[selected_atoms].detach().cpu().tolist()
    isotope = system.isotope[selected_atoms].detach().cpu().tolist()
    isotope_known = system.isotope_known[selected_atoms].detach().cpu().tolist()
    formal_charge = system.formal_charge[selected_atoms].detach().cpu().tolist()
    formal_charge_known = system.formal_charge_known[selected_atoms].detach().cpu().tolist()
    radical_electrons = system.radical_electrons[selected_atoms].detach().cpu().tolist()
    radical_electrons_known = system.radical_electrons_known[selected_atoms].detach().cpu().tolist()
    explicit_hydrogens = system.explicit_hydrogens[selected_atoms].detach().cpu().tolist()
    explicit_hydrogens_known = system.explicit_hydrogens_known[selected_atoms].detach().cpu().tolist()
    implicit_hydrogens_allowed = system.implicit_hydrogens_allowed[selected_atoms].detach().cpu().tolist()
    implicit_hydrogens_allowed_known = system.implicit_hydrogens_allowed_known[selected_atoms].detach().cpu().tolist()
    atom_aromatic = system.atom_aromatic[selected_atoms].detach().cpu().tolist()
    atom_aromatic_known = system.atom_aromatic_known[selected_atoms].detach().cpu().tolist()
    atom_stereo_values = system.atom_stereo[selected_atoms].detach().cpu().tolist()
    canonical_to_local: dict[int, int] = {}
    atom_stereo: dict[int, str] = {}
    for source_index, atom_index in enumerate(atom_indices):
        rdkit_atom = Chem.Atom(atomic_numbers[source_index])
        if isotope_known[source_index]:
            rdkit_atom.SetIsotope(isotope[source_index])
        if formal_charge_known[source_index]:
            rdkit_atom.SetFormalCharge(formal_charge[source_index])
        if radical_electrons_known[source_index]:
            rdkit_atom.SetNumRadicalElectrons(radical_electrons[source_index])
        if explicit_hydrogens_known[source_index]:
            rdkit_atom.SetNumExplicitHs(explicit_hydrogens[source_index])
        if implicit_hydrogens_allowed_known[source_index]:
            rdkit_atom.SetNoImplicit(not implicit_hydrogens_allowed[source_index])
        if atom_aromatic_known[source_index]:
            rdkit_atom.SetIsAromatic(atom_aromatic[source_index])
        local_index = editable.AddAtom(rdkit_atom)
        canonical_to_local[atom_index] = local_index
        stereo_index = atom_stereo_values[source_index]
        if stereo_index >= 0:
            atom_stereo[local_index] = system.symbols[stereo_index].upper()

    bond_types = {
        1.0: Chem.BondType.SINGLE,
        1.5: Chem.BondType.AROMATIC,
        2.0: Chem.BondType.DOUBLE,
        3.0: Chem.BondType.TRIPLE,
        4.0: Chem.BondType.QUADRUPLE,
    }
    selected_edges = torch.tensor(
        edge_indices,
        dtype=torch.long,
        device=system.edge_index.device,
    )
    edge_rows = system.edge_index[:, selected_edges].detach().cpu().T.tolist()
    bond_order = system.bond_order[selected_edges].detach().cpu().tolist()
    bond_order_known = system.bond_order_known[selected_edges].detach().cpu().tolist()
    bond_aromatic = system.bond_aromatic[selected_edges].detach().cpu().tolist()
    bond_aromatic_known = system.bond_aromatic_known[selected_edges].detach().cpu().tolist()
    bond_stereo_values = system.bond_stereo[selected_edges].detach().cpu().tolist()
    bond_stereo_index = system.bond_stereo_index[:, selected_edges].detach().cpu().T.tolist()
    bond_stereo: list[tuple[int, str, tuple[int, int] | None]] = []
    for source_index, (first, second) in enumerate(edge_rows):
        if not bond_order_known[source_index]:
            raise ValueError("Boltz-1 RDKit geometry requires known bond order")
        editable.AddBond(
            canonical_to_local[first],
            canonical_to_local[second],
            bond_types[bond_order[source_index]],
        )
        bond_index = editable.GetNumBonds() - 1
        rdkit_bond = editable.GetBondWithIdx(bond_index)
        if bond_aromatic_known[source_index]:
            rdkit_bond.SetIsAromatic(bond_aromatic[source_index])
        stereo_index = bond_stereo_values[source_index]
        if stereo_index >= 0:
            first_stereo, second_stereo = bond_stereo_index[source_index]
            stereo_atoms = (
                (
                    canonical_to_local[first_stereo],
                    canonical_to_local[second_stereo],
                )
                if first_stereo >= 0 and second_stereo >= 0
                else None
            )
            bond_stereo.append(
                (
                    bond_index,
                    system.symbols[stereo_index],
                    stereo_atoms,
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


def _cyclic_periods(system: MolecularSystem) -> list[int]:
    atom_component = _atom_component_index(system).detach().cpu().tolist()
    edge_rows = system.edge_index.detach().cpu().T.tolist()
    instance_components = system.instance_component_ptr.detach().cpu().tolist()
    instance_entities = system.instance_entity.detach().cpu().tolist()
    entity_types = system.entity_type.detach().cpu().tolist()
    periods: list[int] = []
    for instance_index, entity_index in enumerate(instance_entities):
        component_start = instance_components[instance_index]
        component_stop = instance_components[instance_index + 1]
        component_count = component_stop - component_start
        if entity_types[entity_index] == ENTITY_MOLECULE or component_count <= 2:
            periods.append(0)
            continue
        cyclic = any(
            {
                atom_component[first],
                atom_component[second],
            }
            == {component_start, component_stop - 1}
            for first, second in edge_rows
        )
        periods.append(component_count if cyclic else 0)
    return periods


def _component_polymer_type(
    entity_type: int,
    component_id: str,
) -> str | None:
    polymer_type = _POLYMER_TYPE.get(entity_type)
    if polymer_type is not None:
        return polymer_type
    if entity_type == ENTITY_DNA_RNA_HYBRID:
        if component_id in _STANDARD_COMPONENTS["rna"]:
            return "rna"
        if component_id in _STANDARD_COMPONENTS["dna"]:
            return "dna"
    return None


def _residue_type(component_id: str, polymer_type: str) -> int:
    if polymer_type == "protein":
        return _PROTEIN_RESIDUE_TYPES.get(component_id, 22)
    if polymer_type == "rna":
        return _RNA_RESIDUE_TYPES.get(component_id, 27)
    return _DNA_RESIDUE_TYPES.get(component_id, 32)
