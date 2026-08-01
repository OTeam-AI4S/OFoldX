r"""
Prepare biomolecular features for Protenix.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from ...data import (
    MSA_VOCAB_SIZE,
    TOKEN_GAP_INDEX,
    TOKEN_INDEX,
    TOKEN_VOCAB_SIZE,
    Cropper,
    TokenLayout,
    component_token_index,
    crop_token_layout,
    encode_msa_sequences,
    resolved_token_mask,
    tokenize_molecular_system,
    water_component_mask,
)
from ...data.chemistry import to_rdkit_component
from ...data.evidence import (
    MSA,
    MsaCollection,
    Template,
    TemplateAlignment,
    resolve_msa_bindings,
)
from ...data.molecular import (
    BOND_ROLE_ENTITY,
    ENTITY_DNA,
    ENTITY_DNA_RNA_HYBRID,
    ENTITY_MOLECULE,
    ENTITY_PNA,
    ENTITY_PROTEIN,
    ENTITY_RNA,
    AtomSite,
    ComponentReference,
    MolecularSystem,
    Molecule,
    PolymerSequence,
)
from ...data.structure import Structure
from ...modules.geometry import uniform_random_rotation
from ...processor import Processor
from .config import ProtenixProcessorConfig
from .model import (
    ProtenixAtomContact,
    ProtenixInput,
    ProtenixPocket,
    ProtenixStructureTarget,
    ProtenixTemplateFeatures,
    ProtenixTfgFeatures,
    ProtenixTokenContact,
)

_STANDARD = {
    "protein": frozenset(
        {
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
        }
    ),
    "rna": frozenset(("A", "G", "C", "U", "N")),
    "dna": frozenset(("DA", "DG", "DC", "DT", "DN")),
}
_STANDARD_BY_ENTITY_TYPE = {
    ENTITY_PROTEIN: _STANDARD["protein"],
    ENTITY_RNA: _STANDARD["rna"],
    ENTITY_DNA: _STANDARD["dna"],
    ENTITY_DNA_RNA_HYBRID: _STANDARD["rna"] | _STANDARD["dna"],
    ENTITY_PNA: frozenset(),
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
_POLYMER_TYPE = {
    ENTITY_PROTEIN: "protein",
    ENTITY_RNA: "rna",
    ENTITY_DNA: "dna",
    ENTITY_DNA_RNA_HYBRID: "dna_rna_hybrid",
    ENTITY_PNA: "pna",
}


class ProtenixProcessor(Processor):
    r"""
    Convert supported biomolecular inputs into `ProtenixInput`.

    Args:
        config: Feature limits and reference-position options. Defaults to
            ``ProtenixProcessorConfig()``.
        device: Device for produced tensors.
        dtype: Floating-point dtype for produced tensors.

    Examples:
        >>> from ofoldx.data import ChemicalAtom, ChemicalComponent, MolecularSystem, Molecule
        >>> atom = ChemicalAtom("C", "C", formal_charge=0)
        >>> first = ChemicalComponent(
        ...     "L1", (atom,), frozenset(), reference_coordinates=((0.0, 0.0, 0.0),)
        ... )
        >>> second = ChemicalComponent(
        ...     "L2", (atom,), frozenset(), reference_coordinates=((1.0, 0.0, 0.0),)
        ... )
        >>> system = MolecularSystem.from_entities(
        ...     {"ligand": Molecule((first, second))}, {"L": "ligand"}
        ... )
        >>> config = ProtenixProcessorConfig(augment_reference_positions=False)
        >>> inputs = ProtenixProcessor(config)(system)
        >>> inputs.residue_index.tolist(), inputs.reference_space_id.tolist()
        ([[0, 1]], [[0, 1]])
        >>> from ofoldx.data import TOKEN_INDEX, TOKEN_VOCAB_SIZE
        >>> inputs.token_features.shape[-1] == 384 + 2 * TOKEN_VOCAB_SIZE + 1
        True
        >>> inputs.token_features[0, 0, 384 + TOKEN_INDEX["OTHER"]].item()
        1.0

        A ligand deposited without any reference conformer still processes: its known
        connectivity is embedded to supply the missing reference feature, and the
        canonical graph keeps saying that it has none.

        >>> from ofoldx.data import ChemicalBond
        >>> ligand = ChemicalComponent(
        ...     "L3",
        ...     (ChemicalAtom("C1", "C"), ChemicalAtom("C2", "C")),
        ...     frozenset({ChemicalBond("C1", "C2")}),
        ... )
        >>> observed = MolecularSystem.from_entities(
        ...     {"ligand": Molecule((ligand,))}, {"L": "ligand"}
        ... )
        >>> inputs = ProtenixProcessor(config)(observed)
        >>> bool(inputs.reference_mask.all()), bool(observed.reference_coordinate_mask.any())
        (True, False)
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
        msa: MSA | MsaCollection | None = None,
        prepare_tfg_features: bool = False,
        generator: torch.Generator | None = None,
    ) -> ProtenixInput:
        r"""
        Prepare model features for one molecular assembly.

        Args:
            input: One or more ``PolymerSequence`` values, ``MolecularSystem``,
                or ``Structure``. A structure contributes topology, not observed
                coordinates.
            msa: Optional ``MSA`` or ``MsaCollection``.
            prepare_tfg_features: Whether to prepare the static molecular geometry
                required by training-free guidance.
            generator: Optional PyTorch generator for reference-position augmentation.

        Returns:
            Batched token, atom, and MSA tensors. Reference coordinates are in
            angstroms. ``atom_index`` and ``token_component_index`` bind the
            family axes to the canonical molecular graph.
        """

        return self._lower(
            input,
            msa=msa,
            prepare_tfg_features=prepare_tfg_features,
            generator=generator,
        )

    def process_training(
        self,
        input: Structure,
        *,
        msa: MSA | MsaCollection | None = None,
        cropper: Cropper | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[ProtenixInput, ProtenixStructureTarget]:
        r"""
        Lower one structure into model input and separate coordinate truth.
        """

        processed = self._lower(
            input,
            msa=msa,
            cropper=cropper,
            generator=generator,
        )
        return processed, _structure_target(
            input,
            device=self.device,
            dtype=self.dtype,
        )

    def _resolve_msa(
        self,
        msa: MSA | MsaCollection | None,
        system: MolecularSystem,
    ) -> tuple[tuple[MSA, int | None, int], ...]:
        if msa is None:
            return ()
        resolved = resolve_msa_bindings(msa, system)
        for _value, _target_instance_index, entity_index in resolved:
            if int(system.entity_type[entity_index]) not in {ENTITY_PROTEIN, ENTITY_RNA}:
                raise ValueError("Protenix accepts supplied MSA evidence only for proteins and RNA")
        return resolved

    def _lower(
        self,
        input: PolymerSequence | Sequence[PolymerSequence] | MolecularSystem | Structure,
        *,
        msa: MSA | MsaCollection | None,
        templates: tuple[Template, ...] | None = None,
        atom_contacts: Sequence[ProtenixAtomContact] = (),
        token_contacts: Sequence[ProtenixTokenContact] = (),
        pocket: ProtenixPocket | None = None,
        substructure: Structure | None = None,
        substructure_coordinate_noise_scale: float = 0.0,
        structure_condition: Structure | None = None,
        prepare_tfg_features: bool = False,
        cropper: Cropper | None = None,
        generator: torch.Generator | None = None,
    ) -> ProtenixInput:
        atom_contact_values = tuple(atom_contacts)
        token_contact_values = tuple(token_contacts)
        system = molecular_system_from_input(input)
        resolved_msa = self._resolve_msa(msa, system)
        # Atom contacts require atom-level tokens. Token contacts and pockets use
        # the tokenization already selected for each residue.
        expanded_residues = _atom_constraint_residues(atom_contact_values, system)
        layout = protenix_token_layout(system, expanded_residues)
        if cropper is not None:
            layout = crop_token_layout(
                cast(Structure, input),
                layout,
                cropper,
                required_token_mask=_constraint_token_mask(
                    layout,
                    system,
                    atom_contact_values,
                    token_contact_values,
                    pocket,
                ),
                anchor_token_mask=resolved_token_mask(cast(Structure, input), layout),
                generator=generator,
            )
        tokens = _lower_token_layout(system, layout)
        return _features(
            system,
            tokens=tokens,
            msa=msa,
            msa_bindings=resolved_msa,
            templates=templates,
            atom_contacts=atom_contact_values,
            token_contacts=token_contact_values,
            pocket=pocket,
            substructure=substructure,
            substructure_coordinate_noise_scale=substructure_coordinate_noise_scale,
            structure_condition=structure_condition,
            prepare_tfg_features=prepare_tfg_features,
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
        msa: MSA | MsaCollection | None = None,
        templates: Sequence[Template] = (),
        prepare_tfg_features: bool = False,
        generator: torch.Generator | None = None,
    ) -> ProtenixInput:
        r"""
        Prepare protein/RNA MSA and fixed-four protein-template features.

        Args:
            input: One or more ``PolymerSequence`` values, ``MolecularSystem``,
                or ``Structure``.
            msa: Optional protein and unpaired-RNA MSA evidence.
            templates: Protein templates aligned to target chains.
            prepare_tfg_features: Whether to prepare the static molecular geometry
                required by training-free guidance.
            generator: Optional generator for reference-position augmentation.

        Returns:
            Batched token, atom, MSA, and fixed-four template tensors.
        """

        return self._lower(
            input,
            msa=msa,
            templates=tuple(templates),
            prepare_tfg_features=prepare_tfg_features,
            generator=generator,
        )

    def process_training(
        self,
        input: Structure,
        *,
        msa: MSA | MsaCollection | None = None,
        templates: Sequence[Template] = (),
        cropper: Cropper | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[ProtenixInput, ProtenixStructureTarget]:
        r"""
        Lower template-capable input and separate coordinate truth.
        """

        processed = self._lower(
            input,
            msa=msa,
            templates=tuple(templates),
            cropper=cropper,
            generator=generator,
        )
        return processed, _structure_target(
            input,
            device=self.device,
            dtype=self.dtype,
        )

    def _resolve_msa(
        self,
        msa: MSA | MsaCollection | None,
        system: MolecularSystem,
    ) -> tuple[tuple[MSA, int | None, int], ...]:
        if msa is None:
            return ()
        resolved = resolve_msa_bindings(msa, system)
        for _value, _target_instance_index, entity_index in resolved:
            if int(system.entity_type[entity_index]) not in {ENTITY_PROTEIN, ENTITY_RNA}:
                raise ValueError("template-capable Protenix accepts supplied MSA evidence only for proteins and RNA")
        return resolved


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
        msa: MSA | MsaCollection | None = None,
        prepare_tfg_features: bool = False,
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
            prepare_tfg_features: Whether to prepare the static molecular geometry
                required by training-free guidance.
            generator: Optional generator for reference-position augmentation.

        Returns:
            Batched Protenix tensors with representative coordinates and their
            validity mask aligned to the token axis.
        """

        return self._lower(
            input,
            msa=msa,
            structure_condition=structure_condition,
            prepare_tfg_features=prepare_tfg_features,
            generator=generator,
        )

    def process_training(
        self,
        input: Structure,
        *,
        structure_condition: Structure | None = None,
        msa: MSA | MsaCollection | None = None,
        cropper: Cropper | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[ProtenixInput, ProtenixStructureTarget]:
        r"""
        Lower structure-conditioned input and separate coordinate truth.
        """

        processed = self._lower(
            input,
            structure_condition=structure_condition,
            msa=msa,
            cropper=cropper,
            generator=generator,
        )
        return processed, _structure_target(
            input,
            device=self.device,
            dtype=self.dtype,
        )


class ProtenixConstraintProcessor(ProtenixProcessor):
    r"""
    Convert biomolecular inputs and spatial constraints into `ProtenixInput`.

    The config (``ProtenixProcessorConfig``), device, and dtype are inherited from `ProtenixProcessor`; this
    variant differs only in the spatial-constraint inputs its ``__call__`` accepts.
    """

    def __call__(
        self,
        input: PolymerSequence | Sequence[PolymerSequence] | MolecularSystem | Structure,
        *,
        msa: MSA | MsaCollection | None = None,
        atom_contacts: Sequence[ProtenixAtomContact] = (),
        token_contacts: Sequence[ProtenixTokenContact] = (),
        pocket: ProtenixPocket | None = None,
        substructure: Structure | None = None,
        substructure_coordinate_noise_scale: float = 0.0,
        prepare_tfg_features: bool = False,
        generator: torch.Generator | None = None,
    ) -> ProtenixInput:
        r"""
        Prepare model features and spatial constraints for one assembly.

        Args:
            input: One or more ``PolymerSequence`` values, ``MolecularSystem``,
                or ``Structure``. A structure contributes topology, not observed
                coordinates.
            msa: Optional ``MSA`` or ``MsaCollection``.
            atom_contacts: Upper-bound constraints between exact atom pairs.
            token_contacts: Upper-bound constraints between token-level endpoints.
            pocket: Optional directed binder-to-residue pocket constraint.
            substructure: Optional resolved token-center coordinates used for the
                released four-channel substructure condition.
            substructure_coordinate_noise_scale: Standard deviation of Gaussian
                noise added to selected center coordinates before distance binning.
            prepare_tfg_features: Whether to prepare the static molecular geometry
                required by training-free guidance.
            generator: Optional generator for reference-position augmentation
                and random selection among multi-token constraint endpoints.

        Returns:
            Batched token, atom, MSA, and constraint tensors. Reference
            coordinates and distance bounds are in angstroms. Canonical integer
            indices bind the atom and token axes to the molecular graph.
        """

        return self._lower(
            input,
            msa=msa,
            atom_contacts=atom_contacts,
            token_contacts=token_contacts,
            pocket=pocket,
            substructure=substructure,
            substructure_coordinate_noise_scale=substructure_coordinate_noise_scale,
            prepare_tfg_features=prepare_tfg_features,
            generator=generator,
        )

    def process_training(
        self,
        input: Structure,
        *,
        msa: MSA | MsaCollection | None = None,
        atom_contacts: Sequence[ProtenixAtomContact] = (),
        token_contacts: Sequence[ProtenixTokenContact] = (),
        pocket: ProtenixPocket | None = None,
        substructure: Structure | None = None,
        substructure_coordinate_noise_scale: float = 0.0,
        cropper: Cropper | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[ProtenixInput, ProtenixStructureTarget]:
        r"""
        Lower constrained input and separate coordinate truth.
        """

        processed = self._lower(
            input,
            msa=msa,
            atom_contacts=atom_contacts,
            token_contacts=token_contacts,
            pocket=pocket,
            substructure=substructure,
            substructure_coordinate_noise_scale=substructure_coordinate_noise_scale,
            cropper=cropper,
            generator=generator,
        )
        return processed, _structure_target(
            input,
            device=self.device,
            dtype=self.dtype,
        )


def _structure_target(
    structure: Structure,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> ProtenixStructureTarget:
    r"""
    Move full-axis canonical coordinate truth to the model placement.
    """

    unique_resolutions = torch.unique(structure.resolution)
    if unique_resolutions.numel() > 1:
        raise ValueError("Protenix requires a single experimental-resolution value")
    if unique_resolutions.numel():
        resolution = unique_resolutions
        resolution_mask = torch.ones(1, dtype=torch.bool, device=resolution.device)
    else:
        resolution = structure.resolution.new_tensor([-1.0])
        resolution_mask = torch.zeros(1, dtype=torch.bool, device=resolution.device)

    return ProtenixStructureTarget(  # type: ignore[call-arg]
        coordinates=structure.coordinates.to(device=device, dtype=dtype).unsqueeze(0),
        coordinate_mask=structure.coordinate_mask.to(device=device).unsqueeze(0),
        resolution=resolution.to(device=device, dtype=torch.float32),
        resolution_mask=resolution_mask.to(device=device),
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
        The canonical molecular system. Protenix-visible axes exclude water
        during tokenization without creating a second topology.
    """

    # A Structure supplied as the primary input contributes only topology. The
    # constraint Processor lowers coordinates only from its explicit substructure argument.
    if isinstance(input, Structure):
        return input.system
    elif isinstance(input, MolecularSystem):
        return input
    elif isinstance(input, PolymerSequence):
        return molecular_system_from_named_sequences((("A", input),))
    elif isinstance(input, Sequence):
        sequence_values = tuple(cast(Sequence[object], input))
        if not sequence_values or not all(isinstance(sequence, PolymerSequence) for sequence in sequence_values):
            raise TypeError("Protenix sequence collections must contain PolymerSequence values")
        polymer_sequences = cast(tuple[PolymerSequence, ...], sequence_values)
        return molecular_system_from_named_sequences(
            tuple((f"chain_{index + 1}", sequence) for index, sequence in enumerate(polymer_sequences))
        )
    raise TypeError(
        "Protenix input must be a PolymerSequence, sequence of PolymerSequence values, MolecularSystem, or Structure"
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
    return MolecularSystem.from_entities(entities, instances)


def _atom_constraint_residues(
    atom_contacts: tuple[ProtenixAtomContact, ...],
    system: MolecularSystem,
) -> frozenset[int]:
    # Every polymer residue named by an atom contact must lower to per-atom tokens so the two
    # endpoints occupy distinct atom-level tokens. Non-polymer atoms are already per-atom tokens.
    residues: set[int] = set()
    atom_component = torch.repeat_interleave(
        torch.arange(
            system.component_symbol.numel(),
            device=system.component_atom_ptr.device,
        ),
        system.component_atom_ptr[1:] - system.component_atom_ptr[:-1],
    )
    component_instance = torch.repeat_interleave(
        torch.arange(
            system.instance_symbol.numel(),
            device=system.instance_component_ptr.device,
        ),
        system.instance_component_ptr[1:] - system.instance_component_ptr[:-1],
    )
    for contact in atom_contacts:
        for site in (contact.atom_a, contact.atom_b):
            atom_index = system.atom_index(site)
            component_index = int(atom_component[atom_index])
            instance_index = int(component_instance[component_index])
            entity_index = int(system.instance_entity[instance_index])
            if int(system.entity_type[entity_index]) != ENTITY_MOLECULE:
                residues.add(component_index)
    return frozenset(residues)


def _is_standard(system: MolecularSystem, component_index: int, entity_type: int) -> bool:
    component_id = system.symbols[int(system.component_symbol[component_index])]
    return component_id in _STANDARD_BY_ENTITY_TYPE.get(
        entity_type,
        frozenset(),
    )


def protenix_token_layout(
    system: MolecularSystem,
    expanded_components: frozenset[int] = frozenset(),
) -> TokenLayout:
    r"""Return the canonical token layout consumed by Protenix processors."""

    component_atoms = system.component_atom_ptr.detach().cpu().tolist()
    component_symbols = system.component_symbol.detach().cpu().tolist()
    atom_symbols = system.atom_symbol.detach().cpu().tolist()
    atom_mask = torch.ones_like(system.atomic_number, dtype=torch.bool)
    for component_index, component_symbol in enumerate(component_symbols):
        if system.symbols[component_symbol] not in {"N", "DN"}:
            continue
        for atom_index in range(component_atoms[component_index], component_atoms[component_index + 1]):
            if system.symbols[atom_symbols[atom_index]] == "N":
                atom_mask[atom_index] = False
    return tokenize_molecular_system(
        system,
        expanded_components=expanded_components,
        component_mask=~water_component_mask(system),
        atom_mask=atom_mask,
    )


def _lower_token_layout(system: MolecularSystem, layout: TokenLayout) -> list[dict[str, object]]:
    tokens: list[dict[str, object]] = []
    entity_types = system.entity_type.detach().cpu().tolist()
    component_symbols = system.component_symbol.detach().cpu().tolist()
    atom_symbols = system.atom_symbol.detach().cpu().tolist()
    token_count = int(layout.token_component_index.numel())
    for token_index in range(token_count):
        component_index = int(layout.token_component_index[token_index])
        instance_index = int(layout.token_instance_index[token_index])
        entity_index = int(layout.token_entity_index[token_index])
        entity_type = int(entity_types[entity_index])
        component_id = system.symbols[component_symbols[component_index]]
        start = int(layout.token_atom_ptr[token_index])
        stop = int(layout.token_atom_ptr[token_index + 1])
        group = tuple(layout.atom_index[start:stop].detach().cpu().tolist())
        atom_index = int(layout.token_atom_index[token_index])
        atom_token = atom_index >= 0
        standard = entity_type != ENTITY_MOLECULE and _is_standard(system, component_index, entity_type)
        center = int(layout.token_center_atom_index[token_index])
        if atom_token:
            representative = atom_index
        else:
            by_name = {system.symbols[atom_symbols[index]]: index for index in group}
            polymer_type = _POLYMER_TYPE[entity_type]
            if polymer_type == "protein":
                representative_name = "CA" if component_id == "GLY" else "CB"
            elif component_id in {"A", "G", "DA", "DG"}:
                representative_name = "C4"
            elif component_id in {"C", "U", "DC", "DT"}:
                representative_name = "C2"
            else:
                representative_name = "C1'"
            representative = by_name.get(representative_name, center)
        tokens.append(
            {
                "atoms": group,
                "atom_index": atom_index if atom_token else None,
                "component_index": component_index,
                "instance_index": instance_index,
                "entity_index": entity_index,
                "monomer_index": (
                    None if entity_type == ENTITY_MOLECULE else int(layout.token_residue_index[token_index])
                ),
                "residue_index": int(layout.token_residue_index[token_index]),
                "chain_index": instance_index,
                "symmetry_index": int(layout.token_symmetry_index[token_index]),
                "residue_type": int(layout.token_type_index[token_index]),
                "center": center,
                "representative": representative,
                "standard": standard,
                "expanded": standard and atom_token,
                "reference_space": component_index,
            }
        )
    return tokens


def _constraint_token_mask(
    layout: TokenLayout,
    system: MolecularSystem,
    atom_contacts: tuple[ProtenixAtomContact, ...],
    token_contacts: tuple[ProtenixTokenContact, ...],
    pocket: ProtenixPocket | None,
) -> Tensor | None:
    r"""Mark every token that can realize a discrete spatial constraint."""

    if not atom_contacts and not token_contacts and pocket is None:
        return None

    required = torch.zeros_like(layout.token_component_index, dtype=torch.bool)
    counts = layout.token_atom_ptr[1:] - layout.token_atom_ptr[:-1]
    token_index = torch.repeat_interleave(torch.arange(required.numel(), device=required.device), counts)
    atom_to_token = dict(zip(layout.atom_index.detach().cpu().tolist(), token_index.detach().cpu().tolist()))

    def require_atom(site: AtomSite) -> None:
        try:
            atom = system.atom_index(site)
        except KeyError:
            return
        token = atom_to_token.get(atom)
        if token is not None:
            required[token] = True

    def require_endpoint(endpoint: ComponentReference | AtomSite) -> None:
        if isinstance(endpoint, AtomSite):
            require_atom(endpoint)
        else:
            try:
                component = system.component_index(endpoint)
            except KeyError:
                return
            required[layout.token_component_index == component] = True

    for atom_contact in atom_contacts:
        require_atom(atom_contact.atom_a)
        require_atom(atom_contact.atom_b)
    for token_contact in token_contacts:
        require_endpoint(token_contact.endpoint_a)
        require_endpoint(token_contact.endpoint_b)
    if pocket is not None:
        try:
            binder = system.instance_index(pocket.binder)
        except KeyError:
            binder = -1
        required[layout.token_instance_index == binder] = True
        for residue in pocket.contact_residues:
            require_endpoint(residue)
    return required


def _features(
    system: MolecularSystem,
    *,
    tokens: list[dict[str, object]],
    msa: MSA | MsaCollection | None,
    msa_bindings: tuple[tuple[MSA, int | None, int], ...],
    templates: tuple[Template, ...] | None,
    atom_contacts: tuple[ProtenixAtomContact, ...],
    token_contacts: tuple[ProtenixTokenContact, ...],
    pocket: ProtenixPocket | None,
    substructure: Structure | None,
    substructure_coordinate_noise_scale: float,
    structure_condition: Structure | None,
    prepare_tfg_features: bool,
    generator: torch.Generator | None,
    max_msa_rows: int,
    augment_reference_positions: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> ProtenixInput:
    atom_indices = tuple(atom_index for token in tokens for atom_index in cast(tuple[int, ...], token["atoms"]))
    system_to_model_atom = {atom_index: model_atom_index for model_atom_index, atom_index in enumerate(atom_indices)}
    system_atom_to_token = {
        atom_index: token_index
        for token_index, token in enumerate(tokens)
        for atom_index in cast(tuple[int, ...], token["atoms"])
    }
    atom_to_token = torch.tensor(
        [system_atom_to_token[atom_index] for atom_index in atom_indices],
        dtype=torch.long,
    )
    reference_positions, reference_mask, reference_charges = _reference_features(
        system,
        atom_indices,
        augment=augment_reference_positions,
        generator=generator,
    )
    # Preserve canonical atom IDs; SMILES imports assign element-number IDs during construction.
    atom_names = tuple(system.symbols[int(system.atom_symbol[atom_index])] for atom_index in atom_indices)
    element_indices = (
        system.atomic_number[torch.tensor(atom_indices, device=system.atomic_number.device)].detach().cpu().long() - 1
    )
    name_indices = torch.tensor(
        [[min(63, max(0, ord(character) - 32)) for character in name.ljust(4)[:4]] for name in atom_names],
        dtype=torch.long,
    )
    atom_features = torch.cat(
        (
            reference_mask[:, None].float(),
            F.one_hot(element_indices, num_classes=128).float(),
            F.one_hot(name_indices, num_classes=64).reshape(len(atom_indices), 256).float(),
        ),
        dim=-1,
    )
    component_atoms = system.component_atom_ptr.detach().cpu().tolist()
    atomic_numbers = system.atomic_number.detach().cpu().tolist()
    atom_symbols = system.atom_symbol.detach().cpu().tolist()
    component_symbols = system.component_symbol.detach().cpu().tolist()
    atom_token_index = torch.empty(len(atom_indices), dtype=torch.long)
    for token in tokens:
        group = cast(tuple[int, ...], token["atoms"])
        if cast(bool, token["expanded"]) or not cast(bool, token["standard"]):
            for atom_index in group:
                atom_token_index[system_to_model_atom[atom_index]] = 0
        else:
            component_index = cast(int, token["component_index"])
            component_id = system.symbols[component_symbols[component_index]]
            if component_id == "N":
                heavy_order = {
                    name: index
                    for index, name in enumerate(
                        ("OP3", "P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'")
                    )
                }
            elif component_id == "DN":
                heavy_order = {
                    name: index
                    for index, name in enumerate(
                        ("OP3", "P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'")
                    )
                }
            else:
                heavy_order = {
                    system.symbols[atom_symbols[atom_index]]: index
                    for index, atom_index in enumerate(
                        atom_index
                        for atom_index in range(
                            component_atoms[component_index],
                            component_atoms[component_index + 1],
                        )
                        if atomic_numbers[atom_index] != 1
                    )
                }
            for atom_index in group:
                atom_token_index[system_to_model_atom[atom_index]] = heavy_order[
                    system.symbols[atom_symbols[atom_index]]
                ]

    msa_tokens, profile, deletion_mean, deletion_counts = _msa_features(
        msa,
        msa_bindings,
        tokens,
        system,
        max_rows=max_msa_rows,
    )
    residue_type = torch.tensor([cast(int, token["residue_type"]) for token in tokens])
    token_features = torch.cat(
        (
            torch.zeros(len(tokens), 384),
            F.one_hot(residue_type, num_classes=TOKEN_VOCAB_SIZE).float(),
            profile,
            deletion_mean[:, None],
        ),
        dim=-1,
    )
    esm_tokens, esm_mask, token_to_esm_residue, token_has_esm = _esm_features(system, tokens)

    pocket_feature, token_contact_feature, atom_contact_feature, substructure_feature = _constraint_features(
        system=system,
        tokens=tokens,
        atom_contacts=atom_contacts,
        token_contacts=token_contacts,
        pocket=pocket,
        substructure=substructure,
        substructure_coordinate_noise_scale=substructure_coordinate_noise_scale,
        generator=generator,
        system_atom_to_token=system_atom_to_token,
    )
    token_bonds = _token_bonds(system, system_atom_to_token, len(tokens))
    token_to_rep_atom = torch.tensor(
        [system_to_model_atom[cast(int, token["representative"])] for token in tokens],
        dtype=torch.long,
    )
    structure_rep_positions, structure_rep_mask = _structure_condition_features(
        system,
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
        if not prepare_tfg_features
        else _tfg_features(
            system,
            system_to_model_atom=system_to_model_atom,
            atom_chain=torch.tensor(
                [
                    cast(
                        int,
                        tokens[system_atom_to_token[atom_index]]["chain_index"],
                    )
                    for atom_index in atom_indices
                ],
                dtype=torch.long,
            ),
            element_index=element_indices,
            device=device,
            dtype=dtype,
        )
    )

    def floating(value: Tensor) -> Tensor:
        return value.to(device=device, dtype=dtype).unsqueeze(0)

    def placed(value: Tensor) -> Tensor:
        return value.to(device=device).unsqueeze(0)

    return ProtenixInput(  # type: ignore[call-arg]
        token_mask=placed(torch.ones(len(tokens), dtype=torch.bool)),
        residue_index=placed(torch.tensor([cast(int, token["residue_index"]) for token in tokens])),
        chain_index=placed(torch.tensor([cast(int, token["chain_index"]) for token in tokens])),
        entity_index=placed(torch.tensor([cast(int, token["entity_index"]) for token in tokens])),
        symmetry_index=placed(torch.tensor([cast(int, token["symmetry_index"]) for token in tokens])),
        token_index=placed(torch.arange(len(tokens))),
        atom_features=floating(atom_features),
        atom_to_token=placed(atom_to_token),
        atom_mask=placed(torch.ones(len(atom_indices), dtype=torch.bool)),
        reference_positions=floating(reference_positions),
        reference_space_id=placed(
            torch.tensor(
                [
                    cast(
                        int,
                        tokens[system_atom_to_token[atom_index]]["reference_space"],
                    )
                    for atom_index in atom_indices
                ],
                dtype=torch.long,
            )
        ),
        atom_token_index=placed(atom_token_index),
        token_to_rep_atom=placed(token_to_rep_atom),
        token_features=floating(token_features),
        reference_mask=placed(reference_mask),
        reference_charges=floating(reference_charges[:, None]),
        token_bonds=floating(token_bonds),
        msa_tokens=placed(msa_tokens),
        msa_deletion_counts=placed(deletion_counts),
        esm_tokens=placed(esm_tokens),
        esm_mask=placed(esm_mask),
        token_to_esm_residue=placed(token_to_esm_residue),
        token_has_esm=placed(token_has_esm),
        pocket=floating(pocket_feature),
        token_contact=floating(token_contact_feature),
        atom_contact=floating(atom_contact_feature),
        substructure=floating(substructure_feature),
        structure_rep_positions=floating(structure_rep_positions),
        structure_rep_mask=placed(structure_rep_mask),
        template_features=template_features,
        tfg_features=tfg_features,
        system=system,
        atom_index=placed(torch.tensor(atom_indices, dtype=torch.long)),
        token_component_index=placed(
            torch.tensor(
                [cast(int, token["component_index"]) for token in tokens],
                dtype=torch.long,
            )
        ),
        token_atom_index=placed(
            torch.tensor(
                [-1 if token["atom_index"] is None else cast(int, token["atom_index"]) for token in tokens],
                dtype=torch.long,
            )
        ),
        batch_size=[],
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
    system_to_model_atom: dict[int, int],
    atom_chain: Tensor,
    element_index: Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> ProtenixTfgFeatures:
    r"""
    Prepare the static molecular geometry used by Protenix TFG.
    """

    component_instance = torch.repeat_interleave(
        torch.arange(system.instance_symbol.numel(), device=system.instance_symbol.device),
        system.instance_component_ptr[1:] - system.instance_component_ptr[:-1],
    )
    atom_component = torch.repeat_interleave(
        torch.arange(system.component_symbol.numel(), device=system.component_symbol.device),
        system.component_atom_ptr[1:] - system.component_atom_ptr[:-1],
    )
    atom_instance = component_instance[atom_component].detach().cpu().tolist()
    edges = system.edge_index.detach().cpu().T.tolist()
    interchain_bonds = [
        (system_to_model_atom[first], system_to_model_atom[second])
        for first, second in edges
        if first in system_to_model_atom
        and second in system_to_model_atom
        and atom_instance[first] != atom_instance[second]
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

    component_atoms = system.component_atom_ptr.detach().cpu().tolist()
    instance_entities = system.instance_entity.detach().cpu().tolist()
    entity_types = system.entity_type.detach().cpu().tolist()
    component_instances = component_instance.detach().cpu().tolist()
    component_symbols = system.component_symbol.detach().cpu().tolist()
    atomic_numbers = system.atomic_number.detach().cpu().tolist()
    reference_mask = system.reference_coordinate_mask.detach().cpu().tolist()
    for component_index, instance_index in enumerate(component_instances):
        entity_index = instance_entities[instance_index]
        entity_type = entity_types[entity_index]
        component_id = system.symbols[component_symbols[component_index]]
        if _is_standard(system, component_index, entity_type):
            continue
        atom_start = component_atoms[component_index]
        atom_stop = component_atoms[component_index + 1]
        if any(atomic_number in _METAL_ATOMIC_NUMBERS for atomic_number in atomic_numbers[atom_start:atom_stop]):
            continue
        local_to_global = {
            atom_index - atom_start: system_to_model_atom[atom_index]
            for atom_index in range(atom_start, atom_stop)
            if atom_index in system_to_model_atom
        }
        if any(not reference_mask[atom_start + index] for index in local_to_global):
            raise ValueError(f"TFG requires reference coordinates for modeled atoms in component {component_id!r}")
        geometry = _component_geometry(system, component_index)

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
        for chiral_local_index, orientation in zip(
            geometry.chiral_index,
            geometry.chiral_orientation,
        ):
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
            planar_global_index = _mapped_index(
                planar_local_index,
                local_to_global,
            )
            if planar_global_index is not None:
                planar_improper_index.append(cast(tuple[int, int, int, int], planar_global_index))

    return ProtenixTfgFeatures(  # type: ignore[call-arg]
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
        batch_size=[],
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


def _component_geometry(
    system: MolecularSystem,
    component_index: int,
) -> _ProtenixComponentGeometry:
    r"""
    Extract the released TFG constraints for one non-standard component.
    """

    from itertools import combinations

    from rdkit import Chem
    from rdkit.Chem.rdchem import BondStereo
    from rdkit.Chem.rdDistGeom import GetExperimentalTorsions, GetMoleculeBoundsMatrix
    from rdkit.Chem.rdMolTransforms import GetDihedralRad

    molecule = to_rdkit_component(system, component_index)
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

    instance_entities = system.instance_entity.detach().cpu().tolist()
    entity_types = system.entity_type.detach().cpu().tolist()
    instance_components = system.instance_component_ptr.detach().cpu().tolist()
    component_symbols = system.component_symbol.detach().cpu().tolist()
    protein_entities: list[int] = []
    seen_entities: set[int] = set()
    entity_components: dict[int, tuple[int, int]] = {}
    for instance_index, entity_index in enumerate(instance_entities):
        if entity_index in seen_entities:
            continue
        seen_entities.add(entity_index)
        if entity_types[entity_index] == ENTITY_PROTEIN:
            protein_entities.append(entity_index)
            entity_components[entity_index] = (
                instance_components[instance_index],
                instance_components[instance_index + 1],
            )
    sequence_length = max(
        (
            entity_components[entity_index][1] - entity_components[entity_index][0] + 2
            for entity_index in protein_entities
        ),
        default=0,
    )
    esm_tokens = torch.full(
        (len(protein_entities), sequence_length),
        _ESM_PAD_TOKEN,
        dtype=torch.long,
    )
    esm_mask = torch.zeros_like(esm_tokens, dtype=torch.bool)
    for sequence_index, entity_index in enumerate(protein_entities):
        component_start, component_stop = entity_components[entity_index]
        residue_tokens = torch.tensor(
            [
                _ESM_TOKEN_BY_COMPONENT.get(
                    system.symbols[component_symbols[component_index]],
                    _ESM_UNKNOWN_RESIDUE_TOKEN,
                )
                for component_index in range(component_start, component_stop)
            ],
            dtype=torch.long,
        )
        end = len(residue_tokens) + 1
        esm_tokens[sequence_index, 0] = _ESM_CLS_TOKEN
        esm_tokens[sequence_index, 1:end] = residue_tokens
        esm_tokens[sequence_index, end] = _ESM_EOS_TOKEN
        esm_mask[sequence_index, : end + 1] = True

    entity_to_sequence = {entity_index: index for index, entity_index in enumerate(protein_entities)}
    token_to_esm_residue = torch.zeros(len(tokens), dtype=torch.long)
    token_has_esm = torch.zeros(len(tokens), dtype=torch.bool)
    for token_index, token in enumerate(tokens):
        esm_sequence_index = entity_to_sequence.get(cast(int, token["entity_index"]))
        monomer_index = cast(int | None, token["monomer_index"])
        if esm_sequence_index is None or monomer_index is None:
            continue
        token_to_esm_residue[token_index] = esm_sequence_index * sequence_length + monomer_index + 1
        token_has_esm[token_index] = True
    return esm_tokens, esm_mask, token_to_esm_residue, token_has_esm


def _structure_condition_features(
    system: MolecularSystem,
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

    for token_index, token in enumerate(tokens):
        site = system.atom_site(cast(int, token["representative"]))
        try:
            source_index = structure.system.atom_index(site)
        except KeyError:
            continue
        if bool(structure.coordinate_mask[source_index]):
            positions[token_index] = structure.coordinates[source_index].detach().cpu()
            mask[token_index] = True
    return positions, mask


def _reference_features(
    system: MolecularSystem,
    atom_indices: tuple[int, ...],
    *,
    augment: bool,
    generator: torch.Generator | None,
) -> tuple[Tensor, Tensor, Tensor]:
    index = torch.tensor(atom_indices, device=system.atom_symbol.device)
    positions = system.reference_coordinates[index].detach().cpu().clone()
    mask = system.reference_coordinate_mask[index].detach().cpu().clone()
    charges = system.formal_charge[index].detach().cpu().float()
    # One reference space per component occurrence, matching the assignment in _tokenize.
    atom_component = (
        torch.repeat_interleave(
            torch.arange(system.component_symbol.numel(), device=system.atom_symbol.device),
            system.component_atom_ptr[1:] - system.component_atom_ptr[:-1],
        )
        .detach()
        .cpu()
    )
    grouped: dict[int, list[int]] = {}
    for model_atom_index, atom_index in enumerate(atom_indices):
        grouped.setdefault(int(atom_component[atom_index]), []).append(model_atom_index)

    component_instance = (
        torch.repeat_interleave(
            torch.arange(system.instance_symbol.numel(), device=system.instance_symbol.device),
            system.instance_component_ptr[1:] - system.instance_component_ptr[:-1],
        )
        .detach()
        .cpu()
        .tolist()
    )
    instance_entities = system.instance_entity.detach().cpu().tolist()
    entity_types = system.entity_type.detach().cpu().tolist()

    group_indices: list[Tensor] = []
    component_symbols = system.component_symbol.detach().cpu().tolist()
    for component_index, model_indices in grouped.items():
        indices = torch.tensor(model_indices)
        if not bool(mask[indices].any()):
            entity_index = instance_entities[component_instance[component_index]]
            if entity_types[entity_index] != ENTITY_MOLECULE:
                component_id = system.symbols[component_symbols[component_index]]
                raise ValueError(
                    f"Protenix requires explicit reference_coordinates for polymer component {component_id!r}"
                )
            # An observed-only ligand carries no CCD ideal conformer. A reference space is
            # one conformer, so embed the whole component from its canonical connectivity
            # rather than mixing generated positions into a partially known frame.
            atom_start = int(system.component_atom_ptr[component_index])
            generated = _generated_reference_coordinates(system, component_index)
            positions[indices] = positions.new_tensor(
                generated[[atom_indices[model_index] - atom_start for model_index in model_indices]]
            )
            mask[indices] = True
        positions[indices] -= positions[indices].mean(dim=0)
        if augment:
            group_indices.append(indices)

    if group_indices:
        random_device = positions.device if generator is None else generator.device
        translations = (
            torch.empty(
                len(group_indices),
                3,
                device=random_device,
                dtype=positions.dtype,
            )
            .uniform_(-1.0, 1.0, generator=generator)
            .to(positions.device)
        )
        rotations = uniform_random_rotation(
            len(group_indices),
            device=random_device,
            dtype=positions.dtype,
            generator=generator,
        ).to(positions.device)
        for indices, translation, rotation in zip(group_indices, translations, rotations):
            positions[indices] = torch.matmul(
                positions[indices] + translation,
                rotation.transpose(-1, -2),
            )
    return positions, mask, charges


def _generated_reference_coordinates(system: MolecularSystem, component_index: int) -> np.ndarray:
    r"""
    Embed one non-polymer component's canonical connectivity as a conformer.

    Args:
        system: Canonical molecular system.
        component_index: Component occurrence to embed.

    Returns:
        Angstrom positions in the component's canonical atom order, expressed in
        the conformer's own arbitrary frame. The canonical graph is untouched.
    """

    from rdkit import Chem
    from rdkit.Chem import rdDistGeom, rdForceFieldHelpers

    atom_start = int(system.component_atom_ptr[component_index])
    atom_stop = int(system.component_atom_ptr[component_index + 1])
    molecule = Chem.AddHs(to_rdkit_component(system, component_index))
    options: Any = rdDistGeom.ETKDGv3()
    options.randomSeed = 42
    conformer_id = rdDistGeom.EmbedMolecule(molecule, options)
    if conformer_id == -1:
        options.useRandomCoords = True
        conformer_id = rdDistGeom.EmbedMolecule(molecule, options)
    rdForceFieldHelpers.UFFOptimizeMolecule(molecule, confId=conformer_id, maxIters=1000)
    return molecule.GetConformer(conformer_id).GetPositions()[: atom_stop - atom_start]


def _msa_polymer_type(system: MolecularSystem, entity_index: int) -> str | None:
    return _POLYMER_TYPE.get(int(system.entity_type[entity_index]))


def _msa_plan(
    msa: MSA | MsaCollection | None,
    resolved_bindings: tuple[tuple[MSA, int | None, int], ...],
    system: MolecularSystem,
    *,
    max_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Select source rows and pools for the lowered MSA row axis."""

    column_count = len(resolved_bindings)
    if msa is None:
        return np.full((1, column_count), -1, dtype=np.int64), np.zeros(
            (1, column_count),
            dtype=np.bool_,
        )
    if isinstance(msa, MSA):
        depth = min(len(msa.unpaired_sequences), max_rows)
        source = np.full((depth, column_count), -1, dtype=np.int64)
        if depth > 1:
            source[1:, 0] = np.arange(1, depth)
        return source, np.zeros_like(source, dtype=np.bool_)

    row_pairing = msa.row_pairing
    row_is_paired = msa.row_is_paired
    protein_columns = np.asarray(
        [_msa_polymer_type(system, entity_index) == "protein" for _value, _target, entity_index in resolved_bindings],
        dtype=np.bool_,
    )
    protein_paired = np.count_nonzero(
        (row_pairing != -1) & row_is_paired & protein_columns[None, :],
        axis=1,
    )
    paired = np.flatnonzero(protein_paired[1:] >= 2) + 1
    unpaired = np.flatnonzero(protein_paired[1:] < 2) + 1
    pair_limit = max_rows // 2
    selected = np.concatenate(
        (
            paired[: max(0, pair_limit - 1)],
            unpaired[: max(0, max_rows - 1 - min(len(paired), max(0, pair_limit - 1)))],
        )
    )
    source = np.full((len(selected) + 1, column_count), -1, dtype=np.int64)
    paired_pool = np.zeros_like(source, dtype=np.bool_)
    source[1:] = row_pairing[selected]
    paired_pool[1:] = row_is_paired[selected]

    rna_columns = np.asarray(
        [_msa_polymer_type(system, entity_index) == "rna" for _value, _target, entity_index in resolved_bindings],
        dtype=np.bool_,
    )
    ignored = paired_pool & rna_columns[None, :]
    source[ignored] = -1
    paired_pool[ignored] = False
    return source, paired_pool


def _msa_profile(encoded: np.ndarray) -> np.ndarray:
    r"""Compute a per-column categorical profile without materializing one-hot rows."""

    width = encoded.shape[1]
    counts = np.zeros((width, MSA_VOCAB_SIZE), dtype=np.int64)
    offsets = np.arange(width, dtype=np.int64)[None, :] * MSA_VOCAB_SIZE
    for start in range(0, len(encoded), 4096):
        values = encoded[start : start + 4096] + offsets
        counts += np.bincount(values.ravel(), minlength=width * MSA_VOCAB_SIZE).reshape(counts.shape)
    return counts.astype(np.float32) / len(encoded)


def _msa_features(
    msa: MSA | MsaCollection | None,
    resolved_bindings: tuple[tuple[MSA, int | None, int], ...],
    tokens: list[dict[str, object]],
    system: MolecularSystem,
    *,
    max_rows: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    source_rows, paired_pool = _msa_plan(
        msa,
        resolved_bindings,
        system,
        max_rows=max_rows,
    )
    token_entities = np.fromiter((cast(int, token["entity_index"]) for token in tokens), dtype=np.int64)
    token_instances = np.fromiter((cast(int, token["instance_index"]) for token in tokens), dtype=np.int64)
    token_monomers = np.fromiter(
        (
            -1 if cast(int | None, token["monomer_index"]) is None else cast(int, token["monomer_index"])
            for token in tokens
        ),
        dtype=np.int64,
    )
    residue_types = np.fromiter((cast(int, token["residue_type"]) for token in tokens), dtype=np.int64)
    rows = np.broadcast_to(residue_types, (source_rows.shape[0], len(tokens))).copy()
    deletions = np.zeros(rows.shape, dtype=np.int32)
    profile = np.eye(MSA_VOCAB_SIZE, dtype=np.float32)[residue_types]
    deletion_mean = np.zeros(len(tokens), dtype=np.float32)

    for column, (selected, target_instance, entity_index) in enumerate(resolved_bindings):
        polymer_type = _msa_polymer_type(system, entity_index)
        if polymer_type not in {"protein", "rna"}:
            continue
        token_mask = token_entities == entity_index
        if target_instance is not None:
            token_mask &= token_instances == target_instance
        token_indices = np.flatnonzero(token_mask)
        if not len(token_indices):
            continue
        monomer_indices = token_monomers[token_indices]
        rows[1:, token_indices] = TOKEN_GAP_INDEX
        unpaired = encode_msa_sequences(selected.unpaired_sequences, polymer_type)

        for is_paired in (False, True):
            output_rows = np.flatnonzero((source_rows[:, column] >= 0) & (paired_pool[:, column] == is_paired))
            if not len(output_rows):
                continue
            encoded = encode_msa_sequences(selected.paired_sequences, polymer_type) if is_paired else unpaired
            deletion_pool = selected.paired_deletion_counts if is_paired else selected.unpaired_deletion_counts
            selected_rows = source_rows[output_rows, column]
            rows[np.ix_(output_rows, token_indices)] = encoded[np.ix_(selected_rows, monomer_indices)]
            deletions[np.ix_(output_rows, token_indices)] = deletion_pool[np.ix_(selected_rows, monomer_indices)]

        unique_rows = np.sort(np.unique(unpaired, axis=0, return_index=True)[1])
        unique = unpaired[unique_rows]
        profile[token_indices] = _msa_profile(unique)[monomer_indices]
        deletion_mean[token_indices] = selected.unpaired_deletion_counts[unique_rows].mean(
            axis=0,
            dtype=np.float32,
        )[monomer_indices]

    row_tensor = torch.from_numpy(rows)
    return (
        row_tensor,
        torch.from_numpy(profile),
        torch.from_numpy(deletion_mean),
        torch.from_numpy(deletions),
    )


def _template_to(
    features: ProtenixTemplateFeatures,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> ProtenixTemplateFeatures:
    return ProtenixTemplateFeatures(  # type: ignore[call-arg]
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
    residue_tokens: dict[int, list[int]] = {}
    chain_tokens: dict[int, list[int]] = {}
    for token_index, token in enumerate(tokens):
        instance_index = cast(int, token["instance_index"])
        chain_tokens.setdefault(instance_index, []).append(token_index)
        monomer_index = cast(int | None, token["monomer_index"])
        if monomer_index is not None:
            residue_tokens.setdefault(
                cast(int, token["component_index"]),
                [],
            ).append(token_index)

    instance_entities = system.instance_entity.detach().cpu().tolist()
    chain_templates: dict[
        int,
        list[
            tuple[
                Template,
                tuple[tuple[int, tuple[tuple[int, int], ...]], ...],
            ]
        ],
    ] = {
        instance_index: []
        for instance_index, entity_index in enumerate(instance_entities)
        if int(system.entity_type[entity_index]) == ENTITY_PROTEIN
    }
    for template in templates:
        alignments_by_chain: dict[
            int,
            list[tuple[int, tuple[tuple[int, int], ...]]],
        ] = {}
        for alignment in template.alignments:
            template_instance_index = template.structure.system.instance_index(alignment.template_instance_id)
            template_entity_index = int(template.structure.system.instance_entity[template_instance_index])
            if int(template.structure.system.entity_type[template_entity_index]) != ENTITY_PROTEIN:
                raise ValueError("Protenix templates must align protein template instances")
            for instance_index in _template_target_instances(alignment, system):
                alignments_by_chain.setdefault(instance_index, []).append(
                    (template_instance_index, alignment.residue_pairs)
                )
        for instance_index, alignments in alignments_by_chain.items():
            if len(chain_templates[instance_index]) < 4:
                chain_templates[instance_index].append((template, tuple(alignments)))

    positions = torch.zeros(4, token_count, 4, 3)
    atom_mask = torch.zeros(4, token_count, 4)
    residue_type_ids = torch.zeros(4, token_count, dtype=torch.long)
    # An empty chain contributes one GAP slot; fixed-four padding leaves the remaining slots at residue ID zero.
    residue_type_ids[0] = TOKEN_GAP_INDEX

    target_component_ptr = system.instance_component_ptr.detach().cpu().tolist()
    for instance_index, selected_templates in chain_templates.items():
        target_tokens = chain_tokens.get(instance_index, [])
        for template_index, (template, template_alignments) in enumerate(selected_templates):
            residue_type_ids[template_index, target_tokens] = TOKEN_GAP_INDEX
            template_system = template.structure.system
            template_component_ptr = template_system.instance_component_ptr.detach().cpu().tolist()
            template_component_symbols = template_system.component_symbol.detach().cpu().tolist()
            for template_instance_index, residue_pairs in template_alignments:
                template_entity_index = int(template_system.instance_entity[template_instance_index])
                for target_component, template_component in residue_pairs:
                    target_component_start = target_component_ptr[instance_index]
                    target_component_stop = target_component_ptr[instance_index + 1]
                    if target_component >= target_component_stop - target_component_start:
                        raise ValueError(
                            f"template alignment target "
                            f"{system.instance_id(instance_index)!r}:{target_component} "
                            "does not resolve to a component"
                        )
                    target_component_index = target_component_start + target_component
                    targets = residue_tokens.get(target_component_index)
                    if targets is None:
                        continue
                    template_component_index = template_component_ptr[template_instance_index] + template_component
                    backbone, backbone_mask = _template_residue_backbone(
                        template.structure,
                        template_component_index,
                    )
                    component_id = template_system.symbols[template_component_symbols[template_component_index]]
                    residue_type = component_token_index(component_id, "protein")
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
    return ProtenixTemplateFeatures(  # type: ignore[call-arg]
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
) -> tuple[int, ...]:
    try:
        entity_index = system.entity_index(alignment.target_entity_id)
    except KeyError:
        raise ValueError(f"template target {alignment.target_entity_id!r} is not a protein entity") from None
    if int(system.entity_type[entity_index]) != ENTITY_PROTEIN:
        raise ValueError(f"template target {alignment.target_entity_id!r} is not a protein entity")
    if alignment.target_instance_id is None:
        return tuple(
            instance_index
            for instance_index, actual_entity_index in enumerate(system.instance_entity.detach().cpu().tolist())
            if actual_entity_index == entity_index
        )
    try:
        instance_index = system.instance_index(alignment.target_instance_id)
    except KeyError:
        raise ValueError(f"template alignment targets unknown instance {alignment.target_instance_id!r}")
    actual_entity_index = int(system.instance_entity[instance_index])
    if actual_entity_index != entity_index:
        raise ValueError(
            f"template alignment instance {alignment.target_instance_id!r} belongs to entity "
            f"{system.entity_id(actual_entity_index)!r}, not {alignment.target_entity_id!r}"
        )
    return (instance_index,)


def _template_residue_backbone(
    structure: Structure,
    component_index: int,
) -> tuple[Tensor, Tensor]:
    positions = torch.zeros(4, 3)
    mask = torch.zeros(4)
    system = structure.system
    atom_start = int(system.component_atom_ptr[component_index])
    atom_stop = int(system.component_atom_ptr[component_index + 1])
    atoms_by_name = {system.symbols[int(system.atom_symbol[index])]: index for index in range(atom_start, atom_stop)}
    for row, atom_id in enumerate(("N", "CA", "C", "CB")):
        atom_index = atoms_by_name.get(atom_id)
        if atom_index is not None and bool(structure.coordinate_mask[atom_index]):
            positions[row] = structure.coordinates[atom_index].detach().cpu()
            mask[row] = 1
    return positions, mask


def _template_backbone_features(
    positions: Tensor,
    mask: Tensor,
    residue_type_ids: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    nitrogen, alpha, carbon, beta = positions.unbind(dim=-2)
    nitrogen_mask, alpha_mask, carbon_mask, beta_mask = mask.unbind(dim=-1)

    is_glycine = residue_type_ids == TOKEN_INDEX["G"]
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
    system: MolecularSystem,
    tokens: list[dict[str, object]],
    atom_contacts: tuple[ProtenixAtomContact, ...],
    token_contacts: tuple[ProtenixTokenContact, ...],
    pocket: ProtenixPocket | None,
    substructure: Structure | None,
    substructure_coordinate_noise_scale: float,
    generator: torch.Generator | None,
    system_atom_to_token: dict[int, int],
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
    residue_tokens: dict[int, list[int]] = {}
    instance_tokens: dict[int, list[int]] = {}
    for index, token in enumerate(tokens):
        instance_index = cast(int, token["instance_index"])
        instance_tokens.setdefault(instance_index, []).append(index)
        monomer_index = cast(int | None, token["monomer_index"])
        if monomer_index is not None:
            residue_tokens.setdefault(
                cast(int, token["component_index"]),
                [],
            ).append(index)

    def atom_token(site: AtomSite) -> int:
        try:
            return system_atom_to_token[system.atom_index(site)]
        except (KeyError, IndexError):
            raise ValueError(
                f"Protenix constraint references atom {site!r}, which is not a model-visible token"
            ) from None

    def endpoint_token(endpoint: ComponentReference | AtomSite) -> int:
        if isinstance(endpoint, AtomSite):
            return atom_token(endpoint)
        try:
            component_index = system.component_index(endpoint)
        except (KeyError, IndexError):
            component_index = -1
        candidates = residue_tokens.get(component_index)
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
        try:
            binder_instance = system.instance_index(pocket.binder)
        except KeyError:
            binder_instance = -1
        binder_tokens = instance_tokens.get(binder_instance)
        if not binder_tokens:
            raise ValueError(f"Protenix pocket binder chain {pocket.binder!r} has no model-visible token")
        for residue in pocket.contact_residues:
            try:
                component_index = system.component_index(residue)
            except (KeyError, IndexError):
                component_index = -1
            candidates = residue_tokens.get(component_index)
            if not candidates:
                raise ValueError(f"Protenix pocket residue {residue!r} has no model-visible token")
            pocket_token = _draw_token(candidates, generator)
            for binder_token in binder_tokens:
                pocket_feature[binder_token, pocket_token, 0] = pocket.max_distance

    if substructure is not None:
        selected_token: list[int] = []
        selected_coordinates: list[Tensor] = []
        for token_index, token in enumerate(tokens):
            center = cast(int, token["center"])
            try:
                source_index = substructure.system.atom_index(system.atom_site(center))
            except KeyError:
                continue
            if bool(substructure.coordinate_mask[source_index]):
                selected_token.append(token_index)
                selected_coordinates.append(substructure.coordinates[source_index].detach().cpu())
        if selected_token:
            coordinates = torch.stack(selected_coordinates)
            if substructure_coordinate_noise_scale != 0.0:
                random_device = coordinates.device if generator is None else generator.device
                noise = torch.randn(
                    coordinates.shape,
                    device=random_device,
                    dtype=coordinates.dtype,
                    generator=generator,
                ).to(coordinates.device)
                coordinates = coordinates + substructure_coordinate_noise_scale * noise
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
    random_device = torch.device("cpu") if generator is None else generator.device
    choice = int(
        torch.randint(
            0,
            len(candidates),
            (1,),
            device=random_device,
            generator=generator,
        ).item()
    )
    return candidates[choice]


def _write_symmetric(feature: Tensor, first: int, second: int, value: tuple[float, float]) -> None:
    channels = torch.tensor(value, dtype=feature.dtype)
    previous = feature[first, second]
    if bool((previous != 0).any()) and not torch.equal(previous, channels):
        raise ValueError("conflicting Protenix constraints target the same token pair")
    feature[first, second] = channels
    feature[second, first] = channels


def _token_bonds(
    system: MolecularSystem,
    system_atom_to_token: dict[int, int],
    token_count: int,
) -> Tensor:
    feature = torch.zeros(token_count, token_count)
    component_instance = torch.repeat_interleave(
        torch.arange(system.instance_symbol.numel(), device=system.instance_symbol.device),
        system.instance_component_ptr[1:] - system.instance_component_ptr[:-1],
    )
    atom_component = torch.repeat_interleave(
        torch.arange(system.component_symbol.numel(), device=system.component_symbol.device),
        system.component_atom_ptr[1:] - system.component_atom_ptr[:-1],
    )
    component_instances = component_instance.detach().cpu().tolist()
    atom_components = atom_component.detach().cpu().tolist()
    instance_entities = system.instance_entity.detach().cpu().tolist()
    entity_types = system.entity_type.detach().cpu().tolist()
    edge_rows = system.edge_index.detach().cpu().T.tolist()
    bond_roles = system.bond_role.detach().cpu().tolist()
    for edge_index, (first, second) in enumerate(edge_rows):
        if first not in system_atom_to_token or second not in system_atom_to_token:
            continue
        first_component = atom_components[first]
        second_component = atom_components[second]
        first_instance = component_instances[first_component]
        second_instance = component_instances[second_component]
        first_entity = instance_entities[first_instance]
        second_entity = instance_entities[second_instance]
        first_type = entity_types[first_entity]
        second_type = entity_types[second_entity]
        keep = first_type == ENTITY_MOLECULE or second_type == ENTITY_MOLECULE
        if first_type != ENTITY_MOLECULE and second_type != ENTITY_MOLECULE:
            if first_instance == second_instance and first_component == second_component:
                keep = not _is_standard(
                    system,
                    first_component,
                    first_type,
                )
            elif bond_roles[edge_index] >= BOND_ROLE_ENTITY and (
                first_instance != second_instance or abs(first_component - second_component) > 1
            ):
                keep = True
            else:
                keep = False
        if keep:
            first_token = system_atom_to_token[first]
            second_token = system_atom_to_token[second]
            feature[first_token, second_token] = 1
            feature[second_token, first_token] = 1
    return feature
