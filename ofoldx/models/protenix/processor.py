r"""
Prepare biomolecular features for Protenix.
"""

from __future__ import annotations

import itertools
import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from ...data import water_component_mask
from ...data.ccd import resolve_ccd_component
from ...data.ccd.resolver import validate_ccd_snapshot_path
from ...data.components import get_component
from ...data.evidence import (
    MSA,
    MSACollection,
    Template,
    TemplateAlignment,
    resolve_msa_bindings,
)
from ...data.molecular import (
    BOND_KIND_COVALENT,
    BOND_ROLE_COMPONENT,
    BOND_ROLE_ENTITY,
    ENTITY_DNA,
    ENTITY_DNA_RNA_HYBRID,
    ENTITY_MOLECULE,
    ENTITY_PNA,
    ENTITY_PROTEIN,
    ENTITY_RNA,
    AtomSite,
    ChemicalComponent,
    ComponentReference,
    MolecularSystem,
    Molecule,
    PolymerSequence,
    replace_components,
)
from ...data.structure import Structure
from ...modules.geometry import uniform_random_rotation
from ...processor import Processor, UnresolvedTrainingCropError
from .config import ProtenixProcessorConfig
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
_POLYMER_TYPE = {
    ENTITY_PROTEIN: "protein",
    ENTITY_RNA: "rna",
    ENTITY_DNA: "dna",
    ENTITY_DNA_RNA_HYBRID: "dna_rna_hybrid",
    ENTITY_PNA: "pna",
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


@dataclass(frozen=True)
class _ProtenixTrainingContext:
    full_tokens: list[dict[str, object]]
    current_tokens: list[dict[str, object]]
    crop_method: int | None


class ProtenixCropIndex(Protocol):
    r"""Sampling-row metadata needed to choose spatial crop centres."""

    type: str
    reference_instance_ids: tuple[str, ...]


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
        if self.config.ccd_revision is not None and self.config.ccd_snapshot_path is not None:
            raise ValueError("configure at most one of ccd_revision or ccd_snapshot_path")
        if self.config.ccd_snapshot_path is not None:
            self.config.ccd_snapshot_path = str(validate_ccd_snapshot_path(self.config.ccd_snapshot_path))
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
            angstroms. ``atom_index`` and ``token_component_index`` bind the
            family axes to the canonical molecular graph.
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
        max_tokens: int | None = None,
        crop_method_weights: Sequence[float] = (0.2, 0.4, 0.4),
        indices: ProtenixCropIndex | None = None,
    ) -> tuple[ProtenixInput, ProtenixStructureTarget]:
        r"""
        Lower one structure into cropped model input and full coordinate truth.

        Args:
            input: Full canonical structure used for topology, resolved crop
                centres, and coordinate truth.
            msa: Optional protein MSA evidence.
            generator: Optional generator shared by crop-method selection, chain
                order, crop boundaries, spatial-centre selection, and ordinary
                feature augmentation. CPU and CUDA generators are supported.
            max_tokens: Maximum model-input token count. ``None`` disables
                cropping. Structures already at or below the limit take the
                ordinary lowering path without consuming crop randomness.
            crop_method_weights: Non-negative finite weights for continuous,
                spatial, and interface-spatial cropping, in that order.
            indices: Sampled cluster-row metadata. It is required whenever an
                over-limit structure has positive spatial or interface-spatial
                weight; continuous-only cropping does not require it.

        Returns:
            A possibly cropped ``ProtenixInput`` and a
            ``ProtenixStructureTarget`` that retains the complete canonical atom
            axis. ``input.atom_index`` maps cropped model atoms into that target.
        """

        _validate_crop_arguments(max_tokens, crop_method_weights)
        structure = _hydrate_structure(
            input,
            revision=self.config.ccd_revision,
            snapshot_path=self.config.ccd_snapshot_path,
        )
        training_context: list[_ProtenixTrainingContext] = []
        processed = self._lower(
            structure,
            msa=msa,
            crop_max_tokens=max_tokens,
            crop_method_weights=tuple(float(weight) for weight in crop_method_weights),
            crop_indices=indices,
            generator=generator,
            _training_context=training_context,
        )
        return processed, _structure_target(
            structure,
            processed,
            training_context[0],
            device=self.device,
        )

    def _resolve_msa(
        self,
        msa: MSA | MSACollection | None,
        system: MolecularSystem,
    ) -> tuple[tuple[MSA, int | None, int], ...]:
        if msa is None:
            return ()
        resolved = resolve_msa_bindings(msa, system)
        for _value, _target_instance_index, entity_index in resolved:
            if int(system.entity_type[entity_index]) != ENTITY_PROTEIN:
                raise ValueError("Protenix accepts supplied MSA evidence only for proteins")
        return resolved

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
        crop_max_tokens: int | None = None,
        crop_method_weights: tuple[float, float, float] = (0.2, 0.4, 0.4),
        crop_indices: ProtenixCropIndex | None = None,
        _training_context: list[_ProtenixTrainingContext] | None = None,
    ) -> ProtenixInput:
        atom_contact_values = tuple(atom_contacts)
        token_contact_values = tuple(token_contacts)
        system = molecular_system_from_input(
            input,
            revision=self.config.ccd_revision,
            snapshot_path=self.config.ccd_snapshot_path,
        )
        resolved_msa = self._resolve_msa(msa, system)
        # Atom contacts require atom-level tokens. Token contacts and pockets use
        # the tokenization already selected for each residue.
        expanded_residues = _atom_constraint_residues(atom_contact_values, system)
        tokens = _tokenize(system, expanded_residues)
        full_tokens = tokens
        crop_method: int | None = None
        if crop_max_tokens is not None and len(tokens) > crop_max_tokens:
            if crop_indices is None and any(crop_method_weights[index] > 0 for index in (1, 2)):
                raise ValueError("indices are required when spatial crop methods have positive weight")
            tokens, crop_method = _crop_training_tokens(
                input=cast(Structure, input),
                tokens=tokens,
                max_tokens=crop_max_tokens,
                method_weights=crop_method_weights,
                indices=crop_indices,
                generator=generator,
            )
        processed = _features(
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
            tfg_guidance=tfg_guidance,
            generator=generator,
            max_msa_rows=self.config.max_msa_rows,
            augment_reference_positions=self.config.augment_reference_positions,
            device=self.device,
            dtype=self.dtype,
        )
        if _training_context is not None:
            _training_context.append(
                _ProtenixTrainingContext(
                    full_tokens=full_tokens,
                    current_tokens=tokens,
                    crop_method=crop_method,
                )
            )
        return processed


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
        Lower template-capable input and separate coordinate truth.
        """

        structure = _hydrate_structure(
            input,
            revision=self.config.ccd_revision,
            snapshot_path=self.config.ccd_snapshot_path,
        )
        training_context: list[_ProtenixTrainingContext] = []
        processed = self._lower(
            structure,
            msa=msa,
            templates=tuple(templates),
            generator=generator,
            _training_context=training_context,
        )
        return processed, _structure_target(
            structure,
            processed,
            training_context[0],
            device=self.device,
        )

    def _resolve_msa(
        self,
        msa: MSA | MSACollection | None,
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
        Lower structure-conditioned input and separate coordinate truth.
        """

        structure = _hydrate_structure(
            input,
            revision=self.config.ccd_revision,
            snapshot_path=self.config.ccd_snapshot_path,
        )
        training_context: list[_ProtenixTrainingContext] = []
        processed = self._lower(
            structure,
            structure_condition=structure_condition,
            msa=msa,
            generator=generator,
            _training_context=training_context,
        )
        return processed, _structure_target(
            structure,
            processed,
            training_context[0],
            device=self.device,
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
        Lower constrained input and separate coordinate truth.
        """

        structure = _hydrate_structure(
            input,
            revision=self.config.ccd_revision,
            snapshot_path=self.config.ccd_snapshot_path,
        )
        training_context: list[_ProtenixTrainingContext] = []
        processed = self._lower(
            structure,
            msa=msa,
            atom_contacts=atom_contacts,
            token_contacts=token_contacts,
            pocket=pocket,
            substructure=substructure,
            substructure_coordinate_noise_scale=substructure_coordinate_noise_scale,
            generator=generator,
            _training_context=training_context,
        )
        return processed, _structure_target(
            structure,
            processed,
            training_context[0],
            device=self.device,
        )


def _structure_target(
    structure: Structure,
    input: ProtenixInput,
    context: _ProtenixTrainingContext,
    *,
    device: torch.device,
) -> ProtenixStructureTarget:
    r"""
    Lower full-axis coordinate truth and deterministic loss/permutation metadata.
    """

    system = structure.system
    molecule_id, molecule_equivalence_index, molecule_atom_index, atom_molecule_type = _molecule_metadata(
        system,
        structure.coordinate_mask,
    )
    model_atom_index = tuple(int(index) for index in input.atom_index[0].detach().cpu().tolist() if index >= 0)
    representative_atom_mask = torch.zeros(system.atom_symbol.numel(), dtype=torch.bool)
    representative_atom_mask[
        torch.tensor(
            [cast(int, token["center"]) for token in context.full_tokens],
            dtype=torch.long,
        )
    ] = True
    permutation_pool_mask = _permutation_pool_mask(
        model_atom_index,
        molecule_id,
        molecule_equivalence_index,
        molecule_atom_index,
        restrict_to_current_molecules=context.crop_method in {1, 2},
    )
    symmetry_atom_index, symmetry_permutation = _local_symmetry_features(
        system,
        model_atom_index,
    )
    token_center_atom_index, token_frame_atom_index, token_has_frame = _token_frame_features(
        system,
        context.current_tokens,
        model_atom_index,
    )
    plddt_rep_atom_mask = _plddt_rep_atom_mask(system, model_atom_index)
    ligand_polymer_bond_index, ligand_polymer_bond_mask = _ligand_polymer_bond_features(
        system,
        model_atom_index,
    )
    if structure.resolution is None or structure.resolution_is_known is None:
        resolution = torch.tensor([-1.0], dtype=torch.float32, device=device)
        resolution_is_known = torch.tensor([False], dtype=torch.bool, device=device)
    else:
        resolution = structure.resolution.reshape(-1)
        resolution_is_known = structure.resolution_is_known.reshape(-1)
        if resolution.numel() != 1 or resolution_is_known.numel() != 1:
            raise ValueError("Protenix process_training requires scalar structure resolution metadata")
        resolution = resolution.to(device=device, dtype=torch.float32)
        resolution_is_known = resolution_is_known.to(device=device, dtype=torch.bool)

    def placed(value: Tensor) -> Tensor:
        return value.to(device=device).unsqueeze(0)

    return ProtenixStructureTarget(  # type: ignore[call-arg]
        coordinates=structure.coordinates.to(device=device, dtype=torch.float32).unsqueeze(0),
        coordinate_mask=structure.coordinate_mask.to(device=device).unsqueeze(0),
        molecule_id=placed(molecule_id),
        molecule_equivalence_index=placed(molecule_equivalence_index),
        molecule_atom_index=placed(molecule_atom_index),
        representative_atom_mask=placed(representative_atom_mask),
        permutation_pool_mask=placed(permutation_pool_mask),
        atom_molecule_type=placed(atom_molecule_type),
        symmetry_atom_index=placed(symmetry_atom_index),
        symmetry_permutation=placed(symmetry_permutation),
        token_center_atom_index=placed(token_center_atom_index),
        token_frame_atom_index=placed(token_frame_atom_index),
        token_has_frame=placed(token_has_frame),
        plddt_rep_atom_mask=placed(plddt_rep_atom_mask),
        ligand_polymer_bond_index=placed(ligand_polymer_bond_index),
        ligand_polymer_bond_mask=placed(ligand_polymer_bond_mask),
        resolution=resolution,
        resolution_is_known=resolution_is_known,
        batch_size=[1],
    )


def _molecule_metadata(
    system: MolecularSystem,
    coordinate_mask: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    r"""
    Assign baseline-style instance-graph molecules and exact-layout equivalence.

    Every instance starts as one molecule, even when its component graph is
    disconnected. Resolved inter-instance covalent bonds merge instance nodes,
    except polymer-polymer links such as disulfides. Molecule ids follow first
    canonical atom occurrence. Equivalence intentionally includes both entity
    composition and the complete ordered atom layout, so incomplete or otherwise
    non-isomorphic copies never become permutation peers.
    """

    atom_count = system.atom_symbol.numel()
    if coordinate_mask.numel() != atom_count:
        raise ValueError("coordinate_mask must align with the canonical atom axis")
    resolved = coordinate_mask.reshape(-1).detach().cpu().bool().tolist()
    instance_count = system.instance_symbol.numel()
    parent = list(range(instance_count))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        first_root = root(first)
        second_root = root(second)
        if first_root == second_root:
            return
        if first_root < second_root:
            parent[second_root] = first_root
        else:
            parent[first_root] = second_root

    component_instance = torch.repeat_interleave(
        torch.arange(instance_count, dtype=torch.long),
        (system.instance_component_ptr[1:] - system.instance_component_ptr[:-1]).detach().cpu(),
    ).tolist()
    atom_component = torch.repeat_interleave(
        torch.arange(system.component_symbol.numel(), dtype=torch.long),
        (system.component_atom_ptr[1:] - system.component_atom_ptr[:-1]).detach().cpu(),
    ).tolist()
    atom_instance = [component_instance[component_index] for component_index in atom_component]
    instance_entities = system.instance_entity.detach().cpu().tolist()
    entity_types = system.entity_type.detach().cpu().tolist()
    polymer_types = {
        ENTITY_PROTEIN,
        ENTITY_RNA,
        ENTITY_DNA,
        ENTITY_DNA_RNA_HYBRID,
        ENTITY_PNA,
    }

    edge_index = system.edge_index.detach().cpu()
    bond_kind = system.bond_kind.detach().cpu().tolist()
    for edge_offset, (first, second) in enumerate(edge_index.T.tolist()):
        first_instance = atom_instance[first]
        second_instance = atom_instance[second]
        if first_instance == second_instance or bond_kind[edge_offset] != BOND_KIND_COVALENT:
            continue
        if not resolved[first] or not resolved[second]:
            continue
        first_type = entity_types[instance_entities[first_instance]]
        second_type = entity_types[instance_entities[second_instance]]
        if first_type in polymer_types and second_type in polymer_types:
            continue
        union(first_instance, second_instance)

    atoms_by_root: dict[int, list[int]] = {}
    for atom_index in range(atom_count):
        atoms_by_root.setdefault(root(atom_instance[atom_index]), []).append(atom_index)
    molecule_atoms = list(atoms_by_root.values())

    instance_components = system.instance_component_ptr.detach().cpu().tolist()
    component_atoms = system.component_atom_ptr.detach().cpu().tolist()
    component_symbols = system.component_symbol.detach().cpu().tolist()
    atom_symbols = system.atom_symbol.detach().cpu().tolist()

    molecule_id = torch.empty(atom_count, dtype=torch.long)
    molecule_atom_index = torch.empty(atom_count, dtype=torch.long)
    molecule_equivalence_index = torch.empty(atom_count, dtype=torch.long)
    equivalence_by_layout: dict[tuple[object, ...], int] = {}
    for current_molecule_id, atom_indices in enumerate(molecule_atoms):
        instance_order = tuple(dict.fromkeys(atom_instance[atom_index] for atom_index in atom_indices))
        instance_rank = {instance_index: rank for rank, instance_index in enumerate(instance_order)}
        entity_composition = tuple(sorted(instance_entities[instance_index] for instance_index in instance_order))
        atom_layout = tuple(
            (
                instance_entities[atom_instance[atom_index]],
                instance_rank[atom_instance[atom_index]],
                atom_component[atom_index] - instance_components[atom_instance[atom_index]],
                component_symbols[atom_component[atom_index]],
                atom_index - component_atoms[atom_component[atom_index]],
                atom_symbols[atom_index],
            )
            for atom_index in atom_indices
        )
        layout_key: tuple[object, ...] = (entity_composition, atom_layout)
        equivalence_index = equivalence_by_layout.setdefault(layout_key, len(equivalence_by_layout))
        molecule_id[atom_indices] = current_molecule_id
        molecule_atom_index[atom_indices] = torch.arange(len(atom_indices))
        molecule_equivalence_index[atom_indices] = equivalence_index

    atom_molecule_type = torch.tensor(
        [entity_types[instance_entities[atom_instance[index]]] for index in range(atom_count)],
        dtype=torch.long,
    )
    return molecule_id, molecule_equivalence_index, molecule_atom_index, atom_molecule_type


def _permutation_pool_mask(
    current_atom_index: tuple[int, ...],
    molecule_id: Tensor,
    molecule_equivalence_index: Tensor,
    molecule_atom_index: Tensor,
    *,
    restrict_to_current_molecules: bool,
) -> Tensor:
    current_pairs = {
        (int(molecule_equivalence_index[index]), int(molecule_atom_index[index])) for index in current_atom_index
    }
    current_molecules = {int(molecule_id[index]) for index in current_atom_index}
    return torch.tensor(
        [
            (int(equivalence), int(atom_index)) in current_pairs
            and (not restrict_to_current_molecules or int(molecule) in current_molecules)
            for molecule, equivalence, atom_index in zip(
                molecule_id.tolist(),
                molecule_equivalence_index.tolist(),
                molecule_atom_index.tolist(),
            )
        ],
        dtype=torch.bool,
    )


def _local_symmetry_features(
    system: MolecularSystem,
    atom_indices: tuple[int, ...],
) -> tuple[Tensor, Tensor]:
    r"""Build baseline-compatible CCD permutations on the current model axis."""

    system_to_model_atom = {atom_index: model_index for model_index, atom_index in enumerate(atom_indices)}
    component_instance = torch.repeat_interleave(
        torch.arange(system.instance_symbol.numel(), dtype=torch.long),
        (system.instance_component_ptr[1:] - system.instance_component_ptr[:-1]).detach().cpu(),
    ).tolist()
    instance_entities = system.instance_entity.detach().cpu().tolist()
    entity_types = system.entity_type.detach().cpu().tolist()
    component_symbols = system.component_symbol.detach().cpu().tolist()
    component_atoms = system.component_atom_ptr.detach().cpu().tolist()
    atom_symbols = system.atom_symbol.detach().cpu().tolist()
    current_atom_indices = set(atom_indices)
    groups: list[tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]] = []
    for component_index, instance_index in enumerate(component_instance):
        entity_type = entity_types[instance_entities[instance_index]]
        atom_start = component_atoms[component_index]
        atom_stop = component_atoms[component_index + 1]
        visible_local = [
            atom_index - atom_start
            for atom_index in range(atom_start, atom_stop)
            if atom_index in system_to_model_atom
        ]
        if len(visible_local) < 2:
            continue
        identity = tuple(system_to_model_atom[atom_start + local_index] for local_index in visible_local)
        candidates = {identity}
        boundary_local = {
            atom_index - atom_start
            for atom_index in _component_boundary_atoms(
                system,
                component_index,
                current_atom_indices,
            )
        }
        if bool(system.component_topology_complete[component_index]):
            try:
                molecule = _rdkit_component(system, component_index)
                for boundary_offset, local_index in enumerate(sorted(boundary_local), 1):
                    molecule.GetAtomWithIdx(local_index).SetIsotope(1000 + boundary_offset)
                for permutation in _ccd_component_permutations(molecule, max_matches=1000):
                    if any(permutation[index] != index for index in boundary_local):
                        continue
                    mapped_local = tuple(permutation[index] for index in visible_local)
                    if set(mapped_local) != set(visible_local):
                        continue
                    candidates.add(
                        tuple(system_to_model_atom[atom_start + index] for index in mapped_local)
                    )
            except (KeyError, RuntimeError, ValueError):
                pass
        if len(candidates) == 1 and _is_standard(system, component_index, entity_type):
            component_id = system.symbols[component_symbols[component_index]]
            standard_permutations = _STANDARD_LOCAL_SYMMETRY.get(component_id)
            if standard_permutations is not None:
                by_name = {
                    system.symbols[atom_symbols[atom_start + local_index]]: local_index
                    for local_index in visible_local
                }
                source_names, target_names = standard_permutations
                if all(name in by_name for name in (*source_names, *target_names)):
                    local_permutation = list(range(atom_stop - atom_start))
                    for source_name, target_name in zip(source_names, target_names):
                        local_permutation[by_name[source_name]] = by_name[target_name]
                    if not any(local_permutation[index] != index for index in boundary_local):
                        mapped_local = tuple(local_permutation[index] for index in visible_local)
                        candidates.add(
                            tuple(system_to_model_atom[atom_start + index] for index in mapped_local)
                        )
        if len(candidates) > 1:
            groups.append((identity, (identity, *sorted(candidates - {identity}))))

    if not groups:
        return torch.empty((0, 0), dtype=torch.long), torch.empty((0, 0, 0), dtype=torch.long)
    max_width = max(len(group) for group, _ in groups)
    max_permutations = max(len(permutations) for _, permutations in groups)
    group_atom_index = torch.full((len(groups), max_width), -1, dtype=torch.long)
    permutation = torch.full((len(groups), max_permutations, max_width), -1, dtype=torch.long)
    for group_index, (atom_group, permutations) in enumerate(groups):
        group_atom_index[group_index, : len(atom_group)] = torch.tensor(atom_group)
        for permutation_index, values in enumerate(permutations):
            permutation[group_index, permutation_index, : len(values)] = torch.tensor(values)
    return group_atom_index, permutation


def _neutralize_symmetry_atoms(molecule: Mol) -> Mol:
    r"""Apply the baseline charge/H-count neutralization used for CCD symmetry."""

    from rdkit import Chem

    pattern = Chem.MolFromSmarts(
        "[+1!h0!$([*]~[-1,-2,-3,-4]),-1!#4!#5!$([*]~[+1,+2,+3,+4])]"
    )
    if pattern is None:
        raise RuntimeError("failed to construct the CCD symmetry neutralization pattern")
    for (atom_index,) in molecule.GetSubstructMatches(pattern):
        atom = molecule.GetAtomWithIdx(atom_index)
        charge = atom.GetFormalCharge()
        atom.SetFormalCharge(0)
        atom.SetNumExplicitHs(atom.GetTotalNumHs() - charge)
        atom.UpdatePropertyCache()
    return molecule


def _augment_conjugated_terminal_permutations(
    molecule: Mol,
    permutations: tuple[tuple[int, ...], ...],
    *,
    max_matches: int,
) -> tuple[tuple[int, ...], ...]:
    r"""Augment automorphisms with baseline conjugated terminal-group swaps."""

    from rdkit import Chem

    pattern = Chem.MolFromSmarts(
        "[O,N;D1;$([O,N;D1]-[*]=[O,N;D1]),$([O,N;D1]=[*]-[O,N;D1])]~[*]"
    )
    if pattern is None:
        raise RuntimeError("failed to construct the conjugated terminal-group pattern")
    graph: defaultdict[int, set[int]] = defaultdict(set)
    for first, second in molecule.GetSubstructMatches(pattern):
        graph[first].add(second)
        graph[second].add(first)

    terminal_groups: list[tuple[int, ...]] = []
    for neighbors in graph.values():
        if len(neighbors) < 2:
            continue
        by_atomic_number: defaultdict[int, set[int]] = defaultdict(set)
        for atom_index in neighbors:
            by_atomic_number[molecule.GetAtomWithIdx(atom_index).GetAtomicNum()].add(atom_index)
        terminal_groups.extend(
            tuple(sorted(group)) for group in by_atomic_number.values() if len(group) > 1
        )
    terminal_groups.sort()
    if not terminal_groups:
        return permutations

    max_terminal_groups = max(
        1,
        math.ceil(math.log(max_matches / max(1, len(permutations)), 3)),
    )
    terminal_groups = terminal_groups[:max_terminal_groups]
    augmented = set(permutations)
    for group in terminal_groups:
        next_permutations: set[tuple[int, ...]] = set()
        group_orders = tuple(itertools.permutations(group))
        for permutation in augmented:
            positions = [index for index, value in enumerate(permutation) if value in group]
            for order in group_orders:
                candidate = list(permutation)
                for position, value in zip(positions, order):
                    candidate[position] = value
                next_permutations.add(tuple(candidate))
        augmented = set(sorted(next_permutations)[:max_matches])
    return tuple(sorted(augmented)[:max_matches])


def _rdkit_substructure_permutations(
    molecule: Mol,
    *,
    neutralize: bool,
    max_matches: int,
) -> tuple[tuple[int, ...], ...]:
    r"""Compute one protonation branch of baseline CCD automorphisms."""

    from rdkit import Chem

    working = Chem.Mol(molecule)
    original_atom_count = working.GetNumAtoms()
    for atom in working.GetAtoms():
        atom.SetIntProp("ofoldx_original_atom_index", atom.GetIdx())
    working = Chem.RemoveHs(working)
    if neutralize:
        working = _neutralize_symmetry_atoms(working)
    Chem.AssignStereochemistry(working, force=True, cleanIt=True)
    matches = working.GetSubstructMatches(
        working,
        uniquify=False,
        maxMatches=max_matches,
    )
    if not matches:
        raise RuntimeError("RDKit found no CCD self-match")
    canonical_rank = tuple(Chem.rdmolfiles.CanonicalRankAtoms(working, breakTies=False))
    permutations = tuple(
        tuple(match)
        for match in matches
        if all(canonical_rank[match[index]] == canonical_rank[index] for index in range(len(match)))
    )
    permutations = _augment_conjugated_terminal_permutations(
        working,
        permutations,
        max_matches=max_matches,
    )

    new_to_original = {
        atom.GetIdx(): atom.GetIntProp("ofoldx_original_atom_index") for atom in working.GetAtoms()
    }
    original_to_new = {original: new for new, original in new_to_original.items()}
    expanded: list[tuple[int, ...]] = []
    for permutation in permutations:
        values = []
        for original_index in range(original_atom_count):
            if original_index not in original_to_new:
                values.append(original_index)
                continue
            new_index = original_to_new[original_index]
            values.append(new_to_original[permutation[new_index]])
        expanded.append(tuple(values))
    return tuple(expanded)


def _ccd_component_permutations(
    molecule: Mol,
    *,
    max_matches: int,
) -> tuple[tuple[int, ...], ...]:
    r"""Return deterministic identity-first CCD automorphisms across protonation states."""

    candidates: set[tuple[int, ...]] = set()
    for neutralize in (False, True):
        candidates.update(
            _rdkit_substructure_permutations(
                molecule,
                neutralize=neutralize,
                max_matches=max_matches,
            )
        )
    identity = tuple(range(molecule.GetNumAtoms()))
    candidates.add(identity)
    return (identity, *sorted(candidates - {identity}))[:max_matches]


def _component_boundary_atoms(
    system: MolecularSystem,
    component_index: int,
    current_atom_indices: set[int],
) -> set[int]:
    r"""Return component boundary atoms whose bonded partners survive the crop."""

    atom_start = int(system.component_atom_ptr[component_index])
    atom_stop = int(system.component_atom_ptr[component_index + 1])
    boundary: set[int] = set()
    for first, second in system.edge_index.detach().cpu().T.tolist():
        if first not in current_atom_indices or second not in current_atom_indices:
            continue
        first_inside = atom_start <= first < atom_stop
        second_inside = atom_start <= second < atom_stop
        if first_inside and not second_inside:
            boundary.add(first)
        elif second_inside and not first_inside:
            boundary.add(second)
    return boundary


def _token_frame_features(
    system: MolecularSystem,
    tokens: list[dict[str, object]],
    atom_indices: tuple[int, ...],
) -> tuple[Tensor, Tensor, Tensor]:
    system_to_model_atom = {atom_index: model_index for model_index, atom_index in enumerate(atom_indices)}
    atom_symbols = system.atom_symbol.detach().cpu().tolist()
    reference_positions = system.reference_coordinates.detach().cpu()
    reference_mask = system.reference_coordinate_mask.detach().cpu().bool()
    reference_groups: dict[int, list[int]] = {}
    for token in tokens:
        reference_groups.setdefault(cast(int, token["reference_space"]), []).extend(
            cast(tuple[int, ...], token["atoms"])
        )

    center_atom_index: list[int] = []
    frame_atom_index: list[tuple[int, int, int]] = []
    has_frame: list[bool] = []
    for token in tokens:
        center = cast(int, token["center"])
        center_model_index = system_to_model_atom[center]
        center_atom_index.append(center_model_index)
        token_atoms = cast(tuple[int, ...], token["atoms"])
        entity_type = int(system.entity_type[cast(int, token["entity_index"])])
        standard_aggregate = cast(bool, token["standard"]) and len(token_atoms) > 1
        if standard_aggregate and entity_type in {
            ENTITY_PROTEIN,
            ENTITY_RNA,
            ENTITY_DNA,
            ENTITY_DNA_RNA_HYBRID,
        }:
            names = ("N", "CA", "C") if entity_type == ENTITY_PROTEIN else ("C1'", "C3'", "C4'")
            by_name = {system.symbols[atom_symbols[index]]: index for index in token_atoms}
            if all(name in by_name for name in names):
                frame_atom_index.append(tuple(system_to_model_atom[by_name[name]] for name in names))
                has_frame.append(True)
            else:
                frame_atom_index.append((-1, -1, -1))
                has_frame.append(False)
            continue

        candidates = [
            atom_index
            for atom_index in reference_groups[cast(int, token["reference_space"])]
            if atom_index != center and atom_index in system_to_model_atom
        ]
        candidates.sort(
            key=lambda atom_index: (
                float(torch.sum((reference_positions[atom_index] - reference_positions[center]) ** 2)),
                system_to_model_atom[atom_index],
            )
        )
        if len(candidates) < 2:
            frame_atom_index.append((-1, center_model_index, -1))
            has_frame.append(False)
            continue
        first, third = candidates[:2]
        frame_atom_index.append(
            (system_to_model_atom[first], center_model_index, system_to_model_atom[third])
        )
        valid = bool(reference_mask[[first, center, third]].all())
        first_vector = reference_positions[first] - reference_positions[center]
        third_vector = reference_positions[third] - reference_positions[center]
        first_norm = float(torch.linalg.vector_norm(first_vector))
        third_norm = float(torch.linalg.vector_norm(third_vector))
        if first_norm == 0.0 or third_norm == 0.0:
            valid = False
        elif valid:
            cosine = float(torch.dot(first_vector, third_vector) / (first_norm * third_norm))
            angle = math.degrees(math.acos(min(1.0, max(-1.0, cosine))))
            valid = 25.0 < angle < 155.0
        has_frame.append(valid)

    return (
        torch.tensor(center_atom_index, dtype=torch.long),
        torch.tensor(frame_atom_index, dtype=torch.long).reshape(-1, 3),
        torch.tensor(has_frame, dtype=torch.bool),
    )


def _plddt_rep_atom_mask(system: MolecularSystem, atom_indices: tuple[int, ...]) -> Tensor:
    component_instance = torch.repeat_interleave(
        torch.arange(system.instance_symbol.numel(), dtype=torch.long),
        (system.instance_component_ptr[1:] - system.instance_component_ptr[:-1]).detach().cpu(),
    ).tolist()
    atom_component = torch.repeat_interleave(
        torch.arange(system.component_symbol.numel(), dtype=torch.long),
        (system.component_atom_ptr[1:] - system.component_atom_ptr[:-1]).detach().cpu(),
    ).tolist()
    instance_entities = system.instance_entity.detach().cpu().tolist()
    entity_types = system.entity_type.detach().cpu().tolist()
    atom_symbols = system.atom_symbol.detach().cpu().tolist()
    mask = []
    for atom_index in atom_indices:
        component_index = atom_component[atom_index]
        entity_type = entity_types[instance_entities[component_instance[component_index]]]
        atom_name = system.symbols[atom_symbols[atom_index]]
        expected_name = "CA" if entity_type == ENTITY_PROTEIN else "C1'"
        mask.append(
            entity_type in {ENTITY_PROTEIN, ENTITY_RNA, ENTITY_DNA, ENTITY_DNA_RNA_HYBRID}
            and _is_standard(system, component_index, entity_type)
            and atom_name == expected_name
        )
    return torch.tensor(mask, dtype=torch.bool)


def _ligand_polymer_bond_features(
    system: MolecularSystem,
    atom_indices: tuple[int, ...],
) -> tuple[Tensor, Tensor]:
    system_to_model_atom = {atom_index: model_index for model_index, atom_index in enumerate(atom_indices)}
    component_instance = torch.repeat_interleave(
        torch.arange(system.instance_symbol.numel(), dtype=torch.long),
        (system.instance_component_ptr[1:] - system.instance_component_ptr[:-1]).detach().cpu(),
    ).tolist()
    atom_component = torch.repeat_interleave(
        torch.arange(system.component_symbol.numel(), dtype=torch.long),
        (system.component_atom_ptr[1:] - system.component_atom_ptr[:-1]).detach().cpu(),
    ).tolist()
    atom_instance = [component_instance[component_index] for component_index in atom_component]
    instance_entities = system.instance_entity.detach().cpu().tolist()
    entity_types = system.entity_type.detach().cpu().tolist()
    visible_atom_count_by_instance: dict[int, int] = {}
    for atom_index in atom_indices:
        instance_index = atom_instance[atom_index]
        visible_atom_count_by_instance[instance_index] = (
            visible_atom_count_by_instance.get(instance_index, 0) + 1
        )
    polymer_types = {
        ENTITY_PROTEIN,
        ENTITY_RNA,
        ENTITY_DNA,
        ENTITY_DNA_RNA_HYBRID,
        ENTITY_PNA,
    }
    rows: list[tuple[int, int]] = []
    for edge_offset, (first, second) in enumerate(system.edge_index.detach().cpu().T.tolist()):
        if int(system.bond_kind[edge_offset]) != BOND_KIND_COVALENT:
            continue
        if first not in system_to_model_atom or second not in system_to_model_atom:
            continue
        first_instance = atom_instance[first]
        second_instance = atom_instance[second]
        first_type = entity_types[instance_entities[first_instance]]
        second_type = entity_types[instance_entities[second_instance]]
        if first_type == ENTITY_MOLECULE and second_type in polymer_types:
            ligand_instance = first_instance
        elif second_type == ENTITY_MOLECULE and first_type in polymer_types:
            ligand_instance = second_instance
        else:
            continue
        if visible_atom_count_by_instance.get(ligand_instance, 0) <= 1:
            continue
        rows.append((system_to_model_atom[first], system_to_model_atom[second]))
    rows.sort()
    index = torch.tensor(rows, dtype=torch.long).reshape(-1, 2)
    return index, torch.ones(index.shape[0], dtype=torch.bool)


def _validate_crop_arguments(max_tokens: int | None, method_weights: Sequence[float]) -> None:
    if max_tokens is not None and (isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0):
        raise ValueError("max_tokens must be a positive integer or None")
    if isinstance(method_weights, (str, bytes)):
        raise ValueError("crop_method_weights must be a numeric sequence")
    try:
        weight_count = len(method_weights)
    except TypeError as error:
        raise ValueError("crop_method_weights must be a numeric sequence") from error
    if weight_count != 3:
        raise ValueError("crop_method_weights must contain exactly three values")
    try:
        weights = tuple(float(weight) for weight in method_weights)
    except (TypeError, ValueError) as error:
        raise ValueError("crop_method_weights must be a numeric sequence") from error
    if not all(math.isfinite(weight) for weight in weights):
        raise ValueError("crop_method_weights must be finite")
    if any(weight < 0 for weight in weights):
        raise ValueError("crop_method_weights must be non-negative")
    if sum(weights) <= 0:
        raise ValueError("crop_method_weights must have a positive sum")


def _generator_device(generator: torch.Generator | None) -> torch.device:
    return torch.device("cpu") if generator is None else torch.device(generator.device)


def _random_integer(high: int, generator: torch.Generator | None, *, low: int = 0) -> int:
    if high <= low:
        return low
    return int(
        torch.randint(
            low,
            high,
            (),
            generator=generator,
            device=_generator_device(generator),
        ).item()
    )


def _random_permutation(length: int, generator: torch.Generator | None) -> list[int]:
    return torch.randperm(
        length,
        generator=generator,
        device=_generator_device(generator),
    ).detach().cpu().tolist()


def _drop_partial_reference_spaces(
    tokens: list[dict[str, object]],
    selected_indices: Sequence[int],
) -> list[int]:
    total: dict[int, int] = {}
    selected: dict[int, int] = {}
    for token in tokens:
        reference_space = cast(int, token["reference_space"])
        total[reference_space] = total.get(reference_space, 0) + 1
    for index in selected_indices:
        reference_space = cast(int, tokens[index]["reference_space"])
        selected[reference_space] = selected.get(reference_space, 0) + 1
    return [
        index
        for index in selected_indices
        if selected[cast(int, tokens[index]["reference_space"])]
        == total[cast(int, tokens[index]["reference_space"])]
    ]


def _continuous_crop_indices(
    system: MolecularSystem,
    tokens: list[dict[str, object]],
    max_tokens: int,
    generator: torch.Generator | None,
) -> list[int]:
    del system
    chains: list[list[int]] = []
    for token_index, token in enumerate(tokens):
        chain_index = cast(int, token["chain_index"])
        while len(chains) <= chain_index:
            chains.append([])
        chains[chain_index].append(token_index)

    classified_single_atom: list[bool] = []
    for chain in chains:
        if not chain:
            classified_single_atom.append(False)
            continue
        only_atoms = cast(tuple[int, ...], tokens[chain[0]]["atoms"])
        classified_single_atom.append(len(chain) == 1 and len(only_atoms) == 1)

    first_indices = [0] * len(tokens)
    last_indices = [0] * len(tokens)
    reference_groups: dict[int, list[int]] = {}
    for token_index, token in enumerate(tokens):
        reference_groups.setdefault(cast(int, token["reference_space"]), []).append(token_index)
    for group in reference_groups.values():
        for token_index in group:
            first_indices[token_index] = group[0]
            last_indices[token_index] = group[-1]

    def qualifies(current_size: int, minimum_size: int, already_added: int) -> bool:
        return current_size >= minimum_size and current_size + already_added <= max_tokens

    def complete_boundaries(
        start: int,
        end: int,
        minimum_size: int,
        already_added: int,
    ) -> tuple[int, int]:
        if start == end:
            return start, end

        left_start = right_start = start
        if first_indices[start] != start:
            start_in_middle = True
            left_start = first_indices[start]
            right_start = last_indices[start] + 1
        else:
            start_in_middle = False

        left_end = right_end = end
        if end > 0 and last_indices[end - 1] != end - 1:
            end_in_middle = True
            left_end = first_indices[end - 1]
            right_end = last_indices[end - 1] + 1
        else:
            end_in_middle = False

        if not start_in_middle and not end_in_middle:
            return start, end
        if start_in_middle and end_in_middle:
            # This matches upstream's deterministic "always use left edge"
            # rule when both sampled boundaries split reference spaces.
            start_in_middle = False
            start = left_start

        if not start_in_middle and end_in_middle:
            left_size = left_end - start
            right_size = right_end - start
            left_ok = qualifies(left_size, minimum_size, already_added)
            right_ok = qualifies(right_size, minimum_size, already_added)
            if left_ok and right_ok:
                return (start, left_end) if _random_integer(2, generator) == 0 else (start, right_end)
            if left_ok:
                return start, left_end
            if right_ok:
                return start, right_end
            # ``drop_last=True``: repeatedly discard the final complete
            # reference space until the global token quota is respected.
            completed_end = left_end
            while completed_end - start + already_added > max_tokens:
                if completed_end > start:
                    completed_end = first_indices[completed_end - 1]
                else:
                    break
            return start, completed_end

        # Only the start boundary is in the middle of a reference space.
        left_size = end - left_start
        right_size = end - right_start
        left_ok = qualifies(left_size, minimum_size, already_added)
        right_ok = qualifies(right_size, minimum_size, already_added)
        if left_ok and right_ok:
            return (left_start, end) if _random_integer(2, generator) == 0 else (right_start, end)
        if left_ok:
            return left_start, end
        if right_ok:
            return right_start, end
        # ``drop_last=True`` selects the right edge, dropping the incomplete
        # leading reference space rather than returning a partial group.
        return right_start, end

    selected: list[int] = []
    remaining = len(tokens) - sum(classified_single_atom)
    for chain_offset in _random_permutation(len(chains), generator):
        if len(selected) >= max_tokens:
            break
        chain = chains[chain_offset]
        if classified_single_atom[chain_offset]:
            continue
        chain_length = len(chain)
        remaining -= chain_length
        minimum = min(chain_length, max(0, max_tokens - len(selected) - remaining))
        maximum = min(chain_length, max_tokens - len(selected))
        crop_length = _random_integer(maximum + 1, generator, low=minimum)
        crop_start = _random_integer(chain_length - crop_length + 1, generator)
        start = chain[0] + crop_start
        end = start + crop_length
        start, end = complete_boundaries(start, end, minimum, len(selected))
        selected.extend(range(start, end))

    return sorted(selected)


def _spatial_crop_indices(
    structure: Structure,
    tokens: list[dict[str, object]],
    max_tokens: int,
    indices: ProtenixCropIndex,
    generator: torch.Generator | None,
    *,
    interface: bool,
) -> list[int]:
    if indices.type not in {"chain", "interface"}:
        raise ValueError("indices.type must be 'chain' or 'interface'")
    reference_instances = frozenset(indices.reference_instance_ids)
    if not reference_instances:
        raise ValueError("indices.reference_instance_ids must be non-empty for spatial cropping")

    system = structure.system
    center_indices = torch.tensor([cast(int, token["center"]) for token in tokens], dtype=torch.long)
    coordinates = structure.coordinates.detach().cpu()[center_indices]
    resolved = structure.coordinate_mask.detach().cpu()[center_indices].bool()
    chain_indices = torch.tensor([cast(int, token["chain_index"]) for token in tokens], dtype=torch.long)
    sampled = torch.tensor(
        [
            system.instance_id(cast(int, token["instance_index"])) in reference_instances
            for token in tokens
        ],
        dtype=torch.bool,
    )
    if not bool(sampled.any()):
        raise ValueError("indices.reference_instance_ids do not match any model token")
    reference_candidates = torch.nonzero(resolved & sampled, as_tuple=False).flatten()
    if reference_candidates.numel() == 0:
        raise UnresolvedTrainingCropError("sampled crop row has no resolved token centre candidates")
    if interface:
        interface_candidates = []
        resolved_indices = torch.nonzero(resolved, as_tuple=False).flatten()
        for candidate in reference_candidates.tolist():
            other_chain = resolved_indices[chain_indices[resolved_indices] != chain_indices[candidate]]
            if other_chain.numel() and torch.any(
                torch.linalg.vector_norm(coordinates[other_chain] - coordinates[candidate], dim=-1) < 15.0
            ):
                interface_candidates.append(candidate)
        if interface_candidates:
            reference_candidates = torch.tensor(interface_candidates, dtype=torch.long)
        elif indices.type == "chain":
            # A sampled chain without an interface falls back to any of its
            # resolved centres. Interface rows must describe a real interface.
            reference_candidates = torch.nonzero(resolved & sampled, as_tuple=False).flatten()
        else:
            raise ValueError("sampled interface has no resolved <15A cross-chain centre candidate")

    reference_index = int(
        reference_candidates[_random_integer(reference_candidates.numel(), generator)].item()
    )
    distances = torch.linalg.vector_norm(coordinates - coordinates[reference_index], dim=-1)
    distances = torch.where(resolved, distances, torch.full_like(distances, torch.inf))
    canonical_indices = torch.tensor(
        [cast(int, token["token_index"]) for token in tokens],
        dtype=distances.dtype,
    )
    scores = distances + canonical_indices * 1e-3
    nearest = torch.topk(scores, min(max_tokens, len(tokens)), largest=False).indices.sort().values.tolist()
    return _drop_partial_reference_spaces(tokens, nearest)


def _crop_training_tokens(
    *,
    input: Structure,
    tokens: list[dict[str, object]],
    max_tokens: int,
    method_weights: tuple[float, float, float],
    indices: ProtenixCropIndex | None,
    generator: torch.Generator | None,
) -> tuple[list[dict[str, object]], int]:
    weights = torch.tensor(method_weights, dtype=torch.float64, device=_generator_device(generator))
    method = int(torch.multinomial(weights, 1, generator=generator).item())
    if method == 0:
        selected = _continuous_crop_indices(input.system, tokens, max_tokens, generator)
    else:
        if indices is None:  # guarded before method sampling; retained for type narrowing
            raise ValueError("indices are required for spatial cropping")
        selected = _spatial_crop_indices(
            input,
            tokens,
            max_tokens,
            indices,
            generator,
            interface=method == 2,
        )
    if not selected:
        raise ValueError("training crop selected no complete non-metal token groups")
    atom_indices = [
        atom_index
        for token_index in selected
        for atom_index in cast(tuple[int, ...], tokens[token_index]["atoms"])
    ]
    coordinate_mask = input.coordinate_mask
    mask_indices = torch.tensor(atom_indices, dtype=torch.long, device=coordinate_mask.device)
    if not bool(coordinate_mask[mask_indices].any()):
        raise UnresolvedTrainingCropError("training crop contains no resolved atoms to score")
    return [tokens[index] for index in selected], method


def molecular_system_from_input(
    input: PolymerSequence | Sequence[PolymerSequence] | MolecularSystem | Structure,
    *,
    revision: str | None = None,
    snapshot_path: str | None = None,
) -> MolecularSystem:
    r"""
    Normalize one Protenix biological input to its molecular system.

    Args:
        input: One polymer, a polymer collection, a molecular system, or a
            structure whose topology supplies the system.
        revision: Pinned CCD snapshot used to hydrate reference-less components.
        snapshot_path: Local resolver-schema snapshot directory used offline.

    Returns:
        The canonical molecular system. Protenix-visible axes exclude water
        during tokenization without creating a second topology.
    """

    # A Structure supplied as the primary input contributes only topology. The
    # constraint Processor lowers coordinates only from its explicit substructure argument.
    if isinstance(input, Structure):
        system = input.system
    elif isinstance(input, MolecularSystem):
        system = input
    elif isinstance(input, PolymerSequence):
        system = molecular_system_from_named_sequences((("A", input),))
    elif isinstance(input, Sequence):
        sequence_values = tuple(cast(Sequence[object], input))
        if not sequence_values or not all(isinstance(sequence, PolymerSequence) for sequence in sequence_values):
            raise TypeError("Protenix sequence collections must contain PolymerSequence values")
        polymer_sequences = cast(tuple[PolymerSequence, ...], sequence_values)
        system = molecular_system_from_named_sequences(
            tuple((f"chain_{index + 1}", sequence) for index, sequence in enumerate(polymer_sequences))
        )
    else:
        raise TypeError(
            "Protenix input must be a PolymerSequence, sequence of PolymerSequence "
            "values, MolecularSystem, or Structure"
        )
    return _hydrate_system(system, revision=revision, snapshot_path=snapshot_path)[0]


def _hydrate_system(
    system: MolecularSystem,
    *,
    revision: str | None,
    snapshot_path: str | None,
) -> tuple[MolecularSystem, Tensor]:
    r"""Best-effort hydrate reference-less occurrences from a configured CCD."""

    component_count = system.component_symbol.numel()
    atom_count = system.atom_symbol.numel()
    atom_component = torch.repeat_interleave(
        torch.arange(component_count, dtype=torch.int64, device=system.atom_symbol.device),
        system.component_atom_ptr[1:] - system.component_atom_ptr[:-1],
    )
    reference_count = torch.zeros(component_count, dtype=torch.int64, device=system.atom_symbol.device)
    reference_count.scatter_add_(0, atom_component, system.reference_coordinate_mask.to(torch.int64))
    hydrate_mask = (
        (system.component_atom_ptr[1:] == system.component_atom_ptr[:-1])
        | (reference_count == 0)
        | ~system.component_topology_complete
    )
    component_index = torch.nonzero(hydrate_mask, as_tuple=False).flatten()
    if not component_index.numel():
        return system, torch.arange(atom_count, dtype=torch.int64, device=system.atom_symbol.device)

    cache: dict[str, ChemicalComponent | None] = {}

    component_instance = torch.repeat_interleave(
        torch.arange(
            system.instance_symbol.numel(),
            dtype=torch.int64,
            device=system.instance_component_ptr.device,
        ),
        system.instance_component_ptr[1:] - system.instance_component_ptr[:-1],
    )

    def observed_atom_ids(component: int) -> set[str]:
        start = int(system.component_atom_ptr[component])
        stop = int(system.component_atom_ptr[component + 1])
        return {
            system.symbols[int(symbol)]
            for symbol in system.atom_symbol[start:stop].detach().cpu().tolist()
        }

    def induce_observed(value: ChemicalComponent, atom_ids: set[str]) -> ChemicalComponent:
        keep = tuple(atom.atom_id in atom_ids for atom in value.atoms)
        bonds = value.bonds
        if bonds is not None:
            bonds = frozenset(
                bond
                for bond in bonds
                if bond.atom_a in atom_ids
                and bond.atom_b in atom_ids
                and (bond.stereo_atoms is None or set(bond.stereo_atoms) <= atom_ids)
            )
        return ChemicalComponent(
            component_id=value.component_id,
            atoms=tuple(atom for atom, retained in zip(value.atoms, keep) if retained),
            bonds=bonds,
            reference_coordinates=(
                tuple(
                    coordinate
                    for coordinate, retained in zip(value.reference_coordinates, keep)
                    if retained
                )
                if value.reference_coordinates
                else ()
            ),
        )

    def resolve(component: int) -> ChemicalComponent | None:
        component_id = system.symbols[int(system.component_symbol[component])]
        if component_id == "UNK":
            if bool(system.component_topology_complete[component]):
                return None
            try:
                value = get_component(component_id).without_hydrogens()
            except KeyError:
                return None
            return ChemicalComponent(
                component_id=value.component_id,
                atoms=value.atoms,
                bonds=value.bonds,
                previous_attachment=value.previous_attachment,
                next_attachment=value.next_attachment,
                reference_coordinates=(),
            )
        if component_id not in cache:
            try:
                if revision is None and snapshot_path is None:
                    cache[component_id] = get_component(component_id).without_hydrogens()
                else:
                    cache[component_id] = resolve_ccd_component(
                        component_id,
                        revision=revision,
                        snapshot_path=snapshot_path,
                    ).without_hydrogens()
            except KeyError:
                cache[component_id] = None
        return cache[component_id]

    selected = component_index.detach().cpu().tolist()
    resolved_indices: list[int] = []
    resolved_components: list[ChemicalComponent] = []
    for component in selected:
        value = resolve(component)
        if value is None:
            continue
        instance = int(component_instance[component])
        entity = int(system.instance_entity[instance])
        if int(system.entity_type[entity]) == ENTITY_MOLECULE:
            atom_ids = observed_atom_ids(component)
            value = induce_observed(value, atom_ids)
            if atom_ids and not value.atoms:
                continue
        resolved_indices.append(component)
        resolved_components.append(value)
    if not resolved_indices:
        return system, torch.arange(atom_count, dtype=torch.int64, device=system.atom_symbol.device)
    hydrated, old_to_new, _ = replace_components(
        system,
        torch.tensor(resolved_indices, dtype=torch.int64, device=component_index.device),
        resolved_components,
    )
    return hydrated, old_to_new


def _hydrate_structure(
    structure: Structure,
    *,
    revision: str | None,
    snapshot_path: str | None,
) -> Structure:
    r"""Hydrate topology and map observed fields by component-local atom ID."""

    system, old_to_new = _hydrate_system(
        structure.system,
        revision=revision,
        snapshot_path=snapshot_path,
    )
    if system is structure.system:
        return structure

    retained = old_to_new >= 0
    source_atom = torch.nonzero(retained, as_tuple=False).flatten()
    output_atom = old_to_new[retained]
    atom_count = system.atom_symbol.numel()

    def scatter(value: Tensor) -> Tensor:
        output = value.new_zeros((atom_count, *value.shape[1:]))
        output[output_atom] = value[source_atom]
        return output

    return Structure(  # type: ignore[call-arg]
        system=system,
        coordinates=scatter(structure.coordinates),
        coordinate_mask=scatter(structure.coordinate_mask),
        occupancy=scatter(structure.occupancy),
        occupancy_mask=scatter(structure.occupancy_mask),
        b_factor=scatter(structure.b_factor),
        b_factor_mask=scatter(structure.b_factor_mask),
        source_symbols=structure.source_symbols,
        residue_id=structure.residue_id,
        insertion_code=structure.insertion_code,
        resolution=structure.resolution,
        resolution_is_known=structure.resolution_is_known,
        batch_size=[],
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


def _tokenize(system: MolecularSystem, expanded: frozenset[int]) -> list[dict[str, object]]:
    instance_components = system.instance_component_ptr.detach().cpu().tolist()
    component_atoms = system.component_atom_ptr.detach().cpu().tolist()
    instance_entities = system.instance_entity.detach().cpu().tolist()
    entity_types = system.entity_type.detach().cpu().tolist()
    component_symbols = system.component_symbol.detach().cpu().tolist()
    atom_symbols = system.atom_symbol.detach().cpu().tolist()
    atomic_numbers = system.atomic_number.detach().cpu().tolist()
    water_components = water_component_mask(system).detach().cpu().tolist()
    tokens: list[dict[str, object]] = []
    entity_copies: dict[int, int] = {}
    reference_space = 0
    chain_index = 0

    for instance_index, entity_index in enumerate(instance_entities):
        entity_type = entity_types[entity_index]
        component_start = instance_components[instance_index]
        component_stop = instance_components[instance_index + 1]
        if (
            entity_type == ENTITY_MOLECULE
            and component_stop > component_start
            and all(water_components[component_start:component_stop])
        ):
            continue

        symmetry_index = entity_copies.get(entity_index, 0)
        entity_copies[entity_index] = symmetry_index + 1
        if entity_type != ENTITY_MOLECULE:
            polymer_type = _POLYMER_TYPE[entity_type]
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
                    and not (component_id in {"N", "DN"} and system.symbols[atom_symbols[atom_index]] == "N")
                )
                if not atom_indices:
                    raise ValueError(
                        f"Protenix component {component_id!r} at "
                        f"{system.instance_id(instance_index)}:{monomer_index} "
                        "has no model-visible heavy atoms"
                    )
                standard = _is_standard(system, component_index, entity_type)
                groups = (
                    tuple((atom_index,) for atom_index in atom_indices)
                    if not standard or component_index in expanded
                    else (atom_indices,)
                )
                by_name = {system.symbols[atom_symbols[atom_index]]: atom_index for atom_index in atom_indices}
                center_name = "CA" if polymer_type == "protein" else "C1'"
                if polymer_type == "protein":
                    representative_name = "CA" if component_id == "GLY" else "CB"
                elif component_id in {"A", "G", "DA", "DG"}:
                    representative_name = "C4"
                elif component_id in {"C", "U", "DC", "DT"}:
                    representative_name = "C2"
                else:
                    representative_name = "C1'"
                for group in groups:
                    atom_token = len(group) == 1 and (not standard or len(groups) > 1)
                    center_atom = group[0] if atom_token else by_name[center_name]
                    representative_atom = (
                        group[0]
                        if atom_token
                        else by_name.get(
                            representative_name,
                            center_atom,
                        )
                    )
                    tokens.append(
                        {
                            "token_index": len(tokens),
                            "atoms": group,
                            "atom_index": group[0] if atom_token else None,
                            "component_index": component_index,
                            "instance_index": instance_index,
                            "entity_index": entity_index,
                            "monomer_index": monomer_index,
                            "residue_index": monomer_index,
                            "chain_index": chain_index,
                            "symmetry_index": symmetry_index,
                            "residue_type": _RESIDUE_TYPE.get(component_id, _UNKNOWN) if standard else _UNKNOWN,
                            "center": center_atom,
                            "representative": representative_atom,
                            "standard": standard,
                            "expanded": standard and len(groups) > 1,
                            "reference_space": reference_space,
                        }
                    )
                reference_space += 1
        else:
            molecule_has_atoms = False
            for component_index in range(component_start, component_stop):
                atom_indices = tuple(
                    atom_index
                    for atom_index in range(
                        component_atoms[component_index],
                        component_atoms[component_index + 1],
                    )
                    if atomic_numbers[atom_index] != 1
                )
                if not atom_indices:
                    continue
                molecule_has_atoms = True
                for atom_index in atom_indices:
                    tokens.append(
                        {
                            "token_index": len(tokens),
                            "atoms": (atom_index,),
                            "atom_index": atom_index,
                            "component_index": component_index,
                            "instance_index": instance_index,
                            "entity_index": entity_index,
                            "monomer_index": None,
                            "residue_index": component_index - component_start,
                            "chain_index": chain_index,
                            "symmetry_index": symmetry_index,
                            "residue_type": _UNKNOWN,
                            "center": atom_index,
                            "representative": atom_index,
                            "standard": False,
                            "expanded": False,
                            "reference_space": reference_space,
                        }
                    )
                reference_space += 1
            if not molecule_has_atoms:
                raise ValueError(
                    f"Protenix molecule {system.instance_id(instance_index)!r} " "has no model-visible heavy atoms"
                )
        chain_index += 1
    return tokens


def _features(
    system: MolecularSystem,
    *,
    tokens: list[dict[str, object]],
    msa: MSA | MSACollection | None,
    msa_bindings: tuple[tuple[MSA, int | None, int], ...],
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

    _msa_rows, _msa_mask, profile, deletion_mean = _msa_features(
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
            F.one_hot(residue_type, num_classes=32).float(),
            profile,
            deletion_mean[:, None],
        ),
        dim=-1,
    )
    deletion_counts = _assembled_deletions(
        msa,
        msa_bindings,
        tokens,
        system,
        max_rows=max_msa_rows,
    )
    has_deletion = deletion_counts.clamp(0, 1)
    deletion_value = torch.atan(deletion_counts / 3.0) * (2.0 / math.pi)
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
        if tfg_guidance is None
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
        msa_tokens=placed(_msa_rows),
        has_deletion=floating(has_deletion),
        deletion_value=floating(deletion_value),
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
        tfg_guidance=tfg_guidance,
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
            raise ValueError(f"TFG requires reference coordinates for modeled atoms in " f"component {component_id!r}")
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

    molecule = _rdkit_component(system, component_index)
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


def _rdkit_component(system: MolecularSystem, component_index: int) -> Mol:
    r"""
    Project one canonical component occurrence into RDKit for TFG geometry.
    """

    from rdkit import Chem
    from rdkit.Geometry import Point3D

    component_id = system.symbols[int(system.component_symbol[component_index])]
    if not bool(system.component_topology_complete[component_index]):
        raise ValueError(f"TFG requires known connectivity for component {component_id!r}")
    atom_start = int(system.component_atom_ptr[component_index])
    atom_stop = int(system.component_atom_ptr[component_index + 1])
    atom_count = atom_stop - atom_start

    editable = Chem.RWMol()
    atom_stereo: dict[int, str] = {}
    for atom_index in range(atom_start, atom_stop):
        value = Chem.Atom(int(system.atomic_number[atom_index]))
        if bool(system.isotope_known[atom_index]):
            value.SetIsotope(int(system.isotope[atom_index]))
        if bool(system.formal_charge_known[atom_index]):
            value.SetFormalCharge(int(system.formal_charge[atom_index]))
        if bool(system.radical_electrons_known[atom_index]):
            value.SetNumRadicalElectrons(int(system.radical_electrons[atom_index]))
        if bool(system.explicit_hydrogens_known[atom_index]):
            value.SetNumExplicitHs(int(system.explicit_hydrogens[atom_index]))
        if bool(system.implicit_hydrogens_allowed_known[atom_index]):
            value.SetNoImplicit(not bool(system.implicit_hydrogens_allowed[atom_index]))
        if bool(system.atom_aromatic_known[atom_index]):
            value.SetIsAromatic(bool(system.atom_aromatic[atom_index]))
        local_index = editable.AddAtom(value)
        stereo_index = int(system.atom_stereo[atom_index])
        if stereo_index >= 0:
            atom_stereo[local_index] = system.symbols[stereo_index].upper()

    bond_types = {
        1.0: Chem.BondType.SINGLE,
        1.5: Chem.BondType.AROMATIC,
        2.0: Chem.BondType.DOUBLE,
        3.0: Chem.BondType.TRIPLE,
        4.0: Chem.BondType.QUADRUPLE,
    }
    bond_stereo: list[tuple[int, str, tuple[int, int] | None]] = []
    edge_mask = (
        (system.edge_index[0] >= atom_start)
        & (system.edge_index[1] < atom_stop)
        & (system.bond_role == BOND_ROLE_COMPONENT)
    )
    edge_rows = edge_mask.nonzero(as_tuple=False).flatten().detach().cpu().tolist()
    for edge_index in edge_rows:
        if not bool(system.bond_order_known[edge_index]):
            raise ValueError(f"TFG requires known bond orders for component {component_id!r}")
        first = int(system.edge_index[0, edge_index]) - atom_start
        second = int(system.edge_index[1, edge_index]) - atom_start
        order = float(system.bond_order[edge_index])
        editable.AddBond(first, second, bond_types[order])
        rdkit_bond = editable.GetBondWithIdx(editable.GetNumBonds() - 1)
        if bool(system.bond_aromatic_known[edge_index]):
            rdkit_bond.SetIsAromatic(bool(system.bond_aromatic[edge_index]))
        stereo_index = int(system.bond_stereo[edge_index])
        if stereo_index >= 0:
            stereo_atoms_tensor = system.bond_stereo_index[:, edge_index]
            stereo_atoms = (
                None
                if bool((stereo_atoms_tensor < 0).any())
                else (
                    int(stereo_atoms_tensor[0]) - atom_start,
                    int(stereo_atoms_tensor[1]) - atom_start,
                )
            )
            bond_stereo.append(
                (
                    rdkit_bond.GetIdx(),
                    system.symbols[stereo_index],
                    stereo_atoms,
                )
            )

    molecule = editable.GetMol()
    Chem.SanitizeMol(molecule)
    conformer = Chem.Conformer(atom_count)
    reference_coordinates = system.reference_coordinates[atom_start:atom_stop].detach().cpu()
    reference_mask = system.reference_coordinate_mask[atom_start:atom_stop].detach().cpu()
    for index in range(atom_count):
        x, y, z = reference_coordinates[index].tolist() if bool(reference_mask[index]) else (0.0, 0.0, 0.0)
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

    group_indices: list[Tensor] = []
    for model_indices in grouped.values():
        indices = torch.tensor(model_indices)
        if not bool(mask[indices].any()):
            positions[indices] = 0
            mask[indices] = False
            charges[indices] = 0
            continue
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


def _msa_polymer_type(system: MolecularSystem, entity_index: int) -> str | None:
    return _POLYMER_TYPE.get(int(system.entity_type[entity_index]))


def _msa_symbol_residue_type(symbol: str, polymer_type: str | None) -> int:
    # Cells are UPPERCASE one-letter MSA symbols or ``-``; the query axis binds column i to
    # ``entity.components[i]`` so ``monomer_index`` is the column directly (query_to_entity is now the
    # identity map).
    if polymer_type == "rna":
        return _RNA_MSA_RESIDUE_TYPE.get(symbol, _RESIDUE_TYPE["N"])
    return _MSA_RESIDUE_TYPE.get(symbol, _UNKNOWN)


def _msa_descriptors(
    msa: MSA | MSACollection | None,
    resolved_bindings: tuple[tuple[MSA, int | None, int], ...],
    system: MolecularSystem,
    *,
    max_rows: int,
) -> list[tuple[dict[tuple[int, int | None], tuple[int, bool]], bool]]:
    # Each descriptor maps an entity-broadcast or exact-instance column binding to
    # ``(source_row, is_paired)``. The bool is the query flag.
    if msa is None:
        return [({}, True)]
    if isinstance(msa, MSA):
        # A bare MSA is the entity's own deep alignment: enumerate its unpaired pool (row 0 is the
        # query), which the single-file reader packs as the full alignment.
        _value, target_instance_index, entity_index = resolved_bindings[0]
        key = (entity_index, target_instance_index)
        return [({key: (row, False)}, row == 0) for row in range(min(len(msa.unpaired_sequences), max_rows))]

    row_pairing = msa.row_pairing
    row_is_paired = msa.row_is_paired
    plan_rows = int(row_pairing.shape[0])

    # RNA contributes only its unpaired pool. Protein paired rows keep the RNA query
    # sequence instead of treating RNA columns as another paired alignment.
    protein_columns = np.asarray(
        [_msa_polymer_type(system, entity_index) == "protein" for _value, _target, entity_index in resolved_bindings],
        dtype=np.bool_,
    )
    protein_paired = np.count_nonzero(
        (row_pairing != -1) & row_is_paired & protein_columns[None, :],
        axis=1,
    )
    paired = [row for row in range(1, plan_rows) if protein_paired[row] >= 2]
    paired_rows = set(paired)
    unpaired = [row for row in range(1, plan_rows) if row not in paired_rows]

    pair_limit = max_rows // 2
    ordered: list[int | None] = [None]
    ordered.extend(paired[: max(0, pair_limit - 1)])
    ordered.extend(unpaired[: max_rows - len(ordered)])
    ordered = ordered[:max_rows]

    descriptors: list[tuple[dict[tuple[int, int | None], tuple[int, bool]], bool]] = []
    for row in ordered:
        if row is None:
            descriptors.append(({}, True))
            continue
        mapping: dict[tuple[int, int | None], tuple[int, bool]] = {}
        for column, (_value, target_instance_index, entity_index) in enumerate(resolved_bindings):
            source = int(row_pairing[row, column])
            is_paired = bool(row_is_paired[row, column])
            if source == -1 or (is_paired and _msa_polymer_type(system, entity_index) == "rna"):
                continue
            mapping[(entity_index, target_instance_index)] = (source, is_paired)
        descriptors.append((mapping, False))
    return descriptors


def _msa_features(
    msa: MSA | MSACollection | None,
    resolved_bindings: tuple[tuple[MSA, int | None, int], ...],
    tokens: list[dict[str, object]],
    system: MolecularSystem,
    *,
    max_rows: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    bindings = {
        (entity_index, target_instance_index): value for value, target_instance_index, entity_index in resolved_bindings
    }
    descriptors = _msa_descriptors(
        msa,
        resolved_bindings,
        system,
        max_rows=max_rows,
    )
    rows = torch.empty(len(descriptors), len(tokens), dtype=torch.long)

    for row_index, (descriptor, query) in enumerate(descriptors):
        for token_index, token in enumerate(tokens):
            entity_index = cast(int, token["entity_index"])
            instance_index = cast(int, token["instance_index"])
            monomer_index = cast(int | None, token["monomer_index"])
            polymer_type = _msa_polymer_type(system, entity_index)
            key = (entity_index, instance_index) if (entity_index, instance_index) in bindings else (entity_index, None)
            selected = bindings.get(key)
            if query or selected is None or polymer_type not in {"protein", "rna"}:
                rows[row_index, token_index] = cast(int, token["residue_type"])
            elif key not in descriptor:
                rows[row_index, token_index] = cast(int, token["residue_type"]) if polymer_type == "rna" else _GAP
            else:
                source_row, is_paired = descriptor[key]
                pool = selected.paired_sequences if is_paired else selected.unpaired_sequences
                rows[row_index, token_index] = _msa_symbol_residue_type(
                    pool[source_row][cast(int, monomer_index)],
                    polymer_type,
                )

    profile = torch.zeros(len(tokens), 32)
    deletion_mean = torch.zeros(len(tokens))
    for token_index, token in enumerate(tokens):
        entity_index = cast(int, token["entity_index"])
        instance_index = cast(int, token["instance_index"])
        monomer_index = cast(int | None, token["monomer_index"])
        polymer_type = _msa_polymer_type(system, entity_index)
        selected = bindings.get(
            (entity_index, instance_index),
            bindings.get((entity_index, None)),
        )
        if selected is None or polymer_type not in {"protein", "rna"}:
            profile[token_index, cast(int, token["residue_type"])] = 1
            continue
        # The profile and deletion mean are the entity's own alignment statistics, taken over the full
        # unpaired pool, including the query row.
        column = cast(int, monomer_index)
        encoded = [_msa_symbol_residue_type(sequence[column], polymer_type) for sequence in selected.unpaired_sequences]
        profile[token_index] = F.one_hot(torch.tensor(encoded), num_classes=32).float().mean(dim=0)
        deletion_mean[token_index] = torch.tensor(
            selected.unpaired_deletion_counts[:, column],
            dtype=torch.float32,
        ).mean()
    return rows, torch.ones_like(rows, dtype=torch.float32), profile, deletion_mean


def _assembled_deletions(
    msa: MSA | MSACollection | None,
    resolved_bindings: tuple[tuple[MSA, int | None, int], ...],
    tokens: list[dict[str, object]],
    system: MolecularSystem,
    *,
    max_rows: int,
) -> Tensor:
    bindings = {
        (entity_index, target_instance_index): value for value, target_instance_index, entity_index in resolved_bindings
    }
    descriptors = _msa_descriptors(
        msa,
        resolved_bindings,
        system,
        max_rows=max_rows,
    )
    deletions = torch.zeros(len(descriptors), len(tokens))
    for row_index, (descriptor, query) in enumerate(descriptors):
        if query:
            continue
        for token_index, token in enumerate(tokens):
            entity_index = cast(int, token["entity_index"])
            instance_index = cast(int, token["instance_index"])
            monomer_index = cast(int | None, token["monomer_index"])
            key = (entity_index, instance_index) if (entity_index, instance_index) in bindings else (entity_index, None)
            selected = bindings.get(key)
            if (
                selected is None
                or key not in descriptor
                or _msa_polymer_type(system, entity_index) not in {"protein", "rna"}
            ):
                continue
            source_row, is_paired = descriptor[key]
            pool = selected.paired_deletion_counts if is_paired else selected.unpaired_deletion_counts
            deletions[row_index, token_index] = float(pool[source_row, cast(int, monomer_index)])
    return deletions


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
    residue_type_ids[0] = _GAP

    target_component_ptr = system.instance_component_ptr.detach().cpu().tolist()
    for instance_index, selected_templates in chain_templates.items():
        target_tokens = chain_tokens[instance_index]
        for template_index, (template, template_alignments) in enumerate(selected_templates):
            residue_type_ids[template_index, target_tokens] = _GAP
            template_system = template.structure.system
            template_component_ptr = template_system.instance_component_ptr.detach().cpu().tolist()
            template_component_symbols = template_system.component_symbol.detach().cpu().tolist()
            for template_instance_index, residue_pairs in template_alignments:
                template_entity_index = int(template_system.instance_entity[template_instance_index])
                for target_component, template_component in residue_pairs:
                    target_component_index = target_component_ptr[instance_index] + target_component
                    targets = residue_tokens.get(target_component_index)
                    if targets is None:
                        raise ValueError(
                            f"template alignment target "
                            f"{system.instance_id(instance_index)!r}:{target_component} "
                            "does not resolve to a token"
                        )
                    template_component_index = template_component_ptr[template_instance_index] + template_component
                    backbone, backbone_mask = _template_residue_backbone(
                        template.structure,
                        template_component_index,
                    )
                    component_id = template_system.symbols[template_component_symbols[template_component_index]]
                    residue_type = (
                        _RESIDUE_TYPE.get(component_id, _UNKNOWN)
                        if _is_standard(
                            template_system,
                            template_component_index,
                            int(template_system.entity_type[template_entity_index]),
                        )
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
