r"""
Tensor-native input processing for ODesign structure generation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import cast, get_args

import torch
import torch.nn.functional as F
from torch import Tensor

from ...data.bonds import SymbolicBond
from ...data.components import get_component
from ...data.conditions import Hotspot
from ...data.evidence import MSA, MSACollection, resolve_msa_bindings
from ...data.molecular import (
    BOND_DIRECTION_ENDPOINT_0_DONOR,
    BOND_DIRECTION_ENDPOINT_1_DONOR,
    BOND_DIRECTION_UNDIRECTED,
    BOND_KIND_COVALENT,
    BOND_KIND_DATIVE,
    BOND_ROLE_ASSEMBLY,
    ENTITY_DNA,
    ENTITY_DNA_RNA_HYBRID,
    ENTITY_MOLECULE,
    ENTITY_PNA,
    ENTITY_PROTEIN,
    ENTITY_RNA,
    AtomSite,
    BondDirection,
    BondKind,
    ComponentReference,
    MolecularSystem,
    PolymerSequence,
    PolymerType,
    copy_components,
    disjoint_union,
    form_bond,
    instance_subgraph,
)
from ...data.placement import (
    ExistingSpan,
    GeneratedSegment,
    PlacementMap,
    PlacementSpec,
    SymbolicAtomSite,
    SymbolicResidueSite,
)
from ...data.placement import instance_of as placement_instance_of
from ...data.placement import resolve as resolve_placement
from ...data.structure import Structure
from ...processor import Processor
from .config import ODesignLigandProcessorConfig, ODesignProcessorConfig
from .model import (
    ODesignGeneratedAtoms,
    ODesignInput,
    ODesignLigandInput,
    ODesignPlacements,
    ODesignStructureTarget,
)

_CORE_POLYMER_TYPES: frozenset[str] = frozenset(get_args(PolymerType))

#: One designed polymer type or fixed source residue in an ODesign mask.
ODesignResidueSpec = PolymerType | ComponentReference

_RESIDUE_TYPE_INDEX: dict[str, int] = {
    "ALA": 0,
    "ARG": 1,
    "ASN": 2,
    "ASP": 3,
    "CYS": 4,
    "GLN": 5,
    "GLU": 6,
    "GLY": 7,
    "HIS": 8,
    "ILE": 9,
    "LEU": 10,
    "LYS": 11,
    "MET": 12,
    "PHE": 13,
    "PRO": 14,
    "SER": 15,
    "THR": 16,
    "TRP": 17,
    "TYR": 18,
    "VAL": 19,
    "UNK": 20,
    "A": 21,
    "G": 22,
    "C": 23,
    "U": 24,
    "N": 25,
    "DA": 26,
    "DG": 27,
    "DC": 28,
    "DT": 29,
    "DN": 30,
    "-": 31,
    "-P": 32,
    "-N": 33,
    "-L": 34,
}

_STANDARD_POLYMER_COMPONENTS: frozenset[str] = frozenset(_RESIDUE_TYPE_INDEX) - {"-", "-P", "-N", "-L"}
_GENERATED_COMPONENT_ID = {"protein": "UNK", "rna": "N", "dna": "DN"}
_PLACEHOLDER_COMPONENT_ID = {
    **_GENERATED_COMPONENT_ID,
    "dna_rna_hybrid": "N",
    "pna": "UNK",
}
_POLYMER_TYPE_BY_ENTITY: dict[int, PolymerType] = {
    ENTITY_PROTEIN: "protein",
    ENTITY_RNA: "rna",
    ENTITY_DNA: "dna",
    ENTITY_DNA_RNA_HYBRID: "dna_rna_hybrid",
    ENTITY_PNA: "pna",
}
_MSA_UNKNOWN = _RESIDUE_TYPE_INDEX["UNK"]
_MSA_GAP = _RESIDUE_TYPE_INDEX["-"]
_MSA_SYMBOL_TOKENS = {
    "protein": {
        symbol: _RESIDUE_TYPE_INDEX[component_id]
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
    },
    "rna": {symbol: _RESIDUE_TYPE_INDEX[symbol] for symbol in ("A", "G", "C", "U", "N")},
    "dna": {symbol: _RESIDUE_TYPE_INDEX["D" + symbol] for symbol in ("A", "G", "C", "T", "N")},
}


class ODesignProcessor(Processor):
    r"""
    Lower one realized polymer-design problem to the ODesign equation tensors.
    """

    config_class = ODesignProcessorConfig
    config: ODesignProcessorConfig

    def __init__(
        self,
        config: ODesignProcessorConfig | None = None,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        config = ODesignProcessorConfig() if config is None else config
        self.config = config
        super().__init__(device=device, dtype=dtype)

    def __call__(
        self,
        input: MolecularSystem | Structure | None,
        *,
        placement: ODesignPlacements | None = None,
        msa: MSA | MSACollection | None = None,
        hotspots: Sequence[Hotspot] = (),
        cyclic_bonds: Sequence[SymbolicBond] = (),
        fixed_atoms: Sequence[AtomSite] = (),
        constraint_atoms: Sequence[AtomSite] = (),
        generator: torch.Generator | None = None,
    ) -> ODesignInput:
        r"""
        Build one fixed-topology ODesign polymer input.

        Variable placement lengths are sampled here with ``generator``. Symbolic
        placement and condition metadata is resolved before the typed Input is
        returned.
        """

        model_input, _ = self._lower(
            input,
            placement=placement,
            msa=msa,
            hotspots=hotspots,
            cyclic_bonds=cyclic_bonds,
            fixed_atoms=fixed_atoms,
            constraint_atoms=constraint_atoms,
            generator=generator,
        )
        return model_input

    def process_training(
        self,
        input: MolecularSystem | Structure | None,
        target: Structure | None = None,
        *,
        placement: ODesignPlacements | None = None,
        msa: MSA | MSACollection | None = None,
        hotspots: Sequence[Hotspot] = (),
        cyclic_bonds: Sequence[SymbolicBond] = (),
        fixed_atoms: Sequence[AtomSite] = (),
        constraint_atoms: Sequence[AtomSite] = (),
        generator: torch.Generator | None = None,
    ) -> tuple[ODesignInput, ODesignStructureTarget]:
        r"""
        Lower one polymer problem and gather truth onto its Model atom axis.
        """

        if target is None:
            if not isinstance(input, Structure):
                raise TypeError("ODesign training requires a target Structure")
            target = input
        model_input, output_to_source = self._lower(
            input,
            placement=placement,
            msa=msa,
            hotspots=hotspots,
            cyclic_bonds=cyclic_bonds,
            fixed_atoms=fixed_atoms,
            constraint_atoms=constraint_atoms,
            generator=generator,
        )
        target_rows: list[int] = []
        target_system = target.system
        for output_atom in model_input.atom_index[0].detach().cpu().tolist():
            output_site = model_input.system.atom_site(output_atom)
            try:
                target_row = target_system.atom_index(output_site)
            except KeyError:
                source_atom = output_to_source.get(output_atom)
                if source_atom is None:
                    target_row = -1
                else:
                    try:
                        target_row = target_system.atom_index(
                            cast(MolecularSystem, _source_system(input)).atom_site(source_atom)
                        )
                    except KeyError:
                        target_row = -1
            target_rows.append(target_row)
        coordinates, coordinate_mask = _gather_target(target, target_rows)
        return model_input, ODesignStructureTarget(  # type: ignore[call-arg]
            coordinates=coordinates.to(device=self.device, dtype=self.dtype),
            coordinate_mask=coordinate_mask.to(self.device),
            batch_size=[1],
        )

    def _lower(
        self,
        input: MolecularSystem | Structure | None,
        *,
        placement: ODesignPlacements | None,
        msa: MSA | MSACollection | None,
        hotspots: Sequence[Hotspot],
        cyclic_bonds: Sequence[SymbolicBond],
        fixed_atoms: Sequence[AtomSite],
        constraint_atoms: Sequence[AtomSite],
        generator: torch.Generator | None,
    ) -> tuple[ODesignInput, dict[int, int]]:
        structure = input if isinstance(input, Structure) else None
        source = _source_system(input)
        graph_device = (
            source.atom_symbol.device
            if source is not None
            else torch.device("cpu") if self.device is None else torch.device(self.device)
        )
        if placement is None:
            if source is None:
                raise ValueError("input=None requires placement that determines generated topology")
            system = source
            placement_map: PlacementMap = {}
            source_component_outputs: dict[int, tuple[int, ...]] = {
                index: (index,) for index in range(source.component_symbol.numel())
            }
            source_atom_outputs: dict[int, tuple[int, ...]] = {
                index: (index,) for index in range(source.atom_symbol.numel())
            }
            design_component_mask = torch.zeros_like(
                system.component_symbol,
                dtype=torch.bool,
            )
            cyclic_pairs: tuple[tuple[int, int], ...] = ()
            cyclic_instances: frozenset[int] = frozenset()
        else:
            exact_placement = _realize_placement(
                placement,
                generator=generator,
                device=graph_device,
            )
            (
                system,
                placement_map,
                source_component_outputs,
                source_atom_outputs,
                design_component_mask,
                cyclic_pairs,
                cyclic_instances,
            ) = _materialize_polymer_system(
                exact_placement,
                source,
                cyclic_bonds,
                graph_device,
            )

        generated_types = {
            _POLYMER_TYPE_BY_ENTITY[int(system.entity_type[system.instance_entity[instance]])]
            for instance in range(system.instance_symbol.numel())
            for component in range(
                int(system.instance_component_ptr[instance]),
                int(system.instance_component_ptr[instance + 1]),
            )
            if bool(design_component_mask[component])
        }
        allowed: set[PolymerType] = {"protein"} if self.config.generated_modality == "protein" else {"dna", "rna"}
        if not generated_types <= allowed:
            raise ValueError(
                f"ODesign {self.config.generated_modality} artifact cannot generate "
                f"{sorted(generated_types - allowed)!r}"
            )

        model_atom_index = _model_atom_index(system, design_component_mask)
        if model_atom_index.numel() == 0:
            raise ValueError("ODesign requires at least one heavy atom")
        (
            token_component_index,
            token_atom_index,
            token_instance_index,
            atom_to_token,
        ) = _token_axes(system, model_atom_index)
        token_count = token_component_index.numel()
        canonical_atom_to_token = torch.full(
            (system.atom_symbol.numel(),),
            -1,
            dtype=torch.int64,
            device=system.atom_symbol.device,
        )
        canonical_atom_to_token[model_atom_index] = atom_to_token

        output_to_source = {
            output: source_atom for source_atom, outputs in source_atom_outputs.items() for output in outputs
        }
        observed_positions, observed_mask = _observed_coordinates(
            structure,
            model_atom_index,
            output_to_source,
        )
        fixed_coordinate_mask = _condition_atom_mask(
            fixed_atoms,
            source,
            source_atom_outputs,
            model_atom_index,
        )
        constraint_coordinate_mask = fixed_coordinate_mask | _condition_atom_mask(
            constraint_atoms,
            source,
            source_atom_outputs,
            model_atom_index,
        )
        if bool((constraint_coordinate_mask & ~observed_mask).any()):
            raise ValueError("fixed_atoms and constraint_atoms require observed coordinates")
        if self.config.receptor_mode == "rigid":
            fixed_coordinate_mask |= (
                ~design_component_mask[_atom_component_index(system)[model_atom_index]] & observed_mask
            )
        elif self.config.receptor_mode != "flex":
            raise ValueError(f"unknown ODesign receptor_mode: {self.config.receptor_mode!r}")

        hotspot = _hotspot_tokens(
            hotspots,
            source,
            source_component_outputs,
            source_atom_outputs,
            token_component_index,
            canonical_atom_to_token,
        )
        reference_positions, reference_space_id = _reference_geometry(
            system,
            model_atom_index,
            design_component_mask,
        )
        reference_mask = torch.ones(
            model_atom_index.numel(),
            dtype=torch.bool,
            device=system.atom_symbol.device,
        )
        atom_features = _atom_features(
            system,
            model_atom_index,
            reference_mask,
        )
        msa_features, msa_mask, profile, deletion_mean = _msa_features(
            msa,
            system,
            token_component_index,
            token_instance_index,
            design_component_mask,
        )
        token_features = _token_features(
            system,
            token_component_index,
            token_atom_index,
            design_component_mask,
            profile,
            deletion_mean,
            hotspot,
        )
        token_bonds = _token_bonds(
            system,
            canonical_atom_to_token,
            token_count,
            cyclic_pairs,
        )
        constraint_distogram = _constraint_distogram(
            system,
            model_atom_index,
            token_component_index,
            canonical_atom_to_token,
            observed_positions,
            constraint_coordinate_mask,
        )
        chain_index, entity_index, symmetry_index, residue_index = _token_indices(
            system,
            token_component_index,
            token_instance_index,
        )
        cyclic_period = _cyclic_period(
            token_instance_index,
            cyclic_instances,
        )

        model_input = ODesignInput(  # type: ignore[call-arg]
            token_mask=torch.ones(
                1,
                token_count,
                dtype=torch.bool,
                device=self.device,
            ),
            residue_index=residue_index.unsqueeze(0).to(self.device),
            chain_index=chain_index.unsqueeze(0).to(self.device),
            entity_index=entity_index.unsqueeze(0).to(self.device),
            symmetry_index=symmetry_index.unsqueeze(0).to(self.device),
            token_index=torch.arange(
                token_count,
                dtype=torch.int64,
                device=self.device,
            ).unsqueeze(0),
            cyclic_period=cyclic_period.unsqueeze(0).to(self.device),
            token_features=token_features.unsqueeze(0).to(
                device=self.device,
                dtype=self.dtype,
            ),
            token_bonds=token_bonds.unsqueeze(0).to(
                device=self.device,
                dtype=self.dtype,
            ),
            atom_features=atom_features.unsqueeze(0).to(
                device=self.device,
                dtype=self.dtype,
            ),
            atom_to_token=atom_to_token.unsqueeze(0).to(self.device),
            atom_mask=torch.ones(
                1,
                model_atom_index.numel(),
                dtype=torch.bool,
                device=self.device,
            ),
            reference_positions=reference_positions.unsqueeze(0).to(
                device=self.device,
                dtype=self.dtype,
            ),
            reference_space_id=reference_space_id.unsqueeze(0).to(self.device),
            reference_mask=reference_mask.unsqueeze(0).to(self.device),
            reference_charges=system.formal_charge[model_atom_index]
            .to(dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(-1)
            .to(device=self.device, dtype=self.dtype),
            observed_positions=observed_positions.unsqueeze(0).to(
                device=self.device,
                dtype=self.dtype,
            ),
            observed_mask=observed_mask.unsqueeze(0).to(self.device),
            fixed_coordinate_mask=fixed_coordinate_mask.unsqueeze(0).to(self.device),
            constraint_distogram=constraint_distogram.unsqueeze(0).to(
                device=self.device,
                dtype=self.dtype,
            ),
            msa_features=msa_features.unsqueeze(0).to(
                device=self.device,
                dtype=self.dtype,
            ),
            msa_mask=msa_mask.unsqueeze(0).to(self.device),
            system=system,
            atom_index=model_atom_index.unsqueeze(0).to(self.device),
            token_component_index=token_component_index.unsqueeze(0).to(self.device),
            token_atom_index=token_atom_index.unsqueeze(0).to(self.device),
            design_component_mask=design_component_mask,
            batch_size=[],
        )
        return model_input, output_to_source


class ODesignLigandProcessor(Processor):
    r"""
    Lower canonical context and ordered provisional ligand rows for ODesign.
    """

    config_class = ODesignLigandProcessorConfig
    config: ODesignLigandProcessorConfig

    def __init__(
        self,
        config: ODesignLigandProcessorConfig | None = None,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        config = ODesignLigandProcessorConfig() if config is None else config
        self.config = config
        super().__init__(device=device, dtype=dtype)

    def __call__(
        self,
        input: MolecularSystem | Structure | None,
        *,
        generated_atoms: ODesignGeneratedAtoms,
        msa: MSA | MSACollection | None = None,
        hotspots: Sequence[Hotspot] = (),
        fixed_atoms: Sequence[AtomSite] = (),
        constraint_atoms: Sequence[AtomSite] = (),
        generator: torch.Generator | None = None,
    ) -> ODesignLigandInput:
        r"""
        Build one ligand input, sampling its provisional atom count exactly once.
        """

        return self._lower(
            input,
            generated_atoms=generated_atoms,
            msa=msa,
            hotspots=hotspots,
            fixed_atoms=fixed_atoms,
            constraint_atoms=constraint_atoms,
            generator=generator,
        )

    def process_training(
        self,
        input: MolecularSystem | Structure | None,
        target: Structure,
        *,
        generated_atoms: ODesignGeneratedAtoms,
        generated_target_atom_indices: Sequence[int],
        msa: MSA | MSACollection | None = None,
        hotspots: Sequence[Hotspot] = (),
        fixed_atoms: Sequence[AtomSite] = (),
        constraint_atoms: Sequence[AtomSite] = (),
        generator: torch.Generator | None = None,
    ) -> tuple[ODesignLigandInput, ODesignStructureTarget]:
        r"""
        Lower ligand training truth using explicit ordered generated target rows.
        """

        model_input = self._lower(
            input,
            generated_atoms=generated_atoms,
            msa=msa,
            hotspots=hotspots,
            fixed_atoms=fixed_atoms,
            constraint_atoms=constraint_atoms,
            generator=generator,
        )
        context_rows = []
        for atom in model_input.atom_index[0].detach().cpu().tolist():
            if atom < 0:
                break
            context_rows.append(target.system.atom_index(model_input.system.atom_site(atom)))
        generated_rows = [int(index) for index in generated_target_atom_indices]
        slot_count = int((model_input.atom_mask[0] & (model_input.atom_index[0] < 0)).sum())
        if len(generated_rows) != slot_count:
            raise ValueError("generated_target_atom_indices must have one row per provisional atom")
        coordinates, coordinate_mask = _gather_target(
            target,
            [*context_rows, *generated_rows],
        )
        return model_input, ODesignStructureTarget(  # type: ignore[call-arg]
            coordinates=coordinates.to(device=self.device, dtype=self.dtype),
            coordinate_mask=coordinate_mask.to(self.device),
            batch_size=[1],
        )

    def _lower(
        self,
        input: MolecularSystem | Structure | None,
        *,
        generated_atoms: ODesignGeneratedAtoms,
        msa: MSA | MSACollection | None,
        hotspots: Sequence[Hotspot],
        fixed_atoms: Sequence[AtomSite],
        constraint_atoms: Sequence[AtomSite],
        generator: torch.Generator | None,
    ) -> ODesignLigandInput:
        structure = input if isinstance(input, Structure) else None
        system = _source_system(input)
        if system is None:
            system = MolecularSystem.from_entities({}, {})
        num_slots = _realize_atom_count(
            generated_atoms,
            generator=generator,
            device=system.atom_symbol.device,
        )
        context_design_mask = torch.zeros_like(
            system.component_symbol,
            dtype=torch.bool,
        )
        context_atom_index = _model_atom_index(
            system,
            context_design_mask,
        )
        (
            context_token_component,
            context_token_atom,
            context_token_instance,
            context_atom_to_token,
        ) = _token_axes(system, context_atom_index)
        context_atom_count = context_atom_index.numel()
        context_token_count = context_token_component.numel()
        atom_count = context_atom_count + num_slots
        token_count = context_token_count + num_slots
        if atom_count == 0:
            raise ValueError("ODesign requires at least one context atom or generated atom")

        token_component_index = torch.cat(
            [
                context_token_component,
                torch.full(
                    (num_slots,),
                    -1,
                    dtype=torch.int64,
                    device=system.atom_symbol.device,
                ),
            ]
        )
        token_atom_index = torch.cat(
            [
                context_token_atom,
                torch.full(
                    (num_slots,),
                    -1,
                    dtype=torch.int64,
                    device=system.atom_symbol.device,
                ),
            ]
        )
        atom_index = torch.cat(
            [
                context_atom_index,
                torch.full(
                    (num_slots,),
                    -1,
                    dtype=torch.int64,
                    device=system.atom_symbol.device,
                ),
            ]
        )
        atom_to_token = torch.cat(
            [
                context_atom_to_token,
                torch.arange(
                    context_token_count,
                    token_count,
                    dtype=torch.int64,
                    device=system.atom_symbol.device,
                ),
            ]
        )
        canonical_atom_to_token = torch.full(
            (system.atom_symbol.numel(),),
            -1,
            dtype=torch.int64,
            device=system.atom_symbol.device,
        )
        canonical_atom_to_token[context_atom_index] = context_atom_to_token

        context_reference, context_space = _reference_geometry(
            system,
            context_atom_index,
            context_design_mask,
        )
        generated_space = int(system.component_symbol.numel()) if num_slots else 0
        reference_positions = torch.cat(
            [
                context_reference,
                torch.zeros(
                    num_slots,
                    3,
                    dtype=context_reference.dtype,
                    device=context_reference.device,
                ),
            ]
        )
        reference_space_id = torch.cat(
            [
                context_space,
                torch.full(
                    (num_slots,),
                    generated_space,
                    dtype=torch.int64,
                    device=system.atom_symbol.device,
                ),
            ]
        )
        reference_mask = torch.ones(
            atom_count,
            dtype=torch.bool,
            device=system.atom_symbol.device,
        )
        atom_features = torch.cat(
            [
                _atom_features(
                    system,
                    context_atom_index,
                    torch.ones_like(context_atom_index, dtype=torch.bool),
                ),
                _ligand_slot_atom_features(
                    num_slots,
                    device=system.atom_symbol.device,
                ),
            ]
        )
        observed_positions, observed_mask = _observed_coordinates(
            structure,
            context_atom_index,
            {index: index for index in range(system.atom_symbol.numel())},
        )
        observed_positions = torch.cat(
            [
                observed_positions,
                torch.zeros(
                    num_slots,
                    3,
                    dtype=observed_positions.dtype,
                    device=observed_positions.device,
                ),
            ]
        )
        observed_mask = torch.cat(
            [
                observed_mask,
                torch.zeros(
                    num_slots,
                    dtype=torch.bool,
                    device=observed_mask.device,
                ),
            ]
        )
        identity_atom_outputs: dict[int, tuple[int, ...]] = {
            index: (index,) for index in range(system.atom_symbol.numel())
        }
        fixed_coordinate_mask = torch.cat(
            [
                _condition_atom_mask(
                    fixed_atoms,
                    system,
                    identity_atom_outputs,
                    context_atom_index,
                ),
                torch.zeros(
                    num_slots,
                    dtype=torch.bool,
                    device=system.atom_symbol.device,
                ),
            ]
        )
        constraint_coordinate_mask = fixed_coordinate_mask.clone()
        constraint_coordinate_mask[:context_atom_count] |= _condition_atom_mask(
            constraint_atoms,
            system,
            identity_atom_outputs,
            context_atom_index,
        )
        if bool((constraint_coordinate_mask & ~observed_mask).any()):
            raise ValueError("fixed_atoms and constraint_atoms require observed coordinates")
        if self.config.receptor_mode == "rigid":
            fixed_coordinate_mask[:context_atom_count] |= observed_mask[:context_atom_count]
        elif self.config.receptor_mode != "flex":
            raise ValueError(f"unknown ODesign ligand receptor_mode: " f"{self.config.receptor_mode!r}")

        hotspot = _hotspot_tokens(
            hotspots,
            system,
            {index: (index,) for index in range(system.component_symbol.numel())},
            identity_atom_outputs,
            context_token_component,
            canonical_atom_to_token,
        )
        hotspot = torch.cat(
            [
                hotspot,
                torch.zeros(
                    num_slots,
                    dtype=torch.bool,
                    device=hotspot.device,
                ),
            ]
        )
        (
            context_msa_features,
            context_msa_mask,
            context_profile,
            context_deletion_mean,
        ) = _msa_features(
            msa,
            system,
            context_token_component,
            context_token_instance,
            context_design_mask,
        )
        msa_depth = context_msa_features.shape[0]
        slot_msa_features = torch.zeros(
            msa_depth,
            num_slots,
            34,
            device=context_msa_features.device,
        )
        slot_msa_features[..., 31] = 1
        msa_features = torch.cat(
            [context_msa_features, slot_msa_features],
            dim=1,
        )
        msa_mask = torch.cat(
            [
                context_msa_mask,
                torch.ones(
                    msa_depth,
                    num_slots,
                    dtype=torch.bool,
                    device=context_msa_mask.device,
                ),
            ],
            dim=1,
        )
        slot_profile = torch.zeros(
            num_slots,
            32,
            device=context_profile.device,
        )
        slot_profile[:, 31] = 1
        profile = torch.cat([context_profile, slot_profile])
        deletion_mean = torch.cat(
            [
                context_deletion_mean,
                torch.zeros(
                    num_slots,
                    device=context_deletion_mean.device,
                ),
            ]
        )
        token_features = _token_features(
            system,
            token_component_index,
            token_atom_index,
            context_design_mask,
            profile,
            deletion_mean,
            hotspot,
        )
        token_bonds = torch.zeros(
            token_count,
            token_count,
            dtype=torch.float32,
            device=system.atom_symbol.device,
        )
        token_bonds[:context_token_count, :context_token_count] = _token_bonds(
            system,
            canonical_atom_to_token,
            context_token_count,
            (),
        )
        constraint_distogram = torch.zeros(
            token_count,
            token_count,
            1,
            dtype=torch.float32,
            device=system.atom_symbol.device,
        )
        constraint_distogram[
            :context_token_count,
            :context_token_count,
        ] = _constraint_distogram(
            system,
            context_atom_index,
            context_token_component,
            canonical_atom_to_token,
            observed_positions[:context_atom_count],
            constraint_coordinate_mask[:context_atom_count],
        )
        (
            context_chain,
            context_entity,
            context_symmetry,
            context_residue,
        ) = _token_indices(
            system,
            context_token_component,
            context_token_instance,
        )
        chain_index = torch.cat(
            [
                context_chain,
                torch.full(
                    (num_slots,),
                    system.instance_symbol.numel(),
                    dtype=torch.int64,
                    device=system.atom_symbol.device,
                ),
            ]
        )
        entity_index = torch.cat(
            [
                context_entity,
                torch.full(
                    (num_slots,),
                    system.entity_symbol.numel(),
                    dtype=torch.int64,
                    device=system.atom_symbol.device,
                ),
            ]
        )
        symmetry_index = torch.cat(
            [
                context_symmetry,
                torch.zeros(
                    num_slots,
                    dtype=torch.int64,
                    device=system.atom_symbol.device,
                ),
            ]
        )
        residue_index = torch.cat(
            [
                context_residue,
                torch.zeros(
                    num_slots,
                    dtype=torch.int64,
                    device=system.atom_symbol.device,
                ),
            ]
        )
        context_charges = system.formal_charge[context_atom_index].to(dtype=torch.float32)
        reference_charges = torch.cat(
            [
                context_charges,
                torch.zeros(
                    num_slots,
                    dtype=torch.float32,
                    device=context_charges.device,
                ),
            ]
        )
        return ODesignLigandInput(  # type: ignore[call-arg]
            token_mask=torch.ones(
                1,
                token_count,
                dtype=torch.bool,
                device=self.device,
            ),
            residue_index=residue_index.unsqueeze(0).to(self.device),
            chain_index=chain_index.unsqueeze(0).to(self.device),
            entity_index=entity_index.unsqueeze(0).to(self.device),
            symmetry_index=symmetry_index.unsqueeze(0).to(self.device),
            token_index=torch.arange(
                token_count,
                dtype=torch.int64,
                device=self.device,
            ).unsqueeze(0),
            cyclic_period=torch.zeros(
                1,
                token_count,
                dtype=torch.int64,
                device=self.device,
            ),
            token_features=token_features.unsqueeze(0).to(
                device=self.device,
                dtype=self.dtype,
            ),
            token_bonds=token_bonds.unsqueeze(0).to(
                device=self.device,
                dtype=self.dtype,
            ),
            atom_features=atom_features.unsqueeze(0).to(
                device=self.device,
                dtype=self.dtype,
            ),
            atom_to_token=atom_to_token.unsqueeze(0).to(self.device),
            atom_mask=torch.ones(
                1,
                atom_count,
                dtype=torch.bool,
                device=self.device,
            ),
            reference_positions=reference_positions.unsqueeze(0).to(
                device=self.device,
                dtype=self.dtype,
            ),
            reference_space_id=reference_space_id.unsqueeze(0).to(self.device),
            reference_mask=reference_mask.unsqueeze(0).to(self.device),
            reference_charges=reference_charges.unsqueeze(0).unsqueeze(-1).to(device=self.device, dtype=self.dtype),
            observed_positions=observed_positions.unsqueeze(0).to(
                device=self.device,
                dtype=self.dtype,
            ),
            observed_mask=observed_mask.unsqueeze(0).to(self.device),
            fixed_coordinate_mask=fixed_coordinate_mask.unsqueeze(0).to(self.device),
            constraint_distogram=constraint_distogram.unsqueeze(0).to(
                device=self.device,
                dtype=self.dtype,
            ),
            msa_features=msa_features.unsqueeze(0).to(
                device=self.device,
                dtype=self.dtype,
            ),
            msa_mask=msa_mask.unsqueeze(0).to(self.device),
            system=system,
            atom_index=atom_index.unsqueeze(0).to(self.device),
            token_component_index=token_component_index.unsqueeze(0).to(self.device),
            token_atom_index=token_atom_index.unsqueeze(0).to(self.device),
            batch_size=[],
        )


def from_design_mask(
    residues: Sequence[ODesignResidueSpec],
    *,
    groups: Sequence[str],
    placement_ids: Sequence[str],
) -> PlacementSpec:
    r"""
    Convert ODesign residue specifications to contiguous placement runs.
    """

    placement: PlacementSpec = {}
    seen: set[str] = set()
    current_group: str | None = None
    current_placement_id: str | None = None
    current_polymer_type: PolymerType | None = None
    current_instance_id: str | None = None
    run_start_source = 0
    run_next_source = 0
    run_length = 0

    def flush() -> None:
        if current_placement_id is None:
            return
        assert current_group is not None
        if current_polymer_type is not None:
            value: GeneratedSegment | ExistingSpan = GeneratedSegment(
                current_polymer_type,
                run_length,
                run_length,
            )
        else:
            assert current_instance_id is not None
            value = ExistingSpan(
                current_instance_id,
                run_start_source,
                run_next_source,
            )
        placement.setdefault(current_group, {})[current_placement_id] = value

    for index, (residue, group, placement_id) in enumerate(zip(residues, groups, placement_ids, strict=True)):
        if isinstance(residue, str):
            if residue not in _CORE_POLYMER_TYPES:
                raise ValueError(f"designed residue {index} has non-core polymer_type " f"{residue!r}")
            if group == current_group and placement_id == current_placement_id and current_polymer_type == residue:
                run_length += 1
                continue
            flush()
            if placement_id in seen:
                raise ValueError(f"placement_id {placement_id!r} is not contiguous")
            seen.add(placement_id)
            current_group = group
            current_placement_id = placement_id
            current_polymer_type = residue
            current_instance_id = None
            run_length = 1
        else:
            instance_id, source_index = residue
            if (
                group == current_group
                and placement_id == current_placement_id
                and current_polymer_type is None
                and current_instance_id == instance_id
                and run_next_source == source_index
            ):
                run_next_source = source_index + 1
                continue
            flush()
            if placement_id in seen:
                raise ValueError(f"placement_id {placement_id!r} is not contiguous")
            seen.add(placement_id)
            current_group = group
            current_placement_id = placement_id
            current_polymer_type = None
            current_instance_id = instance_id
            run_start_source = source_index
            run_next_source = source_index + 1
    flush()
    return placement


def to_design_mask(
    placement: PlacementSpec,
    placement_map: PlacementMap,
) -> tuple[list[bool], list[str]]:
    r"""
    Return generated flags and declaration groups in realized component order.
    """

    design_mask: list[bool] = []
    groups: list[str] = []
    for group, chain in placement.items():
        for placement_id, value in chain.items():
            length = (
                len(placement_map[group][placement_id])
                if isinstance(value, GeneratedSegment)
                else value.stop - value.start
            )
            design_mask.extend([isinstance(value, GeneratedSegment)] * length)
            groups.extend([group] * length)
    return design_mask, groups


def _source_system(
    input: MolecularSystem | Structure | None,
) -> MolecularSystem | None:
    return input.system if isinstance(input, Structure) else input


def _realize_atom_count(
    generated_atoms: ODesignGeneratedAtoms,
    *,
    generator: torch.Generator | None,
    device: torch.device,
) -> int:
    if generated_atoms.min_atoms == generated_atoms.max_atoms:
        return generated_atoms.min_atoms
    return int(
        torch.randint(
            generated_atoms.min_atoms,
            generated_atoms.max_atoms + 1,
            (),
            device=device,
            generator=generator,
        )
    )


def _realize_placement(
    placement: ODesignPlacements,
    *,
    generator: torch.Generator | None,
    device: torch.device,
) -> PlacementSpec:
    chain_lengths = (None,) * len(placement.placements) if placement.chain_lengths is None else placement.chain_lengths
    realized: PlacementSpec = {}
    for (group, chain), chain_length in zip(
        placement.placements.items(),
        chain_lengths,
        strict=True,
    ):
        lengths: dict[str, int] = {}
        generated = [
            (placement_id, value) for placement_id, value in chain.items() if isinstance(value, GeneratedSegment)
        ]
        if chain_length is None:
            for placement_id, segment in generated:
                lengths[placement_id] = (
                    segment.min_length
                    if segment.min_length == segment.max_length
                    else int(
                        torch.randint(
                            segment.min_length,
                            segment.max_length + 1,
                            (),
                            device=device,
                            generator=generator,
                        )
                    )
                )
        else:
            minimum = sum(
                value.min_length if isinstance(value, GeneratedSegment) else value.stop - value.start
                for value in chain.values()
            )
            maximum = sum(
                value.max_length if isinstance(value, GeneratedSegment) else value.stop - value.start
                for value in chain.values()
            )
            if not minimum <= chain_length <= maximum:
                raise ValueError(
                    f"ODesign placement chain length {chain_length} cannot be " f"realized in [{minimum}, {maximum}]"
                )
            lengths.update({placement_id: segment.min_length for placement_id, segment in generated})
            for _ in range(chain_length - minimum):
                available = [
                    (placement_id, segment)
                    for placement_id, segment in generated
                    if lengths[placement_id] < segment.max_length
                ]
                selected = int(
                    torch.randint(
                        len(available),
                        (),
                        device=device,
                        generator=generator,
                    )
                )
                lengths[available[selected][0]] += 1
        realized[group] = {
            placement_id: (
                GeneratedSegment(
                    value.polymer_type,
                    lengths[placement_id],
                    lengths[placement_id],
                )
                if isinstance(value, GeneratedSegment)
                else value
            )
            for placement_id, value in chain.items()
        }
    return realized


def _materialize_polymer_system(
    placement: PlacementSpec,
    source: MolecularSystem | None,
    cyclic_bonds: Sequence[SymbolicBond],
    device: torch.device,
) -> tuple[
    MolecularSystem,
    PlacementMap,
    dict[int, tuple[int, ...]],
    dict[int, tuple[int, ...]],
    Tensor,
    tuple[tuple[int, int], ...],
    frozenset[int],
]:
    if source is None:
        context = MolecularSystem.from_entities({}, {}, device=device)
        context_component_map = torch.empty(0, dtype=torch.int64, device=device)
        context_atom_map = torch.empty(0, dtype=torch.int64, device=device)
        used_entity_ids: set[str] = set()
        used_instance_ids: set[str] = set()
    else:
        context_instance_mask = source.entity_type[source.instance_entity] == ENTITY_MOLECULE
        context, context_component_map, context_atom_map = instance_subgraph(
            source,
            context_instance_mask,
        )
        used_entity_ids = {source.entity_id(index) for index in range(source.entity_symbol.numel())}
        used_instance_ids = {source.instance_id(index) for index in range(source.instance_symbol.numel())}

    chain_systems: list[MolecularSystem] = []
    chain_design_masks: list[Tensor] = []
    chain_source_atom_maps: list[Tensor] = []
    chain_source_components: list[list[tuple[int, int]]] = []
    placement_map: PlacementMap = {}
    chain_instances: list[str] = []
    next_chain = 0
    for declaration_index, (group, chain) in enumerate(placement.items()):
        lengths = [
            value.max_length if isinstance(value, GeneratedSegment) else value.stop - value.start
            for value in chain.values()
        ]
        if sum(lengths) == 0:
            placement_map[group] = {placement_id: () for placement_id in chain}
            continue

        instance_id = f"chain_{next_chain}"
        next_chain += 1
        while instance_id in used_instance_ids:
            instance_id += "_"
        used_instance_ids.add(instance_id)
        entity_id = f"odesign_entity_{declaration_index + 1}"
        while entity_id in used_entity_ids:
            entity_id += "_"
        used_entity_ids.add(entity_id)

        polymer_types: list[PolymerType] = []
        for value in chain.values():
            if isinstance(value, GeneratedSegment):
                polymer_types.append(value.polymer_type)
            else:
                if source is None:
                    raise ValueError("ExistingSpan requires a molecular input")
                instance = source.instance_index(value.instance_id)
                entity = int(source.instance_entity[instance])
                entity_type = int(source.entity_type[entity])
                if entity_type == ENTITY_MOLECULE:
                    raise ValueError("ODesign placement ExistingSpan must select a polymer")
                polymer_types.append(_POLYMER_TYPE_BY_ENTITY[entity_type])
        if len(set(polymer_types)) != 1:
            raise ValueError("one ODesign output chain cannot mix polymer types")
        polymer_type = cast(PolymerType, polymer_types[0])
        placeholder = get_component(_PLACEHOLDER_COMPONENT_ID[polymer_type])
        components = tuple(placeholder for _ in range(sum(lengths)))
        chain_system = MolecularSystem.from_entities(
            {entity_id: PolymerSequence(polymer_type, components)},
            {instance_id: entity_id},
            device=device,
        )
        design_mask = torch.zeros(
            len(components),
            dtype=torch.bool,
            device=device,
        )
        target_components: list[int] = []
        source_components: list[int] = []
        source_component_pairs: list[tuple[int, int]] = []
        offset = 0
        realized_group: dict[str, tuple[ComponentReference, ...]] = {}
        for (placement_id, value), length in zip(
            chain.items(),
            lengths,
            strict=True,
        ):
            occurrences = tuple((instance_id, offset + index) for index in range(length))
            realized_group[placement_id] = occurrences
            if isinstance(value, GeneratedSegment):
                design_mask[offset : offset + length] = True
            else:
                assert source is not None
                source_instance = source.instance_index(value.instance_id)
                source_start = int(source.instance_component_ptr[source_instance])
                for local, source_local in enumerate(range(value.start, value.stop)):
                    target_components.append(offset + local)
                    source_components.append(source_start + source_local)
                    source_component_pairs.append((source_start + source_local, offset + local))
            offset += length
        placement_map[group] = realized_group
        if source_components:
            source_system = cast(MolecularSystem, source)
            chain_system, _, source_atom_map = copy_components(
                chain_system,
                torch.tensor(
                    target_components,
                    dtype=torch.int64,
                    device=device,
                ),
                source_system,
                torch.tensor(
                    source_components,
                    dtype=torch.int64,
                    device=source_system.atom_symbol.device,
                ),
            )
        else:
            source_atom_map = torch.full(
                (0 if source is None else source.atom_symbol.numel(),),
                -1,
                dtype=torch.int64,
                device=device,
            )
        chain_systems.append(chain_system)
        chain_design_masks.append(design_mask)
        chain_source_atom_maps.append(source_atom_map)
        chain_source_components.append(source_component_pairs)
        chain_instances.append(instance_id)

    systems = (context, *chain_systems)
    system, component_maps, atom_maps = disjoint_union(systems)
    design_component_mask = torch.cat(
        [
            torch.zeros(
                context.component_symbol.numel(),
                dtype=torch.bool,
                device=device,
            ),
            *chain_design_masks,
        ]
    )
    source_component_outputs: dict[int, list[int]] = {}
    source_atom_outputs: dict[int, list[int]] = {}
    if source is not None:
        for source_component, context_component in enumerate(context_component_map.detach().cpu().tolist()):
            if context_component >= 0:
                source_component_outputs.setdefault(
                    source_component,
                    [],
                ).append(int(component_maps[0][context_component]))
        for source_atom, context_atom in enumerate(context_atom_map.detach().cpu().tolist()):
            if context_atom >= 0:
                source_atom_outputs.setdefault(source_atom, []).append(int(atom_maps[0][context_atom]))
        for chain_index, (
            source_atom_map,
            source_component_pairs,
        ) in enumerate(
            zip(
                chain_source_atom_maps,
                chain_source_components,
                strict=True,
            ),
            start=1,
        ):
            for source_component, chain_component in source_component_pairs:
                source_component_outputs.setdefault(
                    source_component,
                    [],
                ).append(int(component_maps[chain_index][chain_component]))
            for source_atom, chain_atom in enumerate(source_atom_map.detach().cpu().tolist()):
                if chain_atom >= 0:
                    source_atom_outputs.setdefault(source_atom, []).append(int(atom_maps[chain_index][chain_atom]))

        assembly_edges = (
            (source.bond_role == BOND_ROLE_ASSEMBLY).nonzero(as_tuple=False).flatten().detach().cpu().tolist()
        )
        for edge in assembly_edges:
            left_outputs = source_atom_outputs.get(
                int(source.edge_index[0, edge]),
                (),
            )
            right_outputs = source_atom_outputs.get(
                int(source.edge_index[1, edge]),
                (),
            )
            if len(left_outputs) != 1 or len(right_outputs) != 1:
                continue
            stereo_atoms = None
            source_stereo = source.bond_stereo_index[:, edge]
            if bool((source_stereo >= 0).all()):
                stereo_left = source_atom_outputs.get(
                    int(source_stereo[0]),
                    (),
                )
                stereo_right = source_atom_outputs.get(
                    int(source_stereo[1]),
                    (),
                )
                if len(stereo_left) == 1 and len(stereo_right) == 1:
                    stereo_atoms = (stereo_left[0], stereo_right[0])
            system, old_to_new = form_bond(
                system,
                left_outputs[0],
                right_outputs[0],
                system.atom_symbol.new_empty(0),
                system.atom_symbol.new_empty(0),
                order=(float(source.bond_order[edge]) if bool(source.bond_order_known[edge]) else None),
                aromatic=(bool(source.bond_aromatic[edge]) if bool(source.bond_aromatic_known[edge]) else None),
                kind=_bond_kind(int(source.bond_kind[edge])),
                direction=_bond_direction(int(source.bond_direction[edge])),
                stereo=(source.symbols[int(source.bond_stereo[edge])] if int(source.bond_stereo[edge]) >= 0 else None),
                stereo_atoms=stereo_atoms,
                role="assembly",
            )
            _remap_source_atoms(source_atom_outputs, old_to_new)

    instance_lookup = placement_instance_of(placement)
    cyclic_pairs: list[tuple[int, int]] = []
    cyclic_instances: set[int] = set()
    for bond in cyclic_bonds:
        atom_a = _resolve_bond_atom(
            bond.site_a,
            source,
            system,
            placement_map,
            instance_lookup,
            source_atom_outputs,
        )
        atom_b = _resolve_bond_atom(
            bond.site_b,
            source,
            system,
            placement_map,
            instance_lookup,
            source_atom_outputs,
        )
        component_a = _atom_component_index(system)[atom_a]
        component_b = _atom_component_index(system)[atom_b]
        instance_a = _component_instance_index(system)[component_a]
        instance_b = _component_instance_index(system)[component_b]
        if int(instance_a) != int(instance_b):
            raise ValueError("ODesign supports only head-to-tail cyclic polymer links")
        instance = int(instance_a)
        start = int(system.instance_component_ptr[instance])
        stop = int(system.instance_component_ptr[instance + 1])
        if {int(component_a), int(component_b)} != {start, stop - 1}:
            raise ValueError("ODesign supports only head-to-tail cyclic polymer links")
        leaving_a = _named_component_atoms(
            system,
            int(component_a),
            bond.leaving_a,
        )
        leaving_b = _named_component_atoms(
            system,
            int(component_b),
            bond.leaving_b,
        )
        system, old_to_new = form_bond(
            system,
            atom_a,
            atom_b,
            leaving_a,
            leaving_b,
            order=bond.order,
            aromatic=bond.aromatic,
            kind=bond.kind,
            direction=bond.direction,
            stereo=bond.stereo,
            role="assembly",
        )
        _remap_source_atoms(source_atom_outputs, old_to_new)
        cyclic_pairs = [(int(old_to_new[left]), int(old_to_new[right])) for left, right in cyclic_pairs]
        cyclic_pairs.append((int(old_to_new[atom_a]), int(old_to_new[atom_b])))
        cyclic_instances.add(instance)

    return (
        system,
        placement_map,
        {source_component: tuple(outputs) for source_component, outputs in source_component_outputs.items()},
        {source_atom: tuple(outputs) for source_atom, outputs in source_atom_outputs.items()},
        design_component_mask,
        tuple(cyclic_pairs),
        frozenset(cyclic_instances),
    )


def _bond_kind(value: int) -> BondKind:
    return cast(
        BondKind,
        {
            BOND_KIND_COVALENT: "covalent",
            BOND_KIND_DATIVE: "dative",
        }[value],
    )


def _bond_direction(value: int) -> BondDirection:
    return cast(
        BondDirection,
        {
            BOND_DIRECTION_UNDIRECTED: "undirected",
            BOND_DIRECTION_ENDPOINT_0_DONOR: "a_to_b",
            BOND_DIRECTION_ENDPOINT_1_DONOR: "b_to_a",
        }[value],
    )


def _remap_source_atoms(
    source_atom_outputs: dict[int, list[int]],
    old_to_new: Tensor,
) -> None:
    for source_atom, outputs in source_atom_outputs.items():
        source_atom_outputs[source_atom] = [
            int(old_to_new[output]) for output in outputs if int(old_to_new[output]) >= 0
        ]


def _resolve_bond_atom(
    reference: AtomSite | SymbolicAtomSite,
    source: MolecularSystem | None,
    system: MolecularSystem,
    placement_map: PlacementMap,
    instance_lookup: dict[str, str],
    source_atom_outputs: dict[int, list[int]],
) -> int:
    if isinstance(reference, AtomSite):
        if source is None:
            raise ValueError("existing cyclic-bond atoms require molecular input")
        outputs = source_atom_outputs.get(source.atom_index(reference), ())
        if len(outputs) != 1:
            raise ValueError(f"ODesign cyclic-bond atom {reference!r} is not uniquely placed")
        return outputs[0]
    component_reference = resolve_placement(
        reference.residue,
        placement_map,
        instance_lookup,
    )
    return system.atom_index(
        AtomSite(
            component_reference[0],
            component_reference[1],
            reference.atom_id,
        )
    )


def _named_component_atoms(
    system: MolecularSystem,
    component: int,
    atom_ids: frozenset[str],
) -> Tensor:
    start = int(system.component_atom_ptr[component])
    stop = int(system.component_atom_ptr[component + 1])
    return torch.tensor(
        [atom for atom in range(start, stop) if system.symbols[int(system.atom_symbol[atom])] in atom_ids],
        dtype=torch.int64,
        device=system.atom_symbol.device,
    )


def _atom_component_index(system: MolecularSystem) -> Tensor:
    return torch.repeat_interleave(
        torch.arange(
            system.component_symbol.numel(),
            dtype=torch.int64,
            device=system.atom_symbol.device,
        ),
        system.component_atom_ptr[1:] - system.component_atom_ptr[:-1],
    )


def _component_instance_index(system: MolecularSystem) -> Tensor:
    return torch.repeat_interleave(
        torch.arange(
            system.instance_symbol.numel(),
            dtype=torch.int64,
            device=system.atom_symbol.device,
        ),
        system.instance_component_ptr[1:] - system.instance_component_ptr[:-1],
    )


def _model_atom_index(
    system: MolecularSystem,
    design_component_mask: Tensor,
) -> Tensor:
    atom_component = _atom_component_index(system)
    component_instance = _component_instance_index(system)
    atom_entity = system.instance_entity[component_instance[atom_component]]
    model_mask = (system.atomic_number != 1) & system.reference_coordinate_mask
    generated_protein = design_component_mask[atom_component] & (system.entity_type[atom_entity] == ENTITY_PROTEIN)
    backbone_symbols = system.atom_symbol.new_tensor(
        [index for index, symbol in enumerate(system.symbols) if symbol in {"N", "CA", "C", "O"}]
    )
    protein_backbone = torch.isin(system.atom_symbol, backbone_symbols)
    model_mask = torch.where(
        generated_protein,
        protein_backbone,
        model_mask,
    )
    return model_mask.nonzero(as_tuple=False).flatten()


def _token_axes(
    system: MolecularSystem,
    model_atom_index: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    model_row = {atom: row for row, atom in enumerate(model_atom_index.detach().cpu().tolist())}
    token_component: list[int] = []
    token_atom: list[int] = []
    token_instance: list[int] = []
    atom_to_token = torch.empty(
        model_atom_index.numel(),
        dtype=torch.int64,
        device=system.atom_symbol.device,
    )
    for instance in range(system.instance_symbol.numel()):
        entity = int(system.instance_entity[instance])
        is_polymer = int(system.entity_type[entity]) != ENTITY_MOLECULE
        start = int(system.instance_component_ptr[instance])
        stop = int(system.instance_component_ptr[instance + 1])
        for component in range(start, stop):
            atom_start = int(system.component_atom_ptr[component])
            atom_stop = int(system.component_atom_ptr[component + 1])
            atoms = [atom for atom in range(atom_start, atom_stop) if atom in model_row]
            component_id = system.symbols[int(system.component_symbol[component])]
            if is_polymer and component_id in _STANDARD_POLYMER_COMPONENTS:
                token = len(token_component)
                token_component.append(component)
                token_atom.append(-1)
                token_instance.append(instance)
                for atom in atoms:
                    atom_to_token[model_row[atom]] = token
            else:
                for atom in atoms:
                    token = len(token_component)
                    token_component.append(component)
                    token_atom.append(atom)
                    token_instance.append(instance)
                    atom_to_token[model_row[atom]] = token
    return (
        torch.tensor(
            token_component,
            dtype=torch.int64,
            device=system.atom_symbol.device,
        ),
        torch.tensor(
            token_atom,
            dtype=torch.int64,
            device=system.atom_symbol.device,
        ),
        torch.tensor(
            token_instance,
            dtype=torch.int64,
            device=system.atom_symbol.device,
        ),
        atom_to_token,
    )


def _reference_geometry(
    system: MolecularSystem,
    atom_index: Tensor,
    design_component_mask: Tensor,
) -> tuple[Tensor, Tensor]:
    atom_component = _atom_component_index(system)
    component_instance = _component_instance_index(system)
    reference_positions = system.reference_coordinates[atom_index].clone()
    selected_component = atom_component[atom_index]
    selected_entity = system.instance_entity[component_instance[selected_component]]
    generated_protein = design_component_mask[selected_component] & (
        system.entity_type[selected_entity] == ENTITY_PROTEIN
    )
    if bool(generated_protein.any()):
        gly = get_component("GLY")
        gly_coordinates = {
            atom.atom_id: coordinate
            for atom, coordinate in zip(
                gly.atoms,
                gly.reference_coordinates,
                strict=True,
            )
            if coordinate is not None
        }
        for row in generated_protein.nonzero(as_tuple=False).flatten().tolist():
            atom_name = system.symbols[int(system.atom_symbol[atom_index[row]])]
            reference_positions[row] = reference_positions.new_tensor(gly_coordinates[atom_name])
    sums = torch.zeros(
        system.component_symbol.numel(),
        3,
        dtype=reference_positions.dtype,
        device=reference_positions.device,
    )
    counts = torch.zeros(
        system.component_symbol.numel(),
        dtype=reference_positions.dtype,
        device=reference_positions.device,
    )
    sums.index_add_(0, selected_component, reference_positions)
    counts.index_add_(
        0,
        selected_component,
        torch.ones_like(selected_component, dtype=reference_positions.dtype),
    )
    reference_positions -= sums[selected_component] / counts[selected_component].unsqueeze(-1)
    return reference_positions, selected_component


def _observed_coordinates(
    structure: Structure | None,
    model_atom_index: Tensor,
    output_to_source: dict[int, int],
) -> tuple[Tensor, Tensor]:
    coordinates = torch.zeros(
        model_atom_index.numel(),
        3,
        dtype=torch.float32,
        device=model_atom_index.device,
    )
    mask = torch.zeros(
        model_atom_index.numel(),
        dtype=torch.bool,
        device=model_atom_index.device,
    )
    if structure is None:
        return coordinates, mask
    source_index = torch.tensor(
        [output_to_source.get(output, -1) for output in model_atom_index.detach().cpu().tolist()],
        dtype=torch.int64,
        device=structure.coordinates.device,
    )
    present = source_index >= 0
    gathered_mask = structure.coordinate_mask[source_index.clamp_min(0)] & present
    coordinates = structure.coordinates[source_index.clamp_min(0)].to(
        device=model_atom_index.device,
        dtype=torch.float32,
    )
    mask = gathered_mask.to(model_atom_index.device)
    coordinates = coordinates.masked_fill(~mask.unsqueeze(-1), 0)
    return coordinates, mask


def _condition_atom_mask(
    atoms: Sequence[AtomSite],
    source: MolecularSystem | None,
    source_atom_outputs: dict[int, tuple[int, ...]],
    model_atom_index: Tensor,
) -> Tensor:
    mask = torch.zeros(
        model_atom_index.numel(),
        dtype=torch.bool,
        device=model_atom_index.device,
    )
    if not atoms:
        return mask
    if source is None:
        raise ValueError("condition atoms require molecular input")
    model_row = {atom: row for row, atom in enumerate(model_atom_index.detach().cpu().tolist())}
    for site in atoms:
        source_atom = source.atom_index(site)
        outputs = source_atom_outputs.get(source_atom, ())
        if not outputs:
            raise ValueError(f"ODesign output does not contain condition atom {site!r}")
        for output in outputs:
            mask[model_row[output]] = True
    return mask


def _hotspot_tokens(
    hotspots: Sequence[Hotspot],
    source: MolecularSystem | None,
    source_component_outputs: dict[int, tuple[int, ...]],
    source_atom_outputs: dict[int, tuple[int, ...]],
    token_component_index: Tensor,
    canonical_atom_to_token: Tensor,
) -> Tensor:
    hotspot = torch.zeros(
        token_component_index.numel(),
        dtype=torch.bool,
        device=token_component_index.device,
    )
    component_token: dict[int, int] = {}
    for token, component in enumerate(token_component_index.detach().cpu().tolist()):
        component_token.setdefault(component, token)
    for condition in hotspots:
        selected = False
        for reference in condition.residues:
            if isinstance(
                reference,
                (str, SymbolicResidueSite, SymbolicAtomSite),
            ):
                continue
            if source is None:
                raise ValueError("existing hotspot residues require molecular input")
            if isinstance(reference, AtomSite):
                outputs = source_atom_outputs.get(
                    source.atom_index(reference),
                    (),
                )
                for output in outputs:
                    token = int(canonical_atom_to_token[output])
                    if token >= 0:
                        hotspot[token] = True
                        selected = True
            else:
                source_component = source.component_index(reference)
                for output in source_component_outputs.get(
                    source_component,
                    (),
                ):
                    component_token_index = component_token.get(output)
                    if component_token_index is not None:
                        hotspot[component_token_index] = True
                        selected = True
        if not selected:
            raise ValueError("ODesign Hotspot must select at least one input residue")
    return hotspot


def _atom_features(
    system: MolecularSystem,
    atom_index: Tensor,
    reference_mask: Tensor,
) -> Tensor:
    element = F.one_hot(
        system.atomic_number[atom_index].long().clamp(min=1, max=129) - 1,
        129,
    ).to(torch.float32)
    characters = torch.zeros(
        atom_index.numel(),
        4,
        64,
        dtype=torch.float32,
        device=atom_index.device,
    )
    for row, atom in enumerate(atom_index.detach().cpu().tolist()):
        name = system.symbols[int(system.atom_symbol[atom])]
        if len(name) > 4:
            raise ValueError(f"ODesign cannot encode atom name longer than four characters: " f"{name!r}")
        for character_index, character in enumerate(name.ljust(4)):
            characters[
                row,
                character_index,
                ord(character) - 32,
            ] = 1
    return torch.cat(
        [
            reference_mask.to(torch.float32).unsqueeze(-1),
            element,
            characters.flatten(-2),
        ],
        dim=-1,
    )


def _ligand_slot_atom_features(
    num_slots: int,
    *,
    device: torch.device,
) -> Tensor:
    features = torch.zeros(
        num_slots,
        386,
        dtype=torch.float32,
        device=device,
    )
    features[:, 0] = 1
    features[:, 1 + 128] = 1
    character_offset = 1 + 129
    features[:, character_offset + ord("-") - 32] = 1
    for character_index in range(1, 4):
        features[:, character_offset + character_index * 64] = 1
    return features


def _token_features(
    system: MolecularSystem,
    token_component_index: Tensor,
    token_atom_index: Tensor,
    design_component_mask: Tensor,
    profile: Tensor,
    deletion_mean: Tensor,
    hotspot: Tensor,
) -> Tensor:
    features = torch.zeros(
        token_component_index.numel(),
        453,
        dtype=torch.float32,
        device=token_component_index.device,
    )
    component_instance = _component_instance_index(system)
    for token, (component, atom) in enumerate(
        zip(
            token_component_index.detach().cpu().tolist(),
            token_atom_index.detach().cpu().tolist(),
            strict=True,
        )
    ):
        if component < 0 or atom >= 0:
            residue_type = "-L"
        elif bool(design_component_mask[component]):
            entity = int(system.instance_entity[component_instance[component]])
            residue_type = "-P" if int(system.entity_type[entity]) == ENTITY_PROTEIN else "-N"
        else:
            residue_type = system.symbols[int(system.component_symbol[component])]
        features[
            token,
            384 + _RESIDUE_TYPE_INDEX.get(residue_type, 20),
        ] = 1
    features[:, 419:451] = profile
    features[:, 451] = deletion_mean
    features[:, 452] = hotspot.to(torch.float32)
    return features


def _token_bonds(
    system: MolecularSystem,
    canonical_atom_to_token: Tensor,
    token_count: int,
    excluded: Sequence[tuple[int, int]],
) -> Tensor:
    bonds = torch.zeros(
        token_count,
        token_count,
        dtype=torch.float32,
        device=system.atom_symbol.device,
    )
    left = canonical_atom_to_token[system.edge_index[0]]
    right = canonical_atom_to_token[system.edge_index[1]]
    valid = (left >= 0) & (right >= 0) & (left != right)
    for atom_a, atom_b in excluded:
        valid &= ~(
            ((system.edge_index[0] == atom_a) & (system.edge_index[1] == atom_b))
            | ((system.edge_index[0] == atom_b) & (system.edge_index[1] == atom_a))
        )
    bonds[left[valid], right[valid]] = 1
    bonds[right[valid], left[valid]] = 1
    return bonds


def _constraint_distogram(
    system: MolecularSystem,
    model_atom_index: Tensor,
    token_component_index: Tensor,
    canonical_atom_to_token: Tensor,
    observed_positions: Tensor,
    constraint_coordinate_mask: Tensor,
) -> Tensor:
    token_count = token_component_index.numel()
    values = torch.zeros(
        token_count,
        token_count,
        1,
        dtype=observed_positions.dtype,
        device=observed_positions.device,
    )
    atom_component = _atom_component_index(system)
    component_instance = _component_instance_index(system)
    selected: dict[int, int] = {}
    for model_row, atom in enumerate(model_atom_index.detach().cpu().tolist()):
        if not bool(constraint_coordinate_mask[model_row]):
            continue
        component = int(atom_component[atom])
        instance = int(component_instance[component])
        entity = int(system.instance_entity[instance])
        component_id = system.symbols[int(system.component_symbol[component])]
        atom_name = system.symbols[int(system.atom_symbol[atom])]
        if int(system.entity_type[entity]) != ENTITY_MOLECULE and component_id in _STANDARD_POLYMER_COMPONENTS:
            center = "CA" if int(system.entity_type[entity]) == ENTITY_PROTEIN else "C1'"
            if atom_name != center:
                continue
        selected[int(canonical_atom_to_token[atom])] = model_row
    if selected:
        tokens = torch.tensor(
            tuple(selected),
            dtype=torch.int64,
            device=values.device,
        )
        atoms = torch.tensor(
            tuple(selected.values()),
            dtype=torch.int64,
            device=values.device,
        )
        distances = torch.cdist(
            observed_positions[atoms],
            observed_positions[atoms],
        )
        values[tokens[:, None], tokens[None, :], 0] = distances
    return values


def _token_indices(
    system: MolecularSystem,
    token_component_index: Tensor,
    token_instance_index: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    component_instance = _component_instance_index(system)
    instance = token_instance_index
    entity = system.instance_entity[instance]
    symmetry = torch.zeros_like(instance)
    seen: dict[int, int] = {}
    for row, (instance_index, entity_index) in enumerate(
        zip(
            instance.detach().cpu().tolist(),
            entity.detach().cpu().tolist(),
            strict=True,
        )
    ):
        if row and instance_index == int(instance[row - 1]):
            symmetry[row] = symmetry[row - 1]
            continue
        symmetry[row] = seen.get(entity_index, 0)
        seen[entity_index] = int(symmetry[row]) + 1
    residue = torch.zeros_like(token_component_index)
    valid = token_component_index >= 0
    residue[valid] = (
        token_component_index[valid] - system.instance_component_ptr[component_instance[token_component_index[valid]]]
    )
    return instance, entity, symmetry, residue


def _cyclic_period(
    token_instance_index: Tensor,
    cyclic_instances: frozenset[int],
) -> Tensor:
    period = torch.zeros_like(token_instance_index)
    for instance in cyclic_instances:
        mask = token_instance_index == instance
        period[mask] = mask.sum()
    return period


def _msa_symbol_token(symbol: str, polymer_type: str) -> int:
    if symbol == "-":
        return _MSA_GAP
    return _MSA_SYMBOL_TOKENS[polymer_type].get(
        symbol,
        _MSA_UNKNOWN,
    )


def _msa_features(
    msa: MSA | MSACollection | None,
    system: MolecularSystem,
    token_component_index: Tensor,
    token_instance_index: Tensor,
    design_component_mask: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    token_count = token_component_index.numel()
    device = token_component_index.device
    profile = torch.zeros(token_count, 32, device=device)
    deletion_mean = torch.zeros(token_count, device=device)
    if msa is None:
        row = torch.full(
            (token_count,),
            31,
            dtype=torch.int64,
            device=device,
        )
        for token, component in enumerate(token_component_index.detach().cpu().tolist()):
            component_id = (
                "-"
                if component < 0 or bool(design_component_mask[component])
                else system.symbols[int(system.component_symbol[component])]
            )
            residue_token = min(
                _RESIDUE_TYPE_INDEX.get(component_id, 20),
                31,
            )
            row[token] = residue_token
            profile[token, residue_token] = 1
        features = torch.cat(
            [
                F.one_hot(row, 32).to(torch.float32),
                torch.zeros(token_count, 2, device=device),
            ],
            dim=-1,
        ).unsqueeze(0)
        return (
            features,
            torch.ones(
                1,
                token_count,
                dtype=torch.bool,
                device=device,
            ),
            profile,
            deletion_mean,
        )

    resolved = resolve_msa_bindings(msa, system)
    columns = tuple(value for value, _, _ in resolved)
    targets = tuple(target for _, target, _ in resolved)
    entities = tuple(entity for _, _, entity in resolved)
    polymer_types = [_POLYMER_TYPE_BY_ENTITY[int(system.entity_type[entity])] for entity in entities]
    component_instance = _component_instance_index(system)
    tokens_by_entity: dict[int, list[tuple[int, int, int]]] = {}
    for token, component in enumerate(token_component_index.detach().cpu().tolist()):
        if component < 0:
            continue
        instance = int(component_instance[component])
        entity = int(system.instance_entity[instance])
        local = component - int(system.instance_component_ptr[instance])
        tokens_by_entity.setdefault(entity, []).append((token, local, instance))
    plan: list[tuple[tuple[int, int, bool], ...]]
    if isinstance(msa, MSA):
        plan = [((0, row, False),) for row in range(len(msa.unpaired_sequences))]
    else:
        plan = [
            tuple(
                (
                    column,
                    int(msa.row_pairing[row, column]),
                    bool(msa.row_is_paired[row, column]),
                )
                for column in range(len(columns))
                if msa.row_pairing[row, column] != -1
            )
            for row in range(msa.row_pairing.shape[0])
        ]
    rows = []
    deletions = []
    for selections in plan:
        token_row = torch.full(
            (token_count,),
            31,
            dtype=torch.int64,
            device=device,
        )
        deletion_row = torch.zeros(token_count, device=device)
        for column, source_row, paired in selections:
            value = columns[column]
            sequences = value.paired_sequences if paired else value.unpaired_sequences
            deletion_counts = value.paired_deletion_counts if paired else value.unpaired_deletion_counts
            for token, local, instance in tokens_by_entity.get(
                entities[column],
                (),
            ):
                if targets[column] is not None and instance != targets[column]:
                    continue
                token_row[token] = _msa_symbol_token(
                    sequences[source_row][local],
                    polymer_types[column],
                )
                deletion_row[token] = float(deletion_counts[source_row, local])
        rows.append(token_row)
        deletions.append(deletion_row)
    tokens = torch.stack(rows)
    deletion_counts = torch.stack(deletions)
    one_hot = F.one_hot(tokens, 32).to(torch.float32)
    profile = one_hot.mean(dim=0)
    deletion_mean = deletion_counts.mean(dim=0)
    features = torch.cat(
        [
            one_hot,
            (deletion_counts > 0).to(torch.float32).unsqueeze(-1),
            ((2 / math.pi) * torch.atan(deletion_counts / 3)).unsqueeze(-1),
        ],
        dim=-1,
    )
    return (
        features,
        torch.ones_like(tokens, dtype=torch.bool),
        profile,
        deletion_mean,
    )


def _gather_target(
    target: Structure,
    rows: Sequence[int],
) -> tuple[Tensor, Tensor]:
    index = torch.tensor(
        rows,
        dtype=torch.int64,
        device=target.coordinates.device,
    )
    valid = index >= 0
    coordinates = target.coordinates[index.clamp_min(0)].unsqueeze(0)
    coordinate_mask = (target.coordinate_mask[index.clamp_min(0)] & valid).unsqueeze(0)
    return (
        coordinates.masked_fill(
            ~coordinate_mask.unsqueeze(-1),
            0,
        ),
        coordinate_mask,
    )
