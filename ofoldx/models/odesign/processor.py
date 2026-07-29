r"""
Input processing for ODesign structure generation.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import cast, get_args

import torch
import torch.nn.functional as F
from torch import Tensor

from ...data.bonds import SymbolicBond
from ...data.components import get_component
from ...data.conditions import Hotspot
from ...data.evidence import MSA, MSACollection, msa_bindings, resolve_msa_bindings
from ...data.molecular import (
    AtomSite,
    ChemicalComponent,
    MolecularSystem,
    Molecule,
    PolymerSequence,
    PolymerType,
    SystemBond,
)

# Hotspot, SymbolicBond, ODesignPlacements, and ODesignReferenceCoordinates are the exact controls
# this Processor consumes. Hotspot and SymbolicBond reuse shared biological values from
# ``ofoldx.data``; the Processor validates their ODesign semantics inline. The
# generation-topology placement records (GeneratedSegment / ExistingSpan / SymbolicResidueSite
# / SymbolicAtomSite) and resolution helpers come from
# ``ofoldx.data.placement``.
from ...data.placement import (
    ExistingSpan,
    GeneratedSegment,
    InstanceIdAllocator,
    PlacementMap,
    PlacementSpec,
    SymbolicAtomSite,
    SymbolicResidueSite,
)
from ...data.placement import instance_of as _placement_instance_of
from ...data.placement import resolve as _resolve_site
from ...data.placement import validate_placement
from ...data.structure import Structure
from ...processor import Processor
from .config import ODesignLigandProcessorConfig, ODesignProcessorConfig
from .model import (
    ODesignDesignedResidue,
    ODesignFixedResidue,
    ODesignGeneratedAtoms,
    ODesignInput,
    ODesignLayout,
    ODesignLigandInput,
    ODesignLigandLayout,
    ODesignPlacements,
    ODesignProvisionalAtomSlot,
    ODesignReferenceCoordinates,
    ODesignResidueSpec,
    ODesignStructureTarget,
)

_CORE_POLYMER_TYPES: frozenset[str] = frozenset(get_args(PolymerType))


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

_STANDARD_POLYMER_COMPONENTS: frozenset[str] = frozenset(_RESIDUE_TYPE_INDEX) - {
    "-",
    "-P",
    "-N",
    "-L",
}


# ODesign realizes a de-novo polymer GeneratedSegment as a run
# of placeholder components of the segment's polymer type (protein -> UNK, rna -> N, dna -> DN).
# De-novo ligand placeholders do not define a real chemical graph. A generated
# ligand segment therefore has no entry here and is rejected at the realization boundary below:
# a de-novo molecule has no PlacementValue. Existing or fixed ligand context is unaffected.
_GENERATED_COMPONENT_ID = {"protein": "UNK", "rna": "N", "dna": "DN"}

# MSA cells are uppercase one-letter symbols or the ``-`` gap, not ChemicalComponent IDs.
# Standard symbols resolve to residue tokens, gaps to the gap token, and ambiguity symbols to ``UNK``.
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


@dataclass(eq=False)
class _ODesignPlan:
    system: MolecularSystem
    source_structure: Structure | None
    atom_sites: tuple[AtomSite, ...]
    placement: ODesignPlacements | None
    conditions: tuple[Hotspot | SymbolicBond | ODesignReferenceCoordinates, ...]
    placement_map: PlacementMap
    instance_of: dict[str, str]
    source_residue_map: dict[tuple[str, int], tuple[str, int]]
    source_atom_map: dict[AtomSite, AtomSite]
    cyclic_instances: frozenset[str]
    cyclic_links: frozenset[frozenset[AtomSite]]


@dataclass(eq=False)
class _ODesignLigandPlan:
    context_system: MolecularSystem | None
    source_structure: Structure | None
    context_atom_sites: tuple[AtomSite, ...]
    generated_atoms: ODesignGeneratedAtoms
    conditions: tuple[Hotspot | ODesignReferenceCoordinates, ...]
    source_residue_map: dict[tuple[str, int], tuple[str, int]]
    source_atom_map: dict[AtomSite, AtomSite]


class ODesignProcessor(Processor):
    r"""
    Convert supported structure-design inputs into `ODesignInput`.

    Args:
        config: ODesign input-processing options. Defaults to
            ``ODesignProcessorConfig()``.
        device: Device for produced tensors.
        dtype: Floating-point dtype for produced tensors.
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

    def realize_ranges(
        self,
        input: ODesignInput,
        *,
        generator: torch.Generator | None = None,
    ) -> ODesignInput:
        r"""
        Sample variable placement lengths and rebuild a fixed tensor layout.

        Args:
            input: Design input whose placement contains variable-length generated
                segments.
            generator: Random generator used to sample segment lengths.

        Returns:
            An [`ODesignInput`][] with a fixed-length molecular layout and tensor axes.
        """

        return _realize_ranges(input, generator)

    def __call__(
        self,
        input: MolecularSystem | Structure | None,
        *,
        placement: ODesignPlacements | None = None,
        msa: MSA | MSACollection | None = None,
        conditions: Sequence[Hotspot | SymbolicBond | ODesignReferenceCoordinates] = (),
    ) -> ODesignInput:
        r"""
        Build model inputs for one structure-design problem.

        Args:
            input: Optional canonical context system or structure.
            placement: Optional topology of existing and generated molecular segments.
            msa: Optional canonical MSA evidence.
            conditions: Hotspot, symbolic-bond, and reference-coordinate conditions.

        Returns:
            An [`ODesignInput`][] with batch size one and aligned token, atom, MSA,
            condition, placement, and molecular-layout axes.
        """

        plan = self._plan(input, placement, conditions)
        return self._lower(plan, msa)

    def process_training(
        self,
        input: MolecularSystem | Structure | None,
        target: Structure,
        *,
        placement: ODesignPlacements | None = None,
        msa: MSA | MSACollection | None = None,
        conditions: Sequence[Hotspot | SymbolicBond | ODesignReferenceCoordinates] = (),
    ) -> tuple[ODesignInput, ODesignStructureTarget]:
        r"""
        Lower one design problem and its separate structure target.

        Existing input atoms are matched through the placement's source-to-output
        identity map. Generated output atoms match target sites directly.

        Args:
            input: Optional canonical context system or structure visible to the
                Model.
            target: Canonical target Structure whose coordinate rows form the
                complete target pool.
            placement: Optional topology of existing and generated molecular
                segments.
            msa: Optional canonical MSA evidence.
            conditions: Hotspot, symbolic-bond, and reference-coordinate
                conditions.

        Returns:
            The ordinary Model input and a separate target-to-atom-axis mapping.
        """

        plan = self._plan(input, placement, conditions)
        model_input = self._lower(plan, msa)
        target_row = {site: index for index, site in enumerate(target.sites)}
        source_by_output = {output_site: source_site for source_site, output_site in plan.source_atom_map.items()}
        atom_index = []
        for output_site in plan.atom_sites:
            row = target_row.get(output_site)
            if row is None:
                source_site = source_by_output.get(output_site)
                if source_site is not None:
                    row = target_row.get(source_site)
            atom_index.append(-1 if row is None else row)
        structure_target = ODesignStructureTarget(
            coordinates=torch.from_numpy(target.coordinates.copy())
            .unsqueeze(0)
            .to(device=self.device, dtype=self.dtype),
            coordinate_mask=torch.from_numpy(target.coordinate_mask.copy()).unsqueeze(0).to(self.device),
            atom_index=torch.tensor(
                atom_index,
                dtype=torch.int64,
                device=self.device,
            ).unsqueeze(0),
            batch_size=[1],
        )
        return model_input, structure_target

    def _plan(
        self,
        input: MolecularSystem | Structure | None,
        placement: ODesignPlacements | None,
        conditions: Sequence[Hotspot | SymbolicBond | ODesignReferenceCoordinates],
    ) -> _ODesignPlan:
        condition_values = tuple(conditions)
        for condition in condition_values:
            if not isinstance(condition, (ODesignReferenceCoordinates, Hotspot, SymbolicBond)):
                raise TypeError(f"ODesign cannot consume {type(condition).__name__}")

        # ODesign validates condition semantics inline: type and intent above, endpoints and
        # referenced atoms below.
        structure: Structure | None
        source_system: MolecularSystem | None
        if isinstance(input, Structure):
            structure = input
            source_system = input.system
        else:
            structure = None
            source_system = input
        # A GeneratedSegment carries no embedded id; its identity is the
        # globally-unique ``placement_id`` inner key of the PlacementSpec mapping.
        segments = {
            placement_id: part
            for chain in (() if placement is None else placement.placements.values())
            for placement_id, part in chain.items()
            if isinstance(part, GeneratedSegment)
        }
        generated_polymer_types = {segment.polymer_type for segment in segments.values()}
        if self.config.generated_modality == "protein":
            allowed_polymer_types = {"protein"}
        elif self.config.generated_modality == "nucleic_acid":
            allowed_polymer_types = {"dna", "rna"}
        else:
            raise ValueError(f"unknown ODesign generated_modality: {self.config.generated_modality!r}")
        if not generated_polymer_types <= allowed_polymer_types:
            raise ValueError(
                f"ODesign {self.config.generated_modality} artifact cannot generate "
                f"{sorted(generated_polymer_types - allowed_polymer_types)!r}"
            )
        for condition in condition_values:
            if not isinstance(condition, Hotspot):
                continue
            for reference in condition.residues:
                if isinstance(reference, (AtomSite, SymbolicAtomSite)):
                    continue
                if isinstance(reference, str):
                    # A whole-placement reference is the bare globally-unique placement_id.
                    polymer_type = segments[reference].polymer_type
                elif isinstance(reference, SymbolicResidueSite):
                    polymer_type = segments[reference.placement_id].polymer_type
                else:
                    if source_system is None:
                        raise ValueError("existing Hotspot residues require molecular input")
                    reference_entity = source_system.entities[source_system.instances[reference[0]]]
                    if not isinstance(reference_entity, PolymerSequence):
                        raise ValueError("non-protein Hotspot residues require exact atom references")
                    polymer_type = reference_entity.polymer_type
                if polymer_type != "protein":
                    raise ValueError("non-protein Hotspot residues require exact atom references")
        if placement is not None:
            (
                system,
                sites,
                placement_map,
                instance_of,
                source_residue_map,
                source_atom_map,
            ) = _materialize_placement(placement, source_system)
        else:
            if source_system is None:
                raise ValueError("input=None requires one PlacementSpec that determines generated topology")
            system = source_system
            sites = _sites_with_reference_coordinates(system)
            placement_map = {}
            instance_of = {}
            source_residue_map = {
                (instance_id, component_index): (instance_id, component_index)
                for instance_id, entity_id in system.instances.items()
                if isinstance(entity := system.entities[entity_id], PolymerSequence)
                for component_index in range(len(entity))
            }
            source_atom_map = {site: site for site in sites}

        system, cyclic_instances, cyclic_links = _materialize_cyclic_links(
            system,
            condition_values,
            placement_map,
            instance_of,
            source_atom_map,
        )

        unsupported_components = {
            component.component_id
            for entity in system.entities.values()
            if isinstance(entity, PolymerSequence)
            for component in entity.components
            if component.component_id not in _STANDARD_POLYMER_COMPONENTS
        }
        if unsupported_components:
            raise ValueError("ODesign has no residue token for components " f"{sorted(unsupported_components)!r}")

        return _ODesignPlan(
            system=system,
            source_structure=structure,
            atom_sites=sites,
            placement=placement,
            conditions=condition_values,
            placement_map=placement_map,
            instance_of=instance_of,
            source_residue_map=source_residue_map,
            source_atom_map=source_atom_map,
            cyclic_instances=cyclic_instances,
            cyclic_links=cyclic_links,
        )

    def _lower(
        self,
        plan: _ODesignPlan,
        msa: MSA | MSACollection | None,
    ) -> ODesignInput:
        system = plan.system
        structure = plan.source_structure
        sites = plan.atom_sites
        placement = plan.placement
        condition_values = plan.conditions
        placement_map = plan.placement_map
        instance_of = plan.instance_of
        source_residue_map = plan.source_residue_map
        source_atom_map = plan.source_atom_map
        cyclic_instances = plan.cyclic_instances
        cyclic_links = plan.cyclic_links
        msa_value = _validate_msa(msa, system)

        if not sites:
            raise ValueError("ODesign requires at least one heavy atom")
        generated_residues = {
            occurrence
            for placements in placement_map.values()
            for occurrences in placements.values()
            for occurrence in occurrences
        }
        token_sites, token_residues, token_entities = _build_tokens(system, sites)
        site_to_token = {site: token_index for token_index, group in enumerate(token_sites) for site in group}
        site_to_atom = {site: index for index, site in enumerate(sites)}
        observed_positions = torch.zeros(len(sites), 3, dtype=torch.float32)
        observed_mask = torch.zeros(len(sites), dtype=torch.bool)
        if structure is not None:
            observed = {site: index for index, site in enumerate(structure.sites)}
            source_by_output = {output: source for source, output in source_atom_map.items()}
            for atom_index, output_site in enumerate(sites):
                source_site = source_by_output.get(output_site)
                if source_site is None:
                    continue
                source_index = observed.get(source_site)
                if source_index is None or not bool(structure.coordinate_mask[source_index]):
                    continue
                observed_positions[atom_index] = torch.from_numpy(structure.coordinates[source_index].copy())
                observed_mask[atom_index] = True

        reference_positions = torch.empty_like(observed_positions)
        reference_space_id = []
        reference_spaces: dict[tuple[str, int], int] = {}
        for atom_index, site in enumerate(sites):
            entity_id = system.instances[site.instance_id]
            entity = system.entities[entity_id]
            component = system.component((site.instance_id, site.component_index))
            if isinstance(entity, PolymerSequence):
                if (
                    site.instance_id,
                    site.component_index,
                ) in generated_residues and entity.polymer_type == "protein":
                    reference_component = get_component("GLY")
                    owner = "the generated-protein GLY template"
                else:
                    reference_component = component
                    owner = f"component {component.component_id!r}"
            else:
                reference_component = component
                owner = f"molecule {entity_id!r}"
            coordinates = reference_component.reference_coordinates
            atoms = reference_component.atoms
            if not coordinates:
                raise ValueError(f"ODesign requires reference coordinates for {owner}")
            coordinate = coordinates[next(index for index, atom in enumerate(atoms) if atom.atom_id == site.atom_id)]
            if coordinate is None:
                raise ValueError(f"ODesign requires a reference coordinate for {owner} atom {site.atom_id!r}")
            reference_positions[atom_index] = torch.tensor(coordinate)

            reference_space = (site.instance_id, site.component_index)
            if reference_space not in reference_spaces:
                reference_spaces[reference_space] = len(reference_spaces)
            reference_space_id.append(reference_spaces[reference_space])

        reference_space_id = torch.tensor(reference_space_id, dtype=torch.int64)
        for space_id in range(len(reference_spaces)):
            space_mask = reference_space_id == space_id
            reference_positions[space_mask] -= reference_positions[space_mask].mean(dim=0)
        reference_mask = torch.ones(len(sites), dtype=torch.bool)

        hotspot = _hotspot_tokens(
            condition_values,
            placement_map,
            instance_of,
            source_residue_map,
            source_atom_map,
            site_to_token,
        )
        fixed_coordinate_mask = torch.zeros(len(sites), dtype=torch.bool)
        constraint_coordinate_mask = torch.zeros(len(sites), dtype=torch.bool)
        output_atom_by_source = source_atom_map
        for condition in condition_values:
            if not isinstance(condition, ODesignReferenceCoordinates):
                continue
            for site in condition.sites:
                condition_site = output_atom_by_source.get(site)
                if condition_site is None or condition_site not in site_to_atom:
                    raise ValueError(f"ODesign atom layout does not contain reference site {site!r}")
                atom_index = site_to_atom[condition_site]
                constraint_coordinate_mask[atom_index] = True
                if condition.intent == "hard":
                    fixed_coordinate_mask[atom_index] = True
                elif condition.intent != "soft":
                    raise ValueError(f"unknown ODesignReferenceCoordinates intent: {condition.intent!r}")
        if bool((constraint_coordinate_mask & ~observed_mask).any()):
            raise ValueError("ODesignReferenceCoordinates must refer to observed input coordinates")
        # ``receptor_mode="rigid"`` realizes the ``diffusion`` condition by holding
        # every existing (non-generated) receptor heavy atom rigid — it is marked as a hard fixed
        # coordinate that the EDM sampler pins (model.sample reads ``fixed_coordinate_mask``).
        # This is additive over the per-atom ODesignReferenceCoordinates route above and only fixes atoms
        # that are actually observed (a receptor without coordinates cannot be pinned). "flex"
        # regenerates the receptor and leaves this untouched, so only explicit hard
        # ODesignReferenceCoordinates pin atoms.
        mode = self.config.receptor_mode
        if mode == "rigid":
            for atom_index, site in enumerate(sites):
                is_generated = (site.instance_id, site.component_index) in generated_residues
                if not is_generated and bool(observed_mask[atom_index]):
                    fixed_coordinate_mask[atom_index] = True
        elif mode != "flex":
            raise ValueError(f"unknown ODesign receptor_mode: {mode!r}")

        msa_features, msa_mask, profile, deletion_mean = _msa_features(
            msa_value,
            system,
            token_residues,
            token_entities,
            generated_residues,
        )
        token_features = torch.zeros(len(token_sites), 453, dtype=torch.float32)
        for token_index, (residue, entity) in enumerate(zip(token_residues, token_entities)):
            if residue is None:
                residue_type = "-L"
            elif residue in generated_residues:
                residue_type = "-P" if cast(PolymerSequence, entity).polymer_type == "protein" else "-N"
            else:
                sequence = cast(PolymerSequence, system.entities[system.instances[residue[0]]])
                residue_type = sequence.components[residue[1]].component_id
            residue_type_index = _RESIDUE_TYPE_INDEX.get(residue_type, 20)
            token_features[token_index, 384 + residue_type_index] = 1.0
            token_features[token_index, 419 : 419 + 32] = profile[token_index]
            token_features[token_index, 451] = deletion_mean[token_index]
            token_features[token_index, 452] = float(hotspot[token_index])

        atom_to_token = torch.tensor(
            [site_to_token[site] for site in sites],
            dtype=torch.int64,
        )
        atom_features = _atom_features(system, sites, reference_mask)
        token_bonds = _build_token_bonds(
            system,
            token_sites,
            site_to_token,
            excluded=cyclic_links,
        )
        constraint_distogram = _constraint_distogram(
            system,
            sites,
            site_to_token,
            observed_positions,
            constraint_coordinate_mask,
        )
        chain_index, entity_index, symmetry_index, residue_index = _build_token_indices(
            system,
            token_sites,
            token_residues,
            token_entities,
        )
        return ODesignInput(
            token_mask=torch.ones(1, len(token_sites), dtype=torch.bool, device=self.device),
            residue_index=residue_index.unsqueeze(0).to(self.device),
            chain_index=chain_index.unsqueeze(0).to(self.device),
            entity_index=entity_index.unsqueeze(0).to(self.device),
            symmetry_index=symmetry_index.unsqueeze(0).to(self.device),
            token_index=torch.arange(len(token_sites), dtype=torch.int64, device=self.device).unsqueeze(0),
            cyclic_period=_build_cyclic_period(token_sites, cyclic_instances).unsqueeze(0).to(self.device),
            token_features=token_features.unsqueeze(0).to(device=self.device, dtype=self.dtype),
            token_bonds=token_bonds.unsqueeze(0).to(device=self.device, dtype=self.dtype),
            atom_features=atom_features.unsqueeze(0).to(device=self.device, dtype=self.dtype),
            atom_to_token=atom_to_token.unsqueeze(0).to(self.device),
            atom_mask=torch.ones(1, len(sites), dtype=torch.bool, device=self.device),
            reference_positions=reference_positions.unsqueeze(0).to(device=self.device, dtype=self.dtype),
            reference_space_id=reference_space_id.unsqueeze(0).to(self.device),
            reference_mask=reference_mask.unsqueeze(0).to(self.device),
            reference_charges=torch.tensor(
                [system.atom(site).formal_charge for site in sites],
                dtype=self.dtype,
                device=self.device,
            )
            .unsqueeze(0)
            .unsqueeze(-1),
            observed_positions=observed_positions.unsqueeze(0).to(device=self.device, dtype=self.dtype),
            observed_mask=observed_mask.unsqueeze(0).to(self.device),
            fixed_coordinate_mask=fixed_coordinate_mask.unsqueeze(0).to(self.device),
            constraint_distogram=constraint_distogram.unsqueeze(0).to(device=self.device, dtype=self.dtype),
            msa_features=msa_features.unsqueeze(0).to(device=self.device, dtype=self.dtype),
            msa_mask=msa_mask.unsqueeze(0).to(self.device),
            layout=ODesignLayout(
                system=system,
                atom_sites=sites,
                placement=placement,
                conditions=condition_values,
                placement_map=placement_map,
                instance_of=instance_of,
                source_residue_map=source_residue_map,
                source_atom_map=source_atom_map,
            ),
            batch_size=[1],
        )


class ODesignLigandProcessor(Processor):
    r"""
    Lower canonical context and provisional ligand slots for ODesign.

    Generated slots are tensor identities only. They are never inserted into the
    context [`MolecularSystem`][] or represented by [`AtomSite`][].

    Args:
        config: Ligand-artifact input-processing options.
        device: Device for produced tensors.
        dtype: Floating-point dtype for produced tensors.
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

    def realize_atoms(
        self,
        input: ODesignLigandInput,
        *,
        generator: torch.Generator | None = None,
    ) -> ODesignLigandInput:
        r"""
        Sample the generated-atom range and retain its leading stable slots.
        """

        generated_atoms = input.layout.generated_atoms
        if generated_atoms.min_atoms == generated_atoms.max_atoms:
            return input
        num_atoms = int(
            torch.randint(
                generated_atoms.min_atoms,
                generated_atoms.max_atoms + 1,
                (),
                device=input.reference_positions.device,
                generator=generator,
            )
        )
        return _realize_ligand_atoms(input, num_atoms)

    def __call__(
        self,
        input: MolecularSystem | Structure | None,
        *,
        generated_atoms: ODesignGeneratedAtoms,
        msa: MSA | MSACollection | None = None,
        conditions: Sequence[Hotspot | ODesignReferenceCoordinates] = (),
    ) -> ODesignLigandInput:
        r"""
        Build one ligand-backbone input without fabricating generated chemistry.

        Args:
            input: Optional canonical context system or structure.
            generated_atoms: Caller-owned provisional atom range.
            msa: Optional canonical MSA evidence for polymer context.
            conditions: Context hotspots and reference-coordinate conditions.

        Returns:
            Tensor input whose context axes precede stable generated-atom slots.
        """

        return self._lower(self._plan(input, generated_atoms, conditions), msa)

    def _plan(
        self,
        input: MolecularSystem | Structure | None,
        generated_atoms: ODesignGeneratedAtoms,
        conditions: Sequence[Hotspot | ODesignReferenceCoordinates],
    ) -> _ODesignLigandPlan:
        if generated_atoms.min_atoms < 0 or generated_atoms.max_atoms < generated_atoms.min_atoms:
            raise ValueError("generated atom range must satisfy 0 <= min_atoms <= max_atoms")

        condition_values = tuple(conditions)
        for condition in condition_values:
            if not isinstance(condition, (Hotspot, ODesignReferenceCoordinates)):
                raise TypeError(f"ODesign ligand generation cannot consume {type(condition).__name__}")

        structure: Structure | None
        system: MolecularSystem | None
        if isinstance(input, Structure):
            structure = input
            system = input.system
        else:
            structure = None
            system = input
        sites = () if system is None else _sites_with_reference_coordinates(system)
        source_residue_map = (
            {}
            if system is None
            else {
                (instance_id, component_index): (instance_id, component_index)
                for instance_id, entity_id in system.instances.items()
                if isinstance(entity := system.entities[entity_id], PolymerSequence)
                for component_index in range(len(entity))
            }
        )
        source_atom_map = {site: site for site in sites}

        for condition in condition_values:
            if isinstance(condition, ODesignReferenceCoordinates):
                if any(site not in source_atom_map for site in condition.sites):
                    raise ValueError("ligand reference-coordinate conditions require canonical context atoms")
                continue
            for reference in condition.residues:
                if isinstance(reference, AtomSite):
                    if reference not in source_atom_map:
                        raise ValueError("ligand hotspots require canonical context atoms")
                    continue
                if not isinstance(reference, tuple) or system is None:
                    raise ValueError("ligand hotspots cannot refer to provisional generated atoms")
                entity = system.entities[system.instances[reference[0]]]
                if not isinstance(entity, PolymerSequence) or entity.polymer_type != "protein":
                    raise ValueError("non-protein ligand hotspots require exact atom references")

        return _ODesignLigandPlan(
            context_system=system,
            source_structure=structure,
            context_atom_sites=sites,
            generated_atoms=generated_atoms,
            conditions=condition_values,
            source_residue_map=source_residue_map,
            source_atom_map=source_atom_map,
        )

    def _lower(
        self,
        plan: _ODesignLigandPlan,
        msa: MSA | MSACollection | None,
    ) -> ODesignLigandInput:
        system = plan.context_system
        structure = plan.source_structure
        context_sites = plan.context_atom_sites
        num_slots = plan.generated_atoms.max_atoms
        slots = tuple(
            ODesignProvisionalAtomSlot(plan.generated_atoms.request_id, slot_index) for slot_index in range(num_slots)
        )
        if not context_sites and not slots:
            raise ValueError("ODesign requires at least one context atom or generated atom slot")

        if system is None:
            context_token_sites: tuple[tuple[AtomSite, ...], ...] = ()
            token_residues: tuple[tuple[str, int] | None, ...] = ()
            token_entities: tuple[PolymerSequence | Molecule, ...] = ()
        else:
            context_token_sites, token_residues, token_entities = _build_tokens(system, context_sites)
        num_context_tokens = len(context_token_sites)
        num_tokens = num_context_tokens + num_slots
        site_to_token = {site: token_index for token_index, group in enumerate(context_token_sites) for site in group}
        site_to_atom = {site: index for index, site in enumerate(context_sites)}

        observed_positions = torch.zeros(len(context_sites) + num_slots, 3, dtype=torch.float32)
        observed_mask = torch.zeros(len(context_sites) + num_slots, dtype=torch.bool)
        if structure is not None:
            observed = {site: index for index, site in enumerate(structure.sites)}
            for atom_index, site in enumerate(context_sites):
                source_index = observed.get(site)
                if source_index is None or not bool(structure.coordinate_mask[source_index]):
                    continue
                observed_positions[atom_index] = torch.from_numpy(structure.coordinates[source_index].copy())
                observed_mask[atom_index] = True

        reference_positions = torch.zeros_like(observed_positions)
        reference_spaces: dict[tuple[str, int], int] = {}
        reference_space_id = []
        for atom_index, site in enumerate(context_sites):
            component = cast(MolecularSystem, system).component((site.instance_id, site.component_index))
            coordinates = component.reference_coordinates
            atoms = component.atoms
            if not coordinates:
                raise ValueError(f"ODesign requires reference coordinates for component {component.component_id!r}")
            coordinate = coordinates[next(index for index, atom in enumerate(atoms) if atom.atom_id == site.atom_id)]
            if coordinate is None:
                raise ValueError(
                    f"ODesign requires a reference coordinate for component "
                    f"{component.component_id!r} atom {site.atom_id!r}"
                )
            reference_positions[atom_index] = torch.tensor(coordinate)
            reference_space = (site.instance_id, site.component_index)
            if reference_space not in reference_spaces:
                reference_spaces[reference_space] = len(reference_spaces)
            reference_space_id.append(reference_spaces[reference_space])
        context_reference_positions = reference_positions[: len(context_sites)]
        context_reference_space_id = torch.tensor(reference_space_id, dtype=torch.int64)
        for space_id in range(len(reference_spaces)):
            space_mask = context_reference_space_id == space_id
            context_reference_positions[space_mask] -= context_reference_positions[space_mask].mean(dim=0)
        reference_space_id.extend([len(reference_spaces)] * num_slots)
        reference_space_id_tensor = torch.tensor(reference_space_id, dtype=torch.int64)
        reference_mask = torch.ones(len(context_sites) + num_slots, dtype=torch.bool)

        hotspot = (
            _hotspot_tokens(
                plan.conditions,
                {},
                {},
                plan.source_residue_map,
                plan.source_atom_map,
                site_to_token,
            )
            if context_token_sites
            else torch.zeros(0, dtype=torch.bool)
        )
        hotspot = torch.cat([hotspot, torch.zeros(num_slots, dtype=torch.bool)])

        fixed_coordinate_mask = torch.zeros(len(context_sites) + num_slots, dtype=torch.bool)
        constraint_coordinate_mask = torch.zeros(len(context_sites) + num_slots, dtype=torch.bool)
        for condition in plan.conditions:
            if not isinstance(condition, ODesignReferenceCoordinates):
                continue
            for source_site in condition.sites:
                site = plan.source_atom_map[source_site]
                atom_index = site_to_atom[site]
                constraint_coordinate_mask[atom_index] = True
                if condition.intent == "hard":
                    fixed_coordinate_mask[atom_index] = True
                elif condition.intent != "soft":
                    raise ValueError(f"unknown ODesignReferenceCoordinates intent: {condition.intent!r}")
        if bool((constraint_coordinate_mask & ~observed_mask).any()):
            raise ValueError("ODesignReferenceCoordinates must refer to observed input coordinates")
        if self.config.receptor_mode == "rigid":
            fixed_coordinate_mask[: len(context_sites)] |= observed_mask[: len(context_sites)]
        elif self.config.receptor_mode != "flex":
            raise ValueError(f"unknown ODesign ligand receptor_mode: {self.config.receptor_mode!r}")

        if system is None:
            if msa is not None:
                raise ValueError("MSA evidence requires canonical polymer context")
            msa_value = None
            context_msa_features = torch.zeros(1, 0, 34)
            context_msa_mask = torch.ones(1, 0, dtype=torch.bool)
            context_profile = torch.zeros(0, 32)
            context_deletion_mean = torch.zeros(0)
        else:
            msa_value = _validate_msa(msa, system)
            (
                context_msa_features,
                context_msa_mask,
                context_profile,
                context_deletion_mean,
            ) = _msa_features(
                msa_value,
                system,
                token_residues,
                token_entities,
                set(),
            )
        msa_depth = context_msa_features.shape[0]
        slot_msa_features = torch.zeros(msa_depth, num_slots, 34)
        slot_msa_features[..., 31] = 1.0
        msa_features = torch.cat([context_msa_features, slot_msa_features], dim=1)
        msa_mask = torch.cat(
            [context_msa_mask, torch.ones(msa_depth, num_slots, dtype=torch.bool)],
            dim=1,
        )
        slot_profile = torch.zeros(num_slots, 32)
        slot_profile[:, 31] = 1.0
        profile = torch.cat([context_profile, slot_profile])
        deletion_mean = torch.cat([context_deletion_mean, torch.zeros(num_slots)])

        token_features = torch.zeros(num_tokens, 453, dtype=torch.float32)
        for token_index, (residue, entity) in enumerate(zip(token_residues, token_entities)):
            component_id = (
                "-L" if residue is None else cast(PolymerSequence, entity).components[residue[1]].component_id
            )
            token_features[token_index, 384 + _RESIDUE_TYPE_INDEX.get(component_id, 20)] = 1.0
        token_features[num_context_tokens:, 384 + _RESIDUE_TYPE_INDEX["-L"]] = 1.0
        token_features[:, 419 : 419 + 32] = profile
        token_features[:, 451] = deletion_mean
        token_features[:, 452] = hotspot.float()

        context_atom_features = (
            _atom_features(system, context_sites, torch.ones(len(context_sites), dtype=torch.bool))
            if system is not None
            else torch.zeros(0, 386)
        )
        atom_features = torch.cat([context_atom_features, _ligand_slot_atom_features(num_slots)])
        atom_to_token = torch.tensor(
            [site_to_token[site] for site in context_sites] + list(range(num_context_tokens, num_tokens)),
            dtype=torch.int64,
        )

        token_bonds = torch.zeros(num_tokens, num_tokens, dtype=torch.float32)
        if system is not None:
            token_bonds[:num_context_tokens, :num_context_tokens] = _build_token_bonds(
                system,
                context_token_sites,
                site_to_token,
                excluded=frozenset(),
            )
        constraint_distogram = torch.zeros(num_tokens, num_tokens, 1)
        if system is not None and context_token_sites:
            constraint_distogram[:num_context_tokens, :num_context_tokens] = _constraint_distogram(
                system,
                context_sites,
                site_to_token,
                observed_positions[: len(context_sites)],
                constraint_coordinate_mask[: len(context_sites)],
            )

        if system is None:
            chain_index = entity_index = symmetry_index = residue_index = torch.zeros(0, dtype=torch.int64)
            generated_chain_index = 0
            generated_entity_index = 0
        else:
            chain_index, entity_index, symmetry_index, residue_index = _build_token_indices(
                system,
                context_token_sites,
                token_residues,
                token_entities,
            )
            generated_chain_index = len(system.instances)
            generated_entity_index = len(dict.fromkeys(system.instances.values()))
        chain_index = torch.cat([chain_index, torch.full((num_slots,), generated_chain_index, dtype=torch.int64)])
        entity_index = torch.cat([entity_index, torch.full((num_slots,), generated_entity_index, dtype=torch.int64)])
        symmetry_index = torch.cat([symmetry_index, torch.zeros(num_slots, dtype=torch.int64)])
        residue_index = torch.cat([residue_index, torch.zeros(num_slots, dtype=torch.int64)])

        context_charges = [system.atom(site).formal_charge for site in context_sites] if system is not None else []
        return ODesignLigandInput(
            token_mask=torch.ones(1, num_tokens, dtype=torch.bool, device=self.device),
            residue_index=residue_index.unsqueeze(0).to(self.device),
            chain_index=chain_index.unsqueeze(0).to(self.device),
            entity_index=entity_index.unsqueeze(0).to(self.device),
            symmetry_index=symmetry_index.unsqueeze(0).to(self.device),
            token_index=torch.arange(num_tokens, dtype=torch.int64, device=self.device).unsqueeze(0),
            cyclic_period=torch.zeros(1, num_tokens, dtype=torch.int64, device=self.device),
            token_features=token_features.unsqueeze(0).to(device=self.device, dtype=self.dtype),
            token_bonds=token_bonds.unsqueeze(0).to(device=self.device, dtype=self.dtype),
            atom_features=atom_features.unsqueeze(0).to(device=self.device, dtype=self.dtype),
            atom_to_token=atom_to_token.unsqueeze(0).to(self.device),
            atom_mask=torch.ones(1, len(context_sites) + num_slots, dtype=torch.bool, device=self.device),
            reference_positions=reference_positions.unsqueeze(0).to(device=self.device, dtype=self.dtype),
            reference_space_id=reference_space_id_tensor.unsqueeze(0).to(self.device),
            reference_mask=reference_mask.unsqueeze(0).to(self.device),
            reference_charges=torch.tensor(
                [*context_charges, *([0] * num_slots)],
                dtype=self.dtype,
                device=self.device,
            )
            .unsqueeze(0)
            .unsqueeze(-1),
            observed_positions=observed_positions.unsqueeze(0).to(device=self.device, dtype=self.dtype),
            observed_mask=observed_mask.unsqueeze(0).to(self.device),
            fixed_coordinate_mask=fixed_coordinate_mask.unsqueeze(0).to(self.device),
            constraint_distogram=constraint_distogram.unsqueeze(0).to(device=self.device, dtype=self.dtype),
            msa_features=msa_features.unsqueeze(0).to(device=self.device, dtype=self.dtype),
            msa_mask=msa_mask.unsqueeze(0).to(self.device),
            layout=ODesignLigandLayout(
                context_system=system,
                context_token_sites=context_token_sites,
                context_atom_sites=context_sites,
                generated_atoms=plan.generated_atoms,
                slots=slots,
                conditions=plan.conditions,
                source_atom_map=plan.source_atom_map,
            ),
            batch_size=[1],
        )


def from_design_mask(
    residues: Sequence[ODesignResidueSpec],
    *,
    group_of: Callable[[int], str],
    placement_id_of: Callable[[int], str],
) -> PlacementSpec:
    r"""
    Convert ODesign residue specifications to placements.

    Args:
        residues: Designed or fixed residue specifications in output order.
        group_of: Function returning the placement group for each residue index.
        placement_id_of: Function returning the placement identifier for each residue
            index. Each identifier must occupy one contiguous run.

    Returns:
        Placements containing exact-length generated polymer segments for designed
        runs and existing spans for fixed runs.

    Raises:
        ValueError: If a designed residue is not a supported polymer residue, a
            placement identifier is reused, or the resulting placements are invalid.
    """

    placement: PlacementSpec = {}
    seen: set[str] = set()

    current_group: str | None = None
    current_placement_id: str | None = None
    current_polymer_type: PolymerType | None = None
    current_instance_id: str | None = None
    run_start_source: int = 0
    run_next_source: int = 0
    run_length: int = 0

    def flush() -> None:
        if current_placement_id is None:
            return
        assert current_group is not None
        if current_polymer_type is not None:
            value: GeneratedSegment | ExistingSpan = GeneratedSegment(current_polymer_type, run_length, run_length)
        else:
            assert current_instance_id is not None
            value = ExistingSpan(current_instance_id, run_start_source, run_next_source)
        placement.setdefault(current_group, {})[current_placement_id] = value

    for index, residue in enumerate(residues):
        group = group_of(index)
        placement_id = placement_id_of(index)
        if isinstance(residue, ODesignDesignedResidue):
            if residue.polymer_type not in _CORE_POLYMER_TYPES:
                raise ValueError(
                    f"designed residue {index} has non-core polymer_type "
                    f"{residue.polymer_type!r}; a de-novo non-polymer (ligand num_atoms) has no "
                    f"PlacementValue (GAP-1) and cannot be projected onto the placement core"
                )
            extends = (
                group == current_group
                and placement_id == current_placement_id
                and current_polymer_type == residue.polymer_type
            )
            if extends:
                run_length += 1
                continue
            flush()
            if placement_id in seen:
                raise ValueError(
                    f"placement_id {placement_id!r} reused by a non-contiguous run at residue "
                    f"{index}; a placement_id must be globally unique across the PlacementSpec"
                )
            seen.add(placement_id)
            current_group = group
            current_placement_id = placement_id
            current_polymer_type = residue.polymer_type
            current_instance_id = None
            run_length = 1
        else:
            assert isinstance(residue, ODesignFixedResidue)
            extends = (
                group == current_group
                and placement_id == current_placement_id
                and current_polymer_type is None
                and current_instance_id == residue.instance_id
                and run_next_source == residue.source_index
            )
            if extends:
                run_next_source = residue.source_index + 1
                continue
            flush()
            if placement_id in seen:
                raise ValueError(
                    f"placement_id {placement_id!r} reused by a non-contiguous run at residue "
                    f"{index}; a placement_id must be globally unique across the PlacementSpec"
                )
            seen.add(placement_id)
            current_group = group
            current_placement_id = placement_id
            current_polymer_type = None
            current_instance_id = residue.instance_id
            run_start_source = residue.source_index
            run_next_source = residue.source_index + 1

    flush()
    # Enforce globally unique placement IDs across the PlacementSpec.
    validate_placement(placement)
    return placement


def to_design_mask(
    placement: PlacementSpec,
    placement_map: PlacementMap,
) -> tuple[list[bool], list[str]]:
    r"""
    Convert realized ODesign placements to a residue design mask.

    Args:
        placement: Declared generated segments and existing spans.
        placement_map: Realized residue references for each placement.

    Returns:
        A pair containing the designed-status flag and declaration-group label for
        each realized residue, in placement declaration order.
    """

    design_mask: list[bool] = []
    groups: list[str] = []
    for group, chain in placement.items():
        for placement_id, value in chain.items():
            if isinstance(value, GeneratedSegment):
                length = len(placement_map[group][placement_id])
                designed = True
            else:
                length = value.stop - value.start
                designed = False
            design_mask.extend([designed] * length)
            groups.extend([group] * length)
    return design_mask, groups


def _msa_symbol_token(symbol: str, polymer_type: str) -> int:
    r"""
    Lower one uppercase MSA symbol (or ``-`` gap) to an ODesign residue token.
    """

    if symbol == "-":
        return _MSA_GAP
    return _MSA_SYMBOL_TOKENS[polymer_type].get(symbol, _MSA_UNKNOWN)


def _sites_with_reference_coordinates(system: MolecularSystem) -> tuple[AtomSite, ...]:
    r"""
    Select model atoms while leaving canonical-only terminal atoms unresolved.
    """

    sites = []
    for site, atom in system.iter_atoms():
        if atom.element == "H":
            continue
        entity = system.entities[system.instances[site.instance_id]]
        if isinstance(entity, PolymerSequence):
            component = entity.components[site.component_index]
            atom_index = next(index for index, member in enumerate(component.atoms) if member.atom_id == site.atom_id)
            if component.reference_coordinates and component.reference_coordinates[atom_index] is None:
                continue
        sites.append(site)
    return tuple(sites)


def _materialize_placement(
    placement: ODesignPlacements,
    source_system: MolecularSystem | None,
) -> tuple[
    MolecularSystem,
    tuple[AtomSite, ...],
    PlacementMap,
    dict[str, str],
    dict[tuple[str, int], tuple[str, int]],
    dict[AtomSite, AtomSite],
]:
    r"""
    Materialize one exact mixed PlacementSpec and its source-to-output identity maps.

    Returns the realized ``placement_map`` (the output surface for generated
    placements: ``declaration_group -> placement_id -> ordered ComponentReference tuple``)
    and its ``instance_of`` index (``placement_id -> declaration_group``, derived once from the
    PlacementSpec), alongside the source-residue and source-atom maps.
    """

    molecule_entities: dict[str, Molecule] = (
        {entity_id: entity for entity_id, entity in source_system.entities.items() if isinstance(entity, Molecule)}
        if source_system is not None
        else {}
    )
    molecule_instances: dict[str, str] = (
        {
            instance_id: entity_id
            for instance_id, entity_id in source_system.instances.items()
            if isinstance(source_system.entities[entity_id], Molecule)
        }
        if source_system is not None
        else {}
    )
    used_entity_ids = set(molecule_entities)
    used_instance_ids = set(molecule_instances)
    polymer_entities: dict[str, PolymerSequence] = {}
    polymer_instances: dict[str, str] = {}
    placement_map: PlacementMap = {}
    source_residue_map: dict[tuple[str, int], tuple[str, int]] = {}
    generated_residues: set[tuple[str, int]] = set()
    generated_component_ids: set[str] = set()
    new_entities_by_sequence: dict[tuple[PolymerType, tuple[ChemicalComponent, ...]], tuple[str, PolymerSequence]] = {}

    # The shared declaration-order allocator mints the realized instance IDs:
    # canonical ``chain_{n}`` (uniform and 0-based) once per realized output
    # chain that has >=1 occurrence. A declaration group whose parts all realize to zero length
    # mints NO chain and consumes NO index — it is recorded as () and skipped.
    instance_ids = InstanceIdAllocator()
    for chain_index, (group, chain) in enumerate(placement.placements.items()):
        if all(
            part.max_length == 0 if isinstance(part, GeneratedSegment) else part.stop == part.start
            for part in chain.values()
        ):
            for placement_id, part in chain.items():
                if isinstance(part, GeneratedSegment):
                    placement_map.setdefault(group, {})[placement_id] = ()
            continue
        instance_id = instance_ids.allocate()
        while instance_id in used_instance_ids:
            instance_id += "_"
        used_instance_ids.add(instance_id)

        components: list[ChemicalComponent] = []
        segment_offsets: list[tuple[str, GeneratedSegment, int, int]] = []
        polymer_type: str | None = None
        for placement_id, part in chain.items():
            if isinstance(part, GeneratedSegment):
                part_polymer_type = part.polymer_type
                start = len(components)
                length = part.max_length
                if part_polymer_type not in _GENERATED_COMPONENT_ID:
                    raise NotImplementedError(
                        "ODesign cannot realize one dna_rna_hybrid segment without per-residue polymer "
                        "identity; use separate DNA and RNA generated segments"
                    )
                component_id = _GENERATED_COMPONENT_ID[part_polymer_type]
                components.extend([get_component(component_id)] * length)
                generated_component_ids.add(component_id)
                segment_offsets.append((placement_id, part, start, length))
            else:
                part = cast(ExistingSpan, part)
                source_system = cast(MolecularSystem, source_system)
                source_entity = cast(PolymerSequence, source_system.entities[source_system.instances[part.instance_id]])
                part_polymer_type = source_entity.polymer_type
                for source_index in range(part.start, part.stop):
                    output_index = len(components)
                    components.append(source_entity.components[source_index])
                    source_residue_map[(part.instance_id, source_index)] = (instance_id, output_index)

            if polymer_type is not None and polymer_type != part_polymer_type:
                raise ValueError("one ODesign output chain cannot mix polymer types")
            polymer_type = part_polymer_type

        polymer_type = cast(PolymerType, polymer_type)

        reused_entity_id: str | None = None
        chain_parts = tuple(chain.values())
        if len(chain_parts) == 1 and isinstance(chain_parts[0], ExistingSpan) and source_system is not None:
            span = chain_parts[0]
            source_entity_id = source_system.instances[span.instance_id]
            candidate = cast(PolymerSequence, source_system.entities[source_entity_id])
            if span.start == 0 and span.stop == len(candidate):
                reused_entity_id = source_entity_id
                sequence = candidate
        if reused_entity_id is not None:
            entity_id = reused_entity_id
        else:
            signature = (polymer_type, tuple(components))
            existing_entity = new_entities_by_sequence.get(signature)
            if existing_entity is None:
                entity_id = f"odesign_entity_{chain_index + 1}"
                while entity_id in used_entity_ids:
                    entity_id += "_"
                sequence = PolymerSequence(polymer_type, tuple(components))
                new_entities_by_sequence[signature] = (entity_id, sequence)
            else:
                entity_id, sequence = existing_entity
        if entity_id not in used_entity_ids:
            polymer_entities[entity_id] = sequence
            used_entity_ids.add(entity_id)
        polymer_instances[instance_id] = entity_id

        for placement_id, _segment, start, length in segment_offsets:
            occurrences = tuple((instance_id, start + offset) for offset in range(length))
            generated_residues.update(occurrences)
            placement_map.setdefault(group, {})[placement_id] = occurrences

    source_components = (
        {
            component.component_id: component
            for entity in source_system.entities.values()
            for component in entity.components
        }
        if source_system is not None
        else {}
    )
    for component_id in generated_component_ids:
        source_component = source_components.get(component_id)
        if source_component is not None and source_component != get_component(component_id):
            raise ValueError(f"generated {component_id} conflicts with the input system's component graph")

    preliminary = MolecularSystem(
        {**polymer_entities, **molecule_entities},
        {**polymer_instances, **molecule_instances},
    )
    complete_sites = _sites_with_reference_coordinates(preliminary)
    source_residue_by_output = {output: source for source, output in source_residue_map.items()}
    source_heavy_sites = (
        [site for site, atom in source_system.iter_atoms() if atom.element != "H"] if source_system is not None else []
    )
    source_sites = set(source_heavy_sites)
    sites = []
    for site in complete_sites:
        residue = (site.instance_id, site.component_index)
        entity = preliminary.entities[preliminary.instances[site.instance_id]]
        if residue in generated_residues:
            if cast(PolymerSequence, entity).polymer_type == "protein" and site.atom_id not in {"N", "CA", "C", "O"}:
                continue
        elif isinstance(entity, PolymerSequence):
            source_residue = source_residue_by_output[residue]
            if AtomSite(source_residue[0], source_residue[1], site.atom_id) not in source_sites:
                continue
        sites.append(site)
    site_tuple = tuple(sites)
    output_sites = set(site_tuple)
    source_atom_map: dict[AtomSite, AtomSite] = {}
    if source_system is not None:
        source_sites_by_residue: dict[tuple[str, int], list[AtomSite]] = {}
        for source_site in source_heavy_sites:
            source_sites_by_residue.setdefault((source_site.instance_id, source_site.component_index), []).append(
                source_site
            )
        for source_residue, output_residue in source_residue_map.items():
            for source_site in source_sites_by_residue.get(source_residue, ()):
                output_site = AtomSite(output_residue[0], output_residue[1], source_site.atom_id)
                if output_site in output_sites:
                    source_atom_map[source_site] = output_site
        for source_site in source_heavy_sites:
            if source_site.instance_id in molecule_instances and source_site in output_sites:
                source_atom_map[source_site] = source_site

    mapped_atoms = source_atom_map
    mapped_bonds = []
    if source_system is not None:
        for bond in source_system.bonds:
            site_a = mapped_atoms.get(bond.site_a)
            site_b = mapped_atoms.get(bond.site_b)
            if (site_a is None) != (site_b is None):
                raise ValueError(f"ODesign placement cuts exactly one endpoint of source bond {bond!r}")
            if site_a is None or site_b is None:
                continue
            stereo_sites = (
                None
                if bond.stereo_sites is None
                else (mapped_atoms[bond.stereo_sites[0]], mapped_atoms[bond.stereo_sites[1]])
            )
            mapped_bonds.append(
                SystemBond(
                    site_a,
                    site_b,
                    order=bond.order,
                    aromatic=bond.aromatic,
                    stereo=bond.stereo,
                    stereo_sites=stereo_sites,
                    origin=bond.origin,
                )
            )
    system = MolecularSystem(
        preliminary.entities,
        preliminary.instances,
        frozenset(mapped_bonds),
    )
    return (
        system,
        site_tuple,
        placement_map,
        _placement_instance_of(placement.placements),
        source_residue_map,
        source_atom_map,
    )


def _materialize_cyclic_links(
    system: MolecularSystem,
    conditions: tuple[Hotspot | SymbolicBond | ODesignReferenceCoordinates, ...],
    placement_map: PlacementMap,
    instance_of: dict[str, str],
    source_atom_map: dict[AtomSite, AtomSite],
) -> tuple[MolecularSystem, frozenset[str], frozenset[frozenset[AtomSite]]]:
    r"""
    Materialize requested head-to-tail polymer closures.
    """

    source_atoms = source_atom_map
    bonds = set(system.bonds)
    cyclic_instances = set()
    cyclic_links = set()
    for existing_bond in bonds:
        if existing_bond.site_a.instance_id != existing_bond.site_b.instance_id:
            continue
        instance_id = existing_bond.site_a.instance_id
        entity = system.entities[system.instances[instance_id]]
        if not isinstance(entity, PolymerSequence):
            continue
        first = entity.components[0].previous_attachment
        last = entity.components[-1].next_attachment
        if first is None or last is None:
            continue
        expected = frozenset(
            (
                AtomSite(instance_id, 0, first.atom_id),
                AtomSite(instance_id, len(entity) - 1, last.atom_id),
            )
        )
        if frozenset((existing_bond.site_a, existing_bond.site_b)) == expected:
            cyclic_instances.add(instance_id)
            cyclic_links.add(expected)
    for condition in conditions:
        if not isinstance(condition, SymbolicBond):
            continue
        # Endpoints ride on the SymbolicBond (site_a/site_b over the
        # promoted SymbolicBondSite union). An existing-atom endpoint (AtomSite) is remapped
        # source -> output through source_atom_map; a generated-atom endpoint (SymbolicAtomSite)
        # resolves its SymbolicResidueSite through the shared placement resolver.
        symbolic_bond = condition
        endpoints: list[AtomSite] = []
        for reference in (symbolic_bond.site_a, symbolic_bond.site_b):
            if isinstance(reference, AtomSite):
                site = source_atoms.get(reference)
                if site is None:
                    raise ValueError(f"ODesign atom layout does not contain covalent-link site {reference!r}")
            else:
                assert isinstance(reference, SymbolicAtomSite)
                residue = _resolve_site(reference.residue, placement_map, instance_of)
                site = AtomSite(residue[0], residue[1], reference.atom_id)
            system.atom(site)
            endpoints.append(site)

        site_a, site_b = endpoints
        if site_a.instance_id != site_b.instance_id:
            raise ValueError("ODesign supports only head-to-tail cyclic polymer links")
        instance_id = site_a.instance_id
        entity = system.entities[system.instances[instance_id]]
        if not isinstance(entity, PolymerSequence):
            raise ValueError("ODesign supports only head-to-tail cyclic polymer links")
        previous = entity.components[0].previous_attachment
        following = entity.components[-1].next_attachment
        if previous is None or following is None:
            raise ValueError("cyclic polymer endpoints require previous and next attachments")
        expected = frozenset(
            (
                AtomSite(instance_id, 0, previous.atom_id),
                AtomSite(instance_id, len(entity) - 1, following.atom_id),
            )
        )
        endpoint_set = frozenset((site_a, site_b))
        if endpoint_set != expected:
            raise ValueError("ODesign supports only head-to-tail cyclic polymer links")

        existing_bonds = tuple(
            candidate for candidate in bonds if frozenset((candidate.site_a, candidate.site_b)) == endpoint_set
        )
        requested_chemistry = (symbolic_bond.order, symbolic_bond.aromatic, symbolic_bond.stereo, None)
        for candidate in existing_bonds:
            if (candidate.order, candidate.aromatic, candidate.stereo, candidate.stereo_sites) != requested_chemistry:
                raise ValueError("conflicting ODesign cyclic bond already exists at the requested endpoints")
        if not existing_bonds:
            bonds.add(
                SystemBond(
                    site_a,
                    site_b,
                    order=symbolic_bond.order,
                    aromatic=symbolic_bond.aromatic,
                    stereo=symbolic_bond.stereo,
                    origin="user",
                )
            )
        cyclic_instances.add(instance_id)
        cyclic_links.add(endpoint_set)

    if not cyclic_instances:
        return system, frozenset(), frozenset()
    return (
        MolecularSystem(
            system.entities,
            system.instances,
            frozenset(bonds),
        ),
        frozenset(cyclic_instances),
        frozenset(cyclic_links),
    )


def _realize_ranges(input: ODesignInput, generator: torch.Generator | None) -> ODesignInput:
    placement = input.layout.placement
    if placement is None:
        return input
    ranged = {
        placement_id: part
        for chain in placement.placements.values()
        for placement_id, part in chain.items()
        if isinstance(part, GeneratedSegment) and part.min_length != part.max_length
    }
    lengths = {}
    chain_lengths = placement.chain_lengths or (None,) * len(placement.placements)
    for chain, chain_length in zip(placement.placements.values(), chain_lengths, strict=True):
        chain_ranges = [
            (placement_id, part)
            for placement_id, part in chain.items()
            if isinstance(part, GeneratedSegment) and placement_id in ranged
        ]
        if chain_length is None:
            for placement_id, segment in chain_ranges:
                lengths[placement_id] = int(
                    torch.randint(
                        segment.min_length,
                        segment.max_length + 1,
                        (),
                        device=input.reference_positions.device,
                        generator=generator,
                    )
                )
            continue

        # A requested chain length must fall inside the window realizable by this chain's
        # components: every segment at its minimum through every segment at its maximum. When the
        # chain carries no variable GeneratedSegment this window collapses to the fixed/Existing sum,
        # so an exact-match check falls out of the same bounds. Outside the window there is no length
        # assignment to realize the request, mirroring the rfdiffusion "cannot be realized" contract.
        minimum = sum(
            part.min_length if isinstance(part, GeneratedSegment) else part.stop - part.start
            for part in chain.values()
        )
        maximum = sum(
            part.max_length if isinstance(part, GeneratedSegment) else part.stop - part.start
            for part in chain.values()
        )
        if not minimum <= chain_length <= maximum:
            raise ValueError(
                f"ODesign placement chain length {chain_length} cannot be realized by its segments "
                f"(realizable range [{minimum}, {maximum}])"
            )

        for placement_id, segment in chain_ranges:
            lengths[placement_id] = segment.min_length
        remaining = chain_length - minimum
        for _ in range(remaining):
            available = [
                (placement_id, segment)
                for placement_id, segment in chain_ranges
                if lengths[placement_id] < segment.max_length
            ]
            selected = int(
                torch.randint(
                    len(available),
                    (),
                    device=input.reference_positions.device,
                    generator=generator,
                )
            )
            lengths[available[selected][0]] += 1
    if all(length == ranged[placement_id].max_length for placement_id, length in lengths.items()):
        return input

    polymer_instances = [
        instance_id
        for instance_id, entity_id in input.layout.system.instances.items()
        if isinstance(input.layout.system.entities[entity_id], PolymerSequence)
    ]
    residue_map: dict[tuple[str, int], tuple[str, int]] = {}
    placement_map: PlacementMap = {}
    realized_sequences: list[tuple[str, str, PolymerSequence]] = []
    sequence_by_signature: dict[
        tuple[PolymerType, tuple[ChemicalComponent, ...]],
        tuple[str, PolymerSequence],
    ] = {}
    used_entity_ids = {
        entity_id for entity_id, entity in input.layout.system.entities.items() if isinstance(entity, Molecule)
    }
    # Walk declaration groups in order, indexing the realized polymer instances by declaration
    # position among NON-EMPTY groups only: a zero-length declaration group mints no realized
    # chain (matching _materialize_placement), so it is recorded as () and skipped here too.
    chain_position = 0
    for group, chain in placement.placements.items():
        if all(
            part.max_length == 0 if isinstance(part, GeneratedSegment) else part.stop == part.start
            for part in chain.values()
        ):
            for placement_id, part in chain.items():
                if isinstance(part, GeneratedSegment):
                    placement_map.setdefault(group, {})[placement_id] = ()
            continue
        instance_id = polymer_instances[chain_position]
        chain_position += 1
        old_entity_id = input.layout.system.instances[instance_id]
        old_entity = cast(PolymerSequence, input.layout.system.entities[old_entity_id])
        old_index = 0
        new_index = 0
        components: list[ChemicalComponent] = []
        for placement_id, part in chain.items():
            old_length = part.max_length if isinstance(part, GeneratedSegment) else part.stop - part.start
            new_length = (
                lengths.get(placement_id, part.max_length) if isinstance(part, GeneratedSegment) else old_length
            )
            occurrences: list[tuple[str, int]] = []
            for offset in range(new_length):
                residue_map[(instance_id, old_index + offset)] = (instance_id, new_index + offset)
                components.append(old_entity.components[old_index + offset])
                if isinstance(part, GeneratedSegment):
                    occurrences.append((instance_id, new_index + offset))
            if isinstance(part, GeneratedSegment):
                placement_map.setdefault(group, {})[placement_id] = tuple(occurrences)
            old_index += old_length
            new_index += new_length

        signature = (old_entity.polymer_type, tuple(components))
        existing_entity = sequence_by_signature.get(signature)
        if existing_entity is None:
            entity_id = old_entity_id
            while entity_id in used_entity_ids:
                entity_id += "_"
            sequence = PolymerSequence(old_entity.polymer_type, tuple(components))
            sequence_by_signature[signature] = (entity_id, sequence)
            used_entity_ids.add(entity_id)
        else:
            entity_id, sequence = existing_entity
        realized_sequences.append((instance_id, entity_id, sequence))

    molecule_entities = {
        entity_id: entity for entity_id, entity in input.layout.system.entities.items() if isinstance(entity, Molecule)
    }
    molecule_instances = {
        instance_id: entity_id
        for instance_id, entity_id in input.layout.system.instances.items()
        if isinstance(input.layout.system.entities[entity_id], Molecule)
    }
    instances: dict[str, str] = {
        **{instance_id: entity_id for instance_id, entity_id, _ in realized_sequences},
        **molecule_instances,
    }
    entities: dict[str, PolymerSequence | Molecule] = {
        **{entity_id: sequence for _, entity_id, sequence in realized_sequences},
        **molecule_entities,
    }

    def map_site(site: AtomSite) -> AtomSite | None:
        if isinstance(input.layout.system.entities[input.layout.system.instances[site.instance_id]], Molecule):
            return site
        residue = residue_map.get((site.instance_id, site.component_index))
        return None if residue is None else AtomSite(residue[0], residue[1], site.atom_id)

    bonds = []
    for bond in input.layout.system.bonds:
        site_a = map_site(bond.site_a)
        site_b = map_site(bond.site_b)
        if site_a is None or site_b is None:
            continue
        stereo_sites = None
        if bond.stereo_sites is not None:
            stereo_a = map_site(bond.stereo_sites[0])
            stereo_b = map_site(bond.stereo_sites[1])
            if stereo_a is None or stereo_b is None:
                continue
            stereo_sites = (stereo_a, stereo_b)
        bonds.append(
            SystemBond(
                site_a,
                site_b,
                order=bond.order,
                aromatic=bond.aromatic,
                stereo=bond.stereo,
                stereo_sites=stereo_sites,
                origin=bond.origin,
            )
        )
    system = MolecularSystem(entities, instances, frozenset(bonds))
    source_residue_map = {
        source: residue_map[output]
        for source, output in input.layout.source_residue_map.items()
        if output in residue_map
    }
    source_atom_map = {
        source: mapped
        for source, output in input.layout.source_atom_map.items()
        if (mapped := map_site(output)) is not None
    }
    system, cyclic_instances, cyclic_links = _materialize_cyclic_links(
        system,
        input.layout.conditions,
        placement_map,
        input.layout.instance_of,
        source_atom_map,
    )

    old_to_new_site = {site: mapped for site in input.layout.atom_sites if (mapped := map_site(site)) is not None}
    sites = tuple(old_to_new_site.values())
    new_to_old_site = {new: old for old, new in old_to_new_site.items()}
    old_atom_index = {site: index for index, site in enumerate(input.layout.atom_sites)}
    atom_indices = torch.tensor(
        [old_atom_index[new_to_old_site[site]] for site in sites],
        dtype=torch.int64,
        device=input.atom_mask.device,
    )
    token_sites, token_residues, token_entities = _build_tokens(system, sites)
    old_token_indices = torch.tensor(
        [int(input.atom_to_token[0, old_atom_index[new_to_old_site[group[0]]]]) for group in token_sites],
        dtype=torch.int64,
        device=input.token_mask.device,
    )
    atom_to_token = torch.tensor(
        [token_index for token_index, group in enumerate(token_sites) for _ in group],
        dtype=torch.int64,
        device=input.atom_to_token.device,
    ).unsqueeze(0)
    site_to_token = {site: token_index for token_index, group in enumerate(token_sites) for site in group}
    chain_index, entity_index, symmetry_index, residue_index = (
        value.unsqueeze(0).to(input.residue_index.device)
        for value in _build_token_indices(system, token_sites, token_residues, token_entities)
    )
    selected_reference_spaces = input.reference_space_id.index_select(1, atom_indices)
    reference_space_map: dict[int, int] = {}
    reference_space_id: list[int] = []
    for value in selected_reference_spaces[0].tolist():
        if value not in reference_space_map:
            reference_space_map[value] = len(reference_space_map)
        reference_space_id.append(reference_space_map[value])
    reference_space_id = torch.tensor(
        reference_space_id,
        dtype=torch.int64,
        device=input.reference_space_id.device,
    ).unsqueeze(0)
    observed_positions = input.observed_positions.index_select(1, atom_indices)
    observed_mask = input.observed_mask.index_select(1, atom_indices)
    fixed_coordinate_mask = input.fixed_coordinate_mask.index_select(1, atom_indices)
    token_bonds = _build_token_bonds(
        system,
        token_sites,
        site_to_token,
        excluded=cyclic_links,
    ).to(input.token_bonds)
    constraint_distogram = input.constraint_distogram.index_select(1, old_token_indices).index_select(
        2,
        old_token_indices,
    )
    layout = replace(
        input.layout,
        system=system,
        atom_sites=sites,
        placement_map=placement_map,
        source_residue_map=source_residue_map,
        source_atom_map=source_atom_map,
    )
    return input.replace(
        token_mask=input.token_mask.index_select(1, old_token_indices),
        residue_index=residue_index,
        chain_index=chain_index,
        entity_index=entity_index,
        symmetry_index=symmetry_index,
        token_index=torch.arange(len(token_sites), device=input.token_index.device).unsqueeze(0),
        cyclic_period=_build_cyclic_period(token_sites, cyclic_instances).unsqueeze(0).to(input.cyclic_period.device),
        token_features=input.token_features.index_select(1, old_token_indices),
        token_bonds=token_bonds.unsqueeze(0),
        atom_features=input.atom_features.index_select(1, atom_indices),
        atom_to_token=atom_to_token,
        atom_mask=input.atom_mask.index_select(1, atom_indices),
        reference_positions=input.reference_positions.index_select(1, atom_indices),
        reference_space_id=reference_space_id,
        reference_mask=input.reference_mask.index_select(1, atom_indices),
        reference_charges=input.reference_charges.index_select(1, atom_indices),
        observed_positions=observed_positions,
        observed_mask=observed_mask,
        fixed_coordinate_mask=fixed_coordinate_mask,
        constraint_distogram=constraint_distogram,
        msa_features=input.msa_features.index_select(2, old_token_indices),
        msa_mask=input.msa_mask.index_select(2, old_token_indices),
        layout=layout,
    )


def _realize_ligand_atoms(
    input: ODesignLigandInput,
    num_atoms: int,
) -> ODesignLigandInput:
    num_context_atoms = len(input.layout.context_atom_sites)
    num_context_tokens = len(input.layout.context_token_sites)
    atom_count = num_context_atoms + num_atoms
    token_count = num_context_tokens + num_atoms
    generated_atoms = replace(
        input.layout.generated_atoms,
        min_atoms=num_atoms,
        max_atoms=num_atoms,
    )
    layout = replace(
        input.layout,
        generated_atoms=generated_atoms,
        slots=input.layout.slots[:num_atoms],
    )
    return input.replace(
        token_mask=input.token_mask[:, :token_count],
        residue_index=input.residue_index[:, :token_count],
        chain_index=input.chain_index[:, :token_count],
        entity_index=input.entity_index[:, :token_count],
        symmetry_index=input.symmetry_index[:, :token_count],
        token_index=input.token_index[:, :token_count],
        cyclic_period=input.cyclic_period[:, :token_count],
        token_features=input.token_features[:, :token_count],
        token_bonds=input.token_bonds[:, :token_count, :token_count],
        atom_features=input.atom_features[:, :atom_count],
        atom_to_token=input.atom_to_token[:, :atom_count],
        atom_mask=input.atom_mask[:, :atom_count],
        reference_positions=input.reference_positions[:, :atom_count],
        reference_space_id=input.reference_space_id[:, :atom_count],
        reference_mask=input.reference_mask[:, :atom_count],
        reference_charges=input.reference_charges[:, :atom_count],
        observed_positions=input.observed_positions[:, :atom_count],
        observed_mask=input.observed_mask[:, :atom_count],
        fixed_coordinate_mask=input.fixed_coordinate_mask[:, :atom_count],
        constraint_distogram=input.constraint_distogram[:, :token_count, :token_count],
        msa_features=input.msa_features[:, :, :token_count],
        msa_mask=input.msa_mask[:, :, :token_count],
        layout=layout,
    )


def _build_tokens(
    system: MolecularSystem,
    sites: tuple[AtomSite, ...],
) -> tuple[
    tuple[tuple[AtomSite, ...], ...],
    tuple[tuple[str, int] | None, ...],
    tuple[PolymerSequence | Molecule, ...],
]:
    r"""
    Build the ODesign token groups and their molecular identities.
    """

    sites_by_residue: dict[tuple[str, int], list[AtomSite]] = {}
    for site in sites:
        if isinstance(system.entities[system.instances[site.instance_id]], PolymerSequence):
            sites_by_residue.setdefault((site.instance_id, site.component_index), []).append(site)
    token_sites: list[tuple[AtomSite, ...]] = []
    token_residues: list[tuple[str, int] | None] = []
    token_entities: list[PolymerSequence | Molecule] = []
    for instance_id, entity_id in system.instances.items():
        entity = system.entities[entity_id]
        if isinstance(entity, PolymerSequence):
            for component_index in range(len(entity)):
                residue = (instance_id, component_index)
                residue_sites = tuple(sites_by_residue[residue])
                if entity.components[component_index].component_id in _STANDARD_POLYMER_COMPONENTS:
                    token_sites.append(residue_sites)
                    token_residues.append(residue)
                    token_entities.append(entity)
                else:
                    for site in residue_sites:
                        token_sites.append((site,))
                        token_residues.append(residue)
                        token_entities.append(entity)
        else:
            for site in sites:
                if site.instance_id == instance_id:
                    token_sites.append((site,))
                    token_residues.append(None)
                    token_entities.append(entity)
    return tuple(token_sites), tuple(token_residues), tuple(token_entities)


def _build_token_indices(
    system: MolecularSystem,
    token_sites: Sequence[Sequence[AtomSite]],
    token_residues: Sequence[tuple[str, int] | None],
    token_entities: Sequence[PolymerSequence | Molecule],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    r"""
    Build ODesign chain, entity, symmetry, and residue index tensors.
    """

    chain_by_instance = {instance_id: index for index, instance_id in enumerate(system.instances)}
    entity_ids = []
    for entity_id in system.instances.values():
        if entity_id not in entity_ids:
            entity_ids.append(entity_id)
    entity_index_by_id = {value: index for index, value in enumerate(entity_ids)}
    seen_by_entity: dict[str, int] = {}
    symmetry_by_instance = {}
    for instance_id, entity_id in system.instances.items():
        symmetry_by_instance[instance_id] = seen_by_entity.get(entity_id, 0)
        seen_by_entity[entity_id] = symmetry_by_instance[instance_id] + 1
    chain = []
    entity = []
    symmetry = []
    residue = []
    for group, reference, _definition in zip(token_sites, token_residues, token_entities):
        instance_id = reference[0] if reference is not None else group[0].instance_id
        chain.append(chain_by_instance[instance_id])
        entity.append(entity_index_by_id[system.instances[instance_id]])
        symmetry.append(symmetry_by_instance[instance_id])
        if reference is None:
            residue.append(0)
        else:
            residue.append(reference[1])
    return tuple(torch.tensor(values, dtype=torch.int64) for values in (chain, entity, symmetry, residue))


def _build_token_bonds(
    system: MolecularSystem,
    token_sites: Sequence[Sequence[AtomSite]],
    site_to_token: dict[AtomSite, int],
    *,
    excluded: frozenset[frozenset[AtomSite]],
) -> Tensor:
    r"""
    Build the symmetric ODesign token-bond adjacency tensor.
    """

    bonds = torch.zeros(len(token_sites), len(token_sites), dtype=torch.float32)
    for bond in system.iter_bonds():
        if frozenset((bond.site_a, bond.site_b)) in excluded:
            continue
        if bond.site_a not in site_to_token or bond.site_b not in site_to_token:
            continue
        left, right = site_to_token[bond.site_a], site_to_token[bond.site_b]
        if left != right:
            bonds[left, right] = bonds[right, left] = 1.0
    return bonds


def _build_cyclic_period(
    token_sites: Sequence[Sequence[AtomSite]],
    cyclic_instances: frozenset[str],
) -> Tensor:
    r"""
    Build the cyclic-polymer period associated with each ODesign token.
    """

    counts = {
        instance_id: sum(group[0].instance_id == instance_id for group in token_sites)
        for instance_id in cyclic_instances
    }
    return torch.tensor(
        [counts.get(group[0].instance_id, 0) for group in token_sites],
        dtype=torch.int64,
    )


def _validate_msa(
    msa: MSA | MSACollection | None,
    system: MolecularSystem,
) -> MSA | MSACollection | None:
    if msa is None:
        return msa
    # Evidence holds no PolymerSequence: each column names one entity by ``entity_id`` (and an optional
    # exact target instance) and closes over this system. The shared resolver binds the entity, verifies any
    # exact target instance, and checks the query width and chemistry against the target -- so an MSA bound to
    # an unknown instance raises here instead of silently lowering to an all-gap row.
    resolve_msa_bindings(msa, system)
    return msa


def _atom_features(system: MolecularSystem, sites: tuple[AtomSite, ...], reference_mask: Tensor) -> Tensor:
    from rdkit.Chem import GetPeriodicTable

    periodic_table = GetPeriodicTable()
    element = torch.zeros(len(sites), 129)
    for index, site in enumerate(sites):
        atom = system.atom(site)
        element[index, int(periodic_table.GetAtomicNumber(atom.element)) - 1] = 1.0
    characters = torch.zeros(len(sites), 4, 64)
    for atom_index, site in enumerate(sites):
        name = site.atom_id
        if len(name) > 4:
            raise ValueError(f"ODesign cannot encode atom name longer than four characters: {name!r}")
        for char_index, character in enumerate(name.ljust(4)):
            value = ord(character) - 32
            if not 0 <= value < 64:
                raise ValueError(f"ODesign cannot encode atom-name character {character!r}")
            characters[atom_index, char_index, value] = 1.0
    return torch.cat([reference_mask.float().unsqueeze(-1), element, characters.flatten(-2)], dim=-1)


def _ligand_slot_atom_features(num_slots: int) -> Tensor:
    r"""
    Encode valid zero-reference atoms with released element/name mask features.
    """

    features = torch.zeros(num_slots, 386)
    features[:, 0] = 1.0
    element_offset = 1
    element_mask_index = 128
    features[:, element_offset + element_mask_index] = 1.0
    character_offset = 1 + 129
    features[:, character_offset + (ord("-") - 32)] = 1.0
    for char_index in range(1, 4):
        features[:, character_offset + char_index * 64] = 1.0
    return features


def _hotspot_tokens(
    conditions,
    placement_map,
    instance_of,
    source_residue_map,
    source_atom_map,
    site_to_token,
):
    hotspot = torch.zeros(max(site_to_token.values()) + 1, dtype=torch.bool)
    for condition in conditions:
        if not isinstance(condition, Hotspot):
            continue
        selected = False
        for reference in condition.residues:
            # Skip generated endpoints: a whole-placement ``placement_id`` (str), one symbolic
            # residue, or one symbolic atom — hotspots mark EXISTING input residues only.
            if isinstance(reference, (str, SymbolicResidueSite, SymbolicAtomSite)):
                continue
            for site in _reference_sites(
                reference,
                placement_map,
                instance_of,
                source_residue_map,
                source_atom_map,
                site_to_token,
            ):
                hotspot[site_to_token[site]] = True
                selected = True
        if not selected:
            raise ValueError("ODesign Hotspot must select at least one input residue")
    return hotspot


def _constraint_distogram(
    system,
    sites,
    site_to_token,
    observed_positions,
    constraint_coordinate_mask,
):
    size = max(site_to_token.values()) + 1
    values = observed_positions.new_zeros(size, size, 1)
    atom_by_token = {}
    for atom_index, site in enumerate(sites):
        if not bool(constraint_coordinate_mask[atom_index]):
            continue
        entity = system.entities[system.instances[site.instance_id]]
        if isinstance(entity, PolymerSequence):
            component_id = entity.components[site.component_index].component_id
            if component_id in _STANDARD_POLYMER_COMPONENTS:
                center_atom = "CA" if entity.polymer_type == "protein" else "C1'"
                if site.atom_id != center_atom:
                    continue
        atom_by_token[site_to_token[site]] = atom_index
    for left_index, left_atom in atom_by_token.items():
        for right_index, right_atom in atom_by_token.items():
            if left_index != right_index and values[left_index, right_index, 0] == 0:
                distance = torch.linalg.vector_norm(observed_positions[left_atom] - observed_positions[right_atom])
                values[left_index, right_index, 0] = distance
    return values


def _reference_sites(
    reference,
    placement_map,
    instance_of,
    source_residue_map,
    source_atom_map,
    available_sites,
):
    if isinstance(reference, AtomSite):
        return (source_atom_map[reference],)
    if isinstance(reference, SymbolicAtomSite):
        residue = _resolve_site(reference.residue, placement_map, instance_of)
        return (AtomSite(residue[0], residue[1], reference.atom_id),)
    if isinstance(reference, str):
        # A bare globally-unique placement_id names the whole generated placement, in order.
        residues = placement_map[instance_of[reference]][reference]
    elif isinstance(reference, SymbolicResidueSite):
        residues = (_resolve_site(reference, placement_map, instance_of),)
    else:
        residues = (source_residue_map[reference],)
    sites = []
    for residue in residues:
        beta = AtomSite(residue[0], residue[1], "CB")
        sites.append(beta if beta in available_sites else AtomSite(residue[0], residue[1], "CA"))
    return tuple(sites)


def _msa_features(
    msa: MSA | MSACollection | None,
    system: MolecularSystem,
    token_residues: Sequence[tuple[str, int] | None],
    token_entities: Sequence[PolymerSequence | Molecule],
    generated_residues: set[tuple[str, int]],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    token_count = len(token_residues)
    profile = torch.zeros(token_count, 32)
    deletion_mean = torch.zeros(token_count)
    if msa is None:
        row = torch.full((token_count,), 31, dtype=torch.int64)
        for index, residue in enumerate(token_residues):
            component = (
                "-"
                if residue is None or residue in generated_residues
                else token_entities[index].components[residue[1]].component_id
            )
            token = min(_RESIDUE_TYPE_INDEX.get(component, 20), 31)
            profile[index, token] = 1.0
            row[index] = token
        features = torch.cat(
            [F.one_hot(row, 32).float(), torch.zeros(token_count, 2)],
            dim=-1,
        ).unsqueeze(0)
        return features, torch.ones(1, token_count, dtype=torch.bool), profile, deletion_mean
    bindings = msa_bindings(msa)
    msas = tuple(value for value, _target_instance_id in bindings)
    # The MSA holds no PolymerSequence: column ``c`` binds ``msas[c].entity_id`` to a system entity and
    # ``query_to_entity`` is the identity map, so a token's component index is the aligned-row column.
    # ``polymer_types[c]`` resolves the one-letter-symbol lowering; ``tokens_by_entity`` groups every token
    # by entity_id and keeps each token's instance, so a column bound to one exact instance skips its siblings.
    polymer_types = [cast(PolymerSequence, system.entities[value.entity_id]).polymer_type for value in msas]
    tokens_by_entity: dict[str, list[tuple[int, int, str]]] = {}
    for token_index, residue in enumerate(token_residues):
        if residue is None:
            continue
        tokens_by_entity.setdefault(system.instances[residue[0]], []).append((token_index, residue[1], residue[0]))
    # A bare MSA replays its unpaired pool (row 0 is the query); a collection replays its dense
    # row_pairing/row_is_paired plan, selecting the paired or unpaired pool per column (``-1`` = gap).
    if isinstance(msa, MSA):
        plan: list[tuple[tuple[int, int, bool], ...]] = [
            ((0, source_row, False),) for source_row in range(len(msa.unpaired_sequences))
        ]
    else:
        row_pairing = msa.row_pairing
        row_is_paired = msa.row_is_paired
        plan = [
            tuple(
                (column, int(row_pairing[plan_index, column]), bool(row_is_paired[plan_index, column]))
                for column in range(len(msas))
                if row_pairing[plan_index, column] != -1
            )
            for plan_index in range(row_pairing.shape[0])
        ]
    rows = []
    deletions = []
    for selections in plan:
        token_row = torch.full((token_count,), 31, dtype=torch.int64)
        deletion_row = torch.zeros(token_count)
        for column_index, source_row, is_paired in selections:
            value = msas[column_index]
            target_instance_id = bindings[column_index][1]
            polymer_type = polymer_types[column_index]
            aligned = value.paired_sequences[source_row] if is_paired else value.unpaired_sequences[source_row]
            aligned_deletions = (value.paired_deletion_counts if is_paired else value.unpaired_deletion_counts)[
                source_row
            ]
            for token_index, column, instance_id in tokens_by_entity.get(value.entity_id, ()):
                if target_instance_id is not None and instance_id != target_instance_id:
                    continue
                token_row[token_index] = _msa_symbol_token(aligned[column], polymer_type)
                deletion_row[token_index] = float(aligned_deletions[column])
        rows.append(token_row)
        deletions.append(deletion_row)
    tokens = torch.stack(rows)
    deletion_counts = torch.stack(deletions)
    one_hot = F.one_hot(tokens, 32).float()
    profile = one_hot.mean(dim=0)
    deletion_mean = deletion_counts.mean(dim=0)
    has_deletion = (deletion_counts > 0).float()
    deletion_value = (2 / math.pi) * torch.atan(deletion_counts / 3)
    features = torch.cat([one_hot, has_deletion.unsqueeze(-1), deletion_value.unsqueeze(-1)], dim=-1)
    mask = torch.ones(tokens.shape, dtype=torch.bool)
    return features, mask, profile, deletion_mean
