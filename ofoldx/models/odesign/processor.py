r"""
Tensor-native input processing for ODesign structure generation.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias, cast, get_args

import torch
import torch.nn.functional as F
from torch import Tensor

from ...data import (
    MSA_GAP_INDEX,
    MSA_OTHER_INDEX,
    MSA_VOCAB_SIZE,
    TOKEN_OTHER_INDEX,
    TOKEN_VOCAB_SIZE,
    Cropper,
    TokenLayout,
    crop_token_layout,
    msa_component_index,
    msa_residue_index,
    resolved_token_mask,
    tokenize_molecular_system,
)
from ...data.bonds import SymbolicBond
from ...data.components import get_component
from ...data.conditions import Hotspot
from ...data.evidence import MSA, MsaCollection, resolve_msa_bindings
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
from .config import ODesignProcessorConfig
from .model import (
    ODesignInput,
    ODesignLigandInput,
    ODesignStructureTarget,
)

_CORE_POLYMER_TYPES: frozenset[str] = frozenset(get_args(PolymerType))

#: One designed polymer type or fixed source residue in an ODesign mask.
ODesignResidueSpec: TypeAlias = PolymerType | ComponentReference

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
        placement: PlacementSpec | None = None,
        chain_lengths: Sequence[int | None] | None = None,
        msa: MSA | MsaCollection | None = None,
        hotspots: Sequence[Hotspot] = (),
        cyclic_bonds: Sequence[SymbolicBond] = (),
        fixed_atoms: Sequence[AtomSite] = (),
        constraint_atoms: Sequence[AtomSite] = (),
        filter_by_occupancy: bool = True,
        generator: torch.Generator | None = None,
    ) -> ODesignInput:
        r"""
        Build one fixed-topology ODesign polymer input.

        Variable placement lengths are sampled here with ``generator``. Symbolic
        placement and condition metadata is resolved before the typed Input is
        returned. With ``filter_by_occupancy=True``, explicitly zero-occupancy
        coordinates are unresolved; coordinates without occupancy evidence remain
        available.
        """

        model_input, _ = self._lower(
            input,
            placement=placement,
            chain_lengths=chain_lengths,
            msa=msa,
            hotspots=hotspots,
            cyclic_bonds=cyclic_bonds,
            fixed_atoms=fixed_atoms,
            constraint_atoms=constraint_atoms,
            filter_by_occupancy=filter_by_occupancy,
            generator=generator,
            cropper=None,
            training_target=None,
        )
        return model_input

    def process_training(
        self,
        input: MolecularSystem | Structure | None,
        target: Structure | None = None,
        *,
        placement: PlacementSpec | None = None,
        chain_lengths: Sequence[int | None] | None = None,
        msa: MSA | MsaCollection | None = None,
        hotspots: Sequence[Hotspot] = (),
        cyclic_bonds: Sequence[SymbolicBond] = (),
        fixed_atoms: Sequence[AtomSite] = (),
        constraint_atoms: Sequence[AtomSite] = (),
        filter_by_occupancy: bool = True,
        generator: torch.Generator | None = None,
        cropper: Cropper | None = None,
    ) -> tuple[ODesignInput, ODesignStructureTarget]:
        r"""
        Lower one polymer problem and gather truth onto its Model atom axis, applying
        the same requested occupancy policy to conditioning and target coordinates.
        A training-only ``cropper`` retains every explicit condition and anchors its
        selection in tokens with resolved target coordinates.
        """

        if target is None:
            if not isinstance(input, Structure):
                raise TypeError("ODesign training requires a target Structure")
            target = input
        model_input, output_to_source = self._lower(
            input,
            placement=placement,
            chain_lengths=chain_lengths,
            msa=msa,
            hotspots=hotspots,
            cyclic_bonds=cyclic_bonds,
            fixed_atoms=fixed_atoms,
            constraint_atoms=constraint_atoms,
            filter_by_occupancy=filter_by_occupancy,
            generator=generator,
            cropper=cropper,
            training_target=target,
        )
        target_rows = _target_atom_rows(
            target.system,
            model_input.system,
            _source_system(input),
            output_to_source,
            model_input.atom_index[0],
        )
        coordinates, coordinate_mask = _gather_target(
            target,
            target_rows,
            filter_by_occupancy=filter_by_occupancy,
        )
        covalent_bond_index = _target_covalent_bond_index(
            target.system,
            target_rows,
            model_input.system,
            model_input.atom_index[0],
        )
        return model_input, ODesignStructureTarget(  # type: ignore[call-arg]
            coordinates=coordinates.to(device=self.device, dtype=self.dtype),
            coordinate_mask=coordinate_mask.to(self.device),
            covalent_bond_index=covalent_bond_index.to(self.device),
            batch_size=[1],
        )

    def _lower(
        self,
        input: MolecularSystem | Structure | None,
        *,
        placement: PlacementSpec | None,
        chain_lengths: Sequence[int | None] | None,
        msa: MSA | MsaCollection | None,
        hotspots: Sequence[Hotspot],
        cyclic_bonds: Sequence[SymbolicBond],
        fixed_atoms: Sequence[AtomSite],
        constraint_atoms: Sequence[AtomSite],
        filter_by_occupancy: bool,
        generator: torch.Generator | None,
        cropper: Cropper | None,
        training_target: Structure | None,
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
                chain_lengths=chain_lengths,
                generator=generator,
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
        allowed: set[PolymerType] = (
            {"protein"} if self.config.artifact in {"protein_flex", "protein_rigid"} else {"dna", "rna"}
        )
        if not generated_types <= allowed:
            raise ValueError(
                f"ODesign {self.config.artifact} artifact cannot generate {sorted(generated_types - allowed)!r}"
            )

        output_to_source = {
            output: source_atom for source_atom, outputs in source_atom_outputs.items() for output in outputs
        }
        model_atom_index = _model_atom_index(system, design_component_mask)
        if model_atom_index.numel() == 0:
            raise ValueError("ODesign requires at least one heavy atom")
        layout, atom_to_token = _tokenize(system, model_atom_index)
        if cropper is not None:
            assert training_target is not None
            canonical_atom_to_token = _canonical_atom_to_token(system, layout, atom_to_token)
            required_token_mask = _required_crop_tokens(
                system,
                source,
                source_component_outputs,
                source_atom_outputs,
                fixed_atoms,
                constraint_atoms,
                hotspots,
                cyclic_pairs,
                layout,
                atom_to_token,
                canonical_atom_to_token,
            )
            aligned_target = _aligned_target_structure(
                training_target,
                system,
                source,
                output_to_source,
                filter_by_occupancy=filter_by_occupancy,
            )
            layout = crop_token_layout(
                aligned_target,
                layout,
                cropper,
                required_token_mask=required_token_mask,
                anchor_token_mask=resolved_token_mask(aligned_target, layout),
                generator=generator,
            )
            atom_to_token = _layout_atom_to_token(layout)
        model_atom_index = layout.atom_index
        token_component_index = layout.token_component_index
        token_atom_index = layout.token_atom_index
        token_instance_index = layout.token_instance_index
        representative_atom_index = _representative_atom_index(
            system,
            model_atom_index,
            token_component_index,
            token_atom_index,
            atom_to_token,
            design_component_mask,
        )
        token_count = token_component_index.numel()
        canonical_atom_to_token = _canonical_atom_to_token(system, layout, atom_to_token)
        observed_positions, observed_mask = _observed_coordinates(
            structure,
            model_atom_index,
            output_to_source,
            filter_by_occupancy=filter_by_occupancy,
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
        context_coordinate_mask = (
            ~design_component_mask[_atom_component_index(system)[model_atom_index]] & observed_mask
        )
        constraint_coordinate_mask |= context_coordinate_mask
        if bool((constraint_coordinate_mask & ~observed_mask).any()):
            raise ValueError("fixed_atoms and constraint_atoms require observed coordinates")
        if self.config.artifact in {"protein_rigid", "nucleic_acid_rigid"}:
            fixed_coordinate_mask |= context_coordinate_mask

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
        msa_tokens, msa_deletion_counts, msa_mask, profile, deletion_mean = _msa_features(
            msa,
            system,
            token_component_index,
            token_instance_index,
            design_component_mask,
        )
        token_features, design_features = _token_features(
            system,
            token_component_index,
            token_atom_index,
            layout.token_type_index,
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
            token_atom_index,
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
            system,
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
            design_features=design_features.unsqueeze(0).to(
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
            msa_tokens=msa_tokens.unsqueeze(0).to(self.device),
            msa_deletion_counts=msa_deletion_counts.unsqueeze(0).to(self.device),
            msa_mask=msa_mask.unsqueeze(0).to(self.device),
            system=system,
            atom_index=model_atom_index.unsqueeze(0).to(self.device),
            token_component_index=token_component_index.unsqueeze(0).to(self.device),
            token_atom_index=token_atom_index.unsqueeze(0).to(self.device),
            representative_atom_index=representative_atom_index.unsqueeze(0).to(self.device),
            design_component_mask=design_component_mask,
            batch_size=[],
        )
        return model_input, output_to_source


class ODesignLigandProcessor(Processor):
    r"""
    Lower canonical context and ordered provisional ligand rows for ODesign.
    """

    def __call__(
        self,
        input: MolecularSystem | Structure | None,
        *,
        generated_atoms: int | range,
        msa: MSA | MsaCollection | None = None,
        hotspots: Sequence[Hotspot] = (),
        fixed_atoms: Sequence[AtomSite] = (),
        constraint_atoms: Sequence[AtomSite] = (),
        filter_by_occupancy: bool = True,
        generator: torch.Generator | None = None,
    ) -> ODesignLigandInput:
        r"""
        Build one ligand input, sampling its provisional atom count exactly once.

        With ``filter_by_occupancy=True``, explicitly zero-occupancy context
        coordinates are unresolved.
        """

        return self._lower(
            input,
            generated_atoms=generated_atoms,
            msa=msa,
            hotspots=hotspots,
            fixed_atoms=fixed_atoms,
            constraint_atoms=constraint_atoms,
            filter_by_occupancy=filter_by_occupancy,
            generator=generator,
        )

    def process_training(
        self,
        input: MolecularSystem | Structure | None,
        target: Structure,
        *,
        generated_target_instance_ids: Sequence[str],
        generated_atoms: int | range | None = None,
        msa: MSA | MsaCollection | None = None,
        hotspots: Sequence[Hotspot] = (),
        fixed_atoms: Sequence[AtomSite] = (),
        constraint_atoms: Sequence[AtomSite] = (),
        filter_by_occupancy: bool = True,
        generator: torch.Generator | None = None,
    ) -> tuple[ODesignLigandInput, ODesignStructureTarget]:
        r"""
        Lower explicitly selected ligand truth, applying the same requested
        occupancy policy to context and target coordinates.
        """

        if not generated_target_instance_ids:
            raise ValueError("generated_target_instance_ids must not be empty")
        if len(set(generated_target_instance_ids)) != len(generated_target_instance_ids):
            raise ValueError("generated_target_instance_ids must be unique")
        instance_by_id = {
            target.system.instance_id(index): index for index in range(int(target.system.instance_symbol.numel()))
        }
        unknown = set(generated_target_instance_ids) - instance_by_id.keys()
        if unknown:
            raise ValueError(f"target does not contain generated instance {sorted(unknown)[0]!r}")
        generated_rows: list[int] = []
        for instance_id in generated_target_instance_ids:
            instance = instance_by_id[instance_id]
            entity = int(target.system.instance_entity[instance])
            if int(target.system.entity_type[entity]) != ENTITY_MOLECULE:
                raise ValueError(f"generated target instance {instance_id!r} is not a molecule")
            component_start = int(target.system.instance_component_ptr[instance])
            component_stop = int(target.system.instance_component_ptr[instance + 1])
            atom_start = int(target.system.component_atom_ptr[component_start])
            atom_stop = int(target.system.component_atom_ptr[component_stop])
            generated_rows.extend(range(atom_start, atom_stop))
        generated_count = len(generated_rows)
        if generated_count == 0:
            raise ValueError("generated target instances must contain atoms")
        if generated_atoms is not None and (
            (isinstance(generated_atoms, range) and generated_count not in generated_atoms)
            or (isinstance(generated_atoms, int) and generated_count != generated_atoms)
        ):
            raise ValueError("generated_atoms does not contain the target's generated atom count")

        model_input = self._lower(
            input,
            generated_atoms=generated_count,
            msa=msa,
            hotspots=hotspots,
            fixed_atoms=fixed_atoms,
            constraint_atoms=constraint_atoms,
            filter_by_occupancy=filter_by_occupancy,
            generator=generator,
        )
        context_rows = []
        for atom in model_input.atom_index[0].detach().cpu().tolist():
            if atom < 0:
                break
            try:
                target_atom = target.system.atom_index(model_input.system.atom_site(atom))
            except KeyError:
                target_atom = -1
            context_rows.append(target_atom)
        slot_count = int((model_input.atom_mask[0] & (model_input.atom_index[0] < 0)).sum())
        if len(generated_rows) != slot_count:
            raise ValueError("generated target instances must have one atom per provisional atom")
        target_rows = [*context_rows, *generated_rows]
        coordinates, coordinate_mask = _gather_target(
            target,
            target_rows,
            filter_by_occupancy=filter_by_occupancy,
        )
        covalent_bond_index = _target_covalent_bond_index(
            target.system,
            target_rows,
            model_input.system,
            model_input.atom_index[0],
        )
        return model_input, ODesignStructureTarget(  # type: ignore[call-arg]
            coordinates=coordinates.to(device=self.device, dtype=self.dtype),
            coordinate_mask=coordinate_mask.to(self.device),
            covalent_bond_index=covalent_bond_index.to(self.device),
            batch_size=[1],
        )

    def _lower(
        self,
        input: MolecularSystem | Structure | None,
        *,
        generated_atoms: int | range,
        msa: MSA | MsaCollection | None,
        hotspots: Sequence[Hotspot],
        fixed_atoms: Sequence[AtomSite],
        constraint_atoms: Sequence[AtomSite],
        filter_by_occupancy: bool,
        generator: torch.Generator | None,
    ) -> ODesignLigandInput:
        structure = input if isinstance(input, Structure) else None
        system = _source_system(input)
        if system is None:
            system = MolecularSystem.from_entities({}, {})
        if isinstance(generated_atoms, range):
            random_device = None if generator is None else generator.device
            num_slots = generated_atoms[
                int(
                    torch.randint(
                        len(generated_atoms),
                        (),
                        device=random_device,
                        generator=generator,
                    )
                )
            ]
        else:
            num_slots = generated_atoms
        context_design_mask = torch.zeros_like(
            system.component_symbol,
            dtype=torch.bool,
        )
        context_atom_index = _model_atom_index(
            system,
            context_design_mask,
        )
        context_layout, context_atom_to_token = _tokenize(system, context_atom_index)
        context_atom_index = context_layout.atom_index
        context_token_component = context_layout.token_component_index
        context_token_atom = context_layout.token_atom_index
        context_token_instance = context_layout.token_instance_index
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
        representative_atom_index = _representative_atom_index(
            system,
            atom_index,
            token_component_index,
            token_atom_index,
            atom_to_token,
            context_design_mask,
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
            filter_by_occupancy=filter_by_occupancy,
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
        constraint_coordinate_mask[:context_atom_count] |= observed_mask[:context_atom_count]
        if bool((constraint_coordinate_mask & ~observed_mask).any()):
            raise ValueError("fixed_atoms and constraint_atoms require observed coordinates")
        fixed_coordinate_mask[:context_atom_count] |= observed_mask[:context_atom_count]

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
            context_msa_tokens,
            context_msa_deletion_counts,
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
        msa_depth = context_msa_tokens.shape[0]
        slot_msa_tokens = torch.full(
            (msa_depth, num_slots),
            MSA_GAP_INDEX,
            dtype=torch.int64,
            device=context_msa_tokens.device,
        )
        msa_tokens = torch.cat(
            [context_msa_tokens, slot_msa_tokens],
            dim=1,
        )
        msa_deletion_counts = torch.cat(
            [
                context_msa_deletion_counts,
                context_msa_deletion_counts.new_zeros(msa_depth, num_slots),
            ],
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
        slot_profile = context_profile.new_zeros(num_slots, MSA_VOCAB_SIZE)
        slot_profile[:, MSA_GAP_INDEX] = 1
        profile = torch.cat([context_profile, slot_profile])
        deletion_mean = torch.cat(
            [
                context_deletion_mean,
                context_deletion_mean.new_zeros(num_slots),
            ]
        )
        slot_type_index = torch.full(
            (num_slots,),
            TOKEN_OTHER_INDEX,
            dtype=torch.int64,
            device=system.atom_symbol.device,
        )
        token_features, design_features = _token_features(
            system,
            token_component_index,
            token_atom_index,
            torch.cat((context_layout.token_type_index, slot_type_index)),
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
            context_token_atom,
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
            design_features=design_features.unsqueeze(0).to(
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
            msa_tokens=msa_tokens.unsqueeze(0).to(self.device),
            msa_deletion_counts=msa_deletion_counts.unsqueeze(0).to(self.device),
            msa_mask=msa_mask.unsqueeze(0).to(self.device),
            system=system,
            atom_index=atom_index.unsqueeze(0).to(self.device),
            token_component_index=token_component_index.unsqueeze(0).to(self.device),
            token_atom_index=token_atom_index.unsqueeze(0).to(self.device),
            representative_atom_index=representative_atom_index.unsqueeze(0).to(self.device),
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
                raise ValueError(f"designed residue {index} has non-core polymer_type {residue!r}")
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


def _realize_placement(
    placement: PlacementSpec,
    *,
    chain_lengths: Sequence[int | None] | None,
    generator: torch.Generator | None,
) -> PlacementSpec:
    if chain_lengths is None:
        chain_lengths = (None,) * len(placement)
    realized: PlacementSpec = {}
    random_device = None if generator is None else generator.device
    for (group, chain), chain_length in zip(
        placement.items(),
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
                            device=random_device,
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
                    f"ODesign placement chain length {chain_length} cannot be realized in [{minimum}, {maximum}]"
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
                        device=random_device,
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
    chain_source_components: list[list[tuple[int, int]]] = []
    chain_placement_source_atoms: list[dict[str, dict[int, list[int]]]] = []
    placement_map: PlacementMap = {}
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
        source_component_pairs: list[tuple[int, int]] = []
        placement_source_atoms: dict[str, dict[int, list[int]]] = {}
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
                placement_atom_outputs: dict[int, list[int]] = {}
                placement_source_atoms[placement_id] = placement_atom_outputs
                source_components = range(
                    source_start + value.start,
                    source_start + value.stop,
                )
                if length:
                    chain_system, old_to_new, copied_atom_index = copy_components(
                        chain_system,
                        torch.arange(
                            offset,
                            offset + length,
                            dtype=torch.int64,
                            device=device,
                        ),
                        source,
                        torch.arange(
                            source_start + value.start,
                            source_start + value.stop,
                            dtype=torch.int64,
                            device=source.atom_symbol.device,
                        ),
                    )
                    for previous_outputs in placement_source_atoms.values():
                        _remap_source_atoms(previous_outputs, old_to_new)
                    source_atoms = copied_atom_index[0].detach().cpu().tolist()
                    output_atoms = copied_atom_index[1].detach().cpu().tolist()
                    placement_atom_outputs.update(
                        {
                            int(source_atom): [int(output_atom)]
                            for source_atom, output_atom in zip(
                                source_atoms,
                                output_atoms,
                                strict=True,
                            )
                        }
                    )
                source_component_pairs.extend(
                    (source_component, offset + local) for local, source_component in enumerate(source_components)
                )
            offset += length
        placement_map[group] = realized_group
        chain_systems.append(chain_system)
        chain_design_masks.append(design_mask)
        chain_source_components.append(source_component_pairs)
        chain_placement_source_atoms.append(placement_source_atoms)

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
    context_source_atom_outputs: dict[int, list[int]] = {}
    placement_source_atom_outputs: dict[str, dict[int, list[int]]] = {}
    if source is not None:
        context_component_outputs = component_maps[0].detach().cpu().tolist()
        for source_component, context_component in enumerate(context_component_map.detach().cpu().tolist()):
            if context_component >= 0:
                source_component_outputs.setdefault(
                    source_component,
                    [],
                ).append(int(context_component_outputs[context_component]))
        context_atom_outputs = atom_maps[0].detach().cpu().tolist()
        for source_atom, context_atom in enumerate(context_atom_map.detach().cpu().tolist()):
            if context_atom >= 0:
                output_atom = int(context_atom_outputs[context_atom])
                context_source_atom_outputs[source_atom] = [output_atom]
                source_atom_outputs.setdefault(source_atom, []).append(output_atom)
        for chain_index, (
            source_component_pairs,
            placement_source_atoms,
        ) in enumerate(
            zip(
                chain_source_components,
                chain_placement_source_atoms,
                strict=True,
            ),
            start=1,
        ):
            component_outputs = component_maps[chain_index].detach().cpu().tolist()
            for source_component, chain_component in source_component_pairs:
                source_component_outputs.setdefault(
                    source_component,
                    [],
                ).append(int(component_outputs[chain_component]))
            atom_outputs = atom_maps[chain_index].detach().cpu().tolist()
            for chain_placement_id, chain_atom_outputs in placement_source_atoms.items():
                output_atom_map: dict[int, list[int]] = {}
                for source_atom, chain_atoms in chain_atom_outputs.items():
                    mapped = [int(atom_outputs[chain_atom]) for chain_atom in chain_atoms]
                    output_atom_map[source_atom] = mapped
                    source_atom_outputs.setdefault(source_atom, []).extend(mapped)
                placement_source_atom_outputs[chain_placement_id] = output_atom_map

        assembly_edges = (
            (source.bond_role == BOND_ROLE_ASSEMBLY).nonzero(as_tuple=False).flatten().detach().cpu().tolist()
        )
        for edge in assembly_edges:
            source_left = int(source.edge_index[0, edge])
            source_right = int(source.edge_index[1, edge])
            left_outputs = source_atom_outputs.get(
                source_left,
                (),
            )
            right_outputs = source_atom_outputs.get(
                source_right,
                (),
            )
            if not left_outputs or not right_outputs:
                continue

            placement_pairs: list[str | None] = []
            paired_left: set[int] = set()
            paired_right: set[int] = set()
            for candidate_id, candidate_atoms in placement_source_atom_outputs.items():
                placement_left = candidate_atoms.get(source_left, ())
                placement_right = candidate_atoms.get(source_right, ())
                if placement_left and placement_right:
                    placement_pairs.append(candidate_id)
                    paired_left.update(placement_left)
                    paired_right.update(placement_right)

            if len(left_outputs) == 1 and len(right_outputs) == 1:
                if not placement_pairs:
                    placement_pairs.append(None)
            elif paired_left != set(left_outputs) or paired_right != set(right_outputs):
                raise ValueError(
                    "ODesign cannot resolve a source assembly bond across repeated ExistingSpan placements"
                )

            source_stereo = source.bond_stereo_index[:, edge]
            for bond_placement_id in placement_pairs:
                if bond_placement_id is None:
                    left_output = source_atom_outputs[source_left][0]
                    right_output = source_atom_outputs[source_right][0]
                    placement_atoms = None
                else:
                    placement_atoms = placement_source_atom_outputs[bond_placement_id]
                    left_output = placement_atoms[source_left][0]
                    right_output = placement_atoms[source_right][0]

                if (
                    bond_placement_id is None
                    and context_source_atom_outputs.get(source_left) == [left_output]
                    and context_source_atom_outputs.get(source_right) == [right_output]
                ):
                    physical_left = min(left_output, right_output)
                    physical_right = max(left_output, right_output)
                    if bool(((system.edge_index[0] == physical_left) & (system.edge_index[1] == physical_right)).any()):
                        continue

                stereo = source.symbols[int(source.bond_stereo[edge])] if int(source.bond_stereo[edge]) >= 0 else None
                stereo_atoms = None
                if bool((source_stereo >= 0).any()):
                    stereo_outputs: list[int] = []
                    if bool((source_stereo >= 0).all()):
                        for stereo_source in source_stereo.detach().cpu().tolist():
                            candidates = () if placement_atoms is None else placement_atoms.get(int(stereo_source), ())
                            if len(candidates) != 1:
                                candidates = source_atom_outputs.get(int(stereo_source), ())
                            if len(candidates) != 1:
                                break
                            stereo_outputs.append(candidates[0])
                    if len(stereo_outputs) == 2:
                        stereo_atoms = (stereo_outputs[0], stereo_outputs[1])
                    else:
                        stereo = None

                system, _ = form_bond(
                    system,
                    left_output,
                    right_output,
                    system.atom_symbol.new_empty(0),
                    system.atom_symbol.new_empty(0),
                    order=(float(source.bond_order[edge]) if bool(source.bond_order_known[edge]) else None),
                    aromatic=(bool(source.bond_aromatic[edge]) if bool(source.bond_aromatic_known[edge]) else None),
                    kind=cast(
                        BondKind,
                        {
                            BOND_KIND_COVALENT: "covalent",
                            BOND_KIND_DATIVE: "dative",
                        }[int(source.bond_kind[edge])],
                    ),
                    direction=cast(
                        BondDirection,
                        {
                            BOND_DIRECTION_UNDIRECTED: "undirected",
                            BOND_DIRECTION_ENDPOINT_0_DONOR: "a_to_b",
                            BOND_DIRECTION_ENDPOINT_1_DONOR: "b_to_a",
                        }[int(source.bond_direction[edge])],
                    ),
                    stereo=stereo,
                    stereo_atoms=stereo_atoms,
                    role="assembly",
                )

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


def _remap_source_atoms(
    source_atom_outputs: dict[int, list[int]],
    old_to_new: Tensor,
) -> None:
    source_atoms = list(source_atom_outputs)
    output_counts = [len(source_atom_outputs[source_atom]) for source_atom in source_atoms]
    outputs = [output for source_atom in source_atoms for output in source_atom_outputs[source_atom]]
    if not outputs:
        return
    remapped = (
        old_to_new[
            torch.tensor(
                outputs,
                dtype=torch.int64,
                device=old_to_new.device,
            )
        ]
        .detach()
        .cpu()
        .tolist()
    )
    offset = 0
    for source_atom, count in zip(source_atoms, output_counts, strict=True):
        source_atom_outputs[source_atom] = [int(output) for output in remapped[offset : offset + count] if output >= 0]
        offset += count


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


def _tokenize(
    system: MolecularSystem,
    model_atom_index: Tensor,
) -> tuple[TokenLayout, Tensor]:
    atom_mask = torch.zeros_like(system.atom_symbol, dtype=torch.bool)
    atom_mask[model_atom_index] = True
    layout = tokenize_molecular_system(system, atom_mask=atom_mask)
    return layout, _layout_atom_to_token(layout)


def _layout_atom_to_token(layout: TokenLayout) -> Tensor:
    return torch.repeat_interleave(
        torch.arange(
            layout.token_component_index.numel(),
            dtype=torch.int64,
            device=layout.token_component_index.device,
        ),
        layout.token_atom_ptr[1:] - layout.token_atom_ptr[:-1],
    )


def _canonical_atom_to_token(
    system: MolecularSystem,
    layout: TokenLayout,
    atom_to_token: Tensor,
) -> Tensor:
    canonical = torch.full(
        (system.atom_symbol.numel(),),
        -1,
        dtype=torch.int64,
        device=system.atom_symbol.device,
    )
    canonical[layout.atom_index] = atom_to_token
    return canonical


def _required_crop_tokens(
    system: MolecularSystem,
    source: MolecularSystem | None,
    source_component_outputs: dict[int, tuple[int, ...]],
    source_atom_outputs: dict[int, tuple[int, ...]],
    fixed_atoms: Sequence[AtomSite],
    constraint_atoms: Sequence[AtomSite],
    hotspots: Sequence[Hotspot],
    cyclic_pairs: Sequence[tuple[int, int]],
    layout: TokenLayout,
    atom_to_token: Tensor,
    canonical_atom_to_token: Tensor,
) -> Tensor:
    required = _hotspot_tokens(
        hotspots,
        source,
        source_component_outputs,
        source_atom_outputs,
        layout.token_component_index,
        canonical_atom_to_token,
    )
    condition_atom_mask = _condition_atom_mask(
        (*fixed_atoms, *constraint_atoms),
        source,
        source_atom_outputs,
        layout.atom_index,
    )
    required[atom_to_token[condition_atom_mask]] = True
    for atom_a, atom_b in cyclic_pairs:
        tokens = canonical_atom_to_token[
            torch.tensor(
                (atom_a, atom_b),
                dtype=torch.int64,
                device=canonical_atom_to_token.device,
            )
        ]
        required[tokens[tokens >= 0]] = True
    return required


def _representative_atom_index(
    system: MolecularSystem,
    model_atom_index: Tensor,
    token_component_index: Tensor,
    token_atom_index: Tensor,
    atom_to_token: Tensor,
    design_component_mask: Tensor,
) -> Tensor:
    canonical_to_model = {
        int(atom): row for row, atom in enumerate(model_atom_index.detach().cpu().tolist()) if atom >= 0
    }
    component_instance = _component_instance_index(system)
    representatives: list[int] = []
    for token, (component, atom) in enumerate(
        zip(
            token_component_index.detach().cpu().tolist(),
            token_atom_index.detach().cpu().tolist(),
            strict=True,
        )
    ):
        if atom >= 0:
            representatives.append(canonical_to_model[atom])
            continue
        if component < 0:
            representatives.append(int((atom_to_token == token).nonzero(as_tuple=False)[0]))
            continue

        entity = int(system.instance_entity[int(component_instance[component])])
        entity_type = int(system.entity_type[entity])
        component_id = system.symbols[int(system.component_symbol[component])]
        if entity_type == ENTITY_PROTEIN:
            atom_id = "CA" if bool(design_component_mask[component]) or component_id in {"GLY", "UNK"} else "CB"
        elif component_id in {"DA", "DG", "A", "G"}:
            atom_id = "C4"
        elif component_id in {"DC", "DT", "C", "U"}:
            atom_id = "C2"
        else:
            atom_id = "C1'"

        start = int(system.component_atom_ptr[component])
        stop = int(system.component_atom_ptr[component + 1])
        representative = next(
            canonical_to_model[canonical_atom]
            for canonical_atom in range(start, stop)
            if canonical_atom in canonical_to_model
            and system.symbols[int(system.atom_symbol[canonical_atom])] == atom_id
        )
        representatives.append(representative)
    return torch.tensor(
        representatives,
        dtype=torch.int64,
        device=model_atom_index.device,
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
    *,
    filter_by_occupancy: bool,
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
    available = structure.coordinate_mask
    if filter_by_occupancy:
        available = available & (~structure.occupancy_mask | (structure.occupancy > 0))
    gathered_mask = available[source_index.clamp_min(0)] & present
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
            raise ValueError(f"ODesign cannot encode atom name longer than four characters: {name!r}")
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
    token_type_index: Tensor,
    design_component_mask: Tensor,
    profile: Tensor,
    deletion_mean: Tensor,
    hotspot: Tensor,
) -> tuple[Tensor, Tensor]:
    features = torch.zeros(
        token_component_index.numel(),
        TOKEN_VOCAB_SIZE + MSA_VOCAB_SIZE + 2,
        dtype=torch.float32,
        device=token_component_index.device,
    )
    component_instance = _component_instance_index(system)
    valid_component = token_component_index >= 0
    generated = torch.zeros_like(valid_component)
    generated[valid_component] = design_component_mask[token_component_index[valid_component]]
    atom_token = (token_atom_index >= 0) | ~valid_component
    type_index = token_type_index.clone()
    type_index[atom_token] = TOKEN_OTHER_INDEX
    features[
        torch.arange(token_component_index.numel(), device=token_component_index.device),
        type_index,
    ] = 1

    profile_start = TOKEN_VOCAB_SIZE
    features[:, profile_start : profile_start + MSA_VOCAB_SIZE] = profile
    features[:, -2] = deletion_mean
    features[:, -1] = hotspot.to(torch.float32)

    design_features = torch.zeros(
        token_component_index.numel(),
        4,
        dtype=torch.float32,
        device=token_component_index.device,
    )
    generated_rows = generated.nonzero(as_tuple=False).flatten()
    generated_components = token_component_index[generated_rows]
    generated_instances = component_instance[generated_components]
    generated_entities = system.instance_entity[generated_instances]
    entity_type = system.entity_type[generated_entities]
    state = torch.full(
        entity_type.shape,
        2,
        dtype=torch.int64,
        device=entity_type.device,
    )
    state[entity_type == ENTITY_PROTEIN] = 0
    state[entity_type == ENTITY_RNA] = 1
    design_features[generated_rows, state] = 1
    design_features[atom_token, 3] = 1
    return features, design_features


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
    token_atom_index: Tensor,
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
        atom_name = system.symbols[int(system.atom_symbol[atom])]
        token = int(canonical_atom_to_token[atom])
        if token_atom_index[token] < 0:
            center = "CA" if int(system.entity_type[entity]) == ENTITY_PROTEIN else "C1'"
            if atom_name != center:
                continue
        selected[token] = model_row
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
    system: MolecularSystem,
    token_instance_index: Tensor,
    cyclic_instances: frozenset[int],
) -> Tensor:
    r"""
    Project each complete cyclic instance period onto the selected token rows.

    >>> from types import SimpleNamespace
    >>> system = SimpleNamespace(instance_component_ptr=torch.tensor([0, 4]))
    >>> _cyclic_period(system, torch.tensor([0, 0]), frozenset({0}))
    tensor([4, 4])
    """

    period = torch.zeros_like(token_instance_index)
    for instance in cyclic_instances:
        mask = token_instance_index == instance
        period[mask] = system.instance_component_ptr[instance + 1] - system.instance_component_ptr[instance]
    return period


def _component_msa_token(system: MolecularSystem, component: int, entity: int) -> int:
    entity_type = int(system.entity_type[entity])
    if entity_type == ENTITY_MOLECULE:
        return MSA_OTHER_INDEX
    component_id = system.symbols[int(system.component_symbol[component])]
    if entity_type == ENTITY_PROTEIN:
        polymer_type = "protein"
    elif entity_type == ENTITY_RNA:
        polymer_type = "rna"
    elif entity_type == ENTITY_DNA:
        polymer_type = "dna"
    elif entity_type == ENTITY_DNA_RNA_HYBRID:
        polymer_type = "rna" if component_id in {"A", "G", "C", "U", "N"} else "dna"
    else:
        return MSA_OTHER_INDEX
    return msa_component_index(component_id, polymer_type)


def _component_msa_polymer_type(system: MolecularSystem, component: int, polymer_type: str) -> str:
    if polymer_type != "dna_rna_hybrid":
        return polymer_type
    component_id = system.symbols[int(system.component_symbol[component])]
    return "rna" if component_id in {"A", "G", "C", "U", "N"} else "dna"


def _msa_features(
    msa: MSA | MsaCollection | None,
    system: MolecularSystem,
    token_component_index: Tensor,
    token_instance_index: Tensor,
    design_component_mask: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    token_count = token_component_index.numel()
    device = token_component_index.device
    profile = torch.zeros(token_count, MSA_VOCAB_SIZE, dtype=torch.float32, device=device)
    deletion_mean = torch.zeros(token_count, dtype=torch.float32, device=device)
    if msa is None:
        row = torch.full(
            (token_count,),
            MSA_GAP_INDEX,
            dtype=torch.int64,
            device=device,
        )
        for token, component in enumerate(token_component_index.detach().cpu().tolist()):
            if component < 0 or bool(design_component_mask[component]):
                residue_token = MSA_GAP_INDEX
            else:
                instance = int(token_instance_index[token])
                entity = int(system.instance_entity[instance])
                residue_token = _component_msa_token(system, component, entity)
            row[token] = residue_token
            profile[token, residue_token] = 1
        tokens = row.unsqueeze(0)
        deletion_counts = torch.zeros(1, token_count, dtype=torch.int32, device=device)
        return (
            tokens,
            deletion_counts,
            torch.ones_like(tokens, dtype=torch.bool),
            profile,
            deletion_mean,
        )

    resolved = resolve_msa_bindings(msa, system)
    columns = tuple(value for value, _, _ in resolved)
    targets = tuple(target for _, target, _ in resolved)
    entities = tuple(entity for _, _, entity in resolved)
    polymer_types = [_POLYMER_TYPE_BY_ENTITY[int(system.entity_type[entity])] for entity in entities]
    component_instance = _component_instance_index(system)
    tokens_by_entity: dict[int, list[tuple[int, int, int, int]]] = {}
    for token, component in enumerate(token_component_index.detach().cpu().tolist()):
        if component < 0:
            continue
        instance = int(component_instance[component])
        entity = int(system.instance_entity[instance])
        local = component - int(system.instance_component_ptr[instance])
        tokens_by_entity.setdefault(entity, []).append((token, local, instance, component))
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
    rows: list[Tensor] = []
    deletions: list[Tensor] = []
    for selections in plan:
        token_row = torch.full(
            (token_count,),
            MSA_GAP_INDEX,
            dtype=torch.int64,
            device=device,
        )
        deletion_row = torch.zeros(token_count, dtype=torch.int32, device=device)
        for column, source_row, paired in selections:
            value = columns[column]
            sequences = value.paired_sequences if paired else value.unpaired_sequences
            source_deletions = value.paired_deletion_counts if paired else value.unpaired_deletion_counts
            for token, local, instance, component in tokens_by_entity.get(
                entities[column],
                (),
            ):
                if targets[column] is not None and instance != targets[column]:
                    continue
                polymer_type = _component_msa_polymer_type(system, component, polymer_types[column])
                token_row[token] = msa_residue_index(
                    sequences[source_row][local],
                    polymer_type,
                )
                deletion_row[token] = int(source_deletions[source_row, local])
        rows.append(token_row)
        deletions.append(deletion_row)
    tokens = torch.stack(rows)
    deletion_counts = torch.stack(deletions)
    query = torch.full((token_count,), MSA_GAP_INDEX, dtype=torch.int64, device=device)
    for token, component in enumerate(token_component_index.detach().cpu().tolist()):
        if component < 0 or bool(design_component_mask[component]):
            continue
        instance = int(token_instance_index[token])
        entity = int(system.instance_entity[instance])
        query[token] = _component_msa_token(system, component, entity)
    tokens[0] = query
    profile.scatter_add_(
        1,
        tokens.transpose(0, 1),
        torch.ones_like(tokens.transpose(0, 1), dtype=profile.dtype),
    )
    profile /= tokens.shape[0]
    deletion_mean = deletion_counts.to(torch.float32).mean(dim=0)
    return (
        tokens,
        deletion_counts,
        torch.ones_like(tokens, dtype=torch.bool),
        profile,
        deletion_mean,
    )


def _target_atom_rows(
    target_system: MolecularSystem,
    output_system: MolecularSystem,
    source_system: MolecularSystem | None,
    output_to_source: dict[int, int],
    output_atom_index: Tensor,
) -> list[int]:
    rows = []
    for output_atom in output_atom_index.detach().cpu().tolist():
        try:
            target_atom = target_system.atom_index(output_system.atom_site(output_atom))
        except KeyError:
            source_atom = output_to_source.get(output_atom)
            if source_atom is None or source_system is None:
                target_atom = -1
            else:
                try:
                    target_atom = target_system.atom_index(source_system.atom_site(source_atom))
                except KeyError:
                    target_atom = -1
        rows.append(target_atom)
    return rows


def _aligned_target_structure(
    target: Structure,
    output_system: MolecularSystem,
    source_system: MolecularSystem | None,
    output_to_source: dict[int, int],
    *,
    filter_by_occupancy: bool,
) -> Structure:
    output_atom_index = torch.arange(
        output_system.atom_symbol.numel(),
        dtype=torch.int64,
        device=output_system.atom_symbol.device,
    )
    coordinates, coordinate_mask = _gather_target(
        target,
        _target_atom_rows(
            target.system,
            output_system,
            source_system,
            output_to_source,
            output_atom_index,
        ),
        filter_by_occupancy=filter_by_occupancy,
    )
    coordinates = coordinates.squeeze(0).to(output_system.atom_symbol.device)
    coordinate_mask = coordinate_mask.squeeze(0).to(output_system.atom_symbol.device)
    atom_count = output_system.atom_symbol.numel()
    component_count = output_system.component_symbol.numel()
    return Structure(  # type: ignore[call-arg]
        system=output_system,
        coordinates=coordinates,
        coordinate_mask=coordinate_mask,
        occupancy=coordinates.new_zeros(atom_count),
        occupancy_mask=torch.zeros(atom_count, dtype=torch.bool, device=coordinates.device),
        b_factor=coordinates.new_zeros(atom_count),
        b_factor_mask=torch.zeros(atom_count, dtype=torch.bool, device=coordinates.device),
        source_symbols=(),
        residue_id=torch.full(
            (component_count,),
            -1,
            dtype=torch.int64,
            device=coordinates.device,
        ),
        insertion_code=torch.full(
            (component_count,),
            -1,
            dtype=torch.int64,
            device=coordinates.device,
        ),
        standard_component_id=torch.full(
            (component_count,),
            -1,
            dtype=torch.int64,
            device=coordinates.device,
        ),
        batch_size=[],
    )


def _gather_target(
    target: Structure,
    rows: Sequence[int],
    *,
    filter_by_occupancy: bool,
) -> tuple[Tensor, Tensor]:
    index = torch.tensor(
        rows,
        dtype=torch.int64,
        device=target.coordinates.device,
    )
    valid = index >= 0
    coordinates = target.coordinates[index.clamp_min(0)].unsqueeze(0)
    available = target.coordinate_mask
    if filter_by_occupancy:
        available = available & (~target.occupancy_mask | (target.occupancy > 0))
    coordinate_mask = (available[index.clamp_min(0)] & valid).unsqueeze(0)
    return (
        coordinates.masked_fill(
            ~coordinate_mask.unsqueeze(-1),
            0,
        ),
        coordinate_mask,
    )


def _target_covalent_bond_index(
    target_system: MolecularSystem,
    target_rows: Sequence[int],
    output_system: MolecularSystem,
    model_atom_index: Tensor,
) -> Tensor:
    target_to_model: dict[int, list[int]] = {}
    for model_atom, target_atom in enumerate(target_rows):
        if target_atom >= 0:
            target_to_model.setdefault(target_atom, []).append(model_atom)

    output_to_model = {
        output_atom: model_atom
        for model_atom, output_atom in enumerate(model_atom_index.detach().cpu().tolist())
        if output_atom >= 0
    }
    output_bonds: set[tuple[int, int]] = set()
    for edge in range(output_system.edge_index.shape[1]):
        if int(output_system.bond_kind[edge]) != BOND_KIND_COVALENT:
            continue
        model_left = output_to_model.get(int(output_system.edge_index[0, edge]))
        model_right = output_to_model.get(int(output_system.edge_index[1, edge]))
        if model_left is not None and model_right is not None and model_left != model_right:
            output_bonds.add((model_left, model_right) if model_left < model_right else (model_right, model_left))

    bonds: set[tuple[int, int]] = set()
    for edge in range(target_system.edge_index.shape[1]):
        if int(target_system.bond_kind[edge]) != BOND_KIND_COVALENT:
            continue
        left = int(target_system.edge_index[0, edge])
        right = int(target_system.edge_index[1, edge])
        left_candidates = target_to_model.get(left, ())
        right_candidates = target_to_model.get(right, ())
        if not left_candidates or not right_candidates:
            continue
        if len(left_candidates) == 1 and len(right_candidates) == 1:
            model_left = left_candidates[0]
            model_right = right_candidates[0]
            if model_left != model_right:
                bonds.add((model_left, model_right) if model_left < model_right else (model_right, model_left))
            continue

        left_candidate_set = set(left_candidates)
        right_candidate_set = set(right_candidates)
        resolved = {
            (model_left, model_right)
            for model_left, model_right in output_bonds
            if (model_left in left_candidate_set and model_right in right_candidate_set)
            or (model_right in left_candidate_set and model_left in right_candidate_set)
        }
        if not resolved:
            raise ValueError("ODesign cannot resolve a target covalent bond across repeated atom placements")
        bonds.update(resolved)
    return torch.tensor(
        sorted(bonds),
        dtype=torch.int64,
        device=target_system.edge_index.device,
    ).reshape(1, -1, 2)
