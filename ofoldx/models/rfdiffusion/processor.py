r"""Lower canonical molecular inputs and Conditions for RFdiffusion."""

from __future__ import annotations

import operator
from collections.abc import Mapping, Sequence
from os import PathLike
from pathlib import Path
from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor

# AllowedResidueIdentities is shared by sequence-design pipelines.
from ...data.conditions import AllowedResidueIdentities
from ...data.evidence import Template
from ...data.molecular import (
    AtomSite,
    ComponentReference,
    MolecularSystem,
    Molecule,
    PolymerSequence,
)
from ...data.placement import (
    InstanceIdAllocator,
    PlacementSpec,
    PlacementValue,
)
from ...data.placement import instance_of as derive_instance_of
from ...data.placement import symbolic_residue_aliases, validate_placement
from ...data.structure import Structure
from ...io.structure import read_structure
from ...processor import Processor
from .config import RfDiffusionProcessorConfig

# RFdiffusion spatial controls are defined with the model. Topology is a separate
# ``placement`` argument rather than a condition.
from .model import (
    ATOM14_NAMES,
    MASK_TOKEN,
    NUM2AA,
    ExistingSpan,
    GeneratedSegment,
    Hotspot,
    RfDiffusionCoordination,
    RfDiffusionDistance,
    RfDiffusionInput,
    RfDiffusionLayout,
    RfDiffusionPlacements,
    RfDiffusionPointGroupSymmetry,
    RfDiffusionRadiusOfGyration,
    RfDiffusionReferenceCoordinates,
    RfDiffusionSecondaryStructure,
    RfDiffusionSubstrateInteraction,
    SymbolicAtomSite,
    SymbolicBond,
    SymbolicResidueSite,
    residue_token,
    rfdiffusion_secondary_structure,
    terminal_cyclization_instance,
)


class RfDiffusionProcessor(Processor):
    r"""Lower one canonical design problem to RFdiffusion tensors.

    Args:
        config: Diffusion preprocessing options. Defaults to
            ``RfDiffusionProcessorConfig()``.
        device: Device for produced tensors.
        dtype: Floating-point dtype for produced tensors.
    """

    def __init__(
        self,
        config: RfDiffusionProcessorConfig | None = None,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        config = RfDiffusionProcessorConfig() if config is None else config
        self.config = config
        super().__init__(device=device, dtype=dtype)

    def __call__(
        self,
        input: MolecularSystem | Structure | str | PathLike[str] | None = None,
        *,
        placement: RfDiffusionPlacements | None = None,
        templates: Sequence[Template] = (),
        conditions: Sequence[
            AllowedResidueIdentities
            | Hotspot
            | RfDiffusionCoordination
            | RfDiffusionDistance
            | RfDiffusionPointGroupSymmetry
            | RfDiffusionRadiusOfGyration
            | RfDiffusionReferenceCoordinates
            | RfDiffusionSecondaryStructure
            | RfDiffusionSubstrateInteraction
            | SymbolicBond
        ] = (),
        inpaint_sequence: Sequence[ComponentReference] = (),
        length: tuple[int, int] | None = None,
        timestep: int = 50,
    ) -> RfDiffusionInput:
        r"""Lower base, complex, inpainting, active-site, or fold conditioning.

        Args:
            input: Optional context system, structure, or local structure path.
            placement: Output topology containing existing spans and generated protein segments.
            templates: Up to one fold-conditioning template.
            conditions: Sequence, hotspot, geometry, symmetry, and interaction conditions.
            inpaint_sequence: Existing residue occurrences whose identities are masked.
            length: Inclusive bounds for total generated length across all generated segments.
            timestep: Positive diffusion timestep stored on the lowered input.

        Returns:
            An ``RfDiffusionInput`` with aligned target, output, template, and condition axes.
        """

        if length is not None:
            minimum_length, maximum_length = (operator.index(bound) for bound in length)
            if minimum_length < 0 or maximum_length < minimum_length:
                raise ValueError("length bounds must be non-negative with min <= max")
            length = (minimum_length, maximum_length)
        template_values = tuple(templates)
        if len(template_values) > 1:
            raise ValueError("RFdiffusion accepts at most one fold-conditioning Template")
        if isinstance(input, str | PathLike):
            input = read_structure(Path(input).expanduser())
        timestep = operator.index(timestep)
        if timestep <= 0:
            raise ValueError("timestep must be positive")
        condition_values = tuple(conditions)
        for index, condition in enumerate(condition_values):
            if not isinstance(
                condition,
                (
                    AllowedResidueIdentities,
                    Hotspot,
                    RfDiffusionCoordination,
                    RfDiffusionDistance,
                    RfDiffusionPointGroupSymmetry,
                    RfDiffusionRadiusOfGyration,
                    RfDiffusionReferenceCoordinates,
                    RfDiffusionSecondaryStructure,
                    RfDiffusionSubstrateInteraction,
                    SymbolicBond,
                ),
            ):
                raise TypeError(f"conditions[{index}] is not a condition RFdiffusion can consume")
            if isinstance(condition, AllowedResidueIdentities):
                if len(condition.component_ids) != 1:
                    raise ValueError(
                        "RFdiffusion sequence conditioning represents one known residue identity or an "
                        "unknown residue, not a set of alternative identities"
                    )
                component_id = next(iter(condition.component_ids))
                if component_id not in NUM2AA[:MASK_TOKEN]:
                    raise ValueError("RFdiffusion AllowedResidueIdentities must name one residue in its alphabet")

        if isinstance(input, Structure):
            structure = input
            system: MolecularSystem | None = structure.system
        else:
            structure = None
            system = input
        if sum(isinstance(condition, RfDiffusionPointGroupSymmetry) for condition in condition_values) > 1:
            raise ValueError("RFdiffusion accepts at most one RfDiffusionPointGroupSymmetry condition")
        if placement is None and system is not None:
            default_placement: PlacementSpec = {}
            for instance_id, entity_id in system.instances.items():
                entity = system.entities[entity_id]
                if isinstance(entity, PolymerSequence) and entity.polymer_type == "protein":
                    default_placement[instance_id] = {instance_id: ExistingSpan(instance_id, 0, len(entity))}
            placement = RfDiffusionPlacements(placements=default_placement) if default_placement else None
        if placement is None:
            raise ValueError("RFdiffusion requires a placement when no protein input can define one")
        # Every placement_id must be globally unique across the whole PlacementSpec.
        validate_placement(placement.placements)
        for group in placement.placements.values():
            for part in group.values():
                if isinstance(part, GeneratedSegment):
                    if part.polymer_type != "protein":
                        raise ValueError("RFdiffusion generates protein backbones only")
                    if template_values and part.min_length != part.max_length:
                        raise ValueError(
                            "RFdiffusion fold-conditioned Template evidence requires fixed generated lengths"
                        )
                    continue
                if system is None:
                    raise ValueError("RFdiffusion ExistingSpan requires molecular input")
                entity = system.entities[system.instances[part.instance_id]]
                if not isinstance(entity, PolymerSequence) or entity.polymer_type != "protein":
                    raise ValueError("RFdiffusion ExistingSpan must reference a protein polymer")

        target_rows = _protein_rows(system)
        target_residues = tuple(row[0] for row in target_rows)
        target_index = {residue: index for index, residue in enumerate(target_residues)}
        target_coords, target_mask = _atom14_target(structure, target_rows)
        output_residues, output_instances = _resolve_placement(placement)
        # Keep the combined generated-length window distinct from the realized row count.
        combined_length = length
        output_length = len(output_residues)
        output_index = {reference: row for row, reference in enumerate(output_residues)}
        output_index.update(
            {alias: output_index[reference] for alias, reference in symbolic_residue_aliases(output_residues).items()}
        )
        inpainted_residues = set(inpaint_sequence)
        if inpainted_residues:
            if system is None:
                raise ValueError("inpaint_sequence requires a molecular input system")
            if not inpainted_residues <= set(output_index):
                raise ValueError("inpaint_sequence residues must belong to the RFdiffusion output placement")

        cyclic_mask = torch.zeros(output_length, dtype=torch.bool)
        for condition in condition_values:
            if not isinstance(condition, SymbolicBond):
                continue
            instance_id = terminal_cyclization_instance(
                condition,
                output_index,
                output_instances,
            )
            cyclic_mask |= torch.tensor([value == instance_id for value in output_instances])

        reference_sites = {
            site
            for condition in condition_values
            if isinstance(condition, RfDiffusionReferenceCoordinates)
            for site in condition.sites
        }
        if reference_sites and system is None:
            raise ValueError("RfDiffusionReferenceCoordinates requires a molecular input system")
        reference_system = cast(MolecularSystem, system)
        if any(
            isinstance(reference_system.entities[reference_system.instances[site.instance_id]], Molecule)
            for site in reference_sites
        ):
            raise ValueError("RFdiffusion RfDiffusionReferenceCoordinates must identify protein residue atoms")
        referenced_residues = {(site.instance_id, site.component_index) for site in reference_sites}
        for instance_id, component_index in referenced_residues:
            if (instance_id, component_index) not in output_index:
                raise ValueError("RfDiffusionReferenceCoordinates sites must belong to an RFdiffusion output residue")
            backbone = {
                site.atom_id
                for site in reference_sites
                if site.instance_id == instance_id and site.component_index == component_index
            }
            if not {"N", "CA", "C"} <= backbone:
                raise ValueError(
                    "RFdiffusion RfDiffusionReferenceCoordinates works at residue granularity and requires N/CA/C "
                    f"for {instance_id}:{component_index}"
                )
        fixed_structure = torch.tensor(
            [
                isinstance(reference, tuple)
                and any(
                    site.instance_id == reference[0] and site.component_index == reference[1]
                    for site in reference_sites
                )
                for reference in output_residues
            ],
            dtype=torch.bool,
        )
        guided_atoms: list[AtomSite | SymbolicAtomSite] = []
        for condition in condition_values:
            if isinstance(condition, RfDiffusionCoordination):
                for reference in (
                    *condition.side_a,
                    *condition.side_b,
                ):
                    if isinstance(reference, (AtomSite, SymbolicAtomSite)):
                        guided_atoms.append(reference)
            elif isinstance(condition, (RfDiffusionDistance, SymbolicBond)):
                guided_atoms.extend((condition.site_a, condition.site_b))
        structure_index = {} if structure is None else {site: index for index, site in enumerate(structure.sites)}
        for site in guided_atoms:
            if isinstance(site, SymbolicAtomSite):
                if site.atom_id not in {"N", "CA", "C"}:
                    raise ValueError("RFdiffusion guidance supports N/CA/C on generated residues")
                continue
            atom_site = site
            source_system = cast(MolecularSystem, system)
            residue = (
                None
                if isinstance(source_system.entities[source_system.instances[atom_site.instance_id]], Molecule)
                else (atom_site.instance_id, atom_site.component_index)
            )
            row = None if residue is None else output_index.get(residue)
            if row is not None and atom_site.atom_id in {"N", "CA", "C"}:
                continue
            if row is not None:
                entity = source_system.entities[source_system.instances[atom_site.instance_id]]
                atom_names = ATOM14_NAMES[residue_token(entity.components[atom_site.component_index].component_id)]
                if atom_site.atom_id not in atom_names or not bool(fixed_structure[row]):
                    raise ValueError("RFdiffusion guidance supports sidechain/O atoms only on fixed protein residues")
            source = structure_index.get(atom_site)
            source_structure = cast(Structure, structure)
            if source is None or not bool(source_structure.coordinate_mask[source]):
                raise ValueError(f"RFdiffusion guidance atom lacks an observed coordinate: {atom_site!r}")
        external_guidance: list[AtomSite] = []
        source_system = cast(MolecularSystem, system)
        for site in guided_atoms:
            if not isinstance(site, AtomSite):
                continue
            atom_site = site
            if (
                isinstance(source_system.entities[source_system.instances[atom_site.instance_id]], Molecule)
                or (atom_site.instance_id, atom_site.component_index) not in output_index
            ):
                external_guidance.append(atom_site)
        if external_guidance and not bool(fixed_structure.any()):
            raise ValueError(
                "guidance to atoms outside the RFdiffusion output requires fixed RfDiffusionReferenceCoordinates "
                "to define their coordinate frame"
            )
        fixed_sequence = torch.tensor(
            [isinstance(reference, tuple) and reference not in inpainted_residues for reference in output_residues],
            dtype=torch.bool,
        )
        tokens = torch.full((output_length,), MASK_TOKEN, dtype=torch.int64)
        for row, reference in enumerate(output_residues):
            if fixed_sequence[row]:
                component_reference = cast(tuple[str, int], reference)
                tokens[row] = residue_token(target_rows[target_index[component_reference]][1])
        conditioned_tokens: dict[int, int] = {}
        for condition in condition_values:
            if not isinstance(condition, AllowedResidueIdentities):
                continue
            component_id = next(iter(condition.component_ids))
            token = residue_token(component_id)
            identity_rows: list[int] = []
            for target in condition.positions:
                if isinstance(target, str):
                    identity_rows.extend(
                        row
                        for row, reference in enumerate(output_residues)
                        if isinstance(reference, SymbolicResidueSite) and reference.placement_id == target
                    )
                else:
                    identity_rows.append(output_index[target])
            for row in identity_rows:
                previous = conditioned_tokens.get(row)
                if previous is not None and previous != token:
                    raise ValueError("AllowedResidueIdentities conditions assign conflicting identities")
                conditioned_tokens[row] = token
                fixed_sequence[row] = True
                tokens[row] = token

        hotspots: set[tuple[str, int]] = set()
        for condition in condition_values:
            if not isinstance(condition, Hotspot):
                continue
            # ``Hotspot`` is single-sided and distance-free: its
            # ``residues`` name the EXISTING target residues to condition on and the generated binder
            # partner is implicit (the whole design). RFdiffusion realizes them as the per-token
            # ``hotspot_mask``; the interaction head consumes no distance cutoff.
            for hotspot_reference in condition.residues:
                if not isinstance(hotspot_reference, tuple) or hotspot_reference not in output_index:
                    raise ValueError("RFdiffusion hotspot residues must be existing residues in the output placement")
                hotspots.add(hotspot_reference)
        hotspot = torch.tensor(
            [isinstance(reference, tuple) and reference in hotspots for reference in output_residues]
        )
        if bool((hotspot & ~fixed_structure).any()):
            raise ValueError("RFdiffusion hotspot residues must have fixed RfDiffusionReferenceCoordinates")

        coords = torch.full((output_length, 14, 3), float("nan"))
        for output_row, reference in enumerate(output_residues):
            if not isinstance(reference, tuple):
                continue
            target_row = target_index[reference]
            present = target_mask[target_row]
            if fixed_structure[output_row]:
                coords[output_row] = float("nan")
                selected_atoms = {
                    site.atom_id
                    for site in reference_sites
                    if site.instance_id == reference[0] and site.component_index == reference[1]
                }
                atom_names = ATOM14_NAMES[residue_token(target_rows[target_row][1])]
                present = present & torch.tensor(
                    [atom_name in selected_atoms for atom_name in atom_names],
                    dtype=torch.bool,
                )
            coords[output_row, present] = target_coords[target_row, present]

        for condition in condition_values:
            if not isinstance(condition, RfDiffusionSubstrateInteraction):
                continue
            external_atoms: list[AtomSite] = []
            source_system = cast(MolecularSystem, system)
            for substrate_site in condition.substrate_atoms:
                if not isinstance(substrate_site, AtomSite):
                    continue
                atom_site = substrate_site
                if (
                    isinstance(source_system.entities[source_system.instances[atom_site.instance_id]], Molecule)
                    or (atom_site.instance_id, atom_site.component_index) not in output_index
                ):
                    external_atoms.append(atom_site)
            if external_atoms and not bool(fixed_structure.any()):
                raise ValueError(
                    "RfDiffusionSubstrateInteraction with external atoms requires fixed "
                    "RfDiffusionReferenceCoordinates to define the substrate frame"
                )

        secondary_structure, block_adjacency = _fold_features(
            template_values[0] if template_values else None,
            system,
            output_residues,
            output_instances,
        )
        secondary_labels = secondary_structure.argmax(dim=-1)
        label_by_kind = {"helix": 0, "strand": 1, "loop": 2}
        for condition in condition_values:
            if not isinstance(condition, RfDiffusionSecondaryStructure):
                continue
            secondary_rows: list[int] = []
            for secondary_target in condition.targets:
                if isinstance(secondary_target, str):
                    secondary_rows.extend(
                        row
                        for row, reference in enumerate(output_residues)
                        if isinstance(reference, SymbolicResidueSite) and reference.placement_id == secondary_target
                    )
                else:
                    secondary_rows.append(output_index[cast(tuple[str, int] | SymbolicResidueSite, secondary_target)])
            row_tensor = torch.tensor(secondary_rows, dtype=torch.int64)
            if bool(fixed_structure[row_tensor].any()):
                raise ValueError("RfDiffusionSecondaryStructure targets must be structure-inpainted, not fixed")
            label = label_by_kind[condition.kind]
            existing = secondary_labels[row_tensor]
            if bool(((existing != 3) & (existing != label)).any()):
                raise ValueError("RfDiffusionSecondaryStructure contradicts template evidence")
            secondary_labels[row_tensor] = label
        secondary_structure = F.one_hot(secondary_labels, num_classes=4).to(torch.float32)
        residue_indices: list[int] = []
        chain_index = -1
        previous_instance = None
        chain_start = 0
        for row, instance_id in enumerate(output_instances):
            if instance_id != previous_instance:
                chain_index += 1
                chain_start = row
                previous_instance = instance_id
            residue_indices.append(200 * chain_index + row - chain_start)
        residue_index = torch.tensor(residue_indices, dtype=torch.int64)
        return RfDiffusionInput(
            noisy_coords=coords.unsqueeze(0).to(device=self.device, dtype=self.dtype),
            sequence=F.one_hot(tokens, num_classes=22).to(device=self.device, dtype=self.dtype).unsqueeze(0),
            residue_index=residue_index.to(self.device).unsqueeze(0),
            fixed_structure_mask=fixed_structure.to(self.device).unsqueeze(0),
            fixed_sequence_mask=fixed_sequence.to(self.device).unsqueeze(0),
            hotspot_mask=hotspot.to(self.device).unsqueeze(0),
            cyclic_mask=cyclic_mask.to(self.device).unsqueeze(0),
            secondary_structure=secondary_structure.to(device=self.device, dtype=self.dtype).unsqueeze(0),
            block_adjacency=block_adjacency.to(device=self.device, dtype=self.dtype).unsqueeze(0),
            timestep=torch.tensor([timestep], dtype=torch.int64, device=self.device),
            layout=RfDiffusionLayout(
                system=system,
                structure=structure,
                templates=template_values,
                placement=placement,
                conditions=condition_values,
                output_residues=output_residues,
                output_instance_ids=output_instances,
                combined_length=combined_length,
            ),
        )


def _protein_rows(
    system: MolecularSystem | None,
) -> tuple[tuple[tuple[str, int], str], ...]:
    if system is None:
        return ()
    rows: list[tuple[tuple[str, int], str]] = []
    for instance_id, entity_id in system.instances.items():
        entity = system.entities[entity_id]
        if not isinstance(entity, PolymerSequence) or entity.polymer_type != "protein":
            continue
        rows.extend(((instance_id, index), component.component_id) for index, component in enumerate(entity.components))
    return tuple(rows)


def _atom14_target(
    structure: Structure | None,
    rows: tuple[tuple[tuple[str, int], str], ...],
) -> tuple[Tensor, Tensor]:
    coords = torch.zeros((len(rows), 14, 3), dtype=torch.float32)
    mask = torch.zeros((len(rows), 14), dtype=torch.bool)
    if structure is None:
        return coords, mask
    source_index = {site: index for index, site in enumerate(structure.sites)}
    source_coordinates = torch.tensor(structure.coordinates, dtype=torch.float32)
    for row_index, (residue, component_id) in enumerate(rows):
        atom_names = ATOM14_NAMES[residue_token(component_id)]
        for atom_index, atom_name in enumerate(atom_names):
            if atom_name is None:
                continue
            source = source_index.get(AtomSite(residue[0], residue[1], atom_name))
            if source is None or not bool(structure.coordinate_mask[source]):
                continue
            coords[row_index, atom_index] = source_coordinates[source]
            mask[row_index, atom_index] = True
    return coords, mask


def _resolve_placement(
    placement: RfDiffusionPlacements,
) -> tuple[
    tuple[tuple[str, int] | SymbolicResidueSite, ...],
    tuple[str, ...],
]:
    residues: list[tuple[str, int] | SymbolicResidueSite] = []
    instance_ids: list[str] = []
    # Mint realized output instance IDs with the shared `InstanceIdAllocator`, walking declaration groups in
    # order and calling `allocate()` once per realized output chain with >=1 occurrence — the
    # canonical `chain_{n}` IDs (uniform and 0-based). A declaration group that
    # realizes to zero residues mints no chain and consumes no index. Identical to the
    # post-sample re-lowering in `RfDiffusionModel._sample_variable_lengths`. Numerically inert:
    # `residue_index` is computed from chain boundaries below, not from this id string.
    allocator = InstanceIdAllocator()
    for group in placement.placements.values():
        group_residues: list[tuple[str, int] | SymbolicResidueSite] = []
        for placement_id, part in group.items():
            if isinstance(part, ExistingSpan):
                group_references: list[tuple[str, int] | SymbolicResidueSite] = [
                    (part.instance_id, index) for index in range(part.start, part.stop)
                ]
            else:
                # Processor lowering is deterministic. Variable lengths are sampled by
                # RfDiffusionModel.design, where the user RNG is available.
                group_references = [SymbolicResidueSite(placement_id, index) for index in range(part.min_length)]
            group_residues.extend(group_references)
        if not group_residues:
            continue  # zero-length declaration group mints no realized chain, consumes no index
        instance_id = allocator.allocate()
        residues.extend(group_residues)
        instance_ids.extend([instance_id] * len(group_residues))
    return tuple(residues), tuple(instance_ids)


def _fold_features(
    template: Template | None,
    system: MolecularSystem | None,
    output_residues: tuple[tuple[str, int] | SymbolicResidueSite, ...],
    output_instances: tuple[str, ...],
) -> tuple[Tensor, Tensor]:
    length = len(output_residues)
    if template is None:
        secondary = F.one_hot(torch.full((length,), 3), num_classes=4).to(torch.float32)
        adjacency = F.one_hot(torch.full((length, length), 2), num_classes=3).to(torch.float32)
        return secondary, adjacency
    if system is None:
        raise ValueError("RFdiffusion Template alignments require a molecular input system")

    output_index = {
        cast(tuple[str, int], reference): row
        for row, reference in enumerate(output_residues)
        if isinstance(reference, tuple)
    }
    output_rows: list[int] = []
    aligned_coords: list[Tensor] = []
    aligned_masks: list[Tensor] = []
    aligned_labels: list[Tensor] = []
    seen_output_rows: set[int] = set()
    template_features: dict[str, tuple[Tensor, Tensor, Tensor]] = {}

    for alignment in template.alignments:
        target_entity = system.entities.get(alignment.target_entity_id)
        if not isinstance(target_entity, PolymerSequence) or target_entity.polymer_type != "protein":
            raise ValueError("RFdiffusion Template target_entity_id must name a protein entity")
        if alignment.target_instance_id is None:
            target_instances = tuple(
                instance_id
                for instance_id, entity_id in system.instances.items()
                if entity_id == alignment.target_entity_id
            )
            if not target_instances:
                raise ValueError("RFdiffusion Template alignment does not resolve to a target instance")
        else:
            actual_entity_id = system.instances.get(alignment.target_instance_id)
            if actual_entity_id != alignment.target_entity_id:
                raise ValueError("RFdiffusion Template target_instance_id does not belong to target_entity_id")
            target_instances = (alignment.target_instance_id,)

        features = template_features.get(alignment.template_instance_id)
        if features is None:
            template_entity_id = template.structure.system.instances[alignment.template_instance_id]
            template_entity = cast(
                PolymerSequence,
                template.structure.system.entities[template_entity_id],
            )
            template_rows = tuple(
                (
                    (alignment.template_instance_id, template_index),
                    component.component_id,
                )
                for template_index, component in enumerate(template_entity.components)
            )
            template_coords, template_mask = _atom14_target(template.structure, template_rows)
            template_labels = torch.full((len(template_rows),), 2, dtype=torch.int64)
            observed_backbone = template_mask[:, :3].all(dim=-1)
            start = 0
            while start < len(template_rows):
                if not bool(observed_backbone[start]):
                    start += 1
                    continue
                stop = start + 1
                while stop < len(template_rows) and bool(observed_backbone[stop]):
                    stop += 1
                template_labels[start:stop] = rfdiffusion_secondary_structure(template_coords[start:stop, 1])
                start = stop
            features = template_coords, template_mask, template_labels
            template_features[alignment.template_instance_id] = features
        template_coords, template_mask, template_labels = features

        for target_index, template_index in alignment.residue_pairs:
            if target_index >= len(target_entity):
                raise ValueError("RFdiffusion Template target residue is outside target_entity_id")
            for target_instance in target_instances:
                output_row = output_index.get((target_instance, target_index))
                if output_row is None:
                    raise ValueError("RFdiffusion Template alignment target is absent from the output placement")
                if output_row in seen_output_rows:
                    raise ValueError("RFdiffusion Template alignments overlap on one output residue")
                seen_output_rows.add(output_row)
                output_rows.append(output_row)
                aligned_coords.append(template_coords[template_index])
                aligned_masks.append(template_mask[template_index])
                aligned_labels.append(template_labels[template_index])

    if not output_rows:
        raise ValueError("RFdiffusion Template alignments do not cover any output residue")
    coords = torch.stack(aligned_coords)
    mask = torch.stack(aligned_masks)
    if not bool(mask[:, :3].all()):
        raise ValueError("RFdiffusion Template requires complete aligned N/CA/C coordinates")

    secondary_labels = torch.full((length,), 3, dtype=torch.int64)
    output_row_tensor = torch.tensor(output_rows, dtype=torch.int64)
    secondary_labels[output_row_tensor] = torch.stack(aligned_labels)
    secondary = F.one_hot(secondary_labels, num_classes=4).to(torch.float32)
    nitrogen, calpha, carbon = coords[:, :3].unbind(dim=1)
    ca_to_n = calpha - nitrogen
    c_to_ca = carbon - calpha
    cbeta = -0.57910144 * torch.cross(ca_to_n, c_to_ca, dim=-1) + 0.5689693 * ca_to_n - 0.5441217 * c_to_ca + calpha
    distances = torch.cdist(cbeta, cbeta)
    mapped_adjacent = distances < 6.0
    mapped_adjacent.fill_diagonal_(False)
    adjacent = torch.zeros((length, length), dtype=torch.bool)
    adjacent[output_row_tensor[:, None], output_row_tensor[None, :]] = mapped_adjacent
    block = torch.zeros((length, length), dtype=torch.int64)
    segments: list[tuple[int, int, int]] = []
    start = 0
    for index in range(1, length + 1):
        if (
            index == length
            or secondary_labels[index] != secondary_labels[start]
            or output_instances[index] != output_instances[start]
        ):
            segments.append((int(secondary_labels[start]), start, index))
            start = index
    for left, (left_label, left_start, left_stop) in enumerate(segments):
        if left_label == 2:
            continue
        for right_label, right_start, right_stop in segments[left + 1 :]:
            if right_label == 2:
                continue
            if bool(adjacent[left_start:left_stop, right_start:right_stop].any()):
                block[left_start:left_stop, right_start:right_stop] = 1
                block[right_start:right_stop, left_start:left_stop] = 1
    known = secondary_labels != 3
    block = block.masked_fill(~(known[:, None] & known[None, :]), 2)
    return secondary, F.one_hot(block, num_classes=3).to(torch.float32)


def from_contig_string(
    contigs: str,
    *,
    source_instances: Mapping[str, int] | None = None,
) -> tuple[PlacementSpec, tuple[int, int] | None]:
    r"""Parse an RFdiffusion ``ContigMap.contigs`` string into a core ``PlacementSpec``.

    ``contigs`` is the space/``/``-delimited DSL, e.g. ``"10-40/A163-181/10-40 B1-100/0"``.
    Each space-separated block becomes one declaration group (``chain_1``, ``chain_2``,
    …); each ``/``-subcon becomes one placement:

    * ``A163-181`` → ``ExistingSpan("A", 163, 182)`` (inclusive → half-open), ``A5`` →
      ``ExistingSpan("A", 5, 6)``, a bare ``A`` → the whole source chain
      ``ExistingSpan("A", 0, source_instances["A"])``;
    * ``10-40`` → ``GeneratedSegment("protein", 10, 40)``; ``10`` → ``min == max == 10``;
      ``0`` → ``min == max == 0`` (chain break / receptor terminator).

    Returns ``(PlacementSpec, combined_length)``. ``combined_length`` is the RFdiffusion
    ``ContigMap.length`` argument: the combined generated-length bounds ``(Σ min, Σ max)``
    across every generated subcon, excluding ``ExistingSpan`` rows. It is stored on
    :class:`RfDiffusionLayout.combined_length` and is ``None`` when the string generates
    nothing. ``source_instances`` (source chain-letter →
    length) resolves the bare-chain ``A`` form to ``ExistingSpan("A", 0, length)``; explicit
    ``A163-181`` extents are carried literally rather than rebased to zero-based component indices.
    Lossless into ``(PlacementSpec, combined_length)``
    (the precise reference/output permutation, when needed, projects to ordered length-1
    spans).
    """

    placement: PlacementSpec = {}
    generated_min = 0
    generated_max = 0
    has_generated = False
    blocks = contigs.split()
    for block_index, block in enumerate(blocks):
        group_key = f"chain_{block_index + 1}"
        group: dict[str, PlacementValue] = {}
        for sub_index, raw in enumerate(block.split("/")):
            sub = raw.strip()
            if not sub:
                continue
            placement_id = f"{group_key}:{sub_index}"
            if sub[0].isalpha():
                split = 0
                while split < len(sub) and sub[split].isalpha():
                    split += 1
                instance_id = sub[:split]
                extent = sub[split:]
                if extent == "":
                    if source_instances is None or instance_id not in source_instances:
                        raise ValueError(f"bare source chain {instance_id!r} needs its length in source_instances")
                    start, stop = 0, source_instances[instance_id]
                else:
                    bounds = extent.split("-")
                    if len(bounds) == 1:
                        start = stop = int(bounds[0])
                    elif len(bounds) == 2:
                        start, stop = int(bounds[0]), int(bounds[1])
                    else:
                        raise ValueError(f"malformed existing subcon {sub!r}")
                    stop += 1  # inclusive bound -> half-open stop
                if start < 0 or stop < start:
                    raise ValueError(f"malformed existing subcon {sub!r}")
                # RFdiffusion ``contig`` residue numbers are external PDB labels and are carried
                # literally into the ExistingSpan
                # ``A163-181 -> ExistingSpan("A", 163, 182)``), so no upper-bound check against
                # source length is applied — the numbering base is not the core's 0-based index.
                group[placement_id] = ExistingSpan(instance_id, start, stop)
            else:
                bounds = sub.split("-")
                if len(bounds) == 1:
                    low = high = int(bounds[0])
                elif len(bounds) == 2:
                    low, high = int(bounds[0]), int(bounds[1])
                else:
                    raise ValueError(f"malformed generated subcon {sub!r}")
                if low < 0 or high < low:
                    raise ValueError(f"malformed generated subcon {sub!r}")
                group[placement_id] = GeneratedSegment("protein", low, high)
                generated_min += low
                generated_max += high
                has_generated = True
        placement[group_key] = group
    # Reject duplicate placement IDs at construction.
    derive_instance_of(placement)
    combined_length = (generated_min, generated_max) if has_generated else None
    return placement, combined_length


def to_contig_string(
    placement: PlacementSpec,
) -> str:
    r"""Re-emit an RFdiffusion ``contigs`` string from a core ``PlacementSpec``.

    Each declaration group → one space-separated block; each placement → one ``/``-subcon
    (``ExistingSpan("A", 163, 182)`` → ``A163-181`` half-open → inclusive;
    ``GeneratedSegment(min, max)`` → ``min-max`` or the bare ``min`` when fixed, so a
    zero-length placement emits ``0``). The combined-length setting is not part of
    the ``contigs`` grammar. The re-emission is lossless for TOPOLOGY (fixed vs
    generated, order, lengths) and lossy for RF-index spacing and receptor-vs-inpaint role,
    which the contig string does not encode.
    """

    blocks: list[str] = []
    for group in placement.values():
        subcons: list[str] = []
        for value in group.values():
            if isinstance(value, ExistingSpan):
                subcons.append(f"{value.instance_id}{value.start}-{value.stop - 1}")
            else:
                if value.min_length == value.max_length:
                    subcons.append(str(value.min_length))
                else:
                    subcons.append(f"{value.min_length}-{value.max_length}")
        blocks.append("/".join(subcons))
    return " ".join(blocks)
