r"""Lower canonical molecular inputs and Conditions for RFdiffusion."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from os import PathLike
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
from typing_extensions import Self

# AllowedResidueIdentities stays shared: it is the frozen §3.7.1 identity record.
from ...data.conditions import AllowedResidueIdentities
from ...data.evidence import MSA, MSACollection, Template
from ...data.molecular import (
    AtomSite,
    ComponentReference,
    MolecularSystem,
    Molecule,
    PolymerSequence,
)
from ...data.placement import (
    InstanceIdAllocator,
    PlacementMap,
    PlacementSpec,
    PlacementValue,
)
from ...data.placement import instance_of as derive_instance_of
from ...data.placement import validate_placement
from ...data.structure import Structure
from ...io.structure import read_structure
from .chemistry import ATOM14_NAMES, MASK_TOKEN, NUM2AA, residue_token
from .config import RFdiffusionProcessorConfig

# Every record below is a family-owned RFdiffusion spatial/placement condition, re-homed
# family-locally in .conditions after the data-model migration. §3.7.2 keeps the positive API
# family-owned and still under investigation. The covalent-link and target-conditioning records
# are now the FROZEN §3.7.4 core imported through .conditions: the frozen ``SymbolicBond`` (the
# covalent link; the processor collapses a terminal N->C bond to ``cyclic_mask``), the frozen
# distance-free single-sided ``Hotspot`` (owner verdict D3; realized as ``hotspot_mask``), and the
# frozen geometric ``Contact`` (carried to the layout for the family scorer; no released input-side
# potential). The requested covalent link is spelled directly as ``SymbolicBond`` (no alias).
from .model import (
    Contact,
    Coordination,
    Distance,
    ExistingSpan,
    GeneratedSegment,
    Hotspot,
    PointGroupSymmetry,
    RadiusOfGyration,
    ReferenceCoordinates,
    RFdiffusionInput,
    RFdiffusionLayout,
    RFdiffusionPlacements,
    SecondaryStructure,
    SubstrateInteraction,
    SymbolicAtomSite,
    SymbolicBond,
    SymbolicResidueSite,
    symbolic_residue_aliases,
)


class RFdiffusionProcessor:
    r"""Lower one canonical design problem to RFdiffusion tensors."""

    config_class = RFdiffusionProcessorConfig

    def __init__(
        self,
        config: RFdiffusionProcessorConfig | None = None,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        config = RFdiffusionProcessorConfig() if config is None else config
        self.config = config
        self.device = torch.device("cpu" if device is None else device)
        self.dtype = torch.float32 if dtype is None else dtype

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Self:
        r"""Update runtime placement in place and return this Processor."""

        if device is not None:
            self.device = torch.device(device)
        if dtype is not None:
            self.dtype = dtype
        return self

    def __call__(
        self,
        input: MolecularSystem | Structure | str | PathLike[str] | None = None,
        *,
        msa: MSA | MSACollection | str | PathLike[str] | None = None,
        templates: Sequence[Template | str | PathLike[str]] = (),
        conditions: Sequence[
            AllowedResidueIdentities
            | Contact
            | Hotspot
            | RFdiffusionPlacements
            | Coordination
            | Distance
            | PointGroupSymmetry
            | RadiusOfGyration
            | ReferenceCoordinates
            | SecondaryStructure
            | SubstrateInteraction
            | SymbolicBond
        ] = (),
        inpaint_sequence: Sequence[ComponentReference] = (),
        length: tuple[int, int] | None = None,
        timestep: int = 50,
    ) -> RFdiffusionInput:
        r"""Lower base, complex, inpainting, active-site, or fold conditioning.

        ``length`` is the family-owned §3.7 topology argument (RFdiffusion ``ContigMap.length``):
        the combined generated-length bounds across all inpaint chains, excluding receptor
        ``ExistingSpan`` rows. It is a concrete family operation argument (distinct from ODesign
        per-chain lengths and BoltzGen ``total_len``), NOT a shared placement-core invariant; it is
        threaded onto :class:`RFdiffusionLayout` so ``_sample_variable_lengths`` can honor the
        cross-segment total when drawing per-``GeneratedSegment`` lengths.
        """

        if msa is not None:
            raise ValueError("RFdiffusion does not consume MSA evidence")
        if length is not None:
            if (
                type(length) is not tuple
                or len(length) != 2
                or any(type(bound) is not int for bound in length)
                or any(bound < 0 for bound in length)
                or length[0] > length[1]
            ):
                raise ValueError("length must be a (min, max) tuple of non-negative ints with min <= max")
        template_values = tuple(templates)
        if len(template_values) > 1:
            raise ValueError("RFdiffusion accepts at most one fold-conditioning Template")
        if template_values and type(template_values[0]) is not Template:
            raise TypeError("RFdiffusion fold conditioning requires an explicit Template with a canonical alignment")
        if isinstance(input, str | PathLike):
            input = read_structure(Path(input).expanduser())
        if input is not None and type(input) not in (MolecularSystem, Structure):
            raise TypeError("input must be a MolecularSystem, Structure, local path, or None")
        if type(timestep) is not int or timestep <= 0:
            raise ValueError("timestep must be a positive integer")
        condition_values = tuple(conditions)
        for index, condition in enumerate(condition_values):
            if type(condition) not in (
                AllowedResidueIdentities,
                Contact,
                Hotspot,
                RFdiffusionPlacements,
                Coordination,
                Distance,
                PointGroupSymmetry,
                RadiusOfGyration,
                ReferenceCoordinates,
                SecondaryStructure,
                SubstrateInteraction,
                SymbolicBond,
            ):
                raise TypeError(f"conditions[{index}] is not a condition RFdiffusion can consume")
            if type(condition) is AllowedResidueIdentities:
                if len(condition.component_ids) != 1:
                    raise ValueError(
                        "RFdiffusion sequence conditioning represents one known residue identity or an "
                        "unknown residue, not a set of alternative identities"
                    )
                component_id = next(iter(condition.component_ids))
                if component_id not in NUM2AA[:MASK_TOKEN]:
                    raise ValueError(
                        "RFdiffusion AllowedResidueIdentities must name one residue in its released alphabet"
                    )
            if type(condition) is Distance and condition.intent != "soft":
                raise ValueError(f"RFdiffusion consumes {type(condition).__name__} only with intent='soft'")
            if type(condition) is ReferenceCoordinates and condition.intent != "hard":
                raise ValueError("RFdiffusion consumes ReferenceCoordinates only with intent='hard'")
            if type(condition) is PointGroupSymmetry and condition.intent != "hard":
                raise ValueError("RFdiffusion PointGroupSymmetry must have intent='hard'")

        system = input.system if type(input) is Structure else input
        structure = input if type(input) is Structure else None
        placements = tuple(condition for condition in condition_values if type(condition) is RFdiffusionPlacements)
        if len(placements) > 1:
            raise ValueError("RFdiffusion accepts exactly one RFdiffusionPlacements condition")
        if sum(type(condition) is PointGroupSymmetry for condition in condition_values) > 1:
            raise ValueError("RFdiffusion accepts at most one PointGroupSymmetry condition")
        if placements:
            placement = placements[0]
        elif system is not None:
            placements: dict[str, dict[str, ExistingSpan]] = {}
            for instance_id, entity_id in system.instances.items():
                entity = system.entities[entity_id]
                if type(entity) is PolymerSequence and entity.polymer_type == "protein":
                    placements[instance_id] = {instance_id: ExistingSpan(instance_id, 0, len(entity))}
            placement = RFdiffusionPlacements(placements=placements) if placements else None
        else:
            placement = None
        if placement is None:
            raise ValueError("input=None requires one RFdiffusionPlacements condition with GeneratedSegment values")
        # Frozen §3.7: every placement_id must be globally unique across the whole PlacementSpec.
        validate_placement(placement.placements)
        for group in placement.placements.values():
            for part in group.values():
                if type(part) is GeneratedSegment:
                    if part.polymer_type != "protein":
                        raise ValueError("RFdiffusion generates protein backbones only")
                    if (
                        template_values
                        and template_values[0].target_instance_id is None  # type: ignore[union-attr]
                        and part.min_length != part.max_length
                    ):
                        raise ValueError(
                            "RFdiffusion fold-conditioned Template evidence requires fixed generated lengths"
                        )
                    continue
                if system is None:
                    raise ValueError("RFdiffusion ExistingSpan requires molecular input")
                entity = system.entities[system.instances[part.instance_id]]
                if type(entity) is not PolymerSequence or entity.polymer_type != "protein":
                    raise ValueError("RFdiffusion ExistingSpan must reference a protein polymer")

        target_rows = _protein_rows(system)
        target_residues = tuple(row[0] for row in target_rows)
        target_index = {residue: index for index, residue in enumerate(target_residues)}
        target_coords, target_mask = _atom14_target(structure, target_rows)
        output_residues, output_instances = _resolve_placement(placement)
        # Preserve the caller's family-owned (min, max) combined-length tuple (§3.7)
        # before rebinding ``length`` to the concrete residue count below; the Layout
        # carries the tuple, not the int.
        combined_length = length
        length = len(output_residues)
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

        cyclic_mask = torch.zeros(length, dtype=torch.bool)
        for condition in condition_values:
            if type(condition) is not SymbolicBond:
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
            if type(condition) is ReferenceCoordinates
            for site in condition.sites
        }
        if any(
            type(system.entities[system.instances[site.instance_id]]) is Molecule  # type: ignore[union-attr]
            for site in reference_sites
        ):
            raise ValueError("RFdiffusion ReferenceCoordinates must identify protein residue atoms")
        referenced_residues = {
            (site.instance_id, site.component_index)
            for site in reference_sites
            if type(system.entities[system.instances[site.instance_id]]) is not Molecule  # type: ignore[union-attr]
        }
        for instance_id, component_index in referenced_residues:
            if (instance_id, component_index) not in output_index:
                raise ValueError("ReferenceCoordinates sites must belong to an RFdiffusion output residue")
            backbone = {
                site.atom_id
                for site in reference_sites
                if site.instance_id == instance_id and site.component_index == component_index
            }
            if not {"N", "CA", "C"} <= backbone:
                raise ValueError(
                    "RFdiffusion ReferenceCoordinates works at residue granularity and requires N/CA/C "
                    f"for {instance_id}:{component_index}"
                )
        fixed_structure = torch.tensor(
            [
                type(reference) is tuple
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
            if type(condition) is Coordination:
                guided_atoms.extend(
                    reference
                    for reference in (*condition.side_a, *condition.side_b)
                    if type(reference) in (AtomSite, SymbolicAtomSite)
                )
            elif (type(condition) is Distance and condition.intent == "soft") or type(condition) is SymbolicBond:
                guided_atoms.extend((condition.site_a, condition.site_b))
        structure_index = {} if structure is None else {site: index for index, site in enumerate(structure.sites)}
        for site in guided_atoms:
            if type(site) is SymbolicAtomSite:
                if site.atom_id not in {"N", "CA", "C"}:
                    raise ValueError("RFdiffusion guidance supports N/CA/C on generated residues")
                continue
            residue = (
                None
                if type(system.entities[system.instances[site.instance_id]]) is Molecule  # type: ignore[union-attr]
                else (site.instance_id, site.component_index)
            )
            row = None if residue is None else output_index.get(residue)
            if row is not None and site.atom_id in {"N", "CA", "C"}:
                continue
            if row is not None:
                entity = system.entities[system.instances[site.instance_id]]  # type: ignore[union-attr]
                atom_names = ATOM14_NAMES[residue_token(entity.components[site.component_index].component_id)]
                if site.atom_id not in atom_names or not bool(fixed_structure[row]):
                    raise ValueError("RFdiffusion guidance supports sidechain/O atoms only on fixed protein residues")
            source = structure_index.get(site)
            if source is None or not bool(structure.coordinate_mask[source]):  # type: ignore[union-attr]
                raise ValueError(f"RFdiffusion guidance atom lacks an observed coordinate: {site!r}")
        external_guidance = [
            site
            for site in guided_atoms
            if type(site) is AtomSite
            and (
                type(system.entities[system.instances[site.instance_id]]) is Molecule  # type: ignore[union-attr]
                or (site.instance_id, site.component_index) not in output_index
            )
        ]
        if external_guidance and not bool(fixed_structure.any()):
            raise ValueError(
                "guidance to atoms outside the RFdiffusion output requires fixed ReferenceCoordinates "
                "to define their coordinate frame"
            )
        fixed_sequence = torch.tensor(
            [type(reference) is tuple and reference not in inpainted_residues for reference in output_residues],
            dtype=torch.bool,
        )
        tokens = torch.full((length,), MASK_TOKEN, dtype=torch.int64)
        for row, reference in enumerate(output_residues):
            if fixed_sequence[row]:
                tokens[row] = residue_token(target_rows[target_index[reference]][1])
        conditioned_tokens: dict[int, int] = {}
        for condition in condition_values:
            if type(condition) is not AllowedResidueIdentities:
                continue
            component_id = next(iter(condition.component_ids))
            token = residue_token(component_id)
            rows: list[int] = []
            for target in condition.positions:
                if type(target) is str:
                    rows.extend(
                        row
                        for row, reference in enumerate(output_residues)
                        if type(reference) is SymbolicResidueSite and reference.placement_id == target
                    )
                else:
                    rows.append(output_index[target])
            for row in rows:
                previous = conditioned_tokens.get(row)
                if previous is not None and previous != token:
                    raise ValueError("AllowedResidueIdentities conditions assign conflicting identities")
                conditioned_tokens[row] = token
                fixed_sequence[row] = True
                tokens[row] = token

        hotspots: set[tuple[str, int]] = set()
        for condition in condition_values:
            if type(condition) is not Hotspot:
                continue
            # Frozen §3.7.4 ``Hotspot`` is single-sided and distance-free (owner verdict D3): its
            # ``residues`` name the EXISTING target residues to condition on and the generated binder
            # partner is implicit (the whole design). RFdiffusion realizes them as the per-token
            # ``hotspot_mask``; the released ppi head reads no cutoff, so ``Hotspot`` carries none.
            for reference in condition.residues:
                if type(reference) is not tuple or reference not in output_index:
                    raise ValueError("RFdiffusion hotspot residues must be existing residues in the output placement")
                hotspots.add(reference)
        hotspot = torch.tensor([type(reference) is tuple and reference in hotspots for reference in output_residues])
        if bool((hotspot & ~fixed_structure).any()):
            raise ValueError("RFdiffusion hotspot residues must have fixed ReferenceCoordinates")

        coords = torch.full((length, 14, 3), float("nan"))
        for output_row, reference in enumerate(output_residues):
            if type(reference) is not tuple:
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
            if type(condition) is not SubstrateInteraction:
                continue
            external_atoms = [
                site
                for site in condition.substrate_atoms
                if type(system.entities[system.instances[site.instance_id]]) is Molecule  # type: ignore[union-attr]
                or (site.instance_id, site.component_index) not in output_index
            ]
            if external_atoms and not bool(fixed_structure.any()):
                raise ValueError(
                    "SubstrateInteraction with external atoms requires fixed ReferenceCoordinates to "
                    "define the substrate frame"
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
            if type(condition) is not SecondaryStructure:
                continue
            rows: list[int] = []
            for target in condition.targets:
                if type(target) is str:
                    rows.extend(
                        row
                        for row, reference in enumerate(output_residues)
                        if type(reference) is SymbolicResidueSite and reference.placement_id == target
                    )
                else:
                    rows.append(output_index[target])
            row_tensor = torch.tensor(rows, dtype=torch.int64)
            if bool(fixed_structure[row_tensor].any()):
                raise ValueError("SecondaryStructure targets must be structure-inpainted, not fixed")
            label = label_by_kind[condition.kind]
            existing = secondary_labels[row_tensor]
            if bool(((existing != 3) & (existing != label)).any()):
                raise ValueError("SecondaryStructure contradicts template evidence")
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
        return RFdiffusionInput(
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
            layout=RFdiffusionLayout(
                system=system,
                structure=structure,
                templates=template_values,  # type: ignore[arg-type]
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
        if type(entity) is not PolymerSequence or entity.polymer_type != "protein":
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
    placement: RFdiffusionPlacements,
) -> tuple[
    tuple[tuple[str, int] | SymbolicResidueSite, ...],
    tuple[str, ...],
]:
    residues: list[tuple[str, int] | SymbolicResidueSite] = []
    instance_ids: list[str] = []
    # Frozen §3.7 canonical realized-instance-id convention (W6 caveat b): mint the realized
    # OUTPUT instance ids with the shared `InstanceIdAllocator`, walking declaration groups in
    # order and calling `allocate()` once per realized output chain with >=1 occurrence — the
    # canonical `chain_{n}` (uniform, 0-based, no family prefix). A declaration group that
    # realizes to zero residues mints no chain and consumes no index. Identical to the
    # post-sample re-lowering in `RFdiffusionModel._sample_variable_lengths`. Numerically inert:
    # `residue_index` is computed from chain boundaries below, not from this id string.
    allocator = InstanceIdAllocator()
    for group in placement.placements.values():
        group_residues: list[tuple[str, int] | SymbolicResidueSite] = []
        for placement_id, part in group.items():
            if type(part) is ExistingSpan:
                references = [(part.instance_id, index) for index in range(part.start, part.stop)]
            else:
                # Processor lowering is deterministic. Variable lengths are sampled by
                # RFdiffusionModel.design, where the user RNG is available.
                references = [SymbolicResidueSite(placement_id, index) for index in range(part.min_length)]
            group_residues.extend(references)
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
    if template.target_instance_id is not None:
        if system is None:
            raise ValueError("Template.target_instance_id requires a molecular input system")
        if system.entities[system.instances[template.target_instance_id]] != template.target:
            raise ValueError("Template.target does not match target_instance_id in the input system")

    generated_rows = [row for row, reference in enumerate(output_residues) if type(reference) is SymbolicResidueSite]
    generated_target_rows = generated_rows if template.target_instance_id is None else []
    if generated_target_rows and len(generated_target_rows) != len(template.target):
        raise ValueError(
            "RFdiffusion fold conditioning requires the generated scaffold length to match Template.target"
        )
    if len({output_instances[row] for row in generated_target_rows}) > 1:
        raise ValueError("one Template.target cannot describe multiple generated chains")

    target_index_by_output: dict[int, int] = {}
    if generated_target_rows:
        target_index_by_output.update({row: target_index for target_index, row in enumerate(generated_target_rows)})
    if template.target_instance_id is not None:
        target_index_by_output.update(
            {
                row: reference[1]
                for row, reference in enumerate(output_residues)
                if type(reference) is tuple and reference[0] == template.target_instance_id
            }
        )
    if not target_index_by_output:
        raise ValueError("Template.target does not identify any RFdiffusion output residue")

    alignment = dict(template.alignment)
    template_entity = template.structure.system.entities[
        template.structure.system.instances[template.template_instance_id]
    ]
    mapped: list[tuple[int, int, int]] = []
    for output_row, target_index in sorted(target_index_by_output.items()):
        template_index = alignment.get(target_index)
        if template_index is None:
            continue
        mapped.append((output_row, target_index, template_index))
    if not mapped:
        raise ValueError("Template.alignment does not cover any RFdiffusion output residue")
    template_rows = tuple(
        (
            (template.template_instance_id, template_index),
            component.component_id,
        )
        for template_index, component in enumerate(template_entity.components)
    )
    template_coords, template_mask = _atom14_target(template.structure, template_rows)
    template_indices = torch.tensor([value[2] for value in mapped], dtype=torch.int64)
    coords = template_coords[template_indices]
    mask = template_mask[template_indices]
    if not bool(mask[:, :3].all()):
        raise ValueError("RFdiffusion Template requires complete aligned N/CA/C coordinates")

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
    secondary_labels = torch.full((length,), 3, dtype=torch.int64)
    output_rows = torch.tensor([value[0] for value in mapped], dtype=torch.int64)
    secondary_labels[output_rows] = template_labels[template_indices]
    secondary = F.one_hot(secondary_labels, num_classes=4).to(torch.float32)
    nitrogen, calpha, carbon = coords[:, :3].unbind(dim=1)
    ca_to_n = calpha - nitrogen
    c_to_ca = carbon - calpha
    cbeta = -0.57910144 * torch.cross(ca_to_n, c_to_ca, dim=-1) + 0.5689693 * ca_to_n - 0.5441217 * c_to_ca + calpha
    distances = torch.cdist(cbeta, cbeta)
    mapped_adjacent = distances < 6.0
    mapped_adjacent.fill_diagonal_(False)
    adjacent = torch.zeros((length, length), dtype=torch.bool)
    adjacent[output_rows[:, None], output_rows[None, :]] = mapped_adjacent
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


def terminal_cyclization_instance(
    condition: SymbolicBond,
    row_by_reference: dict[object, int],
    output_instances: tuple[str, ...],
) -> str:
    endpoints: list[tuple[int, str]] = []
    for site in (condition.site_a, condition.site_b):
        if type(site) is SymbolicAtomSite:
            reference, atom_name = site.residue, site.atom_id
        elif type(site) is AtomSite:
            # component_index is always an int (§3.3): a non-residue Molecule endpoint is
            # rejected below by row_by_reference, which only holds protein output residues.
            reference, atom_name = (site.instance_id, site.component_index), site.atom_id
        else:
            raise ValueError("RFdiffusion supports only terminal peptide cyclization")
        row = row_by_reference.get(reference)
        if row is None:
            raise ValueError("SymbolicBond endpoints must belong to RFdiffusion output residues")
        endpoints.append((row, atom_name))
    left, right = endpoints
    if output_instances[left[0]] != output_instances[right[0]]:
        raise ValueError("RFdiffusion supports only intrachain terminal peptide cyclization")
    instance_id = output_instances[left[0]]
    rows = [row for row, value in enumerate(output_instances) if value == instance_id]
    if set(endpoints) != {(rows[0], "N"), (rows[-1], "C")}:
        raise ValueError("RFdiffusion cyclic mode requires the terminal N-to-C peptide link")
    return instance_id


def rfdiffusion_secondary_structure(calpha: Tensor) -> Tensor:
    r"""Assign the released helper's P-SEA helix/strand/loop labels."""

    length = calpha.shape[0]
    labels = torch.full((length,), 2, dtype=torch.int64, device=calpha.device)
    if length < 5:
        return labels

    d2 = calpha.new_full((length,), float("nan"))
    d3 = calpha.new_full((length,), float("nan"))
    d4 = calpha.new_full((length,), float("nan"))
    radius_angle = calpha.new_full((length,), float("nan"))
    torsion = calpha.new_full((length,), float("nan"))
    for index in range(1, length - 1):
        d2[index] = torch.linalg.vector_norm(calpha[index + 1] - calpha[index - 1])
        left = calpha[index - 1] - calpha[index]
        right = calpha[index + 1] - calpha[index]
        radius_angle[index] = torch.acos(
            torch.dot(left, right) / (torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right))
        )
    for index in range(1, length - 2):
        d3[index] = torch.linalg.vector_norm(calpha[index + 2] - calpha[index - 1])
        first = calpha[index] - calpha[index - 1]
        second = calpha[index + 1] - calpha[index]
        third = calpha[index + 2] - calpha[index + 1]
        normal_first = torch.cross(first, second, dim=-1)
        normal_second = torch.cross(second, third, dim=-1)
        torsion[index] = torch.atan2(
            torch.dot(torch.cross(normal_first, normal_second, dim=-1), second / torch.linalg.vector_norm(second)),
            torch.dot(normal_first, normal_second),
        )
    for index in range(1, length - 3):
        d4[index] = torch.linalg.vector_norm(calpha[index + 3] - calpha[index - 1])

    degrees = math.pi / 180
    potential_helix = ((d3 >= 4.8) & (d3 <= 5.8) & (d4 >= 5.8) & (d4 <= 7.0)) | (
        (radius_angle >= 77 * degrees)
        & (radius_angle <= 101 * degrees)
        & (torsion >= 30 * degrees)
        & (torsion <= 70 * degrees)
    )
    helix = torch.zeros(length, dtype=torch.bool, device=calpha.device)
    consecutive = 0
    for index in range(length):
        if bool(potential_helix[index]):
            consecutive += 1
        else:
            if consecutive >= 5:
                helix[index - consecutive : index] = True
            consecutive = 0
    for index in torch.where(helix)[0].tolist():
        labels[index] = 0
        if index > 0 and (
            4.8 <= float(d3[index - 1]) <= 5.8 or 77 * degrees <= float(radius_angle[index - 1]) <= 101 * degrees
        ):
            labels[index - 1] = 0
        if index + 1 < length and (
            4.8 <= float(d3[index + 1]) <= 5.8 or 77 * degrees <= float(radius_angle[index + 1]) <= 101 * degrees
        ):
            labels[index + 1] = 0

    potential_strand = ((d2 >= 6.1) & (d2 <= 7.3) & (d3 >= 9.0) & (d3 <= 10.8) & (d4 >= 11.3) & (d4 <= 13.5)) | (
        (radius_angle >= 110 * degrees)
        & (radius_angle <= 138 * degrees)
        & (
            ((torsion >= -180 * degrees) & (torsion <= -125 * degrees))
            | ((torsion >= 145 * degrees) & (torsion <= 180 * degrees))
        )
    )
    strand = torch.zeros(length, dtype=torch.bool, device=calpha.device)
    consecutive = 0
    contacts = 0
    for index in range(length):
        if bool(potential_strand[index]):
            consecutive += 1
            distances = torch.linalg.vector_norm(calpha - calpha[index], dim=-1)
            contacts += int(((distances >= 4.2) & (distances <= 5.2)).sum())
        else:
            if consecutive >= 4 or (consecutive == 3 and contacts >= 5):
                strand[index - consecutive : index] = True
            consecutive = 0
            contacts = 0
    for index in torch.where(strand)[0].tolist():
        labels[index] = 1
        if index > 0 and 9.0 <= float(d3[index - 1]) <= 10.8:
            labels[index - 1] = 1
        if index + 1 < length and 9.0 <= float(d3[index + 1]) <= 10.8:
            labels[index + 1] = 1
    return labels


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

    Returns ``(PlacementSpec, combined_length)``. ``combined_length`` is the family-owned
    ``ContigMap.length`` argument — the combined generated-length bounds ``(Σ min, Σ max)``
    across every generated subcon (excluding ``ExistingSpan`` rows), which the core cannot
    hold and RFdiffusion re-homes onto :class:`RFdiffusionLayout.combined_length`; it is
    ``None`` when the string generates nothing. ``source_instances`` (source chain-letter →
    length) resolves the bare-chain ``A`` form to ``ExistingSpan("A", 0, length)``; explicit
    ``A163-181`` extents are carried literally (the upstream 1-based residue numbering is NOT
    re-based to the core's 0-based index). Lossless into ``(PlacementSpec, combined_length)``
    (the precise ``ref_idx/hal_idx`` permutation, when needed, projects to ordered length-1
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
                    stop += 1  # inclusive upstream bound -> half-open core stop
                if start < 0 or stop < start:
                    raise ValueError(f"malformed existing subcon {sub!r}")
                # NB: RFdiffusion ``contig`` residue numbers are the upstream (often 1-based) PDB
                # numbers and are carried LITERALLY into the ExistingSpan (audit §3:
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
    # Frozen §3.7: reject a duplicate placement_id at construction (globally-unique inner key).
    derive_instance_of(placement)
    combined_length = (generated_min, generated_max) if has_generated else None
    return placement, combined_length


def to_contig_string(
    placement: PlacementSpec,
    combined_length: tuple[int, int] | None = None,
) -> str:
    r"""Re-emit an RFdiffusion ``contigs`` string from a core ``PlacementSpec``.

    Each declaration group → one space-separated block; each placement → one ``/``-subcon
    (``ExistingSpan("A", 163, 182)`` → ``A163-181`` half-open → inclusive;
    ``GeneratedSegment(min, max)`` → ``min-max`` or the bare ``min`` when fixed, so a
    zero-length placement emits ``0``). ``combined_length`` is accepted for signature
    symmetry with :func:`from_contig_string` and is NOT re-serialized into the string: it
    is the family-owned ``ContigMap.length`` argument, carried out-of-band on the Layout,
    not part of the ``contigs`` grammar. The re-emission is lossless for TOPOLOGY (fixed vs
    generated, order, lengths) and lossy for RF-index spacing / receptor-vs-inpaint role —
    the family lowering flags the string never carried, exactly as the audit states.
    """

    blocks: list[str] = []
    for group in placement.values():
        subcons: list[str] = []
        for value in group.values():
            if type(value) is ExistingSpan:
                subcons.append(f"{value.instance_id}{value.start}-{value.stop - 1}")
            else:
                if value.min_length == value.max_length:
                    subcons.append(str(value.min_length))
                else:
                    subcons.append(f"{value.min_length}-{value.max_length}")
        blocks.append("/".join(subcons))
    return " ".join(blocks)


def materialize_placement_map(
    placement: RFdiffusionPlacements,
    output_residues: Sequence[ComponentReference | SymbolicResidueSite],
    residue_map: Mapping[ComponentReference | SymbolicResidueSite, ComponentReference],
) -> tuple[PlacementMap, dict[str, str]]:
    r"""Lift the sampled flat surface into the shared ``placement_map`` (W6 caveat a).

    ``placement`` is the declared :class:`RFdiffusionPlacements` condition; ``output_residues``
    is the post-sample flat identity tuple (each entry a source ``ComponentReference`` for an
    ``ExistingSpan`` row or a :class:`SymbolicResidueSite` for a generated row);
    ``residue_map`` maps each declared output residue to its realized OUTPUT
    ``ComponentReference`` (as built while decoding the Structure). Returns
    ``(placement_map, instance_of)`` where ``instance_of`` is the derived
    ``placement_id -> declaration_group`` index (which also re-validates global
    ``placement_id`` uniqueness).

    Lossless-ness: reconstruction is exact — a generated placement's occurrences are its
    ``SymbolicResidueSite`` rows in ascending offset order, an existing placement's are its
    source components in ``[start, stop)`` order, both projected through ``residue_map``;
    outer/inner iteration is the declared order. The realized id rides per-occurrence in
    ``ComponentReference[0]`` (so a placement MAY span multiple realized instances).
    EXHAUSTIVE coverage is enforced: the union of all placement occurrences is a full
    partition of the realized output surface (no gap, no double-cover, no unaddressed tail).
    """

    spec = placement.placements
    instance_of = derive_instance_of(spec)

    generated: dict[str, list[tuple[int, ComponentReference]]] = {}
    for reference in output_residues:
        if type(reference) is SymbolicResidueSite:
            generated.setdefault(reference.placement_id, []).append((reference.offset, residue_map[reference]))

    placement_map: PlacementMap = {}
    for group_key, placements in spec.items():
        realized: dict[str, tuple[ComponentReference, ...]] = {}
        for placement_id, value in placements.items():
            if type(value) is GeneratedSegment:
                occurrences = tuple(reference for _, reference in sorted(generated.get(placement_id, ())))
            else:
                occurrences = tuple(residue_map[(value.instance_id, index)] for index in range(value.start, value.stop))
            realized[placement_id] = occurrences
        placement_map[group_key] = realized

    _check_exhaustive(placement_map, output_residues, residue_map)
    return placement_map, instance_of


def _check_exhaustive(
    placement_map: PlacementMap,
    output_residues: Sequence[ComponentReference | SymbolicResidueSite],
    residue_map: Mapping[ComponentReference | SymbolicResidueSite, ComponentReference],
) -> None:
    r"""Enforce the adopted §3.7 exhaustion invariant on a materialized ``placement_map``."""

    addressed = [
        reference for group in placement_map.values() for occurrences in group.values() for reference in occurrences
    ]
    if len(set(addressed)) != len(addressed):
        raise ValueError("rfdiffusion placement_map double-covers a realized output residue")
    realized = [residue_map[reference] for reference in output_residues]
    addressed_set = set(addressed)
    realized_set = set(realized)
    if addressed_set != realized_set:
        missing = sorted(realized_set - addressed_set)
        extra = sorted(addressed_set - realized_set)
        raise ValueError(
            "rfdiffusion placement_map is not exhaustive over the realized output surface: "
            f"unaddressed_tail={missing} phantom={extra}"
        )
