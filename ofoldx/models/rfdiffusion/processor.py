r"""
Input processing for RFdiffusion backbone design.
"""

from __future__ import annotations

import operator
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor

from ...data.bonds import SymbolicBond
from ...data.components import get_component
from ...data.conditions import Hotspot
from ...data.evidence import Template
from ...data.molecular import (
    ENTITY_PROTEIN,
    AtomSite,
    ComponentReference,
    MolecularSystem,
    PolymerSequence,
    copy_components,
    form_bond,
)
from ...data.placement import (
    ExistingSpan,
    GeneratedSegment,
    PlacementMap,
    PlacementSpec,
    PlacementValue,
    SymbolicAtomSite,
    SymbolicResidueSite,
)
from ...data.placement import instance_of as derive_instance_of
from ...data.structure import Structure
from ...modules.geometry import rigid_from_three_points
from ...modules.symmetry import point_group_rotations
from ...processor import Processor
from .model import (
    RFDIFFUSION_ATOM14_NAMES,
    RFDIFFUSION_MASK_TOKEN_ID,
    RfDiffusionInput,
    RfDiffusionPotentialInput,
    rfdiffusion_residue_token_id,
    rfdiffusion_secondary_structure,
)


@dataclass(frozen=True)
class RfDiffusionPlacements:
    r"""Declare generated and retained segments for RFdiffusion output chains."""

    placements: PlacementSpec
    receptor_groups: frozenset[str] = frozenset()


@dataclass(frozen=True)
class RfDiffusionCoordination:
    r"""Apply a smooth coordination-number potential between residue C-alpha sets."""

    side_a: tuple[ComponentReference | str | SymbolicResidueSite, ...]
    side_b: tuple[ComponentReference | str | SymbolicResidueSite, ...]
    center_angstrom: float
    scale_angstrom: float
    direction: Literal["maximize", "minimize"]
    weight: float = 1.0


@dataclass(frozen=True)
class RfDiffusionRadiusOfGyration:
    r"""Constrain the compactness of selected RFdiffusion residues."""

    targets: tuple[ComponentReference | str | SymbolicResidueSite, ...]
    upper_angstrom: float = 15.0
    weight: float = 1.0


@dataclass(frozen=True)
class RfDiffusionSubstrateInteraction:
    r"""
    Apply attraction and short-range repulsion to substrate atoms.

    ``nearest_only_repulsion=False`` evaluates repulsion over every
    residue--substrate atom pair, matching the released ``rep_r_min=False``
    equation.
    """

    residues: tuple[ComponentReference | str | SymbolicResidueSite, ...]
    substrate_atoms: tuple[AtomSite | SymbolicAtomSite, ...]
    center_angstrom: float = 2.0
    scale_angstrom: float = 8.0
    attraction_strength: float = 1.0
    repulsion_slope: float = 2.0
    repulsion_cutoff_angstrom: float = 5.0
    nearest_only_repulsion: bool = True
    weight: float = 1.0


@dataclass(frozen=True)
class RfDiffusionPointGroupSymmetry:
    r"""Apply one supported point group to the complete realized output."""

    kind: Literal[
        "cyclic",
        "dihedral",
        "t3",
        "tetrahedral",
        "octahedral",
        "icosahedral",
    ]
    order: int | None = None


@dataclass(frozen=True)
class RfDiffusionSecondaryStructure:
    r"""Assign one secondary-structure label to selected residues."""

    targets: tuple[ComponentReference | str | SymbolicResidueSite, ...]
    kind: Literal["helix", "strand", "loop", "masked"]


class RfDiffusionProcessor(Processor):
    r"""
    Realize RFdiffusion topology and lower its complete numeric Model input.
    """

    def __call__(
        self,
        input: MolecularSystem | Structure | None = None,
        *,
        placement: RfDiffusionPlacements | None = None,
        templates: Sequence[Template] = (),
        secondary_structure: Tensor | None = None,
        block_adjacency: Tensor | None = None,
        secondary_structure_overrides: Sequence[RfDiffusionSecondaryStructure] = (),
        hotspot: Hotspot | None = None,
        inpaint_sequence: Sequence[ComponentReference] = (),
        inpaint_structure: Sequence[ComponentReference] = (),
        provide_sequence: Sequence[SymbolicResidueSite] = (),
        symmetry: RfDiffusionPointGroupSymmetry | None = None,
        potentials: Sequence[
            RfDiffusionCoordination | RfDiffusionRadiusOfGyration | RfDiffusionSubstrateInteraction
        ] = (),
        bonds: Sequence[SymbolicBond] = (),
        length: tuple[int, int] | None = None,
        partial_diffusion: bool = False,
        timestep: int = 50,
        generator: torch.Generator | None = None,
    ) -> RfDiffusionInput:
        processed, _placement_map = self.realize(
            input,
            placement=placement,
            templates=templates,
            secondary_structure=secondary_structure,
            block_adjacency=block_adjacency,
            secondary_structure_overrides=secondary_structure_overrides,
            hotspot=hotspot,
            inpaint_sequence=inpaint_sequence,
            inpaint_structure=inpaint_structure,
            provide_sequence=provide_sequence,
            symmetry=symmetry,
            potentials=potentials,
            bonds=bonds,
            length=length,
            partial_diffusion=partial_diffusion,
            timestep=timestep,
            generator=generator,
        )
        return processed

    def realize(
        self,
        input: MolecularSystem | Structure | None = None,
        *,
        placement: RfDiffusionPlacements | None = None,
        templates: Sequence[Template] = (),
        secondary_structure: Tensor | None = None,
        block_adjacency: Tensor | None = None,
        secondary_structure_overrides: Sequence[RfDiffusionSecondaryStructure] = (),
        hotspot: Hotspot | None = None,
        inpaint_sequence: Sequence[ComponentReference] = (),
        inpaint_structure: Sequence[ComponentReference] = (),
        provide_sequence: Sequence[SymbolicResidueSite] = (),
        symmetry: RfDiffusionPointGroupSymmetry | None = None,
        potentials: Sequence[
            RfDiffusionCoordination | RfDiffusionRadiusOfGyration | RfDiffusionSubstrateInteraction
        ] = (),
        bonds: Sequence[SymbolicBond] = (),
        length: tuple[int, int] | None = None,
        partial_diffusion: bool = False,
        timestep: int = 50,
        generator: torch.Generator | None = None,
    ) -> tuple[RfDiffusionInput, PlacementMap]:
        r"""
        Realize one candidate topology and return its numeric input and placement map.
        """

        structure = input if isinstance(input, Structure) else None
        source = input.system if isinstance(input, Structure) else input
        if placement is None:
            placement = _default_placement(source)
        if placement is None:
            raise ValueError("RFdiffusion requires a placement when no protein input defines one")

        timestep = operator.index(timestep)
        if timestep <= 0:
            raise ValueError("timestep must be positive")
        template_values = tuple(templates)
        if len(template_values) > 1:
            raise ValueError("RFdiffusion accepts at most one structural template")
        if provide_sequence and not partial_diffusion:
            raise ValueError("provide_sequence is available only for partial diffusion")

        rows, group_lengths = _realize_rows(
            placement,
            source,
            length=length,
            generator=generator,
            device=self.device,
        )
        output_length = len(rows)
        if output_length == 0:
            raise ValueError("RFdiffusion requires at least one realized output residue")

        rotations = _symmetry_rotations(symmetry)
        unit_count = len(rotations)
        if unit_count and output_length % unit_count:
            raise ValueError("RFdiffusion symmetry requires equal contiguous units covering the complete output")
        system, output_references, output_instances = _protein_skeleton(
            group_lengths,
            output_length,
            unit_count=unit_count,
            device=self.device,
        )
        placement_map = _placement_map(placement, rows, output_references)
        source_to_rows = _source_rows(rows)
        placement_rows = _placement_rows(rows)

        hidden_sequence = set(inpaint_sequence)
        hidden_structure = set(inpaint_structure)
        _require_selected(hidden_sequence | hidden_structure, source_to_rows)
        provided_rows = {_symbolic_row(site, placement_rows) for site in provide_sequence}

        coordinate_source = [row[2] for row in rows]
        if partial_diffusion:
            if structure is None:
                raise ValueError("partial diffusion requires an input Structure")
            protein_rows = _protein_references(source)
            if len(protein_rows) != output_length:
                raise ValueError("partial diffusion requires positional source/output protein-row correspondence")
            for row, declared_source in enumerate(coordinate_source):
                if declared_source is not None and declared_source != protein_rows[row]:
                    raise ValueError(
                        "partial diffusion ExistingSpan rows must preserve positional source/output correspondence"
                    )
            coordinate_source = list(protein_rows)

        sequence_visible = torch.tensor(
            [row[2] is not None and row[2] not in hidden_sequence for row in rows],
            dtype=torch.bool,
            device=self.device,
        )
        if provided_rows:
            sequence_visible[list(provided_rows)] = True

        identity_source: list[ComponentReference | None] = [
            row[2] if bool(sequence_visible[index]) else None for index, row in enumerate(rows)
        ]
        for row in provided_rows:
            identity_source[row] = coordinate_source[row]

        if any(reference is not None for reference in identity_source):
            if source is None:
                raise ValueError("visible source identities require a molecular input")
            source_on_device = source.to(self.device)  # type: ignore[attr-defined]
            target_rows: list[int] = []
            source_rows: list[int] = []
            for row, reference in enumerate(identity_source):
                if reference is None:
                    continue
                target_rows.append(row)
                source_rows.append(source_on_device.component_index(reference))
            system, _old_to_new, _copied_atom_index = copy_components(
                system,
                torch.tensor(target_rows, dtype=torch.int64, device=self.device),
                source_on_device,
                torch.tensor(source_rows, dtype=torch.int64, device=self.device),
            )

        system, cyclic_rows = _form_bonds(
            system,
            tuple(bonds),
            placement_map,
            source_to_rows,
            output_references,
        )

        component_index = torch.arange(
            output_length,
            dtype=torch.int64,
            device=self.device,
        )
        atom_index, model_atom_names = _model_atom_index(
            system,
            output_references,
            identity_source,
        )
        noisy_coordinates, noisy_coordinate_mask = _source_coordinates(
            structure,
            coordinate_source,
            model_atom_names,
            device=self.device,
            dtype=self.dtype,
        )
        if partial_diffusion and not bool(noisy_coordinate_mask[:, :3].all()):
            raise ValueError("partial diffusion requires observed N/CA/C coordinates for every output residue")

        motif_mask = torch.tensor(
            [structure is not None and row[2] is not None and row[2] not in hidden_structure for row in rows],
            dtype=torch.bool,
            device=self.device,
        )
        if bool(motif_mask.any()) and not bool(noisy_coordinate_mask[motif_mask, :3].all()):
            raise ValueError("RFdiffusion motif rows require observed N/CA/C coordinates")
        tokens = torch.full(
            (output_length,),
            RFDIFFUSION_MASK_TOKEN_ID,
            dtype=torch.int64,
            device=self.device,
        )
        for row, reference in enumerate(identity_source):
            if reference is not None:
                tokens[row] = rfdiffusion_residue_token_id(_component_id(source, reference))

        secondary, adjacency = _fold_features(
            template_values[0] if template_values else None,
            source,
            source_to_rows,
            output_instances,
        )
        if secondary_structure is not None:
            if secondary_structure.shape != (output_length, 4):
                raise ValueError(f"secondary_structure must have shape ({output_length}, 4)")
            secondary = secondary_structure
        if block_adjacency is not None:
            if block_adjacency.shape != (output_length, output_length, 3):
                raise ValueError(f"block_adjacency must have shape ({output_length}, {output_length}, 3)")
            adjacency = block_adjacency
        secondary = secondary.to(device=self.device, dtype=self.dtype)
        adjacency = adjacency.to(device=self.device, dtype=self.dtype)
        labels = secondary.argmax(dim=-1)
        label_by_kind = {"helix": 0, "strand": 1, "loop": 2, "masked": 3}
        for override in secondary_structure_overrides:
            selected = _residue_rows(
                override.targets,
                source_to_rows,
                placement_rows,
            )
            labels[selected] = label_by_kind[override.kind]
        secondary = F.one_hot(labels, num_classes=4).to(self.dtype)

        hotspot_mask = torch.zeros(
            output_length,
            dtype=torch.bool,
            device=self.device,
        )
        if hotspot is not None:
            hotspot_mask[
                _endpoint_rows(
                    hotspot.residues,
                    source_to_rows,
                    placement_rows,
                )
            ] = True

        residue_index = _residue_index(output_instances, device=self.device)
        symmetry_index, symmetry_rotation, symmetry_radial_axis, symmetry_radial_mask = _lower_symmetry(
            rotations, output_length, symmetry, dtype=self.dtype, device=self.device
        )
        potential_input = _lower_potentials(
            tuple(potentials),
            structure,
            source_to_rows,
            placement_rows,
            output_references,
            atom_index,
            model_atom_names,
            noisy_coordinates,
            noisy_coordinate_mask,
            motif_mask,
            dtype=self.dtype,
            device=self.device,
        )
        cyclic_mask = torch.zeros(output_length, dtype=torch.bool, device=self.device)
        if cyclic_rows:
            cyclic_mask[list(cyclic_rows)] = True

        processed = RfDiffusionInput(  # type: ignore[call-arg]
            system=system,
            component_index=component_index[None],
            atom_index=atom_index[None],
            noisy_coordinates=noisy_coordinates[None],
            noisy_coordinate_mask=noisy_coordinate_mask[None],
            residue_type_features=F.one_hot(tokens, num_classes=22).to(self.dtype)[None],
            residue_index=residue_index[None],
            motif_mask=motif_mask[None],
            sequence_visible_mask=sequence_visible[None],
            hotspot_mask=hotspot_mask[None],
            cyclic_mask=cyclic_mask[None],
            secondary_structure=secondary[None],
            block_adjacency=adjacency[None],
            timestep=torch.tensor([timestep], dtype=torch.int64, device=self.device),
            symmetry_index=symmetry_index,
            symmetry_rotation=symmetry_rotation,
            symmetry_radial_axis=symmetry_radial_axis,
            symmetry_radial_mask=symmetry_radial_mask,
            potentials=potential_input,
            batch_size=[],
        )
        return processed, placement_map


def _default_placement(system: MolecularSystem | None) -> RfDiffusionPlacements | None:
    if system is None:
        return None
    placements: PlacementSpec = {}
    for instance in range(system.instance_symbol.numel()):
        entity = int(system.instance_entity[instance])
        if int(system.entity_type[entity]) != ENTITY_PROTEIN:
            continue
        instance_id = system.instance_id(instance)
        start = int(system.instance_component_ptr[instance])
        stop = int(system.instance_component_ptr[instance + 1])
        placements[instance_id] = {instance_id: ExistingSpan(instance_id, 0, stop - start)}
    return RfDiffusionPlacements(placements) if placements else None


def _protein_references(system: MolecularSystem | None) -> tuple[ComponentReference, ...]:
    if system is None:
        return ()
    references: list[ComponentReference] = []
    for instance in range(system.instance_symbol.numel()):
        entity = int(system.instance_entity[instance])
        if int(system.entity_type[entity]) != ENTITY_PROTEIN:
            continue
        instance_id = system.instance_id(instance)
        start = int(system.instance_component_ptr[instance])
        stop = int(system.instance_component_ptr[instance + 1])
        references.extend((instance_id, index) for index in range(stop - start))
    return tuple(references)


def _validate_receptor_groups(placement: RfDiffusionPlacements) -> None:
    unknown = placement.receptor_groups - placement.placements.keys()
    if unknown:
        raise ValueError(f"unknown receptor groups: {sorted(unknown)!r}")
    for group in placement.receptor_groups:
        values = tuple(placement.placements[group].values())
        spans = tuple(value for value in values if isinstance(value, ExistingSpan))
        if not spans or len(spans) != len(values):
            raise ValueError("RFdiffusion receptor groups must contain only existing spans")
        if len({span.instance_id for span in spans}) != 1:
            raise ValueError("one RFdiffusion receptor group must reference one source instance")


def _realize_rows(
    placement: RfDiffusionPlacements,
    source: MolecularSystem | None,
    *,
    length: tuple[int, int] | None,
    generator: torch.Generator | None,
    device: torch.device,
) -> tuple[
    list[tuple[str, str, ComponentReference | None, int]],
    list[tuple[str, int]],
]:
    derive_instance_of(placement.placements)
    _validate_receptor_groups(placement)
    if length is not None:
        lower, upper = (operator.index(value) for value in length)
        if lower < 0 or upper < lower:
            raise ValueError("length bounds must satisfy 0 <= lower <= upper")
        length = lower, upper

    variable: list[tuple[str, GeneratedSegment]] = []
    fixed_nonreceptor = 0
    for group, values in placement.placements.items():
        receptor = group in placement.receptor_groups
        for placement_id, value in values.items():
            if isinstance(value, GeneratedSegment):
                if value.polymer_type != "protein":
                    raise ValueError("RFdiffusion generates protein segments only")
                variable.append((placement_id, value))
            else:
                if source is None:
                    raise ValueError("ExistingSpan requires a molecular input")
                instance = source.instance_index(value.instance_id)
                entity = int(source.instance_entity[instance])
                if int(source.entity_type[entity]) != ENTITY_PROTEIN:
                    raise ValueError("RFdiffusion ExistingSpan must reference a protein instance")
                count = int(source.instance_component_ptr[instance + 1] - source.instance_component_ptr[instance])
                if value.stop > count:
                    raise IndexError(value.stop)
                if not receptor:
                    fixed_nonreceptor += value.stop - value.start

    minimum = fixed_nonreceptor + sum(segment.min_length for _placement_id, segment in variable)
    maximum = fixed_nonreceptor + sum(segment.max_length for _placement_id, segment in variable)
    if length is not None and (length[1] < minimum or length[0] > maximum):
        raise ValueError(f"length {length} cannot be realized by non-receptor output rows in [{minimum}, {maximum}]")

    random_device = device if generator is None else generator.device
    for _attempt in range(100000):
        sampled: dict[str, int] = {}
        for placement_id, segment in variable:
            if segment.min_length == segment.max_length:
                sampled[placement_id] = segment.min_length
            else:
                sampled[placement_id] = int(
                    torch.randint(
                        segment.min_length,
                        segment.max_length + 1,
                        (),
                        generator=generator,
                        device=random_device,
                    )
                )
        realized_nonreceptor = fixed_nonreceptor + sum(sampled[placement_id] for placement_id, _segment in variable)
        if length is None or length[0] <= realized_nonreceptor <= length[1]:
            break
    else:
        raise ValueError("length could not be realized after 100000 upstream-compatible draws")

    ordered_groups = [group for group in placement.placements if group not in placement.receptor_groups]
    ordered_groups.extend(group for group in placement.placements if group in placement.receptor_groups)
    rows: list[tuple[str, str, ComponentReference | None, int]] = []
    group_lengths: list[tuple[str, int]] = []
    for group in ordered_groups:
        start = len(rows)
        for placement_id, value in placement.placements[group].items():
            if isinstance(value, ExistingSpan):
                for offset, component in enumerate(range(value.start, value.stop)):
                    rows.append(
                        (
                            group,
                            placement_id,
                            (value.instance_id, component),
                            offset,
                        )
                    )
            else:
                for offset in range(sampled[placement_id]):
                    rows.append((group, placement_id, None, offset))
        group_lengths.append((group, len(rows) - start))
    return rows, group_lengths


def _symmetry_rotations(
    symmetry: RfDiffusionPointGroupSymmetry | None,
) -> tuple[Tensor, ...]:
    if symmetry is None:
        return ()
    if symmetry.kind in ("cyclic", "dihedral") and symmetry.order is None:
        raise ValueError(f"{symmetry.kind} symmetry requires order")
    return point_group_rotations(symmetry.kind, symmetry.order)


def _protein_skeleton(
    group_lengths: Sequence[tuple[str, int]],
    output_length: int,
    *,
    unit_count: int,
    device: torch.device,
) -> tuple[MolecularSystem, tuple[ComponentReference, ...], tuple[str, ...]]:
    unknown = get_component("UNK")
    if unit_count:
        unit_length = output_length // unit_count
        entity_id = "rfdiffusion_entity_0"
        entities = {
            entity_id: PolymerSequence(
                "protein",
                (unknown,) * unit_length,
            )
        }
        instances = {f"chain_{unit}": entity_id for unit in range(unit_count)}
        references = tuple((f"chain_{row // unit_length}", row % unit_length) for row in range(output_length))
        instance_ids = tuple(reference[0] for reference in references)
    else:
        entities = {}
        instances = {}
        references_list: list[ComponentReference] = []
        instance_ids_list: list[str] = []
        next_instance = 0
        for _group, group_length in group_lengths:
            if group_length == 0:
                continue
            instance_id = f"chain_{next_instance}"
            entity_id = f"rfdiffusion_entity_{next_instance}"
            next_instance += 1
            entities[entity_id] = PolymerSequence(
                "protein",
                (unknown,) * group_length,
            )
            instances[instance_id] = entity_id
            references_list.extend((instance_id, component) for component in range(group_length))
            instance_ids_list.extend((instance_id,) * group_length)
        references = tuple(references_list)
        instance_ids = tuple(instance_ids_list)
    return (
        MolecularSystem.from_entities(entities, instances, device=device),
        references,
        instance_ids,
    )


def _placement_map(
    placement: RfDiffusionPlacements,
    rows: Sequence[tuple[str, str, ComponentReference | None, int]],
    references: Sequence[ComponentReference],
) -> PlacementMap:
    realized: PlacementMap = {
        group: {placement_id: () for placement_id in placements} for group, placements in placement.placements.items()
    }
    collected: dict[tuple[str, str], list[ComponentReference]] = {}
    for row, (group, placement_id, _source, _offset) in enumerate(rows):
        collected.setdefault((group, placement_id), []).append(references[row])
    for (group, placement_id), values in collected.items():
        realized[group][placement_id] = tuple(values)
    return realized


def _source_rows(
    rows: Sequence[tuple[str, str, ComponentReference | None, int]],
) -> dict[ComponentReference, tuple[int, ...]]:
    values: dict[ComponentReference, list[int]] = {}
    for row, (_group, _placement, source, _offset) in enumerate(rows):
        if source is not None:
            values.setdefault(source, []).append(row)
    return {reference: tuple(indices) for reference, indices in values.items()}


def _placement_rows(
    rows: Sequence[tuple[str, str, ComponentReference | None, int]],
) -> dict[str, tuple[int, ...]]:
    values: dict[str, list[int]] = {}
    for row, (_group, placement_id, _source, _offset) in enumerate(rows):
        values.setdefault(placement_id, []).append(row)
    return {placement_id: tuple(indices) for placement_id, indices in values.items()}


def _symbolic_row(
    site: SymbolicResidueSite,
    placement_rows: Mapping[str, tuple[int, ...]],
) -> int:
    return placement_rows[site.placement_id][site.offset]


def _require_selected(
    references: set[ComponentReference],
    source_to_rows: Mapping[ComponentReference, tuple[int, ...]],
) -> None:
    missing = references - source_to_rows.keys()
    if missing:
        raise ValueError(f"source references are absent from the output placement: {sorted(missing)!r}")


def _component_id(
    system: MolecularSystem | None,
    reference: ComponentReference,
) -> str:
    if system is None:
        raise ValueError("a source component requires a molecular system")
    component = system.component_index(reference)
    return system.symbols[int(system.component_symbol[component])]


def _form_bonds(
    system: MolecularSystem,
    bonds: tuple[SymbolicBond, ...],
    placement_map: PlacementMap,
    source_to_rows: Mapping[ComponentReference, tuple[int, ...]],
    output_references: Sequence[ComponentReference],
) -> tuple[MolecularSystem, set[int]]:
    group_by_placement = {
        placement_id: group for group, placements in placement_map.items() for placement_id in placements
    }
    output_by_row = tuple(output_references)
    row_by_output = {reference: row for row, reference in enumerate(output_by_row)}

    def sites(endpoint: AtomSite | SymbolicAtomSite) -> tuple[AtomSite, ...]:
        if isinstance(endpoint, SymbolicAtomSite):
            references = placement_map[group_by_placement[endpoint.residue.placement_id]][endpoint.residue.placement_id]
            reference = references[endpoint.residue.offset]
            return (AtomSite(reference[0], reference[1], endpoint.atom_id),)
        rows = source_to_rows.get((endpoint.instance_id, endpoint.component_index))
        if rows is None:
            return (endpoint,)
        return tuple(
            AtomSite(
                output_by_row[row][0],
                output_by_row[row][1],
                endpoint.atom_id,
            )
            for row in rows
        )

    cyclic_rows: set[int] = set()
    for bond in bonds:
        left = sites(bond.site_a)
        right = sites(bond.site_b)
        if len(left) == 1:
            left = left * len(right)
        if len(right) == 1:
            right = right * len(left)
        if len(left) != len(right):
            raise ValueError("bond endpoints resolve to incompatible occurrence counts")
        for site_a, site_b in zip(left, right, strict=True):
            atom_a = system.atom_index(site_a)
            atom_b = system.atom_index(site_b)
            leaving_a = torch.tensor(
                [
                    system.atom_index(AtomSite(site_a.instance_id, site_a.component_index, atom_id))
                    for atom_id in bond.leaving_a
                ],
                dtype=torch.int64,
                device=system.atom_symbol.device,
            )
            leaving_b = torch.tensor(
                [
                    system.atom_index(AtomSite(site_b.instance_id, site_b.component_index, atom_id))
                    for atom_id in bond.leaving_b
                ],
                dtype=torch.int64,
                device=system.atom_symbol.device,
            )
            system, _old_to_new = form_bond(
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
            )
            if site_a.instance_id == site_b.instance_id and {
                site_a.atom_id,
                site_b.atom_id,
            } == {"N", "C"}:
                instance = system.instance_index(site_a.instance_id)
                start = int(system.instance_component_ptr[instance])
                stop = int(system.instance_component_ptr[instance + 1])
                endpoints = {
                    (row_by_output[(site_a.instance_id, site_a.component_index)], site_a.atom_id),
                    (row_by_output[(site_b.instance_id, site_b.component_index)], site_b.atom_id),
                }
                instance_rows = [
                    row_by_output[system.component_reference(component)] for component in range(start, stop)
                ]
                if endpoints == {
                    (instance_rows[0], "N"),
                    (instance_rows[-1], "C"),
                }:
                    cyclic_rows.update(instance_rows)
    return system, cyclic_rows


def _model_atom_index(
    system: MolecularSystem,
    references: Sequence[ComponentReference],
    identity_source: Sequence[ComponentReference | None],
) -> tuple[Tensor, tuple[tuple[str | None, ...], ...]]:
    indices = torch.full(
        (len(references), 14),
        -1,
        dtype=torch.int64,
        device=system.atom_symbol.device,
    )
    names: list[tuple[str | None, ...]] = []
    for row, reference in enumerate(references):
        component = system.component_index(reference)
        component_id = system.symbols[int(system.component_symbol[component])]
        token = (
            rfdiffusion_residue_token_id(component_id)
            if identity_source[row] is not None
            else RFDIFFUSION_MASK_TOKEN_ID
        )
        atom_names = RFDIFFUSION_ATOM14_NAMES[token]
        names.append(atom_names)
        start = int(system.component_atom_ptr[component])
        stop = int(system.component_atom_ptr[component + 1])
        for slot, atom_name in enumerate(atom_names):
            if atom_name is None:
                continue
            atom = next(
                (index for index in range(start, stop) if system.symbols[int(system.atom_symbol[index])] == atom_name),
                None,
            )
            if atom is not None:
                indices[row, slot] = atom
    return indices, tuple(names)


def _source_coordinates(
    structure: Structure | None,
    sources: Sequence[ComponentReference | None],
    atom_names: Sequence[tuple[str | None, ...]],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    coordinates = torch.zeros(
        (len(sources), 14, 3),
        dtype=dtype,
        device=device,
    )
    mask = torch.zeros(
        (len(sources), 14),
        dtype=torch.bool,
        device=device,
    )
    if structure is None:
        return coordinates, mask
    for row, source in enumerate(sources):
        if source is None:
            continue
        for slot, atom_name in enumerate(atom_names[row]):
            if atom_name is None:
                continue
            try:
                atom = structure.system.atom_index(AtomSite(source[0], source[1], atom_name))
            except KeyError:
                continue
            if not bool(structure.coordinate_mask[atom]):
                continue
            coordinates[row, slot] = structure.coordinates[atom].to(
                device=device,
                dtype=dtype,
            )
            mask[row, slot] = True
    return coordinates, mask


def _residue_rows(
    references: Sequence[ComponentReference | str | SymbolicResidueSite],
    source_to_rows: Mapping[ComponentReference, tuple[int, ...]],
    placement_rows: Mapping[str, tuple[int, ...]],
) -> list[int]:
    rows: list[int] = []
    for reference in references:
        if isinstance(reference, str):
            rows.extend(placement_rows[reference])
        elif isinstance(reference, SymbolicResidueSite):
            rows.append(_symbolic_row(reference, placement_rows))
        else:
            rows.extend(source_to_rows[reference])
    return list(dict.fromkeys(rows))


def _endpoint_rows(
    endpoints: Sequence[ComponentReference | str | SymbolicResidueSite | AtomSite | SymbolicAtomSite],
    source_to_rows: Mapping[ComponentReference, tuple[int, ...]],
    placement_rows: Mapping[str, tuple[int, ...]],
) -> list[int]:
    residues: list[ComponentReference | str | SymbolicResidueSite] = []
    for endpoint in endpoints:
        if isinstance(endpoint, AtomSite):
            residues.append((endpoint.instance_id, endpoint.component_index))
        elif isinstance(endpoint, SymbolicAtomSite):
            residues.append(endpoint.residue)
        else:
            residues.append(endpoint)
    return _residue_rows(residues, source_to_rows, placement_rows)


def _residue_index(
    instance_ids: Sequence[str],
    *,
    device: torch.device,
) -> Tensor:
    values: list[int] = []
    previous = None
    chain = -1
    start = 0
    for row, instance_id in enumerate(instance_ids):
        if instance_id != previous:
            chain += 1
            start = row
            previous = instance_id
        values.append(chain * 200 + row - start)
    return torch.tensor(values, dtype=torch.int64, device=device)


def _lower_symmetry(
    rotations: tuple[Tensor, ...],
    output_length: int,
    symmetry: RfDiffusionPointGroupSymmetry | None,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if not rotations:
        return (
            torch.empty((0, 0), dtype=torch.int64, device=device),
            torch.empty((0, 3, 3), dtype=dtype, device=device),
            torch.empty((0, 3), dtype=dtype, device=device),
            torch.empty((0,), dtype=torch.bool, device=device),
        )
    unit_count = len(rotations)
    rows = torch.arange(output_length, dtype=torch.int64, device=device).reshape(
        unit_count,
        output_length // unit_count,
    )
    matrices = torch.stack(rotations).to(device=device, dtype=dtype)
    radial_axis = torch.einsum(
        "uij,j->ui",
        matrices,
        torch.tensor((1.0, 0.0, 0.0), dtype=dtype, device=device),
    )
    radial_mask = torch.full(
        (unit_count,),
        symmetry is not None and symmetry.kind == "octahedral",
        dtype=torch.bool,
        device=device,
    )
    return rows, matrices, radial_axis, radial_mask


def _fold_features(
    template: Template | None,
    source: MolecularSystem | None,
    source_to_rows: Mapping[ComponentReference, tuple[int, ...]],
    output_instances: Sequence[str],
) -> tuple[Tensor, Tensor]:
    length = len(output_instances)
    secondary = F.one_hot(
        torch.full((length,), 3, dtype=torch.int64),
        num_classes=4,
    ).to(torch.float32)
    adjacency = F.one_hot(
        torch.full((length, length), 2, dtype=torch.int64),
        num_classes=3,
    ).to(torch.float32)
    if template is None:
        return secondary, adjacency
    if source is None:
        raise ValueError("RFdiffusion Template alignments require a molecular input")

    output_rows: list[int] = []
    aligned_coordinates: list[Tensor] = []
    aligned_labels: list[Tensor] = []
    seen: set[int] = set()
    template_cache: dict[str, tuple[Tensor, Tensor, Tensor]] = {}
    for alignment in template.alignments:
        target_entity = source.entity_index(alignment.target_entity_id)
        if int(source.entity_type[target_entity]) != ENTITY_PROTEIN:
            raise ValueError("RFdiffusion Template target must be a protein entity")
        if alignment.target_instance_id is None:
            target_instances = tuple(
                source.instance_id(instance)
                for instance in range(source.instance_symbol.numel())
                if int(source.instance_entity[instance]) == target_entity
            )
        else:
            instance = source.instance_index(alignment.target_instance_id)
            if int(source.instance_entity[instance]) != target_entity:
                raise ValueError("Template target_instance_id does not belong to target_entity_id")
            target_instances = (alignment.target_instance_id,)

        features = template_cache.get(alignment.template_instance_id)
        if features is None:
            template_instance = template.structure.system.instance_index(alignment.template_instance_id)
            start = int(template.structure.system.instance_component_ptr[template_instance])
            stop = int(template.structure.system.instance_component_ptr[template_instance + 1])
            references = tuple((alignment.template_instance_id, index) for index in range(stop - start))
            coordinates, mask = _atom14_coordinates(template.structure, references)
            labels = torch.full((len(references),), 2, dtype=torch.int64)
            observed = mask[:, :3].all(dim=-1)
            segment_start = 0
            while segment_start < len(references):
                if not bool(observed[segment_start]):
                    segment_start += 1
                    continue
                segment_stop = segment_start + 1
                while segment_stop < len(references) and bool(observed[segment_stop]):
                    segment_stop += 1
                labels[segment_start:segment_stop] = rfdiffusion_secondary_structure(
                    coordinates[segment_start:segment_stop, 1]
                )
                segment_start = segment_stop
            features = coordinates, mask, labels
            template_cache[alignment.template_instance_id] = features
        coordinates, mask, labels = features

        for target_index, template_index in alignment.residue_pairs:
            for instance_id in target_instances:
                for output_row in source_to_rows.get((instance_id, target_index), ()):
                    if output_row in seen:
                        raise ValueError("RFdiffusion Template alignments overlap on one output row")
                    if not bool(mask[template_index, :3].all()):
                        raise ValueError("RFdiffusion Template requires aligned N/CA/C coordinates")
                    seen.add(output_row)
                    output_rows.append(output_row)
                    aligned_coordinates.append(coordinates[template_index])
                    aligned_labels.append(labels[template_index])
    if not output_rows:
        raise ValueError("RFdiffusion Template alignments do not cover any output row")

    output = torch.tensor(output_rows, dtype=torch.int64)
    secondary_labels = secondary.argmax(dim=-1)
    secondary_labels[output] = torch.stack(aligned_labels)
    secondary = F.one_hot(secondary_labels, num_classes=4).to(torch.float32)
    coordinates = torch.stack(aligned_coordinates)
    nitrogen, calpha, carbon = coordinates[:, :3].unbind(dim=1)
    cbeta = (
        -0.57910144 * torch.cross(calpha - nitrogen, carbon - calpha, dim=-1)
        + 0.5689693 * (calpha - nitrogen)
        - 0.5441217 * (carbon - calpha)
        + calpha
    )
    contacts = torch.cdist(cbeta, cbeta) < 6.0
    contacts.fill_diagonal_(False)
    mapped = torch.zeros((length, length), dtype=torch.bool)
    mapped[output[:, None], output[None, :]] = contacts

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
            if bool(mapped[left_start:left_stop, right_start:right_stop].any()):
                block[left_start:left_stop, right_start:right_stop] = 1
                block[right_start:right_stop, left_start:left_stop] = 1
    known = secondary_labels != 3
    block = block.masked_fill(~(known[:, None] & known[None, :]), 2)
    return secondary, F.one_hot(block, num_classes=3).to(torch.float32)


def _atom14_coordinates(
    structure: Structure,
    references: Sequence[ComponentReference],
) -> tuple[Tensor, Tensor]:
    coordinates = torch.zeros(
        (len(references), 14, 3),
        dtype=structure.coordinates.dtype,
        device=structure.coordinates.device,
    )
    mask = torch.zeros(
        (len(references), 14),
        dtype=torch.bool,
        device=structure.coordinates.device,
    )
    for row, reference in enumerate(references):
        component_id = _component_id(structure.system, reference)
        atom_names = RFDIFFUSION_ATOM14_NAMES[rfdiffusion_residue_token_id(component_id)]
        for slot, atom_name in enumerate(atom_names):
            if atom_name is None:
                continue
            try:
                atom = structure.system.atom_index(AtomSite(reference[0], reference[1], atom_name))
            except KeyError:
                continue
            if bool(structure.coordinate_mask[atom]):
                coordinates[row, slot] = structure.coordinates[atom]
                mask[row, slot] = True
    return coordinates, mask


def _lower_potentials(
    potentials: tuple[
        RfDiffusionCoordination | RfDiffusionRadiusOfGyration | RfDiffusionSubstrateInteraction,
        ...,
    ],
    structure: Structure | None,
    source_to_rows: Mapping[ComponentReference, tuple[int, ...]],
    placement_rows: Mapping[str, tuple[int, ...]],
    output_references: Sequence[ComponentReference],
    atom_index: Tensor,
    atom_names: Sequence[tuple[str | None, ...]],
    noisy_coordinates: Tensor,
    noisy_coordinate_mask: Tensor,
    motif_mask: Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> RfDiffusionPotentialInput:
    point_atom_index: list[int] = []
    point_local_coordinates: list[Tensor] = []
    frame_atom_index = torch.full((3,), -1, dtype=torch.int64, device=device)
    frame_row = next(
        (
            row
            for row in range(len(output_references))
            if bool(motif_mask[row]) and bool(noisy_coordinate_mask[row, :3].all())
        ),
        None,
    )
    if frame_row is not None:
        frame_atom_index = torch.tensor(
            [frame_row * 14, frame_row * 14 + 1, frame_row * 14 + 2],
            dtype=torch.int64,
            device=device,
        )
        frame_rotation, frame_center = rigid_from_three_points(
            noisy_coordinates[frame_row, 0],
            noisy_coordinates[frame_row, 1],
            noisy_coordinates[frame_row, 2],
        )

    def internal(row: int, atom_name: str) -> int:
        try:
            slot = atom_names[row].index(atom_name)
        except ValueError:
            raise ValueError(f"RFdiffusion model axis has no atom {atom_name!r} at output row {row}") from None
        if int(atom_index[row, slot]) < 0:
            raise ValueError(f"RFdiffusion output graph has no atom {atom_name!r} at output row {row}")
        point_atom_index.append(row * 14 + slot)
        point_local_coordinates.append(torch.zeros(3, dtype=dtype, device=device))
        return len(point_atom_index) - 1

    def external(site: AtomSite) -> int:
        if structure is None:
            raise ValueError("external RFdiffusion potential atoms require an input Structure")
        if frame_row is None:
            raise ValueError("external RFdiffusion potential atoms require an observed motif N/CA/C frame")
        atom = structure.system.atom_index(site)
        if not bool(structure.coordinate_mask[atom]):
            raise ValueError(f"external RFdiffusion potential atom lacks coordinates: {site!r}")
        value = structure.coordinates[atom].to(device=device, dtype=dtype)
        point_atom_index.append(-1)
        point_local_coordinates.append((value - frame_center) @ frame_rotation)
        return len(point_atom_index) - 1

    def atom_points(site: AtomSite | SymbolicAtomSite) -> list[int]:
        if isinstance(site, SymbolicAtomSite):
            row = _symbolic_row(site.residue, placement_rows)
            return [internal(row, site.atom_id)]
        rows = source_to_rows.get((site.instance_id, site.component_index))
        if rows:
            return [internal(row, site.atom_id) for row in rows]
        return [external(site)]

    def residue_points(
        references: Sequence[ComponentReference | str | SymbolicResidueSite],
    ) -> list[int]:
        return [
            internal(row, "CA")
            for row in _residue_rows(
                references,
                source_to_rows,
                placement_rows,
            )
        ]

    coordination_point_index: list[int] = []
    coordination_ptr = [0]
    coordination_center: list[float] = []
    coordination_scale: list[float] = []
    coordination_weight: list[float] = []
    radius_point_index: list[int] = []
    radius_ptr = [0]
    radius_upper: list[float] = []
    radius_weight: list[float] = []
    substrate_point_index: list[int] = []
    substrate_ptr = [0]
    substrate_parameters: list[tuple[float, float, float, float, float, float]] = []
    substrate_nearest_only: list[bool] = []

    for potential in potentials:
        weight = potential.weight
        if isinstance(potential, RfDiffusionCoordination):
            left = residue_points(potential.side_a)
            right = residue_points(potential.side_b)
            coordination_point_index.extend(left)
            coordination_ptr.append(len(coordination_point_index))
            coordination_point_index.extend(right)
            coordination_ptr.append(len(coordination_point_index))
            coordination_center.append(potential.center_angstrom)
            coordination_scale.append(potential.scale_angstrom)
            coordination_weight.append(weight if potential.direction == "maximize" else -weight)
        elif isinstance(potential, RfDiffusionRadiusOfGyration):
            selected = residue_points(potential.targets)
            radius_point_index.extend(selected)
            radius_ptr.append(len(radius_point_index))
            radius_upper.append(potential.upper_angstrom)
            radius_weight.append(weight)
        else:
            selected = residue_points(potential.residues)
            substrate = [point for atom in potential.substrate_atoms for point in atom_points(atom)]
            substrate_point_index.extend(selected)
            substrate_ptr.append(len(substrate_point_index))
            substrate_point_index.extend(substrate)
            substrate_ptr.append(len(substrate_point_index))
            substrate_parameters.append(
                (
                    potential.center_angstrom,
                    potential.scale_angstrom,
                    potential.attraction_strength,
                    potential.repulsion_slope,
                    potential.repulsion_cutoff_angstrom,
                    weight,
                )
            )
            substrate_nearest_only.append(potential.nearest_only_repulsion)

    def integer_tensor(values: object, shape: tuple[int, ...] | None = None) -> Tensor:
        tensor = torch.tensor(values, dtype=torch.int64, device=device)
        return tensor if shape is None or tensor.numel() else tensor.reshape(shape)

    def float_tensor(values: object, shape: tuple[int, ...] | None = None) -> Tensor:
        tensor = torch.tensor(values, dtype=dtype, device=device)
        return tensor if shape is None or tensor.numel() else tensor.reshape(shape)

    return RfDiffusionPotentialInput(  # type: ignore[call-arg]
        point_atom_index=integer_tensor(point_atom_index),
        point_local_coordinates=(
            torch.stack(point_local_coordinates)
            if point_local_coordinates
            else torch.empty((0, 3), dtype=dtype, device=device)
        ),
        frame_atom_index=frame_atom_index,
        coordination_point_index=integer_tensor(coordination_point_index),
        coordination_ptr=integer_tensor(coordination_ptr),
        coordination_center=float_tensor(coordination_center),
        coordination_scale=float_tensor(coordination_scale),
        coordination_weight=float_tensor(coordination_weight),
        radius_point_index=integer_tensor(radius_point_index),
        radius_ptr=integer_tensor(radius_ptr),
        radius_upper=float_tensor(radius_upper),
        radius_weight=float_tensor(radius_weight),
        substrate_point_index=integer_tensor(substrate_point_index),
        substrate_ptr=integer_tensor(substrate_ptr),
        substrate_parameters=float_tensor(substrate_parameters, (0, 6)),
        substrate_nearest_only=torch.tensor(
            substrate_nearest_only,
            dtype=torch.bool,
            device=device,
        ),
        batch_size=[],
    )


def _author_dense_index(source: Structure) -> dict[str, dict[tuple[int, str], int]]:
    author_dense: dict[str, dict[tuple[int, str], int]] = {}
    for component in range(source.system.component_symbol.numel()):
        residue_symbol = int(source.residue_id[component])
        if residue_symbol < 0:
            continue
        insertion_symbol = int(source.insertion_code[component])
        insertion = "" if insertion_symbol < 0 else source.source_symbols[insertion_symbol]
        number = int(source.source_symbols[residue_symbol])
        instance_id, component_index = source.system.component_reference(component)
        author_dense.setdefault(instance_id, {})[(number, insertion)] = component_index
    return author_dense


def _resolve_author_numbering(
    source: Structure | None,
    author_numbering: Mapping[str, Mapping[tuple[int, str], int]] | None,
) -> Mapping[str, Mapping[tuple[int, str], int]]:
    if source is not None:
        return _author_dense_index(source)
    return {} if author_numbering is None else author_numbering


def _existing_bounds(sub: str, extent: str) -> tuple[int, int]:
    bounds = extent.split("-")
    if len(bounds) != 2 or not all(bound.isdigit() for bound in bounds):
        raise ValueError(f"malformed existing subcon {sub!r}")
    low, high = int(bounds[0]), int(bounds[1])
    if high < low:
        raise ValueError(f"malformed existing subcon {sub!r}")
    return low, high


def _rebase_existing_span(
    instance_id: str,
    author_start: int,
    author_end: int,
    author_dense: Mapping[str, Mapping[tuple[int, str], int]],
    sub: str,
) -> tuple[int, int]:
    chain = author_dense.get(instance_id)
    if chain is None:
        raise ValueError(f"existing subcon {sub!r} references source chain {instance_id!r} with no author numbering")
    insertion_numbers = {number for number, insertion in chain if insertion}
    selected: list[int] = []
    for number in range(author_start, author_end + 1):
        if number in insertion_numbers:
            raise ValueError(f"existing subcon {sub!r} crosses insertion-coded author residue {number}")
        try:
            selected.append(chain[(number, "")])
        except KeyError:
            raise ValueError(
                f"existing subcon {sub!r} names author residue {number} absent from source chain {instance_id!r}"
            ) from None
    if selected != list(range(selected[0], selected[0] + len(selected))):
        raise ValueError(
            f"existing subcon {sub!r} does not map to a contiguous residue run of source chain {instance_id!r}"
        )
    return selected[0], selected[-1] + 1


def from_contig_string(
    contigs: str,
    *,
    source: Structure | None = None,
    author_numbering: Mapping[str, Mapping[tuple[int, str], int]] | None = None,
) -> RfDiffusionPlacements:
    r"""
    Parse the released RFdiffusion contig grammar into placements and receptor groups.
    """

    placement: PlacementSpec = {}
    receptor_groups: set[str] = set()
    author_dense = _resolve_author_numbering(source, author_numbering)
    blocks = contigs.split()
    for block_index, block in enumerate(blocks):
        group_key = f"chain_{block_index + 1}"
        raw_parts = block.split("/")
        if any(not value for value in raw_parts):
            raise ValueError(f"malformed contig block {block!r}")
        trailing_zero = raw_parts[-1] == "0"
        if trailing_zero:
            raw_parts.pop()
        group: dict[str, PlacementValue] = {}
        existing_fragments: list[tuple[str, int]] = []
        for sub_index, sub in enumerate(raw_parts):
            placement_id = f"{group_key}:{sub_index}"
            if sub[0].isalpha():
                instance_id = sub[0]
                author_start, author_end = _existing_bounds(sub, sub[1:])
                if not author_dense:
                    raise ValueError(
                        f"existing subcon {sub!r} is author-numbered and needs a source structure "
                        "or author_numbering mapping"
                    )
                start, stop = _rebase_existing_span(
                    instance_id,
                    author_start,
                    author_end,
                    author_dense,
                    sub,
                )
                group[placement_id] = ExistingSpan(instance_id, start, stop)
                existing_fragments.append((instance_id, author_start))
            else:
                bounds = sub.split("-")
                if not all(bound.isdigit() for bound in bounds):
                    raise ValueError(f"malformed generated subcon {sub!r}")
                if len(bounds) == 1:
                    low = high = int(bounds[0])
                elif len(bounds) == 2:
                    low, high = int(bounds[0]), int(bounds[1])
                else:
                    raise ValueError(f"malformed generated subcon {sub!r}")
                if high < low:
                    raise ValueError(f"malformed generated subcon {sub!r}")
                group[placement_id] = GeneratedSegment("protein", low, high)
        placement[group_key] = group
        all_existing = bool(group) and len(existing_fragments) == len(group)
        receptor = all_existing and (trailing_zero or block_index == len(blocks) - 1)
        if receptor:
            receptor_chains = {chain for chain, _start in existing_fragments}
            if len(receptor_chains) != 1:
                raise ValueError("one RFdiffusion receptor block must reference one source chain")
            receptor_starts = [start for _chain, start in existing_fragments]
            if any(
                left >= right
                for left, right in zip(
                    receptor_starts,
                    receptor_starts[1:],
                )
            ):
                raise ValueError("RFdiffusion receptor fragments must be ordered by author residue number")
            receptor_groups.add(group_key)
    return RfDiffusionPlacements(
        placement,
        receptor_groups=frozenset(receptor_groups),
    )


def to_contig_string(
    placement: RfDiffusionPlacements,
    *,
    source: Structure | None = None,
    author_numbering: Mapping[str, Mapping[tuple[int, str], int]] | None = None,
) -> str:
    r"""
    Re-emit RFdiffusion placements with canonical chain breaks.
    """

    author_dense = _resolve_author_numbering(source, author_numbering)
    _validate_receptor_groups(placement)
    blocks: list[str] = []
    placement_groups = tuple(placement.placements.items())
    for group_index, (group_key, group) in enumerate(placement_groups):
        subcons: list[str] = []
        receptor = group_key in placement.receptor_groups
        previous_receptor_start: int | None = None
        for value in group.values():
            if isinstance(value, ExistingSpan):
                if len(value.instance_id) != 1 or not value.instance_id.isalpha():
                    raise ValueError(f"RFdiffusion contig syntax cannot represent source chain {value.instance_id!r}")
                chain = author_dense.get(value.instance_id)
                if chain is None:
                    raise ValueError(
                        f"existing span over chain {value.instance_id!r} needs a source structure "
                        "or author_numbering mapping"
                    )
                positions: list[tuple[int, str]] = []
                for dense in range(value.start, value.stop):
                    matches = [position for position, row in chain.items() if row == dense]
                    if len(matches) != 1:
                        raise ValueError(f"existing span {value!r} has no unique author numbering")
                    positions.append(matches[0])
                if not positions:
                    raise ValueError("RFdiffusion contig syntax cannot represent an empty existing span")
                if any(insertion for _number, insertion in positions):
                    raise ValueError(
                        f"RFdiffusion contig syntax cannot represent insertion-coded residues in {value!r}"
                    )
                numbers = [number for number, _insertion in positions]
                if numbers[0] < 0 or numbers != list(range(numbers[0], numbers[0] + len(numbers))):
                    raise ValueError(
                        f"RFdiffusion contig syntax requires consecutive non-negative author numbering in {value!r}"
                    )
                start_number = numbers[0]
                end_number = numbers[-1]
                if receptor:
                    if previous_receptor_start is not None and start_number <= previous_receptor_start:
                        raise ValueError("RFdiffusion receptor fragments must be ordered by author residue number")
                    previous_receptor_start = start_number
                subcons.append(f"{value.instance_id}{start_number}-{end_number}")
            elif value.min_length == value.max_length:
                subcons.append(str(value.min_length))
            else:
                subcons.append(f"{value.min_length}-{value.max_length}")
        all_existing = bool(group) and all(isinstance(value, ExistingSpan) for value in group.values())
        boundary = group_index < len(placement_groups) - 1
        if receptor or (boundary and not all_existing):
            subcons.append("0")
        blocks.append("/".join(subcons))
    contigs = " ".join(blocks)
    try:
        restored = from_contig_string(contigs, author_numbering=author_dense)
    except (IndexError, KeyError, ValueError) as error:
        raise ValueError("placement cannot be represented by the released RFdiffusion contig syntax") from error
    expected = tuple(
        (tuple(group.values()), group_key in placement.receptor_groups)
        for group_key, group in placement.placements.items()
    )
    observed = tuple(
        (tuple(group.values()), group_key in restored.receptor_groups)
        for group_key, group in restored.placements.items()
    )
    if observed != expected:
        raise ValueError("placement cannot round-trip exactly through the released RFdiffusion contig syntax")
    return contigs
