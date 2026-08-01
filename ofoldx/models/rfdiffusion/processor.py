# OFoldX
# Copyright (C) 2026-Present  Valhalla Technology

# This file is part of OFoldX.

# OFoldX is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# OFoldX is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# For additional terms and clarifications, please refer to our License FAQ at:
# <https://ofoldx.valtech.ai/about/license-faq/>.

r"""
RFdiffusion input preparation, written against the shared preparation contract.

One candidate is one realised output topology: a protein chain per declaration group whose rows are either generated
(the unknown amino acid ``UNK``) or copies of source components, built once by ``chemistry.construction.build_system``
and placed in the source frame -- a copied row carries the source observations, a generated one has no coordinates.
The model's rows are those components' residue tokens, its slots their atom14 names and its tokens the canonical
classes (a hidden identity reads ``<mask>``). Generation regenerates the generated and diffused rows only: the
fixed motif keeps the coordinates it was copied with.
"""

from __future__ import annotations

import operator
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from torch import Tensor

from ...chemistry.construction import (
    INHERITED_COLUMNS,
    InheritedEdges,
    PolymerSequence,
    Recipe,
    RecipeBond,
    build_system,
)
from ...chemistry.definitions import ComponentLibrary, configured_library
from ...data.molecular import AtomSite, BondKind, BondOrigin, ComponentRef, MolecularSystem, MonomerType, Transforms
from ...data.structure import CoordinateEvidence, Structure, replace_coordinates
from ...data.tokenization import CLASS_MONOMER, TOKEN_INDEX, TOKENS, token_classes
from ...errors import RequestRef, StructureRef, fail
from ...modules.geometry import rotation_from_two_vectors
from ...modules.symmetry import point_group_rotations
from ...processor import Processor
from ...provenance import Records, known
from ...templates import (
    ResolvedTemplateAlignment,
    Template,
    resolve_template,
    template_alignment_frames,
    template_atoms,
)
from ..conditions import (
    Endpoint,
    ExistingSpan,
    GeneratedSegment,
    Hotspot,
    PlacementMap,
    PlacementSpec,
    PlacementValue,
    ResidueEndpoint,
    SymbolicAtomSite,
    SymbolicBond,
    SymbolicResidueSite,
    placement_instance_of,
    resolve_placement,
)
from ..design import PositionSelection, resolve_positions
from ..layout import LayoutTable, model_tokens, slot_map, token_layout
from ..preparation import (
    GatingPolicy,
    PreparationContext,
    PreparationRecord,
    PreparedInput,
    RegenerationScope,
    SlotMap,
    component_records,
    gate,
)
from ..processing import project_model_coordinates, project_model_features
from .model import (
    RFDIFFUSION_ATOM14_NAMES,
    RFDIFFUSION_RESIDUE_TYPES,
    RFDIFFUSION_UNKNOWN_TOKEN_ID,
    RfDiffusionInput,
    RfDiffusionPotentialInput,
    rfdiffusion_secondary_structure,
)

_PROTEIN = MonomerType.PROTEIN
#: The component every generated row is built from; its canonical class is protein ``X``.
RFDIFFUSION_GENERATED_COMPONENT = "UNK"
#: The atom14 table of every standard residue id and of the unknown residue.
RFDIFFUSION_ATOM14_TABLE: Mapping[str, tuple[str | None, ...]] = {
    name: RFDIFFUSION_ATOM14_NAMES[index]
    for index, name in enumerate(RFDIFFUSION_RESIDUE_TYPES[: RFDIFFUSION_UNKNOWN_TOKEN_ID + 1])
}
_UNKNOWN_ATOM14 = RFDIFFUSION_ATOM14_NAMES[RFDIFFUSION_UNKNOWN_TOKEN_ID]
#: One row is one protein residue token addressed by its atom14 names (upstream reads an unlisted id as unknown).
RFDIFFUSION_LAYOUT = LayoutTable(
    family="rfdiffusion",
    monomer_types={_PROTEIN: 0},
    residue_slots=RFDIFFUSION_ATOM14_TABLE,
    monomer_slots={_PROTEIN: _UNKNOWN_ATOM14},
    hydrogens="exclude",
    per_atom_components="none",
    exclude_designated_leaving=False,
    centre_atom={_PROTEIN: "CA"},
    frame_atoms={_PROTEIN: ("N", "CA", "C")},
    representative={},
    representative_default={},
)
#: RFdiffusion reads backbone coordinates and residue identities; no charge, bond or hydrogen count is its input.
RFDIFFUSION_GATING = GatingPolicy(
    refuse_undetermined_in_slots=False,
    refuse_unknown_kind_in_slots=False,
    refuse_incomplete_topology_in_slots=False,
    refuse_valence_excess=False,
)
_ATOM14 = len(_UNKNOWN_ATOM14)
_MASK_CLASS = TOKEN_INDEX["<mask>"]
_LENGTH_DRAWS = 100000


@dataclass(frozen=True)
class RfDiffusionPlacements:
    r"""Declare generated and copied segments for RFdiffusion output chains."""

    placements: PlacementSpec
    receptor_groups: frozenset[str] = frozenset()


@dataclass(frozen=True)
class RfDiffusionCoordination:
    r"""Apply a smooth coordination-number potential between residue C-alpha sets."""

    side_a: tuple[ResidueEndpoint, ...]
    side_b: tuple[ResidueEndpoint, ...]
    center_angstrom: float
    scale_angstrom: float
    direction: Literal["maximize", "minimize"]
    weight: float = 1.0


@dataclass(frozen=True)
class RfDiffusionFlatBottomRadialPotential:
    r"""Apply a flat-bottom radial compactness potential to selected RFdiffusion residues.

    Each distance from the selected CA centroid is bounded below by ``radius_threshold_angstrom`` before the
    root-mean-square reduction. Guidance subtracts this value times ``weight``."""

    targets: tuple[ResidueEndpoint, ...]
    radius_threshold_angstrom: float = 15.0
    weight: float = 1.0


@dataclass(frozen=True)
class RfDiffusionSubstrateInteraction:
    r"""
    Apply attraction and short-range repulsion to substrate atoms.

    ``nearest_only_repulsion=False`` evaluates repulsion over every residue--substrate atom pair, matching the
    released ``rep_r_min=False`` equation.
    """

    residues: tuple[ResidueEndpoint, ...]
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

    kind: Literal["cyclic", "dihedral", "t3", "tetrahedral", "octahedral", "icosahedral"]
    order: int | None = None


@dataclass(frozen=True)
class RfDiffusionSecondaryStructure:
    r"""Assign one secondary-structure label to selected residues."""

    targets: tuple[ResidueEndpoint, ...]
    kind: Literal["helix", "strand", "loop", "masked"]


#: One guidance potential of the family.
Potential = RfDiffusionCoordination | RfDiffusionFlatBottomRadialPotential | RfDiffusionSubstrateInteraction
_LABELS = {"helix": 0, "strand": 1, "loop": 2, "masked": 3}


@dataclass(frozen=True)
class _Row:
    r"""One output residue: its declaration group and placement, the source component it copies (``-1`` generated)
    and its offset inside its placement."""

    group: str
    placement: str
    source: int
    offset: int


@dataclass(frozen=True)
class _Realization:
    r"""The realised output: its rows, the target they were built into with the construction's records, and the maps
    the lowering reads them by."""

    rows: tuple[_Row, ...]
    target: Structure
    records: Records
    references: tuple[ComponentRef, ...]
    instance_index: tuple[int, ...]
    placement_map: PlacementMap
    source_rows: Mapping[ComponentRef, tuple[int, ...]]
    placement_rows: Mapping[str, tuple[int, ...]]
    component_ids: tuple[str, ...]


class RfDiffusionProcessor(Processor):
    r"""Realise one RFdiffusion output topology and lower its complete numeric Model input."""

    def prepare_input(
        self,
        structure: Structure | None = None,
        *,
        placement: RfDiffusionPlacements | None = None,
        templates: Sequence[Template] = (),
        secondary_structure: Tensor | None = None,
        block_adjacency: Tensor | None = None,
        secondary_structure_overrides: Sequence[RfDiffusionSecondaryStructure] = (),
        hotspot: Hotspot | None = None,
        inpaint_sequence: PositionSelection | None = None,
        inpaint_structure: PositionSelection | None = None,
        provide_sequence: Sequence[SymbolicResidueSite] = (),
        symmetry: RfDiffusionPointGroupSymmetry | None = None,
        potentials: Sequence[Potential] = (),
        bonds: Sequence[SymbolicBond] = (),
        length: tuple[int, int] | None = None,
        partial_diffusion: bool = False,
        timestep: int = 50,
        library: ComponentLibrary | None = None,
        generator: np.random.Generator | None = None,
    ) -> PreparedInput[RfDiffusionInput]:
        r"""One realised candidate (``realize`` without its placement map)."""
        return self.realize(
            structure,
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
            library=library,
            generator=generator,
        )[0]

    def realize(
        self,
        structure: Structure | None = None,
        *,
        placement: RfDiffusionPlacements | None = None,
        templates: Sequence[Template] = (),
        secondary_structure: Tensor | None = None,
        block_adjacency: Tensor | None = None,
        secondary_structure_overrides: Sequence[RfDiffusionSecondaryStructure] = (),
        hotspot: Hotspot | None = None,
        inpaint_sequence: PositionSelection | None = None,
        inpaint_structure: PositionSelection | None = None,
        provide_sequence: Sequence[SymbolicResidueSite] = (),
        symmetry: RfDiffusionPointGroupSymmetry | None = None,
        potentials: Sequence[Potential] = (),
        bonds: Sequence[SymbolicBond] = (),
        length: tuple[int, int] | None = None,
        partial_diffusion: bool = False,
        timestep: int = 50,
        library: ComponentLibrary | None = None,
        generator: np.random.Generator | None = None,
    ) -> tuple[PreparedInput[RfDiffusionInput], PlacementMap]:
        r"""
        Realise one candidate topology and return its prepared input with the realised components of every placement.

        Args:
            structure: The source every existing span, template alignment and substrate atom is read from.
            placement: The declaration groups of the output (default: one existing span per source protein instance).
            templates: At most one structural template, aligned to the source entities.
            secondary_structure: Optional ``(rows, 4)`` secondary-structure features replacing the template's.
            block_adjacency: Optional ``(rows, rows, 3)`` block-adjacency features replacing the template's.
            secondary_structure_overrides: Per-residue secondary-structure labels applied after the features.
            hotspot: Optional residues labelled as interaction hotspots.
            inpaint_sequence: Source positions whose identity is hidden from the model.
            inpaint_structure: Source positions whose coordinates are diffused instead of fixed.
            provide_sequence: Rows whose identity is provided under partial diffusion.
            symmetry: Optional point group over equal contiguous units of the output.
            potentials: Guidance potentials over the realised rows and source substrate atoms.
            bonds: Covalent bonds the output declares between its rows.
            length: Optional inclusive bounds on the number of non-receptor rows.
            partial_diffusion: Whether every row is diffused from its source coordinates.
            timestep: The diffusion timestep the features are lowered at.
            library: The definitions the output is built from (default the configured artifact).
            generator: The numpy generator variable segment lengths are drawn with.

        Returns:
            The prepared candidate and its placement map.
        """

        if placement is None:
            placement = _default_placement(structure)
        if placement is None:
            fail("value_invalid", RequestRef("placement", "None"), reason="no protein input defines one")
        timestep = operator.index(timestep)
        if timestep <= 0:
            fail("value_invalid", RequestRef("timestep", str(timestep)))
        template_values = tuple(templates)
        if len(template_values) > 1:
            fail("value_invalid", RequestRef("templates", str(len(template_values))), expected=1)
        if provide_sequence and not partial_diffusion:
            fail("value_invalid", RequestRef("provide_sequence", "without partial_diffusion"))

        rotations = _symmetry_rotations(symmetry)
        realization = _realize(
            structure, placement, rotations, bonds=tuple(bonds), length=length, generator=generator, library=library
        )
        rows, target = realization.rows, realization.target
        system = target.system
        if partial_diffusion:
            _require_positional_source(structure, rows)

        layout = token_layout(system, RFDIFFUSION_LAYOUT)
        slots = slot_map(system, layout, RFDIFFUSION_LAYOUT)
        accepted = gate(target, slots, RFDIFFUSION_GATING)
        tokens, losses = model_tokens(system, layout)
        copied = np.array([row.source >= 0 for row in rows])
        hidden_sequence = _selected_rows(structure, inpaint_sequence, "inpaint_sequence", realization.source_rows)
        hidden_structure = _selected_rows(structure, inpaint_structure, "inpaint_structure", realization.source_rows)
        visible = copied & ~_mask_of(hidden_sequence, len(rows))
        visible[[_symbolic_row(site, realization.placement_rows) for site in provide_sequence]] = True
        tokens = np.where(visible, tokens, _MASK_CLASS)
        coordinates, coordinate_mask = _row_coordinates(target, slots)
        fixed = copied & ~_mask_of(hidden_structure, len(rows))
        motif = np.zeros(len(rows), dtype=bool) if partial_diffusion else fixed
        _require_backbone(np.flatnonzero(copied if partial_diffusion else motif), coordinate_mask)
        scope = RegenerationScope.components(system, np.flatnonzero(~motif))

        secondary, adjacency = _fold_features(template_values[0] if template_values else None, structure, realization)
        if secondary_structure is not None:
            if secondary_structure.shape != (len(rows), 4):
                fail("value_invalid", RequestRef("secondary_structure", str(tuple(secondary_structure.shape))))
            secondary = secondary_structure
        if block_adjacency is not None:
            if block_adjacency.shape != (len(rows), len(rows), 3):
                fail("value_invalid", RequestRef("block_adjacency", str(tuple(block_adjacency.shape))))
            adjacency = block_adjacency
        secondary = self._feature(secondary, "RFdiffusion secondary structure")
        adjacency = self._feature(adjacency, "RFdiffusion block adjacency")
        if secondary_structure_overrides:
            secondary = secondary.clone()
            labels = torch.eye(4, device=self.device, dtype=self.dtype)
            for override in secondary_structure_overrides:
                selected = _residue_rows(override.targets, realization.source_rows, realization.placement_rows)
                secondary[selected] = labels[_LABELS[override.kind]]
        hotspot_mask = torch.zeros(len(rows), dtype=torch.bool, device=self.device)
        if hotspot is not None:
            hotspot_mask[_endpoint_rows(hotspot.residues, realization.source_rows, realization.placement_rows)] = True

        noisy = project_model_coordinates(
            torch.from_numpy(coordinates).to(self.device),
            torch.as_tensor(coordinate_mask, device=self.device),
            dtype=self.dtype,
            feature="RFdiffusion source coordinates",
        )
        motif_mask = torch.as_tensor(motif, device=self.device)
        chain_index = np.asarray(realization.instance_index, dtype=np.int64)
        units, unit_rows = _symmetry_unit_rows(len(rows), rotations, device=self.device)
        features = RfDiffusionInput(  # type: ignore[call-arg]
            noisy_coordinates=noisy[None],
            noisy_coordinate_mask=torch.as_tensor(coordinate_mask, device=self.device)[None],
            residue_type_features=F.one_hot(torch.as_tensor(tokens, device=self.device), num_classes=len(TOKENS)).to(
                self.dtype
            )[None],
            residue_index=torch.as_tensor(_residue_index(chain_index), device=self.device)[None],
            chain_index=torch.as_tensor(chain_index, device=self.device)[None],
            motif_mask=motif_mask[None],
            sequence_visible_mask=torch.as_tensor(visible, device=self.device)[None],
            hotspot_mask=hotspot_mask[None],
            cyclic_mask=torch.as_tensor(_cyclic_rows(tuple(bonds), realization, placement), device=self.device)[None],
            secondary_structure=secondary[None],
            block_adjacency=adjacency[None],
            timestep=torch.tensor([timestep], dtype=torch.int64, device=self.device),
            symmetry_residue_index=unit_rows,
            symmetry_rotation=_symmetry_rotation(rotations, dtype=self.dtype, device=self.device),
            symmetry_radial_axis=_symmetry_radial_axis(rotations, dtype=self.dtype, device=self.device),
            symmetry_radial_mask=torch.full(
                (units,), symmetry is not None and symmetry.kind == "octahedral", dtype=torch.bool, device=self.device
            ),
            potentials=_lower_potentials(
                tuple(potentials),
                structure,
                realization,
                slots,
                noisy,
                motif_mask,
                dtype=self.dtype,
                device=self.device,
            ),
            batch_size=[],
        )
        records = Records.concat((realization.records, component_records("encoding_loss", losses.component), accepted))
        record = PreparationRecord(records, np.zeros(len(system.component_symbol), dtype=np.uint8))
        return (
            PreparedInput(features, PreparationContext(target, slots, scope, record, spec=self.preparation_spec)),
            realization.placement_map,
        )

    def _feature(self, values: Tensor, feature: str) -> Tensor:
        r"""One feature tensor lowered to this processor's dtype and device."""
        lowered = project_model_features(
            values, torch.ones_like(values, dtype=torch.bool), dtype=self.dtype, feature=feature
        )
        return lowered.to(self.device)


def _mask_of(rows: Sequence[int], count: int) -> NDArray[np.bool_]:
    mask = np.zeros(count, dtype=bool)
    mask[list(rows)] = True
    return mask


def _protein_instances(structure: Structure) -> list[str | None]:
    r"""Per source instance, its symbol when every component of it is a protein residue, else ``None``."""
    system = structure.system
    monomer = CLASS_MONOMER[token_classes(system).token]
    pointer = system.instance_component_ptr
    symbols = system.text("instance", system.instance_symbol).tolist()
    return [
        str(symbol) if (monomer[pointer[instance] : pointer[instance + 1]] == _PROTEIN).all() else None
        for instance, symbol in enumerate(symbols)
    ]


def _default_placement(structure: Structure | None) -> RfDiffusionPlacements | None:
    r"""One group per source protein instance, each the whole instance."""
    if structure is None:
        return None
    pointer = structure.system.instance_component_ptr
    placements: PlacementSpec = {
        symbol: {symbol: ExistingSpan(symbol, 0, int(pointer[instance + 1] - pointer[instance]))}
        for instance, symbol in enumerate(_protein_instances(structure))
        if symbol is not None
    }
    return RfDiffusionPlacements(placements) if placements else None


def _validate_receptor_groups(placement: RfDiffusionPlacements) -> None:
    r"""A receptor group is one source instance stated as existing spans only."""
    unknown = placement.receptor_groups - placement.placements.keys()
    if unknown:
        fail("value_invalid", RequestRef("receptor_groups", str(sorted(unknown))))
    for group in placement.receptor_groups:
        values = tuple(placement.placements[group].values())
        spans = tuple(value for value in values if isinstance(value, ExistingSpan))
        if not spans or len(spans) != len(values):
            fail("value_invalid", RequestRef("receptor_groups", group), reason="only existing spans")
        if len({span.instance_id for span in spans}) != 1:
            fail("value_invalid", RequestRef("receptor_groups", group), reason="one source instance")


def _realize(
    structure: Structure | None,
    placement: RfDiffusionPlacements,
    rotations: Sequence[Tensor],
    *,
    bonds: Sequence[SymbolicBond],
    length: tuple[int, int] | None,
    generator: np.random.Generator | None,
    library: ComponentLibrary | None,
) -> _Realization:
    r"""Realise the output rows, build the target they describe and bind the maps its lowering reads them by."""
    rows, group_lengths = _realize_rows(placement, structure, length=length, generator=generator)
    if not rows:
        fail("value_invalid", RequestRef("placement", "empty"), reason="no realized output residue")
    units = _symmetry_units(len(rows), rotations)
    instance_index, references = _output_references(rows, group_lengths, units)
    placement_map: PlacementMap = {
        group: {placement_id: () for placement_id in placements} for group, placements in placement.placements.items()
    }
    for row, value in enumerate(rows):
        placement_map[value.group][value.placement] += (references[row],)
    source_rows = _source_rows(structure, rows)
    placement_rows: dict[str, tuple[int, ...]] = {}
    for row, value in enumerate(rows):
        placement_rows[value.placement] = placement_rows.get(value.placement, ()) + (row,)
    ids = _row_component_ids(structure, rows)
    target, records = _build_target(
        structure,
        rows,
        ids,
        instance_index,
        references,
        recipe_bonds=_recipe_bonds(bonds, references, source_rows, placement_map, placement),
        inherited=_inherited_edges(structure, rows, references, instance_index),
        units=units,
        library=library,
    )
    return _Realization(
        tuple(rows),
        target,
        records,
        tuple(references),
        tuple(instance_index),
        placement_map,
        source_rows,
        placement_rows,
        tuple(ids),
    )


def _realize_rows(
    placement: RfDiffusionPlacements,
    structure: Structure | None,
    *,
    length: tuple[int, int] | None,
    generator: np.random.Generator | None,
) -> tuple[list[_Row], list[tuple[str, int]]]:
    r"""The output rows in placement order (non-receptor groups first) and the length of every group."""
    placement_instance_of(placement.placements)
    _validate_receptor_groups(placement)
    if length is not None:
        lower, upper = (operator.index(value) for value in length)
        if lower < 0 or upper < lower:
            fail("value_invalid", RequestRef("length", str(length)))
        length = lower, upper

    variable: list[tuple[str, GeneratedSegment]] = []
    fixed = 0
    spans: dict[str, NDArray[np.int64]] = {}
    protein = [] if structure is None else _protein_instances(structure)
    for group, values in placement.placements.items():
        for placement_id, value in values.items():
            if isinstance(value, GeneratedSegment):
                if value.polymer_type != "protein":
                    fail("value_invalid", RequestRef(placement_id, value.polymer_type), expected="protein")
                variable.append((placement_id, value))
                continue
            spans[placement_id] = _span_components(structure, value, protein)
            if group not in placement.receptor_groups:
                fixed += value.stop - value.start

    minimum = fixed + sum(segment.min_length for _, segment in variable)
    maximum = fixed + sum(segment.max_length for _, segment in variable)
    if length is not None and (length[1] < minimum or length[0] > maximum):
        fail("value_invalid", RequestRef("length", str(length)), realizable=(minimum, maximum))
    sampled = _sample_lengths(variable, fixed, length, generator)

    ordered = [group for group in placement.placements if group not in placement.receptor_groups]
    ordered += [group for group in placement.placements if group in placement.receptor_groups]
    rows: list[_Row] = []
    group_lengths: list[tuple[str, int]] = []
    for group in ordered:
        start = len(rows)
        for placement_id, value in placement.placements[group].items():
            if isinstance(value, ExistingSpan):
                rows += [
                    _Row(group, placement_id, int(component), offset)
                    for offset, component in enumerate(spans[placement_id])
                ]
            else:
                rows += [_Row(group, placement_id, -1, offset) for offset in range(sampled[placement_id])]
        group_lengths.append((group, len(rows) - start))
    return rows, group_lengths


def _span_components(
    structure: Structure | None, span: ExistingSpan, protein: Sequence[str | None]
) -> NDArray[np.int64]:
    r"""The source components of one existing span: the ``[start, stop)`` positions of a protein instance."""
    if structure is None:
        fail("value_invalid", RequestRef("ExistingSpan", span.instance_id), reason="no molecular input")
    instance = int(structure.system.instance_index([span.instance_id])[0])
    if instance < 0 or protein[instance] is None:
        fail("value_invalid", RequestRef("ExistingSpan", span.instance_id), reason="not a protein instance")
    pointer = structure.system.instance_component_ptr
    first, last = int(pointer[instance]), int(pointer[instance + 1])
    if span.stop > last - first:
        fail("value_invalid", RequestRef("ExistingSpan", f"{span.start}..{span.stop}"), length=last - first)
    return np.arange(first + span.start, first + span.stop, dtype=np.int64)


def _sample_lengths(
    variable: Sequence[tuple[str, GeneratedSegment]],
    fixed: int,
    length: tuple[int, int] | None,
    generator: np.random.Generator | None,
) -> dict[str, int]:
    r"""One length per variable segment whose non-receptor total lies within ``length``."""
    draw = np.random.default_rng() if generator is None else generator
    for _attempt in range(_LENGTH_DRAWS):
        sampled = {
            placement_id: (
                segment.min_length
                if segment.min_length == segment.max_length
                else int(draw.integers(segment.min_length, segment.max_length + 1))
            )
            for placement_id, segment in variable
        }
        if length is None or length[0] <= fixed + sum(sampled.values()) <= length[1]:
            return sampled
    fail("length_search_limit", RequestRef("length", str(length)), draws=_LENGTH_DRAWS)


def _symmetry_rotations(symmetry: RfDiffusionPointGroupSymmetry | None) -> tuple[Tensor, ...]:
    r"""The rotations of one point group, empty without symmetry."""
    if symmetry is None:
        return ()
    if symmetry.kind in ("cyclic", "dihedral") and symmetry.order is None:
        fail("value_invalid", RequestRef("symmetry.order", "None"), kind=symmetry.kind)
    return point_group_rotations(symmetry.kind, symmetry.order)


def _symmetry_units(count: int, rotations: Sequence[Tensor]) -> int:
    r"""The number of symmetry units, ``0`` without symmetry: they cover the output in equal contiguous blocks."""
    if not rotations:
        return 0
    if count % len(rotations):
        fail("value_invalid", RequestRef("symmetry", str(len(rotations))), rows=count)
    return len(rotations)


def _symmetry_unit_rows(count: int, rotations: Sequence[Tensor], *, device: torch.device) -> tuple[int, Tensor]:
    r"""The number of symmetry units and the rows of each unit."""
    units = _symmetry_units(count, rotations)
    if not units:
        return 0, torch.empty((0, 0), dtype=torch.int64, device=device)
    return units, torch.arange(count, dtype=torch.int64, device=device).reshape(units, count // units)


def _symmetry_rotation(rotations: Sequence[Tensor], *, dtype: torch.dtype, device: torch.device) -> Tensor:
    if not rotations:
        return torch.empty((0, 3, 3), dtype=dtype, device=device)
    return torch.stack(tuple(rotations)).to(device=device, dtype=dtype)


def _symmetry_radial_axis(rotations: Sequence[Tensor], *, dtype: torch.dtype, device: torch.device) -> Tensor:
    r"""Per unit, the image of the ``x`` axis under its rotation."""
    matrices = _symmetry_rotation(rotations, dtype=dtype, device=device)
    axis = torch.tensor((1.0, 0.0, 0.0), dtype=dtype, device=device)
    return torch.einsum("uij,j->ui", matrices, axis) if len(rotations) else matrices.new_empty((0, 3))


def _output_references(
    rows: Sequence[_Row], group_lengths: Sequence[tuple[str, int]], units: int
) -> tuple[list[int], list[ComponentRef]]:
    r"""Per row, its output instance index and reference: one instance per group, or one per symmetry unit."""
    if units:
        unit = len(rows) // units
        indices = [row // unit for row in range(len(rows))]
        positions = [row % unit for row in range(len(rows))]
    else:
        indices, positions = [], []
        for index, (_group, group_length) in enumerate(group for group in group_lengths if group[1]):
            indices += [index] * group_length
            positions += list(range(group_length))
    return indices, [ComponentRef(f"chain_{index}", position) for index, position in zip(indices, positions)]


def _row_component_ids(structure: Structure | None, rows: Sequence[_Row]) -> list[str]:
    r"""Per row, the component id it is built from."""
    if structure is None:
        return [RFDIFFUSION_GENERATED_COMPONENT] * len(rows)
    system = structure.system
    ids = system.text("component", system.component_symbol)
    return [RFDIFFUSION_GENERATED_COMPONENT if row.source < 0 else str(ids[row.source]) for row in rows]


def _require_stated_definitions(
    structure: Structure, rows: Sequence[_Row], ids: Sequence[str], library: ComponentLibrary
) -> None:
    r"""
    Every copied row is built from the definition its source component states.

    Only the copied rows are compared, by the content digest of the definition: the scope it was read at and the
    artifact it came from do not matter, so two definitions of equal content agree.

    Raises:
        InvalidRequestError: ``definition_set_mismatch`` naming the source component whose chemistry the definitions
            in force state differently; the copy is refused instead of built around another chemistry.
    """

    stated = structure.system.component_definition_digest
    for row, value in enumerate(rows):
        if value.source < 0:
            continue
        definition = library.definition(ids[row])
        if (0 if definition is None else definition.digest) != stated[value.source]:
            fail("definition_set_mismatch", StructureRef(component=value.source), id=ids[row])


def _build_target(
    structure: Structure | None,
    rows: Sequence[_Row],
    ids: Sequence[str],
    instance_index: Sequence[int],
    references: Sequence[ComponentRef],
    *,
    recipe_bonds: Sequence[RecipeBond],
    inherited: InheritedEdges | None,
    units: int,
    library: ComponentLibrary | None,
) -> tuple[Structure, Records]:
    r"""
    The realised output with the records of its construction: one polymer entity per output instance, its rows the
    generated unknown residue or the source component they copy, linked wherever the output declares a junction, plus
    the bonds the request declares and the source connections at copied rows (``_inherited_edges``: edges between
    copied rows, external connections to partners the output does not copy), built in one pass, so a copied residue
    loses exactly the atoms its connections remove. A copied atom keeps the coordinate evidence its source states;
    every other atom has none.

    A copied row is built from the definition of its component id, checked before anything is built
    (``_require_stated_definitions``), so a copy is never built around a chemistry other than its source's.
    """

    library = configured_library() if library is None else library
    if structure is not None:
        _require_stated_definitions(structure, rows, ids, library)
    linked = _row_links(structure, rows, instance_index)
    entities: dict[str, PolymerSequence] = {}
    instances: dict[str, str] = {}
    for index in dict.fromkeys(instance_index):
        selected = [row for row in range(len(rows)) if instance_index[row] == index]
        entity = "rfdiffusion_entity_0" if units else f"rfdiffusion_entity_{index}"
        sequence = PolymerSequence(
            "polypeptide(L)",
            tuple(ids[row] for row in selected),
            tuple(linked[row] for row in selected[:-1]),
        )
        if entities.get(entity, sequence) != sequence:
            fail("value_invalid", RequestRef("symmetry", f"chain_{index}"), reason="symmetry units differ")
        entities[entity] = sequence
        instances[references[selected[0]].instance] = entity
    recipe = Recipe(entities, instances, tuple(recipe_bonds))
    system, _definitions, records = build_system(recipe, library, inherited=inherited)
    target = Structure.unobserved(system)
    if structure is None:
        return target, records
    atoms, coordinates, evidence = _copied_coordinates(structure, system, rows)
    for value in np.unique(evidence):
        group = evidence == value
        scope = np.zeros(len(system.atom_symbol), dtype=bool)
        scope[atoms[group]] = True
        target, _report = replace_coordinates(
            target, atoms[group], coordinates[group], scope=scope, evidence=CoordinateEvidence(int(value))
        )
    return target, records


def _row_links(structure: Structure | None, rows: Sequence[_Row], instance_index: Sequence[int]) -> list[bool]:
    r"""
    Per row, whether the output bonds it to the next row of its instance: a generated row or a placement boundary is
    always a junction; two copied rows of one placement are joined exactly when the source joins them.
    """

    junctions = _source_junctions(structure)
    links = []
    for row in range(len(rows)):
        if row + 1 == len(rows) or instance_index[row] != instance_index[row + 1]:
            links.append(False)
            continue
        left, right = rows[row], rows[row + 1]
        if left.source < 0 or right.source < 0 or left.placement != right.placement:
            links.append(True)
        else:
            links.append((left.source, right.source) in junctions)
    return links


def _source_junctions(structure: Structure | None) -> set[tuple[int, int]]:
    r"""The source's polymer junctions as ``(previous component, next component)``: an edge from one component's next
    port to another's previous port."""
    if structure is None:
        return set()
    system = structure.system
    port = system.component_port.astype(np.int64)
    atoms = len(system.atom_symbol)
    next_owner, previous_owner = np.full(atoms, -1, dtype=np.int64), np.full(atoms, -1, dtype=np.int64)
    has_next, has_previous = port[:, 1] >= 0, port[:, 0] >= 0
    next_owner[port[has_next, 1]] = np.flatnonzero(has_next)
    previous_owner[port[has_previous, 0]] = np.flatnonzero(has_previous)
    first, second = system.edge_index.astype(np.int64)
    junctions: set[tuple[int, int]] = set()
    for left, right in ((first, second), (second, first)):
        joined = (next_owner[left] >= 0) & (previous_owner[right] >= 0)
        junctions.update(zip(next_owner[left][joined].tolist(), previous_owner[right][joined].tolist()))
    return junctions


def _source_rows(structure: Structure | None, rows: Sequence[_Row]) -> dict[ComponentRef, tuple[int, ...]]:
    r"""Per copied source component reference, the output rows that copy it."""
    if structure is None:
        return {}
    components = np.array(sorted({row.source for row in rows if row.source >= 0}), dtype=np.int64)
    reference_of = dict(zip(components.tolist(), structure.system.component_refs(components)))
    found: dict[ComponentRef, tuple[int, ...]] = {}
    for row, value in enumerate(rows):
        if value.source >= 0:
            reference = reference_of[value.source]
            found[reference] = found.get(reference, ()) + (row,)
    return found


def _recipe_bonds(
    bonds: Sequence[SymbolicBond],
    references: Sequence[ComponentRef],
    source_rows: Mapping[ComponentRef, tuple[int, ...]],
    placement_map: PlacementMap,
    placement: RfDiffusionPlacements,
) -> tuple[RecipeBond, ...]:
    r"""The requested bonds as recipe declarations over the output rows."""
    recipe = []
    for bond in bonds:
        if bond.aromatic or bond.stereo is not None or bond.direction != "undirected":
            fail("value_invalid", RequestRef("bonds", "an aromatic, directed or stereo bond"))
        first, second = _bond_sites(bond, references, source_rows, placement_map, placement)
        recipe.append(
            RecipeBond(
                first,
                second,
                None if bond.order is None else int(bond.order),
                BondKind.COVALENT if bond.kind == "covalent" else BondKind.COORDINATION,
                (tuple(sorted(bond.leaving_a)), tuple(sorted(bond.leaving_b))),
            )
        )
    return tuple(recipe)


def _bond_sites(
    bond: SymbolicBond,
    references: Sequence[ComponentRef],
    source_rows: Mapping[ComponentRef, tuple[int, ...]],
    placement_map: PlacementMap,
    placement: RfDiffusionPlacements,
) -> tuple[AtomSite, AtomSite]:
    r"""Both endpoints of one bond as output atoms: a symbolic site through the placement map, a source atom through
    the one row that copies its component."""
    instance_of = placement_instance_of(placement.placements)

    def site(endpoint: AtomSite | SymbolicAtomSite) -> AtomSite:
        if isinstance(endpoint, SymbolicAtomSite):
            reference = resolve_placement(endpoint.residue, placement_map, instance_of)
            return AtomSite(reference.instance, reference.component, endpoint.atom_id)
        found = source_rows.get(ComponentRef(endpoint.instance, endpoint.component), ())
        if len(found) != 1:
            fail("request_invalid_positions", RequestRef("bonds", str(endpoint)), matches=len(found))
        reference = references[found[0]]
        return AtomSite(reference.instance, reference.component, endpoint.atom)

    return site(bond.site_a), site(bond.site_b)


def _inherited_edges(
    structure: Structure | None,
    rows: Sequence[_Row],
    references: Sequence[ComponentRef],
    instance_index: Sequence[int],
) -> InheritedEdges | None:
    r"""
    The source's connections at copied components, restated on the rows that copy them with their resolved states.

    Polymer junctions and caps are never restated: the output's own links state every copied row's ports, joining it
    to its output neighbours or ending its chain. Every other connection (a disulfide, a cross-link, a coordination,
    a branch link) is kept. Between two copied components it is an edge: a component copied once takes it to wherever
    its partner lands; between components copied several times, each copy (placement and output instance) takes the
    connections whose two ends it copies. To a component no row copies it is, as a projection keeps it, an external
    connection of every copy of the kept atom, naming the partner by its source instance symbol, ordinal and atom
    name, so the atoms it removes stay absent; leaving ids its partner declares are not restated, as the output holds
    none of its atoms. An external connection the source holds at a copied atom is kept the same way.

    Raises:
        InvalidRequestError: ``request_invalid_positions`` for a connection between components copied several times
            whose two ends no one copy holds, one to an uncopied partner whose source instance symbol is an output
            instance's (the external connection would name an atom of the output), or a source external connection
            through a symmetry operation (the output does not hold the source's frame); ``value_invalid`` for a
            connection with a bond configuration, which no recipe states.
    """

    if structure is None:
        return None
    system = structure.system
    copies: dict[int, dict[tuple[str, int], list[int]]] = {}
    for row, value in enumerate(rows):
        if value.source >= 0:
            copies.setdefault(value.source, {}).setdefault((value.placement, instance_index[row]), []).append(row)
    owner = np.repeat(np.arange(len(system.component_symbol)), np.diff(system.component_atom_ptr))
    instance_of = np.repeat(np.arange(len(system.instance_symbol)), np.diff(system.instance_component_ptr))
    copied = np.isin(owner, list(copies))
    first, second = system.edge_index.astype(np.int64)
    ports = (BondOrigin.JUNCTION, BondOrigin.CAP)
    linking = ~np.isin(system.bond_origin, ports)
    selected = np.flatnonzero((copied[first] | copied[second]) & (owner[first] != owner[second]) & linking)
    configured = selected[np.isin(selected, system.stereobond_edge)]
    if len(configured):
        edge = int(configured[0])
        fail("value_invalid", StructureRef(atom=int(first[edge])), partner=int(second[edge]), reason="configuration")
    symbols = system.symbols()
    outputs = {reference.instance for reference in references}
    edges: list[int] = []
    sites: list[tuple[AtomSite, AtomSite]] = []
    flipped: list[bool] = []
    outside: list[bool] = []
    for edge in selected.tolist():
        ends = (int(first[edge]), int(second[edge]))
        names = [symbols["atom"][int(system.atom_symbol[atom])] for atom in ends]
        if not (copied[ends[0]] and copied[ends[1]]):  # cut: the partner stays outside the output
            flip = not copied[ends[0]]
            kept, cut = ends[::-1] if flip else ends
            instance = int(instance_of[owner[cut]])
            symbol = symbols["instance"][int(system.instance_symbol[instance])]
            if symbol in outputs:
                fail("request_invalid_positions", StructureRef(atom=kept), partner=cut, outside=symbol)
            partner = AtomSite(symbol, int(owner[cut] - system.instance_component_ptr[instance]), names[flip ^ 1])
            for row in (row for found in copies[int(owner[kept])].values() for row in found):
                edges.append(edge)
                flipped.append(flip)
                outside.append(True)
                sites.append((AtomSite(references[row].instance, references[row].component, names[flip]), partner))
            continue
        left, right = (copies[int(owner[atom])] for atom in ends)
        pairs = _copy_pairs(left, right)
        if pairs is None:
            matches = sum(map(len, left.values())) * sum(map(len, right.values()))
            fail("request_invalid_positions", StructureRef(atom=ends[0]), partner=ends[1], matches=matches)
        for one, other in pairs:
            edges.append(edge)
            flipped.append(False)
            outside.append(False)
            sites.append(
                (
                    AtomSite(references[one].instance, references[one].component, names[0]),
                    AtomSite(references[other].instance, references[other].component, names[1]),
                )
            )
    externals: list[int] = []
    held = copied[system.external_atom] & ~np.isin(system.external_origin, ports)
    for external in np.flatnonzero(held).tolist():
        kept = int(system.external_atom[external])
        if system.external_transform[external] != Transforms.IDENTITY:
            fail("request_invalid_positions", StructureRef(atom=kept, external=external), reason="symmetry")
        symbol = symbols["instance"][int(system.external_partner_instance[external])]
        if symbol in outputs:
            fail("request_invalid_positions", StructureRef(atom=kept, external=external), outside=symbol)
        partner = AtomSite(
            symbol,
            int(system.external_partner_component[external]),
            symbols["atom"][int(system.external_partner_atom[external])],
        )
        name = symbols["atom"][int(system.atom_symbol[kept])]
        for row in (row for found in copies[int(owner[kept])].values() for row in found):
            externals.append(external)
            outside.append(True)
            sites.append((AtomSite(references[row].instance, references[row].component, name), partner))
    if not sites:
        return None

    def ids(table: str, row: int, side: int) -> tuple[str, ...]:
        pointer = getattr(system, f"{table}_declared_leaving_ptr")
        codes, sides = getattr(system, f"{table}_declared_leaving"), getattr(system, f"{table}_declared_leaving_side")
        return tuple(symbols["atom"][int(codes[k])] for k in range(pointer[row], pointer[row + 1]) if sides[k] == side)

    # the partner's own ids are never restated at an outside end: the output holds none of its atoms
    leaving = tuple(
        (ids("bond", edge, int(flip)), () if out else ids("bond", edge, int(not flip)))
        for edge, flip, out in zip(edges, flipped, outside)
    ) + tuple((ids("external", external, 0), ()) for external in externals)
    edge_rows, external_rows = np.array(edges, dtype=np.int64), np.array(externals, dtype=np.int64)
    columns: dict[str, NDArray] = {}
    for name in INHERITED_COLUMNS:
        value, restated = getattr(system, name), getattr(system, name.replace("bond_", "external_", 1))
        columns[name] = np.concatenate((value[edge_rows], restated[external_rows].astype(value.dtype)))
    columns["bond_direction"][: len(edges)] *= np.where(flipped, -1, 1).astype(columns["bond_direction"].dtype)
    types = system.bond_connection_type[edge_rows].tolist() + system.external_connection_type[external_rows].tolist()
    return InheritedEdges(
        tuple(a for a, _ in sites),
        tuple(b for _, b in sites),
        columns,
        tuple(symbols["connection_type"][code] if code >= 0 else None for code in types),
        leaving,  # type: ignore[arg-type]
        tuple(outside),
    )


def _copy_pairs(
    left: Mapping[tuple[str, int], list[int]], right: Mapping[tuple[str, int], list[int]]
) -> list[tuple[int, int]] | None:
    r"""The row pairs one connection lands on: the two rows of components copied once, else per copy holding both
    ends its two rows; ``None`` when a copy holds one end only, or an end twice."""
    once = [row for rows in left.values() for row in rows], [row for rows in right.values() for row in rows]
    if len(once[0]) == 1 and len(once[1]) == 1:
        return [(once[0][0], once[1][0])]
    if left.keys() != right.keys() or any(len(rows) != 1 for rows in (*left.values(), *right.values())):
        return None
    return [(left[key][0], right[key][0]) for key in left]


def _copied_coordinates(
    structure: Structure, system: MolecularSystem, rows: Sequence[_Row]
) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.uint8]]:
    r"""The output atoms of every copied row the source places, with the source coordinates and evidence they take."""
    source = structure.system
    source_names = source.text("atom", source.atom_symbol)
    names = system.text("atom", system.atom_symbol)
    placed = structure.coordinate_mask
    atoms: list[int] = []
    sources: list[int] = []
    for row, value in enumerate(rows):
        if value.source < 0:
            continue
        start, stop = int(source.component_atom_ptr[value.source]), int(source.component_atom_ptr[value.source + 1])
        available = {str(source_names[atom]): atom for atom in range(start, stop) if placed[atom]}
        for atom in range(int(system.component_atom_ptr[row]), int(system.component_atom_ptr[row + 1])):
            found = available.get(str(names[atom]))
            if found is not None:
                atoms.append(atom)
                sources.append(found)
    taken = np.array(sources, dtype=np.int64)
    return np.array(atoms, dtype=np.int64), structure.coordinates[taken], structure.coordinate_evidence[taken]


def _row_coordinates(target: Structure, slots: SlotMap) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    r"""The target's coordinates on the row-by-atom14 model axis, with their presence mask."""
    atom = slots.atom.reshape(-1, _ATOM14)
    mask = (atom >= 0) & target.coordinate_mask[np.maximum(atom, 0)]
    return np.where(mask[..., None], target.coordinates[np.maximum(atom, 0)], 0.0), mask


def _require_backbone(rows: NDArray[np.int64], mask: NDArray[np.bool_]) -> None:
    r"""Every row the model reads as placed needs its N, CA and C: the frame it is diffused around."""
    missing = rows[~mask[rows, :3].all(axis=1)]
    if len(missing):
        fail(
            "prepare_unknown_feature",
            StructureRef(component=int(missing[0])),
            column="coordinates",
            atoms=("N", "CA", "C"),
        )


def _require_positional_source(structure: Structure | None, rows: Sequence[_Row]) -> None:
    r"""Partial diffusion diffuses the whole source: every row copies the source component at its own position."""
    if structure is None:
        fail("value_invalid", RequestRef("partial_diffusion", "without a source structure"))
    pointer = structure.system.instance_component_ptr
    expected = [
        component
        for instance, symbol in enumerate(_protein_instances(structure))
        if symbol is not None
        for component in range(int(pointer[instance]), int(pointer[instance + 1]))
    ]
    if [row.source for row in rows] != expected:
        fail("value_invalid", RequestRef("partial_diffusion", "source/output correspondence"))


def _selected_rows(
    structure: Structure | None,
    selection: PositionSelection | None,
    argument: str,
    source_rows: Mapping[ComponentRef, tuple[int, ...]],
) -> list[int]:
    r"""The output rows a source selection names; a position no row copies is ``request_invalid_positions``."""
    if selection is None:
        return []
    if structure is None:
        fail("request_invalid_positions", RequestRef(argument, "without a source structure"))
    allowed = np.zeros(len(structure.system.component_symbol), dtype=bool)
    allowed[structure.system.component_index(list(source_rows))] = True
    components = resolve_positions(structure, selection, allowed=allowed)
    return [row for reference in structure.system.component_refs(components) for row in source_rows[reference]]


def _symbolic_row(site: SymbolicResidueSite, placement_rows: Mapping[str, tuple[int, ...]]) -> int:
    r"""The row one symbolic residue names."""
    found = placement_rows.get(site.placement_id, ())
    if not -len(found) <= site.offset < len(found):
        fail("value_invalid", RequestRef(site.placement_id, str(site.offset)), length=len(found))
    return found[site.offset]


def _residue_rows(
    references: Sequence[ResidueEndpoint],
    source_rows: Mapping[ComponentRef, tuple[int, ...]],
    placement_rows: Mapping[str, tuple[int, ...]],
) -> list[int]:
    r"""The rows a residue-endpoint sequence names, in order and without repetition."""
    rows: list[int] = []
    for reference in references:
        if isinstance(reference, str):
            rows += list(placement_rows[reference])
        elif isinstance(reference, SymbolicResidueSite):
            rows.append(_symbolic_row(reference, placement_rows))
        else:
            found = source_rows.get(reference)
            if found is None:
                fail("request_invalid_positions", RequestRef("positions", str(reference)))
            rows += list(found)
    return list(dict.fromkeys(rows))


def _endpoint_rows(
    endpoints: Sequence[Endpoint],
    source_rows: Mapping[ComponentRef, tuple[int, ...]],
    placement_rows: Mapping[str, tuple[int, ...]],
) -> list[int]:
    r"""The rows an endpoint sequence names, atom endpoints read as their residue."""
    residues: list[ResidueEndpoint] = []
    for endpoint in endpoints:
        if isinstance(endpoint, AtomSite):
            residues.append(ComponentRef(endpoint.instance, endpoint.component))
        elif isinstance(endpoint, SymbolicAtomSite):
            residues.append(endpoint.residue)
        else:
            residues.append(endpoint)
    return _residue_rows(residues, source_rows, placement_rows)


def _residue_index(chain_index: NDArray[np.int64]) -> NDArray[np.int64]:
    r"""Upstream's residue index: the row plus 200 per earlier chain, so two chains never neighbour in sequence."""
    return np.arange(len(chain_index), dtype=np.int64) + 200 * chain_index


def _cyclic_rows(
    bonds: Sequence[SymbolicBond], realization: _Realization, placement: RfDiffusionPlacements
) -> NDArray[np.bool_]:
    r"""Per row, whether a requested bond closes its instance into a cycle (its first N to its last C)."""
    cyclic = np.zeros(len(realization.rows), dtype=bool)
    row_of = {reference: row for row, reference in enumerate(realization.references)}
    for bond in bonds:
        first, second = _bond_sites(
            bond, realization.references, realization.source_rows, realization.placement_map, placement
        )
        if first.instance != second.instance or {first.atom, second.atom} != {"N", "C"}:
            continue
        members = [row for row, reference in enumerate(realization.references) if reference.instance == first.instance]
        head, tail = (first, second) if first.atom == "N" else (second, first)
        head_row = row_of[ComponentRef(head.instance, head.component)]
        tail_row = row_of[ComponentRef(tail.instance, tail.component)]
        if head_row == members[0] and tail_row == members[-1]:
            cyclic[members] = True
    return cyclic


def _atom14_names(component_id: str) -> tuple[str | None, ...]:
    r"""The atom14 names of one component id: its own table, else the unknown residue's (upstream's fallback)."""
    return RFDIFFUSION_ATOM14_TABLE.get(component_id, _UNKNOWN_ATOM14)


def _fold_features(
    template: Template | None, structure: Structure | None, realization: _Realization
) -> tuple[Tensor, Tensor]:
    r"""
    Lower a template's secondary-structure labels and contact blocks onto the output rows.

    The alignment's target positions are source positions: each is remapped to the output rows copying it, and the
    remapped alignments' placement frames (``templates.template_alignment_frames``) decide which rows may share a
    contact block, so labels and contacts never cross two placements of one template chain.
    """

    count = len(realization.rows)
    labels = torch.full((count,), 3, dtype=torch.int64)
    block = torch.full((count, count), 2, dtype=torch.int64)
    if template is None:
        return F.one_hot(labels, 4).float(), F.one_hot(block, 3).float()
    if structure is None:
        fail("value_invalid", RequestRef("templates", "without a molecular input"))
    resolved = _remap_alignments(template, structure, realization)
    frames = template_alignment_frames(resolved)
    frame_of_row = torch.full((count,), -1, dtype=torch.int64)
    placed: dict[int, Tensor] = {}
    chains: dict[str, tuple[Tensor, Tensor, Tensor]] = {}
    row_of = {
        (instance, reference.component): row
        for row, (instance, reference) in enumerate(zip(realization.instance_index, realization.references))
    }
    for alignment, frame in zip(resolved, frames, strict=True):
        if alignment.template_instance_id not in chains:
            chains[alignment.template_instance_id] = _template_chain(template, alignment.template_instance_id)
        coordinates, mask, chain_labels = chains[alignment.template_instance_id]
        for target_position, source_position in alignment.residue_pairs:
            if not bool(mask[source_position, :3].all()):
                fail("value_invalid", RequestRef("templates", f"position {source_position}"), reason="no N/CA/C")
            row = row_of[(alignment.target_instance_index, target_position)]
            placed[row] = coordinates[source_position]
            labels[row] = chain_labels[source_position]
            frame_of_row[row] = frame
    if not placed:
        fail("value_invalid", RequestRef("templates", "no covered output row"))
    return _fold_blocks(labels, block, frame_of_row, placed, realization.instance_index)


def _remap_alignments(
    template: Template, structure: Structure, realization: _Realization
) -> tuple[ResolvedTemplateAlignment, ...]:
    r"""The template's alignments on the output: each resolved against the source once (``resolve_template``), then
    one alignment per (placement, output instance), its target positions those of the rows that copy the aligned
    source components."""
    pointer = structure.system.instance_component_ptr
    rows_of_component: dict[int, list[int]] = {}
    for row, value in enumerate(realization.rows):
        if value.source >= 0:
            rows_of_component.setdefault(value.source, []).append(row)
    remapped: list[ResolvedTemplateAlignment] = []
    for alignment in resolve_template(template, structure.system, monomer_types=(_PROTEIN,)):
        start = int(pointer[alignment.target_instance_index])
        placements: dict[tuple[str, int], list[tuple[int, int]]] = {}
        for target_position, source_position in alignment.residue_pairs:
            for row in rows_of_component.get(start + target_position, ()):
                key = (realization.rows[row].placement, realization.instance_index[row])
                placements.setdefault(key, []).append((realization.references[row].component, source_position))
        remapped += [
            ResolvedTemplateAlignment(alignment.template_instance_id, instance, tuple(sorted(set(pairs))))
            for (_placement, instance), pairs in placements.items()
        ]
    return tuple(remapped)


def _template_chain(template: Template, chain_id: str) -> tuple[Tensor, Tensor, Tensor]:
    r"""One template chain on the atom14 axis: its coordinates, their mask and the secondary-structure label of every
    observed segment."""
    chain = template.coordinates.chains[chain_id]
    count = len(chain.component_ids)
    coordinates = torch.zeros((count, _ATOM14, 3), dtype=torch.float64)
    mask = torch.zeros((count, _ATOM14), dtype=torch.bool)
    for position in range(count):
        values, present = template_atoms(chain, position, _atom14_names(chain.component_ids[position]))
        coordinates[position] = torch.from_numpy(values)
        mask[position] = torch.from_numpy(present)
    labels = torch.full((count,), 2, dtype=torch.int64)
    observed = mask[:, :3].all(dim=-1)
    start = 0
    while start < count:
        if not bool(observed[start]):
            start += 1
            continue
        stop = start + 1
        while stop < count and bool(observed[stop]):
            stop += 1
        labels[start:stop] = rfdiffusion_secondary_structure(coordinates[start:stop, 1])
        start = stop
    return coordinates, mask, labels


def _fold_blocks(
    labels: Tensor,
    block: Tensor,
    frame_of_row: Tensor,
    placed: Mapping[int, Tensor],
    instance_index: Sequence[int],
) -> tuple[Tensor, Tensor]:
    r"""The fold features of the placed rows: virtual C-beta contacts within one frame, as blocks of equal labels."""
    count = len(instance_index)
    rows = torch.tensor(tuple(placed), dtype=torch.int64)
    coordinates = torch.stack(tuple(placed.values()))
    nitrogen, calpha, carbon = coordinates[:, :3].unbind(dim=1)
    cbeta = (
        -0.57910144 * torch.cross(calpha - nitrogen, carbon - calpha, dim=-1)
        + 0.5689693 * (calpha - nitrogen)
        - 0.5441217 * (carbon - calpha)
        + calpha
    )
    contacts = torch.cdist(cbeta, cbeta) < 6.0
    contacts.fill_diagonal_(False)
    same_frame = (frame_of_row[:, None] == frame_of_row[None, :]) & (frame_of_row[:, None] >= 0)
    mapped = torch.zeros((count, count), dtype=torch.bool)
    mapped[rows[:, None], rows[None, :]] = contacts
    mapped &= same_frame
    segments: list[tuple[int, int, int]] = []
    start = 0
    for index in range(1, count + 1):
        if (
            index == count
            or labels[index] != labels[start]
            or instance_index[index] != instance_index[start]
            or frame_of_row[index] != frame_of_row[start]
        ):
            segments.append((int(labels[start]), start, index))
            start = index
    block.zero_()
    for position, (left_label, left_start, left_stop) in enumerate(segments):
        if left_label in (2, 3):
            continue
        for right_label, right_start, right_stop in segments[position + 1 :]:
            if right_label in (2, 3) or not bool(same_frame[left_start, right_start]):
                continue
            if bool(mapped[left_start:left_stop, right_start:right_stop].any()):
                block[left_start:left_stop, right_start:right_stop] = 1
                block[right_start:right_stop, left_start:left_stop] = 1
    determined = labels != 3
    block.masked_fill_(~(determined[:, None] & determined[None, :] & same_frame), 2)
    return F.one_hot(labels, 4).float(), F.one_hot(block, 3).float()


def _lower_potentials(
    potentials: Sequence[Potential],
    structure: Structure | None,
    realization: _Realization,
    slots: SlotMap,
    coordinates: Tensor,
    motif_mask: Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> RfDiffusionPotentialInput:
    r"""
    The guidance program over the atom14 model axis.

    An output atom is a point of the flattened ``row * 14 + slot`` axis; a source substrate atom has no output row and
    enters as a fixed point in the local frame of the first nondegenerate motif row, so guidance follows that motif.
    """

    atom = slots.atom.reshape(-1, _ATOM14)
    source_rows, placement_rows = realization.source_rows, realization.placement_rows
    program = _PotentialProgram(dtype=dtype, device=device)
    frame_row = _frame_row(coordinates, motif_mask)
    if frame_row is not None:
        frame_center = coordinates[frame_row, 1]
        frame_rotation = rotation_from_two_vectors(
            coordinates[frame_row, 2] - frame_center, coordinates[frame_row, 0] - frame_center, eps=0
        )
        program.frame(frame_row * _ATOM14)

    def internal(row: int, atom_name: str) -> int:
        names = _atom14_names(realization.component_ids[row])
        if atom_name not in names or int(atom[row, names.index(atom_name)]) < 0:
            fail("value_invalid", RequestRef("potentials", f"{atom_name} at row {row}"), reason="no such atom")
        return program.point(row * _ATOM14 + names.index(atom_name), torch.zeros(3, dtype=dtype, device=device))

    def external(site: AtomSite) -> int:
        if structure is None:
            fail("value_invalid", RequestRef("potentials", str(site)), reason="no source structure")
        if frame_row is None:
            fail("model_capacity", RequestRef("potentials", str(site)), reason="no nondegenerate motif frame")
        found = int(structure.system.atom_index([site])[0])
        if found < 0 or not bool(structure.coordinate_mask[found]):
            fail("value_invalid", RequestRef("potentials", str(site)), reason="no observed coordinates")
        value = project_model_coordinates(
            torch.from_numpy(structure.coordinates[found]).to(device),
            torch.ones(3, dtype=torch.bool, device=device),
            dtype=dtype,
            feature="RFdiffusion external potential point",
        )
        return program.point(-1, (value - frame_center) @ frame_rotation)

    def atom_points(site: AtomSite | SymbolicAtomSite) -> list[int]:
        if isinstance(site, SymbolicAtomSite):
            return [internal(_symbolic_row(site.residue, placement_rows), site.atom_id)]
        found = source_rows.get(ComponentRef(site.instance, site.component), ())
        return [internal(row, site.atom) for row in found] if found else [external(site)]

    def residue_points(references: Sequence[ResidueEndpoint]) -> list[int]:
        return [internal(row, "CA") for row in _residue_rows(references, source_rows, placement_rows)]

    for potential in potentials:
        if isinstance(potential, RfDiffusionCoordination):
            program.coordination(residue_points(potential.side_a), residue_points(potential.side_b), potential)
        elif isinstance(potential, RfDiffusionFlatBottomRadialPotential):
            program.radius(residue_points(potential.targets), potential)
        else:
            substrate = [point for site in potential.substrate_atoms for point in atom_points(site)]
            program.substrate(residue_points(potential.residues), substrate, potential)
    return program.lower()


def _frame_row(coordinates: Tensor, motif_mask: Tensor) -> int | None:
    r"""The first motif row whose N, CA and C span a nondegenerate frame."""
    for row in range(coordinates.shape[0]):
        if not bool(motif_mask[row]):
            continue
        normal = torch.cross(
            coordinates[row, 2] - coordinates[row, 1], coordinates[row, 0] - coordinates[row, 1], dim=-1
        )
        if torch.linalg.vector_norm(normal) > 0:
            return row
    return None


class _PotentialProgram:
    r"""The flat guidance program the Model consumes: its points, their frame and per potential its groups."""

    def __init__(self, *, dtype: torch.dtype, device: torch.device) -> None:
        self.dtype, self.device = dtype, device
        self.point_atom_index: list[int] = []
        self.point_local_coordinates: list[Tensor] = []
        self.frame_atom_index = torch.full((3,), -1, dtype=torch.int64, device=device)
        self.coordination_point_index: list[int] = []
        self.coordination_ptr = [0]
        self.coordination_center: list[float] = []
        self.coordination_scale: list[float] = []
        self.coordination_weight: list[float] = []
        self.radius_point_index: list[int] = []
        self.radius_ptr = [0]
        self.radius_threshold: list[float] = []
        self.radius_weight: list[float] = []
        self.substrate_point_index: list[int] = []
        self.substrate_ptr = [0]
        self.substrate_parameters: list[tuple[float, float, float, float, float, float]] = []
        self.substrate_nearest_only: list[bool] = []

    def frame(self, first_slot: int) -> None:
        r"""The N, CA and C points of the frame every external point is expressed in."""
        self.frame_atom_index = torch.tensor(
            [first_slot, first_slot + 1, first_slot + 2], dtype=torch.int64, device=self.device
        )

    def point(self, atom_index: int, local: Tensor) -> int:
        self.point_atom_index.append(atom_index)
        self.point_local_coordinates.append(local)
        return len(self.point_atom_index) - 1

    def coordination(self, left: Sequence[int], right: Sequence[int], potential: RfDiffusionCoordination) -> None:
        for side in (left, right):
            self.coordination_point_index += list(side)
            self.coordination_ptr.append(len(self.coordination_point_index))
        self.coordination_center.append(potential.center_angstrom)
        self.coordination_scale.append(potential.scale_angstrom)
        weight = potential.weight
        self.coordination_weight.append(weight if potential.direction == "maximize" else -weight)

    def radius(self, selected: Sequence[int], potential: RfDiffusionFlatBottomRadialPotential) -> None:
        self.radius_point_index += list(selected)
        self.radius_ptr.append(len(self.radius_point_index))
        self.radius_threshold.append(potential.radius_threshold_angstrom)
        self.radius_weight.append(potential.weight)

    def substrate(
        self, residues: Sequence[int], substrate: Sequence[int], potential: RfDiffusionSubstrateInteraction
    ) -> None:
        for side in (residues, substrate):
            self.substrate_point_index += list(side)
            self.substrate_ptr.append(len(self.substrate_point_index))
        self.substrate_parameters.append(
            (
                potential.center_angstrom,
                potential.scale_angstrom,
                potential.attraction_strength,
                potential.repulsion_slope,
                potential.repulsion_cutoff_angstrom,
                potential.weight,
            )
        )
        self.substrate_nearest_only.append(potential.nearest_only_repulsion)

    def lower(self) -> RfDiffusionPotentialInput:
        r"""The program as the Model's numeric input."""

        def integer(values: Sequence[int]) -> Tensor:
            return torch.tensor(list(values), dtype=torch.int64, device=self.device)

        def real(values: Sequence[object], shape: tuple[int, ...] | None = None) -> Tensor:
            tensor = torch.tensor(list(values), dtype=torch.float64, device=self.device)
            tensor = project_model_features(
                tensor,
                torch.ones_like(tensor, dtype=torch.bool),
                dtype=self.dtype,
                feature="RFdiffusion potential parameters",
            )
            return tensor if shape is None or tensor.numel() else tensor.reshape(shape)

        return RfDiffusionPotentialInput(  # type: ignore[call-arg]
            point_atom_index=integer(self.point_atom_index),
            point_local_coordinates=(
                torch.stack(tuple(self.point_local_coordinates))
                if self.point_local_coordinates
                else torch.empty((0, 3), dtype=self.dtype, device=self.device)
            ),
            frame_atom_index=self.frame_atom_index,
            coordination_point_index=integer(self.coordination_point_index),
            coordination_ptr=integer(self.coordination_ptr),
            coordination_center=real(self.coordination_center),
            coordination_scale=real(self.coordination_scale),
            coordination_weight=real(self.coordination_weight),
            radius_point_index=integer(self.radius_point_index),
            radius_ptr=integer(self.radius_ptr),
            radius_threshold=real(self.radius_threshold),
            radius_weight=real(self.radius_weight),
            substrate_point_index=integer(self.substrate_point_index),
            substrate_ptr=integer(self.substrate_ptr),
            substrate_parameters=real(self.substrate_parameters, (0, 6)),
            substrate_nearest_only=torch.tensor(self.substrate_nearest_only, dtype=torch.bool, device=self.device),
            batch_size=[],
        )


def _author_positions(structure: Structure) -> dict[str, dict[tuple[int, str], int]]:
    r"""Per instance symbol, the position of every author-numbered component, from the identity columns (never a
    parse of the source symbols)."""
    system = structure.system
    insertion = np.asarray((*structure.symbols()["insertion"], ""), dtype=object)[structure.component_insertion]
    instance = np.repeat(np.arange(len(system.instance_symbol)), np.diff(system.instance_component_ptr))
    symbols = system.text("instance", system.instance_symbol)
    positions: dict[str, dict[tuple[int, str], int]] = {}
    for component in np.flatnonzero(known(structure.component_author_number_state)).tolist():
        owner = int(instance[component])
        key = (int(structure.component_author_number[component]), str(insertion[component] or ""))
        positions.setdefault(str(symbols[owner]), {})[key] = component - int(system.instance_component_ptr[owner])
    return positions


def _resolve_author_numbering(
    structure: Structure | None, author_numbering: Mapping[str, Mapping[tuple[int, str], int]] | None
) -> Mapping[str, Mapping[tuple[int, str], int]]:
    if structure is not None:
        return _author_positions(structure)
    return {} if author_numbering is None else author_numbering


def _existing_bounds(sub: str) -> tuple[int, int]:
    r"""The author-number bounds of one existing contig fragment."""
    bounds = sub[1:].split("-")
    if len(bounds) != 2 or not all(bound.isdigit() for bound in bounds):
        fail("value_invalid", RequestRef("contigs", sub))
    low, high = int(bounds[0]), int(bounds[1])
    if high < low:
        fail("value_invalid", RequestRef("contigs", sub))
    return low, high


def _rebase_existing_span(
    instance_id: str,
    author_start: int,
    author_end: int,
    author_positions: Mapping[str, Mapping[tuple[int, str], int]],
    sub: str,
) -> tuple[int, int]:
    r"""One author-numbered contig fragment as a position span of its instance."""
    chain = author_positions.get(instance_id)
    if chain is None:
        fail("value_invalid", RequestRef("contigs", sub), reason=f"no author numbering for {instance_id!r}")
    coded = {number for number, insertion in chain if insertion}
    selected: list[int] = []
    for number in range(author_start, author_end + 1):
        if number in coded:
            fail("value_invalid", RequestRef("contigs", sub), insertion_coded=number)
        if (number, "") not in chain:
            fail("value_invalid", RequestRef("contigs", sub), absent=number)
        selected.append(chain[(number, "")])
    if selected != list(range(selected[0], selected[0] + len(selected))):
        fail("value_invalid", RequestRef("contigs", sub), reason="not a contiguous residue run")
    return selected[0], selected[-1] + 1


def _generated_segment(sub: str) -> GeneratedSegment:
    r"""One generated contig fragment as a segment."""
    bounds = sub.split("-")
    if len(bounds) > 2 or not all(bound.isdigit() for bound in bounds):
        fail("value_invalid", RequestRef("contigs", sub))
    low, high = int(bounds[0]), int(bounds[-1])
    if high < low:
        fail("value_invalid", RequestRef("contigs", sub))
    return GeneratedSegment("protein", low, high)


def from_contig_string(
    contigs: str,
    *,
    source: Structure | None = None,
    author_numbering: Mapping[str, Mapping[tuple[int, str], int]] | None = None,
) -> RfDiffusionPlacements:
    r"""
    Parse the released RFdiffusion contig grammar into placements and receptor groups.

    Args:
        contigs: The released grammar, e.g. ``"10-20/A50-60/10-20 B1-100/0"``.
        source: The source whose author numbering the existing fragments are addressed by.
        author_numbering: That numbering stated directly, when no source is at hand.

    Returns:
        The placements, every all-existing trailing block a receptor group.
    """

    placement: PlacementSpec = {}
    receptor_groups: set[str] = set()
    positions = _resolve_author_numbering(source, author_numbering)
    blocks = contigs.split()
    for block_index, block in enumerate(blocks):
        group_key = f"chain_{block_index + 1}"
        parts = block.split("/")
        if any(not value for value in parts):
            fail("value_invalid", RequestRef("contigs", block))
        trailing_zero = parts[-1] == "0"
        if trailing_zero:
            parts.pop()
        group: dict[str, PlacementValue] = {}
        fragments: list[tuple[str, int]] = []
        for sub_index, sub in enumerate(parts):
            placement_id = f"{group_key}:{sub_index}"
            if not sub[0].isalpha():
                group[placement_id] = _generated_segment(sub)
                continue
            author_start, author_end = _existing_bounds(sub)
            if not positions:
                fail("value_invalid", RequestRef("contigs", sub), reason="author numbering needed")
            start, stop = _rebase_existing_span(sub[0], author_start, author_end, positions, sub)
            group[placement_id] = ExistingSpan(sub[0], start, stop)
            fragments.append((sub[0], author_start))
        placement[group_key] = group
        if bool(group) and len(fragments) == len(group) and (trailing_zero or block_index == len(blocks) - 1):
            if len({chain for chain, _start in fragments}) != 1:
                fail("value_invalid", RequestRef("contigs", block), reason="one receptor source chain")
            starts = [start for _chain, start in fragments]
            if any(left >= right for left, right in zip(starts, starts[1:])):
                fail("value_invalid", RequestRef("contigs", block), reason="receptor fragments out of order")
            receptor_groups.add(group_key)
    return RfDiffusionPlacements(placement, receptor_groups=frozenset(receptor_groups))


def to_contig_string(
    placement: RfDiffusionPlacements,
    *,
    source: Structure | None = None,
    author_numbering: Mapping[str, Mapping[tuple[int, str], int]] | None = None,
) -> str:
    r"""
    Re-emit RFdiffusion placements with canonical chain breaks.

    Raises:
        InvalidRequestError: ``value_invalid`` when the released syntax cannot represent the placements exactly (a
            multi-character chain, insertion-coded or non-consecutive author numbering, an empty span, or a round trip
            that does not restore them).
    """

    positions = _resolve_author_numbering(source, author_numbering)
    _validate_receptor_groups(placement)
    blocks: list[str] = []
    groups = tuple(placement.placements.items())
    for group_index, (group_key, group) in enumerate(groups):
        subcons: list[str] = []
        receptor = group_key in placement.receptor_groups
        previous: int | None = None
        for value in group.values():
            if isinstance(value, GeneratedSegment):
                subcons.append(f"{value.min_length}-{value.max_length}")
                continue
            start_number, end_number = _span_numbers(value, positions)
            if receptor:
                if previous is not None and start_number <= previous:
                    fail("value_invalid", RequestRef("placements", group_key), reason="fragments out of order")
                previous = start_number
            subcons.append(f"{value.instance_id}{start_number}-{end_number}")
        all_existing = bool(group) and all(isinstance(value, ExistingSpan) for value in group.values())
        if receptor or (group_index < len(groups) - 1 and not all_existing):
            subcons.append("0")
        blocks.append("/".join(subcons))
    contigs = " ".join(blocks)
    restored = from_contig_string(contigs, author_numbering=positions)
    expected = tuple(
        (tuple(group.values()), group_key in placement.receptor_groups)
        for group_key, group in placement.placements.items()
    )
    observed = tuple(
        (tuple(group.values()), group_key in restored.receptor_groups)
        for group_key, group in restored.placements.items()
    )
    if observed != expected:
        fail("value_invalid", RequestRef("placements", contigs), reason="no exact contig round trip")
    return contigs


def _span_numbers(span: ExistingSpan, positions: Mapping[str, Mapping[tuple[int, str], int]]) -> tuple[int, int]:
    r"""The first and last author number of one existing span."""
    if len(span.instance_id) != 1 or not span.instance_id.isalpha():
        fail("value_invalid", RequestRef("placements", span.instance_id), reason="chain not representable")
    chain = positions.get(span.instance_id)
    if chain is None:
        fail("value_invalid", RequestRef("placements", span.instance_id), reason="author numbering needed")
    found: list[tuple[int, str]] = []
    for position in range(span.start, span.stop):
        matches = [key for key, value in chain.items() if value == position]
        if len(matches) != 1:
            fail("value_invalid", RequestRef("placements", str(position)), reason="no unique author numbering")
        found.append(matches[0])
    if not found:
        fail("value_invalid", RequestRef("placements", span.instance_id), reason="empty span")
    if any(insertion for _number, insertion in found):
        fail("value_invalid", RequestRef("placements", span.instance_id), reason="insertion coded")
    numbers = [number for number, _insertion in found]
    if numbers[0] < 0 or numbers != list(range(numbers[0], numbers[0] + len(numbers))):
        fail("value_invalid", RequestRef("placements", span.instance_id), reason="numbering not consecutive")
    return numbers[0], numbers[-1]
