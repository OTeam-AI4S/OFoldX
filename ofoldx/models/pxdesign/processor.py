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
PXDesign input processing, written against the shared preparation contract.

``prepare`` settles the generation problem once: the target spans the placement names, projected from the input
structure, and one generated protein chain of unknown residues built from definitions, assembled into one value. The
model's tensors are featurised from that value over the slot axis. The generated residues' reference backbone and the
conditional template of the target's own coordinates are the family's conventions, not evidence, so they stay here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from ...chemistry.construction import PolymerSequence, Recipe, build_system
from ...chemistry.definitions import ComponentLibrary, get_component
from ...data.assembly import union
from ...data.molecular import ComponentRef, MolecularSystem, MonomerType, ReferenceSource
from ...data.projection import ProjectionMap, project_atoms
from ...data.structure import CoordinateEvidence, Structure, replace_coordinates
from ...data.tokenization import TOKENS, TokenLayout
from ...errors import RequestRef, StructureRef, fail
from ...modules.functional import pairwise_distance
from ...processor import Processor
from ...provenance import Records
from ..conditions import ExistingSpan, GeneratedSegment, Hotspot, PlacementSpec, validate_placement
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
    missing_reference,
    reference_conformers,
)
from ..processing import project_model_features
from .config import PxDesignProcessorConfig
from .model import PxDesignInput

_PROTEIN, _RNA, _DNA = MonomerType.PROTEIN, MonomerType.RNA, MonomerType.DNA

PXDESIGN_LAYOUT = LayoutTable(
    family="pxdesign",
    monomer_types={_PROTEIN: 0, _DNA: 1, _RNA: 2},
    residue_slots=None,
    monomer_slots=None,
    hydrogens="exclude",
    per_atom_components="none",
    exclude_designated_leaving=False,
    centre_atom={_PROTEIN: "CA", _DNA: "C1'", _RNA: "C1'"},
    frame_atoms={},
    representative={},
    representative_default={},
)
#: PXDesign conditions on every modelled heavy atom's element, formal charge and reference conformer.
PXDESIGN_GATING = GatingPolicy(required_states=frozenset({"atomic_number", "formal_charge"}))
#: The generated chain's components: the unknown residue, canonical class ``X``.
GENERATED_COMPONENT = "UNK"
#: The atoms a generated residue is modelled by; the rest of its definition has no slot and no coordinate.
GENERATED_ATOMS = ("N", "CA", "C", "O")
#: The residue whose conformer a generated residue's backbone is referenced by: the family's convention, not a claim
#: about the generated residue, whose own chemistry is unknown. Its ``GENERATED_ATOMS`` are the template.
GENERATED_TEMPLATE = "GLY"
#: The conditional template's distance-bin edges: 63 edges from 2 to 22 angstrom, as upstream.
TEMPLATE_EDGES = (2.0, 22.0, 63)

_BINDER_LABEL = "binder"
#: One target chain of a placement: a source instance symbol and its ordered half-open component intervals.
TargetChain = tuple[str, list[tuple[int, int]]]


def _owner(pointer: NDArray[np.int64]) -> NDArray[np.int64]:
    return np.repeat(np.arange(len(pointer) - 1, dtype=np.int64), np.diff(pointer))


def from_binder(num_binder_residues: int, target: Sequence[TargetChain] = ()) -> PlacementSpec:
    r"""
    The placement of one fixed-length generated protein binder after the named target spans.

    Raises:
        InvalidRequestError: ``value_invalid`` for a non-positive length, a chain without an interval, a repeated
            chain or an interval that is not a half-open ``[start, stop)``.
    """

    if num_binder_residues <= 0:
        fail("value_invalid", RequestRef("num_binder_residues", str(num_binder_residues)))
    placement: PlacementSpec = {}
    for instance_id, crops in target:
        intervals = tuple(crops)
        if not intervals or instance_id in placement:
            fail("value_invalid", RequestRef("target", instance_id))
        group: dict[str, GeneratedSegment | ExistingSpan] = {}
        for start, stop in intervals:
            if not 0 <= start < stop:
                fail("value_invalid", RequestRef(f"target[{instance_id}]", f"({start}, {stop})"), half_open=True)
            placement_id = instance_id if len(intervals) == 1 else f"{instance_id}#{start}_{stop}"
            if placement_id in group:
                fail("value_invalid", RequestRef(f"target[{instance_id}]", f"({start}, {stop})"), repeated=True)
            group[placement_id] = ExistingSpan(instance_id, start, stop)
        placement[instance_id] = group
    label = _BINDER_LABEL
    taken = set(placement) | {placement_id for group in placement.values() for placement_id in group}
    while label in taken:
        label += "_"
    placement[label] = {label: GeneratedSegment("protein", num_binder_residues, num_binder_residues)}
    return placement


def to_binder(placement: PlacementSpec) -> tuple[int, list[TargetChain]]:
    r"""
    The binder length and the target chains a placement declares: its trailing group is one generated protein segment
    of one fixed positive length, every other group holds the spans of one source instance.

    Raises:
        InvalidRequestError: ``value_invalid`` for any other shape.
    """

    validate_placement(placement)
    groups = list(placement.items())
    if not groups:
        fail("value_invalid", RequestRef("placement", "{}"))
    label, values = groups[-1][0], tuple(groups[-1][1].values())
    binder = values[0] if len(values) == 1 else None
    if not isinstance(binder, GeneratedSegment) or binder.polymer_type != "protein":
        fail("value_invalid", RequestRef("placement", label), invariant="one_generated_protein_binder")
    if binder.min_length != binder.max_length or binder.max_length <= 0:
        fail("value_invalid", RequestRef("placement", label), invariant="one_fixed_positive_length")
    targets: list[TargetChain] = []
    for group, placements in groups[:-1]:
        spans = tuple(placements.values())
        instances = {span.instance_id for span in spans if isinstance(span, ExistingSpan)}
        if len(instances) != 1 or len(spans) != len([span for span in spans if isinstance(span, ExistingSpan)]):
            fail("value_invalid", RequestRef("placement", group), invariant="spans_of_one_source_instance")
        crops = [(span.start, span.stop) for span in spans if isinstance(span, ExistingSpan)]
        if any(start >= stop for start, stop in crops):
            fail("value_invalid", RequestRef("placement", group), invariant="half_open_spans")
        targets.append((instances.pop(), crops))
    return binder.max_length, targets


def assemble(values: Sequence[Structure]) -> tuple[Structure, tuple[ProjectionMap, ...]]:
    r"""
    One value holding every instance of ``values``, and per value its rows in it.

    The parts are joined as systems (``data.union``) and their coordinates are restated on the joined value with the
    evidence each atom carries. Their source documents, author identity, occupancy, B-factor and resolution do not
    survive: a value holding instances of several documents is not one deposited entry.
    """

    system, maps = union([value.system for value in values])
    joined = Structure.unobserved(system)
    for value, projection in zip(values, maps, strict=True):
        placed = np.flatnonzero(value.coordinate_mask)
        evidence = value.coordinate_evidence[placed]
        for code in np.unique(evidence).tolist():
            rows = placed[evidence == code]
            atoms = projection.atom[rows]
            scope = np.zeros(len(joined.coordinates), dtype=bool)
            scope[atoms] = True
            joined, _ = replace_coordinates(
                joined, atoms, value.coordinates[rows], scope=scope, evidence=CoordinateEvidence(int(code))
            )
    return joined, maps


@dataclass(frozen=True)
class _Problem:
    r"""
    The settled problem: the value, its generated components, and per source component its component in it (``-1``
    for a component the crop holds no atom of).
    """

    structure: Structure
    generated: NDArray[np.int64]
    source_component: NDArray[np.int64]


def _unique_symbol(stem: str, taken: frozenset[str]) -> str:
    symbol = stem
    while symbol in taken:
        symbol += "_"
    return symbol


def _target_atoms(system: MolecularSystem, targets: Sequence[TargetChain]) -> NDArray[np.bool_]:
    r"""
    The atoms of the components the target spans name.

    Raises:
        InvalidRequestError: ``request_invalid_positions`` for a span outside its instance, or when no span holds an
            atom: a target that contributes nothing conditions nothing.
    """
    keep = np.zeros(len(system.atom_symbol), dtype=bool)
    pointer = system.instance_component_ptr
    for instance_id, crops in targets:
        instance = int(system.instance_index([instance_id])[0])
        if instance < 0:
            fail("request_invalid_positions", RequestRef("placement", instance_id))
        start, stop = int(pointer[instance]), int(pointer[instance + 1])
        for first, last in crops:
            if start + last > stop:
                fail("request_invalid_positions", RequestRef("placement", f"{instance_id}[{first}, {last})"))
            atoms = system.component_atom_ptr[[start + first, start + last]]
            keep[atoms[0] : atoms[1]] = True
    if not keep.any():
        instance_id, ((first, last), *_) = targets[0]
        fail("request_invalid_positions", RequestRef("placement", f"{instance_id}[{first}, {last})"), reason="no_atoms")
    return keep


def _problem(
    structure: Structure | None,
    targets: Sequence[TargetChain],
    length: int,
    library: ComponentLibrary | None,
) -> _Problem:
    r"""
    The generation problem: the target spans projected out of ``structure`` and one generated chain of ``length``
    unknown residues, assembled into one value (the target is settled here, once).
    """

    parts: list[Structure] = []
    cropped: ProjectionMap | None = None
    taken: frozenset[str] = frozenset()
    if targets:
        if structure is None:
            fail("request_invalid_positions", RequestRef("placement", targets[0][0]), reason="no_structure")
        target, cropped = project_atoms(structure, _target_atoms(structure.system, targets))
        parts.append(target)
        symbols = structure.system.symbols()
        taken = frozenset(symbols["instance"]) | frozenset(symbols["entity"])
    symbol = _unique_symbol("pxdesign_binder", taken)
    binder, _, _ = build_system(
        Recipe({symbol: PolymerSequence("polypeptide(L)", (GENERATED_COMPONENT,) * length)}, {symbol: symbol}), library
    )
    parts.append(Structure.unobserved(binder))
    value, maps = assemble(parts)
    source_component = (
        np.zeros(0, dtype=np.int64)
        if structure is None
        else np.full(len(structure.system.component_symbol), -1, dtype=np.int64)
    )
    if cropped is not None:
        row = cropped.component
        # an atom projection keeps every component of a kept instance, so a cropped-away span is a component
        # with no atom: it holds no token and no slot, and the problem does not state it
        held = np.append(np.diff(parts[0].system.component_atom_ptr) > 0, False)
        kept = held[row]  # a component the crop keeps no instance of (-1) reads the last, False
        source_component[kept] = maps[0].component[row[kept]]
    return _Problem(value, maps[-1].component, source_component)


def _modelled_atoms(system: MolecularSystem, generated: NDArray[np.int64]) -> NDArray[np.bool_]:
    r"""Every atom but those of a generated residue outside ``GENERATED_ATOMS``: a generated side chain is unknown."""
    component = _owner(system.component_atom_ptr)
    names = system.text("atom", system.atom_symbol)
    return ~(np.isin(component, generated) & ~np.isin(names, GENERATED_ATOMS))


class PxDesignProcessor(Processor):
    r"""
    Prepare one target and placement for PXDesign.

    Args:
        config: Input-processing configuration. Defaults to ``PxDesignProcessorConfig()``.
        seed: Optional override for the configured reference-conformer embedding seed.
        device: Device for produced tensors.
        dtype: Floating-point dtype for produced tensors.
    """

    config_class = PxDesignProcessorConfig
    config: PxDesignProcessorConfig

    def __init__(
        self,
        config: PxDesignProcessorConfig | None = None,
        *,
        seed: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.config = self.config_class() if config is None else self.config_class(config)
        if seed is not None:
            self.config.seed = seed
        super().__init__(device=device, dtype=dtype)

    @property
    def seed(self) -> int:
        r"""Seed used to embed missing reference conformers."""
        return self.config.seed

    @seed.setter
    def seed(self, seed: int) -> None:
        self.config.seed = seed

    def prepare_input(
        self,
        structure: Structure | None = None,
        *,
        placement: PlacementSpec,
        conditions: Sequence[Hotspot] = (),
        library: ComponentLibrary | None = None,
    ) -> PreparedInput[PxDesignInput]:
        r"""
        Prepare one generation problem: the placement's target spans and its generated chain, the layout over the
        modelled atoms, gating, reference conformers and the PXDesign tensors. The scope is the generated components
        in the target's own frame.

        Args:
            structure: The target the placement's spans are taken from; ``None`` generates without a target.
            placement: Target spans followed by one fixed-length generated protein chain.
            conditions: Target components labelled as interaction hotspots.
            library: The definitions the generated residues and their backbone template are read from (default the
                configured artifact).
        """

        length, targets = to_binder(placement)
        problem = _problem(structure, targets, length, library)
        target = problem.structure
        system = target.system
        layout = token_layout(system, PXDESIGN_LAYOUT, atom_mask=_modelled_atoms(system, problem.generated))
        slots = slot_map(system, layout, PXDESIGN_LAYOUT)
        accepted = gate(target, slots, PXDESIGN_GATING)
        references, sources, embedded = reference_conformers(
            target, slots, embed=missing_reference(system), seed=self.seed
        )
        references = _generated_backbone(system, slots, problem.generated, references, library)
        tokens, losses = model_tokens(system, layout)
        generated = np.isin(layout.token_component, problem.generated)
        hotspots = _hotspots(structure, problem, conditions)
        features = _features(
            target,
            slots,
            tokens,
            references,
            generated,
            np.isin(layout.token_component, hotspots),
            device=self.device,
            dtype=self.dtype,
        )
        conformers = np.zeros(len(system.component_symbol), dtype=np.uint8)
        bound = slots.atom >= 0
        conformers[_owner(system.component_atom_ptr)[slots.atom[bound]]] = sources[bound]
        records = Records.concat(
            (
                component_records("encoding_loss", losses.component),
                accepted,
                embedded,
                # the generated components' conformers are GLY's definition, not their own reference pool
                component_records("design_reference_substituted", problem.generated),
            )
        )
        return PreparedInput(
            features,
            PreparationContext(
                target,
                slots,
                RegenerationScope.components(system, problem.generated),
                PreparationRecord(records, conformers),
                spec=self.preparation_spec,
            ),
        )


def _generated_backbone(
    system: MolecularSystem,
    slots: SlotMap,
    generated: NDArray[np.int64],
    references: NDArray[np.float64],
    library: ComponentLibrary | None,
) -> NDArray[np.float64]:
    r"""
    The reference conformers with every generated residue's slots on ``GENERATED_TEMPLATE``'s backbone.

    Raises:
        InsufficientInformationError: ``prepare_reference_missing`` when the template residue's definition carries no
            usable conformer, so the family's convention has no coordinates to state.
    """

    definition = get_component(GENERATED_TEMPLATE, library=library)
    if definition.reference_source not in (ReferenceSource.IDEAL, ReferenceSource.MODEL):
        fail("prepare_reference_missing", RequestRef("component_id", GENERATED_TEMPLATE))
    rows = [definition.atom_names.index(name) for name in GENERATED_ATOMS]
    template = dict(zip(GENERATED_ATOMS, definition.reference[rows].astype(np.float64), strict=True))
    atom = np.maximum(slots.atom, 0)
    component = _owner(system.component_atom_ptr)[atom]
    names = system.text("atom", system.atom_symbol[atom])
    positions = references.copy()
    for slot in np.flatnonzero((slots.atom >= 0) & np.isin(component, generated)).tolist():
        positions[slot] = template[str(names[slot])]
    return positions


def _hotspots(structure: Structure | None, problem: _Problem, conditions: Sequence[Hotspot]) -> NDArray[np.int64]:
    r"""
    The problem's components the conditions label as hotspots; a condition names a component of ``structure``.

    Raises:
        InvalidRequestError: ``request_invalid_positions`` for an endpoint that is not a component reference of the
            input structure, or for a component the placement's spans left out.
    """

    references = [residue for condition in conditions for residue in condition.residues]
    if not references:
        return np.zeros(0, dtype=np.int64)
    if structure is None or any(not isinstance(residue, ComponentRef) for residue in references):
        fail("request_invalid_positions", RequestRef("conditions", str(references[0])))
    source = structure.system.component_index(references)  # type: ignore[arg-type]
    if (source < 0).any():
        fail("request_invalid_positions", RequestRef("conditions", str(references[int(np.argmin(source))])))
    found = problem.source_component[source]
    if (found < 0).any():
        reference = str(references[int(np.argmin(found))])
        fail("request_invalid_positions", RequestRef("conditions", reference), cropped=True)
    return found


def _features(
    structure: Structure,
    slots: SlotMap,
    tokens: NDArray[np.int64],
    references: NDArray[np.float64],
    generated: NDArray[np.bool_],
    hotspot: NDArray[np.bool_],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> PxDesignInput:
    r"""The PXDesign tensors of one prepared problem (upstream featurisation over the slot axis)."""
    system = structure.system
    s = system.arrays()
    layout = slots.layout
    atom = slots.atom
    num_atoms, num_tokens = len(atom), len(layout.token_component)
    atom_component = _owner(s["component_atom_ptr"])
    token_instance = _owner(s["instance_component_ptr"])[layout.token_component]
    entity = s["instance_entity"]
    order = np.argsort(entity, kind="stable")
    copy = np.empty(len(entity), dtype=np.int64)
    copy[order] = np.arange(len(entity)) - np.searchsorted(entity[order], entity[order])
    reference_space = atom_component[atom]
    bins, pair_mask = _template(structure, layout)
    tensors: dict[str, Tensor] = {
        "residue_index": torch.from_numpy(layout.token_component - s["instance_component_ptr"][token_instance]),
        "chain_index": torch.from_numpy(token_instance),
        "entity_index": torch.from_numpy(entity[token_instance].astype(np.int64)),
        "symmetry_index": torch.from_numpy(copy[token_instance]),
        "token_index": torch.arange(num_tokens),
        "atom_features": _atom_features(system, atom),
        "atom_to_token": torch.from_numpy(slots.token),
        "atom_mask": torch.ones(num_atoms, dtype=torch.bool),
        "reference_positions": torch.from_numpy(_centred(references, reference_space)),
        "reference_space_id": torch.from_numpy(reference_space),
        "reference_mask": torch.ones(num_atoms, dtype=torch.bool),
        "reference_charges": torch.from_numpy(s["formal_charge"][atom].astype(np.float64)).reshape(num_atoms, 1),
        "residue_type_features": torch.nn.functional.one_hot(torch.from_numpy(tokens), len(TOKENS)).double(),
        "design_features": torch.from_numpy(generated).double().reshape(num_tokens, 1),
        "plddt": torch.zeros(num_tokens, 1, dtype=torch.float64),
        "hotspot": torch.from_numpy(hotspot).double().reshape(num_tokens, 1),
        "template_distance_bins": bins,
        "template_pair_mask": pair_mask,
    }
    placed = {
        name: (
            project_model_features(
                value, torch.ones_like(value, dtype=torch.bool), dtype=dtype, feature=f"PXDesign prepared {name}"
            ).to(device=device)
            if value.is_floating_point()
            else value.to(device=device)
        )
        for name, value in tensors.items()
    }
    return PxDesignInput(  # type: ignore[call-arg]
        **{name: value.unsqueeze(0) for name, value in placed.items()},
        batch_size=[],
    )


def _centred(references: NDArray[np.float64], space: NDArray[np.int64]) -> NDArray[np.float64]:
    r"""Each reference space's conformer on its own centroid, as upstream centres a space before encoding it."""
    positions = references.copy()
    for value in np.unique(space).tolist():
        rows = space == value
        positions[rows] -= positions[rows].mean(axis=0)
    return positions


def _atom_features(system: MolecularSystem, atom: NDArray[np.int64]) -> Tensor:
    r"""Per slot: the reference-conformer flag, the element one-hot (128) and four atom-name characters (4 x 64)."""
    count = len(atom)
    numbers = system.atomic_number[atom].astype(np.int64)
    element = torch.zeros(count, 128, dtype=torch.float64)
    element[torch.arange(count), torch.from_numpy(numbers - 1)] = 1
    names = system.text("atom", system.atom_symbol[atom]).tolist()
    characters = torch.zeros(count, 4, 64, dtype=torch.float64)
    for row, name in enumerate(names):
        if len(name) > 4 or any(not 32 <= ord(character) < 96 for character in name):
            fail("model_capacity", StructureRef(atom=int(atom[row]), item=name), limit="four printable characters")
        for column, character in enumerate(name.ljust(4)):
            characters[row, column, ord(character) - 32] = 1
    return torch.cat((torch.ones(count, 1, dtype=torch.float64), element, characters.flatten(-2)), dim=-1)


def _template(structure: Structure, layout: TokenLayout) -> tuple[Tensor, Tensor]:
    r"""
    The conditional template: the distance bins of the target's own centre atoms (the family's self-template; a
    generated residue has no coordinate, so its pairs are masked).
    """

    centre = layout.token_centre_atom
    count = len(centre)
    present = (centre >= 0) & structure.coordinate_mask[np.maximum(centre, 0)]
    coordinates = np.zeros((count, 3), dtype=np.float64)
    coordinates[present] = structure.coordinates[centre[present]]
    bins = torch.zeros(count, count, dtype=torch.int64)
    rows = torch.from_numpy(np.flatnonzero(present))
    if rows.numel():
        points = torch.from_numpy(coordinates).index_select(0, rows)
        edges = torch.linspace(TEMPLATE_EDGES[0], TEMPLATE_EDGES[1], int(TEMPLATE_EDGES[2]), dtype=points.dtype)
        values = torch.sum(pairwise_distance(points, points).unsqueeze(-1) > edges, dim=-1)
        first, second = torch.meshgrid(rows, rows, indexing="ij")
        bins[first, second] = values
    return bins, torch.from_numpy(present[:, None] & present[None, :])
