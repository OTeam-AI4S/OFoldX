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
OInvFold input processing, written against the shared preparation contract.

The polymer layout addresses, per residue, the atoms the released graph reads in its order: N, CA, C, O for protein;
seven backbone atoms and the glycosidic nitrogen (N9, else N1) for DNA and RNA. Rows are the residues of one polymer
instance whose atoms are all observed; the model's tokens are the canonical classes (MSE reads ``M`` with its
``encoding_loss`` record), each read as the released output column the modality gives it. Sequence design regenerates
no coordinate: the scope is empty and the design producer builds the final target. The ligand graph reads
coordinates only.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from typing import ClassVar, cast

import numpy as np
import torch
from chanfig import Config
from numpy.typing import NDArray
from torch import Tensor
from torch.nn import functional as F

from ...data.crop import Cropper
from ...data.molecular import EntityType, MolecularSystem, MonomerType
from ...data.structure import Structure
from ...data.tokenization import CLASS_MONOMER, TOKEN_INDEX, TOKENS, TokenLayout, select_tokens
from ...errors import RequestRef, fail
from ...model import Model
from ...modules.functional import radial_basis
from ...processor import Processor
from ...provenance import Records, known
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
from ..processing import model_frame_rotation, project_model_coordinates
from ..targets import SequenceTarget, prepare_sequence_target
from .config import OInvFoldPolymerConfig, OInvFoldPolymerProcessorConfig
from .model import NUM_OINVFOLD_GLOBAL_NODES, OInvFoldGraph, OInvFoldLigandInput, OInvFoldPolymerInput

# The released atom order per modality; a nucleic acid's glycosidic nitrogen is read from the slots N9 then N1.
_SLOTS: dict[str, tuple[str, ...]] = {
    "protein": ("N", "CA", "C", "O"),
    "dna": ("O5'", "C5'", "C4'", "P", "C3'", "O3'", "N9", "N1"),
    "rna": ("P", "C4'", "N9", "N1", "O5'", "C3'", "O3'", "C5'"),
}
#: Per modality, the classes the design chooses among.
OINVFOLD_DESIGN_CLASSES: dict[str, tuple[int, ...]] = {
    "protein": tuple(TOKEN_INDEX[letter] for letter in "ACDEFGHIKLMNPQRSTVWY"),
    "dna": tuple(TOKEN_INDEX[f"DNA_{letter}"] for letter in "ACGT"),
    "rna": tuple(TOKEN_INDEX[f"RNA_{letter}"] for letter in "ACGU"),
}


def _model_classes(modality: str) -> NDArray[np.int64]:
    r"""
    Per canonical class, the class of the released output column the modality reads it as (the released preparation's
    rule, so a hybrid chain or an inosine feeds and scores the column it does upstream): either nucleic type's A, C and
    G read the modality's, inosine reads A, T and U read the modality's pyrimidine; every other class is itself.
    """

    classes = np.arange(len(TOKENS), dtype=np.int64)
    if modality == "protein":
        return classes
    kind, pyrimidine = ("DNA", "T") if modality == "dna" else ("RNA", "U")
    for source in ("RNA", "DNA"):
        for letter, read in (("A", "A"), ("C", "C"), ("G", "G"), ("I", "A")):
            classes[TOKEN_INDEX[f"{source}_{letter}"]] = TOKEN_INDEX[f"{kind}_{read}"]
    classes[[TOKEN_INDEX["RNA_U"], TOKEN_INDEX["DNA_T"]]] = TOKEN_INDEX[f"{kind}_{pyrimidine}"]
    return classes


def _layout(modality: str) -> LayoutTable:
    kinds = (MonomerType.PROTEIN,) if modality == "protein" else (MonomerType.DNA, MonomerType.RNA)
    names = _SLOTS[modality]
    return LayoutTable(
        family="oinvfold",
        monomer_types=dict.fromkeys(kinds, 0),
        residue_slots=None,
        monomer_slots={kind: names for kind in kinds},
        hydrogens="exclude",
        per_atom_components="none",
        exclude_designated_leaving=False,
        centre_atom={kind: names[1] for kind in kinds},
        frame_atoms={},
        representative={},
        representative_default={},
    )


#: Per modality, the polymer layout (a hybrid chain's residues all read the modality's atom order).
OINVFOLD_LAYOUTS: dict[str, LayoutTable] = {modality: _layout(modality) for modality in _SLOTS}
_MODEL_CLASSES = {modality: _model_classes(modality) for modality in _SLOTS}
#: The polymer graph reads coordinates and classes of the slot atoms; an excess valence elsewhere is not its input.
OINVFOLD_GATING = GatingPolicy(refuse_valence_excess=False)
#: The ligand graph reads coordinates only (as the packer's context).
OINVFOLD_LIGAND_GATING = GatingPolicy(
    refuse_undetermined_in_slots=False,
    refuse_unknown_kind_in_slots=False,
    refuse_incomplete_topology_in_slots=False,
    refuse_valence_excess=False,
)
#: Every observed atom of one ligand instance is its own token.
OINVFOLD_LIGAND_LAYOUT = LayoutTable(
    family="oinvfold_ligand",
    monomer_types={},
    residue_slots=None,
    monomer_slots=None,
    hydrogens="include",
    per_atom_components="all",
    exclude_designated_leaving=False,
    centre_atom={},
    frame_atoms={},
    representative={},
    representative_default={},
    exclude_water=True,
)


def _geometry_dtype(dtype: torch.dtype) -> torch.dtype:
    r"""Compute geometry in Float32 for low-precision models and preserve Float64 inputs."""
    return torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype


def _geometry_coordinates(points: Tensor, observed: Tensor, *, dtype: torch.dtype, feature: str) -> Tensor:
    r"""Check model representability before computing geometry from the unquantized source."""
    projected = project_model_coordinates(points, observed, dtype=dtype, feature=feature)
    if dtype in (torch.float16, torch.bfloat16):
        return project_model_coordinates(points, observed, dtype=torch.float32, feature=feature)
    return projected


def _owner(pointer: NDArray[np.int64]) -> NDArray[np.int64]:
    return np.repeat(np.arange(len(pointer) - 1, dtype=np.int64), np.diff(pointer))


def _observed(structure: Structure, filter_by_occupancy: bool) -> NDArray[np.bool_]:
    observed = structure.coordinate_mask.copy()
    if filter_by_occupancy:
        observed &= ~(known(structure.occupancy_state) & (structure.occupancy <= 0))
    return observed


class OInvFoldPolymerProcessor(Processor):
    r"""
    Prepare one protein, DNA or RNA chain for the released OInvFold graph.

    Args:
        config: Polymer modality stored with the paired repository.
        device: Device for produced tensors.
        dtype: Floating-point dtype for produced tensors.
    """

    config_class: ClassVar[type[Config] | None] = OInvFoldPolymerProcessorConfig
    config: OInvFoldPolymerProcessorConfig

    def __init__(
        self,
        config: OInvFoldPolymerProcessorConfig | None = None,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.config = OInvFoldPolymerProcessorConfig() if config is None else config
        super().__init__(device=device, dtype=dtype)

    @property
    def layout(self) -> LayoutTable:
        r"""The layout of this processor's modality."""
        return OINVFOLD_LAYOUTS[self.config.modality]

    def validate_model_input_layout(self, model: Model) -> None:
        r"""Require the model and processor to interpret shared output columns in the same modality.

        The selected modality determines coordinate order and nucleic-class interpretation. It does not restrict
        the physical RNA/DNA chemistry of an input chain.
        """
        modality = cast(OInvFoldPolymerConfig, model.config).modality
        if self.config.modality != modality:
            fail("value_invalid", RequestRef("modality", modality), processor=self.config.modality)

    def prepare_input(
        self,
        structure: Structure,
        *,
        design: PositionSelection = PositionSelection(),
        filter_by_occupancy: bool = True,
        cropper: Cropper | None = None,
        generator: np.random.Generator | None = None,
        _linker_coordinates: Mapping[int, NDArray[np.float64]] | None = None,
    ) -> PreparedInput[OInvFoldPolymerInput]:
        r"""
        Prepare the polymer instance holding the designed positions.

        Args:
            structure: The target; a residue of the modality in a polymer instance is a row when every atom the graph
                reads is observed.
            design: The positions to design among the rows (default every row); they lie in one instance, whose rows
                form the graph.
            filter_by_occupancy: Whether atoms with a KNOWN occupancy at or below zero are treated as unobserved.
            cropper: Optional training crop over the rows, keeping every designed row unless all rows are designed;
                needs ``generator``.
            generator: The numpy generator of the crop.

        Raises:
            InvalidRequestError: ``request_invalid_positions`` when the positions lie in several instances or none.
            UnsupportedError: ``model_capacity`` when fewer than two rows remain.
        """

        if cropper is not None and generator is None:
            fail("request_invalid_generator", RequestRef("generator", "None"))
        modality = self.config.modality
        system = structure.system
        table = self.layout
        observed = _observed(structure, filter_by_occupancy)
        instance = _owner(system.instance_component_ptr)
        entity = system.instance_entity
        polymer = known(system.entity_type_state[entity]) & (system.entity_type[entity] == EntityType.POLYMER)
        layout = token_layout(system, table, atom_mask=_residue_atoms(system, table))
        atoms = _row_atoms(slot_map(system, layout, table), modality)
        present = (atoms >= 0) & observed[np.maximum(atoms, 0)]
        virtual = np.zeros(len(atoms), dtype=bool)
        if _linker_coordinates and modality != "protein":
            linker = _SLOTS[modality].index("N9")
            virtual = np.array([int(c) in _linker_coordinates for c in layout.token_component])
            virtual &= atoms[:, linker] < 0
            present[virtual, linker] = True
        valid = present.all(axis=1)
        valid &= polymer[instance[layout.token_component]]
        allowed = np.zeros(len(system.component_symbol), dtype=bool)
        allowed[layout.token_component[valid]] = True
        positions = resolve_positions(structure, design, allowed=allowed)
        chains = np.unique(instance[positions])
        if len(chains) != 1:
            fail("request_invalid_positions", RequestRef("design", "positions of one instance"), instances=len(chains))
        keep = valid & (instance[layout.token_component] == chains[0])
        virtual = virtual[keep]
        layout, atoms = select_tokens(layout, keep), _two_rows(atoms[keep])
        if virtual.any():
            assert _linker_coordinates is not None
            values = structure.coordinates[np.maximum(atoms, 0)].copy()
            linker = _SLOTS[modality].index("N9")
            for row in np.flatnonzero(virtual):
                values[row, linker] = _linker_coordinates[int(layout.token_component[row])]
            coordinates = _geometry_coordinates(
                torch.from_numpy(values),
                torch.ones(values.shape[:-1], dtype=torch.bool),
                dtype=self.dtype,
                feature="OInvFold backbone",
            ).to(device=self.device)
        else:
            coordinates = self._coordinates(structure, atoms)
        direction = _sequence_direction(coordinates)
        designed = np.isin(layout.token_component, positions)
        if cropper is not None:
            keep = cropper(
                structure,
                layout,
                required=None if designed.all() else designed,
                generator=cast(np.random.Generator, generator),
            )
            layout, atoms, designed = select_tokens(layout, keep), _two_rows(atoms[keep]), designed[keep]
            coordinates = coordinates[torch.as_tensor(keep, device=coordinates.device)]
        slots = _consumed_slots(system, layout, atoms)
        accepted = gate(structure, slots, OINVFOLD_GATING)
        tokens, losses = model_tokens(system, layout)
        rows = len(layout.token_component)
        component_index = torch.as_tensor(layout.token_component, device=self.device)
        graph = _lower_graph(
            coordinates,
            position_index=component_index,
            direction=direction,
            num_neighbors=32,
            node_feature_size=133,
            edge_feature_size=310,
            dtype=self.dtype,
        )
        features = OInvFoldPolymerInput(  # type: ignore[call-arg]
            graph=graph,
            component_index=component_index,
            design_mask=torch.as_tensor(designed, device=self.device),
            chain_index=torch.zeros(rows, dtype=torch.long, device=self.device),
            sequence_tokens=torch.as_tensor(_MODEL_CLASSES[modality][tokens], device=self.device),
            batch_size=[],
        )
        records = Records.concat((component_records("encoding_loss", losses.component), accepted))
        record = PreparationRecord(records, np.zeros(len(system.component_symbol), dtype=np.uint8))
        return PreparedInput(
            features,
            PreparationContext(structure, slots, RegenerationScope.none(system), record, spec=self.preparation_spec),
        )

    def prepare_target(self, target: Structure, *, prepared: PreparedInput[OInvFoldPolymerInput]) -> SequenceTarget:
        r"""Align polymer truth to the prepared rows and declared output-column convention.

        Either physical nucleic-acid type uses the selected DNA/RNA class
        interpretation. Missing and unknown truth stays masked; this operation
        does not replace source sequence or geometric features.
        """
        self._validate_prepared(prepared)
        result = prepare_sequence_target(
            prepared.context.structure,
            target,
            prepared.features.component_index,
            class_map=_model_classes(self.config.modality),
        )
        kind = torch.as_tensor(CLASS_MONOMER, device=result.tokens.device)[result.tokens]
        allowed = (
            kind == MonomerType.PROTEIN
            if self.config.modality == "protein"
            else ((kind == MonomerType.DNA) | (kind == MonomerType.RNA))
        )
        if bool((result.mask & ~allowed).any()):
            fail("model_capacity", RequestRef("target", f"{self.config.modality} sequence supervision is required"))
        return result

    def _coordinates(self, structure: Structure, atoms: NDArray[np.int64]) -> Tensor:
        points = torch.from_numpy(structure.coordinates[atoms])
        observed = torch.ones_like(points[..., 0], dtype=torch.bool)
        return _geometry_coordinates(points, observed, dtype=self.dtype, feature="OInvFold backbone").to(
            device=self.device
        )


def _residue_atoms(system: MolecularSystem, table: LayoutTable) -> NDArray[np.bool_]:
    r"""The atoms of components holding a centre atom: a component without one is no row."""
    component = _owner(system.component_atom_ptr)
    holds = np.zeros(len(system.component_symbol), dtype=bool)
    holds[component[np.isin(system.text("atom", system.atom_symbol), list(table.centre_atom.values()))]] = True
    return holds[component]


def _row_atoms(slots: SlotMap, modality: str) -> NDArray[np.int64]:
    r"""Per row, the atoms the graph reads in its order (``-1`` where absent): N9, else N1, for the glycosidic N."""
    names = _SLOTS[modality]
    atoms = slots.atom.reshape(-1, len(names))
    if "N9" not in names:
        return atoms
    at = names.index("N9")
    glycosidic = np.where(atoms[:, at] >= 0, atoms[:, at], atoms[:, at + 1])
    return np.concatenate((atoms[:, :at], glycosidic[:, None], atoms[:, at + 2 :]), axis=1)


def _two_rows(atoms: NDArray[np.int64]) -> NDArray[np.int64]:
    if len(atoms) < 2:
        fail("model_capacity", RequestRef("OInvFold polymer graph", "fewer than two rows"))
    return atoms


def _consumed_slots(system: MolecularSystem, layout: TokenLayout, atoms: NDArray[np.int64]) -> SlotMap:
    r"""The slot map of exactly the atoms the graph reads."""
    addressed = np.zeros(len(system.atom_symbol), dtype=bool)
    addressed[atoms[atoms >= 0]] = True
    token = np.repeat(np.arange(len(atoms), dtype=np.int64), atoms.shape[1])
    return SlotMap(atoms.reshape(-1), token, layout, np.flatnonzero(~addressed).astype(np.int64))


class OInvFoldLigandProcessor(Processor):
    r"""
    Lower ordered coordinate-bearing atoms to the released ligand graph.
    """

    def __call__(
        self,
        coordinates: Tensor | NDArray,
        *,
        slot_ids: Sequence[str],
        context: Structure | None = None,
        slot_logit_bias: Tensor | NDArray | None = None,
    ) -> OInvFoldLigandInput:
        r"""
        Prepare an atom-element assignment graph.

        Args:
            coordinates: Provisional atom coordinates in slot order.
            slot_ids: Stable caller-owned identifiers in coordinate order.
            context: Optional canonical context retained with decoded candidates.
            slot_logit_bias: Optional additive element-logit bias aligned to slots.

        Returns:
            One atom graph retaining the ordered slot alignment.
        """

        slot_ids = tuple(slot_ids)
        if len(slot_ids) < 3:
            fail("model_capacity", RequestRef("OInvFold ligand graph", "fewer than three atoms"))
        coordinates = torch.as_tensor(coordinates, device=self.device)
        if coordinates.shape[0] != len(slot_ids):
            fail("value_invalid", RequestRef("slot_ids", str(len(slot_ids))), coordinates=coordinates.shape[0])
        available = torch.ones_like(coordinates[..., 0], dtype=torch.bool)
        geometry = project_model_coordinates(
            coordinates, available, dtype=_geometry_dtype(self.dtype), feature="OInvFold ligand slots"
        )
        coordinates = project_model_coordinates(
            coordinates, available, dtype=self.dtype, feature="OInvFold ligand slots"
        )
        precision_context = (
            torch.autocast(device_type=geometry.device.type, enabled=False)
            if self.dtype in (torch.float16, torch.bfloat16)
            else nullcontext()
        )
        with precision_context:
            graph = _lower_graph(
                _ligand_local_slots(geometry),
                num_neighbors=48,
                node_feature_size=114,
                edge_feature_size=272,
                dtype=self.dtype,
            )
        return OInvFoldLigandInput(  # type: ignore[call-arg]
            graph=graph,
            context=context,
            slot_ids=slot_ids,
            coordinates=coordinates,
            slot_logit_bias=(
                None
                if slot_logit_bias is None
                else torch.as_tensor(slot_logit_bias, dtype=self.dtype, device=self.device)
            ),
            batch_size=[],
        )

    def prepare_input(
        self,
        structure: Structure,
        *,
        instance: str | None = None,
        filter_by_occupancy: bool = True,
    ) -> PreparedInput[OInvFoldLigandInput]:
        r"""
        Prepare the observed atoms of one ligand instance; slot ids read ``"instance:component:atom"``.

        Args:
            structure: The context holding the ligand.
            instance: The ligand instance; may be omitted when exactly one non-polymer instance exists.
            filter_by_occupancy: Whether atoms with a KNOWN occupancy at or below zero are treated as unobserved.
        """

        system = structure.system
        if instance is None:
            entity = system.instance_entity
            ligand = known(system.entity_type_state[entity]) & (system.entity_type[entity] == EntityType.NON_POLYMER)
            if ligand.sum() != 1:
                fail("value_invalid", RequestRef("instance", "None"), non_polymer_instances=int(ligand.sum()))
            chosen = int(np.flatnonzero(ligand)[0])
        else:
            chosen = int(system.instance_index([instance])[0])
            if chosen < 0:
                fail("value_invalid", RequestRef("instance", instance))
        mine = _owner(system.instance_component_ptr)[_owner(system.component_atom_ptr)] == chosen
        layout = token_layout(
            system, OINVFOLD_LIGAND_LAYOUT, atom_mask=mine & _observed(structure, filter_by_occupancy)
        )
        slots = slot_map(system, layout, OINVFOLD_LIGAND_LAYOUT)
        accepted = gate(structure, slots, OINVFOLD_LIGAND_GATING)
        refs = system.component_refs(layout.token_component)
        names = system.text("atom", system.atom_symbol[slots.atom])
        features = self(
            structure.coordinates[slots.atom],
            slot_ids=tuple(f"{ref.instance}:{ref.component}:{name}" for ref, name in zip(refs, names)),
            context=structure,
        )
        record = PreparationRecord(accepted, np.zeros(len(system.component_symbol), dtype=np.uint8))
        return PreparedInput(
            features,
            PreparationContext(structure, slots, RegenerationScope.none(system), record, spec=self.preparation_spec),
        )


def _ligand_local_slots(coordinates: Tensor) -> Tensor:
    distances = torch.cdist(coordinates, coordinates)
    distances.fill_diagonal_(torch.inf)
    neighbors = torch.topk(distances, k=2, largest=False).indices
    slots = coordinates.new_zeros((coordinates.shape[0], 6, 3))
    slots[:, 0] = coordinates[neighbors[:, 0]]
    slots[:, 1] = coordinates
    slots[:, 2] = coordinates[neighbors[:, 1]]
    return slots


def _lower_graph(
    coordinates: Tensor,
    *,
    position_index: Tensor | None = None,
    direction: int | None = None,
    num_neighbors: int,
    node_feature_size: int,
    edge_feature_size: int,
    dtype: torch.dtype,
) -> OInvFoldGraph:
    context = (
        torch.autocast(device_type=coordinates.device.type, enabled=False)
        if dtype in (torch.float16, torch.bfloat16)
        else nullcontext()
    )
    with context:
        return _lower_graph_unautocast(
            coordinates,
            position_index=position_index,
            direction=direction,
            num_neighbors=num_neighbors,
            node_feature_size=node_feature_size,
            edge_feature_size=edge_feature_size,
            dtype=dtype,
        )


def _lower_graph_unautocast(
    coordinates: Tensor,
    *,
    position_index: Tensor | None = None,
    direction: int | None = None,
    num_neighbors: int,
    node_feature_size: int,
    edge_feature_size: int,
    dtype: torch.dtype,
) -> OInvFoldGraph:
    num_nodes = coordinates.shape[0]
    if position_index is None:
        position_index = torch.arange(num_nodes, device=coordinates.device)
    centers = coordinates[:, 1]
    distances = torch.cdist(centers, centers)
    distances.fill_diagonal_(torch.inf)
    neighbors = torch.topk(
        distances,
        k=min(num_neighbors, num_nodes - 1),
        largest=False,
    ).indices
    source = torch.arange(num_nodes, device=coordinates.device).unsqueeze(1).expand_as(neighbors)
    edge_index = torch.stack((source.reshape(-1), neighbors.reshape(-1)))
    order = torch.argsort(edge_index[1] * (int(edge_index[0].max()) + 1) + edge_index[0])
    edge_index = edge_index[:, order]

    node_rotations, node_translations = _reference_frames(
        coordinates[:, 0],
        coordinates[:, 1],
        coordinates[:, 2],
    )
    source, destination = edge_index
    edge_rotations, edge_translations = _relative_transforms(
        node_rotations[source],
        node_translations[source],
        node_rotations[destination],
        node_translations[destination],
    )
    node_features, edge_features = _interaction_features(
        coordinates,
        node_rotations,
        node_translations,
        edge_rotations,
        edge_translations,
        edge_index,
        position_index,
        direction,
    )
    node_features = F.pad(node_features, (0, node_feature_size - node_features.shape[-1]))
    edge_features = F.pad(edge_features, (0, edge_feature_size - edge_features.shape[-1]))

    global_rotations, global_translations = _global_frames(
        node_translations,
        NUM_OINVFOLD_GLOBAL_NODES,
    )
    global_nodes = torch.arange(
        num_nodes,
        num_nodes + NUM_OINVFOLD_GLOBAL_NODES,
        device=coordinates.device,
    )
    global_source = global_nodes.repeat_interleave(num_nodes)
    global_destination = torch.arange(num_nodes, device=coordinates.device).repeat(NUM_OINVFOLD_GLOBAL_NODES)
    global_edge_index = torch.stack((global_destination, global_source))
    global_edge_index = torch.cat((global_edge_index, global_edge_index.flip(0)), dim=1)

    all_rotations = torch.cat((node_rotations, global_rotations), dim=0)
    all_translations = torch.cat((node_translations, global_translations), dim=0)
    source, destination = global_edge_index
    global_edge_rotations, global_edge_translations = _relative_transforms(
        all_rotations[source],
        all_translations[source],
        all_rotations[destination],
        all_translations[destination],
    )

    return OInvFoldGraph(  # type: ignore[call-arg]
        node_features=node_features.to(dtype),
        edge_features=edge_features.to(dtype),
        edge_rotations=edge_rotations.to(dtype),
        edge_translations=edge_translations.to(dtype),
        edge_rbf=radial_basis(edge_translations.norm(dim=-1), lower_bound=0.0, upper_bound=50.0, num_bins=16).to(dtype),
        global_edge_rotations=global_edge_rotations.to(dtype),
        global_edge_translations=global_edge_translations.to(dtype),
        global_edge_rbf=radial_basis(
            global_edge_translations.norm(dim=-1),
            lower_bound=0.0,
            upper_bound=50.0,
            num_bins=16,
        ).to(dtype),
        edge_index=edge_index,
        global_edge_index=global_edge_index,
        batch_index=torch.zeros(num_nodes, dtype=torch.long, device=coordinates.device),
        global_batch_index=torch.zeros(
            NUM_OINVFOLD_GLOBAL_NODES,
            dtype=torch.long,
            device=coordinates.device,
        ),
        virtual_indices=torch.arange(
            NUM_OINVFOLD_GLOBAL_NODES,
            dtype=torch.long,
            device=coordinates.device,
        ),
        batch_size=[],
    )


def _reference_frames(
    first: Tensor,
    center: Tensor,
    third: Tensor,
) -> tuple[Tensor, Tensor]:
    x_direction = third.double() - center.double()
    xy_direction = first.double() - center.double()
    finite_directions = torch.isfinite(x_direction).all() & torch.isfinite(xy_direction).all()
    if torch.compiler.is_compiling():
        torch._assert_async(finite_directions, "OInvFold local frame directions exceed the numeric range")
    elif not bool(finite_directions):
        fail("model_numeric", RequestRef("OInvFold local frames", "directions exceed the numeric range"))
    return (
        model_frame_rotation(
            x_direction,
            xy_direction,
            torch.ones_like(center[..., 0], dtype=torch.bool),
            feature="OInvFold local frames",
        ).to(center.dtype),
        center,
    )


def _relative_transforms(
    source_rotations: Tensor,
    source_translations: Tensor,
    destination_rotations: Tensor,
    destination_translations: Tensor,
) -> tuple[Tensor, Tensor]:
    inverse_destination = destination_rotations.transpose(-1, -2)
    rotations = inverse_destination @ source_rotations
    translations = torch.einsum(
        "...ij,...j->...i",
        inverse_destination,
        source_translations - destination_translations,
    )
    return rotations, translations


def _interaction_features(
    coordinates: Tensor,
    node_rotations: Tensor,
    node_translations: Tensor,
    edge_rotations: Tensor,
    edge_translations: Tensor,
    edge_index: Tensor,
    position_index: Tensor,
    direction: int | None,
) -> tuple[Tensor, Tensor]:
    differences = F.pad(
        coordinates.reshape(-1, 3).diff(dim=0),
        (0, 0, 1, 0),
    ).reshape_as(coordinates)
    consecutive = position_index[1:] == position_index[:-1] + 1
    differences[1:, 0] *= consecutive.unsqueeze(-1)
    if direction is None:
        direction = _sequence_direction(coordinates)
    local_differences = torch.einsum(
        "nij,naj->nai",
        node_rotations.transpose(-1, -2),
        differences * direction,
    )
    node_features = _decouple(local_differences).reshape(coordinates.shape[0], -1)
    node_features = node_features.masked_fill(torch.isnan(node_features), 0)

    source, destination = edge_index
    paired = torch.cat((coordinates[source], coordinates[destination]), dim=1)
    local_paired = torch.einsum(
        "eij,eaj->eai",
        node_rotations[source].transpose(-1, -2),
        paired - node_translations[source].unsqueeze(1),
    )
    edge_geometry = _decouple(local_paired).reshape(edge_index.shape[1], -1)
    rotation_geometry = edge_rotations.transpose(-1, -2).reshape(
        edge_index.shape[1],
        9,
    )
    translation_geometry = _decouple(edge_translations)
    position_geometry = _positional_embeddings(
        edge_index,
        16,
        dtype=coordinates.dtype,
        position_index=position_index,
    )
    edge_features = torch.cat(
        (
            edge_geometry,
            rotation_geometry,
            translation_geometry,
            position_geometry,
        ),
        dim=-1,
    )
    return node_features, edge_features


def _sequence_direction(coordinates: Tensor) -> int:
    first_n, first_c = coordinates[0, 0], coordinates[0, 2]
    second_n, second_c = coordinates[1, 0], coordinates[1, 2]
    return 1 if (first_n - second_c).norm() > (first_c - second_n).norm() else -1


def _global_frames(
    translations: Tensor,
    num_global_nodes: int,
) -> tuple[Tensor, Tensor]:
    """Place virtual global nodes at the centroid with Cartesian frame axes.

    The proper polar factor of a positive-definite covariance is identity.
    Using that declared frame directly also defines the singular case without
    differentiating an indeterminate SVD basis.
    """
    mean = translations.mean(dim=0, keepdim=True)
    rotation = torch.eye(3, dtype=translations.dtype, device=translations.device)
    return (
        rotation.unsqueeze(0).expand(num_global_nodes, -1, -1),
        mean.expand(num_global_nodes, -1),
    )


def _decouple(vectors: Tensor) -> Tensor:
    norms = vectors.norm(dim=-1, keepdim=True)
    return torch.cat(
        (
            vectors / (norms + 1e-6),
            radial_basis(norms[..., 0], lower_bound=0.0, upper_bound=20.0, num_bins=16),
        ),
        dim=-1,
    )


def _positional_embeddings(
    edge_index: Tensor,
    embedding_size: int,
    *,
    dtype: torch.dtype,
    position_index: Tensor,
) -> Tensor:
    differences = position_index[edge_index[0]] - position_index[edge_index[1]]
    frequencies = torch.exp(
        torch.arange(
            0,
            embedding_size,
            2,
            dtype=dtype,
            device=edge_index.device,
        )
        * -(math.log(10000.0) / embedding_size)
    )
    angles = differences.unsqueeze(-1) * frequencies
    return torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1)
