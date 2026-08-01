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
gRNAde input processing, written against the shared preparation contract.

The feature equations follow geometric-rna-design v1.0.0 revision
``2453b18778a1981ac25f314790e5395f1e9ab218``. ODesign's released wrapper uses
the same equations with an all-unpaired secondary-structure graph.

The rows are the RNA components of one instance holding C4'; each row consumes its P, C4' and base nitrogen (N9 for
the purine classes, N1 for the pyrimidine classes, else N9 when the component has one). Aligned conformations of the
same RNA add structural features only; the first is the target. Sequence design regenerates no coordinate: the scope is
empty and the design producer builds the final target.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import cast

import numpy as np
import torch
import torch.nn.functional as F
from multimolecule.graph import knn_graph
from multimolecule.utils.rna.secondary_structure import dot_bracket_to_pairs
from numpy.typing import NDArray
from torch import Tensor

from ...data.crop import Cropper, crop_token_layout
from ...data.molecular import MolecularSystem, MonomerType
from ...data.structure import Structure
from ...data.tokenization import CLASS_MONOMER, TOKEN_INDEX, TokenLayout, select_tokens
from ...errors import RequestRef, fail
from ...modules.functional import radial_basis
from ...modules.geometry import dihedral_angle, smoothed_vector_norm
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
from ..processing import project_model_coordinates
from ..targets import SequenceTarget, prepare_sequence_target
from .config import GrnadeV032ProcessorConfig
from .model import GRNADE_CLASSES, GrnadeInput

_RNA = MonomerType.RNA
GRNADE_LAYOUT = LayoutTable(
    family="grnade",
    monomer_types={_RNA: 0},
    residue_slots=None,
    monomer_slots={_RNA: ("P", "C4'", "N9", "N1")},
    hydrogens="exclude",
    per_atom_components="none",
    exclude_designated_leaving=False,
    centre_atom={_RNA: "C4'"},
    frame_atoms={},
    representative={},
    representative_default={},
)
#: gRNAde reads coordinates and classes of P, C4' and the base nitrogen only.
GRNADE_GATING = GatingPolicy(refuse_valence_excess=False)
_PURINES = (TOKEN_INDEX["RNA_A"], TOKEN_INDEX["RNA_G"])
_PYRIMIDINES = (TOKEN_INDEX["RNA_C"], TOKEN_INDEX["RNA_U"])

_FILL_VALUE = 1e-5
_DISTANCE_EPS = 1e-5
_NUM_NEIGHBORS = 32
_NUM_RBF = 32
_POSITIONAL_EMBEDDING_SIZE = 32
_V032_DISTANCE_EPS = 1e-3


def _owner(pointer: NDArray[np.int64]) -> NDArray[np.int64]:
    return np.repeat(np.arange(len(pointer) - 1, dtype=np.int64), np.diff(pointer))


class _Conformations:
    r"""
    The target's rows, its accepted gating outcomes and every conformation's backbone (``[C, rows, 3]`` atoms, fill
    value where unobserved). Every conformation is gated; the record holds the target's outcomes (its subjects index
    the target).
    """

    def __init__(
        self,
        structures: Sequence[Structure],
        instance_id: str | None,
        filter_by_occupancy: bool,
        linker_coordinates: Mapping[int, NDArray[np.float64]] | None = None,
    ) -> None:
        target = structures[0]
        self.layout = _rows(target, instance_id)
        system = target.system
        instance = int(_owner(system.instance_component_ptr)[self.layout.token_component[0]])
        symbol = str(system.text("instance", system.instance_symbol[instance : instance + 1])[0])
        self.residue_index = self.layout.token_component - system.instance_component_ptr[instance]
        identity = _identity(target, self.layout)
        coordinates, masks = [], []
        for index, structure in enumerate(structures):
            layout = self.layout if index == 0 else _rows(structure, symbol)
            if index and not np.array_equal(_identity(structure, layout), identity):
                fail("value_invalid", RequestRef("structure", f"conformation {index}"), reason="sequence_differs")
            slots = _slots(structure.system, layout)
            accepted = gate(structure, slots, GRNADE_GATING)
            if index == 0:
                self.accepted = accepted
            observed = structure.coordinate_mask
            if filter_by_occupancy:
                observed = observed & ~(known(structure.occupancy_state) & (structure.occupancy <= 0))
            atoms = slots.atom.reshape(-1, 3)
            mask = (atoms >= 0) & observed[np.maximum(atoms, 0)]
            points = np.where(mask[..., None], structure.coordinates[np.maximum(atoms, 0)], _FILL_VALUE)
            if linker_coordinates:
                for row, component in enumerate(layout.token_component.tolist()):
                    if component in linker_coordinates and atoms[row, 2] < 0:
                        points[row, 2] = linker_coordinates[component]
                        mask[row, 2] = True
            coordinates.append(points)
            masks.append(mask)
        self.coordinates, self.mask = np.stack(coordinates), np.stack(masks)

    def record(self, system: MolecularSystem, layout: TokenLayout) -> PreparationRecord:
        r"""The target's encoding losses on ``layout`` and its accepted gating outcomes."""
        _, losses = model_tokens(system, layout)
        records = Records.concat((component_records("encoding_loss", losses.component), self.accepted))
        return PreparationRecord(records, np.zeros(len(system.component_symbol), dtype=np.uint8))


class GrnadeProcessor(Processor):
    r"""
    Lower aligned canonical RNA conformations to the released gRNAde v1 tensor graph.

    Args:
        device: Device for produced tensors.
        dtype: Floating-point dtype for produced tensors.
    """

    def prepare_input(
        self,
        structure: Structure | Sequence[Structure],
        *,
        instance_id: str | None = None,
        design: PositionSelection = PositionSelection(),
        secondary_structure: str | None = None,
        use_coordinates: bool = True,
        filter_by_occupancy: bool = True,
        cropper: Cropper | None = None,
        generator: np.random.Generator | None = None,
        _linker_coordinates: Mapping[int, NDArray[np.float64]] | None = None,
    ) -> PreparedInput[GrnadeInput]:
        r"""
        Lower one RNA with optional 2D and multi-conformation 3D context.

        Args:
            structure: One RNA structure, or aligned conformations of the same RNA; the first is the target.
            instance_id: The RNA instance; may be omitted when one instance holds RNA rows.
            design: The positions allowed to change among the instance's rows (default every one).
            secondary_structure: Dot-bracket over the instance's rows (pseudoknots by further bracket types), read by
                MultiMolecule's ``dot_bracket_to_pairs``; ``None`` is all unpaired.
            use_coordinates: Whether to include P, C4' and base-nitrogen geometry (each internal-coordinate term
                requires all its atoms observed); ``False`` builds the released 1D+2D-only graph.
            filter_by_occupancy: Whether atoms with a KNOWN occupancy at or below zero are treated as unobserved.
            cropper: Optional training crop over the rows (the paired rows are kept); needs ``generator``.
            generator: The numpy generator of the crop.

        Notes:
            All conformations contain the same RNA sequence; the first is the design target and later conformations
            supply structural context. Explicit dot-bracket base-pair edges take priority over overlapping spatial or
            sequence edges. ``use_coordinates=False`` builds the released 1D+2D-only graph.
        """

        if cropper is not None and generator is None:
            fail("request_invalid_generator", RequestRef("generator", "None"))
        structures = (structure,) if isinstance(structure, Structure) else tuple(structure)
        target = structures[0]
        system = target.system
        context = _Conformations(structures, instance_id, filter_by_occupancy, _linker_coordinates)
        layout = context.layout
        pairs = _pairs(secondary_structure, len(layout.token_component))
        allowed = np.zeros(len(system.component_symbol), dtype=bool)
        allowed[layout.token_component] = True
        positions = resolve_positions(target, design, allowed=allowed)
        keep = np.ones(len(layout.token_component), dtype=bool)
        if cropper is not None:
            required = np.zeros(len(keep), dtype=bool)
            required[pairs.reshape(-1)] = True
            cropped = crop_token_layout(
                target, layout, cropper, required=required, generator=cast(np.random.Generator, generator)
            )
            keep = np.isin(layout.token_component, cropped.token_component)
        row = np.cumsum(keep) - 1
        layout = select_tokens(layout, keep)
        tokens = layout.token_class.astype(np.int64)
        raw_mask = torch.from_numpy(context.mask[:, keep])
        coordinates = project_model_coordinates(
            torch.from_numpy(context.coordinates[:, keep]),
            raw_mask if use_coordinates else torch.zeros_like(raw_mask),
            dtype=self.dtype,
            feature="RNA backbone",
        ).to(device=self.device)
        atom_mask = raw_mask.to(device=self.device)
        conformation_mask = atom_mask.any(dim=-1).any(dim=-1)
        if not use_coordinates:
            coordinates = torch.zeros_like(coordinates)
            atom_mask = torch.zeros_like(atom_mask)
        node_scalar, node_vector, edge_scalar, edge_vector, edge_index, conformation_mask = _rna_graph_features(
            coordinates,
            atom_mask,
            torch.as_tensor(
                np.isin(tokens, GRNADE_CLASSES) | np.isin(layout.token_component, tuple(_linker_coordinates or ())),
                device=self.device,
            ),
            torch.as_tensor(row[pairs].T.reshape(2, -1), dtype=torch.long, device=self.device),
            conformation_mask,
            torch.as_tensor(context.residue_index[keep], device=self.device),
            num_neighbors=_NUM_NEIGHBORS,
            num_rbf=_NUM_RBF,
            positional_embedding_size=_POSITIONAL_EMBEDDING_SIZE,
            distance_epsilon=_DISTANCE_EPS,
        )
        features = GrnadeInput(  # type: ignore[call-arg]
            node_scalar_features=node_scalar,
            node_vector_features=node_vector,
            edge_scalar_features=edge_scalar,
            edge_vector_features=edge_vector,
            edge_index=edge_index,
            sequence_tokens=torch.as_tensor(tokens, device=self.device),
            conformation_mask=conformation_mask,
            design_mask=torch.as_tensor(np.isin(layout.token_component, positions), device=self.device),
            component_index=torch.as_tensor(layout.token_component, device=self.device),
            batch_size=[],
        )
        slots = _slots(system, layout)
        return PreparedInput(
            features,
            PreparationContext(
                target,
                slots,
                RegenerationScope.none(system),
                context.record(system, layout),
                spec=self.preparation_spec,
            ),
        )

    def prepare_target(self, target: Structure, *, prepared: PreparedInput[GrnadeInput]) -> SequenceTarget:
        r"""Align RNA sequence truth to the same prepared conformation rows.

        Coordinates, conformer choice, graph features, and input sequence fields
        remain unchanged. Missing or unknown truth is explicitly unavailable.
        """
        self._validate_prepared(prepared)
        result = prepare_sequence_target(prepared.context.structure, target, prepared.features.component_index)
        kind = torch.as_tensor(CLASS_MONOMER, device=result.tokens.device)[result.tokens]
        if bool((result.mask & (kind != MonomerType.RNA)).any()):
            fail("model_capacity", RequestRef("target", "RNA sequence supervision is required"))
        return result


class GrnadeV032Processor(Processor):
    r"""
    Lower canonical RNA conformations with the released v0.3.2 graph equation.

    Args:
        config: Released conformation count carried by the paired checkpoint.
        device: Device for produced tensors.
        dtype: Floating-point dtype for produced tensors.
    """

    config_class = GrnadeV032ProcessorConfig
    config: GrnadeV032ProcessorConfig

    def __init__(
        self,
        config: GrnadeV032ProcessorConfig | None = None,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.config = GrnadeV032ProcessorConfig() if config is None else config
        super().__init__(device=device, dtype=dtype)

    def prepare_input(
        self,
        structure: Structure | Sequence[Structure],
        *,
        instance_id: str | None = None,
        design: PositionSelection = PositionSelection(),
        use_coordinates: bool = True,
        filter_by_occupancy: bool = True,
        cropper: Cropper | None = None,
        generator: np.random.Generator | None = None,
    ) -> PreparedInput[GrnadeInput]:
        r"""
        Build the legacy spatial-only graph from selected RNA conformations.

        When fewer conformations are available than the checkpoint count, all are retained in input order and the
        remaining rows are drawn with replacement; otherwise a uniform subset without replacement is used. Rows not
        observed in every selected conformation, and rows of no predicted class, are dropped. ``generator`` draws the
        conformations (a fresh one when ``None``) and the optional crop (required then).

        Notes:
            The first conformation is the design target; the other aligned conformations provide structural context
            for the same RNA sequence. The legacy graph has no base-pair or sequence-edge channels.
            ``use_coordinates=False`` produces zero geometric features and no spatial edges.
        """

        if cropper is not None and generator is None:
            fail("request_invalid_generator", RequestRef("generator", "None"))
        generator = np.random.default_rng() if generator is None else generator
        structures = (structure,) if isinstance(structure, Structure) else tuple(structure)
        target = structures[0]
        system = target.system
        context = _Conformations(structures, instance_id, filter_by_occupancy)
        layout = context.layout
        wanted, available = self.config.max_num_conformations, len(structures)
        if wanted > available:
            selected = np.concatenate((np.arange(available), generator.integers(available, size=wanted - available)))
        else:
            selected = generator.permutation(available)[:wanted]
        mask = context.mask[selected]
        keep = mask.all(axis=(0, 2)) & np.isin(layout.token_class, GRNADE_CLASSES)
        layout = select_tokens(layout, keep)
        allowed = np.zeros(len(system.component_symbol), dtype=bool)
        allowed[layout.token_component] = True
        positions = resolve_positions(target, design, allowed=allowed)
        if cropper is not None:
            cropped = crop_token_layout(target, layout, cropper, generator=generator)
            keep[keep] = np.isin(layout.token_component, cropped.token_component)
            layout = cropped
        coordinates = project_model_coordinates(
            torch.from_numpy(context.coordinates[selected][:, keep]),
            torch.from_numpy(mask[:, keep] if use_coordinates else np.zeros_like(mask[:, keep])),
            dtype=self.dtype,
            feature="RNA backbone",
        ).to(device=self.device)
        conformation_mask = torch.from_numpy(mask.any(axis=(1, 2))).to(device=self.device)
        if not use_coordinates:
            coordinates = torch.zeros_like(coordinates)
        node_scalar, node_vector, edge_scalar, edge_vector, edge_index = _v032_graph_features(
            coordinates,
            torch.as_tensor(context.residue_index[keep], device=self.device),
            num_neighbors=_NUM_NEIGHBORS,
            num_rbf=_NUM_RBF,
            positional_embedding_size=_POSITIONAL_EMBEDDING_SIZE,
            use_coordinates=use_coordinates,
        )
        rows = len(layout.token_component)
        features = GrnadeInput(  # type: ignore[call-arg]
            node_scalar_features=node_scalar,
            node_vector_features=node_vector,
            edge_scalar_features=edge_scalar,
            edge_vector_features=edge_vector,
            edge_index=edge_index,
            sequence_tokens=torch.as_tensor(layout.token_class.astype(np.int64), device=self.device),
            conformation_mask=conformation_mask.unsqueeze(0).expand(rows, -1),
            design_mask=torch.as_tensor(np.isin(layout.token_component, positions), device=self.device),
            component_index=torch.as_tensor(layout.token_component, device=self.device),
            batch_size=[],
        )
        slots = _slots(system, layout)
        return PreparedInput(
            features,
            PreparationContext(
                target,
                slots,
                RegenerationScope.none(system),
                context.record(system, layout),
                spec=self.preparation_spec,
            ),
        )

    def prepare_target(self, target: Structure, *, prepared: PreparedInput[GrnadeInput]) -> SequenceTarget:
        r"""Align RNA sequence truth to the same prepared conformation rows.

        Coordinates, conformer choice, graph features, and input sequence fields
        remain unchanged. Missing or unknown truth is explicitly unavailable.
        """
        self._validate_prepared(prepared)
        result = prepare_sequence_target(prepared.context.structure, target, prepared.features.component_index)
        kind = torch.as_tensor(CLASS_MONOMER, device=result.tokens.device)[result.tokens]
        if bool((result.mask & (kind != MonomerType.RNA)).any()):
            fail("model_capacity", RequestRef("target", "RNA sequence supervision is required"))
        return result


def _residue_atoms(system: MolecularSystem) -> NDArray[np.bool_]:
    r"""The atoms of components holding C4' (the row centre): gRNAde reads residues only."""
    component = _owner(system.component_atom_ptr)
    holds = np.zeros(len(system.component_symbol), dtype=bool)
    holds[component[system.text("atom", system.atom_symbol) == "C4'"]] = True
    return holds[component]


def _rows(structure: Structure, instance_id: str | None) -> TokenLayout:
    r"""The RNA rows of the named instance, or of the one instance holding RNA rows."""
    system = structure.system
    layout = token_layout(system, GRNADE_LAYOUT, atom_mask=_residue_atoms(system))
    instance = _owner(system.instance_component_ptr)[layout.token_component]
    present = np.unique(instance)
    if instance_id is None and len(present) != 1:
        fail("request_invalid_positions", RequestRef("instance_id", "None"), instances=len(present))
    chosen = present[0] if instance_id is None else int(system.instance_index([instance_id])[0])
    if chosen not in present:
        fail("request_invalid_positions", RequestRef("instance_id", str(instance_id)), reason="no_rna_rows")
    return select_tokens(layout, instance == chosen)


def _identity(structure: Structure, layout: TokenLayout) -> NDArray[np.str_]:
    r"""Per row, its component id: conformations of one RNA hold the same."""
    return structure.system.text("component", structure.system.component_symbol[layout.token_component])


def _slots(system: MolecularSystem, layout: TokenLayout) -> SlotMap:
    r"""
    The rows' P, C4' and base nitrogen: N9 for the purine classes, N1 for the pyrimidine classes, else N9 when the
    component has one, else N1 (the layout addresses both; the slot map keeps the one consumed).
    """

    atoms = slot_map(system, layout, GRNADE_LAYOUT).atom.reshape(-1, 4)
    classes = layout.token_class
    purine = np.isin(classes, _PURINES) | (~np.isin(classes, _PYRIMIDINES) & (atoms[:, 2] >= 0))
    atom = np.stack((atoms[:, 0], atoms[:, 1], np.where(purine, atoms[:, 2], atoms[:, 3])), axis=1).reshape(-1)
    addressed = np.zeros(len(system.atom_symbol), dtype=bool)
    addressed[atom[atom >= 0]] = True
    token = np.repeat(np.arange(len(classes), dtype=np.int64), 3)
    return SlotMap(atom=atom, token=token, layout=layout, unaddressed=np.flatnonzero(~addressed).astype(np.int64))


def _pairs(secondary_structure: str | None, rows: int) -> NDArray[np.int64]:
    r"""The base pairs (``[pairs, 2]`` row indices) of a dot-bracket over the rows."""
    if secondary_structure is None:
        return np.zeros((0, 2), dtype=np.int64)
    if len(secondary_structure) != rows:
        fail("value_invalid", RequestRef("secondary_structure", secondary_structure), rows=rows)
    try:
        pairs = dot_bracket_to_pairs(secondary_structure)
    except ValueError as error:
        fail("value_invalid", RequestRef("secondary_structure", secondary_structure), reason=str(error))
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def _rna_graph_features(
    coordinates: Tensor,
    atom_mask: Tensor,
    sequence_mask: Tensor,
    base_pair_index: Tensor,
    valid_conformations: Tensor,
    residue_index: Tensor,
    *,
    num_neighbors: int,
    num_rbf: int,
    positional_embedding_size: int,
    distance_epsilon: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    coordinate_residue_mask = atom_mask.any(dim=-1)
    conformation_mask = valid_conformations.unsqueeze(0).expand(sequence_mask.shape[0], -1)
    residue_mask = coordinate_residue_mask & sequence_mask.unsqueeze(0)
    dihedral, angle, length, length_validity = _internal_coordinates(
        coordinates,
        atom_mask & sequence_mask[None, :, None],
        residue_index,
        distance_epsilon=_DISTANCE_EPS,
    )
    angular = torch.cat((dihedral, angle), dim=-1)
    node_scalar = torch.cat(
        (
            torch.cos(angular),
            torch.sin(angular),
            _log_internal_lengths(length, length_validity, distance_epsilon),
        ),
        dim=-1,
    )
    node_scalar = node_scalar * residue_mask.unsqueeze(-1)
    node_vector = _internal_vectors(
        coordinates,
        atom_mask,
        residue_index,
        distance_epsilon=_DISTANCE_EPS,
    )

    edge_index, edge_type = _rna_edges(
        coordinates,
        atom_mask,
        residue_mask,
        sequence_mask,
        base_pair_index,
        residue_index,
        num_neighbors=num_neighbors,
        distance_epsilon=distance_epsilon,
    )
    coordinates_by_residue = coordinates.permute(1, 0, 2, 3)
    atom_mask_by_residue = atom_mask.permute(1, 0, 2)
    source, destination = edge_index
    edge_displacements = coordinates_by_residue[source] - coordinates_by_residue[destination]
    edge_atom_mask = atom_mask_by_residue[source] & atom_mask_by_residue[destination]
    edge_displacements = edge_displacements * edge_atom_mask.unsqueeze(-1)
    edge_lengths = smoothed_vector_norm(edge_displacements, eps=distance_epsilon)
    edge_lengths = edge_lengths * edge_atom_mask
    edge_rbf = _rbf_expansion(edge_lengths, num_rbf=num_rbf)
    edge_position = _positional_encoding(
        (residue_index[source] - residue_index[destination]).unsqueeze(-1),
        positional_embedding_size,
        dtype=coordinates.dtype,
    ).unsqueeze(1)
    edge_position = edge_position.expand(-1, coordinates.shape[0], -1)
    edge_type_features = F.one_hot(edge_type, num_classes=4).to(coordinates.dtype).unsqueeze(1)
    edge_type_features = edge_type_features.expand(-1, coordinates.shape[0], -1)
    edge_scalar = torch.cat((edge_type_features, edge_rbf, edge_position), dim=-1)
    edge_vector = _normalize(edge_displacements, distance_epsilon=_DISTANCE_EPS)

    return (
        node_scalar.permute(1, 0, 2),
        node_vector.permute(1, 0, 2, 3),
        edge_scalar,
        edge_vector,
        edge_index,
        conformation_mask,
    )


def _v032_graph_features(
    coordinates: Tensor,
    residue_index: Tensor,
    *,
    num_neighbors: int,
    num_rbf: int,
    positional_embedding_size: int,
    use_coordinates: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    num_conformations, num_residues = coordinates.shape[:2]
    if not use_coordinates:
        return (
            coordinates.new_zeros(num_residues, num_conformations, 15),
            coordinates.new_zeros(num_residues, num_conformations, 4, 3),
            coordinates.new_zeros(0, num_conformations, 131),
            coordinates.new_zeros(0, num_conformations, 3, 3),
            torch.empty(2, 0, device=coordinates.device, dtype=torch.long),
        )
    residue_mask = torch.ones(
        num_conformations,
        num_residues,
        device=coordinates.device,
        dtype=torch.bool,
    )
    dihedral, angle, length, length_validity = _internal_coordinates(
        coordinates,
        torch.ones_like(coordinates[..., 0], dtype=torch.bool),
        residue_index,
        distance_epsilon=_V032_DISTANCE_EPS,
    )
    angular = torch.cat((dihedral, angle), dim=-1)
    node_scalar = torch.cat(
        (
            angular.cos(),
            angular.sin(),
            _log_internal_lengths(length, length_validity, _V032_DISTANCE_EPS),
        ),
        dim=-1,
    )
    node_vector = _internal_vectors(
        coordinates,
        torch.ones_like(coordinates[..., 0], dtype=torch.bool),
        residue_index,
        distance_epsilon=_V032_DISTANCE_EPS,
    )

    edges = [
        _knn_edges(conformation.mean(dim=1), residue_mask[index], num_neighbors)
        for index, conformation in enumerate(coordinates)
    ]
    edge_index = _undirected_union(edges, num_residues)
    coordinates_by_residue = coordinates.permute(1, 0, 2, 3)
    source, destination = edge_index
    displacement = coordinates_by_residue[source] - coordinates_by_residue[destination]
    edge_length = smoothed_vector_norm(displacement, eps=_V032_DISTANCE_EPS)
    edge_rbf = _rbf_expansion(edge_length, num_rbf=num_rbf)
    edge_position = _positional_encoding(
        (residue_index[source] - residue_index[destination]).unsqueeze(-1),
        positional_embedding_size,
        dtype=coordinates.dtype,
    ).unsqueeze(1)
    edge_position = edge_position.expand(-1, num_conformations, -1)
    edge_scalar = torch.cat((edge_rbf, edge_position, torch.log(edge_length)), dim=-1)
    edge_vector = _normalize(displacement, distance_epsilon=_V032_DISTANCE_EPS)
    return (
        node_scalar.permute(1, 0, 2),
        node_vector.permute(1, 0, 2, 3),
        edge_scalar,
        edge_vector,
        edge_index,
    )


def _undirected_union(edges: list[Tensor], num_residues: int) -> Tensor:
    edge_index = torch.cat(edges, dim=1)
    edge_index = torch.cat((edge_index, edge_index.flip(0)), dim=1)
    if not edge_index.shape[1]:
        return edge_index
    code = torch.unique(edge_index[0] * num_residues + edge_index[1], sorted=True)
    return torch.stack(
        (
            code.div(num_residues, rounding_mode="floor"),
            code.remainder(num_residues),
        )
    )


def _rna_edges(
    coordinates: Tensor,
    atom_mask: Tensor,
    residue_mask: Tensor,
    sequence_mask: Tensor,
    base_pair_index: Tensor,
    residue_index: Tensor,
    *,
    num_neighbors: int,
    distance_epsilon: float,
) -> tuple[Tensor, Tensor]:
    edge_indices: list[Tensor] = []
    edge_types: list[Tensor] = []
    for conformation_coordinates, conformation_atom_mask, conformation_residue_mask in zip(
        coordinates,
        atom_mask,
        residue_mask,
        strict=True,
    ):
        centroids = conformation_coordinates.sum(dim=1) / (
            conformation_atom_mask.sum(dim=1, keepdim=True) + distance_epsilon
        )
        spatial_edges = _knn_edges(
            centroids,
            conformation_residue_mask,
            num_neighbors,
        )
        edge_indices.append(spatial_edges)
        edge_types.append(
            torch.full(
                (spatial_edges.shape[1],),
                2,
                device=coordinates.device,
                dtype=torch.long,
            )
        )

    edge_indices.append(base_pair_index)
    edge_types.append(
        torch.full(
            (base_pair_index.shape[1],),
            3,
            device=coordinates.device,
            dtype=torch.long,
        )
    )

    sequence_length = sequence_mask.shape[0]
    separation = (residue_index[:, None] - residue_index[None]).abs()
    sequence_edges = (
        ((separation > 0) & (separation <= num_neighbors) & sequence_mask[:, None] & sequence_mask[None])
        .nonzero(as_tuple=False)
        .t()
    )
    edge_indices.append(sequence_edges)
    edge_types.append(
        torch.ones(
            sequence_edges.shape[1],
            device=coordinates.device,
            dtype=torch.long,
        )
    )

    edge_index = torch.cat(edge_indices, dim=1)
    edge_type = torch.cat(edge_types)
    edge_index = torch.cat((edge_index, edge_index.flip(0)), dim=1)
    edge_type = torch.cat((edge_type, edge_type))
    if edge_index.shape[1] == 0:
        return edge_index, edge_type

    code = edge_index[0] * sequence_length + edge_index[1]
    unique_code, inverse = torch.unique(code, sorted=True, return_inverse=True)
    merged_type = torch.zeros(
        unique_code.shape[0],
        device=edge_type.device,
        dtype=edge_type.dtype,
    )
    merged_type.scatter_reduce_(
        0,
        inverse,
        edge_type,
        reduce="amax",
        include_self=True,
    )
    edge_index = torch.stack(
        (
            unique_code.div(sequence_length, rounding_mode="floor"),
            unique_code.remainder(sequence_length),
        )
    )
    keep = sequence_mask[edge_index[0]] & sequence_mask[edge_index[1]]
    return edge_index[:, keep], merged_type[keep]


def _knn_edges(
    coordinates: Tensor,
    mask: Tensor,
    num_neighbors: int,
) -> Tensor:
    r"""Upstream's spatial edges: each residue's nearest neighbours among all residues, kept when both are valid."""
    edges = knn_graph(coordinates, num_neighbors)
    return edges[:, mask[edges[0]] & mask[edges[1]]]


def _internal_coordinates(
    coordinates: Tensor,
    atom_mask: Tensor,
    residue_index: Tensor,
    *,
    distance_epsilon: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    coordinates = coordinates.masked_fill(~atom_mask[..., None], 0.0)
    batch_size, num_residues = coordinates.shape[:2]
    linear = coordinates[:, :, :2].reshape(batch_size, 2 * num_residues, 3)
    linear_length = _length(
        linear[:, 1:],
        linear[:, :-1],
        distance_epsilon=distance_epsilon,
    )
    linear_angle = math.pi - _angle(
        linear[:, :-2],
        linear[:, 1:-1],
        linear[:, 2:],
        distance_epsilon=distance_epsilon,
    )
    linear_dihedral = dihedral_angle(
        linear[:, :-3],
        linear[:, 1:-2],
        linear[:, 2:-1],
        linear[:, 3:],
        eps=distance_epsilon,
    )

    phosphorus, carbon4, nitrogen = coordinates.unbind(dim=2)
    nitrogen_length = _length(
        carbon4,
        nitrogen,
        distance_epsilon=distance_epsilon,
    )
    nitrogen_angle = math.pi - _angle(
        phosphorus,
        carbon4,
        nitrogen,
        distance_epsilon=distance_epsilon,
    )
    nitrogen_dihedral = dihedral_angle(
        phosphorus[:, 1:],
        nitrogen[:, :-1],
        carbon4[:, :-1],
        phosphorus[:, :-1],
        eps=distance_epsilon,
    )

    phosphorus_mask, carbon4_mask, nitrogen_mask = atom_mask.unbind(dim=2)
    linear_mask = atom_mask[:, :, :2].reshape(batch_size, -1)
    linear_position = torch.stack((residue_index * 2, residue_index * 2 + 1), dim=-1).flatten()
    length_mask = linear_mask[:, :-1] & linear_mask[:, 1:] & (linear_position[1:] - linear_position[:-1] == 1)
    angle_mask = (
        linear_mask[:, :-2]
        & linear_mask[:, 1:-1]
        & linear_mask[:, 2:]
        & (linear_position[1:-1] - linear_position[:-2] == 1)
        & (linear_position[2:] - linear_position[1:-1] == 1)
    )
    dihedral_mask = (
        linear_mask[:, :-3]
        & linear_mask[:, 1:-2]
        & linear_mask[:, 2:-1]
        & linear_mask[:, 3:]
        & (linear_position[1:-2] - linear_position[:-3] == 1)
        & (linear_position[2:-1] - linear_position[1:-2] == 1)
        & (linear_position[3:] - linear_position[2:-1] == 1)
    )
    nitrogen_length_mask = carbon4_mask & nitrogen_mask
    nitrogen_angle_mask = phosphorus_mask & carbon4_mask & nitrogen_mask
    nitrogen_dihedral_mask = (
        phosphorus_mask[:, 1:]
        & nitrogen_mask[:, :-1]
        & carbon4_mask[:, :-1]
        & phosphorus_mask[:, :-1]
        & (residue_index[1:] - residue_index[:-1] == 1)
    )

    dihedral = _pad_internal(
        linear_dihedral,
        nitrogen_dihedral,
        left=1,
        right=2,
        num_residues=num_residues,
    )
    angle = _pad_internal(
        linear_angle,
        nitrogen_angle,
        left=1,
        right=1,
        num_residues=num_residues,
    )
    length = _pad_internal(
        linear_length,
        nitrogen_length,
        left=0,
        right=1,
        num_residues=num_residues,
    )
    dihedral_mask = _pad_internal(
        dihedral_mask.to(coordinates.dtype),
        nitrogen_dihedral_mask.to(coordinates.dtype),
        left=1,
        right=2,
        num_residues=num_residues,
    )
    angle_mask = _pad_internal(
        angle_mask.to(coordinates.dtype),
        nitrogen_angle_mask.to(coordinates.dtype),
        left=1,
        right=1,
        num_residues=num_residues,
    )
    length_mask = _pad_internal(
        length_mask.to(coordinates.dtype),
        nitrogen_length_mask.to(coordinates.dtype),
        left=0,
        right=1,
        num_residues=num_residues,
    )
    return (
        torch.where(dihedral_mask.bool(), dihedral, 0.0),
        torch.where(angle_mask.bool(), angle, 0.0),
        torch.where(length_mask.bool(), length, 0.0),
        length_mask.bool(),
    )


def _log_internal_lengths(length: Tensor, validity: Tensor, epsilon: float) -> Tensor:
    operand = torch.where(validity, length + epsilon, 1.0)
    return torch.where(validity, operand.log(), length.new_tensor(epsilon).log())


def _pad_internal(
    linear: Tensor,
    branched: Tensor,
    *,
    left: int,
    right: int,
    num_residues: int,
) -> Tensor:
    linear = F.pad(linear, (left, right)).reshape(linear.shape[0], num_residues, 2)
    branched = F.pad(branched, (0, num_residues - branched.shape[1]))
    return torch.cat((linear, branched.unsqueeze(-1)), dim=-1)


def _internal_vectors(
    coordinates: Tensor,
    atom_mask: Tensor,
    residue_index: Tensor,
    *,
    distance_epsilon: float,
) -> Tensor:
    phosphorus, carbon4, nitrogen = coordinates.unbind(dim=2)
    phosphorus_mask, carbon4_mask, nitrogen_mask = atom_mask.unbind(dim=2)
    nitrogen = (nitrogen - carbon4) * (nitrogen_mask & carbon4_mask).unsqueeze(-1)
    phosphorus = (phosphorus - carbon4) * (phosphorus_mask & carbon4_mask).unsqueeze(-1)
    adjacent = residue_index[1:] - residue_index[:-1] == 1
    adjacent_mask = carbon4_mask[:, 1:] & carbon4_mask[:, :-1] & adjacent
    forward = F.pad((carbon4[:, 1:] - carbon4[:, :-1]) * adjacent_mask.unsqueeze(-1), (0, 0, 0, 1))
    backward = F.pad((carbon4[:, :-1] - carbon4[:, 1:]) * adjacent_mask.unsqueeze(-1), (0, 0, 1, 0))
    return torch.stack(
        (
            _normalize(phosphorus, distance_epsilon=distance_epsilon),
            _normalize(nitrogen, distance_epsilon=distance_epsilon),
            _normalize(forward, distance_epsilon=distance_epsilon),
            _normalize(backward, distance_epsilon=distance_epsilon),
        ),
        dim=-2,
    )


def _length(
    first: Tensor,
    second: Tensor,
    *,
    distance_epsilon: float,
) -> Tensor:
    return smoothed_vector_norm(second - first, eps=distance_epsilon)


def _angle(
    first: Tensor,
    center: Tensor,
    last: Tensor,
    *,
    distance_epsilon: float,
) -> Tensor:
    first_direction = _normalize(first - center, distance_epsilon=distance_epsilon)
    last_direction = _normalize(last - center, distance_epsilon=distance_epsilon)
    return torch.acos((first_direction * last_direction).sum(dim=-1).clamp(-1, 1))


def _normalize(
    vectors: Tensor,
    *,
    distance_epsilon: float,
) -> Tensor:
    return vectors / smoothed_vector_norm(vectors, eps=distance_epsilon, keepdim=True)


def _rbf_expansion(
    values: Tensor,
    *,
    num_rbf: int,
) -> Tensor:
    expanded = radial_basis(values, num_rbf, 0.0, 30.0, width=30.0 / (num_rbf - 1))
    return expanded.reshape(*values.shape[:-1], values.shape[-1] * num_rbf)


def _positional_encoding(
    values: Tensor,
    embedding_size: int,
    *,
    dtype: torch.dtype,
) -> Tensor:
    frequencies = embedding_size // 2
    periods = torch.logspace(
        0,
        3,
        frequencies,
        device=values.device,
        dtype=dtype,
    )
    angular_frequency = 2 * math.pi / periods
    encoded = values.unsqueeze(-1) * angular_frequency
    return torch.cat((encoded.cos(), encoded.sin()), dim=-1).reshape(
        *values.shape[:-1],
        values.shape[-1] * 2 * frequencies,
    )
