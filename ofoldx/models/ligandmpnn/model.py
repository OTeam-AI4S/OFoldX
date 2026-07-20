r"""
LigandMpnn context-aware sequence-design model.

The OFoldX-native reimplementation of LigandMPNN: a message-passing graph model that designs protein
sequences conditioned on backbone geometry and nearby ligand/context atoms. The public surface is
[`LigandMpnnModel`][ofoldx.models.ligandmpnn.model.LigandMpnnModel] and its typed input
[`LigandMpnnInput`][ofoldx.models.ligandmpnn.model.LigandMpnnInput]. Raw forward and sampling return
the shared [`MpnnOutput`][ofoldx.models.mpnn.MpnnOutput] and
[`MpnnSamplingOutput`][ofoldx.models.mpnn.MpnnSamplingOutput]; the four probability operations return the
shared [`MpnnProbabilityResult`][ofoldx.models.mpnn.MpnnProbabilityResult]; ``design`` returns the shared
[`DesignResult`][ofoldx.results.DesignResult]. The root Model owns the released feature Module directly:
there is no ``LigandMpnnBackbone`` checkpoint-prefix wrapper.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ...data.components import get_component
from ...data.molecular import (
    ComponentReference,
    MolecularSystem,
    Molecule,
    PolymerSequence,
)
from ...data.structure import Structure
from ...modules.mpnn import (
    AMINO_ACID_COMPONENT_IDS,
    MpnnDecoderLayer,
    MpnnEncoderLayer,
    MpnnOutput,
    MpnnProteinFeatures,
    MpnnSamplingOutput,
    _autoregressive_masks,
    _decoding_order_to_ranks,
    _radial_basis,
    _virtual_beta_carbon,
    cat_neighbors_nodes,
    gather_nodes,
    periodic_group_period,
)
from ...results import DesignCandidate, DesignResult, MpnnProbabilityResult, Score
from .config import LigandMpnnConfig

_VOCABULARY = len(AMINO_ACID_COMPONENT_IDS) + 1
_GENERATABLE = len(AMINO_ACID_COMPONENT_IDS)


def side_chain_atom_types(*, device: torch.device | str | None = None) -> Tensor:
    r"""Build LigandMPNN's 32-slot side-chain element lookup for atom37 slots ``5:37``."""

    return torch.tensor(
        [6, 6, 6, 8, 8, 16, 6, 6, 6, 7, 7, 8, 8, 16, 6, 6, 6, 6, 7, 7, 7, 8, 8, 6, 7, 7, 8, 6, 6, 6, 7, 8],
        dtype=torch.long,
        device=device,
    )


def _backbone_atoms(coordinates: Tensor) -> dict[str, Tensor]:
    r"""Split N/CA/C/O coordinates and add the idealized virtual Cβ atom."""

    nitrogen = coordinates[:, :, 0]
    alpha_carbon = coordinates[:, :, 1]
    carbon = coordinates[:, :, 2]
    oxygen = coordinates[:, :, 3]
    beta_carbon = _virtual_beta_carbon(nitrogen, alpha_carbon, carbon)
    return {"N": nitrogen, "CA": alpha_carbon, "C": carbon, "O": oxygen, "CB": beta_carbon}


def _local_angle_features(
    nitrogen: Tensor,
    alpha_carbon: Tensor,
    carbon: Tensor,
    points: Tensor,
) -> Tensor:
    r"""Direction features of ``points`` expressed in each residue's local backbone frame."""

    first_axis = F.normalize(nitrogen - alpha_carbon, dim=-1)
    carbon_vector = carbon - alpha_carbon
    projected = torch.einsum("bli,bli->bl", first_axis, carbon_vector)[..., None]
    second_axis = F.normalize(carbon_vector - first_axis * projected, dim=-1)
    third_axis = torch.cross(first_axis, second_axis, dim=-1)
    frame = torch.stack((first_axis, second_axis, third_axis), dim=-1)
    local = torch.einsum("blqp,blyq->blyp", frame, points - alpha_carbon[:, :, None])
    planar_radius = torch.sqrt(local[..., 0] ** 2 + local[..., 1] ** 2 + 1e-8)
    spatial_radius = torch.norm(local, dim=-1) + 1e-8
    return torch.stack(
        (
            local[..., 0] / planar_radius,
            local[..., 1] / planar_radius,
            planar_radius / spatial_radius,
            local[..., 2] / spatial_radius,
        ),
        dim=-1,
    )


def _decode_groups(decode_order: Tensor, tied_groups: list[list[int]]) -> list[list[int]]:
    r"""Order tied groups by the first time any member appears in the decode order."""

    ordered: list[list[int]] = []
    seen: set[int] = set()
    for residue in decode_order.detach().cpu().tolist():
        residue = int(residue)
        if residue in seen:
            continue
        group = next((candidate for candidate in tied_groups if residue in candidate), None)
        selected = list(group) if group is not None else [residue]
        ordered.append(selected)
        seen.update(selected)
    return ordered


def _collapse_tied_order(
    decode_order_row: Tensor,
    tied_groups: tuple[tuple[int, ...], ...],
) -> Tensor:
    r"""Collapse a drawn order so every tied group decodes consecutively from its first member."""

    ordered = _decode_groups(decode_order_row, [list(group) for group in tied_groups])
    return torch.tensor(
        list(itertools.chain.from_iterable(ordered)),
        device=decode_order_row.device,
        dtype=torch.long,
    )


def _realize_designed_system(
    system: MolecularSystem,
    component_references: tuple[ComponentReference | None, ...],
    sampled_tokens: list[int],
    design_mask: list[bool],
) -> MolecularSystem:
    r"""Realize one candidate while preserving entity, instance, and bond identity.

    Designed protein occurrences take their sampled generatable identity; every untouched entity,
    instance, and resolvable stored bond is preserved. Changed copies of a shared entity partition by
    their complete output component-ID sequence exactly as frozen for ProteinMpnn, and entities from
    different originals never merge.
    """

    replacements = {
        reference: get_component(AMINO_ACID_COMPONENT_IDS[int(token)])
        for reference, token, selected in zip(component_references, sampled_tokens, design_mask)
        if reference is not None and selected
    }
    realized_by_instance: dict[str, PolymerSequence | Molecule] = {}
    for instance_id, entity_id in system.instances.items():
        entity = system.entities[entity_id]
        if isinstance(entity, PolymerSequence):
            realized_by_instance[instance_id] = PolymerSequence(
                entity.polymer_type,
                tuple(
                    replacements.get((instance_id, index), component)
                    for index, component in enumerate(entity.components)
                ),
            )
        else:
            realized_by_instance[instance_id] = entity

    # TODO(§3.4.1 reopened): family-owned canonical entity splitting is still reopened for LigandMpnn.
    # Until it freezes we partition changed copies within each original entity by complete output
    # component-ID sequence, exactly as frozen for ProteinMpnn (§4.7); this is the minimal current
    # behavior and never merges entities from different originals.
    partitions: dict[str, list[tuple[tuple[str, ...], PolymerSequence | Molecule, list[str]]]] = {
        entity_id: [] for entity_id in system.entities
    }
    for instance_id, entity_id in system.instances.items():
        entity = realized_by_instance[instance_id]
        signature = tuple(component.component_id for component in entity.components)
        partition = next(
            (candidate for candidate in partitions[entity_id] if candidate[0] == signature),
            None,
        )
        if partition is None:
            partitions[entity_id].append((signature, entity, [instance_id]))
        else:
            partition[2].append(instance_id)

    used_ids = set(system.entities)
    new_entities: dict[str, PolymerSequence | Molecule] = {}
    instance_entities: dict[str, str] = {}
    for entity_id, original in system.entities.items():
        values = partitions[entity_id]
        original_signature = tuple(component.component_id for component in original.components)
        primary_index = next(
            (index for index, value in enumerate(values) if value[0] == original_signature),
            0,
        )
        for partition_index in (primary_index, *(index for index in range(len(values)) if index != primary_index)):
            signature, entity, instance_ids = values[partition_index]
            if partition_index == primary_index:
                output_id = entity_id
            else:
                suffix = 2
                output_id = f"{entity_id}_{suffix}"
                while output_id in used_ids:
                    suffix += 1
                    output_id = f"{entity_id}_{suffix}"
                used_ids.add(output_id)
            new_entities[output_id] = original if signature == original_signature else entity
            for instance_id in instance_ids:
                instance_entities[instance_id] = output_id
    new_instances = {instance_id: instance_entities[instance_id] for instance_id in system.instances}
    candidate = MolecularSystem(new_entities, new_instances)
    retained_bonds = []
    for bond in system.bonds:
        try:
            candidate.atom(bond.site_a)
            candidate.atom(bond.site_b)
            if bond.stereo_sites is not None:
                candidate.atom(bond.stereo_sites[0])
                candidate.atom(bond.stereo_sites[1])
        except KeyError:
            continue
        retained_bonds.append(bond)
    return MolecularSystem(new_entities, new_instances, frozenset(retained_bonds))


class LigandMpnnFeatures(MpnnProteinFeatures):
    r"""
    Input featurizer for the LigandMPNN backbone graph and ligand/side-chain context.

    Derives the virtual Cβ atom, builds the k-nearest-neighbor residue graph and pairwise edge features
    (through the shared full-protein edge graph), and in parallel featurizes the nonprotein context atoms
    (per-atom RBF distances to backbone atoms, periodic-table element one-hots, and local-frame angle
    features). When ``use_side_chain_context`` is set at call time, the observed side-chain atoms of the
    nearest fixed residues are folded into that context and the nearest atoms are reselected.

    Args:
        hidden_size: Width of the per-residue edge features and per-atom context node features.
        num_neighbors: Number of nearest-neighbor residues in the graph.
        num_positional_embeddings: Width of the relative-position embedding.
        num_rbf: Number of radial basis functions per distance feature.
    """

    def __init__(
        self,
        hidden_size: int,
        num_neighbors: int,
        *,
        num_positional_embeddings: int = 16,
        num_rbf: int = 16,
    ) -> None:
        super().__init__(
            hidden_size,
            num_neighbors,
            num_positional_embeddings=num_positional_embeddings,
            num_rbf=num_rbf,
        )
        self.atom_projection = nn.Linear(5 * num_rbf + 64 + 4, hidden_size)
        self.atom_norm = nn.LayerNorm(hidden_size)
        self.element_projection = nn.Linear(147, 64)
        self.context_node_projection = nn.Linear(147, hidden_size, bias=False)
        self.context_edge_projection = nn.Linear(num_rbf, hidden_size, bias=False)
        self.context_edge_norm = nn.LayerNorm(hidden_size)
        self.context_node_norm = nn.LayerNorm(hidden_size)
        buffer_device = self.element_projection.weight.device
        if buffer_device.type == "meta":
            buffer_device = torch.device("cpu")
        group, period = periodic_group_period(device=buffer_device)
        self.register_buffer("periodic_group", group, persistent=False)
        self.register_buffer("periodic_period", period, persistent=False)
        self.register_buffer(
            "side_chain_atom_types",
            side_chain_atom_types(device=buffer_device),
            persistent=False,
        )

    def forward(
        self,
        backbone_coordinates: Tensor,
        valid_mask: Tensor,
        residue_indices: Tensor,
        chain_indices: Tensor,
        context_coordinates: Tensor,
        context_atomic_numbers: Tensor,
        context_mask: Tensor,
        residue_context_coordinates: Tensor,
        residue_context_mask: Tensor,
        design_mask: Tensor,
        *,
        use_side_chain_context: bool,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        r"""Featurize backbone geometry and context atoms into graph inputs.

        Coordinate noise, when requested, is already applied by the caller to ``backbone_coordinates`` and
        ``context_coordinates``; the fixed side-chain coordinates are never perturbed.

        Returns:
            Tuple ``(atom_features, edges, neighbor_indices, context_nodes, context_edges, context_mask)``:
            per-residue context node features, residue-graph edges and neighbor indices, and the
            context-atom node/edge features and validity mask.
        """

        num_context_atoms = context_coordinates.shape[2]
        atoms = _backbone_atoms(backbone_coordinates)
        edges, neighbor_indices = super().forward(
            backbone_coordinates,
            valid_mask,
            residue_indices,
            chain_indices,
        )
        if use_side_chain_context:
            context_coordinates, context_atomic_numbers, context_mask = self._append_side_chain_context(
                atoms["CB"],
                context_coordinates,
                context_atomic_numbers,
                context_mask,
                residue_context_coordinates,
                residue_context_mask,
                design_mask,
                valid_mask,
                neighbor_indices,
                num_context_atoms,
            )

        context_atomic_numbers = context_atomic_numbers.long()
        element_features = self._element_features(context_atomic_numbers)
        element_projection = self.element_projection(element_features.to(dtype=backbone_coordinates.dtype))

        backbone_atoms = torch.stack(
            (atoms["N"], atoms["CA"], atoms["C"], atoms["O"], atoms["CB"]),
            dim=2,
        )
        atom_context_distances = torch.sqrt(
            torch.sum((backbone_atoms[:, :, :, None] - context_coordinates[:, :, None]) ** 2, dim=-1) + 1e-6
        )
        atom_context_features = _radial_basis(atom_context_distances.movedim(2, 3), self.num_rbf).flatten(start_dim=-2)
        atom_features = torch.cat(
            (
                atom_context_features,
                element_projection,
                _local_angle_features(atoms["N"], atoms["CA"], atoms["C"], context_coordinates),
            ),
            dim=-1,
        )
        atom_features = self.atom_norm(self.atom_projection(atom_features))
        context_distances = torch.sqrt(
            torch.sum((context_coordinates[:, :, :, None] - context_coordinates[:, :, None, :]) ** 2, dim=-1) + 1e-6
        )
        context_edges = self.context_edge_norm(
            self.context_edge_projection(_radial_basis(context_distances, self.num_rbf))
        )
        context_nodes = self.context_node_norm(
            self.context_node_projection(element_features.to(dtype=backbone_coordinates.dtype))
        )
        return atom_features, edges, neighbor_indices, context_nodes, context_edges, context_mask

    def _append_side_chain_context(
        self,
        beta_carbon: Tensor,
        context_coordinates: Tensor,
        context_atomic_numbers: Tensor,
        context_mask: Tensor,
        residue_context_coordinates: Tensor,
        residue_context_mask: Tensor,
        design_mask: Tensor,
        valid_mask: Tensor,
        neighbor_indices: Tensor,
        num_context_atoms: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        r"""Append the observed side-chain atoms of the nearest fixed residues, then reselect atoms.

        Only source-fixed rows contribute side-chain atoms; the first sixteen residue KNN neighbors
        supply their observed 32-slot side chains, which are concatenated with the nonprotein context
        before the nearest ``num_context_atoms`` atoms are reselected around the virtual Cβ.
        """

        batch, length = context_mask.shape[:2]
        fixed_mask = residue_context_mask * (1 - design_mask[:, :, None])
        neighbor_subset = neighbor_indices[:, :, :16]
        side_chain_mask = gather_nodes(fixed_mask, neighbor_subset).reshape(batch, length, -1)
        side_chain_coordinates = gather_nodes(
            residue_context_coordinates.reshape(batch, length, -1),
            neighbor_subset,
        ).reshape(batch, length, -1, 3)
        side_chain_types = self.side_chain_atom_types[None, None, None, :].expand(
            batch,
            length,
            neighbor_subset.shape[2],
            -1,
        )
        side_chain_types = side_chain_types.reshape(batch, length, -1)
        context_coordinates = torch.cat((side_chain_coordinates, context_coordinates), dim=2)
        context_mask = torch.cat((side_chain_mask, context_mask), dim=2)
        context_atomic_numbers = torch.cat((side_chain_types, context_atomic_numbers), dim=2)
        distances = torch.sum((beta_carbon[:, :, None] - context_coordinates) ** 2, dim=-1)
        valid = valid_mask[:, :, None] * context_mask
        adjusted = distances * valid + (1.0 - valid) * 10000.0
        atom_count = min(num_context_atoms, adjusted.shape[-1])
        atom_indices = torch.topk(adjusted, atom_count, dim=-1, largest=False).indices
        context_coordinates = torch.gather(context_coordinates, 2, atom_indices[:, :, :, None].expand(-1, -1, -1, 3))
        context_atomic_numbers = torch.gather(context_atomic_numbers, 2, atom_indices)
        context_mask = torch.gather(context_mask, 2, atom_indices)
        return context_coordinates, context_atomic_numbers, context_mask

    def _element_features(self, atomic_numbers: Tensor) -> Tensor:
        r"""One-hot-encode each context atom by element, periodic group, and period."""

        group = self.periodic_group[atomic_numbers]
        period = self.periodic_period[atomic_numbers]
        return torch.cat(
            (
                F.one_hot(atomic_numbers, 120),
                F.one_hot(group, 19),
                F.one_hot(period, 8),
            ),
            dim=-1,
        )


class LigandMpnnModel(nn.Module):
    r"""
    Message-passing model that designs protein sequences for a protein-ligand complex.

    ``forward`` is the ordinary differentiable sequence-conditioned teacher-forced graph. ``sample`` and
    the four probability operations expose the released tensor protocols through the shared MPNN carriers,
    and ``design`` returns self-contained canonical candidates. ``coordinate_noise``,
    ``use_side_chain_context``, and the stochastic controls are per-operation arguments rather than
    persistent configuration.

    Args:
        config: Model configuration; a default
            [`LigandMpnnConfig`][ofoldx.models.ligandmpnn.config.LigandMpnnConfig] is used when ``None``.
    """

    config_class = LigandMpnnConfig

    def __init__(self, config: LigandMpnnConfig | None = None) -> None:
        super().__init__()
        self.config = LigandMpnnConfig() if config is None else config
        hidden_size = self.config.hidden_size
        # Cache the one live width as a plain int: the encoder hot path reads it under
        # torch.compile(..., fullgraph=True), where chanfig's Config attribute access does not trace.
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(self.config.dropout)
        self.feature_encoder = LigandMpnnFeatures(hidden_size, self.config.num_neighbors)
        self.edge_embedding = nn.Linear(hidden_size, hidden_size)
        self.token_embedding = nn.Embedding(_VOCABULARY, hidden_size)
        self.atom_context_embedding = nn.Linear(hidden_size, hidden_size)
        self.residue_context_input = nn.Linear(hidden_size, hidden_size)
        self.atom_node_embedding = nn.Linear(hidden_size, hidden_size)
        self.atom_edge_embedding = nn.Linear(hidden_size, hidden_size)
        self.context_output = nn.Linear(hidden_size, hidden_size, bias=False)
        self.context_norm = nn.LayerNorm(hidden_size)
        self.context_layers = nn.ModuleList(
            MpnnDecoderLayer(hidden_size, hidden_size * 2, dropout=self.config.dropout) for _ in range(2)
        )
        self.atom_context_layers = nn.ModuleList(
            MpnnDecoderLayer(hidden_size, hidden_size, dropout=self.config.dropout) for _ in range(2)
        )
        self.encoder_layers = nn.ModuleList(
            MpnnEncoderLayer(hidden_size, hidden_size * 2, dropout=self.config.dropout)
            for _ in range(self.config.num_encoder_layers)
        )
        self.decoder_layers = nn.ModuleList(
            MpnnDecoderLayer(hidden_size, hidden_size * 3, dropout=self.config.dropout)
            for _ in range(self.config.num_decoder_layers)
        )
        self.output_projection = nn.Linear(hidden_size, _VOCABULARY)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        r"""Apply the released Xavier-uniform initialization to every matrix parameter."""

        for parameter in self.parameters():
            if parameter.ndim > 1:
                nn.init.xavier_uniform_(parameter)

    # -- stochastic and validation controls ------------------------------------------------------

    @staticmethod
    def _validate_coordinate_noise(coordinate_noise: float) -> float:
        r"""Coerce and validate the inference-time coordinate perturbation."""

        coordinate_noise = float(coordinate_noise)
        if not math.isfinite(coordinate_noise) or coordinate_noise < 0:
            raise ValueError("coordinate_noise must be a finite nonnegative number")
        return coordinate_noise

    @staticmethod
    def _validate_num_decoding_orders(num_decoding_orders: int) -> int:
        r"""Validate the independent decoding-order evaluation count."""

        if type(num_decoding_orders) is not int or num_decoding_orders <= 0:
            raise ValueError("num_decoding_orders must be a positive integer")
        return num_decoding_orders

    @staticmethod
    def _require_single_target(input: LigandMpnnInput) -> None:
        r"""Reject an execution-batch axis on one-target biological operations."""

        if input.backbone_coordinates.shape[0] != 1:
            raise ValueError("this LigandMpnn operation returns one target and therefore accepts one input")

    @staticmethod
    def _repeat_input(input: LigandMpnnInput, count: int) -> LigandMpnnInput:
        r"""Repeat one biological input into ``count`` aligned evaluation rows."""

        def repeat(value: Tensor | None) -> Tensor | None:
            return None if value is None else value.repeat_interleave(count, dim=0)

        return LigandMpnnInput(
            backbone_coordinates=repeat(input.backbone_coordinates),
            sequence_tokens=repeat(input.sequence_tokens),
            valid_mask=repeat(input.valid_mask),
            residue_indices=repeat(input.residue_indices),
            chain_indices=repeat(input.chain_indices),
            context_coordinates=repeat(input.context_coordinates),
            context_atomic_numbers=repeat(input.context_atomic_numbers),
            context_mask=repeat(input.context_mask),
            residue_context_coordinates=repeat(input.residue_context_coordinates),
            residue_context_mask=repeat(input.residue_context_mask),
            design_mask=repeat(input.design_mask),
            ligand_interface_mask=repeat(input.ligand_interface_mask),
            structure=input.structure,
            component_references=input.component_references,
            tied_group_ids=repeat(input.tied_group_ids),
            tied_weights=repeat(input.tied_weights),
            allowed_token_mask=repeat(input.allowed_token_mask),
            logit_bias=repeat(input.logit_bias),
        )

    def _probability_result(
        self,
        input: LigandMpnnInput,
        logits: Tensor,
        decoding_ranks: Tensor,
    ) -> MpnnProbabilityResult:
        r"""Wrap aligned per-evaluation logits with the one target's canonical identity."""

        return MpnnProbabilityResult(
            target=input.structure,
            component_references=input.component_references,
            sequence_tokens=input.sequence_tokens[0],
            valid_mask=input.valid_mask[0],
            design_mask=input.design_mask[0],
            logits=logits,
            decoding_ranks=decoding_ranks,
        )

    # -- decode-order generation -----------------------------------------------------------------

    @staticmethod
    def _draw_decode_orders(
        valid_mask: Tensor,
        design_mask: Tensor,
        generator: torch.Generator | None,
    ) -> Tensor:
        r"""Draw the released fixed-first decoding orders directly from aligned masks.

        Fixed rows receive a near-zero weight and therefore decode before every designed row; each row
        of the returned ``[B, L]`` permutation is an independent order.
        """

        chain_mask = valid_mask.to(dtype=torch.float32) * design_mask.to(dtype=torch.float32)
        randn = torch.randn(chain_mask.shape, device=chain_mask.device, generator=generator)
        return torch.argsort((chain_mask + 0.0001) * randn.abs(), dim=-1)

    def _generate_ranks(self, input: LigandMpnnInput, generator: torch.Generator | None) -> Tensor:
        r"""Draw fixed-first causal ranks, collapsing every tied group to one consecutive unit."""

        decode_order = self._draw_decode_orders(input.valid_mask, input.design_mask, generator)
        tied_groups = tied_groups_from_ids(input.tied_group_ids)
        if tied_groups:
            collapsed = _collapse_tied_order(decode_order[0], tied_groups)
            decode_order = collapsed.unsqueeze(0).repeat(decode_order.shape[0], 1)
        return _decoding_order_to_ranks(decode_order)

    # -- shared graph passes ---------------------------------------------------------------------

    def _encode(
        self,
        input: LigandMpnnInput,
        *,
        coordinate_noise: float,
        use_side_chain_context: bool,
        generator: torch.Generator | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        r"""Run the coordinate/context encoder shared by every official operation.

        Coordinate noise perturbs the backbone coordinates and the already selected nonprotein context
        coordinates before residue-neighbor and atom-context features; fixed side-chain coordinates are
        never perturbed.
        """

        backbone_coordinates = input.backbone_coordinates
        context_coordinates = input.context_coordinates
        if coordinate_noise:
            backbone_coordinates = backbone_coordinates + coordinate_noise * torch.randn(
                backbone_coordinates.shape,
                device=backbone_coordinates.device,
                dtype=backbone_coordinates.dtype,
                generator=generator,
            )
            context_coordinates = context_coordinates + coordinate_noise * torch.randn(
                context_coordinates.shape,
                device=context_coordinates.device,
                dtype=context_coordinates.dtype,
                generator=generator,
            )
        mask = input.valid_mask.to(dtype=backbone_coordinates.dtype)
        atom_features, edges, neighbor_indices, atom_nodes, atom_edges, atom_mask = self.feature_encoder(
            backbone_coordinates,
            mask,
            input.residue_indices,
            input.chain_indices,
            context_coordinates,
            input.context_atomic_numbers,
            input.context_mask.to(dtype=backbone_coordinates.dtype),
            input.residue_context_coordinates,
            input.residue_context_mask.to(dtype=backbone_coordinates.dtype),
            input.design_mask.to(dtype=backbone_coordinates.dtype),
            use_side_chain_context=use_side_chain_context,
        )
        node_states = edges.new_zeros(edges.shape[0], edges.shape[1], self.hidden_size)
        edge_states = self.edge_embedding(edges)
        atom_context = self.atom_context_embedding(atom_features)
        attend_mask = gather_nodes(mask.unsqueeze(-1), neighbor_indices).squeeze(-1)
        attend_mask = mask.unsqueeze(-1) * attend_mask
        for layer in self.encoder_layers:
            node_states, edge_states = layer(node_states, edge_states, neighbor_indices, mask, attend_mask)
        residue_context = self.residue_context_input(node_states)
        atom_pair_mask = atom_mask[:, :, :, None] * atom_mask[:, :, None, :]
        atom_nodes = self.atom_node_embedding(atom_nodes)
        atom_edges = self.atom_edge_embedding(atom_edges)
        for layer_index, layer in enumerate(self.context_layers):
            atom_nodes = self.atom_context_layers[layer_index](atom_nodes, atom_edges, atom_mask, atom_pair_mask)
            residue_context = layer(residue_context, torch.cat((atom_context, atom_nodes), dim=-1), mask, atom_mask)
        node_states = node_states + self.context_norm(self.dropout(self.context_output(residue_context)))
        return node_states, edge_states, neighbor_indices

    def _decode_logits(
        self,
        node_states: Tensor,
        edge_states: Tensor,
        neighbor_indices: Tensor,
        sequence_tokens: Tensor,
        valid_mask: Tensor,
        decoding_ranks: Tensor,
    ) -> Tensor:
        r"""Return raw teacher-forced sequence-conditioned logits for one exact causal ranking."""

        mask = valid_mask.to(dtype=node_states.dtype)
        token_states = self.token_embedding(sequence_tokens)
        sequence_edges = cat_neighbors_nodes(token_states, edge_states, neighbor_indices)
        encoder_edges = cat_neighbors_nodes(torch.zeros_like(token_states), edge_states, neighbor_indices)
        encoder_context = cat_neighbors_nodes(node_states, encoder_edges, neighbor_indices)
        backward_mask, forward_mask = _autoregressive_masks(decoding_ranks, neighbor_indices, mask)
        forward_context = forward_mask * encoder_context
        for layer in self.decoder_layers:
            decoder_context = cat_neighbors_nodes(node_states, sequence_edges, neighbor_indices)
            node_states = layer(node_states, backward_mask * decoder_context + forward_context, mask)
        return self.output_projection(node_states)

    def _decode_sequence_free_logits(
        self,
        node_states: Tensor,
        edge_states: Tensor,
        neighbor_indices: Tensor,
        valid_mask: Tensor,
        decoding_ranks: Tensor,
    ) -> Tensor:
        r"""Return raw order-dependent sequence-free logits under one causal ranking."""

        mask = valid_mask.to(dtype=node_states.dtype)
        token_states = torch.zeros_like(node_states)
        encoder_edges = cat_neighbors_nodes(token_states, edge_states, neighbor_indices)
        encoder_context = cat_neighbors_nodes(node_states, encoder_edges, neighbor_indices)
        _, forward_mask = _autoregressive_masks(decoding_ranks, neighbor_indices, mask)
        forward_context = forward_mask * encoder_context
        states = node_states
        for layer in self.decoder_layers:
            states = layer(states, forward_context, mask)
        return self.output_projection(states)

    # -- public operations -----------------------------------------------------------------------

    def encode(
        self,
        input: LigandMpnnInput,
        *,
        coordinate_noise: float = 0.0,
        use_side_chain_context: bool = False,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        r"""Return the shared node states, edge states, and neighbor indices."""

        coordinate_noise = self._validate_coordinate_noise(coordinate_noise)
        return self._encode(
            input,
            coordinate_noise=coordinate_noise,
            use_side_chain_context=use_side_chain_context,
            generator=generator,
        )

    def forward(
        self,
        input: LigandMpnnInput,
        *,
        decoding_ranks: Tensor | None = None,
        coordinate_noise: float = 0.0,
        use_side_chain_context: bool = False,
        generator: torch.Generator | None = None,
    ) -> MpnnOutput:
        r"""Return the single teacher-forced differentiable sequence-conditioned pass."""

        coordinate_noise = self._validate_coordinate_noise(coordinate_noise)
        node_states, edge_states, neighbor_indices = self._encode(
            input,
            coordinate_noise=coordinate_noise,
            use_side_chain_context=use_side_chain_context,
            generator=generator,
        )
        if decoding_ranks is None:
            decoding_ranks = _decoding_order_to_ranks(
                self._draw_decode_orders(input.valid_mask, input.design_mask, generator)
            )
        logits = self._decode_logits(
            node_states,
            edge_states,
            neighbor_indices,
            input.sequence_tokens,
            input.valid_mask,
            decoding_ranks,
        )
        return MpnnOutput(logits=logits, decoding_ranks=decoding_ranks)

    def autoregressive_probabilities(
        self,
        input: LigandMpnnInput,
        *,
        num_decoding_orders: int = 1,
        decoding_ranks: Tensor | None = None,
        coordinate_noise: float = 0.0,
        use_side_chain_context: bool = False,
        generator: torch.Generator | None = None,
    ) -> MpnnProbabilityResult:
        r"""Return native-sequence-conditioned autoregressive probabilities over decoding orders."""

        self._require_single_target(input)
        num_decoding_orders = self._validate_num_decoding_orders(num_decoding_orders)
        coordinate_noise = self._validate_coordinate_noise(coordinate_noise)
        repeated = self._repeat_input(input, num_decoding_orders)
        node_states, edge_states, neighbor_indices = self._encode(
            repeated,
            coordinate_noise=coordinate_noise,
            use_side_chain_context=use_side_chain_context,
            generator=generator,
        )
        ranks = decoding_ranks if decoding_ranks is not None else self._generate_ranks(repeated, generator)
        logits = self._decode_logits(
            node_states,
            edge_states,
            neighbor_indices,
            repeated.sequence_tokens,
            repeated.valid_mask,
            ranks,
        )
        return self._probability_result(input, logits, ranks)

    def sequence_free_autoregressive_probabilities(
        self,
        input: LigandMpnnInput,
        *,
        num_decoding_orders: int = 1,
        decoding_ranks: Tensor | None = None,
        coordinate_noise: float = 0.0,
        use_side_chain_context: bool = False,
        generator: torch.Generator | None = None,
    ) -> MpnnProbabilityResult:
        r"""Return order-dependent sequence-free autoregressive probabilities over decoding orders."""

        self._require_single_target(input)
        num_decoding_orders = self._validate_num_decoding_orders(num_decoding_orders)
        coordinate_noise = self._validate_coordinate_noise(coordinate_noise)
        repeated = self._repeat_input(input, num_decoding_orders)
        node_states, edge_states, neighbor_indices = self._encode(
            repeated,
            coordinate_noise=coordinate_noise,
            use_side_chain_context=use_side_chain_context,
            generator=generator,
        )
        ranks = decoding_ranks if decoding_ranks is not None else self._generate_ranks(repeated, generator)
        logits = self._decode_sequence_free_logits(
            node_states,
            edge_states,
            neighbor_indices,
            repeated.valid_mask,
            ranks,
        )
        return self._probability_result(input, logits, ranks)

    def _single_residue_probabilities(
        self,
        input: LigandMpnnInput,
        *,
        num_decoding_orders: int,
        base_decoding_ranks: Tensor | None,
        coordinate_noise: float,
        use_side_chain_context: bool,
        generator: torch.Generator | None,
        move_last: bool,
    ) -> MpnnProbabilityResult:
        r"""Evaluate every target moved exactly last or first under one base decoding order."""

        self._require_single_target(input)
        num_decoding_orders = self._validate_num_decoding_orders(num_decoding_orders)
        coordinate_noise = self._validate_coordinate_noise(coordinate_noise)
        repeated = self._repeat_input(input, num_decoding_orders)
        device = repeated.backbone_coordinates.device
        node_states, edge_states, neighbor_indices = self._encode(
            repeated,
            coordinate_noise=coordinate_noise,
            use_side_chain_context=use_side_chain_context,
            generator=generator,
        )
        length = repeated.sequence_tokens.shape[1]
        if base_decoding_ranks is None:
            base_priorities = torch.randn(num_decoding_orders, length, device=device, generator=generator).abs()
        else:
            if base_decoding_ranks.shape != (num_decoding_orders, length):
                raise ValueError("base_decoding_ranks has the wrong exact shape")
            if base_decoding_ranks.dtype != torch.long:
                raise TypeError("base_decoding_ranks must have dtype torch.long")
            sorted_ranks, sorted_indices = torch.sort(base_decoding_ranks, dim=-1)
            rank_starts = torch.cat(
                (
                    torch.ones_like(sorted_ranks[:, :1], dtype=torch.bool),
                    sorted_ranks[:, 1:] != sorted_ranks[:, :-1],
                ),
                dim=-1,
            )
            base_ranks = torch.empty_like(base_decoding_ranks)
            base_ranks.scatter_(1, sorted_indices, rank_starts.cumsum(dim=-1) - 1)
            last_ranks = base_ranks.amax(dim=-1) + 1
        logits = node_states.new_zeros(num_decoding_orders, length, _VOCABULARY)
        effective_ranks = torch.empty(num_decoding_orders, length, length, dtype=torch.long, device=device)
        for target in range(length):
            if base_decoding_ranks is None:
                order_mask = torch.full((length,), 0.0 if move_last else 1.0, device=device)
                order_mask[target] = 1.0 if move_last else 0.0
                decode_order = torch.argsort((order_mask + 0.0001) * base_priorities, dim=-1)
                ranks = _decoding_order_to_ranks(decode_order)
            else:
                ranks = base_ranks.clone()
                ranks[:, target] = last_ranks if move_last else -1
            logits[:, target] = self._decode_logits(
                node_states,
                edge_states,
                neighbor_indices,
                repeated.sequence_tokens,
                repeated.valid_mask,
                ranks,
            )[:, target]
            effective_ranks[:, target] = ranks
        return self._probability_result(input, logits, effective_ranks)

    def conditional_probabilities(
        self,
        input: LigandMpnnInput,
        *,
        num_decoding_orders: int = 1,
        base_decoding_ranks: Tensor | None = None,
        coordinate_noise: float = 0.0,
        use_side_chain_context: bool = False,
        generator: torch.Generator | None = None,
    ) -> MpnnProbabilityResult:
        r"""Return target-last conditional probabilities for every row."""

        return self._single_residue_probabilities(
            input,
            num_decoding_orders=num_decoding_orders,
            base_decoding_ranks=base_decoding_ranks,
            coordinate_noise=coordinate_noise,
            use_side_chain_context=use_side_chain_context,
            generator=generator,
            move_last=True,
        )

    def unconditional_probabilities(
        self,
        input: LigandMpnnInput,
        *,
        num_decoding_orders: int = 1,
        base_decoding_ranks: Tensor | None = None,
        coordinate_noise: float = 0.0,
        use_side_chain_context: bool = False,
        generator: torch.Generator | None = None,
    ) -> MpnnProbabilityResult:
        r"""Return target-first unconditional probabilities for every row."""

        return self._single_residue_probabilities(
            input,
            num_decoding_orders=num_decoding_orders,
            base_decoding_ranks=base_decoding_ranks,
            coordinate_noise=coordinate_noise,
            use_side_chain_context=use_side_chain_context,
            generator=generator,
            move_last=False,
        )

    # -- sampling --------------------------------------------------------------------------------

    @staticmethod
    def _sampling_bias(input: LigandMpnnInput) -> Tensor:
        r"""Assemble the aligned generatable logit bias, folding hard support in as large negatives."""

        if input.logit_bias is None:
            bias = torch.zeros(
                input.sequence_tokens.shape[0],
                input.sequence_tokens.shape[1],
                _VOCABULARY,
                device=input.sequence_tokens.device,
                dtype=input.backbone_coordinates.dtype,
            )
        else:
            bias = input.logit_bias
        if input.allowed_token_mask is not None:
            bias = bias + torch.where(
                input.allowed_token_mask,
                torch.zeros((), dtype=bias.dtype, device=bias.device),
                torch.full((), -1e8, dtype=bias.dtype, device=bias.device),
            )
        return bias

    @torch.inference_mode()
    def sample(
        self,
        input: LigandMpnnInput,
        *,
        num_samples: int = 1,
        temperature: float = 0.1,
        coordinate_noise: float = 0.0,
        use_side_chain_context: bool = False,
        decoding_ranks: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> MpnnSamplingOutput:
        r"""Run the complete released autoregressive tensor sampler."""

        self._require_single_target(input)
        if type(num_samples) is not int or num_samples <= 0:
            raise ValueError("num_samples must be a positive integer")
        temperature = float(temperature)
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be a finite positive number")
        coordinate_noise = self._validate_coordinate_noise(coordinate_noise)

        repeated = self._repeat_input(input, num_samples)
        node_states, edge_states, neighbor_indices = self._encode(
            repeated,
            coordinate_noise=coordinate_noise,
            use_side_chain_context=use_side_chain_context,
            generator=generator,
        )
        valid_mask = repeated.valid_mask.to(dtype=node_states.dtype)
        chain_mask = valid_mask * repeated.design_mask.to(dtype=node_states.dtype)
        bias = self._sampling_bias(repeated)
        if decoding_ranks is None:
            decode_order = self._draw_decode_orders(repeated.valid_mask, repeated.design_mask, generator)
        else:
            decode_order = torch.argsort(decoding_ranks, dim=-1)

        tied_groups = tied_groups_from_ids(input.tied_group_ids)
        if not tied_groups:
            result = self._sample_independent(
                node_states,
                edge_states,
                neighbor_indices,
                repeated.sequence_tokens,
                valid_mask,
                chain_mask,
                bias,
                temperature,
                decode_order,
                generator,
            )
        else:
            collapsed = _collapse_tied_order(decode_order[0], tied_groups)
            decode_order = collapsed.unsqueeze(0).repeat(num_samples, 1)
            if repeated.tied_weights is None:
                weights = torch.ones_like(chain_mask)
            else:
                weights = repeated.tied_weights.to(dtype=node_states.dtype)
            result = self._sample_symmetric(
                node_states,
                edge_states,
                neighbor_indices,
                repeated.sequence_tokens,
                valid_mask,
                chain_mask,
                bias,
                temperature,
                decode_order,
                tied_groups,
                weights,
                generator,
            )
        return MpnnSamplingOutput(
            sequences=result["S"],
            log_probabilities=result["log_probs"],
            sampling_probabilities=result["sampling_probs"],
            decoding_ranks=_decoding_order_to_ranks(result["decoding_order"]),
        )

    def _sample_independent(
        self,
        node_states: Tensor,
        edge_states: Tensor,
        neighbor_indices: Tensor,
        sequence_true: Tensor,
        valid_mask: Tensor,
        chain_mask: Tensor,
        bias: Tensor,
        temperature: float,
        decode_order: Tensor,
        generator: torch.Generator | None,
    ) -> dict[str, Tensor]:
        r"""Sample tokens position-by-position with no tied constraints."""

        num_samples, length = sequence_true.shape
        device = sequence_true.device
        sampled = _GENERATABLE * torch.ones(num_samples, length, dtype=torch.long, device=device)
        token_states = torch.zeros_like(node_states)
        layer_states = [node_states] + [torch.zeros_like(node_states) for _ in self.decoder_layers]
        probabilities = node_states.new_zeros(num_samples, length, _GENERATABLE)
        log_probs_out = node_states.new_zeros(num_samples, length, _VOCABULARY)
        logits_out = node_states.new_zeros(num_samples, length, _VOCABULARY)
        backward_mask, forward_mask = _autoregressive_masks(
            _decoding_order_to_ranks(decode_order),
            neighbor_indices,
            valid_mask,
        )
        encoder_edges = cat_neighbors_nodes(torch.zeros_like(token_states), edge_states, neighbor_indices)
        encoder_context = cat_neighbors_nodes(node_states, encoder_edges, neighbor_indices)
        forward_context = forward_mask * encoder_context

        for step in range(length):
            residue = decode_order[:, step]
            chain_mask_t = torch.gather(chain_mask, 1, residue[:, None])[:, 0]
            mask_t = torch.gather(valid_mask, 1, residue[:, None])
            bias_t = torch.gather(bias, 1, residue[:, None, None].repeat(1, 1, bias.shape[-1]))[:, 0, :]
            neighbor_idx_t = torch.gather(
                neighbor_indices,
                1,
                residue[:, None, None].repeat(1, 1, neighbor_indices.shape[-1]),
            )
            edge_t = torch.gather(
                edge_states,
                1,
                residue[:, None, None, None].repeat(1, 1, edge_states.shape[-2], edge_states.shape[-1]),
            )
            edge_token_t = cat_neighbors_nodes(token_states, edge_t, neighbor_idx_t)
            forward_t = torch.gather(
                forward_context,
                1,
                residue[:, None, None, None].repeat(1, 1, forward_context.shape[-2], forward_context.shape[-1]),
            )
            backward_t = torch.gather(
                backward_mask,
                1,
                residue[:, None, None, None].repeat(1, 1, backward_mask.shape[-2], backward_mask.shape[-1]),
            )
            for layer_index, layer in enumerate(self.decoder_layers):
                decoder_context = cat_neighbors_nodes(layer_states[layer_index], edge_token_t, neighbor_idx_t)
                node_t = torch.gather(
                    layer_states[layer_index],
                    1,
                    residue[:, None, None].repeat(1, 1, layer_states[layer_index].shape[-1]),
                )
                updated = layer(node_t, backward_t * decoder_context + forward_t, mask_t)
                layer_states[layer_index + 1].scatter_(
                    1,
                    residue[:, None, None].repeat(1, 1, node_states.shape[-1]),
                    updated,
                )

            final_t = torch.gather(
                layer_states[-1],
                1,
                residue[:, None, None].repeat(1, 1, layer_states[-1].shape[-1]),
            )[:, 0]
            logits = self.output_projection(final_t)
            log_probs = F.log_softmax(logits, dim=-1)
            sample_probs = F.softmax((logits + bias_t) / temperature, dim=-1)
            sample_probs = sample_probs[:, :_GENERATABLE]
            sample_probs = sample_probs / sample_probs.sum(dim=-1, keepdim=True)
            sampled_t = torch.multinomial(sample_probs, 1, generator=generator)[:, 0]
            probabilities.scatter_(
                1,
                residue[:, None, None].repeat(1, 1, _GENERATABLE),
                chain_mask_t[:, None, None] * sample_probs[:, None, :],
            )
            log_probs_out.scatter_(
                1,
                residue[:, None, None].repeat(1, 1, _VOCABULARY),
                chain_mask_t[:, None, None] * log_probs[:, None, :],
            )
            logits_out.scatter_(
                1,
                residue[:, None, None].repeat(1, 1, _VOCABULARY),
                chain_mask_t[:, None, None] * logits[:, None, :],
            )
            true_t = torch.gather(sequence_true, 1, residue[:, None])[:, 0]
            sampled_t = (sampled_t * chain_mask_t + true_t * (1.0 - chain_mask_t)).long()
            token_states.scatter_(
                1,
                residue[:, None, None].repeat(1, 1, token_states.shape[-1]),
                self.token_embedding(sampled_t)[:, None, :],
            )
            sampled.scatter_(1, residue[:, None], sampled_t[:, None])

        return {
            "S": sampled,
            "sampling_probs": probabilities,
            "logits": logits_out,
            "log_probs": log_probs_out,
            "decoding_order": decode_order,
        }

    def _sample_symmetric(
        self,
        node_states: Tensor,
        edge_states: Tensor,
        neighbor_indices: Tensor,
        sequence_true: Tensor,
        valid_mask: Tensor,
        chain_mask: Tensor,
        bias: Tensor,
        temperature: float,
        decode_order: Tensor,
        tied_groups: tuple[tuple[int, ...], ...],
        tied_weights: Tensor,
        generator: torch.Generator | None,
    ) -> dict[str, Tensor]:
        r"""Sample tokens with each tied group drawn as one weighted decision.

        Neural logits combine as ``sum(tied_weight[i] * logits[i])`` with a default coefficient of one;
        the aligned bias applies once at the shared group value and hard support is folded into that
        bias. One group distribution is copied to every member, and every member receives the same token.
        """

        num_samples, length = sequence_true.shape
        device = sequence_true.device
        sampled = _GENERATABLE * torch.ones(num_samples, length, dtype=torch.long, device=device)
        token_states = torch.zeros_like(node_states)
        layer_states = [node_states] + [torch.zeros_like(node_states) for _ in self.decoder_layers]
        probabilities = node_states.new_zeros(num_samples, length, _GENERATABLE)
        log_probs_out = node_states.new_zeros(num_samples, length, _VOCABULARY)
        logits_out = node_states.new_zeros(num_samples, length, _VOCABULARY)
        backward_mask, forward_mask = _autoregressive_masks(
            _decoding_order_to_ranks(decode_order),
            neighbor_indices,
            valid_mask,
        )
        encoder_edges = cat_neighbors_nodes(torch.zeros_like(token_states), edge_states, neighbor_indices)
        encoder_context = cat_neighbors_nodes(node_states, encoder_edges, neighbor_indices)
        forward_context = forward_mask * encoder_context

        ordered_groups = _decode_groups(decode_order[0], [list(group) for group in tied_groups])
        for residue_group in ordered_groups:
            total_logits = node_states.new_zeros(num_samples, _VOCABULARY)
            bias_group = bias[:, residue_group[-1]]
            for residue in residue_group:
                chain_mask_t = chain_mask[:, residue]
                mask_t = valid_mask[:, residue]
                edge_t = edge_states[:, residue : residue + 1]
                neighbor_idx_t = neighbor_indices[:, residue : residue + 1]
                edge_token_t = cat_neighbors_nodes(token_states, edge_t, neighbor_idx_t)
                forward_t = forward_context[:, residue : residue + 1]
                for layer_index, layer in enumerate(self.decoder_layers):
                    decoder_context = cat_neighbors_nodes(layer_states[layer_index], edge_token_t, neighbor_idx_t)
                    node_t = layer_states[layer_index][:, residue : residue + 1]
                    context_t = backward_mask[:, residue : residue + 1] * decoder_context + forward_t
                    layer_states[layer_index + 1][:, residue : residue + 1] = layer(node_t, context_t, mask_t[:, None])
                logits = self.output_projection(layer_states[-1][:, residue])
                log_probs = F.log_softmax(logits, dim=-1)
                log_probs_out[:, residue] = chain_mask_t[:, None] * log_probs
                logits_out[:, residue] = chain_mask_t[:, None] * logits
                total_logits = total_logits + tied_weights[:, residue, None] * logits

            sample_probs = F.softmax((total_logits + bias_group) / temperature, dim=-1)
            sample_probs = sample_probs[:, :_GENERATABLE]
            sample_probs = sample_probs / sample_probs.sum(dim=-1, keepdim=True)
            sampled_t = torch.multinomial(sample_probs, 1, generator=generator)[:, 0]
            for residue in residue_group:
                chain_mask_t = chain_mask[:, residue]
                probabilities[:, residue] = chain_mask_t[:, None] * sample_probs
                true_t = sequence_true[:, residue]
                sampled_t = (sampled_t * chain_mask_t + true_t * (1.0 - chain_mask_t)).long()
                token_states[:, residue] = self.token_embedding(sampled_t)
                sampled[:, residue] = sampled_t

        return {
            "S": sampled,
            "sampling_probs": probabilities,
            "logits": logits_out,
            "log_probs": log_probs_out,
            "decoding_order": decode_order,
        }

    # -- design ----------------------------------------------------------------------------------

    @torch.inference_mode()
    def design(
        self,
        input: LigandMpnnInput,
        *,
        num_candidates: int = 1,
        temperature: float = 0.1,
        coordinate_noise: float = 0.0,
        use_side_chain_context: bool = False,
        generator: torch.Generator | None = None,
    ) -> DesignResult:
        r"""Autoregressively design complete canonical candidates and score each in-pass."""

        self._require_single_target(input)
        if type(num_candidates) is not int or num_candidates <= 0:
            raise ValueError("num_candidates must be a positive integer")
        temperature = float(temperature)
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be a finite positive number")
        coordinate_noise = self._validate_coordinate_noise(coordinate_noise)
        sampled = self.sample(
            input,
            num_samples=num_candidates,
            temperature=temperature,
            coordinate_noise=coordinate_noise,
            use_side_chain_context=use_side_chain_context,
            generator=generator,
        )
        losses = -torch.gather(
            sampled.log_probabilities,
            -1,
            sampled.sequences.unsqueeze(-1),
        ).squeeze(-1)
        designed = (input.valid_mask & input.design_mask)[0]
        interface = designed & input.ligand_interface_mask[0].bool()
        designed_row = designed.to(dtype=losses.dtype)
        interface_row = interface.to(dtype=losses.dtype)
        designed_nll = (losses * designed_row).sum(dim=-1) / designed_row.sum(dim=-1)
        interface_nll = (losses * interface_row).sum(dim=-1) / interface_row.sum(dim=-1)
        designed_confidence = torch.exp(-designed_nll)
        interface_confidence = torch.exp(-interface_nll)
        recovery = (
            (sampled.sequences == input.sequence_tokens.repeat(num_candidates, 1)).to(dtype=losses.dtype) * designed_row
        ).sum(dim=-1) / designed_row.sum(dim=-1)

        references = input.component_references
        designed_host = designed.detach().cpu().tolist()
        interface_host = interface.detach().cpu().tolist()
        designed_scope = frozenset(
            reference for reference, selected in zip(references, designed_host) if reference is not None and selected
        )
        interface_scope = frozenset(
            reference for reference, selected in zip(references, interface_host) if reference is not None and selected
        )
        host_sequences = sampled.sequences.detach().cpu().tolist()
        designed_confidence_host = designed_confidence.detach().cpu().tolist()
        interface_confidence_host = interface_confidence.detach().cpu().tolist()
        recovery_host = recovery.detach().cpu().tolist()

        candidates = tuple(
            DesignCandidate(
                system=_realize_designed_system(
                    input.structure.system,
                    references,
                    host_sequences[index],
                    designed_host,
                ),
                scores={
                    "designed_sequence_confidence": Score(
                        float(designed_confidence_host[index]),
                        preferred="higher",
                        scope=designed_scope,
                    ),
                    "ligand_interface_confidence": Score(
                        float(interface_confidence_host[index]),
                        preferred="higher",
                        scope=interface_scope,
                    ),
                    "sequence_recovery": Score(
                        float(recovery_host[index]),
                        preferred="higher",
                        scope=designed_scope,
                    ),
                },
            )
            for index in range(num_candidates)
        )
        return DesignResult(candidates=candidates)


# ---------------------------------------------------------------------------
# Family-owned tensor boundary (folded from contract.py)
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class LigandMpnnInput:
    r"""Tensors and canonical identities consumed by LigandMPNN operations.

    The canonical Processor operation owns one Structure, so the tensor batch is
    one. ``backbone_coordinates`` is ``[1, L, 4, 3]`` in N/CA/C/O order. The
    ``context_*`` tensors are ``[1, L, A, ...]`` for the padded nearest-context
    extent ``A``. The ``residue_context_*`` tensors are the released ``[1, L, 32]``
    side-chain slots at atom37 indices ``5:37`` (CG through OXT, deliberately
    excluding CB), which the Model consumes directly instead of a redundant full
    atom37 array. ``tied_group_ids`` is aligned ``long [1, L]`` dense causal-unit
    identity in first-row order; every untied row owns its identity. ``tied_weights``
    is aligned ``[1, L]`` when present, and ``None`` selects coefficient one.
    Optional evidence remains ``None`` on the common path rather than allocating
    neutral per-position masks or bias tensors.
    """

    backbone_coordinates: Tensor
    sequence_tokens: Tensor
    valid_mask: Tensor
    residue_indices: Tensor
    chain_indices: Tensor
    context_coordinates: Tensor
    context_atomic_numbers: Tensor
    context_mask: Tensor
    residue_context_coordinates: Tensor
    residue_context_mask: Tensor
    design_mask: Tensor
    ligand_interface_mask: Tensor
    structure: Structure
    component_references: tuple[ComponentReference, ...]
    tied_group_ids: Tensor
    tied_weights: Tensor | None = None
    allowed_token_mask: Tensor | None = None
    logit_bias: Tensor | None = None


def tied_groups_from_ids(
    tied_group_ids: Tensor,
) -> tuple[tuple[int, ...], ...]:
    r"""Reconstruct tied member groups from one row of dense causal-unit IDs.

    The sampling and probability hot paths consume ``tied_group_ids`` directly; this
    helper serves non-compiled orchestration that still reasons about explicit
    member groups. Every returned group holds more than one row and lists members in
    ascending row order, matching the dense first-row identity assignment; untied
    singleton units are omitted. ``tied_group_ids`` is the aligned ``[1, L]`` tensor,
    whose single batch row is the only Processor-owned structure.
    """

    members_by_id: dict[int, list[int]] = {}
    for index, group_id in enumerate(tied_group_ids[0].tolist()):
        members_by_id.setdefault(group_id, []).append(index)
    return tuple(tuple(members) for members in members_by_id.values() if len(members) > 1)
