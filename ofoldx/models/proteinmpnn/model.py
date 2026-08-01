r"""
ProteinMPNN fixed-backbone sequence model.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn.functional as F
from tensordict import tensorclass
from torch import Tensor, nn

from ...data.components import get_component
from ...data.molecular import MolecularSystem, replace_components
from ...data.structure import Structure
from ...model import Model, _TensorCollection
from ...modules.backend import LayerNorm
from ...modules.mpnn import (
    AMINO_ACID_COMPONENT_IDS,
    MpnnDecoderBlock,
    MpnnEncoderBlock,
    MpnnProteinFeatures,
    MpnnRelativePositionEncoding,
    autoregressive_masks,
    cat_neighbors_nodes,
    gather_edges,
    gather_nodes,
    nearest_neighbors,
    nll_loss,
    radial_basis,
)
from ...results import (
    DesignCandidate,
    DesignResult,
    MpnnProbabilityResult,
    Score,
    ScoreResult,
)
from .config import ProteinMpnnConfig


def _component_references(
    system: MolecularSystem,
    component_index: Tensor,
) -> tuple[tuple[str, int] | None, ...]:
    return tuple(
        None if component < 0 else system.component_reference(component)
        for component in component_index.detach().cpu().tolist()
    )


def _repartition_entities_by_sequence(
    original: MolecularSystem,
    realized: MolecularSystem,
) -> MolecularSystem:
    r"""
    Split copies of each original entity by their realized component-ID sequence.
    """

    symbols = list(realized.symbols)
    symbol_index = {symbol: index for index, symbol in enumerate(symbols)}

    def intern(symbol: str) -> int:
        index = symbol_index.get(symbol)
        if index is None:
            index = len(symbols)
            symbols.append(symbol)
            symbol_index[symbol] = index
        return index

    def component_ids(system: MolecularSystem, instance: int) -> tuple[str, ...]:
        start = int(system.instance_component_ptr[instance])
        stop = int(system.instance_component_ptr[instance + 1])
        return tuple(system.symbols[int(symbol)] for symbol in system.component_symbol[start:stop].tolist())

    instances_by_entity: list[list[int]] = [[] for _ in range(original.entity_symbol.numel())]
    for instance, entity in enumerate(original.instance_entity.tolist()):
        instances_by_entity[entity].append(instance)

    used_ids = {original.symbols[int(symbol)] for symbol in original.entity_symbol.tolist()}
    entity_symbol: list[int] = []
    entity_type: list[int] = []
    entity_molecule_type: list[int] = []
    instance_entity = torch.empty_like(realized.instance_entity)
    for entity, instances in enumerate(instances_by_entity):
        entity_id = original.symbols[int(original.entity_symbol[entity])]
        if not instances:
            entity_symbol.append(intern(entity_id))
            entity_type.append(int(original.entity_type[entity]))
            entity_molecule_type.append(int(original.entity_molecule_type[entity]))
            continue
        original_signature = component_ids(original, instances[0])
        groups: dict[tuple[str, ...], list[int]] = {}
        for instance in instances:
            groups.setdefault(component_ids(realized, instance), []).append(instance)
        primary = original_signature if original_signature in groups else next(iter(groups))
        ordered = (primary, *(signature for signature in groups if signature != primary))
        for partition, signature in enumerate(ordered):
            if partition == 0:
                output_id = entity_id
            else:
                suffix = 2
                output_id = f"{entity_id}_{suffix}"
                while output_id in used_ids:
                    suffix += 1
                    output_id = f"{entity_id}_{suffix}"
                used_ids.add(output_id)
            output_entity = len(entity_symbol)
            entity_symbol.append(intern(output_id))
            entity_type.append(int(original.entity_type[entity]))
            entity_molecule_type.append(int(original.entity_molecule_type[entity]))
            instance_entity[groups[signature]] = output_entity

    return realized.replace(  # type: ignore[attr-defined]
        symbols=tuple(symbols),
        system_entity_ptr=realized.system_entity_ptr.new_tensor([0, len(entity_symbol)]),
        entity_symbol=realized.entity_symbol.new_tensor(entity_symbol),
        entity_type=realized.entity_type.new_tensor(entity_type),
        entity_molecule_type=realized.entity_molecule_type.new_tensor(entity_molecule_type),
        instance_entity=instance_entity,
    )


class ProteinMpnnModel(Model):
    r"""
    Design protein sequences for fixed backbones with ProteinMPNN.

    Accepts [`ProteinMpnnInput`][ofoldx.models.proteinmpnn.model.ProteinMpnnInput] and returns
    [`ProteinMpnnOutput`][ofoldx.models.proteinmpnn.model.ProteinMpnnOutput].

    Args:
        config: Model architecture and inference configuration.

    Examples:
        >>> import torch
        >>> from ofoldx.data import AtomSite, MolecularSystem, PolymerSequence, Structure
        >>> from ofoldx.models.proteinmpnn import ProteinMpnnConfig, ProteinMpnnModel, ProteinMpnnProcessor
        >>> sequence = PolymerSequence.from_one_letter("A")
        >>> system = MolecularSystem.from_entities({"protein": sequence}, {"A": "protein"})
        >>> sites = tuple(
        ...     AtomSite("A", residue, atom)
        ...     for residue in range(1)
        ...     for atom in ("N", "CA", "C", "O")
        ... )
        >>> atom_index = torch.tensor([system.atom_index(site) for site in sites])
        >>> coordinates = torch.zeros(system.atom_symbol.numel(), 3)
        >>> coordinates[atom_index] = torch.tensor(
        ...     [[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0],
        ...      [1.0, 0.5, 0.0], [1.0, 1.0, 0.0]]
        ... )
        >>> coordinate_mask = torch.zeros(system.atom_symbol.numel(), dtype=torch.bool)
        >>> coordinate_mask[atom_index] = True
        >>> structure = Structure(
        ...     system=system,
        ...     coordinates=coordinates,
        ...     coordinate_mask=coordinate_mask,
        ...     occupancy=torch.zeros(system.atom_symbol.numel()),
        ...     occupancy_mask=torch.zeros(system.atom_symbol.numel(), dtype=torch.bool),
        ...     b_factor=torch.zeros(system.atom_symbol.numel()),
        ...     b_factor_mask=torch.zeros(system.atom_symbol.numel(), dtype=torch.bool),
        ...     source_symbols=(),
        ...     residue_id=torch.full((system.component_symbol.numel(),), -1),
        ...     insertion_code=torch.full((system.component_symbol.numel(),), -1),
        ...     standard_component_id=torch.full((system.component_symbol.numel(),), -1),
        ...     batch_size=[],
        ... )
        >>> input = ProteinMpnnProcessor()(structure)
        >>> config = ProteinMpnnConfig(
        ...     {"hidden_size": 8, "num_encoder_blocks": 1, "num_decoder_blocks": 1,
        ...      "num_neighbors": 1, "dropout": 0.0}
        ... )
        >>> model = ProteinMpnnModel(config)
        >>> output = model(input, decoding_ranks=torch.zeros_like(input.sequence_tokens))
        >>> output.logits.shape
        torch.Size([1, 1, 21])
        >>> output.decoding_ranks.tolist()
        [[0]]
    """

    config_class = ProteinMpnnConfig
    config: ProteinMpnnConfig
    _gradient_checkpointing_func: Callable[..., object]

    def __init__(self, config: ProteinMpnnConfig | None = None) -> None:
        config = ProteinMpnnConfig() if config is None else config
        super().__init__(config)
        self.gradient_checkpointing = False
        feature_class = ProteinMpnnCaFeatures if self.config.ca_only else MpnnProteinFeatures
        self.feature_encoder = feature_class(self.config.hidden_size, self.config.num_neighbors)
        self.edge_embedding = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.encoder_blocks = nn.ModuleList(
            MpnnEncoderBlock(
                self.config.hidden_size,
                self.config.hidden_size * 2,
                dropout=self.config.dropout,
            )
            for _ in range(self.config.num_encoder_blocks)
        )
        self.token_embedding = nn.Embedding(len(AMINO_ACID_COMPONENT_IDS) + 1, self.config.hidden_size)
        self.decoder_blocks = nn.ModuleList(
            MpnnDecoderBlock(
                self.config.hidden_size,
                self.config.hidden_size * 3,
                dropout=self.config.dropout,
            )
            for _ in range(self.config.num_decoder_blocks)
        )
        self.output_projection = nn.Linear(self.config.hidden_size, len(AMINO_ACID_COMPONENT_IDS) + 1)
        self.post_init()

    def initialize_parameters(self) -> None:
        r"""
        Apply the Xavier initialization to matrix parameters.
        """

        for parameter in self.parameters():
            if parameter.ndim > 1:
                nn.init.xavier_uniform_(parameter)

    @staticmethod
    def _require_single_target(input: ProteinMpnnInput) -> None:
        r"""
        Reject an execution-batch axis on one-target biological operations.
        """

        if input.backbone_coordinates.shape[0] != 1 or input.structure.system.system_entity_ptr.numel() != 2:
            raise ValueError("this ProteinMpnn operation returns one target and therefore accepts one input")

    @staticmethod
    def _repeat_input(input: ProteinMpnnInput, count: int) -> ProteinMpnnInput:
        r"""
        Repeat one biological input into ``count`` aligned evaluation rows.
        """

        def repeat(value: Tensor | None) -> Tensor | None:
            return None if value is None else value.repeat_interleave(count, dim=0)

        return ProteinMpnnInput(  # type: ignore[call-arg]
            backbone_coordinates=repeat(input.backbone_coordinates),
            sequence_tokens=repeat(input.sequence_tokens),
            residue_mask=repeat(input.residue_mask),
            residue_index=repeat(input.residue_index),
            chain_index=repeat(input.chain_index),
            design_positions=repeat(input.design_positions),
            structure=input.structure,
            component_index=repeat(input.component_index),
            tied_group_ids=repeat(input.tied_group_ids),
            tied_weights=repeat(input.tied_weights),
            allowed_token_mask=repeat(input.allowed_token_mask),
            residue_logit_bias=repeat(input.residue_logit_bias),
            pssm_coefficients=repeat(input.pssm_coefficients),
            pssm_probabilities=repeat(input.pssm_probabilities),
            pssm_log_odds=repeat(input.pssm_log_odds),
            batch_size=[],
        )

    def _probability_result(
        self,
        input: ProteinMpnnInput,
        logits: Tensor,
        decoding_ranks: Tensor,
    ) -> MpnnProbabilityResult:
        r"""
        Wrap aligned per-evaluation logits with the one target's identity.
        """

        return MpnnProbabilityResult(
            target=input.structure,
            component_references=_component_references(
                input.structure.system,
                input.component_index[0],
            ),
            sequence_tokens=input.sequence_tokens[0],
            residue_mask=input.residue_mask[0],
            design_positions=input.design_positions[0],
            logits=logits,
            decoding_ranks=decoding_ranks,
        )

    @staticmethod
    def _validate_decoding_ranks(
        decoding_ranks: Tensor,
        tied_group_ids: Tensor,
    ) -> None:
        r"""
        Assert the exact explicit-rank tensor and its causal-unit identity.
        """

        if decoding_ranks.shape != tied_group_ids.shape:
            raise ValueError("decoding_ranks has the wrong exact shape")
        if decoding_ranks.dtype != torch.long:
            raise TypeError("decoding_ranks must have dtype torch.long")

        group_order = torch.argsort(tied_group_ids, dim=1)
        groups_by_group = torch.gather(tied_group_ids, 1, group_order)
        ranks_by_group = torch.gather(decoding_ranks, 1, group_order)
        same_group = groups_by_group[:, 1:] == groups_by_group[:, :-1]
        torch._assert(
            torch.all(~same_group | (ranks_by_group[:, 1:] == ranks_by_group[:, :-1])),
            "tied members must have equal decoding ranks",
        )

        rank_order = torch.argsort(decoding_ranks, dim=1)
        ranks_by_rank = torch.gather(decoding_ranks, 1, rank_order)
        groups_by_rank = torch.gather(tied_group_ids, 1, rank_order)
        same_rank = ranks_by_rank[:, 1:] == ranks_by_rank[:, :-1]
        torch._assert(
            torch.all(~same_rank | (groups_by_rank[:, 1:] == groups_by_rank[:, :-1])),
            "distinct causal units must have distinct decoding ranks",
        )

    @staticmethod
    def _generate_ranks(
        residue_mask: Tensor,
        design_positions: Tensor,
        tied_group_ids: Tensor,
        generator: torch.Generator | None,
    ) -> Tensor:
        r"""
        Draw fixed-first causal ranks directly from aligned causal-unit IDs.

        Every causal unit receives one random priority. Members of a unit share its
        priority through ``tied_group_ids`` and therefore share its rank, while every
        distinct unit receives a distinct rank. A unit is a generated unit only when
        all of its members are designed; those units sort after every fixed unit.

        Every tensor is derived from ``tied_group_ids`` so ragged batches keep
        each structure's order over its own residues, with no padded rows.
        """

        designed = (residue_mask & design_positions).to(torch.long)
        unit_class = torch.full_like(tied_group_ids, 2).scatter_reduce(
            1,
            tied_group_ids,
            designed,
            reduce="amin",
            include_self=True,
        )
        decoding_priority = (
            torch.empty_like(tied_group_ids, dtype=torch.get_default_dtype()).normal_(generator=generator).abs_()
        )
        bounded_priority = decoding_priority / (1 + decoding_priority)
        unit_key = bounded_priority + unit_class.to(decoding_priority.dtype)
        unit_order = torch.argsort(unit_key, dim=1)
        # Inverse permutation: the position at ``unit_order[r]`` decodes at step ``r``. ``cumsum`` of ones is
        # the per-row 0..L-1 step index without a dense ``arange`` that a ragged batch could not align.
        positions = torch.ones_like(tied_group_ids).cumsum(dim=1) - 1
        unit_rank = torch.zeros_like(unit_order).scatter(1, unit_order, positions)
        return torch.gather(unit_rank, 1, tied_group_ids)

    @staticmethod
    def _representative_positions(tied_group_ids: Tensor) -> Tensor:
        r"""
        Return the first-occurrence row of every causal unit for one input.
        """

        device = tied_group_ids.device
        length = tied_group_ids.shape[1]
        row_index = torch.arange(length, device=device)
        unit_first = torch.full((length,), length, dtype=torch.long, device=device)
        unit_first.scatter_reduce_(0, tied_group_ids[0], row_index, reduce="amin", include_self=True)
        first_of_row = unit_first[tied_group_ids[0]]
        return torch.where(row_index == first_of_row)[0]

    def _encode(
        self,
        backbone_coordinates: Tensor,
        residue_mask: Tensor,
        residue_index: Tensor,
        chain_index: Tensor,
        *,
        coordinate_noise: float,
        generator: torch.Generator | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        r"""
        Run the one coordinate encoder shared by every operation.
        """

        if coordinate_noise:
            backbone_coordinates = backbone_coordinates + coordinate_noise * torch.randn(
                backbone_coordinates.shape,
                device=backbone_coordinates.device,
                dtype=backbone_coordinates.dtype,
                generator=generator,
            )
        mask = residue_mask.to(dtype=backbone_coordinates.dtype)
        # ``attend_mask`` comes from the k-NN: it is the real-neighbor mask, which also zeroes filler slots when
        # a structure is shorter than ``num_neighbors`` (the encoder must not attend to a filler neighbor).
        edge_features, neighbor_indices, attend_mask = self.feature_encoder(
            backbone_coordinates,
            mask,
            residue_index,
            chain_index,
        )
        edge_states = self.edge_embedding(edge_features)
        # Zero node states with the residue structure of ``edge_states`` (one neighbor slot, embedding width):
        # deriving them this way keeps a ragged NestedTensor batch ragged, where ``new_zeros`` with an explicit
        # ``(B, N, hidden)`` shape would materialize a dense, padded tensor that no longer aligns with the edges.
        node_states = torch.zeros_like(edge_states[:, :, 0, :])
        checkpointing = self.gradient_checkpointing and self.training and torch.is_grad_enabled()
        for block in self.encoder_blocks:
            if checkpointing:
                node_states, edge_states = cast(
                    tuple[Tensor, Tensor],
                    self._gradient_checkpointing_func(
                        block.__call__,
                        node_states,
                        edge_states,
                        neighbor_indices,
                        mask,
                        attend_mask,
                    ),
                )
            else:
                node_states, edge_states = block(
                    node_states,
                    edge_states,
                    neighbor_indices,
                    mask,
                    attend_mask,
                )
        return node_states, edge_states, neighbor_indices, attend_mask

    def _decode_logits(
        self,
        node_states: Tensor,
        edge_states: Tensor,
        neighbor_indices: Tensor,
        neighbor_valid: Tensor,
        sequence_tokens: Tensor,
        residue_mask: Tensor,
        decoding_ranks: Tensor,
    ) -> Tensor:
        r"""
        Return raw teacher-forced logits for one exact causal ranking.
        """

        token_states = self.token_embedding(sequence_tokens)
        sequence_edges = cat_neighbors_nodes(token_states, edge_states, neighbor_indices)
        encoder_edges = cat_neighbors_nodes(torch.zeros_like(token_states), edge_states, neighbor_indices)
        encoder_context = cat_neighbors_nodes(node_states, encoder_edges, neighbor_indices)
        backward_mask, forward_mask = autoregressive_masks(
            decoding_ranks,
            neighbor_indices,
            residue_mask.to(dtype=node_states.dtype),
            neighbor_valid.to(dtype=node_states.dtype),
        )
        forward_context = forward_mask * encoder_context
        checkpointing = self.gradient_checkpointing and self.training and torch.is_grad_enabled()
        for block in self.decoder_blocks:
            decoder_context = cat_neighbors_nodes(node_states, sequence_edges, neighbor_indices)
            context = backward_mask * decoder_context + forward_context
            mask = residue_mask.to(dtype=node_states.dtype)
            if checkpointing:
                node_states = cast(
                    Tensor,
                    self._gradient_checkpointing_func(
                        block.__call__,
                        node_states,
                        context,
                        mask,
                    ),
                )
            else:
                node_states = block(node_states, context, mask)
        return self.output_projection(node_states)

    def _decode_sequence_free_logits(
        self,
        node_states: Tensor,
        edge_states: Tensor,
        neighbor_indices: Tensor,
        neighbor_valid: Tensor,
        residue_mask: Tensor,
        decoding_ranks: Tensor,
    ) -> Tensor:
        r"""
        Return raw sequence-free logits under one causal ranking.

        Every position attends only its earlier-decoded neighbors through the
        forward encoder context; no decoded identity ever enters the decoder, so the
        computation stays order dependent without conditioning on any token.
        """

        mask = residue_mask.to(dtype=node_states.dtype)
        token_states = torch.zeros_like(node_states)
        encoder_edges = cat_neighbors_nodes(token_states, edge_states, neighbor_indices)
        encoder_context = cat_neighbors_nodes(node_states, encoder_edges, neighbor_indices)
        _, forward_mask = autoregressive_masks(
            decoding_ranks, neighbor_indices, mask, neighbor_valid.to(dtype=node_states.dtype)
        )
        forward_context = forward_mask * encoder_context
        states = node_states
        checkpointing = self.gradient_checkpointing and self.training and torch.is_grad_enabled()
        for block in self.decoder_blocks:
            if checkpointing:
                states = cast(
                    Tensor,
                    self._gradient_checkpointing_func(
                        block.__call__,
                        states,
                        forward_context,
                        mask,
                    ),
                )
            else:
                states = block(states, forward_context, mask)
        return self.output_projection(states)

    def encode(
        self,
        input: ProteinMpnnInput,
        *,
        coordinate_noise: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        r"""
        Encode a batch of protein backbones as residue graph states.

        Args:
            input: ProteinMPNN tensors with batch and residue axes.
            coordinate_noise: Standard deviation of Gaussian coordinate noise.
            generator: Random generator used for coordinate noise.

        Returns:
            Node states with shape ``(B, L, H)``, edge states with shape
            ``(B, L, K, H)``, neighbor indices with shape ``(B, L, K)``, and the
            real-neighbor mask with shape ``(B, L, K)``.
        """

        return self._encode(
            input.backbone_coordinates,
            input.residue_mask,
            input.residue_index,
            input.chain_index,
            coordinate_noise=coordinate_noise,
            generator=generator,
        )

    def forward(
        self,
        input: ProteinMpnnInput,
        *,
        decoding_ranks: Tensor | None = None,
        coordinate_noise: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> ProteinMpnnOutput:
        r"""
        Predict residue logits conditioned on the input sequence and decoding order.

        Args:
            input: ProteinMPNN tensors with batch size ``B`` and ``L`` residue rows.
            decoding_ranks: Optional ranks with shape ``(B, L)``; smaller values
                decode earlier. A fixed-first random order is drawn when omitted.
            coordinate_noise: Standard deviation of Gaussian coordinate noise.
            generator: Random generator used for the order and coordinate noise.

        Returns:
            Vocabulary logits and the decoding ranks used for the forward pass.
        """

        if decoding_ranks is None:
            decoding_ranks = self._generate_ranks(
                input.residue_mask,
                input.design_positions,
                input.tied_group_ids,
                generator,
            )
        else:
            self._validate_decoding_ranks(decoding_ranks, input.tied_group_ids)
        node_states, edge_states, neighbor_indices, neighbor_valid = self._encode(
            input.backbone_coordinates,
            input.residue_mask,
            input.residue_index,
            input.chain_index,
            coordinate_noise=coordinate_noise,
            generator=generator,
        )
        logits = self._decode_logits(
            node_states,
            edge_states,
            neighbor_indices,
            neighbor_valid,
            input.sequence_tokens,
            input.residue_mask,
            decoding_ranks,
        )
        return ProteinMpnnOutput(
            logits=logits,
            decoding_ranks=decoding_ranks,
            batch_size=[logits.shape[0]],
        )

    def autoregressive_probabilities(
        self,
        input: ProteinMpnnInput,
        *,
        num_decoding_orders: int = 1,
        decoding_ranks: Tensor | None = None,
        coordinate_noise: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> MpnnProbabilityResult:
        r"""
        Evaluate sequence-conditioned probabilities over decoding orders.

        Args:
            input: One processed protein target.
            num_decoding_orders: Number of decoding orders to evaluate.
            decoding_ranks: Optional ranks with shape
                ``(num_decoding_orders, num_residues)``.
            coordinate_noise: Standard deviation of Gaussian coordinate noise.
            generator: Random generator used for orders and coordinate noise.

        Returns:
            Logits and ranks whose leading axis enumerates decoding orders.
        """

        self._require_single_target(input)
        if num_decoding_orders <= 0:
            raise ValueError("num_decoding_orders must be positive")
        repeated = self._repeat_input(input, num_decoding_orders)
        output = self.forward(
            repeated,
            decoding_ranks=decoding_ranks,
            coordinate_noise=coordinate_noise,
            generator=generator,
        )
        return self._probability_result(input, output.logits, output.decoding_ranks)

    def sequence_free_autoregressive_probabilities(
        self,
        input: ProteinMpnnInput,
        *,
        num_decoding_orders: int = 1,
        decoding_ranks: Tensor | None = None,
        coordinate_noise: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> MpnnProbabilityResult:
        r"""
        Evaluate sequence-free probabilities over decoding orders.

        Args:
            input: One processed protein target.
            num_decoding_orders: Number of decoding orders to evaluate.
            decoding_ranks: Optional ranks with shape
                ``(num_decoding_orders, num_residues)``.
            coordinate_noise: Standard deviation of Gaussian coordinate noise.
            generator: Random generator used for orders and coordinate noise.

        Returns:
            Sequence-free logits and ranks whose leading axis enumerates decoding
            orders.
        """

        self._require_single_target(input)
        if num_decoding_orders <= 0:
            raise ValueError("num_decoding_orders must be positive")
        repeated = self._repeat_input(input, num_decoding_orders)
        if decoding_ranks is None:
            decoding_ranks = self._generate_ranks(
                repeated.residue_mask,
                repeated.design_positions,
                repeated.tied_group_ids,
                generator,
            )
        else:
            self._validate_decoding_ranks(decoding_ranks, repeated.tied_group_ids)
        node_states, edge_states, neighbor_indices, neighbor_valid = self._encode(
            repeated.backbone_coordinates,
            repeated.residue_mask,
            repeated.residue_index,
            repeated.chain_index,
            coordinate_noise=coordinate_noise,
            generator=generator,
        )
        logits = self._decode_sequence_free_logits(
            node_states,
            edge_states,
            neighbor_indices,
            neighbor_valid,
            repeated.residue_mask,
            decoding_ranks,
        )
        return self._probability_result(input, logits, decoding_ranks)

    def _single_residue_probabilities(
        self,
        input: ProteinMpnnInput,
        *,
        num_decoding_orders: int,
        base_decoding_ranks: Tensor | None,
        coordinate_noise: float,
        generator: torch.Generator | None,
        move_last: bool,
    ) -> MpnnProbabilityResult:
        r"""
        Evaluate every target moved exactly last or first under one base order.
        """

        self._require_single_target(input)
        if num_decoding_orders <= 0:
            raise ValueError("num_decoding_orders must be positive")
        repeated = self._repeat_input(input, num_decoding_orders)
        device = repeated.backbone_coordinates.device
        length = repeated.sequence_tokens.shape[1]
        if base_decoding_ranks is None:
            untied_group_ids = torch.ones_like(repeated.tied_group_ids).cumsum(dim=1) - 1
            base_ranks = self._generate_ranks(
                repeated.residue_mask,
                repeated.design_positions,
                untied_group_ids,
                generator,
            )
        else:
            if base_decoding_ranks.shape != (num_decoding_orders, length):
                raise ValueError("base_decoding_ranks has the wrong exact shape")
            if base_decoding_ranks.dtype != torch.long:
                raise TypeError("base_decoding_ranks must have dtype torch.long")
            base_ranks = base_decoding_ranks
        boundary_ranks = base_ranks.amax(dim=-1) + 1 if move_last else base_ranks.amin(dim=-1) - 1
        node_states, edge_states, neighbor_indices, neighbor_valid = self._encode(
            repeated.backbone_coordinates,
            repeated.residue_mask,
            repeated.residue_index,
            repeated.chain_index,
            coordinate_noise=coordinate_noise,
            generator=generator,
        )
        vocabulary = len(AMINO_ACID_COMPONENT_IDS) + 1
        logits = node_states.new_zeros(num_decoding_orders, length, vocabulary)
        effective_ranks = torch.empty(
            num_decoding_orders,
            length,
            length,
            dtype=torch.long,
            device=device,
        )
        for target in range(length):
            ranks = base_ranks.clone()
            ranks[:, target] = boundary_ranks
            logits[:, target] = self._decode_logits(
                node_states,
                edge_states,
                neighbor_indices,
                neighbor_valid,
                repeated.sequence_tokens,
                repeated.residue_mask,
                ranks,
            )[:, target]
            effective_ranks[:, target] = ranks
        return self._probability_result(input, logits, effective_ranks)

    def conditional_probabilities(
        self,
        input: ProteinMpnnInput,
        *,
        num_decoding_orders: int = 1,
        base_decoding_ranks: Tensor | None = None,
        coordinate_noise: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> MpnnProbabilityResult:
        r"""
        Evaluate each residue after all other residues in the decoding order.

        Args:
            input: One processed protein target.
            num_decoding_orders: Number of base decoding orders to evaluate.
            base_decoding_ranks: Optional base ranks with shape
                ``(num_decoding_orders, num_residues)``.
            coordinate_noise: Standard deviation of Gaussian coordinate noise.
            generator: Random generator used for orders and coordinate noise.

        Returns:
            Logits with shape ``(num_decoding_orders, num_residues, vocabulary_size)``
            and effective ranks with shape
            ``(num_decoding_orders, num_residues, num_residues)``.
        """

        return self._single_residue_probabilities(
            input,
            num_decoding_orders=num_decoding_orders,
            base_decoding_ranks=base_decoding_ranks,
            coordinate_noise=coordinate_noise,
            generator=generator,
            move_last=True,
        )

    def unconditional_probabilities(
        self,
        input: ProteinMpnnInput,
        *,
        num_decoding_orders: int = 1,
        base_decoding_ranks: Tensor | None = None,
        coordinate_noise: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> MpnnProbabilityResult:
        r"""
        Evaluate each residue before all other residues in the decoding order.

        Args:
            input: One processed protein target.
            num_decoding_orders: Number of base decoding orders to evaluate.
            base_decoding_ranks: Optional base ranks with shape
                ``(num_decoding_orders, num_residues)``.
            coordinate_noise: Standard deviation of Gaussian coordinate noise.
            generator: Random generator used for orders and coordinate noise.

        Returns:
            Logits with shape ``(num_decoding_orders, num_residues, vocabulary_size)``
            and effective ranks with shape
            ``(num_decoding_orders, num_residues, num_residues)``.
        """

        return self._single_residue_probabilities(
            input,
            num_decoding_orders=num_decoding_orders,
            base_decoding_ranks=base_decoding_ranks,
            coordinate_noise=coordinate_noise,
            generator=generator,
            move_last=False,
        )

    @torch.inference_mode()
    def score(
        self,
        input: ProteinMpnnInput,
        *,
        num_decoding_orders: int = 1,
        decoding_ranks: Tensor | None = None,
        coordinate_noise: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> ScoreResult:
        r"""
        Score the observed sequence over one or more decoding orders.

        Args:
            input: One processed protein target.
            num_decoding_orders: Number of decoding orders to evaluate.
            decoding_ranks: Optional ranks with shape
                ``(num_decoding_orders, num_residues)``.
            coordinate_noise: Standard deviation of Gaussian coordinate noise.
            generator: Random generator used for orders and coordinate noise.

        Returns:
            Designed-position and whole-sequence negative log-likelihood scores.
        """

        self._require_single_target(input)
        if num_decoding_orders <= 0:
            raise ValueError("num_decoding_orders must be positive")
        repeated = self._repeat_input(input, num_decoding_orders)
        log_probabilities = self.forward(
            repeated,
            decoding_ranks=decoding_ranks,
            coordinate_noise=coordinate_noise,
            generator=generator,
        ).log_probabilities
        losses = nll_loss(repeated.sequence_tokens, log_probabilities)
        valid = repeated.residue_mask.to(dtype=losses.dtype)
        designed = (repeated.residue_mask & repeated.design_positions).to(dtype=losses.dtype)
        designed_values = (losses * designed).sum(dim=-1) / designed.sum(dim=-1)
        global_values = (losses * valid).sum(dim=-1) / valid.sum(dim=-1)
        host_designed_values = designed_values.detach().cpu()
        host_global_values = global_values.detach().cpu()
        if host_designed_values.dtype == torch.bfloat16:
            host_designed_values = host_designed_values.float()
            host_global_values = host_global_values.float()
        designed_evaluations = host_designed_values.numpy()
        global_evaluations = host_global_values.numpy()
        host_design_positions = input.design_positions[0].detach().cpu()
        host_residue_mask = input.residue_mask[0].detach().cpu()
        references = _component_references(
            input.structure.system,
            input.component_index[0],
        )
        designed_scope = frozenset(
            reference
            for reference, selected, present in zip(
                references,
                host_design_positions.tolist(),
                host_residue_mask.tolist(),
            )
            if reference is not None and selected and present
        )
        global_scope = frozenset(
            reference
            for reference, present in zip(references, host_residue_mask.tolist())
            if reference is not None and present
        )
        return ScoreResult(
            target=input.structure,
            scores={
                "designed_sequence_nll": Score(
                    value=float(designed_evaluations.mean()),
                    unit="nat/residue",
                    preferred="lower",
                    scope=designed_scope,
                    evaluations=designed_evaluations,
                ),
                "global_sequence_nll": Score(
                    value=float(global_evaluations.mean()),
                    unit="nat/residue",
                    preferred="lower",
                    scope=global_scope,
                    evaluations=global_evaluations,
                ),
            },
        )

    @staticmethod
    def _sampling_distribution(
        logits: Tensor,
        *,
        temperature: float,
        global_logit_bias: Tensor | None,
        global_omit_token_mask: Tensor | None,
        residue_logit_bias: Tensor | None,
        pssm_coefficient: Tensor | None,
        pssm_probabilities: Tensor | None,
        pssm_mixture: float,
        pssm_log_odds: Tensor | None,
        pssm_log_odds_threshold: float | None,
        allowed_token_mask: Tensor | None,
    ) -> Tensor:
        r"""
        Apply soft controls before intersecting the final hard support.
        """

        scaled_logits = logits / temperature
        if global_logit_bias is not None:
            scaled_logits = scaled_logits + global_logit_bias / temperature
        if residue_logit_bias is not None:
            scaled_logits = scaled_logits + residue_logit_bias / temperature
        probabilities = F.softmax(scaled_logits, dim=-1)
        if pssm_probabilities is not None and pssm_mixture:
            coefficient = (
                torch.ones(logits.shape[0], device=logits.device, dtype=logits.dtype)
                if pssm_coefficient is None
                else pssm_coefficient
            )
            mixture = pssm_mixture * coefficient
            probabilities = (1 - mixture.unsqueeze(-1)) * probabilities + mixture.unsqueeze(-1) * pssm_probabilities

        support = torch.ones_like(probabilities, dtype=torch.bool)
        if global_omit_token_mask is not None:
            support &= ~global_omit_token_mask.bool()
        if pssm_log_odds is not None and pssm_log_odds_threshold is not None:
            support &= pssm_log_odds > pssm_log_odds_threshold
        if allowed_token_mask is not None:
            support &= allowed_token_mask
        supported_probabilities = probabilities * support
        return supported_probabilities / supported_probabilities.sum(dim=-1, keepdim=True)

    @torch.inference_mode()
    def sample(
        self,
        input: ProteinMpnnInput,
        *,
        num_samples: int = 1,
        temperature: float = 0.1,
        decoding_ranks: Tensor | None = None,
        coordinate_noise: float = 0.0,
        global_logit_bias: Tensor | None = None,
        global_omit_token_mask: Tensor | None = None,
        pssm_mixture: float = 0.0,
        pssm_log_odds_threshold: float | None = None,
        generator: torch.Generator | None = None,
    ) -> ProteinMpnnSamplingOutput:
        r"""
        Sample protein sequences under the processed design constraints.

        Args:
            input: One processed protein target.
            num_samples: Number of sequences to sample.
            temperature: Positive sampling temperature.
            decoding_ranks: Optional ranks with shape ``(num_samples, num_residues)``.
            coordinate_noise: Standard deviation of Gaussian coordinate noise.
            global_logit_bias: Optional bias with shape ``(vocabulary_size,)``.
            global_omit_token_mask: Optional exclusion mask with shape
                ``(vocabulary_size,)``.
            pssm_mixture: Weight of the per-residue PSSM probabilities.
            pssm_log_odds_threshold: Optional minimum accepted PSSM log odds.
            generator: Random generator used for orders, noise, and token sampling.

        Returns:
            Sampled token IDs, model log probabilities, sampling probabilities,
            and decoding ranks, each with a leading ``num_samples`` axis.
        """

        self._require_single_target(input)
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0 <= pssm_mixture <= 1:
            raise ValueError("pssm_mixture must be between zero and one")

        repeated = self._repeat_input(input, num_samples)
        device = repeated.backbone_coordinates.device
        if decoding_ranks is None:
            ranks = self._generate_ranks(
                repeated.residue_mask,
                repeated.design_positions,
                repeated.tied_group_ids,
                generator,
            )
        else:
            self._validate_decoding_ranks(decoding_ranks, repeated.tied_group_ids)
            ranks = decoding_ranks

        node_states, edge_states, neighbor_indices, neighbor_valid = self._encode(
            repeated.backbone_coordinates,
            repeated.residue_mask,
            repeated.residue_index,
            repeated.chain_index,
            coordinate_noise=coordinate_noise,
            generator=generator,
        )
        length = repeated.sequence_tokens.shape[1]
        vocabulary = len(AMINO_ACID_COMPONENT_IDS) + 1
        representative_positions = self._representative_positions(input.tied_group_ids)
        ordered_group_ranks = torch.sort(ranks[:, representative_positions], dim=-1).values
        num_units = ordered_group_ranks.shape[1]

        backward_mask, forward_mask = autoregressive_masks(
            ranks,
            neighbor_indices,
            repeated.residue_mask.to(dtype=node_states.dtype),
            neighbor_valid.to(dtype=node_states.dtype),
        )
        token_states = torch.zeros_like(node_states)
        encoder_edges = cat_neighbors_nodes(token_states, edge_states, neighbor_indices)
        encoder_context = cat_neighbors_nodes(node_states, encoder_edges, neighbor_indices)
        forward_context = forward_mask * encoder_context
        block_states = [node_states] + [torch.zeros_like(node_states) for _ in self.decoder_blocks]
        sequence_tokens = repeated.sequence_tokens.clone()
        log_probabilities = node_states.new_zeros(num_samples, length, vocabulary)
        sampling_probabilities = torch.zeros_like(log_probabilities)
        sample_indices = torch.arange(num_samples, device=device)

        for step in range(num_units):
            member_mask = ranks == ordered_group_ranks[:, step, None]
            batch_indices, positions = torch.where(member_mask)
            neighbor_index = neighbor_indices[batch_indices, positions].unsqueeze(1)
            edge_at_position = edge_states[batch_indices, positions].unsqueeze(1)
            sequence_edge = cat_neighbors_nodes(
                token_states[batch_indices],
                edge_at_position,
                neighbor_index,
            )
            encoder_at_position = forward_context[batch_indices, positions].unsqueeze(1)
            mask_at_position = repeated.residue_mask[batch_indices, positions, None].to(node_states.dtype)
            for block_index, block in enumerate(self.decoder_blocks):
                decoder_context = cat_neighbors_nodes(
                    block_states[block_index][batch_indices],
                    sequence_edge,
                    neighbor_index,
                )
                updated = block(
                    block_states[block_index][batch_indices, positions].unsqueeze(1),
                    backward_mask[batch_indices, positions].unsqueeze(1) * decoder_context + encoder_at_position,
                    mask_at_position,
                )
                block_states[block_index + 1][batch_indices, positions] = updated[:, 0]

            logits = self.output_projection(block_states[-1][batch_indices, positions])
            log_probabilities[batch_indices, positions] = F.log_softmax(logits, dim=-1)
            member_counts = member_mask.sum(dim=-1)
            if repeated.tied_weights is None:
                # Default coefficient 1/len(G) per member: the equal mean of the
                # member logits, matching tied decoding with unit betas.
                combined_logits = torch.segment_reduce(
                    logits,
                    "sum",
                    lengths=member_counts,
                ) / member_counts.unsqueeze(-1)
            else:
                # tied_weights holds the final signed coefficients; the combined
                # group logits are the plain weighted sum sum(coeff[i] * logits[i])
                # with no group-size division. Assigning every member coefficient
                # one therefore yields the sum of member logits, while the
                # processor's default coefficient 1/len(G) reproduces the mean.
                weights = repeated.tied_weights[batch_indices, positions]
                combined_logits = torch.segment_reduce(
                    logits * weights.unsqueeze(-1),
                    "sum",
                    lengths=member_counts,
                )
            segment_starts = torch.cat(
                (
                    torch.zeros(1, dtype=torch.long, device=device),
                    member_counts.cumsum(dim=0)[:-1],
                )
            )
            representatives = positions[segment_starts]
            group_allowed_token_mask = (
                None
                if repeated.allowed_token_mask is None
                else torch.segment_reduce(
                    repeated.allowed_token_mask[batch_indices, positions].to(logits.dtype),
                    "min",
                    lengths=member_counts,
                ).bool()
            )
            group_pssm_log_odds = (
                None
                if repeated.pssm_log_odds is None
                else torch.segment_reduce(
                    repeated.pssm_log_odds[batch_indices, positions],
                    "min",
                    lengths=member_counts,
                )
            )
            member_selected = (
                repeated.residue_mask[batch_indices, positions] & repeated.design_positions[batch_indices, positions]
            )
            active = torch.segment_reduce(
                member_selected.to(logits.dtype),
                "min",
                lengths=member_counts,
            ).bool()
            has_designed_member = torch.segment_reduce(
                member_selected.to(logits.dtype),
                "max",
                lengths=member_counts,
            ).bool()
            fixed_mask = member_mask & ~(repeated.residue_mask & repeated.design_positions)
            anchor_positions = fixed_mask.long().argmax(dim=-1)
            anchor_tokens = repeated.sequence_tokens[sample_indices, anchor_positions]
            anchored = ~active & has_designed_member & (member_counts > 1)
            anchor_allowed = (
                torch.ones_like(anchor_tokens, dtype=torch.bool)
                if group_allowed_token_mask is None
                else group_allowed_token_mask[sample_indices, anchor_tokens]
            )
            if global_omit_token_mask is not None:
                anchor_allowed &= ~global_omit_token_mask[anchor_tokens].bool()
            if group_pssm_log_odds is not None and pssm_log_odds_threshold is not None:
                anchor_allowed &= group_pssm_log_odds[sample_indices, anchor_tokens] > pssm_log_odds_threshold
            torch._assert(
                torch.all(~anchored | anchor_allowed),
                "a tied anchor must remain in allowed token support",
            )

            probabilities = self._sampling_distribution(
                combined_logits,
                temperature=temperature,
                global_logit_bias=global_logit_bias,
                global_omit_token_mask=global_omit_token_mask,
                residue_logit_bias=(
                    None
                    if repeated.residue_logit_bias is None
                    else repeated.residue_logit_bias[sample_indices, representatives]
                ),
                pssm_coefficient=(
                    None
                    if repeated.pssm_coefficients is None
                    else repeated.pssm_coefficients[sample_indices, representatives]
                ),
                pssm_probabilities=(
                    None
                    if repeated.pssm_probabilities is None
                    else repeated.pssm_probabilities[sample_indices, representatives]
                ),
                pssm_mixture=pssm_mixture,
                pssm_log_odds=group_pssm_log_odds,
                pssm_log_odds_threshold=pssm_log_odds_threshold,
                allowed_token_mask=group_allowed_token_mask,
            )
            drawn_tokens = torch.multinomial(
                probabilities,
                1,
                generator=generator,
            ).squeeze(-1)
            group_tokens = anchor_tokens.clone()
            group_tokens[active] = drawn_tokens[active]
            sequence_tokens[batch_indices, positions] = group_tokens[batch_indices]
            anchor_probabilities = torch.zeros_like(probabilities).scatter(
                1,
                group_tokens.unsqueeze(-1),
                1,
            )
            final_probabilities = torch.where(
                active.unsqueeze(-1),
                probabilities,
                anchor_probabilities,
            )
            sampling_probabilities[batch_indices, positions] = final_probabilities[
                batch_indices
            ] * member_selected.unsqueeze(-1)
            token_states[batch_indices, positions] = self.token_embedding(group_tokens[batch_indices])

        return ProteinMpnnSamplingOutput(
            sequence_tokens=sequence_tokens,
            log_probabilities=log_probabilities,
            sampling_probabilities=sampling_probabilities,
            decoding_ranks=ranks,
            batch_size=[num_samples],
        )

    @staticmethod
    def _realize_sequence(
        system: MolecularSystem,
        component_index: Tensor,
        sampled_tokens: Tensor,
        design_positions: Tensor,
    ) -> MolecularSystem:
        r"""
        Realize one candidate while preserving entity and bond identity.
        """

        selected_components = []
        replacements = []
        for component, token, selected in zip(
            component_index.detach().cpu().tolist(),
            sampled_tokens.tolist(),
            design_positions.tolist(),
            strict=True,
        ):
            if component < 0 or not selected:
                continue
            component_id = (
                "UNK" if int(token) == len(AMINO_ACID_COMPONENT_IDS) else AMINO_ACID_COMPONENT_IDS[int(token)]
            )
            if system.symbols[int(system.component_symbol[component])] == component_id:
                continue
            selected_components.append(component)
            replacements.append(get_component(component_id))
        if not selected_components:
            return system
        realized, _, _ = replace_components(
            system,
            torch.tensor(
                selected_components,
                dtype=torch.long,
                device=system.component_symbol.device,
            ),
            replacements,
        )
        return _repartition_entities_by_sequence(system, realized)

    @torch.inference_mode()
    def design(
        self,
        input: ProteinMpnnInput,
        *,
        num_candidates: int = 1,
        num_decoding_orders: int = 1,
        temperature: float = 0.1,
        coordinate_noise: float = 0.0,
        global_logit_bias: Tensor | None = None,
        global_omit_token_mask: Tensor | None = None,
        pssm_mixture: float = 0.0,
        pssm_log_odds_threshold: float | None = None,
        generator: torch.Generator | None = None,
    ) -> ProteinMpnnDesignResult:
        r"""
        Design complete sequence candidates and retain each candidate's input score.

        Args:
            input: One processed protein backbone and its design constraints.
            num_candidates: Number of sequence candidates to sample.
            num_decoding_orders: Number of autoregressive orders used to score each candidate.
            temperature: Positive sampling temperature.
            coordinate_noise: Standard deviation of coordinate noise applied before encoding.
            global_logit_bias: Optional vocabulary-wide additive logit bias.
            global_omit_token_mask: Optional vocabulary-wide token exclusion mask.
            pssm_mixture: Weight assigned to the input position-specific scoring matrix.
            pssm_log_odds_threshold: Optional minimum accepted PSSM log odds.
            generator: Optional PyTorch random generator.

        Returns:
            Candidates in sampling order with their designed molecular systems and input scores.
        """

        sampled = self.sample(
            input,
            num_samples=num_candidates,
            temperature=temperature,
            coordinate_noise=coordinate_noise,
            global_logit_bias=global_logit_bias,
            global_omit_token_mask=global_omit_token_mask,
            pssm_mixture=pssm_mixture,
            pssm_log_odds_threshold=pssm_log_odds_threshold,
            generator=generator,
        )
        untied_group_ids = torch.ones_like(input.tied_group_ids).cumsum(dim=1) - 1
        score_input = ProteinMpnnInput(  # type: ignore[call-arg]
            backbone_coordinates=input.backbone_coordinates,
            sequence_tokens=input.sequence_tokens,
            residue_mask=input.residue_mask,
            residue_index=input.residue_index,
            chain_index=input.chain_index,
            design_positions=input.design_positions,
            structure=input.structure,
            component_index=input.component_index,
            tied_group_ids=untied_group_ids,
            tied_weights=None,
            allowed_token_mask=input.allowed_token_mask,
            residue_logit_bias=input.residue_logit_bias,
            pssm_coefficients=input.pssm_coefficients,
            pssm_probabilities=input.pssm_probabilities,
            pssm_log_odds=input.pssm_log_odds,
            batch_size=[],
        )
        input_score = self.score(
            score_input,
            num_decoding_orders=num_decoding_orders,
            coordinate_noise=coordinate_noise,
            generator=generator,
        )
        losses = nll_loss(sampled.sequence_tokens, sampled.log_probabilities)
        valid = input.residue_mask.repeat_interleave(num_candidates, dim=0).to(losses.dtype)
        designed = (
            (input.residue_mask & input.design_positions).repeat_interleave(num_candidates, dim=0).to(losses.dtype)
        )
        designed_nll = (losses * designed).sum(dim=-1) / designed.sum(dim=-1)
        global_nll = (losses * valid).sum(dim=-1) / valid.sum(dim=-1)
        recovery = (sampled.sequence_tokens == input.sequence_tokens.repeat_interleave(num_candidates, dim=0)).to(
            losses.dtype
        ).mul(designed).sum(dim=-1) / designed.sum(dim=-1)
        length = sampled.sequence_tokens.shape[1]
        host_result = (
            torch.cat(
                (
                    sampled.sequence_tokens.reshape(-1),
                    input.design_positions[0],
                    input.residue_mask[0],
                    torch.stack((designed_nll, global_nll, recovery), dim=-1).reshape(-1),
                )
            )
            .detach()
            .cpu()
        )
        sequence_values = num_candidates * length
        host_sequence_tokens = host_result[:sequence_values].reshape(num_candidates, length).long()
        host_design_positions = host_result[sequence_values : sequence_values + length].bool()
        host_residue_mask = host_result[sequence_values + length : sequence_values + 2 * length].bool()
        host_scores = host_result[sequence_values + 2 * length :].reshape(num_candidates, 3)
        references = _component_references(
            input.structure.system,
            input.component_index[0],
        )
        designed_scope = frozenset(
            reference
            for reference, selected, present in zip(
                references,
                host_design_positions.tolist(),
                host_residue_mask.tolist(),
            )
            if reference is not None and selected and present
        )
        global_scope = frozenset(
            reference
            for reference, present in zip(references, host_residue_mask.tolist())
            if reference is not None and present
        )
        candidates = []
        for index in range(num_candidates):
            system = self._realize_sequence(
                input.structure.system,
                input.component_index[0],
                host_sequence_tokens[index],
                host_design_positions,
            )
            candidates.append(
                DesignCandidate(
                    system=system,
                    scores={
                        "designed_sequence_nll": Score(
                            float(host_scores[index, 0]),
                            unit="nat/residue",
                            preferred="lower",
                            scope=designed_scope,
                        ),
                        "global_sequence_nll": Score(
                            float(host_scores[index, 1]),
                            unit="nat/residue",
                            preferred="lower",
                            scope=global_scope,
                        ),
                        "sequence_recovery": Score(
                            float(host_scores[index, 2]),
                            preferred="higher",
                            scope=designed_scope,
                        ),
                    },
                )
            )
        return ProteinMpnnDesignResult(candidates=tuple(candidates), input_score=input_score)


class ProteinMpnnCaFeatures(nn.Module):
    r"""
    Encode C-alpha coordinates as a ProteinMPNN edge graph.

    Args:
        hidden_size: Width of the returned edge features.
        num_neighbors: Maximum number of nearest neighbors per residue.
    """

    def __init__(self, hidden_size: int, num_neighbors: int) -> None:
        super().__init__()
        self.num_neighbors = num_neighbors
        self.relative_position = MpnnRelativePositionEncoding(16)
        self.edge_projection = nn.Linear(16 + 9 * 16 + 7, hidden_size, bias=False)
        self.edge_norm = LayerNorm(hidden_size)

    @staticmethod
    def _quaternions(rotations: Tensor) -> Tensor:
        r"""
        Convert rotation matrices to normalized quaternion features.

        Args:
            rotations: Rotation matrices with shape ``(..., 3, 3)``.

        Returns:
            Quaternions in ``(x, y, z, w)`` order with shape ``(..., 4)``.
        """

        diagonal = torch.diagonal(rotations, dim1=-2, dim2=-1)
        xx, yy, zz = diagonal.unbind(-1)
        magnitudes = 0.5 * torch.sqrt(
            torch.abs(
                1
                + torch.stack(
                    (
                        xx - yy - zz,
                        -xx + yy - zz,
                        -xx - yy + zz,
                    ),
                    dim=-1,
                )
            )
        )
        signs = torch.sign(
            torch.stack(
                (
                    rotations[..., 2, 1] - rotations[..., 1, 2],
                    rotations[..., 0, 2] - rotations[..., 2, 0],
                    rotations[..., 1, 0] - rotations[..., 0, 1],
                ),
                dim=-1,
            )
        )
        xyz = signs * magnitudes
        w = torch.sqrt(F.relu(1 + diagonal.sum(dim=-1, keepdim=True))) / 2
        return F.normalize(torch.cat((xyz, w), dim=-1), dim=-1)

    @classmethod
    def _orientations(
        cls,
        alpha_carbons: Tensor,
        neighbor_indices: Tensor,
    ) -> Tensor:
        r"""
        Compute relative orientation features for neighboring residues.

        Args:
            alpha_carbons: C-alpha coordinates with shape ``(B, L, 3)``.
            neighbor_indices: Neighbor indices with shape ``(B, L, K)``.

        Returns:
            Local directions and relative quaternions with shape ``(B, L, K, 7)``.
        """

        if alpha_carbons.shape[1] < 3:
            frames = alpha_carbons.new_zeros(
                alpha_carbons.shape[0],
                alpha_carbons.shape[1],
                3,
                3,
            )
        else:
            displacements = alpha_carbons[:, 1:] - alpha_carbons[:, :-1]
            displacement_norms = torch.linalg.vector_norm(displacements, dim=-1)
            consecutive = (displacement_norms > 3.6) & (displacement_norms < 4.0)
            unit = F.normalize(displacements * consecutive.unsqueeze(-1), dim=-1)
            u2 = unit[:, :-2]
            u1 = unit[:, 1:-1]
            n2 = F.normalize(torch.cross(u2, u1, dim=-1), dim=-1)

            first_axis = F.normalize(u2 - u1, dim=-1)
            frames = torch.stack(
                (first_axis, n2, torch.cross(first_axis, n2, dim=-1)),
                dim=-2,
            )
            frames = F.pad(frames.flatten(start_dim=-2), (0, 0, 1, 2)).reshape(
                alpha_carbons.shape[0],
                alpha_carbons.shape[1],
                3,
                3,
            )
        neighbor_frames = gather_nodes(frames.flatten(start_dim=-2), neighbor_indices).reshape(
            *neighbor_indices.shape,
            3,
            3,
        )
        neighbor_coordinates = gather_nodes(alpha_carbons, neighbor_indices)
        direction = neighbor_coordinates - alpha_carbons.unsqueeze(-2)
        direction = torch.matmul(frames.unsqueeze(-3), direction.unsqueeze(-1)).squeeze(-1)
        direction = F.normalize(direction, dim=-1)
        rotations = torch.matmul(frames.unsqueeze(-3).transpose(-1, -2), neighbor_frames)
        return torch.cat((direction, cls._quaternions(rotations)), dim=-1)

    def forward(
        self,
        backbone_coordinates: Tensor,
        residue_mask: Tensor,
        residue_index: Tensor,
        chain_index: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        r"""
        Build C-alpha-only residue edge features.

        Args:
            backbone_coordinates: C-alpha coordinates with shape ``(B, L, 3)``.
            residue_mask: Valid-residue mask with shape ``(B, L)``.
            residue_index: Residue indices with shape ``(B, L)``.
            chain_index: Chain indices with shape ``(B, L)``.

        Returns:
            Edge features ``(B, L, K, H)``, neighbor indices ``(B, L, K)``, and the real-neighbor mask
            ``(B, L, K)`` (the encoder's ``attend_mask``).
        """

        neighbor_distances, neighbor_indices, neighbor_valid = nearest_neighbors(
            backbone_coordinates,
            residue_mask,
            self.num_neighbors,
        )
        previous = torch.zeros_like(backbone_coordinates)
        previous[:, 1:] = backbone_coordinates[:, :-1]
        following = torch.zeros_like(backbone_coordinates)
        following[:, :-1] = backbone_coordinates[:, 1:]
        orientation = self._orientations(backbone_coordinates, neighbor_indices)
        distance_features = torch.cat(
            (
                radial_basis(neighbor_distances),
                _pair_radial_basis(previous, previous, neighbor_indices),
                _pair_radial_basis(following, following, neighbor_indices),
                _pair_radial_basis(previous, backbone_coordinates, neighbor_indices),
                _pair_radial_basis(previous, following, neighbor_indices),
                _pair_radial_basis(backbone_coordinates, previous, neighbor_indices),
                _pair_radial_basis(backbone_coordinates, following, neighbor_indices),
                _pair_radial_basis(following, previous, neighbor_indices),
                _pair_radial_basis(following, backbone_coordinates, neighbor_indices),
            ),
            dim=-1,
        )
        residue_offsets = residue_index[:, :, None] - residue_index[:, None, :]
        residue_offsets = gather_edges(residue_offsets.unsqueeze(-1), neighbor_indices).squeeze(-1)
        same_chain = (chain_index[:, :, None] == chain_index[:, None, :]).long()
        same_chain = gather_edges(same_chain.unsqueeze(-1), neighbor_indices).squeeze(-1)
        position_features = self.relative_position(residue_offsets.long(), same_chain)
        edges = torch.cat((position_features, distance_features, orientation), dim=-1)
        return self.edge_norm(self.edge_projection(edges)), neighbor_indices, neighbor_valid


def _pair_radial_basis(
    source_coordinates: Tensor,
    target_coordinates: Tensor,
    neighbor_indices: Tensor,
    num_bins: int = 16,
) -> Tensor:
    r"""
    Encode source-to-neighbor-target distances with radial bases.
    """

    neighbor_coordinates = gather_nodes(target_coordinates, neighbor_indices)
    neighbor_distances = torch.sqrt(
        torch.sum((source_coordinates.unsqueeze(-2) - neighbor_coordinates) ** 2, dim=-1) + 1e-6
    )
    return radial_basis(neighbor_distances, num_bins)


# ===========================================================================
# ProteinMPNN typed inputs, outputs, and Results.
# ===========================================================================


@tensorclass
class ProteinMpnnInput:
    r"""
    Inputs for ProteinMPNN sequence design and scoring.

    Attributes:
        backbone_coordinates: C-alpha coordinates with shape
            ``(batch_size, num_residues, 3)`` for a C-alpha model, or N, C-alpha,
            C, and O coordinates with shape ``(batch_size, num_residues, 4, 3)``.
        sequence_tokens: Residue token IDs with shape ``(batch_size, num_residues)``.
        residue_mask: Valid-residue mask with shape ``(batch_size, num_residues)``.
        residue_index: Residue indices with shape ``(batch_size, num_residues)``.
        chain_index: Chain indices with shape ``(batch_size, num_residues)``.
        design_positions: Positions eligible for design with shape
            ``(batch_size, num_residues)``.
        structure: Canonical source structure. A Runner batch is one disjoint union
            whose system axis follows the row batch.
        component_index: Canonical component index aligned to every residue row;
            ``-1`` marks a transient source-numbering gap.
        tied_group_ids: Autoregressive unit ID for each residue row.
        tied_weights: Optional per-residue weights for tied sampling groups.
        allowed_token_mask: Optional allowed-token mask with shape
            ``(batch_size, num_residues, vocabulary_size)``.
        residue_logit_bias: Optional additive logit bias with shape
            ``(batch_size, num_residues, vocabulary_size)``.
        pssm_coefficients: Optional PSSM mixture coefficients with shape
            ``(batch_size, num_residues)``.
        pssm_probabilities: Optional PSSM probabilities with shape
            ``(batch_size, num_residues, vocabulary_size)``.
        pssm_log_odds: Optional PSSM log odds with shape
            ``(batch_size, num_residues, vocabulary_size)``.
    """

    backbone_coordinates: Tensor
    sequence_tokens: Tensor
    residue_mask: Tensor
    residue_index: Tensor
    chain_index: Tensor
    design_positions: Tensor
    structure: Structure
    component_index: Tensor
    tied_group_ids: Tensor
    tied_weights: Tensor | None = None
    allowed_token_mask: Tensor | None = None
    residue_logit_bias: Tensor | None = None
    pssm_coefficients: Tensor | None = None
    pssm_probabilities: Tensor | None = None
    pssm_log_odds: Tensor | None = None


class ProteinMpnnOutput(_TensorCollection):
    r"""
    Outputs from one ProteinMPNN forward pass.

    Attributes:
        logits: Vocabulary logits with shape
            ``(batch_size, num_residues, vocabulary_size)``.
        decoding_ranks: Autoregressive ranks with shape
            ``(batch_size, num_residues)``.
    """

    logits: Tensor
    decoding_ranks: Tensor

    @property
    def log_probabilities(self) -> Tensor:
        r"""
        Return vocabulary log probabilities.
        """

        return self.logits.log_softmax(dim=-1)

    @property
    def probabilities(self) -> Tensor:
        r"""
        Return vocabulary probabilities.
        """

        return self.logits.softmax(dim=-1)


class ProteinMpnnSamplingOutput(_TensorCollection):
    r"""
    Outputs from ProteinMPNN autoregressive sampling.

    Attributes:
        sequence_tokens: Sampled token IDs with shape
            ``(num_samples, num_residues)``.
        log_probabilities: Model log probabilities with shape
            ``(num_samples, num_residues, vocabulary_size)``.
        sampling_probabilities: Sampling distribution used at each position.
        decoding_ranks: Autoregressive ranks with shape
            ``(num_samples, num_residues)``.
    """

    sequence_tokens: Tensor
    log_probabilities: Tensor
    sampling_probabilities: Tensor
    decoding_ranks: Tensor


@dataclass(eq=False)
class ProteinMpnnDesignResult(DesignResult):
    r"""
    ProteinMPNN designs and the source-sequence score.

    Attributes:
        candidates: Designed molecular systems in generation order.
        input_score: Score of the input sequence under the same structure-conditioned
            model.
    """

    input_score: ScoreResult
