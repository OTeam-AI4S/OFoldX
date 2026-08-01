r"""
LigandMpnn context-aware sequence-design model.
"""

from __future__ import annotations

from collections.abc import Callable
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
from ...modules.geometry import virtual_beta_carbon
from ...modules.mpnn import (
    AMINO_ACID_COMPONENT_IDS,
    MpnnDecoderBlock,
    MpnnEncoderBlock,
    MpnnProteinFeatures,
    autoregressive_masks,
    cat_neighbors_nodes,
    gather_nodes,
    local_backbone_frame_features,
    nll_loss,
    periodic_group_period,
    radial_basis,
)
from ...results import DesignCandidate, DesignResult, MpnnProbabilityResult, Score
from .config import LigandMpnnConfig

_VOCABULARY = len(AMINO_ACID_COMPONENT_IDS) + 1
_GENERATABLE = len(AMINO_ACID_COMPONENT_IDS)
_SIDECHAIN_ATOMIC_NUMBERS = (
    6,
    6,
    6,
    8,
    8,
    16,
    6,
    6,
    6,
    7,
    7,
    8,
    8,
    16,
    6,
    6,
    6,
    6,
    7,
    7,
    7,
    8,
    8,
    6,
    7,
    7,
    8,
    6,
    6,
    6,
    7,
    8,
)


class LigandMpnnModel(Model):
    r"""
    Design protein sequences in protein-ligand contexts with LigandMPNN.

    Accepts [`LigandMpnnInput`][ofoldx.models.ligandmpnn.model.LigandMpnnInput] and returns
    [`LigandMpnnOutput`][ofoldx.models.ligandmpnn.model.LigandMpnnOutput].

    Args:
        config: Model architecture and inference configuration.

    Examples:
        >>> import torch
        >>> from ofoldx.data import (
        ...     MolecularSystem, PolymerSequence, Structure,
        ... )
        >>> from ofoldx.models.ligandmpnn import (
        ...     LigandMpnnConfig, LigandMpnnModel, LigandMpnnProcessor,
        ...     LigandMpnnProcessorConfig,
        ... )
        >>> system = MolecularSystem.from_entities(
        ...     {"protein": PolymerSequence.from_one_letter("A")},
        ...     {"A": "protein"},
        ... )
        >>> atom_count = system.atom_symbol.numel()
        >>> component_count = system.component_symbol.numel()
        >>> structure = Structure(
        ...     system=system,
        ...     coordinates=system.reference_coordinates.clone(),
        ...     coordinate_mask=system.reference_coordinate_mask.clone(),
        ...     occupancy=torch.zeros(atom_count),
        ...     occupancy_mask=torch.zeros(atom_count, dtype=torch.bool),
        ...     b_factor=torch.zeros(atom_count),
        ...     b_factor_mask=torch.zeros(atom_count, dtype=torch.bool),
        ...     source_symbols=(),
        ...     residue_id=torch.full((component_count,), -1, dtype=torch.int64),
        ...     insertion_code=torch.full((component_count,), -1, dtype=torch.int64),
        ...     standard_component_id=torch.full((component_count,), -1, dtype=torch.int64),
        ...     batch_size=[],
        ... )
        >>> input = LigandMpnnProcessor(
        ...     LigandMpnnProcessorConfig({"num_context_atoms": 1})
        ... )(structure)
        >>> config = LigandMpnnConfig(
        ...     {"hidden_size": 8, "num_encoder_blocks": 1, "num_decoder_blocks": 1,
        ...      "num_neighbors": 1, "dropout": 0.0}
        ... )
        >>> model = LigandMpnnModel(config)
        >>> output = model(input, decoding_ranks=torch.zeros_like(input.sequence_tokens))
        >>> output.logits.shape
        torch.Size([1, 1, 21])
        >>> output.decoding_ranks.tolist()
        [[0]]
    """

    config_class = LigandMpnnConfig
    config: LigandMpnnConfig
    _gradient_checkpointing_func: Callable[..., object]

    def __init__(self, config: LigandMpnnConfig | None = None) -> None:
        config = LigandMpnnConfig() if config is None else config
        super().__init__(config)
        self.gradient_checkpointing = False
        hidden_size = self.config.hidden_size
        self.feature_encoder = LigandMpnnFeatures(hidden_size, self.config.num_neighbors)
        self.edge_embedding = nn.Linear(hidden_size, hidden_size)
        self.atom_context_embedding = nn.Linear(hidden_size, hidden_size)
        self.encoder_blocks = nn.ModuleList(
            MpnnEncoderBlock(hidden_size, hidden_size * 2, dropout=self.config.dropout)
            for _ in range(self.config.num_encoder_blocks)
        )
        self.residue_context_input = nn.Linear(hidden_size, hidden_size)
        self.atom_node_embedding = nn.Linear(hidden_size, hidden_size)
        self.atom_edge_embedding = nn.Linear(hidden_size, hidden_size)
        self.atom_context_blocks = nn.ModuleList(
            MpnnDecoderBlock(hidden_size, hidden_size, dropout=self.config.dropout) for _ in range(2)
        )
        self.context_blocks = nn.ModuleList(
            MpnnDecoderBlock(hidden_size, hidden_size * 2, dropout=self.config.dropout) for _ in range(2)
        )
        self.context_output = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(self.config.dropout)
        self.context_norm = LayerNorm(hidden_size)
        self.token_embedding = nn.Embedding(_VOCABULARY, hidden_size)
        self.decoder_blocks = nn.ModuleList(
            MpnnDecoderBlock(hidden_size, hidden_size * 3, dropout=self.config.dropout)
            for _ in range(self.config.num_decoder_blocks)
        )
        self.output_projection = nn.Linear(hidden_size, _VOCABULARY)
        self.post_init()

    def initialize_parameters(self) -> None:
        r"""
        Apply the Xavier-uniform initialization to every matrix parameter.
        """

        for parameter in self.parameters():
            if parameter.ndim > 1:
                nn.init.xavier_uniform_(parameter)

    @staticmethod
    def _require_single_target(input: LigandMpnnInput) -> None:
        r"""
        Reject an execution-batch axis on one-target biological operations.
        """

        if input.backbone_coordinates.shape[0] != 1 or input.structure.system.system_entity_ptr.numel() != 2:
            raise ValueError("this LigandMpnn operation returns one target and therefore accepts one input")

    @staticmethod
    def _repeat_input(input: LigandMpnnInput, count: int) -> LigandMpnnInput:
        r"""
        Repeat one biological input into ``count`` aligned evaluation rows.
        """

        def repeat(value: Tensor | None) -> Tensor | None:
            return None if value is None else value.repeat_interleave(count, dim=0)

        return LigandMpnnInput(  # type: ignore[call-arg]
            backbone_coordinates=repeat(input.backbone_coordinates),
            sequence_tokens=repeat(input.sequence_tokens),
            residue_mask=repeat(input.residue_mask),
            residue_index=repeat(input.residue_index),
            chain_index=repeat(input.chain_index),
            context_coordinates=repeat(input.context_coordinates),
            context_atomic_numbers=repeat(input.context_atomic_numbers),
            context_mask=repeat(input.context_mask),
            residue_context_coordinates=repeat(input.residue_context_coordinates),
            residue_context_mask=repeat(input.residue_context_mask),
            design_positions=repeat(input.design_positions),
            ligand_interface_mask=repeat(input.ligand_interface_mask),
            structure=input.structure,
            component_index=repeat(input.component_index),
            tied_group_ids=repeat(input.tied_group_ids),
            tied_weights=repeat(input.tied_weights),
            allowed_token_mask=repeat(input.allowed_token_mask),
            logit_bias=repeat(input.logit_bias),
            batch_size=[],
        )

    def _probability_result(
        self,
        input: LigandMpnnInput,
        logits: Tensor,
        decoding_ranks: Tensor,
    ) -> MpnnProbabilityResult:
        r"""
        Wrap aligned per-evaluation logits with the one target's canonical identity.
        """

        component_indices = input.component_index[0].detach().cpu().tolist()
        return MpnnProbabilityResult(
            target=input.structure,
            component_references=tuple(
                input.structure.system.component_reference(component) for component in component_indices
            ),
            sequence_tokens=input.sequence_tokens[0],
            residue_mask=input.residue_mask[0],
            design_positions=input.design_positions[0],
            logits=logits,
            decoding_ranks=decoding_ranks,
        )

    # -- decode-order generation -----------------------------------------------------------------

    @staticmethod
    def _validate_decoding_ranks(decoding_ranks: Tensor, tied_group_ids: Tensor) -> None:
        r"""Validate exact ranks and their aligned causal-unit identity."""

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
        r"""Draw independent half-normal priorities for fixed-first causal units."""

        designed = (residue_mask & design_positions).to(torch.long)
        unit_class = torch.full_like(tied_group_ids, 2).scatter_reduce(
            1,
            tied_group_ids,
            designed,
            reduce="amin",
            include_self=True,
        )
        priority = torch.empty_like(tied_group_ids, dtype=torch.get_default_dtype()).normal_(generator=generator).abs_()
        bounded_priority = priority / (1 + priority)
        unit_order = torch.argsort(bounded_priority + unit_class.to(priority.dtype), dim=1)
        positions = torch.ones_like(tied_group_ids).cumsum(dim=1) - 1
        unit_rank = torch.zeros_like(unit_order).scatter(1, unit_order, positions)
        return torch.gather(unit_rank, 1, tied_group_ids)

    @staticmethod
    def _representative_positions(tied_group_ids: Tensor) -> Tensor:
        r"""Return the first row belonging to every causal unit in one input."""

        device = tied_group_ids.device
        length = tied_group_ids.shape[1]
        row_index = torch.arange(length, device=device)
        unit_first = torch.full((length,), length, dtype=torch.long, device=device)
        unit_first.scatter_reduce_(0, tied_group_ids[0], row_index, reduce="amin", include_self=True)
        return torch.where(row_index == unit_first[tied_group_ids[0]])[0]

    # -- shared graph passes ---------------------------------------------------------------------

    def encode(
        self,
        input: LigandMpnnInput,
        *,
        coordinate_noise: float = 0.0,
        use_side_chain_context: bool = False,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        r"""
        Run the coordinate/context encoder shared by every operation.

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
        mask = input.residue_mask.to(dtype=backbone_coordinates.dtype)
        # ``attend_mask`` is the k-NN's real-neighbor mask, which also zeroes filler slots when a structure is
        # shorter than ``num_neighbors`` (the k-NN's ``residue_mask`` is ``mask``, so it matches).
        (
            atom_features,
            edges,
            neighbor_indices,
            atom_nodes,
            atom_edges,
            atom_mask,
            attend_mask,
        ) = self.feature_encoder(
            backbone_coordinates,
            mask,
            input.residue_index,
            input.chain_index,
            context_coordinates,
            input.context_atomic_numbers,
            input.context_mask.to(dtype=backbone_coordinates.dtype),
            input.residue_context_coordinates,
            input.residue_context_mask.to(dtype=backbone_coordinates.dtype),
            input.design_positions.to(dtype=backbone_coordinates.dtype),
            use_side_chain_context=use_side_chain_context,
        )
        # Zero node states with the residue structure of ``edges`` (one neighbor slot, edge width, which
        # equals ``hidden_size``): deriving them from a ragged slice keeps a NestedTensor batch ragged,
        # where ``new_zeros`` with an explicit ``(B, L, hidden)`` shape would materialize a dense, padded
        # tensor that no longer aligns with the edges.
        node_states = torch.zeros_like(edges[:, :, 0, :])
        edge_states = self.edge_embedding(edges)
        atom_context = self.atom_context_embedding(atom_features)
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
                node_states, edge_states = block(node_states, edge_states, neighbor_indices, mask, attend_mask)
        residue_context = self.residue_context_input(node_states)
        atom_pair_mask = atom_mask[:, :, :, None] * atom_mask[:, :, None, :]
        atom_nodes = self.atom_node_embedding(atom_nodes)
        atom_edges = self.atom_edge_embedding(atom_edges)
        for block_index, block in enumerate(self.context_blocks):
            if checkpointing:
                atom_nodes = cast(
                    Tensor,
                    self._gradient_checkpointing_func(
                        self.atom_context_blocks[block_index].__call__,
                        atom_nodes,
                        atom_edges,
                        atom_mask,
                        atom_pair_mask,
                    ),
                )
                residue_context = cast(
                    Tensor,
                    self._gradient_checkpointing_func(
                        block.__call__,
                        residue_context,
                        torch.cat((atom_context, atom_nodes), dim=-1),
                        mask,
                        atom_mask,
                    ),
                )
            else:
                atom_nodes = self.atom_context_blocks[block_index](atom_nodes, atom_edges, atom_mask, atom_pair_mask)
                residue_context = block(
                    residue_context,
                    torch.cat((atom_context, atom_nodes), dim=-1),
                    mask,
                    atom_mask,
                )
        node_states = node_states + self.context_norm(self.dropout(self.context_output(residue_context)))
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
        Return raw teacher-forced sequence-conditioned logits for one exact causal ranking.
        """

        mask = residue_mask.to(dtype=node_states.dtype)
        token_states = self.token_embedding(sequence_tokens)
        sequence_edges = cat_neighbors_nodes(token_states, edge_states, neighbor_indices)
        encoder_edges = cat_neighbors_nodes(torch.zeros_like(token_states), edge_states, neighbor_indices)
        encoder_context = cat_neighbors_nodes(node_states, encoder_edges, neighbor_indices)
        backward_mask, forward_mask = autoregressive_masks(
            decoding_ranks, neighbor_indices, mask, neighbor_valid.to(dtype=node_states.dtype)
        )
        forward_context = forward_mask * encoder_context
        checkpointing = self.gradient_checkpointing and self.training and torch.is_grad_enabled()
        for block in self.decoder_blocks:
            decoder_context = cat_neighbors_nodes(node_states, sequence_edges, neighbor_indices)
            context = backward_mask * decoder_context + forward_context
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
        Return raw order-dependent sequence-free logits under one causal ranking.
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

    # -- public operations -----------------------------------------------------------------------

    def forward(
        self,
        input: LigandMpnnInput,
        *,
        decoding_ranks: Tensor | None = None,
        coordinate_noise: float = 0.0,
        use_side_chain_context: bool = False,
        generator: torch.Generator | None = None,
    ) -> LigandMpnnOutput:
        r"""
        Predict residue logits conditioned on the input sequence and decoding order.

        Args:
            input: LigandMPNN tensors with batch size ``B`` and ``L`` residue rows.
            decoding_ranks: Optional ranks with shape ``(B, L)``; smaller values
                decode earlier. A fixed-first random order is drawn when omitted.
            coordinate_noise: Standard deviation of Gaussian noise in coordinate units.
            use_side_chain_context: Whether fixed protein side chains contribute context atoms.
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
        node_states, edge_states, neighbor_indices, neighbor_valid = self.encode(
            input,
            coordinate_noise=coordinate_noise,
            use_side_chain_context=use_side_chain_context,
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
        return LigandMpnnOutput(
            logits=logits,
            decoding_ranks=decoding_ranks,
            batch_size=[logits.shape[0]],
        )

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
        r"""
        Evaluate sequence-conditioned probabilities over decoding orders.

        Args:
            input: One processed protein-ligand target.
            num_decoding_orders: Number of decoding orders to evaluate.
            decoding_ranks: Optional ranks with shape
                ``(num_decoding_orders, num_residues)``.
            coordinate_noise: Standard deviation of Gaussian noise in coordinate units.
            use_side_chain_context: Whether fixed protein side chains contribute context atoms.
            generator: Random generator used for orders and coordinate noise.

        Returns:
            Logits and ranks whose leading axis enumerates decoding orders.
        """

        self._require_single_target(input)
        if num_decoding_orders <= 0:
            raise ValueError("num_decoding_orders must be positive")
        repeated = self._repeat_input(input, num_decoding_orders)
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
        node_states, edge_states, neighbor_indices, neighbor_valid = self.encode(
            repeated,
            coordinate_noise=coordinate_noise,
            use_side_chain_context=use_side_chain_context,
            generator=generator,
        )
        logits = self._decode_logits(
            node_states,
            edge_states,
            neighbor_indices,
            neighbor_valid,
            repeated.sequence_tokens,
            repeated.residue_mask,
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
        r"""
        Evaluate sequence-free probabilities over decoding orders.

        Args:
            input: One processed protein-ligand target.
            num_decoding_orders: Number of decoding orders to evaluate.
            decoding_ranks: Optional ranks with shape
                ``(num_decoding_orders, num_residues)``.
            coordinate_noise: Standard deviation of Gaussian noise in coordinate units.
            use_side_chain_context: Whether fixed protein side chains contribute context atoms.
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
            ranks = self._generate_ranks(
                repeated.residue_mask,
                repeated.design_positions,
                repeated.tied_group_ids,
                generator,
            )
        else:
            self._validate_decoding_ranks(decoding_ranks, repeated.tied_group_ids)
            ranks = decoding_ranks
        node_states, edge_states, neighbor_indices, neighbor_valid = self.encode(
            repeated,
            coordinate_noise=coordinate_noise,
            use_side_chain_context=use_side_chain_context,
            generator=generator,
        )
        logits = self._decode_sequence_free_logits(
            node_states,
            edge_states,
            neighbor_indices,
            neighbor_valid,
            repeated.residue_mask,
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
        r"""
        Evaluate every target moved exactly last or first under one base decoding order.
        """

        self._require_single_target(input)
        if num_decoding_orders <= 0:
            raise ValueError("num_decoding_orders must be positive")
        repeated = self._repeat_input(input, num_decoding_orders)
        device = repeated.backbone_coordinates.device
        length = repeated.sequence_tokens.shape[1]
        untied_group_ids = torch.ones_like(repeated.tied_group_ids).cumsum(dim=1) - 1
        if base_decoding_ranks is None:
            base_ranks = self._generate_ranks(
                repeated.residue_mask,
                repeated.design_positions,
                untied_group_ids,
                generator,
            )
        else:
            self._validate_decoding_ranks(base_decoding_ranks, untied_group_ids)
            base_ranks = base_decoding_ranks
        node_states, edge_states, neighbor_indices, neighbor_valid = self.encode(
            repeated,
            coordinate_noise=coordinate_noise,
            use_side_chain_context=use_side_chain_context,
            generator=generator,
        )
        first_ranks = base_ranks.amin(dim=-1) - 1
        last_ranks = base_ranks.amax(dim=-1) + 1
        logits = node_states.new_zeros(num_decoding_orders, length, _VOCABULARY)
        effective_ranks = torch.empty(num_decoding_orders, length, length, dtype=torch.long, device=device)
        for target in range(length):
            ranks = base_ranks.clone()
            ranks[:, target] = last_ranks if move_last else first_ranks
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
        input: LigandMpnnInput,
        *,
        num_decoding_orders: int = 1,
        base_decoding_ranks: Tensor | None = None,
        coordinate_noise: float = 0.0,
        use_side_chain_context: bool = False,
        generator: torch.Generator | None = None,
    ) -> MpnnProbabilityResult:
        r"""
        Evaluate each residue after all other residues in the decoding order.

        Args:
            input: One processed protein-ligand target.
            num_decoding_orders: Number of base decoding orders to evaluate.
            base_decoding_ranks: Optional base ranks with shape
                ``(num_decoding_orders, num_residues)``.
            coordinate_noise: Standard deviation of Gaussian noise in coordinate units.
            use_side_chain_context: Whether fixed protein side chains contribute context atoms.
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
        r"""
        Evaluate each residue before all other residues in the decoding order.

        Args:
            input: One processed protein-ligand target.
            num_decoding_orders: Number of base decoding orders to evaluate.
            base_decoding_ranks: Optional base ranks with shape
                ``(num_decoding_orders, num_residues)``.
            coordinate_noise: Standard deviation of Gaussian noise in coordinate units.
            use_side_chain_context: Whether fixed protein side chains contribute context atoms.
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
            use_side_chain_context=use_side_chain_context,
            generator=generator,
            move_last=False,
        )

    # -- sampling --------------------------------------------------------------------------------

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
    ) -> LigandMpnnSamplingOutput:
        r"""
        Sample complete protein sequences one causal unit at a time.

        Tied members share one rank, are evaluated before any member token is
        published, and use one weighted-logit sampling distribution.

        Args:
            input: One processed protein-ligand target.
            num_samples: Number of sequences to sample.
            temperature: Positive sampling temperature.
            coordinate_noise: Standard deviation of coordinate noise.
            use_side_chain_context: Whether fixed protein side chains contribute
                atom context.
            decoding_ranks: Optional exact ranks with shape
                ``(num_samples, num_residues)``.
            generator: Random generator used for ranks, noise, and token sampling.

        Returns:
            Sampled tokens, raw log probabilities, adjusted sampling
            probabilities, and exact causal ranks.
        """

        self._require_single_target(input)
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        repeated = self._repeat_input(input, num_samples)
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
        node_states, edge_states, neighbor_indices, neighbor_valid = self.encode(
            repeated,
            coordinate_noise=coordinate_noise,
            use_side_chain_context=use_side_chain_context,
            generator=generator,
        )

        residue_mask = repeated.residue_mask.to(dtype=node_states.dtype)
        backward_mask, forward_mask = autoregressive_masks(
            ranks,
            neighbor_indices,
            residue_mask,
            neighbor_valid.to(dtype=node_states.dtype),
        )
        token_states = torch.zeros_like(node_states)
        encoder_edges = cat_neighbors_nodes(token_states, edge_states, neighbor_indices)
        encoder_context = cat_neighbors_nodes(node_states, encoder_edges, neighbor_indices)
        forward_context = forward_mask * encoder_context
        block_states = [node_states] + [torch.zeros_like(node_states) for _ in self.decoder_blocks]

        length = repeated.sequence_tokens.shape[1]
        device = repeated.sequence_tokens.device
        sequence_tokens = repeated.sequence_tokens.clone()
        log_probabilities = node_states.new_zeros(num_samples, length, _VOCABULARY)
        sampling_probabilities = node_states.new_zeros(num_samples, length, _GENERATABLE)
        sample_indices = torch.arange(num_samples, device=device)

        representatives = self._representative_positions(input.tied_group_ids)
        ordered_ranks = torch.sort(ranks[:, representatives], dim=-1).values
        for step in range(ordered_ranks.shape[1]):
            member_mask = ranks == ordered_ranks[:, step, None]
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
            selected = (
                repeated.residue_mask[batch_indices, positions] & repeated.design_positions[batch_indices, positions]
            )
            log_probabilities[batch_indices, positions] = F.log_softmax(logits, dim=-1) * selected.unsqueeze(-1)

            member_counts = member_mask.sum(dim=-1)
            if repeated.tied_weights is None:
                combined_logits = torch.segment_reduce(logits, "sum", lengths=member_counts)
            else:
                weights = repeated.tied_weights[batch_indices, positions].to(logits.dtype)
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
            representative_rows = positions[segment_starts]
            if repeated.logit_bias is not None:
                combined_logits = (
                    combined_logits
                    + repeated.logit_bias[
                        sample_indices,
                        representative_rows,
                    ]
                )
            group_support = (
                None
                if repeated.allowed_token_mask is None
                else torch.segment_reduce(
                    repeated.allowed_token_mask[batch_indices, positions].to(logits.dtype),
                    "min",
                    lengths=member_counts,
                ).bool()
            )

            fixed_mask = member_mask & ~(repeated.residue_mask & repeated.design_positions)
            has_fixed = fixed_mask.any(dim=-1)
            anchor_positions = fixed_mask.long().argmax(dim=-1)
            anchor_tokens = repeated.sequence_tokens[sample_indices, anchor_positions]
            has_designed = torch.segment_reduce(
                selected.to(logits.dtype),
                "max",
                lengths=member_counts,
            ).bool()
            anchored = has_fixed & has_designed
            generatable_anchor = (anchor_tokens >= 0) & (anchor_tokens < _GENERATABLE)
            torch._assert(
                torch.all(~anchored | generatable_anchor),
                "a mixed fixed/design causal unit requires a generatable fixed identity",
            )
            if group_support is not None:
                anchor_supported = torch.gather(
                    group_support,
                    1,
                    anchor_tokens.clamp(0, _VOCABULARY - 1).unsqueeze(-1),
                ).squeeze(-1)
                torch._assert(
                    torch.all(~anchored | anchor_supported),
                    "a mixed fixed/design causal unit requires its fixed identity in final support",
                )
            if group_support is not None:
                combined_logits = combined_logits.masked_fill(~group_support, -torch.inf)
            probabilities = F.softmax(combined_logits / temperature, dim=-1)[..., :_GENERATABLE]
            probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)

            drawn_tokens = torch.multinomial(probabilities, 1, generator=generator).squeeze(-1)
            group_tokens = torch.where(has_fixed, anchor_tokens, drawn_tokens)

            probability_anchor = torch.where(anchored, anchor_tokens, 0)
            anchor_probabilities = F.one_hot(
                probability_anchor,
                _GENERATABLE,
            ).to(probabilities.dtype)
            probabilities = torch.where(
                anchored.unsqueeze(-1),
                anchor_probabilities,
                probabilities,
            )
            sampling_probabilities[batch_indices, positions] = probabilities[batch_indices] * selected.unsqueeze(-1)
            sequence_tokens[batch_indices, positions] = group_tokens[batch_indices]
            token_states[batch_indices, positions] = self.token_embedding(group_tokens[batch_indices])

        return LigandMpnnSamplingOutput(
            sequence_tokens=sequence_tokens,
            log_probabilities=log_probabilities,
            sampling_probabilities=sampling_probabilities,
            decoding_ranks=ranks,
            batch_size=[num_samples],
        )

    # -- design ----------------------------------------------------------------------------------

    @staticmethod
    def _realize_sequence(
        system: MolecularSystem,
        component_indices: list[int],
        sampled_tokens: list[int],
        design_positions: list[bool],
    ) -> MolecularSystem:
        r"""
        Realize one candidate while preserving entity and bond identity.
        """

        selected_components = [
            component
            for component, selected in zip(
                component_indices,
                design_positions,
                strict=True,
            )
            if selected
        ]
        replacements = [
            get_component(AMINO_ACID_COMPONENT_IDS[int(token)])
            for token, selected in zip(
                sampled_tokens,
                design_positions,
                strict=True,
            )
            if selected
        ]
        realized, _, _ = replace_components(
            system,
            torch.tensor(
                selected_components,
                dtype=torch.int64,
                device=system.component_symbol.device,
            ),
            replacements,
        )
        return _repartition_design_entities(system, realized)

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
        r"""
        Design protein sequence candidates in a protein-ligand context.

        Args:
            input: One processed protein-ligand target.
            num_candidates: Number of sequence candidates to sample.
            temperature: Positive sampling temperature.
            coordinate_noise: Standard deviation of Gaussian noise in coordinate units.
            use_side_chain_context: Whether fixed protein side chains contribute context atoms.
            generator: Random generator used for orders, noise, and token sampling.

        Returns:
            Designed molecular systems and per-candidate confidence scores in
            sampling order.
        """

        sampled = self.sample(
            input,
            num_samples=num_candidates,
            temperature=temperature,
            coordinate_noise=coordinate_noise,
            use_side_chain_context=use_side_chain_context,
            generator=generator,
        )
        losses = nll_loss(sampled.sequence_tokens, sampled.log_probabilities)
        designed = (input.residue_mask & input.design_positions)[0]
        interface = designed & input.ligand_interface_mask[0].bool()
        designed_row = designed.to(dtype=losses.dtype)
        interface_row = interface.to(dtype=losses.dtype)
        designed_nll = (losses * designed_row).sum(dim=-1) / designed_row.sum(dim=-1)
        interface_nll = (losses * interface_row).sum(dim=-1) / interface_row.sum(dim=-1)
        designed_confidence = torch.exp(-designed_nll)
        interface_confidence = torch.exp(-interface_nll)
        recovery = (
            (sampled.sequence_tokens == input.sequence_tokens.repeat(num_candidates, 1)).to(dtype=losses.dtype)
            * designed_row
        ).sum(dim=-1) / designed_row.sum(dim=-1)

        component_indices = input.component_index[0].detach().cpu().tolist()
        references = tuple(input.structure.system.component_reference(component) for component in component_indices)
        designed_host = designed.detach().cpu().tolist()
        interface_host = interface.detach().cpu().tolist()
        designed_scope = frozenset(
            reference
            for reference, selected in zip(
                references,
                designed_host,
                strict=True,
            )
            if selected
        )
        interface_scope = frozenset(
            reference
            for reference, selected in zip(
                references,
                interface_host,
                strict=True,
            )
            if selected
        )
        host_sequence_tokens = sampled.sequence_tokens.detach().cpu().tolist()
        designed_confidence_host = designed_confidence.detach().cpu().tolist()
        interface_confidence_host = interface_confidence.detach().cpu().tolist()
        recovery_host = recovery.detach().cpu().tolist()

        candidates = tuple(
            DesignCandidate(
                system=self._realize_sequence(
                    input.structure.system,
                    component_indices,
                    host_sequence_tokens[index],
                    designed_host,
                ),
                scores={
                    "designed_sequence_confidence": Score(
                        float(designed_confidence_host[index]),
                        preferred="higher",
                        scope=designed_scope,
                    ),
                    # A target without a ligand has no interface, and the mean over an
                    # empty scope is not a number, so the score is simply not reported.
                    **(
                        {
                            "ligand_interface_confidence": Score(
                                float(interface_confidence_host[index]),
                                preferred="higher",
                                scope=interface_scope,
                            )
                        }
                        if interface_scope
                        else {}
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


class LigandMpnnFeatures(MpnnProteinFeatures):
    r"""
    Build residue-graph and atom-context features for LigandMPNN.

    Args:
        hidden_size: Width of the per-residue edge features and per-atom context node features.
        num_neighbors: Number of nearest-neighbor residues in the graph.
        num_positional_embeddings: Width of the relative-position embedding.
        num_rbf: Number of radial basis functions per distance feature.
    """

    sidechain_atom_types: Tensor
    periodic_group: Tensor
    periodic_period: Tensor

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
        buffer_device = self.edge_projection.weight.device
        if buffer_device.type == "meta":
            buffer_device = torch.device("cpu")
        self.register_buffer(
            "sidechain_atom_types",
            torch.tensor(
                _SIDECHAIN_ATOMIC_NUMBERS,
                dtype=torch.long,
                device=buffer_device,
            ),
            persistent=False,
        )
        group, period = periodic_group_period(device=buffer_device)
        self.register_buffer("periodic_group", group, persistent=False)
        self.register_buffer("periodic_period", period, persistent=False)
        self.element_projection = nn.Linear(147, 64)
        self.atom_projection = nn.Linear(5 * num_rbf + 64 + 4, hidden_size)
        self.atom_norm = LayerNorm(hidden_size)
        self.context_edge_projection = nn.Linear(num_rbf, hidden_size, bias=False)
        self.context_edge_norm = LayerNorm(hidden_size)
        self.context_node_projection = nn.Linear(147, hidden_size, bias=False)
        self.context_node_norm = LayerNorm(hidden_size)

    def forward(  # type: ignore[override]
        self,
        backbone_coordinates: Tensor,
        residue_mask: Tensor,
        residue_index: Tensor,
        chain_index: Tensor,
        context_coordinates: Tensor,
        context_atomic_numbers: Tensor,
        context_mask: Tensor,
        residue_context_coordinates: Tensor,
        residue_context_mask: Tensor,
        design_positions: Tensor,
        *,
        use_side_chain_context: bool,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        r"""
        Build residue-graph and context-atom features.

        Args:
            backbone_coordinates: N, C-alpha, C, and O coordinates with shape
                ``(B, L, 4, 3)``.
            residue_mask: Valid-residue mask with shape ``(B, L)``.
            residue_index: Residue indices with shape ``(B, L)``.
            chain_index: Chain indices with shape ``(B, L)``.
            context_coordinates: Context coordinates with shape ``(B, L, A, 3)``.
            context_atomic_numbers: Context atomic numbers with shape ``(B, L, A)``.
            context_mask: Context-atom mask with shape ``(B, L, A)``.
            residue_context_coordinates: Side-chain coordinates with shape
                ``(B, L, 32, 3)``.
            residue_context_mask: Side-chain atom mask with shape ``(B, L, 32)``.
            design_positions: Design-position mask with shape ``(B, L)``.
            use_side_chain_context: Whether fixed side chains are added to atom context.

        Returns:
            Context features ``(B, L, A, H)``, residue edges ``(B, L, K, H)``,
            neighbor indices ``(B, L, K)``, context nodes ``(B, L, A, H)``,
            context edges ``(B, L, A, A, H)``, context mask ``(B, L, A)``, and
            valid-neighbor mask ``(B, L, K)``.
        """

        num_context_atoms = context_coordinates.shape[2]
        nitrogen = backbone_coordinates[:, :, 0]
        alpha_carbon = backbone_coordinates[:, :, 1]
        carbon = backbone_coordinates[:, :, 2]
        oxygen = backbone_coordinates[:, :, 3]
        beta_carbon = virtual_beta_carbon(nitrogen, alpha_carbon, carbon)
        edges, neighbor_indices, neighbor_valid = super().forward(
            backbone_coordinates,
            residue_mask,
            residue_index,
            chain_index,
        )
        if use_side_chain_context:
            context_coordinates, context_atomic_numbers, context_mask = self._append_sidechain_context(
                beta_carbon,
                context_coordinates,
                context_atomic_numbers,
                context_mask,
                residue_context_coordinates,
                residue_context_mask,
                design_positions,
                residue_mask,
                neighbor_indices,
                neighbor_valid,
                num_context_atoms,
            )

        context_atomic_numbers = context_atomic_numbers.long()
        element_features = self._element_features(context_atomic_numbers)
        element_projection = self.element_projection(element_features.to(dtype=backbone_coordinates.dtype))

        backbone_atoms = torch.stack(
            (nitrogen, alpha_carbon, carbon, oxygen, beta_carbon),
            dim=2,
        )
        atom_context_distances = torch.sqrt(
            torch.sum((backbone_atoms[:, :, :, None] - context_coordinates[:, :, None]) ** 2, dim=-1) + 1e-6
        )
        atom_context_features = radial_basis(atom_context_distances.movedim(2, 3), self.num_rbf).flatten(start_dim=-2)
        atom_features = torch.cat(
            (
                atom_context_features,
                element_projection,
                local_backbone_frame_features(nitrogen, alpha_carbon, carbon, context_coordinates),
            ),
            dim=-1,
        )
        atom_features = self.atom_norm(self.atom_projection(atom_features))
        context_distances = torch.sqrt(
            torch.sum((context_coordinates[:, :, :, None] - context_coordinates[:, :, None, :]) ** 2, dim=-1) + 1e-6
        )
        context_edges = self.context_edge_norm(
            self.context_edge_projection(radial_basis(context_distances, self.num_rbf))
        )
        context_nodes = self.context_node_norm(
            self.context_node_projection(element_features.to(dtype=backbone_coordinates.dtype))
        )
        return atom_features, edges, neighbor_indices, context_nodes, context_edges, context_mask, neighbor_valid

    def _append_sidechain_context(
        self,
        beta_carbon: Tensor,
        context_coordinates: Tensor,
        context_atomic_numbers: Tensor,
        context_mask: Tensor,
        residue_context_coordinates: Tensor,
        residue_context_mask: Tensor,
        design_positions: Tensor,
        residue_mask: Tensor,
        neighbor_indices: Tensor,
        neighbor_valid: Tensor,
        num_context_atoms: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        r"""
        Append the observed side-chain atoms of the nearest fixed residues, then reselect atoms.

        Only source-fixed rows contribute side-chain atoms; the first sixteen residue KNN neighbors
        supply their observed 32-slot side chains, which are concatenated with the nonprotein context
        before the nearest ``num_context_atoms`` atoms are reselected around the virtual Cβ.
        """

        batch, length = context_mask.shape[:2]
        fixed_mask = residue_context_mask.bool() & ~design_positions[:, :, None].bool()
        neighbor_subset = neighbor_indices[:, :, :16]
        neighbor_subset_valid = neighbor_valid[:, :, :16]
        sidechain_mask = gather_nodes(fixed_mask, neighbor_subset)
        sidechain_mask = sidechain_mask * neighbor_subset_valid[..., None]
        sidechain_mask = sidechain_mask.reshape(batch, length, -1)
        sidechain_coordinates = gather_nodes(
            residue_context_coordinates.reshape(batch, length, -1),
            neighbor_subset,
        ).reshape(batch, length, -1, 3)
        sidechain_types = self.sidechain_atom_types[None, None, None, :].expand(
            batch,
            length,
            neighbor_subset.shape[2],
            -1,
        )
        sidechain_types = sidechain_types.reshape(batch, length, -1)
        context_coordinates = torch.cat((sidechain_coordinates, context_coordinates), dim=2)
        context_mask = torch.cat((sidechain_mask, context_mask), dim=2)
        context_atomic_numbers = torch.cat((sidechain_types, context_atomic_numbers), dim=2)
        distances = torch.sum((beta_carbon[:, :, None] - context_coordinates) ** 2, dim=-1)
        valid = residue_mask[:, :, None].bool() & context_mask.bool()
        maximum = torch.where(valid, distances, 0).amax(dim=-1, keepdim=True)
        adjusted = torch.where(valid, distances, maximum + 1)
        atom_count = min(num_context_atoms, adjusted.shape[-1])
        atom_indices = torch.topk(adjusted, atom_count, dim=-1, largest=False).indices
        context_coordinates = torch.gather(context_coordinates, 2, atom_indices[:, :, :, None].expand(-1, -1, -1, 3))
        context_atomic_numbers = torch.gather(context_atomic_numbers, 2, atom_indices)
        context_mask = torch.gather(context_mask, 2, atom_indices)
        return context_coordinates, context_atomic_numbers, context_mask

    def _element_features(self, atomic_numbers: Tensor) -> Tensor:
        r"""
        One-hot-encode each context atom by element, periodic group, and period.
        """

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


def _repartition_design_entities(
    original: MolecularSystem,
    realized: MolecularSystem,
) -> MolecularSystem:
    r"""
    Partition changed copies by complete output sequence within each input entity.
    """

    instance_entity = original.instance_entity.detach().cpu().tolist()
    component_ptr = realized.instance_component_ptr.detach().cpu().tolist()
    component_symbol = realized.component_symbol.detach().cpu().tolist()
    original_component_ptr = original.instance_component_ptr.detach().cpu().tolist()
    original_component_symbol = original.component_symbol.detach().cpu().tolist()
    entity_ids = [original.entity_id(entity) for entity in range(original.entity_symbol.numel())]
    used_entity_ids = set(entity_ids)
    symbols = list(realized.symbols)
    new_entity_symbols: list[int] = []
    new_entity_types: list[int] = []
    new_entity_molecule_types: list[int] = []
    new_instance_entity = [-1] * len(instance_entity)

    def sequence(
        pointers: list[int],
        symbols: list[int],
        instance: int,
    ) -> tuple[int, ...]:
        return tuple(symbols[pointers[instance] : pointers[instance + 1]])

    for entity, entity_id in enumerate(entity_ids):
        instances = [instance for instance, owner in enumerate(instance_entity) if owner == entity]
        if not instances:
            partitions: list[tuple[tuple[int, ...] | None, list[int]]] = [(None, [])]
        else:
            grouped: dict[tuple[int, ...], list[int]] = {}
            for instance in instances:
                grouped.setdefault(
                    sequence(component_ptr, component_symbol, instance),
                    [],
                ).append(instance)
            original_sequence = sequence(
                original_component_ptr,
                original_component_symbol,
                instances[0],
            )
            primary = original_sequence if original_sequence in grouped else next(iter(grouped))
            partitions = [(primary, grouped.pop(primary)), *grouped.items()]

        suffix = 2
        for partition_index, (_, members) in enumerate(partitions):
            if partition_index == 0:
                output_id = entity_id
            else:
                output_id = f"{entity_id}_{suffix}"
                while output_id in used_entity_ids:
                    suffix += 1
                    output_id = f"{entity_id}_{suffix}"
                suffix += 1
                used_entity_ids.add(output_id)
            if output_id not in symbols:
                symbols.append(output_id)
            new_entity = len(new_entity_symbols)
            new_entity_symbols.append(symbols.index(output_id))
            new_entity_types.append(int(original.entity_type[entity]))
            new_entity_molecule_types.append(int(original.entity_molecule_type[entity]))
            for instance in members:
                new_instance_entity[instance] = new_entity

    return realized.replace(  # type: ignore[attr-defined]
        symbols=tuple(symbols),
        system_entity_ptr=realized.system_entity_ptr.new_tensor([0, len(new_entity_symbols)]),
        entity_symbol=realized.entity_symbol.new_tensor(new_entity_symbols),
        entity_type=realized.entity_type.new_tensor(new_entity_types),
        entity_molecule_type=realized.entity_molecule_type.new_tensor(new_entity_molecule_types),
        instance_entity=realized.instance_entity.new_tensor(new_instance_entity),
    )


# ===========================================================================
# LigandMPNN typed inputs and outputs.
# ===========================================================================


@tensorclass
class LigandMpnnInput:
    r"""
    Inputs for LigandMPNN sequence design and scoring.

    Attributes:
        backbone_coordinates: N, CA, C, and O coordinates with shape
            ``(batch_size, num_residues, 4, 3)``.
        sequence_tokens: Residue token IDs with shape ``(batch_size, num_residues)``.
        residue_mask: Valid-residue mask with shape ``(batch_size, num_residues)``.
        residue_index: Residue indices with shape ``(batch_size, num_residues)``.
        chain_index: Chain indices with shape ``(batch_size, num_residues)``.
        context_coordinates: Nearby context-atom coordinates with shape
            ``(batch_size, num_residues, num_context_atoms, 3)``.
        context_atomic_numbers: Context atomic numbers with shape
            ``(batch_size, num_residues, num_context_atoms)``.
        context_mask: Context-atom mask with shape
            ``(batch_size, num_residues, num_context_atoms)``.
        residue_context_coordinates: Side-chain context coordinates with shape
            ``(batch_size, num_residues, 32, 3)``.
        residue_context_mask: Side-chain context mask with shape
            ``(batch_size, num_residues, 32)``.
        design_positions: Positions eligible for design with shape
            ``(batch_size, num_residues)``.
        ligand_interface_mask: Residues at the ligand interface with shape
            ``(batch_size, num_residues)``.
        structure: Source structure.
        component_index: Canonical component indices aligned to residue rows.
        tied_group_ids: Autoregressive unit ID for each residue row.
        tied_weights: Optional tied-group weights with shape ``(batch_size, num_residues)``.
        allowed_token_mask: Optional token mask with shape
            ``(batch_size, num_residues, vocabulary_size)``.
        logit_bias: Optional additive bias with shape
            ``(batch_size, num_residues, vocabulary_size)``.
    """

    backbone_coordinates: Tensor
    sequence_tokens: Tensor
    residue_mask: Tensor
    residue_index: Tensor
    chain_index: Tensor
    context_coordinates: Tensor
    context_atomic_numbers: Tensor
    context_mask: Tensor
    residue_context_coordinates: Tensor
    residue_context_mask: Tensor
    design_positions: Tensor
    ligand_interface_mask: Tensor
    structure: Structure
    component_index: Tensor
    tied_group_ids: Tensor
    tied_weights: Tensor | None = None
    allowed_token_mask: Tensor | None = None
    logit_bias: Tensor | None = None


class LigandMpnnOutput(_TensorCollection):
    r"""
    Outputs from one LigandMPNN forward pass.

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


class LigandMpnnSamplingOutput(_TensorCollection):
    r"""
    Outputs from LigandMPNN autoregressive sampling.

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
