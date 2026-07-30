r"""
LigandMpnn context-aware sequence-design model.
"""

from __future__ import annotations

import itertools
import math

import torch
import torch.nn.functional as F
from tensordict import tensorclass
from torch import Tensor, nn

from ...data.components import get_component
from ...data.molecular import ComponentReference, MolecularSystem
from ...data.structure import Structure
from ...model import Model
from ...modules.backend import LayerNorm
from ...modules.mpnn import (
    AMINO_ACID_COMPONENT_IDS,
    MpnnDecoderBlock,
    MpnnEncoderBlock,
    MpnnProteinFeatures,
    autoregressive_masks,
    cat_neighbors_nodes,
    decoding_order_to_ranks,
    gather_nodes,
    local_backbone_frame_features,
    nll_loss,
    periodic_group_period,
    radial_basis,
    tied_groups_from_ids,
    validate_positive_integer,
    virtual_beta_carbon,
)
from ...results import DesignCandidate, DesignResult, MpnnProbabilityResult, Score
from .config import LigandMpnnConfig

_VOCABULARY = len(AMINO_ACID_COMPONENT_IDS) + 1
_GENERATABLE = len(AMINO_ACID_COMPONENT_IDS)


class LigandMpnnModel(Model):
    r"""
    Design protein sequences in protein-ligand contexts with LigandMPNN.

    Accepts [`LigandMpnnInput`][ofoldx.models.ligandmpnn.model.LigandMpnnInput] and returns
    [`LigandMpnnOutput`][ofoldx.models.ligandmpnn.model.LigandMpnnOutput].

    Args:
        config: Model architecture and inference configuration.

    Examples:
        >>> import numpy as np
        >>> import torch
        >>> from ofoldx.data import (
        ...     AtomSite, ChemicalAtom, ChemicalComponent, MolecularSystem,
        ...     Molecule, PolymerSequence, Structure,
        ... )
        >>> from ofoldx.models.ligandmpnn import (
        ...     LigandMpnnConfig, LigandMpnnModel, LigandMpnnProcessor,
        ...     LigandMpnnProcessorConfig,
        ... )
        >>> sequence = PolymerSequence.from_one_letter("A")
        >>> ligand = Molecule((ChemicalComponent(
        ...     "LIG", (ChemicalAtom("C1", "C"),), frozenset(),
        ...     reference_coordinates=((0.0, 0.0, 0.0),),
        ... ),))
        >>> system = MolecularSystem(
        ...     {"protein": sequence, "ligand": ligand},
        ...     {"A": "protein", "L": "ligand"},
        ... )
        >>> sites = tuple(
        ...     [AtomSite("A", 0, atom) for atom in ("N", "CA", "C", "O")]
        ...     + [AtomSite("L", 0, "C1")]
        ... )
        >>> coordinates = np.array(
        ...     [
        ...         [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.5, 0.0],
        ...         [1.0, 1.0, 0.0], [0.0, 3.0, 0.0],
        ...     ],
        ...     dtype=np.float32,
        ... )
        >>> structure = Structure(system, sites, coordinates, np.ones(len(sites), dtype=np.bool_))
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

    def __init__(self, config: LigandMpnnConfig | None = None) -> None:
        config = LigandMpnnConfig() if config is None else config
        super().__init__(config)
        hidden_size = self.config.hidden_size
        # Cache the one live width as a plain int: the encoder hot path reads it under
        # torch.compile(..., fullgraph=True), where chanfig's Config attribute access does not trace.
        self.hidden_size = hidden_size
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

    # -- stochastic and validation controls ------------------------------------------------------

    @staticmethod
    def _validate_coordinate_noise(coordinate_noise: float) -> float:
        r"""
        Coerce and validate the inference-time coordinate perturbation.
        """

        coordinate_noise = float(coordinate_noise)
        if not math.isfinite(coordinate_noise) or coordinate_noise < 0:
            raise ValueError("coordinate_noise must be a finite nonnegative number")
        return coordinate_noise

    @staticmethod
    def _require_single_target(input: LigandMpnnInput) -> None:
        r"""
        Reject an execution-batch axis on one-target biological operations.
        """

        if input.backbone_coordinates.shape[0] != 1:
            raise ValueError("this LigandMpnn operation returns one target and therefore accepts one input")

    @staticmethod
    def _repeat_input(input: LigandMpnnInput, count: int) -> LigandMpnnInput:
        r"""
        Repeat one biological input into ``count`` aligned evaluation rows.
        """

        def repeat(value: Tensor | None) -> Tensor | None:
            return None if value is None else value.repeat_interleave(count, dim=0)

        return LigandMpnnInput(
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
            component_references=input.component_references,
            tied_group_ids=repeat(input.tied_group_ids),
            tied_weights=repeat(input.tied_weights),
            allowed_token_mask=repeat(input.allowed_token_mask),
            residue_logit_bias=repeat(input.residue_logit_bias),
            batch_size=[input.backbone_coordinates.shape[0] * count],
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

        return MpnnProbabilityResult(
            target=input.structure,
            component_references=input.component_references,
            sequence_tokens=input.sequence_tokens[0],
            residue_mask=input.residue_mask[0],
            design_positions=input.design_positions[0],
            logits=logits,
            decoding_ranks=decoding_ranks,
        )

    # -- decode-order generation -----------------------------------------------------------------

    @staticmethod
    def _draw_decode_orders(
        residue_mask: Tensor,
        design_positions: Tensor,
        generator: torch.Generator | None,
        decoding_priority: Tensor | None = None,
    ) -> Tensor:
        r"""
        Draw the fixed-first decoding orders directly from aligned masks.

        Fixed rows receive a near-zero weight and therefore decode before every designed row; each row
        of the returned ``[B, L]`` permutation is an independent order.

        ``decoding_priority`` may provide nonnegative per-residue priorities
        explicitly; otherwise they are drawn from ``generator`` as
        ``randn.abs()``. The aligned masks keep ragged batches free of padded rows.
        """

        chain_mask = residue_mask.to(dtype=torch.float32) * design_positions.to(dtype=torch.float32)
        if decoding_priority is None:
            decoding_priority = torch.randn(chain_mask.shape, device=chain_mask.device, generator=generator).abs()
        return torch.argsort((chain_mask + 0.0001) * decoding_priority, dim=-1)

    def _generate_ranks(self, input: LigandMpnnInput, generator: torch.Generator | None) -> Tensor:
        r"""
        Draw fixed-first causal ranks, collapsing every tied group to one consecutive unit.
        """

        decode_order = self._draw_decode_orders(input.residue_mask, input.design_positions, generator)
        tied_groups = tied_groups_from_ids(input.tied_group_ids)[0]
        if tied_groups:
            collapsed = _collapse_tied_order(decode_order[0], tied_groups)
            decode_order = collapsed.unsqueeze(0).repeat(decode_order.shape[0], 1)
        return decoding_order_to_ranks(decode_order)

    # -- shared graph passes ---------------------------------------------------------------------

    def _encode(
        self,
        input: LigandMpnnInput,
        *,
        coordinate_noise: float,
        use_sidechain_context: bool,
        generator: torch.Generator | None,
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
        ) = self.feature_encoder.encode(
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
            use_sidechain_context=use_sidechain_context,
        )
        # Zero node states with the residue structure of ``edges`` (one neighbor slot, edge width, which
        # equals ``hidden_size``): deriving them from a ragged slice keeps a NestedTensor batch ragged,
        # where ``new_zeros`` with an explicit ``(B, L, hidden)`` shape would materialize a dense, padded
        # tensor that no longer aligns with the edges.
        node_states = torch.zeros_like(edges[:, :, 0, :])
        edge_states = self.edge_embedding(edges)
        atom_context = self.atom_context_embedding(atom_features)
        for block in self.encoder_blocks:
            node_states, edge_states = block(node_states, edge_states, neighbor_indices, mask, attend_mask)
        residue_context = self.residue_context_input(node_states)
        atom_pair_mask = atom_mask[:, :, :, None] * atom_mask[:, :, None, :]
        atom_nodes = self.atom_node_embedding(atom_nodes)
        atom_edges = self.atom_edge_embedding(atom_edges)
        for block_index, block in enumerate(self.context_blocks):
            atom_nodes = self.atom_context_blocks[block_index](atom_nodes, atom_edges, atom_mask, atom_pair_mask)
            residue_context = block(residue_context, torch.cat((atom_context, atom_nodes), dim=-1), mask, atom_mask)
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
        for block in self.decoder_blocks:
            decoder_context = cat_neighbors_nodes(node_states, sequence_edges, neighbor_indices)
            node_states = block(node_states, backward_mask * decoder_context + forward_context, mask)
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
        for block in self.decoder_blocks:
            states = block(states, forward_context, mask)
        return self.output_projection(states)

    # -- public operations -----------------------------------------------------------------------

    def encode(
        self,
        input: LigandMpnnInput,
        *,
        coordinate_noise: float = 0.0,
        use_sidechain_context: bool = False,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        r"""
        Encode a batch of protein-ligand inputs as residue graph states.

        Args:
            input: LigandMPNN tensors with batch and residue axes.
            coordinate_noise: Standard deviation of Gaussian noise in coordinate units.
            use_sidechain_context: Whether fixed protein side chains contribute context atoms.
            generator: Random generator used for coordinate noise.

        Returns:
            Node states with shape ``(B, L, H)``, edge states with shape
            ``(B, L, K, H)``, neighbor indices with shape ``(B, L, K)``, and the
            real-neighbor mask with shape ``(B, L, K)``.
        """

        coordinate_noise = self._validate_coordinate_noise(coordinate_noise)
        return self._encode(
            input,
            coordinate_noise=coordinate_noise,
            use_sidechain_context=use_sidechain_context,
            generator=generator,
        )

    def forward(
        self,
        input: LigandMpnnInput,
        *,
        decoding_ranks: Tensor | None = None,
        decoding_priority: Tensor | None = None,
        coordinate_noise: float = 0.0,
        use_sidechain_context: bool = False,
        generator: torch.Generator | None = None,
    ) -> LigandMpnnOutput:
        r"""
        Predict residue logits conditioned on the input sequence and decoding order.

        Args:
            input: LigandMPNN tensors with batch size ``B`` and ``L`` residue rows.
            decoding_ranks: Optional ranks with shape ``(B, L)``; smaller values
                decode earlier. A fixed-first random order is drawn when omitted.
            decoding_priority: Optional per-residue nonnegative random priority
                ``(B, L)`` used to draw the order. Ignored when
                ``decoding_ranks`` is given.
            coordinate_noise: Standard deviation of Gaussian noise in coordinate units.
            use_sidechain_context: Whether fixed protein side chains contribute context atoms.
            generator: Random generator used for the order (when no priority is supplied) and coordinate noise.

        Returns:
            Vocabulary logits and the decoding ranks used for the forward pass.
        """

        coordinate_noise = self._validate_coordinate_noise(coordinate_noise)
        node_states, edge_states, neighbor_indices, neighbor_valid = self._encode(
            input,
            coordinate_noise=coordinate_noise,
            use_sidechain_context=use_sidechain_context,
            generator=generator,
        )
        if decoding_ranks is None:
            decoding_ranks = decoding_order_to_ranks(
                self._draw_decode_orders(input.residue_mask, input.design_positions, generator, decoding_priority)
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
        return LigandMpnnOutput(logits=logits, decoding_ranks=decoding_ranks, batch_size=[logits.shape[0]])

    def autoregressive_probabilities(
        self,
        input: LigandMpnnInput,
        *,
        num_decoding_orders: int = 1,
        decoding_ranks: Tensor | None = None,
        coordinate_noise: float = 0.0,
        use_sidechain_context: bool = False,
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
            use_sidechain_context: Whether fixed protein side chains contribute context atoms.
            generator: Random generator used for orders and coordinate noise.

        Returns:
            Logits and ranks whose leading axis enumerates decoding orders.
        """

        self._require_single_target(input)
        num_decoding_orders = validate_positive_integer(num_decoding_orders, "num_decoding_orders")
        coordinate_noise = self._validate_coordinate_noise(coordinate_noise)
        repeated = self._repeat_input(input, num_decoding_orders)
        node_states, edge_states, neighbor_indices, neighbor_valid = self._encode(
            repeated,
            coordinate_noise=coordinate_noise,
            use_sidechain_context=use_sidechain_context,
            generator=generator,
        )
        ranks = decoding_ranks if decoding_ranks is not None else self._generate_ranks(repeated, generator)
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
        use_sidechain_context: bool = False,
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
            use_sidechain_context: Whether fixed protein side chains contribute context atoms.
            generator: Random generator used for orders and coordinate noise.

        Returns:
            Sequence-free logits and ranks whose leading axis enumerates decoding
            orders.
        """

        self._require_single_target(input)
        num_decoding_orders = validate_positive_integer(num_decoding_orders, "num_decoding_orders")
        coordinate_noise = self._validate_coordinate_noise(coordinate_noise)
        repeated = self._repeat_input(input, num_decoding_orders)
        node_states, edge_states, neighbor_indices, neighbor_valid = self._encode(
            repeated,
            coordinate_noise=coordinate_noise,
            use_sidechain_context=use_sidechain_context,
            generator=generator,
        )
        ranks = decoding_ranks if decoding_ranks is not None else self._generate_ranks(repeated, generator)
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
        use_sidechain_context: bool,
        generator: torch.Generator | None,
        move_last: bool,
    ) -> MpnnProbabilityResult:
        r"""
        Evaluate every target moved exactly last or first under one base decoding order.
        """

        self._require_single_target(input)
        num_decoding_orders = validate_positive_integer(num_decoding_orders, "num_decoding_orders")
        coordinate_noise = self._validate_coordinate_noise(coordinate_noise)
        repeated = self._repeat_input(input, num_decoding_orders)
        device = repeated.backbone_coordinates.device
        node_states, edge_states, neighbor_indices, neighbor_valid = self._encode(
            repeated,
            coordinate_noise=coordinate_noise,
            use_sidechain_context=use_sidechain_context,
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
                ranks = decoding_order_to_ranks(decode_order)
            else:
                ranks = base_ranks.clone()
                ranks[:, target] = last_ranks if move_last else -1
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
        use_sidechain_context: bool = False,
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
            use_sidechain_context: Whether fixed protein side chains contribute context atoms.
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
            use_sidechain_context=use_sidechain_context,
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
        use_sidechain_context: bool = False,
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
            use_sidechain_context: Whether fixed protein side chains contribute context atoms.
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
            use_sidechain_context=use_sidechain_context,
            generator=generator,
            move_last=False,
        )

    # -- sampling --------------------------------------------------------------------------------

    @staticmethod
    def _sampling_bias(input: LigandMpnnInput) -> Tensor:
        r"""
        Return the aligned soft logit bias.
        """

        if input.residue_logit_bias is None:
            bias = torch.zeros(
                input.sequence_tokens.shape[0],
                input.sequence_tokens.shape[1],
                _VOCABULARY,
                device=input.sequence_tokens.device,
                dtype=input.backbone_coordinates.dtype,
            )
        else:
            bias = input.residue_logit_bias
        return bias

    @staticmethod
    def _sampling_probabilities(
        logits: Tensor,
        bias: Tensor,
        allowed_token_mask: Tensor | None,
        temperature: float,
    ) -> Tensor:
        r"""
        Apply soft bias first and hard token support last.
        """

        adjusted = logits + bias
        if allowed_token_mask is not None:
            adjusted = adjusted.masked_fill(~allowed_token_mask, -torch.inf)
        probabilities = F.softmax(adjusted / temperature, dim=-1)[..., :_GENERATABLE]
        return probabilities / probabilities.sum(dim=-1, keepdim=True)

    @torch.inference_mode()
    def sample(
        self,
        input: LigandMpnnInput,
        *,
        num_samples: int = 1,
        temperature: float = 0.1,
        coordinate_noise: float = 0.0,
        use_sidechain_context: bool = False,
        decoding_ranks: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> LigandMpnnSamplingOutput:
        r"""
        Sample protein sequences under the processed design constraints.

        Args:
            input: One processed protein-ligand target.
            num_samples: Number of sequences to sample.
            temperature: Positive sampling temperature.
            coordinate_noise: Standard deviation of Gaussian noise in coordinate units.
            use_sidechain_context: Whether fixed protein side chains contribute context atoms.
            decoding_ranks: Optional ranks with shape ``(num_samples, num_residues)``.
            generator: Random generator used for orders, noise, and token sampling.

        Returns:
            Sampled token IDs, model log probabilities, sampling probabilities,
            and decoding ranks, each with a leading ``num_samples`` axis.
        """

        self._require_single_target(input)
        num_samples = validate_positive_integer(num_samples, "num_samples")
        temperature = float(temperature)
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be a finite positive number")
        coordinate_noise = self._validate_coordinate_noise(coordinate_noise)

        repeated = self._repeat_input(input, num_samples)
        node_states, edge_states, neighbor_indices, neighbor_valid = self._encode(
            repeated,
            coordinate_noise=coordinate_noise,
            use_sidechain_context=use_sidechain_context,
            generator=generator,
        )
        residue_mask = repeated.residue_mask.to(dtype=node_states.dtype)
        chain_mask = residue_mask * repeated.design_positions.to(dtype=node_states.dtype)
        bias = self._sampling_bias(repeated)
        allowed_token_mask = repeated.allowed_token_mask
        if decoding_ranks is None:
            decode_order = self._draw_decode_orders(repeated.residue_mask, repeated.design_positions, generator)
        else:
            decode_order = torch.argsort(decoding_ranks, dim=-1)

        tied_groups = tied_groups_from_ids(input.tied_group_ids)[0]
        if not tied_groups:
            result = self._sample_independent(
                node_states,
                edge_states,
                neighbor_indices,
                neighbor_valid,
                repeated.sequence_tokens,
                residue_mask,
                chain_mask,
                bias,
                allowed_token_mask,
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
                neighbor_valid,
                repeated.sequence_tokens,
                residue_mask,
                chain_mask,
                bias,
                allowed_token_mask,
                temperature,
                decode_order,
                tied_groups,
                weights,
                generator,
            )
        return LigandMpnnSamplingOutput(
            sequence_tokens=result["S"],
            log_probabilities=result["log_probs"],
            sampling_probabilities=result["sampling_probs"],
            decoding_ranks=decoding_order_to_ranks(result["decoding_order"]),
            batch_size=[num_samples],
        )

    def _sample_independent(
        self,
        node_states: Tensor,
        edge_states: Tensor,
        neighbor_indices: Tensor,
        neighbor_valid: Tensor,
        sequence_true: Tensor,
        residue_mask: Tensor,
        chain_mask: Tensor,
        bias: Tensor,
        allowed_token_mask: Tensor | None,
        temperature: float,
        decode_order: Tensor,
        generator: torch.Generator | None,
    ) -> dict[str, Tensor]:
        r"""
        Sample tokens position-by-position with no tied constraints.
        """

        num_samples, length = sequence_true.shape
        device = sequence_true.device
        sampled = _GENERATABLE * torch.ones(num_samples, length, dtype=torch.long, device=device)
        token_states = torch.zeros_like(node_states)
        block_states = [node_states] + [torch.zeros_like(node_states) for _ in self.decoder_blocks]
        probabilities = node_states.new_zeros(num_samples, length, _GENERATABLE)
        log_probs_out = node_states.new_zeros(num_samples, length, _VOCABULARY)
        logits_out = node_states.new_zeros(num_samples, length, _VOCABULARY)
        backward_mask, forward_mask = autoregressive_masks(
            decoding_order_to_ranks(decode_order),
            neighbor_indices,
            residue_mask,
            neighbor_valid.to(dtype=node_states.dtype),
        )
        encoder_edges = cat_neighbors_nodes(torch.zeros_like(token_states), edge_states, neighbor_indices)
        encoder_context = cat_neighbors_nodes(node_states, encoder_edges, neighbor_indices)
        forward_context = forward_mask * encoder_context

        for step in range(length):
            residue = decode_order[:, step]
            chain_mask_t = torch.gather(chain_mask, 1, residue[:, None])[:, 0]
            mask_t = torch.gather(residue_mask, 1, residue[:, None])
            bias_t = torch.gather(bias, 1, residue[:, None, None].repeat(1, 1, bias.shape[-1]))[:, 0, :]
            support_t = (
                None
                if allowed_token_mask is None
                else torch.gather(allowed_token_mask, 1, residue[:, None, None].expand(-1, 1, _VOCABULARY))[:, 0]
            )
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
            for block_index, block in enumerate(self.decoder_blocks):
                decoder_context = cat_neighbors_nodes(block_states[block_index], edge_token_t, neighbor_idx_t)
                node_t = torch.gather(
                    block_states[block_index],
                    1,
                    residue[:, None, None].repeat(1, 1, block_states[block_index].shape[-1]),
                )
                updated = block(node_t, backward_t * decoder_context + forward_t, mask_t)
                block_states[block_index + 1].scatter_(
                    1,
                    residue[:, None, None].repeat(1, 1, node_states.shape[-1]),
                    updated,
                )

            final_t = torch.gather(
                block_states[-1],
                1,
                residue[:, None, None].repeat(1, 1, block_states[-1].shape[-1]),
            )[:, 0]
            logits = self.output_projection(final_t)
            log_probs = F.log_softmax(logits, dim=-1)
            sample_probs = self._sampling_probabilities(logits, bias_t, support_t, temperature)
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
        neighbor_valid: Tensor,
        sequence_true: Tensor,
        residue_mask: Tensor,
        chain_mask: Tensor,
        bias: Tensor,
        allowed_token_mask: Tensor | None,
        temperature: float,
        decode_order: Tensor,
        tied_groups: tuple[tuple[int, ...], ...],
        tied_weights: Tensor,
        generator: torch.Generator | None,
    ) -> dict[str, Tensor]:
        r"""
        Sample tokens with each tied group drawn as one weighted decision.

        Neural logits combine as ``sum(tied_weight[i] * logits[i])`` with a default coefficient of one;
        the aligned bias applies once at the shared group value and hard support intersects across
        members last. One group distribution is copied to every member.
        """

        num_samples, length = sequence_true.shape
        device = sequence_true.device
        sampled = _GENERATABLE * torch.ones(num_samples, length, dtype=torch.long, device=device)
        token_states = torch.zeros_like(node_states)
        block_states = [node_states] + [torch.zeros_like(node_states) for _ in self.decoder_blocks]
        probabilities = node_states.new_zeros(num_samples, length, _GENERATABLE)
        log_probs_out = node_states.new_zeros(num_samples, length, _VOCABULARY)
        logits_out = node_states.new_zeros(num_samples, length, _VOCABULARY)
        backward_mask, forward_mask = autoregressive_masks(
            decoding_order_to_ranks(decode_order),
            neighbor_indices,
            residue_mask,
            neighbor_valid.to(dtype=node_states.dtype),
        )
        encoder_edges = cat_neighbors_nodes(torch.zeros_like(token_states), edge_states, neighbor_indices)
        encoder_context = cat_neighbors_nodes(node_states, encoder_edges, neighbor_indices)
        forward_context = forward_mask * encoder_context

        # LigandMPNN DELIBERATELY decodes a tied group as consecutive per-member ranks (each member gets
        # its own distinct rank via ``_collapse_tied_order``/``decoding_order_to_ranks``), not one shared
        # rank. This faithfully mirrors upstream LigandMPNN ``model_utils.py``, whose tied sampling walks a
        # flat distinct-rank permutation and evaluates members one after another. It is why this differs
        # from ProteinMPNN's shared-rank tied decode -- do NOT "unify" them, or upstream parity breaks.
        ordered_groups = _decode_groups(decode_order[0], [list(group) for group in tied_groups])
        for residue_group in ordered_groups:
            total_logits = node_states.new_zeros(num_samples, _VOCABULARY)
            bias_group = bias[:, residue_group[-1]]
            support_group = None if allowed_token_mask is None else allowed_token_mask[:, residue_group].all(dim=1)
            for residue in residue_group:
                chain_mask_t = chain_mask[:, residue]
                mask_t = residue_mask[:, residue]
                edge_t = edge_states[:, residue : residue + 1]
                neighbor_idx_t = neighbor_indices[:, residue : residue + 1]
                edge_token_t = cat_neighbors_nodes(token_states, edge_t, neighbor_idx_t)
                forward_t = forward_context[:, residue : residue + 1]
                for block_index, block in enumerate(self.decoder_blocks):
                    decoder_context = cat_neighbors_nodes(block_states[block_index], edge_token_t, neighbor_idx_t)
                    node_t = block_states[block_index][:, residue : residue + 1]
                    context_t = backward_mask[:, residue : residue + 1] * decoder_context + forward_t
                    block_states[block_index + 1][:, residue : residue + 1] = block(node_t, context_t, mask_t[:, None])
                logits = self.output_projection(block_states[-1][:, residue])
                log_probs = F.log_softmax(logits, dim=-1)
                log_probs_out[:, residue] = chain_mask_t[:, None] * log_probs
                logits_out[:, residue] = chain_mask_t[:, None] * logits
                total_logits = total_logits + tied_weights[:, residue, None] * logits

            sample_probs = self._sampling_probabilities(total_logits, bias_group, support_group, temperature)
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

    @staticmethod
    def _realize_sequence(
        system: MolecularSystem,
        component_references: tuple[ComponentReference | None, ...],
        sampled_tokens: list[int],
        design_positions: list[bool],
    ) -> MolecularSystem:
        r"""
        Realize one candidate while preserving entity and bond identity.
        """

        replacements = {
            reference: get_component(AMINO_ACID_COMPONENT_IDS[int(token)])
            for reference, token, selected in zip(component_references, sampled_tokens, design_positions)
            if reference is not None and selected
        }
        return system.replace_components(replacements)

    @torch.inference_mode()
    def design(
        self,
        input: LigandMpnnInput,
        *,
        num_candidates: int = 1,
        temperature: float = 0.1,
        coordinate_noise: float = 0.0,
        use_sidechain_context: bool = False,
        generator: torch.Generator | None = None,
    ) -> DesignResult:
        r"""
        Design protein sequence candidates in a protein-ligand context.

        Args:
            input: One processed protein-ligand target.
            num_candidates: Number of sequence candidates to sample.
            temperature: Positive sampling temperature.
            coordinate_noise: Standard deviation of Gaussian noise in coordinate units.
            use_sidechain_context: Whether fixed protein side chains contribute context atoms.
            generator: Random generator used for orders, noise, and token sampling.

        Returns:
            Designed molecular systems and per-candidate confidence scores in
            sampling order.
        """

        self._require_single_target(input)
        num_candidates = validate_positive_integer(num_candidates, "num_candidates")
        temperature = float(temperature)
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be a finite positive number")
        coordinate_noise = self._validate_coordinate_noise(coordinate_noise)
        sampled = self.sample(
            input,
            num_samples=num_candidates,
            temperature=temperature,
            coordinate_noise=coordinate_noise,
            use_sidechain_context=use_sidechain_context,
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

        references = input.component_references
        designed_host = designed.detach().cpu().tolist()
        interface_host = interface.detach().cpu().tolist()
        designed_scope = frozenset(
            reference for reference, selected in zip(references, designed_host) if reference is not None and selected
        )
        interface_scope = frozenset(
            reference for reference, selected in zip(references, interface_host) if reference is not None and selected
        )
        host_sequence_tokens = sampled.sequence_tokens.detach().cpu().tolist()
        designed_confidence_host = designed_confidence.detach().cpu().tolist()
        interface_confidence_host = interface_confidence.detach().cpu().tolist()
        recovery_host = recovery.detach().cpu().tolist()

        candidates = tuple(
            DesignCandidate(
                system=self._realize_sequence(
                    input.structure.system,
                    references,
                    host_sequence_tokens[index],
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


class LigandMpnnFeatures(MpnnProteinFeatures):
    r"""
    Build residue-graph and atom-context features for LigandMPNN.

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
        buffer_device = self.edge_projection.weight.device
        if buffer_device.type == "meta":
            buffer_device = torch.device("cpu")
        self.register_buffer(
            "sidechain_atom_types",
            _sidechain_atom_types(device=buffer_device),
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

    def encode(
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
        use_sidechain_context: bool,
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
            use_sidechain_context: Whether fixed side chains are added to atom context.

        Returns:
            Context features ``(B, L, A, H)``, residue edges ``(B, L, K, H)``,
            neighbor indices ``(B, L, K)``, context nodes ``(B, L, A, H)``,
            context edges ``(B, L, A, A, H)``, context mask ``(B, L, A)``, and
            valid-neighbor mask ``(B, L, K)``.
        """

        num_context_atoms = context_coordinates.shape[2]
        atoms = _backbone_atoms(backbone_coordinates)
        edges, neighbor_indices, neighbor_valid = super().forward(
            backbone_coordinates,
            residue_mask,
            residue_index,
            chain_index,
        )
        if use_sidechain_context:
            context_coordinates, context_atomic_numbers, context_mask = self._append_sidechain_context(
                atoms["CB"],
                context_coordinates,
                context_atomic_numbers,
                context_mask,
                residue_context_coordinates,
                residue_context_mask,
                design_positions,
                residue_mask,
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
        atom_context_features = radial_basis(atom_context_distances.movedim(2, 3), self.num_rbf).flatten(start_dim=-2)
        atom_features = torch.cat(
            (
                atom_context_features,
                element_projection,
                local_backbone_frame_features(atoms["N"], atoms["CA"], atoms["C"], context_coordinates),
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
        num_context_atoms: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        r"""
        Append the observed side-chain atoms of the nearest fixed residues, then reselect atoms.

        Only source-fixed rows contribute side-chain atoms; the first sixteen residue KNN neighbors
        supply their observed 32-slot side chains, which are concatenated with the nonprotein context
        before the nearest ``num_context_atoms`` atoms are reselected around the virtual Cβ.
        """

        batch, length = context_mask.shape[:2]
        fixed_mask = residue_context_mask * (1 - design_positions[:, :, None])
        neighbor_subset = neighbor_indices[:, :, :16]
        sidechain_mask = gather_nodes(fixed_mask, neighbor_subset).reshape(batch, length, -1)
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
        valid = residue_mask[:, :, None] * context_mask
        adjusted = distances * valid + (1.0 - valid) * 10000.0
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


def _sidechain_atom_types(*, device: torch.device | str | None = None) -> Tensor:
    r"""
    Return atomic numbers for the atom37 side-chain slots.

    Args:
        device: Device on which to create the lookup tensor.

    Returns:
        Atomic numbers for atom37 slots ``5:37`` with shape ``(32,)``.
    """

    return torch.tensor(
        [6, 6, 6, 8, 8, 16, 6, 6, 6, 7, 7, 8, 8, 16, 6, 6, 6, 6, 7, 7, 7, 8, 8, 6, 7, 7, 8, 6, 6, 6, 7, 8],
        dtype=torch.long,
        device=device,
    )


def _backbone_atoms(coordinates: Tensor) -> dict[str, Tensor]:
    r"""
    Split N/CA/C/O coordinates and add the idealized virtual Cβ atom.
    """

    nitrogen = coordinates[:, :, 0]
    alpha_carbon = coordinates[:, :, 1]
    carbon = coordinates[:, :, 2]
    oxygen = coordinates[:, :, 3]
    beta_carbon = virtual_beta_carbon(nitrogen, alpha_carbon, carbon)
    return {"N": nitrogen, "CA": alpha_carbon, "C": carbon, "O": oxygen, "CB": beta_carbon}


def _decode_groups(decode_order: Tensor, tied_groups: list[list[int]]) -> list[list[int]]:
    r"""
    Order tied groups by the first time any member appears in the decode order.
    """

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
    r"""
    Collapse a drawn order so every tied group decodes consecutively from its first member.

    The members stay as a flat run of DISTINCT consecutive positions (each keeps its own rank once
    :func:`decoding_order_to_ranks` runs), NOT one shared rank -- this faithfully mirrors upstream
    LigandMPNN ``model_utils.py`` and is why LigandMPNN's tied decode differs from ProteinMPNN's
    shared-rank scheme. Do not "unify" them, or upstream parity breaks.
    """

    ordered = _decode_groups(decode_order_row, [list(group) for group in tied_groups])
    return torch.tensor(
        list(itertools.chain.from_iterable(ordered)),
        device=decode_order_row.device,
        dtype=torch.long,
    )


# ---------------------------------------------------------------------------
# LigandMPNN typed inputs and outputs.
# ---------------------------------------------------------------------------


@tensorclass
class LigandMpnnInput:
    r"""
    Inputs for LigandMPNN sequence design and scoring.

    Args:
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
        component_references: Component occurrences aligned to residue rows.
        tied_group_ids: Autoregressive unit ID for each residue row.
        tied_weights: Optional tied-group weights with shape ``(batch_size, num_residues)``.
        allowed_token_mask: Optional token mask with shape
            ``(batch_size, num_residues, vocabulary_size)``.
        residue_logit_bias: Optional additive bias with shape
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
    component_references: tuple[ComponentReference, ...]
    tied_group_ids: Tensor
    tied_weights: Tensor | None = None
    allowed_token_mask: Tensor | None = None
    residue_logit_bias: Tensor | None = None


@tensorclass
class LigandMpnnOutput:
    r"""
    Outputs from one LigandMPNN forward pass.

    Args:
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


@tensorclass
class LigandMpnnSamplingOutput:
    r"""
    Outputs from LigandMPNN autoregressive sampling.

    Args:
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
