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
LigandMpnn context-aware sequence-design model.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch
import torch.nn.functional as F
from tensordict import TensorClass, tensorclass
from torch import Tensor, nn

from ...data.molecular import GROUP_BY_ATOMIC_NUMBER, PERIOD_BY_ATOMIC_NUMBER
from ...data.tokenization import TOKEN_INDEX, TOKENS
from ...errors import RequestRef, fail
from ...model import Model, TensorCollection
from ...modules.backend import LayerNorm, gather_nodes
from ...modules.functional import radial_basis
from ...modules.geometry import local_backbone_frame_features, smoothed_vector_norm, virtual_beta_carbon
from ...modules.mpnn import (
    MpnnDecoderBlock,
    MpnnEncoderBlock,
    MpnnProteinFeatureEncoder,
    autoregressive_masks,
    cat_neighbors_nodes,
    decode_sequence_free,
    validate_decoding_ranks,
)
from ..processing import project_model_categories, project_model_coordinates
from .config import LigandMpnnConfig

#: The classes the released checkpoints were trained on, in their upstream order (canonical spelling); every other
#: canonical class has logit ``-inf``, so identical inputs give the upstream distribution.
LIGANDMPNN_VOCABULARY: tuple[str, ...] = tuple("ACDEFGHIKLMNPQRSTVWYX")
#: The classes sampling draws from: the trained vocabulary without ``X`` (upstream samples the first 20 only).
LIGANDMPNN_GENERATABLE: tuple[str, ...] = LIGANDMPNN_VOCABULARY[:-1]


def _invalid(argument: str, value: object) -> None:
    fail("value_invalid", RequestRef(argument, str(value)))


def normalize_ligandmpnn_input(input: LigandMpnnInput, *, dtype: torch.dtype | None = None) -> LigandMpnnInput:
    """Return finite available model coordinates without mutating the supplied input."""
    result = cast(LigandMpnnInput, cast(TensorClass, input).clone(recurse=False))
    points = input.backbone_coordinates
    observed = input.residue_mask.unsqueeze(-1) & torch.ones_like(points[..., 0], dtype=torch.bool)
    result.backbone_coordinates = project_model_coordinates(points, observed, dtype=dtype, feature="protein backbone")
    result.context_coordinates = project_model_coordinates(
        input.context_coordinates, input.context_mask, dtype=dtype, feature="ligand context"
    )
    result.residue_context_coordinates = project_model_coordinates(
        input.residue_context_coordinates, input.residue_context_mask, dtype=dtype, feature="side-chain context"
    )
    return result


class LigandMpnnModel(Model):
    r"""
    Design protein sequences in protein-ligand contexts with LigandMPNN.

    Accepts [`LigandMpnnInput`][ofoldx.models.ligandmpnn.model.LigandMpnnInput] and returns
    [`LigandMpnnOutput`][ofoldx.models.ligandmpnn.model.LigandMpnnOutput].

    Args:
        config: Model architecture and inference configuration.
    """

    config_class = LigandMpnnConfig
    config: LigandMpnnConfig
    _gradient_checkpointing_func: Callable[..., object]

    def __init__(self, config: LigandMpnnConfig | None = None) -> None:
        config = LigandMpnnConfig() if config is None else config
        super().__init__(config)
        self.gradient_checkpointing = False
        hidden_size = self.config.hidden_size
        self.feature_encoder = LigandMpnnFeatureEncoder(hidden_size, self.config.num_neighbors)
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
        self.token_embedding = nn.Embedding(len(TOKENS), hidden_size)
        self.decoder_blocks = nn.ModuleList(
            MpnnDecoderBlock(hidden_size, hidden_size * 3, dropout=self.config.dropout)
            for _ in range(self.config.num_decoder_blocks)
        )
        self.output_projection = nn.Linear(hidden_size, len(TOKENS))
        trained = torch.zeros(len(TOKENS), dtype=torch.bool)
        trained[[TOKEN_INDEX[token] for token in LIGANDMPNN_VOCABULARY]] = True
        generatable = torch.tensor([TOKEN_INDEX[token] for token in LIGANDMPNN_GENERATABLE])
        self.register_buffer("trained_classes", trained, persistent=False)
        self.register_buffer("generatable_classes", generatable, persistent=False)
        self.post_init()

    def _logits(self, states: Tensor) -> Tensor:
        r"""Vocabulary logits over the canonical classes; classes outside the trained vocabulary are ``-inf``."""
        return self.output_projection(states).masked_fill(~self.trained_classes, -torch.inf)

    def initialize_parameters(self) -> None:
        r"""
        Apply the Xavier-uniform initialization to every matrix parameter.
        """

        for parameter in self.parameters():
            if parameter.ndim > 1:
                nn.init.xavier_uniform_(parameter)

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

        input = normalize_ligandmpnn_input(input, dtype=self.edge_embedding.weight.dtype)
        backbone_coordinates = input.backbone_coordinates
        context_coordinates = input.context_coordinates
        if coordinate_noise:
            backbone_noise = torch.empty_like(backbone_coordinates).normal_(generator=generator)
            context_noise = torch.empty_like(context_coordinates).normal_(generator=generator)
            backbone_coordinates = backbone_coordinates + coordinate_noise * backbone_noise
            context_coordinates = context_coordinates + coordinate_noise * context_noise
        mask = input.residue_mask
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
            input.context_mask,
            input.residue_context_coordinates,
            input.residue_context_atomic_numbers,
            input.residue_context_mask,
            input.design_mask,
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
                input.design_mask,
                input.tied_group_ids,
                generator,
            )
        else:
            validate_decoding_ranks(decoding_ranks, input.tied_group_ids)
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

    @torch.inference_mode()
    def score(
        self,
        input: LigandMpnnInput,
        *,
        num_decoding_orders: int = 1,
        decoding_ranks: Tensor | None = None,
        coordinate_noise: float = 0.0,
        use_side_chain_context: bool = False,
        generator: torch.Generator | None = None,
    ) -> LigandMpnnOutput:
        r"""
        Score the observed sequence over one or more decoding orders.

        Args:
            input: One processed protein-ligand target.
            num_decoding_orders: Number of decoding orders to evaluate.
            decoding_ranks: Optional ranks with shape
                ``(num_decoding_orders, num_residues)``.
            coordinate_noise: Standard deviation of Gaussian noise in coordinate units.
            use_side_chain_context: Whether fixed protein side chains contribute context atoms.
            generator: Random generator used for orders and coordinate noise.

        Returns:
            Raw vocabulary logits and the decoding ranks used for each evaluation.
        """

        return self.autoregressive_probabilities(
            input,
            num_decoding_orders=num_decoding_orders,
            decoding_ranks=decoding_ranks,
            coordinate_noise=coordinate_noise,
            use_side_chain_context=use_side_chain_context,
            generator=generator,
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
    ) -> LigandMpnnOutput:
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
            _invalid("num_decoding_orders", num_decoding_orders)
        repeated = self._repeat_input(input, num_decoding_orders)
        if decoding_ranks is None:
            ranks = self._generate_ranks(
                repeated.residue_mask,
                repeated.design_mask,
                repeated.tied_group_ids,
                generator,
            )
        else:
            validate_decoding_ranks(decoding_ranks, repeated.tied_group_ids)
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
        return LigandMpnnOutput(
            logits=logits,
            decoding_ranks=ranks,
            batch_size=[logits.shape[0]],
        )

    def sequence_free_autoregressive_probabilities(
        self,
        input: LigandMpnnInput,
        *,
        num_decoding_orders: int = 1,
        decoding_ranks: Tensor | None = None,
        coordinate_noise: float = 0.0,
        use_side_chain_context: bool = False,
        generator: torch.Generator | None = None,
    ) -> LigandMpnnOutput:
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
            _invalid("num_decoding_orders", num_decoding_orders)
        repeated = self._repeat_input(input, num_decoding_orders)
        if decoding_ranks is None:
            ranks = self._generate_ranks(
                repeated.residue_mask,
                repeated.design_mask,
                repeated.tied_group_ids,
                generator,
            )
        else:
            validate_decoding_ranks(decoding_ranks, repeated.tied_group_ids)
            ranks = decoding_ranks
        node_states, edge_states, neighbor_indices, neighbor_valid = self.encode(
            repeated,
            coordinate_noise=coordinate_noise,
            use_side_chain_context=use_side_chain_context,
            generator=generator,
        )
        logits = decode_sequence_free(
            node_states,
            edge_states,
            neighbor_indices,
            neighbor_valid,
            repeated.residue_mask,
            ranks,
            decoder_blocks=self.decoder_blocks,
            output_projection=self._logits,
            checkpoint=(
                self._gradient_checkpointing_func
                if self.gradient_checkpointing and self.training and torch.is_grad_enabled()
                else None
            ),
        )
        return LigandMpnnOutput(
            logits=logits,
            decoding_ranks=ranks,
            batch_size=[logits.shape[0]],
        )

    def conditional_probabilities(
        self,
        input: LigandMpnnInput,
        *,
        num_decoding_orders: int = 1,
        base_decoding_ranks: Tensor | None = None,
        coordinate_noise: float = 0.0,
        use_side_chain_context: bool = False,
        generator: torch.Generator | None = None,
    ) -> LigandMpnnOutput:
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
            Logits with shape ``(num_decoding_orders, num_residues, vocab_size)``
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
    ) -> LigandMpnnOutput:
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
            Logits with shape ``(num_decoding_orders, num_residues, vocab_size)``
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
            _invalid("num_samples", num_samples)
        if temperature <= 0:
            _invalid("temperature", temperature)

        repeated = self._repeat_input(input, num_samples)
        repeated.sequence_tokens = project_model_categories(
            repeated.sequence_tokens,
            repeated.residue_mask,
            num_classes=self.token_embedding.num_embeddings,
            feature="sequence tokens",
        )
        if decoding_ranks is None:
            ranks = self._generate_ranks(
                repeated.residue_mask,
                repeated.design_mask,
                repeated.tied_group_ids,
                generator,
            )
        else:
            validate_decoding_ranks(decoding_ranks, repeated.tied_group_ids)
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

        num_residues = repeated.sequence_tokens.shape[1]
        device = repeated.sequence_tokens.device
        sequence_tokens = repeated.sequence_tokens.clone()
        log_probabilities = node_states.new_zeros(num_samples, num_residues, len(TOKENS))
        sampling_probabilities = torch.zeros_like(log_probabilities)
        generatable = self.generatable_classes
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
                    neighbor_valid[batch_indices, positions].unsqueeze(1),
                )
                block_states[block_index + 1][batch_indices, positions] = updated[:, 0]

            # Tied members combine their finite projections; the untrained classes leave the distribution after.
            raw_logits = self.output_projection(block_states[-1][batch_indices, positions])
            logits = raw_logits.masked_fill(~self.trained_classes, -torch.inf)
            selected = repeated.residue_mask[batch_indices, positions] & repeated.design_mask[batch_indices, positions]
            log_probabilities[batch_indices, positions] = torch.where(
                selected.unsqueeze(-1), F.log_softmax(logits, dim=-1), 0.0
            )

            member_counts = member_mask.sum(dim=-1)
            if repeated.tied_weights is None:
                combined_logits = torch.segment_reduce(raw_logits, "sum", lengths=member_counts)
            else:
                weights = repeated.tied_weights[batch_indices, positions].to(logits.dtype)
                combined_logits = torch.segment_reduce(
                    raw_logits * weights.unsqueeze(-1),
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
            if repeated.residue_logit_bias is not None:
                combined_logits = (
                    combined_logits
                    + repeated.residue_logit_bias[
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

            fixed_mask = member_mask & ~(repeated.residue_mask & repeated.design_mask)
            has_fixed = fixed_mask.any(dim=-1)
            anchor_positions = fixed_mask.long().argmax(dim=-1)
            anchor_tokens = repeated.sequence_tokens[sample_indices, anchor_positions]
            has_designed = torch.segment_reduce(
                selected.to(logits.dtype),
                "max",
                lengths=member_counts,
            ).bool()
            anchored = has_fixed & has_designed
            generatable_anchor = torch.isin(anchor_tokens, generatable)
            torch._assert(
                torch.all(~anchored | generatable_anchor),
                "a mixed fixed/design causal unit requires a generatable fixed identity",
            )
            if group_support is not None:
                anchor_supported = torch.gather(
                    group_support,
                    1,
                    anchor_tokens.unsqueeze(-1),
                ).squeeze(-1)
                torch._assert(
                    torch.all(~anchored | anchor_supported),
                    "a mixed fixed/design causal unit requires its fixed identity in final support",
                )
            if group_support is not None:
                combined_logits = combined_logits.masked_fill(~group_support, -torch.inf)
            # Upstream draws from its 20 generatable columns; drawing from exactly those keeps its random stream.
            drawn = F.softmax(combined_logits[..., generatable] / temperature, dim=-1)
            drawn_tokens = generatable[torch.multinomial(drawn, 1, generator=generator).squeeze(-1)]
            group_tokens = torch.where(has_fixed, anchor_tokens, drawn_tokens)
            probabilities = torch.zeros_like(combined_logits).index_copy(-1, generatable, drawn)

            probability_anchor = torch.where(anchored, anchor_tokens, 0)
            anchor_probabilities = F.one_hot(probability_anchor, len(TOKENS)).to(probabilities.dtype)
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
    ) -> LigandMpnnSamplingOutput:
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
            Raw sequence samples in generation order.
        """

        return self.sample(
            input,
            num_samples=num_candidates,
            temperature=temperature,
            coordinate_noise=coordinate_noise,
            use_side_chain_context=use_side_chain_context,
            generator=generator,
        )

    @staticmethod
    def _require_single_target(input: LigandMpnnInput) -> None:
        r"""
        Reject an execution-batch axis on one-target biological operations.
        """

        if input.backbone_coordinates.shape[0] != 1:
            _invalid("input", f"batch of {input.backbone_coordinates.shape[0]}")

    @staticmethod
    def _repeat_input(input: LigandMpnnInput, num_repeats: int) -> LigandMpnnInput:
        r"""
        Repeat one biological input into ``num_repeats`` aligned evaluation rows.
        """

        def repeat(value: Tensor | None) -> Tensor | None:
            return None if value is None else value.repeat_interleave(num_repeats, dim=0)

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
            residue_context_atomic_numbers=repeat(input.residue_context_atomic_numbers),
            residue_context_mask=repeat(input.residue_context_mask),
            design_mask=repeat(input.design_mask),
            ligand_interface_mask=repeat(input.ligand_interface_mask),
            component_index=repeat(input.component_index),
            tied_group_ids=repeat(input.tied_group_ids),
            tied_weights=repeat(input.tied_weights),
            allowed_token_mask=repeat(input.allowed_token_mask),
            residue_logit_bias=repeat(input.residue_logit_bias),
            batch_size=[],
        )

    # -- decode-order generation -----------------------------------------------------------------

    @staticmethod
    def _generate_ranks(
        residue_mask: Tensor,
        design_mask: Tensor,
        tied_group_ids: Tensor,
        generator: torch.Generator | None,
    ) -> Tensor:
        r"""Draw independent half-normal priorities for fixed-first causal units."""

        designed = (residue_mask & design_mask).to(torch.long)
        unit_class = torch.full_like(tied_group_ids, 2).scatter_reduce(
            1,
            tied_group_ids,
            designed,
            reduce="amin",
            include_self=True,
        )
        priority = torch.empty_like(tied_group_ids, dtype=torch.get_default_dtype()).normal_(generator=generator).abs()
        bounded_priority = priority / (1 + priority)
        unit_order = torch.argsort(bounded_priority + unit_class.to(priority.dtype), dim=1)
        positions = torch.ones_like(tied_group_ids).cumsum(dim=1) - 1
        unit_rank = torch.zeros_like(unit_order).scatter(1, unit_order, positions)
        return torch.gather(unit_rank, 1, tied_group_ids)

    @staticmethod
    def _representative_positions(tied_group_ids: Tensor) -> Tensor:
        r"""Return the first row belonging to every causal unit in one input."""

        device = tied_group_ids.device
        num_residues = tied_group_ids.shape[1]
        row_index = torch.arange(num_residues, device=device)
        unit_first = torch.full((num_residues,), num_residues, dtype=torch.long, device=device)
        unit_first.scatter_reduce_(0, tied_group_ids[0], row_index, reduce="amin", include_self=True)
        return torch.where(row_index == unit_first[tied_group_ids[0]])[0]

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
        sequence_tokens = project_model_categories(
            sequence_tokens,
            residue_mask,
            num_classes=self.token_embedding.num_embeddings,
            feature="sequence tokens",
        )
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
                        neighbor_valid,
                    ),
                )
            else:
                node_states = block(node_states, context, mask, neighbor_valid)
        return self._logits(node_states)

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
    ) -> LigandMpnnOutput:
        r"""
        Evaluate every target moved exactly last or first under one base decoding order.
        """

        self._require_single_target(input)
        if num_decoding_orders <= 0:
            _invalid("num_decoding_orders", num_decoding_orders)
        repeated = self._repeat_input(input, num_decoding_orders)
        device = repeated.backbone_coordinates.device
        num_residues = repeated.sequence_tokens.shape[1]
        untied_group_ids = torch.ones_like(repeated.tied_group_ids).cumsum(dim=1) - 1
        if base_decoding_ranks is None:
            base_ranks = self._generate_ranks(
                repeated.residue_mask,
                repeated.design_mask,
                untied_group_ids,
                generator,
            )
        else:
            validate_decoding_ranks(base_decoding_ranks, untied_group_ids)
            base_ranks = base_decoding_ranks
        node_states, edge_states, neighbor_indices, neighbor_valid = self.encode(
            repeated,
            coordinate_noise=coordinate_noise,
            use_side_chain_context=use_side_chain_context,
            generator=generator,
        )
        first_ranks = base_ranks.amin(dim=-1) - 1
        last_ranks = base_ranks.amax(dim=-1) + 1
        logits = node_states.new_zeros(num_decoding_orders, num_residues, len(TOKENS))
        effective_ranks = torch.empty(num_decoding_orders, num_residues, num_residues, dtype=torch.long, device=device)
        for target in range(num_residues):
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
        return LigandMpnnOutput(
            logits=logits,
            decoding_ranks=effective_ranks,
            batch_size=[logits.shape[0]],
        )


class LigandMpnnFeatureEncoder(MpnnProteinFeatureEncoder):
    r"""
    Build residue-graph and atom-context features for LigandMPNN.

    Args:
        hidden_size: Width of the per-residue edge features and per-atom context node features.
        num_neighbors: Number of nearest-neighbor residues in the graph.
        positional_embedding_size: Width of the relative-position embedding.
        num_rbf: Number of radial basis functions per distance feature.
    """

    periodic_group: Tensor
    periodic_period: Tensor

    def __init__(
        self,
        hidden_size: int,
        num_neighbors: int,
        *,
        positional_embedding_size: int = 16,
        num_rbf: int = 16,
    ) -> None:
        super().__init__(
            hidden_size,
            num_neighbors,
            positional_embedding_size=positional_embedding_size,
            num_rbf=num_rbf,
        )
        self.backend = "auto"
        buffer_device = self.edge_projection.weight.device
        if buffer_device.type == "meta":
            buffer_device = torch.device("cpu")
        group = torch.tensor(GROUP_BY_ATOMIC_NUMBER, dtype=torch.long, device=buffer_device)
        period = torch.tensor(PERIOD_BY_ATOMIC_NUMBER, dtype=torch.long, device=buffer_device)
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
        residue_context_atomic_numbers: Tensor,
        residue_context_mask: Tensor,
        design_mask: Tensor,
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
            context_atomic_numbers: Context element IDs with shape ``(B, L, A)``.
                Only available context rows are validated and looked up; excluded payload is ignored.
            context_mask: Context-atom mask with shape ``(B, L, A)``.
            residue_context_coordinates: Side-chain coordinates with shape
                ``(B, L, 32, 3)``.
            residue_context_atomic_numbers: Actual side-chain element numbers with shape ``(B, L, 32)``.
                Zero denotes unknown identity and excludes that optional context atom.
                Excluded, nonfixed and unavailable residue rows do not supply element IDs.
            residue_context_mask: Side-chain atom mask with shape ``(B, L, 32)``.
            design_mask: Design-position mask with shape ``(B, L)``.
            use_side_chain_context: Whether fixed side chains are added to atom context.

        Returns:
            atom_features: Context features with shape ``(B, L, A, H)``.
            edges: Residue edges with shape ``(B, L, K, H)``.
            neighbor_indices: Neighbor indices with shape ``(B, L, K)``.
            context_nodes: Context nodes with shape ``(B, L, A, H)``.
            context_edges: Context edges with shape ``(B, L, A, A, H)``.
            context_mask: Context mask with shape ``(B, L, A)``.
            neighbor_valid: Valid-neighbor mask with shape ``(B, L, K)``.
        """

        backbone_coordinates = project_model_coordinates(
            backbone_coordinates,
            residue_mask.unsqueeze(-1) & torch.ones_like(backbone_coordinates[..., 0], dtype=torch.bool),
            feature="protein backbone",
        )
        context_coordinates = project_model_coordinates(context_coordinates, context_mask, feature="ligand context")
        residue_context_coordinates = project_model_coordinates(
            residue_context_coordinates, residue_context_mask, feature="side-chain context"
        )
        context_atomic_numbers = project_model_categories(
            context_atomic_numbers,
            context_mask & residue_mask.unsqueeze(-1),
            num_classes=min(120, self.periodic_group.numel(), self.periodic_period.numel()),
            feature="context element IDs",
        )
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
            residue_context_atomic_numbers = project_model_categories(
                residue_context_atomic_numbers,
                residue_context_mask & residue_mask.unsqueeze(-1) & ~design_mask.unsqueeze(-1),
                num_classes=min(120, self.periodic_group.numel(), self.periodic_period.numel()),
                feature="side-chain element IDs",
            )
            context_coordinates, context_atomic_numbers, context_mask = self._append_sidechain_context(
                beta_carbon,
                context_coordinates,
                context_atomic_numbers,
                context_mask,
                residue_context_coordinates,
                residue_context_atomic_numbers,
                residue_context_mask,
                design_mask,
                residue_mask,
                neighbor_indices,
                neighbor_valid,
                num_context_atoms,
            )

        element_features = self._element_features(context_atomic_numbers)
        element_projection = self.element_projection(element_features.to(self.element_projection.weight.dtype))

        backbone_atoms = torch.stack(
            (nitrogen, alpha_carbon, carbon, oxygen, beta_carbon),
            dim=2,
        )
        atom_context_distances = smoothed_vector_norm(
            context_coordinates[:, :, :, None] - backbone_atoms[:, :, None], eps=1e-6, backend=self.backend
        )
        atom_context_features = radial_basis(atom_context_distances, self.num_rbf).flatten(start_dim=-2)
        atom_features = torch.cat(
            (
                atom_context_features,
                element_projection,
                local_backbone_frame_features(
                    nitrogen, alpha_carbon, carbon, context_coordinates, backend=self.backend
                ),
            ),
            dim=-1,
        )
        atom_features = self.atom_norm(self.atom_projection(atom_features.to(self.atom_projection.weight.dtype)))
        context_distances = smoothed_vector_norm(
            context_coordinates[:, :, :, None] - context_coordinates[:, :, None, :], eps=1e-6, backend=self.backend
        )
        context_edges = self.context_edge_norm(
            self.context_edge_projection(
                radial_basis(context_distances, self.num_rbf).to(self.context_edge_projection.weight.dtype)
            )
        )
        context_nodes = self.context_node_norm(
            self.context_node_projection(element_features.to(self.context_node_projection.weight.dtype))
        )
        return atom_features, edges, neighbor_indices, context_nodes, context_edges, context_mask, neighbor_valid

    def _append_sidechain_context(
        self,
        beta_carbon: Tensor,
        context_coordinates: Tensor,
        context_atomic_numbers: Tensor,
        context_mask: Tensor,
        residue_context_coordinates: Tensor,
        residue_context_atomic_numbers: Tensor,
        residue_context_mask: Tensor,
        design_mask: Tensor,
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

        batch, num_residues = context_mask.shape[:2]
        fixed_mask = (
            residue_context_mask.bool() & (residue_context_atomic_numbers > 0) & ~design_mask[:, :, None].bool()
        )
        neighbor_subset = neighbor_indices[:, :, :16]
        neighbor_subset_valid = neighbor_valid[:, :, :16]
        sidechain_mask = gather_nodes(fixed_mask, neighbor_subset)
        sidechain_mask = sidechain_mask * neighbor_subset_valid[..., None]
        sidechain_mask = sidechain_mask.reshape(batch, num_residues, -1)
        sidechain_coordinates = gather_nodes(
            residue_context_coordinates.reshape(batch, num_residues, -1),
            neighbor_subset,
        ).reshape(batch, num_residues, -1, 3)
        sidechain_types = gather_nodes(residue_context_atomic_numbers, neighbor_subset).reshape(batch, num_residues, -1)
        context_coordinates = torch.cat((sidechain_coordinates, context_coordinates), dim=2)
        context_mask = torch.cat((sidechain_mask, context_mask), dim=2)
        context_atomic_numbers = torch.cat((sidechain_types, context_atomic_numbers), dim=2)
        distances = torch.sum((beta_carbon[:, :, None] - context_coordinates) ** 2, dim=-1)
        valid = residue_mask[:, :, None].bool() & context_mask.bool()
        maximum = torch.where(valid, distances, 0).amax(dim=-1, keepdim=True)
        adjusted = torch.where(valid, distances, maximum + 1)
        num_selected_context_atoms = min(num_context_atoms, adjusted.shape[-1])
        atom_indices = torch.topk(adjusted, num_selected_context_atoms, dim=-1, largest=False).indices
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
        sequence_tokens: Canonical classes with shape ``(batch_size, num_residues)``.
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
        residue_context_atomic_numbers: Actual side-chain element numbers with shape
            ``(batch_size, num_residues, 32)``; zero is unknown optional context.
        residue_context_mask: Side-chain context mask with shape
            ``(batch_size, num_residues, 32)``.
        design_mask: Positions eligible for design with shape
            ``(batch_size, num_residues)``.
        ligand_interface_mask: Residues at the ligand interface with shape
            ``(batch_size, num_residues)``.
        component_index: Canonical component indices aligned to residue rows.
        tied_group_ids: Autoregressive unit ID for each residue row.
        tied_weights: Optional tied-group weights with shape ``(batch_size, num_residues)``.
        allowed_token_mask: Optional mask over the 72 canonical classes with shape
            ``(batch_size, num_residues, 72)``.
        residue_logit_bias: Optional additive bias over the 72 canonical classes with shape
            ``(batch_size, num_residues, 72)``.
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
    residue_context_atomic_numbers: Tensor
    residue_context_mask: Tensor
    design_mask: Tensor
    ligand_interface_mask: Tensor
    component_index: Tensor
    tied_group_ids: Tensor
    tied_weights: Tensor | None = None
    allowed_token_mask: Tensor | None = None
    residue_logit_bias: Tensor | None = None


class LigandMpnnOutput(TensorCollection):
    r"""
    Outputs from one LigandMPNN forward pass.

    Attributes:
        logits: Vocabulary logits with shape
            ``(batch_size, num_residues, vocab_size)``.
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


class LigandMpnnSamplingOutput(TensorCollection):
    r"""
    Outputs from LigandMPNN autoregressive sampling.

    Attributes:
        sequence_tokens: Sampled token IDs with shape
            ``(num_samples, num_residues)``.
        log_probabilities: Model log probabilities with shape
            ``(num_samples, num_residues, vocab_size)``.
        sampling_probabilities: Sampling distribution used at each position.
        decoding_ranks: Autoregressive ranks with shape
            ``(num_samples, num_residues)``.
    """

    sequence_tokens: Tensor
    log_probabilities: Tensor
    sampling_probabilities: Tensor
    decoding_ranks: Tensor
