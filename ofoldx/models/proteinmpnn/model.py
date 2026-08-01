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
ProteinMPNN fixed-backbone sequence model.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar, NamedTuple, cast

import torch
import torch.nn.functional as F
from chanfig import Config
from tensordict import TensorClass, tensorclass
from torch import Tensor, nn

from ...data.tokenization import TOKEN_INDEX, TOKENS
from ...errors import RequestRef, fail
from ...model import Model, TensorCollection
from ...modules.backend import LayerNorm, gather_nodes, masked_knn, resolve_config_backend
from ...modules.functional import radial_basis
from ...modules.geometry import rotation_from_two_vectors, rotation_to_quaternion, smoothed_vector_norm
from ...modules.mpnn import (
    MpnnDecoderBlock,
    MpnnEncoderBlock,
    MpnnProteinFeatureEncoder,
    MpnnRelativePositionEncoding,
    autoregressive_masks,
    cat_neighbors_nodes,
    decode_sequence_free,
    validate_decoding_ranks,
)
from ..processing import project_model_categories, project_model_coordinates
from .config import ProteinMpnnConfig

#: The classes the released checkpoints were trained on, in their upstream order (canonical spelling); every other
#: canonical class has logit ``-inf``, so identical inputs give the upstream distribution.
PROTEINMPNN_VOCABULARY: tuple[str, ...] = tuple("ACDEFGHIKLMNPQRSTVWYX")


def _invalid(argument: str, value: object) -> None:
    fail("value_invalid", RequestRef(argument, str(value)))


def normalize_proteinmpnn_input(input: ProteinMpnnInput, *, dtype: torch.dtype | None = None) -> ProteinMpnnInput:
    """Return finite available model coordinates without mutating the supplied input."""
    result = cast(ProteinMpnnInput, cast(TensorClass, input).clone(recurse=False))
    points = input.backbone_coordinates
    observed = input.residue_mask
    if points.dim() == observed.dim() + 2:
        observed = observed.unsqueeze(-1) & torch.ones_like(points[..., 0], dtype=torch.bool)
    result.backbone_coordinates = project_model_coordinates(points, observed, dtype=dtype, feature="protein backbone")
    return result


class ProteinMpnnSequenceModel(Model):
    r"""
    Design protein sequences for fixed backbones with ProteinMPNN.

    Accepts [`ProteinMpnnInput`][ofoldx.models.proteinmpnn.model.ProteinMpnnInput] and returns
    [`ProteinMpnnOutput`][ofoldx.models.proteinmpnn.model.ProteinMpnnOutput].

    Args:
        config: Model architecture and inference configuration.
    """

    config_class: ClassVar[type[Config]] = ProteinMpnnConfig
    _gradient_checkpointing_func: Callable[..., object]

    def __init__(self, config: ProteinMpnnConfig | None = None) -> None:
        config = ProteinMpnnConfig() if config is None else config
        super().__init__(config)
        self.gradient_checkpointing = False
        feature_class = ProteinMpnnCaFeatureEncoder if self.config.ca_only else MpnnProteinFeatureEncoder
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
        self.token_embedding = nn.Embedding(len(TOKENS), self.config.hidden_size)
        self.decoder_blocks = nn.ModuleList(
            MpnnDecoderBlock(
                self.config.hidden_size,
                self.config.hidden_size * 3,
                dropout=self.config.dropout,
            )
            for _ in range(self.config.num_decoder_blocks)
        )
        self.output_projection = nn.Linear(self.config.hidden_size, len(TOKENS))
        trained = torch.zeros(len(TOKENS), dtype=torch.bool)
        trained[[TOKEN_INDEX[token] for token in PROTEINMPNN_VOCABULARY]] = True
        self.register_buffer("trained_classes", trained, persistent=False)
        self.post_init()

    def _logits(self, states: Tensor) -> Tensor:
        r"""Vocabulary logits over the canonical classes; classes outside the trained vocabulary are ``-inf``."""
        return self.output_projection(states).masked_fill(~self.trained_classes, -torch.inf)

    def initialize_parameters(self) -> None:
        r"""
        Apply the Xavier initialization to matrix parameters.
        """

        for parameter in self.parameters():
            if parameter.ndim > 1:
                nn.init.xavier_uniform_(parameter)

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
            node_states: Node states with shape ``(B, L, H)``.
            edge_states: Edge states with shape ``(B, L, K, H)``.
            neighbor_indices: Neighbor indices with shape ``(B, L, K)``.
            attend_mask: Real-neighbor mask with shape ``(B, L, K)``.
        """

        input = normalize_proteinmpnn_input(input, dtype=self.edge_embedding.weight.dtype)
        backbone_coordinates = input.backbone_coordinates
        residue_mask = input.residue_mask
        if coordinate_noise:
            noise = torch.empty_like(backbone_coordinates).normal_(generator=generator)
            backbone_coordinates = backbone_coordinates + coordinate_noise * noise
        mask = residue_mask
        # ``attend_mask`` comes from the k-NN: it is the real-neighbor mask, which also zeroes filler slots when
        # a structure is shorter than ``num_neighbors`` (the encoder must not attend to a filler neighbor).
        edge_features, neighbor_indices, attend_mask = self.feature_encoder(
            backbone_coordinates,
            residue_mask,
            input.residue_index,
            input.chain_index,
        )
        edge_states = self.edge_embedding(edge_features)
        # Zero node states with the residue structure of ``edge_states`` (one neighbor slot, embedding width):
        # deriving them this way keeps a ragged NestedTensor batch ragged, where ``new_zeros`` with an explicit
        # ``(B, N, hidden)`` shape would materialize a dense, padded tensor that no longer aligns with the edges.
        node_states = self._initial_node_states(input, edge_states)
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

    def _initial_node_states(self, input: ProteinMpnnInput, edge_states: Tensor) -> Tensor:
        return torch.zeros_like(edge_states[:, :, 0, :])

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
                input.design_mask,
                input.tied_group_ids,
                generator,
            )
        else:
            validate_decoding_ranks(decoding_ranks, input.tied_group_ids)
        node_states, edge_states, neighbor_indices, neighbor_valid = self.encode(
            input,
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

    @torch.inference_mode()
    def score(
        self,
        input: ProteinMpnnInput,
        *,
        num_decoding_orders: int = 1,
        decoding_ranks: Tensor | None = None,
        coordinate_noise: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> ProteinMpnnOutput:
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
            Raw vocabulary logits and the decoding ranks used for each evaluation.
        """

        self._require_single_target(input)
        if num_decoding_orders <= 0:
            _invalid("num_decoding_orders", num_decoding_orders)
        repeated = self._repeat_input(input, num_decoding_orders)
        return self.forward(
            repeated,
            decoding_ranks=decoding_ranks,
            coordinate_noise=coordinate_noise,
            generator=generator,
        )

    def autoregressive_probabilities(
        self,
        input: ProteinMpnnInput,
        *,
        num_decoding_orders: int = 1,
        decoding_ranks: Tensor | None = None,
        coordinate_noise: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> ProteinMpnnOutput:
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
            _invalid("num_decoding_orders", num_decoding_orders)
        repeated = self._repeat_input(input, num_decoding_orders)
        output = self.forward(
            repeated,
            decoding_ranks=decoding_ranks,
            coordinate_noise=coordinate_noise,
            generator=generator,
        )
        return output

    def sequence_free_autoregressive_probabilities(
        self,
        input: ProteinMpnnInput,
        *,
        num_decoding_orders: int = 1,
        decoding_ranks: Tensor | None = None,
        coordinate_noise: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> ProteinMpnnOutput:
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
            _invalid("num_decoding_orders", num_decoding_orders)
        repeated = self._repeat_input(input, num_decoding_orders)
        if decoding_ranks is None:
            decoding_ranks = self._generate_ranks(
                repeated.residue_mask,
                repeated.design_mask,
                repeated.tied_group_ids,
                generator,
            )
        else:
            validate_decoding_ranks(decoding_ranks, repeated.tied_group_ids)
        node_states, edge_states, neighbor_indices, neighbor_valid = self.encode(
            repeated,
            coordinate_noise=coordinate_noise,
            generator=generator,
        )
        logits = decode_sequence_free(
            node_states,
            edge_states,
            neighbor_indices,
            neighbor_valid,
            repeated.residue_mask,
            decoding_ranks,
            decoder_blocks=self.decoder_blocks,
            output_projection=self._logits,
            checkpoint=(
                self._gradient_checkpointing_func
                if self.gradient_checkpointing and self.training and torch.is_grad_enabled()
                else None
            ),
        )
        return ProteinMpnnOutput(
            logits=logits,
            decoding_ranks=decoding_ranks,
            batch_size=[logits.shape[0]],
        )

    def conditional_probabilities(
        self,
        input: ProteinMpnnInput,
        *,
        num_decoding_orders: int = 1,
        base_decoding_ranks: Tensor | None = None,
        coordinate_noise: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> ProteinMpnnOutput:
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
            Logits with shape ``(num_decoding_orders, num_residues, vocab_size)``
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
    ) -> ProteinMpnnOutput:
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
            Logits with shape ``(num_decoding_orders, num_residues, vocab_size)``
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
            global_logit_bias: Optional bias with shape ``(vocab_size,)``.
            global_omit_token_mask: Optional exclusion mask with shape
                ``(vocab_size,)``.
            pssm_mixture: Weight of the per-residue PSSM probabilities.
            pssm_log_odds_threshold: Optional minimum accepted PSSM log odds.
            generator: Random generator used for orders, noise, and token sampling.

        Returns:
            Sampled token IDs, model log probabilities, sampling probabilities,
                and decoding ranks, each with a leading ``num_samples`` axis.
        """

        self._require_single_target(input)
        if num_samples <= 0:
            _invalid("num_samples", num_samples)
        if temperature <= 0:
            _invalid("temperature", temperature)
        if not 0 <= pssm_mixture <= 1:
            _invalid("pssm_mixture", pssm_mixture)

        repeated = self._repeat_input(input, num_samples)
        repeated.sequence_tokens = project_model_categories(
            repeated.sequence_tokens,
            repeated.residue_mask,
            num_classes=self.token_embedding.num_embeddings,
            feature="sequence tokens",
        )
        device = repeated.backbone_coordinates.device
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
            generator=generator,
        )
        num_residues = repeated.sequence_tokens.shape[1]
        vocab_size = len(TOKENS)
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
        log_probabilities = node_states.new_zeros(num_samples, num_residues, vocab_size)
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
                    neighbor_valid[batch_indices, positions].unsqueeze(1),
                )
                block_states[block_index + 1][batch_indices, positions] = updated[:, 0]

            # Tied members combine their finite projections; the untrained classes leave the distribution after.
            raw_logits = self.output_projection(block_states[-1][batch_indices, positions])
            logits = raw_logits.masked_fill(~self.trained_classes, -torch.inf)
            log_probabilities[batch_indices, positions] = F.log_softmax(logits, dim=-1)
            member_counts = member_mask.sum(dim=-1)
            if repeated.tied_weights is None:
                # Default coefficient 1/len(G) per member: the equal mean of the
                # member logits, matching tied decoding with unit betas.
                combined_logits = torch.segment_reduce(
                    raw_logits,
                    "sum",
                    lengths=member_counts,
                ) / member_counts.unsqueeze(-1)
            else:
                # Final signed coefficients act only on neural logits, with no
                # group-size or coefficient-sum normalization.
                weights = repeated.tied_weights[batch_indices, positions]
                combined_logits = torch.segment_reduce(
                    raw_logits * weights.unsqueeze(-1),
                    "sum",
                    lengths=member_counts,
                )
            combined_logits = combined_logits.masked_fill(~self.trained_classes, -torch.inf)
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
                repeated.residue_mask[batch_indices, positions] & repeated.design_mask[batch_indices, positions]
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
            fixed_mask = member_mask & ~(repeated.residue_mask & repeated.design_mask)
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
    def _require_single_target(input: ProteinMpnnInput) -> None:
        r"""
        Reject an execution-batch axis on one-target biological operations.
        """

        if input.backbone_coordinates.shape[0] != 1:
            _invalid("input", f"batch of {input.backbone_coordinates.shape[0]}")

    @staticmethod
    def _repeat_input(input: ProteinMpnnInput, num_repeats: int) -> ProteinMpnnInput:
        r"""
        Repeat one biological input into ``num_repeats`` aligned evaluation rows.
        """

        def repeat(value: Tensor | None) -> Tensor | None:
            return None if value is None else value.repeat_interleave(num_repeats, dim=0)

        return ProteinMpnnInput(  # type: ignore[call-arg]
            backbone_coordinates=repeat(input.backbone_coordinates),
            sequence_tokens=repeat(input.sequence_tokens),
            residue_mask=repeat(input.residue_mask),
            residue_index=repeat(input.residue_index),
            chain_index=repeat(input.chain_index),
            design_mask=repeat(input.design_mask),
            component_index=repeat(input.component_index),
            tied_group_ids=repeat(input.tied_group_ids),
            tied_weights=repeat(input.tied_weights),
            allowed_token_mask=repeat(input.allowed_token_mask),
            residue_logit_bias=repeat(input.residue_logit_bias),
            pssm_coefficients=repeat(input.pssm_coefficients),
            pssm_probabilities=repeat(input.pssm_probabilities),
            pssm_log_odds=repeat(input.pssm_log_odds),
            membrane_labels=repeat(input.membrane_labels),
            batch_size=[],
        )

    @staticmethod
    def _generate_ranks(
        residue_mask: Tensor,
        design_mask: Tensor,
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

        designed = (residue_mask & design_mask).to(torch.long)
        unit_class = torch.full_like(tied_group_ids, 2).scatter_reduce(
            1,
            tied_group_ids,
            designed,
            reduce="amin",
            include_self=True,
        )
        decoding_priority = (
            torch.empty_like(tied_group_ids, dtype=torch.get_default_dtype()).normal_(generator=generator).abs()
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
        num_residues = tied_group_ids.shape[1]
        row_index = torch.arange(num_residues, device=device)
        unit_first = torch.full((num_residues,), num_residues, dtype=torch.long, device=device)
        unit_first.scatter_reduce_(0, tied_group_ids[0], row_index, reduce="amin", include_self=True)
        first_of_row = unit_first[tied_group_ids[0]]
        return torch.where(row_index == first_of_row)[0]

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
                        neighbor_valid,
                    ),
                )
            else:
                node_states = block(node_states, context, mask, neighbor_valid)
        return self._logits(node_states)

    def _single_residue_probabilities(
        self,
        input: ProteinMpnnInput,
        *,
        num_decoding_orders: int,
        base_decoding_ranks: Tensor | None,
        coordinate_noise: float,
        generator: torch.Generator | None,
        move_last: bool,
    ) -> ProteinMpnnOutput:
        r"""
        Evaluate every target moved exactly last or first under one base order.
        """

        self._require_single_target(input)
        if num_decoding_orders <= 0:
            _invalid("num_decoding_orders", num_decoding_orders)
        repeated = self._repeat_input(input, num_decoding_orders)
        device = repeated.backbone_coordinates.device
        num_residues = repeated.sequence_tokens.shape[1]
        if base_decoding_ranks is None:
            untied_group_ids = torch.ones_like(repeated.tied_group_ids).cumsum(dim=1) - 1
            base_ranks = self._generate_ranks(
                repeated.residue_mask,
                repeated.design_mask,
                untied_group_ids,
                generator,
            )
        else:
            if base_decoding_ranks.shape != (num_decoding_orders, num_residues):
                _invalid("base_decoding_ranks", tuple(base_decoding_ranks.shape))
            if base_decoding_ranks.dtype != torch.long:
                _invalid("base_decoding_ranks", base_decoding_ranks.dtype)
            base_ranks = base_decoding_ranks
        boundary_ranks = base_ranks.amax(dim=-1) + 1 if move_last else base_ranks.amin(dim=-1) - 1
        node_states, edge_states, neighbor_indices, neighbor_valid = self.encode(
            repeated,
            coordinate_noise=coordinate_noise,
            generator=generator,
        )
        vocab_size = len(TOKENS)
        logits = node_states.new_zeros(num_decoding_orders, num_residues, vocab_size)
        effective_ranks = torch.empty(
            num_decoding_orders,
            num_residues,
            num_residues,
            dtype=torch.long,
            device=device,
        )
        for target in range(num_residues):
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
        return ProteinMpnnOutput(
            logits=logits,
            decoding_ranks=effective_ranks,
            batch_size=[logits.shape[0]],
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
        Condition the biased neural/PSSM mixture on its final hard support.

        Each component retains its mass on the support before the two are
        combined. Their posterior mixture weights are evaluated in log space,
        so a large excluded logit cannot underflow the allowed neural component.
        """

        support = torch.ones_like(logits, dtype=torch.bool)
        if global_omit_token_mask is not None:
            support &= ~global_omit_token_mask.bool()
        if pssm_log_odds is not None and pssm_log_odds_threshold is not None:
            support &= pssm_log_odds > pssm_log_odds_threshold
        if allowed_token_mask is not None:
            support &= allowed_token_mask
        torch._assert_async(support.any(dim=-1).all(), "ProteinMPNN sampling requires nonempty hard token support")

        scaled_logits = logits / temperature
        if global_logit_bias is not None:
            scaled_logits = scaled_logits + global_logit_bias / temperature
        if residue_logit_bias is not None:
            scaled_logits = scaled_logits + residue_logit_bias / temperature
        supported_logits = scaled_logits.masked_fill(~support, -torch.inf)
        neural_probabilities = F.softmax(supported_logits, dim=-1)
        if pssm_probabilities is None or not pssm_mixture:
            return neural_probabilities

        coefficient = (
            torch.ones(logits.shape[0], device=logits.device, dtype=logits.dtype)
            if pssm_coefficient is None
            else pssm_coefficient
        )
        mixture = pssm_mixture * coefficient
        neural_log_mass = torch.logsumexp(supported_logits, dim=-1) - torch.logsumexp(scaled_logits, dim=-1)
        supported_pssm = pssm_probabilities.masked_fill(~support, 0)
        pssm_mass = supported_pssm.sum(dim=-1)
        # A zero-mass PSSM contributes nothing; the denominator is only used to
        # represent that absent component as zeros, never to restore token support.
        pssm_denominator = torch.where(pssm_mass > 0, pssm_mass, torch.ones_like(pssm_mass))
        conditional_pssm = supported_pssm / pssm_denominator.unsqueeze(-1)
        log_masses = torch.stack(
            (
                neural_log_mass + torch.log1p(-mixture),
                pssm_mass.log() + mixture.log(),
            ),
            dim=-1,
        )
        torch._assert_async(
            log_masses.isfinite().any(dim=-1).all(),
            "ProteinMPNN sampling mixture has zero probability on its hard token support",
        )
        weights = F.softmax(log_masses, dim=-1)
        return weights[:, :1] * neural_probabilities + weights[:, 1:] * conditional_pssm


class ProteinMpnnModel(ProteinMpnnSequenceModel):
    r"""ProteinMPNN sequence design with its native input-score evaluation."""

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
    ) -> ProteinMpnnDesignOutput:
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
            Raw sampling tensors and raw input-sequence score tensors.
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
        if not bool((input.residue_mask & input.design_mask).any()):
            _invalid("design_mask", "no valid scoring position")
        untied_group_ids = torch.ones_like(input.tied_group_ids).cumsum(dim=1) - 1
        score_input = ProteinMpnnInput(  # type: ignore[call-arg]
            backbone_coordinates=input.backbone_coordinates,
            sequence_tokens=input.sequence_tokens,
            residue_mask=input.residue_mask,
            residue_index=input.residue_index,
            chain_index=input.chain_index,
            design_mask=input.design_mask,
            component_index=input.component_index,
            tied_group_ids=untied_group_ids,
            tied_weights=None,
            allowed_token_mask=input.allowed_token_mask,
            residue_logit_bias=input.residue_logit_bias,
            pssm_coefficients=input.pssm_coefficients,
            pssm_probabilities=input.pssm_probabilities,
            pssm_log_odds=input.pssm_log_odds,
            membrane_labels=input.membrane_labels,
            batch_size=[],
        )
        input_score = self.score(
            score_input,
            num_decoding_orders=num_decoding_orders,
            coordinate_noise=coordinate_noise,
            generator=generator,
        )
        return ProteinMpnnDesignOutput(sample=sampled, input_score=input_score)


class ProteinMpnnCaFeatureEncoder(nn.Module):
    r"""
    Encode C-alpha coordinates as a ProteinMPNN edge graph.

    Args:
        hidden_size: Width of the returned edge features.
        num_neighbors: Maximum number of nearest neighbors per residue.
        backend: Backend used by neighbour selection and geometry.
    """

    def __init__(self, hidden_size: int, num_neighbors: int, *, backend: str = "auto") -> None:
        super().__init__()
        self.backend = resolve_config_backend(backend, supported=("oops", "eager"), public_default="eager")
        self.num_neighbors = num_neighbors
        self.relative_position = MpnnRelativePositionEncoding(16)
        self.edge_projection = nn.Linear(16 + 9 * 16 + 7, hidden_size, bias=False)
        self.edge_norm = LayerNorm(hidden_size, backend=backend)

    @staticmethod
    def _quaternions(rotations: Tensor) -> Tensor:
        r"""
        Convert rotation matrices to normalized quaternion features.

        Args:
            rotations: Rotation matrices with shape ``(..., 3, 3)``.

        Returns:
            Quaternions in ``(x, y, z, w)`` order with shape ``(..., 4)``.
        """

        quaternion = rotation_to_quaternion(rotations)
        return torch.cat((quaternion[..., 1:], quaternion[..., :1]), dim=-1)

    @classmethod
    def _orientations(
        cls,
        alpha_carbons: Tensor,
        neighbor_indices: Tensor,
        adjacent: Tensor,
        neighbor_valid: Tensor,
        *,
        backend: str = "auto",
    ) -> Tensor:
        r"""
        Compute relative orientation features for neighboring residues.

        Args:
            alpha_carbons: C-alpha coordinates with shape ``(B, L, 3)``.
            neighbor_indices: Neighbor indices with shape ``(B, L, K)``.
            adjacent: Observed same-chain consecutive-residue bonds ``(B, L - 1)``.
            neighbor_valid: Observed-neighbor mask with shape ``(B, L, K)``.

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
            frame_valid = torch.zeros_like(alpha_carbons[..., 0], dtype=torch.bool)
        else:
            displacements = alpha_carbons[:, 1:] - alpha_carbons[:, :-1]
            displacement_norms = torch.linalg.vector_norm(displacements, dim=-1)
            consecutive = adjacent & (displacement_norms > 3.6) & (displacement_norms < 4.0)
            unit = F.normalize(displacements * consecutive.unsqueeze(-1), dim=-1)
            u2 = unit[:, :-2]
            u1 = unit[:, 1:-1]
            normal = torch.cross(u2, u1, dim=-1)
            frames = rotation_from_two_vectors(u2 - u1, normal, backend=backend).mT
            endpoint = torch.zeros_like(alpha_carbons[:, :1].unsqueeze(-1).expand(-1, -1, 3, 3))
            frames = torch.cat((endpoint, frames, endpoint, endpoint), dim=1)
            interior_valid = consecutive[:, :-2] & consecutive[:, 1:-1] & normal.square().sum(dim=-1).gt(0)
            endpoint_valid = torch.zeros_like(alpha_carbons[..., 0][:, :1], dtype=torch.bool)
            frame_valid = torch.cat((endpoint_valid, interior_valid, endpoint_valid, endpoint_valid), dim=1)
        neighbor_frames = gather_nodes(frames.flatten(start_dim=-2), neighbor_indices).reshape(
            *neighbor_indices.shape,
            3,
            3,
        )
        neighbor_coordinates = gather_nodes(alpha_carbons, neighbor_indices)
        frame_pair_valid = frame_valid.unsqueeze(-1) & neighbor_valid
        residue_row = torch.ones_like(alpha_carbons[..., 0], dtype=torch.long).cumsum(dim=-1) - 1
        direction_valid = frame_pair_valid & neighbor_indices.ne(residue_row.unsqueeze(-1))
        direction = neighbor_coordinates - alpha_carbons.unsqueeze(-2)
        direction = torch.matmul(frames.unsqueeze(-3), direction.unsqueeze(-1)).squeeze(-1)
        direction = direction.masked_fill(~direction_valid.unsqueeze(-1), 0)
        direction = F.normalize(direction, dim=-1)
        neighbor_frame_valid = gather_nodes(frame_valid.unsqueeze(-1), neighbor_indices).squeeze(-1)
        rotation_valid = frame_pair_valid & neighbor_frame_valid
        rotations = torch.matmul(frames.unsqueeze(-3), neighbor_frames.transpose(-1, -2))
        rotations = torch.where(
            rotation_valid[..., None, None],
            rotations,
            torch.eye(3, dtype=rotations.dtype, device=rotations.device),
        )
        quaternions = cls._quaternions(rotations).masked_fill(~rotation_valid.unsqueeze(-1), 0)
        return torch.cat((direction, quaternions), dim=-1)

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
            edges: Edge features with shape ``(B, L, K, H)``.
            neighbor_indices: Neighbor indices with shape ``(B, L, K)``.
            neighbor_valid: Real-neighbor mask with shape ``(B, L, K)`` (the encoder's
                ``attend_mask``).
        """

        backbone_coordinates = project_model_coordinates(backbone_coordinates, residue_mask, feature="C-alpha backbone")
        neighbor_distances, neighbor_indices, neighbor_valid = masked_knn(
            backbone_coordinates,
            residue_mask,
            self.num_neighbors,
            backend=self.backend,
        )
        adjacent = (
            residue_mask[:, :-1]
            & residue_mask[:, 1:]
            & (chain_index[:, :-1] == chain_index[:, 1:])
            & (residue_index[:, 1:] - residue_index[:, :-1] == 1)
        )
        endpoint = torch.zeros_like(backbone_coordinates[:, :1])
        previous = torch.cat((endpoint, backbone_coordinates[:, :-1] * adjacent.unsqueeze(-1)), dim=1)
        following = torch.cat((backbone_coordinates[:, 1:] * adjacent.unsqueeze(-1), endpoint), dim=1)
        endpoint_mask = torch.zeros_like(residue_mask[:, :1], dtype=torch.bool)
        previous_mask = torch.cat((endpoint_mask, adjacent), dim=1)
        following_mask = torch.cat((adjacent, endpoint_mask), dim=1)
        orientation = self._orientations(
            backbone_coordinates, neighbor_indices, adjacent, neighbor_valid, backend=self.backend
        )
        distance_features = torch.cat(
            (
                radial_basis(neighbor_distances).masked_fill(~neighbor_valid.unsqueeze(-1), 0),
                _pair_radial_basis(
                    previous,
                    previous,
                    neighbor_indices,
                    previous_mask,
                    previous_mask,
                    neighbor_valid,
                    backend=self.backend,
                ),
                _pair_radial_basis(
                    following,
                    following,
                    neighbor_indices,
                    following_mask,
                    following_mask,
                    neighbor_valid,
                    backend=self.backend,
                ),
                _pair_radial_basis(
                    previous,
                    backbone_coordinates,
                    neighbor_indices,
                    previous_mask,
                    residue_mask,
                    neighbor_valid,
                    backend=self.backend,
                ),
                _pair_radial_basis(
                    previous,
                    following,
                    neighbor_indices,
                    previous_mask,
                    following_mask,
                    neighbor_valid,
                    backend=self.backend,
                ),
                _pair_radial_basis(
                    backbone_coordinates,
                    previous,
                    neighbor_indices,
                    residue_mask,
                    previous_mask,
                    neighbor_valid,
                    backend=self.backend,
                ),
                _pair_radial_basis(
                    backbone_coordinates,
                    following,
                    neighbor_indices,
                    residue_mask,
                    following_mask,
                    neighbor_valid,
                    backend=self.backend,
                ),
                _pair_radial_basis(
                    following,
                    previous,
                    neighbor_indices,
                    following_mask,
                    previous_mask,
                    neighbor_valid,
                    backend=self.backend,
                ),
                _pair_radial_basis(
                    following,
                    backbone_coordinates,
                    neighbor_indices,
                    following_mask,
                    residue_mask,
                    neighbor_valid,
                    backend=self.backend,
                ),
            ),
            dim=-1,
        )
        neighbor_residue_index = gather_nodes(residue_index.unsqueeze(-1), neighbor_indices).squeeze(-1)
        residue_offsets = residue_index.unsqueeze(-1) - neighbor_residue_index
        neighbor_chain_index = gather_nodes(chain_index.unsqueeze(-1), neighbor_indices).squeeze(-1)
        same_chain = (chain_index.unsqueeze(-1) == neighbor_chain_index).long()
        position_features = self.relative_position(residue_offsets.long(), same_chain)
        edges = torch.cat((position_features, distance_features, orientation), dim=-1)
        return (
            self.edge_norm(self.edge_projection(edges.to(self.edge_projection.weight.dtype))),
            neighbor_indices,
            neighbor_valid,
        )


def _pair_radial_basis(
    source_coordinates: Tensor,
    target_coordinates: Tensor,
    neighbor_indices: Tensor,
    source_mask: Tensor,
    target_mask: Tensor,
    neighbor_valid: Tensor,
    num_bins: int = 16,
    *,
    backend: str = "auto",
) -> Tensor:
    r"""
    Encode observed source-to-neighbor-target distances with radial bases.
    """

    neighbor_coordinates = gather_nodes(target_coordinates, neighbor_indices)
    valid = (
        source_mask.bool().unsqueeze(-1)
        & gather_nodes(target_mask.bool().unsqueeze(-1), neighbor_indices).squeeze(-1)
        & neighbor_valid
    )
    displacement = (source_coordinates.unsqueeze(-2) - neighbor_coordinates).masked_fill(~valid.unsqueeze(-1), 0)
    neighbor_distances = smoothed_vector_norm(displacement, eps=1e-6, backend=backend)
    return radial_basis(neighbor_distances, num_bins).masked_fill(~valid.unsqueeze(-1), 0)


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
        sequence_tokens: Canonical classes with shape ``(batch_size, num_residues)``.
            Available rows must be exact embedding vocabulary indices; excluded payload
            is ignored before lookup without changing native canonical bindings.
        residue_mask: Valid-residue mask with shape ``(batch_size, num_residues)``.
        residue_index: Residue indices with shape ``(batch_size, num_residues)``.
        chain_index: Chain indices with shape ``(batch_size, num_residues)``.
        design_mask: Positions eligible for design with shape
            ``(batch_size, num_residues)``.
        component_index: Canonical component index aligned to every residue row;
            ``-1`` marks a transient source-numbering gap.
        tied_group_ids: Autoregressive unit ID for each residue row.
        tied_weights: Optional per-residue weights for tied sampling groups.
        allowed_token_mask: Optional allowed-token mask with shape
            ``(batch_size, num_residues, vocab_size)``.
        residue_logit_bias: Optional additive logit bias with shape
            ``(batch_size, num_residues, vocab_size)``.
        pssm_coefficients: Optional PSSM mixture coefficients with shape
            ``(batch_size, num_residues)``.
        pssm_probabilities: Optional PSSM probabilities with shape
            ``(batch_size, num_residues, vocab_size)``.
        pssm_log_odds: Optional PSSM log odds with shape
            ``(batch_size, num_residues, vocab_size)``.
        membrane_labels: Optional three-class labels aligned to residue rows,
            required by the membrane-conditioned architecture. Excluded residue payload
            is ignored before one-hot encoding.
    """

    backbone_coordinates: Tensor
    sequence_tokens: Tensor
    residue_mask: Tensor
    residue_index: Tensor
    chain_index: Tensor
    design_mask: Tensor
    component_index: Tensor
    tied_group_ids: Tensor
    tied_weights: Tensor | None = None
    allowed_token_mask: Tensor | None = None
    residue_logit_bias: Tensor | None = None
    pssm_coefficients: Tensor | None = None
    pssm_probabilities: Tensor | None = None
    pssm_log_odds: Tensor | None = None
    membrane_labels: Tensor | None = None


class ProteinMpnnOutput(TensorCollection):
    r"""
    Outputs from one ProteinMPNN forward pass.

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


class ProteinMpnnSamplingOutput(TensorCollection):
    r"""
    Outputs from ProteinMPNN autoregressive sampling.

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


class ProteinMpnnDesignOutput(NamedTuple):
    r"""
    Raw ProteinMPNN design execution tensors.

    Attributes:
        sample: Autoregressive sequence samples in generation order.
        input_score: Raw evaluation of the input sequence after sampling.
    """

    sample: ProteinMpnnSamplingOutput
    input_score: ProteinMpnnOutput
