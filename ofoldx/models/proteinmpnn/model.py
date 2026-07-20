r"""ProteinMPNN fixed-backbone sequence model."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ...data.components import get_component
from ...data.molecular import ComponentReference, MolecularSystem, Molecule, PolymerSequence
from ...data.structure import Structure
from ...modules.mpnn import (
    AMINO_ACID_COMPONENT_IDS,
    MpnnDecoderLayer,
    MpnnEncoderLayer,
    MpnnOutput,
    MpnnProteinFeatures,
    MpnnRelativePositionEncoding,
    MpnnSamplingOutput,
    _autoregressive_masks,
    _nearest_neighbors,
    _radial_basis,
    cat_neighbors_nodes,
    gather_edges,
    gather_nodes,
)
from ...results import (
    DesignCandidate,
    DesignResult,
    MpnnProbabilityResult,
    Score,
    ScoreResult,
)
from .config import ProteinMpnnConfig


class ProteinMpnnModel(nn.Module):
    r"""ProteinMPNN full-backbone or Cα-only sequence model."""

    config_class = ProteinMpnnConfig

    def __init__(self, config: ProteinMpnnConfig | None = None) -> None:
        super().__init__()
        self.config = ProteinMpnnConfig() if config is None else config
        feature_class = CaProteinMpnnFeatures if self.config.ca_only else MpnnProteinFeatures
        self.feature_encoder = feature_class(self.config.hidden_size, self.config.num_neighbors)
        self.edge_embedding = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.token_embedding = nn.Embedding(len(AMINO_ACID_COMPONENT_IDS) + 1, self.config.hidden_size)
        self.encoder_layers = nn.ModuleList(
            MpnnEncoderLayer(
                self.config.hidden_size,
                self.config.hidden_size * 2,
                dropout=self.config.dropout,
            )
            for _ in range(self.config.num_encoder_layers)
        )
        self.decoder_layers = nn.ModuleList(
            MpnnDecoderLayer(
                self.config.hidden_size,
                self.config.hidden_size * 3,
                dropout=self.config.dropout,
            )
            for _ in range(self.config.num_decoder_layers)
        )
        self.output_projection = nn.Linear(self.config.hidden_size, len(AMINO_ACID_COMPONENT_IDS) + 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        r"""Apply the released Xavier initialization to matrix parameters."""

        for parameter in self.parameters():
            if parameter.ndim > 1:
                nn.init.xavier_uniform_(parameter)

    @staticmethod
    def _validate_backbone_noise(backbone_noise: float) -> float:
        r"""Coerce and validate the inference-time coordinate perturbation."""

        backbone_noise = float(backbone_noise)
        if not math.isfinite(backbone_noise) or backbone_noise < 0:
            raise ValueError("backbone_noise must be a finite nonnegative number")
        return backbone_noise

    @staticmethod
    def _validate_num_decoding_orders(num_decoding_orders: int) -> int:
        r"""Validate the independent decoding-order evaluation count."""

        if type(num_decoding_orders) is not int or num_decoding_orders <= 0:
            raise ValueError("num_decoding_orders must be a positive integer")
        return num_decoding_orders

    @staticmethod
    def _require_single_target(input: ProteinMpnnInput) -> None:
        r"""Reject an execution-batch axis on one-target biological operations."""

        if input.backbone_coordinates.shape[0] != 1 or len(input.structures) != 1:
            raise ValueError("this ProteinMpnn operation returns one target and therefore accepts one input")

    @staticmethod
    def _repeat_input(input: ProteinMpnnInput, count: int) -> ProteinMpnnInput:
        r"""Repeat one biological input into ``count`` aligned evaluation rows."""

        def repeat(value: Tensor | None) -> Tensor | None:
            return None if value is None else value.repeat_interleave(count, dim=0)

        return ProteinMpnnInput(
            backbone_coordinates=repeat(input.backbone_coordinates),
            sequence_tokens=repeat(input.sequence_tokens),
            valid_mask=repeat(input.valid_mask),
            residue_indices=repeat(input.residue_indices),
            chain_indices=repeat(input.chain_indices),
            design_mask=repeat(input.design_mask),
            structures=input.structures * count,
            component_references=input.component_references * count,
            tied_group_ids=repeat(input.tied_group_ids),
            tied_weights=repeat(input.tied_weights),
            allowed_token_mask=repeat(input.allowed_token_mask),
            residue_logit_bias=repeat(input.residue_logit_bias),
            pssm_coefficients=repeat(input.pssm_coefficients),
            pssm_probabilities=repeat(input.pssm_probabilities),
            pssm_log_odds=repeat(input.pssm_log_odds),
        )

    def _probability_result(
        self,
        input: ProteinMpnnInput,
        logits: Tensor,
        decoding_ranks: Tensor,
    ) -> MpnnProbabilityResult:
        r"""Wrap aligned per-evaluation logits with the one target's identity."""

        return MpnnProbabilityResult(
            target=input.structures[0],
            component_references=input.component_references[0],
            sequence_tokens=input.sequence_tokens[0],
            valid_mask=input.valid_mask[0],
            design_mask=input.design_mask[0],
            logits=logits,
            decoding_ranks=decoding_ranks,
        )

    @staticmethod
    def _assert_decoding_ranks(
        decoding_ranks: Tensor,
        tied_group_ids: Tensor,
    ) -> None:
        r"""Assert the exact explicit-rank tensor and its causal-unit identity."""

        if decoding_ranks.shape != tied_group_ids.shape:
            raise ValueError("decoding_ranks has the wrong exact shape")
        if decoding_ranks.dtype != torch.long:
            raise TypeError("decoding_ranks must have dtype torch.long")

        length = tied_group_ids.shape[1]
        tied_groups = tied_groups_from_ids(tied_group_ids)
        for batch_index, batch_groups in enumerate(tied_groups):
            grouped = {member for group in batch_groups for member in group}
            causal_units = (
                *batch_groups,
                *((index,) for index in range(length) if index not in grouped),
            )
            for group in batch_groups:
                group_ranks = decoding_ranks[batch_index, list(group)]
                torch._assert(
                    torch.all(group_ranks == group_ranks[:1]),
                    "tied members must have equal decoding ranks",
                )
            representative_ranks = decoding_ranks[
                batch_index,
                [group[0] for group in causal_units],
            ]
            sorted_representative_ranks = torch.sort(representative_ranks).values
            torch._assert(
                torch.all(sorted_representative_ranks[1:] != sorted_representative_ranks[:-1]),
                "distinct causal units must have distinct decoding ranks",
            )

    @staticmethod
    def _generate_ranks(
        valid_mask: Tensor,
        design_mask: Tensor,
        tied_group_ids: Tensor,
        generator: torch.Generator | None,
    ) -> Tensor:
        r"""Draw fixed-first causal ranks directly from aligned causal-unit IDs.

        Every causal unit receives one random priority. Members of a unit share its
        priority through ``tied_group_ids`` and therefore share its rank, while every
        distinct unit receives a distinct rank. A unit is a generated unit only when
        all of its members are designed; those units sort after every fixed unit.
        """

        device = tied_group_ids.device
        batch, length = tied_group_ids.shape
        designed = (valid_mask & design_mask).to(torch.long)
        unit_designed = torch.ones(batch, length, dtype=torch.long, device=device)
        unit_designed.scatter_reduce_(1, tied_group_ids, designed, reduce="amin", include_self=True)
        priority = torch.rand(batch, length, device=device, generator=generator)
        unit_key = priority + 2 * unit_designed.to(priority.dtype)
        unit_order = torch.argsort(unit_key, dim=1)
        unit_rank = torch.empty_like(unit_order)
        unit_rank.scatter_(
            1,
            unit_order,
            torch.arange(length, device=device).expand(batch, length),
        )
        return torch.gather(unit_rank, 1, tied_group_ids)

    @staticmethod
    def _representative_positions(tied_group_ids: Tensor) -> Tensor:
        r"""Return the first-occurrence row of every causal unit for one input."""

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
        valid_mask: Tensor,
        residue_indices: Tensor,
        chain_indices: Tensor,
        *,
        backbone_noise: float,
        generator: torch.Generator | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        r"""Run the one coordinate encoder shared by every official operation."""

        if backbone_noise:
            backbone_coordinates = backbone_coordinates + backbone_noise * torch.randn(
                backbone_coordinates.shape,
                device=backbone_coordinates.device,
                dtype=backbone_coordinates.dtype,
                generator=generator,
            )
        mask = valid_mask.to(dtype=backbone_coordinates.dtype)
        edge_features, neighbor_indices = self.feature_encoder(
            backbone_coordinates,
            mask,
            residue_indices,
            chain_indices,
        )
        node_states = edge_features.new_zeros(
            edge_features.shape[0],
            edge_features.shape[1],
            self.config.hidden_size,
        )
        edge_states = self.edge_embedding(edge_features)
        attend_mask = gather_nodes(mask.unsqueeze(-1), neighbor_indices).squeeze(-1)
        attend_mask = mask.unsqueeze(-1) * attend_mask
        for layer in self.encoder_layers:
            node_states, edge_states = layer(
                node_states,
                edge_states,
                neighbor_indices,
                mask,
                attend_mask,
            )
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
        r"""Return raw teacher-forced logits for one exact causal ranking."""

        token_states = self.token_embedding(sequence_tokens)
        sequence_edges = cat_neighbors_nodes(token_states, edge_states, neighbor_indices)
        encoder_edges = cat_neighbors_nodes(torch.zeros_like(token_states), edge_states, neighbor_indices)
        encoder_context = cat_neighbors_nodes(node_states, encoder_edges, neighbor_indices)
        backward_mask, forward_mask = _autoregressive_masks(
            decoding_ranks,
            neighbor_indices,
            valid_mask.to(dtype=node_states.dtype),
        )
        forward_context = forward_mask * encoder_context
        for layer in self.decoder_layers:
            decoder_context = cat_neighbors_nodes(node_states, sequence_edges, neighbor_indices)
            node_states = layer(
                node_states,
                backward_mask * decoder_context + forward_context,
                valid_mask.to(dtype=node_states.dtype),
            )
        return self.output_projection(node_states)

    def _decode_sequence_free_logits(
        self,
        node_states: Tensor,
        edge_states: Tensor,
        neighbor_indices: Tensor,
        valid_mask: Tensor,
        decoding_ranks: Tensor,
    ) -> Tensor:
        r"""Return raw sequence-free logits under one causal ranking.

        Every position attends only its earlier-decoded neighbors through the
        forward encoder context; no decoded identity ever enters the decoder, so the
        computation stays order dependent without conditioning on any token.
        """

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

    def encode(
        self,
        input: ProteinMpnnInput,
        *,
        backbone_noise: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        r"""Return the shared node states, edge states, and neighbor indices."""

        backbone_noise = self._validate_backbone_noise(backbone_noise)
        return self._encode(
            input.backbone_coordinates,
            input.valid_mask,
            input.residue_indices,
            input.chain_indices,
            backbone_noise=backbone_noise,
            generator=generator,
        )

    def forward(
        self,
        input: ProteinMpnnInput,
        *,
        decoding_ranks: Tensor | None = None,
        backbone_noise: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> MpnnOutput:
        r"""Return the single teacher-forced differentiable sequence-conditioned pass."""

        backbone_noise = self._validate_backbone_noise(backbone_noise)
        if decoding_ranks is not None:
            self._assert_decoding_ranks(decoding_ranks, input.tied_group_ids)
        node_states, edge_states, neighbor_indices = self._encode(
            input.backbone_coordinates,
            input.valid_mask,
            input.residue_indices,
            input.chain_indices,
            backbone_noise=backbone_noise,
            generator=generator,
        )
        if decoding_ranks is None:
            decoding_ranks = self._generate_ranks(
                input.valid_mask,
                input.design_mask,
                input.tied_group_ids,
                generator,
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
        input: ProteinMpnnInput,
        *,
        num_decoding_orders: int = 1,
        decoding_ranks: Tensor | None = None,
        backbone_noise: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> MpnnProbabilityResult:
        r"""Return sequence-conditioned autoregressive probabilities over orders."""

        self._require_single_target(input)
        num_decoding_orders = self._validate_num_decoding_orders(num_decoding_orders)
        backbone_noise = self._validate_backbone_noise(backbone_noise)
        repeated = self._repeat_input(input, num_decoding_orders)
        output = self.forward(
            repeated,
            decoding_ranks=decoding_ranks,
            backbone_noise=backbone_noise,
            generator=generator,
        )
        return self._probability_result(input, output.logits, output.decoding_ranks)

    def sequence_free_autoregressive_probabilities(
        self,
        input: ProteinMpnnInput,
        *,
        num_decoding_orders: int = 1,
        decoding_ranks: Tensor | None = None,
        backbone_noise: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> MpnnProbabilityResult:
        r"""Return order-dependent sequence-free autoregressive probabilities."""

        self._require_single_target(input)
        num_decoding_orders = self._validate_num_decoding_orders(num_decoding_orders)
        backbone_noise = self._validate_backbone_noise(backbone_noise)
        repeated = self._repeat_input(input, num_decoding_orders)
        if decoding_ranks is not None:
            self._assert_decoding_ranks(decoding_ranks, repeated.tied_group_ids)
        node_states, edge_states, neighbor_indices = self._encode(
            repeated.backbone_coordinates,
            repeated.valid_mask,
            repeated.residue_indices,
            repeated.chain_indices,
            backbone_noise=backbone_noise,
            generator=generator,
        )
        if decoding_ranks is None:
            decoding_ranks = self._generate_ranks(
                repeated.valid_mask,
                repeated.design_mask,
                repeated.tied_group_ids,
                generator,
            )
        logits = self._decode_sequence_free_logits(
            node_states,
            edge_states,
            neighbor_indices,
            repeated.valid_mask,
            decoding_ranks,
        )
        return self._probability_result(input, logits, decoding_ranks)

    def _single_residue_probabilities(
        self,
        input: ProteinMpnnInput,
        *,
        num_decoding_orders: int,
        base_decoding_ranks: Tensor | None,
        backbone_noise: float,
        generator: torch.Generator | None,
        move_last: bool,
    ) -> MpnnProbabilityResult:
        r"""Evaluate every target moved exactly last or first under one base order."""

        self._require_single_target(input)
        num_decoding_orders = self._validate_num_decoding_orders(num_decoding_orders)
        backbone_noise = self._validate_backbone_noise(backbone_noise)
        repeated = self._repeat_input(input, num_decoding_orders)
        device = repeated.backbone_coordinates.device
        node_states, edge_states, neighbor_indices = self._encode(
            repeated.backbone_coordinates,
            repeated.valid_mask,
            repeated.residue_indices,
            repeated.chain_indices,
            backbone_noise=backbone_noise,
            generator=generator,
        )
        length = repeated.sequence_tokens.shape[1]
        vocabulary = len(AMINO_ACID_COMPONENT_IDS) + 1
        if base_decoding_ranks is None:
            base_priorities = torch.rand(
                num_decoding_orders,
                length,
                device=device,
                generator=generator,
            )
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
        logits = node_states.new_zeros(num_decoding_orders, length, vocabulary)
        effective_ranks = torch.empty(
            num_decoding_orders,
            length,
            length,
            dtype=torch.long,
            device=device,
        )
        for target in range(length):
            if base_decoding_ranks is None:
                order_values = base_priorities.clone()
                order_values[:, target] = torch.inf if move_last else -torch.inf
                order = torch.argsort(order_values, dim=-1)
                ranks = torch.empty_like(order, dtype=torch.long)
                ranks.scatter_(
                    1,
                    order,
                    torch.arange(length, device=device).expand_as(order),
                )
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
        input: ProteinMpnnInput,
        *,
        num_decoding_orders: int = 1,
        base_decoding_ranks: Tensor | None = None,
        backbone_noise: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> MpnnProbabilityResult:
        r"""Return target-last conditional probabilities for every row."""

        return self._single_residue_probabilities(
            input,
            num_decoding_orders=num_decoding_orders,
            base_decoding_ranks=base_decoding_ranks,
            backbone_noise=backbone_noise,
            generator=generator,
            move_last=True,
        )

    def unconditional_probabilities(
        self,
        input: ProteinMpnnInput,
        *,
        num_decoding_orders: int = 1,
        base_decoding_ranks: Tensor | None = None,
        backbone_noise: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> MpnnProbabilityResult:
        r"""Return target-first sequence-free probabilities for every row."""

        return self._single_residue_probabilities(
            input,
            num_decoding_orders=num_decoding_orders,
            base_decoding_ranks=base_decoding_ranks,
            backbone_noise=backbone_noise,
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
        backbone_noise: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> ScoreResult:
        r"""Score the input sequence over independently drawn decoding orders."""

        self._require_single_target(input)
        num_decoding_orders = self._validate_num_decoding_orders(num_decoding_orders)
        backbone_noise = self._validate_backbone_noise(backbone_noise)
        repeated = self._repeat_input(input, num_decoding_orders)
        log_probabilities = self.forward(
            repeated,
            decoding_ranks=decoding_ranks,
            backbone_noise=backbone_noise,
            generator=generator,
        ).log_probabilities
        losses = -torch.gather(
            log_probabilities,
            -1,
            repeated.sequence_tokens.unsqueeze(-1),
        ).squeeze(-1)
        valid = repeated.valid_mask.to(dtype=losses.dtype)
        designed = (repeated.valid_mask & repeated.design_mask).to(dtype=losses.dtype)
        designed_values = (losses * designed).sum(dim=-1) / designed.sum(dim=-1)
        global_values = (losses * valid).sum(dim=-1) / valid.sum(dim=-1)
        host_result = (
            torch.cat(
                (
                    designed_values,
                    global_values,
                    input.design_mask[0],
                    input.valid_mask[0],
                )
            )
            .detach()
            .cpu()
        )
        designed_evaluations = host_result[:num_decoding_orders].numpy()
        global_evaluations = host_result[num_decoding_orders : 2 * num_decoding_orders].numpy()
        length = input.sequence_tokens.shape[1]
        host_design_mask = host_result[2 * num_decoding_orders : 2 * num_decoding_orders + length].bool()
        host_valid_mask = host_result[2 * num_decoding_orders + length :].bool()
        references = input.component_references[0]
        designed_scope = frozenset(
            reference
            for reference, selected, present in zip(
                references,
                host_design_mask.tolist(),
                host_valid_mask.tolist(),
            )
            if reference is not None and selected and present
        )
        global_scope = frozenset(
            reference
            for reference, present in zip(references, host_valid_mask.tolist())
            if reference is not None and present
        )
        return ScoreResult(
            target=input.structures[0],
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
        residue_logit_bias: Tensor | None,
        pssm_coefficient: Tensor | None,
        pssm_probabilities: Tensor | None,
        pssm_mixture: float,
        hard_support: Tensor,
    ) -> Tensor:
        r"""Apply each soft control once, then intersect final hard support."""

        if global_logit_bias is not None:
            logits = logits + global_logit_bias
        if residue_logit_bias is not None:
            logits = logits + residue_logit_bias
        probabilities = F.softmax(logits / temperature, dim=-1)
        if pssm_probabilities is not None and pssm_mixture:
            coefficient = (
                torch.ones(logits.shape[0], device=logits.device, dtype=logits.dtype)
                if pssm_coefficient is None
                else pssm_coefficient
            )
            mixture = pssm_mixture * coefficient
            probabilities = (1 - mixture.unsqueeze(-1)) * probabilities + mixture.unsqueeze(-1) * pssm_probabilities

        probabilities = probabilities * hard_support
        return probabilities / probabilities.sum(dim=-1, keepdim=True)

    @staticmethod
    def _hard_support(
        valid_mask: Tensor,
        allowed_token_mask: Tensor | None,
        pssm_log_odds: Tensor | None,
        global_omit_token_mask: Tensor | None,
        pssm_log_odds_threshold: float | None,
    ) -> Tensor:
        r"""Construct final per-row support once before autoregressive decoding.

        The raw support starts all-True over every generatable class, including the
        twenty-first (index 20 = ``AMINO_ACID_UNKNOWN_TOKEN`` = X / UNK). X is a
        reachable generated class of the raw Model and is only removed when
        ``global_omit_token_mask``, ``allowed_token_mask``, or the PSSM log-odds
        threshold actually remove it; the Pipeline's default ``("X",)`` omission is
        an operation default rather than a permanent support reduction.
        """

        support = torch.ones(
            *valid_mask.shape,
            len(AMINO_ACID_COMPONENT_IDS) + 1,
            dtype=torch.bool,
            device=valid_mask.device,
        )
        if global_omit_token_mask is not None:
            support &= ~global_omit_token_mask.bool()
        if allowed_token_mask is not None:
            support &= allowed_token_mask.bool()
        if pssm_log_odds is not None and pssm_log_odds_threshold is not None:
            support &= pssm_log_odds > pssm_log_odds_threshold
        return support

    @torch.inference_mode()
    def sample(
        self,
        input: ProteinMpnnInput,
        *,
        num_samples: int = 1,
        temperature: float = 0.1,
        decoding_ranks: Tensor | None = None,
        backbone_noise: float = 0.0,
        global_logit_bias: Tensor | None = None,
        global_omit_token_mask: Tensor | None = None,
        pssm_mixture: float = 0.0,
        pssm_log_odds_threshold: float | None = None,
        generator: torch.Generator | None = None,
    ) -> MpnnSamplingOutput:
        r"""Sample sequences, resolving tied and untied groups on one causal path."""

        self._require_single_target(input)
        if type(num_samples) is not int or num_samples <= 0:
            raise ValueError("num_samples must be a positive integer")
        temperature = float(temperature)
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be a finite positive number")
        backbone_noise = self._validate_backbone_noise(backbone_noise)
        pssm_mixture = float(pssm_mixture)
        if not 0 <= pssm_mixture <= 1:
            raise ValueError("pssm_mixture must be between zero and one")
        if pssm_log_odds_threshold is not None:
            pssm_log_odds_threshold = float(pssm_log_odds_threshold)
            if not math.isfinite(pssm_log_odds_threshold):
                raise ValueError("pssm_log_odds_threshold must be finite")

        repeated = self._repeat_input(input, num_samples)
        device = repeated.backbone_coordinates.device
        if decoding_ranks is not None:
            self._assert_decoding_ranks(decoding_ranks, repeated.tied_group_ids)

        hard_support = self._hard_support(
            repeated.valid_mask,
            repeated.allowed_token_mask,
            repeated.pssm_log_odds,
            global_omit_token_mask,
            pssm_log_odds_threshold,
        )
        node_states, edge_states, neighbor_indices = self._encode(
            repeated.backbone_coordinates,
            repeated.valid_mask,
            repeated.residue_indices,
            repeated.chain_indices,
            backbone_noise=backbone_noise,
            generator=generator,
        )
        length = repeated.sequence_tokens.shape[1]
        vocabulary = len(AMINO_ACID_COMPONENT_IDS) + 1
        ranks = (
            self._generate_ranks(
                repeated.valid_mask,
                repeated.design_mask,
                repeated.tied_group_ids,
                generator,
            )
            if decoding_ranks is None
            else decoding_ranks
        )
        representative_positions = self._representative_positions(input.tied_group_ids)
        ordered_group_ranks = torch.sort(ranks[:, representative_positions], dim=-1).values
        num_units = ordered_group_ranks.shape[1]

        backward_mask, forward_mask = _autoregressive_masks(
            ranks,
            neighbor_indices,
            repeated.valid_mask.to(dtype=node_states.dtype),
        )
        token_states = torch.zeros_like(node_states)
        encoder_edges = cat_neighbors_nodes(token_states, edge_states, neighbor_indices)
        encoder_context = cat_neighbors_nodes(node_states, encoder_edges, neighbor_indices)
        forward_context = forward_mask * encoder_context
        layer_states = [node_states] + [torch.zeros_like(node_states) for _ in self.decoder_layers]
        sequences = repeated.sequence_tokens.clone()
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
            mask_at_position = repeated.valid_mask[batch_indices, positions, None].to(node_states.dtype)
            for layer_index, layer in enumerate(self.decoder_layers):
                decoder_context = cat_neighbors_nodes(
                    layer_states[layer_index][batch_indices],
                    sequence_edge,
                    neighbor_index,
                )
                updated = layer(
                    layer_states[layer_index][batch_indices, positions].unsqueeze(1),
                    backward_mask[batch_indices, positions].unsqueeze(1) * decoder_context + encoder_at_position,
                    mask_at_position,
                )
                layer_states[layer_index + 1][batch_indices, positions] = updated[:, 0]

            logits = self.output_projection(layer_states[-1][batch_indices, positions])
            log_probabilities[batch_indices, positions] = F.log_softmax(logits, dim=-1)
            member_counts = member_mask.sum(dim=-1)
            if repeated.tied_weights is None:
                # Default coefficient 1/len(G) per member: the equal mean of the
                # member logits, matching upstream tied decoding with unit betas.
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
            group_support = torch.segment_reduce(
                hard_support[batch_indices, positions].to(logits.dtype),
                "min",
                lengths=member_counts,
            ).bool()
            member_selected = (
                repeated.valid_mask[batch_indices, positions] & repeated.design_mask[batch_indices, positions]
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
            fixed_mask = member_mask & ~(repeated.valid_mask & repeated.design_mask)
            anchor_positions = fixed_mask.long().argmax(dim=-1)
            anchor_tokens = repeated.sequence_tokens[sample_indices, anchor_positions]
            anchored = ~active & has_designed_member & (member_counts > 1)
            anchor_allowed = group_support[sample_indices, anchor_tokens]
            torch._assert(
                torch.all(~anchored | anchor_allowed),
                "a tied anchor must remain in final hard support",
            )

            probabilities = self._sampling_distribution(
                combined_logits,
                temperature=temperature,
                global_logit_bias=global_logit_bias,
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
                hard_support=group_support,
            )
            drawn_tokens = torch.multinomial(
                probabilities[active],
                1,
                generator=generator,
            ).squeeze(-1)
            group_tokens = anchor_tokens.clone()
            group_tokens[active] = drawn_tokens
            sequences[batch_indices, positions] = group_tokens[batch_indices]
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

        return MpnnSamplingOutput(
            sequences=sequences,
            log_probabilities=log_probabilities,
            sampling_probabilities=sampling_probabilities,
            decoding_ranks=ranks,
        )

    @staticmethod
    def _realize_sequence(
        system: MolecularSystem,
        component_references: tuple[tuple[str, int] | None, ...],
        sampled_tokens: Tensor,
        design_mask: Tensor,
    ) -> MolecularSystem:
        r"""Realize one candidate while preserving entity and bond identity."""

        replacements = {
            reference: get_component(AMINO_ACID_COMPONENT_IDS[int(token)])
            for reference, token, selected in zip(
                component_references,
                sampled_tokens.tolist(),
                design_mask.tolist(),
            )
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

    @torch.inference_mode()
    def design(
        self,
        input: ProteinMpnnInput,
        *,
        num_candidates: int = 1,
        num_decoding_orders: int = 1,
        temperature: float = 0.1,
        backbone_noise: float = 0.0,
        global_logit_bias: Tensor | None = None,
        global_omit_token_mask: Tensor | None = None,
        pssm_mixture: float = 0.0,
        pssm_log_odds_threshold: float | None = None,
        generator: torch.Generator | None = None,
    ) -> ProteinMpnnDesignResult:
        r"""Design complete sequence candidates and retain the native input score."""

        self._require_single_target(input)
        if type(num_candidates) is not int or num_candidates <= 0:
            raise ValueError("num_candidates must be a positive integer")
        num_decoding_orders = self._validate_num_decoding_orders(num_decoding_orders)
        temperature = float(temperature)
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be a finite positive number")
        backbone_noise = self._validate_backbone_noise(backbone_noise)
        pssm_mixture = float(pssm_mixture)
        if not 0 <= pssm_mixture <= 1:
            raise ValueError("pssm_mixture must be between zero and one")
        stream_seeds = torch.randint(
            torch.iinfo(torch.int64).max,
            (2,),
            device=input.backbone_coordinates.device,
            generator=generator,
        ).tolist()
        candidate_generator = torch.Generator(device=input.backbone_coordinates.device).manual_seed(stream_seeds[0])
        score_generator = torch.Generator(device=input.backbone_coordinates.device).manual_seed(stream_seeds[1])
        sampled = self.sample(
            input,
            num_samples=num_candidates,
            temperature=temperature,
            backbone_noise=backbone_noise,
            global_logit_bias=global_logit_bias,
            global_omit_token_mask=global_omit_token_mask,
            pssm_mixture=pssm_mixture,
            pssm_log_odds_threshold=pssm_log_odds_threshold,
            generator=candidate_generator,
        )
        input_score = self.score(
            input,
            num_decoding_orders=num_decoding_orders,
            backbone_noise=backbone_noise,
            generator=score_generator,
        )
        losses = -torch.gather(
            sampled.log_probabilities,
            -1,
            sampled.sequences.unsqueeze(-1),
        ).squeeze(-1)
        valid = input.valid_mask.repeat_interleave(num_candidates, dim=0).to(losses.dtype)
        designed = (input.valid_mask & input.design_mask).repeat_interleave(num_candidates, dim=0).to(losses.dtype)
        designed_nll = (losses * designed).sum(dim=-1) / designed.sum(dim=-1)
        global_nll = (losses * valid).sum(dim=-1) / valid.sum(dim=-1)
        recovery = (sampled.sequences == input.sequence_tokens.repeat_interleave(num_candidates, dim=0)).to(
            losses.dtype
        ).mul(designed).sum(dim=-1) / designed.sum(dim=-1)
        length = sampled.sequences.shape[1]
        host_result = (
            torch.cat(
                (
                    sampled.sequences.reshape(-1),
                    input.design_mask[0],
                    input.valid_mask[0],
                    torch.stack((designed_nll, global_nll, recovery), dim=-1).reshape(-1),
                )
            )
            .detach()
            .cpu()
        )
        sequence_values = num_candidates * length
        host_sequences = host_result[:sequence_values].reshape(num_candidates, length).long()
        host_design_mask = host_result[sequence_values : sequence_values + length].bool()
        host_valid_mask = host_result[sequence_values + length : sequence_values + 2 * length].bool()
        host_scores = host_result[sequence_values + 2 * length :].reshape(num_candidates, 3)
        references = input.component_references[0]
        designed_scope = frozenset(
            reference
            for reference, selected, present in zip(
                references,
                host_design_mask.tolist(),
                host_valid_mask.tolist(),
            )
            if reference is not None and selected and present
        )
        global_scope = frozenset(
            reference
            for reference, present in zip(references, host_valid_mask.tolist())
            if reference is not None and present
        )
        candidates = []
        for index in range(num_candidates):
            system = self._realize_sequence(
                input.structures[0].system,
                references,
                host_sequences[index],
                host_design_mask,
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


def _pair_radial_basis(
    source_coordinates: Tensor,
    target_coordinates: Tensor,
    neighbor_indices: Tensor,
    num_bins: int = 16,
) -> Tensor:
    r"""Encode source-to-neighbor-target distances with radial bases."""

    neighbor_coordinates = gather_nodes(target_coordinates, neighbor_indices)
    neighbor_distances = torch.sqrt(
        torch.sum((source_coordinates.unsqueeze(-2) - neighbor_coordinates) ** 2, dim=-1) + 1e-6
    )
    return _radial_basis(neighbor_distances, num_bins)


class CaProteinMpnnFeatures(nn.Module):
    r"""Released Cα-only ProteinMPNN edge feature graph."""

    def __init__(self, hidden_size: int, num_neighbors: int) -> None:
        super().__init__()
        self.num_neighbors = num_neighbors
        self.relative_position = MpnnRelativePositionEncoding(16)
        self.edge_projection = nn.Linear(16 + 9 * 16 + 7, hidden_size, bias=False)
        self.edge_norm = nn.LayerNorm(hidden_size)

    @staticmethod
    def quaternions(rotations: Tensor) -> Tensor:
        r"""Convert rotation matrices to the released quaternion features."""

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
    def orientations(
        cls,
        alpha_carbons: Tensor,
        neighbor_indices: Tensor,
    ) -> Tensor:
        r"""Return the live neighbor-orientation features."""

        displacements = alpha_carbons[:, 1:] - alpha_carbons[:, :-1]
        displacement_norms = torch.linalg.vector_norm(displacements, dim=-1)
        consecutive = (displacement_norms > 3.6) & (displacement_norms < 4.0)
        unit = F.normalize(displacements * consecutive.unsqueeze(-1), dim=-1)
        u2 = unit[:, :-2]
        u1 = unit[:, 1:-1]
        n2 = F.normalize(torch.cross(u2, u1, dim=-1), dim=-1)

        first_axis = F.normalize(u2 - u1, dim=-1)
        frames = torch.stack((first_axis, n2, torch.cross(first_axis, n2, dim=-1)), dim=-2)
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
        return torch.cat((direction, cls.quaternions(rotations)), dim=-1)

    def forward(
        self,
        backbone_coordinates: Tensor,
        valid_mask: Tensor,
        residue_indices: Tensor,
        chain_indices: Tensor,
    ) -> tuple[Tensor, Tensor]:
        r"""Build the nine-pair Cα-only neighbor graph."""

        neighbor_distances, neighbor_indices = _nearest_neighbors(
            backbone_coordinates,
            valid_mask,
            self.num_neighbors,
        )
        previous = torch.zeros_like(backbone_coordinates)
        previous[:, 1:] = backbone_coordinates[:, :-1]
        following = torch.zeros_like(backbone_coordinates)
        following[:, :-1] = backbone_coordinates[:, 1:]
        orientation = self.orientations(backbone_coordinates, neighbor_indices)
        distance_features = torch.cat(
            (
                _radial_basis(neighbor_distances),
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
        residue_offsets = residue_indices[:, :, None] - residue_indices[:, None, :]
        residue_offsets = gather_edges(residue_offsets.unsqueeze(-1), neighbor_indices).squeeze(-1)
        same_chain = (chain_indices[:, :, None] == chain_indices[:, None, :]).long()
        same_chain = gather_edges(same_chain.unsqueeze(-1), neighbor_indices).squeeze(-1)
        position_features = self.relative_position(residue_offsets.long(), same_chain)
        edges = torch.cat((position_features, distance_features, orientation), dim=-1)
        return self.edge_norm(self.edge_projection(edges)), neighbor_indices


# ---------------------------------------------------------------------------
# Family-owned tensor and result boundaries for ProteinMPNN (folded from
# the former contract.py).
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class ProteinMpnnInput:
    r"""Tensors and canonical identities consumed by ProteinMPNN operations."""

    backbone_coordinates: Tensor
    sequence_tokens: Tensor
    valid_mask: Tensor
    residue_indices: Tensor
    chain_indices: Tensor
    design_mask: Tensor
    structures: tuple[Structure, ...]
    component_references: tuple[tuple[ComponentReference | None, ...], ...]
    tied_group_ids: Tensor
    tied_weights: Tensor | None = None
    allowed_token_mask: Tensor | None = None
    residue_logit_bias: Tensor | None = None
    pssm_coefficients: Tensor | None = None
    pssm_probabilities: Tensor | None = None
    pssm_log_odds: Tensor | None = None


def tied_groups_from_ids(
    tied_group_ids: Tensor,
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    r"""Reconstruct per-batch tied member groups from dense causal-unit IDs.

    Sampling and probability hot paths consume ``tied_group_ids`` directly; this
    helper serves non-compiled orchestration that still reasons about explicit
    member groups. Every returned group holds more than one row and lists members
    in ascending row order, matching the dense first-row ID assignment; untied
    singleton units are omitted.
    """

    batches: list[tuple[tuple[int, ...], ...]] = []
    for row in tied_group_ids.tolist():
        members_by_id: dict[int, list[int]] = {}
        for index, group_id in enumerate(row):
            members_by_id.setdefault(group_id, []).append(index)
        batches.append(tuple(tuple(members) for members in members_by_id.values() if len(members) > 1))
    return tuple(batches)


@dataclass(eq=False)
class ProteinMpnnDesignResult(DesignResult):
    r"""ProteinMPNN designs together with the native input-sequence score."""

    input_score: ScoreResult


def nll_loss(sequence_tokens: Tensor, log_probabilities: Tensor) -> Tensor:
    r"""Return the per-token negative log likelihood of the observed tokens.

    Gathers ``-log_probabilities`` at each target token, matching upstream
    ProteinMPNN's ``NLLLoss(reduction="none")`` without the caller-owned mask
    reduction. The result has shape ``sequence_tokens.shape``.
    """

    return -log_probabilities.gather(-1, sequence_tokens.unsqueeze(-1)).squeeze(-1)


def smoothed_loss(
    sequence_tokens: Tensor,
    log_probabilities: Tensor,
    *,
    weight: float = 0.1,
) -> Tensor:
    r"""Return the per-token label-smoothed cross entropy of the observed tokens.

    Adds ``weight / 21`` to the one-hot target and renormalizes each row before
    the cross entropy, matching upstream ProteinMPNN's label smoothing. At
    ``weight=0`` this reduces to :func:`nll_loss`. The result has shape
    ``sequence_tokens.shape``.
    """

    num_classes = log_probabilities.shape[-1]
    target = F.one_hot(sequence_tokens, num_classes).to(log_probabilities.dtype)
    target = target + weight / num_classes
    target = target / target.sum(dim=-1, keepdim=True)
    return -(target * log_probabilities).sum(dim=-1)
