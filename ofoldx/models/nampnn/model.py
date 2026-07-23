r"""NA-MPNN inverse-folding model for protein, DNA, RNA, and mixed-polymer structures."""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ...data.components import get_component
from ...data.conditions import AllowedResidueIdentities, TiedResidueIdentities
from ...data.molecular import ComponentReference, MolecularSystem
from ...data.structure import Structure
from ...model import Model
from ...modules.backend import LayerNorm
from ...modules.mpnn import (
    MpnnDecoderBlock,
    MpnnEncoderBlock,
    MpnnRelativePositionEncoding,
    autoregressive_masks,
    cat_neighbors_nodes,
    gather_edges,
    gather_nodes,
    nearest_neighbors,
    nll_loss,
    radial_basis,
    virtual_beta_carbon,
)
from ...results import DesignCandidate, DesignResult, Metric, MpnnProbabilityResult, Score, ScoreResult
from .config import NaMpnnConfig

MIXED_POLYMER_ATOM_NAMES = (
    "N",
    "CA",
    "C",
    "O",
    "OP1",
    "OP2",
    "P",
    "O5'",
    "C5'",
    "C4'",
    "O4'",
    "C3'",
    "O3'",
    "C2'",
    "O2'",
    "C1'",
)
MIXED_POLYMER_ATOM_TO_INDEX = {name: index for index, name in enumerate(MIXED_POLYMER_ATOM_NAMES)}
MIXED_POLYMER_COMPONENT_IDS = (
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
    "UNK",
    "DA",
    "DC",
    "DG",
    "DT",
    "DX",
    "A",
    "C",
    "G",
    "U",
    "RX",
    "MAS",
    "PAD",
)
MIXED_POLYMER_TOKEN_BY_COMPONENT = {name: index for index, name in enumerate(MIXED_POLYMER_COMPONENT_IDS)}
PAD_TOKEN = MIXED_POLYMER_TOKEN_BY_COMPONENT["PAD"]
INVALID_GENERATION_TOKEN_IDS = (
    MIXED_POLYMER_TOKEN_BY_COMPONENT["UNK"],
    MIXED_POLYMER_TOKEN_BY_COMPONENT["DX"],
    MIXED_POLYMER_TOKEN_BY_COMPONENT["RX"],
    MIXED_POLYMER_TOKEN_BY_COMPONENT["MAS"],
    MIXED_POLYMER_TOKEN_BY_COMPONENT["PAD"],
)
# Contiguous generatable token columns per polymer type. RNA shares the DNA columns
# in the shared-token mode and owns tokens 26..29 in separate-token mode.
PROTEIN_GENERATABLE_TOKEN_IDS = tuple(range(0, 20))
DNA_GENERATABLE_TOKEN_IDS = tuple(range(21, 25))
RNA_SHARED_GENERATABLE_TOKEN_IDS = tuple(range(21, 25))
RNA_SEPARATE_GENERATABLE_TOKEN_IDS = tuple(range(26, 30))
_RNA_SHARED_INPUT_TOKENS = frozenset((*RNA_SHARED_GENERATABLE_TOKEN_IDS, MIXED_POLYMER_TOKEN_BY_COMPONENT["DX"]))


class NaMpnnModel(Model):
    r"""Message-passing Model for fixed-backbone protein and nucleic-acid design."""

    config_class = NaMpnnConfig
    config: NaMpnnConfig

    def __init__(self, config: NaMpnnConfig | None = None) -> None:
        config = NaMpnnConfig() if config is None else config
        super().__init__(config)
        self.vocab_size = self.config.vocab_size
        self.feature_encoder = NaMpnnFeatures(self.config)
        self.node_embedding = nn.Linear(self.config.feature_size, self.config.hidden_size)
        self.edge_embedding = nn.Linear(self.config.feature_size, self.config.hidden_size)
        self.encoder_blocks = nn.ModuleList(
            MpnnEncoderBlock(self.config.hidden_size, self.config.hidden_size * 2, dropout=self.config.dropout)
            for _ in range(self.config.num_encoder_blocks)
        )
        self.token_embedding = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.decoder_blocks = nn.ModuleList(
            MpnnDecoderBlock(self.config.hidden_size, self.config.hidden_size * 3, dropout=self.config.dropout)
            for _ in range(self.config.num_decoder_blocks)
        )
        self.output_projection = nn.Linear(self.config.hidden_size, self.config.vocab_size)
        self.post_init()

    def initialize_parameters(self) -> None:
        r"""Apply the Xavier-uniform matrix initialization."""

        for parameter in self.parameters():
            if parameter.ndim > 1:
                nn.init.xavier_uniform_(parameter)

    # ------------------------------------------------------------------
    # Shared validation and evaluation-axis helpers.
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_num_decoding_orders(num_decoding_orders: int) -> int:
        r"""Validate the independent decoding-order evaluation count."""

        return _positive_integer(num_decoding_orders, "num_decoding_orders")

    @staticmethod
    def _require_single_target(input: NaMpnnInput) -> None:
        r"""Reject an execution-batch axis on one-target biological operations."""

        if input.atom_coordinates.shape[0] != 1 or len(input.structures) != 1:
            raise ValueError("this NaMpnn operation returns one target and therefore accepts one input")

    @staticmethod
    def _repeat_input(input: NaMpnnInput, count: int) -> NaMpnnInput:
        r"""Repeat one biological input into ``count`` aligned evaluation rows."""

        def repeat(value: Tensor | None) -> Tensor | None:
            return None if value is None else value.repeat_interleave(count, dim=0)

        return NaMpnnInput(
            atom_coordinates=repeat(input.atom_coordinates),
            atom_mask=repeat(input.atom_mask),
            sequence_tokens=repeat(input.sequence_tokens),
            residue_mask=repeat(input.residue_mask),
            residue_index=repeat(input.residue_index),
            chain_index=repeat(input.chain_index),
            polymer_type_index=repeat(input.polymer_type_index),
            protein_mask=repeat(input.protein_mask),
            dna_mask=repeat(input.dna_mask),
            rna_mask=repeat(input.rna_mask),
            design_positions=repeat(input.design_positions),
            tied_group_ids=repeat(input.tied_group_ids),
            structures=input.structures * count,
            component_references=input.component_references * count,
            polymer_types=input.polymer_types * count,
            conditions=input.conditions * count,
            tied_weights=repeat(input.tied_weights),
            residue_logit_bias=repeat(input.residue_logit_bias),
            allowed_token_mask=repeat(input.allowed_token_mask),
            pair_logit_bias=repeat(input.pair_logit_bias),
        )

    def _probability_result(
        self,
        input: NaMpnnInput,
        logits: Tensor,
        decoding_ranks: Tensor,
    ) -> MpnnProbabilityResult:
        r"""Wrap aligned per-evaluation logits with the one target's identity."""

        return MpnnProbabilityResult(
            target=input.structures[0],
            component_references=input.component_references[0],
            sequence_tokens=input.sequence_tokens[0],
            residue_mask=input.residue_mask[0],
            design_positions=input.design_positions[0],
            logits=logits,
            decoding_ranks=decoding_ranks,
        )

    @staticmethod
    def _generate_ranks(
        residue_mask: Tensor,
        design_positions: Tensor,
        tied_group_ids: Tensor,
        generator: torch.Generator | None,
    ) -> Tensor:
        r"""Draw fixed-first causal ranks directly from aligned causal-unit IDs.

        Every causal unit receives one random priority; members of a unit share its
        priority through ``tied_group_ids`` and therefore share its rank, while
        every distinct unit receives a distinct rank. A unit is generated only when
        all of its members are designed; those units sort after every fixed unit.
        """

        device = tied_group_ids.device
        batch, length = tied_group_ids.shape
        designed = (residue_mask & design_positions).to(torch.long)
        unit_designed = torch.ones(batch, length, dtype=torch.long, device=device)
        unit_designed.scatter_reduce_(1, tied_group_ids, designed, reduce="amin", include_self=True)
        priority = torch.rand(batch, length, device=device, generator=generator)
        unit_key = priority + 2 * unit_designed.to(priority.dtype)
        unit_order = torch.argsort(unit_key, dim=1)
        unit_rank = torch.empty_like(unit_order)
        unit_rank.scatter_(1, unit_order, torch.arange(length, device=device).expand(batch, length))
        return torch.gather(unit_rank, 1, tied_group_ids)

    @staticmethod
    def _shared_ranks(
        residue_mask: Tensor,
        design_positions: Tensor,
        tied_group_ids: Tensor,
        generator: torch.Generator | None,
    ) -> Tensor:
        r"""Draw one causal ranking shared across the sampling evaluation axis.

        The sampler decodes one biological structure many times with an
        order shared across the draw axis so tied members stay simultaneous; only
        the multinomial draw differs per sample. Members of a unit share its rank
        and generated units sort after every fixed unit.
        """

        row = NaMpnnModel._generate_ranks(
            residue_mask[:1],
            design_positions[:1],
            tied_group_ids[:1],
            generator,
        )
        return row.expand(tied_group_ids.shape[0], tied_group_ids.shape[1]).contiguous()

    # ------------------------------------------------------------------
    # Encoder / teacher-forced decoder.
    # ------------------------------------------------------------------
    def _encode(
        self,
        input: NaMpnnInput,
        *,
        polymer_noise: tuple[float, float, float] = (0.0, 0.0, 0.0),
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        r"""Run the one coordinate encoder shared by every operation."""

        coordinates = input.atom_coordinates
        protein_noise, dna_noise, rna_noise = polymer_noise
        if protein_noise or dna_noise or rna_noise:
            scale = (
                input.protein_mask.to(coordinates.dtype) * protein_noise
                + input.dna_mask.to(coordinates.dtype) * dna_noise
                + input.rna_mask.to(coordinates.dtype) * rna_noise
            )
            noise = torch.randn(
                coordinates.shape,
                device=coordinates.device,
                dtype=coordinates.dtype,
                generator=generator,
            )
            coordinates = coordinates + scale[:, :, None, None] * noise
        node_features, edge_features, neighbor_indices = self.feature_encoder(
            coordinates,
            input.atom_mask,
            input.residue_mask,
            input.residue_index,
            input.chain_index,
            input.polymer_type_index,
            input.protein_mask,
            input.dna_mask,
            input.rna_mask,
        )
        node_states = self.node_embedding(node_features)
        edge_states = self.edge_embedding(edge_features)
        mask = input.residue_mask.to(dtype=node_states.dtype)
        attend_mask = gather_nodes(mask.unsqueeze(-1), neighbor_indices).squeeze(-1)
        attend_mask = mask.unsqueeze(-1) * attend_mask
        for block in self.encoder_blocks:
            node_states, edge_states = block(node_states, edge_states, neighbor_indices, mask, attend_mask)
        return node_states, edge_states, neighbor_indices

    def _decode_logits(
        self,
        node_states: Tensor,
        edge_states: Tensor,
        neighbor_indices: Tensor,
        sequence_tokens: Tensor,
        residue_mask: Tensor,
        decoding_ranks: Tensor,
    ) -> Tensor:
        r"""Return raw teacher-forced logits for one exact causal ranking."""

        token_states = self.token_embedding(sequence_tokens)
        sequence_edges = cat_neighbors_nodes(token_states, edge_states, neighbor_indices)
        encoder_edges = cat_neighbors_nodes(torch.zeros_like(token_states), edge_states, neighbor_indices)
        encoder_context = cat_neighbors_nodes(node_states, encoder_edges, neighbor_indices)
        backward_mask, forward_mask = autoregressive_masks(
            decoding_ranks,
            neighbor_indices,
            residue_mask.to(dtype=node_states.dtype),
        )
        forward_context = forward_mask * encoder_context
        mask = residue_mask.to(node_states.dtype)
        for block in self.decoder_blocks:
            decoder_context = cat_neighbors_nodes(node_states, sequence_edges, neighbor_indices)
            node_states = block(node_states, backward_mask * decoder_context + forward_context, mask)
        return self.output_projection(node_states)

    def _decode_unconditional_logits(
        self,
        node_states: Tensor,
        edge_states: Tensor,
        neighbor_indices: Tensor,
        residue_mask: Tensor,
    ) -> Tensor:
        r"""Return sequence-free all-equal-rank logits: every row sees no decoded neighbor."""

        mask = residue_mask.to(dtype=node_states.dtype)
        encoder_edges = cat_neighbors_nodes(torch.zeros_like(node_states), edge_states, neighbor_indices)
        encoder_context = cat_neighbors_nodes(node_states, encoder_edges, neighbor_indices)
        forward_context = mask[:, :, None, None] * encoder_context
        states = node_states
        for block in self.decoder_blocks:
            states = block(states, forward_context, mask)
        return self.output_projection(states)

    def encode(
        self,
        input: NaMpnnInput,
        *,
        protein_noise: float = 0.0,
        dna_noise: float = 0.0,
        rna_noise: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        r"""Return the shared node states, edge states, and neighbor indices."""

        return self._encode(
            input,
            polymer_noise=(float(protein_noise), float(dna_noise), float(rna_noise)),
            generator=generator,
        )

    def forward(
        self,
        input: NaMpnnInput,
        *,
        decoding_ranks: Tensor | None = None,
        decode_protein_first: bool = False,
        protein_noise: float = 0.0,
        dna_noise: float = 0.0,
        rna_noise: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> NaMpnnOutput:
        r"""Return the single teacher-forced differentiable sequence-conditioned pass.

        ``decode_protein_first`` and the three polymer coordinate noises are the
        training-only augmentation; inference operations never expose
        them. The returned ranks are the exact causal order used.
        """

        node_states, edge_states, neighbor_indices = self._encode(
            input,
            polymer_noise=(float(protein_noise), float(dna_noise), float(rna_noise)),
            generator=generator,
        )
        if decoding_ranks is None:
            decoding_ranks = self._training_ranks(input, decode_protein_first, generator)
        logits = self._decode_logits(
            node_states,
            edge_states,
            neighbor_indices,
            input.sequence_tokens,
            input.residue_mask,
            decoding_ranks,
        )
        return NaMpnnOutput(logits=logits, decoding_ranks=decoding_ranks)

    @staticmethod
    def _training_ranks(
        input: NaMpnnInput,
        decode_protein_first: bool,
        generator: torch.Generator | None,
    ) -> Tensor:
        r"""Draw the noised training decode order, protein-first when asked."""

        device = input.residue_mask.device
        decode_mask = (input.residue_mask & input.design_positions).to(torch.float32)
        if decode_protein_first:
            decode_mask = decode_mask * (~input.protein_mask.bool()).to(torch.float32)
        noise = torch.rand(input.residue_mask.shape, device=device, generator=generator)
        order = torch.argsort((decode_mask + 1e-4) * noise, dim=1)
        ranks = torch.empty_like(order)
        ranks.scatter_(1, order, torch.arange(order.shape[1], device=device).expand_as(order))
        return ranks

    # ------------------------------------------------------------------
    # Probability operations.
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def score(
        self,
        input: NaMpnnInput,
        *,
        num_decoding_orders: int = 1,
        decoding_ranks: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> MpnnProbabilityResult:
        r"""Return sequence-conditioned autoregressive probabilities over orders."""

        self._require_single_target(input)
        num_decoding_orders = self._validate_num_decoding_orders(num_decoding_orders)
        repeated = self._repeat_input(input, num_decoding_orders)
        node_states, edge_states, neighbor_indices = self._encode(repeated)
        if decoding_ranks is None:
            decoding_ranks = self._generate_ranks(
                repeated.residue_mask,
                repeated.design_positions,
                repeated.tied_group_ids,
                generator,
            )
        logits = self._decode_logits(
            node_states,
            edge_states,
            neighbor_indices,
            repeated.sequence_tokens,
            repeated.residue_mask,
            decoding_ranks,
        )
        return self._probability_result(input, logits, decoding_ranks)

    @torch.inference_mode()
    def unconditional_probabilities(
        self,
        input: NaMpnnInput,
        *,
        num_decoding_orders: int = 1,
    ) -> MpnnProbabilityResult:
        r"""Return sequence-free all-equal-rank probabilities for every row."""

        self._require_single_target(input)
        num_decoding_orders = self._validate_num_decoding_orders(num_decoding_orders)
        repeated = self._repeat_input(input, num_decoding_orders)
        node_states, edge_states, neighbor_indices = self._encode(repeated)
        logits = self._decode_unconditional_logits(
            node_states,
            edge_states,
            neighbor_indices,
            repeated.residue_mask,
        )
        decoding_ranks = torch.zeros(
            num_decoding_orders,
            input.residue_mask.shape[1],
            dtype=torch.long,
            device=input.atom_coordinates.device,
        )
        return self._probability_result(input, logits, decoding_ranks)

    # ------------------------------------------------------------------
    # Autoregressive sampler with the simultaneous-group bias equation.
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def sample(
        self,
        input: NaMpnnInput,
        *,
        num_samples: int = 1,
        temperature: float = 0.1,
        decoding_ranks: Tensor | None = None,
        global_logit_bias: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> NaMpnnSamplingOutput:
        r"""Sample sequences, resolving token-symmetry groups on one causal path."""

        self._require_single_target(input)
        num_samples = _positive_integer(num_samples, "num_samples")
        temperature = float(temperature)
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be a finite positive number")

        repeated = self._repeat_input(input, num_samples)
        device = repeated.atom_coordinates.device
        length = repeated.sequence_tokens.shape[1]
        vocabulary = self.vocab_size

        node_states, edge_states, neighbor_indices = self._encode(repeated)
        if decoding_ranks is None:
            ranks = self._shared_ranks(
                repeated.residue_mask,
                repeated.design_positions,
                repeated.tied_group_ids,
                generator,
            )
        else:
            ranks = decoding_ranks
        hard_support = None if repeated.allowed_token_mask is None else repeated.allowed_token_mask.bool()

        backward_mask, forward_mask = autoregressive_masks(
            ranks,
            neighbor_indices,
            repeated.residue_mask.to(dtype=node_states.dtype),
        )
        token_states = torch.zeros_like(node_states)
        encoder_edges = cat_neighbors_nodes(token_states, edge_states, neighbor_indices)
        encoder_context = cat_neighbors_nodes(node_states, encoder_edges, neighbor_indices)
        forward_context = forward_mask * encoder_context
        block_states = [node_states] + [torch.zeros_like(node_states) for _ in self.decoder_blocks]

        sequence_tokens = torch.full((num_samples, length), PAD_TOKEN, dtype=torch.long, device=device)
        log_probabilities = node_states.new_zeros(num_samples, length, vocabulary)
        sampling_probabilities = torch.zeros_like(log_probabilities)

        designed = (input.design_positions[0] & input.residue_mask[0]).tolist()
        tied_weights = repeated.tied_weights
        residue_bias = repeated.residue_logit_bias
        pair_bias = repeated.pair_logit_bias

        for members in self._ordered_units(input.tied_group_ids[0], ranks[0]):
            member_neural: dict[int, Tensor] = {}
            group_logits = node_states.new_zeros(num_samples, vocabulary)
            group_residue_bias = node_states.new_zeros(num_samples, vocabulary)
            group_pair_bias = node_states.new_zeros(num_samples, vocabulary)
            group_support = torch.ones(num_samples, vocabulary, dtype=torch.bool, device=device)
            for member in members:
                logits = self._decode_position(
                    member,
                    edge_states,
                    neighbor_indices,
                    token_states,
                    block_states,
                    backward_mask,
                    forward_context,
                    repeated.residue_mask,
                )
                member_neural[member] = logits
                weight = 1.0 if tied_weights is None else tied_weights[:, member].unsqueeze(-1)
                group_logits = group_logits + weight * logits
                if residue_bias is not None:
                    group_residue_bias = group_residue_bias + residue_bias[:, member]
                if pair_bias is not None:
                    group_pair_bias = group_pair_bias + self._pair_contribution(
                        pair_bias, member, members, sequence_tokens, vocabulary
                    )
                if hard_support is not None:
                    group_support = group_support & hard_support[:, member]

            total = group_logits + group_residue_bias + group_pair_bias
            if global_logit_bias is not None:
                total = total + global_logit_bias
            # Hard support intersects the combined logit last, before temperature and
            # the one group draw, matching the -1e8 pre-softmax masking; a
            # post-softmax intersection underflows to all-zero support when the top
            # logit is a disallowed token at low temperature.
            total = total.masked_fill(~group_support, float("-inf"))
            probabilities = F.softmax(total / temperature, dim=-1)

            generated_unit = all(designed[member] for member in members)
            if generated_unit:
                drawn = torch.multinomial(probabilities, 1, generator=generator)[:, 0]
            else:
                drawn = None
            for member in members:
                raw_log = F.log_softmax(member_neural[member], dim=-1)
                keep = float(designed[member])
                if generated_unit and drawn is not None:
                    token = drawn
                else:
                    token = repeated.sequence_tokens[:, member]
                sequence_tokens[:, member] = token
                log_probabilities[:, member] = keep * raw_log
                sampling_probabilities[:, member] = keep * probabilities
                token_states[:, member] = self.token_embedding(token)

        return NaMpnnSamplingOutput(
            sequence_tokens=sequence_tokens,
            log_probabilities=log_probabilities,
            sampling_probabilities=sampling_probabilities,
            decoding_ranks=ranks,
        )

    @staticmethod
    def _ordered_units(tied_group_ids: Tensor, ranks: Tensor) -> list[list[int]]:
        r"""Return the shared causal units as ascending-rank member lists."""

        members_by_id: dict[int, list[int]] = {}
        for index, group_id in enumerate(tied_group_ids.tolist()):
            members_by_id.setdefault(int(group_id), []).append(index)
        units = list(members_by_id.values())
        rank_values = ranks.tolist()
        units.sort(key=lambda members: rank_values[members[0]])
        return units

    def _decode_position(
        self,
        position: int,
        edge_states: Tensor,
        neighbor_indices: Tensor,
        token_states: Tensor,
        block_states: list[Tensor],
        backward_mask: Tensor,
        forward_context: Tensor,
        residue_mask: Tensor,
    ) -> Tensor:
        r"""Incrementally decode one position across the whole evaluation axis."""

        edge_slice = edge_states[:, position : position + 1]
        neighbor_slice = neighbor_indices[:, position : position + 1]
        sequence_edge = cat_neighbors_nodes(token_states, edge_slice, neighbor_slice)
        forward_slice = forward_context[:, position : position + 1]
        backward_slice = backward_mask[:, position : position + 1]
        mask_slice = residue_mask[:, position : position + 1].to(dtype=token_states.dtype)
        for block_index, block in enumerate(self.decoder_blocks):
            decoder_context = cat_neighbors_nodes(block_states[block_index], sequence_edge, neighbor_slice)
            node_slice = block_states[block_index][:, position : position + 1]
            updated = block(node_slice, backward_slice * decoder_context + forward_slice, mask_slice)
            block_states[block_index + 1][:, position : position + 1] = updated
        return self.output_projection(block_states[-1][:, position])

    @staticmethod
    def _pair_contribution(
        pair_bias: Tensor,
        member: int,
        members: list[int],
        sequence_tokens: Tensor,
        vocabulary: int,
    ) -> Tensor:
        r"""Directed adjacent-token pair contribution for one target member row.

        External context rows use the pre-rank-block sampled tokens, ``self`` stays
        at PAD, and another internal member is evaluated at the shared candidate
        token via the diagonal. Both directions of a mirrored internal pair
        therefore contribute without averaging.
        """

        samples, length = sequence_tokens.shape
        device = sequence_tokens.device
        row_bias = pair_bias[:, member]  # [S, V, L, V]
        context = sequence_tokens.clone()
        context[:, member] = PAD_TOKEN
        gathered = torch.gather(
            row_bias,
            -1,
            context[:, None, :, None].expand(samples, vocabulary, length, 1),
        ).squeeze(
            -1
        )  # [S, V, L]
        internal_others = [other for other in members if other != member]
        if internal_others:
            gathered[:, :, internal_others] = 0.0
        contribution = gathered.sum(dim=-1)
        diagonal_index = torch.arange(vocabulary, device=device)
        for other in internal_others:
            contribution = contribution + row_bias[:, diagonal_index, other, diagonal_index]
        return contribution

    # ------------------------------------------------------------------
    # Realized design and specificity operations.
    # ------------------------------------------------------------------

    @staticmethod
    def _realize_sequence(
        system: MolecularSystem,
        component_references: tuple[ComponentReference | None, ...],
        polymer_types: tuple[str, ...],
        sampled_tokens: list[int],
        design_positions: list[bool],
    ) -> MolecularSystem:
        r"""Realize one candidate while preserving entity and bond identity."""

        replacements = {
            reference: get_component(_component_from_token(int(token), polymer_type))
            for reference, polymer_type, token, selected in zip(
                component_references,
                polymer_types,
                sampled_tokens,
                design_positions,
            )
            if reference is not None and selected
        }
        return system.replace_components(replacements)

    @torch.inference_mode()
    def design(
        self,
        input: NaMpnnInput,
        *,
        num_candidates: int = 1,
        temperature: float = 0.1,
        global_logit_bias: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> DesignResult:
        r"""Design complete coordinate-free protein/DNA/RNA MolecularSystems."""

        self._require_single_target(input)
        num_candidates = _positive_integer(num_candidates, "num_candidates")
        sampled = self.sample(
            input,
            num_samples=num_candidates,
            temperature=temperature,
            global_logit_bias=global_logit_bias,
            generator=generator,
        )
        losses = nll_loss(sampled.sequence_tokens, sampled.log_probabilities)
        designed = (
            (input.residue_mask & input.design_positions).repeat_interleave(num_candidates, dim=0).to(losses.dtype)
        )
        design_count = designed.sum(dim=-1)
        confidence = torch.exp(-(losses * designed).sum(dim=-1) / design_count.clamp_min(1.0))
        matches = (sampled.sequence_tokens == input.sequence_tokens.repeat_interleave(num_candidates, dim=0)).to(
            losses.dtype
        )
        recovery = (matches * designed).sum(dim=-1) / design_count.clamp_min(1.0)

        references = input.component_references[0]
        polymer_types = input.polymer_types[0]
        design_rows = (input.design_positions[0] & input.residue_mask[0]).tolist()
        designed_scope = frozenset(
            reference for reference, selected in zip(references, design_rows) if reference is not None and selected
        )
        empty_scope = len(designed_scope) == 0

        host_sequence_tokens = sampled.sequence_tokens.detach().cpu().tolist()
        host_confidence = confidence.detach().cpu().tolist()
        host_recovery = recovery.detach().cpu().tolist()
        candidates: list[DesignCandidate] = []
        for index in range(num_candidates):
            system = self._realize_sequence(
                input.structures[0].system,
                references,
                polymer_types,
                host_sequence_tokens[index],
                design_rows,
            )
            candidates.append(
                DesignCandidate(
                    system=system,
                    scores={
                        "designed_sequence_confidence": Score(
                            float("nan") if empty_scope else float(host_confidence[index]),
                            preferred="higher",
                            scope=designed_scope,
                        ),
                        "sequence_recovery": Score(
                            float("nan") if empty_scope else float(host_recovery[index]),
                            preferred="higher",
                            scope=designed_scope,
                        ),
                    },
                )
            )
        return DesignResult(candidates=tuple(candidates))

    @torch.inference_mode()
    def specificity_profile(
        self,
        input: NaMpnnInput,
        *,
        num_samples: int = 30,
        temperature: float = 0.6,
        global_logit_bias: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> ScoreResult:
        r"""Average nucleic-acid sampling probabilities with protein context fixed."""

        self._require_single_target(input)
        na_design = input.design_positions & ~input.protein_mask
        fixed = NaMpnnInput(
            atom_coordinates=input.atom_coordinates,
            atom_mask=input.atom_mask,
            sequence_tokens=input.sequence_tokens,
            residue_mask=input.residue_mask,
            residue_index=input.residue_index,
            chain_index=input.chain_index,
            polymer_type_index=input.polymer_type_index,
            protein_mask=input.protein_mask,
            dna_mask=input.dna_mask,
            rna_mask=input.rna_mask,
            design_positions=na_design,
            tied_group_ids=input.tied_group_ids,
            structures=input.structures,
            component_references=input.component_references,
            polymer_types=input.polymer_types,
            conditions=input.conditions,
            tied_weights=input.tied_weights,
            residue_logit_bias=input.residue_logit_bias,
            allowed_token_mask=input.allowed_token_mask,
            pair_logit_bias=input.pair_logit_bias,
        )
        sampled = self.sample(
            fixed,
            num_samples=num_samples,
            temperature=temperature,
            global_logit_bias=global_logit_bias,
            generator=generator,
        )
        mean_probabilities = sampled.sampling_probabilities.to(torch.float64).mean(dim=0).cpu().numpy()

        references = input.component_references[0]
        polymer_types = input.polymer_types[0]
        selected = (na_design[0] & input.residue_mask[0]).tolist()
        na_rows = [
            (index, reference)
            for index, (reference, polymer, keep) in enumerate(zip(references, polymer_types, selected))
            if reference is not None and keep and polymer in ("dna", "rna")
        ]
        axis = tuple(reference for _index, reference in na_rows)
        token_columns = {
            row: (
                DNA_GENERATABLE_TOKEN_IDS
                if polymer_types[row] == "dna"
                else (
                    RNA_SHARED_GENERATABLE_TOKEN_IDS
                    if int(input.sequence_tokens[0, row]) in _RNA_SHARED_INPUT_TOKENS
                    else RNA_SEPARATE_GENERATABLE_TOKEN_IDS
                )
            )
            for row, _reference in na_rows
        }
        base_names = ("adenine", "cytosine", "guanine", "thymine_uracil")
        metrics: dict[str, Metric] = {}
        for base_index, base in enumerate(base_names):
            values = np.array(
                [
                    mean_probabilities[
                        row,
                        token_columns[row][base_index],
                    ]
                    for row, _reference in na_rows
                ],
                dtype=np.float64,
            )
            metrics[f"{base}_probability"] = Metric(values=values, axis=axis)
        return ScoreResult(target=input.structures[0], scores={}, metrics=metrics)


def _positive_integer(value: int, name: str) -> int:
    result = operator.index(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


class NaMpnnFeatures(nn.Module):
    r"""Geometric feature encoder for unified protein/nucleic-acid polymer graphs."""

    def __init__(self, config: NaMpnnConfig) -> None:
        super().__init__()
        self.num_neighbors = config.get("k_neighbors", config.num_neighbors)
        self.num_rbf = config.num_rbf
        self.num_polymer_types = config.num_polymer_types
        self.include_predicted_na_n = config.include_predicted_na_n
        self.nitrogen_index = MIXED_POLYMER_ATOM_TO_INDEX["N"]
        self.alpha_carbon_index = MIXED_POLYMER_ATOM_TO_INDEX["CA"]
        self.carbonyl_carbon_index = MIXED_POLYMER_ATOM_TO_INDEX["C"]
        self.oxygen_4_prime_index = MIXED_POLYMER_ATOM_TO_INDEX["O4'"]
        self.carbon_1_prime_index = MIXED_POLYMER_ATOM_TO_INDEX["C1'"]
        self.carbon_2_prime_index = MIXED_POLYMER_ATOM_TO_INDEX["C2'"]
        total_atoms = config.num_atom_types + 1 + int(self.include_predicted_na_n)
        self.relative_position = MpnnRelativePositionEncoding(config.num_positional_embeddings)
        self.node_projection = nn.Linear(config.num_polymer_types, config.feature_size, bias=False)
        self.node_norm = LayerNorm(config.feature_size)
        self.edge_projection = nn.Linear(
            config.num_positional_embeddings + config.num_rbf * total_atoms * total_atoms,
            config.feature_size,
            bias=False,
        )
        self.edge_norm = LayerNorm(config.feature_size)

    def forward(
        self,
        atom_coordinates: Tensor,
        atom_mask: Tensor,
        residue_mask: Tensor,
        residue_index: Tensor,
        chain_index: Tensor,
        polymer_type_index: Tensor,
        protein_mask: Tensor,
        dna_mask: Tensor,
        rna_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        r"""Build node features, edge features, and nearest-neighbor indices."""

        dtype = atom_coordinates.dtype
        atom_mask = atom_mask.to(dtype=dtype)
        valid = residue_mask.to(dtype=dtype)
        protein = protein_mask.to(dtype=dtype)
        dna = dna_mask.to(dtype=dtype)
        rna = rna_mask.to(dtype=dtype)

        alpha_carbon = atom_coordinates[:, :, self.alpha_carbon_index]
        nitrogen = atom_coordinates[:, :, self.nitrogen_index]
        carbonyl_carbon = atom_coordinates[:, :, self.carbonyl_carbon_index]
        beta_carbon = virtual_beta_carbon(nitrogen, alpha_carbon, carbonyl_carbon)
        reference = atom_coordinates[:, :, self.carbon_1_prime_index]
        augmented_coordinates = [atom_coordinates, beta_carbon[:, :, None, :]]
        augmented_mask = [atom_mask, protein[:, :, None]]
        if self.include_predicted_na_n:
            oxygen_4_prime = atom_coordinates[:, :, self.oxygen_4_prime_index]
            carbon_1_prime = atom_coordinates[:, :, self.carbon_1_prime_index]
            carbon_2_prime = atom_coordinates[:, :, self.carbon_2_prime_index]
            predicted_nitrogen = _virtual_na_n(
                oxygen_4_prime,
                carbon_1_prime,
                carbon_2_prime,
            )
            augmented_coordinates.append(predicted_nitrogen[:, :, None, :])
            augmented_mask.append((dna + rna).clamp(max=1.0)[:, :, None])
        all_coordinates = torch.cat(augmented_coordinates, dim=-2)
        all_mask = torch.cat(augmented_mask, dim=-1)

        neighbor_indices = nearest_neighbors(
            alpha_carbon + reference,
            valid,
            self.num_neighbors,
        )[1]
        pair_rbf = _all_pair_rbf(all_coordinates, all_mask, neighbor_indices, self.num_rbf)
        residue_offset = residue_index[:, :, None] - residue_index[:, None, :]
        residue_offset = gather_edges(residue_offset[:, :, :, None], neighbor_indices)[:, :, :, 0]
        same_chain = (chain_index[:, :, None] == chain_index[:, None, :]).long()
        same_chain = gather_edges(same_chain[:, :, :, None], neighbor_indices)[:, :, :, 0]
        position_features = self.relative_position(residue_offset.long(), same_chain)
        edge_features = self.edge_norm(self.edge_projection(torch.cat((position_features, pair_rbf), dim=-1)))

        polymer_one_hot = F.one_hot(
            polymer_type_index,
            num_classes=self.num_polymer_types,
        ).to(dtype=dtype)
        node_features = self.node_norm(self.node_projection(polymer_one_hot))
        return node_features, edge_features, neighbor_indices


def _virtual_na_n(o4prime: Tensor, c1prime: Tensor, c2prime: Tensor) -> Tensor:
    r"""Reconstruct the idealized nucleic-acid base nitrogen coordinate."""

    b = c1prime - o4prime
    c = c2prime - c1prime
    a = torch.cross(b, c, dim=-1)
    return -0.56967352 * a + 0.51055973 * b - 0.53122153 * c + c1prime


def _all_pair_rbf(coordinates: Tensor, atom_mask: Tensor, neighbor_indices: Tensor, num_bins: int) -> Tensor:
    r"""RBF-encode every ordered atom-pair distance for each neighbor edge."""

    flat = coordinates.reshape(coordinates.shape[0], coordinates.shape[1], -1)
    neighbor_flat = gather_nodes(flat, neighbor_indices)
    neighbor_coordinates = neighbor_flat.reshape(
        *neighbor_flat.shape[:-1], coordinates.shape[-2], coordinates.shape[-1]
    )
    differences = coordinates[:, :, None, :, None, :] - neighbor_coordinates[:, :, :, None]
    distances = torch.sqrt(torch.sum(differences**2, -1) + 1e-6)
    rbf = radial_basis(distances, num_bins)
    neighbor_mask = gather_nodes(atom_mask, neighbor_indices)
    rbf = rbf * atom_mask[:, :, None, :, None, None] * neighbor_mask[:, :, :, None, :, None]
    return rbf.reshape(coordinates.shape[0], coordinates.shape[1], neighbor_indices.shape[-1], -1)


def _component_from_token(token: int, polymer_type: str) -> str:
    r"""Realize one sampled token as a canonical component under its polymer type."""

    if polymer_type == "protein":
        if token >= 20:
            raise RuntimeError(f"NA-MPNN sampled non-protein token {token} for a protein residue")
        return MIXED_POLYMER_COMPONENT_IDS[token]
    if polymer_type == "dna":
        if not 21 <= token <= 24:
            raise RuntimeError(f"NA-MPNN sampled non-DNA token {token} for a DNA residue")
        return ("DA", "DC", "DG", "DT")[token - 21]
    if 21 <= token <= 24:
        return ("A", "C", "G", "U")[token - 21]
    if 26 <= token <= 29:
        return ("A", "C", "G", "U")[token - 26]
    raise RuntimeError(f"NA-MPNN sampled non-RNA token {token} for an RNA residue")


# Differentiable training objectives published by NA-MPNN.
#
# The NA-MPNN training surface owns two ordinary per-token losses over the
# 33-class protein/DNA/RNA vocabulary. Both consume ``[B, L]`` sequence tokens and
# ``[B, L, 33]`` vocabulary log probabilities and return the unreduced per-row loss
# with shape ``sequence_tokens.shape``. The polymer-specific label smoothing
# distributes its mass only over each row's own polymer residue-type set. Masked mean,
# accuracy, PPM injection, and the training command's fixed token aggregate belong
# to a later Runner and are deliberately excluded here.

# Contiguous vocabulary spans of the 33-token alphabet. Protein occupies
# tokens 0..20 (twenty amino acids plus UNK), DNA tokens 21..25 (DA/DC/DG/DT/DX),
# and RNA tokens 26..30 (A/C/G/U/RX); MAS and PAD are never smoothing targets.
_PROTEIN_RESIDUE_TYPE_SPAN = (0, 21)
_DNA_RESIDUE_TYPE_SPAN = (21, 26)
_RNA_RESIDUE_TYPE_SPAN = (26, 31)
_POLYMER_PROTEIN = 0
_POLYMER_DNA = 1
_POLYMER_RNA = 2


def smoothed_loss(
    sequence_tokens: Tensor,
    log_probabilities: Tensor,
    polymer_type_index: Tensor,
    *,
    weight: float = 0.1,
) -> Tensor:
    r"""Return the per-token polymer-specific label-smoothed cross entropy.

    Each row's smoothing mass ``weight`` is distributed uniformly over that row's
    own polymer residue-type set (protein 21, DNA 5, RNA 5), and the true column is
    scaled by ``1 - weight`` before the eps is added, matching NA-MPNN's
    ``loss_smoothed``. At ``weight=0`` this reduces to [`nll_loss`][]. The
    result has shape ``sequence_tokens.shape``.
    """

    num_classes = log_probabilities.shape[-1]
    device = log_probabilities.device
    dtype = log_probabilities.dtype
    target = F.one_hot(sequence_tokens, num_classes).to(dtype)

    def span_mask(span: tuple[int, int]) -> Tensor:
        mask = torch.zeros(num_classes, device=device, dtype=dtype)
        mask[span[0] : span[1]] = 1.0
        return mask

    protein_set = span_mask(_PROTEIN_RESIDUE_TYPE_SPAN)
    dna_set = span_mask(_DNA_RESIDUE_TYPE_SPAN)
    rna_set = span_mask(_RNA_RESIDUE_TYPE_SPAN)
    residue_type_mask = protein_set + dna_set + rna_set

    target = target * torch.where(
        residue_type_mask.bool(),
        1.0 - weight,
        torch.ones((), device=device, dtype=dtype),
    )

    protein_rows = (polymer_type_index == _POLYMER_PROTEIN).to(dtype).unsqueeze(-1)
    dna_rows = (polymer_type_index == _POLYMER_DNA).to(dtype).unsqueeze(-1)
    rna_rows = (polymer_type_index == _POLYMER_RNA).to(dtype).unsqueeze(-1)
    eps = (
        protein_rows * protein_set * (weight / (_PROTEIN_RESIDUE_TYPE_SPAN[1] - _PROTEIN_RESIDUE_TYPE_SPAN[0]))
        + dna_rows * dna_set * (weight / (_DNA_RESIDUE_TYPE_SPAN[1] - _DNA_RESIDUE_TYPE_SPAN[0]))
        + rna_rows * rna_set * (weight / (_RNA_RESIDUE_TYPE_SPAN[1] - _RNA_RESIDUE_TYPE_SPAN[0]))
    )
    target = target + eps
    return -(target * log_probabilities).sum(dim=-1)


# ---------------------------------------------------------------------------
# NA-MPNN typed inputs and outputs.
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class NaMpnnInput:
    r"""Aligned tensors and canonical identities consumed by NA-MPNN operations.

    One biological Processor call owns one ``Structure``; the leading tensor axis
    is an execution-evaluation axis, not a scientific batch. ``tied_group_ids``
    is the dense causal-unit contract: exact ties, token symmetry groups,
    same-entity shared instances, and bonded fixing all lower into it so the graph
    hot path never reasons about nested Python groups. Aligned residue bias and
    tied weights are ordinary zero/one tensors; the dominant ``[B, L, V, L, V]``
    pair tensor is ``None`` when absent.
    """

    atom_coordinates: Tensor
    atom_mask: Tensor
    sequence_tokens: Tensor
    residue_mask: Tensor
    residue_index: Tensor
    chain_index: Tensor
    polymer_type_index: Tensor
    protein_mask: Tensor
    dna_mask: Tensor
    rna_mask: Tensor
    design_positions: Tensor
    tied_group_ids: Tensor
    structures: tuple[Structure, ...]
    component_references: tuple[tuple[ComponentReference | None, ...], ...]
    polymer_types: tuple[tuple[str, ...], ...]
    conditions: tuple[tuple[AllowedResidueIdentities | TiedResidueIdentities, ...], ...]
    tied_weights: Tensor | None = None
    residue_logit_bias: Tensor | None = None
    allowed_token_mask: Tensor | None = None
    pair_logit_bias: Tensor | None = None


@dataclass(eq=False)
class NaMpnnOutput:
    r"""Raw logits and exact causal ranks from an NA-MPNN sequence graph."""

    logits: Tensor
    decoding_ranks: Tensor

    @property
    def log_probabilities(self) -> Tensor:
        r"""Return vocabulary log probabilities without storing a duplicate tensor."""

        return self.logits.log_softmax(dim=-1)

    @property
    def probabilities(self) -> Tensor:
        r"""Return vocabulary probabilities without storing a duplicate tensor."""

        return self.logits.softmax(dim=-1)


@dataclass(eq=False)
class NaMpnnSamplingOutput:
    r"""Complete aligned tensor output from the NA-MPNN sequence sampler."""

    sequence_tokens: Tensor
    log_probabilities: Tensor
    sampling_probabilities: Tensor
    decoding_ranks: Tensor
