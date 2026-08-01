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
NA-MPNN inverse-folding model for protein, DNA, RNA, and mixed-polymer structures.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch
import torch.nn.functional as F
from tensordict import TensorClass, tensorclass
from torch import Tensor, nn

from ...data.tokenization import TOKEN_INDEX, TOKENS
from ...errors import RequestRef, fail
from ...model import Model, TensorCollection
from ...modules.backend import LayerNorm, gather_nodes, masked_knn, resolve_config_backend
from ...modules.functional import radial_basis
from ...modules.geometry import neighbor_atom_distances, virtual_beta_carbon
from ...modules.mpnn import (
    MpnnDecoderBlock,
    MpnnEncoderBlock,
    MpnnRelativePositionEncoding,
    autoregressive_masks,
    cat_neighbors_nodes,
    decode_sequence_free,
    validate_decoding_ranks,
)
from ..processing import project_model_categories, project_model_coordinates
from .config import NaMpnnConfig

#: Protein, DNA, and RNA atom channels used by NA-MPNN input features.
NA_MPNN_ATOM_NAMES = (
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
NA_MPNN_ATOM_INDEX = {name: index for index, name in enumerate(NA_MPNN_ATOM_NAMES)}
#: The classes the released checkpoints were trained on, in their upstream order (canonical spelling): the protein
#: residues and ``X``, DNA and ``DNA_X``, RNA and ``RNA_X``, mask and padding. Every other canonical class has logit
#: ``-inf``, so identical inputs give the upstream distribution.
NAMPNN_VOCABULARY: tuple[str, ...] = (
    *"ARNDCQEGHILKMFPSTWYVX",
    *(f"DNA_{base}" for base in "ACGTX"),
    *(f"RNA_{base}" for base in "ACGUX"),
    "<mask>",
    "<pad>",
)
_PAD = TOKEN_INDEX["<pad>"]


def _invalid(argument: str, value: object) -> None:
    fail("value_invalid", RequestRef(argument, str(value)))


def normalize_nampnn_input(input: NaMpnnInput, *, dtype: torch.dtype | None = None) -> NaMpnnInput:
    """Return finite available model coordinates without mutating the supplied input."""
    result = cast(NaMpnnInput, cast(TensorClass, input).clone(recurse=False))
    observed = input.atom_mask & input.residue_mask.unsqueeze(-1)
    result.atom_coordinates = project_model_coordinates(
        input.atom_coordinates, observed, dtype=dtype, feature="mixed-polymer backbone"
    )
    return result


class NaMpnnModel(Model):
    r"""
    Design protein and nucleic-acid sequences with NA-MPNN.

    Accepts [`NaMpnnInput`][ofoldx.models.nampnn.model.NaMpnnInput] and returns
    [`NaMpnnOutput`][ofoldx.models.nampnn.model.NaMpnnOutput].

    Args:
        config: Model architecture and inference configuration.

    Notes:
        Inputs and outputs use the 72 canonical classes. The released vocabulary contains 33 trained columns;
        classes outside those columns have ``-inf`` output logits. The paired Processor reads an input class
        outside the released vocabulary as its polymer type's unknown class. Shared nucleic tokens map RNA
        bases to DNA output classes, and decoded RNA residues retain their RNA chemistry.
    """

    config_class = NaMpnnConfig
    config: NaMpnnConfig
    _gradient_checkpointing_func: Callable[..., object]

    def __init__(self, config: NaMpnnConfig | None = None) -> None:
        config = NaMpnnConfig() if config is None else config
        super().__init__(config)
        self.gradient_checkpointing = False
        self.feature_encoder = NaMpnnFeatureEncoder(self.config)
        self.node_embedding = nn.Linear(self.config.feature_size, self.config.hidden_size)
        self.edge_embedding = nn.Linear(self.config.feature_size, self.config.hidden_size)
        self.encoder_blocks = nn.ModuleList(
            MpnnEncoderBlock(self.config.hidden_size, self.config.hidden_size * 2, dropout=self.config.dropout)
            for _ in range(self.config.num_encoder_blocks)
        )
        self.token_embedding = nn.Embedding(len(TOKENS), self.config.hidden_size)
        self.decoder_blocks = nn.ModuleList(
            MpnnDecoderBlock(self.config.hidden_size, self.config.hidden_size * 3, dropout=self.config.dropout)
            for _ in range(self.config.num_decoder_blocks)
        )
        self.output_projection = nn.Linear(self.config.hidden_size, len(TOKENS))
        trained = torch.zeros(len(TOKENS), dtype=torch.bool)
        trained[[TOKEN_INDEX[token] for token in NAMPNN_VOCABULARY]] = True
        self.register_buffer("trained_classes", trained, persistent=False)
        self.post_init()

    def _logits(self, states: Tensor) -> Tensor:
        r"""Vocabulary logits over the canonical classes; classes outside the trained vocabulary are ``-inf``."""
        return self.output_projection(states).masked_fill(~self.trained_classes, -torch.inf)

    def initialize_parameters(self) -> None:
        r"""
        Apply the Xavier-uniform matrix initialization.
        """

        for parameter in self.parameters():
            if parameter.ndim > 1:
                nn.init.xavier_uniform_(parameter)

    def encode(
        self,
        input: NaMpnnInput,
        *,
        num_neighbors: int | None = None,
        protein_coordinate_noise: float = 0.0,
        dna_coordinate_noise: float = 0.0,
        rna_coordinate_noise: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        r"""
        Encode a batch of mixed-polymer backbones as residue graph states.

        Args:
            input: NA-MPNN tensors with batch and residue axes.
            num_neighbors: Optional nearest-neighbor count for this operation.
            protein_coordinate_noise: Protein-coordinate noise standard deviation.
            dna_coordinate_noise: DNA-coordinate noise standard deviation.
            rna_coordinate_noise: RNA-coordinate noise standard deviation.
            generator: Random generator used for coordinate noise.

        Returns:
            node_states: Node states with shape ``(B, L, H)``.
            edge_states: Edge states with shape ``(B, L, K, H)``.
            neighbor_indices: Neighbor indices with shape ``(B, L, K)``.
            attend_mask: Real-neighbor mask with shape ``(B, L, K)``.
        """

        input = normalize_nampnn_input(input, dtype=self.node_embedding.weight.dtype)
        coordinates = input.atom_coordinates
        protein_mask = input.polymer_type_index == 0
        dna_mask = input.polymer_type_index == 1
        rna_mask = input.polymer_type_index == 2
        if protein_coordinate_noise or dna_coordinate_noise or rna_coordinate_noise:
            scale = (
                protein_mask.to(coordinates.dtype) * protein_coordinate_noise
                + dna_mask.to(coordinates.dtype) * dna_coordinate_noise
                + rna_mask.to(coordinates.dtype) * rna_coordinate_noise
            )
            noise = torch.empty_like(coordinates).normal_(generator=generator)
            coordinates = coordinates + scale[:, :, None, None] * input.atom_mask[..., None] * noise
        # ``attend_mask`` comes from the k-NN: the real-neighbor mask, which also zeroes filler slots when a
        # structure is shorter than ``num_neighbors`` (``valid`` there is ``residue_mask``, matching ``mask``).
        node_features, edge_features, neighbor_indices, attend_mask = self.feature_encoder(
            coordinates,
            input.atom_mask,
            input.residue_mask,
            input.residue_index,
            input.chain_index,
            input.polymer_type_index,
            num_neighbors=num_neighbors,
        )
        node_states = self.node_embedding(node_features)
        edge_states = self.edge_embedding(edge_features)
        mask = input.residue_mask.to(dtype=node_states.dtype)
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
        return node_states, edge_states, neighbor_indices, attend_mask

    def forward(
        self,
        input: NaMpnnInput,
        *,
        decoding_ranks: Tensor | None = None,
        decode_protein_first: bool = False,
        protein_coordinate_noise: float = 0.0,
        dna_coordinate_noise: float = 0.0,
        rna_coordinate_noise: float = 0.0,
        num_neighbors: int | None = None,
        generator: torch.Generator | None = None,
    ) -> NaMpnnOutput:
        r"""
        Predict residue logits conditioned on the input sequence and decoding order.

        Args:
            input: NA-MPNN tensors with batch size ``B`` and ``L`` residue rows.
            decoding_ranks: Optional ranks with shape ``(B, L)``; smaller values
                decode earlier.
            decode_protein_first: Whether generated protein rows precede generated
                nucleic-acid rows when ranks are drawn.
            protein_coordinate_noise: Gaussian coordinate-noise standard deviation for proteins.
            dna_coordinate_noise: Gaussian coordinate-noise standard deviation for DNA.
            rna_coordinate_noise: Gaussian coordinate-noise standard deviation for RNA.
            num_neighbors: Optional nearest-neighbor count for this operation.
            generator: Random generator used for the order and coordinate noise.

        Returns:
            Vocabulary logits and the decoding ranks used for the forward pass.
        """

        if decoding_ranks is not None:
            validate_decoding_ranks(decoding_ranks, input.tied_group_ids)
        node_states, edge_states, neighbor_indices, neighbor_valid = self.encode(
            input,
            num_neighbors=num_neighbors,
            protein_coordinate_noise=protein_coordinate_noise,
            dna_coordinate_noise=dna_coordinate_noise,
            rna_coordinate_noise=rna_coordinate_noise,
            generator=generator,
        )
        if decoding_ranks is None:
            protein_mask = input.polymer_type_index == 0 if decode_protein_first else None
            decoding_ranks = self._generate_ranks(
                input.residue_mask,
                input.design_mask,
                input.tied_group_ids,
                generator,
                protein_mask=protein_mask,
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
        return NaMpnnOutput(
            logits=logits,
            decoding_ranks=decoding_ranks,
            batch_size=[logits.shape[0]],
        )

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
        num_neighbors: int | None = None,
        generator: torch.Generator | None = None,
    ) -> NaMpnnOutput:
        r"""
        Score the observed sequence over one or more decoding orders.

        Args:
            input: One processed mixed-polymer target.
            num_decoding_orders: Number of decoding orders to evaluate.
            decoding_ranks: Optional ranks with shape
                ``(num_decoding_orders, num_residues)``.
            num_neighbors: Optional nearest-neighbor count for this operation.
            generator: Random generator used to draw decoding orders.

        Returns:
            Raw vocabulary logits and the decoding ranks used for each evaluation.
        """

        return self.autoregressive_probabilities(
            input,
            num_decoding_orders=num_decoding_orders,
            decoding_ranks=decoding_ranks,
            num_neighbors=num_neighbors,
            generator=generator,
        )

    def autoregressive_probabilities(
        self,
        input: NaMpnnInput,
        *,
        num_decoding_orders: int = 1,
        decoding_ranks: Tensor | None = None,
        num_neighbors: int | None = None,
        generator: torch.Generator | None = None,
    ) -> NaMpnnOutput:
        r"""
        Evaluate sequence-conditioned probabilities over decoding orders.

        Args:
            input: One processed mixed-polymer target.
            num_decoding_orders: Number of decoding orders to evaluate.
            decoding_ranks: Optional ranks with shape
                ``(num_decoding_orders, num_residues)``.
            num_neighbors: Optional nearest-neighbor count for this operation.
            generator: Random generator used to draw decoding orders.

        Returns:
            Logits with shape ``(num_decoding_orders, num_residues, vocab_size)``
                and decoding ranks with shape ``(num_decoding_orders, num_residues)``.
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
            input,
            num_neighbors=num_neighbors,
        )
        node_states = node_states.repeat_interleave(num_decoding_orders, dim=0)
        edge_states = edge_states.repeat_interleave(num_decoding_orders, dim=0)
        neighbor_indices = neighbor_indices.repeat_interleave(num_decoding_orders, dim=0)
        neighbor_valid = neighbor_valid.repeat_interleave(num_decoding_orders, dim=0)
        logits = self._decode_logits(
            node_states,
            edge_states,
            neighbor_indices,
            neighbor_valid,
            repeated.sequence_tokens,
            repeated.residue_mask,
            decoding_ranks,
        )
        return NaMpnnOutput(
            logits=logits,
            decoding_ranks=decoding_ranks,
            batch_size=[logits.shape[0]],
        )

    def unconditional_probabilities(
        self,
        input: NaMpnnInput,
        *,
        num_decoding_orders: int = 1,
        num_neighbors: int | None = None,
    ) -> NaMpnnOutput:
        r"""
        Evaluate sequence-free probabilities without decoded neighbors.

        Args:
            input: One processed mixed-polymer target.
            num_decoding_orders: Number of identical evaluation rows to return.
            num_neighbors: Optional nearest-neighbor count for this operation.

        Returns:
            Logits with shape ``(num_decoding_orders, num_residues, vocab_size)``
                and all-zero ranks with shape ``(num_decoding_orders, num_residues)``.
        """

        self._require_single_target(input)
        if num_decoding_orders <= 0:
            _invalid("num_decoding_orders", num_decoding_orders)
        node_states, edge_states, neighbor_indices, neighbor_valid = self.encode(
            input,
            num_neighbors=num_neighbors,
        )
        repeated = self._repeat_input(input, num_decoding_orders)
        node_states = node_states.repeat_interleave(num_decoding_orders, dim=0)
        edge_states = edge_states.repeat_interleave(num_decoding_orders, dim=0)
        neighbor_indices = neighbor_indices.repeat_interleave(num_decoding_orders, dim=0)
        neighbor_valid = neighbor_valid.repeat_interleave(num_decoding_orders, dim=0)
        decoding_ranks = torch.zeros(
            num_decoding_orders,
            input.residue_mask.shape[1],
            dtype=torch.long,
            device=input.atom_coordinates.device,
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
        return NaMpnnOutput(
            logits=logits,
            decoding_ranks=decoding_ranks,
            batch_size=[logits.shape[0]],
        )

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
        num_neighbors: int | None = None,
        generator: torch.Generator | None = None,
    ) -> NaMpnnSamplingOutput:
        r"""
        Sample mixed-polymer sequences under the processed constraints.

        Args:
            input: One processed mixed-polymer target.
            num_samples: Number of sequences to sample.
            temperature: Positive sampling temperature.
            decoding_ranks: Optional independent ranks with shape
                ``(num_samples, num_residues)``.
            global_logit_bias: Optional bias with shape ``(vocab_size,)``.
            num_neighbors: Optional nearest-neighbor count for this operation.
            generator: Random generator used for orders and token sampling.

        Returns:
            Sampled token IDs, model log probabilities, sampling probabilities,
                and decoding ranks, each with a leading ``num_samples`` axis.
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
        device = repeated.atom_coordinates.device
        num_residues = repeated.sequence_tokens.shape[1]
        vocab_size = len(TOKENS)

        if decoding_ranks is None:
            ranks = self._generate_ranks(
                repeated.residue_mask,
                repeated.design_mask,
                repeated.tied_group_ids,
                generator,
            )
        else:
            ranks = decoding_ranks
            validate_decoding_ranks(ranks, repeated.tied_group_ids)
        node_states, edge_states, neighbor_indices, neighbor_valid = self.encode(
            input,
            num_neighbors=num_neighbors,
        )
        node_states = node_states.repeat_interleave(num_samples, dim=0)
        edge_states = edge_states.repeat_interleave(num_samples, dim=0)
        neighbor_indices = neighbor_indices.repeat_interleave(num_samples, dim=0)
        neighbor_valid = neighbor_valid.repeat_interleave(num_samples, dim=0)
        hard_support = repeated.allowed_token_mask.bool()

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

        sequence_tokens = torch.full((num_samples, num_residues), _PAD, dtype=torch.long, device=device)
        log_probabilities = node_states.new_zeros(num_samples, num_residues, vocab_size)
        sampling_probabilities = torch.zeros_like(log_probabilities)

        designed = repeated.design_mask & repeated.residue_mask
        tied_weights = repeated.tied_weights
        residue_bias = repeated.residue_logit_bias
        pair_logit_bias = repeated.pair_logit_bias

        position_index = torch.arange(num_residues, device=device)
        group_ids = input.tied_group_ids[0]
        first_by_group = torch.full((num_residues,), num_residues, dtype=torch.long, device=device)
        first_by_group.scatter_reduce_(0, group_ids, position_index, reduce="amin", include_self=True)
        representative_positions = position_index[position_index == first_by_group[group_ids]]
        unit_ids = group_ids[representative_positions]
        unit_member_mask = unit_ids[:, None] == group_ids[None, :]
        unit_members = (
            torch.where(
                unit_member_mask,
                position_index[None, :],
                num_residues,
            )
            .sort(dim=1)
            .values
        )
        max_members = int(unit_member_mask.sum(dim=1).max().item())
        unit_order = torch.argsort(ranks[:, representative_positions], dim=1)

        for step in range(representative_positions.numel()):
            active_unit = unit_order[:, step]
            group_member_mask = unit_member_mask[active_unit]
            active_members = unit_members[active_unit, :max_members]
            member_records: list[tuple[Tensor, Tensor, Tensor, Tensor]] = []
            group_logits = node_states.new_zeros(num_samples, vocab_size)
            group_residue_bias = node_states.new_zeros(num_samples, vocab_size)
            group_pair_bias = node_states.new_zeros(num_samples, vocab_size)
            group_support = torch.ones(num_samples, vocab_size, dtype=torch.bool, device=device)
            generated_unit = torch.ones(num_samples, dtype=torch.bool, device=device)
            group_has_designed = torch.zeros(num_samples, dtype=torch.bool, device=device)
            anchor_found = torch.zeros(num_samples, dtype=torch.bool, device=device)
            anchor = torch.full(
                (num_samples,),
                _PAD,
                dtype=torch.long,
                device=device,
            )
            for member_slot in range(max_members):
                member = active_members[:, member_slot]
                active = member < num_residues
                member = member.clamp_max(num_residues - 1)
                logits = self._decode_positions(
                    member,
                    active,
                    edge_states,
                    neighbor_indices,
                    token_states,
                    block_states,
                    backward_mask,
                    forward_context,
                    repeated.residue_mask,
                    neighbor_valid,
                )
                member_index = member[:, None]
                active_float = active[:, None].to(dtype=node_states.dtype)
                weight = torch.gather(tied_weights, 1, member_index).squeeze(1)
                member_bias = torch.gather(
                    residue_bias,
                    1,
                    member_index[:, :, None].expand(-1, 1, vocab_size),
                ).squeeze(1)
                member_support = torch.gather(
                    hard_support,
                    1,
                    member_index[:, :, None].expand(-1, 1, vocab_size),
                ).squeeze(1)
                member_designed = torch.gather(designed, 1, member_index).squeeze(1)
                member_token = torch.gather(repeated.sequence_tokens, 1, member_index).squeeze(1)
                group_logits = group_logits + active_float * weight[:, None] * logits
                group_residue_bias = group_residue_bias + active_float * member_bias
                if pair_logit_bias is not None:
                    group_pair_bias = group_pair_bias + active_float * self._pair_contribution(
                        pair_logit_bias,
                        member,
                        group_member_mask,
                        sequence_tokens,
                        vocab_size,
                    )
                group_support = group_support & (~active[:, None] | member_support)
                generated_unit = generated_unit & (~active | member_designed)
                group_has_designed = group_has_designed | (active & member_designed)
                fixed = active & ~member_designed
                anchor = torch.where(fixed & ~anchor_found, member_token, anchor)
                anchor_found = anchor_found | fixed
                member_records.append((member, active, member_designed, logits))

            total = group_logits + group_residue_bias + group_pair_bias
            if global_logit_bias is not None:
                total = total + global_logit_bias
            # Hard support intersects the combined logit last, before temperature and
            # the one group draw, matching the -1e8 pre-softmax masking; a
            # post-softmax intersection underflows to all-zero support when the top
            # logit is a disallowed token at low temperature.
            total = total.masked_fill(~(group_support & self.trained_classes), float("-inf"))
            probabilities = F.softmax(total / temperature, dim=-1)

            mixed_unit = group_has_designed & ~generated_unit
            anchor_supported = torch.gather(group_support, 1, anchor[:, None]).squeeze(1)
            torch._assert(
                torch.all(~mixed_unit | anchor_supported),
                "a mixed fixed/design causal unit anchor must belong to its common token support",
            )
            drawn = torch.multinomial(probabilities, 1, generator=generator)[:, 0]
            group_token = torch.where(generated_unit, drawn, anchor)
            embedded_token = self.token_embedding(group_token)
            for member, active, member_designed, logits in member_records:
                member_index = member[:, None]
                current_token = torch.gather(sequence_tokens, 1, member_index).squeeze(1)
                written_token = torch.where(active, group_token, current_token)
                sequence_tokens.scatter_(1, member_index, written_token[:, None])

                output_index = member_index[:, :, None].expand(-1, 1, vocab_size)
                raw_log = F.log_softmax(logits.masked_fill(~self.trained_classes, -torch.inf), dim=-1)
                member_log = torch.where(member_designed[:, None], raw_log, 0.0)
                current_log = torch.gather(log_probabilities, 1, output_index).squeeze(1)
                log_probabilities.scatter_(
                    1,
                    output_index,
                    torch.where(active[:, None], member_log, current_log)[:, None],
                )
                member_probability = member_designed[:, None] * probabilities
                current_probability = torch.gather(sampling_probabilities, 1, output_index).squeeze(1)
                sampling_probabilities.scatter_(
                    1,
                    output_index,
                    torch.where(active[:, None], member_probability, current_probability)[:, None],
                )
                state_index = member_index[:, :, None].expand(-1, 1, token_states.shape[-1])
                current_state = torch.gather(token_states, 1, state_index).squeeze(1)
                token_states.scatter_(
                    1,
                    state_index,
                    torch.where(active[:, None], embedded_token, current_state)[:, None],
                )

        return NaMpnnSamplingOutput(
            sequence_tokens=sequence_tokens,
            log_probabilities=log_probabilities,
            sampling_probabilities=sampling_probabilities,
            decoding_ranks=ranks,
            batch_size=[num_samples],
        )

    # ------------------------------------------------------------------
    # Realized design and specificity operations.
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def design(
        self,
        input: NaMpnnInput,
        *,
        num_candidates: int = 1,
        temperature: float = 0.1,
        global_logit_bias: Tensor | None = None,
        num_neighbors: int | None = None,
        generator: torch.Generator | None = None,
    ) -> NaMpnnSamplingOutput:
        r"""
        Design protein, DNA, and RNA sequence candidates on a fixed backbone.

        Args:
            input: One processed mixed-polymer target.
            num_candidates: Number of sequence candidates to sample.
            temperature: Positive sampling temperature.
            global_logit_bias: Optional bias with shape ``(vocab_size,)``.
            num_neighbors: Optional nearest-neighbor count for this operation.
            generator: Random generator used for orders and token sampling.

        Returns:
            Raw sequence samples and their sampling distributions.
        """

        self._require_single_target(input)
        if num_candidates <= 0:
            _invalid("num_candidates", num_candidates)
        return self.sample(
            input,
            num_samples=num_candidates,
            temperature=temperature,
            global_logit_bias=global_logit_bias,
            num_neighbors=num_neighbors,
            generator=generator,
        )

    @torch.inference_mode()
    def specificity_profile(
        self,
        input: NaMpnnInput,
        *,
        num_samples: int = 30,
        temperature: float = 0.6,
        global_logit_bias: Tensor | None = None,
        num_neighbors: int | None = None,
        generator: torch.Generator | None = None,
    ) -> NaMpnnSamplingOutput:
        r"""
        Estimate nucleic-acid base preferences with protein identities fixed.

        Args:
            input: One processed mixed-polymer target.
            num_samples: Number of sampled sequence contexts to average.
            temperature: Positive sampling temperature.
            global_logit_bias: Optional bias with shape ``(vocab_size,)``.
            num_neighbors: Optional nearest-neighbor count for this operation.
            generator: Random generator used for orders and token sampling.

        Returns:
            Raw sequence samples and per-position sampling distributions.
        """

        self._require_single_target(input)
        na_design = input.design_mask & (input.polymer_type_index != 0)
        fixed = NaMpnnInput(  # type: ignore[call-arg]
            atom_coordinates=input.atom_coordinates,
            atom_mask=input.atom_mask,
            sequence_tokens=input.sequence_tokens,
            residue_mask=input.residue_mask,
            residue_index=input.residue_index,
            chain_index=input.chain_index,
            polymer_type_index=input.polymer_type_index,
            design_mask=na_design,
            tied_group_ids=input.tied_group_ids,
            component_index=input.component_index,
            tied_weights=input.tied_weights,
            residue_logit_bias=input.residue_logit_bias,
            allowed_token_mask=input.allowed_token_mask,
            pair_logit_bias=input.pair_logit_bias,
            batch_size=[],
        )
        return self.sample(
            fixed,
            num_samples=num_samples,
            temperature=temperature,
            global_logit_bias=global_logit_bias,
            num_neighbors=num_neighbors,
            generator=generator,
        )

    # ------------------------------------------------------------------
    # Shared validation and evaluation-axis helpers.
    # ------------------------------------------------------------------

    @staticmethod
    def _require_single_target(input: NaMpnnInput) -> None:
        r"""
        Reject an execution-batch axis on one-target biological operations.
        """

        if input.atom_coordinates.shape[0] != 1:
            _invalid("input", f"batch of {input.atom_coordinates.shape[0]}")

    @staticmethod
    def _repeat_input(input: NaMpnnInput, num_repeats: int) -> NaMpnnInput:
        r"""
        Repeat one biological input into ``num_repeats`` aligned evaluation rows.
        """

        def repeat(value: Tensor) -> Tensor:
            return value.repeat_interleave(num_repeats, dim=0)

        pair_logit_bias = input.pair_logit_bias

        return NaMpnnInput(  # type: ignore[call-arg]
            atom_coordinates=repeat(input.atom_coordinates),
            atom_mask=repeat(input.atom_mask),
            sequence_tokens=repeat(input.sequence_tokens),
            residue_mask=repeat(input.residue_mask),
            residue_index=repeat(input.residue_index),
            chain_index=repeat(input.chain_index),
            polymer_type_index=repeat(input.polymer_type_index),
            design_mask=repeat(input.design_mask),
            tied_group_ids=repeat(input.tied_group_ids),
            component_index=repeat(input.component_index),
            tied_weights=repeat(input.tied_weights),
            residue_logit_bias=repeat(input.residue_logit_bias),
            allowed_token_mask=repeat(input.allowed_token_mask),
            pair_logit_bias=None if pair_logit_bias is None else repeat(pair_logit_bias),
            batch_size=[],
        )

    @staticmethod
    def _generate_ranks(
        residue_mask: Tensor,
        design_mask: Tensor,
        tied_group_ids: Tensor,
        generator: torch.Generator | None,
        *,
        protein_mask: Tensor | None = None,
    ) -> Tensor:
        r"""Draw fixed-first causal ranks from half-normal unit priorities."""

        designed = residue_mask & design_mask
        if protein_mask is None:
            row_class = designed.to(torch.long)
        else:
            row_class = torch.where(designed, torch.where(protein_mask, 1, 2), 0)
        unit_class = torch.full_like(tied_group_ids, 3).scatter_reduce(
            1,
            tied_group_ids,
            row_class,
            reduce="amin",
            include_self=True,
        )
        decoding_priority = (
            torch.empty_like(tied_group_ids, dtype=torch.get_default_dtype()).normal_(generator=generator).abs()
        )
        bounded_priority = decoding_priority / (1 + decoding_priority)
        unit_key = bounded_priority + unit_class.to(decoding_priority.dtype)
        unit_order = torch.argsort(unit_key, dim=1)
        positions = torch.ones_like(tied_group_ids).cumsum(dim=1) - 1
        unit_rank = torch.zeros_like(unit_order).scatter(1, unit_order, positions)
        return torch.gather(unit_rank, 1, tied_group_ids)

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
        mask = residue_mask.to(node_states.dtype)
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

    def _decode_positions(
        self,
        positions: Tensor,
        active: Tensor,
        edge_states: Tensor,
        neighbor_indices: Tensor,
        token_states: Tensor,
        block_states: list[Tensor],
        backward_mask: Tensor,
        forward_context: Tensor,
        residue_mask: Tensor,
        neighbor_valid: Tensor,
    ) -> Tensor:
        r"""
        Incrementally decode one independently selected position per sample; the raw projection over every class
        (the sampler excludes untrained classes after its group arithmetic, so that arithmetic stays finite).
        """

        def gather_rows(value: Tensor) -> Tensor:
            index = positions.reshape(positions.shape[0], 1, *([1] * (value.ndim - 2)))
            return torch.gather(value, 1, index.expand(-1, 1, *value.shape[2:]))

        edge_slice = gather_rows(edge_states)
        neighbor_slice = gather_rows(neighbor_indices)
        sequence_edge = cat_neighbors_nodes(token_states, edge_slice, neighbor_slice)
        forward_slice = gather_rows(forward_context)
        backward_slice = gather_rows(backward_mask)
        mask_slice = gather_rows(residue_mask).to(dtype=token_states.dtype) * active[:, None]
        for block_index, block in enumerate(self.decoder_blocks):
            decoder_context = cat_neighbors_nodes(block_states[block_index], sequence_edge, neighbor_slice)
            node_slice = gather_rows(block_states[block_index])
            updated = block(
                node_slice,
                backward_slice * decoder_context + forward_slice,
                mask_slice,
                gather_rows(neighbor_valid),
            )
            state_index = positions[:, None, None].expand(-1, 1, updated.shape[-1])
            current = gather_rows(block_states[block_index + 1])
            block_states[block_index + 1].scatter_(
                1,
                state_index,
                torch.where(active[:, None, None], updated, current),
            )
        return self.output_projection(gather_rows(block_states[-1]).squeeze(1))

    @staticmethod
    def _pair_contribution(
        pair_logit_bias: Tensor,
        member: Tensor,
        group_member_mask: Tensor,
        sequence_tokens: Tensor,
        vocab_size: int,
    ) -> Tensor:
        r"""
        Directed adjacent-token pair contribution for one target member row.

        External context rows use the pre-rank-block sampled tokens, ``self`` stays
        at PAD, and another internal member is evaluated at the shared candidate
        token via the diagonal. Both directions of a mirrored internal pair
        therefore contribute without averaging.
        """

        samples, num_residues = sequence_tokens.shape
        device = sequence_tokens.device
        row_index = member[:, None, None, None, None].expand(-1, 1, vocab_size, num_residues, vocab_size)
        row_bias = torch.gather(pair_logit_bias, 1, row_index).squeeze(1)
        context = sequence_tokens.clone()
        context.scatter_(1, member[:, None], _PAD)
        gathered = torch.gather(
            row_bias,
            -1,
            context[:, None, :, None].expand(samples, vocab_size, num_residues, 1),
        ).squeeze(
            -1
        )  # [S, V, L]
        positions = torch.arange(num_residues, device=device)
        internal_others = group_member_mask & (positions[None, :] != member[:, None])
        gathered = gathered.masked_fill(internal_others[:, None, :], 0.0)
        contribution = gathered.sum(dim=-1)
        diagonal = row_bias.diagonal(dim1=1, dim2=3).transpose(1, 2)
        contribution = contribution + (diagonal * internal_others[:, None, :]).sum(dim=-1)
        return contribution


class NaMpnnFeatureEncoder(nn.Module):
    r"""
    Encode protein and nucleic-acid coordinates as an MPNN graph.

    Args:
        config: NA-MPNN architecture configuration.
        backend: Backend used by neighbour selection and geometry.
    """

    def __init__(self, config: NaMpnnConfig, *, backend: str = "auto") -> None:
        super().__init__()
        self.backend = resolve_config_backend(backend, supported=("oops", "eager"), public_default="eager")
        self.num_neighbors = config.num_neighbors
        self.num_rbf = config.num_rbf
        self.num_polymer_types = 6
        self.include_predicted_na_n = config.include_predicted_na_n
        self.nitrogen_index = NA_MPNN_ATOM_INDEX["N"]
        self.alpha_carbon_index = NA_MPNN_ATOM_INDEX["CA"]
        self.carbonyl_carbon_index = NA_MPNN_ATOM_INDEX["C"]
        self.oxygen_4_prime_index = NA_MPNN_ATOM_INDEX["O4'"]
        self.carbon_1_prime_index = NA_MPNN_ATOM_INDEX["C1'"]
        self.carbon_2_prime_index = NA_MPNN_ATOM_INDEX["C2'"]
        total_atoms = len(NA_MPNN_ATOM_NAMES) + 1 + int(self.include_predicted_na_n)
        self.relative_position = MpnnRelativePositionEncoding(config.positional_embedding_size)
        self.edge_projection = nn.Linear(
            config.positional_embedding_size + config.num_rbf * total_atoms * total_atoms,
            config.feature_size,
            bias=False,
        )
        self.edge_norm = LayerNorm(config.feature_size, backend=backend)
        self.node_projection = nn.Linear(self.num_polymer_types, config.feature_size, bias=False)
        self.node_norm = LayerNorm(config.feature_size, backend=backend)

    def forward(
        self,
        atom_coordinates: Tensor,
        atom_mask: Tensor,
        residue_mask: Tensor,
        residue_index: Tensor,
        chain_index: Tensor,
        polymer_type_index: Tensor,
        *,
        num_neighbors: int | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        r"""
        Build graph features from mixed-polymer coordinates.

        Args:
            atom_coordinates: Atom coordinates with shape ``(B, N, A, 3)``.
            atom_mask: Atom validity mask with shape ``(B, N, A)``.
            residue_mask: Residue validity mask with shape ``(B, N)``.
            residue_index: Chain-local residue indices with shape ``(B, N)``.
            chain_index: Molecular-instance indices with shape ``(B, N)``.
            polymer_type_index: Polymer category IDs with shape ``(B, N)``.
                Available residue rows require exact IDs in the six-class feature vocabulary;
                excluded payload is ignored before lookup.
            num_neighbors: Optional nearest-neighbor count for this operation.

        Returns:
            node_features: Node features with shape ``(B, N, C)``.
            edge_features: Neighbor-edge features with shape ``(B, N, K, C)``.
            neighbor_indices: Neighbor indices with shape ``(B, N, K)``.
            neighbor_valid: Real-neighbor mask with shape ``(B, N, K)`` (the encoder's
                ``attend_mask``: 1 for a real neighbor, 0 for a filler slot).
        """

        polymer_type_index = project_model_categories(
            polymer_type_index,
            residue_mask,
            num_classes=self.num_polymer_types,
            feature="polymer categories",
        )
        atom_coordinates = project_model_coordinates(
            atom_coordinates, atom_mask & residue_mask.unsqueeze(-1), feature="mixed-polymer backbone"
        )
        dtype = atom_coordinates.dtype
        atom_mask = atom_mask.to(dtype=dtype)
        valid = residue_mask.to(dtype=dtype)
        protein = (polymer_type_index == 0).to(dtype=dtype)
        dna = (polymer_type_index == 1).to(dtype=dtype)
        rna = (polymer_type_index == 2).to(dtype=dtype)

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

        _, neighbor_indices, neighbor_valid = masked_knn(
            alpha_carbon + reference,
            valid,
            self.num_neighbors if num_neighbors is None else num_neighbors,
            backend=self.backend,
        )
        pair_rbf = _all_pair_rbf(all_coordinates, all_mask, neighbor_indices, self.num_rbf, backend=self.backend)
        neighbor_residue_index = gather_nodes(residue_index.unsqueeze(-1), neighbor_indices).squeeze(-1)
        residue_offset = residue_index.unsqueeze(-1) - neighbor_residue_index
        neighbor_chain_index = gather_nodes(chain_index.unsqueeze(-1), neighbor_indices).squeeze(-1)
        same_chain = (chain_index.unsqueeze(-1) == neighbor_chain_index).long()
        position_features = self.relative_position(residue_offset.long(), same_chain)
        edge_features = self.edge_norm(
            self.edge_projection(torch.cat((position_features, pair_rbf), dim=-1).to(self.edge_projection.weight.dtype))
        )

        polymer_one_hot = F.one_hot(
            polymer_type_index,
            num_classes=self.num_polymer_types,
        ).to(dtype=self.node_projection.weight.dtype)
        node_features = self.node_norm(self.node_projection(polymer_one_hot))
        return node_features, edge_features, neighbor_indices, neighbor_valid


def _virtual_na_n(o4prime: Tensor, c1prime: Tensor, c2prime: Tensor) -> Tensor:
    r"""
    Reconstruct the idealized nucleic-acid base nitrogen coordinate.
    """

    b = c1prime - o4prime
    c = c2prime - c1prime
    a = torch.cross(b, c, dim=-1)
    return -0.56967352 * a + 0.51055973 * b - 0.53122153 * c + c1prime


def _all_pair_rbf(
    coordinates: Tensor, atom_mask: Tensor, neighbor_indices: Tensor, num_bins: int, *, backend: str = "auto"
) -> Tensor:
    r"""
    RBF-encode every ordered atom-pair distance for each neighbor edge.
    """

    distances = neighbor_atom_distances(coordinates, neighbor_indices, eps=1e-6, backend=backend)
    rbf = radial_basis(distances, num_bins)
    neighbor_mask = gather_nodes(atom_mask, neighbor_indices)
    rbf = rbf * atom_mask[:, :, None, :, None, None] * neighbor_mask[:, :, :, None, :, None]
    return rbf.flatten(start_dim=-3)


# ===========================================================================
# NA-MPNN typed inputs and outputs.
# ===========================================================================


@tensorclass
class NaMpnnInput:
    r"""
    Inputs for NA-MPNN sequence design and scoring.

    Attributes:
        atom_coordinates: Polymer atom coordinates with shape
            ``(batch_size, num_residues, num_atom_types, 3)``.
        atom_mask: Atom-validity mask with shape
            ``(batch_size, num_residues, num_atom_types)``.
        sequence_tokens: Canonical classes with shape ``(batch_size, num_residues)``.
        residue_mask: Valid-residue mask with shape ``(batch_size, num_residues)``.
        residue_index: Residue indices with shape ``(batch_size, num_residues)``.
        chain_index: Chain indices with shape ``(batch_size, num_residues)``.
        polymer_type_index: Polymer-type IDs with shape
            ``(batch_size, num_residues)``.
        design_mask: Positions eligible for design with shape
            ``(batch_size, num_residues)``.
        tied_group_ids: Autoregressive unit ID for each residue row.
        component_index: Canonical component indices aligned to residue rows.
        tied_weights: Tied-group weights with shape ``(batch_size, num_residues)``.
        residue_logit_bias: Additive bias with shape
            ``(batch_size, num_residues, vocab_size)``.
        allowed_token_mask: Token support with shape
            ``(batch_size, num_residues, vocab_size)``.
        pair_logit_bias: Optional pair-conditioned bias with shape
            ``(batch_size, num_residues, vocab_size, num_residues, vocab_size)``.
    """

    atom_coordinates: Tensor
    atom_mask: Tensor
    sequence_tokens: Tensor
    residue_mask: Tensor
    residue_index: Tensor
    chain_index: Tensor
    polymer_type_index: Tensor
    design_mask: Tensor
    tied_group_ids: Tensor
    component_index: Tensor
    tied_weights: Tensor
    residue_logit_bias: Tensor
    allowed_token_mask: Tensor
    pair_logit_bias: Tensor | None = None


class NaMpnnOutput(TensorCollection):
    r"""
    Outputs from one NA-MPNN forward pass.

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


class NaMpnnSamplingOutput(TensorCollection):
    r"""
    Outputs from NA-MPNN autoregressive sampling.

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
