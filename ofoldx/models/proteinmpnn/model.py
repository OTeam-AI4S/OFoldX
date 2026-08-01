r"""
ProteinMPNN fixed-backbone sequence model.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple, cast

import torch
import torch.nn.functional as F
from tensordict import tensorclass
from torch import Tensor, nn

from ...data.molecular import AMINO_ACID_ONE_TO_THREE
from ...data.structure import Structure
from ...data.tokenization import TOKEN_ALPHABET, TOKEN_INDEX
from ...model import Model, _TensorCollection
from ...modules.backend import LayerNorm, gather_nodes, masked_knn
from ...modules.geometry import virtual_beta_carbon
from .config import ProteinMpnnConfig

AMINO_ACID_COMPONENT_IDS = tuple(AMINO_ACID_ONE_TO_THREE[symbol] for symbol in TOKEN_ALPHABET[: TOKEN_INDEX["X"]])
AMINO_ACID_TOKEN_BY_COMPONENT = {component_id: token for token, component_id in enumerate(AMINO_ACID_COMPONENT_IDS)}
AMINO_ACID_UNKNOWN_TOKEN = TOKEN_INDEX["X"]


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
        feature_class = ProteinMpnnCaFeatureEncoder if self.config.ca_only else ProteinMpnnFeatureEncoder
        self.feature_encoder = feature_class(self.config.hidden_size, self.config.num_neighbors)
        self.edge_embedding = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.encoder_blocks = nn.ModuleList(
            ProteinMpnnEncoderBlock(
                self.config.hidden_size,
                self.config.hidden_size * 2,
                dropout=self.config.dropout,
            )
            for _ in range(self.config.num_encoder_blocks)
        )
        self.token_embedding = nn.Embedding(len(AMINO_ACID_COMPONENT_IDS) + 1, self.config.hidden_size)
        self.decoder_blocks = nn.ModuleList(
            ProteinMpnnDecoderBlock(
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

        backbone_coordinates = input.backbone_coordinates
        residue_mask = input.residue_mask
        if coordinate_noise:
            noise = torch.empty_like(backbone_coordinates).normal_(generator=generator)
            backbone_coordinates = backbone_coordinates + coordinate_noise * noise
        mask = residue_mask.to(dtype=backbone_coordinates.dtype)
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
            raise ValueError("num_decoding_orders must be positive")
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
            raise ValueError("num_decoding_orders must be positive")
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
        node_states, edge_states, neighbor_indices, neighbor_valid = self.encode(
            repeated,
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

        node_states, edge_states, neighbor_indices, neighbor_valid = self.encode(
            repeated,
            coordinate_noise=coordinate_noise,
            generator=generator,
        )
        length = repeated.sequence_tokens.shape[1]
        vocabulary = len(AMINO_ACID_COMPONENT_IDS) + 1
        representative_positions = self._representative_positions(input.tied_group_ids)
        ordered_group_ranks = torch.sort(ranks[:, representative_positions], dim=-1).values
        num_units = ordered_group_ranks.shape[1]

        backward_mask, forward_mask = _autoregressive_masks(
            ranks,
            neighbor_indices,
            repeated.residue_mask.to(dtype=node_states.dtype),
            neighbor_valid.to(dtype=node_states.dtype),
        )
        token_states = torch.zeros_like(node_states)
        encoder_edges = _cat_neighbors_nodes(token_states, edge_states, neighbor_indices)
        encoder_context = _cat_neighbors_nodes(node_states, encoder_edges, neighbor_indices)
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
            sequence_edge = _cat_neighbors_nodes(
                token_states[batch_indices],
                edge_at_position,
                neighbor_index,
            )
            encoder_at_position = forward_context[batch_indices, positions].unsqueeze(1)
            mask_at_position = repeated.residue_mask[batch_indices, positions, None].to(node_states.dtype)
            for block_index, block in enumerate(self.decoder_blocks):
                decoder_context = _cat_neighbors_nodes(
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
        if not bool((input.residue_mask & input.design_positions).any()):
            raise ValueError("score requires at least one valid scoring position")
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
        return ProteinMpnnDesignOutput(sample=sampled, input_score=input_score)

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
        length = tied_group_ids.shape[1]
        row_index = torch.arange(length, device=device)
        unit_first = torch.full((length,), length, dtype=torch.long, device=device)
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

        token_states = self.token_embedding(sequence_tokens)
        sequence_edges = _cat_neighbors_nodes(token_states, edge_states, neighbor_indices)
        encoder_edges = _cat_neighbors_nodes(torch.zeros_like(token_states), edge_states, neighbor_indices)
        encoder_context = _cat_neighbors_nodes(node_states, encoder_edges, neighbor_indices)
        backward_mask, forward_mask = _autoregressive_masks(
            decoding_ranks,
            neighbor_indices,
            residue_mask.to(dtype=node_states.dtype),
            neighbor_valid.to(dtype=node_states.dtype),
        )
        forward_context = forward_mask * encoder_context
        checkpointing = self.gradient_checkpointing and self.training and torch.is_grad_enabled()
        for block in self.decoder_blocks:
            decoder_context = _cat_neighbors_nodes(node_states, sequence_edges, neighbor_indices)
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
        encoder_edges = _cat_neighbors_nodes(token_states, edge_states, neighbor_indices)
        encoder_context = _cat_neighbors_nodes(node_states, encoder_edges, neighbor_indices)
        _, forward_mask = _autoregressive_masks(
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
        node_states, edge_states, neighbor_indices, neighbor_valid = self.encode(
            repeated,
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
        total = supported_probabilities.sum(dim=-1, keepdim=True)
        # Fall back to the unconstrained distribution where the hard support mask leaves no eligible token:
        # sampling needs a finite distribution, and dividing by zero would emit NaN sampling probabilities that
        # ``torch.multinomial`` rejects. The reference keeps a ``probs * 0.001`` floor for the same reason
        # (protein_mpnn_utils.py:1174); here the fallback is confined to the degenerate empty-support rows so
        # ordinary masking is unchanged.
        empty = total == 0
        supported_probabilities = torch.where(empty, probabilities, supported_probabilities)
        total = torch.where(empty, probabilities.sum(dim=-1, keepdim=True), total)
        return supported_probabilities / total


class ProteinMpnnMessage(nn.Module):
    r"""Compute ProteinMPNN neighbor messages with three projections and GELU."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.activation = nn.GELU()
        self.hidden_projection = nn.Linear(hidden_size, hidden_size)
        self.output_projection = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        r"""Return neighbor messages with ``hidden_size`` channels."""

        hidden = self.activation(self.input_projection(x))
        hidden = self.activation(self.hidden_projection(hidden))
        return self.output_projection(hidden)


class ProteinMpnnFeedForward(nn.Module):
    r"""Apply the position-wise ProteinMPNN transition."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.input_projection = nn.Linear(hidden_size, intermediate_size)
        self.activation = nn.GELU()
        self.output_projection = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        r"""Return the additive transition update."""

        return self.output_projection(self.activation(self.input_projection(x)))


class ProteinMpnnRelativePositionEncoding(nn.Module):
    r"""Embed signed residue offsets with a distinct inter-chain bucket."""

    def __init__(self, num_embeddings: int, max_relative_feature: int = 32) -> None:
        super().__init__()
        self.max_relative_feature = max_relative_feature
        self.projection = nn.Linear(2 * max_relative_feature + 2, num_embeddings)

    def forward(self, offset: Tensor, same_chain: Tensor) -> Tensor:
        r"""Project bucketed residue offsets to edge-position features."""

        clipped = torch.clamp(offset + self.max_relative_feature, 0, 2 * self.max_relative_feature)
        bucket = clipped * same_chain + (1 - same_chain) * (2 * self.max_relative_feature + 1)
        features = F.one_hot(bucket, 2 * self.max_relative_feature + 2).to(dtype=self.projection.weight.dtype)
        return self.projection(features)


class ProteinMpnnEncoderBlock(nn.Module):
    r"""Apply one ProteinMPNN node, transition, and edge update."""

    def __init__(self, hidden_size: int, edge_input_size: int, *, dropout: float, scale: float = 30.0) -> None:
        super().__init__()
        self.scale = scale
        self.node_message = ProteinMpnnMessage(hidden_size + edge_input_size, hidden_size)
        self.node_dropout = nn.Dropout(dropout)
        self.node_norm = LayerNorm(hidden_size)
        self.transition = ProteinMpnnFeedForward(hidden_size, hidden_size * 4)
        self.transition_dropout = nn.Dropout(dropout)
        self.transition_norm = LayerNorm(hidden_size)
        self.edge_message = ProteinMpnnMessage(hidden_size + edge_input_size, hidden_size)
        self.edge_dropout = nn.Dropout(dropout)
        self.edge_norm = LayerNorm(hidden_size)

    def forward(
        self,
        node_states: Tensor,
        edge_states: Tensor,
        neighbor_indices: Tensor,
        node_mask: Tensor | None,
        attend_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        r"""Update aligned node and edge states."""

        neighbor_states = _cat_neighbors_nodes(node_states, edge_states, neighbor_indices)
        source_states = node_states.unsqueeze(-2).expand(-1, -1, neighbor_states.shape[-2], -1)
        node_message = self.node_message(torch.cat((source_states, neighbor_states), dim=-1))
        if attend_mask is not None:
            node_message = node_message * attend_mask.unsqueeze(-1)
        node_states = self.node_norm(node_states + self.node_dropout(node_message.sum(dim=-2) / self.scale))
        node_states = self.transition_norm(node_states + self.transition_dropout(self.transition(node_states)))
        if node_mask is not None:
            node_states = node_states * node_mask.unsqueeze(-1)

        neighbor_states = _cat_neighbors_nodes(node_states, edge_states, neighbor_indices)
        source_states = node_states.unsqueeze(-2).expand(-1, -1, neighbor_states.shape[-2], -1)
        edge_message = self.edge_message(torch.cat((source_states, neighbor_states), dim=-1))
        edge_states = self.edge_norm(edge_states + self.edge_dropout(edge_message))
        return node_states, edge_states


class ProteinMpnnDecoderBlock(nn.Module):
    r"""Apply one masked ProteinMPNN message update and transition."""

    def __init__(self, hidden_size: int, edge_input_size: int, *, dropout: float, scale: float = 30.0) -> None:
        super().__init__()
        self.scale = scale
        self.message = ProteinMpnnMessage(hidden_size + edge_input_size, hidden_size)
        self.message_dropout = nn.Dropout(dropout)
        self.message_norm = LayerNorm(hidden_size)
        self.transition = ProteinMpnnFeedForward(hidden_size, hidden_size * 4)
        self.transition_dropout = nn.Dropout(dropout)
        self.transition_norm = LayerNorm(hidden_size)

    def forward(
        self,
        node_states: Tensor,
        edge_context: Tensor,
        node_mask: Tensor | None = None,
        attend_mask: Tensor | None = None,
    ) -> Tensor:
        r"""Update node states from caller-assembled neighbor context."""

        source_states = node_states.unsqueeze(-2).expand(*([-1] * (node_states.ndim - 1)), edge_context.shape[-2], -1)
        message = self.message(torch.cat((source_states, edge_context), dim=-1))
        if attend_mask is not None:
            message = message * attend_mask.unsqueeze(-1)
        node_states = self.message_norm(node_states + self.message_dropout(message.sum(dim=-2) / self.scale))
        node_states = self.transition_norm(node_states + self.transition_dropout(self.transition(node_states)))
        if node_mask is not None:
            node_states = node_states * node_mask.unsqueeze(-1)
        return node_states


class ProteinMpnnFeatureEncoder(nn.Module):
    r"""Build ProteinMPNN's 25-distance full-backbone edge graph."""

    distance_indices: Tensor

    def __init__(
        self,
        edge_size: int,
        num_neighbors: int,
        *,
        num_positional_embeddings: int = 16,
        num_rbf: int = 16,
    ) -> None:
        super().__init__()
        self.num_neighbors = num_neighbors
        self.num_rbf = num_rbf
        self.relative_position = ProteinMpnnRelativePositionEncoding(num_positional_embeddings)
        self.edge_projection = nn.Linear(num_positional_embeddings + 25 * num_rbf, edge_size, bias=False)
        self.edge_norm = LayerNorm(edge_size)
        source_atoms = (0, 1, 2, 3, 4, 0, 0, 0, 0, 1, 1, 1, 4, 4, 3, 1, 2, 3, 4, 2, 3, 4, 2, 3, 2)
        target_atoms = (0, 1, 2, 3, 4, 1, 2, 3, 4, 2, 3, 4, 2, 3, 2, 0, 0, 0, 0, 1, 1, 1, 4, 4, 3)
        self.register_buffer(
            "distance_indices",
            torch.tensor(tuple(source * 5 + target for source, target in zip(source_atoms, target_atoms, strict=True))),
            persistent=False,
        )

    def forward(
        self,
        backbone_coordinates: Tensor,
        residue_mask: Tensor,
        residue_index: Tensor,
        chain_index: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        r"""Return edge features, neighbor indices, and the real-neighbor mask."""

        nitrogen = backbone_coordinates[:, :, 0]
        alpha_carbon = backbone_coordinates[:, :, 1]
        carbon = backbone_coordinates[:, :, 2]
        oxygen = backbone_coordinates[:, :, 3]
        beta_carbon = virtual_beta_carbon(nitrogen, alpha_carbon, carbon)
        atoms = torch.stack((alpha_carbon, nitrogen, carbon, oxygen, beta_carbon), dim=2)
        neighbor_distances, neighbor_indices, neighbor_valid = masked_knn(
            alpha_carbon,
            residue_mask,
            self.num_neighbors,
        )

        neighbor_atoms = gather_nodes(atoms.flatten(start_dim=-2), neighbor_indices).unflatten(-1, (5, 3))
        pair_distances = torch.sqrt(
            torch.sum(
                (atoms[:, :, None, :, None] - neighbor_atoms[:, :, :, None]) ** 2,
                dim=-1,
            )
            + 1e-6
        )
        pair_distances = pair_distances.flatten(start_dim=-2).index_select(-1, self.distance_indices)
        pair_distances = torch.cat((neighbor_distances.unsqueeze(-1), pair_distances[..., 1:]), dim=-1)
        distance_features = _radial_basis(pair_distances, self.num_rbf).flatten(start_dim=-2)

        neighbor_residue_index = gather_nodes(residue_index.unsqueeze(-1), neighbor_indices).squeeze(-1)
        residue_offsets = residue_index.unsqueeze(-1) - neighbor_residue_index
        neighbor_chain_index = gather_nodes(chain_index.unsqueeze(-1), neighbor_indices).squeeze(-1)
        same_chain = (chain_index.unsqueeze(-1) == neighbor_chain_index).long()
        position_features = self.relative_position(residue_offsets.long(), same_chain)
        edges = torch.cat((position_features, distance_features), dim=-1)
        return self.edge_norm(self.edge_projection(edges)), neighbor_indices, neighbor_valid


class ProteinMpnnCaFeatureEncoder(nn.Module):
    r"""
    Encode C-alpha coordinates as a ProteinMPNN edge graph.

    Args:
        hidden_size: Width of the returned edge features.
        num_neighbors: Maximum number of nearest neighbors per residue.
    """

    def __init__(self, hidden_size: int, num_neighbors: int) -> None:
        super().__init__()
        self.num_neighbors = num_neighbors
        self.relative_position = ProteinMpnnRelativePositionEncoding(16)
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

        row0, row1, row2 = rotations.unbind(-2)
        xx, xy, xz = row0.unbind(-1)
        yx, yy, yz = row1.unbind(-1)
        zx, zy, zz = row2.unbind(-1)
        diagonal = torch.stack((xx, yy, zz), dim=-1)
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
                    zy - yz,
                    xz - zx,
                    yx - xy,
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
            endpoint = torch.zeros_like(alpha_carbons[:, :1].unsqueeze(-1).expand(-1, -1, 3, 3))
            frames = torch.cat((endpoint, frames, endpoint, endpoint), dim=1)
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
            edges: Edge features with shape ``(B, L, K, H)``.
            neighbor_indices: Neighbor indices with shape ``(B, L, K)``.
            neighbor_valid: Real-neighbor mask with shape ``(B, L, K)`` (the encoder's
                ``attend_mask``).
        """

        neighbor_distances, neighbor_indices, neighbor_valid = masked_knn(
            backbone_coordinates,
            residue_mask,
            self.num_neighbors,
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
        orientation = self._orientations(backbone_coordinates, neighbor_indices)
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
        neighbor_residue_index = gather_nodes(residue_index.unsqueeze(-1), neighbor_indices).squeeze(-1)
        residue_offsets = residue_index.unsqueeze(-1) - neighbor_residue_index
        neighbor_chain_index = gather_nodes(chain_index.unsqueeze(-1), neighbor_indices).squeeze(-1)
        same_chain = (chain_index.unsqueeze(-1) == neighbor_chain_index).long()
        position_features = self.relative_position(residue_offsets.long(), same_chain)
        edges = torch.cat((position_features, distance_features, orientation), dim=-1)
        return self.edge_norm(self.edge_projection(edges)), neighbor_indices, neighbor_valid


def _cat_neighbors_nodes(node_states: Tensor, neighbor_states: Tensor, neighbor_indices: Tensor) -> Tensor:
    r"""Append the gathered neighbor-node states to an aligned edge context."""

    return torch.cat((neighbor_states, gather_nodes(node_states, neighbor_indices)), dim=-1)


def _radial_basis(
    distances: Tensor,
    num_bins: int = 16,
    lower_bound: float = 2.0,
    upper_bound: float = 22.0,
) -> Tensor:
    r"""Expand distances over evenly spaced Gaussian radial-basis centers."""

    centers = torch.linspace(
        lower_bound,
        upper_bound,
        num_bins,
        device=distances.device,
        dtype=distances.dtype,
    )
    width = (upper_bound - lower_bound) / num_bins
    return torch.exp(-(((distances.unsqueeze(-1) - centers) / width) ** 2))


def _autoregressive_masks(
    decoding_ranks: Tensor,
    neighbor_indices: Tensor,
    residue_mask: Tensor,
    neighbor_valid: Tensor,
) -> tuple[Tensor, Tensor]:
    r"""Build backward and forward masks from aligned decoding ranks."""

    neighbor_ranks = gather_nodes(decoding_ranks.unsqueeze(-1), neighbor_indices).squeeze(-1)
    backward = neighbor_ranks < decoding_ranks.unsqueeze(-1)
    query_mask = residue_mask.unsqueeze(-1).unsqueeze(-1) * neighbor_valid.unsqueeze(-1)
    return (
        query_mask * backward.unsqueeze(-1),
        query_mask * (~backward).unsqueeze(-1),
    )


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
    return _radial_basis(neighbor_distances, num_bins)


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


class ProteinMpnnDesignOutput(NamedTuple):
    r"""
    Raw ProteinMPNN design execution tensors.

    Attributes:
        sample: Autoregressive sequence samples in generation order.
        input_score: Raw evaluation of the input sequence after sampling.
    """

    sample: ProteinMpnnSamplingOutput
    input_score: ProteinMpnnOutput
