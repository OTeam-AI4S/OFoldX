r"""
NA-MPNN inverse-folding model for protein, DNA, RNA, and mixed-polymer structures.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch
import torch.nn.functional as F
from tensordict import tensorclass
from torch import Tensor, nn

from ...data.structure import Structure
from ...model import Model, _TensorCollection
from ...modules.backend import LayerNorm, gather_nodes, masked_knn
from ...modules.geometry import virtual_beta_carbon
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
#: Protein, DNA, and RNA residue components in NA-MPNN vocabulary order.
NA_MPNN_COMPONENT_IDS = (
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
NA_MPNN_TOKEN_BY_COMPONENT = {name: index for index, name in enumerate(NA_MPNN_COMPONENT_IDS)}
NA_MPNN_PADDING_TOKEN_ID = NA_MPNN_TOKEN_BY_COMPONENT["PAD"]
# Contiguous generatable token columns per polymer type. RNA shares the DNA columns
# in the shared-token mode and owns tokens 26..29 in separate-token mode.
NA_MPNN_PROTEIN_GENERATABLE_TOKEN_IDS = tuple(range(0, 20))
NA_MPNN_DNA_GENERATABLE_TOKEN_IDS = tuple(range(21, 25))
NA_MPNN_RNA_SHARED_GENERATABLE_TOKEN_IDS = tuple(range(21, 25))
NA_MPNN_RNA_SEPARATE_GENERATABLE_TOKEN_IDS = tuple(range(26, 30))
NA_MPNN_RNA_SHARED_INPUT_TOKENS = frozenset(
    (*NA_MPNN_RNA_SHARED_GENERATABLE_TOKEN_IDS, NA_MPNN_TOKEN_BY_COMPONENT["DX"])
)


class NaMpnnModel(Model):
    r"""
    Design protein and nucleic-acid sequences with NA-MPNN.

    Accepts [`NaMpnnInput`][ofoldx.models.nampnn.model.NaMpnnInput] and returns
    [`NaMpnnOutput`][ofoldx.models.nampnn.model.NaMpnnOutput].

    Args:
        config: Model architecture and inference configuration.

    Examples:
        >>> import torch
        >>> from ofoldx.data import MolecularSystem, PolymerSequence, Structure
        >>> from ofoldx.models.nampnn import NaMpnnConfig, NaMpnnModel, NaMpnnProcessor
        >>> sequence = PolymerSequence.from_one_letter("A", polymer_type="rna")
        >>> system = MolecularSystem.from_entities({"rna": sequence}, {"A": "rna"})
        >>> structure = Structure(
        ...     system=system,
        ...     coordinates=system.reference_coordinates,
        ...     coordinate_mask=system.reference_coordinate_mask,
        ...     occupancy=torch.zeros(system.atom_symbol.numel()),
        ...     occupancy_mask=torch.zeros(system.atom_symbol.numel(), dtype=torch.bool),
        ...     b_factor=torch.zeros(system.atom_symbol.numel()),
        ...     b_factor_mask=torch.zeros(system.atom_symbol.numel(), dtype=torch.bool),
        ...     source_symbols=(),
        ...     residue_id=torch.full((system.component_symbol.numel(),), -1),
        ...     insertion_code=torch.full((system.component_symbol.numel(),), -1),
        ...     batch_size=[],
        ... )
        >>> input = NaMpnnProcessor()(structure)
        >>> config = NaMpnnConfig(
        ...     {"hidden_size": 8, "feature_size": 8, "num_encoder_blocks": 1,
        ...      "num_decoder_blocks": 1, "num_neighbors": 1, "dropout": 0.0,
        ...      "num_positional_embeddings": 4, "num_rbf": 2,
        ...      "include_predicted_na_n": False}
        ... )
        >>> model = NaMpnnModel(config)
        >>> output = model(input, decoding_ranks=torch.zeros_like(input.sequence_tokens))
        >>> output.logits.shape
        torch.Size([1, 1, 33])
        >>> output.decoding_ranks.tolist()
        [[0]]
    """

    config_class = NaMpnnConfig
    config: NaMpnnConfig
    _gradient_checkpointing_func: Callable[..., object]

    def __init__(self, config: NaMpnnConfig | None = None) -> None:
        config = NaMpnnConfig() if config is None else config
        super().__init__(config)
        self.gradient_checkpointing = False
        self.vocab_size = len(NA_MPNN_COMPONENT_IDS)
        self.feature_encoder = NaMpnnFeatureEncoder(self.config)
        self.node_embedding = nn.Linear(self.config.feature_size, self.config.hidden_size)
        self.edge_embedding = nn.Linear(self.config.feature_size, self.config.hidden_size)
        self.encoder_blocks = nn.ModuleList(
            NaMpnnEncoderBlock(self.config.hidden_size, self.config.hidden_size * 2, dropout=self.config.dropout)
            for _ in range(self.config.num_encoder_blocks)
        )
        self.token_embedding = nn.Embedding(self.vocab_size, self.config.hidden_size)
        self.decoder_blocks = nn.ModuleList(
            NaMpnnDecoderBlock(self.config.hidden_size, self.config.hidden_size * 3, dropout=self.config.dropout)
            for _ in range(self.config.num_decoder_blocks)
        )
        self.output_projection = nn.Linear(self.config.hidden_size, self.vocab_size)
        self.post_init()

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
            self._validate_decoding_ranks(decoding_ranks, input.token_group_ids)
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
                input.design_positions,
                input.token_group_ids,
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
            Logits with shape ``(num_decoding_orders, num_residues, vocabulary_size)``
                and decoding ranks with shape ``(num_decoding_orders, num_residues)``.
        """

        self._require_single_target(input)
        if num_decoding_orders <= 0:
            raise ValueError("num_decoding_orders must be positive")
        repeated = self._repeat_input(input, num_decoding_orders)
        if decoding_ranks is None:
            decoding_ranks = self._generate_ranks(
                repeated.residue_mask,
                repeated.design_positions,
                repeated.token_group_ids,
                generator,
            )
        else:
            self._validate_decoding_ranks(decoding_ranks, repeated.token_group_ids)
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
            Logits with shape ``(num_decoding_orders, num_residues, vocabulary_size)``
                and all-zero ranks with shape ``(num_decoding_orders, num_residues)``.
        """

        self._require_single_target(input)
        if num_decoding_orders <= 0:
            raise ValueError("num_decoding_orders must be positive")
        node_states, edge_states, neighbor_indices, neighbor_valid = self.encode(
            input,
            num_neighbors=num_neighbors,
        )
        repeated = self._repeat_input(input, num_decoding_orders)
        node_states = node_states.repeat_interleave(num_decoding_orders, dim=0)
        edge_states = edge_states.repeat_interleave(num_decoding_orders, dim=0)
        neighbor_indices = neighbor_indices.repeat_interleave(num_decoding_orders, dim=0)
        neighbor_valid = neighbor_valid.repeat_interleave(num_decoding_orders, dim=0)
        logits = self._decode_unconditional_logits(
            node_states,
            edge_states,
            neighbor_indices,
            neighbor_valid,
            repeated.residue_mask,
        )
        decoding_ranks = torch.zeros(
            num_decoding_orders,
            input.residue_mask.shape[1],
            dtype=torch.long,
            device=input.atom_coordinates.device,
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
        logit_bias: Tensor | None = None,
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
            logit_bias: Optional bias with shape ``(vocabulary_size,)``.
            num_neighbors: Optional nearest-neighbor count for this operation.
            generator: Random generator used for orders and token sampling.

        Returns:
            Sampled token IDs, model log probabilities, sampling probabilities,
                and decoding ranks, each with a leading ``num_samples`` axis.
        """

        self._require_single_target(input)
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        repeated = self._repeat_input(input, num_samples)
        device = repeated.atom_coordinates.device
        length = repeated.sequence_tokens.shape[1]
        vocabulary = self.vocab_size

        if decoding_ranks is None:
            ranks = self._generate_ranks(
                repeated.residue_mask,
                repeated.design_positions,
                repeated.token_group_ids,
                generator,
            )
        else:
            ranks = decoding_ranks
            self._validate_decoding_ranks(ranks, repeated.token_group_ids)
        node_states, edge_states, neighbor_indices, neighbor_valid = self.encode(
            input,
            num_neighbors=num_neighbors,
        )
        node_states = node_states.repeat_interleave(num_samples, dim=0)
        edge_states = edge_states.repeat_interleave(num_samples, dim=0)
        neighbor_indices = neighbor_indices.repeat_interleave(num_samples, dim=0)
        neighbor_valid = neighbor_valid.repeat_interleave(num_samples, dim=0)
        hard_support = repeated.allowed_token_mask.bool()

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

        sequence_tokens = torch.full((num_samples, length), NA_MPNN_PADDING_TOKEN_ID, dtype=torch.long, device=device)
        log_probabilities = node_states.new_zeros(num_samples, length, vocabulary)
        sampling_probabilities = torch.zeros_like(log_probabilities)

        designed = repeated.design_positions & repeated.residue_mask
        tied_weights = repeated.tied_weights
        residue_bias = repeated.residue_logit_bias
        pair_logit_bias = repeated.pair_logit_bias

        position_index = torch.arange(length, device=device)
        group_ids = input.token_group_ids[0]
        first_by_group = torch.full((length,), length, dtype=torch.long, device=device)
        first_by_group.scatter_reduce_(0, group_ids, position_index, reduce="amin", include_self=True)
        representative_positions = position_index[position_index == first_by_group[group_ids]]
        unit_ids = group_ids[representative_positions]
        unit_member_mask = unit_ids[:, None] == group_ids[None, :]
        unit_members = (
            torch.where(
                unit_member_mask,
                position_index[None, :],
                length,
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
            group_logits = node_states.new_zeros(num_samples, vocabulary)
            group_residue_bias = node_states.new_zeros(num_samples, vocabulary)
            group_pair_bias = node_states.new_zeros(num_samples, vocabulary)
            group_support = torch.ones(num_samples, vocabulary, dtype=torch.bool, device=device)
            generated_unit = torch.ones(num_samples, dtype=torch.bool, device=device)
            group_has_designed = torch.zeros(num_samples, dtype=torch.bool, device=device)
            anchor_found = torch.zeros(num_samples, dtype=torch.bool, device=device)
            anchor = torch.full(
                (num_samples,),
                NA_MPNN_PADDING_TOKEN_ID,
                dtype=torch.long,
                device=device,
            )
            for member_slot in range(max_members):
                member = active_members[:, member_slot]
                active = member < length
                member = member.clamp_max(length - 1)
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
                )
                member_index = member[:, None]
                active_float = active[:, None].to(dtype=node_states.dtype)
                weight = torch.gather(tied_weights, 1, member_index).squeeze(1)
                member_bias = torch.gather(
                    residue_bias,
                    1,
                    member_index[:, :, None].expand(-1, 1, vocabulary),
                ).squeeze(1)
                member_support = torch.gather(
                    hard_support,
                    1,
                    member_index[:, :, None].expand(-1, 1, vocabulary),
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
                        vocabulary,
                    )
                group_support = group_support & (~active[:, None] | member_support)
                generated_unit = generated_unit & (~active | member_designed)
                group_has_designed = group_has_designed | (active & member_designed)
                fixed = active & ~member_designed
                anchor = torch.where(fixed & ~anchor_found, member_token, anchor)
                anchor_found = anchor_found | fixed
                member_records.append((member, active, member_designed, logits))

            total = group_logits + group_residue_bias + group_pair_bias
            if logit_bias is not None:
                total = total + logit_bias
            # Hard support intersects the combined logit last, before temperature and
            # the one group draw, matching the -1e8 pre-softmax masking; a
            # post-softmax intersection underflows to all-zero support when the top
            # logit is a disallowed token at low temperature.
            total = total.masked_fill(~group_support, float("-inf"))
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

                output_index = member_index[:, :, None].expand(-1, 1, vocabulary)
                raw_log = F.log_softmax(logits, dim=-1)
                member_log = member_designed[:, None] * raw_log
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
        logit_bias: Tensor | None = None,
        num_neighbors: int | None = None,
        generator: torch.Generator | None = None,
    ) -> NaMpnnSamplingOutput:
        r"""
        Design protein, DNA, and RNA sequence candidates on a fixed backbone.

        Args:
            input: One processed mixed-polymer target.
            num_candidates: Number of sequence candidates to sample.
            temperature: Positive sampling temperature.
            logit_bias: Optional bias with shape ``(vocabulary_size,)``.
            num_neighbors: Optional nearest-neighbor count for this operation.
            generator: Random generator used for orders and token sampling.

        Returns:
            Raw sequence samples and their sampling distributions.
        """

        self._require_single_target(input)
        if num_candidates <= 0:
            raise ValueError("num_candidates must be positive")
        return self.sample(
            input,
            num_samples=num_candidates,
            temperature=temperature,
            logit_bias=logit_bias,
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
        logit_bias: Tensor | None = None,
        num_neighbors: int | None = None,
        generator: torch.Generator | None = None,
    ) -> NaMpnnSamplingOutput:
        r"""
        Estimate nucleic-acid base preferences with protein identities fixed.

        Args:
            input: One processed mixed-polymer target.
            num_samples: Number of sampled sequence contexts to average.
            temperature: Positive sampling temperature.
            logit_bias: Optional bias with shape ``(vocabulary_size,)``.
            num_neighbors: Optional nearest-neighbor count for this operation.
            generator: Random generator used for orders and token sampling.

        Returns:
            Raw sequence samples and per-position sampling distributions.
        """

        self._require_single_target(input)
        na_design = input.design_positions & (input.polymer_type_index != 0)
        fixed = NaMpnnInput(  # type: ignore[call-arg]
            atom_coordinates=input.atom_coordinates,
            atom_mask=input.atom_mask,
            sequence_tokens=input.sequence_tokens,
            residue_mask=input.residue_mask,
            residue_index=input.residue_index,
            chain_index=input.chain_index,
            polymer_type_index=input.polymer_type_index,
            design_positions=na_design,
            token_group_ids=input.token_group_ids,
            structure=input.structure,
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
            logit_bias=logit_bias,
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

        if input.atom_coordinates.shape[0] != 1 or input.structure.system.system_entity_ptr.numel() != 2:
            raise ValueError("this NaMpnn operation returns one target and therefore accepts one input")

    @staticmethod
    def _repeat_input(input: NaMpnnInput, count: int) -> NaMpnnInput:
        r"""
        Repeat one biological input into ``count`` aligned evaluation rows.
        """

        def repeat(value: Tensor) -> Tensor:
            return value.repeat_interleave(count, dim=0)

        pair_logit_bias = input.pair_logit_bias

        return NaMpnnInput(  # type: ignore[call-arg]
            atom_coordinates=repeat(input.atom_coordinates),
            atom_mask=repeat(input.atom_mask),
            sequence_tokens=repeat(input.sequence_tokens),
            residue_mask=repeat(input.residue_mask),
            residue_index=repeat(input.residue_index),
            chain_index=repeat(input.chain_index),
            polymer_type_index=repeat(input.polymer_type_index),
            design_positions=repeat(input.design_positions),
            token_group_ids=repeat(input.token_group_ids),
            structure=input.structure,
            component_index=repeat(input.component_index),
            tied_weights=repeat(input.tied_weights),
            residue_logit_bias=repeat(input.residue_logit_bias),
            allowed_token_mask=repeat(input.allowed_token_mask),
            pair_logit_bias=None if pair_logit_bias is None else repeat(pair_logit_bias),
            batch_size=[],
        )

    @staticmethod
    def _validate_decoding_ranks(decoding_ranks: Tensor, token_group_ids: Tensor) -> None:
        r"""Validate exact explicit ranks without changing the requested order."""

        if decoding_ranks.shape != token_group_ids.shape:
            raise ValueError("decoding_ranks has the wrong exact shape")
        if decoding_ranks.dtype != torch.long:
            raise TypeError("decoding_ranks must have dtype torch.long")

        group_order = torch.argsort(token_group_ids, dim=1)
        groups_by_group = torch.gather(token_group_ids, 1, group_order)
        ranks_by_group = torch.gather(decoding_ranks, 1, group_order)
        same_group = groups_by_group[:, 1:] == groups_by_group[:, :-1]
        torch._assert(
            torch.all(~same_group | (ranks_by_group[:, 1:] == ranks_by_group[:, :-1])),
            "tied members must have equal decoding ranks",
        )

        rank_order = torch.argsort(decoding_ranks, dim=1)
        ranks_by_rank = torch.gather(decoding_ranks, 1, rank_order)
        groups_by_rank = torch.gather(token_group_ids, 1, rank_order)
        same_rank = ranks_by_rank[:, 1:] == ranks_by_rank[:, :-1]
        torch._assert(
            torch.all(~same_rank | (groups_by_rank[:, 1:] == groups_by_rank[:, :-1])),
            "distinct causal units must have distinct decoding ranks",
        )

    @staticmethod
    def _generate_ranks(
        residue_mask: Tensor,
        design_positions: Tensor,
        token_group_ids: Tensor,
        generator: torch.Generator | None,
        *,
        protein_mask: Tensor | None = None,
    ) -> Tensor:
        r"""Draw fixed-first causal ranks from half-normal unit priorities."""

        designed = residue_mask & design_positions
        if protein_mask is None:
            row_class = designed.to(torch.long)
        else:
            row_class = torch.where(designed, torch.where(protein_mask, 1, 2), 0)
        unit_class = torch.full_like(token_group_ids, 3).scatter_reduce(
            1,
            token_group_ids,
            row_class,
            reduce="amin",
            include_self=True,
        )
        decoding_priority = (
            torch.empty_like(token_group_ids, dtype=torch.get_default_dtype()).normal_(generator=generator).abs()
        )
        bounded_priority = decoding_priority / (1 + decoding_priority)
        unit_key = bounded_priority + unit_class.to(decoding_priority.dtype)
        unit_order = torch.argsort(unit_key, dim=1)
        positions = torch.ones_like(token_group_ids).cumsum(dim=1) - 1
        unit_rank = torch.zeros_like(unit_order).scatter(1, unit_order, positions)
        return torch.gather(unit_rank, 1, token_group_ids)

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
        mask = residue_mask.to(node_states.dtype)
        checkpointing = self.gradient_checkpointing and self.training and torch.is_grad_enabled()
        for block in self.decoder_blocks:
            decoder_context = _cat_neighbors_nodes(node_states, sequence_edges, neighbor_indices)
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

    def _decode_unconditional_logits(
        self,
        node_states: Tensor,
        edge_states: Tensor,
        neighbor_indices: Tensor,
        neighbor_valid: Tensor,
        residue_mask: Tensor,
    ) -> Tensor:
        r"""
        Return sequence-free all-equal-rank logits: every row sees no decoded neighbor.
        """

        mask = residue_mask.to(dtype=node_states.dtype)
        encoder_edges = _cat_neighbors_nodes(torch.zeros_like(node_states), edge_states, neighbor_indices)
        encoder_context = _cat_neighbors_nodes(node_states, encoder_edges, neighbor_indices)
        # Gate the filler neighbor slots exactly as the encoder does, so a structure shorter than
        # ``num_neighbors`` never sees a padded neighbor's message; all-ones for a full structure.
        attend = neighbor_valid.to(dtype=node_states.dtype).unsqueeze(-1)
        forward_context = mask[:, :, None, None] * attend * encoder_context
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
    ) -> Tensor:
        r"""
        Incrementally decode one independently selected position per sample.
        """

        def gather_rows(value: Tensor) -> Tensor:
            index = positions.reshape(positions.shape[0], 1, *([1] * (value.ndim - 2)))
            return torch.gather(value, 1, index.expand(-1, 1, *value.shape[2:]))

        edge_slice = gather_rows(edge_states)
        neighbor_slice = gather_rows(neighbor_indices)
        sequence_edge = _cat_neighbors_nodes(token_states, edge_slice, neighbor_slice)
        forward_slice = gather_rows(forward_context)
        backward_slice = gather_rows(backward_mask)
        mask_slice = gather_rows(residue_mask).to(dtype=token_states.dtype) * active[:, None]
        for block_index, block in enumerate(self.decoder_blocks):
            decoder_context = _cat_neighbors_nodes(block_states[block_index], sequence_edge, neighbor_slice)
            node_slice = gather_rows(block_states[block_index])
            updated = block(node_slice, backward_slice * decoder_context + forward_slice, mask_slice)
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
        vocabulary: int,
    ) -> Tensor:
        r"""
        Directed adjacent-token pair contribution for one target member row.

        External context rows use the pre-rank-block sampled tokens, ``self`` stays
        at PAD, and another internal member is evaluated at the shared candidate
        token via the diagonal. Both directions of a mirrored internal pair
        therefore contribute without averaging.
        """

        samples, length = sequence_tokens.shape
        device = sequence_tokens.device
        row_index = member[:, None, None, None, None].expand(-1, 1, vocabulary, length, vocabulary)
        row_bias = torch.gather(pair_logit_bias, 1, row_index).squeeze(1)
        context = sequence_tokens.clone()
        context.scatter_(1, member[:, None], NA_MPNN_PADDING_TOKEN_ID)
        gathered = torch.gather(
            row_bias,
            -1,
            context[:, None, :, None].expand(samples, vocabulary, length, 1),
        ).squeeze(
            -1
        )  # [S, V, L]
        positions = torch.arange(length, device=device)
        internal_others = group_member_mask & (positions[None, :] != member[:, None])
        gathered = gathered.masked_fill(internal_others[:, None, :], 0.0)
        contribution = gathered.sum(dim=-1)
        diagonal = row_bias.diagonal(dim1=1, dim2=3).transpose(1, 2)
        contribution = contribution + (diagonal * internal_others[:, None, :]).sum(dim=-1)
        return contribution


class NaMpnnMessage(nn.Module):
    r"""Compute neighbor messages with NA-MPNN's three-projection GELU network."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.activation = nn.GELU()
        self.hidden_projection = nn.Linear(hidden_size, hidden_size)
        self.output_projection = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        r"""Apply the message network."""

        hidden = self.activation(self.input_projection(x))
        hidden = self.activation(self.hidden_projection(hidden))
        return self.output_projection(hidden)


class NaMpnnFeedForward(nn.Module):
    r"""Apply NA-MPNN's position-wise feed-forward projection."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.input_projection = nn.Linear(hidden_size, intermediate_size)
        self.activation = nn.GELU()
        self.output_projection = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        r"""Apply the feed-forward computation."""

        return self.output_projection(self.activation(self.input_projection(x)))


class NaMpnnRelativePositionEncoding(nn.Module):
    r"""Embed signed residue offsets with one distinct inter-chain bucket."""

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


class NaMpnnEncoderBlock(nn.Module):
    r"""Apply one NA-MPNN node-message, transition, and edge-message update."""

    def __init__(self, hidden_size: int, edge_input_size: int, *, dropout: float, scale: float = 30.0) -> None:
        super().__init__()
        self.scale = scale
        self.node_message = NaMpnnMessage(hidden_size + edge_input_size, hidden_size)
        self.node_dropout = nn.Dropout(dropout)
        self.node_norm = LayerNorm(hidden_size)
        self.transition = NaMpnnFeedForward(hidden_size, hidden_size * 4)
        self.transition_dropout = nn.Dropout(dropout)
        self.transition_norm = LayerNorm(hidden_size)
        self.edge_message = NaMpnnMessage(hidden_size + edge_input_size, hidden_size)
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
        r"""Update aligned node and edge states from one message-passing round."""

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


class NaMpnnDecoderBlock(nn.Module):
    r"""Apply one masked NA-MPNN neighbor-message update and transition."""

    def __init__(self, hidden_size: int, edge_input_size: int, *, dropout: float, scale: float = 30.0) -> None:
        super().__init__()
        self.scale = scale
        self.message = NaMpnnMessage(hidden_size + edge_input_size, hidden_size)
        self.message_dropout = nn.Dropout(dropout)
        self.message_norm = LayerNorm(hidden_size)
        self.transition = NaMpnnFeedForward(hidden_size, hidden_size * 4)
        self.transition_dropout = nn.Dropout(dropout)
        self.transition_norm = LayerNorm(hidden_size)

    def forward(
        self,
        node_states: Tensor,
        edge_context: Tensor,
        node_mask: Tensor | None = None,
        attend_mask: Tensor | None = None,
    ) -> Tensor:
        r"""Update node states from an assembled neighbor context."""

        source_states = node_states.unsqueeze(-2).expand(*([-1] * (node_states.ndim - 1)), edge_context.shape[-2], -1)
        message = self.message(torch.cat((source_states, edge_context), dim=-1))
        if attend_mask is not None:
            message = message * attend_mask.unsqueeze(-1)
        node_states = self.message_norm(node_states + self.message_dropout(message.sum(dim=-2) / self.scale))
        node_states = self.transition_norm(node_states + self.transition_dropout(self.transition(node_states)))
        if node_mask is not None:
            node_states = node_states * node_mask.unsqueeze(-1)
        return node_states


class NaMpnnFeatureEncoder(nn.Module):
    r"""
    Encode protein and nucleic-acid coordinates as an MPNN graph.

    Args:
        config: NA-MPNN architecture configuration.
    """

    def __init__(self, config: NaMpnnConfig) -> None:
        super().__init__()
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
        self.relative_position = NaMpnnRelativePositionEncoding(config.num_positional_embeddings)
        self.edge_projection = nn.Linear(
            config.num_positional_embeddings + config.num_rbf * total_atoms * total_atoms,
            config.feature_size,
            bias=False,
        )
        self.edge_norm = LayerNorm(config.feature_size)
        self.node_projection = nn.Linear(self.num_polymer_types, config.feature_size, bias=False)
        self.node_norm = LayerNorm(config.feature_size)

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
            num_neighbors: Optional nearest-neighbor count for this operation.

        Returns:
            node_features: Node features with shape ``(B, N, C)``.
            edge_features: Neighbor-edge features with shape ``(B, N, K, C)``.
            neighbor_indices: Neighbor indices with shape ``(B, N, K)``.
            neighbor_valid: Real-neighbor mask with shape ``(B, N, K)`` (the encoder's
                ``attend_mask``: 1 for a real neighbor, 0 for a filler slot).
        """

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
        )
        pair_rbf = _all_pair_rbf(all_coordinates, all_mask, neighbor_indices, self.num_rbf)
        neighbor_residue_index = gather_nodes(residue_index.unsqueeze(-1), neighbor_indices).squeeze(-1)
        residue_offset = residue_index.unsqueeze(-1) - neighbor_residue_index
        neighbor_chain_index = gather_nodes(chain_index.unsqueeze(-1), neighbor_indices).squeeze(-1)
        same_chain = (chain_index.unsqueeze(-1) == neighbor_chain_index).long()
        position_features = self.relative_position(residue_offset.long(), same_chain)
        edge_features = self.edge_norm(self.edge_projection(torch.cat((position_features, pair_rbf), dim=-1)))

        polymer_one_hot = F.one_hot(
            polymer_type_index,
            num_classes=self.num_polymer_types,
        ).to(dtype=dtype)
        node_features = self.node_norm(self.node_projection(polymer_one_hot))
        return node_features, edge_features, neighbor_indices, neighbor_valid


def _cat_neighbors_nodes(node_states: Tensor, neighbor_states: Tensor, neighbor_indices: Tensor) -> Tensor:
    r"""Concatenate gathered neighbor nodes onto aligned edge states."""

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
    r"""Build backward and forward neighbor masks from causal ranks."""

    neighbor_ranks = gather_nodes(decoding_ranks.unsqueeze(-1), neighbor_indices).squeeze(-1)
    backward = neighbor_ranks < decoding_ranks.unsqueeze(-1)
    query_mask = residue_mask.unsqueeze(-1).unsqueeze(-1) * neighbor_valid.unsqueeze(-1)
    return query_mask * backward.unsqueeze(-1), query_mask * (~backward).unsqueeze(-1)


def _virtual_na_n(o4prime: Tensor, c1prime: Tensor, c2prime: Tensor) -> Tensor:
    r"""
    Reconstruct the idealized nucleic-acid base nitrogen coordinate.
    """

    b = c1prime - o4prime
    c = c2prime - c1prime
    a = torch.cross(b, c, dim=-1)
    return -0.56967352 * a + 0.51055973 * b - 0.53122153 * c + c1prime


def _all_pair_rbf(coordinates: Tensor, atom_mask: Tensor, neighbor_indices: Tensor, num_bins: int) -> Tensor:
    r"""
    RBF-encode every ordered atom-pair distance for each neighbor edge.
    """

    flat = coordinates.flatten(start_dim=-2)
    neighbor_flat = gather_nodes(flat, neighbor_indices)
    neighbor_coordinates = neighbor_flat.unflatten(-1, (coordinates.shape[-2], coordinates.shape[-1]))
    differences = coordinates[:, :, None, :, None, :] - neighbor_coordinates[:, :, :, None]
    distances = torch.sqrt(torch.sum(differences**2, -1) + 1e-6)
    rbf = _radial_basis(distances, num_bins)
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
        sequence_tokens: Residue token IDs with shape ``(batch_size, num_residues)``.
        residue_mask: Valid-residue mask with shape ``(batch_size, num_residues)``.
        residue_index: Residue indices with shape ``(batch_size, num_residues)``.
        chain_index: Chain indices with shape ``(batch_size, num_residues)``.
        polymer_type_index: Polymer-type IDs with shape
            ``(batch_size, num_residues)``.
        design_positions: Positions eligible for design with shape
            ``(batch_size, num_residues)``.
        token_group_ids: Autoregressive unit ID for each residue row.
        structure: Source structure.
        component_index: Canonical component indices aligned to residue rows.
        tied_weights: Tied-group weights with shape ``(batch_size, num_residues)``.
        residue_logit_bias: Additive bias with shape
            ``(batch_size, num_residues, vocabulary_size)``.
        allowed_token_mask: Token support with shape
            ``(batch_size, num_residues, vocabulary_size)``.
        pair_logit_bias: Optional pair-conditioned bias with shape
            ``(batch_size, num_residues, vocabulary_size, num_residues, vocabulary_size)``.
    """

    atom_coordinates: Tensor
    atom_mask: Tensor
    sequence_tokens: Tensor
    residue_mask: Tensor
    residue_index: Tensor
    chain_index: Tensor
    polymer_type_index: Tensor
    design_positions: Tensor
    token_group_ids: Tensor
    structure: Structure
    component_index: Tensor
    tied_weights: Tensor
    residue_logit_bias: Tensor
    allowed_token_mask: Tensor
    pair_logit_bias: Tensor | None = None


class NaMpnnOutput(_TensorCollection):
    r"""
    Outputs from one NA-MPNN forward pass.

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


class NaMpnnSamplingOutput(_TensorCollection):
    r"""
    Outputs from NA-MPNN autoregressive sampling.

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
