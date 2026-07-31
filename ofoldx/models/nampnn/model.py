r"""
NA-MPNN inverse-folding model for protein, DNA, RNA, and mixed-polymer structures.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from tensordict import tensorclass
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from ...data.components import get_component
from ...data.molecular import MolecularSystem, replace_components
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
    validate_positive_integer,
    virtual_beta_carbon,
)
from ...results import DesignCandidate, DesignResult, Metric, MpnnProbabilityResult, Score, ScoreResult
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
NA_MPNN_INVALID_GENERATION_TOKEN_IDS = (
    NA_MPNN_TOKEN_BY_COMPONENT["UNK"],
    NA_MPNN_TOKEN_BY_COMPONENT["DX"],
    NA_MPNN_TOKEN_BY_COMPONENT["RX"],
    NA_MPNN_TOKEN_BY_COMPONENT["MAS"],
    NA_MPNN_TOKEN_BY_COMPONENT["PAD"],
)
# Contiguous generatable token columns per polymer type. RNA shares the DNA columns
# in the shared-token mode and owns tokens 26..29 in separate-token mode.
NA_MPNN_PROTEIN_GENERATABLE_TOKEN_IDS = tuple(range(0, 20))
NA_MPNN_DNA_GENERATABLE_TOKEN_IDS = tuple(range(21, 25))
NA_MPNN_RNA_SHARED_GENERATABLE_TOKEN_IDS = tuple(range(21, 25))
NA_MPNN_RNA_SEPARATE_GENERATABLE_TOKEN_IDS = tuple(range(26, 30))
_RNA_SHARED_INPUT_TOKENS = frozenset((*NA_MPNN_RNA_SHARED_GENERATABLE_TOKEN_IDS, NA_MPNN_TOKEN_BY_COMPONENT["DX"]))


def _repartition_entities_by_sequence(
    original: MolecularSystem,
    realized: MolecularSystem,
) -> MolecularSystem:
    r"""
    Split copies of each original entity by their realized component sequence.
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

    def component_ids(
        system: MolecularSystem,
        instance: int,
    ) -> tuple[str, ...]:
        start = int(system.instance_component_ptr[instance])
        stop = int(system.instance_component_ptr[instance + 1])
        return tuple(system.symbols[int(symbol)] for symbol in system.component_symbol[start:stop].tolist())

    instances_by_entity: list[list[int]] = [[] for _ in range(original.entity_symbol.numel())]
    for instance, entity in enumerate(original.instance_entity.tolist()):
        instances_by_entity[entity].append(instance)

    used_ids = {original.symbols[int(symbol)] for symbol in original.entity_symbol.tolist()}
    new_entity_symbols: list[int] = []
    new_entity_types: list[int] = []
    new_entity_molecule_types: list[int] = []
    new_instance_entity = torch.empty_like(realized.instance_entity)
    for entity, instances in enumerate(instances_by_entity):
        entity_id = original.symbols[int(original.entity_symbol[entity])]
        if not instances:
            new_entity_symbols.append(intern(entity_id))
            new_entity_types.append(int(original.entity_type[entity]))
            new_entity_molecule_types.append(int(original.entity_molecule_type[entity]))
            continue
        original_signature = component_ids(original, instances[0])
        groups: dict[tuple[str, ...], list[int]] = {}
        for instance in instances:
            groups.setdefault(component_ids(realized, instance), []).append(instance)
        primary = original_signature if original_signature in groups else next(iter(groups))
        ordered = (
            primary,
            *(signature for signature in groups if signature != primary),
        )
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
            output_entity = len(new_entity_symbols)
            new_entity_symbols.append(intern(output_id))
            new_entity_types.append(int(original.entity_type[entity]))
            new_entity_molecule_types.append(int(original.entity_molecule_type[entity]))
            new_instance_entity[groups[signature]] = output_entity

    return realized.replace(  # type: ignore[attr-defined]
        symbols=tuple(symbols),
        system_entity_ptr=realized.system_entity_ptr.new_tensor([0, len(new_entity_symbols)]),
        entity_symbol=realized.entity_symbol.new_tensor(new_entity_symbols),
        entity_type=realized.entity_type.new_tensor(new_entity_types),
        entity_molecule_type=realized.entity_molecule_type.new_tensor(new_entity_molecule_types),
        instance_entity=new_instance_entity,
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

    def __init__(self, config: NaMpnnConfig | None = None) -> None:
        config = NaMpnnConfig() if config is None else config
        super().__init__(config)
        self.gradient_checkpointing = False
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
        r"""
        Apply the Xavier-uniform matrix initialization.
        """

        for parameter in self.parameters():
            if parameter.ndim > 1:
                nn.init.xavier_uniform_(parameter)

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

        def repeat(value: Tensor | None) -> Tensor | None:
            return None if value is None else value.repeat_interleave(count, dim=0)

        return NaMpnnInput(  # type: ignore[call-arg]
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
            structure=input.structure,
            component_index=repeat(input.component_index),
            tied_weights=repeat(input.tied_weights),
            residue_logit_bias=repeat(input.residue_logit_bias),
            allowed_token_mask=repeat(input.allowed_token_mask),
            pair_logit_bias=repeat(input.pair_logit_bias),
            batch_size=[],
        )

    def _probability_result(
        self,
        input: NaMpnnInput,
        logits: Tensor,
        decoding_ranks: Tensor,
    ) -> MpnnProbabilityResult:
        r"""
        Wrap aligned per-evaluation logits with the one target's identity.
        """

        component_indices = input.component_index[0].detach().cpu().tolist()
        return MpnnProbabilityResult(
            target=input.structure,
            component_references=tuple(
                None if component < 0 else input.structure.system.component_reference(component)
                for component in component_indices
            ),
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
        decoding_priority: Tensor | None = None,
    ) -> Tensor:
        r"""
        Draw fixed-first causal ranks directly from aligned causal-unit IDs.

        Every causal unit receives one random priority; members of a unit share its
        priority through ``tied_group_ids`` and therefore share its rank, while
        every distinct unit receives a distinct rank. A unit is generated only when
        all of its members are designed; those units sort after every fixed unit.

        ``decoding_priority`` may provide the per-residue random priorities
        explicitly; otherwise they are drawn from ``generator``. Every tensor is
        derived from ``tied_group_ids`` so ragged batches keep each structure's
        order over its own residues, with no padded rows.
        """

        designed = (residue_mask & design_positions).to(torch.long)
        unit_designed = torch.ones_like(tied_group_ids).scatter_reduce(
            1, tied_group_ids, designed, reduce="amin", include_self=True
        )
        if decoding_priority is None:
            decoding_priority = torch.rand(tied_group_ids.shape, device=tied_group_ids.device, generator=generator)
        unit_key = decoding_priority + 2 * unit_designed.to(decoding_priority.dtype)
        unit_order = torch.argsort(unit_key, dim=1)
        # ``cumsum`` of ones is the per-row 0..L-1 step index without a dense ``arange`` that a ragged batch
        # could not align; the out-of-place scatter inverts the permutation into per-position ranks.
        positions = torch.ones_like(tied_group_ids).cumsum(dim=1) - 1
        unit_rank = torch.zeros_like(unit_order).scatter(1, unit_order, positions)
        return torch.gather(unit_rank, 1, tied_group_ids)

    @staticmethod
    def _shared_ranks(
        residue_mask: Tensor,
        design_positions: Tensor,
        tied_group_ids: Tensor,
        generator: torch.Generator | None,
    ) -> Tensor:
        r"""
        Draw one causal ranking shared across the sampling evaluation axis.

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
        polymer_coordinate_noise: tuple[float, float, float] = (0.0, 0.0, 0.0),
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        r"""
        Run the one coordinate encoder shared by every operation.
        """

        coordinates = input.atom_coordinates
        protein_coordinate_noise, dna_coordinate_noise, rna_coordinate_noise = polymer_coordinate_noise
        if protein_coordinate_noise or dna_coordinate_noise or rna_coordinate_noise:
            scale = (
                input.protein_mask.to(coordinates.dtype) * protein_coordinate_noise
                + input.dna_mask.to(coordinates.dtype) * dna_coordinate_noise
                + input.rna_mask.to(coordinates.dtype) * rna_coordinate_noise
            )
            noise = torch.randn(
                coordinates.shape,
                device=coordinates.device,
                dtype=coordinates.dtype,
                generator=generator,
            )
            coordinates = coordinates + scale[:, :, None, None] * noise
        # ``attend_mask`` comes from the k-NN: the real-neighbor mask, which also zeroes filler slots when a
        # structure is shorter than ``num_neighbors`` (``valid`` there is ``residue_mask``, matching ``mask``).
        node_features, edge_features, neighbor_indices, attend_mask = self.feature_encoder(
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
        checkpointing = self.gradient_checkpointing and self.training and torch.is_grad_enabled()
        for block in self.encoder_blocks:
            if checkpointing:
                node_states, edge_states = checkpoint(
                    block,
                    node_states,
                    edge_states,
                    neighbor_indices,
                    mask,
                    attend_mask,
                    use_reentrant=False,
                )
            else:
                node_states, edge_states = block(node_states, edge_states, neighbor_indices, mask, attend_mask)
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
        mask = residue_mask.to(node_states.dtype)
        checkpointing = self.gradient_checkpointing and self.training and torch.is_grad_enabled()
        for block in self.decoder_blocks:
            decoder_context = cat_neighbors_nodes(node_states, sequence_edges, neighbor_indices)
            context = backward_mask * decoder_context + forward_context
            if checkpointing:
                node_states = checkpoint(
                    block,
                    node_states,
                    context,
                    mask,
                    use_reentrant=False,
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
        encoder_edges = cat_neighbors_nodes(torch.zeros_like(node_states), edge_states, neighbor_indices)
        encoder_context = cat_neighbors_nodes(node_states, encoder_edges, neighbor_indices)
        # Gate the filler neighbor slots exactly as the encoder does, so a structure shorter than
        # ``num_neighbors`` never sees a padded neighbor's message; all-ones for a full structure.
        attend = neighbor_valid.to(dtype=node_states.dtype).unsqueeze(-1)
        forward_context = mask[:, :, None, None] * attend * encoder_context
        states = node_states
        for block in self.decoder_blocks:
            states = block(states, forward_context, mask)
        return self.output_projection(states)

    def encode(
        self,
        input: NaMpnnInput,
        *,
        protein_coordinate_noise: float = 0.0,
        dna_coordinate_noise: float = 0.0,
        rna_coordinate_noise: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        r"""
        Encode a batch of mixed-polymer backbones as residue graph states.

        Args:
            input: NA-MPNN tensors with batch and residue axes.
            protein_coordinate_noise: Gaussian coordinate-noise standard deviation for proteins.
            dna_coordinate_noise: Gaussian coordinate-noise standard deviation for DNA.
            rna_coordinate_noise: Gaussian coordinate-noise standard deviation for RNA.
            generator: Random generator used for coordinate noise.

        Returns:
            Node states with shape ``(B, L, H)``, edge states with shape
            ``(B, L, K, H)``, neighbor indices with shape ``(B, L, K)``, and the real-neighbor
            mask with shape ``(B, L, K)`` (0 for a filler slot padded in when a structure is
            shorter than ``num_neighbors``).
        """

        return self._encode(
            input,
            polymer_coordinate_noise=(
                protein_coordinate_noise,
                dna_coordinate_noise,
                rna_coordinate_noise,
            ),
            generator=generator,
        )

    def forward(
        self,
        input: NaMpnnInput,
        *,
        decoding_ranks: Tensor | None = None,
        decoding_priority: Tensor | None = None,
        decode_protein_first: bool = False,
        protein_coordinate_noise: float = 0.0,
        dna_coordinate_noise: float = 0.0,
        rna_coordinate_noise: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> NaMpnnOutput:
        r"""
        Predict residue logits conditioned on the input sequence and decoding order.

        Args:
            input: NA-MPNN tensors with batch size ``B`` and ``L`` residue rows.
            decoding_ranks: Optional ranks with shape ``(B, L)``; smaller values
                decode earlier.
            decoding_priority: Optional per-residue random priority ``(B, L)``
                used to draw the order. Ignored when ``decoding_ranks`` is given.
            decode_protein_first: Whether generated protein rows precede generated
                nucleic-acid rows when ranks are drawn.
            protein_coordinate_noise: Gaussian coordinate-noise standard deviation for proteins.
            dna_coordinate_noise: Gaussian coordinate-noise standard deviation for DNA.
            rna_coordinate_noise: Gaussian coordinate-noise standard deviation for RNA.
            generator: Random generator used for the order (when no priority is supplied) and coordinate noise.

        Returns:
            Vocabulary logits and the decoding ranks used for the forward pass.
        """

        node_states, edge_states, neighbor_indices, neighbor_valid = self._encode(
            input,
            polymer_coordinate_noise=(
                protein_coordinate_noise,
                dna_coordinate_noise,
                rna_coordinate_noise,
            ),
            generator=generator,
        )
        if decoding_ranks is None:
            decoding_ranks = self._training_ranks(input, decode_protein_first, generator, decoding_priority)
        logits = self._decode_logits(
            node_states,
            edge_states,
            neighbor_indices,
            neighbor_valid,
            input.sequence_tokens,
            input.residue_mask,
            decoding_ranks,
        )
        return NaMpnnOutput(  # type: ignore[call-arg]
            logits=logits,
            decoding_ranks=decoding_ranks,
            batch_size=[logits.shape[0]],
        )

    @staticmethod
    def _training_ranks(
        input: NaMpnnInput,
        decode_protein_first: bool,
        generator: torch.Generator | None,
        decoding_priority: Tensor | None = None,
    ) -> Tensor:
        r"""
        Draw the noised training decode order, protein-first when asked.

        ``decoding_priority`` may provide the per-residue random priorities
        explicitly; otherwise they are drawn from ``generator``. The order is
        inverted without constructing a dense step tensor, so ragged batches keep
        each structure's own residue axis.
        """

        decode_mask = (input.residue_mask & input.design_positions).to(torch.float32)
        if decode_protein_first:
            decode_mask = decode_mask * (~input.protein_mask.bool()).to(torch.float32)
        if decoding_priority is None:
            decoding_priority = torch.rand(
                input.residue_mask.shape, device=input.residue_mask.device, generator=generator
            )
        order = torch.argsort((decode_mask + 1e-4) * decoding_priority, dim=1)
        positions = torch.ones_like(order).cumsum(dim=1) - 1
        ranks = torch.zeros_like(order).scatter(1, order, positions)
        return ranks

    # ------------------------------------------------------------------
    # Probability operations.
    # ------------------------------------------------------------------
    def score(
        self,
        input: NaMpnnInput,
        *,
        num_decoding_orders: int = 1,
        decoding_ranks: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> MpnnProbabilityResult:
        r"""
        Score a sequence with autoregressive probabilities over decoding orders.

        Args:
            input: One processed mixed-polymer target.
            num_decoding_orders: Number of decoding orders to evaluate.
            decoding_ranks: Optional ranks with shape
                ``(num_decoding_orders, num_residues)``.
            generator: Random generator used to draw decoding orders.

        Returns:
            Logits with shape ``(num_decoding_orders, num_residues, vocabulary_size)``
            and decoding ranks with shape ``(num_decoding_orders, num_residues)``.
        """

        self._require_single_target(input)
        num_decoding_orders = validate_positive_integer(num_decoding_orders, "num_decoding_orders")
        repeated = self._repeat_input(input, num_decoding_orders)
        node_states, edge_states, neighbor_indices, neighbor_valid = self._encode(repeated)
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
            neighbor_valid,
            repeated.sequence_tokens,
            repeated.residue_mask,
            decoding_ranks,
        )
        return self._probability_result(input, logits, decoding_ranks)

    def unconditional_probabilities(
        self,
        input: NaMpnnInput,
        *,
        num_decoding_orders: int = 1,
    ) -> MpnnProbabilityResult:
        r"""
        Evaluate sequence-free probabilities without decoded neighbors.

        Args:
            input: One processed mixed-polymer target.
            num_decoding_orders: Number of identical evaluation rows to return.

        Returns:
            Logits with shape ``(num_decoding_orders, num_residues, vocabulary_size)``
            and all-zero ranks with shape ``(num_decoding_orders, num_residues)``.
        """

        self._require_single_target(input)
        num_decoding_orders = validate_positive_integer(num_decoding_orders, "num_decoding_orders")
        repeated = self._repeat_input(input, num_decoding_orders)
        node_states, edge_states, neighbor_indices, neighbor_valid = self._encode(repeated)
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
        r"""
        Sample mixed-polymer sequences under the processed constraints.

        Args:
            input: One processed mixed-polymer target.
            num_samples: Number of sequences to sample.
            temperature: Positive sampling temperature.
            decoding_ranks: Optional ranks with shape ``(num_samples, num_residues)``. The sampler
                decodes one causal order shared across the whole sample axis, so every row must be
                identical; distinct per-sample orders are rejected with ``ValueError``.
            global_logit_bias: Optional bias with shape ``(vocabulary_size,)``.
            generator: Random generator used for orders and token sampling.

        Returns:
            Sampled token IDs, model log probabilities, sampling probabilities,
            and decoding ranks, each with a leading ``num_samples`` axis.
        """

        self._require_single_target(input)
        num_samples = validate_positive_integer(num_samples, "num_samples")
        temperature = float(temperature)
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be a finite positive number")

        repeated = self._repeat_input(input, num_samples)
        device = repeated.atom_coordinates.device
        length = repeated.sequence_tokens.shape[1]
        vocabulary = self.vocab_size

        node_states, edge_states, neighbor_indices, neighbor_valid = self._encode(repeated)
        if decoding_ranks is None:
            ranks = self._shared_ranks(
                repeated.residue_mask,
                repeated.design_positions,
                repeated.tied_group_ids,
                generator,
            )
        else:
            ranks = decoding_ranks
            # The incremental sampler decodes one causal order (``ranks[0]``) shared across the sample axis,
            # so tied members stay simultaneous and only the multinomial draw differs per sample. Distinct
            # per-sample rows would silently corrupt every sample past the first; reject them rather than
            # honor an order the loop cannot follow.
            if not torch.equal(ranks, ranks[:1].expand_as(ranks)):
                raise ValueError("NaMpnn.sample decodes one shared order; decoding_ranks rows must be identical")
        hard_support = None if repeated.allowed_token_mask is None else repeated.allowed_token_mask.bool()

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

        sequence_tokens = torch.full((num_samples, length), NA_MPNN_PADDING_TOKEN_ID, dtype=torch.long, device=device)
        log_probabilities = node_states.new_zeros(num_samples, length, vocabulary)
        sampling_probabilities = torch.zeros_like(log_probabilities)

        designed = (input.design_positions[0] & input.residue_mask[0]).tolist()
        valid = input.residue_mask[0].tolist()
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
            # A unit that mixes designed and fixed members is fixed to its anchor identity: the fixed
            # members pin the whole unit's token, so every member takes the anchor rather than keeping
            # its own input token (the processor rejects two fixed members with different identities).
            anchor_members = [member for member in members if valid[member] and not designed[member]]
            anchor = repeated.sequence_tokens[:, anchor_members[0]] if anchor_members and not generated_unit else None
            if generated_unit:
                drawn = torch.multinomial(probabilities, 1, generator=generator)[:, 0]
            else:
                drawn = None
            for member in members:
                raw_log = F.log_softmax(member_neural[member], dim=-1)
                keep = float(designed[member])
                if generated_unit and drawn is not None:
                    token = drawn
                elif anchor is not None:
                    token = anchor
                else:
                    token = repeated.sequence_tokens[:, member]
                sequence_tokens[:, member] = token
                log_probabilities[:, member] = keep * raw_log
                sampling_probabilities[:, member] = keep * probabilities
                token_states[:, member] = self.token_embedding(token)

        return NaMpnnSamplingOutput(  # type: ignore[call-arg]
            sequence_tokens=sequence_tokens,
            log_probabilities=log_probabilities,
            sampling_probabilities=sampling_probabilities,
            decoding_ranks=ranks,
            batch_size=[num_samples],
        )

    @staticmethod
    def _ordered_units(tied_group_ids: Tensor, ranks: Tensor) -> list[list[int]]:
        r"""
        Return the shared causal units as ascending-rank member lists.
        """

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
        r"""
        Incrementally decode one position across the whole evaluation axis.
        """

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
        r"""
        Directed adjacent-token pair contribution for one target member row.

        External context rows use the pre-rank-block sampled tokens, ``self`` stays
        at PAD, and another internal member is evaluated at the shared candidate
        token via the diagonal. Both directions of a mirrored internal pair
        therefore contribute without averaging.
        """

        samples, length = sequence_tokens.shape
        device = sequence_tokens.device
        row_bias = pair_bias[:, member]  # [S, V, L, V]
        context = sequence_tokens.clone()
        context[:, member] = NA_MPNN_PADDING_TOKEN_ID
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
        component_index: Tensor,
        polymer_type_index: Tensor,
        sampled_tokens: list[int],
        design_positions: list[bool],
    ) -> MolecularSystem:
        r"""
        Realize one candidate while preserving entity and bond identity.
        """

        selected_components = []
        replacements = []
        for component, polymer_type, token, selected in zip(
            component_index.detach().cpu().tolist(),
            polymer_type_index.detach().cpu().tolist(),
            sampled_tokens,
            design_positions,
            strict=True,
        ):
            if component < 0 or not selected:
                continue
            component_id = _component_from_token(int(token), int(polymer_type))
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
                dtype=torch.int64,
                device=system.component_symbol.device,
            ),
            replacements,
        )
        return _repartition_entities_by_sequence(system, realized)

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
        r"""
        Design protein, DNA, and RNA sequence candidates on a fixed backbone.

        Args:
            input: One processed mixed-polymer target.
            num_candidates: Number of sequence candidates to sample.
            temperature: Positive sampling temperature.
            global_logit_bias: Optional bias with shape ``(vocabulary_size,)``.
            generator: Random generator used for orders and token sampling.

        Returns:
            Designed molecular systems and per-candidate confidence scores in
            sampling order.
        """

        self._require_single_target(input)
        num_candidates = validate_positive_integer(num_candidates, "num_candidates")
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

        component_indices = input.component_index[0].detach().cpu().tolist()
        references = tuple(
            None if component < 0 else input.structure.system.component_reference(component)
            for component in component_indices
        )
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
                input.structure.system,
                input.component_index[0],
                input.polymer_type_index[0],
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
        r"""
        Estimate nucleic-acid base preferences with protein identities fixed.

        Args:
            input: One processed mixed-polymer target.
            num_samples: Number of sampled sequence contexts to average.
            temperature: Positive sampling temperature.
            global_logit_bias: Optional bias with shape ``(vocabulary_size,)``.
            generator: Random generator used for orders and token sampling.

        Returns:
            Per-residue A, C, G, and T/U probability metrics for selected
            nucleic-acid positions.
        """

        self._require_single_target(input)
        na_design = input.design_positions & ~input.protein_mask
        fixed = NaMpnnInput(  # type: ignore[call-arg]
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
            structure=input.structure,
            component_index=input.component_index,
            tied_weights=input.tied_weights,
            residue_logit_bias=input.residue_logit_bias,
            allowed_token_mask=input.allowed_token_mask,
            pair_logit_bias=input.pair_logit_bias,
            batch_size=[],
        )
        sampled = self.sample(
            fixed,
            num_samples=num_samples,
            temperature=temperature,
            global_logit_bias=global_logit_bias,
            generator=generator,
        )
        mean_probabilities = sampled.sampling_probabilities.to(torch.float64).mean(dim=0).cpu().numpy()

        component_indices = input.component_index[0].detach().cpu().tolist()
        references = tuple(
            None if component < 0 else input.structure.system.component_reference(component)
            for component in component_indices
        )
        polymer_types = input.polymer_type_index[0].detach().cpu().tolist()
        selected = (na_design[0] & input.residue_mask[0]).tolist()
        na_rows = [
            (index, reference)
            for index, (reference, polymer, keep) in enumerate(zip(references, polymer_types, selected))
            if reference is not None and keep and polymer in (1, 2)
        ]
        axis = tuple(reference for _index, reference in na_rows)
        token_columns = {
            row: (
                NA_MPNN_DNA_GENERATABLE_TOKEN_IDS
                if polymer_types[row] == 1
                else (
                    NA_MPNN_RNA_SHARED_GENERATABLE_TOKEN_IDS
                    if int(input.sequence_tokens[0, row]) in _RNA_SHARED_INPUT_TOKENS
                    else NA_MPNN_RNA_SEPARATE_GENERATABLE_TOKEN_IDS
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
        return ScoreResult(target=input.structure, scores={}, metrics=metrics)


class NaMpnnFeatures(nn.Module):
    r"""
    Encode protein and nucleic-acid coordinates as an MPNN graph.

    Args:
        config: NA-MPNN architecture configuration.
    """

    def __init__(self, config: NaMpnnConfig) -> None:
        super().__init__()
        self.num_neighbors = config.num_neighbors
        self.num_rbf = config.num_rbf
        self.num_polymer_types = config.num_polymer_types
        self.include_predicted_na_n = config.include_predicted_na_n
        self.nitrogen_index = NA_MPNN_ATOM_INDEX["N"]
        self.alpha_carbon_index = NA_MPNN_ATOM_INDEX["CA"]
        self.carbonyl_carbon_index = NA_MPNN_ATOM_INDEX["C"]
        self.oxygen_4_prime_index = NA_MPNN_ATOM_INDEX["O4'"]
        self.carbon_1_prime_index = NA_MPNN_ATOM_INDEX["C1'"]
        self.carbon_2_prime_index = NA_MPNN_ATOM_INDEX["C2'"]
        total_atoms = config.num_atom_types + 1 + int(self.include_predicted_na_n)
        self.relative_position = MpnnRelativePositionEncoding(config.num_positional_embeddings)
        self.edge_projection = nn.Linear(
            config.num_positional_embeddings + config.num_rbf * total_atoms * total_atoms,
            config.feature_size,
            bias=False,
        )
        self.edge_norm = LayerNorm(config.feature_size)
        self.node_projection = nn.Linear(config.num_polymer_types, config.feature_size, bias=False)
        self.node_norm = LayerNorm(config.feature_size)

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
            protein_mask: Protein-residue mask with shape ``(B, N)``.
            dna_mask: DNA-residue mask with shape ``(B, N)``.
            rna_mask: RNA-residue mask with shape ``(B, N)``.

        Returns:
            Node features ``(B, N, C)``, neighbor-edge features ``(B, N, K, C)``,
            neighbor indices ``(B, N, K)``, and the real-neighbor mask ``(B, N, K)`` (the encoder's
            ``attend_mask``: 1 for a real neighbor, 0 for a filler slot).
        """

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

        _, neighbor_indices, neighbor_valid = nearest_neighbors(
            alpha_carbon + reference,
            valid,
            self.num_neighbors,
        )
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
        return node_features, edge_features, neighbor_indices, neighbor_valid


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


def _component_from_token(token: int, polymer_type: int) -> str:
    r"""
    Realize one sampled token as a canonical component under its polymer type.
    """

    if polymer_type == 0:
        if token >= 20:
            raise RuntimeError(f"NA-MPNN sampled non-protein token {token} for a protein residue")
        return NA_MPNN_COMPONENT_IDS[token]
    if polymer_type == 1:
        if not 21 <= token <= 24:
            raise RuntimeError(f"NA-MPNN sampled non-DNA token {token} for a DNA residue")
        return ("DA", "DC", "DG", "DT")[token - 21]
    if polymer_type == 2 and 21 <= token <= 24:
        return ("A", "C", "G", "U")[token - 21]
    if polymer_type == 2 and 26 <= token <= 29:
        return ("A", "C", "G", "U")[token - 26]
    raise RuntimeError(f"NA-MPNN sampled non-RNA token {token} for an RNA residue")


# ---------------------------------------------------------------------------
# NA-MPNN typed inputs and outputs.
# ---------------------------------------------------------------------------


@tensorclass
class NaMpnnInput:
    r"""
    Inputs for NA-MPNN sequence design and scoring.

    Args:
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
        protein_mask: Protein-residue mask with shape ``(batch_size, num_residues)``.
        dna_mask: DNA-residue mask with shape ``(batch_size, num_residues)``.
        rna_mask: RNA-residue mask with shape ``(batch_size, num_residues)``.
        design_positions: Positions eligible for design with shape
            ``(batch_size, num_residues)``.
        tied_group_ids: Autoregressive unit ID for each residue row.
        structure: Source structure.
        component_index: Canonical component indices aligned to residue rows.
        tied_weights: Optional tied-group weights with shape ``(batch_size, num_residues)``.
        residue_logit_bias: Optional additive bias with shape
            ``(batch_size, num_residues, vocabulary_size)``.
        allowed_token_mask: Optional token mask with shape
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
    protein_mask: Tensor
    dna_mask: Tensor
    rna_mask: Tensor
    design_positions: Tensor
    tied_group_ids: Tensor
    structure: Structure
    component_index: Tensor
    tied_weights: Tensor | None = None
    residue_logit_bias: Tensor | None = None
    allowed_token_mask: Tensor | None = None
    pair_logit_bias: Tensor | None = None


@tensorclass
class NaMpnnOutput:
    r"""
    Outputs from one NA-MPNN forward pass.

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
class NaMpnnSamplingOutput:
    r"""
    Outputs from NA-MPNN autoregressive sampling.

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
