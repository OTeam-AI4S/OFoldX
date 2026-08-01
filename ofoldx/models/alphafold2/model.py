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
AlphaFold 2 top-level model.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Callable
from dataclasses import KW_ONLY, dataclass
from typing import NamedTuple, Protocol, TypeVar, cast

import numpy as np
import torch
import torch.nn.functional as F
from multimolecule.graph import scatter
from tensordict import TensorClass, tensorclass
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from ...data import residue_constants
from ...errors import RequestRef, fail
from ...metrics import predicted_tm_score, tm_weight_from_pae
from ...model import Model, OptionalTensor, TensorCollection
from ...modules.attention import (
    MultiheadAttention,
    MultiheadAttentionProjection,
    TriangleAttention,
)
from ...modules.backend import LayerNorm
from ...modules.dropout import StructuredDropout
from ...modules.evoformer import (
    EvoformerBlock,
    EvoformerPairBlock,
    EvoformerStack,
    ExtraMsaTrackBlock,
    OuterProductFirstEvoformerBlock,
)
from ...modules.feed_forward import FeedForward
from ...modules.functional import coordinate_distogram
from ...modules.geometry import (
    apply_inverse_rigid_transform,
    apply_rigid_transform,
    dihedral_angle,
    frames_to_atom14_positions,
    normalize_vectors,
    quaternion_to_rotation,
    rotate_vectors,
    rotation_from_two_vectors,
    smoothed_vector_norm,
    torsion_angles_to_frames,
    unit_quaternion_to_rotation,
)
from ...modules.head import DistogramHead
from ...modules.pair import TriangleMultiplication
from ...modules.transition import FeedForwardTransition
from ..processing import (
    project_model_categories,
    project_model_coordinates,
    project_model_counts,
    project_model_features,
)
from .config import (
    AlphaFold2Config,
    AlphaFold2EvoformerConfig,
    AlphaFold2InitialGuessConfig,
    AlphaFold2MultimerConfig,
)
from .vocabulary import AMINO_ACIDS, GAP, MASK, NUM_CLASSES, SOURCE_MSA_ALPHABET


class _AlphaFold2TrunkModel(Model):
    r"""Share MSA realization and recycled representations through the actual AF2 components."""

    input_embeddings: AlphaFold2InputEmbeddings | AlphaFold2MultimerInputEmbeddings
    recycling_embeddings: AlphaFold2RecyclingEmbeddings
    extra_msa_projection: nn.Linear
    extra_msa_stack: EvoformerStack
    template_msa_embedder: AlphaFold2TemplateMsaEmbedder | AlphaFold2MultimerTemplateMsaEmbedder | None
    evoformer_stack: EvoformerStack
    single_projection: nn.Linear
    template_embedder: AlphaFold2TemplateEmbedder | AlphaFold2MultimerTemplateEmbedder | None

    def _realize_msa_features(
        self,
        msa_tokens: Tensor,
        msa_deletion_counts: Tensor,
        msa_profile: Tensor,
        *,
        residue_mask: Tensor,
        generator: torch.Generator | None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> _MsaRealization:
        raise NotImplementedError

    def _realize_msa(self, input: _AlphaFold2InputT, *, generator: torch.Generator | None) -> _AlphaFold2InputT:
        r"""
        Realize one MSA-feature ensemble slice from the raw pool.

        Runs MSA sampling, masking, nearest-neighbor clustering, cluster
        summarization, extra-MSA cropping, and feature assembly over the raw
        alignment tensors supplied by the shared realization boundary. Every draw
        consumes the per-sample ``generator`` so the slice is reproducible from the
        fold seed.
        """

        device = input.token_features.device
        dtype = input.token_features.dtype
        realized_batch = [
            self._realize_msa_features(
                msa_tokens,
                msa_deletion_counts,
                msa_profile,
                residue_mask=residue_mask,
                generator=generator,
                device=device,
                dtype=dtype,
            )
            for msa_tokens, msa_deletion_counts, msa_profile, residue_mask in zip(
                input.msa_tokens,
                input.msa_deletion_counts,
                input.msa_profile,
                input.residue_mask,
                strict=True,
            )
        ]
        msa_features, msa_mask, extra_msa_features, extra_msa_mask, bert_mask, true_msa = (
            torch.stack(values) for values in zip(*realized_batch, strict=True)
        )
        realized = cast(_AlphaFold2InputT, cast(TensorClass, input).clone(recurse=False))
        realized.msa_features = msa_features
        realized.msa_mask = msa_mask
        realized.extra_msa_features = extra_msa_features
        realized.extra_msa_mask = extra_msa_mask
        realized.bert_mask = bert_mask
        realized.true_msa = true_msa
        return realized

    def _embed_and_refine(
        self,
        input: _AlphaFold2RecyclingInput,
        *,
        previous_positions: Tensor,
        previous_msa_first_row: Tensor,
        previous_pair: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        msa, pair, msa_mask, pair_mask = self.input_embeddings(input)
        msa, pair = self.recycling_embeddings(
            msa,
            pair,
            input=input,
            previous_positions=previous_positions,
            previous_msa_first_row=previous_msa_first_row,
            previous_pair=previous_pair,
        )
        pair = pair * pair_mask.unsqueeze(-1).to(dtype=pair.dtype)
        pair = self._add_template_embedding(pair, input, pair_mask)
        extra_msa = self.extra_msa_projection(input.extra_msa_features)
        extra_msa_mask = input.extra_msa_mask
        extra_msa, pair = self.extra_msa_stack(
            extra_msa,
            pair,
            msa_mask=extra_msa_mask,
            pair_mask=pair_mask,
        )
        if self.template_msa_embedder is not None:
            template_activations, template_mask = self.template_msa_embedder(input)
            msa = torch.cat([msa, template_activations], dim=-3)
            msa_mask = torch.cat([msa_mask, template_mask], dim=-2)
        msa, pair = self.evoformer_stack(
            msa,
            pair,
            msa_mask=msa_mask,
            pair_mask=pair_mask,
        )
        msa_first_row = msa.select(dim=-3, index=0)
        single = self.single_projection(msa_first_row)
        return msa, single, pair, msa_first_row

    def _initial_recycling(self, input: _AlphaFold2RecyclingInput) -> tuple[Tensor, Tensor, Tensor]:
        token_shape = input.residue_mask.shape
        num_tokens = token_shape[-1]
        dtype = input.token_features.dtype
        device = input.token_features.device
        return (
            torch.zeros(*token_shape, residue_constants.NUM_ATOM_TYPES, 3, device=device, dtype=dtype),
            torch.zeros(
                *token_shape,
                self.input_embeddings.msa_projection.out_features,
                device=device,
                dtype=dtype,
            ),
            torch.zeros(
                *token_shape[:-1],
                num_tokens,
                num_tokens,
                self.input_embeddings.pair_left_projection.out_features,
                device=device,
                dtype=dtype,
            ),
        )

    def _add_template_embedding(self, pair: Tensor, input: _AlphaFold2RecyclingInput, pair_mask: Tensor) -> Tensor:
        if self.template_embedder is None:
            return pair
        return pair + self.template_embedder(pair, input, pair_mask=pair_mask)


class AlphaFold2Model(_AlphaFold2TrunkModel):
    r"""
    Predict monomeric protein structures with AlphaFold 2.

    Accepts [`AlphaFold2Input`][ofoldx.models.alphafold2.model.AlphaFold2Input] and returns
    [`AlphaFold2Output`][ofoldx.models.alphafold2.model.AlphaFold2Output].

    Args:
        config: Model architecture and inference configuration.
    """

    config_class = AlphaFold2Config
    config: AlphaFold2Config

    def __init__(self, config: AlphaFold2Config | None = None) -> None:
        config = AlphaFold2Config() if config is None else config
        super().__init__(config)
        input_config = config.input_embeddings
        recycling_config = config.recycling
        template_config = config.template
        evoformer_config = config.evoformer
        block_config = evoformer_config.block
        structure_module_config = config.structure_module
        heads = config.heads

        self._num_recycles = operator.index(config.num_recycles)
        if self._num_recycles < 1:
            fail("value_invalid", RequestRef("num_recycles", str(self._num_recycles)))
        self.num_ensembles = operator.index(config.num_ensembles)
        if self.num_ensembles < 1:
            fail("value_invalid", RequestRef("num_ensembles", str(self.num_ensembles)))
        self._resample_msa_in_recycling = config.msa.resample_msa_in_recycling
        self.input_embeddings = AlphaFold2InputEmbeddings(
            token_feature_size=input_config.token_feature_size,
            msa_size=evoformer_config.msa_size,
            pair_size=evoformer_config.pair_size,
            max_relative_feature=input_config.max_relative_feature,
        )
        self.recycling_embeddings = AlphaFold2RecyclingEmbeddings(
            msa_size=evoformer_config.msa_size,
            pair_size=evoformer_config.pair_size,
            recycle_positions=recycling_config.recycle_positions,
            recycle_features=recycling_config.recycle_features,
            previous_positions_min_distance=recycling_config.previous_positions.min_distance,
            previous_positions_max_distance=recycling_config.previous_positions.max_distance,
            num_previous_position_distogram_bins=recycling_config.previous_positions.num_distogram_bins,
        )
        self.template_embedder: AlphaFold2TemplateEmbedder | None = None
        if template_config.enabled:
            self.template_embedder = AlphaFold2TemplateEmbedder(
                query_size=evoformer_config.pair_size,
                template_size=template_config.template_pair_stack.pair_size,
                num_blocks=template_config.template_pair_stack.num_blocks,
                num_distogram_bins=template_config.num_distogram_bins,
                min_distance=template_config.min_distance,
                max_distance=template_config.max_distance,
                vocab_size=template_config.vocab_size,
                use_template_unit_vector=template_config.use_template_unit_vector,
                triangle_multiplication_hidden_size=(
                    template_config.template_pair_stack.triangle_multiplication_outgoing.hidden_size
                ),
                triangle_attention_head_size=(
                    template_config.template_pair_stack.triangle_attention_starting_node.head_size
                ),
                num_triangle_attention_heads=(
                    template_config.template_pair_stack.triangle_attention_starting_node.num_attention_heads
                ),
                transition_expansion=template_config.template_pair_stack.transition.expansion,
                dropout=template_config.template_pair_stack.triangle_attention_starting_node.dropout,
                num_attention_heads=template_config.attention.num_attention_heads,
                backend="auto",
            )
        self.extra_msa_projection = nn.Linear(
            NUM_CLASSES + 2,
            evoformer_config.extra_msa_size,
        )
        evoformer_block_type = (
            OuterProductFirstEvoformerBlock if block_config.outer_product_mean.first else EvoformerBlock
        )
        self.extra_msa_stack = EvoformerStack(
            block=evoformer_block_type,
            msa_block=ExtraMsaTrackBlock,
            num_blocks=evoformer_config.num_extra_msa_blocks,
            msa_size=evoformer_config.extra_msa_size,
            pair_size=evoformer_config.pair_size,
            outer_product_hidden_size=block_config.outer_product_mean.hidden_size,
            num_msa_row_attention_heads=block_config.msa_row_attention_with_pair_bias.num_attention_heads,
            num_msa_column_attention_heads=block_config.msa_row_attention_with_pair_bias.num_attention_heads,
            triangle_multiplication_outgoing_hidden_size=(block_config.triangle_multiplication_outgoing.hidden_size),
            triangle_multiplication_incoming_hidden_size=(block_config.triangle_multiplication_outgoing.hidden_size),
            triangle_attention_head_size=block_config.triangle_attention_starting_node.head_size,
            num_triangle_attention_starting_node_heads=(
                block_config.triangle_attention_starting_node.num_attention_heads
            ),
            num_triangle_attention_ending_node_heads=(
                block_config.triangle_attention_starting_node.num_attention_heads
            ),
            msa_transition_expansion=block_config.msa_transition.expansion,
            pair_transition_expansion=block_config.pair_transition.expansion,
            msa_dropout=block_config.msa_row_attention_with_pair_bias.dropout,
            pair_dropout=block_config.triangle_attention_starting_node.dropout,
            backend="auto",
        )
        self.template_msa_embedder: AlphaFold2TemplateMsaEmbedder | None = None
        if template_config.enabled and template_config.embed_torsion_angles:
            self.template_msa_embedder = AlphaFold2TemplateMsaEmbedder(
                vocab_size=template_config.vocab_size,
                msa_size=evoformer_config.msa_size,
            )
        self.evoformer_stack = EvoformerStack(
            block=evoformer_block_type,
            num_blocks=evoformer_config.num_blocks,
            msa_size=evoformer_config.msa_size,
            pair_size=evoformer_config.pair_size,
            outer_product_hidden_size=block_config.outer_product_mean.hidden_size,
            num_msa_row_attention_heads=block_config.msa_row_attention_with_pair_bias.num_attention_heads,
            num_msa_column_attention_heads=block_config.msa_row_attention_with_pair_bias.num_attention_heads,
            triangle_multiplication_outgoing_hidden_size=(block_config.triangle_multiplication_outgoing.hidden_size),
            triangle_multiplication_incoming_hidden_size=(block_config.triangle_multiplication_outgoing.hidden_size),
            triangle_attention_head_size=block_config.triangle_attention_starting_node.head_size,
            num_triangle_attention_starting_node_heads=(
                block_config.triangle_attention_starting_node.num_attention_heads
            ),
            num_triangle_attention_ending_node_heads=(
                block_config.triangle_attention_starting_node.num_attention_heads
            ),
            msa_transition_expansion=block_config.msa_transition.expansion,
            pair_transition_expansion=block_config.pair_transition.expansion,
            msa_dropout=block_config.msa_row_attention_with_pair_bias.dropout,
            pair_dropout=block_config.triangle_attention_starting_node.dropout,
            backend="auto",
        )
        self.single_projection = nn.Linear(evoformer_config.msa_size, evoformer_config.single_size)
        self.structure_module = AlphaFold2StructureModule(
            single_size=evoformer_config.single_size,
            pair_size=evoformer_config.pair_size,
            num_blocks=structure_module_config.num_blocks,
            num_transition_projections=structure_module_config.num_transition_projections,
            hidden_size=structure_module_config.hidden_size,
            num_attention_heads=structure_module_config.num_attention_heads,
            scalar_query_key_size=structure_module_config.scalar_query_key_size,
            scalar_value_size=structure_module_config.scalar_value_size,
            num_point_qk=structure_module_config.num_point_qk,
            num_point_v=structure_module_config.num_point_v,
            dropout=structure_module_config.dropout,
            position_scale=structure_module_config.position_scale,
            sidechain_hidden_size=structure_module_config.sidechain.hidden_size,
            num_sidechain_blocks=structure_module_config.sidechain.num_blocks,
        )
        self.distogram_head = DistogramHead(
            pair_size=evoformer_config.pair_size,
            num_bins=heads.distogram.num_distogram_bins,
        )
        nn.init.zeros_(self.distogram_head.projection.weight)
        nn.init.zeros_(self.distogram_head.projection.bias)
        self.predicted_lddt_head = AlphaFold2PredictedLddtHead(
            input_size=structure_module_config.hidden_size,
            hidden_size=heads.predicted_lddt.hidden_size,
            num_bins=heads.predicted_lddt.num_plddt_bins,
        )
        self.predicted_aligned_error_head: nn.Linear | None = None
        if heads.predicted_aligned_error.enabled:
            self.predicted_aligned_error_head = nn.Linear(
                evoformer_config.pair_size,
                heads.predicted_aligned_error.num_pae_bins,
            )
            nn.init.zeros_(self.predicted_aligned_error_head.weight)
            nn.init.zeros_(self.predicted_aligned_error_head.bias)
        self.experimentally_resolved_head = nn.Linear(
            evoformer_config.single_size,
            heads.experimentally_resolved.num_atom_types,
        )
        nn.init.zeros_(self.experimentally_resolved_head.weight)
        nn.init.zeros_(self.experimentally_resolved_head.bias)
        self.masked_msa_head = nn.Linear(
            evoformer_config.msa_size,
            len(SOURCE_MSA_ALPHABET) + 1,
        )
        nn.init.zeros_(self.masked_msa_head.weight)
        nn.init.zeros_(self.masked_msa_head.bias)
        self.post_init()

    def forward(
        self,
        input: AlphaFold2Input,
        *,
        num_recycles: int | None = None,
        generator: torch.Generator | None = None,
    ) -> AlphaFold2Output:
        r"""
        Run one AlphaFold 2 structure-prediction pass.

        The input tensors use batch and residue axes ``(B, N)``. When
        ``input.msa_features`` is absent, ``generator`` controls MSA row sampling,
        masking, clustering, and extra-MSA cropping. Precomputed MSA features do
        not consume this generator. Dropout follows the module's training mode and
        uses PyTorch's ambient random state.

        Args:
            input: Monomer features aligned to ``B`` systems and ``N``
                residues.
            num_recycles: Total network passes, including the final pass.
                ``None`` uses the configured value.
            generator: Optional generator for stochastic MSA feature construction.

        Returns:
            Predictions with atom37 coordinates in angstroms and tensor axes
                described by [`AlphaFold2Output`][].

        Raises:
            ValueError: If ``num_recycles`` is not positive.
        """
        input = normalize_alphafold2_input(input, dtype=self.input_embeddings.token_projection.weight.dtype)

        num_recycles = operator.index(self._num_recycles if num_recycles is None else num_recycles)
        if num_recycles < 1:
            fail("value_invalid", RequestRef("num_recycles", str(num_recycles)))

        msa, single, pair, structure, realized = self._run_recycling(
            input, num_recycles=num_recycles, generator=generator
        )
        assert realized.msa_features is not None
        num_msa_rows = realized.msa_features.shape[-3]
        distogram_logits = self.distogram_head(pair)
        plddt_logits = self.predicted_lddt_head(structure.structure_module)
        pae_logits = None if self.predicted_aligned_error_head is None else self.predicted_aligned_error_head(pair)
        resolved_logits = self.experimentally_resolved_head(single)
        masked_msa_logits = self.masked_msa_head(msa.narrow(-3, 0, num_msa_rows))
        optional_output = {
            name: value
            for name, value in (
                ("true_msa", realized.true_msa),
                ("bert_mask", realized.bert_mask),
                ("pae_logits", pae_logits),
            )
            if value is not None
        }
        return AlphaFold2Output(
            single=single,
            pair=pair,
            frames=structure.traj.movedim(0, 1),
            atom14_positions=structure.atom14_positions.movedim(0, 1),
            sidechain_frames=structure.sidechain_frames.movedim(0, 1),
            final_atom_positions=structure.final_atom_positions,
            final_atom_mask=structure.final_atom_mask,
            final_atom14_positions=structure.final_atom14_positions,
            final_atom14_mask=structure.final_atom14_mask,
            sidechain_angles=structure.sidechain_angles.movedim(0, 1),
            sidechain_unnormalized_angles=structure.sidechain_unnormalized_angles.movedim(0, 1),
            distogram_logits=distogram_logits,
            masked_msa_logits=masked_msa_logits,
            plddt_logits=plddt_logits,
            resolved_logits=resolved_logits,
            num_recycles=torch.full((single.shape[0],), num_recycles, dtype=torch.int64, device=single.device),
            batch_size=[single.shape[0]],
            **optional_output,
        )

    def _run_recycling(
        self,
        input: _AlphaFold2InputT,
        *,
        num_recycles: int,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, AlphaFold2StructureModuleOutput, _AlphaFold2InputT]:
        needs_realize = input.msa_features is None
        resample = needs_realize and self._resample_msa_in_recycling
        fixed_ensemble: tuple[_AlphaFold2InputT, ...] | None = None
        if needs_realize and not resample:
            fixed_ensemble = tuple(self._realize_msa(input, generator=generator) for _ in range(self.num_ensembles))
        previous_positions, previous_msa_first_row, previous_pair = self._initial_recycling(input)
        for cycle in range(num_recycles):
            if needs_realize:
                if resample:
                    cycle_inputs = tuple(
                        self._realize_msa(input, generator=generator) for _ in range(self.num_ensembles)
                    )
                else:
                    assert fixed_ensemble is not None
                    cycle_inputs = fixed_ensemble
            else:
                cycle_inputs = (input,) * self.num_ensembles
            final_cycle = cycle == num_recycles - 1
            grad_enabled = torch.is_grad_enabled() and final_cycle
            with torch.set_grad_enabled(grad_enabled):
                msa, single, pair, msa_first_row, cycle_input = self._run_ensemble(
                    cycle_inputs,
                    previous_positions=previous_positions,
                    previous_msa_first_row=previous_msa_first_row,
                    previous_pair=previous_pair,
                )
                structure = self.structure_module(single, pair, cycle_input)
            if final_cycle:
                break
            previous_positions = structure.final_atom_positions.detach()
            previous_msa_first_row = msa_first_row.detach()
            previous_pair = pair.detach()
        return msa, single, pair, structure, cycle_input

    def _run_ensemble(
        self,
        inputs: tuple[_AlphaFold2InputT, ...],
        *,
        previous_positions: Tensor,
        previous_msa_first_row: Tensor,
        previous_pair: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, _AlphaFold2InputT]:
        selected_msa = None
        selected_input = inputs[0]
        single_sum = None
        pair_sum = None
        msa_first_row_sum = None
        for index, ensemble_input in enumerate(inputs):
            msa, single, pair, msa_first_row = self._embed_and_refine(
                ensemble_input,
                previous_positions=previous_positions,
                previous_msa_first_row=previous_msa_first_row,
                previous_pair=previous_pair,
            )
            if index == 0:
                selected_msa = msa
                selected_input = ensemble_input
            single_sum = single if single_sum is None else single_sum + single
            pair_sum = pair if pair_sum is None else pair_sum + pair
            msa_first_row_sum = msa_first_row if msa_first_row_sum is None else msa_first_row_sum + msa_first_row

        assert selected_msa is not None
        assert single_sum is not None
        assert pair_sum is not None
        assert msa_first_row_sum is not None
        if len(inputs) == 1:
            return selected_msa, single_sum, pair_sum, msa_first_row_sum, selected_input
        scale = 1.0 / len(inputs)
        return (
            selected_msa,
            single_sum * scale,
            pair_sum * scale,
            msa_first_row_sum * scale,
            selected_input,
        )

    def _realize_msa_features(
        self,
        msa_tokens: Tensor,
        msa_deletion_counts: Tensor,
        msa_profile: Tensor,
        *,
        residue_mask: Tensor,
        generator: torch.Generator | None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> _MsaRealization:
        cfg = self.config.msa
        return _realize_msa_features(
            msa_tokens,
            msa_deletion_counts,
            msa_profile,
            residue_mask=residue_mask,
            max_msa_clusters=cfg.max_msa_clusters,
            max_extra_msa=cfg.max_extra_msa,
            msa_cluster_features=cfg.msa_cluster_features,
            gap_agreement_weight=cfg.gap_agreement_weight,
            masked_msa_enabled=cfg.masked_msa_enabled,
            uniform_prob=cfg.masked_msa_uniform_prob,
            profile_prob=cfg.masked_msa_profile_prob,
            same_prob=cfg.masked_msa_same_prob,
            replace_fraction=cfg.masked_msa_replace_fraction,
            generator=generator,
            device=device,
            dtype=dtype,
        )


class AlphaFold2InitialGuessModel(AlphaFold2Model):
    r"""
    Evaluate a target--binder complex with the AF2 monomer/pTM initial-guess equation.

    The parameter graph is identical to [`AlphaFold2Model`][]. The first pass
    recycles the input atom37 coordinates instead of zeros. This is ColabDesign's
    ``use_initial_guess`` path; it does not initialize Structure Module frames
    (the distinct ``use_initial_atom_pos`` path is disabled by PXDesign). The
    all-GAP template retains backbone/CB geometry and masks pair features
    between the target and binder asymmetric units.

    Args:
        config: Initial-guess model architecture and inference configuration.
    """

    config_class = AlphaFold2InitialGuessConfig
    config: AlphaFold2InitialGuessConfig

    def __init__(self, config: AlphaFold2InitialGuessConfig | None = None) -> None:
        super().__init__(AlphaFold2InitialGuessConfig() if config is None else config)

    def _initial_recycling(
        self,
        input: _AlphaFold2RecyclingInput,
    ) -> tuple[Tensor, Tensor, Tensor]:
        _, previous_msa_first_row, previous_pair = super()._initial_recycling(input)
        initial_guess_input = cast(AlphaFold2InitialGuessInput, input)
        return initial_guess_input.initial_recycling_positions, previous_msa_first_row, previous_pair

    def _add_template_embedding(
        self,
        pair: Tensor,
        input: _AlphaFold2RecyclingInput,
        pair_mask: Tensor,
    ) -> Tensor:
        if self.template_embedder is None:
            return pair
        initial_guess_input = cast(AlphaFold2InitialGuessInput, input)
        template_pair_mask = initial_guess_input.chain_index.unsqueeze(-1) == initial_guess_input.chain_index.unsqueeze(
            -2
        )
        return pair + self.template_embedder(
            pair,
            initial_guess_input,
            pair_mask=pair_mask,
            template_pair_mask=template_pair_mask,
        )


class AlphaFold2MultimerModel(_AlphaFold2TrunkModel):
    r"""
    Predict multimeric protein structures with AlphaFold-Multimer.

    Accepts [`AlphaFold2MultimerInput`][ofoldx.models.alphafold2.model.AlphaFold2MultimerInput] and returns
    [`AlphaFold2Output`][ofoldx.models.alphafold2.model.AlphaFold2Output].

    Args:
        config: Model architecture and inference configuration.
    """

    config_class = AlphaFold2MultimerConfig
    config: AlphaFold2MultimerConfig

    def __init__(self, config: AlphaFold2MultimerConfig | None = None) -> None:
        config = AlphaFold2MultimerConfig() if config is None else config
        super().__init__(config)
        input_config = config.input_embeddings
        recycling_config = config.recycling
        template_config = config.template
        evoformer_config: AlphaFold2EvoformerConfig = config.evoformer
        block_config = evoformer_config.block
        structure_module_config = config.structure_module
        heads = config.heads

        self._num_recycles = operator.index(config.num_recycles)
        if self._num_recycles < 1:
            fail("value_invalid", RequestRef("num_recycles", str(self._num_recycles)))
        self.num_ensembles = operator.index(config.num_ensembles)
        if self.num_ensembles < 1:
            fail("value_invalid", RequestRef("num_ensembles", str(self.num_ensembles)))
        self._resample_msa_in_recycling = config.msa.resample_msa_in_recycling
        self.input_embeddings = AlphaFold2MultimerInputEmbeddings(
            token_feature_size=input_config.token_feature_size,
            msa_size=evoformer_config.msa_size,
            pair_size=evoformer_config.pair_size,
            max_relative_index=input_config.max_relative_index,
            max_relative_chain=input_config.max_relative_chain,
            use_chain_relative=input_config.use_chain_relative,
        )
        self.recycling_embeddings = AlphaFold2RecyclingEmbeddings(
            msa_size=evoformer_config.msa_size,
            pair_size=evoformer_config.pair_size,
            recycle_positions=recycling_config.recycle_positions,
            recycle_features=recycling_config.recycle_features,
            previous_positions_min_distance=recycling_config.previous_positions.min_distance,
            previous_positions_max_distance=recycling_config.previous_positions.max_distance,
            num_previous_position_distogram_bins=recycling_config.previous_positions.num_distogram_bins,
        )
        self.template_embedder: AlphaFold2MultimerTemplateEmbedder | None = None
        if template_config.enabled:
            self.template_embedder = AlphaFold2MultimerTemplateEmbedder(
                query_size=evoformer_config.pair_size,
                template_size=template_config.template_pair_stack.pair_size,
                num_blocks=template_config.template_pair_stack.num_blocks,
                num_distogram_bins=template_config.num_distogram_bins,
                min_distance=template_config.min_distance,
                max_distance=template_config.max_distance,
                vocab_size=template_config.vocab_size,
                triangle_multiplication_hidden_size=(
                    template_config.template_pair_stack.triangle_multiplication_outgoing.hidden_size
                ),
                triangle_attention_head_size=(
                    template_config.template_pair_stack.triangle_attention_starting_node.head_size
                ),
                num_triangle_attention_heads=(
                    template_config.template_pair_stack.triangle_attention_starting_node.num_attention_heads
                ),
                transition_expansion=template_config.template_pair_stack.transition.expansion,
                dropout=template_config.template_pair_stack.triangle_attention_starting_node.dropout,
                backend="auto",
            )
        self.extra_msa_projection = nn.Linear(
            NUM_CLASSES + 2,
            evoformer_config.extra_msa_size,
        )
        evoformer_block_type = (
            OuterProductFirstEvoformerBlock if block_config.outer_product_mean.first else EvoformerBlock
        )
        self.extra_msa_stack = EvoformerStack(
            block=evoformer_block_type,
            msa_block=ExtraMsaTrackBlock,
            num_blocks=evoformer_config.num_extra_msa_blocks,
            msa_size=evoformer_config.extra_msa_size,
            pair_size=evoformer_config.pair_size,
            outer_product_hidden_size=block_config.outer_product_mean.hidden_size,
            num_msa_row_attention_heads=block_config.msa_row_attention_with_pair_bias.num_attention_heads,
            num_msa_column_attention_heads=block_config.msa_row_attention_with_pair_bias.num_attention_heads,
            triangle_multiplication_outgoing_hidden_size=(block_config.triangle_multiplication_outgoing.hidden_size),
            triangle_multiplication_incoming_hidden_size=(block_config.triangle_multiplication_outgoing.hidden_size),
            triangle_attention_head_size=block_config.triangle_attention_starting_node.head_size,
            num_triangle_attention_starting_node_heads=(
                block_config.triangle_attention_starting_node.num_attention_heads
            ),
            num_triangle_attention_ending_node_heads=(
                block_config.triangle_attention_starting_node.num_attention_heads
            ),
            msa_transition_expansion=block_config.msa_transition.expansion,
            pair_transition_expansion=block_config.pair_transition.expansion,
            msa_dropout=block_config.msa_row_attention_with_pair_bias.dropout,
            pair_dropout=block_config.triangle_attention_starting_node.dropout,
            backend="auto",
        )
        self.template_msa_embedder: AlphaFold2MultimerTemplateMsaEmbedder | None = None
        if template_config.enabled:
            self.template_msa_embedder = AlphaFold2MultimerTemplateMsaEmbedder(
                msa_size=evoformer_config.msa_size,
                vocab_size=template_config.vocab_size,
            )
        self.evoformer_stack = EvoformerStack(
            block=evoformer_block_type,
            num_blocks=evoformer_config.num_blocks,
            msa_size=evoformer_config.msa_size,
            pair_size=evoformer_config.pair_size,
            outer_product_hidden_size=block_config.outer_product_mean.hidden_size,
            num_msa_row_attention_heads=block_config.msa_row_attention_with_pair_bias.num_attention_heads,
            num_msa_column_attention_heads=block_config.msa_row_attention_with_pair_bias.num_attention_heads,
            triangle_multiplication_outgoing_hidden_size=(block_config.triangle_multiplication_outgoing.hidden_size),
            triangle_multiplication_incoming_hidden_size=(block_config.triangle_multiplication_outgoing.hidden_size),
            triangle_attention_head_size=block_config.triangle_attention_starting_node.head_size,
            num_triangle_attention_starting_node_heads=(
                block_config.triangle_attention_starting_node.num_attention_heads
            ),
            num_triangle_attention_ending_node_heads=(
                block_config.triangle_attention_starting_node.num_attention_heads
            ),
            msa_transition_expansion=block_config.msa_transition.expansion,
            pair_transition_expansion=block_config.pair_transition.expansion,
            msa_dropout=block_config.msa_row_attention_with_pair_bias.dropout,
            pair_dropout=block_config.triangle_attention_starting_node.dropout,
            backend="auto",
        )
        self.single_projection = nn.Linear(evoformer_config.msa_size, evoformer_config.single_size)
        self.structure_module = AlphaFold2MultimerStructureModule(
            single_size=evoformer_config.single_size,
            pair_size=evoformer_config.pair_size,
            num_blocks=structure_module_config.num_blocks,
            num_transition_projections=structure_module_config.num_transition_projections,
            hidden_size=structure_module_config.hidden_size,
            num_attention_heads=structure_module_config.num_attention_heads,
            scalar_query_key_size=structure_module_config.scalar_query_key_size,
            scalar_value_size=structure_module_config.scalar_value_size,
            num_point_qk=structure_module_config.num_point_qk,
            num_point_v=structure_module_config.num_point_v,
            dropout=structure_module_config.dropout,
            position_scale=structure_module_config.position_scale,
            sidechain_hidden_size=structure_module_config.sidechain.hidden_size,
            num_sidechain_blocks=structure_module_config.sidechain.num_blocks,
        )
        self.distogram_head = DistogramHead(
            pair_size=evoformer_config.pair_size,
            num_bins=heads.distogram.num_distogram_bins,
        )
        nn.init.zeros_(self.distogram_head.projection.weight)
        nn.init.zeros_(self.distogram_head.projection.bias)
        self.predicted_lddt_head = AlphaFold2PredictedLddtHead(
            input_size=structure_module_config.hidden_size,
            hidden_size=heads.predicted_lddt.hidden_size,
            num_bins=heads.predicted_lddt.num_plddt_bins,
        )
        # The multimer TM head is always present.
        self.predicted_aligned_error_head = nn.Linear(
            evoformer_config.pair_size,
            heads.predicted_aligned_error.num_pae_bins,
        )
        nn.init.zeros_(self.predicted_aligned_error_head.weight)
        nn.init.zeros_(self.predicted_aligned_error_head.bias)
        self.experimentally_resolved_head = nn.Linear(
            evoformer_config.single_size,
            heads.experimentally_resolved.num_atom_types,
        )
        nn.init.zeros_(self.experimentally_resolved_head.weight)
        nn.init.zeros_(self.experimentally_resolved_head.bias)
        self.masked_msa_head = nn.Linear(
            evoformer_config.msa_size,
            len(SOURCE_MSA_ALPHABET),
        )
        nn.init.zeros_(self.masked_msa_head.weight)
        nn.init.zeros_(self.masked_msa_head.bias)
        self.post_init()

    def forward(
        self,
        input: AlphaFold2MultimerInput,
        *,
        num_recycles: int | None = None,
        generator: torch.Generator | None = None,
    ) -> AlphaFold2Output:
        r"""
        Run one AlphaFold 2 structure-prediction pass.

        The input tensors use batch and residue axes ``(B, N)``. When
        ``input.msa_features`` is absent, ``generator`` controls MSA row sampling,
        masking, clustering, and extra-MSA cropping. Precomputed MSA features do
        not consume this generator. Dropout follows the module's training mode and
        uses PyTorch's ambient random state.

        Args:
            input: Multimer features aligned to ``B`` systems and ``N``
                residues.
            num_recycles: Maximum total network passes, including the final pass.
                Early stopping may execute fewer. ``None`` uses the configured
                value.
            generator: Optional generator for stochastic MSA feature construction.

        Returns:
            Predictions with atom37 coordinates in angstroms and tensor axes
                described by [`AlphaFold2Output`][].

        Raises:
            ValueError: If ``num_recycles`` is not positive.
        """
        input = normalize_alphafold2_input(input, dtype=self.input_embeddings.token_projection.weight.dtype)

        num_recycles = operator.index(self._num_recycles if num_recycles is None else num_recycles)
        if num_recycles < 1:
            fail("value_invalid", RequestRef("num_recycles", str(num_recycles)))

        msa, single, pair, structure, realized, executed_recycles = self._run_recycling(
            input, num_recycles=num_recycles, generator=generator
        )
        assert realized.msa_features is not None
        num_msa_rows = realized.msa_features.shape[-3]
        distogram_logits = self.distogram_head(pair)
        plddt_logits = self.predicted_lddt_head(structure.structure_module)
        pae_logits = None if self.predicted_aligned_error_head is None else self.predicted_aligned_error_head(pair)
        resolved_logits = self.experimentally_resolved_head(single)
        masked_msa_logits = self.masked_msa_head(msa.narrow(-3, 0, num_msa_rows))
        optional_output = {
            name: value
            for name, value in (
                ("true_msa", realized.true_msa),
                ("bert_mask", realized.bert_mask),
                ("pae_logits", pae_logits),
            )
            if value is not None
        }
        return AlphaFold2Output(
            single=single,
            pair=pair,
            frames=structure.traj.movedim(0, 1),
            atom14_positions=structure.atom14_positions.movedim(0, 1),
            sidechain_frames=structure.sidechain_frames.movedim(0, 1),
            final_atom_positions=structure.final_atom_positions,
            final_atom_mask=structure.final_atom_mask,
            final_atom14_positions=structure.final_atom14_positions,
            final_atom14_mask=structure.final_atom14_mask,
            sidechain_angles=structure.sidechain_angles.movedim(0, 1),
            sidechain_unnormalized_angles=structure.sidechain_unnormalized_angles.movedim(0, 1),
            distogram_logits=distogram_logits,
            masked_msa_logits=masked_msa_logits,
            plddt_logits=plddt_logits,
            resolved_logits=resolved_logits,
            num_recycles=torch.full((single.shape[0],), executed_recycles, dtype=torch.int64, device=single.device),
            batch_size=[single.shape[0]],
            **optional_output,
        )

    def _run_recycling(
        self,
        input: _AlphaFold2InputT,
        *,
        num_recycles: int,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, AlphaFold2StructureModuleOutput, _AlphaFold2InputT, int]:
        needs_realize = input.msa_features is None
        resample = needs_realize and self._resample_msa_in_recycling
        fixed_ensemble: tuple[_AlphaFold2InputT, ...] | None = None
        if needs_realize and not resample:
            fixed_ensemble = tuple(self._realize_msa(input, generator=generator) for _ in range(self.num_ensembles))
        previous_positions, previous_msa_first_row, previous_pair = self._initial_recycling(input)
        early_stop_tolerance = self.config.recycle_early_stop_tolerance
        stop_after_cycle = False
        for cycle in range(num_recycles):
            if needs_realize:
                if resample:
                    cycle_inputs = tuple(
                        self._realize_msa(input, generator=generator) for _ in range(self.num_ensembles)
                    )
                else:
                    assert fixed_ensemble is not None
                    cycle_inputs = fixed_ensemble
            else:
                cycle_inputs = (input,) * self.num_ensembles
            final_cycle = stop_after_cycle or cycle == num_recycles - 1
            grad_enabled = torch.is_grad_enabled() and final_cycle
            with torch.set_grad_enabled(grad_enabled):
                msa, single, pair, msa_first_row, cycle_input = self._run_ensemble(
                    cycle_inputs,
                    previous_positions=previous_positions,
                    previous_msa_first_row=previous_msa_first_row,
                    previous_pair=previous_pair,
                )
                structure = self.structure_module(single, pair, cycle_input)
            if final_cycle:
                break
            next_positions = structure.final_atom_positions.detach()
            if early_stop_tolerance is not None:
                ca_index = residue_constants.atom_order["CA"]
                previous_ca = previous_positions[..., ca_index, :]
                next_ca = next_positions[..., ca_index, :]
                previous_distances = torch.sqrt(
                    torch.square(previous_ca.unsqueeze(-2) - previous_ca.unsqueeze(-3)).sum(dim=-1)
                )
                next_distances = torch.sqrt(torch.square(next_ca.unsqueeze(-2) - next_ca.unsqueeze(-3)).sum(dim=-1))
                squared_difference = torch.square(previous_distances - next_distances)
                pair_mask = input.residue_mask.unsqueeze(-1) * input.residue_mask.unsqueeze(-2)
                # The recycle early-stop is control flow, not a reported metric: clamp so the decision
                # stays well defined. A NaN difference would compare False and silently disable the
                # early stop, burning every remaining cycle.
                mean_squared_difference = (squared_difference * pair_mask).sum(dim=(-2, -1)) / (
                    pair_mask.sum(dim=(-2, -1)).clamp_min(1)
                )
                difference = torch.sqrt(mean_squared_difference + 1e-8)
                stop_after_cycle = bool((difference <= early_stop_tolerance).all())
            previous_positions = next_positions
            previous_msa_first_row = msa_first_row.detach()
            previous_pair = pair.detach()
        return msa, single, pair, structure, cycle_input, cycle + 1

    def _run_ensemble(
        self,
        inputs: tuple[_AlphaFold2InputT, ...],
        *,
        previous_positions: Tensor,
        previous_msa_first_row: Tensor,
        previous_pair: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, _AlphaFold2InputT]:
        selected_msa = None
        selected_input = inputs[0]
        single_sum = None
        pair_sum = None
        msa_first_row_sum = None
        for ensemble_input in inputs:
            msa, single, pair, msa_first_row = self._embed_and_refine(
                ensemble_input,
                previous_positions=previous_positions,
                previous_msa_first_row=previous_msa_first_row,
                previous_pair=previous_pair,
            )
            selected_msa = msa
            selected_input = ensemble_input
            single_sum = single if single_sum is None else single_sum + single
            pair_sum = pair if pair_sum is None else pair_sum + pair
            msa_first_row_sum = msa_first_row if msa_first_row_sum is None else msa_first_row_sum + msa_first_row

        assert selected_msa is not None
        assert single_sum is not None
        assert pair_sum is not None
        assert msa_first_row_sum is not None
        if len(inputs) == 1:
            return selected_msa, single_sum, pair_sum, msa_first_row_sum, selected_input
        scale = 1.0 / len(inputs)
        return (
            selected_msa,
            single_sum * scale,
            pair_sum * scale,
            msa_first_row_sum * scale,
            selected_input,
        )

    def _realize_msa_features(
        self,
        msa_tokens: Tensor,
        msa_deletion_counts: Tensor,
        msa_profile: Tensor,
        *,
        residue_mask: Tensor,
        generator: torch.Generator | None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> _MsaRealization:
        r"""
        Realize one multimer MSA-feature ensemble slice from the raw pool.

        Runs multimer MSA sampling, masking, nearest-neighbor clustering, and feature
        assembly over the raw paired alignment tensors supplied by the shared
        realization boundary. Every draw consumes the per-sample ``generator`` so
        the slice is reproducible from the fold seed.
        """

        cfg = self.config.msa
        return _realize_multimer_msa_features(
            msa_tokens,
            msa_deletion_counts,
            msa_profile,
            residue_mask=residue_mask,
            max_msa_clusters=cfg.max_msa_clusters,
            max_extra_msa=cfg.max_extra_msa,
            msa_cluster_features=cfg.msa_cluster_features,
            gap_agreement_weight=cfg.gap_agreement_weight,
            masked_msa_enabled=cfg.masked_msa_enabled,
            uniform_prob=cfg.masked_msa_uniform_prob,
            profile_prob=cfg.masked_msa_profile_prob,
            same_prob=cfg.masked_msa_same_prob,
            replace_fraction=cfg.masked_msa_replace_fraction,
            generator=generator,
            device=device,
            dtype=dtype,
        )


class AlphaFold2StructureModule(nn.Module):
    r"""
    Native AlphaFold 2 StructureModule inference path.
    """

    def __init__(
        self,
        *,
        single_size: int,
        pair_size: int,
        num_blocks: int,
        num_transition_projections: int,
        hidden_size: int,
        num_attention_heads: int,
        scalar_query_key_size: int,
        scalar_value_size: int,
        num_point_qk: int,
        num_point_v: int,
        sidechain_hidden_size: int,
        num_sidechain_blocks: int,
        position_scale: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.position_scale = position_scale
        self.single_norm = LayerNorm(single_size)
        self.input_projection = nn.Linear(single_size, hidden_size)
        self.pair_norm = LayerNorm(pair_size)
        self.block = AlphaFold2StructureModuleBlock(
            hidden_size=hidden_size,
            pair_size=pair_size,
            num_transition_projections=num_transition_projections,
            num_attention_heads=num_attention_heads,
            scalar_query_key_size=scalar_query_key_size,
            scalar_value_size=scalar_value_size,
            num_point_qk=num_point_qk,
            num_point_v=num_point_v,
            sidechain_hidden_size=sidechain_hidden_size,
            num_sidechain_blocks=num_sidechain_blocks,
            position_scale=position_scale,
            dropout=dropout,
        )

    def forward(self, single: Tensor, pair: Tensor, input: AlphaFold2Input) -> AlphaFold2StructureModuleOutput:
        residue_mask = input.residue_mask.to(dtype=single.dtype)
        residue_type_ids = input.residue_type_ids.clamp(min=0, max=20)
        act = self.single_norm(single)
        initial_act = act
        act = self.input_projection(act)
        pair = self.pair_norm(pair)
        affine = AlphaFold2QuatAffine.identity(residue_mask.shape, device=single.device, dtype=single.dtype)

        affines = []
        atom14_trajectory = []
        sidechain_frame_trajectory = []
        sidechain_angle_trajectory = []
        sidechain_unnormalized_angle_trajectory = []
        for _ in range(self.num_blocks):
            act, affine, angles, unnormalized_angles, sidechain_frames, atom14_positions, raw_affine = self.block(
                act=act,
                pair=pair,
                mask=residue_mask,
                affine=affine,
                initial_act=initial_act,
                residue_type_ids=residue_type_ids,
            )
            affines.append(raw_affine.to_tensor())
            atom14_trajectory.append(atom14_positions)
            sidechain_frame_trajectory.append(sidechain_frames.to_tensor4x4())
            sidechain_angle_trajectory.append(angles)
            sidechain_unnormalized_angle_trajectory.append(unnormalized_angles)

        traj = torch.stack(affines, dim=0)
        atom14_trajectory_tensor = torch.stack(atom14_trajectory, dim=0)
        sidechain_frame_trajectory_tensor = torch.stack(sidechain_frame_trajectory, dim=0)
        sidechain_angle_trajectory_tensor = torch.stack(sidechain_angle_trajectory, dim=0)
        sidechain_unnormalized_angle_trajectory_tensor = torch.stack(
            sidechain_unnormalized_angle_trajectory,
            dim=0,
        )
        traj_scale = traj.new_tensor(
            [1.0, 1.0, 1.0, 1.0, self.position_scale, self.position_scale, self.position_scale],
        )
        traj = traj * traj_scale
        atom14_mask = _residue_type_atom14_mask(residue_type_ids, dtype=single.dtype) * residue_mask.unsqueeze(-1)
        atom37_positions, atom37_mask = _atom14_to_atom37(atom14_positions, residue_type_ids, residue_mask)
        return AlphaFold2StructureModuleOutput(
            final_atom_positions=atom37_positions,
            final_atom_mask=atom37_mask,
            structure_module=act,
            final_atom14_positions=atom14_positions,
            final_atom14_mask=atom14_mask,
            final_affines=traj[-1],
            traj=traj,
            atom14_positions=atom14_trajectory_tensor,
            sidechain_frames=sidechain_frame_trajectory_tensor,
            sidechain_angles=sidechain_angle_trajectory_tensor,
            sidechain_unnormalized_angles=sidechain_unnormalized_angle_trajectory_tensor,
        )


class AlphaFold2MultimerStructureModule(nn.Module):
    r"""
    AF-Multimer structure module.
    """

    def __init__(
        self,
        *,
        single_size: int,
        pair_size: int,
        num_blocks: int,
        num_transition_projections: int,
        hidden_size: int,
        num_attention_heads: int,
        scalar_query_key_size: int,
        scalar_value_size: int,
        num_point_qk: int,
        num_point_v: int,
        sidechain_hidden_size: int,
        num_sidechain_blocks: int,
        position_scale: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.position_scale = position_scale
        self.single_norm = LayerNorm(single_size)
        self.input_projection = nn.Linear(single_size, hidden_size)
        self.pair_norm = LayerNorm(pair_size)
        self.block = AlphaFold2MultimerStructureModuleBlock(
            hidden_size=hidden_size,
            pair_size=pair_size,
            num_transition_projections=num_transition_projections,
            num_attention_heads=num_attention_heads,
            scalar_query_key_size=scalar_query_key_size,
            scalar_value_size=scalar_value_size,
            num_point_qk=num_point_qk,
            num_point_v=num_point_v,
            sidechain_hidden_size=sidechain_hidden_size,
            num_sidechain_blocks=num_sidechain_blocks,
            position_scale=position_scale,
            dropout=dropout,
        )

    def forward(self, single: Tensor, pair: Tensor, input: AlphaFold2MultimerInput) -> AlphaFold2StructureModuleOutput:
        residue_mask = input.residue_mask.to(dtype=single.dtype)
        residue_type_ids = input.residue_type_ids.clamp(min=0, max=20)
        act = self.single_norm(single)
        initial_act = act
        act = self.input_projection(act)
        pair = self.pair_norm(pair)
        rigid = AlphaFold2Rigid.identity(residue_mask.shape, device=single.device, dtype=single.dtype)

        frames = []
        atom14_trajectory = []
        sidechain_frame_trajectory = []
        sidechain_angle_trajectory = []
        sidechain_unnormalized_angle_trajectory = []
        for _ in range(self.num_blocks):
            act, rigid, angles, unnormalized_angles, sidechain_frames, atom14_positions = self.block(
                act=act,
                pair=pair,
                mask=residue_mask,
                rigid=rigid,
                initial_act=initial_act,
                residue_type_ids=residue_type_ids,
            )
            scaled = rigid.scale_translation(self.position_scale)
            frames.append(
                torch.cat([scaled.rotation.reshape(*scaled.rotation.shape[:-2], 9), scaled.translation], dim=-1)
            )
            atom14_trajectory.append(atom14_positions)
            sidechain_frame_trajectory.append(sidechain_frames.to_tensor4x4())
            sidechain_angle_trajectory.append(angles)
            sidechain_unnormalized_angle_trajectory.append(unnormalized_angles)
            rigid = rigid.detach_rotation()

        traj = torch.stack(frames, dim=0)
        atom14_trajectory_tensor = torch.stack(atom14_trajectory, dim=0)
        sidechain_frame_trajectory_tensor = torch.stack(sidechain_frame_trajectory, dim=0)
        sidechain_angle_trajectory_tensor = torch.stack(sidechain_angle_trajectory, dim=0)
        sidechain_unnormalized_angle_trajectory_tensor = torch.stack(
            sidechain_unnormalized_angle_trajectory,
            dim=0,
        )
        atom14_mask = _residue_type_atom14_mask(residue_type_ids, dtype=single.dtype) * residue_mask.unsqueeze(-1)
        atom37_positions, atom37_mask = _atom14_to_atom37(atom14_positions, residue_type_ids, residue_mask)
        return AlphaFold2StructureModuleOutput(
            final_atom_positions=atom37_positions,
            final_atom_mask=atom37_mask,
            structure_module=act,
            final_atom14_positions=atom14_positions,
            final_atom14_mask=atom14_mask,
            final_affines=traj[-1],
            traj=traj,
            atom14_positions=atom14_trajectory_tensor,
            sidechain_frames=sidechain_frame_trajectory_tensor,
            sidechain_angles=sidechain_angle_trajectory_tensor,
            sidechain_unnormalized_angles=sidechain_unnormalized_angle_trajectory_tensor,
        )


class AlphaFold2TemplateEmbedder(nn.Module):
    r"""
    Embed AF2 template pair features.
    """

    def __init__(
        self,
        *,
        query_size: int,
        template_size: int,
        num_blocks: int,
        num_distogram_bins: int,
        min_distance: float,
        max_distance: float,
        vocab_size: int,
        use_template_unit_vector: bool,
        triangle_multiplication_hidden_size: int,
        triangle_attention_head_size: int,
        num_triangle_attention_heads: int,
        transition_expansion: int,
        dropout: float,
        num_attention_heads: int,
        backend: str,
    ) -> None:
        super().__init__()
        self.single_template_embedder = AlphaFold2SingleTemplateEmbedder(
            template_size=template_size,
            num_blocks=num_blocks,
            num_distogram_bins=num_distogram_bins,
            min_distance=min_distance,
            max_distance=max_distance,
            vocab_size=vocab_size,
            use_template_unit_vector=use_template_unit_vector,
            triangle_multiplication_hidden_size=triangle_multiplication_hidden_size,
            triangle_attention_head_size=triangle_attention_head_size,
            num_triangle_attention_heads=num_triangle_attention_heads,
            transition_expansion=transition_expansion,
            dropout=dropout,
            backend=backend,
        )
        self.attention = MultiheadAttention(
            MultiheadAttentionProjection(
                hidden_size=query_size,
                query_size=query_size,
                key_size=template_size,
                value_size=template_size,
                num_attention_heads=num_attention_heads,
                head_size=template_size // num_attention_heads,
                query_bias=False,
                output_bias=True,
            ),
            backend=backend,
        )

    def forward(
        self,
        query_embedding: Tensor,
        input: AlphaFold2Input,
        *,
        pair_mask: Tensor,
        template_pair_mask: Tensor | None = None,
    ) -> Tensor:
        input = normalize_alphafold2_input(input, dtype=query_embedding.dtype)
        template_batch = _template_batch(input)
        template_pair = self.single_template_embedder(
            template_batch,
            pair_mask=pair_mask,
            template_pair_mask=template_pair_mask,
        )
        batch_size, num_templates, num_tokens = template_pair.shape[:3]
        template_mask = template_batch.template_mask.to(dtype=query_embedding.dtype)
        has_template = template_mask.sum(dim=-1) > 0
        safe_template_mask = torch.where(
            has_template.unsqueeze(-1),
            template_mask.to(torch.bool),
            torch.ones_like(template_mask, dtype=torch.bool),
        )

        flat_query = query_embedding.reshape(batch_size * num_tokens * num_tokens, 1, query_embedding.shape[-1])
        flat_templates = template_pair.permute(0, 2, 3, 1, 4).reshape(
            batch_size * num_tokens * num_tokens,
            num_templates,
            template_pair.shape[-1],
        )
        flat_mask = (
            safe_template_mask[:, None, None, :]
            .expand(batch_size, num_tokens, num_tokens, num_templates)
            .reshape(batch_size * num_tokens * num_tokens, 1, num_templates)
        )
        update = self.attention(flat_query, flat_templates, flat_templates, attention_mask=flat_mask)
        update = update.reshape(batch_size, num_tokens, num_tokens, query_embedding.shape[-1])
        return update * has_template[:, None, None, None].to(dtype=update.dtype)


class AlphaFold2MultimerTemplateEmbedder(nn.Module):
    r"""
    AF-Multimer template pair embedder.
    """

    def __init__(
        self,
        *,
        query_size: int,
        template_size: int,
        num_blocks: int,
        num_distogram_bins: int,
        min_distance: float,
        max_distance: float,
        vocab_size: int,
        triangle_multiplication_hidden_size: int,
        triangle_attention_head_size: int,
        num_triangle_attention_heads: int,
        transition_expansion: int,
        dropout: float,
        backend: str,
    ) -> None:
        super().__init__()
        self.backend = backend
        self.single_template_embedder = AlphaFold2MultimerSingleTemplateEmbedder(
            query_size=query_size,
            template_size=template_size,
            num_blocks=num_blocks,
            num_distogram_bins=num_distogram_bins,
            min_distance=min_distance,
            max_distance=max_distance,
            vocab_size=vocab_size,
            triangle_multiplication_hidden_size=triangle_multiplication_hidden_size,
            triangle_attention_head_size=triangle_attention_head_size,
            num_triangle_attention_heads=num_triangle_attention_heads,
            transition_expansion=transition_expansion,
            dropout=dropout,
            backend=backend,
        )
        self.output_projection = nn.Linear(template_size, query_size)

    def forward(self, query_embedding: Tensor, input: AlphaFold2MultimerInput, *, pair_mask: Tensor) -> Tensor:
        input = normalize_alphafold2_input(input, dtype=query_embedding.dtype)
        num_templates = input.template_residue_type_ids.shape[1]
        chain_index = input.chain_index
        multichain_mask_2d = (chain_index.unsqueeze(-1) == chain_index.unsqueeze(-2)).to(dtype=query_embedding.dtype)
        positions = input.template_all_atom_positions
        atom_mask = input.template_all_atom_mask
        pseudo_beta = _pseudo_beta(input.template_residue_type_ids, positions)
        pseudo_beta_mask = _pseudo_beta(input.template_residue_type_ids, atom_mask.unsqueeze(-1)).squeeze(-1)
        unit_vectors, backbone_masks = _backbone_unit_vectors(positions, atom_mask, backend=self.backend)

        accumulated = query_embedding.new_zeros(*query_embedding.shape[:-1], self.output_projection.in_features)
        for index in range(num_templates):
            act = self.single_template_embedder(
                query_embedding,
                residue_type_ids=input.template_residue_type_ids[:, index],
                pseudo_beta=pseudo_beta[:, index],
                pseudo_beta_mask=pseudo_beta_mask[:, index],
                unit_vector=unit_vectors[:, index],
                backbone_mask=backbone_masks[:, index],
                multichain_mask_2d=multichain_mask_2d,
                pair_mask=pair_mask,
            )
            accumulated = accumulated + act
        accumulated = accumulated / max(num_templates, 1)
        return self.output_projection(F.relu(accumulated))


class AlphaFold2InputEmbeddings(nn.Module):
    r"""
    Build initial token/MSA/pair activations before recycling or Evoformer blocks.
    """

    def __init__(
        self,
        *,
        token_feature_size: int,
        msa_size: int,
        pair_size: int,
        max_relative_feature: int,
    ) -> None:
        super().__init__()
        self.max_relative_feature = max_relative_feature
        self.token_projection = nn.Linear(token_feature_size, msa_size)
        self.msa_projection = nn.Linear(2 * NUM_CLASSES + 3, msa_size)
        self.pair_left_projection = nn.Linear(token_feature_size, pair_size)
        self.pair_right_projection = nn.Linear(token_feature_size, pair_size)
        self.relative_position_projection = nn.Linear(2 * max_relative_feature + 1, pair_size)

    def forward(self, input: AlphaFold2Input) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        input = normalize_alphafold2_input(input, dtype=self.token_projection.weight.dtype)
        assert input.msa_features is not None
        assert input.msa_mask is not None
        token_embedding = self.token_projection(input.token_features)
        msa = token_embedding.unsqueeze(-3) + self.msa_projection(input.msa_features)
        pair = self._build_pair_activations(input, input.token_features)
        residue_mask = input.residue_mask.to(torch.bool)
        pair_mask = residue_mask.unsqueeze(-1) & residue_mask.unsqueeze(-2)
        return msa, pair, input.msa_mask, pair_mask

    def _build_pair_activations(self, input: AlphaFold2Input, token_features: Tensor) -> Tensor:
        pair_left = self.pair_left_projection(token_features)
        pair_right = self.pair_right_projection(token_features)
        pair = pair_left.unsqueeze(-2) + pair_right.unsqueeze(-3)
        relpos = _clipped_relative_position(input.residue_index, self.max_relative_feature).to(
            device=pair.device,
            dtype=pair.dtype,
        )
        return pair + self.relative_position_projection(relpos)


# ===========================================================================
# AlphaFold-Multimer top-level model.
#
# The multimer Model reuses the shared Evoformer / extra-MSA / StructureModule
# trunk, its heads, and the AF2 geometry utilities defined above. It differs from
# the monomer Model only in its chain-relative input embedder, its atom37 multimer
# template embedder, its canonical-vocabulary masked-MSA head, an always-present interface-pTM
# branch, and its result-mapping (ipTM and ranking score).
#
# ``AlphaFold2MultimerModel`` owns its multimer graph and result assembly while
# reusing the shared trunk and the AF2 structural components below.
# ===========================================================================


class AlphaFold2MultimerInputEmbeddings(nn.Module):
    r"""
    Build initial token/MSA/pair activations with chain-relative positional encoding.
    """

    def __init__(
        self,
        *,
        token_feature_size: int,
        msa_size: int,
        pair_size: int,
        max_relative_index: int,
        max_relative_chain: int,
        use_chain_relative: bool,
    ) -> None:
        super().__init__()
        self.max_relative_index = max_relative_index
        self.max_relative_chain = max_relative_chain
        self.use_chain_relative = use_chain_relative
        self.token_projection = nn.Linear(token_feature_size, msa_size)
        self.msa_projection = nn.Linear(2 * NUM_CLASSES + 3, msa_size)
        self.pair_left_projection = nn.Linear(token_feature_size, pair_size)
        self.pair_right_projection = nn.Linear(token_feature_size, pair_size)
        self.relative_position_projection = nn.Linear(self.relative_feature_size, pair_size)

    @property
    def relative_feature_size(self) -> int:
        if self.use_chain_relative:
            return (2 * self.max_relative_index + 2) + 1 + (2 * self.max_relative_chain + 2)
        return 2 * self.max_relative_index + 1

    def forward(self, input: AlphaFold2MultimerInput) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        input = normalize_alphafold2_input(input, dtype=self.token_projection.weight.dtype)
        assert input.msa_features is not None
        assert input.msa_mask is not None
        token_embedding = self.token_projection(input.token_features)
        msa = token_embedding.unsqueeze(-3) + self.msa_projection(input.msa_features)
        pair = self._build_pair_activations(input)
        residue_mask = input.residue_mask.to(torch.bool)
        pair_mask = (residue_mask.unsqueeze(-1) & residue_mask.unsqueeze(-2)).to(dtype=pair.dtype)
        return msa, pair, input.msa_mask, pair_mask

    def _build_pair_activations(self, input: AlphaFold2MultimerInput) -> Tensor:
        pair_left = self.pair_left_projection(input.token_features)
        pair_right = self.pair_right_projection(input.token_features)
        pair = pair_left.unsqueeze(-2) + pair_right.unsqueeze(-3)
        relative = self._relative_encoding(input).to(device=pair.device, dtype=pair.dtype)
        return pair + self.relative_position_projection(relative)

    def _relative_encoding(self, input: AlphaFold2MultimerInput) -> Tensor:
        max_relative_index = self.max_relative_index
        residue_index = input.residue_index
        chain_index = input.chain_index
        offset = residue_index.unsqueeze(-1) - residue_index.unsqueeze(-2)
        same_chain = chain_index.unsqueeze(-1) == chain_index.unsqueeze(-2)
        if not self.use_chain_relative:
            clipped = torch.clamp(offset + max_relative_index, min=0, max=2 * max_relative_index)
            return F.one_hot(clipped.long(), num_classes=2 * max_relative_index + 1).to(torch.float32)
        clipped = torch.clamp(offset + max_relative_index, min=0, max=2 * max_relative_index)
        final_offset = torch.where(
            same_chain,
            clipped,
            torch.full_like(clipped, 2 * max_relative_index + 1),
        )
        relative_position = F.one_hot(final_offset.long(), num_classes=2 * max_relative_index + 2).to(torch.float32)

        same_entity = (input.entity_index.unsqueeze(-1) == input.entity_index.unsqueeze(-2)).to(torch.float32)

        # The relative-chain feature uses the per-entity copy index, not the global
        # chain index. These coincide only when same-entity copies are contiguously
        # ordered; for an A-B-A assembly they diverge.
        symmetry_index = input.symmetry_index
        max_chain = self.max_relative_chain
        symmetry_offset = symmetry_index.unsqueeze(-1) - symmetry_index.unsqueeze(-2)
        clipped_chain = torch.clamp(symmetry_offset + max_chain, min=0, max=2 * max_chain)
        final_chain = torch.where(
            same_entity.to(torch.bool),
            clipped_chain,
            torch.full_like(clipped_chain, 2 * max_chain + 1),
        )
        relative_chain = F.one_hot(final_chain.long(), num_classes=2 * max_chain + 2).to(torch.float32)
        return torch.cat([relative_position, same_entity.unsqueeze(-1), relative_chain], dim=-1)


class AlphaFold2RecyclingEmbeddings(nn.Module):
    r"""
    Add previous-cycle features to MSA and pair activations.
    """

    def __init__(
        self,
        *,
        msa_size: int,
        pair_size: int,
        recycle_positions: bool,
        recycle_features: bool,
        previous_positions_min_distance: float,
        previous_positions_max_distance: float,
        num_previous_position_distogram_bins: int,
    ) -> None:
        super().__init__()
        self.recycle_positions = recycle_positions
        self.recycle_features = recycle_features
        self.previous_positions_min_distance = previous_positions_min_distance
        self.previous_positions_max_distance = previous_positions_max_distance
        self.num_previous_position_distogram_bins = num_previous_position_distogram_bins
        self.previous_msa_first_row_norm = LayerNorm(msa_size)
        self.previous_pair_norm = LayerNorm(pair_size)
        self.previous_positions_projection = nn.Linear(num_previous_position_distogram_bins, pair_size)

    def forward(
        self,
        msa: Tensor,
        pair: Tensor,
        *,
        input: AlphaFold2Input,
        previous_positions: Tensor,
        previous_msa_first_row: Tensor,
        previous_pair: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if self.recycle_features:
            msa = self._add_previous_msa_first_row(msa, previous_msa_first_row)
            pair = pair + self.previous_pair_norm(previous_pair)
        if self.recycle_positions:
            model_slots = _gather_residue_type(_RESIDUE_TYPE_ATOM37_MASK, input.residue_type_ids, dtype=torch.bool)
            previous_positions = project_model_coordinates(
                previous_positions,
                model_slots & input.residue_mask.to(torch.bool)[..., None],
                dtype=pair.dtype,
                feature="AlphaFold2 recycled coordinates",
            )
            pair = pair + self.previous_positions_projection(
                self._previous_positions_distogram(input, previous_positions, pair.dtype)
            )
        return msa, pair

    def _add_previous_msa_first_row(self, msa: Tensor, previous_msa_first_row: Tensor) -> Tensor:
        first_row = msa.select(dim=-3, index=0) + self.previous_msa_first_row_norm(previous_msa_first_row)
        if msa.shape[-3] == 1:
            return first_row.unsqueeze(-3)
        return torch.cat([first_row.unsqueeze(-3), msa.narrow(-3, 1, msa.shape[-3] - 1)], dim=-3)

    def _previous_positions_distogram(
        self,
        input: AlphaFold2Input,
        previous_positions: Tensor,
        dtype: torch.dtype,
    ) -> Tensor:
        representative_positions = _pseudo_beta(input.residue_type_ids, previous_positions).to(dtype=dtype)
        return coordinate_distogram(
            representative_positions,
            min_distance=self.previous_positions_min_distance,
            max_distance=self.previous_positions_max_distance,
            num_bins=self.num_previous_position_distogram_bins,
        )


class AlphaFold2SingleTemplateEmbedder(nn.Module):
    r"""
    Embed and refine each template with the shared template pair stack.
    """

    def __init__(
        self,
        *,
        template_size: int,
        num_blocks: int,
        num_distogram_bins: int,
        min_distance: float,
        max_distance: float,
        vocab_size: int,
        use_template_unit_vector: bool,
        triangle_multiplication_hidden_size: int,
        triangle_attention_head_size: int,
        num_triangle_attention_heads: int,
        transition_expansion: int,
        dropout: float,
        backend: str,
    ) -> None:
        super().__init__()
        self.backend = backend
        self.num_distogram_bins = num_distogram_bins
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.vocab_size = vocab_size
        self.use_template_unit_vector = use_template_unit_vector
        template_feature_size = num_distogram_bins + 1 + 2 * vocab_size + 3 + 1
        self.template_pair_projection = nn.Linear(template_feature_size, template_size)
        self.template_pair_stack = AlphaFold2TemplatePairStack(
            num_blocks=num_blocks,
            pair_size=template_size,
            triangle_multiplication_hidden_size=triangle_multiplication_hidden_size,
            triangle_attention_head_size=triangle_attention_head_size,
            num_triangle_attention_heads=num_triangle_attention_heads,
            transition_expansion=transition_expansion,
            dropout=dropout,
            backend=backend,
        )
        self.output_norm = LayerNorm(template_size, backend=backend)

    def forward(
        self,
        template_batch: AlphaFold2TemplateBatch,
        *,
        pair_mask: Tensor,
        template_pair_mask: Tensor | None = None,
    ) -> Tensor:
        template_batch = _normalize_alphafold2_template_batch(
            template_batch, dtype=self.template_pair_projection.weight.dtype
        )
        features, feature_pair_mask = self._features(template_batch)
        batch_size, num_templates, num_tokens = features.shape[:3]
        pair_mask = pair_mask[:, None].expand(batch_size, num_templates, num_tokens, num_tokens)
        if template_pair_mask is not None:
            template_pair_mask = template_pair_mask[:, None].expand(
                batch_size,
                num_templates,
                num_tokens,
                num_tokens,
            )
            features = features * template_pair_mask.unsqueeze(-1)
        pair_mask = pair_mask * feature_pair_mask

        act = self.template_pair_projection(features).reshape(batch_size * num_templates, num_tokens, num_tokens, -1)
        flat_pair_mask = pair_mask.reshape(batch_size * num_templates, num_tokens, num_tokens)
        act = self.template_pair_stack(act, pair_mask=flat_pair_mask)
        act = self.output_norm(act)
        return act.reshape(batch_size, num_templates, num_tokens, num_tokens, -1)

    def _features(self, template_batch: AlphaFold2TemplateBatch) -> tuple[Tensor, Tensor]:
        residue_type_ids = project_model_categories(
            template_batch.template_residue_type_ids,
            template_batch.template_mask.bool()[..., None],
            num_classes=self.vocab_size,
            feature="AlphaFold2 template classes",
        )
        residue_type_one_hot = F.one_hot(residue_type_ids, num_classes=self.vocab_size).to(dtype=template_batch.dtype)
        positions = template_batch.template_all_atom_positions
        atom_mask = template_batch.template_all_atom_mask.to(dtype=template_batch.dtype)
        pseudo_beta = _pseudo_beta(template_batch.template_residue_type_ids, positions)
        pseudo_beta_mask = _pseudo_beta(template_batch.template_residue_type_ids, atom_mask.unsqueeze(-1)).squeeze(-1)
        template_pair_mask = pseudo_beta_mask.unsqueeze(-1) * pseudo_beta_mask.unsqueeze(-2)
        distogram = coordinate_distogram(
            pseudo_beta,
            min_distance=self.min_distance,
            max_distance=self.max_distance,
            num_bins=self.num_distogram_bins,
        ).to(dtype=template_batch.dtype)

        residue_j = residue_type_one_hot.unsqueeze(-3).expand(*template_pair_mask.shape, self.vocab_size)
        residue_i = residue_type_one_hot.unsqueeze(-2).expand(*template_pair_mask.shape, self.vocab_size)
        unit_vector, backbone_mask = _backbone_unit_vectors(positions, atom_mask, backend=self.backend)
        backbone_pair_mask = backbone_mask.unsqueeze(-1) * backbone_mask.unsqueeze(-2)
        if not self.use_template_unit_vector:
            unit_vector = torch.zeros_like(unit_vector)
        features = torch.cat(
            [
                distogram,
                template_pair_mask.unsqueeze(-1),
                residue_j,
                residue_i,
                unit_vector,
                backbone_pair_mask.unsqueeze(-1),
            ],
            dim=-1,
        )
        features = features * backbone_pair_mask.unsqueeze(-1)
        return features, backbone_pair_mask


class AlphaFold2MultimerSingleTemplateEmbedder(nn.Module):
    r"""
    Embed one AF-Multimer template into pair space.
    """

    def __init__(
        self,
        *,
        query_size: int,
        template_size: int,
        num_blocks: int,
        num_distogram_bins: int,
        min_distance: float,
        max_distance: float,
        vocab_size: int,
        triangle_multiplication_hidden_size: int,
        triangle_attention_head_size: int,
        num_triangle_attention_heads: int,
        transition_expansion: int,
        dropout: float,
        backend: str,
    ) -> None:
        super().__init__()
        self.num_distogram_bins = num_distogram_bins
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.vocab_size = vocab_size
        self.distogram_projection = nn.Linear(num_distogram_bins, template_size)
        self.pseudo_beta_mask_projection = nn.Linear(1, template_size)
        self.column_residue_type_projection = nn.Linear(vocab_size, template_size)
        self.row_residue_type_projection = nn.Linear(vocab_size, template_size)
        self.unit_vector_x_projection = nn.Linear(1, template_size)
        self.unit_vector_y_projection = nn.Linear(1, template_size)
        self.unit_vector_z_projection = nn.Linear(1, template_size)
        self.backbone_pair_mask_projection = nn.Linear(1, template_size)
        self.query_embedding_norm = LayerNorm(query_size, backend=backend)
        self.query_pair_projection = nn.Linear(query_size, template_size)
        self.template_pair_stack = AlphaFold2TemplatePairStack(
            block=_multimer_template_pair_block,
            num_blocks=num_blocks,
            pair_size=template_size,
            triangle_multiplication_hidden_size=triangle_multiplication_hidden_size,
            triangle_attention_head_size=triangle_attention_head_size,
            num_triangle_attention_heads=num_triangle_attention_heads,
            transition_expansion=transition_expansion,
            dropout=dropout,
            backend=backend,
        )
        self.output_norm = LayerNorm(template_size, backend=backend)

    def forward(
        self,
        query_embedding: Tensor,
        *,
        residue_type_ids: Tensor,
        pseudo_beta: Tensor,
        pseudo_beta_mask: Tensor,
        unit_vector: Tensor,
        backbone_mask: Tensor,
        multichain_mask_2d: Tensor,
        pair_mask: Tensor,
    ) -> Tensor:
        dtype = query_embedding.dtype
        pseudo_beta = project_model_coordinates(
            pseudo_beta, pseudo_beta_mask.to(torch.bool), dtype=dtype, feature="AlphaFold2 multimer pseudo-beta"
        )
        vector_mask = backbone_mask.to(torch.bool).unsqueeze(-1) & backbone_mask.to(torch.bool).unsqueeze(-2)
        vector_mask = vector_mask & multichain_mask_2d.to(torch.bool)
        unit_vector = project_model_features(
            unit_vector, vector_mask[..., None], dtype=dtype, feature="AlphaFold2 multimer frame vectors"
        )
        pseudo_beta_mask_2d = pseudo_beta_mask.unsqueeze(-1) * pseudo_beta_mask.unsqueeze(-2)
        pseudo_beta_mask_2d = pseudo_beta_mask_2d * multichain_mask_2d
        distogram = coordinate_distogram(
            pseudo_beta,
            min_distance=self.min_distance,
            max_distance=self.max_distance,
            num_bins=self.num_distogram_bins,
        ).to(dtype=dtype)
        distogram = distogram * pseudo_beta_mask_2d.unsqueeze(-1)
        act = self.distogram_projection(distogram)
        act = act + self.pseudo_beta_mask_projection(pseudo_beta_mask_2d.unsqueeze(-1).to(dtype))

        residue_type_one_hot = F.one_hot(
            residue_type_ids.long().clamp(min=0, max=self.vocab_size - 1), self.vocab_size
        ).to(dtype)
        act = act + self.column_residue_type_projection(residue_type_one_hot[..., None, :, :])
        act = act + self.row_residue_type_projection(residue_type_one_hot[..., None, :])

        backbone_mask_2d = backbone_mask.unsqueeze(-1) * backbone_mask.unsqueeze(-2)
        backbone_mask_2d = backbone_mask_2d * multichain_mask_2d
        x, y, z = (unit_vector * backbone_mask_2d.unsqueeze(-1)).to(dtype).unbind(dim=-1)
        act = act + self.unit_vector_x_projection(x.unsqueeze(-1))
        act = act + self.unit_vector_y_projection(y.unsqueeze(-1))
        act = act + self.unit_vector_z_projection(z.unsqueeze(-1))
        act = act + self.backbone_pair_mask_projection(backbone_mask_2d.unsqueeze(-1).to(dtype))

        query = self.query_embedding_norm(query_embedding)
        act = act + self.query_pair_projection(query)

        act = self.template_pair_stack(act, pair_mask=pair_mask)
        return self.output_norm(act)


class AlphaFold2TemplateMsaEmbedder(nn.Module):
    r"""
    Embed template torsion features as extra MSA rows.
    """

    def __init__(self, *, vocab_size: int, msa_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        template_feature_size = vocab_size + 14 + 14 + 7
        self.feed_forward = FeedForward(template_feature_size, msa_size, msa_size, activation="relu")

    def forward(self, input: AlphaFold2Input) -> tuple[Tensor, Tensor]:
        input = normalize_alphafold2_input(input, dtype=self.feed_forward.input_projection.weight.dtype)
        template_batch = _template_batch(input)
        features, torsion_mask = self._features(template_batch)
        activations = self.feed_forward(features)
        msa_mask = torsion_mask[..., 2] * template_batch.template_mask.unsqueeze(-1).to(dtype=torsion_mask.dtype)
        return activations, msa_mask

    def _features(self, template_batch: AlphaFold2TemplateBatch) -> tuple[Tensor, Tensor]:
        residue_type_ids = project_model_categories(
            template_batch.template_residue_type_ids,
            template_batch.template_mask.bool()[..., None],
            num_classes=self.vocab_size,
            feature="AlphaFold2 template classes",
        )
        residue_type_one_hot = F.one_hot(residue_type_ids, num_classes=self.vocab_size).to(dtype=template_batch.dtype)
        torsion = template_batch.template_torsion_angles_sin_cos
        alternative_torsion = template_batch.template_alternative_torsion_angles_sin_cos
        torsion_mask = template_batch.template_torsion_angles_mask
        features = torch.cat(
            [
                residue_type_one_hot,
                torsion.reshape(*residue_type_ids.shape, 14).to(dtype=residue_type_one_hot.dtype),
                alternative_torsion.reshape(*residue_type_ids.shape, 14).to(dtype=residue_type_one_hot.dtype),
                torsion_mask.to(dtype=residue_type_one_hot.dtype),
            ],
            dim=-1,
        )
        return features, torsion_mask.to(dtype=residue_type_one_hot.dtype)


class AlphaFold2MultimerTemplateMsaEmbedder(nn.Module):
    r"""
    Embed AF-Multimer template chi features as MSA rows.
    """

    def __init__(self, *, msa_size: int, vocab_size: int = 22) -> None:
        super().__init__()
        self.backend = "auto"
        self.vocab_size = vocab_size
        template_feature_size = vocab_size + 4 + 4 + 4
        self.feed_forward = FeedForward(template_feature_size, msa_size, msa_size, activation="relu")

    def forward(self, input: AlphaFold2MultimerInput) -> tuple[Tensor, Tensor]:
        input = normalize_alphafold2_input(input, dtype=self.feed_forward.input_projection.weight.dtype)
        residue_type_ids = project_model_categories(
            input.template_residue_type_ids,
            input.template_mask.bool()[..., None] & input.residue_mask.bool().unsqueeze(-2),
            num_classes=self.vocab_size,
            feature="AlphaFold2 template classes",
        )
        residue_type_one_hot = F.one_hot(residue_type_ids, self.vocab_size).to(
            dtype=input.template_all_atom_positions.dtype
        )
        chi, chi_mask = _template_chi_angles(
            input.template_all_atom_positions, input.template_all_atom_mask, residue_type_ids, backend=self.backend
        )
        features = torch.cat(
            [residue_type_one_hot, torch.sin(chi) * chi_mask, torch.cos(chi) * chi_mask, chi_mask],
            dim=-1,
        ).to(dtype=self.feed_forward.input_projection.weight.dtype)
        activations = self.feed_forward(features)
        template_row_mask = chi_mask[..., 0] * input.template_mask.unsqueeze(-1).to(dtype=chi_mask.dtype)
        return activations, template_row_mask


class AlphaFold2TemplatePairBlock(nn.Module):
    r"""
    Update a template pair representation with triangle operations.

    AlphaFold 2 monomer applies triangle attention before triangle multiplication.
    """

    def __init__(
        self,
        *,
        pair_size: int,
        triangle_multiplication_hidden_size: int,
        triangle_attention_head_size: int,
        num_triangle_attention_heads: int,
        transition_expansion: int,
        dropout: float,
        backend: str,
    ) -> None:
        super().__init__()
        self.triangle_attention_starting_node = TriangleAttention(
            pair_size=pair_size,
            head_size=triangle_attention_head_size,
            num_attention_heads=num_triangle_attention_heads,
            output_bias=True,
            gate_bias=True,
            gate_bias_init=1.0,
            backend=backend,
        )
        self.rowwise_dropout = StructuredDropout(dropout, broadcast_dims=-3)
        self.triangle_attention_ending_node = TriangleAttention(
            pair_size=pair_size,
            head_size=triangle_attention_head_size,
            num_attention_heads=num_triangle_attention_heads,
            output_bias=True,
            gate_bias=True,
            gate_bias_init=1.0,
            backend=backend,
        )
        self.triangle_multiplication_outgoing = TriangleMultiplication(
            pair_size=pair_size,
            hidden_size=triangle_multiplication_hidden_size,
            direction="outgoing",
            projection_bias=True,
            backend=backend,
        )
        self.triangle_multiplication_incoming = TriangleMultiplication(
            pair_size=pair_size,
            hidden_size=triangle_multiplication_hidden_size,
            direction="incoming",
            projection_bias=True,
            backend=backend,
        )
        self.transition = FeedForwardTransition(pair_size, expansion=transition_expansion, backend=backend)

    def forward(self, pair: Tensor, *, pair_mask: Tensor) -> Tensor:
        pair = pair + self.rowwise_dropout(self.triangle_attention_starting_node(pair, pair_mask))
        pair = pair + self._ending_node(pair, pair_mask)
        pair = pair + self.rowwise_dropout(self.triangle_multiplication_outgoing(pair, pair_mask))
        pair = pair + self.rowwise_dropout(self.triangle_multiplication_incoming(pair, pair_mask))
        return pair + self.transition(pair, pair_mask)

    def _ending_node(self, pair: Tensor, pair_mask: Tensor) -> Tensor:
        # TriangleAttention preserves the input residue-axis orientation. Ending-node attention
        # therefore transposes the pair and mask residue axes around the call, then restores them.
        end_pair_mask = pair_mask.transpose(-2, -1)
        ending_update = self.triangle_attention_ending_node(pair.transpose(-3, -2), end_pair_mask)
        return self.rowwise_dropout(ending_update).transpose(-3, -2)


def _multimer_template_pair_block(
    *,
    pair_size: int,
    triangle_multiplication_hidden_size: int,
    triangle_attention_head_size: int,
    num_triangle_attention_heads: int,
    transition_expansion: int,
    dropout: float,
    backend: str,
) -> EvoformerPairBlock:
    return EvoformerPairBlock(
        pair_size=pair_size,
        triangle_multiplication_outgoing_hidden_size=triangle_multiplication_hidden_size,
        triangle_multiplication_incoming_hidden_size=triangle_multiplication_hidden_size,
        triangle_attention_head_size=triangle_attention_head_size,
        num_triangle_attention_starting_node_heads=num_triangle_attention_heads,
        num_triangle_attention_ending_node_heads=num_triangle_attention_heads,
        transition_expansion=transition_expansion,
        dropout=dropout,
        backend=backend,
    )


class AlphaFold2TemplatePairStack(nn.Module):
    r"""
    Homogeneous stack of AF2 template-pair blocks.
    """

    def __init__(
        self,
        *,
        num_blocks: int,
        pair_size: int,
        triangle_multiplication_hidden_size: int,
        triangle_attention_head_size: int,
        num_triangle_attention_heads: int,
        transition_expansion: int,
        dropout: float,
        backend: str,
        block: Callable[..., nn.Module] = AlphaFold2TemplatePairBlock,
    ) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.gradient_checkpointing = False
        self.blocks = nn.ModuleList(
            [
                block(
                    pair_size=pair_size,
                    triangle_multiplication_hidden_size=triangle_multiplication_hidden_size,
                    triangle_attention_head_size=triangle_attention_head_size,
                    num_triangle_attention_heads=num_triangle_attention_heads,
                    transition_expansion=transition_expansion,
                    dropout=dropout,
                    backend=backend,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, pair: Tensor, *, pair_mask: Tensor) -> Tensor:
        for block in self.blocks:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                pair = checkpoint(block, pair, pair_mask=pair_mask, use_reentrant=False)
            else:
                pair = block(pair, pair_mask=pair_mask)
        return pair


class AlphaFold2StructureModuleBlock(nn.Module):
    r"""
    One shared-weight AF2 structure-module iteration.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        pair_size: int,
        num_transition_projections: int,
        num_attention_heads: int,
        scalar_query_key_size: int,
        scalar_value_size: int,
        num_point_qk: int,
        num_point_v: int,
        sidechain_hidden_size: int,
        num_sidechain_blocks: int,
        position_scale: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.position_scale = position_scale
        self.invariant_point_attention = AlphaFold2InvariantPointAttention(
            hidden_size=hidden_size,
            pair_size=pair_size,
            num_attention_heads=num_attention_heads,
            scalar_query_key_size=scalar_query_key_size,
            scalar_value_size=scalar_value_size,
            num_point_qk=num_point_qk,
            num_point_v=num_point_v,
        )
        self.dropout = nn.Dropout(dropout)
        self.attention_norm = LayerNorm(hidden_size)
        if num_transition_projections <= 0:
            self.transition: nn.Module = nn.Identity()
        elif num_transition_projections == 1:
            self.transition = nn.Linear(hidden_size, hidden_size)
        else:
            self.transition = FeedForward(
                hidden_size,
                hidden_size,
                num_hidden_layers=num_transition_projections - 1,
                activation="relu",
            )
        final_projection = (
            self.transition.output_projection if isinstance(self.transition, FeedForward) else self.transition
        )
        if isinstance(final_projection, nn.Linear):
            nn.init.zeros_(final_projection.weight)
            nn.init.zeros_(final_projection.bias)
        self.transition_norm = LayerNorm(hidden_size)
        self.backbone_update = nn.Linear(hidden_size, 6)
        self.sidechain = AlphaFold2Sidechain(
            hidden_size=hidden_size,
            sidechain_hidden_size=sidechain_hidden_size,
            num_blocks=num_sidechain_blocks,
        )
        nn.init.zeros_(self.backbone_update.weight)
        nn.init.zeros_(self.backbone_update.bias)

    def forward(
        self,
        *,
        act: Tensor,
        pair: Tensor,
        mask: Tensor,
        affine: AlphaFold2QuatAffine,
        initial_act: Tensor,
        residue_type_ids: Tensor,
    ) -> tuple[Tensor, AlphaFold2QuatAffine, Tensor, Tensor, AlphaFold2Rigid, Tensor, AlphaFold2QuatAffine]:
        attn = self.invariant_point_attention(inputs_1d=act, inputs_2d=pair, mask=mask, affine=affine)
        act = self.attention_norm(act + self.dropout(attn))
        act = self.transition_norm(act + self.dropout(self.transition(act)))
        affine = affine.pre_compose(self.backbone_update(act))
        angles, unnormalized_angles, sidechain_frames, atom14_positions = self.sidechain(
            affine.scale_translation(self.position_scale),
            (act, initial_act),
            residue_type_ids,
        )
        return (
            act,
            affine.detach_rotation(),
            angles,
            unnormalized_angles,
            sidechain_frames,
            atom14_positions,
            affine,
        )


class AlphaFold2MultimerStructureModuleBlock(nn.Module):
    r"""
    One shared-weight AF-Multimer structure-module iteration.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        pair_size: int,
        num_transition_projections: int,
        num_attention_heads: int,
        scalar_query_key_size: int,
        scalar_value_size: int,
        num_point_qk: int,
        num_point_v: int,
        sidechain_hidden_size: int,
        num_sidechain_blocks: int,
        position_scale: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.position_scale = position_scale
        self.invariant_point_attention = AlphaFold2MultimerInvariantPointAttention(
            hidden_size=hidden_size,
            pair_size=pair_size,
            num_attention_heads=num_attention_heads,
            scalar_query_key_size=scalar_query_key_size,
            scalar_value_size=scalar_value_size,
            num_point_qk=num_point_qk,
            num_point_v=num_point_v,
        )
        self.dropout = nn.Dropout(dropout)
        self.attention_norm = LayerNorm(hidden_size)
        if num_transition_projections <= 0:
            self.transition: nn.Module = nn.Identity()
        elif num_transition_projections == 1:
            self.transition = nn.Linear(hidden_size, hidden_size)
        else:
            self.transition = FeedForward(
                hidden_size,
                hidden_size,
                num_hidden_layers=num_transition_projections - 1,
                activation="relu",
            )
        final_projection = (
            self.transition.output_projection if isinstance(self.transition, FeedForward) else self.transition
        )
        if isinstance(final_projection, nn.Linear):
            nn.init.zeros_(final_projection.weight)
            nn.init.zeros_(final_projection.bias)
        self.transition_norm = LayerNorm(hidden_size)
        self.backbone_update = AlphaFold2MultimerBackboneUpdate(hidden_size)
        self.sidechain = AlphaFold2Sidechain(
            hidden_size=hidden_size,
            sidechain_hidden_size=sidechain_hidden_size,
            num_blocks=num_sidechain_blocks,
        )

    def forward(
        self,
        *,
        act: Tensor,
        pair: Tensor,
        mask: Tensor,
        rigid: AlphaFold2Rigid,
        initial_act: Tensor,
        residue_type_ids: Tensor,
    ) -> tuple[Tensor, AlphaFold2Rigid, Tensor, Tensor, AlphaFold2Rigid, Tensor]:
        attn = self.invariant_point_attention(single=act, pair=pair, mask=mask, rigid=rigid)
        act = self.attention_norm(act + self.dropout(attn))
        act = self.transition_norm(act + self.dropout(self.transition(act)))
        rigid = rigid.compose(self.backbone_update(act))
        angles, unnormalized_angles, sidechain_frames, atom14_positions = self.sidechain(
            rigid.scale_translation(self.position_scale),
            (act, initial_act),
            residue_type_ids,
        )
        return act, rigid, angles, unnormalized_angles, sidechain_frames, atom14_positions


class AlphaFold2InvariantPointAttention(nn.Module):
    r"""
    AF2 invariant point attention.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        pair_size: int,
        num_attention_heads: int,
        scalar_query_key_size: int,
        scalar_value_size: int,
        num_point_qk: int,
        num_point_v: int,
        dist_epsilon: float = 1e-8,
    ) -> None:
        super().__init__()
        self.backend = "auto"
        self.num_attention_heads = num_attention_heads
        self.scalar_query_key_size = scalar_query_key_size
        self.scalar_value_size = scalar_value_size
        self.num_point_qk = num_point_qk
        self.num_point_v = num_point_v
        self.dist_epsilon = dist_epsilon
        self.query_scalar_projection = nn.Linear(hidden_size, num_attention_heads * scalar_query_key_size)
        self.key_value_scalar_projection = nn.Linear(
            hidden_size, num_attention_heads * (scalar_value_size + scalar_query_key_size)
        )
        self.query_point_projection = nn.Linear(hidden_size, num_attention_heads * 3 * num_point_qk)
        self.key_value_point_projection = nn.Linear(hidden_size, num_attention_heads * 3 * (num_point_qk + num_point_v))
        self.trainable_point_weights = nn.Parameter(torch.empty(num_attention_heads))
        self.pair_bias_projection = nn.Linear(pair_size, num_attention_heads)
        output_size = (
            num_attention_heads * scalar_value_size
            + 4 * num_attention_heads * num_point_v
            + num_attention_heads * pair_size
        )
        self.output_projection = nn.Linear(output_size, hidden_size)
        nn.init.constant_(self.trainable_point_weights, math.log(math.exp(1.0) - 1.0))
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, *, inputs_1d: Tensor, inputs_2d: Tensor, mask: Tensor, affine: AlphaFold2QuatAffine) -> Tensor:
        num_attention_heads = self.num_attention_heads
        q_scalar = self.query_scalar_projection(inputs_1d).unflatten(
            -1, (num_attention_heads, self.scalar_query_key_size)
        )
        kv_scalar = self.key_value_scalar_projection(inputs_1d).unflatten(
            -1,
            (num_attention_heads, self.scalar_value_size + self.scalar_query_key_size),
        )
        k_scalar, v_scalar = kv_scalar.split((self.scalar_query_key_size, self.scalar_value_size), dim=-1)

        q_point_local = (
            self.query_point_projection(inputs_1d)
            .unflatten(-1, (3, num_attention_heads, self.num_point_qk))
            .movedim(-3, -1)
        )
        q_point = affine.apply_to_point(q_point_local, extra_dims=2)
        kv_point_local = (
            self.key_value_point_projection(inputs_1d)
            .unflatten(-1, (3, num_attention_heads, self.num_point_qk + self.num_point_v))
            .movedim(-3, -1)
        )
        kv_point = affine.apply_to_point(kv_point_local, extra_dims=2)
        k_point, v_point = kv_point.split((self.num_point_qk, self.num_point_v), dim=-2)

        scalar_weights = math.sqrt(1.0 / (3.0 * max(self.scalar_query_key_size, 1)))
        q_scalar = scalar_weights * q_scalar
        attn_qk_scalar = torch.einsum("...qhd,...khd->...hqk", q_scalar, k_scalar)

        point_weight_scale = math.sqrt(1.0 / (3.0 * max(self.num_point_qk, 1) * 9.0 / 2.0))
        point_weights = point_weight_scale * F.softplus(self.trainable_point_weights)
        q_point = q_point.transpose(-4, -3)
        k_point = k_point.transpose(-4, -3)
        dist2 = (q_point.unsqueeze(-3) - k_point.unsqueeze(-4)).square().sum(dim=-1)
        attn_qk_point = -0.5 * (dist2 * _expand_head_weights(point_weights, dist2)).sum(dim=-1)

        attention_2d_weights = math.sqrt(1.0 / 3.0)
        attention_2d = attention_2d_weights * self.pair_bias_projection(inputs_2d).movedim(-1, -3)
        attn_logits = attn_qk_scalar + attn_qk_point + attention_2d
        pair_mask = mask.to(dtype=torch.bool).unsqueeze(-1) & mask.to(dtype=torch.bool).unsqueeze(-2)
        attn_logits = attn_logits.masked_fill(~pair_mask.unsqueeze(-3), torch.finfo(attn_logits.dtype).min)
        attn = torch.softmax(attn_logits, dim=-1)

        v = v_scalar.transpose(-3, -2)
        result_scalar = torch.einsum("...hqk,...hkd->...hqd", attn, v).transpose(-3, -2).flatten(-2)
        v_point = v_point.transpose(-4, -3)
        result_point = torch.einsum("...hqk,...hkpc->...hqpc", attn, v_point).transpose(-4, -3)
        result_point = affine.invert_point(result_point, extra_dims=2)
        result_point_norm = smoothed_vector_norm(result_point, eps=self.dist_epsilon, backend=self.backend)
        result_attention_over_2d = torch.einsum("...hqk,...qkc->...qhc", attn, inputs_2d).flatten(-2)
        output_features = torch.cat(
            [
                result_scalar,
                result_point[..., 0].flatten(-2),
                result_point[..., 1].flatten(-2),
                result_point[..., 2].flatten(-2),
                result_point_norm.flatten(-2),
                result_attention_over_2d,
            ],
            dim=-1,
        )
        return self.output_projection(output_features)


class AlphaFold2MultimerInvariantPointAttention(nn.Module):
    r"""
    AF-Multimer invariant point attention (Algorithm 22, multimer variant).
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        pair_size: int,
        num_attention_heads: int,
        scalar_query_key_size: int,
        scalar_value_size: int,
        num_point_qk: int,
        num_point_v: int,
        inf: float = 1e5,
    ) -> None:
        super().__init__()
        self.backend = "auto"
        self.num_attention_heads = num_attention_heads
        self.scalar_query_key_size = scalar_query_key_size
        self.scalar_value_size = scalar_value_size
        self.num_point_qk = num_point_qk
        self.num_point_v = num_point_v
        self.inf = inf
        self.trainable_point_weights = nn.Parameter(torch.zeros(num_attention_heads))
        self.query_point_projection = AlphaFold2MultimerPointProjection(hidden_size, num_point_qk, num_attention_heads)
        self.key_point_projection = AlphaFold2MultimerPointProjection(hidden_size, num_point_qk, num_attention_heads)
        self.query_scalar_projection = nn.Linear(hidden_size, num_attention_heads * scalar_query_key_size, bias=False)
        self.key_scalar_projection = nn.Linear(hidden_size, num_attention_heads * scalar_query_key_size, bias=False)
        self.pair_bias_projection = nn.Linear(pair_size, num_attention_heads)
        self.value_scalar_projection = nn.Linear(hidden_size, num_attention_heads * scalar_value_size, bias=False)
        self.value_point_projection = AlphaFold2MultimerPointProjection(hidden_size, num_point_v, num_attention_heads)
        output_size = num_attention_heads * (pair_size + scalar_value_size + num_point_v * 4)
        self.output_projection = nn.Linear(output_size, hidden_size)

    def forward(self, *, single: Tensor, pair: Tensor, mask: Tensor, rigid: AlphaFold2Rigid) -> Tensor:
        num_attention_heads = self.num_attention_heads

        point_variance = max(self.num_point_qk, 1) * 9.0 / 2.0
        point_weight_scale = math.sqrt(1.0 / point_variance)
        head_weights = F.softplus(self.trainable_point_weights)
        point_weights = point_weight_scale * head_weights

        q_point = self.query_point_projection(single, rigid)
        k_point = self.key_point_projection(single, rigid)
        # [*, Ni, Nj, H, P]
        dist2 = (q_point.unsqueeze(-4) - k_point.unsqueeze(-5)).square().sum(dim=-1)
        attn_qk_point = -0.5 * (dist2 * point_weights[..., None, :, None]).sum(dim=-1)

        scalar_weights = math.sqrt(1.0 / max(self.scalar_query_key_size, 1))
        q_scalar = self.query_scalar_projection(single).view(
            *single.shape[:-1], num_attention_heads, self.scalar_query_key_size
        )
        k_scalar = self.key_scalar_projection(single).view(
            *single.shape[:-1], num_attention_heads, self.scalar_query_key_size
        )
        q_scalar = q_scalar * scalar_weights
        attn_qk_scalar = torch.einsum("...qhc,...khc->...qkh", q_scalar, k_scalar)

        attn_logits = attn_qk_scalar + attn_qk_point + self.pair_bias_projection(pair)

        square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        square_mask = self.inf * (square_mask - 1.0)
        attn_logits = attn_logits + square_mask.unsqueeze(-1)
        attn_logits = attn_logits * math.sqrt(1.0 / 3.0)
        # softmax over the key axis (Nj), which is dim -2 in the [*, Ni, Nj, H] layout.
        attn = torch.softmax(attn_logits, dim=-2)

        v_scalar = self.value_scalar_projection(single).view(
            *single.shape[:-1], num_attention_heads, self.scalar_value_size
        )
        result_scalar = torch.einsum("...qkh,...khc->...qhc", attn, v_scalar)
        result_scalar = result_scalar.reshape(*result_scalar.shape[:-2], num_attention_heads * self.scalar_value_size)

        v_point = self.value_point_projection(single, rigid)
        # [*, Ni, H, P_v, 3]
        result_point = (attn[..., None, None] * v_point.unsqueeze(-5)).sum(dim=-4)
        result_point = result_point.reshape(*result_point.shape[:-3], num_attention_heads * self.num_point_v, 3)
        result_point = apply_inverse_rigid_transform(
            result_point,
            rigid.rotation,
            rigid.translation,
        )
        result_point_norm = torch.linalg.vector_norm(result_point, dim=-1).clamp_min(1e-8)

        result_pair = torch.einsum("...qkh,...qkc->...qhc", attn, pair)
        result_pair = result_pair.reshape(*result_pair.shape[:-2], num_attention_heads * pair.shape[-1])

        output_features = torch.cat(
            [
                result_scalar,
                result_point[..., 0],
                result_point[..., 1],
                result_point[..., 2],
                result_point_norm,
                result_pair,
            ],
            dim=-1,
        )
        return self.output_projection(output_features)


class AlphaFold2MultimerPointProjection(nn.Module):
    r"""
    Project AF-Multimer per-head points into global backbone frames.
    """

    def __init__(self, hidden_size: int, num_points: int, num_attention_heads: int) -> None:
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.num_points = num_points
        self.point_projection = nn.Linear(hidden_size, num_attention_heads * 3 * num_points)

    def forward(self, single: Tensor, rigid: AlphaFold2Rigid) -> Tensor:
        points_local = self.point_projection(single)
        points_local = points_local.view(*points_local.shape[:-1], self.num_attention_heads, 3, self.num_points)
        points_local = points_local.movedim(-2, -1)
        return apply_rigid_transform(
            points_local,
            rigid.rotation[..., :, None, :, :],
            rigid.translation[..., :, None, :],
        )


class AlphaFold2MultimerBackboneUpdate(nn.Module):
    r"""
    Predict one AF-Multimer backbone rigid update.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.projection = nn.Linear(hidden_size, 6)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, single: Tensor) -> AlphaFold2Rigid:
        rigid_flat = self.projection(single)
        qx, qy, qz, tx, ty, tz = rigid_flat.unbind(dim=-1)
        quaternion = torch.stack([torch.ones_like(qx), qx, qy, qz], dim=-1)
        rotation = quaternion_to_rotation(quaternion)
        translation = torch.stack([tx, ty, tz], dim=-1)
        return AlphaFold2Rigid(rotation=rotation, translation=translation)


class AlphaFold2Sidechain(nn.Module):
    r"""
    AF2 side-chain torsion and atom14 coordinate generator.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        sidechain_hidden_size: int,
        num_blocks: int,
    ) -> None:
        super().__init__()
        self.input_projections = nn.ModuleList(nn.Linear(hidden_size, sidechain_hidden_size) for _ in range(2))
        self.num_blocks = num_blocks
        self.blocks = nn.ModuleList(AlphaFold2SidechainBlock(sidechain_hidden_size) for _ in range(num_blocks))
        self.angle_projection = nn.Linear(sidechain_hidden_size, 14)

    def forward(
        self,
        affine: AlphaFold2QuatAffine,
        representations: tuple[Tensor, Tensor],
        residue_type_ids: Tensor,
    ) -> tuple[Tensor, Tensor, AlphaFold2Rigid, Tensor]:
        act = sum(proj(F.relu(rep)) for proj, rep in zip(self.input_projections, representations, strict=True))
        for block in self.blocks:
            act = block(act)
        unnormalized_angles = self.angle_projection(F.relu(act)).unflatten(-1, (7, 2))
        angles = normalize_vectors(
            torch.where(
                unnormalized_angles.ne(0).any(dim=-1, keepdim=True),
                unnormalized_angles,
                unnormalized_angles.new_tensor((0, 1)),
            )
        )
        backbone = affine.to_rigid()
        rotation, translation = torsion_angles_to_frames(
            backbone.rotation,
            backbone.translation,
            angles,
            default_frames=_gather_residue_type(
                residue_constants.AMINO_ACID_RIGID_GROUP_DEFAULT_FRAME, residue_type_ids, dtype=angles.dtype
            ),
        )
        atom_pos = frames_to_atom14_positions(
            rotation,
            translation,
            atom14_to_rigid_group=_gather_residue_type(
                residue_constants.AMINO_ACID_ATOM14_TO_RIGID_GROUP, residue_type_ids, dtype=torch.long
            ),
            atom14_rigid_group_positions=_gather_residue_type(
                residue_constants.AMINO_ACID_ATOM14_RIGID_GROUP_POSITIONS, residue_type_ids, dtype=angles.dtype
            ),
            atom14_mask=_residue_type_atom14_mask(residue_type_ids, dtype=torch.bool),
        )
        return angles, unnormalized_angles, AlphaFold2Rigid(rotation=rotation, translation=translation), atom_pos


class AlphaFold2SidechainBlock(nn.Module):
    r"""
    Refine side-chain activations with a two-layer residual block.

    The block uses ReLU pre-activations and a zero-initialized final projection.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.feed_forward = FeedForward(hidden_size, hidden_size, activation="relu")
        nn.init.zeros_(self.feed_forward.output_projection.weight)
        nn.init.zeros_(self.feed_forward.output_projection.bias)

    def forward(self, act: Tensor) -> Tensor:
        return act + self.feed_forward(F.relu(act))


@dataclass(eq=False)
class AlphaFold2Rigid:
    r"""
    Batched rigid transform represented by rotation matrices and translations.
    """

    rotation: Tensor
    translation: Tensor

    @classmethod
    def from_tensor4x4(cls, tensor: Tensor) -> AlphaFold2Rigid:
        return cls(rotation=tensor[..., :3, :3], translation=tensor[..., :3, 3])

    def apply(self, points: Tensor) -> Tensor:
        return apply_rigid_transform(points.unsqueeze(-2), self.rotation, self.translation).squeeze(-2)

    def to_tensor4x4(self) -> Tensor:
        top = torch.cat([self.rotation, self.translation.unsqueeze(-1)], dim=-1)
        bottom = torch.cat(
            [
                self.translation.new_zeros((*self.translation.shape[:-1], 1, 3)),
                self.translation.new_ones((*self.translation.shape[:-1], 1, 1)),
            ],
            dim=-1,
        )
        return torch.cat([top, bottom], dim=-2)

    def compose(self, other: AlphaFold2Rigid) -> AlphaFold2Rigid:
        rotation = self.rotation @ other.rotation
        translation = self.apply(other.translation)
        return AlphaFold2Rigid(rotation=rotation, translation=translation)

    def compose_rotation(self, rotation: Tensor) -> AlphaFold2Rigid:
        return AlphaFold2Rigid(rotation=self.rotation @ rotation, translation=self.translation)

    def with_group_axis(self) -> AlphaFold2Rigid:
        return AlphaFold2Rigid(rotation=self.rotation.unsqueeze(-3), translation=self.translation.unsqueeze(-2))

    @classmethod
    def identity(cls, shape: tuple[int, ...], *, device: torch.device, dtype: torch.dtype) -> AlphaFold2Rigid:
        rotation = torch.eye(3, device=device, dtype=dtype).expand(*shape, 3, 3).clone()
        translation = torch.zeros(*shape, 3, device=device, dtype=dtype)
        return cls(rotation=rotation, translation=translation)

    def to_rigid(self) -> AlphaFold2Rigid:
        # Adapter so [`AlphaFold2Sidechain`][] (which calls ``affine.to_rigid()``) can
        # consume a bare rotation-matrix rigid from the multimer structure module.
        return self

    def scale_translation(self, position_scale: float) -> AlphaFold2Rigid:
        return AlphaFold2Rigid(rotation=self.rotation, translation=self.translation * position_scale)

    def detach_rotation(self) -> AlphaFold2Rigid:
        return AlphaFold2Rigid(rotation=self.rotation.detach(), translation=self.translation)


@dataclass(eq=False)
class AlphaFold2QuatAffine:
    r"""
    AF2 backbone affine represented by a quaternion, rotation, and translation.
    """

    quaternion: Tensor
    rotation: Tensor
    translation: Tensor

    @classmethod
    def identity(
        cls, shape: torch.Size | tuple[int, ...], *, device: torch.device, dtype: torch.dtype
    ) -> AlphaFold2QuatAffine:
        quaternion = torch.zeros(*shape, 4, device=device, dtype=dtype)
        quaternion[..., 0] = 1.0
        rotation = torch.eye(3, device=device, dtype=dtype).expand(*shape, 3, 3).clone()
        translation = torch.zeros(*shape, 3, device=device, dtype=dtype)
        return cls(quaternion=quaternion, rotation=rotation, translation=translation)

    @classmethod
    def from_tensor(cls, tensor: Tensor, *, normalize: bool = False) -> AlphaFold2QuatAffine:
        r"""Read scalar-first quaternion and translation coefficients from ``(..., 7)``.

        ``normalize=False`` requires a unit quaternion. ``normalize=True``
        normalizes a nonzero quaternion before storing it and deriving rotation.
        """

        quaternion = tensor[..., :4]
        if normalize:
            quaternion = normalize_vectors(quaternion)
        return cls(
            quaternion=quaternion, rotation=unit_quaternion_to_rotation(quaternion), translation=tensor[..., 4:7]
        )

    def to_tensor(self) -> Tensor:
        return torch.cat([self.quaternion, self.translation], dim=-1)

    def to_rigid(self) -> AlphaFold2Rigid:
        return AlphaFold2Rigid(rotation=self.rotation, translation=self.translation)

    def scale_translation(self, position_scale: float) -> AlphaFold2QuatAffine:
        return AlphaFold2QuatAffine(
            quaternion=self.quaternion,
            rotation=self.rotation,
            translation=self.translation * position_scale,
        )

    def detach_rotation(self) -> AlphaFold2QuatAffine:
        return AlphaFold2QuatAffine(
            quaternion=self.quaternion.detach(),
            rotation=self.rotation.detach(),
            translation=self.translation,
        )

    def pre_compose(self, update: Tensor) -> AlphaFold2QuatAffine:
        vector_quaternion_update = update[..., :3]
        translation_update = update[..., 3:6]
        quaternion = self.quaternion + _quat_multiply_by_vec(self.quaternion, vector_quaternion_update)
        translation = self.translation + rotate_vectors(translation_update.unsqueeze(-2), self.rotation.mT).squeeze(-2)
        quaternion = normalize_vectors(quaternion)
        return AlphaFold2QuatAffine(
            quaternion=quaternion, rotation=unit_quaternion_to_rotation(quaternion), translation=translation
        )

    def apply_to_point(self, point: Tensor, *, extra_dims: int = 0) -> Tensor:
        rotation = self.rotation
        translation = self.translation
        for _ in range(extra_dims):
            rotation = rotation.unsqueeze(-3)
            translation = translation.unsqueeze(-2)
        return apply_rigid_transform(point.unsqueeze(-2), rotation, translation).squeeze(-2)

    def invert_point(self, transformed_point: Tensor, *, extra_dims: int = 0) -> Tensor:
        rotation = self.rotation
        translation = self.translation
        for _ in range(extra_dims):
            rotation = rotation.unsqueeze(-3)
            translation = translation.unsqueeze(-2)
        return apply_inverse_rigid_transform(transformed_point.unsqueeze(-2), rotation, translation).squeeze(-2)


class AlphaFold2PredictedLddtHead(nn.Module):
    r"""
    Predict per-residue pLDDT confidence-bin logits.
    """

    def __init__(self, *, input_size: int, hidden_size: int, num_bins: int) -> None:
        super().__init__()
        self.input_norm = LayerNorm(input_size)
        self.feed_forward = FeedForward(input_size, hidden_size, num_bins, num_hidden_layers=2, activation="relu")
        nn.init.zeros_(self.feed_forward.output_projection.weight)
        nn.init.zeros_(self.feed_forward.output_projection.bias)

    def forward(self, structure_module: Tensor) -> Tensor:
        return self.feed_forward(self.input_norm(structure_module))


class _AlphaFold2RecyclingInput(Protocol):
    r"""
    Tensor fields shared by monomer and multimer recycling.
    """

    residue_type_ids: Tensor
    token_features: Tensor
    residue_mask: Tensor
    msa_tokens: Tensor
    msa_deletion_counts: Tensor
    msa_profile: Tensor
    msa_features: Tensor | None
    msa_mask: Tensor | None
    extra_msa_features: Tensor | None
    extra_msa_mask: Tensor | None
    bert_mask: Tensor | None
    true_msa: Tensor | None


_AlphaFold2InputT = TypeVar("_AlphaFold2InputT", bound=_AlphaFold2RecyclingInput)
_MsaRealization = tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]


def _realize_msa_features(
    msa_tokens: Tensor,
    msa_deletion_counts: Tensor,
    msa_profile: Tensor,
    *,
    residue_mask: Tensor,
    max_msa_clusters: int,
    max_extra_msa: int,
    msa_cluster_features: bool,
    gap_agreement_weight: float,
    masked_msa_enabled: bool,
    uniform_prob: float,
    profile_prob: float,
    same_prob: float,
    replace_fraction: float,
    generator: torch.Generator | None,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    r"""
    Realize one AF2 MSA-feature ensemble slice (unbatched).

    The per-recycle sequence is MSA row sampling, BERT masking before clustering,
    nearest-neighbor clustering, cluster summarization, extra-MSA cropping, and feature
    assembly. Inputs are ``msa_tokens [M, N]`` (canonical MSA indices, row 0 the
    query), ``msa_deletion_counts [M, N]``, and the full-pool canonical profile
    ``msa_profile [N, 72]``. Returns ``msa_features [S, N, 147]``,
    ``msa_mask [S, N]``, ``extra_msa_features [E, N, 74]``, ``extra_msa_mask [E, N]``,
    ``bert_mask [S, N]`` and ``true_msa [S, N]``.
    """

    tokens = msa_tokens.to(device=device)
    deletions = msa_deletion_counts.to(device=device, dtype=torch.float64)
    profile = msa_profile.to(device=device, dtype=dtype)

    sel_tokens, sel_deletions, extra_tokens, extra_deletions = _sample_msa(
        tokens, deletions, max_msa_clusters, generator
    )
    msa_mask = residue_mask.to(device=device, dtype=dtype).expand_as(sel_tokens)
    extra_mask = residue_mask.to(device=device, dtype=dtype).expand_as(extra_tokens)

    if masked_msa_enabled:
        masked_tokens, bert_mask, true_msa = _make_masked_msa(
            sel_tokens,
            profile,
            uniform_prob=uniform_prob,
            profile_prob=profile_prob,
            same_prob=same_prob,
            replace_fraction=replace_fraction,
            generator=generator,
            device=device,
            dtype=dtype,
        )
    else:
        masked_tokens = sel_tokens
        bert_mask = torch.zeros(sel_tokens.shape, device=device, dtype=dtype)
        true_msa = sel_tokens

    if msa_cluster_features:
        assignment = _nearest_neighbor_clusters(
            masked_tokens, msa_mask, extra_tokens, extra_mask, gap_agreement_weight, device, dtype
        )
        cluster_profile, cluster_deletion_mean = _summarize_clusters(
            masked_tokens, msa_mask, sel_deletions, extra_tokens, extra_mask, extra_deletions, assignment, dtype
        )
    else:
        cluster_profile = None
        cluster_deletion_mean = None

    extra_tokens, extra_deletions, extra_mask = _crop_extra_msa(
        extra_tokens, extra_deletions, extra_mask, max_extra_msa, generator, device
    )

    if extra_tokens.shape[0] == 0:
        # Keep a zero-masked row when clustering consumes the entire pool. It contributes nothing to the pair
        # track while preserving a nonempty row axis for the extra-MSA Evoformer stack.
        num_res = tokens.shape[1]
        extra_tokens = tokens[:1]
        extra_deletions = torch.zeros((1, num_res), device=device, dtype=dtype)
        extra_mask = torch.zeros((1, num_res), device=device, dtype=dtype)

    bert_mask = bert_mask * msa_mask
    msa_features = _make_msa_features(masked_tokens, sel_deletions, cluster_profile, cluster_deletion_mean, dtype)
    extra_msa_features = _make_extra_msa_features(extra_tokens, extra_deletions, dtype)
    return msa_features, msa_mask, extra_msa_features, extra_mask, bert_mask, true_msa


def _sample_msa(
    tokens: Tensor,
    deletions: Tensor,
    max_seq: int,
    generator: torch.Generator | None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    r"""
    Permute MSA rows (query fixed at row 0), keep ``max_seq``, spill the rest.
    """

    num_seq = tokens.shape[0]
    device = tokens.device
    if num_seq > 1:
        shuffled = torch.randperm(num_seq - 1, generator=generator, device=device) + 1
        index_order = torch.cat([torch.zeros(1, dtype=torch.long, device=device), shuffled])
    else:
        index_order = torch.zeros(1, dtype=torch.long, device=device)
    num_sel = min(max_seq, num_seq)
    selected = index_order[:num_sel]
    remaining = index_order[num_sel:]
    return tokens[selected], deletions[selected], tokens[remaining], deletions[remaining]


def _make_masked_msa(
    tokens: Tensor,
    profile: Tensor,
    *,
    uniform_prob: float,
    profile_prob: float,
    same_prob: float,
    replace_fraction: float,
    generator: torch.Generator | None,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor, Tensor]:
    r"""
    BERT-corrupt the sampled MSA; emit ``(bert_msa, bert_mask, true_msa)``.
    """

    random_aa = torch.zeros(NUM_CLASSES, dtype=dtype, device=device)
    random_aa[AMINO_ACIDS] = 0.05
    one_hot_msa = F.one_hot(tokens, num_classes=NUM_CLASSES).to(dtype)
    categorical = uniform_prob * random_aa + profile_prob * profile.unsqueeze(0) + same_prob * one_hot_msa
    categorical[..., MASK] = 1.0 - profile_prob - same_prob - uniform_prob
    mask_position = torch.rand(tokens.shape, generator=generator, device=device) < replace_fraction
    sampled = _shaped_categorical(categorical, generator)
    bert_msa = torch.where(mask_position, sampled, tokens)
    return bert_msa, mask_position.to(dtype), tokens


def _shaped_categorical(probs: Tensor, generator: torch.Generator | None, epsilon: float = 1e-10) -> Tensor:
    r"""
    Sample one class per cell from ``probs`` using ``generator`` (reproducible).
    """

    shape = probs.shape
    flat = (probs + epsilon).reshape(-1, shape[-1])
    sampled_indices = torch.multinomial(flat, 1, replacement=True, generator=generator).squeeze(-1)
    return sampled_indices.reshape(shape[:-1])


def _nearest_neighbor_clusters(
    msa_tokens: Tensor,
    msa_mask: Tensor,
    extra_tokens: Tensor,
    extra_mask: Tensor,
    gap_agreement_weight: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    r"""
    Assign each extra sequence to its closest cluster by weighted Hamming agreement.
    """

    weights = torch.ones(NUM_CLASSES, device=device, dtype=dtype)
    weights[GAP] = gap_agreement_weight
    weights[MASK] = 0.0
    sample_one_hot = msa_mask.unsqueeze(-1) * F.one_hot(msa_tokens, num_classes=NUM_CLASSES).to(dtype)
    extra_one_hot = extra_mask.unsqueeze(-1) * F.one_hot(extra_tokens, num_classes=NUM_CLASSES).to(dtype)
    num_seq, num_res, _ = sample_one_hot.shape
    extra_num_seq = extra_one_hot.shape[0]
    agreement = torch.matmul(
        extra_one_hot.reshape(extra_num_seq, num_res * NUM_CLASSES),
        (sample_one_hot * weights).reshape(num_seq, num_res * NUM_CLASSES).transpose(0, 1),
    )
    return torch.argmax(agreement, dim=1).to(torch.int64)


def _summarize_clusters(
    msa_tokens: Tensor,
    msa_mask: Tensor,
    msa_deletion_counts: Tensor,
    extra_tokens: Tensor,
    extra_mask: Tensor,
    extra_deletions: Tensor,
    assignment: Tensor,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    r"""
    Average per-cluster profile and deletion mean, including the cluster center.
    """

    num_seq = msa_tokens.shape[0]

    def cluster_sum(values: Tensor) -> Tensor:
        r"""
        Sum extra-MSA rows into their assigned cluster centers (segment sum over ``assignment``).
        """
        return scatter(values, assignment, dim=0, dim_size=num_seq)

    mask_counts = 1e-6 + msa_mask + cluster_sum(extra_mask)
    msa_sum = cluster_sum(extra_mask.unsqueeze(-1) * F.one_hot(extra_tokens, num_classes=NUM_CLASSES).to(dtype))
    msa_sum = msa_sum + F.one_hot(msa_tokens, num_classes=NUM_CLASSES).to(dtype)
    cluster_profile = msa_sum / mask_counts.unsqueeze(-1)
    deletion_sum = cluster_sum(extra_mask * extra_deletions) + msa_deletion_counts
    cluster_deletion_mean = deletion_sum / mask_counts
    return cluster_profile, cluster_deletion_mean


def _crop_extra_msa(
    extra_tokens: Tensor,
    extra_deletions: Tensor,
    extra_mask: Tensor,
    max_extra_msa: int,
    generator: torch.Generator | None,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    r"""
    Randomly subselect at most ``max_extra_msa`` extra rows.
    """

    num_seq = extra_tokens.shape[0]
    if num_seq == 0 or not max_extra_msa:
        return extra_tokens, extra_deletions, extra_mask
    num_sel = min(max_extra_msa, num_seq)
    selected = torch.randperm(num_seq, generator=generator, device=device)[:num_sel]
    return extra_tokens[selected], extra_deletions[selected], extra_mask[selected]


def _make_msa_features(
    msa_tokens: Tensor,
    msa_deletion_counts: Tensor,
    cluster_profile: Tensor | None,
    cluster_deletion_mean: Tensor | None,
    dtype: torch.dtype,
) -> Tensor:
    r"""
    Concatenate the 147-channel clustered MSA feature.
    """

    msa_one_hot = F.one_hot(msa_tokens, num_classes=NUM_CLASSES).to(dtype)
    has_deletion = msa_deletion_counts.clamp(0.0, 1.0).to(dtype).unsqueeze(-1)
    deletion_value = (torch.atan(msa_deletion_counts / 3.0) * (2.0 / math.pi)).to(dtype).unsqueeze(-1)
    parts = [msa_one_hot, has_deletion, deletion_value]
    if cluster_profile is not None and cluster_deletion_mean is not None:
        deletion_mean_value = (torch.atan(cluster_deletion_mean / 3.0) * (2.0 / math.pi)).to(dtype).unsqueeze(-1)
        parts.extend([cluster_profile.to(dtype), deletion_mean_value])
    return torch.cat(parts, dim=-1)


def _make_extra_msa_features(extra_tokens: Tensor, extra_deletions: Tensor, dtype: torch.dtype) -> Tensor:
    r"""
    Concatenate the 74-channel extra-MSA feature.
    """

    extra_one_hot = F.one_hot(extra_tokens, num_classes=NUM_CLASSES).to(dtype)
    has_deletion = extra_deletions.clamp(0.0, 1.0).to(dtype).unsqueeze(-1)
    deletion_value = (torch.atan(extra_deletions / 3.0) * (2.0 / math.pi)).to(dtype).unsqueeze(-1)
    return torch.cat([extra_one_hot, has_deletion, deletion_value], dim=-1)


# ===========================================================================
# AlphaFold-Multimer stochastic MSA realization.
#
# The multimer feature path samples MSA rows by Gumbel argsort while pinning
# the query row, crops the extra axis in the same operation, corrupts tokens over
# the canonical alphabet with a Gumbel-max draw, and uses soft nearest-neighbor
# cluster assignment. Dedicated helpers keep these multimer semantics out of the
# monomer path.
# ===========================================================================


def _gumbel_noise(
    shape: tuple[int, ...], generator: torch.Generator | None, device: torch.device, eps: float = 1e-6
) -> Tensor:
    r"""
    Draw Gumbel(0, 1) noise from ``generator``.
    """

    uniform = torch.rand(shape, dtype=torch.float32, device=device, generator=generator)
    return -torch.log(-torch.log(uniform + eps) + eps)


def _realize_multimer_msa_features(
    msa_tokens: Tensor,
    msa_deletion_counts: Tensor,
    msa_profile: Tensor,
    *,
    residue_mask: Tensor,
    max_msa_clusters: int,
    max_extra_msa: int,
    msa_cluster_features: bool,
    gap_agreement_weight: float,
    masked_msa_enabled: bool,
    uniform_prob: float,
    profile_prob: float,
    same_prob: float,
    replace_fraction: float,
    generator: torch.Generator | None,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    r"""
    Realize one AF-Multimer MSA-feature ensemble slice (unbatched).

    The multimer per-recycle sequence is MSA row sampling, canonical-alphabet Gumbel-max BERT
    masking before clustering, soft nearest-neighbor assignment with cluster
    summarization, and feature assembly. Inputs are ``msa_tokens [M, N]``
    (long canonical MSA indices, row 0 the concatenated query), ``msa_deletion_counts [M, N]``,
    and the full-pool ``msa_profile [N, 72]``. Returns
    ``msa_features [S, N, 147]``, ``msa_mask [S, N]``, ``extra_msa_features [E, N, 74]``,
    ``extra_msa_mask [E, N]``, ``bert_mask [S, N]``, and ``true_msa [S, N]``.
    """

    tokens = msa_tokens.to(device=device)
    deletions = msa_deletion_counts.to(device=device, dtype=torch.float64)
    profile = msa_profile.to(device=device, dtype=dtype)
    num_seq, num_res = tokens.shape
    # The multimer pipeline pads every raw pool to at least 512 rows
    # before sampling. Padded rows carry token/deletion zeroes and a zero mask.
    min_num_seq = 512
    pool_mask = residue_mask.to(device=device, dtype=dtype).expand(num_seq, num_res)
    if num_seq < min_num_seq:
        pad_rows = min_num_seq - num_seq
        tokens = F.pad(tokens, (0, 0, 0, pad_rows))
        deletions = F.pad(deletions, (0, 0, 0, pad_rows))
        pool_mask = F.pad(pool_mask, (0, 0, 0, pad_rows))

    (
        sel_tokens,
        sel_deletions,
        sel_mask,
        extra_tokens,
        extra_deletions,
        extra_mask,
    ) = _multimer_sample_msa(tokens, deletions, pool_mask, max_msa_clusters, max_extra_msa, generator, device)

    if masked_msa_enabled:
        sel_tokens, bert_mask, true_msa = _multimer_make_masked_msa(
            sel_tokens,
            sel_mask,
            profile,
            uniform_prob=uniform_prob,
            profile_prob=profile_prob,
            same_prob=same_prob,
            replace_fraction=replace_fraction,
            generator=generator,
            device=device,
            dtype=dtype,
        )
    else:
        bert_mask = torch.zeros(sel_tokens.shape, device=device, dtype=dtype)
        true_msa = sel_tokens

    if msa_cluster_features:
        cluster_profile, cluster_deletion_mean = _multimer_nearest_neighbor_clusters(
            sel_tokens, sel_mask, sel_deletions, extra_tokens, extra_mask, extra_deletions, gap_agreement_weight, dtype
        )
    else:
        cluster_profile = F.one_hot(sel_tokens, num_classes=NUM_CLASSES).to(dtype)
        cluster_deletion_mean = sel_deletions

    msa_features = _multimer_make_msa_features(sel_tokens, sel_deletions, cluster_profile, cluster_deletion_mean, dtype)
    extra_msa_features = _make_extra_msa_features(extra_tokens, extra_deletions, dtype)

    if extra_tokens.shape[0] == 0:
        # A query-only / small pool leaves the extra axis empty (every row clusters).
        # The predict-mode ``make_fixed_size`` pads it to a nonzero fixed
        # size with a fully zero-masked row; the extra-MSA Evoformer stack cannot
        # reshape a 0-row tensor, so inject one zeroed masked row (token 0, mask 0).
        extra_msa_features = torch.zeros((1, num_res, NUM_CLASSES + 2), device=device, dtype=dtype)
        extra_msa_features[..., 0] = 1.0
        extra_mask = torch.zeros((1, num_res), device=device, dtype=dtype)

    return msa_features, sel_mask, extra_msa_features, extra_mask, bert_mask, true_msa


def _multimer_sample_msa(
    tokens: Tensor,
    deletions: Tensor,
    mask: Tensor,
    max_seq: int,
    max_extra_msa: int,
    generator: torch.Generator | None,
    device: torch.device,
    inf: float = 1e6,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    r"""
    Gumbel-argsort row selection with the query row pinned; crop extra in one step.

    Rows with no valid position are pushed to ``-inf``, the query row (index 0) is
    pinned to ``+inf`` via ``cluster_bias_mask``, and a Gumbel argsort draws the
    remaining order. The first ``max_seq`` become clusters; the next
    ``max_extra_msa`` become the extra pool.
    """

    num_seq = tokens.shape[0]
    logits = (mask.sum(dim=-1).clamp(0.0, 1.0) - 1.0) * inf
    cluster_bias_mask = torch.zeros(num_seq, device=device, dtype=logits.dtype)
    cluster_bias_mask[0] = 1.0
    logits = logits + cluster_bias_mask * inf
    index_order = torch.argsort(logits + _gumbel_noise(logits.shape, generator, device), dim=-1, descending=True)
    selected_indices = index_order[:max_seq]
    extra_indices = index_order[max_seq:][:max_extra_msa]
    return (
        tokens[selected_indices],
        deletions[selected_indices],
        mask[selected_indices],
        tokens[extra_indices],
        deletions[extra_indices],
        mask[extra_indices],
    )


def _multimer_make_masked_msa(
    tokens: Tensor,
    mask: Tensor,
    profile: Tensor,
    *,
    uniform_prob: float,
    profile_prob: float,
    same_prob: float,
    replace_fraction: float,
    generator: torch.Generator | None,
    device: torch.device,
    dtype: torch.dtype,
    eps: float = 1e-6,
) -> tuple[Tensor, Tensor, Tensor]:
    r"""
    Canonical-alphabet Gumbel-max BERT corruption; emit ``(bert_msa, bert_mask, true_msa)``.

    The categorical is a canonical-class mix (uniform / full-pool profile / identity)
    with the remaining mass on the mask class. The row mask gates corruption sites, and replacement uses a
    Gumbel-max draw rather than a plain multinomial. Corrupted cells can take the mask
    class.
    """

    random_aa = torch.zeros(NUM_CLASSES, dtype=dtype, device=device)
    random_aa[AMINO_ACIDS] = 0.05
    one_hot_msa = F.one_hot(tokens, num_classes=NUM_CLASSES).to(dtype)
    categorical = uniform_prob * random_aa + profile_prob * profile.unsqueeze(0) + same_prob * one_hot_msa
    categorical[..., MASK] = 1.0 - profile_prob - same_prob - uniform_prob
    mask_position = torch.rand(tokens.shape, generator=generator, device=device) < replace_fraction
    mask_position = mask_position & (mask > 0)
    logits = torch.log(categorical + eps)
    sampled = torch.argmax(logits + _gumbel_noise(logits.shape, generator, device), dim=-1)
    bert_msa = torch.where(mask_position, sampled, tokens)
    bert_msa = bert_msa * mask.to(bert_msa.dtype)
    return bert_msa.long(), mask_position.to(dtype), tokens


def _multimer_nearest_neighbor_clusters(
    msa_tokens: Tensor,
    msa_mask: Tensor,
    msa_deletion_counts: Tensor,
    extra_tokens: Tensor,
    extra_mask: Tensor,
    extra_deletions: Tensor,
    gap_agreement_weight: float,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    r"""
    Softmax soft-assign each extra row to clusters; average profile + deletion mean.

    Uses ``softmax(1e3 * agreement)`` over canonical one-hots; the corrupted MSA may
    carry the mask token, and each cluster centre contributes to its own ``+1`` count.
    """

    device = msa_mask.device
    weights = torch.ones(NUM_CLASSES, device=device, dtype=dtype)
    weights[GAP] = gap_agreement_weight
    weights[MASK] = 0.0
    msa_one_hot = msa_mask.unsqueeze(-1) * F.one_hot(msa_tokens, num_classes=NUM_CLASSES).to(dtype)
    extra_one_hot = extra_mask.unsqueeze(-1) * F.one_hot(extra_tokens, num_classes=NUM_CLASSES).to(dtype)
    agreement = torch.einsum("mrc,nrc->nm", extra_one_hot, weights * msa_one_hot)
    cluster_assignment = torch.softmax(1e3 * agreement, dim=0)
    cluster_assignment = cluster_assignment * torch.einsum("mr,nr->mn", msa_mask, extra_mask)
    cluster_count = cluster_assignment.sum(dim=-1) + 1.0
    msa_sum = torch.einsum("nm,mrc->nrc", cluster_assignment, extra_one_hot) + msa_one_hot
    cluster_profile = msa_sum / cluster_count[:, None, None]
    del_sum = (
        torch.einsum("nm,mc->nc", cluster_assignment.to(extra_deletions.dtype), extra_mask * extra_deletions)
        + msa_deletion_counts
    )
    cluster_deletion_mean = del_sum / cluster_count[:, None]
    return cluster_profile, cluster_deletion_mean


def _multimer_make_msa_features(
    msa_tokens: Tensor,
    msa_deletion_counts: Tensor,
    cluster_profile: Tensor,
    cluster_deletion_mean: Tensor,
    dtype: torch.dtype,
) -> Tensor:
    r"""
    Concatenate the multimer 147-channel clustered MSA feature.
    """

    msa_one_hot = F.one_hot(msa_tokens, num_classes=NUM_CLASSES).to(dtype)
    has_deletion = msa_deletion_counts.clamp(0.0, 1.0).to(dtype).unsqueeze(-1)
    deletion_value = (torch.atan(msa_deletion_counts / 3.0) * (2.0 / math.pi)).to(dtype).unsqueeze(-1)
    deletion_mean_value = (torch.atan(cluster_deletion_mean / 3.0) * (2.0 / math.pi)).to(dtype).unsqueeze(-1)
    return torch.cat(
        [msa_one_hot, has_deletion, deletion_value, cluster_profile.to(dtype), deletion_mean_value], dim=-1
    )


def alphafold2_predicted_tm_score(
    logits: Tensor,
    residue_mask: Tensor,
    config: AlphaFold2Config | AlphaFold2MultimerConfig,
    *,
    backend: str = "auto",
) -> float:
    r"""
    Compute monomer pTM confidence from predicted-aligned-error logits.

    Args:
        logits: PAE logits for one protein with shape ``(N, N, D)``.
        residue_mask: Valid-residue mask or fractional confidence weights with shape ``(N,)``.
        config: AlphaFold 2 configuration defining the PAE bins.

    Returns:
        The scalar predicted TM score in ``[0, 1]``.
    """

    breaks = torch.linspace(
        0.0,
        config.heads.predicted_aligned_error.max_error_bin,
        logits.shape[-1] - 1,
        device=logits.device,
        dtype=logits.dtype,
    )
    step = breaks[1] - breaks[0]
    centers = torch.cat([breaks + step / 2, breaks[-1:] + 3 * step / 2])
    num_residues = max(int(residue_mask.sum().item()), 19)
    predicted = tm_weight_from_pae(logits, num_residues, bin_centers=centers)
    weights = residue_mask.to(device=predicted.device, dtype=predicted.dtype)
    pair_weights = weights.unsqueeze(-2)
    return float(
        predicted_tm_score(predicted, pair_weights=pair_weights, reference_weights=weights, backend=backend).item()
    )


def _residue_type_atom37_to_atom14_tables() -> tuple[np.ndarray, np.ndarray]:
    atom37_to_atom14 = np.zeros((21, residue_constants.NUM_ATOM_TYPES), dtype=np.int64)
    atom37_exists = np.zeros((21, residue_constants.NUM_ATOM_TYPES), dtype=np.float32)
    for residue_type_index in range(21):
        component_id = "UNK"
        if residue_type_index < residue_constants.NUM_AMINO_ACID_TYPES:
            one_letter_code = residue_constants.AMINO_ACID_ONE_LETTER_CODES[residue_type_index]
            component_id = residue_constants.AMINO_ACID_ONE_TO_THREE[one_letter_code]
        atom14_names = residue_constants.AMINO_ACID_ATOM14_NAMES[component_id]
        for atom37_index, atom_name in enumerate(residue_constants.atom_types):
            if atom_name in atom14_names:
                atom14_index = atom14_names.index(atom_name)
                if atom14_names[atom14_index]:
                    atom37_to_atom14[residue_type_index, atom37_index] = atom14_index
                    atom37_exists[residue_type_index, atom37_index] = 1.0
    return atom37_to_atom14, atom37_exists


_RESIDUE_TYPE_ATOM37_TO_ATOM14, _RESIDUE_TYPE_ATOM37_MASK = _residue_type_atom37_to_atom14_tables()


def _quat_multiply_by_vec(quat: Tensor, vec: Tensor) -> Tensor:
    r"""
    Hamilton-multiply a quaternion by the pure quaternion ``(0, vec)``, returning the 4-vector product.
    """

    r, i, j, k = quat.unbind(dim=-1)
    x, y, z = vec.unbind(dim=-1)
    return torch.stack(
        [
            -i * x - j * y - k * z,
            r * x + j * z - k * y,
            r * y + k * x - i * z,
            r * z + i * y - j * x,
        ],
        dim=-1,
    )


def _atom14_to_atom37(
    atom14_positions: Tensor, residue_type_ids: Tensor, residue_mask: Tensor
) -> tuple[Tensor, Tensor]:
    r"""
    Scatter dense ``atom14`` coordinates into the ``atom37`` layout.

    Returns ``(atom37_positions, atom37_mask)`` where the mask combines the per-residue-type atom37
    existence table with ``residue_mask``.
    """

    atom37_to_atom14 = _gather_residue_type(_RESIDUE_TYPE_ATOM37_TO_ATOM14, residue_type_ids, dtype=torch.long)
    gather_index = atom37_to_atom14.unsqueeze(-1).expand(*atom37_to_atom14.shape, atom14_positions.shape[-1])
    atom37_positions = torch.gather(atom14_positions, dim=-2, index=gather_index)
    atom37_mask = _gather_residue_type(_RESIDUE_TYPE_ATOM37_MASK, residue_type_ids, dtype=atom14_positions.dtype)
    atom37_mask = atom37_mask * residue_mask.unsqueeze(-1).to(dtype=atom37_mask.dtype)
    return atom37_positions * atom37_mask.unsqueeze(-1), atom37_mask


def _residue_type_atom14_mask(residue_type_ids: Tensor, *, dtype: torch.dtype) -> Tensor:
    r"""
    Gather the per-residue ``atom14`` existence mask for each residue type.
    """

    return _gather_residue_type(residue_constants.AMINO_ACID_ATOM14_MASK, residue_type_ids, dtype=dtype)


def _gather_residue_type(table: np.ndarray, residue_type_ids: Tensor, *, dtype: torch.dtype) -> Tensor:
    source = torch.as_tensor(table, device=residue_type_ids.device, dtype=dtype)
    index = residue_type_ids.clamp(min=0, max=20).long()
    gathered = source.index_select(0, index.reshape(-1))
    return gathered.reshape(*index.shape, *source.shape[1:])


def _expand_head_weights(weights: Tensor, target: Tensor) -> Tensor:
    view_shape = [1] * target.ndim
    view_shape[-4] = weights.shape[0]
    return weights.view(*view_shape)


_GLY_INDEX = residue_constants.AMINO_ACID_ONE_LETTER_CODES.index("G")


def _template_batch(input: AlphaFold2Input) -> AlphaFold2TemplateBatch:
    return AlphaFold2TemplateBatch(
        template_residue_type_ids=input.template_residue_type_ids,
        template_mask=input.template_mask,
        template_all_atom_positions=input.template_all_atom_positions,
        template_all_atom_mask=input.template_all_atom_mask,
        template_torsion_angles_sin_cos=input.template_torsion_angles_sin_cos,
        template_alternative_torsion_angles_sin_cos=input.template_alternative_torsion_angles_sin_cos,
        template_torsion_angles_mask=input.template_torsion_angles_mask,
    )


def _clipped_relative_position(residue_index: Tensor, max_relative_feature: int) -> Tensor:
    r"""
    One-hot encode the relative sequence separation of each residue pair, clipped to
    ``+/- max_relative_feature`` (``2 * max_relative_feature + 1`` bins).
    """

    offset = residue_index.unsqueeze(-1) - residue_index.unsqueeze(-2)
    clipped = torch.clamp(offset + max_relative_feature, min=0, max=2 * max_relative_feature)
    return F.one_hot(clipped.long(), num_classes=2 * max_relative_feature + 1)


def _pseudo_beta(residue_type_ids: Tensor, atom37_values: Tensor) -> Tensor:
    r"""
    Select each residue's pseudo-beta row, CA for glycine and CB otherwise, from ``[..., 37, C]`` values.
    """

    atom_index = torch.where(
        residue_type_ids == _GLY_INDEX,
        residue_constants.atom_order["CA"],
        residue_constants.atom_order["CB"],
    )
    return torch.take_along_dim(atom37_values, atom_index[..., None, None], dim=-2).squeeze(-2)


def _backbone_unit_vectors(
    atom37_positions: Tensor, atom37_mask: Tensor, *, backend: str = "auto"
) -> tuple[Tensor, Tensor]:
    r"""
    Return every residue pair's CA direction in the first residue's backbone frame, and the backbone mask.

    The frame of residue ``i`` has its x-axis along ``C - CA`` and ``N - CA`` in its xy-plane, so the
    ``(i, j)`` unit vector is ``normalize(R_i^T (CA_j - CA_i))``. Shapes: unit vectors ``[..., N, N, 3]``,
    backbone mask ``[..., N]``.
    """

    nitrogen, alpha_carbon, carbon = (residue_constants.atom_order[name] for name in ("N", "CA", "C"))
    origin = atom37_positions[..., alpha_carbon, :]
    first = atom37_positions[..., carbon, :] - origin
    second = atom37_positions[..., nitrogen, :] - origin
    rotation = rotation_from_two_vectors(first, second, backend=backend)
    local = torch.einsum("...icm,...ijc->...ijm", rotation, origin.unsqueeze(-3) - origin.unsqueeze(-2))
    backbone_mask = atom37_mask[..., nitrogen] * atom37_mask[..., alpha_carbon] * atom37_mask[..., carbon]
    length_product = torch.linalg.vector_norm(first, dim=-1) * torch.linalg.vector_norm(second, dim=-1)
    nondegenerate = (length_product > 1e-16) & (
        torch.linalg.vector_norm(torch.linalg.cross(first, second), dim=-1) > 1e-8 * length_product
    )
    backbone_mask = backbone_mask * nondegenerate.to(backbone_mask.dtype)
    return F.normalize(local, dim=-1), backbone_mask


def _build_chi_atom37_tables() -> tuple[np.ndarray, np.ndarray]:
    r"""
    Return atom37 indices ``(21, 4, 4)`` and validity mask ``(21, 4)`` for the chi angles.
    """

    indices = np.zeros((21, 4, 4), dtype=np.int64)
    mask = np.zeros((21, 4), dtype=np.float32)
    for residue_type_index in range(residue_constants.NUM_AMINO_ACID_TYPES):
        one_letter_code = residue_constants.AMINO_ACID_ONE_LETTER_CODES[residue_type_index]
        component_id = residue_constants.AMINO_ACID_ONE_TO_THREE[one_letter_code]
        atoms_for_chi = residue_constants.chi_angles_atoms[component_id]
        for chi_index, atom_names in enumerate(atoms_for_chi):
            for position, atom_name in enumerate(atom_names):
                indices[residue_type_index, chi_index, position] = residue_constants.atom_order[atom_name]
            mask[residue_type_index, chi_index] = 1.0
    return indices, mask


_CHI_ATOM37_INDICES, _CHI_ANGLE_MASK = _build_chi_atom37_tables()


def _template_chi_angles(
    positions: Tensor, atom_mask: Tensor, residue_type_ids: Tensor, *, backend: str = "auto"
) -> tuple[Tensor, Tensor]:
    r"""
    Compute the four backbone-independent chi angles per residue from atom37 positions.

    Returns ``(chi, chi_mask)`` with shapes ``[..., N, 4]``. ``chi`` is the signed dihedral of
    the four defining atoms; ``chi_mask`` is the product of the residue-type chi validity and the
    presence of all four atoms.
    """

    indices = torch.as_tensor(_CHI_ATOM37_INDICES, device=positions.device)
    angle_valid = torch.as_tensor(_CHI_ANGLE_MASK, device=positions.device, dtype=positions.dtype)
    # The chi tables cover residue types 0..19 plus a zero-mask UNK row at index 20;
    # clamp so gap / UNK template tokens index the inert UNK row.
    table_index = residue_type_ids.long().clamp(min=0, max=indices.shape[0] - 1)
    per_res_indices = indices[table_index]  # [..., N, 4, 4]  (4 chi angles, 4 atoms each)
    per_res_valid = angle_valid[table_index]  # [..., N, 4]

    # Gather the four defining atom coordinates for every (residue, chi): [..., N, 4, 4, 3].
    gather_index = per_res_indices.unsqueeze(-1).expand(*per_res_indices.shape, 3)
    expanded_positions = positions.unsqueeze(-3).expand(
        *positions.shape[:-2], 4, positions.shape[-2], positions.shape[-1]
    )
    # index the atom (37) axis for each chi group
    atoms = torch.gather(expanded_positions, -2, gather_index)  # [..., N, 4chi, 4atom, 3]

    present = torch.gather(
        atom_mask.unsqueeze(-2).expand(*atom_mask.shape[:-1], 4, atom_mask.shape[-1]),
        -1,
        per_res_indices,
    )  # [..., N, 4chi, 4atom]
    atoms_present = present.prod(dim=-1)  # [..., N, 4]
    chi_mask = per_res_valid * atoms_present
    # Unavailable torsions do not evaluate their coordinate payload. A fixed
    # nondegenerate quadruplet keeps the excluded branch finite in backward.
    observed = chi_mask > 0
    canonical = atoms.new_tensor([[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
    atoms = torch.where(observed[..., None, None], atoms, canonical)
    a0, a1, a2, a3 = atoms.unbind(dim=-2)
    chi = dihedral_angle(a0, a1, a2, a3, backend=backend)  # [..., N, 4]
    chi = torch.where(observed, chi, 0.0)
    return chi, chi_mask


def alphafold2_predicted_interface_tm_score(
    logits: Tensor,
    chain_index: Tensor,
    residue_mask: Tensor,
    config: AlphaFold2Config | AlphaFold2MultimerConfig,
    *,
    backend: str = "auto",
) -> float:
    r"""
    Compute interface pTM (ipTM) from PAE logits over cross-chain pairs.

    Args:
        logits: PAE logits for one assembly with shape ``(N, N, D)``.
        chain_index: Chain index for each residue with shape ``(N,)``.
        residue_mask: Valid-residue mask or fractional confidence weights with shape ``(N,)``.
        config: AlphaFold 2 configuration defining the PAE bins.

    Returns:
        The scalar interface predicted TM score in ``[0, 1]``. Returns ``0`` when
            no valid cross-chain residue pair exists.
    """

    breaks = torch.linspace(
        0.0,
        config.heads.predicted_aligned_error.max_error_bin,
        logits.shape[-1] - 1,
        device=logits.device,
        dtype=logits.dtype,
    )
    step = breaks[1] - breaks[0]
    centers = torch.cat([breaks + step / 2, breaks[-1:] + 3 * step / 2])
    num_residues = max(int(residue_mask.sum().item()), 19)
    predicted = tm_weight_from_pae(logits, num_residues, bin_centers=centers)

    weights = residue_mask.to(device=predicted.device, dtype=predicted.dtype)
    chain_index = chain_index.to(device=predicted.device)
    cross_chain = (chain_index.unsqueeze(-1) != chain_index.unsqueeze(-2)).to(dtype=predicted.dtype)
    pair_weights = weights.unsqueeze(-2) * cross_chain
    # Only residues that participate in at least one cross-chain pair are eligible.
    eligible = (pair_weights.sum(dim=-1) > 0).to(dtype=predicted.dtype) * weights
    return float(
        predicted_tm_score(predicted, pair_weights=pair_weights, reference_weights=eligible, backend=backend).item()
    )


class AlphaFold2StructureModuleOutput(NamedTuple):
    r"""
    Output tensors from the AlphaFold 2 structure module.

    Attributes:
        final_atom_positions: Atom37 coordinates with shape ``(B, N, 37, 3)`` in
            angstroms.
        final_atom_mask: Atom37 validity mask with shape ``(B, N, 37)``.
        structure_module: Final single representation with shape ``(B, N, C)``.
        final_atom14_positions: Atom14 coordinates with shape ``(B, N, 14, 3)`` in
            angstroms.
        final_atom14_mask: Atom14 validity mask with shape ``(B, N, 14)``.
        final_affines: Final backbone frames with shape ``(B, N, 7)`` for
            monomer models or ``(B, N, 12)`` for multimer models.
        traj: Backbone-frame trajectory with shape ``(K, B, N, 7)`` for
            monomer models or ``(K, B, N, 12)`` for multimer models.
        atom14_positions: Atom14 coordinate trajectory with shape
            ``(K, B, N, 14, 3)``.
        sidechain_frames: Rigid-group frame trajectory with shape
            ``(K, B, N, 8, 4, 4)``.
        sidechain_angles: Normalized torsion angles with shape
            ``(K, B, N, 7, 2)``.
        sidechain_unnormalized_angles: Unnormalized torsion vectors with shape
            ``(K, B, N, 7, 2)``.
    """

    final_atom_positions: Tensor
    final_atom_mask: Tensor
    structure_module: Tensor
    final_atom14_positions: Tensor
    final_atom14_mask: Tensor
    final_affines: Tensor
    traj: Tensor
    atom14_positions: Tensor
    sidechain_frames: Tensor
    sidechain_angles: Tensor
    sidechain_unnormalized_angles: Tensor


class AlphaFold2TemplateBatch(NamedTuple):
    r"""
    Template tensors shared by the pair and torsion embedding modules.
    """

    template_residue_type_ids: Tensor
    template_mask: Tensor
    template_all_atom_positions: Tensor
    template_all_atom_mask: Tensor
    template_torsion_angles_sin_cos: Tensor
    template_alternative_torsion_angles_sin_cos: Tensor
    template_torsion_angles_mask: Tensor

    @property
    def dtype(self) -> torch.dtype:
        return self.template_all_atom_positions.dtype


# ===========================================================================
# Tensor and molecular-identity records for monomer and multimer inputs.
# ===========================================================================


@tensorclass
class AlphaFold2Input:
    r"""
    Tensor input for monomer AlphaFold 2 (its residue axis is the prepared slot map's token axis).

    ``B`` is batch size, ``N`` residue count, ``M`` MSA depth, and ``T`` template
    count. Row zero of the MSA tensors is the query sequence.

    Attributes:
        residue_type_ids: Upstream residue types (``ARNDCQEGHILKMFPSTWYV``, 20 for ``X``) with shape ``(B, N)``;
            they index the atom37/atom14 geometry tables.
        token_features: Target features with shape ``(B, N, 73)``: a chain-break column and the canonical class one-hot.
        residue_index: Chain-local residue indices with shape ``(B, N)``.
        residue_mask: Valid-residue mask with shape ``(B, N)``.
        msa_tokens: Unclustered canonical MSA token IDs with shape ``(B, M, N)``.
        msa_deletion_counts: MSA deletion counts with shape ``(B, M, N)``.
        msa_profile: Full-MSA residue frequencies with shape ``(B, N, 72)``.
        template_mask: Template validity mask with shape ``(B, T)``.
        template_residue_type_ids: Template residue IDs with shape ``(B, T, N)``.
        template_all_atom_positions: Template atom37 coordinates with shape
            ``(B, T, N, 37, 3)`` in angstroms.
        template_all_atom_mask: Template atom37 validity mask with shape
            ``(B, T, N, 37)``.
        template_torsion_angles_sin_cos: Template torsion angles with shape
            ``(B, T, N, 7, 2)``.
        template_alternative_torsion_angles_sin_cos: Alternative template torsion
            angles with shape ``(B, T, N, 7, 2)``.
        template_torsion_angles_mask: Template torsion validity mask with shape
            ``(B, T, N, 7)``.
        msa_features: Optional realized clustered-MSA features with shape
            ``(B, M_clustered, N, 147)``.
        msa_mask: Optional clustered-MSA validity mask with shape
            ``(B, M_clustered, N)``.
        extra_msa_features: Optional extra-MSA features with shape
            ``(B, M_extra, N, 74)``.
        extra_msa_mask: Optional extra-MSA validity mask with shape
            ``(B, M_extra, N)``.
        bert_mask: Optional masked-MSA objective mask with shape
            ``(B, M_clustered, N)``.
        true_msa: Optional uncorrupted clustered-MSA token IDs with shape
            ``(B, M_clustered, N)``.
    """

    _: KW_ONLY
    residue_type_ids: Tensor
    token_features: Tensor
    residue_index: Tensor
    residue_mask: Tensor
    msa_tokens: Tensor
    msa_deletion_counts: Tensor
    msa_profile: Tensor
    template_mask: Tensor
    template_residue_type_ids: Tensor
    template_all_atom_positions: Tensor
    template_all_atom_mask: Tensor
    template_torsion_angles_sin_cos: Tensor
    template_alternative_torsion_angles_sin_cos: Tensor
    template_torsion_angles_mask: Tensor
    msa_features: Tensor | None = None
    msa_mask: Tensor | None = None
    extra_msa_features: Tensor | None = None
    extra_msa_mask: Tensor | None = None
    bert_mask: Tensor | None = None
    true_msa: Tensor | None = None


@tensorclass
class AlphaFold2InitialGuessInput(AlphaFold2Input):
    r"""
    Monomer/pTM features with an explicit first recycling coordinate state.

    Attributes:
        initial_recycling_positions: Generated complex atom37 coordinates used
            only as the first recycled ``prev_pos`` state, with shape
            ``(B, N, 37, 3)``.
        chain_index: ColabDesign target/binder asymmetric-unit index with shape
            ``(B, N)``. All target chains are zero and the binder is one.
        binder_mask: Binder-residue mask with shape ``(B, N)``.
    """

    _: KW_ONLY
    initial_recycling_positions: Tensor
    chain_index: Tensor
    binder_mask: Tensor


class AlphaFold2Output(TensorCollection):
    r"""
    Differentiable predictions from one AlphaFold 2 forward pass.

    Attributes:
        single: Per-residue representation with shape ``(B, N, C_single)``.
        pair: Pair representation with shape ``(B, N, N, C_pair)``.
        frames: Backbone-frame trajectory with shape ``(B, K, N, 7)`` for
            monomer models or ``(B, K, N, 12)`` for multimer models.
        atom14_positions: Atom14 coordinate trajectory with shape
            ``(B, K, N, 14, 3)``.
        sidechain_frames: Rigid-group frame trajectory with shape
            ``(B, K, N, 8, 4, 4)``.
        final_atom_positions: Predicted atom37 coordinates with shape
            ``(B, N, 37, 3)`` in angstroms.
        final_atom_mask: Predicted atom37 validity mask with shape ``(B, N, 37)``.
        final_atom14_positions: Final atom14 coordinates with shape
            ``(B, N, 14, 3)`` in angstroms.
        final_atom14_mask: Final atom14 existence mask with shape
            ``(B, N, 14)``.
        sidechain_angles: Normalized torsion trajectory with shape
            ``(B, K, N, 7, 2)``.
        sidechain_unnormalized_angles: Unnormalized torsion vectors with shape
            ``(B, K, N, 7, 2)``.
        true_msa: Realized uncorrupted clustered-MSA tokens with shape
            ``(B, M, N)``, or ``None`` for precomputed MSA features.
        bert_mask: Realized masked-MSA objective mask with shape ``(B, M, N)``,
            or ``None`` for precomputed MSA features.
        distogram_logits: Distance-bin logits with shape ``(B, N, N, D)``.
        masked_msa_logits: Logits in ``ARNDCQEGHILKMFPSTWYVX-`` order, with
            shape ``(B, M, N, 23)`` for monomer (including a final mask class)
            or ``(B, M, N, 22)`` for multimer.
        plddt_logits: Per-residue pLDDT logits with shape ``(B, N, D_plddt)``.
        pae_logits: Optional predicted-aligned-error logits with shape
            ``(B, N, N, D_pae)``.
        resolved_logits: Atom37 resolved-state logits with shape ``(B, N, 37)``.
        num_recycles: Total number of recycling passes executed, including the
            unconditional final pass, with shape ``(B,)``.
    """

    single: Tensor
    pair: Tensor
    frames: Tensor
    atom14_positions: Tensor
    sidechain_frames: Tensor
    final_atom_positions: Tensor
    final_atom_mask: Tensor
    final_atom14_positions: Tensor
    final_atom14_mask: Tensor
    sidechain_angles: Tensor
    sidechain_unnormalized_angles: Tensor
    true_msa: OptionalTensor
    bert_mask: OptionalTensor
    distogram_logits: Tensor
    masked_msa_logits: Tensor
    plddt_logits: Tensor
    pae_logits: OptionalTensor
    resolved_logits: Tensor
    num_recycles: Tensor


@tensorclass
class AlphaFold2MultimerInput:
    r"""
    Tensor input for AlphaFold-Multimer (its residue axis is the prepared slot map's token axis).

    ``B`` is batch size, ``N`` is the concatenated residue count, ``M`` is MSA
    depth, and ``T`` is template count. Residue indices restart at each chain.

    Attributes:
        residue_type_ids: Upstream residue types (``ARNDCQEGHILKMFPSTWYV``, 20 for ``X``) with shape ``(B, N)``;
            they index the atom37/atom14 geometry tables.
        token_features: Target features with shape ``(B, N, 72)``: the canonical class one-hot.
        residue_index: Chain-local residue indices with shape ``(B, N)``.
        residue_mask: Valid-residue mask with shape ``(B, N)``.
        chain_index: Molecular-instance indices with shape ``(B, N)``.
        entity_index: Shared-sequence entity indices with shape ``(B, N)``.
        symmetry_index: Copy indices within each entity with shape ``(B, N)``.
        msa_tokens: Unclustered canonical paired-MSA token IDs with shape ``(B, M, N)``.
        msa_deletion_counts: MSA deletion counts with shape ``(B, M, N)``.
        msa_profile: Full-MSA residue frequencies with shape ``(B, N, 72)``.
        template_mask: Template validity mask with shape ``(B, T)``.
        template_residue_type_ids: Template residue IDs with shape ``(B, T, N)``.
        template_all_atom_positions: Template atom37 coordinates with shape
            ``(B, T, N, 37, 3)`` in angstroms.
        template_all_atom_mask: Template atom37 validity mask with shape
            ``(B, T, N, 37)``.
        msa_features: Optional realized clustered-MSA features with shape
            ``(B, M_clustered, N, 147)``.
        msa_mask: Optional clustered-MSA validity mask with shape
            ``(B, M_clustered, N)``.
        extra_msa_features: Optional extra-MSA features with shape
            ``(B, M_extra, N, 74)``.
        extra_msa_mask: Optional extra-MSA validity mask with shape
            ``(B, M_extra, N)``.
        bert_mask: Optional masked-MSA objective mask with shape
            ``(B, M_clustered, N)``.
        true_msa: Optional uncorrupted clustered-MSA token IDs with shape
            ``(B, M_clustered, N)``.
    """

    residue_type_ids: Tensor
    token_features: Tensor
    residue_index: Tensor
    residue_mask: Tensor
    chain_index: Tensor
    entity_index: Tensor
    symmetry_index: Tensor
    msa_tokens: Tensor
    msa_deletion_counts: Tensor
    msa_profile: Tensor
    template_mask: Tensor
    template_residue_type_ids: Tensor
    template_all_atom_positions: Tensor
    template_all_atom_mask: Tensor
    msa_features: Tensor | None = None
    msa_mask: Tensor | None = None
    extra_msa_features: Tensor | None = None
    extra_msa_mask: Tensor | None = None
    bert_mask: Tensor | None = None
    true_msa: Tensor | None = None


_AlphaFold2FeatureInputT = TypeVar("_AlphaFold2FeatureInputT", bound=AlphaFold2Input | AlphaFold2MultimerInput)


def normalize_alphafold2_input(
    input: _AlphaFold2FeatureInputT, *, dtype: torch.dtype | None = None
) -> _AlphaFold2FeatureInputT:
    """Project available template and continuous features without altering maps."""
    result = cast(_AlphaFold2FeatureInputT, cast(TensorClass, input).clone(recurse=False))
    residue_mask = input.residue_mask.to(torch.bool)
    raw_available = residue_mask.unsqueeze(-2)
    result.msa_tokens = project_model_categories(
        input.msa_tokens, raw_available, num_classes=NUM_CLASSES, feature="AlphaFold2 raw MSA tokens"
    )
    result.msa_deletion_counts = project_model_counts(
        input.msa_deletion_counts, raw_available, feature="AlphaFold2 raw MSA deletion counts"
    )
    result.template_residue_type_ids = project_model_categories(
        input.template_residue_type_ids,
        input.template_mask.bool()[..., None] & residue_mask.unsqueeze(-2),
        num_classes=22,
        feature="AlphaFold2 template classes",
    )
    template_valid = input.template_mask.to(torch.bool)[..., None, None] & residue_mask[..., None, :, None]
    result.template_all_atom_positions = project_model_coordinates(
        input.template_all_atom_positions,
        input.template_all_atom_mask.to(torch.bool) & template_valid,
        dtype=dtype,
        feature="AlphaFold2 template coordinates",
    )
    for name in ("token_features", "msa_profile"):
        setattr(
            result,
            name,
            project_model_features(
                getattr(input, name), residue_mask[..., None], dtype=dtype, feature=f"AlphaFold2 {name}"
            ),
        )
    for name, mask in (("msa_features", input.msa_mask), ("extra_msa_features", input.extra_msa_mask)):
        values = getattr(input, name)
        if values is not None:
            if mask is None:
                mask = residue_mask.unsqueeze(-2)
            setattr(
                result,
                name,
                project_model_features(
                    values, mask.to(torch.bool)[..., None], dtype=dtype, feature=f"AlphaFold2 {name}"
                ),
            )
    if isinstance(input, AlphaFold2Input):
        torsion_valid = input.template_torsion_angles_mask.to(torch.bool) & template_valid
        for name in ("template_torsion_angles_sin_cos", "template_alternative_torsion_angles_sin_cos"):
            setattr(
                result,
                name,
                project_model_features(
                    getattr(input, name), torsion_valid[..., None], dtype=dtype, feature=f"AlphaFold2 {name}"
                ),
            )
    if isinstance(input, AlphaFold2InitialGuessInput):
        model_slots = _gather_residue_type(_RESIDUE_TYPE_ATOM37_MASK, input.residue_type_ids, dtype=torch.bool)
        cast(AlphaFold2InitialGuessInput, result).initial_recycling_positions = project_model_coordinates(
            input.initial_recycling_positions,
            model_slots & residue_mask[..., None],
            dtype=dtype,
            feature="AlphaFold2 initial recycling coordinates",
        )
    return result


def _normalize_alphafold2_template_batch(
    template: AlphaFold2TemplateBatch, *, dtype: torch.dtype | None = None
) -> AlphaFold2TemplateBatch:
    valid = template.template_mask.to(torch.bool)[..., None, None]
    torsion_valid = template.template_torsion_angles_mask.to(torch.bool) & valid
    return template._replace(
        template_residue_type_ids=project_model_categories(
            template.template_residue_type_ids,
            template.template_mask.bool()[..., None],
            num_classes=22,
            feature="AlphaFold2 template classes",
        ),
        template_all_atom_positions=project_model_coordinates(
            template.template_all_atom_positions,
            template.template_all_atom_mask.to(torch.bool) & valid,
            dtype=dtype,
            feature="AlphaFold2 template coordinates",
        ),
        template_torsion_angles_sin_cos=project_model_features(
            template.template_torsion_angles_sin_cos,
            torsion_valid[..., None],
            dtype=dtype,
            feature="AlphaFold2 template torsions",
        ),
        template_alternative_torsion_angles_sin_cos=project_model_features(
            template.template_alternative_torsion_angles_sin_cos,
            torsion_valid[..., None],
            dtype=dtype,
            feature="AlphaFold2 alternative template torsions",
        ),
    )
