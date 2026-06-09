r"""
AlphaFold 2 top-level model.

This source file owns the AF2 public model class. The implementation is runnable
through OFoldX's structure-prediction contract; native AF2 StructureModule
geometry lives in the sibling ``structure.py`` module.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from oops import LayerNorm
from torch import Tensor, nn

from ...data import residue_constants
from ...modules.functional import pair_mask_from_token_mask
from ...modules.head import DistogramHead
from ...modules.head.frames import gather_atoms_by_index
from ...modules.trunk import EvoformerStack
from ..contract import BaseInput, StructurePredictionOutput
from ..mixin import ModelMixin
from ..registry import MODELS, STRUCTURE_PREDICTION_MODELS
from .config import AlphaFold2Config
from .structure import (
    AlphaFold2StructureModule,
    AlphaFold2StructureModuleOutput,
)
from .template import AlphaFold2TemplateEmbedding, AlphaFold2TemplateMsaEmbedding


@MODELS.register("alphafold2")
@STRUCTURE_PREDICTION_MODELS.register("alphafold2")
class AlphaFold2Model(ModelMixin):
    r"""
    AlphaFold 2 structure-prediction model with AF2-shaped module boundaries.

    The trunk is organized as input embedding, recycling embedding, optional
    template embedding, extra-MSA Evoformer stack, main Evoformer stack, and
    native AF2 structure/head modules.
    """

    config_class = AlphaFold2Config

    def __init__(self, config: AlphaFold2Config) -> None:
        super().__init__(config)
        input_config = config.input_embedding
        recycling_config = config.recycling
        evoformer_config = config.evoformer
        template_config = config.template
        iteration_config = evoformer_config.iteration
        heads = config.heads

        self._recycling_steps = int(config.recycling_steps)
        self.input_embedding = AlphaFold2InputEmbedding(
            target_feature_size=input_config.target_feature_size,
            msa_feature_size=input_config.msa_feature_size,
            msa_channel=evoformer_config.msa_size,
            pair_channel=evoformer_config.pair_size,
            max_relative_feature=input_config.max_relative_feature,
        )
        self.recycling_embedding = AlphaFold2RecyclingEmbedding(
            msa_channel=evoformer_config.msa_size,
            pair_channel=evoformer_config.pair_size,
            recycle_pos=recycling_config.recycle_pos,
            recycle_features=recycling_config.recycle_features,
            prev_pos_min_bin=recycling_config.prev_pos.min_bin,
            prev_pos_max_bin=recycling_config.prev_pos.max_bin,
            prev_pos_num_bins=recycling_config.prev_pos.num_bins,
        )
        self.extra_msa_embedding = AlphaFold2ExtraMsaEmbedding(
            extra_msa_feature_size=input_config.extra_msa_feature_size,
            extra_msa_vocab_size=input_config.extra_msa_vocab_size,
            extra_msa_channel=evoformer_config.extra_msa_size,
        )
        self.template_enabled = bool(template_config.enabled)
        self.template_embedder: AlphaFold2TemplateEmbedding | None = None
        self.template_msa_embedding: AlphaFold2TemplateMsaEmbedding | None = None
        if self.template_enabled:
            self.template_embedder = AlphaFold2TemplateEmbedding(
                query_size=evoformer_config.pair_size,
                template_size=template_config.template_pair_stack.num_channels,
                num_blocks=template_config.template_pair_stack.num_block,
                num_distogram_bins=template_config.num_distogram_bins,
                min_bin=template_config.min_bin,
                max_bin=template_config.max_bin,
                vocab_size=template_config.vocab_size,
                use_template_unit_vector=template_config.use_template_unit_vector,
                triangle_multiplication_hidden_size=(
                    template_config.template_pair_stack.triangle_multiplication_outgoing.num_intermediate_channel
                ),
                triangle_attention_hidden_size=(
                    template_config.template_pair_stack.triangle_attention_starting_node.value_dim
                ),
                num_triangle_attention_heads=(
                    template_config.template_pair_stack.triangle_attention_starting_node.num_head
                ),
                transition_expansion=template_config.template_pair_stack.pair_transition.num_intermediate_factor,
                dropout=template_config.template_pair_stack.triangle_attention_starting_node.dropout_rate,
                num_attention_heads=template_config.attention.num_head,
                operator_backend=evoformer_config.operator_backend,
            )
            self.template_msa_embedding = AlphaFold2TemplateMsaEmbedding(
                vocab_size=template_config.vocab_size,
                msa_size=evoformer_config.msa_size,
            )
        self.extra_msa_stack = EvoformerStack(
            num_blocks=evoformer_config.extra_msa_num_blocks,
            msa_size=evoformer_config.extra_msa_size,
            pair_size=evoformer_config.pair_size,
            outer_product_hidden_size=iteration_config.outer_product_mean.num_outer_channel,
            outer_product_first=iteration_config.outer_product_mean.first,
            num_msa_attention_heads=iteration_config.msa_row_attention_with_pair_bias.num_head,
            triangle_multiplication_hidden_size=(
                iteration_config.triangle_multiplication_outgoing.num_intermediate_channel
            ),
            triangle_attention_hidden_size=iteration_config.triangle_attention_starting_node.value_dim,
            num_triangle_attention_heads=iteration_config.triangle_attention_starting_node.num_head,
            msa_transition_expansion=iteration_config.msa_transition.num_intermediate_factor,
            pair_transition_expansion=iteration_config.pair_transition.num_intermediate_factor,
            msa_dropout=iteration_config.msa_row_attention_with_pair_bias.dropout_rate,
            pair_dropout=iteration_config.triangle_attention_starting_node.dropout_rate,
            is_extra_msa=True,
            operator_backend=evoformer_config.operator_backend,
            blocks_per_checkpoint=evoformer_config.blocks_per_checkpoint,
        )
        self.evoformer = EvoformerStack(
            num_blocks=evoformer_config.num_blocks,
            msa_size=evoformer_config.msa_size,
            pair_size=evoformer_config.pair_size,
            outer_product_hidden_size=iteration_config.outer_product_mean.num_outer_channel,
            outer_product_first=iteration_config.outer_product_mean.first,
            num_msa_attention_heads=iteration_config.msa_row_attention_with_pair_bias.num_head,
            triangle_multiplication_hidden_size=(
                iteration_config.triangle_multiplication_outgoing.num_intermediate_channel
            ),
            triangle_attention_hidden_size=iteration_config.triangle_attention_starting_node.value_dim,
            num_triangle_attention_heads=iteration_config.triangle_attention_starting_node.num_head,
            msa_transition_expansion=iteration_config.msa_transition.num_intermediate_factor,
            pair_transition_expansion=iteration_config.pair_transition.num_intermediate_factor,
            msa_dropout=iteration_config.msa_row_attention_with_pair_bias.dropout_rate,
            pair_dropout=iteration_config.triangle_attention_starting_node.dropout_rate,
            is_extra_msa=False,
            operator_backend=evoformer_config.operator_backend,
            blocks_per_checkpoint=evoformer_config.blocks_per_checkpoint,
        )
        self.single_activations = nn.Linear(evoformer_config.msa_size, evoformer_config.single_size)
        self.structure_module = AlphaFold2StructureModule(
            single_size=evoformer_config.single_size,
            pair_size=evoformer_config.pair_size,
            num_layer=heads.structure_module.num_layer,
            num_layer_in_transition=heads.structure_module.num_layer_in_transition,
            num_channel=heads.structure_module.num_channel,
            num_head=heads.structure_module.num_head,
            num_scalar_qk=heads.structure_module.num_scalar_qk,
            num_scalar_v=heads.structure_module.num_scalar_v,
            num_point_qk=heads.structure_module.num_point_qk,
            num_point_v=heads.structure_module.num_point_v,
            dropout=heads.structure_module.dropout,
            position_scale=heads.structure_module.position_scale,
            sidechain_num_channel=heads.structure_module.sidechain.num_channel,
            sidechain_num_residual_block=heads.structure_module.sidechain.num_residual_block,
            zero_init=config.global_config.zero_init,
        )
        self.distogram_head = DistogramHead(
            pair_size=evoformer_config.pair_size,
            num_bins=heads.distogram.num_bins,
        )
        self.predicted_lddt_head = AlphaFold2PredictedLDDTHead(
            input_size=heads.structure_module.num_channel,
            hidden_size=heads.predicted_lddt.num_channels,
            num_bins=heads.predicted_lddt.num_bins,
        )
        self.predicted_aligned_error_head: AlphaFold2PredictedAlignedErrorHead | None = None
        if heads.predicted_aligned_error.enabled:
            self.predicted_aligned_error_head = AlphaFold2PredictedAlignedErrorHead(
                pair_size=evoformer_config.pair_size,
                num_bins=heads.predicted_aligned_error.num_bins,
            )
        self.experimentally_resolved_head = AlphaFold2ExperimentallyResolvedHead(
            input_size=evoformer_config.single_size,
            num_atom_types=heads.experimentally_resolved.num_atom_types,
        )
        self.masked_msa_head = AlphaFold2MaskedMSAHead(
            msa_channel=evoformer_config.msa_size,
            num_output=heads.masked_msa.num_output,
        )
        self.post_init()

    def reset_parameters(self) -> None:
        r"""
        Start coordinate/confidence final projections from zero.
        """

        self.structure_module.reset_parameters()
        nn.init.zeros_(self.distogram_head.linear.weight)
        nn.init.zeros_(self.distogram_head.linear.bias)
        nn.init.zeros_(self.predicted_lddt_head.logits.weight)
        nn.init.zeros_(self.predicted_lddt_head.logits.bias)
        if self.predicted_aligned_error_head is not None:
            nn.init.zeros_(self.predicted_aligned_error_head.logits.weight)
            nn.init.zeros_(self.predicted_aligned_error_head.logits.bias)
        nn.init.zeros_(self.experimentally_resolved_head.logits.weight)
        nn.init.zeros_(self.experimentally_resolved_head.logits.bias)
        nn.init.zeros_(self.masked_msa_head.logits.weight)
        nn.init.zeros_(self.masked_msa_head.logits.bias)

    def forward(
        self,
        batch: AlphaFold2Input,
        *,
        recycling_steps: int | None = None,
        num_samples: int = 1,
        num_sampling_steps: int | None = None,
        generator: torch.Generator | None = None,
    ) -> StructurePredictionOutput:
        r"""
        Run AF2 recycling, structure generation, and output heads.

        ``recycling_steps`` counts extra recycles after the first representation
        pass, so ``3`` runs four representation/structure iterations.
        """

        del num_sampling_steps, generator
        recycling_steps = self._recycling_steps if recycling_steps is None else recycling_steps
        if recycling_steps < 0:
            raise ValueError(f"recycling_steps must be non-negative, got {recycling_steps}")
        if num_samples < 1:
            raise ValueError(f"num_samples must be positive, got {num_samples}")

        representations, structure = self._run_recycling(batch, recycling_steps=recycling_steps)
        atom_positions = _gather_input_atom_positions(structure.final_atom_positions, batch)
        pred_coords = atom_positions.unsqueeze(-3).expand(
            *atom_positions.shape[:-2],
            num_samples,
            *atom_positions.shape[-2:],
        )
        plddt_logits = self._expand_sample_axis(
            self.predicted_lddt_head(structure.structure_module),
            num_samples=num_samples,
            event_ndim=1,
        )
        if self.predicted_aligned_error_head is None:
            pae_logits = None
        else:
            pae_logits = self._expand_sample_axis(
                self.predicted_aligned_error_head(representations.pair),
                num_samples=num_samples,
                event_ndim=2,
            ).contiguous()
        resolved_logits = self._expand_sample_axis(
            self.experimentally_resolved_head(representations.single),
            num_samples=num_samples,
            event_ndim=1,
        )
        return StructurePredictionOutput(
            pred_coords=pred_coords.contiguous(),
            distogram_logits=self.distogram_head(representations.pair),
            plddt_logits=plddt_logits.contiguous(),
            pae_logits=pae_logits,
            resolved_logits=resolved_logits.contiguous(),
        )

    def _run_recycling(
        self,
        batch: AlphaFold2Input,
        *,
        recycling_steps: int,
    ) -> tuple[AlphaFold2Representations, AlphaFold2StructureModuleOutput]:
        recycling = self._initial_recycling(batch)
        final_representations: AlphaFold2Representations | None = None
        final_structure: AlphaFold2StructureModuleOutput | None = None
        for cycle in range(recycling_steps + 1):
            grad_enabled = torch.is_grad_enabled() and (not self.training or cycle == recycling_steps)
            with torch.set_grad_enabled(grad_enabled):
                representations = self._embed_and_refine(batch, recycling=recycling)
                structure = self.structure_module(representations.single, representations.pair, batch)
                representations.structure_module = structure.structure_module
            final_representations = representations
            final_structure = structure
            if cycle != recycling_steps:
                recycling = AlphaFold2RecyclingInput(
                    prev_pos=structure.final_atom_positions.detach(),
                    prev_msa_first_row=representations.msa_first_row.detach(),
                    prev_pair=representations.pair.detach(),
                )
        if final_representations is None or final_structure is None:
            raise RuntimeError("AF2 recycling produced no output")
        return final_representations, final_structure

    def _embed_and_refine(
        self,
        batch: AlphaFold2Input,
        *,
        recycling: AlphaFold2RecyclingInput,
    ) -> AlphaFold2Representations:
        embedded = self.input_embedding(batch)
        msa, pair = self.recycling_embedding(
            embedded.msa,
            embedded.pair,
            batch=batch,
            recycling=recycling,
        )
        pair = pair * embedded.pair_mask.unsqueeze(-1).to(dtype=pair.dtype)
        if self.template_enabled:
            pair = self._add_template_embedding(pair, batch, embedded.pair_mask)
        extra_msa, extra_msa_mask = self.extra_msa_embedding(batch, pair)
        if extra_msa is not None and extra_msa_mask is not None:
            extra_msa, pair = self.extra_msa_stack(
                extra_msa,
                pair,
                msa_mask=extra_msa_mask,
                pair_mask=embedded.pair_mask,
            )
        msa_mask = embedded.msa_mask
        if (
            self.template_enabled
            and self.template_msa_embedding is not None
            and self.config.template.embed_torsion_angles
        ):
            template_msa = self.template_msa_embedding(batch, dtype=msa.dtype, device=msa.device)
            if template_msa is not None:
                template_activations, template_mask = template_msa
                msa = torch.cat([msa, template_activations], dim=-3)
                msa_mask = torch.cat([msa_mask, template_mask], dim=-2)
        msa, pair = self.evoformer(
            msa,
            pair,
            msa_mask=msa_mask,
            pair_mask=embedded.pair_mask,
        )
        msa_first_row = msa.select(dim=-3, index=0)
        single = self.single_activations(msa_first_row)
        return AlphaFold2Representations(msa=msa, single=single, pair=pair, msa_first_row=msa_first_row)

    def _initial_recycling(self, batch: AlphaFold2Input) -> AlphaFold2RecyclingInput:
        token_shape = batch.token_mask.shape
        num_tokens = token_shape[-1]
        dtype = batch.token_features.dtype
        device = batch.token_features.device
        evoformer_config = self.config.evoformer
        return AlphaFold2RecyclingInput(
            prev_pos=torch.zeros_like(batch.reference_positions, dtype=dtype, device=device),
            prev_msa_first_row=torch.zeros(*token_shape, evoformer_config.msa_size, device=device, dtype=dtype),
            prev_pair=torch.zeros(
                *token_shape[:-1],
                num_tokens,
                num_tokens,
                evoformer_config.pair_size,
                device=device,
                dtype=dtype,
            ),
        )

    def _add_template_embedding(self, pair: Tensor, batch: AlphaFold2Input, pair_mask: Tensor) -> Tensor:
        if batch.template_aatype is None or self.template_embedder is None:
            return pair
        return pair + self.template_embedder(pair, batch, pair_mask=pair_mask)

    @staticmethod
    def _expand_sample_axis(logits: Tensor, *, num_samples: int, event_ndim: int) -> Tensor:
        suffix_ndim = event_ndim + 1
        return logits.unsqueeze(-(suffix_ndim + 1)).expand(
            *logits.shape[:-suffix_ndim],
            num_samples,
            *logits.shape[-suffix_ndim:],
        )


class AlphaFold2InputEmbedding(nn.Module):
    r"""
    Build initial target/MSA/pair activations before recycling or Evoformer blocks.
    """

    def __init__(
        self,
        *,
        target_feature_size: int,
        msa_feature_size: int,
        msa_channel: int,
        pair_channel: int,
        max_relative_feature: int,
    ) -> None:
        super().__init__()
        self.max_relative_feature = max_relative_feature
        self.target_feature_size = target_feature_size
        self.preprocess_1d = nn.Linear(target_feature_size, msa_channel)
        self.preprocess_msa = nn.Linear(msa_feature_size, msa_channel)
        self.left_single = nn.Linear(target_feature_size, pair_channel)
        self.right_single = nn.Linear(target_feature_size, pair_channel)
        self.relpos_projection = nn.Linear(2 * max_relative_feature + 1, pair_channel)

    def forward(self, batch: AlphaFold2Input) -> AlphaFold2EmbeddedInputs:
        target_feat = self._target_feat(batch)
        preprocess_1d = self._project_exact_last_dim(
            target_feat,
            projection=self.preprocess_1d,
            feature_name="target_feat",
        )
        msa, msa_mask = self._build_msa_activations(batch, preprocess_1d)
        pair = self._build_pair_activations(batch, target_feat)
        pair_mask = pair_mask_from_token_mask(batch.token_mask.to(torch.bool))
        return AlphaFold2EmbeddedInputs(msa=msa, pair=pair, msa_mask=msa_mask, pair_mask=pair_mask)

    def _build_msa_activations(self, batch: AlphaFold2Input, preprocess_1d: Tensor) -> tuple[Tensor, Tensor]:
        if batch.msa_feat is not None:
            msa_projected = self._project_exact_last_dim(
                batch.msa_feat,
                projection=self.preprocess_msa,
                feature_name="msa_feat",
            )
            msa_mask = self._default_msa_mask(batch, batch.msa_feat)
        else:
            msa_projected = preprocess_1d.new_zeros(*preprocess_1d.shape[:-2], 1, *preprocess_1d.shape[-2:])
            msa_mask = batch.token_mask.unsqueeze(-2).to(dtype=preprocess_1d.dtype)
        msa = preprocess_1d.unsqueeze(-3) + msa_projected
        return msa, msa_mask.to(device=msa.device, dtype=msa.dtype)

    def _build_pair_activations(self, batch: AlphaFold2Input, target_feat: Tensor) -> Tensor:
        left_single = self._project_exact_last_dim(target_feat, projection=self.left_single, feature_name="target_feat")
        right_single = self._project_exact_last_dim(
            target_feat,
            projection=self.right_single,
            feature_name="target_feat",
        )
        pair = left_single.unsqueeze(-2) + right_single.unsqueeze(-3)
        relpos = clipped_relative_position(batch.residue_index, self.max_relative_feature).to(
            device=pair.device,
            dtype=pair.dtype,
        )
        return pair + self.relpos_projection(relpos)

    @staticmethod
    def _default_msa_mask(batch: AlphaFold2Input, msa_features: Tensor) -> Tensor:
        if batch.msa_mask is not None:
            return batch.msa_mask
        return batch.token_mask.unsqueeze(-2).expand(*msa_features.shape[:-1]).to(dtype=msa_features.dtype)

    def _target_feat(self, batch: AlphaFold2Input) -> Tensor:
        if batch.target_feat is not None:
            return batch.target_feat
        if batch.aatype is None:
            raise ValueError("AlphaFold2Input requires target_feat or aatype for official AF2 input embedding")
        aatype = batch.aatype.to(device=batch.token_features.device, dtype=torch.long)
        aatype_1hot = F.one_hot(
            aatype.clamp(min=0, max=self.target_feature_size - 2),
            num_classes=self.target_feature_size - 1,
        )
        has_break = torch.zeros(*aatype.shape, 1, dtype=aatype_1hot.dtype, device=aatype.device)
        target_feat = torch.cat([has_break, aatype_1hot], dim=-1)
        return target_feat.to(dtype=batch.token_features.dtype)

    @staticmethod
    def _project_exact_last_dim(features: Tensor, *, projection: nn.Linear, feature_name: str) -> Tensor:
        if features.shape[-1] != projection.in_features:
            raise ValueError(
                f"{feature_name} has last dimension {features.shape[-1]}, expected {projection.in_features}"
            )
        return projection(features)


class AlphaFold2RecyclingEmbedding(nn.Module):
    r"""
    Add previous-cycle features to MSA and pair activations.
    """

    def __init__(
        self,
        *,
        msa_channel: int,
        pair_channel: int,
        recycle_pos: bool,
        recycle_features: bool,
        prev_pos_min_bin: float,
        prev_pos_max_bin: float,
        prev_pos_num_bins: int,
    ) -> None:
        super().__init__()
        self.recycle_pos = recycle_pos
        self.recycle_features = recycle_features
        self.prev_pos_min_bin = prev_pos_min_bin
        self.prev_pos_max_bin = prev_pos_max_bin
        self.prev_pos_num_bins = prev_pos_num_bins
        self.prev_pos_linear = nn.Linear(prev_pos_num_bins, pair_channel)
        self.prev_msa_first_row_norm = LayerNorm(msa_channel)
        self.prev_pair_norm = LayerNorm(pair_channel)

    def forward(
        self,
        msa: Tensor,
        pair: Tensor,
        *,
        batch: AlphaFold2Input,
        recycling: AlphaFold2RecyclingInput,
    ) -> tuple[Tensor, Tensor]:
        if self.recycle_features:
            msa = self._add_previous_msa_first_row(msa, recycling.prev_msa_first_row)
            pair = pair + self.prev_pair_norm(recycling.prev_pair.to(device=pair.device, dtype=pair.dtype))
        if self.recycle_pos:
            pair = pair + self.prev_pos_linear(self._previous_position_dgram(batch, recycling.prev_pos, pair.dtype))
        return msa, pair

    def _add_previous_msa_first_row(self, msa: Tensor, prev_msa_first_row: Tensor) -> Tensor:
        previous = prev_msa_first_row.to(device=msa.device, dtype=msa.dtype)
        first_row = msa.select(dim=-3, index=0) + self.prev_msa_first_row_norm(previous)
        if msa.shape[-3] == 1:
            return first_row.unsqueeze(-3)
        return torch.cat([first_row.unsqueeze(-3), msa.narrow(-3, 1, msa.shape[-3] - 1)], dim=-3)

    def _previous_position_dgram(
        self,
        batch: AlphaFold2Input,
        prev_pos: Tensor,
        dtype: torch.dtype,
    ) -> Tensor:
        prev_pos = prev_pos.to(device=batch.token_to_rep_atom.device, dtype=dtype)
        if prev_pos.dim() == batch.token_mask.dim() + 2 and prev_pos.shape[-2] == residue_constants.atom_type_num:
            rep_pos = _pseudo_beta_from_atom37(prev_pos, batch)
        else:
            rep_pos = gather_atoms_by_index(prev_pos, batch.token_to_rep_atom.to(dtype=torch.long))
        squared_distances = (
            (rep_pos.unsqueeze(-2).float() - rep_pos.unsqueeze(-3).float()).square().sum(dim=-1).to(dtype=dtype)
        )
        lower_breaks = torch.linspace(
            self.prev_pos_min_bin,
            self.prev_pos_max_bin,
            steps=self.prev_pos_num_bins,
            device=squared_distances.device,
            dtype=squared_distances.dtype,
        )
        lower_breaks = lower_breaks.square()
        upper_breaks = torch.cat([lower_breaks[1:], lower_breaks.new_tensor([1e8])])
        return ((squared_distances.unsqueeze(-1) > lower_breaks) & (squared_distances.unsqueeze(-1) < upper_breaks)).to(
            dtype=dtype
        )


class AlphaFold2ExtraMsaEmbedding(nn.Module):
    r"""
    Embed extra-MSA features before the extra-MSA Evoformer stack.
    """

    def __init__(
        self,
        *,
        extra_msa_feature_size: int,
        extra_msa_vocab_size: int,
        extra_msa_channel: int,
    ) -> None:
        super().__init__()
        self.extra_msa_vocab_size = extra_msa_vocab_size
        self.extra_msa_projection = nn.Linear(extra_msa_feature_size, extra_msa_channel)

    def forward(self, batch: AlphaFold2Input, pair: Tensor) -> tuple[Tensor | None, Tensor | None]:
        if batch.extra_msa_feat is not None:
            extra_features = batch.extra_msa_feat.to(device=pair.device, dtype=pair.dtype)
        elif batch.extra_msa is not None:
            extra_features = self._create_extra_msa_feature(batch, dtype=pair.dtype)
        else:
            return None, None
        extra_msa = self._project_exact_last_dim(
            extra_features,
            projection=self.extra_msa_projection,
            feature_name="extra_msa_feat",
        )
        if batch.extra_msa_mask is None:
            extra_msa_mask = extra_msa.new_ones(extra_msa.shape[:-1])
        else:
            extra_msa_mask = batch.extra_msa_mask.to(device=pair.device, dtype=pair.dtype)
        return extra_msa, extra_msa_mask

    def _create_extra_msa_feature(self, batch: AlphaFold2Input, *, dtype: torch.dtype) -> Tensor:
        if batch.extra_msa is None:
            raise ValueError("extra_msa is required to build extra MSA features")
        extra_msa = batch.extra_msa.long()
        one_hot = F.one_hot(extra_msa, num_classes=self.extra_msa_vocab_size).to(dtype=dtype)
        zeros = torch.zeros_like(extra_msa, dtype=dtype)
        has_deletion = zeros if batch.extra_has_deletion is None else batch.extra_has_deletion.to(dtype=dtype)
        deletion_value = zeros if batch.extra_deletion_value is None else batch.extra_deletion_value.to(dtype=dtype)
        return torch.cat([one_hot, has_deletion.unsqueeze(-1), deletion_value.unsqueeze(-1)], dim=-1)

    @staticmethod
    def _project_exact_last_dim(features: Tensor, *, projection: nn.Linear, feature_name: str) -> Tensor:
        if features.shape[-1] != projection.in_features:
            raise ValueError(
                f"{feature_name} has last dimension {features.shape[-1]}, expected {projection.in_features}"
            )
        return projection(features)


class AlphaFold2PredictedLDDTHead(nn.Module):
    def __init__(self, *, input_size: int, hidden_size: int, num_bins: int) -> None:
        super().__init__()
        self.input_layer_norm = LayerNorm(input_size)
        self.act_0 = nn.Linear(input_size, hidden_size)
        self.act_1 = nn.Linear(hidden_size, hidden_size)
        self.logits = nn.Linear(hidden_size, num_bins)

    def forward(self, structure_module: Tensor) -> Tensor:
        act = self.input_layer_norm(structure_module)
        act = F.relu(self.act_0(act))
        act = F.relu(self.act_1(act))
        return self.logits(act)


class AlphaFold2PredictedAlignedErrorHead(nn.Module):
    def __init__(self, *, pair_size: int, num_bins: int) -> None:
        super().__init__()
        self.logits = nn.Linear(pair_size, num_bins)

    def forward(self, pair: Tensor) -> Tensor:
        return self.logits(pair)


class AlphaFold2ExperimentallyResolvedHead(nn.Module):
    def __init__(self, *, input_size: int, num_atom_types: int) -> None:
        super().__init__()
        self.logits = nn.Linear(input_size, num_atom_types)

    def forward(self, single: Tensor) -> Tensor:
        return self.logits(single)


class AlphaFold2MaskedMSAHead(nn.Module):
    def __init__(self, *, msa_channel: int, num_output: int) -> None:
        super().__init__()
        self.logits = nn.Linear(msa_channel, num_output)

    def forward(self, msa: Tensor) -> Tensor:
        return self.logits(msa)


def clipped_relative_position(residue_index: Tensor, max_relative_feature: int) -> Tensor:
    offset = residue_index.unsqueeze(-1) - residue_index.unsqueeze(-2)
    clipped = torch.clamp(offset + max_relative_feature, min=0, max=2 * max_relative_feature)
    return F.one_hot(clipped.long(), num_classes=2 * max_relative_feature + 1)


def _gather_input_atom_positions(atom37_positions: Tensor, batch: AlphaFold2Input) -> Tensor:
    if batch.atom37_index is None:
        return atom37_positions
    token_index = batch.atom_to_token.to(device=atom37_positions.device, dtype=torch.long)
    residue_atom37 = atom37_positions.gather(
        -3,
        token_index.unsqueeze(-1).unsqueeze(-1).expand(*token_index.shape, atom37_positions.shape[-2], 3),
    )
    atom37_index = batch.atom37_index.to(device=atom37_positions.device, dtype=torch.long)
    atom_index = atom37_index.unsqueeze(-1).unsqueeze(-1).expand(*atom37_index.shape, 1, 3)
    return residue_atom37.gather(-2, atom_index).squeeze(-2)


def _pseudo_beta_from_atom37(atom37_positions: Tensor, batch: AlphaFold2Input) -> Tensor:
    ca = residue_constants.atom_order["CA"]
    cb = residue_constants.atom_order["CB"]
    aatype = batch.aatype
    if aatype is None:
        atom_index = torch.full(atom37_positions.shape[:-2], ca, device=atom37_positions.device, dtype=torch.long)
    else:
        gly = residue_constants.restypes.index("G")
        aatype = aatype.to(device=atom37_positions.device, dtype=torch.long)
        atom_index = torch.where(aatype == gly, torch.full_like(aatype, ca), torch.full_like(aatype, cb))
    gather_index = atom_index.unsqueeze(-1).unsqueeze(-1).expand(*atom_index.shape, 1, 3)
    return atom37_positions.gather(-2, gather_index).squeeze(-2)


@dataclass(slots=True, kw_only=True)
class AlphaFold2Input(BaseInput):
    r"""
    Input tensors consumed by [`AlphaFold2Model`][ofoldx.models.alphafold2.AlphaFold2Model].

    Required fields are the OFoldX structure-token contract produced by the
    current processor. Optional fields use official AlphaFold 2 names so the
    future sequence/MSA/template processor can populate them directly.
    """

    token_mask: Tensor
    residue_index: Tensor
    chain_index: Tensor
    entity_index: Tensor
    symmetry_index: Tensor
    token_index: Tensor
    atom_features: Tensor
    atom_to_token: Tensor
    atom_mask: Tensor
    reference_positions: Tensor
    reference_space_id: Tensor
    representative_atom_mask: Tensor
    atom_token_index: Tensor
    token_to_rep_atom: Tensor
    token_features: Tensor
    reference_mask: Tensor
    reference_charges: Tensor
    target_positions: Tensor | None = None
    target_mask: Tensor | None = None
    molecule_type: Tensor | None = None
    frame_atom_index: Tensor | None = None
    frame_mask: Tensor | None = None
    token_bonds: Tensor | None = None

    # Official AF2-style optional features.
    aatype: Tensor | None = None
    target_feat: Tensor | None = None
    msa_feat: Tensor | None = None
    msa_features: Tensor | None = None
    msa_mask: Tensor | None = None
    extra_msa: Tensor | None = None
    extra_has_deletion: Tensor | None = None
    extra_deletion_value: Tensor | None = None
    extra_msa_feat: Tensor | None = None
    extra_msa_mask: Tensor | None = None
    atom37_index: Tensor | None = None
    template_mask: Tensor | None = None
    template_aatype: Tensor | None = None
    template_all_atom_positions: Tensor | None = None
    template_all_atom_masks: Tensor | None = None
    template_pseudo_beta: Tensor | None = None
    template_pseudo_beta_mask: Tensor | None = None
    template_torsion_angles_sin_cos: Tensor | None = None
    template_alt_torsion_angles_sin_cos: Tensor | None = None
    template_torsion_angles_mask: Tensor | None = None


@dataclass(slots=True)
class AlphaFold2RecyclingInput:
    r"""
    Previous-cycle tensors injected by AF2 recycling.
    """

    prev_pos: Tensor
    prev_msa_first_row: Tensor
    prev_pair: Tensor


@dataclass(slots=True)
class AlphaFold2EmbeddedInputs:
    r"""
    Embedded inputs passed from input embedding into recycling and Evoformer stacks.
    """

    msa: Tensor
    pair: Tensor
    msa_mask: Tensor
    pair_mask: Tensor


@dataclass(slots=True)
class AlphaFold2Representations:
    r"""
    Representation bundle emitted by the Evoformer stack.
    """

    msa: Tensor
    single: Tensor
    pair: Tensor
    msa_first_row: Tensor
    structure_module: Tensor | None = None
