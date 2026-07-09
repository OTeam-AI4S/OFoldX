r"""
AlphaFold 2 top-level model.

This source file owns the AF2 public model class, template stack, native
StructureModule geometry, and structure-prediction contract.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ...data import residue_constants
from ...modules.attention import GatedMultiheadAttention
from ...modules.backend import LayerNorm
from ...modules.functional import pair_mask_from_token_mask
from ...modules.head import DistogramHead
from ...modules.head.frames import gather_atoms_by_index
from ...modules.representation import Pair, Single
from ...modules.transition import RowwiseDropout
from ...modules.trunk import EvoformerStack
from ...modules.trunk.evoformer import (
    EvoformerTransition,
    TriangleAttention,
    TriangleMultiplication,
)
from ..contract import BaseInput, StructurePredictionOutput, register_pytree_dataclass
from ..mixin import ModelMixin
from ..registry import MODELS, STRUCTURE_PREDICTION_MODELS
from .config import AlphaFold2Config


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
    #: Typed input built by the keyword-tensor call path; assigned to
    #: [`AlphaFold2Input`][ofoldx.models.alphafold2.AlphaFold2Input] below its definition.
    input_class: ClassVar[type[BaseInput]]

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
    ) -> AlphaFold2Output | StructurePredictionOutput:
        r"""
        Training path: run AF2 recycling, structure generation, and every output head.

        ``forward`` is the *training* path. It returns an
        [`AlphaFold2Output`][ofoldx.models.alphafold2.AlphaFold2Output] carrying the raw
        per-token logits and structure trajectory a loss consumes (distogram, masked-MSA,
        pLDDT, PAE, experimentally-resolved, plus the FAPE frame trajectory and side-chain
        angles). Gradients flow through the final recycle only, mirroring the AlphaFold 2
        recycling schedule. Inference (confidence packaging and the sample axis) lives in
        [`fold`][ofoldx.models.alphafold2.AlphaFold2Model.fold]; for backward compatibility
        an eval-mode call delegates to ``fold`` so ``model(batch)`` keeps predicting a
        structure when not training.

        ``recycling_steps`` counts extra recycles after the first representation
        pass, so ``3`` runs four representation/structure iterations.

        Args:
            batch: Typed AlphaFold 2 input.
            recycling_steps: Extra recycles after the first pass; ``None`` uses the config value.
            num_samples: Sample-axis size when delegating to ``fold`` in eval mode.
            num_sampling_steps: Ignored; accepted for the eval-mode ``fold`` delegation signature.
            generator: Ignored; accepted for the eval-mode ``fold`` delegation signature.

        Returns:
            An [`AlphaFold2Output`][ofoldx.models.alphafold2.AlphaFold2Output] in training mode,
            or a [`StructurePredictionOutput`][ofoldx.models.contract.StructurePredictionOutput]
            from the ``fold`` delegation in eval mode.
        """

        if not self.training:
            return self.fold(
                batch,
                recycling_steps=recycling_steps,
                num_samples=num_samples,
                num_sampling_steps=num_sampling_steps,
                generator=generator,
            )
        del num_samples, num_sampling_steps, generator
        recycling_steps = self._recycling_steps if recycling_steps is None else recycling_steps
        if recycling_steps < 0:
            raise ValueError(f"recycling_steps must be non-negative, got {recycling_steps}")

        representations, structure = self._run_recycling(batch, recycling_steps=recycling_steps)
        single = representations.single.tensor
        pair = representations.pair.tensor
        pae_logits = None if self.predicted_aligned_error_head is None else self.predicted_aligned_error_head(pair)
        return AlphaFold2Output(
            single=representations.single,
            pair=representations.pair,
            frames=structure.traj,
            final_atom_positions=structure.final_atom_positions,
            final_atom_mask=structure.final_atom_mask,
            sidechain_angles=structure.sidechain_angles,
            sidechain_unnormalized_angles=structure.sidechain_unnormalized_angles,
            distogram_logits=self.distogram_head(pair),
            masked_msa_logits=self.masked_msa_head(representations.msa),
            plddt_logits=self.predicted_lddt_head(structure.structure_module),
            pae_logits=pae_logits,
            resolved_logits=self.experimentally_resolved_head(single),
        )

    @torch.no_grad()
    def fold(
        self,
        batch: AlphaFold2Input,
        *,
        recycling_steps: int | None = None,
        num_samples: int = 1,
        num_sampling_steps: int | None = None,
        generator: torch.Generator | None = None,
    ) -> StructurePredictionOutput:
        r"""
        Inference path: predict a structure and confidence for ``batch``.

        Runs the same recycling trunk as `forward` under ``no_grad``, maps the
        atom37 output onto the input atom layout, and packages the result into a
        [`StructurePredictionOutput`][ofoldx.models.contract.StructurePredictionOutput]
        carrying an explicit ``num_samples`` axis. AlphaFold 2 is deterministic, so
        ``num_sampling_steps`` and ``generator`` are accepted for a uniform
        structure-prediction verb signature but do not affect the output.

        Args:
            batch: Typed AlphaFold 2 input.
            recycling_steps: Extra recycles after the first pass; ``None`` uses the config value.
            num_samples: Size of the broadcast sample axis on every returned tensor.
            num_sampling_steps: Ignored (AF2 is single-pass); accepted for signature parity.
            generator: Ignored (AF2 is deterministic); accepted for signature parity.

        Returns:
            Structure-prediction output with a ``num_samples`` sample axis.

        Raises:
            ValueError: If ``recycling_steps`` is negative or ``num_samples`` is not positive.

        Examples:
            >>> from ofoldx import AutoModel, AutoProcessor  # doctest: +SKIP
            >>> processor = AutoProcessor.from_pretrained("ofoldx/alphafold2")  # doctest: +SKIP
            >>> model = AutoModel.from_pretrained("ofoldx/alphafold2")  # doctest: +SKIP
            >>> output = model.fold(processor("complex.cif"))  # doctest: +SKIP
        """

        del num_sampling_steps, generator
        recycling_steps = self._recycling_steps if recycling_steps is None else recycling_steps
        if recycling_steps < 0:
            raise ValueError(f"recycling_steps must be non-negative, got {recycling_steps}")
        if num_samples < 1:
            raise ValueError(f"num_samples must be positive, got {num_samples}")

        representations, structure = self._run_recycling(batch, recycling_steps=recycling_steps)
        single = representations.single.tensor
        pair = representations.pair.tensor
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
                self.predicted_aligned_error_head(pair),
                num_samples=num_samples,
                event_ndim=2,
            ).contiguous()
        resolved_logits = self._expand_sample_axis(
            self.experimentally_resolved_head(single),
            num_samples=num_samples,
            event_ndim=1,
        )
        return StructurePredictionOutput(
            pred_coords=pred_coords.contiguous(),
            distogram_logits=self.distogram_head(pair),
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
                structure = self.structure_module(representations.single.tensor, representations.pair.tensor, batch)
                representations.structure_module = structure.structure_module
            final_representations = representations
            final_structure = structure
            if cycle != recycling_steps:
                recycling = AlphaFold2RecyclingInput(
                    prev_pos=structure.final_atom_positions.detach(),
                    prev_msa_first_row=representations.msa_first_row.detach(),
                    prev_pair=representations.pair.tensor.detach(),
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
        return AlphaFold2Representations(
            msa=msa,
            single=Single(single),
            pair=Pair(pair),
            msa_first_row=msa_first_row,
        )

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


def _restype_atom37_to_atom14_tables() -> tuple[np.ndarray, np.ndarray]:
    atom37_to_atom14 = np.zeros((21, residue_constants.atom_type_num), dtype=np.int64)
    atom37_exists = np.zeros((21, residue_constants.atom_type_num), dtype=np.float32)
    for restype_index in range(21):
        resname = "UNK"
        if restype_index < residue_constants.restype_num:
            resname = residue_constants.restype_1to3[residue_constants.restypes[restype_index]]
        atom14_names = residue_constants.restype_name_to_atom14_names[resname]
        for atom37_index, atom_name in enumerate(residue_constants.atom_types):
            if atom_name in atom14_names:
                atom14_index = atom14_names.index(atom_name)
                if atom14_names[atom14_index]:
                    atom37_to_atom14[restype_index, atom37_index] = atom14_index
                    atom37_exists[restype_index, atom37_index] = 1.0
    return atom37_to_atom14, atom37_exists


_RESTYPE_ATOM37_TO_ATOM14, _RESTYPE_ATOM37_MASK = _restype_atom37_to_atom14_tables()


@dataclass(slots=True)
class AlphaFold2StructureModuleOutput:
    r"""
    Output tensors from the native AF2 structure module.
    """

    final_atom_positions: Tensor
    final_atom_mask: Tensor
    structure_module: Tensor
    final_atom14_positions: Tensor
    final_atom14_mask: Tensor
    final_affines: Tensor
    traj: Tensor
    sidechain_angles: Tensor
    sidechain_unnormalized_angles: Tensor


@dataclass(slots=True)
class Rigid:
    r"""
    Batched rigid transform represented by rotation matrices and translations.
    """

    rotation: Tensor
    translation: Tensor

    @classmethod
    def from_tensor4x4(cls, tensor: Tensor) -> Rigid:
        return cls(rotation=tensor[..., :3, :3], translation=tensor[..., :3, 3])

    def apply(self, points: Tensor) -> Tensor:
        return torch.matmul(self.rotation, points.unsqueeze(-1)).squeeze(-1) + self.translation

    def compose(self, other: Rigid) -> Rigid:
        rotation = self.rotation @ other.rotation
        translation = self.apply(other.translation)
        return Rigid(rotation=rotation, translation=translation)

    def compose_rotation(self, rotation: Tensor) -> Rigid:
        return Rigid(rotation=self.rotation @ rotation, translation=self.translation)

    def with_group_axis(self) -> Rigid:
        return Rigid(rotation=self.rotation.unsqueeze(-3), translation=self.translation.unsqueeze(-2))


@dataclass(slots=True)
class QuatAffine:
    r"""
    AF2 backbone affine represented by a quaternion, rotation, and translation.
    """

    quaternion: Tensor
    rotation: Tensor
    translation: Tensor

    @classmethod
    def identity(cls, shape: torch.Size | tuple[int, ...], *, device: torch.device, dtype: torch.dtype) -> QuatAffine:
        quaternion = torch.zeros(*shape, 4, device=device, dtype=dtype)
        quaternion[..., 0] = 1.0
        rotation = torch.eye(3, device=device, dtype=dtype).expand(*shape, 3, 3).clone()
        translation = torch.zeros(*shape, 3, device=device, dtype=dtype)
        return cls(quaternion=quaternion, rotation=rotation, translation=translation)

    @classmethod
    def from_tensor(cls, tensor: Tensor, *, normalize: bool = False) -> QuatAffine:
        quaternion = tensor[..., :4]
        if normalize:
            quaternion = F.normalize(quaternion, dim=-1)
        return cls(quaternion=quaternion, rotation=quat_to_rot(quaternion), translation=tensor[..., 4:7])

    def to_tensor(self) -> Tensor:
        return torch.cat([self.quaternion, self.translation], dim=-1)

    def to_rigid(self) -> Rigid:
        return Rigid(rotation=self.rotation, translation=self.translation)

    def scale_translation(self, position_scale: float) -> QuatAffine:
        return QuatAffine(
            quaternion=self.quaternion,
            rotation=self.rotation,
            translation=self.translation * position_scale,
        )

    def detach_rotation(self) -> QuatAffine:
        return QuatAffine(
            quaternion=self.quaternion.detach(),
            rotation=self.rotation.detach(),
            translation=self.translation,
        )

    def pre_compose(self, update: Tensor) -> QuatAffine:
        vector_quaternion_update = update[..., :3]
        translation_update = update[..., 3:6]
        quaternion = self.quaternion + quat_multiply_by_vec(self.quaternion, vector_quaternion_update)
        translation = self.translation + torch.matmul(self.rotation, translation_update.unsqueeze(-1)).squeeze(-1)
        quaternion = F.normalize(quaternion, dim=-1)
        return QuatAffine(quaternion=quaternion, rotation=quat_to_rot(quaternion), translation=translation)

    def apply_to_point(self, point: Tensor, *, extra_dims: int = 0) -> Tensor:
        rotation = self.rotation
        translation = self.translation
        for _ in range(extra_dims):
            rotation = rotation.unsqueeze(-3)
            translation = translation.unsqueeze(-2)
        return torch.matmul(rotation, point.unsqueeze(-1)).squeeze(-1) + translation

    def invert_point(self, transformed_point: Tensor, *, extra_dims: int = 0) -> Tensor:
        rotation = self.rotation
        translation = self.translation
        for _ in range(extra_dims):
            rotation = rotation.unsqueeze(-3)
            translation = translation.unsqueeze(-2)
        point = transformed_point - translation
        return torch.matmul(rotation.transpose(-1, -2), point.unsqueeze(-1)).squeeze(-1)


class AlphaFold2StructureModule(nn.Module):
    r"""
    Native AlphaFold 2 StructureModule inference path.
    """

    def __init__(
        self,
        *,
        single_size: int,
        pair_size: int,
        num_layer: int,
        num_layer_in_transition: int,
        num_channel: int,
        num_head: int,
        num_scalar_qk: int,
        num_scalar_v: int,
        num_point_qk: int,
        num_point_v: int,
        sidechain_num_channel: int,
        sidechain_num_residual_block: int,
        position_scale: float,
        dropout: float,
        zero_init: bool,
    ) -> None:
        super().__init__()
        self.num_layer = num_layer
        self.position_scale = position_scale
        self.single_layer_norm = LayerNorm(single_size)
        self.initial_projection = nn.Linear(single_size, num_channel)
        self.pair_layer_norm = LayerNorm(pair_size)
        self.fold_iteration = FoldIteration(
            num_channel=num_channel,
            pair_size=pair_size,
            num_layer_in_transition=num_layer_in_transition,
            num_head=num_head,
            num_scalar_qk=num_scalar_qk,
            num_scalar_v=num_scalar_v,
            num_point_qk=num_point_qk,
            num_point_v=num_point_v,
            sidechain_num_channel=sidechain_num_channel,
            sidechain_num_residual_block=sidechain_num_residual_block,
            position_scale=position_scale,
            dropout=dropout,
            zero_init=zero_init,
        )

    def reset_parameters(self) -> None:
        self.fold_iteration.reset_parameters()

    def forward(self, single: Tensor, pair: Tensor, batch: Any) -> AlphaFold2StructureModuleOutput:
        token_mask = batch.token_mask.to(device=single.device, dtype=single.dtype)
        aatype = _aatype_from_batch(batch, token_mask)
        act = self.single_layer_norm(single)
        initial_act = act
        act = self.initial_projection(act)
        pair = self.pair_layer_norm(pair)
        affine = QuatAffine.identity(token_mask.shape, device=single.device, dtype=single.dtype)

        affines = []
        final_sidechain: SidechainOutput | None = None
        for _ in range(self.num_layer):
            act, affine, sidechain, raw_affine = self.fold_iteration(
                act=act,
                pair=pair,
                mask=token_mask,
                affine=affine,
                initial_act=initial_act,
                aatype=aatype,
            )
            affines.append(raw_affine.to_tensor())
            final_sidechain = sidechain

        if final_sidechain is None:
            raise RuntimeError("AF2 structure module produced no fold iterations")

        traj = torch.stack(affines, dim=0)
        traj_scale = traj.new_tensor(
            [1.0, 1.0, 1.0, 1.0, self.position_scale, self.position_scale, self.position_scale],
        )
        traj = traj * traj_scale
        atom14_mask = restype_atom14_mask(aatype, dtype=single.dtype) * token_mask.unsqueeze(-1)
        atom37_positions, atom37_mask = atom14_to_atom37(final_sidechain.atom_pos, aatype, token_mask)
        return AlphaFold2StructureModuleOutput(
            final_atom_positions=atom37_positions,
            final_atom_mask=atom37_mask,
            structure_module=act,
            final_atom14_positions=final_sidechain.atom_pos,
            final_atom14_mask=atom14_mask,
            final_affines=traj[-1],
            traj=traj,
            sidechain_angles=final_sidechain.angles_sin_cos,
            sidechain_unnormalized_angles=final_sidechain.unnormalized_angles_sin_cos,
        )


class FoldIteration(nn.Module):
    r"""
    One shared-weight AF2 structure-module iteration.
    """

    def __init__(
        self,
        *,
        num_channel: int,
        pair_size: int,
        num_layer_in_transition: int,
        num_head: int,
        num_scalar_qk: int,
        num_scalar_v: int,
        num_point_qk: int,
        num_point_v: int,
        sidechain_num_channel: int,
        sidechain_num_residual_block: int,
        position_scale: float,
        dropout: float,
        zero_init: bool,
    ) -> None:
        super().__init__()
        self.position_scale = position_scale
        self.invariant_point_attention = InvariantPointAttention(
            num_channel=num_channel,
            pair_size=pair_size,
            num_head=num_head,
            num_scalar_qk=num_scalar_qk,
            num_scalar_v=num_scalar_v,
            num_point_qk=num_point_qk,
            num_point_v=num_point_v,
            zero_init=zero_init,
        )
        self.dropout = nn.Dropout(dropout)
        self.attention_layer_norm = LayerNorm(num_channel)
        self.transition = StructureTransition(
            num_channel=num_channel,
            num_layer_in_transition=num_layer_in_transition,
            zero_init=zero_init,
        )
        self.transition_layer_norm = LayerNorm(num_channel)
        self.affine_update = nn.Linear(num_channel, 6)
        self.sidechain = MultiRigidSidechain(
            num_channel=num_channel,
            sidechain_num_channel=sidechain_num_channel,
            num_residual_block=sidechain_num_residual_block,
            zero_init=zero_init,
        )

    def reset_parameters(self) -> None:
        self.invariant_point_attention.reset_parameters()
        self.transition.reset_parameters()
        nn.init.zeros_(self.affine_update.weight)
        nn.init.zeros_(self.affine_update.bias)
        self.sidechain.reset_parameters()

    def forward(
        self,
        *,
        act: Tensor,
        pair: Tensor,
        mask: Tensor,
        affine: QuatAffine,
        initial_act: Tensor,
        aatype: Tensor,
    ) -> tuple[Tensor, QuatAffine, SidechainOutput, QuatAffine]:
        attn = self.invariant_point_attention(inputs_1d=act, inputs_2d=pair, mask=mask, affine=affine)
        act = self.attention_layer_norm(act + self.dropout(attn))
        act = self.transition_layer_norm(act + self.dropout(self.transition(act)))
        affine = affine.pre_compose(self.affine_update(act))
        sidechain = self.sidechain(affine.scale_translation(self.position_scale), (act, initial_act), aatype)
        return act, affine.detach_rotation(), sidechain, affine


class InvariantPointAttention(nn.Module):
    r"""
    AF2 invariant point attention.
    """

    def __init__(
        self,
        *,
        num_channel: int,
        pair_size: int,
        num_head: int,
        num_scalar_qk: int,
        num_scalar_v: int,
        num_point_qk: int,
        num_point_v: int,
        zero_init: bool,
        dist_epsilon: float = 1e-8,
    ) -> None:
        super().__init__()
        self.num_head = num_head
        self.num_scalar_qk = num_scalar_qk
        self.num_scalar_v = num_scalar_v
        self.num_point_qk = num_point_qk
        self.num_point_v = num_point_v
        self.dist_epsilon = dist_epsilon
        self.zero_init = zero_init
        self.q_scalar = nn.Linear(num_channel, num_head * num_scalar_qk)
        self.kv_scalar = nn.Linear(num_channel, num_head * (num_scalar_v + num_scalar_qk))
        self.q_point_local = nn.Linear(num_channel, num_head * 3 * num_point_qk)
        self.kv_point_local = nn.Linear(num_channel, num_head * 3 * (num_point_qk + num_point_v))
        self.attention_2d = nn.Linear(pair_size, num_head)
        output_size = num_head * num_scalar_v + 4 * num_head * num_point_v + num_head * pair_size
        self.output_projection = nn.Linear(output_size, num_channel)
        self.trainable_point_weights = nn.Parameter(torch.empty(num_head))
        self._reset_point_weights()

    def reset_parameters(self) -> None:
        self._reset_point_weights()
        if self.zero_init:
            nn.init.zeros_(self.output_projection.weight)
            nn.init.zeros_(self.output_projection.bias)

    def forward(self, *, inputs_1d: Tensor, inputs_2d: Tensor, mask: Tensor, affine: QuatAffine) -> Tensor:
        num_head = self.num_head
        q_scalar = self.q_scalar(inputs_1d).unflatten(-1, (num_head, self.num_scalar_qk))
        kv_scalar = self.kv_scalar(inputs_1d).unflatten(-1, (num_head, self.num_scalar_v + self.num_scalar_qk))
        k_scalar, v_scalar = kv_scalar.split((self.num_scalar_qk, self.num_scalar_v), dim=-1)

        q_point_local = self.q_point_local(inputs_1d).unflatten(-1, (3, num_head, self.num_point_qk)).movedim(-3, -1)
        q_point = affine.apply_to_point(q_point_local, extra_dims=2)
        kv_point_local = (
            self.kv_point_local(inputs_1d)
            .unflatten(-1, (3, num_head, self.num_point_qk + self.num_point_v))
            .movedim(-3, -1)
        )
        kv_point = affine.apply_to_point(kv_point_local, extra_dims=2)
        k_point, v_point = kv_point.split((self.num_point_qk, self.num_point_v), dim=-2)

        scalar_weights = math.sqrt(1.0 / (3.0 * max(self.num_scalar_qk, 1)))
        q_scalar = scalar_weights * q_scalar
        attn_qk_scalar = torch.einsum("...qhd,...khd->...hqk", q_scalar, k_scalar)

        point_weights = math.sqrt(1.0 / (3.0 * max(self.num_point_qk, 1) * 9.0 / 2.0))
        point_weights = point_weights * F.softplus(self.trainable_point_weights)
        q_point = q_point.transpose(-4, -3)
        k_point = k_point.transpose(-4, -3)
        dist2 = (q_point.unsqueeze(-3) - k_point.unsqueeze(-4)).square().sum(dim=-1)
        attn_qk_point = -0.5 * (dist2 * _expand_head_weights(point_weights, dist2)).sum(dim=-1)

        attention_2d_weights = math.sqrt(1.0 / 3.0)
        attention_2d = attention_2d_weights * self.attention_2d(inputs_2d).movedim(-1, -3)
        attn_logits = attn_qk_scalar + attn_qk_point + attention_2d
        pair_mask = mask.to(dtype=torch.bool).unsqueeze(-1) & mask.to(dtype=torch.bool).unsqueeze(-2)
        attn_logits = attn_logits.masked_fill(~pair_mask.unsqueeze(-3), -1e5)
        attn = torch.softmax(attn_logits, dim=-1)

        v = v_scalar.transpose(-3, -2)
        result_scalar = torch.einsum("...hqk,...hkd->...hqd", attn, v).transpose(-3, -2).flatten(-2)
        v_point = v_point.transpose(-4, -3)
        result_point = torch.einsum("...hqk,...hkpc->...hqpc", attn, v_point).transpose(-4, -3)
        result_point = affine.invert_point(result_point, extra_dims=2)
        result_point_norm = torch.sqrt(self.dist_epsilon + result_point.square().sum(dim=-1))
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

    def _reset_point_weights(self) -> None:
        nn.init.constant_(self.trainable_point_weights, math.log(math.exp(1.0) - 1.0))


class StructureTransition(nn.Module):
    def __init__(self, *, num_channel: int, num_layer_in_transition: int, zero_init: bool) -> None:
        super().__init__()
        self.zero_init = zero_init
        self.layers = nn.ModuleList(nn.Linear(num_channel, num_channel) for _ in range(num_layer_in_transition))

    def reset_parameters(self) -> None:
        if self.zero_init and self.layers:
            nn.init.zeros_(self.layers[-1].weight)
            nn.init.zeros_(self.layers[-1].bias)

    def forward(self, act: Tensor) -> Tensor:
        for index, layer in enumerate(self.layers):
            act = layer(act)
            if index != len(self.layers) - 1:
                act = F.relu(act)
        return act


@dataclass(slots=True)
class SidechainOutput:
    angles_sin_cos: Tensor
    unnormalized_angles_sin_cos: Tensor
    atom_pos: Tensor
    frames: Rigid


class MultiRigidSidechain(nn.Module):
    r"""
    AF2 side-chain torsion and atom14 coordinate generator.
    """

    def __init__(
        self,
        *,
        num_channel: int,
        sidechain_num_channel: int,
        num_residual_block: int,
        zero_init: bool,
    ) -> None:
        super().__init__()
        self.zero_init = zero_init
        self.input_projection = nn.ModuleList(nn.Linear(num_channel, sidechain_num_channel) for _ in range(2))
        self.resblocks = nn.ModuleList(
            SidechainResidualBlock(sidechain_num_channel, zero_init=zero_init) for _ in range(num_residual_block)
        )
        self.unnormalized_angles = nn.Linear(sidechain_num_channel, 14)

    def reset_parameters(self) -> None:
        for block in self.resblocks:
            block.reset_parameters()
        if self.zero_init:
            nn.init.zeros_(self.unnormalized_angles.weight)
            nn.init.zeros_(self.unnormalized_angles.bias)

    def forward(self, affine: QuatAffine, representations: tuple[Tensor, Tensor], aatype: Tensor) -> SidechainOutput:
        act = sum(proj(F.relu(rep)) for proj, rep in zip(self.input_projection, representations, strict=True))
        for block in self.resblocks:
            act = block(act)
        unnormalized_angles = self.unnormalized_angles(F.relu(act)).unflatten(-1, (7, 2))
        angles = F.normalize(unnormalized_angles, dim=-1)
        frames = torsion_angles_to_frames(aatype, affine.to_rigid(), angles)
        atom_pos = frames_and_literature_positions_to_atom14_pos(aatype, frames)
        return SidechainOutput(
            angles_sin_cos=angles,
            unnormalized_angles_sin_cos=unnormalized_angles,
            atom_pos=atom_pos,
            frames=frames,
        )


class SidechainResidualBlock(nn.Module):
    def __init__(self, num_channel: int, *, zero_init: bool) -> None:
        super().__init__()
        self.zero_init = zero_init
        self.resblock1 = nn.Linear(num_channel, num_channel)
        self.resblock2 = nn.Linear(num_channel, num_channel)

    def reset_parameters(self) -> None:
        if self.zero_init:
            nn.init.zeros_(self.resblock2.weight)
            nn.init.zeros_(self.resblock2.bias)

    def forward(self, act: Tensor) -> Tensor:
        old_act = act
        act = self.resblock1(F.relu(act))
        act = self.resblock2(F.relu(act))
        return act + old_act


def quat_to_rot(normalized_quat: Tensor) -> Tensor:
    w, x, y, z = normalized_quat.unbind(dim=-1)
    return torch.stack(
        [
            torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], dim=-1),
            torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], dim=-1),
            torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], dim=-1),
        ],
        dim=-2,
    )


def quat_multiply_by_vec(quat: Tensor, vec: Tensor) -> Tensor:
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


def torsion_angles_to_frames(aatype: Tensor, backb_to_global: Rigid, torsion_angles_sin_cos: Tensor) -> Rigid:
    default_frames = Rigid.from_tensor4x4(
        _gather_restype(residue_constants.restype_rigid_group_default_frame, aatype, dtype=torsion_angles_sin_cos.dtype)
    )
    sin_angles = torsion_angles_sin_cos[..., 0]
    cos_angles = torsion_angles_sin_cos[..., 1]
    sin_angles = torch.cat([torch.zeros_like(sin_angles[..., :1]), sin_angles], dim=-1)
    cos_angles = torch.cat([torch.ones_like(cos_angles[..., :1]), cos_angles], dim=-1)
    zeros = torch.zeros_like(sin_angles)
    ones = torch.ones_like(sin_angles)
    all_rots = torch.stack(
        [
            torch.stack([ones, zeros, zeros], dim=-1),
            torch.stack([zeros, cos_angles, -sin_angles], dim=-1),
            torch.stack([zeros, sin_angles, cos_angles], dim=-1),
        ],
        dim=-2,
    )
    all_frames = default_frames.compose_rotation(all_rots)
    chi1 = _frame_index(all_frames, 4)
    chi2 = chi1.compose(_frame_index(all_frames, 5))
    chi3 = chi2.compose(_frame_index(all_frames, 6))
    chi4 = chi3.compose(_frame_index(all_frames, 7))
    frames_to_backbone = Rigid(
        rotation=torch.cat(
            [
                all_frames.rotation[..., :5, :, :],
                chi2.rotation.unsqueeze(-3),
                chi3.rotation.unsqueeze(-3),
                chi4.rotation.unsqueeze(-3),
            ],
            dim=-3,
        ),
        translation=torch.cat(
            [
                all_frames.translation[..., :5, :],
                chi2.translation.unsqueeze(-2),
                chi3.translation.unsqueeze(-2),
                chi4.translation.unsqueeze(-2),
            ],
            dim=-2,
        ),
    )
    return backb_to_global.with_group_axis().compose(frames_to_backbone)


def frames_and_literature_positions_to_atom14_pos(aatype: Tensor, all_frames_to_global: Rigid) -> Tensor:
    group_index = _gather_restype(residue_constants.restype_atom14_to_rigid_group, aatype, dtype=torch.long)
    group_mask = F.one_hot(group_index, num_classes=8).to(dtype=all_frames_to_global.rotation.dtype)
    atom_frames = Rigid(
        rotation=torch.einsum("...nag,...ngij->...naij", group_mask, all_frames_to_global.rotation),
        translation=torch.einsum("...nag,...ngc->...nac", group_mask, all_frames_to_global.translation),
    )
    lit_positions = _gather_restype(
        residue_constants.restype_atom14_rigid_group_positions,
        aatype,
        dtype=all_frames_to_global.rotation.dtype,
    )
    pred_positions = atom_frames.apply(lit_positions)
    return pred_positions * restype_atom14_mask(aatype, dtype=pred_positions.dtype).unsqueeze(-1)


def atom14_to_atom37(atom14_positions: Tensor, aatype: Tensor, token_mask: Tensor) -> tuple[Tensor, Tensor]:
    atom37_to_atom14 = _gather_restype(_RESTYPE_ATOM37_TO_ATOM14, aatype, dtype=torch.long)
    gather_index = atom37_to_atom14.unsqueeze(-1).expand(*atom37_to_atom14.shape, atom14_positions.shape[-1])
    atom37_positions = torch.gather(atom14_positions, dim=-2, index=gather_index)
    atom37_mask = _gather_restype(_RESTYPE_ATOM37_MASK, aatype, dtype=atom14_positions.dtype)
    atom37_mask = atom37_mask * token_mask.unsqueeze(-1).to(dtype=atom37_mask.dtype)
    return atom37_positions * atom37_mask.unsqueeze(-1), atom37_mask


def restype_atom14_mask(aatype: Tensor, *, dtype: torch.dtype) -> Tensor:
    return _gather_restype(residue_constants.restype_atom14_mask, aatype, dtype=dtype)


def _frame_index(rigid: Rigid, index: int) -> Rigid:
    return Rigid(rotation=rigid.rotation[..., index, :, :], translation=rigid.translation[..., index, :])


def _gather_restype(table: np.ndarray, aatype: Tensor, *, dtype: torch.dtype) -> Tensor:
    source = torch.as_tensor(table, device=aatype.device, dtype=dtype)
    index = aatype.clamp(min=0, max=20).long()
    gathered = source.index_select(0, index.reshape(-1))
    return gathered.reshape(*index.shape, *source.shape[1:])


def _aatype_from_batch(batch: Any, token_mask: Tensor) -> Tensor:
    if getattr(batch, "aatype", None) is not None:
        return batch.aatype.to(device=token_mask.device, dtype=torch.long).clamp(min=0, max=20)
    raise ValueError("AlphaFold2StructureModule requires batch.aatype for official AF2 residue geometry")


def _expand_head_weights(weights: Tensor, target: Tensor) -> Tensor:
    view_shape = [1] * target.ndim
    view_shape[-4] = weights.shape[0]
    return weights.view(*view_shape)


_GLY_INDEX = residue_constants.restypes.index("G")


class AlphaFold2TemplateEmbedding(nn.Module):
    r"""
    Official-style AF2 template pair embedding.
    """

    def __init__(
        self,
        *,
        query_size: int,
        template_size: int,
        num_blocks: int,
        num_distogram_bins: int,
        min_bin: float,
        max_bin: float,
        vocab_size: int,
        use_template_unit_vector: bool,
        triangle_multiplication_hidden_size: int,
        triangle_attention_hidden_size: int,
        num_triangle_attention_heads: int,
        transition_expansion: int,
        dropout: float,
        num_attention_heads: int,
        operator_backend: str,
    ) -> None:
        super().__init__()
        self.single_template_embedding = AlphaFold2SingleTemplateEmbedding(
            template_size=template_size,
            num_blocks=num_blocks,
            num_distogram_bins=num_distogram_bins,
            min_bin=min_bin,
            max_bin=max_bin,
            vocab_size=vocab_size,
            use_template_unit_vector=use_template_unit_vector,
            triangle_multiplication_hidden_size=triangle_multiplication_hidden_size,
            triangle_attention_hidden_size=triangle_attention_hidden_size,
            num_triangle_attention_heads=num_triangle_attention_heads,
            transition_expansion=transition_expansion,
            dropout=dropout,
            operator_backend=operator_backend,
        )
        self.attention = GatedMultiheadAttention(
            hidden_size=query_size,
            query_size=query_size,
            key_size=template_size,
            value_size=template_size,
            num_attention_heads=num_attention_heads,
            head_dim=template_size // num_attention_heads,
            query_bias=False,
            output_bias=True,
            gating=False,
            operator_backend=operator_backend,
        )

    def forward(self, query_embedding: Tensor, batch: Any, *, pair_mask: Tensor) -> Tensor:
        template_batch = _template_batch(batch, reference=query_embedding)
        if template_batch is None:
            return query_embedding.new_zeros(query_embedding.shape)

        template_pair = self.single_template_embedding(template_batch, pair_mask=pair_mask)
        batch_size, num_templates, num_tokens = template_pair.shape[:3]
        template_mask = template_batch.template_mask.to(device=query_embedding.device, dtype=query_embedding.dtype)
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


class AlphaFold2SingleTemplateEmbedding(nn.Module):
    r"""
    Embed and refine each template with the shared template pair stack.
    """

    def __init__(
        self,
        *,
        template_size: int,
        num_blocks: int,
        num_distogram_bins: int,
        min_bin: float,
        max_bin: float,
        vocab_size: int,
        use_template_unit_vector: bool,
        triangle_multiplication_hidden_size: int,
        triangle_attention_hidden_size: int,
        num_triangle_attention_heads: int,
        transition_expansion: int,
        dropout: float,
        operator_backend: str,
    ) -> None:
        super().__init__()
        self.num_distogram_bins = num_distogram_bins
        self.min_bin = min_bin
        self.max_bin = max_bin
        self.vocab_size = vocab_size
        self.use_template_unit_vector = use_template_unit_vector
        template_feature_size = num_distogram_bins + 1 + 2 * vocab_size + 3 + 1
        self.embedding2d = nn.Linear(template_feature_size, template_size)
        self.template_pair_stack = AlphaFold2TemplatePairStack(
            num_blocks=num_blocks,
            pair_size=template_size,
            triangle_multiplication_hidden_size=triangle_multiplication_hidden_size,
            triangle_attention_hidden_size=triangle_attention_hidden_size,
            num_triangle_attention_heads=num_triangle_attention_heads,
            transition_expansion=transition_expansion,
            dropout=dropout,
            operator_backend=operator_backend,
        )
        self.output_layer_norm = LayerNorm(template_size)

    def forward(self, template_batch: AlphaFold2TemplateBatch, *, pair_mask: Tensor) -> Tensor:
        features, template_pair_mask = self._features(template_batch)
        batch_size, num_templates, num_tokens = features.shape[:3]
        pair_mask = pair_mask[:, None].expand(batch_size, num_templates, num_tokens, num_tokens)
        pair_mask = pair_mask * template_pair_mask

        act = self.embedding2d(features).reshape(batch_size * num_templates, num_tokens, num_tokens, -1)
        flat_pair_mask = pair_mask.reshape(batch_size * num_templates, num_tokens, num_tokens)
        act = self.template_pair_stack(act, pair_mask=flat_pair_mask)
        act = self.output_layer_norm(act)
        return act.reshape(batch_size, num_templates, num_tokens, num_tokens, -1)

    def _features(self, template_batch: AlphaFold2TemplateBatch) -> tuple[Tensor, Tensor]:
        aatype = template_batch.template_aatype.long().clamp(min=0, max=self.vocab_size - 1)
        aatype_one_hot = F.one_hot(aatype, num_classes=self.vocab_size).to(dtype=template_batch.dtype)
        pseudo_beta, pseudo_beta_mask = template_batch.pseudo_beta()
        template_pair_mask = pseudo_beta_mask.unsqueeze(-1) * pseudo_beta_mask.unsqueeze(-2)
        distogram = _distogram(
            pseudo_beta,
            min_bin=self.min_bin,
            max_bin=self.max_bin,
            num_bins=self.num_distogram_bins,
        ).to(dtype=template_batch.dtype)

        residue_i = aatype_one_hot.unsqueeze(-2).expand(*template_pair_mask.shape, self.vocab_size)
        residue_j = aatype_one_hot.unsqueeze(-3).expand(*template_pair_mask.shape, self.vocab_size)
        unit_vector, backbone_pair_mask = template_batch.backbone_unit_vector()
        if not self.use_template_unit_vector:
            unit_vector = torch.zeros_like(unit_vector)
        features = torch.cat(
            [
                distogram,
                template_pair_mask.unsqueeze(-1),
                residue_i,
                residue_j,
                unit_vector,
                backbone_pair_mask.unsqueeze(-1),
            ],
            dim=-1,
        )
        features = features * backbone_pair_mask.unsqueeze(-1)
        return features, backbone_pair_mask


class AlphaFold2TemplatePairStack(nn.Module):
    r"""
    TemplatePairStack with official template block order.
    """

    def __init__(
        self,
        *,
        num_blocks: int,
        pair_size: int,
        triangle_multiplication_hidden_size: int,
        triangle_attention_hidden_size: int,
        num_triangle_attention_heads: int,
        transition_expansion: int,
        dropout: float,
        operator_backend: str,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                AlphaFold2TemplatePairStackBlock(
                    pair_size=pair_size,
                    triangle_multiplication_hidden_size=triangle_multiplication_hidden_size,
                    triangle_attention_hidden_size=triangle_attention_hidden_size,
                    num_triangle_attention_heads=num_triangle_attention_heads,
                    transition_expansion=transition_expansion,
                    dropout=dropout,
                    operator_backend=operator_backend,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, pair: Tensor, *, pair_mask: Tensor) -> Tensor:
        for block in self.blocks:
            pair = block(pair, pair_mask=pair_mask)
        return pair


class AlphaFold2TemplatePairStackBlock(nn.Module):
    def __init__(
        self,
        *,
        pair_size: int,
        triangle_multiplication_hidden_size: int,
        triangle_attention_hidden_size: int,
        num_triangle_attention_heads: int,
        transition_expansion: int,
        dropout: float,
        operator_backend: str,
    ) -> None:
        super().__init__()
        self.pair_stack = AlphaFold2TemplatePairStackLayer(
            pair_size=pair_size,
            triangle_multiplication_hidden_size=triangle_multiplication_hidden_size,
            triangle_attention_hidden_size=triangle_attention_hidden_size,
            num_triangle_attention_heads=num_triangle_attention_heads,
            transition_expansion=transition_expansion,
            dropout=dropout,
            operator_backend=operator_backend,
        )

    def forward(self, pair: Tensor, *, pair_mask: Tensor) -> Tensor:
        return self.pair_stack(pair, pair_mask=pair_mask)


class AlphaFold2TemplatePairStackLayer(nn.Module):
    def __init__(
        self,
        *,
        pair_size: int,
        triangle_multiplication_hidden_size: int,
        triangle_attention_hidden_size: int,
        num_triangle_attention_heads: int,
        transition_expansion: int,
        dropout: float,
        operator_backend: str,
    ) -> None:
        super().__init__()
        self.triangle_attention_starting_node = TriangleAttention(
            pair_size=pair_size,
            head_dim=triangle_attention_hidden_size,
            num_attention_heads=num_triangle_attention_heads,
            orientation="per_row",
            operator_backend=operator_backend,
        )
        self.triangle_attention_ending_node = TriangleAttention(
            pair_size=pair_size,
            head_dim=triangle_attention_hidden_size,
            num_attention_heads=num_triangle_attention_heads,
            orientation="per_column",
            operator_backend=operator_backend,
        )
        self.triangle_multiplication_outgoing = TriangleMultiplication(
            pair_size=pair_size,
            hidden_size=triangle_multiplication_hidden_size,
            equation="outgoing",
        )
        self.triangle_multiplication_incoming = TriangleMultiplication(
            pair_size=pair_size,
            hidden_size=triangle_multiplication_hidden_size,
            equation="incoming",
        )
        self.feed_forward = EvoformerTransition(pair_size, expansion=transition_expansion)
        self.rowwise_dropout = RowwiseDropout(dropout)

    def forward(self, pair: Tensor, *, pair_mask: Tensor) -> Tensor:
        pair = pair + self.rowwise_dropout(self.triangle_attention_starting_node(pair, pair_mask))
        pair = pair + self.rowwise_dropout(self.triangle_attention_ending_node(pair, pair_mask))
        pair = pair + self.rowwise_dropout(self.triangle_multiplication_outgoing(pair, pair_mask))
        pair = pair + self.rowwise_dropout(self.triangle_multiplication_incoming(pair, pair_mask))
        return pair + self.feed_forward(pair, pair_mask)


class AlphaFold2TemplateMsaEmbedding(nn.Module):
    r"""
    Embed template torsion features as extra MSA rows.
    """

    def __init__(self, *, vocab_size: int, msa_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        template_feature_size = vocab_size + 14 + 14 + 7
        self.template_single_embedding = nn.Linear(template_feature_size, msa_size)
        self.template_projection = nn.Linear(msa_size, msa_size)

    def forward(self, batch: Any, *, dtype: torch.dtype, device: torch.device) -> tuple[Tensor, Tensor] | None:
        template_batch = _template_batch(batch, dtype=dtype, device=device)
        if template_batch is None:
            return None

        features, torsion_mask = self._features(template_batch)
        activations = F.relu(self.template_single_embedding(features))
        activations = self.template_projection(activations)
        msa_mask = torsion_mask[..., 2] * template_batch.template_mask.unsqueeze(-1).to(dtype=torsion_mask.dtype)
        return activations, msa_mask

    def _features(self, template_batch: AlphaFold2TemplateBatch) -> tuple[Tensor, Tensor]:
        aatype = template_batch.template_aatype.long().clamp(min=0, max=self.vocab_size - 1)
        aatype_one_hot = F.one_hot(aatype, num_classes=self.vocab_size).to(dtype=template_batch.dtype)
        torsion = template_batch.template_torsion_angles_sin_cos
        alt_torsion = template_batch.template_alt_torsion_angles_sin_cos
        torsion_mask = template_batch.template_torsion_angles_mask
        if torsion is None or alt_torsion is None or torsion_mask is None:
            shape = (*aatype.shape, 7)
            torsion = aatype_one_hot.new_zeros(*shape, 2)
            alt_torsion = aatype_one_hot.new_zeros(*shape, 2)
            torsion_mask = template_batch.backbone_torsion_mask()
        features = torch.cat(
            [
                aatype_one_hot,
                torsion.reshape(*aatype.shape, 14).to(dtype=aatype_one_hot.dtype),
                alt_torsion.reshape(*aatype.shape, 14).to(dtype=aatype_one_hot.dtype),
                torsion_mask.to(dtype=aatype_one_hot.dtype),
            ],
            dim=-1,
        )
        return features, torsion_mask.to(dtype=aatype_one_hot.dtype)


class AlphaFold2TemplateBatch:
    def __init__(
        self,
        *,
        template_aatype: Tensor,
        template_mask: Tensor,
        template_all_atom_positions: Tensor,
        template_all_atom_masks: Tensor,
        template_pseudo_beta: Tensor | None,
        template_pseudo_beta_mask: Tensor | None,
        template_torsion_angles_sin_cos: Tensor | None,
        template_alt_torsion_angles_sin_cos: Tensor | None,
        template_torsion_angles_mask: Tensor | None,
        dtype: torch.dtype,
    ) -> None:
        self.template_aatype = template_aatype
        self.template_mask = template_mask
        self.template_all_atom_positions = template_all_atom_positions
        self.template_all_atom_masks = template_all_atom_masks
        self.template_pseudo_beta = template_pseudo_beta
        self.template_pseudo_beta_mask = template_pseudo_beta_mask
        self.template_torsion_angles_sin_cos = template_torsion_angles_sin_cos
        self.template_alt_torsion_angles_sin_cos = template_alt_torsion_angles_sin_cos
        self.template_torsion_angles_mask = template_torsion_angles_mask
        self.dtype = dtype

    def pseudo_beta(self) -> tuple[Tensor, Tensor]:
        if self.template_pseudo_beta is not None and self.template_pseudo_beta_mask is not None:
            return self.template_pseudo_beta.to(dtype=self.dtype), self.template_pseudo_beta_mask.to(dtype=self.dtype)
        ca = residue_constants.atom_order["CA"]
        cb = residue_constants.atom_order["CB"]
        is_gly = self.template_aatype == _GLY_INDEX
        atom_index = torch.where(
            is_gly,
            torch.full_like(self.template_aatype, ca),
            torch.full_like(self.template_aatype, cb),
        )
        gather_index = atom_index[..., None, None].expand(*atom_index.shape, 1, 3)
        pseudo_beta = self.template_all_atom_positions.gather(-2, gather_index).squeeze(-2)
        mask_index = atom_index[..., None].expand(*atom_index.shape, 1)
        pseudo_beta_mask = self.template_all_atom_masks.gather(-1, mask_index).squeeze(-1)
        return pseudo_beta.to(dtype=self.dtype), pseudo_beta_mask.to(dtype=self.dtype)

    def backbone_unit_vector(self) -> tuple[Tensor, Tensor]:
        n = residue_constants.atom_order["N"]
        ca = residue_constants.atom_order["CA"]
        c = residue_constants.atom_order["C"]
        atom_pos = self.template_all_atom_positions.to(dtype=self.dtype)
        atom_mask = self.template_all_atom_masks.to(dtype=self.dtype)
        n_pos = atom_pos[..., n, :]
        ca_pos = atom_pos[..., ca, :]
        c_pos = atom_pos[..., c, :]
        backbone_mask = atom_mask[..., n] * atom_mask[..., ca] * atom_mask[..., c]

        x_axis = F.normalize(c_pos - ca_pos, dim=-1)
        n_axis = F.normalize(n_pos - ca_pos, dim=-1)
        z_axis = F.normalize(torch.cross(x_axis, n_axis, dim=-1), dim=-1)
        y_axis = torch.cross(z_axis, x_axis, dim=-1)
        rotation = torch.stack([x_axis, y_axis, z_axis], dim=-1)

        pseudo_beta, _ = self.pseudo_beta()
        vector = pseudo_beta.unsqueeze(-3) - ca_pos.unsqueeze(-2)
        unit_vector = torch.matmul(rotation.unsqueeze(-3).transpose(-1, -2), vector.unsqueeze(-1)).squeeze(-1)
        unit_vector = F.normalize(unit_vector, dim=-1)
        pair_mask = backbone_mask.unsqueeze(-1) * backbone_mask.unsqueeze(-2)
        unit_vector = torch.nan_to_num(unit_vector) * pair_mask.unsqueeze(-1)
        return unit_vector, pair_mask.to(dtype=self.dtype)

    def backbone_torsion_mask(self) -> Tensor:
        n = residue_constants.atom_order["N"]
        ca = residue_constants.atom_order["CA"]
        c = residue_constants.atom_order["C"]
        atom_mask = self.template_all_atom_masks.to(dtype=self.dtype)
        backbone_mask = atom_mask[..., n] * atom_mask[..., ca] * atom_mask[..., c]
        torsion_mask = atom_mask.new_zeros(*self.template_aatype.shape, 7)
        torsion_mask[..., 2] = backbone_mask
        return torsion_mask


def _template_batch(
    batch: Any,
    reference: Tensor | None = None,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> AlphaFold2TemplateBatch | None:
    if batch.template_aatype is None:
        return None
    if batch.template_all_atom_positions is None or batch.template_all_atom_masks is None:
        raise ValueError("AF2 template embedding requires template_all_atom_positions and template_all_atom_masks")
    if reference is not None:
        dtype = reference.dtype
        device = reference.device
    if dtype is None or device is None:
        raise ValueError("template batch normalization requires dtype and device")

    template_aatype = _ensure_batched_templates(batch.template_aatype, 2, device=device)
    template_all_atom_positions = _ensure_batched_templates(
        batch.template_all_atom_positions,
        4,
        device=device,
        dtype=dtype,
    )
    template_all_atom_masks = _ensure_batched_templates(batch.template_all_atom_masks, 3, device=device, dtype=dtype)
    if batch.template_mask is None:
        template_mask = (template_all_atom_masks.sum(dim=(-1, -2)) > 0).to(dtype=dtype)
    else:
        template_mask = _ensure_batched_templates(batch.template_mask, 1, device=device, dtype=dtype)
    return AlphaFold2TemplateBatch(
        template_aatype=template_aatype,
        template_mask=template_mask,
        template_all_atom_positions=template_all_atom_positions,
        template_all_atom_masks=template_all_atom_masks,
        template_pseudo_beta=_optional_batched(batch.template_pseudo_beta, 3, device=device, dtype=dtype),
        template_pseudo_beta_mask=_optional_batched(batch.template_pseudo_beta_mask, 2, device=device, dtype=dtype),
        template_torsion_angles_sin_cos=_optional_batched(
            batch.template_torsion_angles_sin_cos,
            4,
            device=device,
            dtype=dtype,
        ),
        template_alt_torsion_angles_sin_cos=_optional_batched(
            batch.template_alt_torsion_angles_sin_cos,
            4,
            device=device,
            dtype=dtype,
        ),
        template_torsion_angles_mask=_optional_batched(
            batch.template_torsion_angles_mask,
            3,
            device=device,
            dtype=dtype,
        ),
        dtype=dtype,
    )


def _optional_batched(
    tensor: Tensor | None,
    unbatched_ndim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor | None:
    if tensor is None:
        return None
    return _ensure_batched_templates(tensor, unbatched_ndim, device=device, dtype=dtype)


def _ensure_batched_templates(
    tensor: Tensor,
    unbatched_ndim: int,
    *,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> Tensor:
    tensor = tensor.to(device=device, dtype=dtype) if dtype is not None else tensor.to(device=device)
    if tensor.dim() == unbatched_ndim:
        return tensor.unsqueeze(0)
    if tensor.dim() == unbatched_ndim + 1:
        return tensor
    raise ValueError(f"expected template tensor with {unbatched_ndim} or {unbatched_ndim + 1} dims, got {tensor.dim()}")


def _distogram(positions: Tensor, *, min_bin: float, max_bin: float, num_bins: int) -> Tensor:
    distances = torch.cdist(positions.float(), positions.float()).to(dtype=positions.dtype)
    boundaries = torch.linspace(min_bin, max_bin, steps=num_bins - 1, device=positions.device, dtype=positions.dtype)
    bins = torch.bucketize(distances, boundaries)
    return F.one_hot(bins, num_classes=num_bins)


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


@register_pytree_dataclass
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


@register_pytree_dataclass
@dataclass(slots=True)
class AlphaFold2Representations:
    r"""
    Representation bundle emitted by the Evoformer stack.

    ``single`` and ``pair`` are typed [`Single`][ofoldx.modules.representation.Single] /
    [`Pair`][ofoldx.modules.representation.Pair] carriers at this boundary; unwrap
    ``.tensor`` before feeding the structure module or output heads. Both carriers are
    pytree-registered, so this bundle flattens without a graph break under ``torch.compile``.
    """

    msa: Tensor
    single: Single
    pair: Pair
    msa_first_row: Tensor
    structure_module: Tensor | None = None

    def detach(self) -> AlphaFold2Representations:
        r"""
        Return a copy with every tensor (including the ``Single`` / ``Pair`` carriers) detached.
        """

        return AlphaFold2Representations(
            msa=self.msa.detach(),
            single=Single(self.single.tensor.detach()),
            pair=Pair(self.pair.tensor.detach()),
            msa_first_row=self.msa_first_row.detach(),
            structure_module=None if self.structure_module is None else self.structure_module.detach(),
        )


@register_pytree_dataclass
@dataclass(slots=True)
class AlphaFold2Output:
    r"""
    Training output of [`AlphaFold2Model.forward`][ofoldx.models.alphafold2.AlphaFold2Model.forward].

    Bundles the raw per-token logits and structure trajectory a loss consumes. ``single``
    and ``pair`` are typed carriers at the boundary; ``frames`` is the ``[num_layer, ...]``
    quaternion-affine trajectory used for FAPE supervision.
    """

    single: Single
    pair: Pair
    frames: Tensor
    final_atom_positions: Tensor
    final_atom_mask: Tensor
    sidechain_angles: Tensor
    sidechain_unnormalized_angles: Tensor
    distogram_logits: Tensor
    masked_msa_logits: Tensor
    plddt_logits: Tensor
    pae_logits: Tensor | None
    resolved_logits: Tensor


AlphaFold2Model.input_class = AlphaFold2Input
