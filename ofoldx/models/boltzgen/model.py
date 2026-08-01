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
BoltzGen structure-design, confidence, and inverse-folding models with typed tensor contracts.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import partial
from typing import Literal, NamedTuple, TypeVar, cast

import torch
import torch.nn.functional as F
from multimolecule.graph import knn_graph
from tensordict import TensorClass, tensorclass
from torch import Tensor, nn

from ...compilation import HomogeneousBlockRegions
from ...data.molecular import ComponentRef
from ...data.tokenization import TOKEN_INDEX, TOKENS
from ...errors import RequestRef, fail
from ...metrics import (
    categorical_expectation,
    categorical_mean,
    chain_pair_tm_scores,
    directional_ipsae,
    interface_confidence_weights,
    minimum_confidence_error,
    plddt_from_logits,
    predicted_tm_score,
    tm_weight_from_pae,
)
from ...model import Model, OptionalTensor, TensorCollection
from ...modules.atom.coordinates import gather_observed_coordinates
from ...modules.atom.encoders import (
    AtomPairEncoder,
    AtomToTokenEncoder,
    CoordinateAtomToTokenEncoder,
    TokenToAtomDecoder,
    TrunkToAtomConditioning,
)
from ...modules.atom.window import LocalAtomWindowInfo, LocalAtomWindowing
from ...modules.attention import MultiheadAttention, PairBiasedSelfAttention, PerBlockPairBiasProjection
from ...modules.backend import LayerNorm, resolve_config_backend
from ...modules.denoiser import AtomDenoiserConditioning, AtomDenoiserContext
from ...modules.diffusion.conditioning import SingleConditioning
from ...modules.diffusion.edm import EdmPreconditioner, edm_loss_weight
from ...modules.diffusion.schedule import beta_scale_schedule, karras_noise_schedule
from ...modules.dropout import StructuredDropout
from ...modules.embedding.contact import ContactConditioning
from ...modules.embedding.factorized import FactorizedPairLinear
from ...modules.embedding.position import RelativePositionEmbedder, generate_relative_position_features
from ...modules.embedding.radial import GaussianRadialBasis
from ...modules.feed_forward import FeedForward, SwiGluFeedForward
from ...modules.functional import (
    grouped_softmax,
    indexed_sum,
    pair_mask_from_token_mask,
    pairwise_distance,
    weighted_mean,
)
from ...modules.geometry import rotate_vectors, uniform_random_rotation, weighted_rigid_align
from ...modules.head.confidence import token_confidence_logits
from ...modules.head.distogram import DistogramHead
from ...modules.msa import MsaPairBlock
from ...modules.pair import LineDropoutTrianglePairUpdateBlock
from ...modules.pairformer import LineDropoutPairformerBlock, PairformerStack, PostNormLineDropoutPairformerBlock
from ...modules.transformer import (
    ConditionedTransformerEncoder,
    ConditionedTransformerEncoderBlock,
    PostNormConditionedTransformerEncoderBlock,
    PostNormQueryKeyNormConditionedTransformerEncoderBlock,
    QueryKeyNormConditionedTransformerEncoderBlock,
)
from ...modules.transition import ConditionedTransition, Transition, TripleProductConditionedTransition
from ..conditions import Endpoint, Hotspot, SymbolicBond
from ..processing import project_model_categories, project_model_coordinates, project_model_features
from .config import (
    BoltzGenConfidenceConfig,
    BoltzGenConfig,
    BoltzGenInverseFoldingConfig,
    BoltzGenMiniformerConfig,
    BoltzGenMiniformerResidueTypeConfig,
    BoltzGenResidueTypeConfig,
)

_NUM_BOND_TYPES = 6

# Molecular categories are protein, DNA, RNA, and non-polymer.
_NUM_MOLECULE_TYPES = 4

_NUM_RESIDUE_TYPES = len(TOKENS)

#: The 20 residues BoltzGen designs are the canonical protein classes ``A``--``Y``, in one contiguous block.
_CANONICALS_OFFSET = TOKEN_INDEX["A"]

NUM_CANONICAL_TOKENS = 20

#: The 20 canonical amino acids (three-letter CCD codes) in canonical class order, the identities the designer names.
CANONICAL_TOKENS: tuple[str, ...] = (
    "ALA",
    "CYS",
    "ASP",
    "GLU",
    "PHE",
    "GLY",
    "HIS",
    "ILE",
    "LYS",
    "LEU",
    "MET",
    "ASN",
    "PRO",
    "GLN",
    "ARG",
    "SER",
    "THR",
    "VAL",
    "TRP",
    "TYR",
)


class _BoltzGenStructureModel(Model):
    r"""
    Own the common BoltzGen structure trunk, denoiser, sampler, and heads.

    Args:
        config: Model architecture and inference configuration.
    """

    config: BoltzGenConfig
    _miniformer = False
    _predict_residue_type = False

    def __init__(
        self,
        config: BoltzGenConfig | None = None,
    ) -> None:
        config = cast(BoltzGenConfig, self.config_class()) if config is None else config
        if type(config) is not self.config_class:
            fail("value_invalid", RequestRef("config", type(config).__name__), expected=self.config_class.__name__)
        super().__init__(config)
        self._num_recycles = config.num_recycles
        single_size, pair_size = config.single_size, config.pair_size
        input_embeddings_config, trunk, denoiser_config, sampler = (
            config.input_embeddings,
            config.trunk,
            config.denoiser,
            config.sampler,
        )
        design = config.design

        self.input_embeddings = BoltzGenInputEmbeddings(
            single_size=single_size,
            atom_hidden_size=config.atom_hidden_size,
            atom_pair_hidden_size=config.atom_pair_hidden_size,
            atom_feature_size=config.atom_feature_size,
            num_query_atoms=denoiser_config.num_query_atoms,
            num_key_atoms=denoiser_config.num_key_atoms,
            num_atom_encoder_blocks=input_embeddings_config.num_atom_encoder_blocks,
            num_atom_encoder_attention_heads=input_embeddings_config.num_atom_encoder_attention_heads,
            vocab_size=input_embeddings_config.vocab_size,
            num_method_types=input_embeddings_config.num_method_types,
            num_molecule_types=input_embeddings_config.num_molecule_types,
            add_method_conditioning=input_embeddings_config.add_method_conditioning,
            add_modified_flag=input_embeddings_config.add_modified_flag,
            add_cyclic_flag=input_embeddings_config.add_cyclic_flag,
            add_molecule_type_feature=input_embeddings_config.add_molecule_type_feature,
            add_binding_specification=input_embeddings_config.add_binding_specification,
            add_design_mask=input_embeddings_config.add_design_mask,
            add_ss_specification=input_embeddings_config.add_ss_specification,
            backend="auto",
        )

        self.single_init_projection = nn.Linear(single_size, single_size, bias=False)
        self.pair_left_projection = nn.Linear(single_size, pair_size, bias=False)
        self.pair_right_projection = nn.Linear(single_size, pair_size, bias=False)
        self.relative_position_embedder = RelativePositionEmbedder(config.relative_position_size, pair_size)
        self.token_bond_projection = nn.Linear(1, pair_size, bias=False)
        self.token_bond_type_embedding = nn.Embedding(config.num_bond_types, pair_size)
        self.contact_conditioning = ContactConditioning(
            pair_size, min_cutoff=config.min_conditioning_cutoff, max_cutoff=config.max_conditioning_cutoff
        )
        self.single_recycle_norm = LayerNorm(single_size)
        self.single_recycle_projection = nn.Linear(single_size, single_size, bias=False)
        self.pair_recycle_norm = LayerNorm(pair_size)
        self.pair_recycle_projection = nn.Linear(pair_size, pair_size, bias=False)
        nn.init.zeros_(self.single_recycle_projection.weight)
        nn.init.zeros_(self.pair_recycle_projection.weight)

        # Token-distance conditioning precedes template conditioning in the released constructor.
        self.token_distance_embedder: BoltzGenTokenDistanceEmbedder | None = None
        if design.use_token_distance:
            if self._miniformer:
                self.token_distance_embedder = BoltzGenTokenDistanceEmbedder(
                    pair_stack=partial(_mini_pair_stack, post_layer_norm=design.token_distance_post_layer_norm),
                    pair_size=pair_size,
                    token_distance_size=design.token_distance_size,
                    num_blocks=design.num_token_distance_blocks,
                    num_bins=config.template.num_distogram_bins,
                    num_distance_gaussians=design.num_distance_gaussians,
                    min_dist=config.template.min_distance,
                    max_dist=config.template.max_distance,
                    dropout=design.token_distance_dropout,
                    use_token_distance_features=design.use_token_distance_features,
                    backend="auto",
                )
            else:
                self.token_distance_embedder = BoltzGenTokenDistanceEmbedder(
                    pair_stack=partial(
                        _triangle_pair_stack,
                        triangle_attention_head_size=config.template.triangle_attention_head_size,
                        num_triangle_attention_heads=config.template.num_triangle_attention_heads,
                    ),
                    pair_size=pair_size,
                    token_distance_size=design.token_distance_size,
                    num_blocks=design.num_token_distance_blocks,
                    num_bins=config.template.num_distogram_bins,
                    num_distance_gaussians=design.num_distance_gaussians,
                    min_dist=config.template.min_distance,
                    max_dist=config.template.max_distance,
                    dropout=design.token_distance_dropout,
                    use_token_distance_features=design.use_token_distance_features,
                    backend="auto",
                )

        self.template_embedder: BoltzGenTemplateEmbedder
        if self._miniformer:
            self.template_embedder = BoltzGenTemplateEmbedder(
                pair_stack=partial(_mini_pair_stack, post_layer_norm=design.template_post_layer_norm),
                pair_size=pair_size,
                template_size=design.template_size,
                num_blocks=design.num_template_blocks,
                vocab_size=input_embeddings_config.vocab_size,
                num_bins=config.template.num_distogram_bins,
                min_dist=config.template.min_distance,
                max_dist=config.template.max_distance,
                dropout=design.template_dropout,
                backend="auto",
            )
        else:
            self.template_embedder = BoltzGenTemplateEmbedder(
                pair_stack=partial(
                    _triangle_pair_stack,
                    triangle_attention_head_size=config.template.triangle_attention_head_size,
                    num_triangle_attention_heads=config.template.num_triangle_attention_heads,
                ),
                pair_size=pair_size,
                template_size=design.template_size,
                num_blocks=design.num_template_blocks,
                vocab_size=input_embeddings_config.vocab_size,
                num_bins=config.template.num_distogram_bins,
                min_dist=config.template.min_distance,
                max_dist=config.template.max_distance,
                dropout=design.template_dropout,
                backend="auto",
            )

        self.msa_trunk = (
            BoltzGenMsaTrunk(
                single_size=single_size,
                pair_size=pair_size,
                msa_size=trunk.msa_size,
                num_blocks=trunk.num_msa_blocks,
                use_paired_feature=trunk.use_paired_feature,
                outer_product_hidden_size=trunk.outer_product_hidden_size,
                msa_dropout=trunk.msa_dropout,
                pair_dropout=trunk.msa_pair_dropout,
                num_msa_attention_heads=trunk.num_msa_attention_heads,
                msa_attention_head_size=trunk.msa_attention_head_size,
                triangle_attention_head_size=trunk.msa_triangle_attention_head_size,
                num_triangle_attention_heads=trunk.num_msa_triangle_attention_heads,
                backend="auto",
            )
            if not self._miniformer
            else BoltzGenMiniformerMsaTrunk(
                single_size=single_size,
                pair_size=pair_size,
                msa_size=trunk.msa_size,
                num_blocks=trunk.num_msa_blocks,
                use_paired_feature=trunk.use_paired_feature,
                outer_product_hidden_size=trunk.outer_product_hidden_size,
                msa_dropout=trunk.msa_dropout,
                pair_dropout=trunk.msa_pair_dropout,
                num_msa_attention_heads=trunk.num_msa_attention_heads,
                msa_attention_head_size=trunk.msa_attention_head_size,
                backend="auto",
            )
        )
        self.trunk_stack = (
            BoltzGenMiniformerStack(
                num_blocks=trunk.num_trunk_blocks,
                single_size=single_size,
                pair_size=pair_size,
                num_pair_attention_heads=trunk.num_pair_attention_heads,
                dropout=trunk.pair_dropout,
                post_layer_norm=trunk.post_layer_norm,
                use_single_to_pair=trunk.use_single_to_pair,
                backend="auto",
            )
            if self._miniformer
            else PairformerStack(
                num_blocks=trunk.num_trunk_blocks,
                single_size=single_size,
                pair_size=pair_size,
                triangle_attention_head_size=trunk.triangle_attention_head_size,
                num_triangle_attention_heads=trunk.num_triangle_attention_heads,
                num_pair_attention_heads=trunk.num_pair_attention_heads,
                dropout=trunk.pair_dropout,
                block=PostNormLineDropoutPairformerBlock if trunk.post_layer_norm else LineDropoutPairformerBlock,
                backend="auto",
            )
        )
        self.distogram_head = DistogramHead(pair_size=pair_size, num_bins=config.num_distogram_bins)

        self.denoiser_conditioning = AtomDenoiserConditioning(
            atom_feature_bias=True,
            single_size=single_size,
            pair_size=pair_size,
            atom_hidden_size=denoiser_config.atom_hidden_size,
            atom_pair_hidden_size=denoiser_config.atom_pair_hidden_size,
            atom_feature_size=config.atom_feature_size,
            num_query_atoms=denoiser_config.num_query_atoms,
            num_key_atoms=denoiser_config.num_key_atoms,
            num_atom_encoder_blocks=denoiser_config.num_atom_encoder_blocks,
            num_atom_encoder_attention_heads=denoiser_config.num_atom_attention_heads,
            num_atom_decoder_blocks=denoiser_config.num_atom_decoder_blocks,
            num_atom_decoder_attention_heads=denoiser_config.num_atom_attention_heads,
            num_token_encoder_blocks=denoiser_config.num_token_encoder_blocks,
            num_token_encoder_attention_heads=denoiser_config.num_token_encoder_attention_heads,
            num_conditioning_transitions=denoiser_config.num_conditioning_transitions,
        )
        denoiser_type = BoltzGenResidueTypeDenoiser if self._predict_residue_type else BoltzGenDenoiser
        denoiser = denoiser_type(
            single_size=single_size,
            atom_hidden_size=denoiser_config.atom_hidden_size,
            fourier_embedding_size=config.fourier_embedding_size,
            num_atom_encoder_blocks=denoiser_config.num_atom_encoder_blocks,
            num_atom_encoder_attention_heads=denoiser_config.num_atom_attention_heads,
            num_token_encoder_blocks=denoiser_config.num_token_encoder_blocks,
            num_token_encoder_attention_heads=denoiser_config.num_token_encoder_attention_heads,
            num_atom_decoder_blocks=denoiser_config.num_atom_decoder_blocks,
            num_atom_decoder_attention_heads=denoiser_config.num_atom_attention_heads,
            num_conditioning_transitions=denoiser_config.num_conditioning_transitions,
            transformer_size=denoiser_config.transformer_size,
            post_layer_norm=denoiser_config.post_layer_norm,
            use_qk_norm=denoiser_config.use_qk_norm,
            num_coordinate_gaussians=denoiser_config.num_coordinate_gaussians,
            num_residue_types=input_embeddings_config.vocab_size,
            backend="auto",
        )
        self.coordinate_diffusion = BoltzGenCoordinateDiffusion(
            denoiser,
            sigma_data=denoiser_config.sigma_data,
            min_noise_level=sampler.min_noise_level,
            max_noise_level=sampler.max_noise_level,
            rho=sampler.rho,
            churn_rate=sampler.churn_rate,
            min_churn_noise_level=sampler.min_churn_noise_level,
            noise_scale=sampler.noise_scale,
            step_scale=sampler.step_scale,
            step_scale_function=sampler.step_scale_function,
            noise_scale_function=sampler.noise_scale_function,
            min_step_scale=sampler.min_step_scale,
            max_step_scale=sampler.max_step_scale,
            step_scale_alpha=sampler.step_scale_alpha,
            step_scale_beta=sampler.step_scale_beta,
            min_noise_scale=sampler.min_noise_scale,
            max_noise_scale=sampler.max_noise_scale,
            noise_scale_alpha=sampler.noise_scale_alpha,
            noise_scale_beta=sampler.noise_scale_beta,
            step_scale_random=sampler.step_scale_random,
            num_sampling_steps=sampler.num_sampling_steps,
            alignment_reverse_diff=denoiser_config.alignment_reverse_diff,
            sampling_schedule=sampler.sampling_schedule,
            time_dilation=sampler.time_dilation,
            time_dilation_start=sampler.time_dilation_start,
            time_dilation_end=sampler.time_dilation_end,
        )
        self.b_factor_head = nn.Linear(single_size, config.num_distogram_bins)
        self.post_init()

    def initialize_parameters(self) -> None:
        r"""
        Apply the BoltzGen residual-update initialization.
        """

        _zero_initialize_boltzgen_updates(self)

    @torch.no_grad()
    def _sample(
        self,
        input: BoltzGenInput,
        *,
        num_recycles: int | None = None,
        num_samples: int = 1,
        num_sampling_steps: int | None = None,
        generator: torch.Generator | None = None,
        step_scale: float | Tensor | None = None,
        noise_scale: float | Tensor | None = None,
        step_scale_schedule: Tensor | None = None,
        noise_scale_schedule: Tensor | None = None,
        return_trajectory: bool = False,
    ) -> BoltzGenSamplingOutput | BoltzGenResidueTypeSamplingOutput:
        r"""
        Draw reverse-diffusion structure samples from a preprocessed input.

        Args:
            input: Typed model input.
            num_recycles: Number of complete trunk passes. ``None`` uses the
                configured value.
            num_samples: Number of reverse-diffusion samples to draw.
            num_sampling_steps: Optional number of EDM sampling steps. Defaults to the sampler config value.
            generator: Optional PyTorch random generator for diffusion noise.
            step_scale: Optional scalar or per-sample step-scale override.
            noise_scale: Optional scalar or per-sample noise-scale override.
            step_scale_schedule: Actual per-transition step scales. The leading time
                axis precedes broadcast batch/sample axes and overrides ``step_scale``.
            noise_scale_schedule: Actual per-transition noise scales, with the same
                time-axis convention and precedence over ``noise_scale``.
            return_trajectory: Whether to retain the current-coordinate and
                denoised-coordinate histories.

        """

        input = cast(BoltzGenInput, normalize_boltzgen_input(input, dtype=self.single_init_projection.weight.dtype))
        trunk = self._run_trunk(input, num_recycles=num_recycles)
        conditioning = self._build_denoiser_conditioning(trunk, input)
        coordinates, residue_type_logits, coordinate_trajectory, denoised_coordinate_trajectory = (
            self.coordinate_diffusion.sample(
                input.atom_mask,
                reference_positions=input.reference_positions,
                num_sampling_steps=num_sampling_steps,
                num_samples=num_samples,
                generator=generator,
                step_scale=step_scale,
                noise_scale=noise_scale,
                step_scale_schedule=step_scale_schedule,
                noise_scale_schedule=noise_scale_schedule,
                return_trajectory=return_trajectory,
                single_inputs=trunk.single_inputs,
                single_trunk=trunk.single,
                conditioning=conditioning,
                atom_to_token=input.atom_to_token,
                token_mask=input.token_mask,
            )
        )
        batch_size = input.token_mask.shape[0]
        values = {
            "coordinates": coordinates,
            "distogram_logits": trunk.distogram_logits,
            "b_factor_logits": self.b_factor_head(trunk.single),
            **({"coordinate_trajectory": coordinate_trajectory} if coordinate_trajectory is not None else {}),
            **(
                {"denoised_coordinate_trajectory": denoised_coordinate_trajectory}
                if denoised_coordinate_trajectory is not None
                else {}
            ),
            "batch_size": [batch_size],
        }
        if residue_type_logits is not None:
            return BoltzGenResidueTypeSamplingOutput(
                **values,
                residue_type_logits=residue_type_logits,
            )
        return BoltzGenSamplingOutput(
            **values,
        )

    def _run_trunk(
        self,
        input: BoltzGenInput,
        *,
        num_recycles: int | None = None,
    ) -> _BoltzGenTrunkOutput:
        r"""
        Run the structure trunk.

        Args:
            input: Typed model input.
            num_recycles: Number of complete trunk passes. ``None`` uses the
                configured value.

        Returns:
            Trunk representations and distogram logits.

        """

        input = normalize_boltzgen_input(input, dtype=self.single_init_projection.weight.dtype)
        num_recycles = operator.index(self._num_recycles if num_recycles is None else num_recycles)
        if num_recycles < 1:
            fail("value_invalid", RequestRef("num_recycles", str(num_recycles)))
        single_inputs = self.input_embeddings(
            atom_features=input.atom_features,
            reference_positions=input.reference_positions,
            atom_mask=input.atom_mask,
            reference_space_id=input.reference_space_id,
            atom_to_token=input.atom_to_token,
            residue_type_features=input.residue_type_features,
            msa_profile=input.msa_profile,
            deletion_mean=input.deletion_mean,
            method_feature=input.method_feature,
            modified=input.modified,
            cyclic_period=input.cyclic_period,
            molecule_type=input.molecule_type,
            binding_type=input.binding_type,
            design_mask=input.design_mask,
            ss_type=input.ss_type,
        )

        single_init = self.single_init_projection(single_inputs)
        pair_init = self.pair_left_projection(single_inputs).unsqueeze(-2)
        pair_init = pair_init + self.pair_right_projection(single_inputs).unsqueeze(-3)
        relative_features = generate_relative_position_features(
            residue_index=input.residue_index,
            chain_index=input.chain_index,
            entity_index=input.entity_index,
            symmetry_index=input.symmetry_index,
            token_index=input.token_index,
            cyclic_period=input.cyclic_period,
            dtype=single_init.dtype,
        )
        relative_position_encoding = self.relative_position_embedder(relative_features)
        pair_init = pair_init + relative_position_encoding
        pair_init = pair_init + self.token_bond_projection(input.token_bonds.to(pair_init.dtype))
        pair_init = pair_init + self.token_bond_type_embedding(input.type_bonds.long())
        pair_init = pair_init + self.contact_conditioning(
            input.contact_conditioning.to(pair_init.dtype), input.contact_threshold.to(pair_init.dtype)
        )

        token_mask = input.token_mask
        pair_mask = pair_mask_from_token_mask(token_mask, backend=self.backend)
        single = torch.zeros_like(single_init)
        pair = torch.zeros_like(pair_init)
        for cycle in range(num_recycles):
            grad_enabled = torch.is_grad_enabled() and cycle == num_recycles - 1
            with torch.set_grad_enabled(grad_enabled):
                single = single_init + self.single_recycle_projection(self.single_recycle_norm(single))
                pair = pair_init + self.pair_recycle_projection(self.pair_recycle_norm(pair))
                if self.token_distance_embedder is not None:
                    pair = pair + self.token_distance_embedder(
                        pair,
                        input,
                        relative_position_encoding,
                        pair_mask=pair_mask,
                    )
                if input.template_residue_type_features is not None:
                    pair = pair + self.template_embedder(pair, input, pair_mask=pair_mask)
                if input.msa_tokens is not None:
                    pair = pair + self.msa_trunk(
                        pair,
                        single_inputs,
                        msa_tokens=input.msa_tokens,
                        has_deletion=input.has_deletion,
                        deletion_value=input.deletion_value,
                        is_paired=input.msa_paired,
                        msa_mask=input.msa_mask,
                        token_mask=token_mask,
                    )
                single, pair = self.trunk_stack(single, pair, pair_mask=pair_mask)

        return _BoltzGenTrunkOutput(
            single_inputs=single_inputs,
            single=single,
            pair=pair,
            relative_position_encoding=relative_position_encoding,
            distogram_logits=self.distogram_head(pair),
        )

    def _build_denoiser_conditioning(
        self,
        trunk: _BoltzGenTrunkOutput,
        input: BoltzGenInput,
    ) -> AtomDenoiserContext:
        r"""
        Precompute the per-structure diffusion conditioning reused by denoising.
        """

        return self.denoiser_conditioning(
            trunk.single,
            trunk.pair,
            trunk.relative_position_encoding,
            atom_features=input.atom_features,
            reference_positions=input.reference_positions,
            atom_mask=input.atom_mask,
            reference_space_id=input.reference_space_id,
            atom_to_token=input.atom_to_token,
        )


class _BoltzGenDenoisingModel(_BoltzGenStructureModel):
    r"""
    Generate conditioned biomolecular structures with BoltzGen.

    Accepts [`BoltzGenInput`][ofoldx.models.boltzgen.model.BoltzGenInput] and returns
    [`BoltzGenOutput`][ofoldx.models.boltzgen.model.BoltzGenOutput].

    """

    def _denoise_training(
        self,
        input: BoltzGenInput,
        target: BoltzGenStructureTarget,
        *,
        num_samples: int = 1,
        sample_chunk_size: int | None = None,
        synchronize_noise_levels: bool = False,
        coordinate_augmentation: bool = True,
        augmentation_translation_scale: float = 1.0,
        noise_log_mean: float = -1.2,
        noise_log_std: float = 1.5,
        num_recycles: int | None = None,
        generator: torch.Generator | None = None,
    ) -> _BoltzGenTrainingOutput:
        r"""
        Run stochastic denoising samples for structure training.

        ``B`` is batch size, ``S`` is ``num_samples``, ``N`` is token count,
        and ``A`` is atom count.

        Args:
            input: Inference features and canonical molecular graph.
            target: Structure-training labels with one or ``S`` conformers.
            num_samples: Number of independent noisy denoising samples per batch item.
            sample_chunk_size: Maximum samples per input evaluated by one
                denoiser call. ``None`` evaluates all samples together.
            synchronize_noise_levels: Whether all samples of one batch item share
                one sampled noise level.
            coordinate_augmentation: Whether to apply a random rigid transform to
                each repeated target after centering.
            augmentation_translation_scale: Standard deviation of augmentation
                translations in angstroms.
            noise_log_mean: Mean of the log-normal noise distribution.
            noise_log_std: Standard deviation of the log-normal noise distribution.
            num_recycles: Number of complete trunk passes. ``None`` uses the
                configured value.
            generator: Random generator used for rigid augmentation, noise-level
                sampling, and coordinate noise.

        Returns:
            Internal denoising tensors used to construct the concrete public
            output type.
        """

        num_samples = operator.index(num_samples)
        if num_samples < 1:
            fail("value_invalid", RequestRef("num_samples", str(num_samples)))
        if sample_chunk_size is None:
            sample_chunk_size = num_samples
        else:
            sample_chunk_size = operator.index(sample_chunk_size)
            if sample_chunk_size < 1:
                fail("value_invalid", RequestRef("sample_chunk_size", str(sample_chunk_size)))
            sample_chunk_size = min(sample_chunk_size, num_samples)
        input = cast(BoltzGenInput, normalize_boltzgen_input(input, dtype=self.single_init_projection.weight.dtype))
        trunk = self._run_trunk(input, num_recycles=num_recycles)
        conditioning = self._build_denoiser_conditioning(trunk, input)
        conformers = target.coordinates.shape[1]
        model_index = target.coordinate_atom_index[:, None, :].expand(-1, conformers, -1)
        model_mask = input.atom_mask.bool()[:, None, :].expand_as(model_index)
        target_coordinates, target_atom_mask = gather_observed_coordinates(
            target.coordinates,
            target.coordinate_mask[:, None, :].expand(target.coordinates.shape[:-1]),
            model_index,
            model_mask,
        )
        target_coordinates = project_model_coordinates(
            target_coordinates,
            target_atom_mask,
            dtype=trunk.single.dtype,
            feature="BoltzGen training coordinates",
        )
        (
            denoised_coordinates,
            augmented_target_coordinates,
            noise_level,
            residue_type_logits,
        ) = self.coordinate_diffusion.denoise_training(
            target_coordinates,
            input.atom_mask,
            num_samples=num_samples,
            sample_chunk_size=sample_chunk_size,
            synchronize_noise_levels=synchronize_noise_levels,
            coordinate_augmentation=coordinate_augmentation,
            augmentation_translation_scale=augmentation_translation_scale,
            noise_log_mean=noise_log_mean,
            noise_log_std=noise_log_std,
            generator=generator,
            single_inputs=trunk.single_inputs,
            single_trunk=trunk.single,
            conditioning=conditioning,
            atom_to_token=input.atom_to_token,
            token_mask=input.token_mask,
        )
        return _BoltzGenTrainingOutput(
            coordinates=denoised_coordinates,
            distogram_logits=trunk.distogram_logits,
            b_factor_logits=self.b_factor_head(trunk.single),
            augmented_target_coordinates=augmented_target_coordinates,
            noise_level=noise_level,
            residue_type_logits=residue_type_logits,
        )


class BoltzGenModel(_BoltzGenDenoisingModel):
    r"""
    Generate conditioned structures with the full Pairformer BoltzGen architecture.
    """

    config_class = BoltzGenConfig
    config: BoltzGenConfig

    def forward(
        self,
        input: BoltzGenInput,
        target: BoltzGenStructureTarget,
        *,
        num_samples: int = 1,
        sample_chunk_size: int | None = None,
        synchronize_noise_levels: bool = False,
        coordinate_augmentation: bool = True,
        augmentation_translation_scale: float = 1.0,
        noise_log_mean: float = -1.2,
        noise_log_std: float = 1.5,
        num_recycles: int | None = None,
        generator: torch.Generator | None = None,
    ) -> BoltzGenOutput:
        output = self._denoise_training(
            input,
            target,
            num_samples=num_samples,
            sample_chunk_size=sample_chunk_size,
            synchronize_noise_levels=synchronize_noise_levels,
            coordinate_augmentation=coordinate_augmentation,
            augmentation_translation_scale=augmentation_translation_scale,
            noise_log_mean=noise_log_mean,
            noise_log_std=noise_log_std,
            num_recycles=num_recycles,
            generator=generator,
        )
        return BoltzGenOutput(
            coordinates=output.coordinates,
            distogram_logits=output.distogram_logits,
            b_factor_logits=output.b_factor_logits,
            augmented_target_coordinates=output.augmented_target_coordinates,
            noise_level=output.noise_level,
            batch_size=[output.coordinates.shape[0]],
        )

    def sample(
        self,
        input: BoltzGenInput,
        *,
        num_samples: int = 1,
        num_sampling_steps: int | None = None,
        num_recycles: int | None = None,
        generator: torch.Generator | None = None,
        step_scale: float | Tensor | None = None,
        noise_scale: float | Tensor | None = None,
        step_scale_schedule: Tensor | None = None,
        noise_scale_schedule: Tensor | None = None,
        return_trajectory: bool = False,
    ) -> BoltzGenSamplingOutput:
        return cast(
            BoltzGenSamplingOutput,
            self._sample(
                input,
                num_samples=num_samples,
                num_sampling_steps=num_sampling_steps,
                num_recycles=num_recycles,
                generator=generator,
                step_scale=step_scale,
                noise_scale=noise_scale,
                step_scale_schedule=step_scale_schedule,
                noise_scale_schedule=noise_scale_schedule,
                return_trajectory=return_trajectory,
            ),
        )


class BoltzGenResidueTypeModel(_BoltzGenDenoisingModel):
    r"""
    Generate structures and residue-type logits with the full Pairformer architecture.
    """

    config_class = BoltzGenResidueTypeConfig
    config: BoltzGenResidueTypeConfig
    _predict_residue_type = True

    def forward(
        self,
        input: BoltzGenInput,
        target: BoltzGenStructureTarget,
        *,
        num_samples: int = 1,
        sample_chunk_size: int | None = None,
        synchronize_noise_levels: bool = False,
        coordinate_augmentation: bool = True,
        augmentation_translation_scale: float = 1.0,
        noise_log_mean: float = -1.2,
        noise_log_std: float = 1.5,
        num_recycles: int | None = None,
        generator: torch.Generator | None = None,
    ) -> BoltzGenResidueTypeOutput:
        output = self._denoise_training(
            input,
            target,
            num_samples=num_samples,
            sample_chunk_size=sample_chunk_size,
            synchronize_noise_levels=synchronize_noise_levels,
            coordinate_augmentation=coordinate_augmentation,
            augmentation_translation_scale=augmentation_translation_scale,
            noise_log_mean=noise_log_mean,
            noise_log_std=noise_log_std,
            num_recycles=num_recycles,
            generator=generator,
        )
        return BoltzGenResidueTypeOutput(
            coordinates=output.coordinates,
            distogram_logits=output.distogram_logits,
            b_factor_logits=output.b_factor_logits,
            augmented_target_coordinates=output.augmented_target_coordinates,
            noise_level=output.noise_level,
            residue_type_logits=cast(Tensor, output.residue_type_logits),
            batch_size=[output.coordinates.shape[0]],
        )

    def sample(
        self,
        input: BoltzGenInput,
        *,
        num_samples: int = 1,
        num_sampling_steps: int | None = None,
        num_recycles: int | None = None,
        generator: torch.Generator | None = None,
        step_scale: float | Tensor | None = None,
        noise_scale: float | Tensor | None = None,
        step_scale_schedule: Tensor | None = None,
        noise_scale_schedule: Tensor | None = None,
        return_trajectory: bool = False,
    ) -> BoltzGenResidueTypeSamplingOutput:
        return cast(
            BoltzGenResidueTypeSamplingOutput,
            self._sample(
                input,
                num_samples=num_samples,
                num_sampling_steps=num_sampling_steps,
                num_recycles=num_recycles,
                generator=generator,
                step_scale=step_scale,
                noise_scale=noise_scale,
                step_scale_schedule=step_scale_schedule,
                noise_scale_schedule=noise_scale_schedule,
                return_trajectory=return_trajectory,
            ),
        )


class BoltzGenMiniformerModel(BoltzGenModel):
    r"""
    Generate conditioned structures with the reduced Miniformer architecture.
    """

    config_class = BoltzGenMiniformerConfig
    config: BoltzGenMiniformerConfig
    _miniformer = True


class BoltzGenMiniformerResidueTypeModel(BoltzGenResidueTypeModel):
    r"""
    Generate structures and residue-type logits with the reduced Miniformer architecture.
    """

    config_class = BoltzGenMiniformerResidueTypeConfig
    config: BoltzGenMiniformerResidueTypeConfig
    _miniformer = True
    _predict_residue_type = True


class BoltzGenConfidenceModel(_BoltzGenStructureModel):
    r"""
    Score explicit or newly sampled structures with the BoltzGen confidence equation.

    The model owns the ordinary BoltzGen structure trunk and sampler plus the
    independently parameterized confidence head. Its raw ``forward`` consumes
    explicit ``(B, S, A, 3)`` coordinates; ``sample_with_confidence`` samples
    and scores from one shared trunk execution.
    """

    config_class = BoltzGenConfidenceConfig
    config: BoltzGenConfidenceConfig

    def __init__(self, config: BoltzGenConfidenceConfig | None = None) -> None:
        config = BoltzGenConfidenceConfig() if config is None else config
        super().__init__(config)
        confidence = config.confidence
        self.confidence_head = BoltzGenConfidenceHead(
            single_size=config.single_size,
            pair_size=config.pair_size,
            num_blocks=confidence.num_pairformer_blocks,
            num_pair_attention_heads=confidence.num_pair_attention_heads,
            triangle_attention_head_size=confidence.triangle_attention_head_size,
            num_triangle_attention_heads=confidence.num_triangle_attention_heads,
            pair_dropout=confidence.pair_dropout,
            post_layer_norm=confidence.post_layer_norm,
            num_distogram_bins=confidence.num_distogram_bins,
            max_distance=confidence.max_distance,
            token_level_confidence=confidence.token_level_confidence,
            add_single_input=confidence.add_single_input,
            add_single_to_pair_product=confidence.add_single_to_pair_product,
            use_separate_heads=confidence.use_separate_heads,
            num_plddt_bins=confidence.num_plddt_bins,
            num_pae_bins=confidence.num_pae_bins,
            num_pde_bins=confidence.num_pde_bins,
            return_latent_features=confidence.return_latent_features,
            backend="auto",
        )
        _zero_initialize_boltzgen_updates(self.confidence_head)

    def forward(
        self,
        input: BoltzGenInput,
        *,
        coordinates: Tensor,
        num_recycles: int | None = None,
    ) -> BoltzGenConfidenceOutput:
        r"""
        Score explicit sampled coordinates.

        Args:
            input: Processed BoltzGen structure input.
            coordinates: Sampled atom coordinates with shape ``(B, S, A, 3)``.
            num_recycles: Number of complete trunk passes. ``None`` uses the
                configured value.

        Returns:
            Complete learned logits and derived confidence tensors.
        """

        input = cast(BoltzGenInput, normalize_boltzgen_input(input, dtype=self.single_init_projection.weight.dtype))
        trunk = self._run_trunk(input, num_recycles=num_recycles)
        return self._score_confidence(input, trunk=trunk, coordinates=coordinates)

    @torch.no_grad()
    def sample_with_confidence(
        self,
        input: BoltzGenInput,
        *,
        num_samples: int = 1,
        num_sampling_steps: int | None = None,
        num_recycles: int | None = None,
        generator: torch.Generator | None = None,
        step_scale: float | Tensor | None = None,
        noise_scale: float | Tensor | None = None,
        step_scale_schedule: Tensor | None = None,
        noise_scale_schedule: Tensor | None = None,
        return_trajectory: bool = False,
    ) -> BoltzGenSamplingWithConfidenceOutput:
        r"""
        Sample coordinates and score them without recomputing the structure trunk.
        """

        input = cast(BoltzGenInput, normalize_boltzgen_input(input, dtype=self.single_init_projection.weight.dtype))
        trunk = self._run_trunk(input, num_recycles=num_recycles)
        conditioning = self._build_denoiser_conditioning(trunk, input)
        coordinates, _, coordinate_trajectory, denoised_coordinate_trajectory = self.coordinate_diffusion.sample(
            input.atom_mask,
            reference_positions=input.reference_positions,
            num_sampling_steps=num_sampling_steps,
            num_samples=num_samples,
            generator=generator,
            step_scale=step_scale,
            noise_scale=noise_scale,
            step_scale_schedule=step_scale_schedule,
            noise_scale_schedule=noise_scale_schedule,
            return_trajectory=return_trajectory,
            single_inputs=trunk.single_inputs,
            single_trunk=trunk.single,
            conditioning=conditioning,
            atom_to_token=input.atom_to_token,
            token_mask=input.token_mask,
        )
        batch_size = input.token_mask.shape[0]
        b_factor_logits = self.b_factor_head(trunk.single)
        confidence = self._score_confidence(input, trunk=trunk, coordinates=coordinates)
        return BoltzGenSamplingWithConfidenceOutput(
            coordinates=coordinates,
            distogram_logits=trunk.distogram_logits,
            b_factor_logits=b_factor_logits,
            confidence=confidence,
            **({"coordinate_trajectory": coordinate_trajectory} if coordinate_trajectory is not None else {}),
            **(
                {"denoised_coordinate_trajectory": denoised_coordinate_trajectory}
                if denoised_coordinate_trajectory is not None
                else {}
            ),
            batch_size=[batch_size],
        )

    def _score_confidence(
        self,
        input: BoltzGenInput,
        *,
        trunk: _BoltzGenTrunkOutput,
        coordinates: Tensor,
    ) -> BoltzGenConfidenceOutput:
        return self.confidence_head(
            trunk.single_inputs.detach(),
            trunk.single.detach(),
            trunk.pair.detach(),
            coordinates.detach(),
            input=input,
            pred_distogram_logits=trunk.distogram_logits.detach(),
        )


class BoltzGenInverseFoldingModel(Model):
    r"""
    Design sequences for fixed biomolecular structures with BoltzGen.

    Accepts [`BoltzGenInverseFoldingInput`][ofoldx.models.boltzgen.model.BoltzGenInverseFoldingInput] and returns
    [`BoltzGenInverseFoldingOutput`][ofoldx.models.boltzgen.model.BoltzGenInverseFoldingOutput].

    Args:
        config: Model architecture and inference configuration.
    """

    config_class = BoltzGenInverseFoldingConfig
    config: BoltzGenInverseFoldingConfig

    def __init__(self, config: BoltzGenInverseFoldingConfig | None = None) -> None:
        config = BoltzGenInverseFoldingConfig() if config is None else config
        super().__init__(config)
        inverse_folding = config.inverse_folding
        self.input_embeddings = BoltzGenInputEmbeddings(
            single_size=config.single_size,
            atom_hidden_size=config.atom_hidden_size,
            atom_pair_hidden_size=config.atom_pair_hidden_size,
            atom_feature_size=config.atom_feature_size,
            num_query_atoms=inverse_folding.num_query_atoms,
            num_key_atoms=inverse_folding.num_key_atoms,
            num_atom_encoder_blocks=inverse_folding.num_atom_encoder_blocks,
            num_atom_encoder_attention_heads=inverse_folding.num_atom_encoder_attention_heads,
            vocab_size=config.input_embeddings.vocab_size,
            num_method_types=config.input_embeddings.num_method_types,
            num_molecule_types=config.input_embeddings.num_molecule_types,
            add_method_conditioning=config.input_embeddings.add_method_conditioning,
            add_modified_flag=config.input_embeddings.add_modified_flag,
            add_cyclic_flag=config.input_embeddings.add_cyclic_flag,
            add_molecule_type_feature=config.input_embeddings.add_molecule_type_feature,
            add_binding_specification=config.input_embeddings.add_binding_specification,
            add_design_mask=config.input_embeddings.add_design_mask,
            add_ss_specification=config.input_embeddings.add_ss_specification,
            backend="auto",
        )
        self.inverse_folding_encoder = BoltzGenInverseFoldingEncoder(
            single_size=config.single_size,
            node_size=inverse_folding.node_size,
            edge_size=inverse_folding.edge_size,
            hidden_size=inverse_folding.hidden_size,
            num_blocks=inverse_folding.num_encoder_blocks,
            num_attention_heads=inverse_folding.num_attention_heads,
            num_neighbors=inverse_folding.num_neighbors,
            dropout=inverse_folding.dropout,
            softmax_dropout=inverse_folding.softmax_dropout,
            coordinate_noise=inverse_folding.coordinate_noise,
        )
        self.inverse_folding_decoder = BoltzGenInverseFoldingDecoder(
            node_size=inverse_folding.node_size,
            edge_size=inverse_folding.edge_size,
            hidden_size=inverse_folding.hidden_size,
            num_blocks=inverse_folding.num_decoder_blocks,
            num_attention_heads=inverse_folding.num_attention_heads,
            dropout=inverse_folding.dropout,
            softmax_dropout=inverse_folding.softmax_dropout,
        )
        self.post_init()

    def initialize_parameters(self) -> None:
        r"""
        Apply the BoltzGen residual-update initialization.
        """

        _zero_initialize_boltzgen_updates(self)

    def forward(
        self,
        input: BoltzGenInverseFoldingInput,
        target_residue_type: Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> BoltzGenInverseFoldingOutput:
        r"""
        Predict residue-type logits from a fixed structure.

        Args:
            input: Processed inverse-folding input with ``B`` batch items and ``N``
                tokens.
            target_residue_type: Unmasked residue distributions with shape
                ``(B, N, V)`` used only for stochastic teacher forcing.
            generator: Random generator used for training-time coordinate noise
                and teacher-forcing visibility.

        Returns:
            A [`BoltzGenInverseFoldingOutput`][] whose ``logits`` have shape
                ``(B, N, V)``, where ``V`` is the residue vocabulary size.
        """

        input = cast(
            BoltzGenInverseFoldingInput,
            normalize_boltzgen_input(input, dtype=self.input_embeddings.atom_feature_projection.weight.dtype),
        )
        single_inputs = self.input_embeddings(
            atom_features=input.atom_features,
            reference_positions=input.reference_positions,
            atom_mask=input.atom_mask,
            reference_space_id=input.reference_space_id,
            atom_to_token=input.atom_to_token,
            residue_type_features=input.residue_type_features,
            msa_profile=input.msa_profile,
            deletion_mean=input.deletion_mean,
            method_feature=input.method_feature,
            modified=input.modified,
            cyclic_period=input.cyclic_period,
            molecule_type=input.molecule_type,
        )
        edge_index, valid_mask, node, edge = self.inverse_folding_encoder(
            input,
            single_inputs=single_inputs,
            generator=generator,
        )
        logits = self.inverse_folding_decoder(
            node,
            edge,
            edge_index,
            valid_mask,
            target_residue_type,
            generator=generator,
        )
        return BoltzGenInverseFoldingOutput(logits=logits, batch_size=[logits.shape[0]])

    @torch.no_grad()
    def design(
        self,
        input: BoltzGenInverseFoldingInput,
        *,
        temperature: float | None = 0.1,
        tie_symmetric_sequences: bool = True,
        generator: torch.Generator | None = None,
    ) -> BoltzGenInverseFoldingDesignOutput:
        r"""
        Sample residue identities for a fixed structure.

        Args:
            input: Processed inverse-folding input with ``B`` batch items and ``N``
                tokens.
            temperature: Categorical sampling temperature. ``None`` selects the
                highest-logit residue at each decoded position.
            tie_symmetric_sequences: Whether symmetric copies share one sample.
            generator: Random generator used for training-time coordinate noise,
                decode order, and categorical sampling.

        Returns:
            A [`BoltzGenInverseFoldingDesignOutput`][] with ``logits`` of shape
                ``(B, N, V)`` and sampled ``sequence_tokens`` of shape ``(B, N)``.
        """

        input = cast(
            BoltzGenInverseFoldingInput,
            normalize_boltzgen_input(input, dtype=self.input_embeddings.atom_feature_projection.weight.dtype),
        )
        single_inputs = self.input_embeddings(
            atom_features=input.atom_features,
            reference_positions=input.reference_positions,
            atom_mask=input.atom_mask,
            reference_space_id=input.reference_space_id,
            atom_to_token=input.atom_to_token,
            residue_type_features=input.residue_type_features,
            msa_profile=input.msa_profile,
            deletion_mean=input.deletion_mean,
            method_feature=input.method_feature,
            modified=input.modified,
            cyclic_period=input.cyclic_period,
            molecule_type=input.molecule_type,
        )
        edge_index, valid_mask, node, edge = self.inverse_folding_encoder(
            input,
            single_inputs=single_inputs,
            generator=generator,
        )
        logits, residue_type = self.inverse_folding_decoder.sample(
            node,
            edge,
            edge_index,
            valid_mask,
            input,
            temperature=temperature,
            tie_symmetric_sequences=tie_symmetric_sequences,
            generator=generator,
        )
        return BoltzGenInverseFoldingDesignOutput(
            logits=logits,
            sequence_tokens=residue_type.argmax(dim=-1),
            batch_size=[logits.shape[0]],
        )


class BoltzGenInputEmbeddings(nn.Module):
    r"""
    Compose atom-derived token states with token-level encodings and conditioning embeddings.
    """

    def __init__(
        self,
        single_size: int = 384,
        atom_hidden_size: int = 128,
        atom_pair_hidden_size: int = 16,
        atom_feature_size: int = 388,
        num_query_atoms: int = 32,
        num_key_atoms: int = 128,
        num_atom_encoder_blocks: int = 3,
        num_atom_encoder_attention_heads: int = 4,
        vocab_size: int = len(TOKENS),
        num_method_types: int = 12,
        num_molecule_types: int = 4,
        add_method_conditioning: bool = True,
        add_modified_flag: bool = True,
        add_cyclic_flag: bool = True,
        add_molecule_type_feature: bool = True,
        add_binding_specification: bool = False,
        add_design_mask: bool = False,
        add_ss_specification: bool = False,
        num_binding_types: int = 3,
        num_ss_types: int = 4,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.windowing = LocalAtomWindowing(num_query_atoms, num_key_atoms)
        self.atom_pair_encoder = AtomPairEncoder(atom_hidden_size, atom_pair_hidden_size)
        self.atom_feature_projection = nn.Linear(atom_feature_size, atom_hidden_size)
        self.atom_pair_bias_projection = PerBlockPairBiasProjection(
            pair_size=atom_pair_hidden_size,
            num_blocks=num_atom_encoder_blocks,
            num_attention_heads=num_atom_encoder_attention_heads,
            backend=backend,
        )
        self.atom_encoder = AtomToTokenEncoder(
            atom_hidden_size,
            single_size,
            num_atom_encoder_blocks,
            num_atom_encoder_attention_heads,
            backend,
            block=ConditionedTransformerEncoderBlock,
        )

        self.residue_type_projection = nn.Linear(vocab_size, single_size, bias=False)
        self.msa_profile_projection = nn.Linear(len(TOKENS) + 1, single_size, bias=False)
        self.method_embedding: nn.Embedding | None = None
        self.modified_embedding: nn.Embedding | None = None
        self.cyclic_projection: nn.Linear | None = None
        self.molecule_type_embedding: nn.Embedding | None = None
        self.binding_specification_embedding: nn.Embedding | None = None
        self.design_mask_embedding: nn.Embedding | None = None
        self.ss_specification_embedding: nn.Embedding | None = None
        if add_method_conditioning:
            self.method_embedding = nn.Embedding(num_method_types, single_size)
        if add_modified_flag:
            self.modified_embedding = nn.Embedding(2, single_size)
        if add_cyclic_flag:
            self.cyclic_projection = nn.Linear(1, single_size, bias=False)
        if add_molecule_type_feature:
            self.molecule_type_embedding = nn.Embedding(num_molecule_types, single_size)
        if add_binding_specification:
            self.binding_specification_embedding = nn.Embedding(num_binding_types, single_size)
        if add_design_mask:
            self.design_mask_embedding = nn.Embedding(2, single_size)
        if add_ss_specification:
            self.ss_specification_embedding = nn.Embedding(num_ss_types, single_size)
        for module in (
            self.method_embedding,
            self.modified_embedding,
            self.molecule_type_embedding,
            self.binding_specification_embedding,
            self.design_mask_embedding,
            self.ss_specification_embedding,
        ):
            if module is not None:
                nn.init.zeros_(module.weight)
        if self.cyclic_projection is not None:
            nn.init.zeros_(self.cyclic_projection.weight)

    def forward(
        self,
        *,
        atom_features: Tensor,
        reference_positions: Tensor,
        atom_mask: Tensor,
        reference_space_id: Tensor,
        atom_to_token: Tensor,
        residue_type_features: Tensor,
        msa_profile: Tensor,
        deletion_mean: Tensor,
        method_feature: Tensor | None = None,
        modified: Tensor | None = None,
        cyclic_period: Tensor | None = None,
        molecule_type: Tensor | None = None,
        binding_type: Tensor | None = None,
        design_mask: Tensor | None = None,
        ss_type: Tensor | None = None,
    ) -> Tensor:
        reference_positions = project_model_coordinates(
            reference_positions,
            atom_mask.bool(),
            dtype=self.atom_feature_projection.weight.dtype,
            feature="BoltzGen reference positions",
        )
        atom_features = project_model_features(
            atom_features,
            atom_mask.bool()[..., None],
            dtype=self.atom_feature_projection.weight.dtype,
            feature="BoltzGen atom features",
        )
        token_rows = torch.ones_like(deletion_mean, dtype=torch.bool)
        residue_type_features = project_model_features(
            residue_type_features,
            token_rows[..., None],
            dtype=self.residue_type_projection.weight.dtype,
            feature="BoltzGen token classes",
        )
        msa_profile = project_model_features(
            msa_profile,
            token_rows[..., None],
            dtype=self.msa_profile_projection.weight.dtype,
            feature="BoltzGen MSA profile",
        )
        deletion_mean = project_model_features(
            deletion_mean, token_rows, dtype=self.msa_profile_projection.weight.dtype, feature="BoltzGen deletion mean"
        )
        if cyclic_period is not None:
            cyclic_period = project_model_features(cyclic_period, token_rows, feature="BoltzGen cyclic period")
        geometry = self.windowing(reference_positions, reference_space_id, atom_mask)
        atom_single = self.atom_feature_projection(atom_features)
        token_from_atom = self.atom_encoder(
            atom_single,
            self.atom_pair_bias_projection(self.atom_pair_encoder(atom_single, geometry)),
            atom_mask,
            atom_to_token,
            residue_type_features,
            geometry.window_info,
        )

        single = token_from_atom
        single = single + self.residue_type_projection(residue_type_features.to(single.dtype))
        msa_features = torch.cat([msa_profile, deletion_mean.unsqueeze(-1)], dim=-1)
        single = single + self.msa_profile_projection(msa_features.to(single.dtype))

        if self.method_embedding is not None:
            single = single + self.method_embedding(cast(Tensor, method_feature))
        if self.modified_embedding is not None:
            single = single + self.modified_embedding(cast(Tensor, modified))
        if self.cyclic_projection is not None:
            single = single + self.cyclic_projection(
                cast(Tensor, cyclic_period).clamp(max=1.0).unsqueeze(-1).to(single.dtype)
            )
        if self.molecule_type_embedding is not None:
            single = single + self.molecule_type_embedding(cast(Tensor, molecule_type))
        if self.binding_specification_embedding is not None:
            single = single + self.binding_specification_embedding(cast(Tensor, binding_type))
        if self.design_mask_embedding is not None:
            single = single + self.design_mask_embedding(cast(Tensor, design_mask).int())
        if self.ss_specification_embedding is not None:
            single = single + self.ss_specification_embedding(cast(Tensor, ss_type))
        return single


class BoltzGenPairUpdateStack(nn.Module):
    r"""
    Apply a homogeneous sequence of BoltzGen pair-update blocks.

    Template and token-distance embedders differ in feature construction, but
    both repeat the same pair-update equation. The stack owns that repetition
    and its gradient-checkpointing policy.
    """

    _gradient_checkpointing_func: Callable[..., Tensor]

    def __init__(self, blocks: Iterable[nn.Module]) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.num_blocks = len(self.blocks)
        self.gradient_checkpointing = False

    def forward(self, pair: Tensor, pair_mask: Tensor | None = None) -> Tensor:
        for block in self.blocks:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                pair = self._gradient_checkpointing_func(
                    block.__call__,
                    pair,
                    pair_mask=pair_mask,
                )
            else:
                pair = block(pair, pair_mask=pair_mask)
        return pair


def _triangle_pair_stack(
    *,
    pair_size: int,
    num_blocks: int,
    dropout: float,
    backend: str,
    triangle_attention_head_size: int = 32,
    num_triangle_attention_heads: int = 4,
) -> BoltzGenPairUpdateStack:
    return BoltzGenPairUpdateStack(
        LineDropoutTrianglePairUpdateBlock(
            pair_size=pair_size,
            triangle_multiplication_hidden_size=pair_size,
            triangle_attention_head_size=triangle_attention_head_size,
            num_triangle_attention_heads=num_triangle_attention_heads,
            dropout=dropout,
            backend=backend,
        )
        for _ in range(num_blocks)
    )


def _mini_pair_stack(
    *,
    pair_size: int,
    num_blocks: int,
    dropout: float,
    backend: str,
    post_layer_norm: bool = False,
) -> BoltzGenPairUpdateStack:
    return BoltzGenPairUpdateStack(
        BoltzGenMiniTrianglePairUpdateBlock(
            pair_size=pair_size,
            dropout=dropout,
            post_layer_norm=post_layer_norm,
            backend=backend,
        )
        for _ in range(num_blocks)
    )


class BoltzGenTemplateEmbedder(nn.Module):
    r"""
    Embed template geometry as an additive pair update.

    Directions are normalized over their three Cartesian components with a
    ``1e-6`` norm floor. Coincident positions contribute a zero direction.
    """

    def __init__(
        self,
        pair_size: int = 128,
        template_size: int = 64,
        num_blocks: int = 2,
        vocab_size: int = len(TOKENS),
        num_bins: int = 38,
        min_dist: float = 3.25,
        max_dist: float = 50.75,
        dropout: float = 0.25,
        backend: str = "auto",
        *,
        pair_stack: Callable[..., nn.Module] = _triangle_pair_stack,
    ) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.num_bins = num_bins
        self.template_feature_projection = FactorizedPairLinear(
            pair_size=num_bins + 5,
            token_size=vocab_size,
            output_size=template_size,
            bias=False,
            backend=backend,
        )
        self.pair_norm = LayerNorm(pair_size, backend=backend)
        self.pair_projection = nn.Linear(pair_size, template_size, bias=False)
        self.pair_stack = pair_stack(
            pair_size=template_size,
            num_blocks=num_blocks,
            dropout=dropout,
            backend=backend,
        )
        self.hidden_norm = LayerNorm(template_size, backend=backend)
        self.relu = nn.ReLU()
        self.output_projection = nn.Linear(template_size, pair_size, bias=False)

    def forward(self, pair: Tensor, input: BoltzGenInput, pair_mask: Tensor | None = None) -> Tensor:
        input = cast(BoltzGenInput, normalize_boltzgen_input(input, dtype=pair.dtype))
        residue_type_features = cast(Tensor, input.template_residue_type_features)
        frame_mask = cast(Tensor, input.template_frame_mask)
        frame_rotations = cast(Tensor, input.template_frame_rotations)
        frame_translations = cast(Tensor, input.template_frame_translations)
        pseudo_beta_positions = cast(Tensor, input.template_pseudo_beta_positions)
        alpha_carbon_positions = cast(Tensor, input.template_alpha_carbon_positions)
        pseudo_beta_mask = cast(Tensor, input.template_pseudo_beta_mask)
        template_mask = cast(Tensor, input.template_mask)
        visibility_index = cast(Tensor, input.template_visibility_index)
        present = template_mask.any(dim=-1)
        count = present.sum(dim=1).clamp(min=1)

        pair_pseudo_beta_mask = (pseudo_beta_mask[..., :, None] * pseudo_beta_mask[..., None, :]).unsqueeze(-1)
        pair_frame_mask = (frame_mask[..., :, None] * frame_mask[..., None, :]).unsqueeze(-1)
        same_visibility = (visibility_index[..., :, None] == visibility_index[..., None, :]) & (
            visibility_index[..., :, None] >= 0
        )

        beta_pairs = same_visibility & pseudo_beta_mask.bool()[..., :, None] & pseudo_beta_mask.bool()[..., None, :]
        pseudo_beta_displacement = pseudo_beta_positions[..., :, None, :] - pseudo_beta_positions[..., None, :, :]
        pseudo_beta_displacement = pseudo_beta_displacement.masked_fill(~beta_pairs[..., None], 0)
        pseudo_beta_distances = torch.linalg.vector_norm(pseudo_beta_displacement, dim=-1)
        boundaries = torch.linspace(
            self.min_dist,
            self.max_dist,
            self.num_bins - 1,
            device=pseudo_beta_distances.device,
            dtype=pseudo_beta_distances.dtype,
        )
        distogram = F.one_hot(
            (pseudo_beta_distances[..., None] > boundaries).sum(dim=-1).long(),
            num_classes=self.num_bins,
        )

        rotation = frame_rotations.unsqueeze(2).transpose(-1, -2)
        translation = frame_translations.unsqueeze(2).unsqueeze(-1)
        targets = alpha_carbon_positions.unsqueeze(3).unsqueeze(-1)
        frame_pairs = (
            same_visibility
            & frame_mask.bool()[..., None, :]
            & cast(Tensor, input.template_center_mask).bool()[..., :, None]
        )
        displacement = (targets - translation).masked_fill(~frame_pairs[..., None, None], 0)
        vector = (rotation * displacement.transpose(-1, -2)).sum(dim=-1)
        vector = vector.masked_fill(~frame_pairs[..., None], 0)
        unit_vector = F.normalize(vector, dim=-1, eps=1e-6)

        geometry = torch.cat(
            [
                distogram.masked_fill(~beta_pairs[..., None], 0).to(pair.dtype),
                pair_pseudo_beta_mask.masked_fill(~same_visibility[..., None], 0),
                unit_vector,
                pair_frame_mask.masked_fill(~same_visibility[..., None], 0),
            ],
            dim=-1,
        )
        geometry = geometry * same_visibility.unsqueeze(-1)
        template_residue_features = residue_type_features.to(pair.dtype)
        feature_value = self.template_feature_projection(geometry.to(pair.dtype), template_residue_features)
        value = self.pair_projection(self.pair_norm(pair)).unsqueeze(-4) + feature_value
        template_pair_mask = None if pair_mask is None else pair_mask.unsqueeze(-3)
        refined = self.pair_stack(value, pair_mask=template_pair_mask)
        value = self.hidden_norm(value + refined)

        aggregated = (value * present[..., None, None, None]).sum(dim=-4)
        aggregated = aggregated / count[..., None, None, None].to(aggregated.dtype)
        return self.output_projection(self.relu(aggregated))


class BoltzGenTokenDistanceEmbedder(nn.Module):
    r"""
    Encode inter-token backbone geometry into an additive pair update.
    """

    def __init__(
        self,
        pair_size: int = 128,
        token_distance_size: int = 64,
        num_blocks: int = 2,
        num_bins: int = 38,
        min_dist: float = 3.25,
        max_dist: float = 50.75,
        num_distance_gaussians: int = 32,
        dropout: float = 0.25,
        use_token_distance_features: bool = True,
        backend: str = "auto",
        *,
        pair_stack: Callable[..., nn.Module] = _triangle_pair_stack,
    ) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.num_bins = num_bins
        self.use_token_distance_features = use_token_distance_features
        self.distance_encoder = (
            BoltzGenDistanceTokenEncoder(
                num_distance_gaussians=num_distance_gaussians,
                pair_size=pair_size,
                output_size=pair_size,
                backend=backend,
            )
            if use_token_distance_features
            else None
        )
        self.pair_norm = LayerNorm(pair_size, backend=backend)
        self.pair_projection = nn.Linear(pair_size, token_distance_size, bias=False)
        self.distance_feature_projection = nn.Linear(
            num_bins + (4 * pair_size if use_token_distance_features else 0),
            token_distance_size,
            bias=False,
        )
        self.pair_stack = pair_stack(
            pair_size=token_distance_size,
            num_blocks=num_blocks,
            dropout=dropout,
            backend=backend,
        )
        self.hidden_norm = LayerNorm(token_distance_size, backend=backend)
        self.relu = nn.ReLU()
        self.output_projection = nn.Linear(token_distance_size, pair_size, bias=False)

    def forward(
        self,
        pair: Tensor,
        input: BoltzGenInput,
        relative_position_encoding: Tensor,
        pair_mask: Tensor | None = None,
    ) -> Tensor:
        input = normalize_boltzgen_input(input)
        atom_coordinate_mask = input.atom_coordinate_mask.bool() & input.atom_mask.bool()
        center_mask = atom_coordinate_mask.gather(1, input.token_center_atom_index.long()) & input.token_mask.bool()
        center_pair_mask = center_mask.unsqueeze(-1) & center_mask.unsqueeze(-2) & input.token_distance_mask.bool()
        coords = input.center_coordinates.masked_fill(~center_mask.unsqueeze(-1), 0)
        distances = pairwise_distance(coords)
        boundaries = torch.linspace(self.min_dist, self.max_dist, self.num_bins - 1, device=distances.device)
        distogram = F.one_hot((distances[..., None] > boundaries).sum(dim=-1).long(), num_classes=self.num_bins)
        distogram = distogram.masked_fill(~center_pair_mask.unsqueeze(-1), 0)

        distance_features = (
            torch.cat(
                [
                    distogram,
                    cast(BoltzGenDistanceTokenEncoder, self.distance_encoder)(
                        relative_position_encoding,
                        input,
                    ),
                ],
                dim=-1,
            )
            if self.use_token_distance_features
            else distogram
        )
        value = self.pair_projection(self.pair_norm(pair)) + self.distance_feature_projection(
            distance_features.to(pair.dtype)
        )
        refined = self.pair_stack(value, pair_mask=pair_mask)
        value = self.hidden_norm(value + refined)  # one outer residual around the (internally-residual) stack
        return self.output_projection(self.relu(value))


class BoltzGenDistanceTokenEncoder(nn.Module):
    r"""
    Per-backbone-atom pairwise distance features conditioned on the relative-position encoding.
    """

    def __init__(
        self, num_distance_gaussians: int = 32, pair_size: int = 128, output_size: int = 128, backend: str = "auto"
    ) -> None:
        super().__init__()
        self.distance_smearing = GaussianRadialBasis(
            num_bins=num_distance_gaussians,
            lower_bound=0.0,
            upper_bound=2.0,
            width=math.sqrt(2) * 2.0 / (num_distance_gaussians - 1),
        )
        input_size = num_distance_gaussians + 1 + pair_size
        self.distance_norm = LayerNorm(input_size, backend=backend)
        self.distance_feed_forward = SwiGluFeedForward(
            input_size,
            pair_size,
            output_size,
        )

    def forward(self, relative_position_encoding: Tensor, input: BoltzGenInput) -> Tensor:
        input = cast(BoltzGenInput, normalize_boltzgen_input(input))
        batch_size, num_tokens = relative_position_encoding.shape[:2]
        backbone, backbone_mask = _gather_backbone4(
            input.token_to_bb4_atoms,
            input.atom_coordinates,
            input.atom_coordinate_mask.bool() & input.atom_mask.bool(),
            batch_size,
            num_tokens,
        )
        backbone = backbone.permute(0, 2, 1, 3)
        backbone_mask = (backbone_mask & input.token_mask.unsqueeze(-1).bool()).permute(0, 2, 1)
        distance_mask = (
            backbone_mask.unsqueeze(-1) & backbone_mask.unsqueeze(-2) & input.token_distance_mask[:, None].bool()
        )
        displacement = backbone.unsqueeze(-2) - backbone.unsqueeze(-3)
        displacement = displacement.masked_fill(~distance_mask.unsqueeze(-1), 0)
        distance = displacement.norm(dim=-1).unsqueeze(-1)  # (B, 4, N, N, 1)
        gaussian = self.distance_smearing(distance.squeeze(-1))  # (B, 4, N, N, gaussians)
        relative = relative_position_encoding.view(batch_size, 1, num_tokens, num_tokens, -1).expand(-1, 4, -1, -1, -1)
        bias_input = torch.cat([gaussian, distance, relative], dim=-1)
        bias_input = bias_input.masked_fill(~distance_mask.unsqueeze(-1), 0)
        bias = self.distance_feed_forward(self.distance_norm(bias_input))
        bias = bias.masked_fill(~distance_mask.unsqueeze(-1), 0)
        bias = bias.permute(0, 2, 3, 4, 1).reshape(batch_size, num_tokens, num_tokens, -1)
        return bias


class _BoltzGenMsaTrunk(HomogeneousBlockRegions, nn.Module):
    r"""Common feature construction and block loop for BoltzGen MSA trunks."""

    def __init__(
        self,
        *,
        single_size: int,
        msa_size: int,
        use_paired_feature: bool,
        blocks: Iterable[nn.Module],
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.backend = resolve_config_backend(backend, supported=("oops", "eager"), public_default="eager")
        self.use_paired_feature = use_paired_feature
        self.msa_projection = nn.Linear(
            len(TOKENS) + 2 + int(use_paired_feature),
            msa_size,
            bias=False,
        )
        self.token_projection = nn.Linear(single_size, msa_size, bias=False)
        self.blocks = nn.ModuleList(blocks)
        self.num_blocks = len(self.blocks)
        self.gradient_checkpointing = False

    _gradient_checkpointing_func: Callable[..., tuple[Tensor, Tensor]]

    def forward(
        self,
        pair: Tensor,
        single_inputs: Tensor,
        *,
        msa_tokens: Tensor,
        has_deletion: Tensor,
        deletion_value: Tensor,
        is_paired: Tensor,
        msa_mask: Tensor,
        token_mask: Tensor,
    ) -> Tensor:
        msa_mask = msa_mask.bool() & token_mask.bool().unsqueeze(-2)
        msa_tokens = project_model_categories(
            msa_tokens, msa_mask, num_classes=len(TOKENS), feature="BoltzGen MSA tokens"
        )
        has_deletion = project_model_features(
            torch.where(msa_mask, has_deletion, 0).to(torch.float64),
            msa_mask,
            dtype=pair.dtype,
            feature="BoltzGen deletion presence",
        )
        deletion_value = project_model_features(
            deletion_value, msa_mask, dtype=pair.dtype, feature="BoltzGen deletion feature"
        )
        if self.use_paired_feature:
            is_paired = project_model_features(
                torch.where(msa_mask, is_paired, 0).to(torch.float64),
                msa_mask,
                dtype=pair.dtype,
                feature="BoltzGen MSA pairing",
            )
        features = [
            F.one_hot(msa_tokens, num_classes=len(TOKENS)).to(pair.dtype),
            has_deletion.unsqueeze(-1).to(pair.dtype),
            deletion_value.unsqueeze(-1).to(pair.dtype),
        ]
        if self.use_paired_feature:
            features.append(is_paired.unsqueeze(-1).to(pair.dtype))
        msa = self.msa_projection(torch.cat(features, dim=-1))
        msa = msa + self.token_projection(single_inputs).unsqueeze(1)
        pair_mask = pair_mask_from_token_mask(token_mask, backend=self.backend)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                msa, pair = self._gradient_checkpointing_func(
                    block.__call__,
                    msa,
                    pair,
                    msa_mask=msa_mask,
                    pair_mask=pair_mask,
                )
            else:
                msa, pair = block(msa, pair, msa_mask=msa_mask, pair_mask=pair_mask)
        return pair


class BoltzGenMsaTrunk(_BoltzGenMsaTrunk):
    r"""
    Build BoltzGen MSA features and refine the pair representation.
    """

    def __init__(
        self,
        single_size: int = 384,
        pair_size: int = 128,
        msa_size: int = 64,
        num_blocks: int = 4,
        use_paired_feature: bool = True,
        outer_product_hidden_size: int = 32,
        msa_dropout: float = 0.15,
        pair_dropout: float = 0.25,
        num_msa_attention_heads: int = 8,
        msa_attention_head_size: int = 32,
        triangle_attention_head_size: int = 32,
        num_triangle_attention_heads: int = 4,
        expansion: int = 4,
        backend: str = "auto",
    ) -> None:
        super().__init__(
            single_size=single_size,
            msa_size=msa_size,
            use_paired_feature=use_paired_feature,
            backend=backend,
            blocks=(
                MsaPairBlock(
                    msa_size=msa_size,
                    pair_size=pair_size,
                    outer_product_hidden_size=outer_product_hidden_size,
                    msa_dropout=msa_dropout,
                    pair_dropout=pair_dropout,
                    num_msa_attention_heads=num_msa_attention_heads,
                    msa_attention_head_size=msa_attention_head_size,
                    expansion=expansion,
                    backend=backend,
                    pair_block=partial(
                        LineDropoutTrianglePairUpdateBlock,
                        triangle_attention_head_size=triangle_attention_head_size,
                        num_triangle_attention_heads=num_triangle_attention_heads,
                    ),
                )
                for _ in range(num_blocks)
            ),
        )


class BoltzGenMiniformerMsaTrunk(_BoltzGenMsaTrunk):
    r"""
    MSA-module variant whose pair track uses compact pair-only Miniformer blocks.

    Uses the common BoltzGen MSA feature construction with Miniformer blocks.
    """

    def __init__(
        self,
        single_size: int = 384,
        pair_size: int = 128,
        msa_size: int = 64,
        num_blocks: int = 3,
        use_paired_feature: bool = True,
        outer_product_hidden_size: int = 32,
        msa_dropout: float = 0.15,
        pair_dropout: float = 0.25,
        num_msa_attention_heads: int = 8,
        msa_attention_head_size: int = 32,
        expansion: int = 4,
        backend: str = "auto",
    ) -> None:
        super().__init__(
            single_size=single_size,
            msa_size=msa_size,
            use_paired_feature=use_paired_feature,
            backend=backend,
            blocks=(
                MsaPairBlock(
                    msa_size=msa_size,
                    pair_size=pair_size,
                    outer_product_hidden_size=outer_product_hidden_size,
                    msa_dropout=msa_dropout,
                    pair_dropout=pair_dropout,
                    num_msa_attention_heads=num_msa_attention_heads,
                    msa_attention_head_size=msa_attention_head_size,
                    expansion=expansion,
                    backend=backend,
                    pair_block=BoltzGenMiniTrianglePairUpdateBlock,
                )
                for _ in range(num_blocks)
            ),
        )


class BoltzGenMiniformerStack(nn.Module):
    r"""
    Stack of blocks in the concrete reduced BoltzGen Miniformer architecture.
    """

    def __init__(
        self,
        num_blocks: int = 12,
        single_size: int = 384,
        pair_size: int = 128,
        num_pair_attention_heads: int = 16,
        dropout: float = 0.25,
        post_layer_norm: bool = False,
        use_single_to_pair: bool = False,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.gradient_checkpointing = False
        self.blocks = nn.ModuleList(
            [
                BoltzGenMiniformerBlock(
                    single_size=single_size,
                    pair_size=pair_size,
                    num_pair_attention_heads=num_pair_attention_heads,
                    dropout=dropout,
                    post_layer_norm=post_layer_norm,
                    use_single_to_pair=use_single_to_pair,
                    backend=backend,
                )
                for _ in range(num_blocks)
            ]
        )

    _gradient_checkpointing_func: Callable[..., tuple[Tensor, Tensor]]

    def forward(self, single: Tensor, pair: Tensor, pair_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        for block in self.blocks:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                single, pair = self._gradient_checkpointing_func(
                    block.__call__,
                    single,
                    pair,
                    pair_mask=pair_mask,
                )
            else:
                single, pair = block(single, pair, pair_mask=pair_mask)
        return single, pair


class BoltzGenMiniformerBlock(nn.Module):
    r"""
    BoltzGen Miniformer block: compact pair update plus pair-biased single attention.
    """

    def __init__(
        self,
        single_size: int = 384,
        pair_size: int = 128,
        num_pair_attention_heads: int = 16,
        expansion: int = 4,
        dropout: float = 0.25,
        post_layer_norm: bool = False,
        use_single_to_pair: bool = False,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.pair_left_projection = nn.Linear(single_size, pair_size, bias=False) if use_single_to_pair else None
        self.pair_right_projection = nn.Linear(single_size, pair_size, bias=False) if use_single_to_pair else None
        self.triangle_multiplication = BoltzGenBidirectionalTriangleMultiplication(pair_size, backend=backend)
        self.rowwise_dropout = StructuredDropout(dropout, broadcast_dims=(-2, -1))
        self.pair_transition = Transition(pair_size, expansion=expansion, backend=backend)
        self.attention = PairBiasedSelfAttention(
            hidden_size=single_size,
            pair_size=pair_size,
            num_attention_heads=num_pair_attention_heads,
            backend=backend,
        )
        self.single_transition = Transition(single_size, expansion=expansion, backend=backend)
        self.single_post_norm = LayerNorm(single_size, backend=backend) if post_layer_norm else nn.Identity()

    def forward(self, single: Tensor, pair: Tensor, pair_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if pair_mask is None:
            pair_mask = pair.new_ones(pair.shape[:-1])
        left_projection = self.pair_left_projection
        right_projection = self.pair_right_projection
        if left_projection is not None and right_projection is not None:
            pair = pair + left_projection(single).unsqueeze(-2) + right_projection(single).unsqueeze(-3)
        pair = pair + self.rowwise_dropout(self.triangle_multiplication(pair, pair_mask=pair_mask))
        pair = pair + self.pair_transition(pair)
        single = single + self.attention(single, pair, pair_mask=pair_mask)
        single = single + self.single_transition(single)
        return self.single_post_norm(single), pair


class BoltzGenMiniTrianglePairUpdateBlock(nn.Module):
    r"""
    Pair-only Miniformer block used in structure-trained-small MSA / template stacks.
    """

    def __init__(
        self,
        pair_size: int = 128,
        expansion: int = 4,
        dropout: float = 0.25,
        post_layer_norm: bool = False,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.triangle_multiplication = BoltzGenBidirectionalTriangleMultiplication(pair_size, backend=backend)
        self.rowwise_dropout = StructuredDropout(dropout, broadcast_dims=(-2, -1))
        self.pair_transition = Transition(pair_size, expansion=expansion, backend=backend)
        self.pair_post_norm = LayerNorm(pair_size, backend=backend) if post_layer_norm else nn.Identity()

    def forward(self, pair: Tensor, pair_mask: Tensor | None = None) -> Tensor:
        if pair_mask is None:
            pair_mask = pair.new_ones(pair.shape[:-1])
        pair = pair + self.rowwise_dropout(self.triangle_multiplication(pair, pair_mask=pair_mask))
        pair = pair + self.pair_transition(pair)
        return self.pair_post_norm(pair)


class BoltzGenBidirectionalTriangleMultiplication(nn.Module):
    r"""
    Compact bidirectional triangular update used by BoltzGen Miniformer artifacts.
    """

    def __init__(self, pair_size: int = 128, backend: str = "auto") -> None:
        super().__init__()
        self.input_norm = LayerNorm(pair_size, eps=1e-5, backend=backend)
        self.input_projection = nn.Linear(pair_size, pair_size, bias=False)
        self.input_gate_projection = nn.Linear(pair_size, pair_size, bias=False)
        self.output_norm = LayerNorm(pair_size // 2, backend=backend)
        self.output_projection = nn.Linear(pair_size // 2, pair_size, bias=False)
        self.output_gate_projection = nn.Linear(pair_size // 2, pair_size, bias=False)

    def forward(self, pair: Tensor, pair_mask: Tensor | None = None) -> Tensor:
        normalized = self.input_norm(pair)
        update = self.input_projection(normalized) * torch.sigmoid(self.input_gate_projection(normalized))
        if pair_mask is not None:
            update = update * pair_mask.unsqueeze(-1).to(dtype=update.dtype, device=update.device)
        left_out, right_out, left_in, right_in = torch.chunk(update, 4, dim=-1)
        outgoing = torch.einsum("...ikd,...jkd->...ijd", left_out, right_out)
        incoming = torch.einsum("...kid,...kjd->...ijd", left_in, right_in)
        update = torch.cat((outgoing, incoming), dim=-1)
        update = self.output_norm(update)
        return self.output_projection(update) * torch.sigmoid(self.output_gate_projection(update))


class BoltzGenGaussianRandom3DEncoding(nn.Module):
    r"""
    Frozen Gaussian features around randomly initialized three-dimensional centers.
    """

    center: Tensor
    variance: Tensor

    def __init__(self, num_gaussians: int) -> None:
        super().__init__()
        self.num_gaussians = num_gaussians
        self.register_buffer("center", torch.randn(num_gaussians, 3))
        variance = torch.rand(num_gaussians)
        self.register_buffer("variance", variance.clamp_min(torch.finfo(variance.dtype).tiny))

    def forward(self, coordinates: Tensor) -> Tensor:
        squared_distance = (
            (coordinates.unsqueeze(-2) - self.center.view(*(1,) * (coordinates.dim() - 1), self.num_gaussians, 3))
            .square()
            .sum(dim=-1)
        )
        return torch.exp(
            -squared_distance / (2 * self.variance.view(*(1,) * (coordinates.dim() - 1), self.num_gaussians))
        )


class BoltzGenGaussianCoordinateAtomToTokenEncoder(CoordinateAtomToTokenEncoder):
    r"""
    Coordinate atom encoder whose noisy positions also enter through frozen random Gaussian features.

    Args:
        num_coordinate_gaussians: Number of frozen random Gaussian coordinate features.
        atom_hidden_size: Width of the atom query and conditioning states.
        token_hidden_size: Width of the produced token representation.
        num_blocks: Number of conditioned Transformer-encoder blocks.
        num_attention_heads: Number of attention heads per block.
        backend: Backend selector forwarded to the transformer and the pooling.
        block: Conditioned Transformer-encoder block class of every block.
    """

    def __init__(
        self,
        num_coordinate_gaussians: int,
        atom_hidden_size: int = 128,
        token_hidden_size: int = 768,
        num_blocks: int = 3,
        num_attention_heads: int = 4,
        backend: str = "auto",
        *,
        block: type[ConditionedTransformerEncoderBlock],
    ) -> None:
        super().__init__(
            atom_hidden_size,
            token_hidden_size,
            num_blocks,
            num_attention_heads,
            backend,
            block=block,
        )
        self.position_encoding = BoltzGenGaussianRandom3DEncoding(num_coordinate_gaussians)
        self.noisy_position_projection = nn.Linear(3 + num_coordinate_gaussians, atom_hidden_size, bias=False)

    def embed_noisy_positions(self, noisy_positions: Tensor) -> Tensor:
        r"""
        Embed noisy atom coordinates together with their frozen Gaussian features.
        """

        position_features = torch.cat((noisy_positions, self.position_encoding(noisy_positions)), dim=-1)
        return self.noisy_position_projection(position_features)


class BoltzGenResidueTypeTokenToAtomDecoder(TokenToAtomDecoder):
    r"""
    Decode atom coordinate updates and summed per-token residue-type logits.

    Args:
        num_residue_types: Number of residue types scored per token.
        token_hidden_size: Width of the refined token representation.
        atom_hidden_size: Width of the atom query and conditioning states.
        num_blocks: Number of conditioned Transformer-encoder blocks.
        num_attention_heads: Number of attention heads per block.
        backend: Backend selector forwarded to the transformer, the broadcast, and the normalization.
        block: Conditioned Transformer-encoder block class of every block.
    """

    def __init__(
        self,
        num_residue_types: int,
        token_hidden_size: int = 768,
        atom_hidden_size: int = 128,
        num_blocks: int = 3,
        num_attention_heads: int = 4,
        backend: str = "auto",
        *,
        block: type[ConditionedTransformerEncoderBlock],
    ) -> None:
        super().__init__(
            token_hidden_size,
            atom_hidden_size,
            num_blocks,
            num_attention_heads,
            backend,
            block=block,
        )
        self.residue_type_projection = nn.Linear(atom_hidden_size, num_residue_types)

    def forward_with_residue_types(
        self,
        token: Tensor,
        atom_query: Tensor,
        atom_conditioning: Tensor,
        atom_pair_bias: Tensor,
        atom_mask: Tensor,
        atom_to_token: Tensor,
        window_info: LocalAtomWindowInfo,
    ) -> tuple[Tensor, Tensor]:
        atom_query = self.decode_atoms(
            token,
            atom_query,
            atom_conditioning,
            atom_pair_bias,
            atom_mask,
            atom_to_token,
            window_info,
        )
        token_sum = torch.zeros(
            *atom_query.shape[:-2],
            token.shape[-2],
            atom_query.shape[-1],
            device=atom_query.device,
            dtype=atom_query.dtype,
        )
        token_sum.scatter_add_(
            -2,
            atom_to_token.unsqueeze(-2).unsqueeze(-1).expand_as(atom_query),
            atom_query * atom_mask.unsqueeze(-2).unsqueeze(-1).to(atom_query.dtype),
        )
        return (
            self.position_update_projection(self.position_norm(atom_query)),
            self.residue_type_projection(token_sum),
        )


class _BoltzGenDenoiser(nn.Module):
    r"""
    Encode noisy atom and token features shared by the concrete BoltzGen denoisers.
    """

    def __init__(
        self,
        single_size: int = 384,
        atom_hidden_size: int = 128,
        fourier_embedding_size: int = 256,
        num_atom_encoder_blocks: int = 3,
        num_atom_encoder_attention_heads: int = 4,
        num_token_encoder_blocks: int = 24,
        num_token_encoder_attention_heads: int = 16,
        num_atom_decoder_blocks: int = 3,
        num_atom_decoder_attention_heads: int = 4,
        num_conditioning_transitions: int = 2,
        transformer_size: int | None = None,
        post_layer_norm: bool = False,
        use_qk_norm: bool = False,
        num_coordinate_gaussians: int = 0,
        num_residue_types: int = _NUM_RESIDUE_TYPES,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        transformer_size = 2 * single_size if transformer_size is None else transformer_size
        # Query/key normalization reaches every denoiser Transformer; post-normalization only the atom encoder's.
        block: type[ConditionedTransformerEncoderBlock] = (
            QueryKeyNormConditionedTransformerEncoderBlock if use_qk_norm else ConditionedTransformerEncoderBlock
        )
        atom_encoder_block = block
        if post_layer_norm:
            atom_encoder_block = (
                PostNormQueryKeyNormConditionedTransformerEncoderBlock
                if use_qk_norm
                else PostNormConditionedTransformerEncoderBlock
            )
        self.single_conditioner = SingleConditioning(
            2 * single_size,
            transformer_size,
            fourier_embedding_size=fourier_embedding_size,
            num_transitions=num_conditioning_transitions,
            backend=backend,
        )
        self.atom_encoder = (
            BoltzGenGaussianCoordinateAtomToTokenEncoder(
                num_coordinate_gaussians,
                atom_hidden_size,
                transformer_size,
                num_atom_encoder_blocks,
                num_atom_encoder_attention_heads,
                backend,
                block=atom_encoder_block,
            )
            if num_coordinate_gaussians > 0
            else CoordinateAtomToTokenEncoder(
                atom_hidden_size,
                transformer_size,
                num_atom_encoder_blocks,
                num_atom_encoder_attention_heads,
                backend,
                block=atom_encoder_block,
            )
        )
        self.single_to_token_norm = LayerNorm(transformer_size, backend=backend)
        self.single_to_token_projection = nn.Linear(transformer_size, transformer_size, bias=False)
        self.token_encoder = ConditionedTransformerEncoder(
            num_blocks=num_token_encoder_blocks,
            num_attention_heads=num_token_encoder_attention_heads,
            hidden_size=transformer_size,
            conditioning_size=transformer_size,
            backend=backend,
            block=block,
        )
        self.token_norm = LayerNorm(transformer_size, backend=backend)
        self.atom_decoder = self._build_decoder(
            num_residue_types=num_residue_types,
            token_hidden_size=transformer_size,
            atom_hidden_size=atom_hidden_size,
            num_blocks=num_atom_decoder_blocks,
            num_attention_heads=num_atom_decoder_attention_heads,
            backend=backend,
            block=block,
        )

    def _build_decoder(
        self,
        *,
        num_residue_types: int,
        token_hidden_size: int,
        atom_hidden_size: int,
        num_blocks: int,
        num_attention_heads: int,
        backend: str,
        block: type[ConditionedTransformerEncoderBlock],
    ) -> TokenToAtomDecoder:
        raise NotImplementedError

    def _encode(
        self,
        single_inputs: Tensor,
        single_trunk: Tensor,
        noisy_positions: Tensor,
        noise_level: Tensor,
        conditioning: AtomDenoiserContext,
        *,
        atom_mask: Tensor,
        atom_to_token: Tensor,
        token_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        single = self.single_conditioner(noise_level, single_trunk, single_inputs)
        token_repr, atom_query = self.atom_encoder(
            noisy_positions,
            conditioning.atom_query,
            conditioning.atom_conditioning,
            conditioning.atom_encoder_bias,
            atom_mask,
            atom_to_token,
            single_trunk,
            conditioning.window_info,
        )
        token_repr = token_repr + self.single_to_token_projection(self.single_to_token_norm(single))
        token_repr = self.token_encoder(
            token_repr,
            single,
            conditioning.token_encoder_bias,
            token_mask,
        )
        return self.token_norm(token_repr), atom_query


class BoltzGenDenoiser(_BoltzGenDenoiser):
    r"""
    Predict per-atom position updates for EDM sampling.
    """

    def _build_decoder(
        self,
        *,
        num_residue_types: int,
        token_hidden_size: int,
        atom_hidden_size: int,
        num_blocks: int,
        num_attention_heads: int,
        backend: str,
        block: type[ConditionedTransformerEncoderBlock],
    ) -> TokenToAtomDecoder:
        del num_residue_types
        return TokenToAtomDecoder(
            token_hidden_size,
            atom_hidden_size,
            num_blocks,
            num_attention_heads,
            backend,
            block=block,
        )

    def forward(
        self,
        single_inputs: Tensor,
        single_trunk: Tensor,
        noisy_positions: Tensor,
        noise_level: Tensor,
        conditioning: AtomDenoiserContext,
        *,
        atom_mask: Tensor,
        atom_to_token: Tensor,
        token_mask: Tensor,
    ) -> tuple[Tensor, None]:
        token_repr, atom_query = self._encode(
            single_inputs,
            single_trunk,
            noisy_positions,
            noise_level,
            conditioning,
            atom_mask=atom_mask,
            atom_to_token=atom_to_token,
            token_mask=token_mask,
        )
        return (
            self.atom_decoder(
                token_repr,
                atom_query,
                conditioning.atom_conditioning,
                conditioning.atom_decoder_bias,
                atom_mask,
                atom_to_token,
                conditioning.window_info,
            ),
            None,
        )


class BoltzGenResidueTypeDenoiser(_BoltzGenDenoiser):
    r"""
    Predict per-atom position updates and per-token residue identities.
    """

    def _build_decoder(
        self,
        *,
        num_residue_types: int,
        token_hidden_size: int,
        atom_hidden_size: int,
        num_blocks: int,
        num_attention_heads: int,
        backend: str,
        block: type[ConditionedTransformerEncoderBlock],
    ) -> BoltzGenResidueTypeTokenToAtomDecoder:
        return BoltzGenResidueTypeTokenToAtomDecoder(
            num_residue_types,
            token_hidden_size,
            atom_hidden_size,
            num_blocks,
            num_attention_heads,
            backend,
            block=block,
        )

    def forward(
        self,
        single_inputs: Tensor,
        single_trunk: Tensor,
        noisy_positions: Tensor,
        noise_level: Tensor,
        conditioning: AtomDenoiserContext,
        *,
        atom_mask: Tensor,
        atom_to_token: Tensor,
        token_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        token_repr, atom_query = self._encode(
            single_inputs,
            single_trunk,
            noisy_positions,
            noise_level,
            conditioning,
            atom_mask=atom_mask,
            atom_to_token=atom_to_token,
            token_mask=token_mask,
        )
        decoder = cast(BoltzGenResidueTypeTokenToAtomDecoder, self.atom_decoder)
        return decoder.forward_with_residue_types(
            token_repr,
            atom_query,
            conditioning.atom_conditioning,
            conditioning.atom_decoder_bias,
            atom_mask,
            atom_to_token,
            conditioning.window_info,
        )


class BoltzGenCoordinateDiffusion(nn.Module):
    r"""
    EDM sampler wrapping a [`BoltzGenDenoiser`][ofoldx.models.boltzgen.model.BoltzGenDenoiser].
    """

    def __init__(
        self,
        denoiser: BoltzGenDenoiser | BoltzGenResidueTypeDenoiser,
        sigma_data: float = 16.0,
        min_noise_level: float = 4e-4,
        max_noise_level: float = 160.0,
        rho: float = 7.0,
        churn_rate: float = 0.8,
        min_churn_noise_level: float = 1.0,
        noise_scale: float = 0.95,
        step_scale: float = 1.8,
        step_scale_function: str = "constant",
        noise_scale_function: str = "constant",
        min_step_scale: float = 1.0,
        max_step_scale: float = 1.0,
        step_scale_alpha: float = 1.0,
        step_scale_beta: float = 1.0,
        min_noise_scale: float = 1.0,
        max_noise_scale: float = 1.0,
        noise_scale_alpha: float = 1.0,
        noise_scale_beta: float = 1.0,
        step_scale_random: tuple[float, ...] | None = None,
        num_sampling_steps: int = 200,
        alignment_reverse_diff: bool = True,
        sampling_schedule: str = "dilated",
        time_dilation: float = 2.667,
        time_dilation_start: float = 0.6,
        time_dilation_end: float = 0.8,
        noise_log_mean: float = -1.2,
        noise_log_std: float = 1.5,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.backend = resolve_config_backend(backend, supported=("oops", "eager"), public_default="eager")
        self.sigma_data = sigma_data
        self.preconditioner = EdmPreconditioner(sigma_data=sigma_data)
        self.denoiser = denoiser
        self.min_noise_level = min_noise_level
        self.max_noise_level = max_noise_level
        self.rho = rho
        self.churn_rate = churn_rate
        self.min_churn_noise_level = min_churn_noise_level
        self.noise_scale = noise_scale
        self.step_scale = step_scale
        self.step_scale_function = step_scale_function
        self.noise_scale_function = noise_scale_function
        self.min_step_scale = min_step_scale
        self.max_step_scale = max_step_scale
        self.step_scale_alpha = step_scale_alpha
        self.step_scale_beta = step_scale_beta
        self.min_noise_scale = min_noise_scale
        self.max_noise_scale = max_noise_scale
        self.noise_scale_alpha = noise_scale_alpha
        self.noise_scale_beta = noise_scale_beta
        self.step_scale_random = step_scale_random
        self.noise_log_mean = noise_log_mean
        self.noise_log_std = noise_log_std
        self.num_sampling_steps = num_sampling_steps
        self.alignment_reverse_diff = alignment_reverse_diff
        self.sampling_schedule = sampling_schedule
        self.time_dilation = time_dilation
        self.time_dilation_start = time_dilation_start
        self.time_dilation_end = time_dilation_end

    def preconditioned_network_forward(
        self, noisy_coordinates: Tensor, sigma: Tensor | float, **conditioning: object
    ) -> tuple[Tensor, Tensor | None]:
        active = cast(Tensor, conditioning["atom_mask"]).bool()
        while active.ndim < noisy_coordinates.ndim - 1:
            active = active.unsqueeze(-2)
        noisy_coordinates = project_model_coordinates(
            noisy_coordinates, active.expand(noisy_coordinates.shape[:-1]), feature="BoltzGen noisy coordinates"
        )
        sigma = torch.as_tensor(sigma, device=noisy_coordinates.device)
        if sigma.ndim == 0:
            sigma = sigma.expand(noisy_coordinates.shape[:-2])
        padded = sigma[..., None, None]
        update, residue_type_logits = self.denoiser(
            noisy_positions=self.preconditioner.c_in(padded) * noisy_coordinates,
            noise_level=self.preconditioner.c_noise(sigma),
            **conditioning,
        )
        coordinates = (
            self.preconditioner.c_skip(padded) * noisy_coordinates + self.preconditioner.c_out(padded) * update
        )
        return coordinates, residue_type_logits

    def noise_distribution(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        log_mean: float | None = None,
        log_std: float | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        log_mean = self.noise_log_mean if log_mean is None else log_mean
        log_std = self.noise_log_std if log_std is None else log_std
        noise = torch.randn((batch_size,), device=device, dtype=dtype, generator=generator)
        return self.sigma_data * (log_mean + log_std * noise).exp()

    def denoise_training(
        self,
        target_coordinates: Tensor,
        atom_mask: Tensor,
        *,
        num_samples: int = 1,
        sample_chunk_size: int | None = None,
        synchronize_noise_levels: bool = False,
        coordinate_augmentation: bool = True,
        augmentation_translation_scale: float = 1.0,
        noise_log_mean: float = -1.2,
        noise_log_std: float = 1.5,
        generator: torch.Generator | None = None,
        single_inputs: Tensor,
        single_trunk: Tensor,
        conditioning: AtomDenoiserContext,
        atom_to_token: Tensor,
        token_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
        r"""
        Corrupt target conformers and run one preconditioned denoising step.

        A singleton conformer is repeated for every sample; otherwise the
        conformer count must equal ``num_samples``. Returned coordinate tensors
        and noise levels retain the explicit sample axis.
        """

        sample_chunk_size = num_samples if sample_chunk_size is None else min(sample_chunk_size, num_samples)
        device = single_trunk.device
        target_coordinates = target_coordinates.to(device=device)
        target_coordinates = project_model_coordinates(
            target_coordinates,
            atom_mask.bool()[:, None, :].expand(target_coordinates.shape[:-1]),
            feature="BoltzGen corruption coordinates",
        )
        batch_size = target_coordinates.shape[0]
        conformers = target_coordinates.shape[-3]
        if conformers == 1:
            target_coordinates = target_coordinates.expand(
                -1,
                num_samples,
                -1,
                -1,
            )
        elif conformers != num_samples:
            fail("value_invalid", RequestRef("target.coordinates", str(conformers)), num_samples=num_samples)
        # Centre and rigid-augment over every model atom (upstream ``atom_pad_mask``), matching upstream
        # ``center_random_augmentation``. The resolved-atom mask is a loss/alignment concern applied by the
        # criterion, not part of the corruption equation.
        augmentation_mask = atom_mask.to(device=device, dtype=torch.bool)
        augmented_target_coordinates = _center_random_augmentation(
            target_coordinates,
            augmentation_mask,
            augmentation=coordinate_augmentation,
            translation_scale=augmentation_translation_scale,
            generator=generator,
        )
        noise_level = self.noise_distribution(
            batch_size if synchronize_noise_levels else batch_size * num_samples,
            device=augmented_target_coordinates.device,
            dtype=augmented_target_coordinates.dtype,
            log_mean=noise_log_mean,
            log_std=noise_log_std,
            generator=generator,
        )
        if synchronize_noise_levels:
            noise_level = noise_level.unsqueeze(-1).expand(batch_size, num_samples)
        else:
            noise_level = noise_level.reshape(batch_size, num_samples)
        noise = torch.empty_like(augmented_target_coordinates).normal_(generator=generator)
        noisy_coordinates = augmented_target_coordinates + noise_level[..., None, None] * noise
        denoised_chunks = []
        residue_type_chunks = []
        for sample_start in range(0, num_samples, sample_chunk_size):
            sample_end = min(sample_start + sample_chunk_size, num_samples)
            denoised_chunk, residue_type_chunk = self.preconditioned_network_forward(
                torch.narrow(noisy_coordinates, -3, sample_start, sample_end - sample_start),
                torch.narrow(noise_level, -1, sample_start, sample_end - sample_start),
                atom_mask=atom_mask,
                single_inputs=single_inputs,
                single_trunk=single_trunk,
                conditioning=conditioning,
                atom_to_token=atom_to_token,
                token_mask=token_mask,
            )
            denoised_chunks.append(denoised_chunk)
            if residue_type_chunk is not None:
                residue_type_chunks.append(residue_type_chunk)
        denoised_coordinates = denoised_chunks[0] if len(denoised_chunks) == 1 else torch.cat(denoised_chunks, dim=-3)
        residue_type_logits = None
        if residue_type_chunks:
            residue_type_logits = (
                residue_type_chunks[0] if len(residue_type_chunks) == 1 else torch.cat(residue_type_chunks, dim=-3)
            )
        return (
            denoised_coordinates,
            augmented_target_coordinates,
            noise_level,
            residue_type_logits,
        )

    def loss_weight(self, sigma: Tensor) -> Tensor:
        return edm_loss_weight(sigma, self.sigma_data)

    def sample_schedule_karras(
        self,
        num_sampling_steps: int | None = None,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        return karras_noise_schedule(
            operator.index(self.num_sampling_steps if num_sampling_steps is None else num_sampling_steps),
            max_noise_level=self.max_noise_level,
            min_noise_level=self.min_noise_level,
            rho=self.rho,
            sigma_data=self.sigma_data,
            device=device,
            dtype=dtype,
        )

    def sample_schedule_dilated(
        self,
        num_sampling_steps: int | None = None,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        num_sampling_steps = operator.index(
            self.num_sampling_steps if num_sampling_steps is None else num_sampling_steps
        )
        if num_sampling_steps < 1:
            fail("value_invalid", RequestRef("num_sampling_steps", str(num_sampling_steps)))
        inv_rho = 1 / self.rho
        ramp = torch.arange(num_sampling_steps, device=device, dtype=dtype) / max(num_sampling_steps - 1, 1)
        dilated = _dilate_time(
            ramp,
            start=self.time_dilation_start,
            end=self.time_dilation_end,
            dilation=self.time_dilation,
        )
        sigmas = (
            self.max_noise_level**inv_rho + dilated * (self.min_noise_level**inv_rho - self.max_noise_level**inv_rho)
        ) ** self.rho
        sigmas = sigmas * self.sigma_data
        return torch.nn.functional.pad(sigmas, (0, 1), value=0.0)

    def sample_schedule(
        self,
        num_sampling_steps: int | None = None,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        if self.sampling_schedule == "karras":
            return self.sample_schedule_karras(num_sampling_steps, device=device, dtype=dtype)
        if self.sampling_schedule == "dilated":
            return self.sample_schedule_dilated(num_sampling_steps, device=device, dtype=dtype)
        fail("value_invalid", RequestRef("sampling_schedule", str(self.sampling_schedule)))

    def beta_step_scale_schedule(
        self, num_sampling_steps: int | None = None, *, device: torch.device, dtype: torch.dtype
    ) -> Tensor:
        r"""Return the configured beta-CDF step scales, one per transition."""

        return beta_scale_schedule(
            operator.index(self.num_sampling_steps if num_sampling_steps is None else num_sampling_steps),
            min_scale=self.min_step_scale,
            max_scale=self.max_step_scale,
            alpha=self.step_scale_alpha,
            beta=self.step_scale_beta,
            device=device,
            dtype=dtype,
        )

    def beta_noise_scale_schedule(
        self, num_sampling_steps: int | None = None, *, device: torch.device, dtype: torch.dtype
    ) -> Tensor:
        r"""Return the configured reverse-time beta-CDF noise scales."""

        return beta_scale_schedule(
            operator.index(self.num_sampling_steps if num_sampling_steps is None else num_sampling_steps),
            min_scale=self.min_noise_scale,
            max_scale=self.max_noise_scale,
            alpha=self.noise_scale_beta,
            beta=self.noise_scale_alpha,
            device=device,
            dtype=dtype,
        )

    @torch.no_grad()
    def sample(
        self,
        atom_mask: Tensor,
        *,
        reference_positions: Tensor,
        num_sampling_steps: int | None = None,
        num_samples: int = 1,
        generator: torch.Generator | None = None,
        step_scale: float | Tensor | None = None,
        noise_scale: float | Tensor | None = None,
        step_scale_schedule: Tensor | None = None,
        noise_scale_schedule: Tensor | None = None,
        return_trajectory: bool = False,
        **conditioning: object,
    ) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor | None]:
        single_trunk = cast(Tensor, conditioning["single_trunk"])
        device, dtype = single_trunk.device, single_trunk.dtype
        coordinate_template = reference_positions.unsqueeze(-3).expand(
            *reference_positions.shape[:-2],
            num_samples,
            *reference_positions.shape[-2:],
        )
        num_sampling_steps = operator.index(
            self.num_sampling_steps if num_sampling_steps is None else num_sampling_steps
        )
        sigmas = self.sample_schedule(num_sampling_steps, device=device, dtype=dtype)
        gammas = torch.where(sigmas > self.min_churn_noise_level, self.churn_rate, 0.0)
        if step_scale_schedule is not None:
            step_scales = step_scale_schedule.to(device=device)
        elif step_scale is None and self.training and self.step_scale_random is not None:
            random_device = device if generator is None else generator.device
            choices = torch.tensor(self.step_scale_random, device=random_device, dtype=dtype)
            selected = torch.randint(len(self.step_scale_random), (), device=random_device, generator=generator)
            step_scales = choices[selected].to(device=device).expand(num_sampling_steps)
        elif step_scale is None and self.step_scale_function == "beta":
            step_scales = self.beta_step_scale_schedule(num_sampling_steps, device=device, dtype=dtype)
        else:
            value = self.step_scale if step_scale is None else step_scale
            step_scale_value = torch.as_tensor(value, device=device, dtype=None if isinstance(value, Tensor) else dtype)
            if step_scale_value.ndim:
                step_scale_value = step_scale_value.reshape(*coordinate_template.shape[:-2])
            step_scales = step_scale_value.unsqueeze(0).expand(num_sampling_steps, *step_scale_value.shape)
        if noise_scale_schedule is not None:
            noise_scales = noise_scale_schedule.to(device=device)
        elif noise_scale is None and self.noise_scale_function == "beta":
            noise_scales = self.beta_noise_scale_schedule(num_sampling_steps, device=device, dtype=dtype)
        else:
            value = self.noise_scale if noise_scale is None else noise_scale
            noise_scale_value = torch.as_tensor(
                value, device=device, dtype=None if isinstance(value, Tensor) else dtype
            )
            if noise_scale_value.ndim:
                noise_scale_value = noise_scale_value.reshape(*coordinate_template.shape[:-2])
            noise_scales = noise_scale_value.unsqueeze(0).expand(num_sampling_steps, *noise_scale_value.shape)
        step_scales = step_scales[..., None, None]
        noise_scales = noise_scales[..., None, None]

        def randn() -> Tensor:
            r"""
            Draw standard Gaussian noise with the explicit sample and atom axes.
            """
            return torch.empty_like(coordinate_template, device=device, dtype=dtype).normal_(generator=generator)

        coords = sigmas[0] * randn()
        residue_type_logits = None
        if return_trajectory:
            coordinate_steps: list[Tensor] = [coords]
            denoised_steps: list[Tensor] = []
        # The loop owns the squared-noise-scale churn factor, optional mid-step alignment, dilated Karras schedule,
        # rigid augmentation, and random draws as one sampling algorithm.
        # The fixed Python loop controls denoiser invocation; schedule values stay as tensors so sampling does not
        # synchronize them to the host on every step.
        for sigma_current, sigma_next, gamma, step_scale_value, noise_scale_value in zip(
            sigmas[:-1], sigmas[1:], gammas[1:], step_scales, noise_scales, strict=True
        ):
            batch_shape = coords.shape[:-2]
            rotation = uniform_random_rotation(
                math.prod(batch_shape),
                device=device,
                dtype=coords.dtype,
                generator=generator,
            ).reshape(*batch_shape, 3, 3)
            translation = torch.randn(
                *batch_shape,
                1,
                3,
                device=device,
                dtype=dtype,
                generator=generator,
            )
            observed = atom_mask.bool().unsqueeze(-2)
            coords_center = weighted_mean(coords.transpose(-1, -2), observed.unsqueeze(-2)).unsqueeze(-2)
            coords_center = torch.where(observed.any(-1, keepdim=True).unsqueeze(-1), coords_center, 0)
            coords = rotate_vectors(coords - coords_center, rotation) + translation
            sigma_hat = sigma_current * (1 + gamma)
            noise_var = noise_scale_value**2 * (sigma_hat**2 - sigma_current**2)
            noisy = coords + noise_scale_value * torch.sqrt(torch.clamp(noise_var, min=0.0)) * randn()
            step_sigma = torch.ones(coords.shape[:-2], device=coords.device, dtype=coords.dtype) * sigma_hat
            denoised, residue_type_logits = self.preconditioned_network_forward(
                noisy,
                step_sigma,
                atom_mask=atom_mask,
                **conditioning,
            )
            if self.alignment_reverse_diff:
                noisy = weighted_rigid_align(
                    source=noisy,
                    target=denoised,
                    weight=atom_mask.unsqueeze(-2),
                    backend=self.backend,
                )
            update_weight = step_scale_value * (sigma_hat - sigma_next) / sigma_hat
            update_dtype = torch.promote_types(noisy.dtype, torch.promote_types(denoised.dtype, update_weight.dtype))
            coords = torch.lerp(
                noisy.to(dtype=update_dtype),
                denoised.to(dtype=update_dtype),
                update_weight.to(dtype=update_dtype),
            )
            if return_trajectory:
                coordinate_steps.append(coords)
                denoised_steps.append(denoised)
        if not return_trajectory:
            return coords, residue_type_logits, None, None
        return (
            coords,
            residue_type_logits,
            torch.stack(coordinate_steps, dim=-3),
            torch.stack(denoised_steps, dim=-3),
        )


class BoltzGenConfidenceHead(nn.Module):
    r"""
    BoltzGen confidence trunk and complete learned output heads.
    """

    boundaries: Tensor

    def __init__(
        self,
        single_size: int = 384,
        pair_size: int = 128,
        num_blocks: int = 8,
        num_pair_attention_heads: int = 16,
        triangle_attention_head_size: int = 32,
        num_triangle_attention_heads: int = 4,
        pair_dropout: float = 0.25,
        post_layer_norm: bool = False,
        num_distogram_bins: int = 64,
        max_distance: float = 22.0,
        token_level_confidence: bool = True,
        add_single_input: bool = False,
        add_single_to_pair_product: bool = False,
        use_separate_heads: bool = False,
        num_plddt_bins: int = 50,
        num_pae_bins: int = 64,
        num_pde_bins: int = 64,
        return_latent_features: bool = False,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.token_level_confidence = token_level_confidence
        self.backend = resolve_config_backend(backend, supported=("oops", "eager"), public_default="eager")
        self.add_single_input = add_single_input
        self.add_single_to_pair_product = add_single_to_pair_product
        self.return_latent_features = return_latent_features

        self.input_single_norm = LayerNorm(single_size, backend=backend)
        self.single_norm = LayerNorm(single_size, backend=backend)
        if add_single_input:
            self.input_single_projection = nn.Linear(single_size, single_size, bias=False)
        self.pair_norm = LayerNorm(pair_size, backend=backend)
        self.pair_left_projection = nn.Linear(single_size, pair_size, bias=False)
        self.pair_right_projection = nn.Linear(single_size, pair_size, bias=False)
        if add_single_to_pair_product:
            self.single_to_pair_product_left = nn.Linear(single_size, pair_size, bias=False)
            self.single_to_pair_product_right = nn.Linear(single_size, pair_size, bias=False)
            self.single_to_pair_product_out = nn.Linear(pair_size, pair_size, bias=False)
        self.distance_embedding = nn.Embedding(num_distogram_bins, pair_size)
        # The boundaries follow from the configuration, so they are computed rather than stored.
        # A model built on the meta device still needs real boundaries, since loading never fills them.
        buffer_device = self.distance_embedding.weight.device
        if buffer_device.type == "meta":
            buffer_device = torch.device("cpu")
        self.register_buffer(
            "boundaries",
            torch.linspace(2.0, max_distance, num_distogram_bins - 1, device=buffer_device),
            persistent=False,
        )
        self.pairformer_stack = PairformerStack(
            num_blocks=num_blocks,
            single_size=single_size,
            pair_size=pair_size,
            triangle_attention_head_size=triangle_attention_head_size,
            num_triangle_attention_heads=num_triangle_attention_heads,
            num_pair_attention_heads=num_pair_attention_heads,
            dropout=pair_dropout,
            block=PostNormLineDropoutPairformerBlock if post_layer_norm else LineDropoutPairformerBlock,
            backend=backend,
        )
        self.output_head = BoltzGenConfidenceOutputHead(
            single_size=single_size,
            pair_size=pair_size,
            token_level_confidence=token_level_confidence,
            use_separate_heads=use_separate_heads,
            num_plddt_bins=num_plddt_bins,
            num_pae_bins=num_pae_bins,
            num_pde_bins=num_pde_bins,
        )

        nn.init.zeros_(self.distance_embedding.weight)
        nn.init.zeros_(self.pair_left_projection.weight)
        nn.init.zeros_(self.pair_right_projection.weight)
        if add_single_to_pair_product:
            nn.init.zeros_(self.single_to_pair_product_out.weight)
        if add_single_input:
            nn.init.zeros_(self.input_single_projection.weight)

    def forward(
        self,
        single_inputs: Tensor,
        single: Tensor,
        pair: Tensor,
        coordinates: Tensor,
        *,
        input: BoltzGenInput,
        pred_distogram_logits: Tensor,
    ) -> BoltzGenConfidenceOutput:
        batch_size, num_samples, num_atoms, _ = coordinates.shape
        coordinates = project_model_coordinates(
            coordinates,
            input.atom_mask.bool()[:, None, :].expand(coordinates.shape[:-1]),
            feature="BoltzGen confidence coordinates",
        )
        flat_coordinates = coordinates.reshape(batch_size * num_samples, num_atoms, 3)

        single_inputs = self.input_single_norm(single_inputs)
        single = self.single_norm(single)
        if self.add_single_input:
            single = single + self.input_single_projection(single_inputs)
        pair = (
            self.pair_norm(pair)
            + self.pair_left_projection(single_inputs).unsqueeze(-2)
            + self.pair_right_projection(single_inputs).unsqueeze(-3)
        )
        if self.add_single_to_pair_product:
            pair = pair + self.single_to_pair_product_out(
                self.single_to_pair_product_left(single_inputs).unsqueeze(-2)
                * self.single_to_pair_product_right(single_inputs).unsqueeze(-3)
            )

        single = single.repeat_interleave(num_samples, dim=0)
        pair = pair.repeat_interleave(num_samples, dim=0)
        representative_atom_index = input.token_center_atom_index.repeat_interleave(num_samples, dim=0).long()
        representative_coordinates = flat_coordinates.gather(
            1,
            representative_atom_index.unsqueeze(-1).expand(-1, -1, 3),
        )
        representative_distances = pairwise_distance(representative_coordinates)
        distance_bins = (representative_distances.unsqueeze(-1) > self.boundaries).sum(dim=-1)
        pair = pair + self.distance_embedding(distance_bins)

        token_mask = input.token_mask.repeat_interleave(num_samples, dim=0)
        pair_mask = pair_mask_from_token_mask(token_mask, backend=self.backend)
        single, pair = self.pairformer_stack(single, pair, pair_mask=pair_mask)
        pae_logits, pde_logits, plddt_logits, resolved_logits = self.output_head(
            single,
            pair,
            input=input,
            num_samples=num_samples,
        )
        return _boltzgen_confidence_output(
            pae_logits=pae_logits,
            pde_logits=pde_logits,
            plddt_logits=plddt_logits,
            resolved_logits=resolved_logits,
            confidence_single=single if self.return_latent_features else None,
            confidence_pair=pair if self.return_latent_features else None,
            coordinates=flat_coordinates,
            representative_distances=representative_distances,
            pred_distogram_logits=pred_distogram_logits,
            input=input,
            num_samples=num_samples,
            token_level_confidence=self.token_level_confidence,
            backend=self.backend,
        )


class BoltzGenConfidenceOutputHead(nn.Module):
    r"""
    Project confidence representations to PAE, PDE, pLDDT, and resolved logits.
    """

    def __init__(
        self,
        single_size: int = 384,
        pair_size: int = 128,
        token_level_confidence: bool = True,
        use_separate_heads: bool = False,
        num_plddt_bins: int = 50,
        num_pae_bins: int = 64,
        num_pde_bins: int = 64,
    ) -> None:
        super().__init__()
        self.token_level_confidence = token_level_confidence
        self.use_separate_heads = use_separate_heads
        if use_separate_heads:
            self.pae_intra_projection = nn.Linear(pair_size, num_pae_bins, bias=False)
            self.pae_inter_projection = nn.Linear(pair_size, num_pae_bins, bias=False)
            self.pde_intra_projection = nn.Linear(pair_size, num_pde_bins, bias=False)
            self.pde_inter_projection = nn.Linear(pair_size, num_pde_bins, bias=False)
        else:
            self.pae_projection = nn.Linear(pair_size, num_pae_bins, bias=False)
            self.pde_projection = nn.Linear(pair_size, num_pde_bins, bias=False)
        plddt_size = num_plddt_bins if token_level_confidence else 23 * num_plddt_bins
        resolved_size = 2 if token_level_confidence else 23 * 2
        self.plddt_projection = nn.Linear(single_size, plddt_size, bias=False)
        self.resolved_projection = nn.Linear(single_size, resolved_size, bias=False)

    def forward(
        self,
        single: Tensor,
        pair: Tensor,
        *,
        input: BoltzGenInput,
        num_samples: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        chain_index = input.chain_index.repeat_interleave(num_samples, dim=0)
        if self.use_separate_heads:
            same_chain = (chain_index.unsqueeze(-1) == chain_index.unsqueeze(-2)).unsqueeze(-1)
            pae_logits = torch.where(same_chain, self.pae_intra_projection(pair), self.pae_inter_projection(pair))
            symmetric_pair = pair + pair.transpose(-3, -2)
            pde_logits = torch.where(
                same_chain, self.pde_intra_projection(symmetric_pair), self.pde_inter_projection(symmetric_pair)
            )
            plddt_logits = self.plddt_projection(single)
            resolved_logits = self.resolved_projection(single)
        else:
            logits = token_confidence_logits(
                single,
                pair,
                pae_projection=self.pae_projection,
                pde_projection=self.pde_projection,
                plddt_projection=self.plddt_projection,
                resolved_projection=self.resolved_projection,
            )
            pae_logits, pde_logits, plddt_logits, resolved_logits = (
                logits[key] for key in ("pae_logits", "pde_logits", "plddt_logits", "resolved_logits")
            )
        if self.token_level_confidence:
            return pae_logits, pde_logits, plddt_logits, resolved_logits

        num_bins = plddt_logits.shape[-1] // 23
        plddt_slots = plddt_logits.reshape(*plddt_logits.shape[:-1], 23, num_bins)
        resolved_slots = resolved_logits.reshape(*resolved_logits.shape[:-1], 23, 2)
        atom_to_token = input.atom_to_token.repeat_interleave(num_samples, dim=0).long()
        atom_token_index = input.atom_token_index.repeat_interleave(num_samples, dim=0).long()
        batch_index = torch.arange(single.shape[0], device=single.device).unsqueeze(-1)
        plddt_logits = plddt_slots[batch_index, atom_to_token, atom_token_index]
        resolved_logits = resolved_slots[batch_index, atom_to_token, atom_token_index]
        atom_mask = input.atom_mask.repeat_interleave(num_samples, dim=0).to(plddt_logits)
        plddt_logits = plddt_logits * atom_mask.unsqueeze(-1)
        resolved_logits = resolved_logits * atom_mask.unsqueeze(-1)
        return pae_logits, pde_logits, plddt_logits, resolved_logits


class BoltzGenInverseFoldingEncoder(nn.Module):
    r"""
    kNN-graph encoder over tokens (BoltzGen inverse folding).
    """

    def __init__(
        self,
        single_size: int = 384,
        node_size: int = 128,
        edge_size: int = 128,
        hidden_size: int = 128,
        num_blocks: int = 3,
        num_attention_heads: int = 4,
        num_neighbors: int = 30,
        max_relative_position: int = 32,
        dropout: float = 0.1,
        softmax_dropout: float = 0.2,
        coordinate_noise: float = 0.2,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.backend = resolve_config_backend(backend, supported=("oops", "eager"), public_default="eager")
        self.num_blocks = num_blocks
        self.gradient_checkpointing = False
        self.num_neighbors = num_neighbors
        self.max_relative_position = max_relative_position
        self.coordinate_noise = coordinate_noise
        # geo (4x4 backbone-atom distances x 16 RBF = 256)
        # + edge attrs (2*max_relative_position+2 + same_chain + same_residue)
        # + bond (token_bonds + one-hot bond type).
        self.edge_input_size = 256 + (2 * max_relative_position + 2) + 2 + 1 + (_NUM_BOND_TYPES + 1)

        self.distance_smearing = GaussianRadialBasis(
            num_bins=16, lower_bound=0.0, upper_bound=20.0, width=math.sqrt(2) * 20.0 / 15
        )
        self.node_input_projection = nn.Linear(single_size, node_size)
        self.edge_input_projection = nn.Linear(self.edge_input_size, edge_size)
        self.blocks = nn.ModuleList(
            BoltzGenInverseFoldingGraphEncoderBlock(
                node_size=node_size,
                edge_size=edge_size,
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                dropout=dropout,
                softmax_dropout=softmax_dropout,
            )
            for _ in range(num_blocks)
        )

    _gradient_checkpointing_func: Callable[..., tuple[Tensor, Tensor]]

    @torch.no_grad()
    def build_knn_graph(self, input: BoltzGenInverseFoldingInput) -> tuple[Tensor, Tensor, Tensor]:
        r"""
        The decoder's neighbour graph over the valid tokens, packed across the batch.

        Each valid token keeps its ``num_neighbors`` nearest valid tokens of its own sample, itself included
        (``multimolecule.graph.knn_graph``). ``edge_index`` is ``(source, destination)`` in packed token
        indices; the edge features are the token's bond indicator and its one-hot bond type.
        """

        valid_mask = input.token_mask.bool() & input.token_resolved_mask.bool()
        batch_size, num_tokens = valid_mask.shape
        centers = input.center_coordinates.masked_fill(~valid_mask.unsqueeze(-1), 0)
        sample = torch.arange(batch_size, device=centers.device)[:, None].expand(batch_size, num_tokens)
        edge_index = knn_graph(centers[valid_mask], min(self.num_neighbors, num_tokens), sample[valid_mask], loop=True)
        token = torch.nonzero(valid_mask)
        source, destination = token[edge_index[0]], token[edge_index[1]]
        # token_bonds carries a trailing singleton (B, N, N, 1); the reference featurizer drops it the same way.
        token_bonds = input.token_bonds[..., 0][destination[:, 0], destination[:, 1], source[:, 1]]
        type_bonds = input.type_bonds[destination[:, 0], destination[:, 1], source[:, 1]]
        bond = torch.cat([token_bonds[..., None], F.one_hot(type_bonds, num_classes=_NUM_BOND_TYPES + 1)], dim=-1)
        return edge_index, valid_mask, bond

    @torch.no_grad()
    def edge_geometry(
        self,
        input: BoltzGenInverseFoldingInput,
        edge_index: Tensor,
        valid_mask: Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        src, dst = edge_index[0], edge_index[1]
        batch_size, num_tokens = valid_mask.shape
        input = normalize_boltzgen_input(input)
        atom_coordinate_mask = input.atom_coordinate_mask.bool() & input.atom_mask.bool()
        coords = input.atom_coordinates.masked_fill(~atom_coordinate_mask.unsqueeze(-1), 0)
        if self.training and self.coordinate_noise > 0:
            coords = (
                coords
                + torch.randn(
                    coords.shape,
                    device=coords.device,
                    dtype=coords.dtype,
                    generator=generator,
                )
                * self.coordinate_noise
            )
        backbone, backbone_mask = _gather_backbone4(
            input.token_to_bb4_atoms,
            coords,
            atom_coordinate_mask,
            batch_size,
            num_tokens,
        )
        backbone, backbone_mask = backbone[valid_mask], backbone_mask[valid_mask]
        distance_mask = backbone_mask[src, None] & backbone_mask[dst, :, None]
        displacement = backbone[src, None] - backbone[dst, :, None]
        displacement = displacement.masked_fill(~distance_mask.unsqueeze(-1), 0)
        distance = torch.norm(displacement, dim=-1).flatten(start_dim=1)
        geometry = self.distance_smearing(distance)
        geometry = geometry.masked_fill(~distance_mask.flatten(start_dim=1).unsqueeze(-1), 0)
        return geometry.flatten(start_dim=1)

    @torch.no_grad()
    def edge_attributes(self, input: BoltzGenInverseFoldingInput, edge_index: Tensor, valid_mask: Tensor) -> Tensor:
        src, dst = edge_index[0], edge_index[1]
        chain_index = input.chain_index[valid_mask]
        same_chain = chain_index[src] == chain_index[dst]
        residue = input.residue_index[valid_mask]
        same_residue = residue[src] == residue[dst]
        delta = torch.clip(residue[dst] - residue[src] + self.max_relative_position, 0, 2 * self.max_relative_position)
        delta = torch.where(same_chain, delta, torch.zeros_like(delta) + 2 * self.max_relative_position + 1)
        relative = F.one_hot(delta, 2 * self.max_relative_position + 2)
        return torch.cat([relative, same_chain[..., None], same_residue[..., None]], dim=-1).to(
            input.atom_coordinates.dtype
        )

    def forward(
        self,
        input: BoltzGenInverseFoldingInput,
        *,
        single_inputs: Tensor,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        edge_index, valid_mask, bond = self.build_knn_graph(input)
        geometry = self.edge_geometry(
            input,
            edge_index,
            valid_mask,
            generator=generator,
        )
        edge_attr = self.edge_attributes(input, edge_index, valid_mask)

        node = self.node_input_projection(single_inputs[valid_mask])
        edge = self.edge_input_projection(torch.cat([geometry, edge_attr.to(geometry), bond.to(geometry)], dim=-1))
        for block in self.blocks:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                node, edge = self._gradient_checkpointing_func(
                    block.__call__,
                    node,
                    edge,
                    edge_index,
                )
            else:
                node, edge = block(node, edge, edge_index)
        return edge_index, valid_mask, node, edge


class BoltzGenInverseFoldingDecoder(nn.Module):
    r"""
    Graph decoder predicting residue-type logits (BoltzGen inverse folding).
    """

    def __init__(
        self,
        node_size: int = 128,
        edge_size: int = 128,
        hidden_size: int = 128,
        num_blocks: int = 3,
        num_attention_heads: int = 4,
        vocab_size: int = _NUM_RESIDUE_TYPES,
        dropout: float = 0.1,
        softmax_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.gradient_checkpointing = False
        self.vocab_size = vocab_size
        # Each block's edge state is cat[encoder_edge, neighbour_node + residue_rep] = edge_size + node_size.
        decoder_edge_size = edge_size + node_size
        self.residue_type_projection = nn.Linear(vocab_size, node_size)
        self.blocks = nn.ModuleList(
            BoltzGenInverseFoldingGraphDecoderBlock(
                node_size=node_size,
                edge_size=decoder_edge_size,
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                dropout=dropout,
                softmax_dropout=softmax_dropout,
            )
            for _ in range(num_blocks)
        )
        self.residue_type_head = nn.Linear(node_size, vocab_size, bias=False)

    _gradient_checkpointing_func: Callable[..., Tensor]

    def forward(
        self,
        node: Tensor,
        edge: Tensor,
        edge_index: Tensor,
        valid_mask: Tensor,
        target_residue_type: Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        src, dst = edge_index[0], edge_index[1]
        with torch.no_grad():
            # Teacher forcing with random per-edge visibility: an edge reveals its source residue type only when
            # rand[src] < rand[dst], so each node sees a random subset of neighbor identities.
            rand = torch.rand(node.shape[0], device=node.device, generator=generator)
            visibility = rand[src] < rand[dst]
            known = target_residue_type.bool()[valid_mask][src] * visibility[:, None]
            known = known.to(edge.dtype)
        residue_rep = self.residue_type_projection(known)
        neighbors = torch.cat([edge, node[src] + residue_rep], dim=-1)

        for block in self.blocks:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                node = self._gradient_checkpointing_func(
                    block.__call__,
                    node,
                    neighbors,
                    edge_index,
                )
            else:
                node = block(node, neighbors, edge_index)

        logits = self.residue_type_head(node)
        batch_size, num_tokens = valid_mask.shape
        dense = logits.new_zeros(batch_size, num_tokens, self.vocab_size)
        dense[valid_mask] = logits
        return dense

    @torch.no_grad()
    def sample(
        self,
        node: Tensor,
        edge: Tensor,
        edge_index: Tensor,
        valid_mask: Tensor,
        input: BoltzGenInverseFoldingInput,
        *,
        temperature: float | None = 0.1,
        tie_symmetric_sequences: bool = True,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        src, dst = edge_index[0], edge_index[1]
        num_nodes = node.shape[0]
        design_mask = input.design_mask.bool()[valid_mask]
        decoded = node.new_zeros(num_nodes, self.vocab_size)
        logits = node.new_zeros(num_nodes, self.vocab_size)

        # the identity support is stated over the canonical classes; sampling runs in the trained contiguous block
        support = input.allowed_token_mask[valid_mask].to(device=node.device)
        blocked = ~support[..., _CANONICALS_OFFSET : _CANONICALS_OFFSET + NUM_CANONICAL_TOKENS]
        all_blocked = blocked[design_mask].all(dim=-1)
        if all_blocked.any():
            design_positions = torch.where(design_mask)[0]
            fail(
                "value_invalid",
                RequestRef("conditions", "a designed position with no allowed identity"),
                positions=design_positions[all_blocked].tolist(),
            )

        order = torch.randperm(num_nodes, device=node.device, generator=generator).tolist()
        if (~design_mask).any():
            known = input.residue_type_features[valid_mask][~design_mask].to(decoded.dtype)
            decoded[~design_mask] = known
            logits[~design_mask] = known
            fixed = set(torch.where(~design_mask)[0].tolist())
            order = [position for position in order if position not in fixed]

        symmetry_groups, position_to_group = (
            self._build_symmetric_groups(input, valid_mask) if tie_symmetric_sequences else ({}, {})
        )
        group_anchors: dict[int, int] = {}
        for anchor_group_id, group_positions in symmetry_groups.items():
            fixed_positions = [position for position in group_positions if not bool(design_mask[position])]
            if not fixed_positions:
                continue
            identities = decoded[fixed_positions].argmax(dim=-1).unique()
            if identities.numel() != 1:
                fail(
                    "value_invalid",
                    RequestRef("conditions", "a tied group with conflicting fixed identities"),
                    positions=fixed_positions,
                )
            identity = int(identities.item())
            canonical_identity = identity - _CANONICALS_OFFSET
            if not 0 <= canonical_identity < NUM_CANONICAL_TOKENS:
                fail(
                    "value_invalid",
                    RequestRef("conditions", "a tied group with a noncanonical fixed identity"),
                    positions=fixed_positions,
                )
            group_anchors[anchor_group_id] = canonical_identity

        sampled: set[int] = set()

        for position in order:
            if position in sampled:
                continue
            group_id = position_to_group.get(position)
            positions = (
                [member for member in symmetry_groups[group_id] if bool(design_mask[member])]
                if group_id is not None
                else [position]
            )

            aggregated_logits: Tensor | None = None
            for pos in positions:
                incoming = dst == pos
                neighbors = torch.cat(
                    [edge[incoming], node[src[incoming]] + self.residue_type_projection(decoded[src[incoming]])],
                    dim=-1,
                )
                state = node[pos : pos + 1]
                for block in self.blocks:
                    state = cast(BoltzGenInverseFoldingGraphDecoderBlock, block).sample(state, neighbors)
                position_logits = self.residue_type_head(state)
                aggregated_logits = (
                    position_logits if aggregated_logits is None else aggregated_logits + position_logits
                )
            aggregated_logits = cast(Tensor, aggregated_logits) / len(positions)

            shared_blocked = blocked[positions].any(dim=0, keepdim=True)
            if bool(shared_blocked.all()):
                fail(
                    "value_invalid",
                    RequestRef("conditions", "a tied group with no shared identity"),
                    positions=positions,
                )
            canonical = aggregated_logits[
                :, _CANONICALS_OFFSET : _CANONICALS_OFFSET + NUM_CANONICAL_TOKENS
            ].masked_fill(shared_blocked, -torch.inf)
            anchor = group_anchors.get(group_id) if group_id is not None else None
            if anchor is not None:
                if bool(shared_blocked[0, anchor]):
                    fail(
                        "value_invalid",
                        RequestRef("conditions", "a tied group excluding its fixed identity"),
                        positions=positions,
                    )
                chosen = canonical.new_tensor([anchor], dtype=torch.long)
            elif temperature is None:
                chosen = canonical.argmax(dim=-1)
            else:
                chosen = torch.multinomial(
                    torch.softmax(canonical / temperature, dim=-1),
                    num_samples=1,
                    generator=generator,
                ).squeeze(-1)
            one_hot = torch.nn.functional.one_hot(chosen + _CANONICALS_OFFSET, self.vocab_size).to(decoded.dtype)
            for pos in positions:
                decoded[pos] = one_hot
                logits[pos] = aggregated_logits
                sampled.add(pos)

        batch_size, num_tokens = valid_mask.shape
        dense_logits = logits.new_zeros(batch_size, num_tokens, self.vocab_size)
        dense_logits[valid_mask] = logits
        dense_res_type = decoded.new_zeros(batch_size, num_tokens, self.vocab_size)
        dense_res_type[valid_mask] = decoded
        return dense_logits, dense_res_type

    def _build_symmetric_groups(
        self,
        input: BoltzGenInverseFoldingInput,
        valid_mask: Tensor,
    ) -> tuple[dict[int, list[int]], dict[int, int]]:
        batch_indices = valid_mask.nonzero(as_tuple=False)[:, 0].tolist()
        group_indices = input.symmetric_group[valid_mask].tolist()
        key_to_positions: dict[tuple[int, int], list[int]] = {}
        for position, (batch_index, group) in enumerate(zip(batch_indices, group_indices, strict=True)):
            if group <= 0:  # 0 = no symmetry group.
                continue
            key_to_positions.setdefault((batch_index, group), []).append(position)

        groups: dict[int, list[int]] = {}
        position_to_group: dict[int, int] = {}
        group_id = 0
        for positions in key_to_positions.values():
            if len(positions) > 1:
                groups[group_id] = positions
                for position in positions:
                    position_to_group[position] = group_id
                group_id += 1
        return groups, position_to_group


class BoltzGenInverseFoldingGraphEncoderBlock(nn.Module):
    r"""
    One MLP-attention GNN block with node and edge updates.
    """

    def __init__(
        self,
        node_size: int = 128,
        edge_size: int = 128,
        hidden_size: int = 128,
        num_attention_heads: int = 4,
        dropout: float = 0.1,
        softmax_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.edge_feed_forward = FeedForward(
            node_size * 2 + edge_size,
            hidden_size,
            edge_size,
            activation="gelu",
        )
        self.edge_dropout = nn.Dropout(dropout)
        self.edge_norm = nn.BatchNorm1d(edge_size)
        self.attention = BoltzGenInverseFoldingGraphAttention(
            node_size,
            node_size + edge_size,
            hidden_size,
            num_attention_heads,
            softmax_dropout,
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.attention_norm = nn.BatchNorm1d(node_size)
        self.node_feed_forward = FeedForward(
            node_size,
            hidden_size,
            node_size,
            activation="gelu",
        )
        self.node_dropout = nn.Dropout(dropout)
        self.node_norm = nn.BatchNorm1d(node_size)

    def forward(self, node: Tensor, edge: Tensor, edge_index: Tensor) -> tuple[Tensor, Tensor]:
        src, dst = edge_index[0], edge_index[1]
        edge_update = self.edge_feed_forward(torch.cat([node[src], node[dst], edge], dim=1))
        edge = edge + self.edge_norm(self.edge_dropout(edge_update))
        attention_update = self.attention(node, torch.cat([node[src], edge], dim=1), dst)
        node = node + self.attention_norm(self.attention_dropout(attention_update))
        feed_forward_update = self.node_feed_forward(node)
        return node + self.node_norm(self.node_dropout(feed_forward_update)), edge


class BoltzGenInverseFoldingGraphDecoderBlock(nn.Module):
    r"""
    One MLP-attention GNN decoder block updating nodes only.
    """

    def __init__(
        self,
        node_size: int = 128,
        edge_size: int = 256,
        hidden_size: int = 128,
        num_attention_heads: int = 4,
        dropout: float = 0.1,
        softmax_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.attention = BoltzGenInverseFoldingGraphAttention(
            node_size,
            edge_size,
            hidden_size,
            num_attention_heads,
            softmax_dropout,
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.attention_norm = nn.BatchNorm1d(node_size)
        self.node_feed_forward = FeedForward(
            node_size,
            hidden_size,
            node_size,
            activation="gelu",
        )
        self.node_dropout = nn.Dropout(dropout)
        self.node_norm = nn.BatchNorm1d(node_size)

    def forward(self, node: Tensor, edge: Tensor, edge_index: Tensor) -> Tensor:
        return self._update(node, edge, edge_index[1])

    def sample(self, node: Tensor, edge: Tensor) -> Tensor:
        destination_index = torch.zeros(edge.shape[0], dtype=torch.long, device=edge.device)
        return self._update(node, edge, destination_index)

    def _update(self, node: Tensor, context: Tensor, destination_index: Tensor) -> Tensor:
        attention_update = self.attention(node, context, destination_index)
        node = node + self.attention_norm(self.attention_dropout(attention_update))
        feed_forward_update = self.node_feed_forward(node)
        return node + self.node_norm(self.node_dropout(feed_forward_update))


class BoltzGenInverseFoldingGraphAttention(nn.Module):
    r"""
    Aggregate packed edge context with destination-grouped attention.
    """

    def __init__(
        self,
        node_size: int,
        context_size: int,
        hidden_size: int,
        num_attention_heads: int,
        softmax_dropout: float,
    ) -> None:
        super().__init__()
        self.softmax_dropout = softmax_dropout
        self.weight_feed_forward = FeedForward(
            node_size + context_size,
            hidden_size,
            num_attention_heads,
            num_hidden_layers=2,
            activation="gelu",
        )
        self.value_feed_forward = FeedForward(
            context_size,
            hidden_size,
            node_size,
            num_hidden_layers=2,
            activation="gelu",
        )
        self.output_projection = nn.Linear(node_size * num_attention_heads, node_size)

    def forward(self, node: Tensor, context: Tensor, destination_index: Tensor) -> Tensor:
        weight = self.weight_feed_forward(torch.cat([node[destination_index], context], dim=1))
        value = self.value_feed_forward(context)
        if self.training and self.softmax_dropout > 0:
            weight = _softmax_dropout(weight, destination_index, node.shape[0], self.softmax_dropout)
        weight = grouped_softmax(weight, destination_index, dim=0, num_groups=node.shape[0])
        message = indexed_sum(
            weight.unsqueeze(-1) * value.unsqueeze(1),
            destination_index,
            dim=0,
            num_groups=node.shape[0],
        ).flatten(start_dim=1)
        return self.output_projection(message)


# Scoring heads are optional so the design-only generator keeps its base parameter count.


def _boltzgen_confidence_output(
    *,
    pae_logits: Tensor,
    pde_logits: Tensor,
    plddt_logits: Tensor,
    resolved_logits: Tensor,
    confidence_single: Tensor | None,
    confidence_pair: Tensor | None,
    coordinates: Tensor,
    representative_distances: Tensor,
    pred_distogram_logits: Tensor,
    input: BoltzGenInput,
    num_samples: int,
    token_level_confidence: bool,
    backend: str = "auto",
) -> BoltzGenConfidenceOutput:
    r"""
    Derive the complete released BoltzGen confidence tensor set.
    """

    batch_size = input.token_mask.shape[0]
    token_mask = input.token_mask.repeat_interleave(num_samples, dim=0).to(pae_logits)
    chain_index = input.chain_index.repeat_interleave(num_samples, dim=0)
    molecule_type = input.molecule_type.repeat_interleave(num_samples, dim=0)
    is_ligand_token = (molecule_type == 3).to(pae_logits)
    is_protein_token = (molecule_type == 0).to(pae_logits)

    plddt = plddt_from_logits(plddt_logits)
    if token_level_confidence:
        confidence_mask = token_mask
        mean_plddt = weighted_mean(plddt, confidence_mask, backend=backend)
        iplddt_weight = interface_confidence_weights(
            representative_distances,
            chain_index,
            is_ligand_token,
            confidence_mask,
            ligand_weight=20.0,
            interface_weight=10.0,
            background_weight=1.0,
        )
    else:
        atom_mask = input.atom_mask.repeat_interleave(num_samples, dim=0).to(plddt)
        confidence_mask = atom_mask
        mean_plddt = weighted_mean(plddt, confidence_mask, backend=backend)
        atom_to_token = input.atom_to_token.repeat_interleave(num_samples, dim=0).long()
        atom_molecule_type = molecule_type.gather(1, atom_to_token)
        atom_chain_index = chain_index.gather(1, atom_to_token)
        is_ligand_atom = atom_molecule_type == 3
        iplddt_weight = interface_confidence_weights(
            pairwise_distance(coordinates),
            atom_chain_index,
            is_ligand_atom,
            confidence_mask,
            ligand_weight=20.0,
            interface_weight=10.0,
            background_weight=1.0,
        )
    complex_iplddt = weighted_mean(plddt, iplddt_weight, backend=backend)

    pde = categorical_mean(pde_logits, end=32.0)
    pae = categorical_mean(pae_logits, end=32.0)
    pred_distogram_probability = pred_distogram_logits.softmax(dim=-1).repeat_interleave(
        num_samples,
        dim=0,
    )
    contact_probability = pred_distogram_probability[..., :20].sum(dim=-1)
    eye = torch.eye(token_mask.shape[-1], device=token_mask.device, dtype=torch.bool).unsqueeze(0)
    token_pair_mask = token_mask.bool().unsqueeze(-1) & token_mask.bool().unsqueeze(-2) & ~eye

    selected_ligand_mask = is_ligand_token.bool() & (chain_index == 1) & token_mask.bool()
    protein_pair_mask = token_pair_mask & is_protein_token.bool().unsqueeze(-1) & is_protein_token.bool().unsqueeze(-2)
    ligand_pair_mask = token_pair_mask & selected_ligand_mask.unsqueeze(-1) & selected_ligand_mask.unsqueeze(-2)
    interface_protein_mask = (
        (representative_distances < 8)
        & (is_protein_token.bool() & token_mask.bool()).unsqueeze(-1)
        & selected_ligand_mask.unsqueeze(-2)
    ).any(dim=-1)
    protein_ligand_interface_pair_mask = (
        token_pair_mask
        & interface_protein_mask.unsqueeze(-1)
        & selected_ligand_mask.unsqueeze(-2)
        & (representative_distances < 8)
    )

    def pae_average(mask: Tensor, *, contact_weighted: bool) -> Tensor:
        weight = contact_probability.masked_fill(~mask, 0.0) if contact_weighted else mask
        return weighted_mean(pae.flatten(-2), weight.flatten(-2), backend=backend)

    protein_pae_value_weighted = pae_average(protein_pair_mask, contact_weighted=True)
    protein_pae_value_unweighted = pae_average(protein_pair_mask, contact_weighted=False)
    ligand_pae_value_weighted = pae_average(ligand_pair_mask, contact_weighted=True)
    ligand_pae_value_unweighted = pae_average(ligand_pair_mask, contact_weighted=False)
    interface_pae_value_weighted = pae_average(protein_ligand_interface_pair_mask, contact_weighted=True)
    interface_pae_value_unweighted = pae_average(protein_ligand_interface_pair_mask, contact_weighted=False)

    contact_pair_weight = contact_probability.masked_fill(~token_pair_mask, 0.0)
    complex_pde = weighted_mean(pde.flatten(-2), contact_pair_weight.flatten(-2), backend=backend)
    interface_pair_weight = contact_pair_weight.masked_fill(chain_index.unsqueeze(-1) == chain_index.unsqueeze(-2), 0.0)
    complex_ipde = weighted_mean(pde.flatten(-2), interface_pair_weight.flatten(-2), backend=backend)

    design_mask = cast(Tensor, input.design_mask).bool()
    chain_design_mask = cast(Tensor, input.chain_design_mask).bool()
    has_design = design_mask.any(dim=-1, keepdim=True)
    pae_chain_design_mask = torch.where(has_design, chain_design_mask, design_mask)
    pae_target_mask = torch.where(has_design, ~chain_design_mask, design_mask)
    pae_chain_design_mask = pae_chain_design_mask.repeat_interleave(num_samples, dim=0)
    pae_target_mask = pae_target_mask.repeat_interleave(num_samples, dim=0)
    design_mask_repeated = design_mask.repeat_interleave(num_samples, dim=0)
    has_design_repeated = has_design.repeat_interleave(num_samples, dim=0).squeeze(-1)

    chain_design_target_mask = token_pair_mask & (
        (pae_chain_design_mask.unsqueeze(-1) & pae_target_mask.unsqueeze(-2))
        | (pae_target_mask.unsqueeze(-1) & pae_chain_design_mask.unsqueeze(-2))
    )
    design_target_mask = token_pair_mask & (
        (design_mask_repeated.unsqueeze(-1) & pae_target_mask.unsqueeze(-2))
        | (pae_target_mask.unsqueeze(-1) & design_mask_repeated.unsqueeze(-2))
    )
    interaction_pae = weighted_mean(pae.flatten(-2), design_target_mask.flatten(-2), backend=backend)
    min_interaction_pae = minimum_confidence_error(pae, chain_design_target_mask)
    min_design_to_target_pae = minimum_confidence_error(pae, design_target_mask)
    min_design_to_target_pae = torch.where(
        has_design_repeated,
        min_design_to_target_pae,
        min_design_to_target_pae.new_full(min_design_to_target_pae.shape, torch.nan),
    )

    tm_scores = _boltzgen_confidence_scope_scores(
        pae_logits,
        coordinates=coordinates,
        input=input,
        num_samples=num_samples,
        chain_design_mask=chain_design_mask,
        backend=backend,
    )
    ranking_tm = torch.where(
        torch.isclose(tm_scores["iptm"], torch.zeros_like(tm_scores["iptm"])),
        tm_scores["ptm"],
        tm_scores["iptm"],
    )
    confidence_score = (4 * (mean_plddt / 100.0) + ranking_tm) / 5

    def unflatten(value: Tensor) -> Tensor:
        return value.reshape(batch_size, num_samples, *value.shape[1:])

    return BoltzGenConfidenceOutput(
        pae_logits=unflatten(pae_logits),
        pde_logits=unflatten(pde_logits),
        plddt_logits=unflatten(plddt_logits),
        resolved_logits=unflatten(resolved_logits),
        pae=unflatten(pae),
        pde=unflatten(pde),
        plddt=unflatten(plddt),
        mean_plddt=unflatten(mean_plddt),
        complex_iplddt=unflatten(complex_iplddt),
        confidence_score=unflatten(confidence_score),
        complex_pde=unflatten(complex_pde),
        complex_ipde=unflatten(complex_ipde),
        interaction_pae=unflatten(interaction_pae),
        min_design_to_target_pae=unflatten(min_design_to_target_pae),
        min_interaction_pae=unflatten(min_interaction_pae),
        protein_pae_value_weighted=unflatten(protein_pae_value_weighted),
        ligand_pae_value_weighted=unflatten(ligand_pae_value_weighted),
        interface_pae_value_weighted=unflatten(interface_pae_value_weighted),
        protein_pae_value_unweighted=unflatten(protein_pae_value_unweighted),
        ligand_pae_value_unweighted=unflatten(ligand_pae_value_unweighted),
        interface_pae_value_unweighted=unflatten(interface_pae_value_unweighted),
        ptm=unflatten(tm_scores["ptm"]),
        iptm=unflatten(tm_scores["iptm"]),
        ligand_iptm=unflatten(tm_scores["ligand_iptm"]),
        protein_iptm=unflatten(tm_scores["protein_iptm"]),
        chain_pair_iptm=unflatten(tm_scores["chain_pair_iptm"]),
        design_to_target_iptm=unflatten(tm_scores["design_to_target_iptm"]),
        design_residue_iptm=unflatten(tm_scores["design_residue_iptm"]),
        design_iptm=unflatten(tm_scores["design_iptm"]),
        design_iiptm=unflatten(tm_scores["design_iiptm"]),
        target_ptm=unflatten(tm_scores["target_ptm"]),
        design_ptm=unflatten(tm_scores["design_ptm"]),
        design_ipsae_min=unflatten(tm_scores["design_ipsae_min"]),
        design_to_target_ipsae=unflatten(tm_scores["design_to_target_ipsae"]),
        target_to_design_ipsae=unflatten(tm_scores["target_to_design_ipsae"]),
        chain_pair_ipsae=unflatten(tm_scores["chain_pair_ipsae"]),
        **({"confidence_single": unflatten(confidence_single)} if confidence_single is not None else {}),
        **({"confidence_pair": unflatten(confidence_pair)} if confidence_pair is not None else {}),
        batch_size=[batch_size],
    )


def _boltzgen_confidence_scope_scores(
    pae_logits: Tensor,
    *,
    coordinates: Tensor,
    input: BoltzGenInput,
    num_samples: int,
    chain_design_mask: Tensor,
    backend: str = "auto",
) -> dict[str, Tensor]:
    r"""
    Prepare frame and design scopes and assemble their shared TM/iPSAE scores.
    """

    token_mask = input.token_mask.repeat_interleave(num_samples, dim=0).to(pae_logits)
    frame_mask = _boltzgen_predicted_frame_mask(
        coordinates,
        input=input,
        num_samples=num_samples,
    ).to(pae_logits)
    chain_index = input.chain_index.repeat_interleave(num_samples, dim=0)
    pair_mask_ptm = frame_mask.unsqueeze(-1) * token_mask.unsqueeze(-2) * token_mask.unsqueeze(-1)
    different_chain = chain_index.unsqueeze(-1) != chain_index.unsqueeze(-2)
    pair_mask_iptm = pair_mask_ptm * different_chain.to(pae_logits)

    num_bins = pae_logits.shape[-1]
    bin_width = 32.0 / num_bins
    pae_centers = torch.arange(
        start=0.5 * bin_width,
        end=32.0,
        step=bin_width,
        device=pae_logits.device,
        dtype=pae_logits.dtype,
    )
    num_tokens = token_mask.sum(dim=-1, keepdim=True)
    tm_expected = tm_weight_from_pae(
        pae_logits,
        num_tokens.unsqueeze(-1),
        bin_centers=pae_centers,
    )

    molecule_type = input.molecule_type.repeat_interleave(num_samples, dim=0)
    is_ligand = (molecule_type == 3).to(pae_logits)
    is_protein = (molecule_type == 0).to(pae_logits)
    ligand_iptm_mask = pair_mask_iptm * (
        is_ligand.unsqueeze(-1) * is_protein.unsqueeze(-2) + is_protein.unsqueeze(-1) * is_ligand.unsqueeze(-2)
    )
    protein_iptm_mask = pair_mask_iptm * is_protein.unsqueeze(-1) * is_protein.unsqueeze(-2)

    design_mask = cast(Tensor, input.design_mask).bool()
    has_design = design_mask.any(dim=-1, keepdim=True)
    chain_design_mask = torch.where(has_design, chain_design_mask, design_mask)
    target_mask = ~chain_design_mask
    chain_design_mask = chain_design_mask.repeat_interleave(num_samples, dim=0)
    target_mask = target_mask.repeat_interleave(num_samples, dim=0)
    design_mask = design_mask.repeat_interleave(num_samples, dim=0)

    design_iptm_mask = pair_mask_ptm * (
        (chain_design_mask.unsqueeze(-1) & target_mask.unsqueeze(-2))
        | (target_mask.unsqueeze(-1) & chain_design_mask.unsqueeze(-2))
    ).to(pae_logits)
    design_to_target_iptm_mask = pair_mask_ptm * (
        (design_mask.unsqueeze(-1) & target_mask.unsqueeze(-2))
        | (target_mask.unsqueeze(-1) & design_mask.unsqueeze(-2))
    ).to(pae_logits)
    design_residue_iptm_mask = (
        pair_mask_ptm
        * different_chain.to(pae_logits)
        * (design_mask.unsqueeze(-1).to(pae_logits) + design_mask.unsqueeze(-2).to(pae_logits)).clamp(max=1)
    )
    design_ptm_mask = pair_mask_ptm * (chain_design_mask.unsqueeze(-1) & chain_design_mask.unsqueeze(-2)).to(pae_logits)
    target_ptm_mask = pair_mask_ptm * (target_mask.unsqueeze(-1) & target_mask.unsqueeze(-2)).to(pae_logits)

    num_chains = int(input.chain_index.max().item()) + 1
    chain_pair_iptm = chain_pair_tm_scores(
        tm_expected, chain_index, range(num_chains), pair_weights=pair_mask_ptm, backend=backend
    )

    interaction_mask = _boltzgen_design_interaction_mask(
        coordinates,
        input=input,
        num_samples=num_samples,
        chain_design_mask=chain_design_mask,
    )
    design_iiptm_mask = pair_mask_ptm * (
        (interaction_mask.unsqueeze(-1) & target_mask.unsqueeze(-2))
        | (target_mask.unsqueeze(-1) & interaction_mask.unsqueeze(-2))
    ).to(pae_logits)

    expected_pae = categorical_expectation(pae_logits.softmax(dim=-1), pae_centers)
    design_to_target_ipsae = directional_ipsae(
        expected_pae,
        source_mask=chain_design_mask,
        target_mask=target_mask,
        frame_mask=frame_mask,
        token_mask=token_mask,
        backend=backend,
    )
    target_to_design_ipsae = directional_ipsae(
        expected_pae,
        source_mask=target_mask,
        target_mask=chain_design_mask,
        frame_mask=frame_mask,
        token_mask=token_mask,
        backend=backend,
    )
    chain_pair_ipsae = tm_expected.new_zeros(tm_expected.shape[0], num_chains, num_chains)
    for row in range(num_chains):
        for column in range(num_chains):
            if row != column:
                chain_pair_ipsae[:, row, column] = directional_ipsae(
                    expected_pae,
                    source_mask=chain_index == row,
                    target_mask=chain_index == column,
                    frame_mask=frame_mask,
                    token_mask=token_mask,
                    backend=backend,
                )

    return {
        "ptm": predicted_tm_score(tm_expected, pair_weights=pair_mask_ptm, backend=backend),
        "iptm": predicted_tm_score(tm_expected, pair_weights=pair_mask_iptm, backend=backend),
        "ligand_iptm": predicted_tm_score(tm_expected, pair_weights=ligand_iptm_mask, backend=backend),
        "protein_iptm": predicted_tm_score(tm_expected, pair_weights=protein_iptm_mask, backend=backend),
        "chain_pair_iptm": chain_pair_iptm,
        "design_to_target_iptm": predicted_tm_score(
            tm_expected, pair_weights=design_to_target_iptm_mask, backend=backend
        ),
        "design_residue_iptm": predicted_tm_score(tm_expected, pair_weights=design_residue_iptm_mask, backend=backend),
        "design_iptm": predicted_tm_score(tm_expected, pair_weights=design_iptm_mask, backend=backend),
        "design_iiptm": predicted_tm_score(tm_expected, pair_weights=design_iiptm_mask, backend=backend),
        "target_ptm": predicted_tm_score(tm_expected, pair_weights=target_ptm_mask, backend=backend),
        "design_ptm": predicted_tm_score(tm_expected, pair_weights=design_ptm_mask, backend=backend),
        "design_ipsae_min": torch.minimum(design_to_target_ipsae, target_to_design_ipsae),
        "design_to_target_ipsae": design_to_target_ipsae,
        "target_to_design_ipsae": target_to_design_ipsae,
        "chain_pair_ipsae": chain_pair_ipsae,
    }


def _boltzgen_predicted_frame_mask(
    coordinates: Tensor,
    *,
    input: BoltzGenInput,
    num_samples: int,
) -> Tensor:
    return boltzgen_coordinate_frames(
        coordinates,
        input=input,
        num_samples=num_samples,
    )[1]


def boltzgen_coordinate_frames(
    coordinates: Tensor,
    *,
    input: BoltzGenInput,
    num_samples: int,
    atom_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    r"""
    Construct coordinate-dependent token frames and their geometry mask.

    Polymer frame indices come from the Processor. Non-polymer atom-token
    frames use the two nearest eligible atoms around each center atom, matching
    the confidence metric and PAE-target equations.
    """

    active = input.atom_mask.bool().repeat_interleave(num_samples, dim=0)
    if atom_mask is not None:
        active = active & atom_mask.bool()
    coordinates = project_model_coordinates(coordinates, active, feature="BoltzGen frame coordinates")
    if input.frame_atom_index is None:
        fail("value_invalid", RequestRef("frame_atom_index", "None"))
    frame_index = input.frame_atom_index.repeat_interleave(num_samples, dim=0).long().clone()
    chain_index = input.chain_index
    molecule_type = input.molecule_type
    atom_to_token = input.atom_to_token.long()
    atom_chain_index = chain_index.gather(1, atom_to_token)
    structural_atom_mask = input.atom_mask.bool()
    if atom_mask is None:
        atom_mask = structural_atom_mask.repeat_interleave(num_samples, dim=0)
    else:
        atom_mask = atom_mask.bool()

    for batch_index in range(input.token_mask.shape[0]):
        ligand_chains = torch.unique(
            chain_index[batch_index][input.token_mask[batch_index].bool() & (molecule_type[batch_index] == 3)]
        ).tolist()
        for chain in ligand_chains:
            chain_tokens = torch.nonzero(
                input.token_mask[batch_index].bool()
                & (molecule_type[batch_index] == 3)
                & (chain_index[batch_index] == chain),
                as_tuple=False,
            ).flatten()
            chain_atoms = torch.nonzero(
                structural_atom_mask[batch_index] & (atom_chain_index[batch_index] == chain),
                as_tuple=False,
            ).flatten()
            if len(chain_atoms) < 3:
                continue
            for sample_index in range(num_samples):
                row = batch_index * num_samples + sample_index
                centers = input.token_center_atom_index[batch_index, chain_tokens].long()
                distances = pairwise_distance(coordinates[row, centers], coordinates[row, chain_atoms])
                resolved_pair = atom_mask[row, centers, None] & atom_mask[row, chain_atoms][None, :]
                resolved_pair &= centers[:, None] != chain_atoms[None, :]
                nearest = distances.masked_fill(~resolved_pair, torch.inf).argsort(dim=-1)[:, :2]
                frame_index[row, chain_tokens] = torch.stack(
                    (chain_atoms[nearest[:, 0]], centers, chain_atoms[nearest[:, 1]]), dim=-1
                )

    flat_index = frame_index.reshape(frame_index.shape[0], -1)
    points, point_mask = gather_observed_coordinates(
        coordinates, atom_mask, flat_index, torch.ones_like(flat_index, dtype=torch.bool)
    )
    frames = points.reshape(*frame_index.shape, 3)
    frame_resolved = point_mask.reshape(frame_index.shape).all(dim=-1)
    first = frames[..., 1, :] - frames[..., 0, :]
    second = frames[..., 1, :] - frames[..., 2, :]
    first_norm = torch.linalg.norm(first, dim=-1)
    second_norm = torch.linalg.norm(second, dim=-1)
    first_direction = first / (first_norm.unsqueeze(-1) + 1e-6)
    second_direction = second / (second_norm.unsqueeze(-1) + 1e-6)
    cosine = (first_direction * second_direction).sum(dim=-1)
    valid_geometry = (cosine.abs() < 0.9063) & (first_norm > 1e-2) & (second_norm > 1e-2)
    valid_geometry = valid_geometry & frame_resolved & input.token_mask.repeat_interleave(num_samples, dim=0).bool()
    return frame_index.clamp_min(0), valid_geometry


def _boltzgen_design_interaction_mask(
    coordinates: Tensor,
    *,
    input: BoltzGenInput,
    num_samples: int,
    chain_design_mask: Tensor,
) -> Tensor:
    atom_to_token = input.atom_to_token.repeat_interleave(num_samples, dim=0).long()
    atom_mask = input.atom_mask.repeat_interleave(num_samples, dim=0).bool()
    atom_chain_design_mask = chain_design_mask.gather(1, atom_to_token) & atom_mask
    interaction_mask = chain_design_mask.new_zeros(chain_design_mask.shape)
    for sample_index in range(coordinates.shape[0]):
        design_atoms = atom_chain_design_mask[sample_index]
        target_atoms = atom_mask[sample_index] & ~design_atoms
        if design_atoms.any() and target_atoms.any():
            interacting_atoms = (
                pairwise_distance(
                    coordinates[sample_index, design_atoms],
                    coordinates[sample_index, target_atoms],
                ).amin(dim=-1)
                < 8
            )
            interacting_tokens = atom_to_token[sample_index, design_atoms][interacting_atoms]
            interaction_mask[sample_index].scatter_(
                0,
                interacting_tokens,
                torch.ones_like(interacting_tokens, dtype=torch.bool),
            )
    return interaction_mask


def _dilate_time(ts: Tensor, *, start: float, end: float, dilation: float) -> Tensor:
    r"""
    Piecewise remapping used by BoltzGen's dilated design-inference noise schedule.
    """

    x = end - start
    lower = start
    upper = 1 - end
    if (dilation - 1) * x > lower + upper:
        fail("value_invalid", RequestRef("time_dilation", str(dilation)), interval=(start, end))

    if dilation == 1:
        return ts

    inv_dilation = 1 / dilation
    ratio = (lower + upper + (1 - dilation) * x) / (lower + upper)
    if ratio == 0:
        middle = ts * inv_dilation + lower
        return torch.where(ts == 0, 0.0, torch.where(ts == 1, 1.0, middle))

    inv_ratio = 1 / ratio
    lower_prime = lower * ratio
    x_prime = x * dilation

    lower_third = ts * inv_ratio
    middle_third = (ts - lower_prime) * inv_dilation + lower
    upper_third = 1 + (ts - 1) * inv_ratio
    return torch.where(
        ts < lower_prime,
        lower_third,
        torch.where(ts < lower_prime + x_prime, middle_third, upper_third),
    )


def _center_random_augmentation(
    coords: Tensor,
    atom_mask: Tensor,
    *,
    augmentation: bool = True,
    translation_scale: float = 1.0,
    generator: torch.Generator | None = None,
) -> Tensor:
    r"""
    Center masked coordinates and apply a random rotation and translation.
    """

    while atom_mask.dim() < coords.dim() - 1:
        atom_mask = atom_mask.unsqueeze(-2)
    atom_mask = atom_mask.bool()
    coords = project_model_coordinates(
        coords, atom_mask.expand(coords.shape[:-1]), feature="BoltzGen augmentation coordinates"
    )
    centroid = weighted_mean(coords.transpose(-1, -2), atom_mask.unsqueeze(-2)).unsqueeze(-2)
    centroid = torch.where(atom_mask.any(-1, keepdim=True).unsqueeze(-1), centroid, 0)
    centered = coords - centroid
    if not augmentation:
        return centered
    batch_shape = coords.shape[:-2]
    rotation = uniform_random_rotation(
        math.prod(batch_shape),
        device=coords.device,
        dtype=coords.dtype,
        generator=generator,
    ).reshape(*batch_shape, 3, 3)
    rotated = rotate_vectors(centered, rotation)
    translation = (
        torch.randn(*batch_shape, 1, 3, device=coords.device, dtype=coords.dtype, generator=generator)
        * translation_scale
    )
    return rotated + translation


def _gather_backbone4(
    token_to_bb4_atoms: Tensor,
    coords: Tensor,
    atom_coordinate_mask: Tensor,
    batch_size: int,
    num_tokens: int,
) -> tuple[Tensor, Tensor]:
    r"""
    Gather BB4 coordinates ``(B, N_token, 4, 3)`` and availability ``(B, N_token, 4)``.

    Unavailable coordinate payloads are zeroed before geometric arithmetic.
    """
    bb4_index = token_to_bb4_atoms.reshape(batch_size, num_tokens * 4)
    coords = coords.reshape(batch_size, -1, 3)
    backbone, backbone_mask = gather_observed_coordinates(
        coords, atom_coordinate_mask.bool(), bb4_index, torch.ones_like(bb4_index, dtype=torch.bool)
    )
    return backbone.reshape(batch_size, num_tokens, 4, 3), backbone_mask.reshape(batch_size, num_tokens, 4)


def _zero_initialize_boltzgen_updates(module: nn.Module) -> None:
    r"""
    Apply the BoltzGen zero-init scheme to every final-layer / gating / conditioning weight under ``module``.

    Zeroing each residual branch's final projection (and biasing gate projections negative) starts every
    block as a near-identity.

    Args:
        module: Root module to walk; every matching child is zero-initialized in place.
    """

    for child in module.modules():
        if isinstance(child, TrunkToAtomConditioning):
            nn.init.zeros_(child.single_projection.weight)
            nn.init.zeros_(child.pair_projection.weight)
        elif isinstance(child, AtomPairEncoder):
            nn.init.zeros_(child.left_atom_projection.weight)
            nn.init.zeros_(child.right_atom_projection.weight)
            nn.init.zeros_(child.feed_forward.output_projection.weight)
        elif isinstance(child, CoordinateAtomToTokenEncoder):
            nn.init.zeros_(child.noisy_position_projection.weight)
        elif isinstance(child, TokenToAtomDecoder):
            nn.init.zeros_(child.token_projection.weight)
            nn.init.zeros_(child.position_update_projection.weight)
        elif isinstance(child, _BoltzGenDenoiser):
            nn.init.zeros_(child.single_to_token_projection.weight)
        elif isinstance(child, BoltzGenBidirectionalTriangleMultiplication):
            nn.init.zeros_(child.input_gate_projection.weight)
            nn.init.zeros_(child.output_projection.weight)
            nn.init.zeros_(child.output_gate_projection.weight)
        elif isinstance(child, Transition):
            nn.init.zeros_(child.feed_forward.output_projection.weight)
        elif isinstance(child, MultiheadAttention):
            nn.init.zeros_(child.projection.output_projection.weight)
        elif isinstance(child, ConditionedTransformerEncoderBlock):
            nn.init.zeros_(child.output_gate_projection.weight)
            if child.output_gate_projection.bias is not None:
                nn.init.constant_(child.output_gate_projection.bias, -2.0)
        elif isinstance(child, (ConditionedTransition, TripleProductConditionedTransition)):
            nn.init.zeros_(child.conditioning_gate_projection.weight)
            if child.conditioning_gate_projection.bias is not None:
                nn.init.constant_(child.conditioning_gate_projection.bias, -2.0)


def _softmax_dropout(logits: Tensor, index: Tensor, num_groups: int, probability: float) -> Tensor:
    r"""
    Apply the rejection-sampled attention-logit dropout.
    """

    originally_nonempty = indexed_sum(torch.ones_like(logits, dtype=torch.int64), index, num_groups=num_groups).gt(0)
    for _ in range(10):
        dropped = torch.rand_like(logits) < probability
        kept = indexed_sum((~dropped).to(torch.int64), index, num_groups=num_groups)
        if (kept.gt(0) | ~originally_nonempty).all():
            return logits.masked_fill(dropped, -torch.inf)
    raise RuntimeError("attention dropout failed to keep an incoming edge for every destination and head")


# ===========================================================================
# BoltzGen design conditions, typed inputs, structure targets, and outputs.
# ===========================================================================


@dataclass(frozen=True)
class BoltzGenDesignMask:
    r"""
    Select existing residue positions for redesign.

    Attributes:
        positions: Residue occurrences whose identities may change.
    """

    positions: tuple[ComponentRef, ...]


@dataclass(frozen=True)
class BoltzGenBinding:
    r"""
    Positive and negative residue-level binding conditions.

    Attributes:
        binding: Residues encouraged to participate in binding.
        not_binding: Residues discouraged from participating in binding.
    """

    binding: Hotspot = Hotspot(())
    not_binding: tuple[ComponentRef, ...] = ()


@dataclass(frozen=True)
class BoltzGenSecondaryStructure:
    r"""
    Assign secondary-structure conditions to residue positions.

    Attributes:
        assignments: ``(residue, label)`` pairs, where the label is ``"H"``
            (helix), ``"E"`` (strand), or ``"L"`` (loop).
    """

    assignments: tuple[tuple[ComponentRef, Literal["H", "E", "L"]], ...]


@dataclass(frozen=True)
class BoltzGenStructureGroup:
    r"""
    Residues whose relative geometry remains visible during design.

    Distinct nonzero groups move independently; group zero denotes unconstrained residues.

    Attributes:
        members: Residues belonging to the group.
        group_id: Nonnegative group identifier.
    """

    members: tuple[ComponentRef, ...]
    group_id: int

    def __post_init__(self) -> None:
        if self.group_id < 0:
            fail("value_invalid", RequestRef("BoltzGenStructureGroup", str(self.group_id)))


@dataclass(frozen=True)
class BoltzGenContact:
    r"""
    A BoltzGen pairwise proximity condition between two endpoint sets.

    Attributes:
        side_a: Endpoints on one side of the requested contact.
        side_b: Endpoints on the other side of the requested contact.
        cutoff_angstrom: Maximum contact distance in angstroms.
        kind: Directional pocket/binder meaning across the declared sides, or
            ``"contact"`` for symmetric proximity.
    """

    side_a: tuple[Endpoint, ...]
    side_b: tuple[Endpoint, ...]
    cutoff_angstrom: float
    kind: Literal["contact", "pocket>binder", "binder>pocket"] = "contact"


@dataclass(frozen=True)
class BoltzGenSymmetry:
    r"""
    Tie corresponding positions across equal-length symmetry units.

    Attributes:
        units: Residue groups representing symmetric copies. Corresponding offsets
            are tied.
    """

    units: tuple[tuple[ComponentRef, ...], ...]


#: Condition record accepted by BoltzGen structure-design processors and pipelines.
BoltzGenSchema = (
    BoltzGenDesignMask
    | BoltzGenBinding
    | BoltzGenSecondaryStructure
    | BoltzGenStructureGroup
    | BoltzGenContact
    | SymbolicBond
    | BoltzGenSymmetry
)


@tensorclass
class BoltzGenStructureTarget:
    r"""
    Structure-training labels aligned to one [`BoltzGenInput`][].

    Attributes:
        coordinates: Complete-assembly target coordinates with shape
            ``(B, R, A_system, 3)``.
        coordinate_mask: Resolved complete-assembly mask with shape
            ``(B, A_system)``.
        coordinate_atom_index: Model-row to coordinate-source mapping with shape
            ``(B, M)`` and ``-1`` padding.
        chain_span: Half-open canonical-atom chain spans with shape ``(B, C, 2)``.
        chain_equivalence_index: Entity-and-layout equivalence ids with shape
            ``(B, C)``.
        symmetry_atom_index: Current-axis atoms in each local symmetry group
            with shape ``(B, G, W)`` and ``-1`` padding.
        symmetry_permutation: Complete carried permutations with shape
            ``(B, G, P, W)`` and ``-1`` padding.
        residue_type: Unmasked target residue distributions with shape
            ``(B, N, V)``.
        b_factor: B factor of each token's representative atom with shape ``(B, N)``,
            matching the token axis the B-factor head reads.
        b_factor_mask: Deposited-B-factor mask with shape ``(B, N)``.
    """

    coordinates: Tensor
    coordinate_mask: Tensor
    coordinate_atom_index: Tensor
    chain_span: Tensor
    chain_equivalence_index: Tensor
    symmetry_atom_index: Tensor
    symmetry_permutation: Tensor
    residue_type: Tensor
    b_factor: Tensor
    b_factor_mask: Tensor


@tensorclass
class BoltzGenInput:
    r"""
    Input tensors consumed by BoltzGen design models.

    Attributes:
        atom_index: Model-row to canonical-atom identity mapping. Synthetic
            side-chain rows contain ``-1``.
        bond_atom_index: Inter-component model-row bond endpoints.
        bond_mask: Valid inter-component bond mask.
        token_mask: Boolean token validity mask with shape ``(B, N_token)``.
        residue_index: Chain-local residue index used by relative-position features.
        chain_index: Chain ids mapped to contiguous integer indices.
        entity_index: Entity ids shared by chains with the same sequence.
        symmetry_index: Symmetry-copy ids for relative-position features.
        token_index: Token ids in structure order.
        atom_features: Per-atom reference features.
        atom_to_token: Token index for each atom.
        atom_mask: Boolean model-row existence mask, independent of coordinate availability.
        reference_positions: Centered reference-conformer atom coords.
        reference_space_id: Reference-conformer group id for each atom.
        atom_token_index: Per-token atom offset for each atom.
        token_center_atom_index: Center atom index for each token.
        residue_type_features: Query residue type one-hot features.
        msa_profile: Single-sequence MSA profile features.
        deletion_mean: Per-token deletion mean features.
        token_bonds: Token-level covalent adjacency ``(B, N, N, 1)`` (trailing singleton per the featurizer).
        type_bonds: Token-level bond type ids.
        contact_conditioning: Pairwise contact-conditioning class one-hot features.
        contact_threshold: Pairwise contact threshold values.
        molecule_type: Molecular category ids in the model vocabulary.
        center_coordinates: Per-token representative atom coords.
        token_to_bb4_atoms: Per-token ``(N, CA, C, O)`` atom indices.
        atom_coordinates: Atom coordinates used by geometric conditioning features; never a training-label fallback.
        atom_coordinate_mask: Required boolean coordinate-availability mask with shape ``(B, A)``;
            false rows carry no geometric evidence, even when the atom exists.
        token_distance_mask: Pairwise token-distance conditioning visibility, independent of coordinate availability.
        frame_atom_index: Optional frame atom indices for frame-aligned losses.
        method_feature: Optional experimental-method ids.
        modified: Optional modified-residue flags.
        cyclic_period: Optional cyclic-position periods.
        msa_tokens: Optional integer MSA token ids.
        has_deletion: Optional MSA deletion flags.
        deletion_value: Optional MSA deletion values.
        msa_paired: Optional paired-MSA flags.
        msa_mask: Optional MSA validity mask.
        template_residue_type_features: Optional template residue-type features.
        template_frame_rotations: Optional template frame rotations.
        template_frame_translations: Optional template frame translations.
        template_frame_mask: Optional template frame validity mask.
        template_pseudo_beta_positions: Optional template pseudo-beta positions.
        template_alpha_carbon_positions: Optional template alpha-carbon positions.
        template_pseudo_beta_mask: Optional template pseudo-beta validity mask.
        template_mask: Optional template validity mask.
        template_visibility_index: Optional int64 physical placement-frame IDs; -1 is unavailable.
        binding_type: Optional binding conditioning ids.
        design_mask: Optional token mask selecting designable positions.
        chain_design_mask: Optional token mask selecting the complete covalently
            connected design component. Confidence inputs must provide it.
        ss_type: Optional secondary-structure conditioning ids.
    """

    atom_index: Tensor
    bond_atom_index: Tensor
    bond_mask: Tensor
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
    atom_token_index: Tensor
    token_center_atom_index: Tensor
    residue_type_features: Tensor
    msa_profile: Tensor
    deletion_mean: Tensor
    token_bonds: Tensor
    type_bonds: Tensor
    contact_conditioning: Tensor
    contact_threshold: Tensor
    molecule_type: Tensor
    center_coordinates: Tensor
    token_to_bb4_atoms: Tensor
    atom_coordinates: Tensor
    atom_coordinate_mask: Tensor
    token_distance_mask: Tensor
    frame_atom_index: Tensor | None = None
    method_feature: Tensor | None = None
    modified: Tensor | None = None
    cyclic_period: Tensor | None = None
    msa_tokens: Tensor | None = None
    has_deletion: Tensor | None = None
    deletion_value: Tensor | None = None
    msa_paired: Tensor | None = None
    msa_mask: Tensor | None = None
    template_residue_type_features: Tensor | None = None
    template_frame_rotations: Tensor | None = None
    template_frame_translations: Tensor | None = None
    template_frame_mask: Tensor | None = None
    template_pseudo_beta_positions: Tensor | None = None
    template_alpha_carbon_positions: Tensor | None = None
    template_pseudo_beta_mask: Tensor | None = None
    template_mask: Tensor | None = None
    template_center_mask: Tensor | None = None
    template_visibility_index: Tensor | None = None
    binding_type: Tensor | None = None
    design_mask: Tensor | None = None
    chain_design_mask: Tensor | None = None
    ss_type: Tensor | None = None


@tensorclass
class BoltzGenInverseFoldingInput:
    r"""
    Tensor and canonical-graph input for BoltzGen inverse folding.

    ``B`` is batch size, ``N`` token count, and ``A`` atom count.

    Attributes:
        atom_index: Model-row to canonical-atom identity mapping.
        native_token_index: Prepared unmasked native class with shape ``(B, N)``.
            Sampling that same class requests retention of the actual source graph.
        token_mask: Valid-token mask with shape ``(B, N)``.
        residue_index: Chain-local residue indices with shape ``(B, N)``.
        chain_index: Molecular-instance indices with shape ``(B, N)``.
        atom_features: Reference atom features with shape ``(B, A, C_atom)``.
        atom_to_token: Token index for each atom with shape ``(B, A)``.
        atom_mask: Boolean model-row existence mask with shape ``(B, A)``,
            independent of coordinate availability.
        reference_positions: Reference conformer coordinates with shape
            ``(B, A, 3)`` in angstroms.
        reference_space_id: Reference conformer group IDs with shape ``(B, A)``.
        residue_type_features: Residue-type features with shape ``(B, N, C_residue)``.
        msa_profile: Per-token MSA profile with shape ``(B, N, C_msa)``.
        deletion_mean: Per-token mean deletion values with shape ``(B, N)``.
        token_bonds: Token-level bond adjacency with shape ``(B, N, N, 1)``.
        type_bonds: Token-pair bond type IDs with shape ``(B, N, N)``.
        molecule_type: Molecular category IDs with shape ``(B, N)``.
        center_coordinates: Representative token coordinates with shape ``(B, N, 3)``.
        token_to_bb4_atoms: Backbone atom indices with shape ``(B, N, 4)``.
        atom_coordinates: Atom coordinates with shape ``(B, A, 3)`` in angstroms.
        atom_coordinate_mask: Required boolean coordinate-availability mask with shape ``(B, A)``;
            false rows carry no geometric evidence, even when the atom exists.
        token_resolved_mask: Mask selecting tokens whose representative atom is
            resolved in the fixed structure.
        method_feature: Optional experimental-method features.
        modified: Optional modified-residue flags.
        cyclic_period: Optional cyclic-position periods.
        design_mask: Mask selecting positions to redesign.
        allowed_token_mask: Allowed canonical-class mask with shape ``(B, N, V)``.
        symmetric_group: Sequence-symmetry group IDs with shape ``(B, N)``;
            zero means untied.
    """

    atom_index: Tensor
    native_token_index: Tensor
    token_mask: Tensor
    residue_index: Tensor
    chain_index: Tensor
    atom_features: Tensor
    atom_to_token: Tensor
    atom_mask: Tensor
    reference_positions: Tensor
    reference_space_id: Tensor
    residue_type_features: Tensor
    msa_profile: Tensor
    deletion_mean: Tensor
    token_bonds: Tensor
    type_bonds: Tensor
    molecule_type: Tensor
    center_coordinates: Tensor
    token_to_bb4_atoms: Tensor
    atom_coordinates: Tensor
    atom_coordinate_mask: Tensor
    token_resolved_mask: Tensor
    design_mask: Tensor
    allowed_token_mask: Tensor
    symmetric_group: Tensor
    method_feature: Tensor | None = None
    modified: Tensor | None = None
    cyclic_period: Tensor | None = None


class _BoltzGenTrainingOutput(NamedTuple):
    coordinates: Tensor
    distogram_logits: Tensor
    b_factor_logits: Tensor
    augmented_target_coordinates: Tensor
    noise_level: Tensor
    residue_type_logits: Tensor | None


class _BoltzGenTrunkOutput(NamedTuple):
    r"""
    Trunk representations consumed by the diffusion and confidence heads.
    """

    single_inputs: Tensor
    single: Tensor
    pair: Tensor
    relative_position_encoding: Tensor
    distogram_logits: Tensor


class BoltzGenOutput(TensorCollection):
    r"""
    Predictions from one BoltzGen training denoising pass.

    Attributes:
        coordinates: Denoised atom coordinates with shape ``(B, S, A, 3)`` in
            angstroms.
        distogram_logits: Token-pair distance logits with shape ``(B, N, N, D)``.
        b_factor_logits: Per-token B-factor logits with shape ``(B, N, D_b)``.
        augmented_target_coordinates: Training target coordinates after
            rigid augmentation, with shape ``(B, S, A, 3)`` in angstroms.
        noise_level: Sampled diffusion noise levels with shape ``(B, S)``.
    """

    coordinates: Tensor
    distogram_logits: Tensor
    b_factor_logits: Tensor
    augmented_target_coordinates: Tensor
    noise_level: Tensor


class BoltzGenResidueTypeOutput(BoltzGenOutput):
    r"""
    BoltzGen training predictions with required residue-type logits.

    Attributes:
        residue_type_logits: Residue-type logits with shape ``(B, S, N, V)``.
    """

    residue_type_logits: Tensor


class BoltzGenConfidenceOutput(TensorCollection):
    r"""
    Complete raw and derived outputs from the BoltzGen confidence equation.

    Leading dimensions are ``(B, S)``. ``plddt_logits``, ``resolved_logits``,
    and ``plddt`` use the token axis when token-level confidence is configured
    and the realized atom axis for the released 23-slot atom projection.
    Binary ``resolved_logits`` use ``[unresolved, resolved]`` class order.
    Chain-pair tensors use dense zero-based input chain indices.
    """

    pae_logits: Tensor
    pde_logits: Tensor
    plddt_logits: Tensor
    resolved_logits: Tensor
    pae: Tensor
    pde: Tensor
    plddt: Tensor
    mean_plddt: Tensor
    complex_iplddt: Tensor
    confidence_score: Tensor
    complex_pde: Tensor
    complex_ipde: Tensor
    interaction_pae: Tensor
    min_design_to_target_pae: Tensor
    min_interaction_pae: Tensor
    protein_pae_value_weighted: Tensor
    ligand_pae_value_weighted: Tensor
    interface_pae_value_weighted: Tensor
    protein_pae_value_unweighted: Tensor
    ligand_pae_value_unweighted: Tensor
    interface_pae_value_unweighted: Tensor
    ptm: Tensor
    iptm: Tensor
    ligand_iptm: Tensor
    protein_iptm: Tensor
    chain_pair_iptm: Tensor
    design_to_target_iptm: Tensor
    design_residue_iptm: Tensor
    design_iptm: Tensor
    design_iiptm: Tensor
    target_ptm: Tensor
    design_ptm: Tensor
    design_ipsae_min: Tensor
    design_to_target_ipsae: Tensor
    target_to_design_ipsae: Tensor
    chain_pair_ipsae: Tensor
    confidence_single: OptionalTensor
    confidence_pair: OptionalTensor


class BoltzGenSamplingOutput(TensorCollection):
    r"""
    Predictions from one BoltzGen reverse-diffusion sampling pass.

    Attributes:
        coordinates: Sampled atom coordinates with shape ``(B, S, A, 3)`` in
            angstroms.
        distogram_logits: Token-pair distance logits with shape ``(B, N, N, D)``.
        b_factor_logits: Per-token B-factor logits with shape ``(B, N, D_b)``.
        coordinate_trajectory: Initial/current coordinate states with shape
            ``(B, S, K + 1, A, 3)``, when requested.
        denoised_coordinate_trajectory: Per-step denoised ``x0`` predictions
            with shape ``(B, S, K, A, 3)``, when requested.
    """

    coordinates: Tensor
    distogram_logits: Tensor
    b_factor_logits: Tensor
    coordinate_trajectory: OptionalTensor
    denoised_coordinate_trajectory: OptionalTensor


class BoltzGenResidueTypeSamplingOutput(BoltzGenSamplingOutput):
    r"""Reverse-diffusion samples with required final residue-type logits."""

    residue_type_logits: Tensor


class BoltzGenSamplingWithConfidenceOutput(TensorCollection):
    r"""
    Coordinate samples and confidence values produced from one trunk execution.
    """

    coordinates: Tensor
    distogram_logits: Tensor
    b_factor_logits: Tensor
    confidence: BoltzGenConfidenceOutput
    coordinate_trajectory: OptionalTensor
    denoised_coordinate_trajectory: OptionalTensor


class BoltzGenInverseFoldingOutput(TensorCollection):
    r"""
    Logit predictions from one BoltzGen inverse-folding forward pass.

    Attributes:
        logits: Per-token amino-acid logits with shape ``(B, N, V)``.
    """

    logits: Tensor


class BoltzGenInverseFoldingDesignOutput(TensorCollection):
    r"""
    Predictions from one BoltzGen inverse-folding design pass.

    Attributes:
        logits: Per-token amino-acid logits with shape ``(B, N, V)``.
        sequence_tokens: Sampled amino-acid token IDs with shape ``(B, N)``.
    """

    logits: Tensor
    sequence_tokens: Tensor


_BoltzGenInputT = TypeVar("_BoltzGenInputT", BoltzGenInput, BoltzGenInverseFoldingInput)


def normalize_boltzgen_input(
    input: _BoltzGenInputT,
    *,
    dtype: torch.dtype | None = None,
) -> _BoltzGenInputT:
    """Normalize one family's existing carrier at every public numeric boundary.

    A valid reference row has a complete declared model conformer. Source
    coordinate availability is independent; synthetic atom14 rows remain valid
    model rows while their canonical atom indices are -1.
    """
    result = cast(_BoltzGenInputT, cast(TensorClass, input).clone(recurse=False))
    atoms, tokens = input.atom_mask.bool(), input.token_mask.bool()
    observed = input.atom_coordinate_mask.bool() & atoms
    result.reference_positions = project_model_coordinates(
        input.reference_positions, atoms, dtype=dtype, feature="BoltzGen reference positions"
    )
    result.atom_coordinates = project_model_coordinates(
        input.atom_coordinates, observed, dtype=dtype, feature="BoltzGen observed coordinates"
    )
    result.atom_features = project_model_features(
        input.atom_features, atoms[..., None], dtype=dtype, feature="BoltzGen atom features"
    )
    center_index = (
        input.token_center_atom_index if isinstance(input, BoltzGenInput) else input.token_to_bb4_atoms[..., 1]
    )
    _, centers = gather_observed_coordinates(input.atom_coordinates, observed, center_index, tokens)
    result.center_coordinates = project_model_coordinates(
        input.center_coordinates, centers, dtype=dtype, feature="BoltzGen center coordinates"
    )
    for name in ("residue_type_features", "msa_profile", "deletion_mean", "cyclic_period"):
        value = getattr(input, name)
        if value is not None:
            mask = tokens if value.ndim == tokens.ndim else tokens[..., None]
            setattr(result, name, project_model_features(value, mask, dtype=dtype, feature=f"BoltzGen {name}"))
    pairs = tokens[..., :, None] & tokens[..., None, :]
    result.token_bonds = project_model_features(
        input.token_bonds, pairs[..., None], dtype=dtype, feature="BoltzGen token relations"
    )
    if isinstance(input, BoltzGenInverseFoldingInput):
        cast(BoltzGenInverseFoldingInput, result).token_resolved_mask = input.token_resolved_mask.bool() & centers
    if isinstance(input, BoltzGenInput):
        for name in ("contact_conditioning", "contact_threshold"):
            value = getattr(input, name)
            mask = pairs if value.ndim == pairs.ndim else pairs[..., None]
            setattr(result, name, project_model_features(value, mask, dtype=dtype, feature=f"BoltzGen {name}"))
        if input.msa_tokens is not None:
            msa_available = tokens.unsqueeze(-2) & torch.ones_like(input.msa_tokens, dtype=torch.bool)
            if input.msa_mask is not None:
                msa_available = msa_available & input.msa_mask.bool()
            regular = cast(BoltzGenInput, result)
            regular.msa_mask = msa_available
            regular.msa_tokens = project_model_categories(
                input.msa_tokens, msa_available, num_classes=len(TOKENS), feature="BoltzGen MSA tokens"
            )
            for name in ("has_deletion", "deletion_value", "msa_paired"):
                value = getattr(input, name)
                if value is not None:
                    if not value.is_floating_point():
                        value = torch.where(msa_available, value, 0).to(torch.float64)
                    setattr(
                        result,
                        name,
                        project_model_features(value, msa_available, dtype=dtype, feature=f"BoltzGen {name}"),
                    )
        if input.template_residue_type_features is not None:
            present = cast(Tensor, input.template_mask).bool() & tokens[:, None, :]
            frames = cast(Tensor, input.template_frame_mask).bool() & present
            if input.template_center_mask is None:
                fail("value_invalid", RequestRef("template_center_mask", "None"))
            centers = input.template_center_mask.bool() & present
            beta = cast(Tensor, input.template_pseudo_beta_mask).bool() & present
            for name, mask in (
                ("template_frame_translations", frames),
                ("template_alpha_carbon_positions", centers),
                ("template_pseudo_beta_positions", beta),
            ):
                setattr(
                    result,
                    name,
                    project_model_coordinates(
                        cast(Tensor, getattr(input, name)), mask, dtype=dtype, feature=f"BoltzGen {name}"
                    ),
                )
            cast(BoltzGenInput, result).template_frame_rotations = project_model_features(
                cast(Tensor, input.template_frame_rotations),
                frames[..., None, None],
                dtype=dtype,
                feature="BoltzGen template frames",
            )
            classes = input.template_residue_type_features
            if not classes.is_floating_point():
                classes = classes.to(dtype=input.reference_positions.dtype)
            cast(BoltzGenInput, result).template_residue_type_features = project_model_features(
                classes, present[..., None], dtype=dtype, feature="BoltzGen template classes"
            )
    return result
