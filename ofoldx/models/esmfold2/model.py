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
ESMFold2 structure-prediction and binder-design models.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from types import EllipsisType
from typing import Any, NamedTuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from danling.tensors import NestedTensor
from multimolecule import EsmCConfig, EsmCForMaskedLM, EsmCModel
from tensordict import TensorClass, tensorclass
from torch import Tensor

from ...data.tokenization import TOKENS
from ...errors import RequestRef, fail
from ...metrics import (
    categorical_mean,
    chain_pair_mean_tm_scores,
    chain_pair_tm_scores,
    interface_confidence_weights,
    plddt_from_logits,
    predicted_tm_score,
    tm_weight_from_pae,
)
from ...model import Model, TensorCollection
from ...modules.adaptive_norm import AdaptiveLayerNorm
from ...modules.atom import AtomToTokenMean, TokenToAtomBroadcast
from ...modules.attention import GatedMultiheadAttentionProjection, MultiheadAttention
from ...modules.backend import (
    LayerNorm,
    projected_attention,
)
from ...modules.batch import flatten_batch, repeat_batch, unflatten_batch
from ...modules.diffusion.conditioning import condition_pair
from ...modules.diffusion.edm import EdmPreconditioner
from ...modules.diffusion.sampling import CoordinateEdmSampler
from ...modules.diffusion.schedule import karras_noise_schedule
from ...modules.diffusion.types import StepDenoiseFn
from ...modules.embedding.fourier import FrozenFourierEmbedding
from ...modules.embedding.position import RelativePositionEmbedder, generate_relative_position_features
from ...modules.feed_forward import FeedForward, SwiGluFeedForward
from ...modules.functional import banded_pair_mask, weighted_mean
from ...modules.geometry import quaternion_to_rotation, weighted_rigid_align
from ...modules.head import DistogramHead
from ...modules.msa import MsaPairWeightedAveraging, OuterProductMean, PostProjectionOuterProductMean
from ...modules.pair import TriangleMultiplication
from ...modules.transformer import conditioned_transformer_block
from ...modules.transition import ConditionedTransition, Transition
from ..processing import project_model_categories, project_model_coordinates, project_model_features
from ..vocabulary import folding_source_map, map_token_feature_planes
from .config import EsmFold2Config, EsmFold2ExperimentalConfig


class EsmFold2Model(Model):
    r"""
    Predict protein structures with ESMFold 2.

    Args:
        config: Model architecture and inference configuration.
        esmc: Frozen ESMC language-model trunk. ``None`` initializes one from
            ``config.esmc``.
    """

    config_class = EsmFold2Config
    config: EsmFold2Config

    def __init__(self, config: EsmFold2Config | None = None, *, esmc: EsmCModel | None = None) -> None:
        config = EsmFold2Config() if config is None else config
        super().__init__(config)
        self._num_recycles = config.num_recycles
        self.language_model_mask_probability = config.language_model_mask_probability
        self.esmc_bos_token_id = config.esmc.bos_token_id
        self.esmc_eos_token_id = config.esmc.eos_token_id
        self.esmc_mask_token_id = config.esmc.mask_token_id
        self.language_model_dropout = config.language_model_encoder.dropout
        self.per_loop_language_model_dropout = config.language_model_encoder.per_loop_dropout
        self.msa_encoder_overwrite = config.msa_encoder_overwrite
        input_size = config.input_embeddings.input_size
        pair_size = config.pair_size

        self.input_embeddings = EsmFold2InputEmbeddings(config)
        self.pair_left_projection = nn.Linear(input_size, pair_size, bias=False)
        self.pair_right_projection = nn.Linear(input_size, pair_size, bias=False)
        self.relative_position_embedder = RelativePositionEmbedder(
            4 * config.num_relative_residue_bins + 2 * config.num_relative_chain_bins + 7,
            pair_size,
        )
        self.token_bond_projection = nn.Linear(1, pair_size, bias=False)
        if esmc is None:
            esmc = EsmCModel(EsmCConfig(**dict(config.esmc.dict())), add_pooling_layer=False)
        self.esmc = esmc
        self.esmc.requires_grad_(False)
        self.language_model_pair_encoder = EsmFold2LanguageModelPairEncoder(
            pair_size,
            config.esmc.hidden_size,
            config.esmc.num_hidden_layers,
        )

        parcae_decay_init = math.sqrt(1.0 / 5.0)
        parcae_delta_init = -math.log(parcae_decay_init)
        self.parcae_log_delta = nn.Parameter(
            torch.full(
                (pair_size,),
                parcae_delta_init + math.log(-math.expm1(-parcae_delta_init)),
                dtype=torch.float32,
            )
        )
        self.parcae_log_decay = nn.Parameter(torch.zeros(pair_size))
        self.parcae_input_matrix = nn.Parameter(torch.eye(pair_size))

        if config.language_model_encoder.enabled:
            self.language_model_encoder: EsmFold2FoldingStack | None = EsmFold2FoldingStack(
                config.language_model_encoder.num_blocks,
                pair_size,
                expansion=4,
            )
        else:
            self.language_model_encoder = None

        msa_config = config.msa_encoder
        self.msa_encoder: EsmFold2MsaEncoder | None = None
        if msa_config.enabled:
            self.msa_encoder = EsmFold2MsaEncoder(
                msa_size=msa_config.msa_size,
                pair_size=pair_size,
                input_size=input_size,
                hidden_size=msa_config.hidden_size,
                num_stages=msa_config.num_stages,
                num_msa_attention_heads=msa_config.num_msa_attention_heads,
                msa_attention_head_size=msa_config.msa_attention_head_size,
            )

        self.parcae_input_norm = LayerNorm(pair_size)
        folding_stack_config = config.folding_stack
        self.folding_stack = EsmFold2FoldingStack(
            folding_stack_config.num_blocks,
            pair_size,
            expansion=4,
        )
        self.parcae_readout = nn.Linear(pair_size, pair_size, bias=False)
        nn.init.eye_(self.parcae_readout.weight)
        self.parcae_coda = EsmFold2FoldingStack(
            config.parcae.num_coda_blocks,
            pair_size,
            expansion=4,
        )

        self.distogram_head = DistogramHead(pair_size, config.diffusion_structure_head.num_distogram_bins)
        self.diffusion_structure_head = EsmFold2DiffusionStructureHead(config)
        self.confidence_head = EsmFold2ConfidenceHead(config)
        self.post_init()

    def repository_state_dict(self) -> dict[str, Tensor]:
        state = super().repository_state_dict()
        if self.config.esmc_repository is None:
            return state
        return {name: value for name, value in state.items() if not name.startswith("esmc.")}

    def load_repository_state_dict(self, state_dict: Mapping[str, Tensor], *, token: str | None = None) -> None:
        if self.config.esmc_repository is None:
            super().load_repository_state_dict(state_dict, token=token)
            return
        external = {f"esmc.{name}" for name in self.esmc.state_dict()}
        incompatible = self.load_state_dict(state_dict, strict=False, assign=True)
        if set(incompatible.missing_keys) != external or incompatible.unexpected_keys:
            raise RuntimeError("ESMFold2 repository must contain exactly the folding-head parameters")
        esmc, loading = cast(
            tuple[EsmCModel, dict[str, Any]],
            EsmCModel.from_pretrained(
                self.config.esmc_repository,
                revision=self.config.esmc_revision,
                add_pooling_layer=False,
                output_loading_info=True,
                token=token,
            ),
        )
        if loading["missing_keys"]:
            raise RuntimeError(f"ESMC repository has missing encoder weights: {loading['missing_keys']!r}")
        self.esmc.load_state_dict(esmc.state_dict(), strict=True, assign=True)

    def forward(
        self,
        input: EsmFold2Input,
        *,
        noisy_coordinates: Tensor,
        noise_level: Tensor,
        residue_type_probabilities: Tensor | None = None,
        num_recycles: int | None = None,
        language_model_mask_probability: float | None = None,
        language_model_dropout: float | None = None,
        msa_max_depth: int | None = 1024,
        msa_column_mask_rate: float = 0.1,
        msa_subsample_at_inference: bool = True,
        generator: torch.Generator | None = None,
    ) -> EsmFold2Output:
        r"""
        Denoise explicit noisy atom coordinates in one differentiable step.

        ``B`` is batch size, ``S`` is sample count, ``N`` is token count, and
        ``A`` is atom count.

        Args:
            input: Processed molecular input.
            residue_type_probabilities: Optional continuous residue features with shape
                ``(B, N, 72)``. The atom topology and language-model inputs remain
                those supplied in ``input``. Gradients propagate to these features.
            noisy_coordinates: Noisy atom coordinates with shape ``(B, S, A, 3)``
                in angstroms.
            noise_level: Diffusion noise levels with shape ``(B, S)``.
            num_recycles: Number of recurrent trunk passes. ``None`` uses the
                configured value.
            language_model_mask_probability: Probability of masking each nonspecial
                language-model token. ``None`` uses the configured value.
            language_model_dropout: Dropout probability for language-model pair features.
                ``None`` uses the configured value.
            msa_max_depth: Maximum number of MSA rows sampled per recurrent pass. ``None``
                keeps all rows.
            msa_column_mask_rate: Probability of masking a column in nonquery MSA rows.
            msa_subsample_at_inference: Whether to resample MSA rows on each recurrent pass
                during inference.
            generator: Random generator used for token masking, dropout, and MSA sampling.

        Returns:
            An [`EsmFold2Output`][] containing denoised coordinates and raw
                confidence logits.
        """
        input = normalize_esmfold2_input(input, dtype=self.input_embeddings.atom_encoder.atom_projection.weight.dtype)
        noisy_coordinates = _esmfold2_coordinates(
            noisy_coordinates,
            dtype=self.input_embeddings.atom_encoder.atom_projection.weight.dtype,
            feature="ESMFold2 noisy coordinates",
        )

        single_inputs, pair, auxiliary = self._encode_inputs_and_trunk(
            token_index=input.token_index,
            residue_index=input.residue_index,
            chain_index=input.chain_index,
            symmetry_index=input.symmetry_index,
            entity_index=input.entity_index,
            residue_type_ids=input.residue_type_ids,
            residue_type_probabilities=residue_type_probabilities,
            token_bonds=input.token_bonds,
            reference_positions=input.reference_positions,
            reference_elements=input.reference_elements,
            reference_charges=input.reference_charges,
            reference_atom_name_characters=input.reference_atom_name_characters,
            reference_space_id=input.reference_space_id,
            atom_to_token=input.atom_to_token,
            deletion_mean=input.deletion_mean,
            msa_tokens=input.msa_tokens,
            has_deletion=input.has_deletion,
            deletion_value=input.deletion_value,
            msa_mask=input.msa_mask,
            language_model_tokens=input.language_model_tokens,
            language_model_mask=input.language_model_mask,
            token_to_esmc=input.token_to_esmc,
            token_has_esmc=input.token_has_esmc,
            esmc_hidden_states=input.esmc_hidden_states,
            num_recycles=num_recycles,
            language_model_mask_probability=language_model_mask_probability,
            language_model_dropout=language_model_dropout,
            msa_max_depth=msa_max_depth,
            msa_column_mask_rate=msa_column_mask_rate,
            msa_subsample_at_inference=msa_subsample_at_inference,
            generator=generator,
        )
        atom_to_token = auxiliary["atom_to_token"]
        batch_size, num_samples = noisy_coordinates.shape[:2]

        denoiser_output = self.diffusion_structure_head.denoiser(
            noisy_coordinates=flatten_batch(noisy_coordinates),
            noise_level=noise_level.flatten(),
            reference_positions=input.reference_positions,
            reference_charges=input.reference_charges,
            reference_element_features=auxiliary["reference_element_features"],
            reference_atom_name_character_features=auxiliary["reference_atom_name_character_features"],
            reference_space_id=input.reference_space_id,
            atom_to_token=atom_to_token,
            single_inputs=single_inputs,
            pair=pair,
            relative_position_encoding=auxiliary["relative_position_encoding"],
            num_samples=num_samples,
        )
        coordinates = cast(Tensor, denoiser_output["denoised_coordinates"])

        confidence = self.confidence_head(
            single_inputs=single_inputs,
            pair=pair,
            coordinates=coordinates,
            token_representative_atom_index=input.token_representative_atom_index,
            atom_to_token=atom_to_token,
            chain_index=input.chain_index,
            molecule_type=input.molecule_type,
            relative_position_encoding=auxiliary["relative_position_encoding"],
            token_bonds_encoding=auxiliary["token_bonds_encoding"],
            num_samples=num_samples,
        )
        return EsmFold2Output(
            coordinates=unflatten_batch(coordinates, (batch_size, num_samples)),
            distogram_logits=auxiliary["distogram_logits"],
            plddt_logits=unflatten_batch(confidence["plddt_logits"], (batch_size, num_samples)),
            pae_logits=unflatten_batch(confidence["pae_logits"], (batch_size, num_samples)),
            pde_logits=unflatten_batch(confidence["pde_logits"], (batch_size, num_samples)),
            resolved_logits=unflatten_batch(confidence["resolved_logits"], (batch_size, num_samples)),
            batch_size=[batch_size],
        )

    def sample_with_confidence(
        self,
        input: EsmFold2Input,
        /,
        *,
        num_samples: int = 1,
        residue_type_probabilities: Tensor | None = None,
        generator: torch.Generator | None = None,
        num_recycles: int | None = None,
        num_sampling_steps: int | None = None,
        language_model_mask_probability: float | None = None,
        language_model_dropout: float | None = None,
        msa_max_depth: int | None = 1024,
        msa_column_mask_rate: float = 0.1,
        msa_subsample_at_inference: bool = True,
        max_inference_sigma: float | None | EllipsisType = ...,
        noise_scale: float | None = None,
        step_scale: float | None = None,
        denoising_early_exit_rmsd: float | None = None,
    ) -> EsmFold2SamplingWithConfidenceOutput:
        r"""
        Predict raw sampled coordinates and confidence tensors.

        Args:
            input: Processed molecular input.
            num_samples: Number of coordinate samples.
            residue_type_probabilities: Optional continuous residue features aligned
                to the processed token axis, with shape ``(B, N, 72)``.
            generator: PyTorch random generator controlling stochastic inference.
            num_recycles: Number of recurrent trunk passes. ``None`` uses the
                configured value.
            num_sampling_steps: Number of diffusion sampling steps. ``None`` uses the
                configured value.
            language_model_mask_probability: Probability of masking each nonspecial
                language-model token. ``None`` uses the configured value.
            language_model_dropout: Dropout probability for language-model pair features.
                ``None`` uses the configured value.
            msa_max_depth: Maximum number of MSA rows sampled per recurrent pass. ``None``
                keeps all rows.
            msa_column_mask_rate: Probability of masking a column in nonquery MSA rows.
            msa_subsample_at_inference: Whether to resample MSA rows on each recurrent pass
                during inference.
            max_inference_sigma: Largest diffusion sigma included in sampling. Omission
                uses the configured cap; ``None`` keeps the full schedule.
            noise_scale: Sampler noise multiplier. ``None`` uses the configured value.
            step_scale: Sampler step-size multiplier. ``None`` uses the configured value.
            denoising_early_exit_rmsd: Aligned RMSD threshold, in angstroms, for ending
                denoising early. ``None`` disables early exit.

        Returns:
            An [`EsmFold2SamplingWithConfidenceOutput`][] with explicit batch and sample axes.
        """
        input = normalize_esmfold2_input(input, dtype=self.input_embeddings.atom_encoder.atom_projection.weight.dtype)

        if num_samples <= 0:
            fail("value_invalid", RequestRef("num_samples", str(num_samples)))

        raw = self._fold_tensors(
            token_index=input.token_index,
            residue_index=input.residue_index,
            chain_index=input.chain_index,
            symmetry_index=input.symmetry_index,
            entity_index=input.entity_index,
            molecule_type=input.molecule_type,
            residue_type_ids=input.residue_type_ids,
            residue_type_probabilities=residue_type_probabilities,
            token_bonds=input.token_bonds,
            reference_positions=input.reference_positions,
            reference_elements=input.reference_elements,
            reference_charges=input.reference_charges,
            reference_atom_name_characters=input.reference_atom_name_characters,
            reference_space_id=input.reference_space_id,
            atom_to_token=input.atom_to_token,
            token_representative_atom_index=input.token_representative_atom_index,
            deletion_mean=input.deletion_mean,
            msa_tokens=input.msa_tokens,
            has_deletion=input.has_deletion,
            deletion_value=input.deletion_value,
            msa_mask=input.msa_mask,
            language_model_tokens=input.language_model_tokens,
            language_model_mask=input.language_model_mask,
            token_to_esmc=input.token_to_esmc,
            token_has_esmc=input.token_has_esmc,
            esmc_hidden_states=input.esmc_hidden_states,
            num_recycles=num_recycles,
            num_samples=num_samples,
            num_sampling_steps=num_sampling_steps,
            language_model_mask_probability=language_model_mask_probability,
            language_model_dropout=language_model_dropout,
            msa_max_depth=msa_max_depth,
            msa_column_mask_rate=msa_column_mask_rate,
            msa_subsample_at_inference=msa_subsample_at_inference,
            max_inference_sigma=max_inference_sigma,
            noise_scale=noise_scale,
            step_scale=step_scale,
            denoising_early_exit_rmsd=denoising_early_exit_rmsd,
            generator=generator,
        )
        batch_size = input.token_index.shape[0]

        def samples(name: str) -> Tensor:
            return unflatten_batch(raw[name], (batch_size, num_samples))

        return EsmFold2SamplingWithConfidenceOutput(
            coordinates=samples("coordinates"),
            distogram_logits=raw["distogram_logits"],
            plddt_logits=samples("plddt_logits"),
            plddt=samples("plddt"),
            plddt_per_atom=samples("plddt_per_atom"),
            plddt_ca=samples("plddt_ca"),
            mean_plddt=samples("mean_plddt"),
            complex_iplddt=samples("complex_iplddt"),
            pae_logits=samples("pae_logits"),
            pae=samples("pae"),
            pde_logits=samples("pde_logits"),
            pde=samples("pde"),
            resolved_logits=samples("resolved_logits"),
            ptm=samples("ptm"),
            iptm=samples("iptm"),
            chain_pair_iptm=samples("chain_pair_iptm"),
            chain_pair_mean_tm=samples("chain_pair_mean_tm"),
            batch_size=[batch_size],
        )

    def _discretized_dynamics(self) -> tuple[Tensor, Tensor]:
        delta = F.softplus(self.parcae_log_delta)
        decay = torch.exp(-delta * torch.exp(self.parcae_log_decay))
        input_matrix = delta[:, None] * self.parcae_input_matrix
        return decay, input_matrix

    def _init_pair_state(self, reference: Tensor, generator: torch.Generator | None) -> Tensor:
        std = math.sqrt(2.0 / (5.0 * reference.shape[-1]))
        state = torch.empty_like(reference)
        nn.init.trunc_normal_(
            state,
            mean=0.0,
            std=std,
            a=-3 * std,
            b=3 * std,
            generator=generator,
        )
        return state

    def _run_one_loop(
        self,
        *,
        pair_state: Tensor,
        initial_pair: Tensor,
        language_model_pair: Tensor,
        decay: Tensor,
        input_matrix: Tensor,
        num_recycles: int,
        msa_tokens: Tensor | None,
        msa_mask: Tensor | None,
        has_deletion: Tensor | None,
        deletion_value: Tensor | None,
        single_inputs: Tensor,
        msa_max_depth: int | None,
        msa_subsample_at_inference: bool,
        language_model_dropout: float | None,
        generator: torch.Generator | None,
    ) -> Tensor:
        # Keep this loop in a helper so each iteration's pair activations can be
        # released before the distogram and sampling heads run.
        if language_model_dropout is None:
            dropout = self.language_model_dropout
            per_loop_language_model_dropout = self.per_loop_language_model_dropout and dropout > 0.0
        else:
            dropout = language_model_dropout
            per_loop_language_model_dropout = dropout > 0.0

        for _ in range(num_recycles):
            if per_loop_language_model_dropout:
                loop_language_model_pair = _dropout(language_model_pair, dropout, generator)
            else:
                loop_language_model_pair = language_model_pair

            refined_language_model_pair: Tensor | None = None
            if self.language_model_encoder is not None:
                refined_language_model_pair = self.language_model_encoder(
                    loop_language_model_pair.to(initial_pair.dtype),
                )

            injected_pair = initial_pair
            if self.language_model_encoder is None:
                injected_pair = injected_pair + loop_language_model_pair.to(injected_pair.dtype)

            if self.msa_encoder is not None and msa_tokens is not None:
                # Fresh row subsample each iteration (column mask was applied
                # once in forward, before this loop).
                (
                    sampled_msa_tokens,
                    sampled_msa_mask,
                    sampled_has_deletion,
                    sampled_deletion_value,
                ) = _maybe_subsample_msa(
                    msa_tokens,
                    msa_mask,
                    has_deletion,
                    deletion_value,
                    max_depth=msa_max_depth,
                    enabled=msa_subsample_at_inference,
                    generator=generator,
                )
                batch_size, msa_depth, sequence_length = sampled_msa_tokens.shape
                msa_residue_features = F.one_hot(
                    sampled_msa_tokens.permute(0, 2, 1).long(),
                    num_classes=len(TOKENS),
                ).to(initial_pair.dtype)
                msa_attention = (
                    sampled_msa_mask.permute(0, 2, 1).to(initial_pair.dtype)
                    if sampled_msa_mask is not None
                    else initial_pair.new_ones(batch_size, sequence_length, msa_depth)
                )
                # The bias-free MSA projection requires zeroed padding.
                msa_residue_features = msa_residue_features * msa_attention.unsqueeze(-1)
                has_deletion_features = (
                    sampled_has_deletion.permute(0, 2, 1).to(initial_pair.dtype)
                    if sampled_has_deletion is not None
                    else initial_pair.new_zeros(batch_size, sequence_length, msa_depth)
                )
                deletion_features = (
                    sampled_deletion_value.permute(0, 2, 1)
                    if sampled_deletion_value is not None
                    else initial_pair.new_zeros(batch_size, sequence_length, msa_depth)
                )
                msa_pair_update = self.msa_encoder(
                    pair=injected_pair,
                    single_inputs=single_inputs,
                    msa_residue_features=msa_residue_features,
                    has_deletion=has_deletion_features,
                    deletion_value=deletion_features,
                    msa_mask=msa_attention,
                ).to(injected_pair.dtype)
                injected_pair = msa_pair_update if self.msa_encoder_overwrite else (injected_pair + msa_pair_update)

            if refined_language_model_pair is not None:
                injected_pair = injected_pair + refined_language_model_pair.to(injected_pair.dtype)

            normalized_injected_pair = self.parcae_input_norm(injected_pair)
            pair_state = decay * pair_state + F.linear(
                normalized_injected_pair.to(pair_state.dtype),
                input_matrix,
            )
            pair_state = self.folding_stack(pair_state)

        return pair_state

    def _encode_inputs_and_trunk(
        self,
        *,
        token_index: Tensor,
        residue_index: Tensor,
        chain_index: Tensor,
        symmetry_index: Tensor,
        entity_index: Tensor,
        residue_type_ids: Tensor,
        token_bonds: Tensor,
        reference_positions: Tensor,
        reference_elements: Tensor,
        reference_charges: Tensor,
        reference_atom_name_characters: Tensor,
        reference_space_id: Tensor,
        atom_to_token: Tensor,
        deletion_mean: Tensor | None = None,
        msa_tokens: Tensor | None = None,
        has_deletion: Tensor | None = None,
        deletion_value: Tensor | None = None,
        msa_mask: Tensor | None = None,
        language_model_tokens: Tensor | None = None,
        language_model_mask: Tensor | None = None,
        token_to_esmc: Tensor | None = None,
        token_has_esmc: Tensor | None = None,
        esmc_hidden_states: Tensor | None = None,
        residue_type_probabilities: Tensor | None = None,
        num_recycles: int | None = None,
        language_model_mask_probability: float | None = None,
        language_model_dropout: float | None = None,
        msa_max_depth: int | None = 1024,
        msa_column_mask_rate: float = 0.1,
        msa_subsample_at_inference: bool = True,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        r"""
        Run the shared trunk and emit single/pair representations.

        Both the training [`forward`][ofoldx.models.esmfold2.EsmFold2Model.forward] and the inference
        [`sample_with_confidence`][ofoldx.models.esmfold2.EsmFold2Model.sample_with_confidence]
        verb consumes this: it returns the
        single (``single_inputs``) and pair (``pair_state``) trunk streams, plus auxiliary reference features and
        teacher-forced distogram logits.
        """

        num_recycles = num_recycles if num_recycles is not None else self._num_recycles
        if num_recycles <= 0:
            fail("value_invalid", RequestRef("num_recycles", str(num_recycles)))

        residue_type_features = _residue_type_features(
            residue_type_ids, residue_type_probabilities, reference_positions.dtype
        )

        if msa_tokens is not None:
            available = torch.ones_like(msa_tokens, dtype=torch.bool) if msa_mask is None else msa_mask.bool()
            msa_tokens = project_model_categories(
                msa_tokens, available, num_classes=len(TOKENS), feature="ESMFold2 MSA tokens"
            )
            if has_deletion is not None:
                has_deletion = project_model_features(
                    torch.where(available, has_deletion, 0).to(torch.float64),
                    available,
                    dtype=reference_positions.dtype,
                    feature="ESMFold2 deletion presence",
                )
            if deletion_value is not None:
                deletion_value = project_model_features(
                    deletion_value, available, dtype=reference_positions.dtype, feature="ESMFold2 deletion feature"
                )
        if msa_tokens is not None:
            msa_residue_features = F.one_hot(msa_tokens.long(), num_classes=len(TOKENS)).to(reference_positions.dtype)
            if msa_mask is not None:
                mask_f = msa_mask.to(reference_positions.dtype).unsqueeze(-1)
                msa_residue_features = msa_residue_features * mask_f
                valid_seq_count = msa_mask.to(reference_positions.dtype).sum(dim=1).clamp(min=1)
                msa_profile = msa_residue_features.sum(dim=1) / valid_seq_count.unsqueeze(-1)
            else:
                msa_profile = msa_residue_features.mean(dim=1)
        else:
            # Without an alignment the target sequence is the profile: both planes are the canonical classes, so a
            # non-polymer token's mass already sits on ``OTHER`` (upstream's UNK through ``folding_source_map``).
            msa_profile = residue_type_features

        if deletion_mean is None:
            deletion_mean = torch.zeros(
                residue_type_ids.shape[0],
                residue_type_ids.shape[1],
                device=residue_type_ids.device,
                dtype=reference_positions.dtype,
            )

        reference_element_features = F.one_hot(reference_elements.long(), num_classes=_MAX_ATOMIC_NUMBER).to(
            reference_positions.dtype
        )
        reference_atom_name_character_features = F.one_hot(
            reference_atom_name_characters.long(), num_classes=_CHAR_VOCAB_SIZE
        ).to(reference_positions.dtype)

        single_inputs = self.input_embeddings(
            residue_type_features=residue_type_features,
            msa_profile=msa_profile,
            deletion_mean=deletion_mean,
            reference_positions=reference_positions,
            reference_space_id=reference_space_id,
            reference_charges=reference_charges,
            reference_element_features=reference_element_features,
            reference_atom_name_character_features=reference_atom_name_character_features,
            atom_to_token=atom_to_token,
        )

        initial_pair = self.pair_left_projection(single_inputs).unsqueeze(2) + self.pair_right_projection(
            single_inputs
        ).unsqueeze(1)

        relative_features = generate_relative_position_features(
            residue_index=residue_index,
            chain_index=chain_index,
            entity_index=entity_index,
            symmetry_index=symmetry_index,
            token_index=token_index,
            max_relative_residue=self.config.num_relative_residue_bins,
            max_relative_chain=self.config.num_relative_chain_bins,
            chain_gate="not_same_chain",
            dtype=initial_pair.dtype,
        )
        relative_position_encoding = self.relative_position_embedder(relative_features)
        token_bonds_encoding = self.token_bond_projection(token_bonds.to(dtype=self.token_bond_projection.weight.dtype))
        initial_pair = initial_pair + relative_position_encoding + token_bonds_encoding

        if esmc_hidden_states is None:
            if (
                language_model_tokens is None
                or language_model_mask is None
                or token_to_esmc is None
                or token_has_esmc is None
            ):
                raise RuntimeError("ESMFold2 requires ESMC inputs or expert-provided esmc_hidden_states")
            esmc_hidden_states = _compute_esmc_hidden_states(
                self.esmc,
                language_model_tokens,
                language_model_mask,
                token_to_esmc,
                token_has_esmc,
                language_model_mask_probability=(
                    language_model_mask_probability
                    if language_model_mask_probability is not None
                    else self.language_model_mask_probability
                ),
                bos_token_id=self.esmc_bos_token_id,
                eos_token_id=self.esmc_eos_token_id,
                mask_token_id=self.esmc_mask_token_id,
                generator=generator,
            )
        language_model_pair = self.language_model_pair_encoder(esmc_hidden_states)
        del esmc_hidden_states

        pair_state = self._init_pair_state(initial_pair, generator)

        decay, input_matrix = self._discretized_dynamics()
        decay = decay.view(1, 1, 1, -1).to(device=pair_state.device, dtype=pair_state.dtype)
        input_matrix = input_matrix.to(device=pair_state.device, dtype=pair_state.dtype)

        # Inference-time MSA diversity: column mask is applied once here
        # (shared across recycling loops); row subsampling is deferred to
        # per-iter inside _run_one_loop (fresh subset per loop).
        if self.msa_encoder is not None and msa_tokens is not None:
            msa_mask = _maybe_apply_msa_column_masking(
                msa_mask,
                rate=msa_column_mask_rate,
                generator=generator,
            )

        # Method call (not inline loop) frees per-iteration pair locals.
        pair_state = self._run_one_loop(
            pair_state=pair_state,
            initial_pair=initial_pair,
            language_model_pair=language_model_pair,
            decay=decay,
            input_matrix=input_matrix,
            num_recycles=num_recycles,
            msa_tokens=msa_tokens,
            msa_mask=msa_mask,
            has_deletion=has_deletion,
            deletion_value=deletion_value,
            single_inputs=single_inputs,
            msa_max_depth=msa_max_depth,
            msa_subsample_at_inference=msa_subsample_at_inference,
            language_model_dropout=language_model_dropout,
            generator=generator,
        )
        pair_state = self.parcae_readout(pair_state)
        pair_state = self.parcae_coda(pair_state)
        del initial_pair, language_model_pair, decay, input_matrix

        head_dtype = self.distogram_head.projection.weight.dtype
        pair_state = pair_state.to(dtype=head_dtype)
        single_inputs = single_inputs.to(dtype=head_dtype)
        relative_position_encoding = relative_position_encoding.to(dtype=head_dtype)
        token_bonds_encoding = token_bonds_encoding.to(dtype=head_dtype)
        distogram_logits = self.distogram_head(pair_state)

        auxiliary: dict[str, Tensor] = {
            "relative_position_encoding": relative_position_encoding,
            "token_bonds_encoding": token_bonds_encoding,
            "reference_element_features": reference_element_features,
            "reference_atom_name_character_features": reference_atom_name_character_features,
            "atom_to_token": atom_to_token,
            "distogram_logits": distogram_logits,
        }
        return single_inputs, pair_state, auxiliary

    @torch.inference_mode()
    def _fold_tensors(
        self,
        token_index: Tensor,
        residue_index: Tensor,
        chain_index: Tensor,
        symmetry_index: Tensor,
        entity_index: Tensor,
        molecule_type: Tensor,
        residue_type_ids: Tensor,
        token_bonds: Tensor,
        reference_positions: Tensor,
        reference_elements: Tensor,
        reference_charges: Tensor,
        reference_atom_name_characters: Tensor,
        reference_space_id: Tensor,
        atom_to_token: Tensor,
        token_representative_atom_index: Tensor,
        deletion_mean: Tensor | None = None,
        msa_tokens: Tensor | None = None,
        has_deletion: Tensor | None = None,
        deletion_value: Tensor | None = None,
        msa_mask: Tensor | None = None,
        language_model_tokens: Tensor | None = None,
        language_model_mask: Tensor | None = None,
        token_to_esmc: Tensor | None = None,
        token_has_esmc: Tensor | None = None,
        esmc_hidden_states: Tensor | None = None,
        residue_type_probabilities: Tensor | None = None,
        num_recycles: int | None = None,
        num_samples: int | None = None,
        num_sampling_steps: int | None = None,
        language_model_mask_probability: float | None = None,
        language_model_dropout: float | None = None,
        msa_max_depth: int | None = 1024,
        msa_column_mask_rate: float = 0.1,
        msa_subsample_at_inference: bool = True,
        max_inference_sigma: float | None | EllipsisType = ...,
        noise_scale: float | None = None,
        step_scale: float | None = None,
        denoising_early_exit_rmsd: float | None = None,
        generator: torch.Generator | None = None,
    ) -> dict[str, Tensor]:
        num_samples = 1 if num_samples is None else num_samples

        single_inputs, pair, auxiliary = self._encode_inputs_and_trunk(
            token_index=token_index,
            residue_index=residue_index,
            chain_index=chain_index,
            symmetry_index=symmetry_index,
            entity_index=entity_index,
            residue_type_ids=residue_type_ids,
            residue_type_probabilities=residue_type_probabilities,
            token_bonds=token_bonds,
            reference_positions=reference_positions,
            reference_elements=reference_elements,
            reference_charges=reference_charges,
            reference_atom_name_characters=reference_atom_name_characters,
            reference_space_id=reference_space_id,
            atom_to_token=atom_to_token,
            deletion_mean=deletion_mean,
            msa_tokens=msa_tokens,
            has_deletion=has_deletion,
            deletion_value=deletion_value,
            msa_mask=msa_mask,
            language_model_tokens=language_model_tokens,
            language_model_mask=language_model_mask,
            token_to_esmc=token_to_esmc,
            token_has_esmc=token_has_esmc,
            esmc_hidden_states=esmc_hidden_states,
            num_recycles=num_recycles,
            language_model_mask_probability=language_model_mask_probability,
            language_model_dropout=language_model_dropout,
            msa_max_depth=msa_max_depth,
            msa_column_mask_rate=msa_column_mask_rate,
            msa_subsample_at_inference=msa_subsample_at_inference,
            generator=generator,
        )
        relative_position_encoding = auxiliary["relative_position_encoding"]
        token_bonds_encoding = auxiliary["token_bonds_encoding"]
        reference_element_features = auxiliary["reference_element_features"]
        reference_atom_name_character_features = auxiliary["reference_atom_name_character_features"]
        atom_to_token = auxiliary["atom_to_token"]
        distogram_logits = auxiliary["distogram_logits"]

        structure_output = self.diffusion_structure_head.sample(
            pair=pair,
            single_inputs=single_inputs,
            relative_position_encoding=relative_position_encoding,
            reference_positions=reference_positions,
            reference_charges=reference_charges,
            reference_element_features=reference_element_features,
            reference_atom_name_character_features=reference_atom_name_character_features,
            reference_space_id=reference_space_id,
            atom_to_token=atom_to_token,
            num_samples=num_samples,
            num_sampling_steps=num_sampling_steps,
            max_inference_sigma=max_inference_sigma,
            noise_scale=noise_scale,
            step_scale=step_scale,
            return_atom_representations=False,
            denoising_early_exit_rmsd=denoising_early_exit_rmsd,
            generator=generator,
        )

        coordinates = cast(Tensor, structure_output["coordinates"])
        output: dict[str, Tensor] = {"distogram_logits": distogram_logits}
        output["coordinates"] = coordinates

        confidence_output = self.confidence_head(
            single_inputs=single_inputs,
            pair=pair,
            coordinates=coordinates,
            token_representative_atom_index=token_representative_atom_index,
            atom_to_token=atom_to_token,
            chain_index=chain_index,
            molecule_type=molecule_type,
            num_samples=num_samples,
            relative_position_encoding=relative_position_encoding,
            token_bonds_encoding=token_bonds_encoding,
        )
        output.update(confidence_output)
        output["residue_index"] = residue_index
        output["entity_index"] = entity_index
        return output


class EsmFold2ExperimentalModel(Model):
    r"""
    Evaluate the recurrent ESMFold 2 graph used for binder design.

    Args:
        config: Model architecture and inference configuration.
        esmc: Frozen ESMC masked-language model. ``None`` initializes the complete
            child graph from ``config.esmc``.
    """

    config_class = EsmFold2ExperimentalConfig
    config: EsmFold2ExperimentalConfig

    def __init__(
        self,
        config: EsmFold2ExperimentalConfig | None = None,
        *,
        esmc: EsmCForMaskedLM | None = None,
    ) -> None:
        config = EsmFold2ExperimentalConfig() if config is None else config
        super().__init__(config)
        self._num_recycles = config.num_recycles
        self.use_msa_features = config.use_msa_features
        self.language_model_dropout = config.language_model_dropout
        self.force_language_model_dropout = config.force_language_model_dropout
        self.esmc_bos_token_id = config.esmc.bos_token_id
        self.esmc_eos_token_id = config.esmc.eos_token_id
        self.esmc_mask_token_id = config.esmc.mask_token_id
        input_size = config.input_embeddings.input_size
        pair_size = config.pair_size

        self.input_embeddings = EsmFold2InputEmbeddings(config)
        self.pair_left_projection = nn.Linear(input_size, pair_size, bias=False)
        self.pair_right_projection = nn.Linear(input_size, pair_size, bias=False)
        self.relative_position_embedder = RelativePositionEmbedder(
            4 * config.num_relative_residue_bins + 2 * config.num_relative_chain_bins + 7,
            pair_size,
        )
        self.token_bond_projection = nn.Linear(1, pair_size, bias=False)

        if esmc is None:
            esmc = EsmCForMaskedLM(EsmCConfig(**dict(config.esmc.dict())))
        lm_head = esmc.lm_head
        # The head's forward reads ``lm_head.bias`` directly; keep that canonical
        # parameter and remove only the duplicate Linear registration.
        lm_head.decoder.bias = None
        trunk = esmc.model
        trunk.requires_grad_(False)
        self.esmc = trunk.eval()
        lm_head.requires_grad_(False)
        self.esmc_lm_head = lm_head.eval()
        self.language_model_pair_encoder = EsmFold2LanguageModelPairEncoder(
            pair_size,
            config.esmc.hidden_size,
            config.esmc.num_hidden_layers,
        )

        self.recycling_norm = LayerNorm(pair_size)
        self.recycling_projection = nn.Linear(pair_size, pair_size, bias=False)
        nn.init.zeros_(self.recycling_projection.weight)

        msa_config = config.msa_encoder
        self.msa_encoder: EsmFold2ExperimentalMsaEncoder | None = None
        if msa_config.enabled:
            # Every experimental MSA stage is one homogeneous block.
            self.msa_encoder = EsmFold2ExperimentalMsaEncoder(
                msa_size=msa_config.msa_size,
                pair_size=pair_size,
                input_size=input_size,
                hidden_size=msa_config.hidden_size,
                num_blocks=msa_config.num_stages,
                num_msa_attention_heads=msa_config.num_msa_attention_heads,
                msa_attention_head_size=msa_config.msa_attention_head_size,
            )

        folding_stack_config = config.folding_stack
        self.folding_stack = EsmFold2FoldingStack(
            folding_stack_config.num_blocks,
            pair_size,
            expansion=4,
        )

        self.distogram_head = DistogramHead(pair_size, config.diffusion_structure_head.num_distogram_bins)
        self.diffusion_structure_head = EsmFold2DiffusionStructureHead(config)
        # Distogram-only configurations omit the confidence head.
        self.confidence_head: EsmFold2ExperimentalConfidenceHead | None = (
            EsmFold2ExperimentalConfidenceHead(config) if config.confidence.enabled else None
        )
        self.post_init()

    def repository_config(self) -> dict[str, Any]:
        values = super().repository_config()
        if self.config.esmc_repository is not None:
            for key, module in (("esmc_dtype", self.esmc), ("esmc_lm_head_dtype", self.esmc_lm_head)):
                dtypes = {parameter.dtype for parameter in module.parameters() if parameter.is_floating_point()}
                supported = {torch.float16, torch.bfloat16, torch.float32, torch.float64}
                if len(dtypes) != 1 or not dtypes <= supported:
                    fail("value_invalid", RequestRef(key, repr(dtypes)), expected="one supported floating dtype")
                values[key] = str(next(iter(dtypes))).removeprefix("torch.")
        return values

    def repository_state_dict(self) -> dict[str, Tensor]:
        state = super().repository_state_dict()
        if self.config.esmc_repository is None:
            return state
        return {name: value for name, value in state.items() if not name.startswith(("esmc.", "esmc_lm_head."))}

    def load_repository_state_dict(self, state_dict: Mapping[str, Tensor], *, token: str | None = None) -> None:
        if self.config.esmc_repository is None:
            super().load_repository_state_dict(state_dict, token=token)
            return
        external = {
            *(f"esmc.{name}" for name in self.esmc.state_dict()),
            *(f"esmc_lm_head.{name}" for name in self.esmc_lm_head.state_dict()),
        }
        incompatible = self.load_state_dict(state_dict, strict=False, assign=True)
        if set(incompatible.missing_keys) != external or incompatible.unexpected_keys:
            raise RuntimeError("experimental ESMFold2 repository must contain exactly the folding-head parameters")
        reused = getattr(self, "_repository_esmc", None)
        if reused is not None:
            trunk, lm_head = reused
            self.esmc.load_state_dict(trunk.state_dict(), strict=True, assign=True)
            self.esmc_lm_head.load_state_dict(lm_head.state_dict(), strict=True, assign=True)
            del self._repository_esmc
            return
        esmc, loading = cast(
            tuple[EsmCForMaskedLM, dict[str, Any]],
            EsmCForMaskedLM.from_pretrained(
                self.config.esmc_repository,
                revision=self.config.esmc_revision,
                output_loading_info=True,
                token=token,
            ),
        )
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
            if loading.get(key):
                raise RuntimeError(f"ESMC repository has invalid masked-language-model weights: {loading[key]!r}")
        expected = EsmCConfig(**dict(self.config.esmc.dict()))
        differences = {
            name: (getattr(expected, name), getattr(esmc.config, name, None)) for name in self.config.esmc.dict()
        }
        differences = {name: value for name, value in differences.items() if value[0] != value[1]}
        if differences:
            raise RuntimeError(f"ESMC repository configuration does not match the folding head: {differences!r}")
        # MultiMolecule aliases decoder.bias to lm_head.bias; this graph keeps
        # the canonical parameter only, as in __init__.
        esmc.lm_head.decoder.bias = None
        dtypes = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
            "float64": torch.float64,
        }
        for name, module in (("esmc_dtype", esmc.model), ("esmc_lm_head_dtype", esmc.lm_head)):
            dtype = getattr(self.config, name)
            if dtype is not None:
                if dtype not in dtypes:
                    fail("value_invalid", RequestRef(name, repr(dtype)), expected=tuple(dtypes))
                module.to(dtype=dtypes[dtype])
        self.esmc.load_state_dict(esmc.model.state_dict(), strict=True, assign=True)
        self.esmc_lm_head.load_state_dict(esmc.lm_head.state_dict(), strict=True, assign=True)

    def configure_language_model_dropout(self, dropout: float, *, force_during_inference: bool = True) -> None:
        r"""
        Configure dropout on ESMC-derived pair features.

        Args:
            dropout: Dropout probability used by subsequent calls.
            force_during_inference: Whether to apply the dropout outside training mode.
        """

        self.language_model_dropout = dropout
        self.force_language_model_dropout = force_during_inference

    def forward(
        self,
        input: EsmFold2Input,
        *,
        residue_type_probabilities: Tensor | None = None,
        num_recycles: int | None = None,
        language_model_mask_probability: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        r"""
        Predict differentiable distogram logits from discrete or soft residue types.

        ``B`` is batch size and ``N`` is token count.

        Args:
            input: Processed target-binder input.
            residue_type_probabilities: Optional soft residue-type probabilities with
                shape ``(B, N, 72)``. Gradients from ``distogram_logits`` propagate to
                this tensor.
            num_recycles: Number of recurrent trunk passes. ``None`` uses the
                configured value.
            language_model_mask_probability: Probability of masking each nonspecial
                language-model token.
            generator: Caller-owned PyTorch random generator.

        Returns:
            Distogram logits with shape ``(B, N, N, D)``.
        """
        input = normalize_esmfold2_input(input, dtype=self.input_embeddings.atom_encoder.atom_projection.weight.dtype)

        trunk = self._encode_and_trunk(
            input,
            residue_type_probabilities=residue_type_probabilities,
            num_recycles=num_recycles,
            language_model_mask_probability=language_model_mask_probability,
            generator=generator,
        )
        return trunk.distogram_logits

    def sample(
        self,
        input: EsmFold2Input,
        *,
        residue_type_probabilities: Tensor | None = None,
        num_recycles: int | None = None,
        num_samples: int = 1,
        num_sampling_steps: int | None = None,
        language_model_mask_probability: float = 0.0,
        max_inference_sigma: float | None | EllipsisType = ...,
        noise_scale: float | None = None,
        step_scale: float | None = None,
        generator: torch.Generator | None = None,
    ) -> EsmFold2ExperimentalSamplingOutput:
        r"""
        Predict distogram logits and sample atom coordinates.

        Args:
            input: Processed target-binder input.
            residue_type_probabilities: Optional soft residue-type probabilities with
                shape ``(B, N, 72)``.
            num_recycles: Number of recurrent trunk passes. ``None`` uses the
                configured value.
            num_samples: Number of coordinate samples.
            num_sampling_steps: Number of diffusion sampling steps. ``None`` uses the
                configured value.
            language_model_mask_probability: Probability of masking each nonspecial
                language-model token.
            max_inference_sigma: Largest diffusion sigma included in sampling. Omission
                uses the configured cap; ``None`` keeps the full schedule.
            noise_scale: Sampler noise multiplier. ``None`` uses the configured value.
            step_scale: Sampler step-size multiplier. ``None`` uses the configured value.
            generator: Caller-owned PyTorch random generator.

        Returns:
            Distogram logits and sampled atom coordinates.
        """
        input = normalize_esmfold2_input(input, dtype=self.input_embeddings.atom_encoder.atom_projection.weight.dtype)

        trunk = self._encode_and_trunk(
            input,
            residue_type_probabilities=residue_type_probabilities,
            num_recycles=num_recycles,
            language_model_mask_probability=language_model_mask_probability,
            generator=generator,
        )
        return self._sample_from_trunk(
            input,
            trunk,
            num_samples=num_samples,
            num_sampling_steps=num_sampling_steps,
            max_inference_sigma=max_inference_sigma,
            noise_scale=noise_scale,
            step_scale=step_scale,
            generator=generator,
        )

    def sample_with_confidence(
        self,
        input: EsmFold2Input,
        *,
        residue_type_probabilities: Tensor | None = None,
        num_recycles: int | None = None,
        num_samples: int = 1,
        num_sampling_steps: int | None = None,
        language_model_mask_probability: float = 0.0,
        max_inference_sigma: float | None | EllipsisType = ...,
        noise_scale: float | None = None,
        step_scale: float | None = None,
        generator: torch.Generator | None = None,
    ) -> EsmFold2ExperimentalSamplingWithConfidenceOutput:
        r"""
        Predict distogram logits, sample atom coordinates, and compute confidence.

        Args:
            input: Processed target-binder input.
            residue_type_probabilities: Optional soft residue-type probabilities with
                shape ``(B, N, 72)``.
            num_recycles: Number of recurrent trunk passes. ``None`` uses the
                configured value.
            num_samples: Number of coordinate samples.
            num_sampling_steps: Number of diffusion sampling steps. ``None`` uses the
                configured value.
            language_model_mask_probability: Probability of masking each nonspecial
                language-model token.
            max_inference_sigma: Largest diffusion sigma included in sampling. Omission
                uses the configured cap; ``None`` keeps the full schedule.
            noise_scale: Sampler noise multiplier. ``None`` uses the configured value.
            step_scale: Sampler step-size multiplier. ``None`` uses the configured value.
            generator: Caller-owned PyTorch random generator.

        Returns:
            A coordinate-sample carrier and its fixed confidence values.
        """
        input = normalize_esmfold2_input(input, dtype=self.input_embeddings.atom_encoder.atom_projection.weight.dtype)

        trunk = self._encode_and_trunk(
            input,
            residue_type_probabilities=residue_type_probabilities,
            num_recycles=num_recycles,
            language_model_mask_probability=language_model_mask_probability,
            generator=generator,
        )
        sample = self._sample_from_trunk(
            input,
            trunk,
            num_samples=num_samples,
            num_sampling_steps=num_sampling_steps,
            max_inference_sigma=max_inference_sigma,
            noise_scale=noise_scale,
            step_scale=step_scale,
            generator=generator,
        )
        confidence = self._confidence_from_trunk(
            input,
            trunk,
            sample,
        )
        return EsmFold2ExperimentalSamplingWithConfidenceOutput(sample=sample, confidence=confidence)

    def _encode_and_trunk(
        self,
        input: EsmFold2Input,
        *,
        residue_type_probabilities: Tensor | None,
        num_recycles: int | None,
        language_model_mask_probability: float,
        generator: torch.Generator | None,
    ) -> _EsmFold2ExperimentalTrunkOutput:
        r"""
        Run the recurrent trunk and return single/pair/distogram plus sampler auxiliary.
        """

        residue_type_ids = input.residue_type_ids
        num_recycles = self._num_recycles if num_recycles is None else num_recycles
        if num_recycles <= 0:
            fail("value_invalid", RequestRef("num_recycles", str(num_recycles)))

        # Soft one-hot design sequence overrides the discrete residue types on the
        # differentiable embedder path; the atom features stay tied to the argmax
        # residues supplied in the input.
        residue_type_features = _residue_type_features(
            residue_type_ids, residue_type_probabilities, input.reference_positions.dtype
        )

        deletion_mean = torch.zeros(
            residue_type_ids.shape[0],
            residue_type_ids.shape[1],
            device=residue_type_ids.device,
            dtype=input.reference_positions.dtype,
        )

        msa_tokens = input.msa_tokens
        msa_features: Tensor | None = None
        msa_mask = input.msa_mask
        if self.use_msa_features and (residue_type_probabilities is not None or msa_tokens is None):
            # The target sequence supplies the single-row MSA when no evidence is present.
            msa_profile = residue_type_features
            msa_features = msa_profile.unsqueeze(1)
            msa_mask = torch.ones_like(residue_type_ids, dtype=torch.bool).unsqueeze(1)
        elif msa_tokens is not None:
            msa_residue_features = F.one_hot(msa_tokens.long(), num_classes=len(TOKENS)).to(
                input.reference_positions.dtype
            )
            if msa_mask is not None:
                mask_f = msa_mask.to(input.reference_positions.dtype).unsqueeze(-1)
                msa_residue_features = msa_residue_features * mask_f
                valid_seq_count = msa_mask.to(input.reference_positions.dtype).sum(dim=1).clamp(min=1)
                msa_profile = msa_residue_features.sum(dim=1) / valid_seq_count.unsqueeze(-1)
            else:
                msa_profile = msa_residue_features.mean(dim=1)
        else:
            msa_profile = residue_type_features

        if not self.use_msa_features:
            msa_profile = torch.zeros_like(msa_profile)
            deletion_mean = torch.zeros_like(deletion_mean)
            msa_tokens = None
            msa_features = None

        reference_element_features = F.one_hot(input.reference_elements.long(), num_classes=_MAX_ATOMIC_NUMBER).to(
            input.reference_positions.dtype
        )
        reference_atom_name_character_features = F.one_hot(
            input.reference_atom_name_characters.long(), num_classes=_CHAR_VOCAB_SIZE
        ).to(input.reference_positions.dtype)
        atom_to_token = input.atom_to_token

        single_inputs = self.input_embeddings(
            residue_type_features=residue_type_features,
            msa_profile=msa_profile,
            deletion_mean=deletion_mean,
            reference_positions=input.reference_positions,
            reference_space_id=input.reference_space_id,
            reference_charges=input.reference_charges,
            reference_element_features=reference_element_features,
            reference_atom_name_character_features=reference_atom_name_character_features,
            atom_to_token=atom_to_token,
        )

        initial_pair = self.pair_left_projection(single_inputs).unsqueeze(2) + self.pair_right_projection(
            single_inputs
        ).unsqueeze(1)
        relative_features = generate_relative_position_features(
            residue_index=input.residue_index,
            chain_index=input.chain_index,
            entity_index=input.entity_index,
            symmetry_index=input.symmetry_index,
            token_index=input.token_index,
            max_relative_residue=self.config.num_relative_residue_bins,
            max_relative_chain=self.config.num_relative_chain_bins,
            chain_gate="not_same_chain",
            dtype=initial_pair.dtype,
        )
        relative_position_encoding = self.relative_position_embedder(relative_features)
        token_bonds_encoding = self.token_bond_projection(
            input.token_bonds.to(dtype=self.token_bond_projection.weight.dtype)
        )
        initial_pair = initial_pair + relative_position_encoding + token_bonds_encoding

        esmc_hidden_states = input.esmc_hidden_states
        if esmc_hidden_states is None:
            if input.language_model_tokens is None or input.language_model_mask is None or input.token_to_esmc is None:
                raise RuntimeError("ESMFold2 experimental requires ESMC inputs or esmc_hidden_states")
            with torch.no_grad():
                esmc_hidden_states = _compute_esmc_hidden_states(
                    self.esmc,
                    input.language_model_tokens,
                    input.language_model_mask,
                    input.token_to_esmc,
                    input.token_has_esmc,
                    language_model_mask_probability=language_model_mask_probability,
                    bos_token_id=self.esmc_bos_token_id,
                    eos_token_id=self.esmc_eos_token_id,
                    mask_token_id=self.esmc_mask_token_id,
                    generator=generator,
                )
        language_model_pair = self.language_model_pair_encoder(esmc_hidden_states)
        if self.language_model_dropout > 0 and (self.training or self.force_language_model_dropout):
            language_model_pair = _dropout(language_model_pair, self.language_model_dropout, generator)
        initial_pair = initial_pair + language_model_pair.to(initial_pair.dtype)
        del esmc_hidden_states, language_model_pair

        pair = torch.zeros_like(initial_pair)
        for _ in range(num_recycles):
            pair = initial_pair + self.recycling_projection(self.recycling_norm(pair))
            if self.msa_encoder is not None and (msa_tokens is not None or msa_features is not None):
                if msa_features is not None:
                    msa_depth = msa_features.shape[1]
                    msa_residue_features = msa_features.permute(0, 2, 1, 3)
                else:
                    msa_depth = cast(Tensor, msa_tokens).shape[1]
                    msa_residue_features = F.one_hot(
                        cast(Tensor, msa_tokens).permute(0, 2, 1).long(),
                        num_classes=len(TOKENS),
                    ).to(input.reference_positions.dtype)
                msa_attention = (
                    msa_mask.permute(0, 2, 1).to(input.reference_positions.dtype)
                    if msa_mask is not None
                    else torch.ones_like(msa_residue_features[..., 0])
                )
                msa_residue_features = msa_residue_features * msa_attention.unsqueeze(-1)
                zero_deletion_features = torch.zeros(
                    msa_residue_features.shape[0],
                    msa_residue_features.shape[1],
                    msa_depth,
                    device=msa_residue_features.device,
                )
                pair = pair + self.msa_encoder(
                    pair=pair,
                    single_inputs=single_inputs,
                    msa_residue_features=msa_residue_features,
                    has_deletion=zero_deletion_features,
                    deletion_value=zero_deletion_features,
                    msa_mask=msa_attention,
                ).to(pair.dtype)
            pair = self.folding_stack(pair)

        distogram_logits = self.distogram_head(pair)

        return _EsmFold2ExperimentalTrunkOutput(
            single_inputs=single_inputs,
            pair=pair,
            distogram_logits=distogram_logits,
            relative_position_encoding=relative_position_encoding,
            token_bonds_encoding=token_bonds_encoding,
            reference_element_features=reference_element_features,
            reference_atom_name_character_features=reference_atom_name_character_features,
            atom_to_token=atom_to_token,
        )

    def _sample_from_trunk(
        self,
        input: EsmFold2Input,
        trunk: _EsmFold2ExperimentalTrunkOutput,
        *,
        num_samples: int,
        num_sampling_steps: int | None,
        max_inference_sigma: float | None | EllipsisType,
        noise_scale: float | None,
        step_scale: float | None,
        generator: torch.Generator | None,
    ) -> EsmFold2ExperimentalSamplingOutput:
        structure_output = self.diffusion_structure_head.sample(
            pair=trunk.pair,
            single_inputs=trunk.single_inputs,
            relative_position_encoding=trunk.relative_position_encoding,
            reference_positions=input.reference_positions,
            reference_charges=input.reference_charges,
            reference_element_features=trunk.reference_element_features,
            reference_atom_name_character_features=trunk.reference_atom_name_character_features,
            reference_space_id=input.reference_space_id,
            atom_to_token=trunk.atom_to_token,
            num_samples=num_samples,
            num_sampling_steps=num_sampling_steps,
            max_inference_sigma=max_inference_sigma,
            noise_scale=noise_scale,
            step_scale=step_scale,
            return_atom_representations=False,
            generator=generator,
        )
        coordinates = cast(Tensor, structure_output["coordinates"])
        batch_size = input.token_index.shape[0]
        return EsmFold2ExperimentalSamplingOutput(
            distogram_logits=trunk.distogram_logits,
            coordinates=unflatten_batch(coordinates, (batch_size, num_samples)),
            batch_size=[batch_size],
        )

    def _confidence_from_trunk(
        self,
        input: EsmFold2Input,
        trunk: _EsmFold2ExperimentalTrunkOutput,
        sample: EsmFold2ExperimentalSamplingOutput,
    ) -> EsmFold2ExperimentalConfidenceOutput:
        confidence_head = self.confidence_head
        if confidence_head is None:
            fail("model_capacity", RequestRef("confidence.enabled", "False"))
        return confidence_head(
            single_inputs=trunk.single_inputs,
            pair=trunk.pair,
            coordinates=sample.coordinates,
            token_representative_atom_index=input.token_representative_atom_index,
            atom_to_token=trunk.atom_to_token,
            chain_index=input.chain_index,
            molecule_type=input.molecule_type,
            relative_position_encoding=trunk.relative_position_encoding,
            token_bonds_encoding=trunk.token_bonds_encoding,
        )


class EsmFold2InputEmbeddings(nn.Module):
    r"""
    Embeds input features including atom-level encoding via SWA attention.
    """

    def __init__(
        self,
        config: EsmFold2Config | EsmFold2ExperimentalConfig,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        atom_encoder_config = config.input_embeddings.atom_encoder

        self.atom_encoder = EsmFold2AtomEncoder(
            backend=backend,
            atom_hidden_size=atom_encoder_config.atom_hidden_size,
            token_hidden_size=atom_encoder_config.token_hidden_size,
            num_blocks=atom_encoder_config.num_blocks,
            num_attention_heads=atom_encoder_config.num_attention_heads,
            sliding_window_size=atom_encoder_config.sliding_window_size,
            expansion=atom_encoder_config.expansion,
            spatial_rope_base_frequency=atom_encoder_config.spatial_rope_base_frequency,
            num_spatial_rope_pairs_per_axis=atom_encoder_config.num_spatial_rope_pairs_per_axis,
            num_reference_space_rope_pairs=atom_encoder_config.num_reference_space_rope_pairs,
            reference_space_rope_base_frequency=atom_encoder_config.reference_space_rope_base_frequency,
        )

    def forward(
        self,
        residue_type_features: Tensor,
        msa_profile: Tensor,
        deletion_mean: Tensor,
        reference_positions: Tensor,
        reference_space_id: Tensor,
        reference_charges: Tensor,
        reference_element_features: Tensor,
        reference_atom_name_character_features: Tensor,
        atom_to_token: Tensor,
    ) -> Tensor:
        token_features, _atom_query, _atom_conditioning, _intermediates = self.atom_encoder(
            reference_positions=reference_positions,
            reference_space_id=reference_space_id,
            reference_charges=reference_charges,
            reference_element_features=reference_element_features,
            reference_atom_name_character_features=reference_atom_name_character_features,
            atom_to_token=atom_to_token,
            token_reference=residue_type_features,
        )
        return torch.cat(
            [
                token_features,
                residue_type_features.to(dtype=token_features.dtype),
                msa_profile.to(dtype=token_features.dtype),
                deletion_mean.unsqueeze(-1).to(dtype=token_features.dtype),
            ],
            dim=-1,
        )


class EsmFold2LanguageModelPairEncoder(nn.Module):
    r"""
    Project and combine ESMC layer states into the pair representation.
    """

    def __init__(self, pair_size: int = 256, hidden_size: int = 2560, num_hidden_layers: int = 80) -> None:
        super().__init__()

        self.hidden_norm = LayerNorm(hidden_size)
        self.hidden_projection = nn.Linear(hidden_size, pair_size, bias=False)
        self.layer_weights = nn.Parameter(torch.zeros(num_hidden_layers + 1))
        self.single_to_pair = EsmFold2SingleToPair(pair_size, pair_size, pair_size)
        self.output_norm = LayerNorm(pair_size)

    def forward(self, hidden_states: Tensor) -> Tensor:
        if not torch.is_autocast_enabled(hidden_states.device.type):
            hidden_states = hidden_states.to(dtype=self.layer_weights.dtype)
        language_model_pair = self.hidden_projection(self.hidden_norm(hidden_states))  # [B, L, 81, pair_size]
        weights = self.layer_weights.softmax(0)  # [81]
        # Keep the batched-matmul reduction order. Do not squeeze:
        # ``weights @ language_model_pair`` already has shape [B, L, pair_size], including when L=1.
        language_model_pair = weights @ language_model_pair  # [B, L, pair_size]
        return self.output_norm(self.single_to_pair(language_model_pair))  # [B, L, L, pair_size]


class EsmFold2SingleToPair(nn.Module):
    r"""
    Build pair features from projected single features.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.single_projection = nn.Linear(input_size, hidden_size)
        self.feed_forward = FeedForward(2 * hidden_size, output_size, output_size, activation="gelu")

    def forward(self, x: Tensor) -> Tensor:
        x = self.single_projection(x)
        x = torch.cat(
            [(x.unsqueeze(2) * x.unsqueeze(1)), (x.unsqueeze(2) - x.unsqueeze(1))],
            dim=3,
        )
        return self.feed_forward(x)


class EsmFold2FoldingStack(nn.Module):
    r"""
    Homogeneous stack of pair-only folding blocks.
    """

    _gradient_checkpointing_func: Callable[..., Tensor]

    def __init__(
        self,
        num_blocks: int = 24,
        pair_size: int = 256,
        expansion: int = 4,
        *,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.gradient_checkpointing = False
        self.blocks = nn.ModuleList(
            [EsmFold2FoldingBlock(backend, pair_size=pair_size, expansion=expansion) for _ in range(num_blocks)]
        )

    def forward(self, pair: Tensor, pair_attention_mask: Tensor | None = None) -> Tensor:
        for block in self.blocks:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                pair = self._gradient_checkpointing_func(
                    block.__call__,
                    pair,
                    pair_attention_mask=pair_attention_mask,
                )
            else:
                pair = block(pair, pair_attention_mask=pair_attention_mask)
        return pair


class EsmFold2MsaEncoder(nn.Module):
    r"""
    Homogeneous MSA blocks followed by one final OPM and pair update.

    ``num_stages`` counts both the joint MSA/pair blocks and the distinct final
    pair-only stage.
    """

    _gradient_checkpointing_func: Callable[..., tuple[Tensor, Tensor]]

    def __init__(
        self,
        msa_size: int,
        pair_size: int,
        input_size: int,
        hidden_size: int = 32,
        num_stages: int = 4,
        num_msa_attention_heads: int = 8,
        msa_attention_head_size: int = 16,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.num_stages = num_stages
        self.gradient_checkpointing = False
        self.msa_projection = nn.Linear(len(TOKENS) + 2, msa_size, bias=False)
        self.input_projection = nn.Linear(input_size, msa_size, bias=False)
        self.blocks = nn.ModuleList(
            [
                EsmFold2MsaEncoderBlock(
                    msa_size=msa_size,
                    pair_size=pair_size,
                    hidden_size=hidden_size,
                    num_msa_attention_heads=num_msa_attention_heads,
                    msa_attention_head_size=msa_attention_head_size,
                    backend=backend,
                )
                for _ in range(num_stages - 1)
            ]
        )
        self.final_outer_product_mean = (
            EsmFold2OuterProductMean(
                msa_size,
                hidden_size,
                pair_size,
                backend,
            )
            if num_stages
            else None
        )
        self.final_pair_update = (
            EsmFold2FoldingBlock(
                backend,
                pair_size=pair_size,
                expansion=4,
            )
            if num_stages
            else None
        )

    def forward(
        self,
        pair: Tensor,
        single_inputs: Tensor,
        msa_residue_features: Tensor,
        has_deletion: Tensor,
        deletion_value: Tensor,
        msa_mask: Tensor,
    ) -> Tensor:
        # All inputs are pre-transposed to [B, L, M, ...] before calling.
        available = msa_mask.bool()
        msa_residue_features = project_model_features(
            msa_residue_features,
            available[..., None],
            dtype=self.msa_projection.weight.dtype,
            feature="ESMFold2 MSA residue features",
        )
        has_deletion = project_model_features(
            torch.where(available, has_deletion, 0).to(torch.float64),
            available,
            dtype=self.msa_projection.weight.dtype,
            feature="ESMFold2 deletion presence",
        )
        deletion_value = project_model_features(
            deletion_value, available, dtype=self.msa_projection.weight.dtype, feature="ESMFold2 deletion feature"
        )
        msa_features = torch.cat(
            [msa_residue_features, has_deletion.unsqueeze(-1), deletion_value.unsqueeze(-1)],
            dim=-1,
        )
        m = self.msa_projection(msa_features.to(dtype=self.msa_projection.weight.dtype))
        m = m + self.input_projection(single_inputs.to(dtype=self.input_projection.weight.dtype)).unsqueeze(2)
        query_mask = msa_mask[:, :, 0].bool()
        pair_attention_mask = query_mask.unsqueeze(2) & query_mask.unsqueeze(1)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                m, pair = self._gradient_checkpointing_func(
                    block.__call__,
                    m,
                    pair,
                    msa_mask,
                    pair_attention_mask,
                )
            else:
                m, pair = block(m, pair, msa_mask, pair_attention_mask)
        if self.final_outer_product_mean is not None and self.final_pair_update is not None:
            pair = pair + self.final_outer_product_mean(m, msa_mask)
            pair = self.final_pair_update(pair, pair_attention_mask)
        return pair


# Experimental recurrent ESMFold2 architecture. It reinjects the previous pair
# representation through the recycling projection, adds the ESMC projection once,
# and predicts pLDDT and PAE from a 128-bin distogram. residue_type_probabilities keeps the
# distogram differentiable for binder design.


class EsmFold2ExperimentalMsaEncoder(nn.Module):
    r"""
    Homogeneous experimental MSA blocks that update the pair representation.
    """

    _gradient_checkpointing_func: Callable[..., tuple[Tensor, Tensor]]

    def __init__(
        self,
        msa_size: int,
        pair_size: int,
        input_size: int,
        hidden_size: int = 32,
        num_blocks: int = 4,
        num_msa_attention_heads: int = 8,
        msa_attention_head_size: int = 32,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.gradient_checkpointing = False
        self.msa_projection = nn.Linear(len(TOKENS) + 2, msa_size, bias=False)
        self.input_projection = nn.Linear(input_size, msa_size, bias=False)
        self.blocks = nn.ModuleList(
            [
                EsmFold2ExperimentalMsaEncoderBlock(
                    msa_size=msa_size,
                    pair_size=pair_size,
                    hidden_size=hidden_size,
                    num_msa_attention_heads=num_msa_attention_heads,
                    msa_attention_head_size=msa_attention_head_size,
                    backend=backend,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(
        self,
        pair: Tensor,
        single_inputs: Tensor,
        msa_residue_features: Tensor,
        has_deletion: Tensor,
        deletion_value: Tensor,
        msa_mask: Tensor,
    ) -> Tensor:
        # All inputs are pre-transposed to [B, L, M, ...] before calling.
        if not self.blocks:
            return pair
        B, L, M = msa_mask.shape
        available = msa_mask.bool()
        msa_residue_features = project_model_features(
            msa_residue_features,
            available[..., None],
            dtype=self.msa_projection.weight.dtype,
            feature="ESMFold2 MSA residue features",
        )
        has_deletion = project_model_features(
            torch.where(available, has_deletion, 0).to(torch.float64),
            available,
            dtype=self.msa_projection.weight.dtype,
            feature="ESMFold2 deletion presence",
        )
        deletion_value = project_model_features(
            deletion_value, available, dtype=self.msa_projection.weight.dtype, feature="ESMFold2 deletion feature"
        )
        msa_features = torch.cat(
            [msa_residue_features, has_deletion.unsqueeze(-1), deletion_value.unsqueeze(-1)],
            dim=-1,
        )
        m = self.msa_projection(msa_features.to(dtype=self.msa_projection.weight.dtype))
        m = m + self.input_projection(single_inputs.to(dtype=self.input_projection.weight.dtype)).unsqueeze(2)

        if M > 1:
            msa_track_mask = msa_mask[:, :, 1:].any(dim=(1, 2))
        else:
            msa_track_mask = torch.zeros(B, dtype=torch.bool, device=pair.device)

        query_mask = msa_mask[:, :, 0]
        pair_attention_mask = query_mask.unsqueeze(2) * query_mask.unsqueeze(1)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                m, pair = self._gradient_checkpointing_func(
                    block.__call__,
                    m,
                    pair,
                    msa_mask,
                    pair_attention_mask,
                    msa_track_mask,
                )
            else:
                m, pair = block(m, pair, msa_mask, pair_attention_mask, msa_track_mask)
        return pair * msa_track_mask[:, None, None, None].to(pair.dtype)


class _EsmFold2AtomReferenceEncoder(nn.Module):
    r"""
    Prepare projected reference atoms, rotary geometry, and their sample-expanded token map.

    Owns the original atom projection, input normalization, and atom transformer
    fields directly, preserving their parameter/state prefixes. The concrete
    encoder owns its distinct token projection and noisy-coordinate operation.
    """

    def __init__(
        self,
        backend: str = "auto",
        atom_hidden_size: int = 128,
        num_blocks: int = 3,
        num_attention_heads: int = 4,
        sliding_window_size: int = 128,
        expansion: int = 2,
        spatial_rope_base_frequency: float = 20.0,
        num_spatial_rope_pairs_per_axis: int = 2,
        num_reference_space_rope_pairs: int = 10,
        reference_space_rope_base_frequency: float = 10000.0,
    ) -> None:
        super().__init__()

        self.atom_projection = nn.Linear(_ATOM_FEATURE_SIZE, atom_hidden_size, bias=False)
        self.input_norm = LayerNorm(atom_hidden_size, backend=backend)

        self.encoder = EsmFold2AtomTransformerEncoder(
            backend=backend,
            atom_hidden_size=atom_hidden_size,
            num_blocks=num_blocks,
            num_attention_heads=num_attention_heads,
            sliding_window_size=sliding_window_size,
            expansion=expansion,
            spatial_rope_base_frequency=spatial_rope_base_frequency,
            num_spatial_rope_pairs_per_axis=num_spatial_rope_pairs_per_axis,
            num_reference_space_rope_pairs=num_reference_space_rope_pairs,
            reference_space_rope_base_frequency=reference_space_rope_base_frequency,
        )

    def encode_reference(
        self,
        reference_positions: Tensor | NestedTensor,
        reference_space_id: Tensor | NestedTensor,
        reference_charges: Tensor,
        reference_element_features: Tensor,
        reference_atom_name_character_features: Tensor,
        atom_to_token: Tensor | NestedTensor,
        num_samples: int = 1,
    ) -> tuple[
        Tensor | NestedTensor,
        Tensor | NestedTensor,
        Tensor | NestedTensor,
        Tensor | NestedTensor,
    ]:
        reference_positions = _esmfold2_coordinates(
            reference_positions, dtype=self.atom_projection.weight.dtype, feature="ESMFold2 encoded references"
        )
        reference_charges = project_model_features(
            reference_charges.to(torch.float64) if not reference_charges.is_floating_point() else reference_charges,
            torch.ones_like(reference_charges, dtype=torch.bool),
            dtype=self.atom_projection.weight.dtype,
            feature="ESMFold2 reference charges",
        )
        charges = reference_charges.unsqueeze(-1)
        atom_features = torch.cat(
            [
                reference_positions,
                charges,
                # Released atom-validity channel. The atom axis carries no padding, so it is constant.
                torch.ones_like(charges),
                reference_element_features,
                reference_atom_name_character_features.flatten(-2),
            ],
            dim=-1,
        )
        atom_features = project_model_features(
            atom_features,
            torch.ones_like(atom_features, dtype=torch.bool),
            dtype=self.atom_projection.weight.dtype,
            feature="ESMFold2 encoded atom features",
        )
        reference_atom_features = self.input_norm(self.atom_projection(atom_features))
        attention_cos, attention_sin = self.encoder.build_3d_rope(reference_positions, reference_space_id)
        attention_cos = attention_cos.to(dtype=reference_atom_features.dtype)
        attention_sin = attention_sin.to(dtype=reference_atom_features.dtype)
        reference_atom_features = repeat_batch(reference_atom_features, num_samples)
        attention_cos = repeat_batch(attention_cos, num_samples)
        attention_sin = repeat_batch(attention_sin, num_samples)
        expanded_atom_to_token = repeat_batch(atom_to_token, num_samples)
        return (
            reference_atom_features,
            attention_cos,
            attention_sin,
            expanded_atom_to_token,
        )


class EsmFold2AtomEncoder(_EsmFold2AtomReferenceEncoder):
    r"""
    Encode atom features with a sliding-window atom transformer.
    """

    def __init__(
        self,
        backend: str = "auto",
        atom_hidden_size: int = 128,
        token_hidden_size: int = 768,
        num_blocks: int = 3,
        num_attention_heads: int = 4,
        sliding_window_size: int = 128,
        expansion: int = 2,
        spatial_rope_base_frequency: float = 20.0,
        num_spatial_rope_pairs_per_axis: int = 2,
        num_reference_space_rope_pairs: int = 10,
        reference_space_rope_base_frequency: float = 10000.0,
    ) -> None:
        super().__init__(
            backend=backend,
            atom_hidden_size=atom_hidden_size,
            num_blocks=num_blocks,
            num_attention_heads=num_attention_heads,
            sliding_window_size=sliding_window_size,
            expansion=expansion,
            spatial_rope_base_frequency=spatial_rope_base_frequency,
            num_spatial_rope_pairs_per_axis=num_spatial_rope_pairs_per_axis,
            num_reference_space_rope_pairs=num_reference_space_rope_pairs,
            reference_space_rope_base_frequency=reference_space_rope_base_frequency,
        )
        self.token_projection = nn.Linear(atom_hidden_size, token_hidden_size // 2, bias=False)
        self.atom_to_token_mean = AtomToTokenMean(backend=backend)

    def encode(
        self,
        reference_atom_features: Tensor | NestedTensor,
        attention_cos: Tensor | NestedTensor,
        attention_sin: Tensor | NestedTensor,
        expanded_atom_to_token: Tensor | NestedTensor,
        token_reference: Tensor | NestedTensor,
        return_intermediates: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, list[Tensor]]:
        result = self.encoder(
            atom_query=reference_atom_features,
            atom_conditioning=reference_atom_features,
            attention_cos=attention_cos,
            attention_sin=attention_sin,
            return_intermediates=return_intermediates,
        )
        if return_intermediates:
            atom_query, intermediates = result
        else:
            atom_query = result
            intermediates = []

        atom_to_token_update = F.relu(self.token_projection(atom_query))
        token_features = self.atom_to_token_mean(
            atom_to_token_update,
            expanded_atom_to_token,
            _tokens_like_atoms(token_reference, expanded_atom_to_token),
        )
        if isinstance(token_features, NestedTensor):
            token_features = token_features.tensor

        return token_features, atom_query, reference_atom_features, intermediates

    def forward(
        self,
        reference_positions: Tensor | NestedTensor,
        reference_space_id: Tensor | NestedTensor,
        reference_charges: Tensor,
        reference_element_features: Tensor,
        reference_atom_name_character_features: Tensor,
        atom_to_token: Tensor,
        token_reference: Tensor,
        num_samples: int = 1,
        return_intermediates: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, list[Tensor]]:
        (
            reference_atom_features,
            attention_cos,
            attention_sin,
            expanded_atom_to_token,
        ) = self.encode_reference(
            reference_positions,
            reference_space_id,
            reference_charges,
            reference_element_features,
            reference_atom_name_character_features,
            atom_to_token,
            num_samples,
        )
        return self.encode(
            reference_atom_features,
            attention_cos,
            attention_sin,
            expanded_atom_to_token,
            repeat_batch(token_reference, num_samples),
            return_intermediates,
        )


class EsmFold2DenoisingAtomEncoder(_EsmFold2AtomReferenceEncoder):
    r"""
    Encode noisy atom coordinates into full-width diffusion token states.
    """

    def __init__(
        self,
        backend: str = "auto",
        atom_hidden_size: int = 128,
        token_hidden_size: int = 768,
        num_blocks: int = 3,
        num_attention_heads: int = 4,
        sliding_window_size: int = 128,
        expansion: int = 2,
        spatial_rope_base_frequency: float = 20.0,
        num_spatial_rope_pairs_per_axis: int = 2,
        num_reference_space_rope_pairs: int = 10,
        reference_space_rope_base_frequency: float = 10000.0,
    ) -> None:
        super().__init__(
            backend=backend,
            atom_hidden_size=atom_hidden_size,
            num_blocks=num_blocks,
            num_attention_heads=num_attention_heads,
            sliding_window_size=sliding_window_size,
            expansion=expansion,
            spatial_rope_base_frequency=spatial_rope_base_frequency,
            num_spatial_rope_pairs_per_axis=num_spatial_rope_pairs_per_axis,
            num_reference_space_rope_pairs=num_reference_space_rope_pairs,
            reference_space_rope_base_frequency=reference_space_rope_base_frequency,
        )
        self.coordinate_projection = nn.Linear(6, atom_hidden_size, bias=False)
        self.token_projection = nn.Linear(atom_hidden_size, token_hidden_size, bias=False)
        self.atom_to_token_mean = AtomToTokenMean(backend=backend)

    def encode(
        self,
        reference_atom_features: Tensor | NestedTensor,
        attention_cos: Tensor | NestedTensor,
        attention_sin: Tensor | NestedTensor,
        expanded_atom_to_token: Tensor | NestedTensor,
        token_reference: Tensor | NestedTensor,
        noisy_coordinates: Tensor,
        return_intermediates: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, list[Tensor]]:
        noisy_coordinates = _esmfold2_coordinates(
            noisy_coordinates,
            dtype=self.coordinate_projection.weight.dtype,
            feature="ESMFold2 encoded noisy coordinates",
        )
        coordinate_features = torch.cat([noisy_coordinates, torch.zeros_like(noisy_coordinates)], dim=-1)
        coordinate_embedding = self.coordinate_projection(
            coordinate_features.to(dtype=self.coordinate_projection.weight.dtype)
        )
        atom_query = reference_atom_features + coordinate_embedding
        result = self.encoder(
            atom_query=atom_query,
            atom_conditioning=reference_atom_features,
            attention_cos=attention_cos,
            attention_sin=attention_sin,
            return_intermediates=return_intermediates,
        )
        if return_intermediates:
            atom_query, intermediates = result
        else:
            atom_query = result
            intermediates = []
        atom_to_token_update = F.relu(self.token_projection(atom_query))
        token_features = self.atom_to_token_mean(
            atom_to_token_update,
            expanded_atom_to_token,
            _tokens_like_atoms(token_reference, expanded_atom_to_token),
        )
        if isinstance(token_features, NestedTensor):
            token_features = token_features.tensor
        return token_features, atom_query, reference_atom_features, intermediates

    def forward(
        self,
        reference_positions: Tensor | NestedTensor,
        reference_space_id: Tensor | NestedTensor,
        reference_charges: Tensor,
        reference_element_features: Tensor,
        reference_atom_name_character_features: Tensor,
        atom_to_token: Tensor,
        token_reference: Tensor,
        noisy_coordinates: Tensor,
        num_samples: int = 1,
        return_intermediates: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, list[Tensor]]:
        encoded = self.encode_reference(
            reference_positions,
            reference_space_id,
            reference_charges,
            reference_element_features,
            reference_atom_name_character_features,
            atom_to_token,
            num_samples,
        )
        return self.encode(
            *encoded,
            repeat_batch(token_reference, num_samples),
            noisy_coordinates,
            return_intermediates,
        )


class EsmFold2AtomDecoder(nn.Module):
    r"""
    Decode token features into atom-coordinate updates.
    """

    def __init__(
        self,
        backend: str = "auto",
        atom_hidden_size: int = 128,
        token_hidden_size: int = 768,
        num_blocks: int = 3,
        num_attention_heads: int = 4,
        sliding_window_size: int = 128,
        expansion: int = 2,
        spatial_rope_base_frequency: float = 20.0,
        num_spatial_rope_pairs_per_axis: int = 2,
        num_reference_space_rope_pairs: int = 10,
        reference_space_rope_base_frequency: float = 10000.0,
    ) -> None:
        super().__init__()
        self.token_projection = nn.Linear(token_hidden_size, atom_hidden_size, bias=False)
        self.token_to_atom_broadcast = TokenToAtomBroadcast(backend=backend)

        self.encoder = EsmFold2AtomTransformerEncoder(
            backend=backend,
            atom_hidden_size=atom_hidden_size,
            num_blocks=num_blocks,
            num_attention_heads=num_attention_heads,
            sliding_window_size=sliding_window_size,
            expansion=expansion,
            spatial_rope_base_frequency=spatial_rope_base_frequency,
            num_spatial_rope_pairs_per_axis=num_spatial_rope_pairs_per_axis,
            num_reference_space_rope_pairs=num_reference_space_rope_pairs,
            reference_space_rope_base_frequency=reference_space_rope_base_frequency,
        )

        self.output_norm = LayerNorm(atom_hidden_size, backend=backend)
        self.coordinate_projection = nn.Linear(atom_hidden_size, _COORDINATE_SIZE, bias=False)

    def forward(
        self,
        token: Tensor,
        atom_query: Tensor,
        atom_conditioning: Tensor,
        attention_cos: Tensor | NestedTensor,
        attention_sin: Tensor | NestedTensor,
        expanded_atom_to_token: Tensor | NestedTensor,
        return_intermediates: bool = False,
    ) -> tuple[Tensor | NestedTensor, list[Tensor | NestedTensor]]:
        token_to_atom_update = self.token_projection(token)
        token_to_atom_update = self.token_to_atom_broadcast(
            _tokens_like_atoms(token_to_atom_update, expanded_atom_to_token),
            expanded_atom_to_token,
        )
        atom_query = atom_query + token_to_atom_update

        result = self.encoder(
            atom_query=atom_query,
            atom_conditioning=atom_conditioning,
            attention_cos=attention_cos,
            attention_sin=attention_sin,
            return_intermediates=return_intermediates,
        )
        if return_intermediates:
            atom_query, intermediates = result
        else:
            atom_query = result
            intermediates = []

        coordinate_update = self.coordinate_projection(self.output_norm(atom_query))
        return coordinate_update, intermediates


class EsmFold2AtomTransformerEncoder(nn.Module):
    r"""
    Atom Transformer encoder composed of [`EsmFold2AtomBlock`] modules.
    """

    _gradient_checkpointing_func: Callable[..., Tensor]

    def __init__(
        self,
        backend: str = "auto",
        atom_hidden_size: int = 128,
        num_blocks: int = 3,
        num_attention_heads: int = 4,
        sliding_window_size: int = 128,
        expansion: int = 2,
        spatial_rope_base_frequency: float = 20.0,
        num_spatial_rope_pairs_per_axis: int = 2,
        num_reference_space_rope_pairs: int = 10,
        reference_space_rope_base_frequency: float = 10000.0,
    ) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.gradient_checkpointing = False
        self.head_size = atom_hidden_size // num_attention_heads
        self.spatial_rope_base_frequency = spatial_rope_base_frequency
        self.num_spatial_rope_pairs_per_axis = num_spatial_rope_pairs_per_axis
        self.num_reference_space_rope_pairs = num_reference_space_rope_pairs
        self.reference_space_rope_base_frequency = reference_space_rope_base_frequency

        self.blocks = nn.ModuleList(
            [
                EsmFold2AtomBlock(
                    atom_hidden_size=atom_hidden_size,
                    num_attention_heads=num_attention_heads,
                    backend=backend,
                    half_window=sliding_window_size // 2,
                    expansion=expansion,
                )
                for _ in range(num_blocks)
            ]
        )

    def build_3d_rope(self, reference_positions: Tensor, reference_space_id: Tensor) -> tuple[Tensor, Tensor]:
        reference_positions = _esmfold2_coordinates(reference_positions, feature="ESMFold2 spatial references")
        return _build_3d_rope(
            reference_positions=reference_positions,
            reference_space_id=reference_space_id,
            head_size=self.head_size,
            num_spatial_pairs_per_axis=self.num_spatial_rope_pairs_per_axis,
            num_reference_space_pairs=self.num_reference_space_rope_pairs,
            spatial_base_frequency=self.spatial_rope_base_frequency,
            reference_space_base_frequency=self.reference_space_rope_base_frequency,
        )

    def forward(
        self,
        atom_query: Tensor | NestedTensor,
        atom_conditioning: Tensor | NestedTensor,
        attention_cos: Tensor | NestedTensor,
        attention_sin: Tensor | NestedTensor,
        return_intermediates: bool = False,
    ) -> Tensor | NestedTensor | tuple[Tensor | NestedTensor, list[Tensor | NestedTensor]]:
        intermediates: list[Tensor | NestedTensor] = []
        for block in self.blocks:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                atom_query = self._gradient_checkpointing_func(
                    block.__call__,
                    atom_query,
                    atom_conditioning,
                    attention_cos,
                    attention_sin,
                )
            else:
                atom_query = block(atom_query, atom_conditioning, attention_cos, attention_sin)
            if return_intermediates:
                intermediates.append(atom_query)
        if return_intermediates:
            return atom_query, intermediates
        return atom_query


class _PreparedEsmFold2DenoisingContext(NamedTuple):
    r"""Step-independent pair and reference features for one denoising rollout."""

    conditioned_pair: Tensor
    reference_atom_features: Tensor
    attention_cos: Tensor
    attention_sin: Tensor
    expanded_atom_to_token: Tensor


class EsmFold2Denoiser(nn.Module):
    r"""
    Denoise atom coordinates from conditioned token and pair features.
    """

    def __init__(
        self,
        backend: str = "auto",
        atom_hidden_size: int = 128,
        token_hidden_size: int = 768,
        pair_size: int = 256,
        input_size: int = 461,
        sigma_data: float = 16.0,
        fourier_embedding_size: int = 256,
        num_atom_blocks: int = 3,
        num_atom_attention_heads: int = 4,
        num_token_encoder_blocks: int = 12,
        num_token_encoder_attention_heads: int = 16,
        expansion: int = 2,
        sliding_window_size: int = 128,
        spatial_rope_base_frequency: float = 20.0,
        num_spatial_rope_pairs_per_axis: int = 2,
        num_reference_space_rope_pairs: int = 10,
        reference_space_rope_base_frequency: float = 10000.0,
    ) -> None:
        super().__init__()
        self.conditioning = EsmFold2DiffusionConditioning(
            backend=backend,
            pair_size=pair_size,
            single_size=token_hidden_size,  # conditioning s output is token_hidden_size
            input_size=input_size,
            sigma_data=sigma_data,
            fourier_embedding_size=fourier_embedding_size,
            expansion=expansion,
        )
        self.preconditioner = EdmPreconditioner(sigma_data)
        self.atom_encoder = EsmFold2DenoisingAtomEncoder(
            backend=backend,
            atom_hidden_size=atom_hidden_size,
            token_hidden_size=token_hidden_size,
            num_blocks=num_atom_blocks,
            num_attention_heads=num_atom_attention_heads,
            sliding_window_size=sliding_window_size,
            expansion=2,
            spatial_rope_base_frequency=spatial_rope_base_frequency,
            num_spatial_rope_pairs_per_axis=num_spatial_rope_pairs_per_axis,
            num_reference_space_rope_pairs=num_reference_space_rope_pairs,
            reference_space_rope_base_frequency=reference_space_rope_base_frequency,
        )

        self.single_norm = LayerNorm(token_hidden_size, backend=backend)
        self.single_projection = nn.Linear(token_hidden_size, token_hidden_size, bias=False)
        nn.init.zeros_(self.single_projection.weight)
        self.token_encoder = EsmFold2DiffusionTransformerEncoder(
            hidden_size=token_hidden_size,
            pair_size=pair_size,
            num_attention_heads=num_token_encoder_attention_heads,
            num_blocks=num_token_encoder_blocks,
            conditioning_size=token_hidden_size,
            expansion=expansion,
            backend=backend,
        )
        self.token_norm = LayerNorm(token_hidden_size, backend=backend)
        self.atom_decoder = EsmFold2AtomDecoder(
            backend=backend,
            atom_hidden_size=atom_hidden_size,
            token_hidden_size=token_hidden_size,
            num_blocks=num_atom_blocks,
            num_attention_heads=num_atom_attention_heads,
            sliding_window_size=sliding_window_size,
            expansion=2,
            spatial_rope_base_frequency=spatial_rope_base_frequency,
            num_spatial_rope_pairs_per_axis=num_spatial_rope_pairs_per_axis,
            num_reference_space_rope_pairs=num_reference_space_rope_pairs,
            reference_space_rope_base_frequency=reference_space_rope_base_frequency,
        )

    def denoise(
        self,
        *,
        noisy_coordinates: Tensor,
        noise_level: Tensor,
        single_inputs: Tensor,
        conditioned_pair: Tensor,
        reference_atom_features: Tensor,
        attention_cos: Tensor,
        attention_sin: Tensor,
        expanded_atom_to_token: Tensor,
        num_samples: int = 1,
        return_token_representations: bool = False,
        return_atom_representations: bool = False,
    ) -> dict[str, Tensor | None]:
        noisy_coordinates = _esmfold2_coordinates(
            noisy_coordinates,
            dtype=self.atom_encoder.coordinate_projection.weight.dtype,
            feature="ESMFold2 denoising coordinates",
        )
        noise_level = noise_level.to(device=noisy_coordinates.device).reshape(-1)
        conditioning = self.conditioning.condition_single(
            noise_level,
            single_inputs,
            num_samples,
        )

        scaled_noisy_coordinates = self.preconditioner.c_in(noise_level)[:, None, None] * noisy_coordinates

        token, atom_query_skip, atom_conditioning_skip, encoder_intermediates = self.atom_encoder.encode(
            reference_atom_features=reference_atom_features,
            attention_cos=attention_cos,
            attention_sin=attention_sin,
            expanded_atom_to_token=expanded_atom_to_token,
            token_reference=repeat_batch(single_inputs, num_samples),
            noisy_coordinates=scaled_noisy_coordinates,
            return_intermediates=return_atom_representations,
        )

        token = token + self.single_projection(self.single_norm(conditioning))
        token, _ = self.token_encoder(
            token,
            conditioning,
            conditioned_pair,
            num_samples=num_samples,
        )
        token = self.token_norm(token)

        coordinate_update, decoder_intermediates = self.atom_decoder(
            token=token,
            atom_query=atom_query_skip,
            atom_conditioning=atom_conditioning_skip,
            attention_cos=attention_cos,
            attention_sin=attention_sin,
            expanded_atom_to_token=expanded_atom_to_token,
            return_intermediates=return_atom_representations,
        )

        denoised_coordinates = (
            self.preconditioner.c_skip(noise_level)[:, None, None] * noisy_coordinates
            + self.preconditioner.c_out(noise_level)[:, None, None] * coordinate_update
        )

        atom_intermediates = (
            torch.stack(encoder_intermediates + decoder_intermediates, dim=2) if return_atom_representations else None
        )

        return {
            "denoised_coordinates": denoised_coordinates,
            "token_representations": token if return_token_representations else None,
            "atom_intermediates": atom_intermediates,
        }

    def forward(
        self,
        noisy_coordinates: Tensor,
        noise_level: Tensor,
        reference_positions: Tensor,
        reference_charges: Tensor,
        reference_element_features: Tensor,
        reference_atom_name_character_features: Tensor,
        reference_space_id: Tensor,
        atom_to_token: Tensor,
        single_inputs: Tensor,
        pair: Tensor,
        relative_position_encoding: Tensor,
        num_samples: int = 1,
        return_token_representations: bool = False,
        return_atom_representations: bool = False,
        *,
        prepared: _PreparedEsmFold2DenoisingContext | None = None,
    ) -> dict[str, Tensor | None]:
        if prepared is None:
            prepared = _PreparedEsmFold2DenoisingContext(
                self.conditioning.condition_pair(pair, relative_position_encoding),
                *self.atom_encoder.encode_reference(
                    reference_positions,
                    reference_space_id,
                    reference_charges,
                    reference_element_features,
                    reference_atom_name_character_features,
                    atom_to_token,
                    num_samples,
                ),
            )
        return self.denoise(
            noisy_coordinates=noisy_coordinates,
            noise_level=noise_level,
            single_inputs=single_inputs,
            conditioned_pair=prepared.conditioned_pair,
            reference_atom_features=prepared.reference_atom_features,
            attention_cos=prepared.attention_cos,
            attention_sin=prepared.attention_sin,
            expanded_atom_to_token=prepared.expanded_atom_to_token,
            num_samples=num_samples,
            return_token_representations=return_token_representations,
            return_atom_representations=return_atom_representations,
        )


class EsmFold2DiffusionTransformerEncoder(nn.Module):
    r"""
    Diffusion denoising Transformer encoder with attention pair bias.
    """

    _gradient_checkpointing_func: Callable[..., Tensor]

    def __init__(
        self,
        hidden_size: int,
        pair_size: int,
        num_attention_heads: int,
        num_blocks: int,
        conditioning_size: int,
        expansion: int = 2,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.gradient_checkpointing = False
        self.blocks = nn.ModuleList(
            EsmFold2DiffusionTransformerEncoderBlock(
                hidden_size=hidden_size,
                pair_size=pair_size,
                num_attention_heads=num_attention_heads,
                conditioning_size=conditioning_size,
                expansion=expansion,
                backend=backend,
            )
            for _ in range(num_blocks)
        )

    def forward(
        self,
        hidden: Tensor,
        conditioning: Tensor,
        pair: Tensor,
        num_samples: int = 1,
        return_intermediates: bool = False,
    ) -> tuple[Tensor, list[Tensor]]:
        intermediates: list[Tensor] = []
        for block in self.blocks:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                hidden = self._gradient_checkpointing_func(
                    block.__call__,
                    hidden,
                    conditioning,
                    pair,
                    num_samples=num_samples,
                )
            else:
                hidden = block(
                    hidden,
                    conditioning,
                    pair,
                    num_samples=num_samples,
                )
            if return_intermediates:
                intermediates.append(hidden)
        return hidden, intermediates


class EsmFold2FoldingBlock(nn.Module):
    r"""
    Outgoing and incoming triangle multiplication followed by a transition.
    """

    def __init__(self, backend: str = "auto", pair_size: int = 256, expansion: int = 4) -> None:
        super().__init__()
        self.triangle_multiplication_outgoing = TriangleMultiplication(
            pair_size,
            pair_size,
            direction="outgoing",
            backend=backend,
        )
        self.triangle_multiplication_incoming = TriangleMultiplication(
            pair_size,
            pair_size,
            direction="incoming",
            backend=backend,
        )
        self.transition = Transition(pair_size, expansion=expansion, backend=backend)

    def forward(self, pair: Tensor, pair_attention_mask: Tensor | None = None) -> Tensor:
        pair = pair + self.triangle_multiplication_outgoing(pair, pair_mask=pair_attention_mask)
        pair = pair + self.triangle_multiplication_incoming(pair, pair_mask=pair_attention_mask)
        return pair + self.transition(pair)


class EsmFold2MsaEncoderBlock(nn.Module):
    r"""
    One MSA encoder block: OPM into pair, MSA pair-weighted averaging, triangle update.
    """

    def __init__(
        self,
        msa_size: int,
        pair_size: int,
        hidden_size: int,
        num_msa_attention_heads: int,
        msa_attention_head_size: int,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.outer_product_mean = EsmFold2OuterProductMean(msa_size, hidden_size, pair_size, backend)
        self.msa_pair_weighted_averaging = MsaPairWeightedAveraging(
            msa_size=msa_size,
            pair_size=pair_size,
            num_attention_heads=num_msa_attention_heads,
            head_size=msa_attention_head_size,
            backend=backend,
        )
        self.msa_transition = Transition(msa_size, expansion=4, backend=backend)
        self.triangle_multiplication_outgoing = TriangleMultiplication(
            pair_size,
            pair_size,
            direction="outgoing",
            backend=backend,
        )
        self.triangle_multiplication_incoming = TriangleMultiplication(
            pair_size,
            pair_size,
            direction="incoming",
            backend=backend,
        )
        self.pair_transition = Transition(pair_size, expansion=4, backend=backend)

    def forward(
        self,
        m: Tensor,
        pair: Tensor,
        msa_mask: Tensor,
        pair_attention_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        pair = pair + self.outer_product_mean(m, msa_mask)
        m = m + self.msa_pair_weighted_averaging(
            m.transpose(1, 2),
            pair,
            pair_attention_mask,
        ).transpose(1, 2)
        m = m + self.msa_transition(m)
        pair = pair + self.triangle_multiplication_outgoing(pair, pair_mask=pair_attention_mask)
        pair = pair + self.triangle_multiplication_incoming(pair, pair_mask=pair_attention_mask)
        pair = pair + self.pair_transition(pair)
        return m, pair


class EsmFold2ExperimentalMsaEncoderBlock(nn.Module):
    r"""
    Run an MSA update followed by outer-product-mean and triangle updates.
    """

    def __init__(
        self,
        msa_size: int,
        pair_size: int,
        hidden_size: int,
        num_msa_attention_heads: int,
        msa_attention_head_size: int,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.msa_pair_weighted_averaging = MsaPairWeightedAveraging(
            msa_size=msa_size,
            pair_size=pair_size,
            num_attention_heads=num_msa_attention_heads,
            head_size=msa_attention_head_size,
            backend=backend,
        )
        self.msa_transition = Transition(msa_size, expansion=4, backend=backend)
        self.outer_product_mean = OuterProductMean(msa_size, hidden_size, pair_size, backend)
        self.triangle_multiplication_outgoing = TriangleMultiplication(
            pair_size,
            pair_size,
            direction="outgoing",
            backend=backend,
        )
        self.triangle_multiplication_incoming = TriangleMultiplication(
            pair_size,
            pair_size,
            direction="incoming",
            backend=backend,
        )
        self.pair_transition = Transition(pair_size, expansion=4, backend=backend)

    def forward(
        self,
        m: Tensor,
        pair: Tensor,
        msa_mask: Tensor,
        pair_attention_mask: Tensor,
        msa_track_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        mask4d = msa_track_mask[:, None, None, None].to(m.dtype) if msa_track_mask is not None else None

        def masked(x: Tensor) -> Tensor:
            return x * mask4d if mask4d is not None else x

        m = m + masked(
            self.msa_pair_weighted_averaging(
                m.transpose(1, 2),
                pair,
                pair_attention_mask,
            ).transpose(1, 2)
        )
        m = m + masked(self.msa_transition(m))
        # The shared outer-product mean is sequence-major; this block's MSA is token-major.
        pair = pair + masked(self.outer_product_mean(m.transpose(1, 2), msa_mask.transpose(1, 2)))
        pair = pair + masked(self.triangle_multiplication_outgoing(pair, pair_mask=pair_attention_mask))
        pair = pair + masked(self.triangle_multiplication_incoming(pair, pair_mask=pair_attention_mask))
        pair = pair + masked(self.pair_transition(pair))
        return m, pair


class EsmFold2AtomBlock(nn.Module):
    r"""
    Adaptive RMS normalization, sliding-window attention, and transition updates.
    """

    def __init__(
        self,
        atom_hidden_size: int,
        num_attention_heads: int,
        backend: str = "auto",
        half_window: int = 64,
        expansion: int = 2,
    ) -> None:
        super().__init__()
        self.adaptive_norm_activation = nn.SiLU()
        self.adaptive_norm_projection = nn.Linear(atom_hidden_size, 6 * atom_hidden_size, bias=False)
        nn.init.zeros_(self.adaptive_norm_projection.weight)
        self.attention = EsmFold2SpatialAtomAttention(
            atom_hidden_size, num_attention_heads, backend, half_window=half_window
        )
        hidden_size = ((expansion * (atom_hidden_size // 3) * 2) + 255) // 256 * 256
        self.transition = SwiGluFeedForward(atom_hidden_size, hidden_size)

    def forward(
        self,
        x: Tensor | NestedTensor,
        atom_conditioning: Tensor | NestedTensor,
        attention_cos: Tensor | NestedTensor,
        attention_sin: Tensor | NestedTensor,
    ) -> Tensor | NestedTensor:
        mod = self.adaptive_norm_projection(self.adaptive_norm_activation(atom_conditioning))
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = mod.chunk(6, dim=-1)

        attn_input = F.rms_norm(x, (x.shape[-1],)) * (1 + scale_a) + shift_a
        attention_update = self.attention(attn_input, attention_cos, attention_sin)
        x = x + gate_a * attention_update

        transition_input = F.rms_norm(x, (x.shape[-1],)) * (1 + scale_f) + shift_f
        transition_update = self.transition(transition_input)
        return x + gate_f * transition_update


class EsmFold2DiffusionTransformerEncoderBlock(nn.Module):
    r"""
    One attention and transition residual unit in the diffusion Transformer encoder.
    """

    def __init__(
        self,
        hidden_size: int,
        pair_size: int,
        num_attention_heads: int,
        conditioning_size: int,
        expansion: int,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.conditioned_norm = AdaptiveLayerNorm(hidden_size, conditioning_size, backend=backend)
        self.pair_norm = LayerNorm(pair_size, eps=1e-5, backend=backend)
        self.pair_bias_projection = nn.Linear(pair_size, num_attention_heads, bias=False)
        self.attention = MultiheadAttention(
            GatedMultiheadAttentionProjection(hidden_size, num_attention_heads=num_attention_heads, query_bias=True),
            backend=backend,
        )
        self.output_gate_projection = nn.Linear(conditioning_size, hidden_size)
        nn.init.zeros_(self.output_gate_projection.weight)
        nn.init.constant_(self.output_gate_projection.bias, -2.0)
        self.transition = ConditionedTransition(
            hidden_size,
            conditioning_size,
            expansion,
            backend=backend,
        )
        nn.init.zeros_(self.transition.conditioning_gate_projection.weight)
        nn.init.constant_(self.transition.conditioning_gate_projection.bias, -2.0)

    def forward(
        self,
        hidden: Tensor,
        conditioning: Tensor,
        pair: Tensor,
        num_samples: int = 1,
    ) -> Tensor:
        pair = repeat_batch(pair, num_samples)
        bias = self.pair_bias_projection(self.pair_norm(pair))
        return conditioned_transformer_block(
            hidden,
            conditioning,
            bias,
            norm=self.conditioned_norm,
            attention=self.attention,
            output_gate_projection=self.output_gate_projection,
            transition=self.transition,
        )


class EsmFold2OuterProductMean(PostProjectionOuterProductMean):
    r"""
    Outer-product mean: maps a token-major MSA representation into a pair update.

    Adapts ESMFold2's token-major MSA layout (``[B, N_token, N_seq, msa_size]``) to the
    shared outer-product-mean operation. The mask-weighted mean is taken *after* the affine
    output projection, so the projection bias is divided by the clamped valid-row count.
    """

    def __init__(
        self,
        msa_size: int,
        hidden_size: int,
        pair_size: int,
        backend: str = "auto",
    ) -> None:
        super().__init__(msa_size, hidden_size, pair_size, eps=0.0, backend=backend, clamp_denominator=True)

    def forward(self, msa: Tensor, msa_mask: Tensor | None = None) -> Tensor:
        r"""
        Project the masked outer-product sum, then divide by the clamped valid-row count.

        Args:
            msa: MSA representation `[B, N_token, N_seq, msa_size]`.
            msa_mask: Optional MSA validity mask `[B, N_token, N_seq]`.

        Returns:
            The pair contribution `[B, N_token, N_token, pair_size]`.
        """

        return super().forward(
            msa.transpose(-3, -2),
            None if msa_mask is None else msa_mask.transpose(-2, -1),
        )


_NONPOLYMER_ID = 3


class EsmFold2SpatialAtomAttention(nn.Module):
    r"""
    Sliding-window self-attention with three-dimensional RoPE.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        backend: str = "auto",
        half_window: int = 64,
    ) -> None:
        super().__init__()
        self.backend = backend
        self.num_attention_heads = num_attention_heads
        self.head_size = hidden_size // num_attention_heads
        self.scale = self.head_size**-0.5
        self.half_window = half_window

        self.query_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.key_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.value_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.gate_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.output_projection = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(
        self,
        x: Tensor | NestedTensor,
        attention_cos: Tensor | NestedTensor,
        attention_sin: Tensor | NestedTensor,
    ) -> Tensor | NestedTensor:
        r"""
        Attend each atom to the atoms within ``half_window`` positions of it.

        The atom axis carries no padding, so the window is a band over positions: a dense atom axis
        is entirely valid and a nested one is bounded by each batch element's own length.

        Args:
            x: Atom hidden states `[..., N_atom, hidden_size]`.
            attention_cos: Three-dimensional RoPE cosines `[..., N_atom, head_size / 2]`.
            attention_sin: Three-dimensional RoPE sines `[..., N_atom, head_size / 2]`.

        Returns:
            Gated attention update `[..., N_atom, hidden_size]`.
        """

        heads = (self.num_attention_heads, self.head_size)
        query = _qk_norm(self.query_projection(x).unflatten(-1, heads))
        key = _qk_norm(self.key_projection(x).unflatten(-1, heads))
        value = self.value_projection(x).unflatten(-1, heads)

        query = _apply_rotary_emb_3d(query, attention_cos, attention_sin)
        key = _apply_rotary_emb_3d(key, attention_cos, attention_sin)

        output = projected_attention(
            query.transpose(-2, -3),
            key.transpose(-2, -3),
            value.transpose(-2, -3),
            attention_mask=banded_pair_mask(x[..., 0], self.half_window),
            backend=self.backend,
            scale=self.scale,
        ).transpose(-2, -3)
        output = output.flatten(-2) * torch.sigmoid(self.gate_projection(x))
        return self.output_projection(output)


class EsmFold2DiffusionConditioning(nn.Module):
    r"""
    Conditions pair and single representations on noise timestep.
    """

    token_feature_map: Tensor

    def __init__(
        self,
        backend: str = "auto",
        pair_size: int = 256,
        single_size: int = 768,
        input_size: int = 461,
        sigma_data: float = 16.0,
        fourier_embedding_size: int = 256,
        expansion: int = 2,
    ) -> None:
        super().__init__()
        self.sigma_data = float(sigma_data)

        self.pair_input_norm = LayerNorm(2 * pair_size, eps=_EPS, backend=backend)
        self.pair_projection = nn.Linear(2 * pair_size, pair_size, bias=False)
        self.pair_transitions = nn.ModuleList(
            Transition(
                pair_size,
                expansion=expansion,
                backend=backend,
            )
            for _ in range(2)
        )

        self.encoded_size = input_size - 2 * len(TOKENS) - 1
        feature_map = folding_source_map()
        self.register_buffer("token_feature_map", feature_map, persistent=False)
        source_input_size = self.encoded_size + 2 * feature_map.shape[0] + 1
        self.single_input_norm = LayerNorm(source_input_size, eps=_EPS, backend=backend)
        self.single_projection = nn.Linear(source_input_size, single_size, bias=False)
        self.noise_embedding = FrozenFourierEmbedding(fourier_embedding_size)
        self.noise_norm = LayerNorm(fourier_embedding_size, eps=_EPS, backend=backend)
        self.noise_projection = nn.Linear(fourier_embedding_size, single_size, bias=False)
        self.single_transitions = nn.ModuleList(
            Transition(
                single_size,
                expansion=expansion,
                backend=backend,
            )
            for _ in range(2)
        )

    def condition_pair(
        self,
        pair: Tensor,
        relative_position_encoding: Tensor,
    ) -> Tensor:
        dtype = self.pair_projection.weight.dtype
        relative_position_encoding = relative_position_encoding.to(dtype=dtype)
        return condition_pair(
            pair.to(dtype=dtype),
            relative_position_encoding,
            norm=self.pair_input_norm,
            projection=self.pair_projection,
            transitions=self.pair_transitions,
        )

    def condition_single(
        self,
        noise_level: Tensor,
        single_inputs: Tensor,
        num_samples: int = 1,
    ) -> Tensor:
        source_inputs = map_token_feature_planes(single_inputs, self.token_feature_map, self.encoded_size)
        expanded_single_inputs = repeat_batch(source_inputs, num_samples)
        conditioned_single = self.single_projection(
            self.single_input_norm(expanded_single_inputs.to(dtype=self.single_projection.weight.dtype))
        )

        noise_level = noise_level.to(device=conditioned_single.device).reshape(-1)
        embedded_noise_level = 0.25 * torch.log((noise_level / self.sigma_data).clamp(min=1e-20))
        fourier_weight = self.noise_embedding.projection.weight
        noise_embedding = self.noise_embedding(embedded_noise_level.to(fourier_weight).reshape(-1))
        noise_embedding = self.noise_projection(self.noise_norm(noise_embedding))
        conditioned_single = conditioned_single + noise_embedding.unsqueeze(1)

        for transition in self.single_transitions:
            conditioned_single = conditioned_single + transition(conditioned_single)
        return conditioned_single

    def forward(
        self,
        noise_level: Tensor,
        single_inputs: Tensor,
        pair: Tensor,
        relative_position_encoding: Tensor,
        num_samples: int = 1,
    ) -> tuple[Tensor, Tensor]:
        conditioned_pair = self.condition_pair(pair, relative_position_encoding)
        conditioned_single = self.condition_single(
            noise_level,
            single_inputs,
            num_samples,
        )
        return conditioned_single, conditioned_pair


class EsmFold2RowAttentionPooling(nn.Module):
    r"""
    Pool each pair row into a single representation.
    """

    def __init__(self, pair_size: int, single_size: int) -> None:
        super().__init__()
        self.attention_projection = nn.Linear(pair_size, 1, bias=False)
        self.output_projection = nn.Linear(pair_size, single_size, bias=False)

    def forward(self, pair: Tensor) -> Tensor:
        scores = self.attention_projection(pair).squeeze(-1)
        weights = F.softmax(scores, dim=-1)
        pooled = torch.einsum("bnm,bnmd->bnd", weights, pair)
        return self.output_projection(pooled)


class EsmFold2EdmSampler(CoordinateEdmSampler):
    r"""
    ESMFold2 EDM sampler: the noisy coordinates are rigidly realigned onto the denoiser output
    before each solver step.

    The released ESMFold2 sampler owns a truncated schedule, a convergence early exit, and a
    quaternion rigid augmentation that co-transforms the previous denoised coordinates, so
    [`EsmFold2DiffusionStructureHead`][ofoldx.models.esmfold2.model.EsmFold2DiffusionStructureHead]
    keeps the noise schedule, drives the loop, and calls
    [`step`][ofoldx.models.esmfold2.model.EsmFold2EdmSampler.step] for the shared churn,
    noise-injection, alignment, and Euler arithmetic.
    """

    def step(
        self,
        denoise: StepDenoiseFn,
        coords: Tensor,
        sigma_prev: Tensor,
        sigma_next: Tensor,
        *,
        noise: Tensor,
        noise_scale: float,
        step_scale: float,
    ) -> Tensor:
        r"""
        Advance one reverse-diffusion step for a caller that owns the loop.

        Args:
            denoise: Denoiser called with the noisy coordinates and their noise levels.
            coords: Current coordinates with the sample axis folded into the batch axis.
            sigma_prev: Noise level at the start of the step.
            sigma_next: Noise level at the end of the step.
            noise: Pre-drawn churn noise shaped like ``coords``.
            noise_scale: Churn-noise multiplier for this call.
            step_scale: Solver-step multiplier for this call.

        Returns:
            The advanced coordinates.
        """
        coords = _esmfold2_coordinates(coords, feature="ESMFold2 sampler coordinates")

        return self._step(
            denoise,
            coords,
            sigma_prev,
            sigma_next,
            noise=noise,
            churn_rate=self.churn_rate,
            noise_scale=noise_scale,
            step_scale=step_scale,
        )

    def _align(
        self,
        noisy_coords: Tensor,
        denoised_coords: Tensor,
        *,
        atom_mask: Tensor | None = None,
    ) -> Tensor:
        r"""
        Rigidly align the noisy coordinates onto the denoiser output.

        Args:
            noisy_coords: Churn-noised coordinates.
            denoised_coords: Denoiser output with the same shape.
            atom_mask: Unused. The atom axis carries no padding, so every atom weighs the same in
                the fit; a ragged batch weighs each element's own atoms through its own topology.

        Returns:
            The aligned noisy coordinates.
        """

        del atom_mask
        return weighted_rigid_align(
            source=noisy_coords,
            target=denoised_coords,
            weight=torch.ones_like(noisy_coords[..., 0]),
            backend=self.backend,
        )


class EsmFold2DiffusionStructureHead(nn.Module):
    r"""
    Diffusion score network and coordinate sampler.
    """

    def __init__(
        self,
        config: EsmFold2Config | EsmFold2ExperimentalConfig,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.config = config
        self.backend = backend
        denoiser_config = config.diffusion_structure_head.denoiser
        atom_encoder_config = config.input_embeddings.atom_encoder
        structure_head_config = config.diffusion_structure_head

        self.denoiser = EsmFold2Denoiser(
            backend=backend,
            atom_hidden_size=denoiser_config.atom_hidden_size,
            token_hidden_size=denoiser_config.token_hidden_size,
            pair_size=denoiser_config.pair_size,
            input_size=denoiser_config.input_size,
            sigma_data=denoiser_config.sigma_data,
            fourier_embedding_size=denoiser_config.fourier_embedding_size,
            num_atom_blocks=denoiser_config.num_atom_blocks,
            num_atom_attention_heads=denoiser_config.num_atom_attention_heads,
            num_token_encoder_blocks=denoiser_config.num_token_encoder_blocks,
            num_token_encoder_attention_heads=denoiser_config.num_token_encoder_attention_heads,
            expansion=denoiser_config.expansion,
            sliding_window_size=atom_encoder_config.sliding_window_size,
            spatial_rope_base_frequency=atom_encoder_config.spatial_rope_base_frequency,
            num_spatial_rope_pairs_per_axis=atom_encoder_config.num_spatial_rope_pairs_per_axis,
            num_reference_space_rope_pairs=atom_encoder_config.num_reference_space_rope_pairs,
            reference_space_rope_base_frequency=atom_encoder_config.reference_space_rope_base_frequency,
        )

        # Sampling hyperparameters
        self.sigma_data = denoiser_config.sigma_data
        self.churn_rate = structure_head_config.churn_rate
        self.min_churn_noise_level = structure_head_config.min_churn_noise_level
        self.noise_scale = structure_head_config.noise_scale
        self.step_scale = structure_head_config.step_scale
        self.max_noise_level = structure_head_config.max_noise_level
        self.min_noise_level = structure_head_config.min_noise_level
        self.rho = structure_head_config.rho
        self.num_sampling_steps = structure_head_config.num_sampling_steps
        self.sampler = EsmFold2EdmSampler(
            churn_rate=structure_head_config.churn_rate,
            min_churn_noise_level=structure_head_config.min_churn_noise_level,
            noise_scale=structure_head_config.noise_scale,
            step_scale=structure_head_config.step_scale,
            backend=backend,
        )

    def inference_noise_schedule(
        self,
        num_sampling_steps: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        num_sampling_steps = self.num_sampling_steps if num_sampling_steps is None else num_sampling_steps
        dtype = self.denoiser.atom_decoder.coordinate_projection.weight.dtype if dtype is None else dtype
        if num_sampling_steps == 1:
            # The released ESMFold2 schedule returns the endpoint directly for a single step rather
            # than evaluating the ramp, so the first noise level is exact here and carries the
            # ``(max_noise_level ** (1 / rho)) ** rho`` rounding step at every other step count.
            return torch.tensor(
                [self.max_noise_level * self.sigma_data, 0.0],
                device=device,
                dtype=dtype,
            )
        return karras_noise_schedule(
            num_sampling_steps,
            max_noise_level=self.max_noise_level,
            min_noise_level=self.min_noise_level,
            rho=self.rho,
            sigma_data=self.sigma_data,
            device=device,
            dtype=dtype,
        )

    @staticmethod
    def _random_rotations(
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: torch.Generator | None,
    ) -> Tensor:
        quaternion = torch.randn((batch_size, 4), dtype=dtype, device=device, generator=generator)
        return quaternion_to_rotation(quaternion)

    def center_random_augmentation(
        self,
        coordinates: Tensor,
        second_coordinates: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        coordinates = _esmfold2_coordinates(coordinates, feature="ESMFold2 augmentation coordinates")
        if second_coordinates is not None:
            second_coordinates = _esmfold2_coordinates(
                second_coordinates, feature="ESMFold2 paired augmentation coordinates"
            )
        batch_size = coordinates.shape[0]
        center = coordinates.mean(dim=-2, keepdim=True)
        coordinates = coordinates - center
        if second_coordinates is not None:
            second_coordinates = second_coordinates - center

        rotation = self._random_rotations(batch_size, coordinates.dtype, coordinates.device, generator)
        coordinates = torch.einsum("bmd,bds->bms", coordinates, rotation)
        if second_coordinates is not None:
            second_coordinates = torch.einsum("bmd,bds->bms", second_coordinates, rotation)

        translation = torch.randn(
            coordinates.shape[0],
            1,
            coordinates.shape[-1],
            dtype=coordinates.dtype,
            device=coordinates.device,
            generator=generator,
        )
        coordinates = coordinates + translation
        if second_coordinates is not None:
            second_coordinates = second_coordinates + translation
        return coordinates, second_coordinates

    def sample(
        self,
        pair: Tensor,
        single_inputs: Tensor,
        relative_position_encoding: Tensor,
        reference_positions: Tensor,
        reference_charges: Tensor,
        reference_element_features: Tensor,
        reference_atom_name_character_features: Tensor,
        reference_space_id: Tensor,
        atom_to_token: Tensor,
        num_samples: int = 1,
        num_sampling_steps: int | None = None,
        max_inference_sigma: float | None | EllipsisType = ...,
        noise_scale: float | None = None,
        step_scale: float | None = None,
        return_atom_representations: bool = False,
        denoising_early_exit_rmsd: float | None = None,
        generator: torch.Generator | None = None,
    ) -> dict[str, Tensor | None]:
        if num_samples <= 0:
            fail("value_invalid", RequestRef("num_samples", str(num_samples)))
        device = single_inputs.device
        prepared = _PreparedEsmFold2DenoisingContext(
            self.denoiser.conditioning.condition_pair(pair, relative_position_encoding),
            *self.denoiser.atom_encoder.encode_reference(
                reference_positions,
                reference_space_id,
                reference_charges,
                reference_element_features,
                reference_atom_name_character_features,
                atom_to_token,
                num_samples,
            ),
        )

        coordinate_dtype = self.denoiser.atom_decoder.coordinate_projection.weight.dtype
        schedule = self.inference_noise_schedule(num_sampling_steps, device, coordinate_dtype)
        if max_inference_sigma is Ellipsis:
            max_inference_sigma = self.config.diffusion_structure_head.max_inference_sigma
        if max_inference_sigma is not None:
            schedule = schedule[schedule <= float(max_inference_sigma)]
            schedule = F.pad(schedule, (1, 0), value=float(max_inference_sigma))

        applied_noise_scale = self.noise_scale if noise_scale is None else float(noise_scale)
        applied_step_scale = self.step_scale if step_scale is None else float(step_scale)

        sampled_reference_positions = repeat_batch(reference_positions, num_samples).to(dtype=coordinate_dtype)
        coordinates = schedule[0] * torch.empty_like(sampled_reference_positions).normal_(generator=generator)
        previous_denoised_coordinates: Tensor | None = None
        token_representations: Tensor | None = None
        atom_representations: Tensor | None = None

        # Schedule entries stay as tensors and are indexed per step, so sampling never synchronizes
        # them to the host. The shared step derives its own churn gate from the next noise level.
        num_steps = schedule.numel() - 1
        # The schedule is truncated at max_inference_sigma, and sampling may stop when
        # denoised-coordinate RMSD stabilizes, so the loop has a data-dependent step count.
        for step_index in range(num_steps):
            coordinates, previous_denoised_coordinates = self.center_random_augmentation(
                coordinates,
                second_coordinates=previous_denoised_coordinates,
                generator=generator,
            )

            churn_noise = torch.empty_like(coordinates).normal_(generator=generator)

            is_last_step = step_index == num_steps - 1
            request_atom_representations = return_atom_representations and (
                is_last_step or denoising_early_exit_rmsd is not None
            )
            captured: dict[str, Tensor | None] = {}

            def step_denoise(
                noisy: Tensor,
                noise_level: Tensor,
                *,
                request: bool = request_atom_representations,
                into: dict[str, Tensor | None] = captured,
            ) -> Tensor:
                output = self.denoiser(
                    noisy_coordinates=noisy,
                    noise_level=noise_level,
                    reference_positions=reference_positions,
                    reference_charges=reference_charges,
                    reference_element_features=reference_element_features,
                    reference_atom_name_character_features=reference_atom_name_character_features,
                    reference_space_id=reference_space_id,
                    atom_to_token=atom_to_token,
                    single_inputs=single_inputs,
                    pair=pair,
                    relative_position_encoding=relative_position_encoding,
                    num_samples=num_samples,
                    return_token_representations=True,
                    return_atom_representations=request,
                    prepared=prepared,
                )
                # CUDA Graph outputs share replay storage; retain sampler-owned predictions.
                denoised = cast(Tensor, output["denoised_coordinates"]).clone()
                into["denoised"] = denoised
                into["token_representations"] = cast(Tensor, output["token_representations"]).clone()
                if request:
                    into["atom_representations"] = cast(Tensor, output["atom_intermediates"]).clone()
                return denoised

            coordinates = self.sampler.step(
                step_denoise,
                coordinates,
                schedule[step_index],
                schedule[step_index + 1],
                noise=churn_noise,
                noise_scale=applied_noise_scale,
                step_scale=applied_step_scale,
            )
            denoised_coordinates = cast(Tensor, captured["denoised"])
            token_representations = cast(Tensor, captured["token_representations"])
            if request_atom_representations:
                atom_representations = captured.get("atom_representations")

            # Denoising early-exit: stop when consecutive predictions converge
            if denoising_early_exit_rmsd is not None and previous_denoised_coordinates is not None and step_index >= 1:
                aligned = weighted_rigid_align(
                    source=previous_denoised_coordinates,
                    target=denoised_coordinates,
                    weight=torch.ones_like(previous_denoised_coordinates[..., 0]),
                    backend=self.backend,
                )
                coordinate_difference = denoised_coordinates - aligned
                per_sample_rmsd = coordinate_difference.square().sum(dim=-1).mean(dim=-1).sqrt()
                if per_sample_rmsd.max().item() < denoising_early_exit_rmsd:
                    coordinates = denoised_coordinates
                    previous_denoised_coordinates = denoised_coordinates
                    break

            previous_denoised_coordinates = denoised_coordinates

        result: dict[str, Tensor | None] = {
            "coordinates": coordinates,
            "token_representations": token_representations,
        }
        if return_atom_representations:
            result["atom_representations"] = atom_representations
        return result


class EsmFold2ConfidenceHead(nn.Module):
    r"""
    Predicts pLDDT, PAE, PDE, resolved-atom probability and distogram bins.
    """

    distance_boundaries: Tensor
    token_feature_map: Tensor

    def __init__(self, config: EsmFold2Config, backend: str = "auto") -> None:
        super().__init__()
        self.backend = backend
        confidence_config = config.confidence
        single_size = config.single_size
        pair_size = config.pair_size
        input_size = config.input_embeddings.input_size

        self.encoded_size = input_size - 2 * len(TOKENS) - 1
        feature_map = folding_source_map()
        self.register_buffer("token_feature_map", feature_map, persistent=False)
        source_input_size = self.encoded_size + 2 * feature_map.shape[0] + 1
        self.input_norm = LayerNorm(source_input_size, backend=backend)
        self.pair_norm = LayerNorm(pair_size, backend=backend)
        self.pair_left_projection = nn.Linear(source_input_size, pair_size, bias=False)
        self.pair_right_projection = nn.Linear(source_input_size, pair_size, bias=False)
        self.left_product_projection = nn.Linear(source_input_size, pair_size, bias=False)
        self.right_product_projection = nn.Linear(source_input_size, pair_size, bias=False)
        self.product_output_projection = nn.Linear(pair_size, pair_size, bias=False)
        self.distance_embedding = nn.Embedding(confidence_config.num_distogram_bins, pair_size)
        # The boundaries follow from the configuration, so they are computed rather than stored.
        # A model built on the meta device still needs real boundaries, since loading never fills them.
        buffer_device = self.distance_embedding.weight.device
        if buffer_device.type == "meta":
            buffer_device = torch.device("cpu")
        self.register_buffer(
            "distance_boundaries",
            torch.linspace(
                confidence_config.min_distance,
                confidence_config.max_distance,
                confidence_config.num_distogram_bins - 1,
                device=buffer_device,
            ),
            persistent=False,
        )

        folding_stack_config = confidence_config.folding_stack
        self.folding_stack = EsmFold2FoldingStack(
            folding_stack_config.num_blocks,
            pair_size,
            expansion=4,
            backend=backend,
        )
        self.row_attention_pooling = EsmFold2RowAttentionPooling(pair_size, single_size)
        self.token_to_atom_broadcast = TokenToAtomBroadcast(backend=backend)
        self.atom_to_token_mean = AtomToTokenMean(backend=backend)

        # Heads.
        self.plddt_norm = LayerNorm(single_size, backend=backend)
        max_atoms_per_token = 23
        self.plddt_projection_weight = nn.Parameter(
            torch.zeros(max_atoms_per_token, single_size, confidence_config.num_plddt_bins)
        )

        self.pae_norm = LayerNorm(pair_size, backend=backend)
        self.pae_projection = nn.Linear(pair_size, confidence_config.num_pae_bins, bias=False)

        self.pde_norm = LayerNorm(pair_size, backend=backend)
        self.pde_projection = nn.Linear(pair_size, confidence_config.num_pde_bins, bias=False)

        self.resolved_norm = LayerNorm(single_size, backend=backend)
        # 2 = resolved logits ([unresolved, resolved]).
        self.resolved_projection_weight = nn.Parameter(torch.zeros(max_atoms_per_token, single_size, 2))

    def forward(
        self,
        single_inputs: Tensor,
        pair: Tensor,
        coordinates: Tensor,
        token_representative_atom_index: Tensor,
        atom_to_token: Tensor,
        chain_index: Tensor,
        molecule_type: Tensor,
        relative_position_encoding: Tensor,
        token_bonds_encoding: Tensor,
        num_samples: int = 1,
    ) -> dict[str, Tensor]:
        coordinates = _esmfold2_coordinates(coordinates, feature="ESMFold2 confidence coordinates")
        normalized_single_inputs = self.input_norm(
            map_token_feature_planes(single_inputs, self.token_feature_map, self.encoded_size)
        )

        base_pair = self.pair_norm(pair)
        base_pair = base_pair + relative_position_encoding + token_bonds_encoding
        base_pair = base_pair + self.pair_left_projection(normalized_single_inputs).unsqueeze(2)
        base_pair = base_pair + self.pair_right_projection(normalized_single_inputs).unsqueeze(1)
        base_pair = base_pair + self.product_output_projection(
            self.left_product_projection(normalized_single_inputs)[:, :, None, :]
            * self.right_product_projection(normalized_single_inputs)[:, None, :, :]
        )

        pair = repeat_batch(base_pair, num_samples)
        expanded_atom_to_token = repeat_batch(atom_to_token, num_samples)
        expanded_representative_atom_index = repeat_batch(token_representative_atom_index, num_samples).long()
        batch_size = pair.shape[0]

        representative_coordinates = _gather_representative_atom_coordinates(
            coordinates,
            expanded_representative_atom_index,
        )
        representative_distances = torch.cdist(
            representative_coordinates, representative_coordinates, compute_mode="donot_use_mm_for_euclid_dist"
        )
        distogram_bins = (representative_distances.unsqueeze(-1) > self.distance_boundaries).sum(dim=-1).long()
        pair = pair + self.distance_embedding(distogram_bins)

        folded_pair = self.folding_stack(pair)
        pair = pair + folded_pair
        del folded_pair
        single = self.row_attention_pooling(pair)

        atom_single = self.token_to_atom_broadcast(
            _tokens_like_atoms(single, expanded_atom_to_token),
            expanded_atom_to_token,
        )
        normalized_atom_single = self.plddt_norm(atom_single)

        intra_token_index = _compute_intra_token_index(expanded_atom_to_token)
        intra_token_index = intra_token_index.clamp(max=self.plddt_projection_weight.shape[0] - 1)
        if isinstance(intra_token_index, NestedTensor):
            plddt_weights = intra_token_index.packed_with_static_tail(
                self.plddt_projection_weight[intra_token_index.concat]
            )
        else:
            plddt_weights = self.plddt_projection_weight[intra_token_index]
        plddt_logits = torch.einsum("...c,...cb->...b", normalized_atom_single, plddt_weights)
        plddt_per_atom = plddt_from_logits(plddt_logits)

        plddt = self.atom_to_token_mean(
            plddt_per_atom.unsqueeze(-1),
            expanded_atom_to_token,
            _tokens_like_atoms(single, expanded_atom_to_token),
        ).squeeze(-1)
        if isinstance(plddt, NestedTensor):
            plddt = plddt.concat.reshape(single.shape[:-1])

        mean_plddt = plddt_per_atom.mean(dim=-1)

        expanded_type = repeat_batch(molecule_type, num_samples)
        expanded_chain_index = repeat_batch(chain_index, num_samples)
        is_ligand = (expanded_type == _NONPOLYMER_ID).to(plddt_per_atom.dtype)
        iplddt_weight = interface_confidence_weights(
            representative_distances, expanded_chain_index, is_ligand, torch.ones_like(expanded_type, dtype=torch.bool)
        )
        iplddt_weight_atoms = self.token_to_atom_broadcast(
            _tokens_like_atoms(iplddt_weight.unsqueeze(-1), expanded_atom_to_token),
            expanded_atom_to_token,
        ).squeeze(-1)
        complex_iplddt = weighted_mean(plddt_per_atom, iplddt_weight_atoms, backend=self.backend)

        plddt_ca = _gather_representative_atom_coordinates(
            plddt_per_atom.unsqueeze(-1), expanded_representative_atom_index
        ).squeeze(-1)

        # PAE
        pae_logits = self.pae_projection(self.pae_norm(pair))
        pae = categorical_mean(pae_logits, start=0.0, end=32.0)

        # PDE
        pde_logits = self.pde_projection(self.pde_norm(pair))
        pde = categorical_mean(pde_logits, start=0.0, end=32.0)

        # Resolved (per-atom binary).
        resolved_atom_single = self.resolved_norm(atom_single)
        if isinstance(intra_token_index, NestedTensor):
            resolved_weights = intra_token_index.packed_with_static_tail(
                self.resolved_projection_weight[intra_token_index.concat]
            )
        else:
            resolved_weights = self.resolved_projection_weight[intra_token_index]
        resolved_logits = torch.einsum("...c,...cb->...b", resolved_atom_single, resolved_weights)

        # pTM / ipTM from pae_logits.
        num_bins = pae_logits.shape[-1]
        bin_width = 32.0 / num_bins
        bin_centers = torch.arange(0.5 * bin_width, 32.0, bin_width, device=pae_logits.device, dtype=pae_logits.dtype)
        num_residues = pae_logits.new_full((pae_logits.shape[0], 1), pae_logits.shape[1])
        tm_expected = tm_weight_from_pae(pae_logits, num_residues.unsqueeze(-1), bin_centers=bin_centers)

        ptm = predicted_tm_score(tm_expected, backend=self.backend)

        inter_chain_mask = (expanded_chain_index.unsqueeze(-1) != expanded_chain_index.unsqueeze(-2)).to(
            tm_expected.dtype
        )
        iptm = predicted_tm_score(tm_expected, pair_weights=inter_chain_mask, backend=self.backend)

        max_chain_index = int(expanded_chain_index.max().item()) if batch_size > 0 else 0
        num_chains = max_chain_index + 1
        chain_pair_iptm = chain_pair_tm_scores(
            tm_expected, expanded_chain_index, range(num_chains), backend=self.backend
        )
        chain_pair_mean_tm = chain_pair_mean_tm_scores(
            tm_expected, expanded_chain_index, range(num_chains), backend=self.backend
        )

        return {
            "plddt_logits": plddt_logits,
            "plddt": plddt,
            "plddt_per_atom": plddt_per_atom,
            "plddt_ca": plddt_ca,
            "mean_plddt": mean_plddt,
            "complex_iplddt": complex_iplddt,
            "pae_logits": pae_logits,
            "pae": pae,
            "pde_logits": pde_logits,
            "pde": pde,
            "resolved_logits": resolved_logits,
            "ptm": ptm,
            "iptm": iptm,
            "chain_pair_iptm": chain_pair_iptm,
            "chain_pair_mean_tm": chain_pair_mean_tm,
        }


class EsmFold2ExperimentalConfidenceHead(nn.Module):
    r"""
    Narrow confidence head of the experimental recurrent ESMFold2 graph.
    """

    distance_boundaries: Tensor
    token_feature_map: Tensor

    def __init__(self, config: EsmFold2ExperimentalConfig, backend: str = "auto") -> None:
        super().__init__()
        self.backend = backend
        confidence_config = config.confidence
        single_size = config.single_size
        pair_size = config.pair_size
        input_size = config.input_embeddings.input_size

        self.encoded_size = input_size - 2 * len(TOKENS) - 1
        feature_map = folding_source_map()
        self.register_buffer("token_feature_map", feature_map, persistent=False)
        source_input_size = self.encoded_size + 2 * feature_map.shape[0] + 1
        self.input_norm = LayerNorm(source_input_size, backend=backend)
        self.pair_norm = LayerNorm(pair_size, backend=backend)
        self.pair_left_projection = nn.Linear(source_input_size, pair_size, bias=False)
        self.pair_right_projection = nn.Linear(source_input_size, pair_size, bias=False)
        self.left_product_projection = nn.Linear(source_input_size, pair_size, bias=False)
        self.right_product_projection = nn.Linear(source_input_size, pair_size, bias=False)
        self.product_output_projection = nn.Linear(pair_size, pair_size, bias=False)
        self.distance_embedding = nn.Embedding(confidence_config.num_distogram_bins, pair_size)
        # The boundaries follow from the configuration, so they are computed rather than stored.
        # A model built on the meta device still needs real boundaries, since loading never fills them.
        buffer_device = self.distance_embedding.weight.device
        if buffer_device.type == "meta":
            buffer_device = torch.device("cpu")
        self.register_buffer(
            "distance_boundaries",
            torch.linspace(
                confidence_config.min_distance,
                confidence_config.max_distance,
                confidence_config.num_distogram_bins - 1,
                device=buffer_device,
            ),
            persistent=False,
        )

        folding_stack_config = confidence_config.folding_stack
        self.folding_stack = EsmFold2FoldingStack(
            folding_stack_config.num_blocks,
            pair_size,
            expansion=4,
            backend=backend,
        )
        self.row_attention_pooling = EsmFold2RowAttentionPooling(pair_size, single_size)
        self.token_to_atom_broadcast = TokenToAtomBroadcast(backend=backend)
        self.atom_to_token_mean = AtomToTokenMean(backend=backend)

        self.plddt_norm = LayerNorm(single_size, backend=backend)
        max_atoms_per_token = 23
        self.plddt_projection_weight = nn.Parameter(
            torch.zeros(max_atoms_per_token, single_size, confidence_config.num_plddt_bins)
        )

        self.pae_projection = nn.Linear(pair_size, confidence_config.num_pae_bins, bias=False)

    def forward(
        self,
        single_inputs: Tensor,
        pair: Tensor,
        coordinates: Tensor,
        token_representative_atom_index: Tensor,
        atom_to_token: Tensor,
        chain_index: Tensor,
        molecule_type: Tensor,
        relative_position_encoding: Tensor,
        token_bonds_encoding: Tensor,
    ) -> EsmFold2ExperimentalConfidenceOutput:
        coordinates = _esmfold2_coordinates(coordinates, feature="ESMFold2 confidence coordinates")
        batch_size, num_samples = coordinates.shape[:2]
        coordinates = flatten_batch(coordinates)
        normalized_single_inputs = self.input_norm(
            map_token_feature_planes(single_inputs, self.token_feature_map, self.encoded_size)
        )

        base_pair = self.pair_norm(pair)
        base_pair = base_pair + relative_position_encoding + token_bonds_encoding
        base_pair = base_pair + self.pair_left_projection(normalized_single_inputs).unsqueeze(2)
        base_pair = base_pair + self.pair_right_projection(normalized_single_inputs).unsqueeze(1)
        base_pair = base_pair + self.product_output_projection(
            self.left_product_projection(normalized_single_inputs)[:, :, None, :]
            * self.right_product_projection(normalized_single_inputs)[:, None, :, :]
        )

        pair = repeat_batch(base_pair, num_samples)
        expanded_atom_to_token = repeat_batch(atom_to_token, num_samples)
        expanded_representative_atom_index = repeat_batch(token_representative_atom_index, num_samples).long()

        representative_coordinates = _gather_representative_atom_coordinates(
            coordinates,
            expanded_representative_atom_index,
        )
        representative_distances = torch.cdist(
            representative_coordinates, representative_coordinates, compute_mode="donot_use_mm_for_euclid_dist"
        )
        distogram_bins = (representative_distances.unsqueeze(-1) > self.distance_boundaries).sum(dim=-1).long()
        pair = pair + self.distance_embedding(distogram_bins)

        folded_pair = self.folding_stack(pair)
        pair = pair + folded_pair
        del folded_pair
        single = self.row_attention_pooling(pair)

        atom_single = self.token_to_atom_broadcast(
            _tokens_like_atoms(single, expanded_atom_to_token),
            expanded_atom_to_token,
        )
        normalized_atom_single = self.plddt_norm(atom_single)

        intra_token_index = _compute_intra_token_index(expanded_atom_to_token)
        intra_token_index = intra_token_index.clamp(max=self.plddt_projection_weight.shape[0] - 1)
        if isinstance(intra_token_index, NestedTensor):
            plddt_weights = intra_token_index.packed_with_static_tail(
                self.plddt_projection_weight[intra_token_index.concat]
            )
        else:
            plddt_weights = self.plddt_projection_weight[intra_token_index]
        plddt_logits = torch.einsum("...c,...cb->...b", normalized_atom_single, plddt_weights)
        plddt_per_atom = plddt_from_logits(plddt_logits)

        plddt = self.atom_to_token_mean(
            plddt_per_atom.unsqueeze(-1),
            expanded_atom_to_token,
            _tokens_like_atoms(single, expanded_atom_to_token),
        ).squeeze(-1)
        if isinstance(plddt, NestedTensor):
            plddt = plddt.concat.reshape(single.shape[:-1])

        mean_plddt = plddt_per_atom.mean(dim=-1)

        expanded_type = repeat_batch(molecule_type, num_samples)
        expanded_chain_index = repeat_batch(chain_index, num_samples)
        is_ligand = (expanded_type == _NONPOLYMER_ID).to(plddt_per_atom.dtype)
        iplddt_weight = interface_confidence_weights(
            representative_distances, expanded_chain_index, is_ligand, torch.ones_like(expanded_type, dtype=torch.bool)
        )
        iplddt_weight_atoms = self.token_to_atom_broadcast(
            _tokens_like_atoms(iplddt_weight.unsqueeze(-1), expanded_atom_to_token),
            expanded_atom_to_token,
        ).squeeze(-1)
        complex_iplddt = weighted_mean(plddt_per_atom, iplddt_weight_atoms, backend=self.backend)

        plddt_ca = _gather_representative_atom_coordinates(
            plddt_per_atom.unsqueeze(-1), expanded_representative_atom_index
        ).squeeze(-1)

        pae_logits = self.pae_projection(pair)
        pae = categorical_mean(pae_logits, start=0.0, end=32.0)

        num_bins = pae_logits.shape[-1]
        bin_width = 32.0 / num_bins
        bin_centers = torch.arange(0.5 * bin_width, 32.0, bin_width, device=pae_logits.device, dtype=pae_logits.dtype)
        num_residues = pae_logits.new_full((pae_logits.shape[0], 1), pae_logits.shape[1])
        tm_expected = tm_weight_from_pae(pae_logits, num_residues.unsqueeze(-1), bin_centers=bin_centers)

        ptm = predicted_tm_score(tm_expected, backend=self.backend)

        inter_chain_mask = (expanded_chain_index.unsqueeze(-1) != expanded_chain_index.unsqueeze(-2)).to(
            tm_expected.dtype
        )
        iptm = predicted_tm_score(tm_expected, pair_weights=inter_chain_mask, backend=self.backend)

        max_chain_index = int(expanded_chain_index.max().item())
        num_chains = max_chain_index + 1
        chain_pair_iptm = chain_pair_tm_scores(
            tm_expected, expanded_chain_index, range(num_chains), backend=self.backend
        )
        chain_pair_mean_tm = chain_pair_mean_tm_scores(
            tm_expected, expanded_chain_index, range(num_chains), backend=self.backend
        )

        def unflatten(value: Tensor) -> Tensor:
            return unflatten_batch(value, (batch_size, num_samples))

        return EsmFold2ExperimentalConfidenceOutput(
            plddt_logits=unflatten(plddt_logits),
            plddt=unflatten(plddt),
            plddt_per_atom=unflatten(plddt_per_atom),
            plddt_ca=unflatten(plddt_ca),
            mean_plddt=unflatten(mean_plddt),
            complex_iplddt=unflatten(complex_iplddt),
            pae_logits=unflatten(pae_logits),
            pae=unflatten(pae),
            ptm=unflatten(ptm),
            iptm=unflatten(iptm),
            chain_pair_iptm=unflatten(chain_pair_iptm),
            chain_pair_mean_tm=unflatten(chain_pair_mean_tm),
            batch_size=[batch_size],
        )


_CHAR_VOCAB_SIZE: int = 64
_MAX_CHARS: int = 4
_COORDINATE_SIZE: int = 3
_MAX_ATOMIC_NUMBER: int = 128

# Input feature size = 3 + 1 + 1 + 128 + 64*4 = 389
_ATOM_FEATURE_SIZE: int = _COORDINATE_SIZE + 1 + 1 + _MAX_ATOMIC_NUMBER + _CHAR_VOCAB_SIZE * _MAX_CHARS


_NUM_RESIDUE_TYPES: int = len(TOKENS)


def _residue_type_features(
    residue_type_ids: Tensor,
    residue_type_probabilities: Tensor | None,
    dtype: torch.dtype,
) -> Tensor:
    if residue_type_probabilities is not None:
        return residue_type_probabilities
    return F.one_hot(residue_type_ids.long(), num_classes=_NUM_RESIDUE_TYPES).to(dtype)


_EPS = 1e-5


def _maybe_subsample_msa(
    msa_tokens: Tensor,
    msa_mask: Tensor | None,
    has_deletion: Tensor | None,
    deletion_value: Tensor | None,
    *,
    max_depth: int | None,
    enabled: bool,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor | None]:
    r"""
    Randomly subsample the MSA to ``max_depth`` rows, keeping query row 0.

    Accepts encoder-input MSA token tensors shaped ``[B, depth, L]``. No-op when
    disabled, ``max_depth`` is None, or depth <= ``max_depth``.
    """
    if not enabled or max_depth is None:
        return msa_tokens, msa_mask, has_deletion, deletion_value
    depth = msa_tokens.size(1)
    if depth <= 1 or depth <= max_depth:
        return msa_tokens, msa_mask, has_deletion, deletion_value
    indices = torch.zeros(max_depth, dtype=torch.long, device=msa_tokens.device)
    perm = torch.randperm(depth - 1, device=msa_tokens.device, generator=generator)[: max_depth - 1]
    indices[1:] = perm + 1
    indices = indices.sort().values
    msa_tokens = msa_tokens[:, indices]
    if msa_mask is not None:
        msa_mask = msa_mask[:, indices]
    if has_deletion is not None:
        has_deletion = has_deletion[:, indices]
    if deletion_value is not None:
        deletion_value = deletion_value[:, indices]
    return msa_tokens, msa_mask, has_deletion, deletion_value


def _maybe_apply_msa_column_masking(
    msa_mask: Tensor | None,
    rate: float,
    generator: torch.Generator | None = None,
) -> Tensor | None:
    r"""
    Mask fraction ``rate`` of MSA columns in non-query rows of
    ``msa_mask``, keeping query row 0. No-op when absent, ``rate <= 0``,
    or depth <= 1.
    """
    if msa_mask is None or rate <= 0.0 or msa_mask.size(1) <= 1:
        return msa_mask
    B, _M, L = msa_mask.shape
    col_keep = torch.rand(B, L, device=msa_mask.device, generator=generator) >= rate
    masked = msa_mask.bool() & col_keep.unsqueeze(1)
    masked[:, 0, :] = msa_mask[:, 0, :].bool()
    return masked


def _gather_representative_atom_coordinates(
    coordinates: Tensor | NestedTensor, representative_atom_index: Tensor
) -> Tensor:
    r"""
    Gather representative atom coordinates for each token.

    Args:
        coordinates: Dense or nested atom values ``[B, A, C]``.
        representative_atom_index: [B, L] int64

    Returns:
        Dense representative values ``[B, L, C]`` over the shared token axis.
    """
    if isinstance(coordinates, NestedTensor):
        atom_offsets = coordinates.ragged_level_offsets(0, dtype=torch.long).to(device=coordinates.device)
        packed_index = representative_atom_index + atom_offsets[:-1, None]
        return coordinates.concat[packed_index]
    coordinate_index = representative_atom_index.unsqueeze(-1).expand(-1, -1, coordinates.size(-1))
    return torch.gather(coordinates, 1, coordinate_index)


def _compute_intra_token_index(atom_to_token: Tensor) -> Tensor:
    r"""
    Compute local atom index within each token (vectorised).

    Atoms belonging to the same token are contiguous, so this computes a
    running count that resets at each token boundary.

    Args:
        atom_to_token: [B, A] flat index mapping each atom to its token.

    Returns:
        [B, A] tensor with values in [0, max_atoms_per_token - 1].
    """
    same_as_prev = F.pad(atom_to_token[:, 1:] == atom_to_token[:, :-1], (1, 0), value=False)
    ones = torch.ones_like(atom_to_token)
    cumsum = torch.cumsum(ones, dim=-1)
    group_start = cumsum.masked_fill(same_as_prev, 0)
    group_start = torch.cummax(group_start, dim=-1).values
    return cumsum - group_start


def _tokens_like_atoms(token_hidden: Tensor, atom_to_token: Tensor | NestedTensor) -> Tensor | NestedTensor:
    r"""
    Give a dense token axis the batch layout of the atom axis it exchanges features with.

    ESMFold2 batches design candidates over the atom axis alone: candidates share a token count but
    differ by side chain, so the token axis stays dense while the atom axis is ragged. The shared
    atom/token transfers require both sides to agree on layout, so a ragged atom axis lifts the
    token axis to equal-length nested elements. Equal lengths make the lift and the dense view of a
    nested token result exact in both directions.

    Args:
        token_hidden: ``[..., N_token, ...]`` dense token-axis tensor.
        atom_to_token: The atom axis this will exchange features with.

    Returns:
        ``token_hidden`` unchanged for a dense atom axis, nested for a ragged one.
    """

    if not isinstance(atom_to_token, NestedTensor):
        return token_hidden
    return NestedTensor(list(token_hidden), ragged_dims=(0,))


def _apply_rotary_emb_3d(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    r"""
    Apply RoPE with batch-dependent cos/sin.

    Every axis is addressed from the tail, so this serves a dense atom axis and a ragged one alike.

    Args:
        x: ``[..., N_atom, H, D]``
        cos: ``[..., N_atom, D/2]``
        sin: ``[..., N_atom, D/2]``
    """
    rotary_size = cos.shape[-1] * 2
    cos = torch.cat((cos, cos), dim=-1).to(dtype=x.dtype).unsqueeze(-2)
    sin = torch.cat((sin, sin), dim=-1).to(dtype=x.dtype).unsqueeze(-2)
    rotary = x[..., :rotary_size]
    rotary_left, rotary_right = rotary.chunk(2, dim=-1)
    rotated = torch.cat((-rotary_right, rotary_left), dim=-1)
    return torch.cat(
        [rotary * cos + rotated * sin, x[..., rotary_size:]],
        dim=-1,
    )


def _build_3d_rope(
    reference_positions: Tensor,
    reference_space_id: Tensor,
    head_size: int,
    num_spatial_pairs_per_axis: int = 4,
    num_reference_space_pairs: int = 2,
    spatial_base_frequency: float = 10000.0,
    reference_space_base_frequency: float = 10.0,
) -> tuple[Tensor, Tensor]:
    r"""
    Build cos/sin for 3D RoPE + reference-space RoPE.
    """
    device = reference_positions.device
    dtype = reference_positions.dtype
    batch_size, num_atoms = reference_positions.shape[:2]
    half_head_size = head_size // 2
    num_spatial_pairs = 3 * num_spatial_pairs_per_axis

    spatial_inverse_frequency = 1.0 / (
        spatial_base_frequency
        ** (torch.arange(0, num_spatial_pairs_per_axis, dtype=dtype, device=device) / num_spatial_pairs_per_axis)
    )
    reference_space_inverse_frequency = 1.0 / (
        reference_space_base_frequency
        ** (torch.arange(0, num_reference_space_pairs, dtype=dtype, device=device) / num_reference_space_pairs)
    )

    spatial_frequencies = torch.einsum("bna,k->bnak", reference_positions, spatial_inverse_frequency)
    spatial_frequencies = spatial_frequencies.reshape(batch_size, num_atoms, num_spatial_pairs)

    reference_space_values = reference_space_id.to(dtype)
    reference_space_frequencies = torch.einsum(
        "bn,k->bnk",
        reference_space_values,
        reference_space_inverse_frequency,
    )

    num_active_pairs = num_spatial_pairs + num_reference_space_pairs
    frequencies = torch.cat([spatial_frequencies, reference_space_frequencies], dim=-1)

    if num_active_pairs < half_head_size:
        padding = torch.zeros(
            batch_size,
            num_atoms,
            half_head_size - num_active_pairs,
            device=device,
            dtype=dtype,
        )
        frequencies = torch.cat([frequencies, padding], dim=-1)

    return frequencies.cos(), frequencies.sin()


def _qk_norm(x: Tensor) -> Tensor:
    return F.rms_norm(x, (x.size(-1),)).to(x.dtype)


def _compute_esmc_hidden_states(
    esmc: nn.Module,
    language_model_tokens: Tensor,
    language_model_mask: Tensor,
    token_to_esmc: Tensor,
    token_has_esmc: Tensor,
    language_model_mask_probability: float = 0.0,
    bos_token_id: int = 1,
    eos_token_id: int = 2,
    mask_token_id: int = 4,
    generator: torch.Generator | None = None,
) -> Tensor:
    r"""
    Run ESMC and gather per-layer states in structure-token order.
    """
    esmc_model = cast(Any, esmc)
    if language_model_mask_probability > 0.0:
        special_tokens = (
            (language_model_tokens == bos_token_id) | (language_model_tokens == eos_token_id) | ~language_model_mask
        )
        positions_to_mask = (
            torch.rand(
                language_model_tokens.shape,
                device=language_model_tokens.device,
                generator=generator,
            )
            < language_model_mask_probability
        ) & ~special_tokens
        language_model_tokens = language_model_tokens.masked_fill(positions_to_mask, mask_token_id)

    language_model_sequence_index = torch.zeros_like(language_model_tokens).masked_fill(
        ~language_model_mask,
        -1,
    )
    output = esmc_model(
        input_ids=language_model_tokens,
        sequence_id=language_model_sequence_index,
        output_hidden_states=True,
        return_dict=True,
    )
    recorded = output.hidden_states
    assert recorded is not None
    layers = (*recorded[:-1], output.last_hidden_state)
    hidden_states = torch.stack(layers, dim=2)
    hidden_states = hidden_states.flatten(0, 1).unsqueeze(0)
    hidden_states = hidden_states.expand(token_to_esmc.shape[0], -1, -1, -1)
    index = token_to_esmc[:, :, None, None].expand(
        -1,
        -1,
        hidden_states.shape[2],
        hidden_states.shape[3],
    )
    gathered = torch.gather(hidden_states, 1, index)
    return gathered * token_has_esmc[:, :, None, None].to(gathered.dtype)


def _dropout(input: Tensor, p: float, generator: torch.Generator | None) -> Tensor:
    if generator is None:
        return F.dropout(input, p=p, training=True)
    if p == 0.0:
        return input
    if p == 1.0:
        return torch.zeros_like(input)
    if input.is_cuda:
        with torch.random.fork_rng(devices=[input.device]):
            torch.cuda.set_rng_state(generator.get_state(), input.device)
            output = F.dropout(input, p=p, training=True)
            generator.set_state(torch.cuda.get_rng_state(input.device))
        return output
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(generator.get_state())
        output = F.dropout(input, p=p, training=True)
        generator.set_state(torch.get_rng_state())
    return output


# ===========================================================================
# ESMFold2 tensor boundary.
# ===========================================================================


@tensorclass
class EsmFold2Input:
    r"""
    Tensor input for ESMFold2 over the slot axis of one prepared structure (atoms are its slots).

    ``B`` is batch size, ``N`` token count, ``A`` atom count, ``M`` MSA depth,
    ``C`` protein-chain count, and ``L`` padded language-model length.

    Attributes:
        token_index: Token indices within each residue with shape ``(B, N)``.
        residue_index: Chain-local residue indices with shape ``(B, N)``.
        chain_index: Molecular-instance indices with shape ``(B, N)``.
        symmetry_index: Copy indices within each entity with shape ``(B, N)``.
        entity_index: Molecular-entity indices with shape ``(B, N)``.
        molecule_type: Molecular category IDs with shape ``(B, N)``.
        residue_type_ids: Residue-type IDs with shape ``(B, N)``.
        token_bonds: Symmetric positive-relation presence with shape ``(B, N, N, 1)``.
        reference_positions: Reference conformer coordinates with shape
            ``(B, A, 3)`` in angstroms.
        reference_elements: Atomic numbers with shape ``(B, A)``.
        reference_charges: Formal charges with shape ``(B, A)``.
        reference_atom_name_characters: Four encoded atom-name characters with shape
            ``(B, A, 4)``.
        reference_space_id: Reference conformer group IDs with shape ``(B, A)``.
        atom_to_token: Token index for each atom with shape ``(B, A)``.
        token_representative_atom_index: Representative atom index for each token with shape
            ``(B, N)``.
        token_has_esmc: Mask of tokens backed by language-model states with shape
            ``(B, N)``.
        deletion_mean: Optional per-token mean deletion values with shape ``(B, N)``.
        msa_tokens: Optional canonical 72-class MSA token IDs with shape ``(B, M, N)``.
        has_deletion: Optional MSA deletion flags with shape ``(B, M, N)``.
        deletion_value: Optional transformed MSA deletion values with shape
            ``(B, M, N)``.
        msa_mask: Optional MSA validity mask with shape ``(B, M, N)``.
        language_model_tokens: Optional language-model token IDs with shape ``(C, L)``.
        language_model_mask: Optional validity mask with shape ``(C, L)``.
        token_to_esmc: Optional flattened language-model index for each structure token,
            with shape ``(B, N)``.
        esmc_hidden_states: Optional language-model states aligned to structure tokens,
            with shape ``(B, N, H, C_lm)``.
    """

    token_index: Tensor
    residue_index: Tensor
    chain_index: Tensor
    symmetry_index: Tensor
    entity_index: Tensor
    molecule_type: Tensor
    residue_type_ids: Tensor
    token_bonds: Tensor
    reference_positions: Tensor
    reference_elements: Tensor
    reference_charges: Tensor
    reference_atom_name_characters: Tensor
    reference_space_id: Tensor
    atom_to_token: Tensor
    token_representative_atom_index: Tensor
    token_has_esmc: Tensor
    deletion_mean: Tensor | None = None
    msa_tokens: Tensor | None = None
    has_deletion: Tensor | None = None
    deletion_value: Tensor | None = None
    msa_mask: Tensor | None = None
    language_model_tokens: Tensor | None = None
    language_model_mask: Tensor | None = None
    token_to_esmc: Tensor | None = None
    esmc_hidden_states: Tensor | None = None


def _esmfold2_coordinates(
    coordinates: Tensor, *, dtype: torch.dtype | None = None, feature: str = "ESMFold2 coordinates"
) -> Tensor:
    """Project ESM's unpadded, complete model coordinate fields."""
    if coordinates.dim() == 0 or coordinates.shape[-1] != 3:
        fail("value_invalid", RequestRef(feature, str(tuple(coordinates.shape))))
    return project_model_coordinates(
        coordinates, torch.ones_like(coordinates[..., 0], dtype=torch.bool), dtype=dtype, feature=feature
    )


def normalize_esmfold2_input(input: EsmFold2Input, *, dtype: torch.dtype | None = None) -> EsmFold2Input:
    """Return finite complete model references while retaining prepared anchors.

    A public raw input's reference atom axis has no padding or missing conformer coordinates: preparation gives every
    slot a reference or embedded conformer.
    """
    result = cast(EsmFold2Input, cast(TensorClass, input).clone(recurse=False))
    result.reference_positions = _esmfold2_coordinates(
        input.reference_positions, dtype=dtype, feature="ESMFold2 references"
    )
    for name in ("token_bonds", "deletion_mean", "reference_charges"):
        values = getattr(input, name)
        if values is not None:
            if not values.is_floating_point():
                values = values.to(torch.float64)
            setattr(
                result,
                name,
                project_model_features(
                    values, torch.ones_like(values, dtype=torch.bool), dtype=dtype, feature=f"ESMFold2 {name}"
                ),
            )
    if input.msa_tokens is not None:
        msa_available = torch.ones_like(input.msa_tokens, dtype=torch.bool)
        if input.msa_mask is not None:
            msa_available = msa_available & input.msa_mask.bool()
        result.msa_mask = msa_available
        result.msa_tokens = project_model_categories(
            input.msa_tokens, msa_available, num_classes=len(TOKENS), feature="ESMFold2 MSA tokens"
        )
        for name in ("has_deletion", "deletion_value"):
            values = getattr(input, name)
            if values is not None:
                if not values.is_floating_point():
                    values = torch.where(msa_available, values, 0).to(torch.float64)
                setattr(
                    result, name, project_model_features(values, msa_available, dtype=dtype, feature=f"ESMFold2 {name}")
                )
    if input.esmc_hidden_states is not None:
        result.esmc_hidden_states = project_model_features(
            input.esmc_hidden_states,
            input.token_has_esmc.to(torch.bool)[..., None, None],
            dtype=dtype,
            feature="ESMFold2 language states",
        )
    return result


class EsmFold2SamplingWithConfidenceOutput(TensorCollection):
    r"""
    Raw predictions from ESMFold2 coordinate sampling and confidence evaluation.

    Sample-valued fields carry explicit ``(B, S, ...)`` axes; distogram logits
    are shared by the ``S`` coordinate samples and have shape ``(B, N, N, D)``.
    ``chain_pair_iptm`` selects the best eligible source frame;
    ``chain_pair_mean_tm`` averages both chain axes, including intra-chain pairs.
    """

    coordinates: Tensor
    distogram_logits: Tensor
    plddt_logits: Tensor
    plddt: Tensor
    plddt_per_atom: Tensor
    plddt_ca: Tensor
    mean_plddt: Tensor
    complex_iplddt: Tensor
    pae_logits: Tensor
    pae: Tensor
    pde_logits: Tensor
    pde: Tensor
    resolved_logits: Tensor
    ptm: Tensor
    iptm: Tensor
    chain_pair_iptm: Tensor
    chain_pair_mean_tm: Tensor


class EsmFold2Output(TensorCollection):
    r"""
    Predictions from one ESMFold2 denoising call.

    Attributes:
        coordinates: Denoised atom coordinates with shape ``(B, S, A, 3)`` in
            angstroms.
        distogram_logits: Token-pair distance logits with shape ``(B, N, N, D)``.
        plddt_logits: Per-atom pLDDT logits with shape ``(B, S, A, D_plddt)``.
        pae_logits: Token-pair PAE logits with shape ``(B, S, N, N, D_pae)``.
        pde_logits: Token-pair PDE logits with shape ``(B, S, N, N, D_pde)``.
        resolved_logits: Per-atom resolved logits with shape ``(B, S, A, 2)``.
    """

    coordinates: Tensor
    distogram_logits: Tensor
    plddt_logits: Tensor
    pae_logits: Tensor
    pde_logits: Tensor
    resolved_logits: Tensor


class _EsmFold2ExperimentalTrunkOutput(NamedTuple):
    r"""
    Experimental trunk representations consumed by coordinate and confidence heads.
    """

    single_inputs: Tensor
    pair: Tensor
    distogram_logits: Tensor
    relative_position_encoding: Tensor
    token_bonds_encoding: Tensor
    reference_element_features: Tensor
    reference_atom_name_character_features: Tensor
    atom_to_token: Tensor


class EsmFold2ExperimentalSamplingOutput(TensorCollection):
    r"""
    Distogram and coordinate samples from experimental ESMFold2.

    Attributes:
        distogram_logits: Token-pair distance logits with shape ``(B, N, N, D)``.
        coordinates: Sampled atom coordinates with shape ``(B, S, A, 3)`` in
            angstroms.
    """

    distogram_logits: Tensor
    coordinates: Tensor


class EsmFold2ExperimentalConfidenceOutput(TensorCollection):
    r"""
    Fixed confidence values for one experimental ESMFold2 sample carrier.

    Attributes:
        plddt_logits: Per-atom pLDDT logits with shape ``(B, S, A, D_plddt)``.
        plddt: Per-token pLDDT values with shape ``(B, S, N)`` on the
            ``0--100`` scale.
        plddt_per_atom: Per-atom pLDDT values with shape ``(B, S, A)`` on the
            ``0--100`` scale.
        plddt_ca: Representative-atom pLDDT values with shape ``(B, S, N)`` on
            the ``0--100`` scale.
        mean_plddt: Mean pLDDT values with shape ``(B, S)`` on the
            ``0--100`` scale.
        complex_iplddt: Interface-weighted complex pLDDT values on the
            ``0--100`` scale with shape ``(B, S)``.
        pae_logits: Token-pair predicted-aligned-error logits with shape
            ``(B, S, N, N, D_pae)``.
        pae: Token-pair predicted aligned errors in angstroms with shape
            ``(B, S, N, N)``.
        ptm: Predicted TM scores with shape ``(B, S)``.
        iptm: Interface predicted TM scores with shape ``(B, S)``.
        chain_pair_iptm: Directed chain-pair interface predicted TM scores with
            shape ``(B, S, C, C)``.
        chain_pair_mean_tm: Mean calibrated TM expectations over both chain
            axes, including intra-chain entries, with shape ``(B, S, C, C)``.
    """

    plddt_logits: Tensor
    plddt: Tensor
    plddt_per_atom: Tensor
    plddt_ca: Tensor
    mean_plddt: Tensor
    complex_iplddt: Tensor
    pae_logits: Tensor
    pae: Tensor
    ptm: Tensor
    iptm: Tensor
    chain_pair_iptm: Tensor
    chain_pair_mean_tm: Tensor


class EsmFold2ExperimentalSamplingWithConfidenceOutput(NamedTuple):
    r"""
    One experimental ESMFold2 coordinate sample carrier and its confidence values.

    Attributes:
        sample: Distogram logits and sampled atom coordinates.
        confidence: Confidence values computed for ``sample``.
    """

    sample: EsmFold2ExperimentalSamplingOutput
    confidence: EsmFold2ExperimentalConfidenceOutput
