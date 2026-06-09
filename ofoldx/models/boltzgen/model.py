r"""
BoltzGen design models.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from functools import partial
from typing import Any, cast

import torch
import torch.nn.functional as F
from oops import LayerNorm
from torch import Tensor, nn

from ...modules.atom.encoders import PackedAtomAttention
from ...modules.atom.pair_encoding import encode_atom_pair
from ...modules.atom.transfer import dense_mean_to_tokens
from ...modules.atom.window import gather_key_windows
from ...modules.attention import PairBiasedSelfAttention
from ...modules.diffusion.conditioning import condition_pair, condition_single
from ...modules.diffusion.edm import EdmPreconditioner
from ...modules.embedding.fourier import FrozenFourierEmbedding
from ...modules.embedding.position import RelativePositionEmbedder, generate_relative_position_features
from ...modules.functional import pair_mask_from_token_mask
from ...modules.geometry import uniform_random_rotation, weighted_rigid_align
from ...modules.head.distogram import DistogramHead
from ...modules.transformer import ConditionedTransformer
from ...modules.transition import RowwiseDropout, Transition
from ...modules.trunk.msa import MsaModule, MsaPairWeightedAveraging, OuterProductMean, TrianglePairUpdateBlock
from ...modules.trunk.pairformer import PairformerStack
from ..contract import BaseInput, DesignGeneratorOutput, StructurePredictionOutput
from ..mixin import ModelMixin
from ..registry import DESIGN_GENERATOR_MODELS, MODELS
from .config import BoltzGenAffinityConfig, BoltzGenConfig, BoltzGenIFoldConfig


@MODELS.register("boltzgen")
@DESIGN_GENERATOR_MODELS.register("boltzgen")
@MODELS.register("boltzgen_diverse")
@DESIGN_GENERATOR_MODELS.register("boltzgen_diverse")
@MODELS.register("boltzgen_adherence")
@DESIGN_GENERATOR_MODELS.register("boltzgen_adherence")
@MODELS.register("boltzgen_structuretrained_small")
@DESIGN_GENERATOR_MODELS.register("boltzgen_structuretrained_small")
class BoltzGenDesignGenerator(ModelMixin):
    r"""
    BoltzGen structure-generating design model (``boltzgen1_diverse`` / ``boltzgen1_adherence``).

    The model-specific components are defined in this file: the v1
    [`TemplateModule`][ofoldx.models.boltzgen.model.TemplateModule], an added
    [`TokenDistanceModule`][ofoldx.models.boltzgen.model.TokenDistanceModule], design conditioning in the input
    embedder, and a structure-generation path without confidence or affinity heads.
    [`BoltzGenConfig`][ofoldx.models.boltzgen.config.BoltzGenConfig] carries the shared submodule settings plus
    the design-specific modules.

    Args:
        config: Optional BoltzGen model config. Defaults to ``BoltzGenConfig()``.

    Examples:
        >>> from ofoldx import AutoModel, AutoProcessor  # doctest: +SKIP
        >>> processor = AutoProcessor.from_pretrained("oteam/boltzgen")  # doctest: +SKIP
        >>> model = AutoModel.from_pretrained("oteam/boltzgen")  # doctest: +SKIP
        >>> output = model(processor("complex.cif"), num_samples=2)  # doctest: +SKIP
    """

    config_class = BoltzGenConfig

    def __init__(
        self,
        config: BoltzGenConfig | None = None,
    ) -> None:
        config = config or BoltzGenConfig()
        super().__init__(config)
        self._recycling_steps = config.recycling_steps
        single_size, pair_size = config.single_size, config.pair_size
        embedder, trunk, diffusion, sampler = config.input_embedder, config.trunk, config.diffusion, config.sampler
        design = config.design

        self.input_embedder = AtomConditionedInputEmbedder(
            single_size=single_size,
            atom_hidden_size=config.atom_hidden_size,
            atom_pair_hidden_size=config.atom_pair_hidden_size,
            atom_feature_size=config.atom_feature_size,
            num_query_atoms=diffusion.num_query_atoms,
            num_key_atoms=diffusion.num_key_atoms,
            num_atom_encoder_blocks=embedder.num_atom_encoder_blocks,
            num_atom_encoder_attention_heads=embedder.num_atom_encoder_attention_heads,
            vocab_size=embedder.vocab_size,
            msa_profile_size=embedder.msa_profile_size,
            num_method_types=embedder.num_method_types,
            num_molecule_types=embedder.num_molecule_types,
            add_binding_specification=embedder.add_binding_specification,
            add_ss_specification=embedder.add_ss_specification,
            operator_backend=trunk.operator_backend,
        )

        self.single_init_projection = nn.Linear(single_size, single_size, bias=False)
        self.pair_left_projection = nn.Linear(single_size, pair_size, bias=False)
        self.pair_right_projection = nn.Linear(single_size, pair_size, bias=False)
        self.relative_position_embedder = RelativePositionEmbedder(config.relative_position_size, pair_size)
        self.token_bond_projection = nn.Linear(1, pair_size, bias=False)
        self.token_bond_type_embedding = nn.Embedding(config.num_bond_types, pair_size)
        self.contact_conditioning = ContactConditioning(
            pair_size, cutoff_min=config.conditioning_cutoff_min, cutoff_max=config.conditioning_cutoff_max
        )
        self.single_recycle_norm = LayerNorm(single_size)
        self.pair_recycle_norm = LayerNorm(pair_size)
        self.single_recycle_projection = nn.Linear(single_size, single_size, bias=False)
        self.pair_recycle_projection = nn.Linear(pair_size, pair_size, bias=False)

        # BoltzGen swap 1: v1 template module (a_proj/pairformer pattern), not the v2 embedder.
        self.template_module = TemplateModule(
            pair_size=pair_size,
            template_size=design.template_size,
            num_blocks=design.template_blocks,
            vocab_size=embedder.vocab_size,
            num_bins=config.template.num_distogram_bins,
            min_dist=config.template.min_distance,
            max_dist=config.template.max_distance,
            triangle_attention_hidden_size=config.template.triangle_attention_hidden_size,
            num_triangle_attention_heads=config.template.num_triangle_attention_heads,
            miniformer_blocks=config.template.miniformer_blocks,
            operator_backend=trunk.operator_backend,
        )
        # BoltzGen swap 2: token-distance conditioning module. The structuretrained-small artifact omits it.
        self.token_distance = (
            TokenDistanceModule(
                pair_size=pair_size,
                token_distance_size=design.token_distance_size,
                num_blocks=design.token_distance_blocks,
                num_bins=config.template.num_distogram_bins,
                distance_gaussian_dim=design.distance_gaussian_dim,
                min_dist=config.template.min_distance,
                max_dist=config.template.max_distance,
                triangle_attention_hidden_size=config.template.triangle_attention_hidden_size,
                num_triangle_attention_heads=config.template.num_triangle_attention_heads,
                operator_backend=trunk.operator_backend,
            )
            if design.use_token_distance
            else None
        )

        self.msa_module = (
            MsaModule(
                single_size=single_size,
                pair_size=pair_size,
                msa_size=trunk.msa_size,
                msa_feature_size=trunk.msa_feature_size,
                num_blocks=trunk.num_msa_blocks,
                vocab_size=embedder.vocab_size,
                use_paired_feature=trunk.use_paired_feature,
                outer_product_hidden_size=trunk.outer_product_hidden_size,
                msa_dropout=trunk.msa_dropout,
                pair_dropout=trunk.dropout,
                num_msa_attention_heads=trunk.num_msa_attention_heads,
                msa_head_dim=trunk.msa_head_dim,
                triangle_attention_hidden_size=trunk.triangle_attention_hidden_size,
                num_triangle_attention_heads=trunk.num_triangle_attention_heads,
                operator_backend=trunk.operator_backend,
            )
            if not trunk.msa_miniformer_blocks
            else MiniformerMsaModule(
                single_size=single_size,
                pair_size=pair_size,
                msa_size=trunk.msa_size,
                msa_feature_size=trunk.msa_feature_size,
                num_blocks=trunk.num_msa_blocks,
                vocab_size=embedder.vocab_size,
                use_paired_feature=trunk.use_paired_feature,
                outer_product_hidden_size=trunk.outer_product_hidden_size,
                msa_dropout=trunk.msa_dropout,
                pair_dropout=trunk.dropout,
                num_msa_attention_heads=trunk.num_msa_attention_heads,
                msa_head_dim=trunk.msa_head_dim,
                operator_backend=trunk.operator_backend,
            )
        )
        self.pairformer_stack = (
            MiniformerStack(
                num_blocks=trunk.num_pairformer_blocks,
                single_size=single_size,
                pair_size=pair_size,
                num_pair_attention_heads=trunk.num_pair_attention_heads,
                dropout=trunk.dropout,
                operator_backend=trunk.operator_backend,
            )
            if trunk.use_miniformer
            else PairformerStack(
                num_blocks=trunk.num_pairformer_blocks,
                single_size=single_size,
                pair_size=pair_size,
                triangle_attention_hidden_size=trunk.triangle_attention_hidden_size,
                num_triangle_attention_heads=trunk.num_triangle_attention_heads,
                num_pair_attention_heads=trunk.num_pair_attention_heads,
                dropout=trunk.dropout,
                operator_backend=trunk.operator_backend,
            )
        )
        self.distogram_head = DistogramHead(
            pair_size=pair_size, num_bins=config.num_distogram_bins, symmetrize_before_projection=True
        )

        self.diffusion_conditioning = DiffusionConditioning(
            single_size=single_size,
            pair_size=pair_size,
            atom_hidden_size=diffusion.atom_hidden_size,
            atom_pair_hidden_size=diffusion.atom_pair_hidden_size,
            atom_feature_size=config.atom_feature_size,
            num_query_atoms=diffusion.num_query_atoms,
            num_key_atoms=diffusion.num_key_atoms,
            num_atom_encoder_blocks=diffusion.num_atom_encoder_blocks,
            num_atom_encoder_attention_heads=diffusion.num_atom_attention_heads,
            num_atom_decoder_blocks=diffusion.num_atom_decoder_blocks,
            num_atom_decoder_attention_heads=diffusion.num_atom_attention_heads,
            num_token_transformer_blocks=diffusion.num_token_transformer_blocks,
            num_token_transformer_heads=diffusion.num_token_transformer_heads,
            conditioning_transition_layers=diffusion.conditioning_transition_layers,
        )
        score_model = DiffusionScoreModel(
            single_size=single_size,
            atom_hidden_size=diffusion.atom_hidden_size,
            fourier_embedding_size=config.fourier_embedding_size,
            num_query_atoms=diffusion.num_query_atoms,
            num_key_atoms=diffusion.num_key_atoms,
            num_atom_encoder_blocks=diffusion.num_atom_encoder_blocks,
            num_atom_encoder_attention_heads=diffusion.num_atom_attention_heads,
            num_token_transformer_blocks=diffusion.num_token_transformer_blocks,
            num_token_transformer_heads=diffusion.num_token_transformer_heads,
            num_atom_decoder_blocks=diffusion.num_atom_decoder_blocks,
            num_atom_decoder_attention_heads=diffusion.num_atom_attention_heads,
            conditioning_transition_layers=diffusion.conditioning_transition_layers,
            operator_backend=diffusion.operator_backend,
        )
        self.structure_module = AtomDiffusion(
            score_model,
            sigma_data=diffusion.sigma_data,
            min_noise_level=sampler.min_noise_level,
            max_noise_level=sampler.max_noise_level,
            rho=sampler.rho,
            churn_rate=sampler.churn_rate,
            noise_scale=sampler.noise_scale,
            step_scale=sampler.step_scale,
            num_sampling_steps=sampler.num_sampling_steps,
            alignment_reverse_diff=diffusion.alignment_reverse_diff,
            sampling_schedule=sampler.sampling_schedule,
            time_dilation=sampler.time_dilation,
            time_dilation_start=sampler.time_dilation_start,
            time_dilation_end=sampler.time_dilation_end,
        )
        self.bfactor_head = BFactorHead(single_size=single_size, num_bins=config.num_distogram_bins)

        # Opt-in scoring heads (base design generator omits both; enabling either grows the parameter count).
        if config.use_confidence_head:
            confidence = config.confidence
            self.confidence_head = ConfidenceHead(
                single_size=single_size,
                pair_size=pair_size,
                num_blocks=confidence.num_pairformer_blocks,
                num_pair_attention_heads=confidence.num_pair_attention_heads,
                num_distogram_bins=confidence.num_distogram_bins,
                max_distance=confidence.max_distance,
                relative_position_size=config.relative_position_size,
                num_bond_types=config.num_bond_types,
                conditioning_cutoff_min=config.conditioning_cutoff_min,
                conditioning_cutoff_max=config.conditioning_cutoff_max,
                num_plddt_bins=confidence.num_plddt_bins,
                num_pae_bins=confidence.num_pae_bins,
                num_pde_bins=confidence.num_pde_bins,
                operator_backend=trunk.operator_backend,
            )
        if config.use_affinity_head:
            affinity = config.affinity
            if affinity.ensemble:
                self.affinity_module1 = self._build_affinity_head(affinity, affinity.num_pairformer_blocks1)
                self.affinity_module2 = self._build_affinity_head(affinity, affinity.num_pairformer_blocks2)
            else:
                self.affinity_module = self._build_affinity_head(affinity, affinity.num_pairformer_blocks)
        self.post_init()

    def _build_affinity_head(self, affinity: BoltzGenAffinityConfig, num_blocks: int) -> AffinityHead:
        r"""
        Build one affinity head with the shared affinity-config dims and the given Pairformer depth.

        Args:
            affinity: Affinity-head sub-configuration supplying the shared dimensions.
            num_blocks: Number of Pairformer blocks for this head (per-ensemble-member depth).

        Returns:
            The constructed affinity head.
        """

        return AffinityHead(
            single_size=self.config.single_size,
            pair_size=self.config.pair_size,
            num_blocks=num_blocks,
            num_pair_attention_heads=affinity.num_pair_attention_heads,
            num_distogram_bins=affinity.num_distogram_bins,
            max_distance=affinity.max_distance,
            num_conditioning_transitions=affinity.num_conditioning_transitions,
            expansion=affinity.expansion,
            triangle_attention_hidden_size=affinity.triangle_attention_hidden_size,
            num_triangle_attention_heads=affinity.num_triangle_attention_heads,
            dropout=affinity.dropout,
            operator_backend=self.config.trunk.operator_backend,
        )

    def reset_parameters(self) -> None:
        r"""
        Zero the recycle projections and apply the BoltzGen final-layer / conditioning zero-init scheme.
        """

        nn.init.zeros_(self.single_recycle_projection.weight)
        nn.init.zeros_(self.pair_recycle_projection.weight)
        self.input_embedder.reset_parameters()
        _init_boltzgen_final_weights(self)

    def forward(
        self,
        batch: BoltzGenInput,
        *,
        recycling_steps: int | None = None,
        num_samples: int = 1,
        num_sampling_steps: int | None = None,
        generator: torch.Generator | None = None,
        step_scale: float | None = None,
        noise_scale: float | None = None,
    ) -> DesignGeneratorOutput:
        r"""
        Run BoltzGen as a structure-generating design model.

        Args:
            batch: Typed model input.
            recycling_steps: Optional number of recycling passes. Defaults to the config value.
            num_samples: Number of reverse-diffusion samples to draw.
            num_sampling_steps: Optional number of EDM sampling steps. Defaults to the sampler config value.
            generator: Optional PyTorch random generator for diffusion noise.
            step_scale: Optional per-call sampler step-scale override.
            noise_scale: Optional per-call sampler noise-scale override.

        Returns:
            Design-generator output with ``structure_prediction`` populated.

        Raises:
            ValueError: If ``num_samples`` is not positive.

        Examples:
            >>> from ofoldx import AutoModel, AutoProcessor  # doctest: +SKIP
            >>> model = AutoModel.from_pretrained("oteam/boltzgen")  # doctest: +SKIP
            >>> output = model(AutoProcessor.from_pretrained("oteam/boltzgen")("complex.cif"))  # doctest: +SKIP
        """

        if num_samples < 1:
            raise ValueError(f"num_samples must be positive, got {num_samples}")
        steps = self._recycling_steps if recycling_steps is None else recycling_steps
        output = self.predict(
            batch,
            recycling_steps=steps,
            num_sampling_steps=num_sampling_steps,
            num_samples=num_samples,
            generator=generator,
            step_scale=step_scale,
            noise_scale=noise_scale,
        )
        batch_size = batch.token_mask.shape[0]
        bfactor_logits = output.bfactor_logits.unsqueeze(1).expand(-1, num_samples, -1, -1)
        confidence_axis = partial(_sample_axis, batch_size=batch_size, num_samples=num_samples)
        structure_prediction = StructurePredictionOutput(
            pred_coords=_sample_axis(output.pred_coords, batch_size=batch_size, num_samples=num_samples),
            distogram_logits=output.distogram_logits,
            bfactor_logits=bfactor_logits,
            plddt_logits=None if output.plddt_logits is None else confidence_axis(output.plddt_logits),
            pae_logits=None if output.pae_logits is None else confidence_axis(output.pae_logits),
            pde_logits=None if output.pde_logits is None else confidence_axis(output.pde_logits),
            resolved_logits=None if output.resolved_logits is None else confidence_axis(output.resolved_logits),
        )
        return DesignGeneratorOutput(structure_prediction=structure_prediction)

    @torch.no_grad()
    def generate(
        self,
        batch: BoltzGenInput,
        *,
        recycling_steps: int | None = None,
        num_samples: int = 1,
        num_sampling_steps: int | None = None,
        generator: torch.Generator | None = None,
        step_scale: float | None = None,
        noise_scale: float | None = None,
    ) -> DesignGeneratorOutput:
        r"""
        Generate structure-design output without tracking gradients (a ``torch.no_grad`` wrapper around ``forward``).

        Args:
            batch: Typed model input.
            recycling_steps: Optional number of recycling passes. Defaults to the config value.
            num_samples: Number of reverse-diffusion samples to draw.
            num_sampling_steps: Optional number of EDM sampling steps. Defaults to the sampler config value.
            generator: Optional PyTorch random generator for diffusion noise.
            step_scale: Optional per-call sampler step-scale override.
            noise_scale: Optional per-call sampler noise-scale override.

        Returns:
            Design-generator output with ``structure_prediction`` populated.

        Examples:
            >>> from ofoldx import AutoModel, AutoProcessor  # doctest: +SKIP
            >>> model = AutoModel.from_pretrained("oteam/boltzgen")  # doctest: +SKIP
            >>> batch = AutoProcessor.from_pretrained("oteam/boltzgen")("complex.cif")  # doctest: +SKIP
            >>> output = model.generate(batch)  # doctest: +SKIP
        """

        return self.forward(
            batch,
            recycling_steps=recycling_steps,
            num_samples=num_samples,
            num_sampling_steps=num_sampling_steps,
            generator=generator,
            step_scale=step_scale,
            noise_scale=noise_scale,
        )

    def predict(
        self,
        batch: BoltzGenInput,
        *,
        recycling_steps: int | None = None,
        num_sampling_steps: int | None = None,
        num_samples: int = 1,
        generator: torch.Generator | None = None,
        step_scale: float | None = None,
        noise_scale: float | None = None,
    ) -> BoltzGenStructureOutput:
        r"""
        Full structure generation: trunk -> diffusion-sampled coords -> distogram / bfactor logits.

        Args:
            batch: Typed model input.
            recycling_steps: Optional number of recycling passes. Defaults to the config value.
            num_sampling_steps: Optional number of EDM sampling steps. Defaults to the sampler config value.
            num_samples: Number of reverse-diffusion samples to draw.
            generator: Optional PyTorch random generator for diffusion noise.
            step_scale: Optional per-call sampler step-scale override.
            noise_scale: Optional per-call sampler noise-scale override.

        Returns:
            Model-specific structure-generation output.

        Raises:
            ValueError: If ``num_samples`` is not positive.
        """

        if num_samples < 1:
            raise ValueError(f"num_samples must be positive, got {num_samples}")
        representations = self.run_trunk(batch, recycling_steps=recycling_steps)
        conditioning = self.diffusion_conditioning(
            representations.single,
            representations.pair,
            representations.relative_position_encoding,
            atom_features=batch.atom_features,
            reference_positions=batch.reference_positions,
            atom_mask=batch.atom_mask,
            reference_space_id=batch.reference_space_id,
            atom_to_token=batch.atom_to_token,
        )
        pred_coords = self.structure_module.sample(
            batch.atom_mask,
            num_sampling_steps=num_sampling_steps,
            multiplicity=num_samples,
            generator=generator,
            step_scale=step_scale,
            noise_scale=noise_scale,
            single_inputs=representations.single_inputs,
            single_trunk=representations.single,
            conditioning=conditioning,
            atom_to_token=batch.atom_to_token,
            token_mask=batch.token_mask,
        )
        confidence: dict[str, Tensor] = {}
        if hasattr(self, "confidence_head"):
            confidence = self.score_confidence(representations, batch, pred_coords, multiplicity=num_samples)
        return BoltzGenStructureOutput(
            pred_coords=pred_coords,
            distogram_logits=representations.distogram_logits,
            bfactor_logits=self.bfactor_head(representations.single),
            plddt_logits=confidence.get("plddt_logits"),
            pae_logits=confidence.get("pae_logits"),
            pde_logits=confidence.get("pde_logits"),
            resolved_logits=confidence.get("resolved_logits"),
        )

    def score_confidence(
        self,
        representations: BoltzGenRepresentations,
        batch: BoltzGenInput,
        pred_coords: Tensor,
        *,
        multiplicity: int = 1,
    ) -> dict[str, Tensor]:
        r"""
        Score sampled coords with the (opt-in) confidence head, returning pLDDT / pAE / pDE / resolved logits.

        ``pred_coords`` is the flattened ``[B * multiplicity, num_atoms, 3]`` diffusion sample; the trunk
        representations are interleaved internally to match its multiplicity.

        Args:
            representations: Trunk single/pair representations produced by the forward pass.
            batch: The typed model input carrying token/atom indexing and conditioning features.
            pred_coords: Flattened sampled atom coordinates of shape `[B * multiplicity, num_atoms, 3]`.
            multiplicity: Number of samples per input drawn during diffusion; the trunk tensors are repeated this
                many times to align with ``pred_coords``.

        Returns:
            A mapping with ``plddt_logits``, ``pae_logits``, ``pde_logits``, and ``resolved_logits`` tensors from
            the confidence head.

        Raises:
            RuntimeError: If the confidence head was not enabled (``use_confidence_head=False``) in the config.
        """

        if not hasattr(self, "confidence_head"):
            raise RuntimeError("confidence head is disabled; set use_confidence_head=True to enable scoring")
        return self.confidence_head(
            representations.single_inputs,
            representations.single,
            representations.pair,
            pred_coords,
            residue_index=batch.residue_index,
            chain_index=batch.chain_index,
            entity_index=batch.entity_index,
            symmetry_index=batch.symmetry_index,
            token_index=batch.token_index,
            token_bonds=batch.token_bonds,
            type_bonds=batch.type_bonds,
            contact_conditioning=batch.contact_conditioning,
            contact_threshold=batch.contact_threshold,
            token_to_rep_atom=batch.token_to_rep_atom,
            token_mask=batch.token_mask,
            cyclic_period=batch.cyclic_period,
            multiplicity=multiplicity,
        )

    def score_affinity(
        self,
        representations: BoltzGenRepresentations,
        batch: BoltzGenInput,
        pred_coords: Tensor,
        *,
        multiplicity: int = 1,
        affinity_token_mask: Tensor | None = None,
        affinity_mw: Tensor | None = None,
    ) -> dict[str, Tensor]:
        r"""
        Score sampled coords with the (opt-in) affinity head, returning affinity value and binder-probability
        tensors (ensemble-averaged and molecular-weight-corrected when configured).

        ``pred_coords`` is the flattened ``[B * multiplicity, num_atoms, 3]`` diffusion sample.

        Args:
            representations: Trunk single/pair representations produced by the forward pass.
            batch: The typed model input carrying token/atom indexing, chain/molecule types, and affinity metadata.
            pred_coords: Flattened sampled atom coordinates of shape `[B * multiplicity, num_atoms, 3]`.
            multiplicity: Number of samples per input drawn during diffusion; the trunk tensors are repeated this
                many times to align with ``pred_coords``.
            affinity_token_mask: Optional `[..., num_tokens]` boolean mask selecting the ligand tokens; falls back
                to the mask resolved from ``batch`` when ``None``.
            affinity_mw: Optional `[...]` ligand molecular weights (daltons) driving the molecular-weight
                correction; falls back to ``batch.affinity_mw`` when ``None``.

        Returns:
            A mapping with ``affinity_pred_value`` and ``affinity_probability_binary`` (plus per-module
            ``*1`` / ``*2`` entries under the ensemble), after any molecular-weight correction.

        Raises:
            RuntimeError: If the affinity head was not enabled (``use_affinity_head=False``) in the config.
        """

        if not (hasattr(self, "affinity_module") or hasattr(self, "affinity_module1")):
            raise RuntimeError("affinity head is disabled; set use_affinity_head=True to enable scoring")
        ligand_mask = _resolve_affinity_token_mask(batch, affinity_token_mask)
        if batch.molecule_type is None:
            receptor_mask = batch.chain_type == 0
        else:
            receptor_mask = batch.molecule_type == 0
        head_kwargs = {
            "single_inputs": representations.single_inputs,
            "pair": representations.pair,
            "pred_coords": pred_coords,
            "token_to_rep_atom": batch.token_to_rep_atom,
            "token_mask": batch.token_mask,
            "ligand_mask": ligand_mask,
            "receptor_mask": receptor_mask,
            "multiplicity": multiplicity,
        }
        if self.config.affinity.ensemble:
            scores1 = self.affinity_module1(**head_kwargs)
            scores2 = self.affinity_module2(**head_kwargs)
            probability1 = torch.sigmoid(scores1["affinity_logits_binary"])
            probability2 = torch.sigmoid(scores2["affinity_logits_binary"])
            scores = {
                "affinity_pred_value": (scores1["affinity_pred_value"] + scores2["affinity_pred_value"]) / 2,
                "affinity_probability_binary": (probability1 + probability2) / 2,
                "affinity_pred_value1": scores1["affinity_pred_value"],
                "affinity_logits_binary1": scores1["affinity_logits_binary"],
                "affinity_probability_binary1": probability1,
                "affinity_pred_value2": scores2["affinity_pred_value"],
                "affinity_logits_binary2": scores2["affinity_logits_binary"],
                "affinity_probability_binary2": probability2,
            }
        else:
            scores = self.affinity_module(**head_kwargs)
            scores["affinity_probability_binary"] = torch.sigmoid(scores["affinity_logits_binary"])
        scores["affinity_pred_value"] = self._apply_molecular_weight_correction(
            scores["affinity_pred_value"], batch, affinity_mw=affinity_mw
        )
        return scores

    def _apply_molecular_weight_correction(
        self, affinity_pred_value: Tensor, batch: BoltzGenInput, *, affinity_mw: Tensor | None
    ) -> Tensor:
        r"""
        Apply the affinity molecular-weight correction ``a*value + b*MW**e + c`` when enabled.
        """

        affinity = self.config.affinity
        if not affinity.molecular_weight_correction:
            return affinity_pred_value
        if affinity_mw is None:
            affinity_mw = batch.affinity_mw
        if affinity_mw is None:
            raise ValueError("affinity_mw is required when molecular_weight_correction is enabled")
        mw = affinity_mw.to(device=affinity_pred_value.device, dtype=affinity_pred_value.dtype).reshape(-1)
        if mw.shape[0] != affinity_pred_value.shape[0]:
            raise ValueError(
                f"affinity_mw must have one value per row, got {mw.shape[0]} for {affinity_pred_value.shape[0]}"
            )
        correction = affinity.molecular_weight_coefficient * mw.pow(affinity.molecular_weight_exponent)
        return (
            affinity.molecular_weight_model_coefficient * affinity_pred_value
            + correction.unsqueeze(-1)
            + affinity.molecular_weight_bias
        )

    def run_trunk(self, batch: BoltzGenInput, *, recycling_steps: int | None = None) -> BoltzGenRepresentations:
        r"""
        Run the structure trunk.

        Args:
            batch: Typed model input.
            recycling_steps: Optional number of recycling passes. Defaults to the config value.

        Returns:
            Trunk representations and distogram logits.

        Raises:
            ValueError: If ``recycling_steps`` is negative or required design/MSA inputs are missing.
        """

        steps = self._recycling_steps if recycling_steps is None else recycling_steps
        if steps < 0:
            raise ValueError(f"recycling_steps must be non-negative, got {steps}")
        binding_type, design_mask, ss_type = _design_conditioning_for_embedder(batch, self.input_embedder)
        single_inputs = self.input_embedder(
            atom_features=batch.atom_features,
            reference_positions=batch.reference_positions,
            atom_mask=batch.atom_mask,
            reference_space_id=batch.reference_space_id,
            atom_to_token=batch.atom_to_token,
            residue_type=batch.residue_type,
            msa_profile=batch.msa_profile,
            deletion_mean=batch.deletion_mean,
            method_feature=batch.method_feature,
            modified=batch.modified,
            cyclic_period=batch.cyclic_period,
            chain_type=batch.chain_type,
            binding_type=binding_type,
            design_mask=design_mask,
            ss_type=ss_type,
        )

        single_init = self.single_init_projection(single_inputs)
        pair_init = self.pair_left_projection(single_inputs).unsqueeze(-2)
        pair_init = pair_init + self.pair_right_projection(single_inputs).unsqueeze(-3)
        relative_features = generate_relative_position_features(
            residue_index=batch.residue_index,
            chain_index=batch.chain_index,
            entity_index=batch.entity_index,
            symmetry_index=batch.symmetry_index,
            token_index=batch.token_index,
            cyclic_period=batch.cyclic_period,
            dtype=single_init.dtype,
        ).to(device=single_init.device)
        relative_position_encoding = self.relative_position_embedder(relative_features)
        pair_init = pair_init + relative_position_encoding
        pair_init = pair_init + self.token_bond_projection(batch.token_bonds.unsqueeze(-1).to(pair_init.dtype))
        pair_init = pair_init + self.token_bond_type_embedding(batch.type_bonds.long())
        pair_init = pair_init + self.contact_conditioning(
            batch.contact_conditioning.to(pair_init.dtype), batch.contact_threshold.to(pair_init.dtype)
        )

        token_mask = batch.token_mask
        pair_mask = pair_mask_from_token_mask(token_mask)
        single = torch.zeros_like(single_init)
        pair = torch.zeros_like(pair_init)
        for cycle in range(steps + 1):
            grad_enabled = torch.is_grad_enabled()
            if self.training and self.config.recycling_detach and cycle < steps:
                grad_enabled = False
            with torch.set_grad_enabled(grad_enabled):
                single = single_init + self.single_recycle_projection(self.single_recycle_norm(single))
                pair = pair_init + self.pair_recycle_projection(self.pair_recycle_norm(pair))
                if self.token_distance is not None:
                    pair = pair + self.token_distance(pair, batch, relative_position_encoding, pair_mask=pair_mask)
                if batch.template_restype is not None:
                    pair = pair + self.template_module(pair, batch, pair_mask=pair_mask)
                if batch.msa is not None:
                    if (
                        batch.has_deletion is None
                        or batch.deletion_value is None
                        or batch.msa_paired is None
                        or batch.msa_mask is None
                    ):
                        raise ValueError("MSA features are incomplete")
                    pair = pair + self.msa_module(
                        pair,
                        single_inputs,
                        msa=batch.msa,
                        has_deletion=batch.has_deletion,
                        deletion_value=batch.deletion_value,
                        is_paired=batch.msa_paired,
                        msa_mask=batch.msa_mask,
                        token_mask=token_mask,
                    )
                single, pair = self.pairformer_stack(single, pair, pair_mask=pair_mask)

        return BoltzGenRepresentations(
            single_inputs=single_inputs,
            single=single,
            pair=pair,
            relative_position_encoding=relative_position_encoding,
            distogram_logits=self.distogram_head(pair),
        )


@MODELS.register("boltzgen_ifold")
@DESIGN_GENERATOR_MODELS.register("boltzgen_ifold")
class BoltzGenIFoldGenerator(ModelMixin):
    r"""
    BoltzGen inverse-folding model for structure-conditioned sequence design.

    Encodes the input structure into a kNN token graph and autoregressively (or in a single teacher-forced pass)
    decodes per-token residue types.

    Args:
        config: Optional inverse-folding config. Defaults to ``BoltzGenIFoldConfig()``.

    Examples:
        >>> from ofoldx import AutoModel, AutoProcessor  # doctest: +SKIP
        >>> model = AutoModel.from_pretrained("oteam/boltzgen-inversefolding")  # doctest: +SKIP
        >>> processor = AutoProcessor.from_pretrained("oteam/boltzgen-inversefolding")  # doctest: +SKIP
        >>> output = model.generate(processor("bb.cif"))  # doctest: +SKIP
    """

    config_class = BoltzGenIFoldConfig

    def __init__(self, config: BoltzGenIFoldConfig | None = None) -> None:
        config = config or BoltzGenIFoldConfig()
        super().__init__(config)
        ifold = config.ifold
        self.input_embedder = AtomConditionedInputEmbedder(
            single_size=config.single_size,
            atom_hidden_size=config.atom_hidden_size,
            atom_pair_hidden_size=config.atom_pair_hidden_size,
            atom_feature_size=config.atom_feature_size,
            num_query_atoms=config.diffusion.num_query_atoms,
            num_key_atoms=config.diffusion.num_key_atoms,
            num_atom_encoder_blocks=ifold.num_atom_encoder_blocks,
            num_atom_encoder_attention_heads=ifold.num_atom_encoder_attention_heads,
            vocab_size=config.input_embedder.vocab_size,
            msa_profile_size=config.input_embedder.msa_profile_size,
            num_method_types=config.input_embedder.num_method_types,
            num_molecule_types=config.input_embedder.num_molecule_types,
            operator_backend=ifold.operator_backend,
        )
        self.encoder = InverseFoldingEncoder(
            single_size=config.single_size,
            node_size=ifold.node_size,
            edge_size=ifold.edge_size,
            hidden_size=ifold.hidden_size,
            num_layers=ifold.num_encoder_layers,
            num_attention_heads=ifold.num_attention_heads,
            k_neighbors=ifold.k_neighbors,
            enable_input_embedder=True,
        )
        self.decoder = InverseFoldingDecoder(
            node_size=ifold.node_size,
            edge_size=ifold.edge_size,
            hidden_size=ifold.hidden_size,
            num_layers=ifold.num_decoder_layers,
            num_attention_heads=ifold.num_attention_heads,
            inverse_fold_restriction=list(ifold.inverse_fold_restriction),
            tie_symmetric_sequences=ifold.tie_symmetric_sequences,
            constraint_inf=ifold.constraint_inf,
        )
        self.post_init()

    def reset_parameters(self) -> None:
        r"""
        Reset the input embedder and apply the BoltzGen final-layer / conditioning zero-init scheme.
        """

        self.input_embedder.reset_parameters()
        _init_boltzgen_final_weights(self)

    def forward(self, batch: BoltzGenInput) -> DesignGeneratorOutput:
        r"""
        Return per-token residue-type logits for structure-conditioned sequence design.

        Args:
            batch: Typed model input.

        Returns:
            Design-generator output carrying sequence logits.
        """

        single_inputs = self.input_embedder(
            atom_features=batch.atom_features,
            reference_positions=batch.reference_positions,
            atom_mask=batch.atom_mask,
            reference_space_id=batch.reference_space_id,
            atom_to_token=batch.atom_to_token,
            residue_type=batch.residue_type,
            msa_profile=batch.msa_profile,
            deletion_mean=batch.deletion_mean,
            method_feature=batch.method_feature,
            modified=batch.modified,
            cyclic_period=batch.cyclic_period,
            chain_type=batch.chain_type,
        )
        edge_index, valid_mask, node, edge = self.encoder(batch, single_inputs=single_inputs)
        return DesignGeneratorOutput(
            sequence_logits=self.decoder(node, edge, edge_index, valid_mask, batch),
        )

    @torch.no_grad()
    def generate(
        self,
        batch: BoltzGenInput,
        *,
        temperature: float | None = None,
        generator: torch.Generator | None = None,
    ) -> DesignGeneratorOutput:
        r"""
        Autoregressively design residue tokens from the conditioned structure.

        Args:
            batch: Typed model input.
            temperature: Optional categorical sampling temperature. ``None`` selects argmax decoding.
            generator: Optional PyTorch random generator for decode order and categorical sampling.

        Returns:
            Design-generator output carrying sampled sequence tokens and logits.
        """

        single_inputs = self.input_embedder(
            atom_features=batch.atom_features,
            reference_positions=batch.reference_positions,
            atom_mask=batch.atom_mask,
            reference_space_id=batch.reference_space_id,
            atom_to_token=batch.atom_to_token,
            residue_type=batch.residue_type,
            msa_profile=batch.msa_profile,
            deletion_mean=batch.deletion_mean,
            method_feature=batch.method_feature,
            modified=batch.modified,
            cyclic_period=batch.cyclic_period,
            chain_type=batch.chain_type,
        )
        edge_index, valid_mask, node, edge = self.encoder(batch, single_inputs=single_inputs)
        output = self.decoder.sample(
            node,
            edge,
            edge_index,
            valid_mask,
            batch,
            temperature=temperature,
            generator=generator,
        )
        return DesignGeneratorOutput(
            sequence_logits=output.logits,
            sequence_tokens=output.residue_type.argmax(dim=-1),
        )


class AtomConditionedInputEmbedder(nn.Module):
    r"""
    Compose atom-derived token states with token-level encodings and conditioning embeddings.

    Runs the windowed atom encoder / atom-attention encoder to build a per-token embedding, then adds the
    residue-type, MSA-profile, and optional conditioning channels (method, modified, cyclic, molecule type, and
    the design-only binding / design-mask / secondary-structure channels gated by the ``add_*`` flags).

    Args:
        single_size: Width of the token single representation.
        atom_hidden_size: Width of the atom hidden channel.
        atom_pair_hidden_size: Width of the atom-pair hidden channel.
        atom_feature_size: Width of the per-atom reference input features.
        num_query_atoms: Number of query atoms per attention window.
        num_key_atoms: Number of key atoms per attention window.
        num_atom_encoder_blocks: Number of blocks in the atom-attention encoder.
        num_atom_encoder_attention_heads: Number of attention heads per atom-encoder block.
        vocab_size: Number of residue-type tokens.
        msa_profile_size: Width of the single-sequence MSA-profile feature.
        num_method_types: Number of experimental-method conditioning ids.
        num_molecule_types: Number of molecule-type conditioning ids.
        add_method_conditioning: Whether to add the experimental-method embedding.
        add_modified_flag: Whether to add the modified-residue embedding.
        add_cyclic_flag: Whether to add the cyclic-position projection.
        add_mol_type_feat: Whether to add the molecule-type embedding.
        add_binding_specification: Whether to add the binding-specification / design-mask conditioning.
        add_ss_specification: Whether to add the secondary-structure conditioning.
        num_binding_types: Number of binding-specification ids.
        num_ss_types: Number of secondary-structure ids.
        operator_backend: Operator backend selector (``"auto"``, ``"oops"``, or ``"public"``).
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
        vocab_size: int = 33,
        msa_profile_size: int = 34,
        num_method_types: int = 12,
        num_molecule_types: int = 4,
        add_method_conditioning: bool = True,
        add_modified_flag: bool = True,
        add_cyclic_flag: bool = True,
        add_mol_type_feat: bool = True,
        add_binding_specification: bool = False,
        add_ss_specification: bool = False,
        num_binding_types: int = 3,
        num_ss_types: int = 4,
        operator_backend: str = "auto",
    ) -> None:
        super().__init__()
        self.atom_encoder = AtomEncoder(
            atom_feature_size=atom_feature_size,
            atom_hidden_size=atom_hidden_size,
            atom_pair_hidden_size=atom_pair_hidden_size,
            num_query_atoms=num_query_atoms,
            num_key_atoms=num_key_atoms,
        )
        self.atom_pair_bias_projection = nn.Sequential(
            LayerNorm(atom_pair_hidden_size),
            nn.Linear(atom_pair_hidden_size, num_atom_encoder_blocks * num_atom_encoder_attention_heads, bias=False),
        )
        self.atom_attention_encoder = AtomAttentionEncoder(
            atom_hidden_size=atom_hidden_size,
            single_size=single_size,
            num_query_atoms=num_query_atoms,
            num_key_atoms=num_key_atoms,
            num_atom_encoder_blocks=num_atom_encoder_blocks,
            num_atom_encoder_attention_heads=num_atom_encoder_attention_heads,
            operator_backend=operator_backend,
        )

        self.residue_type_projection = nn.Linear(vocab_size, single_size, bias=False)
        self.msa_profile_projection = nn.Linear(msa_profile_size, single_size, bias=False)
        if add_method_conditioning:
            self.method_embedding = nn.Embedding(num_method_types, single_size)
        if add_modified_flag:
            self.modified_embedding = nn.Embedding(2, single_size)
        if add_cyclic_flag:
            self.cyclic_projection = nn.Linear(1, single_size, bias=False)
        if add_mol_type_feat:
            self.molecule_type_embedding = nn.Embedding(num_molecule_types, single_size)
        if add_binding_specification:
            self.binding_specification_embedding = nn.Embedding(num_binding_types, single_size)
            self.design_mask_embedding = nn.Embedding(2, single_size)
        if add_ss_specification:
            self.ss_specification_embedding = nn.Embedding(num_ss_types, single_size)

    def reset_parameters(self) -> None:
        r"""
        Zero-init the optional conditioning embeddings and the cyclic projection so they start as no-ops.
        """

        for name in (
            "method_embedding",
            "modified_embedding",
            "molecule_type_embedding",
            "binding_specification_embedding",
            "design_mask_embedding",
            "ss_specification_embedding",
        ):
            module = getattr(self, name, None)
            if module is not None:
                nn.init.zeros_(module.weight)
        if hasattr(self, "cyclic_projection"):
            nn.init.zeros_(self.cyclic_projection.weight)

    def forward(self, **inputs: Tensor) -> Tensor:
        r"""
        Build the token single embedding from atom-derived states and token encodings.

        Args:
            **inputs: Keyword tensors forwarded to the internal single-embedding builder (atom features and
                reference geometry, ``atom_to_token``, ``residue_type``, ``msa_profile``, and the optional
                conditioning tensors). See ``_forward_single``.

        Returns:
            Token single representation ``(B, N_token, single_size)``.
        """

        return self._forward_single(**inputs)

    def _forward_single(
        self,
        *,
        atom_features: Tensor,
        reference_positions: Tensor,
        atom_mask: Tensor,
        reference_space_id: Tensor,
        atom_to_token: Tensor,
        residue_type: Tensor,
        msa_profile: Tensor,
        deletion_mean: Tensor,
        method_feature: Tensor | None = None,
        modified: Tensor | None = None,
        cyclic_period: Tensor | None = None,
        chain_type: Tensor | None = None,
        binding_type: Tensor | None = None,
        design_mask: Tensor | None = None,
        ss_type: Tensor | None = None,
    ) -> Tensor:
        r"""
        Build the token single embedding: atom-attention token pool plus the additive conditioning channels.

        Args:
            atom_features: Per-atom reference features ``(B, N_atom, atom_feature_size)``.
            reference_positions: Reference-conformer atom coords ``(B, N_atom, 3)``.
            atom_mask: Boolean atom validity mask ``(B, N_atom)``.
            reference_space_id: Reference-conformer group id per atom ``(B, N_atom)``.
            atom_to_token: Token index per atom ``(B, N_atom)``.
            residue_type: Residue-type one-hot ``(B, N_token, vocab_size)``.
            msa_profile: Single-sequence MSA profile ``(B, N_token, msa_profile_size - 1)``.
            deletion_mean: Per-token deletion mean ``(B, N_token)``.
            method_feature: Optional experimental-method ids ``(B, N_token)``.
            modified: Optional modified-residue flags ``(B, N_token)``.
            cyclic_period: Optional cyclic-position periods ``(B, N_token)``.
            chain_type: Optional molecule-type ids ``(B, N_token)``.
            binding_type: Optional binding-specification ids ``(B, N_token)``.
            design_mask: Optional designable-token mask ``(B, N_token)``.
            ss_type: Optional secondary-structure ids ``(B, N_token)``.

        Returns:
            Token single representation ``(B, N_token, single_size)``.

        Raises:
            ValueError: If ``chain_type`` is missing while the molecule-type embedding is enabled.
        """

        atom_query, atom_conditioning, atom_pair = self.atom_encoder(
            atom_features, reference_positions, atom_mask, reference_space_id
        )
        atom_pair_bias = self.atom_pair_bias_projection(atom_pair)
        token_from_atom = self.atom_attention_encoder(
            atom_query, atom_conditioning, atom_pair_bias, atom_mask, atom_to_token, residue_type.shape[-2]
        )

        single = token_from_atom
        single = single + self.residue_type_projection(residue_type.to(single.dtype))
        msa_features = torch.cat([msa_profile, deletion_mean.unsqueeze(-1)], dim=-1)
        single = single + self.msa_profile_projection(msa_features.to(single.dtype))

        if hasattr(self, "method_embedding"):
            single = single + self.method_embedding(method_feature)
        if hasattr(self, "modified_embedding"):
            single = single + self.modified_embedding(modified)
        if hasattr(self, "cyclic_projection"):
            assert cyclic_period is not None
            single = single + self.cyclic_projection(cyclic_period.clamp(max=1.0).unsqueeze(-1).to(single.dtype))
        if hasattr(self, "molecule_type_embedding"):
            if chain_type is None:
                raise ValueError("chain_type is required when molecule_type_embedding is enabled")
            single = single + self.molecule_type_embedding(chain_type)
        if hasattr(self, "design_mask_embedding"):
            assert design_mask is not None
            single = single + self.design_mask_embedding(design_mask.int())
        if hasattr(self, "binding_specification_embedding"):
            single = single + self.binding_specification_embedding(binding_type)
        if hasattr(self, "ss_specification_embedding"):
            single = single + self.ss_specification_embedding(ss_type)
        return single


class AtomEncoder(nn.Module):
    r"""
    Build atom single states and windowed atom-pair states from reference geometry.

    Projects the per-atom features into the atom single channel and encodes windowed atom-pair states from the
    reference distance vectors, inverse distances, and validity mask. When ``structure_prediction`` is set it also
    folds in the trunk single / pair projections (used inside the diffusion conditioning).

    Args:
        atom_feature_size: Width of the per-atom reference input features.
        atom_hidden_size: Width of the atom hidden channel.
        atom_pair_hidden_size: Width of the atom-pair hidden channel.
        single_size: Width of the trunk single representation (only used when ``structure_prediction``).
        pair_size: Width of the trunk pair representation (only used when ``structure_prediction``).
        num_query_atoms: Number of query atoms per attention window.
        num_key_atoms: Number of key atoms per attention window.
        structure_prediction: Whether to consume trunk single / pair projections (diffusion-conditioning mode).
    """

    def __init__(
        self,
        atom_feature_size: int = 388,
        atom_hidden_size: int = 128,
        atom_pair_hidden_size: int = 16,
        single_size: int = 384,
        pair_size: int = 128,
        num_query_atoms: int = 32,
        num_key_atoms: int = 128,
        structure_prediction: bool = False,
    ) -> None:
        super().__init__()
        self.num_query_atoms = num_query_atoms
        self.num_key_atoms = num_key_atoms
        self.structure_prediction = structure_prediction
        self.atom_feature_projection = nn.Linear(atom_feature_size, atom_hidden_size)
        self.distance_vector_projection = nn.Linear(3, atom_pair_hidden_size, bias=False)
        self.inverse_distance_projection = nn.Linear(1, atom_pair_hidden_size, bias=False)
        self.distance_mask_projection = nn.Linear(1, atom_pair_hidden_size, bias=False)
        if structure_prediction:
            self.single_trunk_projection = nn.Sequential(
                LayerNorm(single_size), nn.Linear(single_size, atom_hidden_size, bias=False)
            )
            self.pair_trunk_projection = nn.Sequential(
                LayerNorm(pair_size),
                nn.Linear(pair_size, atom_pair_hidden_size, bias=False),
            )
        self.left_atom_projection = nn.Sequential(
            nn.ReLU(),
            nn.Linear(atom_hidden_size, atom_pair_hidden_size, bias=False),
        )
        self.right_atom_projection = nn.Sequential(
            nn.ReLU(),
            nn.Linear(atom_hidden_size, atom_pair_hidden_size, bias=False),
        )
        self.feed_forward = nn.Sequential(
            nn.ReLU(),
            nn.Linear(atom_pair_hidden_size, atom_pair_hidden_size, bias=False),
            nn.ReLU(),
            nn.Linear(atom_pair_hidden_size, atom_pair_hidden_size, bias=False),
            nn.ReLU(),
            nn.Linear(atom_pair_hidden_size, atom_pair_hidden_size, bias=False),
        )

    def forward(
        self,
        atom_features: Tensor,
        reference_positions: Tensor,
        atom_mask: Tensor,
        reference_space_id: Tensor,
        *,
        atom_to_token: Tensor | None = None,
        single_trunk: Tensor | None = None,
        pair_trunk: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        r"""
        Return atom query states, atom conditioning states, and windowed atom-pair states.

        Args:
            atom_features: Per-atom reference features ``(B, N_atom, atom_feature_size)``.
            reference_positions: Reference-conformer atom coords ``(B, N_atom, 3)``.
            atom_mask: Boolean atom validity mask ``(B, N_atom)``.
            reference_space_id: Reference-conformer group id per atom ``(B, N_atom)``.
            atom_to_token: Token index per atom ``(B, N_atom)``; required in structure-prediction mode.
            single_trunk: Trunk single representation ``(B, N_token, single_size)``; used in structure-prediction
                mode.
            pair_trunk: Trunk pair representation ``(B, N_token, N_token, pair_size)``; used in structure-prediction
                mode.

        Returns:
            Tuple of atom query states ``(B, N_atom, atom_hidden_size)``, atom conditioning states
            ``(B, N_atom, atom_hidden_size)``, and windowed atom-pair states
            ``(B, N_window, N_query, N_key, atom_pair_hidden_size)``.
        """

        atom_query = self.atom_feature_projection(atom_features)
        atom_single = atom_query

        num_query_atoms, num_key_atoms = self.num_query_atoms, self.num_key_atoms
        batch, length = atom_single.shape[:2]
        padding = (-length) % num_query_atoms
        num_windows = (length + padding) // num_query_atoms
        to_keys = partial(
            gather_key_windows,
            num_query_atoms=num_query_atoms,
            num_key_atoms=num_key_atoms,
        )

        def pad_atoms(value: Tensor) -> Tensor:
            r"""
            Right-pad the atom axis (dim 1) of ``value`` by ``padding`` zeros so it tiles into whole windows.

            Args:
                value: A tensor whose dim 1 is the per-atom axis, e.g. `[B, num_atoms, ...]`.

            Returns:
                ``value`` with ``padding`` trailing zeros appended along dim 1 (unchanged when ``padding`` is 0).
            """
            if not padding:
                return value
            return torch.nn.functional.pad(value, [0, 0] * (value.dim() - 2) + [0, padding])

        positions = pad_atoms(reference_positions)
        positions_queries = positions.view(batch, num_windows, num_query_atoms, 1, 3)
        positions_keys = to_keys(positions).view(batch, num_windows, 1, num_key_atoms, 3)
        distance_vectors = positions_keys - positions_queries

        mask = pad_atoms(atom_mask).bool()
        mask_queries = mask.view(batch, num_windows, num_query_atoms, 1)
        mask_keys = to_keys(mask.unsqueeze(-1).float()).view(batch, num_windows, 1, num_key_atoms).bool()
        space_id = pad_atoms(reference_space_id)
        uid_queries = space_id.view(batch, num_windows, num_query_atoms, 1)
        uid_keys = to_keys(space_id.unsqueeze(-1).float()).view(batch, num_windows, 1, num_key_atoms).long()
        valid = (mask_queries & mask_keys & (uid_queries == uid_keys)).to(distance_vectors.dtype).unsqueeze(-1)

        pair_bias = None
        if self.structure_prediction:
            assert atom_to_token is not None
            token_index = atom_to_token
            single_proj = self.single_trunk_projection(single_trunk)
            atom_single = atom_single + single_proj.gather(
                1, token_index.unsqueeze(-1).expand(-1, -1, single_proj.shape[-1])
            )
            pair_from_trunk = self.pair_trunk_projection(pair_trunk)
            token_windows = pad_atoms(token_index)
            query_token = token_windows.view(batch, num_windows, num_query_atoms)
            key_token = to_keys(token_windows.unsqueeze(-1)).reshape(batch, num_windows, num_key_atoms)
            batch_index = torch.arange(batch, device=atom_single.device)[:, None, None, None]
            pair_gathered = pair_from_trunk[batch_index, query_token[:, :, :, None], key_token[:, :, None, :]]
            pair_bias = pair_gathered * mask_keys.unsqueeze(-1).to(distance_vectors.dtype)

        single_windows = pad_atoms(atom_single)
        atom_pair = encode_atom_pair(
            self,
            distance_vectors,
            single_windows.view(batch, num_windows, num_query_atoms, 1, -1),
            to_keys(single_windows).view(batch, num_windows, 1, num_key_atoms, -1),
            distance_term_mask=valid,
            inverse_term_mask=valid,
            valid_mask=valid,
            pair_bias=pair_bias,
            prefix_ndim=1,
        )
        return atom_query, atom_single, atom_pair


class AtomAttentionEncoder(nn.Module):
    r"""
    Windowed atom-attention encoder with an atom-to-token mean pool.

    Runs the packed windowed atom transformer over atom states and mean-pools to token states. In
    structure-prediction mode it also injects the noisy positions and returns the atom / conditioning skip states
    for the decoder.

    Args:
        atom_hidden_size: Width of the atom hidden channel.
        single_size: Width of the token single representation.
        num_query_atoms: Number of query atoms per attention window.
        num_key_atoms: Number of key atoms per attention window.
        num_atom_encoder_blocks: Number of atom-transformer blocks.
        num_atom_encoder_attention_heads: Number of attention heads per block.
        structure_prediction: Whether to inject noisy positions and return diffusion skip states.
        operator_backend: Operator backend selector (``"auto"``, ``"oops"``, or ``"public"``).
    """

    def __init__(
        self,
        atom_hidden_size: int = 128,
        single_size: int = 384,
        num_query_atoms: int = 32,
        num_key_atoms: int = 128,
        num_atom_encoder_blocks: int = 3,
        num_atom_encoder_attention_heads: int = 4,
        structure_prediction: bool = False,
        operator_backend: str = "auto",
    ) -> None:
        super().__init__()
        self.structure_prediction = structure_prediction
        if structure_prediction:
            self.noisy_position_projection = nn.Linear(3, atom_hidden_size, bias=False)
        self.atom_transformer = PackedAtomAttention(
            atom_hidden_size=atom_hidden_size,
            conditioning_size=atom_hidden_size,
            num_blocks=num_atom_encoder_blocks,
            num_attention_heads=num_atom_encoder_attention_heads,
            num_query_atoms=num_query_atoms,
            num_key_atoms=num_key_atoms,
            operator_backend=operator_backend,
            cross_attention_mode=False,
            transition="triple",
        )
        token_out = 2 * single_size if structure_prediction else single_size
        self.token_projection = nn.Sequential(nn.Linear(atom_hidden_size, token_out, bias=False), nn.ReLU())

    def forward(
        self,
        atom_single: Tensor,
        atom_conditioning: Tensor,
        atom_pair_bias: Tensor,
        atom_mask: Tensor,
        atom_to_token: Tensor,
        num_tokens: int,
        noisy_positions: Tensor | None = None,
        multiplicity: int = 1,
    ) -> Tensor | tuple[Tensor, Tensor, Tensor]:
        r"""
        Return token representations, and the diffusion skip states when structure prediction is enabled.

        Args:
            atom_single: Atom single states ``(B, N_atom, atom_hidden_size)``.
            atom_conditioning: Atom conditioning states ``(B, N_atom, atom_hidden_size)``.
            atom_pair_bias: Windowed atom-pair attention bias.
            atom_mask: Boolean atom validity mask ``(B, N_atom)``.
            atom_to_token: Token index per atom ``(B, N_atom)``.
            num_tokens: Number of tokens to pool into.
            noisy_positions: Noised atom coords ``(B*multiplicity, N_atom, 3)`` (structure-prediction mode).
            multiplicity: Number of diffusion samples folded into the leading axis.

        Returns:
            Token representation ``(B, N_token, single_size)``; in structure-prediction mode a tuple of the
            token representation (width ``2*single_size``), the atom skip states, and the conditioning skip states.
        """

        query = atom_single
        conditioning = atom_conditioning
        mask = atom_mask
        token_index = atom_to_token
        if self.structure_prediction:
            query = query.repeat_interleave(multiplicity, 0) + self.noisy_position_projection(noisy_positions)
            conditioning = conditioning.repeat_interleave(multiplicity, 0)
            mask = mask.repeat_interleave(multiplicity, 0)
            token_index = token_index.repeat_interleave(multiplicity, 0)

        atom_query = self.atom_transformer(
            query,
            conditioning,
            atom_pair_bias,
            atom_mask=mask.bool(),
            attention_layout="local",
        )
        token_from_atom = self.token_projection(atom_query)
        token_repr = dense_mean_to_tokens(token_from_atom, token_index, num_tokens, atom_mask=mask, count_eps=1e-6)
        if self.structure_prediction:
            return token_repr, atom_query, conditioning
        return token_repr


class AtomAttentionDecoder(nn.Module):
    r"""
    Broadcast token activations back to atoms and predict a per-atom position update.

    Scatters the token representation to atoms, adds the encoder atom skip states, runs the windowed atom
    transformer, and projects to a 3-vector position update per atom.

    Args:
        atom_hidden_size: Width of the atom hidden channel.
        single_size: Width of the token single representation (the token input is ``2*single_size`` wide).
        num_query_atoms: Number of query atoms per attention window.
        num_key_atoms: Number of key atoms per attention window.
        num_atom_decoder_blocks: Number of atom-transformer blocks.
        num_atom_decoder_attention_heads: Number of attention heads per block.
        operator_backend: Operator backend selector (``"auto"``, ``"oops"``, or ``"public"``).
    """

    def __init__(
        self,
        atom_hidden_size: int = 128,
        single_size: int = 384,
        num_query_atoms: int = 32,
        num_key_atoms: int = 128,
        num_atom_decoder_blocks: int = 3,
        num_atom_decoder_attention_heads: int = 4,
        operator_backend: str = "auto",
    ) -> None:
        super().__init__()
        self.token_projection = nn.Linear(2 * single_size, atom_hidden_size, bias=False)
        self.atom_transformer = PackedAtomAttention(
            atom_hidden_size=atom_hidden_size,
            conditioning_size=atom_hidden_size,
            num_blocks=num_atom_decoder_blocks,
            num_attention_heads=num_atom_decoder_attention_heads,
            num_query_atoms=num_query_atoms,
            num_key_atoms=num_key_atoms,
            operator_backend=operator_backend,
            cross_attention_mode=False,
            transition="triple",
        )
        self.position_update = nn.Sequential(LayerNorm(atom_hidden_size), nn.Linear(atom_hidden_size, 3, bias=False))

    def forward(
        self,
        token_repr: Tensor,
        atom_skip: Tensor,
        conditioning_skip: Tensor,
        atom_pair_bias: Tensor,
        atom_mask: Tensor,
        atom_to_token: Tensor,
        multiplicity: int = 1,
    ) -> Tensor:
        r"""
        Return the per-atom coordinate update.

        Args:
            token_repr: Token representation ``(B*multiplicity, N_token, 2*single_size)``.
            atom_skip: Encoder atom skip states ``(B*multiplicity, N_atom, atom_hidden_size)``.
            conditioning_skip: Encoder conditioning skip states ``(B*multiplicity, N_atom, atom_hidden_size)``.
            atom_pair_bias: Windowed atom-pair attention bias.
            atom_mask: Boolean atom validity mask ``(B, N_atom)``.
            atom_to_token: Token index per atom ``(B, N_atom)``.
            multiplicity: Number of diffusion samples folded into the leading axis.

        Returns:
            Per-atom coordinate update ``(B*multiplicity, N_atom, 3)``.
        """

        token_index = atom_to_token.repeat_interleave(multiplicity, 0)
        token_proj = self.token_projection(token_repr)
        atom_query = atom_skip + token_proj.gather(1, token_index.unsqueeze(-1).expand(-1, -1, token_proj.shape[-1]))
        mask = atom_mask.repeat_interleave(multiplicity, 0)
        atom_query = self.atom_transformer(
            atom_query,
            conditioning_skip,
            atom_pair_bias,
            atom_mask=mask.bool(),
            attention_layout="local",
        )
        return self.position_update(atom_query)


class MiniTriangularUpdate(nn.Module):
    r"""
    Compact bidirectional triangular update used by BoltzGen Miniformer artifacts.

    A single fused module that computes both the outgoing and incoming triangle multiplicative products from one
    gated projection, halving the parameters of the two-module Pairformer triangle-multiplication pair.

    Args:
        pair_size: Width of the pair representation channel.
        operator_backend: Operator backend selector (``"auto"``, ``"oops"``, or ``"public"``).
    """

    def __init__(self, pair_size: int = 128, operator_backend: str = "auto") -> None:
        super().__init__()
        self.operator_backend = operator_backend
        self.norm_in = nn.LayerNorm(pair_size, eps=1e-5)
        self.p_in = nn.Linear(pair_size, pair_size, bias=False)
        self.g_in = nn.Linear(pair_size, pair_size, bias=False)
        self.norm_out = nn.LayerNorm(pair_size // 2)
        self.p_out = nn.Linear(pair_size // 2, pair_size, bias=False)
        self.g_out = nn.Linear(pair_size // 2, pair_size, bias=False)

    def forward(self, pair: Tensor, pair_mask: Tensor | None = None) -> Tensor:
        r"""
        Apply the fused outgoing + incoming triangular multiplicative update.

        The gated input is split into four halves; the outgoing and incoming triangle products are computed in
        fp32 (autocast disabled), concatenated, normalized, and gated back to ``pair_size``.

        Args:
            pair: Pair representation ``(B, N_token, N_token, pair_size)``.
            pair_mask: Optional boolean pair mask ``(B, N_token, N_token)`` zeroing invalid entries.

        Returns:
            Triangular update ``(B, N_token, N_token, pair_size)``.
        """

        normalized = self.norm_in(pair)
        update = self.p_in(normalized) * torch.sigmoid(self.g_in(normalized))
        if pair_mask is not None:
            update = update * pair_mask.unsqueeze(-1).to(dtype=update.dtype, device=update.device)
        with torch.autocast("cuda", enabled=False):
            left_out, right_out, left_in, right_in = torch.chunk(update.float(), 4, dim=-1)
            outgoing = torch.einsum("...ikd,...jkd->...ijd", left_out, right_out)
            incoming = torch.einsum("...kid,...kjd->...ijd", left_in, right_in)
            update = torch.cat((outgoing, incoming), dim=-1).to(pair.dtype)
        update = self.norm_out(update)
        return self.p_out(update) * torch.sigmoid(self.g_out(update))


class MiniTrianglePairUpdateBlock(nn.Module):
    r"""
    Pair-only Miniformer block used in structure-trained-small MSA / template stacks.

    Applies the compact triangular update (row-wise dropout) followed by a pair transition, both residual.

    Args:
        pair_size: Width of the pair representation channel.
        expansion: Feed-forward expansion factor of the pair transition.
        dropout: Row-wise dropout probability on the triangular-update output.
        operator_backend: Operator backend selector (``"auto"``, ``"oops"``, or ``"public"``).
    """

    def __init__(
        self,
        pair_size: int = 128,
        expansion: int = 4,
        dropout: float = 0.25,
        operator_backend: str = "auto",
    ) -> None:
        super().__init__()
        self.triangular = MiniTriangularUpdate(pair_size, operator_backend=operator_backend)
        self.transition_pair = Transition(pair_size, expansion=expansion, operator_backend=operator_backend)
        self.rowwise_dropout = RowwiseDropout(dropout)

    def forward(self, pair: Tensor, pair_mask: Tensor | None = None) -> Tensor:
        r"""
        Apply the triangular update (row-wise dropout) then the pair transition, both residual.

        Args:
            pair: Pair representation ``(B, N_token, N_token, pair_size)``.
            pair_mask: Optional boolean pair mask ``(B, N_token, N_token)``; defaults to all-valid.

        Returns:
            Updated pair representation ``(B, N_token, N_token, pair_size)``.
        """

        if pair_mask is None:
            pair_mask = pair.new_ones(pair.shape[:-1])
        pair = pair + self.rowwise_dropout(self.triangular(pair, pair_mask=pair_mask))
        return pair + self.transition_pair(pair)


class MiniformerBlock(nn.Module):
    r"""
    BoltzGen Miniformer block: compact pair update plus pair-biased single attention.

    Args:
        single_size: Width of the single representation channel.
        pair_size: Width of the pair representation channel.
        num_pair_attention_heads: Number of pair-biased single-attention heads.
        expansion: Feed-forward expansion factor of the transitions.
        dropout: Row-wise dropout probability on the triangular-update output.
        operator_backend: Operator backend selector (``"auto"``, ``"oops"``, or ``"public"``).
    """

    def __init__(
        self,
        single_size: int = 384,
        pair_size: int = 128,
        num_pair_attention_heads: int = 16,
        expansion: int = 4,
        dropout: float = 0.25,
        operator_backend: str = "auto",
    ) -> None:
        super().__init__()
        self.attention = PairBiasedSelfAttention(
            hidden_size=single_size,
            pair_size=pair_size,
            num_attention_heads=num_pair_attention_heads,
            operator_backend=operator_backend,
        )
        self.triangular = MiniTriangularUpdate(pair_size, operator_backend=operator_backend)
        self.transition_single = Transition(single_size, expansion=expansion, operator_backend=operator_backend)
        self.transition_pair = Transition(pair_size, expansion=expansion, operator_backend=operator_backend)
        self.rowwise_dropout = RowwiseDropout(dropout)

    def forward(self, single: Tensor, pair: Tensor, pair_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        r"""
        Update the pair (triangular + transition) then the single (pair-biased attention + transition).

        Args:
            single: Single representation ``(B, N_token, single_size)``.
            pair: Pair representation ``(B, N_token, N_token, pair_size)``.
            pair_mask: Optional boolean pair mask ``(B, N_token, N_token)``; defaults to all-valid.

        Returns:
            Tuple of updated single ``(B, N_token, single_size)`` and pair ``(B, N_token, N_token, pair_size)``.
        """

        if pair_mask is None:
            pair_mask = pair.new_ones(pair.shape[:-1])
        pair = pair + self.rowwise_dropout(self.triangular(pair, pair_mask=pair_mask))
        pair = pair + self.transition_pair(pair)
        single = single + self.attention(single, pair, pair_mask=pair_mask)
        return single + self.transition_single(single), pair


class MiniformerStack(nn.Module):
    r"""
    Stack of BoltzGen Miniformer blocks (the trunk variant selected by ``use_miniformer``).

    Args:
        num_blocks: Number of Miniformer blocks.
        single_size: Width of the single representation channel.
        pair_size: Width of the pair representation channel.
        num_pair_attention_heads: Number of pair-biased single-attention heads per block.
        dropout: Row-wise dropout probability inside each block.
        operator_backend: Operator backend selector (``"auto"``, ``"oops"``, or ``"public"``).
    """

    def __init__(
        self,
        num_blocks: int = 12,
        single_size: int = 384,
        pair_size: int = 128,
        num_pair_attention_heads: int = 16,
        dropout: float = 0.25,
        operator_backend: str = "auto",
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                MiniformerBlock(
                    single_size=single_size,
                    pair_size=pair_size,
                    num_pair_attention_heads=num_pair_attention_heads,
                    dropout=dropout,
                    operator_backend=operator_backend,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, single: Tensor, pair: Tensor, pair_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        r"""
        Run the single and pair representations through every Miniformer block in sequence.

        Args:
            single: Single representation ``(B, N_token, single_size)``.
            pair: Pair representation ``(B, N_token, N_token, pair_size)``.
            pair_mask: Optional boolean pair mask ``(B, N_token, N_token)``.

        Returns:
            Tuple of refined single ``(B, N_token, single_size)`` and pair ``(B, N_token, N_token, pair_size)``.
        """

        for block in self.blocks:
            single, pair = block(single, pair, pair_mask=pair_mask)
        return single, pair


class MiniformerMsaModule(nn.Module):
    r"""
    MSA-module variant whose pair update uses BoltzGen MiniformerNoSeq blocks.

    Args:
        single_size: Width of the single representation channel.
        pair_size: Width of the pair representation channel.
        msa_size: Width of the MSA representation channel.
        msa_feature_size: Width of the raw per-column MSA input features.
        num_blocks: Number of Miniformer MSA layers.
        vocab_size: Number of residue-type tokens (MSA one-hot width).
        use_paired_feature: Whether to concatenate the paired-MSA indicator feature.
        outer_product_hidden_size: Hidden width of the outer-product-mean projection.
        msa_dropout: Row-wise dropout probability on the MSA representation.
        pair_dropout: Row-wise dropout probability inside the pair-only update.
        num_msa_attention_heads: Number of pair-weighted-averaging attention heads.
        msa_head_dim: Per-head dimension of the pair-weighted-averaging attention.
        expansion: Feed-forward expansion factor inside each layer.
        operator_backend: Operator backend selector (``"auto"``, ``"oops"``, or ``"public"``).
    """

    def __init__(
        self,
        single_size: int = 384,
        pair_size: int = 128,
        msa_size: int = 64,
        msa_feature_size: int = 36,
        num_blocks: int = 3,
        vocab_size: int = 33,
        use_paired_feature: bool = True,
        outer_product_hidden_size: int = 32,
        msa_dropout: float = 0.15,
        pair_dropout: float = 0.25,
        num_msa_attention_heads: int = 8,
        msa_head_dim: int = 32,
        expansion: int = 4,
        operator_backend: str = "auto",
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.use_paired_feature = use_paired_feature
        self.token_projection = nn.Linear(single_size, msa_size, bias=False)
        self.msa_projection = nn.Linear(msa_feature_size, msa_size, bias=False)
        self.layers = nn.ModuleList(
            [
                MiniformerMsaLayer(
                    msa_size=msa_size,
                    pair_size=pair_size,
                    outer_product_hidden_size=outer_product_hidden_size,
                    msa_dropout=msa_dropout,
                    pair_dropout=pair_dropout,
                    num_msa_attention_heads=num_msa_attention_heads,
                    msa_head_dim=msa_head_dim,
                    expansion=expansion,
                    operator_backend=operator_backend,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(
        self,
        pair: Tensor,
        single_inputs: Tensor,
        *,
        msa: Tensor,
        has_deletion: Tensor,
        deletion_value: Tensor,
        is_paired: Tensor,
        msa_mask: Tensor,
        token_mask: Tensor,
    ) -> Tensor:
        r"""
        Build the MSA representation from raw MSA features and refine the pair via the Miniformer MSA layers.

        Concatenates the one-hot MSA, deletion, and (optional) paired features, projects to the MSA channel, adds
        the token projection of the input single, then applies each Miniformer MSA layer and returns the pair.

        Args:
            pair: Trunk pair representation ``(B, N_token, N_token, pair_size)``.
            single_inputs: Input single representation ``(B, N_token, single_size)``.
            msa: Integer MSA token ids ``(B, N_msa, N_token)``.
            has_deletion: MSA deletion flags ``(B, N_msa, N_token)``.
            deletion_value: MSA deletion values ``(B, N_msa, N_token)``.
            is_paired: Paired-MSA indicator ``(B, N_msa, N_token)``.
            msa_mask: Boolean MSA validity mask ``(B, N_msa, N_token)``.
            token_mask: Boolean token validity mask ``(B, N_token)``.

        Returns:
            Updated pair representation ``(B, N_token, N_token, pair_size)``.
        """

        features = [
            F.one_hot(msa.long(), num_classes=self.vocab_size).to(pair.dtype),
            has_deletion.unsqueeze(-1).to(pair.dtype),
            deletion_value.unsqueeze(-1).to(pair.dtype),
        ]
        if self.use_paired_feature:
            features.append(is_paired.unsqueeze(-1).to(pair.dtype))
        msa_state = self.msa_projection(torch.cat(features, dim=-1))
        msa_state = msa_state + self.token_projection(single_inputs).unsqueeze(1)
        pair_mask = pair_mask_from_token_mask(token_mask)
        for layer in self.layers:
            pair, msa_state = layer(pair, msa_state, pair_mask, msa_mask)
        return pair


class MiniformerMsaLayer(nn.Module):
    r"""
    One MSA layer for the Miniformer MSA module: pair-weighted averaging, MSA transition, outer-product mean,
    then a compact pair-only Miniformer update.

    Args:
        msa_size: Width of the MSA representation channel.
        pair_size: Width of the pair representation channel.
        outer_product_hidden_size: Hidden width of the outer-product-mean projection.
        msa_dropout: Row-wise dropout probability on the MSA representation.
        pair_dropout: Row-wise dropout probability inside the pair-only update.
        num_msa_attention_heads: Number of pair-weighted-averaging attention heads.
        msa_head_dim: Per-head dimension of the pair-weighted-averaging attention.
        expansion: Feed-forward expansion factor inside the layer.
        operator_backend: Operator backend selector (``"auto"``, ``"oops"``, or ``"public"``).
    """

    def __init__(
        self,
        msa_size: int = 64,
        pair_size: int = 128,
        outer_product_hidden_size: int = 32,
        msa_dropout: float = 0.15,
        pair_dropout: float = 0.25,
        num_msa_attention_heads: int = 8,
        msa_head_dim: int = 32,
        expansion: int = 4,
        operator_backend: str = "auto",
    ) -> None:
        super().__init__()
        self.pair_weighted_averaging = MsaPairWeightedAveraging(
            msa_size=msa_size,
            pair_size=pair_size,
            num_attention_heads=num_msa_attention_heads,
            head_dim=msa_head_dim,
            operator_backend=operator_backend,
        )
        self.msa_transition = Transition(msa_size, expansion=expansion, operator_backend=operator_backend)
        self.outer_product_mean = OuterProductMean(
            msa_size=msa_size,
            hidden_size=outer_product_hidden_size,
            pair_size=pair_size,
            divide_before_projection=True,
            operator_backend=operator_backend,
        )
        self.triangle_pair_update = MiniTrianglePairUpdateBlock(
            pair_size=pair_size,
            expansion=expansion,
            dropout=pair_dropout,
            operator_backend=operator_backend,
        )
        self.msa_dropout = RowwiseDropout(msa_dropout)

    def forward(self, pair: Tensor, msa: Tensor, pair_mask: Tensor, msa_mask: Tensor) -> tuple[Tensor, Tensor]:
        r"""
        Update the MSA (pair-weighted averaging + transition) and the pair (outer-product mean + Miniformer update).

        Args:
            pair: Pair representation ``(B, N_token, N_token, pair_size)``.
            msa: MSA representation ``(B, N_msa, N_token, msa_size)``.
            pair_mask: Boolean pair mask ``(B, N_token, N_token)``.
            msa_mask: Boolean MSA validity mask ``(B, N_msa, N_token)``.

        Returns:
            Tuple of updated pair ``(B, N_token, N_token, pair_size)`` and MSA
            ``(B, N_msa, N_token, msa_size)``.
        """

        msa = msa + self.msa_dropout(self.pair_weighted_averaging(msa, pair, pair_mask=pair_mask))
        msa = msa + self.msa_transition(msa)
        pair = pair + self.outer_product_mean(msa, msa_mask)
        pair = self.triangle_pair_update(pair, pair_mask=pair_mask)
        return pair, msa


CONTACT_CONDITIONING_CLASSES: tuple[str, ...] = (
    "UNSPECIFIED",
    "UNSELECTED",
    "POCKET>BINDER",
    "BINDER>POCKET",
    "CONTACT",
)


class ContactConditioning(nn.Module):
    r"""
    Additive pair bias encoding pocket and contact constraints.

    Encodes the per-pair contact class (pocket / binder direction, contact) and a Fourier-embedded distance
    threshold into an additive pair bias, with dedicated learned biases for the unspecified / unselected classes.

    Args:
        pair_size: Width of the pair representation channel.
        cutoff_min: Lower distance cutoff (angstrom) used to normalize the threshold.
        cutoff_max: Upper distance cutoff (angstrom) used to normalize the threshold.
        num_classes: Number of contact-conditioning classes. Defaults to the built-in class list length.
    """

    def __init__(
        self,
        pair_size: int,
        *,
        cutoff_min: float,
        cutoff_max: float,
        num_classes: int | None = None,
    ) -> None:
        super().__init__()
        num_classes = len(CONTACT_CONDITIONING_CLASSES) if num_classes is None else num_classes
        self.cutoff_min = cutoff_min
        self.cutoff_max = cutoff_max
        self.num_classes = num_classes
        self.threshold_fourier = FrozenFourierEmbedding(pair_size)
        self.constraint_encoder = nn.Linear(pair_size + num_classes - 1, pair_size)
        self.unspecified_bias = nn.Parameter(torch.zeros(pair_size))
        self.unselected_bias = nn.Parameter(torch.zeros(pair_size))

    def forward(self, contact_conditioning: Tensor, contact_threshold: Tensor) -> Tensor:
        r"""
        Encode the contact-conditioning classes and distance thresholds into an additive pairwise bias.

        Args:
            contact_conditioning: Per-pair contact-class one-hot ``(B, N_token, N_token, num_classes)`` whose
                first two channels are the unspecified / unselected classes.
            contact_threshold: Per-pair distance threshold ``(B, N_token, N_token)``.

        Returns:
            Additive pair bias ``(B, N_token, N_token, pair_size)``.
        """

        active = contact_conditioning[..., 2:]
        threshold_normalized = (contact_threshold - self.cutoff_min) / (self.cutoff_max - self.cutoff_min)
        threshold_fourier = self.threshold_fourier(threshold_normalized)

        encoder_input = torch.cat(
            [active, threshold_normalized.unsqueeze(-1), threshold_fourier],
            dim=-1,
        )
        encoded = self.constraint_encoder(encoder_input)

        gate = 1 - contact_conditioning[..., 0:2].sum(dim=-1, keepdim=True)
        return (
            encoded * gate
            + self.unspecified_bias * contact_conditioning[..., 0:1]
            + self.unselected_bias * contact_conditioning[..., 1:2]
        )


class DiffusionConditioning(nn.Module):
    r"""
    Precompute the diffusion conditioning shared across denoiser calls.

    Builds the conditioned token pair and the atom-pair / token-pair states once per structure, then projects them
    into the per-block attention biases consumed by every step of the score model.

    Args:
        single_size: Width of the single representation channel.
        pair_size: Width of the pair representation channel.
        atom_hidden_size: Width of the atom hidden channel.
        atom_pair_hidden_size: Width of the atom-pair hidden channel.
        atom_feature_size: Width of the per-atom reference input features.
        num_query_atoms: Number of query atoms per attention window.
        num_key_atoms: Number of key atoms per attention window.
        num_atom_encoder_blocks: Number of atom-encoder blocks (one bias projection each).
        num_atom_encoder_attention_heads: Number of attention heads per atom-encoder block.
        num_atom_decoder_blocks: Number of atom-decoder blocks (one bias projection each).
        num_atom_decoder_attention_heads: Number of attention heads per atom-decoder block.
        num_token_transformer_blocks: Number of token-transformer blocks (one bias projection each).
        num_token_transformer_heads: Number of attention heads per token-transformer block.
        conditioning_transition_layers: Number of transition layers in the pairwise conditioner.
    """

    def __init__(
        self,
        single_size: int = 384,
        pair_size: int = 128,
        atom_hidden_size: int = 128,
        atom_pair_hidden_size: int = 16,
        atom_feature_size: int = 128,
        num_query_atoms: int = 32,
        num_key_atoms: int = 128,
        num_atom_encoder_blocks: int = 3,
        num_atom_encoder_attention_heads: int = 4,
        num_atom_decoder_blocks: int = 3,
        num_atom_decoder_attention_heads: int = 4,
        num_token_transformer_blocks: int = 24,
        num_token_transformer_heads: int = 16,
        conditioning_transition_layers: int = 2,
    ) -> None:
        super().__init__()
        self.pairwise_conditioner = PairwiseConditioning(
            pair_size=pair_size, relative_position_size=pair_size, num_transitions=conditioning_transition_layers
        )
        self.atom_encoder = AtomEncoder(
            atom_feature_size=atom_feature_size,
            atom_hidden_size=atom_hidden_size,
            atom_pair_hidden_size=atom_pair_hidden_size,
            single_size=single_size,
            pair_size=pair_size,
            num_query_atoms=num_query_atoms,
            num_key_atoms=num_key_atoms,
            structure_prediction=True,
        )
        self.atom_enc_proj_pair = nn.ModuleList(
            nn.Sequential(
                LayerNorm(atom_pair_hidden_size),
                nn.Linear(atom_pair_hidden_size, num_atom_encoder_attention_heads, bias=False),
            )
            for _ in range(num_atom_encoder_blocks)
        )
        self.atom_dec_proj_pair = nn.ModuleList(
            nn.Sequential(
                LayerNorm(atom_pair_hidden_size),
                nn.Linear(atom_pair_hidden_size, num_atom_decoder_attention_heads, bias=False),
            )
            for _ in range(num_atom_decoder_blocks)
        )
        self.token_trans_proj_pair = nn.ModuleList(
            nn.Sequential(LayerNorm(pair_size), nn.Linear(pair_size, num_token_transformer_heads, bias=False))
            for _ in range(num_token_transformer_blocks)
        )

    def forward(
        self,
        single_trunk: Tensor,
        pair_trunk: Tensor,
        relative_position_encoding: Tensor,
        *,
        atom_features: Tensor,
        reference_positions: Tensor,
        atom_mask: Tensor,
        reference_space_id: Tensor,
        atom_to_token: Tensor,
    ) -> dict[str, object]:
        r"""
        Precompute the noise-independent diffusion conditioning shared across all denoiser calls.

        Builds the conditioned token pair, runs the structure-prediction
        [`AtomEncoder`][ofoldx.models.boltzgen.model.AtomEncoder] over the trunk single/pair, and projects the
        atom-pair / token-pair states into the per-block attention biases consumed by the score model.

        Args:
            single_trunk: Trunk single representation ``(B, N_token, single_size)``.
            pair_trunk: Trunk pair representation ``(B, N_token, N_token, pair_size)``.
            relative_position_encoding: Relative-position pair features ``(B, N_token, N_token, pair_size)``.
            atom_features: Per-atom reference features ``(B, N_atom, atom_feature_size)``.
            reference_positions: Reference-conformer atom coords ``(B, N_atom, 3)``.
            atom_mask: Boolean atom validity mask ``(B, N_atom)``.
            reference_space_id: Reference-conformer group id per atom ``(B, N_atom)``.
            atom_to_token: Token index per atom ``(B, N_atom)``.

        Returns:
            Conditioning dict with keys:
                ``"atom_query"``: atom query states ``(B, N_atom, atom_hidden_size)``.
                ``"atom_conditioning"``: atom conditioning states ``(B, N_atom, atom_hidden_size)``.
                ``"atom_enc_bias"``: per-encoder-block atom-pair attention bias windows,
                    last dim ``num_atom_encoder_blocks * num_atom_encoder_attention_heads``.
                ``"atom_dec_bias"``: per-decoder-block atom-pair attention bias windows,
                    last dim ``num_atom_decoder_blocks * num_atom_decoder_attention_heads``.
                ``"token_trans_bias"``: per-token-transformer-block pair attention bias
                    ``(B, N_token, N_token, num_token_transformer_blocks * num_token_transformer_heads)``.
        """

        pair = self.pairwise_conditioner(pair_trunk, relative_position_encoding)
        atom_query, atom_conditioning, atom_pair = self.atom_encoder(
            atom_features,
            reference_positions,
            atom_mask,
            reference_space_id,
            atom_to_token=atom_to_token,
            single_trunk=single_trunk,
            pair_trunk=pair,
        )
        atom_enc_bias = torch.cat([layer(atom_pair) for layer in self.atom_enc_proj_pair], dim=-1)
        atom_dec_bias = torch.cat([layer(atom_pair) for layer in self.atom_dec_proj_pair], dim=-1)
        token_trans_bias = torch.cat([layer(pair) for layer in self.token_trans_proj_pair], dim=-1)
        return {
            "atom_query": atom_query,
            "atom_conditioning": atom_conditioning,
            "atom_enc_bias": atom_enc_bias,
            "atom_dec_bias": atom_dec_bias,
            "token_trans_bias": token_trans_bias,
        }


class PairwiseConditioning(nn.Module):
    r"""
    Condition the trunk pair representation with the relative-position encoding.

    Args:
        pair_size: Width of the pair representation channel.
        relative_position_size: Width of the relative-position encoding concatenated with the pair.
        num_transitions: Number of refinement transition layers.
        transition_expansion: Feed-forward expansion factor of each transition.
    """

    def __init__(
        self,
        pair_size: int = 128,
        relative_position_size: int = 128,
        num_transitions: int = 2,
        transition_expansion: int = 2,
    ) -> None:
        super().__init__()
        self.init_projection = nn.Sequential(
            LayerNorm(pair_size + relative_position_size),
            nn.Linear(pair_size + relative_position_size, pair_size, bias=False),
        )
        self.transitions = nn.ModuleList(
            Transition(pair_size, expansion=transition_expansion) for _ in range(num_transitions)
        )

    def forward(self, pair_trunk: Tensor, relative_position_encoding: Tensor) -> Tensor:
        r"""
        Concatenate trunk pair and relative-position encoding, project to ``pair_size``, then refine via transitions.

        Args:
            pair_trunk: Trunk pair representation ``(B, N_token, N_token, pair_size)``.
            relative_position_encoding: Relative-position pair features
                ``(B, N_token, N_token, relative_position_size)``.

        Returns:
            Conditioned pair representation ``(B, N_token, N_token, pair_size)``.
        """

        return condition_pair(
            pair_trunk,
            relative_position_encoding,
            norm=self.init_projection[0],
            projection=self.init_projection[1],
            transitions=self.transitions,
        )


class DiffusionScoreModel(nn.Module):
    r"""
    EDM score network mapping noisy positions, noise level, trunk, and conditioning to a per-atom position update.

    Args:
        single_size: Width of the single representation channel.
        atom_hidden_size: Width of the atom hidden channel.
        fourier_embedding_size: Width of the noise-level Fourier embedding.
        num_query_atoms: Number of query atoms per attention window.
        num_key_atoms: Number of key atoms per attention window.
        num_atom_encoder_blocks: Number of atom-encoder blocks.
        num_atom_encoder_attention_heads: Number of attention heads per atom-encoder block.
        num_token_transformer_blocks: Number of conditioned token-transformer blocks.
        num_token_transformer_heads: Number of attention heads per token-transformer block.
        num_atom_decoder_blocks: Number of atom-decoder blocks.
        num_atom_decoder_attention_heads: Number of attention heads per atom-decoder block.
        conditioning_transition_layers: Number of transition layers in the single conditioner.
        operator_backend: Operator backend selector (``"auto"``, ``"oops"``, or ``"public"``).
    """

    def __init__(
        self,
        single_size: int = 384,
        atom_hidden_size: int = 128,
        fourier_embedding_size: int = 256,
        num_query_atoms: int = 32,
        num_key_atoms: int = 128,
        num_atom_encoder_blocks: int = 3,
        num_atom_encoder_attention_heads: int = 4,
        num_token_transformer_blocks: int = 24,
        num_token_transformer_heads: int = 16,
        num_atom_decoder_blocks: int = 3,
        num_atom_decoder_attention_heads: int = 4,
        conditioning_transition_layers: int = 2,
        operator_backend: str = "auto",
    ) -> None:
        super().__init__()
        hidden = 2 * single_size
        self.single_conditioner = SingleConditioning(
            single_size=single_size,
            fourier_embedding_size=fourier_embedding_size,
            num_transitions=conditioning_transition_layers,
        )
        self.atom_attention_encoder = AtomAttentionEncoder(
            atom_hidden_size=atom_hidden_size,
            single_size=single_size,
            num_query_atoms=num_query_atoms,
            num_key_atoms=num_key_atoms,
            num_atom_encoder_blocks=num_atom_encoder_blocks,
            num_atom_encoder_attention_heads=num_atom_encoder_attention_heads,
            structure_prediction=True,
            operator_backend=operator_backend,
        )
        self.single_to_token = nn.Sequential(LayerNorm(hidden), nn.Linear(hidden, hidden, bias=False))
        self.token_transformer = ConditionedTransformer(
            num_blocks=num_token_transformer_blocks,
            num_attention_heads=num_token_transformer_heads,
            hidden_size=hidden,
            conditioning_size=hidden,
            operator_backend=operator_backend,
        )
        self.token_norm = LayerNorm(hidden)
        self.atom_attention_decoder = AtomAttentionDecoder(
            atom_hidden_size=atom_hidden_size,
            single_size=single_size,
            num_query_atoms=num_query_atoms,
            num_key_atoms=num_key_atoms,
            num_atom_decoder_blocks=num_atom_decoder_blocks,
            num_atom_decoder_attention_heads=num_atom_decoder_attention_heads,
            operator_backend=operator_backend,
        )

    def forward(
        self,
        single_inputs: Tensor,
        single_trunk: Tensor,
        noisy_positions: Tensor,
        noise_level: Tensor,
        conditioning: dict[str, object],
        *,
        atom_mask: Tensor,
        atom_to_token: Tensor,
        token_mask: Tensor,
        multiplicity: int = 1,
    ) -> Tensor:
        r"""
        Map noisy atom positions and noise level to a per-atom position update (the EDM network output).

        Conditions the single state on the noise level, runs the atom encoder (with noisy positions) to token
        states, refines them through the conditioned token transformer, then decodes back to atoms. The
        precomputed ``conditioning`` biases are ``repeat_interleave``'d by ``multiplicity`` to match the sample axis.

        Args:
            single_inputs: Input single representation ``(B, N_token, single_size)``.
            single_trunk: Trunk single representation ``(B, N_token, single_size)``.
            noisy_positions: Noised atom coords ``(B*multiplicity, N_atom, 3)``.
            noise_level: Per-sample noise level ``(B*multiplicity,)``.
            conditioning: Dict from
                [`DiffusionConditioning`][ofoldx.models.boltzgen.model.DiffusionConditioning] (atom queries,
                conditioning, and the encoder/decoder/token attention biases).
            atom_mask: Boolean atom validity mask ``(B, N_atom)``.
            atom_to_token: Token index per atom ``(B, N_atom)``.
            token_mask: Boolean token validity mask ``(B, N_token)``.
            multiplicity: Number of diffusion samples folded into the leading axis.

        Returns:
            Per-atom position update ``(B*multiplicity, N_atom, 3)``.
        """

        single, _ = self.single_conditioner(
            noise_level,
            single_trunk.repeat_interleave(multiplicity, 0),
            single_inputs.repeat_interleave(multiplicity, 0),
        )

        token_repr, atom_skip, conditioning_skip = self.atom_attention_encoder(
            conditioning["atom_query"],
            conditioning["atom_conditioning"],
            cast(Any, conditioning["atom_enc_bias"]).repeat_interleave(multiplicity, 0),
            atom_mask,
            atom_to_token,
            token_mask.shape[-1],
            noisy_positions=noisy_positions,
            multiplicity=multiplicity,
        )
        token_repr = token_repr + self.single_to_token(single)

        token_repr = self.token_transformer(
            token_repr,
            single,
            cast(Any, conditioning["token_trans_bias"]).repeat_interleave(multiplicity, 0),
            token_mask.repeat_interleave(multiplicity, 0).float(),
        )
        token_repr = self.token_norm(token_repr)

        return self.atom_attention_decoder(
            token_repr,
            atom_skip,
            conditioning_skip,
            cast(Any, conditioning["atom_dec_bias"]).repeat_interleave(multiplicity, 0),
            atom_mask,
            atom_to_token,
            multiplicity=multiplicity,
        )


class SingleConditioning(nn.Module):
    r"""
    Condition the concatenated trunk and input single representation on the diffusion noise level.

    Args:
        single_size: Width of the single representation channel (the conditioned state is ``2*single_size`` wide).
        fourier_embedding_size: Width of the noise-level Fourier embedding.
        num_transitions: Number of refinement transition layers.
        transition_expansion: Feed-forward expansion factor of each transition.
    """

    def __init__(
        self,
        single_size: int = 384,
        fourier_embedding_size: int = 256,
        num_transitions: int = 2,
        transition_expansion: int = 2,
    ) -> None:
        super().__init__()
        hidden = 2 * single_size
        self.single_norm = LayerNorm(hidden)
        self.single_projection = nn.Linear(hidden, hidden)
        self.noise_fourier = FrozenFourierEmbedding(fourier_embedding_size)
        self.fourier_norm = LayerNorm(fourier_embedding_size)
        self.fourier_projection = nn.Linear(fourier_embedding_size, hidden, bias=False)
        self.transitions = nn.ModuleList(
            Transition(hidden, expansion=transition_expansion) for _ in range(num_transitions)
        )

    def forward(self, noise_level: Tensor, single_trunk: Tensor, single_inputs: Tensor) -> tuple[Tensor, Tensor]:
        r"""
        Concatenate trunk and input single states, add the Fourier-embedded noise level, and refine via transitions.

        Args:
            noise_level: Per-sample diffusion noise level ``(B,)``.
            single_trunk: Trunk single representation ``(B, N_token, single_size)``.
            single_inputs: Input single representation ``(B, N_token, single_size)``.

        Returns:
            Tuple of the conditioned single state ``(B, N_token, 2*single_size)`` and the pre-transition single state
            ``(B, N_token, 2*single_size)``.
        """

        return condition_single(
            single_trunk,
            single_inputs,
            noise_level,
            norm=self.single_norm,
            projection=self.single_projection,
            noise_fourier=self.noise_fourier,
            noise_norm=self.fourier_norm,
            noise_projection=self.fourier_projection,
            transitions=self.transitions,
            sample_axis=False,
        )


class AtomDiffusion(nn.Module):
    r"""
    EDM sampler wrapping a [`DiffusionScoreModel`][ofoldx.models.boltzgen.model.DiffusionScoreModel].

    Holds the EDM preconditioner and the Karras noise schedule / stochastic-churn hyperparameters, and drives the
    reverse-diffusion sampling loop (with optional per-step rigid re-alignment of the noisy coords).

    Args:
        score_model: The wrapped EDM score network.
        sigma_data: EDM data-scale hyperparameter.
        min_noise_level: Lowest non-zero schedule noise level.
        max_noise_level: Highest schedule noise level (before ``sigma_data`` scaling).
        rho: Karras schedule ``rho`` exponent.
        churn_rate: Fraction of extra noise injected per step above the churn floor.
        min_churn: Noise-level floor below which no churn noise is injected.
        noise_scale: Multiplier on the injected stochastic-churn noise.
        step_scale: Multiplier on the deterministic step.
        num_sampling_steps: Default number of sampling steps.
        alignment_reverse_diff: Whether to rigidly re-align the noisy coords onto the denoised coords each step.
        sampling_schedule: Noise schedule variant, either ``"af3"`` or ``"dilated"``.
        time_dilation: Dilation factor for the ``"dilated"`` schedule.
        time_dilation_start: Start of the normalized schedule interval to dilate.
        time_dilation_end: End of the normalized schedule interval to dilate.
    """

    def __init__(
        self,
        score_model: DiffusionScoreModel,
        sigma_data: float = 16.0,
        min_noise_level: float = 4e-4,
        max_noise_level: float = 160.0,
        rho: float = 7.0,
        churn_rate: float = 0.8,
        min_churn: float = 1.0,
        noise_scale: float = 0.95,
        step_scale: float = 1.8,
        num_sampling_steps: int = 200,
        alignment_reverse_diff: bool = True,
        sampling_schedule: str = "dilated",
        time_dilation: float = 2.667,
        time_dilation_start: float = 0.6,
        time_dilation_end: float = 0.8,
    ) -> None:
        super().__init__()
        self.score_model = score_model
        self.sigma_data = sigma_data
        self.preconditioner = EdmPreconditioner(sigma_data=sigma_data)
        self.min_noise_level = min_noise_level
        self.max_noise_level = max_noise_level
        self.rho = rho
        self.churn_rate = churn_rate
        self.min_churn = min_churn
        self.noise_scale = noise_scale
        self.step_scale = step_scale
        self.num_sampling_steps = num_sampling_steps
        self.alignment_reverse_diff = alignment_reverse_diff
        self.sampling_schedule = sampling_schedule
        self.time_dilation = time_dilation
        self.time_dilation_start = time_dilation_start
        self.time_dilation_end = time_dilation_end

    def preconditioned_network_forward(
        self, noised_coords: Tensor, sigma: Tensor | float, *, multiplicity: int = 1, **conditioning: object
    ) -> Tensor:
        r"""
        Run one EDM-preconditioned denoising evaluation: noisy coords to denoised coords.

        Args:
            noised_coords: Noised atom coords ``(B*multiplicity, N_atom, 3)``.
            sigma: Noise level, either a scalar or a per-row tensor ``(B*multiplicity,)``.
            multiplicity: Number of diffusion samples folded into the leading axis.
            **conditioning: Keyword conditioning tensors forwarded to the score model.

        Returns:
            Denoised atom coords ``(B*multiplicity, N_atom, 3)``.
        """

        batch = noised_coords.shape[0]
        if not torch.is_tensor(sigma):
            sigma = torch.full((batch,), float(sigma), device=noised_coords.device, dtype=noised_coords.dtype)
        padded = cast(Tensor, sigma).reshape(batch, 1, 1)
        update = self.score_model(
            noisy_positions=self.preconditioner.c_in(padded) * noised_coords,
            noise_level=self.preconditioner.c_noise(sigma),
            multiplicity=multiplicity,
            **conditioning,
        )
        return self.preconditioner.c_skip(padded) * noised_coords + self.preconditioner.c_out(padded) * update

    def sample_schedule_af3(self, num_sampling_steps: int | None = None, *, device: torch.device) -> Tensor:
        r"""
        Build the Karras ``rho``-power noise schedule, scaled by ``sigma_data`` and terminated with a trailing 0.

        Args:
            num_sampling_steps: Number of schedule steps. Defaults to the configured ``num_sampling_steps``.
            device: Device for the returned tensor.

        Returns:
            Descending noise levels ``(num_sampling_steps + 1,)`` from ``max_noise_level*sigma_data`` down to ``0``.
        """

        steps = self.num_sampling_steps if num_sampling_steps is None else num_sampling_steps
        inv_rho = 1 / self.rho
        ramp = torch.arange(steps, device=device, dtype=torch.float32) / max(steps - 1, 1)
        sigmas = (
            self.max_noise_level**inv_rho + ramp * (self.min_noise_level**inv_rho - self.max_noise_level**inv_rho)
        ) ** self.rho
        sigmas = sigmas * self.sigma_data
        return torch.nn.functional.pad(sigmas, (0, 1), value=0.0)

    def sample_schedule_dilated(self, num_sampling_steps: int | None = None, *, device: torch.device) -> Tensor:
        r"""
        Build the BoltzGen dilated Karras schedule used by the released design inference config.

        The schedule first maps uniformly spaced normalized times through the upstream piecewise dilation
        transform, then applies the same Karras ``rho`` curve as the AF3 schedule.
        """

        steps = self.num_sampling_steps if num_sampling_steps is None else num_sampling_steps
        inv_rho = 1 / self.rho
        ramp = torch.arange(steps, device=device, dtype=torch.float32) / max(steps - 1, 1)
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

    def sample_schedule(self, num_sampling_steps: int | None = None, *, device: torch.device) -> Tensor:
        r"""
        Build the configured BoltzGen sampling schedule.

        Returns:
            Descending noise levels ``(num_sampling_steps + 1,)`` with a trailing ``0``.
        """

        if self.sampling_schedule == "af3":
            return self.sample_schedule_af3(num_sampling_steps, device=device)
        if self.sampling_schedule == "dilated":
            return self.sample_schedule_dilated(num_sampling_steps, device=device)
        raise ValueError(f"unknown BoltzGen sampling_schedule: {self.sampling_schedule!r}")

    @torch.no_grad()
    def sample(
        self,
        atom_mask: Tensor,
        *,
        num_sampling_steps: int | None = None,
        multiplicity: int = 1,
        generator: torch.Generator | None = None,
        step_scale: float | None = None,
        noise_scale: float | None = None,
        **conditioning: object,
    ) -> Tensor:
        r"""
        Run reverse diffusion and return sampled atom coords.

        Args:
            atom_mask: Boolean atom validity mask ``(B, N_atom)``.
            num_sampling_steps: Number of denoising steps. Defaults to the configured value.
            multiplicity: Number of samples drawn per batch element (folded into the leading axis).
            generator: Optional PyTorch random generator for the diffusion noise.
            step_scale: Optional per-call deterministic-step scale override.
            noise_scale: Optional per-call stochastic-noise scale override.
            **conditioning: Keyword conditioning tensors forwarded to the score model.

        Returns:
            Sampled atom coords ``(B*multiplicity, N_atom, 3)``.
        """

        device = atom_mask.device
        base_atom_mask = atom_mask
        expanded_atom_mask = atom_mask.repeat_interleave(multiplicity, 0)
        shape = (*expanded_atom_mask.shape, 3)
        sigmas = self.sample_schedule(num_sampling_steps, device=device)
        gammas = torch.where(sigmas > self.min_churn, self.churn_rate, 0.0)
        step_scale_value = self.step_scale if step_scale is None else step_scale
        noise_scale_value = self.noise_scale if noise_scale is None else noise_scale

        def randn() -> Tensor:
            r"""Draw standard Gaussian noise of the expanded coordinate shape ``(B*multiplicity, N_atom, 3)``."""
            return torch.randn(shape, device=device, generator=generator)

        coords = sigmas[0] * randn()
        denoised = None
        for index in range(len(sigmas) - 1):
            sigma_current, sigma_next, gamma = sigmas[index].item(), sigmas[index + 1].item(), gammas[index + 1].item()
            rotation = uniform_random_rotation(
                expanded_atom_mask.shape[0], device=device, dtype=coords.dtype, generator=generator
            )
            translation = torch.randn(expanded_atom_mask.shape[0], 1, 3, device=device, generator=generator)
            coords = torch.einsum("bmd,bds->bms", coords - coords.mean(-2, keepdim=True), rotation) + translation
            if denoised is not None:
                denoised = (
                    torch.einsum("bmd,bds->bms", denoised - denoised.mean(-2, keepdim=True), rotation) + translation
                )

            sigma_hat = sigma_current * (1 + gamma)
            noise_var = noise_scale_value**2 * (sigma_hat**2 - sigma_current**2)
            noisy = coords + noise_scale_value * math.sqrt(max(noise_var, 0.0)) * randn()
            denoised = self.preconditioned_network_forward(
                noisy, sigma_hat, multiplicity=multiplicity, atom_mask=base_atom_mask, **conditioning
            )
            if self.alignment_reverse_diff:
                noisy = weighted_rigid_align(
                    source=noisy.float(), target=denoised.float(), weight=expanded_atom_mask.float()
                ).to(denoised)
            derivative = (noisy - denoised) / sigma_hat
            coords = noisy + step_scale_value * (sigma_next - sigma_hat) * derivative
        return coords


def _dilate_time(ts: Tensor, *, start: float, end: float, dilation: float) -> Tensor:
    r"""
    Piecewise remapping used by BoltzGen's dilated design-inference noise schedule.
    """

    x = end - start
    lower = start
    upper = 1 - end
    if (dilation - 1) * x > lower + upper:
        raise ValueError("time dilation is too large for the requested interval")

    inv_dilation = 1 / dilation
    ratio = (lower + upper + (1 - dilation) * x) / (lower + upper)
    inv_ratio = 1 / ratio
    lower_prime = lower * ratio
    x_prime = x * dilation

    lower_third = ts * inv_ratio
    middle_third = (ts - lower_prime) * inv_dilation + lower
    upper_third = (ts - (lower_prime + x_prime)) * inv_ratio + lower + x
    return torch.where(
        ts < lower_prime,
        lower_third,
        torch.where(ts < lower_prime + x_prime, middle_third, upper_third),
    )


class BFactorHead(nn.Module):
    r"""
    Predict per-token B-factor logits from the trunk single representation.

    Args:
        single_size: Width of the single representation channel.
        num_bins: Number of B-factor output bins.
    """

    def __init__(self, single_size: int = 384, num_bins: int = 64) -> None:
        super().__init__()
        self.projection = nn.Linear(single_size, num_bins)

    def forward(self, single: Tensor) -> Tensor:
        r"""
        Project the single representation to per-token B-factor logits.

        Args:
            single: Trunk single representation of shape `[B, num_tokens, single_size]`.

        Returns:
            Per-token B-factor logits of shape `[B, num_tokens, num_bins]`.
        """

        return self.projection(single)


# Scoring heads (Boltz-2 backbone; reused verbatim by BoltzGen). Opt-in, so the design-only generator keeps its
# base parameter count. Faithful to the OFoldX Boltz-2 confidence/affinity heads (forward-parity-verified
# against boltz2_conf / boltz2_aff), which are in turn the upstream boltzgen ConfidenceModule / AffinityModule.


class ConfidenceHead(nn.Module):
    r"""
    Token-level confidence head (pLDDT / pAE / pDE / resolved) scoring sampled coords against the trunk.

    Args:
        single_size: Width of the single representation channel.
        pair_size: Width of the pair representation channel.
        num_blocks: Number of Pairformer blocks.
        num_pair_attention_heads: Number of pair-biased single-attention heads per block.
        num_distogram_bins: Number of distance-distogram bins built from the sampled coords.
        max_distance: Upper edge (angstrom) of the distogram binning.
        relative_position_size: Width of the relative-position input features.
        num_bond_types: Number of token bond-type ids.
        conditioning_cutoff_min: Lower distance cutoff (angstrom) for contact conditioning.
        conditioning_cutoff_max: Upper distance cutoff (angstrom) for contact conditioning.
        num_plddt_bins: Number of pLDDT output bins.
        num_pae_bins: Number of predicted-aligned-error output bins.
        num_pde_bins: Number of predicted-distance-error output bins.
        operator_backend: Operator backend selector (``"auto"``, ``"oops"``, or ``"public"``).
    """

    def __init__(
        self,
        single_size: int = 384,
        pair_size: int = 128,
        num_blocks: int = 8,
        num_pair_attention_heads: int = 16,
        num_distogram_bins: int = 64,
        max_distance: float = 22.0,
        relative_position_size: int = 139,
        num_bond_types: int = 7,
        conditioning_cutoff_min: float = 4.0,
        conditioning_cutoff_max: float = 20.0,
        num_plddt_bins: int = 50,
        num_pae_bins: int = 64,
        num_pde_bins: int = 64,
        operator_backend: str = "auto",
    ) -> None:
        super().__init__()
        self.register_buffer("boundaries", torch.linspace(2.0, max_distance, num_distogram_bins - 1))
        self.distance_embedding = nn.Embedding(num_distogram_bins, pair_size)

        self.input_single_norm = LayerNorm(single_size)
        self.single_norm = LayerNorm(single_size)
        self.norm_pair = LayerNorm(pair_size)
        self.input_single_projection = nn.Linear(single_size, single_size, bias=False)
        self.single_to_pair_row = nn.Linear(single_size, pair_size, bias=False)
        self.single_to_pair_col = nn.Linear(single_size, pair_size, bias=False)
        self.single_to_pair_product_left = nn.Linear(single_size, pair_size, bias=False)
        self.single_to_pair_product_right = nn.Linear(single_size, pair_size, bias=False)
        self.single_to_pair_product_out = nn.Linear(pair_size, pair_size, bias=False)

        self.relative_position_embedder = RelativePositionEmbedder(relative_position_size, pair_size)
        self.token_bond_projection = nn.Linear(1, pair_size, bias=False)
        self.token_bond_type_embedding = nn.Embedding(num_bond_types, pair_size)
        self.contact_conditioning = ContactConditioning(
            pair_size, cutoff_min=conditioning_cutoff_min, cutoff_max=conditioning_cutoff_max
        )

        self.pairformer_stack = PairformerStack(
            num_blocks=num_blocks,
            single_size=single_size,
            pair_size=pair_size,
            num_pair_attention_heads=num_pair_attention_heads,
            operator_backend=operator_backend,
        )
        self.confidence_heads = ConfidenceHeads(
            single_size, pair_size, num_plddt_bins=num_plddt_bins, num_pae_bins=num_pae_bins, num_pde_bins=num_pde_bins
        )

    def forward(
        self,
        single_inputs: Tensor,
        single: Tensor,
        pair: Tensor,
        pred_coords: Tensor,
        *,
        residue_index: Tensor,
        chain_index: Tensor,
        entity_index: Tensor,
        symmetry_index: Tensor,
        token_index: Tensor,
        token_bonds: Tensor,
        type_bonds: Tensor,
        contact_conditioning: Tensor,
        contact_threshold: Tensor,
        token_to_rep_atom: Tensor,
        token_mask: Tensor,
        cyclic_period: Tensor | None = None,
        multiplicity: int = 1,
    ) -> dict[str, Tensor]:
        r"""
        Rebuild the pair representation (relative-position, bonds, contact, and a binned distogram of
        representative-atom distances), refine with a Pairformer stack, and project to confidence logits.

        Args:
            single_inputs: Trunk input single ``(B, N, single_size)``.
            single: Trunk refined single ``(B, N, single_size)``.
            pair: Trunk pair ``(B, N, N, pair_size)``.
            pred_coords: Sampled atom coords ``(B*multiplicity, M, 3)``.
            residue_index, chain_index, entity_index, symmetry_index, token_index: Relative-position features.
            token_bonds: Token covalent adjacency ``(B, N, N)``.
            type_bonds: Token bond-type ids ``(B, N, N)``.
            contact_conditioning: Contact-class one-hot ``(B, N, N, num_classes)``.
            contact_threshold: Contact thresholds ``(B, N, N)``.
            token_to_rep_atom: Representative atom index per token ``(B, N)``.
            token_mask: Boolean token validity mask ``(B, N)``.
            cyclic_period: Optional cyclic-position periods ``(B, N)``.
            multiplicity: Samples per batch element; trunk tensors are interleaved to ``B*multiplicity``.

        Returns:
            Logit dict from [`ConfidenceHeads`][ofoldx.models.boltzgen.model.ConfidenceHeads]: ``pae_logits``,
            ``pde_logits``, ``plddt_logits``, ``resolved_logits``.
        """

        single_inputs = self.input_single_norm(single_inputs)
        single = self.single_norm(single)
        single = single + self.input_single_projection(single_inputs)
        pair = self.norm_pair(pair)

        relative_features = generate_relative_position_features(
            residue_index=residue_index,
            chain_index=chain_index,
            entity_index=entity_index,
            symmetry_index=symmetry_index,
            token_index=token_index,
            cyclic_period=cyclic_period,
            dtype=single.dtype,
        ).to(device=single.device)
        pair = pair + self.relative_position_embedder(relative_features)
        pair = pair + self.token_bond_projection(token_bonds.unsqueeze(-1).to(pair.dtype))
        pair = pair + self.token_bond_type_embedding(type_bonds.long())
        pair = pair + self.contact_conditioning(contact_conditioning.to(pair.dtype), contact_threshold.to(pair.dtype))

        single = single.repeat_interleave(multiplicity, 0)
        pair = (
            pair
            + self.single_to_pair_row(single_inputs).unsqueeze(-2)
            + self.single_to_pair_col(single_inputs).unsqueeze(-3)
        )
        pair = pair + self.single_to_pair_product_out(
            self.single_to_pair_product_left(single_inputs).unsqueeze(-2)
            * self.single_to_pair_product_right(single_inputs).unsqueeze(-3)
        )
        pair = pair.repeat_interleave(multiplicity, 0)

        rep_atom_index = token_to_rep_atom.repeat_interleave(multiplicity, 0)
        representative = pred_coords.gather(1, rep_atom_index.unsqueeze(-1).expand(-1, -1, pred_coords.shape[-1]))
        distances = torch.cdist(representative, representative)
        distogram = (distances.unsqueeze(-1) > self.boundaries).sum(dim=-1).long()
        pair = pair + self.distance_embedding(distogram)

        token_mask = token_mask.repeat_interleave(multiplicity, 0)
        pair_mask = pair_mask_from_token_mask(token_mask)
        single, pair = self.pairformer_stack(single, pair, pair_mask=pair_mask)

        chain_index = chain_index.repeat_interleave(multiplicity, 0)
        return self.confidence_heads(single, pair, chain_index)


class ConfidenceHeads(nn.Module):
    r"""
    Project the confidence single / pair representations to pLDDT / resolved / pAE / pDE logits.

    Args:
        single_size: Width of the single representation channel.
        pair_size: Width of the pair representation channel.
        num_plddt_bins: Number of pLDDT output bins.
        num_pae_bins: Number of predicted-aligned-error output bins.
        num_pde_bins: Number of predicted-distance-error output bins.
    """

    def __init__(
        self, single_size: int, pair_size: int, num_plddt_bins: int = 50, num_pae_bins: int = 64, num_pde_bins: int = 64
    ) -> None:
        super().__init__()
        self.pae_intra_projection = nn.Linear(pair_size, num_pae_bins, bias=False)
        self.pae_inter_projection = nn.Linear(pair_size, num_pae_bins, bias=False)
        self.pde_intra_projection = nn.Linear(pair_size, num_pde_bins, bias=False)
        self.pde_inter_projection = nn.Linear(pair_size, num_pde_bins, bias=False)
        self.plddt_projection = nn.Linear(single_size, num_plddt_bins, bias=False)
        self.resolved_projection = nn.Linear(single_size, 2, bias=False)

    def forward(self, single: Tensor, pair: Tensor, chain_index: Tensor) -> dict[str, Tensor]:
        r"""
        Project the confidence single/pair representations to error logits, selecting intra- vs inter-chain
        projections per token pair via ``chain_index``; pDE uses the symmetrized pair.

        Args:
            single: Confidence single representation of shape `[B, num_tokens, single_size]`.
            pair: Confidence pair representation of shape `[B, num_tokens, num_tokens, pair_size]`.
            chain_index: Per-token chain id of shape `[B, num_tokens]`, used to gate the intra-/inter-chain heads.

        Returns:
            A mapping with ``pae_logits`` `[B, num_tokens, num_tokens, num_pae_bins]`, ``pde_logits``
            `[B, num_tokens, num_tokens, num_pde_bins]`, ``plddt_logits`` `[B, num_tokens, num_plddt_bins]`, and
            ``resolved_logits`` `[B, num_tokens, 2]`.
        """

        same_chain = (chain_index.unsqueeze(-1) == chain_index.unsqueeze(-2)).float().unsqueeze(-1)
        different_chain = 1.0 - same_chain
        pae_logits = self.pae_intra_projection(pair) * same_chain + self.pae_inter_projection(pair) * different_chain
        symmetric = pair + pair.transpose(-3, -2)
        pde_logits = (
            self.pde_intra_projection(symmetric) * same_chain + self.pde_inter_projection(symmetric) * different_chain
        )
        return {
            "pae_logits": pae_logits,
            "pde_logits": pde_logits,
            "plddt_logits": self.plddt_projection(single),
            "resolved_logits": self.resolved_projection(single),
        }


class AffinityHead(nn.Module):
    r"""
    Pair-only affinity head: distogram-condition the trunk pair over the ligand/receptor sub-graph, refine with a
    Pairformer stack, then pool to scalar affinity value and binary-binder logits (Boltz-2 / boltzgen affinity).

    Args:
        single_size: Width of the single representation channel.
        pair_size: Width of the pair representation channel.
        num_blocks: Number of pair-only Pairformer blocks.
        num_pair_attention_heads: Number of pair-biased attention heads per block.
        num_distogram_bins: Number of distance-distogram bins built from the sampled coords.
        max_distance: Upper edge (angstrom) of the distogram binning.
        num_conditioning_transitions: Number of transition layers in the pairwise conditioner.
        expansion: Feed-forward / conditioning-transition expansion factor.
        triangle_attention_hidden_size: Per-head width of the triangle attention.
        num_triangle_attention_heads: Number of triangle-attention heads.
        dropout: Dropout probability inside the Pairformer.
        operator_backend: Operator backend selector (``"auto"``, ``"oops"``, or ``"public"``).
    """

    def __init__(
        self,
        single_size: int = 384,
        pair_size: int = 128,
        num_blocks: int = 4,
        num_pair_attention_heads: int = 16,
        num_distogram_bins: int = 64,
        max_distance: float = 22.0,
        num_conditioning_transitions: int = 2,
        expansion: int = 2,
        triangle_attention_hidden_size: int = 32,
        num_triangle_attention_heads: int = 4,
        dropout: float = 0.0,
        operator_backend: str = "auto",
    ) -> None:
        super().__init__()
        self.register_buffer("boundaries", torch.linspace(2.0, max_distance, num_distogram_bins - 1))
        self.dist_bin_pairwise_embed = nn.Embedding(num_distogram_bins, pair_size)
        self.single_to_pair_product_left = nn.Linear(single_size, pair_size, bias=False)
        self.single_to_pair_product_right = nn.Linear(single_size, pair_size, bias=False)
        self.pair_norm = LayerNorm(pair_size)
        self.pair_linear = nn.Linear(pair_size, pair_size, bias=False)
        self.pairwise_conditioner = PairwiseConditioning(
            pair_size=pair_size,
            relative_position_size=pair_size,
            num_transitions=num_conditioning_transitions,
            transition_expansion=expansion,
        )
        self.pairformer_stack = PairformerStack(
            num_blocks=num_blocks,
            single_size=0,
            pair_size=pair_size,
            triangle_attention_hidden_size=triangle_attention_hidden_size,
            num_triangle_attention_heads=num_triangle_attention_heads,
            num_pair_attention_heads=num_pair_attention_heads,
            dropout=dropout,
            operator_backend=operator_backend,
        )
        self.affinity_heads = AffinityHeads(pair_size=pair_size, single_size=single_size)

    def forward(
        self,
        *,
        single_inputs: Tensor,
        pair: Tensor,
        pred_coords: Tensor,
        token_to_rep_atom: Tensor,
        token_mask: Tensor,
        ligand_mask: Tensor,
        receptor_mask: Tensor,
        multiplicity: int = 1,
    ) -> dict[str, Tensor]:
        r"""
        Score affinity from trunk pair features and predicted atom coords over the ligand/receptor sub-graph.

        Distogram-conditions the trunk pair on the representative-atom distances, restricts attention to the
        receptor/ligand and ligand/ligand pair mask, refines with the pair-only Pairformer, and pools to scalars.

        Args:
            single_inputs: Trunk input single ``(B, N_token, single_size)``.
            pair: Trunk pair ``(B, N_token, N_token, pair_size)``.
            pred_coords: Sampled atom coords ``(B*multiplicity, N_atom, 3)``.
            token_to_rep_atom: Representative atom index per token ``(B, N_token)``.
            token_mask: Boolean token validity mask ``(B, N_token)``.
            ligand_mask: Boolean ligand-token mask ``(B, N_token)``.
            receptor_mask: Boolean receptor-token mask ``(B, N_token)``.
            multiplicity: Samples per batch element; trunk tensors are interleaved to ``B*multiplicity``.

        Returns:
            Dict with ``affinity_pred_value`` ``(B*multiplicity, 1)`` and ``affinity_logits_binary``
            ``(B*multiplicity, 1)``.
        """

        pair = self.pair_linear(self.pair_norm(pair))
        pair = pair + self.single_to_pair_product_left(single_inputs).unsqueeze(-2)
        pair = pair + self.single_to_pair_product_right(single_inputs).unsqueeze(-3)
        pair = pair.repeat_interleave(multiplicity, 0)

        rep_atom_index = token_to_rep_atom.repeat_interleave(multiplicity, 0).long()
        representative = pred_coords.gather(
            1,
            rep_atom_index.unsqueeze(-1).expand(-1, -1, pred_coords.shape[-1]),
        )
        distances = torch.cdist(representative, representative)
        distogram = (distances.unsqueeze(-1) > self.boundaries).sum(dim=-1).long()
        pair = pair + self.pairwise_conditioner(pair, self.dist_bin_pairwise_embed(distogram))

        token_mask = token_mask.repeat_interleave(multiplicity, 0).bool()
        ligand_mask = ligand_mask.repeat_interleave(multiplicity, 0).bool() & token_mask
        receptor_mask = receptor_mask.repeat_interleave(multiplicity, 0).bool() & token_mask
        affinity_pair_mask = _affinity_pair_mask(ligand_mask, receptor_mask)
        _, pair = self.pairformer_stack(None, pair, pair_mask=affinity_pair_mask)
        return self.affinity_heads(pair, affinity_pair_mask)


class AffinityHeads(nn.Module):
    r"""
    Project pooled affinity pair features to an affinity value and binary-binder logits.

    Args:
        pair_size: Width of the pair representation channel.
        single_size: Width of the intermediate MLP channel.
    """

    def __init__(self, pair_size: int = 128, single_size: int = 384) -> None:
        super().__init__()
        self.affinity_out_mlp = nn.Sequential(
            nn.Linear(pair_size, pair_size),
            nn.ReLU(),
            nn.Linear(pair_size, single_size),
            nn.ReLU(),
        )
        self.to_affinity_pred_value = nn.Sequential(
            nn.Linear(single_size, single_size),
            nn.ReLU(),
            nn.Linear(single_size, single_size),
            nn.ReLU(),
            nn.Linear(single_size, 1),
        )
        self.to_affinity_pred_score = nn.Sequential(
            nn.Linear(single_size, single_size),
            nn.ReLU(),
            nn.Linear(single_size, single_size),
            nn.ReLU(),
            nn.Linear(single_size, 1),
        )
        self.to_affinity_logits_binary = nn.Linear(1, 1)

    def forward(self, pair: Tensor, affinity_pair_mask: Tensor) -> dict[str, Tensor]:
        r"""
        Mean-pool the receptor/ligand and ligand/ligand pair states (excluding the diagonal) and project to scalars.

        Args:
            pair: Refined affinity pair states ``(B*multiplicity, N_token, N_token, pair_size)``.
            affinity_pair_mask: Boolean receptor/ligand + ligand/ligand pair mask ``(B*multiplicity, N, N)``.

        Returns:
            Dict with ``affinity_pred_value`` ``(B*multiplicity, 1)`` and ``affinity_logits_binary``
            ``(B*multiplicity, 1)``.
        """

        num_tokens = affinity_pair_mask.shape[-1]
        non_diagonal = ~torch.eye(num_tokens, device=affinity_pair_mask.device, dtype=torch.bool).unsqueeze(0)
        pool_mask = (affinity_pair_mask.bool() & non_diagonal).to(dtype=pair.dtype).unsqueeze(-1)
        pooled = torch.sum(pair * pool_mask, dim=(-3, -2)) / torch.sum(pool_mask, dim=(-3, -2)).clamp_min(1e-7)
        pooled = self.affinity_out_mlp(pooled)
        affinity_pred_value = self.to_affinity_pred_value(pooled).reshape(-1, 1)
        affinity_pred_score = self.to_affinity_pred_score(pooled).reshape(-1, 1)
        affinity_logits_binary = self.to_affinity_logits_binary(affinity_pred_score).reshape(-1, 1)
        return {
            "affinity_pred_value": affinity_pred_value,
            "affinity_logits_binary": affinity_logits_binary,
        }


def _affinity_pair_mask(ligand_mask: Tensor, receptor_mask: Tensor) -> Tensor:
    r"""
    Build the boolean pair mask over receptor/ligand and ligand/ligand token pairs used by the affinity head.

    Args:
        ligand_mask: Boolean ligand-token mask ``(..., N_token)``.
        receptor_mask: Boolean receptor-token mask ``(..., N_token)``.

    Returns:
        Boolean pair mask ``(..., N_token, N_token)`` true on receptor/ligand and ligand/ligand pairs.
    """

    return (
        ligand_mask.unsqueeze(-1) & receptor_mask.unsqueeze(-2)
        | receptor_mask.unsqueeze(-1) & ligand_mask.unsqueeze(-2)
        | ligand_mask.unsqueeze(-1) & ligand_mask.unsqueeze(-2)
    )


def _resolve_affinity_token_mask(batch: BoltzGenInput, affinity_token_mask: Tensor | None = None) -> Tensor:
    r"""
    Resolve the ligand-token mask for affinity scoring, defaulting to ``molecule_type == 3`` (NONPOLYMER) tokens.
    """

    if affinity_token_mask is None:
        affinity_token_mask = batch.affinity_token_mask
    if affinity_token_mask is None:
        if batch.molecule_type is None:
            raise ValueError("affinity_token_mask is required when molecule_type is unavailable")
        affinity_token_mask = batch.molecule_type == 3
    affinity_token_mask = affinity_token_mask.to(device=batch.token_mask.device, dtype=torch.bool)
    if affinity_token_mask.shape != batch.token_mask.shape:
        raise ValueError(
            f"affinity_token_mask shape {tuple(affinity_token_mask.shape)} must match token_mask "
            f"{tuple(batch.token_mask.shape)}"
        )
    if not bool((affinity_token_mask & batch.token_mask.bool()).any()):
        raise ValueError("affinity_token_mask selects no valid tokens")
    return affinity_token_mask


def _design_conditioning_for_embedder(
    batch: BoltzGenInput, embedder: AtomConditionedInputEmbedder
) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
    r"""
    Resolve and validate the design-conditioning tensors required by the input embedder's enabled channels.

    Args:
        batch: Typed model input.
        embedder: The atom-conditioned input embedder (its enabled conditioning channels decide what is required).

    Returns:
        Tuple of ``(binding_type, design_mask, ss_type)`` (each may be ``None`` when the channel is disabled).

    Raises:
        ValueError: If a conditioning tensor required by an enabled channel is missing.
    """

    binding_type, design_mask, ss_type = batch.binding_type, batch.design_mask, batch.ss_type
    if hasattr(embedder, "binding_specification_embedding") and binding_type is None:
        raise ValueError("binding_type is required for BoltzGen structure design")
    if hasattr(embedder, "design_mask_embedding") and design_mask is None:
        raise ValueError("design_mask is required for BoltzGen structure design")
    if hasattr(embedder, "ss_specification_embedding") and ss_type is None:
        raise ValueError("ss_type is required for BoltzGen structure design")
    return binding_type, design_mask, ss_type


def _require_template_features(
    batch: BoltzGenInput,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    r"""
    Return the nine template feature tensors required by the template module, raising if any is missing.

    Args:
        batch: Typed model input.

    Returns:
        Tuple of ``(template_restype, template_mask_frame, template_frame_rot, template_frame_t, template_cb,
        template_ca, template_mask_cb, template_mask, visibility_ids)``.

    Raises:
        ValueError: If any required template feature is missing.
    """

    if (
        batch.template_restype is None
        or batch.template_mask_frame is None
        or batch.template_frame_rot is None
        or batch.template_frame_t is None
        or batch.template_cb is None
        or batch.template_ca is None
        or batch.template_mask_cb is None
        or batch.template_mask is None
        or batch.visibility_ids is None
    ):
        raise ValueError("template features are incomplete")
    return (
        batch.template_restype,
        batch.template_mask_frame,
        batch.template_frame_rot,
        batch.template_frame_t,
        batch.template_cb,
        batch.template_ca,
        batch.template_mask_cb,
        batch.template_mask,
        batch.visibility_ids,
    )


def _token_bonds_with_channel(batch: BoltzGenInput) -> Tensor:
    r"""
    Return ``batch.token_bonds`` with a trailing singleton channel, adding one when it is rank-3.

    Args:
        batch: Typed model input.

    Returns:
        Token bonds ``(B, N_token, N_token, 1)``.
    """

    if batch.token_bonds.ndim == 3:
        return batch.token_bonds.unsqueeze(-1)
    return batch.token_bonds


def _gather_backbone4(batch: BoltzGenInput, batch_size: int, num_tokens: int) -> Tensor:
    r"""Gather the four backbone-atom coordinates per token -> ``(B, N_token, 4, 3)``."""
    bb4_index = batch.token_to_bb4_atoms.reshape(batch_size, num_tokens * 4)
    coords = batch.coords.reshape(batch_size, -1, 3)
    backbone = coords.gather(1, bb4_index.unsqueeze(-1).expand(-1, -1, 3))
    return backbone.reshape(batch_size, num_tokens, 4, 3)


def _init_boltzgen_final_weights(module: nn.Module) -> None:
    r"""
    Apply the BoltzGen zero-init scheme to every final-layer / gating / conditioning weight under ``module``.

    Zeroing the last projection of each residual sub-block (and biasing the gate projections negative) starts every
    block as a near-identity, matching the upstream BoltzGen initialization.

    Args:
        module: Root module to walk; every matching child is zero-initialized in place.
    """

    for child in module.modules():
        if isinstance(child, AtomConditionedInputEmbedder):
            child.reset_parameters()
        if isinstance(child, AtomEncoder):
            if hasattr(child, "single_trunk_projection"):
                nn.init.zeros_(child.single_trunk_projection[1].weight)
            if hasattr(child, "pair_trunk_projection"):
                nn.init.zeros_(child.pair_trunk_projection[1].weight)
            nn.init.zeros_(child.left_atom_projection[1].weight)
            nn.init.zeros_(child.right_atom_projection[1].weight)
            nn.init.zeros_(child.feed_forward[5].weight)
        elif isinstance(child, AtomAttentionEncoder) and hasattr(child, "noisy_position_projection"):
            nn.init.zeros_(child.noisy_position_projection.weight)
        elif isinstance(child, AtomAttentionDecoder):
            nn.init.zeros_(child.token_projection.weight)
            nn.init.zeros_(child.position_update[1].weight)
        elif isinstance(child, DiffusionScoreModel):
            nn.init.zeros_(child.single_to_token[1].weight)
        elif isinstance(child, MiniTriangularUpdate):
            nn.init.zeros_(child.g_in.weight)
            nn.init.zeros_(child.p_out.weight)
            nn.init.zeros_(child.g_out.weight)
        elif isinstance(child, Transition):
            nn.init.zeros_(child.output_projection.weight)

        projection = getattr(getattr(child, "attention", None), "projection", None)
        if projection is not None and hasattr(projection, "output_projection"):
            nn.init.zeros_(projection.output_projection.weight)
        if hasattr(child, "output_gate_projection"):
            nn.init.zeros_(child.output_gate_projection.weight)
            if child.output_gate_projection.bias is not None:
                nn.init.constant_(child.output_gate_projection.bias, -2.0)
        if hasattr(child, "conditioning_gate_projection"):
            nn.init.zeros_(child.conditioning_gate_projection.weight)
            if child.conditioning_gate_projection.bias is not None:
                nn.init.constant_(child.conditioning_gate_projection.bias, -2.0)


NUM_BOND_TYPES = 6

# const.chain_type_ids = {PROTEIN, DNA, RNA, NONPOLYMER}.
NUM_CHAIN_TYPES = 4

NUM_RESIDUE_TYPES = 33  # const.num_tokens

CANONICALS_OFFSET = 2  # const.canonicals_offset: the 20 canonical amino acids occupy ids [2, 22)

NUM_CANONICAL_TOKENS = 20  # len(const.canonical_tokens)

# The 20 canonical amino acids (three-letter CCD codes) in vocabulary order (residue-token ids ``[2, 22)``);
# matches ``const.canonical_tokens``. Indexed by ``inverse_fold_restriction`` and ``aa_constraint_mask``.
CANONICAL_TOKENS: tuple[str, ...] = (
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
)


def build_constraint_logit_mask(
    num_nodes: int,
    aa_constraint_mask: Tensor | None,
    inverse_fold_restriction: list[str],
    inf: float,
    device: torch.device,
) -> Tensor:
    r"""
    Build the additive per-position canonical-amino-acid logit mask for constrained inverse folding.

    Combines a per-residue block mask (``aa_constraint_mask``, ``(num_nodes, 20)``, ``> 0`` = blocked) with a
    global block list (``inverse_fold_restriction``, three-letter CCD codes blocked everywhere). Positions left
    with no allowed amino acid have their per-residue constraints relaxed; if a position is still fully blocked by
    the global list alone, a ``ValueError`` is raised. Faithful to upstream ``build_constraint_logit_mask``.

    Args:
        num_nodes: Number of packed valid tokens.
        aa_constraint_mask: Optional per-residue block mask ``(num_nodes, 20)``; ``None`` = no per-residue blocks.
        inverse_fold_restriction: Canonical amino acids (three-letter codes) to block at every position.
        inf: Positive magnitude of the additive penalty applied to a blocked amino acid.
        device: Device for the returned mask.

    Returns:
        Additive logit bias ``(num_nodes, 20)``: ``0.0`` where allowed, ``-inf`` where blocked.

    Raises:
        ValueError: If a position is fully blocked by the global restriction alone.
    """

    num_aa = NUM_CANONICAL_TOKENS
    has_per_residue = False
    if aa_constraint_mask is None:
        per_residue_blocked = torch.zeros(num_nodes, num_aa, dtype=torch.bool, device=device)
    elif aa_constraint_mask.shape != (num_nodes, num_aa):
        warnings.warn(
            f"aa_constraint_mask shape {tuple(aa_constraint_mask.shape)} != {(num_nodes, num_aa)}; "
            "ignoring per-residue constraints.",
            RuntimeWarning,
            stacklevel=2,
        )
        per_residue_blocked = torch.zeros(num_nodes, num_aa, dtype=torch.bool, device=device)
    else:
        has_per_residue = True
        per_residue_blocked = aa_constraint_mask.to(device=device) > 0

    global_blocked = torch.zeros(num_aa, dtype=torch.bool, device=device)
    for res_type in inverse_fold_restriction:
        global_blocked[CANONICAL_TOKENS.index(res_type)] = True

    combined = per_residue_blocked | global_blocked.unsqueeze(0)
    all_blocked = combined.all(dim=1)
    if all_blocked.any() and has_per_residue:
        warnings.warn(
            f"Positions {torch.where(all_blocked)[0].tolist()} have all amino acids blocked; relaxing "
            "per-residue constraints there.",
            RuntimeWarning,
            stacklevel=2,
        )
        per_residue_blocked = per_residue_blocked.clone()
        per_residue_blocked[all_blocked] = False
        combined = per_residue_blocked | global_blocked.unsqueeze(0)

    if combined.all(dim=1).any():
        raise ValueError(
            f"Inverse folding has no valid amino acids at positions {torch.where(combined.all(dim=1))[0].tolist()} "
            "after applying the global restriction; reduce inverse_fold_restriction."
        )
    return combined.to(dtype=torch.float32) * (-inf)


class InverseFoldingEncoder(nn.Module):
    r"""
    kNN-graph encoder over tokens (BoltzGen inverse folding).

    Builds a per-structure kNN token graph from the representative-atom coordinates, featurizes each edge (backbone
    distance RBF plus relative-position / bond attributes), and refines node and edge states with graph-attention
    layers.

    Args:
        single_size: Width of the input-embedder token single representation.
        node_size: Width of the GNN node channel.
        edge_size: Width of the GNN edge channel.
        hidden_size: Width of the graph-attention hidden channel.
        num_layers: Number of graph-attention layers.
        num_attention_heads: Number of attention heads per layer.
        k_neighbors: Number of nearest neighbours per token.
        max_relative_position: Clamp range for the relative-position edge feature.
        dropout: Dropout probability inside the graph-attention layers.
        enable_input_embedder: Whether node features come from the token single embedding (else raw node features).
        node_feature_size: Width of the raw node features when the input embedder is disabled.
    """

    def __init__(
        self,
        single_size: int = 384,
        node_size: int = 128,
        edge_size: int = 128,
        hidden_size: int = 128,
        num_layers: int = 3,
        num_attention_heads: int = 4,
        k_neighbors: int = 30,
        max_relative_position: int = 32,
        dropout: float = 0.0,
        enable_input_embedder: bool = True,
        node_feature_size: int = 135,
    ) -> None:
        super().__init__()
        self.k_neighbors = k_neighbors
        self.max_relative_position = max_relative_position
        self.enable_input_embedder = enable_input_embedder
        # geo (4x4 backbone-atom distances x 16 RBF = 256)
        # + edge attrs (2*max_relative_position+2 + same_chain + same_residue)
        # + bond (token_bonds + one-hot bond type).
        self.edge_input_size = 256 + (2 * max_relative_position + 2) + 2 + 1 + (NUM_BOND_TYPES + 1)

        self.node_input_projection = nn.Linear(single_size if enable_input_embedder else node_feature_size, node_size)
        self.edge_input_projection = nn.Linear(self.edge_input_size, edge_size)
        self.layers = nn.ModuleList(
            GraphAttentionLayer(
                node_size=node_size,
                edge_size=edge_size,
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                dropout=dropout,
            )
            for _ in range(num_layers)
        )
        self.distance_smearing = GaussianSmearing(start=0.0, stop=20.0, num_gaussians=16)

    @torch.no_grad()
    def build_knn_graph(self, batch: BoltzGenInput) -> tuple[Tensor, Tensor, Tensor]:
        r"""
        Build the kNN token graph from representative-atom distances.

        Args:
            batch: Typed model input (supplies ``token_mask``, ``center_coords``, and token bonds).

        Returns:
            Tuple of ``edge_index`` ``(2, E)`` into packed valid tokens, ``valid_mask`` ``(B, N_token)``, and
            per-edge bond features ``(E, 1 + NUM_BOND_TYPES + 1)``.
        """

        valid_mask = batch.token_mask.bool()
        batch_size, num_tokens = valid_mask.shape
        valid_pair = valid_mask.unsqueeze(1) & valid_mask.unsqueeze(2)
        packed_index = (valid_mask.flatten().cumsum(dim=0) - 1).view(batch_size, num_tokens)
        k_neighbors = min(self.k_neighbors, num_tokens)

        distances = torch.cdist(batch.center_coords, batch.center_coords)
        distances = distances.masked_fill(~valid_pair, float("inf"))
        src = torch.topk(distances, k_neighbors, largest=False).indices  # (B, N, k_neighbors)
        dst = torch.arange(num_tokens, device=src.device)[None, :, None].expand(batch_size, -1, k_neighbors)

        # token_bonds carries a trailing singleton (B, N, N, 1); drop it so the gather index (B, N, k_neighbors)
        # matches its rank (the BoltzGen reference does the same, asserting shape[-1] == 1). type_bonds is (B, N, N).
        token_bonds = torch.gather(_token_bonds_with_channel(batch)[..., 0], 2, src)
        type_bonds = torch.gather(batch.type_bonds, 2, src)

        src, dst = src.flatten(start_dim=1), dst.flatten(start_dim=1)
        token_bonds, type_bonds = token_bonds.flatten(), type_bonds.flatten()
        edge_valid = torch.gather(valid_mask, 1, src).flatten() & torch.gather(valid_mask, 1, dst).flatten()

        src = torch.gather(packed_index, 1, src).flatten()[edge_valid]
        dst = torch.gather(packed_index, 1, dst).flatten()[edge_valid]
        edge_index = torch.stack([src, dst], dim=0)

        token_bonds, type_bonds = token_bonds[edge_valid], type_bonds[edge_valid]
        type_bonds = F.one_hot(type_bonds, num_classes=NUM_BOND_TYPES + 1)
        bond = torch.cat([token_bonds[..., None], type_bonds], dim=-1)
        return edge_index, valid_mask, bond

    @torch.no_grad()
    def edge_geometry(self, batch: BoltzGenInput, edge_index: Tensor, valid_mask: Tensor) -> Tensor:
        r"""
        Compute the per-edge backbone-atom (4x4) pairwise-distance RBF feature.

        Args:
            batch: Typed model input (supplies backbone-atom coordinates).
            edge_index: Edge endpoints ``(2, E)`` into packed valid tokens.
            valid_mask: Boolean token validity mask ``(B, N_token)``.

        Returns:
            Per-edge backbone-distance RBF feature ``(E, 256)``.
        """

        src, dst = edge_index[0], edge_index[1]
        batch_size, num_tokens = valid_mask.shape
        backbone = _gather_backbone4(batch, batch_size, num_tokens)[valid_mask]  # (Nvalid, 4, 3)
        distance = torch.norm(backbone[src, None] - backbone[dst, :, None], dim=-1).flatten(start_dim=1)
        return self.distance_smearing(distance)

    @torch.no_grad()
    def edge_attributes(self, batch: BoltzGenInput, edge_index: Tensor, valid_mask: Tensor) -> Tensor:
        r"""
        Compute the per-edge relative-position / same-chain / same-residue attributes.

        Args:
            batch: Typed model input (supplies ``chain_index`` and ``residue_index``).
            edge_index: Edge endpoints ``(2, E)`` into packed valid tokens.
            valid_mask: Boolean token validity mask ``(B, N_token)``.

        Returns:
            Per-edge attribute feature ``(E, 2*max_relative_position + 2 + 2)`` (relative-position one-hot plus
            same-chain and same-residue flags).
        """

        src, dst = edge_index[0], edge_index[1]
        asym = batch.chain_index[valid_mask]
        same_chain = asym[src] == asym[dst]
        residue = batch.residue_index[valid_mask]
        same_residue = residue[src] == residue[dst]
        delta = torch.clip(residue[dst] - residue[src] + self.max_relative_position, 0, 2 * self.max_relative_position)
        delta = torch.where(same_chain, delta, torch.zeros_like(delta) + 2 * self.max_relative_position + 1)
        relative = F.one_hot(delta, 2 * self.max_relative_position + 2)
        return torch.cat([relative.float(), same_chain[..., None].float(), same_residue[..., None].float()], dim=-1)

    def forward(
        self,
        batch: BoltzGenInput,
        *,
        single_inputs: Tensor | None = None,
        node_features: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        r"""
        Encode the structure into a packed kNN token graph with attention-refined node and edge states.

        Builds the kNN graph, featurizes edges (backbone-distance RBF + relative-position/bond attributes),
        projects node features (from ``single_inputs`` when the input embedder is enabled, else ``node_features``),
        and runs the graph-attention layers.

        Args:
            batch: Typed model input.
            single_inputs: Token single embedding ``(B, N_token, single_size)``; required when
                ``enable_input_embedder`` is True.
            node_features: Precomputed node features ``(B, N_token, node_feature_size)``; required when
                ``enable_input_embedder`` is False.

        Returns:
            Tuple of ``edge_index`` ``(2, E)`` into packed valid tokens, ``valid_mask`` ``(B, N_token)``, packed
            node states ``(N_valid, node_size)``, and packed edge states ``(E, edge_size)``.

        Raises:
            ValueError: If the required node-feature source for the configured mode is missing.
        """

        edge_index, valid_mask, bond = self.build_knn_graph(batch)
        geometry = self.edge_geometry(batch, edge_index, valid_mask)
        edge_attr = self.edge_attributes(batch, edge_index, valid_mask)

        if self.enable_input_embedder:
            if single_inputs is None:
                raise ValueError("single_inputs is required when enable_input_embedder=True")
            node_attr = single_inputs[valid_mask]
        else:
            if node_features is None:
                raise ValueError("node_features is required when enable_input_embedder=False")
            node_attr = node_features[valid_mask]

        node = self.node_input_projection(node_attr)
        edge = self.edge_input_projection(torch.cat([geometry, edge_attr, bond], dim=-1))
        for layer in self.layers:
            node, edge = layer(node, edge, edge_index)
        return edge_index, valid_mask, node, edge


class InverseFoldingDecoder(nn.Module):
    r"""
    Graph decoder predicting residue-type logits (BoltzGen inverse folding).

    Refines the encoder node states with edge-free graph-attention layers and projects to per-token residue-type
    logits. The training ``forward`` uses random per-edge teacher-forcing visibility; ``sample`` decodes
    autoregressively with optional per-residue amino-acid constraints and homomer-symmetry tying.

    Args:
        node_size: Width of the GNN node channel.
        edge_size: Width of the encoder edge channel (the decoder edge is ``edge_size + node_size`` wide).
        hidden_size: Width of the graph-attention hidden channel.
        num_layers: Number of graph-attention layers.
        num_attention_heads: Number of attention heads per layer.
        vocab_size: Number of residue-type tokens.
        dropout: Dropout probability inside the graph-attention layers.
        inverse_fold_restriction: Canonical amino acids (three-letter codes) globally blocked during sampling.
        tie_symmetric_sequences: Whether to tie homomer copies during autoregressive decoding.
        constraint_inf: Additive logit penalty magnitude applied to a blocked amino acid.
    """

    def __init__(
        self,
        node_size: int = 128,
        edge_size: int = 128,
        hidden_size: int = 128,
        num_layers: int = 3,
        num_attention_heads: int = 4,
        vocab_size: int = NUM_RESIDUE_TYPES,
        dropout: float = 0.0,
        inverse_fold_restriction: list[str] | None = None,
        tie_symmetric_sequences: bool = False,
        constraint_inf: float = 1e6,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.inverse_fold_restriction = list(inverse_fold_restriction or [])
        self.tie_symmetric_sequences = tie_symmetric_sequences
        self.constraint_inf = constraint_inf
        # Each layer's edge state is cat[encoder_edge, neighbour_node + residue_rep] = edge_size + node_size.
        decoder_edge_size = edge_size + node_size
        self.layers = nn.ModuleList(
            GraphAttentionLayerNoEdge(
                node_size=node_size,
                edge_size=decoder_edge_size,
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                dropout=dropout,
            )
            for _ in range(num_layers)
        )
        self.residue_type_projection = nn.Linear(vocab_size, node_size)
        self.residue_type_head = nn.Linear(node_size, vocab_size, bias=False)

    def forward(
        self, node: Tensor, edge: Tensor, edge_index: Tensor, valid_mask: Tensor, batch: BoltzGenInput
    ) -> Tensor:
        r"""
        Predict per-token residue-type logits with teacher forcing and random per-edge neighbour visibility.

        Each edge reveals its source residue type only when ``rand[src] < rand[dst]``, so every node attends to a
        random subset of ground-truth neighbour identities; the decoder graph layers then refine the nodes and the
        residue head produces logits, scattered back to the dense token grid.

        Args:
            node: Packed encoder node states ``(N_valid, node_size)``.
            edge: Packed encoder edge states ``(E, edge_size)``.
            edge_index: Edge endpoints ``(2, E)`` into packed valid tokens.
            valid_mask: Boolean token validity mask ``(B, N_token)``.
            batch: Typed model input (supplies ground-truth ``residue_type``).

        Returns:
            Dense per-token residue-type logits ``(B, N_token, vocab_size)`` (zero at invalid tokens).
        """

        src, dst = edge_index[0], edge_index[1]
        with torch.no_grad():
            # Teacher forcing with random per-edge visibility: an edge reveals its source residue type only when
            # rand[src] < rand[dst], so each node sees a random subset of neighbour identities.
            rand = torch.rand(int(valid_mask.sum()), device=valid_mask.device)
            visibility = rand[src] < rand[dst]
            known = batch.residue_type.bool()[valid_mask][src] * visibility[:, None]
            known = known.to(edge.dtype)
        residue_rep = self.residue_type_projection(known)
        neighbours = torch.cat([edge, node[src] + residue_rep], dim=-1)

        for layer in self.layers:
            node = layer(node, neighbours, edge_index)

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
        batch: BoltzGenInput,
        *,
        temperature: float | None = None,
        generator: torch.Generator | None = None,
    ) -> InverseFoldingSample:
        r"""
        Autoregressively decode residue types one position at a time, in a random order.

        Each position is decoded from its (already-decoded) neighbours: the edge state carries
        ``neighbour_node + residue_rep(decoded_neighbour)``, the decoder layers run via their single-node
        ``sample`` path, and the canonical-amino-acid logits are argmax'd (or temperature-sampled). Non-design
        positions are fixed to ``res_type_clone``.

        Faithful to the upstream boltzgen ``InverseFoldingDecoder.sample``, including per-residue amino-acid
        constraints (``batch.aa_constraint_mask`` + global ``inverse_fold_restriction``, combined into an additive
        canonical-logit mask via ``build_constraint_logit_mask``) and, when ``tie_symmetric_sequences`` is set,
        homomer-symmetry tying (logits are averaged across a symmetric group, sampled once, and applied to every
        copy). Input-side masks (``aa_constraint_mask``, ``symmetric_group``) are built by the processor.

        Args:
            node: Packed encoder node states of shape `[num_valid, node_size]`.
            edge: Packed encoder edge states of shape `[E, edge_size]`.
            edge_index: Edge endpoints of shape `[2, E]` indexing into the packed valid tokens.
            valid_mask: Boolean token validity mask of shape `[B, num_tokens]`.
            batch: The typed model input, supplying ``design_mask``, ground-truth ``residue_type``, and the
                per-residue / symmetry constraint masks.
            temperature: Optional softmax temperature for stochastic sampling; ``None`` (or 0) argmaxes instead.
            generator: Optional random generator seeding the decode order and any temperature sampling.

        Returns:
            An [`InverseFoldingSample`][ofoldx.models.boltzgen.model.InverseFoldingSample] carrying the decoded
            per-token residue types and their logits.
        """

        src, dst = edge_index[0], edge_index[1]
        num_nodes = node.shape[0]
        if batch.design_mask is None:
            raise ValueError("design_mask is required for inverse-folding sampling")
        design_mask = batch.design_mask.bool()[valid_mask]
        decoded = node.new_zeros(num_nodes, self.vocab_size)
        logits = node.new_zeros(num_nodes, self.vocab_size)

        constraint_mask = None if batch.aa_constraint_mask is None else batch.aa_constraint_mask[valid_mask]
        per_residue_mask = build_constraint_logit_mask(
            num_nodes=num_nodes,
            aa_constraint_mask=constraint_mask,
            inverse_fold_restriction=self.inverse_fold_restriction,
            inf=self.constraint_inf,
            device=node.device,
        )

        order = torch.randperm(num_nodes, device=node.device, generator=generator).tolist()
        if (~design_mask).any():
            known = batch.residue_type[valid_mask][~design_mask].to(decoded.dtype)
            decoded[~design_mask] = known
            logits[~design_mask] = known
            fixed = set(torch.where(~design_mask)[0].tolist())
            order = [position for position in order if position not in fixed]

        symmetry_groups, position_to_group = self._build_symmetric_groups(batch, valid_mask, design_mask)
        sampled: set[int] = set()

        for position in order:
            if position in sampled:
                continue
            positions = (
                symmetry_groups.get(position_to_group[position], [position])
                if position in position_to_group
                else [position]
            )

            aggregated_logits: Tensor | None = None
            for pos in positions:
                incoming = dst == pos
                neighbours = torch.cat(
                    [edge[incoming], node[src[incoming]] + self.residue_type_projection(decoded[src[incoming]])],
                    dim=-1,
                )
                state = node[pos : pos + 1]
                for layer in self.layers:
                    state = layer.sample(state, neighbours)
                position_logits = self.residue_type_head(state)
                aggregated_logits = (
                    position_logits if aggregated_logits is None else aggregated_logits + position_logits
                )
            assert aggregated_logits is not None
            aggregated_logits = aggregated_logits / len(positions)

            canonical = (
                aggregated_logits[:, CANONICALS_OFFSET : CANONICALS_OFFSET + NUM_CANONICAL_TOKENS]
                + per_residue_mask[position : position + 1]
            )
            if temperature is None:
                chosen = canonical.argmax(dim=-1)
            else:
                chosen = torch.multinomial(
                    torch.softmax(canonical / temperature, dim=-1),
                    num_samples=1,
                    generator=generator,
                ).squeeze(-1)
            one_hot = torch.nn.functional.one_hot(chosen + CANONICALS_OFFSET, self.vocab_size).to(decoded.dtype)
            for pos in positions:
                decoded[pos] = one_hot
                logits[pos] = aggregated_logits
                sampled.add(pos)

        batch_size, num_tokens = valid_mask.shape
        dense_logits = logits.new_zeros(batch_size, num_tokens, self.vocab_size)
        dense_logits[valid_mask] = logits
        dense_res_type = decoded.new_zeros(batch_size, num_tokens, self.vocab_size)
        dense_res_type[valid_mask] = decoded
        return InverseFoldingSample(logits=dense_logits, residue_type=dense_res_type, valid_mask=valid_mask)

    def _build_symmetric_groups(
        self, batch: BoltzGenInput, valid_mask: Tensor, design_mask: Tensor
    ) -> tuple[dict[int, list[int]], dict[int, int]]:
        r"""
        Group designable packed-token positions that must share a residue type under homomer symmetry.

        Returns ``(groups, position_to_group)`` where ``groups`` maps a group id to its member positions (only
        groups with more than one member are kept) and ``position_to_group`` maps a position to its group id.
        Returns empty maps unless ``tie_symmetric_sequences`` is set and ``symmetric_group`` is available; positions
        are tied when they share the same ``(symmetric_group, residue_index)`` key.
        """

        if not self.tie_symmetric_sequences or batch.symmetric_group is None:
            return {}, {}
        symmetric_group = batch.symmetric_group[valid_mask]
        residue_index = batch.residue_index[valid_mask]
        key_to_positions: dict[tuple[int, int], list[int]] = {}
        for position in range(symmetric_group.shape[0]):
            if not bool(design_mask[position]):
                continue
            group = int(symmetric_group[position])
            if group <= 0:  # 0 = no symmetry group.
                continue
            key = (group, int(residue_index[position]))
            key_to_positions.setdefault(key, []).append(position)

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


class TokenDistanceModule(nn.Module):
    r"""
    Encode inter-token backbone geometry into an additive pair update (BoltzGen TokenDistanceModule).

    Args:
        pair_size: Width of the trunk pair representation channel.
        token_distance_size: Hidden width of the module.
        num_blocks: Number of pair-only pairformer blocks.
        num_bins: Number of center-distance distogram bins.
        min_dist: Lower edge (angstrom) of the distogram binning.
        max_dist: Upper edge (angstrom) of the distogram binning.
        distance_gaussian_dim: Number of Gaussian basis functions in the distance encoder.
        triangle_attention_hidden_size: Per-head width of the triangle attention.
        num_triangle_attention_heads: Number of triangle-attention heads.
        operator_backend: Operator backend selector (``"auto"``, ``"oops"``, or ``"public"``).
    """

    def __init__(
        self,
        pair_size: int = 128,
        token_distance_size: int = 64,
        num_blocks: int = 2,
        num_bins: int = 38,
        min_dist: float = 3.25,
        max_dist: float = 50.75,
        distance_gaussian_dim: int = 32,
        triangle_attention_hidden_size: int = 32,
        num_triangle_attention_heads: int = 4,
        operator_backend: str = "auto",
    ) -> None:
        super().__init__()
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.num_bins = num_bins
        self.relu = nn.ReLU()
        self.norm_pair = nn.LayerNorm(pair_size)
        self.hidden_norm = nn.LayerNorm(token_distance_size)
        self.pair_projection = nn.Linear(pair_size, token_distance_size, bias=False)
        self.distance_feature_projection = nn.Linear(num_bins + 4 * pair_size, token_distance_size, bias=False)
        self.output_projection = nn.Linear(token_distance_size, pair_size, bias=False)
        self.pairformer = nn.ModuleList(
            TrianglePairUpdateBlock(
                pair_size=token_distance_size,
                triangle_multiplication_hidden_size=token_distance_size,
                triangle_attention_hidden_size=triangle_attention_hidden_size,
                num_triangle_attention_heads=num_triangle_attention_heads,
                operator_backend=operator_backend,
            )
            for _ in range(num_blocks)
        )
        self.distance_encoder = DistanceTokenEncoder(
            distance_gaussian_dim=distance_gaussian_dim, pair_size=pair_size, out_dim=pair_size
        )

    def forward(
        self,
        pair: Tensor,
        batch: BoltzGenInput,
        relative_position_encoding: Tensor,
        pair_mask: Tensor | None = None,
    ) -> Tensor:
        r"""
        Encode inter-token center-atom distances into an additive pair update.

        Bins center-coordinate distances into a distogram, concatenates per-backbone-atom distance features,
        masks by ``token_distance_mask``, projects together with the trunk pair, refines through the pair-only
        pairformer (with one outer residual), and projects back to ``pair_size``.

        Args:
            pair: Trunk pair representation ``(B, N_token, N_token, pair_size)``.
            batch: Typed model input (supplies ``center_coords`` and ``token_distance_mask``).
            relative_position_encoding: Relative-position pair features ``(B, N_token, N_token, pair_size)``.
            pair_mask: Optional boolean pair mask ``(B, N_token, N_token)``.

        Returns:
            Additive pair update ``(B, N_token, N_token, pair_size)``.
        """

        coords = batch.center_coords
        distances = torch.cdist(coords, coords)
        boundaries = torch.linspace(self.min_dist, self.max_dist, self.num_bins - 1, device=distances.device)
        distogram = F.one_hot((distances[..., None] > boundaries).sum(dim=-1).long(), num_classes=self.num_bins)

        distance_features = torch.cat([distogram, self.distance_encoder(relative_position_encoding, batch)], dim=-1)
        distance_features = distance_features * batch.token_distance_mask.unsqueeze(-1)
        value = self.pair_projection(self.norm_pair(pair)) + self.distance_feature_projection(
            distance_features.to(pair.dtype)
        )
        refined = value
        for block in self.pairformer:
            refined = block(refined, pair_mask=pair_mask)
        value = self.hidden_norm(value + refined)  # one outer residual around the (internally-residual) stack
        return self.output_projection(self.relu(value))


class TemplateModule(nn.Module):
    r"""
    BoltzGen v1 template module: on-the-fly template geometry -> additive pair update.

    Same wrapper as [`TokenDistanceModule`][ofoldx.models.boltzgen.model.TokenDistanceModule] (projections +
    pair-only pairformer), but the per-edge feature is the template geometry featurization (CB distogram +
    per-frame CA unit vectors + cb/frame/visibility masks + paired residue types), aggregated over the template
    axis.

    Args:
        pair_size: Width of the trunk pair representation channel.
        template_size: Hidden width of the module.
        num_blocks: Number of pair-only pairformer blocks.
        vocab_size: Number of residue-type tokens used by the template residue-type projection.
        num_bins: Number of template CB-distance distogram bins.
        min_dist: Lower edge (angstrom) of the distogram binning.
        max_dist: Upper edge (angstrom) of the distogram binning.
        triangle_attention_hidden_size: Per-head width of the triangle attention.
        num_triangle_attention_heads: Number of triangle-attention heads.
        miniformer_blocks: Whether the pairformer uses compact Miniformer pair updates.
        operator_backend: Operator backend selector (``"auto"``, ``"oops"``, or ``"public"``).
    """

    def __init__(
        self,
        pair_size: int = 128,
        template_size: int = 64,
        num_blocks: int = 2,
        vocab_size: int = 33,
        num_bins: int = 38,
        min_dist: float = 3.25,
        max_dist: float = 50.75,
        triangle_attention_hidden_size: int = 32,
        num_triangle_attention_heads: int = 4,
        miniformer_blocks: bool = False,
        operator_backend: str = "auto",
    ) -> None:
        super().__init__()
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.num_bins = num_bins
        self.relu = nn.ReLU()
        self.norm_pair = nn.LayerNorm(pair_size)
        self.hidden_norm = nn.LayerNorm(template_size)
        self.pair_projection = nn.Linear(pair_size, template_size, bias=False)
        self.template_feature_projection = nn.Linear(2 * vocab_size + num_bins + 5, template_size, bias=False)
        self.output_projection = nn.Linear(template_size, pair_size, bias=False)
        if miniformer_blocks:
            self.pairformer = nn.ModuleList(
                MiniTrianglePairUpdateBlock(pair_size=template_size, operator_backend=operator_backend)
                for _ in range(num_blocks)
            )
        else:
            self.pairformer = nn.ModuleList(
                TrianglePairUpdateBlock(
                    pair_size=template_size,
                    triangle_multiplication_hidden_size=template_size,
                    triangle_attention_hidden_size=triangle_attention_hidden_size,
                    num_triangle_attention_heads=num_triangle_attention_heads,
                    operator_backend=operator_backend,
                )
                for _ in range(num_blocks)
            )

    def forward(self, pair: Tensor, batch: BoltzGenInput, pair_mask: Tensor | None = None) -> Tensor:
        r"""
        Featurize template geometry into an additive pair update, aggregated over the template axis.

        Per template, builds the per-edge feature (CB distogram + per-frame CA unit vectors + cb/frame/visibility
        masks + query/key residue-type projections), projects together with the trunk pair, refines through the
        pair-only pairformer (batched over templates with one outer residual), then averages over present
        templates and projects back to ``pair_size``.

        Args:
            pair: Trunk pair representation ``(B, N_token, N_token, pair_size)``.
            batch: Typed model input (supplies the template features via ``_require_template_features``).
            pair_mask: Optional boolean pair mask ``(B, N_token, N_token)``.

        Returns:
            Additive pair update ``(B, N_token, N_token, pair_size)``.

        Raises:
            ValueError: If any required template feature is missing.
        """

        (
            res_type,
            frame_mask,
            frame_rotation,
            frame_translation,
            cb_coords,
            ca_coords,
            cb_mask,
            template_mask,
            visibility,
        ) = _require_template_features(batch)
        present = template_mask.any(dim=2).float()
        count = present.sum(dim=1).clamp(min=1)
        batch_size, num_templates, num_tokens = res_type.shape[:3]

        pair_cb_mask = (cb_mask[..., :, None] * cb_mask[..., None, :]).unsqueeze(-1)
        pair_frame_mask = (frame_mask[..., :, None] * frame_mask[..., None, :]).unsqueeze(-1)
        same_visibility = (visibility[..., :, None] == visibility[..., None, :]).float()

        cb_distances = torch.cdist(cb_coords, cb_coords)
        boundaries = torch.linspace(self.min_dist, self.max_dist, self.num_bins - 1, device=cb_distances.device)
        distogram = F.one_hot((cb_distances[..., None] > boundaries).sum(dim=-1).long(), num_classes=self.num_bins)

        rotation = frame_rotation.unsqueeze(2).transpose(-1, -2)
        translation = frame_translation.unsqueeze(2).unsqueeze(-1)
        targets = ca_coords.unsqueeze(3).unsqueeze(-1)
        vector = torch.matmul(rotation, targets - translation)
        norm = torch.norm(vector, dim=-1, keepdim=True)
        unit_vector = torch.where(norm > 0, vector / norm, torch.zeros_like(vector)).squeeze(-1)

        geom = torch.cat([distogram.to(pair.dtype), pair_cb_mask, unit_vector, pair_frame_mask], dim=-1)
        geom = geom * same_visibility.unsqueeze(-1)
        # Project the per-token residue types and broadcast over the pair, instead of materialising the
        # (B, T, N, N, 2*K) expanded one-hots: F.linear(one_hot, W_slice) == one_hot @ W_slice columns (exact).
        residue = res_type.to(pair.dtype)
        vocab_size = residue.shape[-1]
        weight = self.template_feature_projection.weight  # (template_size, num_bins + 5 + 2*vocab_size)
        geom_dim = weight.shape[1] - 2 * vocab_size
        feature_value = (
            F.linear(geom.to(pair.dtype), weight[:, :geom_dim])
            + F.linear(residue, weight[:, geom_dim : geom_dim + vocab_size]).unsqueeze(-2)  # query (i) term
            + F.linear(residue, weight[:, geom_dim + vocab_size :]).unsqueeze(-3)  # key (j) term
        )
        value = self.pair_projection(self.norm_pair(pair[:, None])) + feature_value
        value = value.reshape(batch_size * num_templates, num_tokens, num_tokens, -1)
        if pair_mask is not None:
            pair_mask = (
                pair_mask[:, None]
                .expand(-1, num_templates, -1, -1)
                .reshape(batch_size * num_templates, num_tokens, num_tokens)
            )
        refined = value
        for block in self.pairformer:
            refined = block(refined, pair_mask=pair_mask)
        value = self.hidden_norm(value + refined).reshape(batch_size, num_templates, num_tokens, num_tokens, -1)

        aggregated = (value * present[:, :, None, None, None]).sum(dim=1) / count[:, None, None, None].to(value.dtype)
        return self.output_projection(self.relu(aggregated))


class DistanceTokenEncoder(nn.Module):
    r"""
    Per-backbone-atom pairwise distance features conditioned on the relative-position encoding.

    Args:
        distance_gaussian_dim: Number of Gaussian basis functions in the distance smearing.
        pair_size: Width of the relative-position encoding concatenated with the distance features.
        out_dim: Output feature width per backbone-atom-pair channel.
    """

    def __init__(self, distance_gaussian_dim: int = 32, pair_size: int = 128, out_dim: int = 128) -> None:
        super().__init__()
        self.distance_smearing = BroadcastGaussianSmearing(start=0.0, stop=2.0, num_gaussians=distance_gaussian_dim)
        self.distance_token_bias_trans = ProjectionTransition(distance_gaussian_dim + 1 + pair_size, pair_size, out_dim)

    def forward(self, relative_position_encoding: Tensor, batch: BoltzGenInput) -> Tensor:
        r"""
        Compute per-backbone-atom pairwise distance features conditioned on the relative-position encoding.

        Gathers the four backbone atoms per token, RBF-smears their pairwise distances, concatenates the raw
        distance and the (broadcast) relative-position encoding, and runs a transition; the four per-atom-pair
        channels are flattened into the output feature dimension.

        Args:
            relative_position_encoding: Relative-position pair features ``(B, N_token, N_token, pair_size)``.
            batch: Typed model input (supplies ``token_to_bb4_atoms`` and ``coords``).

        Returns:
            Distance-conditioned pair features ``(B, N_token, N_token, 4 * out_dim)``.
        """

        batch_size, num_tokens = relative_position_encoding.shape[:2]
        backbone = _gather_backbone4(batch, batch_size, num_tokens).permute(0, 2, 1, 3)  # (B, 4, N, 3)
        distance = (backbone.unsqueeze(-2) - backbone.unsqueeze(-3)).norm(dim=-1).unsqueeze(-1)  # (B, 4, N, N, 1)
        gaussian = self.distance_smearing(distance)  # (B, 4, N, N, gaussians)
        relative = relative_position_encoding.view(batch_size, 1, num_tokens, num_tokens, -1).expand(-1, 4, -1, -1, -1)
        bias_input = torch.cat([gaussian, distance, relative], dim=-1)
        bias = (
            self.distance_token_bias_trans(bias_input)
            .permute(0, 2, 3, 4, 1)
            .reshape(batch_size, num_tokens, num_tokens, -1)
        )
        return bias


class GraphAttentionLayer(nn.Module):
    r"""
    One MLP-attention GNN layer with node and edge updates (encoder).

    Args:
        node_size: Width of the node channel.
        edge_size: Width of the edge channel.
        hidden_size: Width of the attention hidden channel.
        num_attention_heads: Number of attention heads.
        dropout: Dropout probability inside the layer.
    """

    def __init__(
        self,
        node_size: int = 128,
        edge_size: int = 128,
        hidden_size: int = 128,
        num_attention_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_weight_mlp = _mlp([node_size * 2 + edge_size, hidden_size, hidden_size, num_attention_heads])
        self.attention_value_mlp = _mlp([node_size + edge_size, hidden_size, hidden_size, node_size])
        self.attention_output = nn.Sequential(
            nn.Linear(hidden_size * num_attention_heads, node_size), nn.Dropout(dropout), nn.BatchNorm1d(node_size)
        )
        self.node_transition = nn.Sequential(
            nn.Linear(node_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, node_size),
            nn.Dropout(dropout),
            nn.BatchNorm1d(node_size),
        )
        self.edge_transition = nn.Sequential(
            nn.Linear(node_size * 2 + edge_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, edge_size),
            nn.Dropout(dropout),
            nn.BatchNorm1d(edge_size),
        )

    def forward(self, node: Tensor, edge: Tensor, edge_index: Tensor) -> tuple[Tensor, Tensor]:
        r"""
        Apply one multi-head MLP-attention update to both node and edge states (residual).

        Updates edges from their endpoint nodes, computes per-head attention weights (softmax over each
        destination's incoming edges) and per-edge values, scatters the messages to destinations, and adds node
        attention output plus a node transition.

        Args:
            node: Packed node states ``(N_valid, node_size)``.
            edge: Packed edge states ``(E, edge_size)``.
            edge_index: Edge endpoints ``(2, E)`` (row 0 source, row 1 destination).

        Returns:
            Tuple of updated node states ``(N_valid, node_size)`` and updated edge states ``(E, edge_size)``.
        """

        src, dst = edge_index[0], edge_index[1]
        edge = edge + self.edge_transition(torch.cat([node[src], node[dst], edge], dim=1))
        weight = self.attention_weight_mlp(torch.cat([node[dst], node[src], edge], dim=1))
        value = self.attention_value_mlp(torch.cat([node[src], edge], dim=1))
        weight = scatter_softmax(weight, dst, dim=0)
        message = scatter_sum(weight.unsqueeze(-1) * value.unsqueeze(1), dst, dim=0, dim_size=node.shape[0]).flatten(
            start_dim=1
        )
        node = node + self.attention_output(message)
        return node + self.node_transition(node), edge


class GraphAttentionLayerNoEdge(nn.Module):
    r"""
    One MLP-attention GNN layer updating nodes only (decoder).

    Unlike the encoder layer this does NOT mirror the message construction: the decoder's edge state already
    carries the source-node information, so the attention weight is a function of only the destination node and
    the edge (``cat[node[dst], edge]``) and the value is a function of only the edge (``edge``). The edge dim is
    therefore wider than the encoder's (the reference uses ``pair_dim`` = 256). Faithful to boltzgen
    ``MLPAttnGNNDecoder.forward``.

    Args:
        node_size: Width of the node channel.
        edge_size: Width of the (wider) decoder edge channel.
        hidden_size: Width of the attention hidden channel.
        num_attention_heads: Number of attention heads.
        dropout: Dropout probability inside the layer.
    """

    def __init__(
        self,
        node_size: int = 128,
        edge_size: int = 256,
        hidden_size: int = 128,
        num_attention_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_weight_mlp = _mlp([node_size + edge_size, hidden_size, hidden_size, num_attention_heads])
        self.attention_value_mlp = _mlp([edge_size, hidden_size, hidden_size, node_size])
        self.attention_output = nn.Sequential(
            nn.Linear(hidden_size * num_attention_heads, node_size), nn.Dropout(dropout), nn.BatchNorm1d(node_size)
        )
        self.node_transition = nn.Sequential(
            nn.Linear(node_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, node_size),
            nn.Dropout(dropout),
            nn.BatchNorm1d(node_size),
        )

    def forward(self, node: Tensor, edge: Tensor, edge_index: Tensor) -> Tensor:
        r"""
        Apply one multi-head MLP-attention update to node states only (edges held fixed; residual).

        Attention weights depend on ``cat[node[dst], edge]`` and values on ``edge`` alone; messages are softmaxed
        per destination, scattered, and added to the node plus a node transition.

        Args:
            node: Packed node states ``(N_valid, node_size)``.
            edge: Packed decoder edge states ``(E, edge_size)`` carrying neighbour-node + residue information.
            edge_index: Edge endpoints ``(2, E)`` (destination in row 1).

        Returns:
            Updated node states ``(N_valid, node_size)``.
        """

        dst = edge_index[1]
        weight = self.attention_weight_mlp(torch.cat([node[dst], edge], dim=1))
        value = self.attention_value_mlp(edge)
        weight = scatter_softmax(weight, dst, dim=0)
        message = scatter_sum(weight.unsqueeze(-1) * value.unsqueeze(1), dst, dim=0, dim_size=node.shape[0]).flatten(
            start_dim=1
        )
        node = node + self.attention_output(message)
        return node + self.node_transition(node)

    def sample(self, node: Tensor, edge: Tensor) -> Tensor:
        r"""
        Run one node-update for a single decode position (the batched ``forward`` specialised to one node).

        All incoming edges point at node 0, so no scatter index is needed.

        Args:
            node: State of the single node being decoded, of shape `[1, node_size]`.
            edge: Incoming decoder edge states of shape `[E, edge_size]`, all directed at node 0.

        Returns:
            Updated single-node state of shape `[1, node_size]`.
        """
        dst = torch.zeros(edge.shape[0], dtype=torch.long, device=edge.device)
        weight = self.attention_weight_mlp(torch.cat([node[dst], edge], dim=1))
        value = self.attention_value_mlp(edge)
        weight = scatter_softmax(weight, dst, dim=0)
        message = scatter_sum(weight.unsqueeze(-1) * value.unsqueeze(1), dst, dim=0, dim_size=1).flatten(start_dim=1)
        node = node + self.attention_output(message)
        return node + self.node_transition(node)


class BroadcastGaussianSmearing(nn.Module):
    r"""
    Radial-basis distance features ``exp(coeff (d - offset)^2)`` broadcasting over a trailing singleton.

    Distinct from the inverse-folding [`GaussianSmearing`][ofoldx.models.boltzgen.model.GaussianSmearing],
    which flattens; here ``d`` already carries a trailing size-1 axis and the gaussians broadcast onto it.

    Args:
        start: Lowest RBF center.
        stop: Highest RBF center.
        num_gaussians: Number of RBF centers.
    """

    def __init__(self, start: float = 0.0, stop: float = 2.0, num_gaussians: int = 32) -> None:
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, distance: Tensor) -> Tensor:
        r"""
        RBF-smear distances over ``num_gaussians`` centers, broadcasting them onto the trailing singleton axis.

        Args:
            distance: Distances of shape `[B, ..., 1]` (the trailing size-1 axis is broadcast over the centers).

        Returns:
            RBF features of shape `[B, ..., num_gaussians]`.
        """

        return torch.exp(self.coeff * (distance - self.offset.view(1, 1, 1, -1)).pow(2))


class GaussianSmearing(nn.Module):
    r"""
    Radial-basis distance featurization (frozen ``offset`` buffer).

    Args:
        start: Lowest RBF center.
        stop: Highest RBF center.
        num_gaussians: Number of RBF centers.
    """

    def __init__(self, start: float = 0.0, stop: float = 5.0, num_gaussians: int = 16) -> None:
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, distances: Tensor) -> Tensor:
        r"""
        RBF-smear distances over ``num_gaussians`` centers, flattening the gaussian axis into the feature dim.

        Args:
            distances: Per-edge distances of shape `[E, D]`.

        Returns:
            RBF features of shape `[E, D * num_gaussians]`.
        """

        delta = (distances.unsqueeze(-1) - self.offset.view(1, 1, -1)).flatten(start_dim=1)
        return torch.exp(self.coeff * delta.pow(2))


class ProjectionTransition(nn.Module):
    r"""
    Channel-reshaping transition without a residual connection.

    Unlike the residual [`Transition`][ofoldx.modules.transition.Transition], input / hidden / output widths are
    independent and there is no residual. Used where a transition reshapes channels.

    Args:
        input_size: Width of the input channel.
        hidden_size: Width of the SwiGLU hidden channel.
        output_size: Width of the output channel.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(input_size)
        self.gate_projection = nn.Linear(input_size, hidden_size, bias=False)
        self.value_projection = nn.Linear(input_size, hidden_size, bias=False)
        self.output_projection = nn.Linear(hidden_size, output_size, bias=False)
        self.silu = nn.SiLU()

    def forward(self, features: Tensor) -> Tensor:
        r"""
        SwiGLU channel reshape (no residual) via ``silu(gate) * value``.

        Args:
            features: Input features of shape `[..., input_size]`.

        Returns:
            Reshaped features of shape `[..., output_size]`.
        """

        normalized = self.norm(features)
        return self.output_projection(self.silu(self.gate_projection(normalized)) * self.value_projection(normalized))


def scatter_sum(src: Tensor, index: Tensor, dim: int = 0, dim_size: int | None = None) -> Tensor:
    r"""
    Sum ``src`` rows into buckets given by ``index`` (native reimplementation of ``torch_scatter.scatter_sum``).

    Args:
        src: Source tensor whose slices along ``dim`` are summed per bucket.
        index: Destination bucket index for each slice along ``dim``.
        dim: Dimension along which to scatter.
        dim_size: Number of output buckets. Defaults to ``index.max() + 1``.

    Returns:
        Tensor with ``dim_size`` buckets along ``dim``, each the sum of its assigned ``src`` slices.

    Examples:
        >>> import torch
        >>> src = torch.tensor([[1.0], [2.0], [3.0]])
        >>> scatter_sum(src, torch.tensor([0, 0, 1])).tolist()
        [[3.0], [3.0]]
    """

    if dim_size is None:
        dim_size = int(index.max()) + 1 if index.numel() > 0 else 0
    shape = list(src.shape)
    shape[dim] = dim_size
    out = torch.zeros(shape, dtype=src.dtype, device=src.device)
    expanded = index
    for _ in range(src.dim() - index.dim()):
        expanded = expanded.unsqueeze(-1)
    out.scatter_add_(dim, expanded.expand_as(src), src)
    return out


def scatter_softmax(src: Tensor, index: Tensor, dim: int = 0) -> Tensor:
    r"""
    Softmax over rows sharing a destination ``index`` (per-group / segmented softmax).

    Args:
        src: Source tensor whose slices along ``dim`` are softmaxed within each group.
        index: Group index for each slice along ``dim``.
        dim: Dimension along which to group.

    Returns:
        Tensor the same shape as ``src`` whose slices sum to one within each group.

    Examples:
        >>> import torch
        >>> weight = scatter_softmax(torch.tensor([1.0, 1.0, 5.0]), torch.tensor([0, 0, 1]))
        >>> weight.round(decimals=4).tolist()
        [0.5, 0.5, 1.0]
    """

    dim_size = int(index.max()) + 1 if index.numel() > 0 else 0
    group_max = src.new_full((dim_size, *src.shape[1:]), float("-inf"))
    expanded = index
    for _ in range(src.dim() - index.dim()):
        expanded = expanded.unsqueeze(-1)
    group_max.scatter_reduce_(0, expanded.expand_as(src), src, reduce="amax", include_self=True)
    centered = src - group_max.gather(0, expanded.expand_as(src))
    exp = centered.exp()
    denom = scatter_sum(exp, index, dim=dim, dim_size=dim_size).gather(0, expanded.expand_as(src))
    return exp / (denom + 1e-16)


def _mlp(sizes: list[int]) -> nn.Sequential:
    r"""
    Build a GELU-activated MLP from consecutive ``sizes`` (linear layers with GELU between, none after the last).

    Args:
        sizes: Layer widths; ``len(sizes) - 1`` linear layers are created.

    Returns:
        The MLP as an ``nn.Sequential``.
    """

    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.GELU())
    return nn.Sequential(*layers)


def _sample_axis(coords: Tensor, *, batch_size: int, num_samples: int) -> Tensor:
    r"""
    Reshape a flattened ``(batch_size*num_samples, ...)`` tensor back into a ``(batch_size, num_samples, ...)`` one.

    Args:
        coords: Tensor whose leading axis is ``batch_size*num_samples`` (or ``batch_size`` when ``num_samples`` is 1).
        batch_size: Number of batch elements.
        num_samples: Number of samples per batch element.

    Returns:
        Tensor with the sample axis restored: ``(batch_size, num_samples, ...)``.

    Raises:
        ValueError: If the leading axis is not ``batch_size*num_samples``.
    """

    if coords.shape[0] == batch_size * num_samples:
        return coords.reshape(batch_size, num_samples, *coords.shape[1:])
    if num_samples == 1 and coords.shape[0] == batch_size:
        return coords.unsqueeze(1)
    raise ValueError(
        "pred_coords leading axis must equal batch_size*num_samples, got "
        f"{coords.shape[0]} for batch_size={batch_size} and num_samples={num_samples}"
    )


@dataclass(slots=True)
class BoltzGenRepresentations:
    r"""
    Trunk outputs consumed by BoltzGen diffusion and heads.

    Attributes:
        single_inputs: Initial token single representations.
        single: Final token single representations.
        pair: Final token pair representations.
        relative_position_encoding: Relative-position pair features.
        distogram_logits: Trunk distogram logits.
    """

    single_inputs: Tensor
    single: Tensor
    pair: Tensor
    relative_position_encoding: Tensor
    distogram_logits: Tensor


@dataclass(slots=True, kw_only=True)
class BoltzGenInput(BaseInput):
    r"""
    Input tensors consumed by BoltzGen design models.

    Attributes:
        token_mask: Boolean token validity mask with shape ``(B, N_token)``.
        residue_index: Chain-local residue index used by relative-position features.
        chain_index: Chain ids mapped to contiguous integer indices.
        entity_index: Entity ids shared by chains with the same sequence.
        symmetry_index: Symmetry-copy ids for relative-position features.
        token_index: Token ids in structure order.
        atom_features: Per-atom reference features.
        atom_to_token: Token index for each atom.
        atom_mask: Boolean atom validity mask.
        reference_positions: Centered reference-conformer atom coords.
        reference_space_id: Reference-conformer group id for each atom.
        representative_atom_mask: Mask selecting the distogram representative atom per token.
        atom_token_index: Per-token atom offset for each atom.
        token_to_rep_atom: Distogram representative atom index for each token.
        residue_type: Query residue type one-hot features.
        msa_profile: Single-sequence MSA profile features.
        deletion_mean: Per-token deletion mean features.
        token_bonds: Token-level covalent adjacency.
        type_bonds: Token-level bond type ids.
        contact_conditioning: Pairwise contact-conditioning class one-hot features.
        contact_threshold: Pairwise contact threshold values.
        chain_type: Chain type ids in the model vocabulary.
        center_coords: Per-token representative atom coords.
        token_to_bb4_atoms: Per-token ``(N, CA, C, O)`` atom indices.
        coords: Training or conditioning atom coords.
        token_distance_mask: Pairwise token-distance conditioning mask.
        target_positions: Optional experimental atom coords for training.
        target_mask: Optional experimental-coordinate validity mask.
        molecule_type: Optional molecule type id per token.
        frame_atom_index: Optional frame atom indices for frame-aligned losses.
        frame_mask: Optional frame validity mask.
        method_feature: Optional experimental-method ids.
        modified: Optional modified-residue flags.
        cyclic_period: Optional cyclic-position periods.
        msa: Optional integer MSA token ids.
        has_deletion: Optional MSA deletion flags.
        deletion_value: Optional MSA deletion values.
        msa_paired: Optional paired-MSA flags.
        msa_mask: Optional MSA validity mask.
        template_restype: Optional template residue type ids.
        template_frame_rot: Optional template frame rotations.
        template_frame_t: Optional template frame translations.
        template_mask_frame: Optional template frame validity mask.
        template_cb: Optional template representative-atom coords.
        template_ca: Optional template alpha-carbon coords.
        template_mask_cb: Optional template representative-atom mask.
        template_mask: Optional template validity mask.
        visibility_ids: Optional visibility ids for stochastic conditioning.
        binding_type: Optional binding conditioning ids.
        design_mask: Optional token mask selecting designable positions.
        ss_type: Optional secondary-structure conditioning ids.
        affinity_token_mask: Optional mask selecting the ligand tokens scored by the affinity head.
        affinity_mw: Optional ligand molecular weight used by the affinity molecular-weight correction.
        aa_constraint_mask: Optional per-token per-canonical-AA blocking mask for constrained inverse folding
            (``(B, N_token, 20)``, ``1`` = blocked).
        symmetric_group: Optional per-token symmetry-copy id used to tie homomer sequences during decoding.
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
    residue_type: Tensor
    msa_profile: Tensor
    deletion_mean: Tensor
    token_bonds: Tensor
    type_bonds: Tensor
    contact_conditioning: Tensor
    contact_threshold: Tensor
    chain_type: Tensor
    center_coords: Tensor
    token_to_bb4_atoms: Tensor
    coords: Tensor
    token_distance_mask: Tensor
    target_positions: Tensor | None = None
    target_mask: Tensor | None = None
    molecule_type: Tensor | None = None
    frame_atom_index: Tensor | None = None
    frame_mask: Tensor | None = None
    method_feature: Tensor | None = None
    modified: Tensor | None = None
    cyclic_period: Tensor | None = None
    msa: Tensor | None = None
    has_deletion: Tensor | None = None
    deletion_value: Tensor | None = None
    msa_paired: Tensor | None = None
    msa_mask: Tensor | None = None
    template_restype: Tensor | None = None
    template_frame_rot: Tensor | None = None
    template_frame_t: Tensor | None = None
    template_mask_frame: Tensor | None = None
    template_cb: Tensor | None = None
    template_ca: Tensor | None = None
    template_mask_cb: Tensor | None = None
    template_mask: Tensor | None = None
    visibility_ids: Tensor | None = None
    binding_type: Tensor | None = None
    design_mask: Tensor | None = None
    ss_type: Tensor | None = None
    affinity_token_mask: Tensor | None = None
    affinity_mw: Tensor | None = None
    aa_constraint_mask: Tensor | None = None
    symmetric_group: Tensor | None = None


@dataclass(slots=True)
class BoltzGenStructureOutput:
    r"""
    Structure-design inference outputs before sample-axis restoration.

    Attributes:
        pred_coords: Sampled atom coords.
        distogram_logits: Pairwise distance logits.
        bfactor_logits: Predicted B-factor logits.
        plddt_logits: Optional per-token pLDDT logits (confidence head).
        pae_logits: Optional predicted aligned-error logits (confidence head).
        pde_logits: Optional predicted distance-error logits (confidence head).
        resolved_logits: Optional resolved-atom logits (confidence head).
    """

    pred_coords: Tensor
    distogram_logits: Tensor
    bfactor_logits: Tensor
    plddt_logits: Tensor | None = None
    pae_logits: Tensor | None = None
    pde_logits: Tensor | None = None
    resolved_logits: Tensor | None = None


@dataclass(slots=True)
class InverseFoldingSample:
    r"""
    Autoregressive inverse-folding sample result.

    Attributes:
        logits: Per-step residue logits.
        residue_type: Sampled residue type ids.
        valid_mask: Boolean mask for sampled positions.
    """

    logits: Tensor
    residue_type: Tensor
    valid_mask: Tensor
