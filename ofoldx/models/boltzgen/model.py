r"""BoltzGen raw-tensor design models.

Model-first order: the public ``BoltzGenModel`` (structure design) and ``BoltzGenIFoldModel``
(inverse folding) come first, then their supporting ``nn.Module`` layers/helpers, then the typed
contract dataclasses (Input/Output/Layout/Result) and family-local conditions last. The family is
registered (see ``__init__``); its multi-stage ``design`` / ``inverse_fold`` flow is carried by the
concrete ``BoltzGenPipeline``, exported for manual construction and not yet routed through the
(unfrozen) concrete-Pipeline registry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import partial
from typing import Literal, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from typing_extensions import Self

from ...data.bonds import SymbolicBond
from ...data.conditions import Contact, Hotspot
from ...data.molecular import AtomSite, ComponentReference, MolecularSystem
from ...data.placement import (
    PlacementMap,
    PlacementSpec,
    validate_placement,
)
from ...data.structure import Structure
from ...modules.adaptive_norm import AdaptiveLayerNorm
from ...modules.atom.encoders import PackedAtomAttention
from ...modules.atom.pair_encoding import encode_atom_pair
from ...modules.atom.transfer import (
    TokenPairToLocalAtomPairBroadcast,
    TokenToAtomBroadcast,
    dense_mean_to_tokens,
)
from ...modules.atom.window import gather_key_windows
from ...modules.attention import GatedMultiheadAttention, PairBiasedSelfAttention, TriangleMultiplication
from ...modules.backend import LayerNorm
from ...modules.diffusion.conditioning import condition_pair, condition_single
from ...modules.diffusion.edm import EdmPreconditioner
from ...modules.embedding.fourier import FrozenFourierEmbedding
from ...modules.embedding.position import RelativePositionEmbedder, generate_relative_position_features
from ...modules.functional import pair_mask_from_token_mask
from ...modules.geometry import uniform_random_rotation, weighted_rigid_align
from ...modules.head.distogram import DistogramHead
from ...modules.transformer import ConditionedTransformer, ConditionedTransformerLayer, ConditionedTransitionBlock
from ...modules.transition import ConditionedTransition, RowwiseDropout, Transition
from ...modules.trunk.msa import MsaModule, MsaPairWeightedAveraging, OuterProductMean, TrianglePairUpdateBlock
from ...modules.trunk.pairformer import PairformerStack
from ...results import Metric, PairMetric, Score
from .config import BoltzGenConfig, BoltzGenIFoldConfig


class BoltzGenModel(nn.Module):
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

    """

    config_class = BoltzGenConfig

    def __init__(
        self,
        config: BoltzGenConfig | None = None,
    ) -> None:
        config = BoltzGenConfig() if config is None else config
        super().__init__()
        self.config = config
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
            operator_backend="auto",
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
        nn.init.zeros_(self.single_recycle_projection.weight)
        nn.init.zeros_(self.pair_recycle_projection.weight)

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
            operator_backend="auto",
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
                operator_backend="auto",
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
                operator_backend="auto",
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
                operator_backend="auto",
            )
        )
        self.pairformer_stack = (
            MiniformerStack(
                num_blocks=trunk.num_pairformer_blocks,
                single_size=single_size,
                pair_size=pair_size,
                num_pair_attention_heads=trunk.num_pair_attention_heads,
                dropout=trunk.dropout,
                operator_backend="auto",
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
                operator_backend="auto",
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
            operator_backend="auto",
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
        self.set_backend("auto")
        _zero_initialize_boltzgen_updates(self)

    def set_backend(self, backend: str) -> Self:
        r"""Select the operator implementation used by this model."""

        if backend not in {"auto", "torch", "oops"}:
            raise ValueError(f"unknown backend: {backend!r}")
        self.backend = backend
        attention_backend = "eager" if backend == "torch" else backend
        operator_backend = "eager" if backend == "torch" else backend
        for module in self.modules():
            if isinstance(module, GatedMultiheadAttention):
                module.torch_backend = "eager"
                module.operator_backend = attention_backend
            elif isinstance(
                module,
                (
                    AdaptiveLayerNorm,
                    Transition,
                    TriangleMultiplication,
                    MsaPairWeightedAveraging,
                    OuterProductMean,
                    TokenToAtomBroadcast,
                    TokenPairToLocalAtomPairBroadcast,
                ),
            ):
                module.operator_backend = operator_backend
        return self

    def forward(
        self,
        batch: BoltzGenInput,
        *,
        recycling_steps: int | None = None,
        generator: torch.Generator | None = None,
    ) -> BoltzGenOutput:
        r"""
        Training path: one EDM denoising step of the structure diffusion module.

        This is the structure-training graph in both ``train()`` and ``eval()`` modes; those modes only control
        ordinary stochastic layers such as dropout. It mirrors the upstream BoltzGen training diffusion step
        (``AtomDiffusion.forward``): run the trunk,
        center-and-augment the experimental target coords, corrupt them at a log-normal-sampled noise level, and
        denoise once with the EDM-preconditioned score network. The returned candidate carries the single-step
        training tensors (``denoised_coords``, ``augmented_target``, ``noise_level``) from which the EDM-weighted
        denoising loss is computed. This is the compile-friendly gradient path.

        Args:
            batch: Typed model input. ``target_positions`` supplies the training target; ``coords`` is the
                fallback when it is absent.
            recycling_steps: Optional number of recycling passes. Defaults to the config value.
            generator: Optional PyTorch random generator for the augmentation, noise level, and Gaussian noise.

        Returns:
            The single-step structure-training tensors.

        """

        single_inputs, single, pair, relative_position_encoding, distogram_logits = self._run_trunk(
            batch,
            recycling_steps=recycling_steps,
        )
        conditioning = self.diffusion_conditioning(
            single,
            pair,
            relative_position_encoding,
            atom_features=batch.atom_features,
            reference_positions=batch.reference_positions,
            atom_mask=batch.atom_mask,
            reference_space_id=batch.reference_space_id,
            atom_to_token=batch.atom_to_token,
        )
        target = batch.coords if batch.target_positions is None else batch.target_positions
        atom_mask = batch.atom_mask
        augmented_target = _center_random_augmentation(target, atom_mask, generator=generator)
        sigmas = self.structure_module.noise_distribution(
            atom_mask.shape[0], device=augmented_target.device, dtype=augmented_target.dtype, generator=generator
        )
        noise = torch.randn(
            augmented_target.shape, device=augmented_target.device, dtype=augmented_target.dtype, generator=generator
        )
        noised_coords = augmented_target + sigmas.reshape(-1, 1, 1) * noise
        denoised_coords = self.structure_module.preconditioned_network_forward(
            noised_coords,
            sigmas,
            multiplicity=1,
            atom_mask=atom_mask,
            single_inputs=single_inputs,
            single_trunk=single,
            conditioning=conditioning,
            atom_to_token=batch.atom_to_token,
            token_mask=batch.token_mask,
        )
        return BoltzGenOutput(
            pred_coords=denoised_coords,
            distogram_logits=distogram_logits,
            bfactor_logits=self.bfactor_head(single),
            denoised_coords=denoised_coords,
            augmented_target=augmented_target,
            noise_level=sigmas,
        )

    def design(
        self,
        batch: BoltzGenInput,
        *,
        num_samples: int = 1,
        num_sampling_steps: int | None = None,
        recycling_steps: int | None = None,
        generator: torch.Generator | None = None,
        step_scale: float | None = None,
        noise_scale: float | None = None,
    ) -> BoltzGenOutput:
        r"""Sample whole-assembly design coordinates (canonical expert operation).

        Thin public wrapper over the parity-passing ``_design`` reverse-diffusion body;
        the numerics are unchanged. Produces ``pred_coords [B, num_samples, A, 3]`` plus
        the trunk ``distogram_logits`` and ``bfactor_logits``. Sequence assignment and the
        canonical ``BoltzGenDesignResult`` decoding are owned by ``BoltzGenPipeline``.

        Args:
            batch: Typed structure-design input from ``BoltzGenProcessor``.
            num_samples: Number of reverse-diffusion samples to draw.
            num_sampling_steps: Optional EDM sampling steps. Defaults to the sampler config value.
            recycling_steps: Optional recycling passes. Defaults to the config value.
            generator: Optional PyTorch random generator for diffusion noise.
            step_scale: Optional per-call sampler step-scale override.
            noise_scale: Optional per-call sampler noise-scale override.

        Returns:
            The sampled structure output.
        """

        return self._design(
            batch,
            recycling_steps=recycling_steps,
            num_samples=num_samples,
            num_sampling_steps=num_sampling_steps,
            generator=generator,
            step_scale=step_scale,
            noise_scale=noise_scale,
        )

    @torch.no_grad()
    def _design(
        self,
        batch: BoltzGenInput,
        *,
        recycling_steps: int | None = None,
        num_samples: int = 1,
        num_sampling_steps: int | None = None,
        generator: torch.Generator | None = None,
        step_scale: float | None = None,
        noise_scale: float | None = None,
    ) -> BoltzGenOutput:
        r"""
        Draw reverse-diffusion structure samples from a preprocessed batch.

        Args:
            batch: Typed model input.
            recycling_steps: Optional number of recycling passes. Defaults to the config value.
            num_samples: Number of reverse-diffusion samples to draw.
            num_sampling_steps: Optional number of EDM sampling steps. Defaults to the sampler config value.
            generator: Optional PyTorch random generator for diffusion noise.
            step_scale: Optional per-call sampler step-scale override.
            noise_scale: Optional per-call sampler noise-scale override.

        This remains private until a processor can decode it into the canonical design result.

        """

        single_inputs, single, pair, relative_position_encoding, distogram_logits = self._run_trunk(
            batch,
            recycling_steps=recycling_steps,
        )
        conditioning = self.diffusion_conditioning(
            single,
            pair,
            relative_position_encoding,
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
            single_inputs=single_inputs,
            single_trunk=single,
            conditioning=conditioning,
            atom_to_token=batch.atom_to_token,
            token_mask=batch.token_mask,
        )
        batch_size = batch.token_mask.shape[0]
        return BoltzGenOutput(
            pred_coords=pred_coords.reshape(batch_size, num_samples, *pred_coords.shape[1:]),
            distogram_logits=distogram_logits,
            bfactor_logits=self.bfactor_head(single).unsqueeze(1).expand(-1, num_samples, -1, -1),
        )

    def _run_trunk(
        self,
        batch: BoltzGenInput,
        *,
        recycling_steps: int | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        r"""
        Run the structure trunk.

        Args:
            batch: Typed model input.
            recycling_steps: Optional number of recycling passes. Defaults to the config value.

        Returns:
            ``(single_inputs, single, pair, relative_position_encoding, distogram_logits)``.

        """

        steps = self._recycling_steps if recycling_steps is None else recycling_steps
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
            binding_type=batch.binding_type,
            design_mask=batch.design_mask,
            ss_type=batch.ss_type,
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
        )
        relative_position_encoding = self.relative_position_embedder(relative_features)
        pair_init = pair_init + relative_position_encoding
        pair_init = pair_init + self.token_bond_projection(batch.token_bonds.to(pair_init.dtype))
        pair_init = pair_init + self.token_bond_type_embedding(batch.type_bonds.long())
        pair_init = pair_init + self.contact_conditioning(
            batch.contact_conditioning.to(pair_init.dtype), batch.contact_threshold.to(pair_init.dtype)
        )

        token_mask = batch.token_mask
        pair_mask = pair_mask_from_token_mask(token_mask)
        single = torch.zeros_like(single_init)
        pair = torch.zeros_like(pair_init)
        for cycle in range(steps + 1):
            grad_enabled = torch.is_grad_enabled() and cycle == steps
            with torch.set_grad_enabled(grad_enabled):
                single = single_init + self.single_recycle_projection(self.single_recycle_norm(single))
                pair = pair_init + self.pair_recycle_projection(self.pair_recycle_norm(pair))
                if self.token_distance is not None:
                    pair = pair + self.token_distance(pair, batch, relative_position_encoding, pair_mask=pair_mask)
                if batch.template_restype is not None:
                    pair = pair + self.template_module(pair, batch, pair_mask=pair_mask)
                if batch.msa is not None:
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

        return (
            single_inputs,
            single,
            pair,
            relative_position_encoding,
            self.distogram_head(pair),
        )


class BoltzGenIFoldModel(nn.Module):
    r"""
    BoltzGen inverse-folding model for structure-conditioned sequence design.

    Encodes the input structure into a kNN token graph and autoregressively (or in a single teacher-forced pass)
    decodes per-token residue types.

    Args:
        config: Optional inverse-folding config. Defaults to ``BoltzGenIFoldConfig()``.

    """

    config_class = BoltzGenIFoldConfig

    def __init__(self, config: BoltzGenIFoldConfig | None = None) -> None:
        config = BoltzGenIFoldConfig() if config is None else config
        super().__init__()
        self.config = config
        ifold = config.ifold
        self.input_embedder = AtomConditionedInputEmbedder(
            single_size=config.single_size,
            atom_hidden_size=config.atom_hidden_size,
            atom_pair_hidden_size=config.atom_pair_hidden_size,
            atom_feature_size=config.atom_feature_size,
            num_query_atoms=ifold.num_query_atoms,
            num_key_atoms=ifold.num_key_atoms,
            num_atom_encoder_blocks=ifold.num_atom_encoder_blocks,
            num_atom_encoder_attention_heads=ifold.num_atom_encoder_attention_heads,
            vocab_size=config.input_embedder.vocab_size,
            msa_profile_size=config.input_embedder.msa_profile_size,
            num_method_types=config.input_embedder.num_method_types,
            num_molecule_types=config.input_embedder.num_molecule_types,
            operator_backend="auto",
        )
        self.encoder = InverseFoldingEncoder(
            single_size=config.single_size,
            node_size=ifold.node_size,
            edge_size=ifold.edge_size,
            hidden_size=ifold.hidden_size,
            num_layers=ifold.num_encoder_layers,
            num_attention_heads=ifold.num_attention_heads,
            num_neighbors=ifold.num_neighbors,
            dropout=ifold.dropout,
            softmax_dropout=ifold.softmax_dropout,
            inverse_fold_noise=ifold.inverse_fold_noise,
        )
        self.decoder = InverseFoldingDecoder(
            node_size=ifold.node_size,
            edge_size=ifold.edge_size,
            hidden_size=ifold.hidden_size,
            num_layers=ifold.num_decoder_layers,
            num_attention_heads=ifold.num_attention_heads,
            dropout=ifold.dropout,
            softmax_dropout=ifold.softmax_dropout,
        )
        self.set_backend("auto")
        _zero_initialize_boltzgen_updates(self)

    def set_backend(self, backend: str) -> Self:
        r"""Select the operator implementation used by this model."""

        if backend not in {"auto", "torch", "oops"}:
            raise ValueError(f"unknown backend: {backend!r}")
        self.backend = backend
        attention_backend = "eager" if backend == "torch" else backend
        operator_backend = "eager" if backend == "torch" else backend
        for module in self.modules():
            if isinstance(module, GatedMultiheadAttention):
                module.torch_backend = "eager"
                module.operator_backend = attention_backend
            elif isinstance(
                module,
                (
                    AdaptiveLayerNorm,
                    Transition,
                    TriangleMultiplication,
                    MsaPairWeightedAveraging,
                    OuterProductMean,
                    TokenToAtomBroadcast,
                    TokenPairToLocalAtomPairBroadcast,
                ),
            ):
                module.operator_backend = operator_backend
        return self

    def forward(self, batch: BoltzGenIFoldInput) -> BoltzGenIFoldOutput:
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
        return BoltzGenIFoldOutput(
            logits=self.decoder(node, edge, edge_index, valid_mask, batch),
        )

    def design(
        self,
        batch: BoltzGenIFoldInput,
        *,
        temperature: float | None = None,
        inverse_fold_restriction: tuple[str, ...] = (),
        tie_symmetric_sequences: bool = False,
        generator: torch.Generator | None = None,
    ) -> BoltzGenIFoldOutput:
        r"""Design residue tokens on a fixed backbone (canonical expert operation).

        Thin public wrapper over the parity-passing ``_design`` autoregressive body;
        the numerics are unchanged. Returns per-token ``logits`` and sampled ``tokens``.

        Args:
            batch: Typed inverse-fold input from ``BoltzGenIFoldProcessor``.
            temperature: Optional categorical sampling temperature. ``None`` selects argmax.
            inverse_fold_restriction: Canonical amino acids to exclude from this call.
            tie_symmetric_sequences: Whether symmetric copies share one sample.
            generator: Optional PyTorch random generator for decode order and sampling.

        Returns:
            The per-token sequence logits and sampled tokens.
        """

        return self._design(
            batch,
            temperature=temperature,
            inverse_fold_restriction=inverse_fold_restriction,
            tie_symmetric_sequences=tie_symmetric_sequences,
            generator=generator,
        )

    @torch.no_grad()
    def _design(
        self,
        batch: BoltzGenIFoldInput,
        *,
        temperature: float | None = None,
        inverse_fold_restriction: tuple[str, ...] = (),
        tie_symmetric_sequences: bool = False,
        generator: torch.Generator | None = None,
    ) -> BoltzGenIFoldOutput:
        r"""
        Autoregressively design residue tokens from the conditioned structure.

        Args:
            batch: Typed model input.
            temperature: Optional categorical sampling temperature. ``None`` selects argmax decoding.
            inverse_fold_restriction: Canonical amino acids to exclude from this design call.
            tie_symmetric_sequences: Whether matching positions across symmetric copies share one sample.
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
        logits, residue_type = self.decoder.sample(
            node,
            edge,
            edge_index,
            valid_mask,
            batch,
            temperature=temperature,
            inverse_fold_restriction=inverse_fold_restriction,
            tie_symmetric_sequences=tie_symmetric_sequences,
            generator=generator,
        )
        return BoltzGenIFoldOutput(
            logits=logits,
            tokens=residue_type.argmax(dim=-1),
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
        operator_backend: Operator backend selector (``"auto"``, ``"oops"``, or ``"eager"``).
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
        if add_mol_type_feat:
            self.molecule_type_embedding = nn.Embedding(num_molecule_types, single_size)
        if add_binding_specification:
            self.binding_specification_embedding = nn.Embedding(num_binding_types, single_size)
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

        if self.method_embedding is not None:
            single = single + self.method_embedding(cast(Tensor, method_feature))
        if self.modified_embedding is not None:
            single = single + self.modified_embedding(cast(Tensor, modified))
        if self.cyclic_projection is not None:
            single = single + self.cyclic_projection(
                cast(Tensor, cyclic_period).clamp(max=1.0).unsqueeze(-1).to(single.dtype)
            )
        if self.molecule_type_embedding is not None:
            single = single + self.molecule_type_embedding(cast(Tensor, chain_type))
        if self.binding_specification_embedding is not None:
            single = single + self.binding_specification_embedding(cast(Tensor, binding_type))
        if self.design_mask_embedding is not None:
            single = single + self.design_mask_embedding(cast(Tensor, design_mask).int())
        if self.ss_specification_embedding is not None:
            single = single + self.ss_specification_embedding(cast(Tensor, ss_type))
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
        self.atom_feature_projection = nn.Linear(atom_feature_size, atom_hidden_size)
        self.distance_vector_projection = nn.Linear(3, atom_pair_hidden_size, bias=False)
        self.inverse_distance_projection = nn.Linear(1, atom_pair_hidden_size, bias=False)
        self.distance_mask_projection = nn.Linear(1, atom_pair_hidden_size, bias=False)
        self.single_trunk_projection: nn.Sequential | None = None
        self.pair_trunk_projection: nn.Sequential | None = None
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
        single_trunk_projection = self.single_trunk_projection
        if single_trunk_projection is not None:
            token_index = cast(Tensor, atom_to_token)
            single_proj = single_trunk_projection(cast(Tensor, single_trunk))
            atom_single = atom_single + single_proj.gather(
                1, token_index.unsqueeze(-1).expand(-1, -1, single_proj.shape[-1])
            )
            pair_from_trunk = cast(nn.Sequential, self.pair_trunk_projection)(cast(Tensor, pair_trunk))
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
        operator_backend: Operator backend selector (``"auto"``, ``"oops"``, or ``"eager"``).
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
        self.noisy_position_projection: nn.Linear | None = None
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
        noisy_position_projection = self.noisy_position_projection
        if noisy_position_projection is not None:
            query = query.repeat_interleave(multiplicity, 0) + noisy_position_projection(cast(Tensor, noisy_positions))
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
        if noisy_position_projection is not None:
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
        operator_backend: Operator backend selector (``"auto"``, ``"oops"``, or ``"eager"``).
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
    """

    def __init__(self, pair_size: int = 128) -> None:
        super().__init__()
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
        operator_backend: Operator backend selector (``"auto"``, ``"oops"``, or ``"eager"``).
    """

    def __init__(
        self,
        pair_size: int = 128,
        expansion: int = 4,
        dropout: float = 0.25,
        operator_backend: str = "auto",
    ) -> None:
        super().__init__()
        self.triangular = MiniTriangularUpdate(pair_size)
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
        operator_backend: Operator backend selector (``"auto"``, ``"oops"``, or ``"eager"``).
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
        self.triangular = MiniTriangularUpdate(pair_size)
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
        operator_backend: Operator backend selector (``"auto"``, ``"oops"``, or ``"eager"``).
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
        operator_backend: Operator backend selector (``"auto"``, ``"oops"``, or ``"eager"``).
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
        operator_backend: Operator backend selector (``"auto"``, ``"oops"``, or ``"eager"``).
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
    ) -> dict[str, Tensor]:
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
        operator_backend: Operator backend selector (``"auto"``, ``"oops"``, or ``"eager"``).
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
        conditioning: dict[str, Tensor],
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
            conditioning["atom_enc_bias"].repeat_interleave(multiplicity, 0),
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
            conditioning["token_trans_bias"].repeat_interleave(multiplicity, 0),
            token_mask.repeat_interleave(multiplicity, 0).float(),
        )
        token_repr = self.token_norm(token_repr)

        return self.atom_attention_decoder(
            token_repr,
            atom_skip,
            conditioning_skip,
            conditioning["atom_dec_bias"].repeat_interleave(multiplicity, 0),
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
        p_mean: float = -1.2,
        p_std: float = 1.5,
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
        self.p_mean = p_mean
        self.p_std = p_std
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
        sigma = torch.as_tensor(sigma, device=noised_coords.device, dtype=noised_coords.dtype).expand(batch)
        padded = sigma.reshape(batch, 1, 1)
        update = self.score_model(
            noisy_positions=self.preconditioner.c_in(padded) * noised_coords,
            noise_level=self.preconditioner.c_noise(sigma),
            multiplicity=multiplicity,
            **conditioning,
        )
        return self.preconditioner.c_skip(padded) * noised_coords + self.preconditioner.c_out(padded) * update

    def noise_distribution(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        r"""
        Draw per-sample EDM training noise levels from the ``sigma_data``-scaled log-normal.

        Mirrors the upstream BoltzGen ``AtomDiffusion.noise_distribution``:
        ``sigma = sigma_data * exp(P_mean + P_std * z)`` with ``z ~ N(0, 1)``. Returned as a tensor (never
        ``.item()``-ed) so the value feeds tensor math end to end.

        Args:
            batch_size: Number of noise levels to draw.
            device: Device for the returned tensor.
            dtype: Floating dtype for the returned tensor.
            generator: Optional PyTorch random generator.

        Returns:
            Noise levels ``(batch_size,)``.
        """

        noise = torch.randn((batch_size,), device=device, dtype=dtype, generator=generator)
        return self.sigma_data * (self.p_mean + self.p_std * noise).exp()

    def loss_weight(self, sigma: Tensor) -> Tensor:
        r"""
        EDM denoising-loss weighting ``(sigma^2 + sigma_data^2) / (sigma * sigma_data)^2`` for each noise level.

        Args:
            sigma: Per-sample noise levels.

        Returns:
            Per-sample loss weights of the same shape as ``sigma``.
        """

        return (sigma**2 + self.sigma_data**2) / ((sigma * self.sigma_data) ** 2)

    def sample_schedule_karras(self, num_sampling_steps: int | None = None, *, device: torch.device) -> Tensor:
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
        transform, then applies the same Karras ``rho`` curve as the ``karras`` schedule.
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
            return self.sample_schedule_karras(num_sampling_steps, device=device)
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
        # The EDM *step math* matches BoltzGenCoordinateEdmSampler (the shared sampler subclass that injects
        # BoltzGen's faithful ``noise_scale**2 * sqrt(sigma_hat**2 - sigma_current**2)`` churn factor) plus the
        # opt-in ``alignment_reverse_diff`` mid-step align. The loop stays in-model for the same three
        # RNG-load-bearing framing divergences as Boltz-2 -- here with the *dilated* Karras schedule
        # (sample_schedule_dilated) rather than EdmScheduler, the right-multiply (R^T @ x) augmentation
        # convention, and torch.randn draws the structure_diffusion_parity harness patches. Genuine structural
        # divergence in the sampler framing, so it is left separate (Frankenstein principle).
        # The fixed Python loop controls denoiser invocation; schedule values stay as tensors so sampling does not
        # synchronize them to the host on every step.
        for index in range(len(sigmas) - 1):
            sigma_current, sigma_next, gamma = sigmas[index], sigmas[index + 1], gammas[index + 1]
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
            noisy = coords + noise_scale_value * torch.sqrt(torch.clamp(noise_var, min=0.0)) * randn()
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


def _center_random_augmentation(
    coords: Tensor, atom_mask: Tensor, *, s_trans: float = 1.0, generator: torch.Generator | None = None
) -> Tensor:
    r"""
    Center coords on the masked centroid and apply a Haar-uniform rotation plus Gaussian translation.

    Mirrors the upstream BoltzGen training augmentation (``center_random_augmentation``, Algorithm 19): the
    coordinate-frame augmentation applied to the diffusion target before noising. Fully tensor-valued (no
    ``.item()``) so it stays inside the compiled training graph.

    Args:
        coords: Atom coords ``(B, N_atom, 3)``.
        atom_mask: Boolean atom validity mask ``(B, N_atom)``.
        s_trans: Standard deviation of the random per-structure translation.
        generator: Optional PyTorch random generator.

    Returns:
        Augmented coords ``(B, N_atom, 3)``.
    """

    mask = atom_mask.unsqueeze(-1).to(coords.dtype)
    centroid = (coords * mask).sum(dim=-2, keepdim=True) / mask.sum(dim=-2, keepdim=True).clamp(min=1.0)
    centered = coords - centroid
    rotation = uniform_random_rotation(coords.shape[0], device=coords.device, dtype=coords.dtype, generator=generator)
    rotated = torch.einsum("bmd,bds->bms", centered, rotation)
    translation = (
        torch.randn(coords.shape[0], 1, 3, device=coords.device, dtype=coords.dtype, generator=generator) * s_trans
    )
    return rotated + translation


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


def _gather_backbone4(
    token_to_bb4_atoms: Tensor,
    coords: Tensor,
    batch_size: int,
    num_tokens: int,
) -> Tensor:
    r"""Gather the four backbone-atom coordinates per token -> ``(B, N_token, 4, 3)``."""
    bb4_index = token_to_bb4_atoms.reshape(batch_size, num_tokens * 4)
    coords = coords.reshape(batch_size, -1, 3)
    backbone = coords.gather(1, bb4_index.unsqueeze(-1).expand(-1, -1, 3))
    return backbone.reshape(batch_size, num_tokens, 4, 3)


def _zero_initialize_boltzgen_updates(module: nn.Module) -> None:
    r"""
    Apply the BoltzGen zero-init scheme to every final-layer / gating / conditioning weight under ``module``.

    Zeroing the last projection of each residual sub-block (and biasing the gate projections negative) starts every
    block as a near-identity, matching the upstream BoltzGen initialization.

    Args:
        module: Root module to walk; every matching child is zero-initialized in place.
    """

    for child in module.modules():
        if isinstance(child, AtomEncoder):
            if child.single_trunk_projection is not None:
                nn.init.zeros_(child.single_trunk_projection[1].weight)
            if child.pair_trunk_projection is not None:
                nn.init.zeros_(child.pair_trunk_projection[1].weight)
            nn.init.zeros_(child.left_atom_projection[1].weight)
            nn.init.zeros_(child.right_atom_projection[1].weight)
            nn.init.zeros_(child.feed_forward[5].weight)
        elif isinstance(child, AtomAttentionEncoder) and child.noisy_position_projection is not None:
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
        elif isinstance(child, GatedMultiheadAttention):
            nn.init.zeros_(child.projection.output_projection.weight)
        elif isinstance(child, ConditionedTransformerLayer):
            nn.init.zeros_(child.output_gate_projection.weight)
            if child.output_gate_projection.bias is not None:
                nn.init.constant_(child.output_gate_projection.bias, -2.0)
        elif isinstance(child, (ConditionedTransition, ConditionedTransitionBlock)):
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


def _build_constraint_logit_mask(
    num_nodes: int,
    aa_constraint_mask: Tensor | None,
    inverse_fold_restriction: tuple[str, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    r"""
    Build the additive per-position canonical-amino-acid logit mask for constrained inverse folding.

    Combines a per-residue block mask (``aa_constraint_mask``, ``(num_nodes, 20)``, ``> 0`` = blocked) with a
    global block list (``inverse_fold_restriction``, three-letter CCD codes blocked everywhere). A position with no
    allowed amino acid raises instead of silently relaxing the explicit per-residue constraints.

    Args:
        num_nodes: Number of packed valid tokens.
        aa_constraint_mask: Optional per-residue block mask ``(num_nodes, 20)``; ``None`` = no per-residue blocks.
        inverse_fold_restriction: Canonical amino acids (three-letter codes) to block at every position.
        device: Device for the returned mask.
        dtype: Floating dtype for the returned additive mask.

    Returns:
        Additive logit bias ``(num_nodes, 20)``: ``0.0`` where allowed, ``-1e6`` where blocked.

    Raises:
        ValueError: If the combined restrictions block every identity at a position.
    """

    num_aa = NUM_CANONICAL_TOKENS
    if aa_constraint_mask is None:
        per_residue_blocked = torch.zeros(num_nodes, num_aa, dtype=torch.bool, device=device)
    else:
        per_residue_blocked = aa_constraint_mask > 0

    global_blocked = torch.zeros(num_aa, dtype=torch.bool, device=device)
    for res_type in inverse_fold_restriction:
        global_blocked[CANONICAL_TOKENS.index(res_type)] = True

    combined = per_residue_blocked | global_blocked.unsqueeze(0)
    all_blocked = combined.all(dim=1)
    if all_blocked.any():
        raise ValueError(
            f"inverse folding has no valid amino acids at positions {torch.where(all_blocked)[0].tolist()}"
        )
    return combined.to(dtype=dtype) * -1e6


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
        num_neighbors: Number of nearest neighbours per token.
        max_relative_position: Clamp range for the relative-position edge feature.
        dropout: Dropout probability inside the graph-attention layers.
        softmax_dropout: Probability of masking an attention logit before its destination-wise softmax.
        inverse_fold_noise: Coordinate-noise standard deviation used by the training graph.
    """

    def __init__(
        self,
        single_size: int = 384,
        node_size: int = 128,
        edge_size: int = 128,
        hidden_size: int = 128,
        num_layers: int = 3,
        num_attention_heads: int = 4,
        num_neighbors: int = 30,
        max_relative_position: int = 32,
        dropout: float = 0.1,
        softmax_dropout: float = 0.2,
        inverse_fold_noise: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_neighbors = num_neighbors
        self.max_relative_position = max_relative_position
        self.inverse_fold_noise = inverse_fold_noise
        # geo (4x4 backbone-atom distances x 16 RBF = 256)
        # + edge attrs (2*max_relative_position+2 + same_chain + same_residue)
        # + bond (token_bonds + one-hot bond type).
        self.edge_input_size = 256 + (2 * max_relative_position + 2) + 2 + 1 + (NUM_BOND_TYPES + 1)

        self.node_input_projection = nn.Linear(single_size, node_size)
        self.edge_input_projection = nn.Linear(self.edge_input_size, edge_size)
        self.layers = nn.ModuleList(
            GraphAttentionLayer(
                node_size=node_size,
                edge_size=edge_size,
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                dropout=dropout,
                softmax_dropout=softmax_dropout,
            )
            for _ in range(num_layers)
        )
        self.distance_smearing = GaussianSmearing(start=0.0, stop=20.0, num_gaussians=16)

    @torch.no_grad()
    def build_knn_graph(self, batch: BoltzGenIFoldInput) -> tuple[Tensor, Tensor, Tensor]:
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
        num_neighbors = min(self.num_neighbors, num_tokens)

        distances = torch.cdist(batch.center_coords, batch.center_coords)
        distances = distances.masked_fill(~valid_pair, float("inf"))
        src = torch.topk(distances, num_neighbors, largest=False).indices  # (B, N, num_neighbors)
        dst = torch.arange(num_tokens, device=src.device)[None, :, None].expand(batch_size, -1, num_neighbors)

        # token_bonds carries a trailing singleton (B, N, N, 1); drop it so the gather index (B, N, num_neighbors)
        # matches its rank (the BoltzGen reference does the same, asserting shape[-1] == 1). type_bonds is (B, N, N).
        token_bonds = torch.gather(batch.token_bonds[..., 0], 2, src)
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
    def edge_geometry(self, batch: BoltzGenIFoldInput, edge_index: Tensor, valid_mask: Tensor) -> Tensor:
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
        coords = batch.coords
        if self.training and self.inverse_fold_noise > 0:
            coords = coords + torch.randn_like(coords) * self.inverse_fold_noise
        backbone = _gather_backbone4(
            batch.token_to_bb4_atoms,
            coords,
            batch_size,
            num_tokens,
        )[valid_mask]
        distance = torch.norm(backbone[src, None] - backbone[dst, :, None], dim=-1).flatten(start_dim=1)
        return self.distance_smearing(distance)

    @torch.no_grad()
    def edge_attributes(self, batch: BoltzGenIFoldInput, edge_index: Tensor, valid_mask: Tensor) -> Tensor:
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
        batch: BoltzGenIFoldInput,
        *,
        single_inputs: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        r"""
        Encode the structure into a packed kNN token graph with attention-refined node and edge states.

        Builds the kNN graph, featurizes edges (backbone-distance RBF + relative-position/bond attributes),
        projects node features (from ``single_inputs`` when the input embedder is enabled, else ``node_features``),
        and runs the graph-attention layers.

        Args:
            batch: Typed model input.
            single_inputs: Token single embedding ``(B, N_token, single_size)``.

        Returns:
            Tuple of ``edge_index`` ``(2, E)`` into packed valid tokens, ``valid_mask`` ``(B, N_token)``, packed
            node states ``(N_valid, node_size)``, and packed edge states ``(E, edge_size)``.

        """

        edge_index, valid_mask, bond = self.build_knn_graph(batch)
        geometry = self.edge_geometry(batch, edge_index, valid_mask)
        edge_attr = self.edge_attributes(batch, edge_index, valid_mask)

        node = self.node_input_projection(single_inputs[valid_mask])
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
        softmax_dropout: Probability of masking an attention logit before its destination-wise softmax.
    """

    def __init__(
        self,
        node_size: int = 128,
        edge_size: int = 128,
        hidden_size: int = 128,
        num_layers: int = 3,
        num_attention_heads: int = 4,
        vocab_size: int = NUM_RESIDUE_TYPES,
        dropout: float = 0.1,
        softmax_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        # Each layer's edge state is cat[encoder_edge, neighbour_node + residue_rep] = edge_size + node_size.
        decoder_edge_size = edge_size + node_size
        self.layers = nn.ModuleList(
            GraphAttentionLayerNoEdge(
                node_size=node_size,
                edge_size=decoder_edge_size,
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                dropout=dropout,
                softmax_dropout=softmax_dropout,
            )
            for _ in range(num_layers)
        )
        self.residue_type_projection = nn.Linear(vocab_size, node_size)
        self.residue_type_head = nn.Linear(node_size, vocab_size, bias=False)

    def forward(
        self, node: Tensor, edge: Tensor, edge_index: Tensor, valid_mask: Tensor, batch: BoltzGenIFoldInput
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
            rand = torch.rand(node.shape[0], device=node.device)
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
        batch: BoltzGenIFoldInput,
        *,
        temperature: float | None = None,
        inverse_fold_restriction: tuple[str, ...] = (),
        tie_symmetric_sequences: bool = False,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        r"""
        Autoregressively decode residue types one position at a time, in a random order.

        Each position is decoded from its (already-decoded) neighbours: the edge state carries
        ``neighbour_node + residue_rep(decoded_neighbour)``, the decoder layers run via their single-node
        ``sample`` path, and the canonical-amino-acid logits are argmax'd (or temperature-sampled). Non-design
        positions are fixed to ``res_type_clone``.

        Faithful to the upstream boltzgen ``InverseFoldingDecoder.sample``, including per-residue amino-acid
        constraints (``batch.aa_constraint_mask`` + call-time ``inverse_fold_restriction``, combined into an additive
        canonical-logit mask via ``_build_constraint_logit_mask``) and, when ``tie_symmetric_sequences`` is set,
        homomer-symmetry tying (logits are averaged across a symmetric group, sampled once, and applied to every
        copy). Input-side masks (``aa_constraint_mask``, ``symmetric_group``) must be supplied by the caller until
        the canonical Processor exists.

        Args:
            node: Packed encoder node states of shape `[num_valid, node_size]`.
            edge: Packed encoder edge states of shape `[E, edge_size]`.
            edge_index: Edge endpoints of shape `[2, E]` indexing into the packed valid tokens.
            valid_mask: Boolean token validity mask of shape `[B, num_tokens]`.
            batch: The typed model input, supplying ``design_mask``, ground-truth ``residue_type``, and the
                per-residue / symmetry constraint masks.
            temperature: Optional softmax temperature for stochastic sampling; ``None`` (or 0) argmaxes instead.
            inverse_fold_restriction: Canonical amino acids to block in this sampling call.
            tie_symmetric_sequences: Whether matching positions across symmetric copies share one sample.
            generator: Optional random generator seeding the decode order and any temperature sampling.

        Returns:
            The decoded logits and one-hot residue types on the dense token grid.
        """

        src, dst = edge_index[0], edge_index[1]
        num_nodes = node.shape[0]
        design_mask = cast(Tensor, batch.design_mask).bool()[valid_mask]
        decoded = node.new_zeros(num_nodes, self.vocab_size)
        logits = node.new_zeros(num_nodes, self.vocab_size)

        constraint_mask = None if batch.aa_constraint_mask is None else batch.aa_constraint_mask[valid_mask]
        per_residue_mask = _build_constraint_logit_mask(
            num_nodes=num_nodes,
            aa_constraint_mask=constraint_mask,
            inverse_fold_restriction=inverse_fold_restriction,
            device=node.device,
            dtype=node.dtype,
        )

        order = torch.randperm(num_nodes, device=node.device, generator=generator).tolist()
        if (~design_mask).any():
            known = batch.residue_type[valid_mask][~design_mask].to(decoded.dtype)
            decoded[~design_mask] = known
            logits[~design_mask] = known
            fixed = set(torch.where(~design_mask)[0].tolist())
            order = [position for position in order if position not in fixed]

        symmetry_groups, position_to_group = (
            self._build_symmetric_groups(batch, valid_mask, design_mask)
            if tie_symmetric_sequences and batch.symmetric_group is not None
            else ({}, {})
        )
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
            aggregated_logits = cast(Tensor, aggregated_logits) / len(positions)

            canonical = (
                aggregated_logits[:, CANONICALS_OFFSET : CANONICALS_OFFSET + NUM_CANONICAL_TOKENS]
                + per_residue_mask[position : position + 1]
            )
            if temperature is None or temperature == 0:
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
        return dense_logits, dense_res_type

    def _build_symmetric_groups(
        self, batch: BoltzGenIFoldInput, valid_mask: Tensor, design_mask: Tensor
    ) -> tuple[dict[int, list[int]], dict[int, int]]:
        r"""
        Group designable packed-token positions that must share a residue type under homomer symmetry.

        Returns ``(groups, position_to_group)`` where ``groups`` maps a group id to its member positions (only
        groups with more than one member are kept) and ``position_to_group`` maps a position to its group id.
        Positions are tied when they share the same ``(symmetric_group, residue_index)`` key.
        """

        metadata = torch.stack(
            [design_mask, cast(Tensor, batch.symmetric_group)[valid_mask], batch.residue_index[valid_mask]], dim=-1
        ).tolist()
        key_to_positions: dict[tuple[int, int], list[int]] = {}
        for position, (designable, group, residue_index) in enumerate(metadata):
            if not designable:
                continue
            if group <= 0:  # 0 = no symmetry group.
                continue
            key = (group, residue_index)
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
        operator_backend: Operator backend selector (``"auto"``, ``"oops"``, or ``"eager"``).
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
        operator_backend: Operator backend selector (``"auto"``, ``"oops"``, or ``"eager"``).
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
            batch: Typed model input supplying the template features.
            pair_mask: Optional boolean pair mask ``(B, N_token, N_token)``.

        Returns:
            Additive pair update ``(B, N_token, N_token, pair_size)``.

        """

        res_type = cast(Tensor, batch.template_restype)
        frame_mask = cast(Tensor, batch.template_mask_frame)
        frame_rotation = cast(Tensor, batch.template_frame_rot)
        frame_translation = cast(Tensor, batch.template_frame_t)
        cb_coords = cast(Tensor, batch.template_cb)
        ca_coords = cast(Tensor, batch.template_ca)
        cb_mask = cast(Tensor, batch.template_mask_cb)
        template_mask = cast(Tensor, batch.template_mask)
        visibility = cast(Tensor, batch.visibility_ids)
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
        backbone = _gather_backbone4(
            batch.token_to_bb4_atoms,
            batch.coords,
            batch_size,
            num_tokens,
        ).permute(0, 2, 1, 3)
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
        softmax_dropout: Probability of masking an attention logit before its destination-wise softmax.
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
        self.num_attention_heads = num_attention_heads
        self.softmax_dropout = softmax_dropout
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
        if self.training and self.softmax_dropout > 0:
            weight = _softmax_dropout(weight, dst, node.shape[0], self.softmax_dropout)
        weight = scatter_softmax(weight, dst, dim=0, dim_size=node.shape[0])
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
        softmax_dropout: Probability of masking an attention logit before its destination-wise softmax.
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
        self.num_attention_heads = num_attention_heads
        self.softmax_dropout = softmax_dropout
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
        if self.training and self.softmax_dropout > 0:
            weight = _softmax_dropout(weight, dst, node.shape[0], self.softmax_dropout)
        weight = scatter_softmax(weight, dst, dim=0, dim_size=node.shape[0])
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
        if self.training and self.softmax_dropout > 0:
            weight = _softmax_dropout(weight, dst, 1, self.softmax_dropout)
        weight = scatter_softmax(weight, dst, dim=0, dim_size=1)
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
        # Match the BoltzGen reference: derive the RBF width from the float32 ``offset`` spacing, not the float64
        # analytic spacing (see GaussianSmearing above) so the distance features are bit-faithful to upstream.
        # Compute the spacing on a real CPU tensor so construction under ``torch.device("meta")`` (the
        # converter / hub load mode) does not call ``.item()`` on an unmaterialized meta tensor.
        spacing = torch.linspace(start, stop, num_gaussians, device="cpu")
        self.coeff = -0.5 / (spacing[1] - spacing[0]).item() ** 2
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
        # Derive the RBF width from the float32 ``offset`` spacing (matching the BoltzGen reference) rather than
        # the float64 analytic spacing: for non-float32-representable spacings (e.g. 20/15) the two differ by
        # ~6e-8 relative, and that discrepancy amplifies through the 6-layer encoder to ~1e-3 in the logits.
        # Compute the spacing on a real CPU tensor so construction under ``torch.device("meta")`` (the
        # converter / hub load mode) does not call ``.item()`` on an unmaterialized meta tensor.
        spacing = torch.linspace(start, stop, num_gaussians, device="cpu")
        self.coeff = -0.5 / (spacing[1] - spacing[0]).item() ** 2
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


def _softmax_dropout(logits: Tensor, index: Tensor, num_groups: int, probability: float) -> Tensor:
    r"""Apply the released rejection-sampled attention-logit dropout."""

    for _ in range(10):
        dropped = torch.rand_like(logits) < probability
        kept = scatter_sum((~dropped).to(torch.int64), index, dim_size=num_groups)
        if kept.all():
            return logits.masked_fill(dropped, -torch.inf)
    raise RuntimeError("attention dropout failed to keep an incoming edge for every destination and head")


def scatter_sum(src: Tensor, index: Tensor, dim_size: int, dim: int = 0) -> Tensor:
    r"""
    Sum ``src`` rows into buckets given by ``index`` (native reimplementation of ``torch_scatter.scatter_sum``).

    Args:
        src: Source tensor whose slices along ``dim`` are summed per bucket.
        index: Destination bucket index for each slice along ``dim``.
        dim: Dimension along which to scatter.
        dim_size: Number of output buckets.

    Returns:
        Tensor with ``dim_size`` buckets along ``dim``, each the sum of its assigned ``src`` slices.

    Examples:
        >>> import torch
        >>> src = torch.tensor([[1.0], [2.0], [3.0]])
        >>> scatter_sum(src, torch.tensor([0, 0, 1]), dim_size=2).tolist()
        [[3.0], [3.0]]
    """

    shape = list(src.shape)
    shape[dim] = dim_size
    out = torch.zeros(shape, dtype=src.dtype, device=src.device)
    expanded = index
    for _ in range(src.dim() - index.dim()):
        expanded = expanded.unsqueeze(-1)
    out.scatter_add_(dim, expanded.expand_as(src), src)
    return out


def scatter_softmax(src: Tensor, index: Tensor, dim_size: int, dim: int = 0) -> Tensor:
    r"""
    Softmax over rows sharing a destination ``index`` (per-group / segmented softmax).

    Args:
        src: Source tensor whose slices along ``dim`` are softmaxed within each group.
        index: Group index for each slice along ``dim``.
        dim: Dimension along which to group.
        dim_size: Number of groups.

    Returns:
        Tensor the same shape as ``src`` whose slices sum to one within each group.

    Examples:
        >>> import torch
        >>> weight = scatter_softmax(torch.tensor([1.0, 1.0, 5.0]), torch.tensor([0, 0, 1]), dim_size=2)
        >>> weight.round(decimals=4).tolist()
        [0.5, 0.5, 1.0]
    """

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


@dataclass(kw_only=True)
class BoltzGenInput:
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
        token_bonds: Token-level covalent adjacency ``(B, N, N, 1)`` (trailing singleton per the released featurizer).
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
    # Additive, backward-compatible: the canonical identity layout populated by
    # ``BoltzGenProcessor`` and read by ``BoltzGenPipeline`` when it decodes rows back into
    # molecular identities. Ignored by the model graph. See ``BoltzGenLayout`` below.
    layout: BoltzGenLayout | None = None


@dataclass(kw_only=True)
class BoltzGenIFoldInput:
    r"""Input tensors consumed by the distinct inverse-folding architecture."""

    token_mask: Tensor
    residue_index: Tensor
    chain_index: Tensor
    atom_features: Tensor
    atom_to_token: Tensor
    atom_mask: Tensor
    reference_positions: Tensor
    reference_space_id: Tensor
    residue_type: Tensor
    msa_profile: Tensor
    deletion_mean: Tensor
    token_bonds: Tensor
    type_bonds: Tensor
    chain_type: Tensor
    center_coords: Tensor
    token_to_bb4_atoms: Tensor
    coords: Tensor
    method_feature: Tensor | None = None
    modified: Tensor | None = None
    cyclic_period: Tensor | None = None
    design_mask: Tensor | None = None
    aa_constraint_mask: Tensor | None = None
    symmetric_group: Tensor | None = None
    # Additive, backward-compatible: the canonical identity layout populated by
    # ``BoltzGenIFoldProcessor`` and read by ``BoltzGenPipeline``. Ignored by the model
    # graph. See ``BoltzGenIFoldLayout`` below.
    layout: BoltzGenIFoldLayout | None = None


# ---------------------------------------------------------------------------
# Canonical identity layouts (folded in from the former ``contract.py``).
#
# These are the non-tensor leaves the design/inverse-fold Processors populate and
# the design Pipeline reads when it decodes tensor rows back into canonical molecular
# identities; the model graph above deliberately ignores them.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BoltzGenLayout:
    r"""Canonical molecular identities aligned to one ``BoltzGenModel`` tensor layout.

    A structure-design assembly may be partly or wholly generated (``total_len``
    minted rows), so ``output_residues[i]`` names token row ``i`` as either an
    existing canonical [`ComponentReference`][ofoldx.data.molecular.ComponentReference]
    or one symbolic generated residue
    ([`SymbolicResidueSite`][ofoldx.models.boltzgen.model.SymbolicResidueSite]).
    The symbolic rows follow the RFdiffusion pattern (§3.7 owner-gated): the shared
    PlacementSpec / placement_map core stays unresolved, so these are family-local
    provisional identities, not a frozen public placement contract.

    Attributes:
        system: The canonical molecular system the tensor batch was lowered from
            (``None`` for a fully-generated assembly with no input system).
        atom_sites: Model atom order aligned to the atom axis (heavy atoms, sans padding).
        token_sites: Model token order aligned to the token axis; polymer tokens carry
            ``(instance_id, monomer_index)`` and ligand-atom tokens carry the single ``AtomSite``.
        output_residues: Per-token canonical identity (existing residue or symbolic generated residue).
        output_instance_ids: Per-token owning instance id (explicit because generated rows have no
            canonical instance until the placement core resolves).
        placement_map: Frozen §3.7 realized output surface for a whole-assembly ``total_len``
            generation (``declaration_group -> placement_id -> ordered ComponentReference
            occurrences``). Empty for an existing-input design that generates no topology.
        instance_of: The derived ``placement_id -> declaration_group`` index (built once from the
            PlacementSpec), used with ``ofoldx.data.placement.resolve`` to resolve a ``SymbolicResidueSite``.
    """

    system: MolecularSystem | None
    atom_sites: tuple[AtomSite, ...]
    token_sites: tuple[AtomSite | tuple[str, int], ...]
    output_residues: tuple[ComponentReference | SymbolicResidueSite, ...]
    output_instance_ids: tuple[str, ...]
    placement_map: PlacementMap = field(default_factory=dict)
    instance_of: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class BoltzGenIFoldLayout:
    r"""Canonical molecular identities aligned to one ``BoltzGenIFoldModel`` tensor layout.

    Inverse folding never generates topology: every row maps to an existing input
    backbone residue, so ``output_residues`` is a tuple of canonical
    ``ComponentReference`` values only.

    Attributes:
        system: The canonical molecular system the fixed backbone was lowered from.
        atom_sites: Model atom order aligned to the atom axis (heavy atoms, sans padding).
        token_sites: Model token order aligned to the token axis.
        output_residues: Per-token canonical residue identity of the fixed backbone.
        output_instance_ids: Per-token owning instance id.
    """

    system: MolecularSystem
    atom_sites: tuple[AtomSite, ...]
    token_sites: tuple[AtomSite | tuple[str, int], ...]
    output_residues: tuple[ComponentReference, ...]
    output_instance_ids: tuple[str, ...]


@dataclass(frozen=True)
class SymbolicResidueSite:
    r"""A pre-realization reference to one generated residue by segment and offset.

    §3.7 owner-gated family-local placeholder (RFdiffusion spelling). Used only to
    name rows of a fully-generated ``total_len`` assembly before the shared
    placement core resolves; NOT a frozen public shape.
    """

    segment_id: str
    offset: int


@dataclass(frozen=True)
class BoltzGenPlacements:
    r"""One BoltzGen whole-assembly generation-topology request (frozen §3.7 core, X1-B).

    ``placements`` is the frozen core :data:`~ofoldx.data.placement.PlacementSpec` mapping
    ``declaration_group -> placement_id -> value``; the inner ``placement_id`` is globally
    unique across the whole mapping and is the SOLE pre-realization reference target
    (validated at construction). Unlike ODesign — which declares one group **per output
    chain** — a BoltzGen assembly is ONE declaration group whose inner placements are its
    chains, so the outer group key is a pure whole-assembly grouping label that no reference
    names (frozen §3.7 "BoltzGen whole-assembly handling (X1-B)").

    Per-chain lengths are drawn at realization from the family-owned whole-assembly
    ``total_len`` budget (the shared core carries no total-length field), so a ranged
    :class:`~ofoldx.data.placement.GeneratedSegment` (``min_length != max_length``) is resolved
    against ``total_len`` rather than fixed on the core. ``total_len is None`` realizes every
    generated placement at its ``max_length``.
    """

    placements: PlacementSpec

    def __post_init__(self) -> None:
        # Frozen §3.7: a placement_id must be globally unique across the whole PlacementSpec; this is
        # the construction-time uniqueness check (raises ValueError on any duplicate).
        validate_placement(self.placements)


@dataclass(frozen=True)
class BoltzGenDesignMask:
    r"""Positions to (re)design → ``design_mask`` on the model input.

    ``positions`` are the designable residues (upstream ``design`` / complement of
    ``not_design``); every named token is marked designable.
    """

    positions: tuple[ComponentReference, ...]


@dataclass(frozen=True)
class BoltzGenBinding:
    r"""Binding-specification conditioning → ``binding_type`` ids.

    ``binding`` is the frozen §3.7.4 shared :class:`~ofoldx.data.conditions.Hotspot` — the
    distance-free per-residue POSITIVE binding label (owner verdict D3: BoltzGen ``binding`` is
    the canonical hotspot analog). ``not_binding`` is the BoltzGen-owned NEGATIVE signed label
    (embedding id 2) that has no analog in the other families, so — like ``kind`` on
    :class:`BoltzGenContact` — it stays on this family envelope rather than on the shared
    ``Hotspot`` core. Realized as the released binding-specification embedding ids
    (0 = unspecified, 1 = binding, 2 = not-binding).
    """

    binding: Hotspot = Hotspot(())
    not_binding: tuple[ComponentReference, ...] = ()


@dataclass(frozen=True)
class BoltzGenSecondaryStructure:
    r"""Per-residue secondary-structure conditioning → ``ss_type`` ids.

    Each assignment pairs a residue with one of the released labels ``"H"`` (helix),
    ``"E"`` (strand), or ``"L"`` (loop); unlisted residues take the masked/unspecified
    class.
    """

    assignments: tuple[tuple[ComponentReference, Literal["H", "E", "L"]], ...]


@dataclass(frozen=True)
class BoltzGenStructureGroup:
    r"""One structure group → per-token ``structure_group`` → ``token_distance_mask``.

    §3.7 family-owned (not fixed coordinates): geometry is exposed only *within* the
    same **nonzero** group and distinct groups move independently. Realized through the
    per-token ``structure_group`` id and its pairwise ``token_distance_mask``, never by
    writing absolute coordinates; ``reference_space_id`` stays per-residue (the released
    featurizer does NOT regroup it). ``group_id == 0`` is the free (ungrouped) class.
    Ratified/frozen under the §3.7.2 family-owned spatial-conditioning contract,
    validated exact-equal to the released featurizer (``process_token_features``,
    2026-07-19); see the Processor lowering (``_structure_group_ids`` /
    ``_token_distance_mask``).
    """

    members: tuple[ComponentReference, ...]
    group_id: int


@dataclass(frozen=True)
class BoltzGenResidueConstraint:
    r"""Per-residue amino-acid identity constraint → ``aa_constraint_mask`` (inverse folding).

    ``allowed`` is the set of canonical three-letter residue codes permitted at
    ``position`` (upstream ``residue_constraints``); consumed by
    ``BoltzGenIFoldModel`` as an ``aa_constraint_mask`` gate.
    """

    position: ComponentReference
    allowed: tuple[str, ...]


@dataclass(frozen=True)
class BoltzGenContact:
    r"""A pairwise contact declaration → ``contact_conditioning`` / ``contact_threshold``.

    ``contact`` is the frozen §3.7.4 shared two-sided geometric
    :class:`~ofoldx.data.conditions.Contact`; its ``side_a`` / ``side_b`` name the two
    contacting endpoint sets and its ``cutoff_angstrom`` is the contact distance cutoff used by
    the released contact-conditioning encoder (BoltzGen writes it as the ``contact_threshold``
    matrix). ``kind`` is the BoltzGen-owned released directed contact CLASS
    (``pocket>binder`` / ``binder>pocket`` / ``contact``); directionality is a family-owned
    extension that stays OFF the symmetric shared core and rides on this envelope.
    """

    contact: Contact
    kind: Literal["contact", "pocket>binder", "binder>pocket"] = "contact"


@dataclass(frozen=True)
class BoltzGenCovalentBond:
    r"""A requested covalent link → derived ``cyclic_period`` (envelope over ``SymbolicBond``).

    ``bond`` is the frozen §3.7.4 shared :class:`~ofoldx.data.bonds.SymbolicBond`; its endpoints
    may be EXISTING input atoms (:class:`~ofoldx.data.molecular.AtomSite`) or GENERATED atoms
    (:class:`~ofoldx.data.placement.SymbolicAtomSite`) resolved through the frozen placement
    core. ``BoltzGenProcessor`` lowers it by resolving the bond to a realized
    :class:`~ofoldx.data.molecular.SystemBond` and injecting it into the lowered system, so the
    released head-to-tail detector derives ``cyclic_period`` exactly as it does for a bond the
    input system already carried (this family declares NO physical topology beyond that — a
    non-terminal link simply surfaces as a token bond). ``intent`` is the BoltzGen-owned
    hard/soft label kept OFF the shared bond core, per owner verdict (no ``intent`` on
    ``SymbolicBond``).
    """

    bond: SymbolicBond
    intent: Literal["hard", "soft"] = "hard"


@dataclass(frozen=True)
class BoltzGenSymmetry:
    r"""A symmetric group of equal-length residue units → ``symmetry_index`` / ``symmetric_group``.

    Each unit in ``units`` is one symmetric copy; matching offsets across units are
    tied. Consumed by structure design as ``symmetry_index`` and by inverse folding
    as ``symmetric_group`` (with ``tie_symmetric_sequences``).
    """

    units: tuple[tuple[ComponentReference, ...], ...]


# The ``conditions=`` parameter type on ``BoltzGenProcessor`` (structure design). The
# inverse-fold Processor consumes ``BoltzGenResidueConstraint`` / ``BoltzGenSymmetry``
# through their own dedicated keyword arguments, not this union. ``BoltzGenPlacements`` carries
# the frozen §3.7 whole-assembly generation topology (X1-B) alongside the spatial records.
BoltzGenSchema = (
    BoltzGenPlacements
    | BoltzGenDesignMask
    | BoltzGenBinding
    | BoltzGenSecondaryStructure
    | BoltzGenStructureGroup
    | BoltzGenContact
    | BoltzGenCovalentBond
    | BoltzGenSymmetry
)


def symbolic_residue_aliases(
    references: tuple[ComponentReference | SymbolicResidueSite, ...],
) -> dict[SymbolicResidueSite, SymbolicResidueSite]:
    r"""Map canonical negative segment offsets to concrete positive trajectory identities.

    Mirrors ``rfdiffusion.model.symbolic_residue_aliases`` so a fully-generated
    ``total_len`` assembly can address its generated rows by segment/offset before
    the shared placement core resolves.
    """

    segments: dict[str, list[SymbolicResidueSite]] = {}
    for reference in references:
        if type(reference) is SymbolicResidueSite:
            segments.setdefault(reference.segment_id, []).append(reference)
    return {
        SymbolicResidueSite(segment_id, index - len(rows)): reference
        for segment_id, rows in segments.items()
        for index, reference in enumerate(rows)
    }


@dataclass(eq=False)
class BoltzGenDesignCandidate:
    r"""One designed (and optionally refolded) complex and its aligned scientific values.

    Each candidate owns the complete ``Structure`` of the designed assembly. The
    assembly length varies across candidates, which is the coordinate-producing,
    variable-length property that forbids reuse of the frozen fixed-length
    ``DesignCandidate``.

    Attributes:
        structure: The designed complex (post inverse folding, optionally refolded by boltz2).
        scores: Whole-candidate canonical [`Score`][ofoldx.results.Score] values keyed by name.
        metrics: Axis-aligned [`Metric`][ofoldx.results.Metric] values (e.g. per-residue NLL / pLDDT).
        pair_metrics: Two-axis [`PairMetric`][ofoldx.results.PairMetric] values (e.g. interaction PAE).
    """

    structure: Structure
    scores: dict[str, Score] = field(default_factory=dict)
    metrics: dict[str, Metric] = field(default_factory=dict)
    pair_metrics: dict[str, PairMetric] = field(default_factory=dict)


@dataclass(eq=False)
class BoltzGenDesignResult:
    r"""One generation-ordered collection of coordinate-producing BoltzGen designs.

    Tuple position is candidate identity for one generation context; there is no
    auxiliary index field, and no input/context/provenance field is carried
    (matching ``ESMFold2DesignResult``). The composite upstream ranking key is a
    Pipeline-level argument, not part of this Result: the Result exposes the
    primitive Scores and ``best(score)`` so a caller ranks by any single canonical
    Score.
    """

    candidates: tuple[BoltzGenDesignCandidate, ...]

    def best(self, score: str) -> BoltzGenDesignCandidate:
        r"""Return the earliest best candidate under one explicit score.

        Mirrors ``FoldResult.best`` / ``ESMFold2DesignResult.best`` exactly: the
        score's ``preferred`` direction must be consistently ``"higher"`` or
        ``"lower"``; unit and scope must match across every candidate; a NaN value or
        an undefined direction raises at ranking time; infinity is rankable; and the
        earliest candidate wins a tie.
        """

        scores = tuple(candidate.scores[score] for candidate in self.candidates)
        directions = {value.preferred for value in scores}
        if directions not in ({"higher"}, {"lower"}):
            raise ValueError(f"score {score!r} must consistently prefer higher or lower values")
        if len({value.unit for value in scores}) != 1:
            raise ValueError(f"score {score!r} must use one identical unit")
        if len({value.scope for value in scores}) != 1:
            raise ValueError(f"score {score!r} must use one identical scope")
        if any(math.isnan(value.value) for value in scores):
            raise ValueError(f"score {score!r} contains NaN and cannot be ranked")
        if directions == {"higher"}:
            return max(self.candidates, key=lambda candidate: candidate.scores[score].value)
        return min(self.candidates, key=lambda candidate: candidate.scores[score].value)


@dataclass(eq=False)
class BoltzGenOutput:
    r"""Tensor output of one BoltzGen structure-design forward or sampling pass."""

    pred_coords: Tensor
    distogram_logits: Tensor
    bfactor_logits: Tensor | None = None
    denoised_coords: Tensor | None = None
    augmented_target: Tensor | None = None
    noise_level: Tensor | None = None


@dataclass(eq=False)
class BoltzGenIFoldOutput:
    r"""Per-token inverse-folding logits and optional sampled tokens."""

    logits: Tensor
    tokens: Tensor | None = None
