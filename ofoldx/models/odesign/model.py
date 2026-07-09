r"""
ODesign top-level model and its submodules.

Defines the ODesign structure-prediction model, its design-generator wrapper, the input embedder, the
confidence head and its projectors, the ODesign-specific token bond-type and constraint-distogram heads, the
training-noise sampler, and the typed input/representation/output dataclasses that thread through the
forward pass. The two model classes register themselves under their architecture keys in the model registry.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import ClassVar, Literal, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ...modules.atom import AtomToTokenEncoder, LocalAtomGeometry, TokenToAtomBroadcast
from ...modules.backend import LayerNorm
from ...modules.denoiser import CoordinateDenoiser
from ...modules.diffusion import CoordinateDenoiseContext, CoordinateEdmSampler, EdmScheduler
from ...modules.embedding import (
    RelativePositionBuckets,
    RelativePositionEmbedder,
    TemplateEmbedder,
    TemplateFeatures,
    generate_relative_position_features,
)
from ...modules.functional import pair_mask_from_token_mask
from ...modules.head import DistogramHead
from ...modules.head.frames import gather_atoms_by_index, insert_sample_axis
from ...modules.representation import Pair, Single
from ...modules.trunk import MsaTrunk, PairformerStack
from ..contract import BaseInput, StructureDesignCandidate, StructurePredictionOutput, register_pytree_dataclass
from ..mixin import ModelMixin
from ..registry import DESIGN_MODELS, MODELS, STRUCTURE_PREDICTION_MODELS
from .config import ODesignConfig, ODesignDesignConfig


@MODELS.register("odesign")
@STRUCTURE_PREDICTION_MODELS.register("odesign")
class ODesignModel(ModelMixin):
    r"""
    Predict all-atom structure with a recycling trunk and EDM coordinate diffusion.

    Wires the full ODesign pipeline: an atom/token input embedder, a recycling trunk (MSA, template,
    constraint-distogram, and Pairformer stacks), an EDM coordinate diffusion denoiser/sampler, and the
    distogram, confidence, and token bond-type heads. In training mode the diffusion module learns from a
    single denoising step on noised ground truth while a no-grad mini-rollout supplies coords for the
    confidence head; in eval/inference mode the full multi-step EDM sampler generates the predicted coords,
    optionally ensembled over several model seeds.

    Args:
        config: ODesign model configuration.

    Examples:
        >>> from ofoldx import AutoModel, AutoProcessor  # doctest: +SKIP
        >>> processor = AutoProcessor.from_pretrained("ofoldx/odesign")  # doctest: +SKIP
        >>> model = AutoModel.from_pretrained("ofoldx/odesign")  # doctest: +SKIP
        >>> output = model(processor("complex.cif"))  # doctest: +SKIP
    """

    config_class = ODesignConfig
    #: Typed input this model's ``forward`` consumes; enables the ``model(**tensors)`` keyword call path.
    #: Bound to [`ODesignInput`][] at module load (assigned after the dataclass is defined, below).
    input_class: ClassVar[type[BaseInput]]

    def __init__(self, config: ODesignConfig) -> None:
        super().__init__(config)
        self._recycling_steps = config.recycling_steps
        self._num_model_seeds = config.num_model_seeds
        self.relative_position_embedder = RelativePositionEmbedder(config.relative_position_size, config.pair_size)
        self.single_init_projection = nn.Linear(config.token_feature_size, config.single_size, bias=False)
        self.pair_left_projection = nn.Linear(config.single_size, config.pair_size, bias=False)
        self.pair_right_projection = nn.Linear(config.single_size, config.pair_size, bias=False)
        self.token_bond_projection = nn.Linear(1, config.pair_size, bias=False)
        self.pair_recycle_projection = nn.Linear(config.pair_size, config.pair_size, bias=False)
        self.single_recycle_projection = nn.Linear(config.single_size, config.single_size, bias=False)
        self.pair_recycle_norm = LayerNorm(config.pair_size)
        self.single_recycle_norm = LayerNorm(config.single_size)
        self.input_embedder = InputEmbedder(
            single_size=config.single_size,
            token_feature_size=config.token_feature_size,
            atom_feature_size=config.atom_feature_size,
            operator_backend=config.trunk.operator_backend,
        )
        self.pairformer_stack = PairformerStack(
            num_blocks=config.trunk.num_pairformer_blocks,
            single_size=config.single_size,
            pair_size=config.pair_size,
            dropout=config.trunk.pairformer_dropout,
            blocks_per_checkpoint=config.trunk.blocks_per_checkpoint,
            operator_backend=config.trunk.operator_backend,
        )
        self.coordinate_denoiser = self._build_coordinate_denoiser(config)
        self.coordinate_sampler = self._build_coordinate_sampler(config)
        self.training_noise_sampler = TrainingNoiseSampler(
            noise_log_mean=config.diffusion_training.noise_log_mean,
            noise_log_std=config.diffusion_training.noise_log_std,
            sigma_data=config.denoiser.sigma_data,
        )
        self._num_sampling_steps = config.sampler.num_sampling_steps
        self._diffusion_train_num_samples = config.diffusion_training.num_samples
        self._diffusion_rollout_steps = config.diffusion_training.rollout_steps
        self._diffusion_rollout_num_samples = config.diffusion_training.rollout_num_samples
        self.distogram_head = DistogramHead(config.pair_size, config.distogram.num_bins)
        self.confidence_head = ConfidenceHead(
            single_size=config.single_size,
            pair_size=config.pair_size,
            token_feature_size=config.token_feature_size,
            distance_bin_start=config.confidence.distance_bin_start,
            distance_bin_end=config.confidence.distance_bin_end,
            distance_bin_step=config.confidence.distance_bin_step,
            num_plddt_bins=config.confidence.num_plddt_bins,
            num_resolved_bins=config.confidence.num_resolved_bins,
            max_atoms_per_token=config.confidence.max_atoms_per_token,
            num_pae_bins=config.confidence.num_pae_bins,
            num_pde_bins=config.confidence.num_pde_bins,
            num_blocks=config.confidence.num_blocks,
            blocks_per_checkpoint=config.confidence.blocks_per_checkpoint,
            operator_backend=config.confidence.operator_backend,
            num_pair_attention_heads=config.confidence.num_pair_attention_heads,
            triangle_multiplication_hidden_size=config.confidence.triangle_multiplication_hidden_size,
            triangle_attention_hidden_size=config.confidence.triangle_attention_hidden_size,
            num_triangle_attention_heads=config.confidence.num_triangle_attention_heads,
            expansion=config.confidence.expansion,
            dropout=config.confidence.pairformer_dropout,
            single_max_abs=config.confidence.single_max_abs,
        )
        self.msa_trunk = MsaTrunk(
            pair_size=config.pair_size,
            token_feature_size=config.token_feature_size,
            msa_dropout=config.trunk.msa_dropout,
            pair_dropout=config.trunk.pair_dropout,
            operator_backend=config.trunk.operator_backend,
            blocks_per_checkpoint=config.trunk.blocks_per_checkpoint,
        )
        self.template_embedder = TemplateEmbedder(
            pair_size=config.pair_size,
            hidden_size=config.template.hidden_size,
            num_blocks=config.template.num_blocks,
            vocab_size=config.template.vocab_size,
            num_distogram_bins=config.template.num_distogram_bins,
            expansion=config.template.expansion,
            dropout=config.template.dropout,
            operator_backend=config.trunk.operator_backend,
        )
        self._build_additional_modules(config)
        self.post_init()

    def _build_coordinate_denoiser(self, config: ODesignConfig) -> CoordinateDenoiser:
        return CoordinateDenoiser(
            num_encoder_blocks=config.denoiser.num_token_encoder_blocks,
            num_encoder_attention_heads=config.denoiser.num_token_encoder_attention_heads,
            sigma_data=config.denoiser.sigma_data,
            atom_hidden_size=config.denoiser.atom_hidden_size,
            atom_pair_hidden_size=config.denoiser.atom_pair_hidden_size,
            token_hidden_size=config.denoiser.token_hidden_size,
            single_size=config.single_size,
            pair_size=config.pair_size,
            token_feature_size=config.token_feature_size,
            atom_feature_size=config.atom_feature_size,
            num_atom_encoder_blocks=config.denoiser.num_atom_encoder_blocks,
            num_atom_encoder_attention_heads=config.denoiser.num_atom_encoder_attention_heads,
            num_atom_decoder_blocks=config.denoiser.num_atom_decoder_blocks,
            num_atom_decoder_attention_heads=config.denoiser.num_atom_decoder_attention_heads,
            num_query_atoms=config.denoiser.num_query_atoms,
            num_key_atoms=config.denoiser.num_key_atoms,
            blocks_per_checkpoint=config.denoiser.blocks_per_checkpoint,
            operator_backend=config.denoiser.operator_backend,
        )

    def _build_coordinate_sampler(self, config: ODesignConfig) -> CoordinateEdmSampler:
        return CoordinateEdmSampler(
            scheduler=EdmScheduler(
                max_noise_level=config.sampler.max_noise_level,
                min_noise_level=config.sampler.min_noise_level,
                rho=config.sampler.rho,
                sigma_data=config.denoiser.sigma_data,
            ),
            solver=cast(Literal["euler", "heun"], config.sampler.solver),
            churn_rate=config.sampler.churn_rate,
            min_churn_noise_level=config.sampler.min_churn_noise_level,
            noise_scale=config.sampler.noise_scale,
            step_scale=config.sampler.step_scale,
            center_random_augmentation=config.sampler.center_random_augmentation,
            reverse_augmentation=True,
        )

    def _build_additional_modules(self, config: ODesignConfig) -> None:
        self.token_bond_type_head: TokenBondTypeHead | None
        if config.pairwise.bond_reconstruction:
            self.token_bond_type_head = TokenBondTypeHead(
                pair_size=config.pair_size,
                hidden_size=config.pairwise.hidden_size,
                num_bond_types=config.pairwise.num_bond_types,
            )
        else:
            self.token_bond_type_head = None

        self.constraint_distogram_embedder: ConstraintTemplateEmbedder | None
        if config.constraint.enabled:
            self.constraint_distogram_embedder = ConstraintTemplateEmbedder(
                pair_size=config.pair_size,
                hidden_size=config.constraint.hidden_size,
                num_blocks=config.constraint.num_blocks,
                num_bins=config.constraint.num_bins,
                min_distance=config.constraint.min_distance,
                max_distance=config.constraint.max_distance,
                dropout=config.constraint.dropout,
                blocks_per_checkpoint=config.trunk.blocks_per_checkpoint,
                operator_backend=config.trunk.operator_backend,
            )
        else:
            self.constraint_distogram_embedder = None

    def reset_parameters(self) -> None:
        r"""
        Zero-initialize the recycle projections and the PAE/PDE confidence projections.

        Called after construction. Zeroing the recycle projections makes the first recycling cycle a no-op
        update, and zeroing the pairwise-error projections starts the confidence head at uniform logits, both
        matching the upstream initialization scheme.
        """

        nn.init.zeros_(self.pair_recycle_projection.weight)
        nn.init.zeros_(self.single_recycle_projection.weight)
        nn.init.zeros_(self.confidence_head.pair_projector.pae_projection.weight)
        nn.init.zeros_(self.confidence_head.pair_projector.pde_projection.weight)

    def forward(
        self,
        batch: ODesignInput,
        *,
        recycling_steps: int | None = None,
        generator: torch.Generator | None = None,
    ) -> StructurePredictionOutput:
        r"""
        Run one training step (the TRAINING path).

        ``forward`` is the training path: it runs the recycling trunk, trains the diffusion denoiser with a
        single denoising step on noised ground truth (with grad), and attaches the denoised coords, augmented
        target, and per-sample noise level for the EDM-weighted coordinate loss. A separate no-grad EDM
        mini-rollout supplies the coords the confidence head scores. Use
        [`fold`][ofoldx.models.odesign.model.ODesignModel.fold] for inference-time multi-step sampling.

        Args:
            batch: Typed model input; must carry ``target_positions`` for the diffusion loss.
            recycling_steps: Optional number of recycling passes. Defaults to the config value.
            generator: Optional PyTorch random generator for augmentation and diffusion noise.

        Returns:
            Structure-prediction output with ``denoised_coords``, ``augmented_target``, and ``noise_level``
            populated for the diffusion loss, plus token bond-type logits when that head is enabled.

        Raises:
            ValueError: If ``recycling_steps`` is not positive, or ``batch.target_positions`` is ``None``.

        Examples:
            >>> from ofoldx import AutoModel, AutoProcessor  # doctest: +SKIP
            >>> processor = AutoProcessor.from_pretrained("ofoldx/odesign")  # doctest: +SKIP
            >>> model = AutoModel.from_pretrained("ofoldx/odesign").train()  # doctest: +SKIP
            >>> output = model(processor("complex.cif"))  # doctest: +SKIP
        """

        recycling_steps = self._recycling_steps if recycling_steps is None else recycling_steps
        if recycling_steps < 1:
            raise ValueError(f"recycling_steps must be positive for this model, got {recycling_steps}")
        representations = self._run_trunk(batch, recycling_steps=recycling_steps)
        return self._forward_train(batch, representations=representations, generator=generator)

    @torch.no_grad()
    def fold(
        self,
        batch: ODesignInput,
        *,
        recycling_steps: int | None = None,
        num_samples: int = 1,
        num_sampling_steps: int | None = None,
        num_model_seeds: int | None = None,
        generator: torch.Generator | None = None,
    ) -> StructurePredictionOutput:
        r"""
        Predict all-atom structure with the multi-step EDM sampler (the house verb for structure prediction).

        ``fold`` is the DISTINCT inference path: it runs the recycling trunk and the full reverse-diffusion
        sampler (optionally ensembled over several model seeds) under ``torch.no_grad``, and returns the raw
        typed [`StructurePredictionOutput`][ofoldx.models.contract.StructurePredictionOutput] (the pipeline
        wraps it into a result). ``forward`` remains the training path.

        Args:
            batch: Typed model input.
            recycling_steps: Optional number of recycling passes. Defaults to the config value.
            num_samples: Number of reverse-diffusion samples to draw.
            num_sampling_steps: Optional number of EDM sampling steps. Defaults to the sampler config value.
            num_model_seeds: Optional number of independent model seeds to ensemble. Each seed reruns the trunk
                and diffusion sampler and its outputs are stacked along a leading seed axis. Defaults to the
                config value; ``1`` keeps the single-seed output shape unchanged.
            generator: Optional PyTorch random generator for diffusion noise.

        Returns:
            Structure-prediction output, including token bond-type logits when that head is enabled. With more
            than one model seed the tensor fields carry a leading ``num_model_seeds`` axis.

        Raises:
            ValueError: If ``recycling_steps`` or ``num_model_seeds`` is not positive.

        Examples:
            >>> from ofoldx import AutoModel, AutoProcessor  # doctest: +SKIP
            >>> processor = AutoProcessor.from_pretrained("ofoldx/odesign")  # doctest: +SKIP
            >>> model = AutoModel.from_pretrained("ofoldx/odesign").eval()  # doctest: +SKIP
            >>> output = model.fold(processor("complex.cif"), num_samples=1)  # doctest: +SKIP
        """

        recycling_steps = self._recycling_steps if recycling_steps is None else recycling_steps
        if recycling_steps < 1:
            raise ValueError(f"recycling_steps must be positive for this model, got {recycling_steps}")
        num_model_seeds = self._num_model_seeds if num_model_seeds is None else num_model_seeds
        if num_model_seeds < 1:
            raise ValueError(f"num_model_seeds must be positive, got {num_model_seeds}")
        return self._forward_eval(
            batch,
            recycling_steps=recycling_steps,
            num_samples=num_samples,
            num_sampling_steps=num_sampling_steps,
            num_model_seeds=num_model_seeds,
            generator=generator,
        )

    def _forward_eval(
        self,
        batch: ODesignInput,
        *,
        recycling_steps: int,
        num_samples: int,
        num_sampling_steps: int | None,
        num_model_seeds: int,
        generator: torch.Generator | None,
    ) -> StructurePredictionOutput:
        r"""
        Run one eval/inference pass per model seed and ensemble the results.

        Each seed reruns the trunk and diffusion sampler independently (upstream ``main_inference_loop``);
        for a single seed the output shapes are identical to the non-ensembled path. Multiple seeds stack
        every tensor field along a new leading axis.

        Args:
            batch: Typed ODesign input.
            recycling_steps: Number of recycling passes through the trunk.
            num_samples: Number of reverse-diffusion samples to draw per seed.
            num_sampling_steps: Optional number of EDM sampling steps; ``None`` uses the sampler config value.
            num_model_seeds: Number of independent model seeds to ensemble.
            generator: Optional PyTorch random generator for diffusion noise.

        Returns:
            Structure-prediction output; with more than one seed every tensor field gains a leading
            ``num_model_seeds`` axis.
        """

        outputs = [
            self._forward_eval_single(
                batch,
                recycling_steps=recycling_steps,
                num_samples=num_samples,
                num_sampling_steps=num_sampling_steps,
                generator=generator,
            )
            for _ in range(num_model_seeds)
        ]
        if num_model_seeds == 1:
            return outputs[0]
        return _stack_outputs(outputs)

    def _forward_eval_single(
        self,
        batch: ODesignInput,
        *,
        recycling_steps: int,
        num_samples: int,
        num_sampling_steps: int | None,
        generator: torch.Generator | None,
    ) -> StructurePredictionOutput:
        r"""
        Run a single (non-ensembled) eval pass: trunk, EDM sampling, then output assembly.

        Args:
            batch: Typed ODesign input.
            recycling_steps: Number of recycling passes through the trunk.
            num_samples: Number of reverse-diffusion samples to draw.
            num_sampling_steps: Optional number of EDM sampling steps; ``None`` uses the sampler config value.
            generator: Optional PyTorch random generator for diffusion noise.

        Returns:
            Structure-prediction output for this seed.
        """

        representations = self._run_trunk(batch, recycling_steps=recycling_steps)
        pred_coords = self._predict_coords(
            batch,
            representations=representations,
            num_samples=num_samples,
            num_sampling_steps=num_sampling_steps,
            generator=generator,
        )
        return self._assemble_output(batch, representations=representations, confidence_coords=pred_coords)

    def _forward_train(
        self,
        batch: ODesignInput,
        *,
        representations: ODesignRepresentations,
        generator: torch.Generator | None,
    ) -> StructurePredictionOutput:
        r"""
        Single-step diffusion training + no-grad mini-rollout for the confidence head.

        Trains the denoiser with a single denoising step on noised ground truth (with grad), then runs a
        separate no-grad EDM mini-rollout whose coords are what the confidence head scores. The denoised
        coords, augmented target, and per-sample noise level for the EDM-weighted loss are attached to the
        returned output.

        Args:
            batch: Typed ODesign input; must carry ``target_positions``.
            representations: Trunk representations for this batch.
            generator: Optional PyTorch random generator for diffusion noise.

        Returns:
            Structure-prediction output with ``denoised_coords``, ``augmented_target``, and ``noise_level``
            populated for the diffusion loss.
        """

        denoised_coords, augmented_target, noise_level = self._train_denoise(
            batch,
            representations=representations,
            num_samples=self._diffusion_train_num_samples,
            generator=generator,
        )
        with torch.no_grad():
            rollout_coords = self._predict_coords(
                batch,
                representations=representations.detach(),
                num_samples=self._diffusion_rollout_num_samples,
                num_sampling_steps=self._diffusion_rollout_steps,
                generator=generator,
            )
        output = self._assemble_output(batch, representations=representations, confidence_coords=rollout_coords)
        output.denoised_coords = denoised_coords
        output.augmented_target = augmented_target
        output.noise_level = noise_level
        return output

    def _assemble_output(
        self,
        batch: ODesignInput,
        *,
        representations: ODesignRepresentations,
        confidence_coords: torch.Tensor,
    ) -> StructurePredictionOutput:
        r"""
        Assemble the structure-prediction output from trunk representations and coords.

        Runs the confidence head on ``confidence_coords`` and packages the predicted coords, distogram
        logits, and confidence logits; adds token bond-type logits when that head is enabled.

        Args:
            batch: Typed ODesign input.
            representations: Trunk representations for this batch.
            confidence_coords: Coordinates the confidence head scores, shape `[..., N_sample, N_atom, 3]`.

        Returns:
            Structure-prediction output with the predicted coords and every logit field populated.
        """

        confidence = self.confidence_head(
            representations.single.tensor,
            pair=representations.pair.tensor,
            token_embedding=representations.token_embedding,
            pred_coords=confidence_coords,
            token_to_rep_atom=batch.token_to_rep_atom,
            atom_to_token=batch.atom_to_token,
            atom_token_index=batch.atom_token_index,
        )
        output = StructurePredictionOutput(
            pred_coords=confidence_coords,
            distogram_logits=self.distogram_head(representations.pair.tensor),
            plddt_logits=confidence.plddt_logits,
            pae_logits=confidence.pae_logits,
            pde_logits=confidence.pde_logits,
            resolved_logits=confidence.resolved_logits,
        )
        if self.token_bond_type_head is not None:
            output.token_bond_type_logits = self.token_bond_type_head(representations.pair.tensor)
        return output

    def _train_denoise(
        self,
        batch: ODesignInput,
        *,
        representations: ODesignRepresentations,
        num_samples: int,
        generator: torch.Generator | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Noise the (augmented) ground truth once per sample and denoise in a single step.

        Args:
            batch: Typed ODesign input; must carry ``target_positions``.
            representations: Trunk representations for this batch.
            num_samples: Number of independent noised samples to draw.
            generator: Optional PyTorch random generator for augmentation and noise.

        Returns:
            ``(denoised_coords, augmented_target, noise_level)``, all with a leading ``num_samples`` axis, for
            the EDM-weighted coordinate loss.

        Raises:
            ValueError: If ``batch.target_positions`` is ``None``.
        """

        if batch.target_positions is None:
            raise ValueError("diffusion training requires batch.target_positions")
        target = batch.target_positions.to(dtype=representations.single.tensor.dtype)
        atom_mask = batch.atom_mask.to(device=target.device)
        # Center/augment the diffusion target over RESOLVED atoms only — materialized missing atoms
        # have no true coordinate. Equals atom_mask when nothing was materialized.
        resolved_mask = (batch.target_mask if batch.target_mask is not None else batch.atom_mask).to(
            device=target.device
        )

        augmented_target = self.coordinate_sampler.augment_target(
            target,
            num_samples=num_samples,
            atom_mask=resolved_mask,
            generator=generator,
        )
        batch_shape = augmented_target.shape[:-2]
        noise_level = self.training_noise_sampler(
            batch_shape,
            device=augmented_target.device,
            dtype=augmented_target.dtype,
            generator=generator,
        )
        # randn_like (not randn(shape, ...)) so torch.compile with dynamic shapes can trace
        # the noise: a symbolic atom-count dim is not a concrete SymIntArrayRef for randn.
        noise = torch.empty_like(augmented_target).normal_(generator=generator)
        noisy_coords = augmented_target + noise * noise_level[..., None, None]

        local_geometry = self._build_denoising_local_geometry(batch)
        context = CoordinateDenoiseContext(
            atom_features=batch.atom_features,
            atom_to_token=batch.atom_to_token,
            reference_positions=batch.reference_positions,
            reference_charges=batch.reference_charges,
            atom_mask=atom_mask,
            token_embedding=representations.token_embedding,
            single=representations.single.tensor,
            pair=representations.pair.tensor,
            relative_features=representations.relative_features,
            local_geometry=local_geometry,
        )
        denoised_coords = self.coordinate_denoiser(
            context.to_denoising_input(noisy_coords=noisy_coords, noise_level=noise_level)
        )
        return denoised_coords, augmented_target, noise_level

    def _run_trunk(
        self,
        batch: ODesignInput,
        *,
        recycling_steps: int,
    ) -> ODesignRepresentations:
        r"""
        Run the input embedder and the recycling trunk.

        Builds the token embedding and initial single/pair states, then recycles them through the MSA,
        template, constraint, and Pairformer stacks, returning the bundled representations for the heads and
        diffusion path.

        Args:
            batch: Typed ODesign input.
            recycling_steps: Number of recycling passes through the trunk.

        Returns:
            Trunk representations (token embedding, single, pair, relative features, and pair mask).
        """

        local_geometry = self._build_input_local_geometry(batch)
        token_embedding = self.input_embedder(
            batch.atom_features,
            token_features=batch.token_features,
            reference_positions=batch.reference_positions,
            reference_charges=batch.reference_charges,
            reference_mask=batch.reference_mask,
            atom_to_token=batch.atom_to_token,
            num_tokens=batch.token_features.shape[-2],
            atom_mask=batch.atom_mask,
            token_mask=batch.token_mask,
            local_geometry=local_geometry,
        )

        single_init = self.single_init_projection(token_embedding)
        relative_features = generate_relative_position_features(
            residue_index=batch.residue_index,
            chain_index=batch.chain_index,
            entity_index=batch.entity_index,
            symmetry_index=batch.symmetry_index,
            token_index=batch.token_index,
            dtype=single_init.dtype,
        ).to(device=single_init.device)
        pair_init = self._initial_pair(single_init, batch, relative_features)

        pair_mask = pair_mask_from_token_mask(batch.token_mask)
        state = ODesignRecyclingState(
            token_embedding=token_embedding,
            single_init=single_init,
            pair_init=pair_init,
            pair_mask=pair_mask,
        )
        single, pair = self._run_recycling(batch, state=state, recycling_steps=recycling_steps)

        return ODesignRepresentations(
            token_embedding=token_embedding,
            single=Single(single),
            pair=Pair(pair),
            relative_features=relative_features,
            pair_mask=pair_mask,
        )

    def _run_recycling(
        self,
        batch: ODesignInput,
        *,
        state: ODesignRecyclingState,
        recycling_steps: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        Run recycling, keeping gradients only for the final training cycle.

        Args:
            batch: Typed ODesign input.
            state: Init/context bundle threaded through recycling.
            recycling_steps: Number of recycling passes to run.

        Returns:
            ``(single, pair)`` representations after the final recycling cycle.
        """

        single = torch.zeros_like(state.single_init)
        pair = torch.zeros_like(state.pair_init)
        for cycle in range(recycling_steps):
            if self.training:
                with torch.set_grad_enabled(cycle == recycling_steps - 1):
                    single, pair = self._recycle_once(batch, state=state, single=single, pair=pair)
            else:
                single, pair = self._recycle_once(batch, state=state, single=single, pair=pair)

        return single, pair

    def _recycle_once(
        self,
        batch: ODesignInput,
        *,
        state: ODesignRecyclingState,
        single: torch.Tensor,
        pair: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        Run one recycling cycle: fold in the previous states, embedders, and the Pairformer stack.

        Args:
            batch: Typed ODesign input.
            state: Init/context bundle threaded through recycling.
            single: Single representation from the previous cycle.
            pair: Pair representation from the previous cycle.

        Returns:
            ``(single, pair)`` representations after this cycle.
        """

        pair = state.pair_init + self.pair_recycle_projection(self.pair_recycle_norm(pair))
        if self.constraint_distogram_embedder is not None and batch.constraint_distogram is not None:
            constraint_distogram = batch.constraint_distogram.to(device=pair.device, dtype=pair.dtype)
            pair = pair + self.constraint_distogram_embedder(pair, constraint_distogram, pair_mask=state.pair_mask)
        if batch.template_features is not None:
            pair = pair + self.template_embedder(pair, batch.template_features, pair_mask=state.pair_mask)
        if batch.msa_features is not None:
            _, pair = self.msa_trunk(
                batch.msa_features, pair, state.token_embedding, msa_mask=batch.msa_mask, pair_mask=state.pair_mask
            )
        single = state.single_init + self.single_recycle_projection(self.single_recycle_norm(single))
        single, pair = self.pairformer_stack(single, pair, pair_mask=state.pair_mask)
        return single, pair

    def _initial_pair(
        self,
        single: torch.Tensor,
        batch: ODesignInput,
        relative_features: RelativePositionBuckets,
    ) -> torch.Tensor:
        r"""
        Build the initial pair representation from single projections, relative position, and token bonds.

        Args:
            single: Initial single representation, shape `[..., N_token, single_size]`.
            batch: Typed ODesign input (supplies optional token bonds and the token mask).
            relative_features: Relative-position feature bundle.

        Returns:
            Initial pair representation `[..., N_token, N_token, pair_size]`, masked by the token-pair mask.
        """

        left = self.pair_left_projection(single).unsqueeze(-2)
        right = self.pair_right_projection(single).unsqueeze(-3)
        if batch.token_bonds is None:
            token_bonds = single.new_zeros(*single.shape[:-1], single.shape[-2])
        else:
            token_bonds = batch.token_bonds.to(device=single.device, dtype=single.dtype)
            token_bonds = torch.broadcast_to(token_bonds, (*single.shape[:-1], single.shape[-2]))
        token_bonds = token_bonds.unsqueeze(-1)
        pair = left + right
        pair = pair + self.relative_position_embedder(relative_features)
        pair = pair + self.token_bond_projection(token_bonds)

        pair_mask = pair_mask_from_token_mask(batch.token_mask)
        return pair * pair_mask.unsqueeze(-1).to(dtype=pair.dtype)

    def _predict_coords(
        self,
        batch: ODesignInput,
        *,
        representations: ODesignRepresentations,
        num_samples: int,
        num_sampling_steps: int | None,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        r"""
        Run the EDM sampler over the coordinate denoiser to produce predicted coords.

        Args:
            batch: Typed ODesign input.
            representations: Trunk representations conditioning the denoiser.
            num_samples: Number of reverse-diffusion samples to draw.
            num_sampling_steps: Optional number of EDM sampling steps; ``None`` uses the sampler config value.
            generator: Optional PyTorch random generator for diffusion noise.

        Returns:
            Predicted coordinates of shape `[..., N_sample, N_atom, 3]`.
        """

        local_geometry = self._build_denoising_local_geometry(batch)
        condition_mask = self._condition_atom_mask(batch)
        return self.coordinate_sampler(
            self.coordinate_denoiser,
            context=CoordinateDenoiseContext(
                atom_features=batch.atom_features,
                atom_to_token=batch.atom_to_token,
                reference_positions=batch.reference_positions,
                reference_charges=batch.reference_charges,
                atom_mask=batch.atom_mask,
                token_embedding=representations.token_embedding,
                single=representations.single.tensor,
                pair=representations.pair.tensor,
                relative_features=representations.relative_features,
                local_geometry=local_geometry,
            ),
            num_samples=num_samples,
            num_sampling_steps=self._num_sampling_steps if num_sampling_steps is None else num_sampling_steps,
            condition_coords=batch.reference_positions,
            condition_mask=condition_mask,
            generator=generator,
        )

    def _condition_atom_mask(self, batch: ODesignInput) -> Tensor:
        restype_size = 35
        profile_size = 32
        deletion_mean_size = 1
        hotspot_index = self.config.single_size + restype_size + profile_size + deletion_mean_size
        if batch.token_features.shape[-1] <= hotspot_index:
            return torch.zeros_like(batch.atom_mask, dtype=torch.bool)
        condition_token = (batch.token_features[..., hotspot_index] != 0) & batch.token_mask.bool()
        atom_token = batch.atom_to_token.clamp_min(0)
        return torch.gather(condition_token, dim=-1, index=atom_token).bool() & batch.atom_mask.bool()

    def _build_input_local_geometry(self, batch: ODesignInput) -> LocalAtomGeometry:
        r"""
        Build the local atom geometry using the input embedder's atom encoder layout.

        Args:
            batch: Typed ODesign input (supplies reference positions and space ids).

        Returns:
            Local atom geometry for the input embedder.
        """

        reference_positions = batch.reference_positions.to(device=batch.atom_features.device)
        return self.input_embedder.build_local_geometry(reference_positions, batch.reference_space_id)

    def _build_denoising_local_geometry(self, batch: ODesignInput) -> LocalAtomGeometry:
        r"""
        Build the local atom geometry using the coordinate denoiser's atom encoder layout.

        Args:
            batch: Typed ODesign input (supplies reference positions and space ids).

        Returns:
            Local atom geometry for the coordinate denoiser.
        """

        reference_positions = batch.reference_positions.to(device=batch.atom_features.device)
        return self.coordinate_denoiser.build_local_geometry(reference_positions, batch.reference_space_id)


@MODELS.register("odesign_design")
@DESIGN_MODELS.register("odesign_design")
class ODesignDesigner(ODesignModel):
    r"""
    ODesign all-atom structure generator for design workflows.

    The current ODesign implementation does not include a sequence decoder. This entry point exposes the
    existing structure-generation path through [`StructureDesignCandidate`][], with the generated structure nested
    under ``structure_prediction`` and sequence fields left unset.

    Args:
        config: ODesign design-generator config. Defaults to ``ODesignDesignConfig()``.
    """

    config_class = ODesignDesignConfig

    def __init__(self, config: ODesignDesignConfig | None = None) -> None:
        super().__init__(ODesignDesignConfig() if config is None else config)

    # Design-generator variant: ``design`` emits StructureDesignCandidate, a deliberately different contract
    # from the structure-prediction parent (callers reach it via its own type, never as ODesignModel).
    # ``forward`` stays the inherited TRAINING path; ``design`` is the DISTINCT inference verb.
    @torch.no_grad()
    def design(
        self,
        batch: ODesignInput,
        *,
        recycling_steps: int | None = None,
        num_samples: int = 1,
        num_sampling_steps: int | None = None,
        num_model_seeds: int | None = None,
        generator: torch.Generator | None = None,
    ) -> StructureDesignCandidate:
        r"""
        Generate structure candidates and normalize them into the design-generator output contract.

        ``design`` is the house verb for design (the inference path): it runs the eval structure-prediction
        path under ``torch.no_grad`` and nests the result under ``structure_prediction``, returning the raw
        typed [`StructureDesignCandidate`][ofoldx.models.contract.StructureDesignCandidate] (the pipeline
        wraps it into a design output). ``forward`` remains the inherited training path.

        Args:
            batch: Typed ODesign input.
            recycling_steps: Optional number of recycling passes. Defaults to the config value.
            num_samples: Number of reverse-diffusion samples to draw.
            num_sampling_steps: Optional number of EDM sampling steps. Defaults to the sampler config value.
            num_model_seeds: Optional number of independent model seeds to ensemble. Defaults to the config
                value; ``1`` keeps the single-seed output shape unchanged.
            generator: Optional PyTorch random generator for diffusion noise.

        Returns:
            Design-generator output with ``structure_prediction`` populated. ``sequence_logits`` and
            ``sequence_tokens`` are ``None`` until an ODesign sequence head is added.

        Raises:
            ValueError: If ``recycling_steps``, ``num_samples``, or ``num_model_seeds`` is not positive.

        Examples:
            >>> from ofoldx import AutoModel, AutoProcessor  # doctest: +SKIP
            >>> processor = AutoProcessor.from_pretrained("ofoldx/odesign_design")  # doctest: +SKIP
            >>> model = AutoModel.from_pretrained("ofoldx/odesign_design")  # doctest: +SKIP
            >>> output = model.design(processor("complex.cif"))  # doctest: +SKIP
        """

        structure_prediction = self._predict_structure_for_design(
            batch,
            recycling_steps=recycling_steps,
            num_samples=num_samples,
            num_sampling_steps=num_sampling_steps,
            num_model_seeds=num_model_seeds,
            generator=generator,
        )
        return StructureDesignCandidate(structure_prediction=structure_prediction)

    def _predict_structure_for_design(
        self,
        batch: ODesignInput,
        *,
        recycling_steps: int | None,
        num_samples: int,
        num_sampling_steps: int | None,
        num_model_seeds: int | None,
        generator: torch.Generator | None,
    ) -> StructurePredictionOutput:
        r"""
        Validate the generation knobs and run the eval structure-prediction path.

        Args:
            batch: Typed ODesign input.
            recycling_steps: Optional number of recycling passes; ``None`` uses the config value.
            num_samples: Number of reverse-diffusion samples to draw.
            num_sampling_steps: Optional number of EDM sampling steps; ``None`` uses the sampler config value.
            num_model_seeds: Optional number of model seeds to ensemble; ``None`` uses the config value.
            generator: Optional PyTorch random generator for diffusion noise.

        Returns:
            Structure-prediction output to nest under the design-generator contract.

        Raises:
            ValueError: If ``num_samples``, ``recycling_steps``, or ``num_model_seeds`` is not positive.
        """

        if num_samples < 1:
            raise ValueError(f"num_samples must be positive, got {num_samples}")
        recycling_steps = self._recycling_steps if recycling_steps is None else recycling_steps
        if recycling_steps < 1:
            raise ValueError(f"recycling_steps must be positive for this model, got {recycling_steps}")
        num_model_seeds = self._num_model_seeds if num_model_seeds is None else num_model_seeds
        if num_model_seeds < 1:
            raise ValueError(f"num_model_seeds must be positive, got {num_model_seeds}")
        return self._forward_eval(
            batch,
            recycling_steps=recycling_steps,
            num_samples=num_samples,
            num_sampling_steps=num_sampling_steps,
            num_model_seeds=num_model_seeds,
            generator=generator,
        )


class InputEmbedder(nn.Module):
    r"""
    Build token input embeddings from packed atom and token features.

    Runs the atom-to-token encoder to derive the leading token-single channels, then concatenates the
    trailing per-token feature channels (everything past ``single_size``) to form the full token embedding.

    Args:
        single_size: Width of the atom-derived token-single prefix.
        token_feature_size: Total per-token input feature width the embedder expects.
        atom_feature_size: Per-atom input feature width fed to the atom encoder.
        operator_backend: Operator backend selector passed to the atom encoder.
    """

    def __init__(
        self,
        single_size: int = 384,
        token_feature_size: int = 449,
        atom_feature_size: int = 385,
        operator_backend: str = "auto",
    ) -> None:
        super().__init__()
        self.single_size = single_size
        self.token_feature_size = token_feature_size
        self.atom_encoder = AtomToTokenEncoder(
            token_hidden_size=single_size,
            atom_feature_size=atom_feature_size,
            operator_backend=operator_backend,
        )

    def forward(
        self,
        atom_features: Tensor,
        *,
        reference_positions: Tensor,
        reference_charges: Tensor,
        reference_mask: Tensor,
        token_features: Tensor,
        atom_to_token: Tensor,
        num_tokens: int,
        atom_mask: Tensor | None = None,
        token_mask: Tensor | None = None,
        local_geometry: LocalAtomGeometry | None = None,
    ) -> Tensor:
        r"""
        Return token embeddings with atom-derived channels followed by token features.

        Args:
            atom_features: Per-atom input features `[..., N_atom, atom_feature_size]`.
            reference_positions: Centered reference-conformer atom coords `[..., N_atom, 3]`.
            reference_charges: Per-atom reference formal charges (arcsinh-compressed internally).
            reference_mask: Boolean mask for valid reference-conformer atoms.
            token_features: Per-token input features `[..., N_token, token_feature_size]`.
            atom_to_token: Token index for each atom `[..., N_atom]`.
            num_tokens: Number of tokens the atom encoder aggregates into.
            atom_mask: Optional boolean atom validity mask.
            token_mask: Optional boolean token validity mask; zeros out padded tokens when given.
            local_geometry: Optional precomputed local atom geometry.

        Returns:
            Token embedding `[..., N_token, single_size + token_tail]` combining atom-derived single channels
            and the trailing token features.
        """

        encoder_reference_charges = torch.arcsinh(reference_charges)
        atom_encoder_output = self.atom_encoder(
            atom_features,
            reference_positions=reference_positions,
            reference_charges=encoder_reference_charges,
            reference_mask=reference_mask,
            atom_mask=atom_mask,
            atom_to_token=atom_to_token,
            num_tokens=num_tokens,
            local_geometry=local_geometry,
        )
        token_single = atom_encoder_output.token_hidden

        token_tail = self._token_tail(token_single, token_features)
        token_embedding = torch.cat([token_single, token_tail], dim=-1)

        if token_mask is not None:
            token_embedding = token_embedding * token_mask.unsqueeze(-1).to(dtype=token_embedding.dtype)
        return token_embedding

    def build_local_geometry(self, reference_positions: Tensor, reference_space_id: Tensor) -> LocalAtomGeometry:
        r"""
        Build local atom geometry using the input atom encoder layout.

        Args:
            reference_positions: Centered reference-conformer atom coords `[..., N_atom, 3]`.
            reference_space_id: Reference-conformer group id for each atom `[..., N_atom]`.

        Returns:
            Local atom geometry for the input atom encoder.
        """

        return self.atom_encoder.build_local_geometry(reference_positions, reference_space_id)

    def _token_tail(self, token_single: Tensor, token_features: Tensor) -> Tensor:
        r"""
        Slice the trailing per-token feature channels to concatenate onto the atom-derived single.

        Args:
            token_single: Atom-derived token-single tensor `[..., N_token, single_size]`.
            token_features: Full per-token feature tensor `[..., N_token, token_feature_size]`.

        Returns:
            The trailing feature channels `[..., N_token, token_feature_size - single_size]`, matched to
            ``token_single``'s device and dtype.

        Raises:
            ValueError: If ``token_features`` has the wrong last dimension, or its leading dimensions do not
                match ``token_single``.
        """

        if token_features.shape[-1] != self.token_feature_size:
            raise ValueError(
                f"token_features must have last dimension {self.token_feature_size}, got {token_features.shape[-1]}"
            )
        token_tail = token_features[..., self.single_size :]
        if token_tail.shape[:-1] != token_single.shape[:-1]:
            raise ValueError(
                "token_features leading dimensions must match token embeddings, got "
                f"{tuple(token_tail.shape[:-1])} and {tuple(token_single.shape[:-1])}"
            )
        return token_tail.to(device=token_single.device, dtype=token_single.dtype)


class TrainingNoiseSampler(nn.Module):
    r"""
    Sample EDM training noise levels from a log-normal distribution.

    Draw
    ``sigma = exp(noise_log_mean + noise_log_std * N(0, 1)) * sigma_data``. Each training sample gets an
    independent noise level, so the denoiser learns to denoise across the whole sigma range
    in a single step (no reverse-sampling loop during training).

    Args:
        noise_log_mean: Mean of the underlying log-normal in log space.
        noise_log_std: Standard deviation of the underlying log-normal in log space.
        sigma_data: Data-scale constant the sampled sigma is multiplied by.
    """

    def __init__(self, *, noise_log_mean: float = -1.2, noise_log_std: float = 1.5, sigma_data: float = 16.0) -> None:
        super().__init__()
        self.noise_log_mean = noise_log_mean
        self.noise_log_std = noise_log_std
        self.sigma_data = sigma_data

    def forward(
        self,
        size: tuple[int, ...] | torch.Size,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        r"""
        Return a tensor of independent noise levels with shape ``size``.

        Args:
            size: Leading batch shape of the noise-level tensor to draw.
            device: Optional device for the sampled tensor.
            dtype: Optional dtype for the sampled tensor.
            generator: Optional PyTorch random generator.

        Returns:
            Positive noise levels (sigma) of shape ``size``.

        Examples:
            >>> import torch
            >>> _ = torch.manual_seed(0)
            >>> sampler = TrainingNoiseSampler()
            >>> noise_level = sampler((2,))
            >>> noise_level.shape
            torch.Size([2])
            >>> bool((noise_level > 0).all())
            True
        """

        standard_normal = torch.randn(size, device=device, dtype=dtype, generator=generator)
        return (standard_normal * self.noise_log_std + self.noise_log_mean).exp() * self.sigma_data


class ConfidenceHead(nn.Module):
    r"""
    Predict confidence logits from single, pair, and coordinate outputs.

    Embeds predicted representative-atom distances into the pair state, runs a Pairformer stack, and projects
    the result to per-atom (plddt, resolved) and per-token-pair (PAE, PDE) confidence logits. Handles the
    sample axis by running each drawn coordinate sample through the head and stacking the outputs.

    Args:
        single_size: Width of the token single representation.
        pair_size: Width of the token pair representation.
        token_feature_size: Width of the per-token input features fed to the pair embedder.
        distance_bin_start: Lower edge of the representative-distance binning range, in angstroms.
        distance_bin_end: Upper edge of the representative-distance binning range, in angstroms.
        distance_bin_step: Width of each representative-distance bin, in angstroms.
        num_plddt_bins: Number of per-atom lDDT bins.
        num_resolved_bins: Number of per-atom resolved bins.
        max_atoms_per_token: Maximum atoms per token for the per-offset weight tables.
        num_pae_bins: Number of pairwise aligned-error bins.
        num_pde_bins: Number of pairwise distance-error bins.
        num_blocks: Number of Pairformer blocks in the confidence stack.
        num_pair_attention_heads: Number of pair (single) attention heads in the stack.
        triangle_multiplication_hidden_size: Hidden width of the triangle multiplication.
        triangle_attention_hidden_size: Hidden width of the triangle attention.
        num_triangle_attention_heads: Number of triangle attention heads.
        expansion: Transition expansion factor in the stack.
        dropout: Dropout probability in the Pairformer stack.
        blocks_per_checkpoint: Optional gradient-checkpointing block granularity.
        operator_backend: Operator backend selector for the stack.
        single_max_abs: Optional clamp on the single representation's absolute value before norming; ``None``
            disables clamping.
    """

    def __init__(
        self,
        single_size: int = 384,
        pair_size: int = 128,
        token_feature_size: int = 449,
        distance_bin_start: float = 3.25,
        distance_bin_end: float = 52.0,
        distance_bin_step: float = 1.25,
        num_plddt_bins: int = 50,
        num_resolved_bins: int = 2,
        max_atoms_per_token: int = 24,
        num_pae_bins: int = 64,
        num_pde_bins: int = 64,
        num_blocks: int = 4,
        num_pair_attention_heads: int = 16,
        triangle_multiplication_hidden_size: int = 128,
        triangle_attention_hidden_size: int = 32,
        num_triangle_attention_heads: int = 4,
        expansion: int = 4,
        dropout: float = 0.0,
        blocks_per_checkpoint: int | None = None,
        operator_backend: str = "auto",
        single_max_abs: float | None = 512.0,
    ) -> None:
        super().__init__()
        self.single_max_abs = single_max_abs
        self.pair_embedder = ConfidencePairEmbedder(
            pair_size=pair_size,
            token_feature_size=token_feature_size,
            distance_bin_start=distance_bin_start,
            distance_bin_end=distance_bin_end,
            distance_bin_step=distance_bin_step,
        )
        self.pairformer_stack = PairformerStack(
            num_blocks=num_blocks,
            single_size=single_size,
            pair_size=pair_size,
            triangle_multiplication_hidden_size=triangle_multiplication_hidden_size,
            triangle_attention_hidden_size=triangle_attention_hidden_size,
            num_triangle_attention_heads=num_triangle_attention_heads,
            num_pair_attention_heads=num_pair_attention_heads,
            expansion=expansion,
            dropout=dropout,
            blocks_per_checkpoint=blocks_per_checkpoint,
            operator_backend=operator_backend,
        )
        self.single_norm = LayerNorm(single_size)
        self.atom_projector = AtomConfidenceProjector(
            single_size=single_size,
            num_plddt_bins=num_plddt_bins,
            num_resolved_bins=num_resolved_bins,
            max_atoms_per_token=max_atoms_per_token,
        )
        self.pair_projector = PairConfidenceProjector(
            pair_size=pair_size, num_pae_bins=num_pae_bins, num_pde_bins=num_pde_bins
        )

    def forward(
        self,
        single: Tensor,
        pair: Tensor,
        token_embedding: Tensor,
        pred_coords: Tensor,
        token_to_rep_atom: Tensor,
        atom_to_token: Tensor,
        atom_token_index: Tensor,
        pair_mask: Tensor | None = None,
    ) -> ConfidenceOutput:
        r"""
        Return confidence logits for atom-level and token-pair predictions.

        Args:
            single: Token single representation `[..., N_token, single_size]`.
            pair: Token pair representation `[..., N_token, N_token, pair_size]`.
            token_embedding: Per-token input embedding `[..., N_token, token_feature_size]`.
            pred_coords: Predicted coords `[..., N_sample, N_atom, 3]`.
            token_to_rep_atom: Representative-atom index for each token `[..., N_token]`.
            atom_to_token: Token index for each atom `[..., N_atom]`.
            atom_token_index: Per-token atom offset for each atom `[..., N_atom]`.
            pair_mask: Optional token-pair validity mask `[..., N_token, N_token]`.

        Returns:
            Confidence logits with the plddt/resolved fields carrying a sample axis and the PAE/PDE fields a
            sample axis ahead of the token-pair axes.
        """

        if self.single_max_abs is not None:
            single = torch.clamp(single, min=-self.single_max_abs, max=self.single_max_abs)
        single = self.single_norm(single)
        # Select each token's representative-atom coordinate by gathering on the per-token global atom index
        # (replaces a boolean mask over the atom axis; supports batched/sampled leading dims). pred_coords
        # is ``[..., N_sample, N_atom, 3]``; insert the sample axis into the per-token index before gathering.
        rep_index = token_to_rep_atom.to(device=pred_coords.device, dtype=torch.long)
        num_samples = pred_coords.shape[-3]
        sample_rep_index = insert_sample_axis(rep_index, num_samples)
        representative_coords = gather_atoms_by_index(pred_coords, sample_rep_index)

        plddt_logits = []
        pae_logits = []
        pde_logits = []
        resolved_logits = []
        for representative_coords_sample in representative_coords.unbind(dim=-3):
            sample_output = self._forward_single_sample(
                single=single,
                pair=pair,
                token_embedding=token_embedding,
                pair_mask=pair_mask,
                representative_coords=representative_coords_sample,
                atom_to_token=atom_to_token,
                atom_token_index=atom_token_index,
            )
            plddt_logits.append(sample_output.plddt_logits)
            pae_logits.append(sample_output.pae_logits)
            pde_logits.append(sample_output.pde_logits)
            resolved_logits.append(sample_output.resolved_logits)

        return ConfidenceOutput(
            plddt_logits=torch.stack(plddt_logits, dim=-3),
            pae_logits=torch.stack(pae_logits, dim=-4),
            pde_logits=torch.stack(pde_logits, dim=-4),
            resolved_logits=torch.stack(resolved_logits, dim=-3),
        )

    def _forward_single_sample(
        self,
        single: Tensor,
        pair: Tensor,
        token_embedding: Tensor,
        pair_mask: Tensor | None,
        representative_coords: Tensor,
        atom_to_token: Tensor,
        atom_token_index: Tensor,
    ) -> ConfidenceOutput:
        r"""
        Run the confidence head for a single coordinate sample.

        Args:
            single: Token single representation `[..., N_token, single_size]`.
            pair: Token pair representation `[..., N_token, N_token, pair_size]`.
            token_embedding: Per-token input embedding `[..., N_token, token_feature_size]`.
            pair_mask: Optional token-pair validity mask `[..., N_token, N_token]`.
            representative_coords: Per-token representative-atom coords `[..., N_token, 3]`.
            atom_to_token: Token index for each atom `[..., N_atom]`.
            atom_token_index: Per-token atom offset for each atom `[..., N_atom]`.

        Returns:
            Confidence logits (plddt, PAE, PDE, resolved) for this single sample.
        """

        pair = self.pair_embedder(pair, token_embedding, representative_coords)
        single, pair = self.pairformer_stack(single, pair, pair_mask=pair_mask)
        plddt_logits, resolved_logits = self.atom_projector(single, atom_to_token, atom_token_index)
        pae_logits, pde_logits = self.pair_projector(pair)

        return ConfidenceOutput(
            plddt_logits=plddt_logits,
            pae_logits=pae_logits,
            pde_logits=pde_logits,
            resolved_logits=resolved_logits,
        )


class ConfidencePairEmbedder(nn.Module):
    r"""
    Build confidence pair states from token embeddings and predicted distances.

    Adds left/right token-embedding projections onto the pair state, then folds in both a one-hot-binned and
    a raw-scalar projection of the pairwise representative-atom distances.

    Args:
        pair_size: Width of the pair representation.
        token_feature_size: Width of the per-token input embedding projected into the pair state.
        distance_bin_start: Lower edge of the distance binning range, in angstroms.
        distance_bin_end: Upper edge of the distance binning range, in angstroms.
        distance_bin_step: Width of each distance bin, in angstroms.
    """

    def __init__(
        self,
        pair_size: int = 128,
        token_feature_size: int = 449,
        distance_bin_start: float = 3.25,
        distance_bin_end: float = 52.0,
        distance_bin_step: float = 1.25,
    ) -> None:
        super().__init__()
        lower_bins = torch.arange(distance_bin_start, distance_bin_end, distance_bin_step)
        upper_bins = torch.cat([lower_bins[1:], lower_bins.new_tensor([torch.inf])], dim=-1)
        self.lower_bins: Tensor
        self.upper_bins: Tensor
        self.register_buffer("lower_bins", lower_bins)
        self.register_buffer("upper_bins", upper_bins)
        self.left_token_projection = nn.Linear(token_feature_size, pair_size, bias=False)
        self.right_token_projection = nn.Linear(token_feature_size, pair_size, bias=False)
        self.distance_one_hot_projection = nn.Linear(len(lower_bins), pair_size, bias=False)
        self.distance_projection = nn.Linear(1, pair_size, bias=False)

    def forward(self, pair: Tensor, token_embedding: Tensor, representative_coords: Tensor) -> Tensor:
        r"""
        Augment the pair state with token and predicted-distance signals.

        Args:
            pair: ``(B, N_token, N_token, pair_size)`` pair representation.
            token_embedding: ``(B, N_token, token_feature_size)`` token embeddings, projected
                left/right and added along the row/column axes.
            representative_coords: ``(B, N_token, 3)`` representative-atom coords; their
                pairwise distances feed both a one-hot-binned and a raw-scalar projection.

        Returns:
            ``(B, N_token, N_token, pair_size)`` updated pair representation.
        """

        pair = pair + self.left_token_projection(token_embedding).unsqueeze(-2)
        pair = pair + self.right_token_projection(token_embedding).unsqueeze(-3)
        # torch.cdist does not implement CUDA bf16; keep the cast local to this distance calculation.
        distance = torch.cdist(representative_coords.to(torch.float32), representative_coords.to(torch.float32))
        pair = pair + self.distance_one_hot_projection(
            _one_hot(distance, lower_bins=self.lower_bins, upper_bins=self.upper_bins),
        )
        return pair + self.distance_projection(distance.unsqueeze(-1))


class AtomConfidenceProjector(nn.Module):
    r"""
    Project token single representations to atom-level confidence logits.

    Broadcasts each token's single representation to its atoms, then contracts it against a per-within-token
    atom-offset weight table to produce per-atom plddt and resolved logits.

    Args:
        single_size: Width of the token single representation.
        num_plddt_bins: Number of per-atom lDDT bins.
        num_resolved_bins: Number of per-atom resolved bins.
        max_atoms_per_token: Maximum atoms per token, sizing the per-offset weight tables.
    """

    def __init__(
        self,
        single_size: int = 384,
        num_plddt_bins: int = 50,
        num_resolved_bins: int = 2,
        max_atoms_per_token: int = 24,
    ):
        super().__init__()
        self.max_atoms_per_token = max_atoms_per_token
        self.plddt_weight = nn.Parameter(torch.empty(max_atoms_per_token, single_size, num_plddt_bins))
        self.resolved_weight = nn.Parameter(torch.empty(max_atoms_per_token, single_size, num_resolved_bins))
        self.plddt_norm = LayerNorm(single_size)
        self.resolved_norm = LayerNorm(single_size)
        self.token_to_atom_broadcast = TokenToAtomBroadcast()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        r"""
        Zero-initialize the per-atom-offset plddt and resolved weight tables (zero logits at init).
        """

        nn.init.zeros_(self.plddt_weight)
        nn.init.zeros_(self.resolved_weight)

    def forward(self, single: Tensor, atom_to_token: Tensor, atom_token_index: Tensor) -> tuple[Tensor, Tensor]:
        r"""
        Produce per-atom plddt and resolved logits.

        Broadcasts the token single to atoms via ``atom_to_token``, gathers each atom's
        within-token-offset weight matrix using ``atom_token_index``, then contracts (after a LayerNorm)
        the atom single with its weight.

        Args:
            single: Token single representation `[..., N_token, single_size]`.
            atom_to_token: Per-atom index into the token axis, mapping each atom to its parent token,
                shape `[..., N_atom]`.
            atom_token_index: Per-atom within-token offset (0-based position of the atom inside its token),
                selecting the atom's weight matrix, shape `[..., N_atom]`.

        Returns:
            Tuple ``(plddt_logits, resolved_logits)`` with shapes `[..., N_atom, num_plddt_bins]` and
            `[..., N_atom, num_resolved_bins]`.
        """

        atom_single = self.token_to_atom_broadcast(single, atom_to_token)
        atom_token_index = atom_token_index.to(device=single.device, dtype=torch.long)
        plddt_weight = self._gather_atom_weight(self.plddt_weight, atom_token_index)
        resolved_weight = self._gather_atom_weight(self.resolved_weight, atom_token_index)
        plddt_logits = torch.einsum("...nc,...ncb->...nb", self.plddt_norm(atom_single), plddt_weight)
        resolved_logits = torch.einsum("...nc,...ncb->...nb", self.resolved_norm(atom_single), resolved_weight)
        return plddt_logits, resolved_logits

    def _gather_atom_weight(self, weight: Tensor, atom_token_index: Tensor) -> Tensor:
        r"""
        Select each atom's per-within-token-offset weight matrix ``[..., N_atom, C, B]`` per element.

        ``F.embedding`` on the flattened ``[max_atoms_per_token, C*B]`` table gathers a per-element (batched)
        ``atom_token_index`` correctly — unlike dense-parameter advanced indexing ``weight[atom_token_index]``
        with the ``ncb`` einsum, which assumes one shared 1-D index (bsz=1). The trailing ``C*B -> (C, B)`` split
        is over static dims. (Out-of-range offsets raise in ``F.embedding``; no ``.max()`` sync needed.)
        """

        num_offsets, channels, bins = weight.shape
        gathered = F.embedding(atom_token_index, weight.reshape(num_offsets, channels * bins))
        return gathered.reshape(*gathered.shape[:-1], channels, bins)


class PairConfidenceProjector(nn.Module):
    r"""
    Project pair representations to pair-level confidence logits.

    Emits PAE logits from the (asymmetric) normed pair and PDE logits from the symmetrized pair, since
    distance error is symmetric.

    Args:
        pair_size: Width of the pair representation.
        num_pae_bins: Number of pairwise aligned-error bins.
        num_pde_bins: Number of pairwise distance-error bins.
    """

    def __init__(self, pair_size: int = 128, num_pae_bins: int = 64, num_pde_bins: int = 64) -> None:
        super().__init__()
        self.pae_norm = LayerNorm(pair_size)
        self.pde_norm = LayerNorm(pair_size)
        self.pae_projection = nn.Linear(pair_size, num_pae_bins, bias=False)
        self.pde_projection = nn.Linear(pair_size, num_pde_bins, bias=False)

    def forward(self, pair: Tensor) -> tuple[Tensor, Tensor]:
        r"""
        Map pair states to PAE and PDE logits.

        PAE uses the (asymmetric) normed pair; PDE uses the symmetrized pair (``pair + pair^T``)
        since distance error is symmetric.

        Args:
            pair: Pair representation `[B, N_token, N_token, pair_size]`.

        Returns:
            Tuple ``(pae_logits, pde_logits)`` with shapes `[B, N_token, N_token, num_pae_bins]` and
            `[B, N_token, N_token, num_pde_bins]`.
        """

        pae_logits = self.pae_projection(self.pae_norm(pair))
        pde_logits = self.pde_projection(self.pde_norm(pair + pair.transpose(-2, -3)))
        return pae_logits, pde_logits


class TokenBondTypeHead(nn.Module):
    r"""
    Predict ODesign token-pair bond type logits from pair representations.

    A two-layer Linear classifier over the pair state, emitting per-token-pair bond-type logits.

    Args:
        pair_size: Width of the pair representation.
        hidden_size: Width of the classifier's hidden layer.
        num_bond_types: Number of bond-type classes the head predicts.
    """

    def __init__(self, *, pair_size: int = 128, hidden_size: int = 128, num_bond_types: int = 2) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(pair_size, hidden_size),
            nn.Linear(hidden_size, num_bond_types),
        )

    def forward(self, pair: Tensor) -> Tensor:
        r"""
        Map pair states to token-pair bond-type logits via a two-layer Linear classifier.

        Args:
            pair: Pair representation `[B, N_token, N_token, pair_size]`.

        Returns:
            Bond-type logits `[B, N_token, N_token, num_bond_types]`.
        """

        return self.classifier(pair)


class ConstraintTemplateEmbedder(nn.Module):
    r"""
    Fold distance constraints into the pair representation (upstream ODesign Algorithm 16 template path).

    A ``constraint_distogram`` carries raw pairwise distances (``0`` where unconstrained) between condition
    tokens. This module bins those distances into a one-hot distogram, mixes them with the current pair
    state through a pair-only Pairformer stack, and projects the result back to ``pair_size`` so it can be
    added into the trunk pair each recycling cycle. It is the mechanism that makes user-supplied constraints
    actually take effect; without it the ``constraint_distogram`` input is silently ignored.

    Args:
        pair_size: Width of the trunk pair representation the update is added back into.
        hidden_size: Width of the embedder's internal pair channel.
        num_blocks: Number of pair-only Pairformer blocks.
        num_bins: Number of distance bins the raw constraint distances are one-hot encoded into.
        min_distance: Lower edge of the binned distance range, in angstroms.
        max_distance: Upper edge of the binned distance range, in angstroms.
        dropout: Dropout probability inside the Pairformer stack.
        blocks_per_checkpoint: Optional gradient-checkpointing block granularity.
        operator_backend: Operator backend selector for the stack.
    """

    def __init__(
        self,
        *,
        pair_size: int = 128,
        hidden_size: int = 64,
        num_blocks: int = 2,
        num_bins: int = 39,
        min_distance: float = 3.25,
        max_distance: float = 50.75,
        dropout: float = 0.25,
        blocks_per_checkpoint: int | None = None,
        operator_backend: str = "auto",
    ) -> None:
        super().__init__()
        self.num_bins = num_bins
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.pair_projection = nn.Linear(pair_size, hidden_size, bias=False)
        self.pair_norm = LayerNorm(pair_size)
        self.distogram_projection = nn.Linear(num_bins, hidden_size, bias=False)
        self.pairformer_stack = PairformerStack(
            num_blocks=num_blocks,
            single_size=0,
            pair_size=hidden_size,
            dropout=dropout,
            blocks_per_checkpoint=blocks_per_checkpoint,
            operator_backend=operator_backend,
        )
        self.hidden_norm = LayerNorm(hidden_size)
        self.activation = nn.ReLU()
        self.output_projection = nn.Linear(hidden_size, pair_size, bias=False)

    def forward(
        self,
        pair: Tensor,
        constraint_distogram: Tensor,
        pair_mask: Tensor | None = None,
    ) -> Tensor:
        r"""
        Return the constraint pair update added into the trunk pair representation.

        Args:
            pair: ``(B, N_token, N_token, pair_size)`` current pair state.
            constraint_distogram: ``(B, N_token, N_token, 1)`` raw constraint distances (``0`` = unconstrained).
            pair_mask: Optional ``(B, N_token, N_token)`` token-pair validity mask.

        Returns:
            ``(B, N_token, N_token, pair_size)`` constraint pair update.
        """

        # Bin the raw distance into a one-hot distogram over ``num_bins`` bins (last bin = beyond max).
        boundaries = torch.linspace(self.min_distance, self.max_distance, self.num_bins - 1, device=pair.device)
        true_bins = (constraint_distogram > boundaries).sum(dim=-1)
        distogram = F.one_hot(true_bins, self.num_bins).to(dtype=pair.dtype)
        hidden = self.pair_projection(self.pair_norm(pair)) + self.distogram_projection(distogram)
        _, hidden = self.pairformer_stack(None, hidden, pair_mask=pair_mask)
        return self.output_projection(self.activation(self.hidden_norm(hidden)))


def _one_hot(distance: Tensor, lower_bins: Tensor, upper_bins: Tensor) -> Tensor:
    r"""
    One-hot encode each distance into the half-open bin ``(lower, upper)`` it falls in.

    Args:
        distance: Distances to bin `[...]`.
        lower_bins: Lower edge of each bin `[num_bins]`.
        upper_bins: Upper edge of each bin `[num_bins]`.

    Returns:
        One-hot bin indicators `[..., num_bins]`, in ``distance``'s dtype.

    Examples:
        >>> import torch
        >>> lower = torch.tensor([0.0, 1.0, 2.0])
        >>> upper = torch.tensor([1.0, 2.0, 3.0])
        >>> _one_hot(torch.tensor([0.5, 1.5]), lower, upper).tolist()
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    """

    return ((distance[..., None] > lower_bins) & (distance[..., None] < upper_bins)).to(dtype=distance.dtype)


def _stack_outputs(outputs: list[StructurePredictionOutput]) -> StructurePredictionOutput:
    r"""
    Stack per-seed structure-prediction outputs along a new leading model-seed axis.

    Every present tensor field is stacked; fields left ``None`` by the model (e.g. training-only tensors in
    eval mode) stay ``None``. All seeds share the same field set, so the first output's fields drive the merge.
    """

    fields = [field.name for field in dataclasses.fields(StructurePredictionOutput)]
    merged: dict[str, Tensor | None] = {}
    for name in fields:
        values = [getattr(output, name) for output in outputs]
        merged[name] = None if values[0] is None else torch.stack(values, dim=0)
    return StructurePredictionOutput(**merged)  # type: ignore[arg-type]


@register_pytree_dataclass
@dataclass(slots=True)
class ODesignRepresentations:
    r"""
    Intermediate representations produced by the trunk.

    The ``single`` and ``pair`` fields carry the typed [`Single`][ofoldx.modules.representation.Single] /
    [`Pair`][ofoldx.modules.representation.Pair] carriers at this boundary (the same way ``Coords`` is used
    on the output types); unwrap ``.tensor`` before feeding the denoiser or the eager tensor heads. The
    carriers are pytree-registered, so this container flattens without a graph break under ``torch.compile``.

    Attributes:
        token_embedding: Token embedding before trunk updates.
        single: Final token single representation (a ``Single`` carrier).
        pair: Final token pair representation (a ``Pair`` carrier).
        relative_features: Relative-position feature bundle used by denoising.
        pair_mask: Boolean token-pair validity mask.
    """

    token_embedding: torch.Tensor
    single: Single
    pair: Pair
    relative_features: RelativePositionBuckets
    pair_mask: torch.Tensor

    def detach(self) -> ODesignRepresentations:
        r"""
        Return a gradient-detached copy (used for the no-grad confidence mini-rollout).

        Returns:
            Representation container with tensor fields detached.
        """

        return ODesignRepresentations(
            token_embedding=self.token_embedding.detach(),
            single=Single(self.single.tensor.detach()),
            pair=Pair(self.pair.tensor.detach()),
            relative_features=self.relative_features.detach(),
            pair_mask=self.pair_mask,
        )


@dataclass(slots=True)
class ODesignRecyclingState:
    r"""
    Init/context bundle passed through recycling.

    Attributes:
        token_embedding: Token embedding before trunk updates.
        single_init: Initial single representation for recycling.
        pair_init: Initial pair representation for recycling.
        pair_mask: Boolean token-pair validity mask.
    """

    token_embedding: torch.Tensor
    single_init: torch.Tensor
    pair_init: torch.Tensor
    pair_mask: torch.Tensor


@register_pytree_dataclass
@dataclass(slots=True, kw_only=True)
class ODesignInput(BaseInput):
    r"""
    Input tensors consumed by ODesignModel.

    Attributes:
        token_mask: Boolean token validity mask with shape ``(B, N_token)``.
        residue_index: Residue numbers for relative-position encoding.
        chain_index: Chain ids mapped to contiguous integer indices.
        entity_index: Entity ids shared by chains with the same sequence.
        symmetry_index: Symmetry-copy ids for relative-position encoding.
        token_index: Token ids in structure order.
        atom_features: Per-atom input features.
        atom_to_token: Token index for each atom.
        atom_mask: Boolean atom validity mask.
        reference_positions: Centered reference-conformer atom coords.
        reference_space_id: Reference-conformer group id for each atom.
        representative_atom_mask: Mask selecting one representative atom per token.
        atom_token_index: Per-token atom offset for each atom.
        token_to_rep_atom: Representative atom index for each token.
        token_features: Per-token input features.
        reference_mask: Boolean mask for valid reference-conformer atoms.
        reference_charges: Per-atom reference formal charges.
        target_positions: Optional experimental atom coords for training.
        target_mask: Optional experimental-coordinate validity mask.
        molecule_type: Optional molecule type id per token.
        frame_atom_index: Optional frame atom indices for frame-aligned losses.
        frame_mask: Optional frame validity mask.
        token_bonds: Optional token-level covalent adjacency.
        contact_constraints: Optional per-token-pair contact-constraint categories (reserved for the
            contact/pocket constraint path; not consumed by the base model).
        constraint_distogram: Optional ``(B, N_token, N_token, 1)`` pairwise distance constraints
            (``0`` = unconstrained). Consumed by [`ConstraintTemplateEmbedder`][] each recycling cycle
            when the constraint embedder is enabled.
        msa_features: Optional prebuilt MSA features.
        msa_mask: Optional MSA validity mask.
        template_features: Optional template feature bundle.
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
    contact_constraints: Tensor | None = None
    constraint_distogram: Tensor | None = None
    msa_features: Tensor | None = None
    msa_mask: Tensor | None = None
    template_features: TemplateFeatures | None = None


@dataclass(slots=True)
class ConfidenceOutput:
    r"""
    Structured confidence logits.

    Attributes:
        plddt_logits: Per-token lDDT logits.
        pae_logits: Pairwise aligned-error logits.
        pde_logits: Pairwise distance-error logits.
        resolved_logits: Per-atom resolved logits.
    """

    plddt_logits: Tensor
    pae_logits: Tensor
    pde_logits: Tensor
    resolved_logits: Tensor


# ``ODesignInput`` is defined below the model classes, so bind the class-level ``input_class`` here (once the
# dataclass exists). ``ODesignDesigner`` inherits it, enabling the ``model(**tensors)`` keyword call path.
ODesignModel.input_class = ODesignInput
