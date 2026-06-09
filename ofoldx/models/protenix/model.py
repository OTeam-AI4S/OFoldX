r"""
Protenix v0.5 constraint predictor: the AF3-shaped trunk plus the constraint pair-bias path, sampler, and heads.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

import torch
import torch.nn.functional as F
from oops import LayerNorm
from torch import Tensor, nn

from ...modules.atom import AtomToTokenEncoder, LocalAtomGeometry, TokenToAtomBroadcast
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
from ...modules.trunk import MsaTrunk, PairformerStack
from ..contract import BaseInput, StructurePredictionOutput
from ..mixin import ModelMixin
from ..registry import MODELS, STRUCTURE_PREDICTION_MODELS
from .config import (
    ProtenixConfig,
    ProtenixConstraintContactAtomConfig,
    ProtenixConstraintContactConfig,
    ProtenixConstraintEmbedderConfig,
    ProtenixConstraintPocketConfig,
)
from .guidance import (
    ProtenixGuidanceFeatures,
    ProtenixTFGEngine,
    TFGConfig,
    _guidance_features_from_batch,
    parse_tfg_config,
)


@MODELS.register("protenix")
@STRUCTURE_PREDICTION_MODELS.register("protenix")
class ProtenixModel(ModelMixin):
    r"""
    Protenix v0.5 structure-prediction model.

    The architecture is the AF3/Protenix base predictor plus the released constraint pair-bias path:
    pocket, token-contact, atom-contact, and substructure features are embedded and added to the
    initial pair representation before the recycling trunk, matching ``protenix/model/protenix.py`` in
    the official implementation. The trunk output feeds the EDM diffusion sampler and the
    distogram/confidence heads; inference optionally applies Protenix training-free guidance.

    Args:
        config: Model configuration selecting trunk, sampler, confidence, template, and constraint
            embedder hyperparameters.

    Examples:
        >>> from ofoldx import AutoModel, AutoProcessor  # doctest: +SKIP
        >>> processor = AutoProcessor.from_pretrained("oteam/protenix-base")  # doctest: +SKIP
        >>> model = AutoModel.from_pretrained("oteam/protenix-base").eval()  # doctest: +SKIP
        >>> batch = processor("complex.cif")  # doctest: +SKIP
        >>> output = model(batch)  # doctest: +SKIP
        >>> output.pred_coords.shape  # doctest: +SKIP
        torch.Size([1, 1, N_atom, 3])
    """

    config_class = ProtenixConfig

    def __init__(self, config: ProtenixConfig) -> None:
        super().__init__(config)
        self._recycling_steps = config.recycling_steps
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
        self.constraint_embedder = ProtenixConstraintEmbedder(config.constraint_embedder, pair_size=config.pair_size)
        self.post_init()

    def _build_coordinate_denoiser(self, config: ProtenixConfig) -> CoordinateDenoiser:
        r"""
        Construct the EDM coordinate denoiser from the denoiser config block.

        Args:
            config: The model configuration.

        Returns:
            The configured coordinate denoiser.
        """

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

    def _build_coordinate_sampler(self, config: ProtenixConfig) -> CoordinateEdmSampler:
        r"""
        Construct the EDM coordinate sampler from the sampler config block.

        Args:
            config: The model configuration.

        Returns:
            The configured EDM sampler wrapping the noise scheduler.
        """

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
        )

    def reset_parameters(self) -> None:
        r"""
        Zero-initialize the recycling and PAE/PDE projections and the constraint embedder.

        Restores the reference initialization: the recycle projections start at zero (so recycling
        adds nothing on the first cycle), the confidence PAE/PDE heads start at zero, and the
        constraint embedder is reset to its configured init.
        """

        nn.init.zeros_(self.pair_recycle_projection.weight)
        nn.init.zeros_(self.single_recycle_projection.weight)
        nn.init.zeros_(self.confidence_head.pair_projector.pae_projection.weight)
        nn.init.zeros_(self.confidence_head.pair_projector.pde_projection.weight)
        self.constraint_embedder.reset_parameters()

    def forward(
        self,
        batch: ProtenixInput,
        *,
        recycling_steps: int | None = None,
        num_samples: int = 1,
        num_sampling_steps: int | None = None,
        generator: torch.Generator | None = None,
        guidance_config: Mapping[str, Any] | None = None,
    ) -> StructurePredictionOutput:
        r"""
        Run the Protenix model on a typed input batch.

        Runs the recycling trunk, then branches on training mode: training returns the diffusion
        training targets alongside a rolled-out confidence prediction; inference runs the EDM sampling
        loop (optionally with Protenix training-free guidance) followed by the distogram and
        confidence heads.

        Args:
            batch: Typed input tensors (atom/token features, masks, indices, and optional constraint,
                template, MSA, and guidance features).
            recycling_steps: Number of trunk recycling cycles; defaults to the config value.
            num_samples: Number of coordinate samples to draw at inference.
            num_sampling_steps: Number of EDM sampling steps; defaults to the config value.
            generator: Optional RNG for reproducible noise/augmentation draws.
            guidance_config: Optional Protenix training-free-guidance config; guidance is applied only
                when its ``enable`` flag is set (inference only).

        Returns:
            A [`StructurePredictionOutput`][ofoldx.models.contract.StructurePredictionOutput] with
            predicted coordinates, distogram logits, and confidence (pLDDT/PAE/PDE/resolved) logits;
            training additionally attaches the denoised coordinates, augmented target, and noise level.

        Raises:
            ValueError: If guidance is requested during training, or if ``recycling_steps`` is not
                positive.

        Examples:
            >>> from ofoldx import AutoModel, AutoProcessor  # doctest: +SKIP
            >>> name = "oteam/protenix-base"  # doctest: +SKIP
            >>> model = AutoModel.from_pretrained(name).eval()  # doctest: +SKIP
            >>> batch = AutoProcessor.from_pretrained(name)("complex.cif")  # doctest: +SKIP
            >>> output = model(batch, num_samples=1)  # doctest: +SKIP
        """

        tfg_config = parse_tfg_config(guidance_config)
        if self.training and tfg_config.enable:
            raise ValueError("Protenix TFG guidance is an inference-time sampler feature")
        recycling_steps = self._recycling_steps if recycling_steps is None else recycling_steps
        if recycling_steps < 1:
            raise ValueError(f"recycling_steps must be positive for this model, got {recycling_steps}")
        representations = self._run_trunk(batch, recycling_steps=recycling_steps)

        if self.training:
            return self._forward_train(batch, representations=representations, generator=generator)

        pred_coords = self._predict_coords(
            batch,
            representations=representations,
            num_samples=num_samples,
            num_sampling_steps=num_sampling_steps,
            generator=generator,
            tfg_config=tfg_config,
        )
        return self._assemble_output(batch, representations=representations, confidence_coords=pred_coords)

    def _forward_train(
        self,
        batch: ProtenixInput,
        *,
        representations: ProtenixRepresentations,
        generator: torch.Generator | None,
    ) -> StructurePredictionOutput:
        r"""
        Build the training-mode output: single-step denoise plus a no-grad rollout for confidence.

        Args:
            batch: Typed input tensors.
            representations: Trunk representations from `_run_trunk`.
            generator: Optional RNG for reproducible noise/augmentation.

        Returns:
            The structure output with confidence logits plus attached diffusion training targets.
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
        batch: ProtenixInput,
        *,
        representations: ProtenixRepresentations,
        confidence_coords: Tensor,
    ) -> StructurePredictionOutput:
        r"""
        Assemble the structure output by running the distogram and confidence heads.

        Args:
            batch: Typed input tensors.
            representations: Trunk representations from `_run_trunk`.
            confidence_coords: Predicted coordinates the confidence head scores.

        Returns:
            The structure output with predicted coordinates and distogram/confidence logits.
        """

        confidence = self.confidence_head(
            representations.single,
            pair=representations.pair,
            token_embedding=representations.token_embedding,
            pred_coords=confidence_coords,
            token_to_rep_atom=batch.token_to_rep_atom,
            atom_to_token=batch.atom_to_token,
            atom_token_index=batch.atom_token_index,
        )
        return StructurePredictionOutput(
            pred_coords=confidence_coords,
            distogram_logits=self.distogram_head(representations.pair),
            plddt_logits=confidence.plddt_logits,
            pae_logits=confidence.pae_logits,
            pde_logits=confidence.pde_logits,
            resolved_logits=confidence.resolved_logits,
        )

    def _train_denoise(
        self,
        batch: ProtenixInput,
        *,
        representations: ProtenixRepresentations,
        num_samples: int,
        generator: torch.Generator | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        r"""
        Draw a training noise level, add noise to the augmented target, and denoise once.

        Args:
            batch: Typed input tensors; requires ``target_positions``.
            representations: Trunk representations from `_run_trunk`.
            num_samples: Number of augmented target samples to draw.
            generator: Optional RNG for reproducible augmentation/noise.

        Returns:
            ``(denoised_coords, augmented_target, noise_level)``.

        Raises:
            ValueError: If ``batch.target_positions`` is ``None``.
        """

        if batch.target_positions is None:
            raise ValueError("diffusion training requires batch.target_positions")
        target = batch.target_positions.to(dtype=representations.single.dtype)
        atom_mask = batch.atom_mask.to(device=target.device)
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
            single=representations.single,
            pair=representations.pair,
            relative_features=representations.relative_features,
            local_geometry=local_geometry,
        )
        denoised_coords = self.coordinate_denoiser(
            context.to_denoising_input(noisy_coords=noisy_coords, noise_level=noise_level)
        )
        return denoised_coords, augmented_target, noise_level

    def _run_trunk(
        self,
        batch: ProtenixInput,
        *,
        recycling_steps: int,
    ) -> ProtenixRepresentations:
        r"""
        Embed inputs, build the initial single/pair representations, and run the recycling trunk.

        Args:
            batch: Typed input tensors.
            recycling_steps: Number of recycling cycles to run.

        Returns:
            The trunk representations (token embedding, single, pair, relative features, pair mask).
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
        state = ProtenixRecyclingState(
            token_embedding=token_embedding,
            single_init=single_init,
            pair_init=pair_init,
            pair_mask=pair_mask,
        )
        single, pair = self._run_recycling(batch, state=state, recycling_steps=recycling_steps)

        return ProtenixRepresentations(
            token_embedding=token_embedding,
            single=single,
            pair=pair,
            relative_features=relative_features,
            pair_mask=pair_mask,
        )

    def _run_recycling(
        self,
        batch: ProtenixInput,
        *,
        state: ProtenixRecyclingState,
        recycling_steps: int,
    ) -> tuple[Tensor, Tensor]:
        r"""
        Iterate the recycling loop, enabling gradients only on the final cycle during training.

        Args:
            batch: Typed input tensors.
            state: Init/context bundle for the recycling trunk.
            recycling_steps: Number of recycling cycles to run.

        Returns:
            The final ``(single, pair)`` representations.
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

    def _initial_pair(
        self,
        single: Tensor,
        batch: ProtenixInput,
        relative_features: RelativePositionBuckets,
    ) -> Tensor:
        r"""
        Build the initial pair representation, adding the constraint pair-bias when features exist.

        Combines the left/right single projections, relative-position embedding, and token-bond
        projection, adds any enabled constraint pair-bias update, and masks invalid token pairs.

        Args:
            single: Initial single representation ``[..., N_token, single_size]``.
            batch: Typed input tensors (token bonds, masks, constraint features).
            relative_features: Bucketed relative-position features.

        Returns:
            The masked initial pair representation ``[..., N_token, N_token, pair_size]``.
        """

        left = self.pair_left_projection(single).unsqueeze(-2)
        right = self.pair_right_projection(single).unsqueeze(-3)
        if batch.token_bonds is None:
            token_bonds = single.new_zeros(*single.shape[:-1], single.shape[-2])
        else:
            token_bonds = batch.token_bonds.to(device=single.device, dtype=single.dtype)
            token_bonds = torch.broadcast_to(token_bonds, (*single.shape[:-1], single.shape[-2]))
        pair = left + right
        pair = pair + self.relative_position_embedder(relative_features)
        pair = pair + self.token_bond_projection(token_bonds.unsqueeze(-1))

        features = _constraint_features_from_batch(
            batch,
            channels=self.constraint_embedder.feature_channels,
            dtype=pair.dtype,
            device=pair.device,
        )
        if features is not None:
            pair = pair + self.constraint_embedder(
                features,
                pair_shape=pair.shape[:-1],
                dtype=pair.dtype,
                device=pair.device,
            )

        pair_mask = pair_mask_from_token_mask(batch.token_mask)
        return pair * pair_mask.unsqueeze(-1).to(dtype=pair.dtype)

    def _recycle_once(
        self,
        batch: ProtenixInput,
        *,
        state: ProtenixRecyclingState,
        single: Tensor,
        pair: Tensor,
    ) -> tuple[Tensor, Tensor]:
        r"""
        Run one recycling cycle: fold prior state, apply template/MSA trunks, then the Pairformer.

        Args:
            batch: Typed input tensors (optional template and MSA features).
            state: Init/context bundle for the recycling trunk.
            single: Single representation from the previous cycle.
            pair: Pair representation from the previous cycle.

        Returns:
            The updated ``(single, pair)`` representations.
        """

        pair = state.pair_init + self.pair_recycle_projection(self.pair_recycle_norm(pair))
        if batch.template_features is not None:
            pair = pair + self.template_embedder(pair, batch.template_features, pair_mask=state.pair_mask)
        if batch.msa_features is not None:
            _, pair = self.msa_trunk(
                batch.msa_features, pair, state.token_embedding, msa_mask=batch.msa_mask, pair_mask=state.pair_mask
            )
        single = state.single_init + self.single_recycle_projection(self.single_recycle_norm(single))
        single, pair = self.pairformer_stack(single, pair, pair_mask=state.pair_mask)
        return single, pair

    def _predict_coords(
        self,
        batch: ProtenixInput,
        *,
        representations: ProtenixRepresentations,
        num_samples: int,
        num_sampling_steps: int | None,
        generator: torch.Generator | None,
        tfg_config: TFGConfig | None = None,
    ) -> Tensor:
        r"""
        Sample predicted coordinates via the EDM sampler, dispatching to guidance when enabled.

        Args:
            batch: Typed input tensors.
            representations: Trunk representations from `_run_trunk`.
            num_samples: Number of coordinate samples to draw.
            num_sampling_steps: Number of EDM steps; defaults to the config value.
            generator: Optional RNG for reproducible sampling.
            tfg_config: Optional guidance config; guided sampling runs only when its ``enable`` is set.

        Returns:
            Predicted coordinates ``[..., N_sample, N_atom, 3]``.
        """

        local_geometry = self._build_denoising_local_geometry(batch)
        context = CoordinateDenoiseContext(
            atom_features=batch.atom_features,
            atom_to_token=batch.atom_to_token,
            reference_positions=batch.reference_positions,
            reference_charges=batch.reference_charges,
            atom_mask=batch.atom_mask,
            token_embedding=representations.token_embedding,
            single=representations.single,
            pair=representations.pair,
            relative_features=representations.relative_features,
            local_geometry=local_geometry,
        )
        num_sampling_steps = self._num_sampling_steps if num_sampling_steps is None else num_sampling_steps
        if tfg_config is None or not tfg_config.enable:
            return self.coordinate_sampler(
                self.coordinate_denoiser,
                context=context,
                num_samples=num_samples,
                num_sampling_steps=num_sampling_steps,
                generator=generator,
            )
        return self._predict_coords_with_guidance(
            batch,
            context=context,
            num_samples=num_samples,
            num_sampling_steps=num_sampling_steps,
            generator=generator,
            tfg_config=tfg_config,
        )

    def _predict_coords_with_guidance(
        self,
        batch: ProtenixInput,
        *,
        context: CoordinateDenoiseContext,
        num_samples: int,
        num_sampling_steps: int,
        generator: torch.Generator | None,
        tfg_config: TFGConfig,
    ) -> Tensor:
        r"""
        Run the guided EDM sampling loop using the Protenix training-free-guidance engine.

        Reimplements the sampler's churn/step schedule so each step is routed through the guidance
        engine, which applies the path shift, projection, and gradient-ascent updates.

        Args:
            batch: Typed input tensors carrying (or defaulting) the guidance geometry features.
            context: Denoiser context bundling the trunk representations and atom features.
            num_samples: Number of coordinate samples to draw.
            num_sampling_steps: Number of EDM steps.
            generator: Optional RNG for reproducible sampling.
            tfg_config: The resolved (enabled) guidance configuration.

        Returns:
            Predicted coordinates ``[..., N_sample, N_atom, 3]``.
        """

        features = _guidance_features_from_batch(batch, device=context.device, dtype=context.dtype)
        engine = ProtenixTFGEngine(tfg_config, device=context.device, dtype=context.dtype)
        sampler = self.coordinate_sampler
        noise_schedule = sampler.scheduler(
            num_sampling_steps=num_sampling_steps, device=context.device, dtype=context.dtype
        )
        coords = sampler._initial_coords(
            context=context,
            num_samples=num_samples,
            noise_level=noise_schedule[0],
            initial_coords=None,
            generator=generator,
        )
        num_transitions = noise_schedule.size(0) - 1
        for step_index, (sigma_prev, sigma_next) in enumerate(zip(noise_schedule[:-1], noise_schedule[1:])):
            noise = None
            if sampler.sample_noise_before_augmentation:
                noise = torch.empty_like(coords).normal_(generator=generator)
            if sampler.center_random_augmentation:
                coords = sampler._center_random_augmentation(
                    coords,
                    num_samples=1,
                    generator=generator,
                ).squeeze(dim=-3)
            if noise is None:
                noise = torch.empty_like(coords).normal_(generator=generator)

            churn = torch.where(
                sigma_next > sampler.min_churn_noise_level,
                sigma_next.new_tensor(sampler.churn_rate),
                sigma_next.new_zeros(()),
            )
            sigma_hat = sigma_prev * (churn + 1)
            delta_noise = torch.sqrt((sigma_hat**2 - sigma_prev**2).clamp_min(0.0))
            noisy_coords = coords + sampler.noise_scale * delta_noise * noise
            batch_shape = coords.shape[:-3]
            noise_level = (
                sigma_hat.reshape((1,) * (len(batch_shape) + 1)).expand(*batch_shape, num_samples).to(coords.dtype)
            )
            next_noise_level = (
                sigma_next.reshape((1,) * (len(batch_shape) + 1)).expand(*batch_shape, num_samples).to(coords.dtype)
            )
            coords = engine.step(
                self.coordinate_denoiser,
                context=context,
                coords=noisy_coords,
                noise_level=noise_level,
                next_noise_level=next_noise_level,
                step_scale=sampler._step_scale(sigma_next),
                step_index=step_index,
                num_diffusion_steps=num_transitions,
                features=features,
                generator=generator,
            )
        return coords

    def _build_input_local_geometry(self, batch: ProtenixInput) -> LocalAtomGeometry:
        r"""
        Build the input-embedder local atom geometry from the batch reference positions.

        Args:
            batch: Typed input tensors.

        Returns:
            Local atom geometry laid out for the input atom encoder.
        """

        reference_positions = batch.reference_positions.to(device=batch.atom_features.device)
        return self.input_embedder.build_local_geometry(reference_positions, batch.reference_space_id)

    def _build_denoising_local_geometry(self, batch: ProtenixInput) -> LocalAtomGeometry:
        r"""
        Build the denoiser local atom geometry from the batch reference positions.

        Args:
            batch: Typed input tensors.

        Returns:
            Local atom geometry laid out for the coordinate denoiser.
        """

        reference_positions = batch.reference_positions.to(device=batch.atom_features.device)
        return self.coordinate_denoiser.build_local_geometry(reference_positions, batch.reference_space_id)


class InputEmbedder(nn.Module):
    r"""
    Build token input embeddings from packed atom and token features.

    Runs the atom-to-token encoder to produce the leading token-single channels, then concatenates
    the trailing token-feature tail so the output width matches the token feature contract.

    Args:
        single_size: Width of the atom-derived token-single channel.
        token_feature_size: Expected width of the packed per-token features.
        atom_feature_size: Width of the per-atom input features consumed by the atom encoder.
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
        Return token embeddings concatenating atom-derived channels and the token-feature tail.

        Args:
            atom_features: Per-atom input features ``[..., N_atom, atom_feature_size]``.
            reference_positions: Reference conformer atom positions ``[..., N_atom, 3]``.
            reference_charges: Per-atom reference charges ``[..., N_atom]`` (arcsinh-compressed here).
            reference_mask: Per-atom reference validity mask ``[..., N_atom]``.
            token_features: Packed per-token features ``[..., N_token, token_feature_size]``.
            atom_to_token: Token index per atom ``[..., N_atom]``.
            num_tokens: Number of tokens.
            atom_mask: Optional per-atom validity mask ``[..., N_atom]``.
            token_mask: Optional per-token validity mask ``[..., N_token]``; zeroes masked rows.
            local_geometry: Optional precomputed local atom geometry.

        Returns:
            Token embeddings ``[..., N_token, single_size + tail]`` with the token-feature tail appended.
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
            reference_positions: Reference conformer atom positions ``[..., N_atom, 3]``.
            reference_space_id: Per-atom reference-space grouping id ``[..., N_atom]``.

        Returns:
            Local atom geometry for the input atom encoder.
        """

        return self.atom_encoder.build_local_geometry(reference_positions, reference_space_id)

    def _token_tail(self, token_single: Tensor, token_features: Tensor) -> Tensor:
        r"""
        Slice the trailing token-feature channels appended after the atom-derived single channels.

        Args:
            token_single: Atom-derived token-single channels ``[..., N_token, single_size]``.
            token_features: Packed per-token features ``[..., N_token, token_feature_size]``.

        Returns:
            The token-feature tail ``[..., N_token, token_feature_size - single_size]`` on the
            device/dtype of ``token_single``.

        Raises:
            ValueError: If ``token_features`` has the wrong last dimension or mismatched leading dims.
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


class ProtenixConstraintEmbedder(nn.Module):
    r"""
    Aggregate constraint pair-bias embedder from Protenix v0.5.

    Holds the enabled pocket, token-contact, atom-contact, and substructure sub-embedders, projecting
    each constraint feature into the pair channel and summing the results into a single additive pair
    update.

    Args:
        config: Constraint embedder configuration selecting sub-embedders and the init method.
        pair_size: Width of the pair channel the sub-embedders project into; must equal
            ``config.constraint_pair_size``.

    Raises:
        ValueError: If ``config.constraint_pair_size`` does not equal ``pair_size``, or the configured
            ``initialize_method`` is not one of ``"zero"``, ``"kaiming"``, ``"default"``.
    """

    def __init__(self, config: ProtenixConstraintEmbedderConfig, pair_size: int) -> None:
        super().__init__()
        if int(config.constraint_pair_size) != int(pair_size):
            raise ValueError(
                f"constraint_embedder.constraint_pair_size ({config.constraint_pair_size}) "
                f"must match pair_size ({pair_size})"
            )
        self.initialize_method = config.initialize_method
        if self.initialize_method not in {"zero", "kaiming", "default"}:
            raise ValueError("constraint_embedder.initialize_method must be 'zero', 'kaiming', or 'default'")

        self.pocket_embedder_config = config.pocket_embedder
        self.contact_embedder_config = config.contact_embedder
        self.contact_atom_embedder_config = config.contact_atom_embedder
        self.substructure_embedder_config = config.substructure_embedder
        self.feature_channels = {
            "pocket": int(config.pocket_embedder.pair_input_size),
            "contact": int(config.contact_embedder.pair_input_size),
            "contact_atom": int(config.contact_atom_embedder.pair_input_size),
            "substructure": int(config.substructure_embedder.n_classes),
        }

        self.pocket_z_embedder: nn.Linear | None = _linear_if_enabled(config.pocket_embedder, pair_size)
        self.contact_z_embedder: nn.Linear | None = _linear_if_enabled(config.contact_embedder, pair_size)
        self.contact_atom_z_embedder: nn.Linear | None = _linear_if_enabled(config.contact_atom_embedder, pair_size)
        self.substructure_z_embedder: ProtenixConstraintSubstructureEmbedder | None
        if config.substructure_embedder.enable:
            self.substructure_z_embedder = ProtenixConstraintSubstructureEmbedder(
                n_classes=config.substructure_embedder.n_classes,
                c_pair_dim=pair_size,
                architecture=cast(Literal["mlp", "transformer"], config.substructure_embedder.architecture.lower()),
                hidden_size=config.substructure_embedder.hidden_size,
                n_layers=config.substructure_embedder.n_layers,
                dropout=config.substructure_embedder.dropout,
            )
        else:
            self.substructure_z_embedder = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        r"""
        Re-initialize the constraint projection weights per the configured method.

        Zeros every linear weight for ``"zero"`` init (the reference default, so constraints start
        with no effect) or applies Kaiming-uniform for ``"kaiming"``; ``"default"`` leaves the
        PyTorch defaults untouched. Biases are always zeroed.
        """

        if self.initialize_method == "default":
            return
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if self.initialize_method == "zero":
                    nn.init.zeros_(module.weight)
                else:
                    nn.init.kaiming_uniform_(module.weight, a=5**0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        features: ProtenixConstraintFeatures,
        *,
        pair_shape: torch.Size | tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        r"""
        Project every enabled constraint feature into the pair channel and sum the updates.

        Each enabled sub-embedder broadcasts its feature to the pair shape and projects it; the
        additive updates are summed. At least one sub-embedder must be enabled.

        Args:
            features: Constraint feature bundle (pocket/contact/contact-atom/substructure tensors).
            pair_shape: Target pair-tensor shape without the channel axis, e.g. ``[..., N, N]``.
            dtype: Dtype the features are cast to before projection.
            device: Device the features are moved to before projection.

        Returns:
            The summed constraint pair update ``[..., N, N, pair_size]``.

        Raises:
            ValueError: If no constraint sub-embedder is enabled.
        """

        updates: list[Tensor] = []
        if self.pocket_z_embedder is not None:
            updates.append(
                self.pocket_z_embedder(
                    _feature(
                        features.pocket,
                        pair_shape,
                        self.pocket_z_embedder.in_features,
                        "pocket",
                        dtype,
                        device,
                    )
                )
            )
        if self.contact_z_embedder is not None:
            updates.append(
                self.contact_z_embedder(
                    _feature(
                        features.contact,
                        pair_shape,
                        self.contact_z_embedder.in_features,
                        "contact",
                        dtype,
                        device,
                    )
                )
            )
        if self.contact_atom_z_embedder is not None:
            updates.append(
                self.contact_atom_z_embedder(
                    _feature(
                        features.contact_atom,
                        pair_shape,
                        self.contact_atom_z_embedder.in_features,
                        "contact_atom",
                        dtype,
                        device,
                    )
                )
            )
        if self.substructure_z_embedder is not None:
            updates.append(
                self.substructure_z_embedder(
                    _feature(
                        features.substructure,
                        pair_shape,
                        self.substructure_z_embedder.n_classes,
                        "substructure",
                        dtype,
                        device,
                    )
                )
            )
        if not updates:
            raise ValueError("at least one constraint sub-embedder must be enabled to embed constraint features")
        return _add_update(updates)


class ProtenixConstraintSubstructureEmbedder(nn.Module):
    r"""
    Pairwise substructure-distance feature embedder from Protenix v0.5.

    Projects a one-hot binned substructure-distance feature into the pair channel using either an MLP
    or a small transformer over the flattened token-pair sequence.

    Args:
        n_classes: Number of distance-bin classes in the input feature.
        c_pair_dim: Output pair channel width.
        architecture: Sub-embedder architecture, ``"transformer"`` or ``"mlp"``.
        hidden_size: Width of the internal hidden channel.
        n_layers: Number of MLP/transformer layers.
        dropout: Dropout probability inside the sub-embedder.

    Raises:
        ValueError: If ``architecture`` is neither ``"mlp"`` nor ``"transformer"``.
    """

    def __init__(
        self,
        *,
        n_classes: int,
        c_pair_dim: int,
        architecture: Literal["mlp", "transformer"] = "transformer",
        hidden_size: int = 128,
        n_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.architecture = architecture.lower()
        if self.architecture == "mlp":
            layers: list[nn.Module] = [
                nn.Linear(n_classes, hidden_size, bias=False),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            for _ in range(max(n_layers - 2, 0)):
                layers.extend(
                    [
                        nn.Linear(hidden_size, hidden_size, bias=False),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                    ]
                )
            layers.append(nn.Linear(hidden_size, c_pair_dim, bias=False))
            self.network = nn.Sequential(*layers)
        elif self.architecture == "transformer":
            self.input_proj = nn.Linear(n_classes, hidden_size, bias=False)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=4,
                dim_feedforward=hidden_size * 4,
                dropout=dropout,
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.output_proj = nn.Linear(hidden_size, c_pair_dim, bias=False)
        else:
            raise ValueError(f"unknown substructure architecture: {architecture!r}")

    def forward(self, x: Tensor) -> Tensor:
        r"""
        Project one-hot substructure features to a pair-channel bias.

        The MLP architecture applies pointwise; the transformer flattens the token-pair grid into a
        sequence, attends over it, and reshapes back.

        Args:
            x: Substructure feature ``[..., N_token, N_token, n_classes]``.

        Returns:
            Pair bias ``[..., N_token, N_token, c_pair_dim]``.
        """

        if self.architecture == "mlp":
            return self.network(x)

        original_shape = x.shape
        x = x.reshape(-1, original_shape[-3], original_shape[-2], original_shape[-1])
        batch_size, num_tokens, _, _ = x.shape
        hidden = self.input_proj(x).reshape(batch_size, num_tokens * num_tokens, -1)
        hidden = self.transformer(hidden)
        return self.output_proj(hidden).reshape(*original_shape[:-1], -1)


class TrainingNoiseSampler(nn.Module):
    r"""
    Sample EDM training noise levels from a log-normal distribution.

    Mirrors the AlphaFold 3 family training noise schedule: draw
    ``sigma = exp(noise_log_mean + noise_log_std * N(0, 1)) * sigma_data``. Each training sample gets an
    independent noise level, so the denoiser learns to denoise across the whole sigma range
    in a single step (no reverse-sampling loop during training).

    Args:
        noise_log_mean: Mean of the log-normal noise level.
        noise_log_std: Standard deviation of the log-normal noise level.
        sigma_data: Data-scale factor multiplying the sampled noise level.
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
        Return a tensor of independent log-normal noise levels.

        Args:
            size: Output shape; one independent noise level per element.
            device: Target device for the sampled tensor.
            dtype: Target dtype for the sampled tensor.
            generator: Optional RNG for reproducible sampling.

        Returns:
            Noise levels of shape ``size``.
        """

        standard_normal = torch.randn(size, device=device, dtype=dtype, generator=generator)
        return (standard_normal * self.noise_log_std + self.noise_log_mean).exp() * self.sigma_data


class ConfidenceHead(nn.Module):
    r"""
    Predict per-atom and pairwise confidence logits from the trunk and predicted coordinates.

    Embeds the predicted representative-atom distances into the pair state, runs a small Pairformer
    stack, then projects to per-atom pLDDT/resolved logits and pairwise PAE/PDE logits. Each
    coordinate sample is scored independently and the results are stacked.

    Args:
        single_size: Width of the token-single representation.
        pair_size: Width of the token-pair representation.
        token_feature_size: Width of the token features projected into the pair embedder.
        distance_bin_start: Start of the representative-distance one-hot bins (Angstrom).
        distance_bin_end: End of the representative-distance one-hot bins (Angstrom).
        distance_bin_step: Width of each representative-distance bin (Angstrom).
        num_plddt_bins: Number of pLDDT output bins.
        num_resolved_bins: Number of resolved-atom output bins.
        max_atoms_per_token: Maximum atoms per token for the per-offset weight tables.
        num_pae_bins: Number of PAE output bins.
        num_pde_bins: Number of PDE output bins.
        num_blocks: Number of Pairformer blocks in the confidence stack.
        num_pair_attention_heads: Attention heads for the pair-biased single attention.
        triangle_multiplication_hidden_size: Hidden width of the triangle-multiplication updates.
        triangle_attention_hidden_size: Per-head width of the triangle attention.
        num_triangle_attention_heads: Number of triangle-attention heads.
        expansion: Feed-forward expansion factor inside the Pairformer.
        dropout: Dropout probability inside the Pairformer.
        blocks_per_checkpoint: Number of blocks per gradient-checkpoint segment, or ``None``.
        operator_backend: Operator backend selector for the Pairformer.
        single_max_abs: Optional clamp applied to the single representation before scoring.
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

        Gathers each token's representative-atom coordinate per sample, scores every sample through
        the confidence stack, and stacks the per-sample logits along a new sample axis.

        Args:
            single: Token-single representation ``[..., N_token, single_size]``.
            pair: Token-pair representation ``[..., N_token, N_token, pair_size]``.
            token_embedding: Token features ``[..., N_token, token_feature_size]``.
            pred_coords: Predicted coordinates ``[..., N_sample, N_atom, 3]``.
            token_to_rep_atom: Per-token representative global atom index ``[..., N_token]``.
            atom_to_token: Token index per atom ``[..., N_atom]``.
            atom_token_index: Within-token atom offset per atom ``[..., N_atom]``.
            pair_mask: Optional token-pair validity mask ``[..., N_token, N_token]``.

        Returns:
            A [`ConfidenceOutput`][ofoldx.models.protenix.model.ConfidenceOutput] with
            pLDDT, PAE, PDE, and resolved logits stacked over the sample axis.
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
        Score a single coordinate sample through the confidence stack and projectors.

        Args:
            single: Token-single representation ``[..., N_token, single_size]``.
            pair: Token-pair representation ``[..., N_token, N_token, pair_size]``.
            token_embedding: Token features ``[..., N_token, token_feature_size]``.
            pair_mask: Optional token-pair validity mask ``[..., N_token, N_token]``.
            representative_coords: Representative-atom coords for this sample ``[..., N_token, 3]``.
            atom_to_token: Token index per atom ``[..., N_atom]``.
            atom_token_index: Within-token atom offset per atom ``[..., N_atom]``.

        Returns:
            The per-sample confidence logits.
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

    Adds left/right token-embedding projections and both one-hot-binned and raw projections of the
    representative-atom pairwise distances to the incoming pair state.

    Args:
        pair_size: Width of the pair channel.
        token_feature_size: Width of the token features projected into the pair channel.
        distance_bin_start: Start of the distance one-hot bins (Angstrom).
        distance_bin_end: End of the distance one-hot bins (Angstrom).
        distance_bin_step: Width of each distance bin (Angstrom).
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
        Add token-projection and representative-distance channels to the pair state.

        Adds left/right token-embedding projections plus a one-hot binned distance projection
        and a raw-distance projection of the representative-atom pairwise distances.

        Args:
            pair: Token pair state ``(..., N_token, N_token, pair_size)``.
            token_embedding: Token features ``(..., N_token, token_feature_size)``.
            representative_coords: Representative-atom coords ``(..., N_token, 3)``.

        Returns:
            Updated pair state ``(..., N_token, N_token, pair_size)``.
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

    Broadcasts token singles to atoms, then contracts each atom's normalized features against a
    per-within-token-offset weight matrix to produce pLDDT and resolved logits.

    Args:
        single_size: Width of the token-single representation.
        num_plddt_bins: Number of pLDDT output bins.
        num_resolved_bins: Number of resolved-atom output bins.
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
        Zero-initialize the per-atom-offset pLDDT and resolved weight tensors.
        """

        nn.init.zeros_(self.plddt_weight)
        nn.init.zeros_(self.resolved_weight)

    def forward(self, single: Tensor, atom_to_token: Tensor, atom_token_index: Tensor) -> tuple[Tensor, Tensor]:
        r"""
        Broadcast token singles to atoms and project to per-atom pLDDT/resolved logits.

        Broadcasts the token single representation to atoms, then contracts each atom's
        normalized features against the weight matrix selected by its within-token offset.

        Args:
            single: Token single representation ``(..., N_token, single_size)``.
            atom_to_token: Token index per atom ``(..., N_atom)``.
            atom_token_index: Within-token atom offset per atom ``(..., N_atom)``.

        Returns:
            ``(plddt_logits, resolved_logits)`` with shapes ``(..., N_atom, num_plddt_bins)`` and
            ``(..., N_atom, num_resolved_bins)``.
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

    Produces PAE logits from the normalized pair state and PDE logits from the normalized symmetrized
    pair state.

    Args:
        pair_size: Width of the pair channel.
        num_pae_bins: Number of PAE output bins.
        num_pde_bins: Number of PDE output bins.
    """

    def __init__(self, pair_size: int = 128, num_pae_bins: int = 64, num_pde_bins: int = 64) -> None:
        super().__init__()
        self.pae_norm = LayerNorm(pair_size)
        self.pde_norm = LayerNorm(pair_size)
        self.pae_projection = nn.Linear(pair_size, num_pae_bins, bias=False)
        self.pde_projection = nn.Linear(pair_size, num_pde_bins, bias=False)

    def forward(self, pair: Tensor) -> tuple[Tensor, Tensor]:
        r"""
        Project the pair state to PAE and (symmetrized) PDE logits.

        PAE is projected from the normalized pair state; PDE is projected from the normalized
        symmetrized pair state ``pair + pair.transpose(-2, -3)``.

        Args:
            pair: Token pair state ``(..., N_token, N_token, pair_size)``.

        Returns:
            ``(pae_logits, pde_logits)`` with shapes ``(..., N_token, N_token, num_pae_bins)`` and
            ``(..., N_token, N_token, num_pde_bins)``.
        """

        pae_logits = self.pae_projection(self.pae_norm(pair))
        pde_logits = self.pde_projection(self.pde_norm(pair + pair.transpose(-2, -3)))
        return pae_logits, pde_logits


def _one_hot(distance: Tensor, lower_bins: Tensor, upper_bins: Tensor) -> Tensor:
    r"""
    One-hot-encode distances into half-open ``(lower, upper)`` bins.

    Args:
        distance: Pairwise distances ``[..., N, N]``.
        lower_bins: Per-bin lower edges ``[num_bins]``.
        upper_bins: Per-bin upper edges ``[num_bins]``.

    Returns:
        One-hot bin membership ``[..., N, N, num_bins]`` in the distance dtype.
    """

    return ((distance[..., None] > lower_bins) & (distance[..., None] < upper_bins)).to(dtype=distance.dtype)


def _linear_if_enabled(
    config: ProtenixConstraintPocketConfig | ProtenixConstraintContactConfig | ProtenixConstraintContactAtomConfig,
    pair_size: int,
) -> nn.Linear | None:
    r"""
    Build a bias-free projection for a constraint sub-embedder when it is enabled.

    Args:
        config: A pocket/contact/contact-atom sub-embedder config with ``enable`` and
            ``pair_input_size``.
        pair_size: Output pair channel width.

    Returns:
        A linear projecting ``pair_input_size -> pair_size``, or ``None`` when disabled.
    """

    if not config.enable:
        return None
    return nn.Linear(config.pair_input_size, pair_size, bias=False)


def _constraint_features_from_batch(
    batch: ProtenixInput,
    *,
    channels: Mapping[str, int],
    dtype: torch.dtype,
    device: torch.device,
) -> ProtenixConstraintFeatures | None:
    r"""
    Resolve a constraint feature bundle from a batch, or ``None`` when no constraints are present.

    Prefers an explicit ``constraint_feature`` bundle; otherwise assembles one from the direct
    ``*_constraints`` tensors, filling absent channels with minimal zero tensors. Returns ``None`` when
    neither source provides any constraint.

    Args:
        batch: Typed input tensors.
        channels: Per-channel widths keyed by constraint name.
        dtype: Dtype for synthesized zero placeholders.
        device: Device for synthesized zero placeholders.

    Returns:
        The constraint feature bundle, or ``None`` when no constraints are supplied.
    """

    if batch.constraint_feature is not None:
        if isinstance(batch.constraint_feature, ProtenixConstraintFeatures):
            return batch.constraint_feature
        return ProtenixConstraintFeatures.from_mapping(batch.constraint_feature)

    direct = (
        batch.pocket_constraints,
        batch.contact_constraints,
        batch.contact_atom_constraints,
        batch.substructure_constraints,
    )
    if not any(value is not None for value in direct):
        return None

    # Return raw tensors (or minimal per-channel zeros for absent constraints); the embedder
    # performs the single authoritative dtype/device cast and broadcast to the pair shape.
    return ProtenixConstraintFeatures(
        pocket=_feature_or_zeros(batch.pocket_constraints, channels["pocket"], dtype, device),
        contact=_feature_or_zeros(batch.contact_constraints, channels["contact"], dtype, device),
        contact_atom=_feature_or_zeros(batch.contact_atom_constraints, channels["contact_atom"], dtype, device),
        substructure=_feature_or_zeros(batch.substructure_constraints, channels["substructure"], dtype, device),
    )


def _feature(
    value: Tensor,
    pair_shape: torch.Size | tuple[int, ...],
    channels: int,
    name: str,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    r"""
    Cast a constraint feature and broadcast it to the target pair shape.

    Args:
        value: The raw constraint feature tensor.
        pair_shape: Target pair shape without the channel axis, e.g. ``[..., N, N]``.
        channels: Expected number of feature channels.
        name: Constraint name, used in error messages.
        dtype: Target dtype.
        device: Target device.

    Returns:
        The feature broadcast to ``[*pair_shape, channels]``.

    Raises:
        ValueError: If the feature's channel count is wrong or it cannot broadcast to the pair shape.
    """

    value = value.to(device=device, dtype=dtype)
    if value.shape[-1] != channels:
        raise ValueError(f"constraint_feature['{name}'] must have last dimension {channels}, got {value.shape[-1]}")
    target_shape = (*tuple(pair_shape), channels)
    try:
        return torch.broadcast_to(value, target_shape)
    except RuntimeError as error:
        raise ValueError(
            f"constraint_feature['{name}'] shape {tuple(value.shape)} cannot broadcast to {target_shape}"
        ) from error


def _feature_or_zeros(
    value: Tensor | None,
    channels: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    r"""
    Return the feature tensor, or a broadcastable zero placeholder when it is absent.

    Args:
        value: The raw feature tensor, or ``None``.
        channels: Number of channels for the zero placeholder.
        dtype: Dtype for the zero placeholder.
        device: Device for the zero placeholder.

    Returns:
        The feature, or a ``[1, 1, channels]`` zero tensor.
    """

    if value is None:
        return torch.zeros((1, 1, channels), dtype=dtype, device=device)
    return value


def _add_update(updates: list[Tensor]) -> Tensor:
    r"""
    Sum a non-empty list of pair-update tensors.

    Args:
        updates: The per-sub-embedder pair updates to sum.

    Returns:
        The elementwise sum of the updates.
    """

    return functools.reduce(torch.add, updates)


def _move_tensor(
    tensor: Tensor,
    device: torch.device | str | None,
    dtype: torch.dtype | None,
    non_blocking: bool,
    copy: bool,
) -> Tensor:
    r"""
    Move a tensor to a device/dtype, casting only floating-point tensors.

    Args:
        tensor: The tensor to move.
        device: Target device, or ``None`` to leave unchanged.
        dtype: Target floating dtype; ignored for integer tensors.
        non_blocking: Whether to request an asynchronous copy.
        copy: Whether to force a copy.

    Returns:
        The moved tensor.
    """

    cast_dtype = dtype if tensor.is_floating_point() else None
    return tensor.to(device=device, dtype=cast_dtype, non_blocking=non_blocking, copy=copy)


@dataclass(slots=True)
class ProtenixRepresentations:
    r"""
    Trunk representations produced by [`ProtenixModel`][].

    Attributes:
        token_embedding: Token input embedding ``[..., N_token, token_feature_size]``.
        single: Trunk single representation ``[..., N_token, single_size]``.
        pair: Trunk pair representation ``[..., N_token, N_token, pair_size]``.
        relative_features: Bucketed relative-position features reused by the denoiser.
        pair_mask: Token-pair validity mask ``[..., N_token, N_token]``.
    """

    token_embedding: Tensor
    single: Tensor
    pair: Tensor
    relative_features: RelativePositionBuckets
    pair_mask: Tensor

    def detach(self) -> ProtenixRepresentations:
        r"""
        Return a copy with every tensor detached from autograd.

        Used for the no-grad confidence rollout during training so the rollout does not build graph.

        Returns:
            A new representations bundle with detached tensors.
        """

        return type(self)(
            token_embedding=self.token_embedding.detach(),
            single=self.single.detach(),
            pair=self.pair.detach(),
            relative_features=self.relative_features.detach(),
            pair_mask=self.pair_mask.detach(),
        )


@dataclass(slots=True)
class ProtenixRecyclingState:
    r"""
    Init/context bundle for the Protenix recycling trunk.

    Attributes:
        token_embedding: Token input embedding ``[..., N_token, token_feature_size]``.
        single_init: Initial single representation ``[..., N_token, single_size]``.
        pair_init: Initial pair representation ``[..., N_token, N_token, pair_size]``.
        pair_mask: Token-pair validity mask ``[..., N_token, N_token]``.
    """

    token_embedding: Tensor
    single_init: Tensor
    pair_init: Tensor
    pair_mask: Tensor


@dataclass(slots=True, kw_only=True)
class ProtenixConstraintFeatures:
    r"""
    Constraint feature tensor bundle mirroring the official Protenix v0.5 ``constraint_feature`` dict.

    The keys and shapes mirror the reference dictionary: ``pocket`` has shape ``[..., N, N, 1]``,
    ``contact`` and ``contact_atom`` have shape ``[..., N, N, 2]``, and ``substructure`` has shape
    ``[..., N, N, 4]``.

    Attributes:
        pocket: Pocket constraint feature ``[..., N, N, 1]``.
        contact: Token-contact constraint feature ``[..., N, N, 2]`` (min/max distance).
        contact_atom: Atom-contact constraint feature ``[..., N, N, 2]`` (min/max distance).
        substructure: One-hot binned substructure-distance feature ``[..., N, N, 4]``.
    """

    pocket: Tensor
    contact: Tensor
    contact_atom: Tensor
    substructure: Tensor

    @classmethod
    def from_mapping(cls, values: Mapping[str, Tensor]) -> ProtenixConstraintFeatures:
        r"""
        Build a feature bundle from the official ``constraint_feature`` key names.

        Args:
            values: Mapping with ``pocket``, ``contact``, ``contact_atom``, and ``substructure`` keys.

        Returns:
            The constructed feature bundle.

        Raises:
            ValueError: If any required key is missing.
        """

        missing = {"pocket", "contact", "contact_atom", "substructure"} - set(values)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"constraint_feature is missing required tensor(s): {names}")
        return cls(
            pocket=values["pocket"],
            contact=values["contact"],
            contact_atom=values["contact_atom"],
            substructure=values["substructure"],
        )

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        non_blocking: bool = False,
        copy: bool = False,
    ) -> ProtenixConstraintFeatures:
        r"""
        Return a copy with every feature tensor moved to the given device/dtype.

        Only floating-point tensors are cast to ``dtype``; integer tensors keep their dtype.

        Args:
            device: Target device, or ``None`` to leave unchanged.
            dtype: Target floating dtype, or ``None`` to leave unchanged.
            non_blocking: Whether to request an asynchronous copy.
            copy: Whether to force a copy.

        Returns:
            A new bundle with the moved tensors.
        """

        return type(self)(
            pocket=_move_tensor(self.pocket, device, dtype, non_blocking, copy),
            contact=_move_tensor(self.contact, device, dtype, non_blocking, copy),
            contact_atom=_move_tensor(self.contact_atom, device, dtype, non_blocking, copy),
            substructure=_move_tensor(self.substructure, device, dtype, non_blocking, copy),
        )

    def as_dict(self) -> dict[str, Tensor]:
        r"""
        Return the bundle as the reference Protenix ``constraint_feature`` dictionary.

        Returns:
            Mapping with ``pocket``, ``contact``, ``contact_atom``, and ``substructure`` tensors.
        """

        return {
            "pocket": self.pocket,
            "contact": self.contact,
            "contact_atom": self.contact_atom,
            "substructure": self.substructure,
        }


@dataclass(slots=True, kw_only=True)
class ProtenixInput(BaseInput):
    r"""
    Input tensors consumed by [`ProtenixModel`][].

    Extends [`BaseInput`][ofoldx.models.contract.BaseInput] with the structural, template, MSA,
    and constraint fields the model reads. ``constraint_feature`` is the native Protenix v0.5 feature
    bundle. The four direct ``*_constraints`` fields are accepted as an OFoldX convenience for
    callers that already own individual tensors; when no constraint fields are provided, the model
    follows the reference path and omits the constraint update entirely.

    Attributes:
        token_mask: Per-token validity mask ``[..., N_token]``.
        residue_index: Per-token residue position index ``[..., N_token]``.
        chain_index: Per-token chain index ``[..., N_token]``.
        entity_index: Per-token entity index ``[..., N_token]``.
        symmetry_index: Per-token symmetry-copy index ``[..., N_token]``.
        token_index: Per-token ordinal index ``[..., N_token]``.
        atom_features: Per-atom input features ``[..., N_atom, atom_feature_size]``.
        atom_to_token: Token index per atom ``[..., N_atom]``.
        atom_mask: Per-atom validity mask ``[..., N_atom]``.
        reference_positions: Reference conformer atom positions ``[..., N_atom, 3]``.
        reference_space_id: Per-atom reference-space grouping id ``[..., N_atom]``.
        representative_atom_mask: Per-atom mask marking token representative atoms ``[..., N_atom]``.
        atom_token_index: Within-token atom offset per atom ``[..., N_atom]``.
        token_to_rep_atom: Per-token representative global atom index ``[..., N_token]``.
        token_features: Packed per-token features ``[..., N_token, token_feature_size]``.
        reference_mask: Per-atom reference validity mask ``[..., N_atom]``.
        reference_charges: Per-atom reference charges ``[..., N_atom]``.
        target_positions: Ground-truth atom positions for training ``[..., N_atom, 3]``, or ``None``.
        target_mask: Resolved-atom mask for the training target ``[..., N_atom]``, or ``None``.
        molecule_type: Per-token molecule-type id ``[..., N_token]``, or ``None``.
        frame_atom_index: Per-token frame atom indices ``[..., N_token, 3]``, or ``None``.
        frame_mask: Per-token frame validity mask ``[..., N_token]``, or ``None``.
        token_bonds: Token-bond adjacency ``[..., N_token, N_token]``, or ``None``.
        constraint_feature: Native Protenix constraint feature bundle or mapping, or ``None``.
        pocket_constraints: Direct pocket constraint tensor ``[..., N, N, 1]``, or ``None``.
        contact_constraints: Direct token-contact constraint tensor ``[..., N, N, 2]``, or ``None``.
        contact_atom_constraints: Direct atom-contact constraint tensor ``[..., N, N, 2]``, or ``None``.
        substructure_constraints: Direct substructure constraint tensor ``[..., N, N, 4]``, or ``None``.
        guidance_features: Training-free-guidance geometry features or mapping, or ``None``.
        msa_features: MSA features for the MSA trunk, or ``None``.
        msa_mask: MSA validity mask, or ``None``.
        template_features: Template features for the template embedder, or ``None``.
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
    constraint_feature: ProtenixConstraintFeatures | Mapping[str, Tensor] | None = None
    pocket_constraints: Tensor | None = None
    contact_constraints: Tensor | None = None
    contact_atom_constraints: Tensor | None = None
    substructure_constraints: Tensor | None = None
    guidance_features: ProtenixGuidanceFeatures | Mapping[str, Tensor] | None = None
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
