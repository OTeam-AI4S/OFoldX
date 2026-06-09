r"""
AlphaFold 3 top-level model.

Assembles the native AlphaFold 3 structure-prediction path from the shared OFoldX modules: atom-to-token
input embedding, the recycled Pairformer trunk with optional MSA and template updates, the EDM coordinate
denoiser/sampler diffusion path, and the distogram and pLDDT/PAE/PDE/resolved confidence heads. The model owns
the AlphaFold 3 input contract and the training (single-step denoise plus no-grad confidence rollout) versus
eval (full reverse-diffusion sampling) branching; generic neural-network blocks live in the shared modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

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
from .confidence import ConfidenceSummary, confidence_summary
from .config import AlphaFold3Config


@MODELS.register("alphafold3")
@STRUCTURE_PREDICTION_MODELS.register("alphafold3")
class AlphaFold3Model(ModelMixin):
    r"""
    AlphaFold 3 structure-prediction model assembled from the shared OFoldX modules.

    Wires the input embedder, MSA and template embedders, the recycled Pairformer trunk, the EDM
    coordinate denoiser/sampler, and the distogram and confidence heads into a single model. Training runs
    a single-step diffusion objective plus a no-grad mini-rollout for the confidence head; evaluation runs
    the full multi-step reverse-diffusion sampler. Registered under ``"alphafold3"`` in both the model and
    structure-prediction registries.

    Args:
        config: Model configuration; see
            [`AlphaFold3Config`][ofoldx.models.alphafold3.AlphaFold3Config].

    Examples:
        >>> from ofoldx import AutoModel, AutoProcessor  # doctest: +SKIP
        >>> processor = AutoProcessor.from_pretrained("oteam/alphafold3")  # doctest: +SKIP
        >>> model = AutoModel.from_pretrained("oteam/alphafold3").eval()  # doctest: +SKIP
        >>> output = model(processor("complex.cif"))  # doctest: +SKIP
        >>> output.pred_coords.shape  # doctest: +SKIP
        torch.Size([1, 1, 512, 3])
    """

    config_class = AlphaFold3Config

    def __init__(self, config: AlphaFold3Config) -> None:
        super().__init__(config)
        self._recycling_steps = config.recycling_steps
        self.relative_position_embedder = RelativePositionEmbedder(config.relative_position_size, config.pair_size)
        self.single_init_projection = nn.Linear(config.token_feature_size, config.single_size, bias=False)
        self.pair_left_projection = nn.Linear(config.token_feature_size, config.pair_size, bias=False)
        self.pair_right_projection = nn.Linear(config.token_feature_size, config.pair_size, bias=False)
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
        self.distogram_head = DistogramHead(config.pair_size, config.distogram.num_bins, bias=False)
        self.confidence_head = ConfidenceHead(
            single_size=config.single_size,
            pair_size=config.pair_size,
            token_feature_size=config.token_feature_size,
            distance_bin_start=config.confidence.distance_bin_start,
            distance_bin_end=config.confidence.distance_bin_end,
            distance_bin_step=config.confidence.distance_bin_step,
            num_bins=config.confidence.num_plddt_bins,
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
            triangle_attention_hidden_size=config.template.triangle_attention_hidden_size,
            num_triangle_attention_heads=config.template.num_triangle_attention_heads,
            expansion=config.template.expansion,
            dropout=config.template.dropout,
            operator_backend=config.trunk.operator_backend,
        )
        self._build_additional_modules(config)
        self.post_init()

    def _build_coordinate_denoiser(self, config: AlphaFold3Config) -> CoordinateDenoiser:
        r"""
        Construct the EDM coordinate denoiser from the denoiser sub-config.

        Args:
            config: Model configuration (reads ``config.denoiser`` and the shared widths).

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
            embed_relative_position=False,
            num_atom_encoder_blocks=config.denoiser.num_atom_encoder_blocks,
            num_atom_encoder_attention_heads=config.denoiser.num_atom_encoder_attention_heads,
            num_atom_decoder_blocks=config.denoiser.num_atom_decoder_blocks,
            num_atom_decoder_attention_heads=config.denoiser.num_atom_decoder_attention_heads,
            num_query_atoms=config.denoiser.num_query_atoms,
            num_key_atoms=config.denoiser.num_key_atoms,
            blocks_per_checkpoint=config.denoiser.blocks_per_checkpoint,
            operator_backend=config.denoiser.operator_backend,
        )

    def _build_coordinate_sampler(self, config: AlphaFold3Config) -> CoordinateEdmSampler:
        r"""
        Construct the EDM reverse-diffusion sampler from the sampler sub-config.

        Args:
            config: Model configuration (reads ``config.sampler`` and ``config.denoiser.sigma_data``).

        Returns:
            The configured reverse-diffusion sampler.
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

    def _build_additional_modules(self, config: AlphaFold3Config) -> None:
        r"""
        Build model-specific extension modules; a no-op hook for subclasses to override.

        Args:
            config: Model configuration passed through from the constructor.
        """

    def reset_parameters(self) -> None:
        r"""
        Zero-initialize the recycling and PAE/PDE projection weights after construction.

        The recycling projections and the PAE/PDE output projections start at zero so the first
        recycling pass and the untrained confidence head contribute nothing, matching the upstream
        AlphaFold 3 initialization.

        Examples:
            >>> from ofoldx.models.alphafold3 import AlphaFold3Config, AlphaFold3Model  # doctest: +SKIP
            >>> model = AlphaFold3Model(AlphaFold3Config())  # doctest: +SKIP
            >>> model.reset_parameters()  # doctest: +SKIP
        """

        nn.init.zeros_(self.pair_recycle_projection.weight)
        nn.init.zeros_(self.single_recycle_projection.weight)
        nn.init.zeros_(self.confidence_head.pair_projector.pae_projection.weight)
        nn.init.zeros_(self.confidence_head.pair_projector.pde_projection.weight)

    def forward(
        self,
        batch: AlphaFold3Input,
        *,
        recycling_steps: int | None = None,
        num_samples: int = 1,
        num_sampling_steps: int | None = None,
        generator: torch.Generator | None = None,
    ) -> StructurePredictionOutput:
        r"""
        Run the model on typed input.

        In training mode the diffusion module is trained with a single denoising step on
        noised ground truth (grad), and a separate no-grad mini-rollout produces the
        coords the confidence head scores. In eval/inference mode the full multi-step
        EDM sampler produces the predicted coords.

        Args:
            batch: Typed model input.
            recycling_steps: Optional number of recycling passes. Defaults to the config value.
            num_samples: Number of reverse-diffusion samples to draw in eval/inference mode.
            num_sampling_steps: Optional number of EDM sampling steps. Defaults to the sampler config value.
            generator: Optional PyTorch random generator for diffusion noise.

        Returns:
            Structure-prediction output carrying predicted coords, distogram logits, and the pLDDT /
            PAE / PDE / resolved confidence logits.

        Raises:
            ValueError: If ``recycling_steps`` is not positive.

        Examples:
            >>> from ofoldx import AutoModel, AutoProcessor  # doctest: +SKIP
            >>> processor = AutoProcessor.from_pretrained("oteam/alphafold3")  # doctest: +SKIP
            >>> model = AutoModel.from_pretrained("oteam/alphafold3").eval()  # doctest: +SKIP
            >>> output = model(processor("complex.cif"), num_samples=5)  # doctest: +SKIP
            >>> output.pred_coords.shape  # doctest: +SKIP
            torch.Size([1, 5, 512, 3])
        """

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
        )
        return self._assemble_output(batch, representations=representations, confidence_coords=pred_coords)

    def _forward_train(
        self,
        batch: AlphaFold3Input,
        *,
        representations: AlphaFold3Representations,
        generator: torch.Generator | None,
    ) -> StructurePredictionOutput:
        r"""
        Run one training step: single-step diffusion denoising plus a no-grad mini-rollout.

        The rollout coords (detached representations, no grad) feed the confidence head, while the
        single-step denoised coords carry the gradient for the EDM-weighted coordinate loss.

        Args:
            batch: Typed model input.
            representations: Trunk representations for this batch (kept attached for the loss path).
            generator: Optional PyTorch random generator for diffusion noise.

        Returns:
            Structure-prediction output with ``denoised_coords``, ``augmented_target``, and
            ``noise_level`` attached for the diffusion loss.
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
        batch: AlphaFold3Input,
        *,
        representations: AlphaFold3Representations,
        confidence_coords: torch.Tensor,
    ) -> StructurePredictionOutput:
        r"""
        Score ``confidence_coords`` and pack the distogram and confidence logits into the output.

        Args:
            batch: Typed model input.
            representations: Trunk representations for this batch.
            confidence_coords: Predicted coords the confidence head scores, ``[..., N_sample, N_atom, 3]``.

        Returns:
            Structure-prediction output with predicted coords, distogram logits, and confidence logits.
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

    def confidence_summary(
        self, output: StructurePredictionOutput, batch: AlphaFold3Input, *, max_error_bin: float = 31.0
    ) -> ConfidenceSummary:
        r"""
        Compute the pTM / ipTM / ranking-score summary from a model output.

        This is an opt-in, parameter-free post-processing step over the PAE logits already produced by
        the confidence head: callers run `forward` and then pass the output plus the batch here to
        get the scalar ranking summaries. Chains are read from ``batch.chain_index`` and the valid frame
        mask from ``batch.frame_mask`` (all tokens are treated as framed when it is absent).

        Args:
            output: Structure-prediction output carrying ``pae_logits``.
            batch: The model input the output was produced from.
            max_error_bin: Upper edge of the resolved PAE bins in Angstrom.

        Returns:
            Scalar pTM / ipTM / ranking-score summaries per leading batch and sample axis.

        Raises:
            ValueError: If the output has no PAE logits.

        Examples:
            >>> model = AlphaFold3Model(AlphaFold3Config()).eval()  # doctest: +SKIP
            >>> batch = processor("complex.cif")  # doctest: +SKIP
            >>> summary = model.confidence_summary(model(batch), batch)  # doctest: +SKIP
            >>> summary.ranking_score.shape  # doctest: +SKIP
            torch.Size([1, 1])
        """

        if output.pae_logits is None:
            raise ValueError("confidence_summary requires pae_logits; run the model in eval mode first")
        num_tokens = output.pae_logits.shape[-2]
        frame_mask = batch.frame_mask
        if frame_mask is None:
            frame_mask = batch.token_mask.to(torch.bool)
        # Broadcast token-axis fields across the per-sample logit axis so the summary is per-sample.
        sample_axes = output.pae_logits.dim() - 3
        asym_id = batch.chain_index
        for _ in range(sample_axes - asym_id.dim() + 1):
            asym_id = asym_id.unsqueeze(1)
            frame_mask = frame_mask.unsqueeze(1)
        return confidence_summary(
            output.pae_logits,
            frame_mask=frame_mask.expand(*output.pae_logits.shape[:-3], num_tokens),
            asym_id=asym_id.expand(*output.pae_logits.shape[:-3], num_tokens),
            max_error_bin=max_error_bin,
        )

    def _train_denoise(
        self,
        batch: AlphaFold3Input,
        *,
        representations: AlphaFold3Representations,
        num_samples: int,
        generator: torch.Generator | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Noise the (augmented) ground truth once per sample and denoise it in a single step.

        Centers and augments the diffusion target over resolved atoms only, draws an independent
        log-normal noise level per sample, and runs one denoiser pass — the single-step training
        objective the EDM-weighted coordinate loss scores.

        Args:
            batch: Typed model input; must carry ``target_positions``.
            representations: Trunk representations for this batch.
            num_samples: Number of independent noised samples to draw.
            generator: Optional PyTorch random generator for augmentation and noise.

        Returns:
            ``(denoised_coords, augmented_target, noise_level)``, all with a leading ``num_samples`` axis,
            for the EDM-weighted coordinate loss.

        Raises:
            ValueError: If ``batch.target_positions`` is ``None``.
        """

        if batch.target_positions is None:
            raise ValueError("diffusion training requires batch.target_positions")
        target = batch.target_positions.to(dtype=representations.single.dtype)
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
        batch: AlphaFold3Input,
        *,
        recycling_steps: int,
    ) -> AlphaFold3Representations:
        r"""
        Build the token embedding and run the recycled trunk to final single/pair representations.

        Args:
            batch: Typed model input.
            recycling_steps: Number of recycling passes to run.

        Returns:
            Trunk representations (token embedding, single, pair, relative features, pair mask).
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
        pair_init = self._initial_pair(token_embedding, batch, relative_features)

        pair_mask = pair_mask_from_token_mask(batch.token_mask)
        state = AlphaFold3RecyclingState(
            token_embedding=token_embedding,
            single_init=single_init,
            pair_init=pair_init,
            pair_mask=pair_mask,
        )
        single, pair = self._run_recycling(batch, state=state, recycling_steps=recycling_steps)

        return AlphaFold3Representations(
            token_embedding=token_embedding,
            single=single,
            pair=pair,
            relative_features=relative_features,
            pair_mask=pair_mask,
        )

    def _run_recycling(
        self,
        batch: AlphaFold3Input,
        *,
        state: AlphaFold3RecyclingState,
        recycling_steps: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        Iterate the trunk ``recycling_steps`` times, keeping gradients only for the final training cycle.

        Args:
            batch: Typed model input.
            state: Init/context bundle carrying the initial single/pair states and pair mask.
            recycling_steps: Number of recycling passes to run.

        Returns:
            ``(single, pair)`` representations after the final recycling pass.
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
        batch: AlphaFold3Input,
        *,
        state: AlphaFold3RecyclingState,
        single: torch.Tensor,
        pair: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        Run a single recycling pass: fold the previous states back in, apply templates/MSA, then Pairformer.

        Args:
            batch: Typed model input.
            state: Init/context bundle carrying the initial single/pair states and pair mask.
            single: Single representation from the previous cycle.
            pair: Pair representation from the previous cycle.

        Returns:
            ``(single, pair)`` representations after this recycling pass.
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

    def _initial_pair(
        self,
        token_embedding: torch.Tensor,
        batch: AlphaFold3Input,
        relative_features: RelativePositionBuckets,
    ) -> torch.Tensor:
        r"""
        Build the initial pair representation from single projections, relative positions, and token bonds.

        Args:
            token_embedding: Initial target-feature representation ``(..., N_token, token_feature_size)``.
            batch: Typed model input (provides optional token bonds and the token mask).
            relative_features: Relative-position bucket features for the pair embedder.

        Returns:
            Masked initial pair representation ``(..., N_token, N_token, pair_size)``.
        """

        left = self.pair_left_projection(token_embedding).unsqueeze(-2)
        right = self.pair_right_projection(token_embedding).unsqueeze(-3)
        if batch.token_bonds is None:
            token_bonds = token_embedding.new_zeros(*token_embedding.shape[:-1], token_embedding.shape[-2])
        else:
            token_bonds = batch.token_bonds.to(device=token_embedding.device, dtype=token_embedding.dtype)
            token_bonds = torch.broadcast_to(token_bonds, (*token_embedding.shape[:-1], token_embedding.shape[-2]))
        token_bonds = token_bonds.unsqueeze(-1)
        pair = left + right
        pair = pair + self.relative_position_embedder(relative_features)
        pair = pair + self.token_bond_projection(token_bonds)

        pair_mask = pair_mask_from_token_mask(batch.token_mask)
        return pair * pair_mask.unsqueeze(-1).to(dtype=pair.dtype)

    def _predict_coords(
        self,
        batch: AlphaFold3Input,
        *,
        representations: AlphaFold3Representations,
        num_samples: int,
        num_sampling_steps: int | None,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        r"""
        Run the full reverse-diffusion sampler to produce predicted atom coords.

        Args:
            batch: Typed model input.
            representations: Trunk representations conditioning the denoiser.
            num_samples: Number of reverse-diffusion samples to draw.
            num_sampling_steps: Number of EDM sampling steps, or ``None`` to use the sampler config value.
            generator: Optional PyTorch random generator for diffusion noise.

        Returns:
            Predicted atom coords ``[..., num_samples, N_atom, 3]``.
        """

        local_geometry = self._build_denoising_local_geometry(batch)
        return self.coordinate_sampler(
            self.coordinate_denoiser,
            context=CoordinateDenoiseContext(
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
            ),
            num_samples=num_samples,
            num_sampling_steps=self._num_sampling_steps if num_sampling_steps is None else num_sampling_steps,
            generator=generator,
        )

    def _build_input_local_geometry(self, batch: AlphaFold3Input) -> LocalAtomGeometry:
        r"""
        Build the input-embedder local atom geometry from the batch reference conformer.

        Args:
            batch: Typed model input (provides reference positions and space ids).

        Returns:
            Local atom geometry in the input atom-encoder layout.
        """

        reference_positions = batch.reference_positions.to(device=batch.atom_features.device)
        return self.input_embedder.build_local_geometry(reference_positions, batch.reference_space_id)

    def _build_denoising_local_geometry(self, batch: AlphaFold3Input) -> LocalAtomGeometry:
        r"""
        Build the denoiser local atom geometry from the batch reference conformer.

        Args:
            batch: Typed model input (provides reference positions and space ids).

        Returns:
            Local atom geometry in the coordinate-denoiser atom layout.
        """

        reference_positions = batch.reference_positions.to(device=batch.atom_features.device)
        return self.coordinate_denoiser.build_local_geometry(reference_positions, batch.reference_space_id)


class InputEmbedder(nn.Module):
    r"""
    Build token input embeddings by encoding atoms to tokens and appending the token-feature tail.

    Runs the atom-to-token encoder over the packed atom features, then concatenates the encoder's token
    channels with the trailing per-token features to form the initial token embedding.

    Args:
        single_size: Width of the single (token) representation.
        token_feature_size: Width of the packed per-token feature vector.
        atom_feature_size: Width of the packed per-atom feature vector.
        operator_backend: Operator backend selector for the atom encoder (``"auto"`` by default).
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
        Return token embeddings concatenating atom-derived channels with the token-feature tail.

        Args:
            atom_features: Packed per-atom features ``(..., N_atom, atom_feature_size)``.
            reference_positions: Centered reference-conformer atom coords ``(..., N_atom, 3)``.
            reference_charges: Per-atom reference formal charges ``(..., N_atom)`` (``arcsinh``-compressed
                before encoding).
            reference_mask: Boolean mask for valid reference-conformer atoms ``(..., N_atom)``.
            token_features: Packed per-token features ``(..., N_token, token_feature_size)``.
            atom_to_token: Token index for each atom ``(..., N_atom)``.
            num_tokens: Number of tokens ``N_token``.
            atom_mask: Optional boolean atom validity mask ``(..., N_atom)``.
            token_mask: Optional boolean token validity mask ``(..., N_token)``; masked tokens are zeroed.
            local_geometry: Optional precomputed local atom geometry for the atom encoder.

        Returns:
            Token embedding ``(..., N_token, single_size + token_tail)``.
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
        Build local atom geometry using the input atom-encoder layout.

        Args:
            reference_positions: Centered reference-conformer atom coords ``(..., N_atom, 3)``.
            reference_space_id: Reference-conformer group id for each atom ``(..., N_atom)``.

        Returns:
            Local atom geometry for the input atom encoder.
        """

        return self.atom_encoder.build_local_geometry(reference_positions, reference_space_id)

    def _token_tail(self, token_single: Tensor, token_features: Tensor) -> Tensor:
        r"""
        Slice the trailing per-token feature channels and align them to the token embedding.

        Args:
            token_single: Atom-encoder token channels ``(..., N_token, single_size)``.
            token_features: Packed per-token features ``(..., N_token, token_feature_size)``.

        Returns:
            The token-feature tail ``(..., N_token, token_feature_size - single_size)`` on the token
            embedding's device and dtype.

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


class TrainingNoiseSampler(nn.Module):
    r"""
    Sample EDM training noise levels from a log-normal distribution.

    Mirrors the AlphaFold 3 / Protenix training noise schedule: draw
    ``sigma = exp(noise_log_mean + noise_log_std * N(0, 1)) * sigma_data``. Each training sample gets an
    independent noise level, so the denoiser learns to denoise across the whole sigma range
    in a single step (no reverse-sampling loop during training).

    Args:
        noise_log_mean: Mean of the log-normal noise distribution.
        noise_log_std: Standard deviation of the log-normal noise distribution.
        sigma_data: Data scale multiplying the sampled noise level.
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
        Return a tensor of independent, strictly positive noise levels with shape ``size``.

        Args:
            size: Output shape; one independent noise level is drawn per element.
            device: Optional device for the sampled tensor.
            dtype: Optional dtype for the sampled tensor.
            generator: Optional PyTorch random generator.

        Returns:
            Log-normal noise levels of shape ``size``.

        Examples:
            >>> import torch
            >>> sampler = TrainingNoiseSampler()
            >>> _ = torch.manual_seed(0)
            >>> noise = sampler((2,))
            >>> noise.shape
            torch.Size([2])
            >>> bool((noise > 0).all())
            True
        """

        standard_normal = torch.randn(size, device=device, dtype=dtype, generator=generator)
        return (standard_normal * self.noise_log_std + self.noise_log_mean).exp() * self.sigma_data


class ConfidenceHead(nn.Module):
    r"""
    Predict pLDDT, PAE, PDE, and resolved logits from the trunk single/pair states and predicted coords.

    Embeds representative-atom distances into the pair state, runs a small Pairformer stack, and projects to
    the atom-level (pLDDT / resolved) and pair-level (PAE / PDE) confidence logits. Each diffusion sample is
    scored independently and the per-sample logits are stacked on a leading sample axis.

    Args:
        single_size: Width of the single (token) representation.
        pair_size: Width of the pair representation.
        token_feature_size: Width of the packed per-token feature vector.
        distance_bin_start: Lower edge of the representative-distance one-hot bins in Angstrom.
        distance_bin_end: Upper edge of the representative-distance one-hot bins in Angstrom.
        distance_bin_step: Width of each representative-distance bin in Angstrom.
        num_bins: Number of pLDDT bins.
        num_resolved_bins: Number of resolved-classification bins.
        max_atoms_per_token: Maximum atoms per token, sizing the per-offset weight tables.
        num_pae_bins: Number of predicted-aligned-error bins.
        num_pde_bins: Number of predicted-distance-error bins.
        num_blocks: Number of Pairformer blocks in the confidence stack.
        num_pair_attention_heads: Number of pair-attention heads in the Pairformer stack.
        triangle_multiplication_hidden_size: Hidden width of the triangle-multiplication updates.
        triangle_attention_hidden_size: Per-head width of the triangle-attention updates.
        num_triangle_attention_heads: Number of triangle-attention heads.
        expansion: Feed-forward expansion factor in the Pairformer transitions.
        dropout: Pairformer dropout probability.
        blocks_per_checkpoint: Blocks per gradient-checkpoint segment, or ``None`` to disable.
        operator_backend: Operator backend selector for the Pairformer stack.
        single_max_abs: Optional clamp on the single representation before normalization; ``None`` disables it.
    """

    def __init__(
        self,
        single_size: int = 384,
        pair_size: int = 128,
        token_feature_size: int = 449,
        distance_bin_start: float = 3.25,
        distance_bin_end: float = 52.0,
        distance_bin_step: float = 1.25,
        num_bins: int = 50,
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
            use_distance_projection=False,
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
        self.atom_projector = AtomConfidenceProjector(
            single_size=single_size,
            num_bins=num_bins,
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
        Return atom-level and token-pair confidence logits for every predicted sample.

        Args:
            single: Token single representation ``(..., N_token, single_size)``.
            pair: Token pair representation ``(..., N_token, N_token, pair_size)``.
            token_embedding: Token embedding ``(..., N_token, token_feature_size)``.
            pred_coords: Predicted atom coords ``(..., N_sample, N_atom, 3)``.
            token_to_rep_atom: Representative atom index for each token ``(..., N_token)``.
            atom_to_token: Token index for each atom ``(..., N_atom)``.
            atom_token_index: Within-token atom offset for each atom ``(..., N_atom)``.
            pair_mask: Optional boolean token-pair validity mask ``(..., N_token, N_token)``.

        Returns:
            Confidence logits with a leading per-sample axis on each field.
        """

        if self.single_max_abs is not None:
            single = torch.clamp(single, min=-self.single_max_abs, max=self.single_max_abs)
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
        Score a single predicted sample: embed distances, run the Pairformer, and project the logits.

        Args:
            single: Token single representation ``(..., N_token, single_size)``.
            pair: Token pair representation ``(..., N_token, N_token, pair_size)``.
            token_embedding: Token embedding ``(..., N_token, token_feature_size)``.
            pair_mask: Optional boolean token-pair validity mask ``(..., N_token, N_token)``.
            representative_coords: Representative-atom coords for this sample ``(..., N_token, 3)``.
            atom_to_token: Token index for each atom ``(..., N_atom)``.
            atom_token_index: Within-token atom offset for each atom ``(..., N_atom)``.

        Returns:
            Confidence logits for this single sample (no leading sample axis).
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
    Augment the pair state with token-embedding projections and representative-atom distance features.

    Adds left/right token-embedding projections plus a binned one-hot and a raw projection of the
    representative-atom pairwise distances, so the confidence Pairformer sees the predicted geometry.

    Args:
        pair_size: Width of the pair representation.
        token_feature_size: Width of the packed per-token feature vector.
        distance_bin_start: Lower edge of the distance one-hot bins in Angstrom.
        distance_bin_end: Upper edge of the distance one-hot bins in Angstrom.
        distance_bin_step: Width of each distance bin in Angstrom.
    """

    def __init__(
        self,
        pair_size: int = 128,
        token_feature_size: int = 449,
        distance_bin_start: float = 3.25,
        distance_bin_end: float = 52.0,
        distance_bin_step: float = 1.25,
        use_distance_projection: bool = True,
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
        self.distance_projection = nn.Linear(1, pair_size, bias=False) if use_distance_projection else None

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
        if self.distance_projection is not None:
            pair = pair + self.distance_projection(distance.unsqueeze(-1))
        return pair


class AtomConfidenceProjector(nn.Module):
    r"""
    Project token single representations to per-atom pLDDT and resolved logits.

    Broadcasts each token's single representation to its atoms, then contracts each atom's normalized
    features against the weight matrix selected by its within-token offset, giving a distinct projection
    per atom position.

    Args:
        single_size: Width of the single (token) representation.
        num_bins: Number of pLDDT bins.
        num_resolved_bins: Number of resolved-classification bins.
        max_atoms_per_token: Maximum atoms per token, sizing the per-offset weight tables.
    """

    def __init__(
        self, single_size: int = 384, num_bins: int = 50, num_resolved_bins: int = 2, max_atoms_per_token: int = 24
    ):
        super().__init__()
        self.max_atoms_per_token = max_atoms_per_token
        self.plddt_weight = nn.Parameter(torch.empty(max_atoms_per_token, single_size, num_bins))
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
            ``(plddt_logits, resolved_logits)`` with shapes ``(..., N_atom, num_bins)`` and
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
    Project the pair representation to PAE and (symmetrized) PDE logits.

    Args:
        pair_size: Width of the pair representation.
        num_pae_bins: Number of predicted-aligned-error bins.
        num_pde_bins: Number of predicted-distance-error bins.
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
    One-hot-encode each distance into the half-open bin ``(lower, upper)`` it falls in.

    Args:
        distance: Pairwise distances ``(..., N, N)``.
        lower_bins: Lower edges of the bins ``(num_bins,)``.
        upper_bins: Upper edges of the bins ``(num_bins,)``.

    Returns:
        One-hot bin membership ``(..., N, N, num_bins)`` in the distance dtype.
    """

    return ((distance[..., None] > lower_bins) & (distance[..., None] < upper_bins)).to(dtype=distance.dtype)


@dataclass(slots=True)
class AlphaFold3Representations:
    r"""
    Intermediate AlphaFold 3 representations produced by the trunk.

    Attributes:
        token_embedding: Token embedding before trunk updates.
        single: Final token single representation.
        pair: Final token pair representation.
        relative_features: Relative-position feature bundle used by denoising.
        pair_mask: Boolean token-pair validity mask.
    """

    token_embedding: torch.Tensor
    single: torch.Tensor
    pair: torch.Tensor
    relative_features: RelativePositionBuckets
    pair_mask: torch.Tensor

    def detach(self) -> AlphaFold3Representations:
        r"""
        Return a gradient-detached copy (used for the no-grad confidence mini-rollout).

        Returns:
            Representation container with tensor fields detached.
        """

        return AlphaFold3Representations(
            token_embedding=self.token_embedding.detach(),
            single=self.single.detach(),
            pair=self.pair.detach(),
            relative_features=self.relative_features.detach(),
            pair_mask=self.pair_mask,
        )


@dataclass(slots=True)
class AlphaFold3RecyclingState:
    r"""
    Init/context bundle passed through AlphaFold 3 recycling.

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


@dataclass(slots=True, kw_only=True)
class AlphaFold3Input(BaseInput):
    r"""
    Input tensors consumed by AlphaFold3Model.

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
        contact_constraints: Optional contact-constraint categories.
        constraint_distogram: Optional contact-constraint distogram features.
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
