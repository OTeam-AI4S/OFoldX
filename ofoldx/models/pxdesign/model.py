r"""
OFoldX-native reimplementation of the PxDesign backbone diffusion generator.

Defines the [`PxDesignDesigner`][ofoldx.models.pxdesign.model.PxDesignDesigner] entry point,
its condition-embedding and EDM-sampling sub-modules, and the
[`PxDesignInput`][ofoldx.models.pxdesign.model.PxDesignInput] tensor contract that drive coordinate
sampling from design conditioning.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ...modules.atom import AtomToTokenEncoder
from ...modules.denoiser import CoordinateDenoiser
from ...modules.diffusion import CoordinateDenoiseContext, EdmScheduler
from ...modules.diffusion.sampling import CoordinateEdmSampler
from ...modules.embedding import RelativePositionBuckets, generate_relative_position_features
from ...modules.functional import pair_mask_from_token_mask
from ...modules.head import DistogramHead
from ...modules.representation import Pair, Single
from ..contract import BaseInput, StructureDesignCandidate, StructurePredictionOutput, register_pytree_dataclass
from ..mixin import ModelMixin
from ..registry import DESIGN_MODELS, MODELS
from .config import PxDesignConfig


@MODELS.register("pxdesign")
@DESIGN_MODELS.register("pxdesign")
class PxDesignDesigner(ModelMixin):
    r"""
    Samples design-backbone coordinates from PxDesign conditioning via AF3-style EDM diffusion.

    This entry point implements the runnable PxDesign-d core and the executable
    OFoldX-native pipeline hooks used by the official family: ProteinMPNN
    sequence assignment and structure-predictor-based filtering. It embeds the
    per-atom/design conditioning, runs the AF3-style EDM coordinate sampler, and
    exposes the design distogram head.

    Args:
        config: PxDesign configuration; when ``None`` the defaults of
            [`PxDesignConfig`][ofoldx.models.pxdesign.config.PxDesignConfig] are used.

    Examples:
        >>> from ofoldx import AutoModel, AutoProcessor  # doctest: +SKIP
        >>> processor = AutoProcessor.from_pretrained("ofoldx/pxdesign")  # doctest: +SKIP
        >>> model = AutoModel.from_pretrained("ofoldx/pxdesign")  # doctest: +SKIP
        >>> output = model.design(processor("design.json"))  # doctest: +SKIP
    """

    config_class = PxDesignConfig
    #: Typed input this model's ``forward`` consumes; enables the ``model(**inputs)`` keyword-tensor call path.
    #: Bound after ``PxDesignInput`` is defined (below the class body) to avoid a forward reference.
    input_class: ClassVar[type[BaseInput]]

    def __init__(self, config: PxDesignConfig | None = None) -> None:
        config = PxDesignConfig() if config is None else config
        super().__init__(config)
        self.condition_embedder = PxDesignConditionEmbedder(config)
        self.coordinate_denoiser = self._build_coordinate_denoiser(config)
        self.coordinate_sampler = PxDesignCoordinateSampler(config)
        self.training_noise_sampler = TrainingNoiseSampler(
            noise_log_mean=config.diffusion_training.noise_log_mean,
            noise_log_std=config.diffusion_training.noise_log_std,
            sigma_data=config.denoiser.sigma_data,
        )
        self._diffusion_train_num_samples = config.diffusion_training.num_samples
        self.distogram_head = DistogramHead(
            pair_size=config.pair_size,
            num_bins=config.distogram.num_distogram_bins,
            symmetrize_before_projection=config.distogram.symmetrize_before_projection,
        )
        self.post_init()

    def reset_parameters(self) -> None:
        r"""
        Initialize embedding tables not covered by the shared ``ModelMixin`` initializer.

        Draws the conditional-template embedding weight from a truncated normal
        scaled by ``config.initializer_range``. Called by ``post_init``.
        """

        nn.init.trunc_normal_(
            self.condition_embedder.template_embedder.embedding.weight,
            mean=0.0,
            std=float(self.config.initializer_range),
        )

    def _build_coordinate_denoiser(self, config: PxDesignConfig) -> CoordinateDenoiser:
        r"""
        Construct the AF3-style coordinate denoiser from the denoiser sub-config.

        Args:
            config: PxDesign configuration whose ``denoiser`` and top-level width
                fields parameterize the denoiser.

        Returns:
            The configured [`CoordinateDenoiser`][ofoldx.modules.denoiser.CoordinateDenoiser].
        """

        denoiser = config.denoiser
        return CoordinateDenoiser(
            num_encoder_blocks=denoiser.num_token_encoder_blocks,
            num_encoder_attention_heads=denoiser.num_token_encoder_attention_heads,
            sigma_data=denoiser.sigma_data,
            atom_hidden_size=denoiser.atom_hidden_size,
            atom_pair_hidden_size=denoiser.atom_pair_hidden_size,
            token_hidden_size=denoiser.token_hidden_size,
            single_size=config.single_size,
            pair_size=config.pair_size,
            token_feature_size=config.token_feature_size,
            atom_feature_size=config.atom_feature_size,
            num_atom_encoder_blocks=denoiser.num_atom_encoder_blocks,
            num_atom_encoder_attention_heads=denoiser.num_atom_encoder_attention_heads,
            num_atom_decoder_blocks=denoiser.num_atom_decoder_blocks,
            num_atom_decoder_attention_heads=denoiser.num_atom_decoder_attention_heads,
            num_query_atoms=denoiser.num_query_atoms,
            num_key_atoms=denoiser.num_key_atoms,
            blocks_per_checkpoint=denoiser.blocks_per_checkpoint,
            operator_backend=denoiser.operator_backend,
        )

    def forward(
        self,
        batch: PxDesignInput,
        *,
        num_samples: int | None = None,
        generator: torch.Generator | None = None,
    ) -> StructureDesignCandidate:
        r"""
        Run the single-step diffusion training objective on PxDesign conditioning.

        This is the TRAINING path. It encodes the design conditioning, draws an independent EDM noise level
        per training sample, noises the (augmented) ground-truth ``target_positions`` once, and predicts the
        clean coordinates in a single denoiser step. The design distogram logits are attached for the
        auxiliary distogram loss. Multi-step coordinate *sampling* is the distinct inference path exposed by
        [`design`][ofoldx.models.pxdesign.model.PxDesignDesigner.design].

        The public PxDesign release ships inference-only; this objective mirrors the single-step EDM denoise
        of the sister Protenix/ODesign diffusion trainers, which share PxDesign's denoiser and EDM sampler.

        Args:
            batch: PxDesign model input; must carry ``target_positions`` for the diffusion target.
            num_samples: Number of noised training samples; defaults to the diffusion-training config.
            generator: Optional random generator seeding the augmentation and noise draws.

        Returns:
            A [`StructureDesignCandidate`][ofoldx.models.contract.StructureDesignCandidate] whose structure
            prediction carries the single-step ``denoised_coords`` (also surfaced as ``pred_coords``), the
            ``augmented_target`` and per-sample ``noise_level`` for the EDM-weighted diffusion loss, and the
            design distogram logits.

        Raises:
            ValueError: If ``batch.target_positions`` is ``None``.

        Examples:
            >>> model = PxDesignDesigner().train()  # doctest: +SKIP
            >>> loss_inputs = model(batch)  # doctest: +SKIP
        """

        representations = self.encode_condition(batch)
        num_samples = self._diffusion_train_num_samples if num_samples is None else num_samples
        denoised_coords, augmented_target, noise_level = self._train_denoise(
            batch,
            representations=representations,
            num_samples=num_samples,
            generator=generator,
        )
        structure_prediction = StructurePredictionOutput(
            pred_coords=denoised_coords,
            distogram_logits=self.distogram_head(representations.pair.tensor),
            denoised_coords=denoised_coords,
            augmented_target=augmented_target,
            noise_level=noise_level,
        )
        return StructureDesignCandidate(
            structure_prediction=structure_prediction,
            metadata={
                "architecture": "pxdesign",
                "backend": "native_pxdesign",
                "checkpoint_variant": self.config.checkpoint_variant,
                "mode": "diffusion_training",
                "num_samples": int(denoised_coords.shape[-3]),
            },
        )

    @torch.no_grad()
    def design(
        self,
        batch: PxDesignInput,
        *,
        num_samples: int | None = None,
        num_sampling_steps: int | None = None,
        sample_chunk_size: int | None = None,
        initial_coords: Tensor | None = None,
        preserve_condition_coords: bool = False,
        generator: torch.Generator | None = None,
    ) -> StructureDesignCandidate:
        r"""
        Generate design-backbone coordinates from PxDesign conditioning.

        This is the capability verb and the DISTINCT inference path (``forward`` is the training path). It
        encodes the design conditioning, runs the multi-step EDM coordinate sampler under ``torch.no_grad``
        to draw ``num_samples`` backbone samples, and attaches the design distogram logits. The facade wraps
        the returned candidate into a design result.

        Args:
            batch: PxDesign model input.
            num_samples: Number of coordinate samples; defaults to the sampler config.
            num_sampling_steps: Number of EDM steps; defaults to the sampler config.
            sample_chunk_size: Samples per forward chunk; defaults to the sampler config.
            initial_coords: Optional starting coords `[..., N_atom, 3]` for sampling.
            preserve_condition_coords: When ``True``, restore the fixed condition atoms in the sampled
                coords from ``batch.condition_coords``.
            generator: Optional random generator seeding the sampling noise.

        Returns:
            A [`StructureDesignCandidate`][ofoldx.models.contract.StructureDesignCandidate] whose structure
            prediction carries the sampled ``pred_coords`` of shape `[..., num_samples, N_atom, 3]` and the
            distogram logits, plus run metadata.

        Examples:
            >>> from ofoldx import AutoModel, AutoProcessor  # doctest: +SKIP
            >>> model = AutoModel.from_pretrained("ofoldx/pxdesign")  # doctest: +SKIP
            >>> processor = AutoProcessor.from_pretrained("ofoldx/pxdesign")  # doctest: +SKIP
            >>> output = model.design(processor("design.json"))  # doctest: +SKIP
        """

        representations = self.encode_condition(batch)
        num_sampling_steps = (
            self.config.sampler.num_sampling_steps if num_sampling_steps is None else num_sampling_steps
        )
        pred_coords = self._predict_coords(
            batch,
            representations=representations,
            num_samples=self.config.sampler.num_samples if num_samples is None else num_samples,
            num_sampling_steps=num_sampling_steps,
            sample_chunk_size=(
                self.config.sampler.sample_chunk_size if sample_chunk_size is None else sample_chunk_size
            ),
            initial_coords=initial_coords,
            generator=generator,
        )
        if preserve_condition_coords:
            pred_coords = _restore_condition_coords(batch, pred_coords)

        structure_prediction = StructurePredictionOutput(
            pred_coords=pred_coords,
            distogram_logits=self.distogram_head(representations.pair.tensor),
        )
        return StructureDesignCandidate(
            structure_prediction=structure_prediction,
            metadata={
                "architecture": "pxdesign",
                "backend": "native_pxdesign",
                "checkpoint_variant": self.config.checkpoint_variant,
                "num_samples": int(pred_coords.shape[-3]),
                "num_sampling_steps": num_sampling_steps,
                "sequence_assignment": "not_requested",
                "filtering": "not_requested",
            },
        )

    @staticmethod
    def official_pipeline_capabilities() -> dict[str, object]:
        r"""
        Return the official PxDesign model-family stages represented by this entry.

        PxDesign-d backbone generation is native here. Sequence assignment uses the
        OFoldX ProteinMPNN/MPNN surface. Filtering uses an injected or loaded
        OFoldX structure predictor and attaches the predictor scores used for
        candidate selection.

        Returns:
            A mapping from pipeline stage name (``"backbone_generation"``,
            ``"sequence_assignment"``, ``"filtering"``) to the backend(s) that
            realize it.

        Examples:
            >>> caps = PxDesignDesigner.official_pipeline_capabilities()
            >>> caps["backbone_generation"]
            'native_pxdesign_d'
        """

        return {
            "backbone_generation": "native_pxdesign_d",
            "sequence_assignment": "native_proteinmpnn",
            "filtering": ("native_structure_predictor_hook", "protenix", "alphafold3", "boltz"),
        }

    def encode_condition(self, batch: PxDesignInput) -> PxDesignRepresentations:  # type: ignore[override]
        r"""
        Build PxDesign token, pair, and relative-position conditioning.

        Validates the input, runs the condition embedder for the token embedding and
        template pair states, computes the relative-position features, and masks the
        pair states to valid token pairs.

        Args:
            batch: PxDesign model input.

        Returns:
            A [`PxDesignRepresentations`][ofoldx.models.pxdesign.model.PxDesignRepresentations]
            holding the token embedding, a zero-initialized single representation, the
            masked pair representation, and the relative-position features.

        Raises:
            ValueError: If any conditioning tensor in ``batch`` has an inconsistent shape.

        Examples:
            >>> model = PxDesignDesigner()  # doctest: +SKIP
            >>> representations = model.encode_condition(batch)  # doctest: +SKIP
        """

        _validate_pxdesign_input(batch, vocab_size=self.config.condition_embedder.vocab_size)
        token_embedding, pair = self.condition_embedder(batch)
        relative_features = generate_relative_position_features(
            residue_index=batch.residue_index,
            chain_index=batch.chain_index,
            entity_index=batch.entity_index,
            symmetry_index=batch.symmetry_index,
            token_index=batch.token_index,
            dtype=token_embedding.dtype,
        ).to(device=token_embedding.device)
        pair_mask = pair_mask_from_token_mask(batch.token_mask).to(device=pair.device, dtype=pair.dtype)
        pair = pair * pair_mask.unsqueeze(-1)
        single = token_embedding.new_zeros(*token_embedding.shape[:-1], self.config.single_size)
        return PxDesignRepresentations(
            token_embedding=token_embedding,
            single=Single(single),
            pair=Pair(pair),
            relative_features=relative_features,
        )

    def _predict_coords(
        self,
        batch: PxDesignInput,
        *,
        representations: PxDesignRepresentations,
        num_samples: int,
        num_sampling_steps: int,
        sample_chunk_size: int | None,
        initial_coords: Tensor | None,
        generator: torch.Generator | None,
    ) -> Tensor:
        r"""
        Run the EDM coordinate sampler over the encoded conditioning.

        Args:
            batch: PxDesign model input (supplies the atom-level tensors).
            representations: Encoded conditioning from ``encode_condition``.
            num_samples: Number of coordinate samples to draw.
            num_sampling_steps: Number of EDM denoising steps.
            sample_chunk_size: Samples per forward chunk, or ``None`` for one pass.
            initial_coords: Optional starting coords `[..., N_atom, 3]`.
            generator: Optional random generator seeding the sampling noise.

        Returns:
            Sampled coordinates of shape `[..., num_samples, N_atom, 3]`.
        """

        local_geometry = self.coordinate_denoiser.build_local_geometry(
            batch.reference_positions.to(device=batch.atom_features.device),
            batch.reference_space_id,
        )
        context = CoordinateDenoiseContext(
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
        )
        return self.coordinate_sampler(
            self.coordinate_denoiser,
            context=context,
            num_samples=num_samples,
            num_sampling_steps=num_sampling_steps,
            sample_chunk_size=sample_chunk_size,
            initial_coords=initial_coords,
            generator=generator,
        )

    def _train_denoise(
        self,
        batch: PxDesignInput,
        *,
        representations: PxDesignRepresentations,
        num_samples: int,
        generator: torch.Generator | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        r"""
        Noise the (augmented) ground-truth target once per sample and denoise in a single step.

        Draws ``num_samples`` centered/augmented copies of ``batch.target_positions``, samples an
        independent EDM noise level per sample, noises the target once, and runs one denoiser step over the
        fixed design conditioning. This is the grad-carrying core of the training ``forward``.

        Args:
            batch: PxDesign model input; must carry ``target_positions``.
            representations: Encoded conditioning from ``encode_condition``.
            num_samples: Number of independent noised samples to draw.
            generator: Optional random generator for the augmentation and noise draws.

        Returns:
            ``(denoised_coords, augmented_target, noise_level)``, each with a leading ``num_samples`` axis,
            for the EDM-weighted coordinate loss.

        Raises:
            ValueError: If ``batch.target_positions`` is ``None``.
        """

        if batch.target_positions is None:
            raise ValueError("diffusion training requires batch.target_positions")
        single = representations.single.tensor
        target = batch.target_positions.to(device=single.device, dtype=single.dtype)
        atom_mask = batch.atom_mask.to(device=target.device)
        # Center/augment the diffusion target over RESOLVED atoms only; materialized/absent atoms have no
        # true coordinate. Falls back to ``atom_mask`` when no explicit target mask is provided.
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
        # ``empty_like().normal_`` (not ``randn(shape)``) so a symbolic atom-count dim traces under
        # torch.compile with dynamic shapes instead of becoming a concrete SymInt array.
        noise = torch.empty_like(augmented_target).normal_(generator=generator)
        noisy_coords = augmented_target + noise * noise_level[..., None, None]

        local_geometry = self.coordinate_denoiser.build_local_geometry(
            batch.reference_positions.to(device=batch.atom_features.device),
            batch.reference_space_id,
        )
        context = CoordinateDenoiseContext(
            atom_features=batch.atom_features,
            atom_to_token=batch.atom_to_token,
            reference_positions=batch.reference_positions,
            reference_charges=batch.reference_charges,
            atom_mask=atom_mask,
            token_embedding=representations.token_embedding,
            single=single,
            pair=representations.pair.tensor,
            relative_features=representations.relative_features,
            local_geometry=local_geometry,
        )
        denoised_coords = self.coordinate_denoiser(
            context.to_denoising_input(noisy_coords=noisy_coords, noise_level=noise_level)
        )
        return denoised_coords, augmented_target, noise_level


class PxDesignConditionEmbedder(nn.Module):
    r"""
    Build PxDesign condition embeddings from atom features and design token features.

    Runs the atom-to-token encoder, concatenates the aggregated per-token hidden with
    the restype/pLDDT/hotspot/design features, projects to the token feature width, and
    embeds the conditional-template distance bins into the pair representation.

    Args:
        config: PxDesign configuration; its ``condition_embedder`` and
            ``conditional_template`` sub-configs parameterize this module.
    """

    def __init__(self, config: PxDesignConfig) -> None:
        super().__init__()
        embedder_config = config.condition_embedder
        self.vocab_size = embedder_config.vocab_size
        self.add_feature_size = embedder_config.add_feature_size
        self.atom_encoder = AtomToTokenEncoder(
            token_hidden_size=embedder_config.token_hidden_size,
            atom_hidden_size=embedder_config.atom_hidden_size,
            atom_pair_hidden_size=embedder_config.atom_pair_hidden_size,
            num_blocks=embedder_config.num_atom_encoder_blocks,
            num_attention_heads=embedder_config.num_atom_encoder_attention_heads,
            num_query_atoms=embedder_config.num_query_atoms,
            num_key_atoms=embedder_config.num_key_atoms,
            atom_feature_size=config.atom_feature_size,
            operator_backend=embedder_config.operator_backend,
        )
        raw_feature_size = (
            embedder_config.token_hidden_size
            + embedder_config.vocab_size
            + embedder_config.plddt_size
            + embedder_config.hotspot_size
            + 2 * embedder_config.add_feature_size
        )
        self.input_map = nn.Linear(raw_feature_size, embedder_config.token_feature_size)
        self.template_embedder = PxDesignConditionalTemplateEmbedder(
            num_template_bins=config.conditional_template.num_template_bins,
            pair_size=config.pair_size,
        )

    def forward(self, batch: PxDesignInput) -> tuple[Tensor, Tensor]:
        r"""
        Return ``(token_embedding, pair_embedding)`` PxDesign conditioning tensors.

        Args:
            batch: PxDesign model input supplying atom features, reference geometry,
                design token features, and the conditional template.

        Returns:
            A tuple ``(token_embedding, pair)`` where ``token_embedding`` has shape
            `[..., N_token, token_feature_size]` (masked to valid tokens) and ``pair``
            has shape `[..., N_token, N_token, pair_size]`.
        """

        local_geometry = self.atom_encoder.build_local_geometry(
            batch.reference_positions.to(device=batch.atom_features.device),
            batch.reference_space_id,
        )
        atom_encoder_output = self.atom_encoder(
            batch.atom_features,
            reference_positions=batch.reference_positions,
            reference_charges=torch.arcsinh(batch.reference_charges),
            reference_mask=batch.reference_mask,
            atom_mask=batch.atom_mask,
            atom_to_token=batch.atom_to_token,
            num_tokens=batch.token_mask.shape[-1],
            local_geometry=local_geometry,
        )
        token_features = self._token_features(batch, atom_encoder_output.token_hidden)
        token_embedding = self.input_map(token_features)
        token_embedding = token_embedding * batch.token_mask.unsqueeze(-1).to(
            device=token_embedding.device,
            dtype=token_embedding.dtype,
        )
        pair = self.template_embedder(batch.conditional_templ, batch.conditional_templ_mask)
        return token_embedding, pair.to(device=token_embedding.device, dtype=token_embedding.dtype)

    def _token_features(self, batch: PxDesignInput, token_hidden: Tensor) -> Tensor:
        r"""
        Concatenate the atom-encoder hidden with per-token design features.

        Args:
            batch: PxDesign model input (supplies restype, pLDDT, and hotspot).
            token_hidden: Aggregated per-token hidden from the atom encoder,
                shape `[..., N_token, token_hidden_size]`.

        Returns:
            The concatenated token feature tensor of shape
            `[..., N_token, raw_feature_size]` fed to the input projection.
        """

        restype = _restype_one_hot(batch.restype, num_classes=self.vocab_size).to(
            device=token_hidden.device,
            dtype=token_hidden.dtype,
        )
        plddt = _optional_scalar_token_feature(batch.plddt, like=restype[..., :1])
        hotspot = _optional_scalar_token_feature(batch.hotspot, like=restype[..., :1])
        add_feature = F.one_hot(
            torch.zeros(restype.shape[:-1], device=restype.device, dtype=torch.long),
            num_classes=self.add_feature_size,
        ).to(dtype=token_hidden.dtype)
        return torch.cat([token_hidden, restype, plddt, hotspot, add_feature, add_feature], dim=-1)


class PxDesignConditionalTemplateEmbedder(nn.Module):
    r"""
    Embed PxDesign conditional-template distance bins into pair states.

    Args:
        num_template_bins: Number of condition-condition distance bins; one extra
            embedding id is reserved for unconditioned pairs.
        pair_size: Width of the emitted pair representation.
    """

    def __init__(self, *, num_template_bins: int = 64, pair_size: int = 128) -> None:
        super().__init__()
        self.num_template_bins = num_template_bins
        self.embedding = nn.Embedding(num_template_bins + 1, pair_size)

    def forward(self, conditional_templ: Tensor, conditional_templ_mask: Tensor) -> Tensor:
        r"""
        Return pair embeddings for condition-template bins.

        Bin id zero is reserved for unconditioned pairs. Conditioned pairs use
        ``1 + conditional_templ`` so the 64 official bins occupy ids ``1..64``.

        Args:
            conditional_templ: Distance-bin ids per token pair, shape
                `[..., N_token, N_token]`.
            conditional_templ_mask: Boolean mask selecting conditioned pairs,
                shape `[..., N_token, N_token]`.

        Returns:
            Pair embeddings of shape `[..., N_token, N_token, pair_size]`; masked-out
            pairs receive the reserved unconditioned embedding.

        Examples:
            >>> import torch
            >>> embedder = PxDesignConditionalTemplateEmbedder(num_template_bins=4, pair_size=8)
            >>> templ = torch.tensor([[[0, 1], [2, 3]]])
            >>> mask = torch.tensor([[[True, False], [True, True]]])
            >>> embedder(templ, mask).shape
            torch.Size([1, 2, 2, 8])
        """

        bins = conditional_templ.to(dtype=torch.long).clamp(min=0, max=self.num_template_bins - 1)
        mask = conditional_templ_mask.to(device=bins.device, dtype=torch.long)
        embedding_ids = mask * (bins + 1)
        return self.embedding(embedding_ids)


class TrainingNoiseSampler(nn.Module):
    r"""
    Sample EDM training noise levels from a log-normal distribution.

    Draw ``sigma = exp(noise_log_mean + noise_log_std * N(0, 1)) * sigma_data``. Each training sample gets an
    independent noise level so the denoiser learns to denoise across the whole sigma range in a single step
    (no reverse-sampling loop during training).

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
            generator: Optional random generator.

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


class PxDesignCoordinateSampler(nn.Module):
    r"""
    PxDesign EDM sampler with the public step-index step-scale schedules.

    Wraps an [`EdmScheduler`][ofoldx.modules.diffusion.EdmScheduler] and applies the
    named per-step step-scale schedule and optional churn from the sampler config.

    Args:
        config: PxDesign configuration; its ``sampler`` sub-config parameterizes the
            schedule, churn, and step-scale behavior.
    """

    def __init__(self, config: PxDesignConfig) -> None:
        super().__init__()
        sampler = config.sampler
        self.scheduler = EdmScheduler(
            max_noise_level=sampler.max_noise_level,
            min_noise_level=sampler.min_noise_level,
            rho=sampler.rho,
            sigma_data=sampler.sigma_data,
        )
        self.gamma0 = sampler.gamma0
        self.gamma_min = sampler.gamma_min
        self.noise_scale = sampler.noise_scale
        self.step_scale_min = sampler.step_scale_min
        self.step_scale_max = sampler.step_scale_max
        self.step_scale_schedule = sampler.step_scale_schedule
        self.center_random_augmentation = sampler.center_random_augmentation

    def augment_target(
        self,
        coords: Tensor,
        *,
        num_samples: int,
        atom_mask: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        r"""
        Return ``num_samples`` centered, randomly rigid-augmented copies of a training target.

        Reuses the shared EDM centering/augmentation used by the sampler so single-step diffusion training
        draws per-sample ground-truth targets the same way inference initializes coords. Honors
        ``center_random_augmentation``: when disabled, only mask-weighted centering is applied.

        Args:
            coords: Ground-truth coords of shape `[..., N_atom, 3]`.
            num_samples: Number of augmented copies to draw along a new sample axis.
            atom_mask: Optional `[..., N_atom]` mask for weighted centering and output masking.
            generator: Optional random generator for reproducible augmentation.

        Returns:
            Augmented coords of shape `[..., num_samples, N_atom, 3]`.
        """

        return CoordinateEdmSampler._center_random_augmentation(
            coords,
            num_samples=num_samples,
            center_only=not self.center_random_augmentation,
            atom_mask=atom_mask,
            generator=generator,
        )

    def forward(
        self,
        denoise: nn.Module,
        *,
        context: CoordinateDenoiseContext,
        num_samples: int,
        num_sampling_steps: int,
        sample_chunk_size: int | None = None,
        initial_coords: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        r"""
        Run PxDesign coordinate sampling.

        Builds the EDM noise schedule and samples ``num_samples`` coordinate sets,
        optionally in chunks of ``sample_chunk_size`` to bound peak memory.

        Args:
            denoise: The coordinate denoiser invoked at each step.
            context: Static per-input denoising context.
            num_samples: Number of coordinate samples to draw.
            num_sampling_steps: Number of EDM denoising steps.
            sample_chunk_size: Samples per chunk, or ``None`` to draw all at once.
            initial_coords: Optional starting coords `[..., N_atom, 3]`.
            generator: Optional random generator seeding the sampling noise.

        Returns:
            Sampled coordinates of shape `[..., num_samples, N_atom, 3]`.

        Raises:
            ValueError: If ``num_samples`` or ``sample_chunk_size`` is not positive.
        """

        if num_samples < 1:
            raise ValueError(f"num_samples must be positive, got {num_samples}")
        if sample_chunk_size is not None and sample_chunk_size < 1:
            raise ValueError(f"sample_chunk_size must be positive, got {sample_chunk_size}")

        noise_schedule = self.scheduler(
            num_sampling_steps=num_sampling_steps, device=context.device, dtype=context.dtype
        )
        if sample_chunk_size is None:
            return self._sample_chunk(
                denoise,
                context=context,
                num_samples=num_samples,
                noise_schedule=noise_schedule,
                initial_coords=initial_coords,
                generator=generator,
            )

        chunks = []
        num_chunks = num_samples // sample_chunk_size + (num_samples % sample_chunk_size != 0)
        for chunk_index in range(num_chunks):
            chunk_num_samples = (
                sample_chunk_size if chunk_index < num_chunks - 1 else num_samples - chunk_index * sample_chunk_size
            )
            chunks.append(
                self._sample_chunk(
                    denoise,
                    context=context,
                    num_samples=chunk_num_samples,
                    noise_schedule=noise_schedule,
                    initial_coords=CoordinateEdmSampler._initial_coords_chunk(
                        initial_coords,
                        context=context,
                        num_samples=num_samples,
                        sample_start=chunk_index * sample_chunk_size,
                        chunk_num_samples=chunk_num_samples,
                    ),
                    generator=generator,
                )
            )
        return torch.cat(chunks, dim=-3)

    def _sample_chunk(
        self,
        denoise: nn.Module,
        *,
        context: CoordinateDenoiseContext,
        num_samples: int,
        noise_schedule: Tensor,
        initial_coords: Tensor | None,
        generator: torch.Generator | None,
    ) -> Tensor:
        r"""
        Sample one chunk of coordinates over the full noise schedule.

        Args:
            denoise: The coordinate denoiser invoked at each step.
            context: Static per-input denoising context.
            num_samples: Number of samples in this chunk.
            noise_schedule: Descending EDM sigma schedule, shape `[num_steps + 1]`.
            initial_coords: Optional starting coords `[..., N_atom, 3]`.
            generator: Optional random generator seeding the sampling noise.

        Returns:
            Sampled coordinates for the chunk, shape `[..., num_samples, N_atom, 3]`.
        """

        coords = CoordinateEdmSampler._initial_coords(
            context=context,
            num_samples=num_samples,
            noise_level=noise_schedule[0],
            initial_coords=initial_coords,
            generator=generator,
        )
        schedule_length = noise_schedule.shape[0]
        for step_index, (sigma_prev, sigma_next) in enumerate(zip(noise_schedule[:-1], noise_schedule[1:])):
            if self.center_random_augmentation:
                coords = CoordinateEdmSampler._center_random_augmentation(
                    coords,
                    num_samples=1,
                    generator=generator,
                ).squeeze(dim=-3)
            coords = self._step(
                denoise,
                coords,
                sigma_prev,
                sigma_next,
                context=context,
                step_index=step_index,
                schedule_length=schedule_length,
                generator=generator,
            )
        return coords

    def _step(
        self,
        denoise: nn.Module,
        coords: Tensor,
        sigma_prev: Tensor,
        sigma_next: Tensor,
        *,
        context: CoordinateDenoiseContext,
        step_index: int,
        schedule_length: int,
        generator: torch.Generator | None,
    ) -> Tensor:
        r"""
        Take one EDM sampler step from ``sigma_prev`` toward ``sigma_next``.

        Applies churn (raising the noise level to ``sigma_hat``), denoises the noised
        coords, and advances along the estimated score with the scheduled step scale.

        Args:
            denoise: The coordinate denoiser invoked for this step.
            coords: Current coordinates, shape `[..., num_samples, N_atom, 3]`.
            sigma_prev: Noise level at the current step.
            sigma_next: Noise level at the next step.
            context: Static per-input denoising context.
            step_index: Zero-based index of this step in the schedule.
            schedule_length: Total number of sigmas in the schedule.
            generator: Optional random generator seeding the churn noise.

        Returns:
            Updated coordinates, shape `[..., num_samples, N_atom, 3]`.
        """

        gamma = torch.where(
            sigma_next > self.gamma_min,
            sigma_next.new_tensor(self.gamma0),
            sigma_next.new_zeros(()),
        )
        sigma_hat = sigma_prev * (gamma + 1)
        delta_noise_level = torch.sqrt(sigma_hat**2 - sigma_prev**2)
        noisy_coords = coords + self.noise_scale * delta_noise_level * torch.empty_like(coords).normal_(
            generator=generator
        )

        batch_shape = coords.shape[:-3]
        num_samples = coords.shape[-3]
        noise_level = (
            sigma_hat.reshape((1,) * (len(batch_shape) + 1)).expand(*batch_shape, num_samples).to(coords.dtype)
        )
        denoised_coords = denoise(context.to_denoising_input(noisy_coords=noisy_coords, noise_level=noise_level))
        delta = (noisy_coords - denoised_coords) / noise_level[..., None, None]
        step_scale = self._step_scale(step_index, schedule_length, device=coords.device, dtype=coords.dtype)
        return noisy_coords + step_scale * (sigma_next - sigma_hat)[..., None, None] * delta

    def _step_scale(
        self,
        step_index: int,
        schedule_length: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        r"""
        Compute the per-step update scale for the configured schedule.

        Args:
            step_index: Zero-based index of the current step.
            schedule_length: Total number of sigmas in the schedule.
            device: Device for the returned scalar.
            dtype: Dtype for the returned scalar.

        Returns:
            A scalar tensor holding the step scale interpolated between
            ``step_scale_min`` and ``step_scale_max`` per ``step_scale_schedule``.

        Raises:
            ValueError: If ``step_scale_schedule`` is not a supported schedule name.
        """

        schedule = self.step_scale_schedule
        if schedule_length <= 0:
            fraction = 1.0
        else:
            fraction = step_index / schedule_length
        step_scale_min = float(self.step_scale_min)
        step_scale_max = float(self.step_scale_max)

        if schedule == "const":
            value = step_scale_min
        elif schedule == "linear":
            value = step_scale_min + (step_scale_max - step_scale_min) * fraction
        elif schedule == "poly":
            value = step_scale_min + (step_scale_max - step_scale_min) * fraction**2
        elif schedule == "cos":
            value = step_scale_min + 0.5 * (step_scale_max - step_scale_min) * (1 - math.cos(math.pi * fraction))
        elif schedule == "piecewise":
            value = step_scale_min if fraction < 0.5 else step_scale_max
        elif schedule == "piecewise_65":
            value = step_scale_min if fraction < 0.65 else step_scale_max
        elif schedule == "piecewise_70":
            value = step_scale_min if fraction < 0.70 else step_scale_max
        else:
            raise ValueError(f"unsupported PxDesign step_scale_schedule: {schedule!r}")
        return torch.tensor(value, device=device, dtype=dtype)


def _validate_pxdesign_input(batch: PxDesignInput, *, vocab_size: int) -> None:
    r"""
    Validate the shape consistency of a PxDesign input.

    Args:
        batch: PxDesign model input to check.
        vocab_size: Expected restype vocabulary size for one-hot restype features.

    Raises:
        ValueError: If any conditioning tensor has a shape inconsistent with the
            token mask, restype vocabulary, atom leading dims, or coordinate width.
    """

    token_shape = tuple(batch.token_mask.shape)
    for name in ("residue_index", "chain_index", "entity_index", "symmetry_index", "token_index"):
        value = getattr(batch, name)
        if tuple(value.shape) != token_shape:
            raise ValueError(f"{name} must have shape {token_shape}, got {tuple(value.shape)}")
    if batch.restype.shape not in {batch.token_mask.shape, (*batch.token_mask.shape, vocab_size)}:
        raise ValueError(
            "restype must have shape "
            f"{token_shape} or {(*token_shape, vocab_size)}, got {tuple(batch.restype.shape)}"
        )
    pair_shape = (*token_shape, token_shape[-1])
    if tuple(batch.conditional_templ.shape) != pair_shape:
        raise ValueError(f"conditional_templ must have shape {pair_shape}, got {tuple(batch.conditional_templ.shape)}")
    if tuple(batch.conditional_templ_mask.shape) != pair_shape:
        raise ValueError(
            f"conditional_templ_mask must have shape {pair_shape}, got {tuple(batch.conditional_templ_mask.shape)}"
        )
    if batch.atom_features.shape[:-1] != batch.atom_to_token.shape:
        raise ValueError("atom_features and atom_to_token must share leading atom dimensions")
    if batch.reference_positions.shape[-1] != 3:
        raise ValueError("reference_positions must end in coordinate dimension 3")
    if batch.backbone_atom_indices is not None and tuple(batch.backbone_atom_indices.shape) != (*token_shape, 4):
        raise ValueError(
            "backbone_atom_indices must have shape "
            f"{(*token_shape, 4)}, got {tuple(batch.backbone_atom_indices.shape)}"
        )


def _restype_one_hot(restype: Tensor, *, num_classes: int) -> Tensor:
    r"""
    Return restype features as a one-hot tensor.

    Passes through inputs that are already one-hot (last dim ``num_classes`` and rank
    ``>= 3``); otherwise clamps the integer ids into range and one-hot encodes them.

    Args:
        restype: Restype ids `[..., N_token]` or one-hot features `[..., N_token, num_classes]`.
        num_classes: Restype vocabulary size.

    Returns:
        One-hot restype features of shape `[..., N_token, num_classes]`.

    Examples:
        >>> import torch
        >>> _restype_one_hot(torch.tensor([[0, 2]]), num_classes=4).tolist()
        [[[1, 0, 0, 0], [0, 0, 1, 0]]]
    """

    if restype.shape[-1:] == (num_classes,) and restype.dim() >= 3:
        return restype
    return F.one_hot(restype.to(dtype=torch.long).clamp(min=0, max=num_classes - 1), num_classes=num_classes)


def _optional_scalar_token_feature(value: Tensor | None, *, like: Tensor) -> Tensor:
    r"""
    Normalize an optional per-token scalar feature to a trailing channel dim.

    A missing value yields zeros shaped like ``like``; a present value is broadcast to
    a trailing singleton channel when needed.

    Args:
        value: Optional per-token scalar `[..., N_token]` or `[..., N_token, 1]`.
        like: Reference feature `[..., N_token, 1]` supplying the target shape/dtype/device.

    Returns:
        The feature with shape matching ``like``.

    Raises:
        ValueError: If ``value`` does not match the reference token shape.

    Examples:
        >>> import torch
        >>> like = torch.zeros(1, 3, 1)
        >>> _optional_scalar_token_feature(None, like=like).shape
        torch.Size([1, 3, 1])
        >>> _optional_scalar_token_feature(torch.ones(1, 3), like=like).shape
        torch.Size([1, 3, 1])
    """

    if value is None:
        return torch.zeros_like(like)
    value = value.to(device=like.device, dtype=like.dtype)
    if value.shape == like.shape[:-1]:
        value = value.unsqueeze(-1)
    if value.shape != like.shape:
        raise ValueError(f"token scalar feature must have shape {tuple(like.shape[:-1])} or {tuple(like.shape)}")
    return value


def _restore_condition_coords(batch: PxDesignInput, pred_coords: Tensor) -> Tensor:
    r"""
    Overwrite fixed condition atoms in the sampled coords with their input coords.

    When the batch carries condition coords and a condition-atom mask, the masked
    atoms are replaced by ``batch.condition_coords`` (broadcast across the sample dim);
    otherwise the predicted coords are returned unchanged.

    Args:
        batch: PxDesign model input; uses ``condition_coords`` and ``condition_atom_mask``.
        pred_coords: Sampled coords `[..., num_samples, N_atom, 3]`.

    Returns:
        Coordinates with condition atoms restored, shape `[..., num_samples, N_atom, 3]`.
    """

    if batch.condition_coords is None or batch.condition_atom_mask is None:
        return pred_coords
    condition_coords = batch.condition_coords.to(device=pred_coords.device, dtype=pred_coords.dtype)
    if condition_coords.shape == pred_coords.shape[:-3] + pred_coords.shape[-2:]:
        condition_coords = condition_coords.unsqueeze(-3)
    condition_mask = batch.condition_atom_mask.to(device=pred_coords.device, dtype=torch.bool)
    while condition_mask.dim() < pred_coords.dim() - 1:
        condition_mask = condition_mask.unsqueeze(-2)
    return torch.where(condition_mask.unsqueeze(-1), condition_coords, pred_coords)


@register_pytree_dataclass
@dataclass(slots=True)
class PxDesignRepresentations:
    r"""
    Encoded PxDesign conditioning held fixed across all EDM sampling steps.

    Holds the per-input conditioning tensors that stay fixed across all EDM sampling steps and are passed
    into the coordinate denoiser context. The token single/pair states use the typed
    [`Single`][ofoldx.modules.representation.Single] / [`Pair`][ofoldx.modules.representation.Pair] carriers
    at this boundary (unwrapped via ``.tensor`` before the denoiser); both carriers and this dataclass are
    pytree-registered so ``torch.compile`` traces through them without a graph break.

    Attributes:
        token_embedding: Per-token conditioning embedding `[..., N_token, token_feature_size]`.
        single: Single (token) representation `[..., N_token, single_size]` carrier,
            zero-initialized for PxDesign.
        pair: Pair representation `[..., N_token, N_token, pair_size]` carrier, masked to
            valid token pairs.
        relative_features: Bucketed relative-position features shared across steps.
    """

    token_embedding: Tensor
    single: Single
    pair: Pair
    relative_features: RelativePositionBuckets


@register_pytree_dataclass
@dataclass(slots=True, kw_only=True)
class PxDesignInput(BaseInput):
    r"""
    Batched tensor contract consumed by the native PxDesign generator.

    Carries the per-token/per-atom conditioning, reference-conformer geometry, restype and template features,
    and optional design/condition annotations that drive coordinate sampling. Registered as a pytree node so
    ``torch.compile`` traces through the batch and the ``model(**inputs)`` keyword-tensor call path can build
    it outside the compiled graph.

    Attributes:
        token_mask: Valid token mask with shape ``(B, N_token)``.
        residue_index: Residue numbers for relative-position features.
        chain_index: Chain ids mapped to contiguous integer indices.
        entity_index: Entity ids shared by equivalent chains.
        symmetry_index: Symmetry/copy ids.
        token_index: Token ids in structure order.
        atom_features: Per-atom reference features.
        atom_to_token: Token index for each atom.
        atom_mask: Valid atom mask.
        reference_positions: Centered reference-conformer atom coords.
        reference_space_id: Reference-conformer group id per atom.
        reference_mask: Valid reference-conformer atom mask.
        reference_charges: Per-atom reference formal charges.
        restype: PxDesign restype ids ``(B, N_token)`` or one-hot features
            ``(B, N_token, 36)``. Official design placeholders occupy ids
            ``32..35`` (``xpb``, ``xpa``, ``rbb``, ``raa``).
        conditional_templ: Conditional-template distance-bin ids in ``[0, 63]``.
        conditional_templ_mask: Mask selecting condition-condition token pairs.
        design_token_mask: Optional mask for generated/design tokens.
        condition_token_mask: Optional mask for fixed/conditioning tokens.
        condition_atom_mask: Optional atom mask for fixed/conditioning atoms.
        condition_coords: Optional coords used to restore fixed condition atoms.
        backbone_atom_indices: Optional atom indices for ProteinMPNN ``N/CA/C/O``
            backbone coords, shaped ``(B, N_token, 4)`` with ``-1`` for missing
            atoms.
        plddt: Optional per-token pLDDT conditioning scalar.
        hotspot: Optional per-token hotspot indicator.
        distogram_rep_atom_mask: Optional ``(B, N_atom)`` mask selecting the AF3
            distogram representative atom per token (Cβ/Cα, C4/C2/C1', all-ligand).
            Preserved from Protenix-coupled featurization for faithful design
            conditioning rather than lossily re-derived.
        frame_atom_index: Optional ``(B, N_token, 3)`` token-frame atom indices.
        frame_mask: Optional ``(B, N_token)`` token-frame validity mask.
        conditional_label: Optional ``(B, N_atom)`` bool, ``True`` for condition
            atoms and ``False`` for design atoms (PxDesign design annotation).
        is_resolved: Optional ``(B, N_atom)`` bool marking atoms with a real
            experimental coordinate (PxDesign design annotation).
        target_positions: Optional ground-truth atom coords ``(B, N_atom, 3)`` used as the
            single-step diffusion training target; required by ``forward`` (training) and unused at inference.
        target_mask: Optional ``(B, N_atom)`` mask selecting atoms with a real training coordinate for
            masked target centering; defaults to ``atom_mask`` when absent.
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
    reference_mask: Tensor
    reference_charges: Tensor
    restype: Tensor
    conditional_templ: Tensor
    conditional_templ_mask: Tensor
    design_token_mask: Tensor | None = None
    condition_token_mask: Tensor | None = None
    condition_atom_mask: Tensor | None = None
    condition_coords: Tensor | None = None
    backbone_atom_indices: Tensor | None = None
    plddt: Tensor | None = None
    hotspot: Tensor | None = None
    distogram_rep_atom_mask: Tensor | None = None
    frame_atom_index: Tensor | None = None
    frame_mask: Tensor | None = None
    conditional_label: Tensor | None = None
    is_resolved: Tensor | None = None
    target_positions: Tensor | None = None
    target_mask: Tensor | None = None


PxDesignDesigner.input_class = PxDesignInput
