r"""
ODesign structure-generation model and tensor interfaces.
"""

from __future__ import annotations

import math
import operator
from typing import NamedTuple

import torch
import torch.nn.functional as F
from tensordict import tensorclass
from torch import Tensor, nn

from ...data import MSA_VOCAB_SIZE, TOKEN_VOCAB_SIZE
from ...data.molecular import MolecularSystem
from ...model import Model, _TensorCollection
from ...modules.atom import (
    LocalAtomGeometry,
    NormalizedKeyAtomToTokenEncoder,
)
from ...modules.backend import LayerNorm
from ...modules.denoiser import CoordinateDenoiser
from ...modules.diffusion import (
    CoordinateDenoiseContext,
    CoordinateEdmSampler,
    denoise_coordinates,
)
from ...modules.embedding import (
    RelativePositionBuckets,
    RelativePositionEmbedder,
    generate_relative_position_features,
)
from ...modules.functional import pair_mask_from_token_mask
from ...modules.geometry import uniform_random_rotation
from ...modules.head import DistogramHead
from ...modules.trunk import MsaTrunk, PairformerStack, TrianglePairUpdateStack
from .config import ODesignConfig

_ATOM_FEATURE_SIZE = 386
# Canonical residue identity + MSA profile + deletion mean + hotspot.
_TOKEN_FEATURE_SIZE = TOKEN_VOCAB_SIZE + MSA_VOCAB_SIZE + 2
_DESIGN_FEATURE_SIZE = 4
_NUM_BOND_CLASSES = 2
_DEFAULT_RECYCLING_STEPS = 10


class ODesignModel(Model):
    r"""
    Generate all-atom biomolecular structures with ODesign.

    Accepts [`ODesignInput`][ofoldx.models.odesign.model.ODesignInput] and returns
    [`ODesignOutput`][ofoldx.models.odesign.model.ODesignOutput].

    Args:
        config: Model architecture configuration.

    Examples:
        >>> from ofoldx.models.odesign import ODesignConfig, ODesignModel
        >>> config = ODesignConfig({
        ...     "single_size": 16,
        ...     "pair_size": 8,
        ...     "constraint_embeddings": {
        ...         "hidden_size": 8, "num_blocks": 1,
        ...         "num_distogram_bins": 4, "dropout": 0.0,
        ...     },
        ...     "trunk": {
        ...         "num_msa_stages": 1, "num_pairformer_blocks": 1,
        ...         "pairformer_dropout": 0.0, "msa_dropout": 0.0, "pair_dropout": 0.0,
        ...     },
        ...     "denoiser": {
        ...         "num_token_encoder_blocks": 1,
        ...         "num_token_encoder_attention_heads": 1,
        ...         "atom_hidden_size": 8,
        ...         "atom_pair_hidden_size": 4,
        ...         "token_hidden_size": 16,
        ...         "num_atom_encoder_blocks": 1,
        ...         "num_atom_encoder_attention_heads": 1,
        ...         "num_atom_decoder_blocks": 1,
        ...         "num_atom_decoder_attention_heads": 1,
        ...         "num_query_atoms": 4,
        ...         "num_key_atoms": 4,
        ...     },
        ...     "pairwise": {
        ...         "num_distogram_bins": 4,
        ...         "bond_hidden_size": 8,
        ...     },
        ... })
        >>> model = ODesignModel(config).eval()
        >>> model.distogram_head.projection.out_features
        4
        >>> import torch
        >>> from ofoldx.data import Structure
        >>> from ofoldx.models.odesign import ODesignProcessor
        >>> input = ODesignProcessor()(Structure.from_sequence("ACDEF"))
        >>> output = model.sample(
        ...     input, num_sampling_steps=2, generator=torch.Generator().manual_seed(0)
        ... )
        >>> tuple(output.coordinates.shape)
        (1, 1, 1, 40, 3)
    """

    config_class = ODesignConfig
    config: ODesignConfig

    def __init__(self, config: ODesignConfig | None = None) -> None:
        config = ODesignConfig() if config is None else config
        super().__init__(config)
        self.input_embeddings = ODesignInputEmbeddings(
            single_size=config.single_size,
        )
        token_embedding_size = config.single_size + _TOKEN_FEATURE_SIZE + _DESIGN_FEATURE_SIZE
        self.single_init_projection = nn.Linear(token_embedding_size, config.single_size, bias=False)
        self.pair_left_projection = nn.Linear(config.single_size, config.pair_size, bias=False)
        self.pair_right_projection = nn.Linear(config.single_size, config.pair_size, bias=False)
        self.relative_position_embedder = RelativePositionEmbedder(
            4 * config.max_relative_residue + 2 * config.max_relative_chain + 7,
            config.pair_size,
        )
        self.token_bond_projection = nn.Linear(1, config.pair_size, bias=False)
        self.pair_recycle_norm = LayerNorm(config.pair_size)
        self.pair_recycle_projection = nn.Linear(config.pair_size, config.pair_size, bias=False)
        self.constraint_embeddings = ODesignConstraintEmbeddings(
            pair_size=config.pair_size,
            hidden_size=config.constraint_embeddings.hidden_size,
            num_blocks=config.constraint_embeddings.num_blocks,
            num_bins=config.constraint_embeddings.num_distogram_bins,
            min_distance=config.constraint_embeddings.min_distance,
            max_distance=config.constraint_embeddings.max_distance,
            dropout=config.constraint_embeddings.dropout,
            backend="auto",
        )
        self.msa_trunk = MsaTrunk(
            pair_size=config.pair_size,
            msa_vocab_size=MSA_VOCAB_SIZE,
            token_feature_size=token_embedding_size,
            num_stages=config.trunk.num_msa_stages,
            msa_dropout=config.trunk.msa_dropout,
            pair_dropout=config.trunk.pair_dropout,
        )
        self.single_recycle_norm = LayerNorm(config.single_size)
        self.single_recycle_projection = nn.Linear(config.single_size, config.single_size, bias=False)
        self.pairformer_stack = PairformerStack(
            num_blocks=config.trunk.num_pairformer_blocks,
            single_size=config.single_size,
            pair_size=config.pair_size,
            dropout=config.trunk.pairformer_dropout,
        )
        self.coordinate_denoiser = CoordinateDenoiser(
            num_encoder_blocks=config.denoiser.num_token_encoder_blocks,
            num_encoder_attention_heads=config.denoiser.num_token_encoder_attention_heads,
            sigma_data=config.denoiser.sigma_data,
            atom_hidden_size=config.denoiser.atom_hidden_size,
            atom_pair_hidden_size=config.denoiser.atom_pair_hidden_size,
            token_hidden_size=config.denoiser.token_hidden_size,
            single_size=config.single_size,
            pair_size=config.pair_size,
            token_feature_size=token_embedding_size,
            atom_feature_size=_ATOM_FEATURE_SIZE,
            num_atom_encoder_blocks=config.denoiser.num_atom_encoder_blocks,
            num_atom_encoder_attention_heads=config.denoiser.num_atom_encoder_attention_heads,
            num_atom_decoder_blocks=config.denoiser.num_atom_decoder_blocks,
            num_atom_decoder_attention_heads=config.denoiser.num_atom_decoder_attention_heads,
            num_query_atoms=config.denoiser.num_query_atoms,
            num_key_atoms=config.denoiser.num_key_atoms,
        )
        self.coordinate_sampler = CoordinateEdmSampler(reverse_augmentation=True)
        self.distogram_head = DistogramHead(config.pair_size, config.pairwise.num_distogram_bins)
        self.token_bond_head = ODesignTokenBondHead(
            pair_size=config.pair_size,
            hidden_size=config.pairwise.bond_hidden_size,
        )
        nn.init.zeros_(self.pair_recycle_projection.weight)
        nn.init.zeros_(self.single_recycle_projection.weight)
        nn.init.zeros_(self.distogram_head.projection.weight)
        if self.distogram_head.projection.bias is not None:
            nn.init.zeros_(self.distogram_head.projection.bias)
        self.post_init()

    def forward(
        self,
        input: ODesignInput | ODesignLigandInput,
        target: ODesignStructureTarget,
        *,
        num_samples: int = 1,
        sample_chunk_size: int | None = None,
        synchronize_noise_levels: bool = False,
        coordinate_augmentation: bool = True,
        augmentation_translation_scale: float = 1.0,
        noise_log_mean: float = -1.2,
        noise_log_std: float = 1.5,
        recycling_steps: int | None = None,
        use_conditioning: bool = True,
        generator: torch.Generator | None = None,
    ) -> ODesignOutput:
        r"""
        Denoise randomly corrupted target coordinates.

        ``B`` is batch size, ``S`` is the coordinate-sample axis, ``N`` is token
        count, and ``A`` is atom count.

        Args:
            input: Processed structure-design input.
            target: Coordinates already aligned to the Model atom axis.
            num_samples: Number ``S`` of independently corrupted copies per input.
            sample_chunk_size: Maximum samples per input evaluated by one
                denoiser call. ``None`` evaluates all samples together.
            synchronize_noise_levels: Whether samples from one input item share one
                noise level.
            coordinate_augmentation: Whether to rotate and translate the centered
                target independently for each sample.
            augmentation_translation_scale: Standard deviation of augmentation
                translations in angstroms.
            noise_log_mean: Mean of the log-normal noise distribution.
            noise_log_std: Standard deviation of the log-normal noise distribution.
            recycling_steps: Number of trunk recycling passes. ``None`` uses the
                released default of 10.
            use_conditioning: Whether trunk single and pair values condition the
                coordinate denoiser.
            generator: Optional generator for rigid augmentation, noise levels, and
                coordinate noise.

        Returns:
            Denoised coordinates in the Target's original frame with shape
            ``(B, S, A, 3)``, noise levels with shape ``(B, S)``, and pairwise
            logits computed once from the trunk.
        """

        num_samples = operator.index(num_samples)
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        if sample_chunk_size is None:
            sample_chunk_size = num_samples
        else:
            sample_chunk_size = operator.index(sample_chunk_size)
            if sample_chunk_size < 1:
                raise ValueError("sample_chunk_size must be positive")
            sample_chunk_size = min(sample_chunk_size, num_samples)
        steps = _DEFAULT_RECYCLING_STEPS if recycling_steps is None else recycling_steps
        trunk = self._run_trunk(input, recycling_steps=steps)
        target_coordinates = target.coordinates.to(input.atom_features)
        target_mask = target.coordinate_mask.to(device=target_coordinates.device).bool() & input.atom_mask.bool()
        batch_size = target_coordinates.shape[0]
        augmented_target_coordinates, augmentation_transform = _augment_odesign_target(
            target_coordinates,
            target_mask,
            num_samples=num_samples,
            augmentation=coordinate_augmentation,
            translation_scale=augmentation_translation_scale,
            generator=generator,
        )
        noise_count = batch_size if synchronize_noise_levels else batch_size * num_samples
        normal = torch.empty(
            noise_count,
            device=target_coordinates.device,
            dtype=target_coordinates.dtype,
        ).normal_(generator=generator)
        noise_level = (
            self.coordinate_denoiser.preconditioner.sigma_data * (noise_log_mean + noise_log_std * normal).exp()
        )
        if synchronize_noise_levels:
            noise_level = noise_level.repeat_interleave(num_samples)
        noise_level = noise_level.reshape(batch_size, num_samples)
        coordinate_noise = torch.empty_like(augmented_target_coordinates).normal_(generator=generator)
        fixed_coordinate_mask = target_mask & input.fixed_coordinate_mask.bool()
        fixed_coordinate_mask = fixed_coordinate_mask.unsqueeze(-2).expand(-1, num_samples, -1)
        coordinate_noise = coordinate_noise.masked_fill(fixed_coordinate_mask.unsqueeze(-1), 0)
        noisy_coordinates = augmented_target_coordinates + noise_level[..., None, None] * coordinate_noise
        raw_output = self._denoise_from_trunk(
            input,
            trunk,
            noisy_coordinates=noisy_coordinates,
            noise_level=noise_level,
            use_conditioning=use_conditioning,
            sample_chunk_size=sample_chunk_size,
        )
        augmented_coordinates = torch.where(
            fixed_coordinate_mask.unsqueeze(-1),
            augmented_target_coordinates,
            raw_output.coordinates,
        )
        coordinates = _reverse_odesign_augmentation(
            augmented_coordinates,
            augmentation_transform,
        )
        coordinates = torch.where(
            fixed_coordinate_mask.unsqueeze(-1),
            target_coordinates.unsqueeze(-3).expand(-1, num_samples, -1, -1),
            coordinates,
        )
        return ODesignOutput(
            coordinates=coordinates,
            distogram_logits=raw_output.distogram_logits,
            token_bond_logits=raw_output.token_bond_logits,
            noise_level=noise_level,
            batch_size=[coordinates.shape[0]],
        )

    def denoise(
        self,
        input: ODesignInput | ODesignLigandInput,
        *,
        noisy_coordinates: Tensor,
        noise_level: Tensor,
        recycling_steps: int | None = None,
        use_conditioning: bool = True,
    ) -> ODesignRawOutput:
        r"""
        Denoise atom coordinates at specified EDM noise levels.

        Args:
            input: Processed structure-design input.
            noisy_coordinates: Noisy atom coordinates with shape ``(B, S, A, 3)`` in
                angstroms.
            noise_level: Diffusion noise levels with shape ``(B, S)``.
            recycling_steps: Number of trunk recycling passes. ``None`` uses the
                released default of 10.
            use_conditioning: Whether trunk single and pair values condition the
                coordinate denoiser.

        Returns:
            Denoised coordinates and pairwise logits.
        """

        steps = _DEFAULT_RECYCLING_STEPS if recycling_steps is None else recycling_steps
        trunk = self._run_trunk(input, recycling_steps=steps)
        return self._denoise_from_trunk(
            input,
            trunk,
            noisy_coordinates=noisy_coordinates,
            noise_level=noise_level,
            use_conditioning=use_conditioning,
        )

    @torch.inference_mode()
    def sample(
        self,
        input: ODesignInput | ODesignLigandInput,
        *,
        num_samples: int = 1,
        recycling_steps: int | None = None,
        num_sampling_steps: int = 200,
        num_model_seeds: int = 1,
        partial_diffusion_snr: float | None = None,
        generator: torch.Generator | None = None,
    ) -> ODesignSamplingOutput:
        r"""
        Sample coordinates while retaining the model's pairwise predictions.

        ``K`` is ``num_model_seeds``, ``B`` is batch size, ``S`` is ``num_samples``,
        ``N`` is token count, and ``A`` is atom count.

        Args:
            input: Processed design input.
            num_samples: Number of coordinate samples per model seed.
            recycling_steps: Number of trunk recycling passes. ``None`` uses the
                released default of 10.
            num_sampling_steps: Number of coordinate-diffusion steps.
            num_model_seeds: Number of independent trunk evaluations.
            partial_diffusion_snr: Signal-to-noise ratio for partial diffusion from
                ``input.observed_positions``. ``None`` starts from noise.
            generator: Random generator used by coordinate sampling.

        Returns:
            An [`ODesignSamplingOutput`][] with ``coordinates`` of shape
            ``(B, K, S, A, 3)`` in angstroms, ``distogram_logits`` of shape
            ``(B, K, N, N, D)``, and ``token_bond_logits`` of shape
            ``(B, K, N, N, 2)``.
        """

        steps = _DEFAULT_RECYCLING_STEPS if recycling_steps is None else recycling_steps
        partial_diffusion = partial_diffusion_snr is not None
        if partial_diffusion and not bool(input.observed_mask.all()):
            raise ValueError("partial diffusion requires observed coordinates for every model atom")
        if partial_diffusion and partial_diffusion_snr <= 0:
            raise ValueError("partial_diffusion_snr must be positive")
        max_noise_level = 1 / math.sqrt(partial_diffusion_snr + 1e-6) if partial_diffusion_snr is not None else None
        coordinates = []
        distograms = []
        bond_logits = []
        for _ in range(num_model_seeds):
            trunk = self._run_trunk(input, recycling_steps=steps)
            distograms.append(self.distogram_head(trunk.pair))
            bond_logits.append(self.token_bond_head(trunk.pair))
            context = self._denoise_context(input, trunk)
            coordinates.append(
                self.coordinate_sampler(
                    self.coordinate_denoiser.prepare(context),
                    context=context,
                    num_samples=num_samples,
                    num_sampling_steps=num_sampling_steps,
                    initial_coords=input.observed_positions if partial_diffusion else None,
                    condition_coords=input.observed_positions,
                    condition_mask=input.fixed_coordinate_mask,
                    max_noise_level=max_noise_level,
                    generator=generator,
                )
            )
        return ODesignSamplingOutput(
            coordinates=torch.stack(coordinates, dim=1),
            distogram_logits=torch.stack(distograms, dim=1),
            token_bond_logits=torch.stack(bond_logits, dim=1),
            batch_size=[input.atom_features.shape[0]],
        )

    def _denoise_from_trunk(
        self,
        input: ODesignInput | ODesignLigandInput,
        trunk: _ODesignTrunkOutput,
        *,
        noisy_coordinates: Tensor,
        noise_level: Tensor,
        use_conditioning: bool,
        sample_chunk_size: int | None = None,
    ) -> ODesignRawOutput:
        context = self._denoise_context(
            input,
            trunk,
            use_conditioning=use_conditioning,
        )
        coordinates = denoise_coordinates(
            self.coordinate_denoiser.prepare(context),
            context=context,
            noisy_coords=noisy_coordinates,
            noise_level=noise_level,
            sample_chunk_size=sample_chunk_size,
        )
        return ODesignRawOutput(
            coordinates=coordinates,
            distogram_logits=self.distogram_head(trunk.pair),
            token_bond_logits=self.token_bond_head(trunk.pair),
            batch_size=[coordinates.shape[0]],
        )

    def _run_trunk(
        self,
        input: ODesignInput | ODesignLigandInput,
        *,
        recycling_steps: int,
    ) -> _ODesignTrunkOutput:
        local_geometry = self.input_embeddings.build_local_geometry(
            input.reference_positions,
            input.reference_space_id,
            input.atom_mask,
        )
        token_embedding = self.input_embeddings(
            input.atom_features,
            reference_positions=input.reference_positions,
            reference_charges=input.reference_charges,
            reference_mask=input.reference_mask,
            token_features=input.token_features,
            design_features=input.design_features,
            atom_to_token=input.atom_to_token,
            num_tokens=input.token_features.shape[-2],
            atom_mask=input.atom_mask,
            token_mask=input.token_mask,
            local_geometry=local_geometry,
        )
        single_init = self.single_init_projection(token_embedding)
        relative_features = generate_relative_position_features(
            residue_index=input.residue_index,
            chain_index=input.chain_index,
            entity_index=input.entity_index,
            symmetry_index=input.symmetry_index,
            token_index=input.token_index,
            cyclic_period=input.cyclic_period,
            max_relative_residue=self.config.max_relative_residue,
            max_relative_chain=self.config.max_relative_chain,
            dtype=single_init.dtype,
        )
        pair_mask = pair_mask_from_token_mask(input.token_mask)
        pair_init = self._initial_pair(single_init, input, relative_features)
        deletion_counts = input.msa_deletion_counts.to(torch.float32)
        msa_continuous_features = torch.stack(
            (
                (deletion_counts > 0).to(torch.float32),
                (2 / math.pi) * torch.atan(deletion_counts / 3),
            ),
            dim=-1,
        ).to(token_embedding.dtype)
        single = torch.zeros_like(single_init)
        pair = torch.zeros_like(pair_init)
        grad_enabled = torch.is_grad_enabled()
        for cycle in range(recycling_steps):
            with torch.set_grad_enabled(grad_enabled and cycle == recycling_steps - 1):
                pair = pair_init + self.pair_recycle_projection(self.pair_recycle_norm(pair))
                pair = pair + self.constraint_embeddings(
                    pair,
                    input.constraint_distogram,
                    pair_mask=pair_mask,
                )
                _, pair = self.msa_trunk(
                    input.msa_tokens,
                    msa_continuous_features,
                    pair,
                    token_embedding,
                    msa_mask=input.msa_mask,
                    pair_mask=pair_mask,
                )
                single = single_init + self.single_recycle_projection(self.single_recycle_norm(single))
                single, pair = self.pairformer_stack(single, pair, pair_mask=pair_mask)
        return _ODesignTrunkOutput(
            token_embedding=token_embedding,
            single=single,
            pair=pair,
            relative_features=relative_features,
        )

    def _initial_pair(
        self,
        single: Tensor,
        input: ODesignInput | ODesignLigandInput,
        relative_features: RelativePositionBuckets,
    ) -> Tensor:
        pair = self.pair_left_projection(single).unsqueeze(-2)
        pair = pair + self.pair_right_projection(single).unsqueeze(-3)
        pair = pair + self.relative_position_embedder(relative_features)
        pair = pair + self.token_bond_projection(input.token_bonds.unsqueeze(-1))
        return pair * pair_mask_from_token_mask(input.token_mask).unsqueeze(-1).to(pair.dtype)

    def _denoise_context(
        self,
        input: ODesignInput | ODesignLigandInput,
        trunk: _ODesignTrunkOutput,
        *,
        use_conditioning: bool = True,
    ) -> CoordinateDenoiseContext:
        conditioning_single = trunk.single if use_conditioning else 0 * trunk.single
        conditioning_pair = trunk.pair if use_conditioning else 0 * trunk.pair
        return CoordinateDenoiseContext(
            atom_features=input.atom_features,
            atom_to_token=input.atom_to_token,
            reference_positions=input.reference_positions,
            reference_charges=input.reference_charges,
            reference_mask=input.reference_mask,
            atom_mask=input.atom_mask,
            token_embedding=trunk.token_embedding,
            single=trunk.single,
            conditioning_single=conditioning_single,
            conditioning_pair=conditioning_pair,
            relative_features=trunk.relative_features,
            local_geometry=self.coordinate_denoiser.build_local_geometry(
                input.reference_positions,
                input.reference_space_id,
                input.atom_mask,
            ),
        )


class ODesignInputEmbeddings(nn.Module):
    r"""
    Aggregate reference atoms and append ODesign token features.
    """

    def __init__(
        self,
        single_size: int = 384,
    ) -> None:
        super().__init__()
        self.atom_encoder = NormalizedKeyAtomToTokenEncoder(
            token_hidden_size=single_size,
            atom_feature_size=_ATOM_FEATURE_SIZE,
        )

    def forward(
        self,
        atom_features: Tensor,
        *,
        reference_positions: Tensor,
        reference_charges: Tensor,
        reference_mask: Tensor,
        token_features: Tensor,
        design_features: Tensor,
        atom_to_token: Tensor,
        num_tokens: int,
        atom_mask: Tensor,
        token_mask: Tensor,
        local_geometry: LocalAtomGeometry,
    ) -> Tensor:
        encoded = self.atom_encoder(
            atom_features,
            reference_positions=reference_positions,
            reference_charges=torch.arcsinh(reference_charges),
            reference_mask=reference_mask,
            atom_mask=atom_mask,
            atom_to_token=atom_to_token,
            num_tokens=num_tokens,
            local_geometry=local_geometry,
        ).token_hidden
        output = torch.cat(
            [encoded, token_features.to(encoded), design_features.to(encoded)],
            dim=-1,
        )
        return output * token_mask.unsqueeze(-1).to(output.dtype)

    def build_local_geometry(
        self,
        reference_positions: Tensor,
        reference_space_id: Tensor,
        atom_mask: Tensor,
    ) -> LocalAtomGeometry:
        return self.atom_encoder.build_local_geometry(
            reference_positions,
            reference_space_id,
            atom_mask,
        )


class ODesignConstraintEmbeddings(nn.Module):
    r"""
    Embed pairwise target distances into the ODesign pair representation.
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
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.num_bins = num_bins
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.pair_norm = LayerNorm(pair_size, backend=backend)
        self.pair_projection = nn.Linear(pair_size, hidden_size, bias=False)
        self.distogram_projection = nn.Linear(num_bins, hidden_size, bias=False)
        self.pair_update_stack = TrianglePairUpdateStack(
            num_blocks=num_blocks,
            pair_size=hidden_size,
            dropout=dropout,
            backend=backend,
        )
        self.hidden_norm = LayerNorm(hidden_size, backend=backend)
        self.activation = nn.ReLU()
        self.output_projection = nn.Linear(hidden_size, pair_size, bias=False)

    def forward(self, pair: Tensor, distances: Tensor, pair_mask: Tensor | None = None) -> Tensor:
        boundaries = torch.linspace(
            self.min_distance,
            self.max_distance,
            self.num_bins - 1,
            device=pair.device,
            dtype=pair.dtype,
        )
        bins = (distances.to(pair.dtype) > boundaries).sum(dim=-1)
        distogram = F.one_hot(bins, self.num_bins).to(pair.dtype)
        hidden = self.pair_projection(self.pair_norm(pair)) + self.distogram_projection(distogram)
        hidden = self.pair_update_stack(hidden, pair_mask=pair_mask)
        return self.output_projection(self.activation(self.hidden_norm(hidden)))


class ODesignTokenBondHead(nn.Module):
    r"""
    Predict whether each ODesign token pair is covalently connected.
    """

    def __init__(self, *, pair_size: int = 128, hidden_size: int = 128) -> None:
        super().__init__()
        self.input_projection = nn.Linear(pair_size, hidden_size)
        self.output_projection = nn.Linear(hidden_size, _NUM_BOND_CLASSES)

    def forward(self, pair: Tensor) -> Tensor:
        return self.output_projection(self.input_projection(pair))


def _augment_odesign_target(
    coordinates: Tensor,
    coordinate_mask: Tensor,
    *,
    num_samples: int,
    augmentation: bool,
    translation_scale: float,
    generator: torch.Generator | None,
) -> tuple[Tensor, tuple[Tensor | None, Tensor, Tensor]]:
    mask = coordinate_mask.unsqueeze(-1).to(dtype=coordinates.dtype)
    center = (coordinates * mask).sum(dim=-2, keepdim=True) / mask.sum(
        dim=-2,
        keepdim=True,
    ).clamp_min(1e-12)
    centered = (coordinates - center).unsqueeze(-3).expand(-1, num_samples, -1, -1)
    batch_size = coordinates.shape[0]
    rotation: Tensor | None
    if augmentation:
        rotation = uniform_random_rotation(
            batch_size * num_samples,
            device=coordinates.device,
            dtype=coordinates.dtype,
            generator=generator,
        ).reshape(batch_size, num_samples, 3, 3)
        translation = translation_scale * torch.empty(
            batch_size,
            num_samples,
            3,
            device=coordinates.device,
            dtype=coordinates.dtype,
        ).normal_(generator=generator)
        augmented = torch.einsum("bsad,bsde->bsae", centered, rotation)
        augmented = augmented + translation.unsqueeze(-2)
    else:
        rotation = None
        translation = torch.zeros(
            batch_size,
            num_samples,
            3,
            device=coordinates.device,
            dtype=coordinates.dtype,
        )
        augmented = centered
    augmented = augmented * mask.unsqueeze(-3)
    return augmented, (rotation, translation, center)


def _reverse_odesign_augmentation(
    coordinates: Tensor,
    transform: tuple[Tensor | None, Tensor, Tensor],
) -> Tensor:
    rotation, translation, center = transform
    coordinates = coordinates - translation.unsqueeze(-2)
    if rotation is not None:
        coordinates = torch.einsum(
            "bsad,bsde->bsae",
            coordinates,
            rotation.transpose(-1, -2),
        )
    return coordinates + center.unsqueeze(-3)


# ===========================================================================
# Typed tensor and semantic records for ODesign.
# ===========================================================================


class _ODesignTrunkOutput(NamedTuple):
    r"""
    Named representations produced by the ODesign trunk.
    """

    token_embedding: Tensor
    single: Tensor
    pair: Tensor
    relative_features: RelativePositionBuckets


@tensorclass
class ODesignInput:
    r"""
    Tensor and molecular-layout input for ODesign.

    ``B`` is batch size, ``N`` token count, ``A`` atom count, and ``M`` MSA depth.
    Processors produce ``B=1``. A larger explicit ``B`` represents repeated
    tensors for the same ``system`` and layout, not heterogeneous structures.

    Attributes:
        token_mask: Valid-token mask with shape ``(B, N)``.
        residue_index: Chain-local residue indices with shape ``(B, N)``.
        chain_index: Molecular-instance indices with shape ``(B, N)``.
        entity_index: Molecular-entity indices with shape ``(B, N)``.
        symmetry_index: Copy indices within each entity with shape ``(B, N)``.
        token_index: Token indices within each residue with shape ``(B, N)``.
        cyclic_period: Optional cyclic-position periods with shape ``(B, N)``.
        token_features: Canonical residue, MSA-profile, deletion, and hotspot
            features with shape ``(B, N, 78)``.
        design_features: Generated-protein, generated-RNA, generated-DNA, and
            atom-token indicators with shape ``(B, N, 4)``. The Model prepends
            its atom embedding to both feature tensors.
        token_bonds: Token-level bond adjacency with shape ``(B, N, N)``.
        atom_features: Reference atom features with shape ``(B, A, 386)``.
        atom_to_token: Token index for each atom with shape ``(B, A)``.
        atom_mask: Valid-atom mask with shape ``(B, A)``.
        reference_positions: Reference conformer coordinates with shape
            ``(B, A, 3)`` in angstroms.
        reference_space_id: Reference conformer group IDs with shape ``(B, A)``.
        reference_mask: Reference-coordinate validity mask with shape ``(B, A)``.
        reference_charges: Reference formal charges with shape ``(B, A, 1)``.
        observed_positions: Observed atom coordinates with shape ``(B, A, 3)`` in
            angstroms.
        observed_mask: Observed-coordinate validity mask with shape ``(B, A)``.
        fixed_coordinate_mask: Atoms fixed during sampling with shape ``(B, A)``.
        constraint_distogram: Raw observed pair distances with shape
            ``(B, N, N, 1)``; the Model performs released 39-bin encoding.
        msa_tokens: Compact MSA token indices with shape ``(B, M, N)``.
        msa_deletion_counts: Raw int32 MSA deletion counts with shape ``(B, M, N)``.
        msa_mask: MSA validity mask with shape ``(B, M, N)``.
        system: Complete canonical output topology.
        atom_index: Model atom to canonical atom mapping with shape ``(B, A)``.
        token_component_index: Token to canonical component mapping with shape
            ``(B, N)``.
        token_atom_index: Atom-token to canonical atom mapping with shape
            ``(B, N)``; residue tokens use ``-1``.
        representative_atom_index: Model atom row used by the distogram target
            for each token, with shape ``(B, N)``.
        design_component_mask: Generated polymer occurrences on the complete
            canonical component axis.
    """

    token_mask: Tensor
    residue_index: Tensor
    chain_index: Tensor
    entity_index: Tensor
    symmetry_index: Tensor
    token_index: Tensor
    cyclic_period: Tensor
    token_features: Tensor
    design_features: Tensor
    token_bonds: Tensor
    atom_features: Tensor
    atom_to_token: Tensor
    atom_mask: Tensor
    reference_positions: Tensor
    reference_space_id: Tensor
    reference_mask: Tensor
    reference_charges: Tensor
    observed_positions: Tensor
    observed_mask: Tensor
    fixed_coordinate_mask: Tensor
    constraint_distogram: Tensor
    msa_tokens: Tensor
    msa_deletion_counts: Tensor
    msa_mask: Tensor
    system: MolecularSystem
    atom_index: Tensor
    token_component_index: Tensor
    token_atom_index: Tensor
    representative_atom_index: Tensor
    design_component_mask: Tensor


@tensorclass
class ODesignLigandInput:
    r"""
    ODesign tensor input with canonical context and provisional generated rows.

    The equation fields deliberately repeat [`ODesignInput`][] because ligand
    output has no canonical generated chemistry. Valid atom rows with
    ``atom_index < 0`` are the ordered provisional slots.
    """

    token_mask: Tensor
    residue_index: Tensor
    chain_index: Tensor
    entity_index: Tensor
    symmetry_index: Tensor
    token_index: Tensor
    cyclic_period: Tensor
    token_features: Tensor
    design_features: Tensor
    token_bonds: Tensor
    atom_features: Tensor
    atom_to_token: Tensor
    atom_mask: Tensor
    reference_positions: Tensor
    reference_space_id: Tensor
    reference_mask: Tensor
    reference_charges: Tensor
    observed_positions: Tensor
    observed_mask: Tensor
    fixed_coordinate_mask: Tensor
    constraint_distogram: Tensor
    msa_tokens: Tensor
    msa_deletion_counts: Tensor
    msa_mask: Tensor
    system: MolecularSystem
    atom_index: Tensor
    token_component_index: Tensor
    token_atom_index: Tensor
    representative_atom_index: Tensor


@tensorclass
class ODesignStructureTarget:
    r"""
    Ground-truth coordinates already aligned to the ODesign Model atom axis.

    Attributes:
        coordinates: Target coordinates with shape ``(B, A, 3)``.
        coordinate_mask: Valid-coordinate mask with shape ``(B, A)``.
        covalent_bond_index: Undirected covalent target edges on the Model atom
            axis, stored once with shape ``(B, E, 2)``.
    """

    coordinates: Tensor
    coordinate_mask: Tensor
    covalent_bond_index: Tensor


class ODesignRawOutput(_TensorCollection):
    r"""
    Predictions from one explicit ODesign denoising call.

    Attributes:
        coordinates: Denoised atom coordinates with shape ``(B, S, A, 3)`` in
            angstroms.
        distogram_logits: Token-pair distance logits with shape ``(B, N, N, D)``.
        token_bond_logits: Binary token-pair bond logits with shape
            ``(B, N, N, 2)``.
    """

    coordinates: Tensor
    distogram_logits: Tensor
    token_bond_logits: Tensor


class ODesignOutput(ODesignRawOutput):
    r"""
    Predictions from stochastic target corruption and denoising.

    Attributes:
        coordinates: Denoised atom coordinates in the Target's original frame,
            with shape ``(B, S, A, 3)``.
        distogram_logits: Token-pair distance logits with shape ``(B, N, N, D)``.
        token_bond_logits: Binary token-pair bond logits with shape
            ``(B, N, N, 2)``.
        noise_level: Sampled EDM noise levels with shape ``(B, S)``.
    """

    noise_level: Tensor


class ODesignSamplingOutput(_TensorCollection):
    r"""
    Predictions from one ODesign sampling call.

    Attributes:
        coordinates: Sampled atom coordinates with shape ``(B, K, S, A, 3)`` in
            angstroms.
        distogram_logits: Token-pair distance logits with shape
            ``(B, K, N, N, D)``.
        token_bond_logits: Binary token-pair bond logits with shape
            ``(B, K, N, N, 2)``.
    """

    coordinates: Tensor
    distogram_logits: Tensor
    token_bond_logits: Tensor
