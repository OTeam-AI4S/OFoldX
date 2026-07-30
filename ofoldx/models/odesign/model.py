r"""
ODesign structure-generation model and tensor interfaces.
"""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass, field
from typing import Generic, Literal, TypeVar

import numpy as np
import torch
import torch.nn.functional as F
from tensordict import tensorclass
from torch import Tensor, nn

from ...data.bonds import SymbolicBond
from ...data.conditions import Hotspot

# ODesign's spatial and placement records drive the hard-condition checks below.
from ...data.molecular import (
    AtomSite,
    MolecularSystem,
)
from ...data.placement import (
    PlacementMap,
    PlacementSpec,
    SymbolicResidueSite,
)
from ...data.placement import resolve as _resolve_site
from ...data.placement import validate_placement
from ...data.structure import Structure
from ...model import Model
from ...modules.atom import (
    LocalAtomGeometry,
    NormalizedKeyAtomToTokenEncoder,
)
from ...modules.backend import LayerNorm
from ...modules.denoiser import CoordinateDenoiser
from ...modules.diffusion import (
    CoordinateDenoiseContext,
    CoordinateEdmSampler,
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


class ODesignModel(Model):
    r"""
    Generate all-atom biomolecular structures with ODesign.

    Accepts [`ODesignInput`][ofoldx.models.odesign.model.ODesignInput] and returns
    [`ODesignOutput`][ofoldx.models.odesign.model.ODesignOutput].

    Args:
        config: Model architecture and inference configuration.

    Examples:
        >>> import torch
        >>> from ofoldx.data import GeneratedSegment
        >>> from ofoldx.models.odesign import (
        ...     ODesignConfig,
        ...     ODesignModel,
        ...     ODesignProcessor,
        ...     ODesignPlacements,
        ... )
        >>> config = ODesignConfig({
        ...     "single_size": 16,
        ...     "pair_size": 8,
        ...     "constraint_embeddings": {
        ...         "hidden_size": 8, "num_blocks": 1,
        ...         "num_distogram_bins": 4, "dropout": 0.0,
        ...     },
        ...     "trunk": {
        ...         "num_msa_blocks": 1, "num_pairformer_blocks": 1,
        ...         "pairformer_dropout": 0.0, "msa_dropout": 0.0, "pair_dropout": 0.0,
        ...     },
        ...     "recycling_steps": 1,
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
        >>> placement = ODesignPlacements(
        ...     {"design": {"binder": GeneratedSegment("protein", 1, 1)}}
        ... )
        >>> inputs = ODesignProcessor()(None, placement=placement)
        >>> atom_count = inputs.atom_features.shape[1]
        >>> model = ODesignModel(config).eval()
        >>> with torch.inference_mode():
        ...     output = model.denoise(
        ...         inputs,
        ...         noisy_coordinates=torch.zeros(1, 1, atom_count, 3),
        ...         noise_level=torch.ones(1, 1),
        ...         recycling_steps=1,
        ...     )
        >>> tuple(output.coordinates.shape) == (1, 1, atom_count, 3)
        True
        >>> output.distogram_logits.shape[-1]
        4
    """

    config_class = ODesignConfig
    config: ODesignConfig

    def __init__(self, config: ODesignConfig | None = None) -> None:
        config = ODesignConfig() if config is None else config
        super().__init__(config)
        self._recycling_steps = config.recycling_steps
        self.input_embeddings = ODesignInputEmbeddings(
            single_size=config.single_size,
            atom_feature_size=config.atom_feature_size,
        )
        self.single_init_projection = nn.Linear(config.token_feature_size, config.single_size, bias=False)
        self.pair_left_projection = nn.Linear(config.single_size, config.pair_size, bias=False)
        self.pair_right_projection = nn.Linear(config.single_size, config.pair_size, bias=False)
        self.relative_position_embedder = RelativePositionEmbedder(
            config.relative_position_size,
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
            token_feature_size=config.token_feature_size,
            num_stages=config.trunk.num_msa_blocks,
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
            token_feature_size=config.token_feature_size,
            atom_feature_size=config.atom_feature_size,
            num_atom_encoder_blocks=config.denoiser.num_atom_encoder_blocks,
            num_atom_encoder_attention_heads=config.denoiser.num_atom_encoder_attention_heads,
            num_atom_decoder_blocks=config.denoiser.num_atom_decoder_blocks,
            num_atom_decoder_attention_heads=config.denoiser.num_atom_decoder_attention_heads,
            num_query_atoms=config.denoiser.num_query_atoms,
            num_key_atoms=config.denoiser.num_key_atoms,
        )
        self.coordinate_sampler = CoordinateEdmSampler(reverse_augmentation=True)
        self.distogram_head = DistogramHead(config.pair_size, config.pairwise.num_distogram_bins)
        self.token_bond_type_head = ODesignTokenBondTypeHead(
            pair_size=config.pair_size,
            hidden_size=config.pairwise.bond_hidden_size,
            num_bond_types=config.pairwise.num_bond_types,
        )
        nn.init.zeros_(self.pair_recycle_projection.weight)
        nn.init.zeros_(self.single_recycle_projection.weight)
        nn.init.zeros_(self.distogram_head.projection.weight)
        if self.distogram_head.projection.bias is not None:
            nn.init.zeros_(self.distogram_head.projection.bias)
        self.post_init()

    def forward(
        self,
        input: ODesignInput,
        target: ODesignStructureTarget,
        *,
        num_samples: int = 1,
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
            target: Structure coordinates and their mapping onto the Model atom axis.
            num_samples: Number ``S`` of independently corrupted copies per input.
            synchronize_noise_levels: Whether samples from one input item share one
                noise level.
            coordinate_augmentation: Whether to rotate and translate the centered
                target independently for each sample.
            augmentation_translation_scale: Standard deviation of augmentation
                translations in angstroms.
            noise_log_mean: Mean of the log-normal noise distribution.
            noise_log_std: Standard deviation of the log-normal noise distribution.
            recycling_steps: Number of trunk recycling passes. ``None`` uses the
                configured value.
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
        steps = self._recycling_steps if recycling_steps is None else recycling_steps
        if steps < 1:
            raise ValueError("recycling_steps must be positive")
        trunk = self._run_trunk(input, recycling_steps=steps)
        target_coordinates, target_mask = _gather_odesign_structure_target(target, input)
        target_coordinates = target_coordinates.to(input.atom_features)
        target_mask = target_mask.to(device=target_coordinates.device)
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
        return ODesignOutput(  # type: ignore[call-arg]
            coordinates=coordinates,
            distogram_logits=raw_output.distogram_logits,
            token_bond_type_logits=raw_output.token_bond_type_logits,
            noise_level=noise_level,
            batch_size=[coordinates.shape[0]],
        )

    def denoise(
        self,
        input: ODesignInput,
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
                configured value.
            use_conditioning: Whether trunk single and pair values condition the
                coordinate denoiser.

        Returns:
            Denoised coordinates and pairwise logits.
        """

        steps = self._recycling_steps if recycling_steps is None else recycling_steps
        if steps < 1:
            raise ValueError("recycling_steps must be positive")
        trunk = self._run_trunk(input, recycling_steps=steps)
        return self._denoise_from_trunk(
            input,
            trunk,
            noisy_coordinates=noisy_coordinates,
            noise_level=noise_level,
            use_conditioning=use_conditioning,
        )

    @torch.inference_mode()
    def design(
        self,
        input: ODesignInput,
        *,
        num_candidates: int = 1,
        generator: torch.Generator | None = None,
        recycling_steps: int | None = None,
        num_sampling_steps: int = 200,
        partial_diffusion_snr: float | None = None,
    ) -> tuple[Structure, ...]:
        r"""
        Generate decoded structure candidates for one processed design problem.

        Args:
            input: Processed design input with batch size one and a realized fixed-length
                layout.
            num_candidates: Number of backbone candidates.
            generator: Random generator used by coordinate sampling.
            recycling_steps: Number of trunk recycling passes. ``None`` uses the
                configured value.
            num_sampling_steps: Number of coordinate-diffusion steps.
            partial_diffusion_snr: Signal-to-noise ratio for partial diffusion from
                ``input.observed_positions``. ``None`` starts from noise.

        Returns:
            Backbone structures in sampling order. Coordinates are in angstroms.
        """

        if isinstance(input, ODesignLigandInput):
            raise TypeError("ODesignModel.design accepts polymer layouts; use design_ligand for provisional slots")
        output = self.sample(
            input,
            num_samples=num_candidates,
            recycling_steps=recycling_steps,
            num_sampling_steps=num_sampling_steps,
            num_model_seeds=1,
            partial_diffusion_snr=partial_diffusion_snr,
            generator=generator,
        )
        structures = tuple(
            _structure_from_atom_coordinates(
                output.coordinates[0, 0, index],
                input.layout.system,
                input.layout.atom_sites,
            )
            for index in range(num_candidates)
        )
        for structure in structures:
            _require_hard_conditions(structure, input)
        return structures

    @torch.inference_mode()
    def design_ligand(
        self,
        input: ODesignLigandInput,
        *,
        num_candidates: int = 1,
        generator: torch.Generator | None = None,
        recycling_steps: int | None = None,
        num_sampling_steps: int = 200,
        partial_diffusion_snr: float | None = None,
    ) -> tuple[ODesignLigandBackbone, ...]:
        r"""
        Generate provisional ligand-atom backbone candidates.

        Generated atom slots remain separate from the canonical context system.
        Their predicted coordinates and pairwise bond logits do not establish a
        molecule or chemical connectivity.

        Args:
            input: Processed ligand-backbone input with a realized slot count.
            num_candidates: Number of backbone candidates.
            generator: Random generator used by coordinate sampling.
            recycling_steps: Number of trunk recycling passes. ``None`` uses the
                configured value.
            num_sampling_steps: Number of coordinate-diffusion steps.
            partial_diffusion_snr: Signal-to-noise ratio for partial diffusion from
                ``input.observed_positions``. ``None`` starts from noise.

        Returns:
            Provisional ligand backbones in sampling order.
        """

        output = self.sample(
            input,
            num_samples=num_candidates,
            recycling_steps=recycling_steps,
            num_sampling_steps=num_sampling_steps,
            num_model_seeds=1,
            partial_diffusion_snr=partial_diffusion_snr,
            generator=generator,
        )
        candidates = tuple(
            _ligand_backbone_from_coordinates(
                output.coordinates[0, 0, index],
                output.token_bond_type_logits[0, 0],
                input.layout,
            )
            for index in range(num_candidates)
        )
        for candidate in candidates:
            _require_ligand_hard_conditions(candidate, input)
        return candidates

    @torch.inference_mode()
    def sample(
        self,
        input: ODesignInput,
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
                configured value.
            num_sampling_steps: Number of coordinate-diffusion steps.
            num_model_seeds: Number of independent trunk evaluations.
            partial_diffusion_snr: Signal-to-noise ratio for partial diffusion from
                ``input.observed_positions``. ``None`` starts from noise.
            generator: Random generator used by coordinate sampling.

        Returns:
            An [`ODesignSamplingOutput`][] with ``coordinates`` of shape
            ``(B, K, S, A, 3)`` in angstroms, ``distogram_logits`` of shape
            ``(B, K, N, N, D)``, and ``token_bond_type_logits`` of shape
            ``(B, K, N, N, D_b)``.
        """

        steps = self._recycling_steps if recycling_steps is None else recycling_steps
        if steps < 1:
            raise ValueError("recycling_steps must be positive")
        partial_diffusion = partial_diffusion_snr is not None
        if partial_diffusion and not bool(input.observed_mask.all()):
            raise ValueError("partial diffusion requires observed coordinates for every model atom")
        max_noise_level = 1 / math.sqrt(partial_diffusion_snr + 1e-6) if partial_diffusion_snr is not None else None
        coordinates = []
        distograms = []
        bond_logits = []
        for _ in range(num_model_seeds):
            trunk = self._run_trunk(input, recycling_steps=steps)
            distograms.append(self.distogram_head(trunk.pair))
            bond_logits.append(self.token_bond_type_head(trunk.pair))
            coordinates.append(
                self.coordinate_sampler(
                    self.coordinate_denoiser,
                    context=self._denoise_context(input, trunk),
                    num_samples=num_samples,
                    num_sampling_steps=num_sampling_steps,
                    initial_coords=input.observed_positions if partial_diffusion else None,
                    condition_coords=input.observed_positions,
                    condition_mask=input.fixed_coordinate_mask,
                    max_noise_level=max_noise_level,
                    generator=generator,
                )
            )
        return ODesignSamplingOutput(  # type: ignore[call-arg]
            coordinates=torch.stack(coordinates, dim=1),
            distogram_logits=torch.stack(distograms, dim=1),
            token_bond_type_logits=torch.stack(bond_logits, dim=1),
            batch_size=[input.atom_features.shape[0]],
        )

    def _denoise_from_trunk(
        self,
        input: ODesignInput,
        trunk: _ODesignTrunkOutput,
        *,
        noisy_coordinates: Tensor,
        noise_level: Tensor,
        use_conditioning: bool,
    ) -> ODesignRawOutput:
        context = self._denoise_context(
            input,
            trunk,
            use_conditioning=use_conditioning,
        )
        coordinates = self.coordinate_denoiser(
            context.to_denoising_input(
                noisy_coords=noisy_coordinates,
                noise_level=noise_level,
            )
        )
        return ODesignRawOutput(  # type: ignore[call-arg]
            coordinates=coordinates,
            distogram_logits=self.distogram_head(trunk.pair),
            token_bond_type_logits=self.token_bond_type_head(trunk.pair),
            batch_size=[coordinates.shape[0]],
        )

    def _run_trunk(
        self,
        input: ODesignInput,
        *,
        recycling_steps: int,
    ) -> _ODesignTrunkOutput:
        local_geometry = self.input_embeddings.build_local_geometry(
            input.reference_positions,
            input.reference_space_id,
        )
        token_embedding = self.input_embeddings(
            input.atom_features,
            reference_positions=input.reference_positions,
            reference_charges=input.reference_charges,
            reference_mask=input.reference_mask,
            token_features=input.token_features,
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
            dtype=single_init.dtype,
        )
        pair_mask = pair_mask_from_token_mask(input.token_mask)
        pair_init = self._initial_pair(single_init, input, relative_features)
        single = torch.zeros_like(single_init)
        pair = torch.zeros_like(pair_init)
        grad_enabled = torch.is_grad_enabled()
        for cycle in range(recycling_steps):
            with torch.set_grad_enabled(grad_enabled and (not self.training or cycle == recycling_steps - 1)):
                pair = pair_init + self.pair_recycle_projection(self.pair_recycle_norm(pair))
                pair = pair + self.constraint_embeddings(
                    pair,
                    input.constraint_distogram,
                    pair_mask=pair_mask,
                )
                _, pair = self.msa_trunk(
                    input.msa_features,
                    pair,
                    token_embedding,
                    msa_mask=input.msa_mask,
                    pair_mask=pair_mask,
                )
                single = single_init + self.single_recycle_projection(self.single_recycle_norm(single))
                single, pair = self.pairformer_stack(single, pair, pair_mask=pair_mask)
        return _ODesignTrunkOutput(  # type: ignore[call-arg]
            token_embedding=token_embedding,
            single=single,
            pair=pair,
            relative_features=relative_features,
            batch_size=[single.shape[0]],
        )

    def _initial_pair(
        self,
        single: Tensor,
        input: ODesignInput,
        relative_features: RelativePositionBuckets,
    ) -> Tensor:
        pair = self.pair_left_projection(single).unsqueeze(-2)
        pair = pair + self.pair_right_projection(single).unsqueeze(-3)
        pair = pair + self.relative_position_embedder(relative_features)
        pair = pair + self.token_bond_projection(input.token_bonds.unsqueeze(-1))
        return pair * pair_mask_from_token_mask(input.token_mask).unsqueeze(-1).to(pair.dtype)

    def _denoise_context(
        self,
        input: ODesignInput,
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
            ),
        )


class ODesignInputEmbeddings(nn.Module):
    r"""
    Aggregate reference atoms and append ODesign token features.
    """

    def __init__(
        self,
        single_size: int = 384,
        atom_feature_size: int = 386,
    ) -> None:
        super().__init__()
        self.single_size = single_size
        self.atom_encoder = NormalizedKeyAtomToTokenEncoder(
            token_hidden_size=single_size,
            atom_feature_size=atom_feature_size,
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
            [encoded, token_features[..., self.single_size :].to(encoded)],
            dim=-1,
        )
        return output * token_mask.unsqueeze(-1).to(output.dtype)

    def build_local_geometry(self, reference_positions: Tensor, reference_space_id: Tensor) -> LocalAtomGeometry:
        return self.atom_encoder.build_local_geometry(reference_positions, reference_space_id)


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


class ODesignTokenBondTypeHead(nn.Module):
    r"""
    Predict whether each ODesign token pair is covalently connected.
    """

    def __init__(self, *, pair_size: int = 128, hidden_size: int = 128, num_bond_types: int = 2) -> None:
        super().__init__()
        self.input_projection = nn.Linear(pair_size, hidden_size)
        self.output_projection = nn.Linear(hidden_size, num_bond_types)

    def forward(self, pair: Tensor) -> Tensor:
        return self.output_projection(self.input_projection(pair))


def _gather_odesign_structure_target(
    target: ODesignStructureTarget,
    input: ODesignInput,
) -> tuple[Tensor, Tensor]:
    atom_index = target.atom_index.long()
    valid = atom_index >= 0
    gather_index = atom_index.clamp_min(0)
    coordinates = target.coordinates.gather(
        1,
        gather_index.unsqueeze(-1).expand(-1, -1, 3),
    )
    coordinate_mask = target.coordinate_mask.gather(1, gather_index)
    coordinate_mask = coordinate_mask & valid & input.atom_mask.bool()
    coordinates = coordinates.masked_fill(~coordinate_mask.unsqueeze(-1), 0)
    return coordinates, coordinate_mask


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


def _structure_from_atom_coordinates(
    coordinates: Tensor,
    system: MolecularSystem,
    atom_sites: tuple[AtomSite, ...],
) -> Structure:
    values = coordinates.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    sites = tuple(site for site, atom in system.iter_atoms() if atom.element != "H")
    site_index = {site: index for index, site in enumerate(sites)}
    complete = np.zeros((len(sites), 3), dtype=np.float32)
    coordinate_mask = np.zeros(len(sites), dtype=np.bool_)
    for predicted_index, site in enumerate(atom_sites):
        output_index = site_index[site]
        complete[output_index] = values[predicted_index]
        coordinate_mask[output_index] = True
    return Structure(
        system=system,
        sites=sites,
        coordinates=complete,
        coordinate_mask=coordinate_mask,
    )


def _ligand_backbone_from_coordinates(
    coordinates: Tensor,
    token_bond_type_logits: Tensor,
    layout: ODesignLigandLayout,
) -> ODesignLigandBackbone:
    context_atom_count = len(layout.context_atom_sites)
    context_system = layout.context_system
    context = (
        _structure_from_atom_coordinates(
            coordinates[:context_atom_count],
            context_system,
            layout.context_atom_sites,
        )
        if context_system is not None and layout.context_atom_sites
        else None
    )
    slot_coordinates = (
        coordinates[context_atom_count:].detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    )
    bond_logits = token_bond_type_logits.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    return ODesignLigandBackbone(
        context=context,
        context_token_sites=layout.context_token_sites,
        slots=layout.slots,
        coordinates=slot_coordinates,
        token_bond_type_logits=bond_logits,
    )


def _require_hard_conditions(structure: Structure, input: ODesignInput) -> None:
    r"""
    Raise when a candidate violates fixed-coordinate evidence.
    """

    source_residues = input.layout.source_residue_map
    placement_map = input.layout.placement_map
    instance_of = input.layout.instance_of
    source_atoms = input.layout.source_atom_map
    coordinates = {site: structure.coordinates[index] for index, site in enumerate(structure.sites)}
    reference = {
        site: input.observed_positions[0, index].detach().to(device="cpu", dtype=torch.float32).numpy()
        for index, site in enumerate(input.layout.atom_sites)
    }
    for index, condition in enumerate(input.layout.conditions):
        if not isinstance(condition, ODesignReferenceCoordinates) or condition.intent != "hard":
            continue
        sites = tuple(
            _resolve_atom(site, source_residues, placement_map, instance_of, source_atoms) for site in condition.sites
        )
        predicted = np.stack([coordinates[site] for site in sites])
        target = np.stack([reference[site] for site in sites])
        if _maximum_aligned_distance(predicted, target) > condition.tolerance_angstrom:
            raise RuntimeError(f"ODesign candidate violated hard conditions[{index}]")


def _require_ligand_hard_conditions(
    candidate: ODesignLigandBackbone,
    input: ODesignLigandInput,
) -> None:
    if not any(
        isinstance(condition, ODesignReferenceCoordinates) and condition.intent == "hard"
        for condition in input.layout.conditions
    ):
        return

    context = candidate.context
    if context is None:
        raise RuntimeError("ODesign ligand candidate omitted conditioned context coordinates")
    coordinates = {site: context.coordinates[index] for index, site in enumerate(context.sites)}
    reference = {
        site: input.observed_positions[0, index].detach().to(device="cpu", dtype=torch.float32).numpy()
        for index, site in enumerate(input.layout.context_atom_sites)
    }
    for index, condition in enumerate(input.layout.conditions):
        if not isinstance(condition, ODesignReferenceCoordinates) or condition.intent != "hard":
            continue
        sites = tuple(input.layout.source_atom_map[site] for site in condition.sites)
        predicted = np.stack([coordinates[site] for site in sites])
        target = np.stack([reference[site] for site in sites])
        if _maximum_aligned_distance(predicted, target) > condition.tolerance_angstrom:
            raise RuntimeError(f"ODesign ligand candidate violated hard conditions[{index}]")


def _maximum_aligned_distance(predicted: np.ndarray, target: np.ndarray) -> float:
    predicted_centered = predicted - predicted.mean(axis=0)
    target_centered = target - target.mean(axis=0)
    left, _, right = np.linalg.svd(predicted_centered.T @ target_centered)
    if np.linalg.det(left @ right) < 0:
        left[:, -1] *= -1
    aligned = predicted_centered @ (left @ right) + target.mean(axis=0)
    return float(np.linalg.norm(aligned - target, axis=-1).max())


def _resolve_residue(
    reference: tuple[str, int] | SymbolicResidueSite,
    source_residues,
    placement_map,
    instance_of,
) -> tuple[str, int]:
    if isinstance(reference, tuple):
        return source_residues[reference]
    return _resolve_site(reference, placement_map, instance_of)


def _resolve_atom(reference, source_residues, placement_map, instance_of, source_atoms) -> AtomSite:
    if isinstance(reference, AtomSite):
        return source_atoms[reference]
    instance_id, monomer_index = _resolve_residue(
        reference.residue,
        source_residues,
        placement_map,
        instance_of,
    )
    return AtomSite(instance_id, monomer_index, reference.atom_id)


# ---------------------------------------------------------------------------
# Typed tensor and semantic records for ODesign.
# ---------------------------------------------------------------------------


@tensorclass
class _ODesignTrunkOutput:
    r"""
    Named representations produced by the ODesign trunk.
    """

    token_embedding: Tensor
    single: Tensor
    pair: Tensor
    relative_features: RelativePositionBuckets


@dataclass(frozen=True)
class ODesignGeneratedAtoms:
    r"""
    Declare one caller-owned range of provisional ligand atom slots.

    Args:
        request_id: Caller identity retained by every realized slot.
        min_atoms: Minimum number of generated atom slots.
        max_atoms: Maximum number of generated atom slots.
    """

    request_id: str
    min_atoms: int
    max_atoms: int

    def __post_init__(self) -> None:
        if self.min_atoms < 0 or self.max_atoms < self.min_atoms:
            raise ValueError("generated atom range must satisfy 0 <= min_atoms <= max_atoms")


@dataclass(frozen=True)
class ODesignProvisionalAtomSlot:
    r"""
    Stable identity for one generated ligand atom before chemical realization.

    Args:
        request_id: Identity of the caller's generated-atom request.
        slot_index: Zero-based position within that request.
    """

    request_id: str
    slot_index: int


@dataclass(frozen=True)
class ODesignPlacements:
    r"""
    Declare the generated and retained segments of an ODesign output.

    Each declaration group represents one output chain.

    Args:
        placements: Ordered placement declaration. Placement IDs must be globally
            unique.
        chain_lengths: Optional exact output length for each declaration group.
    """

    placements: PlacementSpec
    chain_lengths: tuple[int | None, ...] | None = None

    def __post_init__(self) -> None:
        # A placement_id must be globally unique across the whole PlacementSpec.
        validate_placement(self.placements)


@dataclass(frozen=True)
class ODesignReferenceCoordinates:
    r"""
    Fixed or constrained coordinates for a set of existing input atoms.

    Args:
        sites: Existing atom occurrences carrying the reference coordinates.
        tolerance_angstrom: Accepted coordinate deviation in angstroms.
        intent: ``"hard"`` fixes selected atoms during sampling; ``"soft"`` only
            supplies geometric conditioning.
    """

    sites: tuple[AtomSite, ...]
    tolerance_angstrom: float
    intent: Literal["hard", "soft"] = "hard"


@dataclass(eq=False)
class ODesignLayout:
    r"""
    Molecular identities and conditions aligned to ODesign tensor axes.
    """

    system: MolecularSystem
    atom_sites: tuple[AtomSite, ...]
    placement: ODesignPlacements | None
    conditions: tuple[Hotspot | SymbolicBond | ODesignReferenceCoordinates, ...]
    placement_map: PlacementMap = field(default_factory=dict)
    instance_of: dict[str, str] = field(default_factory=dict)
    source_residue_map: dict[tuple[str, int], tuple[str, int]] = field(default_factory=dict)
    source_atom_map: dict[AtomSite, AtomSite] = field(default_factory=dict)


@dataclass(eq=False)
class ODesignLigandLayout:
    r"""
    Canonical context and provisional identities aligned to ligand tensor axes.

    Context token groups precede generated slots on the token axis. Context atom
    sites precede generated slots on the atom axis.
    """

    context_system: MolecularSystem | None
    context_token_sites: tuple[tuple[AtomSite, ...], ...]
    context_atom_sites: tuple[AtomSite, ...]
    generated_atoms: ODesignGeneratedAtoms
    slots: tuple[ODesignProvisionalAtomSlot, ...]
    conditions: tuple[Hotspot | ODesignReferenceCoordinates, ...]
    source_atom_map: dict[AtomSite, AtomSite] = field(default_factory=dict)


_ODesignLayoutT = TypeVar("_ODesignLayoutT", ODesignLayout, ODesignLigandLayout)


@tensorclass
class ODesignInput(Generic[_ODesignLayoutT]):
    r"""
    Tensor and molecular-layout input for ODesign.

    ``B`` is batch size, ``N`` token count, ``A`` atom count, and ``M`` MSA depth.

    Args:
        token_mask: Valid-token mask with shape ``(B, N)``.
        residue_index: Chain-local residue indices with shape ``(B, N)``.
        chain_index: Molecular-instance indices with shape ``(B, N)``.
        entity_index: Molecular-entity indices with shape ``(B, N)``.
        symmetry_index: Copy indices within each entity with shape ``(B, N)``.
        token_index: Token indices within each residue with shape ``(B, N)``.
        cyclic_period: Optional cyclic-position periods with shape ``(B, N)``.
        token_features: Per-token input features with shape ``(B, N, 453)``.
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
        constraint_distogram: Geometric condition features with shape
            ``(B, N, N, 39)``.
        msa_features: MSA features with shape ``(B, M, N, 34)``.
        msa_mask: MSA validity mask with shape ``(B, M, N)``.
        layout: Molecular identities and conditions aligned to the token and atom axes.
    """

    token_mask: Tensor
    residue_index: Tensor
    chain_index: Tensor
    entity_index: Tensor
    symmetry_index: Tensor
    token_index: Tensor
    cyclic_period: Tensor
    token_features: Tensor
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
    msa_features: Tensor
    msa_mask: Tensor
    layout: _ODesignLayoutT


@tensorclass
class ODesignLigandInput(ODesignInput[ODesignLigandLayout]):
    r"""
    ODesign tensor input with a separate provisional ligand layout.
    """


@tensorclass
class ODesignStructureTarget:
    r"""
    Structure coordinates mapped onto an ODesign Model atom axis.

    Args:
        coordinates: Target coordinate pool with shape ``(B, T, 3)``.
        coordinate_mask: Valid-coordinate mask with shape ``(B, T)``.
        atom_index: Target-pool index for every Model atom with shape ``(B, A)``;
            ``-1`` marks atoms absent from the target.
    """

    coordinates: Tensor
    coordinate_mask: Tensor
    atom_index: Tensor


@tensorclass
class ODesignRawOutput:
    r"""
    Predictions from one explicit ODesign denoising call.

    Args:
        coordinates: Denoised atom coordinates with shape ``(B, S, A, 3)`` in
            angstroms.
        distogram_logits: Token-pair distance logits with shape ``(B, N, N, D)``.
        token_bond_type_logits: Token-pair bond-type logits with shape
            ``(B, N, N, D_b)``.
    """

    coordinates: Tensor
    distogram_logits: Tensor
    token_bond_type_logits: Tensor


@tensorclass
class ODesignOutput(ODesignRawOutput):
    r"""
    Predictions from stochastic target corruption and denoising.

    Args:
        coordinates: Denoised atom coordinates in the Target's original frame,
            with shape ``(B, S, A, 3)``.
        distogram_logits: Token-pair distance logits with shape ``(B, N, N, D)``.
        token_bond_type_logits: Token-pair bond-type logits with shape
            ``(B, N, N, D_b)``.
        noise_level: Sampled EDM noise levels with shape ``(B, S)``.
    """

    noise_level: Tensor


@tensorclass
class ODesignSamplingOutput:
    r"""
    Predictions from one ODesign sampling call.

    Args:
        coordinates: Sampled atom coordinates with shape ``(B, K, S, A, 3)`` in
            angstroms.
        distogram_logits: Token-pair distance logits with shape
            ``(B, K, N, N, D)``.
        token_bond_type_logits: Token-pair bond-type logits with shape
            ``(B, K, N, N, D_b)``.
    """

    coordinates: Tensor
    distogram_logits: Tensor
    token_bond_type_logits: Tensor


@dataclass(eq=False)
class ODesignLigandBackbone:
    r"""
    Decoded ODesign ligand backbone before atom identities and bonds are resolved.

    ``token_bond_type_logits`` is aligned first to ``context_token_sites`` and then
    to the singleton tokens represented by ``slots``. It is evidence about
    connectivity, not a realized bond graph.

    Args:
        context: Predicted coordinates over the canonical context system.
        context_token_sites: Canonical atom groups on the leading token axis.
        slots: Ordered provisional generated-atom identities.
        coordinates: Slot coordinates with shape ``(L, 3)`` in angstroms.
        token_bond_type_logits: Full token-pair bond logits with shape
            ``(N, N, D_b)``.
        bonds: Unresolved generated chemical connectivity.
    """

    context: Structure | None
    context_token_sites: tuple[tuple[AtomSite, ...], ...]
    slots: tuple[ODesignProvisionalAtomSlot, ...]
    coordinates: np.ndarray
    token_bond_type_logits: np.ndarray
    bonds: None = None
