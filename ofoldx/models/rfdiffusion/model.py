r"""
RFdiffusion denoiser, sampler, and public scientific modules.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from operator import index as integer_index
from typing import Literal, cast

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from tensordict import tensorclass
from torch import Tensor, einsum, nn

# RFdiffusion spatial and placement records (RfDiffusionPlacements,
# RfDiffusionCoordination, RfDiffusionDistance, RfDiffusionRadiusOfGyration,
# RfDiffusionReferenceCoordinates, RfDiffusionSecondaryStructure,
# RfDiffusionSubstrateInteraction, and RfDiffusionPointGroupSymmetry) are defined below. They build on
# the shared placement records and constraint bond imported from ``ofoldx.data``.
from ...data.bonds import SymbolicBond
from ...data.components import get_component

# AllowedResidueIdentities and Hotspot are shared identity and target-conditioning records.
from ...data.conditions import AllowedResidueIdentities, Hotspot
from ...data.evidence import Template
from ...data.molecular import (
    AtomSite,
    ChemicalComponent,
    ComponentReference,
    MolecularSystem,
    Molecule,
    PolymerSequence,
    SystemBond,
)
from ...data.placement import (
    ExistingSpan,
    GeneratedSegment,
    PlacementMap,
)
from ...data.placement import PlacementSpec as PlacementMapping
from ...data.placement import SymbolicAtomSite, SymbolicResidueSite
from ...data.placement import instance_of as derive_instance_of
from ...data.structure import Structure
from ...model import Model
from ...modules.attention import GatedMultiheadAttention, ProjectedAttention
from ...modules.backend import (
    LayerNorm,
    Se3Basis,
    SegmentSoftmaxCsr,
    SegmentSumCsr,
    WeightedTensorProduct,
)
from ...modules.diffusion.ddpm import ddpm_forward_step, ddpm_posterior
from ...modules.diffusion.igso3 import (
    calculate_igso3,
    so3_exp_map,
    so3_log_map,
    so3_project_to_rotation,
)
from ...modules.dropout import StructuredDropout
from ...modules.geometry import rigid_from_three_points
from ...modules.potential import (
    harmonic_distance_potential,
    interval_distance_penalty,
    smooth_coordination_potential,
    substrate_interaction_potential,
    upper_bounded_radius_of_gyration,
)
from ...modules.symmetry import point_group_rotations
from ...modules.trunk.msa import OuterProductMean
from ...results import Metric, Score, best
from .config import RfDiffusionConfig

# Constraint endpoints use atom-granularity ``AtomSite | SymbolicAtomSite`` values and residue-granularity
# ``ComponentReference | str | SymbolicResidueSite`` values.

# Fixed dimensions of the RFdiffusion IGSO(3) lookup table (num_sigma x num_omega). The
# table cannot be regenerated exactly from compact parameters, so it is stored as persistent float64 buffers.
_IGSO3_NUM_SIGMA = 500
_IGSO3_NUM_OMEGA = 1000
_IGSO3_FIELDS: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("cdf", (_IGSO3_NUM_SIGMA, _IGSO3_NUM_OMEGA)),
    ("score_norm", (_IGSO3_NUM_SIGMA, _IGSO3_NUM_OMEGA)),
    ("exp_score_norms", (_IGSO3_NUM_SIGMA,)),
    ("discrete_omega", (_IGSO3_NUM_OMEGA,)),
    ("discrete_sigma", (_IGSO3_NUM_SIGMA,)),
)


def _index_output_residues(
    references: Sequence[ComponentReference | SymbolicResidueSite],
) -> tuple[
    dict[ComponentReference | SymbolicResidueSite, int],
    dict[str, tuple[int, ...]],
]:
    rows = {reference: row for row, reference in enumerate(references)}
    symbolic: dict[str, list[int]] = {}
    for row, reference in enumerate(references):
        if isinstance(reference, SymbolicResidueSite):
            symbolic.setdefault(reference.placement_id, []).append(row)
    return rows, {placement_id: tuple(indices) for placement_id, indices in symbolic.items()}


def _output_residue_row(
    reference: ComponentReference | SymbolicResidueSite,
    rows: Mapping[ComponentReference | SymbolicResidueSite, int],
    symbolic_rows: Mapping[str, tuple[int, ...]],
) -> int:
    if isinstance(reference, SymbolicResidueSite) and reference.offset < 0:
        return symbolic_rows[reference.placement_id][reference.offset]
    return rows[reference]


class RfDiffusionModel(Model):
    r"""
    Generate protein backbones with RFdiffusion.

    Accepts [`RfDiffusionInput`][ofoldx.models.rfdiffusion.model.RfDiffusionInput] and returns
    [`RfDiffusionOutput`][ofoldx.models.rfdiffusion.model.RfDiffusionOutput] from one denoising pass.

    Args:
        config: Model architecture and inference configuration.

    Examples:
        >>> import torch
        >>> from ofoldx.data import GeneratedSegment
        >>> from ofoldx.models.rfdiffusion import (
        ...     RfDiffusionConfig,
        ...     RfDiffusionModel,
        ...     RfDiffusionPlacements,
        ...     RfDiffusionProcessor,
        ... )
        >>> placement = RfDiffusionPlacements(
        ...     {"chain": {"segment": GeneratedSegment("protein", 1, 1)}}
        ... )
        >>> input = RfDiffusionProcessor()(placement=placement, timestep=1)
        >>> noisy_coordinates = torch.zeros_like(input.noisy_coordinates)
        >>> noisy_coordinates[0, 0, :3] = torch.tensor(
        ...     [[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.5, 0.0]]
        ... )
        >>> input.noisy_coordinates = noisy_coordinates
        >>> se3 = {
        ...     "num_blocks": 1,
        ...     "num_channels": 4,
        ...     "num_degrees": 2,
        ...     "num_attention_heads": 1,
        ...     "div": 1,
        ...     "l0_in_features": 4,
        ...     "l0_out_features": 4,
        ...     "l1_in_features": 3,
        ...     "l1_out_features": 2,
        ...     "num_edge_features": 4,
        ... }
        >>> config = RfDiffusionConfig(
        ...     {
        ...         "denoiser": {
        ...             "num_extra_blocks": 0,
        ...             "num_main_blocks": 1,
        ...             "num_refinement_blocks": 0,
        ...             "msa_size": 8,
        ...             "full_msa_size": 8,
        ...             "pair_size": 8,
        ...             "template_size": 8,
        ...             "num_msa_attention_heads": 1,
        ...             "num_pair_attention_heads": 1,
        ...             "num_template_attention_heads": 1,
        ...             "hidden_size": 4,
        ...             "template_hidden_size": 4,
        ...             "dropout": 0.0,
        ...             "trunk_se3": se3,
        ...             "refinement_se3": se3,
        ...         },
        ...         "diffuser": {"num_timesteps": 1},
        ...     }
        ... )
        >>> model = RfDiffusionModel(config).eval()
        >>> with torch.inference_mode():
        ...     output = model(input)
        >>> output.backbone_coordinates.shape
        torch.Size([1, 1, 3, 3])
        >>> output.sequence_logits.shape
        torch.Size([1, 1, 21])
    """

    config_class = RfDiffusionConfig
    config: RfDiffusionConfig
    _igso3_cache_cdf: Tensor | None

    def __init__(self, config: RfDiffusionConfig | None = None) -> None:
        config = RfDiffusionConfig() if config is None else config
        super().__init__(config)
        denoiser, preprocess = config.denoiser, config.preprocess
        self.denoiser = RfDiffusionDenoiser(
            num_extra_blocks=denoiser.num_extra_blocks,
            num_main_blocks=denoiser.num_main_blocks,
            num_refinement_blocks=denoiser.num_refinement_blocks,
            msa_size=denoiser.msa_size,
            full_msa_size=denoiser.full_msa_size,
            pair_size=denoiser.pair_size,
            template_size=denoiser.template_size,
            num_msa_attention_heads=denoiser.num_msa_attention_heads,
            num_pair_attention_heads=denoiser.num_pair_attention_heads,
            num_template_attention_heads=denoiser.num_template_attention_heads,
            hidden_size=denoiser.hidden_size,
            template_hidden_size=denoiser.template_hidden_size,
            dropout=denoiser.dropout,
            template_single_feature_size=preprocess.template_single_feature_size,
            template_pair_feature_size=preprocess.template_pair_feature_size,
            freeze_track_motif=denoiser.freeze_track_motif,
            trunk_se3=denoiser.trunk_se3,
            refinement_se3=denoiser.refinement_se3,
            vocab_size=denoiser.vocab_size,
        )
        self.num_timesteps = config.diffuser.num_timesteps
        self.template_single_feature_size = preprocess.template_single_feature_size
        self.template_pair_feature_size = preprocess.template_pair_feature_size
        self.sidechain_input = preprocess.sidechain_input
        self.structure_self_conditioning = preprocess.structure_self_conditioning
        self.all_atom_reconstruction = RfDiffusionAllAtomReconstruction()
        # Conversion fills these persistent buffers from the released IGSO(3)
        # lookup table before the model repository is saved.
        for name, shape in _IGSO3_FIELDS:
            self.register_buffer(f"igso3_{name}", torch.zeros(shape, dtype=torch.float64), persistent=True)
            self.register_buffer(f"_igso3_cache_{name}", None, persistent=False)
        self.register_buffer(
            "igso3_schedule_present",
            torch.tensor(False, dtype=torch.bool),
            persistent=True,
        )
        self.post_init()

    def _igso3_schedule(self) -> dict[str, Tensor]:
        r"""
        Return a populated IGSO(3) lookup table.

        Converted Models carry the released persistent table. A fresh Model computes
        its config's table once into non-persistent buffers on the Model's current
        device and dtype.
        """

        if bool(self.igso3_schedule_present):
            return {name: getattr(self, f"igso3_{name}") for name, _ in _IGSO3_FIELDS}
        if self._igso3_cache_cdf is None:
            config = self.config.diffuser
            max_sigma = (
                config.min_rotation_sigma
                + config.min_rotation_beta
                + 0.5 * (config.max_rotation_beta - config.min_rotation_beta)
                if config.rotation_schedule_type == "linear"
                else config.max_rotation_sigma
            )
            schedule = calculate_igso3(
                num_sigma=_IGSO3_NUM_SIGMA,
                min_sigma=config.min_rotation_sigma,
                max_sigma=max_sigma,
                num_omega=_IGSO3_NUM_OMEGA,
                num_terms=2000,
            )
            for name, _shape in _IGSO3_FIELDS:
                release_buffer = getattr(self, f"igso3_{name}")
                setattr(
                    self,
                    f"_igso3_cache_{name}",
                    torch.as_tensor(
                        schedule[name],
                        device=release_buffer.device,
                        dtype=release_buffer.dtype,
                    ),
                )
        return {name: cast(Tensor, getattr(self, f"_igso3_cache_{name}")) for name, _ in _IGSO3_FIELDS}

    def forward(
        self,
        input: RfDiffusionInput,
        *,
        previous_single: Tensor | None = None,
        previous_pair: Tensor | None = None,
        previous_state: Tensor | None = None,
    ) -> RfDiffusionOutput:
        r"""
        Run one differentiable RFdiffusion denoising pass.

        ``B`` is batch size, ``N`` is residue count, and ``T`` is the number of
        structure-update blocks.

        Args:
            input: Processed input containing noisy atom coordinates with shape
                ``(B, N, A, 3)`` in angstroms and one diffusion timestep per batch item.
            previous_single: Recycled single representation with shape
                ``(B, N, C_single)``, or `None` to start from zeros.
            previous_pair: Recycled pair representation with shape
                ``(B, N, N, C_pair)``, or `None` to start from zeros.
            previous_state: Recycled structure-state representation with shape
                ``(B, N, C_state)``, or `None` to start from zeros.

        Returns:
            An [`RfDiffusionOutput`][] containing final backbone coordinates with shape
            ``(B, N, 3, 3)``, residue and pair representations, sequence and confidence
            logits, and structure trajectories with a leading ``T`` axis.
        """

        (
            latent_msa,
            full_msa,
            template_single_features,
            template_pair_features,
            template_coordinates,
            template_torsion_features,
        ) = self._features(input)
        denoiser_output = self.denoiser(
            latent_msa,
            full_msa,
            input.residue_type_features,
            input.noisy_coordinates,
            input.residue_index,
            template_single_features=template_single_features,
            template_pair_features=template_pair_features,
            template_coordinates=template_coordinates,
            template_torsion_features=template_torsion_features,
            previous_msa=previous_single,
            previous_pair=previous_pair,
            previous_state=previous_state,
            fixed_structure_mask=input.fixed_structure_mask,
            cyclic_mask=input.cyclic_mask,
        )
        geometry_logits = denoiser_output.geometry_logits
        return RfDiffusionOutput(
            single=denoiser_output.single,
            pair=denoiser_output.pair,
            backbone_coordinates=denoiser_output.backbone_coordinates,
            state=denoiser_output.state,
            torsion_angles=denoiser_output.torsion_angles,
            sequence_logits=denoiser_output.sequence_logits,
            plddt_fraction=denoiser_output.plddt_fraction,
            distance_logits=geometry_logits.distance_logits.movedim(1, -1),
            omega_logits=geometry_logits.omega_logits.movedim(1, -1),
            theta_logits=geometry_logits.theta_logits.movedim(1, -1),
            phi_logits=geometry_logits.phi_logits.movedim(1, -1),
            plddt_logits=denoiser_output.plddt_logits.movedim(1, -1),
            resolved_log_odds=denoiser_output.resolved_log_odds,
            backbone_trajectory=denoiser_output.backbone_trajectory,
            torsion_trajectory=denoiser_output.torsion_trajectory,
        )

    def _features(
        self,
        input: RfDiffusionInput,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        r"""
        Build the exact MSA/template feature channels.
        """

        residue_type_features = input.residue_type_features
        batch_size, length = residue_type_features.shape[:2]
        latent_msa = residue_type_features.new_zeros((batch_size, 1, length, 48))
        latent_msa[..., :22] = residue_type_features[:, None]
        latent_msa[..., 22:44] = residue_type_features[:, None]
        latent_msa[:, :, 0, 46] = 1
        latent_msa[:, :, -1, 47] = 1
        full_msa = residue_type_features.new_zeros((batch_size, 1, length, 25))
        full_msa[..., :22] = residue_type_features[:, None]
        full_msa[:, :, 0, 23] = 1
        full_msa[:, :, -1, 24] = 1

        template_single_features = residue_type_features.new_zeros((batch_size, 1, length, 21))
        visible_residue_type_features = residue_type_features[..., :21].clone()
        masked = residue_type_features[..., RFDIFFUSION_MASK_TOKEN_ID] > 0.5
        visible_residue_type_features[..., 20] = visible_residue_type_features[..., 20].masked_fill(masked, 1)
        template_single_features[..., :21] = visible_residue_type_features[:, None]
        time = 1 - input.timestep.to(residue_type_features.dtype) / float(self.num_timesteps)
        time = (
            time[:, None]
            .expand_as(input.fixed_structure_mask)
            .masked_fill(
                input.fixed_structure_mask,
                1,
            )
        )
        template_single_features = torch.cat((template_single_features, time[:, None, :, None]), dim=-1)
        if self.template_single_feature_size >= 24:
            template_single_features = torch.cat(
                (
                    template_single_features,
                    template_single_features.new_zeros((batch_size, 1, length, 1)),
                    input.hotspot_mask[:, None, :, None].to(residue_type_features.dtype),
                ),
                dim=-1,
            )
        if self.template_single_feature_size == 28:
            template_single_features = torch.cat((template_single_features, input.secondary_structure[:, None]), dim=-1)

        coords = input.noisy_coordinates.clone()
        if self.sidechain_input:
            coords[:, :, 3:] = coords[:, :, 3:].masked_fill(masked[:, :, None, None], float("nan"))
        else:
            coords[:, :, 3:] = coords[:, :, 3:].masked_fill(
                (~input.fixed_structure_mask)[:, :, None, None],
                float("nan"),
            )
        template_coordinates = torch.cat(
            (coords[:, None], coords.new_full((batch_size, 1, length, 13, 3), float("nan"))),
            dim=3,
        )
        template_pair_features = _template_pair_features(template_coordinates)
        template_torsion_features = _template_torsion_features(
            template_coordinates,
            template_single_features[..., :21].argmax(dim=-1).reshape(batch_size, length),
        )
        if self.structure_self_conditioning:
            template_pair_features[..., :44] = 0
            template_coordinates = torch.zeros_like(template_coordinates)
        if self.template_pair_feature_size == 47:
            template_pair_features = torch.cat((template_pair_features, input.block_adjacency[:, None]), dim=-1)
        return (
            latent_msa,
            full_msa,
            template_single_features,
            template_pair_features,
            template_coordinates,
            template_torsion_features,
        )

    @torch.inference_mode()
    def design(
        self,
        input: RfDiffusionInput,
        *,
        num_candidates: int = 1,
        num_sampling_steps: int | None = None,
        final_step: int = 1,
        partial_steps: int | None = None,
        noise_scale_ca: float = 1.0,
        final_noise_scale_ca: float | None = None,
        ca_noise_schedule: Literal["constant", "linear"] = "constant",
        noise_scale_frame: float = 1.0,
        final_noise_scale_frame: float | None = None,
        frame_noise_schedule: Literal["constant", "linear"] = "constant",
        align_motif: bool = True,
        return_trajectory: bool = False,
        symmetry_recenter: bool = True,
        symmetry_radius_angstrom: float = 10.0,
        guidance_scale: float = 10.0,
        guidance_decay: str = "constant",
        generator: torch.Generator | None = None,
    ) -> RfDiffusionDesignResult:
        r"""
        Generate protein-backbone structures from one processed design request.

        Args:
            input: Processed design input with batch size one. Variable generated-segment
                lengths are sampled independently for each candidate.
            num_candidates: Number of independently sampled backbone candidates.
            num_sampling_steps: Number of active reverse-diffusion timesteps. ``None``
                uses the configured value.
            final_step: Last reverse-diffusion timestep to execute, inclusive.
            partial_steps: Starting timestep for partial diffusion from observed
                coordinates. ``None`` starts from the active schedule maximum.
            noise_scale_ca: Initial multiplier for C-alpha translation noise.
            final_noise_scale_ca: Final translation-noise multiplier for a linear
                schedule. ``None`` uses ``noise_scale_ca``.
            ca_noise_schedule: Translation-noise schedule, ``"constant"`` or ``"linear"``.
            noise_scale_frame: Initial multiplier for backbone-frame rotation noise.
            final_noise_scale_frame: Final frame-noise multiplier for a linear schedule.
                ``None`` uses ``noise_scale_frame``.
            frame_noise_schedule: Frame-noise schedule, ``"constant"`` or ``"linear"``.
            align_motif: Whether to align each sampled structure to fixed motif residues.
            return_trajectory: Whether each candidate includes its reverse-diffusion
                trajectory.
            symmetry_recenter: Whether to recenter symmetry units before radial placement.
            symmetry_radius_angstrom: Radial placement distance in angstroms.
            guidance_scale: Multiplier for differentiable geometric-condition guidance.
            guidance_decay: Guidance schedule: ``"constant"``, ``"linear"``,
                ``"quadratic"``, or ``"cubic"``.
            generator: Random generator used for length realization and diffusion noise.

        Returns:
            An [`RfDiffusionDesignResult`][] containing structures in sampling order.
            Coordinates are in angstroms; each candidate's pLDDT metric uses a 0--100
            scale.
        """

        if num_candidates <= 0:
            raise ValueError("num_candidates must be a positive integer")
        if input.residue_type_features.shape[0] != 1:
            raise ValueError("RFdiffusion design accepts one canonical system at a time")
        prepared = [_sample_variable_lengths(input, generator) for _ in range(num_candidates)]
        sampler = RfDiffusionSampler(
            self.config,
            denoiser=self.denoiser,
            all_atom_reconstruction=self.all_atom_reconstruction,
            igso3_schedule=self._igso3_schedule(),
            final_step=final_step,
            partial_steps=partial_steps,
            noise_scale_ca=noise_scale_ca,
            final_noise_scale_ca=final_noise_scale_ca,
            ca_noise_schedule=ca_noise_schedule,
            noise_scale_frame=noise_scale_frame,
            final_noise_scale_frame=final_noise_scale_frame,
            frame_noise_schedule=frame_noise_schedule,
            align_motif=align_motif,
            symmetry_recenter=symmetry_recenter,
            symmetry_radius_angstrom=symmetry_radius_angstrom,
            guidance_scale=guidance_scale,
            guidance_decay=guidance_decay,
        )
        return _decode_design(
            (
                sampler.sample(
                    candidate_input,
                    num_sampling_steps=num_sampling_steps,
                    generator=generator,
                    return_trajectory=return_trajectory,
                )
                for candidate_input in prepared
            ),
            prepared,
            align_motif=align_motif,
            return_trajectory=return_trajectory,
        )


class RfDiffusionDenoiser(nn.Module):
    r"""
    RoseTTAFold trunk driving one RFdiffusion denoising step.
    """

    def __init__(
        self,
        num_extra_blocks: int,
        num_main_blocks: int,
        num_refinement_blocks: int,
        msa_size: int,
        full_msa_size: int,
        pair_size: int,
        template_size: int,
        num_msa_attention_heads: int,
        num_pair_attention_heads: int,
        num_template_attention_heads: int,
        hidden_size: int,
        template_hidden_size: int,
        dropout: float,
        template_single_feature_size: int,
        template_pair_feature_size: int,
        freeze_track_motif: bool,
        trunk_se3: dict | None = None,
        refinement_se3: dict | None = None,
        vocab_size: int = 21,
    ) -> None:
        super().__init__()
        trunk_se3_config = _se3_config(trunk_se3)
        refinement_se3_config = _se3_config(refinement_se3)
        state_size = refinement_se3_config["l0_out_features"]

        self.freeze_track_motif = freeze_track_motif
        self.msa_embeddings = RfDiffusionMsaEmbeddings(
            msa_size=msa_size,
            pair_size=pair_size,
            state_size=state_size,
            dropout=dropout,
        )
        self.extra_msa_embeddings = RfDiffusionExtraMsaEmbeddings(
            msa_size=full_msa_size,
            input_feature_size=25,
            dropout=dropout,
        )
        self.recycling_embeddings = RfDiffusionRecyclingEmbeddings(
            msa_size=msa_size,
            pair_size=pair_size,
            state_size=state_size,
        )
        self.template_embedder = RfDiffusionTemplateEmbedder(
            pair_size=pair_size,
            template_size=template_size,
            state_size=state_size,
            num_attention_heads=num_template_attention_heads,
            hidden_size=template_hidden_size,
            dropout=0.25,
            template_single_feature_size=template_single_feature_size,
            template_pair_feature_size=template_pair_feature_size,
        )
        self.trunk = RfDiffusionTrunk(
            num_extra_blocks=num_extra_blocks,
            num_main_blocks=num_main_blocks,
            num_refinement_blocks=num_refinement_blocks,
            msa_size=msa_size,
            full_msa_size=full_msa_size,
            pair_size=pair_size,
            hidden_size=hidden_size,
            num_msa_attention_heads=num_msa_attention_heads,
            num_pair_attention_heads=num_pair_attention_heads,
            trunk_se3=trunk_se3_config,
            refinement_se3=refinement_se3_config,
            dropout=dropout,
        )
        self.pairwise_geometry_head = RfDiffusionPairwiseGeometryHead(pair_size)
        self.masked_token_head = RfDiffusionMaskedTokenHead(msa_size, vocab_size=vocab_size)
        self.predicted_lddt_head = RfDiffusionPredictedLddtHead(state_size)
        self.experimentally_resolved_head = RfDiffusionExperimentallyResolvedHead(msa_size, state_size)

    def forward(
        self,
        latent_msa: Tensor,
        full_msa: Tensor,
        residue_type_features: Tensor,
        coords: Tensor,
        residue_index: Tensor,
        template_single_features: Tensor | None = None,
        template_pair_features: Tensor | None = None,
        template_coordinates: Tensor | None = None,
        template_torsion_features: Tensor | None = None,
        previous_msa: Tensor | None = None,
        previous_pair: Tensor | None = None,
        previous_state: Tensor | None = None,
        fixed_structure_mask: Tensor | None = None,
        cyclic_mask: Tensor | None = None,
    ) -> _RfDiffusionDenoiserOutput:
        batch_size, _, length = latent_msa.shape[:3]
        latent_msa, pair, state = self.msa_embeddings(
            latent_msa,
            residue_type_features,
            residue_index,
            cyclic_mask,
        )
        full_msa = self.extra_msa_embeddings(full_msa, residue_type_features)

        if previous_msa is None:
            previous_msa = torch.zeros_like(latent_msa[:, 0])
            previous_pair = torch.zeros_like(pair)
            previous_state = torch.zeros_like(state)
        msa_recycle, pair_recycle, state_recycle = self.recycling_embeddings(
            previous_msa,
            cast(Tensor, previous_pair),
            coords,
            cast(Tensor, previous_state),
        )
        latent_msa[:, 0] = latent_msa[:, 0] + msa_recycle.reshape(batch_size, length, -1)
        pair = pair + pair_recycle
        state = state + state_recycle

        # Time features are injected by the sampler's preprocess(); no timestep_embedder here.
        pair, state = self.template_embedder(
            template_single_features,
            template_pair_features,
            template_torsion_features,
            template_coordinates,
            pair,
            state,
        )
        if fixed_structure_mask is None:
            fixed_structure_mask = torch.zeros(
                (batch_size, length),
                dtype=torch.bool,
                device=latent_msa.device,
            )
        elif fixed_structure_mask.ndim == 1:
            fixed_structure_mask = fixed_structure_mask.unsqueeze(0).expand(batch_size, -1)
        frozen_mask = fixed_structure_mask if self.freeze_track_motif else torch.zeros_like(fixed_structure_mask)
        trunk_output = self.trunk(
            residue_type_features,
            latent_msa,
            full_msa,
            pair,
            coords[:, :, :3],
            state,
            residue_index,
            fixed_structure_mask=frozen_mask,
            cyclic_mask=cyclic_mask,
        )

        local = coords[:, :, :3] - coords[:, :, 1].unsqueeze(-2)
        backbone_trajectory = torch.einsum(
            "sblij,blaj->sblai",
            trunk_output.rotation_trajectory,
            local,
        )
        backbone_trajectory = backbone_trajectory + trunk_output.translation_trajectory.unsqueeze(-2)
        final_coords = backbone_trajectory[-1]

        geometry_logits = self.pairwise_geometry_head(trunk_output.pair)
        sequence_logits = self.masked_token_head(trunk_output.msa)
        plddt_logits = self.predicted_lddt_head(trunk_output.state)
        resolved_log_odds = self.experimentally_resolved_head(
            trunk_output.msa[:, 0],
            trunk_output.state,
        )
        return _RfDiffusionDenoiserOutput(
            single=trunk_output.msa[:, 0],
            pair=trunk_output.pair,
            backbone_coordinates=final_coords,
            state=trunk_output.state,
            torsion_angles=trunk_output.torsion_trajectory[-1],
            sequence_logits=sequence_logits.permute(0, 2, 1),
            plddt_fraction=_plddt_expectation(plddt_logits),
            geometry_logits=geometry_logits,
            plddt_logits=plddt_logits,
            resolved_log_odds=resolved_log_odds,
            backbone_trajectory=backbone_trajectory,
            torsion_trajectory=trunk_output.torsion_trajectory,
        )


class RfDiffusionMsaEmbeddings(nn.Module):
    r"""
    Embed the latent MSA, pair, and node-state representations from residue-type features.
    """

    def __init__(
        self,
        msa_size: int = 256,
        pair_size: int = 128,
        state_size: int = 32,
        input_feature_size: int = 22 + 22 + 2 + 2,
        minpos: int = -32,
        maxpos: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_feature_size, msa_size)
        self.query_embedding = nn.Embedding(22, msa_size)
        self.dropout = nn.Dropout(dropout)
        self.left_pair_embedding = nn.Embedding(22, pair_size)
        self.right_pair_embedding = nn.Embedding(22, pair_size)
        self.relative_position_embedding = RfDiffusionRelativePositionEmbedding(
            pair_size, minpos=minpos, maxpos=maxpos, dropout=dropout
        )
        self.state_embedding = nn.Embedding(22, state_size)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.input_projection = _initialize_lecun_normal(self.input_projection)
        self.query_embedding = _initialize_lecun_normal(self.query_embedding)
        self.left_pair_embedding = _initialize_lecun_normal(self.left_pair_embedding)
        self.right_pair_embedding = _initialize_lecun_normal(self.right_pair_embedding)
        self.state_embedding = _initialize_lecun_normal(self.state_embedding)
        _zero_bias(self.input_projection)

    def forward(
        self,
        msa: torch.Tensor,
        residue_type_features: torch.Tensor,
        residue_index: torch.Tensor,
        cyclize: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        msa_depth = msa.shape[1]
        msa = self.input_projection(msa)
        query = _one_hot_lookup(residue_type_features, self.query_embedding).unsqueeze(1)
        msa = self.dropout(msa + query.expand(-1, msa_depth, -1, -1))

        left = _one_hot_lookup(residue_type_features, self.left_pair_embedding).unsqueeze(1)
        right = _one_hot_lookup(residue_type_features, self.right_pair_embedding).unsqueeze(2)
        pair = self.relative_position_embedding(left + right, residue_index, cyclize)

        state = self.dropout(_one_hot_lookup(residue_type_features, self.state_embedding))
        return msa, pair, state


class RfDiffusionExtraMsaEmbeddings(nn.Module):
    r"""
    Embed the extra (full) MSA stack from its raw features.
    """

    def __init__(
        self,
        msa_size: int = 256,
        input_feature_size: int = 22 + 1 + 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_feature_size, msa_size)
        self.query_embedding = nn.Embedding(22, msa_size)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.input_projection = _initialize_lecun_normal(self.input_projection)
        _zero_bias(self.input_projection)

    def forward(self, msa: torch.Tensor, residue_type_features: torch.Tensor) -> torch.Tensor:
        msa_depth = msa.shape[1]
        msa = self.input_projection(msa)
        query = _one_hot_lookup(residue_type_features, self.query_embedding).unsqueeze(1)
        return self.dropout(msa + query.expand(-1, msa_depth, -1, -1))


class RfDiffusionTemplateEmbedder(nn.Module):
    r"""
    Fold template features into the pair and node-state representations.
    """

    def __init__(
        self,
        template_single_feature_size: int = 21 + 1 + 1,
        template_pair_feature_size: int = 43 + 1,
        torsion_size: int = 30,
        pair_size: int = 128,
        state_size: int = 32,
        num_blocks: int = 2,
        template_size: int = 64,
        num_attention_heads: int = 4,
        hidden_size: int = 16,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.pair_projection = nn.Linear(template_single_feature_size * 2 + template_pair_feature_size, template_size)
        self.template_pair_stack = RfDiffusionTemplatePairStack(
            num_blocks=num_blocks,
            template_size=template_size,
            num_attention_heads=num_attention_heads,
            hidden_size=hidden_size,
            dropout=dropout,
        )
        self.template_pair_norm = LayerNorm(template_size)
        self.single_input_projection = nn.Linear(template_single_feature_size + torsion_size, template_size)
        self.single_output_projection = nn.Linear(template_size, template_size)
        self.state_attention = GatedMultiheadAttention(
            hidden_size=state_size,
            num_attention_heads=num_attention_heads,
            head_dim=hidden_size,
            query_size=state_size,
            key_size=template_size,
            value_size=template_size,
            query_bias=False,
            output_bias=True,
            gating=False,
            backend="eager",
        )
        self.pair_attention = GatedMultiheadAttention(
            hidden_size=pair_size,
            num_attention_heads=num_attention_heads,
            head_dim=hidden_size,
            query_size=pair_size,
            key_size=template_size,
            value_size=template_size,
            query_bias=False,
            output_bias=True,
            gating=False,
            backend="eager",
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.pair_projection = _initialize_lecun_normal(self.pair_projection)
        _zero_bias(self.pair_projection)
        nn.init.kaiming_normal_(self.single_input_projection.weight, nonlinearity="relu")
        _zero_bias(self.single_input_projection)
        self.single_output_projection = _initialize_lecun_normal(self.single_output_projection)
        _zero_bias(self.single_output_projection)
        for attention in (self.state_attention, self.pair_attention):
            projection = attention.projection
            nn.init.xavier_uniform_(projection.query_projection.weight)
            nn.init.xavier_uniform_(projection.key_projection.weight)
            nn.init.xavier_uniform_(projection.value_projection.weight)
            nn.init.zeros_(projection.output_projection.weight)
            nn.init.zeros_(projection.output_projection.bias)

    def forward(
        self,
        template_single_features: torch.Tensor,
        template_pair_features: torch.Tensor,
        template_torsion_features: torch.Tensor,
        template_coordinates: torch.Tensor,
        pair: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, template_count, length, _ = template_single_features.shape

        left = template_single_features.unsqueeze(3).expand(-1, -1, -1, length, -1)
        right = template_single_features.unsqueeze(2).expand(-1, -1, length, -1, -1)
        template_pair = self.pair_projection(torch.cat((template_pair_features, left, right), dim=-1))

        template_coordinates = template_coordinates.reshape(batch_size * template_count, length, -1, 3)
        radial_features = _radial_basis(torch.cdist(template_coordinates[:, :, 1], template_coordinates[:, :, 1]))
        template_pair = self.template_pair_norm(self.template_pair_stack(template_pair, radial_features))

        template_single = torch.cat((template_single_features, template_torsion_features), dim=-1)
        template_single = self.single_output_projection(F.relu(self.single_input_projection(template_single)))

        state_query = state.reshape(batch_size * length, 1, -1)
        template_single = template_single.permute(0, 2, 1, 3).reshape(batch_size * length, template_count, -1)
        state_update = self.state_attention(state_query, template_single, template_single)
        state = state + state_update.reshape(batch_size, length, -1)

        pair_query = pair.reshape(batch_size * length * length, 1, -1)
        template_pair = template_pair.permute(0, 2, 3, 1, 4).reshape(batch_size * length * length, template_count, -1)
        pair_update = self.pair_attention(pair_query, template_pair, template_pair)
        pair = pair + pair_update.reshape(batch_size, length, length, -1)

        return pair, state


class RfDiffusionTemplatePairStack(nn.Module):
    r"""
    Stack of pair-update blocks that refines the template pair representation.
    """

    def __init__(
        self,
        num_blocks: int = 2,
        template_size: int = 64,
        num_attention_heads: int = 4,
        hidden_size: int = 16,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.blocks = nn.ModuleList(
            RfDiffusionPairBlock(
                pair_size=template_size,
                num_attention_heads=num_attention_heads,
                hidden_size=hidden_size,
                dropout=dropout,
            )
            for _ in range(num_blocks)
        )

    def forward(self, template_pair: torch.Tensor, radial_features: torch.Tensor) -> torch.Tensor:
        batch_size, template_count, length = template_pair.shape[:3]
        template_pair = template_pair.reshape(batch_size * template_count, length, length, -1)
        for block in self.blocks:
            template_pair = block(template_pair, radial_features)
        return template_pair.reshape(batch_size, template_count, length, length, -1)


class RfDiffusionRecyclingEmbeddings(nn.Module):
    r"""
    Fold the previous iteration's MSA, pair, state, and structure back into the trunk inputs.
    """

    def __init__(self, msa_size: int = 256, pair_size: int = 128, state_size: int = 32) -> None:
        super().__init__()
        self.state_norm = LayerNorm(state_size)
        self.pair_projection = nn.Linear(36 + state_size * 2, pair_size)
        self.msa_norm = LayerNorm(msa_size)
        self.pair_norm = LayerNorm(pair_size)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.pair_projection = _initialize_lecun_normal(self.pair_projection)
        _zero_bias(self.pair_projection)

    def forward(
        self,
        msa: torch.Tensor,
        pair: torch.Tensor,
        coords: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, length = pair.shape[:2]
        state = self.state_norm(state)
        left_state = state.unsqueeze(2).expand(-1, -1, length, -1)
        right_state = state.unsqueeze(1).expand(-1, length, -1, -1)

        cbeta = _beta_carbon(coords[:, :, 0], coords[:, :, 1], coords[:, :, 2])
        dist = _radial_basis(torch.cdist(cbeta, cbeta))
        dist = self.pair_projection(torch.cat((dist, left_state, right_state), dim=-1))

        msa = self.msa_norm(msa)
        pair = self.pair_norm(pair) + dist
        return msa, pair, state


class RfDiffusionTrunk(nn.Module):
    r"""
    The RoseTTAFold track: stacked iteration blocks plus a top-k structure refiner.
    """

    def __init__(
        self,
        num_extra_blocks: int = 4,
        num_main_blocks: int = 12,
        num_refinement_blocks: int = 4,
        msa_size: int = 256,
        full_msa_size: int = 64,
        pair_size: int = 128,
        hidden_size: int = 32,
        num_msa_attention_heads: int = 8,
        num_pair_attention_heads: int = 4,
        trunk_se3: Mapping[str, int] | None = None,
        refinement_se3: Mapping[str, int] | None = None,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        trunk_se3 = _se3_config(trunk_se3)
        refinement_se3 = _se3_config(refinement_se3)

        self.num_extra_blocks = num_extra_blocks
        self.num_main_blocks = num_main_blocks
        self.num_refinement_blocks = num_refinement_blocks
        self.state_input_projection = nn.Linear(refinement_se3["l0_out_features"], trunk_se3["l0_out_features"])

        self.extra_blocks = nn.ModuleList(
            [
                RfDiffusionExtraTrunkBlock(
                    msa_size=full_msa_size,
                    pair_size=pair_size,
                    num_msa_attention_heads=num_msa_attention_heads,
                    num_pair_attention_heads=num_pair_attention_heads,
                    msa_hidden_size=8,
                    hidden_size=hidden_size,
                    dropout=dropout,
                    se3_config=trunk_se3,
                )
                for _ in range(num_extra_blocks)
            ]
        )

        self.main_blocks = nn.ModuleList(
            [
                RfDiffusionTrunkBlock(
                    msa_size=msa_size,
                    pair_size=pair_size,
                    num_msa_attention_heads=num_msa_attention_heads,
                    num_pair_attention_heads=num_pair_attention_heads,
                    hidden_size=hidden_size,
                    dropout=dropout,
                    se3_config=trunk_se3,
                )
                for _ in range(num_main_blocks)
            ]
        )

        self.state_output_projection = nn.Linear(trunk_se3["l0_out_features"], refinement_se3["l0_out_features"])
        if num_refinement_blocks > 0:
            self.refinement_update = RfDiffusionStructureUpdate(
                msa_size=msa_size,
                pair_size=pair_size,
                state_size=refinement_se3["l0_out_features"],
                se3_config=refinement_se3,
            )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.state_input_projection = _initialize_lecun_normal(self.state_input_projection)
        nn.init.zeros_(self.state_input_projection.bias)
        self.state_output_projection = _initialize_lecun_normal(self.state_output_projection)
        nn.init.zeros_(self.state_output_projection.bias)

    def forward(
        self,
        residue_type_features: Tensor,
        msa: Tensor,
        full_msa: Tensor,
        pair: Tensor,
        coords_in: Tensor,
        state: Tensor,
        residue_index: Tensor,
        cyclic_mask: Tensor | None = None,
        fixed_structure_mask: Tensor | None = None,
    ) -> _RfDiffusionTrunkOutput:
        fixed_structure_mask = (
            torch.zeros_like(residue_index, dtype=torch.bool) if fixed_structure_mask is None else fixed_structure_mask
        )
        if fixed_structure_mask.ndim == 1:
            fixed_structure_mask = fixed_structure_mask.unsqueeze(0).expand(residue_index.shape[0], -1)
        rotations, translations, local_coords = _initial_frames(coords_in)
        state = self.state_input_projection(state)

        rotation_steps: list[Tensor] = []
        translation_steps: list[Tensor] = []
        torsion_steps: list[Tensor] = []

        for block in self.extra_blocks:
            rotations = rotations.detach()
            translations = translations.detach()
            coords = _apply_frames(rotations, translations, local_coords)
            block_output = block(
                full_msa,
                pair,
                rotations,
                translations,
                coords,
                state,
                residue_index,
                fixed_structure_mask=fixed_structure_mask,
                cyclic_mask=cyclic_mask,
            )
            full_msa = block_output.msa
            pair = block_output.pair
            rotations = block_output.rotations
            translations = block_output.translations
            state = block_output.state
            rotation_steps.append(rotations)
            translation_steps.append(translations)
            torsion_steps.append(block_output.torsion_angles)

        for block in self.main_blocks:
            rotations = rotations.detach()
            translations = translations.detach()
            coords = _apply_frames(rotations, translations, local_coords)
            block_output = block(
                msa,
                pair,
                rotations,
                translations,
                coords,
                state,
                residue_index,
                fixed_structure_mask=fixed_structure_mask,
                cyclic_mask=cyclic_mask,
            )
            msa = block_output.msa
            pair = block_output.pair
            rotations = block_output.rotations
            translations = block_output.translations
            state = block_output.state
            rotation_steps.append(rotations)
            translation_steps.append(translations)
            torsion_steps.append(block_output.torsion_angles)

        state = self.state_output_projection(state)
        for _ in range(self.num_refinement_blocks):
            rotations = rotations.detach()
            translations = translations.detach()
            coords = _apply_frames(rotations, translations, local_coords)
            structure_output = self.refinement_update(
                msa,
                pair,
                rotations,
                translations,
                coords,
                state,
                residue_index,
                top_k=64,
                fixed_structure_mask=fixed_structure_mask,
                cyclic_mask=cyclic_mask,
            )
            rotations = structure_output.rotations
            translations = structure_output.translations
            state = structure_output.state
            rotation_steps.append(rotations)
            translation_steps.append(translations)
            torsion_steps.append(structure_output.torsion_angles)

        return _RfDiffusionTrunkOutput(
            msa=msa,
            pair=pair,
            rotation_trajectory=torch.stack(rotation_steps, dim=0),
            translation_trajectory=torch.stack(translation_steps, dim=0),
            torsion_trajectory=torch.stack(torsion_steps, dim=0),
            state=state,
        )


class RfDiffusionExtraTrunkBlock(nn.Module):
    r"""
    One RoseTTAFold iteration block over the MSA, pair, and structure representations.
    """

    def __init__(
        self,
        msa_size: int = 256,
        pair_size: int = 128,
        num_msa_attention_heads: int = 8,
        num_pair_attention_heads: int = 4,
        hidden_size: int = 32,
        msa_hidden_size: int | None = None,
        dropout: float = 0.15,
        se3_config: Mapping[str, int] | None = None,
    ) -> None:
        super().__init__()
        se3_config = _se3_config(se3_config)
        msa_hidden_size = hidden_size if msa_hidden_size is None else msa_hidden_size
        state_size = se3_config["l0_out_features"]

        self.msa_block = RfDiffusionExtraMsaBlock(
            msa_size=msa_size,
            pair_size=pair_size,
            num_attention_heads=num_msa_attention_heads,
            state_size=state_size,
            hidden_size=msa_hidden_size,
            dropout=dropout,
        )
        self.outer_product_mean = RfDiffusionOuterProductMean(
            msa_size=msa_size, pair_size=pair_size, hidden_size=hidden_size // 2
        )
        self.pair_block = RfDiffusionPairBlock(
            pair_size=pair_size, num_attention_heads=num_pair_attention_heads, hidden_size=hidden_size, dropout=dropout
        )
        self.structure_update = RfDiffusionStructureUpdate(
            msa_size=msa_size,
            pair_size=pair_size,
            state_size=state_size,
            se3_config=se3_config,
        )

    def forward(
        self,
        msa: Tensor,
        pair: Tensor,
        input_rotations: Tensor,
        input_translations: Tensor,
        coords: Tensor,
        state: Tensor,
        residue_index: Tensor,
        fixed_structure_mask: Tensor,
        cyclic_mask: Tensor | None = None,
    ) -> _RfDiffusionTrunkBlockOutput:
        radial_features = _backbone_ca_distogram(coords)
        msa = self.msa_block(msa, pair, radial_features, state)
        pair = pair + self.outer_product_mean(msa)
        pair = self.pair_block(pair, radial_features)
        structure_output = self.structure_update(
            msa,
            pair,
            input_rotations,
            input_translations,
            coords,
            state,
            residue_index,
            fixed_structure_mask=fixed_structure_mask,
            cyclic_mask=cyclic_mask,
            top_k=0,
        )
        return _RfDiffusionTrunkBlockOutput(  # type: ignore[call-arg]
            msa=msa,
            pair=pair,
            rotations=structure_output.rotations,
            translations=structure_output.translations,
            state=structure_output.state,
            torsion_angles=structure_output.torsion_angles,
            batch_size=[msa.shape[0]],
        )


class RfDiffusionTrunkBlock(nn.Module):
    r"""
    One main RoseTTAFold iteration block over the MSA, pair, and structure representations.
    """

    def __init__(
        self,
        msa_size: int = 256,
        pair_size: int = 128,
        num_msa_attention_heads: int = 8,
        num_pair_attention_heads: int = 4,
        hidden_size: int = 32,
        msa_hidden_size: int | None = None,
        dropout: float = 0.15,
        se3_config: Mapping[str, int] | None = None,
    ) -> None:
        super().__init__()
        se3_config = _se3_config(se3_config)
        msa_hidden_size = hidden_size if msa_hidden_size is None else msa_hidden_size
        state_size = se3_config["l0_out_features"]

        self.msa_block = RfDiffusionMsaBlock(
            msa_size=msa_size,
            pair_size=pair_size,
            num_attention_heads=num_msa_attention_heads,
            state_size=state_size,
            hidden_size=msa_hidden_size,
            dropout=dropout,
        )
        self.outer_product_mean = RfDiffusionOuterProductMean(
            msa_size=msa_size, pair_size=pair_size, hidden_size=hidden_size // 2
        )
        self.pair_block = RfDiffusionPairBlock(
            pair_size=pair_size, num_attention_heads=num_pair_attention_heads, hidden_size=hidden_size, dropout=dropout
        )
        self.structure_update = RfDiffusionStructureUpdate(
            msa_size=msa_size,
            pair_size=pair_size,
            state_size=state_size,
            se3_config=se3_config,
        )

    def forward(
        self,
        msa: Tensor,
        pair: Tensor,
        input_rotations: Tensor,
        input_translations: Tensor,
        coords: Tensor,
        state: Tensor,
        residue_index: Tensor,
        fixed_structure_mask: Tensor,
        cyclic_mask: Tensor | None = None,
    ) -> _RfDiffusionTrunkBlockOutput:
        radial_features = _backbone_ca_distogram(coords)
        msa = self.msa_block(msa, pair, radial_features, state)
        pair = pair + self.outer_product_mean(msa)
        pair = self.pair_block(pair, radial_features)
        structure_output = self.structure_update(
            msa,
            pair,
            input_rotations,
            input_translations,
            coords,
            state,
            residue_index,
            fixed_structure_mask=fixed_structure_mask,
            cyclic_mask=cyclic_mask,
            top_k=0,
        )
        return _RfDiffusionTrunkBlockOutput(  # type: ignore[call-arg]
            msa=msa,
            pair=pair,
            rotations=structure_output.rotations,
            translations=structure_output.translations,
            state=structure_output.state,
            torsion_angles=structure_output.torsion_angles,
            batch_size=[msa.shape[0]],
        )


class RfDiffusionRelativePositionEmbedding(nn.Module):
    r"""
    Relative-position embedding added to the pair representation.
    """

    def __init__(self, hidden_size: int, minpos: int = -32, maxpos: int = 32, dropout: float = 0.1) -> None:
        super().__init__()
        self.minpos = minpos
        self.maxpos = maxpos
        self.num_bins = abs(minpos) + maxpos + 1
        self.embedding = nn.Embedding(self.num_bins, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, pair: torch.Tensor, residue_index: torch.Tensor, cyclize: torch.Tensor | None = None
    ) -> torch.Tensor:
        boundaries = torch.arange(self.minpos, self.maxpos, device=pair.device)
        seqsep = residue_index[:, None, :] - residue_index[:, :, None]
        cyclic_residues = _cyclic_mask(cyclize, residue_index)

        if cyclic_residues is not None:
            chain_indices = _chain_indices(residue_index)
            same_chain = chain_indices[:, :, None] == chain_indices[:, None, :]
            same_cycle = same_chain & cyclic_residues[:, :, None] & cyclic_residues[:, None, :]
            cycle_length = (same_chain & cyclic_residues[:, None, :]).sum(dim=-1)[:, :, None]
            half = cycle_length // 2
            seqsep = torch.where(same_cycle & (seqsep > half), seqsep - cycle_length, seqsep)
            seqsep = torch.where(same_cycle & (seqsep < -half), seqsep + cycle_length, seqsep)

        relpos = torch.bucketize(seqsep, boundaries).long()
        return self.dropout(pair + self.embedding(relpos))


class RfDiffusionMsaBlock(nn.Module):
    r"""
    Update the MSA representation from the pair, structure, and node-state signals.
    """

    def __init__(
        self,
        msa_size: int = 256,
        pair_size: int = 128,
        num_attention_heads: int = 8,
        state_size: int = 16,
        hidden_size: int = 32,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.pair_norm = LayerNorm(pair_size)
        self.pair_projection = nn.Linear(pair_size + 36, pair_size)
        self.state_norm = LayerNorm(state_size)
        self.state_projection = nn.Linear(state_size, msa_size)
        self.row_attention = RfDiffusionMsaRowAttentionWithPairBias(
            msa_size=msa_size, pair_size=pair_size, num_attention_heads=num_attention_heads, head_dim=hidden_size
        )
        self.row_dropout = StructuredDropout(dropout, broadcast_dims=1)
        self.column_attention = RfDiffusionMsaColumnAttention(
            msa_size=msa_size, num_attention_heads=num_attention_heads, head_dim=hidden_size
        )
        self.transition = RfDiffusionTransition(msa_size, 4, dropout=dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.pair_projection = _initialize_lecun_normal(self.pair_projection)
        self.state_projection = _initialize_lecun_normal(self.state_projection)
        nn.init.zeros_(self.pair_projection.bias)
        nn.init.zeros_(self.state_projection.bias)

    def forward(self, msa: Tensor, pair: Tensor, radial_features: Tensor, state: Tensor) -> Tensor:
        batch_size, _, length = msa.shape[:3]
        pair_bias = torch.cat((self.pair_norm(pair), radial_features), dim=-1)
        pair_bias = self.pair_projection(pair_bias)

        query_update = self.state_projection(self.state_norm(state)).reshape(batch_size, 1, length, -1)
        msa = _query_state_update(msa, query_update)

        msa = msa + self.row_dropout(self.row_attention(msa, pair_bias))
        msa = msa + self.column_attention(msa)
        return msa + self.transition(msa)


class RfDiffusionExtraMsaBlock(nn.Module):
    r"""
    Update an extra-MSA representation with global column attention.
    """

    def __init__(
        self,
        msa_size: int = 256,
        pair_size: int = 128,
        num_attention_heads: int = 8,
        state_size: int = 16,
        hidden_size: int = 32,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.pair_norm = LayerNorm(pair_size)
        self.pair_projection = nn.Linear(pair_size + 36, pair_size)
        self.state_norm = LayerNorm(state_size)
        self.state_projection = nn.Linear(state_size, msa_size)
        self.row_attention = RfDiffusionMsaRowAttentionWithPairBias(
            msa_size=msa_size, pair_size=pair_size, num_attention_heads=num_attention_heads, head_dim=hidden_size
        )
        self.row_dropout = StructuredDropout(dropout, broadcast_dims=1)
        self.column_attention = RfDiffusionMsaColumnGlobalAttention(
            msa_size=msa_size, num_attention_heads=num_attention_heads, head_dim=hidden_size
        )
        self.transition = RfDiffusionTransition(msa_size, 4, dropout=dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.pair_projection = _initialize_lecun_normal(self.pair_projection)
        self.state_projection = _initialize_lecun_normal(self.state_projection)
        nn.init.zeros_(self.pair_projection.bias)
        nn.init.zeros_(self.state_projection.bias)

    def forward(self, msa: Tensor, pair: Tensor, radial_features: Tensor, state: Tensor) -> Tensor:
        batch_size, _, length = msa.shape[:3]
        pair_bias = torch.cat((self.pair_norm(pair), radial_features), dim=-1)
        pair_bias = self.pair_projection(pair_bias)

        query_update = self.state_projection(self.state_norm(state)).reshape(batch_size, 1, length, -1)
        msa = _query_state_update(msa, query_update)

        msa = msa + self.row_dropout(self.row_attention(msa, pair_bias))
        msa = msa + self.column_attention(msa)
        return msa + self.transition(msa)


class RfDiffusionPairBlock(nn.Module):
    r"""
    Update the pair representation with structure-biased axial attention.
    """

    def __init__(
        self,
        pair_size: int = 128,
        num_attention_heads: int = 4,
        hidden_size: int = 32,
        rbf_size: int = 36,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.radial_embedding = nn.Linear(rbf_size, hidden_size)
        self.radial_projection = nn.Linear(hidden_size, pair_size)
        self.row_attention = RfDiffusionPairAxialAttention(
            pair_size, pair_size, num_attention_heads, hidden_size, is_row=True
        )
        self.row_dropout = StructuredDropout(dropout, broadcast_dims=1)
        self.column_attention = RfDiffusionPairAxialAttention(
            pair_size, pair_size, num_attention_heads, hidden_size, is_row=False
        )
        self.column_dropout = StructuredDropout(dropout, broadcast_dims=2)
        self.transition = RfDiffusionTransition(pair_size, 2)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_normal_(self.radial_embedding.weight, nonlinearity="relu")
        nn.init.zeros_(self.radial_embedding.bias)
        self.radial_projection = _initialize_lecun_normal(self.radial_projection)
        nn.init.zeros_(self.radial_projection.bias)

    def forward(self, pair: Tensor, radial_features: Tensor) -> Tensor:
        structure_bias = self.radial_projection(F.relu_(self.radial_embedding(radial_features)))
        pair = pair + self.row_dropout(self.row_attention(pair, structure_bias))
        pair = pair + self.column_dropout(self.column_attention(pair, structure_bias))
        return pair + self.transition(pair)


class RfDiffusionOuterProductMean(OuterProductMean):
    r"""
    Compute the MSA outer-product update for the pair representation.
    """

    def __init__(
        self,
        msa_size: int = 256,
        pair_size: int = 128,
        hidden_size: int = 32,
    ) -> None:
        super().__init__(
            msa_size=msa_size,
            hidden_size=hidden_size,
            pair_size=pair_size,
        )
        self.left_projection = nn.Linear(msa_size, hidden_size)
        self.right_projection = nn.Linear(msa_size, hidden_size)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.left_projection = _initialize_lecun_normal(self.left_projection)
        self.right_projection = _initialize_lecun_normal(self.right_projection)
        nn.init.zeros_(self.left_projection.bias)
        nn.init.zeros_(self.right_projection.bias)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)


class RfDiffusionStructureUpdate(nn.Module):
    r"""
    Structure-update module: refine backbone frames with an SE(3)-equivariant transformer.
    """

    def __init__(
        self,
        msa_size: int = 256,
        pair_size: int = 128,
        state_size: int = 16,
        se3_config: Mapping[str, int] | None = None,
    ) -> None:
        super().__init__()
        se3_config = _se3_config(se3_config)
        l0_in_features = se3_config["l0_in_features"]
        l0_out_features = se3_config["l0_out_features"]
        num_edge_features = se3_config["num_edge_features"]

        self.msa_norm = LayerNorm(msa_size)
        self.state_norm = LayerNorm(state_size)
        self.node_projection = nn.Linear(msa_size + state_size, l0_in_features)
        self.node_norm = LayerNorm(l0_in_features)
        self.pair_norm = LayerNorm(pair_size)
        self.pair_projection = nn.Linear(pair_size, num_edge_features)
        self.pair_edge_norm = LayerNorm(num_edge_features)
        self.geometry_projection = nn.Linear(num_edge_features + 36 + 1, num_edge_features)
        self.geometry_norm = LayerNorm(num_edge_features)
        self.structure_module = RfDiffusionSe3StructureModule(**se3_config)
        self.torsion_head = RfDiffusionTorsionHead(msa_size=msa_size, state_size=l0_out_features)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.node_projection = _initialize_lecun_normal(self.node_projection)
        self.pair_projection = _initialize_lecun_normal(self.pair_projection)
        self.geometry_projection = _initialize_lecun_normal(self.geometry_projection)
        nn.init.zeros_(self.node_projection.bias)
        nn.init.zeros_(self.pair_projection.bias)
        nn.init.zeros_(self.geometry_projection.bias)

    def forward(
        self,
        msa: Tensor,
        pair: Tensor,
        input_rotations: Tensor,
        input_translations: Tensor,
        coords: Tensor,
        state: Tensor,
        residue_index: Tensor,
        fixed_structure_mask: Tensor | None,
        cyclic_mask: Tensor | None = None,
        top_k: int = 64,
    ) -> _RfDiffusionStructureUpdateOutput:
        # SE(3)/frame geometry must run in fp32; disable autocast device-generically. A "cuda"-hardcoded
        # guard is a no-op on CPU/MPS/XPU, where an ambient autocast would silently downcast this math.
        with torch.autocast(device_type=msa.device.type, enabled=False):
            batch_size, _, length = msa.shape[:3]
            fixed_structure_mask = (
                torch.zeros_like(residue_index, dtype=torch.bool)
                if fixed_structure_mask is None
                else fixed_structure_mask
            )
            if fixed_structure_mask.ndim == 1:
                fixed_structure_mask = fixed_structure_mask.unsqueeze(0).expand(batch_size, -1)

            node = torch.cat((self.msa_norm(msa[:, 0]), self.state_norm(state)), dim=-1)
            node = self.node_norm(self.node_projection(node))
            pair = self.pair_edge_norm(self.pair_projection(self.pair_norm(pair)))

            neighbor = _sequence_separation(residue_index, cyclic_mask)
            pair = torch.cat((pair, _backbone_ca_distogram(coords), neighbor), dim=-1)
            pair = self.geometry_norm(self.geometry_projection(pair))

            ca_coords = coords[:, :, 1, :]
            graph, edge_features = (
                _top_k_graph(ca_coords, pair, residue_index, top_k=top_k)
                if top_k != 0
                else _full_graph(ca_coords, pair, residue_index)
            )
            vector_features = (coords - ca_coords.unsqueeze(2)).reshape(batch_size * length, -1, 3)
            shift = self.structure_module(
                graph,
                node.reshape(batch_size * length, -1, 1),
                vector_features,
                edge_features,
            )

            state = shift["0"].reshape(batch_size, length, -1)
            offset = shift["1"].reshape(batch_size, length, 2, 3)
            offset = offset.masked_fill(fixed_structure_mask[:, :, None, None], 0)

            translations = offset[:, :, 0, :] / 10.0
            rotation_vector = offset[:, :, 1, :] / 100.0
            delta_rotations = _quaternion_update_matrix(rotation_vector, device=coords.device)
            rotations = einsum("bnij,bnjk->bnik", delta_rotations, input_rotations)
            translations = translations + input_translations
            torsion_angles = self.torsion_head(msa[:, 0], state)
            return _RfDiffusionStructureUpdateOutput(  # type: ignore[call-arg]
                rotations=rotations,
                translations=translations,
                state=state,
                torsion_angles=torsion_angles,
                batch_size=[batch_size],
            )


class RfDiffusionTransition(nn.Module):
    r"""
    Pre-norm transition with a ReLU-activated expansion.
    """

    def __init__(self, hidden_size: int, expansion: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.input_norm = LayerNorm(hidden_size)
        self.input_projection = nn.Linear(hidden_size, hidden_size * expansion)
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(hidden_size * expansion, hidden_size)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_normal_(self.input_projection.weight, nonlinearity="relu")
        nn.init.zeros_(self.input_projection.bias)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        src = self.input_norm(src)
        return self.output_projection(self.dropout(F.relu_(self.input_projection(src))))


class RfDiffusionMsaSequenceWeighting(nn.Module):
    r"""
    Per-head softmax weights over MSA rows relative to the query row.
    """

    def __init__(self, msa_size: int, num_attention_heads: int, head_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.query_projection = nn.Linear(msa_size, num_attention_heads * head_dim)
        self.key_projection = nn.Linear(msa_size, num_attention_heads * head_dim)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.query_projection.weight)
        nn.init.xavier_uniform_(self.key_projection.weight)

    def forward(self, msa: torch.Tensor) -> torch.Tensor:
        batch_size, msa_depth, length = msa.shape[:3]
        query = self.query_projection(msa[:, 0]).view(batch_size, 1, length, self.num_attention_heads, self.head_dim)
        key = self.key_projection(msa).view(batch_size, msa_depth, length, self.num_attention_heads, self.head_dim)
        attn = einsum("bqihd,bkihd->bkihq", query * self.scale, key)
        return self.dropout(F.softmax(attn, dim=1))


class RfDiffusionMsaRowAttentionWithPairBias(nn.Module):
    r"""
    Row (per-column) MSA attention biased by the pair representation and gated.
    """

    def __init__(
        self, msa_size: int = 256, pair_size: int = 128, num_attention_heads: int = 8, head_dim: int = 32
    ) -> None:
        super().__init__()
        self.msa_norm = LayerNorm(msa_size)
        self.pair_norm = LayerNorm(pair_size)
        self.sequence_weighting = RfDiffusionMsaSequenceWeighting(msa_size, num_attention_heads, head_dim, dropout=0.1)
        self.query_projection = nn.Linear(msa_size, num_attention_heads * head_dim, bias=False)
        self.key_projection = nn.Linear(msa_size, num_attention_heads * head_dim, bias=False)
        self.value_projection = nn.Linear(msa_size, num_attention_heads * head_dim, bias=False)
        self.bias_projection = nn.Linear(pair_size, num_attention_heads, bias=False)
        self.gate_projection = nn.Linear(msa_size, num_attention_heads * head_dim)
        self.attention = ProjectedAttention()
        self.output_projection = nn.Linear(num_attention_heads * head_dim, msa_size)
        self.scale = 1 / math.sqrt(head_dim)
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.query_projection.weight)
        nn.init.xavier_uniform_(self.key_projection.weight)
        nn.init.xavier_uniform_(self.value_projection.weight)
        self.bias_projection = _initialize_lecun_normal(self.bias_projection)
        nn.init.zeros_(self.gate_projection.weight)
        nn.init.ones_(self.gate_projection.bias)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, msa: torch.Tensor, pair: torch.Tensor) -> torch.Tensor:
        batch_size, msa_depth, length = msa.shape[:3]
        msa = self.msa_norm(msa)
        pair = self.pair_norm(pair)
        seq_weight = self.sequence_weighting(msa)
        query = self.query_projection(msa).reshape(
            batch_size, msa_depth, length, self.num_attention_heads, self.head_dim
        )
        key = self.key_projection(msa).reshape(batch_size, msa_depth, length, self.num_attention_heads, self.head_dim)
        value = self.value_projection(msa).reshape(
            batch_size, msa_depth, length, self.num_attention_heads, self.head_dim
        )
        bias = self.bias_projection(pair)
        gate = torch.sigmoid(self.gate_projection(msa))
        query = query * seq_weight.expand(-1, -1, -1, -1, self.head_dim)
        query = query.permute(0, 3, 2, 1, 4).reshape(
            batch_size,
            self.num_attention_heads,
            length,
            msa_depth * self.head_dim,
        )
        key = key.permute(0, 3, 2, 1, 4).reshape(
            batch_size,
            self.num_attention_heads,
            length,
            msa_depth * self.head_dim,
        )
        value = value.permute(0, 3, 2, 1, 4).reshape(
            batch_size,
            self.num_attention_heads,
            length,
            msa_depth * self.head_dim,
        )
        out = self.attention(
            query,
            key,
            value,
            biases=(bias.permute(0, 3, 1, 2),),
            scale=self.scale,
        )
        out = (
            out.reshape(
                batch_size,
                self.num_attention_heads,
                length,
                msa_depth,
                self.head_dim,
            )
            .permute(0, 3, 2, 1, 4)
            .reshape(batch_size, msa_depth, length, -1)
        )
        return self.output_projection(gate * out)


class RfDiffusionMsaColumnAttention(nn.Module):
    r"""
    Column (per-row) MSA attention across sequences, gated.
    """

    def __init__(self, msa_size: int = 256, num_attention_heads: int = 8, head_dim: int = 32) -> None:
        super().__init__()
        self.msa_norm = LayerNorm(msa_size)
        self.query_projection = nn.Linear(msa_size, num_attention_heads * head_dim, bias=False)
        self.key_projection = nn.Linear(msa_size, num_attention_heads * head_dim, bias=False)
        self.value_projection = nn.Linear(msa_size, num_attention_heads * head_dim, bias=False)
        self.gate_projection = nn.Linear(msa_size, num_attention_heads * head_dim)
        self.attention = ProjectedAttention()
        self.output_projection = nn.Linear(num_attention_heads * head_dim, msa_size)
        self.scale = 1 / math.sqrt(head_dim)
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.query_projection.weight)
        nn.init.xavier_uniform_(self.key_projection.weight)
        nn.init.xavier_uniform_(self.value_projection.weight)
        nn.init.zeros_(self.gate_projection.weight)
        nn.init.ones_(self.gate_projection.bias)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, msa: torch.Tensor) -> torch.Tensor:
        batch_size, msa_depth, length = msa.shape[:3]
        msa = self.msa_norm(msa)
        query = self.query_projection(msa).reshape(
            batch_size, msa_depth, length, self.num_attention_heads, self.head_dim
        )
        key = self.key_projection(msa).reshape(batch_size, msa_depth, length, self.num_attention_heads, self.head_dim)
        value = self.value_projection(msa).reshape(
            batch_size, msa_depth, length, self.num_attention_heads, self.head_dim
        )
        gate = torch.sigmoid(self.gate_projection(msa))
        query = query.permute(0, 2, 3, 1, 4)
        key = key.permute(0, 2, 3, 1, 4)
        value = value.permute(0, 2, 3, 1, 4)
        out = self.attention(query, key, value, scale=self.scale)
        out = out.permute(0, 3, 1, 2, 4).reshape(batch_size, msa_depth, length, -1)
        return self.output_projection(gate * out)


class RfDiffusionMsaColumnGlobalAttention(nn.Module):
    r"""
    Global column MSA attention with a mean-pooled query, gated.
    """

    def __init__(self, msa_size: int = 64, num_attention_heads: int = 8, head_dim: int = 8) -> None:
        super().__init__()
        self.msa_norm = LayerNorm(msa_size)
        self.query_projection = nn.Linear(msa_size, num_attention_heads * head_dim, bias=False)
        self.key_projection = nn.Linear(msa_size, head_dim, bias=False)
        self.value_projection = nn.Linear(msa_size, head_dim, bias=False)
        self.gate_projection = nn.Linear(msa_size, num_attention_heads * head_dim)
        self.attention = ProjectedAttention()
        self.output_projection = nn.Linear(num_attention_heads * head_dim, msa_size)
        self.scale = 1 / math.sqrt(head_dim)
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.query_projection.weight)
        nn.init.xavier_uniform_(self.key_projection.weight)
        nn.init.xavier_uniform_(self.value_projection.weight)
        nn.init.zeros_(self.gate_projection.weight)
        nn.init.ones_(self.gate_projection.bias)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, msa: torch.Tensor) -> torch.Tensor:
        batch_size, msa_depth, length = msa.shape[:3]
        msa = self.msa_norm(msa)
        query = (
            self.query_projection(msa)
            .reshape(batch_size, msa_depth, length, self.num_attention_heads, self.head_dim)
            .mean(dim=1)
        )
        key = self.key_projection(msa)
        value = self.value_projection(msa)
        gate = torch.sigmoid(self.gate_projection(msa))
        query = query.unsqueeze(-2)
        key = key.permute(0, 2, 1, 3).unsqueeze(2).expand(-1, -1, self.num_attention_heads, -1, -1)
        value = value.permute(0, 2, 1, 3).unsqueeze(2).expand(-1, -1, self.num_attention_heads, -1, -1)
        out = self.attention(query, key, value, scale=self.scale)
        out = out.squeeze(-2).reshape(batch_size, 1, length, -1)
        return self.output_projection(gate * out)


class RfDiffusionPairAxialAttention(nn.Module):
    r"""
    Axial (row or column) attention over the pair representation with an additive bias, gated.
    """

    def __init__(
        self,
        pair_size: int,
        bias_size: int,
        num_attention_heads: int,
        head_dim: int,
        is_row: bool = True,
    ) -> None:
        super().__init__()
        self.is_row = is_row
        self.pair_norm = LayerNorm(pair_size)
        self.bias_norm = LayerNorm(bias_size)
        self.query_projection = nn.Linear(pair_size, num_attention_heads * head_dim, bias=False)
        self.key_projection = nn.Linear(pair_size, num_attention_heads * head_dim, bias=False)
        self.value_projection = nn.Linear(pair_size, num_attention_heads * head_dim, bias=False)
        self.bias_projection = nn.Linear(bias_size, num_attention_heads, bias=False)
        self.gate_projection = nn.Linear(pair_size, num_attention_heads * head_dim)
        self.attention = ProjectedAttention()
        self.output_projection = nn.Linear(num_attention_heads * head_dim, pair_size)
        self.scale = 1 / math.sqrt(head_dim)
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.query_projection.weight)
        nn.init.xavier_uniform_(self.key_projection.weight)
        nn.init.xavier_uniform_(self.value_projection.weight)
        self.bias_projection = _initialize_lecun_normal(self.bias_projection)
        nn.init.zeros_(self.gate_projection.weight)
        nn.init.ones_(self.gate_projection.bias)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, pair: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        batch_size, length = pair.shape[:2]
        if self.is_row:
            pair = pair.permute(0, 2, 1, 3)
            bias = bias.permute(0, 2, 1, 3)
        pair = self.pair_norm(pair)
        bias = self.bias_norm(bias)
        query = self.query_projection(pair).reshape(batch_size, length, length, self.num_attention_heads, self.head_dim)
        key = self.key_projection(pair).reshape(batch_size, length, length, self.num_attention_heads, self.head_dim)
        value = self.value_projection(pair).reshape(batch_size, length, length, self.num_attention_heads, self.head_dim)
        bias = self.bias_projection(bias)
        gate = torch.sigmoid(self.gate_projection(pair))
        query = query.permute(0, 3, 2, 1, 4).reshape(
            batch_size,
            self.num_attention_heads,
            length,
            length * self.head_dim,
        )
        key = key.permute(0, 3, 2, 1, 4).reshape(
            batch_size,
            self.num_attention_heads,
            length,
            length * self.head_dim,
        )
        value = value.permute(0, 3, 2, 1, 4).reshape(
            batch_size,
            self.num_attention_heads,
            length,
            length * self.head_dim,
        )
        out = self.attention(
            query,
            key,
            value,
            biases=(bias.permute(0, 3, 1, 2),),
            scale=self.scale / math.sqrt(length),
        )
        out = (
            out.reshape(
                batch_size,
                self.num_attention_heads,
                length,
                length,
                self.head_dim,
            )
            .permute(0, 2, 3, 1, 4)
            .reshape(batch_size, length, length, -1)
        )
        out = self.output_projection(gate * out)
        if self.is_row:
            out = out.permute(0, 2, 1, 3)
        return out


class RfDiffusionGraph:
    r"""
    Sparse residue graph consumed by the SE(3) transformer.
    """

    def __init__(
        self,
        coordinates: torch.Tensor,
        pair: torch.Tensor,
        batch: torch.Tensor,
        source_residue_index: torch.Tensor,
        destination_residue_index: torch.Tensor,
        edge_mask: torch.Tensor,
        *,
        batch_size: int,
        length: int,
    ) -> None:
        source = batch * length + source_residue_index
        destination = batch * length + destination_residue_index
        order = torch.argsort(destination * (batch_size * length) + source, stable=True)
        source, destination = source[order], destination[order]
        batch = batch[order]
        source_residue_index = source_residue_index[order]
        destination_residue_index = destination_residue_index[order]
        self.edge_mask = edge_mask[order]
        self.source = source
        self.destination = destination
        self.relative_positions = (
            coordinates[batch, destination_residue_index, :] - coordinates[batch, source_residue_index, :]
        ).detach()
        self.relative_positions = self.relative_positions.masked_fill(~self.edge_mask[:, None], 0)
        self.relative_positions[:, 0] = self.relative_positions[:, 0].masked_fill(~self.edge_mask, 1)
        self.edge_features = pair[batch, source_residue_index, destination_residue_index][..., None]
        self.edge_features = self.edge_features.masked_fill(~self.edge_mask[:, None, None], 0)
        self.num_nodes = batch_size * length
        counts = torch.zeros(self.num_nodes, dtype=torch.long, device=coordinates.device)
        counts.scatter_add_(0, destination, torch.ones_like(destination))
        self.row_offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))

    def edges(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.source, self.destination


class RfDiffusionSe3Fiber(dict[int, int]):
    r"""
    Degree-to-channel structure of an SE(3)-equivariant feature space.
    """

    def __init__(self, structure: Mapping[int, int] | Iterable[tuple[int, int]]) -> None:
        super().__init__(sorted(dict(structure).items()))

    def __missing__(self, degree: int) -> int:
        return 0

    @property
    def degrees(self) -> list[int]:
        return sorted(self)

    @property
    def channels(self) -> list[int]:
        return [self[degree] for degree in self.degrees]

    @property
    def num_features(self) -> int:
        return sum(channels * _degree_to_dimension(degree) for degree, channels in self.items())

    @staticmethod
    def create(num_degrees: int, num_channels: int) -> RfDiffusionSe3Fiber:
        return RfDiffusionSe3Fiber((degree, num_channels) for degree in range(num_degrees))

    def __add__(self, other):
        if isinstance(other, RfDiffusionSe3Fiber):
            return RfDiffusionSe3Fiber({degree: channels + other[degree] for degree, channels in self.items()})
        if isinstance(other, int):
            return RfDiffusionSe3Fiber({degree: channels + other for degree, channels in self.items()})
        return NotImplemented


class RfDiffusionSe3RadialProfile(nn.Module):
    r"""
    MLP mapping invariant edge features to the radial weights of an SE(3) convolution.
    """

    def __init__(
        self,
        num_freq: int,
        channels_in: int,
        channels_out: int,
        edge_dim: int = 1,
        mid_dim: int = 32,
        use_layer_norm: bool = False,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(edge_dim, mid_dim)
        self.input_norm = LayerNorm(mid_dim) if use_layer_norm else None
        self.input_activation = nn.ReLU()
        self.hidden_projection = nn.Linear(mid_dim, mid_dim)
        self.hidden_norm = LayerNorm(mid_dim) if use_layer_norm else None
        self.hidden_activation = nn.ReLU()
        self.output_projection = nn.Linear(
            mid_dim,
            num_freq * channels_in * channels_out,
            bias=False,
        )

    def forward(self, features: Tensor) -> Tensor:
        features = self.input_projection(features)
        if self.input_norm is not None:
            features = self.input_norm(features)
        features = self.input_activation(features)
        features = self.hidden_projection(features)
        if self.hidden_norm is not None:
            features = self.hidden_norm(features)
        return self.output_projection(self.hidden_activation(features))


class RfDiffusionSe3TensorProductConvolution(nn.Module):
    r"""
    A single degree-in to degree-out SE(3) convolution parameterized by a radial profile.
    """

    def __init__(
        self,
        freq_sum: int,
        channels_in: int,
        channels_out: int,
        edge_dim: int,
        use_layer_norm: bool,
    ) -> None:
        super().__init__()
        self.freq_sum = freq_sum
        self.channels_out = channels_out
        self.channels_in = channels_in
        self.radial_profile = RfDiffusionSe3RadialProfile(
            num_freq=freq_sum,
            channels_in=channels_in,
            channels_out=channels_out,
            edge_dim=edge_dim,
            use_layer_norm=use_layer_norm,
        )
        self.tensor_product = WeightedTensorProduct()

    def forward(self, features: Tensor, invariant_edge_features: Tensor, basis: Tensor | None) -> Tensor:
        radial_weights = self.radial_profile(invariant_edge_features).view(
            -1, self.channels_out, self.channels_in * self.freq_sum
        )
        if basis is None:
            return radial_weights @ features
        return self.tensor_product(features, basis, radial_weights)


class RfDiffusionSe3Linear(nn.Module):
    r"""
    Degree-wise equivariant linear layer: an independent channel mix per representation degree.
    """

    def __init__(self, fiber_in: RfDiffusionSe3Fiber, fiber_out: RfDiffusionSe3Fiber) -> None:
        super().__init__()
        self.weights = nn.ParameterDict(
            {
                str(degree_out): nn.Parameter(
                    torch.randn(channels_out, fiber_in[degree_out]) / math.sqrt(fiber_in[degree_out])
                )
                for degree_out, channels_out in fiber_out.items()
            }
        )

    def forward(self, features: dict[str, Tensor]) -> dict[str, Tensor]:
        return {
            degree: torch.einsum("oc,ncd->nod", weight, features[degree]) for degree, weight in self.weights.items()
        }


class RfDiffusionSe3Norm(nn.Module):
    r"""
    Equivariant norm-nonlinearity: rescale each degree's feature by a learned function of its norm.
    """

    NORM_CLAMP = 2**-24

    def __init__(self, fiber: RfDiffusionSe3Fiber, nonlinearity: nn.Module | None = None) -> None:
        super().__init__()
        self.fiber = fiber
        self.group_norm = nn.GroupNorm(num_groups=len(fiber.degrees), num_channels=sum(fiber.channels))
        self.nonlinearity = nn.ReLU() if nonlinearity is None else nonlinearity

    def forward(self, features: dict[str, Tensor]) -> dict[str, Tensor]:
        output: dict[str, Tensor] = {}
        norms = [
            features[str(degree)].norm(dim=-1, keepdim=True).clamp(min=self.NORM_CLAMP) for degree in self.fiber.degrees
        ]
        fused_norms = torch.cat(norms, dim=-2)
        new_norms = self.nonlinearity(self.group_norm(fused_norms.squeeze(-1))).unsqueeze(-1)
        new_norms = torch.chunk(new_norms, chunks=len(self.fiber.degrees), dim=-2)
        for norm, new_norm, degree in zip(norms, new_norms, self.fiber.degrees):
            output[str(degree)] = features[str(degree)] / norm * new_norm
        return output


class RfDiffusionSe3Convolution(nn.Module):
    r"""
    SE(3)-equivariant graph convolution (message passing) between two fibers.
    """

    def __init__(
        self,
        fiber_in: RfDiffusionSe3Fiber,
        fiber_out: RfDiffusionSe3Fiber,
        fiber_edge: RfDiffusionSe3Fiber,
        pool: bool = True,
        use_layer_norm: bool = False,
        self_interaction: bool = False,
        max_degree: int = 1,
        allow_fused_output: bool = False,
    ) -> None:
        super().__init__()
        self.pool = pool
        self.fiber_in = fiber_in
        self.fiber_out = fiber_out
        self.self_interaction = self_interaction
        self.max_degree = max_degree
        self.allow_fused_output = allow_fused_output
        channels_in_set = {channels + fiber_edge[degree] * (degree > 0) for degree, channels in self.fiber_in.items()}
        channels_out_set = set(self.fiber_out.values())
        unique_channels_in = len(channels_in_set) == 1
        degrees_up_to_max = list(range(max_degree + 1))
        edge_dim = fiber_edge[0] + 1
        self.fused_output = unique_channels_in and fiber_in.degrees == degrees_up_to_max
        self.output_convolutions = nn.ModuleDict()
        self.input_convolutions = nn.ModuleDict()

        if self.fused_output:
            for degree_out, channels_out in fiber_out.items():
                sum_freq = sum(_degree_to_dimension(min(degree_out, degree)) for degree in fiber_in.degrees)
                self.output_convolutions[str(degree_out)] = RfDiffusionSe3TensorProductConvolution(
                    sum_freq,
                    list(channels_in_set)[0],
                    channels_out,
                    edge_dim=edge_dim,
                    use_layer_norm=use_layer_norm,
                )
        else:
            for degree_in, channels_in in fiber_in.items():
                sum_freq = sum(_degree_to_dimension(min(degree_in, degree)) for degree in fiber_out.degrees)
                self.input_convolutions[str(degree_in)] = RfDiffusionSe3TensorProductConvolution(
                    sum_freq,
                    channels_in,
                    list(channels_out_set)[0],
                    edge_dim=edge_dim,
                    use_layer_norm=use_layer_norm,
                )

        if self_interaction:
            self.self_interaction_weights = nn.ParameterDict()
            for degree_out, channels_out in fiber_out.items():
                if fiber_in[degree_out]:
                    self.self_interaction_weights[str(degree_out)] = nn.Parameter(
                        torch.randn(channels_out, fiber_in[degree_out]) / math.sqrt(fiber_in[degree_out])
                    )
        self.segment_sum = SegmentSumCsr()

    def forward(
        self,
        node_feats: dict[str, Tensor],
        edge_features: dict[str, Tensor],
        graph,
        basis: dict[str, Tensor],
    ) -> dict[str, Tensor] | Tensor:
        invariant_edge_features = edge_features["0"].squeeze(-1)
        source, destination = graph.edges()
        in_features = []
        for degree_in in self.fiber_in.degrees:
            source_node_features = node_feats[str(degree_in)][source]
            if degree_in > 0 and str(degree_in) in edge_features:
                source_node_features = torch.cat([source_node_features, edge_features[str(degree_in)]], dim=1)
            in_features.append(source_node_features)

        out: dict[str, Tensor]
        if self.fused_output:
            in_features_fused = torch.cat(in_features, dim=-1)
            out = {
                str(degree_out): self.output_convolutions[str(degree_out)](
                    in_features_fused, invariant_edge_features, basis[f"out{degree_out}_fused"]
                )
                for degree_out in self.fiber_out.degrees
            }
        else:
            out_tensor = torch.zeros((), dtype=in_features[0].dtype, device=in_features[0].device)
            for degree_in, feature in zip(self.fiber_in.degrees, in_features):
                out_tensor = out_tensor + self.input_convolutions[str(degree_in)](
                    feature, invariant_edge_features, basis[f"in{degree_in}_fused"]
                )
            if not self.allow_fused_output or self.self_interaction or self.pool:
                out = _unfuse_features(out_tensor, self.fiber_out.degrees)
            else:
                return out_tensor

        for degree_out in self.fiber_out.degrees:
            if self.self_interaction and str(degree_out) in self.self_interaction_weights:
                destination_features = node_feats[str(degree_out)][destination]
                out[str(degree_out)] = out[str(degree_out)] + torch.einsum(
                    "oc,ecd->eod", self.self_interaction_weights[str(degree_out)], destination_features
                )
            out[str(degree_out)] = out[str(degree_out)].masked_fill(
                ~graph.edge_mask[:, None, None],
                0,
            )
            if self.pool:
                out[str(degree_out)] = self.segment_sum(
                    out[str(degree_out)],
                    graph.row_offsets,
                    num_segments=graph.num_nodes,
                )
        return out


class RfDiffusionSe3Attention(nn.Module):
    r"""
    Multi-head SE(3)-equivariant attention over graph edges.
    """

    def __init__(
        self, num_attention_heads: int, key_fiber: RfDiffusionSe3Fiber, value_fiber: RfDiffusionSe3Fiber
    ) -> None:
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.key_fiber = key_fiber
        self.value_fiber = value_fiber
        self.segment_softmax = SegmentSoftmaxCsr()
        self.segment_sum = SegmentSumCsr()

    def forward(
        self,
        value: Tensor,
        key: Tensor,
        query: dict[str, Tensor],
        graph,
    ) -> dict[str, Tensor]:
        key_heads = key.reshape(key.shape[0], self.num_attention_heads, -1)
        query_tensor = torch.cat([query[str(degree)] for degree in self.key_fiber.degrees], dim=-1)
        query_heads = query_tensor.reshape(query_tensor.shape[0], self.num_attention_heads, -1)

        edge_weights = (key_heads * query_heads[graph.destination]).sum(dim=-1) / math.sqrt(self.key_fiber.num_features)
        edge_weights = edge_weights.masked_fill(
            ~graph.edge_mask[:, None],
            torch.finfo(edge_weights.dtype).min,
        )
        edge_weights = self.segment_softmax(edge_weights, graph.row_offsets)
        edge_weights = edge_weights.masked_fill(~graph.edge_mask[:, None], 0)[..., None, None]

        value_heads = value.view(value.shape[0], self.num_attention_heads, -1, value.shape[-1])
        feat_out = self.segment_sum(
            edge_weights * value_heads,
            graph.row_offsets,
            num_segments=graph.num_nodes,
        )
        feat_out = feat_out.view(feat_out.shape[0], -1, feat_out.shape[-1])
        return _unfuse_features(feat_out, self.value_fiber.degrees)


class RfDiffusionSe3AttentionBlock(nn.Module):
    r"""
    One SE(3)-equivariant attention block: key/value convolution, query linear, attention, and projection.
    """

    def __init__(
        self,
        fiber_in: RfDiffusionSe3Fiber,
        fiber_out: RfDiffusionSe3Fiber,
        fiber_edge: RfDiffusionSe3Fiber,
        num_attention_heads: int = 4,
        channels_div: int = 2,
        use_layer_norm: bool = False,
        max_degree: int = 1,
    ) -> None:
        super().__init__()
        value_fiber = RfDiffusionSe3Fiber((degree, channels // channels_div) for degree, channels in fiber_out.items())
        key_query_fiber = RfDiffusionSe3Fiber(
            (degree, channels) for degree, channels in value_fiber.items() if degree in fiber_in.degrees
        )
        self.key_value_projection = RfDiffusionSe3Convolution(
            fiber_in,
            value_fiber + key_query_fiber,
            pool=False,
            fiber_edge=fiber_edge,
            use_layer_norm=use_layer_norm,
            max_degree=max_degree,
            allow_fused_output=True,
        )
        self.query_projection = RfDiffusionSe3Linear(fiber_in, key_query_fiber)
        self.attention = RfDiffusionSe3Attention(num_attention_heads, key_query_fiber, value_fiber)
        self.output_projection = RfDiffusionSe3Linear(value_fiber + fiber_in, fiber_out)
        self.output_norm = RfDiffusionSe3Norm(fiber_out)

    def forward(
        self,
        node_features: dict[str, Tensor],
        edge_features: dict[str, Tensor],
        graph: RfDiffusionGraph,
        basis: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        fused_key_value = self.key_value_projection(node_features, edge_features, graph, basis)
        key, value = self._get_key_value_from_fused(fused_key_value)
        query = self.query_projection(node_features)
        z = self.attention(value, key, query, graph)
        z_concat = {
            degree: torch.cat([value, node_features[degree]], dim=1) if degree in node_features else value
            for degree, value in z.items()
        }
        return self.output_norm(self.output_projection(z_concat))

    def _get_key_value_from_fused(self, fused_key_value: Tensor) -> tuple[Tensor, Tensor]:
        value, key = torch.chunk(fused_key_value, chunks=2, dim=-2)
        return key, value


class RfDiffusionSe3Transformer(nn.Module):
    r"""
    Stack of equivariant attention blocks plus a final convolution over a residue graph.
    """

    def __init__(
        self,
        num_blocks: int,
        fiber_in: RfDiffusionSe3Fiber,
        fiber_hidden: RfDiffusionSe3Fiber,
        fiber_out: RfDiffusionSe3Fiber,
        num_attention_heads: int,
        channels_div: int,
        fiber_edge: RfDiffusionSe3Fiber | None = None,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        fiber_edge = RfDiffusionSe3Fiber({}) if fiber_edge is None else fiber_edge
        self.num_blocks = num_blocks
        self.fiber_edge = fiber_edge
        self.num_attention_heads = num_attention_heads
        self.channels_div = channels_div
        self.max_degree = max(*fiber_in.degrees, *fiber_hidden.degrees, *fiber_out.degrees)
        self.basis = Se3Basis(self.max_degree)
        blocks = []
        for _ in range(num_blocks):
            blocks.append(
                RfDiffusionSe3AttentionBlock(
                    fiber_in=fiber_in,
                    fiber_out=fiber_hidden,
                    fiber_edge=fiber_edge,
                    num_attention_heads=num_attention_heads,
                    channels_div=channels_div,
                    use_layer_norm=use_layer_norm,
                    max_degree=self.max_degree,
                )
            )
            fiber_in = fiber_hidden
        self.blocks = nn.ModuleList(blocks)
        self.output_projection = RfDiffusionSe3Convolution(
            fiber_in=fiber_in,
            fiber_out=fiber_out,
            fiber_edge=fiber_edge,
            self_interaction=True,
            use_layer_norm=use_layer_norm,
            max_degree=self.max_degree,
        )

    def forward(
        self,
        graph,
        node_feats: dict[str, Tensor],
        edge_features: dict[str, Tensor] | None = None,
        basis: dict[str, Tensor] | None = None,
    ) -> dict[str, Tensor]:
        basis = basis or self.basis(graph.relative_positions)
        edge_features = _populated_edge_features(graph.relative_positions, edge_features)
        for block in self.blocks:
            node_feats = block(node_feats, edge_features, graph, basis)
        return self.output_projection(node_feats, edge_features, graph, basis)


class RfDiffusionSe3StructureModule(nn.Module):
    r"""
    Adapt the SE(3) transformer to RFdiffusion structure features.
    """

    def __init__(
        self,
        num_blocks: int = 2,
        num_channels: int = 32,
        num_degrees: int = 3,
        num_attention_heads: int = 4,
        div: int = 4,
        l0_in_features: int = 32,
        l0_out_features: int = 32,
        l1_in_features: int = 3,
        l1_out_features: int = 2,
        num_edge_features: int = 32,
    ) -> None:
        super().__init__()
        self.num_vector_inputs = l1_in_features
        fiber_edge = RfDiffusionSe3Fiber({0: num_edge_features})
        fiber_in = RfDiffusionSe3Fiber({0: l0_in_features, 1: l1_in_features})
        fiber_hidden = RfDiffusionSe3Fiber.create(num_degrees, num_channels)
        fiber_out = RfDiffusionSe3Fiber({0: l0_out_features, 1: l1_out_features})
        self.transformer = RfDiffusionSe3Transformer(
            num_blocks=num_blocks,
            fiber_in=fiber_in,
            fiber_hidden=fiber_hidden,
            fiber_out=fiber_out,
            num_attention_heads=num_attention_heads,
            channels_div=div,
            fiber_edge=fiber_edge,
            use_layer_norm=True,
        )

    def forward(
        self,
        graph,
        type_0_features: Tensor,
        type_1_features: Tensor | None = None,
        edge_features: Tensor | None = None,
    ) -> dict[str, Tensor]:
        node_features = {"0": type_0_features}
        if self.num_vector_inputs > 0:
            node_features["1"] = type_1_features
        return self.transformer(graph, node_features, {"0": edge_features})


class RfDiffusionAllAtomReconstruction(nn.Module):
    r"""
    Reconstruct RFdiffusion atom14 coords from backbone frames and predicted torsion angles.
    """

    def __init__(self) -> None:
        super().__init__()
        frame_indices, base_frame_transforms, base_frame_coordinates = _RF_ATOM14_KINEMATICS
        self.register_buffer("base_frame_transforms", base_frame_transforms, persistent=False)
        self.register_buffer("base_frame_coordinates", base_frame_coordinates, persistent=False)
        self.register_buffer("frame_indices", frame_indices, persistent=False)

    def forward(
        self,
        seq: torch.Tensor,
        coords: torch.Tensor,
        alphas: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq = seq.long()
        batch_size, length = coords.shape[:2]
        if alphas is None:
            alphas = torch.zeros((batch_size, length, 10, 2), dtype=coords.dtype, device=coords.device)
            alphas[..., 0] = 1.0
        alphas = alphas.to(device=coords.device, dtype=coords.dtype)

        rotations, translations = rigid_from_three_points(coords[..., 0, :], coords[..., 1, :], coords[..., 2, :])
        rtf0 = torch.eye(4, dtype=coords.dtype, device=coords.device).repeat(batch_size, length, 1, 1)
        rtf0[:, :, :3, :3] = rotations
        rtf0[:, :, :3, 3] = translations

        rts = self.base_frame_transforms.to(device=coords.device, dtype=coords.dtype)[seq]
        rtf1 = torch.einsum("blij,bljk,blkm->blim", rtf0, rts[:, :, 0], _rotation_x(alphas[:, :, 0]))
        rtf2 = torch.einsum("blij,bljk,blkm->blim", rtf0, rts[:, :, 1], _rotation_x(alphas[:, :, 1]))
        rtf3 = torch.einsum("blij,bljk,blkm->blim", rtf0, rts[:, :, 2], _rotation_x(alphas[:, :, 2]))

        base_coords = self.base_frame_coordinates.to(device=coords.device, dtype=coords.dtype)[seq]
        nc_midpoint = 0.5 * (base_coords[:, :, 2, :3] + base_coords[:, :, 0, :3])
        ca_ref = base_coords[:, :, 1, :3]
        cb_ref = base_coords[:, :, 4, :3]
        cb_axis_1 = torch.cross(cb_ref - ca_ref, nc_midpoint - ca_ref, dim=-1)
        cb_axis_1 = cb_axis_1 / torch.linalg.norm(cb_axis_1, dim=-1, keepdim=True).clamp_min(1e-8)

        nc_projection = base_coords[:, :, 2, :3] - base_coords[:, :, 0, :3]
        nc_projection = nc_projection - (
            torch.sum(nc_projection * nc_midpoint, dim=-1, keepdim=True)
            / torch.sum(nc_midpoint * nc_midpoint, dim=-1, keepdim=True).clamp_min(1e-8)
            * nc_midpoint
        )
        cb_axis_2 = torch.cross(cb_ref - ca_ref, nc_projection, dim=-1)
        cb_axis_2 = cb_axis_2 / torch.linalg.norm(cb_axis_2, dim=-1, keepdim=True).clamp_min(1e-8)
        cb_rot_1 = _rotation_about_axis(alphas[:, :, 7], cb_axis_1)
        cb_rot_2 = _rotation_about_axis(alphas[:, :, 8], cb_axis_2)
        rtf8 = torch.einsum("blij,bljk,blkm->blim", rtf0, cb_rot_1, cb_rot_2)

        rtf4 = torch.einsum(
            "blij,bljk,blkm,blmn->blin",
            rtf8,
            rts[:, :, 3],
            _rotation_x(alphas[:, :, 3]),
            _rotation_z(alphas[:, :, 9]),
        )
        rtf5 = torch.einsum("blij,bljk,blkm->blim", rtf4, rts[:, :, 4], _rotation_x(alphas[:, :, 4]))
        rtf6 = torch.einsum("blij,bljk,blkm->blim", rtf5, rts[:, :, 5], _rotation_x(alphas[:, :, 5]))
        rtf7 = torch.einsum("blij,bljk,blkm->blim", rtf6, rts[:, :, 6], _rotation_x(alphas[:, :, 6]))
        frames = torch.stack((rtf0, rtf1, rtf2, rtf3, rtf4, rtf5, rtf6, rtf7, rtf8), dim=2)

        gather_index = self.frame_indices.to(device=coords.device)[seq][..., None, None].repeat(1, 1, 1, 4, 4)
        atom_frames = frames.gather(2, gather_index)
        coords_out = torch.einsum("bltij,bltj->blti", atom_frames, base_coords)
        return frames, coords_out[..., :14, :3]


class RfDiffusionTorsionBlock(nn.Module):
    r"""
    One residual ReLU feed-forward unit in the torsion head.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.input_projection = nn.Linear(hidden_size, hidden_size)
        self.output_projection = nn.Linear(hidden_size, hidden_size)
        nn.init.kaiming_normal_(self.input_projection.weight, nonlinearity="relu")
        nn.init.zeros_(self.input_projection.bias)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, hidden: Tensor) -> Tensor:
        return hidden + self.output_projection(F.relu_(self.input_projection(F.relu_(hidden))))


class RfDiffusionTorsionHead(nn.Module):
    r"""
    Predict sidechain and backbone torsion angles from the MSA row and node state.
    """

    def __init__(
        self,
        msa_size: int = 256,
        state_size: int = 32,
        hidden_size: int = 128,
    ) -> None:
        super().__init__()
        self.msa_norm = LayerNorm(msa_size)
        self.msa_projection = nn.Linear(msa_size, hidden_size)
        self.state_norm = LayerNorm(state_size)
        self.state_projection = nn.Linear(state_size, hidden_size)
        self.blocks = nn.ModuleList(RfDiffusionTorsionBlock(hidden_size) for _ in range(2))
        self.output_projection = nn.Linear(hidden_size, 20)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.msa_projection = _initialize_lecun_normal(self.msa_projection)
        self.state_projection = _initialize_lecun_normal(self.state_projection)
        self.output_projection = _initialize_lecun_normal(self.output_projection)
        nn.init.zeros_(self.msa_projection.bias)
        nn.init.zeros_(self.state_projection.bias)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, seq: Tensor, state: Tensor) -> Tensor:
        batch_size, length = seq.shape[:2]
        hidden = self.msa_projection(self.msa_norm(seq)) + self.state_projection(self.state_norm(state))
        for block in self.blocks:
            hidden = block(hidden)
        return self.output_projection(F.relu_(hidden)).view(batch_size, length, 10, 2)


class RfDiffusionPairwiseGeometryHead(nn.Module):
    r"""
    Predict the RfDiffusion 6D distance/orientation logits from the pair representation.
    """

    def __init__(self, n_feat: int) -> None:
        super().__init__()
        self.asymmetric_projection = nn.Linear(n_feat, _THETA_BINS + _PHI_BINS)
        self.symmetric_projection = nn.Linear(n_feat, _DISTANCE_BINS + _OMEGA_BINS)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        _zero_init(self.symmetric_projection)
        _zero_init(self.asymmetric_projection)

    def forward(
        self,
        pair: torch.Tensor,
    ) -> _RfDiffusionGeometryLogits:
        theta_logits, phi_logits = self.asymmetric_projection(pair).split((_THETA_BINS, _PHI_BINS), dim=-1)

        symmetric_logits = self.symmetric_projection(pair)
        symmetric_logits = symmetric_logits + symmetric_logits.transpose(1, 2)
        distance_logits, omega_logits = symmetric_logits.split((_DISTANCE_BINS, _OMEGA_BINS), dim=-1)

        return _RfDiffusionGeometryLogits(  # type: ignore[call-arg]
            distance_logits=_channel_first(distance_logits),
            omega_logits=_channel_first(omega_logits),
            theta_logits=_channel_first(theta_logits),
            phi_logits=_channel_first(phi_logits),
            batch_size=[pair.shape[0]],
        )


class RfDiffusionMaskedTokenHead(nn.Module):
    r"""
    Predict per-position amino-acid logits from the MSA representation.
    """

    def __init__(self, n_feat: int, vocab_size: int = 21) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.projection = nn.Linear(n_feat, self.vocab_size)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        _zero_init(self.projection)

    def forward(self, msa: torch.Tensor) -> torch.Tensor:
        batch_size, msa_depth, length = msa.shape[:3]
        logits = self.projection(msa).movedim(-1, 1)
        return logits.reshape(batch_size, self.vocab_size, msa_depth * length)


class RfDiffusionPredictedLddtHead(nn.Module):
    r"""
    Predict binned per-residue pLDDT logits from the node state.
    """

    def __init__(self, n_feat: int, n_bin_lddt: int = 50) -> None:
        super().__init__()
        self.projection = nn.Linear(n_feat, n_bin_lddt)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        _zero_init(self.projection)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.projection(state).movedim(-1, 1)


class RfDiffusionExperimentallyResolvedHead(nn.Module):
    r"""
    Predict the per-residue "experimentally resolved" logit from the MSA row and node state.
    """

    def __init__(self, msa_size: int, state_size: int) -> None:
        super().__init__()
        self.msa_norm = LayerNorm(msa_size)
        self.state_norm = LayerNorm(state_size)
        self.projection = nn.Linear(msa_size + state_size, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        _zero_init(self.projection)

    def forward(self, seq: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        normalized_seq = self.msa_norm(seq)
        normalized_state = self.state_norm(state)
        logits = self.projection(torch.cat((normalized_seq, normalized_state), dim=-1))
        return logits.squeeze(-1)


class RfDiffusionTranslationDiffuser:
    r"""
    Discrete Gaussian translation diffuser for backbone CA positions.
    """

    def __init__(
        self,
        num_timesteps: int,
        beta_initial: float,
        beta_final: float,
        *,
        schedule_type: str = "linear",
    ) -> None:
        self.num_timesteps = integer_index(num_timesteps)
        self.beta_schedule, _, _ = _beta_schedule(
            self.num_timesteps,
            beta_initial,
            beta_final,
            schedule_type,
        )

    def diffuse_translations(
        self,
        coords: Tensor,
        *,
        diffusion_mask: Tensor | None = None,
        var_scale: float = 1.0,
        num_steps: int | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        bb_stack: list[Tensor] = []
        delta_stack: list[Tensor] = []
        current = coords
        steps = self.num_timesteps if num_steps is None else integer_index(num_steps)
        if not 1 <= steps <= self.num_timesteps:
            raise ValueError("num_steps must be in [1, num_timesteps]")
        for t in range(1, steps + 1):
            current, delta = self.apply_kernel(
                current,
                t,
                diffusion_mask=diffusion_mask,
                var_scale=var_scale,
                generator=generator,
            )
            bb_stack.append(current)
            delta_stack.append(delta)
        return torch.stack(bb_stack).transpose(0, 1), torch.stack(delta_stack).transpose(0, 1)

    def apply_kernel(
        self,
        x: Tensor,
        t: int,
        *,
        diffusion_mask: Tensor | None = None,
        var_scale: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        t_idx = t - 1
        ca_coords = x[:, 1]
        beta_t = self.beta_schedule.to(device=x.device, dtype=x.dtype)[t_idx]
        _, delta = ddpm_forward_step(
            ca_coords,
            beta_t,
            variance_scale=var_scale,
            generator=generator,
        )
        if diffusion_mask is not None:
            delta[diffusion_mask.to(device=delta.device).bool()] = 0
        return x + delta[:, None], delta


def _interp1d(x: Tensor, xp: Tensor, fp: Tensor) -> Tensor:
    r"""
    Linearly interpolate increasing one-dimensional samples.

    Query points outside the sample range use the nearest endpoint value.
    """

    index = torch.searchsorted(xp, x, right=True).clamp(1, xp.shape[-1] - 1)
    left = xp[index - 1]
    right = xp[index]
    slope = (fp[index] - fp[index - 1]) / (right - left)
    interpolated = fp[index - 1] + slope * (x - left)
    interpolated = torch.where(x <= xp[0], fp[0], interpolated)
    return torch.where(x >= xp[-1], fp[-1], interpolated)


class RfDiffusionIgso3Diffuser(nn.Module):
    r"""
    Isotropic-Gaussian-on-SO(3) (IGSO3) rotation diffuser and reverse sampler.

    The IGSO(3) lookup table is held in non-persistent buffers. Direct construction
    without a table calculates it once.
    """

    # The score-norm table buffer avoids the ``score_norm`` method name; every other
    # ``calculate_igso3`` key registers under its own name.
    _BUFFER_NAMES: dict[str, str] = {"score_norm": "score_norm_table"}

    def __init__(
        self,
        *,
        num_timesteps: int,
        min_sigma: float,
        max_sigma: float,
        min_b: float,
        max_b: float,
        num_omega: int = 1000,
        num_sigma: int = 500,
        schedule: str = "linear",
        truncation_level: int = 2000,
        igso3_schedule: dict[str, Tensor] | None = None,
    ) -> None:
        super().__init__()
        self.num_timesteps = integer_index(num_timesteps)
        self.schedule = schedule
        self.min_sigma = float(min_sigma)
        self.max_sigma = float(max_sigma)
        self.min_b = float(min_b)
        self.max_b = float(max_b)
        if self.schedule == "linear":
            self.max_sigma = float(self.sigma(torch.tensor(1.0)))
        if self.schedule not in {"linear", "exponential"}:
            raise ValueError(f"unsupported IGSO3 schedule {schedule!r}")
        self.num_omega = integer_index(num_omega)
        self.num_sigma = integer_index(num_sigma)
        self.truncation_level = integer_index(truncation_level)
        self.step_size = 1 / self.num_timesteps
        if igso3_schedule is None:
            igso3_schedule = {
                key: torch.as_tensor(value)
                for key, value in calculate_igso3(
                    num_sigma=self.num_sigma,
                    min_sigma=self.min_sigma,
                    max_sigma=self.max_sigma,
                    num_omega=self.num_omega,
                    num_terms=self.truncation_level,
                ).items()
            }
        for key, value in igso3_schedule.items():
            self.register_buffer(
                self._BUFFER_NAMES.get(key, key),
                torch.as_tensor(value),
                persistent=False,
            )

    def sigma_idx(self, sigma: Tensor | float) -> Tensor:
        sigma_tensor = torch.as_tensor(
            sigma,
            dtype=self.discrete_sigma.dtype,
            device=self.discrete_sigma.device,
        )
        return torch.searchsorted(self.discrete_sigma, sigma_tensor, right=True) - 1

    def t_to_idx(self, t: int | Tensor) -> Tensor:
        sigma = self.sigma(torch.as_tensor(t, dtype=torch.float32) / self.num_timesteps).to(
            dtype=self.discrete_sigma.dtype,
            device=self.discrete_sigma.device,
        )
        return self.sigma_idx(sigma)

    def sigma(self, t: Tensor | float) -> Tensor:
        t_tensor = torch.as_tensor(t, dtype=torch.float32)
        if torch.any(t_tensor < 0) or torch.any(t_tensor > 1):
            raise ValueError(f"invalid IGSO3 time {t}")
        if self.schedule == "exponential":
            sigma = t_tensor * np.log10(self.max_sigma) + (1 - t_tensor) * np.log10(self.min_sigma)
            return 10**sigma
        return self.min_sigma + t_tensor * self.min_b + 0.5 * (t_tensor**2) * (self.max_b - self.min_b)

    def diffusion_coefficient(self, t: float | Tensor) -> Tensor:
        with torch.inference_mode(False), torch.enable_grad():
            t_tensor = torch.as_tensor(t, dtype=torch.float32).requires_grad_(True)
            sigma_sq = self.sigma(t_tensor) ** 2
            grad = torch.autograd.grad(sigma_sq.sum(), t_tensor)[0]
            return torch.sqrt(grad)

    def sample(
        self,
        ts: np.ndarray,
        *,
        n_samples: int = 1,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        if np.any(np.asarray(ts) == 0):
            raise ValueError("IGSO3 sample expects one-indexed timesteps")
        device = self.discrete_omega.device
        all_samples = []
        for t in ts:
            sigma_idx = int(self.t_to_idx(int(t)))
            cdf = self.cdf[sigma_idx]
            rng_device = torch.device("cpu") if generator is None else generator.device
            uniform = torch.rand(n_samples, device=rng_device, generator=generator)
            all_samples.append(
                _interp1d(
                    uniform.to(device=device, dtype=self.discrete_omega.dtype),
                    cdf,
                    self.discrete_omega,
                )
            )
        return torch.stack(all_samples, dim=0)

    def sample_vec(
        self,
        ts: np.ndarray,
        *,
        n_samples: int = 1,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        rng_device = torch.device("cpu") if generator is None else generator.device
        axes = torch.randn((len(ts), n_samples, 3), device=rng_device, generator=generator)
        axes = axes.to(device=self.discrete_omega.device, dtype=self.discrete_omega.dtype)
        axes = axes / torch.linalg.vector_norm(axes, dim=-1, keepdim=True)
        return axes * self.sample(ts, n_samples=n_samples, generator=generator)[..., None]

    def score_norm(self, t: int, omega: Tensor) -> Tensor:
        omega_tensor = torch.as_tensor(
            omega,
            dtype=self.discrete_omega.dtype,
            device=self.discrete_omega.device,
        )
        sigma_idx = int(self.t_to_idx(t))
        return _interp1d(omega_tensor, self.discrete_omega, self.score_norm_table[sigma_idx])

    def diffuse_frames(
        self,
        coords: Tensor | np.ndarray,
        *,
        t_list: np.ndarray | None = None,
        diffusion_mask: np.ndarray | Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        device = self.discrete_omega.device
        coords_tensor = torch.as_tensor(coords, dtype=self.discrete_omega.dtype, device=device)
        t_values = np.arange(self.num_timesteps) + 1 if t_list is None else np.asarray(t_list, dtype=np.int64)
        num_res = coords_tensor.shape[0]
        N = coords_tensor[None, :, 0]
        Ca = coords_tensor[None, :, 1]
        C = coords_tensor[None, :, 2]
        R_true, Ca_true = rigid_from_three_points(N, Ca, C)
        R_true = R_true[0]
        Ca_true = Ca_true[0]
        sampled_rots = self.sample_vec(t_values, n_samples=num_res, generator=generator)
        if diffusion_mask is not None:
            mask = torch.as_tensor(diffusion_mask, dtype=coords_tensor.dtype, device=device)
            sampled_rots = sampled_rots * (1 - mask[None, :, None])
        R_sampled = so3_exp_map(sampled_rots.reshape(-1, 3)).reshape(len(t_values), num_res, 3, 3)
        R_perturbed = torch.einsum("tnij,njk->tnik", R_sampled, R_true)
        perturbed = (
            torch.einsum("tnij,naj->tnai", R_sampled, coords_tensor[:, :3] - Ca_true[:, None]) + Ca_true[None, :, None]
        )
        return perturbed.transpose(0, 1), R_perturbed.transpose(0, 1)

    def reverse_sample_vectorized(
        self,
        rotations_t: Tensor | np.ndarray,
        rotations_0: Tensor | np.ndarray,
        t: int,
        noise_level: float,
        *,
        mask: Tensor | None = None,
        return_perturb: bool = False,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        R_t_t = torch.as_tensor(rotations_t, dtype=torch.float32)
        R_0_t = torch.as_tensor(rotations_0, dtype=torch.float32, device=R_t_t.device)
        R_0t = torch.einsum("...ij,...kj->...ik", R_t_t, R_0_t)
        # ``Rotation.from_matrix(R_0t).as_rotvec()`` in pure torch: SVD-orthonormalise the (float32)
        # relative rotation in float64, then take the rotation logarithm.
        rotvec = so3_log_map(so3_project_to_rotation(R_0t.double())).to(dtype=R_t_t.dtype)
        omega_value = torch.linalg.norm(rotvec, dim=-1)
        score_norm = self.score_norm(t, omega_value)
        denom = omega_value.to(device=score_norm.device, dtype=score_norm.dtype).clamp_min(1e-8)
        scale = score_norm / denom
        score = rotvec * scale.to(device=rotvec.device, dtype=rotvec.dtype)[:, None]
        rot_g = self.diffusion_coefficient(t / self.num_timesteps).to(device=score.device, dtype=score.dtype)
        rng_device = score.device if generator is None else generator.device
        z = torch.randn(score.shape, dtype=score.dtype, device=rng_device, generator=generator).to(score.device)
        z = z * noise_level
        delta_r = (rot_g**2) * self.step_size * score
        perturb_tangent = delta_r + rot_g * math.sqrt(self.step_size) * z
        if mask is not None:
            perturb_tangent = perturb_tangent * (1 - mask.long()).to(device=perturb_tangent.device)[:, None]
        perturb = so3_exp_map(perturb_tangent)
        if return_perturb:
            return perturb
        return torch.einsum("...ij,...jk->...ik", perturb, R_t_t)


class RfDiffusionDiffuser:
    r"""
    RFdiffusion forward pose diffuser coupling translation and rotation noise.
    """

    def __init__(
        self,
        num_timesteps: int,
        beta_initial: float,
        beta_final: float,
        min_sigma: float,
        max_sigma: float,
        min_b: float,
        max_b: float,
        schedule_type: str,
        so3_schedule_type: str,
        so3_type: str,
        crd_scale: float,
        *,
        var_scale: float = 1.0,
        partial_timesteps: int | None = None,
        truncation_level: int = 2000,
        igso3_num_omega: int = 1000,
        igso3_num_sigma: int = 500,
        igso3_schedule: dict[str, Tensor] | None = None,
    ) -> None:
        if so3_type != "igso3":
            raise ValueError(f"unsupported SO(3) diffuser {so3_type!r}")
        self.num_timesteps = integer_index(num_timesteps)
        self.partial_timesteps = None if partial_timesteps is None else integer_index(partial_timesteps)
        if self.partial_timesteps is not None and not 0 < self.partial_timesteps <= self.num_timesteps:
            raise ValueError("partial_timesteps must be in the interval [1, T]")
        self.min_sigma = float(min_sigma)
        self.max_sigma = float(max_sigma)
        self.crd_scale = float(crd_scale)
        self.var_scale = float(var_scale)
        self.so3_diffuser = RfDiffusionIgso3Diffuser(
            num_timesteps=self.num_timesteps,
            min_sigma=self.min_sigma,
            max_sigma=self.max_sigma,
            schedule=so3_schedule_type,
            min_b=min_b,
            max_b=max_b,
            truncation_level=truncation_level,
            num_omega=igso3_num_omega,
            num_sigma=igso3_num_sigma,
            igso3_schedule=igso3_schedule,
        )
        self.eucl_diffuser = RfDiffusionTranslationDiffuser(
            self.num_timesteps,
            beta_initial,
            beta_final,
            schedule_type=schedule_type,
        )

    def diffuse_pose(
        self,
        coords: Tensor,
        atom_mask: Tensor,
        *,
        include_motif_sidechains: bool = True,
        diffusion_mask: Tensor | None = None,
        t_list: np.ndarray | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        if diffusion_mask is None:
            diffusion_mask = torch.zeros(len(coords.squeeze()), dtype=torch.bool, device=coords.device)
        diffusion_mask = diffusion_mask.to(device=coords.device).bool()
        if diffusion_mask.any():
            motif_com = coords[diffusion_mask, 1].mean(dim=0)
            coords = coords - motif_com
        else:
            coords = coords - coords[:, 1].mean(dim=0)
        true_coords = coords
        t_values = (
            np.arange(1, self.num_timesteps + 1, dtype=np.int64)
            if t_list is None
            else np.asarray(t_list, dtype=np.int64)
        )
        if t_values.ndim != 1 or len(t_values) == 0 or np.any(t_values < 1) or np.any(t_values > self.num_timesteps):
            raise ValueError("t_list must contain one-indexed timesteps in the active diffusion schedule")
        max_step = int(t_values.max())
        scaled = coords * self.crd_scale
        _diffused_T, deltas = self.eucl_diffuser.diffuse_translations(
            scaled[:, :3],
            diffusion_mask=diffusion_mask,
            var_scale=self.var_scale,
            num_steps=max_step,
            generator=generator,
        )
        deltas = deltas / self.crd_scale
        frame_coords, _frames = self.so3_diffuser.diffuse_frames(
            scaled[:, :3],
            diffusion_mask=diffusion_mask,
            t_list=t_values,
            generator=generator,
        )
        frame_coords = frame_coords.to(device=coords.device, dtype=coords.dtype) / self.crd_scale
        cum_delta = deltas.cumsum(dim=1)
        requested = torch.as_tensor(t_values - 1, dtype=torch.long, device=cum_delta.device)
        cum_delta = cum_delta.index_select(1, requested)
        diffused_bb = (frame_coords + cum_delta[:, :, None]).transpose(0, 1)
        t_steps, length = diffused_bb.shape[:2]
        diffused_fa = torch.full(
            (t_steps, length, 27, 3),
            float("nan"),
            dtype=coords.dtype,
            device=coords.device,
        )
        diffused_fa[:, :, :3] = diffused_bb
        if include_motif_sidechains:
            preserve = diffusion_mask[:, None] & atom_mask.to(device=coords.device, dtype=torch.bool)
            diffused_fa = torch.where(preserve[None, :, :, None], true_coords[None], diffused_fa)
        return diffused_fa, true_coords


class RfDiffusionSampler:
    r"""
    PyTorch reverse-diffusion loop over one canonical RFdiffusion input.
    """

    def __init__(
        self,
        config: RfDiffusionConfig,
        *,
        denoiser: RfDiffusionDenoiser,
        all_atom_reconstruction: RfDiffusionAllAtomReconstruction,
        igso3_schedule: dict[str, Tensor] | None = None,
        final_step: int = 1,
        partial_steps: int | None = None,
        noise_scale_ca: float = 1.0,
        final_noise_scale_ca: float | None = None,
        ca_noise_schedule: Literal["constant", "linear"] = "constant",
        noise_scale_frame: float = 1.0,
        final_noise_scale_frame: float | None = None,
        frame_noise_schedule: Literal["constant", "linear"] = "constant",
        align_motif: bool = True,
        symmetry_recenter: bool = True,
        symmetry_radius_angstrom: float = 10.0,
        guidance_scale: float = 10.0,
        guidance_decay: str = "constant",
    ) -> None:
        final_step = integer_index(final_step)
        partial_steps = None if partial_steps is None else integer_index(partial_steps)
        if not 1 <= final_step <= config.diffuser.num_timesteps:
            raise ValueError("final_step must be in [1, config.diffuser.num_timesteps]")
        if partial_steps is not None and not 1 <= partial_steps <= config.diffuser.num_timesteps:
            raise ValueError("partial_steps must be in [1, config.diffuser.num_timesteps]")
        if not math.isfinite(guidance_scale) or guidance_scale < 0:
            raise ValueError("guidance_scale must be finite and non-negative")
        if guidance_decay not in {"constant", "linear", "quadratic", "cubic"}:
            raise ValueError("guidance_decay must be constant, linear, quadratic, or cubic")
        self.config = config
        self.denoiser = denoiser
        self.all_atom_reconstruction = all_atom_reconstruction
        self.igso3_schedule = igso3_schedule
        self.final_step = final_step
        self.partial_steps = partial_steps
        self.noise_scale_ca = float(noise_scale_ca)
        self.final_noise_scale_ca = self.noise_scale_ca if final_noise_scale_ca is None else float(final_noise_scale_ca)
        self.ca_noise_schedule = ca_noise_schedule
        self.noise_scale_frame = float(noise_scale_frame)
        self.final_noise_scale_frame = (
            self.noise_scale_frame if final_noise_scale_frame is None else float(final_noise_scale_frame)
        )
        self.frame_noise_schedule = frame_noise_schedule
        if ca_noise_schedule not in ("constant", "linear"):
            raise ValueError(f"unknown CA noise schedule {ca_noise_schedule!r}")
        if frame_noise_schedule not in ("constant", "linear"):
            raise ValueError(f"unknown frame noise schedule {frame_noise_schedule!r}")
        if (
            ca_noise_schedule == "constant"
            and final_noise_scale_ca is not None
            and self.final_noise_scale_ca != self.noise_scale_ca
        ):
            raise ValueError("final_noise_scale_ca can differ from noise_scale_ca only for a linear schedule")
        if (
            frame_noise_schedule == "constant"
            and final_noise_scale_frame is not None
            and self.final_noise_scale_frame != self.noise_scale_frame
        ):
            raise ValueError("final_noise_scale_frame can differ from noise_scale_frame only for a linear schedule")
        self.align_motif = align_motif
        self.symmetry_recenter = symmetry_recenter
        self.symmetry_radius_angstrom = float(symmetry_radius_angstrom)
        self.guidance_scale = float(guidance_scale)
        self.guidance_decay = guidance_decay
        self.igso3_truncation_level = 2000
        self.igso3_num_omega = 1000
        self.igso3_num_sigma = 500
        self.diffuser: RfDiffusionDiffuser | None = None

    def sample(
        self,
        input: RfDiffusionInput,
        *,
        num_sampling_steps: int | None = None,
        final_step: int | None = None,
        generator: torch.Generator | None = None,
        return_trajectory: bool = False,
    ) -> _RfDiffusionSample:
        active_steps = (
            self.config.diffuser.num_timesteps if num_sampling_steps is None else integer_index(num_sampling_steps)
        )
        if active_steps <= 0:
            raise ValueError("num_sampling_steps must be positive")
        stop_t = self.final_step if final_step is None else integer_index(final_step)
        if input.layout.templates and (
            self.config.preprocess.template_single_feature_size != 28
            or self.config.preprocess.template_pair_feature_size != 47
        ):
            raise ValueError("Template evidence requires an RFdiffusion fold-conditioned configuration")
        if any(isinstance(condition, RfDiffusionSecondaryStructure) for condition in input.layout.conditions) and (
            self.config.preprocess.template_single_feature_size != 28
        ):
            raise ValueError("RfDiffusionSecondaryStructure requires an RFdiffusion fold-conditioned configuration")
        mask_sequence = input.fixed_sequence_mask
        mask_structure = input.fixed_structure_mask
        diffusion_mask = mask_structure.squeeze(0)
        diffuser = self._make_diffuser(active_steps)
        length = input.noisy_coordinates.shape[1]
        coords27 = input.noisy_coordinates.new_full((length, 27, 3), float("nan"))
        coords27[:, :14] = input.noisy_coordinates[0]
        atom_mask = torch.isfinite(coords27).all(dim=-1)
        original_sequence = input.residue_type_features[0].argmax(dim=-1)
        if bool(diffusion_mask.any()):
            center = coords27[diffusion_mask, 1].mean(dim=0)
        else:
            observed_calpha = torch.isfinite(coords27[:, 1]).all(dim=-1)
            center = coords27[observed_calpha, 1].mean(dim=0) if bool(observed_calpha.any()) else coords27.new_zeros(3)
        if self.partial_steps is not None:
            if input.layout.structure is None:
                raise ValueError("partial diffusion requires coordinates for every output residue")
            source_index = {site: index for index, site in enumerate(input.layout.structure.sites)}
            for reference in input.layout.output_residues:
                if isinstance(reference, SymbolicResidueSite):
                    raise ValueError("partial diffusion requires coordinates for every output residue")
                component_reference = cast(tuple[str, int], reference)
                for atom_name in ("N", "CA", "C"):
                    source = source_index.get(AtomSite(component_reference[0], component_reference[1], atom_name))
                    if source is None or not bool(input.layout.structure.coordinate_mask[source]):
                        raise ValueError(
                            "partial diffusion requires observed N/CA/C coordinates for every output residue"
                        )
            initial_coords = coords27
        else:
            initial_coords = _initial_coordinates(coords27[None, None]).squeeze(0).squeeze(0)

        trajectory_steps = active_steps
        if self.partial_steps is not None:
            if self.partial_steps > trajectory_steps:
                raise ValueError("partial_steps must not exceed the active diffusion schedule")
            trajectory_steps = self.partial_steps
        start_t = trajectory_steps
        if not 1 <= stop_t <= start_t:
            raise ValueError("final_step must be between 1 and the starting timestep")
        residue_type_features = original_sequence.clone()
        residue_type_features[~mask_sequence.squeeze(0)] = RFDIFFUSION_MASK_TOKEN_ID
        residue_type_features = _one_hot_sequence(
            residue_type_features,
            dtype=input.residue_type_features.dtype,
        )
        diffused, _clean = diffuser.diffuse_pose(
            initial_coords,
            atom_mask,
            diffusion_mask=diffusion_mask,
            t_list=np.arange(1, start_t + 1),
            include_motif_sidechains=self.config.preprocess.motif_sidechain_input,
            generator=generator,
        )
        noisy_coords = diffused[-1, :, :14]
        symmetry_conditions = tuple(
            condition for condition in input.layout.conditions if isinstance(condition, RfDiffusionPointGroupSymmetry)
        )
        symmetry = None if not symmetry_conditions else RfDiffusionSymmetry(symmetry_conditions[0], input.layout)
        if symmetry is not None:
            noisy_coords, residue_type_features = symmetry.apply(
                noisy_coords,
                residue_type_features,
                recenter=self.symmetry_recenter,
                radius_angstrom=self.symmetry_radius_angstrom,
            )
        guidance = RfDiffusionGuidance(input.layout)
        sampler_step = self._make_sampler_step(
            length,
            diffuser=diffuser,
            guidance=guidance if guidance else None,
        )

        if return_trajectory:
            timesteps: list[int] = []
            plddt_trajectory: list[Tensor] = []
            denoised_trajectory: list[Tensor] = []
            next_coordinates_trajectory: list[Tensor] = []
        previous_prediction: Tensor | None = None
        residue_index = input.residue_index
        if symmetry is not None:
            residue_index = symmetry.chain_broken_residue_index(residue_index)
        secondary_structure = input.secondary_structure[0]
        block_adjacency = input.block_adjacency[0]
        hotspot_mask = input.hotspot_mask[0]
        cyclic_mask = input.cyclic_mask[0]
        for timestep in range(start_t, stop_t - 1, -1):
            (
                latent_msa_features,
                full_msa_features,
                denoiser_residue_type_features,
                noisy_coordinates_input,
                template_single_features,
                template_pair_features,
                template_coordinates,
                template_torsion_features,
            ) = self._preprocess(
                residue_type_features,
                noisy_coords,
                timestep,
                fixed_structure_mask=mask_structure,
                diffuser=diffuser,
                previous_prediction=previous_prediction,
                secondary_structure=secondary_structure,
                block_adjacency=block_adjacency,
                hotspot_mask=hotspot_mask,
            )
            output = self.denoiser(
                latent_msa_features,
                full_msa_features,
                denoiser_residue_type_features,
                noisy_coordinates_input,
                residue_index,
                template_single_features=template_single_features,
                template_pair_features=template_pair_features,
                template_coordinates=template_coordinates,
                template_torsion_features=template_torsion_features,
                previous_msa=None,
                previous_pair=None,
                previous_state=None,
                fixed_structure_mask=diffusion_mask,
                cyclic_mask=cyclic_mask,
            )
            predicted_backbone = output.backbone_coordinates
            torsion_angles = output.torsion_angles
            plddt = output.plddt_fraction
            sequence_tokens = denoiser_residue_type_features.argmax(dim=-1)
            _frames, predicted_all_atom_coordinates = self.all_atom_reconstruction(
                sequence_tokens, predicted_backbone, torsion_angles
            )
            predicted_coordinates = predicted_all_atom_coordinates.squeeze(0)[:, :14]
            if symmetry is not None:
                symmetric_coords, _symmetric_residue_type_features = symmetry.apply(
                    predicted_coordinates,
                    residue_type_features,
                    recenter=self.symmetry_recenter,
                    radius_angstrom=self.symmetry_radius_angstrom,
                )
                previous_prediction = symmetric_coords[:, :3].detach()
                _frames, predicted_all_atom_coordinates = self.all_atom_reconstruction(
                    sequence_tokens,
                    previous_prediction[None],
                    torsion_angles,
                )
                predicted_coordinates = predicted_all_atom_coordinates.squeeze(0)[:, :14]
            else:
                previous_prediction = predicted_backbone.squeeze(0).detach()
            denoised_coordinates = predicted_coordinates
            if timestep > stop_t:
                noisy_coords, denoised_coordinates = sampler_step.get_next_pose(
                    noisy_coords=noisy_coords,
                    pred_coords=predicted_coordinates,
                    t=timestep,
                    diffusion_mask=diffusion_mask,
                    align_motif=self.align_motif,
                    include_motif_sidechains=self.config.preprocess.motif_sidechain_input,
                    generator=generator,
                )
            else:
                noisy_coords = predicted_coordinates
            if symmetry is not None:
                noisy_coords, residue_type_features = symmetry.apply(
                    noisy_coords,
                    residue_type_features,
                    recenter=self.symmetry_recenter,
                    radius_angstrom=self.symmetry_radius_angstrom,
                )
            final_timestep = timestep
            final_plddt = plddt.squeeze(0)
            final_denoised_coordinates = denoised_coordinates
            final_next_coordinates = noisy_coords
            if return_trajectory:
                timesteps.append(timestep)
                plddt_trajectory.append(final_plddt.detach().cpu())
                denoised_trajectory.append(final_denoised_coordinates.detach().cpu())
                next_coordinates_trajectory.append(final_next_coordinates.detach().cpu())
        if return_trajectory:
            center_cpu = center.detach().cpu()
            timestep_values = torch.tensor(timesteps, dtype=torch.int64)
            plddt_values = torch.stack(plddt_trajectory)
            denoised_values = torch.stack(denoised_trajectory) + center_cpu
            next_coordinates_values = torch.stack(next_coordinates_trajectory) + center_cpu
        else:
            timestep_values = noisy_coords.new_tensor((final_timestep,), dtype=torch.int64)
            plddt_values = final_plddt[None]
            denoised_values = (final_denoised_coordinates + center)[None]
            next_coordinates_values = (final_next_coordinates + center)[None]
        return _RfDiffusionSample(
            coordinates=final_denoised_coordinates + center,
            sequence_tokens=residue_type_features.argmax(dim=-1),
            timesteps=timestep_values,
            plddt_fraction=plddt_values,
            predicted_clean_coordinates=denoised_values,
            next_step_coordinates=next_coordinates_values,
        )

    def _preprocess(
        self,
        residue_type_features: Tensor,
        template_coordinates: Tensor,
        timestep: int,
        *,
        fixed_structure_mask: Tensor,
        diffuser: RfDiffusionDiffuser,
        previous_prediction: Tensor | None,
        secondary_structure: Tensor,
        block_adjacency: Tensor,
        hotspot_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        length = residue_type_features.shape[0]
        latent_msa_features = residue_type_features.new_zeros((1, 1, length, 48))
        latent_msa_features[:, :, :, :22] = residue_type_features[None, None]
        latent_msa_features[:, :, :, 22:44] = residue_type_features[None, None]
        latent_msa_features[:, :, 0, 46] = 1.0
        latent_msa_features[:, :, -1, 47] = 1.0
        full_msa_features = residue_type_features.new_zeros((1, 1, length, 25))
        full_msa_features[:, :, :, :22] = residue_type_features[None, None]
        full_msa_features[:, :, 0, 23] = 1.0
        full_msa_features[:, :, -1, 24] = 1.0
        template_single_features = residue_type_features.new_zeros((1, 1, length, 21))
        visible_residue_type_features = residue_type_features.clone()
        mask_token_positions = visible_residue_type_features[:, RFDIFFUSION_MASK_TOKEN_ID] == 1
        visible_residue_type_features[mask_token_positions, 20] = 1
        visible_residue_type_features[mask_token_positions, RFDIFFUSION_MASK_TOKEN_ID] = 0
        template_single_features[:, :, :, :21] = visible_residue_type_features[None, None, :, :21]
        time_feature = residue_type_features.new_zeros((length,))
        time_feature[fixed_structure_mask.squeeze(0)] = 1
        time_feature[~fixed_structure_mask.squeeze(0)] = 1 - timestep / diffuser.num_timesteps
        template_single_features = torch.cat((template_single_features, time_feature[None, None, :, None]), dim=-1)

        working_coordinates = template_coordinates.clone()
        if self.config.preprocess.sidechain_input:
            working_coordinates[mask_token_positions, 3:] = float("nan")
        else:
            working_coordinates[~fixed_structure_mask.squeeze(0), 3:] = float("nan")
        template_coordinates = working_coordinates[None, None]
        template_coordinates = torch.cat(
            (
                template_coordinates,
                template_coordinates.new_full((1, 1, length, 13, 3), float("nan")),
            ),
            dim=3,
        )
        noisy_coordinates_input = template_coordinates.squeeze(0)
        template_pair_features = _template_pair_features(template_coordinates)
        template_sequence = template_single_features[..., :-1].argmax(dim=-1).reshape(1, length)
        template_torsion_features = _template_torsion_features(template_coordinates, template_sequence)
        if self.config.preprocess.structure_self_conditioning:
            if previous_prediction is not None and (
                timestep < diffuser.num_timesteps and timestep != self.partial_steps
            ):
                previous_coordinates = previous_prediction[:, :3][None, None]
                zeros = previous_coordinates.new_zeros((1, 1, length, 24, 3))
                template_coordinates = torch.cat((previous_coordinates, zeros), dim=-2)
                template_pair_features[..., :44] = _template_pair_features(template_coordinates)
            else:
                template_pair_features[..., :44] = 0
                template_coordinates = torch.zeros_like(template_coordinates)

        if self.config.preprocess.template_single_feature_size >= 24:
            hotspot_feature = hotspot_mask.to(dtype=residue_type_features.dtype)
            template_single_features = torch.cat(
                (
                    template_single_features,
                    torch.zeros_like(template_single_features[..., :1]),
                    hotspot_feature[None, None, :, None],
                ),
                dim=-1,
            )

        if self.config.preprocess.template_single_feature_size == 28:
            template_single_features = torch.cat(
                (
                    template_single_features,
                    secondary_structure[None, None].to(dtype=residue_type_features.dtype),
                ),
                dim=-1,
            )
        if self.config.preprocess.template_pair_feature_size == 47:
            template_pair_features = torch.cat(
                (
                    template_pair_features,
                    block_adjacency[None, None].to(dtype=template_pair_features.dtype),
                ),
                dim=-1,
            )

        return (
            latent_msa_features,
            full_msa_features,
            residue_type_features[None],
            noisy_coordinates_input,
            template_single_features,
            template_pair_features,
            template_coordinates,
            template_torsion_features,
        )

    def _make_diffuser(self, num_timesteps: int) -> RfDiffusionDiffuser:
        if self.diffuser is not None and self.diffuser.num_timesteps == num_timesteps:
            return self.diffuser
        config = self.config.diffuser
        diffuser = RfDiffusionDiffuser(
            num_timesteps=num_timesteps,
            beta_initial=config.beta_initial,
            beta_final=config.beta_final,
            min_sigma=config.min_rotation_sigma,
            max_sigma=config.max_rotation_sigma,
            min_b=config.min_rotation_beta,
            max_b=config.max_rotation_beta,
            schedule_type=config.schedule_type,
            so3_schedule_type=config.rotation_schedule_type,
            so3_type=config.rotation_diffuser,
            crd_scale=config.coordinate_scale,
            partial_timesteps=self.partial_steps,
            truncation_level=self.igso3_truncation_level,
            igso3_num_omega=self.igso3_num_omega,
            igso3_num_sigma=self.igso3_num_sigma,
            igso3_schedule=self.igso3_schedule,
        )
        self.diffuser = diffuser
        return diffuser

    def _make_sampler_step(
        self,
        length: int,
        *,
        diffuser: RfDiffusionDiffuser,
        guidance: RfDiffusionGuidance | None,
    ) -> RfDiffusionSamplerStep:
        config = self.config.diffuser
        return RfDiffusionSamplerStep(
            num_timesteps=diffuser.num_timesteps,
            truncation_level=length,
            diffuser=diffuser,
            beta_initial=config.beta_initial,
            beta_final=config.beta_final,
            schedule_type=config.schedule_type,
            so3_type=config.rotation_diffuser,
            noise_scale_ca=self.noise_scale_ca,
            final_noise_scale_ca=self.final_noise_scale_ca,
            ca_noise_schedule=self.ca_noise_schedule,
            noise_scale_frame=self.noise_scale_frame,
            final_noise_scale_frame=self.final_noise_scale_frame,
            frame_noise_schedule=self.frame_noise_schedule,
            crd_scale=config.coordinate_scale,
            partial_timesteps=self.partial_steps,
            guidance=guidance,
            guidance_scale=self.guidance_scale,
            guidance_decay=self.guidance_decay,
        )


class RfDiffusionSamplerStep:
    r"""
    Reverse RFdiffusion pose updater stepping ``x_t`` toward ``x_{t-1}``.
    """

    def __init__(
        self,
        num_timesteps: int,
        truncation_level: int,
        diffuser: RfDiffusionDiffuser,
        *,
        beta_initial: float = 0.001,
        beta_final: float = 0.1,
        schedule_type: str = "linear",
        so3_type: str = "igso3",
        noise_scale_ca: float = 1.0,
        final_noise_scale_ca: float | None = None,
        ca_noise_schedule: Literal["constant", "linear"] = "constant",
        noise_scale_frame: float = 0.5,
        final_noise_scale_frame: float | None = None,
        frame_noise_schedule: Literal["constant", "linear"] = "constant",
        crd_scale: float = 1 / 15,
        partial_timesteps: int | None = None,
        guidance: RfDiffusionGuidance | None = None,
        guidance_scale: float = 10.0,
        guidance_decay: str = "constant",
    ) -> None:
        self.num_timesteps = integer_index(num_timesteps)
        self.truncation_level = integer_index(truncation_level)
        self.partial_timesteps = None if partial_timesteps is None else integer_index(partial_timesteps)
        if self.partial_timesteps is not None and not 0 < self.partial_timesteps <= self.num_timesteps:
            raise ValueError("partial_timesteps must be in the interval [1, T]")
        self.diffuser = diffuser
        self.beta_initial = float(beta_initial)
        self.beta_final = float(beta_final)
        self.schedule_type = schedule_type
        self.so3_type = so3_type
        self.crd_scale = float(crd_scale)
        self.noise_scale_ca = float(noise_scale_ca)
        self.final_noise_scale_ca = self.noise_scale_ca if final_noise_scale_ca is None else float(final_noise_scale_ca)
        self.ca_noise_schedule = ca_noise_schedule
        self.noise_scale_frame = float(noise_scale_frame)
        self.final_noise_scale_frame = (
            self.noise_scale_frame if final_noise_scale_frame is None else float(final_noise_scale_frame)
        )
        self.frame_noise_schedule = frame_noise_schedule
        if (
            ca_noise_schedule == "constant"
            and final_noise_scale_ca is not None
            and self.final_noise_scale_ca != self.noise_scale_ca
        ):
            raise ValueError("final_noise_scale_ca can differ from noise_scale_ca only for a linear schedule")
        if (
            frame_noise_schedule == "constant"
            and final_noise_scale_frame is not None
            and self.final_noise_scale_frame != self.noise_scale_frame
        ):
            raise ValueError("final_noise_scale_frame can differ from noise_scale_frame only for a linear schedule")
        self.guidance = guidance
        self.guidance_scale = guidance_scale
        self.guidance_decay = guidance_decay
        self.schedule, _, self.alphabar_schedule = _beta_schedule(
            self.num_timesteps,
            self.beta_initial,
            self.beta_final,
            self.schedule_type,
        )

    @staticmethod
    def align_prediction_to_noisy_motif(pred_coords: Tensor, final_coords: Tensor, diffusion_mask: Tensor) -> Tensor:
        output_dtype = pred_coords.dtype
        atom_mask = ~torch.isnan(pred_coords)
        working_coords = pred_coords.float()
        fixed_structure_mask = diffusion_mask.to(device=pred_coords.device, dtype=torch.bool)
        pred_coords_motif = working_coords[fixed_structure_mask, :3].reshape(-1, 3)
        noisy_coords_motif = final_coords.to(device=pred_coords.device, dtype=torch.float32)[
            fixed_structure_mask, :3
        ].reshape(-1, 3)
        if pred_coords_motif.numel() == 0:
            return pred_coords
        pred_coords_mean = pred_coords_motif.mean(0)
        noisy_coords_mean = noisy_coords_motif.mean(0)
        A = noisy_coords_motif - noisy_coords_mean
        B = pred_coords_motif - pred_coords_mean
        cov = A.T @ B
        U, _S, Vt = torch.linalg.svd(cov)
        handed = torch.eye(3, dtype=torch.float32, device=pred_coords.device)
        handed[-1, -1] = torch.sign(torch.linalg.det(Vt.T @ U.T))
        rotation = Vt.T @ handed @ U.T
        aligned = (torch.nan_to_num(working_coords).reshape(-1, 3) - pred_coords_mean) @ rotation + noisy_coords_mean
        aligned = torch.where(atom_mask, aligned.reshape_as(working_coords), torch.nan)
        return aligned.to(dtype=output_dtype)

    def get_next_pose(
        self,
        noisy_coords: Tensor,
        pred_coords: Tensor,
        t: int,
        diffusion_mask: Tensor,
        *,
        align_motif: bool = True,
        include_motif_sidechains: bool = True,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        diffusion_mask = diffusion_mask.to(device=noisy_coords.device).bool()
        if align_motif and diffusion_mask.any():
            pred_coords = self.align_prediction_to_noisy_motif(pred_coords, noisy_coords, diffusion_mask)
        pred_coords = pred_coords.to(device=noisy_coords.device, dtype=noisy_coords.dtype)
        noise_scale_ca = _noise_scale(
            t,
            self.num_timesteps,
            self.noise_scale_ca,
            self.final_noise_scale_ca,
            self.ca_noise_schedule,
        )
        _out, ca_deltas = _next_alpha_carbon(
            noisy_coords,
            pred_coords,
            t,
            diffusion_mask,
            crd_scale=self.crd_scale,
            beta_schedule=self.schedule,
            alphabar_schedule=self.alphabar_schedule,
            noise_scale=noise_scale_ca,
            generator=generator,
        )
        noise_scale_frame = _noise_scale(
            t,
            self.num_timesteps,
            self.noise_scale_frame,
            self.final_noise_scale_frame,
            self.frame_noise_schedule,
        )
        frames_next = _next_frames(
            noisy_coords,
            pred_coords,
            t,
            diffuser=self.diffuser,
            so3_type=self.so3_type,
            diffusion_mask=diffusion_mask,
            noise_scale=noise_scale_frame,
            generator=generator,
        )
        if self.guidance is not None and self.guidance_scale:
            with torch.inference_mode(False):
                guided_coords = noisy_coords.detach().clone().requires_grad_(True)
                potential = self.guidance(guided_coords)
                gradient = torch.autograd.grad(potential, guided_coords)[0][:, 1]
            gradient = gradient.masked_fill(diffusion_mask[:, None], 0)
            fraction = t / self.num_timesteps
            decay = {
                "constant": 1.0,
                "linear": fraction,
                "quadratic": fraction**2,
                "cubic": fraction**3,
            }[self.guidance_decay]
            ca_deltas = ca_deltas + self.guidance_scale * decay * gradient
        frames_next = frames_next + ca_deltas[:, None]
        fullatom_next = torch.full_like(noisy_coords, float("nan")).unsqueeze(0)
        fullatom_next[:, :, :3] = frames_next[None]
        if include_motif_sidechains:
            fullatom_next[:, diffusion_mask, :14] = noisy_coords[None, diffusion_mask]
        return fullatom_next.squeeze(0)[:, :14], pred_coords


class RfDiffusionSymmetry:
    r"""
    Project declared RfDiffusionPointGroupSymmetry units onto one exact rotation group.
    """

    def __init__(self, condition: RfDiffusionPointGroupSymmetry, layout: RfDiffusionLayout) -> None:
        row_by_reference, symbolic_rows = _index_output_residues(layout.output_residues)
        units = []
        for unit in condition.units:
            rows: list[int] = []
            for target in unit:
                if isinstance(target, str):
                    rows.extend(
                        row
                        for row, reference in enumerate(layout.output_residues)
                        if isinstance(reference, SymbolicResidueSite) and reference.placement_id == target
                    )
                else:
                    rows.append(
                        _output_residue_row(
                            cast(tuple[str, int] | SymbolicResidueSite, target),
                            row_by_reference,
                            symbolic_rows,
                        )
                    )
            units.append(tuple(rows))
        lengths = {len(unit) for unit in units}
        if len(lengths) != 1:
            raise ValueError("RfDiffusionPointGroupSymmetry units must resolve to equal concrete lengths")
        if len({row for unit in units for row in unit}) != sum(map(len, units)):
            raise ValueError("RfDiffusionPointGroupSymmetry units must not overlap")
        if {row for unit in units for row in unit} != set(range(len(layout.output_residues))):
            raise ValueError("RFdiffusion RfDiffusionPointGroupSymmetry must cover every output residue exactly once")
        self.units = tuple(units)
        self.kind = condition.kind
        self.rotations = point_group_rotations(condition.kind, condition.order)

    def apply(
        self,
        coords: Tensor,
        residue_type_features: Tensor,
        *,
        recenter: bool,
        radius_angstrom: float,
    ) -> tuple[Tensor, Tensor]:
        projected_coords = coords.clone()
        projected_residue_type_features = residue_type_features.clone()
        base_index = torch.tensor(self.units[0], device=coords.device)
        base_coords = coords[base_index]
        base_residue_type_features = residue_type_features[base_index]
        for rows, rotation in zip(self.units, self.rotations, strict=True):
            index = torch.tensor(rows, device=coords.device)
            matrix = rotation.to(device=coords.device, dtype=coords.dtype)
            rotated = torch.einsum("raj,kj->rak", base_coords, matrix)
            if self.kind == "octahedral" and recenter:
                offset = coords.new_tensor((radius_angstrom, 0.0, 0.0))
                rotated = rotated - rotated[:, 1].mean(dim=0) + torch.einsum("j,kj->k", offset, matrix)
            projected_coords[index] = rotated
            projected_residue_type_features[index] = base_residue_type_features
        return projected_coords, projected_residue_type_features

    def chain_broken_residue_index(self, residue_index: Tensor) -> Tensor:
        result = residue_index.clone()
        offset = result.shape[-1]
        for unit_index, rows in enumerate(self.units):
            result[..., torch.tensor(rows, device=result.device)] += offset * (unit_index + 1)
        return result


class RfDiffusionGuidance:
    r"""
    Differentiate canonical soft geometric Conditions during reverse diffusion.
    """

    def __init__(
        self,
        layout: RfDiffusionLayout,
    ) -> None:
        self.layout = layout
        self.row_by_reference, self.symbolic_rows = _index_output_residues(layout.output_residues)
        self.structure_index = (
            {} if layout.structure is None else {site: index for index, site in enumerate(layout.structure.sites)}
        )
        reference_sites = {
            site
            for condition in layout.conditions
            if isinstance(condition, RfDiffusionReferenceCoordinates)
            for site in condition.sites
        }
        anchor_rows: list[int] = []
        anchor_atoms: list[int] = []
        anchor_coordinates: list[np.ndarray] = []
        if layout.structure is not None:
            for row, reference in enumerate(layout.output_residues):
                if not isinstance(reference, tuple):
                    continue
                sites = tuple(AtomSite(reference[0], reference[1], atom) for atom in ("N", "CA", "C"))
                if not all(site in reference_sites for site in sites):
                    continue
                for atom_index, site in enumerate(sites):
                    source = self.structure_index[site]
                    anchor_rows.append(row)
                    anchor_atoms.append(atom_index)
                    anchor_coordinates.append(layout.structure.coordinates[source])
                break
        self.anchor_rows = tuple(anchor_rows)
        self.anchor_atoms = tuple(anchor_atoms)
        self.anchor_coordinates = np.asarray(anchor_coordinates, dtype=np.float32)
        guidance_conditions: list[
            RfDiffusionCoordination
            | RfDiffusionDistance
            | RfDiffusionRadiusOfGyration
            | RfDiffusionSubstrateInteraction
            | SymbolicBond
        ] = []
        for condition in layout.conditions:
            if isinstance(
                condition,
                (
                    RfDiffusionCoordination,
                    RfDiffusionDistance,
                    RfDiffusionRadiusOfGyration,
                    RfDiffusionSubstrateInteraction,
                ),
            ):
                guidance_conditions.append(
                    cast(
                        RfDiffusionCoordination
                        | RfDiffusionDistance
                        | RfDiffusionRadiusOfGyration
                        | RfDiffusionSubstrateInteraction,
                        condition,
                    )
                )
            elif isinstance(condition, SymbolicBond):
                bond = cast(SymbolicBond, condition)
                if {bond.site_a.atom_id, bond.site_b.atom_id} == {"N", "C"}:
                    guidance_conditions.append(bond)
        self.conditions = tuple(guidance_conditions)
        if (
            any(isinstance(condition, RfDiffusionSubstrateInteraction) for condition in self.conditions)
            and not self.anchor_rows
        ):
            raise ValueError(
                "RfDiffusionSubstrateInteraction with external atoms requires fixed "
                "RfDiffusionReferenceCoordinates to define the substrate frame"
            )

    def __bool__(self) -> bool:
        return bool(self.conditions)

    def __call__(self, coords: Tensor) -> Tensor:
        potential = coords.new_zeros(())
        for condition in self.conditions:
            if isinstance(condition, RfDiffusionCoordination):
                coordination_condition = cast(RfDiffusionCoordination, condition)
                weight = 1.0 if coordination_condition.weight is None else coordination_condition.weight
                left = self._coordinates(coordination_condition.side_a, coords)
                right = self._coordinates(coordination_condition.side_b, coords)
                coordination = smooth_coordination_potential(
                    torch.cdist(left, right),
                    coordination_condition.center_angstrom,
                    coordination_condition.scale_angstrom,
                )
                if coordination_condition.direction == "maximize":
                    direction = 1
                elif coordination_condition.direction == "minimize":
                    direction = -1
                else:
                    raise ValueError(f"unknown RfDiffusionCoordination direction: {coordination_condition.direction!r}")
                potential = potential + direction * weight * coordination
            elif isinstance(condition, RfDiffusionDistance):
                distance_condition = cast(RfDiffusionDistance, condition)
                weight = 1.0 if distance_condition.weight is None else distance_condition.weight
                distance = torch.linalg.vector_norm(
                    self._atom(distance_condition.site_a, coords) - self._atom(distance_condition.site_b, coords)
                )
                lower = max(
                    0.0,
                    distance_condition.lower_angstrom - distance_condition.tolerance_angstrom,
                )
                upper = distance_condition.upper_angstrom + distance_condition.tolerance_angstrom
                violation = interval_distance_penalty(distance, lower, upper)
                potential = potential - weight * violation
            elif isinstance(condition, RfDiffusionRadiusOfGyration):
                radius_condition = cast(RfDiffusionRadiusOfGyration, condition)
                weight = 1.0 if radius_condition.weight is None else radius_condition.weight
                rows = self._residue_rows(radius_condition.targets)
                calpha = coords[torch.tensor(rows, device=coords.device), 1]
                radius = upper_bounded_radius_of_gyration(calpha, radius_condition.upper_angstrom)
                potential = potential - weight * radius
            elif isinstance(condition, RfDiffusionSubstrateInteraction):
                substrate_condition = cast(RfDiffusionSubstrateInteraction, condition)
                weight = 1.0 if substrate_condition.weight is None else substrate_condition.weight
                rows = self._residue_rows(substrate_condition.residues)
                residue_coords = coords[torch.tensor(rows, device=coords.device), 1]
                substrate_coords = self._coordinates(substrate_condition.substrate_atoms, coords)
                interaction = substrate_interaction_potential(
                    residue_coords,
                    substrate_coords,
                    center=substrate_condition.center_angstrom,
                    scale=substrate_condition.scale_angstrom,
                    attraction_strength=substrate_condition.attraction_strength,
                    repulsion_slope=substrate_condition.repulsion_slope,
                    repulsion_cutoff=substrate_condition.repulsion_cutoff_angstrom,
                    nearest_only_repulsion=substrate_condition.nearest_only_repulsion,
                )
                potential = potential + weight * interaction
            else:
                bond = cast(SymbolicBond, condition)
                distance = torch.linalg.vector_norm(self._atom(bond.site_a, coords) - self._atom(bond.site_b, coords))
                potential = potential + harmonic_distance_potential(distance, 1.33)
        return potential

    def _residue_rows(self, references: Iterable[object]) -> tuple[int, ...]:
        rows: list[int] = []
        for reference in references:
            if isinstance(reference, str):
                rows.extend(
                    row
                    for row, output in enumerate(self.layout.output_residues)
                    if isinstance(output, SymbolicResidueSite) and output.placement_id == reference
                )
            elif isinstance(reference, (tuple, SymbolicResidueSite)):
                rows.append(
                    _output_residue_row(
                        cast(tuple[str, int] | SymbolicResidueSite, reference),
                        self.row_by_reference,
                        self.symbolic_rows,
                    )
                )
            elif isinstance(reference, SymbolicAtomSite):
                symbolic_site = cast(SymbolicAtomSite, reference)
                rows.append(
                    _output_residue_row(
                        symbolic_site.residue,
                        self.row_by_reference,
                        self.symbolic_rows,
                    )
                )
            elif isinstance(reference, AtomSite):
                atom_site = cast(AtomSite, reference)
                system = cast(MolecularSystem, self.layout.system)
                entity = system.entities[system.instances[atom_site.instance_id]]
                if not isinstance(entity, Molecule):
                    rows.append(self.row_by_reference[(atom_site.instance_id, atom_site.component_index)])
        return tuple(dict.fromkeys(rows))

    def _coordinates(self, references: Iterable[object], coords: Tensor) -> Tensor:
        values: list[Tensor] = []
        for reference in references:
            if isinstance(reference, (str, tuple, SymbolicResidueSite)):
                rows = self._residue_rows((reference,))
                values.extend(self._residue_coordinate(row, coords) for row in rows)
            else:
                values.append(self._atom(cast(AtomSite | SymbolicAtomSite, reference), coords))
        return torch.stack(values)

    def _residue_coordinate(self, row: int, coords: Tensor) -> Tensor:
        reference = self.layout.output_residues[row]
        if isinstance(reference, tuple) and self.layout.system is not None:
            entity = self.layout.system.entities[self.layout.system.instances[reference[0]]]
            if entity.components[reference[1]].component_id == "GLY":
                return coords[row, 1]
        return _beta_carbon(coords[row, 0], coords[row, 1], coords[row, 2])

    def _atom(self, reference: AtomSite | SymbolicAtomSite, coords: Tensor) -> Tensor:
        if isinstance(reference, SymbolicAtomSite):
            symbolic_site = cast(SymbolicAtomSite, reference)
            symbolic_row = _output_residue_row(
                symbolic_site.residue,
                self.row_by_reference,
                self.symbolic_rows,
            )
            symbolic_atom_names = ("N", "CA", "C")
            return coords[symbolic_row, symbolic_atom_names.index(symbolic_site.atom_id)]
        atom_site = cast(AtomSite, reference)
        row = self.row_by_reference.get((atom_site.instance_id, atom_site.component_index))
        if row is not None:
            system = cast(MolecularSystem, self.layout.system)
            entity = system.entities[system.instances[atom_site.instance_id]]
            component_id = entity.components[atom_site.component_index].component_id
            protein_atom_names = RFDIFFUSION_ATOM14_NAMES[rfdiffusion_residue_token_id(component_id)]
            return coords[row, protein_atom_names.index(atom_site.atom_id)]
        source_index = self.structure_index[atom_site]
        value = torch.tensor(
            cast(Structure, self.layout.structure).coordinates[source_index],
            device=coords.device,
            dtype=coords.dtype,
        )
        if not self.anchor_rows:
            raise ValueError(
                "guidance to atoms outside the RFdiffusion output requires fixed RfDiffusionReferenceCoordinates "
                "to define their coordinate frame"
            )
        source_anchors = torch.as_tensor(
            self.anchor_coordinates,
            device=coords.device,
            dtype=coords.dtype,
        )
        target_anchors = coords[
            torch.tensor(self.anchor_rows, device=coords.device),
            torch.tensor(self.anchor_atoms, device=coords.device),
        ]
        source_rotation, source_calpha = rigid_from_three_points(
            source_anchors[0],
            source_anchors[1],
            source_anchors[2],
        )
        target_rotation, target_calpha = rigid_from_three_points(
            target_anchors[0],
            target_anchors[1],
            target_anchors[2],
        )
        local = (value - source_calpha) @ source_rotation
        return local @ target_rotation.transpose(-1, -2) + target_calpha


def rfdiffusion_terminal_cyclization_instance(
    condition: SymbolicBond,
    output_residues: Sequence[ComponentReference | SymbolicResidueSite],
    output_instances: tuple[str, ...],
) -> str:
    r"""
    Resolve an intrachain terminal peptide bond to its output instance.

    Args:
        condition: Requested symbolic bond between the terminal nitrogen and carbon.
        output_residues: Concrete and symbolic output residues in row order.
        output_instances: Output instance identifier for each residue row.

    Returns:
        The instance identifier of the cyclic protein chain.

    Raises:
        ValueError: If the bond is not an intrachain terminal peptide cyclization.
    """

    row_by_reference, symbolic_rows = _index_output_residues(output_residues)
    endpoints: list[tuple[int, str]] = []
    for site in (condition.site_a, condition.site_b):
        if isinstance(site, SymbolicAtomSite):
            reference: ComponentReference | SymbolicResidueSite = site.residue
            atom_name = site.atom_id
        elif isinstance(site, AtomSite):
            # component_index is always an int; a non-residue Molecule endpoint is
            # rejected below by row_by_reference, which only holds protein output residues.
            reference = (site.instance_id, site.component_index)
            atom_name = site.atom_id
        else:
            raise ValueError("RFdiffusion supports only terminal peptide cyclization")
        try:
            row = _output_residue_row(reference, row_by_reference, symbolic_rows)
        except (IndexError, KeyError):
            raise ValueError("SymbolicBond endpoints must belong to RFdiffusion output residues")
        endpoints.append((row, atom_name))
    left, right = endpoints
    if output_instances[left[0]] != output_instances[right[0]]:
        raise ValueError("RFdiffusion supports only intrachain terminal peptide cyclization")
    instance_id = output_instances[left[0]]
    rows = [row for row, value in enumerate(output_instances) if value == instance_id]
    if set(endpoints) != {(rows[0], "N"), (rows[-1], "C")}:
        raise ValueError("RFdiffusion cyclic mode requires the terminal N-to-C peptide link")
    return instance_id


def rfdiffusion_secondary_structure(calpha: Tensor) -> Tensor:
    r"""
    Assign the helper's P-SEA helix/strand/loop labels.
    """

    length = calpha.shape[0]
    labels = torch.full((length,), 2, dtype=torch.int64, device=calpha.device)
    if length < 5:
        return labels

    d2 = calpha.new_full((length,), float("nan"))
    d3 = calpha.new_full((length,), float("nan"))
    d4 = calpha.new_full((length,), float("nan"))
    radius_angle = calpha.new_full((length,), float("nan"))
    torsion = calpha.new_full((length,), float("nan"))
    for index in range(1, length - 1):
        d2[index] = torch.linalg.vector_norm(calpha[index + 1] - calpha[index - 1])
        left = calpha[index - 1] - calpha[index]
        right = calpha[index + 1] - calpha[index]
        radius_angle[index] = torch.acos(
            torch.dot(left, right) / (torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right))
        )
    for index in range(1, length - 2):
        d3[index] = torch.linalg.vector_norm(calpha[index + 2] - calpha[index - 1])
        first = calpha[index] - calpha[index - 1]
        second = calpha[index + 1] - calpha[index]
        third = calpha[index + 2] - calpha[index + 1]
        normal_first = torch.cross(first, second, dim=-1)
        normal_second = torch.cross(second, third, dim=-1)
        torsion[index] = torch.atan2(
            torch.dot(torch.cross(normal_first, normal_second, dim=-1), second / torch.linalg.vector_norm(second)),
            torch.dot(normal_first, normal_second),
        )
    for index in range(1, length - 3):
        d4[index] = torch.linalg.vector_norm(calpha[index + 3] - calpha[index - 1])

    degrees = math.pi / 180
    potential_helix = ((d3 >= 4.8) & (d3 <= 5.8) & (d4 >= 5.8) & (d4 <= 7.0)) | (
        (radius_angle >= 77 * degrees)
        & (radius_angle <= 101 * degrees)
        & (torsion >= 30 * degrees)
        & (torsion <= 70 * degrees)
    )
    helix = torch.zeros(length, dtype=torch.bool, device=calpha.device)
    consecutive = 0
    for index in range(length):
        if bool(potential_helix[index]):
            consecutive += 1
        else:
            if consecutive >= 5:
                helix[index - consecutive : index] = True
            consecutive = 0
    for index in torch.where(helix)[0].tolist():
        labels[index] = 0
        if index > 0 and (
            4.8 <= float(d3[index - 1]) <= 5.8 or 77 * degrees <= float(radius_angle[index - 1]) <= 101 * degrees
        ):
            labels[index - 1] = 0
        if index + 1 < length and (
            4.8 <= float(d3[index + 1]) <= 5.8 or 77 * degrees <= float(radius_angle[index + 1]) <= 101 * degrees
        ):
            labels[index + 1] = 0

    potential_strand = ((d2 >= 6.1) & (d2 <= 7.3) & (d3 >= 9.0) & (d3 <= 10.8) & (d4 >= 11.3) & (d4 <= 13.5)) | (
        (radius_angle >= 110 * degrees)
        & (radius_angle <= 138 * degrees)
        & (
            ((torsion >= -180 * degrees) & (torsion <= -125 * degrees))
            | ((torsion >= 145 * degrees) & (torsion <= 180 * degrees))
        )
    )
    strand = torch.zeros(length, dtype=torch.bool, device=calpha.device)
    consecutive = 0
    contacts = 0
    for index in range(length):
        if bool(potential_strand[index]):
            consecutive += 1
            distances = torch.linalg.vector_norm(calpha - calpha[index], dim=-1)
            contacts += int(((distances >= 4.2) & (distances <= 5.2)).sum())
        else:
            if consecutive >= 4 or (consecutive == 3 and contacts >= 5):
                strand[index - consecutive : index] = True
            consecutive = 0
            contacts = 0
    for index in torch.where(strand)[0].tolist():
        labels[index] = 1
        if index > 0 and 9.0 <= float(d3[index - 1]) <= 10.8:
            labels[index - 1] = 1
        if index + 1 < length and 9.0 <= float(d3[index + 1]) <= 10.8:
            labels[index + 1] = 1
    return labels


def _materialize_placement_map(
    placement: RfDiffusionPlacements,
    output_residues: Sequence[ComponentReference | SymbolicResidueSite],
    residue_map: Mapping[ComponentReference | SymbolicResidueSite, ComponentReference],
) -> tuple[PlacementMap, dict[str, str]]:
    r"""
    Lift the sampled flat surface into the shared ``placement_map``.

    ``placement`` is the declared [`RfDiffusionPlacements`][] condition; ``output_residues``
    is the post-sample flat identity tuple (each entry a source ``ComponentReference`` for an
    ``ExistingSpan`` row or a [`SymbolicResidueSite`][] for a generated row);
    ``residue_map`` maps each declared output residue to its realized OUTPUT
    ``ComponentReference`` (as built while decoding the Structure). Returns
    ``(placement_map, instance_of)`` where ``instance_of`` is the derived
    ``placement_id -> declaration_group`` index (which also re-validates global
    ``placement_id`` uniqueness).

    Lossless-ness: reconstruction is exact -- a generated placement's occurrences are its
    ``SymbolicResidueSite`` rows in ascending offset order, an existing placement's are its
    source components in ``[start, stop)`` order, both projected through ``residue_map``;
    outer/inner iteration is the declared order. The realized id rides per-occurrence in
    ``ComponentReference[0]`` (so a placement MAY span multiple realized instances).
    EXHAUSTIVE coverage is enforced: the union of all placement occurrences is a full
    partition of the realized output surface (no gap, no double-cover, no unaddressed tail).
    """

    spec = placement.placements
    instance_of = derive_instance_of(spec)

    generated: dict[str, list[tuple[int, ComponentReference]]] = {}
    for reference in output_residues:
        if isinstance(reference, SymbolicResidueSite):
            generated.setdefault(reference.placement_id, []).append((reference.offset, residue_map[reference]))

    placement_map: PlacementMap = {}
    for group_key, placements in spec.items():
        realized: dict[str, tuple[ComponentReference, ...]] = {}
        for placement_id, value in placements.items():
            if isinstance(value, GeneratedSegment):
                occurrences = tuple(reference for _, reference in sorted(generated.get(placement_id, ())))
            else:
                occurrences = tuple(residue_map[(value.instance_id, index)] for index in range(value.start, value.stop))
            realized[placement_id] = occurrences
        placement_map[group_key] = realized

    _check_exhaustive(placement_map, output_residues, residue_map)
    return placement_map, instance_of


def _check_exhaustive(
    placement_map: PlacementMap,
    output_residues: Sequence[ComponentReference | SymbolicResidueSite],
    residue_map: Mapping[ComponentReference | SymbolicResidueSite, ComponentReference],
) -> None:
    r"""
    Require ``placement_map`` to partition the materialized output exactly once.
    """

    addressed = [
        reference for group in placement_map.values() for occurrences in group.values() for reference in occurrences
    ]
    if len(set(addressed)) != len(addressed):
        raise ValueError("rfdiffusion placement_map double-covers a realized output residue")
    realized = [residue_map[reference] for reference in output_residues]
    addressed_set = set(addressed)
    realized_set = set(realized)
    if addressed_set != realized_set:
        missing = sorted(realized_set - addressed_set)
        extra = sorted(addressed_set - realized_set)
        raise ValueError(
            "rfdiffusion placement_map is not exhaustive over the realized output surface: "
            f"unaddressed_tail={missing} phantom={extra}"
        )


def _plddt_expectation(logits: Tensor) -> Tensor:
    r"""
    Convert per-bin pLDDT logits `[B, num_bins, L]` into the expected pLDDT value `[B, L]`.
    """

    num_bins = logits.shape[1]
    bins = torch.linspace(1.0 / num_bins, 1.0, num_bins, dtype=logits.dtype, device=logits.device)
    return torch.sum(torch.softmax(logits, dim=1) * bins[None, :, None], dim=1)


def _chain_indices(residue_index: torch.Tensor, threshold: int = 35) -> torch.Tensor:
    r"""
    Assign a chain id per residue by detecting large jumps (> ``threshold``) in the residue index.
    """

    starts = (residue_index[:, 1:] - residue_index[:, :-1] > threshold).to(torch.int64)
    return torch.cat((torch.zeros_like(residue_index[:, :1]), starts), dim=1).cumsum(dim=1)


def _cyclic_mask(cyclize: torch.Tensor | None, residue_index: torch.Tensor) -> torch.Tensor | None:
    r"""
    Broadcast a cyclic-residue mask to `[B, L]` on ``residue_index``'s device, or return `None` when unset.
    """

    if cyclize is None:
        return None
    cyclize = cyclize.to(device=residue_index.device, dtype=torch.bool)
    if cyclize.ndim == 1:
        cyclize = cyclize.unsqueeze(0).expand(residue_index.shape[0], -1)
    return cyclize


def _one_hot_lookup(one_hot: torch.Tensor, embedding: nn.Embedding) -> torch.Tensor:
    r"""
    Look up embedding rows via a one-hot matmul, including soft residue-type features.
    """

    return torch.matmul(one_hot, embedding.weight)


def _zero_bias(linear: nn.Linear) -> None:
    r"""
    Zero the bias of a linear layer in place.
    """

    nn.init.zeros_(linear.bias)


_DEFAULT_SE3_CONFIG = {
    "l0_in_features": 32,
    "l0_out_features": 16,
    "num_edge_features": 32,
}


def _se3_config(params: Mapping[str, int] | None) -> dict[str, int]:
    r"""
    Return a copy of ``params`` (or the default SE(3) sizing) as a plain mutable dict.
    """

    return dict(_DEFAULT_SE3_CONFIG if params is None else params)


def _query_state_update(msa: Tensor, state: Tensor) -> Tensor:
    r"""
    Add the projected node ``state`` into the query (row 0) slot of ``msa``.
    """

    query_index = torch.zeros(1, dtype=torch.long, device=state.device)
    return msa.index_add(1, query_index, state)


def _backbone_ca_distogram(coords: Tensor) -> Tensor:
    r"""
    Return the radial-basis distogram of the backbone CA-CA distances from coords `[B, L, A, 3]`.
    """

    return _radial_basis(torch.cdist(coords[:, :, 1], coords[:, :, 1]))


def _apply_frames(rotations: Tensor, translations: Tensor, local_coords: Tensor) -> Tensor:
    r"""
    Apply per-residue frames ``(rotations, translations)`` to ``local_coords`` `[B, L, A, 3]`.
    """

    return einsum("bnij,bnaj->bnai", rotations, local_coords) + translations.unsqueeze(-2)


def _initial_frames(coords: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    r"""
    Build identity rotations, CA translations, and CA-centered local coords from backbone coords.
    """

    batch_size, length = coords.shape[:2]
    rotations = (
        torch.eye(3, device=coords.device, dtype=coords.dtype).reshape(1, 1, 3, 3).expand(batch_size, length, -1, -1)
    )
    translations = coords[:, :, 1]
    local_coords = coords - translations.unsqueeze(-2)
    return rotations, translations, local_coords


def _quaternion_update_matrix(rotation_vector: Tensor, *, device: torch.device) -> Tensor:
    r"""
    Convert a per-residue rotation vector `[B, L, 3]` into a rotation matrix `[B, L, 3, 3]` via a quaternion.
    """

    norm = torch.sqrt(1 + torch.sum(rotation_vector * rotation_vector, dim=-1))
    q_a = 1 / norm
    q_b = rotation_vector[:, :, 0] / norm
    q_c = rotation_vector[:, :, 1] / norm
    q_d = rotation_vector[:, :, 2] / norm

    row0 = torch.stack(
        (
            q_a * q_a + q_b * q_b - q_c * q_c - q_d * q_d,
            2 * q_b * q_c - 2 * q_a * q_d,
            2 * q_b * q_d + 2 * q_a * q_c,
        ),
        dim=-1,
    )
    row1 = torch.stack(
        (
            2 * q_b * q_c + 2 * q_a * q_d,
            q_a * q_a - q_b * q_b + q_c * q_c - q_d * q_d,
            2 * q_c * q_d - 2 * q_a * q_b,
        ),
        dim=-1,
    )
    row2 = torch.stack(
        (
            2 * q_b * q_d - 2 * q_a * q_c,
            2 * q_c * q_d + 2 * q_a * q_b,
            q_a * q_a - q_b * q_b - q_c * q_c + q_d * q_d,
        ),
        dim=-1,
    )
    return torch.stack((row0, row1, row2), dim=-2).to(device=device)


def _degree_to_dimension(degree: int) -> int:
    r"""
    Return the dimension ``2*degree + 1`` of a spherical-harmonic feature of the given degree.

    Args:
        degree: SE(3) representation degree (0 for scalars, 1 for vectors, ...).

    Returns:
        The number of components a single feature of that degree occupies.
    """

    return 2 * degree + 1


def _unfuse_features(features: Tensor, degrees: Iterable[int]) -> dict[str, Tensor]:
    r"""
    Split a fused feature tensor into a per-degree dict.

    Args:
        features: Fused features `[..., sum(_degree_to_dimension(d) for d in degrees)]`.
        degrees: Degrees present, in the order they are packed along the last axis.

    Returns:
        A dict keyed by string degree, each holding the slice for that degree.
    """

    degree_list = list(degrees)
    return dict(
        zip(
            [str(degree) for degree in degree_list],
            features.split([_degree_to_dimension(degree) for degree in degree_list], dim=-1),
        )
    )


def _populated_edge_features(relative_pos: Tensor, edge_features: dict[str, Tensor] | None = None) -> dict[str, Tensor]:
    r"""
    Append the edge length as an invariant (degree-0) feature to the edge feature dict.

    Args:
        relative_pos: Per-edge relative position vectors `[E, 3]`.
        edge_features: Existing per-degree edge features, or `None` to start empty.

    Returns:
        A new edge-feature dict with the pairwise distance concatenated into degree ``"0"``.
    """

    edge_features = edge_features.copy() if edge_features else {}
    radius = relative_pos.norm(dim=-1, keepdim=True)
    if "0" in edge_features:
        edge_features["0"] = torch.cat([edge_features["0"], radius[..., None]], dim=1)
    else:
        edge_features["0"] = radius[..., None]
    return edge_features


_DISTANCE_BINS = 37
_OMEGA_BINS = 37
_THETA_BINS = 37
_PHI_BINS = 19


def _zero_init(linear: nn.Linear) -> None:
    r"""
    Zero both the weight and bias of a linear head in place.
    """

    nn.init.zeros_(linear.weight)
    nn.init.zeros_(linear.bias)


def _channel_first(logits: torch.Tensor) -> torch.Tensor:
    r"""
    Move the last (channel) axis of ``logits`` to position 1, matching the channel-first head convention.
    """

    return logits.movedim(-1, 1)


def _initialize_lecun_normal(module: nn.Module) -> nn.Module:
    r"""
    Re-initialize a layer's weight in place with the truncated Lecun-normal scheme, returning the layer.

    Args:
        module: A layer exposing a ``weight`` parameter.

    Returns:
        The same ``module`` with its weight replaced.
    """

    module.weight = nn.Parameter(_lecun_normal_tensor(module.weight))
    return module


def _lecun_normal_tensor(weight: torch.Tensor) -> torch.Tensor:
    r"""
    Draw a truncated Lecun-normal initialization tensor shaped like ``weight``.

    Args:
        weight: Tensor whose shape (and its fan-in) sets the output shape and scale.

    Returns:
        A new tensor of the same shape, drawn from the truncated Lecun-normal distribution.
    """

    if weight.is_meta:
        return torch.empty_like(weight)

    def truncated_normal(
        uniform: torch.Tensor,
        mu: float = 0.0,
        sigma: float = 1.0,
        a: int = -2,
        b: int = 2,
    ) -> Tensor:
        r"""
        Map uniform samples to a normal distribution truncated to ``[a, b]`` via the inverse-CDF.

        Args:
            uniform: Samples drawn from `U(0, 1)` to be transformed; any shape.
            mu: Mean of the underlying (untruncated) normal.
            sigma: Standard deviation of the underlying (untruncated) normal.
            a: Lower truncation bound, in standard deviations from ``mu``.
            b: Upper truncation bound, in standard deviations from ``mu``.

        Returns:
            Truncated-normal samples of the same shape as ``uniform``, clamped to `[a, b]`.
        """

        normal = torch.distributions.normal.Normal(0, 1)
        alpha = (a - mu) / sigma
        beta = (b - mu) / sigma
        alpha_cdf = normal.cdf(torch.tensor(alpha))
        p = alpha_cdf + (normal.cdf(torch.tensor(beta)) - alpha_cdf) * uniform
        v = torch.clamp(2 * p - 1, -1 + 1e-8, 1 - 1e-8)
        return torch.clamp(mu + sigma * math.sqrt(2) * torch.erfinv(v), a, b)

    stddev = math.sqrt(1.0 / weight.shape[-1]) / 0.87962566103423978
    return stddev * truncated_normal(torch.rand(weight.shape, device=weight.device, dtype=weight.dtype))


def _radial_basis(distances: torch.Tensor) -> torch.Tensor:
    r"""
    Expand pairwise ``distances`` into 36 Gaussian radial-basis features over 0-20 Angstrom.

    Args:
        distances: Pairwise distances of any shape `[...]`.

    Returns:
        Radial-basis features `[..., 36]`.
    """

    d_min, d_max, d_count = 0.0, 20.0, 36
    d_mu = torch.linspace(d_min, d_max, d_count, device=distances.device, dtype=distances.dtype)[None, :]
    d_sigma = (d_max - d_min) / d_count
    return torch.exp(-(((distances.unsqueeze(-1) - d_mu) / d_sigma) ** 2))


def _sequence_separation(residue_index: torch.Tensor, cyclic: torch.Tensor | None = None) -> torch.Tensor:
    r"""
    Build the signed sequence-neighbor edge feature (+/-1 for adjacent residues, 0 otherwise).

    Adjacent-in-sequence residues within a chain get a signed one; cyclic chains additionally link their first
    and last residues so the topology closes.

    Args:
        residue_index: Residue-index tensor `[B, L]`.
        cyclic: Optional cyclic-residue mask closing each cyclic chain.

    Returns:
        The neighbor feature `[B, L, L, 1]`.
    """

    seqsep = residue_index[:, None, :] - residue_index[:, :, None]
    sign = torch.sign(seqsep)
    neigh = torch.abs(seqsep)
    neigh[neigh > 1] = 0.0
    neigh = sign * neigh

    cyclic_mask = _cyclic_mask(cyclic, residue_index)
    if cyclic_mask is not None:
        chain_indices = _chain_indices(residue_index)
        same_chain = chain_indices[:, :, None] == chain_indices[:, None, :]
        positions = torch.arange(residue_index.shape[1], device=residue_index.device)
        earlier = positions[None, None, :] < positions[None, :, None]
        later = positions[None, None, :] > positions[None, :, None]
        cyclic_peers = same_chain & cyclic_mask[:, None, :]
        first = cyclic_mask & ~(cyclic_peers & earlier).any(dim=-1)
        last = cyclic_mask & ~(cyclic_peers & later).any(dim=-1)
        forward_closure = last[:, :, None] & first[:, None, :] & same_chain
        reverse_closure = first[:, :, None] & last[:, None, :] & same_chain
        neigh = torch.where(forward_closure, torch.ones_like(neigh), neigh)
        neigh = torch.where(reverse_closure, -torch.ones_like(neigh), neigh)

    return neigh.unsqueeze(-1)


def _full_graph(
    coords: torch.Tensor,
    pair: torch.Tensor,
    residue_index: torch.Tensor,
) -> tuple[RfDiffusionGraph, torch.Tensor]:
    r"""
    Build a fully-connected residue graph over all distinct residue pairs.

    Args:
        coords: Backbone CA coords `[B, L, 3]`.
        pair: Pair representation `[B, L, L, C]`.
        residue_index: Residue-index tensor `[B, L]`.

    Returns:
        A ``(graph, edge_features)`` pair: the
        [`RfDiffusionGraph`][ofoldx.models.rfdiffusion.model.RfDiffusionGraph] and its per-edge features.
    """

    batch_size, length = coords.shape[:2]
    residues = torch.arange(length, device=coords.device)
    batch = torch.arange(batch_size, device=coords.device)[:, None, None].expand(-1, length, length).reshape(-1)
    source_residue_index = residues[None, :, None].expand(batch_size, -1, length).reshape(-1)
    destination_residue_index = residues[None, None, :].expand(batch_size, length, -1).reshape(-1)
    edge_mask = (residue_index[:, None, :] != residue_index[:, :, None]).reshape(-1)
    graph = RfDiffusionGraph(
        coords,
        pair,
        batch,
        source_residue_index,
        destination_residue_index,
        edge_mask,
        batch_size=batch_size,
        length=length,
    )
    return graph, graph.edge_features


def _top_k_graph(
    coords: torch.Tensor,
    pair: torch.Tensor,
    residue_index: torch.Tensor,
    top_k: int = 64,
    kmin: int = 32,
    eps: float = 1e-6,
) -> tuple[RfDiffusionGraph, torch.Tensor]:
    r"""
    Build a k-nearest-neighbor residue graph (plus close-in-sequence edges).

    Keeps each residue's ``top_k`` nearest neighbors by CA distance, breaking ties by sequence separation, and
    additionally includes all pairs closer than ``kmin`` in sequence.

    Args:
        coords: Backbone CA coords `[B, L, 3]`.
        pair: Pair representation `[B, L, L, C]`.
        residue_index: Residue-index tensor `[B, L]`.
        top_k: Number of spatial neighbors per residue.
        kmin: Sequence-separation threshold below which pairs are always connected.
        eps: Tie-breaking weight on sequence separation.

    Returns:
        A ``(graph, edge_features)`` pair: the
        [`RfDiffusionGraph`][ofoldx.models.rfdiffusion.model.RfDiffusionGraph] and its per-edge features.
    """

    batch_size, length = coords.shape[:2]
    distances = torch.cdist(coords, coords) + torch.eye(length, device=coords.device).unsqueeze(0) * 999.9
    sep = (residue_index[:, None, :] - residue_index[:, :, None]).abs() + torch.eye(
        length, device=coords.device
    ).unsqueeze(0) * 999.9
    distances = distances + sep * eps
    _, topk_targets = torch.topk(distances, min(top_k, length), largest=False)
    residues = torch.arange(length, device=coords.device)
    local_offsets = torch.arange(1 - kmin, kmin, device=coords.device)
    local_targets = residues[:, None] + local_offsets[None]
    local_valid = (local_targets >= 0) & (local_targets < length)
    local_targets = local_targets.clamp(0, length - 1)
    gathered_separation = torch.gather(
        sep,
        2,
        local_targets[None].expand(batch_size, -1, -1),
    )
    local_valid = local_valid[None] & (gathered_separation < kmin)
    local_valid = local_valid & ~(local_targets[None, :, :, None] == topk_targets[:, :, None, :]).any(dim=-1)
    targets = torch.cat(
        (topk_targets, local_targets[None].expand(batch_size, -1, -1)),
        dim=-1,
    )
    edge_mask = torch.cat((torch.ones_like(topk_targets, dtype=torch.bool), local_valid), dim=-1)
    edges_per_source = targets.shape[-1]
    batch = (
        torch.arange(batch_size, device=coords.device)[:, None, None].expand(-1, length, edges_per_source).reshape(-1)
    )
    source_residue_index = residues[None, :, None].expand(batch_size, -1, edges_per_source).reshape(-1)
    destination_residue_index = targets.reshape(-1)
    graph = RfDiffusionGraph(
        coords,
        pair,
        batch,
        source_residue_index,
        destination_residue_index,
        edge_mask.reshape(-1),
        batch_size=batch_size,
        length=length,
    )
    return graph, graph.edge_features


def _beta_schedule(
    num_timesteps: int,
    beta_initial: float,
    beta_final: float,
    schedule_type: str,
) -> tuple[Tensor, Tensor, Tensor]:
    r"""
    Build the RFdiffusion translation-noise beta, alpha, and cumulative alpha-bar schedules.

    Args:
        num_timesteps: Total number of diffusion timesteps (must be >= 15).
        beta_initial: Initial beta (rescaled by ``200 / num_timesteps``).
        beta_final: Final beta (rescaled by ``200 / num_timesteps``).
        schedule_type: Schedule shape; only `"linear"` is supported.

    Returns:
        A tuple ``(beta, alpha, alpha_bar)`` of `[num_timesteps]` schedule tensors.

    Raises:
        ValueError: If ``schedule_type`` is not `"linear"` or ``num_timesteps`` is below 15.
    """

    if schedule_type != "linear":
        raise ValueError(f"unsupported RFdiffusion beta schedule {schedule_type!r}")
    if num_timesteps < 15:
        raise ValueError("RFdiffusion beta schedule expects num_timesteps >= 15")
    beta0 = beta_initial * 200 / num_timesteps
    betaT = beta_final * 200 / num_timesteps
    beta = torch.linspace(beta0, betaT, num_timesteps)
    alpha = 1 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    return beta, alpha, alpha_bar


def _next_frames(
    noisy_coords: Tensor,
    pred_coords: Tensor,
    t: int,
    diffuser: RfDiffusionDiffuser,
    so3_type: str,
    diffusion_mask: Tensor,
    *,
    noise_scale: float = 1.0,
    generator: torch.Generator | None = None,
) -> Tensor:
    r"""
    Update backbone frames from ``noisy_coords`` one reverse step toward predicted ``x_0``.

    Args:
        noisy_coords: Current noised backbone coords `[L, A, 3]`.
        pred_coords: Predicted clean backbone coords `[L, A, 3]`.
        t: Current timestep.
        diffuser: The forward
            [`RfDiffusionDiffuser`][ofoldx.models.rfdiffusion.model.RfDiffusionDiffuser]
            whose IGSO3 reverse sampler is used.
        so3_type: Rotation diffuser type; only `"igso3"` is supported.
        diffusion_mask: Per-residue mask; masked (motif) frames stay fixed.
        noise_scale: Scale on the reverse rotation noise.

    Returns:
        The updated backbone coords `[L, 3, 3]`.

    Raises:
        ValueError: If ``so3_type`` is not `"igso3"`.
    """

    rotations_0, _Ca_0 = rigid_from_three_points(
        pred_coords[None, :, 0], pred_coords[None, :, 1], pred_coords[None, :, 2]
    )
    rotations_t, Ca_t = rigid_from_three_points(
        noisy_coords[None, :, 0], noisy_coords[None, :, 1], noisy_coords[None, :, 2]
    )
    device = noisy_coords.device
    # ``Rotation.from_matrix(...).as_matrix()`` in pure torch: project the near-orthonormal frames
    # onto the nearest proper rotation in float64.
    R_0 = so3_project_to_rotation(rotations_0.squeeze(0).to(dtype=torch.float64))
    R_t = so3_project_to_rotation(rotations_t.squeeze(0).to(dtype=torch.float64))
    length = R_t.shape[0]
    diffusion_mask = diffusion_mask.to(device=device).bool()
    mobile = ~diffusion_mask
    transitions = torch.eye(3, dtype=torch.float64, device=device).expand(length, 3, 3).clone()
    if so3_type == "igso3":
        if bool(mobile.any()):
            transitions[mobile] = diffuser.so3_diffuser.reverse_sample_vectorized(
                R_t[mobile],
                R_0[mobile],
                t,
                noise_level=noise_scale,
                return_perturb=True,
                generator=generator,
            ).to(dtype=torch.float64)
    else:
        raise ValueError(f"unsupported SO(3) diffusion type {so3_type!r}")
    transitions = transitions[:, None]
    ca_t = Ca_t.squeeze(0).to(dtype=torch.float64)
    noisy_bb = noisy_coords[:, :3].to(dtype=torch.float64)
    next_coords = (torch.einsum("lrij,laj->lrai", transitions, noisy_bb - ca_t[:, None]) + ca_t[:, None, None]).squeeze(
        1
    )
    return next_coords.to(dtype=noisy_coords.dtype, device=noisy_coords.device)


def _next_alpha_carbon(
    noisy_coords: Tensor,
    pred_coords: Tensor,
    t: int,
    diffusion_mask: Tensor,
    crd_scale: float,
    beta_schedule: Tensor,
    alphabar_schedule: Tensor,
    *,
    noise_scale: float = 1.0,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    r"""
    Sample the reverse CA-translation update for one timestep.

    Args:
        noisy_coords: Current noised coords `[L, A, 3]`.
        pred_coords: Predicted clean coords `[L, A, 3]`.
        t: Current timestep.
        diffusion_mask: Per-residue mask; masked (motif) CA positions stay fixed.
        crd_scale: Coordinate-scaling factor.
        beta_schedule: The beta schedule `[T]`.
        alphabar_schedule: The cumulative alpha-bar schedule `[T]`.
        noise_scale: Scale on the injected posterior noise.
        generator: Optional RNG for reproducible sampling.

    Returns:
        A tuple ``(next_coords, delta)`` of the updated coords and the CA delta, both unscaled.
    """

    pred_coords_scaled = pred_coords * crd_scale
    noisy_coords_scaled = noisy_coords * crd_scale
    mu, sigma = ddpm_posterior(
        noisy_coords_scaled[:, 1],
        pred_coords_scaled[:, 1],
        t,
        beta_schedule,
        alphabar_schedule,
    )
    sampled = mu + torch.randn(mu.shape, dtype=mu.dtype, device=mu.device, generator=generator) * torch.sqrt(
        sigma * noise_scale
    )
    delta = sampled - noisy_coords_scaled[:, 1]
    delta[diffusion_mask.to(device=delta.device).bool()] = 0
    out = noisy_coords_scaled + delta[:, None]
    return out / crd_scale, delta / crd_scale


def _noise_scale(
    timestep: int,
    num_timesteps: int,
    initial_noise: float,
    final_noise: float,
    schedule_type: Literal["constant", "linear"],
) -> float:
    r"""
    Return the noise scale at one reverse-diffusion timestep.

    Args:
        timestep: Current one-indexed reverse-diffusion timestep.
        num_timesteps: Total number of timesteps (for the linear interpolation).
        initial_noise: Noise scale at the initial timestep ``t = num_timesteps``.
        final_noise: Linear-interpolation endpoint at timestep ``t = 1``.
        schedule_type: Schedule shape (`"constant"` or `"linear"`).

    Returns:
        The noise scale for ``timestep``.
    """

    if schedule_type == "constant":
        return initial_noise
    return ((timestep - 1) / (num_timesteps - 1)) * (initial_noise - final_noise) + final_noise


_PAIR_MIN_DISTANCE = 2.0
_PAIR_MAX_DISTANCE = 20.0
_PAIR_DISTANCE_BINS = 36


def _one_hot_sequence(
    seq: Tensor,
    *,
    num_classes: int = 22,
    dtype: torch.dtype,
) -> Tensor:
    r"""
    Encode RFdiffusion residue IDs as one-hot residue-type features.

    Args:
        seq: Residue ids `[L]`.
        num_classes: Number of classes (residue vocabulary size).
        dtype: Floating dtype of the encoded features.

    Returns:
        One-hot residue-type features `[L, num_classes]`.
    """

    return F.one_hot(seq.long(), num_classes=num_classes).to(dtype=dtype)


def _beta_carbon(nitrogen: Tensor, alpha_carbon: Tensor, carbon: Tensor) -> Tensor:
    r"""
    Reconstruct C-beta coords from backbone N/CA/C coords.

    Args:
        nitrogen: N coords `[..., 3]`.
        alpha_carbon: CA coords `[..., 3]`.
        carbon: C coords `[..., 3]`.

    Returns:
        The reconstructed C-beta coords `[..., 3]`.
    """

    b = alpha_carbon - nitrogen
    c = carbon - alpha_carbon
    a = torch.cross(b, c, dim=-1)
    return -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + alpha_carbon


def _planar_angle(a: Tensor, b: Tensor, c: Tensor) -> Tensor:
    r"""
    Return the angle at vertex ``b`` of the triples ``(a, b, c)`` in radians.

    Args:
        a: First point(s) `[..., 3]`.
        b: Vertex point(s) `[..., 3]` where the angle is measured.
        c: Third point(s) `[..., 3]`.

    Returns:
        The planar angle `[...]` at ``b`` in radians, in ``[0, pi]``.
    """

    v = a - b
    w = c - b
    v = v / torch.norm(v, dim=-1, keepdim=True)
    w = w / torch.norm(w, dim=-1, keepdim=True)
    return torch.acos(torch.sum(v * w, dim=-1))


def _dihedral_angle(a: Tensor, b: Tensor, c: Tensor, d: Tensor) -> Tensor:
    r"""
    Return the signed _dihedral_angle angle of the four-point sequence ``(a, b, c, d)`` in radians.

    Args:
        a: First point(s) `[..., 3]`.
        b: Second point(s) `[..., 3]` (start of the central bond).
        c: Third point(s) `[..., 3]` (end of the central bond).
        d: Fourth point(s) `[..., 3]`.

    Returns:
        The signed _dihedral_angle angle `[...]` in radians, in ``(-pi, pi]``.
    """

    beta_initial = a - b
    b1 = c - b
    b2 = d - c
    b1 = b1 / torch.norm(b1, dim=-1, keepdim=True)
    v = beta_initial - torch.sum(beta_initial * b1, dim=-1, keepdim=True) * b1
    w = b2 - torch.sum(b2 * b1, dim=-1, keepdim=True) * b1
    x = torch.sum(v * w, dim=-1)
    y = torch.sum(torch.cross(b1, v, dim=-1) * w, dim=-1)
    return torch.atan2(y, x)


def _backbone_pair_geometry(
    coordinates: Tensor,
    *,
    max_distance: float = _PAIR_MAX_DISTANCE,
) -> tuple[Tensor, Tensor]:
    r"""
    Convert backbone coordinates into distance and orientation pair features.

    Args:
        coordinates: Backbone coordinates `[B, L, 3, 3]` in N/CA/C order.
        max_distance: Maximum valid beta-carbon pair distance in angstroms.

    Returns:
        Pair geometry `[B, L, L, 4]` containing one distance and three angles, plus its validity mask.
    """

    batch_size, num_residues = coordinates.shape[:2]
    nitrogen, alpha_carbon, carbon = coordinates[:, :, 0], coordinates[:, :, 1], coordinates[:, :, 2]
    beta_carbon = _beta_carbon(nitrogen, alpha_carbon, carbon)
    pair_geometry = torch.zeros(
        (batch_size, num_residues, num_residues, 4), dtype=coordinates.dtype, device=coordinates.device
    )
    distance = torch.cdist(beta_carbon, beta_carbon)
    distance = torch.where(torch.isnan(distance), 999.9, distance)
    pair_geometry[..., 0] = distance + 999.9 * torch.eye(num_residues, device=coordinates.device)[None]
    valid = pair_geometry[..., 0] < max_distance
    left_ca = alpha_carbon[:, :, None]
    left_cb = beta_carbon[:, :, None]
    right_cb = beta_carbon[:, None, :]
    right_ca = alpha_carbon[:, None, :]
    pair_geometry[..., 1] = torch.where(
        valid,
        _dihedral_angle(left_ca, left_cb, right_cb, right_ca),
        0,
    )
    pair_geometry[..., 2] = torch.where(
        valid,
        _dihedral_angle(nitrogen[:, :, None], left_ca, left_cb, right_cb),
        0,
    )
    pair_geometry[..., 3] = torch.where(
        valid,
        _planar_angle(left_ca, left_cb, right_cb),
        0,
    )
    pair_geometry[..., 0] = torch.where(valid, pair_geometry[..., 0], 999.9)
    return pair_geometry, valid.to(dtype=coordinates.dtype)


def _distance_one_hot(
    distance: Tensor,
    *,
    min_distance: float = _PAIR_MIN_DISTANCE,
    max_distance: float = _PAIR_MAX_DISTANCE,
    num_bins: int = _PAIR_DISTANCE_BINS,
) -> Tensor:
    r"""
    Bin a distance map into one-hot distance features (with one extra out-of-range bin).

    Args:
        distance: Pairwise distance map; NaNs are treated as out of range.
        min_distance: Lower edge of the distance range in angstroms.
        max_distance: Upper edge of the distance range in angstroms.
        num_bins: Number of in-range distance bins.

    Returns:
        One-hot distance features `[..., num_bins + 1]`; the final bin catches out-of-range distances.
    """

    distance = torch.nan_to_num(distance, nan=999.9)
    bin_width = (max_distance - min_distance) / num_bins
    boundaries = torch.linspace(
        min_distance + bin_width,
        max_distance,
        num_bins,
        dtype=distance.dtype,
        device=distance.device,
    )
    bins = torch.bucketize(distance.contiguous(), boundaries).long()
    return F.one_hot(bins, num_classes=num_bins + 1).to(dtype=distance.dtype)


def _template_pair_features(
    template_coordinates: Tensor,
    *,
    min_distance: float = _PAIR_MIN_DISTANCE,
    max_distance: float = _PAIR_MAX_DISTANCE,
    num_bins: int = _PAIR_DISTANCE_BINS,
) -> Tensor:
    r"""
    Convert template backbone coords into the RFdiffusion ``template_pair_features`` pair features.

    Concatenates one-hot distance bins, sine/cosine orientation features, and a validity flag.

    Args:
        template_coordinates: Template coords `[B, T, L, A, 3]`.
        min_distance: Lower edge of the distance range in angstroms.
        max_distance: Upper edge of the distance range in angstroms.
        num_bins: Number of in-range distance bins.

    Returns:
        The ``template_pair_features`` features `[B, T, L, L, C]`.
    """

    batch_size, template_count, length = template_coordinates.shape[:3]
    pair_geometry, mask = _backbone_pair_geometry(
        template_coordinates[:, :, :, :3].reshape(batch_size * template_count, length, 3, 3),
        max_distance=max_distance,
    )
    pair_geometry = pair_geometry.reshape(batch_size, template_count, length, length, 4)
    mask = mask.reshape(batch_size, template_count, length, length, 1)
    distance = _distance_one_hot(
        pair_geometry[..., 0], min_distance=min_distance, max_distance=max_distance, num_bins=num_bins
    )
    orientation = torch.cat((torch.sin(pair_geometry[..., 1:]), torch.cos(pair_geometry[..., 1:])), dim=-1) * mask
    valid = ~torch.isnan(pair_geometry[..., 0])
    template_pair_features = torch.cat(
        (
            distance,
            orientation,
            valid.unsqueeze(-1).to(dtype=pair_geometry.dtype),
        ),
        dim=-1,
    )
    return torch.nan_to_num(template_pair_features, nan=0.0)


def _initial_coordinates(template_coordinates: Tensor) -> Tensor:
    r"""
    Fill missing template coords with the RFdiffusion ideal residue frame placed near the nearest known CA.

    Args:
        template_coordinates: Template coords `[B, T, L, 27, 3]` with NaNs for missing atoms.

    Returns:
        The filled template coords `[B, T, L, 27, 3]`.
    """

    B, T, length = template_coordinates.shape[:3]
    init = (
        _INITIAL_COORDINATES.to(device=template_coordinates.device, dtype=template_coordinates.dtype)
        .reshape(1, 1, 1, 27, 3)
        .repeat(B, T, length, 1, 1)
    )
    if torch.isnan(template_coordinates).all():
        return init
    missing = torch.isnan(template_coordinates[:, :, :, :3]).any(dim=-1).any(dim=-1)
    center_ca = ((~missing[:, :, :, None]) * torch.nan_to_num(template_coordinates[:, :, :, 1])).sum(dim=2)
    center_ca = center_ca / ((~missing[:, :, :, None]).sum(dim=2) + 1e-4)
    centered_coords = template_coordinates - center_ca.reshape(B, T, 1, 1, 3)
    for b_idx in range(B):
        for t_idx in range(T):
            if missing[b_idx, t_idx].all():
                continue
            existing = torch.where(~missing[b_idx, t_idx])[0]
            nearest = torch.argmin(
                (torch.arange(length, device=template_coordinates.device)[:, None] - existing[None]).abs(), dim=-1
            )
            gather_idx = torch.gather(existing, 0, nearest)
            offset = torch.gather(centered_coords[b_idx, t_idx, :, 1], 0, gather_idx[:, None].expand(-1, 3))
            init[b_idx, t_idx] += offset.reshape(length, 1, 3)
    return torch.where(missing.reshape(B, T, length, 1, 1), init, centered_coords)


def _torsion_features(coords: Tensor, seq: Tensor) -> tuple[Tensor, Tensor]:
    r"""
    Compute the RFdiffusion alpha torsion features and their validity masks.

    Computes backbone (omega/phi/psi) and sidechain chi torsions and encodes them as ``(cos, sin)`` pairs.

    Args:
        coords: RFdiffusion atom27 coords `[B, L, 27, 3]`.
        seq: Per-token residue ids `[B, L]` (or flattenable to it).

    Returns:
        A tuple ``(features, mask)`` of the torsion features `[B, L, 10, 2]` and the validity mask
        `[B, L, 10]`.

    """

    def normalize(value: Tensor) -> Tensor:
        return value / torch.sqrt(value.square().sum(dim=-1, keepdim=True) + 1e-8)

    def angle_pair(left: Tensor, right: Tensor) -> Tensor:
        left = normalize(left)
        right = normalize(right)
        cosine = torch.clamp((left * right).sum(dim=-1), -1, 1)
        sine = torch.sqrt(1 - cosine.square() + 1e-8)
        return torch.stack((cosine, sine), dim=-1)

    def dihedral_pair(a: Tensor, b: Tensor, c: Tensor, d: Tensor) -> Tensor:
        ab = normalize(a - b)
        bc = normalize(b - c)
        cd = normalize(c - d)
        first = normalize(torch.cross(ab, bc, dim=-1))
        second = normalize(torch.cross(bc, cd, dim=-1))
        sine = (torch.cross(first, bc, dim=-1) * second).sum(dim=-1)
        cosine = (first * second).sum(dim=-1)
        return torch.stack((cosine, sine), dim=-1)

    B, L = coords.shape[:2]
    seq = seq.long().reshape(B, L)
    coords = coords.clone()
    rotation, translation = rigid_from_three_points(coords[..., 0, :], coords[..., 1, :], coords[..., 2, :])
    ideal_nitrogen = coords.new_tensor((-0.5272, 1.3593, 0.0))
    ideal_carbon = coords.new_tensor((1.5233, 0.0, 0.0))
    coords[..., 0, :] = torch.einsum("...ij,j->...i", rotation, ideal_nitrogen) + translation
    coords[..., 2, :] = torch.einsum("...ij,j->...i", rotation, ideal_carbon) + translation

    features = coords.new_zeros((B, L, 10, 2))
    mask = torch.ones((B, L, 10), dtype=torch.bool, device=coords.device)
    features[:, 0, 1, 0] = 1
    features[:, -1, 0, 0] = 1
    mask[:, 0, 1] = False
    mask[:, -1, 0] = False
    if L > 1:
        features[:, :-1, 0] = dihedral_pair(coords[:, :-1, 1], coords[:, :-1, 2], coords[:, 1:, 0], coords[:, 1:, 1])
        features[:, 1:, 1] = dihedral_pair(coords[:, :-1, 2], coords[:, 1:, 0], coords[:, 1:, 1], coords[:, 1:, 2])
    features[:, :, 2] = -dihedral_pair(coords[:, :, 0], coords[:, :, 1], coords[:, :, 2], coords[:, :, 3])

    mask[:, :, 3:7] = False
    for aa_idx, chi_defs in enumerate(_CHI_ATOM_NAMES):
        aa_positions = seq == aa_idx
        atom_names = RFDIFFUSION_ATOM14_NAMES[aa_idx]
        for chi_idx, chi_atoms in enumerate(chi_defs):
            if chi_atoms is None:
                continue
            atom_indices = tuple(atom_names.index(atom_name) for atom_name in chi_atoms)
            chi = dihedral_pair(*(coords[:, :, atom_index] for atom_index in atom_indices))
            features[:, :, 3 + chi_idx] = torch.where(aa_positions[..., None], chi, features[:, :, 3 + chi_idx])
            mask[:, :, 3 + chi_idx] |= aa_positions
    for aa_idx, chi_idx, atom_indices in (
        (rfdiffusion_residue_token_id("CYS"), 1, (1, 4, 5, 18)),
        (rfdiffusion_residue_token_id("HIS"), 2, (7, 8, 19, 9)),
        (rfdiffusion_residue_token_id("PRO"), 2, (4, 5, 6, 19)),
        (rfdiffusion_residue_token_id("SER"), 1, (1, 4, 5, 15)),
        (rfdiffusion_residue_token_id("THR"), 1, (1, 4, 5, 15)),
        (rfdiffusion_residue_token_id("TYR"), 2, (8, 10, 11, 22)),
    ):
        aa_positions = seq == aa_idx
        chi = dihedral_pair(*(coords[:, :, atom_index] for atom_index in atom_indices))
        features[:, :, 3 + chi_idx] = torch.where(aa_positions[..., None], chi, features[:, :, 3 + chi_idx])
        mask[:, :, 3 + chi_idx] |= aa_positions

    _, _, reference_coords = _RF_ATOM14_KINEMATICS
    reference_coords = reference_coords.to(device=coords.device, dtype=coords.dtype)
    reference_nc = 0.5 * (reference_coords[:, 0, :3] + reference_coords[:, 2, :3])
    reference_ca = reference_coords[:, 1, :3]
    reference_cb = reference_coords[:, 4, :3]
    reference_cg = reference_coords[:, 5, :3]
    reference_cb_bend = angle_pair(reference_cb - reference_ca, reference_nc - reference_ca)
    reference_nc_projection = reference_coords[:, 2, :3] - reference_coords[:, 0, :3]
    reference_nc_projection = reference_nc_projection - (
        (reference_nc_projection * reference_nc).sum(dim=-1, keepdim=True)
        / reference_nc.square().sum(dim=-1, keepdim=True)
        * reference_nc
    )
    reference_cb_twist = angle_pair(reference_cb - reference_ca, reference_nc_projection)
    reference_cg_bend = angle_pair(reference_cg, reference_coords.new_tensor((-1.0, 0.0, 0.0)))

    nc = 0.5 * (coords[:, :, 0] + coords[:, :, 2])
    ca = coords[:, :, 1]
    cb = coords[:, :, 4]
    cb_bend = angle_pair(cb - ca, nc - ca)
    nc_axis = nc - ca
    nc_projection = coords[:, :, 2] - coords[:, :, 0]
    nc_projection = nc_projection - (
        (nc_projection * nc_axis).sum(dim=-1, keepdim=True) / nc_axis.square().sum(dim=-1, keepdim=True) * nc_axis
    )
    cb_twist = angle_pair(cb - ca, nc_projection)
    cg_bend = angle_pair(coords[:, :, 5] - cb, ca - cb)
    for index, (observed, reference) in enumerate(
        ((cb_bend, reference_cb_bend), (cb_twist, reference_cb_twist), (cg_bend, reference_cg_bend)),
        start=7,
    ):
        expected = reference[seq]
        features[:, :, index] = torch.stack(
            (
                (observed * expected).sum(dim=-1),
                observed[..., 0] * expected[..., 1] - observed[..., 1] * expected[..., 0],
            ),
            dim=-1,
        )

    glycine = rfdiffusion_residue_token_id("GLY")
    mask[:, :, 7] = seq != glycine
    mask[:, :, 8] = seq != glycine
    mask[:, :, 9] = (
        (seq != glycine)
        & (seq != rfdiffusion_residue_token_id("ALA"))
        & (seq != rfdiffusion_residue_token_id("UNK"))
        & (seq != RFDIFFUSION_MASK_TOKEN_ID)
    )
    features[..., 0] = torch.nan_to_num(features[..., 0], nan=1.0)
    features[..., 1] = torch.nan_to_num(features[..., 1], nan=0.0)
    return features, mask


def _template_torsion_features(template_coordinates: Tensor, seq: Tensor) -> Tensor:
    r"""
    Build the flattened ``template_torsion_features`` torsion tensor expected by RFdiffusion inference.

    Args:
        template_coordinates: Template atom27 coords `[B, T, L, 27, 3]`.
        seq: Per-token residue ids broadcastable to `[B*T, L]`.

    Returns:
        The flattened torsion features `[B, T, L, 30]` (angles plus masks).
    """

    B, T, length = template_coordinates.shape[:3]
    seq_flat = seq.reshape(B * T, length) if seq.ndim == 3 else seq.reshape(B, length).repeat_interleave(T, dim=0)
    alpha, alpha_mask = _torsion_features(
        template_coordinates.reshape(B * T, length, template_coordinates.shape[-2], 3), seq_flat
    )
    alpha = alpha.reshape(B, T, length, 10, 2)
    alpha_mask = alpha_mask.reshape(B, T, length, 10, 1).to(dtype=alpha.dtype)
    return torch.cat((alpha, alpha_mask), dim=-1).reshape(B, T, length, 30)


def _rotation_x(angs: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    r"""
    Build 4x4 homogeneous rotation matrices about the x-axis from ``(cos, sin)`` angle pairs `[B, L, 2]`.

    Args:
        angs: Per-position ``(cos, sin)`` angle pairs `[B, L, 2]`.
        eps: Floor added to the pair norm to avoid division by zero.

    Returns:
        Homogeneous rotation matrices `[B, L, 4, 4]`.
    """

    batch_size, length = angs.shape[:2]
    norm = torch.linalg.norm(angs, dim=-1) + eps
    rotations = torch.eye(4, device=angs.device, dtype=angs.dtype).repeat(batch_size, length, 1, 1)
    rotations[:, :, 1, 1] = angs[:, :, 0] / norm
    rotations[:, :, 1, 2] = -angs[:, :, 1] / norm
    rotations[:, :, 2, 1] = angs[:, :, 1] / norm
    rotations[:, :, 2, 2] = angs[:, :, 0] / norm
    return rotations


def _rotation_z(angs: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    r"""
    Build 4x4 homogeneous rotation matrices about the z-axis from ``(cos, sin)`` angle pairs `[B, L, 2]`.

    Args:
        angs: Per-position ``(cos, sin)`` angle pairs `[B, L, 2]`.
        eps: Floor added to the pair norm to avoid division by zero.

    Returns:
        Homogeneous rotation matrices `[B, L, 4, 4]`.
    """

    batch_size, length = angs.shape[:2]
    norm = torch.linalg.norm(angs, dim=-1) + eps
    rotations = torch.eye(4, device=angs.device, dtype=angs.dtype).repeat(batch_size, length, 1, 1)
    rotations[:, :, 0, 0] = angs[:, :, 0] / norm
    rotations[:, :, 0, 1] = -angs[:, :, 1] / norm
    rotations[:, :, 1, 0] = angs[:, :, 1] / norm
    rotations[:, :, 1, 1] = angs[:, :, 0] / norm
    return rotations


def _rotation_about_axis(angs: torch.Tensor, axis: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    r"""
    Build 4x4 homogeneous rotations about an arbitrary ``axis`` from ``(cos, sin)`` angle pairs (Rodrigues).

    Args:
        angs: Per-position ``(cos, sin)`` angle pairs `[B, L, 2]`.
        axis: Rotation axis per position `[B, L, 3]`; normalized internally.
        eps: Floor on the pair and axis norms to avoid division by zero.

    Returns:
        Homogeneous rotation matrices `[B, L, 4, 4]`.
    """

    batch_size, length = angs.shape[:2]
    norm = torch.linalg.norm(angs, dim=-1).clamp_min(eps)
    axis = axis / torch.linalg.norm(axis, dim=-1, keepdim=True).clamp_min(eps)
    cos_theta = angs[:, :, 0] / norm
    sin_theta = angs[:, :, 1] / norm
    u0, u1, u2 = axis.unbind(dim=-1)

    rotations = torch.eye(4, device=angs.device, dtype=angs.dtype).repeat(batch_size, length, 1, 1)
    one_minus_cos = 1 - cos_theta
    rotations[:, :, 0, 0] = cos_theta + u0 * u0 * one_minus_cos
    rotations[:, :, 0, 1] = u0 * u1 * one_minus_cos - u2 * sin_theta
    rotations[:, :, 0, 2] = u0 * u2 * one_minus_cos + u1 * sin_theta
    rotations[:, :, 1, 0] = u0 * u1 * one_minus_cos + u2 * sin_theta
    rotations[:, :, 1, 1] = cos_theta + u1 * u1 * one_minus_cos
    rotations[:, :, 1, 2] = u1 * u2 * one_minus_cos - u0 * sin_theta
    rotations[:, :, 2, 0] = u0 * u2 * one_minus_cos - u1 * sin_theta
    rotations[:, :, 2, 1] = u1 * u2 * one_minus_cos + u0 * sin_theta
    rotations[:, :, 2, 2] = cos_theta + u2 * u2 * one_minus_cos
    return rotations


def _sample_variable_lengths(
    input: RfDiffusionInput,
    generator: torch.Generator | None,
) -> RfDiffusionInput:
    placement = input.layout.placement
    if placement is None:
        return input
    rng_device = torch.device("cpu") if generator is None else generator.device

    def draw(low: int, high: int) -> int:
        if low == high:
            return low
        return int(
            torch.randint(
                low,
                high + 1,
                (),
                device=rng_device,
                generator=generator,
            ).item()
        )

    # Resample generated segment lengths until their combined total falls inside the requested window. A bounded
    # attempt count turns incompatible constraints into an error instead of silently dropping the length request.
    combined_length = input.layout.combined_length
    # placement_id -> GeneratedSegment, so a whole-placement reference (spelled as the bare
    # placement_id string, e.g. in RfDiffusionPointGroupSymmetry units) recovers its length bounds
    # without the value carrying its own id.
    segments_by_id: dict[str, GeneratedSegment] = {
        placement_id: cast(GeneratedSegment, value)
        for group in placement.placements.values()
        for placement_id, value in group.items()
        if isinstance(value, GeneratedSegment)
    }
    chain_lengths = (None,) * len(placement.placements) if placement.chain_lengths is None else placement.chain_lengths

    def sample_segment_lengths() -> dict[str, int]:
        sampled_lengths: dict[str, int] = {}
        for group, chain_length in zip(placement.placements.values(), chain_lengths, strict=True):
            if chain_length is None:
                continue
            generated_segments = [
                (placement_id, cast(GeneratedSegment, value))
                for placement_id, value in group.items()
                if isinstance(value, GeneratedSegment)
            ]
            remaining = chain_length - sum(
                value.stop - value.start for value in group.values() if isinstance(value, ExistingSpan)
            )
            for index, (placement_id, segment) in enumerate(generated_segments):
                following = [value for _, value in generated_segments[index + 1 :]]
                low = max(segment.min_length, remaining - sum(value.max_length for value in following))
                high = min(segment.max_length, remaining - sum(value.min_length for value in following))
                if low > high:
                    raise ValueError("placement chain length cannot be realized by its generated segments")
                length = draw(low, high)
                sampled_lengths[placement_id] = length
                remaining -= length

        symmetry = cast(
            RfDiffusionPointGroupSymmetry | None,
            next(
                (
                    condition
                    for condition in input.layout.conditions
                    if isinstance(condition, RfDiffusionPointGroupSymmetry)
                ),
                None,
            ),
        )
        if symmetry is not None:
            unit_ranges = []
            for unit in symmetry.units:
                minimum = sum(
                    sampled_lengths.get(target, segments_by_id[target].min_length) if isinstance(target, str) else 1
                    for target in unit
                )
                maximum = sum(
                    sampled_lengths.get(target, segments_by_id[target].max_length) if isinstance(target, str) else 1
                    for target in unit
                )
                unit_ranges.append((minimum, maximum))
            minimum = max(value[0] for value in unit_ranges)
            maximum = min(value[1] for value in unit_ranges)
            if minimum > maximum:
                raise ValueError("placement chain lengths are incompatible with RfDiffusionPointGroupSymmetry")
            total = draw(minimum, maximum)
            for unit in symmetry.units:
                unresolved_segments = [
                    cast(str, target) for target in unit if isinstance(target, str) and target not in sampled_lengths
                ]
                remaining = (
                    total
                    - sum(not isinstance(target, str) for target in unit)
                    - sum(
                        sampled_lengths[target]
                        for target in unit
                        if isinstance(target, str) and target in sampled_lengths
                    )
                )
                for index, placement_id in enumerate(unresolved_segments):
                    segment = segments_by_id[placement_id]
                    following = [segments_by_id[value] for value in unresolved_segments[index + 1 :]]
                    low = max(segment.min_length, remaining - sum(value.max_length for value in following))
                    high = min(segment.max_length, remaining - sum(value.min_length for value in following))
                    if low > high:
                        raise ValueError("placement chain lengths are incompatible with RfDiffusionPointGroupSymmetry")
                    length = draw(low, high)
                    sampled_lengths[placement_id] = length
                    remaining -= length

        # Any GeneratedSegment not pinned by a chain-length or symmetry coupling is drawn freely
        # within its own [min_length, max_length]; hoisted here (out of the reference-building loop
        # below) so the combined generated total is known before it is checked against the window.
        for group in placement.placements.values():
            for placement_id, part in group.items():
                if isinstance(part, GeneratedSegment) and placement_id not in sampled_lengths:
                    sampled_lengths[placement_id] = draw(part.min_length, part.max_length)
        return sampled_lengths

    # A GeneratedSegment always realizes within [min_length, max_length], so the combined generated
    # total is always within these summed bounds regardless of the per-chain / symmetry couplings;
    # if the requested window misses this envelope it can never be satisfied -- report that directly
    # instead of spinning the resampler to the cap.
    reachable_low = sum(segment.min_length for segment in segments_by_id.values())
    reachable_high = sum(segment.max_length for segment in segments_by_id.values())
    if combined_length is not None and (combined_length[1] < reachable_low or combined_length[0] > reachable_high):
        raise ValueError(
            f"combined_length {combined_length} cannot be realized: the placement's generated segments "
            f"admit a combined generated length in [{reachable_low}, {reachable_high}]"
        )
    max_attempts = 100000  # matches RFdiffusion get_sampled_mask
    sampled_lengths = sample_segment_lengths()
    if combined_length is not None:
        attempt = 1
        while not combined_length[0] <= sum(sampled_lengths[pid] for pid in segments_by_id) <= combined_length[1]:
            if attempt >= max_attempts:
                raise ValueError(
                    f"combined_length {combined_length} could not be satisfied after {max_attempts} "
                    f"resampling attempts of the placement's generated segment lengths"
                )
            sampled_lengths = sample_segment_lengths()
            attempt += 1

    output_residues: list[tuple[str, int] | SymbolicResidueSite] = []
    output_instances: list[str] = []
    # Mint canonical `chain_{n}` IDs in declaration order for nonempty groups.
    next_instance = 0
    for group in placement.placements.values():
        group_residues: list[tuple[str, int] | SymbolicResidueSite] = []
        for placement_id, part in group.items():
            if isinstance(part, ExistingSpan):
                existing = cast(ExistingSpan, part)
                group_references: list[tuple[str, int] | SymbolicResidueSite] = [
                    (existing.instance_id, index) for index in range(existing.start, existing.stop)
                ]
            else:
                length = sampled_lengths[placement_id]
                group_references = [SymbolicResidueSite(placement_id, index) for index in range(length)]
            group_residues.extend(group_references)
        if not group_residues:
            continue  # zero-length declaration group mints no realized chain, consumes no index
        instance_id = f"chain_{next_instance}"
        next_instance += 1
        output_residues.extend(group_residues)
        output_instances.extend([instance_id] * len(group_residues))

    realized_references = tuple(output_residues)
    instances = tuple(output_instances)
    if realized_references == input.layout.output_residues and instances == input.layout.output_instance_ids:
        return input
    if input.layout.templates:
        raise ValueError("variable generated lengths cannot be combined with aligned Template evidence")

    old_index = {reference: index for index, reference in enumerate(input.layout.output_residues)}
    length = len(realized_references)
    coords = input.noisy_coordinates.new_empty((1, length, 14, 3))
    residue_type_features = input.residue_type_features.new_zeros((1, length, 22))
    residue_type_features[..., RFDIFFUSION_MASK_TOKEN_ID] = 1
    fixed_structure = torch.zeros((1, length), dtype=torch.bool, device=input.noisy_coordinates.device)
    fixed_sequence = torch.zeros_like(fixed_structure)
    hotspots = torch.zeros_like(fixed_structure)
    for row, reference in enumerate(realized_references):
        previous = old_index.get(reference)
        if previous is None:
            coords[0, row] = float("nan")
            continue
        coords[0, row] = input.noisy_coordinates[0, previous]
        residue_type_features[0, row] = input.residue_type_features[0, previous]
        fixed_structure[0, row] = input.fixed_structure_mask[0, previous]
        fixed_sequence[0, row] = input.fixed_sequence_mask[0, previous]
        hotspots[0, row] = input.hotspot_mask[0, previous]
    for row, reference in enumerate(realized_references):
        if isinstance(reference, SymbolicResidueSite):
            residue_type_features[0, row].zero_()
            residue_type_features[0, row, RFDIFFUSION_MASK_TOKEN_ID] = 1
            fixed_sequence[0, row] = False
    residue_indices: list[int] = []
    previous_instance = None
    chain_index = -1
    chain_start = 0
    for row, instance_id in enumerate(instances):
        if instance_id != previous_instance:
            chain_index += 1
            chain_start = row
            previous_instance = instance_id
        residue_indices.append(200 * chain_index + row - chain_start)
    row_by_reference, symbolic_rows = _index_output_residues(realized_references)
    for condition in input.layout.conditions:
        if not isinstance(condition, AllowedResidueIdentities):
            continue
        identity_condition = cast(AllowedResidueIdentities, condition)
        token = rfdiffusion_residue_token_id(next(iter(identity_condition.component_ids)))
        identity_rows: list[int] = []
        for target in identity_condition.positions:
            if isinstance(target, str):
                identity_rows.extend(
                    row
                    for row, reference in enumerate(realized_references)
                    if isinstance(reference, SymbolicResidueSite)
                    and cast(SymbolicResidueSite, reference).placement_id == target
                )
            else:
                identity_rows.append(_output_residue_row(target, row_by_reference, symbolic_rows))
        for row in identity_rows:
            residue_type_features[0, row].zero_()
            residue_type_features[0, row, token] = 1
            fixed_sequence[0, row] = True
    secondary_labels = torch.full((1, length), 3, dtype=torch.int64, device=coords.device)
    label_by_kind = {"helix": 0, "strand": 1, "loop": 2}
    for condition in input.layout.conditions:
        if not isinstance(condition, RfDiffusionSecondaryStructure):
            continue
        secondary_condition = cast(RfDiffusionSecondaryStructure, condition)
        secondary_rows: list[int] = []
        for secondary_target in secondary_condition.targets:
            if isinstance(secondary_target, str):
                secondary_rows.extend(
                    row
                    for row, reference in enumerate(realized_references)
                    if isinstance(reference, SymbolicResidueSite)
                    and cast(SymbolicResidueSite, reference).placement_id == secondary_target
                )
            else:
                secondary_rows.append(
                    _output_residue_row(
                        cast(tuple[str, int] | SymbolicResidueSite, secondary_target),
                        row_by_reference,
                        symbolic_rows,
                    )
                )
        secondary_labels[0, torch.tensor(secondary_rows, device=coords.device)] = label_by_kind[
            secondary_condition.kind
        ]
    cyclic_mask = torch.zeros((1, length), dtype=torch.bool, device=coords.device)
    for condition in input.layout.conditions:
        if not isinstance(condition, SymbolicBond):
            continue
        bond = cast(SymbolicBond, condition)
        instance_id = rfdiffusion_terminal_cyclization_instance(
            bond,
            realized_references,
            instances,
        )
        cyclic_mask[0] |= torch.tensor(
            [value == instance_id for value in instances],
            device=coords.device,
        )
    return RfDiffusionInput(  # type: ignore[call-arg]
        noisy_coordinates=coords,
        residue_type_features=residue_type_features,
        residue_index=torch.tensor(residue_indices, dtype=torch.int64, device=coords.device)[None],
        fixed_structure_mask=fixed_structure,
        fixed_sequence_mask=fixed_sequence,
        hotspot_mask=hotspots,
        cyclic_mask=cyclic_mask,
        secondary_structure=F.one_hot(
            secondary_labels,
            num_classes=4,
        ).to(coords.dtype),
        block_adjacency=F.one_hot(
            torch.full((1, length, length), 2, dtype=torch.int64, device=coords.device),
            num_classes=3,
        ).to(coords.dtype),
        timestep=input.timestep,
        layout=RfDiffusionLayout(
            system=input.layout.system,
            structure=input.layout.structure,
            templates=input.layout.templates,
            placement=input.layout.placement,
            conditions=input.layout.conditions,
            output_residues=realized_references,
            output_instance_ids=instances,
            combined_length=input.layout.combined_length,
        ),
        batch_size=[coords.shape[0]],
    )


def _decode_design(
    samples: Iterable[_RfDiffusionSample],
    inputs: list[RfDiffusionInput],
    *,
    align_motif: bool,
    return_trajectory: bool,
) -> RfDiffusionDesignResult:
    # RFdiffusion is coordinate-producing and variable-topology. Each sample carries the decoded
    # Structure and per-sample values in RfDiffusionDesignCandidate/RfDiffusionDesignResult.
    candidates: list[RfDiffusionDesignCandidate] = []
    for sample, input in zip(samples, inputs, strict=True):
        coordinates = sample.coordinates.clone()
        fixed = input.fixed_structure_mask[0].to(device=coordinates.device, dtype=torch.bool)
        if bool(fixed.any()):
            reference = input.noisy_coordinates[0].to(device=coordinates.device, dtype=coordinates.dtype)
            if align_motif:
                coordinates = RfDiffusionSamplerStep.align_prediction_to_noisy_motif(coordinates, reference, fixed)
            coordinates[fixed] = reference[fixed]
        layout = input.layout
        candidate, _identity_map, residue_map = _decode_structure(
            coordinates,
            sample.sequence_tokens,
            input.fixed_sequence_mask[0],
            layout,
        )
        hard_condition_outcomes = _hard_condition_outcomes(candidate, layout, residue_map)
        _require_hard_conditions(layout, hard_condition_outcomes)
        # sample.plddt_fraction is [steps, L]; the final step is the per-residue design pLDDT,
        # aligned row-for-row to the output residues (residue_map keys are the ComponentReference
        # / SymbolicResidueSite identities, its values the realized output ComponentReferences).
        # The expectation lies on (0, 1] (bins linspace(1/num_bins, 1, num_bins)); the
        # Decode multiplies by 100 so the exported `plddt` value uses its 0-100 scale. This
        # mapping is applied only during decode and does not change the sampler or denoiser.
        plddt = sample.plddt_fraction[-1].detach().to(device="cpu", dtype=torch.float32).numpy() * 100.0
        plddt_axis = tuple(residue_map[reference] for reference in layout.output_residues)
        # Use the shared grouped placement_map as the per-candidate output identity. It is
        # reconstructed losslessly from output_residues and residue_map, with
        # exhaustive coverage enforced inside _materialize_placement_map). The realized ids in
        # its ComponentReference[0] values are this candidate's Structure.system instance ids.
        placement_condition = layout.placement
        placement_map = (
            {}
            if placement_condition is None
            else _materialize_placement_map(placement_condition, layout.output_residues, residue_map)[0]
        )
        candidates.append(
            RfDiffusionDesignCandidate(
                structure=candidate,
                placement_map=placement_map,
                scores={
                    "mean_plddt": Score(
                        value=float(plddt.mean()),
                        unit="0-100",
                        preferred="higher",
                    )
                },
                metrics={"plddt": Metric(values=plddt, axis=plddt_axis, unit="0-100")},
                trajectory=(
                    RfDiffusionTrajectory(
                        timesteps=sample.timesteps,
                        plddt_fraction=sample.plddt_fraction,
                        predicted_clean_coordinates=sample.predicted_clean_coordinates,
                        next_step_coordinates=sample.next_step_coordinates,
                    )
                    if return_trajectory
                    else None
                ),
            )
        )
    return RfDiffusionDesignResult(candidates=tuple(candidates))


def _decode_structure(
    coords: Tensor,
    sequence: Tensor,
    fixed_sequence_mask: Tensor,
    layout: RfDiffusionLayout,
) -> tuple[
    Structure,
    tuple[tuple[SymbolicResidueSite, tuple[str, int]], ...],
    dict[tuple[str, int] | SymbolicResidueSite, tuple[str, int]],
]:
    source = layout.system
    rows_by_instance: dict[str, list[tuple[int, tuple[str, int] | SymbolicResidueSite]]] = {}
    for row, (reference, instance_id) in enumerate(zip(layout.output_residues, layout.output_instance_ids)):
        rows_by_instance.setdefault(instance_id, []).append((row, reference))

    row_by_reference, symbolic_rows = _index_output_residues(layout.output_residues)
    allowed_references: set[tuple[str, int] | SymbolicResidueSite] = set()
    for condition in layout.conditions:
        if not isinstance(condition, AllowedResidueIdentities):
            continue
        identity_condition = cast(AllowedResidueIdentities, condition)
        for identity_target in identity_condition.positions:
            if isinstance(identity_target, str):
                for reference in layout.output_residues:
                    if (
                        isinstance(reference, SymbolicResidueSite)
                        and cast(SymbolicResidueSite, reference).placement_id == identity_target
                    ):
                        allowed_references.add(reference)
            else:
                residue_target = cast(
                    tuple[str, int] | SymbolicResidueSite,
                    identity_target,
                )
                if isinstance(residue_target, SymbolicResidueSite):
                    row = _output_residue_row(residue_target, row_by_reference, symbolic_rows)
                    allowed_references.add(layout.output_residues[row])
                else:
                    allowed_references.add(cast(tuple[str, int], residue_target))

    residue_map: dict[tuple[str, int] | SymbolicResidueSite, tuple[str, int]] = {}
    output_sequences: dict[str, tuple[ChemicalComponent, ...]] = {}
    for instance_id, instance_rows in rows_by_instance.items():
        residue_components: list[ChemicalComponent] = []
        for component_index, (row, reference) in enumerate(instance_rows):
            token = int(sequence[row])
            if (
                not isinstance(reference, SymbolicResidueSite)
                and bool(fixed_sequence_mask[row])
                and reference not in allowed_references
            ):
                source_system = cast(MolecularSystem, source)
                component_reference = cast(tuple[str, int], reference)
                entity = source_system.entities[source_system.instances[component_reference[0]]]
                residue_components.append(entity.components[component_reference[1]])
            else:
                residue_components.append(
                    get_component("UNK" if token == RFDIFFUSION_MASK_TOKEN_ID else RFDIFFUSION_RESIDUE_TYPES[token])
                )
            residue_map[reference] = (instance_id, component_index)
        output_sequences[instance_id] = tuple(residue_components)

    touched_instances = {reference[0] for reference in layout.output_residues if isinstance(reference, tuple)}
    retained_instance_ids = (
        []
        if source is None
        else [instance_id for instance_id in source.instances if instance_id not in touched_instances]
    )
    entities: dict[str, PolymerSequence | Molecule] = {}
    instances: dict[str, str] = {}
    entity_ids: set[str] = set()
    if source is not None:
        for instance_id in retained_instance_ids:
            entity_id = source.instances[instance_id]
            entities[entity_id] = source.entities[entity_id]
            instances[instance_id] = entity_id
            entity_ids.add(entity_id)
    # Group instances by their realized sequence value so symmetric copies (which share one
    # sequence) collapse to a single entity instantiated N times -- the canonical homo-oligomer
    # representation -- instead of one distinct entity per copy (a spurious heterotrimer).
    # Genuinely divergent chains still resolve to separate entities.
    instances_by_sequence: dict[PolymerSequence, list[str]] = {}
    for instance_id, sequence_components in output_sequences.items():
        realized = PolymerSequence("protein", sequence_components)
        instances_by_sequence.setdefault(realized, []).append(instance_id)
    for realized, grouped_instance_ids in instances_by_sequence.items():
        base = f"{grouped_instance_ids[0]}_rfdesign"
        entity_id = base
        suffix = 2
        while entity_id in entity_ids:
            entity_id = f"{base}_{suffix}"
            suffix += 1
        entity_ids.add(entity_id)
        entities[entity_id] = realized
        for instance_id in grouped_instance_ids:
            instances[instance_id] = entity_id

    valid_instance_ids = set(instances)
    bonds: set[SystemBond] = set()
    if source is not None:
        for bond in source.bonds:
            remapped = _remap_bond(bond, residue_map, touched_instances)
            if remapped.site_a.instance_id in valid_instance_ids and remapped.site_b.instance_id in valid_instance_ids:
                bonds.add(remapped)
    for condition in layout.conditions:
        if not isinstance(condition, SymbolicBond):
            continue
        symbolic_bond = cast(SymbolicBond, condition)
        endpoints: list[AtomSite] = []
        for endpoint in (symbolic_bond.site_a, symbolic_bond.site_b):
            if isinstance(endpoint, SymbolicAtomSite):
                symbolic_endpoint = cast(SymbolicAtomSite, endpoint)
                reference = layout.output_residues[
                    _output_residue_row(
                        symbolic_endpoint.residue,
                        row_by_reference,
                        symbolic_rows,
                    )
                ]
                instance_id, component_index = residue_map[reference]
                endpoints.append(AtomSite(instance_id, component_index, symbolic_endpoint.atom_id))
            else:
                atom_endpoint = cast(AtomSite, endpoint)
                mapped_residue = residue_map.get((atom_endpoint.instance_id, atom_endpoint.component_index))
                endpoints.append(
                    atom_endpoint
                    if mapped_residue is None
                    else AtomSite(mapped_residue[0], mapped_residue[1], atom_endpoint.atom_id)
                )
        bonds.add(SystemBond(endpoints[0], endpoints[1], origin="user"))
    system = MolecularSystem(entities, instances, frozenset(bonds))

    sites: list[AtomSite] = []
    coordinate_rows: list[np.ndarray] = []
    masks: list[bool] = []
    coords_cpu = coords.detach().to(device="cpu", dtype=torch.float32).numpy()
    for row, reference in enumerate(layout.output_residues):
        instance_id, component_index = residue_map[reference]
        for atom_index, atom_name in enumerate(("N", "CA", "C", "O")):
            sites.append(AtomSite(instance_id, component_index, atom_name))
            value = np.asarray(coords_cpu[row, atom_index], dtype=np.float32)
            present = bool(np.isfinite(value).all())
            coordinate_rows.append(value if present else np.zeros(3, dtype=np.float32))
            masks.append(present)

    site_set = set(sites)
    if layout.structure is not None:
        source_index = {site: index for index, site in enumerate(layout.structure.sites)}
        for condition in layout.conditions:
            if not isinstance(condition, RfDiffusionReferenceCoordinates):
                continue
            reference_condition = cast(RfDiffusionReferenceCoordinates, condition)
            for original in reference_condition.sites:
                mapped_residue = residue_map[(original.instance_id, original.component_index)]
                mapped_site = AtomSite(mapped_residue[0], mapped_residue[1], original.atom_id)
                if mapped_site in site_set:
                    continue
                source_row = source_index[original]
                sites.append(mapped_site)
                site_set.add(mapped_site)
                coordinate_rows.append(np.asarray(layout.structure.coordinates[source_row], dtype=np.float32))
                masks.append(True)

    if layout.structure is not None and source is not None:
        retained_ids = set(retained_instance_ids)
        for index, site in enumerate(layout.structure.sites):
            if site.instance_id not in retained_ids or site in sites:
                continue
            sites.append(site)
            coordinate_rows.append(np.asarray(layout.structure.coordinates[index], dtype=np.float32))
            masks.append(bool(layout.structure.coordinate_mask[index]))
    structure = Structure(
        system=system,
        sites=tuple(sites),
        coordinates=np.asarray(coordinate_rows, dtype=np.float32),
        coordinate_mask=np.asarray(masks, dtype=np.bool_),
    )
    identity_map = tuple(
        (reference, residue_map[reference])
        for reference in layout.output_residues
        if isinstance(reference, SymbolicResidueSite)
    )
    return structure, identity_map, residue_map


def _require_hard_conditions(
    layout: RfDiffusionLayout,
    outcomes: tuple[bool | None, ...],
) -> None:
    for index, (condition, outcome) in enumerate(zip(layout.conditions, outcomes, strict=True)):
        is_hard = isinstance(
            condition,
            (
                AllowedResidueIdentities,
                RfDiffusionReferenceCoordinates,
                RfDiffusionPointGroupSymmetry,
                SymbolicBond,
            ),
        )
        # Raise only on a MEASURED hard violation. An unmeasured outcome is ``None`` -- the
        # conditioning-only Hotspot biases sampling through hotspot_mask but its satisfaction is
        # never verified post-hoc, so it cannot be asserted as a violation. Every hard
        # condition and the measured records (AllowedResidueIdentities / SymbolicBond) report
        # a concrete bool here, so ``is False`` preserves their existing raise behavior exactly.
        if is_hard and outcome is False:
            raise RuntimeError(f"RFdiffusion violated hard conditions[{index}] during result decoding")


def _remap_bond(
    bond: SystemBond,
    residue_map: Mapping[tuple[str, int] | SymbolicResidueSite, tuple[str, int]],
    touched_instances: set[str],
) -> SystemBond:
    def site(value: AtomSite) -> AtomSite:
        if value.instance_id not in touched_instances:
            return value
        mapped = residue_map.get((value.instance_id, value.component_index))
        if mapped is None:
            raise ValueError("placement omits a residue participating in an input system bond")
        return AtomSite(mapped[0], mapped[1], value.atom_id)

    stereo_sites = None if bond.stereo_sites is None else (site(bond.stereo_sites[0]), site(bond.stereo_sites[1]))
    return SystemBond(
        site_a=site(bond.site_a),
        site_b=site(bond.site_b),
        order=bond.order,
        aromatic=bond.aromatic,
        stereo=bond.stereo,
        stereo_sites=stereo_sites,
        origin=bond.origin,
    )


def _hard_condition_outcomes(
    candidate: Structure,
    layout: RfDiffusionLayout,
    residue_map: Mapping[tuple[str, int] | SymbolicResidueSite, tuple[str, int]],
) -> tuple[bool | None, ...]:
    coordinate_index = {site: index for index, site in enumerate(candidate.sites)}
    row_by_reference, symbolic_rows = _index_output_residues(layout.output_residues)
    source_index = (
        {} if layout.structure is None else {site: index for index, site in enumerate(layout.structure.sites)}
    )
    outcomes: list[bool | None] = []
    for condition in layout.conditions:
        if isinstance(condition, AllowedResidueIdentities):
            identity_condition = cast(AllowedResidueIdentities, condition)
            references: list[tuple[str, int] | SymbolicResidueSite] = []
            for identity_target in identity_condition.positions:
                if isinstance(identity_target, str):
                    references.extend(
                        reference
                        for reference in layout.output_residues
                        if isinstance(reference, SymbolicResidueSite)
                        and cast(SymbolicResidueSite, reference).placement_id == identity_target
                    )
                else:
                    reference = cast(tuple[str, int] | SymbolicResidueSite, identity_target)
                    if isinstance(reference, SymbolicResidueSite):
                        reference = layout.output_residues[
                            _output_residue_row(
                                reference,
                                row_by_reference,
                                symbolic_rows,
                            )
                        ]
                    references.append(reference)
            component_ids = []
            for reference in references:
                instance_id, component_index = residue_map[reference]
                entity = candidate.system.entities[candidate.system.instances[instance_id]]
                component_ids.append(entity.components[component_index].component_id)
            outcomes.append(all(component_id in identity_condition.component_ids for component_id in component_ids))
            continue
        if isinstance(condition, RfDiffusionReferenceCoordinates) and layout.structure is not None:
            reference_condition = cast(RfDiffusionReferenceCoordinates, condition)
            original_coordinates: list[np.ndarray] = []
            candidate_coordinates: list[np.ndarray] = []
            complete = True
            for original in reference_condition.sites:
                system = cast(MolecularSystem, layout.system)
                entity = system.entities[system.instances[original.instance_id]]
                if isinstance(entity, Molecule):
                    mapped_site: AtomSite | None = original
                else:
                    mapped_residue = residue_map.get((original.instance_id, original.component_index))
                    mapped_site = (
                        None
                        if mapped_residue is None
                        else AtomSite(mapped_residue[0], mapped_residue[1], original.atom_id)
                    )
                old_index = source_index.get(original)
                new_index = None if mapped_site is None else coordinate_index.get(mapped_site)
                if (
                    old_index is None
                    or new_index is None
                    or not bool(layout.structure.coordinate_mask[old_index])
                    or not bool(candidate.coordinate_mask[new_index])
                ):
                    complete = False
                    break
                original_coordinates.append(layout.structure.coordinates[old_index])
                candidate_coordinates.append(candidate.coordinates[new_index])
            distances = (
                _proper_alignment_distances(
                    np.asarray(candidate_coordinates, dtype=np.float64),
                    np.asarray(original_coordinates, dtype=np.float64),
                )
                if complete
                else None
            )
            outcomes.append(distances is not None and float(distances.max()) <= reference_condition.tolerance_angstrom)
            continue
        if isinstance(condition, RfDiffusionPointGroupSymmetry):
            symmetry_condition = cast(RfDiffusionPointGroupSymmetry, condition)
            symmetry = RfDiffusionSymmetry(symmetry_condition, layout)
            units: list[np.ndarray] = []
            complete = True
            for unit_rows in symmetry.units:
                unit = []
                for row in unit_rows:
                    output_residue = residue_map[layout.output_residues[row]]
                    residue_atoms = []
                    for atom_name in ("N", "CA", "C"):
                        coordinate = coordinate_index.get(AtomSite(output_residue[0], output_residue[1], atom_name))
                        if coordinate is None or not bool(candidate.coordinate_mask[coordinate]):
                            complete = False
                            break
                        residue_atoms.append(candidate.coordinates[coordinate])
                    if not complete:
                        break
                    unit.append(residue_atoms)
                if not complete:
                    break
                units.append(np.asarray(unit, dtype=np.float64))
            if complete:
                observed = np.asarray(units, dtype=np.float64)
                rotations = np.stack([rotation.cpu().numpy() for rotation in symmetry.rotations])
                rmsd = _point_group_rmsd(observed, rotations)
            else:
                rmsd = None
            outcomes.append(rmsd is not None and rmsd <= symmetry_condition.tolerance_angstrom)
            continue
        if isinstance(condition, SymbolicBond):
            symbolic_bond = cast(SymbolicBond, condition)
            endpoint_sites: list[AtomSite] = []
            for endpoint in (symbolic_bond.site_a, symbolic_bond.site_b):
                if isinstance(endpoint, SymbolicAtomSite):
                    symbolic_endpoint = cast(SymbolicAtomSite, endpoint)
                    reference = layout.output_residues[
                        _output_residue_row(
                            symbolic_endpoint.residue,
                            row_by_reference,
                            symbolic_rows,
                        )
                    ]
                    mapped_residue = residue_map[reference]
                    endpoint_sites.append(AtomSite(mapped_residue[0], mapped_residue[1], symbolic_endpoint.atom_id))
                else:
                    atom_endpoint = cast(AtomSite, endpoint)
                    mapped_residue = residue_map.get((atom_endpoint.instance_id, atom_endpoint.component_index))
                    endpoint_sites.append(
                        atom_endpoint
                        if mapped_residue is None
                        else AtomSite(mapped_residue[0], mapped_residue[1], atom_endpoint.atom_id)
                    )
            bond = any({value.site_a, value.site_b} == set(endpoint_sites) for value in candidate.system.bonds)
            outcomes.append(bond)
            continue
        outcomes.append(None)
    return tuple(outcomes)


def _proper_alignment_distances(moving: np.ndarray, target: np.ndarray) -> np.ndarray:
    if np.array_equal(moving, target):
        return np.zeros(len(moving), dtype=np.float64)
    moving_centered = moving - moving.mean(axis=0)
    target_centered = target - target.mean(axis=0)
    left, _singular, right = np.linalg.svd(moving_centered.T @ target_centered)
    rotation = left @ right
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right
    aligned = moving_centered @ rotation + target.mean(axis=0)
    return np.linalg.norm(aligned - target, axis=-1)


def _point_group_rmsd(observed: np.ndarray, rotations: np.ndarray) -> float:
    center = observed.reshape(-1, 3).mean(axis=0)

    def residual(parameters: np.ndarray) -> np.ndarray:
        frame = Rotation.from_rotvec(parameters[:3]).as_matrix()
        origin = parameters[3:]
        base = observed[0] - origin
        predicted = np.stack([base @ frame @ rotation.T @ frame.T + origin for rotation in rotations])
        return (predicted - observed).reshape(-1)

    starts = (
        np.zeros(3),
        np.asarray([math.pi / 2, 0.0, 0.0]),
        np.asarray([0.0, math.pi / 2, 0.0]),
        np.asarray([0.0, 0.0, math.pi / 2]),
        np.asarray([math.pi, 0.0, 0.0]),
        np.asarray([0.0, math.pi, 0.0]),
        np.asarray([0.0, 0.0, math.pi]),
    )
    fitted = least_squares(residual, np.concatenate((starts[0], center)))
    best = residual(fitted.x).reshape(-1, 3)
    for frame in starts[1:]:
        fitted = least_squares(residual, np.concatenate((frame, center)))
        values = residual(fitted.x).reshape(-1, 3)
        if np.square(values).sum() < np.square(best).sum():
            best = values
    return float(np.sqrt(np.square(best).sum(axis=-1).mean()))


RFDIFFUSION_RESIDUE_TYPES = (
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
    "UNK",
    "MAS",
)
RFDIFFUSION_RESIDUE_TYPE_INDEX = {name: index for index, name in enumerate(RFDIFFUSION_RESIDUE_TYPES)}
RFDIFFUSION_UNKNOWN_TOKEN_ID = RFDIFFUSION_RESIDUE_TYPE_INDEX["UNK"]
RFDIFFUSION_MASK_TOKEN_ID = RFDIFFUSION_RESIDUE_TYPE_INDEX["MAS"]

RFDIFFUSION_ATOM14_NAMES = (
    ("N", "CA", "C", "O", "CB", None, None, None, None, None, None, None, None, None),
    ("N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2", None, None, None),
    ("N", "CA", "C", "O", "CB", "CG", "OD1", "ND2", None, None, None, None, None, None),
    ("N", "CA", "C", "O", "CB", "CG", "OD1", "OD2", None, None, None, None, None, None),
    ("N", "CA", "C", "O", "CB", "SG", None, None, None, None, None, None, None, None),
    ("N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2", None, None, None, None, None),
    ("N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2", None, None, None, None, None),
    ("N", "CA", "C", "O", None, None, None, None, None, None, None, None, None, None),
    ("N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2", None, None, None, None),
    ("N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1", None, None, None, None, None, None),
    ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", None, None, None, None, None, None),
    ("N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ", None, None, None, None, None),
    ("N", "CA", "C", "O", "CB", "CG", "SD", "CE", None, None, None, None, None, None),
    ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", None, None, None),
    ("N", "CA", "C", "O", "CB", "CG", "CD", None, None, None, None, None, None, None),
    ("N", "CA", "C", "O", "CB", "OG", None, None, None, None, None, None, None, None),
    ("N", "CA", "C", "O", "CB", "OG1", "CG2", None, None, None, None, None, None, None),
    ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"),
    ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH", None, None),
    ("N", "CA", "C", "O", "CB", "CG1", "CG2", None, None, None, None, None, None, None),
    ("N", "CA", "C", "O", "CB", None, None, None, None, None, None, None, None, None),
    ("N", "CA", "C", "O", "CB", None, None, None, None, None, None, None, None, None),
)

_CHI_ATOM_NAMES: tuple[tuple[tuple[str, str, str, str] | None, ...], ...] = (
    (None, None, None, None),
    (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"), ("CB", "CG", "CD", "NE"), ("CG", "CD", "NE", "CZ")),
    (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "OD1"), None, None),
    (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "OD1"), None, None),
    (("N", "CA", "CB", "SG"), None, None, None),
    (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"), ("CB", "CG", "CD", "OE1"), None),
    (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"), ("CB", "CG", "CD", "OE1"), None),
    (None, None, None, None),
    (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "ND1"), None, None),
    (("N", "CA", "CB", "CG1"), ("CA", "CB", "CG1", "CD1"), None, None),
    (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1"), None, None),
    (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"), ("CB", "CG", "CD", "CE"), ("CG", "CD", "CE", "NZ")),
    (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "SD"), ("CB", "CG", "SD", "CE"), None),
    (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1"), None, None),
    (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"), None, None),
    (("N", "CA", "CB", "OG"), None, None, None),
    (("N", "CA", "CB", "OG1"), None, None, None),
    (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1"), None, None),
    (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1"), None, None),
    (("N", "CA", "CB", "CG1"), None, None, None),
    (None, None, None, None),
    (None, None, None, None),
)

_INITIAL_COORDINATES = torch.full((27, 3), float("nan"))
_INITIAL_COORDINATES[0] = torch.tensor([-0.5272, 1.3593, 0.0000])
_INITIAL_COORDINATES[1] = torch.tensor([0.0000, 0.0000, 0.0000])
_INITIAL_COORDINATES[2] = torch.tensor([1.5233, 0.0000, 0.0000])
_INITIAL_COORDINATES[3] = torch.tensor([2.0505, 1.3593, 0.0000])
_INITIAL_COORDINATES[4] = torch.tensor([-0.5291, -0.7749, -1.2056])


def rfdiffusion_residue_token_id(residue_name: str) -> int:
    r"""
    Map a three-letter residue name to its RFdiffusion integer id.

    Lookup is case-insensitive; unrecognized names fall back to the unknown token so parsing never fails on a
    non-standard residue.

    Args:
        residue_name: Three-letter residue name (e.g. `"ALA"`); case is ignored.

    Returns:
        The residue id from
        [`RFDIFFUSION_RESIDUE_TYPE_INDEX`][ofoldx.models.rfdiffusion.model.RFDIFFUSION_RESIDUE_TYPE_INDEX],
        or ``RFDIFFUSION_UNKNOWN_TOKEN_ID`` for an unrecognized name.
    """

    return RFDIFFUSION_RESIDUE_TYPE_INDEX.get(residue_name.upper(), RFDIFFUSION_UNKNOWN_TOKEN_ID)


_RF_IDEAL_ATOM14_COORDINATES: tuple[
    tuple[tuple[str, int, tuple[float, float, float]], ...],
    ...,
] = (
    (  # ALA
        ("N", 0, (-0.5272, 1.3593, 0.0)),
        ("CA", 0, (0.0, 0.0, 0.0)),
        ("C", 0, (1.5233, 0.0, 0.0)),
        ("O", 3, (0.6303, 1.0574, 0.0)),
        ("CB", 8, (-0.5289, -0.7734, -1.1991)),
    ),
    (  # ARG
        ("N", 0, (-0.5272, 1.3593, 0.0)),
        ("CA", 0, (0.0, 0.0, 0.0)),
        ("C", 0, (1.5233, 0.0, 0.0)),
        ("O", 3, (0.6303, 1.0574, 0.0)),
        ("CB", 8, (-0.5042, -0.7698, -1.2118)),
        ("CG", 4, (0.6396, 1.3794, 0.0)),
        ("CD", 5, (0.5492, 1.3801, 0.0)),
        ("NE", 6, (0.5423, 1.3491, 0.0)),
        ("CZ", 7, (0.7650, 1.1090, 0.0)),
        ("NH1", 7, (0.2012, 2.2965, 0.0)),
        ("NH2", 7, (2.0824, 1.0030, 0.0)),
    ),
    (  # ASN
        ("N", 0, (-0.5272, 1.3593, 0.0)),
        ("CA", 0, (0.0, 0.0, 0.0)),
        ("C", 0, (1.5233, 0.0, 0.0)),
        ("O", 3, (0.6303, 1.0574, 0.0)),
        ("CB", 8, (-0.5341, -0.7799, -1.1874)),
        ("CG", 4, (0.5778, 1.3881, 0.0)),
        ("OD1", 5, (0.6331, 1.0620, 0.0)),
        ("ND2", 5, (0.5839, -1.1711, 0.0)),
    ),
    (  # ASP
        ("N", 0, (-0.5272, 1.3593, 0.0)),
        ("CA", 0, (0.0, 0.0, 0.0)),
        ("C", 0, (1.5233, 0.0, 0.0)),
        ("O", 3, (0.6303, 1.0574, 0.0)),
        ("CB", 8, (-0.5162, -0.7757, -1.2144)),
        ("CG", 4, (0.5926, 1.4028, 0.0)),
        ("OD1", 5, (0.5746, 1.0629, 0.0)),
        ("OD2", 5, (0.5738, -1.0627, 0.0)),
    ),
    (  # CYS
        ("N", 0, (-0.5272, 1.3593, 0.0)),
        ("CA", 0, (0.0, 0.0, 0.0)),
        ("C", 0, (1.5233, 0.0, 0.0)),
        ("O", 3, (0.6303, 1.0574, 0.0)),
        ("CB", 8, (-0.5046, -0.7727, -1.2189)),
        ("SG", 4, (0.7386, 1.6511, 0.0)),
    ),
    (  # GLN
        ("N", 0, (-0.5272, 1.3593, 0.0)),
        ("CA", 0, (0.0, 0.0, 0.0)),
        ("C", 0, (1.5233, 0.0, 0.0)),
        ("O", 3, (0.6303, 1.0574, 0.0)),
        ("CB", 8, (-0.5226, -0.7776, -1.2109)),
        ("CG", 4, (0.6225, 1.3857, 0.0)),
        ("CD", 5, (0.5788, 1.4021, 0.0)),
        ("OE1", 6, (0.6347, 1.0584, 0.0)),
        ("NE2", 6, (0.5908, -1.1895, 0.0)),
    ),
    (  # GLU
        ("N", 0, (-0.5272, 1.3593, 0.0)),
        ("CA", 0, (0.0, 0.0, 0.0)),
        ("C", 0, (1.5233, 0.0, 0.0)),
        ("O", 3, (0.6303, 1.0574, 0.0)),
        ("CB", 8, (-0.5197, -0.7737, -1.2137)),
        ("CG", 4, (0.6287, 1.3862, 0.0)),
        ("CD", 5, (0.5850, 1.3849, 0.0)),
        ("OE1", 6, (0.5752, 1.0618, 0.0)),
        ("OE2", 6, (0.5741, -1.0635, 0.0)),
    ),
    (  # GLY
        ("N", 0, (-0.5272, 1.3593, 0.0)),
        ("CA", 0, (0.0, 0.0, 0.0)),
        ("C", 0, (1.5233, 0.0, 0.0)),
        ("O", 3, (0.6303, 1.0574, 0.0)),
    ),
    (  # HIS
        ("N", 0, (-0.5272, 1.3593, 0.0)),
        ("CA", 0, (0.0, 0.0, 0.0)),
        ("C", 0, (1.5233, 0.0, 0.0)),
        ("O", 3, (0.6303, 1.0574, 0.0)),
        ("CB", 8, (-0.5163, -0.7809, -1.2129)),
        ("CG", 4, (0.6016, 1.3710, 0.0)),
        ("ND1", 6, (-1.8631, -1.0722, 0.0)),
        ("CD2", 5, (0.8918, -1.0184, 0.0)),
        ("CE1", 5, (2.0299, 0.8564, 0.0)),
        ("NE2", 6, (-1.8625, 1.0707, 0.0)),
    ),
    (  # ILE
        ("N", 0, (-0.5272, 1.3593, 0.0)),
        ("CA", 0, (0.0, 0.0, 0.0)),
        ("C", 0, (1.5233, 0.0, 0.0)),
        ("O", 3, (0.6303, 1.0574, 0.0)),
        ("CB", 8, (-0.5140, -0.7885, -1.2184)),
        ("CG1", 4, (0.5339, 1.4348, 0.0)),
        ("CG2", 4, (0.5319, -0.7693, -1.1994)),
        ("CD1", 5, (0.6106, 1.3829, 0.0)),
    ),
    (  # LEU
        ("N", 0, (-0.5272, 1.3593, 0.0)),
        ("CA", 0, (0.0, 0.0, 0.0)),
        ("C", 0, (1.525, -0.0, -0.0)),
        ("O", 3, (0.6303, 1.0574, 0.0)),
        ("CB", 8, (-0.5175, -0.7692, -1.2220)),
        ("CG", 4, (0.6652, 1.3823, 0.0)),
        ("CD1", 5, (0.5083, 1.4353, 0.0)),
        ("CD2", 5, (0.5079, -0.7600, 1.2163)),
    ),
    (  # LYS
        ("N", 0, (-0.5272, 1.3593, 0.0)),
        ("CA", 0, (0.0, 0.0, 0.0)),
        ("C", 0, (1.5233, 0.0, 0.0)),
        ("O", 3, (0.6303, 1.0574, 0.0)),
        ("CB", 8, (-0.5259, -0.7785, -1.2069)),
        ("CG", 4, (0.6291, 1.3869, 0.0)),
        ("CD", 5, (0.5526, 1.4174, 0.0)),
        ("CE", 6, (0.5544, 1.4170, 0.0)),
        ("NZ", 7, (0.5566, 1.3801, 0.0)),
    ),
    (  # MET
        ("N", 0, (-0.5272, 1.3593, 0.0)),
        ("CA", 0, (0.0, 0.0, 0.0)),
        ("C", 0, (1.5233, 0.0, 0.0)),
        ("O", 3, (0.6303, 1.0574, 0.0)),
        ("CB", 8, (-0.5331, -0.7727, -1.2048)),
        ("CG", 4, (0.6298, 1.3858, 0.0)),
        ("SD", 5, (0.6953, 1.6645, 0.0)),
        ("CE", 6, (0.3383, 1.7581, 0.0)),
    ),
    (  # PHE
        ("N", 0, (-0.5272, 1.3593, 0.0)),
        ("CA", 0, (0.0, 0.0, 0.0)),
        ("C", 0, (1.5233, 0.0, 0.0)),
        ("O", 3, (0.6303, 1.0574, 0.0)),
        ("CB", 8, (-0.5150, -0.7729, -1.2156)),
        ("CG", 4, (0.6060, 1.3746, 0.0)),
        ("CD1", 5, (0.7078, 1.1928, 0.0)),
        ("CD2", 5, (0.7084, -1.1920, 0.0)),
        ("CE1", 5, (2.0900, 1.1940, 0.0)),
        ("CE2", 5, (2.0897, -1.1939, 0.0)),
        ("CZ", 5, (2.7809, 0.0, 0.0)),
    ),
    (  # PRO
        ("N", 0, (-0.5272, 1.3593, 0.0)),
        ("CA", 0, (0.0, 0.0, 0.0)),
        ("C", 0, (1.5233, 0.0, 0.0)),
        ("O", 3, (0.6303, 1.0574, 0.0)),
        ("CB", 8, (-0.5649, -0.5888, -1.2966)),
        ("CG", 4, (0.3657, 1.4451, 0.0)),
        ("CD", 5, (0.3744, 1.4582, 0.0)),
    ),
    (  # SER
        ("N", 0, (-0.5272, 1.3593, 0.0)),
        ("CA", 0, (0.0, 0.0, 0.0)),
        ("C", 0, (1.5233, 0.0, 0.0)),
        ("O", 3, (0.6303, 1.0574, 0.0)),
        ("CB", 8, (-0.5146, -0.7595, -1.2073)),
        ("OG", 4, (0.5021, 1.3081, 0.0)),
    ),
    (  # THR
        ("N", 0, (-0.5272, 1.3593, 0.0)),
        ("CA", 0, (0.0, 0.0, 0.0)),
        ("C", 0, (1.5233, 0.0, 0.0)),
        ("O", 3, (0.6303, 1.0574, 0.0)),
        ("CB", 8, (-0.5172, -0.7952, -1.2130)),
        ("OG1", 4, (0.4804, 1.3506, 0.0)),
        ("CG2", 4, (0.5334, -0.7239, -1.2267)),
    ),
    (  # TRP
        ("N", 0, (-0.5272, 1.3593, 0.0)),
        ("CA", 0, (0.0, 0.0, 0.0)),
        ("C", 0, (1.5233, 0.0, 0.0)),
        ("O", 3, (0.6303, 1.0574, 0.0)),
        ("CB", 8, (-0.5136, -0.7712, -1.2173)),
        ("CG", 4, (0.5984, 1.3741, 0.0)),
        ("CD1", 5, (0.8151, 1.0921, 0.0)),
        ("CD2", 5, (0.8753, -1.1538, 0.0)),
        ("NE1", 5, (2.1309, 0.7003, 0.0)),
        ("CE2", 5, (2.1865, -0.6707, 0.0)),
        ("CE3", 5, (0.6541, -2.5366, 0.0)),
        ("CZ2", 5, (3.2813, -1.5205, 0.0)),
        ("CZ3", 5, (1.7521, -3.3888, 0.0)),
        ("CH2", 5, (3.0315, -2.8930, 0.0)),
    ),
    (  # TYR
        ("N", 0, (-0.5272, 1.3593, 0.0)),
        ("CA", 0, (0.0, 0.0, 0.0)),
        ("C", 0, (1.5233, 0.0, 0.0)),
        ("O", 3, (0.6303, 1.0574, 0.0)),
        ("CB", 8, (-0.5305, -0.7799, -1.2051)),
        ("CG", 4, (0.6104, 1.3840, 0.0)),
        ("CD1", 5, (0.6936, 1.2013, 0.0)),
        ("CD2", 5, (0.6934, -1.2011, 0.0)),
        ("CE1", 5, (2.0751, 1.2013, 0.0)),
        ("CE2", 5, (2.0748, -1.2011, 0.0)),
        ("CZ", 5, (2.7648, 0.0, 0.0)),
        ("OH", 5, (4.1408, 0.0, 0.0)),
    ),
    (  # VAL
        ("N", 0, (-0.5272, 1.3593, 0.0)),
        ("CA", 0, (0.0, 0.0, 0.0)),
        ("C", 0, (1.5233, 0.0, 0.0)),
        ("O", 3, (0.6303, 1.0574, 0.0)),
        ("CB", 8, (-0.5105, -0.7712, -1.2317)),
        ("CG1", 4, (0.5326, 1.4252, 0.0)),
        ("CG2", 4, (0.5177, -0.7693, 1.2057)),
    ),
    (  # UNK
        ("N", 0, (-0.5272, 1.3593, 0.0)),
        ("CA", 0, (0.0, 0.0, 0.0)),
        ("C", 0, (1.5233, 0.0, 0.0)),
        ("O", 3, (0.6303, 1.0574, 0.0)),
        ("CB", 8, (-0.5289, -0.7734, -1.1991)),
    ),
    (  # MAS
        ("N", 0, (-0.5272, 1.3593, 0.0)),
        ("CA", 0, (0.0, 0.0, 0.0)),
        ("C", 0, (1.5233, 0.0, 0.0)),
        ("O", 3, (0.6303, 1.0574, 0.0)),
        ("CB", 8, (-0.5289, -0.7734, -1.1991)),
    ),
)


def _rfdiffusion_base_frame(x_axis: Tensor, y_axis: Tensor) -> Tensor:
    x_axis = x_axis / torch.linalg.norm(x_axis)
    y_axis = y_axis - torch.dot(y_axis, x_axis) * x_axis
    y_axis = y_axis / torch.linalg.norm(y_axis)
    z_axis = torch.cross(x_axis, y_axis, dim=0)
    z_axis = z_axis / torch.linalg.norm(z_axis)
    return torch.stack((x_axis, y_axis, z_axis), dim=-1)


def _build_rfdiffusion_atom14_kinematics() -> tuple[Tensor, Tensor, Tensor]:
    residue_count = len(RFDIFFUSION_RESIDUE_TYPES)
    frame_indices = torch.zeros((residue_count, 14), dtype=torch.long, device="cpu")
    base_frame_coordinates = torch.ones((residue_count, 14, 4), dtype=torch.float32, device="cpu")
    base_frame_transforms = torch.eye(4, dtype=torch.float32, device="cpu").repeat(residue_count, 7, 1, 1)

    for residue_index, coordinates in enumerate(_RF_IDEAL_ATOM14_COORDINATES):
        atom_names = RFDIFFUSION_ATOM14_NAMES[residue_index]
        for atom_name, frame_index, position in coordinates:
            atom_index = atom_names.index(atom_name)
            frame_indices[residue_index, atom_index] = frame_index
            base_frame_coordinates[residue_index, atom_index, :3] = base_frame_coordinates.new_tensor(position)

        nitrogen = base_frame_coordinates[residue_index, atom_names.index("N"), :3]
        alpha_carbon = base_frame_coordinates[residue_index, atom_names.index("CA"), :3]
        carbon = base_frame_coordinates[residue_index, atom_names.index("C"), :3]
        base_frame_transforms[residue_index, 1, :3, :3] = _rfdiffusion_base_frame(
            nitrogen - alpha_carbon,
            base_frame_coordinates.new_tensor((1.0, 0.0, 0.0)),
        )
        base_frame_transforms[residue_index, 1, :3, 3] = nitrogen
        base_frame_transforms[residue_index, 2, :3, :3] = _rfdiffusion_base_frame(
            carbon - alpha_carbon,
            alpha_carbon - nitrogen,
        )
        base_frame_transforms[residue_index, 2, :3, 3] = carbon

        for chi_index, chi_atoms in enumerate(_CHI_ATOM_NAMES[residue_index]):
            if chi_atoms is None:
                continue
            atom_indices = tuple(atom_names.index(atom_name) for atom_name in chi_atoms[:3])
            first, second, third = (base_frame_coordinates[residue_index, index, :3] for index in atom_indices)
            if chi_index == 0:
                rotation = _rfdiffusion_base_frame(third - second, first - second)
            else:
                rotation = _rfdiffusion_base_frame(
                    third,
                    base_frame_coordinates.new_tensor((-1.0, 0.0, 0.0)),
                )
            base_frame_transforms[residue_index, 3 + chi_index, :3, :3] = rotation
            base_frame_transforms[residue_index, 3 + chi_index, :3, 3] = third

    # These released protonation/ring-closure torsions do not add atom14 sites,
    # but their frames remain observable in the reconstruction return value.
    for residue_name, chi_index, frame_atoms, special_case in (
        ("CYS", 1, ("CA", "CB", "SG"), False),
        ("PRO", 2, ("CB", "CG", "CD"), False),
        ("SER", 1, ("CA", "CB", "OG"), False),
        ("THR", 1, ("CA", "CB", "OG1"), False),
        ("TYR", 2, ("CE1", "CZ", "OH"), True),
    ):
        residue_index = RFDIFFUSION_RESIDUE_TYPE_INDEX[residue_name]
        atom_names = RFDIFFUSION_ATOM14_NAMES[residue_index]
        coordinates = base_frame_coordinates[residue_index]
        first, second, third = (coordinates[atom_names.index(atom_name), :3] for atom_name in frame_atoms)
        if special_case:
            rotation = _rfdiffusion_base_frame(third - second, first - second)
        else:
            rotation = _rfdiffusion_base_frame(
                third,
                coordinates.new_tensor((-1.0, 0.0, 0.0)),
            )
        base_frame_transforms[residue_index, 3 + chi_index, :3, :3] = rotation
        base_frame_transforms[residue_index, 3 + chi_index, :3, 3] = third

    # HIS group 6 follows the released protonation pseudo-torsion. HE1 is not
    # an atom14 output, but its ideal point defines the frame used by ND1/NE2.
    histidine_index = RFDIFFUSION_RESIDUE_TYPE_INDEX["HIS"]
    histidine_atoms = RFDIFFUSION_ATOM14_NAMES[histidine_index]
    histidine_coordinates = base_frame_coordinates[histidine_index]
    cd2 = histidine_coordinates[histidine_atoms.index("CD2"), :3]
    ce1 = histidine_coordinates[histidine_atoms.index("CE1"), :3]
    he1 = histidine_coordinates.new_tensor((2.8542, 1.5693, 0.0))
    base_frame_transforms[histidine_index, 5, :3, :3] = _rfdiffusion_base_frame(he1 - ce1, cd2 - ce1)
    base_frame_transforms[histidine_index, 5, :3, 3] = he1

    return frame_indices, base_frame_transforms, base_frame_coordinates


_RF_ATOM14_KINEMATICS = _build_rfdiffusion_atom14_kinematics()
# ---------------------------------------------------------------------------
# RFdiffusion tensor boundary and semantic layout.
# ---------------------------------------------------------------------------


@tensorclass
class _RfDiffusionStructureUpdateOutput:
    rotations: Tensor
    translations: Tensor
    state: Tensor
    torsion_angles: Tensor


@tensorclass
class _RfDiffusionTrunkBlockOutput:
    msa: Tensor
    pair: Tensor
    rotations: Tensor
    translations: Tensor
    state: Tensor
    torsion_angles: Tensor


@dataclass(eq=False)
class _RfDiffusionTrunkOutput:
    msa: Tensor
    pair: Tensor
    rotation_trajectory: Tensor
    translation_trajectory: Tensor
    torsion_trajectory: Tensor
    state: Tensor


@tensorclass
class _RfDiffusionGeometryLogits:
    distance_logits: Tensor
    omega_logits: Tensor
    theta_logits: Tensor
    phi_logits: Tensor


@dataclass(eq=False)
class _RfDiffusionDenoiserOutput:
    single: Tensor
    pair: Tensor
    backbone_coordinates: Tensor
    state: Tensor
    torsion_angles: Tensor
    sequence_logits: Tensor
    plddt_fraction: Tensor
    geometry_logits: _RfDiffusionGeometryLogits
    plddt_logits: Tensor
    resolved_log_odds: Tensor
    backbone_trajectory: Tensor
    torsion_trajectory: Tensor


@dataclass(eq=False)
class _RfDiffusionSample:
    coordinates: Tensor
    sequence_tokens: Tensor
    timesteps: Tensor
    plddt_fraction: Tensor
    predicted_clean_coordinates: Tensor
    next_step_coordinates: Tensor


@dataclass(frozen=True)
class RfDiffusionLayout:
    r"""
    Molecular identities aligned to RFdiffusion residue rows.

    Args:
        system: Optional input molecular system.
        structure: Optional input structure carrying observed coordinates.
        templates: Structural templates supplied with the input.
        placement: Optional generated and retained output segments.
        conditions: Spatial and sequence conditions applied to the design.
        output_residues: Existing or symbolic residue reference for each output row.
        output_instance_ids: Molecular instance ID for each output row.
        combined_length: Optional lower and upper bound for the total generated
            length.
    """

    system: MolecularSystem | None
    structure: Structure | None
    templates: tuple[Template, ...]
    placement: RfDiffusionPlacements | None
    conditions: tuple[
        AllowedResidueIdentities
        | Hotspot
        | RfDiffusionCoordination
        | RfDiffusionDistance
        | RfDiffusionPointGroupSymmetry
        | RfDiffusionRadiusOfGyration
        | RfDiffusionReferenceCoordinates
        | RfDiffusionSecondaryStructure
        | RfDiffusionSubstrateInteraction
        | SymbolicBond,
        ...,
    ]
    output_residues: tuple[ComponentReference | SymbolicResidueSite, ...]
    output_instance_ids: tuple[str, ...]
    # Combined generated-length bounds across all generated segments, excluding
    # retained existing spans. The denoiser ignores this result-decoding metadata.
    combined_length: tuple[int, int] | None = None


@tensorclass
class RfDiffusionInput:
    r"""
    Tensor and molecular-layout input for RFdiffusion.

    ``B`` is batch size, ``N`` residue count, and ``A`` atom slots per residue.

    Args:
        noisy_coordinates: Noisy atom coordinates with shape ``(B, N, A, 3)`` in
            angstroms.
        residue_type_features: One-hot residue-type features with shape
            ``(B, N, 22)``.
        residue_index: Chain-aware residue indices with shape ``(B, N)``.
        fixed_structure_mask: Residues whose coordinates remain fixed, with shape
            ``(B, N)``.
        fixed_sequence_mask: Residues whose identities remain fixed, with shape
            ``(B, N)``.
        hotspot_mask: Interaction-hotspot mask with shape ``(B, N)``.
        cyclic_mask: Cyclic-chain mask with shape ``(B, N)``.
        secondary_structure: Secondary-structure condition features with shape
            ``(B, N, 4)``.
        block_adjacency: Block-contact condition features with shape ``(B, N, N, 3)``.
        timestep: Diffusion timestep with shape ``(B,)``.
        layout: Molecular identities and conditions aligned to residue rows.
    """

    noisy_coordinates: Tensor
    residue_type_features: Tensor
    residue_index: Tensor
    fixed_structure_mask: Tensor
    fixed_sequence_mask: Tensor
    hotspot_mask: Tensor
    cyclic_mask: Tensor
    secondary_structure: Tensor
    block_adjacency: Tensor
    timestep: Tensor
    layout: RfDiffusionLayout


@dataclass(eq=False)
class RfDiffusionOutput:
    r"""
    Predictions and intermediate representations from RFdiffusion.

    ``B`` is batch size, ``N`` is residue count, and ``T`` is the number of
    structure-update blocks.

    Args:
        single: Per-residue representation with shape ``(B, N, C_single)``.
        pair: Pair representation with shape ``(B, N, N, C_pair)``.
        backbone_coordinates: Predicted ``(N, CA, C)`` coordinates with shape
            ``(B, N, 3, 3)`` in angstroms.
        state: Final structure-state representation with shape ``(B, N, C_state)``.
        torsion_angles: Ten normalized ``(cos, sin)`` torsion pairs with shape
            ``(B, N, 10, 2)``.
        sequence_logits: Per-residue amino-acid logits with shape ``(B, N, V)``.
        plddt_fraction: Expected per-residue pLDDT fraction with shape ``(B, N)``
            on the 0--1 scale.
        distance_logits: Pairwise distance-bin logits with shape
            ``(B, N, N, D_distance)``.
        omega_logits: Pairwise omega-bin logits with shape
            ``(B, N, N, D_omega)``.
        theta_logits: Pairwise theta-bin logits with shape
            ``(B, N, N, D_theta)``.
        phi_logits: Pairwise phi-bin logits with shape ``(B, N, N, D_phi)``.
        plddt_logits: Per-residue pLDDT-bin logits with shape ``(B, N, D_plddt)``.
        resolved_log_odds: Per-residue experimentally-resolved log-odds with
            shape ``(B, N)``.
        backbone_trajectory: Backbone coordinates with shape
            ``(T, B, N, 3, 3)`` in angstroms.
        torsion_trajectory: Torsion pairs with shape ``(T, B, N, 10, 2)``.
    """

    single: Tensor
    pair: Tensor
    backbone_coordinates: Tensor
    state: Tensor
    torsion_angles: Tensor
    sequence_logits: Tensor
    plddt_fraction: Tensor
    distance_logits: Tensor
    omega_logits: Tensor
    theta_logits: Tensor
    phi_logits: Tensor
    plddt_logits: Tensor
    resolved_log_odds: Tensor
    backbone_trajectory: Tensor
    torsion_trajectory: Tensor


@dataclass(eq=False)
class RfDiffusionTrajectory:
    r"""
    One RFdiffusion reverse-sampling trajectory.

    Args:
        timesteps: Descending diffusion timesteps with shape ``(S,)``.
        plddt_fraction: Per-step, per-residue pLDDT fractions with shape ``(S, L)``.
        predicted_clean_coordinates: Predicted-clean ``pX0`` atom coordinates with
            shape ``(S, L, A, 3)`` in angstroms.
        next_step_coordinates: Next-step ``Xt-1`` atom coordinates with shape
            ``(S, L, A, 3)`` in angstroms.
    """

    timesteps: Tensor
    plddt_fraction: Tensor
    predicted_clean_coordinates: Tensor
    next_step_coordinates: Tensor


@dataclass(eq=False)
class RfDiffusionDesignCandidate:
    r"""
    One RFdiffusion sample and its aligned scientific values.

    Args:
        structure: Designed backbone structure.
        placement_map: Realized component occurrences for every placement.
        scores: Scalar scores keyed by stable scientific names.
        metrics: One-axis metrics keyed by stable scientific names.
        trajectory: Reverse-sampling trajectory when requested.
    """

    structure: Structure
    placement_map: PlacementMap = field(default_factory=dict)
    scores: dict[str, Score] = field(default_factory=dict)
    metrics: dict[str, Metric] = field(default_factory=dict)
    trajectory: RfDiffusionTrajectory | None = None


@dataclass(eq=False)
class RfDiffusionDesignResult:
    r"""
    A generation-ordered collection of RFdiffusion designs.

    Args:
        candidates: Designed backbones in generation order.
    """

    candidates: tuple[RfDiffusionDesignCandidate, ...]

    def best(self, score: str) -> RfDiffusionDesignCandidate:
        r"""
        Return the earliest best candidate under a named score.

        Args:
            score: Comparable score name present on every candidate.

        Returns:
            The best candidate, with generation order breaking ties.
        """

        return best(self.candidates, score)


# --------------------------------------------------------------------------------------
# RFdiffusion placement mapping and per-group lengths.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RfDiffusionPlacements:
    r"""
    Declare generated and retained segments for RFdiffusion output chains.

    Args:
        placements: Ordered placement declaration.
        chain_lengths: Optional generated-length bound for each declaration group.
    """

    placements: PlacementMapping
    chain_lengths: tuple[int | None, ...] | None = None

    def __post_init__(self) -> None:
        derive_instance_of(self.placements)


# --------------------------------------------------------------------------------------
# RFdiffusion spatial and placement conditions.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RfDiffusionReferenceCoordinates:
    r"""
    Fix selected existing atoms to their observed coordinates.

    Args:
        sites: Existing atom occurrences to fix during sampling.
        tolerance_angstrom: Accepted coordinate deviation in angstroms.
    """

    sites: tuple[AtomSite, ...]
    tolerance_angstrom: float


@dataclass(frozen=True)
class RfDiffusionCoordination:
    r"""
    Apply a smooth coordination-number potential between endpoint sets.

    Args:
        side_a: Endpoints on the first side of the potential.
        side_b: Endpoints on the second side of the potential.
        center_angstrom: Midpoint of the distance transition in angstroms.
        scale_angstrom: Width of the distance transition in angstroms.
        direction: Whether to maximize or minimize coordination.
        weight: Optional potential weight.
    """

    side_a: tuple[ComponentReference | str | SymbolicResidueSite | AtomSite | SymbolicAtomSite, ...]
    side_b: tuple[ComponentReference | str | SymbolicResidueSite | AtomSite | SymbolicAtomSite, ...]
    center_angstrom: float
    scale_angstrom: float
    direction: Literal["maximize", "minimize"]
    weight: float | None = None


@dataclass(frozen=True)
class RfDiffusionDistance:
    r"""
    Constrain the distance between two atoms to an interval.

    Args:
        site_a: First atom occurrence.
        site_b: Second atom occurrence.
        lower_angstrom: Lower distance bound in angstroms.
        upper_angstrom: Upper distance bound in angstroms.
        tolerance_angstrom: Additional accepted distance outside the interval.
        weight: Optional potential weight.
    """

    site_a: AtomSite | SymbolicAtomSite
    site_b: AtomSite | SymbolicAtomSite
    lower_angstrom: float
    upper_angstrom: float
    tolerance_angstrom: float = 0.0
    weight: float | None = None


@dataclass(frozen=True)
class RfDiffusionRadiusOfGyration:
    r"""
    Constrain the compactness of a set of residues.

    Args:
        targets: Residues included in the radius-of-gyration calculation.
        upper_angstrom: Preferred upper radius of gyration in angstroms.
        weight: Optional potential weight.
    """

    targets: tuple[ComponentReference | str | SymbolicResidueSite, ...]
    upper_angstrom: float
    weight: float | None = None


@dataclass(frozen=True)
class RfDiffusionSubstrateInteraction:
    r"""
    Apply attraction and short-range repulsion to substrate atoms.

    Args:
        residues: Designed residues participating in the interaction.
        substrate_atoms: Substrate atom occurrences.
        center_angstrom: Midpoint of the attractive distance transition.
        scale_angstrom: Width of the attractive distance transition.
        attraction_strength: Attractive-potential multiplier.
        repulsion_slope: Slope of the short-range repulsion.
        repulsion_cutoff_angstrom: Repulsion cutoff distance in angstroms.
        nearest_only_repulsion: Whether repulsion uses only each residue's nearest
            substrate atom instead of every residue--substrate atom pair.
        weight: Optional overall potential weight.
    """

    residues: tuple[ComponentReference | str | SymbolicResidueSite, ...]
    substrate_atoms: tuple[AtomSite | SymbolicAtomSite, ...]
    center_angstrom: float
    scale_angstrom: float
    attraction_strength: float
    repulsion_slope: float
    repulsion_cutoff_angstrom: float = 5.0
    nearest_only_repulsion: bool = True
    weight: float | None = None


@dataclass(frozen=True)
class RfDiffusionPointGroupSymmetry:
    r"""
    Apply point-group symmetry to equal-length residue units.

    Args:
        units: Residue groups representing symmetric copies.
        kind: Point-group kind, such as cyclic or dihedral.
        order: Symmetry order when required by ``kind``.
        tolerance_angstrom: Accepted symmetry deviation in angstroms.
    """

    units: tuple[tuple[ComponentReference | str | SymbolicResidueSite, ...], ...]
    kind: str
    order: int | None
    tolerance_angstrom: float


@dataclass(frozen=True)
class RfDiffusionSecondaryStructure:
    r"""
    Assign one secondary-structure label to selected residues.

    Args:
        targets: Residues receiving the condition.
        kind: Secondary-structure label.
    """

    targets: tuple[ComponentReference | str | SymbolicResidueSite, ...]
    kind: str
