r"""
Protein backbone design operations for RFdiffusion.
"""

from __future__ import annotations

from collections.abc import Sequence
from os import PathLike
from typing import Literal

import torch

# RFdiffusion represents requested covalent links directly as SymbolicBond values.
from ...data.bonds import SymbolicBond

# AllowedResidueIdentities is shared by sequence-design pipelines.
from ...data.conditions import AllowedResidueIdentities, Hotspot
from ...data.evidence import Template
from ...data.molecular import ComponentReference, MolecularSystem
from ...data.structure import Structure
from ...io import read_structure
from ...pipelines import PairedPipeline, resolve_generator
from .model import (
    RfDiffusionCoordination,
    RfDiffusionDesignResult,
    RfDiffusionDistance,
    RfDiffusionModel,
    RfDiffusionPlacements,
    RfDiffusionPointGroupSymmetry,
    RfDiffusionRadiusOfGyration,
    RfDiffusionReferenceCoordinates,
    RfDiffusionSecondaryStructure,
    RfDiffusionSubstrateInteraction,
)
from .processor import RfDiffusionProcessor


class RfDiffusionPipeline(PairedPipeline[RfDiffusionModel, RfDiffusionProcessor]):
    r"""
    Protein backbone design pipeline for RFdiffusion.
    """

    model_architecture = "rfdiffusion"

    def design(
        self,
        input: MolecularSystem | Structure | str | PathLike[str] | None = None,
        *,
        placement: RfDiffusionPlacements | None = None,
        templates: Sequence[Template] = (),
        conditions: Sequence[
            AllowedResidueIdentities
            | Hotspot
            | RfDiffusionCoordination
            | SymbolicBond
            | RfDiffusionDistance
            | RfDiffusionPointGroupSymmetry
            | RfDiffusionRadiusOfGyration
            | RfDiffusionReferenceCoordinates
            | RfDiffusionSecondaryStructure
            | RfDiffusionSubstrateInteraction
        ] = (),
        inpaint_sequence: Sequence[ComponentReference] = (),
        length: tuple[int, int] | None = None,
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
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> RfDiffusionDesignResult:
        r"""
        Generate protein-backbone structures.

        Args:
            input: Optional context system, observed structure, or local structure path.
            placement: Topology containing existing spans and generated protein segments.
            templates: Zero or one structural template.
            conditions: Residue-identity, hotspot, geometry, symmetry, bond, and
                interaction conditions.
            inpaint_sequence: Existing residue occurrences whose identities are masked.
            length: Inclusive lower and upper bounds for total generated length across
                generated segments, excluding retained receptor spans.
            num_candidates: Number of independently sampled backbone candidates.
            num_sampling_steps: Number of active reverse-diffusion timesteps. ``None``
                uses the model configuration.
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
            seed: Integer seed, mutually exclusive with ``generator``.
            generator: PyTorch random generator, mutually exclusive with ``seed``.

        Returns:
            An [`RfDiffusionDesignResult`][] containing structures in sampling order.
            Coordinates are in angstroms; each candidate's pLDDT metric uses a 0--100
            scale.
        """

        if isinstance(input, str | PathLike):
            input = read_structure(input)
        batch = self.processor(
            input,
            placement=placement,
            templates=templates,
            conditions=conditions,
            inpaint_sequence=inpaint_sequence,
            length=length,
        )
        generator = resolve_generator(seed, generator, device=batch.noisy_coordinates.device)
        return self.model.design(
            batch,
            num_candidates=num_candidates,
            num_sampling_steps=num_sampling_steps,
            final_step=final_step,
            partial_steps=partial_steps,
            noise_scale_ca=noise_scale_ca,
            final_noise_scale_ca=final_noise_scale_ca,
            ca_noise_schedule=ca_noise_schedule,
            noise_scale_frame=noise_scale_frame,
            final_noise_scale_frame=final_noise_scale_frame,
            frame_noise_schedule=frame_noise_schedule,
            align_motif=align_motif,
            return_trajectory=return_trajectory,
            symmetry_recenter=symmetry_recenter,
            symmetry_radius_angstrom=symmetry_radius_angstrom,
            guidance_scale=guidance_scale,
            guidance_decay=guidance_decay,
            generator=generator,
        )
