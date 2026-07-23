r"""RFdiffusion generation and design pipeline.

``design`` lowers one generation input through :class:`RfDiffusionProcessor`, runs the
reverse-diffusion sampler via :meth:`RfDiffusionModel.design`, and returns the coordinate-producing
:class:`~ofoldx.models.rfdiffusion.model.RfDiffusionDesignResult`. A single caller-controlled
``seed`` or ``generator`` supplies the random stream for every requested candidate without
unintended reseeding.
"""

from __future__ import annotations

from collections.abc import Sequence
from os import PathLike
from typing import Literal

import torch

# RFdiffusion represents requested covalent links directly as SymbolicBond values.
from ...data.bonds import SymbolicBond

# AllowedResidueIdentities is shared by sequence-design pipelines.
from ...data.conditions import AllowedResidueIdentities
from ...data.evidence import Template
from ...data.molecular import ComponentReference, MolecularSystem
from ...data.structure import Structure
from ...pipelines import PairedPipeline, resolve_generator
from .model import (
    Hotspot,
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
    r"""Bind one RFdiffusion Model and Processor for coordinate-producing design."""

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
        timestep: int = 50,
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
        r"""Generate RFdiffusion backbones and return an ``RfDiffusionDesignResult``.

        Args:
            input: Optional target or context system, structure, or structure path.
            placement: Generation topology for existing spans and generated protein segments.
            templates: Up to one fold-conditioning template.
            conditions: Residue, geometry, symmetry, and interaction conditions consumed by
                RFdiffusion.
            inpaint_sequence: Existing residue occurrences whose sequence identities are masked.
            length: Inclusive bounds for the combined generated length across inpaint chains,
                excluding receptor spans.
            timestep: Positive diffusion timestep stored on the lowered ``RfDiffusionInput``.
            num_candidates: Number of backbone candidates to generate.
            num_sampling_steps: Active reverse-diffusion schedule length. ``None`` uses the
                checkpoint schedule.
            final_step: Last reverse timestep to execute.
            partial_steps: Optional starting timestep for partial diffusion.
            noise_scale_ca: Initial CA-translation noise scale.
            final_noise_scale_ca: Final CA-translation noise scale. A differing value requires a
                linear schedule.
            ca_noise_schedule: CA-translation noise schedule.
            noise_scale_frame: Initial backbone-frame noise scale.
            final_noise_scale_frame: Final backbone-frame noise scale. A differing value requires
                a linear schedule.
            frame_noise_schedule: Backbone-frame noise schedule.
            align_motif: Whether to align predictions to fixed motif coordinates.
            return_trajectory: Whether to retain every reverse step during sampling.
            symmetry_recenter: Whether to recenter symmetry units before radial placement.
            symmetry_radius_angstrom: Radial placement distance for recentered symmetry units.
            guidance_scale: Overall scale for differentiable condition guidance.
            guidance_decay: Guidance schedule: ``"constant"``, ``"linear"``, ``"quadratic"``,
                or ``"cubic"``.
            seed: Optional integer seed.
            generator: Optional PyTorch random generator.

        Returns:
            An ``RfDiffusionDesignResult`` containing one coordinate-bearing candidate per sample.

        Notes:
            ``seed`` and ``generator`` are mutually exclusive. Candidate order is sampling order;
            each ``plddt`` metric axis follows that candidate's realized output residues.
        """

        batch = self.processor(
            input,
            placement=placement,
            templates=templates,
            conditions=conditions,
            inpaint_sequence=inpaint_sequence,
            length=length,
            timestep=timestep,
        )
        generator = resolve_generator(seed, generator, device=batch.noisy_coords.device)
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
