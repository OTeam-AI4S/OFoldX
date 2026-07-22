r"""RFdiffusion generation and design pipeline.

``design`` lowers one generation input through :class:`RFdiffusionProcessor`, runs the released
reverse-diffusion sampler via :meth:`RFdiffusionModel.design`, and returns the coordinate-producing
:class:`~ofoldx.models.rfdiffusion.model.RFdiffusionDesignResult`. A single caller-controlled
``seed`` or ``generator`` supplies the random stream for every requested candidate without
unintended reseeding.

The marker ID is ``"ofoldx.rfdiffusion"``. A checkpoint containing
``pipeline.json = {"pipeline": "ofoldx.rfdiffusion"}`` routes ``design`` to this pipeline.
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
    Coordination,
    Distance,
    Hotspot,
    PointGroupSymmetry,
    RadiusOfGyration,
    ReferenceCoordinates,
    RFdiffusionDesignResult,
    RFdiffusionModel,
    RFdiffusionPlacements,
    SecondaryStructure,
    SubstrateInteraction,
)
from .processor import RFdiffusionProcessor


class RFdiffusionPipeline(PairedPipeline[RFdiffusionModel, RFdiffusionProcessor]):
    r"""Bind one RFdiffusion Model-Processor pair for coordinate-producing design.

    Marker id ``"ofoldx.rfdiffusion"``. :meth:`design` returns an
    :class:`RFdiffusionDesignResult`.
    """

    def design(
        self,
        input: MolecularSystem | Structure | str | PathLike[str] | None = None,
        *,
        placement: RFdiffusionPlacements | None = None,
        templates: Sequence[Template] = (),
        conditions: Sequence[
            AllowedResidueIdentities
            | Hotspot
            | Coordination
            | SymbolicBond
            | Distance
            | PointGroupSymmetry
            | RadiusOfGyration
            | ReferenceCoordinates
            | SecondaryStructure
            | SubstrateInteraction
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
    ) -> RFdiffusionDesignResult:
        r"""Lower one generation input and run released RFdiffusion design into an ``RFdiffusionDesignResult``.

        Lowers base, complex, inpaint, active-site, or fold conditioning through the Processor,
        honoring ``inpaint_sequence`` and the RFdiffusion ``length``
        (``ContigMap.length``, combined generated length across inpaint chains, excluding receptor)
        -- then forwards every effective released sampling control to
        :meth:`RFdiffusionModel.design`. ``seed`` and ``generator`` are mutually exclusive and are
        the single caller-controlled run-level random stream for the ``num_candidates`` samples.
        Each candidate contains the complete decoded backbone ``Structure`` of one generated design.
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
