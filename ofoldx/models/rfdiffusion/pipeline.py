r"""OFoldX's biological operation boundary for released RFdiffusion generation & design.

Owns the one Processor call and the family's released coordinate-producing ``design`` verb.
``design`` lowers one generation input through :class:`RFdiffusionProcessor` and runs the
released reverse-diffusion sampler via :meth:`RFdiffusionModel.design`, returning the
family-owned, coordinate-producing :class:`~ofoldx.models.rfdiffusion.model.RFdiffusionResult`
(NOT the frozen coordinate-free shared ``DesignResult``; §5.3.1/§5.3.3, §6.1). The Pipeline
coordinates the run-level random streams for its ``num_candidates`` calls without unintended
reseeding (§3.7.3), threading a single caller-controlled ``seed``/``generator`` through to the
sampler.

Marker id ``"ofoldx.rfdiffusion"``. Per DESIGN.md §7.2 and Principle 13 this concrete Pipeline
is NOT registered (no ``register_pipeline`` / ``pipeline.json``) this wave -- deferred, exactly
as ``ESMFold2Pipeline`` defers, until real-weight sampled-output acceptance passes. This wave
freezes its signature and family Result type at contract level only.
"""

from __future__ import annotations

from collections.abc import Sequence
from os import PathLike
from typing import Literal

import torch
from typing_extensions import Self

# The requested covalent link IS the frozen §3.7.4 SymbolicBond (no family alias; DESIGN §10.2).
from ...data.bonds import SymbolicBond

# AllowedResidueIdentities stays shared: it is the frozen §3.7.1 identity record.
from ...data.conditions import AllowedResidueIdentities
from ...data.evidence import Template
from ...data.molecular import ComponentReference, MolecularSystem
from ...data.structure import Structure
from ...pipelines import resolve_generator
from .model import (
    Contact,
    Coordination,
    Distance,
    PointGroupSymmetry,
    RadiusOfGyration,
    ReferenceCoordinates,
    RFdiffusionModel,
    RFdiffusionPlacements,
    RFdiffusionResult,
    SecondaryStructure,
    SubstrateInteraction,
)
from .processor import RFdiffusionProcessor


class RFdiffusionPipeline:
    r"""Bind one RFdiffusion Model-Processor pair for the released coordinate-producing design verb.

    Marker id ``"ofoldx.rfdiffusion"``. The Pipeline is not a base class and shares no runtime
    request object with any other family. ``design`` returns the family-owned
    :class:`RFdiffusionResult`.
    """

    model: RFdiffusionModel
    processor: RFdiffusionProcessor

    def __init__(
        self,
        *,
        model: RFdiffusionModel,
        processor: RFdiffusionProcessor,
    ) -> None:
        self.model = model
        self.processor = processor

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Self:
        r"""Place the Model and Processor on one device and dtype in place."""

        self.model.to(device=device, dtype=dtype)
        self.processor.to(device=device, dtype=dtype)
        return self

    def design(
        self,
        input: MolecularSystem | Structure | str | PathLike[str] | None = None,
        *,
        templates: Sequence[Template | str | PathLike[str]] = (),
        conditions: Sequence[
            AllowedResidueIdentities
            | Contact
            | RFdiffusionPlacements
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
    ) -> RFdiffusionResult:
        r"""Lower one generation input and run released RFdiffusion design into an ``RFdiffusionResult``.

        Lowers base/complex/inpaint/active-site/fold conditioning through the Processor -- honoring
        the frozen §3.7.3 ``inpaint_sequence`` and the family-owned §3.7 ``length``
        (``ContigMap.length``, combined generated length across inpaint chains, excluding receptor)
        -- then forwards every effective released sampling control to
        :meth:`RFdiffusionModel.design`. ``seed`` and ``generator`` are mutually exclusive and are
        the single caller-controlled run-level random stream for the ``num_candidates`` samples.
        Each candidate owns the complete decoded backbone ``Structure`` of one generated design.
        """

        batch = self.processor(
            input,
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
