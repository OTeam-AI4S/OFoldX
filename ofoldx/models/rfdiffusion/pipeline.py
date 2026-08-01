# OFoldX
# Copyright (C) 2026-Present  Valhalla Technology

# This file is part of OFoldX.

# OFoldX is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# OFoldX is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# For additional terms and clarifications, please refer to our License FAQ at:
# <https://ofoldx.valtech.ai/about/license-faq/>.

r"""
RFdiffusion backbone generation Pipeline.

Candidates are independent: each is realised on its own (its generated segments draw their own lengths), sampled from
one random stream, and decoded by the one decode (``decode_coordinates``) over its regeneration scope -- the
generated and diffused rows in the source frame, as GENERATED. A fixed motif keeps the coordinates it was copied with
and the evidence they carry, and atoms of a regenerated row the model does not place (its side chain) become
unobserved and are counted.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from os import PathLike
from typing import Literal

import numpy as np
import torch
from torch import Tensor

from ...chemistry.construction import PolymerSequence
from ...chemistry.definitions import ComponentLibrary
from ...data.structure import CoordinateEvidence, Structure
from ...errors import RequestRef, fail
from ...pipelines import PairedPipeline, resolve_generator
from ...pipelines.base import task_call
from ...pipelines.input import resolve_polymer_input, resolve_template_input
from ...results import Metric, Score, best
from ...templates import Template
from ..conditions import Hotspot, PlacementMap, SymbolicBond, SymbolicResidueSite
from ..design import PositionSelection
from ..preparation import PreparedInput, decode_coordinates
from .model import (
    RfDiffusionInput,
    RfDiffusionModel,
    RfDiffusionSamplingOutput,
    RfDiffusionTrajectory,
)
from .processor import (
    Potential,
    RfDiffusionPlacements,
    RfDiffusionPointGroupSymmetry,
    RfDiffusionProcessor,
    RfDiffusionSecondaryStructure,
)

#: The model places the backbone of every row: its first four atom14 slots (N, CA, C, O).
_BACKBONE_SLOTS = 4
_ATOM14 = 14
#: Upper bound of the seed the segment-length stream is drawn with.
_SEED_LIMIT = 2**62


def _length_stream(generator: torch.Generator | None) -> np.random.Generator | None:
    r"""
    The numpy stream the candidates draw their segment lengths from: one draw of the sampling stream, so a ``seed=``
    and a caller's ``generator=`` both control the lengths. ``None`` (the ambient stream) when neither is given.
    """

    if generator is None:
        return None
    return np.random.default_rng(int(torch.randint(_SEED_LIMIT, (1,), generator=generator, device=generator.device)))


@dataclass(eq=False)
class RfDiffusionDesignCandidate:
    r"""
    One decoded RFdiffusion backbone candidate.
    """

    structure: Structure
    placement_map: PlacementMap
    scores: dict[str, Score]
    metrics: dict[str, Metric]
    trajectory: RfDiffusionTrajectory | None


@dataclass(eq=False)
class RfDiffusionDesignResult:
    r"""
    RFdiffusion candidates in generation order.
    """

    candidates: tuple[RfDiffusionDesignCandidate, ...]

    @property
    def best(self) -> RfDiffusionDesignCandidate:
        r"""
        Return the earliest candidate with the highest mean pLDDT.
        """

        return best(self.candidates, "mean_plddt")


@dataclass
class RfDiffusionPipelineInput:
    r"""The source, its evidence and the two random streams every candidate of one call shares."""

    source: Structure | None
    placement: RfDiffusionPlacements | None
    templates: tuple[Template, ...]
    secondary_structure: Tensor | None
    block_adjacency: Tensor | None
    secondary_structure_overrides: Sequence[RfDiffusionSecondaryStructure]
    hotspot: Hotspot | None
    inpaint_sequence: PositionSelection | None
    inpaint_structure: PositionSelection | None
    provide_sequence: Sequence[SymbolicResidueSite]
    symmetry: RfDiffusionPointGroupSymmetry | None
    potentials: Sequence[Potential]
    bonds: Sequence[SymbolicBond]
    length: tuple[int, int] | None
    library: ComponentLibrary | None
    generator: torch.Generator | None
    lengths: np.random.Generator | None


class RfDiffusionPipeline(PairedPipeline[RfDiffusionModel, RfDiffusionProcessor]):
    r"""
    Protein backbone generation pipeline for RFdiffusion.
    """

    task = "design"

    model_architecture = "rfdiffusion"

    def prepare(
        self,
        input: Structure | PolymerSequence | Sequence[PolymerSequence] | str | PathLike[str] | None = None,
        *,
        placement: RfDiffusionPlacements | None = None,
        templates: Sequence[Template] | str | PathLike[str] = (),
        secondary_structure: Tensor | None = None,
        block_adjacency: Tensor | None = None,
        secondary_structure_overrides: Sequence[RfDiffusionSecondaryStructure] = (),
        hotspot: Hotspot | None = None,
        inpaint_sequence: PositionSelection | None = None,
        inpaint_structure: PositionSelection | None = None,
        provide_sequence: Sequence[SymbolicResidueSite] = (),
        symmetry: RfDiffusionPointGroupSymmetry | None = None,
        potentials: Sequence[Potential] = (),
        bonds: Sequence[SymbolicBond] = (),
        length: tuple[int, int] | None = None,
        library: ComponentLibrary | None = None,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> RfDiffusionPipelineInput:
        r"""
        Bind the source, its evidence and one random stream for independent candidates.

        Notes:
            Each declaration group becomes an output protein chain. Existing spans copy source components and generated
            rows use ``UNK`` because the model predicts no sequence. Copies retain coordinate evidence, including
            PREDICTED
            evidence from a predicted motif, and their source frame. Generated rows start with unavailable coordinates.
            Copied component definitions must match those used by the source. Point-group symmetry creates units of one
            entity,
            with the same sequence in each unit.

            Source positions use ``ComponentRef`` or the source's author numbering in the contig grammar. Generated
            positions
            use placement and offset. A new ``SymbolicBond`` must select one occurrence at each endpoint. An
            ``AtomSite`` with
            several output copies is ambiguous; use ``SymbolicAtomSite`` to name a copy.

            Generated rows and placement boundaries join the next row. Two copied rows within one placement join exactly
            when
            the source joins them. Output links determine each copied row's junctions and caps; source junctions and
            caps are
            not copied independently. Other source connections, including disulfides, cross-links, coordination and
            branch
            links, retain their states. A copied disulfide therefore does not regain HG. Connections between copied
            partners
            become edges; connections to uncopied or already external partners retain the partner's source instance,
            position
            and atom as an external connection.

            Unsupported connections include ones whose ends share no single copy, uncopied partners whose instance name
            collides with an output instance, source external connections through symmetry operations, and connections
            carrying
            bond configurations.
        """
        sampling = resolve_generator(seed, generator, device=self.processor.device)
        return RfDiffusionPipelineInput(
            source=None if input is None else resolve_polymer_input(input, library=library),
            placement=placement,
            templates=resolve_template_input(templates),
            secondary_structure=secondary_structure,
            block_adjacency=block_adjacency,
            secondary_structure_overrides=secondary_structure_overrides,
            hotspot=hotspot,
            inpaint_sequence=inpaint_sequence,
            inpaint_structure=inpaint_structure,
            provide_sequence=provide_sequence,
            symmetry=symmetry,
            potentials=potentials,
            bonds=bonds,
            length=length,
            library=library,
            generator=sampling,
            lengths=_length_stream(sampling),
        )

    def prepare_candidate(
        self, input: RfDiffusionPipelineInput, *, partial_diffusion: bool = False, timestep: int = 50
    ) -> tuple[PreparedInput[RfDiffusionInput], PlacementMap]:
        r"""Realise one candidate of a bound input: its own segment lengths, its target and its numeric input."""
        return self.processor.realize(
            input.source,
            placement=input.placement,
            templates=input.templates,
            secondary_structure=input.secondary_structure,
            block_adjacency=input.block_adjacency,
            secondary_structure_overrides=input.secondary_structure_overrides,
            hotspot=input.hotspot,
            inpaint_sequence=input.inpaint_sequence,
            inpaint_structure=input.inpaint_structure,
            provide_sequence=input.provide_sequence,
            symmetry=input.symmetry,
            potentials=input.potentials,
            bonds=input.bonds,
            length=input.length,
            partial_diffusion=partial_diffusion,
            timestep=timestep,
            library=input.library,
            generator=input.lengths,
        )

    def forward(
        self,
        input: RfDiffusionPipelineInput,
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
        self_conditioning: bool = True,
        symmetric_self_conditioning: bool = True,
        return_trajectory: bool = False,
        symmetry_recenter: bool = True,
        symmetry_radius_angstrom: float = 10.0,
        guidance_scale: float = 10.0,
        guidance_decay: str = "constant",
    ) -> tuple[RfDiffusionDesignCandidate, ...]:
        r"""Realise, sample and decode candidates in one continuous random stream."""
        if num_candidates <= 0:
            fail("value_invalid", RequestRef("num_candidates", str(num_candidates)))
        candidates = []
        for _candidate in range(num_candidates):
            prepared, placement_map = self.prepare_candidate(input, partial_diffusion=partial_steps is not None)
            sample = self.model.sample(
                prepared.features,
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
                self_conditioning=self_conditioning,
                symmetric_self_conditioning=symmetric_self_conditioning,
                return_trajectory=return_trajectory,
                symmetry_recenter=symmetry_recenter,
                symmetry_radius_angstrom=symmetry_radius_angstrom,
                guidance_scale=guidance_scale,
                guidance_decay=guidance_decay,
                generator=input.generator,
            )
            candidates.append(self.decode_candidate(prepared, sample, placement_map))
        return tuple(candidates)

    def postprocess(
        self,
        input: RfDiffusionPipelineInput,
        output: tuple[RfDiffusionDesignCandidate, ...],
    ) -> RfDiffusionDesignResult:
        r"""
        Collect independently realised candidates in generation order.

        Notes:
            Only generated and diffused rows receive regenerated coordinates. Fixed motif coordinates and their evidence
            remain unchanged. Each regenerated row receives N, CA, C and O predictions; side-chain coordinates are
            unavailable
            and counted, since reconstruction of a side chain with masked identity is not a prediction. Rows used as
            placed
            backbone frames require complete N, CA and C coordinates.
        """

        return RfDiffusionDesignResult(output)

    def design(
        self,
        input: Structure | PolymerSequence | Sequence[PolymerSequence] | str | PathLike[str] | None = None,
        *,
        placement: RfDiffusionPlacements | None = None,
        templates: Sequence[Template] | str | PathLike[str] = (),
        secondary_structure: Tensor | None = None,
        block_adjacency: Tensor | None = None,
        secondary_structure_overrides: Sequence[RfDiffusionSecondaryStructure] = (),
        hotspot: Hotspot | None = None,
        inpaint_sequence: PositionSelection | None = None,
        inpaint_structure: PositionSelection | None = None,
        provide_sequence: Sequence[SymbolicResidueSite] = (),
        symmetry: RfDiffusionPointGroupSymmetry | None = None,
        potentials: Sequence[Potential] = (),
        bonds: Sequence[SymbolicBond] = (),
        length: tuple[int, int] | None = None,
        library: ComponentLibrary | None = None,
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
        self_conditioning: bool = True,
        symmetric_self_conditioning: bool = True,
        return_trajectory: bool = False,
        symmetry_recenter: bool = True,
        symmetry_radius_angstrom: float = 10.0,
        guidance_scale: float = 10.0,
        guidance_decay: str = "constant",
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> RfDiffusionDesignResult:
        r"""
        Generate independent RFdiffusion candidates.

        Args:
            input: The source every existing span, template alignment and substrate atom is read from; ``None``
                generates without one.
            placement: The declaration groups of the output (default: one existing span per source protein instance).
            templates: At most one structural template, or the path of a native template artifact.
            secondary_structure: Optional ``(rows, 4)`` secondary-structure features replacing the template's.
            block_adjacency: Optional ``(rows, rows, 3)`` block-adjacency features replacing the template's.
            secondary_structure_overrides: Per-residue secondary-structure labels applied after the features.
            hotspot: Optional residues labelled as interaction hotspots.
            inpaint_sequence: Source positions whose identity is hidden from the model.
            inpaint_structure: Source positions whose coordinates are diffused instead of fixed.
            provide_sequence: Rows whose identity is provided under partial diffusion.
            symmetry: Optional point group over equal contiguous units of the output.
            potentials: Guidance potentials over the realised rows and source substrate atoms.
            bonds: Covalent bonds the output declares between its rows.
            length: Optional inclusive bounds on the number of non-receptor rows.
            library: The definitions the output is built from (default the configured artifact).
            num_candidates: Number of independently realised candidates.
            num_sampling_steps: Number of reverse-diffusion steps. ``None`` uses the Model configuration.
            final_step: The timestep sampling stops at.
            partial_steps: Diffuse the whole source for this many steps instead of generating.
            noise_scale_ca: Translation-noise scale.
            final_noise_scale_ca: Translation-noise scale at the final step, with ``ca_noise_schedule``.
            ca_noise_schedule: Interpolation between the two translation-noise scales.
            noise_scale_frame: Rotation-noise scale.
            final_noise_scale_frame: Rotation-noise scale at the final step, with ``frame_noise_schedule``.
            frame_noise_schedule: Interpolation between the two rotation-noise scales.
            align_motif: Whether each step aligns the prediction onto the fixed motif.
            self_conditioning: Whether the preceding predicted-clean backbone conditions the next step.
            symmetric_self_conditioning: Whether that backbone is first projected through the point group.
            return_trajectory: Whether every step's prediction is retained.
            symmetry_recenter: Whether the symmetric copies are recentred each step.
            symmetry_radius_angstrom: Radial offset of the symmetric copies.
            guidance_scale: Scale of the guidance-potential gradients.
            guidance_decay: Decay of the guidance scale over the reverse trajectory.
            seed: Optional seed for the segment lengths and the reverse sampling. Mutually exclusive with
                ``generator``.
            generator: Optional generator advanced by the reverse sampling. Mutually exclusive with ``seed``.

        Returns:
            The candidates in generation order, each decoded onto its own realised target.
        """

        prepared = self.prepare(
            input,
            placement=placement,
            templates=templates,
            secondary_structure=secondary_structure,
            block_adjacency=block_adjacency,
            secondary_structure_overrides=secondary_structure_overrides,
            hotspot=hotspot,
            inpaint_sequence=inpaint_sequence,
            inpaint_structure=inpaint_structure,
            provide_sequence=provide_sequence,
            symmetry=symmetry,
            potentials=potentials,
            bonds=bonds,
            length=length,
            library=library,
            seed=seed,
            generator=generator,
        )
        output = self.forward(
            prepared,
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
            self_conditioning=self_conditioning,
            symmetric_self_conditioning=symmetric_self_conditioning,
            return_trajectory=return_trajectory,
            symmetry_recenter=symmetry_recenter,
            symmetry_radius_angstrom=symmetry_radius_angstrom,
            guidance_scale=guidance_scale,
            guidance_decay=guidance_decay,
        )
        return self.postprocess(prepared, output)

    def decode_candidate(
        self,
        prepared: PreparedInput[RfDiffusionInput],
        output: RfDiffusionSamplingOutput,
        placement_map: PlacementMap,
    ) -> RfDiffusionDesignCandidate:
        r"""Decode one raw RFdiffusion sample without executing the Model."""

        coordinates = output.coordinates.detach().cpu().double().numpy().reshape(-1, 3)
        atom = prepared.context.slots.atom
        backbone = np.tile(np.arange(_ATOM14) < _BACKBONE_SLOTS, len(atom) // _ATOM14)
        valid = backbone & prepared.context.scope.atoms[np.maximum(atom, 0)]
        structure, _report = decode_coordinates(
            prepared, coordinates, valid=valid, evidence=CoordinateEvidence.GENERATED
        )
        plddt = output.plddt_fraction.detach().to(device="cpu", dtype=torch.float32).numpy() * 100.0
        system = prepared.context.structure.system
        return RfDiffusionDesignCandidate(
            structure=structure,
            placement_map=placement_map,
            scores={"mean_plddt": Score(float(np.mean(plddt)), unit="0-100", preferred="higher")},
            metrics={
                "plddt": Metric(
                    values=plddt,
                    axis=system.component_refs(prepared.context.slots.layout.token_component),
                    unit="0-100",
                )
            },
            trajectory=output.trajectory,
        )

    __call__ = task_call(design)
