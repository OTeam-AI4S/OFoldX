r"""
Protein backbone design operations for RFdiffusion.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from os import PathLike
from typing import Literal

import numpy as np
import torch
from torch import Tensor

from ...data.bonds import SymbolicBond
from ...data.conditions import Hotspot
from ...data.evidence import Template
from ...data.molecular import ENTITY_PROTEIN, ComponentReference, MolecularSystem
from ...data.placement import ExistingSpan, PlacementMap, SymbolicResidueSite
from ...data.structure import Structure
from ...io import read_structure
from ...pipelines import PairedPipeline, resolve_generator
from ...results import Metric, Score, best
from .model import (
    RfDiffusionCoordination,
    RfDiffusionDistance,
    RfDiffusionInput,
    RfDiffusionModel,
    RfDiffusionPlacements,
    RfDiffusionPointGroupSymmetry,
    RfDiffusionRadiusOfGyration,
    RfDiffusionSamplingOutput,
    RfDiffusionSecondaryStructure,
    RfDiffusionSubstrateInteraction,
    RfDiffusionTrajectory,
)
from .processor import RfDiffusionProcessor


@dataclass(eq=False)
class RfDiffusionDesignCandidate:
    r"""
    One decoded RFdiffusion backbone candidate.
    """

    structure: Structure
    placement_map: PlacementMap = field(default_factory=dict)
    scores: dict[str, Score] = field(default_factory=dict)
    metrics: dict[str, Metric] = field(default_factory=dict)
    trajectory: RfDiffusionTrajectory | None = None


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
        secondary_structure: Tensor | None = None,
        block_adjacency: Tensor | None = None,
        secondary_structure_overrides: Sequence[RfDiffusionSecondaryStructure] = (),
        hotspot: Hotspot | None = None,
        inpaint_sequence: Sequence[ComponentReference] = (),
        inpaint_structure: Sequence[ComponentReference] = (),
        provide_sequence: Sequence[SymbolicResidueSite] = (),
        symmetry: RfDiffusionPointGroupSymmetry | None = None,
        potentials: Sequence[
            RfDiffusionCoordination
            | RfDiffusionDistance
            | RfDiffusionRadiusOfGyration
            | RfDiffusionSubstrateInteraction
        ] = (),
        bonds: Sequence[SymbolicBond] = (),
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
        Realize and sample independent RFdiffusion candidates.
        """

        if num_candidates <= 0:
            raise ValueError("num_candidates must be positive")
        if isinstance(input, str | PathLike):
            input = read_structure(input)
        generator = resolve_generator(
            seed,
            generator,
            device=self.processor.device,
        )
        candidates = []
        for _candidate in range(num_candidates):
            processed, placement_map = self.processor.realize(
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
                partial_diffusion=partial_steps is not None,
                generator=generator,
            )
            sample = self.model.sample(
                processed,
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
            candidates.append(
                _candidate_result(
                    sample,
                    processed,
                    placement_map,
                    source=input if isinstance(input, Structure) else None,
                    placement=placement,
                )
            )
        return RfDiffusionDesignResult(tuple(candidates))


def _candidate_result(
    sample: RfDiffusionSamplingOutput,
    input: RfDiffusionInput,
    placement_map: PlacementMap,
    *,
    source: Structure | None,
    placement: RfDiffusionPlacements | None,
) -> RfDiffusionDesignCandidate:
    system = input.system.to("cpu")  # type: ignore[attr-defined]
    canonical_atom_index = input.atom_index[0].detach().cpu()
    coordinates = torch.zeros(
        (system.atom_symbol.numel(), 3),
        dtype=torch.float32,
    )
    coordinate_mask = torch.zeros(
        system.atom_symbol.numel(),
        dtype=torch.bool,
    )
    model_coordinates = sample.coordinates.detach().to(
        device="cpu",
        dtype=torch.float32,
    )
    valid = canonical_atom_index >= 0
    observed = valid & torch.isfinite(model_coordinates).all(dim=-1)
    coordinates[canonical_atom_index[observed]] = model_coordinates[observed]
    coordinate_mask[canonical_atom_index[observed]] = True

    source_symbols: tuple[str, ...] = ()
    residue_id = torch.full(
        (system.component_symbol.numel(),),
        -1,
        dtype=torch.int64,
    )
    insertion_code = torch.full_like(residue_id, -1)
    if source is not None and placement is not None:
        source_symbols = source.source_symbols
        for group, declarations in placement.placements.items():
            for placement_id, value in declarations.items():
                if not isinstance(value, ExistingSpan):
                    continue
                output = placement_map[group][placement_id]
                for source_component, output_reference in zip(
                    range(value.start, value.stop),
                    output,
                    strict=True,
                ):
                    source_index = source.system.component_index((value.instance_id, source_component))
                    output_index = system.component_index(output_reference)
                    residue_id[output_index] = source.residue_id[source_index]
                    insertion_code[output_index] = source.insertion_code[source_index]
    elif source is not None:
        source_symbols = source.source_symbols
        for instance in range(source.system.instance_symbol.numel()):
            entity = int(source.system.instance_entity[instance])
            if int(source.system.entity_type[entity]) != ENTITY_PROTEIN:
                continue
            instance_id = source.system.instance_id(instance)
            output = placement_map[instance_id][instance_id]
            start = int(source.system.instance_component_ptr[instance])
            stop = int(source.system.instance_component_ptr[instance + 1])
            for source_index, output_reference in zip(
                range(start, stop),
                output,
                strict=True,
            ):
                output_index = system.component_index(output_reference)
                residue_id[output_index] = source.residue_id[source_index]
                insertion_code[output_index] = source.insertion_code[source_index]

    structure = Structure(  # type: ignore[call-arg]
        system=system,
        coordinates=coordinates,
        coordinate_mask=coordinate_mask,
        occupancy=torch.zeros(system.atom_symbol.numel(), dtype=torch.float32),
        occupancy_mask=torch.zeros(system.atom_symbol.numel(), dtype=torch.bool),
        b_factor=torch.zeros(system.atom_symbol.numel(), dtype=torch.float32),
        b_factor_mask=torch.zeros(system.atom_symbol.numel(), dtype=torch.bool),
        source_symbols=source_symbols,
        residue_id=residue_id,
        insertion_code=insertion_code,
        batch_size=[],
    )
    plddt = sample.plddt_fraction.detach().to(device="cpu", dtype=torch.float32).numpy() * 100.0
    axis = tuple(system.component_reference(int(index)) for index in input.component_index[0].detach().cpu())
    return RfDiffusionDesignCandidate(
        structure=structure,
        placement_map=placement_map,
        scores={
            "mean_plddt": Score(
                float(np.mean(plddt)),
                unit="0-100",
                preferred="higher",
            )
        },
        metrics={
            "plddt": Metric(
                values=plddt,
                axis=axis,
                unit="0-100",
            )
        },
        trajectory=sample.trajectory,
    )
