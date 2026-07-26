r"""
Protein backbone design operations for PXDesign.
"""

from __future__ import annotations

from collections.abc import Sequence
from os import PathLike

import torch

from ...data import PlacementSpec
from ...data.conditions import Hotspot
from ...data.molecular import MolecularSystem
from ...data.structure import Structure
from ...io import read_structure
from ...pipelines import PairedPipeline, resolve_generator
from .model import PxDesignDesignResult, PxDesignModel
from .processor import PxDesignProcessor


class PxDesignPipeline(PairedPipeline[PxDesignModel, PxDesignProcessor]):
    r"""
    Protein backbone design pipeline for PXDesign.
    """

    model_architecture = "pxdesign"

    def design(
        self,
        input: MolecularSystem | Structure | str | PathLike[str] | None = None,
        *,
        placement: PlacementSpec,
        conditions: Sequence[Hotspot] = (),
        num_candidates: int = 1,
        num_sampling_steps: int = 400,
        sample_chunk_size: int | None = 10,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> PxDesignDesignResult:
        r"""
        Generate target-binder backbone candidates.

        Args:
            input: Optional target system, observed structure, or local structure path.
            placement: Topology containing complete target spans and one fixed-length
                generated protein chain.
            conditions: Target hotspot labels.
            num_candidates: Number of backbone candidates.
            num_sampling_steps: Number of EDM sampling steps.
            sample_chunk_size: Maximum number of candidates denoised together. ``None``
                processes all candidates together.
            seed: Integer seed, mutually exclusive with ``generator``.
            generator: PyTorch random generator, mutually exclusive with ``seed``.

        Returns:
            A [`PxDesignDesignResult`][] containing coordinate-bearing structures in
            sampling order. Coordinates are in angstroms.
        """

        if isinstance(input, str | PathLike):
            input = read_structure(input)
        model_input = self.processor(input, placement=placement, conditions=conditions)
        generator = resolve_generator(seed, generator, device=model_input.reference_positions.device)
        return self.model.design(
            model_input,
            num_candidates=num_candidates,
            num_sampling_steps=num_sampling_steps,
            sample_chunk_size=sample_chunk_size,
            generator=generator,
        )
