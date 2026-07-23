r"""Coordinate-producing binder-design Pipeline for PXDesign.

[`PxDesignPipeline`][] binds the native PXDesign-d generator Model+Processor and owns
the ``design`` verb over the PXDesign topology argument (``PlacementSpec``) and
per-residue target label (``Hotspot``), returning [`PxDesignDesignResult`][].
"""

from __future__ import annotations

from collections.abc import Sequence
from os import PathLike

import torch

from ...data import PlacementSpec
from ...data.molecular import MolecularSystem
from ...data.structure import Structure
from ...pipelines import PairedPipeline, resolve_generator
from .model import Hotspot, PxDesignDesignResult, PxDesignModel
from .processor import PxDesignProcessor


class PxDesignPipeline(PairedPipeline[PxDesignModel, PxDesignProcessor]):
    r"""Bind one PXDesign-d Model--Processor pair and own the ``design`` verb."""

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
        r"""Generate target-binder backbone candidates.

        Args:
            input: Optional target system, observed structure, or local structure path.
            placement: Topology containing target spans and one generated binder chain.
            conditions: Target hotspot labels.
            num_candidates: Number of backbone candidates to generate.
            num_sampling_steps: Number of diffusion steps.
            sample_chunk_size: Maximum candidates generated per model call. ``None`` generates
                all candidates together.
            seed: Optional integer seed; mutually exclusive with ``generator``.
            generator: Optional PyTorch random generator; mutually exclusive with ``seed``.

        Returns:
            A ``PxDesignDesignResult`` containing coordinate-bearing candidates in generation
            order.
        """

        model_input = self.processor(input, placement=placement, conditions=conditions)
        generator = resolve_generator(seed, generator, device=model_input.reference_positions.device)
        return self.model.design(
            model_input,
            num_candidates=num_candidates,
            num_sampling_steps=num_sampling_steps,
            sample_chunk_size=sample_chunk_size,
            generator=generator,
        )
