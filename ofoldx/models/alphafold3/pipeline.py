r"""The released AlphaFold 3 folding Pipeline."""

from __future__ import annotations

from collections.abc import Sequence
from os import PathLike

import torch

from ...data.evidence import MSA, MSACollection, Template
from ...data.molecular import MolecularSystem, PolymerSequence
from ...data.structure import Structure
from ...pipelines import PairedPipeline, processor_device, resolve_generator
from .model import AlphaFold3FoldResult, AlphaFold3Model, AlphaFold3SeedGroup
from .processor import AlphaFold3Processor


class AlphaFold3Pipeline(PairedPipeline[AlphaFold3Model, AlphaFold3Processor]):
    r"""Bind the official AF3 Model and Processor to its seed-group-preserving fold operation."""

    def fold(
        self,
        input: str | PathLike[str] | PolymerSequence | Sequence[PolymerSequence] | MolecularSystem | Structure,
        *,
        msa: MSA | MSACollection | str | PathLike[str] | None = None,
        templates: Sequence[Template] = (),
        num_samples: int | None = None,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        recycling_steps: int | None = None,
        num_sampling_steps: int | None = None,
    ) -> AlphaFold3FoldResult:
        r"""Lower one assembly and run one official AF3 seed group.

        The official parameters may be stored in a private Hub repository; loading
        and authentication are repository concerns, while this Pipeline remains the
        same concrete task boundary for local and private snapshots.
        """

        generator = resolve_generator(seed, generator, device=processor_device(self.processor))
        features = self.processor(input, msa=msa, templates=templates)
        result = self.model.fold(
            features,
            num_samples=num_samples,
            generator=generator,
            recycling_steps=recycling_steps,
            num_sampling_steps=num_sampling_steps,
        )
        if seed is None:
            return result
        group = result.seed_groups[0]
        return AlphaFold3FoldResult(seed_groups=(AlphaFold3SeedGroup(seed=seed, predictions=group.predictions),))
