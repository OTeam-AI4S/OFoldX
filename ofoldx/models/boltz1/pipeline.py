r"""OFoldX's biological operation boundary for released Boltz-1 folding."""

from __future__ import annotations

from collections.abc import Mapping
from os import PathLike

import torch

from ...data.evidence import MSA, MSACollection
from ...data.molecular import MolecularSystem
from ...data.structure import Structure
from ...pipelines import PairedPipeline, processor_device, resolve_generator
from ...results import FoldResult
from .model import Boltz1Model, Boltz1Pocket
from .processor import Boltz1Processor


class Boltz1Pipeline(PairedPipeline[Boltz1Model, Boltz1Processor]):
    r"""Bind one Boltz-1 Model-Processor pair for the released folding operation.

    Runs one Processor call followed by one ``Boltz1Model.fold`` invocation, exposing
    Boltz-1 pocket conditioning and Boltz-1x steering controls. The decoded output is the
    shared ``FoldResult`` unchanged.
    """

    def fold(
        self,
        input: MolecularSystem | Structure | str | PathLike[str],
        *,
        msa: MSA | MSACollection | str | PathLike[str] | None = None,
        pocket: Boltz1Pocket | None = None,
        num_samples: int = 1,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        recycling_steps: int | None = None,
        num_sampling_steps: int | None = None,
        use_potentials: bool = False,
        steering_args: Mapping[str, object] | None = None,
    ) -> FoldResult:
        r"""Lower one folding input and run released Boltz-1 folding into a canonical ``FoldResult``."""

        generator = resolve_generator(seed, generator, device=processor_device(self.processor))
        batch = self.processor(input, msa=msa, pocket=pocket, generator=generator)
        return self.model.fold(
            batch,
            num_samples=num_samples,
            generator=generator,
            recycling_steps=recycling_steps,
            num_sampling_steps=num_sampling_steps,
            use_potentials=use_potentials,
            steering_args=steering_args,
        )
