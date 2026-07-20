r"""The explicit folding Pipeline."""

from __future__ import annotations

from collections.abc import Sequence
from os import PathLike

import torch

from ..data.evidence import MSA, MSACollection, Template
from ..data.molecular import MolecularSystem, PolymerSequence
from ..data.structure import Structure
from ..results import FoldResult
from ._paired import PairedPipeline
from .generator import inputs_device, resolve_generator


class FoldPipeline(PairedPipeline):
    r"""Bind one folding Model to its paired Processor."""

    def fold(
        self,
        input: str | PathLike[str] | PolymerSequence | MolecularSystem | Structure,
        *,
        msa: MSA | MSACollection | None = None,
        templates: Sequence[Template] = (),
        num_samples: int = 1,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> FoldResult:
        processor_arguments: dict[str, object] = {}
        if msa is not None:
            processor_arguments["msa"] = msa
        if templates:
            processor_arguments["templates"] = templates
        inputs = self.processor(input, **processor_arguments)  # type: ignore[operator]
        generator = resolve_generator(seed, generator, device=inputs_device(inputs))
        return self.model.fold(  # type: ignore[attr-defined]
            inputs,
            num_samples=num_samples,
            generator=generator,
        )
