r"""The explicit molecular scoring Pipeline."""

from __future__ import annotations

from os import PathLike

import torch

from ..data.molecular import MolecularSystem
from ..data.structure import Structure
from ..results import ScoreResult
from ._paired import PairedPipeline
from .generator import inputs_device, resolve_generator


class ScorePipeline(PairedPipeline):
    r"""Bind one scoring Model to its paired Processor."""

    def score(
        self,
        input: MolecularSystem | Structure | str | PathLike[str],
        *,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> ScoreResult:
        inputs = self.processor(input)  # type: ignore[operator]
        generator = resolve_generator(seed, generator, device=inputs_device(inputs))
        return self.model.score(  # type: ignore[attr-defined]
            inputs,
            generator=generator,
        )
