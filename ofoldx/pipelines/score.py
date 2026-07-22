r"""The explicit molecular scoring Pipeline."""

from __future__ import annotations

from os import PathLike
from typing import Generic, Protocol, TypeVar

import torch

from ..artifacts import SaveableModel
from ..data.molecular import MolecularSystem
from ..data.structure import Structure
from ..results import ScoreResult
from ._paired import PairedPipeline
from .generator import PipelineProcessor, processor_device, resolve_generator

_ModelInputT = TypeVar("_ModelInputT")
_ProcessorOutputT_co = TypeVar("_ProcessorOutputT_co", covariant=True)
_ModelInputT_contra = TypeVar("_ModelInputT_contra", contravariant=True)


class _ScoreProcessor(PipelineProcessor, Protocol[_ProcessorOutputT_co]):
    def __call__(
        self,
        input: MolecularSystem | Structure | str | PathLike[str],
    ) -> _ProcessorOutputT_co: ...


class _ScoreModel(SaveableModel, Protocol[_ModelInputT_contra]):
    def score(
        self,
        input: _ModelInputT_contra,
        *,
        generator: torch.Generator | None = None,
    ) -> ScoreResult: ...


class ScorePipeline(
    PairedPipeline[_ScoreModel[_ModelInputT], _ScoreProcessor[_ModelInputT]],
    Generic[_ModelInputT],
):
    r"""Bind one scoring Model to its paired Processor."""

    def score(
        self,
        input: MolecularSystem | Structure | str | PathLike[str],
        *,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> ScoreResult:
        generator = resolve_generator(seed, generator, device=processor_device(self.processor))
        model_input = self.processor(input)
        return self.model.score(
            model_input,
            generator=generator,
        )
