r"""The explicit folding Pipeline."""

from __future__ import annotations

from collections.abc import Sequence
from os import PathLike
from typing import Generic, Protocol, TypeVar

import torch

from ..artifacts import SaveableModel
from ..data.evidence import MSA, MSACollection, Template
from ..data.molecular import MolecularSystem, PolymerSequence
from ..data.structure import Structure
from ..results import FoldResult
from ._paired import PairedPipeline
from .generator import PipelineProcessor, processor_device, resolve_generator

_ModelInputT = TypeVar("_ModelInputT")
_ProcessorOutputT_co = TypeVar("_ProcessorOutputT_co", covariant=True)
_ModelInputT_contra = TypeVar("_ModelInputT_contra", contravariant=True)


class _FoldProcessor(PipelineProcessor, Protocol[_ProcessorOutputT_co]):
    def __call__(
        self,
        input: str | PathLike[str] | PolymerSequence | MolecularSystem | Structure,
        *,
        msa: MSA | MSACollection | None = None,
        templates: Sequence[Template] = (),
    ) -> _ProcessorOutputT_co: ...


class _FoldModel(SaveableModel, Protocol[_ModelInputT_contra]):
    def fold(
        self,
        input: _ModelInputT_contra,
        *,
        num_samples: int = 1,
        generator: torch.Generator | None = None,
    ) -> FoldResult: ...


class FoldPipeline(
    PairedPipeline[_FoldModel[_ModelInputT], _FoldProcessor[_ModelInputT]],
    Generic[_ModelInputT],
):
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
        generator = resolve_generator(seed, generator, device=processor_device(self.processor))
        model_input = self.processor(input, msa=msa, templates=templates)
        return self.model.fold(
            model_input,
            num_samples=num_samples,
            generator=generator,
        )
