r"""The explicit protein side-chain packing Pipeline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from os import PathLike
from typing import Generic, Protocol, TypeVar

import torch

from ..artifacts import SaveableModel
from ..data.molecular import ComponentReference
from ..data.structure import Structure
from ..results import PackResult
from ._paired import PairedPipeline
from .generator import PipelineProcessor, processor_device, resolve_generator

_FeaturesT = TypeVar("_FeaturesT")
_ProcessorOutputT_co = TypeVar("_ProcessorOutputT_co", covariant=True)
_ModelInputT_contra = TypeVar("_ModelInputT_contra", contravariant=True)


class _PackProcessor(PipelineProcessor, Protocol[_ProcessorOutputT_co]):
    def __call__(
        self,
        input: Structure | str | PathLike[str],
        *,
        sequence: str | Mapping[str, str | Sequence[str]] | None = None,
        positions: Sequence[ComponentReference] | None = None,
    ) -> _ProcessorOutputT_co: ...


class _PackModel(SaveableModel, Protocol[_ModelInputT_contra]):
    def pack(
        self,
        input: _ModelInputT_contra,
        *,
        num_samples: int = 1,
        generator: torch.Generator | None = None,
    ) -> PackResult: ...


class PackPipeline(
    PairedPipeline[_PackModel[_FeaturesT], _PackProcessor[_FeaturesT]],
    Generic[_FeaturesT],
):
    r"""Bind one side-chain packing Model to its paired Processor."""

    def pack(
        self,
        input: Structure | str | PathLike[str],
        *,
        sequence: str | Mapping[str, str | Sequence[str]] | None = None,
        positions: Sequence[ComponentReference] | None = None,
        num_samples: int = 1,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> PackResult:
        generator = resolve_generator(seed, generator, device=processor_device(self.processor))
        features = self.processor(input, sequence=sequence, positions=positions)
        return self.model.pack(
            features,
            num_samples=num_samples,
            generator=generator,
        )
