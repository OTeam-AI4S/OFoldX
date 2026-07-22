r"""The reusable Pipeline for sequence design."""

from __future__ import annotations

from collections.abc import Sequence
from os import PathLike
from typing import Generic, Protocol, TypeVar

import torch

from ..artifacts import SaveableModel
from ..data.conditions import AllowedResidueIdentities, TiedResidueIdentities
from ..data.evidence import MSA, MSACollection, Template
from ..data.molecular import MolecularSystem
from ..data.structure import Structure
from ..results import DesignResult
from ._paired import PairedPipeline
from .generator import PipelineProcessor, processor_device, resolve_generator

_FeaturesT = TypeVar("_FeaturesT")
_ProcessorOutputT_co = TypeVar("_ProcessorOutputT_co", covariant=True)
_ModelInputT_contra = TypeVar("_ModelInputT_contra", contravariant=True)


class _DesignProcessor(PipelineProcessor, Protocol[_ProcessorOutputT_co]):
    def __call__(
        self,
        input: MolecularSystem | Structure | str | PathLike[str],
        *,
        msa: MSA | MSACollection | None = None,
        templates: Sequence[Template] = (),
        conditions: Sequence[AllowedResidueIdentities | TiedResidueIdentities] = (),
    ) -> _ProcessorOutputT_co: ...


class _DesignModel(SaveableModel, Protocol[_ModelInputT_contra]):
    def design(
        self,
        input: _ModelInputT_contra,
        *,
        num_candidates: int = 1,
        generator: torch.Generator | None = None,
    ) -> DesignResult: ...


class DesignPipeline(
    PairedPipeline[_DesignModel[_FeaturesT], _DesignProcessor[_FeaturesT]],
    Generic[_FeaturesT],
):
    r"""Reusable binding of one design Model and its paired Processor."""

    def design(
        self,
        input: MolecularSystem | Structure | str | PathLike[str],
        *,
        msa: MSA | MSACollection | None = None,
        templates: Sequence[Template] = (),
        conditions: Sequence[AllowedResidueIdentities | TiedResidueIdentities] = (),
        num_candidates: int = 1,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> DesignResult:
        generator = resolve_generator(seed, generator, device=processor_device(self.processor))
        features = self.processor(input, msa=msa, templates=templates, conditions=conditions)
        return self.model.design(
            features,
            num_candidates=num_candidates,
            generator=generator,
        )
