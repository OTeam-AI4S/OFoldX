r"""
The explicit folding Pipeline.
"""

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
from .generator import PipelineProcessor, resolve_generator
from .paired import PairedPipeline

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
    r"""
    Structure prediction pipeline for one model-processor pair.
    """

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
        r"""
        Fold one biological input with the paired Processor and Model.

        Args:
            input: Sequence, molecular system, structure, or path to process.
            msa: Optional multiple-sequence-alignment evidence.
            templates: Template evidence to process with ``input``.
            num_samples: Positive number of folding samples to generate.
            seed: Optional integer seed used to create the operation generator; mutually exclusive
                with ``generator``.
            generator: Optional caller-supplied random generator; mutually exclusive with ``seed``.

        Returns:
            A ``FoldResult`` containing generation-ordered folding predictions.

        Raises:
            ValueError: If both ``seed`` and ``generator`` are supplied.
        """

        generator = resolve_generator(seed, generator, device=self.processor.device)
        model_input = self.processor(input, msa=msa, templates=templates)
        return self.model.fold(
            model_input,
            num_samples=num_samples,
            generator=generator,
        )
