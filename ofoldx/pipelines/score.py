r"""
The explicit molecular scoring Pipeline.
"""

from __future__ import annotations

from os import PathLike
from typing import Generic, Protocol, TypeVar

import torch

from ..artifacts import SaveableModel
from ..data.molecular import MolecularSystem
from ..data.structure import Structure
from ..results import ScoreResult
from .generator import PipelineProcessor, resolve_generator
from .paired import PairedPipeline

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
    r"""
    Molecular scoring pipeline for one model-processor pair.
    """

    def score(
        self,
        input: MolecularSystem | Structure | str | PathLike[str],
        *,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> ScoreResult:
        r"""
        Score one molecular input with the paired Processor and Model.

        Args:
            input: Molecular system, structure, or path to process.
            seed: Optional integer seed used to create the operation generator; mutually exclusive
                with ``generator``.
            generator: Optional caller-supplied random generator; mutually exclusive with ``seed``.

        Returns:
            A ``ScoreResult`` containing the coordinate-conditioned target and its scientific values.

        Raises:
            ValueError: If both ``seed`` and ``generator`` are supplied.
        """

        generator = resolve_generator(seed, generator, device=self.processor.device)
        model_input = self.processor(input)
        return self.model.score(
            model_input,
            generator=generator,
        )
