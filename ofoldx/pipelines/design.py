r"""
The reusable Pipeline for sequence design.
"""

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
from .generator import PipelineProcessor, resolve_generator
from .paired import PairedPipeline

_ModelInputT = TypeVar("_ModelInputT")
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
    PairedPipeline[_DesignModel[_ModelInputT], _DesignProcessor[_ModelInputT]],
    Generic[_ModelInputT],
):
    r"""
    Molecular design pipeline for one model-processor pair.
    """

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
        r"""
        Design fixed-length sequences with the paired Processor and Model.

        Args:
            input: Molecular system, structure, or path to process.
            msa: Optional multiple-sequence-alignment evidence.
            templates: Template evidence to process with ``input``.
            conditions: Residue-identity conditions to apply during design.
            num_candidates: Positive number of design candidates to generate.
            seed: Optional integer seed used to create the operation generator; mutually exclusive
                with ``generator``.
            generator: Optional caller-supplied random generator; mutually exclusive with ``seed``.

        Returns:
            A ``DesignResult`` containing generation-ordered fixed-length designs.

        Raises:
            ValueError: If both ``seed`` and ``generator`` are supplied.
        """

        generator = resolve_generator(seed, generator, device=self.processor.device)
        model_input = self.processor(input, msa=msa, templates=templates, conditions=conditions)
        return self.model.design(
            model_input,
            num_candidates=num_candidates,
            generator=generator,
        )
