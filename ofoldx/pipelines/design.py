r"""The reusable Pipeline for sequence design."""

from __future__ import annotations

from collections.abc import Sequence
from os import PathLike

import torch

from ..data.conditions import AllowedResidueIdentities, TiedResidueIdentities
from ..data.evidence import MSA, MSACollection, Template
from ..data.molecular import MolecularSystem
from ..data.structure import Structure
from ..results import DesignResult
from ._generator import inputs_device, resolve_generator
from ._paired import _PairedPipeline


class DesignPipeline(_PairedPipeline):
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
        processor_arguments: dict[str, object] = {}
        if msa is not None:
            processor_arguments["msa"] = msa
        if templates:
            processor_arguments["templates"] = templates
        if conditions:
            processor_arguments["conditions"] = conditions
        inputs = self.processor(input, **processor_arguments)  # type: ignore[operator]
        generator = resolve_generator(seed, generator, device=inputs_device(inputs))
        return self.model.design(  # type: ignore[attr-defined]
            inputs,
            num_candidates=num_candidates,
            generator=generator,
        )
