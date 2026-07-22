r"""Concrete AlphaFold 2 monomer and multimer folding Pipelines."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from os import PathLike

import torch

from ...data.evidence import MSA, MSACollection, Template
from ...data.molecular import MolecularSystem, PolymerSequence
from ...data.structure import Structure
from ...pipelines import PairedPipeline, processor_device, resolve_generator
from ...results import FoldResult
from .model import AlphaFold2Model, AlphaFold2MultimerModel
from .multimer_processor import AlphaFold2MultimerProcessor
from .processor import AlphaFold2Processor


class AlphaFold2Pipeline(PairedPipeline[AlphaFold2Model, AlphaFold2Processor]):
    r"""Bind the released AlphaFold 2 monomer Model and Processor."""

    def fold(
        self,
        input: str | PathLike[str] | PolymerSequence | Sequence[PolymerSequence] | MolecularSystem | Structure,
        *,
        msa: MSA | MSACollection | str | PathLike[str] | None = None,
        templates: Sequence[Template] = (),
        num_samples: int = 1,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        recycling_steps: int | None = None,
    ) -> FoldResult:
        r"""Lower one monomer and fold it with configured representation ensembles."""

        generator = resolve_generator(seed, generator, device=processor_device(self.processor))
        features = self.processor(input, msa=msa, templates=templates)
        return self.model.fold(
            features,
            num_samples=num_samples,
            generator=generator,
            recycling_steps=recycling_steps,
        )


class AlphaFold2MultimerPipeline(PairedPipeline[AlphaFold2MultimerModel, AlphaFold2MultimerProcessor]):
    r"""Bind the released AlphaFold-Multimer Model and Processor."""

    def fold(
        self,
        input: MolecularSystem | Sequence[PolymerSequence] | Mapping[str, PolymerSequence],
        *,
        msa: MSACollection | None = None,
        templates: Sequence[Template] = (),
        num_samples: int = 1,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        recycling_steps: int | None = None,
    ) -> FoldResult:
        r"""Lower one assembly and fold it with configured representation ensembles."""

        generator = resolve_generator(seed, generator, device=processor_device(self.processor))
        features = self.processor(input, msa=msa, templates=templates)
        return self.model.fold(
            features,
            num_samples=num_samples,
            generator=generator,
            recycling_steps=recycling_steps,
        )
