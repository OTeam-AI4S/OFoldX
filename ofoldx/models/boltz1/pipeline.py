r"""OFoldX's biological operation boundary for released Boltz-1 folding."""

from __future__ import annotations

from collections.abc import Mapping
from os import PathLike

import torch
from typing_extensions import Self

from ...data.evidence import MSA, MSACollection
from ...data.molecular import MolecularSystem
from ...data.structure import Structure
from ...pipelines import resolve_generator
from ...results import FoldResult
from .model import Boltz1Model, Boltz1Pocket
from .processor import Boltz1Processor


class Boltz1Pipeline:
    r"""Bind one Boltz-1 Model-Processor pair for the released folding operation.

    Owns the one Processor call and the one ``Boltz1Model.fold`` invocation, exposing the family-owned pocket
    conditioning and Boltz-1x steering controls that the common ``fold`` verb cannot carry. The decoded output
    is the shared ``FoldResult`` unchanged; the Pipeline is not a base class and shares no runtime request
    object with any other family.
    """

    model: Boltz1Model
    processor: Boltz1Processor

    def __init__(
        self,
        *,
        model: Boltz1Model,
        processor: Boltz1Processor,
    ) -> None:
        self.model = model
        self.processor = processor

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Self:
        r"""Place the Model and Processor on one device and dtype in place."""

        self.model.to(device=device, dtype=dtype)
        self.processor.to(device=device, dtype=dtype)
        return self

    def fold(
        self,
        input: MolecularSystem | Structure | str | PathLike[str],
        *,
        msa: MSA | MSACollection | str | PathLike[str] | None = None,
        pocket: Boltz1Pocket | None = None,
        num_samples: int = 1,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        recycling_steps: int | None = None,
        num_sampling_steps: int | None = None,
        use_potentials: bool = False,
        steering_args: Mapping[str, object] | None = None,
    ) -> FoldResult:
        r"""Lower one folding input and run released Boltz-1 folding into a canonical ``FoldResult``."""

        batch = self.processor(input, msa=msa, pocket=pocket)
        generator = resolve_generator(seed, generator, device=batch.atom_mask.device)
        return self.model.fold(
            batch,
            num_samples=num_samples,
            generator=generator,
            recycling_steps=recycling_steps,
            num_sampling_steps=num_sampling_steps,
            use_potentials=use_potentials,
            steering_args=steering_args,
        )
