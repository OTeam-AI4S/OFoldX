r"""The explicit protein side-chain packing Pipeline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from os import PathLike

import torch

from ..data.molecular import ComponentReference
from ..data.structure import Structure
from ..results import PackResult
from ._paired import PairedPipeline
from .generator import inputs_device, resolve_generator


class PackPipeline(PairedPipeline):
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
        inputs = self.processor(  # type: ignore[operator]
            input,
            sequence=sequence,
            positions=positions,
        )
        generator = resolve_generator(seed, generator, device=inputs_device(inputs))
        return self.model.pack(  # type: ignore[attr-defined]
            inputs,
            num_samples=num_samples,
            generator=generator,
        )
