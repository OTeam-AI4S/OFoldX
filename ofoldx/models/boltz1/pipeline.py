r"""Boltz-1 folding Pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from os import PathLike

import torch

from ...data.evidence import MSA, MSACollection
from ...data.molecular import MolecularSystem
from ...data.structure import Structure
from ...pipelines import PairedPipeline, resolve_generator
from ...results import FoldResult
from .model import Boltz1Model, Boltz1Pocket
from .processor import Boltz1Processor


class Boltz1Pipeline(PairedPipeline[Boltz1Model, Boltz1Processor]):
    r"""Bind one Boltz-1 Model and Processor for folding.

    Runs one Processor call followed by one ``Boltz1Model.fold`` invocation, exposing
    Boltz-1 pocket conditioning and Boltz-1x steering controls. The decoded output is the
    shared ``FoldResult`` unchanged.
    """

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
        r"""Lower one molecular input and run Boltz-1 folding.

        Args:
            input: Canonical molecular system, structure, or local structure path.
            msa: Optional MSA value, collection, or alignment path.
            pocket: Optional binder and pocket-residue specification.
            num_samples: Number of structure predictions to generate.
            seed: Optional integer seed; mutually exclusive with ``generator``.
            generator: Optional PyTorch random generator; mutually exclusive with ``seed``.
            recycling_steps: Number of representation recycling iterations. ``None`` uses the
                Model configuration.
            num_sampling_steps: Number of diffusion steps. ``None`` uses the Model configuration.
            use_potentials: Whether to apply configured physical-potential guidance.
            steering_args: Optional potential-guidance controls.

        Returns:
            A ``FoldResult`` whose predictions follow generation order.
        """

        generator = resolve_generator(seed, generator, device=self.processor.device)
        batch = self.processor(input, msa=msa, pocket=pocket, generator=generator)
        return self.model.fold(
            batch,
            num_samples=num_samples,
            generator=generator,
            recycling_steps=recycling_steps,
            num_sampling_steps=num_sampling_steps,
            use_potentials=use_potentials,
            steering_args=steering_args,
        )
