r"""
AlphaFold 3 folding Pipeline.
"""

from __future__ import annotations

from collections.abc import Sequence
from os import PathLike

import torch

from ...data.evidence import MSA, MSACollection, Template
from ...data.molecular import MolecularSystem, PolymerSequence
from ...data.structure import Structure
from ...pipelines import PairedPipeline, resolve_generator
from .model import AlphaFold3FoldResult, AlphaFold3Model, AlphaFold3SeedGroup
from .processor import AlphaFold3Processor


class AlphaFold3Pipeline(PairedPipeline[AlphaFold3Model, AlphaFold3Processor]):
    r"""
    Structure prediction pipeline for AlphaFold 3.
    """

    def fold(
        self,
        input: str | PathLike[str] | PolymerSequence | Sequence[PolymerSequence] | MolecularSystem | Structure,
        *,
        msa: MSA | MSACollection | str | PathLike[str] | None = None,
        templates: Sequence[Template] = (),
        num_samples: int | None = None,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        recycling_steps: int | None = None,
        num_sampling_steps: int | None = None,
    ) -> AlphaFold3FoldResult:
        r"""
        Predict structures for one molecular assembly with AlphaFold 3.

        Args:
            input: Inline sequence, FASTA or structure path, polymer sequence collection,
                molecular system, or structure topology.
            msa: Optional MSA value, collection, or alignment path.
            templates: Template structures aligned to target polymers.
            num_samples: Number of diffusion samples. ``None`` uses five.
            seed: Optional diffusion seed recorded on the returned group.
                Mutually exclusive with ``generator``.
            generator: Optional generator for initial coordinates, churn, and rigid
                augmentations. Mutually exclusive with ``seed``.
            recycling_steps: Number of representation recycling iterations. ``None`` uses the
                Model configuration.
            num_sampling_steps: Number of diffusion steps. ``None`` uses the Model configuration.

        Returns:
            One seed group containing generation-ordered predictions. Coordinates,
            PAE, and PDE values are in angstroms; pLDDT is on ``0--100``.
        """

        generator = resolve_generator(seed, generator, device=self.processor.device)
        model_input = self.processor(input, msa=msa, templates=templates)
        result = self.model.fold(
            model_input,
            num_samples=num_samples,
            generator=generator,
            recycling_steps=recycling_steps,
            num_sampling_steps=num_sampling_steps,
        )
        if seed is None:
            return result
        group = result.seed_groups[0]
        return AlphaFold3FoldResult(seed_groups=(AlphaFold3SeedGroup(seed=seed, predictions=group.predictions),))
