r"""
Concrete AlphaFold 2 monomer and multimer folding Pipelines.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from os import PathLike

import torch

from ...data.evidence import MSA, MSACollection, Template
from ...data.molecular import MolecularSystem, PolymerSequence
from ...data.structure import Structure
from ...pipelines import PairedPipeline, resolve_generator
from ...results import FoldResult
from .model import AlphaFold2Model, AlphaFold2MultimerModel
from .multimer_processor import AlphaFold2MultimerProcessor
from .processor import AlphaFold2Processor


class AlphaFold2Pipeline(PairedPipeline[AlphaFold2Model, AlphaFold2Processor]):
    r"""
    Structure prediction pipeline for AlphaFold 2.
    """

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
        r"""
        Predict the structure of one protein with AlphaFold 2.

        Args:
            input: Inline sequence, FASTA path, protein sequence, molecular system, or structure
                topology containing exactly one protein instance.
            msa: Optional monomer MSA value or alignment path.
            templates: Template structures with explicit target alignments.
            num_samples: Number of structure predictions to generate.
            seed: Optional seed for stochastic MSA feature construction. Mutually
                exclusive with ``generator``.
            generator: Optional generator for stochastic MSA feature construction.
                Mutually exclusive with ``seed``.
            recycling_steps: Number of representation recycling iterations. ``None`` uses the
                Model configuration.

        Returns:
            Generation-ordered structure predictions. Coordinates and PAE values
            are in angstroms; pLDDT is on the ``0--100`` scale.
        """

        generator = resolve_generator(seed, generator, device=self.processor.device)
        model_input = self.processor(input, msa=msa, templates=templates)
        return self.model.fold(
            model_input,
            num_samples=num_samples,
            generator=generator,
            recycling_steps=recycling_steps,
        )


class AlphaFold2MultimerPipeline(PairedPipeline[AlphaFold2MultimerModel, AlphaFold2MultimerProcessor]):
    r"""
    Multimer structure prediction pipeline for AlphaFold-Multimer.
    """

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
        r"""
        Predict the structure of one protein assembly with AlphaFold-Multimer.

        Args:
            input: Molecular system or named/ordered protein sequences forming one assembly.
            msa: Optional multimer MSA collection.
            templates: Template structures with explicit target alignments.
            num_samples: Number of structure predictions to generate.
            seed: Optional seed for stochastic MSA feature construction. Mutually
                exclusive with ``generator``.
            generator: Optional generator for stochastic MSA feature construction.
                Mutually exclusive with ``seed``.
            recycling_steps: Number of representation recycling iterations. ``None`` uses the
                Model configuration.

        Returns:
            Generation-ordered structure predictions. Coordinates and PAE values
            are in angstroms; pLDDT is on the ``0--100`` scale.
        """

        generator = resolve_generator(seed, generator, device=self.processor.device)
        model_input = self.processor(input, msa=msa, templates=templates)
        return self.model.fold(
            model_input,
            num_samples=num_samples,
            generator=generator,
            recycling_steps=recycling_steps,
        )
