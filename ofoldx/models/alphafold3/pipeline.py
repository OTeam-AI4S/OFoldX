r"""
AlphaFold 3 folding Pipeline.
"""

from __future__ import annotations

from collections.abc import Sequence
from os import PathLike

import torch

from ...data.evidence import MSA, MsaCollection, Template
from ...data.molecular import ENTITY_PROTEIN, ENTITY_RNA, MolecularSystem, PolymerSequence
from ...data.structure import Structure
from ...pipelines import PairedPipeline, resolve_generator
from ...pipelines.input import resolve_msa_input, resolve_polymer_input
from ...results import FoldResult
from .model import AlphaFold3Model
from .processor import AlphaFold3Processor, molecular_system_from_named_sequences


class AlphaFold3Pipeline(PairedPipeline[AlphaFold3Model, AlphaFold3Processor]):
    r"""
    Structure prediction pipeline for AlphaFold 3.
    """

    model_architecture = "alphafold3"

    def fold(
        self,
        input: str | PathLike[str] | PolymerSequence | Sequence[PolymerSequence] | MolecularSystem | Structure,
        *,
        msa: MSA | MsaCollection | str | PathLike[str] | None = None,
        templates: Sequence[Template] = (),
        num_samples: int | None = None,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        recycling_steps: int | None = None,
        num_sampling_steps: int | None = None,
    ) -> FoldResult:
        r"""
        Predict structures for one molecular assembly with AlphaFold 3.

        Args:
            input: Inline sequence, FASTA or structure path, polymer sequence collection,
                molecular system, or structure topology.
            msa: Optional MSA value, collection, or alignment path.
            templates: Template structures aligned to target polymers.
            num_samples: Number of diffusion samples. ``None`` uses five.
            seed: Optional diffusion seed, mutually exclusive with ``generator``.
            generator: Optional generator for per-recycle MSA permutations, initial
                coordinates, churn, and rigid augmentations. Mutually exclusive with ``seed``.
            recycling_steps: Number of complete trunk passes. ``None`` uses the
                Model configuration.
            num_sampling_steps: Number of diffusion steps. ``None`` uses the Model configuration.

        Returns:
            Generation-ordered predictions. Coordinates, PAE, and PDE values are
            in angstroms; pLDDT is on ``0--100``.
        """

        generator = resolve_generator(seed, generator, device=self.processor.device)
        resolved_input = resolve_polymer_input(input)
        canonical_input = (
            resolved_input
            if isinstance(resolved_input, MolecularSystem | Structure)
            else molecular_system_from_named_sequences(
                resolved_input,
                revision=self.processor.config.ccd_revision,
            )
        )
        system = canonical_input.system if isinstance(canonical_input, Structure) else canonical_input
        model_input = self.processor(
            canonical_input,
            msa=resolve_msa_input(msa, system, entity_types={ENTITY_PROTEIN, ENTITY_RNA}),
            templates=templates,
        )
        return self.model.fold(
            model_input,
            num_samples=num_samples,
            generator=generator,
            recycling_steps=recycling_steps,
            num_sampling_steps=num_sampling_steps,
        )
