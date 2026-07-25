r"""
AlphaFold 3 folding Pipeline.
"""

from __future__ import annotations

from collections.abc import Sequence
from os import PathLike
from pathlib import Path

import torch

from ...data.evidence import MSA, MSACollection, Template
from ...data.molecular import MolecularSystem, PolymerSequence
from ...data.structure import Structure
from ...io import read_fasta_records, read_msa, read_structure
from ...pipelines import PairedPipeline, resolve_generator
from ...references import classify_reference
from ...results import FoldResult
from .model import AlphaFold3Model
from .processor import (
    AlphaFold3Processor,
    molecular_system_from_input,
    molecular_system_from_named_sequences,
)

_FASTA_SUFFIXES = frozenset((".fa", ".faa", ".fasta", ".fna", ".seq"))


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
            generator: Optional generator for initial coordinates, churn, and rigid
                augmentations. Mutually exclusive with ``seed``.
            recycling_steps: Number of representation recycling iterations. ``None`` uses the
                Model configuration.
            num_sampling_steps: Number of diffusion steps. ``None`` uses the Model configuration.

        Returns:
            Generation-ordered predictions. Coordinates, PAE, and PDE values are
            in angstroms; pLDDT is on ``0--100``.
        """

        generator = resolve_generator(seed, generator, device=self.processor.device)
        canonical_input = _canonical_input(input)
        system = canonical_input.system if isinstance(canonical_input, Structure) else canonical_input
        model_input = self.processor(
            canonical_input,
            msa=_canonical_msa(msa, system),
            templates=templates,
        )
        return self.model.fold(
            model_input,
            num_samples=num_samples,
            generator=generator,
            recycling_steps=recycling_steps,
            num_sampling_steps=num_sampling_steps,
        )


def _canonical_input(
    input: str | PathLike[str] | PolymerSequence | Sequence[PolymerSequence] | MolecularSystem | Structure,
) -> MolecularSystem | Structure:
    r"""
    Resolve AlphaFold 3's inline-sequence, FASTA, and structure conveniences.
    """

    if not isinstance(input, str | PathLike):
        if isinstance(input, MolecularSystem | Structure):
            return input
        return molecular_system_from_input(input)
    classified = classify_reference(input)
    if not isinstance(classified, Path):
        sequence = PolymerSequence.from_one_letter(classified.strip().upper())
        return molecular_system_from_named_sequences((("protein", sequence),))
    if any(suffix.lower() in _FASTA_SUFFIXES for suffix in classified.suffixes):
        sequences = tuple(
            (
                record.id or f"chain_{index + 1}",
                PolymerSequence.from_one_letter(record.sequence.strip().upper()),
            )
            for index, record in enumerate(read_fasta_records(classified))
        )
        return molecular_system_from_named_sequences(sequences)
    return read_structure(classified)


def _canonical_msa(
    msa: MSA | MSACollection | str | PathLike[str] | None,
    system: MolecularSystem,
) -> MSA | MSACollection | None:
    r"""
    Read a path MSA against the only unambiguous protein entity.
    """

    if not isinstance(msa, str | PathLike):
        return msa
    proteins = [
        entity_id
        for entity_id, entity in system.entities.items()
        if isinstance(entity, PolymerSequence) and entity.polymer_type == "protein"
    ]
    if len(proteins) != 1:
        raise ValueError("an aligned FASTA MSA is unambiguous only for one protein entity")
    return read_msa(Path(msa).expanduser(), entity_id=proteins[0])
