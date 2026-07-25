r"""
Concrete AlphaFold 2 monomer and multimer folding Pipelines.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from os import PathLike
from pathlib import Path

import torch

from ...data.evidence import MSA, MSACollection, Template
from ...data.molecular import MolecularSystem, PolymerSequence
from ...data.structure import Structure
from ...io import read_fasta_records, read_msa
from ...pipelines import PairedPipeline, resolve_generator
from ...references import classify_reference
from ...results import FoldResult
from .model import AlphaFold2Model, AlphaFold2MultimerModel
from .multimer_processor import AlphaFold2MultimerProcessor
from .processor import AlphaFold2Processor, monomer_system_from_sequences, protein_monomer

_FASTA_SUFFIXES = frozenset((".fa", ".faa", ".fasta", ".seq"))


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
        canonical_input = _monomer_input(input)
        system = canonical_input.system if isinstance(canonical_input, Structure) else canonical_input
        model_input = self.processor(
            canonical_input,
            msa=_monomer_msa(msa, system),
            templates=templates,
        )
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


def _monomer_input(
    input: str | PathLike[str] | PolymerSequence | Sequence[PolymerSequence] | MolecularSystem | Structure,
) -> MolecularSystem | Structure:
    r"""
    Resolve AlphaFold 2's inline-sequence and FASTA conveniences.
    """

    if isinstance(input, str | PathLike):
        classified = classify_reference(input)
        if isinstance(classified, Path):
            if not any(suffix.lower() in _FASTA_SUFFIXES for suffix in classified.suffixes):
                raise ValueError(
                    "AlphaFold 2 monomer cannot consume a coordinate-bearing structure path; use a FASTA path, "
                    "input.system, or templates="
                )
            named = [
                (record.id or "protein", PolymerSequence.from_one_letter(record.sequence.strip().upper()))
                for record in read_fasta_records(classified)
            ]
            if len(named) != 1:
                raise ValueError("AlphaFold 2 monomer requires exactly one PolymerSequence")
            entity_id, sequence = named[0]
            return MolecularSystem(
                entities={entity_id: sequence},
                instances={entity_id: entity_id},
            )
        return monomer_system_from_sequences(PolymerSequence.from_one_letter(classified.strip().upper()))
    if isinstance(input, Structure):
        return input
    if isinstance(input, MolecularSystem):
        return input
    return monomer_system_from_sequences(input)


def _monomer_msa(
    msa: MSA | MSACollection | str | PathLike[str] | None,
    system: MolecularSystem,
) -> MSA | MSACollection | None:
    r"""
    Read path evidence against AlphaFold 2's one exact protein entity.
    """

    if not isinstance(msa, str | PathLike):
        return msa
    _target, instance_id = protein_monomer(system)
    return read_msa(Path(msa).expanduser(), entity_id=system.instances[instance_id])
