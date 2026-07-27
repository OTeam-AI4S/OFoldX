r"""
Concrete AlphaFold 2 monomer and multimer folding Pipelines.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Literal

import torch

from ...data.evidence import MSA, MSACollection, Template
from ...data.molecular import MolecularSystem, PolymerSequence
from ...data.structure import Structure
from ...io import read_fasta_records, read_msa
from ...pipelines import CompositePair, CompositePipeline, PairedPipeline, resolve_generator
from ...references import classify_reference
from ...results import FoldPrediction, FoldResult
from .model import AlphaFold2InitialGuessModel, AlphaFold2Model, AlphaFold2MultimerModel
from .multimer_processor import AlphaFold2MultimerProcessor
from .processor import (
    AlphaFold2InitialGuessProcessor,
    AlphaFold2Processor,
    monomer_system_from_sequences,
    protein_monomer,
)
from .relax import AlphaFold2AmberRelaxation, AlphaFold2RelaxationResult

_FASTA_SUFFIXES = frozenset((".fa", ".faa", ".fasta", ".seq"))


class AlphaFold2Pipeline(PairedPipeline[AlphaFold2Model, AlphaFold2Processor]):
    r"""
    Structure prediction pipeline for AlphaFold 2.
    """

    model_architecture = "alphafold2"

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
            recycling_steps: Total network passes, including the final pass.
                ``None`` uses the Model configuration.

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


class AlphaFold2InitialGuessPipeline(PairedPipeline[AlphaFold2InitialGuessModel, AlphaFold2InitialGuessProcessor]):
    r"""
    Initial-guess target--binder evaluation with the monomer/pTM graph.
    """

    model_architecture = "alphafold2_initial_guess"

    def fold(
        self,
        input: Structure,
        *,
        binder_instance_id: str,
        num_samples: int = 1,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        num_recycles: int = 3,
    ) -> FoldResult:
        r"""
        Predict a complex from its generated atom37 recycling state.

        Args:
            input: Sequence-assigned generated target--binder complex.
            binder_instance_id: Exact protein instance treated as the binder.
            num_samples: Number of predictions.
            seed: Optional MSA sampling seed, mutually exclusive with ``generator``.
            generator: Optional MSA sampling generator, mutually exclusive with
                ``seed``.
            num_recycles: Preliminary recycling iterations. The Model always runs
                one additional final pass, so the released default executes four
                total passes.

        Returns:
            Complex predictions with binder pLDDT, pTM, iPTM, and raw and
            normalized interface PAE scores.
        """

        generator = resolve_generator(seed, generator, device=self.processor.device)
        model_input = self.processor(
            input,
            binder_instance_id=binder_instance_id,
        )
        return self.model.fold(
            model_input,
            num_samples=num_samples,
            generator=generator,
            recycling_steps=num_recycles + 1,
        )


class AlphaFold2MultimerPipeline(PairedPipeline[AlphaFold2MultimerModel, AlphaFold2MultimerProcessor]):
    r"""
    Multimer structure prediction pipeline for AlphaFold-Multimer.
    """

    model_architecture = "alphafold2_multimer"

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
            recycling_steps: Total network passes, including the final pass.
                ``None`` uses the Model configuration.

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


@dataclass(eq=False)
class AlphaFold2Casp14Prediction:
    r"""
    One ranked prediction from the fixed five-model CASP14 ensemble.

    Args:
        model_name: Official checkpoint name, from ``"model_1"`` through
            ``"model_5"``.
        sample_index: Zero-based sample index within that checkpoint.
        prediction: Decoded AlphaFold 2 prediction.
        relaxation: Amber result when this ranked prediction was selected for
            relaxation.
    """

    model_name: str
    sample_index: int
    prediction: FoldPrediction
    relaxation: AlphaFold2RelaxationResult | None = None


@dataclass(eq=False)
class AlphaFold2Casp14Result:
    r"""
    Complete stable ranking from the fixed five-model CASP14 ensemble.

    Args:
        ranked: Predictions in descending mean-pLDDT order, with checkpoint and
            sample generation order breaking ties.
    """

    ranked: tuple[AlphaFold2Casp14Prediction, ...]

    @property
    def best(self) -> AlphaFold2Casp14Prediction:
        r"""
        Return the highest-ranked prediction.
        """

        return self.ranked[0]


class AlphaFold2Casp14Pipeline(CompositePipeline):
    r"""
    Fixed five-checkpoint AlphaFold 2 CASP14 inference and relaxation.

    Args:
        model_1: Ordinary eight-ensemble ``model_1`` monomer pipeline.
        model_2: Ordinary eight-ensemble ``model_2`` monomer pipeline.
        model_3: Ordinary eight-ensemble ``model_3`` monomer pipeline.
        model_4: Ordinary eight-ensemble ``model_4`` monomer pipeline.
        model_5: Ordinary eight-ensemble ``model_5`` monomer pipeline.
    """

    model_1: AlphaFold2Pipeline
    model_2: AlphaFold2Pipeline
    model_3: AlphaFold2Pipeline
    model_4: AlphaFold2Pipeline
    model_5: AlphaFold2Pipeline
    relaxation: AlphaFold2AmberRelaxation

    pair_layout = (
        CompositePair("", "alphafold2", "model_1", None, wrap=AlphaFold2Pipeline),
        CompositePair("model_2", "alphafold2", "model_2", None, wrap=AlphaFold2Pipeline),
        CompositePair("model_3", "alphafold2", "model_3", None, wrap=AlphaFold2Pipeline),
        CompositePair("model_4", "alphafold2", "model_4", None, wrap=AlphaFold2Pipeline),
        CompositePair("model_5", "alphafold2", "model_5", None, wrap=AlphaFold2Pipeline),
    )

    def __init__(
        self,
        *,
        model_1: AlphaFold2Pipeline,
        model_2: AlphaFold2Pipeline,
        model_3: AlphaFold2Pipeline,
        model_4: AlphaFold2Pipeline,
        model_5: AlphaFold2Pipeline,
    ) -> None:
        self.model_1 = model_1
        self.model_2 = model_2
        self.model_3 = model_3
        self.model_4 = model_4
        self.model_5 = model_5
        self.relaxation = AlphaFold2AmberRelaxation()
        self._validate_components()
        for model_name, pipeline in self._models_in_generation_order():
            if pipeline.model.config.num_ensembles != 8:
                raise ValueError(f"{model_name} must use the CASP14 eight-ensemble configuration")

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
        models_to_relax: Literal["none", "best", "all"] = "best",
        platform: str = "CPU",
        platform_properties: Mapping[str, str] | None = None,
    ) -> AlphaFold2Casp14Result:
        r"""
        Predict, rank, and optionally relax the fixed CASP14 five-model ensemble.

        The biological input is lowered once through ``model_1``'s Processor.
        Every checkpoint consumes that same tensor input sequentially.

        Args:
            input: Inline sequence, FASTA path, protein sequence, molecular
                system, or structure topology containing one protein instance.
            msa: Optional monomer MSA value or alignment path.
            templates: Template structures with explicit target alignments.
            num_samples: Number of predictions generated by each checkpoint.
            seed: Optional MSA sampling seed, mutually exclusive with
                ``generator``.
            generator: Optional MSA sampling generator, mutually exclusive with
                ``seed``.
            recycling_steps: Total network passes, including the final pass.
                ``None`` uses each Model configuration.
            models_to_relax: Whether Amber relaxation is applied to no
                prediction, only the best prediction, or every prediction.
            platform: Exact OpenMM platform name passed to Amber relaxation.
            platform_properties: Properties passed directly to the selected
                OpenMM platform.

        Returns:
            Every prediction in stable descending mean-pLDDT order, including
            checkpoint identity, sample identity, and any relaxation result.
        """

        if models_to_relax not in ("none", "best", "all"):
            raise ValueError("models_to_relax must be 'none', 'best', or 'all'")
        generator = resolve_generator(seed, generator, device=self.model_1.processor.device)
        canonical_input = _monomer_input(input)
        system = canonical_input.system if isinstance(canonical_input, Structure) else canonical_input
        model_input = self.model_1.processor(
            canonical_input,
            msa=_monomer_msa(msa, system),
            templates=templates,
        )

        generated: list[AlphaFold2Casp14Prediction] = []
        for model_name, pipeline in self._models_in_generation_order():
            result = pipeline.model.fold(
                model_input,
                num_samples=num_samples,
                generator=generator,
                recycling_steps=recycling_steps,
            )
            generated.extend(
                AlphaFold2Casp14Prediction(
                    model_name=model_name,
                    sample_index=sample_index,
                    prediction=prediction,
                )
                for sample_index, prediction in enumerate(result.predictions)
            )
        ranked = tuple(
            sorted(
                generated,
                key=lambda item: item.prediction.scores["mean_plddt"].value,
                reverse=True,
            )
        )

        if models_to_relax == "none":
            relax_indices: tuple[int, ...] = ()
        elif models_to_relax == "best":
            relax_indices = (0,)
        else:
            relax_indices = tuple(range(len(ranked)))

        relaxed = list(ranked)
        for index in relax_indices:
            candidate = relaxed[index]
            relaxed[index] = AlphaFold2Casp14Prediction(
                model_name=candidate.model_name,
                sample_index=candidate.sample_index,
                prediction=candidate.prediction,
                relaxation=self.relaxation.relax(
                    candidate.prediction.structure,
                    platform=platform,
                    platform_properties=platform_properties,
                ),
            )
        return AlphaFold2Casp14Result(ranked=tuple(relaxed))

    def _models_in_generation_order(self) -> tuple[tuple[str, AlphaFold2Pipeline], ...]:
        return (
            ("model_1", self.model_1),
            ("model_2", self.model_2),
            ("model_3", self.model_3),
            ("model_4", self.model_4),
            ("model_5", self.model_5),
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
