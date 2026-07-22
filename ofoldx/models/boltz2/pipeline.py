r"""Pipelines for Boltz-2 folding and two-stage affinity scoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from os import PathLike

import torch

from ...data.evidence import MSA, MSACollection
from ...data.molecular import MolecularSystem
from ...data.structure import Structure
from ...pipelines import CompositePair, CompositePipeline, PairedPipeline, processor_device, resolve_generator
from ...results import FoldPrediction, FoldResult, Metric, Score
from .model import Boltz2AffinityModel, Boltz2Model
from .processor import Boltz2AffinityProcessor, Boltz2Processor


class Boltz2Pipeline(PairedPipeline[Boltz2Model, Boltz2Processor]):
    r"""Bind the Boltz-2 structure Model and Processor."""

    def fold(
        self,
        input: MolecularSystem | Structure | str | PathLike[str],
        *,
        msa: MSA | MSACollection | str | PathLike[str] | None = None,
        num_samples: int = 1,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        recycling_steps: int | None = None,
        num_sampling_steps: int | None = None,
    ) -> FoldResult:
        r"""Lower one input and run Boltz-2 folding."""

        generator = resolve_generator(seed, generator, device=processor_device(self.processor))
        features = self.processor(input, msa=msa, generator=generator)
        return self.model.fold(
            features,
            num_samples=num_samples,
            generator=generator,
            recycling_steps=recycling_steps,
            num_sampling_steps=num_sampling_steps,
        )


class Boltz2AffinityPipeline(CompositePipeline):
    r"""Run Boltz-2 folding followed by affinity scoring.

    The pipeline owns separate folding and affinity Model/Processor pairs. ``score``
    folds the complete system, ranks predictions by ``confidence_score``, and passes
    the highest-ranked Structure to the affinity pair. The two passes have independent
    sampling counts and default recycling settings, share one RNG stream, and both
    receive the supplied MSA.

    The result preserves the confidence-sorted folding predictions alongside the
    affinity target, scores, and metrics.
    """

    model: Boltz2AffinityModel
    processor: Boltz2AffinityProcessor
    structure_model: Boltz2Model
    structure_processor: Boltz2Processor

    # The affinity pass owns the repository root; the structure/confidence pair is stored in
    # the fixed ``structure/`` subdirectory.
    _PAIR_LAYOUT = (
        CompositePair("", "model", "processor"),
        CompositePair("structure", "structure_model", "structure_processor"),
    )

    def __init__(
        self,
        *,
        model: Boltz2AffinityModel,
        processor: Boltz2AffinityProcessor,
        structure_model: Boltz2Model,
        structure_processor: Boltz2Processor,
    ) -> None:
        self.model = model
        self.processor = processor
        self.structure_model = structure_model
        self.structure_processor = structure_processor

    def score(
        self,
        input: MolecularSystem | Structure | str | PathLike[str],
        *,
        ligand: str | None = None,
        msa: MSA | MSACollection | str | PathLike[str] | None = None,
        num_samples: int = 1,
        num_sampling_steps: int | None = None,
        affinity_num_samples: int = 5,
        affinity_num_sampling_steps: int | None = None,
        molecular_weight_correction: bool = False,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> Boltz2AffinityResult:
        r"""Run two-stage Boltz-2 affinity selection and return a flat ``Boltz2AffinityResult``.

        Args:
            input: The molecular system, canonical ``Structure``, or path lowered by the structure Processor.
            ligand: The affinity ligand instance selector consumed by the affinity Processor.
            msa: MSA evidence routed to both Processors.
            num_samples: Structure-pass diffusion samples.
            num_sampling_steps: Structure-pass EDM steps.
            affinity_num_samples: Affinity-pass diffusion samples.
            affinity_num_sampling_steps: Affinity-pass EDM steps.
            molecular_weight_correction: Whether to apply molecular-weight calibration.
            seed: Optional integer seed shared by both passes; mutually exclusive with
                ``generator``.
            generator: Optional RNG shared across both passes; mutually exclusive with ``seed``.
        """

        # Reference augmentation and both model passes advance one RNG stream.
        generator = resolve_generator(seed, generator, device=processor_device(self.structure_processor))

        # The structure and affinity passes use independent sampling counts.
        fold_batch = self.structure_processor(input, msa=msa, generator=generator)

        fold_result = self.structure_model.fold(
            fold_batch,
            num_samples=num_samples,
            num_sampling_steps=num_sampling_steps,
            generator=generator,
        )

        # Rank first-stage predictions by descending confidence_score; Python's stable sort keeps generation
        # order among ties (earliest-tie stable), so structure_predictions is confidence-descending, not
        # generation-ordered.
        structure_predictions = tuple(
            sorted(
                fold_result.predictions,
                key=lambda prediction: prediction.scores["confidence_score"].value,
                reverse=True,
            )
        )
        best_structure = structure_predictions[0].structure

        affinity_batch = self.processor(best_structure, ligand=ligand, msa=msa, generator=generator)
        score_result = self.model.score(
            affinity_batch,
            num_samples=affinity_num_samples,
            num_sampling_steps=affinity_num_sampling_steps,
            molecular_weight_correction=molecular_weight_correction,
            generator=generator,
        )
        return Boltz2AffinityResult(
            target=score_result.target,
            structure_predictions=structure_predictions,
            scores=score_result.scores,
            metrics=score_result.metrics,
        )


@dataclass(eq=False)
class Boltz2AffinityResult:
    r"""One two-stage Boltz-2 affinity result.

    Contains the selected second-pass ``target``, confidence-sorted first-pass
    ``structure_predictions``, and affinity scores and metrics.
    """

    target: Structure
    structure_predictions: tuple[FoldPrediction, ...]
    scores: dict[str, Score] = field(default_factory=dict)
    metrics: dict[str, Metric] = field(default_factory=dict)
