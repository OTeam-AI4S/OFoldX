r"""
Pipelines for Boltz-2 folding and two-stage affinity scoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from os import PathLike

import torch

from ...data.evidence import MSA, MSACollection
from ...data.molecular import MolecularSystem
from ...data.structure import Structure
from ...pipelines import CompositePair, CompositePipeline, PairedPipeline, resolve_generator
from ...results import FoldPrediction, FoldResult, Metric, Score
from .model import Boltz2AffinityModel, Boltz2Model
from .processor import Boltz2AffinityProcessor, Boltz2Processor


class Boltz2Pipeline(PairedPipeline[Boltz2Model, Boltz2Processor]):
    r"""
    Structure prediction pipeline for Boltz 2.
    """

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
        r"""
        Predict structures for one molecular system with Boltz 2.

        Args:
            input: Molecular system, structure, or local structure path.
            msa: Optional MSA value, collection, or alignment path.
            num_samples: Number of structure predictions to generate.
            seed: Optional seed for reference-conformer augmentation and
                coordinate sampling. Mutually exclusive with ``generator``.
            generator: Optional generator advanced by preprocessing and coordinate
                sampling. Mutually exclusive with ``seed``.
            recycling_steps: Number of representation recycling iterations. ``None`` uses the
                Model configuration.
            num_sampling_steps: Number of diffusion steps. ``None`` uses the Model configuration.

        Returns:
            Generation-ordered structure predictions. Structure, PAE, and PDE values
            are in angstroms; pLDDT is on the ``0--100`` scale.
        """

        generator = resolve_generator(seed, generator, device=self.processor.device)
        features = self.processor(input, msa=msa, generator=generator)
        return self.model.fold(
            features,
            num_samples=num_samples,
            generator=generator,
            recycling_steps=recycling_steps,
            num_sampling_steps=num_sampling_steps,
        )


class Boltz2AffinityPipeline(CompositePipeline):
    r"""
    Two-pass structure prediction and binding-affinity pipeline for Boltz 2.

    Args:
        model: Affinity prediction model.
        processor: Processor for the affinity pass.
        structure_model: Structure prediction model.
        structure_processor: Processor for the structure prediction pass.
    """

    model: Boltz2AffinityModel
    processor: Boltz2AffinityProcessor
    structure_model: Boltz2Model
    structure_processor: Boltz2Processor

    # The affinity pass owns the repository root; the structure/confidence pair is stored in
    # the fixed ``structure/`` subdirectory.
    pair_layout = (
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
        r"""
        Predict candidate structures and score protein-ligand affinity.

        Args:
            input: Molecular system, structure, or local structure path.
            ligand: Instance ID of the molecule to score. May be omitted when the
                system contains exactly one molecule.
            msa: Optional MSA value, collection, or alignment path used by both passes.
            num_samples: Number of first-pass structure samples.
            num_sampling_steps: Number of first-pass diffusion transitions.
            affinity_num_samples: Number of second-pass candidate samples.
            affinity_num_sampling_steps: Number of second-pass diffusion transitions.
            molecular_weight_correction: Whether to apply molecular-weight calibration.
            seed: Optional seed shared by preprocessing and both model passes; mutually exclusive with
                ``generator``.
            generator: Optional generator advanced by preprocessing and both model
                passes; mutually exclusive with ``seed``.

        Returns:
            First-pass predictions sorted by descending confidence, the selected
            structure, and affinity scores. Affinity values use
            ``log10(IC50 / micromolar)`` units; binder probabilities lie in
            ``[0, 1]``.
        """

        # Reference augmentation and both model passes advance one RNG stream.
        generator = resolve_generator(seed, generator, device=self.structure_processor.device)

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
    r"""
    One two-stage Boltz-2 affinity result.

    Contains the selected second-pass ``target``, confidence-sorted first-pass
    ``structure_predictions``, and affinity scores and metrics.

    Args:
        target: Structure selected for affinity scoring.
        structure_predictions: First-stage predictions in descending confidence order.
        scores: Scalar affinity scores keyed by stable scientific names.
        metrics: Affinity metrics keyed by stable scientific names.
    """

    target: Structure
    structure_predictions: tuple[FoldPrediction, ...]
    scores: dict[str, Score] = field(default_factory=dict)
    metrics: dict[str, Metric] = field(default_factory=dict)
