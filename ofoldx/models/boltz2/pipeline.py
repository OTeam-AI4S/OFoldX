r"""
Pipelines for Boltz-2 folding and two-stage affinity scoring.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path

import torch

from ...data.evidence import MSA, MSACollection, Template
from ...data.molecular import MolecularSystem, PolymerSequence
from ...data.structure import Structure
from ...io import read_msa, read_structure
from ...pipelines import CompositePair, CompositePipeline, PairedPipeline, resolve_generator
from ...results import FoldPrediction, FoldResult, Metric, Score
from .model import (
    Boltz2AffinityModel,
    Boltz2Contact,
    Boltz2Model,
    Boltz2Pocket,
    Boltz2Template,
)
from .processor import Boltz2AffinityProcessor, Boltz2Processor


class Boltz2Pipeline(PairedPipeline[Boltz2Model, Boltz2Processor]):
    r"""
    Structure prediction pipeline for Boltz 2.
    """

    model_architecture = "boltz2"

    def fold(
        self,
        input: MolecularSystem | Structure | str | PathLike[str],
        *,
        msa: MSA | MSACollection | str | PathLike[str] | None = None,
        templates: Sequence[Template | Boltz2Template] = (),
        contacts: Sequence[Boltz2Contact] = (),
        pockets: Sequence[Boltz2Pocket] = (),
        method: str = "x-ray diffraction",
        num_samples: int = 1,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        recycling_steps: int | None = None,
        msa_max_depth: int | None = None,
        max_msa_rows: int | None = 8192,
        num_sampling_steps: int | None = None,
        step_scale: float | None = None,
        max_parallel_samples: int | None = None,
        use_potentials: bool = False,
        feynman_kac: bool = False,
        physical_guidance: bool = False,
        contact_guidance: bool = True,
        num_particles: int = 3,
        feynman_kac_lambda: float = 4.0,
        feynman_kac_resampling_interval: int = 3,
        num_guidance_steps: int = 20,
    ) -> FoldResult:
        r"""
        Predict structures for one molecular system with Boltz 2.

        Args:
            input: Molecular system, structure, or local structure path.
            msa: Optional MSA value, collection, or alignment path.
            templates: Structural protein templates with canonical residue alignments.
            contacts: Learned pairwise contact conditions.
            pockets: Learned binder-to-pocket conditions.
            method: Experimental method used by the learned method embedding.
            num_samples: Number of structure predictions to generate.
            seed: Optional seed for reference-conformer augmentation and
                coordinate sampling. Mutually exclusive with ``generator``.
            generator: Optional generator advanced by preprocessing and coordinate
                sampling. Mutually exclusive with ``seed``.
            recycling_steps: Number of representation recycling iterations. ``None`` uses the
                Model configuration.
            msa_max_depth: Optional rows independently sampled for each MSA-module
                invocation. ``None`` retains all rows.
            max_msa_rows: Optional cap applied only while reading an MSA path.
            num_sampling_steps: Number of diffusion steps. ``None`` uses the Model configuration.
            step_scale: Optional scale on each reverse-diffusion Euler update.
            max_parallel_samples: Maximum samples evaluated by one denoiser call.
            use_potentials: Enable the released Feynman--Kac and physical-guidance
                preset.
            feynman_kac: Whether to use particle resampling.
            physical_guidance: Whether to apply physical-potential gradients.
            contact_guidance: Whether forced contact, pocket, and template
                conditions apply their potential gradients.
            num_particles: Number of particles per requested sample under
                Feynman--Kac steering.
            feynman_kac_lambda: Feynman--Kac energy-log-weight scale.
            feynman_kac_resampling_interval: Reverse-diffusion step interval
                between particle resampling operations.
            num_guidance_steps: Gradient updates applied at each guided
                reverse-diffusion step.

        Returns:
            Generation-ordered structure predictions. Structure, PAE, and PDE values
            are in angstroms; pLDDT is on the ``0--100`` scale.
        """

        generator = resolve_generator(seed, generator, device=self.processor.device)
        input = _read_boltz2_input(input)
        msa = _read_boltz2_msa(
            msa,
            input.system if isinstance(input, Structure) else input,
            max_msa_rows=max_msa_rows,
        )
        features = self.processor(
            input,
            msa=msa,
            templates=templates,
            contacts=contacts,
            pockets=pockets,
            method=method,
            generator=generator,
        )
        return self.model.fold(
            features,
            num_samples=num_samples,
            generator=generator,
            recycling_steps=recycling_steps,
            msa_max_depth=msa_max_depth,
            num_sampling_steps=num_sampling_steps,
            step_scale=step_scale,
            max_parallel_samples=max_parallel_samples,
            use_potentials=use_potentials,
            feynman_kac=feynman_kac,
            physical_guidance=physical_guidance,
            contact_guidance=contact_guidance,
            num_particles=num_particles,
            feynman_kac_lambda=feynman_kac_lambda,
            feynman_kac_resampling_interval=feynman_kac_resampling_interval,
            num_guidance_steps=num_guidance_steps,
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
        CompositePair("", "boltz2_affinity", "model", "processor"),
        CompositePair("structure", "boltz2", "structure_model", "structure_processor"),
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
        self._validate_components()

    def score(
        self,
        input: MolecularSystem | Structure | str | PathLike[str],
        *,
        ligand: str | None = None,
        msa: MSA | MSACollection | str | PathLike[str] | None = None,
        templates: Sequence[Template | Boltz2Template] = (),
        contacts: Sequence[Boltz2Contact] = (),
        pockets: Sequence[Boltz2Pocket] = (),
        method: str = "x-ray diffraction",
        num_samples: int = 1,
        num_sampling_steps: int | None = None,
        step_scale: float | None = None,
        max_parallel_samples: int | None = None,
        use_potentials: bool = False,
        feynman_kac: bool = False,
        physical_guidance: bool = False,
        contact_guidance: bool = True,
        num_particles: int = 3,
        feynman_kac_lambda: float = 4.0,
        feynman_kac_resampling_interval: int = 3,
        num_guidance_steps: int = 20,
        affinity_num_samples: int = 5,
        affinity_num_sampling_steps: int | None = None,
        affinity_step_scale: float | None = None,
        msa_max_depth: int | None = None,
        max_msa_rows: int | None = 8192,
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
            templates: Structural protein templates used by the first structure pass.
            contacts: Learned pairwise contact conditions used by both passes.
            pockets: Learned binder-to-pocket conditions used by both passes.
            method: Experimental method used by the first structure pass.
            num_samples: Number of first-pass structure samples.
            num_sampling_steps: Number of first-pass diffusion transitions.
            step_scale: Optional first-pass reverse-diffusion Euler-step scale.
            max_parallel_samples: Maximum first-pass samples evaluated by one
                denoiser call.
            use_potentials: Enable the released first-pass Feynman--Kac and
                physical-guidance preset.
            feynman_kac: Whether the first pass uses particle resampling.
            physical_guidance: Whether the first pass applies physical-potential
                gradients.
            contact_guidance: Whether forced first-pass contact, pocket, and
                template conditions apply their potential gradients.
            num_particles: Number of first-pass particles per requested sample.
            feynman_kac_lambda: First-pass Feynman--Kac energy-log-weight scale.
            feynman_kac_resampling_interval: First-pass reverse-diffusion step
                interval between particle resampling operations.
            num_guidance_steps: Gradient updates applied at each guided
                first-pass reverse-diffusion step.
            affinity_num_samples: Number of second-pass candidate samples.
            affinity_num_sampling_steps: Number of second-pass diffusion transitions.
            affinity_step_scale: Optional second-pass reverse-diffusion Euler-step scale.
            msa_max_depth: Optional rows independently sampled for every MSA-module
                invocation in both passes. ``None`` retains all rows.
            max_msa_rows: Optional cap applied only while reading an MSA path.
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

        input = _read_boltz2_input(input)
        msa = _read_boltz2_msa(
            msa,
            input.system if isinstance(input, Structure) else input,
            max_msa_rows=max_msa_rows,
        )

        # The structure and affinity passes use independent sampling counts.
        fold_batch = self.structure_processor(
            input,
            msa=msa,
            templates=templates,
            contacts=contacts,
            pockets=pockets,
            method=method,
            generator=generator,
        )

        fold_result = self.structure_model.fold(
            fold_batch,
            num_samples=num_samples,
            num_sampling_steps=num_sampling_steps,
            step_scale=step_scale,
            msa_max_depth=msa_max_depth,
            max_parallel_samples=max_parallel_samples,
            use_potentials=use_potentials,
            feynman_kac=feynman_kac,
            physical_guidance=physical_guidance,
            contact_guidance=contact_guidance,
            num_particles=num_particles,
            feynman_kac_lambda=feynman_kac_lambda,
            feynman_kac_resampling_interval=feynman_kac_resampling_interval,
            num_guidance_steps=num_guidance_steps,
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

        affinity_batch = self.processor(
            best_structure,
            ligand=ligand,
            msa=msa,
            contacts=contacts,
            pockets=pockets,
            method="other",
            generator=generator,
        )
        score_result = self.model.score(
            affinity_batch,
            num_samples=affinity_num_samples,
            num_sampling_steps=affinity_num_sampling_steps,
            step_scale=affinity_step_scale,
            msa_max_depth=msa_max_depth,
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


def _read_boltz2_input(
    input: MolecularSystem | Structure | str | PathLike[str],
) -> MolecularSystem | Structure:
    if isinstance(input, str | PathLike):
        return read_structure(Path(input).expanduser())
    return input


def _read_boltz2_msa(
    msa: MSA | MSACollection | str | PathLike[str] | None,
    system: MolecularSystem,
    *,
    max_msa_rows: int | None,
) -> MSA | MSACollection | None:
    if not isinstance(msa, str | PathLike):
        return msa
    proteins = [
        entity_id
        for entity_id, entity in system.entities.items()
        if isinstance(entity, PolymerSequence) and entity.polymer_type == "protein"
    ]
    if len(proteins) != 1:
        raise ValueError("an aligned FASTA is unambiguous only for one protein entity; " "pass MSACollection")
    return read_msa(msa, entity_id=proteins[0], max_rows=max_msa_rows)
