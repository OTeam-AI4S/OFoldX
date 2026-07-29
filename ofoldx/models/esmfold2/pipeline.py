r"""
ESMFold2 structure-prediction and binder-design pipelines.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field, replace
from functools import cache
from os import PathLike
from pathlib import Path
from typing import cast

import numpy as np
import torch
import torch.nn.functional as F

from ...data.evidence import MSA, MSACollection
from ...data.molecular import AMINO_ACID_THREE_TO_ONE, MolecularSystem, PolymerSequence
from ...data.structure import Structure
from ...io import read_fasta_records, read_msa
from ...pipelines import CompositePair, CompositePipeline, PairedPipeline, resolve_generator
from ...references import classify_reference
from ...results import FoldPrediction, FoldResult, Metric, PairMetric, Score
from ...results.values import best
from .model import (
    EsmFold2ExperimentalModel,
    EsmFold2ExperimentalPrediction,
    EsmFold2Input,
    EsmFold2Model,
    compute_esmc_masked_lm_logits,
)
from .processor import (
    EsmFold2ExperimentalProcessor,
    EsmFold2Processor,
    antibody_cdr_indices,
    molecular_system_from_named_sequences,
    molecular_system_from_sequences,
)

# 20-dim design amino-acid space, ordered ALA..VAL (residue-type ids 2..21 minus 2), the
# offset used by the experimental trunk's residue-type features. ``F.pad(design, (2, 11))``
# lifts it to the 33-class ``residue_type`` space (2 leading pad/gap slots, 11 trailing
# nucleic slots).
_AA3_ORDER = (
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
)
_AMINO_ACID_TO_INDEX = {AMINO_ACID_THREE_TO_ONE[three]: index for index, three in enumerate(_AA3_ORDER)}
_INDEX_TO_AMINO_ACID = {index: AMINO_ACID_THREE_TO_ONE[three] for index, three in enumerate(_AA3_ORDER)}
_AA_DIMS = 20
_CYSTEINE_INDEX = _AMINO_ACID_TO_INDEX["C"]
_MUTABLE_TOKEN = "#"

# Binder-design optimization constants.
_LOSS_WEIGHTS = {"intra_contact": 0.5, "inter_contact": 0.5, "glob": 0.2}
_LEARNING_RATE = 0.1
_TEMPERATURE_MIN = 1e-2

# ESMC pseudo-perplexity regularizer mask-sampling constants.
_ESMC_MASK_FRACTION = 0.15
_LM_MASK_PASSES = 4
_LM_LOSS_CHUNK = 128

_FAST_CHECKPOINT = "ESMFold2-Experimental-Fast"
_FAST_CUTOFF2025_CHECKPOINT = "ESMFold2-Experimental-Fast-Cutoff2025"
_EXPERIMENTAL_CHECKPOINT = "ESMFold2-Experimental"
_EXPERIMENTAL_CUTOFF2025_CHECKPOINT = "ESMFold2-Experimental-Cutoff2025"


class EsmFold2Pipeline(PairedPipeline[EsmFold2Model, EsmFold2Processor]):
    r"""
    Structure prediction pipeline for ESMFold 2.
    """

    model_architecture = "esmfold2"

    def fold(
        self,
        input: str | PathLike[str] | PolymerSequence | Sequence[PolymerSequence] | MolecularSystem | Structure,
        *,
        msa: MSA | MSACollection | str | PathLike[str] | None = None,
        num_samples: int = 1,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        recycling_steps: int | None = None,
        num_sampling_steps: int | None = None,
        language_model_mask_probability: float | None = None,
        language_model_dropout: float | None = 0.3,
        msa_max_depth: int | None = 1024,
        msa_column_mask_rate: float = 0.1,
        msa_subsample_at_inference: bool = True,
        max_inference_sigma: float | None = 256.0,
        noise_scale: float | None = None,
        step_scale: float | None = None,
        denoising_early_exit_rmsd: float | None = None,
    ) -> FoldResult:
        r"""
        Predict structures from a sequence or molecular input.

        Args:
            input: Sequence text, FASTA path, one or more protein sequences, molecular
                system, or structure topology. Coordinates in a ``Structure`` are ignored.
            msa: Optional MSA evidence or alignment path. Configurations without MSA
                conditioning reject this argument.
            num_samples: Number of coordinate samples and returned predictions.
            seed: Integer seed, mutually exclusive with ``generator``.
            generator: PyTorch random generator, mutually exclusive with ``seed``.
            recycling_steps: Number of recurrent trunk passes. ``None`` uses the model
                configuration.
            num_sampling_steps: Number of diffusion sampling steps. ``None`` uses the
                model configuration.
            language_model_mask_probability: Probability of masking each nonspecial
                language-model token. ``None`` uses the model configuration.
            language_model_dropout: Dropout probability for language-model pair features.
                ``None`` uses the model configuration.
            msa_max_depth: Maximum number of MSA rows sampled per recurrent pass. ``None``
                keeps all rows.
            msa_column_mask_rate: Probability of masking a column in nonquery MSA rows.
            msa_subsample_at_inference: Whether to resample MSA rows on each recurrent pass.
            max_inference_sigma: Largest diffusion sigma included in sampling. ``None``
                keeps the full schedule.
            noise_scale: Sampler noise multiplier. ``None`` uses the model configuration.
            step_scale: Sampler step-size multiplier. ``None`` uses the model configuration.
            denoising_early_exit_rmsd: Aligned RMSD threshold, in angstroms, for ending
                denoising early. ``None`` disables early exit.

        Returns:
            A [`FoldResult`][ofoldx.results.FoldResult] containing predictions in sampling
            order.
        """

        if msa is not None and not self.processor.config.msa_conditioning:
            raise ValueError("this ESMFold2 checkpoint is the single-sequence Fast variant")
        system = _fold_system(input)
        msa = _fold_msa(msa, system)
        generator = resolve_generator(seed, generator, device=self.processor.device)
        batch = self.processor(system, msa=msa)
        return self.model.fold(
            batch,
            num_samples=num_samples,
            generator=generator,
            recycling_steps=recycling_steps,
            num_sampling_steps=num_sampling_steps,
            language_model_mask_probability=language_model_mask_probability,
            language_model_dropout=language_model_dropout,
            msa_max_depth=msa_max_depth,
            msa_column_mask_rate=msa_column_mask_rate,
            msa_subsample_at_inference=msa_subsample_at_inference,
            max_inference_sigma=max_inference_sigma,
            noise_scale=noise_scale,
            step_scale=step_scale,
            denoising_early_exit_rmsd=denoising_early_exit_rmsd,
        )


class EsmFold2ExperimentalPipeline(PairedPipeline[EsmFold2ExperimentalModel, EsmFold2ExperimentalProcessor]):
    r"""
    Independently callable experimental ESMFold2 Model--Processor pair.
    """

    model_architecture = "esmfold2_experimental"

    @torch.inference_mode()
    def fold(
        self,
        input: str | PathLike[str] | PolymerSequence | Sequence[PolymerSequence] | MolecularSystem | Structure,
        *,
        msa: MSA | MSACollection | str | PathLike[str] | None = None,
        num_samples: int = 1,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        recycling_steps: int | None = None,
        num_sampling_steps: int | None = None,
        language_model_mask_probability: float = 0.0,
        max_inference_sigma: float | None = 256.0,
        noise_scale: float | None = None,
        step_scale: float | None = None,
    ) -> FoldResult:
        r"""
        Fold one system with an independently saved experimental pair.

        Args:
            input: Sequence text, FASTA path, one or more protein sequences,
                molecular system, or structure topology.
            msa: Optional MSA evidence or alignment path. Fast checkpoints reject
                this argument.
            num_samples: Number of coordinate samples and returned predictions.
            seed: Integer seed, mutually exclusive with ``generator``.
            generator: PyTorch random generator, mutually exclusive with ``seed``.
            recycling_steps: Number of recurrent trunk passes. ``None`` uses the
                configured value.
            num_sampling_steps: Number of diffusion sampling steps. ``None`` uses
                the configured value.
            language_model_mask_probability: Probability of masking each nonspecial
                language-model token.
            max_inference_sigma: Largest diffusion sigma included in sampling.
                ``None`` keeps the full schedule.
            noise_scale: Sampler noise multiplier. ``None`` uses the configured value.
            step_scale: Sampler step-size multiplier. ``None`` uses the configured value.

        Returns:
            A [`FoldResult`][ofoldx.results.FoldResult] containing structures and
            the confidence values published by the experimental head.
        """

        if msa is not None and not self.processor.config.msa_conditioning:
            raise ValueError("this ESMFold2 checkpoint is the single-sequence Fast variant")
        system = _fold_system(input)
        msa = _fold_msa(msa, system)
        generator = resolve_generator(seed, generator, device=self.processor.device)
        processed = self.processor(system, msa=msa)
        prediction = self.model.sample_with_confidence(
            processed,
            recycling_steps=recycling_steps,
            num_samples=num_samples,
            num_sampling_steps=num_sampling_steps,
            language_model_mask_probability=language_model_mask_probability,
            max_inference_sigma=max_inference_sigma,
            noise_scale=noise_scale,
            step_scale=step_scale,
            generator=generator,
        )

        return FoldResult(
            predictions=tuple(
                _experimental_fold_prediction(
                    processed,
                    prediction,
                    batch_index=0,
                    sample_index=sample_index,
                )
                for sample_index in range(num_samples)
            )
        )


@dataclass(frozen=True)
class EsmFold2DesignStep:
    r"""
    One candidate's scientific values from one inversion optimization step.

    Args:
        temperature: Softmax temperature for this step.
        inversion_checkpoint: Released inversion checkpoint selected for this step.
        intra_contact_loss: Binder intra-contact term.
        inter_contact_loss: Target--binder contact term.
        globularity_loss: Binder globularity term.
        pseudoperplexity_loss: ESMC pseudo-perplexity term.
        total_loss: Reported structure-plus-pseudo-perplexity loss.
    """

    temperature: float
    inversion_checkpoint: str
    intra_contact_loss: float
    inter_contact_loss: float
    globularity_loss: float
    pseudoperplexity_loss: float
    total_loss: float


@dataclass(eq=False)
class EsmFold2CriticEvaluation:
    r"""
    One coherent released-critic evaluation of a designed sequence.

    Args:
        checkpoint: Exact released critic checkpoint role.
        prediction: Structure and all aligned decoded confidence values from one fold.
        distogram_iptm_proxy: Whole-binder distogram interface proxy.
        cdr_distogram_iptm_proxy: Antibody CDR-only proxy, or ``None`` for
            non-antibody design.
    """

    checkpoint: str
    prediction: FoldPrediction
    distogram_iptm_proxy: float
    cdr_distogram_iptm_proxy: float | None


@dataclass(eq=False)
class EsmFold2DesignCandidate:
    r"""
    One folded target-binder complex and its aligned scientific values.

    Args:
        sequence: Selected binder amino-acid sequence.
        structure: Experimental-Cutoff2025 target-binder complex used for selection.
        final_logits: Final optimized binder logits with shape ``(L, 20)``.
        best_iptm: Highest actual low-temperature inversion iPTM.
        trajectory: Complete scientific optimization trajectory.
        inversion_checkpoint: Released inversion checkpoint that produced
            ``inversion_prediction`` and ``best_iptm``.
        inversion_prediction: Coherent structure and confidence values from the
            inversion step attaining ``best_iptm``.
        inversion_distogram_iptm_proxy: Distogram interface proxy aligned to
            ``inversion_prediction``.
        critic_evaluations: Fixed hero and executed scaling evaluations.
        is_antibody: Whether the released antibody-specific filter and proxy
            semantics apply.
        isoelectric_point: Bjellqvist isoelectric point used by the released filter.
        passes_isoelectric_point_filter: Whether the candidate passes selection's
            non-antibody pI gate.
        hero_mean_iptm: Mean actual iPTM from the four fixed hero critics.
        scaling_mean_iptm_proxy: Mean fixed scaling-ensemble proxy, or zero when
            scaling evaluation was skipped.
        selection_score: Equal-weight hero/scaling selection score.
        scores: Scalar scores keyed by stable scientific names.
        metrics: One-axis metrics keyed by stable scientific names.
        pair_metrics: Two-axis metrics keyed by stable scientific names.
    """

    sequence: str
    structure: Structure
    final_logits: torch.Tensor
    best_iptm: float
    trajectory: tuple[EsmFold2DesignStep, ...]
    inversion_checkpoint: str
    inversion_prediction: FoldPrediction
    inversion_distogram_iptm_proxy: float
    critic_evaluations: tuple[EsmFold2CriticEvaluation, ...] = ()
    is_antibody: bool = False
    isoelectric_point: float | None = None
    passes_isoelectric_point_filter: bool = False
    hero_mean_iptm: float | None = None
    scaling_mean_iptm_proxy: float | None = None
    selection_score: float | None = None
    scores: dict[str, Score] = field(default_factory=dict)
    metrics: dict[str, Metric] = field(default_factory=dict)
    pair_metrics: dict[str, PairMetric] = field(default_factory=dict)


@dataclass(eq=False)
class EsmFold2DesignResult:
    r"""
    A generation-ordered collection of folded binder designs.

    Args:
        candidates: Folded target-binder complexes in generation order.
        selected_indices: Stable descending-score indices of the at most 84
            candidates passing the released filter.
    """

    candidates: tuple[EsmFold2DesignCandidate, ...]
    selected_indices: tuple[int, ...]

    @property
    def selected(self) -> tuple[EsmFold2DesignCandidate, ...]:
        r"""
        Return selected candidates in stable descending score order.
        """

        return tuple(self.candidates[index] for index in self.selected_indices)

    def best(self, score: str) -> EsmFold2DesignCandidate:
        r"""
        Return the earliest best candidate under a named score.

        Args:
            score: Comparable score name present on every candidate.

        Returns:
            The best candidate, with generation order breaking ties.
        """

        return best(self.candidates, score)


class EsmFold2DesignPipeline(CompositePipeline):
    r"""
    Complete released ESMFold2 binder-design composition.
    """

    pair_layout = (
        CompositePair("", "esmfold2_experimental", "fast", None, wrap=EsmFold2ExperimentalPipeline),
        CompositePair(
            "fast_cutoff2025",
            "esmfold2_experimental",
            "fast_cutoff2025",
            None,
            wrap=EsmFold2ExperimentalPipeline,
        ),
        CompositePair(
            "experimental",
            "esmfold2_experimental",
            "experimental",
            None,
            wrap=EsmFold2ExperimentalPipeline,
        ),
        CompositePair(
            "experimental_cutoff2025",
            "esmfold2_experimental",
            "experimental_cutoff2025",
            None,
            wrap=EsmFold2ExperimentalPipeline,
        ),
        CompositePair(
            "scaling/base300m_step250k",
            "esmfold2_experimental",
            "base300m_step250k",
            None,
            wrap=EsmFold2ExperimentalPipeline,
        ),
        CompositePair(
            "scaling/base300m_step500k",
            "esmfold2_experimental",
            "base300m_step500k",
            None,
            wrap=EsmFold2ExperimentalPipeline,
        ),
        CompositePair(
            "scaling/base300m_step750k",
            "esmfold2_experimental",
            "base300m_step750k",
            None,
            wrap=EsmFold2ExperimentalPipeline,
        ),
        CompositePair(
            "scaling/base300m_step1000k",
            "esmfold2_experimental",
            "base300m_step1000k",
            None,
            wrap=EsmFold2ExperimentalPipeline,
        ),
        CompositePair(
            "scaling/base300m_step1500k",
            "esmfold2_experimental",
            "base300m_step1500k",
            None,
            wrap=EsmFold2ExperimentalPipeline,
        ),
        CompositePair(
            "scaling/base600m_step250k",
            "esmfold2_experimental",
            "base600m_step250k",
            None,
            wrap=EsmFold2ExperimentalPipeline,
        ),
        CompositePair(
            "scaling/base600m_step500k",
            "esmfold2_experimental",
            "base600m_step500k",
            None,
            wrap=EsmFold2ExperimentalPipeline,
        ),
        CompositePair(
            "scaling/base600m_step750k",
            "esmfold2_experimental",
            "base600m_step750k",
            None,
            wrap=EsmFold2ExperimentalPipeline,
        ),
        CompositePair(
            "scaling/base600m_step1000k",
            "esmfold2_experimental",
            "base600m_step1000k",
            None,
            wrap=EsmFold2ExperimentalPipeline,
        ),
        CompositePair(
            "scaling/base600m_step1500k",
            "esmfold2_experimental",
            "base600m_step1500k",
            None,
            wrap=EsmFold2ExperimentalPipeline,
        ),
        CompositePair(
            "scaling/base6b_step250k",
            "esmfold2_experimental",
            "base6b_step250k",
            None,
            wrap=EsmFold2ExperimentalPipeline,
        ),
        CompositePair(
            "scaling/base6b_step500k",
            "esmfold2_experimental",
            "base6b_step500k",
            None,
            wrap=EsmFold2ExperimentalPipeline,
        ),
        CompositePair(
            "scaling/base6b_step750k",
            "esmfold2_experimental",
            "base6b_step750k",
            None,
            wrap=EsmFold2ExperimentalPipeline,
        ),
        CompositePair(
            "scaling/base6b_step1000k",
            "esmfold2_experimental",
            "base6b_step1000k",
            None,
            wrap=EsmFold2ExperimentalPipeline,
        ),
        CompositePair(
            "scaling/base6b_step1500k",
            "esmfold2_experimental",
            "base6b_step1500k",
            None,
            wrap=EsmFold2ExperimentalPipeline,
        ),
    )

    def __init__(
        self,
        *,
        fast: EsmFold2ExperimentalPipeline,
        fast_cutoff2025: EsmFold2ExperimentalPipeline,
        experimental: EsmFold2ExperimentalPipeline,
        experimental_cutoff2025: EsmFold2ExperimentalPipeline,
        base300m_step250k: EsmFold2ExperimentalPipeline,
        base300m_step500k: EsmFold2ExperimentalPipeline,
        base300m_step750k: EsmFold2ExperimentalPipeline,
        base300m_step1000k: EsmFold2ExperimentalPipeline,
        base300m_step1500k: EsmFold2ExperimentalPipeline,
        base600m_step250k: EsmFold2ExperimentalPipeline,
        base600m_step500k: EsmFold2ExperimentalPipeline,
        base600m_step750k: EsmFold2ExperimentalPipeline,
        base600m_step1000k: EsmFold2ExperimentalPipeline,
        base600m_step1500k: EsmFold2ExperimentalPipeline,
        base6b_step250k: EsmFold2ExperimentalPipeline,
        base6b_step500k: EsmFold2ExperimentalPipeline,
        base6b_step750k: EsmFold2ExperimentalPipeline,
        base6b_step1000k: EsmFold2ExperimentalPipeline,
        base6b_step1500k: EsmFold2ExperimentalPipeline,
    ) -> None:
        self.fast = fast
        self.fast_cutoff2025 = fast_cutoff2025
        self.experimental = experimental
        self.experimental_cutoff2025 = experimental_cutoff2025
        self.base300m_step250k = base300m_step250k
        self.base300m_step500k = base300m_step500k
        self.base300m_step750k = base300m_step750k
        self.base300m_step1000k = base300m_step1000k
        self.base300m_step1500k = base300m_step1500k
        self.base600m_step250k = base600m_step250k
        self.base600m_step500k = base600m_step500k
        self.base600m_step750k = base600m_step750k
        self.base600m_step1000k = base600m_step1000k
        self.base600m_step1500k = base600m_step1500k
        self.base6b_step250k = base6b_step250k
        self.base6b_step500k = base6b_step500k
        self.base6b_step750k = base6b_step750k
        self.base6b_step1000k = base6b_step1000k
        self.base6b_step1500k = base6b_step1500k
        self._validate_components()

    def design(
        self,
        input: str | PathLike[str] | Structure | MolecularSystem,
        *,
        binder_length: int,
        binder_prompt: str | None = None,
        num_candidates: int = 1,
        num_optimization_steps: int = 150,
        is_antibody: bool = False,
        use_scaling_critics: bool = True,
        language_model_weight: float | None = None,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> EsmFold2DesignResult:
        r"""
        Run the released inversion, critic-evaluation, filtering, and selection protocol.

        Args:
            input: One-letter sequence, FASTA path, ``Structure``, or
                ``MolecularSystem`` resolving to one protein chain.
            binder_length: Number of residues in each designed binder.
            binder_prompt: Optional prompt of length ``binder_length``. One-letter
                amino acids fix positions and ``'#'`` marks mutable positions;
                ``None`` makes every position mutable.
            num_candidates: Number of jointly optimized binder candidates.
            num_optimization_steps: Number of soft-sequence optimization steps.
            is_antibody: Whether the prompt is an antibody design.
            use_scaling_critics: Whether to execute the fixed fifteen-checkpoint
                scaling ensemble. All fifteen pairs remain required by the repository.
            language_model_weight: ESMC gradient coefficient. ``None`` selects
                ``0.05`` for antibodies and ``0.15`` otherwise.
            seed: Optional integer seed, mutually exclusive with ``generator``.
            generator: Optional caller-owned generator advanced by checkpoint
                selection, model stochasticity, diffusion, and ESMC masking.

        Returns:
            Every generated design, its critic evaluations and scores, and the
            stable indices of at most 84 selected designs.
        """

        if binder_length <= 0:
            raise ValueError("binder_length must be positive")
        if num_candidates <= 0:
            raise ValueError("num_candidates must be positive")
        if num_optimization_steps <= 0:
            raise ValueError("num_optimization_steps must be positive")
        prompt = _MUTABLE_TOKEN * binder_length if binder_prompt is None else binder_prompt
        if len(prompt) != binder_length:
            raise ValueError("binder_prompt length must equal binder_length")

        coefficient = (0.05 if is_antibody else 0.15) if language_model_weight is None else language_model_weight
        inversion_pairs = (
            (_FAST_CHECKPOINT, self.fast),
            (_FAST_CUTOFF2025_CHECKPOINT, self.fast_cutoff2025),
        )
        hero_critics = (
            (_FAST_CHECKPOINT, self.fast),
            (_FAST_CUTOFF2025_CHECKPOINT, self.fast_cutoff2025),
            (_EXPERIMENTAL_CHECKPOINT, self.experimental),
            (_EXPERIMENTAL_CUTOFF2025_CHECKPOINT, self.experimental_cutoff2025),
        )
        scaling_critics = (
            ("ESMFold2-Experimental-Fast-base300M-step250k", self.base300m_step250k),
            ("ESMFold2-Experimental-Fast-base300M-step500k", self.base300m_step500k),
            ("ESMFold2-Experimental-Fast-base300M-step750k", self.base300m_step750k),
            ("ESMFold2-Experimental-Fast-base300M-step1000k", self.base300m_step1000k),
            ("ESMFold2-Experimental-Fast-base300M-step1500k", self.base300m_step1500k),
            ("ESMFold2-Experimental-Fast-base600M-step250k", self.base600m_step250k),
            ("ESMFold2-Experimental-Fast-base600M-step500k", self.base600m_step500k),
            ("ESMFold2-Experimental-Fast-base600M-step750k", self.base600m_step750k),
            ("ESMFold2-Experimental-Fast-base600M-step1000k", self.base600m_step1000k),
            ("ESMFold2-Experimental-Fast-base600M-step1500k", self.base600m_step1500k),
            ("ESMFold2-Experimental-Fast-base6B-step250k", self.base6b_step250k),
            ("ESMFold2-Experimental-Fast-base6B-step500k", self.base6b_step500k),
            ("ESMFold2-Experimental-Fast-base6B-step750k", self.base6b_step750k),
            ("ESMFold2-Experimental-Fast-base6B-step1000k", self.base6b_step1000k),
            ("ESMFold2-Experimental-Fast-base6B-step1500k", self.base6b_step1500k),
        )
        target_sequence = _target_one_letter(input)
        device = self.fast.model.left_pair_projection.weight.device
        generator = resolve_generator(seed, generator, device=device)
        active_scaling_critics = scaling_critics if use_scaling_critics else ()
        active_critics = hero_critics + active_scaling_critics
        models = tuple(pipeline.model for _, pipeline in active_critics)
        dropout_state = tuple((model.language_model_dropout, model.force_language_model_dropout) for model in models)
        try:
            for _, pipeline in inversion_pairs:
                model = pipeline.model
                model.configure_language_model_dropout(0.5, force_during_inference=True)
            with _frozen_eval(self.fast.model), _frozen_eval(self.fast_cutoff2025.model):
                candidates = self._run_design(
                    pairs=inversion_pairs,
                    target_sequence=target_sequence,
                    prompt=prompt,
                    binder_length=binder_length,
                    num_candidates=num_candidates,
                    num_optimization_steps=num_optimization_steps,
                    language_model_weight=coefficient,
                    generator=generator,
                )
            for model in models:
                model.configure_language_model_dropout(0.25, force_during_inference=True)
            with ExitStack() as stack:
                for _, pipeline in active_critics:
                    stack.enter_context(_frozen_eval(pipeline.model))
                return self._score_candidates(
                    candidates,
                    target_sequence=target_sequence,
                    is_antibody=is_antibody,
                    hero_critics=hero_critics,
                    scaling_critics=active_scaling_critics,
                    generator=generator,
                )
        finally:
            for model, (dropout, force) in zip(models, dropout_state):
                model.configure_language_model_dropout(dropout, force_during_inference=force)

    def _score_candidates(
        self,
        candidates: tuple[EsmFold2DesignCandidate, ...],
        *,
        target_sequence: str,
        is_antibody: bool,
        hero_critics: tuple[tuple[str, EsmFold2ExperimentalPipeline], ...],
        scaling_critics: tuple[tuple[str, EsmFold2ExperimentalPipeline], ...],
        generator: torch.Generator | None,
    ) -> EsmFold2DesignResult:
        target_length = len(target_sequence)
        evaluations: list[list[EsmFold2CriticEvaluation]] = [[] for _ in candidates]
        for checkpoint, pipeline in hero_critics + scaling_critics:
            individual_inputs = tuple(
                pipeline.processor(
                    [
                        PolymerSequence.from_one_letter(target_sequence),
                        PolymerSequence.from_one_letter(candidate.sequence),
                    ]
                )
                for candidate in candidates
            )
            batch = _stack_design_inputs(individual_inputs)
            with torch.inference_mode():
                prediction = pipeline.model.sample_with_confidence(
                    batch,
                    recycling_steps=4,
                    num_samples=1,
                    num_sampling_steps=200,
                    language_model_mask_probability=0.0,
                    generator=generator,
                )
                for candidate_index, candidate in enumerate(candidates):
                    distogram = prediction.sample.distogram_logits[candidate_index].float()
                    evaluations[candidate_index].append(
                        EsmFold2CriticEvaluation(
                            checkpoint=checkpoint,
                            prediction=_experimental_fold_prediction(
                                individual_inputs[candidate_index],
                                prediction,
                                batch_index=candidate_index,
                                sample_index=0,
                            ),
                            distogram_iptm_proxy=_distogram_iptm_proxy(
                                distogram,
                                target_length,
                                len(candidate.sequence),
                            ),
                            cdr_distogram_iptm_proxy=(
                                cdr_distogram_iptm_proxy(
                                    distogram,
                                    target_length,
                                    candidate.sequence,
                                )
                                if is_antibody
                                else None
                            ),
                        )
                    )

        scored: list[EsmFold2DesignCandidate] = []
        num_hero_critics = len(hero_critics)
        for candidate, candidate_evaluations in zip(candidates, evaluations):
            hero_evaluations = candidate_evaluations[:num_hero_critics]
            scaling_evaluations = candidate_evaluations[num_hero_critics:]
            hero_mean_iptm = (
                sum(evaluation.prediction.scores["iptm"].value for evaluation in hero_evaluations) / num_hero_critics
            )
            if scaling_evaluations:
                if is_antibody:
                    scaling_mean_proxy = sum(
                        cast(float, evaluation.cdr_distogram_iptm_proxy) for evaluation in scaling_evaluations
                    ) / len(scaling_evaluations)
                else:
                    scaling_mean_proxy = sum(
                        evaluation.distogram_iptm_proxy for evaluation in scaling_evaluations
                    ) / len(scaling_evaluations)
            else:
                scaling_mean_proxy = 0.0
            selection_score = 0.5 * hero_mean_iptm + 0.5 * scaling_mean_proxy
            isoelectric_point = _isoelectric_point(candidate.sequence)
            passes_filter = is_antibody or isoelectric_point < 6.0
            selected_evaluation = next(
                evaluation
                for evaluation in hero_evaluations
                if evaluation.checkpoint == _EXPERIMENTAL_CUTOFF2025_CHECKPOINT
            )
            selected_prediction = selected_evaluation.prediction
            scores = {
                **selected_prediction.scores,
                "best_inversion_iptm": Score(candidate.best_iptm, preferred="higher"),
                "inversion_distogram_iptm_proxy": Score(
                    candidate.inversion_distogram_iptm_proxy,
                    preferred="higher",
                ),
                "isoelectric_point": Score(isoelectric_point, unit="pH", preferred="lower"),
                "hero_mean_iptm": Score(hero_mean_iptm, preferred="higher"),
                "scaling_mean_iptm_proxy": Score(scaling_mean_proxy, preferred="higher"),
                "selection_score": Score(selection_score, preferred="higher"),
            }
            scored.append(
                replace(
                    candidate,
                    structure=selected_prediction.structure,
                    critic_evaluations=tuple(candidate_evaluations),
                    is_antibody=is_antibody,
                    isoelectric_point=isoelectric_point,
                    passes_isoelectric_point_filter=passes_filter,
                    hero_mean_iptm=hero_mean_iptm,
                    scaling_mean_iptm_proxy=scaling_mean_proxy,
                    selection_score=selection_score,
                    scores=scores,
                    metrics=selected_prediction.metrics,
                    pair_metrics=selected_prediction.pair_metrics,
                )
            )

        eligible = [index for index, candidate in enumerate(scored) if candidate.passes_isoelectric_point_filter]
        selected_indices = tuple(
            sorted(
                eligible,
                key=lambda index: cast(float, scored[index].selection_score),
                reverse=True,
            )[:84]
        )
        return EsmFold2DesignResult(
            candidates=tuple(scored),
            selected_indices=selected_indices,
        )

    def _run_design(
        self,
        *,
        pairs: tuple[
            tuple[str, EsmFold2ExperimentalPipeline],
            tuple[str, EsmFold2ExperimentalPipeline],
        ],
        target_sequence: str,
        prompt: str,
        binder_length: int,
        num_candidates: int,
        num_optimization_steps: int,
        language_model_weight: float,
        generator: torch.Generator | None,
    ) -> tuple[EsmFold2DesignCandidate, ...]:
        device = pairs[0][1].model.left_pair_projection.weight.device
        target_length = len(target_sequence)
        logits, gradient_mask = self._initial_logits(
            prompt,
            num_candidates,
            generator,
            device,
        )
        optimizer = torch.optim.SGD([logits], lr=_LEARNING_RATE)
        trajectories: list[list[EsmFold2DesignStep]] = [[] for _ in range(num_candidates)]
        best_iptm = [-math.inf] * num_candidates
        selected: list[EsmFold2DesignCandidate | None] = [None] * num_candidates

        for step in range(num_optimization_steps):
            choice = int(torch.randint(len(pairs), (), device=device, generator=generator).item())
            checkpoint, pipeline = pairs[choice]
            model = pipeline.model
            processor = pipeline.processor
            t = (step + 1) / num_optimization_steps
            temperature = _TEMPERATURE_MIN + (1 - _TEMPERATURE_MIN) * 0.5 * (1 + math.cos(math.pi * t))

            optimizer.zero_grad()
            design = F.softmax(logits / temperature, dim=-1)
            binder_sequences = tuple(
                "".join(_INDEX_TO_AMINO_ACID[int(index)] for index in row) for row in design.argmax(dim=-1)
            )
            individual_inputs = tuple(
                processor(
                    [
                        PolymerSequence.from_one_letter(target_sequence),
                        PolymerSequence.from_one_letter(binder_sequence),
                    ]
                )
                for binder_sequence in binder_sequences
            )
            batch = _stack_design_inputs(individual_inputs)
            residue_type_probabilities = torch.cat(
                [
                    F.one_hot(
                        batch.residue_type_ids[:, :target_length].long(),
                        num_classes=33,
                    ).float(),
                    F.pad(design, (2, 11)),
                ],
                dim=1,
            )

            prediction: EsmFold2ExperimentalPrediction | None = None
            if temperature < 0.05:
                prediction = model.sample_with_confidence(
                    batch,
                    residue_type_probabilities=residue_type_probabilities,
                    recycling_steps=2,
                    num_samples=1,
                    num_sampling_steps=50,
                    generator=generator,
                )
                distogram = prediction.sample.distogram_logits.float()
            else:
                sample = model.sample(
                    batch,
                    residue_type_probabilities=residue_type_probabilities,
                    recycling_steps=2,
                    num_samples=1,
                    num_sampling_steps=1,
                    generator=generator,
                )
                distogram = sample.distogram_logits.float()

            losses = _structure_losses(distogram, binder_length)
            structure_grad = torch.autograd.grad(losses["total"].mean(), logits)[0]
            design_lm = F.softmax(logits / temperature, dim=-1)
            plm_loss = _esmc_pseudo_perplexity_nll(
                model,
                design_lm,
                gradient_mask.sum(dim=-1) > 0,
                generator=generator,
            )
            plm_grad = torch.autograd.grad(plm_loss.mean(), logits)[0]
            logits.grad = _normalized_gradient(structure_grad, gradient_mask) + language_model_weight * (
                _normalized_gradient(plm_grad, gradient_mask)
            )
            for group in optimizer.param_groups:
                group["lr"] = _LEARNING_RATE * temperature
            optimizer.step()

            reported_total = losses["total"] + plm_loss
            for candidate_index in range(num_candidates):
                trajectories[candidate_index].append(
                    EsmFold2DesignStep(
                        temperature=temperature,
                        inversion_checkpoint=checkpoint,
                        intra_contact_loss=float(losses["intra_contact"][candidate_index].detach().cpu()),
                        inter_contact_loss=float(losses["inter_contact"][candidate_index].detach().cpu()),
                        globularity_loss=float(losses["glob"][candidate_index].detach().cpu()),
                        pseudoperplexity_loss=float(plm_loss[candidate_index].detach().cpu()),
                        total_loss=float(reported_total[candidate_index].detach().cpu()),
                    )
                )

            if prediction is not None:
                for candidate_index, binder_sequence in enumerate(binder_sequences):
                    iptm = float(prediction.confidence.iptm[candidate_index, 0].detach().cpu())
                    if iptm > best_iptm[candidate_index]:
                        best_iptm[candidate_index] = iptm
                        selected[candidate_index] = self._decode_candidate(
                            individual_inputs[candidate_index],
                            prediction,
                            candidate_index,
                            target_length,
                            checkpoint,
                            binder_sequence,
                            iptm,
                            logits[candidate_index].detach().cpu(),
                            tuple(trajectories[candidate_index]),
                        )

        candidates: list[EsmFold2DesignCandidate] = []
        for candidate_index, candidate in enumerate(selected):
            assert candidate is not None
            candidate.final_logits = logits[candidate_index].detach().cpu()
            candidate.trajectory = tuple(trajectories[candidate_index])
            candidates.append(candidate)
        return tuple(candidates)

    @staticmethod
    def _initial_logits(
        prompt: str,
        num_candidates: int,
        generator: torch.Generator | None,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        length = len(prompt)
        if all(symbol == _MUTABLE_TOKEN for symbol in prompt):
            logits = 0.01 * torch.randn(
                num_candidates,
                length,
                _AA_DIMS,
                generator=generator,
                device=device,
            )
            logits[:, :, _CYSTEINE_INDEX] = -1e6
        else:
            logits = torch.zeros(num_candidates, length, _AA_DIMS, device=device)
            for index, symbol in enumerate(prompt):
                if symbol == _MUTABLE_TOKEN:
                    logits[:, index, :] = 0.01 * torch.randn(
                        num_candidates,
                        _AA_DIMS,
                        generator=generator,
                        device=device,
                    )
                    logits[:, index, _CYSTEINE_INDEX] = -1e6
                else:
                    if symbol not in _AMINO_ACID_TO_INDEX:
                        raise ValueError(f"binder_prompt symbol {symbol!r} is not a standard amino acid or '#'")
                    logits[:, index, _AMINO_ACID_TO_INDEX[symbol]] = 10.0

        mask = torch.ones(num_candidates, length, _AA_DIMS, device=device)
        for index, symbol in enumerate(prompt):
            if symbol != _MUTABLE_TOKEN:
                mask[:, index, :] = 0.0
        mask[:, :, _CYSTEINE_INDEX] = 0.0
        return logits.requires_grad_(True), mask

    def _decode_candidate(
        self,
        input: EsmFold2Input,
        prediction: EsmFold2ExperimentalPrediction,
        batch_index: int,
        target_length: int,
        inversion_checkpoint: str,
        binder_sequence: str,
        iptm: float,
        final_logits: torch.Tensor,
        trajectory: tuple[EsmFold2DesignStep, ...],
    ) -> EsmFold2DesignCandidate:
        sample = prediction.sample
        inversion_prediction = _experimental_fold_prediction(
            input,
            prediction,
            batch_index=batch_index,
            sample_index=0,
        )
        proxy = _distogram_iptm_proxy(
            sample.distogram_logits[batch_index].float(),
            target_length,
            len(binder_sequence),
        )
        scores = {
            **inversion_prediction.scores,
            "distogram_iptm_proxy": Score(proxy, preferred="higher"),
        }
        return EsmFold2DesignCandidate(
            sequence=binder_sequence,
            structure=inversion_prediction.structure,
            final_logits=final_logits,
            best_iptm=iptm,
            trajectory=trajectory,
            inversion_checkpoint=inversion_checkpoint,
            inversion_prediction=inversion_prediction,
            inversion_distogram_iptm_proxy=proxy,
            scores=scores,
            metrics=inversion_prediction.metrics,
            pair_metrics=inversion_prediction.pair_metrics,
        )


def _experimental_fold_prediction(
    input: EsmFold2Input,
    prediction: EsmFold2ExperimentalPrediction,
    *,
    batch_index: int,
    sample_index: int,
) -> FoldPrediction:
    layout = input.layout
    atom_count = len(layout.atom_sites)
    raw_coordinates = (
        prediction.sample.coordinates[batch_index, sample_index, :atom_count]
        .detach()
        .to(device="cpu", dtype=torch.float32)
        .numpy()
    )
    complete_sites = tuple(site for site, atom in layout.system.iter_atoms() if atom.element != "H")
    complete_index = {site: index for index, site in enumerate(complete_sites)}
    model_to_complete = np.asarray([complete_index[site] for site in layout.atom_sites], dtype=np.intp)
    coordinates = np.zeros((len(complete_sites), 3), dtype=np.float32)
    coordinate_mask = np.zeros(len(complete_sites), dtype=np.bool_)
    coordinates[model_to_complete] = raw_coordinates
    coordinate_mask[model_to_complete] = True
    confidence = prediction.confidence
    return FoldPrediction(
        structure=Structure(
            system=layout.system,
            sites=complete_sites,
            coordinates=coordinates,
            coordinate_mask=coordinate_mask,
        ),
        scores={
            "ptm": Score(
                float(confidence.ptm[batch_index, sample_index].detach().cpu()),
                preferred="higher",
            ),
            "iptm": Score(
                float(confidence.iptm[batch_index, sample_index].detach().cpu()),
                preferred="higher",
            ),
            "mean_plddt": Score(
                float(confidence.mean_plddt[batch_index, sample_index].detach().cpu()),
                preferred="higher",
            ),
            "complex_iplddt": Score(
                float(confidence.complex_iplddt[batch_index, sample_index].detach().cpu()),
                preferred="higher",
            ),
        },
        metrics={
            "plddt": Metric(
                values=confidence.plddt[batch_index, sample_index].detach().float().cpu().numpy(),
                axis=layout.token_sites,
            ),
            "plddt_per_atom": Metric(
                values=confidence.plddt_per_atom[batch_index, sample_index, :atom_count].detach().float().cpu().numpy(),
                axis=layout.atom_sites,
            ),
            "plddt_ca": Metric(
                values=confidence.plddt_ca[batch_index, sample_index].detach().float().cpu().numpy(),
                axis=layout.token_sites,
            ),
        },
        pair_metrics={
            "pae": PairMetric(
                values=confidence.pae[batch_index, sample_index].detach().float().cpu().numpy(),
                row_axis=layout.token_sites,
                column_axis=layout.token_sites,
                unit="angstrom",
            ),
            "chain_pair_iptm": PairMetric(
                values=confidence.chain_pair_iptm[batch_index, sample_index].detach().float().cpu().numpy(),
                row_axis=tuple(layout.system.instances),
                column_axis=tuple(layout.system.instances),
            ),
        },
    )


def _stack_design_inputs(inputs: tuple[EsmFold2Input, ...]) -> EsmFold2Input:
    r"""
    Stack same-length target--binder inputs while retaining each candidate's atom padding.
    """

    max_atoms = max(value.atom_mask.shape[1] for value in inputs)

    def stack(name: str) -> torch.Tensor:
        return torch.cat([cast(torch.Tensor, getattr(value, name)) for value in inputs], dim=0)

    def stack_optional(name: str) -> torch.Tensor | None:
        values = [getattr(value, name) for value in inputs]
        if values[0] is None:
            return None
        return torch.cat([cast(torch.Tensor, value) for value in values], dim=0)

    def stack_atoms(name: str) -> torch.Tensor:
        values: list[torch.Tensor] = []
        for input in inputs:
            value = cast(torch.Tensor, getattr(input, name))
            padding = max_atoms - value.shape[1]
            if padding:
                value = torch.cat(
                    [
                        value,
                        value.new_zeros(value.shape[0], padding, *value.shape[2:]),
                    ],
                    dim=1,
                )
            values.append(value)
        return torch.cat(values, dim=0)

    language_model_tokens = [cast(torch.Tensor, value.language_model_tokens) for value in inputs]
    language_model_masks = [cast(torch.Tensor, value.language_model_mask) for value in inputs]
    token_to_esmc: list[torch.Tensor] = []
    offset = 0
    for value, tokens in zip(inputs, language_model_tokens):
        mapping = cast(torch.Tensor, value.token_to_esmc)
        has_esmc = value.token_has_esmc
        token_to_esmc.append(torch.where(has_esmc, mapping + offset, mapping))
        offset += tokens.numel()

    first = inputs[0]
    return replace(
        first,
        token_index=stack("token_index"),
        residue_index=stack("residue_index"),
        chain_index=stack("chain_index"),
        symmetry_index=stack("symmetry_index"),
        entity_index=stack("entity_index"),
        molecule_type=stack("molecule_type"),
        residue_type_ids=stack("residue_type_ids"),
        token_bonds=stack("token_bonds"),
        token_mask=stack("token_mask"),
        reference_positions=stack_atoms("reference_positions"),
        reference_elements=stack_atoms("reference_elements"),
        reference_charges=stack_atoms("reference_charges"),
        reference_atom_name_characters=stack_atoms("reference_atom_name_characters"),
        reference_space_id=stack_atoms("reference_space_id"),
        atom_mask=stack_atoms("atom_mask"),
        atom_to_token=stack_atoms("atom_to_token"),
        distogram_atom_index=stack("distogram_atom_index"),
        token_has_esmc=stack("token_has_esmc"),
        deletion_mean=stack_optional("deletion_mean"),
        msa_tokens=stack_optional("msa_tokens"),
        has_deletion=stack_optional("has_deletion"),
        deletion_value=stack_optional("deletion_value"),
        msa_mask=stack_optional("msa_mask"),
        language_model_tokens=torch.cat(language_model_tokens, dim=0),
        language_model_mask=torch.cat(language_model_masks, dim=0),
        token_to_esmc=torch.cat(token_to_esmc, dim=0),
        esmc_hidden_states=stack_optional("esmc_hidden_states"),
    )


@contextmanager
def _frozen_eval(model: torch.nn.Module) -> Iterator[None]:
    r"""
    Run one design call in frozen eval mode, then restore the exact module state.
    """

    training = tuple((module, module.training) for module in model.modules())
    requires_grad = tuple((parameter, parameter.requires_grad) for parameter in model.parameters())
    model.eval().requires_grad_(False)
    try:
        yield
    finally:
        for parameter, enabled in requires_grad:
            parameter.requires_grad_(enabled)
        for module, enabled in training:
            module.training = enabled


def _fold_system(
    input: str | PathLike[str] | PolymerSequence | Sequence[PolymerSequence] | MolecularSystem | Structure,
) -> MolecularSystem:
    r"""
    Resolve the fold Pipeline's inline-sequence and FASTA convenience.
    """

    if isinstance(input, Structure):
        return input.system
    if isinstance(input, MolecularSystem):
        return input
    if not isinstance(input, str | PathLike):
        return molecular_system_from_sequences(input)

    classified = classify_reference(input)
    if isinstance(classified, Path):
        named = [
            (
                record.id or f"chain_{index + 1}",
                PolymerSequence.from_one_letter(record.sequence.strip().upper()),
            )
            for index, record in enumerate(read_fasta_records(classified))
        ]
    else:
        named = [("protein", PolymerSequence.from_one_letter(classified.strip().upper()))]
    return molecular_system_from_named_sequences(named)


def _fold_msa(
    msa: MSA | MSACollection | str | PathLike[str] | None,
    system: MolecularSystem,
) -> MSA | MSACollection | None:
    r"""
    Read path evidence once and bind a single-entity alignment to its protein.
    """

    if msa is None or isinstance(msa, MSA | MSACollection):
        return msa
    proteins = [
        entity_id
        for entity_id, entity in system.entities.items()
        if isinstance(entity, PolymerSequence) and entity.polymer_type == "protein"
    ]
    if len(proteins) != 1:
        raise ValueError("an alignment path is unambiguous only for one protein entity; pass MSACollection")
    return read_msa(Path(msa).expanduser(), entity_id=proteins[0])


def _target_one_letter(input: str | PathLike[str] | Structure | MolecularSystem) -> str:
    r"""
    Resolve one single-chain protein target to an unambiguous one-letter sequence.
    """

    system: MolecularSystem | None = None
    if isinstance(input, Structure):
        system = input.system
    elif isinstance(input, MolecularSystem):
        system = input
    if system is not None:
        proteins: list[PolymerSequence] = []
        for entity_key in system.instances.values():
            entity = system.entities[entity_key]
            if isinstance(entity, PolymerSequence) and entity.polymer_type == "protein":
                proteins.append(entity)
        if len(proteins) != 1:
            raise ValueError("ESMFold2 design requires one single-chain protein target")
        return "".join(AMINO_ACID_THREE_TO_ONE[component.component_id] for component in proteins[0].components)

    classified = classify_reference(cast(str | PathLike[str], input))
    if isinstance(classified, Path):
        records = tuple(read_fasta_records(classified))
        if len(records) != 1:
            raise ValueError("ESMFold2 design requires exactly one target FASTA record")
        return records[0].sequence.strip().upper()
    return str(classified).strip().upper()


def _isoelectric_point(sequence: str) -> float:
    r"""
    Calculate the Bjellqvist isoelectric point used by Biopython 1.87.
    """

    positive_pks = {"Nterm": 7.5, "K": 10.0, "R": 12.0, "H": 5.98}
    negative_pks = {"Cterm": 3.55, "D": 4.05, "E": 4.45, "C": 9.0, "Y": 10.0}
    n_terminal_pks = {
        "A": 7.59,
        "M": 7.0,
        "S": 6.93,
        "P": 8.36,
        "T": 6.82,
        "V": 7.44,
        "E": 7.7,
    }
    c_terminal_pks = {"D": 4.55, "E": 4.75}
    if sequence[0] in n_terminal_pks:
        positive_pks["Nterm"] = n_terminal_pks[sequence[0]]
    if sequence[-1] in c_terminal_pks:
        negative_pks["Cterm"] = c_terminal_pks[sequence[-1]]

    charged_content = {
        amino_acid: float(sequence.count(amino_acid)) for amino_acid in ("K", "R", "H", "D", "E", "C", "Y")
    }
    charged_content["Nterm"] = 1.0
    charged_content["Cterm"] = 1.0

    minimum = 4.05
    maximum = 12.0
    ph = 7.775
    while maximum - minimum > 0.0001:
        positive_charge = sum(
            charged_content[amino_acid] / (10 ** (ph - pk) + 1.0) for amino_acid, pk in positive_pks.items()
        )
        negative_charge = sum(
            charged_content[amino_acid] / (10 ** (pk - ph) + 1.0) for amino_acid, pk in negative_pks.items()
        )
        if positive_charge - negative_charge > 0.0:
            minimum = ph
        else:
            maximum = ph
        ph = (minimum + maximum) / 2
    return ph


@cache
def _language_model_amino_acid_layout() -> tuple[int, int, tuple[int, ...]]:
    r"""
    Locate the 20 standard amino acids in the MultiMolecule ESMC vocabulary.

    Returns ``(amino_acid_start, amino_acid_stop, permutation)`` where the returned range is the
    contiguous vocabulary block holding the 20 standard amino acids and
    ``permutation[folding_trunk_index]`` is the column of the folding-trunk amino acid
    ``folding_trunk_index`` (``_AA3_ORDER`` order) inside that block. The permutation is derived
    from the tokenizer rather than assuming fixed vocabulary ids.
    """

    from multimolecule import ProteinTokenizer

    vocabulary = ProteinTokenizer().get_vocab()
    amino_acid_token_indices = {amino_acid: vocabulary[amino_acid] for amino_acid in _INDEX_TO_AMINO_ACID.values()}
    amino_acid_start = min(amino_acid_token_indices.values())
    amino_acid_stop = max(amino_acid_token_indices.values()) + 1
    if amino_acid_stop - amino_acid_start != _AA_DIMS:
        raise ValueError("ESMC standard amino-acid vocabulary ids are not contiguous")
    language_model_order = [
        amino_acid
        for amino_acid, _ in sorted(
            amino_acid_token_indices.items(),
            key=lambda item: item[1],
        )
    ]
    permutation = tuple(language_model_order.index(_INDEX_TO_AMINO_ACID[index]) for index in range(_AA_DIMS))
    return amino_acid_start, amino_acid_stop, permutation


def _folding_trunk_to_language_model_amino_acid_matrix(device: torch.device) -> torch.Tensor:
    r"""
    ``[20, 20]`` permutation from folding-trunk amino-acid order to ESMC LM order.
    """

    _, _, permutation = _language_model_amino_acid_layout()
    matrix = torch.zeros(_AA_DIMS, _AA_DIMS, device=device)
    for folding_trunk_index, language_model_index in enumerate(permutation):
        matrix[folding_trunk_index, language_model_index] = 1.0
    return matrix


def _hard_categorical_probabilities(probabilities: torch.Tensor) -> torch.Tensor:
    return F.one_hot(torch.argmax(probabilities, dim=-1), num_classes=probabilities.size(-1)).to(probabilities.dtype)


def _straight_through(discrete: torch.Tensor, continuous: torch.Tensor) -> torch.Tensor:
    r"""
    Discrete value on the forward pass, continuous gradient on the backward pass.
    """

    return continuous + (discrete - continuous).detach()


def _esmc_pseudo_perplexity_nll(
    model: EsmFold2ExperimentalModel,
    design: torch.Tensor,
    score_mask: torch.Tensor,
    *,
    generator: torch.Generator | None,
    n_passes: int = _LM_MASK_PASSES,
    mask_fraction: float = _ESMC_MASK_FRACTION,
    chunk_size: int = _LM_LOSS_CHUNK,
) -> torch.Tensor:
    r"""
    Approximate ESMC pseudo-perplexity NLL of a soft designed sequence.

    ``design`` is ``[B, L, 20]`` soft amino-acid probabilities in folding-trunk order (the
    binder's per-position softmax). ``score_mask`` is ``[B, L]`` marking the mutable
    positions eligible for masking. Builds a differentiable categorical ESMC input (with a
    straight-through discretisation), samples ``n_passes`` random ``mask_fraction`` masks over
    the mutable positions, and returns the per-sample mean masked-token NLL of the designed
    distribution under the frozen ESMC masked-LM head. Gradient flows back to ``design``
    through the frozen trunk, giving a differentiable sequence-naturalness prior.
    """

    device = design.device
    esmc = model.esmc
    lm_head = model.esmc_lm_head
    esmc_config = model.config.esmc
    vocabulary_size = esmc_config.vocab_size
    embed_dtype = esmc.get_input_embeddings().weight.dtype
    amino_acid_start, amino_acid_stop, _ = _language_model_amino_acid_layout()
    folding_trunk_to_language_model = _folding_trunk_to_language_model_amino_acid_matrix(device)

    batch_size, binder_length = design.shape[0], design.shape[1]
    target_language_model_probabilities = design @ folding_trunk_to_language_model
    input_language_model_probabilities = _straight_through(
        _hard_categorical_probabilities(target_language_model_probabilities),
        target_language_model_probabilities,
    )

    input_token_probabilities = torch.zeros(
        batch_size,
        binder_length + 2,
        vocabulary_size,
        dtype=embed_dtype,
        device=device,
    )
    input_token_probabilities[:, 0, esmc_config.bos_token_id] = 1
    input_token_probabilities[:, -1, esmc_config.eos_token_id] = 1
    input_token_probabilities[:, 1:-1, amino_acid_start:amino_acid_stop] = input_language_model_probabilities.to(
        embed_dtype
    )

    if score_mask.ndim == 1:
        score_mask = score_mask.unsqueeze(0).expand(batch_size, -1)
    score_mask = score_mask.to(device=device, dtype=torch.bool)

    mask_token = torch.zeros(vocabulary_size, dtype=embed_dtype, device=device)
    mask_token[esmc_config.mask_token_id] = 1

    all_masked_rows: list[torch.Tensor] = []
    all_pass_masks: list[torch.Tensor] = []
    for sample in range(batch_size):
        position_indices = score_mask[sample].nonzero(as_tuple=False).flatten()
        num_positions = int(position_indices.numel())
        if num_positions == 0:
            raise ValueError("ESMC pseudo-perplexity score mask selected zero positions")
        num_masked = max(1, math.ceil(mask_fraction * num_positions))
        random_scores = torch.rand(
            (n_passes, num_positions),
            device=device,
            generator=generator,
        )
        masked_offsets = random_scores.topk(num_masked, dim=-1, largest=False).indices
        pass_masks = torch.zeros((n_passes, binder_length), dtype=torch.bool, device=device)
        pass_masks[torch.arange(n_passes, device=device)[:, None], position_indices[masked_offsets]] = True
        masked = input_token_probabilities[sample : sample + 1].repeat(n_passes, 1, 1)
        mask_rows, mask_cols = pass_masks.nonzero(as_tuple=True)
        masked[mask_rows, mask_cols + 1] = mask_token
        all_masked_rows.append(masked)
        all_pass_masks.append(pass_masks)

    masked_sequences = torch.cat(all_masked_rows, dim=0)
    language_model_mask = torch.ones(
        masked_sequences.shape[0],
        masked_sequences.shape[1],
        dtype=torch.bool,
        device=device,
    )
    logit_chunks: list[torch.Tensor] = []
    for start in range(0, masked_sequences.shape[0], chunk_size):
        chunk = masked_sequences[start : start + chunk_size]
        logit_chunks.append(
            compute_esmc_masked_lm_logits(
                esmc,
                lm_head,
                chunk,
                language_model_mask[start : start + chunk_size],
            )
        )
    logits = torch.cat(logit_chunks, dim=0)

    losses: list[torch.Tensor] = []
    for sample, pass_masks in enumerate(all_pass_masks):
        start = sample * n_passes
        stop = start + n_passes
        target_weights = target_language_model_probabilities[sample]
        log_probs = logits[start:stop].log_softmax(dim=-1)[
            :,
            1:-1,
            amino_acid_start:amino_acid_stop,
        ]
        nlls = -(log_probs * target_weights.to(log_probs.dtype).unsqueeze(0)).sum(dim=-1)
        losses.append(nlls[pass_masks].mean())
    return torch.stack(losses, dim=0)


def _get_mid_points(device: torch.device) -> torch.Tensor:
    r"""
    128 distance-bin midpoints over the 1-57 angstrom experimental distogram range.
    """

    boundaries = torch.linspace(2, 52.0, 127, device=device)
    lower = torch.tensor([1.0], device=device)
    upper = torch.tensor([52.0 + 5.0], device=device)
    exp_boundaries = torch.cat((lower, boundaries, upper))
    return (exp_boundaries[:-1] + exp_boundaries[1:]) / 2


def _binned_entropy(dgram: torch.Tensor, bin_distance: torch.Tensor, cutoff: float) -> torch.Tensor:
    bin_mask = ~(bin_distance < cutoff)
    masked_dgram = dgram - (1e7 * bin_mask)
    px = torch.softmax(masked_dgram, dim=-1)
    log_px = torch.log_softmax(dgram, dim=-1)
    return -(px * log_px).sum(-1)


def _masked_min_k(x: torch.Tensor, mask: torch.Tensor, k: int) -> torch.Tensor:
    mask = mask.bool()
    y = torch.sort(torch.where(mask, x, float("nan")))[0]
    k_mask = (torch.arange(y.shape[-1], device=y.device) < k) & (~torch.isnan(y))
    return torch.where(k_mask, y, 0).sum(-1) / (k_mask.sum(-1) + 1e-8)


def _masked_average(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.bool()
    return torch.where(mask, x, 0).sum(-1) / (torch.where(mask, 1, 0).sum(-1) + 1e-8)


def _contact_loss(
    distogram_logits: torch.Tensor,
    bin_distance: torch.Tensor,
    num_contacts: int,
    min_sep: int,
    cutoff: float,
    chain_mask: torch.Tensor,
    binder_mask: torch.Tensor,
) -> torch.Tensor:
    r"""
    Compute entropy contact loss with a sequence-separation constraint.
    """

    con_loss = _binned_entropy(distogram_logits, bin_distance, cutoff)
    position = torch.arange(distogram_logits.shape[1], device=distogram_logits.device)
    p_dist = position[:, None] - position[None, :]
    if min_sep > 0:
        separation_mask = torch.abs(p_dist) >= min_sep
        binder_mask = torch.logical_and(separation_mask, binder_mask)
    per_residue = _masked_min_k(con_loss, mask=binder_mask, k=num_contacts)
    return _masked_average(per_residue, mask=chain_mask)


def _structure_losses(distogram_logits: torch.Tensor, binder_length: int) -> dict[str, torch.Tensor]:
    r"""
    Weighted intra-contact + inter-contact + globularity objective (binder at suffix).
    """

    bin_distance = _get_mid_points(distogram_logits.device)
    full_len = distogram_logits.shape[1]
    is_binder = torch.ones(full_len, device=distogram_logits.device)
    is_binder[:-binder_length] *= 0.0

    intra = _contact_loss(
        distogram_logits,
        bin_distance,
        num_contacts=2,
        min_sep=9,
        cutoff=14.0,
        chain_mask=is_binder,
        binder_mask=is_binder,
    )
    inter = _contact_loss(
        distogram_logits,
        bin_distance,
        num_contacts=1,
        min_sep=0,
        cutoff=22.0,
        chain_mask=1 - is_binder,
        binder_mask=is_binder,
    )

    binder_disto = distogram_logits[:, -binder_length:, -binder_length:, :]
    n = binder_disto.shape[1]
    disto_probs = torch.softmax(binder_disto, dim=-1)
    rg_bins = bin_distance.clamp(max=27)
    e_sq_dist = torch.sum(disto_probs * torch.square(rg_bins), dim=-1)
    sum_sq_dist = torch.sum(torch.tril(e_sq_dist, diagonal=-1), dim=(1, 2))
    rg_term = torch.sqrt(sum_sq_dist / (n * n))
    rg_th = 2.38 * (n**0.365)
    glob = F.elu(rg_term - rg_th)

    total = (
        _LOSS_WEIGHTS["intra_contact"] * intra + _LOSS_WEIGHTS["inter_contact"] * inter + _LOSS_WEIGHTS["glob"] * glob
    )
    return {"intra_contact": intra, "inter_contact": inter, "glob": glob, "total": total}


def _binding_confidence_entropy(dgram: torch.Tensor, bin_distance: torch.Tensor, cutoff: float) -> torch.Tensor:
    probs = torch.softmax(dgram, dim=-1)
    cutoff_mask = bin_distance < cutoff
    p_cut = probs[..., cutoff_mask]
    p_cut = p_cut / (p_cut.sum(-1, keepdim=True) + 1e-8)
    return -(p_cut * torch.log(p_cut + 1e-10)).sum(-1)


def _distogram_iptm_proxy(distogram_logits: torch.Tensor, target_length: int, binder_length: int) -> float:
    r"""
    Return an entropy-of-binding proxy for target-binder ipTM.
    """

    if distogram_logits.ndim == 4:
        distogram_logits = distogram_logits[0]
    bin_distance = _get_mid_points(distogram_logits.device)
    binder_to_target = _binding_confidence_entropy(
        distogram_logits[target_length:, :target_length, :], bin_distance, cutoff=22.0
    )
    sorted_entropies, _ = torch.sort(binder_to_target.reshape(-1))
    k = min(binder_length, sorted_entropies.numel())
    mean_entropy = float(sorted_entropies[:k].mean())
    return float(max(0.0, min(1.0, 1.0 - mean_entropy / math.log(51))))


def cdr_distogram_iptm_proxy(distogram_logits: torch.Tensor, target_length: int, binder_sequence: str) -> float:
    r"""
    Score target contacts made by antibody CDR residues.

    Args:
        distogram_logits: Pairwise distance-bin logits with shape ``(N, N, D)`` or
            ``(B, N, N, D)``. For a four-dimensional tensor, only the first batch item
            is scored.
        target_length: Number of target tokens preceding the binder tokens.
        binder_sequence: Antibody Fv sequence aligned to the binder-token rows.

    Returns:
        An entropy-based contact confidence in the closed interval ``[0, 1]``.
    """

    if distogram_logits.ndim == 4:
        distogram_logits = distogram_logits[0]
    bin_distance = _get_mid_points(distogram_logits.device)
    cdr_indices = antibody_cdr_indices(binder_sequence)
    cdr_rows = [target_length + i for i in cdr_indices]
    cdr_to_target = _binding_confidence_entropy(
        distogram_logits[cdr_rows, :target_length, :], bin_distance, cutoff=22.0
    )
    sorted_entropies, _ = torch.sort(cdr_to_target.reshape(-1))
    k = min(len(cdr_indices), sorted_entropies.numel())
    mean_entropy = float(sorted_entropies[:k].mean())
    return float(max(0.0, min(1.0, 1.0 - mean_entropy / math.log(51))))


def _normalized_gradient(grad: torch.Tensor, gradient_mask: torch.Tensor) -> torch.Tensor:
    r"""
    Return a per-sample length-normalized gradient over mutable positions.
    """

    masked_grad = grad * gradient_mask
    index_has_grad = torch.square(masked_grad).sum(-1) > 0
    eff_len = index_has_grad.sum(-1)
    grad_norm = torch.linalg.norm(masked_grad, dim=(-1, -2))
    normalized = (masked_grad / (grad_norm[:, None, None] + 1e-7)) * torch.sqrt(eff_len[:, None, None])
    return normalized * gradient_mask
