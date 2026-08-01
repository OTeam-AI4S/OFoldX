r"""
ESMFold2 structure-prediction and binder-design pipelines.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from functools import cache
from os import PathLike
from pathlib import Path
from typing import Any, NamedTuple, cast

import torch
import torch.nn.functional as F

from ...data.evidence import MSA, MsaCollection
from ...data.molecular import (
    AMINO_ACID_THREE_TO_ONE,
    ENTITY_PROTEIN,
    MolecularSystem,
    PolymerSequence,
)
from ...data.structure import Structure
from ...data.tokenization import TOKEN_INDEX, TOKEN_VOCAB_SIZE
from ...pipelines import CompositePair, CompositePipeline, PairedPipeline, resolve_generator
from ...pipelines.input import resolve_msa_input, resolve_polymer_input
from ...references import classify_reference
from ...results import FoldPrediction, FoldResult, Metric, PairMetric, Score
from ...results.values import best
from .model import (
    EsmFold2ExperimentalModel,
    EsmFold2ExperimentalSamplingOutput,
    EsmFold2ExperimentalSamplingWithConfidenceOutput,
    EsmFold2Input,
    EsmFold2Model,
    EsmFold2SamplingWithConfidenceOutput,
)
from .processor import (
    EsmFold2ExperimentalProcessor,
    EsmFold2Processor,
    antibody_cdr_indices,
    esmfold2_molecular_system_from_named_sequences,
    esmfold2_molecular_system_from_sequences,
    is_antibody_sequence,
)

# 20-dimensional design amino-acid space, ordered ALA..VAL.
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
_DESIGN_TOKEN_INDICES = tuple(TOKEN_INDEX[AMINO_ACID_THREE_TO_ONE[component]] for component in _AA3_ORDER)
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

    def prepare_fold(
        self,
        input: str | PathLike[str] | PolymerSequence | Sequence[PolymerSequence] | MolecularSystem | Structure,
        *,
        msa: MSA | MsaCollection | str | PathLike[str] | None = None,
    ) -> EsmFold2Input:
        r"""Prepare one molecular system and its optional MSA for ESMFold2 folding."""

        system = _fold_system(input)
        prepared_msa = _fold_msa(msa, system)
        return self.processor(system, msa=prepared_msa)

    @torch.inference_mode()
    def fold(
        self,
        input: str | PathLike[str] | PolymerSequence | Sequence[PolymerSequence] | MolecularSystem | Structure,
        *,
        msa: MSA | MsaCollection | str | PathLike[str] | None = None,
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

        generator = resolve_generator(seed, generator, device=self.processor.device)
        batch = self.prepare_fold(input, msa=msa)
        output = self.model.sample_with_confidence(
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
        return self.decode_fold(batch, output)

    def decode_fold(
        self,
        input: EsmFold2Input,
        output: EsmFold2SamplingWithConfidenceOutput,
    ) -> FoldResult:
        r"""Decode one raw ESMFold2 folding output without executing the model."""

        if input.token_index.shape[0] != 1 or input.system.system_entity_ptr.numel() != 2:
            raise ValueError("EsmFold2Pipeline.decode_fold accepts one canonical system per call")

        system = input.system.cpu()  # type: ignore[attr-defined]
        model_atom_mask = input.atom_index[0] >= 0
        canonical_atom_index = input.atom_index[0, model_atom_mask].detach().cpu()
        token_component_index = input.token_component_index[0].detach().cpu()
        token_atom_index = input.token_atom_index[0].detach().cpu()
        token_axis = tuple(
            (
                system.atom_site(int(atom_index))
                if int(atom_index) >= 0
                else system.component_reference(int(component_index))
            )
            for component_index, atom_index in zip(token_component_index, token_atom_index)
        )
        atom_axis = tuple(system.atom_site(int(atom_index)) for atom_index in canonical_atom_index)
        plddt = output.plddt[0].detach().to(device="cpu", dtype=torch.float32).numpy()
        plddt_ca = output.plddt_ca[0].detach().to(device="cpu", dtype=torch.float32).numpy()
        pae = output.pae[0].detach().to(device="cpu", dtype=torch.float32).numpy()
        resolved_probability = (
            output.resolved_logits[0, :, model_atom_mask].detach().float().softmax(dim=-1)[..., 1].cpu().numpy()
        )
        chain_axis = tuple(system.instance_id(index) for index in range(system.instance_symbol.numel()))
        predictions: list[FoldPrediction] = []
        for sample_index in range(output.coordinates.shape[1]):
            coordinates = torch.zeros(system.atom_symbol.numel(), 3, dtype=torch.float32)
            coordinate_mask = torch.zeros(system.atom_symbol.numel(), dtype=torch.bool)
            coordinates[canonical_atom_index] = (
                output.coordinates[0, sample_index, model_atom_mask].detach().to(device="cpu", dtype=torch.float32)
            )
            coordinate_mask[canonical_atom_index] = True
            structure = Structure(  # type: ignore[call-arg]
                system=system,
                coordinates=coordinates,
                coordinate_mask=coordinate_mask,
                occupancy=torch.zeros(system.atom_symbol.numel(), dtype=torch.float32),
                occupancy_mask=torch.zeros(system.atom_symbol.numel(), dtype=torch.bool),
                b_factor=torch.zeros(system.atom_symbol.numel(), dtype=torch.float32),
                b_factor_mask=torch.zeros(system.atom_symbol.numel(), dtype=torch.bool),
                source_symbols=(),
                residue_id=torch.full((system.component_symbol.numel(),), -1, dtype=torch.long),
                insertion_code=torch.full((system.component_symbol.numel(),), -1, dtype=torch.long),
                batch_size=[],
            )
            predictions.append(
                FoldPrediction(
                    structure=structure,
                    scores={
                        "ptm": Score(
                            float(output.ptm[0, sample_index].detach().cpu()),
                            preferred="higher",
                        ),
                        "iptm": Score(
                            float(output.iptm[0, sample_index].detach().cpu()),
                            preferred="higher",
                        ),
                        "mean_plddt": Score(
                            float(output.mean_plddt[0, sample_index].detach().cpu()),
                            preferred="higher",
                        ),
                        "complex_iplddt": Score(
                            float(output.complex_iplddt[0, sample_index].detach().cpu()),
                            preferred="higher",
                        ),
                    },
                    metrics={
                        "plddt": Metric(values=plddt[sample_index], axis=token_axis),
                        "plddt_per_atom": Metric(
                            values=(
                                output.plddt_per_atom[0, sample_index, model_atom_mask].detach().float().cpu().numpy()
                            ),
                            axis=atom_axis,
                        ),
                        "plddt_ca": Metric(
                            values=plddt_ca[sample_index],
                            axis=token_axis,
                        ),
                        "resolved_probability": Metric(
                            values=resolved_probability[sample_index],
                            axis=atom_axis,
                        ),
                    },
                    pair_metrics={
                        "pae": PairMetric(
                            values=pae[sample_index],
                            row_axis=token_axis,
                            column_axis=token_axis,
                            unit="angstrom",
                        ),
                        "pde": PairMetric(
                            values=output.pde[0, sample_index].detach().float().cpu().numpy(),
                            row_axis=token_axis,
                            column_axis=token_axis,
                            unit="angstrom",
                        ),
                        "chain_pair_iptm": PairMetric(
                            values=output.chain_pair_iptm[0, sample_index].detach().float().cpu().numpy(),
                            row_axis=chain_axis,
                            column_axis=chain_axis,
                        ),
                    },
                )
            )
        return FoldResult(predictions=tuple(predictions))


class EsmFold2ExperimentalPipeline(PairedPipeline[EsmFold2ExperimentalModel, EsmFold2ExperimentalProcessor]):
    r"""
    Independently callable experimental ESMFold2 Model--Processor pair.
    """

    model_architecture = "esmfold2_experimental"

    def prepare_fold(
        self,
        input: str | PathLike[str] | PolymerSequence | Sequence[PolymerSequence] | MolecularSystem | Structure,
        *,
        msa: MSA | MsaCollection | str | PathLike[str] | None = None,
    ) -> EsmFold2Input:
        r"""Prepare one molecular system and its optional MSA for experimental folding."""

        system = _fold_system(input)
        prepared_msa = _fold_msa(msa, system)
        return self.processor(system, msa=prepared_msa)

    @torch.inference_mode()
    def fold(
        self,
        input: str | PathLike[str] | PolymerSequence | Sequence[PolymerSequence] | MolecularSystem | Structure,
        *,
        msa: MSA | MsaCollection | str | PathLike[str] | None = None,
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
            A [`FoldResult`][ofoldx.results.FoldResult] containing structures and,
                when enabled by the checkpoint, confidence values.
        """

        generator = resolve_generator(seed, generator, device=self.processor.device)
        processed = self.prepare_fold(input, msa=msa)
        output: EsmFold2ExperimentalSamplingOutput | EsmFold2ExperimentalSamplingWithConfidenceOutput
        if self.model.confidence_head is None:
            output = self.model.sample(
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
        else:
            output = self.model.sample_with_confidence(
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
        return self.decode_fold(processed, output)

    def decode_fold(
        self,
        input: EsmFold2Input,
        output: EsmFold2ExperimentalSamplingOutput | EsmFold2ExperimentalSamplingWithConfidenceOutput,
    ) -> FoldResult:
        r"""Decode one raw experimental folding output without executing the model."""

        if input.token_index.shape[0] != 1 or input.system.system_entity_ptr.numel() != 2:
            raise ValueError("EsmFold2ExperimentalPipeline.decode_fold accepts one canonical system per call")
        if isinstance(output, EsmFold2ExperimentalSamplingWithConfidenceOutput):
            return FoldResult(
                predictions=tuple(
                    _experimental_fold_prediction(input, output, batch_index=0, sample_index=sample_index)
                    for sample_index in range(output.sample.coordinates.shape[1])
                )
            )
        return FoldResult(
            predictions=tuple(
                FoldPrediction(
                    structure=_experimental_fold_structure(
                        input,
                        output,
                        batch_index=0,
                        sample_index=sample_index,
                    ),
                    scores={},
                    metrics={},
                    pair_metrics={},
                )
                for sample_index in range(output.coordinates.shape[1])
            )
        )

    @contextmanager
    def _design_mode(self, language_model_dropout: float) -> Iterator[None]:
        r"""Freeze the pair in eval mode with one temporary language-model dropout."""

        model = self.model
        dropout = model.language_model_dropout
        force = model.force_language_model_dropout
        model.configure_language_model_dropout(language_model_dropout, force_during_inference=True)
        try:
            with _frozen_eval(model):
                yield
        finally:
            model.configure_language_model_dropout(dropout, force_during_inference=force)


@dataclass(frozen=True)
class EsmFold2DesignStep:
    r"""
    One candidate's scientific values from one inversion optimization step.

    Attributes:
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
class EsmFold2HeroCriticEvaluation:
    r"""
    One confidence-enabled hero-critic evaluation of a designed sequence.

    Attributes:
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
class EsmFold2ScalingCriticEvaluation:
    r"""
    One confidence-disabled scaling-critic evaluation of a designed sequence.

    Attributes:
        checkpoint: Exact released critic checkpoint role.
        structure: Sampled target--binder complex.
        distogram_iptm_proxy: Whole-binder distogram interface proxy.
        cdr_distogram_iptm_proxy: Antibody CDR-only proxy, or ``None`` for
            non-antibody design.
    """

    checkpoint: str
    structure: Structure
    distogram_iptm_proxy: float
    cdr_distogram_iptm_proxy: float | None


class _EsmFold2InversionCandidate(NamedTuple):
    sequence: str
    best_iptm: float
    inversion_checkpoint: str
    inversion_prediction: FoldPrediction
    inversion_distogram_iptm_proxy: float
    final_logits: torch.Tensor
    trajectory: tuple[EsmFold2DesignStep, ...]


@dataclass(eq=False)
class EsmFold2DesignCandidate:
    r"""
    One folded target-binder complex and its aligned scientific values.

    Attributes:
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
        hero_critic_evaluations: Four confidence-enabled hero evaluations.
        scaling_critic_evaluations: Executed confidence-disabled scaling
            evaluations.
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
    hero_critic_evaluations: tuple[EsmFold2HeroCriticEvaluation, ...]
    scaling_critic_evaluations: tuple[EsmFold2ScalingCriticEvaluation, ...]
    is_antibody: bool
    isoelectric_point: float
    passes_isoelectric_point_filter: bool
    hero_mean_iptm: float
    scaling_mean_iptm_proxy: float
    selection_score: float
    scores: dict[str, Score]
    metrics: dict[str, Metric]
    pair_metrics: dict[str, PairMetric]


@dataclass(eq=False)
class EsmFold2DesignResult:
    r"""
    A generation-ordered collection of folded binder designs.

    Attributes:
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

    @property
    def best(self) -> EsmFold2DesignCandidate:
        r"""Return the highest-ranked candidate under the released selection."""

        return self.candidates[self.selected_indices[0]]

    def best_by(self, score: str) -> EsmFold2DesignCandidate:
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
        CompositePair(
            "fast",
            "",
            EsmFold2ExperimentalPipeline,
        ),
        CompositePair(
            "fast_cutoff2025",
            "fast_cutoff2025",
            EsmFold2ExperimentalPipeline,
        ),
        CompositePair(
            "experimental",
            "experimental",
            EsmFold2ExperimentalPipeline,
        ),
        CompositePair(
            "experimental_cutoff2025",
            "experimental_cutoff2025",
            EsmFold2ExperimentalPipeline,
        ),
        CompositePair(
            "base300m_step250k",
            "scaling/base300m_step250k",
            EsmFold2ExperimentalPipeline,
        ),
        CompositePair(
            "base300m_step500k",
            "scaling/base300m_step500k",
            EsmFold2ExperimentalPipeline,
        ),
        CompositePair(
            "base300m_step750k",
            "scaling/base300m_step750k",
            EsmFold2ExperimentalPipeline,
        ),
        CompositePair(
            "base300m_step1000k",
            "scaling/base300m_step1000k",
            EsmFold2ExperimentalPipeline,
        ),
        CompositePair(
            "base300m_step1500k",
            "scaling/base300m_step1500k",
            EsmFold2ExperimentalPipeline,
        ),
        CompositePair(
            "base600m_step250k",
            "scaling/base600m_step250k",
            EsmFold2ExperimentalPipeline,
        ),
        CompositePair(
            "base600m_step500k",
            "scaling/base600m_step500k",
            EsmFold2ExperimentalPipeline,
        ),
        CompositePair(
            "base600m_step750k",
            "scaling/base600m_step750k",
            EsmFold2ExperimentalPipeline,
        ),
        CompositePair(
            "base600m_step1000k",
            "scaling/base600m_step1000k",
            EsmFold2ExperimentalPipeline,
        ),
        CompositePair(
            "base600m_step1500k",
            "scaling/base600m_step1500k",
            EsmFold2ExperimentalPipeline,
        ),
        CompositePair(
            "base6b_step250k",
            "scaling/base6b_step250k",
            EsmFold2ExperimentalPipeline,
        ),
        CompositePair(
            "base6b_step500k",
            "scaling/base6b_step500k",
            EsmFold2ExperimentalPipeline,
        ),
        CompositePair(
            "base6b_step750k",
            "scaling/base6b_step750k",
            EsmFold2ExperimentalPipeline,
        ),
        CompositePair(
            "base6b_step1000k",
            "scaling/base6b_step1000k",
            EsmFold2ExperimentalPipeline,
        ),
        CompositePair(
            "base6b_step1500k",
            "scaling/base6b_step1500k",
            EsmFold2ExperimentalPipeline,
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
        super().__init__(
            {
                "fast": fast,
                "fast_cutoff2025": fast_cutoff2025,
                "experimental": experimental,
                "experimental_cutoff2025": experimental_cutoff2025,
                "base300m_step250k": base300m_step250k,
                "base300m_step500k": base300m_step500k,
                "base300m_step750k": base300m_step750k,
                "base300m_step1000k": base300m_step1000k,
                "base300m_step1500k": base300m_step1500k,
                "base600m_step250k": base600m_step250k,
                "base600m_step500k": base600m_step500k,
                "base600m_step750k": base600m_step750k,
                "base600m_step1000k": base600m_step1000k,
                "base600m_step1500k": base600m_step1500k,
                "base6b_step250k": base6b_step250k,
                "base6b_step500k": base6b_step500k,
                "base6b_step750k": base6b_step750k,
                "base6b_step1000k": base6b_step1000k,
                "base6b_step1500k": base6b_step1500k,
            }
        )

    @property
    def fast(self) -> EsmFold2ExperimentalPipeline:
        return cast(EsmFold2ExperimentalPipeline, self.components["fast"])

    @property
    def fast_cutoff2025(self) -> EsmFold2ExperimentalPipeline:
        return cast(EsmFold2ExperimentalPipeline, self.components["fast_cutoff2025"])

    @property
    def experimental(self) -> EsmFold2ExperimentalPipeline:
        return cast(EsmFold2ExperimentalPipeline, self.components["experimental"])

    @property
    def experimental_cutoff2025(self) -> EsmFold2ExperimentalPipeline:
        return cast(EsmFold2ExperimentalPipeline, self.components["experimental_cutoff2025"])

    @property
    def base300m_step250k(self) -> EsmFold2ExperimentalPipeline:
        return cast(EsmFold2ExperimentalPipeline, self.components["base300m_step250k"])

    @property
    def base300m_step500k(self) -> EsmFold2ExperimentalPipeline:
        return cast(EsmFold2ExperimentalPipeline, self.components["base300m_step500k"])

    @property
    def base300m_step750k(self) -> EsmFold2ExperimentalPipeline:
        return cast(EsmFold2ExperimentalPipeline, self.components["base300m_step750k"])

    @property
    def base300m_step1000k(self) -> EsmFold2ExperimentalPipeline:
        return cast(EsmFold2ExperimentalPipeline, self.components["base300m_step1000k"])

    @property
    def base300m_step1500k(self) -> EsmFold2ExperimentalPipeline:
        return cast(EsmFold2ExperimentalPipeline, self.components["base300m_step1500k"])

    @property
    def base600m_step250k(self) -> EsmFold2ExperimentalPipeline:
        return cast(EsmFold2ExperimentalPipeline, self.components["base600m_step250k"])

    @property
    def base600m_step500k(self) -> EsmFold2ExperimentalPipeline:
        return cast(EsmFold2ExperimentalPipeline, self.components["base600m_step500k"])

    @property
    def base600m_step750k(self) -> EsmFold2ExperimentalPipeline:
        return cast(EsmFold2ExperimentalPipeline, self.components["base600m_step750k"])

    @property
    def base600m_step1000k(self) -> EsmFold2ExperimentalPipeline:
        return cast(EsmFold2ExperimentalPipeline, self.components["base600m_step1000k"])

    @property
    def base600m_step1500k(self) -> EsmFold2ExperimentalPipeline:
        return cast(EsmFold2ExperimentalPipeline, self.components["base600m_step1500k"])

    @property
    def base6b_step250k(self) -> EsmFold2ExperimentalPipeline:
        return cast(EsmFold2ExperimentalPipeline, self.components["base6b_step250k"])

    @property
    def base6b_step500k(self) -> EsmFold2ExperimentalPipeline:
        return cast(EsmFold2ExperimentalPipeline, self.components["base6b_step500k"])

    @property
    def base6b_step750k(self) -> EsmFold2ExperimentalPipeline:
        return cast(EsmFold2ExperimentalPipeline, self.components["base6b_step750k"])

    @property
    def base6b_step1000k(self) -> EsmFold2ExperimentalPipeline:
        return cast(EsmFold2ExperimentalPipeline, self.components["base6b_step1000k"])

    @property
    def base6b_step1500k(self) -> EsmFold2ExperimentalPipeline:
        return cast(EsmFold2ExperimentalPipeline, self.components["base6b_step1500k"])

    def design(
        self,
        input: str | PathLike[str] | Structure | MolecularSystem,
        *,
        binder_length: int,
        binder_prompt: str | None = None,
        num_candidates: int = 1,
        num_optimization_steps: int = 150,
        is_antibody: bool | None = None,
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
            is_antibody: Whether the prompt is an antibody design. ``None`` detects
                antibody variable domains from ``binder_prompt``.
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
        is_antibody = is_antibody_sequence(prompt) if is_antibody is None else is_antibody

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
        generator = resolve_generator(seed, generator, device=self.fast.processor.device)
        active_scaling_critics = scaling_critics if use_scaling_critics else ()
        active_critics = hero_critics + active_scaling_critics
        with self.fast._design_mode(0.5), self.fast_cutoff2025._design_mode(0.5):
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
        with ExitStack() as stack:
            for _, pipeline in active_critics:
                stack.enter_context(pipeline._design_mode(0.25))
            return self._score_candidates(
                candidates,
                target_sequence=target_sequence,
                is_antibody=is_antibody,
                hero_critics=hero_critics,
                scaling_critics=active_scaling_critics,
                generator=generator,
            )

    def _score_candidates(
        self,
        candidates: tuple[_EsmFold2InversionCandidate, ...],
        *,
        target_sequence: str,
        is_antibody: bool,
        hero_critics: tuple[tuple[str, EsmFold2ExperimentalPipeline], ...],
        scaling_critics: tuple[tuple[str, EsmFold2ExperimentalPipeline], ...],
        generator: torch.Generator | None,
    ) -> EsmFold2DesignResult:
        target_length = len(target_sequence)
        hero_evaluations: list[list[EsmFold2HeroCriticEvaluation]] = [[] for _ in candidates]
        scaling_evaluations: list[list[EsmFold2ScalingCriticEvaluation]] = [[] for _ in candidates]

        for checkpoint, pipeline in hero_critics:
            individual_inputs = tuple(
                pipeline.prepare_fold(
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
                    hero_evaluations[candidate_index].append(
                        EsmFold2HeroCriticEvaluation(
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

        for checkpoint, pipeline in scaling_critics:
            individual_inputs = tuple(
                pipeline.prepare_fold(
                    [
                        PolymerSequence.from_one_letter(target_sequence),
                        PolymerSequence.from_one_letter(candidate.sequence),
                    ]
                )
                for candidate in candidates
            )
            batch = _stack_design_inputs(individual_inputs)
            with torch.inference_mode():
                sample = pipeline.model.sample(
                    batch,
                    recycling_steps=4,
                    num_samples=1,
                    num_sampling_steps=200,
                    language_model_mask_probability=0.0,
                    generator=generator,
                )
                for candidate_index, candidate in enumerate(candidates):
                    distogram = sample.distogram_logits[candidate_index].float()
                    scaling_evaluations[candidate_index].append(
                        EsmFold2ScalingCriticEvaluation(
                            checkpoint=checkpoint,
                            structure=_experimental_fold_structure(
                                individual_inputs[candidate_index],
                                sample,
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
        for candidate, candidate_hero_evaluations, candidate_scaling_evaluations in zip(
            candidates,
            hero_evaluations,
            scaling_evaluations,
        ):
            hero_mean_iptm = sum(
                evaluation.prediction.scores["iptm"].value for evaluation in candidate_hero_evaluations
            ) / len(candidate_hero_evaluations)
            if candidate_scaling_evaluations:
                if is_antibody:
                    scaling_mean_proxy = sum(
                        cast(float, evaluation.cdr_distogram_iptm_proxy) for evaluation in candidate_scaling_evaluations
                    ) / len(candidate_scaling_evaluations)
                else:
                    scaling_mean_proxy = sum(
                        evaluation.distogram_iptm_proxy for evaluation in candidate_scaling_evaluations
                    ) / len(candidate_scaling_evaluations)
            else:
                scaling_mean_proxy = 0.0
            selection_score = 0.5 * hero_mean_iptm + 0.5 * scaling_mean_proxy
            isoelectric_point = _isoelectric_point(candidate.sequence)
            passes_filter = is_antibody or isoelectric_point < 6.0
            selected_evaluation = next(
                evaluation
                for evaluation in candidate_hero_evaluations
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
                EsmFold2DesignCandidate(
                    sequence=candidate.sequence,
                    structure=selected_prediction.structure,
                    final_logits=candidate.final_logits,
                    best_iptm=candidate.best_iptm,
                    trajectory=candidate.trajectory,
                    inversion_checkpoint=candidate.inversion_checkpoint,
                    inversion_prediction=candidate.inversion_prediction,
                    inversion_distogram_iptm_proxy=candidate.inversion_distogram_iptm_proxy,
                    hero_critic_evaluations=tuple(candidate_hero_evaluations),
                    scaling_critic_evaluations=tuple(candidate_scaling_evaluations),
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
                key=lambda index: scored[index].selection_score,
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
    ) -> tuple[_EsmFold2InversionCandidate, ...]:
        device = pairs[0][1].processor.device
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
        selected: list[_EsmFold2InversionCandidate | None] = [None] * num_candidates

        for step in range(num_optimization_steps):
            choice = int(torch.randint(len(pairs), (), device=device, generator=generator).item())
            checkpoint, pipeline = pairs[choice]
            t = (step + 1) / num_optimization_steps
            temperature = _TEMPERATURE_MIN + (1 - _TEMPERATURE_MIN) * 0.5 * (1 + math.cos(math.pi * t))

            optimizer.zero_grad()
            design = F.softmax(logits / temperature, dim=-1)
            binder_sequences = tuple(
                "".join(_INDEX_TO_AMINO_ACID[int(index)] for index in row) for row in design.argmax(dim=-1)
            )
            individual_inputs = tuple(
                pipeline.prepare_fold(
                    [
                        PolymerSequence.from_one_letter(target_sequence),
                        PolymerSequence.from_one_letter(binder_sequence),
                    ]
                )
                for binder_sequence in binder_sequences
            )
            batch = _stack_design_inputs(individual_inputs)
            binder_probabilities = design.new_zeros(*design.shape[:-1], TOKEN_VOCAB_SIZE)
            binder_probabilities[..., list(_DESIGN_TOKEN_INDICES)] = design
            residue_type_probabilities = torch.cat(
                (
                    F.one_hot(
                        batch.residue_type_ids[:, :target_length].long(),
                        num_classes=TOKEN_VOCAB_SIZE,
                    ).float(),
                    binder_probabilities,
                ),
                dim=1,
            )

            prediction: EsmFold2ExperimentalSamplingWithConfidenceOutput | None = None
            if temperature < 0.05:
                prediction = pipeline.model.sample_with_confidence(
                    batch,
                    residue_type_probabilities=residue_type_probabilities,
                    recycling_steps=2,
                    num_samples=1,
                    num_sampling_steps=50,
                    generator=generator,
                )
                distogram = prediction.sample.distogram_logits.float()
            else:
                sample = pipeline.model.sample(
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
                pipeline.model,
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

        candidates: list[_EsmFold2InversionCandidate] = []
        for candidate_index, candidate in enumerate(selected):
            assert candidate is not None
            candidates.append(
                _EsmFold2InversionCandidate(
                    sequence=candidate.sequence,
                    best_iptm=candidate.best_iptm,
                    inversion_checkpoint=candidate.inversion_checkpoint,
                    inversion_prediction=candidate.inversion_prediction,
                    inversion_distogram_iptm_proxy=candidate.inversion_distogram_iptm_proxy,
                    final_logits=logits[candidate_index].detach().cpu(),
                    trajectory=tuple(trajectories[candidate_index]),
                )
            )
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
        prediction: EsmFold2ExperimentalSamplingWithConfidenceOutput,
        batch_index: int,
        target_length: int,
        inversion_checkpoint: str,
        binder_sequence: str,
        iptm: float,
        final_logits: torch.Tensor,
        trajectory: tuple[EsmFold2DesignStep, ...],
    ) -> _EsmFold2InversionCandidate:
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
        return _EsmFold2InversionCandidate(
            sequence=binder_sequence,
            best_iptm=iptm,
            inversion_checkpoint=inversion_checkpoint,
            inversion_prediction=inversion_prediction,
            inversion_distogram_iptm_proxy=proxy,
            final_logits=final_logits,
            trajectory=trajectory,
        )


def _experimental_fold_structure(
    input: EsmFold2Input,
    sample: EsmFold2ExperimentalSamplingOutput,
    *,
    batch_index: int,
    sample_index: int,
) -> Structure:
    system = input.system.cpu()  # type: ignore[attr-defined]
    model_atom_mask = input.atom_index[0] >= 0
    canonical_atom_index = input.atom_index[0, model_atom_mask].detach().cpu()
    coordinates = torch.zeros(system.atom_symbol.numel(), 3, dtype=torch.float32)
    coordinate_mask = torch.zeros(system.atom_symbol.numel(), dtype=torch.bool)
    coordinates[canonical_atom_index] = (
        sample.coordinates[batch_index, sample_index, model_atom_mask].detach().to(device="cpu", dtype=torch.float32)
    )
    coordinate_mask[canonical_atom_index] = True
    return Structure(  # type: ignore[call-arg]
        system=system,
        coordinates=coordinates,
        coordinate_mask=coordinate_mask,
        occupancy=torch.zeros(system.atom_symbol.numel(), dtype=torch.float32),
        occupancy_mask=torch.zeros(system.atom_symbol.numel(), dtype=torch.bool),
        b_factor=torch.zeros(system.atom_symbol.numel(), dtype=torch.float32),
        b_factor_mask=torch.zeros(system.atom_symbol.numel(), dtype=torch.bool),
        source_symbols=(),
        residue_id=torch.full((system.component_symbol.numel(),), -1, dtype=torch.long),
        insertion_code=torch.full((system.component_symbol.numel(),), -1, dtype=torch.long),
        batch_size=[],
    )


def _experimental_fold_prediction(
    input: EsmFold2Input,
    prediction: EsmFold2ExperimentalSamplingWithConfidenceOutput,
    *,
    batch_index: int,
    sample_index: int,
) -> FoldPrediction:
    system = input.system.cpu()  # type: ignore[attr-defined]
    model_atom_mask = input.atom_index[0] >= 0
    canonical_atom_index = input.atom_index[0, model_atom_mask].detach().cpu()
    token_component_index = input.token_component_index[0].detach().cpu()
    token_atom_index = input.token_atom_index[0].detach().cpu()
    token_axis = tuple(
        (
            system.atom_site(int(atom_index))
            if int(atom_index) >= 0
            else system.component_reference(int(component_index))
        )
        for component_index, atom_index in zip(token_component_index, token_atom_index)
    )
    atom_axis = tuple(system.atom_site(int(atom_index)) for atom_index in canonical_atom_index)
    chain_axis = tuple(system.instance_id(index) for index in range(system.instance_symbol.numel()))
    confidence = prediction.confidence
    return FoldPrediction(
        structure=_experimental_fold_structure(
            input,
            prediction.sample,
            batch_index=batch_index,
            sample_index=sample_index,
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
                100.0 * float(confidence.mean_plddt[batch_index, sample_index].detach().cpu()),
                preferred="higher",
            ),
            "complex_iplddt": Score(
                100.0 * float(confidence.complex_iplddt[batch_index, sample_index].detach().cpu()),
                preferred="higher",
            ),
        },
        metrics={
            "plddt": Metric(
                values=(100.0 * confidence.plddt[batch_index, sample_index]).detach().float().cpu().numpy(),
                axis=token_axis,
            ),
            "plddt_per_atom": Metric(
                values=(100.0 * confidence.plddt_per_atom[batch_index, sample_index, model_atom_mask])
                .detach()
                .float()
                .cpu()
                .numpy(),
                axis=atom_axis,
            ),
            "plddt_ca": Metric(
                values=(100.0 * confidence.plddt_ca[batch_index, sample_index]).detach().float().cpu().numpy(),
                axis=token_axis,
            ),
        },
        pair_metrics={
            "pae": PairMetric(
                values=confidence.pae[batch_index, sample_index].detach().float().cpu().numpy(),
                row_axis=token_axis,
                column_axis=token_axis,
                unit="angstrom",
            ),
            "chain_pair_iptm": PairMetric(
                values=confidence.chain_pair_iptm[batch_index, sample_index].detach().float().cpu().numpy(),
                row_axis=chain_axis,
                column_axis=chain_axis,
            ),
        },
    )


def _stack_design_inputs(inputs: tuple[EsmFold2Input, ...]) -> EsmFold2Input:
    r"""
    Stack same-length target--binder inputs while retaining each candidate's atom padding.
    """

    max_atoms = max(value.atom_mask.shape[1] for value in inputs)

    def stack(values: tuple[torch.Tensor, ...]) -> torch.Tensor:
        return torch.cat(values, dim=0)

    def stack_optional(values: tuple[torch.Tensor | None, ...]) -> torch.Tensor | None:
        if values[0] is None:
            return None
        return torch.cat(cast(tuple[torch.Tensor, ...], values), dim=0)

    def stack_atoms(atom_values: tuple[torch.Tensor, ...]) -> torch.Tensor:
        padded_values: list[torch.Tensor] = []
        for value in atom_values:
            padding = max_atoms - value.shape[1]
            if padding:
                value = torch.cat(
                    [
                        value,
                        value.new_zeros(value.shape[0], padding, *value.shape[2:]),
                    ],
                    dim=1,
                )
            padded_values.append(value)
        return torch.cat(padded_values, dim=0)

    atom_offsets: list[int] = []
    component_offsets: list[int] = []
    atom_offset = 0
    component_offset = 0
    for value in inputs:
        atom_offsets.append(atom_offset)
        component_offsets.append(component_offset)
        atom_offset += value.system.atom_symbol.numel()
        component_offset += value.system.component_symbol.numel()

    def stack_canonical_indices(
        index_values: tuple[torch.Tensor, ...],
        offsets: list[int],
        *,
        pad_atoms: bool = False,
    ) -> torch.Tensor:
        values = []
        for indices, offset in zip(index_values, offsets, strict=True):
            indices = torch.where(indices >= 0, indices + offset, indices)
            padding = max_atoms - indices.shape[1] if pad_atoms else 0
            if padding:
                indices = torch.cat(
                    [
                        indices,
                        indices.new_full((indices.shape[0], padding), -1),
                    ],
                    dim=1,
                )
            values.append(indices)
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

    return EsmFold2Input(  # type: ignore[call-arg]
        system=MolecularSystem.batch([value.system for value in inputs]),
        token_index=stack(tuple(value.token_index for value in inputs)),
        residue_index=stack(tuple(value.residue_index for value in inputs)),
        chain_index=stack(tuple(value.chain_index for value in inputs)),
        symmetry_index=stack(tuple(value.symmetry_index for value in inputs)),
        entity_index=stack(tuple(value.entity_index for value in inputs)),
        molecule_type=stack(tuple(value.molecule_type for value in inputs)),
        residue_type_ids=stack(tuple(value.residue_type_ids for value in inputs)),
        token_bonds=stack(tuple(value.token_bonds for value in inputs)),
        token_mask=stack(tuple(value.token_mask for value in inputs)),
        reference_positions=stack_atoms(tuple(value.reference_positions for value in inputs)),
        reference_elements=stack_atoms(tuple(value.reference_elements for value in inputs)),
        reference_charges=stack_atoms(tuple(value.reference_charges for value in inputs)),
        reference_atom_name_characters=stack_atoms(tuple(value.reference_atom_name_characters for value in inputs)),
        reference_space_id=stack_atoms(tuple(value.reference_space_id for value in inputs)),
        atom_mask=stack_atoms(tuple(value.atom_mask for value in inputs)),
        atom_to_token=stack_atoms(tuple(value.atom_to_token for value in inputs)),
        atom_index=stack_canonical_indices(
            tuple(value.atom_index for value in inputs),
            atom_offsets,
            pad_atoms=True,
        ),
        token_component_index=stack_canonical_indices(
            tuple(value.token_component_index for value in inputs),
            component_offsets,
        ),
        token_atom_index=stack_canonical_indices(
            tuple(value.token_atom_index for value in inputs),
            atom_offsets,
        ),
        distogram_atom_index=stack(tuple(value.distogram_atom_index for value in inputs)),
        token_has_esmc=stack(tuple(value.token_has_esmc for value in inputs)),
        deletion_mean=stack_optional(tuple(value.deletion_mean for value in inputs)),
        msa_tokens=stack_optional(tuple(value.msa_tokens for value in inputs)),
        has_deletion=stack_optional(tuple(value.has_deletion for value in inputs)),
        deletion_value=stack_optional(tuple(value.deletion_value for value in inputs)),
        msa_mask=stack_optional(tuple(value.msa_mask for value in inputs)),
        language_model_tokens=torch.cat(language_model_tokens, dim=0),
        language_model_mask=torch.cat(language_model_masks, dim=0),
        token_to_esmc=torch.cat(token_to_esmc, dim=0),
        esmc_hidden_states=stack_optional(tuple(value.esmc_hidden_states for value in inputs)),
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
        return esmfold2_molecular_system_from_sequences(input)

    classified = classify_reference(input)
    if isinstance(classified, Path):
        resolved = resolve_polymer_input(classified)
        if isinstance(resolved, Structure):
            return resolved.system
        if isinstance(resolved, MolecularSystem):
            return resolved
        named = resolved
    else:
        named = (("protein", PolymerSequence.from_one_letter(classified.strip().upper())),)
    return esmfold2_molecular_system_from_named_sequences(named)


def _fold_msa(
    msa: MSA | MsaCollection | str | PathLike[str] | None,
    system: MolecularSystem,
) -> MSA | MsaCollection | None:
    r"""
    Read path evidence once and bind a single-entity alignment to its protein.
    """

    return resolve_msa_input(msa, system, entity_types={ENTITY_PROTEIN})


def _target_one_letter(input: str | PathLike[str] | Structure | MolecularSystem) -> str:
    r"""
    Resolve one single-chain protein target to an unambiguous one-letter sequence.
    """

    resolved = resolve_polymer_input(input) if isinstance(input, str | PathLike) else input
    if isinstance(resolved, tuple):
        if len(resolved) != 1 or resolved[0][1].polymer_type != "protein":
            raise ValueError("ESMFold2 design requires one single-chain protein target")
        return resolved[0][1].to_one_letter()
    system: MolecularSystem | None = resolved.system if isinstance(resolved, Structure) else resolved
    if system is not None:
        protein_instances = [
            instance_index
            for instance_index, entity_index in enumerate(system.instance_entity.detach().cpu().tolist())
            if int(system.entity_type[entity_index]) == ENTITY_PROTEIN
        ]
        if len(protein_instances) != 1:
            raise ValueError("ESMFold2 design requires one single-chain protein target")
        instance_index = protein_instances[0]
        component_start = int(system.instance_component_ptr[instance_index])
        component_stop = int(system.instance_component_ptr[instance_index + 1])
        return "".join(
            AMINO_ACID_THREE_TO_ONE[system.symbols[int(component_symbol)]]
            for component_symbol in system.component_symbol[component_start:component_stop].detach().cpu()
        )

    raise TypeError("ESMFold2 design requires a protein sequence or structure")


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


def _compute_esmc_masked_lm_logits(
    esmc: torch.nn.Module,
    lm_head: torch.nn.Module,
    input_token_probabilities: torch.Tensor,
    language_model_mask: torch.Tensor,
) -> torch.Tensor:
    r"""
    Compute ESMC masked-language-model logits from soft token probabilities.
    """

    esmc_model = cast(Any, esmc)
    masked_lm_head = cast(Any, lm_head)
    embed_weight = esmc_model.get_input_embeddings().weight
    language_model_embeddings = input_token_probabilities.to(embed_weight.dtype) @ embed_weight
    language_model_sequence_index = torch.zeros_like(language_model_mask, dtype=torch.long).masked_fill(
        ~language_model_mask,
        -1,
    )
    output = esmc_model(
        inputs_embeds=language_model_embeddings,
        sequence_id=language_model_sequence_index,
        return_dict=True,
    )
    logits = masked_lm_head({"last_hidden_state": output.last_hidden_state}).logits
    return logits


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
            _compute_esmc_masked_lm_logits(
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
