# OFoldX
# Copyright (C) 2026-Present  Valhalla Technology

# This file is part of OFoldX.

# OFoldX is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# OFoldX is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# For additional terms and clarifications, please refer to our License FAQ at:
# <https://ofoldx.valtech.ai/about/license-faq/>.

r"""
ESMFold2 structure-prediction and binder-design Pipelines.

Folding prepares one structure (or one built from polymer sequences, ``resolve_polymer_input``) through the shared
contract and decodes every sample with ``decode_coordinates`` (every atom regenerated in a new frame). Binder
design builds each target--binder complex from its one-letter sequences (``build_system``) and folds it with the
experimental pairs.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from functools import cache
from os import PathLike
from pathlib import Path, PurePosixPath
from types import EllipsisType
from typing import Any, NamedTuple, Self, cast

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import init_empty_weights
from danling.tensors import NestedTensor
from multimolecule import ProteinTokenizer
from numpy.typing import NDArray
from torch import Tensor

from ...chemistry.construction import PolymerSequence, Recipe, build_system, polymer_sequence
from ...chemistry.definitions import ComponentLibrary, configured_library
from ...data.molecular import AtomSite, ComponentRef, MonomerType
from ...data.structure import Structure
from ...data.tokenization import TOKEN_INDEX, TOKENS
from ...errors import RequestRef, fail
from ...hub import (
    PROCESSOR_METADATA,
    ResolvedRepository,
    hydrate_weights,
    load_json_object,
    load_processor,
    load_registered_model,
    model_values,
    resolve_repository,
    upload_repository,
)
from ...metrics import categorical_entropy_from_logits, resolved_probability_from_logits, sequence_isoelectric_point
from ...modules.backend import Backend
from ...msa import MSA, MsaCollection
from ...pipelines import ComponentSpec, CompositePipeline, PairedPipeline, resolve_generator
from ...pipelines.base import task_call
from ...pipelines.input import resolve_msa_input, resolve_polymer_input
from ...results import FoldPrediction, FoldResult, Metric, PairMetric, Score, floating_numpy
from ...results.values import best
from ..preparation import PreparedInput, decode_coordinates
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
    is_antibody_sequence,
)

# 20-dimensional design amino-acid space, ordered ALA..VAL.
_AMINO_ACIDS = "ARNDCQEGHILKMFPSTWYV"
_DESIGN_TOKEN_INDICES = tuple(TOKEN_INDEX[letter] for letter in _AMINO_ACIDS)
_AMINO_ACID_TO_INDEX = {letter: index for index, letter in enumerate(_AMINO_ACIDS)}
_INDEX_TO_AMINO_ACID = dict(enumerate(_AMINO_ACIDS))
_AMINO_ACID_VOCAB_SIZE = 20
_CYSTEINE_INDEX = _AMINO_ACID_TO_INDEX["C"]
_MUTABLE_TOKEN = "#"

# Binder-design optimization constants.
_LOSS_WEIGHTS = {"intra_contact": 0.5, "inter_contact": 0.5, "glob": 0.2}
_LEARNING_RATE = 0.1
_TEMPERATURE_MIN = 1e-2

# ESMC pseudo-perplexity regularizer mask-sampling constants.
_ESMC_MASK_FRACTION = 0.15
_NUM_LM_MASK_PASSES = 4
_LM_LOSS_CHUNK = 128

_FAST_CHECKPOINT = "ESMFold2-Experimental-Fast"
_FAST_CUTOFF2025_CHECKPOINT = "ESMFold2-Experimental-Fast-Cutoff2025"
_EXPERIMENTAL_CHECKPOINT = "ESMFold2-Experimental"
_EXPERIMENTAL_CUTOFF2025_CHECKPOINT = "ESMFold2-Experimental-Cutoff2025"

_FoldInput = Structure | PolymerSequence | Sequence[PolymerSequence] | str | PathLike[str]
_MsaInput = Mapping[str, MSA] | MsaCollection | str | PathLike[str] | None


def _prepare(
    processor: EsmFold2Processor, input: _FoldInput, msa: _MsaInput, library: ComponentLibrary | None
) -> PreparedInput[EsmFold2Input]:
    structure = resolve_polymer_input(input, library=library)
    pools = resolve_msa_input(msa, structure.system, monomer_types=(MonomerType.PROTEIN,))
    return processor.prepare_input(structure, msa=pools)


class EsmFold2Pipeline(PairedPipeline[EsmFold2Model, EsmFold2Processor]):
    r"""
    Structure prediction pipeline for ESMFold 2.
    """

    task = "fold"

    model_architecture = "esmfold2"

    def prepare(
        self,
        input: _FoldInput,
        *,
        msa: _MsaInput = None,
        library: ComponentLibrary | None = None,
    ) -> PreparedInput[EsmFold2Input]:
        r"""
        Prepare one structure (or polymer sequences) and its optional protein MSA for ESMFold2 folding.

        Notes:
            The reference atom axis is complete and unpadded. A component whose represented atoms lack a usable
            reference conformer is embedded from its determined graph using the processor seed, with one
            ``reference_embedded`` record. Generated model references leave canonical coordinate knownness unchanged.

            Representative atoms follow the declared convention: protein CB, then CA, then the first admitted atom;
            nucleic
            C4 for purine classes or C2 for other known classes, then C1-prime, then the first atom. Unknown nucleic
            classes
            start at C1-prime. Coordinate availability does not change this choice. The atom mapping is retained for
            feature
            gathering and decoding even when a component's biological label disagrees with its actual atoms. The
            relation bit
            records symmetric positive-edge presence and discards source relation types.
        """

        return _prepare(self.processor, input, msa, library)

    @torch.inference_mode()
    def fold(
        self,
        input: _FoldInput,
        *,
        msa: _MsaInput = None,
        library: ComponentLibrary | None = None,
        num_samples: int = 1,
        residue_type_probabilities: Tensor | None = None,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        num_recycles: int | None = None,
        num_sampling_steps: int | None = None,
        language_model_mask_probability: float | None = None,
        language_model_dropout: float | None = 0.3,
        msa_max_depth: int | None = 1024,
        msa_column_mask_rate: float = 0.1,
        msa_subsample_at_inference: bool = True,
        max_inference_sigma: float | None | EllipsisType = ...,
        noise_scale: float | None = None,
        step_scale: float | None = None,
        denoising_early_exit_rmsd: float | None = None,
    ) -> FoldResult:
        r"""
        Predict structures for one structure or polymer sequences with ESMFold 2.

        Args:
            input: The target (read or constructed; its chemistry is consumed as is), or polymer sequences, inline
                protein one-letter text or a protein FASTA path (``resolve_polymer_input``).
            msa: Optional protein MSA pools by instance, an exact plan, or an alignment path (``resolve_msa_input``).
                Configurations without MSA conditioning refuse it.
            library: The definitions sequences are built from (default the configured artifact).
            num_samples: Number of coordinate samples and returned predictions.
            residue_type_probabilities: Optional continuous residue features with
                shape ``(B, N, 72)``, aligned to the prepared token axis.
            seed: Integer seed, mutually exclusive with ``generator``.
            generator: PyTorch random generator, mutually exclusive with ``seed``.
            num_recycles: Number of recurrent trunk passes. ``None`` uses the model
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
            max_inference_sigma: Largest diffusion sigma included in sampling. Omission
                uses the configured cap; ``None`` keeps the full schedule.
            noise_scale: Sampler noise multiplier. ``None`` uses the model configuration.
            step_scale: Sampler step-size multiplier. ``None`` uses the model configuration.
            denoising_early_exit_rmsd: Aligned RMSD threshold, in angstroms, for ending
                denoising early. ``None`` disables early exit.

        Returns:
            Sampling-ordered predictions decoded onto the target (every atom regenerated in a new frame).
        """

        generator = resolve_generator(seed, generator, device=self.processor.device)
        prepared = self.prepare(input, msa=msa, library=library)
        output = self.forward(
            prepared.features,
            num_samples=num_samples,
            residue_type_probabilities=residue_type_probabilities,
            generator=generator,
            num_recycles=num_recycles,
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
        return self.postprocess(prepared, output)

    __call__ = task_call(fold)

    @torch.inference_mode()
    def forward(
        self,
        input: EsmFold2Input,
        *,
        num_samples: int = 1,
        residue_type_probabilities: Tensor | None = None,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        num_recycles: int | None = None,
        num_sampling_steps: int | None = None,
        language_model_mask_probability: float | None = None,
        language_model_dropout: float | None = 0.3,
        msa_max_depth: int | None = 1024,
        msa_column_mask_rate: float = 0.1,
        msa_subsample_at_inference: bool = True,
        max_inference_sigma: float | None | EllipsisType = ...,
        noise_scale: float | None = None,
        step_scale: float | None = None,
        denoising_early_exit_rmsd: float | None = None,
    ) -> EsmFold2SamplingWithConfidenceOutput:
        r"""Execute the configured folding sampler on one prepared input."""

        generator = resolve_generator(seed, generator, device=self.processor.device)
        return self.model.sample_with_confidence(
            input,
            num_samples=num_samples,
            residue_type_probabilities=residue_type_probabilities,
            generator=generator,
            num_recycles=num_recycles,
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

    def postprocess(
        self, prepared: PreparedInput[EsmFold2Input], output: EsmFold2SamplingWithConfidenceOutput
    ) -> FoldResult:
        r"""
        Decode one raw output without executing the model: every sample through ``decode_coordinates`` (evidence
        PREDICTED, scope every atom, new frame); confidence values on the prepared token, slot and chain axes.
        """

        token_axis, chain_axis = _axes(prepared)
        atom_axis = _atom_axis(prepared)
        chains = torch.from_numpy(_chains(prepared)).to(output.chain_pair_iptm.device)
        chain_pair_iptm = floating_numpy(output.chain_pair_iptm[0].index_select(-2, chains).index_select(-1, chains))
        chain_pair_mean_tm = floating_numpy(
            output.chain_pair_mean_tm[0].index_select(-2, chains).index_select(-1, chains)
        )
        resolved = floating_numpy(resolved_probability_from_logits(output.resolved_logits[0].detach()))
        values = {
            name: floating_numpy(getattr(output, name)[0])
            for name in ("plddt", "plddt_per_atom", "plddt_ca", "pae", "pde", "ptm", "iptm", "mean_plddt")
        }
        values["complex_iplddt"] = floating_numpy(output.complex_iplddt[0])
        predictions = []
        for index, sample in enumerate(output.coordinates[0].detach().cpu().double().numpy()):
            structure, _ = decode_coordinates(prepared, sample)
            predictions.append(
                FoldPrediction(
                    structure,
                    {
                        name: Score(float(values[name][index]), preferred="higher")
                        for name in ("ptm", "iptm", "mean_plddt", "complex_iplddt")
                    },
                    {
                        "plddt": Metric(values=values["plddt"][index], axis=token_axis),
                        "plddt_per_atom": Metric(values=values["plddt_per_atom"][index], axis=atom_axis),
                        "plddt_ca": Metric(values=values["plddt_ca"][index], axis=token_axis),
                        "resolved_probability": Metric(values=resolved[index], axis=atom_axis),
                    },
                    {
                        "pae": PairMetric(
                            values=values["pae"][index], row_axis=token_axis, column_axis=token_axis, unit="angstrom"
                        ),
                        "pde": PairMetric(
                            values=values["pde"][index], row_axis=token_axis, column_axis=token_axis, unit="angstrom"
                        ),
                        "chain_pair_iptm": PairMetric(
                            values=chain_pair_iptm[index], row_axis=chain_axis, column_axis=chain_axis
                        ),
                        "chain_pair_mean_tm": PairMetric(
                            values=chain_pair_mean_tm[index], row_axis=chain_axis, column_axis=chain_axis
                        ),
                    },
                )
            )
        return FoldResult(predictions=tuple(predictions))


class EsmFold2ExperimentalPipeline(PairedPipeline[EsmFold2ExperimentalModel, EsmFold2ExperimentalProcessor]):
    r"""
    Independently callable experimental ESMFold2 Model--Processor pair.
    """

    task = "fold"

    model_architecture = "esmfold2_experimental"

    def prepare(
        self,
        input: _FoldInput,
        *,
        msa: _MsaInput = None,
        library: ComponentLibrary | None = None,
    ) -> PreparedInput[EsmFold2Input]:
        r"""Prepare one structure (or polymer sequences) and its optional protein MSA for experimental folding."""

        return _prepare(self.processor, input, msa, library)

    @torch.inference_mode()
    def fold(
        self,
        input: _FoldInput,
        *,
        msa: _MsaInput = None,
        library: ComponentLibrary | None = None,
        num_samples: int = 1,
        residue_type_probabilities: Tensor | None = None,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        num_recycles: int | None = None,
        num_sampling_steps: int | None = None,
        language_model_mask_probability: float = 0.0,
        max_inference_sigma: float | None | EllipsisType = ...,
        noise_scale: float | None = None,
        step_scale: float | None = None,
    ) -> FoldResult:
        r"""
        Fold one structure with an independently saved experimental pair.

        Args:
            input: The target (read or constructed; its chemistry is consumed as is), or polymer sequences, inline
                protein one-letter text or a protein FASTA path (``resolve_polymer_input``).
            msa: Optional protein MSA pools by instance, an exact plan, or an alignment path (``resolve_msa_input``).
                Fast checkpoints refuse it.
            library: The definitions sequences are built from (default the configured artifact).
            num_samples: Number of coordinate samples and returned predictions.
            residue_type_probabilities: Optional continuous residue features with
                shape ``(B, N, 72)``, aligned to the prepared token axis.
            seed: Integer seed, mutually exclusive with ``generator``.
            generator: PyTorch random generator, mutually exclusive with ``seed``.
            num_recycles: Number of recurrent trunk passes. ``None`` uses the
                configured value.
            num_sampling_steps: Number of diffusion sampling steps. ``None`` uses
                the configured value.
            language_model_mask_probability: Probability of masking each nonspecial
                language-model token.
            max_inference_sigma: Largest diffusion sigma included in sampling. Omission
                uses the configured cap; ``None`` keeps the full schedule.
            noise_scale: Sampler noise multiplier. ``None`` uses the configured value.
            step_scale: Sampler step-size multiplier. ``None`` uses the configured value.

        Returns:
            Predictions decoded onto the target and, when the checkpoint has a confidence head, confidence values.
        """

        generator = resolve_generator(seed, generator, device=self.processor.device)
        prepared = self.prepare(input, msa=msa, library=library)
        output = self.forward(
            prepared.features,
            num_samples=num_samples,
            residue_type_probabilities=residue_type_probabilities,
            generator=generator,
            num_recycles=num_recycles,
            num_sampling_steps=num_sampling_steps,
            language_model_mask_probability=language_model_mask_probability,
            max_inference_sigma=max_inference_sigma,
            noise_scale=noise_scale,
            step_scale=step_scale,
        )
        return self.postprocess(prepared, output)

    __call__ = task_call(fold)

    @torch.inference_mode()
    def forward(
        self,
        input: EsmFold2Input,
        *,
        num_samples: int = 1,
        residue_type_probabilities: Tensor | None = None,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        num_recycles: int | None = None,
        num_sampling_steps: int | None = None,
        language_model_mask_probability: float = 0.0,
        max_inference_sigma: float | None | EllipsisType = ...,
        noise_scale: float | None = None,
        step_scale: float | None = None,
    ) -> EsmFold2ExperimentalSamplingOutput | EsmFold2ExperimentalSamplingWithConfidenceOutput:
        r"""Execute the configured folding sampler on one prepared input."""

        generator = resolve_generator(seed, generator, device=self.processor.device)
        sample = self.model.sample if self.model.confidence_head is None else self.model.sample_with_confidence
        return sample(
            input,
            num_recycles=num_recycles,
            num_samples=num_samples,
            residue_type_probabilities=residue_type_probabilities,
            num_sampling_steps=num_sampling_steps,
            language_model_mask_probability=language_model_mask_probability,
            max_inference_sigma=max_inference_sigma,
            noise_scale=noise_scale,
            step_scale=step_scale,
            generator=generator,
        )

    def postprocess(
        self,
        prepared: PreparedInput[EsmFold2Input],
        output: EsmFold2ExperimentalSamplingOutput | EsmFold2ExperimentalSamplingWithConfidenceOutput,
    ) -> FoldResult:
        r"""Decode one raw experimental folding output without executing the model."""

        if isinstance(output, EsmFold2ExperimentalSamplingWithConfidenceOutput):
            return FoldResult(
                predictions=tuple(
                    _experimental_fold_prediction(prepared, output, batch_index=0, sample_index=index)
                    for index in range(output.sample.coordinates.shape[1])
                )
            )
        return FoldResult(
            predictions=tuple(
                FoldPrediction(_experimental_fold_structure(prepared, output, batch_index=0, sample_index=index))
                for index in range(output.coordinates.shape[1])
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


class EsmFold2DesignPipelineInput(NamedTuple):
    r"""Canonical target/prompt and dimensions of one binder-optimization operation."""

    target_sequence: str
    prompt: str
    num_candidates: int
    num_optimization_steps: int
    is_antibody: bool
    language_model_weight: float


class EsmFold2DesignPipelineOutput(NamedTuple):
    r"""Optimized binder candidates and their aligned hero/scaling evaluations."""

    candidates: tuple[_EsmFold2InversionCandidate, ...]
    hero_critic_evaluations: tuple[tuple[EsmFold2HeroCriticEvaluation, ...], ...]
    scaling_critic_evaluations: tuple[tuple[EsmFold2ScalingCriticEvaluation, ...], ...]


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


class _Components(Mapping[str, "EsmFold2ExperimentalPipeline"]):
    def __init__(self, store: _CompositionStore) -> None:
        self.store = store

    def __iter__(self) -> Iterator[str]:
        return iter(self.store.slots)

    def __len__(self) -> int:
        return len(self.store.slots)

    def __getitem__(self, role: str) -> EsmFold2ExperimentalPipeline:
        return self.store.pin(role)


class _CompositionStore:
    """Hydrate only active children and share exact pinned external language models."""

    def __init__(
        self,
        source: ResolvedRepository,
        layout: tuple[ComponentSpec, ...],
        values: Mapping[str, Mapping[str, str]],
        *,
        device: torch.device | str | None,
        dtype: torch.dtype | None,
        backend: Backend | None,
        token: str | None,
    ) -> None:
        self.source = source
        self.slots = {slot.role: slot for slot in layout}
        if set(values) != set(self.slots):
            raise RuntimeError("ESMFold2 design repository must contain exactly its nineteen components")
        for slot in layout:
            value = values[slot.role]
            if value.get("directory") != slot.subdirectory or value.get("pipeline") != "esmfold2_experimental":
                raise RuntimeError(f"invalid ESMFold2 design component: {slot.role}")
        self.values = values
        self.device = device
        self.dtype = dtype
        self.backend = backend
        self.token = token
        self.loaded: dict[str, EsmFold2ExperimentalPipeline] = {}
        self.pinned: dict[str, EsmFold2ExperimentalPipeline] = {}
        self.backbones: dict[tuple[Any, ...], tuple[torch.nn.Module, torch.nn.Module]] = {}
        self.components = _Components(self)
        self.active = False

    def pin(self, role: str) -> EsmFold2ExperimentalPipeline:
        if self.active:
            fail("value_invalid", RequestRef("pipeline", "active"))
        if role not in self.pinned:
            self.pinned[role] = self.load(role, share=False)
            self.loaded.pop(role, None)
        return self.pinned[role]

    def load(self, role: str, *, share: bool = True) -> EsmFold2ExperimentalPipeline:
        if role in self.pinned:
            return self.pinned[role]
        if role in self.loaded:
            return self.loaded[role]
        directory = self.values[role]["directory"]
        subdirectory = (
            str(PurePosixPath(self.source.subdirectory) / directory) if self.source.subdirectory else directory
        )
        child = ResolvedRepository(
            self.source.path / directory,
            self.source.repository,
            self.source.commit,
            subdirectory,
            self.source.repo_type,
        )
        child = resolve_repository(child, allow_patterns=PROCESSOR_METADATA, token=self.token)
        registration, values = model_values(child.path)
        if registration.model is not EsmFold2ExperimentalModel:
            raise RuntimeError(f"invalid ESMFold2 design model architecture: {role}")
        config = EsmFold2ExperimentalModel.config_class(values)
        key = None
        revision = config.esmc_revision
        if share and config.esmc_repository is not None and isinstance(revision, str) and len(revision) == 40:
            key = (
                config.esmc_repository,
                revision,
                tuple(sorted(config.esmc.dict().items())),
                config.esmc_dtype,
                config.esmc_lm_head_dtype,
                str(self.device),
                self.dtype,
            )
        processor = cast(EsmFold2ExperimentalProcessor, load_processor(child, device=self.device, dtype=self.dtype))
        with init_empty_weights(include_buffers=False):
            model = EsmFold2ExperimentalModel(config)
        if key is not None and key in self.backbones:
            model._repository_esmc = self.backbones[key]
        model = load_registered_model(
            hydrate_weights(child, token=self.token), model, device=self.device, dtype=self.dtype, token=self.token
        )
        if self.backend is not None:
            model.set_backend(self.backend)
        pipeline = EsmFold2ExperimentalPipeline(model=model, processor=processor, source=child)
        self.loaded[role] = pipeline
        if key is not None and key not in self.backbones:
            self.backbones[key] = (pipeline.model.esmc, pipeline.model.esmc_lm_head)
        return pipeline

    @contextmanager
    def session(self) -> Iterator[None]:
        if self.active:
            fail("value_invalid", RequestRef("pipeline", "reentrant"))
        self.active = True
        try:
            yield
        finally:
            self.loaded.clear()
            self.backbones.clear()
            self.active = False

    @contextmanager
    def use(self, *roles: str) -> Iterator[tuple[EsmFold2ExperimentalPipeline, ...]]:
        try:
            yield tuple(self.load(role) for role in roles)
        finally:
            for role in roles:
                self.loaded.pop(role, None)

    def clear_backbones(self) -> None:
        self.backbones.clear()


class EsmFold2DesignPipeline(CompositePipeline):
    r"""
    Complete released ESMFold2 binder-design composition.

    Notes:
        The composition requires all nineteen children, including when ``use_scaling_critics=False`` skips scaling
        evaluations. Four hero roles occupy ``fast/``, ``fast_cutoff2025/``, ``experimental/``, and
        ``experimental_cutoff2025/``. The remaining roles occupy
        ``scaling/base{300m,600m,6b}_step{250k,500k,750k,1000k,1500k}/``. Each child uses a frozen ESMC trunk and
        masked-language-model head, has its own pipeline marker, and can be loaded independently from the same
        resolved repository revision. A child's ESMC weights may be stored in a separately referenced repository.

        A repository-backed workflow loads children when needed. Explicit access to a child retains that object and
        its caller-made changes; internally loaded children are released after their stage. This bounds ordinary
        execution by the active models, while explicitly retaining many children increases memory use.
        A repository-backed workflow cannot be entered again, expose a child, save, change device or dtype, or change
        backend while one of its stages is active (``value_invalid``). A workflow constructed from in-memory children
        retains their ordinary access behavior.

        Inversion uses the raw models to preserve gradients through soft binder-sequence logits. The default 150
        optimization steps use the distogram contact objective and ESMC pseudo-perplexity regularizer; each candidate
        retains its best actual low-temperature iPTM sequence and structure. Hero critics return confidence values,
        while scaling critics return structures and distogram proxies without confidence heads. Sampling overrides on
        each critic's forward stage apply inside the composition, and critic batches are decoded separately from
        single-system folding postprocessing.

        Postprocessing applies the released confidence combination, the non-antibody pI filter, and stable selection.
        Selected candidates use coherent Experimental-Cutoff2025 structures and confidence values. Skipping scaling
        critics sets their score component to zero; it does not remove the required child repositories.
    """

    task = "design"

    component_layout = (
        ComponentSpec(
            "fast",
            "fast",
            EsmFold2ExperimentalPipeline,
        ),
        ComponentSpec(
            "fast_cutoff2025",
            "fast_cutoff2025",
            EsmFold2ExperimentalPipeline,
        ),
        ComponentSpec(
            "experimental",
            "experimental",
            EsmFold2ExperimentalPipeline,
        ),
        ComponentSpec(
            "experimental_cutoff2025",
            "experimental_cutoff2025",
            EsmFold2ExperimentalPipeline,
        ),
        ComponentSpec(
            "base300m_step250k",
            "scaling/base300m_step250k",
            EsmFold2ExperimentalPipeline,
        ),
        ComponentSpec(
            "base300m_step500k",
            "scaling/base300m_step500k",
            EsmFold2ExperimentalPipeline,
        ),
        ComponentSpec(
            "base300m_step750k",
            "scaling/base300m_step750k",
            EsmFold2ExperimentalPipeline,
        ),
        ComponentSpec(
            "base300m_step1000k",
            "scaling/base300m_step1000k",
            EsmFold2ExperimentalPipeline,
        ),
        ComponentSpec(
            "base300m_step1500k",
            "scaling/base300m_step1500k",
            EsmFold2ExperimentalPipeline,
        ),
        ComponentSpec(
            "base600m_step250k",
            "scaling/base600m_step250k",
            EsmFold2ExperimentalPipeline,
        ),
        ComponentSpec(
            "base600m_step500k",
            "scaling/base600m_step500k",
            EsmFold2ExperimentalPipeline,
        ),
        ComponentSpec(
            "base600m_step750k",
            "scaling/base600m_step750k",
            EsmFold2ExperimentalPipeline,
        ),
        ComponentSpec(
            "base600m_step1000k",
            "scaling/base600m_step1000k",
            EsmFold2ExperimentalPipeline,
        ),
        ComponentSpec(
            "base600m_step1500k",
            "scaling/base600m_step1500k",
            EsmFold2ExperimentalPipeline,
        ),
        ComponentSpec(
            "base6b_step250k",
            "scaling/base6b_step250k",
            EsmFold2ExperimentalPipeline,
        ),
        ComponentSpec(
            "base6b_step500k",
            "scaling/base6b_step500k",
            EsmFold2ExperimentalPipeline,
        ),
        ComponentSpec(
            "base6b_step750k",
            "scaling/base6b_step750k",
            EsmFold2ExperimentalPipeline,
        ),
        ComponentSpec(
            "base6b_step1000k",
            "scaling/base6b_step1000k",
            EsmFold2ExperimentalPipeline,
        ),
        ComponentSpec(
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
        self._composition_store: _CompositionStore | None = None
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

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | PathLike[str] | ResolvedRepository,
        *,
        revision: str | None = None,
        token: str | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        backend: Backend | None = None,
    ) -> Self:
        r"""Load the design composition without materializing inactive children."""
        resolved = resolve_repository(
            pretrained_model_name_or_path, revision=revision, allow_patterns=("pipeline.json",), token=token
        )
        marker = load_json_object(resolved.path / "pipeline.json")
        if marker.get("pipeline") != "esmfold2_design" or not isinstance(marker.get("components"), dict):
            raise RuntimeError("invalid ESMFold2 design repository")
        instance = cls.__new__(cls)
        CompositePipeline.__init__(instance, {})
        instance._source = resolved
        instance._composition_store = _CompositionStore(
            resolved,
            cls.component_layout,
            cast(Mapping[str, Mapping[str, str]], marker["components"]),
            device=device,
            dtype=dtype,
            backend=backend,
            token=token,
        )
        return instance

    @property
    def components(self) -> Mapping[str, EsmFold2ExperimentalPipeline]:
        if self._composition_store is None:
            return cast(Mapping[str, EsmFold2ExperimentalPipeline], super().components)
        return self._composition_store.components

    def repository_components(self) -> Iterator[tuple[str, EsmFold2ExperimentalPipeline]]:
        if self._composition_store is None:
            yield from cast(Mapping[str, EsmFold2ExperimentalPipeline], super().components).items()
            return
        with self._composition_store.session():
            for slot in self.component_layout:
                with self._composition_store.use(slot.role) as (pipeline,):
                    yield slot.role, pipeline
                self._composition_store.clear_backbones()

    def to(self, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> Self:
        if self._composition_store is None:
            return super().to(device=device, dtype=dtype)
        store = self._composition_store
        with store.session():
            if device is not None:
                store.device = device
            if dtype is not None:
                store.dtype = dtype
            for component in store.pinned.values():
                component.to(device=device, dtype=dtype)
        return self

    def set_backend(self, backend: Backend) -> Self:
        if self._composition_store is None:
            return super().set_backend(backend)
        store = self._composition_store
        with store.session():
            store.backend = backend
            for component in store.pinned.values():
                component.set_backend(backend)
        return self

    def save(
        self,
        directory: str | PathLike[str],
        *,
        max_shard_size: int | str = "5GB",
        repo_id: str | None = None,
        token: str | None = None,
        private: bool = False,
        commit_message: str | None = None,
    ) -> None:
        if self._composition_store is None:
            super().save(
                directory,
                max_shard_size=max_shard_size,
                repo_id=repo_id,
                token=token,
                private=private,
                commit_message=commit_message,
            )
            return
        with self._composition_store.session():
            destination = Path(directory).expanduser()
            destination.mkdir(parents=True, exist_ok=True)
            marker = destination / "pipeline.json"
            marker.unlink(missing_ok=True)
            components = {}
            for slot in self.component_layout:
                with self._composition_store.use(slot.role) as (pipeline,):
                    pipeline.save(destination / slot.subdirectory, max_shard_size=max_shard_size)
                components[slot.role] = {"directory": slot.subdirectory, "pipeline": "esmfold2_experimental"}
                self._composition_store.clear_backbones()
            marker.write_text(
                json.dumps({"pipeline": self.id, "components": components}, indent=2) + "\n", encoding="utf-8"
            )
        if repo_id is not None:
            upload_repository(destination, repo_id, token=token, private=private, commit_message=commit_message)

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
        target: str,
        *,
        num_binder_residues: int,
        binder_prompt: str | None = None,
        num_candidates: int = 1,
        num_optimization_steps: int = 150,
        is_antibody: bool | None = None,
        use_scaling_critics: bool = True,
        language_model_weight: float | None = None,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        library: ComponentLibrary | None = None,
    ) -> EsmFold2DesignResult:
        r"""
        Run the released inversion, critic-evaluation, filtering, and selection protocol.

        Args:
            target: One-letter sequence of the single protein target chain.
            num_binder_residues: Number of residues in each designed binder.
            binder_prompt: Optional prompt of length ``num_binder_residues``. One-letter
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
            library: The definitions every target--binder complex is built from (default the configured artifact).

        Returns:
            Every generated design, its critic evaluations and scores, and the
                stable indices of at most 84 selected designs.
        """

        prepared = self.prepare(
            target,
            num_binder_residues=num_binder_residues,
            binder_prompt=binder_prompt,
            num_candidates=num_candidates,
            num_optimization_steps=num_optimization_steps,
            is_antibody=is_antibody,
            language_model_weight=language_model_weight,
        )
        output = self.forward(
            prepared, use_scaling_critics=use_scaling_critics, seed=seed, generator=generator, library=library
        )
        return self.postprocess(prepared, output)

    __call__ = task_call(design)

    def prepare(
        self,
        target: str,
        *,
        num_binder_residues: int,
        binder_prompt: str | None = None,
        num_candidates: int = 1,
        num_optimization_steps: int = 150,
        is_antibody: bool | None = None,
        language_model_weight: float | None = None,
    ) -> EsmFold2DesignPipelineInput:
        r"""Resolve the target sequence, binder prompt and optimization dimensions."""

        for name, value in (
            ("num_binder_residues", num_binder_residues),
            ("num_candidates", num_candidates),
            ("num_optimization_steps", num_optimization_steps),
        ):
            if value <= 0:
                fail("value_invalid", RequestRef(name, str(value)))
        polymer_sequence(target, MonomerType.PROTEIN)
        prompt = _MUTABLE_TOKEN * num_binder_residues if binder_prompt is None else binder_prompt
        if len(prompt) != num_binder_residues:
            fail("value_invalid", RequestRef("binder_prompt", prompt), length=num_binder_residues)
        for position, symbol in enumerate(prompt):
            if symbol != _MUTABLE_TOKEN and symbol not in _AMINO_ACID_TO_INDEX:
                fail("request_invalid_sequence", RequestRef("binder_prompt", prompt), position=position, symbol=symbol)
        is_antibody = is_antibody_sequence(prompt) if is_antibody is None else is_antibody
        coefficient = (0.05 if is_antibody else 0.15) if language_model_weight is None else language_model_weight
        return EsmFold2DesignPipelineInput(
            target_sequence=target,
            prompt=prompt,
            num_candidates=num_candidates,
            num_optimization_steps=num_optimization_steps,
            is_antibody=is_antibody,
            language_model_weight=coefficient,
        )

    def forward(
        self,
        input: EsmFold2DesignPipelineInput,
        *,
        use_scaling_critics: bool = True,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        library: ComponentLibrary | None = None,
    ) -> EsmFold2DesignPipelineOutput:
        r"""Optimize soft binder sequences and execute the released critic ensemble."""

        if self._composition_store is not None:
            return self._forward_staged(
                input, use_scaling_critics=use_scaling_critics, seed=seed, generator=generator, library=library
            )

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
        generator = resolve_generator(seed, generator, device=self.fast.processor.device)
        library = configured_library() if library is None else library
        complexes: dict[str, Structure] = {}

        def complex_of(binder: str) -> Structure:
            if binder not in complexes:
                complexes[binder] = _complex(input.target_sequence, binder, library)
            return complexes[binder]

        active_scaling_critics = scaling_critics if use_scaling_critics else ()
        with self.fast._design_mode(0.5), self.fast_cutoff2025._design_mode(0.5):
            candidates = self._run_design(
                pairs=inversion_pairs,
                complex_of=complex_of,
                num_target_tokens=len(input.target_sequence),
                prompt=input.prompt,
                num_candidates=input.num_candidates,
                num_optimization_steps=input.num_optimization_steps,
                language_model_weight=input.language_model_weight,
                generator=generator,
            )
        with ExitStack() as stack:
            for _, pipeline in hero_critics + active_scaling_critics:
                stack.enter_context(pipeline._design_mode(0.25))
            return self._evaluate_candidates(
                candidates,
                complex_of=complex_of,
                num_target_tokens=len(input.target_sequence),
                is_antibody=input.is_antibody,
                hero_critics=hero_critics,
                scaling_critics=active_scaling_critics,
                generator=generator,
            )

    def _forward_staged(
        self,
        input: EsmFold2DesignPipelineInput,
        *,
        use_scaling_critics: bool,
        seed: int | None,
        generator: torch.Generator | None,
        library: ComponentLibrary | None,
    ) -> EsmFold2DesignPipelineOutput:
        store = cast(_CompositionStore, self._composition_store)
        library = configured_library() if library is None else library
        complexes: dict[str, Structure] = {}

        def complex_of(binder: str) -> Structure:
            if binder not in complexes:
                complexes[binder] = _complex(input.target_sequence, binder, library)
            return complexes[binder]

        heroes = (
            (_FAST_CHECKPOINT, "fast"),
            (_FAST_CUTOFF2025_CHECKPOINT, "fast_cutoff2025"),
            (_EXPERIMENTAL_CHECKPOINT, "experimental"),
            (_EXPERIMENTAL_CUTOFF2025_CHECKPOINT, "experimental_cutoff2025"),
        )
        scaling = tuple(
            (f"ESMFold2-Experimental-Fast-base{size}-step{step}k", f"base{size.lower()}_step{step}k")
            for size in ("300M", "600M", "6B")
            for step in (250, 500, 750, 1000, 1500)
        )
        with store.session():
            with store.use("fast", "fast_cutoff2025") as (fast, cutoff):
                generator = resolve_generator(seed, generator, device=fast.processor.device)
                with fast._design_mode(0.5), cutoff._design_mode(0.5):
                    candidates = self._run_design(
                        pairs=((_FAST_CHECKPOINT, fast), (_FAST_CUTOFF2025_CHECKPOINT, cutoff)),
                        complex_of=complex_of,
                        num_target_tokens=len(input.target_sequence),
                        prompt=input.prompt,
                        num_candidates=input.num_candidates,
                        num_optimization_steps=input.num_optimization_steps,
                        language_model_weight=input.language_model_weight,
                        generator=generator,
                    )
            del fast, cutoff
            hero_values: list[list[EsmFold2HeroCriticEvaluation]] = [[] for _ in candidates]
            scaling_values: list[list[EsmFold2ScalingCriticEvaluation]] = [[] for _ in candidates]
            for is_hero, roles in ((True, heroes), (False, scaling if use_scaling_critics else ())):
                if not is_hero:
                    store.clear_backbones()
                for checkpoint, role in roles:
                    with store.use(role) as (pipeline,), pipeline._design_mode(0.25):
                        result = self._evaluate_candidates(
                            candidates,
                            complex_of=complex_of,
                            num_target_tokens=len(input.target_sequence),
                            is_antibody=input.is_antibody,
                            hero_critics=((checkpoint, pipeline),) if is_hero else (),
                            scaling_critics=() if is_hero else ((checkpoint, pipeline),),
                            generator=generator,
                        )
                    del pipeline
                    for index, hero_evaluations in enumerate(result.hero_critic_evaluations):
                        hero_values[index].extend(hero_evaluations)
                    for index, scaling_evaluations in enumerate(result.scaling_critic_evaluations):
                        scaling_values[index].extend(scaling_evaluations)
                    if not is_hero:
                        store.clear_backbones()
            return EsmFold2DesignPipelineOutput(
                candidates,
                tuple(tuple(values) for values in hero_values),
                tuple(tuple(values) for values in scaling_values),
            )

    def _evaluate_candidates(
        self,
        candidates: tuple[_EsmFold2InversionCandidate, ...],
        *,
        complex_of: Callable[[str], Structure],
        num_target_tokens: int,
        is_antibody: bool,
        hero_critics: tuple[tuple[str, EsmFold2ExperimentalPipeline], ...],
        scaling_critics: tuple[tuple[str, EsmFold2ExperimentalPipeline], ...],
        generator: torch.Generator | None,
    ) -> EsmFold2DesignPipelineOutput:
        hero_evaluations: list[list[EsmFold2HeroCriticEvaluation]] = [[] for _ in candidates]
        scaling_evaluations: list[list[EsmFold2ScalingCriticEvaluation]] = [[] for _ in candidates]

        def proxies(distogram: Tensor, sequence: str) -> tuple[float, float | None]:
            whole = _distogram_iptm_proxy(distogram, num_target_tokens, len(sequence))
            return whole, cdr_distogram_iptm_proxy(distogram, num_target_tokens, sequence) if is_antibody else None

        for hero, critics in ((True, hero_critics), (False, scaling_critics)):
            for checkpoint, pipeline in critics:
                prepared = tuple(pipeline.prepare(complex_of(candidate.sequence)) for candidate in candidates)
                batch = _stack_design_inputs(tuple(value.features for value in prepared))
                with torch.inference_mode():
                    output = pipeline.forward(
                        batch,
                        num_recycles=4,
                        num_samples=1,
                        num_sampling_steps=200,
                        language_model_mask_probability=0.0,
                        generator=generator,
                    )
                for index, candidate in enumerate(candidates):
                    if hero:
                        confident = cast(EsmFold2ExperimentalSamplingWithConfidenceOutput, output)
                        whole, cdr = proxies(confident.sample.distogram_logits[index], candidate.sequence)
                        prediction = _experimental_fold_prediction(
                            prepared[index], confident, batch_index=index, sample_index=0
                        )
                        hero_evaluations[index].append(EsmFold2HeroCriticEvaluation(checkpoint, prediction, whole, cdr))
                    else:
                        sample = cast(EsmFold2ExperimentalSamplingOutput, output)
                        whole, cdr = proxies(sample.distogram_logits[index], candidate.sequence)
                        structure = _experimental_fold_structure(
                            prepared[index], sample, batch_index=index, sample_index=0
                        )
                        scaling_evaluations[index].append(
                            EsmFold2ScalingCriticEvaluation(checkpoint, structure, whole, cdr)
                        )

        return EsmFold2DesignPipelineOutput(
            candidates,
            tuple(tuple(values) for values in hero_evaluations),
            tuple(tuple(values) for values in scaling_evaluations),
        )

    def postprocess(
        self,
        input: EsmFold2DesignPipelineInput,
        output: EsmFold2DesignPipelineOutput,
    ) -> EsmFold2DesignResult:
        r"""Bind coherent confidence, apply the released pI filter and select designs."""

        candidates = output.candidates
        hero_evaluations = output.hero_critic_evaluations
        scaling_evaluations = output.scaling_critic_evaluations
        is_antibody = input.is_antibody
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
            isoelectric_point = sequence_isoelectric_point(candidate.sequence)
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
        complex_of: Callable[[str], Structure],
        num_target_tokens: int,
        prompt: str,
        num_candidates: int,
        num_optimization_steps: int,
        language_model_weight: float,
        generator: torch.Generator | None,
    ) -> tuple[_EsmFold2InversionCandidate, ...]:
        device = pairs[0][1].processor.device
        num_binder_residues = len(prompt)
        logits, gradient_mask = self._initial_logits(prompt, num_candidates, generator, device)
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
            prepared = tuple(pipeline.prepare(complex_of(binder)) for binder in binder_sequences)
            batch = _stack_design_inputs(tuple(value.features for value in prepared))
            binder_probabilities = design.new_zeros(*design.shape[:-1], len(TOKENS))
            binder_probabilities[..., list(_DESIGN_TOKEN_INDICES)] = design
            residue_type_probabilities = torch.cat(
                (
                    F.one_hot(batch.residue_type_ids[:, :num_target_tokens].long(), num_classes=len(TOKENS)),
                    binder_probabilities,
                ),
                dim=1,
            )

            prediction: EsmFold2ExperimentalSamplingWithConfidenceOutput | None = None
            if temperature < 0.05:
                prediction = pipeline.model.sample_with_confidence(
                    batch,
                    residue_type_probabilities=residue_type_probabilities,
                    num_recycles=2,
                    num_samples=1,
                    num_sampling_steps=50,
                    generator=generator,
                )
                distogram = prediction.sample.distogram_logits
            else:
                sample = pipeline.model.sample(
                    batch,
                    residue_type_probabilities=residue_type_probabilities,
                    num_recycles=2,
                    num_samples=1,
                    num_sampling_steps=1,
                    generator=generator,
                )
                distogram = sample.distogram_logits

            losses = _structure_losses(distogram, num_binder_residues)
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
                            prepared[candidate_index],
                            prediction,
                            candidate_index,
                            num_target_tokens,
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
                candidate._replace(
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
        num_binder_residues = len(prompt)
        if all(symbol == _MUTABLE_TOKEN for symbol in prompt):
            logits = 0.01 * torch.randn(
                num_candidates,
                num_binder_residues,
                _AMINO_ACID_VOCAB_SIZE,
                generator=generator,
                device=device,
            )
            logits[:, :, _CYSTEINE_INDEX] = -1e6
        else:
            logits = torch.zeros(num_candidates, num_binder_residues, _AMINO_ACID_VOCAB_SIZE, device=device)
            for index, symbol in enumerate(prompt):
                if symbol == _MUTABLE_TOKEN:
                    logits[:, index, :] = 0.01 * torch.randn(
                        num_candidates,
                        _AMINO_ACID_VOCAB_SIZE,
                        generator=generator,
                        device=device,
                    )
                    logits[:, index, _CYSTEINE_INDEX] = -1e6
                else:
                    logits[:, index, _AMINO_ACID_TO_INDEX[symbol]] = 10.0

        mask = torch.ones(num_candidates, num_binder_residues, _AMINO_ACID_VOCAB_SIZE, device=device)
        for index, symbol in enumerate(prompt):
            if symbol != _MUTABLE_TOKEN:
                mask[:, index, :] = 0.0
        mask[:, :, _CYSTEINE_INDEX] = 0.0
        return logits.requires_grad_(True), mask

    def _decode_candidate(
        self,
        prepared: PreparedInput[EsmFold2Input],
        prediction: EsmFold2ExperimentalSamplingWithConfidenceOutput,
        batch_index: int,
        num_target_tokens: int,
        inversion_checkpoint: str,
        binder_sequence: str,
        iptm: float,
        final_logits: torch.Tensor,
        trajectory: tuple[EsmFold2DesignStep, ...],
    ) -> _EsmFold2InversionCandidate:
        return _EsmFold2InversionCandidate(
            sequence=binder_sequence,
            best_iptm=iptm,
            inversion_checkpoint=inversion_checkpoint,
            inversion_prediction=_experimental_fold_prediction(
                prepared, prediction, batch_index=batch_index, sample_index=0
            ),
            inversion_distogram_iptm_proxy=_distogram_iptm_proxy(
                prediction.sample.distogram_logits[batch_index], num_target_tokens, len(binder_sequence)
            ),
            final_logits=final_logits,
            trajectory=trajectory,
        )


def _complex(target: str, binder: str, library: ComponentLibrary) -> Structure:
    r"""The target--binder complex of two one-letter protein sequences (chains ``A`` and ``B``), nothing placed."""
    recipe = Recipe(
        {
            "target": polymer_sequence(target, MonomerType.PROTEIN),
            "binder": polymer_sequence(binder, MonomerType.PROTEIN),
        },
        {"A": "target", "B": "binder"},
    )
    system, _, _ = build_system(recipe, library)
    return Structure.unobserved(system)


def _owner(pointer: NDArray[np.int64]) -> NDArray[np.int64]:
    return np.repeat(np.arange(len(pointer) - 1, dtype=np.int64), np.diff(pointer))


def _chains(prepared: PreparedInput[EsmFold2Input]) -> NDArray[np.int64]:
    r"""The instances with a token, in system order (the model's chain index values)."""
    system = prepared.context.structure.system
    return np.unique(_owner(system.instance_component_ptr)[prepared.context.slots.layout.token_component])


def _axes(prepared: PreparedInput[EsmFold2Input]) -> tuple[tuple[ComponentRef | AtomSite, ...], tuple[str, ...]]:
    r"""The prepared token axis (a residue token is its component, an atom token its atom) and the chain axis."""
    system = prepared.context.structure.system
    layout = prepared.context.slots.layout
    counts = np.bincount(layout.token_component, minlength=len(system.component_symbol))
    atomic = (counts[layout.token_component] > 1) | (layout.token_class == TOKEN_INDEX["OTHER"])
    refs = system.component_refs(layout.token_component)
    names = system.text("atom", system.atom_symbol[layout.token_centre_atom]).tolist()
    axis = tuple(
        AtomSite(ref.instance, ref.component, name) if atom else ref
        for ref, name, atom in zip(refs, names, atomic.tolist())
    )
    return axis, tuple(system.text("instance", system.instance_symbol[_chains(prepared)]).tolist())


def _atom_axis(prepared: PreparedInput[EsmFold2Input]) -> tuple[AtomSite, ...]:
    r"""The prepared atom (slot) axis."""
    system = prepared.context.structure.system
    atoms = prepared.context.slots.atom
    refs = system.component_refs(_owner(system.component_atom_ptr)[atoms])
    names = system.text("atom", system.atom_symbol[atoms]).tolist()
    return tuple(AtomSite(ref.instance, ref.component, name) for ref, name in zip(refs, names))


def _experimental_fold_structure(
    prepared: PreparedInput[EsmFold2Input],
    sample: EsmFold2ExperimentalSamplingOutput,
    *,
    batch_index: int,
    sample_index: int,
) -> Structure:
    coordinates = sample.coordinates[batch_index, sample_index].detach().cpu().double().numpy()
    structure, _ = decode_coordinates(prepared, coordinates)
    return structure


def _experimental_fold_prediction(
    prepared: PreparedInput[EsmFold2Input],
    prediction: EsmFold2ExperimentalSamplingWithConfidenceOutput,
    *,
    batch_index: int,
    sample_index: int,
) -> FoldPrediction:
    token_axis, chain_axis = _axes(prepared)
    confidence = prediction.confidence
    chains = torch.from_numpy(_chains(prepared)).to(confidence.chain_pair_iptm.device)
    chain_pair_iptm = (
        confidence.chain_pair_iptm[batch_index, sample_index].index_select(-2, chains).index_select(-1, chains)
    )
    chain_pair_mean_tm = (
        confidence.chain_pair_mean_tm[batch_index, sample_index].index_select(-2, chains).index_select(-1, chains)
    )

    def value(name: str) -> Any:
        return floating_numpy(getattr(confidence, name)[batch_index, sample_index])

    return FoldPrediction(
        _experimental_fold_structure(prepared, prediction.sample, batch_index=batch_index, sample_index=sample_index),
        {
            name: Score(float(value(name)), preferred="higher")
            for name in ("ptm", "iptm", "mean_plddt", "complex_iplddt")
        },
        {
            "plddt": Metric(values=value("plddt"), axis=token_axis),
            "plddt_per_atom": Metric(values=value("plddt_per_atom"), axis=_atom_axis(prepared)),
            "plddt_ca": Metric(values=value("plddt_ca"), axis=token_axis),
        },
        {
            "pae": PairMetric(values=value("pae"), row_axis=token_axis, column_axis=token_axis, unit="angstrom"),
            "chain_pair_iptm": PairMetric(
                values=floating_numpy(chain_pair_iptm), row_axis=chain_axis, column_axis=chain_axis
            ),
            "chain_pair_mean_tm": PairMetric(
                values=floating_numpy(chain_pair_mean_tm), row_axis=chain_axis, column_axis=chain_axis
            ),
        },
    )


def _stack_design_inputs(inputs: tuple[EsmFold2Input, ...]) -> EsmFold2Input:
    r"""
    Batch same-length target--binder inputs over a ragged atom axis.

    Candidates differ by side chain, so their atom counts differ. The token axis is shared and
    stacks densely; the atom axis becomes a [`NestedTensor`][danling.NestedTensor] whose elements
    keep their own lengths, so no atom slot is padding.
    """

    if len(inputs) == 1:
        return inputs[0]

    def stack(values: tuple[torch.Tensor, ...]) -> torch.Tensor:
        return torch.cat(values, dim=0)

    def stack_optional(values: tuple[torch.Tensor | None, ...]) -> torch.Tensor | None:
        if values[0] is None:
            return None
        return torch.cat(cast(tuple[torch.Tensor, ...], values), dim=0)

    def stack_atoms(atom_values: tuple[torch.Tensor, ...]) -> NestedTensor:
        return NestedTensor([value.squeeze(0) for value in atom_values], ragged_dims=(0,))

    language_model_tokens = [cast(torch.Tensor, value.language_model_tokens) for value in inputs]
    language_model_masks = [cast(torch.Tensor, value.language_model_mask) for value in inputs]
    token_to_esmc: list[torch.Tensor] = []
    offset = 0
    for value, tokens in zip(inputs, language_model_tokens):
        mapping = cast(torch.Tensor, value.token_to_esmc)
        token_to_esmc.append(torch.where(value.token_has_esmc, mapping + offset, mapping))
        offset += tokens.numel()

    def tokens_of(name: str) -> torch.Tensor:
        return stack(tuple(getattr(value, name) for value in inputs))

    def atoms_of(name: str) -> NestedTensor:
        return stack_atoms(tuple(getattr(value, name) for value in inputs))

    return EsmFold2Input(  # type: ignore[call-arg]
        **{
            name: tokens_of(name)
            for name in (
                "token_index",
                "residue_index",
                "chain_index",
                "symmetry_index",
                "entity_index",
                "molecule_type",
                "residue_type_ids",
                "token_bonds",
                "token_representative_atom_index",
                "token_has_esmc",
            )
        },
        **{
            name: atoms_of(name)
            for name in (
                "reference_positions",
                "reference_elements",
                "reference_charges",
                "reference_atom_name_characters",
                "reference_space_id",
                "atom_to_token",
            )
        },
        **{
            name: stack_optional(tuple(getattr(value, name) for value in inputs))
            for name in (
                "deletion_mean",
                "msa_tokens",
                "has_deletion",
                "deletion_value",
                "msa_mask",
                "esmc_hidden_states",
            )
        },
        language_model_tokens=torch.cat(language_model_tokens, dim=0),
        language_model_mask=torch.cat(language_model_masks, dim=0),
        token_to_esmc=torch.cat(token_to_esmc, dim=0),
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

    vocabulary = ProteinTokenizer().get_vocab()
    amino_acid_token_indices = {amino_acid: vocabulary[amino_acid] for amino_acid in _INDEX_TO_AMINO_ACID.values()}
    amino_acid_start = min(amino_acid_token_indices.values())
    amino_acid_stop = max(amino_acid_token_indices.values()) + 1
    if amino_acid_stop - amino_acid_start != _AMINO_ACID_VOCAB_SIZE:
        raise RuntimeError("ESMC standard amino-acid vocabulary ids are not contiguous")
    language_model_order = [
        amino_acid
        for amino_acid, _ in sorted(
            amino_acid_token_indices.items(),
            key=lambda item: item[1],
        )
    ]
    permutation = tuple(
        language_model_order.index(_INDEX_TO_AMINO_ACID[index]) for index in range(_AMINO_ACID_VOCAB_SIZE)
    )
    return amino_acid_start, amino_acid_stop, permutation


def _folding_trunk_to_language_model_amino_acid_matrix(device: torch.device) -> torch.Tensor:
    r"""
    ``[20, 20]`` permutation from folding-trunk amino-acid order to ESMC LM order.
    """

    _, _, permutation = _language_model_amino_acid_layout()
    matrix = torch.zeros(_AMINO_ACID_VOCAB_SIZE, _AMINO_ACID_VOCAB_SIZE, device=device)
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
    num_mask_passes: int = _NUM_LM_MASK_PASSES,
    mask_fraction: float = _ESMC_MASK_FRACTION,
    chunk_size: int = _LM_LOSS_CHUNK,
) -> torch.Tensor:
    r"""
    Approximate ESMC pseudo-perplexity NLL of a soft designed sequence.

    ``design`` is ``[B, L, 20]`` soft amino-acid probabilities in folding-trunk order (the
    binder's per-position softmax). ``score_mask`` is ``[B, L]`` marking the mutable
    positions eligible for masking. Builds a differentiable categorical ESMC input (with a
    straight-through discretisation), samples ``num_mask_passes`` random ``mask_fraction`` masks over
    the mutable positions, and returns the per-sample mean masked-token NLL of the designed
    distribution under the frozen ESMC masked-LM head. Gradient flows back to ``design``
    through the frozen trunk, giving a differentiable sequence-naturalness prior.
    """

    device = design.device
    esmc = model.esmc
    lm_head = model.esmc_lm_head
    esmc_config = model.config.esmc
    vocab_size = esmc_config.vocab_size
    embed_dtype = esmc.get_input_embeddings().weight.dtype
    amino_acid_start, amino_acid_stop, _ = _language_model_amino_acid_layout()
    folding_trunk_to_language_model = _folding_trunk_to_language_model_amino_acid_matrix(device)

    batch_size, num_binder_residues = design.shape[0], design.shape[1]
    target_language_model_probabilities = design @ folding_trunk_to_language_model
    input_language_model_probabilities = _straight_through(
        _hard_categorical_probabilities(target_language_model_probabilities),
        target_language_model_probabilities,
    )

    input_token_probabilities = torch.zeros(
        batch_size,
        num_binder_residues + 2,
        vocab_size,
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

    mask_token = torch.zeros(vocab_size, dtype=embed_dtype, device=device)
    mask_token[esmc_config.mask_token_id] = 1

    all_masked_rows: list[torch.Tensor] = []
    all_pass_masks: list[torch.Tensor] = []
    for sample in range(batch_size):
        position_indices = score_mask[sample].nonzero(as_tuple=False).flatten()
        num_positions = int(position_indices.numel())
        if num_positions == 0:
            fail("value_invalid", RequestRef("score_mask", f"sample {sample}: no position"))
        num_masked = max(1, math.ceil(mask_fraction * num_positions))
        random_scores = torch.rand(
            (num_mask_passes, num_positions),
            device=device,
            generator=generator,
        )
        masked_offsets = random_scores.topk(num_masked, dim=-1, largest=False).indices
        pass_masks = torch.zeros((num_mask_passes, num_binder_residues), dtype=torch.bool, device=device)
        pass_masks[torch.arange(num_mask_passes, device=device)[:, None], position_indices[masked_offsets]] = True
        masked = input_token_probabilities[sample : sample + 1].repeat(num_mask_passes, 1, 1)
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
        start = sample * num_mask_passes
        stop = start + num_mask_passes
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


def _masked_min_k(x: torch.Tensor, mask: torch.Tensor, num_values: int) -> torch.Tensor:
    mask = mask.bool()
    y = torch.sort(torch.where(mask, x, float("nan")))[0]
    k_mask = (torch.arange(y.shape[-1], device=y.device) < num_values) & (~torch.isnan(y))
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
    per_residue = _masked_min_k(con_loss, mask=binder_mask, num_values=num_contacts)
    return _masked_average(per_residue, mask=chain_mask)


def _structure_losses(distogram_logits: torch.Tensor, num_binder_residues: int) -> dict[str, torch.Tensor]:
    r"""
    Weighted intra-contact + inter-contact + globularity objective (binder at suffix).
    """

    bin_distance = _get_mid_points(distogram_logits.device)
    full_len = distogram_logits.shape[1]
    is_binder = torch.ones(full_len, device=distogram_logits.device)
    is_binder[:-num_binder_residues] *= 0.0

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

    binder_disto = distogram_logits[:, -num_binder_residues:, -num_binder_residues:, :]
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
    return categorical_entropy_from_logits(dgram[..., bin_distance < cutoff])


def _distogram_iptm_proxy(distogram_logits: torch.Tensor, num_target_tokens: int, num_binder_residues: int) -> float:
    r"""
    Return an entropy-of-binding proxy for target-binder ipTM.
    """

    if distogram_logits.ndim == 4:
        distogram_logits = distogram_logits[0]
    bin_distance = _get_mid_points(distogram_logits.device)
    binder_to_target = _binding_confidence_entropy(
        distogram_logits[num_target_tokens:, :num_target_tokens, :], bin_distance, cutoff=22.0
    )
    sorted_entropies, _ = torch.sort(binder_to_target.reshape(-1))
    k = min(num_binder_residues, sorted_entropies.numel())
    mean_entropy = sorted_entropies[:k].mean()
    return float((1.0 - mean_entropy / math.log(51)).clamp(0.0, 1.0))


def cdr_distogram_iptm_proxy(distogram_logits: torch.Tensor, num_target_tokens: int, binder_sequence: str) -> float:
    r"""
    Score target contacts made by antibody CDR residues.

    Args:
        distogram_logits: Pairwise distance-bin logits with shape ``(N, N, D)`` or
            ``(B, N, N, D)``. For a four-dimensional tensor, only the first batch item
            is scored.
        num_target_tokens: Number of target tokens preceding the binder tokens.
        binder_sequence: Antibody Fv sequence aligned to the binder-token rows.

    Returns:
        An entropy-based contact confidence in the closed interval ``[0, 1]``.
    """

    if distogram_logits.ndim == 4:
        distogram_logits = distogram_logits[0]
    bin_distance = _get_mid_points(distogram_logits.device)
    cdr_indices = antibody_cdr_indices(binder_sequence)
    cdr_rows = [num_target_tokens + i for i in cdr_indices]
    cdr_to_target = _binding_confidence_entropy(
        distogram_logits[cdr_rows, :num_target_tokens, :], bin_distance, cutoff=22.0
    )
    sorted_entropies, _ = torch.sort(cdr_to_target.reshape(-1))
    k = min(len(cdr_indices), sorted_entropies.numel())
    mean_entropy = sorted_entropies[:k].mean()
    return float((1.0 - mean_entropy / math.log(51)).clamp(0.0, 1.0))


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
