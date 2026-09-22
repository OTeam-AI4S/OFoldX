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
Sequence and structural diversity reduction over generated candidates.

This module contains Vendi-score aggregation, deduplication, greedy-selection, and Foldseek
boundary operations shared by real model families. It does not own metric
computation, candidate filtering, ranking, or protocol selection.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from ..data.structure import Structure
from ..metrics import diversity as diversity_metrics
from ..metrics import sequence as sequence_metrics
from ..metrics import structure as structure_metrics


@dataclass(frozen=True)
class FoldseekMatch:
    r"""
    One Foldseek reference match used by the shared novelty axis.

    Attributes:
        query_index: Zero-based index into the query structure sequence.
        query_tm_score: TM score normalized by query length.
        target_tm_score: TM score normalized by reference length.
    """

    query_index: int
    query_tm_score: float
    target_tm_score: float


FoldseekNoveltyOperation: TypeAlias = Callable[[Sequence[Structure]], Sequence[FoldseekMatch]]


@dataclass(frozen=True)
class DiversityMetrics:
    r"""
    Candidate metrics that condition the shared structural Vendi scores.

    All-atom and backbone RMSD/lDDT values are kept distinct.  Design iPTM and
    minimum interaction PAE are confidence values shared by both metric scopes.
    """

    rmsd: float
    design_rmsd: float
    target_aligned_design_rmsd: float
    lddt_intra_design: float | None
    bb_rmsd: float
    bb_design_rmsd: float
    bb_target_aligned_design_rmsd: float
    bb_lddt_intra_design: float | None
    design_iptm: float
    min_interaction_pae: float


@dataclass(frozen=True, eq=False)
class DiversitySample:
    r"""
    One target-labelled design used by the shared diversity reductions.
    """

    target_id: str
    sequence: str
    original_coordinates: NDArray
    refolded_coordinates: NDArray
    metrics: DiversityMetrics


@dataclass(frozen=True)
class ConditionedVendiScores:
    r"""
    Structural Vendi scores after the shared candidate-quality conditions.
    """

    rmsd_below_2_5: float
    design_rmsd_below_2: float
    design_rmsd_below_4: float
    target_aligned_design_rmsd_below_2_5: float
    design_lddt_at_least_0_60: float | None
    design_lddt_at_least_0_65: float | None
    design_lddt_at_least_0_70: float | None
    design_lddt_at_least_0_75: float | None
    design_lddt_at_least_0_80: float | None
    design_lddt_at_least_0_85: float | None
    design_lddt_at_least_0_90: float | None
    design_iptm_at_least_0_80: float
    design_iptm_at_least_0_70: float
    design_iptm_at_least_0_60: float
    design_iptm_at_least_0_50: float
    min_interaction_pae_at_most_1_5: float
    min_interaction_pae_at_most_2: float
    min_interaction_pae_at_most_2_5: float
    min_interaction_pae_at_most_3: float
    min_interaction_pae_at_most_4: float
    min_interaction_pae_at_most_5: float


@dataclass(frozen=True)
class StructureVendiScores:
    r"""
    Shared base and conditionally filtered structural Vendi scores.
    """

    vendi_tm_fixed: float
    vendi_tm_align: float
    all_atom: ConditionedVendiScores | None = None
    backbone: ConditionedVendiScores | None = None


@dataclass(frozen=True)
class DiversityAggregate:
    r"""
    Structure and optional sequence diversity for one candidate collection.
    """

    structure: StructureVendiScores
    vendi_seq_sim: float | None
    num_structures: int
    num_sequences: int


@dataclass(frozen=True)
class DiversityResult:
    r"""
    Shared original, refolded, and per-target diversity aggregates.
    """

    original: DiversityAggregate
    refolded: DiversityAggregate
    per_target_original: dict[str, DiversityAggregate]
    per_target_refolded: dict[str, DiversityAggregate]


def deduplicate_sequences(sequences: Sequence[str]) -> tuple[int, ...]:
    r"""
    Return indices of the earliest occurrence of every designed sequence.
    """

    seen: set[str] = set()
    indices: list[int] = []
    for index, sequence in enumerate(sequences):
        if sequence not in seen:
            seen.add(sequence)
            indices.append(index)
    return tuple(indices)


def greedy_diversity_selection(
    sequences: Sequence[str],
    quality: Sequence[float],
    *,
    budget: int = 30,
    alpha: float = 0.001,
    size_buckets: Sequence[tuple[int, int, int]] = (),
    size_values: Sequence[int] | None = None,
) -> tuple[int, ...]:
    r"""
    Select the shared quality--sequence-diversity candidate set.

    Each size bucket is ``(minimum_inclusive, maximum_exclusive, budget)``;
    the first matching bucket limits every selection, including the first.
    Returned indices are in their original stable order. Selection may contain
    fewer than ``budget`` candidates when the eligible buckets are exhausted.
    ``size_values`` supplies each candidate's actual bucket measure. By default,
    size is its number of residues, excluding colon chain delimiters.
    """

    quality_array = np.asarray(quality, dtype=np.float64)
    if not size_buckets and len(quality_array) <= budget:
        return tuple(range(len(quality_array)))

    selected: list[int] = []
    bucket_counts = np.zeros(len(size_buckets), dtype=np.int64)

    def bucket_index(index: int) -> int | None:
        size_value = len(sequences[index].replace(":", "")) if size_values is None else size_values[index]
        for position, (minimum, maximum, _) in enumerate(size_buckets):
            if minimum <= size_value < maximum:
                return position
        return None

    remaining = set(range(len(quality_array)))
    while len(selected) < budget:
        best_index = None
        best_gain = -float("inf")
        for index in sorted(remaining):
            bucket = bucket_index(index)
            if bucket is not None and bucket_counts[bucket] >= size_buckets[bucket][2]:
                continue
            gain = quality_array[index]
            if selected:
                maximum_similarity = max(
                    sequence_metrics.sequence_alignment_score(sequences[index], sequences[chosen])
                    for chosen in selected
                )
                gain = (1.0 - alpha) * quality_array[index] + alpha * (1.0 - maximum_similarity)
            if gain > best_gain:
                best_gain = gain
                best_index = index
        if best_index is None:
            break
        selected.append(best_index)
        remaining.remove(best_index)
        selected_bucket = bucket_index(best_index)
        if selected_bucket is not None:
            bucket_counts[selected_bucket] += 1
    return tuple(sorted(selected))


def structure_vendi_scores(
    coordinates: Sequence[NDArray],
    metrics: Sequence[DiversityMetrics] | None = None,
    *,
    all_atom: bool = True,
    backbone: bool = False,
) -> StructureVendiScores:
    r"""
    Return Vendi scores from normalized PSD-projected structural TM kernels.

    ``metrics`` enables the shared RMSD, lDDT, design-iPTM, and minimum-PAE
    conditions.  The same structural similarity matrix is filtered separately
    by all-atom and backbone candidate metrics.
    """

    rmsd_aligned = np.eye(len(coordinates), dtype=np.float64)
    homology_aligned = np.eye(len(coordinates), dtype=np.float64)
    for first in range(len(coordinates)):
        for second in range(first + 1, len(coordinates)):
            rmsd_score, homology_score = structure_metrics.structure_tm_scores(
                coordinates[first],
                coordinates[second],
            )
            rmsd_aligned[first, second] = rmsd_aligned[second, first] = rmsd_score
            homology_aligned[first, second] = homology_score
            homology_aligned[second, first] = homology_score
    rmsd_aligned = diversity_metrics.normalized_psd_projection(rmsd_aligned)
    homology_aligned = diversity_metrics.normalized_psd_projection(homology_aligned)
    return StructureVendiScores(
        vendi_tm_fixed=diversity_metrics.vendi_score(homology_aligned),
        vendi_tm_align=diversity_metrics.vendi_score(rmsd_aligned),
        all_atom=(
            _conditioned_vendi_scores(homology_aligned, metrics, backbone=False)
            if metrics is not None and all_atom
            else None
        ),
        backbone=(
            _conditioned_vendi_scores(homology_aligned, metrics, backbone=True)
            if metrics is not None and backbone
            else None
        ),
    )


def analyze_diversity(
    samples: Sequence[DiversitySample],
    *,
    all_atom: bool = True,
    backbone: bool = True,
) -> DiversityResult:
    r"""
    Aggregate shared original, refolded, and per-target diversity values.

    Sequence diversity belongs to the original candidate collection; refolding
    changes coordinates but not sequences.  Global values use the first
    generation-ordered design for each target; per-target values use every
    design for that target.  Structures shorter than three design residues are
    excluded from structural Vendi values.
    """

    sample_values = tuple(samples)
    grouped: dict[str, list[DiversitySample]] = {}
    for sample in sample_values:
        grouped.setdefault(sample.target_id, []).append(sample)
    representative_samples = tuple(target_samples[0] for target_samples in grouped.values())
    return DiversityResult(
        original=_diversity_aggregate(
            representative_samples,
            refolded=False,
            include_sequence=True,
            all_atom=all_atom,
            backbone=backbone,
        ),
        refolded=_diversity_aggregate(
            representative_samples,
            refolded=True,
            include_sequence=False,
            all_atom=all_atom,
            backbone=backbone,
        ),
        per_target_original={
            target_id: _diversity_aggregate(
                target_samples,
                refolded=False,
                include_sequence=True,
                all_atom=all_atom,
                backbone=backbone,
            )
            for target_id, target_samples in grouped.items()
        },
        per_target_refolded={
            target_id: _diversity_aggregate(
                target_samples,
                refolded=True,
                include_sequence=False,
                all_atom=all_atom,
                backbone=backbone,
            )
            for target_id, target_samples in grouped.items()
        },
    )


def foldseek_novelty(
    structures: Sequence[Structure],
    operation: FoldseekNoveltyOperation,
) -> NDArray[np.float64]:
    r"""
    Run an explicit in-memory Foldseek boundary for shared novelty values.

    The supplied operation owns any Foldseek repository, process, or transport
    details.  This function retains the shared scientific reduction: average
    query- and target-normalized TM scores, then take the maximum per query,
    filling unmatched queries with zero.
    """

    novelty = np.zeros(len(structures), dtype=np.float64)
    for match in operation(structures):
        score = (match.query_tm_score + match.target_tm_score) / 2.0
        novelty[match.query_index] = max(novelty[match.query_index], score)
    return novelty


def _diversity_aggregate(
    samples: Sequence[DiversitySample],
    *,
    refolded: bool,
    include_sequence: bool,
    all_atom: bool,
    backbone: bool,
) -> DiversityAggregate:
    structural_samples = tuple(
        sample
        for sample in samples
        if len(sample.refolded_coordinates if refolded else sample.original_coordinates) >= 3
    )
    coordinates = tuple(
        sample.refolded_coordinates if refolded else sample.original_coordinates for sample in structural_samples
    )
    return DiversityAggregate(
        structure=structure_vendi_scores(
            coordinates,
            tuple(sample.metrics for sample in structural_samples),
            all_atom=all_atom,
            backbone=backbone,
        ),
        vendi_seq_sim=(
            diversity_metrics.sequence_vendi_score(tuple(sample.sequence for sample in samples))
            if include_sequence
            else None
        ),
        num_structures=len(structural_samples),
        num_sequences=len(samples) if include_sequence else 0,
    )


def _conditioned_vendi_scores(
    similarity: NDArray,
    metrics: Sequence[DiversityMetrics],
    *,
    backbone: bool,
) -> ConditionedVendiScores:
    if backbone:
        rmsd = np.array([metric.bb_rmsd for metric in metrics])
        design_rmsd = np.array([metric.bb_design_rmsd for metric in metrics])
        target_aligned_rmsd = np.array([metric.bb_target_aligned_design_rmsd for metric in metrics])
        design_lddt = tuple(metric.bb_lddt_intra_design for metric in metrics)
    else:
        rmsd = np.array([metric.rmsd for metric in metrics])
        design_rmsd = np.array([metric.design_rmsd for metric in metrics])
        target_aligned_rmsd = np.array([metric.target_aligned_design_rmsd for metric in metrics])
        design_lddt = tuple(metric.lddt_intra_design for metric in metrics)
    design_iptm = np.array([metric.design_iptm for metric in metrics])
    min_interaction_pae = np.array([metric.min_interaction_pae for metric in metrics])

    lddt_scores: tuple[float, ...] | None = None
    if all(value is not None for value in design_lddt):
        lddt = np.array([float(value) for value in design_lddt if value is not None])
        lddt_scores = tuple(
            _subset_vendi_score(similarity, lddt >= threshold)
            for threshold in (0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90)
        )

    return ConditionedVendiScores(
        rmsd_below_2_5=_subset_vendi_score(similarity, rmsd <= 2.5),
        design_rmsd_below_2=_subset_vendi_score(similarity, design_rmsd <= 2.0),
        design_rmsd_below_4=_subset_vendi_score(similarity, design_rmsd <= 4.0),
        target_aligned_design_rmsd_below_2_5=_subset_vendi_score(
            similarity,
            target_aligned_rmsd <= 2.5,
        ),
        design_lddt_at_least_0_60=None if lddt_scores is None else lddt_scores[0],
        design_lddt_at_least_0_65=None if lddt_scores is None else lddt_scores[1],
        design_lddt_at_least_0_70=None if lddt_scores is None else lddt_scores[2],
        design_lddt_at_least_0_75=None if lddt_scores is None else lddt_scores[3],
        design_lddt_at_least_0_80=None if lddt_scores is None else lddt_scores[4],
        design_lddt_at_least_0_85=None if lddt_scores is None else lddt_scores[5],
        design_lddt_at_least_0_90=None if lddt_scores is None else lddt_scores[6],
        design_iptm_at_least_0_80=_subset_vendi_score(
            similarity,
            design_iptm >= 0.8,
        ),
        design_iptm_at_least_0_70=_subset_vendi_score(
            similarity,
            design_iptm >= 0.7,
        ),
        design_iptm_at_least_0_60=_subset_vendi_score(
            similarity,
            design_iptm >= 0.6,
        ),
        design_iptm_at_least_0_50=_subset_vendi_score(
            similarity,
            design_iptm >= 0.5,
        ),
        min_interaction_pae_at_most_1_5=_subset_vendi_score(
            similarity,
            min_interaction_pae <= 1.5,
        ),
        min_interaction_pae_at_most_2=_subset_vendi_score(
            similarity,
            min_interaction_pae <= 2.0,
        ),
        min_interaction_pae_at_most_2_5=_subset_vendi_score(
            similarity,
            min_interaction_pae <= 2.5,
        ),
        min_interaction_pae_at_most_3=_subset_vendi_score(
            similarity,
            min_interaction_pae <= 3.0,
        ),
        min_interaction_pae_at_most_4=_subset_vendi_score(
            similarity,
            min_interaction_pae <= 4.0,
        ),
        min_interaction_pae_at_most_5=_subset_vendi_score(
            similarity,
            min_interaction_pae <= 5.0,
        ),
    )


def _subset_vendi_score(
    similarity: NDArray,
    mask: NDArray[np.bool_],
) -> float:
    return diversity_metrics.vendi_score(similarity[mask][:, mask]) if bool(np.any(mask)) else 0.0
