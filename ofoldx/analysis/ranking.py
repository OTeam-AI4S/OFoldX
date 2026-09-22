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
Generic candidate filtering and ranking primitives.

This module contains named-metric filter and worst-rank operations shared by
real model families. It does not own metric computation, diversity reduction,
or protocol selection.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

AnalysisValue: TypeAlias = float | int | bool | str


@dataclass(frozen=True)
class CandidateFilter:
    r"""
    One explicit candidate filter.

    Attributes:
        feature: Named scalar candidate score.
        threshold: Inclusive threshold.
        lower_is_better: Whether values at or below the threshold pass.
    """

    feature: str
    threshold: float
    lower_is_better: bool


@dataclass(frozen=True)
class CandidateRanking:
    r"""
    Stable ranking values aligned to input candidates.

    Attributes:
        order: Zero-based input indices in best-first order.
        rank: One-based final rank for every input row.
        quality: Shared rank-derived quality for every input row.
        filters_passed: Number of individual filters passed by every row.
        passes_all_filters: Whether every filter passed for every row.
    """

    order: tuple[int, ...]
    rank: NDArray[np.int64]
    quality: NDArray[np.float64]
    filters_passed: NDArray[np.int64]
    passes_all_filters: NDArray[np.bool_]


def candidate_filter_results(
    row: Mapping[str, AnalysisValue],
    filters: Sequence[CandidateFilter],
) -> dict[str, bool]:
    r"""
    Return every named inclusive filter outcome for one candidate.
    """

    return {
        f"pass_{candidate_filter.feature}_filter": (
            float(row[candidate_filter.feature]) <= candidate_filter.threshold
            if candidate_filter.lower_is_better
            else float(row[candidate_filter.feature]) >= candidate_filter.threshold
        )
        for candidate_filter in filters
    }


def rank_candidates(
    rows: Sequence[Mapping[str, AnalysisValue]],
    *,
    filters: Sequence[CandidateFilter],
    metric_weights: Mapping[str, float],
    tie_breaker: str,
) -> CandidateRanking:
    r"""
    Filter and rank candidates with the shared worst-rank rule.

    Every metric is min-ranked descending on
    ``(number of individual filters passed, metric value)`` and divided by its
    inverse-importance weight.  The worst scaled metric rank determines a dense
    rank group; ``tie_breaker`` then produces the stable final order.
    """

    named_filter_results = [candidate_filter_results(row, filters) for row in rows]
    filter_passes = np.array(
        [
            [results[f"pass_{candidate_filter.feature}_filter"] for candidate_filter in filters]
            for results in named_filter_results
        ],
        dtype=np.bool_,
    )
    filters_passed: NDArray[np.int64] = np.asarray(
        np.sum(filter_passes, axis=1, dtype=np.int64),
        dtype=np.int64,
    )
    passes_all_filters = np.all(filter_passes, axis=1)

    scaled_ranks: list[NDArray] = []
    for metric, inverse_importance in metric_weights.items():
        values = np.array([float(row[metric]) for row in rows], dtype=np.float64)
        metric_rank = _minimum_descending_rank(filters_passed, values)
        scaled_ranks.append(metric_rank / inverse_importance)
    worst_rank = np.max(np.stack(scaled_ranks), axis=0)
    dense_rank = np.searchsorted(np.unique(np.sort(worst_rank)), worst_rank) + 1
    tie_values = np.array([float(row[tie_breaker]) for row in rows], dtype=np.float64)
    order = tuple(sorted(range(len(rows)), key=lambda index: (dense_rank[index], -tie_values[index], index)))

    final_rank = np.empty(len(rows), dtype=np.int64)
    for index, row_index in enumerate(order, start=1):
        final_rank[row_index] = index
    if len(rows) == 1:
        quality = np.ones(1, dtype=np.float64)
    else:
        quality = 1.0 - (final_rank - 1) / (len(rows) - 1)
    return CandidateRanking(order, final_rank, quality, filters_passed, passes_all_filters)


def _minimum_descending_rank(primary: NDArray, secondary: NDArray) -> NDArray[np.int64]:
    order = sorted(range(len(primary)), key=lambda index: (-primary[index], -secondary[index], index))
    rank = np.empty(len(primary), dtype=np.int64)
    previous: tuple[object, object] | None = None
    current_rank = 0
    for position, index in enumerate(order, start=1):
        key = (primary[index], secondary[index])
        if key != previous:
            current_rank = position
            previous = key
        rank[index] = current_rank
    return rank
