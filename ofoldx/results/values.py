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
Scientific result values returned by OFoldX tasks.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol, TypeAlias, TypeVar

import numpy as np
import torch
from torch import Tensor

from ..data.molecular import AtomSite, ComponentRef
from ..data.structure import Structure

#: Molecular entity, component occurrence, or atom occurrence addressed by a metric.
MetricReference: TypeAlias = str | ComponentRef | AtomSite


def floating_numpy(value: Tensor) -> np.ndarray:
    r"""Represent floating tensor observations as a NumPy array."""
    return value.detach().to(device="cpu", dtype=torch.promote_types(value.dtype, torch.float32)).numpy()


@dataclass(eq=False)
class Score:
    r"""
    One named scalar value and the subtarget it describes.

    Attributes:
        value: Scalar score value.
        unit: Physical or statistical unit, or ``None`` for a dimensionless score.
        preferred: Direction used to rank comparable results, or ``None`` when the score is not
            rankable.
        scope: Components or atoms summarized by the score. ``None`` denotes the
            whole result.
        evaluations: Optional raw values whose aggregate is reported by ``value``. Their leading
            axis preserves evaluation order.
    """

    value: float
    unit: str | None = None
    preferred: Literal["higher", "lower"] | None = None
    scope: frozenset[MetricReference] | None = None
    evaluations: np.ndarray | None = None


class Scored(Protocol):
    r"""
    Structural contract for a result item carrying named scores.
    """

    @property
    def scores(self) -> Mapping[str, Score]:
        r"""
        Return scores keyed by their stable scientific names.

        Returns:
            A mapping from stable score names to scalar score records.
        """

        ...


_ScoredT = TypeVar("_ScoredT", bound=Scored)


def best(values: Sequence[_ScoredT], score: str) -> _ScoredT:
    r"""
    Return the earliest best item under a named score.

    Args:
        values: Candidate values in tie-breaking order.
        score: Comparable score name present on every candidate.

    Returns:
        The candidate selected according to ``Score.preferred``. Input order breaks
            ties.

    Raises:
        KeyError: If a candidate does not contain ``score``.
        ValueError: If ``values`` is empty, scores are not consistently rankable, units
            or scopes differ, or every score is NaN.
    """

    scores = tuple(value.scores[score] for value in values)
    directions = {value.preferred for value in scores}
    if directions not in ({"higher"}, {"lower"}):
        raise ValueError(f"score {score!r} must consistently prefer higher or lower values")
    if len({value.unit for value in scores}) != 1:
        raise ValueError(f"score {score!r} must use one identical unit")
    if len({value.scope for value in scores}) != 1:
        raise ValueError(f"score {score!r} must use one identical scope")
    # A NaN score reports an undefined scope, not a bad candidate: an interface score is undefined for
    # a monomer because it has no interface. Such a candidate is unrankable under this score alone, so
    # it is skipped rather than aborting the ranking; only an all-undefined score has no answer.
    rankable = tuple(value for value in values if not math.isnan(value.scores[score].value))
    if not rankable:
        raise ValueError(f"score {score!r} is NaN for every candidate and cannot be ranked")

    def score_value(value: _ScoredT) -> float:
        return value.scores[score].value

    return max(rankable, key=score_value) if directions == {"higher"} else min(rankable, key=score_value)


@dataclass(eq=False)
class Metric:
    r"""
    Numerical values aligned to one molecular axis.

    The leading dimension of ``values`` corresponds to ``axis``.

    Attributes:
        values: Numerical array with shape ``(len(axis), ...)``.
        axis: Component or atom reference for each leading-axis value.
        unit: Physical or statistical unit, or ``None`` for dimensionless values.
    """

    values: np.ndarray
    axis: tuple[MetricReference, ...]
    unit: str | None = None


@dataclass(eq=False)
class PairMetric:
    r"""
    Numerical values aligned to two molecular axes.

    The first two dimensions of ``values`` correspond to ``row_axis`` and
    ``column_axis``.

    Attributes:
        values: Numerical array with shape
            ``(len(row_axis), len(column_axis), ...)``.
        row_axis: Component or atom references for rows.
        column_axis: Component or atom references for columns.
        unit: Physical or statistical unit, or ``None`` for dimensionless values.
    """

    values: np.ndarray
    row_axis: tuple[MetricReference, ...]
    column_axis: tuple[MetricReference, ...]
    unit: str | None = None


@dataclass(eq=False)
class FoldPrediction:
    r"""
    One folding prediction and its aligned scientific values.

    Attributes:
        structure: Predicted structure.
        scores: Scalar scores keyed by stable scientific names.
        metrics: One-axis metrics keyed by stable scientific names.
        pair_metrics: Two-axis metrics keyed by stable scientific names.
    """

    structure: Structure
    scores: dict[str, Score] = field(default_factory=dict)
    metrics: dict[str, Metric] = field(default_factory=dict)
    pair_metrics: dict[str, PairMetric] = field(default_factory=dict)


@dataclass(eq=False)
class FoldResult:
    r"""
    One generation-ordered collection of folding predictions.

    Attributes:
        predictions: Predictions in generation order.
    """

    predictions: tuple[FoldPrediction, ...]

    def best_by(self, score: str) -> FoldPrediction:
        r"""
        Return the earliest best prediction under one explicit score.

        Args:
            score: Comparable score name present on every prediction.

        Returns:
            The best prediction, with generation order breaking ties.
        """

        return best(self.predictions, score)


@dataclass(eq=False)
class DesignCandidate:
    r"""
    One fixed-length target design and its scientific values.

    Attributes:
        target: Designed chemistry with available guide coordinates. New atoms
            may have unavailable coordinates. Retained observations preserve
            their own provenance; replacement guide geometry does not assert
            experimental observation metadata.
        scores: Scalar scores keyed by stable scientific names.
    """

    target: Structure
    scores: dict[str, Score] = field(default_factory=dict)


@dataclass(eq=False)
class DesignResult:
    r"""
    One generation-ordered collection of fixed-length designs.

    Attributes:
        candidates: Designed systems in generation order.
    """

    candidates: tuple[DesignCandidate, ...]

    def best_by(self, score: str) -> DesignCandidate:
        r"""
        Return the earliest best candidate under one explicit score.

        Args:
            score: Comparable score name present on every candidate.

        Returns:
            The best candidate, with generation order breaking ties.
        """

        return best(self.candidates, score)


@dataclass(eq=False)
class ScoreResult:
    r"""
    One coordinate-conditioned target and its scientific values.

    Attributes:
        target: Structure that was scored.
        scores: Scalar scores keyed by stable scientific names.
        metrics: One-axis metrics keyed by stable scientific names.
    """

    target: Structure
    scores: dict[str, Score] = field(default_factory=dict)
    metrics: dict[str, Metric] = field(default_factory=dict)


@dataclass(eq=False)
class PackPrediction:
    r"""
    One packed structure and the metrics produced for that sample.

    Attributes:
        structure: Structure with packed side-chain coordinates.
        metrics: One-axis metrics keyed by stable scientific names.
    """

    structure: Structure
    metrics: dict[str, Metric] = field(default_factory=dict)


@dataclass(eq=False)
class PackResult:
    r"""
    One generation-ordered collection of side-chain packing predictions.

    Attributes:
        predictions: Packed structures in sampling order.
    """

    predictions: tuple[PackPrediction, ...]


@dataclass(eq=False)
class DesignAndPackResult:
    r"""Sequence designs and the side-chain packing results aligned to them."""

    designs: DesignResult
    packs: tuple[PackResult, ...]

    @property
    def candidates(self) -> tuple[DesignCandidate, ...]:
        r"""Return the sequence candidates aligned one-to-one with ``packs``."""

        return self.designs.candidates
