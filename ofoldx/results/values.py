r"""Canonical scientific results returned by OFoldX tasks."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, TypeAlias

import numpy as np

from ..data.molecular import AtomSite, ComponentReference, MolecularSystem
from ..data.structure import Structure

MetricReference: TypeAlias = str | ComponentReference | AtomSite


@dataclass(eq=False)
class Score:
    r"""One named scalar value and the canonical subtarget it describes."""

    value: float
    unit: str | None = None
    preferred: Literal["higher", "lower"] | None = None
    scope: frozenset[MetricReference] | None = None
    evaluations: np.ndarray | None = None


@dataclass(eq=False)
class Metric:
    r"""One numerical value aligned to one canonical axis."""

    values: np.ndarray
    axis: tuple[MetricReference, ...]
    unit: str | None = None


@dataclass(eq=False)
class PairMetric:
    r"""One numerical value aligned to two canonical axes."""

    values: np.ndarray
    row_axis: tuple[MetricReference, ...]
    column_axis: tuple[MetricReference, ...]
    unit: str | None = None


@dataclass(eq=False)
class FoldPrediction:
    r"""One folding prediction and its aligned scientific values."""

    structure: Structure
    scores: dict[str, Score] = field(default_factory=dict)
    metrics: dict[str, Metric] = field(default_factory=dict)
    pair_metrics: dict[str, PairMetric] = field(default_factory=dict)


@dataclass(eq=False)
class FoldResult:
    r"""One generation-ordered collection of folding predictions."""

    predictions: tuple[FoldPrediction, ...]

    def best(self, score: str) -> FoldPrediction:
        r"""Return the earliest best prediction under one explicit score."""

        values = tuple(prediction.scores[score] for prediction in self.predictions)
        directions = {value.preferred for value in values}
        if directions not in ({"higher"}, {"lower"}):
            raise ValueError(f"score {score!r} must consistently prefer higher or lower values")
        if len({value.unit for value in values}) != 1:
            raise ValueError(f"score {score!r} must use one identical unit")
        if len({value.scope for value in values}) != 1:
            raise ValueError(f"score {score!r} must use one identical scope")
        if any(math.isnan(value.value) for value in values):
            raise ValueError(f"score {score!r} contains NaN and cannot be ranked")
        if directions == {"higher"}:
            return max(self.predictions, key=lambda prediction: prediction.scores[score].value)
        return min(self.predictions, key=lambda prediction: prediction.scores[score].value)


@dataclass(eq=False)
class DesignCandidate:
    r"""One coordinate-free fixed-length design and its scientific values."""

    system: MolecularSystem
    scores: dict[str, Score] = field(default_factory=dict)


@dataclass(eq=False)
class DesignResult:
    r"""One generation-ordered collection of fixed-length designs."""

    candidates: tuple[DesignCandidate, ...]

    def best(self, score: str) -> DesignCandidate:
        r"""Return the earliest best candidate under one explicit score."""

        values = tuple(candidate.scores[score] for candidate in self.candidates)
        directions = {value.preferred for value in values}
        if directions not in ({"higher"}, {"lower"}):
            raise ValueError(f"score {score!r} must consistently prefer higher or lower values")
        if len({value.unit for value in values}) != 1:
            raise ValueError(f"score {score!r} must use one identical unit")
        if len({value.scope for value in values}) != 1:
            raise ValueError(f"score {score!r} must use one identical scope")
        if any(math.isnan(value.value) for value in values):
            raise ValueError(f"score {score!r} contains NaN and cannot be ranked")
        if directions == {"higher"}:
            return max(self.candidates, key=lambda candidate: candidate.scores[score].value)
        return min(self.candidates, key=lambda candidate: candidate.scores[score].value)


@dataclass(eq=False)
class ScoreResult:
    r"""One coordinate-conditioned target and its scientific values."""

    target: Structure
    scores: dict[str, Score] = field(default_factory=dict)
    metrics: dict[str, Metric] = field(default_factory=dict)


@dataclass(eq=False)
class PackPrediction:
    r"""One packed structure and the metrics produced for that sample."""

    structure: Structure
    metrics: dict[str, Metric] = field(default_factory=dict)


@dataclass(eq=False)
class PackResult:
    r"""One generation-ordered collection of side-chain packing predictions."""

    predictions: tuple[PackPrediction, ...]
