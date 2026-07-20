r"""OFoldX's public model, task, Pipeline, and Result API."""

from __future__ import annotations

from ._version import __version__, version
from .api import design, fold, pack, pipeline, score
from .auto import AutoModel, AutoProcessor
from .pipelines import DesignPipeline, FoldPipeline, PackPipeline, ScorePipeline
from .results import (
    AffinityResult,
    DesignCandidate,
    DesignResult,
    FoldPrediction,
    FoldResult,
    Metric,
    MetricReference,
    PackPrediction,
    PackResult,
    PairMetric,
    Score,
    ScoreResult,
)
from .runners import Runner
