r"""OFoldX's public model, task, Pipeline, and Result API."""

from __future__ import annotations

from ._version import __version__, version
from .api import design, fold, pack, pipeline, score
from .artifacts import save_model
from .auto import AutoModel, AutoProcessor
from .data.conditions import Contact, Hotspot
from .data.evidence import MSA, Template
from .data.molecular import MolecularSystem, PolymerSequence
from .data.structure import Structure
from .model import Model, ModelBackend
from .models import TASKS, Task
from .pipelines import DesignPipeline, FoldPipeline, PackPipeline, ScorePipeline
from .results import (
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


def __getattr__(name: str):
    r"""Lazily expose the training ``Runner``.

    ``Runner`` pulls the training stack (danling -> torchmetrics, several seconds), which the
    ``fold``/``design``/``score``/``pack`` inference surface never needs; importing it lazily keeps
    ``import ofoldx`` cheap for inference users. ``ofoldx.Runner`` still resolves on first access.
    """

    if name == "Runner":
        from .runners import Runner

        return Runner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
