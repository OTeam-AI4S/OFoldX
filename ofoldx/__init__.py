r"""
Public models, processors, task pipelines, and scientific results.
"""

from __future__ import annotations

from ._version import __version__, version
from .api import TASKS, Task, design, fold, pack, pipeline, score
from .artifacts import save_model
from .auto import AutoModel, AutoProcessor
from .data.conditions import Contact, Hotspot
from .data.evidence import MSA, Template
from .data.molecular import MolecularSystem, PolymerSequence
from .data.structure import Structure
from .model import Model
from .modules.backend import Backend
from .processor import Processor
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
    Scored,
    best,
)


def __getattr__(name: str):
    if name == "Runner":
        from .runners import Runner

        return Runner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
