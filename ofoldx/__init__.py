r"""
Open biomolecular structure models.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._version import __version__ as __version__  # noqa: F401
    from ._version import version
    from .api import pack, pipeline, score
    from .artifacts import save_model
    from .auto import AutoModel, AutoProcessor
    from .data import MSA, Hotspot, MolecularSystem, PolymerSequence, Structure, SymbolicBond, Template
    from .model import Model
    from .modules.backend import Backend
    from .processor import Processor
    from .results import (
        AutoregressiveProbabilityResult,
        DesignAndPackResult,
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
        Scored,
        ScoreResult,
        best,
    )
    from .runners import Runner

__all__ = [
    "AutoModel",
    "AutoProcessor",
    "AutoregressiveProbabilityResult",
    "Backend",
    "DesignAndPackResult",
    "DesignCandidate",
    "DesignResult",
    "FoldPrediction",
    "FoldResult",
    "Hotspot",
    "MSA",
    "Metric",
    "MetricReference",
    "Model",
    "MolecularSystem",
    "PackPrediction",
    "PackResult",
    "PairMetric",
    "PolymerSequence",
    "Processor",
    "Runner",
    "Score",
    "ScoreResult",
    "Scored",
    "Structure",
    "SymbolicBond",
    "Template",
    "best",
    "pack",
    "pipeline",
    "save_model",
    "score",
    "version",
]


_EXPORTS = {
    "__version__": ("._version", "__version__"),
    "version": ("._version", "version"),
    "pack": (".api", "pack"),
    "pipeline": (".api", "pipeline"),
    "score": (".api", "score"),
    "save_model": (".artifacts", "save_model"),
    "AutoModel": (".auto", "AutoModel"),
    "AutoProcessor": (".auto", "AutoProcessor"),
    "MSA": (".data", "MSA"),
    "Hotspot": (".data", "Hotspot"),
    "MolecularSystem": (".data", "MolecularSystem"),
    "PolymerSequence": (".data", "PolymerSequence"),
    "Structure": (".data", "Structure"),
    "SymbolicBond": (".data", "SymbolicBond"),
    "Template": (".data", "Template"),
    "Model": (".model", "Model"),
    "Backend": (".modules.backend", "Backend"),
    "Processor": (".processor", "Processor"),
    "AutoregressiveProbabilityResult": (".results", "AutoregressiveProbabilityResult"),
    "DesignAndPackResult": (".results", "DesignAndPackResult"),
    "DesignCandidate": (".results", "DesignCandidate"),
    "DesignResult": (".results", "DesignResult"),
    "FoldPrediction": (".results", "FoldPrediction"),
    "FoldResult": (".results", "FoldResult"),
    "Metric": (".results", "Metric"),
    "MetricReference": (".results", "MetricReference"),
    "PackPrediction": (".results", "PackPrediction"),
    "PackResult": (".results", "PackResult"),
    "PairMetric": (".results", "PairMetric"),
    "Score": (".results", "Score"),
    "Scored": (".results", "Scored"),
    "ScoreResult": (".results", "ScoreResult"),
    "best": (".results", "best"),
    "Runner": (".runners", "Runner"),
}


def __getattr__(name):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | _EXPORTS.keys())
