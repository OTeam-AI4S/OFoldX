r"""
OFoldX public API.

Tier 0 -- verbs: [`fold`][ofoldx.fold] / [`design`][ofoldx.design] / [`score`][ofoldx.score] /
[`pack`][ofoldx.pack], plus [`pipeline`][ofoldx.pipeline] to load a task pipeline.
Tier 2 -- builders: [`AutoModel`][ofoldx.AutoModel] / [`AutoConfig`][ofoldx.AutoConfig] /
[`AutoProcessor`][ofoldx.AutoProcessor]. Training lives in [`ofoldx.training`][ofoldx.training].
"""

from __future__ import annotations

from ._version import __version__, version
from .models import AutoConfig, AutoModel
from .pipelines import (
    BindingAffinityPipeline,
    DesignPipeline,
    Pipeline,
    SideChainPackingPipeline,
    StructurePredictionPipeline,
)
from .processors import AutoProcessor
from .tasks import design, fold, pack, pipeline, register_default_model, score

__all__ = [
    "AutoConfig",
    "AutoModel",
    "AutoProcessor",
    "BindingAffinityPipeline",
    "DesignPipeline",
    "Pipeline",
    "SideChainPackingPipeline",
    "StructurePredictionPipeline",
    "__version__",
    "design",
    "fold",
    "pack",
    "pipeline",
    "register_default_model",
    "score",
    "version",
]
