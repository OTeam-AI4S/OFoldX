r"""
LigandMPNN protein-sequence design implementation.
"""

from __future__ import annotations

from ...pipelines.registry import PIPELINES as _PIPELINES
from ..registry import MODELS as _MODELS
from .config import LigandMpnnConfig, LigandMpnnProcessorConfig
from .model import (
    LigandMpnnFeatures,
    LigandMpnnInput,
    LigandMpnnModel,
    LigandMpnnOutput,
    LigandMpnnSamplingOutput,
)
from .objectives import criterion
from .pipeline import LigandMpnnDesignAndPackPipeline, LigandMpnnPipeline
from .processor import LigandMpnnProcessor

__all__ = [
    "LigandMpnnConfig",
    "LigandMpnnDesignAndPackPipeline",
    "LigandMpnnFeatures",
    "LigandMpnnInput",
    "LigandMpnnModel",
    "LigandMpnnOutput",
    "LigandMpnnPipeline",
    "LigandMpnnProcessor",
    "LigandMpnnProcessorConfig",
    "LigandMpnnSamplingOutput",
    "criterion",
]

_MODELS.register(
    "ligandmpnn",
    model=LigandMpnnModel,
    processor=LigandMpnnProcessor,
)
_PIPELINES.register(
    "ligandmpnn",
    pipeline=LigandMpnnPipeline,
)
_PIPELINES.register(
    "ligandmpnn_design_and_pack",
    pipeline=LigandMpnnDesignAndPackPipeline,
)
