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
from .pipeline import LigandMpnnDesignAndPackPipeline, LigandMpnnPipeline
from .processor import LigandMpnnProcessor

_MODELS.register_model(
    "ofoldx.ligandmpnn",
    model=LigandMpnnModel,
    processor=LigandMpnnProcessor,
    processor_config=LigandMpnnProcessorConfig,
    standard_tasks=(),
)
_PIPELINES.register_pipeline(
    "ofoldx.ligandmpnn",
    pipeline=LigandMpnnPipeline,
    tasks=("design",),
    architectures=("ofoldx.ligandmpnn",),
)
