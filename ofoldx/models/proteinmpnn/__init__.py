r"""
ProteinMPNN inverse-folding implementation.
"""

from __future__ import annotations

from ...pipelines.registry import PIPELINES as _PIPELINES
from ..registry import MODELS as _MODELS
from .config import ProteinMpnnConfig, ProteinMpnnProcessorConfig
from .model import (
    ProteinMpnnCaFeatures,
    ProteinMpnnDesignResult,
    ProteinMpnnInput,
    ProteinMpnnModel,
    ProteinMpnnOutput,
    ProteinMpnnSamplingOutput,
)
from .pipeline import ProteinMpnnDesignAndPackPipeline, ProteinMpnnPipeline
from .processor import ProteinMpnnProcessor

_MODELS.register_model(
    "ofoldx.proteinmpnn",
    model=ProteinMpnnModel,
    processor=ProteinMpnnProcessor,
    processor_config=ProteinMpnnProcessorConfig,
    standard_tasks=(),
)
_PIPELINES.register_pipeline(
    "ofoldx.proteinmpnn",
    pipeline=ProteinMpnnPipeline,
    tasks=("design", "score"),
    architectures=("ofoldx.proteinmpnn",),
)
