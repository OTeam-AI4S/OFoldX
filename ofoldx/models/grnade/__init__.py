r"""
gRNAde RNA inverse-folding implementation.
"""

from __future__ import annotations

from ...pipelines.registry import PIPELINES as _PIPELINES
from ..registry import MODELS as _MODELS
from .config import GrnadeConfig, GrnadeV032ProcessorConfig
from .model import (
    GRNADE_COMPONENT_IDS,
    GRNADE_TOKEN_BY_COMPONENT,
    GRNADE_UNKNOWN_TOKEN,
    GrnadeInput,
    GrnadeModel,
    GrnadeOutput,
    GrnadeSamplingOutput,
)
from .objectives import criterion
from .pipeline import GrnadePipeline, GrnadeV032Pipeline
from .processor import GrnadeProcessor, GrnadeV032Processor

__all__ = [
    "GRNADE_COMPONENT_IDS",
    "GRNADE_TOKEN_BY_COMPONENT",
    "GRNADE_UNKNOWN_TOKEN",
    "GrnadeConfig",
    "GrnadeInput",
    "GrnadeModel",
    "GrnadeOutput",
    "GrnadePipeline",
    "GrnadeProcessor",
    "GrnadeSamplingOutput",
    "GrnadeV032Pipeline",
    "GrnadeV032Processor",
    "GrnadeV032ProcessorConfig",
    "criterion",
]

_MODELS.register(
    "grnade",
    model=GrnadeModel,
    processor=GrnadeProcessor,
)
_PIPELINES.register(
    "grnade",
    pipeline=GrnadePipeline,
)
_MODELS.register(
    "grnade_v0_3_2",
    model=GrnadeModel,
    processor=GrnadeV032Processor,
)
_PIPELINES.register(
    "grnade_v0_3_2",
    pipeline=GrnadeV032Pipeline,
)
