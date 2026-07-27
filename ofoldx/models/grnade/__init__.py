r"""
gRNAde RNA inverse-folding implementation.
"""

from __future__ import annotations

from ...pipelines.registry import PIPELINES as _PIPELINES
from ..registry import MODELS as _MODELS
from .config import GrnadeConfig, GrnadeProcessorConfig
from .model import (
    GRNADE_COMPONENT_IDS,
    GRNADE_TOKEN_BY_COMPONENT,
    GRNADE_UNKNOWN_TOKEN,
    GrnadeInput,
    GrnadeModel,
    GrnadeOutput,
    GrnadeSamplingOutput,
)
from .pipeline import GrnadeDesignResult, GrnadePipeline
from .processor import GrnadeProcessor

_MODELS.register(
    "grnade",
    model=GrnadeModel,
    processor=GrnadeProcessor,
)
_PIPELINES.register(
    "grnade",
    pipeline=GrnadePipeline,
)
