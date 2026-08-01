r"""
NA-MPNN mixed-polymer inverse-folding implementation.
"""

from __future__ import annotations

from ...pipelines.registry import PIPELINES as _PIPELINES
from ..registry import MODELS as _MODELS
from .config import NaMpnnConfig, NaMpnnProcessorConfig
from .model import (
    NaMpnnFeatures,
    NaMpnnInput,
    NaMpnnModel,
    NaMpnnOutput,
    NaMpnnSamplingOutput,
)
from .objectives import criterion, smoothed_loss
from .pipeline import NaMpnnPipeline
from .processor import NaMpnnProcessor

__all__ = [
    "NaMpnnConfig",
    "NaMpnnFeatures",
    "NaMpnnInput",
    "NaMpnnModel",
    "NaMpnnOutput",
    "NaMpnnPipeline",
    "NaMpnnProcessor",
    "NaMpnnProcessorConfig",
    "NaMpnnSamplingOutput",
    "criterion",
    "smoothed_loss",
]

_MODELS.register(
    "nampnn",
    model=NaMpnnModel,
    processor=NaMpnnProcessor,
)
_PIPELINES.register(
    "nampnn",
    pipeline=NaMpnnPipeline,
)
