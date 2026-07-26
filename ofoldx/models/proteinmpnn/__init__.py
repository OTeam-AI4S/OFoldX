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
from .objectives import nll_loss, smoothed_loss
from .pipeline import ProteinMpnnDesignAndPackPipeline, ProteinMpnnPipeline
from .processor import ProteinMpnnProcessor

_MODELS.register(
    "proteinmpnn",
    model=ProteinMpnnModel,
    processor=ProteinMpnnProcessor,
)
_PIPELINES.register(
    "proteinmpnn",
    pipeline=ProteinMpnnPipeline,
)
