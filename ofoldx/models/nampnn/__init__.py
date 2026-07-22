r"""NA-MPNN mixed-polymer inverse-folding implementation."""

from __future__ import annotations

from ...pipelines.registry import PIPELINES
from ..registry import MODELS
from .config import NaMpnnConfig, NaMpnnProcessorConfig
from .model import NaMpnnInput, NaMpnnModel
from .pipeline import NaMpnnPipeline
from .processor import NaMpnnProcessor

MODELS.register_model(
    "ofoldx.nampnn",
    model=NaMpnnModel,
    processor=NaMpnnProcessor,
    processor_config=NaMpnnProcessorConfig,
    standard_tasks=(),
)
PIPELINES.register_pipeline(
    "ofoldx.nampnn",
    pipeline=NaMpnnPipeline,
    tasks=("design", "score"),
    architectures=("ofoldx.nampnn",),
)
