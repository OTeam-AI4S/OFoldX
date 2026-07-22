r"""RFdiffusion protein-backbone design implementation."""

from __future__ import annotations

from ...pipelines.registry import PIPELINES
from ..registry import MODELS
from .config import RFdiffusionConfig, RFdiffusionProcessorConfig
from .model import (
    RFdiffusionConditionEvaluation,
    RFdiffusionDesignCandidate,
    RFdiffusionDesignResult,
    RFdiffusionInput,
    RFdiffusionModel,
    RFdiffusionOutput,
)
from .pipeline import RFdiffusionPipeline
from .processor import RFdiffusionProcessor

MODELS.register_model(
    "ofoldx.rfdiffusion",
    model=RFdiffusionModel,
    processor=RFdiffusionProcessor,
    processor_config=RFdiffusionProcessorConfig,
    standard_tasks=(),
)

# Coordinate-bearing results require the concrete design pipeline.
PIPELINES.register_pipeline(
    "ofoldx.rfdiffusion",
    pipeline=RFdiffusionPipeline,
    tasks=("design",),
    architectures=("ofoldx.rfdiffusion",),
)
