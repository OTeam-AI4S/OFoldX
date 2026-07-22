r"""RFdiffusion protein-backbone design implementation."""

from __future__ import annotations

from ...pipelines.registry import PIPELINES
from ..registry import MODELS, Task
from .config import (
    RfDiffusionConfig,
    RfDiffusionDenoiserConfig,
    RfDiffusionDiffuserConfig,
    RfDiffusionPreprocessConfig,
    RfDiffusionProcessorConfig,
    RfDiffusionSe3Config,
)
from .model import (
    RfDiffusionConditionEvaluation,
    RfDiffusionCoordination,
    RfDiffusionDesignCandidate,
    RfDiffusionDesignResult,
    RfDiffusionDistance,
    RfDiffusionInput,
    RfDiffusionLayout,
    RfDiffusionModel,
    RfDiffusionOutput,
    RfDiffusionPlacements,
    RfDiffusionPointGroupSymmetry,
    RfDiffusionRadiusOfGyration,
    RfDiffusionReferenceCoordinates,
    RfDiffusionSecondaryStructure,
    RfDiffusionSubstrateInteraction,
)
from .pipeline import RfDiffusionPipeline
from .processor import RfDiffusionProcessor

_DESIGN_TASKS: tuple[Task, ...] = ("design",)

MODELS.register_model(
    "ofoldx.rfdiffusion",
    model=RfDiffusionModel,
    processor=RfDiffusionProcessor,
    processor_config=RfDiffusionProcessorConfig,
    standard_tasks=(),
)

# Coordinate-bearing results require the concrete design pipeline.
PIPELINES.register_pipeline(
    "ofoldx.rfdiffusion",
    pipeline=RfDiffusionPipeline,
    tasks=_DESIGN_TASKS,
    architectures=("ofoldx.rfdiffusion",),
)
