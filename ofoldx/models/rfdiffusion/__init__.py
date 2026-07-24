r"""
RFdiffusion protein-backbone design implementation.
"""

from __future__ import annotations

from ...pipelines.registry import PIPELINES as _PIPELINES
from ..registry import MODELS as _MODELS
from ..registry import Task as _Task
from .config import (
    RfDiffusionConfig,
    RfDiffusionDenoiserConfig,
    RfDiffusionDiffuserConfig,
    RfDiffusionPreprocessConfig,
    RfDiffusionProcessorConfig,
    RfDiffusionSe3Config,
)
from .model import (
    RFDIFFUSION_ATOM14_NAMES,
    RFDIFFUSION_MASK_TOKEN_ID,
    RFDIFFUSION_RESIDUE_TYPE_INDEX,
    RFDIFFUSION_RESIDUE_TYPES,
    RFDIFFUSION_UNKNOWN_TOKEN_ID,
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
    rfdiffusion_residue_token_id,
)
from .pipeline import RfDiffusionPipeline
from .processor import RfDiffusionProcessor, from_contig_string, to_contig_string

_DESIGN_TASKS: tuple[_Task, ...] = ("design",)

_MODELS.register_model(
    "ofoldx.rfdiffusion",
    model=RfDiffusionModel,
    processor=RfDiffusionProcessor,
    processor_config=RfDiffusionProcessorConfig,
    standard_tasks=(),
)

_PIPELINES.register_pipeline(
    "ofoldx.rfdiffusion",
    pipeline=RfDiffusionPipeline,
    tasks=_DESIGN_TASKS,
    architectures=("ofoldx.rfdiffusion",),
)
