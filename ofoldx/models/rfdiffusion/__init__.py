r"""
RFdiffusion protein-backbone design implementation.
"""

from __future__ import annotations

from ...pipelines import PIPELINES, PipelineExecutionStep
from ..registry import MODELS
from .config import (
    RfDiffusionConfig,
    RfDiffusionDenoiserConfig,
    RfDiffusionDiffuserConfig,
    RfDiffusionPreprocessConfig,
    RfDiffusionSe3Config,
)
from .model import (
    RFDIFFUSION_ATOM14_NAMES,
    RFDIFFUSION_MASK_TOKEN_ID,
    RFDIFFUSION_RESIDUE_TYPE_INDEX,
    RFDIFFUSION_RESIDUE_TYPES,
    RFDIFFUSION_UNKNOWN_TOKEN_ID,
    RfDiffusionInput,
    RfDiffusionModel,
    RfDiffusionOutput,
    RfDiffusionPotentialInput,
    RfDiffusionSamplingOutput,
    RfDiffusionTrajectory,
    rfdiffusion_residue_token_id,
)
from .pipeline import (
    RfDiffusionDesignCandidate,
    RfDiffusionDesignResult,
    RfDiffusionPipeline,
)
from .processor import (
    RfDiffusionCoordination,
    RfDiffusionPlacements,
    RfDiffusionPointGroupSymmetry,
    RfDiffusionProcessor,
    RfDiffusionRadiusOfGyration,
    RfDiffusionSecondaryStructure,
    RfDiffusionSubstrateInteraction,
    from_contig_string,
    to_contig_string,
)

__all__ = [
    "RFDIFFUSION_ATOM14_NAMES",
    "RFDIFFUSION_MASK_TOKEN_ID",
    "RFDIFFUSION_RESIDUE_TYPES",
    "RFDIFFUSION_RESIDUE_TYPE_INDEX",
    "RFDIFFUSION_UNKNOWN_TOKEN_ID",
    "RfDiffusionConfig",
    "RfDiffusionCoordination",
    "RfDiffusionDenoiserConfig",
    "RfDiffusionDesignCandidate",
    "RfDiffusionDesignResult",
    "RfDiffusionDiffuserConfig",
    "RfDiffusionInput",
    "RfDiffusionModel",
    "RfDiffusionOutput",
    "RfDiffusionPipeline",
    "RfDiffusionPlacements",
    "RfDiffusionPointGroupSymmetry",
    "RfDiffusionPotentialInput",
    "RfDiffusionPreprocessConfig",
    "RfDiffusionProcessor",
    "RfDiffusionRadiusOfGyration",
    "RfDiffusionSamplingOutput",
    "RfDiffusionSe3Config",
    "RfDiffusionSecondaryStructure",
    "RfDiffusionSubstrateInteraction",
    "RfDiffusionTrajectory",
    "from_contig_string",
    "rfdiffusion_residue_token_id",
    "to_contig_string",
]

MODELS.register(
    "rfdiffusion",
    model=RfDiffusionModel,
    processor=RfDiffusionProcessor,
)

PIPELINES.register(
    "rfdiffusion",
    pipeline=RfDiffusionPipeline,
    executions={
        "design": (
            PipelineExecutionStep("prepare", "pipeline", RfDiffusionPipeline.prepare_design),
            PipelineExecutionStep("forward", "model", RfDiffusionModel.sample),
            PipelineExecutionStep("decode", "pipeline", RfDiffusionPipeline.decode_design),
        )
    },
)
