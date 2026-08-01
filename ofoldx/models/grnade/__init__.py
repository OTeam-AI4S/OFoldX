r"""
gRNAde RNA inverse-folding implementation.
"""

from __future__ import annotations

from ...pipelines import PIPELINES, PipelineExecutionStep
from ..registry import MODELS
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

MODELS.register(
    "grnade",
    model=GrnadeModel,
    processor=GrnadeProcessor,
)
PIPELINES.register(
    "grnade",
    pipeline=GrnadePipeline,
    executions={
        "design": (
            PipelineExecutionStep("prepare", "pipeline", GrnadePipeline.prepare_design),
            PipelineExecutionStep("forward", "model", GrnadeModel.sample),
            PipelineExecutionStep("decode", "pipeline", GrnadePipeline.decode_design),
        )
    },
)
MODELS.register(
    "grnade_v0_3_2",
    model=GrnadeModel,
    processor=GrnadeV032Processor,
)
PIPELINES.register(
    "grnade_v0_3_2",
    pipeline=GrnadeV032Pipeline,
    executions={
        "design": (
            PipelineExecutionStep("prepare", "pipeline", GrnadeV032Pipeline.prepare_design),
            PipelineExecutionStep("forward", "model", GrnadeModel.sample),
            PipelineExecutionStep("decode", "pipeline", GrnadeV032Pipeline.decode_design),
        )
    },
)
