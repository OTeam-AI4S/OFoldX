r"""
PxDesign protein-structure design implementation.
"""

from __future__ import annotations

from ...pipelines import PIPELINES, PipelineExecutionStep
from ..registry import MODELS
from .config import (
    PxDesignConditionEmbeddingsConfig,
    PxDesignConfig,
    PxDesignDenoiserConfig,
    PxDesignSamplerConfig,
)
from .model import (
    PxDesignInput,
    PxDesignModel,
    PxDesignSamplingOutput,
)
from .pipeline import (
    PxDesignBackbonePipeline,
    PxDesignBackboneResult,
    PxDesignCandidate,
    PxDesignFilterResult,
    PxDesignPipeline,
    PxDesignResult,
)
from .processor import PxDesignProcessor, TargetChain, from_binder, to_binder

__all__ = [
    "PxDesignBackbonePipeline",
    "PxDesignBackboneResult",
    "PxDesignCandidate",
    "PxDesignConditionEmbeddingsConfig",
    "PxDesignConfig",
    "PxDesignDenoiserConfig",
    "PxDesignFilterResult",
    "PxDesignInput",
    "PxDesignModel",
    "PxDesignSamplingOutput",
    "PxDesignPipeline",
    "PxDesignProcessor",
    "PxDesignResult",
    "PxDesignSamplerConfig",
    "TargetChain",
    "from_binder",
    "to_binder",
]

MODELS.register(
    "pxdesign",
    model=PxDesignModel,
    processor=PxDesignProcessor,
)
PIPELINES.register(
    "pxdesign_backbone",
    pipeline=PxDesignBackbonePipeline,
    executions={
        "design": (
            PipelineExecutionStep("prepare", "pipeline", PxDesignBackbonePipeline.prepare_design),
            PipelineExecutionStep("forward", "model", PxDesignModel.sample),
            PipelineExecutionStep("decode", "pipeline", PxDesignBackbonePipeline.decode_design),
        )
    },
)
PIPELINES.register(
    "pxdesign",
    pipeline=PxDesignPipeline,
    executions={"design": (PipelineExecutionStep("execute", "pipeline", PxDesignPipeline.design),)},
)
