r"""
Boltz-1 structure-prediction implementation.
"""

from __future__ import annotations

from ...pipelines import PIPELINES, PipelineExecutionStep
from ..registry import MODELS
from .config import (
    Boltz1ConfidenceConfig,
    Boltz1Config,
    Boltz1DenoiserConfig,
    Boltz1InputEmbeddingsConfig,
    Boltz1SamplerConfig,
    Boltz1TrunkConfig,
)
from .model import (
    Boltz1ConfidenceOutput,
    Boltz1Input,
    Boltz1Model,
    Boltz1Output,
    Boltz1Pocket,
    Boltz1Representations,
    Boltz1SamplingOutput,
    Boltz1SamplingWithConfidenceOutput,
    Boltz1StructureTarget,
)
from .pipeline import Boltz1Pipeline
from .processor import Boltz1Processor

__all__ = [
    "Boltz1ConfidenceConfig",
    "Boltz1ConfidenceOutput",
    "Boltz1Config",
    "Boltz1DenoiserConfig",
    "Boltz1Input",
    "Boltz1InputEmbeddingsConfig",
    "Boltz1Model",
    "Boltz1Output",
    "Boltz1Pipeline",
    "Boltz1Pocket",
    "Boltz1Processor",
    "Boltz1Representations",
    "Boltz1SamplerConfig",
    "Boltz1SamplingOutput",
    "Boltz1SamplingWithConfidenceOutput",
    "Boltz1StructureTarget",
    "Boltz1TrunkConfig",
]

MODELS.register(
    "boltz1",
    model=Boltz1Model,
    processor=Boltz1Processor,
)
PIPELINES.register(
    "boltz1",
    pipeline=Boltz1Pipeline,
    executions={
        "fold": (
            PipelineExecutionStep("prepare", "pipeline", Boltz1Pipeline.prepare_fold),
            PipelineExecutionStep("forward", "model", Boltz1Model.sample_with_confidence),
            PipelineExecutionStep("decode", "pipeline", Boltz1Pipeline.decode_fold),
        )
    },
)
