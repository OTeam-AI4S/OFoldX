r"""
NA-MPNN mixed-polymer inverse-folding implementation.
"""

from __future__ import annotations

from ...pipelines import PIPELINES, PipelineExecutionStep
from ..registry import MODELS
from .config import NaMpnnConfig, NaMpnnProcessorConfig
from .model import (
    NaMpnnFeatureEncoder,
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
    "NaMpnnFeatureEncoder",
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

MODELS.register(
    "nampnn",
    model=NaMpnnModel,
    processor=NaMpnnProcessor,
)
PIPELINES.register(
    "nampnn",
    pipeline=NaMpnnPipeline,
    executions={
        "design": (
            PipelineExecutionStep("prepare", "pipeline", NaMpnnPipeline.prepare_design),
            PipelineExecutionStep("forward", "model", NaMpnnModel.design),
            PipelineExecutionStep("decode", "pipeline", NaMpnnPipeline.decode_design),
        ),
        "score": (
            PipelineExecutionStep("prepare", "pipeline", NaMpnnPipeline.prepare_probabilities),
            PipelineExecutionStep("forward", "model", NaMpnnModel.score),
            PipelineExecutionStep("decode", "pipeline", NaMpnnPipeline.decode_score),
        ),
    },
)
