r"""
LigandMPNN protein-sequence design implementation.
"""

from __future__ import annotations

from ...pipelines import PIPELINES, PipelineExecutionStep
from ..registry import MODELS
from .config import LigandMpnnConfig, LigandMpnnProcessorConfig
from .model import (
    LigandMpnnFeatureEncoder,
    LigandMpnnInput,
    LigandMpnnModel,
    LigandMpnnOutput,
    LigandMpnnSamplingOutput,
)
from .objectives import criterion
from .pipeline import LigandMpnnDesignAndPackPipeline, LigandMpnnPipeline
from .processor import LigandMpnnProcessor

__all__ = [
    "LigandMpnnConfig",
    "LigandMpnnDesignAndPackPipeline",
    "LigandMpnnFeatureEncoder",
    "LigandMpnnInput",
    "LigandMpnnModel",
    "LigandMpnnOutput",
    "LigandMpnnPipeline",
    "LigandMpnnProcessor",
    "LigandMpnnProcessorConfig",
    "LigandMpnnSamplingOutput",
    "criterion",
]

MODELS.register(
    "ligandmpnn",
    model=LigandMpnnModel,
    processor=LigandMpnnProcessor,
)
PIPELINES.register(
    "ligandmpnn",
    pipeline=LigandMpnnPipeline,
    executions={
        "design": (
            PipelineExecutionStep("prepare", "pipeline", LigandMpnnPipeline.prepare_design),
            PipelineExecutionStep("forward", "model", LigandMpnnModel.design),
            PipelineExecutionStep("decode", "pipeline", LigandMpnnPipeline.decode_design),
        ),
        "score": (
            PipelineExecutionStep("prepare", "pipeline", LigandMpnnPipeline.prepare_score),
            PipelineExecutionStep("forward", "model", LigandMpnnModel.score),
            PipelineExecutionStep("decode", "pipeline", LigandMpnnPipeline.decode_score),
        ),
    },
)
PIPELINES.register(
    "ligandmpnn_design_and_pack",
    pipeline=LigandMpnnDesignAndPackPipeline,
    executions={"design": (PipelineExecutionStep("execute", "pipeline", LigandMpnnDesignAndPackPipeline.design),)},
)
