r"""
ProteinMPNN inverse-folding implementation.
"""

from __future__ import annotations

from ...pipelines import PIPELINES, PipelineExecutionStep
from ..registry import MODELS
from .config import ProteinMpnnConfig, ProteinMpnnProcessorConfig
from .model import (
    ProteinMpnnCaFeatureEncoder,
    ProteinMpnnDesignOutput,
    ProteinMpnnFeatureEncoder,
    ProteinMpnnInput,
    ProteinMpnnModel,
    ProteinMpnnOutput,
    ProteinMpnnSamplingOutput,
)
from .objectives import criterion
from .pipeline import ProteinMpnnDesignAndPackPipeline, ProteinMpnnDesignResult, ProteinMpnnPipeline
from .processor import ProteinMpnnProcessor

__all__ = [
    "ProteinMpnnCaFeatureEncoder",
    "ProteinMpnnConfig",
    "ProteinMpnnDesignAndPackPipeline",
    "ProteinMpnnDesignOutput",
    "ProteinMpnnDesignResult",
    "ProteinMpnnFeatureEncoder",
    "ProteinMpnnInput",
    "ProteinMpnnModel",
    "ProteinMpnnOutput",
    "ProteinMpnnPipeline",
    "ProteinMpnnProcessor",
    "ProteinMpnnProcessorConfig",
    "ProteinMpnnSamplingOutput",
    "criterion",
]

MODELS.register(
    "proteinmpnn",
    model=ProteinMpnnModel,
    processor=ProteinMpnnProcessor,
)
PIPELINES.register(
    "proteinmpnn",
    pipeline=ProteinMpnnPipeline,
    executions={
        "design": (
            PipelineExecutionStep("prepare", "pipeline", ProteinMpnnPipeline.prepare_design),
            PipelineExecutionStep("forward", "model", ProteinMpnnModel.design),
            PipelineExecutionStep("decode", "pipeline", ProteinMpnnPipeline.decode_design),
        ),
        "score": (
            PipelineExecutionStep("prepare", "pipeline", ProteinMpnnPipeline.prepare_score),
            PipelineExecutionStep("forward", "model", ProteinMpnnModel.score),
            PipelineExecutionStep("decode", "pipeline", ProteinMpnnPipeline.decode_score),
        ),
    },
)
PIPELINES.register(
    "proteinmpnn_design_and_pack",
    pipeline=ProteinMpnnDesignAndPackPipeline,
    executions={"design": (PipelineExecutionStep("execute", "pipeline", ProteinMpnnDesignAndPackPipeline.design),)},
)
