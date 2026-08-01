r"""
MPNN side-chain packing implementation.
"""

from __future__ import annotations

from ...pipelines import PIPELINES, PipelineExecutionStep
from ..registry import MODELS
from .config import MpnnPackerConfig, MpnnPackerProcessorConfig
from .density import structure_with_chi_density
from .model import (
    MpnnPackerFeatureEncoder,
    MpnnPackerInput,
    MpnnPackerModel,
    MpnnPackerOutput,
    MpnnPackerSamplingOutput,
)
from .pipeline import MpnnPackerPipeline
from .processor import MpnnPackerProcessor

__all__ = [
    "MpnnPackerConfig",
    "MpnnPackerFeatureEncoder",
    "MpnnPackerInput",
    "MpnnPackerModel",
    "MpnnPackerOutput",
    "MpnnPackerPipeline",
    "MpnnPackerProcessor",
    "MpnnPackerProcessorConfig",
    "MpnnPackerSamplingOutput",
    "structure_with_chi_density",
]

MODELS.register(
    "mpnn_packer",
    model=MpnnPackerModel,
    processor=MpnnPackerProcessor,
)
PIPELINES.register(
    "mpnn_packer",
    pipeline=MpnnPackerPipeline,
    executions={
        "pack": (
            PipelineExecutionStep("prepare", "pipeline", MpnnPackerPipeline.prepare_pack),
            PipelineExecutionStep("forward", "pipeline", MpnnPackerPipeline.sample_pack),
            PipelineExecutionStep("decode", "pipeline", MpnnPackerPipeline.decode_pack),
        )
    },
)
