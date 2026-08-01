r"""
Boltz-2 folding and affinity implementations.
"""

from __future__ import annotations

from ...pipelines import PIPELINES, PipelineExecutionStep
from ..registry import MODELS
from .chemistry import standardize_boltz2_affinity_smiles
from .config import (
    Boltz2AffinityConfig,
    Boltz2AffinityHeadConfig,
    Boltz2ConfidenceConfig,
    Boltz2Config,
    Boltz2DenoiserConfig,
    Boltz2InputEmbeddingsConfig,
    Boltz2SamplerConfig,
    Boltz2TemplateConfig,
    Boltz2TrunkConfig,
)
from .model import (
    Boltz2AffinityModel,
    Boltz2AffinityOutput,
    Boltz2AffinityScoreOutput,
    Boltz2ConfidenceOutput,
    Boltz2Contact,
    Boltz2Input,
    Boltz2Model,
    Boltz2Output,
    Boltz2Pocket,
    Boltz2Representations,
    Boltz2SamplingOutput,
    Boltz2StructureTarget,
    Boltz2Template,
)
from .pipeline import Boltz2AffinityPipeline, Boltz2AffinityResult, Boltz2AffinityScorePipeline, Boltz2Pipeline
from .processor import Boltz2AffinityProcessor, Boltz2Processor

__all__ = [
    "Boltz2AffinityConfig",
    "Boltz2AffinityHeadConfig",
    "Boltz2AffinityModel",
    "Boltz2AffinityOutput",
    "Boltz2AffinityPipeline",
    "Boltz2AffinityProcessor",
    "Boltz2AffinityResult",
    "Boltz2AffinityScoreOutput",
    "Boltz2AffinityScorePipeline",
    "Boltz2ConfidenceConfig",
    "Boltz2ConfidenceOutput",
    "Boltz2Config",
    "Boltz2Contact",
    "Boltz2DenoiserConfig",
    "Boltz2Input",
    "Boltz2InputEmbeddingsConfig",
    "Boltz2Model",
    "Boltz2Output",
    "Boltz2Pipeline",
    "Boltz2Pocket",
    "Boltz2Processor",
    "Boltz2Representations",
    "Boltz2SamplerConfig",
    "Boltz2SamplingOutput",
    "Boltz2StructureTarget",
    "Boltz2Template",
    "Boltz2TemplateConfig",
    "Boltz2TrunkConfig",
    "standardize_boltz2_affinity_smiles",
]

MODELS.register(
    "boltz2",
    model=Boltz2Model,
    processor=Boltz2Processor,
)
MODELS.register(
    "boltz2_affinity",
    model=Boltz2AffinityModel,
    processor=Boltz2AffinityProcessor,
)
PIPELINES.register(
    "boltz2",
    pipeline=Boltz2Pipeline,
    executions={
        "fold": (
            PipelineExecutionStep("prepare", "pipeline", Boltz2Pipeline.prepare_fold),
            PipelineExecutionStep("forward", "model", Boltz2Model.sample_with_confidence),
            PipelineExecutionStep("decode", "pipeline", Boltz2Pipeline.decode_fold),
        )
    },
)
PIPELINES.register(
    "boltz2_affinity",
    pipeline=Boltz2AffinityPipeline,
    executions={"score": (PipelineExecutionStep("execute", "pipeline", Boltz2AffinityPipeline.score),)},
)
PIPELINES.register(
    "boltz2_affinity_score",
    pipeline=Boltz2AffinityScorePipeline,
    executions={
        "score": (
            PipelineExecutionStep("prepare", "pipeline", Boltz2AffinityScorePipeline.prepare_score),
            PipelineExecutionStep("forward", "model", Boltz2AffinityModel.score),
            PipelineExecutionStep("decode", "pipeline", Boltz2AffinityScorePipeline.decode_score),
        )
    },
)
