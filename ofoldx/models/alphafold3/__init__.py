r"""
AlphaFold 3 structure-prediction implementation.
"""

from __future__ import annotations

from ...pipelines import PIPELINES, PipelineExecutionStep
from ..registry import MODELS
from .config import (
    AlphaFold3ConfidenceConfig,
    AlphaFold3Config,
    AlphaFold3ProcessorConfig,
    AlphaFold3TemplateConfig,
    AlphaFold3TrunkConfig,
)
from .model import (
    AlphaFold3Input,
    AlphaFold3Model,
    AlphaFold3Output,
    AlphaFold3SamplingOutput,
    AlphaFold3StructureTarget,
    AlphaFold3TemplateFeatures,
    alphafold3_confidence_summary,
    alphafold3_pae_bin_centers,
    alphafold3_predicted_tm_score,
    alphafold3_ranking_score,
    alphafold3_tm_adjusted_pae,
)
from .pipeline import AlphaFold3Pipeline, alphafold3_fraction_disordered, alphafold3_has_clash
from .processor import AlphaFold3Processor

__all__ = [
    "AlphaFold3ConfidenceConfig",
    "AlphaFold3Config",
    "AlphaFold3Input",
    "AlphaFold3Model",
    "AlphaFold3Output",
    "AlphaFold3Pipeline",
    "AlphaFold3Processor",
    "AlphaFold3ProcessorConfig",
    "AlphaFold3SamplingOutput",
    "AlphaFold3StructureTarget",
    "AlphaFold3TemplateFeatures",
    "AlphaFold3TemplateConfig",
    "AlphaFold3TrunkConfig",
    "alphafold3_confidence_summary",
    "alphafold3_fraction_disordered",
    "alphafold3_has_clash",
    "alphafold3_pae_bin_centers",
    "alphafold3_predicted_tm_score",
    "alphafold3_ranking_score",
    "alphafold3_tm_adjusted_pae",
]

MODELS.register(
    "alphafold3",
    model=AlphaFold3Model,
    processor=AlphaFold3Processor,
)

PIPELINES.register(
    "alphafold3",
    pipeline=AlphaFold3Pipeline,
    executions={
        "fold": (
            PipelineExecutionStep("prepare", "pipeline", AlphaFold3Pipeline.prepare_fold),
            PipelineExecutionStep("forward", "model", AlphaFold3Model.sample),
            PipelineExecutionStep("decode", "pipeline", AlphaFold3Pipeline.decode_fold),
        )
    },
)
