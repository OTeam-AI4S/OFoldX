r"""Boltz-2 folding and affinity implementations."""

from __future__ import annotations

from ...pipelines.registry import PIPELINES
from ..registry import MODELS
from .chemistry import standardize_boltz2_affinity_smiles
from .config import (
    Boltz2AffinityConfig,
    Boltz2AffinityModuleConfig,
    Boltz2ConfidenceModuleConfig,
    Boltz2Config,
    Boltz2DenoiserConfig,
    Boltz2InputEmbeddingsConfig,
    Boltz2SamplerConfig,
    Boltz2TemplateConfig,
    Boltz2TrunkConfig,
)
from .model import Boltz2AffinityModel, Boltz2Input, Boltz2Model, Boltz2Output
from .pipeline import Boltz2AffinityPipeline, Boltz2AffinityResult, Boltz2Pipeline
from .processor import Boltz2AffinityProcessor, Boltz2Processor

MODELS.register_model(
    "ofoldx.boltz2",
    model=Boltz2Model,
    processor=Boltz2Processor,
    processor_config=None,
    standard_tasks=("fold",),
)
MODELS.register_model(
    "ofoldx.boltz2_affinity",
    model=Boltz2AffinityModel,
    processor=Boltz2AffinityProcessor,
    processor_config=None,
    standard_tasks=(),
)
PIPELINES.register_pipeline(
    "ofoldx.boltz2",
    pipeline=Boltz2Pipeline,
    tasks=("fold",),
    architectures=("ofoldx.boltz2",),
)
PIPELINES.register_pipeline(
    "ofoldx.boltz2_affinity",
    pipeline=Boltz2AffinityPipeline,
    tasks=("score",),
    repository_loader=Boltz2AffinityPipeline.from_repository,
)
