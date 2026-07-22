r"""AlphaFold 2 monomer and multimer implementations."""

from __future__ import annotations

from ...pipelines.registry import PIPELINES
from ..registry import MODELS
from .config import (
    AlphaFold2Config,
    AlphaFold2EvoformerConfig,
    AlphaFold2EvoformerIterationConfig,
    AlphaFold2HeadsConfig,
    AlphaFold2InputEmbeddingConfig,
    AlphaFold2MSAConfig,
    AlphaFold2MultimerConfig,
    AlphaFold2MultimerHeadsConfig,
    AlphaFold2MultimerInputEmbeddingConfig,
    AlphaFold2MultimerProcessorConfig,
    AlphaFold2MultimerTemplateConfig,
    AlphaFold2ProcessorConfig,
    AlphaFold2RecyclingConfig,
    AlphaFold2StructureModuleHeadConfig,
    AlphaFold2TemplateConfig,
)
from .model import (
    AlphaFold2Input,
    AlphaFold2Model,
    AlphaFold2MultimerInput,
    AlphaFold2MultimerModel,
    AlphaFold2Output,
    AlphaFold2StructureModuleOutput,
    predicted_interface_tm_score,
)
from .multimer_processor import AlphaFold2MultimerProcessor
from .pipeline import AlphaFold2MultimerPipeline, AlphaFold2Pipeline
from .processor import AlphaFold2Processor

MODELS.register_model(
    "ofoldx.alphafold2",
    model=AlphaFold2Model,
    processor=AlphaFold2Processor,
    processor_config=AlphaFold2ProcessorConfig,
    standard_tasks=("fold",),
)
MODELS.register_model(
    "ofoldx.alphafold2_multimer",
    model=AlphaFold2MultimerModel,
    processor=AlphaFold2MultimerProcessor,
    processor_config=AlphaFold2MultimerProcessorConfig,
    standard_tasks=("fold",),
)
PIPELINES.register_pipeline(
    "ofoldx.alphafold2",
    pipeline=AlphaFold2Pipeline,
    tasks=("fold",),
    architectures=("ofoldx.alphafold2",),
)
PIPELINES.register_pipeline(
    "ofoldx.alphafold2_multimer",
    pipeline=AlphaFold2MultimerPipeline,
    tasks=("fold",),
    architectures=("ofoldx.alphafold2_multimer",),
)
