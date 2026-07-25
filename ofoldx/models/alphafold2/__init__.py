r"""
AlphaFold 2 monomer and multimer implementations.
"""

from __future__ import annotations

from ...pipelines.registry import PIPELINES as _PIPELINES
from ..registry import MODELS as _MODELS
from .config import (
    AlphaFold2Config,
    AlphaFold2EvoformerBlockConfig,
    AlphaFold2EvoformerConfig,
    AlphaFold2HeadsConfig,
    AlphaFold2InputEmbeddingsConfig,
    AlphaFold2MsaConfig,
    AlphaFold2MultimerConfig,
    AlphaFold2MultimerHeadsConfig,
    AlphaFold2MultimerInputEmbeddingsConfig,
    AlphaFold2MultimerProcessorConfig,
    AlphaFold2MultimerStructureModuleConfig,
    AlphaFold2MultimerTemplateConfig,
    AlphaFold2ProcessorConfig,
    AlphaFold2RecyclingConfig,
    AlphaFold2StructureModuleConfig,
    AlphaFold2TemplateConfig,
)
from .model import (
    AlphaFold2Input,
    AlphaFold2Model,
    AlphaFold2MultimerInput,
    AlphaFold2MultimerModel,
    AlphaFold2Output,
    AlphaFold2StructureModuleOutput,
    alphafold2_predicted_interface_tm_score,
    alphafold2_predicted_tm_score,
)
from .multimer_processor import AlphaFold2MultimerProcessor
from .pipeline import AlphaFold2MultimerPipeline, AlphaFold2Pipeline
from .processor import AlphaFold2Processor

_MODELS.register(
    "ofoldx.alphafold2",
    model=AlphaFold2Model,
    processor=AlphaFold2Processor,
)
_MODELS.register(
    "ofoldx.alphafold2_multimer",
    model=AlphaFold2MultimerModel,
    processor=AlphaFold2MultimerProcessor,
)
_PIPELINES.register(
    "ofoldx.alphafold2",
    pipeline=AlphaFold2Pipeline,
)
_PIPELINES.register(
    "ofoldx.alphafold2_multimer",
    pipeline=AlphaFold2MultimerPipeline,
)
