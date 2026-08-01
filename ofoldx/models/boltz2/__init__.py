r"""
Boltz-2 folding and affinity implementations.
"""

from __future__ import annotations

from ...pipelines.registry import PIPELINES as _PIPELINES
from ..registry import MODELS as _MODELS
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
    Boltz2ConfidenceOutput,
    Boltz2Contact,
    Boltz2Input,
    Boltz2Model,
    Boltz2Output,
    Boltz2Pocket,
    Boltz2Representations,
    Boltz2StructureTarget,
    Boltz2Template,
)
from .pipeline import Boltz2AffinityPipeline, Boltz2AffinityResult, Boltz2Pipeline
from .processor import Boltz2AffinityProcessor, Boltz2Processor

__all__ = [
    "Boltz2AffinityConfig",
    "Boltz2AffinityHeadConfig",
    "Boltz2AffinityModel",
    "Boltz2AffinityOutput",
    "Boltz2AffinityPipeline",
    "Boltz2AffinityProcessor",
    "Boltz2AffinityResult",
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
    "Boltz2StructureTarget",
    "Boltz2Template",
    "Boltz2TemplateConfig",
    "Boltz2TrunkConfig",
    "standardize_boltz2_affinity_smiles",
]

_MODELS.register(
    "boltz2",
    model=Boltz2Model,
    processor=Boltz2Processor,
)
_MODELS.register(
    "boltz2_affinity",
    model=Boltz2AffinityModel,
    processor=Boltz2AffinityProcessor,
)
_PIPELINES.register(
    "boltz2",
    pipeline=Boltz2Pipeline,
)
_PIPELINES.register(
    "boltz2_affinity",
    pipeline=Boltz2AffinityPipeline,
)
