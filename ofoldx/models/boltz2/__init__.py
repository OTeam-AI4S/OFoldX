r"""Boltz-2 folding and two-stage affinity family, its Processors, Pipeline, and registration."""

from __future__ import annotations

from ...pipelines.registry import register_pipeline as _register_pipeline
from ..registry import register_model as _register_model
from .chemistry import standardize_boltz2_affinity_smiles
from .config import (
    Boltz2AffinityConfig,
    Boltz2AffinityHeadConfig,
    Boltz2ConfidenceConfig,
    Boltz2Config,
    Boltz2DiffusionConfig,
    Boltz2InputEmbedderConfig,
    Boltz2SamplerConfig,
    Boltz2TemplateConfig,
    Boltz2TrunkConfig,
)
from .model import Boltz2AffinityModel, Boltz2Input, Boltz2Layout, Boltz2Model, Boltz2Output
from .pipeline import Boltz2AffinityPipeline
from .processor import Boltz2AffinityProcessor, Boltz2Processor

# Two registered Model/Processor pairs over two distinct checkpoints plus one concrete Pipeline. The standard
# ``ofoldx.boltz2`` fold pair resolves through the generic standard fold Pipeline (standard_tasks=("fold",)).
# The ``ofoldx.boltz2_affinity`` pair has no lossless standard task (its score is the concrete two-stage op),
# so standard_tasks=(); the concrete two-stage ``Boltz2AffinityPipeline`` registers under the same string in
# the independent pipeline registry (the one-owner convention MpnnPacker already proves).
_register_model(
    "ofoldx.boltz2",
    model=Boltz2Model,
    processor=Boltz2Processor,
    standard_tasks=("fold",),
)
_register_model(
    "ofoldx.boltz2_affinity",
    model=Boltz2AffinityModel,
    processor=Boltz2AffinityProcessor,
    standard_tasks=(),
)
_register_pipeline("ofoldx.boltz2_affinity", pipeline=Boltz2AffinityPipeline)
