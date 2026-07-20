r"""Boltz-2 folding and two-stage affinity family, its Processors, Pipeline, and registration."""

from __future__ import annotations

from ...pipelines.registry import PIPELINES
from ..registry import MODELS
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

# Two registered Model/Processor pairs over two distinct checkpoints. The standard ``ofoldx.boltz2`` fold pair
# resolves through the generic standard fold Pipeline (standard_tasks=("fold",)) and is fully published.
#
# The ``ofoldx.boltz2_affinity`` pair has no lossless standard task (its score is the concrete two-stage op),
# so standard_tasks=(). Its concrete ``Boltz2AffinityPipeline`` is a DESIGN §7.4 multi-pair Pipeline: it binds
# the affinity root pair and the ``structure/`` structure/confidence pair, declares a class-level
# ``_PAIR_LAYOUT``, and carries the shared ``save`` / ``from_repository`` (``ofoldx/pipelines/_composite.py``).
# It is registered below, so a marked repository's ``score`` routes through the factory's multi-pair loader
# (``api/tasks.py`` dispatches to ``from_repository``) instead of the one-pair ``(model, processor)`` call.
MODELS.register(
    "ofoldx.boltz2",
    model=Boltz2Model,
    processor=Boltz2Processor,
    standard_tasks=("fold",),
)
MODELS.register(
    "ofoldx.boltz2_affinity",
    model=Boltz2AffinityModel,
    processor=Boltz2AffinityProcessor,
    standard_tasks=(),
)
PIPELINES.register(
    "ofoldx.boltz2_affinity",
    pipeline=Boltz2AffinityPipeline,
)
