r"""Boltz-2 folding and two-stage affinity family, its Processors, Pipeline, and registration."""

from __future__ import annotations

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

# Two registered Model/Processor pairs over two distinct checkpoints. The standard ``ofoldx.boltz2`` fold pair
# resolves through the generic standard fold Pipeline (standard_tasks=("fold",)) and is fully published.
#
# The ``ofoldx.boltz2_affinity`` pair has no lossless standard task (its score is the concrete two-stage op),
# so standard_tasks=(). Its concrete ``Boltz2AffinityPipeline`` is intentionally NOT ``register_pipeline``-ed
# this wave -- deferred, matching the reserved-id contract in ``ofoldx/pipelines/registry.py``
# (``_RESERVED_PIPELINE_IDS``). The routed factory (``api/tasks.py``) builds every registered pipeline through
# the ``(model, processor)`` path, but this Pipeline binds two checkpoints (a 4-arg
# ``model``/``processor``/``structure_model``/``structure_processor`` ctor) and has no multi-pair loader/factory
# nor a ``.save()``, so a routed ``score`` would crash on construction. Registration is withheld until it gains
# that two-checkpoint loader + ``.save()`` (see the registry TODO). ``Boltz2AffinityPipeline`` stays importable
# and exported for manual construction; the id remains reserved so no third party can squat it.
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
