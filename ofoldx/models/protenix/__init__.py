r"""Protenix v0.5 default and constraint model family."""

from __future__ import annotations

from ...pipelines.registry import PIPELINES
from ..registry import MODELS
from .config import (
    ProtenixConfig,
    ProtenixConstraintConfig,
    ProtenixConstraintProcessorConfig,
    ProtenixProcessorConfig,
)
from .model import (
    AtomConfidenceProjector,
    ConfidenceHead,
    ConfidencePairEmbedder,
    InputEmbedder,
    PairConfidenceProjector,
    ProtenixAtomContact,
    ProtenixConstraintEmbedder,
    ProtenixConstraintModel,
    ProtenixConstraintSubstructureEmbedder,
    ProtenixInput,
    ProtenixModel,
    ProtenixOutput,
    ProtenixPocket,
    ProtenixTokenContact,
)
from .pipeline import ProtenixConstraintPipeline
from .processor import ProtenixConstraintProcessor, ProtenixProcessor

MODELS.register(
    "ofoldx.protenix",
    model=ProtenixModel,
    processor=ProtenixProcessor,
    standard_tasks=("fold",),
)
# The default v0.5 pair keeps the generic standard ``fold`` route (its ``fold`` takes only
# msa/templates, which the generic ``FoldPipeline`` forwards). The constraint pair also registers
# ``standard_tasks=("fold",)``, but its released ``atom_contacts``/``token_contacts``/``pocket``
# channels live only on ``ProtenixConstraintProcessor`` and cannot be forwarded by the generic
# ``FoldPipeline``. So the concrete ``ProtenixConstraintPipeline`` is published under the family id
# and a checkpoint shipping ``pipeline.json = {"pipeline": "ofoldx.protenix_constraint"}`` routes
# ``fold`` to it, making the constraint channels reachable through ``ofoldx.fold`` (see api/tasks.py).
MODELS.register(
    "ofoldx.protenix_constraint",
    model=ProtenixConstraintModel,
    processor=ProtenixConstraintProcessor,
    standard_tasks=("fold",),
)
PIPELINES.register(
    "ofoldx.protenix_constraint",
    pipeline=ProtenixConstraintPipeline,
)
