r"""Protenix v0.5 default and constraint model family."""

from __future__ import annotations

from ..registry import register_model as _register_model
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
from .processor import ProtenixConstraintProcessor, ProtenixProcessor

_register_model(
    "ofoldx.protenix",
    model=ProtenixModel,
    processor=ProtenixProcessor,
    standard_tasks=("fold",),
)
_register_model(
    "ofoldx.protenix_constraint",
    model=ProtenixConstraintModel,
    processor=ProtenixConstraintProcessor,
    standard_tasks=("fold",),
)
