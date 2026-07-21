r"""OFoldX-native NA-MPNN mixed-polymer inverse-folding architecture."""

from __future__ import annotations

from ...pipelines.registry import PIPELINES
from ..registry import MODELS
from .config import NaMpnnConfig, NaMpnnProcessorConfig
from .model import (
    MIXED_POLYMER_ATOM_NAMES,
    MIXED_POLYMER_COMPONENT_IDS,
    NaMpnnFeatureEncoder,
    NaMpnnInput,
    NaMpnnModel,
    nll_loss,
    smoothed_loss,
    token_groups_from_ids,
)
from .pipeline import NaMpnnPipeline
from .processor import NaMpnnProcessor

__all__ = [
    "NaMpnnConfig",
    "NaMpnnProcessorConfig",
    "NaMpnnModel",
    "NaMpnnProcessor",
    "NaMpnnInput",
    "NaMpnnPipeline",
    "nll_loss",
    "smoothed_loss",
]

MODELS.register(
    "ofoldx.nampnn",
    model=NaMpnnModel,
    processor=NaMpnnProcessor,
    standard_tasks=(),
)
PIPELINES.register(
    "ofoldx.nampnn",
    pipeline=NaMpnnPipeline,
)
