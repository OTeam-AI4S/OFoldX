r"""OFoldX-native NA-MPNN mixed-polymer inverse-folding architecture."""

from __future__ import annotations

from ...pipelines.registry import register_pipeline as _register_pipeline
from ..registry import register_model as _register_model
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

_register_model(
    "ofoldx.nampnn",
    model=NaMpnnModel,
    processor=NaMpnnProcessor,
    standard_tasks=(),
)
_register_pipeline(
    "ofoldx.nampnn",
    pipeline=NaMpnnPipeline,
)
