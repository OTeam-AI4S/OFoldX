r"""NA-MPNN mixed-polymer inverse-folding implementation."""

from __future__ import annotations

from ...modules.mpnn import nll_loss, tied_groups_from_ids
from ...pipelines.registry import PIPELINES
from ..registry import MODELS
from .config import NaMpnnConfig, NaMpnnProcessorConfig
from .model import (
    MIXED_POLYMER_ATOM_NAMES,
    MIXED_POLYMER_COMPONENT_IDS,
    NaMpnnFeatures,
    NaMpnnInput,
    NaMpnnModel,
    NaMpnnOutput,
    NaMpnnSamplingOutput,
    smoothed_loss,
)
from .pipeline import NaMpnnPipeline
from .processor import NaMpnnProcessor

MODELS.register_model(
    "ofoldx.nampnn",
    model=NaMpnnModel,
    processor=NaMpnnProcessor,
    processor_config=NaMpnnProcessorConfig,
    standard_tasks=(),
)
PIPELINES.register_pipeline(
    "ofoldx.nampnn",
    pipeline=NaMpnnPipeline,
    tasks=("design", "score"),
    architectures=("ofoldx.nampnn",),
)
