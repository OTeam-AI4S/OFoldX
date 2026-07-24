r"""
NA-MPNN mixed-polymer inverse-folding implementation.
"""

from __future__ import annotations

from ...pipelines.registry import PIPELINES as _PIPELINES
from ..registry import MODELS as _MODELS
from .config import NaMpnnConfig, NaMpnnProcessorConfig
from .model import (
    NA_MPNN_ATOM_INDEX,
    NA_MPNN_ATOM_NAMES,
    NA_MPNN_COMPONENT_IDS,
    NA_MPNN_DNA_GENERATABLE_TOKEN_IDS,
    NA_MPNN_INVALID_GENERATION_TOKEN_IDS,
    NA_MPNN_PADDING_TOKEN_ID,
    NA_MPNN_PROTEIN_GENERATABLE_TOKEN_IDS,
    NA_MPNN_RNA_SEPARATE_GENERATABLE_TOKEN_IDS,
    NA_MPNN_RNA_SHARED_GENERATABLE_TOKEN_IDS,
    NA_MPNN_TOKEN_BY_COMPONENT,
    NaMpnnFeatures,
    NaMpnnInput,
    NaMpnnModel,
    NaMpnnOutput,
    NaMpnnSamplingOutput,
    nampnn_polymer_label_smoothed_nll_loss,
)
from .pipeline import NaMpnnPipeline
from .processor import NaMpnnProcessor

_MODELS.register_model(
    "ofoldx.nampnn",
    model=NaMpnnModel,
    processor=NaMpnnProcessor,
    processor_config=NaMpnnProcessorConfig,
    standard_tasks=(),
)
_PIPELINES.register_pipeline(
    "ofoldx.nampnn",
    pipeline=NaMpnnPipeline,
    tasks=("design", "score"),
    architectures=("ofoldx.nampnn",),
)
