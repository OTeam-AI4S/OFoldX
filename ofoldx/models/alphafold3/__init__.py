r"""
AlphaFold 3 structure-prediction implementation.
"""

from __future__ import annotations

from ...pipelines.registry import PIPELINES as _PIPELINES
from ..registry import MODELS as _MODELS
from .config import AlphaFold3Config, AlphaFold3ProcessorConfig
from .model import (
    AlphaFold3Input,
    AlphaFold3Model,
    AlphaFold3Output,
    TemplateFeatures,
    alphafold3_confidence_summary,
    alphafold3_fraction_disordered,
    alphafold3_has_clash,
    alphafold3_pae_bin_centers,
    alphafold3_predicted_tm_score,
    alphafold3_ranking_score,
    alphafold3_tm_adjusted_pae,
)
from .pipeline import AlphaFold3Pipeline
from .processor import AlphaFold3Processor

_MODELS.register(
    "ofoldx.alphafold3",
    model=AlphaFold3Model,
    processor=AlphaFold3Processor,
)

_PIPELINES.register(
    "ofoldx.alphafold3",
    pipeline=AlphaFold3Pipeline,
)
