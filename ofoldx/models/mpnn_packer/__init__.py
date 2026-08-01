r"""
MPNN side-chain packing implementation.
"""

from __future__ import annotations

from ...pipelines.registry import PIPELINES as _PIPELINES
from ..registry import MODELS as _MODELS
from .config import MpnnPackerConfig, MpnnPackerProcessorConfig
from .density import structure_with_chi_density
from .model import (
    MpnnPackerFeatureEncoder,
    MpnnPackerInput,
    MpnnPackerModel,
    MpnnPackerOutput,
    MpnnPackerSamplingOutput,
)
from .pipeline import MpnnPackerPipeline
from .processor import MpnnPackerProcessor

__all__ = [
    "MpnnPackerConfig",
    "MpnnPackerFeatureEncoder",
    "MpnnPackerInput",
    "MpnnPackerModel",
    "MpnnPackerOutput",
    "MpnnPackerPipeline",
    "MpnnPackerProcessor",
    "MpnnPackerProcessorConfig",
    "MpnnPackerSamplingOutput",
    "structure_with_chi_density",
]

_MODELS.register(
    "mpnn_packer",
    model=MpnnPackerModel,
    processor=MpnnPackerProcessor,
)
_PIPELINES.register(
    "mpnn_packer",
    pipeline=MpnnPackerPipeline,
)
