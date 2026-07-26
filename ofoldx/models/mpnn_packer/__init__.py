r"""
MPNN side-chain packing implementation.
"""

from __future__ import annotations

from ...pipelines.registry import PIPELINES as _PIPELINES
from ..registry import MODELS as _MODELS
from . import pdb as _pdb_io
from .config import MpnnPackerConfig, MpnnPackerProcessorConfig
from .model import (
    MpnnPackerFeatureEncoder,
    MpnnPackerFrame,
    MpnnPackerInput,
    MpnnPackerModel,
    MpnnPackerOutput,
    MpnnPackerSamplingOutput,
)
from .pipeline import MpnnPackerPipeline
from .processor import MpnnPackerProcessor

write_pdb = _pdb_io.write_pdb

_MODELS.register(
    "mpnn_packer",
    model=MpnnPackerModel,
    processor=MpnnPackerProcessor,
)
_PIPELINES.register(
    "mpnn_packer",
    pipeline=MpnnPackerPipeline,
)
