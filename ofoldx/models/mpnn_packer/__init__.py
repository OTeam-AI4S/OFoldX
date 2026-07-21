r"""OFoldX implementation of the released MpnnPacker."""

from __future__ import annotations

from ...io import read_pdb
from ...pipelines.registry import PIPELINES
from ..registry import MODELS
from .config import MpnnPackerConfig, MpnnPackerProcessorConfig
from .model import MpnnPackerFeatures, MpnnPackerInput, MpnnPackerModel, MpnnPackerOutput, MpnnPackerSamplingOutput
from .pdb import write_pdb
from .pipeline import MpnnPackerPipeline
from .processor import MpnnPackerProcessor

__all__ = [
    "MpnnPackerConfig",
    "MpnnPackerProcessorConfig",
    "MpnnPackerModel",
    "MpnnPackerProcessor",
    "MpnnPackerInput",
    "MpnnPackerOutput",
    "MpnnPackerSamplingOutput",
    "MpnnPackerPipeline",
    "write_pdb",
]

MODELS.register(
    "ofoldx.mpnn_packer",
    model=MpnnPackerModel,
    processor=MpnnPackerProcessor,
    standard_tasks=(),
)
PIPELINES.register(
    "ofoldx.mpnn_packer",
    pipeline=MpnnPackerPipeline,
)
