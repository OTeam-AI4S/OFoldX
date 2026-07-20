r"""OFoldX implementation of the released MpnnPacker."""

from __future__ import annotations

from ...io import read_pdb, write_pdb
from ...pipelines.registry import register_pipeline as _register_pipeline
from ..registry import register_model as _register_model
from .config import MpnnPackerConfig, MpnnPackerProcessorConfig
from .model import MpnnPackerFeatures, MpnnPackerInput, MpnnPackerModel, MpnnPackerOutput, MpnnPackerSamplingOutput
from .pipeline import MpnnPackerPipeline
from .processor import MpnnPackerProcessor

_register_model(
    "ofoldx.mpnn_packer",
    model=MpnnPackerModel,
    processor=MpnnPackerProcessor,
    standard_tasks=(),
)
_register_pipeline(
    "ofoldx.mpnn_packer",
    pipeline=MpnnPackerPipeline,
)
