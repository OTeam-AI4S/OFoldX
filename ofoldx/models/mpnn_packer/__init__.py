r"""MPNN side-chain packing implementation."""

from __future__ import annotations

from ...pipelines.registry import PIPELINES
from ..registry import MODELS
from . import pdb as _pdb_io
from .config import MpnnPackerConfig, MpnnPackerProcessorConfig
from .model import (
    MpnnPackerFeatures,
    MpnnPackerFrame,
    MpnnPackerInput,
    MpnnPackerModel,
    MpnnPackerOutput,
    MpnnPackerSamplingOutput,
    pack_candidate,
)
from .pipeline import MpnnPackerPipeline
from .processor import MpnnPackerProcessor

write_pdb = _pdb_io.write_pdb

MODELS.register_model(
    "ofoldx.mpnn_packer",
    model=MpnnPackerModel,
    processor=MpnnPackerProcessor,
    processor_config=MpnnPackerProcessorConfig,
    standard_tasks=(),
)
PIPELINES.register_pipeline(
    "ofoldx.mpnn_packer",
    pipeline=MpnnPackerPipeline,
    tasks=("pack",),
    architectures=("ofoldx.mpnn_packer",),
)
