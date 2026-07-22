r"""MPNN side-chain packing implementation."""

from __future__ import annotations

from ...pipelines.registry import PIPELINES
from ..registry import MODELS
from .config import MpnnPackerConfig, MpnnPackerProcessorConfig
from .model import (
    MpnnPackerFeatures,
    MpnnPackerInput,
    MpnnPackerModel,
    MpnnPackerOutput,
    MpnnPackerSamplingOutput,
    pack_candidate,
)
from .pdb import write_pdb
from .pipeline import MpnnPackerPipeline
from .processor import MpnnPackerProcessor

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
