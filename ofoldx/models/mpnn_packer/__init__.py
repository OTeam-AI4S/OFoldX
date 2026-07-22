r"""MPNN side-chain packing implementation."""

from __future__ import annotations

from typing import cast

from ...pipelines.registry import PIPELINES
from ..registry import MODELS, Task
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
    tasks=cast(tuple[Task], ("pack",)),
    architectures=("ofoldx.mpnn_packer",),
)
