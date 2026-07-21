r"""OFoldX-native implementation of the released RFdiffusion architecture."""

from __future__ import annotations

from ...pipelines.registry import PIPELINES
from ..registry import MODELS
from .config import RFdiffusionConfig, RFdiffusionProcessorConfig
from .model import (
    ComputeAllAtomCoords,
    RFdiffusionDesignCandidate,
    RFdiffusionInput,
    RFdiffusionModel,
    RFdiffusionOutput,
    RFdiffusionResult,
)
from .pipeline import RFdiffusionPipeline
from .processor import RFdiffusionProcessor

__all__ = [
    "RFdiffusionConfig",
    "RFdiffusionProcessorConfig",
    "RFdiffusionModel",
    "RFdiffusionProcessor",
    "RFdiffusionPipeline",
    "RFdiffusionInput",
    "RFdiffusionOutput",
    "RFdiffusionResult",
    "RFdiffusionDesignCandidate",
    "ComputeAllAtomCoords",
]

MODELS.register(
    "ofoldx.rfdiffusion",
    model=RFdiffusionModel,
    processor=RFdiffusionProcessor,
    standard_tasks=(),
)

# `standard_tasks` stays empty (§6.1): the root pair must NOT be wrapped by the generic standard
# "design" Pipeline, because `design` returns the family-owned `RFdiffusionResult`, which is not
# losslessly the shared `DesignResult` (§7). Instead the concrete coordinate-producing
# `RFdiffusionPipeline` is published under the family id, exactly as `ProteinMpnnPipeline` is: a
# checkpoint shipping `pipeline.json = {"pipeline": "ofoldx.rfdiffusion"}` routes `design` to it and
# receives the family-owned Result (see `api/tasks.py`). Publication is wired now (Principle 13): the
# validated combined-length + hotspot `design` verb has a working Pipeline + Result carrier.
PIPELINES.register(
    "ofoldx.rfdiffusion",
    pipeline=RFdiffusionPipeline,
)
