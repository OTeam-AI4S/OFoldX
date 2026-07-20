r"""OFoldX-native implementation of the released RFdiffusion architecture."""

from __future__ import annotations

from ..registry import register_model as _register_model
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

_register_model(
    "ofoldx.rfdiffusion",
    model=RFdiffusionModel,
    processor=RFdiffusionProcessor,
    standard_tasks=(),
)

# `standard_tasks` stays empty (§6.1): the root pair must NOT be wrapped by the generic
# standard "design" Pipeline, because `design` returns the family-owned `RFdiffusionResult`,
# which is not losslessly the shared `DesignResult` (§7). The concrete `RFdiffusionPipeline`
# registration (`register_pipeline("ofoldx.rfdiffusion", ...)` + `pipeline.json` marker) is
# deferred under Principle 13 / §7.2 until real-weight sampled-output acceptance passes, exactly
# as `ESMFold2Pipeline` defers; this wave freezes its signature and family Result type only.
