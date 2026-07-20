r"""Public tensor-level API of the PxDesign family."""

from __future__ import annotations

from ...pipelines.registry import register_pipeline as _register_pipeline
from ..registry import register_model as _register_model
from .config import (
    PxDesignConditionEmbedderConfig,
    PxDesignConfig,
    PxDesignDenoiserConfig,
    PxDesignProcessorConfig,
    PxDesignSamplerConfig,
)
from .model import (
    PxDesignCandidate,
    PxDesignConditionalTemplateEmbedder,
    PxDesignConditionEmbedder,
    PxDesignInput,
    PxDesignLayout,
    PxDesignModel,
    PxDesignOutput,
    PxDesignResult,
    PxDesignSampler,
)
from .pipeline import PxDesignPipeline, PxDesignSuitePipeline
from .processor import PxDesignProcessor, TargetChain, from_binder, to_binder

# `standard_tasks` stays () (§6.1, mirroring rfdiffusion): the root pair must NOT be wrapped by
# the generic standard `DesignPipeline`, which returns the shared coordinate-free `DesignResult`
# and lowers `AllowedResidueIdentities`/`TiedResidueIdentities`. PxDesign is coordinate-producing:
# its `design` returns the family-owned `PxDesignResult` (§5.3) over its own `PlacementSpec`/
# `Hotspot` conditions. Publication (Principle 13) is therefore the concrete `PxDesignPipeline`
# registered below — exactly as `PxDesignPipeline` mirrors `ProteinMpnnPipeline` (pipeline.py).
# The `PxDesignSuitePipeline` composition stays UNregistered: it is dependency-gated on the
# AF2-IG fold entry point and the un-vendored `pxdbench` submodule (honest deferred gap).
_register_model(
    "ofoldx.pxdesign",
    model=PxDesignModel,
    processor=PxDesignProcessor,
    standard_tasks=(),
)
_register_pipeline(
    "ofoldx.pxdesign",
    pipeline=PxDesignPipeline,
)
