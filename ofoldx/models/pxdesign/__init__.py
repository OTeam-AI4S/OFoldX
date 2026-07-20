r"""Public tensor-level API of the PxDesign family."""

from __future__ import annotations

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

_register_model(
    "ofoldx.pxdesign",
    model=PxDesignModel,
    processor=PxDesignProcessor,
    standard_tasks=(),
)
