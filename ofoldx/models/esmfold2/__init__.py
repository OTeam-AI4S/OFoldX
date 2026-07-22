r"""ESMFold2 folding and experimental design implementations."""

from __future__ import annotations

from ...pipelines.registry import PIPELINES
from ..registry import MODELS
from .config import (
    ESMFold2AtomAttentionConfig,
    ESMFold2ConfidenceHeadConfig,
    ESMFold2Config,
    ESMFold2DiffusionModuleConfig,
    ESMFold2ESMCConfig,
    ESMFold2ExperimentalConfidenceHeadConfig,
    ESMFold2ExperimentalConfig,
    ESMFold2FoldingTrunkConfig,
    ESMFold2InputsEmbedderConfig,
    ESMFold2LMEncoderConfig,
    ESMFold2MSAEncoderConfig,
    ESMFold2ParcaeConfig,
    ESMFold2ProcessorConfig,
    ESMFold2StructureHeadConfig,
)
from .model import (
    ESMFold2BackendSettings,
    ESMFold2DesignCandidate,
    ESMFold2DesignResult,
    ESMFold2ExperimentalModel,
    ESMFold2Input,
    ESMFold2Model,
    ESMFold2Output,
)
from .pipeline import ESMFold2DesignPipeline, ESMFold2Pipeline, cdr_distogram_iptm_proxy
from .processor import (
    ESMFold2ExperimentalProcessor,
    ESMFold2Processor,
    antibody_cdr_indices,
    antibody_cdr_prompt,
    is_antibody_sequence,
)

MODELS.register_model(
    "ofoldx.esmfold2",
    model=ESMFold2Model,
    processor=ESMFold2Processor,
    processor_config=ESMFold2ProcessorConfig,
    standard_tasks={"fold"},
)

# The experimental graph uses a distinct processor and the concrete design pipeline.
MODELS.register_model(
    "ofoldx.esmfold2_experimental",
    model=ESMFold2ExperimentalModel,
    processor=ESMFold2ExperimentalProcessor,
    processor_config=ESMFold2ExperimentalConfig,
    standard_tasks=(),
)

PIPELINES.register_pipeline(
    "ofoldx.esmfold2",
    pipeline=ESMFold2Pipeline,
    tasks=("fold",),
    architectures=("ofoldx.esmfold2",),
)
PIPELINES.register_pipeline(
    "ofoldx.esmfold2_design",
    pipeline=ESMFold2DesignPipeline,
    tasks=("design",),
    architectures=("ofoldx.esmfold2_experimental",),
    repository_loader=ESMFold2DesignPipeline.from_repository,
)
