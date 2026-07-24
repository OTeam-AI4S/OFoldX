r"""
ESMFold2 folding and experimental design implementations.
"""

from __future__ import annotations

from ...pipelines.registry import PIPELINES as _PIPELINES
from ..registry import MODELS as _MODELS
from ..registry import Task as _Task
from .config import (
    EsmFold2AtomEncoderConfig,
    EsmFold2ConfidenceHeadConfig,
    EsmFold2Config,
    EsmFold2DenoiserConfig,
    EsmFold2DiffusionStructureHeadConfig,
    EsmFold2EsmCConfig,
    EsmFold2ExperimentalConfidenceHeadConfig,
    EsmFold2ExperimentalConfig,
    EsmFold2ExperimentalProcessorConfig,
    EsmFold2FoldingTrunkConfig,
    EsmFold2InputEmbeddingsConfig,
    EsmFold2LanguageModelEncoderConfig,
    EsmFold2MsaEncoderConfig,
    EsmFold2ParcaeConfig,
    EsmFold2ProcessorConfig,
)
from .model import (
    EsmFold2DesignCandidate,
    EsmFold2DesignResult,
    EsmFold2ExperimentalModel,
    EsmFold2Input,
    EsmFold2Model,
    EsmFold2Output,
)
from .pipeline import EsmFold2DesignPipeline, EsmFold2Pipeline, cdr_distogram_iptm_proxy
from .processor import (
    EsmFold2ExperimentalProcessor,
    EsmFold2Processor,
    antibody_cdr_indices,
    antibody_cdr_prompt,
    is_antibody_sequence,
)

_FOLD_TASKS: tuple[_Task, ...] = ("fold",)
_DESIGN_TASKS: tuple[_Task, ...] = ("design",)

_MODELS.register_model(
    "ofoldx.esmfold2",
    model=EsmFold2Model,
    processor=EsmFold2Processor,
    processor_config=EsmFold2ProcessorConfig,
    standard_tasks=_FOLD_TASKS,
)

_MODELS.register_model(
    "ofoldx.esmfold2_experimental",
    model=EsmFold2ExperimentalModel,
    processor=EsmFold2ExperimentalProcessor,
    processor_config=EsmFold2ExperimentalProcessorConfig,
    standard_tasks=(),
)

_PIPELINES.register_pipeline(
    "ofoldx.esmfold2",
    pipeline=EsmFold2Pipeline,
    tasks=_FOLD_TASKS,
    architectures=("ofoldx.esmfold2",),
)
_PIPELINES.register_pipeline(
    "ofoldx.esmfold2_design",
    pipeline=EsmFold2DesignPipeline,
    tasks=_DESIGN_TASKS,
    architectures=("ofoldx.esmfold2_experimental",),
    repository_loader=EsmFold2DesignPipeline.from_repository,
)
