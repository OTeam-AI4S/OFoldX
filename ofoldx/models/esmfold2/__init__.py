r"""
ESMFold2 folding and experimental design implementations.
"""

from __future__ import annotations

from ...pipelines.registry import PIPELINES as _PIPELINES
from ..registry import MODELS as _MODELS
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
    EsmFold2ExperimentalConfidence,
    EsmFold2ExperimentalModel,
    EsmFold2ExperimentalPrediction,
    EsmFold2ExperimentalSample,
    EsmFold2Input,
    EsmFold2Model,
    EsmFold2Output,
)
from .pipeline import (
    EsmFold2CriticEvaluation,
    EsmFold2DesignCandidate,
    EsmFold2DesignPipeline,
    EsmFold2DesignResult,
    EsmFold2DesignStep,
    EsmFold2ExperimentalPipeline,
    EsmFold2Pipeline,
    cdr_distogram_iptm_proxy,
)
from .processor import (
    EsmFold2ExperimentalProcessor,
    EsmFold2Processor,
    antibody_cdr_indices,
    antibody_cdr_prompt,
    is_antibody_sequence,
)

_MODELS.register(
    "esmfold2",
    model=EsmFold2Model,
    processor=EsmFold2Processor,
)

_MODELS.register(
    "esmfold2_experimental",
    model=EsmFold2ExperimentalModel,
    processor=EsmFold2ExperimentalProcessor,
)

_PIPELINES.register(
    "esmfold2",
    pipeline=EsmFold2Pipeline,
)

_PIPELINES.register(
    "esmfold2_experimental",
    pipeline=EsmFold2ExperimentalPipeline,
)

_PIPELINES.register(
    "esmfold2_design",
    pipeline=EsmFold2DesignPipeline,
)
