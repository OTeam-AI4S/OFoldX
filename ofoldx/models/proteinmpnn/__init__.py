r"""ProteinMPNN inverse-folding implementation."""

from __future__ import annotations

from ...pipelines.registry import PIPELINES
from ..registry import MODELS
from .config import ProteinMpnnConfig, ProteinMpnnProcessorConfig
from .model import ProteinMpnnDesignResult, ProteinMpnnInput, ProteinMpnnModel
from .pipeline import ProteinMpnnDesignAndPackPipeline, ProteinMpnnPipeline
from .processor import ProteinMpnnProcessor

MODELS.register_model(
    "ofoldx.proteinmpnn",
    model=ProteinMpnnModel,
    processor=ProteinMpnnProcessor,
    processor_config=ProteinMpnnProcessorConfig,
    standard_tasks=(),
)
PIPELINES.register_pipeline(
    "ofoldx.proteinmpnn",
    pipeline=ProteinMpnnPipeline,
    tasks=("design", "score"),
    architectures=("ofoldx.proteinmpnn",),
)
