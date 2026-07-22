r"""LigandMPNN protein-sequence design implementation."""

from __future__ import annotations

from ...pipelines.registry import PIPELINES
from ..registry import MODELS
from .config import LigandMpnnConfig, LigandMpnnProcessorConfig
from .model import LigandMpnnInput, LigandMpnnModel
from .pipeline import LigandMpnnDesignAndPackPipeline, LigandMpnnPipeline
from .processor import LigandMpnnProcessor

MODELS.register_model(
    "ofoldx.ligandmpnn",
    model=LigandMpnnModel,
    processor=LigandMpnnProcessor,
    processor_config=LigandMpnnProcessorConfig,
    standard_tasks=(),
)
PIPELINES.register_pipeline(
    "ofoldx.ligandmpnn",
    pipeline=LigandMpnnPipeline,
    tasks=("design",),
    architectures=("ofoldx.ligandmpnn",),
)
