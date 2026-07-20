r"""OFoldX-native ProteinMPNN architecture."""

from __future__ import annotations

from ...modules.mpnn import MpnnProteinFeatures
from ...pipelines.registry import PIPELINES
from ..registry import MODELS
from .config import ProteinMpnnConfig, ProteinMpnnProcessorConfig
from .model import CaProteinMpnnFeatures, ProteinMpnnDesignResult, ProteinMpnnInput, ProteinMpnnModel
from .pipeline import ProteinMpnnDesignAndPackPipeline, ProteinMpnnPipeline
from .processor import ProteinMpnnProcessor

MODELS.register(
    "ofoldx.proteinmpnn",
    model=ProteinMpnnModel,
    processor=ProteinMpnnProcessor,
    standard_tasks=(),
)
PIPELINES.register(
    "ofoldx.proteinmpnn",
    pipeline=ProteinMpnnPipeline,
)
