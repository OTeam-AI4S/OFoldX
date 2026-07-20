r"""OFoldX-native ProteinMPNN architecture."""

from __future__ import annotations

from ...modules.mpnn import MpnnProteinFeatures
from ...pipelines.registry import register_pipeline as _register_pipeline
from ..registry import register_model as _register_model
from .config import ProteinMpnnConfig, ProteinMpnnProcessorConfig
from .model import CaProteinMpnnFeatures, ProteinMpnnDesignResult, ProteinMpnnInput, ProteinMpnnModel
from .pipeline import ProteinMpnnDesignAndPackPipeline, ProteinMpnnPipeline
from .processor import ProteinMpnnProcessor

_register_model(
    "ofoldx.proteinmpnn",
    model=ProteinMpnnModel,
    processor=ProteinMpnnProcessor,
    standard_tasks=(),
)
_register_pipeline(
    "ofoldx.proteinmpnn",
    pipeline=ProteinMpnnPipeline,
)
