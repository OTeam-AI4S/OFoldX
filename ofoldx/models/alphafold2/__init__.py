r"""
AlphaFold 2 monomer and multimer implementations.
"""

from __future__ import annotations

from ...pipelines import PIPELINES, PipelineExecutionStep
from ..registry import MODELS
from .config import (
    AlphaFold2Config,
    AlphaFold2EvoformerBlockConfig,
    AlphaFold2EvoformerConfig,
    AlphaFold2HeadsConfig,
    AlphaFold2InitialGuessConfig,
    AlphaFold2InputEmbeddingsConfig,
    AlphaFold2MsaConfig,
    AlphaFold2MultimerConfig,
    AlphaFold2MultimerHeadsConfig,
    AlphaFold2MultimerInputEmbeddingsConfig,
    AlphaFold2MultimerProcessorConfig,
    AlphaFold2MultimerStructureModuleConfig,
    AlphaFold2MultimerTemplateConfig,
    AlphaFold2ProcessorConfig,
    AlphaFold2RecyclingConfig,
    AlphaFold2StructureModuleConfig,
    AlphaFold2TemplateConfig,
)
from .model import (
    AlphaFold2InitialGuessInput,
    AlphaFold2InitialGuessModel,
    AlphaFold2Input,
    AlphaFold2Model,
    AlphaFold2MultimerInput,
    AlphaFold2MultimerModel,
    AlphaFold2Output,
    AlphaFold2StructureModuleOutput,
    alphafold2_predicted_interface_tm_score,
    alphafold2_predicted_tm_score,
)
from .multimer_processor import AlphaFold2MultimerProcessor
from .objectives import masked_msa_criterion
from .pipeline import (
    AlphaFold2Casp14Pipeline,
    AlphaFold2Casp14Prediction,
    AlphaFold2Casp14Result,
    AlphaFold2InitialGuessPipeline,
    AlphaFold2MultimerPipeline,
    AlphaFold2Pipeline,
)
from .processor import AlphaFold2InitialGuessProcessor, AlphaFold2Processor
from .relax import (
    AlphaFold2AmberRelaxation,
    AlphaFold2RelaxationResult,
    alphafold2_structural_violations,
)

__all__ = [
    "AlphaFold2AmberRelaxation",
    "AlphaFold2Casp14Pipeline",
    "AlphaFold2Casp14Prediction",
    "AlphaFold2Casp14Result",
    "AlphaFold2Config",
    "AlphaFold2EvoformerBlockConfig",
    "AlphaFold2EvoformerConfig",
    "AlphaFold2HeadsConfig",
    "AlphaFold2InitialGuessConfig",
    "AlphaFold2InitialGuessInput",
    "AlphaFold2InitialGuessModel",
    "AlphaFold2InitialGuessPipeline",
    "AlphaFold2InitialGuessProcessor",
    "AlphaFold2Input",
    "AlphaFold2InputEmbeddingsConfig",
    "AlphaFold2Model",
    "AlphaFold2MsaConfig",
    "AlphaFold2MultimerConfig",
    "AlphaFold2MultimerHeadsConfig",
    "AlphaFold2MultimerInput",
    "AlphaFold2MultimerInputEmbeddingsConfig",
    "AlphaFold2MultimerModel",
    "AlphaFold2MultimerPipeline",
    "AlphaFold2MultimerProcessor",
    "AlphaFold2MultimerProcessorConfig",
    "AlphaFold2MultimerStructureModuleConfig",
    "AlphaFold2MultimerTemplateConfig",
    "AlphaFold2Output",
    "AlphaFold2Pipeline",
    "AlphaFold2Processor",
    "AlphaFold2ProcessorConfig",
    "AlphaFold2RecyclingConfig",
    "AlphaFold2RelaxationResult",
    "AlphaFold2StructureModuleConfig",
    "AlphaFold2StructureModuleOutput",
    "AlphaFold2TemplateConfig",
    "alphafold2_predicted_interface_tm_score",
    "alphafold2_predicted_tm_score",
    "alphafold2_structural_violations",
    "masked_msa_criterion",
]

MODELS.register(
    "alphafold2",
    model=AlphaFold2Model,
    processor=AlphaFold2Processor,
)
MODELS.register(
    "alphafold2_initial_guess",
    model=AlphaFold2InitialGuessModel,
    processor=AlphaFold2InitialGuessProcessor,
)
MODELS.register(
    "alphafold2_multimer",
    model=AlphaFold2MultimerModel,
    processor=AlphaFold2MultimerProcessor,
)
PIPELINES.register(
    "alphafold2",
    pipeline=AlphaFold2Pipeline,
    executions={
        "fold": (
            PipelineExecutionStep("prepare", "pipeline", AlphaFold2Pipeline.prepare_fold),
            PipelineExecutionStep("forward", "pipeline", AlphaFold2Pipeline.sample_fold),
            PipelineExecutionStep("decode", "pipeline", AlphaFold2Pipeline.decode_fold),
        )
    },
)
PIPELINES.register(
    "alphafold2_casp14",
    pipeline=AlphaFold2Casp14Pipeline,
    executions={"fold": (PipelineExecutionStep("execute", "pipeline", AlphaFold2Casp14Pipeline.fold),)},
)
PIPELINES.register(
    "alphafold2_initial_guess",
    pipeline=AlphaFold2InitialGuessPipeline,
    executions={
        "fold": (
            PipelineExecutionStep("prepare", "pipeline", AlphaFold2InitialGuessPipeline.prepare_fold),
            PipelineExecutionStep("forward", "pipeline", AlphaFold2InitialGuessPipeline.sample_fold),
            PipelineExecutionStep("decode", "pipeline", AlphaFold2InitialGuessPipeline.decode_fold),
        )
    },
)
PIPELINES.register(
    "alphafold2_multimer",
    pipeline=AlphaFold2MultimerPipeline,
    executions={
        "fold": (
            PipelineExecutionStep("prepare", "pipeline", AlphaFold2MultimerPipeline.prepare_fold),
            PipelineExecutionStep("forward", "pipeline", AlphaFold2MultimerPipeline.sample_fold),
            PipelineExecutionStep("decode", "pipeline", AlphaFold2MultimerPipeline.decode_fold),
        )
    },
)
