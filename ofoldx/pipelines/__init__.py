r"""
Task pipelines that pair a model with its structure processor behind a single callable.

Each pipeline is a thin inference boundary: it accepts a parsed [`Structure`][ofoldx.data.structure.Structure],
a structure file path, or an already-typed model input, runs the processor when needed, moves the batch to the model
device, and normalizes the raw model output into a task result.
[`Pipeline.from_pretrained`][ofoldx.pipelines.base.Pipeline.from_pretrained] dispatches to the concrete task class
from either a plain model artifact or a saved pipeline package, using the artifact task contract for structure
prediction, design generation, binding-affinity prediction, and side-chain packing.
"""

from __future__ import annotations

from .base import Pipeline, StructureProcessor
from .binding_affinity import BindingAffinityInput, BindingAffinityPipeline
from .design import (
    AllAtomDesignPipeline,
    BackboneDesignPipeline,
    DesignInput,
    DesignPipeline,
    SequenceDesignPipeline,
)
from .side_chain_packing import SideChainPackingInput, SideChainPackingPipeline
from .structure_prediction import StructurePredictionInput, StructurePredictionPipeline

__all__ = [
    "AllAtomDesignPipeline",
    "BackboneDesignPipeline",
    "DesignInput",
    "DesignPipeline",
    "SequenceDesignPipeline",
    "BindingAffinityInput",
    "BindingAffinityPipeline",
    "Pipeline",
    "SideChainPackingInput",
    "SideChainPackingPipeline",
    "StructurePredictionInput",
    "StructurePredictionPipeline",
    "StructureProcessor",
]
