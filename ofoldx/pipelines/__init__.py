r"""
Task pipeline utilities.
"""

from __future__ import annotations

from .composite import CompositePair, CompositePipeline
from .generator import resolve_generator
from .paired import PairedComponent, PairedPipeline
from .registry import (
    PIPELINES,
    PipelineExecution,
    PipelineExecutionStep,
    PipelineRegistration,
    PipelineTask,
)

__all__ = [
    "CompositePair",
    "CompositePipeline",
    "PIPELINES",
    "PairedComponent",
    "PairedPipeline",
    "PipelineExecution",
    "PipelineExecutionStep",
    "PipelineRegistration",
    "PipelineTask",
    "resolve_generator",
]
