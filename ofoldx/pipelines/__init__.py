r"""
Task pipeline utilities.
"""

from __future__ import annotations

from .composite import CompositePair, CompositePipeline
from .generator import resolve_generator
from .paired import PairedPipeline
from .registry import PIPELINES

__all__ = ["CompositePair", "CompositePipeline", "PIPELINES", "PairedPipeline", "resolve_generator"]
