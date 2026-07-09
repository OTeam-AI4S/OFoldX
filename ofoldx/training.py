r"""
Training entry points.

The DanLing-based [`Runner`][ofoldx.runners.Runner] and its [`Config`][ofoldx.runners.Config] live here,
not in the top-level namespace: someone who only wants to fold/design/score/pack a structure should never
import a training noun. This module is the ``ofoldx.training`` re-export of the training surface.
"""

from __future__ import annotations

from .runners import Config, DataConfig, LossConfig, Runner, SamplingConfig

__all__ = [
    "Config",
    "DataConfig",
    "LossConfig",
    "Runner",
    "SamplingConfig",
]
