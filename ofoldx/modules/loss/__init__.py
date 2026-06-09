r"""
Training losses.

Coordinate and diffusion-denoising loss terms plus the
[`CompositeLoss`][ofoldx.modules.loss.composite.CompositeLoss] that assembles them (together
with the distogram and confidence heads) into a single weighted training objective, returning a
[`LossOutput`][ofoldx.modules.loss.composite.LossOutput].
"""

from __future__ import annotations

from .composite import CompositeLoss, LossOutput
from .coordinate import CoordinateMseLoss, DiffusionDenoisingLoss

__all__ = ["CompositeLoss", "CoordinateMseLoss", "DiffusionDenoisingLoss", "LossOutput"]
