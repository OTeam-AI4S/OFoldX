r"""
Boltz-1 training contract: the shared EDM coordinate-denoising objective.
"""

from __future__ import annotations

from ...modules.diffusion.loss import denoising_criterion_factory
from ..training import TrainingContract

TRAINING = TrainingContract(
    architectures=("boltz1",),
    criterion=denoising_criterion_factory,
    strategy="denoising",
)
