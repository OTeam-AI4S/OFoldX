r"""
Boltz-1 training contract: the shared EDM coordinate-denoising objective. The distogram and confidence
heads never feed the denoised coordinates, so they are frozen (Boltz-1 has no template embedder).
"""

from __future__ import annotations

from ...modules.diffusion.loss import denoising_criterion_factory
from ..training import TRAINING_CONTRACTS, TrainingContract

TRAINING_CONTRACTS.register(
    TrainingContract(
        architectures=("boltz1",),
        criterion=denoising_criterion_factory,
        strategy="denoising",
        freeze=("distogram_head", "confidence"),
    )
)
