r"""
Boltz-2 training contract: the shared EDM coordinate-denoising objective. The distogram, confidence, and
B-factor heads are off the loss path; the template embedder is unused for template-free structures. All are
frozen so DDP has no unreduced parameters.
"""

from __future__ import annotations

from ...modules.diffusion.loss import denoising_criterion_factory
from ..training import TRAINING_CONTRACTS, TrainingContract

TRAINING_CONTRACTS.register(
    TrainingContract(
        architectures=("boltz2",),
        criterion=denoising_criterion_factory,
        strategy="denoising",
        freeze=("distogram_head", "confidence_head", "bfactor_head", "template_embedder"),
    )
)
