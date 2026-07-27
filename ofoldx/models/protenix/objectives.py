r"""
Protenix training contract: the shared EDM coordinate-denoising objective, covering the base model and its
conditioning variants (they share ``ProtenixModel.forward`` and ``ProtenixOutput``). The distogram and
confidence heads and the template embedder are off the loss path and frozen.
"""

from __future__ import annotations

from ...modules.diffusion.loss import denoising_criterion_factory
from ..training import TRAINING_CONTRACTS, TrainingContract

TRAINING_CONTRACTS.register(
    TrainingContract(
        architectures=("protenix", "protenix_template", "protenix_structure_condition", "protenix_constraint"),
        criterion=denoising_criterion_factory,
        strategy="denoising",
        freeze=("distogram_head", "confidence_head", "template_embedder"),
    )
)
