r"""
Protenix training contract: the shared EDM coordinate-denoising objective, covering the base model and its
conditioning variants (they share ``ProtenixModel.forward`` and ``ProtenixOutput``). DDP ``static_graph``
handles the conditioning modules that a given training set does not exercise.
"""

from __future__ import annotations

from ...modules.diffusion.loss import denoising_criterion_factory
from ..training import TrainingContract

TRAINING = TrainingContract(
    architectures=("protenix", "protenix_template", "protenix_structure_condition", "protenix_constraint"),
    criterion=denoising_criterion_factory,
    strategy="denoising",
)
