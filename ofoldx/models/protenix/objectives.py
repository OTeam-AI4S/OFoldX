r"""
Protenix training contract: the shared EDM coordinate-denoising objective, covering the base model and its
conditioning variants (they share ``ProtenixModel.forward`` and ``ProtenixOutput``). DDP
``find_unused_parameters=True`` handles the conditioning modules that a given training set does not exercise.

Each variant carries its own model/config class (``protenix_template`` -> ``ProtenixTemplateModel`` with the
template embedder on, ``protenix_structure_condition``, ``protenix_constraint``), so a fresh build from the
registered architecture already constructs the correct graph -- no training-side config patching.
"""

from __future__ import annotations

from ...modules.diffusion.loss import denoising_criterion_factory
from ..training import TrainingContract

TRAINING = TrainingContract(
    architectures=("protenix", "protenix_template", "protenix_structure_condition", "protenix_constraint"),
    criterion=denoising_criterion_factory,
    strategy="denoising",
)
