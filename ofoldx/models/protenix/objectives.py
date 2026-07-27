r"""
Protenix training contract: the shared EDM coordinate-denoising objective, covering the base model and its
conditioning variants (they share ``ProtenixModel.forward`` and ``ProtenixOutput``). DDP
``find_unused_parameters=True`` handles the conditioning modules that a given training set does not exercise.

``protenix_template`` reuses ``ProtenixModel`` / ``ProtenixConfig`` (whose ``template`` defaults to ``None``)
but pairs with a template-producing Processor, so a fresh build must enable the template embedder or the
template features would be silently ignored; ``fresh_configs`` supplies that graph. The other variants carry
their own model/config classes, so their base defaults already match.
"""

from __future__ import annotations

from ...modules.diffusion.loss import denoising_criterion_factory
from ..training import TrainingContract
from .config import ProtenixConfig, ProtenixTemplateConfig

TRAINING = TrainingContract(
    architectures=("protenix", "protenix_template", "protenix_structure_condition", "protenix_constraint"),
    criterion=denoising_criterion_factory,
    strategy="denoising",
    fresh_configs={
        # Overrides win over the template default, so an explicit ``model.template`` still applies.
        "protenix_template": lambda **overrides: ProtenixConfig(**{"template": ProtenixTemplateConfig(), **overrides}),
    },
)
