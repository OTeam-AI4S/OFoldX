r"""
Boltz-2 training objective.

Boltz-2 noises the target internally and exposes its denoised coordinates alongside the augmented target
and per-sample noise level, so the shared EDM coordinate-denoising loss grounds directly against the
model output. The batch is DanLing's ``{"input", "target", "loss_normalizer"}`` from
``collate_denoising``.
"""

from __future__ import annotations

from ...modules.collate import collate_denoising as collate
from ...modules.diffusion.loss import denoising_criterion as criterion

__all__ = ["collate", "criterion"]
