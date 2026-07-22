r"""
Shared diffusion building blocks.

Gathers the reusable EDM diffusion primitives: the noise
[`EdmScheduler`][ofoldx.modules.diffusion.schedule.EdmScheduler] and
[`EdmPreconditioner`][ofoldx.modules.diffusion.edm.EdmPreconditioner], the
[`CoordinateEdmSampler`][ofoldx.modules.diffusion.sampling.CoordinateEdmSampler] reverse-diffusion loop with its
context and denoising input, the
[`PairwiseConditioning`][ofoldx.modules.diffusion.conditioning.PairwiseConditioning] module, and the
[`condition_single`][ofoldx.modules.diffusion.conditioning.condition_single] /
[`condition_pair`][ofoldx.modules.diffusion.conditioning.condition_pair] conditioning helpers.
"""

from __future__ import annotations

from .conditioning import PairwiseConditioning, condition_pair, condition_single
from .edm import EdmPreconditioner
from .sampling import CoordinateDenoiseContext, CoordinateEdmSampler
from .schedule import EdmScheduler
from .types import CoordinateDenoiseFn, CoordinateDenoisingInput
