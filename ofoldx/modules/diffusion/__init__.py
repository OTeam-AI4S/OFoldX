# OFoldX
# Copyright (C) 2026-Present  Valhalla Technology

# This file is part of OFoldX.

# OFoldX is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# OFoldX is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# For additional terms and clarifications, please refer to our License FAQ at:
# <https://ofoldx.valtech.ai/about/license-faq/>.

r"""
Reusable diffusion schedules, samplers, and conditioning components.
"""

from __future__ import annotations

from .conditioning import PairwiseConditioning, SingleConditioning, condition_pair
from .ddpm import ddpm_forward_step, ddpm_posterior
from .edm import EdmPreconditioner, edm_loss_weight
from .igso3 import (
    calculate_igso3,
    igso3_angle_density,
    igso3_expansion,
    igso3_log_derivative,
    so3_exp_map,
    so3_log_map,
    so3_project_to_rotation,
)
from .loss import edm_denoising_loss, resolved_atom_mask, smooth_lddt_loss, smooth_lddt_loss_with_counts
from .sampling import (
    CoordinateDenoiseContext,
    CoordinateEdmSampler,
    HeunCoordinateEdmSampler,
    UnaugmentedCoordinateEdmSampler,
    denoise_coordinates,
)
from .schedule import EdmScheduler, karras_noise_schedule
from .types import CoordinateDenoiseFn, CoordinateDenoisingInput, CoordinateDenoisingOutput
