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

r"""Gaussian radial-basis embeddings with deterministic evenly spaced centers."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..functional import gaussian_radial_basis


class GaussianRadialBasis(nn.Module):
    r"""Embed scalar distances using a fixed grid of Gaussian centers.

    Args:
        num_bins: Number of centers spanning the inclusive bounds.
        lower_bound: First center, in the input distance units.
        upper_bound: Last center, in the input distance units.
        width: Gaussian width; the default is the range divided by bin count.
    """

    centers: Tensor

    def __init__(
        self,
        num_bins: int = 16,
        lower_bound: float = 2.0,
        upper_bound: float = 22.0,
        *,
        width: float | None = None,
    ) -> None:
        super().__init__()
        self.width = (upper_bound - lower_bound) / num_bins if width is None else width
        device = torch.get_default_device()
        # Deterministic nonpersistent constants must remain real when learned
        # parameters are constructed on meta: checkpoint loading never fills them.
        if device.type == "meta":
            device = torch.device("cpu")
        self.register_buffer(
            "centers", torch.linspace(lower_bound, upper_bound, num_bins, device=device), persistent=False
        )

    def forward(self, distances: Tensor) -> Tensor:
        r"""Return ``[..., num_bins]`` features for scalar distances ``[...]``."""

        return gaussian_radial_basis(distances, self.centers, self.width)
