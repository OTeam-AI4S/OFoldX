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
Pairwise distance prediction.
"""

from torch import Tensor, nn


class DistogramHead(nn.Module):
    r"""
    Project pair representations and add the transposed distance-bin logits.

    The projection's bias contributes twice. A checkpoint that projects the
    symmetrized pair representation maps its source bias to half this value.
    """

    def __init__(self, pair_size: int = 128, num_bins: int = 64, bias: bool = True) -> None:
        super().__init__()
        self.projection = nn.Linear(pair_size, num_bins, bias=bias)

    def forward(self, pair: Tensor) -> Tensor:
        logits = self.projection(pair)
        return logits + logits.transpose(-2, -3)
