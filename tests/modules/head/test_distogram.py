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

import torch
import torch.nn.functional as F

from ofoldx.modules.head import DistogramHead


class TestDistogramHead:
    def test_equation(self) -> None:
        head = DistogramHead(pair_size=2, num_bins=2)
        weight = torch.tensor([[1.0, 2.0], [-1.0, 3.0]])
        bias = torch.tensor([0.5, -2.0])
        pair = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]])
        head.projection.weight = torch.nn.Parameter(weight)
        head.projection.bias = torch.nn.Parameter(bias)

        expected = F.linear(pair, weight) + F.linear(pair.transpose(-2, -3), weight) + 2 * bias

        torch.testing.assert_close(head(pair), expected)

    def test_pair_symmetrized_source_bias_mapping(self) -> None:
        head = DistogramHead(pair_size=2, num_bins=2)
        weight = torch.tensor([[1.0, 2.0], [-1.0, 3.0]])
        bias = torch.tensor([0.5, -2.0])
        pair = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]])
        head.projection.weight = torch.nn.Parameter(weight)
        head.projection.bias = torch.nn.Parameter(bias / 2)

        symmetrized_pair = pair + pair.transpose(-2, -3)
        expected = F.linear(symmetrized_pair, weight, bias)

        torch.testing.assert_close(head(pair), expected)
