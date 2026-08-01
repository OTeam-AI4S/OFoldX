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

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from ofoldx.modules.embedding import FactorizedPairLinear


class TestFactorizedPairLinear:
    @pytest.mark.parametrize("backend", ["eager", "auto"])
    def test_linear_contract(self, backend: str) -> None:
        torch.manual_seed(0)
        pair_prefix_size = 3
        pair = torch.randn(2, 4, 4, 5, dtype=torch.float64, requires_grad=True)
        token = torch.randn(2, 4, 3, dtype=torch.float64, requires_grad=True)
        projection = FactorizedPairLinear(
            pair_size=5,
            token_size=3,
            output_size=7,
            pair_prefix_size=pair_prefix_size,
            bias=False,
            backend=backend,
        ).double()

        prefix = pair[..., :pair_prefix_size]
        suffix = pair[..., pair_prefix_size:]
        left = token.unsqueeze(-2).expand(-1, -1, token.shape[-2], -1)
        right = token.unsqueeze(-3).expand(-1, token.shape[-2], -1, -1)
        packed = torch.cat((prefix, left, right, suffix), dim=-1)

        actual = projection(pair, token)
        expected = F.linear(packed, projection.weight)
        torch.testing.assert_close(actual, expected)

        grad_output = torch.randn_like(actual)
        actual_grads = torch.autograd.grad(actual, (pair, token, projection.weight), grad_output)
        expected_grads = torch.autograd.grad(expected, (pair, token, projection.weight), grad_output)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad)

        assert tuple(projection.state_dict()) == ("weight",)
        assert projection.weight.shape == (7, 11)
