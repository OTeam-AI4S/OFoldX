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

from ofoldx.modules.backend import categorical_msa_projection


class TestCategoricalMsaProjection:
    def test_equation(self) -> None:
        tokens = torch.tensor([[[0, 2, 4], [3, 1, 0], [4, 3, 2]]])
        continuous = torch.randn(1, 3, 3, 2, requires_grad=True)
        weight = torch.randn(4, 7, requires_grad=True)
        expected_continuous = continuous.detach().clone().requires_grad_()
        expected_weight = weight.detach().clone().requires_grad_()
        features = torch.cat((F.one_hot(tokens, num_classes=5), expected_continuous), dim=-1)

        expected = F.linear(features, expected_weight)
        actual = categorical_msa_projection(tokens, continuous, weight, backend="eager")
        expected.square().sum().backward()
        actual.square().sum().backward()

        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(continuous.grad, expected_continuous.grad)
        torch.testing.assert_close(weight.grad, expected_weight.grad)

    def test_shape(self) -> None:
        with pytest.raises(ValueError, match="continuous_features must have shape"):
            categorical_msa_projection(
                torch.zeros(2, 3, dtype=torch.long),
                torch.zeros(1, 2, 3, 2),
                torch.zeros(4, 7),
                backend="eager",
            )
