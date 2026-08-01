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

import torch

from ofoldx.modules.transition import ConditionedTransition, Transition


class TestTransition:
    def test_returns_feed_forward_update(self) -> None:
        transition = Transition(hidden_size=8, expansion=2, backend="eager")
        hidden = torch.randn(2, 3, 8, requires_grad=True)

        output = transition(hidden)
        expected = transition.feed_forward(transition.input_norm(hidden))

        torch.testing.assert_close(output, expected)
        output.sum().backward()
        assert hidden.grad is not None


class TestConditionedTransition:
    def test_equation(self) -> None:
        transition = ConditionedTransition(hidden_size=8, conditioning_size=4, expansion=2)
        hidden = torch.randn(2, 3, 8, requires_grad=True)
        conditioning = torch.randn(2, 3, 4, requires_grad=True)

        output = transition(hidden, conditioning)
        normalized = transition.conditioned_norm(hidden, conditioning)
        expected = transition.feed_forward(normalized) * torch.sigmoid(
            transition.conditioning_gate_projection(conditioning)
        )

        torch.testing.assert_close(output, expected)
        output.sum().backward()
        assert hidden.grad is not None
        assert conditioning.grad is not None
