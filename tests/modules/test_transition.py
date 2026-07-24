from __future__ import annotations

import torch

from ofoldx.modules.transition import ConditionedTransition, Transition


class TestTransition:
    def test_returns_feed_forward_update(self) -> None:
        transition = Transition(hidden_size=8, expansion=2, operator_backend="eager")
        hidden = torch.randn(2, 3, 8, requires_grad=True)

        output = transition(hidden)
        expected = transition.feed_forward(transition.input_norm(hidden))

        torch.testing.assert_close(output, expected)
        output.sum().backward()
        assert hidden.grad is not None


class TestConditionedTransition:
    def test_conditions_norm_and_output_gate(self) -> None:
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
