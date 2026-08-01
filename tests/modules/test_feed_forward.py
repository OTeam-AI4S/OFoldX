from __future__ import annotations

import torch
import torch.nn.functional as F

from ofoldx.modules.feed_forward import SwiGluFeedForward


class TestSwiGluFeedForward:
    def test_forward_and_gradients(self) -> None:
        module = SwiGluFeedForward(3, 5, 2, bias=True)
        hidden = torch.randn(4, 3, requires_grad=True)

        actual = module(hidden)
        expected = F.linear(
            F.silu(F.linear(hidden, module.gate_projection.weight, module.gate_projection.bias))
            * F.linear(hidden, module.value_projection.weight, module.value_projection.bias),
            module.output_projection.weight,
            module.output_projection.bias,
        )
        torch.testing.assert_close(actual, expected)

        actual.sum().backward()
        assert hidden.grad is not None
