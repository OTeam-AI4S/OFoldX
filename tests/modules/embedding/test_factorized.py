from __future__ import annotations

import torch
import torch.nn.functional as F

from ofoldx.modules.embedding import FactorizedPairLinear


class TestFactorizedPairLinear:
    def test_linear_contract(self) -> None:
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
            backend="eager",
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
