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

from functools import partial

import pytest
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ofoldx.modules.pair import TriangleMultiplication, TrianglePairUpdateStack


def _stack() -> TrianglePairUpdateStack:
    return TrianglePairUpdateStack(
        num_blocks=2,
        pair_size=4,
        triangle_multiplication_hidden_size=4,
        triangle_attention_head_size=2,
        num_triangle_attention_heads=2,
        dropout=0.0,
        backend="eager",
    )


class TestTrianglePairUpdateStack:
    def test_shape_and_gradient(self) -> None:
        pair = torch.randn(1, 3, 3, 4, requires_grad=True)

        output = _stack()(pair)
        output.sum().backward()

        assert output.shape == pair.shape
        assert pair.grad is not None

    def test_gradient_checkpointing(self) -> None:
        eager = _stack()
        checkpointed = _stack()
        checkpointed.load_state_dict(eager.state_dict())
        checkpointed.gradient_checkpointing = True
        checkpointed._gradient_checkpointing_func = partial(checkpoint, use_reentrant=False)
        eager_pair = torch.randn(1, 3, 3, 4, requires_grad=True)
        checkpointed_pair = eager_pair.detach().clone().requires_grad_()

        eager_output = eager(eager_pair)
        checkpointed_output = checkpointed(checkpointed_pair)
        eager_output.sum().backward()
        checkpointed_output.sum().backward()

        torch.testing.assert_close(checkpointed_output, eager_output)
        torch.testing.assert_close(checkpointed_pair.grad, eager_pair.grad)


@pytest.mark.parametrize(
    ("direction", "equation"),
    [("outgoing", "...ikh,...jkh->...ijh"), ("incoming", "...kih,...kjh->...ijh")],
)
def test_triangle_multiplication(direction: str, equation: str) -> None:
    torch.manual_seed(0)
    module = TriangleMultiplication(
        pair_size=4,
        hidden_size=3,
        direction=direction,
        projection_bias=True,
        backend="eager",
    )
    pair = torch.randn(1, 3, 3, 4, requires_grad=True)
    pair_mask = torch.tensor([[[True, True, False], [True, False, True], [False, True, True]]])

    output = module(pair, pair_mask)

    normalized = F.layer_norm(
        pair,
        (4,),
        module.input_norm.weight,
        module.input_norm.bias,
        module.input_norm.eps,
    )
    left = F.linear(normalized, module.left_projection.weight, module.left_projection.bias) * torch.sigmoid(
        F.linear(normalized, module.left_gate_projection.weight, module.left_gate_projection.bias)
    )
    right = F.linear(normalized, module.right_projection.weight, module.right_projection.bias) * torch.sigmoid(
        F.linear(normalized, module.right_gate_projection.weight, module.right_gate_projection.bias)
    )
    factor_mask = pair_mask.unsqueeze(-1)
    contracted = torch.einsum(equation, left * factor_mask, right * factor_mask)
    contracted = F.layer_norm(
        contracted,
        (3,),
        module.output_norm.weight,
        module.output_norm.bias,
        module.output_norm.eps,
    )
    expected = F.linear(contracted, module.output_projection.weight, module.output_projection.bias) * torch.sigmoid(
        F.linear(normalized, module.gate_projection.weight, module.gate_projection.bias)
    )

    torch.testing.assert_close(output, expected)
    inputs = (pair, *module.parameters())
    actual_grads = torch.autograd.grad(output.sum(), inputs, retain_graph=True)
    expected_grads = torch.autograd.grad(expected.sum(), inputs)
    for actual, reference in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual, reference)
