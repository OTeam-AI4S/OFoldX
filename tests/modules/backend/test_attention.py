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
from danling.tensors import NestedTensor

from ofoldx.modules.backend import attention as attention_backend
from ofoldx.modules.backend.attention import projected_attention


@pytest.mark.parametrize("sampled", [False, True])
@pytest.mark.parametrize("query_specific", [False, True])
def test_oops_mask_lowering(monkeypatch, sampled, query_specific):
    def provider(query, key, value, *, mask, pair_bias, scale):
        prefix_rank = query.dim() - 3
        assert mask.dim() in (prefix_rank + 1, prefix_rank + 2)
        if mask.dim() == prefix_rank + 1:
            mask = mask.unsqueeze(-2)
        return torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=mask.unsqueeze(-3), scale=scale
        )

    monkeypatch.setattr(attention_backend, "_oops_pair_bias_attention", provider)
    prefix = (2, 3) if sampled else (2,)
    query = torch.randn(*prefix, 2, 3, 4, dtype=torch.float64, requires_grad=True)
    key = torch.randn(*prefix, 2, 4, 4, dtype=torch.float64, requires_grad=True)
    value = torch.randn_like(key, requires_grad=True)
    mask = torch.ones(*prefix, 3 if query_specific else 1, 4, dtype=torch.bool)
    mask[..., 0, -1] = False
    actual = projected_attention(query, key, value, attention_mask=mask, backend="oops", scale=1.0)
    expected = torch.nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask=mask.unsqueeze(-3), scale=1.0
    )
    torch.testing.assert_close(actual, expected)
    actual_gradients = torch.autograd.grad(actual.square().sum(), (query, key, value), retain_graph=True)
    expected_gradients = torch.autograd.grad(expected.square().sum(), (query, key, value))
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual_gradient, expected_gradient)


@pytest.mark.parametrize("backend", ["eager", "sdpa"])
def test_nested_key_mask_matches_dense(backend):
    queries = [torch.randn(2, count, 3, dtype=torch.float64, requires_grad=True) for count in (2, 3)]
    keys = [torch.randn(2, count, 3, dtype=torch.float64, requires_grad=True) for count in (4, 2)]
    values = [torch.randn_like(key, requires_grad=True) for key in keys]
    masks = [torch.tensor([True, False, True, False]), torch.tensor([False, False])]
    output = projected_attention(
        NestedTensor(queries, ragged_dims=(1,)),
        NestedTensor(keys, ragged_dims=(1,)),
        NestedTensor(values, ragged_dims=(1,)),
        attention_mask=NestedTensor(masks, ragged_dims=(0,)),
        backend=backend,
    )
    expected = [
        projected_attention(q, k, v, attention_mask=mask, backend=backend)
        for q, k, v, mask in zip(queries, keys, values, masks, strict=True)
    ]
    for actual, reference in zip(output.unbind(0), expected, strict=True):
        torch.testing.assert_close(actual, reference)
    inputs = (*queries, *keys, *values)
    actual_gradients = torch.autograd.grad(output.concat.square().sum(), inputs, retain_graph=True)
    expected_gradients = torch.autograd.grad(sum(part.square().sum() for part in expected), inputs)
    for actual, reference in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual, reference)


def test_triangle_multiple_prefixes(monkeypatch):
    from ofoldx.modules.attention.multihead import MultiheadAttentionProjection

    def provider(query, key, value, mask, bias, *, scale):
        assert query.dim() == 5
        assert mask.shape == query.shape[:3]
        q, k, v = (tensor.transpose(-3, -2) for tensor in (query, key, value))
        output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=bias + torch.where(mask[:, :, None, None, :], 0.0, -torch.inf), scale=scale
        )
        return output.transpose(-3, -2)

    monkeypatch.setattr(attention_backend, "_oops_triangle_attention", provider)
    projection = MultiheadAttentionProjection(hidden_size=4, num_attention_heads=2, head_size=2).double()
    pair = torch.randn(2, 3, 4, 4, 4, dtype=torch.float64, requires_grad=True)
    bias = torch.randn(2, 3, 1, 2, 4, 4, dtype=torch.float64, requires_grad=True)
    mask = torch.ones(2, 3, 4, 4, dtype=torch.bool)
    actual = attention_backend.triangle_attention(pair, bias, mask, projection=projection, backend="oops")
    expected = attention_backend.triangle_attention(pair, bias, mask, projection=projection, backend="eager")
    torch.testing.assert_close(actual, expected)
    operands = (pair, bias, *projection.parameters())
    actual_gradients = torch.autograd.grad(actual.square().sum(), operands, retain_graph=True)
    expected_gradients = torch.autograd.grad(expected.square().sum(), operands)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual_gradient, expected_gradient)
