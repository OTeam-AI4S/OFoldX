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
from torch._dynamo.testing import CompileCounter

from ofoldx.modules.attention import TriangleAttention


def test_triangle_bias_couples_rows() -> None:
    # Regression: the shared bias b_{jk}=Linear(z_{jk}) must couple untouched rows at query column
    # == the perturbed row; the earlier row-tied bias [row, key] left other rows invariant.
    torch.manual_seed(0)
    module = TriangleAttention(pair_size=8, head_size=4, num_attention_heads=2, backend="sdpa").eval()
    pair = torch.randn(1, 5, 5, 8)

    with torch.no_grad():
        baseline = module(pair)
        perturbed = pair.clone()
        perturbed[0, 0] = torch.randn(5, 8)  # a direction change survives LayerNorm
        updated = module(perturbed)

    # Row 2's q/k/v are untouched, so the shared bias couples it to row 0 only at query column 0.
    assert not torch.allclose(baseline[0, 2], updated[0, 2], atol=1e-6)
    torch.testing.assert_close(baseline[0, 2, 1:], updated[0, 2, 1:], atol=1e-6, rtol=0)


@pytest.mark.parametrize("bias_pair_dims", [(-3, -2), (-2, -3)])
@pytest.mark.parametrize("extra_prefix", [(), (3,)])
@pytest.mark.parametrize("projection_bias", [False, True])
def test_triangle_bias_index_map_complete_equation_and_all_gradients(
    bias_pair_dims: tuple[int, int], extra_prefix: tuple[int, ...], projection_bias: bool
) -> None:
    module = TriangleAttention(
        pair_size=4,
        head_size=3,
        num_attention_heads=2,
        output_bias=projection_bias,
        gate_bias=projection_bias,
        bias_pair_dims=bias_pair_dims,
        backend="eager",
    ).double()
    actual_input = torch.randn(2, *extra_prefix, 4, 4, 4, dtype=torch.float64, requires_grad=True)
    reference_input = actual_input.detach().clone().requires_grad_()
    references = {name: value.detach().clone().requires_grad_() for name, value in module.named_parameters()}
    mask = torch.ones(2, *extra_prefix, 4, 4, dtype=torch.bool)
    mask[..., 0, :] = False
    mask[..., 1, 2] = False
    actual = module(actual_input, mask)
    normalized = F.layer_norm(
        reference_input, (4,), references["pair_norm.weight"], references["pair_norm.bias"], module.pair_norm.eps
    )
    projected = {}
    prefix = "attention.projection."
    for name in ("query", "key", "value"):
        projected[name] = (
            F.linear(
                normalized,
                references[f"{prefix}{name}_projection.weight"],
                references.get(f"{prefix}{name}_projection.bias"),
            )
            .unflatten(-1, (2, 3))
            .transpose(-2, -3)
        )
    bias_source = normalized if bias_pair_dims == (-3, -2) else normalized.transpose(-3, -2)
    bias = F.linear(bias_source, references["bias_projection.weight"]).movedim(-1, -3).unsqueeze(-4)
    logits = projected["query"] @ projected["key"].transpose(-1, -2) / (3**0.5) + bias
    selected = mask[..., None, None, :]
    present = selected.any(-1, keepdim=True)
    logits = logits.masked_fill(~selected, -torch.inf)
    probabilities = torch.where(present, logits, 0.0).softmax(-1)
    probabilities = probabilities.masked_fill(~present, 0.0)
    context = (probabilities @ projected["value"]).transpose(-2, -3).flatten(-2)
    gate = F.linear(
        normalized, references[f"{prefix}gate_projection.weight"], references.get(f"{prefix}gate_projection.bias")
    ).sigmoid()
    expected = F.linear(
        context * gate,
        references[f"{prefix}output_projection.weight"],
        references.get(f"{prefix}output_projection.bias"),
    )
    torch.testing.assert_close(actual, expected, rtol=2e-10, atol=2e-11)
    cotangent = torch.randn_like(actual)
    actual_gradients = torch.autograd.grad(actual, (actual_input, *module.parameters()), cotangent)
    expected_gradients = torch.autograd.grad(expected, (reference_input, *references.values()), cotangent)
    for observed, wanted in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(observed, wanted, rtol=2e-10, atol=2e-11)


def test_triangle_bias_index_map_preserves_qkv_when_zero_query_key() -> None:
    ordinary = TriangleAttention(pair_size=4, head_size=2, num_attention_heads=1, backend="eager").double()
    transposed = TriangleAttention(
        pair_size=4, head_size=2, num_attention_heads=1, bias_pair_dims=(-2, -3), backend="eager"
    ).double()
    with torch.no_grad():
        projection = ordinary.attention.projection
        projection.query_projection.weight.zero_()
        projection.key_projection.weight.zero_()
        ordinary.bias_projection.weight.copy_(torch.tensor([[1.0, -0.4, 0.8, -0.6]], dtype=torch.float64))
        projection.value_projection.weight.copy_(
            torch.tensor([[1.0, 0.0, -0.2, 0.4], [0.1, -0.3, 0.7, 1.0]], dtype=torch.float64)
        )
        projection.gate_projection.weight.zero_()
        projection.output_projection.weight.copy_(
            torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.3, 0.2], [-0.2, 0.5]], dtype=torch.float64)
        )
    transposed.load_state_dict(ordinary.state_dict(), strict=True)
    pair = torch.tensor(
        [[[[0.0, 1.0, 0.3, -0.2], [1.0, -0.7, 0.2, 0.9]], [[-0.6, 0.4, 1.2, -0.1], [0.7, -0.2, -0.8, 0.5]]]],
        dtype=torch.float64,
    )
    normalized = ordinary.pair_norm(pair)
    original_bias = ordinary.bias_projection(normalized)
    mapped_bias = transposed.bias_projection(normalized.transpose(-3, -2))
    torch.testing.assert_close(mapped_bias, original_bias.transpose(-3, -2), rtol=2e-12, atol=2e-12)
    assert not torch.allclose(ordinary(pair), transposed(pair))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA OOps kernels")
def test_oops_nested_ending_node_reuses_one_dynamic_fullgraph() -> None:
    pytest.importorskip("oops")
    NestedTensor = pytest.importorskip("danling.tensors").NestedTensor
    device = torch.device("cuda")
    module = TriangleAttention(
        pair_size=8,
        head_size=4,
        num_attention_heads=2,
        backend="oops",
    ).to(device=device, dtype=torch.bfloat16)
    counter = CompileCounter()
    compiled = torch.compile(
        lambda pair, mask: module(pair.transpose(1, 2), mask.transpose(1, 2)).transpose(1, 2).concat,
        backend=counter,
        fullgraph=True,
        dynamic=True,
    )

    for lengths in ((2, 3), (3, 5)):
        pair_template = NestedTensor(
            [torch.empty(length, length, 8) for length in lengths],
            ragged_dims=(0, 1),
        )
        pair_values = torch.randn(
            pair_template.concat.shape,
            device=device,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        pair = pair_template.packed_like(pair_values)
        mask = NestedTensor(
            [torch.ones(length, length, device=device, dtype=torch.bool) for length in lengths],
            ragged_dims=(0, 1),
        )
        with torch.no_grad():
            expected = torch.cat(
                tuple(
                    module(element.transpose(0, 1), element_mask.transpose(0, 1)).transpose(0, 1).reshape(-1, 8)
                    for element, element_mask in zip(torch.unbind(pair), torch.unbind(mask), strict=True)
                )
            )
        actual = compiled(pair, mask)
        torch.testing.assert_close(actual, expected, rtol=5e-2, atol=5e-2)
        pair_gradient, gate_gradient = torch.autograd.grad(
            actual.sum(),
            (pair_values, module.attention.projection.gate_projection.weight),
        )
        assert pair_gradient is not None
        assert gate_gradient is not None
    assert counter.frame_count == 1
