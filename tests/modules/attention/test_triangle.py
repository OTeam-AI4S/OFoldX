from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch._dynamo.testing import CompileCounter

from ofoldx.modules.attention import TriangleAttention, TriangleMultiplication


def _nested_guard():
    tensors = pytest.importorskip("danling.tensors")
    return tensors.nested_execution_guard(
        forbid_iteration=True,
        forbid_storage_map=True,
        forbid_eager_fallback=True,
        forbid_padded_materialization=True,
        forbid_dense_repack=True,
    )


def test_triangle_bias_couples_rows() -> None:
    # Regression: the shared bias b_{jk}=Linear(z_{jk}) must couple untouched rows at query column
    # == the perturbed row; the earlier row-tied bias [row, key] left other rows invariant.
    torch.manual_seed(0)
    module = TriangleAttention(pair_size=8, head_dim=4, num_attention_heads=2, backend="sdpa").eval()
    pair = torch.randn(1, 5, 5, 8)

    with torch.no_grad():
        baseline = module(pair)
        perturbed = pair.clone()
        perturbed[0, 0] = torch.randn(5, 8)  # a direction change survives LayerNorm
        updated = module(perturbed)

    # Row 2's q/k/v are untouched, so the shared bias couples it to row 0 only at query column 0.
    assert not torch.allclose(baseline[0, 2], updated[0, 2], atol=1e-6)
    torch.testing.assert_close(baseline[0, 2, 1:], updated[0, 2, 1:], atol=1e-6, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA OOps kernels")
def test_oops_nested_ending_node_reuses_one_dynamic_fullgraph() -> None:
    pytest.importorskip("oops")
    NestedTensor = pytest.importorskip("danling.tensors").NestedTensor
    device = torch.device("cuda")
    module = TriangleAttention(
        pair_size=8,
        head_dim=4,
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
        with _nested_guard():
            actual = compiled(pair, mask)
        torch.testing.assert_close(actual, expected, rtol=5e-2, atol=5e-2)
        pair_gradient, gate_gradient = torch.autograd.grad(
            actual.sum(),
            (pair_values, module.attention.projection.gate_projection.weight),
        )
        assert pair_gradient is not None
        assert gate_gradient is not None
    assert counter.frame_count == 1


@pytest.mark.parametrize("outgoing", [False, True])
def test_triangle_multiplication(outgoing: bool) -> None:
    torch.manual_seed(0)
    module = TriangleMultiplication(
        pair_size=4,
        hidden_size=3,
        outgoing=outgoing,
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
    equation = "...ikh,...jkh->...ijh" if outgoing else "...kih,...kjh->...ijh"
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
