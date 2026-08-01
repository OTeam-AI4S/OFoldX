from __future__ import annotations

import torch
import torch.nn.functional as F

from ofoldx.modules.backend.transition import transition


def test_mixed_precision_equation() -> None:
    hidden = torch.randn(2, 3, 8, dtype=torch.bfloat16, requires_grad=True)
    norm_weight = torch.randn(8, requires_grad=True)
    norm_bias = torch.randn(8, requires_grad=True)
    gate_weight = torch.randn(16, 8, dtype=torch.bfloat16, requires_grad=True)
    up_weight = torch.randn(16, 8, dtype=torch.bfloat16, requires_grad=True)
    down_weight = torch.randn(8, 16, dtype=torch.bfloat16, requires_grad=True)
    tensors = (hidden, norm_weight, norm_bias, gate_weight, up_weight, down_weight)

    output = transition(
        hidden,
        input_norm_weight=norm_weight,
        input_norm_bias=norm_bias,
        gate_weight=gate_weight,
        up_weight=up_weight,
        down_weight=down_weight,
        input_norm_eps=1e-5,
        backend="eager",
    )

    normalized = hidden.float()
    mean = normalized.mean(dim=-1, keepdim=True)
    variance = normalized.square().mean(dim=-1, keepdim=True) - mean.square()
    normalized = (norm_weight * torch.rsqrt(variance + 1e-5) * (normalized - mean) + norm_bias).to(hidden.dtype)
    gate = F.linear(normalized, gate_weight)
    value = F.linear(normalized, up_weight)
    expected = F.linear((F.silu(gate.float()) * value.float()).to(hidden.dtype), down_weight)

    torch.testing.assert_close(output, expected)
    actual_grads = torch.autograd.grad(output.float().sum(), tensors, retain_graph=True)
    expected_grads = torch.autograd.grad(expected.float().sum(), tensors)
    for actual, reference in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual, reference)
