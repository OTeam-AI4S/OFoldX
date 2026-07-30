from __future__ import annotations

import pytest
import torch
from torch import nn

from ofoldx.modules.backend import LayerNorm
from ofoldx.modules.backend._oops import OOPS_AVAILABLE


@pytest.mark.parametrize("elementwise_affine", [False, True])
@pytest.mark.parametrize("bias", [False, True])
def test_layer_norm_matches_torch(elementwise_affine: bool, bias: bool) -> None:
    reference = nn.LayerNorm(8, elementwise_affine=elementwise_affine, bias=bias)
    actual = LayerNorm(8, elementwise_affine=elementwise_affine, bias=bias, backend="eager")
    actual.load_state_dict(reference.state_dict())
    hidden = torch.randn(2, 3, 8)

    torch.testing.assert_close(actual(hidden), reference(hidden))
    assert tuple(actual.state_dict()) == tuple(reference.state_dict())


@pytest.mark.parametrize("backend", ["sdpa", "cueq", "apex", "deepspeed"])
def test_provider_uses_eager_normalization_when_not_supported(backend: str) -> None:
    norm = LayerNorm(8, backend="eager")
    hidden = torch.randn(2, 3, 8)
    expected = norm(hidden)
    norm.backend = backend

    torch.testing.assert_close(norm(hidden), expected)


@pytest.mark.skipif(not OOPS_AVAILABLE, reason="OOps is not installed")
def test_oops_layer_norm_preserves_mixed_precision_affine_contract() -> None:
    norm = LayerNorm(8, dtype=torch.float32, backend="oops")
    hidden = torch.randn(2, 3, 8, dtype=torch.bfloat16, requires_grad=True)

    output = norm(hidden)
    output.float().square().mean().backward()

    assert output.dtype == torch.bfloat16
    assert hidden.grad is not None
    assert hidden.grad.dtype == torch.bfloat16
    assert norm.weight.grad is not None
    assert norm.weight.grad.dtype == torch.float32
    assert torch.isfinite(output).all()
    assert torch.isfinite(hidden.grad).all()
    assert torch.isfinite(norm.weight.grad).all()
