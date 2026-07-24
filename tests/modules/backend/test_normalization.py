from __future__ import annotations

import pytest
import torch
from torch import nn

from ofoldx.modules.backend import LayerNorm


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
