from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from ofoldx.modules.backend import pair_norm_projection


def test_source_precision() -> None:
    torch.manual_seed(0)
    pair = torch.randn(2, 3, 5, 8, dtype=torch.bfloat16)
    norm_weight = torch.randn(8, dtype=torch.float32)
    norm_bias = torch.randn(8, dtype=torch.float32)
    projection_weight = torch.randn(3, 8, dtype=torch.bfloat16)

    actual = pair_norm_projection(
        pair,
        norm_weight,
        norm_bias,
        projection_weight,
        output_layout="head_major",
        backend="eager",
    )

    pair_float = pair.float()
    mean = pair_float.mean(dim=-1, keepdim=True)
    variance = pair_float.square().mean(dim=-1, keepdim=True) - mean.square()
    normalized = (norm_weight * torch.rsqrt(variance + 1e-5) * (pair_float - mean) + norm_bias).to(pair.dtype)
    expected = F.linear(normalized, projection_weight).movedim(-1, -3).contiguous()

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert actual.is_contiguous()


def test_public_shape_domain() -> None:
    pair = torch.randn(3, 8)
    norm_weight = torch.randn(8)

    with pytest.raises(ValueError, match="pair must have shape"):
        pair_norm_projection(pair, norm_weight, None, torch.randn(2, 8), backend="eager")
    with pytest.raises(ValueError, match="H > 0"):
        pair_norm_projection(pair[None], norm_weight, None, torch.empty(0, 8), backend="eager")
