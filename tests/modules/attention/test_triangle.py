from __future__ import annotations

import torch

from ofoldx.modules.attention import TriangleAttention


def test_triangle_bias_couples_rows() -> None:
    # Regression: the shared bias b_{jk}=Linear(z_{jk}) must couple untouched rows at query column
    # == the perturbed row; the earlier row-tied bias [row, key] left other rows invariant.
    torch.manual_seed(0)
    module = TriangleAttention(pair_size=8, head_dim=4, num_attention_heads=2, operator_backend="sdpa").eval()
    pair = torch.randn(1, 5, 5, 8)

    with torch.no_grad():
        baseline = module(pair)
        perturbed = pair.clone()
        perturbed[0, 0] = torch.randn(5, 8)  # a direction change survives LayerNorm
        updated = module(perturbed)

    # Row 2's q/k/v are untouched, so the shared bias couples it to row 0 only at query column 0.
    assert not torch.allclose(baseline[0, 2], updated[0, 2], atol=1e-6)
    torch.testing.assert_close(baseline[0, 2, 1:], updated[0, 2, 1:], atol=1e-6, rtol=0)
