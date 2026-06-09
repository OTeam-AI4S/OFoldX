from __future__ import annotations

import pytest
import torch

from ofoldx.modules.attention import GatedMultiheadAttention, MultiheadAttentionProjection


@pytest.mark.parametrize(
    "build",
    [
        lambda: MultiheadAttentionProjection(hidden_size=8, num_attention_heads=3),
        lambda: GatedMultiheadAttention(hidden_size=8, num_attention_heads=2, operator_backend="bogus"),
    ],
)
def test_invalid_inputs(build) -> None:
    with pytest.raises(ValueError):
        build()


def test_projection_bias_switches_match_af2_attention_layout() -> None:
    projection = MultiheadAttentionProjection(
        hidden_size=8,
        num_attention_heads=2,
        query_bias=False,
        output_bias=True,
        gate_bias=True,
        gate_bias_init=1.0,
    )

    assert projection.query_projection.bias is None
    assert projection.key_projection.bias is None
    assert projection.value_projection.bias is None
    assert projection.output_projection.bias is not None
    assert projection.gate_projection.bias is not None
    torch.testing.assert_close(projection.gate_projection.bias, torch.ones(8))

    query = torch.randn(1, 3, 8)
    output = torch.randn(1, 3, 2, 4)

    assert projection.merge(output, query).shape == (1, 3, 8)
