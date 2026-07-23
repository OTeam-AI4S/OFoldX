from __future__ import annotations

import pytest
import torch

from ofoldx.modules.attention import GatedMultiheadAttention, MultiheadAttentionProjection
from ofoldx.modules.backend import resolve as resolve_module


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


def test_explicit_head_dim_defines_the_inner_width() -> None:
    projection = MultiheadAttentionProjection(hidden_size=8, num_attention_heads=3, head_dim=4)

    assert projection.inner_size == 12
    assert projection.query_projection.out_features == 12
    assert projection.output_projection.in_features == 12


@pytest.mark.parametrize("backend", ("auto", "torch"))
def test_public_torch_paths_use_sdpa_without_changing_state_dict(monkeypatch, backend: str) -> None:
    monkeypatch.setattr(resolve_module, "oops_available", lambda: False)
    public = GatedMultiheadAttention(hidden_size=8, num_attention_heads=2, operator_backend=backend)
    sdpa = GatedMultiheadAttention(hidden_size=8, num_attention_heads=2, operator_backend="sdpa")
    sdpa.load_state_dict(public.state_dict())

    query = torch.randn(2, 4, 8)
    bias = torch.randn(2, 2, 4, 4)
    mask = torch.tensor([[True, True, False, True], [True, False, True, True]]).unsqueeze(1).expand(-1, 4, -1)

    torch.testing.assert_close(
        public(query, biases=(bias,), attention_mask=mask),
        sdpa(query, biases=(bias,), attention_mask=mask),
        rtol=0,
        atol=0,
    )
    assert public.state_dict().keys() == sdpa.state_dict().keys()
    assert all("backend" not in key for key in public.state_dict())
