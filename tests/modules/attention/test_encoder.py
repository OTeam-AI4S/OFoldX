from __future__ import annotations

import pytest
import torch
from torch._dynamo.testing import CompileCounter

from ofoldx.modules.attention import PairBiasedSelfAttention


def _nested_guard():
    tensors = pytest.importorskip("danling.tensors")
    return tensors.nested_execution_guard(
        forbid_iteration=True,
        forbid_storage_map=True,
        forbid_eager_fallback=True,
        forbid_padded_materialization=True,
        forbid_dense_repack=True,
    )


def test_pair_bias_coupling() -> None:
    # The pair representation enters only as a per-(query, key) attention-bias logit
    # b_qk = Linear(LayerNorm(z_qk)). Perturbing a single entry z[:, q, k] must therefore
    # change the output at query row q and leave every other query row bit-for-bit identical.
    torch.manual_seed(0)
    attention = PairBiasedSelfAttention(hidden_size=8, pair_size=4, num_attention_heads=2, backend="sdpa").eval()
    hidden = torch.randn(1, 5, 8)
    pair = torch.randn(1, 5, 5, 4)

    with torch.no_grad():
        baseline = attention(hidden, pair)
        perturbed = pair.clone()
        perturbed[0, 2, 3] = torch.randn(4)  # bias for query=2, key=3 only
        updated = attention(hidden, perturbed)

    assert baseline.shape == hidden.shape
    assert torch.isfinite(baseline).all()
    assert not torch.allclose(baseline[0, 2], updated[0, 2], atol=1e-6)
    for query in (0, 1, 3, 4):
        assert torch.allclose(baseline[0, query], updated[0, query], atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA OOps kernels")
def test_oops_nested_implicit_sample_reuses_one_dynamic_fullgraph() -> None:
    pytest.importorskip("oops")
    NestedTensor = pytest.importorskip("danling.tensors").NestedTensor
    device = torch.device("cuda")
    module = PairBiasedSelfAttention(
        hidden_size=8,
        pair_size=8,
        num_attention_heads=2,
        backend="oops",
    ).to(device=device, dtype=torch.bfloat16)
    counter = CompileCounter()
    compiled = torch.compile(
        lambda hidden, pair, mask: module(hidden, pair, mask).concat,
        backend=counter,
        fullgraph=True,
        dynamic=True,
    )

    for lengths in ((2, 3), (3, 5)):
        hidden_template = NestedTensor([torch.empty(length, 8) for length in lengths], ragged_dims=(0,))
        hidden_values = torch.randn(
            hidden_template.concat.shape,
            device=device,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        hidden = hidden_template.packed_like(hidden_values)
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
                    module(element_hidden, element_pair, element_mask)
                    for element_hidden, element_pair, element_mask in zip(
                        torch.unbind(hidden),
                        torch.unbind(pair),
                        torch.unbind(mask),
                        strict=True,
                    )
                )
            )
        with _nested_guard():
            actual = compiled(hidden, pair, mask)
        torch.testing.assert_close(actual, expected, rtol=5e-2, atol=5e-2)
        hidden_gradient, pair_gradient, gate_gradient = torch.autograd.grad(
            actual.sum(),
            (hidden_values, pair_values, module.attention.projection.gate_projection.weight),
        )
        assert hidden_gradient is not None
        assert pair_gradient is not None
        assert gate_gradient is not None
    assert counter.frame_count == 1
