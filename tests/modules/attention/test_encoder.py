from __future__ import annotations

import torch

from ofoldx.modules.attention import PairBiasedSelfAttention


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
