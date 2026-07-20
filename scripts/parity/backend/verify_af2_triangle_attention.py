#!/usr/bin/env python
r"""
VERIFY-FIRST: legacy AF2 ``TriangleAttention`` vs. the unified module + caller-side transpose.

B-nv3 collapses AlphaFold 2's ``TriangleAttention`` (internal ``per_row``/``per_column`` transpose,
``query_norm``/``feat_2d_weights`` names, biased output/gate) onto the shared bias-free
``ofoldx.modules.attention.triangle.TriangleAttention`` as the ``AlphaFold2TriangleAttention`` subclass.
Per TRUNK_EINSUM_LOCK.md §3 the unified module NEVER transposes internally: the ending node's residue
swap moves to the caller. This is the second-riskiest step in the trunk wave (the mirror trap: an ending
node accidentally built with ``transpose_bias=True`` silently switches to the DeepMind AF3 bias source).

This script inlines the *legacy* AF2 module verbatim and drives it against the new module fed
converter-mapped weights, fp64, masked fixtures, multiple seeds, for BOTH the starting node (no
transpose) and the ending node (caller transposes ``(-3, -2)`` + mask, transposes output back). Byte-
exact (max_abs == 0) ⇒ the refactor transitively inherits AF2's OpenFold parity.

Run (CPU is fine; fp64/sdpa)::

    PYTHONPATH=/home/z/ofoldx /data/users/z/envs/zyc/bin/python \
        /home/z/ofoldx/scripts/parity/backend/verify_af2_triangle_attention.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import Tensor, nn

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ofoldx.modules.attention.multihead import GatedMultiheadAttention  # noqa: E402
from ofoldx.modules.backend import LayerNorm  # noqa: E402
from ofoldx.modules.trunk.evoformer import AlphaFold2TriangleAttention  # noqa: E402


class LegacyAlphaFold2TriangleAttention(nn.Module):
    r"""The pre-refactor AF2 TriangleAttention (internal transpose), copied verbatim."""

    def __init__(self, *, pair_size, head_dim, num_attention_heads, orientation, operator_backend):
        super().__init__()
        self.orientation = orientation
        self.query_norm = LayerNorm(pair_size)
        self.feat_2d_weights = nn.Linear(pair_size, num_attention_heads, bias=False)
        self.attention = GatedMultiheadAttention(
            pair_size,
            num_attention_heads=num_attention_heads,
            head_dim=head_dim,
            query_bias=False,
            output_bias=True,
            gate_bias=True,
            gate_bias_init=1.0,
            operator_backend=operator_backend,
        )

    def forward(self, pair: Tensor, pair_mask: Tensor) -> Tensor:
        if self.orientation == "per_column":
            pair = pair.transpose(-3, -2)
            pair_mask = pair_mask.transpose(-2, -1)
        normalized = self.query_norm(pair)
        pair_bias = self.feat_2d_weights(normalized).movedim(-1, -3).unsqueeze(-4)
        output = self.attention(normalized, biases=(pair_bias,), attention_mask=pair_mask.unsqueeze(-2).bool())
        if self.orientation == "per_column":
            output = output.transpose(-3, -2)
        return output


def _copy_weights(legacy: LegacyAlphaFold2TriangleAttention, unified: AlphaFold2TriangleAttention) -> None:
    # Converter name mapping: query_norm -> norm_pair, feat_2d_weights -> bias_projection, attention.*
    # identical (both wrap GatedMultiheadAttention with the same biased projections).
    unified.norm_pair.load_state_dict(legacy.query_norm.state_dict())
    unified.bias_projection.load_state_dict(legacy.feat_2d_weights.state_dict())
    unified.attention.load_state_dict(legacy.attention.state_dict())


def _new_forward(module: AlphaFold2TriangleAttention, orientation: str, pair: Tensor, pair_mask: Tensor) -> Tensor:
    # The caller owns the ending-node transpose now (TRUNK_EINSUM_LOCK.md §3).
    if orientation == "per_column":
        end_mask = pair_mask.transpose(-2, -1)
        return module(pair.transpose(-3, -2), end_mask).transpose(-3, -2)
    return module(pair, pair_mask)


def run() -> int:
    torch.set_default_dtype(torch.float64)
    pair_size, head_dim, heads = 24, 8, 4
    batch, tokens = 2, 9
    worst = 0.0
    rows = []
    for orientation in ("per_row", "per_column"):
        for seed in range(4):
            gen = torch.Generator().manual_seed(1000 + seed)
            legacy = (
                LegacyAlphaFold2TriangleAttention(
                    pair_size=pair_size,
                    head_dim=head_dim,
                    num_attention_heads=heads,
                    orientation=orientation,
                    operator_backend="sdpa",
                )
                .double()
                .eval()
            )
            unified = (
                AlphaFold2TriangleAttention(
                    pair_size=pair_size,
                    head_dim=head_dim,
                    num_attention_heads=heads,
                    operator_backend="sdpa",
                )
                .double()
                .eval()
            )
            # Randomise the legacy weights, then mirror onto the unified module by the converter mapping.
            for p in legacy.parameters():
                with torch.no_grad():
                    p.copy_(torch.randn(p.shape, generator=gen, dtype=torch.float64))
            _copy_weights(legacy, unified)
            pair = torch.randn(batch, tokens, tokens, pair_size, generator=gen, dtype=torch.float64)
            mask = torch.rand(batch, tokens, tokens, generator=gen) > 0.25
            mask = mask | torch.eye(tokens, dtype=torch.bool).unsqueeze(0)  # keep the diagonal key
            with torch.no_grad():
                out_legacy = legacy(pair, mask)
                out_new = _new_forward(unified, orientation, pair, mask)
            max_abs = (out_legacy - out_new).abs().max().item()
            worst = max(worst, max_abs)
            rows.append((orientation, seed, max_abs))

    print("legacy-vs-unified AF2 TriangleAttention (fp64, masked)")
    print(f"{'orientation':12s} {'seed':>4s} {'max_abs':>12s}")
    for orientation, seed, max_abs in rows:
        print(f"{orientation:12s} {seed:4d} {max_abs:12.3e}")
    print(f"\nworst max_abs = {worst:.3e}")
    ok = worst == 0.0
    print("VERDICT:", "BYTE-EXACT PASS" if ok else "FAIL (non-zero divergence)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(run())
