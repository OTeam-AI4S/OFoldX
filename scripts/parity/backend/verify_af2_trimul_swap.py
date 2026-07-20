#!/usr/bin/env python
r"""VERIFY-FIRST for TRUNK_EINSUM_LOCK.md B-nv2 (AF2 incoming triangle-mult swap).

Proves the canonical unified TriangleMultiplication + the incoming left<->right converter
swap reproduces the *legacy* AF2 TriangleMultiplication byte-for-byte, on random biased
weights, before the legacy class is deleted. If this regresses, the swap is backwards
(the recorded ~21A / 9-of-10-keys transpose); do not proceed.

The legacy forward is inlined verbatim from the pre-refactor evoformer.py so the check does
not depend on the class still existing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import Tensor, nn

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ofoldx.modules.trunk.evoformer import AlphaFold2TriangleMultiplication  # noqa: E402


class LegacyAF2TriangleMultiplication(nn.Module):
    """Verbatim pre-refactor AF2 triangle-mult (evoformer.py:427, DeepMind operand order)."""

    def __init__(self, *, pair_size: int, hidden_size: int, equation: str) -> None:
        super().__init__()
        self.equation = equation
        self.layer_norm_input = nn.LayerNorm(pair_size)
        self.left_projection = nn.Linear(pair_size, hidden_size)
        self.right_projection = nn.Linear(pair_size, hidden_size)
        self.left_gate = nn.Linear(pair_size, hidden_size)
        self.right_gate = nn.Linear(pair_size, hidden_size)
        self.center_layer_norm = nn.LayerNorm(hidden_size)
        self.output_projection = nn.Linear(hidden_size, pair_size)
        self.gating_linear = nn.Linear(pair_size, pair_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, pair: Tensor, pair_mask: Tensor) -> Tensor:
        mask = pair_mask.unsqueeze(-1).to(dtype=pair.dtype)
        normalized = self.layer_norm_input(pair)
        left = self.left_projection(normalized) * self.sigmoid(self.left_gate(normalized)) * mask
        right = self.right_projection(normalized) * self.sigmoid(self.right_gate(normalized)) * mask
        if self.equation == "outgoing":
            update = torch.einsum("...ikc,...jkc->...ijc", left, right)
        else:
            update = torch.einsum("...kjc,...kic->...ijc", left, right)
        update = self.output_projection(self.center_layer_norm(update))
        return update * self.sigmoid(self.gating_linear(normalized))


def _swapped_state(legacy: LegacyAF2TriangleMultiplication, *, outgoing: bool) -> dict[str, Tensor]:
    """Replicate the AF2 converter mapping legacy-name -> unified-name (incoming swaps L<->R)."""
    ls = legacy.state_dict()
    out: dict[str, Tensor] = {}
    # LayerNorms + always-straight leaves.
    for a, b in (("layer_norm_input", "norm_input"), ("center_layer_norm", "norm_output")):
        out[f"{b}.weight"] = ls[f"{a}.weight"]
        out[f"{b}.bias"] = ls[f"{a}.bias"]
    for a, b in (("output_projection", "output_projection"), ("gating_linear", "gate_projection")):
        out[f"{b}.weight"] = ls[f"{a}.weight"]
        out[f"{b}.bias"] = ls[f"{a}.bias"]
    # Value + gate projections: straight for outgoing, L<->R swapped for incoming.
    if outgoing:
        vg = {
            "left_projection": "left_projection",
            "right_projection": "right_projection",
            "left_gate": "left_gate_projection",
            "right_gate": "right_gate_projection",
        }
    else:
        vg = {
            "left_projection": "right_projection",
            "right_projection": "left_projection",
            "left_gate": "right_gate_projection",
            "right_gate": "left_gate_projection",
        }
    for a, b in vg.items():
        out[f"{b}.weight"] = ls[f"{a}.weight"]
        out[f"{b}.bias"] = ls[f"{a}.bias"]
    return out


def _run(outgoing: bool, seed: int) -> float:
    torch.manual_seed(seed)
    pair_size, hidden_size, n = 16, 12, 7
    legacy = LegacyAF2TriangleMultiplication(
        pair_size=pair_size, hidden_size=hidden_size, equation="outgoing" if outgoing else "incoming"
    ).double()
    unified = AlphaFold2TriangleMultiplication(pair_size=pair_size, hidden_size=hidden_size, outgoing=outgoing).double()
    unified.load_state_dict(_swapped_state(legacy, outgoing=outgoing), strict=True)
    legacy.eval()
    unified.eval()

    pair = torch.randn(2, n, n, pair_size, dtype=torch.float64)
    mask = torch.rand(2, n, n) > 0.25
    with torch.no_grad():
        want = legacy(pair, mask)
        got = unified(pair, mask.bool())
    return (want - got).abs().max().item()


def main() -> None:
    rows = []
    for outgoing in (True, False):
        worst = max(_run(outgoing, seed) for seed in range(4))
        rows.append((("outgoing" if outgoing else "incoming"), worst))
    for name, worst in rows:
        status = "OK" if worst < 1e-10 else "FAIL"
        print(f"[{status}] {name}: legacy-vs-unified(+swap) max_abs = {worst:.3e}")
    ok = all(worst < 1e-10 for _, worst in rows)
    print("VERIFY-FIRST:", "PASS" if ok else "REGRESSION -- swap is backwards, do not proceed")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
