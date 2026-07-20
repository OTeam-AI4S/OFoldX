#!/usr/bin/env python
r"""
CUDA-gated parity: OFoldX ``operator_backend="oops"`` vs ``operator_backend="eager"``.

This closes the highest OOps risk: OFoldX's fused-kernel compute path was previously untied to the
eager reference (the model-level parity harness forces ``operator_backend="eager"``). Here we drive
the *same* random inputs and weights through each wired backend dispatcher on both backends and assert
``oops`` == ``eager`` within a floating-point tolerance.

Covered dispatchers (all under ``ofoldx.modules.backend``):

- ``triangle_multiplication``
- ``outer_product_mean``
- ``pair_weighted_average`` (incl. the dual-eps concern at pair_weighted_average.py:118)
- ``transition``
- ``gated_multihead_attention`` -- both the ungated ``pair_bias_attention`` path and the fused
  ``gated_pair_bias_attention`` fast path
- ``LayerNorm`` (fused ``oops.LayerNorm`` vs ``torch.nn.LayerNorm``)
- ``atom_to_token_mean`` / ``token_to_atom_broadcast`` / ``token_pair_to_atom_window_gather``

Plus the AF3 triangle-attention block end-to-end (``ofoldx.modules.attention.triangle.TriangleAttention``).

Method (VERIFY-FIRST): every op runs in **both** fp32 and bf16.

- A divergence in **fp32** is a real issue (adapter bug or kernel bug) -- root-cause it.
- A divergence only in **bf16** while fp32 matches is a floating-point/precision artifact.

Requires CUDA + the ``oops`` package. NEVER fabricates a pass: if CUDA or OOps is missing, or an OOps
kernel raises, that is reported as a hard failure, not a match.

Run (on a GPU node)::

    srun --partition=gpu --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=00:40:00 \
        env PYTHONPATH=/home/z/ofoldx /data/users/z/envs/oops-parity/bin/python \
        /home/z/ofoldx/scripts/parity/backend/oops_parity.py
"""

from __future__ import annotations

import math
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor

# Allow direct execution (python scripts/parity/backend/oops_parity.py) as well as -m.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ofoldx.modules.attention.multihead import MultiheadAttentionProjection  # noqa: E402
from ofoldx.modules.attention.triangle import TriangleAttention  # noqa: E402
from ofoldx.modules.backend._oops import OOPS_AVAILABLE  # noqa: E402
from ofoldx.modules.backend._oops import LayerNorm as OOpsLayerNorm  # noqa: E402
from ofoldx.modules.backend.atom_window import (  # noqa: E402
    atom_to_token_mean,
    token_pair_to_atom_window_gather,
    token_to_atom_broadcast,
)
from ofoldx.modules.backend.attention import gated_multihead_attention  # noqa: E402
from ofoldx.modules.backend.outer_product_mean import outer_product_mean  # noqa: E402
from ofoldx.modules.backend.pair_weighted_average import pair_weighted_average  # noqa: E402
from ofoldx.modules.backend.resolve import oops_available  # noqa: E402
from ofoldx.modules.backend.transition import transition  # noqa: E402
from ofoldx.modules.backend.triangle_multiplication import triangle_multiplication  # noqa: E402

# Relative-L2 MATCH thresholds. Fused kernels reorder reductions vs the eager composition, so exact
# equality is not expected even in fp32; these bound the gap that a correct kernel can produce.
TOL_REL_L2 = {torch.float32: 5e-3, torch.bfloat16: 5e-2}
# Index/gather ops are exact on both backends -- hold them to a much tighter bound.
TOL_REL_L2_EXACT = {torch.float32: 1e-6, torch.bfloat16: 1e-2}

DTYPES = (torch.float32, torch.bfloat16)
DTYPE_NAME = {torch.float32: "fp32", torch.bfloat16: "bf16"}


def _generator(seed: int) -> torch.Generator:
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    return gen


def randn(*shape: int, gen: torch.Generator, dtype: torch.dtype) -> Tensor:
    # Draw in fp32 for a stable stream, then cast, so fp32/bf16 runs share the same underlying values.
    return torch.randn(*shape, generator=gen, device="cuda", dtype=torch.float32).to(dtype)


@dataclass
class Metrics:
    rel_l2: float
    max_abs: float
    cosine: float

    @classmethod
    def compare(cls, got: Tensor, ref: Tensor) -> "Metrics":
        got64 = got.detach().to(torch.float64).flatten()
        ref64 = ref.detach().to(torch.float64).flatten()
        diff = got64 - ref64
        ref_norm = ref64.norm().item()
        rel_l2 = (diff.norm().item() / ref_norm) if ref_norm > 0 else diff.norm().item()
        max_abs = diff.abs().max().item() if diff.numel() else 0.0
        denom = got64.norm().item() * ref64.norm().item()
        cosine = (torch.dot(got64, ref64).item() / denom) if denom > 0 else 1.0
        return cls(rel_l2=rel_l2, max_abs=max_abs, cosine=cosine)


@dataclass
class Result:
    op: str
    dtype: torch.dtype
    metrics: Metrics | None = None
    error: str | None = None
    tol: float = 0.0
    note: str = ""
    # ``expect_divergent`` marks probe cases (e.g. dual-eps) where a divergence is the finding itself.
    expect_divergent: bool = False

    @property
    def matched(self) -> bool:
        return self.error is None and self.metrics is not None and self.metrics.rel_l2 <= self.tol

    @property
    def verdict(self) -> str:
        if self.error is not None:
            return "ERROR"
        return "MATCH" if self.matched else "DIVERGENT"


@dataclass
class Report:
    results: list[Result] = field(default_factory=list)

    def run(
        self,
        op: str,
        dtype: torch.dtype,
        fn,
        *,
        exact: bool = False,
        note: str = "",
        expect_divergent: bool = False,
    ) -> None:
        tol_table = TOL_REL_L2_EXACT if exact else TOL_REL_L2
        result = Result(op=op, dtype=dtype, tol=tol_table[dtype], note=note, expect_divergent=expect_divergent)
        try:
            got, ref = fn(dtype)
            result.metrics = Metrics.compare(got, ref)
        except Exception as exc:  # noqa: BLE001 -- surface any kernel/adapter failure verbatim
            result.error = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
        self.results.append(result)


# --------------------------------------------------------------------------------------------------
# Per-op input builders. Each returns (oops_output, eager_output) for a given dtype, using identical
# inputs and weights for both backends.
# --------------------------------------------------------------------------------------------------


def case_triangle_multiplication(dtype: torch.dtype, *, outgoing: bool, masked: bool):
    gen = _generator(101)
    batch, tokens, pair_size, hidden = 2, 32, 64, 64
    pair = randn(batch, tokens, tokens, pair_size, gen=gen, dtype=dtype)
    mask = None
    if masked:
        mask = (torch.rand(batch, tokens, tokens, generator=gen, device="cuda") > 0.2).to(dtype)
    kwargs = dict(
        outgoing=outgoing,
        norm_input_weight=randn(pair_size, gen=gen, dtype=dtype),
        norm_input_bias=randn(pair_size, gen=gen, dtype=dtype),
        left_value_weight=randn(hidden, pair_size, gen=gen, dtype=dtype),
        left_gate_weight=randn(hidden, pair_size, gen=gen, dtype=dtype),
        right_value_weight=randn(hidden, pair_size, gen=gen, dtype=dtype),
        right_gate_weight=randn(hidden, pair_size, gen=gen, dtype=dtype),
        norm_output_weight=randn(hidden, gen=gen, dtype=dtype),
        norm_output_bias=randn(hidden, gen=gen, dtype=dtype),
        output_weight=randn(pair_size, hidden, gen=gen, dtype=dtype),
        output_gate_weight=randn(pair_size, pair_size, gen=gen, dtype=dtype),
        eps=1e-5,
    )
    oops = triangle_multiplication(pair, mask, operator_backend="oops", **kwargs)
    eager = triangle_multiplication(pair, mask, operator_backend="eager", **kwargs)
    return oops, eager


def case_outer_product_mean(dtype: torch.dtype, *, divide_before: bool, masked: bool):
    gen = _generator(202)
    batch, depth, tokens, msa_size, proj, pair_size = 2, 8, 24, 32, 16, 48
    msa = randn(batch, depth, tokens, msa_size, gen=gen, dtype=dtype)
    mask = None
    if masked:
        mask = (torch.rand(batch, depth, tokens, generator=gen, device="cuda") > 0.2).to(dtype)
    kwargs = dict(
        norm_weight=randn(msa_size, gen=gen, dtype=dtype),
        norm_bias=randn(msa_size, gen=gen, dtype=dtype),
        proj_left_weight=randn(proj, msa_size, gen=gen, dtype=dtype),
        proj_right_weight=randn(proj, msa_size, gen=gen, dtype=dtype),
        proj_out_weight=randn(pair_size, proj * proj, gen=gen, dtype=dtype),
        proj_out_bias=randn(pair_size, gen=gen, dtype=dtype),
        norm_eps=1e-5,
        denom_eps=1e-3,
        divide_before_projection=divide_before,
    )
    oops = outer_product_mean(msa, mask, operator_backend="oops", **kwargs)
    eager = outer_product_mean(msa, mask, operator_backend="eager", **kwargs)
    return oops, eager


def _pwa_inputs(gen: torch.Generator, dtype: torch.dtype, *, masked: bool):
    batch, depth, tokens, msa_size, pair_size = 2, 4, 32, 32, 48
    num_heads, head_dim = 4, 8
    msa = randn(batch, depth, tokens, msa_size, gen=gen, dtype=dtype)
    pair = randn(batch, tokens, tokens, pair_size, gen=gen, dtype=dtype)
    mask = None
    if masked:
        mask = (torch.rand(batch, tokens, tokens, generator=gen, device="cuda") > 0.2).to(dtype)
    weights = dict(
        norm_msa_weight=randn(msa_size, gen=gen, dtype=dtype),
        norm_msa_bias=randn(msa_size, gen=gen, dtype=dtype),
        norm_pair_weight=randn(pair_size, gen=gen, dtype=dtype),
        norm_pair_bias=randn(pair_size, gen=gen, dtype=dtype),
        value_weight=randn(num_heads * head_dim, msa_size, gen=gen, dtype=dtype),
        gate_weight=randn(num_heads * head_dim, msa_size, gen=gen, dtype=dtype),
        pair_logits_weight=randn(num_heads, pair_size, gen=gen, dtype=dtype),
        output_weight=randn(msa_size, num_heads * head_dim, gen=gen, dtype=dtype),
        num_heads=num_heads,
        head_dim=head_dim,
    )
    return msa, pair, mask, weights


def case_pair_weighted_average(dtype: torch.dtype, *, masked: bool):
    gen = _generator(303)
    msa, pair, mask, weights = _pwa_inputs(gen, dtype, masked=masked)
    oops = pair_weighted_average(
        msa, pair, mask, operator_backend="oops", norm_msa_eps=1e-5, norm_pair_eps=1e-5, **weights
    )
    eager = pair_weighted_average(
        msa, pair, mask, operator_backend="eager", norm_msa_eps=1e-5, norm_pair_eps=1e-5, **weights
    )
    return oops, eager


def case_pair_weighted_average_dual_eps(dtype: torch.dtype):
    r"""
    Probe the flagged dual-eps concern at backend/pair_weighted_average.py:118.

    The adapter forwards only ``eps=norm_msa_eps`` to the single-eps OOps kernel, which then uses it for
    *both* the MSA and the pair LayerNorm. The eager path uses ``norm_pair_eps`` for the pair LayerNorm.
    Setting the two epsilons far apart makes any behavioural difference observable.
    """
    gen = _generator(304)
    msa, pair, mask, weights = _pwa_inputs(gen, dtype, masked=False)
    # Exaggerated pair eps so the pair LayerNorm is materially sensitive to which eps is used.
    oops = pair_weighted_average(
        msa, pair, mask, operator_backend="oops", norm_msa_eps=1e-5, norm_pair_eps=0.5, **weights
    )
    eager = pair_weighted_average(
        msa, pair, mask, operator_backend="eager", norm_msa_eps=1e-5, norm_pair_eps=0.5, **weights
    )
    return oops, eager


def case_transition(dtype: torch.dtype):
    gen = _generator(404)
    batch, tokens, hidden, inter = 2, 64, 128, 256
    hidden_in = randn(batch, tokens, hidden, gen=gen, dtype=dtype)
    kwargs = dict(
        norm_weight=randn(hidden, gen=gen, dtype=dtype),
        norm_bias=randn(hidden, gen=gen, dtype=dtype),
        gate_weight=randn(inter, hidden, gen=gen, dtype=dtype),
        up_weight=randn(inter, hidden, gen=gen, dtype=dtype),
        down_weight=randn(hidden, inter, gen=gen, dtype=dtype),
        eps=1e-5,
    )
    oops = transition(hidden_in, operator_backend="oops", **kwargs)
    eager = transition(hidden_in, operator_backend="eager", **kwargs)
    return oops, eager


def _build_projection(dtype: torch.dtype, *, gating: bool, gen: torch.Generator) -> MultiheadAttentionProjection:
    hidden, heads, head_dim = 64, 4, 16
    proj = MultiheadAttentionProjection(
        hidden_size=hidden,
        num_attention_heads=heads,
        head_dim=head_dim,
        query_bias=False,
        output_bias=False,
        gating=gating,
        gate_bias=False,
    )
    proj = proj.to(device="cuda", dtype=dtype)
    # Deterministically re-initialise every projection weight from the shared generator.
    for module in proj.modules():
        if isinstance(module, torch.nn.Linear):
            with torch.no_grad():
                module.weight.copy_(randn(*module.weight.shape, gen=gen, dtype=dtype))
    return proj


def case_attention(dtype: torch.dtype, *, gating: bool, masked: bool):
    r"""
    ``gating=True`` with a pair bias exercises the fused ``gated_pair_bias_attention`` fast path;
    ``gating=False`` exercises the ungated ``pair_bias_attention`` kernel.
    """
    gen = _generator(505 if gating else 506)
    batch, tokens, hidden, heads = 2, 32, 64, 4
    proj = _build_projection(dtype, gating=gating, gen=gen)
    query = randn(batch, tokens, hidden, gen=gen, dtype=dtype)
    pair_bias = randn(batch, heads, tokens, tokens, gen=gen, dtype=dtype)
    mask = None
    if masked:
        # Full (batch, q, k) mask, the general form both backends support. Callers only ever pass masks
        # carrying an explicit query axis (e.g. ``msa_mask.unsqueeze(-2)``); a bare key-only (..., k) mask
        # is accepted by the OOps path but not broadcast over queries by ``_eager_attention``.
        mask = torch.rand(batch, tokens, tokens, generator=gen, device="cuda") > 0.2
        # Guarantee every query keeps at least its diagonal key so no softmax row is fully masked.
        mask = mask | torch.eye(tokens, dtype=torch.bool, device="cuda")
    common = dict(
        biases=[pair_bias],
        attention_mask=mask,
        projection=proj,
        torch_backend="eager",
    )
    oops = gated_multihead_attention(query, None, None, operator_backend="oops", **common)
    eager = gated_multihead_attention(query, None, None, operator_backend="eager", **common)
    return oops, eager


def case_layer_norm(dtype: torch.dtype):
    gen = _generator(606)
    batch, tokens, dim = 4, 64, 128
    x = randn(batch, tokens, dim, gen=gen, dtype=dtype)
    weight = randn(dim, gen=gen, dtype=dtype)
    bias = randn(dim, gen=gen, dtype=dtype)
    fused = OOpsLayerNorm(dim, eps=1e-5).to(device="cuda", dtype=dtype)
    ref = torch.nn.LayerNorm(dim, eps=1e-5).to(device="cuda", dtype=dtype)
    with torch.no_grad():
        fused.weight.copy_(weight)
        fused.bias.copy_(bias)
        ref.weight.copy_(weight)
        ref.bias.copy_(bias)
    return fused(x), ref(x)


def case_atom_to_token_mean(dtype: torch.dtype, *, masked: bool):
    gen = _generator(707)
    batch, n_atom, n_token, channels = 2, 96, 24, 32
    atom = randn(batch, n_atom, channels, gen=gen, dtype=dtype)
    # Each token gets a contiguous run of atoms so no token is empty.
    per = n_atom // n_token
    atom_to_token = (torch.arange(n_atom, device="cuda") // per).clamp(max=n_token - 1)
    atom_to_token = atom_to_token.unsqueeze(0).expand(batch, -1).contiguous()
    mask = None
    if masked:
        mask = torch.rand(batch, n_atom, generator=gen, device="cuda") > 0.2
    oops = atom_to_token_mean(
        atom, atom_to_token, num_tokens=n_token, atom_mask=mask, eps=1e-6, operator_backend="oops"
    )
    eager = atom_to_token_mean(
        atom, atom_to_token, num_tokens=n_token, atom_mask=mask, eps=1e-6, operator_backend="eager"
    )
    return oops, eager


def case_token_to_atom_broadcast(dtype: torch.dtype):
    gen = _generator(808)
    batch, n_atom, n_token, channels = 2, 96, 24, 32
    token = randn(batch, n_token, channels, gen=gen, dtype=dtype)
    per = n_atom // n_token
    atom_to_token = (torch.arange(n_atom, device="cuda") // per).clamp(max=n_token - 1)
    atom_to_token = atom_to_token.unsqueeze(0).expand(batch, -1).contiguous()
    oops = token_to_atom_broadcast(token, atom_to_token, operator_backend="oops")
    eager = token_to_atom_broadcast(token, atom_to_token, operator_backend="eager")
    return oops, eager


def case_token_pair_to_atom_window_gather(dtype: torch.dtype):
    gen = _generator(909)
    batch, n_atom, n_token, channels = 2, 64, 16, 32
    n_queries, n_keys = 32, 128
    token_pair = randn(batch, n_token, n_token, channels, gen=gen, dtype=dtype)
    per = n_atom // n_token
    atom_to_token = (torch.arange(n_atom, device="cuda") // per).clamp(max=n_token - 1)
    oops = token_pair_to_atom_window_gather(
        token_pair, atom_to_token, n_queries=n_queries, n_keys=n_keys, operator_backend="oops"
    )
    eager = token_pair_to_atom_window_gather(
        token_pair, atom_to_token, n_queries=n_queries, n_keys=n_keys, operator_backend="eager"
    )
    return oops, eager


def case_triangle_attention_block(dtype: torch.dtype, *, masked: bool):
    r"""AF3 triangle-attention block end-to-end on both backends (same module, same weights)."""
    gen = _generator(1010)
    batch, tokens, pair_size, heads, head_dim = 2, 24, 64, 4, 16
    module = TriangleAttention(
        pair_size=pair_size, head_dim=head_dim, num_attention_heads=heads, operator_backend="eager"
    )
    module = module.to(device="cuda", dtype=dtype)
    for m in module.modules():
        if isinstance(m, torch.nn.Linear):
            with torch.no_grad():
                m.weight.copy_(randn(*m.weight.shape, gen=gen, dtype=dtype))
                if m.bias is not None:
                    m.bias.copy_(randn(*m.bias.shape, gen=gen, dtype=dtype))
    pair = randn(batch, tokens, tokens, pair_size, gen=gen, dtype=dtype)
    mask = None
    if masked:
        mask = torch.rand(batch, tokens, tokens, generator=gen, device="cuda") > 0.2
    module.attention.operator_backend = "oops"
    oops = module(pair, mask)
    module.attention.operator_backend = "eager"
    eager = module(pair, mask)
    return oops, eager


# --------------------------------------------------------------------------------------------------


def build_report() -> Report:
    report = Report()
    for dtype in DTYPES:
        report.run(
            "triangle_multiplication[out]",
            dtype,
            lambda d: case_triangle_multiplication(d, outgoing=True, masked=False),
        )
        report.run(
            "triangle_multiplication[in,mask]",
            dtype,
            lambda d: case_triangle_multiplication(d, outgoing=False, masked=True),
        )
        report.run(
            "outer_product_mean[after]",
            dtype,
            lambda d: case_outer_product_mean(d, divide_before=False, masked=True),
        )
        report.run(
            "outer_product_mean[before]",
            dtype,
            lambda d: case_outer_product_mean(d, divide_before=True, masked=False),
        )
        report.run("pair_weighted_average", dtype, lambda d: case_pair_weighted_average(d, masked=False))
        report.run("pair_weighted_average[mask]", dtype, lambda d: case_pair_weighted_average(d, masked=True))
        report.run(
            "pair_weighted_average[dual-eps probe]",
            dtype,
            case_pair_weighted_average_dual_eps,
            note="msa_eps=1e-5 pair_eps=0.5",
            expect_divergent=True,
        )
        report.run("transition", dtype, case_transition)
        report.run("attention[gated_pair_bias]", dtype, lambda d: case_attention(d, gating=True, masked=False))
        report.run("attention[gated_pair_bias,mask]", dtype, lambda d: case_attention(d, gating=True, masked=True))
        report.run("attention[pair_bias]", dtype, lambda d: case_attention(d, gating=False, masked=False))
        report.run("layer_norm", dtype, case_layer_norm)
        report.run("atom_to_token_mean", dtype, lambda d: case_atom_to_token_mean(d, masked=False), exact=True)
        report.run("atom_to_token_mean[mask]", dtype, lambda d: case_atom_to_token_mean(d, masked=True), exact=True)
        report.run("token_to_atom_broadcast", dtype, case_token_to_atom_broadcast, exact=True)
        report.run("token_pair_to_atom_window_gather", dtype, case_token_pair_to_atom_window_gather, exact=True)
        report.run("TriangleAttention(e2e)", dtype, lambda d: case_triangle_attention_block(d, masked=False))
        report.run("TriangleAttention(e2e,mask)", dtype, lambda d: case_triangle_attention_block(d, masked=True))
    return report


def print_report(report: Report) -> int:
    header = f"{'op':40s} {'dtype':5s} {'rel_l2':>11s} {'max_abs':>11s} {'cosine':>10s}  verdict"
    print("\n" + "=" * len(header))
    print("OFoldX OOps-vs-eager backend parity")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in report.results:
        dname = DTYPE_NAME[r.dtype]
        if r.error is not None:
            print(f"{r.op:40s} {dname:5s} {'':>11s} {'':>11s} {'':>10s}  ERROR  {r.error}")
            continue
        m = r.metrics
        assert m is not None
        flag = r.verdict
        extra = ""
        if r.note:
            extra = f"  ({r.note})"
        if r.expect_divergent:
            flag = f"{flag} [probe]"
        print(f"{r.op:40s} {dname:5s} {m.rel_l2:11.3e} {m.max_abs:11.3e} {m.cosine:10.6f}  {flag}{extra}")

    print("-" * len(header))

    # Root-cause summary: separate real (fp32) divergences from bf16-only precision artifacts.
    by_op: dict[str, dict[torch.dtype, Result]] = {}
    for r in report.results:
        by_op.setdefault(r.op, {})[r.dtype] = r

    errors = [r for r in report.results if r.error is not None]
    real_divergences: list[str] = []
    precision_artifacts: list[str] = []
    probe_findings: list[str] = []

    for op, per_dtype in by_op.items():
        fp32 = per_dtype.get(torch.float32)
        bf16 = per_dtype.get(torch.bfloat16)
        if fp32 is not None and fp32.expect_divergent:
            state = "DIVERGENT" if not fp32.matched else "MATCH"
            probe_findings.append(f"{op}: fp32 {state} (rel_l2={fp32.metrics.rel_l2:.3e})" if fp32.metrics else op)
            continue
        if fp32 is not None and fp32.error is None and not fp32.matched:
            real_divergences.append(f"{op}: fp32 rel_l2={fp32.metrics.rel_l2:.3e} cos={fp32.metrics.cosine:.6f}")
        elif fp32 is not None and bf16 is not None and fp32.matched and bf16.error is None and not bf16.matched:
            precision_artifacts.append(f"{op}: bf16 rel_l2={bf16.metrics.rel_l2:.3e} (fp32 matches)")

    print("SUMMARY")
    print("-" * len(header))
    if errors:
        print(f"  ERRORS ({len(errors)}): kernels that failed to run --")
        for r in errors:
            print(f"    - {r.op} [{DTYPE_NAME[r.dtype]}]: {r.error}")
    if real_divergences:
        print(f"  REAL DIVERGENCES in fp32 ({len(real_divergences)}) -- adapter or kernel bug:")
        for line in real_divergences:
            print(f"    - {line}")
    if precision_artifacts:
        print(f"  bf16-only precision artifacts ({len(precision_artifacts)}) -- fp32 matches, expected:")
        for line in precision_artifacts:
            print(f"    - {line}")
    if probe_findings:
        print("  PROBES (divergence is the finding, not a regression):")
        for line in probe_findings:
            print(f"    - {line}")

    # Verdict: validated iff no errors and no real fp32 divergences on the non-probe ops.
    validated = not errors and not real_divergences
    print("-" * len(header))
    print(
        f"  VERDICT: OOps backend {'VALIDATED' if validated else 'NOT VALIDATED'} "
        f"(errors={len(errors)}, fp32-divergences={len(real_divergences)})"
    )
    print("=" * len(header) + "\n")
    return 0 if validated else 1


def main() -> int:
    print(f"torch {torch.__version__}  cuda_available={torch.cuda.is_available()}  oops_available={oops_available()}")
    if not torch.cuda.is_available():
        print("FAIL: CUDA is not available; the OOps backend is CUDA-only and cannot be validated here.")
        return 2
    if not OOPS_AVAILABLE:
        print("FAIL: the 'oops' package is not importable; cannot validate operator_backend='oops'.")
        return 2
    print(f"device: {torch.cuda.get_device_name(0)}")
    try:
        import oops

        print(f"oops {oops.__version__} @ {oops.__file__}")
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: could not import oops for version banner: {exc}")
        return 2
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    report = build_report()
    return print_report(report)


if __name__ == "__main__":
    raise SystemExit(main())
