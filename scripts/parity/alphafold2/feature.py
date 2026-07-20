#!/usr/bin/env python
r"""Stage-1 stochastic MSA-feature parity for monomer AlphaFold 2 (CPU, no weights).

The released stochastic fold realizes MSA features per recycle from the raw
alignment pool: ``sample_msa`` -> ``make_masked_msa`` (BERT) ->
``nearest_neighbor_clusters`` / ``summarize_clusters`` -> ``crop_extra_msa`` ->
``make_msa_feat``. OFoldX moved this draw into the Model; this harness proves the
new numerical code is BITWISE identical to OpenFold's reference transforms.

RNG alignment (the load-bearing subtlety). OpenFold does NOT drive these stages
from one stream: ``sample_msa`` opens its own ``torch.Generator`` seeded from its
``seed=`` argument, while ``make_masked_msa`` and ``crop_extra_msa`` draw from the
process-global default RNG. OFoldX threads a single per-sample generator through
all three. There is therefore no single seed that reproduces both end to end.
We align EXACTLY and per stage: each stage's OFoldX helper is fed a fresh
generator seeded ``S``, and the matching OpenFold transform is seeded ``S`` too
(via its ``seed=`` argument for ``sample_msa``; via ``torch.manual_seed(S)``
immediately before the global-RNG stages). Identical inputs + identical draws =>
the outputs must match to the last bit; any nonzero max_abs is a real defect.

A separate check confirms OFoldX's *integrated* ``realize_msa_features`` threads
its single generator through sample->mask->crop in that order with no hidden
draws (integrated output == a manual replay of the helpers off one generator).

Everything here is CPU + integer/float32 transforms; no GPU, weights, or model
trunk are needed. Run under the ``zyc`` venv.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ofoldx.models.alphafold2.model import (  # noqa: E402
    _crop_extra_msa,
    _make_extra_msa_feat,
    _make_masked_msa,
    _make_msa_feat,
    _nearest_neighbor_clusters,
    _sample_generator,
    _sample_msa,
    _summarize_clusters,
    realize_msa_features,
)
from scripts.parity.paths import reference_root  # noqa: E402

# Released predict-mode MSA constants (mirror AlphaFold2MSAConfig).
MAX_MSA = 512
MAX_EXTRA = 1024
UNIFORM = 0.10
PROFILE = 0.10
SAME = 0.10
REPLACE = 0.15
GAP_W = 0.0


# --------------------------------------------------------------------------------------
# Synthetic raw alignment pool (deterministic across processes)
# --------------------------------------------------------------------------------------
def build_pool(num_seq: int, num_res: int, pool_seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Build a deterministic raw pool big enough to exercise sampling / clustering / crop.

    ``msa_tokens [M, N]`` long restype ids (0..21; 20 unknown, 21 gap; row 0 the
    query), ``msa_deletions [M, N]`` float insertion counts, and the full-pool
    ``hhblits_profile [N, 22]`` -- computed by the exact ``make_hhblits_profile``
    formula (mean of one_hot(tokens, 22)) so both pipelines start from the same
    profile.
    """

    g = torch.Generator().manual_seed(pool_seed)
    # Bias toward real residues so clustering is non-degenerate; allow gaps (21) and unknown (20).
    tokens = torch.randint(0, 22, (num_seq, num_res), generator=g)
    query = torch.randint(0, 20, (num_res,), generator=g)
    tokens[0] = query
    deletions = (torch.rand((num_seq, num_res), generator=g) * 6.0).floor()
    deletions[0] = 0.0
    profile = torch.nn.functional.one_hot(tokens, num_classes=22).float().mean(dim=0)
    return tokens.long(), deletions.float(), profile.float()


# --------------------------------------------------------------------------------------
# OpenFold reference transforms
# --------------------------------------------------------------------------------------
def load_openfold(reference: Path) -> Any:
    if str(reference) not in sys.path:
        sys.path.insert(0, str(reference))
    from openfold.data import data_transforms

    return data_transforms


def of_sample(dt: Any, tokens: torch.Tensor, deletions: torch.Tensor, seed: int) -> dict[str, torch.Tensor]:
    protein = {
        "msa": tokens.clone(),
        "deletion_matrix": deletions.clone(),
        "msa_mask": torch.ones(tokens.shape, dtype=torch.float32),
    }
    dt.sample_msa(MAX_MSA, True, seed)(protein)
    return protein


def of_masked(dt: Any, sel_tokens: torch.Tensor, profile: torch.Tensor, aatype: torch.Tensor, seed: int) -> dict:
    protein = {
        "msa": sel_tokens.clone(),
        "aatype": aatype.clone(),
        "hhblits_profile": profile.clone(),
    }
    config = types.SimpleNamespace(uniform_prob=UNIFORM, profile_prob=PROFILE, same_prob=SAME)
    torch.manual_seed(seed)
    dt.make_masked_msa(config, REPLACE)(protein)
    return protein


def of_cluster(
    dt: Any,
    masked_tokens: torch.Tensor,
    sel_deletions: torch.Tensor,
    extra_tokens: torch.Tensor,
    extra_deletions: torch.Tensor,
) -> dict:
    protein = {
        "msa": masked_tokens.clone(),
        "msa_mask": torch.ones(masked_tokens.shape, dtype=torch.float32),
        "deletion_matrix": sel_deletions.clone(),
        "extra_msa": extra_tokens.clone(),
        "extra_msa_mask": torch.ones(extra_tokens.shape, dtype=torch.float32),
        "extra_deletion_matrix": extra_deletions.clone(),
    }
    dt.nearest_neighbor_clusters(GAP_W)(protein)
    dt.summarize_clusters()(protein)
    return protein


def of_crop(dt: Any, extra_tokens: torch.Tensor, extra_deletions: torch.Tensor, seed: int) -> dict:
    protein = {
        "extra_msa": extra_tokens.clone(),
        "extra_deletion_matrix": extra_deletions.clone(),
        "extra_msa_mask": torch.ones(extra_tokens.shape, dtype=torch.float32),
    }
    torch.manual_seed(seed)
    dt.crop_extra_msa(MAX_EXTRA)(protein)
    return protein


# --------------------------------------------------------------------------------------
# Comparison helpers
# --------------------------------------------------------------------------------------
def max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().float()
    b = b.detach().float()
    if a.shape != b.shape:
        return float("inf")
    if a.numel() == 0:
        return 0.0
    return float((a - b).abs().max())


def signature(*tensors: torch.Tensor) -> str:
    h = hashlib.sha256()
    for t in tensors:
        t = t.detach().cpu().contiguous()
        h.update(str(tuple(t.shape)).encode())
        h.update(t.float().numpy().tobytes())
    return h.hexdigest()


def realize_signature(tokens, deletions, profile, seed: int) -> str:
    g = torch.Generator().manual_seed(seed)
    feats = realize_msa_features(
        tokens,
        deletions,
        profile,
        max_msa_clusters=MAX_MSA,
        max_extra_msa=MAX_EXTRA,
        msa_cluster_features=True,
        gap_agreement_weight=GAP_W,
        masked_msa_enabled=True,
        uniform_prob=UNIFORM,
        profile_prob=PROFILE,
        same_prob=SAME,
        replace_fraction=REPLACE,
        generator=g,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    return signature(*feats)


# --------------------------------------------------------------------------------------
# Stage 1: per-stage bitwise parity vs OpenFold
# --------------------------------------------------------------------------------------
def stage1_per_stage(dt, tokens, deletions, profile, aatype) -> dict[str, Any]:
    device = torch.device("cpu")
    dtype = torch.float32
    S_SAMPLE, S_MASK, S_CROP = 101, 202, 303
    out: dict[str, Any] = {}

    # --- sample_msa ---
    g = torch.Generator(device=device).manual_seed(S_SAMPLE)
    sel_t, sel_d, ext_t, ext_d = _sample_msa(tokens, deletions, MAX_MSA, g)
    ref = of_sample(dt, tokens, deletions, S_SAMPLE)
    out["sample_msa.sel_tokens"] = max_abs(sel_t, ref["msa"])
    out["sample_msa.sel_deletions"] = max_abs(sel_d, ref["deletion_matrix"])
    out["sample_msa.extra_tokens"] = max_abs(ext_t, ref["extra_msa"])
    out["sample_msa.extra_deletions"] = max_abs(ext_d, ref["extra_deletion_matrix"])

    # --- make_masked_msa (uses the aligned sampled slice) ---
    g = torch.Generator(device=device).manual_seed(S_MASK)
    masked_t, bert_mask, true_msa = _make_masked_msa(
        sel_t,
        profile,
        uniform_prob=UNIFORM,
        profile_prob=PROFILE,
        same_prob=SAME,
        replace_fraction=REPLACE,
        generator=g,
        device=device,
        dtype=dtype,
    )
    refm = of_masked(dt, sel_t, profile, aatype, S_MASK)
    out["make_masked_msa.bert_msa"] = max_abs(masked_t, refm["msa"])
    out["make_masked_msa.bert_mask"] = max_abs(bert_mask, refm["bert_mask"])
    out["make_masked_msa.true_msa"] = max_abs(true_msa, refm["true_msa"])

    # --- clustering (deterministic) ---
    msa_mask = torch.ones(masked_t.shape, dtype=dtype)
    ext_mask = torch.ones(ext_t.shape, dtype=dtype)
    assignment = _nearest_neighbor_clusters(masked_t, msa_mask, ext_t, ext_mask, GAP_W, device, dtype)
    cluster_profile, cluster_deletion_mean = _summarize_clusters(
        masked_t, msa_mask, sel_d, ext_t, ext_mask, ext_d, assignment, dtype
    )
    refc = of_cluster(dt, masked_t, sel_d, ext_t, ext_d)
    out["nearest_neighbor.assignment"] = max_abs(assignment, refc["extra_cluster_assignment"])
    out["summarize.cluster_profile"] = max_abs(cluster_profile, refc["cluster_profile"])
    out["summarize.cluster_deletion_mean"] = max_abs(cluster_deletion_mean, refc["cluster_deletion_mean"])

    # --- crop_extra_msa ---
    g = torch.Generator(device=device).manual_seed(S_CROP)
    crop_t, crop_d, crop_mask = _crop_extra_msa(ext_t, ext_d, ext_mask, MAX_EXTRA, g, device)
    refx = of_crop(dt, ext_t, ext_d, S_CROP)
    out["crop_extra_msa.extra_tokens"] = max_abs(crop_t, refx["extra_msa"])
    out["crop_extra_msa.extra_deletions"] = max_abs(crop_d, refx["extra_deletion_matrix"])

    # --- make_msa_feat (49ch) + extra feat (25ch), both deterministic ---
    msa_feat = _make_msa_feat(masked_t, sel_d, cluster_profile, cluster_deletion_mean, dtype)
    extra_feat = _make_extra_msa_feat(crop_t, crop_d, dtype)
    ref_feat = of_make_feat(dt, masked_t, sel_d, aatype, cluster_profile, cluster_deletion_mean, crop_t, crop_d)
    out["make_msa_feat.msa_feat[49]"] = max_abs(msa_feat, ref_feat["msa_feat"])
    out["make_msa_feat.extra_msa_feat[25]"] = max_abs(extra_feat, ref_feat["extra_msa_feat"])
    return out


def of_make_feat(dt, masked_t, sel_d, aatype, cluster_profile, cluster_deletion_mean, crop_t, crop_d) -> dict:
    protein = {
        "msa": masked_t.clone(),
        "deletion_matrix": sel_d.clone(),
        "aatype": aatype.clone(),
        "between_segment_residues": torch.zeros(aatype.shape[0], dtype=torch.long),
        "cluster_profile": cluster_profile.clone(),
        "cluster_deletion_mean": cluster_deletion_mean.clone(),
        "extra_msa": crop_t.clone(),
        "extra_deletion_matrix": crop_d.clone(),
    }
    dt.make_msa_feat()(protein)
    from openfold.utils.feats import build_extra_msa_feat

    protein["extra_msa_feat"] = build_extra_msa_feat(protein)
    return protein


# --------------------------------------------------------------------------------------
# Integrated realize_msa_features: single-generator threading is correct
# --------------------------------------------------------------------------------------
def integrated_threading(tokens, deletions, profile, seed: int) -> dict[str, Any]:
    device = torch.device("cpu")
    dtype = torch.float32
    g = torch.Generator(device=device).manual_seed(seed)
    feats = realize_msa_features(
        tokens,
        deletions,
        profile,
        max_msa_clusters=MAX_MSA,
        max_extra_msa=MAX_EXTRA,
        msa_cluster_features=True,
        gap_agreement_weight=GAP_W,
        masked_msa_enabled=True,
        uniform_prob=UNIFORM,
        profile_prob=PROFILE,
        same_prob=SAME,
        replace_fraction=REPLACE,
        generator=g,
        device=device,
        dtype=dtype,
    )
    msa_feat, msa_mask, extra_feat, extra_mask, bert_mask, true_msa = feats

    # Manual replay off ONE generator, sample->mask->cluster->crop, mirroring realize order.
    g2 = torch.Generator(device=device).manual_seed(seed)
    sel_t, sel_d, ext_t, ext_d = _sample_msa(tokens, deletions, MAX_MSA, g2)
    m_mask = torch.ones(sel_t.shape, dtype=dtype)
    e_mask = torch.ones(ext_t.shape, dtype=dtype)
    masked_t, b_mask, t_msa = _make_masked_msa(
        sel_t,
        profile,
        uniform_prob=UNIFORM,
        profile_prob=PROFILE,
        same_prob=SAME,
        replace_fraction=REPLACE,
        generator=g2,
        device=device,
        dtype=dtype,
    )
    assign = _nearest_neighbor_clusters(masked_t, m_mask, ext_t, e_mask, GAP_W, device, dtype)
    cp, cdm = _summarize_clusters(masked_t, m_mask, sel_d, ext_t, e_mask, ext_d, assign, dtype)
    crop_t, crop_d, crop_mask = _crop_extra_msa(ext_t, ext_d, e_mask, MAX_EXTRA, g2, device)
    exp_msa_feat = _make_msa_feat(masked_t, sel_d, cp, cdm, dtype)
    exp_extra = _make_extra_msa_feat(crop_t, crop_d, dtype)
    return {
        "integrated.msa_feat": max_abs(msa_feat, exp_msa_feat),
        "integrated.extra_msa_feat": max_abs(extra_feat, exp_extra),
        "integrated.bert_mask": max_abs(bert_mask, b_mask),
        "integrated.true_msa": max_abs(true_msa, t_msa),
        "integrated.msa_mask": max_abs(msa_mask, m_mask),
        "integrated.extra_msa_mask": max_abs(extra_mask, crop_mask),
    }


# --------------------------------------------------------------------------------------
# Adversarial checks
# --------------------------------------------------------------------------------------
def adversarial(tokens, deletions, profile, dt, aatype) -> dict[str, Any]:
    device = torch.device("cpu")
    checks: dict[str, Any] = {}

    # (1) same seed -> identical draws in-process
    checks["same_seed_reproducible_inproc"] = realize_signature(tokens, deletions, profile, 7) == realize_signature(
        tokens, deletions, profile, 7
    )

    # (2) wrong seed -> DIFFERENT features (stage-1 is not trivially passing)
    checks["wrong_seed_differs"] = realize_signature(tokens, deletions, profile, 7) != realize_signature(
        tokens, deletions, profile, 8
    )

    # (3) wrong seed changes the *sampled* set too (not just masking)
    g_a = torch.Generator(device=device).manual_seed(11)
    g_b = torch.Generator(device=device).manual_seed(12)
    sa = _sample_msa(tokens, deletions, MAX_MSA, g_a)[0]
    sb = _sample_msa(tokens, deletions, MAX_MSA, g_b)[0]
    checks["wrong_seed_changes_sample"] = not torch.equal(sa, sb)

    # (4) A WRONG seed on the OpenFold side also mismatches OFoldX (guards against
    #     an accidental seed-independent pass).
    g = torch.Generator(device=device).manual_seed(101)
    sel_t = _sample_msa(tokens, deletions, MAX_MSA, g)[0]
    ref_wrong = of_sample(dt, tokens, deletions, 999)["msa"]
    checks["mismatched_seed_is_detected"] = max_abs(sel_t, ref_wrong) > 0.0

    # (5) fold() per-sample sub-seed machinery: k distinct, in order, reproducible
    def draw_samples(root_seed: int, k: int) -> list[str]:
        root = torch.Generator(device=device).manual_seed(root_seed)
        sigs = []
        for _ in range(k):
            gs = _sample_generator(root, device)
            feats = realize_msa_features(
                tokens,
                deletions,
                profile,
                max_msa_clusters=MAX_MSA,
                max_extra_msa=MAX_EXTRA,
                msa_cluster_features=True,
                gap_agreement_weight=GAP_W,
                masked_msa_enabled=True,
                uniform_prob=UNIFORM,
                profile_prob=PROFILE,
                same_prob=SAME,
                replace_fraction=REPLACE,
                generator=gs,
                device=device,
                dtype=torch.float32,
            )
            sigs.append(signature(*feats))
        return sigs

    first = draw_samples(42, 4)
    second = draw_samples(42, 4)
    checks["num_samples_distinct"] = len(set(first)) == 4
    checks["num_samples_ordered_reproducible"] = first == second
    checks["num_samples_root_seed_matters"] = first != draw_samples(43, 4)

    # (6) per-recycle resampling: consecutive draws off one advancing generator differ, reproducibly
    def two_recycles(seed: int) -> tuple[str, str]:
        g = torch.Generator(device=device).manual_seed(seed)
        kw = dict(
            max_msa_clusters=MAX_MSA,
            max_extra_msa=MAX_EXTRA,
            msa_cluster_features=True,
            gap_agreement_weight=GAP_W,
            masked_msa_enabled=True,
            uniform_prob=UNIFORM,
            profile_prob=PROFILE,
            same_prob=SAME,
            replace_fraction=REPLACE,
            device=device,
            dtype=torch.float32,
        )
        r0 = signature(*realize_msa_features(tokens, deletions, profile, generator=g, **kw))
        r1 = signature(*realize_msa_features(tokens, deletions, profile, generator=g, **kw))
        return r0, r1

    a0, a1 = two_recycles(5)
    b0, b1 = two_recycles(5)
    checks["per_recycle_resample_advances"] = a0 != a1
    checks["per_recycle_reproducible"] = (a0, a1) == (b0, b1)
    return checks


def guard_checks() -> dict[str, Any]:
    r"""fold() input-guard behavior (mutual exclusion, num_samples) via a tiny model."""

    import copy

    from ofoldx.models.alphafold2.config import AlphaFold2Config
    from ofoldx.models.alphafold2.model import AlphaFold2Model

    config = AlphaFold2Config()
    config.evoformer.num_blocks = 1
    config.evoformer.extra_msa_num_blocks = 1
    config.template.enabled = False
    config.heads.structure_module.num_layer = 1
    model = AlphaFold2Model(copy.deepcopy(config)).eval()

    results: dict[str, Any] = {}

    def raises(fn) -> bool:
        try:
            fn()
            return False
        except ValueError:
            return True

    dummy = types.SimpleNamespace(aatype=torch.ones(1, 4, dtype=torch.long))
    g = torch.Generator().manual_seed(0)
    results["mutual_exclusion_raises"] = raises(lambda: model.fold(dummy, seed=1, generator=g))
    results["num_samples_zero_raises"] = raises(lambda: model.fold(dummy, num_samples=0))
    results["num_samples_negative_raises"] = raises(lambda: model.fold(dummy, num_samples=-2))
    bad = types.SimpleNamespace(aatype=torch.ones(2, 4, dtype=torch.long))
    results["multi_system_raises"] = raises(lambda: model.fold(bad, num_samples=1))
    return results


# --------------------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference-root", type=Path, default=reference_root("openfold"))
    parser.add_argument("--num-seq", type=int, default=2048)
    parser.add_argument("--num-res", type=int, default=37)
    parser.add_argument("--pool-seed", type=int, default=2024)
    parser.add_argument("--emit-signature", type=int, default=None, help="internal: print realize signature for a seed")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    tokens, deletions, profile = build_pool(args.num_seq, args.num_res, args.pool_seed)

    if args.emit_signature is not None:
        print(realize_signature(tokens, deletions, profile, args.emit_signature))
        return

    aatype = tokens[0].clone()
    dt = load_openfold(args.reference_root)

    stage1 = stage1_per_stage(dt, tokens, deletions, profile, aatype)
    integrated = integrated_threading(tokens, deletions, profile, 314)
    adv = adversarial(tokens, deletions, profile, dt, aatype)
    guards = guard_checks()

    # Cross-process reproducibility: relaunch this script and compare signatures.
    same = _subprocess_signature(args, 7)
    diff = _subprocess_signature(args, 8)
    inproc = realize_signature(tokens, deletions, profile, 7)
    adv["cross_process_same_seed_identical"] = same == inproc
    adv["cross_process_wrong_seed_differs"] = diff != inproc

    stage1_max = max(v for v in stage1.values())
    integrated_max = max(v for v in integrated.values())
    bool_checks = {**adv, **guards}
    all_bool = all(bool(v) for v in bool_checks.values())

    report = {
        "stage1_per_stage_feature_max_abs": stage1,
        "stage1_max_abs_over_all_features": stage1_max,
        "integrated_threading_max_abs": integrated,
        "integrated_max_abs": integrated_max,
        "adversarial_and_guards": bool_checks,
        "stage1_bitwise_pass": stage1_max == 0.0,
        "integrated_threading_pass": integrated_max == 0.0,
        "all_adversarial_and_guards_pass": all_bool,
        "passed": stage1_max == 0.0 and integrated_max == 0.0 and all_bool,
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n", encoding="utf-8")
    raise SystemExit(0 if report["passed"] else 1)


def _subprocess_signature(args: argparse.Namespace, seed: int) -> str:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--num-seq",
        str(args.num_seq),
        "--num-res",
        str(args.num_res),
        "--pool-seed",
        str(args.pool_seed),
        "--emit-signature",
        str(seed),
    ]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return proc.stdout.strip()


if __name__ == "__main__":
    main()
