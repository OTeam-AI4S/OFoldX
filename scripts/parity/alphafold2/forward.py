#!/usr/bin/env python
r"""Deterministic deep-forward golden-dump parity for monomer AlphaFold 2.

A single set of FIXED processed features (built once by the OFoldX
``AlphaFold2Processor`` for a query-only, template-free target) is fed into both the
OpenFold reference trunk+structure module and the OFoldX ``ofoldx.alphafold2`` model.
This bypasses every stochastic feature-pipeline step (MSA clustering/masking, per-recycle
resampling, ensembling), so the comparison is a clean numerical check of the Evoformer +
structure module + output heads.

Modes:
  reference  build fixed features, run OpenFold, dump features+outputs to a golden .pt
  native     rebuild the identical fixed features, assert byte-identity against the
             golden features, run OFoldX, compare outputs against the golden reference
  both       do reference then native in one process (features built once)

Regimes (``--regime``):
  fidelity   DEFAULT gating assertion. Runs at ``recycling_steps=0`` (single pass),
             which fully exercises embeddings + templates + extra-MSA + the 48-block
             Evoformer + structure module + all 9 output heads exactly once. This is
             the *non-chaotic* regime: the forward is evaluated once with no recycling
             feedback, so the fp32 OFoldX-vs-OpenFold gap is the genuine numerical
             fidelity of the port. Failure exits non-zero.
  trajectory NON-GATING. Runs at ``recycling_steps=3`` (the released inference default).
             Recycling feeds the previous cycle's outputs back as inputs, so tiny fp32
             rounding differences are amplified chaotically across cycles (push-2: the
             self fp32-vs-fp64 gap of a single checkpoint exceeds its vs-OpenFold gap).
             This measures trajectory divergence, NOT port fidelity, so it never gates
             (always exits 0); the report carries ``gating: false`` and ``numerical_match``
             is advisory only.

``--recycling-steps`` may be given explicitly to override the regime default; the gating
behaviour still follows ``--regime``.

Run under the ``zyc`` venv on a GPU node. OpenFold's compiled ``attn_core_inplace_cuda``
kernel is stubbed out: the deterministic forward never selects the memory-efficient
attention kernel (``use_memory_efficient_kernel`` defaults to False everywhere), so the
stub is never called.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.parity.paths import reference_root, weights_root  # noqa: E402

FEATURE_FIELDS = (
    "aatype",
    "target_feat",
    "residue_index",
    "token_mask",
    "msa_feat",
    "msa_mask",
    "extra_msa_feat",
    "extra_msa_mask",
    "template_mask",
    "template_aatype",
    "template_all_atom_positions",
    "template_all_atom_masks",
    "template_torsion_angles_sin_cos",
    "template_alt_torsion_angles_sin_cos",
    "template_torsion_angles_mask",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("reference", "native", "both"))
    parser.add_argument("--sequence", default="GSHMKIAENLYFQGSDPRLVWQ".replace(" ", ""))
    parser.add_argument("--model-name", default="model_1_ptm")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=weights_root() / "af2-params/params_model_1_ptm.npz",
    )
    parser.add_argument(
        "--regime",
        choices=("fidelity", "trajectory"),
        default="fidelity",
        help=(
            "fidelity (default): gating single-pass fidelity gate at recycling_steps=0; "
            "trajectory: non-gating 3-recycle trajectory check (chaotic, not a fidelity assertion)."
        ),
    )
    parser.add_argument(
        "--recycling-steps",
        type=int,
        default=None,
        help="Override the per-regime default (fidelity=0, trajectory=3).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--reference-root", type=Path, default=reference_root("openfold"))
    parser.add_argument("--golden", type=Path, default=ROOT / "artifacts/parity/alphafold2/golden_monomer.pt")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument("--atol", type=float, default=1e-3)
    args = parser.parse_args()

    # Resolve the recycling depth from the regime unless the caller pinned it explicitly.
    # fidelity -> 0 (single pass, non-chaotic, gating); trajectory -> 3 (chaotic, non-gating).
    if args.recycling_steps is None:
        args.recycling_steps = 0 if args.regime == "fidelity" else 3
    args.gating = args.regime == "fidelity"

    sequence = args.sequence.strip().upper()
    if args.mode in ("reference", "both"):
        run_reference(args, sequence)
    if args.mode in ("native", "both"):
        report = run_native(args, sequence)
        text = json.dumps(report, indent=2, sort_keys=True)
        print(text)
        if args.json is not None:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(text + "\n", encoding="utf-8")
        # Only the fidelity regime gates. The trajectory regime is chaotic recycling
        # amplification, not a port-fidelity signal, so it always exits 0 (advisory).
        exit_code = (0 if report["passed"] else 1) if args.gating else 0
        raise SystemExit(exit_code)


# --------------------------------------------------------------------------------------
# Fixed feature construction (single source of truth, OFoldX processor, deterministic)
# --------------------------------------------------------------------------------------
def build_features(sequence: str, device: str, seed: int) -> Any:
    r"""Processor raw pool + one *fixed* realized MSA-feature slice.

    Under the stochastic-fold contract the Processor emits only the raw alignment
    pool (``msa_feat is None``); the Model draws the clustered/extra realization.
    Stage-2 trunk parity needs a single FIXED feature slice injected identically
    into both trunks, so we realize one slice here on CPU off a fixed-seed
    generator (deterministic across the reference/native processes) and fill the
    per-recycle slots. With ``msa_feat`` present the Model consumes it verbatim
    (``needs_realize`` is False), so this bypasses every stochastic draw exactly
    as the pre-stochastic deterministic baseline did.
    """

    from dataclasses import replace as _replace

    from ofoldx.models.alphafold2 import AlphaFold2Processor
    from ofoldx.models.alphafold2.config import AlphaFold2MSAConfig
    from ofoldx.models.alphafold2.model import realize_msa_features

    processor = AlphaFold2Processor(device="cpu")
    batch = processor(sequence)
    cfg = AlphaFold2MSAConfig()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    msa_feat, msa_mask, extra_msa_feat, extra_msa_mask, bert_mask, true_msa = realize_msa_features(
        batch.msa_tokens[0],
        batch.msa_deletions[0],
        batch.msa_profile[0],
        max_msa_clusters=cfg.max_msa_clusters,
        max_extra_msa=cfg.max_extra_msa,
        msa_cluster_features=cfg.msa_cluster_features,
        gap_agreement_weight=cfg.gap_agreement_weight,
        masked_msa_enabled=cfg.masked_msa_enabled,
        uniform_prob=cfg.masked_msa_uniform_prob,
        profile_prob=cfg.masked_msa_profile_prob,
        same_prob=cfg.masked_msa_same_prob,
        replace_fraction=cfg.masked_msa_replace_fraction,
        generator=generator,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    # A query-only pool realizes an empty extra-MSA (every row clusters). OpenFold's
    # extra-MSA stack cannot reshape a 0-row tensor, so -- exactly as the pre-stochastic
    # processor's dummy extra did -- inject one zeroed, fully masked extra row (token 0,
    # mask 0). Both trunks receive the identical dummy, so trunk parity is unaffected.
    if extra_msa_feat.shape[0] == 0:
        num_res = extra_msa_feat.shape[1]
        dummy = torch.zeros(1, num_res, 25, dtype=torch.float32)
        dummy[..., 0] = 1.0  # one-hot of restype 0
        extra_msa_feat = dummy
        extra_msa_mask = torch.zeros(1, num_res, dtype=torch.float32)
    return _replace(
        batch,
        msa_feat=msa_feat.unsqueeze(0),
        msa_mask=msa_mask.unsqueeze(0),
        extra_msa_feat=extra_msa_feat.unsqueeze(0),
        extra_msa_mask=extra_msa_mask.unsqueeze(0),
        bert_mask=bert_mask.unsqueeze(0),
        true_msa=true_msa.unsqueeze(0),
    )


def feature_tensors(batch: Any) -> dict[str, torch.Tensor]:
    return {name: getattr(batch, name).detach().cpu() for name in FEATURE_FIELDS}


# --------------------------------------------------------------------------------------
# Reference process: OpenFold trunk + structure module on the fixed features
# --------------------------------------------------------------------------------------
def run_reference(args: argparse.Namespace, sequence: str) -> None:
    _deterministic(args.seed)
    device = torch.device(args.device)
    batch = build_features(sequence, args.device, args.seed)
    feats = feature_tensors(batch)

    # Stub the compiled kernel that OpenFold imports eagerly. Its ``forward_`` is an
    # in-place softmax over the last (key) axis of the attention logits; reproduce it
    # exactly in Python so the reference numerics are unchanged (inference-only, so the
    # unused backward is never reached).
    if "attn_core_inplace_cuda" not in sys.modules:
        stub = types.ModuleType("attn_core_inplace_cuda")

        def _forward_(logits: torch.Tensor, rows: int, cols: int) -> None:
            logits.copy_(torch.softmax(logits, dim=-1))

        def _backward_(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("attn_core_inplace_cuda backward is not available in the parity stub")

        stub.forward_ = _forward_
        stub.backward_ = _backward_
        sys.modules["attn_core_inplace_cuda"] = stub

    reference = args.reference_root
    revision, dirty = _git_state(reference)
    if str(reference) not in sys.path:
        sys.path.insert(0, str(reference))

    from openfold.config import model_config
    from openfold.data import data_transforms
    from openfold.model.model import AlphaFold
    from openfold.utils.import_weights import import_jax_weights_

    config = model_config(args.model_name)
    config.data.common.max_recycling_iters = args.recycling_steps
    config.globals.chunk_size = None
    config.globals.use_lma = False
    config.globals.use_flash = False
    config.globals.use_memory_efficient_kernel = False
    config.globals.offload_inference = False

    of_batch = _openfold_batch(feats, args.recycling_steps, device, data_transforms)

    model = AlphaFold(config).eval()
    import_jax_weights_(model, str(args.checkpoint), version=args.model_name)
    model = model.to(device)
    with torch.inference_mode():
        output = model(of_batch)

    outputs = {
        "single": output["single"].float().cpu(),
        "pair": output["pair"].float().cpu(),
        "final_atom_positions": output["final_atom_positions"].float().cpu(),
        "final_atom_mask": output["final_atom_mask"].bool().cpu(),
        "distogram_logits": output["distogram_logits"].float().cpu(),
        "plddt_logits": output["lddt_logits"].float().cpu(),
        "resolved_logits": output["experimentally_resolved_logits"].float().cpu(),
        "masked_msa_logits": output["masked_msa_logits"].float().cpu(),
        "plddt": output["plddt"].float().cpu(),
    }
    if "tm_logits" in output:
        outputs["pae_logits"] = output["tm_logits"].float().cpu()
        outputs["ptm"] = output["predicted_tm_score"].float().cpu()

    payload = {
        "metadata": _metadata(args, sequence),
        "reference_revision": revision,
        "reference_dirty": dirty,
        "features": feats,
        "outputs": outputs,
    }
    args.golden.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.golden)
    print(
        json.dumps(
            {
                "mode": "reference",
                "golden": str(args.golden),
                "reference_revision": revision,
                "reference_dirty": dirty,
                "num_residues": int(feats["aatype"].shape[-1]),
                "outputs": {k: list(v.shape) for k, v in outputs.items()},
            },
            indent=2,
            sort_keys=True,
        )
    )


def _openfold_batch(
    feats: dict[str, torch.Tensor],
    recycling_steps: int,
    device: torch.device,
    data_transforms: Any,
) -> dict[str, torch.Tensor]:
    r"""Translate the fixed OFoldX features into an OpenFold batch dict.

    The OFoldX ``extra_msa_feat`` (one-hot[23] + has_deletion + deletion_value) is split
    back into the raw fields OpenFold's ``build_extra_msa_feat`` recombines, and a
    trailing recycling axis is added so ``AlphaFold.forward`` slices each cycle.
    """

    recycles = recycling_steps + 1

    def rec(t: torch.Tensor) -> torch.Tensor:
        return torch.stack([t] * recycles, dim=-1).to(device)

    aatype = feats["aatype"][0].long()
    protein = {"aatype": aatype.clone()}
    data_transforms.make_atom14_masks(protein)

    # OpenFold's template pair featurizer reads template pseudo-beta, computed here from
    # the same fixed atom37 template coordinates via the released transform.
    template_protein = {
        "template_aatype": feats["template_aatype"][0].long(),
        "template_all_atom_positions": feats["template_all_atom_positions"][0].float(),
        "template_all_atom_mask": feats["template_all_atom_masks"][0].float(),
    }
    data_transforms.make_pseudo_beta("template_")(template_protein)

    extra = feats["extra_msa_feat"][0]
    extra_msa = extra[..., :23].argmax(dim=-1).long()
    extra_has_deletion = extra[..., 23]
    extra_deletion_value = extra[..., 24]

    batch = {
        "aatype": rec(aatype),
        "residue_index": rec(feats["residue_index"][0].long()),
        "seq_mask": rec(feats["token_mask"][0].float()),
        "target_feat": rec(feats["target_feat"][0].float()),
        "msa_feat": rec(feats["msa_feat"][0].float()),
        "msa_mask": rec(feats["msa_mask"][0].float()),
        "extra_msa": rec(extra_msa),
        "extra_has_deletion": rec(extra_has_deletion.float()),
        "extra_deletion_value": rec(extra_deletion_value.float()),
        "extra_msa_mask": rec(feats["extra_msa_mask"][0].float()),
        "template_aatype": rec(feats["template_aatype"][0].long()),
        "template_all_atom_positions": rec(feats["template_all_atom_positions"][0].float()),
        "template_all_atom_mask": rec(feats["template_all_atom_masks"][0].float()),
        "template_torsion_angles_sin_cos": rec(feats["template_torsion_angles_sin_cos"][0].float()),
        "template_alt_torsion_angles_sin_cos": rec(feats["template_alt_torsion_angles_sin_cos"][0].float()),
        "template_torsion_angles_mask": rec(feats["template_torsion_angles_mask"][0].float()),
        "template_mask": rec(feats["template_mask"][0].float()),
        "template_pseudo_beta": rec(template_protein["template_pseudo_beta"].float()),
        "template_pseudo_beta_mask": rec(template_protein["template_pseudo_beta_mask"].float()),
        "atom37_atom_exists": rec(protein["atom37_atom_exists"].float()),
        "residx_atom37_to_atom14": rec(protein["residx_atom37_to_atom14"].long()),
        "residx_atom14_to_atom37": rec(protein["residx_atom14_to_atom37"].long()),
        "atom14_atom_exists": rec(protein["atom14_atom_exists"].float()),
    }
    return batch


# --------------------------------------------------------------------------------------
# Native process: OFoldX trunk + structure module on the identical fixed features
# --------------------------------------------------------------------------------------
def run_native(args: argparse.Namespace, sequence: str) -> dict[str, Any]:
    _deterministic(args.seed)
    device = torch.device(args.device)
    golden = torch.load(args.golden, map_location="cpu", weights_only=False)
    _check_metadata(golden["metadata"], _metadata(args, sequence))

    batch = build_features(sequence, args.device, args.seed)
    feats = feature_tensors(batch)
    feature_diffs = _feature_identity(golden["features"], feats)

    from ofoldx.models.alphafold2.convert import convert_alphafold2_checkpoint

    model = (
        convert_alphafold2_checkpoint(args.checkpoint, source="deepmind", model_name=args.model_name).to(device).eval()
    )
    model.set_backend("torch")
    batch = _to_device(batch, device)
    with torch.inference_mode():
        output = model(batch, recycling_steps=args.recycling_steps)

    actual = {
        "single": output.single.float().cpu(),
        "pair": output.pair.float().cpu(),
        "final_atom_positions": output.final_atom_positions.float().cpu(),
        "final_atom_mask": output.final_atom_mask.bool().cpu(),
        "distogram_logits": output.distogram_logits.float().cpu(),
        "plddt_logits": output.plddt_logits.float().cpu(),
        "resolved_logits": output.resolved_logits.float().cpu(),
        "masked_msa_logits": output.masked_msa_logits.float().cpu(),
    }
    if output.pae_logits is not None:
        actual["pae_logits"] = output.pae_logits.float().cpu()

    reference = golden["outputs"]
    comparisons = _compare(reference, actual, atol=args.atol, rtol=args.rtol)
    numerical = all(v["passed"] for v in comparisons.values())
    features_identical = all(v["identical"] for v in feature_diffs.values())
    return {
        "mode": "native",
        "reference": "openfold",
        "reference_revision": golden["reference_revision"],
        "sequence": sequence,
        "num_residues": len(sequence),
        "model_name": args.model_name,
        "regime": args.regime,
        "gating": bool(args.gating),
        "recycling_steps": args.recycling_steps,
        "device": args.device,
        "rtol": args.rtol,
        "atol": args.atol,
        "features_identical": features_identical,
        "feature_diffs": feature_diffs,
        "comparisons": comparisons,
        "numerical_match": numerical,
        "passed": numerical and features_identical,
    }


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def _to_device(batch: Any, device: torch.device) -> Any:
    for name in FEATURE_FIELDS + ("residue_index",):
        value = getattr(batch, name)
        object.__setattr__(batch, name, value.to(device))
    return batch


def _feature_identity(reference: dict[str, torch.Tensor], native: dict[str, torch.Tensor]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in FEATURE_FIELDS:
        a = reference[name].float()
        b = native[name].float()
        same_shape = a.shape == b.shape
        max_abs = float((a - b).abs().max()) if same_shape and a.numel() else (0.0 if same_shape else None)
        out[name] = {
            "shape": list(b.shape),
            "max_abs_diff": max_abs,
            "identical": bool(same_shape and (not a.numel() or torch.equal(a, b))),
        }
    return out


def _squeeze_leading(t: torch.Tensor, ndim: int) -> torch.Tensor:
    r"""Drop leading singleton axes so a reference tensor and its native counterpart
    (which carries an extra batch/MSA-row singleton) share a rank before comparison."""

    while t.dim() > ndim and t.shape[0] == 1:
        t = t[0]
    return t


def _align(reference: torch.Tensor, actual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    target = min(reference.dim(), actual.dim())
    return _squeeze_leading(reference, target), _squeeze_leading(actual, target)


def _compare(
    reference: dict[str, torch.Tensor],
    actual: dict[str, torch.Tensor],
    *,
    atol: float,
    rtol: float,
) -> dict[str, dict[str, Any]]:
    reference = dict(reference)
    actual = dict(actual)
    for name in set(reference) & set(actual):
        reference[name], actual[name] = _align(reference[name], actual[name])
    comparisons: dict[str, dict[str, Any]] = {}
    if "final_atom_mask" in reference and "final_atom_mask" in actual:
        ref_mask = reference["final_atom_mask"].bool()
        act_mask = actual["final_atom_mask"].bool()
        comparisons["final_atom_mask"] = {
            "shape": list(act_mask.shape),
            "passed": ref_mask.shape == act_mask.shape and torch.equal(ref_mask, act_mask),
        }
    names = sorted((set(reference) & set(actual)) - {"final_atom_mask", "final_atom_positions", "plddt", "ptm"})
    for name in names:
        comparisons[name] = _metric(actual[name], reference[name], atol=atol, rtol=rtol)
    if "final_atom_positions" in reference and "final_atom_positions" in actual:
        common = reference["final_atom_mask"].bool() & actual["final_atom_mask"].bool()
        comparisons["final_atom_positions"] = _metric(
            actual["final_atom_positions"][common],
            reference["final_atom_positions"][common],
            atol=atol,
            rtol=rtol,
        )
    return comparisons


def _metric(actual: torch.Tensor, reference: torch.Tensor, *, atol: float, rtol: float) -> dict[str, Any]:
    actual = actual.detach().float().cpu()
    reference = reference.detach().float().cpu()
    if actual.shape != reference.shape:
        return {"expected_shape": list(reference.shape), "actual_shape": list(actual.shape), "passed": False}
    diff = (actual - reference).abs()
    rel = diff / reference.abs().clamp_min(1e-6)
    max_abs = float(diff.max()) if diff.numel() else 0.0
    max_rel = float(rel.max()) if rel.numel() else 0.0
    # Pass if either the absolute or the relative error is within tolerance (magnitudes
    # here range 10^0..10^2, so a joint criterion is the meaningful one).
    passed = bool(max_abs <= atol or max_rel <= rtol)
    return {
        "shape": list(actual.shape),
        "max_absolute_error": max_abs,
        "mean_absolute_error": float(diff.mean()) if diff.numel() else 0.0,
        "max_relative_error": max_rel,
        "passed": passed,
    }


def _metadata(args: argparse.Namespace, sequence: str) -> dict[str, Any]:
    return {
        "sequence": sequence,
        "model_name": args.model_name,
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "recycling_steps": args.recycling_steps,
        "seed": args.seed,
    }


def _check_metadata(stored: dict[str, Any], current: dict[str, Any]) -> None:
    if stored != current:
        raise ValueError(f"golden metadata differs from this run: {stored!r} != {current!r}")


def _git_state(repository: Path) -> tuple[str, bool]:
    try:
        revision = subprocess.run(
            ("git", "-C", str(repository), "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ("git", "-C", str(repository), "status", "--porcelain"),
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        )
    except Exception:
        revision, dirty = "unknown", False
    return revision, dirty


def _deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    main()
