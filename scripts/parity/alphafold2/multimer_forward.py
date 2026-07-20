#!/usr/bin/env python
r"""Deterministic deep-forward golden parity for AlphaFold-Multimer.

ONE set of FIXED processed multimer features (built once by the OFoldX
``AlphaFold2MultimerProcessor`` for a query-only ``A-B-A`` complex, plus one
synthetic template so the Gram-Schmidt backbone-frame path is exercised) is fed
into BOTH the OpenFold >= v2.0 multimer trunk+structure+heads and the OFoldX
``AlphaFold2MultimerModel``. Identical official ``model_1_multimer_v3`` weights are
loaded into each (OFoldX via ``convert_deepmind_multimer_checkpoint`` from the
DeepMind tar; OpenFold via ``import_jax_weights_`` from the extracted npz). This
bypasses every stochastic feature-pipeline step, so the comparison is a clean
numerical check of Evoformer + template embedder + extra-MSA + structure module +
output heads over the multi-chain token axis with real ``asym/entity/sym`` identity.

The ``A-B-A`` ordering (two copies of entity A separated by entity B) makes the
per-entity ``sym_id`` copy index diverge from the global ``asym_id`` chain index,
so the multimer relative-position ``rel_chain`` featurization is genuinely tested.

Runs "both" in one process on a GPU node under the ``zyc`` venv. OpenFold's compiled
``attn_core_inplace_cuda`` kernel and the eager ``modelcif`` writers are stubbed; the
deterministic forward never selects the memory-efficient kernel.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import random
import sys
import types
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.parity.paths import reference_root, weights_root  # noqa: E402

# Quantities compared between the two multimer deep-forwards.
COMPARE_KEYS = (
    "single",
    "pair",
    "distogram_logits",
    "plddt_logits",
    "resolved_logits",
    "masked_msa_logits",
    "pae_logits",
    "final_atom_positions",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seq-a", default="GSHMKIAEN")
    parser.add_argument("--seq-b", default="DPRLVWQ")
    parser.add_argument("--model-name", default="model_1_multimer_v3")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=weights_root() / "af2-params/alphafold_params_2022-12-06.tar",
        help="DeepMind params tar (member params_<model>.npz) for the OFoldX converter",
    )
    parser.add_argument(
        "--npz",
        type=Path,
        default=None,
        help="Extracted params_<model>.npz for OpenFold import_jax_weights_ (defaults next to checkpoint)",
    )
    parser.add_argument("--recycling-steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--reference-root", type=Path, default=None, help="OpenFold >= v2.0 checkout root")
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument("--cos", type=float, default=0.9995, help="minimum cosine similarity to pass")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    _deterministic(args.seed)
    device = torch.device(args.device)

    reference_root_path = args.reference_root or reference_root("openfold-v2.2.0")
    npz_path = args.npz
    if npz_path is None:
        # Prefer a sibling extracted npz; fall back to the scratch extraction path.
        candidate = args.checkpoint.parent / f"params_{args.model_name}.npz"
        npz_path = candidate

    batch = build_multimer_features(args.seq_a, args.seq_b, seed=args.seed)
    num_res = int(batch.aatype.shape[-1])
    identity = {
        "chain_index": batch.chain_index[0].tolist(),
        "entity_index": batch.entity_index[0].tolist(),
        "symmetry_index": batch.symmetry_index[0].tolist(),
    }

    reference = run_reference(
        batch,
        reference_root=Path(reference_root_path),
        npz=Path(npz_path),
        model_name=args.model_name,
        recycling_steps=args.recycling_steps,
        device=device,
    )
    native = run_native(
        batch,
        checkpoint=args.checkpoint,
        model_name=args.model_name,
        recycling_steps=args.recycling_steps,
        device=device,
    )

    comparisons = compare(reference, native, rtol=args.rtol, cos=args.cos)
    passed = all(v["passed"] for v in comparisons.values())
    report = {
        "mode": "multimer_deep_forward",
        "reference": "openfold-multimer",
        "reference_root": str(reference_root_path),
        "model_name": args.model_name,
        "seq_a": args.seq_a,
        "seq_b": args.seq_b,
        "chain_order": "A-B-A",
        "num_residues": num_res,
        "recycling_steps": args.recycling_steps,
        "device": args.device,
        "rtol": args.rtol,
        "min_cosine": args.cos,
        "identity": identity,
        "comparisons": comparisons,
        "passed": bool(passed),
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n", encoding="utf-8")
    raise SystemExit(0 if passed else 1)


# --------------------------------------------------------------------------------------
# Fixed multimer feature construction (single source of truth, OFoldX processor)
# --------------------------------------------------------------------------------------
def build_multimer_features(seq_a: str, seq_b: str, *, seed: int) -> Any:
    r"""Build a fixed query-only ``A-B-A`` multimer input with one synthetic template.

    The template atom37 coordinates are random but non-degenerate, so the processor's
    Gram-Schmidt N/CA/C backbone frame (``_template_backbone_frame``) is a real rotation
    and the template unit-vector path is exercised end-to-end. The pseudo-beta and
    backbone frame are computed with the exact processor helpers OFoldX uses, while the
    OpenFold reference recomputes both internally from the same atom coordinates.
    """

    from dataclasses import replace as _replace

    from ofoldx.data.molecular import PolymerSequence
    from ofoldx.models.alphafold2.config import AlphaFold2MultimerMSAConfig
    from ofoldx.models.alphafold2.model import realize_multimer_msa_features
    from ofoldx.models.alphafold2.processor import (
        AlphaFold2MultimerProcessor,
        _template_backbone_frame,
        _template_pseudo_beta,
    )

    chain_a = PolymerSequence.from_one_letter(seq_a)
    chain_b = PolymerSequence.from_one_letter(seq_b)
    processor = AlphaFold2MultimerProcessor(device="cpu")
    # A-B-A: two copies of entity A separated by B, so sym_id != asym_id for the A pair.
    batch = processor({"A": chain_a, "B": chain_b, "A2": chain_a})

    # Under the stochastic-fold contract the Processor emits only the raw paired pool
    # (``msa_feat is None``); the Model draws the clustered/extra realization. Trunk
    # parity needs one FIXED feature slice injected identically into both trunks, so
    # realize a single slice here on CPU off a fixed-seed generator and fill the
    # per-recycle slots. With ``msa_feat`` present the Model consumes it verbatim
    # (``needs_realize`` is False), bypassing every stochastic draw.
    cfg = AlphaFold2MultimerMSAConfig()
    msa_generator = torch.Generator(device="cpu").manual_seed(seed)
    msa_feat, msa_mask, extra_msa_feat, extra_msa_mask, bert_mask, true_msa = realize_multimer_msa_features(
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
        generator=msa_generator,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    batch = _replace(
        batch,
        msa_feat=msa_feat.unsqueeze(0),
        msa_mask=msa_mask.unsqueeze(0),
        extra_msa_feat=extra_msa_feat.unsqueeze(0),
        extra_msa_mask=extra_msa_mask.unsqueeze(0),
        bert_mask=bert_mask.unsqueeze(0),
        true_msa=true_msa.unsqueeze(0),
    )

    num_res = int(batch.aatype.shape[-1])
    template_aatype = batch.aatype.detach().cpu()[0].unsqueeze(0).long()  # [T=1, N]
    generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    positions = torch.randn(1, num_res, 37, 3, generator=generator, dtype=torch.float32) * 8.0
    masks = torch.ones(1, num_res, 37, dtype=torch.float32)
    pseudo_beta, pseudo_beta_mask = _template_pseudo_beta(template_aatype, positions, masks)
    frame, frame_mask = _template_backbone_frame(positions, masks)

    return replace(
        batch,
        template_mask=torch.ones(1, 1),
        template_aatype=template_aatype.unsqueeze(0),
        template_all_atom_positions=positions.unsqueeze(0),
        template_all_atom_mask=masks.unsqueeze(0),
        template_pseudo_beta=pseudo_beta.unsqueeze(0),
        template_pseudo_beta_mask=pseudo_beta_mask.unsqueeze(0),
        template_backbone_frame=frame.unsqueeze(0),
        template_backbone_frame_mask=frame_mask.unsqueeze(0),
    )


# --------------------------------------------------------------------------------------
# Reference: OpenFold >= v2.0 multimer trunk+structure+heads on the fixed features
# --------------------------------------------------------------------------------------
def run_reference(
    batch: Any,
    *,
    reference_root: Path,
    npz: Path,
    model_name: str,
    recycling_steps: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    _install_stubs()
    if str(reference_root) not in sys.path:
        sys.path.insert(0, str(reference_root))

    from openfold.config import model_config
    from openfold.data import data_transforms
    from openfold.model.model import AlphaFold
    from openfold.utils.import_weights import import_jax_weights_

    config = model_config(model_name)
    config.data.common.max_recycling_iters = recycling_steps
    config.globals.chunk_size = None
    config.globals.use_lma = False
    config.globals.use_flash = False
    config.globals.use_deepspeed_evo_attention = False
    config.globals.use_memory_efficient_kernel = False
    config.globals.offload_inference = False
    # Never early-stop: OFoldX always runs the full recycle count, so the reference must too.
    with contextlib.suppress(Exception):
        config.model.recycle_early_stop_tolerance = -1.0

    of_batch = _openfold_multimer_batch(batch, recycling_steps, device, data_transforms)

    model = AlphaFold(config).eval()
    import_jax_weights_(model, str(npz), version=model_name)
    model = model.to(device)
    with torch.inference_mode():
        output = model(of_batch)

    out = {
        "single": output["single"].float().cpu(),
        "pair": output["pair"].float().cpu(),
        "final_atom_positions": output["final_atom_positions"].float().cpu(),
        "final_atom_mask": output["final_atom_mask"].bool().cpu(),
        "distogram_logits": output["distogram_logits"].float().cpu(),
        "masked_msa_logits": output["masked_msa_logits"].float().cpu(),
    }
    for key, dst in (
        ("lddt_logits", "plddt_logits"),
        ("experimentally_resolved_logits", "resolved_logits"),
        ("tm_logits", "pae_logits"),
    ):
        if key in output:
            out[dst] = output[key].float().cpu()
    return out


def _openfold_multimer_batch(
    batch: Any,
    recycling_steps: int,
    device: torch.device,
    data_transforms: Any,
) -> dict[str, torch.Tensor]:
    r"""Translate the fixed OFoldX multimer features into an OpenFold multimer batch."""

    recycles = recycling_steps + 1

    def rec(t: torch.Tensor) -> torch.Tensor:
        return torch.stack([t] * recycles, dim=-1).to(device)

    aatype = batch.aatype[0].long().cpu()
    protein = {"aatype": aatype.clone()}
    data_transforms.make_atom14_masks(protein)

    # Extra MSA: OpenFold's multimer build_extra_msa_feat rebuilds the 25-wide feature
    # from raw (extra_msa index, extra_deletion_matrix). The query-only pool yields a
    # single fully-masked dummy row; recover the raw fields from the one-hot block.
    extra = batch.extra_msa_feat[0].cpu()  # [E, N, 25]
    extra_msa = extra[..., :23].argmax(dim=-1).long()
    # deletion channels are zero for the query-only dummy row, so raw counts are zero.
    extra_deletion_matrix = torch.zeros_like(extra_msa, dtype=torch.float32)

    of: dict[str, torch.Tensor] = {
        "aatype": rec(aatype),
        "residue_index": rec(batch.residue_index[0].long().cpu()),
        "seq_mask": rec(batch.token_mask[0].float().cpu()),
        "asym_id": rec(batch.chain_index[0].long().cpu().float()),
        "entity_id": rec(batch.entity_index[0].long().cpu().float()),
        "sym_id": rec(batch.symmetry_index[0].long().cpu().float()),
        "target_feat": rec(batch.target_feat[0].float().cpu()),
        "msa_feat": rec(batch.msa_feat[0].float().cpu()),
        "msa_mask": rec(batch.msa_mask[0].float().cpu()),
        "extra_msa": rec(extra_msa),
        "extra_deletion_matrix": rec(extra_deletion_matrix),
        "extra_msa_mask": rec(batch.extra_msa_mask[0].float().cpu()),
        "template_mask": rec(batch.template_mask[0].float().cpu()),
        "template_aatype": rec(batch.template_aatype[0].long().cpu()),
        "template_all_atom_positions": rec(batch.template_all_atom_positions[0].float().cpu()),
        "template_all_atom_mask": rec(batch.template_all_atom_mask[0].float().cpu()),
        "template_pseudo_beta": rec(batch.template_pseudo_beta[0].float().cpu()),
        "template_pseudo_beta_mask": rec(batch.template_pseudo_beta_mask[0].float().cpu()),
        "atom37_atom_exists": rec(protein["atom37_atom_exists"].float()),
        "residx_atom37_to_atom14": rec(protein["residx_atom37_to_atom14"].long()),
        "residx_atom14_to_atom37": rec(protein["residx_atom14_to_atom37"].long()),
        "atom14_atom_exists": rec(protein["atom14_atom_exists"].float()),
    }
    return of


# --------------------------------------------------------------------------------------
# Native: OFoldX multimer trunk+structure+heads on the identical fixed features
# --------------------------------------------------------------------------------------
def run_native(
    batch: Any,
    *,
    checkpoint: Path,
    model_name: str,
    recycling_steps: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    from ofoldx.models.alphafold2.converters import deepmind
    from ofoldx.models.alphafold2.converters.deepmind_multimer import (
        convert_deepmind_multimer_checkpoint,
    )

    params, _resolved = deepmind._load_official_alphafold2_params(checkpoint, model_name=model_name)
    model = convert_deepmind_multimer_checkpoint(dict(params), model_name=model_name).to(device).eval()
    model.set_backend("torch")

    device_batch = _batch_to_device(batch, device)
    with torch.inference_mode():
        output = model(device_batch, recycling_steps=recycling_steps)

    out = {
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
        out["pae_logits"] = output.pae_logits.float().cpu()
    return out


def _batch_to_device(batch: Any, device: torch.device) -> Any:
    fields = (
        "aatype",
        "target_feat",
        "residue_index",
        "token_mask",
        "chain_index",
        "entity_index",
        "symmetry_index",
        "msa_feat",
        "msa_mask",
        "extra_msa_feat",
        "extra_msa_mask",
        "template_mask",
        "template_aatype",
        "template_all_atom_positions",
        "template_all_atom_mask",
        "template_pseudo_beta",
        "template_pseudo_beta_mask",
        "template_backbone_frame",
        "template_backbone_frame_mask",
    )
    moved = {name: getattr(batch, name).to(device) for name in fields}
    return replace(batch, **moved)


# --------------------------------------------------------------------------------------
# Comparison (relative L2 + cosine + max abs, mask-aware for coordinates)
# --------------------------------------------------------------------------------------
def compare(
    reference: dict[str, torch.Tensor],
    native: dict[str, torch.Tensor],
    *,
    rtol: float,
    cos: float,
) -> dict[str, dict[str, Any]]:
    comparisons: dict[str, dict[str, Any]] = {}
    for key in COMPARE_KEYS:
        if key not in reference or key not in native:
            continue
        if key == "final_atom_positions":
            # Squeeze both to canonical ranks: positions [N, 37, 3], mask [N, 37].
            ref = _squeeze_to(reference[key], 3)
            act = _squeeze_to(native[key], 3)
            ref_mask = _squeeze_to(reference["final_atom_mask"].bool(), 2)
            act_mask = _squeeze_to(native["final_atom_mask"].bool(), 2)
            comparisons["final_atom_mask_equal"] = {
                "passed": bool(ref_mask.shape == act_mask.shape and torch.equal(ref_mask, act_mask)),
            }
            common = ref_mask & act_mask
            comparisons[key] = _metric(act[common], ref[common], rtol=rtol, cos=cos)
        else:
            ref, act = _align(reference[key], native[key])
            comparisons[key] = _metric(act, ref, rtol=rtol, cos=cos)
    return comparisons


def _squeeze_to(t: torch.Tensor, ndim: int) -> torch.Tensor:
    while t.dim() > ndim and t.shape[0] == 1:
        t = t[0]
    return t


def _align(reference: torch.Tensor, actual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    target = min(reference.dim(), actual.dim())
    return _squeeze_to(reference, target), _squeeze_to(actual, target)


def _metric(actual: torch.Tensor, reference: torch.Tensor, *, rtol: float, cos: float) -> dict[str, Any]:
    actual = actual.detach().float().cpu().reshape(-1)
    reference = reference.detach().float().cpu().reshape(-1)
    if actual.shape != reference.shape:
        return {
            "expected_shape": list(reference.shape),
            "actual_shape": list(actual.shape),
            "passed": False,
        }
    diff = actual - reference
    max_abs = float(diff.abs().max()) if diff.numel() else 0.0
    ref_norm = float(reference.norm())
    rel_l2 = float(diff.norm() / (ref_norm + 1e-12)) if diff.numel() else 0.0
    denom = (actual.norm() * reference.norm()).clamp_min(1e-12)
    cosine = float((actual @ reference) / denom) if diff.numel() else 1.0
    passed = bool(rel_l2 <= rtol and cosine >= cos)
    return {
        "count": int(reference.numel()),
        "max_absolute_error": max_abs,
        "relative_l2": rel_l2,
        "cosine": cosine,
        "passed": passed,
    }


# --------------------------------------------------------------------------------------
# Environment stubs / determinism
# --------------------------------------------------------------------------------------
def _install_stubs() -> None:
    for name in (
        "modelcif",
        "modelcif.dumper",
        "modelcif.model",
        "modelcif.protocol",
        "modelcif.reference",
        "modelcif.alignment",
        "modelcif.qa_metric",
        "modelcif.dictionary",
    ):
        sys.modules.setdefault(name, types.ModuleType(name))
    if "attn_core_inplace_cuda" not in sys.modules:
        stub = types.ModuleType("attn_core_inplace_cuda")

        def _forward_(logits: torch.Tensor, rows: int, cols: int) -> None:
            logits.copy_(torch.softmax(logits, dim=-1))

        def _backward_(*a: Any, **k: Any) -> None:
            raise RuntimeError("attn_core_inplace_cuda backward is unavailable in the parity stub")

        stub.forward_ = _forward_
        stub.backward_ = _backward_
        sys.modules["attn_core_inplace_cuda"] = stub


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
