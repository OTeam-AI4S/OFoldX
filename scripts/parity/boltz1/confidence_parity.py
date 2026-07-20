"""Boltz-1 confidence-head numerical-parity harness (pLDDT / PAE / PDE / resolved logits + pTM/ipTM).

Mirrors ``scripts.parity.boltz2.confidence_parity``, adapted for Boltz-1's ``imitate_trunk`` confidence
module (which re-embeds inputs and runs its own input-embedder -> recycle -> MSA -> Pairformer stack before
projecting to logits). Drives the released Boltz-1 :class:`ConfidenceModule` (loaded module-by-module from
``boltz1_conf.ckpt``) and the OFoldX :class:`~ofoldx.models.boltz1.model.Boltz1ConfidenceHead` (loaded via
the released converter from the same checkpoint) on one shared fixed structure input, comparing the four
confidence logit tensors and the derived pTM / ipTM / ligand-ipTM / protein-ipTM ranking scores
(model.py:3861-3864).

The pTM/ipTM frame masks are compared on an all-protein structure with fixed, non-collinear three-atom
frames so the reference frame-derivation heuristic (which re-derives nonpolymer frames by distance sorting
and uses a 25-degree collinearity gate) and OFoldX's fixed-frame path agree; that heuristic difference is a
known design divergence and is out of scope for the pTM arithmetic tested here.

The reference and OFoldX stacks share the zyc environment, so this runs in one process (CPU/fp32).

    python -m scripts.parity.boltz1.confidence_parity
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.parity.boltz1.forward_parity import (  # noqa: E402
    NUM_TOKENS,
    B,
    M,
    N,
    _ensure_fairscale_stub,
    allo_input,
    build_feats,
    load_checkpoint,
    subset,
)
from scripts.parity.paths import reference_root, weight_path  # noqa: E402

REFERENCE_SRC = reference_root("boltz") / "src"
if str(REFERENCE_SRC) not in sys.path:
    sys.path.insert(0, str(REFERENCE_SRC))

TOKEN_S, TOKEN_Z = 384, 128


def compare(name: str, a: torch.Tensor, b: torch.Tensor, *, atol: float = 1e-3) -> tuple[bool, float]:
    a, b = a.float(), b.float()
    if a.shape != b.shape:
        print(f"[SHAPE] {name}: {tuple(a.shape)} vs {tuple(b.shape)}", flush=True)
        return False, float("inf")
    diff = (a - b).abs()
    max_abs = float(diff.max()) if diff.numel() else 0.0
    rel = max_abs / (float(b.abs().max()) + 1e-9)
    ok = max_abs < atol
    print(
        f"[{'ok ' if ok else 'BAD'}] {name}: max_abs={max_abs:.3e} mean_abs="
        f"{float(diff.mean()) if diff.numel() else 0.0:.3e} max_rel={rel:.3e} shape={tuple(a.shape)}",
        flush=True,
    )
    return ok, max_abs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=weight_path("boltz1_conf.ckpt"))
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--atol", type=float, default=1e-3)
    return parser.parse_args()


def build_reference(state: dict, hparams: dict, device: torch.device):
    from boltz.model.modules.confidence import ConfidenceModule

    unused = {"activation_checkpointing", "offload_to_cpu"}
    pairformer_args = {k: v for k, v in hparams["pairformer_args"].items() if k not in unused}
    msa_args = {k: v for k, v in hparams["msa_args"].items() if k not in unused}
    full_embedder_args = {
        "atom_s": int(hparams["atom_s"]),
        "atom_z": int(hparams["atom_z"]),
        "token_s": TOKEN_S,
        "token_z": TOKEN_Z,
        "atoms_per_window_queries": int(hparams["atoms_per_window_queries"]),
        "atoms_per_window_keys": int(hparams["atoms_per_window_keys"]),
        "atom_feature_dim": int(hparams["atom_feature_dim"]),
        "no_atom_encoder": bool(hparams.get("no_atom_encoder", False)),
        **hparams["embedder_args"],
    }
    module = ConfidenceModule(
        TOKEN_S,
        TOKEN_Z,
        compute_pae=float(hparams["alpha_pae"]) > 0,
        imitate_trunk=True,
        pairformer_args=pairformer_args,
        full_embedder_args=full_embedder_args,
        msa_args=msa_args,
        **hparams["confidence_model_args"],
    )
    module.load_state_dict(subset(state, "confidence_module"), strict=True)
    return module.to(device).eval()


def build_native(state: dict, device: torch.device):
    from ofoldx.models.boltz1 import Boltz1Config
    from ofoldx.models.boltz1.convert import convert_boltz1_state_dict
    from ofoldx.models.boltz1.model import Boltz1Model

    with torch.device("meta"):
        model = Boltz1Model(Boltz1Config())
    model.load_state_dict(convert_boltz1_state_dict(state), strict=True, assign=True)
    model = model.to(device).eval().set_backend("torch")
    assert model.confidence_head is not None
    return model.confidence_head


def build_structure_inputs(device: torch.device, seed: int):
    r"""One shared fixed structure input: all-protein tokens with fixed non-collinear three-atom frames."""

    feats = build_feats(seed=1, device=device)
    feats["mol_type"] = torch.zeros(B, N, dtype=torch.long, device=device)  # all-protein -> shared frame path
    gen = torch.Generator().manual_seed(seed)

    rep_atom_index = torch.randint(0, M, (B, N), generator=gen).to(device)
    frames_idx = torch.tensor([0, 1, 2], device=device).view(1, 1, 3).expand(B, N, 3).contiguous()
    frame_mask = torch.ones(B, N, dtype=torch.bool, device=device)

    x_pred = torch.randn(B, M, 3, generator=gen).to(device)
    # Force atoms 0/1/2 into a clearly non-collinear (near right-angle) frame so both frame-validity criteria
    # (reference 25-degree gate; OFoldX cross-product norm) mark every token's frame valid.
    x_pred[:, 0] = torch.tensor([0.0, 0.0, 0.0], device=device)
    x_pred[:, 1] = torch.tensor([10.0, 0.0, 0.0], device=device)
    x_pred[:, 2] = torch.tensor([0.0, 10.0, 0.0], device=device)

    single = torch.randn(B, N, TOKEN_S, generator=gen).to(device)
    pair = torch.randn(B, N, N, TOKEN_Z, generator=gen).to(device)
    s_diffusion = torch.randn(B, N, 2 * TOKEN_S, generator=gen).to(device)
    s_inputs = torch.randn(B, N, 455, generator=gen).to(device)  # ignored by imitate_trunk (re-embedded)
    pred_distogram = torch.randn(B, N, N, 64, generator=gen).to(device)

    # OFoldX batch: index contracts for atom_to_token / token_to_rep_atom + explicit frames.
    batch = allo_input(feats)
    batch.token_to_rep_atom = rep_atom_index
    batch.frame_atom_index = frames_idx
    batch.frame_mask = frame_mask

    # Reference feats: dense one-hot selection matrices, one-hot MSA, and the [..., 1] token-bond channel.
    ref_feats = {
        **feats,
        "token_to_rep_atom": torch.nn.functional.one_hot(rep_atom_index, M).float(),
        "frames_idx": frames_idx,
        "frame_resolved_mask": frame_mask.float(),
        "token_bonds": feats["token_bonds"].unsqueeze(-1),
        "msa": torch.nn.functional.one_hot(feats["msa"].long(), NUM_TOKENS).to(feats["ref_pos"].dtype),
    }
    shared = {
        "single": single,
        "pair": pair,
        "s_diffusion": s_diffusion,
        "s_inputs": s_inputs,
        "x_pred": x_pred,
        "pred_distogram": pred_distogram,
    }
    return batch, feats, ref_feats, shared


def main() -> None:
    args = parse_args()
    _ensure_fairscale_stub()
    device = torch.device("cpu")
    print(f"[device] {device}", flush=True)

    state, hparams = load_checkpoint(args.checkpoint)
    reference = build_reference(state, hparams, device)
    native = build_native(state, device)
    batch, feats, ref_feats, shared = build_structure_inputs(device, args.seed)

    with torch.no_grad():
        ref_out = reference(
            shared["s_inputs"],
            shared["single"],
            shared["pair"],
            shared["x_pred"],
            ref_feats,
            shared["pred_distogram"],
            multiplicity=1,
            s_diffusion=shared["s_diffusion"],
        )
        mine_out = native(
            shared["s_inputs"],
            shared["single"],
            shared["pair"],
            shared["x_pred"],
            batch=batch,
            pred_distogram_logits=shared["pred_distogram"],
            single_diffusion=shared["s_diffusion"],
            multiplicity=1,
        )

    results: list[tuple[bool, float]] = []
    print("\n=== confidence logits ===", flush=True)
    for name in ("pae_logits", "pde_logits", "plddt_logits", "resolved_logits"):
        results.append(compare(name, mine_out[name], ref_out[name], atol=args.atol))
    print("\n=== pTM / ipTM ranking metrics ===", flush=True)
    for name in ("ptm", "iptm", "ligand_iptm", "protein_iptm"):
        results.append(compare(name, mine_out[name], ref_out[name], atol=args.atol))

    worst = max((deviation for _, deviation in results), default=0.0)
    ok = all(passed for passed, _ in results)
    print(f"\n[boltz1] confidence worst max_abs = {worst:.3e}  tolerance = {args.atol:.1e}", flush=True)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
