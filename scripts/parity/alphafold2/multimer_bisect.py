#!/usr/bin/env python
r"""Stage-by-stage bisection of the AF-Multimer deep-forward (single cycle).

Feeds ONE fixed feature set into both OFoldX and OpenFold multimer models with
recycling disabled (recycling_steps=0) and captures, via forward hooks, the pair /
single activations at each trunk stage (input embedder -> template -> extra-MSA ->
evoformer -> structure). Reports rel-L2 + cosine per stage so the FIRST divergence
is localized instead of comparing chaotically-amplified 3-recycle outputs.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.parity.alphafold2.multimer_forward import (  # noqa: E402
    _batch_to_device,
    _deterministic,
    _install_stubs,
    _openfold_multimer_batch,
    build_multimer_features,
)


def _t(x: Any) -> torch.Tensor | None:
    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu()
    if isinstance(x, (tuple, list)) and x and isinstance(x[0], torch.Tensor):
        return x[0].detach().float().cpu()
    return None


def _sq(t: torch.Tensor) -> torch.Tensor:
    while t.dim() > 1 and t.shape[0] == 1:
        t = t[0]
    return t


def _cmp(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    a = _sq(a).reshape(-1)
    b = _sq(b).reshape(-1)
    if a.shape != b.shape:
        return {"a_shape": list(a.shape), "b_shape": list(b.shape), "shape_ok": False}
    diff = a - b
    rel = float(diff.norm() / (b.norm() + 1e-12))
    cos = float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-12))
    return {
        "n": int(a.numel()),
        "rel_l2": round(rel, 6),
        "cosine": round(cos, 6),
        "max_abs": round(float(diff.abs().max()), 5),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-a", default="GSHMKIAEN")
    ap.add_argument("--seq-b", default="DPRLVWQ")
    ap.add_argument("--model-name", default="model_1_multimer_v3")
    ap.add_argument(
        "--checkpoint", type=Path, default=Path("/data/users/z/weights/af2-params/alphafold_params_2022-12-06.tar")
    )
    ap.add_argument("--npz", type=Path, default=Path("/data/users/z/weights/af2-params/params_model_1_multimer_v3.npz"))
    ap.add_argument("--reference-root", type=Path, default=Path("/data/users/z/reference/openfold-v2.2.0"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    _deterministic(0)
    device = torch.device(args.device)
    batch = build_multimer_features(args.seq_a, args.seq_b, seed=0)

    # ---- OpenFold, recycle 0 ----
    _install_stubs()
    if str(args.reference_root) not in sys.path:
        sys.path.insert(0, str(args.reference_root))
    from openfold.config import model_config
    from openfold.data import data_transforms
    from openfold.model.model import AlphaFold
    from openfold.utils.import_weights import import_jax_weights_

    cfg = model_config(args.model_name)
    cfg.data.common.max_recycling_iters = 0
    cfg.globals.chunk_size = None
    cfg.globals.use_lma = False
    cfg.globals.use_flash = False
    cfg.globals.use_deepspeed_evo_attention = False
    cfg.globals.use_memory_efficient_kernel = False
    cfg.globals.offload_inference = False
    with contextlib.suppress(Exception):
        cfg.model.recycle_early_stop_tolerance = -1.0
    of_batch = _openfold_multimer_batch(batch, 0, device, data_transforms)
    of_model = AlphaFold(cfg).eval()
    import_jax_weights_(of_model, str(args.npz), version=args.model_name)
    of_model = of_model.to(device)

    of_cap: dict[str, torch.Tensor] = {}
    of_model.input_embedder.register_forward_hook(lambda m, i, o: of_cap.update(ie_m=_t(o[0]), ie_z=_t(o[1])))
    of_model.template_embedder.register_forward_hook(
        lambda m, i, o: of_cap.update(tmpl=_t(o.get("template_pair_embedding") if isinstance(o, dict) else o))
    )
    of_model.extra_msa_stack.register_forward_hook(lambda m, i, o: of_cap.update(extra_z=_t(o)))
    of_model.evoformer.register_forward_hook(
        lambda m, i, o: of_cap.update(evo_m=_t(o[0]), evo_z=_t(o[1]), evo_s=_t(o[2]))
    )
    of_model.evoformer.register_forward_pre_hook(lambda m, i: of_cap.update(evo_in_m=_t(i[0]), evo_in_z=_t(i[1])))

    def _of_evo_prehook(module, args, kwargs):
        of_cap.update(evo_in_msa_mask=_t(kwargs.get("msa_mask")))

    of_model.evoformer.register_forward_pre_hook(_of_evo_prehook, with_kwargs=True)
    with torch.inference_mode():
        of_out = of_model(of_batch)
    of_cap["single"] = _t(of_out["single"])
    of_cap["pair"] = _t(of_out["pair"])

    # ---- OFoldX, recycle 0 ----
    from ofoldx.models.alphafold2.converters import deepmind
    from ofoldx.models.alphafold2.converters.deepmind_multimer import convert_deepmind_multimer_checkpoint

    params, _ = deepmind._load_official_alphafold2_params(args.checkpoint, model_name=args.model_name)
    nx = convert_deepmind_multimer_checkpoint(dict(params), model_name=args.model_name).to(device).eval()
    nx.set_backend("torch")
    dbatch = _batch_to_device(batch, device)

    nx_cap: dict[str, torch.Tensor] = {}
    nx.input_embedding.register_forward_hook(lambda m, i, o: nx_cap.update(ie_m=_t(o[0]), ie_z=_t(o[1])))
    if nx.template_embedder is not None:
        nx.template_embedder.register_forward_hook(lambda m, i, o: nx_cap.update(tmpl=_t(o)))
    nx.extra_msa_stack.register_forward_hook(
        lambda m, i, o: nx_cap.update(extra_z=_t(o[1] if isinstance(o, (tuple, list)) else o))
    )
    nx.evoformer.register_forward_hook(lambda m, i, o: nx_cap.update(evo_m=_t(o[0]), evo_z=_t(o[1])))
    nx.evoformer.register_forward_pre_hook(lambda m, i: nx_cap.update(evo_in_m=_t(i[0]), evo_in_z=_t(i[1])))

    # msa_mask fed to the evoformer (kw), captured via a wrapper pre-hook with kwargs
    def _nx_evo_prehook(module, args, kwargs):
        nx_cap.update(evo_in_msa_mask=_t(kwargs.get("msa_mask")))

    nx.evoformer.register_forward_pre_hook(_nx_evo_prehook, with_kwargs=True)
    with torch.inference_mode():
        nx_out = nx(dbatch, recycling_steps=0)
    nx_cap["single"] = _t(nx_out.single)
    nx_cap["pair"] = _t(nx_out.pair)

    stages = [
        "ie_z",
        "ie_m",
        "tmpl",
        "extra_z",
        "evo_in_m",
        "evo_in_z",
        "evo_in_msa_mask",
        "evo_z",
        "evo_m",
        "evo_s",
        "pair",
        "single",
    ]
    report = {}
    for s in stages:
        if s in of_cap and s in nx_cap and of_cap[s] is not None and nx_cap[s] is not None:
            report[s] = _cmp(nx_cap[s], of_cap[s])
        else:
            report[s] = {
                "present_of": s in of_cap and of_cap.get(s) is not None,
                "present_nx": s in nx_cap and nx_cap.get(s) is not None,
            }
    text = json.dumps(report, indent=2)
    print(text)
    if args.json:
        args.json.write_text(text + "\n")


if __name__ == "__main__":
    main()
