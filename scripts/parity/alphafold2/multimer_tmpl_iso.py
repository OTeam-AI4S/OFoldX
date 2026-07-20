#!/usr/bin/env python
r"""Isolate the AF-Multimer template embedder: OFoldX vs OpenFold, identical inputs.

Both template embedders are taken from models loaded from the SAME official weights and
driven with an identical random query pair embedding and identical template geometry
(atom coordinates -> OFoldX processor frame/pseudo-beta, validated bit-exact against
OpenFold's make_backbone_affine). Removes every upstream confound (input/recycling
embedders). Hooks capture the pre-pair-stack ``act`` to localize divergence.
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _cmp(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    a = a.detach().float().cpu().reshape(-1)
    b = b.detach().float().cpu().reshape(-1)
    if a.shape != b.shape:
        return {"a": list(a.shape), "b": list(b.shape), "shape_ok": False}
    d = a - b
    return {
        "n": int(a.numel()),
        "rel_l2": round(float(d.norm() / (b.norm() + 1e-12)), 6),
        "cosine": round(float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-12)), 6),
        "max_abs": round(float(d.abs().max()), 6),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference-root", type=Path, default=Path("/data/users/z/reference/openfold-v2.2.0"))
    ap.add_argument(
        "--checkpoint", type=Path, default=Path("/data/users/z/weights/af2-params/alphafold_params_2022-12-06.tar")
    )
    ap.add_argument("--npz", type=Path, default=Path("/data/users/z/weights/af2-params/params_model_1_multimer_v3.npz"))
    ap.add_argument("--model-name", default="model_1_multimer_v3")
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from scripts.parity.alphafold2.multimer_forward import _install_stubs

    _install_stubs()
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
    if str(args.reference_root) not in sys.path:
        sys.path.insert(0, str(args.reference_root))

    torch.manual_seed(3)
    device = torch.device(args.device)
    n = args.n

    from ofoldx.models.alphafold2.converters import deepmind
    from ofoldx.models.alphafold2.converters.deepmind_multimer import convert_deepmind_multimer_checkpoint
    from ofoldx.models.alphafold2.model import AlphaFold2MultimerInput
    from ofoldx.models.alphafold2.processor import _template_backbone_frame, _template_pseudo_beta

    params, _ = deepmind._load_official_alphafold2_params(args.checkpoint, model_name=args.model_name)
    nx = convert_deepmind_multimer_checkpoint(dict(params), model_name=args.model_name).to(device).eval()
    nx.set_backend("torch")

    from openfold.config import model_config
    from openfold.model.model import AlphaFold
    from openfold.utils import geometry
    from openfold.utils.import_weights import import_jax_weights_

    cfg = model_config(args.model_name)
    cfg.globals.use_lma = False
    cfg.globals.use_flash = False
    cfg.globals.use_deepspeed_evo_attention = False
    cfg.globals.chunk_size = None
    of = AlphaFold(cfg).eval().to(device)
    import_jax_weights_(of, str(args.npz), version=args.model_name)

    c_z = 128
    z = torch.randn(1, n, n, c_z, device=device)
    aatype = torch.randint(0, 20, (1, n), dtype=torch.long, device=device)
    positions = torch.randn(1, n, 37, 3, device=device) * 8.0
    masks = torch.ones(1, n, 37, device=device)
    chain_index = torch.tensor([[1] * (n // 2) + [2] * (n - n // 2)], device=device)
    seq_mask = torch.ones(1, n, device=device)
    pair_mask = seq_mask[..., None] * seq_mask[..., None, :]

    pb, pbm = _template_pseudo_beta(aatype, positions, masks)
    frame, fmask = _template_backbone_frame(positions, masks)

    # OFoldX batch (only the fields the template embedder reads)
    nx_batch = AlphaFold2MultimerInput(
        aatype=aatype,
        target_feat=torch.zeros(1, n, 21, device=device),
        residue_index=torch.arange(n, device=device).unsqueeze(0),
        token_mask=seq_mask,
        chain_index=chain_index,
        entity_index=chain_index,
        symmetry_index=chain_index,
        msa_feat=torch.zeros(1, 1, n, 49, device=device),
        msa_mask=torch.ones(1, 1, n, device=device),
        extra_msa_feat=torch.zeros(1, 1, n, 25, device=device),
        extra_msa_mask=torch.zeros(1, 1, n, device=device),
        template_mask=torch.ones(1, 1, device=device),
        template_aatype=aatype.unsqueeze(1),
        template_all_atom_positions=positions.unsqueeze(1),
        template_all_atom_mask=masks.unsqueeze(1),
        template_pseudo_beta=pb.unsqueeze(1),
        template_pseudo_beta_mask=pbm.unsqueeze(1),
        template_backbone_frame=frame.unsqueeze(1),
        template_backbone_frame_mask=fmask.unsqueeze(1),
        layout=None,
    )

    # Hooks: pre-pair-stack act
    caps: dict[str, torch.Tensor] = {}
    nx.template_embedder.single_template_embedding.template_embedding_iteration.register_forward_pre_hook(
        lambda m, inp: caps.update(nx_prestack=inp[0].detach().float().cpu())
    )
    of.template_embedder.template_pair_stack.register_forward_pre_hook(
        lambda m, inp: caps.update(of_prestack=inp[0].detach().float().cpu())
    )
    # Layer-normed pair-stack output: OFoldX applies output_layer_norm after the stack;
    # OpenFold's TemplatePairStack applies its final layer_norm internally.
    nx.template_embedder.single_template_embedding.output_layer_norm.register_forward_hook(
        lambda m, i, o: caps.update(nx_poststack=o.detach().float().cpu())
    )
    of.template_embedder.template_pair_stack.register_forward_hook(
        lambda m, i, o: caps.update(of_poststack=(o[0] if isinstance(o, (tuple, list)) else o).detach().float().cpu())
    )

    with torch.inference_mode():
        nx_out = nx.template_embedder(z, nx_batch, pair_mask=pair_mask)

    # OpenFold template embedder
    multichain_mask_2d = chain_index[..., None] == chain_index[..., None, :]
    of_batch = {
        "template_aatype": aatype.unsqueeze(1),
        "template_all_atom_positions": positions.unsqueeze(1),
        "template_all_atom_mask": masks.unsqueeze(1),
        "template_pseudo_beta": pb.unsqueeze(1),
        "template_pseudo_beta_mask": pbm.unsqueeze(1),
    }
    with torch.inference_mode():
        of_out = of.template_embedder(
            of_batch,
            z,
            pair_mask,
            1,
            chunk_size=None,
            multichain_mask_2d=multichain_mask_2d,
        )["template_pair_embedding"]

    # --- template SINGLE (MSA-row) embedding ---
    with torch.inference_mode():
        nx_single, nx_single_mask = nx.template_msa_embedding(nx_batch)  # [1,1,N,c_m]
    of_atom_pos = geometry.Vec3Array.from_array(positions.to(torch.float32))  # [1,N]
    aatype_one_hot22 = torch.nn.functional.one_hot(aatype.clamp(max=21), 22).float()
    of_single_batch = {
        "template_all_atom_positions": positions,
        "template_all_atom_mask": masks,
        "template_aatype": aatype,
    }
    with torch.inference_mode():
        of_single = of.template_embedder.template_single_embedder(of_single_batch, of_atom_pos, aatype_one_hot22)
    of_single_act = of_single["template_single_embedding"]

    report = {
        "template_embedder_output": _cmp(nx_out, of_out),
        "template_single_embedding": _cmp(nx_single[:, 0], of_single_act),
    }
    if "nx_prestack" in caps and "of_prestack" in caps:
        report["pre_pair_stack_act"] = _cmp(caps["nx_prestack"], caps["of_prestack"])
    if "nx_poststack" in caps and "of_poststack" in caps:
        # OFoldX post is per-template [1,N,N,C]; OpenFold post is [1,S_t=1,N,N,C].
        report["post_pair_stack_ln"] = _cmp(caps["nx_poststack"], caps["of_poststack"])
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
