#!/usr/bin/env python
r"""Check the official->linear weight mapping of the AF-Multimer template pair embedder.

Loads the SAME official ``model_1_multimer_v3`` weights into both OFoldX and OpenFold and
compares each of the nine ``template_pair_embedding_i`` linears (OFoldX) against the
correspondingly-named linear in OpenFold's ``TemplatePairEmbedderMultimer``. A mismatch
localizes a wrong concat-order / official-scope assignment.
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# OFoldX index -> OpenFold TemplatePairEmbedderMultimer attribute (per OpenFold forward order)
MAPPING = {
    "template_pair_embedding_0": "dgram_linear",
    "template_pair_embedding_1": "pseudo_beta_mask_linear",
    "template_pair_embedding_2": "aatype_linear_1",
    "template_pair_embedding_3": "aatype_linear_2",
    "template_pair_embedding_4": "x_linear",
    "template_pair_embedding_5": "y_linear",
    "template_pair_embedding_6": "z_linear",
    "template_pair_embedding_7": "backbone_mask_linear",
    "template_pair_embedding_8": "query_embedding_linear",
}


def _eq(a: torch.Tensor, b: torch.Tensor) -> dict:
    a = a.detach().float().cpu()
    b = b.detach().float().cpu()
    if a.shape != b.shape:
        return {"a": list(a.shape), "b": list(b.shape), "match": False}
    return {"max_abs": round(float((a - b).abs().max()), 8), "match": bool(torch.allclose(a, b, atol=1e-6))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference-root", type=Path, default=Path("/data/users/z/reference/openfold-v2.2.0"))
    ap.add_argument(
        "--checkpoint", type=Path, default=Path("/data/users/z/weights/af2-params/alphafold_params_2022-12-06.tar")
    )
    ap.add_argument("--npz", type=Path, default=Path("/data/users/z/weights/af2-params/params_model_1_multimer_v3.npz"))
    ap.add_argument("--model-name", default="model_1_multimer_v3")
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

    # OFoldX
    from ofoldx.models.alphafold2.converters import deepmind
    from ofoldx.models.alphafold2.converters.deepmind_multimer import convert_deepmind_multimer_checkpoint

    params, _ = deepmind._load_official_alphafold2_params(args.checkpoint, model_name=args.model_name)
    nx = convert_deepmind_multimer_checkpoint(dict(params), model_name=args.model_name).eval()
    nx_pe = nx.template_embedder.single_template_embedding

    # OpenFold
    from openfold.config import model_config
    from openfold.model.model import AlphaFold
    from openfold.utils.import_weights import import_jax_weights_

    cfg = model_config(args.model_name)
    cfg.globals.use_lma = False
    cfg.globals.use_flash = False
    cfg.globals.use_deepspeed_evo_attention = False
    of = AlphaFold(cfg).eval()
    import_jax_weights_(of, str(args.npz), version=args.model_name)
    of_pe = of.template_embedder.template_pair_embedder

    report = {}
    for nx_name, of_name in MAPPING.items():
        nxl = getattr(nx_pe, nx_name)
        ofl = getattr(of_pe, of_name)
        report[f"{nx_name} <-> {of_name}"] = {
            "weight": _eq(nxl.weight, ofl.weight),
            "bias": _eq(nxl.bias, ofl.bias) if (nxl.bias is not None and ofl.bias is not None) else "n/a",
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
