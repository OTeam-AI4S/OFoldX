#!/usr/bin/env python
r"""Bisect the FIRST main-Evoformer block op that diverges (OFoldX vs OpenFold, recycle 0).

Hooks block-0 sub-ops (row attention, column attention, outer-product-mean, pair stack)
and block outputs, on identical fixed features, to localize the diverging Evoformer op.
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

from scripts.parity.alphafold2.multimer_bisect import _cmp, _t  # noqa: E402
from scripts.parity.alphafold2.multimer_forward import (  # noqa: E402
    _batch_to_device,
    _deterministic,
    _install_stubs,
    _openfold_multimer_batch,
    build_multimer_features,
)


def main() -> None:
    ap = argparse.ArgumentParser()
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
    batch = build_multimer_features("GSHMKIAEN", "DPRLVWQ", seed=0)

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
    of = AlphaFold(cfg).eval()
    import_jax_weights_(of, str(args.npz), version=args.model_name)
    of = of.to(device)

    ofc: dict[str, Any] = {}
    b0 = of.evoformer.blocks[0]
    b0.msa_att_row.register_forward_hook(lambda m, i, o: ofc.update(row=_t(o)))
    b0.msa_att_col.register_forward_hook(lambda m, i, o: ofc.update(col=_t(o)))
    b0.outer_product_mean.register_forward_hook(lambda m, i, o: ofc.update(opm=_t(o)))
    b0.msa_transition.register_forward_hook(lambda m, i, o: ofc.update(mtrans=_t(o)))
    of.evoformer.blocks[0].register_forward_hook(lambda m, i, o: ofc.update(blk0_m=_t(o[0]), blk0_z=_t(o[1])))
    of.evoformer.blocks[1].register_forward_hook(lambda m, i, o: ofc.update(blk1_m=_t(o[0]), blk1_z=_t(o[1])))
    with torch.inference_mode():
        of(of_batch)

    from ofoldx.models.alphafold2.converters import deepmind
    from ofoldx.models.alphafold2.converters.deepmind_multimer import convert_deepmind_multimer_checkpoint

    params, _ = deepmind._load_official_alphafold2_params(args.checkpoint, model_name=args.model_name)
    nx = convert_deepmind_multimer_checkpoint(dict(params), model_name=args.model_name).to(device).eval()
    nx.set_backend("torch")
    dbatch = _batch_to_device(batch, device)

    nxc: dict[str, Any] = {}
    n0 = nx.evoformer.blocks[0]
    n0.msa_stack.msa_row_attention_with_pair_bias.register_forward_hook(lambda m, i, o: nxc.update(row=_t(o)))
    n0.msa_stack.msa_column_attention.register_forward_hook(lambda m, i, o: nxc.update(col=_t(o)))
    n0.outer_product_mean.register_forward_hook(lambda m, i, o: nxc.update(opm=_t(o)))
    n0.msa_stack.feed_forward.register_forward_hook(lambda m, i, o: nxc.update(mtrans=_t(o)))
    nx.evoformer.blocks[0].register_forward_hook(lambda m, i, o: nxc.update(blk0_m=_t(o[0]), blk0_z=_t(o[1])))
    nx.evoformer.blocks[1].register_forward_hook(lambda m, i, o: nxc.update(blk1_m=_t(o[0]), blk1_z=_t(o[1])))
    with torch.inference_mode():
        nx(dbatch, recycling_steps=0)

    keys = ["opm", "row", "col", "mtrans", "blk0_m", "blk0_z", "blk1_m", "blk1_z"]
    report = {}
    for k in keys:
        if ofc.get(k) is not None and nxc.get(k) is not None:
            report[k] = _cmp(nxc[k], ofc[k])
        else:
            report[k] = {"of": ofc.get(k) is not None, "nx": nxc.get(k) is not None}
    text = json.dumps(report, indent=2)
    print(text)
    if args.json:
        args.json.write_text(text + "\n")


if __name__ == "__main__":
    main()
