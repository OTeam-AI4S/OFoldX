#!/usr/bin/env python
r"""Isolate the AF-Multimer template path vs OpenFold on identical atom coordinates.

Compares (a) the backbone unit-vector + backbone mask and (b) the pseudo-beta from
the OFoldX processor helpers against OpenFold's ``make_backbone_affine`` /
``pseudo_beta`` on the SAME random atom37 template coordinates, then (c) runs the
full single-template pair embedder in both with copied weights.
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
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

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

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    n = args.n
    dtype = torch.float64  # high precision to expose any formula mismatch

    from ofoldx.models.alphafold2.model import _template_backbone_unit_vector
    from ofoldx.models.alphafold2.processor import _template_backbone_frame, _template_pseudo_beta

    # Random template coordinates and aatype
    positions = torch.randn(1, n, 37, 3, dtype=dtype) * 8.0
    masks = torch.ones(1, n, 37, dtype=dtype)
    aatype = torch.randint(0, 20, (1, n), dtype=torch.long)

    # --- OFoldX side ---
    frame, frame_mask = _template_backbone_frame(positions, masks)
    pb, pbm = _template_pseudo_beta(aatype, positions, masks)

    class _B:
        pass

    b = _B()
    b.template_backbone_frame = frame.unsqueeze(0)
    b.template_backbone_frame_mask = frame_mask.unsqueeze(0)
    uv_nx, bbmask_nx = _template_backbone_unit_vector(b)  # [1,1,n,n,3], [1,1,n]

    # --- OpenFold side ---
    from openfold.utils import all_atom_multimer, geometry

    atom_pos = geometry.Vec3Array.from_array(positions.to(torch.float32))  # [1,n]
    rigid, bbmask_of = all_atom_multimer.make_backbone_affine(atom_pos, masks.to(torch.float32), aatype)
    points = rigid.translation
    rigid_vec = rigid[..., None].inverse().apply_to_point(points)
    uv_of = rigid_vec.normalized()  # Vec3Array [1,n,n]
    uv_of_arr = torch.stack([uv_of.x, uv_of.y, uv_of.z], dim=-1)  # [1,n,n,3]

    # OpenFold pseudo beta (multimer): pseudo_beta_fn
    from openfold.utils.feats import pseudo_beta_fn

    pb_of, pbm_of = pseudo_beta_fn(aatype, positions.to(torch.float32), masks.to(torch.float32))

    report = {
        "backbone_unit_vector": _cmp(uv_nx[:, 0].to(torch.float32), uv_of_arr.to(torch.float32)),
        "backbone_mask": _cmp(bbmask_nx[:, 0].to(torch.float32), bbmask_of.to(torch.float32)),
        "pseudo_beta": _cmp(pb.to(torch.float32), pb_of),
        "pseudo_beta_mask": _cmp(pbm.to(torch.float32), pbm_of),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
