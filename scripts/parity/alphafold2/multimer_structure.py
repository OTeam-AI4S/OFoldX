#!/usr/bin/env python
r"""Isolation parity for the AF-Multimer structure-module primitives vs OpenFold (>=2.0).

The genuinely new AF-Multimer structure-module math -- the invariant point attention variant
(:class:`InvariantPointAttentionMultimer`: separate bias-free q/k/v scalar projections,
global-frame :class:`PointProjection` points, the multimer logit normalization and key-axis
softmax) and the :class:`QuatRigid` backbone update composed onto a rotation-matrix rigid --
is validated here directly against a multimer-capable OpenFold reference. OpenFold's weights
are copied into the native modules and both are run on identical random ``(s, z, rigid, mask)``;
a bit-exact match confirms the ported forward math.

This complements ``scripts/parity/alphafold2/forward.py`` (the monomer deep-forward golden) and
the strict weight-load probe ``multimer_probe.py``. Run under a GPU node with a multimer-capable
OpenFold checkout (``--reference-root``, an OpenFold >= v2.0.0 worktree); the monomer reference
pinned for ``forward.py`` (v1.0.1) predates the multimer modules.
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


def _stub_optional_modules() -> None:
    # OpenFold v2.x imports mmCIF writers eagerly; they are irrelevant to the numeric path.
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


def _rot_to_matrix(rot) -> torch.Tensor:
    return torch.stack(
        [
            torch.stack([rot.xx, rot.xy, rot.xz], dim=-1),
            torch.stack([rot.yx, rot.yy, rot.yz], dim=-1),
            torch.stack([rot.zx, rot.zy, rot.zz], dim=-1),
        ],
        dim=-2,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference-root", type=Path, required=True, help="OpenFold >= v2.0.0 checkout root")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-tokens", type=int, default=7)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    _stub_optional_modules()
    if str(args.reference_root) not in sys.path:
        sys.path.insert(0, str(args.reference_root))

    from openfold.model.structure_module import InvariantPointAttentionMultimer as RefIpa
    from openfold.utils.geometry.quat_rigid import QuatRigid as RefQuatRigid
    from openfold.utils.geometry.rigid_matrix_vector import Rigid3Array
    from openfold.utils.geometry.rotation_matrix import Rot3Array
    from openfold.utils.geometry.vector import Vec3Array

    from ofoldx.models.alphafold2.model import InvariantPointAttentionMultimer as NativeIpa
    from ofoldx.models.alphafold2.model import QuatRigid as NativeQuatRigid
    from ofoldx.models.alphafold2.model import Rigid as NativeRigid

    device = torch.device(args.device)
    c_s, c_z, c_hidden, num_head, p_qk, p_v = 384, 128, 16, 12, 4, 8
    n = args.num_tokens

    single = torch.randn(1, n, c_s, dtype=torch.double, device=device)
    pair = torch.randn(1, n, n, c_z, dtype=torch.double, device=device)
    mask = torch.ones(1, n, dtype=torch.double, device=device)

    quat = torch.randn(1, n, 4, dtype=torch.double, device=device)
    quat = quat / quat.norm(dim=-1, keepdim=True)
    qw, qx, qy, qz = quat.unbind(-1)
    ref_rot = Rot3Array.from_quaternion(qw, qx, qy, qz, normalize=True)
    trans = torch.randn(1, n, 3, dtype=torch.double, device=device)
    ref_rigid = Rigid3Array(ref_rot, Vec3Array(trans[..., 0], trans[..., 1], trans[..., 2]))
    native_rigid = NativeRigid(rotation=_rot_to_matrix(ref_rot), translation=trans)

    # --- Invariant point attention (multimer) ---
    ref_ipa = RefIpa(c_s=c_s, c_z=c_z, c_hidden=c_hidden, no_heads=num_head, no_qk_points=p_qk, no_v_points=p_v)
    ref_ipa = ref_ipa.to(device).double().eval()
    native_ipa = (
        NativeIpa(
            c_s=c_s,
            c_z=c_z,
            num_head=num_head,
            num_scalar_qk=c_hidden,
            num_scalar_v=c_hidden,
            num_point_qk=p_qk,
            num_point_v=p_v,
        )
        .to(device)
        .double()
        .eval()
    )
    with torch.no_grad():
        native_ipa.q_scalar_projection.weight.copy_(ref_ipa.linear_q.weight)
        native_ipa.k_scalar_projection.weight.copy_(ref_ipa.linear_k.weight)
        native_ipa.v_scalar_projection.weight.copy_(ref_ipa.linear_v.weight)
        native_ipa.q_point_projection.point_projection.weight.copy_(ref_ipa.linear_q_points.linear.weight)
        native_ipa.q_point_projection.point_projection.bias.copy_(ref_ipa.linear_q_points.linear.bias)
        native_ipa.k_point_projection.point_projection.weight.copy_(ref_ipa.linear_k_points.linear.weight)
        native_ipa.k_point_projection.point_projection.bias.copy_(ref_ipa.linear_k_points.linear.bias)
        native_ipa.v_point_projection.point_projection.weight.copy_(ref_ipa.linear_v_points.linear.weight)
        native_ipa.v_point_projection.point_projection.bias.copy_(ref_ipa.linear_v_points.linear.bias)
        native_ipa.attention_2d.weight.copy_(ref_ipa.linear_b.weight)
        native_ipa.attention_2d.bias.copy_(ref_ipa.linear_b.bias)
        native_ipa.trainable_point_weights.copy_(ref_ipa.head_weights)
        native_ipa.output_projection.weight.copy_(ref_ipa.linear_out.weight)
        native_ipa.output_projection.bias.copy_(ref_ipa.linear_out.bias)
    with torch.no_grad():
        ref_ipa_out = ref_ipa(single, pair, ref_rigid, mask)
        native_ipa_out = native_ipa(single=single, pair=pair, mask=mask, rigid=native_rigid)
    ipa_max_abs = float((ref_ipa_out - native_ipa_out).abs().max())

    # --- QuatRigid backbone update + rigid composition ---
    ref_qr = RefQuatRigid(c_s, full_quat=False).to(device).double().eval()
    native_qr = NativeQuatRigid(c_s).to(device).double().eval()
    with torch.no_grad():
        native_qr.rigid.weight.copy_(ref_qr.linear.weight)
        native_qr.rigid.bias.copy_(ref_qr.linear.bias)
        ref_comp = ref_rigid @ ref_qr(single)
        native_comp = native_rigid.compose(native_qr(single))
    ref_comp_rot = _rot_to_matrix(ref_comp.rotation)
    ref_comp_trans = torch.stack([ref_comp.translation.x, ref_comp.translation.y, ref_comp.translation.z], dim=-1)
    qr_rot_max_abs = float((ref_comp_rot - native_comp.rotation).abs().max())
    qr_trans_max_abs = float((ref_comp_trans - native_comp.translation).abs().max())

    report = {
        "reference": "openfold-multimer",
        "reference_root": str(args.reference_root),
        "device": args.device,
        "num_tokens": n,
        "atol": args.atol,
        "invariant_point_attention_multimer": {"max_absolute_error": ipa_max_abs, "passed": ipa_max_abs <= args.atol},
        "quat_rigid_compose": {
            "max_absolute_error_rotation": qr_rot_max_abs,
            "max_absolute_error_translation": qr_trans_max_abs,
            "passed": max(qr_rot_max_abs, qr_trans_max_abs) <= args.atol,
        },
    }
    report["passed"] = bool(
        report["invariant_point_attention_multimer"]["passed"] and report["quat_rigid_compose"]["passed"]
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
