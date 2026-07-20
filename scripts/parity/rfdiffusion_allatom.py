#!/usr/bin/env python
r"""Compare native RFdiffusion atom14 reconstruction with the official reference.

This component diagnostic is not public reverse-sampler or DesignResult signoff.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[2]


def main() -> None:
    args = _parser().parse_args()
    device = torch.device(args.device)

    # Cross-environment split: --dump-reference runs ONLY the official reference
    # (DGL/e3nn SE3nv env) and serializes its atom14 frames/xyz plus the exact
    # golden inputs; --load-reference runs ONLY the OFoldX-native reconstruction
    # (zyc) against those dumped tensors, at the identical tolerance.
    if args.dump_reference is not None:
        if args.reference_root is None:
            raise SystemExit("--dump-reference requires --reference-root")
        inputs = _inputs(device=device, seed=args.seed)
        ref_frames, ref_xyz = _run_reference(args.reference_root, device=device, seed=args.seed, inputs=inputs)
        args.dump_reference.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "frames": ref_frames.cpu(),
                "xyz": ref_xyz.cpu(),
                "inputs": [tensor.cpu() for tensor in inputs],
                "seed": args.seed,
            },
            args.dump_reference,
        )
        print(f"wrote reference dump {args.dump_reference}")
        return

    if args.load_reference is not None:
        blob = torch.load(args.load_reference, map_location="cpu", weights_only=True)
        if blob.get("seed") != args.seed:
            raise SystemExit("reference dump seed does not match this invocation")
        native_frames, native_xyz = _run_native(device=device, seed=args.seed, inputs=tuple(blob["inputs"]))
        ref_frames, ref_xyz = blob["frames"], blob["xyz"]
    else:
        if args.reference_root is None:
            raise SystemExit("in-process parity requires --reference-root")
        native_frames, native_xyz = _run_native(device=device, seed=args.seed)
        ref_frames, ref_xyz = _run_reference(args.reference_root, device=device, seed=args.seed)
    frame_diff = torch.nan_to_num((native_frames - ref_frames).abs(), nan=0.0)
    xyz_diff = torch.nan_to_num((native_xyz - ref_xyz).abs(), nan=0.0)
    report: dict[str, Any] = {
        "device": str(device),
        "seed": args.seed,
        "reference_root": str(args.reference_root),
        "frames_max_abs": float(frame_diff.max().item()),
        "xyz_max_abs": float(xyz_diff.max().item()),
        "nan_mask_equal": bool(torch.equal(torch.isnan(native_xyz), torch.isnan(ref_xyz))),
        "shape": list(native_xyz.shape),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["frames_max_abs"] > args.atol or report["xyz_max_abs"] > args.atol or not report["nan_mask_equal"]:
        raise SystemExit("RFdiffusion all-atom parity failed")


def _run_native(
    *, device: torch.device, seed: int, inputs: tuple[torch.Tensor, ...] | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    sys.path.insert(0, str(REPO))
    from ofoldx.models.rfdiffusion import ComputeAllAtomCoords  # noqa: PLC0415

    if inputs is None:
        seq, xyz, alphas = _inputs(device=device, seed=seed)
    else:
        seq, xyz, alphas = (tensor.to(device) for tensor in inputs)
    with torch.no_grad():
        return tuple(value.cpu() for value in ComputeAllAtomCoords().to(device).eval()(seq, xyz, alphas))


def _run_reference(
    reference_root: Path,
    *,
    device: torch.device,
    seed: int,
    inputs: tuple[torch.Tensor, ...] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    sys.path.insert(0, str(reference_root.resolve()))
    from rfdiffusion.util_module import ComputeAllAtomCoords  # noqa: PLC0415

    if inputs is None:
        seq, xyz, alphas = _inputs(device=device, seed=seed)
    else:
        seq, xyz, alphas = (tensor.to(device) for tensor in inputs)
    with torch.no_grad():
        return tuple(value.cpu() for value in ComputeAllAtomCoords().to(device).eval()(seq, xyz, alphas, use_H=False))


def _inputs(*, device: torch.device, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    seq = torch.tensor(
        [[0, 1, 7, 12, 19, 20], [3, 5, 8, 14, 17, 21]],
        dtype=torch.long,
        device=device,
    )
    batch_size, length = seq.shape
    xyz = torch.randn(batch_size, length, 3, 3, device=device)
    xyz[..., 1, :] = torch.randn(batch_size, length, 3, device=device)
    xyz[..., 0, :] = xyz[..., 1, :] + torch.tensor([-0.5, 1.3, 0.1], device=device)
    xyz[..., 2, :] = xyz[..., 1, :] + torch.tensor([1.5, 0.1, 0.2], device=device)
    alphas = torch.randn(batch_size, length, 10, 2, device=device)
    alphas = alphas / torch.linalg.norm(alphas, dim=-1, keepdim=True).clamp_min(1e-8)
    return seq, xyz, alphas


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=None,
        help="Official RFdiffusion checkout (required for in-process or --dump-reference)",
    )
    parser.add_argument(
        "--dump-reference",
        type=Path,
        default=None,
        help="Run only the official reference (SE3nv env) and serialize frames/xyz+inputs to this .pt",
    )
    parser.add_argument(
        "--load-reference",
        type=Path,
        default=None,
        help="Run only the OFoldX-native reconstruction (zyc) and compare against this dumped .pt",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--atol", type=float, default=1e-4)
    return parser


if __name__ == "__main__":
    main()
