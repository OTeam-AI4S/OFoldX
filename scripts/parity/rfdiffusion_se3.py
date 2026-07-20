#!/usr/bin/env python
"""RFDiffusion SE(3) block parity harness.

This script compares the OFoldX-native SE(3) block against the official
RosettaCommons/RFdiffusion implementation when a reference checkout with its
official DGL/e3nn SE3Transformer environment is available. Without
``--reference-root`` it emits the OFoldX output signature used by local
golden-regression tests.

This component diagnostic is not public reverse-sampler or DesignResult signoff.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.parity.paths import weight_path  # noqa: E402

DEFAULT_CHECKPOINT = weight_path("rfdiffusion-weights/Base_ckpt.pt")
TENSOR_NAMES = ("R", "T", "state", "alpha")
REFERENCE_MODEL_KEYS = (
    "n_extra_block",
    "n_main_block",
    "n_ref_block",
    "d_msa",
    "d_msa_full",
    "d_pair",
    "d_templ",
    "n_head_msa",
    "n_head_pair",
    "n_head_templ",
    "d_hidden",
    "d_hidden_templ",
    "p_drop",
    "use_motif_timestep",
    "freeze_track_motif",
    "SE3_param_full",
    "SE3_param_topk",
    "input_seq_onehot",
)


def main() -> None:
    args = _parser().parse_args()
    device = torch.device(args.device)
    checkpoint = _load_checkpoint(args.checkpoint)

    # Cross-environment split: --dump-reference runs ONLY the official reference
    # (in its DGL/e3nn SE3nv env) and serializes its outputs plus the exact golden
    # inputs; --load-reference runs ONLY the OFoldX-native block (in zyc) against
    # those dumped tensors, at the identical tolerance used by the in-process path.
    if args.dump_reference is not None:
        _mc = checkpoint["config_dict"]["model"]
        inputs = _golden_inputs(_mc["d_msa_full"], _mc["d_pair"], 8, device=device, seed=args.seed)
        reference_outputs = _run_reference(
            args.reference_root or (Path(__file__).resolve().parents[2] / "reference"),
            checkpoint,
            device=device,
            seed=args.seed,
            inputs=inputs,
        )
        args.dump_reference.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "outputs": [tensor.cpu() for tensor in reference_outputs],
                "inputs": [tensor.cpu() for tensor in inputs],
                "seed": args.seed,
                "checkpoint": str(args.checkpoint),
            },
            args.dump_reference,
        )
        print(f"wrote reference dump {args.dump_reference}")
        return

    inputs = None
    dumped_reference: tuple[Tensor, ...] | None = None
    if args.load_reference is not None:
        blob = torch.load(args.load_reference, map_location="cpu", weights_only=True)
        if blob.get("checkpoint") != str(args.checkpoint) or blob.get("seed") != args.seed:
            raise SystemExit("reference dump checkpoint or seed does not match this invocation")
        inputs = tuple(blob["inputs"])
        dumped_reference = tuple(blob["outputs"])

    native_outputs = _run_ofoldx(checkpoint, device=device, seed=args.seed, backend=args.backend, inputs=inputs)
    report: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "seed": args.seed,
        "device": str(device),
        "native": _summarize_outputs(native_outputs),
    }

    if dumped_reference is not None:
        report["reference"] = _summarize_outputs(dumped_reference)
        report["diff"] = _diff_outputs(native_outputs, dumped_reference)
    elif args.reference_root is not None:
        reference_outputs = _run_reference(
            args.reference_root,
            checkpoint,
            device=device,
            seed=args.seed,
        )
        report["reference"] = _summarize_outputs(reference_outputs)
        report["diff"] = _diff_outputs(native_outputs, reference_outputs)

    text = json.dumps(report, indent=2, sort_keys=True)
    if args.json is None:
        print(text)
    else:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n")
        print(f"wrote {args.json}")

    if "diff" in report and report["diff"]["max_abs"] > args.atol:
        raise SystemExit(f"parity failed: max_abs={report['diff']['max_abs']:.6g} exceeds atol={args.atol:.6g}")


def _load_checkpoint(path: Path) -> dict[str, Any]:
    """Load the official checkpoint under both torch 2.x (native) and torch 1.9 (reference)."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _run_ofoldx(
    checkpoint: dict[str, Any],
    *,
    device: torch.device,
    seed: int,
    backend: str,
    inputs: tuple[Tensor, ...] | None = None,
) -> tuple[Tensor, ...]:
    sys.path.insert(0, str(REPO))
    from ofoldx.models.rfdiffusion import RFdiffusionModel  # noqa: PLC0415
    from ofoldx.models.rfdiffusion.convert import (  # noqa: PLC0415
        convert_rfdiffusion_state_dict,
        rfdiffusion_config_from_checkpoint,
    )

    config = rfdiffusion_config_from_checkpoint(checkpoint)
    model = RFdiffusionModel(config).set_backend(backend).to(device).eval()
    state = convert_rfdiffusion_state_dict(checkpoint["model_state_dict"])
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"OFoldX strict load failed: {result}")

    block = model.network.simulator.extra_block[0].str2str.eval()
    if inputs is None:
        _mc = checkpoint["config_dict"]["model"]
        inputs = _golden_inputs(_mc["d_msa_full"], _mc["d_pair"], 8, device=device, seed=seed)
    else:
        inputs = tuple(value.to(device) for value in inputs)
    with torch.no_grad():
        return tuple(output.detach().cpu() for output in block(*inputs, top_k=0))


def _run_reference(
    reference_root: Path,
    checkpoint: dict[str, Any],
    *,
    device: torch.device,
    seed: int,
    inputs: tuple[Tensor, ...] | None = None,
) -> tuple[Tensor, ...]:
    reference_root = reference_root.resolve()
    se3_root = reference_root / "env" / "SE3Transformer"
    sys.path.insert(0, str(se3_root))
    sys.path.insert(0, str(reference_root))
    try:
        importlib.import_module("dgl")
        module = importlib.import_module("rfdiffusion.RoseTTAFoldModel")
    except Exception as error:
        raise RuntimeError(
            "Official RFDiffusion reference import failed. The reference path needs "
            "its official DGL/e3nn SE3Transformer environment."
        ) from error

    source_model_config = checkpoint["config_dict"]["model"]
    preprocess = checkpoint["config_dict"].get("preprocess", {})
    diffuser = checkpoint["config_dict"].get("diffuser", {})
    # ``input_seq_onehot`` is not serialized in released checkpoints; fall back to the
    # reference RoseTTAFoldModule constructor default (False). It only affects the MSA
    # embedders, never the str2str SE(3) block under test here.
    model_config = {key: source_model_config[key] for key in REFERENCE_MODEL_KEYS if key in source_model_config}
    model_config["d_t1d"] = preprocess["d_t1d"]
    model_config["d_t2d"] = preprocess["d_t2d"]
    model_config["T"] = diffuser["T"]
    ref_model = module.RoseTTAFoldModule(**model_config).to(device).eval()
    result = ref_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"reference strict load failed: {result}")

    block = ref_model.simulator.extra_block[0].str2str.eval()
    if inputs is None:
        inputs = _golden_inputs(model_config["d_msa_full"], model_config["d_pair"], 8, device=device, seed=seed)
    else:
        inputs = tuple(value.to(device) for value in inputs)
    with torch.no_grad():
        return tuple(output.detach().cpu() for output in block(*inputs, top_k=0))


def _golden_inputs(
    d_msa_full: int,
    d_pair: int,
    d_state: int,
    *,
    device: torch.device,
    seed: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    torch.manual_seed(seed)
    batch_size = 1
    length = 4
    msa_depth = 1
    msa = torch.randn(batch_size, msa_depth, length, d_msa_full, device=device)
    pair = torch.randn(batch_size, length, length, d_pair, device=device)
    rotations = torch.eye(3, device=device).expand(batch_size, length, 3, 3).clone()
    translations = torch.randn(batch_size, length, 3, device=device) * 0.1
    xyz = torch.randn(batch_size, length, 3, 3, device=device)
    state = torch.randn(batch_size, length, d_state, device=device)
    residue_index = torch.arange(length, device=device).reshape(1, length)
    motif_mask = torch.tensor([False, True, False, False], device=device)
    return msa, pair, rotations, translations, xyz, state, residue_index, motif_mask


def _summarize_outputs(outputs: tuple[Tensor, ...]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for name, tensor in zip(TENSOR_NAMES, outputs, strict=True):
        summary[name] = {
            "shape": list(tensor.shape),
            "sum": float(tensor.sum().item()),
            "mean": float(tensor.mean().item()),
            "std": float(tensor.std(unbiased=False).item()),
            "max_abs": float(tensor.abs().max().item()),
            "values": tensor.tolist(),
        }
    return summary


def _diff_outputs(native_outputs: tuple[Tensor, ...], reference_outputs: tuple[Tensor, ...]) -> dict[str, Any]:
    per_tensor = {}
    max_abs = 0.0
    for name, native, reference in zip(TENSOR_NAMES, native_outputs, reference_outputs, strict=True):
        diff = (native - reference).abs()
        tensor_max = float(diff.max().item())
        max_abs = max(max_abs, tensor_max)
        per_tensor[name] = {
            "max_abs": tensor_max,
            "mean_abs": float(diff.mean().item()),
        }
    return {"max_abs": max_abs, "tensors": per_tensor}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--reference-root", type=Path, default=None)
    parser.add_argument(
        "--dump-reference",
        type=Path,
        default=None,
        help="Run only the official reference (SE3nv env) and serialize outputs+inputs to this .pt",
    )
    parser.add_argument(
        "--load-reference",
        type=Path,
        default=None,
        help="Run only the OFoldX-native block (zyc) and compare against this dumped reference .pt",
    )
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260702)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--backend", choices=("torch", "oops"), default="torch")
    return parser


if __name__ == "__main__":
    main()
