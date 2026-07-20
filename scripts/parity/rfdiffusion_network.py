#!/usr/bin/env python
r"""Compare RFdiffusion's released training heads with the official network.

Run ``reference`` with the official RFdiffusion environment, then run ``native``
with the OFoldX environment against the captured tensors.  Production weights are
local inputs and are never downloaded by this script.

This is a denoiser diagnostic, not public sampler signoff. Sampler acceptance must
compare the raw final pX0 and the canonical motif-repaired DesignResult separately.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_NAMES = (
    "distance",
    "omega",
    "theta",
    "phi",
    "sequence",
    "resolved",
    "backbone_trajectory",
    "torsion_trajectory",
    "allatom",
    "plddt",
)


def inputs(config: dict[str, object], *, length: int = 4) -> tuple[torch.Tensor, ...]:
    preprocess = config["preprocess"]
    tokens = torch.tensor([0, 21, 4, 7], dtype=torch.long)[:length]
    sequence = torch.nn.functional.one_hot(tokens, num_classes=22).float()[None]
    msa = torch.zeros((1, 1, length, 48))
    msa[..., :22] = sequence[:, None]
    msa[..., 22:44] = sequence[:, None]
    msa[:, :, 0, 46] = 1
    msa[:, :, -1, 47] = 1
    msa_full = torch.zeros((1, 1, length, 25))
    msa_full[..., :22] = sequence[:, None]
    msa_full[:, :, 0, 23] = 1
    msa_full[:, :, -1, 24] = 1

    residue = torch.arange(length, dtype=torch.float32)
    calpha = torch.stack((3.8 * residue, 0.2 * (residue.remainder(2) * 2 - 1), 0.1 * residue), dim=-1)
    coords = torch.zeros((1, length, 14, 3))
    coords[0, :, 0] = calpha + torch.tensor([-1.2, 0.7, -0.1])
    coords[0, :, 1] = calpha
    coords[0, :, 2] = calpha + torch.tensor([1.3, 0.5, 0.2])
    coords[0, :, 3:] = calpha[:, None] + torch.arange(11, dtype=torch.float32)[None, :, None] / 20
    residue_index = torch.arange(length, dtype=torch.long)[None]

    template_1d = torch.zeros((1, 1, length, preprocess["d_t1d"]))
    template_1d[..., :21] = sequence[:, None, :, :21]
    template_1d[..., 21] = torch.tensor([0.1, 1.0, 0.4, 0.7])[:length]
    template_2d = torch.arange(length * length * preprocess["d_t2d"], dtype=torch.float32).reshape(
        1, 1, length, length, preprocess["d_t2d"]
    )
    template_2d = template_2d.remainder(17) / 17
    template_coords = torch.zeros((1, 1, length, 27, 3))
    template_coords[:, :, :, :14] = coords[:, None]
    template_coords[:, :, :, 14:] = calpha[None, None, :, None]
    torsion = torch.tensor([1.0, 0.0, 1.0]).repeat(1, 1, length, 10).reshape(1, 1, length, 30)
    motif_mask = torch.tensor([False, True, False, True])[:length]
    return (
        msa,
        msa_full,
        sequence,
        coords,
        residue_index,
        template_1d,
        template_2d,
        template_coords,
        torsion,
        motif_mask,
    )


def reference(checkpoint_path: Path, reference_root: Path) -> dict[str, torch.Tensor]:
    if not torch.cuda.is_available():
        torch.cuda.nvtx.range = lambda *_args, **_kwargs: contextlib.nullcontext()
    sys.path.insert(0, str(reference_root / "env" / "SE3Transformer"))
    sys.path.insert(0, str(reference_root))
    from rfdiffusion.RoseTTAFoldModel import RoseTTAFoldModule
    from rfdiffusion.util_module import ComputeAllAtomCoords

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    config = checkpoint["config_dict"]
    source = config["model"]
    names = (
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
    )
    kwargs = {name: source[name] for name in names}
    kwargs.update(
        d_t1d=config["preprocess"]["d_t1d"],
        d_t2d=config["preprocess"]["d_t2d"],
        T=config["diffuser"]["T"],
    )
    model = RoseTTAFoldModule(**kwargs).eval()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    msa, msa_full, sequence, coords, index, t1d, t2d, xyz_t, alpha_t, motif = inputs(config)
    with torch.no_grad():
        distance, sequence_logits, resolved, trajectory, torsion, plddt = model(
            msa,
            msa_full,
            sequence,
            coords,
            index,
            torch.tensor(25),
            t1d=t1d,
            t2d=t2d,
            xyz_t=xyz_t,
            alpha_t=alpha_t,
            motif_mask=motif,
        )
    _, allatom = ComputeAllAtomCoords()(sequence.argmax(dim=-1), trajectory[-1], torsion[-1])
    values = (*distance, sequence_logits.permute(0, 2, 1), resolved, trajectory, torsion, allatom, plddt)
    return dict(zip(OUTPUT_NAMES, (value.cpu() for value in values)))


def native(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    sys.path.insert(0, str(ROOT))
    from ofoldx.models.rfdiffusion.convert import (
        convert_rfdiffusion_checkpoint,
        load_rfdiffusion_checkpoint,
    )
    from ofoldx.models.rfdiffusion.model import ComputeAllAtomCoords

    source = load_rfdiffusion_checkpoint(checkpoint_path)
    model = convert_rfdiffusion_checkpoint(checkpoint_path).eval()
    msa, msa_full, sequence, coords, index, t1d, t2d, xyz_t, alpha_t, motif = inputs(source["config_dict"])
    with torch.no_grad():
        output = model.network(
            msa,
            msa_full,
            sequence,
            coords,
            index,
            template_1d=t1d,
            template_2d=t2d,
            template_coords=xyz_t,
            alpha_t=alpha_t,
            motif_mask=motif,
        )
    _, allatom = ComputeAllAtomCoords()(sequence.argmax(dim=-1), output[2], output[4])
    values = (*output[7], output[5], output[9], output[10], output[11], allatom, output[8])
    return dict(zip(OUTPUT_NAMES, (value.cpu() for value in values)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("reference", "native"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--reference-output", type=Path)
    parser.add_argument("--atol", type=float, default=5e-3)
    args = parser.parse_args()

    if args.mode == "reference":
        if args.reference_root is None:
            parser.error("reference mode requires --reference-root")
        torch.save(reference(args.checkpoint, args.reference_root), args.output)
        return

    if args.reference_output is None:
        parser.error("native mode requires --reference-output")
    expected = torch.load(args.reference_output, map_location="cpu", weights_only=True)
    actual = native(args.checkpoint)
    comparison = {}
    maximum = 0.0
    for name in OUTPUT_NAMES:
        expected_value = expected[name][..., :14, :] if name == "allatom" else expected[name]
        difference = (actual[name] - expected_value).abs()
        value = float(difference.max())
        maximum = max(maximum, value)
        comparison[name] = {
            "shape": list(actual[name].shape),
            "max_absolute_error": value,
            "mean_absolute_error": float(difference.mean()),
        }
    report = {
        "checkpoint": args.checkpoint.name,
        "tensor_count": len(actual),
        "maximum_absolute_error": maximum,
        "tolerance": args.atol,
        "passed": maximum <= args.atol,
        "comparison": comparison,
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit("RFdiffusion network parity failed")


if __name__ == "__main__":
    main()
