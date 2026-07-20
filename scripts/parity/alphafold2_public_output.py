#!/usr/bin/env python
r"""Compare public AlphaFold 2 outputs with an OpenFold reference.

The capture records the supplied OpenFold checkout's actual revision and dirty
state. Both modes read the same official DeepMind NPZ, query-only sequence input,
recycle count, and seed. Run ``reference`` in the OpenFold environment, then
``native`` in the OFoldX environment against the local capture.

This is not yet publication signoff: OFoldX still lacks released stochastic MSA
sampling/masking, per-recycle resampling, and ensemble inference. The native report
therefore remains blocked even when this deterministic fixture happens to match.
Captures and JSON reports are local run outputs; keep them under the ignored
``artifacts/parity`` directory or another scratch directory.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.parity.paths import reference_root, weights_root  # noqa: E402

CURRENT_BLOCKERS = (
    "OFoldX does not yet implement released stochastic MSA sampling and masking.",
    "OFoldX reuses one processed feature slice instead of resampling each recycle.",
    "OFoldX does not yet expose released ensemble inference or its seed contract.",
    "Variant-specific MSA/template limits are not yet complete in the public Processor.",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("mode", choices=("reference", "native"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=weights_root() / "af2-params/params_model_1_ptm.npz",
    )
    parser.add_argument("--model-name", default="model_1_ptm")
    parser.add_argument("--sequence", default="ACDEFGHIK")
    parser.add_argument("--recycling-steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--reference-root", type=Path, default=reference_root("openfold"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reference-output", type=Path)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    args = parser.parse_args()

    if args.recycling_steps < 0:
        parser.error("--recycling-steps must be non-negative")
    sequence = args.sequence.strip().upper()
    if not sequence:
        parser.error("--sequence must not be empty")

    if args.mode == "reference":
        if args.output is None:
            parser.error("reference mode requires --output")
        payload = run_reference(
            checkpoint=args.checkpoint,
            model_name=args.model_name,
            sequence=sequence,
            recycling_steps=args.recycling_steps,
            seed=args.seed,
            device=args.device,
            reference=args.reference_root,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, args.output)
        print(
            json.dumps(
                {
                    "mode": "reference",
                    "output": str(args.output),
                    "reference_revision": payload["reference_revision"],
                    "reference_dirty": payload["reference_dirty"],
                    "checkpoint": str(args.checkpoint),
                    "model_name": args.model_name,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    if args.reference_output is None:
        parser.error("native mode requires --reference-output")
    expected = torch.load(args.reference_output, map_location="cpu", weights_only=True)
    _check_metadata(
        expected["metadata"],
        checkpoint=args.checkpoint,
        model_name=args.model_name,
        sequence=sequence,
        recycling_steps=args.recycling_steps,
        seed=args.seed,
        device=args.device,
    )
    actual = run_native(
        checkpoint=args.checkpoint,
        model_name=args.model_name,
        sequence=sequence,
        recycling_steps=args.recycling_steps,
        seed=args.seed,
        device=args.device,
    )
    comparisons = compare_outputs(expected["outputs"], actual, atol=args.atol, rtol=args.rtol)
    numerical_match = all(value["passed"] for value in comparisons.values())
    report = {
        "reference_repository": "https://github.com/aqlaboratory/openfold",
        "reference_revision": expected["reference_revision"],
        "reference_dirty": expected["reference_dirty"],
        "checkpoint": str(args.checkpoint),
        "model_name": args.model_name,
        "sequence": sequence,
        "msa": "query-only",
        "recycling_steps": args.recycling_steps,
        "seed": args.seed,
        "device": args.device,
        "comparisons": comparisons,
        "numerical_match": numerical_match,
        "blockers": CURRENT_BLOCKERS,
        "passed": numerical_match and not CURRENT_BLOCKERS,
    }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text, encoding="utf-8")
    raise SystemExit(2 if CURRENT_BLOCKERS else int(not numerical_match))


def run_reference(
    *,
    checkpoint: Path,
    model_name: str,
    sequence: str,
    recycling_steps: int,
    seed: int,
    device: str,
    reference: Path,
) -> dict[str, object]:
    revision = subprocess.run(
        ("git", "-C", str(reference), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ("git", "-C", str(reference), "status", "--porcelain"),
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    sys.path.insert(0, str(reference))

    from openfold.config import model_config
    from openfold.data.data_pipeline import empty_template_feats, make_msa_features, make_sequence_features
    from openfold.data.feature_pipeline import FeaturePipeline
    from openfold.model.model import AlphaFold
    from openfold.utils.import_weights import import_jax_weights_

    _deterministic(seed)
    config = model_config(model_name)
    config.data.common.max_recycling_iters = recycling_steps
    config.globals.chunk_size = None
    config.globals.use_lma = False
    config.globals.use_flash = False
    config.globals.offload_inference = False

    raw_features = dict(make_sequence_features(sequence, "ofoldx_parity", len(sequence)))
    aligned = [sequence]
    deletions = [[0] * len(sequence)]
    raw_features.update(make_msa_features([aligned], [deletions]))
    raw_features.update(empty_template_feats(len(sequence)))

    processed = FeaturePipeline(config.data).process_features(raw_features, mode="predict")
    batch = {name: value.to(device) for name, value in processed.items()}
    model = AlphaFold(config).eval()
    import_jax_weights_(model, str(checkpoint), version=model_name)
    model = model.to(device)
    with torch.inference_mode():
        output = model(batch)
    outputs = {
        "final_atom_positions": output["final_atom_positions"].detach().float().cpu(),
        "final_atom_mask": output["final_atom_mask"].detach().bool().cpu(),
        "plddt": output["plddt"].detach().float().cpu(),
        "distogram_logits": output["distogram_logits"].detach().float().cpu(),
        "plddt_logits": output["lddt_logits"].detach().float().cpu(),
        "resolved_logits": output["experimentally_resolved_logits"].detach().float().cpu(),
        "mean_plddt": output["plddt"].detach().float().mean().cpu(),
    }
    if "predicted_aligned_error" in output:
        outputs.update(
            pae=output["predicted_aligned_error"].detach().float().cpu(),
            pae_logits=output["tm_logits"].detach().float().cpu(),
            ptm=output["predicted_tm_score"].detach().float().cpu(),
        )
    return {
        "reference_revision": revision,
        "reference_dirty": dirty,
        "metadata": _metadata(
            checkpoint=checkpoint,
            model_name=model_name,
            sequence=sequence,
            recycling_steps=recycling_steps,
            seed=seed,
            device=device,
        ),
        "outputs": outputs,
    }


def run_native(
    *,
    checkpoint: Path,
    model_name: str,
    sequence: str,
    recycling_steps: int,
    seed: int,
    device: str,
) -> dict[str, torch.Tensor]:
    from ofoldx.data import residue_constants
    from ofoldx.data.molecular import AtomSite
    from ofoldx.models.alphafold2 import AlphaFold2Processor
    from ofoldx.models.alphafold2.convert import convert_alphafold2_checkpoint

    _deterministic(seed)
    processor = AlphaFold2Processor(device=device)
    batch = processor(sequence)
    model = (
        convert_alphafold2_checkpoint(
            checkpoint,
            source="deepmind",
            model_name=model_name,
        )
        .to(device)
        .eval()
    )
    model.set_backend("torch")

    # The released distogram/plddt/resolved/pae LOGITS are not carried on the public
    # FoldPrediction; capture them from a direct raw AlphaFold2Output forward, then take
    # coordinates and confidences from the deterministic public fold() surface.
    with torch.inference_mode():
        output = model(batch, recycling_steps=recycling_steps)
    result = model.fold(batch, recycling_steps=recycling_steps)
    prediction = result.predictions[0]
    structure = prediction.structure
    coordinate_index = {site: index for index, site in enumerate(structure.sites)}
    positions = torch.zeros(len(batch.layout.residue_sites), residue_constants.atom_type_num, 3)
    mask = torch.zeros(len(batch.layout.residue_sites), residue_constants.atom_type_num, dtype=torch.bool)
    for residue, (instance_id, monomer_index) in enumerate(batch.layout.residue_sites):
        for atom_name, atom_index in residue_constants.atom_order.items():
            index = coordinate_index.get(AtomSite(instance_id, monomer_index, atom_name))
            if index is None or not structure.coordinate_mask[index]:
                continue
            positions[residue, atom_index] = torch.from_numpy(structure.coordinates[index].copy())
            mask[residue, atom_index] = True

    outputs = {
        "final_atom_positions": positions,
        "final_atom_mask": mask,
        "plddt": torch.from_numpy(prediction.metrics["plddt"].values.copy()),
        "distogram_logits": output.distogram_logits[0].detach().float().cpu(),
        "plddt_logits": output.plddt_logits[0].detach().float().cpu(),
        "resolved_logits": output.resolved_logits[0].detach().float().cpu(),
        "mean_plddt": torch.tensor(prediction.scores["mean_plddt"].value),
    }
    if output.pae_logits is not None:
        outputs.update(
            pae=torch.from_numpy(prediction.pair_metrics["pae"].values.copy()),
            pae_logits=output.pae_logits[0].detach().float().cpu(),
            ptm=torch.tensor(prediction.scores["ptm"].value),
        )
    return outputs


def compare_outputs(
    expected: dict[str, torch.Tensor],
    actual: dict[str, torch.Tensor],
    *,
    atol: float,
    rtol: float,
) -> dict[str, dict[str, Any]]:
    comparisons: dict[str, dict[str, Any]] = {}
    expected_names = set(expected)
    actual_names = set(actual)
    if expected_names != actual_names:
        comparisons["output_names"] = {
            "expected": sorted(expected_names),
            "actual": sorted(actual_names),
            "passed": False,
        }
    if "final_atom_mask" in expected_names & actual_names:
        reference_mask = expected["final_atom_mask"].bool()
        native_mask = actual["final_atom_mask"].bool()
        comparisons["final_atom_mask"] = {
            "shape": list(native_mask.shape),
            "different_values": (
                int((reference_mask != native_mask).sum()) if reference_mask.shape == native_mask.shape else None
            ),
            "passed": reference_mask.shape == native_mask.shape and torch.equal(reference_mask, native_mask),
        }
    for name in sorted((expected_names & actual_names) - {"final_atom_mask", "final_atom_positions"}):
        comparisons[name] = _comparison(actual[name], expected[name], atol=atol, rtol=rtol)
    if "final_atom_positions" in expected_names & actual_names:
        reference = expected["final_atom_positions"]
        native = actual["final_atom_positions"]
        if reference.shape == native.shape and expected["final_atom_mask"].shape == actual["final_atom_mask"].shape:
            common = expected["final_atom_mask"].bool() & actual["final_atom_mask"].bool()
            comparisons["final_atom_positions"] = _comparison(
                native[common],
                reference[common],
                atol=atol,
                rtol=rtol,
            )
        else:
            comparisons["final_atom_positions"] = {
                "expected_shape": list(reference.shape),
                "actual_shape": list(native.shape),
                "passed": False,
            }
    return comparisons


def _comparison(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    actual = actual.detach().float().cpu()
    expected = expected.detach().float().cpu()
    if actual.shape != expected.shape:
        return {
            "expected_shape": list(expected.shape),
            "actual_shape": list(actual.shape),
            "passed": False,
        }
    difference = (actual - expected).abs()
    relative = difference / expected.abs().clamp_min(1e-9)
    return {
        "shape": list(actual.shape),
        "max_absolute_error": float(difference.max()) if difference.numel() else 0.0,
        "mean_absolute_error": float(difference.mean()) if difference.numel() else 0.0,
        "max_relative_error": float(relative.max()) if relative.numel() else 0.0,
        "passed": torch.allclose(actual, expected, atol=atol, rtol=rtol),
    }


def _metadata(
    *,
    checkpoint: Path,
    model_name: str,
    sequence: str,
    recycling_steps: int,
    seed: int,
    device: str,
) -> dict[str, object]:
    return {
        "checkpoint": str(checkpoint.expanduser().resolve()),
        "model_name": model_name,
        "sequence": sequence,
        "msa": "query-only",
        "recycling_steps": recycling_steps,
        "seed": seed,
        "device": device,
    }


def _check_metadata(
    metadata: dict[str, object],
    *,
    checkpoint: Path,
    model_name: str,
    sequence: str,
    recycling_steps: int,
    seed: int,
    device: str,
) -> None:
    current = _metadata(
        checkpoint=checkpoint,
        model_name=model_name,
        sequence=sequence,
        recycling_steps=recycling_steps,
        seed=seed,
        device=device,
    )
    if metadata != current:
        raise ValueError(f"reference capture input differs from this run: {metadata!r} != {current!r}")


def _deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


if __name__ == "__main__":
    main()
