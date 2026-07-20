#!/usr/bin/env python
r"""Two-process golden-dump parity for BioHub ESMFold2 vs OFoldX ESMFold2.

Single-process parity is impossible: the released reference (BioHub transformers
4.57 fork + huggingface-hub<1.0 + tokenizers<=0.23) and the OFoldX side (EsmC via
multimolecule, transformers 5.x + huggingface-hub 1.x) need mutually exclusive
transformers / huggingface-hub majors and cannot coexist in one interpreter.

The parity therefore runs in two processes sharing a golden dump on disk:

  * ``dump`` runs inside the reference env (``ref-esmfold2``): it loads the
    released ``ESMFold2Model`` + BioHub-format ``ESMC-6B`` head, folds one
    sequence, and writes the golden reference tensors (positions, plddt/pae/pde
    logits, distogram logits, derived scores, and lowered inputs) to a ``.pt``.

  * ``compare`` runs inside the OFoldX env (``zyc-esmfold2``): it folds the same
    sequence with OFoldX ESMFold2 (multimolecule EsmC) and reports per-quantity
    ``max_abs`` / ``mean_abs`` errors against the golden dump.

``semantic`` keeps the model-free public-boundary checks. ``full`` retains the
historical single-process path for environments where both stacks can coexist.

RNG determinism across the process boundary: both processes seed identically
(``--seed``) before the diffusion sampler and run on the same torch build, so the
noise stream matches exactly as it did in-process; ``compare`` reads the run
parameters from the golden metadata to guarantee identical settings.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.parity.paths import reference_root, repo_root  # noqa: E402

_HEADS = {
    "standard": "biohub/ESMFold2",
    "fast": "biohub/ESMFold2-Fast",
}
_DEFAULT_SEQUENCE = "MQIFVKTLTGKTITLEVEPS"


def _default_golden(variant: str) -> Path:
    return repo_root() / "artifacts/parity/esmfold2" / f"golden_{variant}.pt"


def _add_common_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--variant", choices=tuple(_HEADS), default="standard")
    parser.add_argument("--sequence", default=_DEFAULT_SEQUENCE)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-loops", type=int, default=3)
    parser.add_argument("--num-sampling-steps", type=int, default=14)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--lm-mask-probability", type=float, default=None)


def _lm_mask_probability(variant: str, override: float | None) -> float:
    if override is not None:
        return override
    return 0.1 if variant == "fast" else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    dump = sub.add_parser("dump", help="Reference process: fold + write golden dump")
    _add_common_run_args(dump)
    dump.add_argument("--head", type=Path, required=True, help="Local released folding-head repository")
    dump.add_argument("--esmc", type=Path, required=True, help="Local BioHub-format ESMC-6B repository")
    dump.add_argument("--ccd", type=Path, required=True, help="Local BioHub ESMFold2 ccd.pkl")
    dump.add_argument(
        "--biohub-transformers",
        type=Path,
        default=reference_root("biohub_transformers"),
    )
    dump.add_argument("--esmc-precision", default="bf16", choices=("bf16", "fp32"))
    dump.add_argument("--output", type=Path, default=None, help="Golden dump destination (.pt)")

    compare = sub.add_parser("compare", help="OFoldX process: fold + compare against golden dump")
    _add_common_run_args(compare)
    compare.add_argument("--head", type=Path, required=True, help="Local released folding-head repository")
    compare.add_argument("--ofoldx-esmc", type=Path, required=True, help="Local MultiMolecule ESMC-6B repository")
    compare.add_argument("--golden", type=Path, default=None, help="Golden dump produced by `dump` (.pt)")
    compare.add_argument("--atol", type=float, default=1e-4)
    compare.add_argument("--rtol", type=float, default=1e-4)
    compare.add_argument(
        "--deterministic",
        action="store_true",
        help="Force torch.use_deterministic_algorithms(True) to kill run-to-run CUDA variance",
    )
    compare.add_argument(
        "--force-fp32-trunk",
        action="store_true",
        help="Monkeypatch torch.autocast to a no-op so the OFoldX head/trunk runs in fp32 (match fp32 golden)",
    )

    full = sub.add_parser("full", help="Legacy single-process parity (needs both stacks in one env)")
    _add_common_run_args(full)
    full.add_argument("--head", type=Path, required=True)
    full.add_argument("--upstream-esmc", type=Path, required=True)
    full.add_argument("--ofoldx-esmc", type=Path, required=True)
    full.add_argument("--ccd", type=Path, required=True)
    full.add_argument("--biohub-esm", type=Path, default=reference_root("biohub_esm"))
    full.add_argument("--biohub-transformers", type=Path, default=reference_root("biohub_transformers"))
    full.add_argument("--atol", type=float, default=1e-4)
    full.add_argument("--rtol", type=float, default=1e-4)

    sub.add_parser("semantic", help="Model-free public-boundary checks")

    args = parser.parse_args()

    if args.mode == "semantic":
        semantic_checks()
    elif args.mode == "dump":
        dump_reference(args)
    elif args.mode == "compare":
        compare_ofoldx(args)
    elif args.mode == "full":
        run_full(args)
    else:  # pragma: no cover - argparse enforces the choices
        parser.error(f"unknown mode {args.mode!r}")


# ---------------------------------------------------------------------------
# Process 1 — reference golden dump (BioHub transformers 4.57 fork)
# ---------------------------------------------------------------------------
def dump_reference(args: argparse.Namespace) -> None:
    head = args.head.expanduser()
    esmc = args.esmc.expanduser()
    ccd = args.ccd.expanduser()
    transformers_repository = args.biohub_transformers.expanduser()
    transformers_source = transformers_repository / "src"
    for label, path in (("folding-head repository", head), ("BioHub ESMC repository", esmc)):
        if not path.is_dir():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    if not ccd.is_file():
        raise FileNotFoundError(f"BioHub ESMFold2 CCD pickle does not exist: {ccd}")
    if not transformers_source.is_dir():
        raise FileNotFoundError(f"BioHub Transformers source does not exist: {transformers_source}")

    transformers_revision, transformers_dirty = _git_state(transformers_repository)

    os.environ["ESMCFOLD_CCD_PATH"] = str(ccd)
    sys.path.insert(0, str(transformers_source))

    import torch
    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model as ReferenceModel

    device = torch.device(args.device)
    lm_mask_probability = _lm_mask_probability(args.variant, args.lm_mask_probability)

    with redirect_stdout(sys.stderr):
        reference_model = (
            ReferenceModel.from_pretrained(str(head), load_esmc=False, local_files_only=True).to(device).eval()
        )
        reference_model.load_esmc(str(esmc), precision=args.esmc_precision)
        reference_model.set_kernel_backend(None)
        reference_model.set_chunk_size(args.chunk_size)
        with _seeded(args.seed, device):
            output = reference_model.infer_protein(
                args.sequence,
                num_loops=args.num_loops,
                num_diffusion_samples=1,
                num_sampling_steps=args.num_sampling_steps,
                lm_mask_pct=lm_mask_probability,
            )

    values = _reference_values(output)
    meta = {
        "sequence": args.sequence,
        "variant": args.variant,
        "published_head": _HEADS[args.variant],
        "head": str(head),
        "esmc": str(esmc),
        "esmc_precision": args.esmc_precision,
        "ccd": str(ccd),
        "seed": args.seed,
        "device": str(device),
        "num_loops": args.num_loops,
        "num_sampling_steps": args.num_sampling_steps,
        "chunk_size": args.chunk_size,
        "lm_mask_probability": lm_mask_probability,
        "torch_version": torch.__version__,
        "biohub_transformers": {"revision": transformers_revision, "dirty": transformers_dirty},
    }

    output_path = args.output.expanduser() if args.output is not None else _default_golden(args.variant)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"meta": meta, "values": values}, output_path)

    summary = {
        "mode": "dump",
        "golden": str(output_path),
        "meta": meta,
        "tensor_shapes": {
            key: (list(value.shape) if hasattr(value, "shape") else len(value))
            for key, value in values.items()
            if key not in {"inputs", "logits"}
        },
        "logit_shapes": {key: list(value.shape) for key, value in values["logits"].items()},
        "num_atoms": len(values["atom_names"]),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


# ---------------------------------------------------------------------------
# Process 2 — OFoldX fold + compare against the golden dump (transformers 5.x)
# ---------------------------------------------------------------------------
def compare_ofoldx(args: argparse.Namespace) -> None:
    head = args.head.expanduser()
    ofoldx_esmc = args.ofoldx_esmc.expanduser()
    golden_path = args.golden.expanduser() if args.golden is not None else _default_golden(args.variant)
    for label, path in (("folding-head repository", head), ("MultiMolecule ESMC repository", ofoldx_esmc)):
        if not path.is_dir():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    if not golden_path.is_file():
        raise FileNotFoundError(f"golden dump does not exist (run `dump` first): {golden_path}")

    import torch

    # (a) CUDA determinism: kill the run-to-run coord variance from nondeterministic
    # reduction kernels. CUBLAS_WORKSPACE_CONFIG must already be in the environment
    # (set before the CUDA context is created); we only flip the algorithm switch here.
    determinism = {"requested": bool(args.deterministic)}
    if args.deterministic:
        cublas_cfg = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        determinism["cublas_workspace_config"] = cublas_cfg
        determinism["cublas_ok"] = cublas_cfg in (":4096:8", ":16:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # (b) Precision match: the fp32 golden was produced by a reference head that runs
    # fully in fp32 (no autocast). The OFoldX head wraps its trunk in
    # ``torch.autocast(..., bfloat16)`` (model.py:374/420/439/1903/3147). Rather than
    # edit model code, neutralise autocast in this process only so the whole OFoldX
    # forward runs at its fp32 parameter precision.
    force_fp32_trunk = bool(args.force_fp32_trunk)
    if force_fp32_trunk:

        class _NoAutocast:
            def __init__(self, *a: Any, **k: Any) -> None:
                pass

            def __enter__(self) -> "_NoAutocast":
                return self

            def __exit__(self, *exc: Any) -> bool:
                return False

        torch.autocast = _NoAutocast  # type: ignore[assignment]

    golden = torch.load(golden_path, map_location="cpu", weights_only=False)
    meta = golden["meta"]
    reference_values = golden["values"]

    # Mirror the reference run parameters exactly so the diffusion RNG stream and
    # trunk inputs match; the golden metadata is the single source of truth.
    sequence = meta["sequence"]
    seed = meta["seed"]
    num_loops = meta["num_loops"]
    num_sampling_steps = meta["num_sampling_steps"]
    chunk_size = meta["chunk_size"]
    lm_mask_probability = meta["lm_mask_probability"]

    from multimolecule import EsmCModel

    import ofoldx.models.esmfold2.convert as conversion
    from ofoldx.models.esmfold2 import (
        ESMFold2Config,
        ESMFold2ESMCConfig,
        ESMFold2Model,
        ESMFold2Processor,
        ESMFold2ProcessorConfig,
    )

    device = torch.device(args.device)

    with redirect_stdout(sys.stderr):
        esmc = EsmCModel.from_pretrained(str(ofoldx_esmc), local_files_only=True).eval()
        checkpoint_path, config_path = conversion.resolve_esmfold2_source(head, local_files_only=True)
        raw_config = json.loads(config_path.read_text(encoding="utf-8"))
        translated = conversion._translate_biohub_config(raw_config)
        esmc_fields = ESMFold2ESMCConfig().dict()
        translated["esmc"] = {name: getattr(esmc.config, name) for name in esmc_fields}
        config = ESMFold2Config(translated)
        model = ESMFold2Model(config, esmc=esmc)
        head_state = conversion.load_esmfold2_state_dict(checkpoint_path)
        for name in conversion._BIOHUB_UNUSED_STATE_KEYS:
            head_state.pop(name, None)
        incompatible = model.load_state_dict(head_state, strict=False)
        expected_missing = {f"esmc.{name}" for name in model.esmc.state_dict()}
        if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "converted ESMFold2 head does not strict-load: "
                f"missing={sorted(set(incompatible.missing_keys) - expected_missing)!r}, "
                f"unexpected={sorted(incompatible.unexpected_keys)!r}"
            )
        head_tensors = len(head_state)
        del head_state
        # Mirror the reference precision split exactly: the released reference runs
        # the folding head in float32 (``reference_model.to(device)``) and the ESMC
        # LM at ``meta["esmc_precision"]`` (``load_esmc(precision=...)``). The OFoldX
        # head is built for float32 params + internal autocast, so casting the whole
        # model to bf16 both diverges from the reference and breaks the non-autocast
        # distogram head.
        model.to(device=device).eval()
        esmc_dtype = torch.bfloat16 if meta.get("esmc_precision", "bf16") == "bf16" else torch.float32
        model.esmc.to(dtype=esmc_dtype)
        model.set_backend("torch")
        model.set_chunk_size(chunk_size)
        processor = ESMFold2Processor(
            ESMFold2ProcessorConfig(msa_conditioning=config.msa_encoder.enabled),
            device=device,
        )
        batch = processor(sequence)
        result = model.fold(
            batch,
            num_samples=1,
            seed=seed,
            num_loops=num_loops,
            num_sampling_steps=num_sampling_steps,
            language_model_mask_probability=lm_mask_probability,
            language_model_dropout=None,
            msa_column_mask_rate=0.0,
            msa_subsample_at_inference=False,
            max_inference_sigma=256.0,
        )

    prediction = result.predictions[0]
    visible_atoms = prediction.structure.coordinate_mask
    ofoldx_inputs = {
        "res_type": batch.res_type,
        "atom_to_token": batch.atom_to_token,
        "ref_atom_name_chars": batch.ref_atom_name_chars,
        "atom_attention_mask": batch.atom_attention_mask,
        "token_attention_mask": batch.token_attention_mask,
        "residue_index": batch.residue_index,
    }
    comparisons = {
        **{
            f"input.{name}": _comparison(value, reference_values["inputs"][name])
            for name, value in ofoldx_inputs.items()
        },
        "fold.coordinates": _comparison(
            prediction.structure.coordinates[visible_atoms],
            reference_values["coordinates"],
        ),
        "fold.plddt": _comparison(prediction.metrics["plddt"].values, reference_values["plddt"]),
        "fold.pae": _comparison(prediction.pair_metrics["pae"].values, reference_values["pae"]),
        "fold.pde": _comparison(prediction.pair_metrics["pde"].values, reference_values["pde"]),
        "fold.ptm": _comparison(prediction.scores["ptm"].value, reference_values["ptm"]),
        "fold.iptm": _comparison(prediction.scores["iptm"].value, reference_values["iptm"]),
        "fold.complex_plddt": _comparison(
            prediction.scores["complex_plddt"].value,
            reference_values["complex_plddt"],
        ),
        "fold.complex_iplddt": _comparison(
            prediction.scores["complex_iplddt"].value,
            reference_values["complex_iplddt"],
        ),
        "fold.pair_chains_iptm": _comparison(
            prediction.pair_metrics["pair_chains_iptm"].values,
            reference_values["pair_chains_iptm"],
        ),
    }
    ofoldx_atom_names = [site.atom_id for site, present in zip(prediction.structure.sites, visible_atoms) if present]
    identity = {
        "atom_names": ofoldx_atom_names == reference_values["atom_names"],
        "visible_atom_count": len(ofoldx_atom_names) == len(reference_values["atom_names"]),
    }
    passed = all(identity.values()) and all(
        comparison["max_absolute_error"] <= args.atol or comparison["max_relative_error"] <= args.rtol
        for comparison in comparisons.values()
    )
    report = {
        "mode": "compare",
        "golden": str(golden_path),
        "golden_meta": meta,
        "head": str(head),
        "ofoldx_esmc": str(ofoldx_esmc),
        "checkpoint": str(checkpoint_path),
        "head_tensors": head_tensors,
        "strict_head_load": True,
        "device": str(device),
        "atol": args.atol,
        "rtol": args.rtol,
        "deterministic": determinism,
        "force_fp32_trunk": force_fp32_trunk,
        "identity": identity,
        "comparisons": comparisons,
        "passed": passed,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# Legacy single-process path (kept for coexisting-stack environments)
# ---------------------------------------------------------------------------
def run_full(args: argparse.Namespace) -> None:
    head = args.head.expanduser()
    upstream_esmc = args.upstream_esmc.expanduser()
    ofoldx_esmc = args.ofoldx_esmc.expanduser()
    ccd = args.ccd.expanduser()
    transformers_repository = args.biohub_transformers.expanduser()
    esm_repository = args.biohub_esm.expanduser()
    transformers_source = transformers_repository / "src"
    esm_source = esm_repository
    for label, path in (
        ("folding-head repository", head),
        ("BioHub ESMC repository", upstream_esmc),
        ("MultiMolecule ESMC repository", ofoldx_esmc),
    ):
        if not path.is_dir():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    if not ccd.is_file():
        raise FileNotFoundError(f"BioHub ESMFold2 CCD pickle does not exist: {ccd}")
    if not transformers_source.is_dir():
        raise FileNotFoundError(f"BioHub Transformers source does not exist: {transformers_source}")
    if not esm_source.is_dir():
        raise FileNotFoundError(f"BioHub ESM source does not exist: {esm_source}")

    esm_revision, esm_dirty = _git_state(esm_repository)
    transformers_revision, transformers_dirty = _git_state(transformers_repository)

    os.environ["ESMCFOLD_CCD_PATH"] = str(ccd)
    sys.path.insert(0, str(esm_source))
    sys.path.insert(0, str(transformers_source))

    import torch
    from multimolecule import EsmCModel
    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model as ReferenceModel

    import ofoldx.models.esmfold2.convert as conversion
    from ofoldx.models.esmfold2 import (
        ESMFold2Config,
        ESMFold2ESMCConfig,
        ESMFold2Model,
        ESMFold2Processor,
        ESMFold2ProcessorConfig,
    )

    device = torch.device(args.device)
    lm_mask_probability = _lm_mask_probability(args.variant, args.lm_mask_probability)

    with redirect_stdout(sys.stderr):
        reference_model = (
            ReferenceModel.from_pretrained(str(head), load_esmc=False, local_files_only=True).to(device).eval()
        )
        reference_model.load_esmc(str(upstream_esmc), precision="bf16")
        reference_model.set_kernel_backend(None)
        reference_model.set_chunk_size(args.chunk_size)
        with _seeded(args.seed, device):
            reference_output = reference_model.infer_protein(
                args.sequence,
                num_loops=args.num_loops,
                num_diffusion_samples=1,
                num_sampling_steps=args.num_sampling_steps,
                lm_mask_pct=lm_mask_probability,
            )
    reference_values = _reference_values(reference_output)
    del reference_output, reference_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    with redirect_stdout(sys.stderr):
        esmc = EsmCModel.from_pretrained(str(ofoldx_esmc), local_files_only=True).eval()
        checkpoint_path, config_path = conversion.resolve_esmfold2_source(head, local_files_only=True)
        raw_config = json.loads(config_path.read_text(encoding="utf-8"))
        translated = conversion._translate_biohub_config(raw_config)
        esmc_fields = ESMFold2ESMCConfig().dict()
        translated["esmc"] = {name: getattr(esmc.config, name) for name in esmc_fields}
        config = ESMFold2Config(translated)
        model = ESMFold2Model(config, esmc=esmc)
        head_state = conversion.load_esmfold2_state_dict(checkpoint_path)
        for name in conversion._BIOHUB_UNUSED_STATE_KEYS:
            head_state.pop(name, None)
        incompatible = model.load_state_dict(head_state, strict=False)
        expected_missing = {f"esmc.{name}" for name in model.esmc.state_dict()}
        if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "converted ESMFold2 head does not strict-load: "
                f"missing={sorted(set(incompatible.missing_keys) - expected_missing)!r}, "
                f"unexpected={sorted(incompatible.unexpected_keys)!r}"
            )
        head_tensors = len(head_state)
        del head_state
        # Reference precision split: float32 folding head + bf16 ESMC LM.
        model.to(device=device).eval()
        model.esmc.to(dtype=torch.bfloat16)
        model.set_backend("torch")
        model.set_chunk_size(args.chunk_size)
        processor = ESMFold2Processor(
            ESMFold2ProcessorConfig(msa_conditioning=config.msa_encoder.enabled),
            device=device,
        )
        batch = processor(args.sequence)
        result = model.fold(
            batch,
            num_samples=1,
            seed=args.seed,
            num_loops=args.num_loops,
            num_sampling_steps=args.num_sampling_steps,
            language_model_mask_probability=lm_mask_probability,
            language_model_dropout=None,
            msa_column_mask_rate=0.0,
            msa_subsample_at_inference=False,
            max_inference_sigma=256.0,
        )

    prediction = result.predictions[0]
    visible_atoms = prediction.structure.coordinate_mask
    ofoldx_inputs = {
        "res_type": batch.res_type,
        "atom_to_token": batch.atom_to_token,
        "ref_atom_name_chars": batch.ref_atom_name_chars,
        "atom_attention_mask": batch.atom_attention_mask,
        "token_attention_mask": batch.token_attention_mask,
        "residue_index": batch.residue_index,
    }
    comparisons = {
        **{
            f"input.{name}": _comparison(value, reference_values["inputs"][name])
            for name, value in ofoldx_inputs.items()
        },
        "fold.coordinates": _comparison(
            prediction.structure.coordinates[visible_atoms],
            reference_values["coordinates"],
        ),
        "fold.plddt": _comparison(prediction.metrics["plddt"].values, reference_values["plddt"]),
        "fold.pae": _comparison(prediction.pair_metrics["pae"].values, reference_values["pae"]),
        "fold.pde": _comparison(prediction.pair_metrics["pde"].values, reference_values["pde"]),
        "fold.ptm": _comparison(prediction.scores["ptm"].value, reference_values["ptm"]),
        "fold.iptm": _comparison(prediction.scores["iptm"].value, reference_values["iptm"]),
        "fold.complex_plddt": _comparison(
            prediction.scores["complex_plddt"].value,
            reference_values["complex_plddt"],
        ),
        "fold.complex_iplddt": _comparison(
            prediction.scores["complex_iplddt"].value,
            reference_values["complex_iplddt"],
        ),
        "fold.pair_chains_iptm": _comparison(
            prediction.pair_metrics["pair_chains_iptm"].values,
            reference_values["pair_chains_iptm"],
        ),
    }
    ofoldx_atom_names = [site.atom_id for site, present in zip(prediction.structure.sites, visible_atoms) if present]
    identity = {
        "atom_names": ofoldx_atom_names == reference_values["atom_names"],
        "visible_atom_count": len(ofoldx_atom_names) == len(reference_values["atom_names"]),
    }
    passed = all(identity.values()) and all(
        comparison["max_absolute_error"] <= args.atol or comparison["max_relative_error"] <= args.rtol
        for comparison in comparisons.values()
    )
    report = {
        "mode": "full",
        "reference_repositories": {
            "Biohub/esm": {"revision": esm_revision, "dirty": esm_dirty},
            "Biohub/transformers": {"revision": transformers_revision, "dirty": transformers_dirty},
        },
        "variant": args.variant,
        "published_head": _HEADS[args.variant],
        "head": str(head),
        "upstream_esmc": str(upstream_esmc),
        "ofoldx_esmc": str(ofoldx_esmc),
        "ccd": str(ccd),
        "checkpoint": str(checkpoint_path),
        "head_tensors": head_tensors,
        "strict_head_load": True,
        "sequence": args.sequence,
        "device": str(device),
        "seed": args.seed,
        "num_loops": args.num_loops,
        "num_sampling_steps": args.num_sampling_steps,
        "lm_mask_probability": lm_mask_probability,
        "identity": identity,
        "comparisons": comparisons,
        "passed": passed,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


def _git_state(repository: Path) -> tuple[str, bool]:
    revision = subprocess.run(
        ("git", "-C", str(repository), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ("git", "-C", str(repository), "status", "--porcelain"),
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    return revision, dirty


def semantic_checks() -> None:
    r"""Check ESMFold2's public semantic boundary without loading model weights.

    Ported to the post-3.6 evidence and molecular-value model (evidence.py rewrite
    + esmfold2 processor MSA-lowering migration). The MSA-lowering assertions
    reproduce the exact lowered ``msa``/``has_deletion``/``deletion_value`` tensors
    the pre-migration check demanded, expressed through the new
    ``MSA``/``MSACollection`` paired/unpaired-pool + dense row-plan API, so the new
    evidence model is confirmed to yield identical released features rather than
    freshly invented expectations.
    """

    from dataclasses import replace

    import numpy as np
    import torch

    from ofoldx.data import (
        MSA,
        AtomSite,
        ChemicalAtom,
        ChemicalBond,
        ChemicalComponent,
        MolecularSystem,
        Molecule,
        MSACollection,
        PolymerSequence,
        SystemBond,
        get_component,
    )
    from ofoldx.models.esmfold2 import ESMFold2Processor, ESMFold2ProcessorConfig

    processor = ESMFold2Processor()

    # --- Mixed protein + covalent ligand: tokens, token bonds, reference coords ---
    protein = PolymerSequence.from_one_letter("A")
    ligand_component = ChemicalComponent(
        component_id="LIG",
        atoms=(ChemicalAtom("C1", "C", formal_charge=0), ChemicalAtom("O1", "O", formal_charge=0)),
        bonds=frozenset({ChemicalBond("C1", "O1")}),
        reference_coordinates=((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)),
    )
    ligand = Molecule(components=(ligand_component,))
    mixed_system = MolecularSystem(
        entities={"protein": protein, "ligand": ligand},
        instances={"A": "protein", "L": "ligand"},
        bonds=frozenset({SystemBond(AtomSite("A", 0, "CB"), AtomSite("L", 0, "C1"))}),
    )
    mixed = processor(mixed_system)
    ligand_c = AtomSite("L", 0, "C1")
    ligand_o = AtomSite("L", 0, "O1")
    expected_token_sites = (("A", 0), ligand_c, ligand_o)
    if mixed.layout.token_sites != expected_token_sites:
        raise AssertionError(f"mixed token identities differ: {mixed.layout.token_sites!r}")
    expected_token_bonds = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=mixed.token_bonds.dtype,
        device=mixed.token_bonds.device,
    )
    if not torch.equal(mixed.token_bonds[0, :3, :3, 0], expected_token_bonds):
        raise AssertionError("mixed covalent token bonds differ")
    ligand_atom_indices = [mixed.layout.atom_sites.index(site) for site in (ligand_c, ligand_o)]
    expected_reference_coordinates = torch.tensor(
        ligand_component.reference_coordinates,
        dtype=mixed.ref_pos.dtype,
        device=mixed.ref_pos.device,
    )
    if not torch.equal(mixed.ref_pos[0, ligand_atom_indices], expected_reference_coordinates):
        raise AssertionError("ligand reference coordinates differ")

    # --- Canonical N/DN residue grouping and modified-nucleotide atom tokenization ---
    canonical_rna = PolymerSequence("rna", (get_component("N"),))
    canonical_dna = PolymerSequence("dna", (get_component("DN"),))
    adenine = get_component("A")
    modified_component = ChemicalComponent(
        component_id="MOD",
        atoms=(adenine.atoms[0], replace(adenine.atoms[1], isotope=31), *adenine.atoms[2:]),
        bonds=adenine.bonds,
        previous_attachment=adenine.previous_attachment,
        next_attachment=adenine.next_attachment,
        reference_coordinates=adenine.reference_coordinates,
    )
    modified_rna = PolymerSequence("rna", (modified_component,))
    nucleic_system = MolecularSystem(
        entities={"rna_n": canonical_rna, "dna_n": canonical_dna, "modified": modified_rna},
        instances={"R": "rna_n", "D": "dna_n", "M": "modified"},
    )
    nucleic = processor(nucleic_system)
    if nucleic.layout.token_sites[:2] != (("R", 0), ("D", 0)):
        raise AssertionError("canonical N/DN residues were not grouped into residue tokens")
    if nucleic.res_type[0, :2].tolist() != [27, 32]:
        raise AssertionError(f"canonical N/DN token IDs differ: {nucleic.res_type[0, :2].tolist()!r}")
    modified_sites = nucleic.layout.token_sites[2:]
    expected_modified_tokens = sum(
        atom.element != "H" and atom.atom_id not in {"OXT", "OP3"} for atom in modified_component.atoms
    )
    if len(modified_sites) != expected_modified_tokens or not all(
        type(site) is AtomSite and site.instance_id == "M" for site in modified_sites
    ):
        raise AssertionError("caller-supplied modified nucleotide was not atom-tokenized")
    if nucleic.res_type[0, 2:].tolist() != [22] * expected_modified_tokens:
        raise AssertionError("caller-supplied modified nucleotide did not use token ID 22")

    # --- Paired/unpaired MSACollection rows and deletion features (§3.6 evidence) ---
    # Two protein columns AC / GD; the dense row plan assembles one paired non-query
    # row, one column-0-only unpaired row, and one column-1-only unpaired row. Pool
    # rows and deletion counts are chosen so the lowered features reproduce exactly
    # the tensors the pre-migration check asserted (A=2 C=6 G=9 D=5, gap=1).
    protein_a = PolymerSequence.from_one_letter("AC")
    protein_b = PolymerSequence.from_one_letter("GD")
    msa_a = MSA(
        entity_id="protein_a",
        unpaired_sequences=("AC", "GC"),
        unpaired_deletion_counts=np.array([[0, 0], [3, 4]], dtype=np.uint32),
        paired_sequences=("AC", "A-"),
        paired_deletion_counts=np.array([[0, 0], [1, 2]], dtype=np.uint32),
    )
    msa_b = MSA(
        entity_id="protein_b",
        unpaired_sequences=("GD", "-D"),
        unpaired_deletion_counts=np.array([[0, 0], [7, 8]], dtype=np.uint32),
        paired_sequences=("GD", "GD"),
        paired_deletion_counts=np.array([[0, 0], [5, 6]], dtype=np.uint32),
    )
    msa_collection = MSACollection(
        msas=(msa_a, msa_b),
        row_pairing=np.array([[0, 0], [1, 1], [1, -1], [-1, 1]], dtype=np.int64),
        row_is_paired=np.array(
            [[True, True], [True, True], [False, False], [False, False]],
            dtype=np.bool_,
        ),
    )
    msa_system = MolecularSystem(
        entities={"protein_a": protein_a, "protein_b": protein_b},
        instances={"P1": "protein_a", "P2": "protein_b"},
    )
    msa_features = processor(msa_system, msa=msa_collection)
    expected_msa = torch.tensor(
        [[[2, 6, 9, 5], [2, 1, 9, 5], [9, 6, 1, 1], [1, 1, 1, 5]]],
        dtype=msa_features.msa.dtype,
        device=msa_features.msa.device,
    )
    expected_deletions = torch.tensor(
        [[[0, 0, 0, 0], [1, 2, 5, 6], [3, 4, 0, 0], [0, 0, 7, 8]]],
        dtype=msa_features.deletion_value.dtype,
        device=msa_features.deletion_value.device,
    )
    if not torch.equal(msa_features.msa, expected_msa):
        raise AssertionError(f"paired/unpaired MSA rows differ: {msa_features.msa.tolist()!r}")
    if not torch.equal(msa_features.has_deletion, expected_deletions > 0):
        raise AssertionError("paired/unpaired MSA deletion masks differ")
    expected_deletion_values = (torch.pi / 2) * torch.atan(expected_deletions / 3)
    if not torch.equal(msa_features.deletion_value, expected_deletion_values):
        raise AssertionError("paired/unpaired MSA deletion values differ")
    if msa_features.layout.msa is not msa_collection:
        raise AssertionError("MSACollection identity was not retained in the layout")

    # --- Fast (single-sequence) variant rejects explicit MSA evidence ---
    fast = ESMFold2Processor(ESMFold2ProcessorConfig(msa_conditioning=False))
    try:
        fast(msa_system, msa=msa_collection)
    except ValueError as error:
        if "Fast variant" not in str(error):
            raise AssertionError(f"Fast Processor raised an unexpected error: {error}") from error
    else:
        raise AssertionError("Fast Processor accepted explicit MSA evidence")

    checks = (
        "mixed protein/covalent-ligand tokens, bonds, and reference coordinates",
        "canonical N/DN grouping and modified-nucleotide atom tokenization",
        "paired/unpaired MSACollection rows and deletion features",
        "Fast Processor explicit-MSA rejection",
    )
    print(json.dumps({"mode": "semantic", "checks": checks, "passed": True}, indent=2))
    print("ESMFold2 semantic-only checks passed")


@contextmanager
def _seeded(seed: int, device: Any) -> Iterator[None]:
    import torch

    devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices, enabled=True):
        torch.manual_seed(seed)
        if device.type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.manual_seed(seed)
        yield


def _reference_values(output: dict[str, Any]) -> dict[str, Any]:
    atom_mask = output["atom_attention_mask"][0].bool()
    atom_names = [
        "".join(chr(int(char) + 32) for char in chars if int(char) != 0).strip()
        for chars in output["ref_atom_name_chars"][0, atom_mask]
    ]
    return {
        "inputs": {
            name: output[name].detach().cpu()
            for name in (
                "res_type",
                "atom_to_token",
                "ref_atom_name_chars",
                "atom_attention_mask",
                "token_attention_mask",
                "residue_index",
            )
        },
        "coordinates": output["sample_atom_coords"][0, atom_mask].detach().float().cpu(),
        "plddt": output["plddt"][0].detach().float().cpu(),
        "pae": output["pae"][0].detach().float().cpu(),
        "pde": output["pde"][0].detach().float().cpu(),
        "ptm": output["ptm"][0].detach().float().cpu(),
        "iptm": output["iptm"][0].detach().float().cpu(),
        "complex_plddt": output["complex_plddt"][0].detach().float().cpu(),
        "complex_iplddt": output["complex_iplddt"][0].detach().float().cpu(),
        "pair_chains_iptm": output["pair_chains_iptm"][0].detach().float().cpu(),
        "atom_names": atom_names,
        # Raw logits (positions/plddt/pae logits + distogram) for reference-side
        # inspection; the OFoldX public result exposes only derived values, so the
        # `compare` step scores the derived quantities above.
        "logits": {
            "distogram_logits": output["distogram_logits"][0].detach().float().cpu(),
            "plddt_logits": output["plddt_logits"][0].detach().float().cpu(),
            "pae_logits": output["pae_logits"][0].detach().float().cpu(),
            "pde_logits": output["pde_logits"][0].detach().float().cpu(),
            "resolved_logits": output["resolved_logits"][0].detach().float().cpu(),
        },
    }


def _comparison(ours: Any, reference: Any) -> dict[str, Any]:
    ours = _tensor(ours)
    reference = _tensor(reference)
    if ours.shape != reference.shape:
        raise ValueError(f"parity shapes differ: OFoldX {tuple(ours.shape)} != reference {tuple(reference.shape)}")
    delta = (ours - reference).abs()
    relative = delta / reference.abs().clamp_min(1e-9)
    return {
        "shape": list(ours.shape),
        "max_absolute_error": float(delta.max()),
        "mean_absolute_error": float(delta.mean()),
        "max_relative_error": float(relative.max()),
    }


def _tensor(value: Any) -> Any:
    import numpy as np
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu()
    return torch.as_tensor(np.asarray(value)).detach().float().cpu()


if __name__ == "__main__":
    main()
