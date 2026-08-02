r"""Train an OFoldX model with the native :class:`ofoldx.runners.Runner`.

The fresh default is a full-sized, randomly initialized Protenix model intended
for GPU training. For Protenix, the native Runner currently optimizes its
built-in EDM coordinate-denoising objective; this is not the complete upstream
Protenix multi-stage total loss. MSA manifests work with MSA-capable processors,
including Protenix, while LigandMPNN and ProteinMPNN do not accept MSA input.
Indexed Protenix MSAs can instead be resolved for every structure by passing
``--seq-to-pdb-index`` together with ``--msa-path``.

Cluster sampling accepts asymmetric-unit CIF/mmCIF inputs only. It eagerly scans
the asymmetric-unit CIFs to form sampling rows, then maps every selected row
back to the full CIF model input. ``--cluster-epoch-size`` is a global draw
count; distributed sampling pads it so every rank receives the same number of
rows. Weighted index rows are cached as versioned OFoldX JSONL, either at
``--cluster-indices-file`` or under the workspace cache directory.

Examples:
  python -m ofoldx.runners.train data/train --epochs 10 --device cuda --precision bf16
  python -m ofoldx.runners.train data/cifs --cluster-file data/clusters.txt --cluster-epoch-size 10000
  python -m ofoldx.runners.train data/msa_manifest.jsonl --steps 100 --device cuda --precision bf16
  python -m ofoldx.runners.train data/cifs --seq-to-pdb-index seq_to_pdb_index.json --msa-path data/mmcif_msa
  python -m ofoldx.runners.train data/train --model ligandmpnn --device cpu --steps 1 --no-checkpoint
  torchrun --standalone --nproc-per-node=2 --module ofoldx.runners.train data/train --steps 1000
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Any, Sequence


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"expected a non-negative integer, got {value!r}")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive number, got {value!r}")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError(f"expected a non-negative number, got {value!r}")
    return parsed


def _json_object(value: str) -> dict[str, Any]:
    r"""Parse an inline JSON object or a path to a UTF-8 JSON object."""

    try:
        if value.lstrip().startswith("{"):
            parsed = json.loads(value)
        else:
            path = Path(value)
            if not path.is_file():
                raise argparse.ArgumentTypeError(
                    f"expected an inline JSON object or an existing JSON file, got {value!r}"
                )
            parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise argparse.ArgumentTypeError(f"could not read JSON object from {value!r}: {error}") from error
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError(f"expected a JSON object, got {type(parsed).__name__}")
    return parsed


def _nonempty_string(value: str) -> str:
    if not value.strip():
        raise argparse.ArgumentTypeError("expected a non-empty string")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train an OFoldX model with the native OFoldX Runner.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python -m ofoldx.runners.train data/train --epochs 10 --val-source data/val --device cuda --precision bf16
  python -m ofoldx.runners.train data/train --eval-source validation=data/val --eval-source benchmark=data/test
  python -m ofoldx.runners.train data/cifs --cluster-file clusters.txt --cluster-epoch-size 10000
  python -m ofoldx.runners.train msa.jsonl --steps 100 --device cuda --precision bf16
  python -m ofoldx.runners.train data/cifs --seq-to-pdb-index seq_to_pdb_index.json --msa-path data/mmcif_msa
  python -m ofoldx.runners.train chain.pdb --model ligandmpnn --device cpu --steps 1 --no-checkpoint
  torchrun --standalone --nproc-per-node=2 --module ofoldx.runners.train data/train --steps 1000

The full-sized fresh Protenix default is intended for GPU training. Its native Runner objective is EDM
coordinate denoising, not the complete upstream Protenix multi-stage total loss. MSA manifests work with
MSA-capable processors including Protenix; LigandMPNN and ProteinMPNN do not accept MSA.
Cluster sampling eagerly indexes asymmetric-unit CIF/mmCIF files, samples ASU chain/interface rows, and
maps each row to the full CIF input. Its epoch size is global and is padded evenly across distributed ranks.
""",
    )
    parser.add_argument("train_source", help="training structure source: file, directory, glob, or manifest")
    parser.add_argument("--val-source", help="optional validation structure source (split name: val)")
    parser.add_argument(
        "--eval-source",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="named evaluation structure source; repeat for multiple splits",
    )
    parser.add_argument(
        "--score-split",
        metavar="NAME",
        help="evaluation split used for best-checkpoint scoring (default: val, or first named split)",
    )
    parser.add_argument(
        "--seq-to-pdb-index",
        metavar="PATH",
        help="JSON sequence-to-MSA-directory index (requires --msa-path)",
    )
    parser.add_argument(
        "--msa-path",
        metavar="DIR",
        help="root directory containing indexed MSA directories (requires --seq-to-pdb-index)",
    )

    model = parser.add_argument_group("model")
    model.add_argument("--model", help="registered architecture ID (default: protenix for a fresh model)")
    model.add_argument("--pretrained", help="local or Hugging Face model repository")
    model.add_argument("--revision", help="pretrained repository branch, tag, or commit")
    model.add_argument("--model-config", type=_json_object, default={}, metavar="JSON_OR_PATH")
    model.add_argument("--processor-config", type=_json_object, default={}, metavar="JSON_OR_PATH")
    ccd = model.add_mutually_exclusive_group()
    ccd.add_argument(
        "--ccd-revision",
        type=_nonempty_string,
        help="CCD repository revision for a fresh Processor (not the pretrained model revision)",
    )
    ccd.add_argument(
        "--ccd-snapshot-path",
        type=_nonempty_string,
        metavar="PATH",
        help="local CCD snapshot path for a fresh Processor",
    )

    boundary = parser.add_mutually_exclusive_group()
    boundary.add_argument("--epochs", type=_positive_int, help="number of training epochs (default: 1)")
    boundary.add_argument("--steps", type=_positive_int, help="number of optimizer steps")

    training = parser.add_argument_group("optimization")
    training.add_argument("--lr", type=_positive_float, default=1e-4)
    training.add_argument("--weight-decay", type=_nonnegative_float, default=0.01)
    training.add_argument("--batch-size", type=_positive_int, default=1)
    training.add_argument("--accum-steps", type=_positive_int, default=1)
    training.add_argument("--workers", type=_nonnegative_int, default=0)
    training.add_argument("--precision", choices=("bf16", "fp16"))
    training.add_argument("--max-grad-norm", type=_positive_float)
    training.add_argument("--seed", type=_nonnegative_int, default=1016)
    training.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=False)
    training.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="enable model activation recomputation to reduce memory; independent of training-state --checkpoint",
    )

    crop = parser.add_argument_group("Protenix training crop")
    crop.add_argument(
        "--max-tokens",
        "--crop-size",
        dest="max_tokens",
        type=_positive_int,
        metavar="N",
        help="maximum training tokens per sample (alias: --crop-size)",
    )
    crop.add_argument(
        "--crop-method-weights",
        nargs=3,
        type=_nonnegative_float,
        metavar=("CONTINUOUS", "SPATIAL", "INTERFACE_SPATIAL"),
        help="Protenix continuous, spatial, and interface-spatial crop weights (requires --max-tokens)",
    )

    scheduler = parser.add_argument_group("learning-rate scheduler")
    scheduler.add_argument("--scheduler", choices=("none", "cosine", "linear"), default="none")
    scheduler.add_argument(
        "--warmup",
        type=_nonnegative_int,
        default=0,
        metavar="OPTIMIZER_STEPS",
        help="linear warmup length in optimizer steps",
    )
    scheduler.add_argument(
        "--final-lr-ratio",
        type=_nonnegative_float,
        default=0.0,
        help="final learning rate divided by initial learning rate",
    )

    runtime = parser.add_argument_group("runtime and output")
    runtime.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    runtime.add_argument("--workspace", default="experiments", help="workspace root directory")
    runtime.add_argument("--run-name", help="optional run name")
    restore = runtime.add_mutually_exclusive_group()
    restore.add_argument("--resume", action="store_true", help="resume the latest checkpoint for this run")
    restore.add_argument("--resume-from", metavar="CHECKPOINT", help="resume from an explicit checkpoint")
    runtime.add_argument("--checkpoint", action=argparse.BooleanOptionalAction, default=True)
    runtime.add_argument("--checkpoint-interval", type=_positive_int)
    runtime.add_argument("--keep-latest", type=_nonnegative_int, default=0)
    runtime.add_argument("--logging", action=argparse.BooleanOptionalAction, default=True)
    runtime.add_argument("--log-interval", type=_positive_int)
    runtime.add_argument(
        "--tensorboard",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="enable TensorBoard metric logging",
    )
    runtime.add_argument("--tensorboard-log-dir", type=_nonempty_string, metavar="PATH")
    runtime.add_argument("--tensorboard-flush-secs", type=_positive_int, metavar="N")
    runtime.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="enable Weights & Biases metric logging",
    )
    runtime.add_argument("--wandb-mode", choices=("online", "offline", "disabled"))
    runtime.add_argument("--wandb-project", type=_nonempty_string)
    runtime.add_argument("--wandb-entity", type=_nonempty_string)
    runtime.add_argument("--wandb-id", type=_nonempty_string)
    runtime.add_argument("--wandb-name", type=_nonempty_string)
    runtime.add_argument("--wandb-group", type=_nonempty_string)
    runtime.add_argument("--wandb-resume", choices=("allow", "must", "never", "auto"))
    runtime.add_argument("--wandb-tag", action="append", type=_nonempty_string, default=[], metavar="TAG")
    runtime.add_argument(
        "--print-config",
        action="store_true",
        help="print JSON without constructing Runner/model/dataset or starting training",
    )

    cluster = parser.add_argument_group("cluster-weighted CIF sampling")
    cluster.add_argument("--cluster-file", help="RCSB entity-cluster file; enables cluster sampling")
    cluster.add_argument(
        "--cluster-indices-file",
        metavar="PATH",
        help="OFoldX JSONL cache for final weighted cluster indices (requires --cluster-file)",
    )
    cluster.add_argument("--cluster-epoch-size", type=_positive_int, help="global samples drawn per epoch")
    cluster.add_argument("--cluster-seed", type=_nonnegative_int)
    cluster.add_argument("--cluster-replacement", action=argparse.BooleanOptionalAction, default=True)
    cluster.add_argument(
        "--cluster-contact-cutoff",
        "--cluster-cutoff",
        dest="cluster_cutoff",
        type=_positive_float,
        default=5.0,
        metavar="ANGSTROMS",
    )
    cluster.add_argument(
        "--cluster-all-chain-pairs",
        "--cluster-all-pairs",
        dest="cluster_all_pairs",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    cluster.add_argument("--cluster-exclude-mol-types", nargs="*", default=["ions"], metavar="TYPE")
    cluster.add_argument("--cluster-beta-chain", type=_nonnegative_float, default=0.5)
    cluster.add_argument("--cluster-beta-interface", type=_nonnegative_float, default=1.0)
    cluster.add_argument("--cluster-alpha-prot", type=_nonnegative_float, default=3.0)
    cluster.add_argument("--cluster-alpha-nuc", type=_nonnegative_float, default=3.0)
    cluster.add_argument("--cluster-alpha-ligand", type=_nonnegative_float, default=1.0)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    r"""Parse and validate command-line arguments without constructing a Runner."""

    parser = _parser()
    args = parser.parse_args(argv)
    named_eval_sources: list[tuple[str, str]] = []
    seen_eval_names: set[str] = set()
    for specification in args.eval_source:
        if "=" not in specification:
            parser.error(f"--eval-source must use NAME=PATH syntax, got {specification!r}")
        name, source = specification.split("=", 1)
        if not name or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", name):
            parser.error(
                f"--eval-source name must match ^[A-Za-z][A-Za-z0-9_-]*$, got {name!r}"
            )
        if name == "train":
            parser.error("--eval-source name 'train' is reserved")
        if not source.strip():
            parser.error(f"--eval-source {name!r} must have a non-empty path")
        if name in seen_eval_names:
            parser.error(f"duplicate evaluation split name: {name!r}")
        seen_eval_names.add(name)
        named_eval_sources.append((name, source))
    if args.val_source is not None and "val" in seen_eval_names:
        parser.error("--val-source conflicts with --eval-source val=PATH")
    args.eval_sources = named_eval_sources
    configured_eval_names = (["val"] if args.val_source is not None else []) + [
        name for name, _ in named_eval_sources
    ]
    if args.score_split is not None and args.score_split not in configured_eval_names:
        if configured_eval_names:
            parser.error(
                f"--score-split must name a configured evaluation split: {', '.join(configured_eval_names)}"
            )
        parser.error("--score-split requires at least one evaluation source")
    if args.score_split is None and configured_eval_names:
        args.score_split = "val" if "val" in configured_eval_names else configured_eval_names[0]
    if args.epochs is None and args.steps is None:
        args.epochs = 1
    if args.revision and not args.pretrained:
        parser.error("--revision requires --pretrained")
    if args.pretrained and args.model_config:
        parser.error("--pretrained cannot be combined with nonempty --model-config")
    if args.pretrained and args.processor_config:
        parser.error("--pretrained cannot be combined with nonempty --processor-config")
    if args.pretrained and (args.ccd_revision is not None or args.ccd_snapshot_path is not None):
        parser.error("--pretrained cannot be combined with explicit CCD locator flags")
    processor_ccd_keys = {"ccd_revision", "ccd_snapshot_path"}.intersection(args.processor_config)
    if processor_ccd_keys and (args.ccd_revision is not None or args.ccd_snapshot_path is not None):
        parser.error("explicit CCD locator flags cannot be combined with CCD keys in --processor-config")
    configured_ccd_revision = args.processor_config.get("ccd_revision")
    configured_ccd_snapshot = args.processor_config.get("ccd_snapshot_path")
    for key, value in (
        ("ccd_revision", configured_ccd_revision),
        ("ccd_snapshot_path", configured_ccd_snapshot),
    ):
        if value is not None and (not isinstance(value, str) or not value.strip()):
            parser.error(f"--processor-config {key} must be a non-empty string or null")
    if configured_ccd_revision is not None and configured_ccd_snapshot is not None:
        parser.error("--processor-config must configure at most one CCD locator")
    if (args.seq_to_pdb_index is None) != (args.msa_path is None):
        parser.error("--seq-to-pdb-index and --msa-path must be provided together")
    if not args.tensorboard and (
        args.tensorboard_log_dir is not None or args.tensorboard_flush_secs is not None
    ):
        parser.error("TensorBoard-specific options require --tensorboard")
    wandb_specific = (
        args.wandb_mode,
        args.wandb_project,
        args.wandb_entity,
        args.wandb_id,
        args.wandb_name,
        args.wandb_group,
        args.wandb_resume,
        args.wandb_tag,
    )
    if not args.wandb and any(wandb_specific):
        parser.error("W&B-specific options require --wandb")
    if args.crop_method_weights is not None and args.max_tokens is None:
        parser.error("--crop-method-weights requires --max-tokens")
    if args.crop_method_weights is not None and not any(args.crop_method_weights):
        parser.error("--crop-method-weights cannot be all zero")
    if args.max_tokens is not None:
        if args.model is not None and args.model != "protenix":
            parser.error("training crop options only support --model protenix")
        if args.crop_method_weights is None:
            args.crop_method_weights = [0.2, 0.4, 0.4] if args.cluster_file else [1.0, 0.0, 0.0]
        if not args.cluster_file and any(args.crop_method_weights[1:]):
            parser.error("positive spatial crop weights require --cluster-file")
    reserved_model_keys = sorted({"name", "pretrained", "revision"}.intersection(args.model_config))
    if reserved_model_keys:
        parser.error(f"--model-config cannot set reserved identity key(s): {', '.join(reserved_model_keys)}")
    cluster_only = [
        args.cluster_indices_file,
        args.cluster_epoch_size,
        args.cluster_seed,
        args.cluster_replacement is not True,
        args.cluster_cutoff != 5.0,
        args.cluster_all_pairs,
        args.cluster_exclude_mol_types != ["ions"],
        args.cluster_beta_chain != 0.5,
        args.cluster_beta_interface != 1.0,
        args.cluster_alpha_prot != 3.0,
        args.cluster_alpha_nuc != 3.0,
        args.cluster_alpha_ligand != 1.0,
    ]
    if not args.cluster_file and any(value is not None and value is not False for value in cluster_only):
        parser.error("cluster sampling options require --cluster-file")
    return args


def build_config(args: argparse.Namespace) -> dict[str, Any]:
    r"""Build the plain mapping consumed by :class:`ofoldx.runners.Runner`."""

    model_name = args.model
    if not args.pretrained and model_name is None:
        model_name = "protenix"
    model: dict[str, Any] = {
        "name": model_name or "",
        "pretrained": args.pretrained,
        "revision": args.revision,
        **args.model_config,
    }
    train: dict[str, Any] = {"source": args.train_source}
    if args.max_tokens is not None:
        train["max_tokens"] = args.max_tokens
        train["crop_method_weights"] = args.crop_method_weights
    if args.seq_to_pdb_index is not None:
        train["seq_to_pdb_index"] = args.seq_to_pdb_index
        train["msa_path"] = args.msa_path
    if args.cluster_file:
        train["cluster_sampling"] = {
            "enabled": True,
            "cluster_file": args.cluster_file,
            "indices_file": args.cluster_indices_file,
            "epoch_size": args.cluster_epoch_size,
            "replacement": args.cluster_replacement,
            "contact_cutoff": args.cluster_cutoff,
            "include_all_chain_pairs": args.cluster_all_pairs,
            "exclude_mol_types": args.cluster_exclude_mol_types,
            "seed": args.cluster_seed,
            "beta": {"chain": args.cluster_beta_chain, "interface": args.cluster_beta_interface},
            "alpha": {
                "prot": args.cluster_alpha_prot,
                "nuc": args.cluster_alpha_nuc,
                "ligand": args.cluster_alpha_ligand,
            },
        }
    data: dict[str, Any] = {"train": train}
    evaluate_splits: list[str] = []
    eval_sources = ([("val", args.val_source)] if args.val_source is not None else []) + args.eval_sources
    for name, source in eval_sources:
        data[name] = {"source": source}
        if args.seq_to_pdb_index is not None:
            data[name]["seq_to_pdb_index"] = args.seq_to_pdb_index
            data[name]["msa_path"] = args.msa_path
        evaluate_splits.append(name)

    processor = dict(args.processor_config)
    if args.ccd_revision is not None:
        processor["ccd_revision"] = args.ccd_revision
    elif args.ccd_snapshot_path is not None:
        processor["ccd_snapshot_path"] = args.ccd_snapshot_path

    config: dict[str, Any] = {
        "stack": "ddp",
        "model": model,
        "processor": processor,
        "seed": args.seed,
        "deterministic": args.deterministic,
        "gradient_checkpointing": args.gradient_checkpointing,
        "accum_steps": args.accum_steps,
        "train_splits": ["train"],
        "evaluate_splits": evaluate_splits,
        "data": data,
        "optim": {"type": "torch_adamw", "lr": args.lr, "weight_decay": args.weight_decay},
        "dataloader": {"batch_size": args.batch_size, "num_workers": args.workers, "drop_last": False},
        "logging": {"enabled": args.logging, "interval": args.log_interval},
        "ckpt": {
            "enabled": args.checkpoint,
            "interval": args.checkpoint_interval,
            "keep_latest_k": args.keep_latest,
        },
    }
    run_name = args.run_name or model_name or "pretrained"
    config["name"] = run_name
    config["workspace"] = {"root": args.workspace, "lineage": "ofoldx", "experiment": run_name}
    if args.steps is not None:
        config["steps"] = args.steps
    else:
        config["epochs"] = args.epochs
    if args.precision is not None:
        config["precision"] = args.precision
    if args.max_grad_norm is not None:
        config["max_grad_norm"] = args.max_grad_norm
    if evaluate_splits:
        config["score"] = {"split": args.score_split, "metric": "loss"}
    if args.scheduler != "none":
        config["sched"] = {
            "type": args.scheduler,
            "interval": "step",
            "warmup_steps": args.warmup,
            "final_lr_ratio": args.final_lr_ratio,
        }
    if args.tensorboard:
        tensorboard: dict[str, Any] = {"enabled": True}
        if args.tensorboard_log_dir is not None:
            tensorboard["log_dir"] = args.tensorboard_log_dir
        if args.tensorboard_flush_secs is not None:
            tensorboard["flush_secs"] = args.tensorboard_flush_secs
        config["tensorboard"] = tensorboard
    if args.wandb:
        wandb: dict[str, Any] = {"enabled": True}
        for argument, key in (
            (args.wandb_mode, "mode"),
            (args.wandb_project, "project"),
            (args.wandb_entity, "entity"),
            (args.wandb_id, "id"),
            (args.wandb_name, "name"),
            (args.wandb_group, "group"),
            (args.wandb_resume, "resume"),
        ):
            if argument is not None:
                wandb[key] = argument
        if args.wandb_tag:
            wandb["tags"] = args.wandb_tag
        config["wandb"] = wandb
    if args.resume:
        config["resume"] = True
    if args.resume_from:
        config["checkpoint"] = args.resume_from
    return config


def _distributed_fail_fast() -> None:
    r"""Report the active exception and terminate without distributed teardown."""

    traceback.print_exc(file=sys.stderr)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)


def main(argv: Sequence[str] | None = None) -> int:
    r"""Run training, returning zero after a completed Runner lifecycle."""

    args = parse_args(argv)
    config = build_config(args)
    if args.print_config:
        print(json.dumps(config, indent=2, sort_keys=True))
        return 0
    if not args.train_source.strip():
        raise ValueError("training source must not be empty")
    for name in config["evaluate_splits"]:
        if not config["data"][name]["source"].strip():
            raise ValueError(f"evaluation source {name!r} must not be empty")

    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available")
    resolved_device = "cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu"

    from .runner import Runner

    runner = Runner(config)
    close_runner = True
    try:
        runner.unwrap(runner.model).set_backend("oops")  # NOTE
        if len(runner.datasets["train"]) == 0:
            raise ValueError(f"training source {args.train_source!r} produced an empty dataset")
        for name in config["evaluate_splits"]:
            if len(runner.datasets[name]) == 0:
                source = config["data"][name]["source"]
                raise ValueError(f"evaluation source {name!r} ({source!r}) produced an empty dataset")
        if runner.rank == 0:
            actual_device = getattr(runner, "device", resolved_device)
            print(f"OFoldX training on {actual_device}; workspace: {runner.workspace.dir}")
            train_dataset = runner.datasets["train"]
            cache_path = getattr(train_dataset, "cluster_indices_file", None)
            if cache_path is not None:
                print(
                    f"Cluster indices cache: {cache_path} "
                    f"({getattr(train_dataset, 'cluster_indices_status', 'unknown')})"
                )
        runner.train()
    except Exception:
        if bool(getattr(runner, "distributed", False)):
            close_runner = False
            _distributed_fail_fast()
        raise
    finally:
        if close_runner:
            runner.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
