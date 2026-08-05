r"""Train an OFoldX model with the native :class:`ofoldx.runners.Runner`.

The fresh default is a full-sized, randomly initialized Protenix model intended
for GPU training. Its default ``protenix_baseline_v1`` objective recipe selects
the versioned baseline-aligned training forward and aggregate objective;
``ofoldx_edm_v1`` retains the historical coordinate-denoising recipe. MSA
manifests work with MSA-capable processors,
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
import copy
import json
import math
import os
import re
import sys
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Sequence

import yaml


_YAML_BOOLEAN_TAG = "tag:yaml.org,2002:bool"
_YAML_FLOAT_TAG = "tag:yaml.org,2002:float"
_YAML_INTEGER_TAG = "tag:yaml.org,2002:int"
_YAML_MERGE_TAG = "tag:yaml.org,2002:merge"
_YAML_NULL_TAG = "tag:yaml.org,2002:null"
_YAML_INTERPOLATION = re.compile(r"\$\{")
_JSON_INTEGER = re.compile(r"^-?(?:0|[1-9][0-9]*)$")
_JSON_FLOAT = re.compile(
    r"^-?(?:(?:0|[1-9][0-9]*)\.[0-9]+(?:[eE][-+]?[0-9]+)?|"
    r"(?:0|[1-9][0-9]*)[eE][-+]?[0-9]+)$"
)
_PROTENIX_ARCHITECTURES = frozenset(
    {"protenix", "protenix_template", "protenix_structure_condition", "protenix_constraint"}
)
_OBJECTIVE_RECIPES = ("protenix_baseline_v1", "ofoldx_edm_v1")


class _LaunchConfigLoader(yaml.SafeLoader):
    r"""Safe YAML loader with JSON-like scalars, unique keys, and no aliases."""

    yaml_implicit_resolvers: dict[Any, list[Any]] = {}

    def compose_node(self, parent: Any, index: Any) -> yaml.Node:
        if self.check_event(yaml.AliasEvent):
            event = self.get_event()
            raise yaml.constructor.ConstructorError(
                None,
                None,
                "YAML aliases are not allowed in launch configs",
                event.start_mark,
            )
        event = self.peek_event()
        if isinstance(
            event,
            (yaml.ScalarEvent, yaml.MappingStartEvent, yaml.SequenceStartEvent),
        ) and event.tag is not None:
            raise yaml.constructor.ConstructorError(
                None,
                None,
                "explicit YAML tags are not allowed in launch configs",
                event.start_mark,
            )
        return super().compose_node(parent, index)

    def construct_mapping(self, node: yaml.Node, deep: bool = False) -> dict[Any, Any]:
        if not isinstance(node, yaml.MappingNode):
            raise yaml.constructor.ConstructorError(
                None,
                None,
                f"expected a mapping node, got {node.id}",
                node.start_mark,
            )
        for key_node, _ in node.value:
            if key_node.tag == _YAML_MERGE_TAG or (
                isinstance(key_node, yaml.ScalarNode)
                and key_node.style is None
                and key_node.value == "<<"
            ):
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    "YAML merge keys are not allowed in launch configs",
                    key_node.start_mark,
                )
        self.flatten_mapping(node)
        mapping: dict[Any, Any] = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                duplicate = key in mapping
            except TypeError as error:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    "found an unhashable mapping key",
                    key_node.start_mark,
                ) from error
            if duplicate:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"found duplicate mapping key {key!r}",
                    key_node.start_mark,
                )
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


_LaunchConfigLoader.add_implicit_resolver(
    _YAML_BOOLEAN_TAG,
    re.compile(r"^(?:true|false)$"),
    list("tf"),
)
_LaunchConfigLoader.add_implicit_resolver(
    _YAML_INTEGER_TAG,
    _JSON_INTEGER,
    list("-0123456789"),
)
_LaunchConfigLoader.add_implicit_resolver(
    _YAML_FLOAT_TAG,
    _JSON_FLOAT,
    list("-0123456789"),
)
_LaunchConfigLoader.add_implicit_resolver(
    _YAML_NULL_TAG,
    re.compile(r"^null$"),
    ["n"],
)


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


def _is_positive_finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value)) and value > 0
    except (OverflowError, TypeError, ValueError):
        return False


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError(f"expected a non-negative number, got {value!r}")
    return parsed


def _ema_decay(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 < parsed < 1.0:
        raise argparse.ArgumentTypeError(
            f"expected an EMA decay strictly between 0 and 1, got {value!r}"
        )
    return parsed


def _json_object(value: str) -> dict[str, Any]:
    r"""Parse an inline JSON object or a path to a UTF-8 JSON object."""

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        parsed_object: dict[str, Any] = {}
        for key, item in pairs:
            if key in parsed_object:
                raise ValueError(f"duplicate JSON object key: {key!r}")
            parsed_object[key] = item
        return parsed_object

    try:
        if value.lstrip().startswith("{"):
            parsed = json.loads(value, object_pairs_hook=reject_duplicate_keys)
        else:
            path = Path(value).expanduser()
            if not path.is_file():
                raise argparse.ArgumentTypeError(
                    f"expected an inline JSON object or an existing JSON file, got {value!r}"
                )
            parsed = json.loads(
                path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate_keys
            )
    except (OSError, ValueError) as error:
        raise argparse.ArgumentTypeError(f"could not read JSON object from {value!r}: {error}") from error
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError(f"expected a JSON object, got {type(parsed).__name__}")
    return parsed


def _validate_mixture_cluster_sampling(
    parser: argparse.ArgumentParser,
    component_name: str,
    cluster_sampling: Any,
) -> None:
    r"""Validate the complete per-component cluster-sampling CLI schema."""

    prefix = f"--train-datasets component {component_name!r} cluster_sampling"
    if not isinstance(cluster_sampling, dict):
        parser.error(f"{prefix} must be a JSON object")
    allowed_keys = {
        "enabled",
        "cluster_file",
        "indices_file",
        "epoch_size",
        "replacement",
        "contact_cutoff",
        "include_all_chain_pairs",
        "exclude_mol_types",
        "seed",
        "beta",
        "alpha",
    }
    unknown_keys = sorted(set(cluster_sampling).difference(allowed_keys))
    if unknown_keys:
        parser.error(f"{prefix} has unknown key(s): {', '.join(unknown_keys)}")

    enabled = cluster_sampling.get("enabled", False)
    if not isinstance(enabled, bool):
        parser.error(f"{prefix}.enabled must be a boolean")
    for key in ("cluster_file", "indices_file"):
        value = cluster_sampling.get(key)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            parser.error(f"{prefix}.{key} must be a non-empty string or null")
    if enabled and not cluster_sampling.get("cluster_file"):
        parser.error(f"{prefix}.cluster_file must be a non-empty string when enabled")

    for key in ("epoch_size", "seed"):
        if cluster_sampling.get(key) is not None:
            parser.error(f"{prefix}.{key} must be null or omitted for a dataset mixture")
    replacement = cluster_sampling.get("replacement", True)
    if replacement is not True:
        parser.error(f"{prefix}.replacement must be true for a dataset mixture")

    cutoff = cluster_sampling.get("contact_cutoff", 5.0)
    if (
        isinstance(cutoff, bool)
        or not isinstance(cutoff, (int, float))
        or not math.isfinite(float(cutoff))
        or cutoff <= 0
    ):
        parser.error(f"{prefix}.contact_cutoff must be a positive finite number")
    include_all_pairs = cluster_sampling.get("include_all_chain_pairs", False)
    if not isinstance(include_all_pairs, bool):
        parser.error(f"{prefix}.include_all_chain_pairs must be a boolean")
    excluded = cluster_sampling.get("exclude_mol_types", ["ions"])
    if not isinstance(excluded, list) or any(
        not isinstance(item, str) or not item.strip() for item in excluded
    ):
        parser.error(f"{prefix}.exclude_mol_types must be a list of non-empty strings")

    nested_schemas = {
        "beta": ({"chain", "interface"}, {"chain": 0.5, "interface": 1.0}),
        "alpha": ({"prot", "nuc", "ligand"}, {"prot": 3.0, "nuc": 3.0, "ligand": 1.0}),
    }
    for key, (allowed_nested_keys, defaults) in nested_schemas.items():
        nested = cluster_sampling.get(key, defaults)
        if not isinstance(nested, dict):
            parser.error(f"{prefix}.{key} must be a JSON object")
        unknown_nested = sorted(set(nested).difference(allowed_nested_keys))
        if unknown_nested:
            parser.error(
                f"{prefix}.{key} has unknown key(s): {', '.join(unknown_nested)}"
            )
        for nested_key, value in nested.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value < 0
            ):
                parser.error(
                    f"{prefix}.{key}.{nested_key} must be a non-negative finite number"
                )


def _nonempty_string(value: str) -> str:
    if not value.strip():
        raise argparse.ArgumentTypeError("expected a non-empty string")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train an OFoldX model with the native OFoldX Runner.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
        epilog="""\
Examples:
  python -m ofoldx.runners.train --config configs/train.yaml
  python -m ofoldx.runners.train data/train --epochs 10 --val-source data/val --device cuda --precision bf16
  python -m ofoldx.runners.train data/train --eval-source validation=data/val --eval-source benchmark=data/test
  python -m ofoldx.runners.train data/cifs --cluster-file clusters.txt --cluster-epoch-size 10000
  python -m ofoldx.runners.train msa.jsonl --steps 100 --device cuda --precision bf16
  python -m ofoldx.runners.train data/cifs --seq-to-pdb-index seq_to_pdb_index.json --msa-path data/mmcif_msa
  python -m ofoldx.runners.train chain.pdb --model ligandmpnn --device cpu --steps 1 --no-checkpoint
  torchrun --standalone --nproc-per-node=2 --module ofoldx.runners.train data/train --steps 1000

The full-sized fresh Protenix default is intended for GPU training. It selects the versioned
protenix_baseline_v1 training recipe with S=48 and C=4 unless overridden; ofoldx_edm_v1 retains the
historical EDM-only objective. MSA manifests work with MSA-capable processors including Protenix;
LigandMPNN and ProteinMPNN do not accept MSA.
Cluster sampling eagerly indexes asymmetric-unit CIF/mmCIF files, samples ASU chain/interface rows, and
maps each row to the full CIF input. Its epoch size is global and is padded evenly across distributed ranks.

--config is exclusive with training arguments; only --print-config and -h/--help may accompany it.
Help is displayed without opening or parsing YAML_PATH.
""",
    )
    parser.add_argument(
        "-c",
        "--config",
        metavar="YAML_PATH",
        help="load training arguments from a YAML mapping keyed by argparse destination",
    )
    parser.add_argument(
        "train_source",
        nargs="?",
        help="training structure source: file, directory, glob, or manifest",
    )
    parser.add_argument(
        "--train-datasets",
        type=_json_object,
        metavar="JSON_OR_PATH",
        help="named training dataset mixture as an inline JSON object or JSON file",
    )
    parser.add_argument(
        "--train-epoch-size",
        type=_positive_int,
        metavar="N",
        help="number of samples drawn per mixed-dataset training epoch",
    )
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
        type=_nonempty_string,
        metavar="PATH",
        help="JSON sequence-to-MSA-directory index (requires --msa-path)",
    )
    parser.add_argument(
        "--msa-path",
        type=_nonempty_string,
        metavar="DIR",
        help="root directory containing indexed MSA directories (requires --seq-to-pdb-index)",
    )

    model = parser.add_argument_group("model")
    model.add_argument("--model", help="registered architecture ID (default: protenix for a fresh model)")
    model.add_argument("--pretrained", help="local or Hugging Face model repository")
    model.add_argument("--revision", help="pretrained repository branch, tag, or commit")
    model.add_argument(
        "--objective-recipe",
        choices=_OBJECTIVE_RECIPES,
        help=(
            "versioned objective contract (known Protenix defaults to protenix_baseline_v1; "
            "known non-Protenix models omit ofoldx_edm_v1 and reject the Protenix baseline; "
            "an explicit value is retained while a pretrained architecture is unresolved)"
        ),
    )
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
    training.add_argument(
        "--diffusion-batch-size",
        type=_positive_int,
        metavar="S",
        help="independent diffusion corruptions drawn per Protenix input structure",
    )
    training.add_argument(
        "--diffusion-chunk-size",
        type=_positive_int,
        metavar="S_CHUNK",
        help="maximum Protenix diffusion samples evaluated in one denoiser call",
    )
    training.add_argument("--accum-steps", type=_positive_int, default=1)
    training.add_argument("--workers", type=_nonnegative_int, default=0)
    training.add_argument("--precision", choices=("bf16", "fp16"))
    training.add_argument("--max-grad-norm", type=_positive_float)
    training.add_argument(
        "--ema-decay",
        type=_ema_decay,
        metavar="FLOAT",
        help="enable a model-parameter EMA with a finite decay strictly between 0 and 1",
    )
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


def _yaml_value_location(parent: str, key: str | int) -> str:
    if isinstance(key, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
        return f"{parent}.{key}"
    return f"{parent}[{key!r}]"


def _validate_yaml_tree(value: Any, location: str = "config") -> None:
    r"""Require a finite JSON-compatible tree with string mapping keys."""

    if isinstance(value, str):
        if _YAML_INTERPOLATION.search(value):
            raise ValueError(f"{location} contains a forbidden interpolation marker")
        return
    if value is None or isinstance(value, bool) or isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{location} must be finite")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_yaml_tree(item, _yaml_value_location(location, index))
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(
                    f"{location} has non-string mapping key {key!r} "
                    f"({type(key).__name__})"
                )
            if _YAML_INTERPOLATION.search(key):
                raise ValueError(
                    f"{location} has a mapping key containing a forbidden interpolation marker"
                )
            _validate_yaml_tree(item, _yaml_value_location(location, key))
        return
    raise ValueError(
        f"{location} contains a non-JSON-compatible {type(value).__name__} value"
    )


def _load_launch_config(parser: argparse.ArgumentParser, value: str) -> dict[str, Any]:
    try:
        path = Path(value).expanduser()
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        parser.error(f"could not resolve YAML config path {value!r}: {error}")
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, RuntimeError, UnicodeError, ValueError) as error:
        parser.error(f"could not read YAML config {path}: {error}")
    try:
        config = yaml.load(text, Loader=_LaunchConfigLoader)
    except yaml.YAMLError as error:
        parser.error(f"could not parse YAML config {path}: {error}")
    if not isinstance(config, Mapping):
        parser.error(
            f"YAML config {path} must contain one mapping document, "
            f"got {type(config).__name__}"
        )
    try:
        _validate_yaml_tree(config)
    except ValueError as error:
        parser.error(f"invalid YAML config {path}: {error}")
    return dict(config)


def _yaml_action_option(action: argparse.Action, *, enabled: bool = True) -> str:
    candidates = [option for option in action.option_strings if option.startswith("--")]
    if not enabled:
        candidates = [option for option in candidates if option.startswith("--no-")]
    else:
        candidates = [option for option in candidates if not option.startswith("--no-")]
    if not candidates:
        raise RuntimeError(f"no matching long option for argparse destination {action.dest!r}")
    return candidates[0]


def _yaml_scalar_to_cli(
    parser: argparse.ArgumentParser,
    action: argparse.Action,
    key: str,
    value: Any,
) -> str:
    if action.type in (_positive_int, _nonnegative_int):
        if isinstance(value, bool) or not isinstance(value, int):
            parser.error(f"YAML key {key!r} must be an integer, got {type(value).__name__}")
        return str(value)
    if action.type in (_positive_float, _nonnegative_float, _ema_decay):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            parser.error(f"YAML key {key!r} must be a number, got {type(value).__name__}")
        return str(value)
    if action.type is _json_object:
        if not isinstance(value, Mapping):
            parser.error(f"YAML key {key!r} must be a native mapping")
        try:
            return json.dumps(value, allow_nan=False, separators=(",", ":"))
        except (TypeError, ValueError) as error:
            parser.error(f"YAML key {key!r} could not be serialized as JSON: {error}")
    if not isinstance(value, str):
        parser.error(f"YAML key {key!r} must be a string, got {type(value).__name__}")
    return value


def _launch_config_to_argv(
    parser: argparse.ArgumentParser,
    config: Mapping[str, Any],
) -> list[str]:
    actions = {
        action.dest: action
        for action in parser._actions
        if action.dest not in {"help", "config", "print_config"}
    }
    unknown_keys = sorted(set(config).difference(actions))
    if unknown_keys:
        parser.error(f"unknown YAML config key(s): {', '.join(unknown_keys)}")

    optional_argv: list[str] = []
    positional_argv: list[str] = []
    for key, value in config.items():
        action = actions[key]
        if value is None:
            if action.default is not None:
                parser.error(f"YAML key {key!r} may not be null")
            continue
        if action.nargs == 0:
            if not isinstance(value, bool):
                parser.error(f"YAML key {key!r} must be true or false")
            if isinstance(action, argparse.BooleanOptionalAction):
                optional_argv.append(_yaml_action_option(action, enabled=value))
            elif value != action.default:
                if value != action.const:
                    parser.error(f"YAML key {key!r} cannot represent {value!r}")
                optional_argv.append(_yaml_action_option(action))
            continue
        if key in {"eval_source", "wandb_tag"}:
            if not isinstance(value, list):
                parser.error(f"YAML key {key!r} must be a list")
            option = _yaml_action_option(action)
            for item in value:
                cli_value = _yaml_scalar_to_cli(parser, action, key, item)
                optional_argv.append(f"{option}={cli_value}")
            continue
        if action.nargs not in (None, "?"):
            if not isinstance(value, list):
                parser.error(f"YAML key {key!r} must be a list")
            option = _yaml_action_option(action)
            optional_argv.append(option)
            optional_argv.extend(
                _yaml_scalar_to_cli(parser, action, key, item) for item in value
            )
            continue
        cli_value = _yaml_scalar_to_cli(parser, action, key, value)
        if action.option_strings:
            optional_argv.append(f"{_yaml_action_option(action)}={cli_value}")
        else:
            positional_argv.append(cli_value)
    if positional_argv:
        optional_argv.append("--")
        optional_argv.extend(positional_argv)
    return optional_argv


def _has_config_option(argv: Sequence[str]) -> bool:
    for value in argv:
        if value == "--":
            return False
        if value in {"-c", "--config"} or value.startswith("--config="):
            return True
        if value.startswith("-c") and not value.startswith("--"):
            return True
    return False


def _expand_config_argv(
    parser: argparse.ArgumentParser,
    raw_argv: list[str],
) -> list[str]:
    if not _has_config_option(raw_argv):
        return raw_argv
    controls = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    controls.add_argument("-c", "--config", action="append")
    controls.add_argument("--print-config", action="store_true")
    controls.add_argument("-h", "--help", action="store_true")
    control_args, other_args = controls.parse_known_args(raw_argv)
    config_values = control_args.config or []
    if len(config_values) != 1:
        parser.error("exactly one -c/--config option must be provided")
    if other_args:
        parser.error(
            "--config cannot be combined with other training arguments: "
            + " ".join(other_args)
        )
    if control_args.help:
        parser.parse_args(["--help"])
    config = _load_launch_config(parser, config_values[0])
    config_argv = _launch_config_to_argv(parser, config)
    if control_args.print_config:
        try:
            separator_index = config_argv.index("--")
        except ValueError:
            config_argv.append("--print-config")
        else:
            config_argv.insert(separator_index, "--print-config")
    return [f"--config={config_values[0]}", *config_argv]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    r"""Parse and validate command-line arguments without constructing a Runner."""

    parser = _parser()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    raw_argv = _expand_config_argv(parser, raw_argv)
    args = parser.parse_args(raw_argv)
    if (args.train_source is None) == (args.train_datasets is None):
        parser.error("exactly one of train_source or --train-datasets must be provided")
    if args.train_source is not None and not args.train_source.strip():
        parser.error("train_source must be a non-empty path")
    if args.train_epoch_size is not None and args.train_datasets is None:
        parser.error("--train-epoch-size requires --train-datasets")
    legacy_cluster_options = [
        value.split("=", 1)[0]
        for value in raw_argv
        if value.startswith(("--cluster-", "--no-cluster-"))
    ]
    if args.train_datasets is not None and legacy_cluster_options:
        parser.error(
            "legacy --cluster-* options are ambiguous with --train-datasets; "
            "configure cluster_sampling inside each component"
        )
    if args.train_datasets is not None:
        if not args.train_datasets:
            parser.error("--train-datasets must contain at least one named component")
        normalized_datasets: dict[str, dict[str, Any]] = {}
        indices_file_owners: dict[Path, str] = {}
        allowed_component_keys = {
            "source",
            "sampling_ratio",
            "seq_to_pdb_index",
            "msa_path",
            "processing_timeout_seconds",
            "max_consecutive_processing_errors",
            "cluster_sampling",
        }
        for name, raw_spec in args.train_datasets.items():
            if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", name):
                parser.error(
                    "--train-datasets component names must match "
                    f"^[A-Za-z][A-Za-z0-9_-]*$, got {name!r}"
                )
            if not isinstance(raw_spec, dict):
                parser.error(f"--train-datasets component {name!r} must be a JSON object")
            unknown_keys = sorted(set(raw_spec).difference(allowed_component_keys))
            if unknown_keys:
                parser.error(
                    f"--train-datasets component {name!r} has unknown key(s): {', '.join(unknown_keys)}"
                )
            source = raw_spec.get("source")
            if not isinstance(source, str) or not source.strip():
                parser.error(f"--train-datasets component {name!r} source must be a non-empty string")
            ratio = raw_spec.get("sampling_ratio", 1)
            if not _is_positive_finite_number(ratio):
                parser.error(
                    f"--train-datasets component {name!r} sampling_ratio must be a positive finite number"
                )
            processing_timeout_seconds = raw_spec.get("processing_timeout_seconds")
            if "processing_timeout_seconds" in raw_spec and processing_timeout_seconds is not None:
                if not _is_positive_finite_number(processing_timeout_seconds):
                    parser.error(
                        f"--train-datasets component {name!r} processing_timeout_seconds "
                        "must be a positive finite number or null"
                    )
            max_consecutive_processing_errors = raw_spec.get("max_consecutive_processing_errors")
            if "max_consecutive_processing_errors" in raw_spec and max_consecutive_processing_errors is not None:
                if (
                    isinstance(max_consecutive_processing_errors, bool)
                    or not isinstance(max_consecutive_processing_errors, int)
                    or max_consecutive_processing_errors <= 0
                ):
                    parser.error(
                        f"--train-datasets component {name!r} max_consecutive_processing_errors "
                        "must be a positive integer or null"
                    )
            has_local_index = "seq_to_pdb_index" in raw_spec
            has_local_msa = "msa_path" in raw_spec
            if has_local_index != has_local_msa:
                parser.error(
                    f"--train-datasets component {name!r} must provide seq_to_pdb_index and msa_path together"
                )
            if has_local_index:
                for key in ("seq_to_pdb_index", "msa_path"):
                    value = raw_spec[key]
                    if not isinstance(value, str) or not value.strip():
                        parser.error(
                            f"--train-datasets component {name!r} {key} must be a non-empty string"
                        )
            cluster_sampling = raw_spec.get("cluster_sampling")
            if "cluster_sampling" in raw_spec:
                _validate_mixture_cluster_sampling(parser, name, cluster_sampling)
            spec = copy.deepcopy(raw_spec)
            spec["sampling_ratio"] = ratio
            if cluster_sampling is not None:
                indices_file = cluster_sampling.get("indices_file")
                if indices_file is not None:
                    try:
                        resolved_indices_file = Path(indices_file).expanduser().resolve()
                    except (OSError, RuntimeError) as error:
                        parser.error(
                            f"--train-datasets component {name!r} cluster_sampling.indices_file "
                            f"could not be resolved: {error}"
                        )
                    previous_owner = indices_file_owners.get(resolved_indices_file)
                    if previous_owner is not None:
                        parser.error(
                            "--train-datasets components "
                            f"{previous_owner!r} and {name!r} resolve cluster_sampling.indices_file "
                            f"to the same path: {resolved_indices_file}"
                        )
                    indices_file_owners[resolved_indices_file] = name
                    spec["cluster_sampling"]["indices_file"] = str(resolved_indices_file)
            if not has_local_index and args.seq_to_pdb_index is not None:
                spec["seq_to_pdb_index"] = args.seq_to_pdb_index
                spec["msa_path"] = args.msa_path
            normalized_datasets[name] = spec
        args.train_datasets = normalized_datasets
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
            all_clustered = bool(args.cluster_file)
            if args.train_datasets is not None:
                all_clustered = all(
                    component.get("cluster_sampling", {}).get("enabled") is True
                    for component in args.train_datasets.values()
                )
            args.crop_method_weights = [0.2, 0.4, 0.4] if all_clustered else [1.0, 0.0, 0.0]
        if any(args.crop_method_weights[1:]):
            if args.train_datasets is None:
                if not args.cluster_file:
                    parser.error("positive spatial crop weights require --cluster-file")
            elif not all(
                component.get("cluster_sampling", {}).get("enabled") is True
                for component in args.train_datasets.values()
            ):
                parser.error(
                    "positive spatial crop weights require cluster sampling to be enabled for every "
                    "--train-datasets component"
                )
    if args.diffusion_batch_size is not None or args.diffusion_chunk_size is not None:
        if args.model is not None and args.model not in _PROTENIX_ARCHITECTURES:
            parser.error("diffusion batch options only support Protenix architectures")
    if (
        args.objective_recipe == "protenix_baseline_v1"
        and args.model is not None
        and args.model not in _PROTENIX_ARCHITECTURES
    ):
        parser.error("--objective-recipe protenix_baseline_v1 only supports Protenix architectures")
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
    objective_recipe = args.objective_recipe
    if model_name in _PROTENIX_ARCHITECTURES:
        objective_recipe = objective_recipe or "protenix_baseline_v1"
    elif model_name is not None and objective_recipe == "ofoldx_edm_v1":
        objective_recipe = None
    model: dict[str, Any] = {
        "name": model_name or "",
        "pretrained": args.pretrained,
        "revision": args.revision,
        **args.model_config,
    }
    if args.train_datasets is not None:
        datasets = copy.deepcopy(args.train_datasets)
        if args.max_tokens is not None:
            for component in datasets.values():
                component["max_tokens"] = args.max_tokens
                component["crop_method_weights"] = list(args.crop_method_weights)
        train: dict[str, Any] = {
            "datasets": datasets,
            "epoch_size": args.train_epoch_size,
            "replacement": True,
            "seed": args.seed,
        }
    else:
        train = {"source": args.train_source}
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
            "async_mode": "disabled",
        },
    }
    if objective_recipe is not None:
        config["objective_recipe"] = objective_recipe
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
    if args.ema_decay is not None:
        config["ema_decay"] = args.ema_decay
    if args.diffusion_batch_size is not None:
        config["diffusion_batch_size"] = args.diffusion_batch_size
    if args.diffusion_chunk_size is not None:
        config["diffusion_chunk_size"] = args.diffusion_chunk_size
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


def _cluster_cache_diagnostics(dataset: Any) -> list[tuple[str | None, Any, str]]:
    r"""Return cluster-index cache details for a dataset or mixed dataset."""

    diagnostics: list[tuple[str | None, Any, str]] = []
    cache_path = getattr(dataset, "cluster_indices_file", None)
    if cache_path is not None:
        diagnostics.append((None, cache_path, getattr(dataset, "cluster_indices_status", "unknown")))
    components = getattr(dataset, "datasets", None)
    if components is None:
        components = getattr(dataset, "components", None)
    if isinstance(components, Mapping):
        component_items = components.items()
    elif components is not None and not isinstance(components, (str, bytes)):
        component_items = enumerate(components)
    else:
        component_items = ()
    for name, component in component_items:
        component_path = getattr(component, "cluster_indices_file", None)
        if component_path is not None:
            diagnostics.append(
                (str(name), component_path, getattr(component, "cluster_indices_status", "unknown"))
            )
    return diagnostics


def main(argv: Sequence[str] | None = None) -> int:
    r"""Run training, returning zero after a completed Runner lifecycle."""

    args = parse_args(argv)
    config = build_config(args)
    if args.print_config:
        print(json.dumps(config, indent=2, sort_keys=True))
        return 0
    if args.train_source is not None and not args.train_source.strip():
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
        runner.set_backend("oops")
        if len(runner.datasets["train"]) == 0:
            if args.train_datasets is not None:
                names = ", ".join(args.train_datasets)
                raise ValueError(f"training dataset mixture ({names}) produced an empty dataset")
            raise ValueError(f"training source {args.train_source!r} produced an empty dataset")
        for name in config["evaluate_splits"]:
            if len(runner.datasets[name]) == 0:
                source = config["data"][name]["source"]
                raise ValueError(f"evaluation source {name!r} ({source!r}) produced an empty dataset")
        if runner.rank == 0:
            actual_device = getattr(runner, "device", resolved_device)
            print(f"OFoldX training on {actual_device}; workspace: {runner.workspace.dir}")
            train_dataset = runner.datasets["train"]
            for component_name, cache_path, cache_status in _cluster_cache_diagnostics(train_dataset):
                component_label = f" ({component_name})" if component_name is not None else ""
                print(
                    f"Cluster indices cache{component_label}: {cache_path} "
                    f"({cache_status})"
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
