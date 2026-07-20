"""Inspect PyTorch checkpoint state dictionaries.

This is a development utility for understanding external checkpoints. It does
not load weights into OFoldX models.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

GB = 1024**3
DEFAULT_DEPTHS = (1, 2, 3)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Path to a .pt checkpoint")
    parser.add_argument("--strip-prefix", default="module.", help="Prefix to strip from tensor keys before stats")
    parser.add_argument("--depths", nargs="+", type=int, default=list(DEFAULT_DEPTHS), help="Prefix depths to count")
    parser.add_argument("--limit", type=int, default=20, help="Number of sample tensor keys to print")
    parser.add_argument("--json", type=Path, default=None, help="Optional path for a structured JSON summary")
    parser.add_argument(
        "--compare-model",
        default=None,
        metavar="MODULE:CLASS",
        help="Optional model class to instantiate and compare state_dict keys against",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = inspect_checkpoint(
        args.checkpoint,
        strip_prefix=args.strip_prefix,
        depths=tuple(args.depths),
        limit=args.limit,
        compare_model=args.compare_model,
    )
    print_summary(summary)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def inspect_checkpoint(
    path: Path,
    *,
    strip_prefix: str,
    depths: tuple[int, ...],
    limit: int,
    compare_model: str | None,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu")
    state_dict, source = select_state_dict(checkpoint)
    entries = collect_entries(state_dict, strip_prefix=strip_prefix)
    tensor_entries = [entry for entry in entries if entry["is_tensor"]]
    non_tensor_count = len(entries) - len(tensor_entries)
    tensor_keys = [entry["key"] for entry in tensor_entries]

    summary: dict[str, Any] = {
        "path": str(path),
        "size_gb": path.stat().st_size / GB,
        "checkpoint_type": type(checkpoint).__name__,
        "top_level": summarize_top_level(checkpoint),
        "state_dict_source": source,
        "strip_prefix": strip_prefix,
        "all_tensor_keys_have_module_prefix": all_have_prefix(tensor_entries, "module."),
        "tensor_count": len(tensor_entries),
        "non_tensor_count": non_tensor_count,
        "total_numel": sum(entry["numel"] for entry in tensor_entries),
        "tensor_bytes_gb": sum(entry["bytes"] for entry in tensor_entries) / GB,
        "dtype_counts": dict(sorted(Counter(entry["dtype"] for entry in tensor_entries).items())),
        "prefix_counts": prefix_counts(tensor_keys, depths),
        "sample_tensors": tensor_entries[:limit],
    }
    if compare_model is not None:
        summary["compare_model"] = compare_to_model(tensor_entries, compare_model, depths, limit)
    return summary


def select_state_dict(checkpoint: Any) -> tuple[Mapping[str, Any], str]:
    if isinstance(checkpoint, Mapping):
        if isinstance(checkpoint.get("model"), Mapping):
            return checkpoint["model"], "model"
        if isinstance(checkpoint.get("state_dict"), Mapping):
            return checkpoint["state_dict"], "state_dict"
        if looks_like_state_dict(checkpoint):
            return checkpoint, "root"
    raise TypeError("checkpoint does not contain a recognizable state_dict mapping")


def looks_like_state_dict(value: Mapping[str, Any]) -> bool:
    return bool(value) and any(isinstance(key, str) and isinstance(item, Tensor) for key, item in value.items())


def collect_entries(state_dict: Mapping[str, Any], *, strip_prefix: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for key in sorted(state_dict):
        value = state_dict[key]
        stripped_key = strip_key(key, strip_prefix)
        if isinstance(value, Tensor):
            entries.append(
                {
                    "key": stripped_key,
                    "original_key": key,
                    "is_tensor": True,
                    "shape": list(value.shape),
                    "dtype": str(value.dtype).removeprefix("torch."),
                    "numel": value.numel(),
                    "bytes": value.numel() * value.element_size(),
                }
            )
        else:
            entries.append(
                {
                    "key": stripped_key,
                    "original_key": key,
                    "is_tensor": False,
                    "type": type(value).__name__,
                }
            )
    return entries


def strip_key(key: str, prefix: str) -> str:
    if prefix and key.startswith(prefix):
        return key.removeprefix(prefix)
    return key


def all_have_prefix(entries: list[dict[str, Any]], prefix: str) -> bool:
    return bool(entries) and all(str(entry["original_key"]).startswith(prefix) for entry in entries)


def prefix_counts(keys: list[str], depths: tuple[int, ...]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for depth in depths:
        counter = Counter(prefix_at_depth(key, depth) for key in keys)
        counts[str(depth)] = dict(sorted(counter.items()))
    return counts


def prefix_at_depth(key: str, depth: int) -> str:
    parts = key.split(".")
    return ".".join(parts[:depth])


def summarize_top_level(checkpoint: Any) -> dict[str, Any]:
    if not isinstance(checkpoint, Mapping):
        return {"type": type(checkpoint).__name__}
    return {
        "type": type(checkpoint).__name__,
        "keys": [
            {"key": str(key), "type": type(value).__name__}
            for key, value in sorted(checkpoint.items(), key=lambda item: str(item[0]))
        ],
    }


def compare_to_model(
    tensor_entries: list[dict[str, Any]],
    class_path: str,
    depths: tuple[int, ...],
    limit: int,
) -> dict[str, Any]:
    model = instantiate_model(class_path)
    model_state = model.state_dict()
    checkpoint_shapes = {entry["key"]: tuple(entry["shape"]) for entry in tensor_entries}
    model_shapes = {key: tuple(value.shape) for key, value in model_state.items()}
    checkpoint_keys = set(checkpoint_shapes)
    model_keys = set(model_shapes)
    common = sorted(checkpoint_keys & model_keys)
    shape_matches = [key for key in common if checkpoint_shapes[key] == model_shapes[key]]
    shape_mismatches = [
        {"key": key, "checkpoint_shape": list(checkpoint_shapes[key]), "model_shape": list(model_shapes[key])}
        for key in common
        if checkpoint_shapes[key] != model_shapes[key]
    ]
    missing = sorted(model_keys - checkpoint_keys)
    unexpected = sorted(checkpoint_keys - model_keys)
    return {
        "class": class_path,
        "model_tensor_count": len(model_keys),
        "checkpoint_tensor_count": len(checkpoint_keys),
        "common_keys": len(common),
        "shape_match": len(shape_matches),
        "shape_mismatch": len(shape_mismatches),
        "missing": len(missing),
        "unexpected": len(unexpected),
        "missing_prefix_counts": prefix_counts(missing, depths),
        "unexpected_prefix_counts": prefix_counts(unexpected, depths),
        "sample_shape_mismatches": shape_mismatches[:limit],
        "sample_missing": missing[:limit],
        "sample_unexpected": unexpected[:limit],
    }


def instantiate_model(class_path: str) -> nn.Module:
    module_name, _, class_name = class_path.partition(":")
    if not module_name or not class_name:
        raise ValueError("--compare-model must use MODULE:CLASS format")
    module = importlib.import_module(module_name)
    model_cls = getattr(module, class_name)
    config_cls = getattr(model_cls, "config_class", None)
    if config_cls is None:
        raise AttributeError(f"{class_path} does not define config_class")
    model = model_cls(config_cls())
    if not isinstance(model, nn.Module):
        raise TypeError(f"{class_path} did not instantiate a torch.nn.Module")
    return model


def print_summary(summary: dict[str, Any]) -> None:
    print(f"path: {summary['path']}")
    print(f"size_gb: {summary['size_gb']:.6f}")
    print(f"checkpoint_type: {summary['checkpoint_type']}")
    print(f"state_dict_source: {summary['state_dict_source']}")
    print("top_level:")
    top_level = summary["top_level"]
    print(f"  type: {top_level['type']}")
    for item in top_level.get("keys", []):
        print(f"  - {item['key']}: {item['type']}")
    print(f"strip_prefix: {summary['strip_prefix']!r}")
    print(f"all_tensor_keys_have_module_prefix: {summary['all_tensor_keys_have_module_prefix']}")
    print(f"tensor_count: {summary['tensor_count']}")
    print(f"non_tensor_count: {summary['non_tensor_count']}")
    print(f"total_numel: {summary['total_numel']}")
    print(f"tensor_bytes_gb: {summary['tensor_bytes_gb']:.6f}")
    print("dtype_counts:")
    for dtype, count in summary["dtype_counts"].items():
        print(f"  {dtype}: {count}")
    print_prefix_counts(summary["prefix_counts"])
    print("sample_tensors:")
    for entry in summary["sample_tensors"]:
        print(f"  - {entry['key']}: shape={entry['shape']} dtype={entry['dtype']}")
    if "compare_model" in summary:
        print_compare_summary(summary["compare_model"])


def print_prefix_counts(counts: dict[str, dict[str, int]], *, title: str = "prefix_counts") -> None:
    print(f"{title}:")
    for depth, depth_counts in counts.items():
        print(f"  depth_{depth}:")
        for prefix, count in depth_counts.items():
            print(f"    {prefix}: {count}")


def print_compare_summary(compare: dict[str, Any]) -> None:
    print("compare_model:")
    print(f"  class: {compare['class']}")
    print(f"  model_tensor_count: {compare['model_tensor_count']}")
    print(f"  checkpoint_tensor_count: {compare['checkpoint_tensor_count']}")
    print(f"  common_keys: {compare['common_keys']}")
    print(f"  shape_match: {compare['shape_match']}")
    print(f"  shape_mismatch: {compare['shape_mismatch']}")
    print(f"  missing: {compare['missing']}")
    print(f"  unexpected: {compare['unexpected']}")
    print_prefix_counts(compare["missing_prefix_counts"], title="  missing_prefix_counts")
    print_prefix_counts(compare["unexpected_prefix_counts"], title="  unexpected_prefix_counts")
    if compare["sample_shape_mismatches"]:
        print("  sample_shape_mismatches:")
        for item in compare["sample_shape_mismatches"]:
            print(
                "  "
                f"- {item['key']}: checkpoint_shape={item['checkpoint_shape']} "
                f"model_shape={item['model_shape']}"
            )
    if compare["sample_missing"]:
        print("  sample_missing:")
        for key in compare["sample_missing"]:
            print(f"  - {key}")
    if compare["sample_unexpected"]:
        print("  sample_unexpected:")
        for key in compare["sample_unexpected"]:
            print(f"  - {key}")


if __name__ == "__main__":
    main()
