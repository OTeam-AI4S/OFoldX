#!/usr/bin/env python
r"""Probe AF-Multimer strict weight loading to enumerate node4 forward-blockers.

Loads the official ``params_model_*_multimer_v3.npz`` from the DeepMind tar, runs the
OFoldX multimer key normalization, builds a meta ``AlphaFold2MultimerModel``, and prints
the full key/shape reconciliation (missing / unexpected / shape-mismatch) instead of
failing on the first ``load_state_dict`` error.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.parity.paths import weights_root  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=Path, default=weights_root() / "af2-params/alphafold_params_2022-12-06.tar"
    )
    parser.add_argument("--model-name", default="model_1_multimer_v3")
    parser.add_argument("--show", type=int, default=200)
    args = parser.parse_args()

    from ofoldx.models.alphafold2.config import AlphaFold2MultimerConfig
    from ofoldx.models.alphafold2.converters import deepmind, deepmind_multimer
    from ofoldx.models.alphafold2.model import AlphaFold2MultimerModel

    params, resolved = deepmind._load_official_alphafold2_params(args.checkpoint, model_name=args.model_name)
    params = dict(params)
    print(f"resolved model: {resolved}; official tensors: {len(params)}")

    converted = deepmind_multimer.convert_official_multimer_state_dict(params)
    print(f"converted native keys: {len(converted)}")

    overrides = deepmind_multimer._multimer_config_overrides(resolved)
    with torch.device("meta"):
        model = AlphaFold2MultimerModel(AlphaFold2MultimerConfig(overrides))
    model_state = model.state_dict()
    print(f"model params: {len(model_state)}")

    model_keys = set(model_state)
    conv_keys = set(converted)
    missing = sorted(model_keys - conv_keys)
    unexpected = sorted(conv_keys - model_keys)
    shape_mismatch = []
    for key in sorted(model_keys & conv_keys):
        want = tuple(model_state[key].shape)
        got = tuple(converted[key].shape)
        if want != got:
            shape_mismatch.append((key, want, got))

    print(f"\n=== MISSING (in model, absent from converted): {len(missing)} ===")
    for key in missing[: args.show]:
        print("  ", key, tuple(model_state[key].shape))
    print(f"\n=== UNEXPECTED (in converted, absent from model): {len(unexpected)} ===")
    for key in unexpected[: args.show]:
        print("  ", key, tuple(converted[key].shape))
    print(f"\n=== SHAPE MISMATCH: {len(shape_mismatch)} ===")
    for key, want, got in shape_mismatch[: args.show]:
        print("  ", key, "want", want, "got", got)

    print("\n=== sample official scopes (unique second path segment) ===")
    scopes: dict[str, int] = {}
    for key in params:
        scope = key.split("//")[0]
        parts = [p for p in scope.split("/") if p]
        if parts[:2] == ["alphafold", "alphafold_iteration"]:
            parts = parts[2:]
        tag = "/".join(parts[:2])
        scopes[tag] = scopes.get(tag, 0) + 1
    for tag, count in sorted(scopes.items()):
        print("  ", tag, count)


if __name__ == "__main__":
    main()
