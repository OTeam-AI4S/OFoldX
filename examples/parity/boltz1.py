"""Load a released Boltz-1 checkpoint and convert it.

Prefers boltz1_conf.ckpt (3.6GB) when present; falls back to boltz1.ckpt.
Uses mmap when possible so the 15GB CPU machine can hold the converted tensors.

Run::

    python examples/parity/boltz1.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from ofoldx.models.boltz1.convert import (
    boltz1_config_from_checkpoint,
    convert_boltz1_model,
    convert_boltz1_state_dict,
)
from ofoldx.models.boltz1.model import Boltz1Model

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import hf_download, report_convert

_REPO = "boltz-community/boltz-1"
_FILE = "boltz1_conf.ckpt"
_FALLBACK = "boltz1.ckpt"


def _load_checkpoint(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu", weights_only=False)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.checkpoint is not None:
        checkpoint = args.checkpoint
    else:
        try:
            checkpoint = hf_download(_REPO, _FILE)
        except Exception as error:
            print(f"boltz1: {_FILE} unavailable ({error}); trying {_FALLBACK}", flush=True)
            checkpoint = hf_download(_REPO, _FALLBACK)
    print(f"boltz1: source checkpoint {checkpoint}", flush=True)
    try:
        model = convert_boltz1_model(checkpoint)
    except (RuntimeError, MemoryError) as error:
        print(f"boltz1: convert_boltz1_model failed ({error!r}); retrying mmap+meta", flush=True)
        loaded = _load_checkpoint(checkpoint)
        config = boltz1_config_from_checkpoint(loaded)
        converted = convert_boltz1_state_dict(loaded["state_dict"], config=config)
        del loaded
        with torch.device("meta"):
            model = Boltz1Model(config)
        model.load_state_dict(converted, strict=True, assign=True)
    model.eval()
    report_convert("boltz1/convert", model)


if __name__ == "__main__":
    main()
