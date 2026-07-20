"""Whole-checkpoint load parity for BoltzGenModel vs boltzgen1_diverse."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ofoldx.models.boltzgen.config import BoltzGenConfig
from ofoldx.models.boltzgen.convert import convert_structure_design
from ofoldx.models.boltzgen.model import BoltzGenModel
from scripts.parity.boltzgen import reference
from scripts.parity.paths import boltzgen_checkpoint

CKPT = boltzgen_checkpoint("boltzgen1_diverse.ckpt")


def main() -> None:
    args = parse_args()
    cfg = BoltzGenConfig()
    model = BoltzGenModel(cfg).set_backend("torch")
    ck = reference.load_checkpoint(args.checkpoint)
    converted = convert_structure_design(ck["state_dict"])
    missing, unexpected = model.load_state_dict(converted, strict=True)
    print(f"ckpt {len(ck['state_dict'])} -> converted {len(converted)} | model {len(model.state_dict())} params")
    print(f"missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        from collections import Counter

        print("  missing top groups:", dict(Counter(k.split(".")[0] for k in missing)))
        print("  missing sample:", missing[:8])
    if unexpected:
        from collections import Counter

        print("  unexpected top groups:", dict(Counter(k.split(".")[0] for k in unexpected)))
        print("  unexpected sample:", unexpected[:8])
    if not missing and not unexpected:
        print("[ok] BoltzGenModel whole-ckpt load parity 0/0")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CKPT)
    return parser.parse_args()


if __name__ == "__main__":
    main()
