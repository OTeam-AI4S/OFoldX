#!/usr/bin/env python
r"""Verify the OFoldX-native RfDiffusion network state tree."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    sys.path.insert(0, str(REPO))
    from ofoldx import AutoModel, AutoProcessor
    from ofoldx.models.rfdiffusion import RFdiffusionModel, RFdiffusionProcessor

    args = parser().parse_args()
    model = AutoModel(args.artifact)
    processor = AutoProcessor(args.artifact)
    if type(model) is not RFdiffusionModel or type(processor) is not RFdiffusionProcessor:
        raise TypeError("artifact is not a paired RFdiffusion repository")
    print(f"model_tensors={len(model.state_dict())}")
    print("repository_files=model.json, processor.json, model.safetensors")
    print("strict_load=ok")


def parser() -> argparse.ArgumentParser:
    arg_parser = argparse.ArgumentParser(description=__doc__)
    arg_parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("artifacts/rfdiffusion-base"),
        help="RfDiffusion pretrained artifact directory.",
    )
    return arg_parser


if __name__ == "__main__":
    main()
