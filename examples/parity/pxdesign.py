"""Load the released PXDesign-d backbone checkpoint and convert it.

Uses convert_pxdesign_model (meta device, strict load). The complete six-pair
pipeline also needs ProteinMPNN + AlphaFold2 + two Protenix checkpoints; this
harness covers the PXDesign family backbone converter.

Run::

    python examples/parity/pxdesign.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ofoldx.models.pxdesign.convert import convert_pxdesign_model

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import hf_download, report_convert

_REPO = "TMF001/pxdesign-weights"
_FILE = "checkpoint/pxdesign_v0.1.0.pt"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args(argv)
    checkpoint = args.checkpoint if args.checkpoint is not None else hf_download(_REPO, _FILE)
    print(f"pxdesign: source checkpoint {checkpoint}", flush=True)
    model = convert_pxdesign_model(checkpoint)
    report_convert("pxdesign/backbone/convert", model)


if __name__ == "__main__":
    main()
