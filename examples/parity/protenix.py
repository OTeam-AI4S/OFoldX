"""Load a released Protenix mini checkpoint and convert it.

Uses convert_protenix_model with variant protenix_mini_default_v0.5.0 (no ESM
companion). Mini is the smallest publicly listed default checkpoint.

Run::

    python examples/parity/protenix.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ofoldx.models.protenix.convert import convert_protenix_model

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import hf_download, report_convert

_REPO = "TMF001/pxdesign-weights"
_FILE = "checkpoint/protenix_mini_default_v0.5.0.pt"
_VARIANT = "protenix_mini_default_v0.5.0"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args(argv)
    checkpoint = args.checkpoint if args.checkpoint is not None else hf_download(_REPO, _FILE)
    print(f"protenix: source checkpoint {checkpoint} variant={_VARIANT}", flush=True)
    model = convert_protenix_model(checkpoint, variant=_VARIANT)
    report_convert("protenix/mini_default/convert", model)


if __name__ == "__main__":
    main()
