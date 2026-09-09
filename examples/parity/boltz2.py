"""Load a released Boltz-2 structure checkpoint and convert it.

Uses convert_boltz2_model on boltz2_conf.ckpt (structure+confidence, 2.3GB).

Run::

    python examples/parity/boltz2.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ofoldx.models.boltz2.convert import convert_boltz2_model

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import hf_download, report_convert

_REPO = "boltz-community/boltz-2"
_FILE = "boltz2_conf.ckpt"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args(argv)
    checkpoint = args.checkpoint if args.checkpoint is not None else hf_download(_REPO, _FILE)
    print(f"boltz2: source checkpoint {checkpoint}", flush=True)
    model = convert_boltz2_model(checkpoint)
    model.eval()
    report_convert("boltz2/conf/convert", model)


if __name__ == "__main__":
    main()
