"""Load a released OpenFold AlphaFold 2 checkpoint and convert it.

Official DeepMind CASP14 tarballs are multi-GB. This harness uses the public
OpenFold finetuning_no_templ weights (nz/OpenFold) with source='openfold'.

Run::

    python examples/parity/alphafold2.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ofoldx.models.alphafold2.convert import convert_alphafold2_model

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import hf_download, report_convert

_REPO = "nz/OpenFold"
_FILE = "finetuning_no_templ_1.pt"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args(argv)
    checkpoint = args.checkpoint if args.checkpoint is not None else hf_download(_REPO, _FILE)
    print(f"alphafold2: source checkpoint {checkpoint} source=openfold", flush=True)
    model = convert_alphafold2_model(checkpoint, source="openfold")
    report_convert("alphafold2/openfold/convert", model)


if __name__ == "__main__":
    main()
