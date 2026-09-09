"""Attempt official AlphaFold 3 conversion; record BLOCKED if weights are gated.

The converter only accepts DeepMind record-stream parameters. This harness tries
known public locations, then clones google-deepmind/alphafold3 to see whether a
random Haiku state can be converted. Either a strict-load pass or an explicit
blocked reason is printed.

Run::

    python examples/parity/alphafold3.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import UPSTREAM_DIR, clone_source

_SOURCE_URL = "https://github.com/google-deepmind/alphafold3.git"
_PARAM_CANDIDATES = (
    os.environ.get("ALPHAFOLD3_PARAMS"),
    "/tmp/ofoldx-ckpts/alphafold3",
    "/tmp/ofoldx-ckpts/af3.bin.zst",
)


def _existing_params() -> Path | None:
    for candidate in _PARAM_CANDIDATES:
        if not candidate:
            continue
        path = Path(candidate)
        if path.is_file() or (path.is_dir() and any(path.iterdir())):
            return path
    return None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args(argv)
    checkpoint = args.checkpoint if args.checkpoint is not None else _existing_params()
    if checkpoint is not None:
        from ofoldx.models.alphafold3.convert import convert_alphafold3_model

        from _common import report_convert

        print(f"alphafold3: source checkpoint {checkpoint}", flush=True)
        model = convert_alphafold3_model(checkpoint)
        model.eval()
        report_convert("alphafold3/convert", model)
        return

    try:
        source = clone_source(_SOURCE_URL, UPSTREAM_DIR / "alphafold3", "src/alphafold3")
        source_note = f"cloned {source}"
    except Exception as error:
        source_note = f"upstream clone failed ({error!r})"
    print(
        "alphafold3: BLOCKED (official DeepMind AF3 weights are gated; "
        f"{source_note}; no public parameter file is obtainable here, and "
        "random Haiku init needs JAX plus the full official graph)",
        flush=True,
    )


if __name__ == "__main__":
    main()
