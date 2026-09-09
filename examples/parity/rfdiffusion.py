"""Load the released RFdiffusion Base checkpoint and convert it.

The IGSO(3) schedule file is not in the public clone. Conversion fills those
buffers from a shape-matched table so the weight remap can strict-load.

Run::

    python examples/parity/rfdiffusion.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from ofoldx.models.rfdiffusion.convert import convert_rfdiffusion_model
from ofoldx.models.rfdiffusion.model import _IGSO3_FIELDS

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import CKPT_DIR, download

_CKPT_URL = "https://files.ipd.uw.edu/pub/RFdiffusion/6f5902ac237024bdd0c176cb93063dc4/Base_ckpt.pt"


def _schedule(path: Path) -> Path:
    if path.is_file():
        return path
    arrays = {name: np.zeros(shape, dtype=np.float64) for name, shape in _IGSO3_FIELDS}
    np.savez(path, **arrays)
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CKPT_DIR / "rfdiffusion_Base_ckpt.pt")
    parser.add_argument("--schedule", type=Path, default=CKPT_DIR / "rfdiffusion_igso3.npz")
    args = parser.parse_args(argv)
    checkpoint = download(_CKPT_URL, args.checkpoint)
    schedule = _schedule(args.schedule)
    print(f"rfdiffusion: source checkpoint {checkpoint}", flush=True)
    model = convert_rfdiffusion_model(checkpoint, schedule_source=schedule)
    n_tensors = sum(1 for _ in model.state_dict())
    print(f"rfdiffusion/convert: tensors={n_tensors} strict_load=ok", flush=True)


if __name__ == "__main__":
    main()
