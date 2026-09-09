"""Load the released BoltzGen inverse-folding checkpoint and convert it.

boltzgen1_ifold.ckpt is 12MB. The Lightning pickle references upstream boltzgen
objects; the harness extracts the tensor state_dict with a tolerant unpickler
and calls convert_inverse_folding.

Run::

    python examples/parity/boltzgen.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from ofoldx.models.boltzgen.convert import convert_inverse_folding
from ofoldx.models.boltzgen.model import BoltzGenInverseFoldingModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import hf_download, load_pickle_checkpoint, report_convert

_REPO = "boltzgen/boltzgen-1"
_FILE = "boltzgen1_ifold.ckpt"


def _state_dict_from_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    loaded = load_pickle_checkpoint(path)
    if not isinstance(loaded, dict) or "state_dict" not in loaded:
        raise TypeError(f"unsupported BoltzGen checkpoint type {type(loaded)!r}")
    stripped: dict[str, torch.Tensor] = {}
    for key, value in loaded["state_dict"].items():
        if not torch.is_tensor(value):
            continue
        name = key.removeprefix("module.").removeprefix("model.")
        stripped[name] = value
    return stripped


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args(argv)
    checkpoint = args.checkpoint if args.checkpoint is not None else hf_download(_REPO, _FILE)
    print(f"boltzgen: source checkpoint {checkpoint}", flush=True)
    converted = convert_inverse_folding(_state_dict_from_checkpoint(checkpoint))
    with torch.device("meta"):
        model = BoltzGenInverseFoldingModel()
    missing, unexpected = model.load_state_dict(converted, strict=True, assign=True)
    if missing or unexpected:
        raise RuntimeError(f"boltzgen strict load failed missing={missing} unexpected={unexpected}")
    model.eval()
    report_convert("boltzgen/ifold/convert", model)


if __name__ == "__main__":
    main()
