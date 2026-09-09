"""Load the released BoltzGen inverse-folding checkpoint and convert it.

boltzgen1_ifold.ckpt is 12MB. The Lightning pickle may require upstream
boltzgen objects; if so the harness extracts the tensor state_dict and calls
convert_inverse_folding directly.

Run::

    python examples/parity/boltzgen.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from ofoldx.models.boltzgen.convert import convert_boltzgen_inverse_folding_model, convert_inverse_folding
from ofoldx.models.boltzgen.model import BoltzGenInverseFoldingModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import hf_download, report_convert

_REPO = "boltzgen/boltzgen-1"
_FILE = "boltzgen1_ifold.ckpt"


def _state_dict_from_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    try:
        loaded = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        loaded = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(loaded, dict) and "state_dict" in loaded:
        state = loaded["state_dict"]
    elif isinstance(loaded, dict):
        state = loaded
    else:
        raise TypeError(f"unsupported BoltzGen checkpoint type {type(loaded)!r}")
    stripped: dict[str, torch.Tensor] = {}
    for key, value in state.items():
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
    try:
        model = convert_boltzgen_inverse_folding_model(checkpoint)
    except Exception as error:
        print(f"boltzgen: convert_boltzgen_inverse_folding_model failed ({error!r}); using state_dict path", flush=True)
        converted = convert_inverse_folding(_state_dict_from_checkpoint(checkpoint))
        with torch.device("meta"):
            model = BoltzGenInverseFoldingModel()
        model.load_state_dict(converted, strict=True, assign=True)
    model.eval()
    report_convert("boltzgen/ifold/convert", model)


if __name__ == "__main__":
    main()
