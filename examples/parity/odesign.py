"""Load a released ODesign backbone checkpoint and convert it.

convert_odesign_model instantiates a full CPU model before loading, which OOMs
on a 15GB machine for the 4.3GB checkpoint. This harness uses the same
load_reference_state + translate_odesign_state path with a meta module and
assign=True, then strict-loads the backbone (no MPNN children).

Run::

    python examples/parity/odesign.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from ofoldx.models.odesign.convert import load_reference_state, translate_odesign_state
from ofoldx.models.odesign.model import ODesignModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import hf_download, report_convert

_REPO = "The-Institute-for-AI-Molecular-Design/ODesign"
_FILE = "ckpt/odesign_base_prot_rigid.pt"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args(argv)
    checkpoint = args.checkpoint if args.checkpoint is not None else hf_download(_REPO, _FILE)
    print(f"odesign: source checkpoint {checkpoint}", flush=True)
    with torch.device("meta"):
        model = ODesignModel()
    translated = translate_odesign_state(model, load_reference_state(checkpoint))
    missing, unexpected = model.load_state_dict(translated, strict=True, assign=True)
    model.eval()
    if missing or unexpected:
        raise RuntimeError(f"odesign strict load failed missing={missing} unexpected={unexpected}")
    report_convert("odesign/prot_rigid/convert", model)


if __name__ == "__main__":
    main()
