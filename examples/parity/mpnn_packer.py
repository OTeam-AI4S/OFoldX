"""Load a released LigandMPNN side-chain packer checkpoint and convert it.

Strict-loads the converted state into MpnnPackerModel. Numeric upstream compare is
left to a follow-up once the Packer feature_dict is wired.

Run::

    python examples/parity/mpnn_packer.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from ofoldx.models.mpnn_packer import MpnnPackerConfig, MpnnPackerModel
from ofoldx.models.mpnn_packer.convert import convert_mpnn_packer_state_dict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import CKPT_DIR, download

_CKPT_URL = "https://files.ipd.uw.edu/pub/ligandmpnn/ligandmpnn_sc_v_32_002_16.pt"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CKPT_DIR / "ligandmpnn_sc_v_32_002_16.pt")
    args = parser.parse_args(argv)
    checkpoint = download(_CKPT_URL, args.checkpoint)
    print(f"mpnn_packer: source checkpoint {checkpoint}", flush=True)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = MpnnPackerConfig(
        num_neighbors=int(ckpt["num_edges"]),
        num_context_atoms=int(ckpt["atom_context_num"]),
    )
    model = MpnnPackerModel(config)
    converted = convert_mpnn_packer_state_dict(ckpt["model_state_dict"])
    missing, unexpected = model.load_state_dict(converted, strict=True, assign=True)
    n_params = sum(value.numel() for value in converted.values())
    print(
        f"mpnn_packer/convert: tensors={len(converted)} params={n_params} "
        f"missing={len(missing)} unexpected={len(unexpected)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
