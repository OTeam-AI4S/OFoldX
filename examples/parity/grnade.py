"""Load the released gRNAde checkpoint and convert it into GrnadeModel.

Run::

    python examples/parity/grnade.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

from ofoldx.models.grnade import GrnadeConfig, GrnadeModel
from ofoldx.models.grnade.convert import convert_grnade_state_dict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import CKPT_DIR


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.checkpoint is None:
        checkpoint = Path(
            hf_hub_download(
                "chaitjo/gRNAde",
                "gRNAde_drop3d@0.75_maxlen@500.h5",
                local_dir=str(CKPT_DIR),
            )
        )
    else:
        checkpoint = args.checkpoint
    print(f"grnade: source checkpoint {checkpoint}", flush=True)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    converted = convert_grnade_state_dict(state)
    model = GrnadeModel(GrnadeConfig(edge_scalar_size=132, num_token_embeddings=5))
    missing, unexpected = model.load_state_dict(converted, strict=True, assign=True)
    model.eval()
    print(
        f"grnade/convert: tensors={len(converted)} missing={len(missing)} unexpected={len(unexpected)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
