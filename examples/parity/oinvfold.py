"""Load a released OInvFold Lightning checkpoint and convert it.

Run::

    python examples/parity/oinvfold.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

from ofoldx.models.oinvfold import OInvFoldPolymerConfig, OInvFoldPolymerModel
from ofoldx.models.oinvfold.convert import (
    _lightning_state_dict,
    _live_state_dict,
    _polymer_omitted_source_keys,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import CKPT_DIR


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.checkpoint is None:
        checkpoint = Path(
            hf_hub_download(
                "The-Institute-for-AI-Molecular-Design/OInvFold",
                "oinvfold_protein.ckpt",
                local_dir=str(CKPT_DIR),
            )
        )
    else:
        checkpoint = args.checkpoint
    print(f"oinvfold: source checkpoint {checkpoint}", flush=True)
    model = OInvFoldPolymerModel(OInvFoldPolymerConfig())
    state = _live_state_dict(
        _lightning_state_dict(checkpoint),
        model.state_dict(),
        _polymer_omitted_source_keys(model.config.num_encoder_blocks),
        architecture="polymer",
        config=model.config,
    )
    missing, unexpected = model.load_state_dict(state, strict=True, assign=True)
    model.eval()
    print(
        f"oinvfold/protein/convert: tensors={len(state)} missing={len(missing)} unexpected={len(unexpected)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
