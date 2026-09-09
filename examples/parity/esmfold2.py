"""Load the Biohub ESMFold2-Fast folding head and convert it.

The native converter also loads frozen ESMC-6B weights. This environment cannot
materialize that language model, so the harness translates the official head
with load_esmfold2_state_dict and strict-loads it onto a meta EsmFold2Model
whose ESMC child stays on the meta device (the same missing-key set as
convert_esmfold2_checkpoint).

Run::

    python examples/parity/esmfold2.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from multimolecule import EsmCConfig, EsmCModel

from ofoldx.models.esmfold2 import EsmFold2Config, EsmFold2Model
from ofoldx.models.esmfold2.convert import (
    _BIOHUB_UNUSED_STATE_KEYS,
    _translate_biohub_config,
    load_esmfold2_state_dict,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import hf_download, report_convert

_REPO = "biohub/ESMFold2-Fast"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)
    checkpoint = args.checkpoint if args.checkpoint is not None else hf_download(_REPO, "model.safetensors")
    config_path = args.config if args.config is not None else hf_download(_REPO, "config.json")
    print(f"esmfold2: source checkpoint {checkpoint}", flush=True)
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    if raw_config["model_type"] != "esmfold2" or raw_config["type"] != "release":
        raise ValueError("checkpoint config is not the released Biohub ESMFold2 architecture")
    config = EsmFold2Config(_translate_biohub_config(raw_config))
    with torch.device("meta"):
        esmc = EsmCModel(EsmCConfig(**dict(config.esmc.dict())))
        model = EsmFold2Model(config, esmc=esmc)
    head_state = load_esmfold2_state_dict(checkpoint)
    for name in _BIOHUB_UNUSED_STATE_KEYS:
        head_state.pop(name, None)
    incompatible = model.load_state_dict(head_state, strict=False, assign=True)
    expected_missing = {f"esmc.{name}" for name in model.esmc.state_dict()}
    missing = set(incompatible.missing_keys)
    if missing != expected_missing or incompatible.unexpected_keys:
        missing_head = sorted(missing - expected_missing)
        unexpected = sorted(incompatible.unexpected_keys)
        raise RuntimeError(
            "ESMFold2 checkpoint does not match the released head; "
            f"missing_head={missing_head!r}, unexpected={unexpected!r}"
        )
    model.eval()
    report_convert("esmfold2/fast/convert", model)
    print(
        f"esmfold2/fast/convert: head_tensors={len(head_state)} esmc_missing={len(expected_missing)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
