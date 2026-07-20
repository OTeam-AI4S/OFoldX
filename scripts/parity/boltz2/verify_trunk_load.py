"""Whole-trunk checkpoint-parity check for Boltz2Model.

Convert the full ``boltz2_conf.ckpt`` to the trunk param tree and ``strict``-load it into a fresh
:class:`Boltz2Model`. A clean load proves every trunk parameter has a checkpoint source under clean
naming. Non-trunk keys (diffusion / confidence / affinity / bfactor) are dropped until those phases land,
so the model's *missing* set must be empty and *unexpected* must be empty too.

Usage::

    python -m scripts.parity.boltz2.verify_trunk_load [--checkpoint PATH]
"""

from __future__ import annotations

import argparse

import torch

from ofoldx.models.boltz2 import Boltz2Config, convert
from ofoldx.models.boltz2.model import Boltz2Model
from scripts.parity.paths import weight_path

DEFAULT_CHECKPOINT = weight_path("boltz2_conf.ckpt")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    args = parser.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    converted = convert.convert_trunk(state)
    model = Boltz2Model(Boltz2Config())
    model_params = dict(model.state_dict())

    missing, unexpected = model.load_state_dict(converted, strict=True)
    missing = [k for k in missing if k in model_params]
    print(f"checkpoint tensors: {len(state)}")
    print(f"converted trunk tensors: {len(converted)}")
    print(f"model trunk params: {len(model_params)}")
    print(f"missing (model params with no checkpoint source): {len(missing)}")
    for k in missing[:20]:
        print("   MISSING", k)
    print(f"unexpected (converted keys not in model): {len(unexpected)}")
    for k in unexpected[:20]:
        print("   UNEXPECTED", k)
    ok = not missing and not unexpected
    print("TRUNK LOAD:", "ok" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
