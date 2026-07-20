#!/usr/bin/env python
r"""Compare OFoldX RFdiffusion T/O/I matrix order with the released upstream asset.

The runtime implementation owns a deterministic rounded-lexicographic canonical
ordering and maps it to the released RFdiffusion order. This script checks the
result against ``rfdiffusion/inference/sym_rots.npz`` from a reference checkout.
It is a component diagnostic, not public reverse-sampler or DesignResult signoff.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ofoldx.models.rfdiffusion.model import RFdiffusionSymmetry  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=Path.home() / "reference" / "RFdiffusion",
        help="RosettaCommons/RFdiffusion reference checkout.",
    )
    args = parser.parse_args()
    reference_path = args.reference_root / "rfdiffusion" / "inference" / "sym_rots.npz"
    reference = np.load(reference_path)

    errors = {}
    for kind in ("tetrahedral", "octahedral", "icosahedral"):
        native = np.stack([matrix.numpy() for matrix in RFdiffusionSymmetry.rotation_matrices(kind, None)])
        expected = reference[kind]
        np.testing.assert_allclose(native, expected, rtol=0.0, atol=1e-6)
        errors[kind] = float(np.max(np.abs(native - expected)))
    print(json.dumps(errors, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
