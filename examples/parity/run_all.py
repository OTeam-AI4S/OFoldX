"""Run every local conversion/parity harness that currently exists."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CHECKS = (
    "proteinmpnn.py",
    "ligandmpnn.py",
    "nampnn.py",
    "mpnn_packer.py",
    "grnade.py",
    "oinvfold.py",
    "rfdiffusion.py",
    "alphafold2.py",
    "pxdesign.py",
    "protenix.py",
    "esmfold2.py",
    "boltzgen.py",
    "boltz2.py",
    "boltz1.py",
    "odesign.py",
    "alphafold3.py",
)


def main() -> None:
    failures = []
    for name in CHECKS:
        print(f"\n=== {name} ===", flush=True)
        result = subprocess.run([sys.executable, str(ROOT / name)], cwd=str(ROOT.parent.parent))
        if result.returncode != 0:
            failures.append(name)
    if failures:
        raise SystemExit(f"failed: {', '.join(failures)}")
    print("\nall conversion checks passed", flush=True)


if __name__ == "__main__":
    main()
