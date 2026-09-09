"""Families whose official source checkpoints are not exercised in this environment.

Prints an explicit blocked reason. These are not CI skips: the harness records why
convert+parity could not be run here (size, license gate, or missing public file).
"""

from __future__ import annotations

BLOCKED = (
    (
        "alphafold2",
        "official CASP14/multimer parameter tarballs are multi-GB licensed Google downloads",
    ),
    (
        "alphafold3",
        "official AlphaFold 3 weights are gated and not downloadable in this environment",
    ),
    (
        "esmfold2",
        "facebook/esmfold_v1 pytorch_model.bin is 8.4GB; deferred",
    ),
    (
        "boltz1",
        "boltz-community/boltz-1 boltz1.ckpt is 6.9GB; deferred",
    ),
    (
        "boltz2",
        "released Boltz-2 checkpoints are multi-GB; deferred",
    ),
    (
        "boltzgen",
        "composite five-role checkpoint set is multi-GB; deferred",
    ),
    (
        "protenix",
        "official Protenix checkpoints are not anonymously listed on Hugging Face here",
    ),
    (
        "pxdesign",
        "PXDesign-d conversion requires ProteinMPNN + AlphaFold2 + two Protenix checkpoints",
    ),
    (
        "odesign",
        "odesign_base_prot_rigid.pt is 4.3GB; deferred after smaller families",
    ),
)


def main() -> None:
    for family, reason in BLOCKED:
        print(f"{family}: BLOCKED ({reason})", flush=True)


if __name__ == "__main__":
    main()
