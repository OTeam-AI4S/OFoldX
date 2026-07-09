r"""
RoseTTAFold residue/atom reference data for RfDiffusion.
"""

from __future__ import annotations

import base64
import io
import zlib

import torch
from torch import Tensor

NUM2AA = (
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
    "UNK",
    "MAS",
)
AA2NUM = {name: index for index, name in enumerate(NUM2AA)}
UNKNOWN_TOKEN = AA2NUM["UNK"]
MASK_TOKEN = AA2NUM["MAS"]

AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "UNK": "?",
    "MAS": "-",
}

ATOM14_NAMES = (
    ("N", "CA", "C", "O", "CB", None, None, None, None, None, None, None, None, None),
    ("N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2", None, None, None),
    ("N", "CA", "C", "O", "CB", "CG", "OD1", "ND2", None, None, None, None, None, None),
    ("N", "CA", "C", "O", "CB", "CG", "OD1", "OD2", None, None, None, None, None, None),
    ("N", "CA", "C", "O", "CB", "SG", None, None, None, None, None, None, None, None),
    ("N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2", None, None, None, None, None),
    ("N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2", None, None, None, None, None),
    ("N", "CA", "C", "O", None, None, None, None, None, None, None, None, None, None),
    ("N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2", None, None, None, None),
    ("N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1", None, None, None, None, None, None),
    ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", None, None, None, None, None, None),
    ("N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ", None, None, None, None, None),
    ("N", "CA", "C", "O", "CB", "CG", "SD", "CE", None, None, None, None, None, None),
    ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", None, None, None),
    ("N", "CA", "C", "O", "CB", "CG", "CD", None, None, None, None, None, None, None),
    ("N", "CA", "C", "O", "CB", "OG", None, None, None, None, None, None, None, None),
    ("N", "CA", "C", "O", "CB", "OG1", "CG2", None, None, None, None, None, None, None),
    ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"),
    ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH", None, None),
    ("N", "CA", "C", "O", "CB", "CG1", "CG2", None, None, None, None, None, None, None),
    ("N", "CA", "C", "O", "CB", None, None, None, None, None, None, None, None, None),
    ("N", "CA", "C", "O", "CB", None, None, None, None, None, None, None, None, None),
)

CHI_ATOM_NAMES: tuple[tuple[tuple[str, str, str, str] | None, ...], ...] = (
    (None, None, None, None),
    (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"), ("CB", "CG", "CD", "NE"), ("CG", "CD", "NE", "CZ")),
    (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "OD1"), None, None),
    (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "OD1"), None, None),
    (("N", "CA", "CB", "SG"), None, None, None),
    (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"), ("CB", "CG", "CD", "OE1"), None),
    (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"), ("CB", "CG", "CD", "OE1"), None),
    (None, None, None, None),
    (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "ND1"), None, None),
    (("N", "CA", "CB", "CG1"), ("CA", "CB", "CG1", "CD1"), None, None),
    (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1"), None, None),
    (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"), ("CB", "CG", "CD", "CE"), ("CG", "CD", "CE", "NZ")),
    (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "SD"), ("CB", "CG", "SD", "CE"), None),
    (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1"), None, None),
    (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"), None, None),
    (("N", "CA", "CB", "OG"), None, None, None),
    (("N", "CA", "CB", "OG1"), None, None, None),
    (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1"), None, None),
    (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1"), None, None),
    (("N", "CA", "CB", "CG1"), None, None, None),
    (None, None, None, None),
    (None, None, None, None),
)

INIT_CRDS = torch.full((27, 3), float("nan"))
INIT_CRDS[0] = torch.tensor([-0.5272, 1.3593, 0.0000])
INIT_CRDS[1] = torch.tensor([0.0000, 0.0000, 0.0000])
INIT_CRDS[2] = torch.tensor([1.5233, 0.0000, 0.0000])
INIT_CRDS[3] = torch.tensor([2.0505, 1.3593, 0.0000])
INIT_CRDS[4] = torch.tensor([-0.5291, -0.7749, -1.2056])


def residue_token(residue_name: str) -> int:
    r"""
    Map a three-letter residue name to its RFdiffusion integer id.

    Lookup is case-insensitive; unrecognized names fall back to the unknown token so parsing never fails on a
    non-standard residue.

    Args:
        residue_name: Three-letter residue name (e.g. `"ALA"`); case is ignored.

    Returns:
        The residue id from [`AA2NUM`][ofoldx.models.rfdiffusion.chemical.AA2NUM], or ``UNKNOWN_TOKEN``
        for an unrecognized name.

    Examples:
        >>> from ofoldx.models.rfdiffusion.chemical import residue_token, UNKNOWN_TOKEN
        >>> residue_token("gly")
        7
        >>> residue_token("XYZ") == UNKNOWN_TOKEN
        True
    """

    return AA2NUM.get(residue_name.upper(), UNKNOWN_TOKEN)


_RF_ALL_ATOM_KINEMATICS = (
    "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"
)
_RF_ALL_ATOM_CACHE: tuple[Tensor, Tensor, Tensor] | None = None


def rf_all_atom_kinematics() -> tuple[Tensor, Tensor, Tensor]:
    r"""
    Return the RFdiffusion residue-frame kinematics constants for atom14 reconstruction.

    Decompresses and caches the embedded RoseTTAFold all-atom tables on first call, so subsequent calls are
    free. The tables drive [`ComputeAllAtomCoords`][ofoldx.models.rfdiffusion.model.ComputeAllAtomCoords]
    when it rebuilds sidechain atoms from backbone frames and torsion angles.

    Returns:
        A tuple ``(base_indices, RTs_by_torsion, xyzs_in_base_frame)``: the per-atom frame indices
        (``long``), the rigid transforms per torsion group (``float``), and the reference atom coords in each
        base frame (``float``).

    Examples:
        >>> from ofoldx.models.rfdiffusion.chemical import rf_all_atom_kinematics
        >>> base_indices, rts, xyzs = rf_all_atom_kinematics()  # doctest: +SKIP
        >>> base_indices.dtype  # doctest: +SKIP
        torch.int64
    """

    global _RF_ALL_ATOM_CACHE
    if _RF_ALL_ATOM_CACHE is None:
        payload = zlib.decompress(base64.b64decode(_RF_ALL_ATOM_KINEMATICS))
        tensors = torch.load(io.BytesIO(payload), weights_only=True)
        _RF_ALL_ATOM_CACHE = (
            tensors["base_indices"].long(),
            tensors["RTs_by_torsion"].float(),
            tensors["xyzs_in_base_frame"].float(),
        )
    return _RF_ALL_ATOM_CACHE
