r"""PDBx chemical-component field semantics shared by CCD and structure IO."""

from __future__ import annotations

from typing import Literal

_STEREO_VALUES = {"atom": frozenset({"R", "S"}), "bond": frozenset({"E", "Z"})}


def parse_chemical_stereo(value: str | None, *, category: Literal["atom", "bond"]) -> str | None:
    r"""Parse case-insensitive PDBx ucode; CIF r/s are not pseudoasymmetric CIP.

    The atom item allows N/R/S and the bond item N/E/Z. N denotes no stereo;
    missing values convey no descriptor. Callers retain explicit-N provenance
    separately where their schema distinguishes it from missing evidence.
    """

    token = None if value is None else value.strip().upper()
    if token in {None, "N"}:
        return None
    if token not in _STEREO_VALUES[category]:
        raise ValueError(f"invalid _chem_comp_{category}.pdbx_stereo_config: {value!r}")
    return token


def format_chemical_stereo(value: str | None, *, category: Literal["atom", "bond"]) -> str:
    r"""Encode a canonical descriptor without changing its stereochemical meaning."""

    if value is None:
        return "N"
    # A canonical RDKit r/s has a different meaning from R/S. Uppercasing it
    # here or writing it literally into ucode would silently change chemistry.
    if value not in _STEREO_VALUES[category]:
        raise ValueError(f"mmCIF _chem_comp_{category}.pdbx_stereo_config cannot encode {value!r}")
    return value


def parse_optional_flag(value: str | None, *, item: str) -> bool | None:
    """Decode a CIF Y/N item after lexical nulls have been decoded to None."""
    if value is None:
        return None
    token = value.strip().upper()
    if token not in {"Y", "N"}:
        raise ValueError(f"invalid {item}: {value!r}; expected Y or N")
    return token == "Y"
