# OFoldX
# Copyright (C) 2026-Present  Valhalla Technology

# This file is part of OFoldX.

# OFoldX is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# OFoldX is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# For additional terms and clarifications, please refer to our License FAQ at:
# <https://ofoldx.valtech.ai/about/license-faq/>.

r"""
The fixed-column records of PDB v3.3, used by the decoder and the encoder.

``start``/``stop`` are zero-based and half-open (the format's columns ``start + 1`` to ``stop``); ``blank`` is the state
byte of a blank field: ``INAPPLICABLE×DECLARED`` for chain, altloc and insertion (a known absence),
``UNKNOWN×DECLARED`` for numbers, ``KNOWN×DEFINED`` for a ``LINK`` symmetry (identity), ``UNKNOWN×NONE`` where the
format requires the field (the decoder refuses a blank identity field; a blank ``SSBOND`` symmetry stays not stated).
``column`` names the table column the field fills (``""`` for a field read for its record only).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ...provenance import Basis, Resolution, state

_ColumnKind = Literal["int", "float", "text", "hybrid36"]
_ColumnSpec = tuple[str, int, int, _ColumnKind, int, str]

_ABSENCE = state(Resolution.INAPPLICABLE, Basis.DECLARED)
_UNSTATED = state(Resolution.UNKNOWN, Basis.DECLARED)
_REQUIRED = state(Resolution.UNKNOWN, Basis.NONE)
_IDENTITY = state(Resolution.KNOWN, Basis.DEFINED)  # a blank LINK symmetry field: identity, no lattice translation

#: Every record name of PDB v3.3 (columns 1–6 without trailing blanks): a line naming none is no record.
RECORD_NAMES = frozenset(
    b"HEADER OBSLTE TITLE SPLIT CAVEAT COMPND SOURCE KEYWDS EXPDTA NUMMDL MDLTYP AUTHOR REVDAT SPRSDE JRNL REMARK "
    b"DBREF DBREF1 DBREF2 SEQADV SEQRES MODRES HET HETNAM HETSYN FORMUL HELIX SHEET SSBOND LINK CISPEP SITE CRYST1 "
    b"ORIGX1 ORIGX2 ORIGX3 SCALE1 SCALE2 SCALE3 MTRIX1 MTRIX2 MTRIX3 MODEL ATOM ANISOU TER HETATM ENDMDL CONECT "
    b"MASTER END".split()
)


@dataclass(frozen=True)
class PdbLines:
    r"""Every line of a PDB file: ``record`` its first six bytes without trailing blanks, ``text`` ``[lines, 80]``
    zero-padded, ``line`` its zero-based line number. Re-exported by ``io.source``."""

    record: NDArray[np.bytes_]
    text: NDArray[np.uint8]
    line: NDArray[np.int64]

    def __post_init__(self) -> None:
        for column in (self.record, self.text, self.line):
            column.flags.writeable = False


@dataclass(frozen=True)
class RecordColumn:
    record: str
    name: str
    start: int
    stop: int
    kind: _ColumnKind
    blank: int
    column: str


def _columns(record: str, *rows: _ColumnSpec) -> tuple[RecordColumn, ...]:
    return tuple(
        RecordColumn(record, name, start - 1, stop, kind, blank, column)
        for name, start, stop, kind, blank, column in rows
    )


_ATOM: tuple[_ColumnSpec, ...] = (
    ("serial", 7, 11, "hybrid36", _REQUIRED, "serial"),
    ("name", 13, 16, "text", _REQUIRED, "auth_atom"),
    ("altLoc", 17, 17, "text", _ABSENCE, "label_alt"),
    ("resName", 18, 20, "text", _REQUIRED, "auth_comp"),
    ("chainID", 22, 22, "text", _ABSENCE, "auth_asym"),
    ("resSeq", 23, 26, "hybrid36", _UNSTATED, "auth_seq"),
    ("iCode", 27, 27, "text", _ABSENCE, "insertion"),
    ("x", 31, 38, "float", _UNSTATED, "x"),
    ("y", 39, 46, "float", _UNSTATED, "y"),
    ("z", 47, 54, "float", _UNSTATED, "z"),
    ("occupancy", 55, 60, "float", _UNSTATED, "occupancy"),
    ("tempFactor", 61, 66, "float", _UNSTATED, "b_factor"),
    ("element", 77, 78, "text", _REQUIRED, "element"),
    ("charge", 79, 80, "text", _UNSTATED, "charge"),
)
_LINK_SIDE: tuple[_ColumnSpec, ...] = (  # LINK: one endpoint; second side's columns are 30 later
    ("name", 13, 16, "text", _REQUIRED, "auth_atom"),
    ("altLoc", 17, 17, "text", _ABSENCE, "label_alt"),
    ("resName", 18, 20, "text", _REQUIRED, "auth_comp"),
    ("chainID", 22, 22, "text", _ABSENCE, "auth_asym"),
    ("resSeq", 23, 26, "hybrid36", _UNSTATED, "auth_seq"),
    ("iCode", 27, 27, "text", _ABSENCE, "auth_ins"),
)
_SSBOND_SIDE: tuple[_ColumnSpec, ...] = (  # SSBOND: the second side is 14 columns to the right
    ("resName", 12, 14, "text", _REQUIRED, "auth_comp"),
    ("chainID", 16, 16, "text", _ABSENCE, "auth_asym"),
    ("seqNum", 18, 21, "hybrid36", _UNSTATED, "auth_seq"),
    ("icode", 22, 22, "text", _ABSENCE, "auth_ins"),
)


def _sides(rows: tuple[_ColumnSpec, ...], shift: int) -> tuple[_ColumnSpec, ...]:
    first = tuple((f"{name}1", start, stop, kind, blank, column) for name, start, stop, kind, blank, column in rows)
    second = tuple(
        (f"{name}2", start + shift, stop + shift, kind, blank, column)
        for name, start, stop, kind, blank, column in rows
    )
    return first + second


RECORDS: Mapping[str, tuple[RecordColumn, ...]] = MappingProxyType(
    {
        "HEADER": _columns(
            "HEADER",
            ("classification", 11, 50, "text", _UNSTATED, ""),
            ("depDate", 51, 59, "text", _UNSTATED, "deposition_date"),
            ("idCode", 63, 66, "text", _UNSTATED, "entry_id"),
        ),
        "COMPND": _columns(
            "COMPND", ("continuation", 8, 10, "text", _UNSTATED, ""), ("compound", 11, 80, "text", _UNSTATED, "")
        ),
        "EXPDTA": _columns(
            "EXPDTA", ("continuation", 9, 10, "text", _UNSTATED, ""), ("technique", 11, 79, "text", _UNSTATED, "method")
        ),
        "REVDAT": _columns(
            "REVDAT", ("modNum", 8, 10, "int", _UNSTATED, ""), ("modDate", 14, 22, "text", _UNSTATED, "release_date")
        ),
        "ATOM": _columns("ATOM", *_ATOM),
        "HETATM": _columns("HETATM", *_ATOM),
        "MODEL": _columns("MODEL", ("serial", 11, 14, "int", _REQUIRED, "model")),
        "ENDMDL": (),
        "TER": _columns(
            "TER",
            ("serial", 7, 11, "hybrid36", _UNSTATED, ""),
            ("resName", 18, 20, "text", _UNSTATED, ""),
            ("chainID", 22, 22, "text", _ABSENCE, ""),
            ("resSeq", 23, 26, "hybrid36", _UNSTATED, ""),
            ("iCode", 27, 27, "text", _ABSENCE, ""),
        ),
        "SEQRES": _columns(
            "SEQRES",
            ("serNum", 8, 10, "int", _REQUIRED, ""),
            ("chainID", 12, 12, "text", _ABSENCE, "entity"),
            ("numRes", 14, 17, "int", _REQUIRED, ""),
            *((f"resName{k + 1}", 20 + 4 * k, 22 + 4 * k, "text", _UNSTATED, "mon_id") for k in range(13)),
        ),
        "MODRES": _columns(
            "MODRES",
            ("idCode", 8, 11, "text", _UNSTATED, ""),
            ("resName", 13, 15, "text", _REQUIRED, "comp"),
            ("chainID", 17, 17, "text", _ABSENCE, "asym"),
            ("seqNum", 19, 22, "hybrid36", _REQUIRED, "seq"),
            ("iCode", 23, 23, "text", _ABSENCE, "ins"),
            ("stdRes", 25, 27, "text", _REQUIRED, "parent"),
            ("comment", 30, 70, "text", _UNSTATED, ""),
        ),
        "SSBOND": _columns(
            "SSBOND",
            ("serNum", 8, 10, "int", _UNSTATED, ""),
            *_sides(_SSBOND_SIDE, 14),
            ("sym1", 60, 65, "text", _REQUIRED, "symmetry"),
            ("sym2", 67, 72, "text", _REQUIRED, "symmetry"),
            ("Length", 74, 78, "float", _UNSTATED, ""),
        ),
        "LINK": _columns(
            "LINK",
            *_sides(_LINK_SIDE, 30),
            ("sym1", 60, 65, "text", _IDENTITY, "symmetry"),
            ("sym2", 67, 72, "text", _IDENTITY, "symmetry"),
            ("Length", 74, 78, "float", _UNSTATED, ""),
        ),
        "CONECT": _columns(
            "CONECT",
            ("serial", 7, 11, "hybrid36", _REQUIRED, "serial"),
            *((f"bonded{k + 1}", 12 + 5 * k, 16 + 5 * k, "hybrid36", _UNSTATED, "serial") for k in range(4)),
            *((f"contact{k + 1}", 32 + 5 * k, 36 + 5 * k, "hybrid36", _UNSTATED, "contact") for k in range(6)),
        ),
        "CRYST1": _columns(
            "CRYST1",
            ("a", 7, 15, "float", _UNSTATED, "cell"),
            ("b", 16, 24, "float", _UNSTATED, "cell"),
            ("c", 25, 33, "float", _UNSTATED, "cell"),
            ("alpha", 34, 40, "float", _UNSTATED, "cell"),
            ("beta", 41, 47, "float", _UNSTATED, "cell"),
            ("gamma", 48, 54, "float", _UNSTATED, "cell"),
            ("sGroup", 56, 66, "text", _UNSTATED, "space_group"),
            ("z", 67, 70, "int", _UNSTATED, ""),
        ),
        **{
            f"SCALE{n}": _columns(
                f"SCALE{n}",
                *(
                    (f"s{n}{k}", 11 + 10 * (k - 1), 20 + 10 * (k - 1), "float", _UNSTATED, "deposited")
                    for k in (1, 2, 3)
                ),
                (f"u{n}", 46, 55, "float", _UNSTATED, "deposited"),
            )
            for n in (1, 2, 3)
        },
        "REMARK 465": _columns(
            "REMARK 465",
            ("model", 12, 14, "int", _UNSTATED, "model"),
            ("resName", 16, 18, "text", _REQUIRED, "comp"),
            ("chainID", 20, 20, "text", _ABSENCE, "chain"),
            ("seqNum", 22, 26, "hybrid36", _REQUIRED, "seq"),
            ("iCode", 27, 27, "text", _ABSENCE, "ins"),
        ),
    }
)
