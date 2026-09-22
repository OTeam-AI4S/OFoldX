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
The one writer entry: a ``Structure`` to mmCIF or PDB. Every fact the format can carry is
written and every other one is reported, never stated as something else; an identifier the format requires and the
value does not know is assigned only when the caller asks. The file is published atomically. The reader round trip
(our reader; gemmi and biotite as third-party readers) is a test and corpus property, never control flow; a frame
stated only by ``_atom_sites.Cartn_transf_*`` is outside it, and the writer reports the loss (``frame``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Literal

from ..chemistry.definitions import ComponentLibrary, Definitions
from ..data.structure import Structure
from ..errors import RequestRef, fail
from ..provenance import Records
from .mmcif.encode import block_code, encode_mmcif
from .pdb.encode import encode_pdb
from .writing import publish

_SUFFIXES: dict[str, Literal["mmcif", "pdb"]] = {".cif": "mmcif", ".mmcif": "mmcif", ".pdb": "pdb", ".ent": "pdb"}


@dataclass(frozen=True)
class WriteReport:
    r"""What one write did: ``records`` holds an ``unencodable`` record (family, item) per fact the format did not
    carry, one per instance for those of its atoms or components (count: how many), and an ``identity_assigned``
    record (item, value) per identifier ``identity="assign"`` assigned."""

    path: str
    format: Literal["mmcif", "pdb"]
    records: Records


def write_structure(
    structure: Structure,
    path: str | PathLike,
    *,
    format: Literal["mmcif", "pdb"] | None = None,
    identity: Literal["source", "assign"] = "source",
    chains: Mapping[str, str] | None = None,
    block: str | None = None,
    definitions: Definitions | None = None,
    library: ComponentLibrary | None = None,
) -> WriteReport:
    r"""
    Write ``structure`` to ``path``: every fact the format can carry, ``WriteReport.records`` naming each other one.

    A structure the reader built, written in its source's format, reads back as itself under the same CCD artifact,
    rules and configuration (chemical graph, atom identities, coordinate validity, knowledge states),
    except a frame stated only by ``_atom_sites.Cartn_transf_*``, whose loss is reported (``frame``), and a
    branch link's ``arom``/``delo``/``pi``/``poly`` order, which the value holds as ``?``: every unknown declared
    branch-link order is written ``?`` and reported (``order``), and an atom whose leaving state is undetermined, which
    no item of either format states per atom: written as the value holds it and reported per instance with its
    atoms (``atom_fate``), so a re-read is not guaranteed to restore that state; a
    transformed one (a projection, an edit, a model's output) is written best effort, and the reader may rebuild from
    the CCD what it no longer holds (an atom a projection removed is neither written nor stated as leaving).

    Args:
        format: ``mmcif`` or ``pdb``; by default from the suffix (``.cif``, ``.mmcif``, ``.pdb``, ``.ent``).
        identity: ``source`` writes the identifiers the value carries; ``assign`` also assigns those the format
            requires and the value does not know (author number := sequence number, PDB chain from ``chains`` or a
            one-character instance symbol), each one reported.
        chains: PDB chain per instance symbol.
        block: The mmCIF data block code (default: the file name up to its first ``.``, each character a code cannot
            hold replaced by ``_``, cut to 75 characters).
        definitions: The definitions of the value's components whose definition is not the archive's (mmCIF
            writes them as ``chem_comp*`` and names the CCD artifact they were resolved against; their digests must be
            the components').
        library: The CCD artifact a re-read resolves definitions and PDB polymer types from: a definition equal to
            its entry is the archive's whatever its scope (no ``chem_comp*`` rows, no ``component_definition`` record;
            a re-read gives it archive scope), any other is not; ``configured_library()`` when None. With none given
            or configured, or (mmCIF) one other than the artifact ``definitions`` were resolved against, every
            definition not of archive scope is written (mmCIF with ``definitions``) or reported as not the archive's;
            with none given or configured, a PDB polymer type is checked by its state alone: a write never requires
            an artifact.

    Raises:
        UnsupportedError: ``write_identity_unknown`` for an identifier the format requires that the value does not
            know (under ``identity="source"``); ``write_identifier_unfit`` for a known one, or a number, the format's
            columns cannot hold (a cut would name another chain, residue or atom, or state another number). Nothing is
            written.
        InvalidRequestError: ``format_unknown`` for an unrecognised suffix or a ``format`` other than ``mmcif`` and
            ``pdb``; ``definition_set_mismatch``; ``value_invalid`` for a ``block`` that is no data block code or an
            ``identity`` other than ``source`` and ``assign``.
    """
    path = Path(path)
    if format is None:
        format = _SUFFIXES.get(path.suffix.lower())
        if format is None:
            fail("format_unknown", RequestRef("path", str(path)))
    if format not in ("mmcif", "pdb"):
        fail("format_unknown", RequestRef("format", str(format)))
    if identity not in ("source", "assign"):
        fail("value_invalid", RequestRef("identity", str(identity)))
    if format == "mmcif":
        name = block_code(path.name.split(".")[0]) if block is None else block
        data, records = encode_mmcif(structure, block=name, identity=identity, definitions=definitions, library=library)
    else:
        data, records = encode_pdb(structure, identity=identity, chains=chains, library=library)
    publish(path, data)
    return WriteReport(str(path), format, records)
