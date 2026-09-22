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
Format dispatch and the ``Source`` value: the raw content of one mmCIF block or one PDB file, decoded into the shared
tables on request.

A ``Source`` holds every category of the block as typed columns (mmCIF) or every record line (PDB), uninterpreted.
``Source.tables(model)`` decodes once per source: the all-model ``AtomTable`` and every other table are shared by all
models' views, which differ only in the selected model's cross-category checks (``io.tables.model_view``).
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, cast

import numpy as np

from ..errors import RequestRef, SourceRef, fail
from .cif import CifCategory, TokenState, decompress, parse_cif
from .mmcif.decode import decode_mmcif, mmcif_metadata
from .pdb.decode import decode_pdb, pdb_metadata, pdb_models, split_lines
from .pdb.records import RECORD_NAMES, PdbLines
from .tables import DocumentMetadata, SourceTables, model_view

__all__ = ["PdbLines", "Source", "parse_source", "read_source"]

_BOM = b"\xef\xbb\xbf"  # a UTF-8 byte order mark: no content (CIF 2.0 permits one; editors add one)


@dataclass(frozen=True, eq=False)
class Source:
    r"""
    One source document. ``sha256`` is the digest of the decompressed content (a leading UTF-8 byte order mark
    included; the decoders never see one); ``models`` the ``pdbx_PDB_model_num`` / ``MODEL`` serials in file order
    (``(1,)`` when the format states none).
    """

    path: str | None
    sha256: str
    format: Literal["mmcif", "pdb"]
    block: str | None
    categories: Mapping[str, CifCategory]
    lines: PdbLines | None
    models: tuple[int, ...]
    _cache: dict = field(default_factory=dict, repr=False)

    @property
    def metadata(self) -> DocumentMetadata:
        if "metadata" not in self._cache:
            if self.format == "mmcif":
                self._cache["metadata"] = mmcif_metadata(self.categories)
            else:
                self._cache["metadata"] = pdb_metadata(cast(PdbLines, self.lines))
        return self._cache["metadata"]

    def tables(self, model: int) -> SourceTables:
        r"""The tables of one model's view; the first call decodes, later calls share its tables."""
        base = self._cache.get("tables")
        if base is not None:
            return model_view(base, model)
        if self.format == "mmcif":
            view = decode_mmcif(self.categories, model, block=self.block)
        else:
            view = decode_pdb(cast(PdbLines, self.lines), model)
        self._cache["tables"] = view
        return view


def read_source(
    path: str | os.PathLike[str], *, format: Literal["mmcif", "pdb"] | None = None, block: str | None = None
) -> Source:
    r"""Read one source file, gzip-compressed or not (by content); the format is detected by content when not
    given (``format_unknown`` for a ``format`` other than ``mmcif`` and ``pdb``)."""
    with open(path, "rb") as stream:
        data = stream.read()
    return _source(data, os.fspath(path), format, block)


def parse_source(data: str | bytes, *, format: Literal["mmcif", "pdb"], block: str | None = None) -> Source:
    return _source(data.encode() if isinstance(data, str) else data, None, format, block)


def _source(data: bytes, path: str | None, format: str | None, block: str | None) -> Source:
    if format not in (None, "mmcif", "pdb"):
        fail("format_unknown", RequestRef("format", str(format)))
    data = decompress(data)
    sha256 = hashlib.sha256(data).hexdigest()
    data = data.removeprefix(_BOM)
    format = format or _detect(data)
    if format == "pdb":
        lines = split_lines(data)
        return Source(path, sha256, "pdb", None, {}, lines, pdb_models(lines))
    cif = parse_cif(data)
    if block is None and len(cif.blocks) > 1:
        candidates = [SourceRef("data", i, None, b.name) for i, b in enumerate(cif.blocks)]
        fail("multiple_data_blocks", candidates=candidates)
    if not cif.blocks:
        fail("cif_syntax", detail="no data block")
    chosen = cif.block(block if block is not None else 0)
    models = _cif_models(chosen.categories, chosen.name)
    return Source(path, sha256, "mmcif", chosen.name, chosen.categories, None, models)


def _detect(data: bytes) -> str:
    r"""``mmcif`` when the first statement is a ``data_`` block header, ``pdb`` when the first non-blank line names a
    PDB record in columns 1–6."""
    head = data[:4096]
    if head and (0x80 <= head[0] <= 0x8F or head[0] in (0xDE, 0xDF)):  # a MessagePack map
        fail("format_unsupported", detail="BinaryCIF (MessagePack)")
    for line in head.splitlines():
        text = line.strip()
        if not text or text.startswith(b"#"):
            continue
        if text[:5].lower() == b"data_":
            return "mmcif"
        if line[:6].split(b" ")[0] in RECORD_NAMES:
            return "pdb"
        break
    fail("format_unknown", RequestRef("format", "None"))


def _cif_models(categories: Mapping[str, CifCategory], block: str) -> tuple[int, ...]:
    category = categories.get("atom_site")
    if category is None or not category.nrows:
        return (1,)
    numbers, states = category.ints("pdbx_PDB_model_num")
    if (states == TokenState.ABSENT).all():
        return (1,)
    null = np.flatnonzero(states != TokenState.VALUE)
    if len(null):
        fail("model_number_null", SourceRef("atom_site", int(null[0]), "pdbx_PDB_model_num", block))
    unique, first = np.unique(numbers, return_index=True)
    return tuple(int(number) for number in unique[np.argsort(first)])
