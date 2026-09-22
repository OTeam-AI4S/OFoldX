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
PDB records → the shared source tables, fixed columns read vectorised through ``RECORDS``.

The author namespace is filled and every label column is absent (no author-into-label substitution). A blank chain,
altloc or insertion field is a known absence (INAPPLICABLE×DECLARED), a blank number is UNKNOWN×DECLARED, and a blank
charge field is UNKNOWN×DECLARED: never normalised to an absent item, so the mmCIF default charge 0 never
applies here. ``SEQRES`` gives the sequence of each chain (``SequenceTable.entity`` the chain code), ``COMPND
MOL_ID``/``CHAIN`` the entities and their chains, ``SSBOND`` and ``LINK`` the connections (a blank ``LINK`` symmetry is
the identity, a blank ``SSBOND`` one is not stated), ``REMARK 290`` the file's own operation table by its own
serials, ``REMARK 350`` the declared operators, ``TER`` the lines that end a chain. Polymer types and instances are
derived later (``io.binding``). A line naming no PDB v3.3 record is not read: one ``pdb_line_unrecognised`` record
counts them.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from collections.abc import Callable
from typing import cast

import numpy as np
from numpy.typing import NDArray

from ...chemistry.definitions import DefinitionRows
from ...data.molecular import ELEMENTS
from ...data.structure import ResolutionKind
from ...errors import SourceRef, fail
from ...provenance import Basis, Resolution, state
from ..cif import intern, parse_integers, parse_numbers
from ..tables import (
    AssemblyTable,
    AsymTable,
    AtomTable,
    BranchLinkTable,
    ConectTable,
    ConnectionTable,
    CrystalTable,
    DocumentMetadata,
    EntityTable,
    ExperimentTable,
    ModifiedResidueTable,
    OperatorTable,
    PolymerTable,
    SchemeTable,
    SequenceTable,
    SourceTables,
    TableBuilder,
    UnobservedTable,
    model_view,
)
from .records import RECORD_NAMES, RECORDS, PdbLines, RecordColumn

_KNOWN = state(Resolution.KNOWN, Basis.DECLARED)
_UNSTATED = state(Resolution.UNKNOWN, Basis.DECLARED)
_ABSENCE = state(Resolution.INAPPLICABLE, Basis.DECLARED)
_ABSENT = state(Resolution.UNKNOWN, Basis.NONE)
_DEFINED = state(Resolution.KNOWN, Basis.DEFINED)
_CONFLICT = state(Resolution.CONFLICT, Basis.NONE)
_SSBOND, _LINK = 2, 3  # io.binding.QuerySource
_ELEMENTS = {symbol.upper(): (z, 0) for z, symbol in enumerate(ELEMENTS)} | {"D": (1, 2), "T": (1, 3)}
_UPPER36, _LOWER36 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ", "0123456789abcdefghijklmnopqrstuvwxyz"
_OPERATION = re.compile(rb"\s*(\d{4,})\s+(\S(?:.*\S)?)\s*")  # NNNMMM and the operator as stated, spaces included
_BIOMT = re.compile(rb"\s*BIOMT([123])\s+(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*")
_RESOLUTION = re.compile(rb"\s*RESOLUTION\.\s+(\S+)\s+ANGSTROMS\.?\s*")
_READ = [name.encode() for name in RECORDS if " " not in name]  # the records the decoder reads, and its REMARKs
_REMARKS = [b"  2", b"290", b"350", b"465"]
_FIRST_UNRECOGNISED = 3  # the lines a pdb_line_unrecognised record names


def split_lines(data: bytes) -> PdbLines:
    r"""
    The lines of a PDB file as zero-padded 80-column rows.

    Raises:
        ContradictionError: ``non_ascii_source``; ``pdb_column`` for a line of a record the decoder reads with text
            beyond column 80 (no other line is read, so its width is not checked).
    """
    if not data.isascii():
        fail("non_ascii_source", detail="a PDB file is ASCII text")
    lines = data.splitlines()
    text = np.array(lines, dtype="S80").view(np.uint8).reshape(len(lines), 80)  # columns past 80 cut
    record = np.strings.rstrip(text[:, :6].copy().view("S6").reshape(-1))
    remark = np.isin(text[:, 7:10].copy().view("S3").reshape(-1), _REMARKS) & (record == b"REMARK")
    read = np.isin(record, _READ) | remark
    for i in np.flatnonzero(np.fromiter(map(len, lines), np.int64, len(lines)) > 80).tolist():
        if read[i] and lines[i][80:].strip(b" \0"):
            fail("pdb_column", SourceRef("line", i))
    return PdbLines(record, text, np.arange(len(lines), dtype=np.int64))


def _rows(lines: PdbLines, record: str) -> NDArray[np.int64]:
    r"""The lines of a record; ``"REMARK n"`` selects the remark number in columns 8–10."""
    if record.startswith("REMARK "):
        number = record.split()[1].rjust(3).encode()
        remarks = np.flatnonzero(lines.record == b"REMARK")
        return remarks[lines.text[remarks, 7:10].copy().view("S3").reshape(-1) == number]
    return np.flatnonzero(lines.record == record.encode())


def _field(lines: PdbLines, rows: NDArray[np.int64], column: RecordColumn) -> NDArray[np.bytes_]:
    r"""One fixed-column field of the given lines, blanks stripped (``b""`` for a blank field)."""
    chunk = lines.text[rows, column.start : column.stop]
    chunk = np.where(chunk == 0, ord(" "), chunk).astype(np.uint8)
    return np.strings.strip(np.ascontiguousarray(chunk).view(f"S{column.stop - column.start}").reshape(-1))


def _fields(lines: PdbLines, record: str, rows: NDArray[np.int64]) -> dict[str, NDArray[np.bytes_]]:
    return {column.name: _field(lines, rows, column) for column in RECORDS[record]}


def _where(lines: PdbLines, rows: NDArray[np.int64], record: str, item: str) -> Callable[[int], SourceRef]:
    return lambda i: SourceRef(record, int(lines.line[rows[i]]), item)


def _required(values: NDArray[np.bytes_], where: Callable[[int], SourceRef], code: str = "pdb_column") -> None:
    blank = np.flatnonzero(values == b"")
    if len(blank):
        fail(code, where(int(blank[0])))


def _hybrid36(values: NDArray[np.bytes_], width: int, where: Callable[[int], SourceRef]) -> NDArray[np.int64]:
    r"""Decimal or hybrid-36 integers, 0 for a blank field."""
    stated = values != b""
    first = values.view(np.uint8).reshape(len(values), -1)[:, 0] if len(values) else np.zeros(0, np.uint8)
    decimal = stated & (((first >= ord("0")) & (first <= ord("9"))) | (first == ord("-")) | (first == ord("+")))
    out = parse_integers(values, decimal, where)
    for i in np.flatnonzero(stated & ~decimal):
        text = values[i].decode()
        for offset, digits in enumerate((_UPPER36, _LOWER36)):
            if len(text) == width and text[0] in digits[10:] and all(character in digits for character in text):
                out[i] = int(text, 36) - 10 * 36 ** (width - 1) + 10**width + offset * 26 * 36 ** (width - 1)
                break
        else:
            fail("pdb_hybrid36_invalid", where(int(i)))
    return out


def _numbers(values: NDArray[np.bytes_], where: Callable[[int], SourceRef]) -> tuple[NDArray, NDArray, NDArray]:
    r"""Value, state (UNKNOWN×DECLARED for a blank field) and printed half-unit of a real field."""
    stated = values != b""
    value, _, quantum = parse_numbers(values, stated, where)
    return value, np.where(stated, _KNOWN, _UNSTATED).astype(np.uint8), quantum / 2


def _codes(builder: TableBuilder, family: str, values: NDArray[np.bytes_], blank: int) -> tuple[NDArray, NDArray]:
    stated = values != b""
    codes, symbols = intern(values, stated)
    return builder.code(family, codes, symbols), np.where(stated, _KNOWN, blank).astype(np.uint8)


def _absent(n: int, dtype=np.int32) -> tuple[NDArray, NDArray]:
    return np.full(n, -1, dtype), np.full(n, _ABSENT, np.uint8)


def decode_pdb(lines: PdbLines, model: int) -> SourceTables:
    r"""
    The source tables of one PDB file, viewed for ``model``.

    Raises:
        ContradictionError: ``pdb_column`` for a required field left blank or a malformed one (an atom outside every
            MODEL…ENDMDL block, a REMARK 350 chain list without BIOMT), ``pdb_hybrid36_invalid``, ``number_grammar``,
            ``element_missing``, ``element_unknown``, ``key_invalid`` (a chain in two COMPND molecules, a doubled MODEL
            serial, a CRYST1 or EXPDTA restating another value), the per-atom cross checks of ``model``'s rows.
        UnsupportedError: ``numeric_range``.
        InvalidRequestError: ``model_unknown``.
    """
    builder = TableBuilder(None)
    unrecognised = np.flatnonzero(~np.isin(lines.record, list(RECORD_NAMES)))
    unrecognised = unrecognised[(lines.text[unrecognised] > ord(" ")).any(axis=1)]  # a blank line states nothing
    if len(unrecognised):
        first = builder.refs("line", lines.line[unrecognised[:_FIRST_UNRECOGNISED]])
        builder.record("pdb_line_unrecognised", first, len(unrecognised))
    atoms = _atoms(lines, builder)
    entities, asyms = _compound(lines, builder)
    sequences = _sequences(lines, builder)
    connections = _connections(lines, builder)
    conect = _conect(lines, builder)
    modified = _modified(lines, builder)
    unobserved = _unobserved(lines, builder)
    crystal = _crystal(lines, builder)
    operators, assemblies = _biomolecules(lines, builder)
    experiment = _experiment(lines, builder)
    symbols = builder.symbols()
    refs = builder.source_refs()
    int32, uint8, int64 = np.int32, np.uint8, np.int64
    numbered = (np.char.rstrip(lines.record) == b"MODEL").any()
    tables = SourceTables(
        AtomTable(**atoms, symbols=symbols),
        _KNOWN if numbered else _ABSENT,
        entities,
        PolymerTable(*(np.zeros(0, t) for t in (int32, int32, uint8, uint8, uint8, int64))),
        sequences,
        asyms,
        SchemeTable(*(np.zeros(0, t) for t in (uint8, *(int32,) * 4, int64, uint8, int32, *(int32, uint8) * 2, int64))),
        BranchLinkTable(
            np.zeros(0, int32),
            np.zeros((0, 2), int32),
            np.zeros((0, 2), int32),
            np.zeros((0, 2), int32),
            np.zeros((0, 2), int32),
            np.zeros(0, uint8),
            np.zeros(0, uint8),
            np.zeros(0, int64),
        ),
        connections,
        conect,
        modified,
        unobserved,
        lines.line[_rows(lines, "TER")],
        CrystalTable(**crystal, symbols=symbols),
        operators,
        assemblies,
        experiment,
        DefinitionRows({}),
        refs,
        builder.records(refs),
    )
    return model_view(tables, model)


def pdb_models(lines: PdbLines) -> tuple[int, ...]:
    r"""The ``MODEL`` serials in file order; ``(1,)`` when the file has no MODEL record."""
    return tuple(_serials(lines)[1].tolist()) or (1,)


def _serials(lines: PdbLines) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    r"""The MODEL lines and their serials (``key_invalid`` for a doubled serial)."""
    rows = _rows(lines, "MODEL")
    serials = _fields(lines, "MODEL", rows)["serial"]
    where = _where(lines, rows, "MODEL", "serial")
    _required(serials, where)
    numbers = parse_integers(serials, np.ones(len(rows), bool), where)
    if len(set(numbers.tolist())) != len(numbers):
        fail("key_invalid", where(int(len(numbers) - 1)))
    return rows, numbers


def _models(lines: PdbLines) -> NDArray[np.int64]:
    r"""
    Per line the serial of its MODEL block: 1 when the file has no MODEL record, 0 outside every MODEL…ENDMDL block
    (before the first MODEL, or after an ENDMDL before the next MODEL).
    """
    rows, numbers = _serials(lines)
    if not len(rows):
        return np.ones(len(lines.line), np.int64)
    block = np.full(len(lines.line), -1, np.int64)  # per line the index into [0, *numbers] it opens, -1 where none
    block[rows] = np.arange(1, len(rows) + 1)
    block[_rows(lines, "ENDMDL")] = 0
    position = np.maximum.accumulate(np.where(block >= 0, np.arange(len(block)), -1))
    index = np.where(position >= 0, block[np.maximum(position, 0)], 0)
    return np.concatenate([[0], numbers])[index]


def _atoms(lines: PdbLines, builder: TableBuilder) -> dict:
    rows = np.sort(np.concatenate([_rows(lines, "ATOM"), _rows(lines, "HETATM")]))
    n = len(rows)
    f = _fields(lines, "ATOM", rows)
    record = np.where(lines.record[rows] == b"HETATM", "HETATM", "ATOM")

    def where(item: str) -> Callable[[int], SourceRef]:
        return lambda i: SourceRef(str(record[i]), int(lines.line[rows[i]]), item)

    model = _models(lines)[rows]
    if (model == 0).any():
        fail("pdb_column", where("serial")(int(np.flatnonzero(model == 0)[0])))
    for name in ("serial", "name", "resName"):
        _required(f[name], where(name))
    _required(f["element"], where("element"), "element_missing")
    codes, symbols = intern(f["element"], np.ones(n, bool))
    pairs = []
    for i, symbol in enumerate(symbols):
        if symbol.upper() not in _ELEMENTS:
            fail("element_unknown", where("element")(int(np.flatnonzero(codes == i)[0])))
        pairs.append(_ELEMENTS[symbol.upper()])
    elements = np.array([*pairs, (0, 0)], np.int64).reshape(-1, 2)[codes]
    axes = [_numbers(f[axis], where(axis)) for axis in "xyz"]
    coordinate_state = axes[0][1].copy()
    coordinate_state[(axes[1][1] != axes[0][1]) | (axes[2][1] != axes[0][1])] = _CONFLICT
    occupancy, occupancy_state, _ = _numbers(f["occupancy"], where("occupancy"))
    b_factor, b_factor_state, _ = _numbers(f["tempFactor"], where("tempFactor"))
    charge = f["charge"]
    stated = charge != b""
    pair = charge.astype("S2").view(np.uint8).reshape(n, 2)  # a short field NUL-padded: neither digit nor sign
    digit = (pair[:, 0] >= ord("0")) & (pair[:, 0] <= ord("9"))
    malformed = np.flatnonzero(stated & ~(digit & np.isin(pair[:, 1], (ord("+"), ord("-")))))
    if len(malformed):
        fail("pdb_column", where("charge")(int(malformed[0])))
    magnitude = np.where(stated, pair[:, 0].astype(np.int64) - ord("0"), 0) * np.where(pair[:, 1] == ord("-"), -1, 1)
    out = {
        "row": lines.line[rows].astype(np.int64),
        "serial": _hybrid36(f["serial"], 5, where("serial")),
        "model": model.astype(np.int32),
        "group": np.where(record == "HETATM", 2, 1).astype(np.uint8),
        "element": elements[:, 0].astype(np.uint8),
        "isotope": elements[:, 1].astype(np.uint16),
        "coordinates": np.stack([axis[0] for axis in axes], axis=1).reshape(n, 3),
        "coordinate_state": coordinate_state,
        "coordinate_quantum": np.max([axis[2] for axis in axes], axis=0).astype(np.float32).reshape(n),
        "occupancy": occupancy,
        "occupancy_state": occupancy_state,
        "occupancy_uncertainty": np.zeros(n),  # PDB states no standard uncertainty
        "occupancy_uncertainty_state": np.full(n, _ABSENT, np.uint8),
        "b_factor": b_factor,
        "b_factor_state": b_factor_state,
        "charge": magnitude.astype(np.int8),
        "charge_state": np.where(stated, _KNOWN, _UNSTATED).astype(np.uint8),
    }
    for column in ("label_asym", "label_entity", "label_comp", "label_atom"):
        out[column], out[f"{column}_state"] = _absent(n)
    out["label_seq"], out["label_seq_state"] = np.zeros(n, np.int64), np.full(n, _ABSENT, np.uint8)
    out["label_alt"], out["label_alt_state"] = _codes(builder, "alt", f["altLoc"], _ABSENCE)
    out["auth_asym"], out["auth_asym_state"] = _codes(builder, "chain", f["chainID"], _ABSENCE)
    out["auth_comp"], out["auth_comp_state"] = _codes(builder, "comp", f["resName"], _ABSENT)
    out["auth_atom"], out["auth_atom_state"] = _codes(builder, "atom", f["name"], _ABSENT)
    out["insertion"], out["insertion_state"] = _codes(builder, "ins", f["iCode"], _ABSENCE)
    out["auth_seq"] = _hybrid36(f["resSeq"], 4, where("resSeq"))
    out["auth_seq_text"], out["auth_seq_state"] = _codes(builder, "number", f["resSeq"], _UNSTATED)
    return out


def _text_lines(lines: PdbLines, record: str, field: str) -> list[tuple[int, str]]:
    r"""A free-text field of each line of a record, with its line number."""
    rows = _rows(lines, record)
    values = _fields(lines, record, rows)[field] if len(rows) else []
    return [(int(lines.line[row]), value.decode()) for row, value in zip(rows, values)]


def _statements(lines: PdbLines) -> list[tuple[int, str, str]]:
    r"""``COMPND`` as (line, token, value) statements: the continued text split at ``;`` (PDB v3.3 specification)."""
    text, starts, lines_of = "", [], []
    for line, value in _text_lines(lines, "COMPND", "compound"):
        starts.append(len(text))
        lines_of.append(line)
        text += value + " "
    out, offset = [], 0
    for statement in text.split(";"):
        token, colon, value = statement.partition(":")
        if colon:
            line = lines_of[bisect_right(starts, offset + len(statement) - len(statement.lstrip())) - 1]
            out.append((line, token.strip(), value.strip()))
        offset += len(statement) + 1
    return out


def _compound(lines: PdbLines, builder: TableBuilder) -> tuple[EntityTable, AsymTable]:
    r"""Entities (``MOL_ID`` with its ``MOLECULE``) and their chains (``CHAIN``; ``NULL`` a blank chain)."""
    entities: list[list] = []  # [id code, description code, line]
    chains: list[tuple[int, int, int]] = []  # chain code, entity code, line
    owner: dict[int, int] = {}
    for line, token, value in _statements(lines):
        if token == "MOL_ID":
            if not value.isdigit() or int(value) < 1:
                fail("pdb_column", SourceRef("COMPND", line, "MOL_ID"))
            entities.append([builder.symbol("entity", value), -1, line])
        elif token == "MOLECULE" and entities:
            entities[-1][1] = builder.symbol("text", value)
        elif token == "CHAIN" and entities:
            for chain in (item.strip() for item in value.split(",") if item.strip()):
                code = -1 if chain == "NULL" else builder.symbol("chain", chain)
                if owner.setdefault(code, entities[-1][0]) != entities[-1][0]:
                    fail("key_invalid", SourceRef("COMPND", line, "CHAIN"))
                chains.append((code, entities[-1][0], line))
    n, m = len(entities), len(chains)
    entity_table = EntityTable(
        np.array([e[0] for e in entities], np.int32).reshape(n),
        np.zeros(n, np.uint8),
        np.full(n, _ABSENT, np.uint8),
        np.array([e[1] for e in entities], np.int32).reshape(n),
        builder.refs("COMPND", [e[2] for e in entities], "MOL_ID"),
    )
    asym_table = AsymTable(
        np.array([c[0] for c in chains], np.int32).reshape(m),
        np.array([c[1] for c in chains], np.int32).reshape(m),
        np.arange(m, dtype=np.int32),
        builder.refs("COMPND", [c[2] for c in chains], "CHAIN"),
    )
    return entity_table, asym_table


def _sequences(lines: PdbLines, builder: TableBuilder) -> SequenceTable:
    r"""``SEQRES``: per chain its components in record order, numbered from 1; ``numRes`` must count them."""
    rows = _rows(lines, "SEQRES")
    f = _fields(lines, "SEQRES", rows)
    where = _where(lines, rows, "SEQRES", "numRes")
    _required(f["numRes"], where)
    declared = parse_integers(f["numRes"], np.ones(len(rows), bool), where)
    chain, _ = _codes(builder, "chain", f["chainID"], _ABSENCE)
    names = np.stack([f[f"resName{k + 1}"] for k in range(13)], axis=1).reshape(len(rows), 13)
    entity, mon_id, line, num = [], [], [], []
    counts: dict[int, int] = {}
    for i, (code, residues) in enumerate(zip(chain.tolist(), names)):
        for residue in residues:
            if residue != b"":
                counts[code] = counts.get(code, 0) + 1
                entity.append(code)
                mon_id.append(residue)
                line.append(int(lines.line[rows[i]]))
                num.append(counts[code])
    for i, code in enumerate(chain.tolist()):
        if declared[i] != counts.get(code, 0):
            fail("pdb_column", where(i))
    values = np.array(mon_id, dtype="S3") if mon_id else np.zeros(0, "S3")
    codes, symbols = intern(values, np.ones(len(values), bool))
    return SequenceTable(
        np.array(entity, np.int32),
        np.array(num, np.int32),
        builder.code("comp", codes, symbols),
        np.zeros(len(values), bool),
        builder.refs("SEQRES", line),
    )


def _symmetry_codes(builder: TableBuilder, values: NDArray[np.bytes_], blank: int) -> tuple[NDArray, NDArray]:
    r"""Six-digit ``NNNMMM`` fields as ``n_klm`` text (any other text kept for the symmetry grammar to judge)."""
    texts = [v.decode() for v in values.tolist()]
    texts = [f"{t[:-3]}_{t[-3:]}" if len(t) >= 4 and t.isdigit() else t for t in texts]
    return _codes(builder, "symmetry", np.array(texts, dtype="S8").reshape(len(texts)), blank)


def _connections(lines: PdbLines, builder: TableBuilder) -> ConnectionTable:
    parts = []
    for record, origin in (("SSBOND", _SSBOND), ("LINK", _LINK)):
        rows = _rows(lines, record)
        n = len(rows)
        f = _fields(lines, record, rows)
        columns = {c.name: c for c in RECORDS[record]}
        part = {
            "origin": np.full(n, origin, np.uint8),
            "ref": builder.refs(record, lines.line[rows]),
            "conn_type": np.full(n, builder.symbol("conn_type", record), np.int32),
            "conn_type_state": np.full(n, _KNOWN, np.uint8),
        }
        field_state = np.zeros((n, 2, 10), np.uint8)
        side: dict[str, list] = {}
        for k in (1, 2):
            for f_index, column in enumerate(("label_asym", "label_seq", "label_comp", "label_atom")):
                side.setdefault(column, []).append(
                    np.zeros(n, np.int64) if column == "label_seq" else np.full(n, -1, np.int32)
                )
                field_state[:, k - 1, f_index] = _ABSENT
            if record == "LINK":
                value, value_state = _codes(builder, "alt", f[f"altLoc{k}"], _ABSENCE)
                atom, atom_state = _codes(builder, "atom", f[f"name{k}"], _ABSENT)
                _required(f[f"name{k}"], _where(lines, rows, record, f"name{k}"))
                number = f[f"resSeq{k}"]
            else:
                value, value_state = _absent(n)
                atom, atom_state = np.full(n, builder.symbol("atom", "SG"), np.int32), np.full(n, _DEFINED, np.uint8)
                number = f[f"seqNum{k}"]
            _required(f[f"resName{k}"], _where(lines, rows, record, f"resName{k}"))
            chain, chain_state = _codes(builder, "chain", f[f"chainID{k}"], _ABSENCE)
            insertion, insertion_state = _codes(
                builder, "ins", f[f"iCode{k}" if record == "LINK" else f"icode{k}"], _ABSENCE
            )
            comp, comp_state = _codes(builder, "comp", f[f"resName{k}"], _ABSENT)
            seq = _hybrid36(number, 4, _where(lines, rows, record, f"resSeq{k}"))
            seq_state = np.where(number != b"", _KNOWN, _UNSTATED).astype(np.uint8)
            side.setdefault("auth_seq_text", []).append(_codes(builder, "number", number, _UNSTATED)[0])
            for f_index, (column, v, st) in enumerate(
                (
                    ("label_alt", value, value_state),
                    ("auth_asym", chain, chain_state),
                    ("auth_seq", seq, seq_state),
                    ("auth_ins", insertion, insertion_state),
                    ("auth_comp", comp, comp_state),
                    ("auth_atom", atom, atom_state),
                ),
                4,
            ):
                side.setdefault(column, []).append(v)
                field_state[:, k - 1, f_index] = st
        for column, values in side.items():
            part[column] = np.stack(values, axis=1).reshape(n, 2)
        part["field_state"] = field_state
        symmetry = [_symmetry_codes(builder, f[f"sym{k}"], columns[f"sym{k}"].blank) for k in (1, 2)]
        part["symmetry"] = np.stack([s[0] for s in symmetry], axis=1).reshape(n, 2)
        part["symmetry_state"] = np.stack([s[1] for s in symmetry], axis=1).reshape(n, 2)
        # neither record states an order: an SSBOND's comes from its endpoints' elements (io.connections)
        part["order"], part["order_state"] = np.zeros(n, np.uint8), np.full(n, _ABSENT, np.uint8)
        part["leaving"], part["leaving_state"] = (np.full((n, 2), -1, np.int32), np.full((n, 2), _ABSENT, np.uint8))
        part["leaving_flag"] = np.zeros(n, np.uint8)
        part["role"], part["role_state"] = (np.full((n, 2), -1, np.int32), np.full((n, 2), _ABSENT, np.uint8))
        part["pdbx_role"], part["pdbx_role_state"] = _absent(n)
        parts.append(part)
    return ConnectionTable(**{column: np.concatenate([part[column] for part in parts]) for column in parts[0]})


def _conect(lines: PdbLines, builder: TableBuilder) -> ConectTable:
    r"""``CONECT`` pairs (the record's atom, each listed atom); reverse duplicates are kept for the reader to merge."""
    rows = _rows(lines, "CONECT")
    f = _fields(lines, "CONECT", rows)
    where = _where(lines, rows, "CONECT", "serial")
    _required(f["serial"], where)
    first = _hybrid36(f["serial"], 5, where)
    pairs, contact, line = [], [], []
    for name, is_contact in [(f"bonded{k + 1}", False) for k in range(4)] + [
        (f"contact{k + 1}", True) for k in range(6)
    ]:
        values = f[name]
        partner = _hybrid36(values, 5, _where(lines, rows, "CONECT", name))
        stated = np.flatnonzero(values != b"")
        pairs.append(np.stack([first[stated], partner[stated]], axis=1).reshape(-1, 2))
        contact.append(np.full(len(stated), is_contact))
        line.append(lines.line[rows[stated]])
    order = np.argsort(np.concatenate(line), kind="stable") if line else np.zeros(0, np.int64)
    serial = np.concatenate(pairs)[order] if pairs else np.zeros((0, 2), np.int64)
    return ConectTable(serial, np.concatenate(contact)[order], builder.refs("CONECT", np.concatenate(line)[order]))


def _modified(lines: PdbLines, builder: TableBuilder) -> ModifiedResidueTable:
    rows = _rows(lines, "MODRES")
    f = _fields(lines, "MODRES", rows)
    for name in ("resName", "seqNum", "stdRes"):
        _required(f[name], _where(lines, rows, "MODRES", name))
    text, text_state = _codes(builder, "number", f["seqNum"], _UNSTATED)
    return ModifiedResidueTable(
        *_absent(len(rows)),
        *_absent(len(rows), np.int64),
        *_codes(builder, "chain", f["chainID"], _ABSENCE),
        _hybrid36(f["seqNum"], 4, _where(lines, rows, "MODRES", "seqNum")),
        text_state,
        text,
        _codes(builder, "ins", f["iCode"], _ABSENCE)[0],
        _codes(builder, "comp", f["resName"], _ABSENT)[0],
        _codes(builder, "comp", f["stdRes"], _ABSENT)[0],
        np.full(len(rows), -1, np.int32),  # PDB states no label part
        builder.refs("MODRES", lines.line[rows]),
    )


def _unobserved(lines: PdbLines, builder: TableBuilder) -> UnobservedTable:
    r"""``REMARK 465`` data lines: those after the ``M RES C SSSEQI`` header."""
    rows = _rows(lines, "REMARK 465")
    header = [i for i, row in enumerate(rows) if b"RES C SSSEQI" in lines.text[row].tobytes()]
    rows = rows[header[0] + 1 :] if header else rows[:0]
    f = _fields(lines, "REMARK 465", rows)
    for name in ("resName", "seqNum"):
        _required(f[name], _where(lines, rows, "REMARK 465", name))
    model = f["model"]
    return UnobservedTable(
        parse_integers(model, model != b"", _where(lines, rows, "REMARK 465", "model")).astype(np.int32),
        _codes(builder, "chain", f["chainID"], _ABSENCE)[0],
        _codes(builder, "comp", f["resName"], _ABSENT)[0],
        _hybrid36(f["seqNum"], 4, _where(lines, rows, "REMARK 465", "seqNum")),
        _codes(builder, "number", f["seqNum"], _UNSTATED)[0],
        _codes(builder, "ins", f["iCode"], _ABSENCE)[0],
        builder.refs("REMARK 465", lines.line[rows]),
    )


def _crystal(lines: PdbLines, builder: TableBuilder) -> dict:
    r"""
    ``CRYST1``, the ``REMARK 290`` operations and ``SCALEn``. ``CRYST1`` states the entry's one cell: a restatement
    of the same cell and space group is the first statement.

    Raises:
        ContradictionError: ``key_invalid`` for a ``CRYST1`` restating another cell or space group.
    """
    rows = _rows(lines, "CRYST1")
    out = {"cell": np.zeros(6), "cell_quantum": np.zeros(6), "cell_state": np.full(6, _ABSENT, np.uint8)}
    out |= {"space_group": np.full(1, -1, np.int32), "space_group_state": np.full(1, _ABSENT, np.uint8)}
    out |= {"space_group_ref": np.full(1, -1, np.int64), "cell_ref": -1}
    out |= {"number": np.zeros(0, np.int64), "number_state": np.zeros(0, np.uint8), "number_ref": np.zeros(0, np.int64)}
    if len(rows):
        f = _fields(lines, "CRYST1", rows)
        other = f["sGroup"] != f["sGroup"][0]
        for i, name in enumerate(("a", "b", "c", "alpha", "beta", "gamma")):
            value, value_state, quantum = _numbers(f[name], _where(lines, rows, "CRYST1", name))
            out["cell"][i], out["cell_state"][i], out["cell_quantum"][i] = value[0], value_state[0], quantum[0]
            other |= (value != value[0]) | (value_state != value_state[0])
        if other.any():
            fail("key_invalid", *(SourceRef("CRYST1", int(lines.line[rows[k]])) for k in (0, np.argmax(other))))
        group, group_state = _codes(builder, "space_group", f["sGroup"], _UNSTATED)
        out["space_group"], out["space_group_state"] = group[:1].astype(np.int32), group_state[:1].astype(np.uint8)
        out["cell_ref"] = builder.ref("CRYST1", int(lines.line[rows[0]]))
        out["space_group_ref"] = np.array([builder.ref("CRYST1", int(lines.line[rows[0]]), "sGroup")], np.int64)
    operations = [
        (int(lines.line[row]), match)
        for row in _rows(lines, "REMARK 290")
        if (match := _OPERATION.fullmatch(lines.text[row, 10:].tobytes().rstrip(b"\0")))
    ]
    out["operation_id"] = np.array(
        [builder.symbol("operation_id", m[1][:-3].decode()) for _, m in operations], np.int32
    )
    out["operation_text"] = np.array([builder.symbol("operation_text", m[2].decode()) for _, m in operations], np.int32)
    out["operation_ref"] = builder.refs("REMARK 290", [line for line, _ in operations])
    frame, quantum, frame_state = np.zeros((2, 3, 4)), np.zeros((2, 3, 4)), np.full((2, 3, 4), _ABSENT, np.uint8)
    deposited = -1
    for i in range(3):
        scale = _rows(lines, f"SCALE{i + 1}")[:1]
        if not len(scale):
            continue
        f = _fields(lines, f"SCALE{i + 1}", scale)
        for j, name in enumerate((f"s{i + 1}1", f"s{i + 1}2", f"s{i + 1}3", f"u{i + 1}")):
            value, value_state, value_quantum = _numbers(f[name], _where(lines, scale, f"SCALE{i + 1}", name))
            frame[0, i, j], frame_state[0, i, j], quantum[0, i, j] = value[0], value_state[0], value_quantum[0]
        if deposited < 0:
            deposited = builder.ref(f"SCALE{i + 1}", int(lines.line[scale[0]]))
    out |= {"deposited_frame": frame, "deposited_quantum": quantum, "deposited_state": frame_state}
    out["deposited_ref"], out["axes_ref"], out["wwpdb_ref"] = np.array([deposited, -1], np.int64), -1, -1
    return out


def _remark_350(lines: PdbLines) -> list[dict]:
    r"""``REMARK 350`` biomolecules: id, and per chain list its line, chains and BIOMT matrices (by serial, rows as
    printed tokens, first line)."""
    molecules: list[dict] = []
    for row in _rows(lines, "REMARK 350"):
        line, text = int(lines.line[row]), lines.text[row, 10:].tobytes().rstrip(b"\0").decode()
        content = text.strip()
        if content.startswith("BIOMOLECULE:"):
            molecules.append({"id": content.partition(":")[2].strip(), "groups": [], "line": line})
        elif not molecules:
            continue
        elif content.startswith("APPLY THE FOLLOWING TO CHAINS:"):
            chains = [c.strip() for c in content.partition(":")[2].split(",") if c.strip()]
            molecules[-1]["groups"].append({"line": line, "chains": chains, "matrices": {}})
        elif content.startswith("AND CHAINS:") and molecules[-1]["groups"]:
            molecules[-1]["groups"][-1]["chains"] += [
                c.strip() for c in content.partition(":")[2].split(",") if c.strip()
            ]
        elif content.startswith("BIOMT") and molecules[-1]["groups"]:
            if (match := _BIOMT.fullmatch(text.encode())) is None:  # a matrix row out of its columns names no operator
                fail("pdb_column", SourceRef("REMARK 350", line, "BIOMT"))
            matrices = molecules[-1]["groups"][-1]["matrices"]
            entry = matrices.setdefault(match[2].decode(), {"rows": {}, "line": line})
            entry["rows"][int(match[1])] = match.group(3, 4, 5, 6)
    return molecules


def _biomolecules(lines: PdbLines, builder: TableBuilder) -> tuple[OperatorTable, AssemblyTable]:
    r"""``REMARK 350`` as declared operators (numbered 1… by first appearance of each distinct printed matrix) and
    assemblies (one generator row per chain list; an empty list joins to ``""``, the blank chain's key)."""
    operators: dict[tuple, int] = {}
    tokens, operator_lines = [], []
    assembly, gen_assembly, gen_expression, gen_asyms, gen_lines = [], [], [], [], []
    for molecule in _remark_350(lines):
        aid = builder.symbol("assembly", molecule["id"])
        assembly.append(aid)
        for group in molecule["groups"]:
            ids = []
            for entry in group["matrices"].values():
                if sorted(entry["rows"]) != [1, 2, 3]:
                    fail("pdb_column", SourceRef("REMARK 350", entry["line"], "BIOMT"))
                key = tuple(entry["rows"][i] for i in (1, 2, 3))
                if key not in operators:
                    operators[key] = len(operators) + 1
                    tokens.append(key)
                    operator_lines.append(entry["line"])
                ids.append(str(operators[key]))
            if not ids:
                fail("pdb_column", SourceRef("REMARK 350", group["line"], "BIOMT"))
            gen_assembly.append(aid)
            gen_expression.append(builder.symbol("text", ids[0] if len(ids) == 1 else f"({','.join(ids)})"))
            gen_asyms.append(builder.symbol("text", ",".join(group["chains"])))
            gen_lines.append(group["line"])
    n = len(tokens)
    flat = np.array([token for key in tokens for row in key for token in row], dtype=np.bytes_).reshape(n * 12)
    value, _, quantum = parse_numbers(
        flat, np.ones(len(flat), bool), lambda i: SourceRef("REMARK 350", operator_lines[i // 12], "BIOMT")
    )
    operator_table = OperatorTable(  # BIOMT states no operator type or name
        np.array([builder.symbol("operator", str(i + 1)) for i in range(n)], np.int32),
        np.full(n, -1, np.int32),
        np.full(n, -1, np.int32),
        value.reshape(n, 3, 4),
        (quantum / 2).reshape(n, 3, 4),
        builder.refs("REMARK 350", operator_lines, "BIOMT"),
    )
    assembly_table = AssemblyTable(
        np.array(assembly, np.int32),
        np.array(gen_assembly, np.int32),
        np.array(gen_expression, np.int32),
        np.array(gen_asyms, np.int32),
        builder.refs("REMARK 350", gen_lines),
    )
    return operator_table, assembly_table


def _experiment(lines: PdbLines, builder: TableBuilder) -> ExperimentTable:
    r"""
    ``EXPDTA`` methods and ``REMARK 2`` resolutions. ``EXPDTA`` states the entry's methods once: a line with a
    continuation number continues the text, which splits at ``;``; another ``EXPDTA`` restating the same methods is
    the first.

    Raises:
        ContradictionError: ``key_invalid`` for an ``EXPDTA`` restating other methods.
    """
    rows = _rows(lines, "EXPDTA")
    f = _fields(lines, "EXPDTA", rows)
    statements: list[list] = []  # [line, text] per record, its continuation lines joined
    for row, continuation, text in zip(rows.tolist(), f["continuation"], f["technique"]):
        if continuation == b"" or not statements:
            statements.append([int(lines.line[row]), text.decode()])
        else:
            statements[-1][1] += " " + text.decode()
    stated = [frozenset(m.strip() for m in text.split(";") if m.strip()) for _, text in statements]
    k = next((k for k, other in enumerate(stated) if other != stated[0]), None)
    if k is not None:
        fail("key_invalid", SourceRef("EXPDTA", statements[0][0]), SourceRef("EXPDTA", statements[k][0]))
    methods = [m.strip() for m in statements[0][1].split(";") if m.strip()] if statements else []
    method = [builder.symbol("text", m) for m in methods]
    resolution, kind, resolution_state = (
        [0.0] * len(methods),
        [ResolutionKind.NONE] * len(methods),
        [_ABSENT] * len(methods),
    )
    ref = list(builder.refs("EXPDTA", [statements[0][0]] * len(methods))) if methods else []
    for row in _rows(lines, "REMARK 2"):
        line, text = int(lines.line[row]), lines.text[row, 10:].tobytes().rstrip(b"\0")
        if b"NOT APPLICABLE" in text and b"RESOLUTION" in text:
            value, value_state = 0.0, state(Resolution.INAPPLICABLE, Basis.DECLARED)
        elif match := _RESOLUTION.fullmatch(text):

            def where(row: int, source_line: int = line) -> SourceRef:
                return SourceRef("REMARK 2", source_line)

            number, _, _ = parse_numbers(np.array([match[1]]), np.ones(1, bool), where)
            value, value_state = float(number[0]), _KNOWN
        else:
            continue
        index = builder.ref("REMARK 2", line, "RESOLUTION")
        if value_state == _KNOWN and not value > 0:  # stated, but no resolution: unknown, never refused
            builder.record("resolution_invalid", [index])
            value_state = _UNSTATED
        method.append(-1)
        resolution.append(value)
        kind.append(ResolutionKind.PDB_REMARK_2)
        resolution_state.append(value_state)
        ref.append(index)
    return ExperimentTable(
        np.array(method, np.int32),
        np.array(resolution, np.float64),
        np.array(kind, np.uint8),
        np.array(resolution_state, np.uint8),
        np.array(ref, np.int64),
    )


def pdb_metadata(lines: PdbLines) -> DocumentMetadata:
    r"""The document facts of a PDB file; dates as written (``DD-MON-YY``)."""
    header = _rows(lines, "HEADER")[:1]
    f = _fields(lines, "HEADER", header) if len(header) else {}

    def head(name: str) -> str | None:
        return (f[name][0].decode() or None) if f else None

    revisions = _rows(lines, "REVDAT")
    revision = _fields(lines, "REVDAT", revisions)
    release = next(
        (date.decode() for number, date in zip(revision["modNum"], revision["modDate"]) if number == b"1"), None
    )
    entities, molecule = [], None
    for _, token, value in _statements(lines):
        if token == "MOL_ID":
            molecule = [value, "", None, None, None]
            entities.append(molecule)
        elif token == "MOLECULE" and molecule is not None:
            molecule[2] = value
    metadata = tuple(cast(tuple[str, str, str | None, str | None, str | None], tuple(e)) for e in entities)
    return DocumentMetadata(head("idCode"), head("depDate"), release, metadata)
