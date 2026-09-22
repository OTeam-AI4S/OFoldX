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
``Structure`` → PDB records through ``RECORDS``: every field is placed at the columns the decoder reads
it from, vectorised over the rows of a record.

PDB carries no fact state beyond a value or a blank and no private record, so the write reports every fact it cannot
restore (``unencodable``), written as the nearest thing the format can say: a number its decimals
round, a state other than the one a written value or a blank re-reads as (``basis``), a coordinate that is not
observed, a non-deposited copy, a configuration the reader would not re-derive, a non-archive definition, a connection
kind, order or direction its ``SSBOND``/``LINK`` does not re-read as, a connection held from one side only or whose
transform has no code (an ``SSBOND``'s unstated one is its blank symmetry), a consumer assumption, a literal ``?``/``.``
identifier, a parent name ``MODRES`` cannot hold (its line left out), the id of an entity that is no polymer or
whose id is no ``MOL_ID`` (the reader derives that entity). An identifier the format requires whose evidence is
unknown (chain, the residue number of a residue some line names) refuses (``write_identity_unknown``) unless
``identity="assign"``; a known one its columns cannot hold (a multi-character chain unless ``chains=`` maps it,
a residue name, number or insertion code, an atom name, altloc or serial), which a cut would turn into another
identifier, or a residue address another component holds, or a chain another polymer or water instance holds, refuses
(``write_identifier_unfit``), and so does a number wider than its field (a model serial, a sequence length, a
coordinate, occupancy, B, charge or cell): no field is ever cut, each width is the field's in ``RECORDS``. A
``REMARK 465`` row leaves out a model its three columns cannot hold, and an ``SSBOND`` its line number past 999: a
row stating no model applies to the file's one model, and the reader takes no meaning from the line number.
A source asym (``asym``, the mmCIF ``label_asym_id``) and a polymer type or ``nstd_linkage`` state other than those a
re-read derives are reported per instance; a fact lost on atoms or components (a rounded coordinate, ``width``; the
unit of an exact one printed at a unit other than 0.001 Å, ``coordinate_quantum``; an unplaced atom; a non-archive
definition) once per instance holding such ones, with their count.
"""

from __future__ import annotations

import re
import textwrap
from collections.abc import Mapping
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

from ...chemistry.definitions import ComponentLibrary
from ...data.molecular import ELEMENTS, AtomSite, BondKind, BondOrigin, EntityType
from ...data.structure import CoordinateEvidence, CopyKind, OperationNumbering, ResolutionKind, Structure
from ...errors import RequestRef, fail
from ...provenance import Basis, Records, Resolution, SubjectKind, basis, known, resolution, state
from ..binding import derived_polymer_type
from ..symmetry import SymOp, triplet
from ..writing import (
    Notes,
    archive_library,
    assumptions,
    charge_cell,
    one_way,
    settle,
    unarchived,
    undetermined,
    unframed,
    unplaced,
    untabled,
    written_transform,
)
from .records import RECORDS

Identity = Literal["source", "assign"]

_KD = state(Resolution.KNOWN, Basis.DECLARED)
_UD = state(Resolution.UNKNOWN, Basis.DECLARED)
_UN = state(Resolution.UNKNOWN, Basis.NONE)
_UDER = state(Resolution.UNKNOWN, Basis.DERIVED)
_KDEF = state(Resolution.KNOWN, Basis.DEFINED)
_KDER = state(Resolution.KNOWN, Basis.DERIVED)
_ID = state(Resolution.INAPPLICABLE, Basis.DECLARED)  # a blank field of an identifier: the stated none
_IDEF = state(Resolution.INAPPLICABLE, Basis.DEFINED)
_S = ELEMENTS.index("S")
_LINKED = (BondOrigin.CONNECTION, BondOrigin.RECIPE, BondOrigin.BRANCH_LINK)
_FIELDS = {record: {column.name: column for column in columns} for record, columns in RECORDS.items()}
_NUMBERS = ("int", "float", "hybrid36")
_WATER = ("HOH", "DOD")  # the residue names the reader takes as a chain's waters


def pdb_capability(
    structure: Structure,
    *,
    identity: Identity,
    chains: Mapping[str, str] | None,
    library: ComponentLibrary | None = None,
) -> Records:
    r"""The ``unencodable`` records of the facts PDB cannot carry from ``structure`` (empty: a file carries all)."""
    return _plan(structure, identity, chains, library)[1].records()


def encode_pdb(
    structure: Structure,
    *,
    identity: Identity,
    chains: Mapping[str, str] | None,
    library: ComponentLibrary | None = None,
) -> tuple[bytes, Records]:
    r"""
    The PDB bytes of ``structure`` and the records of the write (facts not carried, identifiers assigned); a
    definition equal to ``library``'s entry is the archive's (``unarchived``).

    Raises:
        UnsupportedError: ``write_identity_unknown`` for an identifier the format requires that is unknown (under
            ``identity="source"``); ``write_identifier_unfit`` for one, or a number, wider than its columns.
        InvalidRequestError: ``value_invalid`` for a value of several systems.
    """
    lines, notes = _plan(structure, identity, chains, library)
    records = settle(notes)
    return ("\n".join(lines) + "\n").encode("ascii"), records


def _owner(pointer: NDArray[np.int64]) -> NDArray[np.int64]:
    return np.repeat(np.arange(len(pointer) - 1, dtype=np.int64), np.diff(pointer))


def _symbols(table: tuple[str, ...], codes: NDArray) -> NDArray:
    return np.asarray((*table, ""), dtype=object)[np.asarray(codes, dtype=np.int64)]


def _lines(record: str, n: int, fields: Mapping[str, NDArray]) -> list[str]:
    r"""``n`` lines of ``record`` with each field placed at its ``RECORDS`` columns: numbers and residue names right-,
    other text left-justified; a field wider than its columns is cut, never published (the caller has recorded it
    ``unfit``, which refuses the write)."""
    matrix = np.full((n, 80), ord(" "), dtype=np.uint8)
    head = record.encode()
    matrix[:, : len(head)] = np.frombuffer(head, np.uint8)
    for name, values in fields.items():
        column = _FIELDS[record][name]
        width = column.stop - column.start
        values = np.asarray(values, dtype=str).astype(f"S{width}")
        right = column.kind in _NUMBERS or name.startswith("resName")
        values = np.strings.rjust(values, width) if right else np.strings.ljust(values, width)
        matrix[:, column.start : column.stop] = np.ascontiguousarray(values).view(np.uint8).reshape(n, width)
    return [line.decode().rstrip() for line in matrix.view("S80").reshape(n).tolist()]


def _fixed(
    notes: Notes, values: NDArray[np.float64], form: str, field: tuple[str, str], item: str, kind, subjects
) -> tuple[NDArray, NDArray[np.bool_]]:
    r"""``values`` printed with ``form`` and whether each reads back exactly; one wider than ``field``'s columns
    (``RECORDS``: record, field) is ``unfit`` (cut, it would state another number), one only rounded is not exact."""
    texts = np.char.mod(form, values) if len(values) else np.zeros(0, dtype=str)
    fits = _fit(notes, *field, item, kind, subjects, texts)
    exact = fits & (texts.astype(np.float64) == values) if len(values) else fits
    return texts, exact


class _Value:
    r"""The indices and PDB identifiers of one single-system ``Structure``."""

    def __init__(self, structure: Structure, identity: Identity, chains: Mapping[str, str] | None, notes: Notes):
        if len(structure.system_model_number) != 1:
            fail("value_invalid", RequestRef("structure", f"{len(structure.system_model_number)} systems"))
        self.structure, self.notes = structure, notes
        system = self.system = structure.system
        sym, tsym = system.symbols(), structure.symbols()
        self.atom_component = _owner(system.component_atom_ptr)
        self.component_instance = _owner(system.instance_component_ptr)
        self.component_entity = system.instance_entity.astype(np.int64)[self.component_instance]
        self.ordinal = np.arange(len(self.component_instance)) - system.instance_component_ptr[self.component_instance]
        self.instance_name = _symbols(sym["instance"], system.instance_symbol)
        self.component_name = _symbols(sym["component"], system.component_symbol)
        self.atom_name = _symbols(sym["atom"], system.atom_symbol)
        kind = np.where(known(system.entity_type_state), system.entity_type, 0)
        self.entity_kind = kind
        self.polymer = kind[self.component_entity] == EntityType.POLYMER
        # the entities ``COMPND`` states as molecules: an id that is a ``MOL_ID`` (a positive integer), polymers only
        entities = _symbols(sym["entity"], system.entity_symbol)
        self.molecule = np.array([e.isascii() and e.isdigit() and int(e) > 0 for e in entities], dtype=bool)
        self.molecule[system.instance_entity[kind[system.instance_entity] != EntityType.POLYMER]] = False
        self.partner = system.atom_index(
            [
                AtomSite(str(i), int(c), str(a))
                for i, c, a in zip(
                    _symbols(sym["instance"], system.external_partner_instance),
                    system.external_partner_component.tolist(),
                    _symbols(sym["atom"], system.external_partner_atom),
                )
            ]
        )
        text = _symbols(tsym["author_number"], structure.component_author_text)

        # the residues a line names: atoms, REMARK 465 (an unplaced polymer residue with an author number), SSBOND/LINK
        # endpoints, MODRES; a residue only SEQRES lists carries no number
        self.placed = np.zeros(len(self.component_instance), dtype=bool)
        self.placed[self.atom_component[structure.coordinate_evidence != CoordinateEvidence.NONE]] = True
        self.unplaced = self.polymer & ~self.placed & (text != "")
        first, second = system.edge_index.astype(np.int64)
        linked = np.isin(system.bond_origin, _LINKED) & (
            _connection_kind(system, system.bond_connection_type) != "CONECT"
        )
        modified = np.diff(system.component_parent_ptr) > 0
        written = self.partner >= 0
        ends = (first[linked], second[linked], system.external_atom[written].astype(np.int64), self.partner[written])
        connected = np.zeros(len(self.component_instance), dtype=bool)
        connected[self.atom_component[np.concatenate(ends)]] = True
        named = self.placed | self.unplaced | modified | connected
        # the narrowest residue-number field of the records naming each residue (REMARK 465's holds one more column)
        width = np.full(len(named), _width("REMARK 465", "seqNum"), dtype=np.int64)
        for naming, record, field in (
            (self.placed, "ATOM", "resSeq"),
            (modified, "MODRES", "seqNum"),
            (connected, "LINK", "resSeq1"),
            (connected, "SSBOND", "seqNum1"),
        ):
            width = np.where(naming, np.minimum(width, _width(record, field)), width)

        # chains: the caller's map, else the author chain, blank where none applies (re-read as the stated none);
        # unknown → refused, or assigned from a 1-character symbol
        chain_state = structure.instance_author_chain_state
        author = _symbols(tsym["chain"], structure.instance_author_chain)
        stated = known(chain_state)
        blank = resolution(chain_state) == Resolution.INAPPLICABLE
        chain = np.where(stated, author, "").astype(object)
        mapped = np.array([chains is not None and name in chains for name in self.instance_name], dtype=bool)
        chain[mapped] = (
            [cast(Mapping[str, str], chains)[name] for name in self.instance_name[mapped]] if mapped.any() else []
        )
        other_basis = ~mapped & blank & (chain_state != _ID)  # a blank re-reads INAPPLICABLE×DECLARED
        notes.unencodable("basis", "chainID", SubjectKind.INSTANCE, np.flatnonzero(other_basis))
        missing = ~mapped & ~stated & ~blank
        if identity == "assign":
            for i in np.flatnonzero(missing).tolist():
                chain[i] = self.instance_name[i]
                notes.assigned("chainID", chain[i], SubjectKind.INSTANCE, i)
        else:
            notes.unencodable("identity", "chainID", SubjectKind.INSTANCE, np.flatnonzero(missing))
        _fit(notes, "ATOM", "chainID", "chainID", SubjectKind.INSTANCE, np.arange(len(chain)), chain)
        # one polymer per chain: the reader takes a chain's SEQRES and ATOM sites as one polymer, of one COMPND
        # molecule; one water instance per chain: it takes a chain's HOH/DOD heterogens as one instance
        water = np.zeros(len(chain), dtype=bool)
        water[self.component_instance[self.placed & ~self.polymer & np.isin(self.component_name, _WATER)]] = True
        for held, limit in (
            (kind[system.instance_entity] == EntityType.POLYMER, "one polymer per chain"),
            (water, "one water instance per chain"),
        ):
            holder: dict[str, int] = {}
            rows = np.flatnonzero(held & (~missing | (identity == "assign"))).tolist()
            shared = [i for i in rows if holder.setdefault(chain[i], i) != i]
            notes.unfit("chainID", limit, SubjectKind.INSTANCE, shared, chain[shared])
        self.chain = chain
        # the polymer instances SEQRES lists: a known sequence, or positions ``identity="assign"`` assigns
        sequenced = self.polymer & (known(structure.component_sequence_number_state) | (identity == "assign"))
        self.sequenced = np.zeros(len(chain), dtype=bool)
        self.sequenced[self.component_instance[sequenced]] = True

        # residue numbers: the author text of a known number, which re-reads as that integer (decimal or hybrid-36, as
        # its source read it); unknown → refused, or the sequence number (else the ordinal + 1)
        integer = known(structure.component_author_number_state)
        number = np.where(integer, text, "").astype(object)
        reading = np.where(integer, structure.component_author_number, 0)  # the integer the reader keys a site by
        unwritable = np.flatnonzero(named & (text != "") & ~integer)
        notes.unfit("resSeq", "an integer", SubjectKind.COMPONENT, unwritable, text[unwritable])
        missing = named & (text == "")
        if identity == "assign":
            sequence = known(structure.component_sequence_number_state)
            fallback = np.where(sequence, structure.component_sequence_number, self.ordinal + 1)
            for c in np.flatnonzero(missing).tolist():
                number[c], reading[c] = str(fallback[c]), fallback[c]
                notes.assigned("resSeq", number[c], SubjectKind.COMPONENT, c)
        else:
            notes.unencodable("identity", "resSeq", SubjectKind.COMPONENT, np.flatnonzero(missing))
        wide = named & (_lengths(number) > width)
        for columns in np.unique(width[wide]).tolist():
            rows = np.flatnonzero(wide & (width == columns))
            notes.unfit("resSeq", f"{columns} characters", SubjectKind.COMPONENT, rows, number[rows])
        self.number = number
        self.insertion = _symbols(tsym["insertion"], structure.component_insertion)
        # one residue per site: the reader takes the residues of one (chainID, resSeq, iCode) as one position's
        # candidates (microheterogeneity), so a second component written there would name the first one's site
        site: dict[tuple, int] = {}
        written = np.flatnonzero(named & (number != ""))
        keys = zip(chain[self.component_instance[written]], reading[written].tolist(), self.insertion[written])
        shared = [c for c, key in zip(written.tolist(), keys) if site.setdefault(key, c) != c]
        notes.unfit("resSeq", "one residue per site", SubjectKind.COMPONENT, shared, number[shared])
        for name, texts, subject in (
            ("chainID", self.chain, SubjectKind.INSTANCE),
            ("resName", self.component_name, SubjectKind.COMPONENT),
            ("iCode", self.insertion, SubjectKind.COMPONENT),
        ):
            notes.unencodable("identifier", name, subject, np.flatnonzero(np.isin(texts, ("?", "."))))
        # an identifier a line holds, cut to its columns, would name another residue
        rows = np.flatnonzero(named | self.polymer)  # SEQRES names every polymer residue
        _fit(notes, "ATOM", "resName", "resName", SubjectKind.COMPONENT, rows, self.component_name[rows])
        rows = np.flatnonzero(named)
        _fit(notes, "ATOM", "iCode", "iCode", SubjectKind.COMPONENT, rows, self.insertion[rows])

    def residue(self, components: NDArray[np.int64]) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        r"""resName, chainID, resSeq and iCode of ``components``."""
        return (
            self.component_name[components],
            self.chain[self.component_instance[components]],
            self.number[components],
            self.insertion[components],
        )


def _lengths(texts: NDArray) -> NDArray[np.int64]:
    return np.array([len(t) for t in texts], dtype=np.int64)


def _width(record: str, field: str) -> int:
    r"""The number of columns of one ``RECORDS`` field."""
    column = _FIELDS[record][field]
    return column.stop - column.start


def _fit(notes: Notes, record: str, field: str, item: str, kind, subjects, texts) -> NDArray[np.bool_]:
    r"""Which ``texts`` fit the columns of ``record``'s ``field``; one ``unfit`` record for each other one."""
    texts = np.asarray(texts, dtype=str)
    width = _width(record, field)
    wide = np.char.str_len(texts) > width if len(texts) else np.zeros(0, dtype=bool)
    limit = f"{width} character" + "s" * (width > 1)
    notes.unfit(item, limit, kind, np.asarray(subjects, dtype=np.int64)[wide], texts[wide])
    return ~wide


def _stateful(
    notes: Notes, item: str, states: NDArray[np.uint8], subjects: NDArray[np.int64], kind, value: int = _KD
) -> NDArray:
    r"""Where a field is written blank (UNKNOWN×DECLARED); records every other state but ``value``, the state the reader
    gives a written value (an assumed one is the ``assumed`` records')."""
    assumed = basis(states) == Basis.ASSUMED
    notes.unencodable("basis", item, kind, subjects[~assumed & (states != value) & (states != _UD)])
    return states != value


def _plan(
    structure: Structure, identity: Identity, chains: Mapping[str, str] | None, library: ComponentLibrary | None
) -> tuple[list[str], Notes]:
    notes = Notes(structure)
    value = _Value(structure, identity, chains, notes)
    assumptions(notes, structure)
    lines = _header(value)
    lines += _missing(value)
    lines += _compound(value)
    lines += _sequences(value)
    lines += _modified(value)
    lines += _connections(value)
    lines += _crystal(value)
    atoms, serial = _atoms(value)
    lines += atoms
    lines += _conect(value, serial)
    _unrepresentable(value, library)
    return [*lines, "END"], notes


def _header(value: _Value) -> list[str]:
    r"""``EXPDTA`` (methods joined by ``;``) and ``REMARK 2`` resolutions; another resolution source cannot be named."""
    structure, lines = value.structure, []
    methods = structure.symbols()["method"]
    method = structure.method[structure.system_method_ptr[0] : structure.system_method_ptr[1]]
    text = "; ".join(_symbols(methods, method).tolist())
    for k, chunk in enumerate(textwrap.wrap(text, 69, break_long_words=False)):
        lines.append(f"EXPDTA  {'' if k == 0 else k + 1:>2}{chunk}")
    rows = np.arange(*structure.system_resolution_ptr[:2])
    kinds = structure.resolution_kind[rows]
    remark = rows[kinds == ResolutionKind.PDB_REMARK_2]
    sourced = (ResolutionKind.REFINEMENT, ResolutionKind.REFLECTIONS, ResolutionKind.EM_RECONSTRUCTION)
    other = int(np.isin(kinds, sourced).sum())  # a NONE row states no resolution
    value.notes.unencodable("resolution", "REMARK 2", SubjectKind.NONE, np.full(other, -1))
    inapplicable = state(Resolution.INAPPLICABLE, Basis.DECLARED)
    for r in remark.tolist():
        if structure.resolution_state[r] == _KD:
            lines.append(f"REMARK   2 RESOLUTION. {float(structure.resolution[r])!r} ANGSTROMS.")
        else:  # the remark states a resolution or that none applies
            if structure.resolution_state[r] != inapplicable:
                value.notes.unencodable("basis", "REMARK 2", SubjectKind.NONE, [-1])
            lines.append("REMARK   2 RESOLUTION. NOT APPLICABLE.")
    return lines


def _missing(value: _Value) -> list[str]:
    r"""``REMARK 465``: the polymer residues without a written atom whose author number is known (the reader takes
    their numbers from here)."""
    structure, components = value.structure, np.flatnonzero(value.unplaced)
    if not len(components):
        return []
    res_name, chain, number, insertion = value.residue(components)
    model = str(structure.system_model_number[0]) if structure.system_model_number_state[0] == _KD else ""
    if len(model) > _width("REMARK 465", "model"):  # a row stating no model applies to the file's one model
        model = ""
    fields = {"model": np.full(len(components), model), "resName": res_name, "chainID": chain}
    fields |= {"seqNum": number, "iCode": insertion}
    return ["REMARK 465   M RES C SSSEQI", *_lines("REMARK 465", len(components), fields)]


def _modified(value: _Value) -> list[str]:
    r"""``MODRES``: a line per parent of each residue's own annotation (``component_parent``, occurrence scope), but a
    parent wider than ``stdRes``. A definition's parents are restored by the definitions."""
    system, notes = value.system, value.notes
    owner = _owner(system.component_parent_ptr)
    parents = _symbols(system.symbols()["parent"], system.component_parent)
    wide = _lengths(parents) > _width("MODRES", "stdRes")
    notes.unencodable("width", "stdRes", SubjectKind.COMPONENT, owner[wide])
    owner, parents = owner[~wide], parents[~wide]
    if not len(owner):
        return []
    res_name, chain, number, insertion = value.residue(owner)
    fields = {"resName": res_name, "chainID": chain, "seqNum": number, "iCode": insertion, "stdRes": parents}
    return _lines("MODRES", len(owner), fields)


def _compound(value: _Value) -> list[str]:
    r"""``COMPND`` ``MOL_ID``/``CHAIN`` statements: the entities whose id is a ``MOL_ID`` (a positive integer) and whose
    instances are polymers, with their chains (``NULL`` a blank chain). A ``COMPND`` molecule is the entity of its
    chains' polymers, and PDB requires no molecule statement: the reader derives the entity of every other instance (a
    heterogen, a chain's waters, a chain without ``COMPND``), so any other entity's id is not carried."""
    system, notes = value.system, value.notes
    entities = _symbols(system.symbols()["entity"], system.entity_symbol)
    notes.unencodable("identifier", "COMPND MOL_ID", SubjectKind.NONE, np.full(int((~value.molecule).sum()), -1))
    statements = []
    for e in np.flatnonzero(value.molecule).tolist():
        mol_id, instances = entities[e], np.flatnonzero(system.instance_entity == e)
        names = list(dict.fromkeys(value.chain[i] or "NULL" for i in instances))
        statements.append(f"MOL_ID: {mol_id};")
        if names:
            statements.append(f"CHAIN: {', '.join(names)};")
    lines = []
    for k, chunk in enumerate(textwrap.wrap(" ".join(statements), 70, break_long_words=False)):
        lines.append(f"COMPND {'' if k == 0 else k + 1:>3}{chunk}")
    return lines


def _sequences(value: _Value) -> list[str]:
    r"""
    ``SEQRES``: every component of each polymer instance in order, 13 per line, for an instance whose sequence the
    value knows (a component's sequence number is known; PDB's correspondence leaves a position unknown only inside a
    stated SEQRES). An instance with none (a PDB source without SEQRES, whose order is ATOM order) has no
    SEQRES, unless ``identity="assign"`` assigns its positions (ordinal + 1), each one reported.
    """
    lines = []
    known_number = known(value.structure.component_sequence_number_state)
    for i in np.flatnonzero(value.sequenced).tolist():
        components = np.flatnonzero((value.component_instance == i) & value.polymer)
        if not known_number[components].any():
            for c in components.tolist():
                value.notes.assigned("SEQRES", str(value.ordinal[c] + 1), SubjectKind.COMPONENT, c)
        names = value.component_name[components].tolist()
        _fit(value.notes, "SEQRES", "numRes", "SEQRES numRes", SubjectKind.INSTANCE, [i], [str(len(names))])
        for k in range(0, len(names), 13):  # serNum fits where numRes does (13 names a line)
            chunk = names[k : k + 13]
            fields = {"serNum": [str(k // 13 + 1)], "chainID": [value.chain[i]], "numRes": [str(len(names))]}
            fields |= {f"resName{j + 1}": [name] for j, name in enumerate(chunk)}
            lines += _lines("SEQRES", 1, fields)
    return lines


def _crystal(value: _Value) -> list[str]:
    r"""``CRYST1`` from the cell, ``SCALEn`` from ``writing.written_transform`` (a DERIVED frame is derived again from
    ``CRYST1``; ``writing.unframed``), and the file's own operation table as ``REMARK 290`` (an ITA numbering
    re-resolves from the space-group symbol; a table not in force is none, ``writing.untabled``)."""
    structure, notes, lines = value.structure, value.notes, []
    none = np.array([-1])
    cell_state = structure.cell_state
    if cell_state[0] != _UN or structure.space_group_state[0] != _UN:  # the cell and the group are separate facts
        cell, three = np.where(known(cell_state[0]), structure.cell[0], 0.0), np.full(3, -1)
        lengths, exact_lengths = _fixed(notes, cell[:3], "%.3f", ("CRYST1", "a"), "CRYST1", SubjectKind.NONE, three)
        angles, exact_angles = _fixed(notes, cell[3:], "%.2f", ("CRYST1", "alpha"), "CRYST1", SubjectKind.NONE, three)
        _stateful(notes, "CRYST1", cell_state, none, SubjectKind.NONE)  # a blank cell re-reads UNKNOWN×DECLARED
        units = np.isclose(structure.cell_quantum[0], [1e-3] * 3 + [1e-2] * 3, rtol=1e-9, atol=0)
        units |= not known(cell_state[0])  # a re-read takes the columns' units, which a KNOWN cell's bounds enter
        if not (exact_lengths.all() and exact_angles.all() and units.all()):
            notes.unencodable("width", "CRYST1", SubjectKind.NONE, none)
        group = _symbols(structure.symbols()["space_group"], structure.space_group)
        group_blank = _stateful(notes, "CRYST1 sGroup", structure.space_group_state, none, SubjectKind.NONE)
        _fit(notes, "CRYST1", "sGroup", "CRYST1 sGroup", SubjectKind.NONE, none, np.where(group_blank, "", group))
        cell = np.where(known(cell_state), [*lengths, *angles], "")  # a cell not known is blank, never zeros
        fields = {name: [text] for name, text in zip(("a", "b", "c", "alpha", "beta", "gamma"), cell)}
        fields["sGroup"] = np.where(group_blank, "", group)
        lines += _lines("CRYST1", 1, fields)
    if written_transform(structure):  # [A | u] in the columns' units: 1e-6 and 1e-5
        frame, unit, kind = structure.frame[0], structure.frame_quantum[0], SubjectKind.NONE
        exact = printed = True
        for n in (1, 2, 3):
            scale, fits = _fixed(notes, frame[n - 1, :3], "%.6f", (f"SCALE{n}", f"s{n}1"), "SCALEn", kind, [-1] * 3)
            shift, moves = _fixed(notes, frame[n - 1, 3:], "%.5f", (f"SCALE{n}", f"u{n}"), "SCALEn", kind, none)
            exact &= bool(fits.all() and moves.all())
            printed &= bool(np.isclose(unit[n - 1], [1e-6, 1e-6, 1e-6, 1e-5], rtol=1e-9, atol=0).all())
            fields = {f"s{n}{k}": [scale[k - 1]] for k in (1, 2, 3)} | {f"u{n}": shift}
            lines += _lines(f"SCALE{n}", 1, fields)
        if not exact:  # a re-read states other numbers
            notes.unencodable("width", "SCALEn", SubjectKind.NONE, none)
        if not printed:  # a re-read bounds by another resolution (a Cartn_transf frame's propagated units)
            notes.unencodable("frame", "SCALEn", SubjectKind.NONE, none)
    unframed(notes, structure, "SCALEn")
    if structure.operation_numbering[0] == OperationNumbering.FILE:
        rows = slice(*structure.operation_ptr[:2])
        ids = _symbols(structure.symbols()["operation"], structure.operation_id[rows])
        numeric = np.array([i.isdigit() for i in ids], dtype=bool)
        notes.unencodable("operation_table", "REMARK 290", SubjectKind.NONE, np.full(int((~numeric).sum()), -1))
        for i, r, t in zip(ids, structure.operation_rotation[rows], structure.operation_translation[rows]):
            text = triplet(SymOp(tuple(r.reshape(9).tolist()), tuple(t.tolist()))).upper()
            lines.append(f"REMARK 290 {i + '555':>10}   {text}")
    untabled(notes, structure, "REMARK 290")  # a table not in force is written as none
    return lines


def _atoms(value: _Value) -> tuple[list[str], NDArray[np.int64]]:
    r"""``MODEL``, ``ATOM``/``HETATM`` and ``TER`` lines; per atom its serial (−1 not written)."""
    structure, system, notes = value.structure, value.system, value.notes
    evidence = structure.coordinate_evidence
    notes.unencodable("coordinates", "ATOM", SubjectKind.ATOM, unplaced(structure))
    atoms = np.flatnonzero(evidence != CoordinateEvidence.NONE)
    notes.unencodable(
        "coordinate_evidence", "ATOM", SubjectKind.ATOM, atoms[evidence[atoms] != CoordinateEvidence.OBSERVED]
    )
    n = len(atoms)
    serial = np.full(len(evidence), -1, dtype=np.int64)
    serial[atoms] = np.arange(1, n + 1)
    each = (SubjectKind.ATOM, atoms)
    _fit(notes, "ATOM", "serial", "serial", *each, serial[atoms])
    c = value.atom_component[atoms]
    res_name, chain, number, insertion = value.residue(c)

    coordinates, exact = [], np.ones(n, dtype=bool)
    for i, axis in enumerate("xyz"):
        texts, ok = _fixed(notes, structure.coordinates[atoms, i], "%.3f", ("ATOM", axis), axis, *each)
        coordinates.append(texts)
        exact &= ok
    notes.unencodable("width", "x y z", SubjectKind.ATOM, atoms[~exact])
    # an exact value printed at another unit keeps its value and re-reads at 0.001 Å: its printed unit is lost
    unit = np.isclose(structure.coordinate_quantum[atoms], 1e-3, rtol=1e-6, atol=0)  # float32
    notes.unencodable("coordinate_quantum", "x y z", SubjectKind.ATOM, atoms[exact & ~unit])
    observations = {}
    for name, column in (("occupancy", "occupancy"), ("tempFactor", "b_factor")):
        blank = _stateful(notes, name, structure.arrays()[f"{column}_state"][atoms], atoms, SubjectKind.ATOM)
        values = np.where(blank, 0.0, structure.arrays()[column][atoms])  # a blank field states no number
        texts, ok = _fixed(notes, values, "%.2f", ("ATOM", name), name, *each)
        notes.unencodable("width", name, SubjectKind.ATOM, atoms[~blank & ~ok])
        observations[name] = np.where(blank, "", texts)
    su = structure.occupancy_uncertainty_state[atoms]  # PDB states no su (the reader's UNKNOWN×NONE); none observed
    stated_su = (su != _UN) & (su != state(Resolution.INAPPLICABLE, Basis.DEFINED))
    notes.unencodable("uncertainty", "occupancy", SubjectKind.ATOM, atoms[stated_su])

    charge, charge_state = charge_cell(notes, structure, atoms, "charge")
    charge = charge.astype(np.int64)
    blank = _stateful(notes, "charge", charge_state, atoms, SubjectKind.ATOM)
    charges = np.where(blank, "", np.char.add(np.abs(charge).astype(str), np.where(charge < 0, "-", "+")))
    _fit(notes, "ATOM", "charge", "charge", *each, charges)

    z, isotope = system.atomic_number[atoms], system.isotope[atoms]
    element = np.char.upper(np.array(ELEMENTS)[np.where(known(system.atomic_number_state[atoms]), z, 0)])
    heavy = (z == 1) & np.isin(isotope, (2, 3))
    element = np.where(heavy, np.where(isotope == 2, "D", "T"), element)
    notes.unencodable("isotope", "element", SubjectKind.ATOM, atoms[~heavy & (isotope > 0)])
    notes.unencodable("element", "element", SubjectKind.ATOM, atoms[~known(system.atomic_number_state[atoms])])

    names = value.atom_name[atoms]
    _fit(notes, "ATOM", "name", "name", *each, names)
    altloc = _symbols(structure.symbols()["altloc"], structure.atom_altloc[atoms])
    _fit(notes, "ATOM", "altLoc", "altLoc", *each, altloc)
    for name, texts in (("name", names), ("altLoc", altloc)):
        notes.unencodable("identifier", name, SubjectKind.ATOM, atoms[np.isin(texts, ("?", "."))])
    # the PDB alignment: a name of fewer than four characters with a one-letter element starts in column 14
    aligned = np.where(
        (_lengths(names) < 4) & (np.char.str_len(element.astype(str)) == 1), np.char.add(" ", names.astype(str)), names
    )
    polymer = value.polymer[c]
    fields = {
        "serial": serial[atoms],
        "name": aligned,
        "altLoc": altloc,
        "resName": res_name,
        "chainID": chain,
        "resSeq": number,
        "iCode": insertion,
        "x": coordinates[0],
        "y": coordinates[1],
        "z": coordinates[2],
        "occupancy": observations["occupancy"],
        "tempFactor": observations["tempFactor"],
        "element": element,
        "charge": charges,
    }
    atom_lines = np.array(_lines("ATOM", n, fields), dtype=object)
    atom_lines[~polymer] = ["HETATM" + line[6:] for line in atom_lines[~polymer]]
    lines = []
    instance = value.component_instance[c]
    last = np.flatnonzero(polymer & np.append(instance[1:] != instance[:-1], True))  # TER after each polymer chain
    ends = set(last.tolist())
    for k, line in enumerate(atom_lines.tolist()):
        lines.append(line)
        if k in ends:
            lines.append("TER")
    model = structure.system_model_number_state[0]
    if model == _KD:
        model_serial = [str(structure.system_model_number[0])]
        _fit(notes, "MODEL", "serial", "MODEL serial", SubjectKind.NONE, [-1], model_serial)
        lines = [*_lines("MODEL", 1, {"serial": model_serial}), *lines, "ENDMDL"]
    elif model != _UN and basis(model) != Basis.ASSUMED:  # a MODEL serial re-reads declared, none UNKNOWN×NONE
        notes.unencodable("basis", "MODEL", SubjectKind.NONE, [-1])
    return lines, serial


def _connection_kind(system, codes: NDArray) -> NDArray:
    r"""The record of each connection: ``SSBOND`` for a disulfide, ``CONECT`` for an unclassified CONECT connection,
    else ``LINK``."""
    names = _symbols(system.symbols()["connection_type"], codes)
    return np.where(np.isin(names, ("disulf", "SSBOND")), "SSBOND", np.where(names == "CONECT", "CONECT", "LINK"))


def _records(value: _Value, ends: tuple[NDArray, NDArray], codes: NDArray) -> NDArray:
    r"""The record of each connection; ``SSBOND`` names no atom (the reader takes SG): a disulfide between other atoms
    is a ``LINK``."""
    records = _connection_kind(value.system, codes)
    sg = (value.atom_name[ends[0]] == "SG") & (value.atom_name[ends[1]] == "SG")
    return np.where((records == "SSBOND") & ~sg, "LINK", records)


def _reread(
    notes: Notes, records: NDArray, facts: tuple[NDArray, ...], sulfur: NDArray, subjects: NDArray, kind: SubjectKind
) -> None:
    r"""
    Records each fact of a written connection its record does not re-read as: ``SSBOND`` reads as a covalent
    (KNOWN×DEFINED) connection, a single bond (KNOWN×DEFINED) between two known S and no order (UNKNOWN×NONE) between
    any other elements; ``LINK`` as a connection of undeclared kind (UNKNOWN×DECLARED) and no order (UNKNOWN×NONE);
    neither states a direction. An assumed fact is the ``assumed`` records'.

    ``facts`` is (kind, kind state, order, order state, direction state) of each connection; ``sulfur`` whether both
    its endpoints are known S.
    """
    bond_kind, kind_state, order, order_state, direction_state = facts
    ssbond = records == "SSBOND"
    checks = (
        (
            "connection_kind",
            kind_state,
            np.where(ssbond, (kind_state == _KDEF) & (bond_kind == BondKind.COVALENT), kind_state == _UD),
        ),
        ("order", order_state, np.where(ssbond & sulfur, (order_state == _KDEF) & (order == 1), order_state == _UN)),
        ("direction", direction_state, direction_state == _UN),
    )
    for family, states, same in checks:
        lost = ~same & (basis(states) != Basis.ASSUMED)
        for record in ("SSBOND", "LINK"):
            notes.unencodable(family, record, kind, subjects[lost & (records == record)])


def _connections(value: _Value) -> list[str]:
    r"""``SSBOND`` and ``LINK`` of connection edges and external connections (a symmetry partner's code as
    ``NNNKLM``); a junction's declared order cannot be carried."""
    system, structure, notes = value.system, value.structure, value.notes
    first, second = system.edge_index.astype(np.int64)
    junctions = np.flatnonzero((system.bond_origin == BondOrigin.JUNCTION) & (system.bond_order_state == _KD))
    notes.unencodable("order", "LINK", SubjectKind.EDGE, junctions)
    edges = np.flatnonzero(np.isin(system.bond_origin, _LINKED))
    edges = edges[_connection_kind(system, system.bond_connection_type[edges]) != "CONECT"]
    kinds = _records(value, (first[edges], second[edges]), system.bond_connection_type[edges])
    facts = (system.bond_kind, system.bond_kind_state, system.bond_order, system.bond_order_state)
    sulfur = known(system.atomic_number_state) & (system.atomic_number == _S)  # the SSBOND order's element rule
    facts = (*(f[edges] for f in facts), system.bond_direction_state[edges])
    _reread(notes, kinds, facts, sulfur[first[edges]] & sulfur[second[edges]], edges, SubjectKind.EDGE)
    pairs: list[tuple[int, int, str | None, str]] = [
        (int(first[e]), int(second[e]), "", k) for e, k in zip(edges.tolist(), kinds.tolist())
    ]

    # an external connection: its symmetry partner's code as NNNKLM; an SSBOND's unstated transform (UNKNOWN×NONE) as
    # its blank symmetry fields, which re-read so; every other transform without a code (a blank LINK is the identity)
    # and a connection held from one side only are not carried. An end's own frame is written as the identity's code
    # (1555 under the ITA numbering; under a file's own table the id of its x,y,z row, which need not be 1), so a
    # re-read relates the two codes as stored; a file table with no such row carries none
    x = np.arange(len(system.external_atom))
    partner = value.partner
    transform = system.external_transform.astype(np.int64)
    code, transform_state = system.transform_code[transform], system.transform_state[transform]
    names = _symbols(system.symbols()["atom"], system.external_partner_atom)
    sg = (value.atom_name[system.external_atom] == "SG") & (names == "SG")
    records = np.where(_connection_kind(system, system.external_connection_type) == "SSBOND", "SSBOND", "LINK")
    records = np.where((records == "SSBOND") & ~sg, "LINK", records)
    blank = (records == "SSBOND") & (transform_state == _UN)
    unresolvable = (code < 0) | (structure.operation_numbering[0] == OperationNumbering.NONE)
    own = "1555"
    if structure.operation_numbering[0] == OperationNumbering.FILE:
        rows = slice(*structure.operation_ptr[:2])
        ids = _symbols(structure.symbols()["operation"], structure.operation_id[rows])
        rotation, translation = structure.operation_rotation[rows], structure.operation_translation[rows]
        exact = (rotation == np.eye(3)).all(axis=(1, 2)) & ~translation.any(axis=1)
        found = [n for n in ids[exact].tolist() if re.fullmatch("[1-9][0-9]{0,2}", n)]  # NNN of the 6-column code
        own = f"{found[0]}555" if found else own
        unresolvable |= not found
    lost = (partner < 0) | (unresolvable & ~blank)
    x, sided = one_way(structure, partner, x[~lost])
    for record in ("SSBOND", "LINK"):
        dropped = np.union1d(np.flatnonzero(lost & (records == record)), sided[records[sided] == record])
        notes.unencodable("external_connection", record, SubjectKind.EXTERNAL, dropped)
    atoms = system.external_atom[x].astype(np.int64)
    external_kinds = records[x]
    facts = (system.external_kind, system.external_kind_state, system.external_order, system.external_order_state)
    facts = (*(f[x] for f in facts), system.external_direction_state[x])
    _reread(notes, external_kinds, facts, sulfur[atoms] & sulfur[partner[x]], x, SubjectKind.EXTERNAL)
    pairs += [
        (int(a), int(partner[i]), None if blank[i] else f"{code[i] // 1000}{code[i] % 1000:03d}", k)
        for a, i, k in zip(atoms.tolist(), x.tolist(), external_kinds.tolist())
    ]
    lines = []
    for record in ("SSBOND", "LINK"):
        mine = [p for p in pairs if p[3] == record]
        if not mine:
            continue
        ends = np.array([[a, b] for a, b, _, _ in mine], dtype=np.int64).reshape(-1, 2)
        fields: dict[str, NDArray] = {}
        for side in (0, 1):
            res_name, chain, number, insertion = value.residue(value.atom_component[ends[:, side]])
            k = side + 1
            if record == "LINK":
                fields |= {f"name{k}": value.atom_name[ends[:, side]], f"resSeq{k}": number, f"iCode{k}": insertion}
                fields[f"altLoc{k}"] = _symbols(structure.symbols()["altloc"], structure.atom_altloc[ends[:, side]])
                for item in ("name", "altLoc"):  # an unplaced endpoint has no ATOM line checking them
                    _fit(notes, "LINK", f"{item}{k}", item, SubjectKind.ATOM, ends[:, side], fields[f"{item}{k}"])
            else:
                fields |= {f"seqNum{k}": number, f"icode{k}": insertion}
            fields |= {f"resName{k}": res_name, f"chainID{k}": chain}
        symmetry = np.array([s for _, _, s, _ in mine], dtype=object)
        if record == "SSBOND":  # a line number: none past its columns
            serials = np.arange(1, len(mine) + 1).astype(str)
            fields["serNum"] = np.where(np.char.str_len(serials) <= _width("SSBOND", "serNum"), serials, "")
            unstated = np.array([s is None for s in symmetry], dtype=bool)
            fields["sym1"] = np.where(unstated, "", np.where(symmetry == "", "1555", own))  # equal codes: identity
            fields["sym2"] = np.where(unstated, "", np.where(symmetry == "", "1555", symmetry))
        else:
            fields |= {"sym1": np.where(symmetry == "", "", own), "sym2": symmetry}
        lines += _lines(record, len(mine), fields)
    return lines


def _conect(value: _Value, serial: NDArray[np.int64]) -> list[str]:
    r"""``CONECT`` pairs of the connections a source stated only as ``CONECT`` (kind undetermined)."""
    system = value.system
    first, second = system.edge_index.astype(np.int64)
    edges = np.flatnonzero(_connection_kind(system, system.bond_connection_type) == "CONECT")
    unwritten = edges[(serial[first[edges]] < 0) | (serial[second[edges]] < 0)]
    value.notes.unencodable("coordinates", "CONECT", SubjectKind.EDGE, unwritten)
    edges = np.setdiff1d(edges, unwritten)
    fields = {"serial": serial[first[edges]], "bonded1": serial[second[edges]]}
    return _lines("CONECT", len(edges), fields) if len(edges) else []


def _polymer_types(value: _Value, library: ComponentLibrary) -> tuple[NDArray, NDArray[np.uint8]]:
    r"""Per instance the polymer type (text) and state a re-read under ``library`` derives from what is written: an
    instance's SEQRES components derive it (``binding.derived_polymer_type``: KNOWN×DERIVED, UNKNOWN×DERIVED when none
    is determined), none without SEQRES (UNKNOWN×NONE); a ``COMPND`` molecule takes its chains' one type,
    UNKNOWN×DERIVED when they differ."""
    system = value.system
    texts, states = np.full(len(value.chain), "", dtype=object), np.full(len(value.chain), _UN, dtype=np.uint8)
    entries = {cid: library.definition(cid) for cid in set(value.component_name[value.polymer].tolist())}
    definitions = {cid: d for cid, d in entries.items() if d is not None}
    for i in np.flatnonzero(value.sequenced).tolist():
        names = set(value.component_name[(value.component_instance == i) & value.polymer].tolist())
        derived = derived_polymer_type(names, definitions)
        texts[i], states[i] = ("", _UDER) if derived is None else (derived, _KDER)
    for e in np.flatnonzero(value.molecule).tolist():
        instances = np.flatnonzero(system.instance_entity == e)
        types = set(zip(texts[instances].tolist(), states[instances].tolist()))
        texts[instances], states[instances] = types.pop() if len(types) == 1 else ("", _UDER)
    return texts, states


def _unrepresentable(value: _Value, library: ComponentLibrary | None) -> None:
    r"""Facts PDB has no record for: copies other than deposited, a source asym (a re-read names none), a polymer
    type other than the one a re-read derives from what is written (``_polymer_types``: a known type it derives as
    the same type is kept, only its basis now derived; with no library given or configured, by state alone:
    KNOWN×DERIVED or UNKNOWN×DERIVED with SEQRES, UNKNOWN×NONE without) and an ``nstd_linkage`` state other than
    INAPPLICABLE×DEFINED per polymer instance, connections not all represented, configurations not retained from a
    definition (an assumed fact is the ``assumed`` records'), definitions that are not the archive's
    (``unarchived``), folded hydrogens, and atoms whose leaving state is undetermined."""
    structure, system, notes = value.structure, value.system, value.notes
    notes.unencodable(
        "instance", "chainID", SubjectKind.INSTANCE, np.flatnonzero(structure.instance_copy_kind != CopyKind.DEPOSITED)
    )
    notes.unencodable("asym", "chainID", SubjectKind.INSTANCE, np.flatnonzero(structure.instance_source_asym >= 0))
    polymer = value.entity_kind[system.instance_entity] == EntityType.POLYMER
    types, linkage = system.entity_polymer_type_state, system.entity_nstd_linkage_state
    types, linkage = types[system.instance_entity], linkage[system.instance_entity]
    reread = archive_library(library) if polymer.any() else None
    if reread is None:
        typed = np.where(value.sequenced, np.isin(types, (_KDER, _UDER)), types == _UN)
    else:
        texts, derived = _polymer_types(value, reread)
        held = _symbols(system.symbols()["polymer_type"], system.entity_polymer_type)[system.instance_entity]
        same = known(types) & (derived == _KDER) & (held == texts)  # the same specific type, whatever it was based on
        typed = same | ((types == derived) & (derived != _KDER))
    for family, states, kept in (("polymer_type", types, typed), ("nstd_linkage", linkage, linkage == _IDEF)):
        lost = ~kept & (basis(states) != Basis.ASSUMED)
        notes.unencodable(family, "SEQRES", SubjectKind.INSTANCE, np.flatnonzero(polymer & lost))
    open_ = np.flatnonzero(~system.instance_connections_complete)
    notes.unencodable("connections", "LINK", SubjectKind.INSTANCE, open_)
    excluded = (Basis.REFERENCE, Basis.ASSUMED)
    centres = np.flatnonzero(~np.isin(basis(system.stereocentre_state), excluded))
    notes.unencodable("stereo", "ATOM", SubjectKind.STEREOCENTRE, centres)
    bonds = system.stereobond_edge[~np.isin(basis(system.stereobond_state), excluded)]
    notes.unencodable("stereo", "ATOM", SubjectKind.EDGE, bonds)
    notes.unencodable("component_definition", "HETATM", SubjectKind.COMPONENT, unarchived(structure, library))
    hydrogens = known(system.implicit_hydrogens_state) & (system.implicit_hydrogens > 0)
    notes.unencodable("hydrogens", "ATOM", SubjectKind.ATOM, np.flatnonzero(hydrogens))
    undetermined(notes, structure, value.component_instance[value.atom_component], "ATOM")
