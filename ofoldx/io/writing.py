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
What the two encoders share: the outcome records of a write (``unencodable``,
``identity_assigned``), the consumer assumptions no format carries, the atoms whose row states no coordinates, the
components whose definition is not the archive's, the atoms whose leaving state is undetermined, the charge cell of
each row, the external connections a row states, the two refusals (an identifier the value does not know, one the
format cannot hold), and the atomic publish of a written file.
"""

from __future__ import annotations

import os
import secrets
import stat
from collections.abc import Collection
from os import PathLike
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ..chemistry.definitions import ArtifactProvenance, ComponentLibrary, configured_library
from ..data.molecular import AtomFate, DefinitionScope
from ..data.structure import CoordinateEvidence, OperationNumbering, Structure
from ..errors import CODE_NAMES, StructureRef, fail
from ..provenance import Basis, Records, Resolution, SourceRefs, SubjectKind, basis, known, resolution, state
from .symmetry import placeholder_frame, placeholder_transform, space_group_setting

_REF = {SubjectKind.INSTANCE: "instance", SubjectKind.COMPONENT: "component", SubjectKind.ATOM: "atom"}
_REFUSING = frozenset({"identity", "unfit"})  # the families that refuse the write: each names its subjects
_KNOWN_DECLARED = state(Resolution.KNOWN, Basis.DECLARED)
_INAPPLICABLE_DEFINED = state(Resolution.INAPPLICABLE, Basis.DEFINED)
_UNKNOWN_DECLARED = state(Resolution.UNKNOWN, Basis.DECLARED)


def _owner(pointer: NDArray[np.int64]) -> NDArray[np.int64]:
    return np.repeat(np.arange(len(pointer) - 1, dtype=np.int64), np.diff(pointer))


class Notes:
    r"""The ``unencodable`` and ``identity_assigned`` records one encoding of ``structure`` collects, one call per fact
    and item."""

    def __init__(self, structure: Structure) -> None:
        system = structure.system
        component = _owner(system.instance_component_ptr)
        atom = component[_owner(system.component_atom_ptr)]
        self._instance = {SubjectKind.COMPONENT: component, SubjectKind.ATOM: atom}  # each one's instance
        self._parts: list[
            tuple[str, SubjectKind, NDArray[np.int64], str, str, NDArray[np.int64], list[NDArray[np.int64]] | None]
        ] = []
        self._unfit: tuple[StructureRef, str, str, str] | None = None

    def unencodable(self, family: str, item: str, kind: SubjectKind, subjects, atoms=None) -> None:
        r"""``family`` cannot be carried by ``item``: one record per subject, but atoms or components give one per
        instance holding such ones, counting them (``count``), unless the family refuses the write (``identity``,
        ``unfit``: each subject named); ``atoms``, one array per subject, the atoms the record names (none by
        default)."""
        subjects = np.asarray(subjects, dtype=np.int64).reshape(-1)
        counts = np.zeros(len(subjects), dtype=np.int64)
        if kind in self._instance and family not in _REFUSING:
            subjects, counts = np.unique(self._instance[kind][subjects], return_counts=True)
            kind = SubjectKind.INSTANCE
        if len(subjects):
            self._parts.append(("unencodable", kind, subjects, family, item, counts, atoms))

    def unfit(self, item: str, limit: str, kind: SubjectKind, subjects, values) -> None:
        r"""One ``unfit`` record per subject: its identifier or number (``values``, one per subject) exceeds
        ``item``'s ``limit``; the first one named is the refusal's."""
        subjects = np.asarray(subjects, dtype=np.int64).reshape(-1)
        self.unencodable("unfit", item, kind, subjects)
        if len(subjects) and self._unfit is None:
            part = {_REF[kind]: int(subjects[0])} if kind in _REF else {}
            self._unfit = (StructureRef(item=None, **part), item, str(values[0]), limit)

    def assigned(self, item: str, value: str, kind: SubjectKind, subject: int) -> None:
        r"""``identity="assign"`` gave ``subject`` the identifier ``value`` for ``item``."""
        subjects = np.array([subject], dtype=np.int64)
        self._parts.append(("identity_assigned", kind, subjects, item, value, np.zeros(1, np.int64), None))

    def families(self) -> frozenset[str]:
        return frozenset(part[3] for part in self._parts if part[0] == "unencodable")

    def records(self, families: Collection[str] | None = None) -> Records:
        r"""The records, every assignment and the ``unencodable`` ones of ``families`` (all when ``None``)."""
        parts = [p for p in self._parts if p[0] != "unencodable" or families is None or p[3] in families]
        texts: dict[str, int] = {}
        counts = [len(p[2]) for p in parts]
        payload = np.zeros((sum(counts), 4), dtype=np.int64)
        lists = [atoms for p, count in zip(parts, counts) for atoms in (p[6] or [np.zeros(0, np.int64)] * count)]
        start = 0
        for (_, _, _subjects, first, second, tally, _), count in zip(parts, counts):
            payload[start : start + count, 0] = texts.setdefault(first, len(texts))
            payload[start : start + count, 1] = texts.setdefault(second, len(texts))
            payload[start : start + count, 2] = tally
            start += count
        n = len(payload)
        return Records(
            np.repeat([CODE_NAMES.index(p[0]) for p in parts], counts).astype(np.uint16),
            np.repeat([int(p[1]) for p in parts], counts).astype(np.uint8),
            np.concatenate([p[2] for p in parts]) if parts else np.zeros(0, dtype=np.int64),
            np.zeros(n + 1, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            SourceRefs(np.zeros(0, np.int16), np.zeros(0, np.int64), np.zeros(0, np.int16), (), (), None),
            payload,
            np.cumsum([0, *map(len, lists)], dtype=np.int64),
            np.concatenate(lists).astype(np.int64) if lists else np.zeros(0, dtype=np.int64),
            tuple(texts),
        )


_SUBJECTS = {
    "instance": SubjectKind.INSTANCE,
    "component": SubjectKind.COMPONENT,
    "atom": SubjectKind.ATOM,
    "edge": SubjectKind.EDGE,
    "external": SubjectKind.EXTERNAL,
    "stereocentre": SubjectKind.STEREOCENTRE,
}


def assumptions(notes: Notes, structure: Structure) -> None:
    r"""
    One ``assumed`` record (item: the state column) per fact of ``structure`` whose basis is a consumer assumption:
    an assumption is never written as a source statement, whichever item would carry it. A stereobond names
    its edge; a fact of an axis without a subject kind names no subject.
    """
    system = structure.system
    for record in (system, structure):
        columns = record.arrays()
        for field in record.SCHEMA.fields:
            if not field.state:
                continue
            rows = np.flatnonzero(basis(columns[field.name]) == Basis.ASSUMED)
            if field.axis == "stereobond":
                notes.unencodable("assumed", field.name, SubjectKind.EDGE, system.stereobond_edge[rows])
            elif field.axis in _SUBJECTS:
                notes.unencodable("assumed", field.name, _SUBJECTS[field.axis], rows)
            else:
                notes.unencodable("assumed", field.name, SubjectKind.NONE, np.full(len(rows), -1))


def unplaced(structure: Structure) -> NDArray[np.int64]:
    r"""
    The atoms without coordinates whose own row states an observation (occupancy, its su or B): a source row whose
    coordinates are ``?``. A withdrawal of coordinates withdraws the observations too (``replace_coordinates``), never
    the row's altloc or charge cell, so these decide nothing here. mmCIF writes the row with ``?`` coordinates; PDB has
    no such row and reports it.
    """
    stated = np.zeros(len(structure.coordinate_evidence), dtype=bool)
    for name in ("occupancy", "occupancy_uncertainty", "b_factor"):
        stated |= structure.arrays()[f"{name}_state"] != _INAPPLICABLE_DEFINED
    return np.flatnonzero((structure.coordinate_evidence == CoordinateEvidence.NONE) & stated)


def archive_library(library: ComponentLibrary | None) -> ComponentLibrary | None:
    r"""The library a re-read resolves definitions from: ``library``, else ``configured_library()``; None when neither
    is given nor configured (``OFOLDX_CCD_ARTIFACT`` unset or empty): a write never requires an artifact."""
    if library is None and os.environ.get("OFOLDX_CCD_ARTIFACT"):
        return configured_library()
    return library


def unarchived(
    structure: Structure, library: ComponentLibrary | None, named: ArtifactProvenance | None = None
) -> NDArray[np.int64]:
    r"""
    The components whose definition is not the archive's: one whose content (digest) is not that of the entry for its
    id in ``archive_library(library)`` (loaded only for a value holding a definition), whatever its scope: a re-read
    resolves every component from that library, so a definition equal to its entry is the archive's and any other is
    not. With no library given or configured, or one other than ``named`` (the artifact the file names), every
    definition of a scope other than archive counts as not the archive's (compared by scope alone): a write never
    requires an artifact, and a file leaves no definition to an artifact it does not name.
    """
    system = structure.system
    scope = system.component_definition_scope
    defined = np.flatnonzero(scope != DefinitionScope.NONE)
    library = archive_library(library) if len(defined) else None
    if library is None or (
        named is not None and (library.provenance is None or library.provenance.sha256 != named.sha256)
    ):
        return defined[scope[defined] != DefinitionScope.ARCHIVE]
    names = system.text("component", system.component_symbol[defined]).tolist()
    entries = {cid: library.definition(cid) for cid in set(names)}
    archive = np.array([0 if (entry := entries[cid]) is None else entry.digest for cid in names], dtype=np.int64)
    return defined[system.component_definition_digest[defined] != archive]  # a defined component's digest is never 0


def undetermined(notes: Notes, structure: Structure, atom_instance: NDArray[np.int64], item: str) -> None:
    r"""
    One ``atom_fate`` record per instance (``item``: the atom rows) naming its atoms whose leaving state is
    undetermined (``AtomFate.UNDETERMINED``). Neither format states a leaving state per atom: a component
    definition's leaving flag is shared by every residue of the component, and a connection's describes its two sides.
    The atoms are written as the value holds them, never deleted and never flagged as leaving or retained, and the
    definition is not edited, so a re-read is not guaranteed to restore their state. ``atom_instance`` is each atom's
    instance.
    """
    atoms = np.flatnonzero(structure.system.atom_fate == AtomFate.UNDETERMINED)
    instances, starts = np.unique(atom_instance[atoms], return_index=True)  # atoms run instance by instance
    notes.unencodable("atom_fate", item, SubjectKind.INSTANCE, instances, np.split(atoms, starts[1:]))


def charge_cell(notes: Notes, structure: Structure, atoms: NDArray[np.int64], item: str) -> tuple[NDArray, NDArray]:
    r"""
    The ``atom_site`` charge cell of each of the ``atoms`` rows (the raw cell, never the effective charge) as
    the value holds it: a declared formal charge the cell does not state (an edit's, which leaves the source's cell) is
    the cell, and a stated cell the atom's charge no longer declares is ``?``, reported (``charge``). A reader-built
    value declares exactly the charges its cells state.
    """
    system = structure.system
    cell, cell_state = structure.atom_site_charge[atoms], structure.atom_site_charge_state[atoms]
    declared = system.formal_charge_state[atoms] == _KNOWN_DECLARED
    stale = (cell_state == _KNOWN_DECLARED) & ~declared
    notes.unencodable("charge", item, SubjectKind.ATOM, atoms[stale])
    cell = np.where(declared, system.formal_charge[atoms], np.where(stale, 0, cell))
    cell_state = np.where(declared, _KNOWN_DECLARED, np.where(stale, _UNKNOWN_DECLARED, cell_state))
    return cell.astype(np.int8), cell_state.astype(np.uint8)


def one_way(structure: Structure, partner: NDArray[np.int64], rows: NDArray[np.int64]) -> tuple[NDArray, NDArray]:
    r"""
    The external ``rows`` to write, one per connection, and those no row can state: the reader realises both externals
    of a connection from one row, so of one the value also holds from its partner (the partner's external under the
    inverse transform; a transform that is not KNOWN is its own inverse) only the first is written, and one the value
    holds from one side only (an assembly copy's, ``data.assembly``) is not, since its row would add the other side.
    ``partner`` is each external's partner atom (-1 outside the value).
    """
    system = structure.system
    rotation = system.transform_rotation.astype(np.int64)
    translation = system.transform_translation.astype(np.int64)
    atom, transform = system.external_atom.astype(np.int64), system.external_transform.astype(np.int64)

    def key(k: int, inverse: bool) -> tuple:
        t = int(transform[k])
        r, v = rotation[t], translation[t]
        if inverse and known(system.transform_state[t]):
            r = np.rint(np.linalg.inv(r)).astype(np.int64)
            v = -r @ v
        ends = (int(partner[k]), int(atom[k])) if inverse else (int(atom[k]), int(partner[k]))
        return (*ends, int(system.transform_state[t]), r.tobytes(), v.tobytes())

    held, written = {key(k, False) for k in np.flatnonzero(partner >= 0).tolist()}, set()
    keep, sided = np.zeros(len(rows), dtype=bool), np.zeros(len(rows), dtype=bool)
    for i, k in enumerate(rows.tolist()):
        back = key(k, True)
        sided[i] = back not in held
        keep[i] = not sided[i] and key(k, False) not in written
        if keep[i]:
            written.add(back)
    return rows[keep], rows[sided]


def untabled(notes: Notes, structure: Structure, item: str) -> None:
    r"""
    One ``symmetry`` record (``item``: the table) when the source's operator table was not in force:
    numbering NONE beside a space group that names a setting, which only a stated table that is not in force
    leaves (a row of it states no operation, or every id names two; ``io.symmetry.symmetry``). The writers write no
    table and keep the space group: neither the table nor its state is kept, and a re-read may rebuild the operations
    from the space group. No code names the table (its transforms are Unknown), so none is rebound.
    """
    group = structure.space_group[0]
    if structure.operation_numbering[0] != OperationNumbering.NONE or not known(structure.space_group_state[0]):
        return
    if space_group_setting(structure.symbols()["space_group"][group]) is not None:
        notes.unencodable("symmetry", item, SubjectKind.NONE, [-1])


def written_transform(structure: Structure) -> bool:
    r"""Whether the writers write the frame's transform [A | u]: the source's (KNOWN×DECLARED), or the one an
    INAPPLICABLE frame rests on while the value states no cell value (UNKNOWN or INAPPLICABLE). A
    DERIVED frame is derived again from the cell."""
    frame, cell = int(structure.frame_state[0]), resolution(np.uint8(structure.cell_state[0]))
    unstated = cell in (Resolution.UNKNOWN, Resolution.INAPPLICABLE)
    rests = resolution(np.uint8(frame)) == Resolution.INAPPLICABLE and unstated
    return frame == _KNOWN_DECLARED or rests and bool(structure.frame[0].any())


def unframed(notes: Notes, structure: Structure, item: str) -> None:
    r"""
    One ``frame`` record (``item``: the frame's item) when a re-read of what the writers write would state another
    frame: they write the cell and ``written_transform``, and no DERIVED frame, which the reader derives
    from the cell again. A CONFLICT frame is never written, and an UNKNOWN one beside a KNOWN cell would read back
    derived from it. A re-read is INAPPLICABLE exactly where the value states non-crystal methods and its placeholder
    cell (``io.symmetry.placeholder_frame``) or, no cell value written, a written transform of the placeholder metric
    (``io.symmetry.placeholder_transform``), so a KNOWN frame there reads back otherwise too.
    """
    frame, cell = int(structure.frame_state[0]), int(structure.cell_state[0])
    methods = structure.method[structure.system_method_ptr[0] : structure.system_method_ptr[1]]
    texts = [structure.symbols()["method"][k] for k in methods.tolist()]
    if known(np.uint8(cell)):
        placeholder = placeholder_frame(structure.cell[0], texts)
    else:
        placeholder = written_transform(structure) and placeholder_transform(structure.frame[0], texts)
    if resolution(np.uint8(frame)) == Resolution.INAPPLICABLE:
        lost = not placeholder
    else:
        conflict = resolution(np.uint8(frame)) == Resolution.CONFLICT
        lost = conflict or placeholder or not known(np.uint8(frame)) and known(np.uint8(cell))
    if lost:
        notes.unencodable("frame", item, SubjectKind.NONE, [-1])


def settle(notes: Notes) -> Records:
    r"""
    The records a write reports: every fact it does not carry and every identifier it assigned.

    Raises:
        UnsupportedError: ``write_identity_unknown`` for an ``identity`` record, an identifier the format requires that
            the value does not know (the writer never invents one; payload: those records);
            ``write_identifier_unfit`` for an ``unfit`` record, a known identifier or number the format's columns
            cannot hold (the writer never cuts one; payload: the first one's item, value and limit, and those
            records).
    """
    if "identity" in notes.families():
        fail("write_identity_unknown", records=notes.records({"identity"}))
    if notes._unfit is not None:
        ref, item, value, limit = notes._unfit
        fail("write_identifier_unfit", ref, item=item, value=value, limit=limit, records=notes.records({"unfit"}))
    return notes.records()


def publish(path: str | PathLike, data: bytes) -> None:
    r"""Replace ``path`` by ``data`` only once every byte is written (a reader never sees a partial file): a new file
    has the mode the umask (and a default ACL) leaves of 0666, a replaced one keeps its mode."""
    destination = Path(path)
    temporary = destination.with_name(f"{destination.name}.{secrets.token_hex(8)}")
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            if destination.exists():
                os.fchmod(stream.fileno(), stat.S_IMODE(destination.stat().st_mode))
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
