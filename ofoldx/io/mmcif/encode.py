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
``Structure`` → mmCIF through the codec table: per category the rows are gathered with numpy and each
item's ``write`` map turns a fact state into the token the decoder reads back as the same state.

A state no token restores is written as its nearest token (a value, ``?`` or ``.``) plus one ``_ofoldx_fact_basis`` row
naming the category, row, item and the state byte, so a reader knows both what the file states and what the fact's
basis was. Facts of a component or instance repeated on every ``_atom_site`` row carry their basis rows on their scheme
row only. A disulfide's KNOWN×DEFINED order is written ``?`` (the reader re-derives it), the identity relation of an
edge ``.``, the stated ``_atom_site.pdbx_formal_charge`` cell, never the effective charge (a declared charge
the cell does not state is the cell, ``writing.charge_cell``). A column mixing the dictionary default with other
states (its default rows written as the default value with a basis row), a consumer assumption (written as ``?`` or
not at all), and every fact the format cannot carry is reported (``unencodable``; a fact lost on atoms or components
once per instance holding such ones, with their count); one plan serves the capability check and the encoding
(encode once).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ...chemistry.definitions import ChemicalComponent, ComponentLibrary, Definitions, OrderToken, StereoBasis
from ...chemistry.fragments import POLYMER_FAMILIES
from ...data.molecular import ELEMENTS, AtomSite, BondKind, BondOrigin, DefinitionScope, EntityType, ReferenceSource
from ...data.structure import CoordinateEvidence, CopyKind, OperationNumbering, ResolutionKind, Structure
from ...errors import RequestRef, StructureRef, fail
from ...provenance import Basis, Records, Resolution, SubjectKind, basis, determined, known, resolution, state
from ..cif import parse_numbers
from ..symmetry import SymOp, triplet
from ..writing import (
    Notes,
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
from .codec import CONN_TYPES, MMCIF_CODECS, ItemCodec
from .decode import author_readings

Identity = Literal["source", "assign"]

_KD = state(Resolution.KNOWN, Basis.DECLARED)
_KDEF = state(Resolution.KNOWN, Basis.DEFINED)
_KDER = state(Resolution.KNOWN, Basis.DERIVED)
_KDFT = state(Resolution.KNOWN, Basis.DEFAULT)
_KREF = state(Resolution.KNOWN, Basis.REFERENCE)
_UD = state(Resolution.UNKNOWN, Basis.DECLARED)
_UN = state(Resolution.UNKNOWN, Basis.NONE)
_UDER = state(Resolution.UNKNOWN, Basis.DERIVED)
_UREF = state(Resolution.UNKNOWN, Basis.REFERENCE)
_IDEC = state(Resolution.INAPPLICABLE, Basis.DECLARED)
_IDEF = state(Resolution.INAPPLICABLE, Basis.DEFINED)
_FALLBACK = {Resolution.KNOWN: "value", Resolution.UNKNOWN: "?", Resolution.INAPPLICABLE: ".", Resolution.CONFLICT: "?"}
_ORDER_TOKENS = np.array(["", "sing", "doub", "trip", "quad"])
_TOKEN_NAMES = np.array([token.name for token in OrderToken])
_ENTITY_TYPES = np.array(["", "polymer", "non-polymer", "branched", "macrolide", "water"])
_FLAG = np.array(["?", "N", "Y"])
_EVIDENCE = {2: "predicted", 3: "generated", 4: "reference"}
_RESOLUTION = {  # the category each resolution source reads from
    ResolutionKind.REFINEMENT: "refine",
    ResolutionKind.REFLECTIONS: "reflns",
    ResolutionKind.EM_RECONSTRUCTION: "em_3d_reconstruction",
}
_BARE = re.compile(r"[^\s_#$'\"\[\];][^\s]*")
_RESERVED = re.compile(r"(?i)(data_|save_|loop_|global_|stop_)")
_BLOCK = re.compile(r"[!-~]{1,75}")  # a CIF 1.1 data block code: 1-75 non-blank printable ASCII characters

Column = tuple[NDArray, NDArray[np.uint8]]  # texts, states


def mmcif_capability(
    structure: Structure,
    *,
    identity: Identity,
    definitions: Definitions | None = None,
    library: ComponentLibrary | None = None,
) -> Records:
    r"""The ``unencodable`` records of the facts mmCIF cannot carry from ``structure`` (empty: a file carries all)."""
    notes = _plan(structure, "ofoldx", identity, definitions, library)[1]
    return notes.records()


def block_code(name: str) -> str:
    r"""The data block code of a name: each character a code cannot hold replaced by ``_``, cut to 75 characters,
    ``ofoldx`` when nothing is left."""
    return re.sub(r"[^!-~]", "_", name)[:75] or "ofoldx"


def encode_mmcif(
    structure: Structure,
    *,
    block: str,
    identity: Identity,
    definitions: Definitions | None = None,
    library: ComponentLibrary | None = None,
) -> tuple[bytes, Records]:
    r"""
    The mmCIF bytes of ``structure`` and the records of the write (facts not carried, identifiers assigned); a
    definition equal to ``library``'s entry is the archive's (``unarchived``) unless ``definitions`` name another
    artifact, the one the file names.

    Raises:
        UnsupportedError: ``write_identity_unknown`` for a polymer or branched position without a sequence number
            (``identity="source"``).
        InvalidRequestError: ``definition_set_mismatch`` when a supplied definition is not the one a component was
            built from; ``value_invalid`` for a value of several systems or a ``block`` that is no CIF 1.1 data block
            code (1-75 non-blank printable ASCII characters).
    """
    if not _BLOCK.fullmatch(block):
        fail("value_invalid", RequestRef("block", block))
    categories, notes = _plan(structure, block, identity, definitions, library)
    records = settle(notes)
    return _serialize(block, categories), records


class _Plan:
    r"""The categories of one block in writing order and the ``_ofoldx_fact_basis`` rows they need."""

    def __init__(self, notes: Notes) -> None:
        self.notes = notes
        self.categories: list[tuple[str, int, dict[str, tuple[NDArray, NDArray[np.bool_]]]]] = []
        self.basis: list[tuple[str, NDArray[np.int64], str, NDArray[np.uint8]]] = []

    def add(
        self,
        name: str,
        columns: Mapping[str, Column],
        subjects: tuple[SubjectKind | NDArray, NDArray[np.int64]],
        *,
        repeated: frozenset[str] = frozenset(),
    ) -> None:
        r"""
        One category from codec columns. ``subjects`` is the subject kind (one, or one per row) and index of each row;
        ``repeated`` columns restate facts whose basis rows and records another category carries (no basis row, no
        record here).
        """
        n = len(next(iter(columns.values()))[1])
        if not n:
            return
        rows = np.asarray(subjects[1], dtype=np.int64)
        kinds = np.broadcast_to(np.asarray(subjects[0], dtype=np.uint8), rows.shape)
        written: dict[str, tuple[NDArray, NDArray[np.bool_]]] = {}
        for item in MMCIF_CODECS[name].items:
            if item.column not in columns:
                continue
            values, states = columns[item.column]
            tokens = _tokens(item, np.asarray(values, dtype=object), np.asarray(states, dtype=np.uint8))
            if tokens is None:
                continue
            text, null, carried, mixed = tokens
            written[item.item] = (text, null)
            if item.column in repeated:
                continue
            for kind in np.unique(kinds[mixed]).tolist():
                self.notes.unencodable(
                    "default_mixed", f"_{name}.{item.item}", SubjectKind(kind), rows[mixed & (kinds == kind)]
                )
            if carried.any():
                self.basis.append((name, np.flatnonzero(carried), item.item, np.asarray(states)[carried]))
        self.categories.append((name, n, written))

    def finish(self) -> list[tuple[str, int, dict[str, tuple[NDArray, NDArray[np.bool_]]]]]:
        if self.basis:
            names = np.concatenate([np.full(len(rows), name, dtype=object) for name, rows, _, _ in self.basis])
            items = np.concatenate([np.full(len(rows), item, dtype=object) for _, rows, item, _ in self.basis])
            rows = np.concatenate([rows for _, rows, _, _ in self.basis])
            states = np.concatenate([states for _, _, _, states in self.basis])
            columns = {"category": names, "row": rows, "item": items, "state": states}
            self.add("ofoldx_fact_basis", _stated(columns), (SubjectKind.NONE, np.full(len(rows), -1)))
        return self.categories


def _tokens(item: ItemCodec, values: NDArray, states: NDArray[np.uint8]):
    r"""
    The tokens of one item, or None when the item is omitted (every cell reads back from an absent item).

    Returns:
        texts, null mask (an unquoted ``?``/``.``), the rows needing a basis row and the DEFAULT rows of a column that
        also holds other states (written as the default value with a basis row). A consumer assumption is written
        ``?`` without a basis row (its ``assumed`` record is the plan's).
    """
    policy = np.empty(len(states), dtype=object)
    carried = np.zeros(len(states), dtype=bool)
    for s in np.unique(states).tolist():
        rows = states == s
        if basis(np.uint8(s)) == Basis.ASSUMED:
            policy[rows] = "?"
        elif s in item.write:
            policy[rows] = item.write[s]
        else:
            policy[rows], carried[rows] = _FALLBACK[Resolution(int(resolution(np.uint8(s))))], True
    absent = policy == "absent"
    if absent.all():
        return None
    mixed = absent & (states == _KDFT)
    values = values.copy()
    values[mixed], policy[mixed] = item.default, "value"
    policy[absent & ~mixed] = "?"
    carried |= absent
    null = policy != "value"
    return np.where(null, policy, values), null, carried, mixed


def _stated(columns: Mapping[str, NDArray]) -> dict[str, Column]:
    r"""Columns of values every row states (KNOWN×DECLARED)."""
    return {name: (_text(values), np.full(len(values), _KD, np.uint8)) for name, values in columns.items()}


def _text(values) -> NDArray:
    r"""Values as text; floats by their shortest exact decimal (``repr``), so a written value reads back bit-equal."""
    values = np.asarray(values)
    if values.dtype.kind == "f":
        return np.array([repr(v) for v in values.tolist()], dtype=object)
    return values.astype(str).astype(object)


def _printed(values: NDArray, units: NDArray) -> NDArray:
    r"""Floats at the place of their printed unit (``12e2`` above the units digit), so the reader derives that unit
    again; by ``repr`` where the unit is 0 or does not print the value exactly (compared in the values' dtype)."""
    values = np.asarray(values)
    flat, units = values.reshape(-1), np.broadcast_to(units, values.shape).reshape(-1)
    texts = _text(flat)
    places = np.round(np.log10(np.where(units > 0, units, 1))).astype(int)
    for place in np.unique(places[units > 0]).tolist():
        rows = np.flatnonzero((units > 0) & (places == place))
        if place <= 0:
            text = np.char.mod(f"%.{-place}f", flat[rows].astype(np.float64))
        else:
            text = np.char.add(np.char.mod("%.0f", flat[rows] / 10.0**place), f"e{place}")
        exact = text.astype(np.float64).astype(values.dtype) == flat[rows]
        texts[rows[exact]] = text[exact]
    return texts.reshape(values.shape)


def _owner(pointer: NDArray[np.int64]) -> NDArray[np.int64]:
    return np.repeat(np.arange(len(pointer) - 1, dtype=np.int64), np.diff(pointer))


def _symbols(table: tuple[str, ...], codes: NDArray) -> NDArray:
    r"""The symbols of ``codes``; ``""`` for ``-1``."""
    return np.asarray((*table, ""), dtype=object)[np.asarray(codes, dtype=np.int64)]


class _Value:
    r"""The indices and texts of one single-system ``Structure`` every category reads."""

    def __init__(self, structure: Structure, identity: Identity, notes: Notes) -> None:
        if len(structure.system_model_number) != 1:
            fail("value_invalid", RequestRef("structure", f"{len(structure.system_model_number)} systems"))
        self.structure, self.notes = structure, notes
        system = self.system = structure.system
        sym, tsym = system.symbols(), structure.symbols()
        self.atom_component = _owner(system.component_atom_ptr)
        self.component_instance = _owner(system.instance_component_ptr)
        self.instance_entity = system.instance_entity.astype(np.int64)
        self.component_entity = self.instance_entity[self.component_instance]
        self.ordinal = np.arange(len(self.component_instance)) - system.instance_component_ptr[self.component_instance]
        self.instance_name = _symbols(sym["instance"], system.instance_symbol)
        self.component_name = _symbols(sym["component"], system.component_symbol)
        self.atom_name = _symbols(sym["atom"], system.atom_symbol)
        self.entity_name = _symbols(sym["entity"], system.entity_symbol)
        typed = known(system.entity_type_state)
        self.entity_kind = np.where(typed, system.entity_type, 0)
        kind = self.entity_kind[self.component_entity]
        self.numbered = np.isin(kind, (EntityType.POLYMER, EntityType.BRANCHED))
        self.polymer_component = kind == EntityType.POLYMER

        # identity: the sequence number of a polymer or branched position is required (entity_poly_seq, schemes)
        number, number_state = structure.component_sequence_number, structure.component_sequence_number_state
        missing = self.numbered & ~known(number_state)
        if identity == "assign":
            number = np.where(missing, self.ordinal + 1, number)
            number_state = np.where(missing, _KDER, number_state).astype(np.uint8)
            for component in np.flatnonzero(missing).tolist():
                notes.assigned("label_seq_id", str(self.ordinal[component] + 1), SubjectKind.COMPONENT, component)
        else:
            notes.unencodable("identity", "_atom_site.label_seq_id", SubjectKind.COMPONENT, np.flatnonzero(missing))
        self.sequence = (_text(number), number_state)
        self.sequence_number = number

        text = structure.component_author_text
        readable = np.append(author_readings(tsym["author_number"])[1], True)
        unread = known(structure.component_author_number_state) & ~readable[text]  # a hybrid-36 resSeq, say
        notes.unencodable("author_number", "_atom_site.auth_seq_id", SubjectKind.COMPONENT, np.flatnonzero(unread))
        self.author = (
            _symbols(tsym["author_number"], text),
            np.where(text >= 0, _KD, structure.component_author_number_state).astype(np.uint8),
        )
        insertion = structure.component_insertion
        self.insertion = (_symbols(tsym["insertion"], insertion), np.where(insertion >= 0, _KD, _IDEF).astype(np.uint8))
        self.chain = (_symbols(tsym["chain"], structure.instance_author_chain), structure.instance_author_chain_state)

        evidence = structure.coordinate_evidence
        # a row per placed atom and per source row whose coordinates are ``?``
        self.written = np.union1d(np.flatnonzero(evidence != CoordinateEvidence.NONE), unplaced(structure))
        self.atom_site_id = np.full(len(evidence), -1, dtype=np.int64)
        self.atom_site_id[self.written] = np.arange(1, len(self.written) + 1)

    def label_seq(self, components: NDArray[np.int64]) -> Column:
        r"""``label_seq_id`` of each component: its number, ``.`` for a branched one (whose number is its branch
        scheme's ``num``)."""
        branched = self.numbered[components] & ~self.polymer_component[components]
        return self.sequence[0][components], np.where(branched, _IDEC, self.sequence[1][components]).astype(np.uint8)


def _plan(
    structure: Structure,
    block: str,
    identity: Identity,
    definitions: Definitions | None,
    library: ComponentLibrary | None,
):
    notes = Notes(structure)
    value = _Value(structure, identity, notes)
    assumptions(notes, structure)
    plan = _Plan(notes)
    one = (SubjectKind.NONE, np.array([-1]))
    plan.add("entry", _stated({"id": np.array([block])}), one)
    _experiment(plan, value)
    _crystal(plan, value)
    family, restored = _entities(plan, value)
    _components(plan, value, definitions, library)
    _modified(plan, value)
    _atom_site(plan, value)
    _connections(plan, value, family, restored)
    _private(plan, value, definitions)
    return plan.finish(), notes


def _experiment(plan: _Plan, value: _Value) -> None:
    structure = value.structure
    methods = structure.symbols()["method"]
    method = structure.method[structure.system_method_ptr[0] : structure.system_method_ptr[1]]
    plan.add("exptl", _stated({"method": _symbols(methods, method)}), (SubjectKind.NONE, np.full(len(method), -1)))
    rows = np.arange(*structure.system_resolution_ptr[:2])
    for kind, name in _RESOLUTION.items():
        picked = rows[structure.resolution_kind[rows] == kind]
        columns = {"resolution": (_text(structure.resolution[picked]), structure.resolution_state[picked])}
        plan.add(name, columns, (SubjectKind.NONE, np.full(len(picked), -1)))
    lost = rows[structure.resolution_kind[rows] == ResolutionKind.PDB_REMARK_2]  # no mmCIF item names its source
    plan.notes.unencodable("resolution", "_refine.ls_d_res_high", SubjectKind.NONE, np.full(len(lost), -1))


def _crystal(plan: _Plan, value: _Value) -> None:
    structure, one = value.structure, (SubjectKind.NONE, np.array([-1]))
    if structure.cell_state[0] != _UN:
        names = ("length_a", "length_b", "length_c", "angle_alpha", "angle_beta", "angle_gamma")
        cell = _printed(structure.cell[0], structure.cell_quantum[0])
        plan.add("cell", {n: (cell[i : i + 1], structure.cell_state) for i, n in enumerate(names)}, one)
    if structure.space_group_state[0] != _UN:
        name = _symbols(structure.symbols()["space_group"], structure.space_group)
        plan.add("symmetry", {"space_group": (name, structure.space_group_state)}, one)
    if structure.operation_numbering[0] == OperationNumbering.FILE:  # ITA numbering re-resolves from the H-M name
        rows = slice(*structure.operation_ptr[:2])
        ids = _symbols(structure.symbols()["operation"], structure.operation_id[rows])
        texts = [
            triplet(SymOp(tuple(r.reshape(9).tolist()), tuple(t.tolist())))
            for r, t in zip(structure.operation_rotation[rows], structure.operation_translation[rows])
        ]
        columns = _stated({"operation_id": ids, "operation_text": np.array(texts, dtype=object)})
        plan.add("space_group_symop", columns, (SubjectKind.NONE, np.full(len(ids), -1)))
    untabled(plan.notes, structure, "_space_group_symop")  # a table not in force is written as none
    item = "_atom_sites.fract_transf_matrix"
    if written_transform(structure):  # the source's transform; a derived one is derived again from the cell
        unit = structure.frame_quantum[0]
        frame = _printed(structure.frame[0], unit)
        stated = np.array([_KD], np.uint8)
        columns = {f"fract_{i + 1}{j + 1}": (frame[i, j : j + 1], stated) for i in range(3) for j in range(4)}
        plan.add("atom_sites", columns, one)
        texts = np.array(frame.ravel().tolist(), dtype=bytes)
        printed = parse_numbers(texts, np.ones(len(texts), dtype=bool), lambda row: RequestRef("frame", str(row)))[2]
        if not np.isclose(printed, unit.ravel(), rtol=1e-9, atol=0).all():  # a re-read bounds by another resolution
            plan.notes.unencodable("frame", item, *one)
    unframed(plan.notes, structure, item)


def _entities(plan: _Plan, value: _Value) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    system, entities = value.system, np.arange(len(value.entity_name))
    rows = (SubjectKind.NONE, np.full(len(entities), -1))
    types = (_ENTITY_TYPES[system.entity_type], system.entity_type_state)
    plan.add("entity", {"id": (value.entity_name, np.full(len(entities), _KD, np.uint8)), "type": types}, rows)

    # an _entity_poly row is a statement (it puts the entity under the junction rule): none where the source stated none
    stated = (system.entity_polymer_type_state != _UN) | (system.entity_nstd_linkage_state != _UN)
    polymer = np.flatnonzero((value.entity_kind == EntityType.POLYMER) & stated)
    polymer_type = _symbols(system.symbols()["polymer_type"], system.entity_polymer_type[polymer])
    linkage, linkage_state = _linkage(value, polymer)
    # the entities whose written row lets a re-read derive their junctions (a type naming a polymer family), and
    # restore those no row states (under a known `no`); the others' junctions are written as rows (_connections)
    named = np.array([POLYMER_FAMILIES.get(t) is not None for t in polymer_type.tolist()], dtype=bool)
    family = np.zeros(len(entities), dtype=bool)
    family[polymer] = named & determined(system.entity_polymer_type_state[polymer])
    restored = np.zeros(len(entities), dtype=bool)
    restored[polymer] = family[polymer] & (linkage == "no") & np.isin(linkage_state, (_KD, _KDER))
    columns = {
        "entity": (value.entity_name[polymer], np.full(len(polymer), _KD, np.uint8)),
        "type": (polymer_type, system.entity_polymer_type_state[polymer]),
        "nstd_linkage": (linkage, linkage_state),
    }
    plan.add("entity_poly", columns, (SubjectKind.NONE, np.full(len(polymer), -1)))

    # the union of the positions of every instance of the entity; a position with two components is heterogeneous
    picked = np.flatnonzero(value.polymer_component & known(value.sequence[1]))
    keys = sorted(
        {(int(value.component_entity[c]), int(value.sequence_number[c]), value.component_name[c]) for c in picked}
    )
    entity = np.array([k[0] for k in keys], dtype=np.int64)
    number = np.array([k[1] for k in keys], dtype=np.int64)
    pairs = entity * (1 << 32) + number
    hetero = np.isin(pairs, pairs[np.flatnonzero(pairs[1:] == pairs[:-1])])
    columns = _stated(
        {
            "entity": value.entity_name[entity],
            "num": number,
            "mon_id": np.array([k[2] for k in keys], dtype=object),
            "hetero": np.where(hetero, "yes", "no"),
        }
    )
    plan.add("entity_poly_seq", columns, (SubjectKind.NONE, np.full(len(keys), -1)))

    instances = np.arange(len(value.instance_name))
    columns = _stated({"id": value.instance_name, "entity": value.entity_name[value.instance_entity]})
    plan.add("struct_asym", columns, (SubjectKind.INSTANCE, instances))

    kinds = value.entity_kind[value.component_entity]
    for name, mask in (
        ("pdbx_poly_seq_scheme", kinds == EntityType.POLYMER),
        ("pdbx_branch_scheme", kinds == EntityType.BRANCHED),
        ("pdbx_nonpoly_scheme", ~value.numbered & (value.sequence[1] == _IDEF)),  # else its atom rows declared it
    ):
        c = np.flatnonzero(mask)
        k = value.component_instance[c]
        columns = {
            "asym": (value.instance_name[k], np.full(len(c), _KD, np.uint8)),
            "entity": (value.entity_name[value.component_entity[c]], np.full(len(c), _KD, np.uint8)),
            "mon_id": (value.component_name[c], np.full(len(c), _KD, np.uint8)),
            "pdb_seq": (value.author[0][c], value.author[1][c]),
            "pdb_ins": (value.insertion[0][c], value.insertion[1][c]),
            "pdb_strand": (value.chain[0][k], value.chain[1][k]),
        }
        if name != "pdbx_nonpoly_scheme":
            columns["number"] = (value.sequence[0][c], value.sequence[1][c])
        plan.add(name, columns, (SubjectKind.COMPONENT, c))
    return family, restored


def _linkage(value: _Value, entities: NDArray[np.int64]) -> Column:
    r"""
    ``_entity_poly.nstd_linkage`` of each polymer entity: the stored statement, else (INAPPLICABLE×DEFINED: a PDB
    source; a stated ``.`` is a statement) the value the graph derives: ``yes`` iff a consecutive pair is
    linked otherwise than by its default junction, else ``?`` iff a pair has no determined link, else ``no``. Read
    back, a derived ``no`` restores every default junction; under a derived ``yes`` or ``?`` they are candidates, so
    ``_connections`` writes the default junctions no row states as rows.
    """
    system = value.system
    linkage, linkage_state = system.entity_nstd_linkage[entities], system.entity_nstd_linkage_state[entities]
    stored = np.array(["", "no", "yes"], dtype=object)[linkage]
    derive = linkage_state == _IDEF
    if not derive.any():
        return stored, linkage_state
    first, second = system.edge_index.astype(np.int64)
    a, b = value.atom_component[first], value.atom_component[second]
    consecutive = (b == a + 1) & (value.component_instance[a] == value.component_instance[b])
    port = system.component_port.astype(np.int64)
    default = consecutive & (first == port[a, 1]) & (second == port[b, 0])
    count = len(value.component_instance)
    linked, otherwise = np.zeros(count, dtype=bool), np.zeros(count, dtype=bool)
    linked[a[default]] = True
    otherwise[a[consecutive & ~default]] = True
    pair = np.zeros(count, dtype=bool)  # component k and k + 1 in one polymer instance
    pair[:-1] = (value.component_instance[1:] == value.component_instance[:-1]) & value.polymer_component[:-1]
    entity = value.component_entity
    yes = np.bincount(entity[pair & ~linked & otherwise], minlength=len(value.entity_name)) > 0
    open_ = np.bincount(entity[pair & ~linked & ~otherwise], minlength=len(value.entity_name)) > 0
    derived = np.where(yes[entities], "yes", np.where(open_[entities], "?", "no"))
    derived_state = np.where(~yes[entities] & open_[entities], _UDER, _KDER)
    return np.where(derive, derived, stored), np.where(derive, derived_state, linkage_state).astype(np.uint8)


def _definition_states(states) -> NDArray[np.uint8]:
    r"""Definition item states as the token states they were read from (the definition pass reads them REFERENCE)."""
    states = np.asarray(states, dtype=np.uint8)
    return np.where(states == _KREF, _KD, np.where(states == _UREF, _UD, states)).astype(np.uint8)


def _components(plan: _Plan, value: _Value, definitions: Definitions | None, library: ComponentLibrary | None) -> None:
    r"""``chem_comp*`` of every component whose definition is not the archive's (``unarchived``): written from
    ``definitions``, whose digests must be the ones the components were built from. A component whose definition is
    not supplied is reported ``component_definition`` with item ``_chem_comp`` (no rows written), one whose rows do not
    define it again unchanged with item ``_chem_comp_atom``."""
    system = value.system
    local = unarchived(value.structure, library, None if definitions is None else definitions.library)
    by_id = {} if definitions is None else definitions.by_id
    chosen, missing, unrestated = [], [], []
    for cid in dict.fromkeys(value.component_name[local].tolist()):
        components = local[value.component_name[local] == cid]
        definition = by_id.get(cid)
        if definition is None:
            missing += components.tolist()
            continue
        wrong = components[system.component_definition_digest[components] != definition.digest]
        if len(wrong):
            fail("definition_set_mismatch", StructureRef(component=int(wrong[0])), id=cid)
        if not _restated(definition):
            unrestated += components.tolist()
        chosen.append(definition)
    plan.notes.unencodable("component_definition", "_chem_comp", SubjectKind.COMPONENT, missing)
    plan.notes.unencodable("component_definition", "_chem_comp_atom", SubjectKind.COMPONENT, unrestated)
    none = (SubjectKind.NONE, np.full(len(chosen), -1))
    plan.add(
        "chem_comp",
        {
            "id": (np.array([d.component_id for d in chosen], dtype=object), np.full(len(chosen), _KD, np.uint8)),
            "type": (
                np.array([d.chem_comp_type or "" for d in chosen], dtype=object),
                _definition_states([d.chem_comp_type_state for d in chosen]),
            ),
            "one_letter_code": (
                np.array([d.one_letter_code or "" for d in chosen], dtype=object),
                _definition_states([d.one_letter_code_state for d in chosen]),
            ),
            "mon_nstd_parent_comp_id": (
                np.array([",".join(d.parents) for d in chosen], dtype=object),
                _definition_states([d.parents_state for d in chosen]),
            ),
        },
        none,
    )
    atoms: dict[str, list] = {}
    bonds: dict[str, list] = {}
    for d in chosen:
        for name, column in _definition_atoms(d).items():
            atoms.setdefault(name, []).append(column)
        for name, column in _definition_bonds(d).items():
            bonds.setdefault(name, []).append(column)
    for name, parts in (("chem_comp_atom", atoms), ("chem_comp_bond", bonds)):
        if parts:
            columns = {
                k: (np.concatenate([p[0] for p in v]), np.concatenate([p[1] for p in v])) for k, v in parts.items()
            }
            plan.add(name, columns, (SubjectKind.NONE, np.full(len(columns["comp_id"][1]), -1)))


def _restated(d: ChemicalComponent) -> bool:
    r"""Whether the ``chem_comp*`` rows written for ``d`` define it again unchanged: no isotope but D/T, no parity
    resting on CX tags or a descriptor, no enhanced stereo group and no cis/trans row without its E/Z label (none of
    these has an item; no CIP at write). A descriptor-verified parity of a file-scope definition is the archive's
    arbitration, which the reader keeps again over the written conformer."""
    isotope = (d.isotope > 0) & ~((d.element == 1) & np.isin(d.isotope, (2, 3)))
    verified = () if d.scope == DefinitionScope.FILE else (StereoBasis.DESCRIPTOR_VERIFIED,)
    parity = np.isin(d.stereo_basis, (StereoBasis.STATED, *verified))
    unlabelled = any(label is None for label in d.bond_stereo_declared)
    return not (isotope.any() or parity.any() or len(d.stereo_group_kind) or unlabelled)


def _modified(plan: _Plan, value: _Value) -> None:
    r"""``_pdbx_struct_mod_residue``: a row per parent of each component's own annotation (``component_parent``,
    occurrence scope), at its label address too, which tells the copies of one author chain apart. A definition's
    parents are written in its ``chem_comp``."""
    system = value.system
    c = _owner(system.component_parent_ptr)
    k = value.component_instance[c]
    stated = np.full(len(c), _KD, np.uint8)
    columns = {
        "label_asym": (value.instance_name[k], stated),
        "label_seq": value.label_seq(c),
        "asym": (value.chain[0][k], value.chain[1][k]),
        "seq": (value.author[0][c], value.author[1][c]),
        "comp": (value.component_name[c], stated),
        "parent": (_symbols(system.symbols()["parent"], system.component_parent), stated),
        "ins": (value.insertion[0][c], value.insertion[1][c]),
    }
    repeated = frozenset({"label_seq", "asym", "seq", "ins"})
    plan.add("pdbx_struct_mod_residue", columns, (SubjectKind.COMPONENT, c), repeated=repeated)


def _definition_atoms(d) -> dict[str, Column]:
    n = len(d.atom_names)
    stated = np.full(n, _KD, np.uint8)
    element = np.array(ELEMENTS, dtype=object)[d.element]
    element = np.where(
        (d.element == 1) & (d.isotope == 2), "D", np.where((d.element == 1) & (d.isotope == 3), "T", element)
    )
    flag = np.where(np.arange(3) > 0, _KD, _UD)
    columns = {
        "comp_id": (np.full(n, d.component_id, dtype=object), stated),
        "atom_id": (np.array(d.atom_names, dtype=object), stated),
        "type_symbol": (element, stated),
        "charge": (_text(d.charge), _definition_states(d.charge_state)),
        "pdbx_aromatic_flag": (np.where(d.aromatic, "Y", "N"), _definition_states(d.aromatic_state)),
        "pdbx_leaving_atom_flag": (_FLAG[d.leaving_flag], flag[d.leaving_flag]),
        "pdbx_stereo_config": (
            np.array([label or "" for label in d.cip_declared], dtype=object),
            np.array([_UD if label is None else _KD for label in d.cip_declared], np.uint8),
        ),
        "pdbx_n_terminal_atom_flag": (_FLAG[d.terminal_flag[:, 0]], flag[d.terminal_flag[:, 0]]),
        "pdbx_c_terminal_atom_flag": (_FLAG[d.terminal_flag[:, 1]], flag[d.terminal_flag[:, 1]]),
    }
    # the conformer at its printed unit, so the definition (quantum included) is derived again unchanged
    texts = _printed(d.reference, d.reference_quantum[:, None]).reshape(n, 3)
    for template, source in (
        ("model_Cartn_{}", ReferenceSource.MODEL),
        ("pdbx_model_Cartn_{}_ideal", ReferenceSource.IDEAL),
    ):
        given = np.full(n, _KD if d.reference_source == source else _UD, np.uint8)
        for axis, name in enumerate("xyz"):
            columns[template.format(name)] = (texts[:, axis], given)
    return columns


def _definition_bonds(d) -> dict[str, Column]:
    m = len(d.bond_atoms)
    names = np.array(d.atom_names, dtype=object)
    labels = np.full(m, "N", dtype=object)
    for bond, label in zip(d.bond_stereo_bond.tolist(), d.bond_stereo_declared):
        labels[bond] = label or "?"
    stated = np.full(m, _KD, np.uint8)
    typed = d.bond_order_token > OrderToken.QUAD  # AROM–POLY: the stated token, which states no order
    order = np.where(typed, _TOKEN_NAMES[d.bond_order_token], _ORDER_TOKENS[d.bond_order])
    return {
        "comp_id": (np.full(m, d.component_id, dtype=object), stated),
        "atom_id_1": (names[d.bond_atoms[:, 0]], stated),
        "atom_id_2": (names[d.bond_atoms[:, 1]], stated),
        "value_order": (
            np.char.upper(order.astype(str)).astype(object),
            np.where(typed, _KD, _definition_states(d.bond_order_state)).astype(np.uint8),
        ),
        "pdbx_aromatic_flag": (np.where(d.bond_aromatic, "Y", "N"), _definition_states(d.bond_aromatic_state)),
        "pdbx_stereo_config": (labels, np.where(labels == "?", _UD, _KD).astype(np.uint8)),
    }


def _element(value: _Value, atoms: NDArray[np.int64]) -> NDArray:
    r"""``type_symbol`` of each atom: its element, ``D``/``T`` for hydrogen 2/3; another isotope and an element that is
    not KNOWN cannot be stated."""
    system, notes = value.system, value.notes
    z, z_state = system.atomic_number[atoms], system.atomic_number_state[atoms]
    isotope = system.isotope[atoms]
    symbol = np.array(ELEMENTS, dtype=object)[np.where(known(z_state), z, 0)]
    heavy = (z == 1) & np.isin(isotope, (2, 3))
    symbol[heavy] = np.where(isotope[heavy] == 2, "D", "T")
    notes.unencodable("isotope", "_atom_site.type_symbol", SubjectKind.ATOM, atoms[~heavy & (isotope > 0)])
    notes.unencodable("element", "_atom_site.type_symbol", SubjectKind.ATOM, atoms[~known(z_state)])
    return symbol


def _atom_site(plan: _Plan, value: _Value) -> None:
    structure, system, atoms = value.structure, value.system, value.written
    c = value.atom_component[atoms]
    k = value.component_instance[c]
    n = len(atoms)
    stated = np.full(n, _KD, np.uint8)
    evidence = structure.coordinate_evidence[atoms]
    placed = evidence != CoordinateEvidence.NONE
    unobserved = np.isin(
        evidence, (CoordinateEvidence.PREDICTED, CoordinateEvidence.GENERATED, CoordinateEvidence.REFERENCE)
    )
    altloc = structure.atom_altloc[atoms]
    coordinates = _printed(structure.coordinates[atoms], structure.coordinate_quantum[atoms, None])

    def observation(name: str) -> Column:  # unobserved atoms: INAPPLICABLE×DEFINED (a Structure invariant), written "."
        return _text(structure.arrays()[name][atoms]), np.where(
            unobserved, _IDEC, structure.arrays()[f"{name}_state"][atoms]
        )

    hydrogens = known(system.implicit_hydrogens_state) & (system.implicit_hydrogens > 0)
    value.notes.unencodable("hydrogens", "_atom_site", SubjectKind.ATOM, np.flatnonzero(hydrogens))
    undetermined(value.notes, structure, value.component_instance[value.atom_component], "_atom_site")
    model = structure.system_model_number_state[0]
    charge, charge_state = charge_cell(value.notes, structure, atoms, "_atom_site.pdbx_formal_charge")
    columns = {
        "id": (_text(value.atom_site_id[atoms]), stated),
        "group": (np.where(value.polymer_component[c], "ATOM", "HETATM"), stated),
        "element": (_element(value, atoms), stated),
        "label_atom": (value.atom_name[atoms], stated),
        "label_alt": (_symbols(structure.symbols()["altloc"], altloc), np.where(altloc >= 0, _KD, _IDEC)),
        "label_comp": (value.component_name[c], stated),
        "label_asym": (value.instance_name[k], stated),
        "label_entity": (value.entity_name[value.instance_entity[k]], stated),
        "label_seq": value.label_seq(c),
        "insertion": (value.insertion[0][c], value.insertion[1][c]),
        **{axis: (coordinates[:, i], np.where(placed, _KD, _UD)) for i, axis in enumerate("xyz")},
        "occupancy": observation("occupancy"),
        # its su as occupancy_esd; an unobserved atom's absent (the reader gives it none)
        "occupancy_esd": (
            _text(structure.occupancy_uncertainty[atoms]),
            np.where(unobserved, _UN, structure.occupancy_uncertainty_state[atoms]),
        ),
        "b_factor": observation("b_factor"),
        "charge": (_text(charge), charge_state),
        "auth_seq": (value.author[0][c], value.author[1][c]),
        "auth_comp": (value.component_name[c], stated),
        "auth_asym": (value.chain[0][k], value.chain[1][k]),
        "auth_atom": (value.atom_name[atoms], stated),
    }
    # a known number is stated on every row and reads KNOWN×DECLARED, an omitted item UNKNOWN×NONE; any other state
    # is reported, never restored; an assumed number is never written
    assumed = basis(model) == Basis.ASSUMED
    stating = bool(n) and bool(known(model)) and not assumed
    if stating:
        columns["model"] = (_text(np.repeat(structure.system_model_number, n)), stated)
    if not assumed and model != (_KD if stating else _UN):
        family = "model" if known(model) and not n else "basis"  # a known number with no row to state it on
        value.notes.unencodable(family, "_atom_site.pdbx_PDB_model_num", SubjectKind.NONE, [-1])
    repeated = frozenset({"label_seq", "insertion", "auth_seq", "auth_asym"})
    plan.add("atom_site", columns, (SubjectKind.ATOM, atoms), repeated=repeated)


def _conn_type(
    value: _Value, codes: NDArray, kinds: NDArray, kind_states: NDArray, subjects, subject_kind
) -> tuple[Column, NDArray[np.bool_]]:
    r"""``_struct_conn.conn_type_id`` of each row and whether the row is written: the stated type, else the determined
    kind's (covalent ``covale``, coordination ``metalc``), else the undetermined kind's own state (``?``, ``.`` or
    absent, never a guessed type; an assumed kind is written ``?``, not as a statement). An unclassified connection
    (CONECT only) is not written: a struct_conn row would declare its endpoints."""
    stated = _symbols(value.system.symbols()["connection_type"], codes)
    stated = np.where(stated == "SSBOND", "disulf", stated)
    listed = np.isin(stated, CONN_TYPES)
    kinds = np.where(determined(kind_states), kinds, 0)
    by_kind = np.where(kinds == BondKind.COVALENT, "covale", np.where(kinds == BondKind.COORDINATION, "metalc", ""))
    out = np.where(listed, stated, by_kind)
    conect = (stated == "CONECT") & (out == "")
    value.notes.unencodable("connection_kind", "_struct_conn.conn_type_id", subject_kind, subjects[conect])
    return (out, np.where(out == "", kind_states, _KD).astype(np.uint8)), ~conect


def _leaving(value: _Value, pointer, leaving, side, rows, subjects, subject_kind) -> list[Column]:
    r"""The declared leaving atom of each side of each row (one per side; more cannot be stated)."""
    names = _symbols(value.system.symbols()["atom"], leaving)
    out = []
    for s in (0, 1):
        text, states = np.full(len(rows), "", dtype=object), np.full(len(rows), _UN, np.uint8)
        for i, r in enumerate(rows.tolist()):
            mine = np.flatnonzero(side[pointer[r] : pointer[r + 1]] == s) + pointer[r]
            if len(mine) > 1:
                value.notes.unencodable(
                    "leaving", f"_struct_conn.pdbx_ptnr{s + 1}_leaving_atom_id", subject_kind, [subjects[i]]
                )
            if len(mine):
                text[i], states[i] = names[mine[0]], _KD
        out.append((text, states))
    return out


def _endpoint(value: _Value, atoms: NDArray[np.int64], k: int) -> dict[str, Column]:
    structure = value.structure
    c = value.atom_component[atoms]
    i = value.component_instance[c]
    altloc = structure.atom_altloc[atoms]
    stated = np.full(len(atoms), _KD, np.uint8)
    return {
        f"label_asym_{k}": (value.instance_name[i], stated),
        f"label_seq_{k}": value.label_seq(c),
        f"label_comp_{k}": (value.component_name[c], stated),
        f"label_atom_{k}": (value.atom_name[atoms], stated),
        f"label_alt_{k}": (_symbols(structure.symbols()["altloc"], altloc), np.where(altloc >= 0, _KD, _UD)),
        f"auth_asym_{k}": (value.chain[0][i], value.chain[1][i]),
        f"auth_seq_{k}": (value.author[0][c], value.author[1][c]),
        f"auth_ins_{k}": (value.insertion[0][c], value.insertion[1][c]),
        f"auth_comp_{k}": (value.component_name[c], stated),
        f"auth_atom_{k}": (value.atom_name[atoms], stated),
    }


def _order(orders: NDArray, states: NDArray, conn: NDArray, restating: NDArray[np.bool_]) -> Column:
    r"""
    ``pdbx_value_order`` of each row. The reader derives a disulfide's KNOWN×DEFINED order and a ``metalc`` row's
    INAPPLICABLE×DEFINED or reference one from the type whatever the item says, and a ``restating`` row's (one
    stating an edge its junction, cap, branch link or definition makes) defined or reference order from what it
    restates: where every other row is the item's default, all follow it (the item is omitted, no ``default_mixed``
    record); otherwise the ``metalc`` row's inapplicable order is written ``.`` and every other ``?``. The
    inapplicable order of a row whose kind is written undetermined follows from an assumed kind: it is free the same
    way, written ``?`` like the kind. Every other defined order (a glycosylation role's, or a disulfide's whose edge's
    first typed row is not ``disulf``) is its value with a basis row: ``pdbx_role`` is not a column, so the role would
    not re-derive it.
    """
    free = ((conn == "disulf") & (states == _KDEF)) | ((conn == "metalc") & np.isin(states, (_IDEF, _KREF)))
    free |= (restating & np.isin(states, (_KDEF, _KREF))) | ((conn == "") & (states == _IDEF))
    if (states[~free] == _KDFT).all():
        states = np.where(free, _KDFT, states)
    states = np.where(free & (states != _KDFT), np.where((states == _IDEF) & (conn == "metalc"), _IDEC, _UD), states)
    return _ORDER_TOKENS[np.minimum(orders, 4)].astype(object), states.astype(np.uint8)


def _roles(value: _Value, directions: NDArray, states: NDArray, subjects, subject_kind) -> tuple[Column, Column]:
    r"""``ptnr1_role``/``ptnr2_role`` of each row, the only items stating a direction (+1: partner 1 donates). A
    known undirected row has no role pair saying so."""
    undirected = known(states) & (directions == 0)
    value.notes.unencodable("direction", "_struct_conn.ptnr1_role", subject_kind, subjects[undirected])
    states = np.where(undirected, _UN, states).astype(np.uint8)
    donor = directions > 0
    return (np.where(donor, "donor", "acceptor").astype(object), states), (
        np.where(donor, "acceptor", "donor").astype(object),
        states,
    )


def _connections(plan: _Plan, value: _Value, family: NDArray[np.bool_], restored: NDArray[np.bool_]) -> None:
    r"""
    ``_struct_conn`` (connections, junctions with a declared order, edges a stated row restates, external connections,
    and the junctions a re-read would not restore: those of an entity whose written ``_entity_poly`` row names no
    polymer family or no known ``no`` linkage) and ``_pdbx_entity_branch_link`` (one row per distinct link of a
    branched entity; an unknown declared order is written ``?`` and reported, since the value holds a no-order token
    the same way). Such a junction is written when its order is determined and both residues have atom rows (a row
    binds no residue without one); every other one, and one whose entity's row names no family (a re-read holds its
    row as a connection), is reported ``junction``: its edge names the entity, chain and atoms.
    """
    system, structure, notes = value.system, value.structure, value.notes
    first, second = system.edge_index.astype(np.int64)
    origin = system.bond_origin
    entity = value.component_entity[value.atom_component[first]]
    restating = ~np.isin(origin, (BondOrigin.CONNECTION, BondOrigin.RECIPE)) & (system.bond_connection_type >= 0)
    restating &= (origin != BondOrigin.JUNCTION) | family[entity]  # a re-read derives no familyless junction
    junction = (origin == BondOrigin.JUNCTION) & ~restating
    declared = junction & (system.bond_order_state == _KD)
    rowed = np.zeros(len(value.component_instance), dtype=bool)
    rowed[value.atom_component[value.written]] = True
    unrestored = junction & ~declared & ~restored[entity]
    stated = unrestored & determined(system.bond_order_state)
    stated &= rowed[value.atom_component[first]] & rowed[value.atom_component[second]]
    notes.unencodable("junction", "_struct_conn", SubjectKind.EDGE, np.flatnonzero(unrestored & ~stated))
    rows = declared | stated
    notes.unencodable("junction", "_entity_poly.type", SubjectKind.EDGE, np.flatnonzero(rows & ~family[entity]))
    restating |= stated & family[entity]  # a re-read derives the junction its row states again
    edges = np.flatnonzero(np.isin(origin, (BondOrigin.CONNECTION, BondOrigin.RECIPE)) | rows | restating)
    (conn, conn_state), keep = _conn_type(
        value,
        system.bond_connection_type[edges],
        system.bond_kind[edges],
        system.bond_kind_state[edges],
        edges,
        SubjectKind.EDGE,
    )
    covale = junction[edges] & (conn == "")
    conn, conn_state = np.where(covale, "covale", conn), np.where(covale, _KD, conn_state)
    edges, conn, conn_state = edges[keep], conn[keep], conn_state[keep]
    edge_rows = {
        **_endpoint(value, first[edges], 1),
        **_endpoint(value, second[edges], 2),
        "symmetry_1": (np.full(len(edges), "", dtype=object), np.full(len(edges), _KDEF, np.uint8)),
        "symmetry_2": (np.full(len(edges), "", dtype=object), np.full(len(edges), _KDEF, np.uint8)),
    }
    edge_rows["role_1"], edge_rows["role_2"] = _roles(
        value, system.bond_direction[edges], system.bond_direction_state[edges], edges, SubjectKind.EDGE
    )
    edge_rows["leaving_1"], edge_rows["leaving_2"] = _leaving(
        value,
        system.bond_declared_leaving_ptr,
        system.bond_declared_leaving,
        system.bond_declared_leaving_side,
        edges,
        edges,
        SubjectKind.EDGE,
    )

    # external connections: the partner must be written, and a known relation must have a resolvable code; an unknown
    # one is stated as unknown, a derived code with its basis row
    x = np.arange(len(system.external_atom))
    partner = system.atom_index(
        [
            AtomSite(str(i), int(c), str(a))
            for i, c, a in zip(
                _symbols(system.symbols()["instance"], system.external_partner_instance),
                system.external_partner_component,
                _symbols(system.symbols()["atom"], system.external_partner_atom),
            )
        ]
    )
    transform = system.external_transform.astype(np.int64)
    code, transform_state = system.transform_code[transform], system.transform_state[transform]
    cut = partner < 0
    numbering = structure.operation_numbering[0]
    unresolvable = ~cut & known(transform_state) & ((code < 0) | (numbering == OperationNumbering.NONE))
    notes.unencodable("external_connection", "_struct_conn.ptnr2_label_atom_id", SubjectKind.EXTERNAL, x[cut])
    notes.unencodable("external_connection", "_struct_conn.ptnr2_symmetry", SubjectKind.EXTERNAL, x[unresolvable])
    x, sided = one_way(structure, partner, x[~cut & ~unresolvable])
    notes.unencodable("external_connection", "_struct_conn", SubjectKind.EXTERNAL, sided)
    (external_conn, external_state), keep = _conn_type(
        value,
        system.external_connection_type[x],
        system.external_kind[x],
        system.external_kind_state[x],
        x,
        SubjectKind.EXTERNAL,
    )
    x, external_conn, external_state = x[keep], external_conn[keep], external_state[keep]
    symmetry = np.array([f"{p // 1000}_{p % 1000:03d}" if p >= 0 else "" for p in code[x].tolist()], dtype=object)
    external_rows = {
        **_endpoint(value, system.external_atom[x].astype(np.int64), 1),
        **_endpoint(value, partner[x], 2),
        "symmetry_1": (np.full(len(x), "", dtype=object), np.full(len(x), _KDEF, np.uint8)),
        "symmetry_2": (symmetry, transform_state[x]),
    }
    external_rows["role_1"], external_rows["role_2"] = _roles(
        value, system.external_direction[x], system.external_direction_state[x], x, SubjectKind.EXTERNAL
    )
    external_rows["leaving_1"], external_rows["leaving_2"] = _leaving(
        value,
        system.external_declared_leaving_ptr,
        system.external_declared_leaving,
        system.external_declared_leaving_side,
        x,
        x,
        SubjectKind.EXTERNAL,
    )
    types = np.concatenate((conn, external_conn)).astype(object)
    ids = np.empty(len(types), dtype=object)
    for name in dict.fromkeys(types.tolist()):
        mine = np.flatnonzero(types == name)
        ids[mine] = [f"{name or 'conn'}{j + 1}" for j in range(len(mine))]
    columns = {
        name: (
            np.concatenate((edge_rows[name][0], external_rows[name][0])),
            np.concatenate((edge_rows[name][1], external_rows[name][1])),
        )
        for name in edge_rows
    }
    columns |= _stated({"id": ids})
    columns["conn_type"] = (types, np.concatenate((conn_state, external_state)).astype(np.uint8))
    columns["order"] = _order(
        np.concatenate((system.bond_order[edges], system.external_order[x])),
        np.concatenate((system.bond_order_state[edges], system.external_order_state[x])),
        types,
        np.concatenate((restating[edges], np.zeros(len(x), dtype=bool))),
    )
    kinds = np.repeat([SubjectKind.EDGE, SubjectKind.EXTERNAL], [len(edges), len(x)])
    repeated = frozenset(f"{name}_{k}" for name in ("label_seq", "auth_asym", "auth_seq", "auth_ins") for k in (1, 2))
    plan.add("struct_conn", columns, (kinds, np.concatenate((edges, x))), repeated=repeated)

    branch = np.flatnonzero(origin == BondOrigin.BRANCH_LINK)
    links: dict[tuple, int] = {}
    for e in branch.tolist():
        a, b = value.atom_component[first[e]], value.atom_component[second[e]]
        key = (
            int(value.component_entity[a]),
            int(value.sequence_number[a]),
            value.component_name[a],
            value.atom_name[first[e]],
            int(value.sequence_number[b]),
            value.component_name[b],
            value.atom_name[second[e]],
            int(system.bond_order[e]),
            int(system.bond_order_state[e]),
        )
        links.setdefault(key, e)
    picked = np.array(list(links.values()), dtype=np.int64)
    keys = list(links)
    columns = _stated(
        {
            "link_id": np.arange(1, len(keys) + 1),
            "entity": value.entity_name[[k[0] for k in keys]] if keys else np.zeros(0, dtype=object),
            "num_1": np.array([k[1] for k in keys], dtype=np.int64),
            "comp_1": np.array([k[2] for k in keys], dtype=object),
            "atom_1": np.array([k[3] for k in keys], dtype=object),
            "num_2": np.array([k[4] for k in keys], dtype=np.int64),
            "comp_2": np.array([k[5] for k in keys], dtype=object),
            "atom_2": np.array([k[6] for k in keys], dtype=object),
        }
    )
    columns["order"] = (
        _ORDER_TOKENS[np.minimum(system.bond_order[picked], 4)].astype(object),
        system.bond_order_state[picked],
    )
    # the value holds a stated arom, delo, pi or poly (no integer order) as `?` does: written `?`, reported
    stated_none = picked[system.bond_order_state[picked] == _UD]
    notes.unencodable("order", "_pdbx_entity_branch_link.value_order", SubjectKind.EDGE, stated_none)
    columns["leaving_1"], columns["leaving_2"] = _leaving(
        value,
        system.bond_declared_leaving_ptr,
        system.bond_declared_leaving,
        system.bond_declared_leaving_side,
        picked,
        picked,
        SubjectKind.EDGE,
    )
    plan.add("pdbx_entity_branch_link", columns, (SubjectKind.EDGE, picked))


def _site(value: _Value, slots: NDArray[np.int64], prefix: str) -> dict[str, Column]:
    r"""Atoms named natively (instance, component ordinal, atom name); the implicit slot ``-1`` is ``.``."""
    atoms = np.maximum(slots, 0)
    c = value.atom_component[atoms]
    implicit = slots == -1
    states = np.where(implicit, _IDEC, _KD).astype(np.uint8)
    return {
        f"{prefix}asym_id": (value.instance_name[value.component_instance[c]], states),
        f"{prefix}component": (_text(value.ordinal[c]), states),
        f"{prefix}atom_id": (value.atom_name[atoms], states),
    }


def _private(plan: _Plan, value: _Value, definitions: Definitions | None) -> None:
    r"""The private categories: coordinate evidence other than observed, instances that are not the deposited
    asym itself or whose connections are not all represented (their rows may name what the file does not hold),
    configurations the written definitions do not reproduce (those not retained from a definition), and the CCD artifact
    the definitions were resolved against, so a reader holding another one knows that what the file leaves to the
    library is derived from its own."""
    structure, system, notes = value.structure, value.system, value.notes
    atoms = value.written
    evidence = structure.coordinate_evidence[atoms]
    other = np.isin(evidence, list(_EVIDENCE))
    names = np.array([_EVIDENCE.get(int(e), "") for e in evidence[other]], dtype=object)
    columns = _stated({"atom_site_id": value.atom_site_id[atoms[other]], "evidence": names})
    plan.add("ofoldx_coordinate_evidence", columns, (SubjectKind.ATOM, atoms[other]))

    kind, open_ = structure.instance_copy_kind, ~system.instance_connections_complete
    source = _symbols(structure.symbols()["asym"], structure.instance_source_asym)
    copies = np.flatnonzero((kind != CopyKind.DEPOSITED) | (source != value.instance_name) | open_)
    operators = [",".join(structure.copy_id(int(i)).operators) or "." for i in copies]
    columns = {
        "asym_id": (value.instance_name[copies], np.full(len(copies), _KD, np.uint8)),
        "copy_kind": (
            np.array([CopyKind(int(kind[i])).name.lower() for i in copies], dtype=object),
            np.full(len(copies), _KD, np.uint8),
        ),
        "source_asym": (source[copies], np.where(structure.instance_source_asym[copies] >= 0, _KD, _IDEC)),
        "operators": (
            np.array(operators, dtype=object),
            np.where(structure.instance_copy_operators[copies] >= 0, _KD, _IDEC),
        ),
        "connections": (
            np.where(open_[copies], "open", "complete").astype(object),
            np.full(len(copies), _KD, np.uint8),
        ),
    }
    plan.add("ofoldx_instance", columns, (SubjectKind.INSTANCE, copies))

    # a configuration retained from a definition is re-derived from it; an assumed one is never written
    centres = np.flatnonzero(~np.isin(basis(system.stereocentre_state), (Basis.REFERENCE, Basis.ASSUMED)))
    neighbors = system.stereocentre_neighbors[centres].astype(np.int64)
    external = (neighbors <= -2).any(axis=1)
    notes.unencodable("stereo", "_ofoldx_atom_parity", SubjectKind.STEREOCENTRE, centres[external])
    centres, neighbors = centres[~external], neighbors[~external]
    columns = _site(value, system.stereocentre_atom[centres].astype(np.int64), "")
    for slot in range(4):
        columns |= _site(value, neighbors[:, slot], f"neighbor_{slot + 1}_")
    columns |= _stated({"parity": system.stereocentre_parity[centres], "state": system.stereocentre_state[centres]})
    plan.add("ofoldx_atom_parity", columns, (SubjectKind.STEREOCENTRE, centres))

    bonds = np.flatnonzero(~np.isin(basis(system.stereobond_state), (Basis.REFERENCE, Basis.ASSUMED)))
    references = system.stereobond_reference[bonds].astype(np.int64)
    external = (references <= -2).any(axis=1)
    notes.unencodable("stereo", "_ofoldx_bond_stereo", SubjectKind.EDGE, system.stereobond_edge[bonds[external]])
    bonds, references = bonds[~external], references[~external]
    edge = system.stereobond_edge[bonds].astype(np.int64)
    columns = {}
    for k in (0, 1):
        columns |= _site(value, system.edge_index[k, edge].astype(np.int64), f"end_{k + 1}_")
        columns |= _site(value, references[:, k], f"reference_{k + 1}_")
    columns |= _stated({"value": system.stereobond_value[bonds], "state": system.stereobond_state[bonds]})
    plan.add("ofoldx_bond_stereo", columns, (SubjectKind.EDGE, edge))

    library = None if definitions is None else definitions.library
    if library is not None:
        columns = _stated({"content_version": [library.content_version], "sha256": [library.sha256]})
        plan.add("ofoldx_library", columns, (SubjectKind.NONE, np.array([-1])))


def _quote(text: str) -> str:
    r"""A CIF 1.1 token for a value: bare when the grammar allows, else quoted, else a text field. A literal ``?`` or
    ``.`` is quoted, so it never reads as a null."""
    if _BARE.fullmatch(text) and text not in ("?", ".") and not _RESERVED.match(text):
        return text
    if "\n" not in text:
        if "' " not in text and not text.endswith("'"):
            return f"'{text}'"
        if '" ' not in text and not text.endswith('"'):
            return f'"{text}"'
    return f"\n;{text}\n;\n"


def _serialize(block: str, categories) -> bytes:
    lines = [f"data_{block}", "#"]
    for name, n, items in categories:
        kinds = {item.item: item.kind for item in MMCIF_CODECS[name].items}
        tokens = {}
        for item, (text, null) in items.items():
            text = np.asarray(text, dtype=object)
            if kinds[item] in ("code", "text", "enum") and (~null).any():
                unique, inverse = np.unique(text[~null].astype(str), return_inverse=True)
                quoted = np.array([_quote(t) for t in unique.tolist()], dtype=object)
                text = text.copy()
                text[~null] = quoted[inverse.reshape(-1)]
            tokens[item] = text
        if n == 1:
            lines += [f"_{name}.{item} {column[0]}" for item, column in tokens.items()]
        else:
            lines += ["loop_", *(f"_{name}.{item}" for item in tokens)]
            lines += [" ".join(row) for row in zip(*(column.tolist() for column in tokens.values()))]
        lines.append("#")
    return ("\n".join(lines) + "\n").encode()
