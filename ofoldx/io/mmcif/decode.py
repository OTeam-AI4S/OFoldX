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
mmCIF categories → the shared source tables, every item through its ``MMCIF_CODECS`` row.

Label and author namespaces decode into separate columns and are never spliced; ``auth_atom_id`` is its own column;
``type_symbol`` ``D``/``T`` reads as H with isotope 2/3. An instance's entity statements are checked once, by
``io.binding``; the per-atom checks raise for the selected model and are recorded for the others
(``io.tables.model_view``). ``decode_components`` is the one ``chem_comp*`` decoder, for embedded definitions and for
the CCD build alike.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType
from typing import cast

import numpy as np
from numpy.typing import NDArray

from ...chemistry.definitions import ComponentItems, DefinitionRows
from ...data.molecular import ELEMENTS, EntityType
from ...data.structure import ResolutionKind
from ...errors import SourceRef, fail
from ...provenance import ALLOWED_STATES, Basis, Resolution, SourceRefs, known, state
from ..cif import CifCategory, TokenState, integer_tokens, intern
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
from .codec import MMCIF_CODECS, CategoryCodec, ItemCodec

_KNOWN = state(Resolution.KNOWN, Basis.DECLARED)
_ABSENT = state(Resolution.UNKNOWN, Basis.NONE)
_UNSTATED = state(Resolution.UNKNOWN, Basis.DECLARED)
_CONFLICT = state(Resolution.CONFLICT, Basis.NONE)
_NOT_INTEGER = state(Resolution.UNKNOWN, Basis.DERIVED)  # a token without an integer reading, still the key
_STRUCT_CONN = 1  # io.binding.QuerySource.STRUCT_CONN
_ELEMENTS = {symbol.upper(): (z, 0) for z, symbol in enumerate(ELEMENTS)} | {"D": (1, 2), "T": (1, 3)}
# the wwPDB archive sites _pdbx_database_status.process_site names: the marker of the wwPDB numbering of
# symmetry codes, no proof that the file is unmodified
_ARCHIVE_SITES = frozenset(("BNL", "NDB", "PDBC", "PDBE", "PDBJ", "RCSB"))

Columns = dict[str, tuple]
# category → column → written rows, the states they restore and their _ofoldx_fact_basis rows
FactBasis = Mapping[str, Mapping[str, tuple[NDArray[np.int64], NDArray[np.uint8], NDArray[np.int64]]]]


def decode_mmcif(categories: Mapping[str, CifCategory], model: int, *, block: str | None) -> SourceTables:
    r"""
    The source tables of one block, viewed for ``model``.

    Raises:
        ContradictionError: a number outside the CIF grammar, a stated token outside its dictionary enumeration, a
            missing or doubled category key, a stated null model number, a missing or unknown element symbol, or a
            disagreement between categories (the per-atom ones only for ``model``'s rows).
        UnsupportedError: a number outside its column's range, a stated author altloc.
        InvalidRequestError: ``model_unknown``.
    """
    builder = TableBuilder(block)
    basis = fact_basis(categories, block)
    for name, columns in basis.items():  # a written Conflict is a Conflict of the source
        for column, (rows, states, basis_rows) in columns.items():
            item, restored = _codec(name, column).item, states == _CONFLICT
            for row, k in zip(rows[restored].tolist(), basis_rows[restored].tolist()):
                builder.record("conflict_restored", [builder.ref(name, row, item), builder.ref("ofoldx_fact_basis", k)])
    numbered = "atom_site" in categories and "pdbx_pdb_model_num" in map(str.lower, categories["atom_site"].items)
    model_state = _KNOWN if numbered else _ABSENT

    def read(name: str) -> tuple[int, Columns]:
        return _read(categories, name, block, basis)

    atoms = _atoms(read, builder, block)
    entities = _entities(read, builder)
    asyms = _asyms(read, builder)
    schemes = _schemes(read, builder, block)
    connections = _connections(read, builder, block)
    crystal = _crystal(read, builder)
    operators, assemblies = _operators(read, builder), _assemblies(read, builder)
    experiment = _experiment(read, builder)
    polymers, sequences = _polymers(read, builder), _sequences(read, builder)
    branch_links, modified = _branch_links(read, builder), _modified(read, builder)
    symbols = builder.symbols()
    refs = builder.source_refs()
    tables = SourceTables(
        AtomTable(**atoms, symbols=symbols),
        model_state,
        entities,
        polymers,
        sequences,
        asyms,
        schemes,
        branch_links,
        connections,
        ConectTable(np.zeros((0, 2), np.int64), np.zeros(0, bool), np.zeros(0, np.int64)),
        modified,
        UnobservedTable(*(np.zeros(0, t) for t in (np.int32,) * 3 + (np.int64, np.int32, np.int32, np.int64))),
        np.zeros(0, np.int64),
        CrystalTable(**crystal, symbols=symbols),
        operators,
        assemblies,
        experiment,
        decode_components(categories, block=block, basis=basis),
        refs,
        builder.records(refs),
    )
    return model_view(tables, model)


def _read(
    categories: Mapping[str, CifCategory], name: str, block: str | None, basis: FactBasis = MappingProxyType({})
) -> tuple[int, Columns]:
    r"""Every codec item of one category, decoded by its row: ``float`` (value, state, quantum, standard uncertainty:
    ``nan`` where none is printed), ``int`` (value,
    state), ``code`` (local codes, state, symbols), ``text`` (bytes, state), ``enum`` (value, state; −1 for a stated
    token outside the enumeration, 0 where not stated). A token with a ``basis`` row reads as the state it carries."""
    codec, category = MMCIF_CODECS[name], categories.get(name)
    n = 0 if category is None else category.nrows
    columns = {item.column: _decode(category, item, n, name, block) for item in codec.items}
    for column, (rows, states, _) in basis.get(name, {}).items():
        columns[column][1][rows] = states
    return n, columns


def fact_basis(categories: Mapping[str, CifCategory], block: str | None) -> FactBasis:
    r"""
    ``_ofoldx_fact_basis``: per category and item column, the written rows, the state byte each
    restores and the basis row.

    Raises:
        ContradictionError: ``private_row_invalid`` for a row naming no written token of a codec item, naming
            ``_atom_site.pdbx_PDB_model_num`` (one fact of the system, no row state) or carrying no state that token
            can carry (KNOWN exactly on a value; only OFoldX writes the category); ``key_invalid`` for a row stating
            no key.
    """
    n, c = _read(categories, "ofoldx_fact_basis", block)
    # Both fields are required by their codec, so _read has rejected every unstated value.
    names, items = cast(list[str], _strings(c["category"])), cast(list[str], _strings(c["item"]))
    found: dict[str, dict[str, dict[int, tuple[int, int]]]] = {}
    for k, (name, item, row, value) in enumerate(zip(names, items, c["row"][0].tolist(), c["state"][0].tolist())):
        codec, category = MMCIF_CODECS.get(name), categories.get(name)
        column = next((i.column for i in codec.items if i.item == item), None) if codec else None
        tokens = category.text(item)[1] if category is not None else np.zeros(0, np.uint8)
        columns = found.setdefault(name, {})
        system = (name, item) == ("atom_site", "pdbx_PDB_model_num")
        if (
            column is None
            or cast(CategoryCodec, codec).rows == "private"
            or system
            or not 0 <= row < len(tokens)
            or row in columns.get(column, {})
        ):
            fail("private_row_invalid", SourceRef("ofoldx_fact_basis", k, "row", block))
        cells = columns.setdefault(column, {})
        token = tokens[row]
        written = token != TokenState.ABSENT and (token == TokenState.VALUE) == bool(known(value))
        if value not in ALLOWED_STATES or not written:
            fail("private_row_invalid", SourceRef("ofoldx_fact_basis", k, "state", block))
        cells[row] = (value, k)
    return MappingProxyType(
        {
            name: MappingProxyType(
                {
                    column: (
                        np.array(list(cells), np.int64),
                        np.array([value for value, _ in cells.values()], np.uint8),
                        np.array([k for _, k in cells.values()], np.int64),
                    )
                    for column, cells in columns.items()
                }
            )
            for name, columns in found.items()
        }
    )


def private_rows(categories: Mapping[str, CifCategory], name: str, block: str | None) -> list[dict]:
    r"""The rows of one OFoldX private category other than ``_ofoldx_fact_basis``: per row its items as text,
    integer or enumerated values (None where not a stated value), for the read driver."""
    n, c = _read(categories, name, block)
    values: dict[str, Sequence[int | str | None]] = {}
    for item in MMCIF_CODECS[name].items:
        decoded = c[item.column]
        if item.kind in ("int", "enum"):
            values[item.item] = [int(v) if known(s) else None for v, s in zip(decoded[0].tolist(), decoded[1])]
        else:
            values[item.item] = _strings(decoded)
    return [{item: column[k] for item, column in values.items()} for k in range(n)]


def _decode(category: CifCategory | None, item: ItemCodec, n: int, name: str, block: str | None) -> tuple:
    raw = category.text(item.item)[1] if category is not None else np.zeros(0, np.uint8)
    states = np.array([item.read[TokenState(i)] for i in range(4)], dtype=np.uint8)[raw]
    absent = raw == TokenState.ABSENT
    if item.required and (raw != TokenState.VALUE).any():
        row = int(np.flatnonzero(raw != TokenState.VALUE)[0])
        fail("element_missing" if item.item == "type_symbol" else "key_invalid", SourceRef(name, row, item.item, block))
    if item.kind == "float":
        value, _, uncertainty, quantum = category.floats(item.item) if category is not None else (np.zeros(0),) * 4
        if item.default is not None:
            value = np.where(absent, float(item.default), value)
        return value, states, quantum, uncertainty
    if item.kind == "int":
        value = category.ints(item.item)[0] if category is not None else np.zeros(0, np.int64)
        if item.default is not None:
            value = np.where(absent, int(item.default), value)
        return value, states
    if item.kind == "text":
        value = category.text(item.item)[0] if category is not None else np.zeros(0, "S1")
        if item.default is not None and absent.any():
            value = np.where(absent, item.default.encode(), value)
        return value, states
    codes, _, symbols = category.codes(item.item) if category is not None else (np.zeros(0, np.int32), None, ())
    if item.kind == "code":
        return codes, states, symbols
    enumeration = cast(Mapping[str, int], item.enumeration)  # Every enum codec declares its enumeration.
    table = np.array([enumeration.get(symbol.lower(), -1) for symbol in symbols] + [0], dtype=np.int64)
    value = table[codes]
    if item.default is not None:
        value = np.where(absent, enumeration[item.default], value)
    return value, states


def _codec(name: str, column: str) -> ItemCodec:
    return next(item for item in MMCIF_CODECS[name].items if item.column == column)


def _refuse_unlisted(value: NDArray, name: str, item: str, block: str | None, code: str = "item_enumeration") -> None:
    rows = np.flatnonzero(value < 0)
    if len(rows):
        fail(code, SourceRef(name, int(rows[0]), item, block))


def author_readings(tokens: Sequence[str]) -> tuple[NDArray[np.int64], NDArray[np.bool_]]:
    r"""The integer each mmCIF author-number token reads as, its CIF integer (``cif.integer_tokens``), and whether it
    reads as one."""
    value, _, readable = integer_tokens(np.array([token.encode() for token in tokens], dtype=np.bytes_))
    return value, readable


def _number(builder: TableBuilder, decoded: tuple) -> tuple[NDArray[np.int64], NDArray[np.uint8], NDArray[np.int32]]:
    r"""An author number (a code item): its integer reading, its state (UNKNOWN×DERIVED where the text has none) and
    its text in the ``number`` family."""
    codes, states, symbols = decoded
    value, readable = author_readings(symbols)  # code -1 (no token): its state as decoded
    state = np.where(np.append(readable, True)[codes], states, _NOT_INTEGER).astype(np.uint8)
    return np.append(value, 0)[codes], state, builder.code("number", codes, symbols)


def _code(builder: TableBuilder, family: str, decoded: tuple) -> tuple[NDArray[np.int32], NDArray[np.uint8]]:
    codes, states, symbols = decoded
    return builder.code(family, codes, symbols), states


def _elements(decoded: tuple, where: Callable[[int], SourceRef] | None) -> tuple[NDArray, NDArray]:
    r"""Atomic number and isotope of ``type_symbol`` (case-insensitive; ``D``/``T`` → H, 2/3). An unknown symbol is
    refused (``where`` given) or reads as 0."""
    values, states = decoded
    codes, symbols = intern(values, known(states))
    pairs = []
    for i, symbol in enumerate(symbols):
        pair = _ELEMENTS.get(symbol.strip().upper())
        if pair is None and where is not None:
            fail("element_unknown", where(int(np.flatnonzero(codes == i)[0])))
        pairs.append(pair or (0, 0))
    table = np.array([*pairs, (0, 0)], dtype=np.int64).reshape(-1, 2)
    return table[codes, 0].astype(np.uint8), table[codes, 1].astype(np.uint16)


def _range(value: NDArray, dtype, name: str, item: str, block: str | None) -> NDArray:
    limits = np.iinfo(dtype)
    bad = np.flatnonzero((value < limits.min) | (value > limits.max))
    if len(bad):
        fail("numeric_range", SourceRef(name, int(bad[0]), item, block))
    return value.astype(dtype)


def _atoms(read, builder: TableBuilder, block: str | None) -> dict:
    n, c = read("atom_site")
    model, model_state = c["model"]
    null = np.flatnonzero(~known(model_state))
    if len(null):
        fail("model_number_null", SourceRef("atom_site", int(null[0]), "pdbx_PDB_model_num", block))
    model = np.where(model_state == _KNOWN, model, 1)
    group, _ = c["group"]
    _refuse_unlisted(group, "atom_site", "group_PDB", block)
    element, isotope = _elements(c["element"], lambda row: SourceRef("atom_site", row, "type_symbol", block))
    axes = [c[axis] for axis in "xyz"]
    coordinate_state = axes[0][1].copy()
    coordinate_state[(axes[1][1] != axes[0][1]) | (axes[2][1] != axes[0][1])] = _CONFLICT
    charge, charge_state = c["charge"]
    occupancy, occupancy_state, _, braced = c["occupancy"]
    esd, esd_state, _, _ = c["occupancy_esd"]
    # the occupancy's su: its parenthesised one or occupancy_esd; both stated and different is a Conflict, neither
    # leaves occupancy_esd's state (absent: UNKNOWN×NONE)
    printed = ~np.isnan(braced)
    uncertainty = np.where(printed, braced, esd)
    uncertainty_state = np.where(printed, _KNOWN, esd_state).astype(np.uint8)
    uncertainty_state[printed & known(esd_state) & (esd != braced)] = _CONFLICT
    out = {
        "row": np.arange(n, dtype=np.int64),
        "serial": np.full(n, -1, np.int64),
        "model": _range(model, np.int32, "atom_site", "pdbx_PDB_model_num", block),
        "group": group.astype(np.uint8),
        "label_seq": c["label_seq"][0],
        "label_seq_state": c["label_seq"][1],
        "element": element,
        "isotope": isotope,
        "coordinates": np.stack([axis[0] for axis in axes], axis=1).reshape(n, 3),
        "coordinate_state": coordinate_state,
        "coordinate_quantum": (np.max([axis[2] for axis in axes], axis=0) / 2).astype(np.float32).reshape(n),
        "occupancy": occupancy,
        "occupancy_state": occupancy_state,
        "occupancy_uncertainty": uncertainty,
        "occupancy_uncertainty_state": uncertainty_state,
        "b_factor": c["b_factor"][0],
        "b_factor_state": c["b_factor"][1],
        "charge": _range(charge, np.int8, "atom_site", "pdbx_formal_charge", block),
        "charge_state": charge_state,
    }
    for column, family in (
        ("label_asym", "chain"),
        ("label_entity", "entity"),
        ("label_comp", "comp"),
        ("label_atom", "atom"),
        ("label_alt", "alt"),
        ("auth_asym", "chain"),
        ("auth_comp", "comp"),
        ("auth_atom", "atom"),
        ("insertion", "ins"),
    ):
        out[column], out[f"{column}_state"] = _code(builder, family, c[column])
    out["auth_seq"], out["auth_seq_state"], out["auth_seq_text"] = _number(builder, c["auth_seq"])
    return out


def _unique_key(codes: NDArray[np.int32], name: str, item: str, block: str | None) -> None:
    stated = np.flatnonzero(codes >= 0)
    _, first, counts = np.unique(codes[stated], return_index=True, return_counts=True)
    if (counts > 1).any():
        doubled = codes[stated][first[counts > 1][0]]
        fail("key_invalid", SourceRef(name, int(np.flatnonzero(codes == doubled)[1]), item, block))


def _entities(read, builder: TableBuilder) -> EntityTable:
    n, c = read("entity")
    block = builder.block
    identifier, _ = _code(builder, "entity", c["id"])
    _unique_key(identifier, "entity", "id", block)
    kind, kind_state = c["type"]
    _refuse_unlisted(kind, "entity", "type", block, "entity_type_enumeration")
    description, _ = _code(builder, "text", c["description"])
    return EntityTable(identifier, kind.astype(np.uint8), kind_state, description, builder.refs("entity", range(n)))


def _polymers(read, builder: TableBuilder) -> PolymerTable:
    n, c = read("entity_poly")
    entity, _ = _code(builder, "entity", c["entity"])
    _unique_key(entity, "entity_poly", "entity_id", builder.block)
    kind, kind_state = _code(builder, "polymer_type", c["type"])
    linkage, linkage_state = c["nstd_linkage"]
    _refuse_unlisted(linkage, "entity_poly", "nstd_linkage", builder.block)
    refs = builder.refs("entity_poly", range(n))
    return PolymerTable(entity, kind, kind_state, linkage.astype(np.uint8), linkage_state, refs)


def _sequences(read, builder: TableBuilder) -> SequenceTable:
    n, c = read("entity_poly_seq")
    hetero, _ = c["hetero"]
    _refuse_unlisted(hetero, "entity_poly_seq", "hetero", builder.block)
    return SequenceTable(
        _code(builder, "entity", c["entity"])[0],
        _range(c["num"][0], np.int32, "entity_poly_seq", "num", builder.block),
        _code(builder, "comp", c["mon_id"])[0],
        hetero == 2,
        builder.refs("entity_poly_seq", range(n)),
    )


def _asyms(read, builder: TableBuilder) -> AsymTable:
    n, c = read("struct_asym")
    identifier, _ = _code(builder, "chain", c["id"])
    _unique_key(identifier, "struct_asym", "id", builder.block)
    entity, _ = _code(builder, "entity", c["entity"])
    return AsymTable(identifier, entity, np.arange(n, dtype=np.int32), builder.refs("struct_asym", range(n)))


def _schemes(read, builder: TableBuilder, block: str | None) -> SchemeTable:
    parts: list[dict] = []
    for kind, name in enumerate(("pdbx_poly_seq_scheme", "pdbx_nonpoly_scheme", "pdbx_branch_scheme"), 1):
        n, c = read(name)
        part = {"kind": np.full(n, kind, np.uint8), "ref": builder.refs(name, range(n))}
        for column, family in (("asym", "chain"), ("entity", "entity"), ("mon_id", "comp")):
            part[column] = _code(builder, family, c[column])[0]
        part["number"] = (
            _range(c["number"][0], np.int32, name, "seq_id", block) if "number" in c else np.zeros(n, np.int32)
        )
        part["pdb_seq_num"], part["pdb_seq_state"], part["pdb_seq_text"] = _number(builder, c["pdb_seq"])
        part["pdb_ins_code"], part["pdb_ins_state"] = _code(builder, "ins", c["pdb_ins"])
        part["pdb_strand_id"], part["pdb_strand_state"] = _code(builder, "chain", c["pdb_strand"])
        parts.append(part)
    return SchemeTable(**{column: np.concatenate([part[column] for part in parts]) for column in parts[0]})


def _branch_links(read, builder: TableBuilder) -> BranchLinkTable:
    n, c = read("pdbx_entity_branch_link")
    block = builder.block
    order, order_state = c["order"]
    _refuse_unlisted(order, "pdbx_entity_branch_link", "value_order", block)

    def pair(column: str, family: str) -> NDArray[np.int32]:
        return np.stack([_code(builder, family, c[f"{column}_{k}"])[0] for k in (1, 2)], axis=1).reshape(n, 2)

    num = np.stack([c[f"num_{k}"][0] for k in (1, 2)], axis=1).reshape(n, 2)
    return BranchLinkTable(
        _code(builder, "entity", c["entity"])[0],
        pair("comp", "comp"),
        _range(num, np.int32, "pdbx_entity_branch_link", "entity_branch_list_num_1", block),
        pair("atom", "atom"),
        pair("leaving", "atom"),
        order.astype(np.uint8),
        order_state.astype(np.uint8),
        builder.refs("pdbx_entity_branch_link", range(n)),
    )


_ENDPOINT = (  # field_state order: label asym, seq, comp, atom, alt; author asym, seq, ins, comp, atom
    ("label_asym", "chain"),
    ("label_seq", None),
    ("label_comp", "comp"),
    ("label_atom", "atom"),
    ("label_alt", "alt"),
    ("auth_asym", "chain"),
    ("auth_seq", "number"),
    ("auth_ins", "ins"),
    ("auth_comp", "comp"),
    ("auth_atom", "atom"),
)


def _connections(read, builder: TableBuilder, block: str | None) -> ConnectionTable:
    n, c = read("struct_conn")
    refs = builder.refs("struct_conn", range(n))
    for k in (1, 2):
        stated = np.flatnonzero(known(c[f"auth_alt_{k}"][1]))
        if len(stated):
            fail(
                "author_altloc_unsupported",
                SourceRef("struct_conn", int(stated[0]), f"pdbx_ptnr{k}_auth_alt_id", block),
            )
    conn_type, conn_type_state = _code(builder, "conn_type", c["conn_type"])
    local, _, symbols = c["conn_type"]
    enumeration = cast(Mapping[str, int], _codec("struct_conn", "conn_type").enumeration)
    listed = np.array([symbol.lower() in enumeration for symbol in symbols] + [True])
    for row in np.flatnonzero(~listed[local]):
        builder.record("conn_type_enumeration", [builder.ref("struct_conn", int(row), "conn_type_id")])
    conn_type = np.where(listed[local], conn_type, -1).astype(np.int32)
    conn_type_state = np.where(listed[local], conn_type_state, _CONFLICT).astype(np.uint8)
    columns: dict[str, list] = {}
    field_state = np.zeros((n, 2, len(_ENDPOINT)), np.uint8)
    for k in (1, 2):
        for f, (column, family) in enumerate(_ENDPOINT):
            if column == "label_seq":
                value, states = c[f"label_seq_{k}"]
            elif column == "auth_seq":
                value, states, text = _number(builder, c[f"auth_seq_{k}"])
                columns.setdefault("auth_seq_text", []).append(text)
            else:
                value, states = _code(builder, cast(str, family), c[f"{column}_{k}"])
            columns.setdefault(column, []).append(value)
            field_state[:, k - 1, f] = states
    side = {column: np.stack(values, axis=1).reshape(n, 2) for column, values in columns.items()}
    symmetry = [_code(builder, "symmetry", c[f"symmetry_{k}"]) for k in (1, 2)]
    leaving = [_code(builder, "atom", c[f"leaving_{k}"]) for k in (1, 2)]
    role = [_code(builder, "text", c[f"role_{k}"]) for k in (1, 2)]
    order, order_state = c["order"]
    _refuse_unlisted(order, "struct_conn", "pdbx_value_order", block)
    flag, _ = c["leaving_flag"]
    _refuse_unlisted(flag, "struct_conn", "pdbx_leaving_atom_flag", block)
    pdbx_role, pdbx_role_state = _code(builder, "text", c["pdbx_role"])

    def stack(values: list[tuple], at: int) -> NDArray:
        return np.stack([value[at] for value in values], axis=1).reshape(n, 2)

    return ConnectionTable(
        np.full(n, _STRUCT_CONN, np.uint8),
        conn_type,
        conn_type_state,
        side["label_asym"],
        side["label_seq"],
        side["label_comp"],
        side["label_atom"],
        side["label_alt"],
        side["auth_asym"],
        side["auth_seq"],
        side["auth_seq_text"],
        side["auth_ins"],
        side["auth_comp"],
        side["auth_atom"],
        field_state,
        stack(symmetry, 0),
        stack(symmetry, 1),
        order.astype(np.uint8),
        order_state.astype(np.uint8),
        stack(leaving, 0),
        stack(leaving, 1),
        flag.astype(np.uint8),
        stack(role, 0),
        stack(role, 1),
        pdbx_role,
        pdbx_role_state,
        refs,
    )


def _modified(read, builder: TableBuilder) -> ModifiedResidueTable:
    n, c = read("pdbx_struct_mod_residue")
    label_comp, label_comp_state = _code(builder, "comp", c["label_comp"])
    label_seq, label_seq_state = c["label_seq"]
    return ModifiedResidueTable(
        *_code(builder, "chain", c["label_asym"]),
        label_seq,
        label_seq_state,
        *_code(builder, "chain", c["asym"]),
        *_number(builder, c["seq"]),
        _code(builder, "ins", c["ins"])[0],
        _code(builder, "comp", c["comp"])[0],
        _code(builder, "comp", c["parent"])[0],
        np.where(known(label_comp_state), label_comp, -1).astype(np.int32),
        builder.refs("pdbx_struct_mod_residue", range(n)),
    )


def _first(builder: TableBuilder, name: str, n: int, decoded: tuple, item: str, family: str | None = None):
    r"""Row 0 of a single-row category item: (value or code, state, reference); absent → (−1/0, UNKNOWN×NONE, −1)."""
    if not n:
        return (-1 if family else 0), _ABSENT, -1
    value = builder.code(family, decoded[0][:1], decoded[2])[0] if family else decoded[0][0]
    return int(value), int(decoded[1][0]), builder.ref(name, 0, item)


def _crystal(read, builder: TableBuilder) -> dict:
    n, c = read("cell")
    names = ("length_a", "length_b", "length_c", "angle_alpha", "angle_beta", "angle_gamma")
    cell = np.array([c[name][0][0] if n else 0.0 for name in names])
    cell_state = np.array([c[name][1][0] if n else _ABSENT for name in names], np.uint8)
    cell_quantum = np.array([c[name][2][0] / 2 if n else 0.0 for name in names])
    out = {
        "cell": cell,
        "cell_quantum": cell_quantum,
        "cell_state": cell_state,
        "cell_ref": builder.ref("cell", 0) if n else -1,
    }
    for column, items, dtype in (
        ("space_group", ("space_group_name_H-M", "name_H-M_alt"), np.int32),
        ("number", ("Int_Tables_number", "IT_number"), np.int64),
    ):  # every statement of the fact, one per category
        family = "space_group" if column == "space_group" else None
        statements = []
        for name, item in zip(("symmetry", "space_group"), items):
            rows, columns = read(name)
            if rows and columns[column][1][0] != _ABSENT:
                statements.append(_first(builder, name, rows, columns[column], item, family))
            else:
                statements.append((-1 if family else 0, _ABSENT, -1))
        value, value_state, ref = zip(*statements)
        out[column], out[f"{column}_state"] = np.array(value, dtype), np.array(value_state, np.uint8)
        out[f"{column}_ref"] = np.array(ref, np.int64)
    ids, texts, refs = [np.zeros(0, np.int32)], [np.zeros(0, np.int32)], [np.zeros(0, np.int64)]
    for name in ("space_group_symop", "symmetry_equiv"):  # every statement of the table, rows in file order
        rows, columns = read(name)
        if rows:
            ids.append(_code(builder, "operation_id", columns["operation_id"])[0])
            texts.append(_code(builder, "operation_text", columns["operation_text"])[0])
            refs.append(builder.refs(name, range(rows)))
    out["operation_id"] = np.concatenate(ids).astype(np.int32)
    out["operation_text"] = np.concatenate(texts).astype(np.int32)
    out["operation_ref"] = np.concatenate(refs).astype(np.int64)
    rows, columns = read("atom_sites")
    frame, quantum, frame_state = np.zeros((2, 3, 4)), np.zeros((2, 3, 4)), np.full((2, 3, 4), _ABSENT, np.uint8)
    for g, prefix in enumerate(("fract", "cartn")):
        for i in range(3):
            for j in range(4):
                if rows:
                    value, value_state, value_quantum, _ = columns[f"{prefix}_{i + 1}{j + 1}"]
                    frame[g, i, j], frame_state[g, i, j], quantum[g, i, j] = (
                        value[0],
                        value_state[0],
                        value_quantum[0] / 2,
                    )
    out |= {"deposited_frame": frame, "deposited_quantum": quantum, "deposited_state": frame_state}
    out["deposited_ref"] = np.array([builder.ref("atom_sites", 0) if rows else -1] * 2, np.int64)
    described = rows and known(columns["axes"][1][0])
    out["axes_ref"] = builder.ref("atom_sites", 0, "Cartn_transform_axes") if described else -1
    rows, columns = read("pdbx_database_status")
    processed = rows and _strings(columns["process_site"])[0] in _ARCHIVE_SITES
    out["wwpdb_ref"] = builder.ref("pdbx_database_status", 0, "process_site") if processed else -1
    return out


def _operators(read, builder: TableBuilder) -> OperatorTable:
    n, c = read("pdbx_struct_oper_list")
    matrix, quantum = np.full((n, 3, 4), np.nan), np.full((n, 3, 4), np.nan)
    for i in range(3):
        for j in range(4):
            value, value_state, value_quantum, _ = c[f"matrix_{i + 1}{j + 1}"]
            stated = known(value_state)
            matrix[stated, i, j], quantum[stated, i, j] = value[stated], value_quantum[stated] / 2
    identifier, _ = _code(builder, "operator", c["id"])
    _unique_key(identifier, "pdbx_struct_oper_list", "id", builder.block)
    return OperatorTable(
        identifier,
        _code(builder, "text", c["type"])[0],
        _code(builder, "text", c["name"])[0],
        matrix,
        quantum,
        builder.refs("pdbx_struct_oper_list", range(n)),
    )


def _assemblies(read, builder: TableBuilder) -> AssemblyTable:
    _, c = read("pdbx_struct_assembly")
    gen_n, gen = read("pdbx_struct_assembly_gen")
    return AssemblyTable(
        _code(builder, "assembly", c["id"])[0],
        _code(builder, "assembly", gen["assembly"])[0],
        _code(builder, "text", gen["expression"])[0],
        _code(builder, "text", gen["asyms"])[0],
        builder.refs("pdbx_struct_assembly_gen", range(gen_n)),
    )


def _experiment(read, builder: TableBuilder) -> ExperimentTable:
    method, resolution, kind, resolution_state, ref = [], [], [], [], []
    n, c = read("exptl")
    method.append(_code(builder, "text", c["method"])[0])
    resolution.append(np.zeros(n))
    kind.append(np.full(n, ResolutionKind.NONE, np.uint8))
    resolution_state.append(np.full(n, _ABSENT, np.uint8))
    ref.append(builder.refs("exptl", range(n)))
    for name, item, resolution_kind in (
        ("refine", "ls_d_res_high", ResolutionKind.REFINEMENT),
        ("reflns", "d_resolution_high", ResolutionKind.REFLECTIONS),
        ("em_3d_reconstruction", "resolution", ResolutionKind.EM_RECONSTRUCTION),
    ):
        n, c = read(name)
        value, value_state, _, _ = c["resolution"]
        method.append(_code(builder, "text", c["method"])[0] if "method" in c else np.full(n, -1, np.int32))
        rows = builder.refs(name, range(n), item)
        invalid = known(value_state) & ~(value > 0)
        for row in np.flatnonzero(invalid):
            builder.record("resolution_invalid", [int(rows[row])])
        resolution.append(value)
        kind.append(np.full(n, resolution_kind, np.uint8))
        resolution_state.append(np.where(invalid, _UNSTATED, value_state).astype(np.uint8))  # never refused
        ref.append(rows)
    return ExperimentTable(*(np.concatenate(part) for part in (method, resolution, kind, resolution_state, ref)))


def decode_components(
    categories: Mapping[str, CifCategory], *, block: str | None, basis: FactBasis = MappingProxyType({})
) -> DefinitionRows:
    r"""
    The ``chem_comp*`` rows of one block (embedded definitions, or one CCD entry) as ``ComponentItems`` per component
    id, in ``_chem_comp`` order then first appearance: text items as bytes and codec states, numbers
    with their printed quantum, ``type_symbol`` as atomic number and isotope (an unknown symbol reads as 0), the
    default token filled for an absent item with a verified default.

    Raises:
        ContradictionError: ``key_invalid`` for a ``_chem_comp.id`` stated twice or not stated, or a row with no
            ``comp_id``; ``number_grammar`` for a number outside the CIF grammar.
    """
    names = ("chem_comp", "chem_comp_atom", "chem_comp_bond", "pdbx_chem_comp_descriptor")
    read = {name: _read(categories, name, block, basis) for name in names}
    n, comp = read["chem_comp"]
    ids = [value.decode() if stated else None for value, stated in zip(comp["id"][0].tolist(), known(comp["id"][1]))]
    first: dict[str, int] = {}
    for row, identifier in enumerate(ids):
        if identifier is None or first.setdefault(identifier, row) != row:
            fail("key_invalid", SourceRef("chem_comp", row, "id", block))
    groups = {name: _groups(read[name][1]) for name in names[1:]}
    order = list(dict.fromkeys([*first, *(cid for group in groups.values() for cid in group)]))
    components = {}
    for cid in order:
        rows: dict[str, Sequence[int] | NDArray[np.int64]] = {"chem_comp": [first[cid]] if cid in first else []}
        rows |= {name: groups[name].get(cid, np.zeros(0, np.int64)) for name in names[1:]}
        components[cid] = _component(read, rows, block)
    return DefinitionRows(MappingProxyType(components))


def _groups(columns: Columns) -> dict[str, NDArray[np.int64]]:
    r"""The rows of each ``comp_id`` in first-appearance order."""
    values, _ = columns["comp_id"]
    out: dict[str, list[int]] = {}
    for row, value in enumerate(values.tolist()):
        out.setdefault(value.decode(), []).append(row)
    return {cid: np.array(rows, np.int64) for cid, rows in out.items()}


def _component(read: dict, rows: Mapping[str, Sequence[int] | NDArray[np.int64]], block: str | None) -> ComponentItems:
    _, comp = read["chem_comp"]
    component = {}
    if len(rows["chem_comp"]):
        row = rows["chem_comp"][0]
        for item, (values, states) in comp.items():
            component[item] = (values[row].decode() if known(states[row]) else None, int(states[row]))
    _, atom = read["chem_comp_atom"]
    at = rows["chem_comp_atom"]
    atoms, numbers = {}, {}
    for item in MMCIF_CODECS["chem_comp_atom"].items:
        if item.kind == "text" and item.column != "comp_id":
            values, states = atom[item.column]
            atoms[item.column] = (values[at], states[at])
        elif item.kind == "float":
            value, states, quantum, _ = atom[item.column]
            numbers[item.column] = (value[at], states[at], quantum[at])
    charge, charge_state = atom["charge"]
    numbers["charge"] = (charge[at].astype(np.float64), charge_state[at], np.where(known(charge_state[at]), 1.0, 0.0))
    symbols, symbol_states = atom["type_symbol"]
    z, isotope = _elements((symbols[at], symbol_states[at]), None)
    _, bond = read["chem_comp_bond"]
    bonds = {
        item: (values[rows["chem_comp_bond"]], states[rows["chem_comp_bond"]])
        for item, (values, states) in bond.items()
    }
    del bonds["comp_id"]
    _, descriptor = read["pdbx_chem_comp_descriptor"]
    texts = [_strings(descriptor[item]) for item in ("type", "program", "descriptor")]
    picked = cast(NDArray[np.int64], rows["pdbx_chem_comp_descriptor"]).tolist()
    descriptors = tuple((texts[0][r] or "", texts[1][r] or "", texts[2][r] or "") for r in picked)
    order = ("chem_comp", "chem_comp_atom", "chem_comp_bond", "pdbx_chem_comp_descriptor")
    lengths = [len(rows[name]) for name in order]
    refs = SourceRefs(
        np.repeat(np.arange(len(order), dtype=np.int16), lengths),
        np.concatenate([np.asarray(rows[name], np.int64) for name in order]),
        np.full(sum(lengths), -1, np.int16),
        order,
        (),
        block,
    )
    element = (z, isotope, symbol_states[at])
    for column in (*atoms.values(), *numbers.values(), *bonds.values(), element):
        for array in column:  # owning copies (fancy indexing), made read-only
            array.flags.writeable = False
    return ComponentItems(
        MappingProxyType(component),
        MappingProxyType(atoms),
        MappingProxyType(numbers),
        element,
        MappingProxyType(bonds),
        descriptors,
        refs,
    )


def _strings(decoded: tuple) -> list[str | None]:
    r"""A text or code item as one string per row (None where not a stated value): document facts, a few rows."""
    if len(decoded) == 3:
        codes, _, symbols = decoded
        return [symbols[code] if code >= 0 else None for code in codes.tolist()]
    values, states = decoded
    return [value.decode() if stated else None for value, stated in zip(values.tolist(), known(states))]


def mmcif_metadata(categories: Mapping[str, CifCategory]) -> DocumentMetadata:
    r"""The document facts of one block: ``release_date`` is the revision date of revision 1."""

    def read(name: str) -> Columns:
        return _read(categories, name, None)[1]

    def first(name: str, column: str) -> str | None:
        values = _strings(read(name)[column])
        return values[0] if values else None

    history = read("pdbx_audit_revision_history")
    ordinal, ordinal_state = history["ordinal"]
    dates = _strings(history["revision_date"])
    release = next((d for o, s, d in zip(ordinal.tolist(), known(ordinal_state), dates) if s and o == 1), None)
    poly = read("entity_poly")
    polymers = {
        entity: (kind, sequence.replace("\n", "").replace(" ", "") if sequence else sequence)
        for entity, kind, sequence in zip(*(_strings(poly[column]) for column in ("entity", "type", "sequence")))
    }
    entity = read("entity")
    kinds = [EntityType(k).name.lower().replace("_", "-") if k > 0 else "" for k in entity["type"][0].tolist()]
    entities = tuple(
        (cid or "", kind, description, *polymers.get(cid, (None, None)))
        for cid, kind, description in zip(_strings(entity["id"]), kinds, _strings(entity["description"]))
    )
    return DocumentMetadata(first("entry", "id"), first("pdbx_database_status", "deposition_date"), release, entities)
