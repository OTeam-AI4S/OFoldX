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
The format-independent typed source tables both decoders emit.

Every table is columns: identifiers as ``int32`` codes into per-source symbol families (``AtomTable.symbols``, the
same mapping as ``CrystalTable.symbols``), states as ``uint8`` fact-state bytes (``ofoldx.provenance``), values as
``float64``, row references as ``int64`` rows of ``SourceTables.refs``. Every item that can be ``?``, ``.`` or absent
carries a state decoded at this boundary: a code is −1 wherever its state is not a stated value, and a code
column alone never stands for a null. Families are shared across tables, so joins are integer joins:

- ``chain`` asym ids and author chain ids; ``entity``; ``comp`` component ids; ``atom`` atom names; ``alt`` altloc
  labels; ``ins`` insertion codes; ``number`` author residue numbers as text; ``conn_type``; ``symmetry`` symmetry codes
  as text; ``text`` free text (descriptions, roles, methods, expressions, lists); ``polymer_type``; ``assembly``;
  ``operator``; ``space_group``, ``operation_id``, ``operation_text`` (the crystal table).

A per-model view (``model_view``) is the same tables: ``AtomTable`` holds every model's rows, distinguished by
``model``; only the cross-category checks of the selected model and the records of the other models differ.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

from ..chemistry.definitions import DefinitionRows
from ..data.schema import occupancy_in_range
from ..errors import RequestRef, SourceRef, fail
from ..provenance import Basis, Records, Resolution, SourceRefs, SubjectKind, known, state

_CONFLICT = state(Resolution.CONFLICT, Basis.NONE)


class _Columns:
    def __post_init__(self) -> None:
        for f in fields(self):  # type: ignore[arg-type]
            value = getattr(self, f.name)
            if isinstance(value, np.ndarray):
                value.flags.writeable = False


@dataclass(frozen=True, eq=False)
class AtomTable(_Columns):
    r"""
    The atom rows of every model (``_atom_site`` / ``ATOM``, ``HETATM``). ``row`` is the category row (mmCIF) or the
    zero-based line (PDB); ``serial`` the PDB atom serial ``CONECT`` names (−1 for mmCIF, whose connections name atoms
    by identity). ``group``: 0 not stated, 1 ATOM, 2 HETATM. Label and author namespaces are separate columns
    and are never spliced; PDB fills the author namespace only, its altloc in ``label_alt`` (the format's one altloc).
    ``auth_seq`` is the integer reading of the author number text ``auth_seq_text`` (UNKNOWN×DERIVED when the text is
    no integer). ``element`` is the atomic number (0 for ``X``), ``isotope`` 2/3 for ``D``/``T``, else 0.
    ``coordinate_state`` is one state for the three axes (CONFLICT×NONE when they disagree); ``coordinate_quantum`` the
    largest printed half-unit of the three. ``occupancy_uncertainty`` is the occupancy's standard uncertainty u as the
    source states it (its parenthesised su or ``_atom_site.occupancy_esd``, CONFLICT×NONE when both differ; PDB states
    none: UNKNOWN×NONE).
    ``charge_state`` follows the codec: an mmCIF ``atom_site`` loop without
    ``pdbx_formal_charge`` reads KNOWN×DEFAULT 0; a blank PDB charge is UNKNOWN×DECLARED.
    """

    row: NDArray[np.int64]
    serial: NDArray[np.int64]
    model: NDArray[np.int32]
    group: NDArray[np.uint8]
    label_asym: NDArray[np.int32]
    label_asym_state: NDArray[np.uint8]
    label_entity: NDArray[np.int32]
    label_entity_state: NDArray[np.uint8]
    label_comp: NDArray[np.int32]
    label_comp_state: NDArray[np.uint8]
    label_atom: NDArray[np.int32]
    label_atom_state: NDArray[np.uint8]
    label_alt: NDArray[np.int32]
    label_alt_state: NDArray[np.uint8]
    label_seq: NDArray[np.int64]
    label_seq_state: NDArray[np.uint8]
    auth_asym: NDArray[np.int32]
    auth_asym_state: NDArray[np.uint8]
    auth_comp: NDArray[np.int32]
    auth_comp_state: NDArray[np.uint8]
    auth_atom: NDArray[np.int32]
    auth_atom_state: NDArray[np.uint8]
    auth_seq: NDArray[np.int64]
    auth_seq_state: NDArray[np.uint8]
    auth_seq_text: NDArray[np.int32]
    insertion: NDArray[np.int32]
    insertion_state: NDArray[np.uint8]
    element: NDArray[np.uint8]
    isotope: NDArray[np.uint16]
    coordinates: NDArray[np.float64]
    coordinate_state: NDArray[np.uint8]
    coordinate_quantum: NDArray[np.float32]
    occupancy: NDArray[np.float64]
    occupancy_state: NDArray[np.uint8]
    occupancy_uncertainty: NDArray[np.float64]
    occupancy_uncertainty_state: NDArray[np.uint8]
    b_factor: NDArray[np.float64]
    b_factor_state: NDArray[np.uint8]
    charge: NDArray[np.int8]
    charge_state: NDArray[np.uint8]
    symbols: Mapping[str, tuple[str, ...]]


@dataclass(frozen=True, eq=False)
class EntityTable(_Columns):
    r"""``_entity`` / ``COMPND MOL_ID``; ``type`` an ``EntityType`` (PDB: UNKNOWN×NONE)."""

    id: NDArray[np.int32]
    type: NDArray[np.uint8]
    type_state: NDArray[np.uint8]
    description: NDArray[np.int32]
    ref: NDArray[np.int64]


@dataclass(frozen=True, eq=False)
class PolymerTable(_Columns):
    r"""``_entity_poly``. ``type`` codes into ``polymer_type``; ``nstd_linkage`` an ``NstdLinkage`` with no default.
    PDB has no rows: its polymer types are derived from SEQRES (``io.binding.pdb_polymer_types``)."""

    entity: NDArray[np.int32]
    type: NDArray[np.int32]
    type_state: NDArray[np.uint8]
    nstd_linkage: NDArray[np.uint8]
    nstd_linkage_state: NDArray[np.uint8]
    ref: NDArray[np.int64]


@dataclass(frozen=True, eq=False)
class SequenceTable(_Columns):
    r"""``_entity_poly_seq`` / ``SEQRES`` (PDB: ``entity`` is the chain code, ``num`` the position from 1)."""

    entity: NDArray[np.int32]
    num: NDArray[np.int32]
    mon_id: NDArray[np.int32]
    hetero: NDArray[np.bool_]
    ref: NDArray[np.int64]


@dataclass(frozen=True, eq=False)
class AsymTable(_Columns):
    r"""``_struct_asym`` in file order (``order``); PDB: one row per ``COMPND CHAIN`` (author chains)."""

    id: NDArray[np.int32]
    entity: NDArray[np.int32]
    order: NDArray[np.int32]
    ref: NDArray[np.int64]


@dataclass(frozen=True, eq=False)
class SchemeTable(_Columns):
    r"""
    The three schemes: ``kind`` 1 ``_pdbx_poly_seq_scheme`` (``number`` = ``seq_id``), 2 ``_pdbx_nonpoly_scheme``
    (``number`` 0), 3 ``_pdbx_branch_scheme`` (``number`` = ``num``; ``pdb_strand_id`` = ``pdb_asym_id``).
    ``pdb_seq_num`` is the integer reading of its text ``pdb_seq_text``.
    """

    kind: NDArray[np.uint8]
    asym: NDArray[np.int32]
    entity: NDArray[np.int32]
    number: NDArray[np.int32]
    mon_id: NDArray[np.int32]
    pdb_seq_num: NDArray[np.int64]
    pdb_seq_state: NDArray[np.uint8]
    pdb_seq_text: NDArray[np.int32]
    pdb_ins_code: NDArray[np.int32]
    pdb_ins_state: NDArray[np.uint8]
    pdb_strand_id: NDArray[np.int32]
    pdb_strand_state: NDArray[np.uint8]
    ref: NDArray[np.int64]


@dataclass(frozen=True, eq=False)
class BranchLinkTable(_Columns):
    r"""``_pdbx_entity_branch_link``; ``comp``, ``num``, ``atom``, ``leaving`` are ``[n, 2]`` (per side; ``leaving`` −1
    where not stated)."""

    entity: NDArray[np.int32]
    comp: NDArray[np.int32]
    num: NDArray[np.int32]
    atom: NDArray[np.int32]
    leaving: NDArray[np.int32]
    order: NDArray[np.uint8]
    order_state: NDArray[np.uint8]
    ref: NDArray[np.int64]


@dataclass(frozen=True, eq=False)
class ConnectionTable(_Columns):
    r"""
    ``_struct_conn``, PDB ``SSBOND`` and ``LINK``. ``origin`` is the ``io.binding.QuerySource`` value (1 struct_conn,
    2 SSBOND, 3 LINK). Endpoint columns are ``[n, 2]``; ``field_state`` ``[n, 2, 10]`` holds the state of label asym,
    seq, comp, atom, alt, then author asym, seq, ins, comp, atom; ``auth_seq`` is the integer reading of the author
    number text ``auth_seq_text`` (as ``AtomTable``). ``symmetry`` codes the code text (PDB six digits as
    ``n_klm``); its state is KNOWN×DECLARED, KNOWN×DEFINED (``.``; a blank ``LINK`` field: identity), UNKNOWN×DECLARED
    (``?``) or UNKNOWN×NONE (absent; a blank ``SSBOND`` field). ``order`` 1–4 (``value_order`` sing–quad; another stated
    order is UNKNOWN×DECLARED). ``leaving_flag``: 0 not stated, 1 none, 2 one, 3 both.
    """

    origin: NDArray[np.uint8]
    conn_type: NDArray[np.int32]
    conn_type_state: NDArray[np.uint8]
    label_asym: NDArray[np.int32]
    label_seq: NDArray[np.int64]
    label_comp: NDArray[np.int32]
    label_atom: NDArray[np.int32]
    label_alt: NDArray[np.int32]
    auth_asym: NDArray[np.int32]
    auth_seq: NDArray[np.int64]
    auth_seq_text: NDArray[np.int32]
    auth_ins: NDArray[np.int32]
    auth_comp: NDArray[np.int32]
    auth_atom: NDArray[np.int32]
    field_state: NDArray[np.uint8]
    symmetry: NDArray[np.int32]
    symmetry_state: NDArray[np.uint8]
    order: NDArray[np.uint8]
    order_state: NDArray[np.uint8]
    leaving: NDArray[np.int32]
    leaving_state: NDArray[np.uint8]
    leaving_flag: NDArray[np.uint8]
    role: NDArray[np.int32]
    role_state: NDArray[np.uint8]
    pdbx_role: NDArray[np.int32]
    pdbx_role_state: NDArray[np.uint8]
    ref: NDArray[np.int64]


@dataclass(frozen=True, eq=False)
class ConectTable(_Columns):
    r"""PDB ``CONECT`` pairs: ``serial`` ``[n, 2]`` (the record's atom, the listed atom); ``contact`` for the v2.x
    hydrogen-bond and salt-bridge columns."""

    serial: NDArray[np.int64]
    contact: NDArray[np.bool_]
    ref: NDArray[np.int64]


@dataclass(frozen=True, eq=False)
class ModifiedResidueTable(_Columns):
    r"""``_pdbx_struct_mod_residue`` / ``MODRES`` in the author namespace both formats state: ``asym`` the author chain,
    ``seq`` the integer reading of the author number text ``seq_text`` (with its state, as ``AtomTable.auth_seq``),
    ``comp`` the modified component, ``parent`` the standard parent; and the label address mmCIF may state
    (``label_asym``, ``label_seq``, ``label_comp`` −1 where not stated; PDB states none: UNKNOWN×NONE)."""

    label_asym: NDArray[np.int32]
    label_asym_state: NDArray[np.uint8]
    label_seq: NDArray[np.int64]
    label_seq_state: NDArray[np.uint8]
    asym: NDArray[np.int32]
    asym_state: NDArray[np.uint8]
    seq: NDArray[np.int64]
    seq_state: NDArray[np.uint8]
    seq_text: NDArray[np.int32]
    ins: NDArray[np.int32]
    comp: NDArray[np.int32]
    parent: NDArray[np.int32]
    label_comp: NDArray[np.int32]
    ref: NDArray[np.int64]


@dataclass(frozen=True, eq=False)
class UnobservedTable(_Columns):
    r"""PDB ``REMARK 465``; ``model`` 0 where the row states none; ``seq`` the integer reading of the author number
    text ``seq_text``."""

    model: NDArray[np.int32]
    chain: NDArray[np.int32]
    comp: NDArray[np.int32]
    seq: NDArray[np.int64]
    seq_text: NDArray[np.int32]
    ins: NDArray[np.int32]
    ref: NDArray[np.int64]


@dataclass(frozen=True, eq=False)
class CrystalTable(_Columns):
    r"""
    The crystal statements, each statement of one fact kept (``io.symmetry`` decides between them): ``cell`` a, b, c,
    α, β, γ with printed half-units; ``space_group`` codes each statement of the full H-M name
    (``_symmetry.space_group_name_H-M`` and ``_space_group.name_H-M_alt``; ``CRYST1``), ``number`` each of the
    ``Int_Tables_number`` (``_symmetry.Int_Tables_number`` and ``_space_group.IT_number``), absent ones UNKNOWN×NONE;
    the rows of every in-file operation table (``_space_group_symop`` then ``_symmetry_equiv``; ``REMARK 290``) in
    file order (``operation_id`` and ``operation_text`` −1 for a row that states none); ``deposited_frame``
    ``[2, 3, 4]`` the stated transforms with printed half-units: the fractionalisation [A | u], f = A·x + u
    (``_atom_sites.fract_transf_matrix``/``vector``, ``SCALEn``), and its inverse (``_atom_sites.Cartn_transf_*``; PDB
    states none, and ``ORIGXn`` is no frame: it maps the entry's coordinates to the submitted ones). ``axes_ref``: a
    stated ``_atom_sites.Cartn_transform_axes`` (−1 none, ``?`` and ``.`` included; PDB states none). ``wwpdb_ref``: a
    ``_pdbx_database_status.process_site`` naming a wwPDB archive site (BNL, NDB, PDBC, PDBE, PDBJ, RCSB), the site
    that processed the entry, whose ``n_klm`` codes wwPDB numbers in the ITA order of the setting whatever the order of
    its operation table (−1 none: absent, ``?``, ``.`` and any other value; PDB states none:
    ``REMARK 290`` numbers its codes). The marker is not proof that the file is unmodified. The ``*_ref`` items index
    ``SourceTables.refs``, −1 none.
    """

    cell: NDArray[np.float64]
    cell_quantum: NDArray[np.float64]
    cell_state: NDArray[np.uint8]
    space_group: NDArray[np.int32]
    space_group_state: NDArray[np.uint8]
    number: NDArray[np.int64]
    number_state: NDArray[np.uint8]
    operation_id: NDArray[np.int32]
    operation_text: NDArray[np.int32]
    operation_ref: NDArray[np.int64]
    deposited_frame: NDArray[np.float64]
    deposited_quantum: NDArray[np.float64]
    deposited_state: NDArray[np.uint8]
    symbols: Mapping[str, tuple[str, ...]]
    space_group_ref: NDArray[np.int64]
    number_ref: NDArray[np.int64]
    cell_ref: int
    deposited_ref: NDArray[np.int64]
    axes_ref: int
    wwpdb_ref: int


@dataclass(frozen=True, eq=False)
class OperatorTable(_Columns):
    r"""``_pdbx_struct_oper_list`` / ``REMARK 350 BIOMT``: ``matrix`` ``[n, 3, 4]`` (``nan`` where not stated) with
    printed half-units. PDB operators are numbered ``1…`` by first appearance of each distinct printed matrix."""

    id: NDArray[np.int32]
    type: NDArray[np.int32]
    name: NDArray[np.int32]
    matrix: NDArray[np.float64]
    quantum: NDArray[np.float64]
    ref: NDArray[np.int64]


@dataclass(frozen=True, eq=False)
class AssemblyTable(_Columns):
    r"""``_pdbx_struct_assembly`` and its generator rows ``_pdbx_struct_assembly_gen`` (``gen_assembly`` codes the
    assembly id, ``gen_asyms`` the asym list text); PDB: one assembly per ``REMARK 350 BIOMOLECULE``, one generator row
    per chain list (the blank chain's id blank)."""

    id: NDArray[np.int32]
    gen_assembly: NDArray[np.int32]
    gen_expression: NDArray[np.int32]
    gen_asyms: NDArray[np.int32]
    gen_ref: NDArray[np.int64]


@dataclass(frozen=True, eq=False)
class ExperimentTable(_Columns):
    r"""Method rows (``_exptl.method`` / ``EXPDTA``; ``resolution_kind`` NONE) and resolution rows (a
    ``ResolutionKind``; ``method`` −1 except ``_refine.pdbx_refine_id``)."""

    method: NDArray[np.int32]
    resolution: NDArray[np.float64]
    resolution_kind: NDArray[np.uint8]
    resolution_state: NDArray[np.uint8]
    ref: NDArray[np.int64]


@dataclass(frozen=True)
class DocumentMetadata:
    r"""Document facts: the entry id, the dates and per entity its id, type, description,
    polymer type and sequence; None or empty where the source states none."""

    entry_id: str | None
    deposition_date: str | None
    release_date: str | None
    entities: tuple[tuple[str, str, str | None, str | None, str | None], ...]


@dataclass(frozen=True, eq=False)
class SourceTables:
    r"""The tables of one source; ``model_state`` is the state of its model numbering: KNOWN×DECLARED when stated
    (``pdbx_PDB_model_num``, a ``MODEL`` record), else UNKNOWN×NONE; ``terminators`` the lines of the PDB ``TER``
    records in file order (mmCIF states none)."""

    atoms: AtomTable
    model_state: int
    entities: EntityTable
    polymers: PolymerTable
    sequences: SequenceTable
    asyms: AsymTable
    schemes: SchemeTable
    branch_links: BranchLinkTable
    connections: ConnectionTable
    conect: ConectTable
    modified: ModifiedResidueTable
    unobserved: UnobservedTable
    terminators: NDArray[np.int64]
    crystal: CrystalTable
    operators: OperatorTable
    assemblies: AssemblyTable
    experiment: ExperimentTable
    definitions: DefinitionRows
    refs: SourceRefs
    records: Records


class TableBuilder:
    r"""The symbol families, source references and records a decoder builds the tables of one source with."""

    def __init__(self, block: str | None) -> None:
        self.block = block
        self._families: dict[str, dict[str, int]] = {}
        self._categories: dict[str, int] = {}
        self._items: dict[str, int] = {}
        self._refs: list[tuple[NDArray, NDArray, NDArray]] = []
        self._count = 0
        self._records: list[tuple] = []

    def code(self, family: str, codes: NDArray[np.int32], symbols: Sequence[str]) -> NDArray[np.int32]:
        r"""Local codes (−1 none) into ``symbols`` as codes into the family."""
        table = self._families.setdefault(family, {})
        remap = np.array([table.setdefault(symbol, len(table)) for symbol in symbols] + [-1], dtype=np.int32)
        return remap[codes]

    def symbol(self, family: str, text: str) -> int:
        table = self._families.setdefault(family, {})
        return table.setdefault(text, len(table))

    def refs(self, category: str, rows: NDArray | Sequence[int], item: str | None = None) -> NDArray:
        r"""Append one reference per row; returns their indices in ``SourceRefs``."""
        rows = np.asarray(rows, dtype=np.int64).reshape(-1)
        n = len(rows)
        category_code = self._categories.setdefault(category, len(self._categories))
        item_code = -1 if item is None else self._items.setdefault(item, len(self._items))
        self._refs.append((np.full(n, category_code, np.int16), rows, np.full(n, item_code, np.int16)))
        self._count += n
        return np.arange(self._count - n, self._count, dtype=np.int64)

    def ref(self, category: str, row: int, item: str | None = None) -> int:
        return int(self.refs(category, [row], item)[0])

    def record(self, code: str, refs: Sequence[int], *payload: object) -> None:
        r"""One record about the source (subject NONE); a TEXT payload slot takes a str."""
        self._records.append((code, SubjectKind.NONE, -1, refs, payload))

    def symbols(self) -> Mapping[str, tuple[str, ...]]:
        return MappingProxyType({family: tuple(table) for family, table in self._families.items()})

    def source_refs(self) -> SourceRefs:
        dtypes = (np.int16, np.int64, np.int16)
        columns = [
            np.concatenate([part[i] for part in self._refs] or [np.zeros(0)]).astype(t) for i, t in enumerate(dtypes)
        ]
        return SourceRefs(columns[0], columns[1], columns[2], tuple(self._categories), tuple(self._items), self.block)

    def records(self, refs: SourceRefs) -> Records:
        return Records.from_rows(self._records, refs)


def models(atoms: AtomTable) -> tuple[int, ...]:
    r"""The model numbers of the atom rows in file order; ``(1,)`` for a source without atoms."""
    numbers, first = np.unique(atoms.model, return_index=True)
    return tuple(int(number) for number in numbers[np.argsort(first)]) or (1,)


_ENTRY_LEVEL = "entry_level_violation"


def model_view(tables: SourceTables, model: int) -> SourceTables:
    r"""
    The view of one model: the cross-category checks raise for the selected model's atom rows and are
    recorded as ``entry_level_violation`` (the row's model, the check's code and the row's asym in the payload) for
    every other model's (1ts6).

    Raises:
        InvalidRequestError: ``model_unknown`` when no atom row has the model.
        ContradictionError: ``coordinate_axis_partial``, ``occupancy_uncertainty_conflict``,
            ``occupancy_uncertainty_negative``, ``occupancy_out_of_range`` or ``asym_undeclared`` for a row of the
            selected model.
    """
    if model not in models(tables.atoms):
        fail("model_unknown", RequestRef("model", str(model)))
    atoms, refs = tables.atoms, tables.refs
    builder = TableBuilder(refs.block)
    for code, item, rows, payload in _violations(tables):
        selected = rows[atoms.model[rows] == model]
        if len(selected):
            row = int(selected[0])
            fail(code, _atom_ref(tables, row, item), **(payload(row) if payload else {}))
        for row in rows:
            category, line = _atom_address(tables, int(row))
            asym = atoms.symbols["chain"][atoms.label_asym[row]] if known(atoms.label_asym_state[row]) else ""
            builder.record(_ENTRY_LEVEL, [builder.ref(category, line, item)], int(atoms.model[row]), code, asym)
    kept = [tables.records.where(code=code) for code in tables.records.counts() if code != _ENTRY_LEVEL]
    records = Records.concat([*kept, builder.records(builder.source_refs())])
    values = {f.name: getattr(tables, f.name) for f in fields(SourceTables)}
    return SourceTables(**{**values, "records": records})


def _violations(tables: SourceTables) -> list[tuple[str, str, NDArray[np.int64], Callable[[int], dict] | None]]:
    r"""Per check: its code, the item it names, the atom rows (table positions) that fail it, and the payload of its
    failure at a row (an occupancy's range is a Structure invariant, ``data.schema.occupancy_in_range``)."""
    atoms, asyms = tables.atoms, tables.asyms
    occupancy, su, su_state = atoms.occupancy, atoms.occupancy_uncertainty, atoms.occupancy_uncertainty_state

    def occupancy_payload(row: int) -> dict:  # the value and its su (None: Unknown)
        return {"value": float(occupancy[row]), "uncertainty": float(su[row]) if known(su_state[row]) else None}

    def su_payload(row: int) -> dict:
        return {"value": float(su[row])}

    # as Structure requires, no state goes unchecked: a stated su is valid whatever the occupancy; a known one in range
    invalid = known(su_state) & ~(np.isfinite(su) & (su >= 0))
    outside = known(atoms.occupancy_state) & ~invalid & ~occupancy_in_range(occupancy, su, su_state)
    out: list[tuple[str, str, NDArray[np.int64], Callable[[int], dict] | None]] = [
        ("coordinate_axis_partial", "Cartn_x", np.flatnonzero(atoms.coordinate_state == _CONFLICT), None),
        ("occupancy_uncertainty_conflict", "occupancy_esd", np.flatnonzero(su_state == _CONFLICT), None),
        ("occupancy_uncertainty_negative", "occupancy_esd", np.flatnonzero(invalid), su_payload),
        ("occupancy_out_of_range", "occupancy", np.flatnonzero(outside), occupancy_payload),
    ]
    stated = known(atoms.label_asym_state)
    if len(asyms.id) and stated.any():
        order = np.argsort(asyms.id, kind="stable")
        position = np.searchsorted(asyms.id[order], atoms.label_asym).clip(max=len(order) - 1)
        found = stated & (asyms.id[order][position] == atoms.label_asym)
        out.append(("asym_undeclared", "label_asym_id", np.flatnonzero(stated & ~found), None))
    return out


def _atom_address(tables: SourceTables, position: int) -> tuple[str, int]:
    r"""The category and row of an atom row: ``atom_site`` (mmCIF, which has a block) or the PDB record."""
    atoms = tables.atoms
    if tables.refs.block is not None:
        return "atom_site", int(atoms.row[position])
    return ("HETATM" if atoms.group[position] == 2 else "ATOM"), int(atoms.row[position])


def _atom_ref(tables: SourceTables, position: int, item: str) -> SourceRef:
    category, row = _atom_address(tables, position)
    return SourceRef(category, row, item, tables.refs.block, int(tables.atoms.model[position]))
