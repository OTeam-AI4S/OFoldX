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
The exception base, the five failure classes, the references every failure and record names, and the closed code
table.

Every named condition OFoldX detects is one row of ``CODES``: the row fixes whether it is raised or recorded, the
class it raises, the record kind and cause it is reported under, the basis that defines it and the typed payload of
its records. ``fail`` is the only raise of a typed failure; internal defects stay ``AssertionError``/``RuntimeError``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, fields
from enum import Enum, IntEnum, StrEnum
from functools import partial
from types import MappingProxyType
from typing import NoReturn, Protocol, cast


@dataclass(frozen=True, slots=True)
class SourceRef:
    r"""A statement of a source: a category row (mmCIF), a zero-based line (PDB), ``("smiles", 0, item)``, an artifact
    entry."""

    category: str
    row: int
    item: str | None = None
    block: str | None = None
    model: int | None = None


@dataclass(frozen=True, slots=True)
class StructureRef:
    r"""A part of a value: named by writer, operation and preparation refusals."""

    instance: int | None = None
    component: int | None = None
    atom: int | None = None
    external: int | None = None
    item: str | None = None


@dataclass(frozen=True, slots=True)
class RequestRef:
    r"""One caller argument and the value it was given."""

    argument: str
    value: str


_Ref = SourceRef | StructureRef | RequestRef


class OFoldXError(Exception):
    r"""
    A detected condition OFoldX refuses to continue past.

    The class is fixed by the code: ``type(error) is CODES[error.code].error``. The message is rendered from the code
    row, the references and the payload, never written at the raise site.
    """

    code: str
    refs: tuple[_Ref, ...]
    detail: Mapping[str, object]

    def __init__(self, code: str, refs: Sequence[_Ref] = (), /, **detail: object) -> None:
        row = CODES[code]
        assert row.error is type(self), f"{code} raises {row.error}, not {type(self).__name__}"
        self.code = code
        self.refs = tuple(refs)
        self.detail = detail
        super().__init__(_message(row, self.refs, detail))

    def __reduce__(self):
        # the message alone cannot rebuild the error (e.g. across processes)
        return partial(type(self), self.code, self.refs, **self.detail), ()


class ContradictionError(OFoldXError):
    r"""Statements contradict each other, the dictionary, the format or a definition."""


class InsufficientInformationError(OFoldXError):
    r"""The requested result is not unique: no interpretation, or several (``candidates``)."""

    def __init__(
        self,
        code: str,
        refs: Sequence[_Ref] = (),
        /,
        *,
        candidates: Iterable[SourceRef | StructureRef] = (),
        **detail: object,
    ) -> None:
        candidates = tuple(candidates)
        assert len(candidates) != 1, f"{code}: one candidate is a unique result"
        super().__init__(code, refs, candidates=candidates, **detail)

    @property
    def candidates(self) -> tuple[SourceRef | StructureRef, ...]:
        r"""``()`` when no interpretation exists; two or more when the evidence admits several."""
        return cast(tuple[SourceRef | StructureRef, ...], self.detail["candidates"])


class UnsupportedError(OFoldXError):
    r"""A well-defined fact or request OFoldX cannot represent ("we", never "the file")."""


class InvalidRequestError(OFoldXError):
    r"""Caller arguments or caller-built values are inconsistent."""


class ResourceLimitError(OFoldXError):
    r"""A finite search or the machine ended without a result."""


class Disposition(StrEnum):
    RAISE = "raise"  # detected → raise at once
    RECORD = "record"  # detected → a record of the read/operation report


class RecordKind(IntEnum):
    CONFLICT = 0
    UNKNOWN = 1
    DERIVED = 2
    UNREALISED = 3
    FINDING = 4


class Cause(IntEnum):
    r"""The accounting cause of an Unknown record; ``NONE`` for every other code and for the Unknowns accounted with
    another record (counted once, under that record)."""

    NONE = 0
    INSUFFICIENT = 1
    AMBIGUOUS = 2
    UNSUPPORTED = 3


class PayloadKind(IntEnum):
    r"""How a payload slot is read: a plain integer, an index into ``Records.texts``, an atom or component of
    the returned value, a declaration row, or an index into ``CODE_NAMES``."""

    INT = 0
    TEXT = 1
    ATOM = 2
    COMPONENT = 3
    DECLARATION = 4
    CODE = 5


@dataclass(frozen=True, slots=True)
class Code:
    r"""One row of the closed code table."""

    name: str
    error: type[OFoldXError] | None  # None only for record codes never raised
    disposition: Disposition
    kind: RecordKind | None  # None for raise-only codes
    cause: Cause
    basis: str  # the dictionary item, format clause or rule that defines the condition
    witness: str | None  # corpus entry demonstrating the condition
    payload: tuple[tuple[str, PayloadKind], ...] = ()  # <= 4 named fixed slots of a record of this code
    payload_list: tuple[str, PayloadKind] | None = None  # one variable-length list field (CSR)


def _message(row: Code, refs: tuple[_Ref, ...], detail: Mapping[str, object]) -> str:
    parts = [f"{row.name}: {row.basis}"]
    if refs:
        parts.append("at " + ", ".join(_text(ref) for ref in refs))
    parts.extend(f"{key}={_text(value)}" for key, value in detail.items())
    return "; ".join(parts)


class _RecordPayload(Protocol):
    @property
    def subject(self) -> tuple[IntEnum, int]: ...

    @property
    def detail(self) -> Mapping[str, object]: ...


def _text(value: object) -> str:
    r"""A short rendering: references name only their stated fields, a record its code, subject, payload and sources,
    and tables or reports only their type."""

    if isinstance(value, (SourceRef, StructureRef, RequestRef)):
        stated = (f"{f.name}={getattr(value, f.name)!r}" for f in fields(value) if getattr(value, f.name) is not None)
        return f"{type(value).__name__}({', '.join(stated)})"
    if hasattr(value, "code") and hasattr(value, "sources"):  # a provenance.Record (provenance imports this module)
        payload = cast(_RecordPayload, value)
        kind, index = payload.subject
        parts = [f"subject=({kind.name.lower()}, {index})", *(f"{k}={_text(v)}" for k, v in payload.detail.items())]
        return f"{value.code}({', '.join(parts + [_text(ref) for ref in value.sources])})"
    if isinstance(value, (tuple, list, frozenset)):
        return "(" + ", ".join(_text(item) for item in value) + ")"
    if value is None or isinstance(value, (str, int, float, Enum)):
        return repr(value)
    return f"<{type(value).__name__}>"


CON, INS, UNS, REQ, RES = (
    ContradictionError,
    InsufficientInformationError,
    UnsupportedError,
    InvalidRequestError,
    ResourceLimitError,
)
C, U, R, F = RecordKind.CONFLICT, RecordKind.UNKNOWN, RecordKind.UNREALISED, RecordKind.FINDING
INSUFFICIENT, AMBIGUOUS, UNSUPPORTED = Cause.INSUFFICIENT, Cause.AMBIGUOUS, Cause.UNSUPPORTED
TEXT, INT, ATOM, DECLARATION, CODE = (
    PayloadKind.TEXT,
    PayloadKind.INT,
    PayloadKind.ATOM,
    PayloadKind.DECLARATION,
    PayloadKind.CODE,
)


def _raise(names: str, error: type[OFoldXError] | tuple[type[OFoldXError], ...], basis: str, witness=None):
    r"""Raise-only codes of one table row; ``error`` is one class or one class per name."""
    split = names.split()
    errors = error if isinstance(error, tuple) else (error,) * len(split)
    return [Code(n, e, Disposition.RAISE, None, Cause.NONE, basis, witness) for n, e in zip(split, errors, strict=True)]


def _record(names: str, kind: RecordKind, basis: str, witness=None, *, cause=Cause.NONE, payload=None, lists=None):
    r"""Record codes of one table row; ``payload`` maps a name of the row to its typed slots, ``lists`` to its list
    field."""
    payload, lists = payload or {}, lists or {}
    return [
        Code(n, None, Disposition.RECORD, kind, cause, basis, witness, payload.get(n, ()), lists.get(n))
        for n in names.split()
    ]


_TABLE = [
    # Lexical and format
    *_raise(
        "cif_syntax cif_loop_count cif_duplicate_item non_ascii_source",
        CON,
        "CIF 1.1 grammar (gemmi's syntax layer reports; the adapter maps)",
    ),
    *_raise("cif_frame_unsupported", UNS, "a CIF save frame: valid CIF syntax that no mmCIF category reads"),
    *_raise(
        "cif_form_unsupported",
        UNS,
        "a global_ block, a tag not _category.item, or a loop over several categories: valid CIF, not mmCIF",
    ),
    *_raise("number_grammar", CON, "CIF numeric grammar with su"),
    *_raise("item_enumeration", CON, "a stated token outside the dictionary enumeration of its item (codec row)"),
    *_raise(
        "private_row_invalid",
        CON,
        "an _ofoldx_* row (only OFoldX writes them) naming no written token, atom, instance, configuration or "
        "library",
    ),
    # sources: the written cell and its _ofoldx_fact_basis row
    *_record("conflict_restored", C, "an _ofoldx_fact_basis row restores a written Conflict"),
    *_record(
        "connections_open_restored",
        R,
        "an _ofoldx_instance row restores an instance whose connections the value does not all represent",
    ),
    *_record(
        "written_library_differs",
        F,
        "the _ofoldx_library row names a CCD artifact of other content than the one in use: what the file leaves to "
        "the library (reference facts, a file-scope definition's arbitration) is derived from the one in use",
        payload={
            "written_library_differs": (
                ("written_version", TEXT),
                ("written_sha256", TEXT),
                ("content_version", TEXT),
                ("sha256", TEXT),
            )
        },
    ),
    *_raise("numeric_range", UNS, "value outside the column dtype (our limit)"),
    # the candidates are the block names; `candidates` holds references (see InsufficientInformationError)
    *_raise("multiple_data_blocks", INS, "> 1 block, none requested (candidates = block names)"),
    *_raise("format_unknown", REQ, "not recognisable as mmCIF, PDB, A3M or aligned FASTA"),
    *_raise("format_unsupported", UNS, "recognised, unsupported (BinaryCIF)"),
    *_raise("pdb_column pdb_hybrid36_invalid", CON, "PDB v3.3 fixed columns; hybrid-36"),
    *_record(
        "pdb_line_unrecognised",
        F,
        "a non-blank line whose columns 1–6 name no PDB v3.3 record is not read (a cut file, a shifted or foreign "
        "line): one record per file, referencing its first three such lines",
        payload={"pdb_line_unrecognised": (("count", INT),)},
    ),
    *_raise("element_missing element_unknown", CON, "_atom_site.type_symbol / PDB cols 77–78"),
    # Atom site, declarations, identity
    *_raise("model_number_null", CON, "explicit null pdbx_PDB_model_num"),
    *_raise(
        "molecule_absent",
        INS,
        "the source states no atom row and declares no instance (a bare data block, a component definition): no "
        "structure to read",
    ),
    *_raise(
        "model_unknown assembly_unknown altloc_policy_invalid block_unknown",
        REQ,
        "read option names nothing in the source",
    ),
    *_raise("coordinate_axis_partial", CON, "Cartn_x/y/z states disagree"),
    *_raise(
        "occupancy_out_of_range",
        CON,
        "_atom_site.occupancy outside −3u ≤ occ ≤ 1 + 3u (IUCr coreCIF _atom_site_occupancy), u its stated su, 0 when "
        "Unknown; payload value, uncertainty (None: Unknown)",
        "1btw, 1ecp",
    ),
    *_raise(
        "occupancy_uncertainty_conflict",
        CON,
        "the occupancy's parenthesised su and _atom_site.occupancy_esd both stated and different",
    ),
    *_raise(
        "occupancy_uncertainty_negative",
        CON,
        "_atom_site.occupancy_esd below zero (a standard uncertainty is not negative), whatever the occupancy's state; "
        "payload value",
    ),
    *_raise(
        "label_auth_conflict duplicate_atom residue_unbound asym_two_chains asym_undeclared asym_entity_mismatch "
        "scheme_mismatch entity_type_enumeration key_invalid",
        CON,
        "dictionary keys and cross-category agreement in the selected model; key_invalid: a category key stated "
        "twice or not stated, or a PDB record stated once per entry (CRYST1, EXPDTA) restated with another value",
        "2m9a, 7v3g",
    ),
    *_raise(
        "nonpolymer_address_ambiguous",
        INS,
        "the rows of a non-polymer instance state no one address: a water row without an author number, or rows of "
        "one molecule with different numbers, insertion codes or components",
    ),
    *_raise(
        "modified_residue_ambiguous",
        INS,
        "a _pdbx_struct_mod_residue / MODRES row's address (label when stated, else author) names several occurrences "
        "(candidates)",
    ),
    *_record(
        "modified_residue_unbound",
        R,
        "a _pdbx_struct_mod_residue / MODRES row stating a parent whose address (label when stated, else author) "
        "names no occurrence: its annotation is declared but unrealised",
        "5n7g, 1ct5 (mutated)",
    ),
    *_record(
        "seqres_correspondence_ambiguous",
        U,
        "PDB SEQRES embedding not unique; publishes no determined endpoint for the pairs whose adjacency differs "
        "(formerly a raise)",
        "1a33 (1a33_seqres_ambiguous.pdb.gz)",
        cause=AMBIGUOUS,
        # the number of admissible embeddings
        payload={"seqres_correspondence_ambiguous": (("embeddings", INT),)},
    ),
    *_record(
        "polymer_type_unstated",
        U,
        "PDB chain whose SEQRES components do not derive one polymer type (replaces the raise "
        "polymer_type_ambiguous)",
        cause=INSUFFICIENT,
    ),
    *_record(
        "parent_annotation_mismatch",
        F,
        "an occurrence's _pdbx_struct_mod_residue.parent_comp_id / MODRES annotation (component_parent; its rows "
        "named) differs as a set from the parents its definition's _chem_comp.mon_nstd_parent_comp_id lists "
        "(component_definition_parent; the statements it was resolved from named): two scopes, no contradiction; a "
        "definition listing none states no set to differ from",
        "1b9c",
        # both parent sets, comma-separated as mon_nstd_parent_comp_id lists them
        payload={"parent_annotation_mismatch": (("annotation", TEXT), ("definition", TEXT))},
    ),
    *_raise(
        "altloc_unknown altloc_occupancy_unknown",
        INS,
        "an Unknown altloc at a position with lettered rows; the highest-occupancy rule meets an Unknown occupancy",
    ),
    *_raise(
        "microheterogeneity_unobserved",
        INS,
        "a sequence position declares several components and the model observes none (candidates = the declaring "
        "rows)",
    ),
    *_raise("author_altloc_unsupported", UNS, "pdbx_ptnr*_auth_alt_id stated; no author-altloc column"),
    *_record(
        "entry_level_violation scheme_absent group_vs_seqres seqres_absent resolution_invalid isotope_name_mapped "
        "author_component_differs author_atom_differs altloc_tie_by_label",
        F,
        "a violation in a model other than the selected one; an observed instance without scheme rows, its residues "
        "bound by atom keys; a PDB HETATM residue in SEQRES that no MODRES names; a PDB chain without SEQRES; an "
        "invalid resolution; an isotope atom name bound to its definition atom; author component or atom names that "
        "differ from the label ones; an alternative chosen by its label on an occupancy tie",
        "1ts6, bdbseq, 4jlj",
        # entry_level_violation: the model number of the violating atom row, the check it fails (the code that would
        # refuse it in the selected model) and its asym ("" where the row states none)
        payload={"entry_level_violation": (("model", INT), ("check", CODE), ("asym", TEXT))},
    ),
    *_record("instance_unobserved", R, "declared asym with no observed atom"),
    *_record("conn_type_enumeration", C, "_struct_conn.conn_type_id enumeration; the row yields no declaration"),
    # Endpoints
    *_raise(
        "endpoint_dangling endpoint_label_auth_conflict",
        CON,
        "a connection endpoint whose label or author address names no residue of the selected model, or whose label "
        "and author addresses name different residues or atoms",
        "2m9a, 3cye, 1jtw, 1jba",
    ),
    *_raise("endpoint_ambiguous", INS, "≥ 2 candidates"),
    *_record(
        "absent_in_model unselected_alternative",
        R,
        "a declared endpoint the selected model does not observe, or one in an alternative the selection did not "
        "choose: the declaration is kept, unrealised",
        "1zbl m2, 5wlj, 2grz",
    ),
    *_record(
        "unresolved_asym endpoint_atom_unbound",
        U,
        "an endpoint naming a declared asym that has no residues, or an unobserved atom no definition names",
        cause=INSUFFICIENT,
    ),
    *_record("endpoint_atom_not_in_reference", C, "the named atom is in neither the definition nor the observations"),
    *_record("contact_row", F, "hydrog/saltbr/mismat never an edge"),
    # Symmetry and worlds
    # side: the declaration side 0 | 1 a record about a symmetry code names (-1 for a record of the read's symmetry)
    *_record(
        "symmetry_code_grammar",
        C,
        "n_klm dictionary regex",
        "5m85",
        payload={"symmetry_code_grammar": (("side", INT), ("text", TEXT))},
    ),
    *_record(
        "symmetry_code_ambiguous",
        U,
        "several digit splits",
        cause=AMBIGUOUS,
        payload={"symmetry_code_ambiguous": (("side", INT), ("text", TEXT), ("reading", TEXT))},
    ),
    *_record(
        "symmetry_absent ssbond_symmetry_absent space_group_setting_unresolved frame_incomplete",
        U,
        "a connection side states no symmetry code (an SSBOND's blank symmetry included); the stated space group "
        "resolves no setting; frame_incomplete: a stated transform or cell in part, or a cell whose stated axis "
        "convention is not interpreted",
        "1i88",
        cause=INSUFFICIENT,
        # name: the H-M name as stated ("" none); number: Int_Tables_number (-1 none); on a declaration's record both
        # are the read's own record's ("", -1 when the read has none: an id absent from the table in force); code: the
        # unresolved code ("" for the read's own record)
        payload={
            "symmetry_absent": (("side", INT),),
            "ssbond_symmetry_absent": (("side", INT),),
            "space_group_setting_unresolved": (("side", INT), ("name", TEXT), ("number", INT), ("code", TEXT)),
        },
    ),
    *_record(
        "special_position_undecided",
        U,
        "stabiliser operation in the band between the inner and outer bounds, or failing the group property",
        cause=AMBIGUOUS,
        # 1 when the numerically decided members fail the group property, 0 for an operation in the band
        payload={"special_position_undecided": (("group_property_failed", INT),)},
    ),
    *_record(
        "special_position_unplaced",
        U,
        "an endpoint without a position (or frame) under an operation with fixed points",
        cause=INSUFFICIENT,
    ),
    *_record(
        "operation_table_vs_setting frame_conflict contact_implausible operator_matrix_deviates "
        "non_isometric_operator",
        C,
        "the file's operation table against the resolved setting (always a Conflict); the frame [A | u] against the "
        "cell's metric, against its stated inverse, or no lattice basis; a symmetry image farther from its partner "
        "than any contact; an operator matrix off its named operation, or no rotation, beyond what its printing "
        "accounts for",
        "5vjz, 1fhj, 6kpq, 1c50",
        # operations are x,y,z triplets, "" where that side states none (a setting operation the table lacks); a
        # frame_conflict names the quantity furthest beyond its rounding bound, a _cell item (the frame's metric
        # against the cell) or Cartn_transf (a unit-cell corner placed by the inverse), and both in thousandths of its
        # unit, Å or degree, the difference rounded up and the bound down ("", -1, -1: nothing measured, a stated
        # matrix or the cell itself describes no lattice basis); distance_mA in integer milliångström
        payload={
            "operation_table_vs_setting": (
                ("side", INT),
                ("file_operation", TEXT),
                ("setting_operation", TEXT),
                ("setting", TEXT),
            ),
            "frame_conflict": (("item", TEXT), ("difference", INT), ("bound", INT)),
            "contact_implausible": (("distance_mA", INT),),
        },
    ),
    *_record(
        "crystal_operation_inapplicable",
        C,
        "a declaration places its partner by a crystal operation other than the identity while the crystal frame is "
        "inapplicable (the format's placeholder cell, stated as the cell or with no cell through the transform, in a "
        "non-crystal experiment); the references are the declaration and what the frame rests on: the "
        "cell and/or the transform, and the methods",
        "4bog",
    ),
    # two statements of one symmetry fact contradict each other; operations are triplets
    *_record(
        "space_group_name_vs_number",
        C,
        "the H-M name fixes the setting; Int_Tables_number modulo 1000 only cross-checks it and names another group; "
        "the references are both items",
        payload={"space_group_name_vs_number": (("name", TEXT), ("number", INT), ("setting_number", INT))},
    ),
    *_record(
        "space_group_restatement_conflict",
        C,
        "_symmetry and _space_group state the space group differently: H-M names of two settings, or two "
        "Int_Tables numbers naming two groups (modulo 1000); the references are both items",
        payload={"space_group_restatement_conflict": (("value", TEXT), ("other_value", TEXT))},
    ),
    *_record(
        "operation_id_conflict",
        C,
        "one operation id names two different operations (the file's own ids); the references are both rows",
        payload={"operation_id_conflict": (("id", TEXT), ("operation", TEXT), ("other_operation", TEXT))},
    ),
    *_record(
        "operation_table_not_group",
        C,
        "the file's operation list, with no resolved setting, is not closed modulo the lattice; the references "
        "are its rows",
        # generated = the cosets the listed ones generate, -1 when they generate no space group
        payload={"operation_table_not_group": (("listed", INT), ("generated", INT))},
    ),
    *_record(
        "operation_table_unparsable operator_defect_recorded operator_2d3d_present",
        F,
        "an operation row whose operation is not parsed (its id's operation Unknown); an operator matrix that "
        "departs from a rotation within what its printing accounts for; a 2D or 3D crystal symmetry operator row",
        "1l6j, 3hq7, 1rb8",
        # the reference is the operator row; id its operation id ("" when it states none), text its operation token as
        # stated: the operation of the id is Unknown
        payload={"operation_table_unparsable": (("id", TEXT), ("text", TEXT))},
    ),
    *_raise(
        "frame_unknown operator_unresolved copy_identity_conflict operator_matrix_unknown",
        (INS, CON, CON, INS),
        "a request (an assembly) needs a placement that is unknown or contradicted (frame_unknown: a "
        "frame not KNOWN, an inapplicable one included)",
    ),
    *_raise(
        "operator_expression_grammar",
        CON,
        "_pdbx_struct_assembly_gen.oper_expression: a comma list of operator ids and m-n ranges, or a product of "
        "parenthesised lists (PDBx)",
    ),
    *_raise(
        "operator_2d3d improper_copy copy_symbol_collision",
        UNS,
        "an assembly applies a 2D or 3D crystal symmetry operator; mirror-image identities not represented; a "
        "generated copy symbol equals an existing asym id",
    ),
    # Definitions
    *_record(
        "reference_incomplete_absent",
        U,
        "the artifact's entry for the id states no atoms and no other scope supplies any",
        "CCD UNL (components.cif:8055515, no _chem_comp_atom rows)",
        cause=INSUFFICIENT,
    ),
    *_record(
        "reference_incomplete_not_in_library",
        U,
        "the configured artifact has no entry and no scope supplies atoms; payload id + content_version",
        "synthetic fixture (every corpus file embeds its definitions)",
        cause=UNSUPPORTED,
        payload={"reference_incomplete_not_in_library": (("id", TEXT), ("content_version", TEXT))},
    ),
    # accounted with its element_differs_from_reference Conflict, never twice: no cause of its own
    *_record(
        "reference_incomplete_unusable",
        U,
        "source contradiction in substance: an observed element contradicts the definition, which is therefore "
        "unusable for the occurrence",
    ),
    *_record(
        "edge_endpoint_unrealised",
        R,
        "a declared edge whose endpoint is a named port of a definition-less component with no row",
    ),
    *_raise("definition_set_mismatch", REQ, "supplied Definitions digests differ from component_definition_digest"),
    # Definition records: subject (NONE, -1), the component id in the first slot
    *_record(
        "definition_conflict element_differs_from_reference",
        C,
        "statements of one definition that cannot form one (an unkeyed atom, a bond naming an absent atom or itself, "
        "duplicate rows that disagree) or two same-scope definitions that differ: withheld; observed vs definition "
        "element (the dependent endpoint Unknowns are accounted under this "
        "conflict, never twice)",
        payload={"definition_conflict": (("id", TEXT),)},
    ),
    *_record(
        "file_scope_null_over_archive_value definition_atom_sets_differ reference_coordinates_contradicted "
        "leaving_flag_unusable two_fragment_port port_atom_absent port_unverified parity_refused "
        "chem_comp_type_unlisted",
        F,
        "a file-scope null over an archive value, or file-scope atoms other than the archive's (the file scope "
        "wins); artifact findings (a contradicted reference conformer, a refused parity, an unusable leaving flag); "
        "port verification; a _chem_comp.type outside the junction-family table and its no-rule types",
        "18 components; 7WC (next side)",
        # side as in TerminalCap ("previous" | "next"); condition a chemistry.fragments.PortCondition value; item the
        # dictionary item name; atom an atom id of the definition
        payload={
            "file_scope_null_over_archive_value": (("id", TEXT), ("item", TEXT)),
            "definition_atom_sets_differ": (("id", TEXT),),
            "reference_coordinates_contradicted": (("id", TEXT),),
            "leaving_flag_unusable": (("id", TEXT), ("atom", TEXT)),
            "two_fragment_port": (("id", TEXT), ("side", TEXT)),
            "port_atom_absent": (("id", TEXT), ("atom", TEXT)),
            "port_unverified": (("id", TEXT), ("side", TEXT), ("condition", INT)),
            "parity_refused": (("id", TEXT), ("atom", TEXT)),
            "chem_comp_type_unlisted": (("id", TEXT), ("type", TEXT)),
        },
    ),
    *_raise("artifact_schema_unsupported", UNS, "a CCD artifact of a schema this version does not read"),
    *_raise("ccd_artifact_unconfigured", REQ, "no CCD artifact configured (no silent fallback)"),
    *_raise("component_not_in_library", INS, "get_component miss"),
    # Attachments — Conflict reasons
    *_record(
        "endpoint_named_as_leaving declaration_not_terminal declaration_not_in_reference observed_leaving_atom "
        "placed_leaving_atom bonded_leaving_atom declaration_vs_declaration declaration_vs_reference "
        "declaration_vs_designation declaration_exceeds_order type_vs_value type_vs_element definition_vs_definition "
        "altloc_exclusive_endpoints self_bond",
        C,
        "a declared connection contradicts the endpoint's definition, its leaving atoms, another declaration or "
        "the stated type, element or order",
        "5xrf, 8cdo, 1nd5, 9jog, 1adz, OXT ×37",
        # carbonyl_o_observed on a linked carbon whose designated OXT is observed: 0 legacy carbonyl naming (O absent),
        # 1 a pentavalent carbon (O present)
        # the leaving-atom codes name the atom (an occurrence atom until lowered); their sources are the declarations
        # and the atom's atom_site row
        payload={
            "observed_leaving_atom": (
                ("atom", ATOM),
                ("declaration", DECLARATION),
                ("atom_site_row", INT),
                ("carbonyl_o_observed", INT),
            ),
            "placed_leaving_atom": (("atom", ATOM),),
            "bonded_leaving_atom": (("atom", ATOM),),
        },
    ),
    # one record per atom pair whose rows admit no reading, a CONFLICT transform among them (subject a row of it,
    # sources every row, the two atoms), the existence reason of its rows no KNOWN transform or fixed topology makes an
    # edge (not recorded again at their endpoints)
    *_record(
        "conflicting_connection",
        C,
        "several rows name one atom pair and one has a CONFLICT transform: no interpretation of them as physical edges "
        "is compatible with the source",
        lists={"conflicting_connection": ("atoms", ATOM)},
    ),
    # Attachments — Unknown reasons (never raised by a read)
    *_record(
        "order_unknown symmetry_unknown kind_unknown junction_unresolved_nstd_linkage_unstated "
        "junction_unresolved_nstd_linkage_yes junction_unresolved_no_family junction_unresolved_no_port",
        U,
        "insufficient source information",
        "1abz SIN 1 C4 – ASP 2 N, 5dgf",
        cause=INSUFFICIENT,
    ),
    *_record(
        "ambiguous_candidates internal_edit_unresolved junction_unresolved_sequence_ambiguous",
        U,
        "genuine ambiguity",
        "1hrc HEC, 6af9",
        cause=AMBIGUOUS,
        # the roots of the candidate fragments the inequivalent balancing sets draw from
        lists={"ambiguous_candidates": ("candidates", ATOM)},
    ),
    # one record per atom pair (subject a row of it, sources every row, the two atoms), the existence reason of its
    # rows no KNOWN transform or fixed topology makes an edge (not recorded again at their endpoints); covalent: 1 when
    # a row of it is covalent or of undetermined kind, 0 when every row is a stated coordination; a default read
    # escalates a covalent one (read_ambiguous_connections)
    *_record(
        "ambiguous_connection",
        U,
        "several rows name one atom pair and are not provably one physical edge: their source-compatible "
        "interpretations (one restated edge or distinct edges) give different graphs",
        cause=AMBIGUOUS,
        payload={"ambiguous_connection": (("covalent", INT),)},
        lists={"ambiguous_connection": ("atoms", ATOM)},
    ),
    *_record(
        "valence_model_inapplicable designation_exceeds_order charge_differs no_balancing_fragment_absent "
        "unsupported_polymer_type unsupported_coordination_substitution",
        U,
        "unsupported definition / model scope",
        "MET SD, 6n80",
        cause=UNSUPPORTED,
        # charge_differs names the charge-model applicability condition that fails: "class_b" | "aromatic" (an unknown
        # order or an incomplete neighbourhood at the endpoint fails earlier: reference_incomplete_absent)
        payload={"valence_model_inapplicable": (("open_valence", INT),), "charge_differs": (("condition", TEXT),)},
    ),
    # accounted with its placed_leaving_atom/bonded_leaving_atom Conflict record: no cause of its own
    *_record("no_balancing_fragment_blocked", U, "source contradiction in substance", "1adz (modelled HG)"),
    # accounted under assumption_application, never source_missing or a source conflict: no cause of its own
    *_record(
        "assumption_falsified",
        U,
        "an assumption rule the caller applied is contradicted by the source (recorded only when an assumption runs)",
        "1u19 covale4 under unstated_covalent_order_single",
        # the falsified assumption rule; the source rows are the record's references
        payload={"assumption_falsified": (("rule", TEXT),)},
    ),
    # Attachments — findings
    *_record(
        "edge_not_in_reference kind_order_mismatch kind_element_mismatch consecutive_pair_unlinked "
        "hydrogen_count_unknown hydrogen_count_observed_only implied_value_overridden leaving_flag_inconsistent "
        "order_on_coordination inapplicable_on_covalent conect_unclassified endpoint_unobserved "
        "phantom_heavy_neighbours observed_atom_not_in_reference nonstandard_linkage_declared nstd_linkage_unstated "
        "nstd_linkage_contradicted junction_realised_otherwise role_atoms_incompatible role_environment_unknown "
        "coordination_substitution_unresolved reference_charge_unverified_at_coordination disulf_selenium_endpoint "
        "parity_cleared_heavy_neighbour_removed "
        "parity_cleared_new_neighbour_without_vacated_slot parity_cleared_two_virtual_slots "
        "parity_cleared_nonport_link replacement_atom_absent "
        "replacement_element_differs replacement_endpoint_undetermined",
        F,
        "attachment facts recorded, not refused: how a connection or its endpoints depart from the definitions",
        "5a4f, 1e61, 3rqd, 3Q9H",
        # hydrogen_count_unknown cause: "reference_unresolved" | "endpoint_unknown" | "endpoint_conflict" |
        # "coordination_incompatible" | "unclassified_connection" | "hydrogen_parent_unknown" (a candidate host of an
        # observed hydrogen with no reference counterpart and no decided parent; source: the hydrogen's row);
        # consecutive_pair_unlinked: the entity id and the two sequence numbers
        payload={
            "hydrogen_count_unknown": (("cause", TEXT),),
            "consecutive_pair_unlinked": (("entity", TEXT), ("first", INT), ("second", INT)),
        },
        lists={
            "phantom_heavy_neighbours": ("atoms", ATOM),
            "coordination_substitution_unresolved": ("atoms", ATOM),
        },
    ),
    # Read escalation
    *_raise(
        "read_conflicts",
        CON,
        'on_conflict="raise" and ≥ 1 kind-C record; payload = every Conflict record + the report '
        "(build_system: the Conflict records of its definitions)",
        "~252 modelled-HG files",
    ),
    *_raise(
        "read_ambiguous_connections",
        INS,
        'on_conflict="raise" and ≥ 1 ambiguous_connection record whose rows can be covalent (its readings '
        "can differ in the covalent graph or a budget); refs = the atoms and source rows, candidates = the "
        "interpretations, payload = every ambiguous_connection record + the report",
    ),
    # Operations
    *_raise("hydrogen_view_loses_stereo", UNS, "the H fold would lose a stored configuration"),
    *_raise(
        "value_invalid edit_invalid projection_invalid stub_mismatch instance_collision scope_violation",
        REQ,
        "caller-built value or recipe inconsistent; select naming a chain absent from the namespace",
    ),
    *_raise("precision_loss", UNS, "integer range"),
    # Writers and native
    *_raise(
        "write_identity_unknown",
        UNS,
        'an identifier the format requires that the value does not know (unless identity="assign"): never invented; '
        "payload = those unencodable records",
    ),
    *_raise(
        "write_identifier_unfit",
        UNS,
        "a known identifier or number the format's columns cannot hold (a PDB chain, residue name, number or "
        "insertion code, atom name, altloc, serial, model serial, coordinate, charge): never cut; "
        "payload = the item, its value, the limit, the records",
    ),
    # records of WriteReport
    *_record(
        "unencodable identity_assigned",
        F,
        "a fact the format cannot carry, written around (family, item; one record per instance for its atoms or "
        "components, count: how many; the atoms an instance-level record names); an "
        'identifier identity="assign" assigned',
        payload={
            "unencodable": (("family", TEXT), ("item", TEXT), ("count", INT)),
            "identity_assigned": (("item", TEXT), ("value", TEXT)),
        },
        lists={"unencodable": ("atoms", ATOM)},
    ),
    *_raise("native_schema_version", REQ, "schema digest mismatch"),
    *_raise("native_corrupt", CON, "header and payload disagree, or the content digest does not match the bytes"),
    *_raise(
        "dataset_not_native",
        REQ,
        "a training dataset reads native structure artifacts (.safetensors) only and never parses mmCIF or PDB; "
        "read and save those beforehand (read_structure, save_structure)",
    ),
    # MSA, templates, SMILES
    *_raise("msa_syntax msa_length_mismatch msa_query_mismatch", CON, "A3M/FASTA grammar"),
    *_raise(
        "sequence_name_duplicate",
        CON,
        "two FASTA sequence records name one instance (an id stated twice, or an id equal to a generated chain_N)",
    ),
    *_raise("msa_index_ambiguous", INS, "an MSA index value names more than one directory"),
    *_raise(
        "msa_entity_unresolved",
        INS,
        "a text MSA path binds to no entity of the target or to several (candidates = one instance per entity)",
    ),
    *_raise(
        "msa_plan_with_crop msa_plan_too_deep msa_entity_unknown template_target_unknown",
        REQ,
        "an exact MSA plan given with a training crop; an exact plan with more rows than the family's depth constant; "
        "an MSA naming an entity the target does not have; "
        "a template alignment naming no instance of its entity or a position past the instance",
    ),
    *_raise(
        "query_class_unrepresentable template_target_unsupported template_frame_incompatible",
        UNS,
        "a class with no sequence letter (OTHER, specials); a template target type the family cannot featurise; "
        "incompatible placement frames",
    ),
    *_raise(
        "smiles_syntax cx_conflicting_declaration invalid_stereo_class", CON, "OpenSMILES / ChemAxon CX", r"C/C(\F)=C/F"
    ),
    *_raise("smiles_chemistry", CON, "the string admits no valence or Kekulé structure (RDKit sanitization)"),
    *_raise(
        "cx_unsupported_declaration unsupported_stereo_class",
        UNS,
        "a CX extension a definition cannot represent, or a non-tetrahedral stereo class (@AL, @SP, @TB, @OH)",
        "TLB",
    ),
    *_raise("smiles_unsupported", UNS, "a dummy atom or a bond type outside single–quadruple and aromatic"),
    *_raise(
        "smiles_atom_mapping_unsupported",
        UNS,
        "an atom-mapping annotation (an atom-map number, as RDKit parses it): a definition neither keeps nor uses "
        "atom mapping, so the string is refused rather than read without it; payload the string atoms and their map "
        "numbers",
        "[CH4:1]",
    ),
    *_raise(
        "unsupported_radical",
        UNS,
        "radical electrons a CX ^n: key states for an atom, or RDKit assigns to a non-metal atom "
        "(GetNumRadicalElectrons > 0): a definition represents none; RDKit's computed count on a metal atom is not "
        "consulted, nor claimed zero",
        "[CH3]",
    ),
    *_raise("smiles_excluded", UNS, "the ChEMBL structure pipeline excludes the molecule (its exclusion flag)"),
    # the component bond row whose double bond carries one directional mark
    *_record(
        "cx_incomplete_declaration",
        F,
        "a double bond with a directional mark on one side only declares no configuration",
        payload={"cx_incomplete_declaration": (("bond", INT),)},
    ),
    # Chemistry views and models
    *_raise("rdkit_incomplete", INS, "projection needs an UNKNOWN fact"),
    *_raise(
        "rdkit_unsupported model_capacity model_numeric design_breaks_connection",
        UNS,
        "our representation or the architecture's vocabulary",
        "MSE in atom14",
    ),
    *_raise(
        "checkpoint_invalid",
        CON,
        "a checkpoint contradicts the released layout its converter maps (an unmapped or colliding key, a malformed "
        "record, a buffer or configuration disagreeing with the architecture)",
    ),
    *_raise(
        "oops_operator_unavailable",
        REQ,
        'backend="oops" where the installed OOps package does not provide the operator called, or OOps is not '
        "installed: the explicit request never runs another backend instead",
    ),
    *_raise("embedding_failed", RES, "the conformer search ended without a conformer"),
    *_raise("symmetry_search_limit", RES, "more automorphisms than the explicit search limit"),
    *_raise(
        "length_search_limit",
        RES,
        "the segment-length rejection search ended without a total inside the requested bounds (RFdiffusion)",
    ),
    *_record(
        "symmetry_unknown_incomplete",
        U,
        "a component whose topology, atoms or hydrogens are not complete has no determined automorphisms",
        cause=INSUFFICIENT,
    ),
    *_raise(
        "prepare_undetermined_slot prepare_unknown_kind prepare_incomplete_topology prepare_unknown_feature "
        "prepare_reference_missing template_chain_ambiguous",
        INS,
        "the consumed atoms lack a fact or reference conformer the family needs; template projection",
    ),
    *_raise(
        "affinity_ligand_unobserved",
        INS,
        "the affinity crop is centred on the ligand's observed token centres (upstream Boltz-2 AffinityCropper)",
    ),
    *_raise(
        "alignment_unresolved",
        INS,
        "no rigid transform under which the compared chains all pair (PXDesign's chain-permuted CA RMSD)",
    ),
    *_raise(
        "prepare_valence_excess prepare_source_conflict",
        CON,
        "a KNOWN positive valence excess at a consumed atom (an Unknown one is not refused); a CONFLICT endpoint at "
        "a consumed atom or at an atom bonded to one",
        "a constructed value (no read carries a KNOWN positive excess); 5xrf",
    ),
    *_raise(
        "request_invalid_positions request_invalid_sequence request_invalid_generator request_invalid_placement "
        "design_token_unmappable",
        REQ,
        "one position validator; one-letter construction; a crop without a generator; a generation placement naming "
        "nothing, or a length outside its segments' budget; a sampled class with no single component",
    ),
    *_record(
        "encoding_loss",
        F,
        "a position whose canonical class does not encode its identity exactly; per position",
        "MSE → M",
    ),
    *_record(
        "reference_embedded",
        F,
        "a component without a usable reference conformer was embedded from its determined graph",
        # the seed of the embedding
        payload={"reference_embedded": (("seed", INT),)},
    ),
    *_record(
        "design_reference_substituted",
        F,
        "a generated component's reference conformer is another component's definition conformer",
        "a PXDesign UNK residue on GLY's backbone",
    ),
    *_record(
        "prepare_accepted_incomplete",
        F,
        "a component's topology is incomplete only at atoms the prepared task does not consume",
        "1u19 under ProteinMPNN",
        # the code that makes the component's topology incomplete
        payload={"prepare_accepted_incomplete": (("cause", CODE),)},
    ),
]

CODES: Mapping[str, Code] = MappingProxyType({code.name: code for code in sorted(_TABLE, key=lambda c: c.name)})
assert len(CODES) == len(_TABLE), "a code is named twice"
CODE_NAMES: tuple[str, ...] = tuple(CODES)
del CON, INS, UNS, REQ, RES, C, U, R, F, INSUFFICIENT, AMBIGUOUS, UNSUPPORTED, _TABLE
del TEXT, INT, ATOM, DECLARATION, CODE


def fail(code: str, *refs: _Ref, **detail: object) -> NoReturn:
    r"""
    Raise the failure of a raise code: the one raise of a typed failure in OFoldX.

    Args:
        code: A key of ``CODES`` whose disposition is ``RAISE``.
        *refs: The statements or parts the condition is detected at.
        **detail: The typed payload of the code (``candidates`` for an ``InsufficientInformationError``).

    Raises:
        OFoldXError: The code's class, carrying ``code``, ``refs`` and ``detail``.
    """

    row = CODES[code]
    assert row.disposition is Disposition.RAISE, f"{code} is recorded, never raised"
    raise row.error(code, refs, **detail)  # type: ignore[misc]
