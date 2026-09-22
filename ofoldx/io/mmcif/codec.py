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
One codec row per mmCIF item OFoldX reads: the item, the table column it fills, how its tokens
decode, how each lexical state reads as a fact state and back, and its dictionary default.

``read`` maps a ``TokenState`` to a state byte: a value is KNOWN×DECLARED, ``?`` UNKNOWN×DECLARED, ``.``
INAPPLICABLE×DECLARED, an absent item KNOWN×DEFAULT with ``default`` when the codec row carries one, else UNKNOWN×NONE.
``write`` is its inverse. A ``default`` is the ``_item_default.value`` of the item's page in ``mmcif_pdbx_v50``, entered
only after that page was read (dictionary version 5.417, read 2026-10-19): ``_atom_site.occupancy``
1.0, ``_atom_site.pdbx_formal_charge`` 0, ``_chem_comp_atom.charge`` 0, ``value_order`` ``sing`` of
``_struct_conn``, ``_chem_comp_bond`` and ``_pdbx_entity_branch_link``, ``_entity_poly_seq.hetero`` ``no``. Items whose
page states no default (``_entity_poly.nstd_linkage``, ``_struct_conn.ptnr*_symmetry``) have none.

Readings encoded as rows (each reversible here): ``*_ins_code`` ``?``/``.``/absent → no insertion code;
``ptnr*_symmetry`` ``.`` → identity; ``pdbx_PDB_model_num`` absent → one model (a stated null is refused by the
decoder). ``enumeration`` lists the dictionary values (matched case-insensitively) with the integer each
reads as.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from ...chemistry.declarations import OrderToken
from ...provenance import Basis, Resolution, state
from ..cif import TokenState

_ItemKind = Literal["float", "int", "code", "text", "enum"]
_OutputToken = Literal["value", "?", ".", "absent"]

_KNOWN = state(Resolution.KNOWN, Basis.DECLARED)
_UNKNOWN = state(Resolution.UNKNOWN, Basis.DECLARED)
_INAPPLICABLE = state(Resolution.INAPPLICABLE, Basis.DECLARED)
_ABSENT = state(Resolution.UNKNOWN, Basis.NONE)
_DEFAULT = state(Resolution.KNOWN, Basis.DEFAULT)
_DEFINED = state(Resolution.KNOWN, Basis.DEFINED)
_NO_VALUE = state(Resolution.INAPPLICABLE, Basis.DEFINED)


@dataclass(frozen=True)
class ItemCodec:
    item: str
    column: str
    kind: _ItemKind
    read: Mapping[TokenState, int]
    default: str | None
    write: Mapping[int, _OutputToken]
    required: bool
    enumeration: Mapping[str, int] | None = None


@dataclass(frozen=True)
class CategoryCodec:
    r"""``key`` the category's key items; ``rows`` the ``SourceTables`` field its rows decode into."""

    category: str
    key: tuple[str, ...]
    items: tuple[ItemCodec, ...]
    rows: str


_STANDARD = {TokenState.VALUE: _KNOWN, TokenState.UNKNOWN: _UNKNOWN, TokenState.INAPPLICABLE: _INAPPLICABLE}
_INSERTION = {**_STANDARD, TokenState.UNKNOWN: _NO_VALUE, TokenState.ABSENT: _NO_VALUE}
_SYMMETRY = {**_STANDARD, TokenState.INAPPLICABLE: _DEFINED, TokenState.ABSENT: _ABSENT}
_MODEL = {**_STANDARD, TokenState.ABSENT: _DEFINED}
_TOKENS: dict[TokenState, _OutputToken] = {
    TokenState.VALUE: "value",
    TokenState.UNKNOWN: "?",
    TokenState.INAPPLICABLE: ".",
    TokenState.ABSENT: "absent",
}

_ENTITY_TYPES = {"polymer": 1, "non-polymer": 2, "branched": 3, "macrolide": 4, "water": 5}  # data.molecular.EntityType
_CONN_ORDERS = {"sing": 1, "doub": 2, "trip": 3, "quad": 4}  # _struct_conn.pdbx_value_order
# _pdbx_entity_branch_link.value_order also lists tokens that state no integer order
_ORDERS = _CONN_ORDERS | {token.name.lower(): int(token) for token in OrderToken if token > OrderToken.QUAD}
_YES_NO = {"no": 1, "n": 1, "yes": 2, "y": 2}  # data.molecular.NstdLinkage; hetero reads yes as 2
CONN_TYPES = [
    "covale",
    "disulf",
    "hydrog",
    "metalc",
    "mismat",
    "saltbr",
    "modres",
    "covale_base",
    "covale_sugar",
    "covale_phosphate",
]


def _item(
    item: str,
    column: str,
    kind: _ItemKind,
    *,
    default: str | None = None,
    required: bool = False,
    enumeration: Mapping[str, int] | None = None,
    read: Mapping[TokenState, int] = _STANDARD,
) -> ItemCodec:
    states = {TokenState.ABSENT: _DEFAULT if default is not None else _ABSENT, **read}
    write: dict[int, _OutputToken] = {}
    for token in TokenState:  # the first token reading as a state writes it back
        write.setdefault(states[token], _TOKENS[token])
    enumeration = None if enumeration is None else MappingProxyType(dict(enumeration))
    return ItemCodec(
        item, column, kind, MappingProxyType(states), default, MappingProxyType(write), required, enumeration
    )


def _codes(*pairs: str) -> tuple[ItemCodec, ...]:
    r"""Code items from ``"item:column"`` pairs (the column is the item when no colon is given)."""
    return tuple(
        _item(item, column, "code")
        for item, column in (pair.split(":") if ":" in pair else (pair, pair) for pair in pairs)
    )


def _partners(template: str, column: str, kind: _ItemKind, **options) -> tuple[ItemCodec, ...]:
    return tuple(_item(template.format(k), f"{column}_{k}", kind, **options) for k in (1, 2))


def _matrix(prefix: str, column: str) -> tuple[ItemCodec, ...]:
    r"""The 3×4 items ``{prefix}_matrix[i][j]`` and ``{prefix}_vector[i]`` (``prefix`` may be empty) as row-major
    columns ``{column}_{i}{j}`` (j = 4 the vector)."""
    joint = f"{prefix}_" if prefix else ""
    items = []
    for i in (1, 2, 3):
        items += [_item(f"{joint}matrix[{i}][{j}]", f"{column}_{i}{j}", "float") for j in (1, 2, 3)]
        items.append(_item(f"{joint}vector[{i}]", f"{column}_{i}4", "float"))
    return tuple(items)


def _scheme(*extra: ItemCodec) -> tuple[ItemCodec, ...]:
    return (
        *_codes("asym_id:asym", "entity_id:entity", "mon_id", "pdb_seq_num:pdb_seq"),
        _item("pdb_ins_code", "pdb_ins", "code", read=_INSERTION),
        *extra,
    )


_CATEGORIES = (
    CategoryCodec(
        "atom_site",
        ("id",),
        (
            _item("group_PDB", "group", "enum", enumeration={"atom": 1, "hetatm": 2}),
            _item("id", "id", "text"),
            _item("type_symbol", "element", "text", required=True),
            *_codes("label_atom_id:label_atom", "label_alt_id:label_alt", "label_comp_id:label_comp"),
            *_codes("label_asym_id:label_asym", "label_entity_id:label_entity"),
            _item("label_seq_id", "label_seq", "int"),
            _item("pdbx_PDB_ins_code", "insertion", "code", read=_INSERTION),
            *(_item(f"Cartn_{axis}", axis, "float") for axis in "xyz"),
            _item("occupancy", "occupancy", "float", default="1.0"),
            _item("occupancy_esd", "occupancy_esd", "float"),
            _item("B_iso_or_equiv", "b_factor", "float"),
            _item("pdbx_formal_charge", "charge", "int", default="0"),
            *_codes("auth_seq_id:auth_seq", "auth_comp_id:auth_comp", "auth_asym_id:auth_asym"),
            *_codes("auth_atom_id:auth_atom"),
            _item("pdbx_PDB_model_num", "model", "int", read=_MODEL),
        ),
        "atoms",
    ),
    CategoryCodec(
        "entity",
        ("id",),
        (
            _item("id", "id", "code", required=True),
            _item("type", "type", "enum", enumeration=_ENTITY_TYPES),
            _item("pdbx_description", "description", "code"),
        ),
        "entities",
    ),
    CategoryCodec(
        "entity_poly",
        ("entity_id",),
        (
            _item("entity_id", "entity", "code", required=True),
            _item("type", "type", "code"),
            _item("nstd_linkage", "nstd_linkage", "enum", enumeration=_YES_NO),
            _item("pdbx_seq_one_letter_code_can", "sequence", "text"),
        ),
        "polymers",
    ),
    CategoryCodec(
        "entity_poly_seq",
        ("entity_id", "num", "mon_id"),
        (
            _item("entity_id", "entity", "code", required=True),
            _item("num", "num", "int", required=True),
            _item("mon_id", "mon_id", "code", required=True),
            _item("hetero", "hetero", "enum", default="no", enumeration=_YES_NO),
        ),
        "sequences",
    ),
    CategoryCodec(
        "struct_asym",
        ("id",),
        (_item("id", "id", "code", required=True), _item("entity_id", "entity", "code")),
        "asyms",
    ),
    CategoryCodec(
        "pdbx_poly_seq_scheme",
        ("asym_id", "entity_id", "seq_id", "mon_id"),
        _scheme(_item("seq_id", "number", "int"), _item("pdb_strand_id", "pdb_strand", "code")),
        "schemes",
    ),
    CategoryCodec(
        "pdbx_nonpoly_scheme",
        ("asym_id", "ndb_seq_num"),
        _scheme(_item("pdb_strand_id", "pdb_strand", "code")),
        "schemes",
    ),
    CategoryCodec(
        "pdbx_branch_scheme",
        ("asym_id", "entity_id", "num", "mon_id"),
        _scheme(_item("num", "number", "int"), _item("pdb_asym_id", "pdb_strand", "code")),
        "schemes",
    ),
    CategoryCodec(
        "pdbx_entity_branch_link",
        ("link_id",),
        (
            _item("link_id", "link_id", "int"),
            _item("entity_id", "entity", "code"),
            *_partners("entity_branch_list_num_{}", "num", "int"),
            *_partners("comp_id_{}", "comp", "code"),
            *_partners("atom_id_{}", "atom", "code"),
            *_partners("leaving_atom_id_{}", "leaving", "code"),
            _item("value_order", "order", "enum", default="sing", enumeration=_ORDERS),
        ),
        "branch_links",
    ),
    CategoryCodec(
        "struct_conn",
        ("id",),
        (
            _item("id", "id", "code"),
            _item("conn_type_id", "conn_type", "code", enumeration={name: i for i, name in enumerate(CONN_TYPES)}),
            *_partners("ptnr{}_label_asym_id", "label_asym", "code"),
            *_partners("ptnr{}_label_seq_id", "label_seq", "int"),
            *_partners("ptnr{}_label_comp_id", "label_comp", "code"),
            *_partners("ptnr{}_label_atom_id", "label_atom", "code"),
            *_partners("pdbx_ptnr{}_label_alt_id", "label_alt", "code"),
            *_partners("ptnr{}_auth_asym_id", "auth_asym", "code"),
            *_partners("ptnr{}_auth_seq_id", "auth_seq", "code"),
            *_partners("pdbx_ptnr{}_PDB_ins_code", "auth_ins", "code", read=_INSERTION),
            *_partners("ptnr{}_auth_comp_id", "auth_comp", "code"),
            *_partners("ptnr{}_auth_atom_id", "auth_atom", "code"),
            *_partners("pdbx_ptnr{}_auth_alt_id", "auth_alt", "code"),
            *_partners("ptnr{}_symmetry", "symmetry", "code", read=_SYMMETRY),
            _item("pdbx_value_order", "order", "enum", default="sing", enumeration=_CONN_ORDERS),
            *_partners("pdbx_ptnr{}_leaving_atom_id", "leaving", "code"),
            _item("pdbx_leaving_atom_flag", "leaving_flag", "enum", enumeration={"none": 1, "one": 2, "both": 3}),
            *_partners("ptnr{}_role", "role", "code"),
            _item("pdbx_role", "pdbx_role", "code"),
        ),
        "connections",
    ),
    CategoryCodec(
        "pdbx_struct_mod_residue",
        ("id",),
        (
            *_codes("label_asym_id:label_asym"),
            _item("label_seq_id", "label_seq", "int"),
            *_codes("auth_asym_id:asym", "auth_seq_id:seq", "auth_comp_id:comp", "parent_comp_id:parent"),
            _item("PDB_ins_code", "ins", "code", read=_INSERTION),
            *_codes("label_comp_id:label_comp"),
        ),
        "modified",
    ),
    CategoryCodec(
        "cell",
        ("entry_id",),
        tuple(
            _item(item, item, "float")
            for item in ("length_a", "length_b", "length_c", "angle_alpha", "angle_beta", "angle_gamma")
        ),
        "crystal",
    ),
    CategoryCodec(
        "symmetry",
        ("entry_id",),
        (_item("space_group_name_H-M", "space_group", "code"), _item("Int_Tables_number", "number", "int")),
        "crystal",
    ),
    CategoryCodec(
        "space_group",
        ("id",),
        (_item("name_H-M_alt", "space_group", "code"), _item("IT_number", "number", "int")),
        "crystal",
    ),
    CategoryCodec(
        "space_group_symop",
        ("id",),
        (_item("id", "operation_id", "code"), _item("operation_xyz", "operation_text", "code")),
        "crystal",
    ),
    CategoryCodec(
        "symmetry_equiv",
        ("id",),
        (_item("id", "operation_id", "code"), _item("pos_as_xyz", "operation_text", "code")),
        "crystal",
    ),
    CategoryCodec(
        "atom_sites",
        ("entry_id",),
        (
            *_matrix("fract_transf", "fract"),
            *_matrix("Cartn_transf", "cartn"),
            _item("Cartn_transform_axes", "axes", "code"),
        ),
        "crystal",
    ),
    CategoryCodec(
        "pdbx_struct_oper_list",
        ("id",),
        (_item("id", "id", "code", required=True), *_codes("type", "name"), *_matrix("", "matrix")),
        "operators",
    ),
    CategoryCodec("pdbx_struct_assembly", ("id",), _codes("id"), "assemblies"),
    CategoryCodec(
        "pdbx_struct_assembly_gen",
        ("assembly_id", "oper_expression", "asym_id_list"),
        (
            _item("assembly_id", "assembly", "code", required=True),
            _item("oper_expression", "expression", "code", required=True),
            _item("asym_id_list", "asyms", "code", required=True),
        ),
        "assemblies",
    ),
    CategoryCodec("exptl", ("entry_id", "method"), _codes("method"), "experiment"),
    CategoryCodec(
        "refine",
        ("pdbx_refine_id",),
        (_item("pdbx_refine_id", "method", "code"), _item("ls_d_res_high", "resolution", "float")),
        "experiment",
    ),
    CategoryCodec("reflns", ("pdbx_ordinal",), (_item("d_resolution_high", "resolution", "float"),), "experiment"),
    CategoryCodec("em_3d_reconstruction", ("id",), (_item("resolution", "resolution", "float"),), "experiment"),
    CategoryCodec(
        "chem_comp",
        ("id",),
        tuple(
            _item(item, item, "text")
            for item in ("id", "type", "name", "one_letter_code", "three_letter_code", "mon_nstd_parent_comp_id")
        ),
        "definitions",
    ),
    CategoryCodec(
        "chem_comp_atom",
        ("comp_id", "atom_id"),
        (
            _item("comp_id", "comp_id", "text", required=True),
            *(
                _item(item, item, "text")
                for item in (
                    "atom_id",
                    "alt_atom_id",
                    "type_symbol",
                    "pdbx_aromatic_flag",
                    "pdbx_leaving_atom_flag",
                    "pdbx_stereo_config",
                    "pdbx_n_terminal_atom_flag",
                    "pdbx_c_terminal_atom_flag",
                )
            ),
            _item("charge", "charge", "int", default="0"),
            *(
                _item(template.format(axis), template.format(axis), "float")
                for template in ("model_Cartn_{}", "pdbx_model_Cartn_{}_ideal")
                for axis in "xyz"
            ),
        ),
        "definitions",
    ),
    CategoryCodec(
        "chem_comp_bond",
        ("comp_id", "atom_id_1", "atom_id_2"),
        (
            _item("comp_id", "comp_id", "text", required=True),
            *(
                _item(item, item, "text")
                for item in ("atom_id_1", "atom_id_2", "pdbx_aromatic_flag", "pdbx_stereo_config")
            ),
            _item("value_order", "value_order", "text", default="sing"),
        ),
        "definitions",
    ),
    CategoryCodec(
        "pdbx_chem_comp_descriptor",
        ("comp_id", "type", "program", "program_version"),
        (
            _item("comp_id", "comp_id", "text", required=True),
            *(_item(item, item, "text") for item in ("type", "program", "descriptor")),
        ),
        "definitions",
    ),
    CategoryCodec("entry", ("id",), (_item("id", "id", "text"),), "metadata"),
    CategoryCodec(
        "pdbx_database_status",
        ("entry_id",),
        (
            _item("recvd_initial_deposition_date", "deposition_date", "text"),
            _item("process_site", "process_site", "text"),
        ),
        "metadata",
    ),
    CategoryCodec(
        "pdbx_audit_revision_history",
        ("ordinal",),
        (_item("ordinal", "ordinal", "int"), _item("revision_date", "revision_date", "text")),
        "metadata",
    ),
)


def _site(prefix: str) -> tuple[ItemCodec, ...]:
    r"""One atom named natively: instance symbol, component ordinal in the instance, atom name (``.`` the implicit
    slot of a stereocentre)."""
    return _codes(f"{prefix}asym_id", f"{prefix}atom_id") + (_item(f"{prefix}component", f"{prefix}component", "int"),)


# Private categories: facts with no dictionary item that a reader cannot re-derive. Each row restores exactly
# the state it carries. `rows` names no SourceTables field: the read driver reads them (io.read).
_PRIVATE = (
    CategoryCodec(
        "ofoldx_fact_basis",
        ("category", "row", "item"),
        (
            _item("category", "category", "code", required=True),
            _item("row", "row", "int", required=True),
            _item("item", "item", "code", required=True),
            _item("state", "state", "int", required=True),
        ),
        "private",
    ),
    CategoryCodec(
        "ofoldx_coordinate_evidence",
        ("atom_site_id",),
        (
            _item("atom_site_id", "atom_site_id", "int", required=True),
            _item("evidence", "evidence", "enum", enumeration={"predicted": 2, "generated": 3, "reference": 4}),
        ),
        "private",
    ),
    CategoryCodec(
        "ofoldx_instance",
        ("asym_id",),
        (
            *_codes("asym_id", "copy_kind", "source_asym"),
            _item("operators", "operators", "text"),
            _item("connections", "connections", "enum", enumeration={"complete": 1, "open": 2}),
        ),
        "private",
    ),
    CategoryCodec(
        "ofoldx_atom_parity",
        ("asym_id", "component", "atom_id"),
        (
            *_site(""),
            *(item for k in (1, 2, 3, 4) for item in _site(f"neighbor_{k}_")),
            _item("parity", "parity", "int"),
            _item("state", "state", "int", required=True),
        ),
        "private",
    ),
    CategoryCodec(
        "ofoldx_bond_stereo",
        tuple(f"end_{k}_{item}" for k in (1, 2) for item in ("asym_id", "component", "atom_id")),
        (
            *(item for k in (1, 2) for item in _site(f"end_{k}_")),
            *(item for k in (1, 2) for item in _site(f"reference_{k}_")),
            _item("value", "value", "int"),
            _item("state", "state", "int", required=True),
        ),
        "private",
    ),
    CategoryCodec(  # the CCD artifact the definitions were resolved against
        "ofoldx_library",
        ("sha256",),
        (_item("content_version", "content_version", "text"), _item("sha256", "sha256", "text")),
        "private",
    ),
)

MMCIF_CODECS: Mapping[str, CategoryCodec] = MappingProxyType(
    {codec.category: codec for codec in (*_CATEGORIES, *_PRIVATE)}
)
