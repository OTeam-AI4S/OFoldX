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
Construction of a ``MolecularSystem`` from an explicit recipe through the same attachment core and the one lowering a
read uses; one-letter sequence construction; component instantiation and replacement admissibility for design.

A recipe states every sequence, every link and every bond it wants: junctions and caps of each consecutive pair of a
``PolymerSequence`` (unless ``linked`` says the pair is not bonded), recipe bonds as declarations with the caller's
order and leaving ids, and edges another value resolved (``InheritedEdges``) with their resolved states. Nothing is
placed or observed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import IntEnum
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

from ..data.editing import ComponentRows
from ..data.molecular import (
    AtomSite,
    BondKind,
    EntityType,
    MolecularSystem,
    MonomerType,
    NstdLinkage,
    Transforms,
    endpoint_resolution,
)
from ..data.stereo import StereoColumns
from ..errors import CODE_NAMES, RecordKind, RequestRef, fail
from ..provenance import Basis, Records, Resolution, SourceRefs, known, state
from .attachments import resolve_attachments
from .declarations import (
    AttachmentInputs,
    DeclarationTable,
    EndpointTable,
    LinkKind,
    OccurrenceTable,
    declaration_columns,
    declaration_table,
    in_use,
    inherited_declarations,
    join_declarations,
    polymer_link_declarations,
    recipe_declarations,
)
from .definitions import (
    ChemicalComponent,
    ComponentLibrary,
    DefinitionRows,
    Definitions,
    configured_library,
    resolve_definitions,
)
from .fragments import FAMILY_TABLE, POLYMER_FAMILIES, TERMINAL_CAPS, LinkingFamily, PortStatus, ports
from .realize import OccurrenceLayout, realize_graph

_DECLARED = state(Resolution.KNOWN, Basis.DECLARED)
_DEFINED = state(Resolution.KNOWN, Basis.DEFINED)
_ABSENT = state(Resolution.UNKNOWN, Basis.NONE)
_NOT_APPLICABLE = state(Resolution.INAPPLICABLE, Basis.DECLARED)
_CLEARED = state(Resolution.UNKNOWN, Basis.DERIVED)

_PROTEIN = dict(
    zip(
        "ACDEFGHIKLMNPQRSTVWY",
        [
            "ALA",
            "CYS",
            "ASP",
            "GLU",
            "PHE",
            "GLY",
            "HIS",
            "ILE",
            "LYS",
            "LEU",
            "MET",
            "ASN",
            "PRO",
            "GLN",
            "ARG",
            "SER",
            "THR",
            "VAL",
            "TRP",
            "TYR",
        ],
    )
)
_ALPHABETS = {
    MonomerType.PROTEIN: _PROTEIN | {"U": "SEC", "O": "PYL", "X": "UNK"},
    MonomerType.RNA: {symbol: symbol for symbol in "ACGUIN"} | {"T": "U"},  # T read as U
    MonomerType.DNA: {symbol: f"D{symbol}" for symbol in "ACGTIN"} | {"U": "DT"},  # U read as T
}
# The entity polymer type of the standard components of each alphabet (every chiral amino acid L; RNA only; DNA only).
_POLYMER_TYPES = {
    MonomerType.PROTEIN: "polypeptide(L)",
    MonomerType.RNA: "polyribonucleotide",
    MonomerType.DNA: "polydeoxyribonucleotide",
}


@dataclass(frozen=True)
class PolymerSequence:
    r"""An entity's component sequence; ``linked`` per consecutive pair whether it is bonded (None: every pair)."""

    polymer_type: str
    components: tuple[str, ...]
    linked: tuple[bool, ...] | None = None


def polymer_sequence(text: str, monomer: MonomerType) -> PolymerSequence:
    r"""
    One-letter text as CCD ids: protein — the 20 standard letters, ``U`` SEC, ``O`` PYL, ``X`` UNK; RNA —
    ``A C G U I N``, ``T`` read as ``U``; DNA — ``A C G T I N`` as ``DA``…``DN``, ``U`` read as ``T``.

    Raises:
        InvalidRequestError: ``request_invalid_sequence`` naming the first symbol with no single component (an
            ambiguity class, a gap or stop symbol, lower case); no member of an ambiguity class is guessed.

    Examples:
        >>> polymer_sequence("GAU", MonomerType.RNA).components
        ('G', 'A', 'U')
    """

    alphabet = _ALPHABETS[monomer]
    for position, symbol in enumerate(text):
        if symbol not in alphabet:
            fail("request_invalid_sequence", RequestRef("text", text), position=position, symbol=symbol)
    return PolymerSequence(_POLYMER_TYPES[monomer], tuple(alphabet[symbol] for symbol in text))


@dataclass(frozen=True)
class Molecule:
    r"""A non-polymer entity's components, joined only by recipe bonds."""

    components: tuple[str, ...]


@dataclass(frozen=True)
class RecipeBond:
    r"""A bond the caller declares, with its order (None: not stated) and per side the leaving atom ids it declares
    (a dictionary-class override)."""

    first: AtomSite
    second: AtomSite
    order: int | None
    kind: BondKind
    leaving: tuple[tuple[str, ...], tuple[str, ...]] = ((), ())


@dataclass(frozen=True)
class Recipe:
    r"""Entities by symbol, instances (instance symbol → entity symbol, in output order) and recipe bonds."""

    entities: Mapping[str, PolymerSequence | Molecule]
    instances: Mapping[str, str]
    bonds: tuple[RecipeBond, ...] = ()


#: The resolved edge columns an inherited edge carries, each with its state but the origin.
INHERITED_COLUMNS = (
    *(f"bond_{name}{suffix}" for name in ("order", "aromatic", "kind", "direction") for suffix in ("", "_state")),
    "bond_origin",
)


@dataclass(frozen=True)
class InheritedEdges:
    r"""
    Edges another value already resolved, restated in a recipe as they are (a caller's request is a ``RecipeBond``).
    Per edge: its two atoms, ``first`` the resolved edge's first, or its end in the value when ``second`` is outside;
    ``columns`` its ``INHERITED_COLUMNS`` (``bond_direction`` relative to ``first``); its connection type (None:
    none); its declared leaving ids per side; whether ``second`` is outside the value (``outside``, default none): an
    atom the value does not hold, named by its own instance symbol (never one of the recipe's), ordinal and atom
    name, which the value keeps as an external connection of ``first``.
    """

    first: tuple[AtomSite, ...]
    second: tuple[AtomSite, ...]
    columns: Mapping[str, NDArray]
    connection_type: tuple[str | None, ...]
    leaving: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...]
    outside: tuple[bool, ...] = ()


def build_system(
    recipe: Recipe,
    library: ComponentLibrary | None = None,
    components: Mapping[str, ChemicalComponent] = MappingProxyType({}),
    *,
    inherited: InheritedEdges | None = None,
) -> tuple[MolecularSystem, Definitions, Records]:
    r"""
    The value a recipe states: its component ids resolved by ``resolve_definitions`` with ``components`` at
    archive scope (the read's precedence; a SMILES definition is ``read_smiles(...).component``), one occurrence per
    (instance, position), the junctions and caps of every linked consecutive pair (existence KNOWN), recipe bonds as
    declarations (origin RECIPE) and ``inherited`` edges as the declarations that restate their resolved states
    (``declarations.inherited_declarations``; one to an atom outside the value through an endpoint outside it, kept as
    an external connection of its end in the value), merged per atom pair, then ``resolve_attachments`` and
    ``realize_graph``. ``library`` defaults to the configured one. Returns the value, the definitions it was built
    from and the lowering's records.

    Raises:
        ContradictionError: ``read_conflicts`` carrying the Conflict records of the definitions, as the default read
            refuses them: a caller entry that disagrees with the library entry of its id is never built around (a
            withheld id would leave its components without atoms).
        InvalidRequestError: ``value_invalid`` for an unverifiable or family-incompatible polymer junction, a recipe
            bond or inherited edge naming an atom no definition in
            force has, an inherited edge whose states no declaration restates, or one outside the value that names an
            instance of the recipe or declares a leaving id of its outside atom.
    """

    ids = sorted({cid for entity in recipe.entities.values() for cid in entity.components})
    library = configured_library() if library is None else library
    definitions = resolve_definitions(ids, DefinitionRows(MappingProxyType({})), library, components)
    conflicts = definitions.records.where(kind=RecordKind.CONFLICT)
    if len(conflicts):
        fail("read_conflicts", conflicts=tuple(conflicts[k] for k in range(len(conflicts))))
    system, records = _build(recipe, definitions, inherited)
    return system, definitions, records


def _build(
    recipe: Recipe, definitions: Definitions, inherited: InheritedEdges | None = None
) -> tuple[MolecularSystem, Records]:
    entity_symbols = tuple(recipe.entities)
    instance_symbols = tuple(recipe.instances)
    occurrence_ids: list[str] = []
    instance_ptr = [0]
    for instance in instance_symbols:
        occurrence_ids += recipe.entities[recipe.instances[instance]].components
        instance_ptr.append(len(occurrence_ids))
    used = sorted({cid for cid in occurrence_ids if cid in definitions.by_id and definitions.by_id[cid].atom_names})
    defs = tuple(definitions.by_id[cid] for cid in used)
    row = {cid: k for k, cid in enumerate(used)}
    definition = np.array([row.get(cid, -1) for cid in occurrence_ids], dtype=np.int32)
    sizes = np.array([len(defs[d].atom_names) if d >= 0 else 0 for d in definition.tolist()], dtype=np.int64)
    total = int(sizes.sum())
    occurrences = OccurrenceTable(
        ordinal=np.arange(len(definition), dtype=np.int32),
        definition=definition,
        unusable=np.zeros(len(definition), dtype=bool),
        unsupported=np.zeros(len(definition), dtype=bool),
        atom_ptr=np.concatenate(([0], np.cumsum(sizes))).astype(np.int64),
        placed=np.zeros(total, dtype=bool),
        observed=np.zeros(total, dtype=bool),
        zero_occupancy=np.zeros(total, dtype=bool),
        atom_ref=np.full(total, -1, dtype=np.int64),
        charge=np.zeros(total, dtype=np.int8),
        charge_state=np.full(total, _ABSENT, dtype=np.uint8),
        observed_only=np.zeros(total, dtype=np.uint8),
    )
    outside_ends = () if inherited is None else zip(inherited.second, inherited.outside)
    outside_names = {site.atom for site, out in outside_ends if out}  # an atom outside the value is named, never built
    names = tuple(sorted({name for d in defs for name in d.atom_names} | outside_names))
    code = {name: k for k, name in enumerate(names)}

    pairs, kinds = [], []
    for k, instance in enumerate(instance_symbols):
        entity = recipe.entities[recipe.instances[instance]]
        if isinstance(entity, PolymerSequence):
            for i in range(len(entity.components) - 1):
                if entity.linked is None or entity.linked[i]:
                    capped = {entity.components[i], entity.components[i + 1]} & set(TERMINAL_CAPS)
                    family = POLYMER_FAMILIES.get(entity.polymer_type)
                    if capped:
                        if family != LinkingFamily.PEPTIDE:
                            fail("value_invalid", RequestRef("polymer_type", entity.polymer_type), reason="cap_family")
                    else:
                        for occurrence in (instance_ptr[k] + i, instance_ptr[k] + i + 1):
                            index = definition[occurrence]
                            if index < 0:
                                continue  # the shared declaration builder reports the missing definition
                            component = defs[index]
                            rule = (
                                FAMILY_TABLE.get((component.chem_comp_type or "").strip().upper())
                                if known(component.chem_comp_type_state)
                                else None
                            )
                            if family is None or rule is None or rule[0] != family:
                                fail(
                                    "value_invalid",
                                    RequestRef("linked", f"{instance} position {i}: {component.component_id}"),
                                    reason="junction_family",
                                    polymer_type=entity.polymer_type,
                                )
                    pairs.append((instance_ptr[k] + i, instance_ptr[k] + i + 1))
                    kinds.append(LinkKind.CAP if capped else LinkKind.JUNCTION)
    count = len(pairs)
    links = polymer_link_declarations(
        np.array(pairs, dtype=np.int32).reshape(-1, 2),
        np.array(kinds, dtype=np.uint8),
        np.full(count, _DEFINED, dtype=np.uint8),
        np.zeros(count, dtype=np.uint16),
        occurrences,
        defs,
        code,
    )
    sites = {
        (instance, ordinal): instance_ptr[k] + ordinal
        for k, instance in enumerate(instance_symbols)
        for ordinal in range(instance_ptr[k + 1] - instance_ptr[k])
    }
    parts = [links, _recipe_bonds(recipe.bonds, sites, occurrences, defs, code)]
    types: tuple[str, ...] = ()
    outside_symbols: tuple[str, ...] = ()
    if inherited is not None:
        away = inherited.outside or (False,) * len(inherited.first)
        ends = list(zip(inherited.first, inherited.second, inherited.leaving))
        first, second, leaving = _ends(ends, "InheritedEdges", sites, occurrences, defs, code, away)
        outside_symbols = tuple(sorted({site.instance for site, out in zip(inherited.second, away) if out}))
        if taken := set(outside_symbols) & set(instance_symbols):
            fail("value_invalid", RequestRef("InheritedEdges", min(taken)), reason="an outside atom names an instance")
        named = np.array(
            [
                (len(instance_symbols) + outside_symbols.index(site.instance), site.component) if out else (-1, -1)
                for site, out in zip(inherited.second, away)
            ],
            dtype=np.int64,
        ).reshape(-1, 2)
        parts.append(inherited_declarations(first, second, inherited.columns, leaving, named))
        types = tuple(sorted({text for text in inherited.connection_type if text is not None}))
    endpoints, declarations = join_declarations(parts)
    # one physical edge per atom pair, numbered in the order of its first row: a bond stated twice is merged once
    located = np.flatnonzero(declarations.second >= 0)
    physical = np.full(len(declarations.first), -1, dtype=np.int32)
    if len(located):
        pair = np.sort(np.stack((declarations.first[located], declarations.second[located]), axis=1), axis=1)
        _, start, group = np.unique(pair, axis=0, return_index=True, return_inverse=True)
        rank = np.empty(len(start), dtype=np.int32)
        rank[np.argsort(start)] = np.arange(len(start), dtype=np.int32)
        physical[located] = rank[group.ravel()]
    declarations = declaration_table(**(declaration_columns(declarations) | {"physical_edge": physical}))
    connection = np.full(len(physical), -1, dtype=np.int32)
    if inherited is not None and inherited.first:
        stated = [types.index(text) if text is not None else -1 for text in inherited.connection_type]
        connection[-len(stated) :] = stated

    identity = Transforms(
        np.eye(3, dtype=np.int8)[None],
        np.zeros((1, 3), dtype=np.int16),
        np.full(1, -1, dtype=np.int32),
        np.full(1, _DEFINED, dtype=np.uint8),
    )
    inputs = AttachmentInputs(
        occurrences,
        defs,
        endpoints,
        declarations,
        identity,
        in_use(occurrences, endpoints, declarations),
        _no_refs(),
        (),
        np.zeros(len(defs) + 1, dtype=np.int64),  # a recipe has no source to name
        np.zeros(0, dtype=np.int64),
    )
    result = resolve_attachments(inputs)
    layout = _layout(
        recipe,
        entity_symbols,
        instance_symbols,
        instance_ptr,
        occurrence_ids,
        names,
        total,
        connection,
        types,
        outside_symbols,
    )
    graph = realize_graph(inputs, result, layout)
    return MolecularSystem.from_arrays(graph.system), graph.records


def _ends(
    ends: Sequence[tuple[AtomSite, AtomSite, tuple[tuple[str, ...], tuple[str, ...]]]],
    argument: str,
    sites: Mapping[tuple[str, int], int],
    occurrences: OccurrenceTable,
    defs: tuple[ChemicalComponent, ...],
    code: Mapping[str, int],
    outside: Sequence[bool] = (),
) -> tuple[NDArray[np.int64], NDArray[np.int64], list[list[tuple[int, int]]]]:
    r"""Per bond its two endpoints (occurrence, definition atom, name code) and its declared leaving ids as (definition
    atom, side), named by the definitions in force; an atom none of them has is ``value_invalid``. A second end
    ``outside`` the value is its name code only (-1 occurrence and atom), and declares no leaving id (``value_invalid``:
    no definition in force names its atoms)."""

    def atom(site: AtomSite, name: str) -> tuple[int, int]:
        occurrence = sites.get((site.instance, site.component), -1)
        d = int(occurrences.definition[occurrence]) if occurrence >= 0 else -1
        if d < 0 or name not in defs[d].atom_names:
            fail("value_invalid", RequestRef(argument, f"{site.instance} {site.component} {name}"))
        return occurrence, defs[d].atom_names.index(name)

    first, second, leaving = [], [], []
    for k, (a, b, declared) in enumerate(ends):
        away = k < len(outside) and outside[k]
        if away and declared[1]:
            fail("value_invalid", RequestRef(argument, f"{b.instance} {b.component} {declared[1][0]}"), outside=True)
        first.append((*atom(a, a.atom), code[a.atom]))
        second.append((-1, -1, code[b.atom]) if away else (*atom(b, b.atom), code[b.atom]))
        leaving.append([(atom(site, name)[1], side) for side, site in enumerate((a, b)) for name in declared[side]])
    return np.array(first, dtype=np.int64).reshape(-1, 3), np.array(second, dtype=np.int64).reshape(-1, 3), leaving


def _recipe_bonds(
    bonds: tuple[RecipeBond, ...],
    sites: Mapping[tuple[str, int], int],
    occurrences: OccurrenceTable,
    defs: tuple[ChemicalComponent, ...],
    code: Mapping[str, int],
) -> tuple[EndpointTable, DeclarationTable]:
    r"""One declaration per recipe bond (``declarations.recipe_declarations``), its atoms named by the definitions in
    force."""

    ends = [(bond.first, bond.second, bond.leaving) for bond in bonds]
    first, second, leaving = _ends(ends, "RecipeBond", sites, occurrences, defs, code)
    return recipe_declarations(
        first,
        second,
        np.array([bond.kind for bond in bonds], dtype=np.uint8),
        np.array([bond.order or 0 for bond in bonds], dtype=np.uint8),
        leaving,
        occurrences,
        defs,
    )


def _no_refs() -> SourceRefs:
    return SourceRefs(
        np.zeros(0, dtype=np.int16),
        np.zeros(0, dtype=np.int64),
        np.zeros(0, dtype=np.int16),
        (),
        (),
        None,
    )


def _layout(
    recipe, entity_symbols, instance_symbols, instance_ptr, occurrence_ids, names, atoms, connection, types, outside
):
    r"""Entities and instances as the recipe states them: a polymer's type and ``nstd_linkage`` NO (a recipe states
    every link), a one-component molecule NON-POLYMER, the type of a several-component molecule unstated; nothing is
    observed. ``connection`` is each declaration row's connection type, a code of ``types`` (-1 none); ``outside`` the
    instance symbols endpoints outside the value name."""

    entities = [recipe.entities[symbol] for symbol in entity_symbols]
    polymer_types = tuple(sorted({e.polymer_type for e in entities if isinstance(e, PolymerSequence)}))
    components = tuple(sorted(set(occurrence_ids)))
    polymer = np.array([isinstance(e, PolymerSequence) for e in entities], dtype=bool)
    single = np.array([len(e.components) == 1 for e in entities], dtype=bool)
    entity_type = np.where(polymer, EntityType.POLYMER, np.where(single, EntityType.NON_POLYMER, 0))
    occurrences = instance_ptr[-1]
    return OccurrenceLayout(
        entity_symbol=entity_symbols,
        entity_type=entity_type.astype(np.uint8),
        entity_type_state=np.where(polymer | single, _DECLARED, _ABSENT).astype(np.uint8),
        entity_polymer_type=np.array(
            [polymer_types.index(e.polymer_type) if p else -1 for e, p in zip(entities, polymer)], dtype=np.int32
        ),
        entity_polymer_type_state=np.where(polymer, _DECLARED, _NOT_APPLICABLE).astype(np.uint8),
        entity_nstd_linkage=np.where(polymer, NstdLinkage.NO, 0).astype(np.uint8),
        entity_nstd_linkage_state=np.where(polymer, _DECLARED, _NOT_APPLICABLE).astype(np.uint8),
        instance_symbol=instance_symbols,
        instance_entity=np.array([entity_symbols.index(recipe.instances[i]) for i in instance_symbols], dtype=np.int32),
        instance_occurrence_ptr=np.array(instance_ptr, dtype=np.int64),
        occurrence_component=np.array([components.index(cid) for cid in occurrence_ids], dtype=np.int32),
        observed_element=np.zeros(atoms, dtype=np.uint8),
        observed_isotope=np.zeros(atoms, dtype=np.uint16),
        extra_ptr=np.zeros(occurrences + 1, dtype=np.int64),
        extra_name=np.zeros(0, dtype=np.int32),
        extra_element=np.zeros(0, dtype=np.uint8),
        extra_isotope=np.zeros(0, dtype=np.uint16),
        extra_charge=np.zeros(0, dtype=np.int8),
        extra_charge_state=np.zeros(0, dtype=np.uint8),
        extra_ref=np.zeros(0, dtype=np.int64),
        declared_parent_ptr=np.zeros(occurrences + 1, dtype=np.int64),
        declared_parent=np.zeros(0, dtype=np.int32),
        declared_parent_ref=np.zeros(0, dtype=np.int64),
        declaration_connection_type=connection,
        unlinked_pairs=np.zeros((0, 2), dtype=np.int32),
        connections_open=np.zeros(len(instance_symbols), dtype=bool),
        symbols=MappingProxyType(
            {
                "atom": names,
                "component": components,
                "polymer_type": polymer_types,
                "parent": (),
                "connection_type": types,
                "outside_instance": outside,
            }
        ),
    )


def instantiate_component(definition: ChemicalComponent, *, linked: tuple[bool, bool]) -> ComponentRows:
    r"""
    One component as final rows: the definition built by the same core and lowering as the middle of a
    sequence of itself, so Rule L at each linked port removes exactly what a junction there removes (the new residue
    loses OXT/HXT or H2 where linked). A stereocentre slot or stereobond reference on the partner side becomes the
    implicit slot ``-1``; a row left with two such slots is cleared.

    Raises:
        InvalidRequestError: ``value_invalid`` for a linked side whose port is not VERIFIED.
    """

    for side, (wanted, status) in enumerate(zip(linked, ports(definition).status)):
        if wanted and status != PortStatus.VERIFIED:
            fail(
                "value_invalid",
                RequestRef("linked", f"{definition.component_id} side {side}: port {PortStatus(status).name}"),
            )
    family = FAMILY_TABLE.get((definition.chem_comp_type or "").upper())
    polymer_type = _POLYMER_TYPES.get(family[1] if family else MonomerType.NONE, "other")
    cid = definition.component_id
    recipe = Recipe({"P": PolymerSequence(polymer_type, (cid,) * (1 + sum(linked)))}, {"P": "P"})
    built = Definitions(MappingProxyType({cid: definition}), Records.empty(), None)
    system, _ = _build(recipe, built)
    return _component_rows(system, int(linked[0]))


def _component_rows(system: MolecularSystem, component: int) -> ComponentRows:
    r"""The rows of one component of a value, local to it (``data.editing.ComponentRows``)."""

    start, stop = (int(x) for x in system.component_atom_ptr[component : component + 2])
    local = np.full(len(system.atom_symbol), -1, dtype=np.int64)
    local[start:stop] = np.arange(stop - start)
    atom_columns = [f.name for f in MolecularSystem.SCHEMA.fields if f.axis == "atom" and f.name != "atom_symbol"]
    atoms = {name: getattr(system, name)[start:stop].copy() for name in atom_columns}
    rows = system.atom_reference_row[start:stop].astype(np.int64)
    pool = np.unique(rows[rows >= 0])
    atoms["atom_reference_row"] = np.where(rows >= 0, np.searchsorted(pool, rows), -1).astype(np.int32)

    first, second = system.edge_index.astype(np.int64)
    edges = np.flatnonzero((local[first] >= 0) & (local[second] >= 0))
    edge_local = np.full(len(first), -1, dtype=np.int64)
    edge_local[edges] = np.arange(len(edges))
    bond_columns = [
        f.name
        for f in MolecularSystem.SCHEMA.fields
        if f.axis == "edge" and f.name not in ("edge_index", "bond_connection_type")
    ]

    centres = system.stereocentre_atom.astype(np.int64)
    kept = np.flatnonzero(local[centres] >= 0)
    slots = system.stereocentre_neighbors[kept].astype(np.int64)
    slots = np.where(slots >= 0, local[np.maximum(slots, 0)], -1)  # the partner (or an external row) → -1
    cleared = (slots == -1).sum(axis=1) > 1
    bond_rows = np.flatnonzero(edge_local[system.stereobond_edge] >= 0)
    references = local[system.stereobond_reference[bond_rows].astype(np.int64)]
    bond_cleared = (references < 0).any(axis=1)
    row_of = np.full(len(centres), -1, dtype=np.int64)
    row_of[kept] = np.arange(len(kept))
    member = row_of[system.stereogroup_member]
    group = np.repeat(np.arange(len(system.stereogroup_kind)), np.diff(system.stereogroup_ptr))
    stays = member >= 0
    sizes = np.bincount(group[stays], minlength=len(system.stereogroup_kind))
    stereo = StereoColumns(
        centre=local[centres[kept]].astype(np.int32),
        parity=np.where(cleared, 0, system.stereocentre_parity[kept]).astype(np.int8),
        neighbors=np.where(cleared[:, None], -1, slots).astype(np.int32),
        centre_state=np.where(cleared, _CLEARED, system.stereocentre_state[kept]).astype(np.uint8),
        bond_edge=edge_local[system.stereobond_edge[bond_rows]].astype(np.int32),
        bond_value=np.where(bond_cleared, 0, system.stereobond_value[bond_rows]).astype(np.int8),
        bond_reference=np.where(bond_cleared[:, None], -1, references).astype(np.int32),
        bond_state=np.where(bond_cleared, _CLEARED, system.stereobond_state[bond_rows]).astype(np.uint8),
        group_kind=system.stereogroup_kind[sizes > 0].copy(),
        group_ptr=np.concatenate(([0], np.cumsum(sizes[sizes > 0]))).astype(np.int64),
        group_member=member[stays].astype(np.int32),
    )

    def parents(name: str) -> tuple[str, ...]:
        pointer = getattr(system, f"{name}_ptr")
        return tuple(system.text("parent", getattr(system, name)[pointer[component] : pointer[component + 1]]).tolist())

    port = system.component_port[component].astype(np.int64)
    topology, heavy_atoms = (
        bool(getattr(system, f"component_{c}_complete")[component]) for c in ("topology", "heavy_atoms")
    )
    previous_port, next_port = (int(local[p]) if p >= 0 else -1 for p in port.tolist())
    return ComponentRows(
        symbol=str(system.text("component", system.component_symbol[component : component + 1])[0]),
        names=tuple(system.text("atom", system.atom_symbol[start:stop]).tolist()),
        atoms=atoms,
        edges=np.stack((local[first[edges]], local[second[edges]])),
        bonds={name: getattr(system, name)[edges].copy() for name in bond_columns},
        stereo=stereo,
        claims=(topology, heavy_atoms),
        parents=parents("component_parent"),
        definition_parents=parents("component_definition_parent"),
        monomer_type=int(system.component_monomer_type[component]),
        monomer_type_state=int(system.component_monomer_type_state[component]),
        definition_scope=int(system.component_definition_scope[component]),
        definition_digest=int(system.component_definition_digest[component]),
        port=(previous_port, next_port),
        reference=system.reference_pool_coordinates[pool].copy(),
        reference_source=system.reference_pool_source[pool].copy(),
    )


class Admissibility(IntEnum):
    ADMISSIBLE = 0
    UNDETERMINED = 1
    INADMISSIBLE = 2


@dataclass(frozen=True)
class ReplacementCheck:
    r"""Per edge between the component and another (edge index), the outcome and its reason (``CODE_NAMES`` index;
    0 when admissible)."""

    edge: NDArray[np.int64]
    outcome: NDArray[np.uint8]
    reason: NDArray[np.uint16]


_ATOM_ABSENT, _ELEMENT_DIFFERS, _ENDPOINT_UNDETERMINED = (
    CODE_NAMES.index(f"replacement_{name}") for name in ("atom_absent", "element_differs", "endpoint_undetermined")
)


def replacement_admissibility(
    system: MolecularSystem, component: int, replacement: ChemicalComponent
) -> ReplacementCheck:
    r"""
    Whether ``replacement`` can take ``component``'s place at each edge to another component: UNDETERMINED when the
    endpoint's state or element is not KNOWN (``replacement_endpoint_undetermined``); INADMISSIBLE when the
    replacement has no atom of the endpoint's name (``replacement_atom_absent``) or one of another element
    (``replacement_element_differs``); ADMISSIBLE otherwise. External connections are not edges and are not checked.
    """

    start, stop = system.component_atom_ptr[component], system.component_atom_ptr[component + 1]
    first, second = system.edge_index.astype(np.int64)
    inside_first, inside_second = (first >= start) & (first < stop), (second >= start) & (second < stop)
    edges = np.flatnonzero(inside_first != inside_second)
    endpoint = np.where(inside_first[edges], first[edges], second[edges])
    index = {name: k for k, name in enumerate(replacement.atom_names)}
    resolution = endpoint_resolution(system.atom_endpoint_state[endpoint])
    element_known = known(system.atomic_number_state[endpoint])
    outcome = np.zeros(len(edges), dtype=np.uint8)
    reason = np.zeros(len(edges), dtype=np.uint16)
    for k, name in enumerate(system.text("atom", system.atom_symbol[endpoint]).tolist()):
        if resolution[k] in (Resolution.UNKNOWN, Resolution.CONFLICT) or not element_known[k]:
            outcome[k], reason[k] = Admissibility.UNDETERMINED, _ENDPOINT_UNDETERMINED
        elif name not in index:
            outcome[k], reason[k] = Admissibility.INADMISSIBLE, _ATOM_ABSENT
        elif replacement.element[index[name]] != system.atomic_number[endpoint[k]]:
            outcome[k], reason[k] = Admissibility.INADMISSIBLE, _ELEMENT_DIFFERS
    return ReplacementCheck(edges.astype(np.int64), outcome, reason)
