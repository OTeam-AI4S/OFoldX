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
The one lowering of attachment results into graph arrays, and the valence excess read off a lowered value.

``realize_graph`` turns the attachment result of one chemical state into ``SystemArrays``: one component per native
occurrence (component ``k`` is occurrence ``k``), its created atoms, facts with their states, the reference bonds and
the merged new edges, external connections with exact transforms, parity and claims. A read (``io.read``) and a recipe
(``chemistry.construction``) call it; no second lowering exists.

``valence_excess`` needs no definition and no library: everything it uses was lowered.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

from ..data.molecular import (
    ELEMENTS,
    AtomFate,
    BondKind,
    BondOrigin,
    EndpointState,
    MolecularSystem,
    ReferenceSource,
    SystemArrays,
    Transforms,
)
from ..data.stereo import StereoColumns, remap_parity
from ..errors import CODE_NAMES
from ..provenance import Basis, Records, Resolution, SubjectKind, known, state
from .attachments import AttachmentResult, Dependency, Hydrogens, StereoAction
from .declarations import AttachmentInputs, MergedEdges, OrderToken, Restatement
from .definitions import ChemicalComponent, StereoBasis
from .fragments import FAMILY_TABLE, PortStatus, ports

_H = ELEMENTS.index("H")
_KNOWN_DECLARED = state(Resolution.KNOWN, Basis.DECLARED)
_KNOWN_DEFINED = state(Resolution.KNOWN, Basis.DEFINED)
_KNOWN_DERIVED = state(Resolution.KNOWN, Basis.DERIVED)
_KNOWN_REFERENCE = state(Resolution.KNOWN, Basis.REFERENCE)
_KNOWN_ASSUMED = state(Resolution.KNOWN, Basis.ASSUMED)
_UNKNOWN = state(Resolution.UNKNOWN, Basis.NONE)
_UNKNOWN_DECLARED = state(Resolution.UNKNOWN, Basis.DECLARED)
_UNKNOWN_DERIVED = state(Resolution.UNKNOWN, Basis.DERIVED)
_UNKNOWN_REFERENCE = state(Resolution.UNKNOWN, Basis.REFERENCE)
_INAPPLICABLE_DEFINED = state(Resolution.INAPPLICABLE, Basis.DEFINED)
_OPEN = (EndpointState.UNKNOWN, EndpointState.CONFLICT)


@dataclass(frozen=True, eq=False)
class OccurrenceLayout:
    r"""
    How native occurrences form entities, instances and components, and the observed facts the lowering needs beyond
    ``AttachmentInputs``. Occurrences are in instance order (``instance_occurrence_ptr``); ``occurrence_component``
    codes each occurrence's component id. ``observed_element`` and ``observed_isotope`` are the element and isotope (0
    none) of each occurrence atom's selected row (0 unplaced): an unusable definition is replaced by what was observed,
    and a stated isotope is the atom's own fact over the definition's. Extras are observed atoms with no definition
    atom (CSR ``extra_ptr``), with their name, element, isotope (0 none), raw charge fact and row of
    ``AttachmentInputs.refs``. ``declared_parent_*`` are the parents an occurrence's ``_pdbx_struct_mod_residue`` or
    MODRES rows state (its annotation); ``declaration_connection_type`` the
    ``conn_type_id``/``SSBOND``/``LINK``/``CONECT`` of each declaration row (-1 none); ``unlinked_pairs`` occurrence
    pairs with ``consecutive_pair_unlinked``; ``connections_open`` io's evidence that a connection of an instance is
    unrepresented. ``symbols`` holds the families the codes index: ``atom`` (also
    ``EndpointTable.name``), ``component``, ``polymer_type``, ``parent``, ``connection_type`` and, for a built value
    with endpoints outside it, ``outside_instance``: the instance symbols they name, the instance family's codes after
    ``instance_symbol``'s.
    """

    entity_symbol: tuple[str, ...]
    entity_type: NDArray[np.uint8]
    entity_type_state: NDArray[np.uint8]
    entity_polymer_type: NDArray[np.int32]
    entity_polymer_type_state: NDArray[np.uint8]
    entity_nstd_linkage: NDArray[np.uint8]
    entity_nstd_linkage_state: NDArray[np.uint8]
    instance_symbol: tuple[str, ...]
    instance_entity: NDArray[np.int32]
    instance_occurrence_ptr: NDArray[np.int64]
    occurrence_component: NDArray[np.int32]
    observed_element: NDArray[np.uint8]
    observed_isotope: NDArray[np.uint16]
    extra_ptr: NDArray[np.int64]
    extra_name: NDArray[np.int32]
    extra_element: NDArray[np.uint8]
    extra_isotope: NDArray[np.uint16]
    extra_charge: NDArray[np.int8]
    extra_charge_state: NDArray[np.uint8]
    extra_ref: NDArray[np.int64]
    declared_parent_ptr: NDArray[np.int64]
    declared_parent: NDArray[np.int32]
    declared_parent_ref: NDArray[np.int64]
    declaration_connection_type: NDArray[np.int32]
    unlinked_pairs: NDArray[np.int32]
    connections_open: NDArray[np.bool_]
    symbols: Mapping[str, tuple[str, ...]]


@dataclass(frozen=True, eq=False)
class GraphArrays:
    r"""
    The lowered native value and its maps: occurrence atom, extra and endpoint → graph atom (-1 not created);
    ``MergedEdges`` row and declaration row → graph edge, ``-2 - x`` for external row ``x`` (the one at the edge's
    first endpoint), -1 not realised; ``records`` in graph indices (ATOM, EDGE, EXTERNAL, COMPONENT = occurrence row;
    OCCURRENCE stays the native ordinal).
    """

    system: SystemArrays
    occurrence_atom: NDArray[np.int64]
    extra_atom: NDArray[np.int64]
    endpoint_atom: NDArray[np.int64]
    edge_of_physical: NDArray[np.int64]
    edge_of_declaration: NDArray[np.int64]
    records: Records


def _owner(pointer: NDArray[np.int64]) -> NDArray[np.int64]:
    return np.repeat(np.arange(len(pointer) - 1, dtype=np.int64), np.diff(pointer))


def _local(pointer: NDArray[np.int64]) -> NDArray[np.int64]:
    r"""Each row's position inside its segment of the CSR ``pointer``."""
    return np.arange(int(pointer[-1]), dtype=np.int64) - np.repeat(pointer[:-1], np.diff(pointer))


def _ptr(counts: NDArray) -> NDArray[np.int64]:
    return np.concatenate(([0], np.cumsum(counts))).astype(np.int64)


def _any(owner: NDArray[np.int64], mask: NDArray[np.bool_], count: int) -> NDArray[np.bool_]:
    return np.bincount(owner[mask], minlength=count) > 0


class _Tables:
    r"""The per-atom, per-bond and per-row arrays of the definitions of one lowering, concatenated, with offsets; a last
    empty definition answers index -1 (no definition), so every per-definition lookup holds at any count, zero
    included."""

    def __init__(self, definitions: tuple[ChemicalComponent | None, ...], names: Mapping[str, int]) -> None:
        definitions = (*definitions, None)
        found = [d for d in definitions if d is not None]

        def offsets(name: str) -> NDArray[np.int64]:
            return _ptr([len(getattr(d, name)) if d is not None else 0 for d in definitions])

        def join(name: str, dtype, shape=()) -> NDArray:
            parts = [np.asarray(getattr(d, name)) for d in found]
            return np.concatenate(parts).astype(dtype) if parts else np.zeros((0, *shape), dtype)

        self.atom = offsets("element")
        self.element = join("element", np.uint8)
        self.isotope = join("isotope", np.uint16)
        self.aromatic = join("aromatic", bool)
        self.aromatic_state = join("aromatic_state", np.uint8)
        self.name = np.array([names[name] for d in found for name in d.atom_names], dtype=np.int32)
        self.parent = np.concatenate([_parents(d) for d in found] or [np.zeros(0, np.int64)])
        self.designation = np.concatenate([_designation(d) for d in found] or [np.zeros(0, np.uint8)])
        self.bond = offsets("bond_order")
        raw = join("bond_atoms", np.int64, (2,))
        self.bond_atoms = np.sort(raw, axis=1)
        self.bond_order = join("bond_order", np.uint8)
        self.bond_order_state = join("bond_order_state", np.uint8)
        self.bond_aromatic = join("bond_aromatic", bool)
        self.bond_aromatic_state = join("bond_aromatic_state", np.uint8)
        self.centre = offsets("stereo_centre")
        self.centre_atom = join("stereo_centre", np.int64)
        self.parity = join("stereo_parity", np.int8)
        self.slots = join("stereo_neighbors", np.int64, (4,))
        self.stereo_basis = join("stereo_basis", np.uint8)
        self.group = offsets("stereo_group_kind")
        self.group_kind = join("stereo_group_kind", np.uint8)
        self.group_size = np.concatenate([np.diff(d.stereo_group_ptr) for d in found] or [np.zeros(0, np.int64)])
        self.member = offsets("stereo_group_member")
        self.group_member = join("stereo_group_member", np.int64)
        self.bond_stereo = offsets("bond_stereo_bond")
        self.bond_stereo_bond = join("bond_stereo_bond", np.int64)
        self.bond_stereo_value = join("bond_stereo_value", np.int8)
        reference = join("bond_stereo_reference", np.int64, (2,))
        # references follow the low → high order of the sorted bond ends
        rows = self.bond[np.repeat(np.arange(len(definitions)), np.diff(self.bond_stereo))] + self.bond_stereo_bond
        flipped = raw[rows, 0] > raw[rows, 1] if len(rows) else np.zeros(0, bool)
        self.bond_stereo_reference = np.where(flipped[:, None], reference[:, ::-1], reference)
        # a row with no E/Z label was stated by SMILES/CX marks (a CCD row exists only for a label)
        self.bond_stereo_stated = np.array([label is None for d in found for label in d.bond_stereo_declared], bool)
        pooled = np.array([d is not None and d.reference_source != ReferenceSource.NONE for d in definitions], bool)
        sizes = np.diff(self.atom)
        self.pool = np.where(pooled, _ptr(np.where(pooled, sizes, 0))[:-1], -1).astype(np.int64)
        kept = [cast(ChemicalComponent, d) for d, p in zip(definitions, pooled.tolist()) if p]
        self.pool_coordinates = np.concatenate([d.reference for d in kept] or [np.zeros((0, 3))]).astype(np.float32)
        self.pool_source = np.repeat([int(d.reference_source) for d in kept], [len(d.atom_names) for d in kept])


def _parents(definition: ChemicalComponent) -> NDArray[np.int64]:
    r"""Per definition atom, the heavy atom a hydrogen is bonded to (-1 for a heavy atom or an unbonded hydrogen)."""
    parent = np.full(len(definition.atom_names), -1, dtype=np.int64)
    for i, j in definition.bond_atoms.astype(np.int64).tolist():
        for h, x in ((i, j), (j, i)):
            if definition.element[h] == _H and definition.element[x] != _H:
                parent[h] = x
    return parent


def _designation(definition: ChemicalComponent) -> NDArray[np.uint8]:
    r"""Per definition atom, the port whose designated leaving fragment holds it (``data.molecular.Designation``)."""
    designation = np.zeros(len(definition.atom_names), dtype=np.uint8)
    record = ports(definition)
    designation[list(record.previous_designated)] = 1
    designation[list(record.next_designated)] = 2
    return designation


def realize_graph(inputs: AttachmentInputs, result: AttachmentResult, layout: OccurrenceLayout) -> GraphArrays:
    r"""
    Lower one chemical state's attachment result.

    1. Atoms per occurrence: definition atoms with fate RETAINED, UNDETERMINED or UNOBSERVED_ENDPOINT in definition
       order — a hydrogen only when observed or when its parent's decision is MATERIALISE — then the extra observed
       atoms (``observed_atom_not_in_reference`` where a definition stating atoms is in force). An unusable definition
       contributes only its observed atoms, as observed. 2. Component and atom facts with their states (charge and
       allowed valence from the result; an occurrence's parents: its annotation where stated, else its definition's).
       3. Reference bonds whose atoms exist (a restating row's merged fields replace its bond's reference fields on the
       one edge) and the merged new edges; a non-identity or non-KNOWN transform makes external connections at the
       native endpoints (the second one with the inverse transform), an edge to an endpoint outside the value one
       external connection at its endpoint in the value (the identity); an endpoint with no atom leaves the edge
       unrealised (``edge_endpoint_unrealised``). 4. Endpoint states. 5. Parity: definition rows remapped by
       ``data.stereo.remap_parity`` with the TRANSPORT substitutions; UNDETERMINED_NONPORT and CLEAR rows, and every
       centre or stereobond the remap clears, recorded ``parity_cleared_*``. 6. Claims. 7. Records in graph
       indices.
    """

    occ, ends, decl, edges = inputs.occurrences, inputs.endpoints, inputs.declarations, result.edges
    symbols = layout.symbols
    t = _Tables(inputs.definitions, {name: code for code, name in enumerate(symbols["atom"])})
    n_occ, n_inst = len(occ.definition), len(layout.instance_symbol)
    atom_occ, local = _owner(occ.atom_ptr), _local(occ.atom_ptr)
    definition = occ.definition.astype(np.int64)
    usable = (definition >= 0) & ~occ.unusable
    row = t.atom[definition][atom_occ] + local  # each occurrence atom's row in the definition tables
    element = t.element[row]
    is_h = element == _H
    fate = result.fate
    log: list[tuple] = []

    # ---- 1. atoms
    parent = t.parent[row]
    parent_atom = np.where(parent >= 0, occ.atom_ptr[atom_occ] + parent, -1)
    materialised = np.append(result.hydrogens, Hydrogens.NONE)[parent_atom] == Hydrogens.MATERIALISE
    created = (fate != AtomFate.LEFT) & np.where(usable[atom_occ], ~is_h | occ.placed | materialised, occ.placed)
    extra_owner = _owner(layout.extra_ptr)
    created_count = np.bincount(atom_occ[created], minlength=n_occ)
    component_ptr = _ptr(created_count + np.diff(layout.extra_ptr))
    before = _ptr(created_count)[:-1]
    occurrence_atom = np.where(created, component_ptr[atom_occ] + np.cumsum(created) - 1 - before[atom_occ], -1)
    extra_atom = component_ptr[extra_owner] + created_count[extra_owner] + _local(layout.extra_ptr)
    n = int(component_ptr[-1])
    endpoint_atom = np.full(len(ends.occurrence), -1, dtype=np.int64)
    by_atom, by_extra = ends.atom >= 0, ends.extra >= 0
    endpoint_atom[by_atom] = occurrence_atom[occ.atom_ptr[ends.occurrence[by_atom]] + ends.atom[by_atom]]
    endpoint_atom[by_extra] = extra_atom[layout.extra_ptr[ends.occurrence[by_extra]] + ends.extra[by_extra]]

    c = _atoms(t, layout, result, occ, row, element, is_h, created, usable[atom_occ], occurrence_atom, extra_atom, n)
    assumed = endpoint_atom[result.leaving.endpoint[(result.leaving.dependencies & Dependency.ASSUMPTION) > 0]]
    for name in ("implicit_hydrogens_state", "atom_valence_state"):  # results depending on an assumption
        hit = assumed[assumed >= 0]
        c[name][hit] = np.where(known(c[name][hit]), _KNOWN_ASSUMED, c[name][hit])
    stated_end = endpoint_atom[result.leaving.endpoint]
    c["atom_endpoint_state"][stated_end[stated_end >= 0]] = result.leaving.state[stated_end >= 0]
    pooled = created & usable[atom_occ] & (t.pool[definition][atom_occ] >= 0)
    c["atom_reference_row"][occurrence_atom[pooled]] = t.pool[definition[atom_occ[pooled]]] + local[pooled]
    c["reference_pool_coordinates"] = t.pool_coordinates
    c["reference_pool_source"] = t.pool_source.astype(np.uint8)
    # an atom-less or absent reference is reported once, by its reference_incomplete_* record
    resolved = usable & (np.diff(t.atom)[definition] > 0)
    for k in np.flatnonzero(resolved[extra_owner]).tolist():
        log.append(("observed_atom_not_in_reference", SubjectKind.ATOM, int(extra_atom[k]), [layout.extra_ref[k]], ()))

    # ---- 3. edges and external connections
    bond_ptr = _ptr(np.where(usable, np.diff(t.bond)[definition], 0))
    bond_occ = _owner(bond_ptr)
    bond_row = t.bond[definition[bond_occ]] + _local(bond_ptr)
    old_edges = (occ.atom_ptr[bond_occ][:, None] + t.bond_atoms[bond_row]).T.reshape(2, -1)
    ga, gb = occurrence_atom[old_edges[0]], occurrence_atom[old_edges[1]]
    reference = np.flatnonzero((ga >= 0) & (gb >= 0))

    n_edge = len(edges.first)
    edge_group = _owner(edges.declaration_ptr)
    first_end, second_end = edges.first.astype(np.int64), edges.second.astype(np.int64)
    ea, eb = endpoint_atom[first_end], endpoint_atom[second_end]
    tr = inputs.transforms
    identity = known(tr.state) & (tr.rotation == np.eye(3, dtype=np.int8)).all(axis=(1, 2))
    identity &= (tr.translation == 0).all(axis=1)
    restated = edges.edge_class == Restatement.RESTATEMENT
    realised = (ea >= 0) & (eb >= 0)
    internal = realised & identity[edges.transform]
    new_internal = np.flatnonzero(internal & ~restated)
    restating = np.flatnonzero(internal & restated)
    connection_type = np.full(n_edge, -1, dtype=np.int64)  # the first typed row of each edge (representative first)
    typed = np.flatnonzero(layout.declaration_connection_type[edges.declaration] >= 0)
    groups, first_typed = np.unique(edge_group[typed], return_index=True)
    connection_type[groups] = layout.declaration_connection_type[edges.declaration[typed[first_typed]]]

    lo = np.concatenate((np.minimum(ga, gb)[reference], np.minimum(ea, eb)[new_internal]))
    hi = np.concatenate((np.maximum(ga, gb)[reference], np.maximum(ea, eb)[new_internal]))
    order = np.lexsort((hi, lo))
    position = np.empty(len(order), dtype=np.int64)
    position[order] = np.arange(len(order))
    m, m_ref = len(order), len(reference)
    c["edge_index"] = np.stack((lo[order], hi[order])).astype(np.int32).reshape(2, -1)
    edge_map = np.full(len(bond_row), -1, dtype=np.int64)  # old reference bond → graph edge
    edge_map[reference] = position[:m_ref]
    c.update(_reference_bonds(t, bond_row[reference], position[:m_ref], m))

    # a restating row's merged fields replace its reference bond's fields on the one edge (1e61)
    edge_of_physical = np.full(n_edge, -1, dtype=np.int64)
    width = max(len(occ.placed), 1)
    occurrence_end = occ.atom_ptr[ends.occurrence] + ends.atom
    pair = np.sort(np.stack((occurrence_end[first_end[restating]], occurrence_end[second_end[restating]])), axis=0)
    bond_key = old_edges[0] * width + old_edges[1]
    key_order = np.argsort(bond_key)
    found = key_order[np.searchsorted(bond_key[key_order], pair[0] * width + pair[1])] if len(restating) else restating
    edge_of_physical[restating] = edge_map[found]
    edge_of_physical[new_internal] = position[m_ref:]
    on_edge = np.full(m, -1, dtype=np.int64)
    on_edge[edge_of_physical[edge_of_physical >= 0]] = np.flatnonzero(edge_of_physical >= 0)
    merged = np.flatnonzero(on_edge >= 0)
    g = on_edge[merged]
    swap = ea[g] > eb[g]
    # a declared AROM, DELO, PI or POLY token states the row but no order
    token = (edges.order_state == _KNOWN_DECLARED) & (edges.order > OrderToken.QUAD)
    edge_order = np.where(token, 0, edges.order).astype(np.uint8)
    edge_order_state = np.where(token, _UNKNOWN_DECLARED, edges.order_state).astype(np.uint8)
    for name, values in (
        ("bond_order", edge_order[g]),
        ("bond_order_state", edge_order_state[g]),
        ("bond_aromatic", edges.aromatic[g]),
        ("bond_aromatic_state", edges.aromatic_state[g]),
        ("bond_kind", edges.kind[g]),
        ("bond_kind_state", edges.kind_state[g]),
        ("bond_direction", np.where(swap, -edges.direction[g], edges.direction[g])),
        ("bond_direction_state", edges.direction_state[g]),
        ("bond_connection_type", connection_type[g]),
    ):
        c[name][merged] = values
    c["bond_origin"][position[m_ref:]] = edges.origin[new_internal]
    owner, code, side = _leaving(edges, g, swap, ends, definition, t)
    c["bond_declared_leaving_ptr"] = _ptr(np.bincount(merged[owner], minlength=m))
    by_edge = np.argsort(merged[owner], kind="stable")
    c["bond_declared_leaving"], c["bond_declared_leaving_side"] = code[by_edge], side[by_edge]

    # external connections: at the first endpoint (transform g), at the second (g⁻¹) unless both are one endpoint
    instance_of = _owner(layout.instance_occurrence_ptr)
    outward = np.flatnonzero(realised & ~identity[edges.transform])
    inward = outward[first_end[outward] != second_end[outward]]
    ext = np.concatenate((outward, inward))
    flip = np.arange(len(ext)) >= len(outward)
    here, there = np.where(flip, second_end[ext], first_end[ext]), np.where(flip, first_end[ext], second_end[ext])
    there_occ = ends.occurrence[there].astype(np.int64)
    # an edge to an endpoint outside the value (a built connection to an atom it does not hold) is one external
    # connection at its endpoint in the value: the identity, the partner as the outside endpoint names it
    away = ends.occurrence < 0
    cut = np.flatnonzero((away[first_end] | away[second_end]) & (np.maximum(ea, eb) >= 0))
    cut_flip = away[first_end[cut]]
    cut_here = np.where(cut_flip, second_end[cut], first_end[cut])
    cut_there = np.where(cut_flip, first_end[cut], second_end[cut])
    c["external_atom"] = endpoint_atom[np.concatenate((here, cut_here))]
    c["external_partner_instance"] = np.concatenate((instance_of[there_occ], ends.outside_instance[cut_there]))
    c["external_partner_component"] = np.concatenate(
        (there_occ - layout.instance_occurrence_ptr[instance_of[there_occ]], ends.outside_component[cut_there])
    )
    c["external_partner_atom"] = ends.name[np.concatenate((there, cut_there))]
    c["external_transform"] = np.concatenate(
        (
            np.where(flip, _inverse_rows(tr, edges.transform[ext[flip]], len(ext)), edges.transform[ext]),
            np.zeros(len(cut), dtype=np.int64),  # the identity is row 0
        )
    )
    n_ext = len(ext)
    ext, flip = np.concatenate((ext, cut)), np.concatenate((flip, cut_flip))
    for name, values in (
        ("order", edge_order),
        ("order_state", edge_order_state),
        ("aromatic", edges.aromatic),
        ("aromatic_state", edges.aromatic_state),
        ("kind", edges.kind),
        ("kind_state", edges.kind_state),
        ("direction_state", edges.direction_state),
        ("origin", edges.origin),
        ("connection_type", connection_type),
    ):
        c[f"external_{name}"] = values[ext]
    c["external_direction"] = np.where(flip, -edges.direction[ext], edges.direction[ext])
    owner, code, side = _leaving(edges, ext, flip, ends, definition, t)
    c["external_declared_leaving_ptr"] = _ptr(np.bincount(owner, minlength=len(ext)))
    c["external_declared_leaving"], c["external_declared_leaving_side"] = code, side
    edge_of_physical[outward] = -2 - np.arange(len(outward))
    edge_of_physical[cut] = -2 - (n_ext + np.arange(len(cut)))
    inward_row = np.full(n_edge, -1, dtype=np.int64)
    inward_row[inward] = len(outward) + np.arange(len(inward))
    inward_row[cut] = n_ext + np.arange(len(cut))  # its one external row, whichever end the centre is
    # the value holds the identity and the transforms its externals use, never a row only the source's codes named
    used = np.unique(np.concatenate(([0], c["external_transform"]))).astype(np.int64)
    renumber = np.zeros(len(tr.state), dtype=np.int64)
    renumber[used] = np.arange(len(used))
    c["external_transform"] = renumber[c["external_transform"]]
    c["transform_rotation"], c["transform_translation"] = tr.rotation[used], tr.translation[used]
    c["transform_code"], c["transform_state"] = tr.code[used], tr.state[used]

    # an endpoint with no atom (a port of a definition-less component, an atom not created): no edge
    unrealised_occ = np.zeros(n_occ, dtype=bool)
    externalised = np.zeros(n_edge, dtype=bool)
    externalised[cut] = True
    for k in np.flatnonzero(~realised & ~externalised).tolist():
        members = edges.declaration[edges.declaration_ptr[k] : edges.declaration_ptr[k + 1]].tolist()
        sources = sorted({int(r) for d in members for r in decl.ref[decl.ref_ptr[d] : decl.ref_ptr[d + 1]]})
        for e in (int(first_end[k]), int(second_end[k])):
            if away[e]:  # the endpoint outside the value has no occurrence to leave open
                continue
            unrealised_occ[ends.occurrence[e]] = True
            if endpoint_atom[e] < 0:
                log.append(
                    (
                        "edge_endpoint_unrealised",
                        SubjectKind.OCCURRENCE,
                        int(occ.ordinal[ends.occurrence[e]]),
                        sources,
                        (),
                    )
                )
    edge_of_declaration = np.full(len(decl.first), -1, dtype=np.int64)
    edge_of_declaration[edges.declaration] = edge_of_physical[edge_group]

    # ---- 2. components; 5. parity
    families = _components(c, inputs, definition, usable, layout, occurrence_atom, occ.atom_ptr, log)
    columns, clears = _stereo(t, inputs, result, definition, usable, occurrence_atom)
    folded = is_h & ~created & (fate != AtomFate.LEFT)  # hydrogens that are not atoms: implicit slots
    remapped, cleared = remap_parity(
        columns,
        occurrence_atom,
        removed_hydrogen=folded,
        substitutions=_substitutions(inputs.occurrences, result, endpoint_atom, edge_of_physical, inward_row),
        externalised=np.full(len(occurrence_atom), -1, dtype=np.int64),
        edge_map=edge_map,
        edges=old_edges,
    )
    remapped.write(c)
    log += clears + [
        (CODE_NAMES[reason], SubjectKind.ATOM, int(remapped.centre[r]), [], ())
        for r, reason in zip(cleared.centre.tolist(), cleared.reason.tolist())
    ]
    log += [  # a stereobond whose reference left with no replacement
        (CODE_NAMES[reason], SubjectKind.EDGE, int(remapped.bond_edge[r]), [], ())
        for r, reason in zip(cleared.bond.tolist(), cleared.bond_reason.tolist())
    ]

    # ---- 6. claims
    leaving = result.leaving
    owner_atom = _owner(component_ptr)
    heavy = c["atomic_number"] != _H
    atom_fate = c["atom_fate"]

    def touched(endpoints: NDArray) -> NDArray[np.bool_]:
        mask = np.zeros(n_occ, dtype=bool)
        occurrence = ends.occurrence[endpoints[endpoints >= 0]]
        mask[occurrence[occurrence >= 0]] = True  # an endpoint outside the value touches no occurrence
        return mask

    applied = np.zeros(len(decl.first), dtype=bool)
    applied[[a.declaration for a in inputs.assumptions if a.item == "existence"]] = True
    candidates = np.flatnonzero(~known(decl.existence) & ~applied)
    loose = (edges.kind_state == _UNKNOWN) & ~restated  # an unclassified connection
    unlinked = np.zeros(n_occ, dtype=bool)
    unlinked[layout.unlinked_pairs.ravel()] = True
    undetermined = _any(owner_atom, atom_fate == AtomFate.UNDETERMINED, n_occ)
    unobserved = _any(owner_atom, atom_fate == AtomFate.UNOBSERVED_ENDPOINT, n_occ)
    blocked = undetermined | unobserved | unlinked | unrealised_occ | (np.diff(layout.extra_ptr) > 0)
    blocked |= touched(leaving.endpoint[np.isin(leaving.state, _OPEN)])
    blocked |= touched(np.concatenate((decl.first[candidates], decl.second[candidates])))
    blocked |= touched(np.concatenate((first_end[loose], second_end[loose])))
    c["component_topology_complete"] = usable & ~blocked
    phantom = touched(leaving.endpoint[np.diff(leaving.phantom_ptr) > 0])
    heavy_undetermined = _any(owner_atom, (atom_fate == AtomFate.UNDETERMINED) & heavy, n_occ)
    c["component_heavy_atoms_complete"] = usable & ~(heavy_undetermined | phantom | unobserved)
    open_instance = layout.connections_open.copy()
    open_instance[instance_of[np.flatnonzero(unrealised_occ)]] = True
    c["instance_connections_complete"] = ~open_instance
    c["component_atom_ptr"] = component_ptr
    c["instance_component_ptr"] = layout.instance_occurrence_ptr
    c["instance_symbol"] = np.arange(n_inst)
    c["instance_entity"] = layout.instance_entity
    n_entity = len(layout.entity_symbol)
    c["system_entity_ptr"], c["system_instance_ptr"] = np.array([0, n_entity]), np.array([0, n_inst])
    c["entity_symbol"] = np.arange(n_entity)
    for name in ("entity_type", "entity_polymer_type", "entity_nstd_linkage"):
        c[name], c[f"{name}_state"] = getattr(layout, name), getattr(layout, f"{name}_state")

    families |= {
        "entity": layout.entity_symbol,
        "instance": layout.instance_symbol + tuple(symbols.get("outside_instance", ())),
        "component": symbols["component"],
        "atom": symbols["atom"],
        "connection_type": symbols.get("connection_type", ()),
        "polymer_type": symbols["polymer_type"],
    }
    graph = GraphArrays(
        SystemArrays(families, c),
        occurrence_atom,
        extra_atom,
        endpoint_atom,
        edge_of_physical,
        edge_of_declaration,
        Records.empty(),
    )
    records = Records.concat([graph_records(result.records, graph), Records.from_rows(log, inputs.refs)])
    return GraphArrays(
        graph.system, occurrence_atom, extra_atom, endpoint_atom, edge_of_physical, edge_of_declaration, records
    )


def graph_records(records: Records, graph: GraphArrays) -> Records:
    r"""
    Records of the attachment stage (subjects: ATOM = occurrence atom, EDGE = declaration row) re-indexed into the
    graph of ``graph``: an occurrence atom becomes its atom (-1 not created), a declaration row its edge or, when the
    edge became an external connection, (EXTERNAL, row); payload atoms follow their atoms.
    """

    moved = records.remap(SubjectKind.ATOM, graph.occurrence_atom)
    kind, subject = moved.subject_kind.copy(), moved.subject.copy()
    edge = (kind == SubjectKind.EDGE) & (subject >= 0)
    target = graph.edge_of_declaration[subject[edge]]
    kind[edge] = np.where(target <= -2, SubjectKind.EXTERNAL, SubjectKind.EDGE)
    subject[edge] = np.where(target <= -2, -2 - target, target)
    return Records(
        moved.code,
        kind,
        subject,
        moved.ref_ptr,
        moved.ref,
        moved.refs,
        moved.payload,
        moved.list_ptr,
        moved.list_value,
        moved.texts,
    )


def atom_elements(
    reference: NDArray[np.uint8], observed: NDArray[np.uint8], usable: NDArray[np.bool_]
) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    r"""
    Each occurrence atom's element and its state, decided once: the lowering writes it and io classifies an endpoint
    by it. The element a usable definition states (KNOWN×REFERENCE), else the one the atom's selected row states
    (KNOWN×DECLARED; ``observed`` 0 for an unplaced atom), else unknown: UNKNOWN×REFERENCE under a usable definition,
    UNKNOWN×DECLARED under an unusable one, which contributes only what was observed.
    """

    defined = usable & (reference > 0)
    element = np.where(defined, reference, observed).astype(np.uint8)
    states = np.select(
        [defined, observed > 0, usable], [_KNOWN_REFERENCE, _KNOWN_DECLARED, _UNKNOWN_REFERENCE], _UNKNOWN_DECLARED
    )
    return element, states.astype(np.uint8)


def _atoms(t, layout, result, occ, row, element, is_h, created, usable, occurrence_atom, extra_atom, n) -> dict:
    r"""The atom columns: definition atoms of usable occurrences with their reference facts (the element by
    ``atom_elements``) and the result's charge, valence and hydrogen decision; observed atoms (of unusable occurrences,
    and extras) with what was observed."""

    def column(dtype, fill=0) -> NDArray:
        return np.full(n, fill, dtype=dtype)

    kept, raw = created & usable, created & ~usable
    at_def, at_raw, extra = occurrence_atom[kept], occurrence_atom[raw], extra_atom
    c = {"atom_symbol": column(np.int32)}
    c["atom_symbol"][occurrence_atom[created]] = t.name[row[created]]
    c["atom_symbol"][extra] = layout.extra_name

    z, z_state = column(np.uint8), column(np.uint8, _UNKNOWN)
    decided, decided_state = atom_elements(element, layout.observed_element, usable)
    z[occurrence_atom[created]], z_state[occurrence_atom[created]] = decided[created], decided_state[created]
    z[extra] = layout.extra_element
    z_state[extra] = np.where(layout.extra_element > 0, _KNOWN_DECLARED, _UNKNOWN_DECLARED)
    c["atomic_number"], c["atomic_number_state"] = z, z_state

    # a mass number is a fact, the source's (D/T) over the definition's; natural abundance has none (0)
    isotope = column(np.uint16)
    declared, reference = layout.observed_isotope[kept], t.isotope[row[kept]]
    isotope[at_def] = np.where(declared > 0, declared, reference)
    isotope[np.concatenate((at_raw, extra))] = np.concatenate((layout.observed_isotope[raw], layout.extra_isotope))
    c["isotope"] = isotope

    charge, charge_state = column(np.int8), column(np.uint8, _UNKNOWN)
    charge[at_def] = np.where(known(result.charge_state), result.charge, 0)[kept]
    charge_state[at_def] = result.charge_state[kept]
    stated_charge = known(occ.charge_state)
    charge[at_raw] = np.where(stated_charge, occ.charge, 0)[raw]
    charge_state[at_raw] = np.where(stated_charge, occ.charge_state, _UNKNOWN)[raw]
    extra_known = known(layout.extra_charge_state)
    charge[extra] = np.where(extra_known, layout.extra_charge, 0)
    charge_state[extra] = np.where(extra_known, layout.extra_charge_state, _UNKNOWN)
    c["formal_charge"], c["formal_charge_state"] = charge, charge_state

    # hydrogens a charge change adds join a materialised count; observed-only and undetermined counts are unknown
    added = result.implicit_added.astype(np.int64)
    materialise = result.hydrogens == Hydrogens.MATERIALISE
    counted = np.select(
        [is_h, materialise & (added > 0), materialise],
        [_KNOWN_REFERENCE, _KNOWN_DERIVED, _KNOWN_REFERENCE],
        _UNKNOWN_DERIVED,
    )
    c["implicit_hydrogens"] = column(np.int8)
    c["implicit_hydrogens"][at_def] = np.where(materialise, added, 0)[kept]
    c["implicit_hydrogens_state"] = column(np.uint8, _UNKNOWN)
    c["implicit_hydrogens_state"][at_def] = counted[kept]
    c["atom_valence"], c["atom_valence_state"] = column(np.int8), column(np.uint8, _UNKNOWN)
    c["atom_valence"][at_def], c["atom_valence_state"][at_def] = result.valence[kept], result.valence_state[kept]
    c["atom_aromatic"], c["atom_aromatic_state"] = column(bool), column(np.uint8, _UNKNOWN)
    c["atom_aromatic"][at_def], c["atom_aromatic_state"][at_def] = t.aromatic[row[kept]], t.aromatic_state[row[kept]]
    c["atom_fate"] = column(np.uint8, AtomFate.RETAINED)
    c["atom_fate"][occurrence_atom[created]] = result.fate[created]
    c["atom_endpoint_state"] = column(np.uint8)
    c["atom_designation"] = column(np.uint8)
    c["atom_designation"][at_def] = t.designation[row[kept]]
    c["atom_reference_row"] = column(np.int32, -1)
    return c


def _reference_bonds(t: _Tables, bond_row: NDArray[np.int64], at: NDArray[np.int64], m: int) -> dict:
    r"""The edge columns, with the definition's bond facts at the reference edges (origin REFERENCE; a covalent bond
    has no donor direction)."""
    c = {
        "bond_order": np.zeros(m, np.uint8),
        "bond_order_state": np.full(m, _UNKNOWN, np.uint8),
        "bond_aromatic": np.zeros(m, bool),
        "bond_aromatic_state": np.full(m, _UNKNOWN, np.uint8),
        "bond_kind": np.zeros(m, np.uint8),
        "bond_kind_state": np.full(m, _UNKNOWN, np.uint8),
        "bond_direction": np.zeros(m, np.int8),
        "bond_direction_state": np.full(m, _UNKNOWN, np.uint8),
        "bond_origin": np.zeros(m, np.uint8),
        "bond_connection_type": np.full(m, -1, np.int32),
    }
    c["bond_order"][at], c["bond_order_state"][at] = t.bond_order[bond_row], t.bond_order_state[bond_row]
    c["bond_aromatic"][at], c["bond_aromatic_state"][at] = t.bond_aromatic[bond_row], t.bond_aromatic_state[bond_row]
    c["bond_kind"][at], c["bond_kind_state"][at] = BondKind.COVALENT, _KNOWN_REFERENCE
    c["bond_direction_state"][at] = _INAPPLICABLE_DEFINED
    c["bond_origin"][at] = BondOrigin.REFERENCE
    return c


def _leaving(edges: MergedEdges, groups, flip, ends, definition, t) -> tuple[NDArray, NDArray, NDArray]:
    r"""The declared leaving ids of the merged edges ``groups``: per entry its position in ``groups``, its atom-name
    code and its side (0 = the edge's first atom, or the external atom; ``flip`` swaps an edge's sides); an id that
    names no atom of its side's reference is not a name the value can hold (its Conflict record names it)."""
    lengths = np.diff(edges.declared_leaving_ptr)[groups]
    owner = np.repeat(np.arange(len(groups)), lengths)
    entries = np.repeat(edges.declared_leaving_ptr[:-1][groups], lengths) + _local(_ptr(lengths))
    atom, side = edges.declared_leaving[entries].astype(np.int64), edges.declared_leaving_side[entries].astype(np.int64)
    endpoint = np.where(side == 0, edges.first[groups[owner]], edges.second[groups[owner]])
    occurrence = ends.occurrence[endpoint]
    d = np.where(occurrence >= 0, definition[occurrence], -1)  # an endpoint outside the value has no reference
    keep = (atom >= 0) & (d >= 0)
    code = t.name[t.atom[d[keep]] + atom[keep]]
    sides = np.where(flip[owner], 1 - side, side)[keep]
    order = np.lexsort((code, sides, owner[keep]))  # by side then name, whichever row represents the edge
    return owner[keep][order], code[order].astype(np.int32), sides[order].astype(np.uint8)


def _inverse_rows(transforms: Transforms, rows: NDArray[np.int64], count: int) -> NDArray[np.int64]:
    r"""The row of the inverse of each given transform row, placed at the last ``len(rows)`` of ``count`` positions: the
    row itself when it is not KNOWN (zero payload). io puts the inverse of every KNOWN relative transform into the table
    (``io.symmetry.resolve_transforms``); a recipe has the identity only."""
    out = np.zeros(count, dtype=np.int64)
    inverse = np.asarray(rows, dtype=np.int64).copy()
    stated = np.flatnonzero(known(transforms.state[inverse]))
    r = transforms.rotation[inverse[stated]].astype(np.int64)
    adjugate = np.stack([np.cross(r[:, 1], r[:, 2]), np.cross(r[:, 2], r[:, 0]), np.cross(r[:, 0], r[:, 1])], axis=2)
    rotation = adjugate * np.einsum("ij,ij->i", r[:, 0], np.cross(r[:, 1], r[:, 2]))[:, None, None]
    translation = -np.einsum("tij,tj->ti", rotation, transforms.translation[inverse[stated]].astype(np.int64))
    table = np.flatnonzero(known(transforms.state))
    index = {(transforms.rotation[k].tobytes(), transforms.translation[k].tobytes()): k for k in table[::-1].tolist()}
    for k, at in enumerate(stated.tolist()):
        inverse[at] = index[(rotation[k].astype(np.int8).tobytes(), translation[k].astype(np.int16).tobytes())]
    out[count - len(inverse) :] = inverse
    return out


def _components(c, inputs, definition, usable, layout, occurrence_atom, atom_ptr, log) -> dict[str, tuple[str, ...]]:
    r"""The component columns: definition scope and digest, monomer type with its state, the verified ports, the
    definition's parents and the occurrence's own annotation.
    Returns the symbol families these columns code into."""

    defs = inputs.definitions
    n_occ, count = len(definition), len(defs)
    stated_parent = [layout.symbols["parent"][p] for p in layout.declared_parent.tolist()]
    reference_parents = [
        sorted(set(d.parents)) if d is not None and known(np.uint8(d.parents_state)) else [] for d in defs
    ]
    parents = sorted({p for listed in reference_parents for p in listed} | set(stated_parent))
    parent_code = {p: k for k, p in enumerate(parents)}

    per = {name: np.zeros(count + 1, dtype) for name, dtype in (("scope", np.uint8), ("digest", np.int64))}
    per["monomer_state"] = np.full(count + 1, _UNKNOWN, np.uint8)
    per["monomer"] = np.zeros(count + 1, np.uint8)
    port = np.full((count + 1, 2), -1, dtype=np.int64)
    for k, d in enumerate(defs):
        if d is None:
            continue
        per["scope"][k], per["digest"][k] = d.scope, d.digest
        typed = known(np.uint8(d.chem_comp_type_state))
        if typed and (family := FAMILY_TABLE.get(d.chem_comp_type.strip().upper())) is not None:
            per["monomer"][k], per["monomer_state"][k] = family[1], _KNOWN_DEFINED
        record = ports(d)
        port[k] = np.where(np.array(record.status) == PortStatus.VERIFIED, (record.previous, record.next), -1)
    pick = np.where(usable, definition, count)
    c["component_symbol"] = layout.occurrence_component
    c["component_definition_scope"], c["component_definition_digest"] = per["scope"][pick], per["digest"][pick]
    c["component_monomer_type"], c["component_monomer_type_state"] = per["monomer"][pick], per["monomer_state"][pick]
    ported = port[pick] >= 0
    index = np.where(ported, atom_ptr[:-1][:, None] + port[pick], 0)
    c["component_port"] = np.where(ported, np.append(occurrence_atom, -1)[np.where(ported, index, -1)], -1)

    # parents: the definition's own → component_definition_parent; the occurrence's own
    # _pdbx_struct_mod_residue / MODRES annotation → component_parent, empty where the source annotates none (never
    # filled from the definition; an annotation equal to the definition's stays). Two scopes, never a Conflict: an
    # annotation differing as a set from the parents the definition lists → finding ``parent_annotation_mismatch``
    # naming both statements and both sets; a definition listing none (null, absent or Conflict) states no parent set
    # to differ from
    segments = [[parent_code[p] for p in listed] for listed in reference_parents] + [[]]
    flat = np.array([p for s in segments for p in s], dtype=np.int32)
    start = _ptr([len(s) for s in segments])
    lengths = np.diff(start)[pick]
    pointer = _ptr(lengths)
    values = flat[np.repeat(start[pick], lengths) + _local(pointer)]
    c["component_definition_parent_ptr"], c["component_definition_parent"] = pointer, values

    declared, defined = layout.declared_parent_ptr, inputs.definition_ref_ptr
    annotated = np.flatnonzero(np.diff(declared) > 0)
    annotation = {}
    for o in annotated.tolist():
        stated = sorted({stated_parent[k] for k in range(declared[o], declared[o + 1])})
        listed = reference_parents[definition[o]] if usable[o] else []
        if listed and stated != listed:  # the annotation's rows, then the definition's statements
            rows = layout.declared_parent_ref[declared[o] : declared[o + 1]].tolist()
            rows += inputs.definition_ref[defined[definition[o]] : defined[definition[o] + 1]].tolist()
            sets = (",".join(stated), ",".join(listed))
            log.append(("parent_annotation_mismatch", SubjectKind.COMPONENT, o, rows, sets))
        annotation[o] = [parent_code[p] for p in stated]
    slots = [annotation.get(o, []) for o in range(n_occ)]
    c["component_parent_ptr"] = _ptr([len(s) for s in slots])
    c["component_parent"] = np.array([p for s in slots for p in s], dtype=np.int32)
    return {"parent": tuple(parents)}


def _stereo(t: _Tables, inputs, result, definition, usable, occurrence_atom) -> tuple[StereoColumns, list[tuple]]:
    r"""The definition stereo rows of every usable occurrence in occurrence-atom indices (state REFERENCE, DECLARED for
    SMILES tags, UNKNOWN×REFERENCE for a refused or undetermined parity), with the plan's UNDETERMINED_NONPORT and
    CLEAR rows already cleared (with their records); bond rows over the old reference bonds."""

    occ = inputs.occurrences
    centre_ptr = _ptr(np.where(usable, np.diff(t.centre)[definition], 0))
    centre_occ = _owner(centre_ptr)
    centre_row = t.centre[definition[centre_occ]] + _local(centre_ptr)
    base = occ.atom_ptr[centre_occ]
    centre = base + t.centre_atom[centre_row]
    slots = np.where(t.slots[centre_row] >= 0, base[:, None] + t.slots[centre_row], -1)
    parity = t.parity[centre_row].copy()
    stereo_basis = t.stereo_basis[centre_row]
    determined = np.isin(parity, (1, 2)) & (stereo_basis != StereoBasis.REFUSED)
    centre_state = np.where(stereo_basis == StereoBasis.STATED, _KNOWN_DECLARED, _KNOWN_REFERENCE)
    centre_state = np.where(determined, centre_state, _UNKNOWN_REFERENCE).astype(np.uint8)

    plan = result.stereo
    plan_centre = occ.atom_ptr[plan.occurrence] + plan.centre.astype(np.int64)
    order = np.argsort(centre)
    plan_row = order[np.searchsorted(centre[order], plan_centre)] if len(plan_centre) else np.zeros(0, np.int64)
    clearing = np.isin(plan.action, (StereoAction.UNDETERMINED_NONPORT, StereoAction.CLEAR)) & determined[plan_row]
    determined[plan_row[clearing]] = False
    centre_state[plan_row[clearing]] = _UNKNOWN_DERIVED
    parity[~determined], slots[~determined] = 0, -1
    clears: list[tuple[str, SubjectKind, int, list[int], tuple[()]]] = [
        (CODE_NAMES[reason], SubjectKind.ATOM, int(occurrence_atom[at]), [], ())
        for at, reason in zip(plan_centre[clearing].tolist(), plan.reason[clearing].tolist())
    ]

    # AND/OR groups of the occurrences that state them; a row that is not determined leaves its group
    kinds, sizes, members = [], [], []
    for o in np.flatnonzero(usable & (np.diff(t.group)[definition] > 0)).tolist():
        d = definition[o]
        start = 0
        for g in range(t.group[d], t.group[d + 1]):
            rows = t.group_member[t.member[d] + start : t.member[d] + start + t.group_size[g]] + centre_ptr[o]
            start += int(t.group_size[g])
            rows = rows[determined[rows]]
            if len(rows):
                kinds.append(t.group_kind[g])
                sizes.append(len(rows))
                members += rows.tolist()

    bond_stereo_ptr = _ptr(np.where(usable, np.diff(t.bond_stereo)[definition], 0))
    bond_occ = _owner(bond_stereo_ptr)
    bond_stereo_row = t.bond_stereo[definition[bond_occ]] + _local(bond_stereo_ptr)
    value = t.bond_stereo_value[bond_stereo_row]
    bond_known = np.isin(value, (1, 2))
    bond_state = np.where(t.bond_stereo_stated[bond_stereo_row], _KNOWN_DECLARED, _KNOWN_REFERENCE)
    bond_base = occ.atom_ptr[bond_occ]
    # the old bond of a row: its occurrence's bond rows come in definition order (see realize_graph's bond_ptr)
    occurrence_bonds = _ptr(np.where(usable, np.diff(t.bond)[definition], 0))
    columns = StereoColumns(
        centre=centre.astype(np.int32),
        parity=parity.astype(np.int8),
        neighbors=slots.astype(np.int32),
        centre_state=centre_state,
        bond_edge=(occurrence_bonds[bond_occ] + t.bond_stereo_bond[bond_stereo_row]).astype(np.int32),
        bond_value=np.where(bond_known, value, 0).astype(np.int8),
        bond_reference=np.where(
            bond_known[:, None], bond_base[:, None] + t.bond_stereo_reference[bond_stereo_row], -1
        ).astype(np.int32),
        bond_state=np.where(bond_known, bond_state, _UNKNOWN_REFERENCE).astype(np.uint8),
        group_kind=np.array(kinds, dtype=np.uint8),
        group_ptr=_ptr(sizes),
        group_member=np.array(members, dtype=np.int32),
    )
    return columns, clears


def _substitutions(occ, result, endpoint_atom, edge_of_physical, inward_row) -> NDArray[np.int64]:
    r"""The substitutions of the TRANSPORT rows: ``(centre, departed, partner)`` in occurrence-atom
    indices for the centre and the departed neighbour, the partner as a graph atom or, when it is reached through an
    external connection at the centre, as that external row ``-(x + 2)``."""

    edges, plan = result.edges, result.stereo
    substitutions = []
    for k in np.flatnonzero(plan.action == StereoAction.TRANSPORT).tolist():
        g, partner = int(plan.partner_edge[k]), int(plan.partner_endpoint[k])
        target = int(edge_of_physical[g])
        if target >= 0:
            code = int(endpoint_atom[partner])
        elif target <= -2:  # the centre sits at the other endpoint; its external row carries the partner
            at_first = int(edges.second[g]) == partner or int(edges.first[g]) == int(edges.second[g])
            code = -(((-2 - target) if at_first else int(inward_row[g])) + 2)
        else:
            continue
        base = int(occ.atom_ptr[plan.occurrence[k]])
        substitutions.append((base + int(plan.centre[k]), base + int(plan.departed[k]), code))
    return np.array(substitutions, dtype=np.int64).reshape(-1, 3)


@dataclass(frozen=True)
class ValenceExcess:
    r"""Per atom that is not UNDETERMINED: realised covalent valence − ``atom_valence`` and its state."""

    atom: NDArray[np.int64]
    excess: NDArray[np.int8]
    state: NDArray[np.uint8]


def valence_excess(system: MolecularSystem) -> ValenceExcess:
    r"""
    The valence excess of every atom that is not UNDETERMINED: realised covalent valence = Σ KNOWN
    orders of its covalent edges and covalent external connections (a bond to a graph hydrogen is one of them) + its
    KNOWN ``implicit_hydrogens``; excess = realised − ``atom_valence``. KNOWN×DERIVED only when the atom's valence and
    hydrogen count, and the kind (and, if covalent, the order) of every incident connection are KNOWN and no neighbour
    is UNDETERMINED (consumers computing valence exclude it); otherwise UNKNOWN×DERIVED with payload 0. Only a
    KNOWN positive excess is a contradiction (``prepare_valence_excess``).
    """

    undetermined = system.atom_fate == AtomFate.UNDETERMINED
    realised = np.where(known(system.implicit_hydrogens_state), system.implicit_hydrogens, 0).astype(np.int64)
    determined = known(system.implicit_hydrogens_state) & known(system.atom_valence_state)

    first, second = system.edge_index.astype(np.int64)
    for atom, partner, kind, kind_state, order, order_state in (
        (first, second, system.bond_kind, system.bond_kind_state, system.bond_order, system.bond_order_state),
        (second, first, system.bond_kind, system.bond_kind_state, system.bond_order, system.bond_order_state),
        (
            system.external_atom.astype(np.int64),
            np.full(len(system.external_atom), -1, dtype=np.int64),
            system.external_kind,
            system.external_kind_state,
            system.external_order,
            system.external_order_state,
        ),
    ):
        covalent = known(kind_state) & (kind == BondKind.COVALENT)
        counted = covalent & known(order_state)
        np.add.at(realised, atom[counted], order[counted].astype(np.int64))
        blocked = ~known(kind_state) | (covalent & ~known(order_state))
        blocked |= np.append(undetermined, False)[partner]
        determined[atom[blocked]] = False

    atoms = np.flatnonzero(~undetermined)
    excess = np.where(determined, realised - system.atom_valence, 0)[atoms]
    states = np.where(determined[atoms], _KNOWN_DERIVED, _UNKNOWN_DERIVED).astype(np.uint8)
    return ValenceExcess(atoms.astype(np.int64), excess.astype(np.int8), states)
