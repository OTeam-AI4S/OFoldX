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

r"""Select molecular views while retaining identity, connectivity, and stereochemistry.

Atom and instance projections update index maps and narrow completeness claims as
needed. A connection to an omitted atom of another component becomes an external
connection, identified by instance, component ordinal, and atom name, with an identity
transform. An omitted hydrogen is folded into the retained neighbor's count.

Component ordinals remain stable within each retained instance. Atom selection keeps
all components of that instance, including components left without atoms; an instance
is removed when none of its atoms remains. Parity transport records stereocenters that
become undetermined. `select` combines these operations for `Structure.select`, while
`project_arrays` gathers their build buffers and those used by molecular copies.
"""

from __future__ import annotations

from collections.abc import MutableMapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypeVar

import numpy as np
from numpy.typing import NDArray

from ..errors import CODE_NAMES, RequestRef, fail
from ..provenance import Basis, Resolution, known, state
from .molecular import EntityType, MolecularSystem, SystemArrays
from .schema import check_mask
from .stereo import ParityClears, StereoColumns, remap_parity

if TYPE_CHECKING:
    from .structure import Structure, StructureArrays

V = TypeVar("V", MolecularSystem, "Structure")
_FACTS = ("order", "aromatic", "kind", "direction")
_TWO_VIRTUAL = CODE_NAMES.index("parity_cleared_two_virtual_slots")
_KNOWN_DERIVED, _UNKNOWN_DERIVED = state(Resolution.KNOWN, Basis.DERIVED), state(Resolution.UNKNOWN, Basis.DERIVED)


@dataclass(frozen=True)
class ProjectionMap:
    r"""Old → new indices (``-1`` removed) and the parity rows the projection cleared."""

    atom: NDArray[np.int64]
    component: NDArray[np.int64]
    instance: NDArray[np.int64]
    external: NDArray[np.int64]
    edge: NDArray[np.int64]
    parity_clears: ParityClears


def _owner(pointer: NDArray[np.int64]) -> NDArray[np.int64]:
    return np.repeat(np.arange(len(pointer) - 1, dtype=np.int64), np.diff(pointer))


def _pointer(counts: NDArray[np.int64]) -> NDArray[np.int64]:
    return np.concatenate(([0], np.cumsum(counts))).astype(np.int64)


def _ragged(pointer: NDArray[np.int64], rows: NDArray[np.int64]) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    r"""The pointer of segments ``rows`` (in that order) and the old index of every value they hold."""
    counts = pointer[rows + 1] - pointer[rows]
    new = _pointer(counts)
    return new, np.repeat(pointer[rows] - new[:-1], counts) + np.arange(new[-1], dtype=np.int64)


def _inverse(rows: NDArray[np.int64], size: int) -> NDArray[np.int64]:
    out = np.full(size, -1, dtype=np.int64)
    out[rows] = np.arange(len(rows))
    return out


def _take(columns: MutableMapping[str, NDArray], schema, axis: str, rows: NDArray[np.int64]) -> None:
    for field in schema.fields:
        if field.axis == axis:
            columns[field.name] = np.take(columns[field.name], rows, axis=len(field.leading))


def _hydrogen(columns) -> NDArray[np.bool_]:
    return known(columns["atomic_number_state"]) & (columns["atomic_number"] == 1)


def project_arrays(
    system: SystemArrays,
    structure: StructureArrays | None,
    instances: NDArray[np.int64],
    atoms: NDArray[np.bool_],
    *,
    externalise: bool = True,
) -> tuple[SystemArrays, StructureArrays | None, ProjectionMap]:
    r"""Gather selected instances and atoms into new molecular build buffers.

    Args:
        system: Source molecular-system buffers.
        structure: Source coordinate buffers, or None for a coordinate-free system.
        instances: Instance rows in the requested output order. Instances of one
            system must remain contiguous.
        atoms: Boolean selection over the source atom axis.
        externalise: Whether an edge to an omitted atom of another component becomes
            an external connection. False drops those edges.

    Returns:
        New system buffers, optional structure buffers, and an old-to-new index map.
        The supplied buffers remain unchanged.

    Notes:
        Retained atoms keep their facts, including caller-maintained hydrogen counts.
        Edges between retained atoms are reordered with their direction and leaving
        sides. Edges to omitted atoms of the same component are dropped. External
        connections use the identity transform.

        Parity remapping represents a removed hydrogen as an implicit neighbor slot
        and a removed external neighbor as an external slot. For a neighbor connected
        to several retained atoms, the slot identifies the stereocenter's cut.
        Entities, transforms, and system rows remain; unused conformer rows are removed.
    """

    c0 = system.columns
    schema = MolecularSystem.SCHEMA
    instances = np.asarray(instances, dtype=np.int64)
    instance_system = _owner(c0["system_instance_ptr"])
    atom_component, component_instance = _owner(c0["component_atom_ptr"]), _owner(c0["instance_component_ptr"])
    n = len(atom_component)
    _, rows_component = _ragged(c0["instance_component_ptr"], instances)
    _, candidates = _ragged(c0["component_atom_ptr"], rows_component)
    rows_atom = candidates[np.asarray(atoms, dtype=bool)[candidates]]
    atom_map, kept = _inverse(rows_atom, n), np.zeros(n, dtype=bool)
    kept[rows_atom] = True
    c = dict(c0)
    c["system_instance_ptr"] = _pointer(
        np.bincount(instance_system[instances], minlength=len(c0["system_instance_ptr"]) - 1)
    )
    c["instance_component_ptr"] = _pointer(np.diff(c0["instance_component_ptr"])[instances])
    c["component_atom_ptr"] = _pointer(
        np.bincount(atom_component[rows_atom], minlength=len(component_instance))[rows_component]
    )
    for name in ("component_parent", "component_definition_parent"):
        c[f"{name}_ptr"], parents = _ragged(c0[f"{name}_ptr"], rows_component)
        c[name] = c0[name][parents]
    _take(c, schema, "instance", instances)
    _take(c, schema, "component", rows_component)
    c["component_port"] = np.append(atom_map, -1)[c["component_port"]]  # a port -1 (none) reads the last, -1
    _take(c, schema, "atom", rows_atom)
    referenced = np.unique(c["atom_reference_row"][c["atom_reference_row"] >= 0])
    c["atom_reference_row"] = np.append(_inverse(referenced, len(c0["reference_pool_source"])), -1)[
        c["atom_reference_row"]
    ]
    _take(c, schema, "reference", referenced)
    # edges: kept (re-sorted), dropped inside a component, or cut across components into external connections
    first, second = (end.astype(np.int64) for end in c0["edge_index"])
    rows_edge = np.flatnonzero(kept[first] & kept[second])
    ends = np.stack((atom_map[first[rows_edge]], atom_map[second[rows_edge]]))
    order = np.lexsort((ends.max(axis=0), ends.min(axis=0)))
    rows_edge, ends = rows_edge[order], ends[:, order]
    swapped = ends[0] > ends[1]
    edge_map = _inverse(rows_edge, len(first))
    _take(c, schema, "edge", rows_edge)
    c["edge_index"] = np.sort(ends, axis=0)
    c["bond_direction"] = np.where(swapped, -c["bond_direction"], c["bond_direction"])
    c["bond_declared_leaving_ptr"], leaving = _ragged(c0["bond_declared_leaving_ptr"], rows_edge)
    c["bond_declared_leaving"] = c0["bond_declared_leaving"][leaving]
    sides = c0["bond_declared_leaving_side"][leaving]
    c["bond_declared_leaving_side"] = np.where(
        np.repeat(swapped, np.diff(c["bond_declared_leaving_ptr"])), 1 - sides, sides
    )
    cross = atom_component[first] != atom_component[second]
    cut = np.flatnonzero((kept[first] ^ kept[second]) & cross & externalise)
    flip = kept[second[cut]]
    kept_end, gone = np.where(flip, second[cut], first[cut]), np.where(flip, first[cut], second[cut])
    partner_instance = component_instance[atom_component[gone]]
    new = {
        "external_atom": atom_map[kept_end],
        "external_partner_instance": c0["instance_symbol"][partner_instance],
        "external_partner_component": atom_component[gone] - c0["instance_component_ptr"][partner_instance],
        "external_partner_atom": c0["atom_symbol"][gone],
        "external_transform": np.zeros(len(cut), dtype=np.int64),
        "external_origin": c0["bond_origin"][cut],
        "external_connection_type": c0["bond_connection_type"][cut],
    }
    for fact in _FACTS:
        new[f"external_{fact}"] = c0[f"bond_{fact}"][cut]
        new[f"external_{fact}_state"] = c0[f"bond_{fact}_state"][cut]
    new["external_direction"] = np.where(flip, -1, 1) * new["external_direction"]
    present = np.flatnonzero(kept[c0["external_atom"]])
    pointer, leaving = _ragged(c0["external_declared_leaving_ptr"], present)
    cut_pointer, cut_leaving = _ragged(c0["bond_declared_leaving_ptr"], cut)
    c["external_declared_leaving_ptr"] = np.concatenate((pointer, pointer[-1] + cut_pointer[1:]))
    cut_sides = c0["bond_declared_leaving_side"][cut_leaving]
    cut_sides = np.where(np.repeat(flip, np.diff(cut_pointer)), 1 - cut_sides, cut_sides)
    c["external_declared_leaving"] = np.concatenate(
        (c0["external_declared_leaving"][leaving], c0["bond_declared_leaving"][cut_leaving])
    )
    c["external_declared_leaving_side"] = np.concatenate((c0["external_declared_leaving_side"][leaving], cut_sides))
    _take(c, schema, "external", present)
    c["external_atom"] = atom_map[c["external_atom"]]
    for name, values in new.items():
        c[name] = np.concatenate((c[name], values.astype(c[name].dtype)))
    stereo = StereoColumns.read(c0)
    centres = np.isin(kept_end, stereo.centre[known(stereo.centre_state)])
    order = np.argsort(~centres, kind="stable")  # a removed atom cut from a stereocentre names that centre's row
    gone_atoms, first_row = np.unique(gone[order], return_index=True)
    externalised = np.full(n, -1, dtype=np.int64)
    externalised[gone_atoms] = len(present) + order[first_row]
    external_map = np.full(len(c0["external_atom"]), -1, dtype=np.int64)
    external_map[present] = -np.arange(len(present)) - 2
    remapped, clears = remap_parity(
        stereo,
        atom_map,
        removed_hydrogen=_hydrogen(c0) & ~kept,
        substitutions=np.zeros((0, 3), dtype=np.int64),
        externalised=externalised,
        edge_map=edge_map,
        external_map=external_map,
    )
    remapped.write(c)
    projected = None if structure is None else structure.gather(rows_atom, rows_component, instances)
    component_map = _inverse(rows_component, len(component_instance))
    projection = ProjectionMap(
        atom_map,
        component_map,
        _inverse(instances, len(c0["instance_symbol"])),
        _inverse(present, len(external_map)),
        edge_map,
        clears,
    )
    return SystemArrays(system.symbols, c), projected, projection


def _fold_counts(
    c: MutableMapping[str, NDArray], folded: NDArray[np.bool_], receiving: NDArray[np.bool_]
) -> NDArray[np.bool_]:
    r"""
    Add the hydrogens ``folded`` to the count of the atom each ``receiving`` edge joins them to, in place; returns the
    atoms whose fold is not exact.

    A count becomes KNOWN×DERIVED when it was known and every hydrogen it receives had exactly that one neighbour and no
    stated isotope (a count cannot carry a mass number), else UNKNOWN×DERIVED.
    """

    n = len(c["atom_symbol"])
    first, second = (end.astype(np.int64) for end in c["edge_index"])
    from_first, from_second = receiving & folded[first] & ~folded[second], receiving & folded[second] & ~folded[first]
    h_end = np.concatenate((first[from_first], second[from_second]))
    heavy_end = np.concatenate((second[from_first], first[from_second]))
    degree = np.bincount(first, minlength=n) + np.bincount(second, minlength=n)
    degree += np.bincount(c["external_atom"].astype(np.int64), minlength=n)
    exact_h = folded & (degree == 1) & (c["isotope"] == 0)
    receives = np.bincount(heavy_end, minlength=n) > 0
    inexact = np.zeros(n, dtype=bool)
    inexact[heavy_end[~exact_h[h_end]]] = True
    determined = receives & known(c["implicit_hydrogens_state"]) & ~inexact
    counts = c["implicit_hydrogens"].astype(np.int64) + np.bincount(heavy_end, minlength=n)
    c["implicit_hydrogens"] = np.where(receives, np.where(determined, counts, 0), c["implicit_hydrogens"]).astype(
        np.int8
    )
    c["implicit_hydrogens_state"] = np.where(
        receives, np.where(determined, _KNOWN_DERIVED, _UNKNOWN_DERIVED), c["implicit_hydrogens_state"]
    ).astype(np.uint8)
    return (folded & ~exact_h) | (receives & ~determined)


def _finish(value, system: SystemArrays, structure):
    built = MolecularSystem.from_arrays(system)
    return built if structure is None else type(value).from_arrays(built, structure)


def _parts(value) -> tuple[MolecularSystem, SystemArrays, StructureArrays | None]:
    system = value if isinstance(value, MolecularSystem) else value.system
    return system, system.build(), None if value is system else value.build()


def _instances_with(c, atom_keep: NDArray[np.bool_]) -> NDArray[np.int64]:
    r"""Instances that keep an atom, and instances without atoms (nothing of them is removed)."""
    instance_of_atom = _owner(c["instance_component_ptr"])[_owner(c["component_atom_ptr"])]
    count = len(c["instance_symbol"])
    total = np.bincount(instance_of_atom, minlength=count)
    return np.flatnonzero((np.bincount(instance_of_atom[atom_keep], minlength=count) > 0) | (total == 0))


def project_atoms(value: V, keep: NDArray[np.bool_]) -> tuple[V, ProjectionMap]:
    r"""
    The atoms ``keep``: every fact of a kept atom is kept but its hydrogen count, which takes the removed hydrogens of
    its own component (the ``fold_hydrogens`` count rule); an edge to a removed atom of the same component is dropped,
    an edge to a removed atom of another component becomes an external connection; every component that loses an atom
    has its topology and heavy-atom claims False. Parity: a removed hydrogen becomes the implicit slot (two → cleared),
    a removed heavy neighbour of the same component clears the row, a neighbour across the cut becomes an external slot.
    """

    _, arrays, structure = _parts(value)
    c = arrays.columns
    keep = check_mask("keep", keep, len(c["atom_symbol"]), code="projection_invalid")
    atom_component = _owner(c["component_atom_ptr"])
    first, second = (end.astype(np.int64) for end in c["edge_index"])
    removed_h = _hydrogen(c) & ~keep
    _fold_counts(c, removed_h, (keep[first] ^ keep[second]) & (atom_component[first] == atom_component[second]))
    touched = np.bincount(atom_component[~keep], minlength=len(c["component_symbol"])) > 0
    for claim in ("topology", "heavy_atoms"):
        c[f"component_{claim}_complete"] = c[f"component_{claim}_complete"] & ~touched
    arrays, projected, projection = project_arrays(arrays, structure, _instances_with(c, keep), keep)
    return _finish(value, arrays, projected), projection


def project_instances(value: V, keep: NDArray[np.bool_]) -> tuple[V, ProjectionMap]:
    r"""
    The instances ``keep``, whole: every edge to a removed instance becomes an external connection of the kept side and
    its parity slot an external slot; no claim narrows.
    """

    system, arrays, structure = _parts(value)
    keep = check_mask("keep", keep, len(system.instance_symbol), code="projection_invalid")
    atoms = np.ones(len(system.atom_symbol), dtype=bool)
    arrays, projected, projection = project_arrays(arrays, structure, np.flatnonzero(keep), atoms)
    return _finish(value, arrays, projected), projection


def fold_hydrogens(value: V) -> tuple[V, ProjectionMap]:
    r"""
    Fold every hydrogen atom into the implicit count of its neighbour.

    A count becomes KNOWN×DERIVED when the old count was known and every hydrogen bonded to the atom had exactly that
    one neighbour and no stated isotope, else UNKNOWN×DERIVED; a component keeps its topology claim only when its fold
    is exact (no heavy atom is removed, so the heavy-atom claim stays). A stored configuration the fold would lose (a
    second implicit slot at one centre, or a cis/trans reference hydrogen whose double-bond atom has no other heavy
    neighbour) is ``UnsupportedError(hydrogen_view_loses_stereo)``; a reference hydrogen with another heavy neighbour on
    the same atom is re-pointed to it and the value flipped.
    """

    _, arrays, structure = _parts(value)
    c = arrays.columns
    hydrogen = _hydrogen(c)
    first, second = (end.astype(np.int64) for end in c["edge_index"])
    lossy = _fold_counts(c, hydrogen, hydrogen[first] ^ hydrogen[second])
    exact = np.bincount(_owner(c["component_atom_ptr"])[lossy], minlength=len(c["component_symbol"])) == 0
    c["component_topology_complete"] = c["component_topology_complete"] & exact
    _repoint(c, hydrogen, first, second)
    arrays, projected, projection = project_arrays(
        arrays, structure, _instances_with(c, ~hydrogen), ~hydrogen, externalise=False
    )
    two = projection.parity_clears.reason == _TWO_VIRTUAL
    if two.any():
        row = int(projection.parity_clears.centre[two][0])
        fail("hydrogen_view_loses_stereo", RequestRef("stereocentre", str(row)), rule="two_virtual_slots")
    empty = ParityClears(np.zeros(0, np.int64), np.zeros(0, np.uint16))
    projection = ProjectionMap(
        projection.atom, projection.component, projection.instance, projection.external, projection.edge, empty
    )
    return _finish(value, arrays, projected), projection


def _repoint(c: MutableMapping[str, NDArray], hydrogen, first, second) -> None:
    r"""Re-point each KNOWN cis/trans reference hydrogen to another heavy neighbour of its atom, flipping the value."""
    edge, references = c["stereobond_edge"].astype(np.int64), c["stereobond_reference"].astype(np.int64)
    values = c["stereobond_value"].copy()
    for row in np.flatnonzero(known(c["stereobond_state"])):
        ends = (int(first[edge[row]]), int(second[edge[row]]))
        for k in (0, 1):
            reference = references[row, k]
            if reference < 0 or not hydrogen[reference]:
                continue
            atom, other = ends[k], ends[1 - k]
            neighbours = np.concatenate((second[first == atom], first[second == atom]))
            heavy = neighbours[~hydrogen[neighbours] & (neighbours != other)]
            if not len(heavy):
                fail("hydrogen_view_loses_stereo", RequestRef("stereobond", str(row)), rule="reference_hydrogen")
            references[row, k] = heavy.min()
            values[row] = 3 - values[row]
    c["stereobond_reference"], c["stereobond_value"] = references.astype(np.int32), values


def select(
    structure: Structure,
    *,
    chains: Sequence[str] | None = None,
    namespace: Literal["author", "label"] = "author",
    water: bool = True,
    hydrogens: bool = True,
) -> Structure:
    r"""
    The view a reader asks for after the read: chains, water, hydrogens.

    ``chains`` in the author namespace selects every instance whose author chain is one of them (an author chain can
    span several label instances); in the label namespace, the instances of the named label asyms, every copy of each.
    A chain matching nothing in the namespace (an author chain of UNKNOWN state matches nothing) is
    ``InvalidRequestError(projection_invalid)``, never a fallback to the other namespace. ``water=False`` drops
    instances of water entities (by entity type, not name); ``hydrogens=False`` folds hydrogens into counts. Each part
    is the projection it names: claims, externals and counts carry every effect.
    """

    system = structure.system
    keep = np.ones(len(system.instance_symbol), dtype=bool)
    if chains is not None:
        if namespace == "author":
            codes, present = structure.instance_author_chain, known(structure.instance_author_chain_state)
            family = structure.symbols()["chain"]
        elif namespace == "label":
            codes, family = structure.instance_source_asym, structure.symbols()["asym"]
            present = codes >= 0
        else:
            fail("value_invalid", RequestRef("namespace", str(namespace)))
        lookup = {text: code for code, text in enumerate(family)}
        requested = np.array([lookup.get(chain, -2) for chain in chains], dtype=np.int64)
        for chain, code in zip(chains, requested):
            if not (present & (codes == code)).any():
                fail("projection_invalid", RequestRef("chains", chain), namespace=namespace)
        keep &= present & np.isin(codes, requested)
    if not water:
        entity = system.instance_entity
        keep &= ~(known(system.entity_type_state[entity]) & (system.entity_type[entity] == EntityType.WATER))
    value = structure if keep.all() else project_instances(structure, keep)[0]
    return value if hydrogens else fold_hydrogens(value)[0]
