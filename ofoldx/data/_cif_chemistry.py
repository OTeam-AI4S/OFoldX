r"""Standard mmCIF projection of the canonical tensor molecular graph."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from numpy.typing import NDArray

from ..io.identity import Endpoint, IdentityConflictError, SiteIndex, _intersect_namespaces
from ..io.records import BranchResidueKey
from ..io.symmetry import (
    PlacementIndex,
    SymmetryContext,
)
from ._cif import parse_chemical_stereo, parse_optional_flag
from ._materialization.plan import ResolvedConnections
from ._placed_atoms import PlacedAtoms
from .components import (
    polymer_attachment_anchors,
    terminal_leaving_groups,
)
from .elements import NON_METAL_ELEMENTS
from .molecular import (
    HYDROGEN_ELEMENTS,
    AtomSite,
    BondDirection,
    ChemicalAtom,
    ChemicalBond,
    ChemicalComponent,
    Molecule,
    MoleculeBond,
    PolymerAttachment,
    SystemBond,
)

if TYPE_CHECKING:
    from ..io.cif import CifDocument
    from ..io.source import CifSource

from ..io.cif import column_integers, column_points
from ..io.declarations import (
    _cif_column,
    _cif_string,
    read_component_parents,
)
from ..io.selection import _select_cif_declarations
from ..io.source import StructureView, _require_cif_source

#: ``value_order`` to ``(order, aromatic)``. A Kekulé order carries no aromaticity claim:
#: the wwPDB writes ring bonds as ``sing``/``doub`` alongside ``pdbx_aromatic_flag Y``,
#: so only the explicit ``arom`` order determines the flag.
_CIF_BOND_ORDERS: dict[str, tuple[float, bool | None]] = {
    "sing": (1.0, None),
    "doub": (2.0, None),
    "trip": (3.0, None),
    "quad": (4.0, None),
    "arom": (1.5, True),
}


def _is_graph_connection_type(value: str | None) -> bool:
    if value is None:
        return False
    normalized = value.casefold()
    return normalized.startswith(("covale", "disulf", "metalc")) or normalized == "modres"


def _validate_mmcif_atom_provenance(
    cif_file: CifDocument,
    provenance: frozenset[AtomSite],
    components: Mapping[str, ChemicalComponent],
) -> None:
    r"""Require every OFoldX provenance row to name one declared source atom."""

    if not provenance:
        return

    block = cif_file.block
    struct_asym = block.get("struct_asym")
    declared_asym_ids = frozenset() if struct_asym is None else frozenset(_cif_column(struct_asym, "id"))
    components_by_position: dict[tuple[str, int], set[str]] = {}

    def add_positions(
        category_name: str,
        position_name: str | None,
    ) -> None:
        category = block.get(category_name)
        if category is None:
            return
        asym_ids = _cif_column(category, "asym_id")
        positions = (1,) * len(asym_ids) if position_name is None else column_integers(category, position_name)
        for asym_id, position, component_id in zip(
            asym_ids,
            positions,
            _cif_column(category, "mon_id"),
            strict=True,
        ):
            component_index = position
            if asym_id is None or component_index is None or component_index < 1 or component_id is None:
                raise ValueError(f"mmCIF _{category_name} cannot identify an OFoldX provenance component")
            key = asym_id, component_index - 1
            components_by_position.setdefault(key, set()).add(component_id)

    add_positions("pdbx_poly_seq_scheme", "seq_id")
    add_positions("pdbx_branch_scheme", "num")
    add_positions("pdbx_nonpoly_scheme", None)

    atom_ids_by_component = {
        component_id: {atom.atom_id for atom in component.atoms} for component_id, component in components.items()
    }
    for category_name, component_field, atom_field in (
        ("chem_comp_atom", "comp_id", "atom_id"),
        ("atom_site", "label_comp_id", "label_atom_id"),
    ):
        category = block.get(category_name)
        if category is not None:
            for component, atom in zip(category.strings(component_field), category.strings(atom_field), strict=True):
                if component is not None and atom is not None:
                    atom_ids_by_component.setdefault(component, set()).add(atom)
    for site in provenance:
        component_ids = components_by_position.get((site.instance_id, site.component_index), ())
        if (
            site.instance_id not in declared_asym_ids
            or not component_ids
            or not any(site.atom_id in atom_ids_by_component.get(component_id, ()) for component_id in component_ids)
        ):
            raise ValueError(f"mmCIF _ofoldx_source_formal_charge references unknown atom {site!r}")


class _CifOccurrence(NamedTuple):
    label: tuple[object, ...]
    auth: tuple[object, ...]
    branch_number: int | None
    rows: tuple[int, ...] = ()


class _CifSourceSites(NamedTuple):
    occurrences: tuple[_CifOccurrence, ...]
    sites: SiteIndex
    observed_rows: dict[AtomSite, tuple[int, ...]]


def _mmcif_source_occurrences(source: CifSource, model_index: int) -> tuple[_CifOccurrence, ...]:
    """Collect residue identities before any component atom dictionary is needed."""
    rows = source.atoms.by_index(model_index)
    occurrences: dict[tuple[object, ...], _CifOccurrence] = {}
    keys_by_label: dict[tuple[object, ...], dict[object, set[tuple[object, ...]]]] = {}
    represented_asym_ids: set[object] = set()

    def add(
        key: tuple[object, ...],
        label: tuple[object, ...],
        auth: tuple[object, ...],
        positions: tuple[int, ...] = (),
        *,
        branch_number: int | None = None,
    ) -> None:
        previous = occurrences.get(key)
        if previous is not None:
            merged = []
            for old, new in ((previous.label, label), (previous.auth, auth)):
                if any(a is not None and b is not None and a != b for a, b in zip(old, new, strict=True)):
                    raise IdentityConflictError(f"conflicting source occurrence declarations for {key!r}")
                merged.append(tuple(a if a is not None else b for a, b in zip(old, new, strict=True)))
            label, auth = merged
            positions = previous.rows + positions
            branch_number = previous.branch_number
            if previous.label != label or previous.auth[1] != auth[1]:
                keys_by_label[previous.label][previous.auth[1]].remove(key)
        occurrences[key] = _CifOccurrence(label, auth, branch_number, positions)
        keys_by_label.setdefault(label, {}).setdefault(auth[1], set()).add(key)
        represented_asym_ids.add(label[0])

    for category_name, kind, number_field, chain_field in (
        ("pdbx_poly_seq_scheme", "polymer", "seq_id", "pdb_strand_id"),
        ("pdbx_branch_scheme", "branched", "num", "pdb_asym_id"),
    ):
        category = source.block.get(category_name)
        if category is None:
            continue
        numbers = column_integers(category, number_field)
        for asym, number, insertion, component, author, author_number in zip(
            category.strings("asym_id"),
            numbers,
            category.strings("pdb_ins_code"),
            category.strings("mon_id"),
            category.strings(chain_field),
            category.strings("pdb_seq_num"),
            strict=True,
        ):
            if asym is not None and component is not None:
                add(
                    (kind, asym, number, component),
                    (asym, number if kind == "polymer" else None, insertion, component),
                    (author, author_number, insertion, component),
                    branch_number=number if kind == "branched" else None,
                )
    declarations = source.declarations
    for asym, declared in declarations.nonpolymer_occurrences.items():
        for ndb_number, _, component, author_number, insertion, author in declared:
            add(
                ("non-polymer", asym, ndb_number, component),
                (asym, None, insertion, component),
                (author, author_number, insertion, component),
            )
    branch_components: dict[str, dict[str, set[int]]] = {}
    branch = source.block.get("pdbx_entity_branch_list")
    if branch is not None:
        for entity, number, component in zip(
            branch.strings("entity_id"), column_integers(branch, "num"), branch.strings("comp_id"), strict=True
        ):
            if entity is not None and number is not None and component is not None:
                branch_components.setdefault(entity, {}).setdefault(component, set()).add(number)
    key: tuple[object, ...]
    for asym, entity in declarations.asym_entities.items():
        for component, branch_numbers in branch_components.get(entity, {}).items():
            for number in sorted(branch_numbers):
                key = ("branched", asym, number, component)
                if key not in occurrences:
                    add(key, (asym, None, None, component), (None, None, None, None), branch_number=number)
    for position, row in enumerate(rows):
        label = (row.chain_id, row.label_seq_id, row.insertion_code, row.component_id)
        auth = (row.auth_chain_id, row.auth_residue_id, row.insertion_code, row.auth_component_id)
        branch_numbers = branch_components.get(row.entity_id, {}).get(row.component_id, set())
        branch_number = None
        if row.label_seq_id is not None:
            key = ("polymer", row.chain_id, row.label_seq_id, row.component_id)
        elif len(branch_numbers) == 1:
            branch_number = next(iter(branch_numbers))
            key = ("branched", row.chain_id, branch_number, row.component_id)
        else:
            # Water/ligand occurrences can share a label tuple. Known author
            # numbers narrow the lookup while unknown numbers remain candidates.
            numbered = keys_by_label.get(label, {})
            groups = numbered.values() if auth[1] is None else (numbered.get(auth[1], ()), numbered.get(None, ()))
            matches = [
                key
                for group in groups
                for key in group
                if all(a is None or b is None or a == b for a, b in zip(occurrences[key].auth, auth, strict=True))
            ]
            key = matches[0] if len(matches) == 1 else ("observed", label, auth)
        add(key, label, auth, (position,), branch_number=branch_number)
    observed_chains = set(rows.chain_id)
    for asym, entity in declarations.asym_entities.items():
        for number, candidates in (declarations.polymer_monomers or {}).get(entity, ()):
            for component in candidates:
                key = ("polymer", asym, number, component)
                if key not in occurrences:
                    add(key, (asym, number, None, component), (None, None, None, component))
        component = declarations.nonpolymer_components.get(entity)
        if component is not None and asym not in observed_chains and asym not in represented_asym_ids:
            add(("non-polymer", asym), (asym, None, None, component), (None, None, None, component))
    return tuple(occurrences.values())


def _resolve_mmcif_occurrences(occurrences: Sequence[_CifOccurrence], endpoint: Endpoint) -> tuple[_CifOccurrence, ...]:
    """Collect compatible occurrences; atom evidence establishes identity after chemistry resolution."""
    matches: list[frozenset[int] | None] = []
    for namespace, query in enumerate((endpoint.label[:-1], endpoint.auth[:-1])):
        fields = tuple(i for i, value in enumerate(query) if value is not None)
        if not fields or fields == (2,):
            matches.append(None)
            continue
        possible: set[int] = set()
        for position, occurrence in enumerate(occurrences):
            identity = occurrence.label if namespace == 0 else occurrence.auth
            known = tuple(i for i in fields if namespace == 0 or identity[i] is not None)
            if all(identity[i] == query[i] for i in known):
                possible.add(position)
        matches.append(frozenset(possible))
    if all(match is None for match in matches):
        # An atom-only reference can identify its occurrence only after atom validation.
        return tuple(occurrences)
    candidates = _intersect_namespaces(matches[0], matches[1], endpoint)
    return tuple(occurrences[position] for position in sorted(candidates))


def _mmcif_source_sites(
    source: CifSource,
    model_index: int,
    components: Mapping[str, ChemicalComponent],
    occurrences: tuple[_CifOccurrence, ...],
) -> _CifSourceSites:
    """Index source atoms and validate canonical endpoints before projection."""
    atoms = {key: {atom.atom_id for atom in value.atoms} for key, value in components.items()}
    category = source.block.get("chem_comp_atom")
    if category is not None:
        for component, atom in zip(category.strings("comp_id"), category.strings("atom_id"), strict=True):
            if component is not None and atom is not None:
                atoms.setdefault(component, set()).add(atom)
    rows = source.atoms.by_index(model_index)
    for component, atom in zip(rows.component_id, rows.atom_name, strict=True):
        atoms.setdefault(str(component), set()).add(str(atom))
    sites = SiteIndex()
    observed: dict[AtomSite, list[int]] = {}
    for position, occurrence in enumerate(occurrences):
        label, auth = occurrence.label, occurrence.auth
        for name in atoms.get(str(label[3]), ()):
            site = AtomSite(str(label[0]), position, name)
            sites.add("label", (None, *label, name), site)
            sites.add("auth", (None, *auth, None), site)
        for row_position in occurrence.rows:
            row = rows[row_position]
            # Observed author atom names are positive evidence, including renamed atoms.
            site = AtomSite(str(row.chain_id), position, str(row.atom_name))
            sites.add("auth", (None, *auth, row.auth_atom_name), site)
            observed.setdefault(site, []).append(row_position)
    for connection in source.connections:
        if not _is_graph_connection_type(connection.kind):
            continue
        for endpoint_number, endpoint in enumerate((connection.first, connection.second), start=1):
            canonical_site = sites.unique(endpoint)
            if canonical_site is None:
                raise ValueError(
                    f"mmCIF struct_conn row {connection.row} cannot resolve endpoint {endpoint_number} "
                    "in the source molecular declaration"
                )
            if endpoint.altloc is not None and not any(
                rows.altloc_id[position] == endpoint.altloc for position in observed.get(canonical_site, ())
            ):
                raise ValueError(
                    f"mmCIF struct_conn row {connection.row} cannot resolve explicit alternate "
                    f"{endpoint.altloc!r} in source atom_site"
                )
    return _CifSourceSites(occurrences, sites, {site: tuple(positions) for site, positions in observed.items()})


def _materialize_mmcif_crystallographic_copies(
    cif_file: StructureView,
    rows,
    selected_rows,
    placements,
    *,
    source_sites: _CifSourceSites,
    model_occurrences: frozenset[int],
    include_water: bool,
):
    """Plan physical copies by connected molecular units, never atom-table masks.

    Identity connections form a disjoint-set graph once. Each crystal seed then
    gathers its connected units by index; Cartesian transforms stay deferred.
    """
    if any(symmetry is not None for symmetry, _ in placements):
        return rows, dict(placements), set()
    from ._placed_atoms import PlacedAtoms

    source = _require_cif_source(cif_file.source)
    native_sites, observed_rows_by_site = source_sites.sites, source_sites.observed_rows
    frames = SymmetryContext(source.block)
    connections = tuple(connection for connection in source.connections if _is_graph_connection_type(connection.kind))
    operations = {
        connection.row: frames.pair(connection.first.symmetry, connection.second.symmetry) for connection in connections
    }
    if all(all(operation[0] is None for operation in pair) for pair in operations.values()):
        return rows, dict(placements), set()
    declarations = _select_cif_declarations(cif_file)
    entity_types = declarations.entity_types
    asym_entities = declarations.asym_entities
    nonpoly_occurrences = declarations.nonpolymer_occurrences
    branched = source.block.get("pdbx_entity_branch_list")
    branched_entities = set() if branched is None else set(branched.strings("entity_id"))

    def chain_unit(entity):
        return entity_types.get(entity) in {"polymer", "branched"} or entity in branched_entities

    unit_keys: dict[tuple[object, ...], int] = {}
    unit_rows: list[list[int]] = []
    unit_declarations, parents = [], []

    def unit(key, declaration=None):
        identifier = unit_keys.get(key)
        if identifier is None:
            identifier = len(unit_rows)
            unit_keys[key] = identifier
            unit_rows.append([])
            unit_declarations.append(declaration)
            parents.append(identifier)
        return identifier

    row_units = []
    for position, row in enumerate(selected_rows):
        key = (
            ("chain", row.chain_id)
            if chain_unit(row.entity_id)
            else (
                "occurrence",
                row.chain_id,
                row.residue_id,
                row.auth_residue_id,
                row.insertion_code,
                row.component_id,
            )
        )
        identifier = unit(key)
        unit_rows[identifier].append(position)
        row_units.append(identifier)

    deposited_rows = source.atoms.by_index(cif_file.model_index)
    selected_positions = {int(row_id): position for position, row_id in enumerate(selected_rows.row_index)}
    source_occurrences = source_sites.occurrences
    units_by_occurrence = {
        index: {
            row_units[selected_positions[int(deposited_rows.row_index[position])]]
            for position in occurrence.rows
            if int(deposited_rows.row_index[position]) in selected_positions
        }
        for index, occurrence in enumerate(source_occurrences)
    }

    def resolve_unit(endpoint):
        site = native_sites.unique(endpoint)
        if site is None or site.component_index not in model_occurrences:
            return None
        source_positions = set(observed_rows_by_site.get(site, ()))
        observed = [
            selected_positions[int(deposited_rows.row_index[position])]
            for position in source_positions
            if int(deposited_rows.row_index[position]) in selected_positions
            and (endpoint.altloc is None or deposited_rows.altloc_id[position] == endpoint.altloc)
        ]
        if len(observed) > 1:
            raise ValueError("mmCIF struct_conn endpoint is ambiguous in the selected atom rows")
        if observed:
            return row_units[observed[0]]
        # Omission by a requested altloc/water selection is distinct from an
        # unobserved atom still belonging to a declared molecular unit.
        if source_positions:
            return None
        occurrence = source_occurrences[site.component_index]
        candidates = units_by_occurrence[site.component_index]
        if len(candidates) > 1:
            raise ValueError("mmCIF struct_conn endpoint occurrence is ambiguous")
        if candidates:
            return next(iter(candidates))
        asym = str(occurrence.label[0])
        entity = asym_entities.get(asym)
        if entity is None or (not include_water and entity_types.get(entity) == "water"):
            return None
        if chain_unit(entity):
            return unit(("chain", asym), (asym, None, entity))
        label, auth = occurrence.label, occurrence.auth
        candidates = [
            ndb
            for ndb, declared_entity, monomer, number, insertion, author_chain in nonpoly_occurrences.get(asym, ())
            if declared_entity == entity
            and (label[3] is None or label[3] == monomer)
            and (auth[1] is None or number is None or auth[1] == number)
            and (label[2] is None or insertion is None or label[2] == insertion)
            and (auth[0] is None or author_chain is None or auth[0] == author_chain)
        ]
        if len(candidates) > 1:
            raise ValueError("mmCIF struct_conn endpoint occurrence is ambiguous")
        if not candidates:
            return None
        return unit(("declared", asym, candidates[0]), (asym, candidates[0], entity))

    def root(identifier):
        while parents[identifier] != identifier:
            parents[identifier] = parents[parents[identifier]]
            identifier = parents[identifier]
        return identifier

    seeds: dict[str, set[int]] = {}
    transforms_by_symmetry: dict[str, NDArray[np.float64]] = {}
    for connection in connections:
        first, second = resolve_unit(connection.first), resolve_unit(connection.second)
        if first is None or second is None:
            continue
        pair = operations[connection.row]
        if all(operation[0] is None for operation in pair):
            parents[root(first)] = root(second)
        else:
            for identifier, (symmetry, transform) in zip((first, second), pair, strict=True):
                if symmetry is not None:
                    previous = transforms_by_symmetry.setdefault(symmetry, transform)
                    if previous is not transform and not np.allclose(previous, transform, rtol=0, atol=1e-8):
                        raise ValueError(f"mmCIF crystallographic symmetry {symmetry!r} has conflicting transforms")
                    seeds.setdefault(symmetry, set()).add(identifier)
    if not seeds:
        return rows, dict(placements), set()
    connected: dict[int, list[int]] = {}
    for identifier in range(len(unit_rows)):
        connected.setdefault(root(identifier), []).append(identifier)
    source_positions = {int(row_id): i for i, row_id in enumerate(rows.row_index)}
    positions, copies = list(range(len(rows))), [0] * len(rows)
    transforms: list[tuple[str | None, NDArray[np.float64]]] = [(None, np.eye(4))]
    materialized_placements, topology_placements = dict(placements), set()
    for symmetry, seed_units in seeds.items():
        transform = transforms_by_symmetry[symmetry]
        selected_units = {
            identifier for component in {root(seed) for seed in seed_units} for identifier in connected[component]
        }
        chosen = sorted(position for identifier in selected_units for position in unit_rows[identifier])
        positions.extend(source_positions[int(selected_rows.row_index[position])] for position in chosen)
        copies.extend([len(transforms)] * len(chosen))
        transforms.append((symmetry, transform))
        for asym in set(selected_rows.chain_id[chosen]):
            materialized_placements[(symmetry, asym)] = transform
        for identifier in sorted(selected_units):
            declaration = unit_declarations[identifier]
            if declaration is not None:
                asym, ndb, _ = declaration
                materialized_placements[(symmetry, asym)] = transform
                topology_placements.add((symmetry, asym, ndb))
    return PlacedAtoms(rows, positions, copies, transforms), materialized_placements, topology_placements


def read_mmcif_components(
    cif_file: CifDocument,
    *,
    owned: Collection[str] = (),
    selected: Collection[str] | None = None,
) -> tuple[
    dict[str, ChemicalComponent],
    set[str],
    dict[str, tuple[str, ...]],
    dict[str, tuple[frozenset[str], frozenset[frozenset[str]]]],
]:
    r"""
    Read embedded component definitions and every declared component ID.

    ``chem_comp_atom`` and ``chem_comp_bond`` carry the component graph, ``_chem_comp.type``
    its polymer chemistry, and ``pdbx_leaving_atom_flag`` the atoms a realized polymer link
    removes; together the last two give the component its polymer attachments, exactly as
    [`ofoldx.data.ccd.build`][] reads the same items out of the wwPDB dictionary. Reference
    coordinates come from the idealized conformer where the source declares one and from the
    representative model coordinates otherwise.

    An embedded definition explicitly supplied through ``owned`` is still declared but
    never built. All other source chemistry is retained; a later dictionary completion may
    fill facts it leaves unstated but cannot replace facts it states.

    Args:
        cif_file: Parsed mmCIF document.
        owned: Component IDs whose chemistry the caller supplies itself.
        selected: Component IDs needed for the requested occurrences and their
            connection endpoints; `None` reads every embedded definition.

    Returns:
        Embedded component definitions by component ID, every declared component ID,
        nonstandard-to-parent component IDs from ``_chem_comp``, and explicit atom/bond
        stereo fields (including explicit ``N`` values) by component ID.

    Raises:
        ValueError: If a ``chem_comp_atom`` or ``chem_comp_bond`` row omits an identifier,
            an atom states an invalid leaving flag, or a bond states an unsupported order
            or an aromaticity disagreeing with it.
    """

    block = cif_file.block
    declared_components: set[str] = set()
    component_parents = read_component_parents(block.get("chem_comp"))
    for category_name, column_name in (
        ("entity_poly_seq", "mon_id"),
        ("pdbx_entity_branch_list", "comp_id"),
        ("pdbx_entity_nonpoly", "comp_id"),
    ):
        if category_name in block:
            declared_components.update(
                value for value in _cif_column(block[category_name], column_name) if value is not None
            )
    if "chem_comp_atom" not in block:
        return {}, declared_components, component_parents, {}
    component_types = (
        dict(
            zip(
                _cif_column(block["chem_comp"], "id"),
                _cif_column(block["chem_comp"], "type"),
            )
        )
        if "chem_comp" in block
        else {}
    )
    atom_category = block["chem_comp_atom"]
    component_ids = _cif_column(atom_category, "comp_id")
    declared_components.update(component_id for component_id in component_ids if component_id is not None)
    ignored_components: set[str | None] = set(owned)
    if selected is not None:
        ignored_components.update(set(component_ids) - set(selected))
    atom_ids = _cif_column(atom_category, "atom_id")
    elements = _cif_column(atom_category, "type_symbol")
    charges = column_integers(atom_category, "charge")
    aromatic_flags = _cif_column(atom_category, "pdbx_aromatic_flag")
    stereo_values = _cif_column(atom_category, "pdbx_stereo_config")
    leaving_flags = _cif_column(atom_category, "pdbx_leaving_atom_flag")

    atoms: dict[str, list[ChemicalAtom]] = {}
    leaving_atoms: dict[str, set[str]] = {}
    non_leaving_atoms: dict[str, set[str]] = {}
    model_coordinates: dict[str, list[tuple[float, float, float] | None]] = {}
    ideal_coordinates: dict[str, list[tuple[float, float, float] | None]] = {}
    atom_stereo_known: dict[str, set[str]] = {}
    for values in zip(
        component_ids,
        atom_ids,
        elements,
        charges,
        aromatic_flags,
        stereo_values,
        leaving_flags,
        column_points(atom_category, "model_Cartn_x", "model_Cartn_y", "model_Cartn_z"),
        column_points(
            atom_category,
            "pdbx_model_Cartn_x_ideal",
            "pdbx_model_Cartn_y_ideal",
            "pdbx_model_Cartn_z_ideal",
        ),
    ):
        component_id, atom_id, element, charge, aromatic, stereo, leaving, model_point, ideal_point = values
        if component_id is None:
            raise ValueError("mmCIF chem_comp_atom row is missing comp_id, atom_id, or type_symbol")
        if component_id in ignored_components:
            continue
        if atom_id is None or element is None:
            raise ValueError("mmCIF chem_comp_atom row is missing comp_id, atom_id, or type_symbol")
        atoms.setdefault(component_id, []).append(
            ChemicalAtom(
                atom_id=atom_id,
                element=element.capitalize(),
                formal_charge=charge,
                aromatic=parse_optional_flag(aromatic, item="_chem_comp_atom.pdbx_aromatic_flag"),
                stereo=parse_chemical_stereo(stereo, category="atom"),
            )
        )
        if stereo is not None:
            atom_stereo_known.setdefault(component_id, set()).add(atom_id)
        if leaving is not None:
            normalized_leaving = leaving.upper()
            if normalized_leaving not in {"Y", "N"}:
                raise ValueError("mmCIF chem_comp_atom pdbx_leaving_atom_flag must be Y or N")
            if normalized_leaving == "Y":
                leaving_atoms.setdefault(component_id, set()).add(atom_id)
            else:
                non_leaving_atoms.setdefault(component_id, set()).add(atom_id)
        model_coordinates.setdefault(component_id, []).append(model_point)
        ideal_coordinates.setdefault(component_id, []).append(ideal_point)

    bonds_by_component: dict[str, set[ChemicalBond]] = {component_id: set() for component_id in atoms}
    components_with_bonds: set[str] = set()
    bond_stereo_known: dict[str, set[frozenset[str]]] = {}
    if "chem_comp_bond" in block:
        bond_category = block["chem_comp_bond"]
        order_names = (
            _cif_column(bond_category, "value_order")
            if "value_order" in bond_category
            else ("sing",) * bond_category.row_count
        )
        for component_id, atom_a, atom_b, order_name, aromatic, stereo in zip(
            _cif_column(bond_category, "comp_id"),
            _cif_column(bond_category, "atom_id_1"),
            _cif_column(bond_category, "atom_id_2"),
            order_names,
            _cif_column(bond_category, "pdbx_aromatic_flag"),
            _cif_column(bond_category, "pdbx_stereo_config"),
        ):
            if component_id is None:
                raise ValueError("mmCIF chem_comp_bond row is missing comp_id or atom IDs")
            if component_id in ignored_components:
                continue
            if atom_a is None or atom_b is None:
                raise ValueError("mmCIF chem_comp_bond row is missing comp_id or atom IDs")
            components_with_bonds.add(component_id)
            order, order_aromatic = _cif_bond_order(order_name)
            if aromatic is None:
                aromatic_value = order_aromatic
            else:
                if aromatic.upper() not in {"Y", "N"}:
                    raise ValueError("mmCIF chem_comp_bond pdbx_aromatic_flag must be Y or N")
                aromatic_value = aromatic.upper() == "Y"
                if order_aromatic is not None and aromatic_value != order_aromatic:
                    raise ValueError("mmCIF chem_comp_bond value_order and pdbx_aromatic_flag disagree")
            bonds_by_component.setdefault(component_id, set()).add(
                ChemicalBond(
                    atom_a,
                    atom_b,
                    order=order,
                    aromatic=aromatic_value,
                    stereo=parse_chemical_stereo(stereo, category="bond"),
                )
            )
            if stereo is not None:
                bond_stereo_known.setdefault(component_id, set()).add(frozenset((atom_a, atom_b)))
    components: dict[str, ChemicalComponent] = {}
    for component_id, component_atoms in atoms.items():
        component_leaving_atoms = leaving_atoms.get(component_id, set())
        component_non_leaving_atoms = non_leaving_atoms.get(component_id, set())
        component_atom_ids = {atom.atom_id for atom in component_atoms}
        leaving_flags_complete = component_leaving_atoms | component_non_leaving_atoms == component_atom_ids
        component_bonds = (
            frozenset(bonds_by_component[component_id])
            if len(component_atoms) == 1 or component_id in components_with_bonds
            else None
        )
        previous_attachment, next_attachment = _polymer_attachments(
            component_types.get(component_id),
            component_atoms,
            component_leaving_atoms if leaving_flags_complete else None,
            component_bonds,
        )
        ideal_points = ideal_coordinates[component_id]
        components[component_id] = ChemicalComponent(
            component_id=component_id,
            atoms=tuple(component_atoms),
            bonds=component_bonds,
            previous_attachment=previous_attachment,
            next_attachment=next_attachment,
            # Each column triple is one whole conformer, so a component takes every
            # reference coordinate from the same one: the idealized conformer this project
            # reads as the reference geometry, or the representative model coordinates for a
            # component that declares no idealized one.
            reference_coordinates=tuple(
                ideal_points if any(point is not None for point in ideal_points) else model_coordinates[component_id]
            ),
            leaving_atom_ids=frozenset(component_leaving_atoms),
            non_leaving_atom_ids=frozenset(component_non_leaving_atoms),
        )
    stereo_known = {
        component_id: (
            frozenset(atom_stereo_known.get(component_id, ())),
            frozenset(bond_stereo_known.get(component_id, ())),
        )
        for component_id in components
    }
    return components, declared_components, component_parents, stereo_known


def _polymer_attachments(
    chem_comp_type: str | None,
    atoms: Sequence[ChemicalAtom],
    leaving_atom_ids: set[str] | None,
    bonds: frozenset[ChemicalBond] | None,
) -> tuple[PolymerAttachment | None, PolymerAttachment | None]:
    r"""
    Derive the linear-polymer attachments an embedded component declares.

    ``_chem_comp.type`` names the polymer chemistry while ``pdbx_leaving_atom_flag``
    names the atoms a realized link removes. Each connected leaving group belongs to
    its unique non-leaving graph boundary. This matters for beta- and gamma-peptides,
    whose outgoing carbonyl is not the conventional atom ``C``.
    """

    if leaving_atom_ids is None or bonds is None:
        return None, None
    previous_anchor, next_anchor = polymer_attachment_anchors(
        chem_comp_type,
        tuple(atom.atom_id for atom in atoms),
    )
    if previous_anchor is None and next_anchor is None:
        return None, None

    atom_ids = tuple(atom.atom_id for atom in atoms)
    present = set(atom_ids)
    bond_pairs = tuple((bond.atom_a, bond.atom_b) for bond in bonds)

    def attachment(anchor_id: str | None) -> PolymerAttachment | None:
        if anchor_id is None or anchor_id not in present:
            return None
        groups = terminal_leaving_groups(atom_ids, bond_pairs, leaving_atom_ids, anchor_id)
        if len(groups) > 1:
            return None
        return PolymerAttachment(anchor_id, groups[0] if groups else frozenset())

    return attachment(previous_anchor), attachment(next_anchor)


def _cif_bond_order(value: str | None) -> tuple[float | None, bool | None]:
    if value is None:
        return None, None
    try:
        return _CIF_BOND_ORDERS[value.lower()]
    except KeyError:
        raise ValueError(f"unsupported mmCIF covalent bond order: {value!r}") from None


def _remove_component_atoms(component: ChemicalComponent, atom_ids: set[str]) -> ChemicalComponent:
    atom_by_id = {atom.atom_id: atom for atom in component.atoms}
    removed = set(atom_by_id).intersection(atom_ids)
    if not removed:
        return component
    bonds = component.bonds
    if bonds is not None:
        for bond in bonds:
            if bond.atom_a in removed and atom_by_id[bond.atom_b].element in HYDROGEN_ELEMENTS:
                removed.add(bond.atom_b)
            if bond.atom_b in removed and atom_by_id[bond.atom_a].element in HYDROGEN_ELEMENTS:
                removed.add(bond.atom_a)
        for bond in bonds:
            if (
                bond.stereo_atoms is not None
                and removed.intersection(bond.stereo_atoms)
                and not removed.intersection((bond.atom_a, bond.atom_b))
            ):
                raise ValueError("a declared leaving atom is required to define retained bond stereochemistry")
    keep = tuple(atom.atom_id not in removed for atom in component.atoms)
    retained = set(atom_by_id) - removed
    previous = component.previous_attachment
    if previous is not None:
        previous = (
            PolymerAttachment(
                previous.atom_id,
                previous.leaving_atom_ids & retained,
            )
            if previous.atom_id in retained
            else None
        )
    following = component.next_attachment
    if following is not None:
        following = (
            PolymerAttachment(
                following.atom_id,
                following.leaving_atom_ids & retained,
            )
            if following.atom_id in retained
            else None
        )
    return ChemicalComponent(
        component_id=component.component_id,
        atoms=tuple(atom for atom, present in zip(component.atoms, keep) if present),
        bonds=(
            None
            if bonds is None
            else frozenset(bond for bond in bonds if bond.atom_a not in removed and bond.atom_b not in removed)
        ),
        previous_attachment=previous,
        next_attachment=following,
        reference_coordinates=(
            tuple(coordinate for coordinate, present in zip(component.reference_coordinates, keep) if present)
            if component.reference_coordinates
            else ()
        ),
        leaving_atom_ids=component.leaving_atom_ids & retained,
        non_leaving_atom_ids=component.non_leaving_atom_ids & retained,
    )


def _component_atom_groups(
    component: ChemicalComponent,
    atom_ids: Collection[str],
) -> tuple[tuple[frozenset[str], frozenset[str]], ...]:
    r"""Connected selected atom groups and their boundaries in a component graph."""

    selected = set(atom_ids)
    if not selected:
        return ()
    component_atoms = {atom.atom_id for atom in component.atoms}
    if not selected <= component_atoms:
        raise ValueError(f"component {component.component_id!r} leaving group references an unknown atom")
    if component.bonds is None:
        raise ValueError(f"component {component.component_id!r} has no known bond graph")
    adjacency: dict[str, set[str]] = {atom_id: set() for atom_id in component_atoms}
    for bond in component.bonds:
        adjacency[bond.atom_a].add(bond.atom_b)
        adjacency[bond.atom_b].add(bond.atom_a)

    groups: list[tuple[frozenset[str], frozenset[str]]] = []
    unseen = set(selected)
    for atom in component.atoms:
        if atom.atom_id not in unseen:
            continue
        unseen.remove(atom.atom_id)
        group = {atom.atom_id}
        pending = [atom.atom_id]
        while pending:
            for neighbor in adjacency[pending.pop()] & unseen:
                unseen.remove(neighbor)
                group.add(neighbor)
                pending.append(neighbor)
        boundary = {neighbor for member in group for neighbor in adjacency[member] if neighbor not in group}
        groups.append((frozenset(group), frozenset(boundary)))
    return tuple(groups)


def _component_leaving_atom(
    component: ChemicalComponent,
    group: Collection[str],
    endpoint: str,
) -> str:
    r"""Return the unique atom through which a leaving group is bonded to its endpoint."""

    if component.bonds is None:
        raise ValueError(f"component {component.component_id!r} has no known bond graph")
    leaving_atoms = set(group)
    candidates = {
        bond.atom_b if bond.atom_a == endpoint else bond.atom_a
        for bond in component.bonds
        if (bond.atom_a == endpoint and bond.atom_b in leaving_atoms)
        or (bond.atom_b == endpoint and bond.atom_a in leaving_atoms)
    }
    if len(candidates) != 1:
        raise ValueError(f"mmCIF cannot infer a unique leaving atom for {component.component_id!r}:{endpoint}")
    return next(iter(candidates))


def read_mmcif_branch_entities(
    cif_file: CifDocument,
    component_table: Mapping[str, ChemicalComponent],
    selected_rows: np.recarray | PlacedAtoms,
    *,
    source_rows: np.recarray,
    source_occurrences: Sequence[_CifOccurrence],
    selected_entities: Collection[str] | None = None,
) -> tuple[
    dict[str, dict[str | None, Molecule]],
    dict[BranchResidueKey, int],
    dict[tuple[str, str | None], tuple[tuple[MoleculeBond, bool, bool], ...]],
]:
    r"""
    Decode selected mmCIF branched entities and source component indices.

    Args:
        cif_file: Parsed mmCIF document.
        component_table: Chemical components available by component ID.
        selected_rows: Atom rows after alternate-location selection.
        source_rows: Original atom rows for the deposited model, before selection
            or placement expansion.
        source_occurrences: Validated source identities, with branch declaration
            numbers and positions into ``source_rows`` for their observed atoms.
        selected_entities: Entities whose chemistry is requested; all source branch
            references are still validated before this projection.

    Returns:
        Branched-entity realizations by entity and label asym ID, selected source
        residue keys mapped to component indices, and the corresponding source link
        declarations before their leaving atoms are removed. ``None`` identifies an
        entity-wide realization whose declaration has no microheterogeneous positions.

    Raises:
        ValueError: If branch declarations, links, bond orders, component references,
            or source numbering are invalid.
    """

    block = cif_file.block
    if "pdbx_entity_branch_list" not in block:
        return {}, {}, {}

    category = block["pdbx_entity_branch_list"]
    entity_ids = _cif_column(category, "entity_id")
    component_ids = _cif_column(category, "comp_id")
    numbers = column_integers(category, "num")
    declarations: dict[str, dict[int, list[str]]] = {}
    for entity_id, component_id, number in zip(entity_ids, component_ids, numbers):
        if entity_id is None:
            raise ValueError("mmCIF branched-entity component is missing entity_id")
        component_index = number
        if component_id is None or component_index is None:
            raise ValueError("mmCIF branched-entity components require entity_id, comp_id, and num")
        candidates = declarations.setdefault(entity_id, {}).setdefault(component_index, [])
        if component_id in candidates:
            raise ValueError(
                f"duplicate branched-entity component {component_id!r} at number "
                f"{component_index} for entity {entity_id!r}"
            )
        candidates.append(component_id)

    number_to_index = {
        entity_id: {number: index for index, number in enumerate(sorted(components))}
        for entity_id, components in declarations.items()
    }
    source_choices: dict[BranchResidueKey, set[tuple[str, str, int, str]]] = {}
    residue_records: list[
        tuple[
            str,
            str,
            int,
            str,
            tuple[BranchResidueKey, ...],
        ]
    ] = []
    branch_asym_ids: dict[str, list[str]] = {}
    if "pdbx_branch_scheme" in block:
        scheme = block["pdbx_branch_scheme"]
        label_asym_ids = _cif_column(scheme, "asym_id")
        scheme_entities = _cif_column(scheme, "entity_id")
        monomer_ids = _cif_column(scheme, "mon_id")
        scheme_numbers = column_integers(scheme, "num")
        pdb_asym_ids = _cif_column(scheme, "pdb_asym_id")
        pdb_monomer_ids = _cif_column(scheme, "pdb_mon_id")
        pdb_sequence_numbers = _cif_column(scheme, "pdb_seq_num")
        pdb_insertion_codes = _cif_column(scheme, "pdb_ins_code")
        for values in zip(
            label_asym_ids,
            scheme_entities,
            monomer_ids,
            scheme_numbers,
            pdb_asym_ids,
            pdb_monomer_ids,
            pdb_sequence_numbers,
            pdb_insertion_codes,
        ):
            (
                label_asym_id,
                entity_id,
                monomer_id,
                number,
                pdb_asym_id,
                pdb_monomer_id,
                pdb_sequence_number,
                pdb_insertion_code,
            ) = values
            if label_asym_id is None or entity_id is None or entity_id not in declarations:
                raise ValueError(f"mmCIF branch scheme references undeclared entity {entity_id!r}")
            number_int = number
            if number_int is None:
                raise ValueError("mmCIF branch scheme requires an integer component number")
            declared_candidates = declarations[entity_id].get(number_int)
            if declared_candidates is None or monomer_id is None or monomer_id not in declared_candidates:
                raise ValueError(f"mmCIF branch scheme disagrees with component declaration for entity {entity_id!r}")
            pdb_sequence_number = _cif_string(pdb_sequence_number)
            pdb_insertion_code = _cif_string(pdb_insertion_code)
            source_keys: set[BranchResidueKey] = {
                BranchResidueKey(label_asym_id, entity_id, monomer_id, label_asym_id, None, pdb_insertion_code),
            }
            if pdb_asym_id is not None and pdb_monomer_id is not None and pdb_sequence_number is not None:
                source_keys.add(
                    BranchResidueKey(
                        label_asym_id,
                        entity_id,
                        pdb_monomer_id,
                        pdb_asym_id,
                        pdb_sequence_number,
                        pdb_insertion_code,
                    )
                )
            choice = (label_asym_id, entity_id, number_int, monomer_id)
            for source_key in source_keys:
                source_choices.setdefault(source_key, set()).add(choice)
            residue_records.append((label_asym_id, entity_id, number_int, monomer_id, tuple(source_keys)))
            asym_ids = branch_asym_ids.setdefault(entity_id, [])
            if label_asym_id not in asym_ids:
                asym_ids.append(label_asym_id)

    for occurrence in source_occurrences:
        if occurrence.branch_number is None or not occurrence.rows:
            continue
        entity_id = _cif_string(source_rows.entity_id[occurrence.rows[0]])
        if entity_id not in declarations:
            continue
        assert entity_id is not None
        label_asym_id = str(occurrence.label[0])
        monomer_id = str(occurrence.label[3])
        source_keys = set()
        for position in occurrence.rows:
            source_row = source_rows[position]
            source_keys.update(
                (
                    BranchResidueKey(
                        label_asym_id,
                        entity_id,
                        source_row.component_id,
                        label_asym_id,
                        None,
                        source_row.insertion_code,
                    ),
                    BranchResidueKey(
                        label_asym_id,
                        entity_id,
                        source_row.auth_component_id,
                        source_row.auth_chain_id,
                        source_row.auth_residue_id,
                        source_row.insertion_code,
                    ),
                )
            )
        choice = (label_asym_id, entity_id, occurrence.branch_number, monomer_id)
        for source_key in source_keys:
            source_choices.setdefault(source_key, set()).add(choice)
        residue_records.append((label_asym_id, entity_id, occurrence.branch_number, monomer_id, tuple(source_keys)))
        asym_ids = branch_asym_ids.setdefault(entity_id, [])
        if label_asym_id not in asym_ids:
            asym_ids.append(label_asym_id)

    selected_components: dict[tuple[str, str, int], str] = {}
    selected_atom_ids: dict[tuple[str, int, str], set[str]] = {}
    for row in selected_rows:
        entity_id = _cif_string(row.entity_id)
        if entity_id not in declarations:
            continue
        choices: set[tuple[str, str, int, str]] = set()
        source_keys = {
            BranchResidueKey(row.chain_id, row.entity_id, row.component_id, row.chain_id, None, row.insertion_code),
            BranchResidueKey(
                row.chain_id,
                row.entity_id,
                row.auth_component_id,
                row.auth_chain_id,
                row.auth_residue_id,
                row.insertion_code,
            ),
        }
        for source_key in source_keys:
            source_candidate_choices = source_choices.get(source_key, ())
            if len(source_candidate_choices) == 1:
                choices.update(source_candidate_choices)
        if len(choices) > 1:
            raise ValueError("selected atom rows ambiguously identify a branched-entity component")
        if not choices:
            raise ValueError("selected atom rows have missing or ambiguous branched-entity component identity")
        selected_asym_id, selected_entity_id, selected_number, selected_component_id = next(iter(choices))
        selected_key = selected_entity_id, selected_asym_id, selected_number
        previous_selection = selected_components.get(selected_key)
        if previous_selection is not None and previous_selection != selected_component_id:
            raise ValueError(
                f"selected atom rows contain multiple branched-entity candidates at "
                f"number {selected_number} for asym {selected_asym_id!r}"
            )
        selected_components[selected_key] = selected_component_id
        if bool(row.coordinate_present):
            selected_atom_ids.setdefault(
                (selected_entity_id, selected_number, selected_component_id),
                set(),
            ).add(str(row.atom_name))

    realized_components: dict[tuple[str, str | None], dict[int, str]] = {}
    for declared_entity_id, numbered_candidates in declarations.items():
        if selected_entities is not None and declared_entity_id not in selected_entities:
            continue
        if all(len(candidates) == 1 for candidates in numbered_candidates.values()):
            realized_components[(declared_entity_id, None)] = {
                component_number: candidates[0] for component_number, candidates in numbered_candidates.items()
            }
            continue
        for realization_asym_id in branch_asym_ids.get(declared_entity_id, ()):
            realization: dict[int, str] = {}
            for component_number, candidates in numbered_candidates.items():
                if len(candidates) == 1:
                    realization[component_number] = candidates[0]
                    continue
                selected = selected_components.get((declared_entity_id, realization_asym_id, component_number))
                if selected is None:
                    break
                realization[component_number] = selected
            else:
                realized_components[(declared_entity_id, realization_asym_id)] = realization

    branch_links: list[
        tuple[
            str,
            int,
            str,
            str,
            str | None,
            int,
            str,
            str,
            str | None,
            float | None,
            bool | None,
        ]
    ] = []
    if "pdbx_entity_branch_link" in block:
        links = block["pdbx_entity_branch_link"]
        link_entities = _cif_column(links, "entity_id")
        number_1 = column_integers(links, "entity_branch_list_num_1")
        number_2 = column_integers(links, "entity_branch_list_num_2")
        component_1 = _cif_column(links, "comp_id_1")
        component_2 = _cif_column(links, "comp_id_2")
        atom_1 = _cif_column(links, "atom_id_1")
        atom_2 = _cif_column(links, "atom_id_2")
        leaving_1 = _cif_column(links, "leaving_atom_id_1")
        leaving_2 = _cif_column(links, "leaving_atom_id_2")
        orders = _cif_column(links, "value_order") if "value_order" in links else ("sing",) * links.row_count
        for values in zip(
            link_entities,
            number_1,
            component_1,
            atom_1,
            leaving_1,
            number_2,
            component_2,
            atom_2,
            leaving_2,
            orders,
        ):
            (
                entity_id,
                first_number,
                first_component,
                first_atom,
                first_leaving,
                second_number,
                second_component,
                second_atom,
                second_leaving,
                order_name,
            ) = values
            if entity_id not in declarations:
                raise ValueError(f"mmCIF branched-entity link references undeclared entity {entity_id!r}")
            first_number_int = first_number
            second_number_int = second_number
            if first_number_int is None or second_number_int is None or first_atom is None or second_atom is None:
                raise ValueError("mmCIF branched-entity links require two component numbers and atom IDs")
            declared = declarations[entity_id]
            if (
                first_component is None
                or second_component is None
                or first_component not in declared.get(first_number_int, ())
                or second_component not in declared.get(second_number_int, ())
            ):
                raise ValueError(f"branched-entity link disagrees with component declaration for entity {entity_id!r}")
            order, aromatic = _cif_bond_order(order_name)
            if order is not None and aromatic is None:
                aromatic = False
            branch_links.append(
                (
                    entity_id,
                    first_number_int,
                    first_component,
                    first_atom,
                    first_leaving,
                    second_number_int,
                    second_component,
                    second_atom,
                    second_leaving,
                    order,
                    aromatic,
                )
            )

    entities: dict[str, dict[str | None, Molecule]] = {entity_id: {} for entity_id in declarations}
    source_bonds: dict[tuple[str, str | None], tuple[tuple[MoleculeBond, bool, bool], ...]] = {}
    for (entity_id, label_asym_id), numbered_components in realized_components.items():
        removed_atoms: dict[int, set[str]] = {}
        bonds: set[MoleculeBond] = set()
        declared_bonds: list[tuple[MoleculeBond, bool, bool]] = []

        def branch_leaving_group(
            component_id: str,
            endpoint: str,
            leaving_atom: str | None,
        ) -> frozenset[str]:
            try:
                component = component_table[component_id]
            except KeyError:
                raise ValueError(
                    f"unknown component {component_id!r}; pass components={{...}} with its complete declared graph"
                ) from None
            atom_ids = tuple(atom.atom_id for atom in component.atoms)
            if endpoint not in atom_ids:
                raise ValueError(
                    f"mmCIF branched-entity link endpoint {component_id!r}:{endpoint} is not in its component"
                )
            if leaving_atom is None:
                return frozenset()
            if leaving_atom not in atom_ids or leaving_atom == endpoint:
                raise ValueError(
                    f"mmCIF branched-entity link {component_id!r}:{endpoint} has invalid leaving atom {leaving_atom!r}"
                )
            if component.bonds is None:
                raise ValueError(f"component {component_id!r} has no known bond graph for its leaving atom")
            groups = terminal_leaving_groups(
                atom_ids,
                ((bond.atom_a, bond.atom_b) for bond in component.bonds),
                {leaving_atom},
                endpoint,
            )
            if len(groups) != 1:
                raise ValueError(
                    f"mmCIF branched-entity link {component_id!r}:{endpoint} has invalid leaving atom {leaving_atom!r}"
                )
            return groups[0]

        for (
            link_entity_id,
            first_number,
            first_component,
            first_atom,
            first_leaving,
            second_number,
            second_component,
            second_atom,
            second_leaving,
            order,
            aromatic,
        ) in branch_links:
            if link_entity_id != entity_id or (
                numbered_components[first_number] != first_component
                or numbered_components[second_number] != second_component
            ):
                continue
            first_index = number_to_index[entity_id][first_number]
            second_index = number_to_index[entity_id][second_number]
            first_leaving_group = branch_leaving_group(first_component, first_atom, first_leaving)
            second_leaving_group = branch_leaving_group(second_component, second_atom, second_leaving)
            for component_number, component_id, leaving_group in (
                (first_number, first_component, first_leaving_group),
                (second_number, second_component, second_leaving_group),
            ):
                observed_leaving = leaving_group & selected_atom_ids.get(
                    (entity_id, component_number, component_id),
                    set(),
                )
                if observed_leaving:
                    raise ValueError(
                        f"mmCIF branched-entity link for entity {entity_id!r} component "
                        f"{component_number} {component_id!r} declares observed leaving "
                        f"atom {sorted(observed_leaving)[0]!r}"
                    )
            bonds.add(MoleculeBond(first_index, first_atom, second_index, second_atom, order=order, aromatic=aromatic))
            declared_bonds.append(
                (
                    MoleculeBond(
                        first_index,
                        first_atom,
                        second_index,
                        second_atom,
                        order=order,
                        aromatic=aromatic,
                        leaving_a=first_leaving_group,
                        leaving_b=second_leaving_group,
                    ),
                    first_leaving is not None,
                    second_leaving is not None,
                )
            )
            removed_atoms.setdefault(first_index, set()).update(first_leaving_group)
            removed_atoms.setdefault(second_index, set()).update(second_leaving_group)

        components: list[ChemicalComponent] = []
        for component_index, component_number in enumerate(sorted(numbered_components)):
            component_id = numbered_components[component_number]
            try:
                component = component_table[component_id]
            except KeyError:
                raise ValueError(
                    f"unknown component {component_id!r}; pass components={{...}} with its complete declared graph"
                ) from None
            components.append(_remove_component_atoms(component, removed_atoms.get(component_index, set())))
        entities[entity_id][label_asym_id] = Molecule(
            components=tuple(components),
            bonds=frozenset(bonds),
            molecule_type="branched",
        )
        source_bonds[(entity_id, label_asym_id)] = tuple(declared_bonds)

    component_indices: dict[BranchResidueKey, int] = {}
    ambiguous_component_indices: set[BranchResidueKey] = set()
    for scheme_asym_id, scheme_entity_id, scheme_number, scheme_component_id, scheme_source_keys in residue_records:
        selected_realization = realized_components.get((scheme_entity_id, scheme_asym_id))
        if selected_realization is None:
            selected_realization = realized_components.get((scheme_entity_id, None))
        if selected_realization is None or selected_realization[scheme_number] != scheme_component_id:
            continue
        selected_component_index = number_to_index[scheme_entity_id][scheme_number]
        for selected_source_key in scheme_source_keys:
            if selected_source_key in ambiguous_component_indices:
                continue
            previous_component_index = component_indices.get(selected_source_key)
            if previous_component_index is not None and previous_component_index != selected_component_index:
                component_indices.pop(selected_source_key)
                ambiguous_component_indices.add(selected_source_key)
                continue
            component_indices[selected_source_key] = selected_component_index
    return entities, component_indices, source_bonds


def _is_metal_site(site: AtomSite, element_by_site: dict[AtomSite, str]) -> bool:
    r"""Whether one atom occurrence is a metal center, from source or component chemistry."""

    element = element_by_site.get(site)
    return element is not None and element not in NON_METAL_ELEMENTS


def read_mmcif_bonds(
    cif_file: StructureView,
    source_rows: np.recarray | PlacedAtoms,
    selected_rows: np.recarray | PlacedAtoms,
    site_by_row_index: Mapping[int, AtomSite],
    site_index: SiteIndex,
    component_table: Mapping[str, ChemicalComponent] | None = None,
    *,
    source_sites: _CifSourceSites,
    model_occurrences: frozenset[int],
    placements: Mapping[tuple[str | None, str], np.ndarray],
    model_rows: np.recarray | PlacedAtoms,
    occurrence_components: Mapping[tuple[str, int], ChemicalComponent],
    component_topology_complete: Mapping[str, bool],
    source_non_leaving_atom_ids: Mapping[str, frozenset[str]],
    declared_branch_bonds: Mapping[frozenset[AtomSite], tuple[SystemBond, bool, bool]] | None = None,
) -> ResolvedConnections:
    r"""
    Decode selected mmCIF covalent connections into system bonds.

    Args:
        cif_file: Selected view of a native CIF source.
        source_rows: Atom rows before user-requested alternate-location and water
            projection.
        selected_rows: Atom rows retained for the selected structure.
        site_by_row_index: Canonical atom sites indexed by source row.
        site_index: Selected canonical sites, including unobserved declared atoms.
        component_table: Component definitions used to validate or uniquely infer
            connection leaving atoms retained in the canonical graph.
        source_sites: Validated complete-model atom identities shared with placements.
        model_occurrences: Source occurrences retained by component/alternate selection,
            including declared occurrences with no observed atoms.
        placements: Cartesian placement of each selected label-asym copy relative to
            the source asymmetric unit.
        model_rows: Complete model after alternate-location selection, retaining
            occurrence evidence for endpoints outside the requested chains/assembly.
        occurrence_components: Component definitions addressed by retained occurrences.
        component_topology_complete: Whether each component definition closes its
            atom and bond graph.
        source_non_leaving_atom_ids: Component atoms the source itself explicitly
            marks as non-leaving.
        declared_branch_bonds: Occurrence-local bonds already represented by
            ``pdbx_entity_branch_link`` in a branched entity, keyed by physical edge.

    Returns:
        Retained bonds and leaving atoms on retained ends of projected connections.

    Raises:
        ValueError: If a selected covalent connection declares an unsupported bond order,
            unresolved crystallographic operation, or conflicting or ambiguous atom identities.
    """

    source = _require_cif_source(cif_file.source)
    block = source.block
    placement_index = PlacementIndex(block, placements)
    if declared_branch_bonds is None:
        declared_branch_bonds = {}
    if "struct_conn" not in block:
        return ResolvedConnections()
    category = block["struct_conn"]
    from ._placed_atoms import PlacedAtoms

    deposited_rows = source.atoms.by_index(cif_file.model_index)
    native_sites, observed_rows_by_site = source_sites.sites, source_sites.observed_rows
    native_components = {site: str(key[-2]) for key, site in native_sites.entries("label")}
    model_ids = {int(row) for row in model_rows.row_index}
    model_positions = {i for i, row in enumerate(deposited_rows.row_index) if int(row) in model_ids}
    native_observations: dict[tuple[object, ...], set[str]] = {}
    for row in model_rows:
        key = (row.chain_id, row.label_seq_id, row.auth_residue_id, row.insertion_code, row.component_id)
        native_observations.setdefault(key, set()).add(str(row.atom_name))
    source_positions = {int(row_id): position for position, row_id in enumerate(deposited_rows.row_index)}
    selected_ids = (
        selected_rows.base.row_index[selected_rows.source_positions]
        if isinstance(selected_rows, PlacedAtoms)
        else selected_rows.row_index
    )
    selected_source_positions = frozenset(source_positions[int(row_id)] for row_id in selected_ids)

    conn_types = tuple(connection.kind for connection in source.connections)
    alt_1 = tuple(connection.first.altloc for connection in source.connections)
    alt_2 = tuple(connection.second.altloc for connection in source.connections)
    leaving_flags = _cif_column(category, "pdbx_leaving_atom_flag")
    leaving_ids_1 = _cif_column(category, "pdbx_ptnr1_leaving_atom_id")
    leaving_ids_2 = _cif_column(category, "pdbx_ptnr2_leaving_atom_id")
    roles_1 = _cif_column(category, "ptnr1_role")
    roles_2 = _cif_column(category, "ptnr2_role")
    symmetry_1 = _cif_column(category, "ptnr1_symmetry")
    symmetry_2 = _cif_column(category, "ptnr2_symmetry")
    # PDBx v50 declares default "sing" for an absent item. Explicit CIF nulls
    # remain unknown/inapplicable; neither is a declared single bond.
    # https://mmcif.wwpdb.org/dictionaries/mmcif_pdbx_v50.dic/Items/_struct_conn.pdbx_value_order.html
    value_orders = (
        _cif_column(category, "pdbx_value_order") if "pdbx_value_order" in category else ("sing",) * category.row_count
    )
    element_by_site: dict[AtomSite, str] = {}
    altloc_by_site: dict[AtomSite, str | None] = {}
    observed_atom_id_sets: dict[tuple[str, int], set[str]] = {}
    for row_id, altloc, observed, element in zip(
        selected_rows.row_index,
        selected_rows.altloc_id,
        selected_rows.coordinate_present,
        selected_rows.element,
        strict=True,
    ):
        site = site_by_row_index[int(row_id)]
        altloc_by_site[site] = altloc
        if observed:
            observed_atom_id_sets.setdefault((site.instance_id, site.component_index), set()).add(site.atom_id)
        if element:
            element_by_site[site] = str(element)
    observed_atom_ids_by_occurrence = {
        occurrence: frozenset(atom_ids) for occurrence, atom_ids in observed_atom_id_sets.items()
    }
    component_elements = (
        {}
        if component_table is None
        else {
            component_id: {atom.atom_id: atom.element for atom in component.atoms}
            for component_id, component in component_table.items()
        }
    )
    for source_key, site in site_index.entries("label"):
        component_id, atom_id = source_key[-2:]
        element = component_elements.get(str(component_id), {}).get(str(atom_id))
        if element is not None:
            element_by_site.setdefault(site, element)

    def candidate_leaving_groups(
        component: ChemicalComponent,
        endpoint: str,
        known_non_leaving: frozenset[str],
        *,
        include_unmarked: bool,
    ) -> set[frozenset[str]]:
        assert component.bonds is not None
        atom_ids = tuple(atom.atom_id for atom in component.atoms)
        bond_pairs = tuple((bond.atom_a, bond.atom_b) for bond in component.bonds)
        possible_leaving = component.leaving_atom_ids
        if not possible_leaving and include_unmarked:
            possible_leaving = frozenset(atom_ids) - known_non_leaving - {endpoint}
        candidates = set(terminal_leaving_groups(atom_ids, bond_pairs, possible_leaving, endpoint))
        for attachment in (component.previous_attachment, component.next_attachment):
            if (
                attachment is None
                or attachment.atom_id != endpoint
                or not attachment.leaving_atom_ids
                or attachment.leaving_atom_ids & known_non_leaving
            ):
                continue
            groups = terminal_leaving_groups(atom_ids, bond_pairs, attachment.leaving_atom_ids, endpoint)
            if len(groups) != 1:
                raise ValueError(f"component {component.component_id!r} has an ambiguous attachment leaving group")
            candidates.add(groups[0])
        return candidates

    def explicit_leaving_group(
        component: ChemicalComponent,
        endpoint: str,
        leaving_atom: str,
        candidates: set[frozenset[str]],
        known_non_leaving: frozenset[str],
        *,
        chemistry_complete: bool,
    ) -> frozenset[str]:
        atom_by_id = {atom.atom_id: atom for atom in component.atoms}
        if leaving_atom not in atom_by_id or leaving_atom == endpoint:
            raise ValueError(
                f"mmCIF struct_conn endpoint {component.component_id!r}:{endpoint} "
                f"has invalid leaving atom {leaving_atom!r}"
            )
        if leaving_atom in known_non_leaving:
            raise ValueError(
                f"mmCIF struct_conn endpoint {component.component_id!r}:{endpoint} "
                f"declares known non-leaving atom {leaving_atom!r} as leaving"
            )
        if not chemistry_complete or component.bonds is None:
            raise ValueError(
                f"mmCIF struct_conn endpoint {component.component_id!r}:{endpoint} "
                "has incomplete chemistry for its leaving atom"
            )
        matches = {group for group in candidates if leaving_atom in group}
        if len(matches) > 1:
            raise ValueError(
                f"mmCIF struct_conn endpoint {component.component_id!r}:{endpoint} "
                "has an ambiguous component leaving group"
            )
        if matches:
            return next(iter(matches))

        assert component.bonds is not None
        groups = terminal_leaving_groups(
            tuple(atom_by_id),
            ((bond.atom_a, bond.atom_b) for bond in component.bonds),
            {leaving_atom},
            endpoint,
        )
        if len(groups) != 1:
            raise ValueError(
                f"mmCIF struct_conn endpoint {component.component_id!r}:{endpoint} "
                f"has invalid leaving atom {leaving_atom!r}"
            )
        return groups[0]

    def unobserved_leaving_group(
        component: ChemicalComponent,
        endpoint: str,
        group: frozenset[str],
        observed_atom_ids: frozenset[str],
    ) -> frozenset[str]:
        r"""Relabel only graph-equivalent terminal hydrogens to avoid deleting observations."""

        observed_leaving = group & observed_atom_ids
        if not observed_leaving:
            return group
        if component.bonds is None:
            raise ValueError(f"component {component.component_id!r} has no known bond graph")
        atom_by_id = {atom.atom_id: atom for atom in component.atoms}
        incident: dict[str, list[ChemicalBond]] = {atom.atom_id: [] for atom in component.atoms}
        stereo_atoms = {
            atom_id for bond in component.bonds if bond.stereo_atoms is not None for atom_id in bond.stereo_atoms
        }
        for bond in component.bonds:
            incident[bond.atom_a].append(bond)
            incident[bond.atom_b].append(bond)

        def signature(atom_id: str) -> tuple[object, ...] | None:
            atom = atom_by_id[atom_id]
            bonds = incident[atom_id]
            if (
                atom.element not in HYDROGEN_ELEMENTS
                or atom.stereo is not None
                or atom_id in stereo_atoms
                or len(bonds) != 1
            ):
                return None
            bond = bonds[0]
            neighbor = bond.atom_b if bond.atom_a == atom_id else bond.atom_a
            if neighbor != endpoint or bond.stereo is not None or bond.stereo_atoms is not None:
                return None
            return (
                atom.element,
                atom.isotope,
                atom.formal_charge,
                atom.radical_electrons,
                atom.explicit_hydrogens,
                atom.implicit_hydrogens,
                atom.implicit_hydrogens_allowed,
                atom.aromatic,
                bond.order,
                bond.aromatic,
                bond.kind,
                bond.direction,
            )

        replacement = set(group)
        available = set(atom_by_id) - set(group) - set(observed_atom_ids)
        for atom_id in sorted(observed_leaving):
            atom_signature = signature(atom_id)
            alternatives = sorted(candidate for candidate in available if signature(candidate) == atom_signature)
            if atom_signature is None or not alternatives:
                raise ValueError(
                    f"component {component.component_id!r} leaving group contains observed atom {atom_id!r}"
                )
            replacement.remove(atom_id)
            replacement.add(alternatives[0])
            available.remove(alternatives[0])
        return frozenset(replacement)

    def leaving_groups(
        index: int,
        conn_type: str,
        leaving_flag: str | None,
        atom_ids: Sequence[str],
        component_values: Sequence[ChemicalComponent | None],
        observed_values: Sequence[frozenset[str]],
    ) -> tuple[frozenset[str], ...]:
        explicit_leaving = (leaving_ids_1[index], leaving_ids_2[index])
        leaving_sets: list[frozenset[str]] = []
        candidate_sets: list[set[frozenset[str]]] = []
        retained_atom_ids: list[frozenset[str]] = []
        endpoint_components: list[ChemicalComponent | None] = []
        observed_atom_ids: list[frozenset[str]] = []
        for endpoint_number, (atom_id, component, leaving_id) in enumerate(
            zip(atom_ids, component_values, explicit_leaving),
            start=1,
        ):
            component_id = None if component is None else component.component_id
            endpoint_components.append(component)
            chemistry_complete = (
                component is not None
                and component_topology_complete.get(component.component_id, False)
                and component.bonds is not None
                and all(atom.element != "X" for atom in component.atoms)
            )
            observed_atom_ids.append(observed_values[endpoint_number - 1])
            retained_atom_ids.append(
                frozenset() if component is None else frozenset(atom.atom_id for atom in component.atoms)
            )
            known_non_leaving = (
                frozenset() if component_id is None else source_non_leaving_atom_ids.get(component_id, frozenset())
            )
            leaving_candidates = (
                set()
                if component is None or not chemistry_complete
                else candidate_leaving_groups(
                    component,
                    atom_id,
                    known_non_leaving,
                    include_unmarked=leaving_id is not None or leaving_flag in {"one", "both"},
                )
            )
            if conn_type.startswith("disulf") and leaving_id is None and chemistry_complete:
                assert component is not None and component.bonds is not None
                # A disulfide condenses away each cysteine's thiol hydrogen. When that
                # hydrogen is actually modelled -- a reduced-cysteine structure whose
                # declared "disulfide" is not a real S-S contact -- it is an observation,
                # not a leaving atom: excluding it here keeps the read deterministic (the
                # candidate never reaches the observed-leaving check) and preserves the
                # modelled atom, while an unmodelled thiol hydrogen still leaves as usual.
                hydrogen_ids = {
                    atom.atom_id
                    for atom in component.atoms
                    if atom.element in HYDROGEN_ELEMENTS and atom.atom_id not in observed_atom_ids[-1]
                }
                leaving_candidates.update(
                    terminal_leaving_groups(
                        tuple(atom.atom_id for atom in component.atoms),
                        ((bond.atom_a, bond.atom_b) for bond in component.bonds),
                        hydrogen_ids,
                        atom_id,
                    )
                )
            if leaving_id is None and component is not None:
                viable_candidates: set[frozenset[str]] = set()
                for candidate in leaving_candidates:
                    try:
                        viable_candidates.add(
                            unobserved_leaving_group(
                                component,
                                atom_id,
                                candidate,
                                observed_atom_ids[-1],
                            )
                        )
                    except ValueError:
                        if conn_type.startswith("disulf"):
                            raise
                        continue
                leaving_candidates = viable_candidates
            candidate_sets.append(leaving_candidates)
            if leaving_id is None:
                leaving_sets.append(frozenset())
            elif component is None:
                raise ValueError(
                    f"mmCIF struct_conn row {index} endpoint {endpoint_number} "
                    "has a leaving atom without a component definition"
                )
            else:
                leaving_sets.append(
                    explicit_leaving_group(
                        component,
                        atom_id,
                        leaving_id,
                        leaving_candidates,
                        known_non_leaving,
                        chemistry_complete=chemistry_complete,
                    )
                )

        explicit_count = sum(bool(values) for values in leaving_sets)
        if leaving_flag == "none" and explicit_count:
            raise ValueError(f"mmCIF struct_conn row {index} contradicts its leaving-atom flag")
        if leaving_flag is None:
            for endpoint_index, candidates in enumerate(candidate_sets):
                projected = {values & retained_atom_ids[endpoint_index] for values in candidates}
                if len(projected) > 1:
                    raise ValueError(
                        f"mmCIF struct_conn row {index} cannot identify endpoint "
                        f"{endpoint_index + 1}'s leaving group"
                    )
                if projected and not leaving_sets[endpoint_index]:
                    leaving_sets[endpoint_index] = next(iter(projected))
        elif leaving_flag == "both":
            for endpoint_index in range(2):
                if leaving_sets[endpoint_index]:
                    continue
                candidates = candidate_sets[endpoint_index]
                projected = {values & retained_atom_ids[endpoint_index] for values in candidates}
                if not candidates or len(projected) != 1:
                    raise ValueError(
                        f"mmCIF struct_conn row {index} cannot identify endpoint "
                        f"{endpoint_index + 1}'s leaving group"
                    )
                leaving_sets[endpoint_index] = next(iter(projected))
        elif leaving_flag == "one":
            if explicit_count > 1:
                raise ValueError(f"mmCIF struct_conn row {index} contradicts its leaving-atom flag")
            if explicit_count == 0:
                inferred = {
                    (
                        group & retained_atom_ids[0] if endpoint_index == 0 else frozenset(),
                        group & retained_atom_ids[1] if endpoint_index == 1 else frozenset(),
                    )
                    for endpoint_index, candidates in enumerate(candidate_sets)
                    for group in candidates
                }
                if not any(candidate_sets) or len(inferred) != 1:
                    raise ValueError(f"mmCIF struct_conn row {index} cannot identify its leaving group")
                leaving_sets = list(next(iter(inferred)))
        leaving_sets = [values & retained for values, retained in zip(leaving_sets, retained_atom_ids)]
        for endpoint_index, component in enumerate(endpoint_components):
            observed_leaving = leaving_sets[endpoint_index] & observed_atom_ids[endpoint_index]
            if not observed_leaving:
                continue
            if explicit_leaving[endpoint_index] is not None:
                atom_id = sorted(observed_leaving)[0]
                raise ValueError(
                    f"mmCIF struct_conn row {index} endpoint {endpoint_index + 1} "
                    f"declares observed leaving atom {atom_id!r}"
                )
            assert component is not None
            leaving_sets[endpoint_index] = unobserved_leaving_group(
                component,
                atom_ids[endpoint_index],
                leaving_sets[endpoint_index],
                observed_atom_ids[endpoint_index],
            )
        return tuple(leaving_sets)

    def endpoint_asym(endpoint: Endpoint, matches: Sequence[int], row_index: int) -> str:
        if endpoint.label[0] is not None:
            return str(endpoint.label[0])
        if matches:
            candidates = {deposited_rows.chain_id[position] for position in matches}
        else:
            candidates = native_sites.asym_ids(native_sites.resolve(endpoint))
        if len(candidates) != 1:
            raise ValueError(f"mmCIF struct_conn row {row_index} cannot uniquely identify its label-asym placement")
        return str(next(iter(candidates)))

    bonds: list[SystemBond] = []
    boundary_leaving: dict[AtomSite, frozenset[str]] = {}
    boundary_evidence: dict[tuple[AtomSite, object], list[tuple[np.ndarray, tuple[object, ...]]]] = {}
    for index, raw_conn_type in enumerate(conn_types):
        if raw_conn_type is None:
            raise ValueError(f"mmCIF struct_conn row {index} has no connection type")
        conn_type = raw_conn_type.casefold()
        if conn_type in {"hydrog", "mismat", "saltbr"}:
            continue
        if not (
            conn_type.startswith("covale")
            or conn_type.startswith("disulf")
            or conn_type.startswith("metalc")
            or conn_type == "modres"
        ):
            raise ValueError(f"mmCIF struct_conn row {index} has unsupported connection type {raw_conn_type!r}")
        value_order = value_orders[index]
        if value_order is not None and value_order.lower() == "arom":
            raise ValueError("mmCIF struct_conn pdbx_value_order does not support aromatic bonds")
        order, aromatic = _cif_bond_order(value_order)
        # `struct_conn` has no aromatic-flag column, so unlike `chem_comp_bond` its order
        # is the only channel: an explicit non-aromatic order states non-aromaticity.
        if order is not None and aromatic is None:
            aromatic = False
        if conn_type.startswith("disulf") and order is None:
            order, aromatic = 1.0, False
        direction: BondDirection | None = "undirected"
        if conn_type.startswith("metalc"):
            direction_evidence: set[BondDirection] = set()
            for role_endpoint, role in enumerate(
                (roles_1[index], roles_2[index]),
                start=1,
            ):
                normalized = None if role is None else role.casefold()
                if normalized in {"donor", "metal coordination"}:
                    direction_evidence.add("a_to_b" if role_endpoint == 1 else "b_to_a")
                elif normalized in {"acceptor", "metal"}:
                    direction_evidence.add("b_to_a" if role_endpoint == 1 else "a_to_b")
            direction = direction_evidence.pop() if len(direction_evidence) == 1 else None
        connection = source.connections[index]
        endpoint_1, auth_endpoint_1 = connection.first.label, connection.first.auth
        endpoint_2, auth_endpoint_2 = connection.second.label, connection.second.auth

        native_matches = tuple(native_sites.resolve(endpoint) for endpoint in (connection.first, connection.second))
        endpoint_matches = tuple(
            tuple(
                sorted(
                    position
                    for site in matches
                    for position in observed_rows_by_site.get(site, ())
                    if endpoint.altloc is None or deposited_rows.altloc_id[position] == endpoint.altloc
                )
            )
            for endpoint, matches in zip((connection.first, connection.second), native_matches, strict=True)
        )
        selected_matches = tuple(bool(selected_source_positions.intersection(matches)) for matches in endpoint_matches)
        raw_leaving_flag = leaving_flags[index]
        leaving_flag = None if raw_leaving_flag is None else raw_leaving_flag.casefold()
        if leaving_flag not in {None, "none", "one", "both"}:
            raise ValueError(f"unsupported mmCIF struct_conn leaving-atom flag: {leaving_flags[index]!r}")

        # Validate source identity first, then omit reactions of a component or
        # alternate removed from the complete model, before chain projection.
        if any(
            matches and not any(site.component_index in model_occurrences for site in matches)
            for matches in native_matches
        ) or any(matches and not model_positions.intersection(matches) for matches in endpoint_matches):
            continue
        asym_a = endpoint_asym(connection.first, endpoint_matches[0], index)
        asym_b = endpoint_asym(connection.second, endpoint_matches[1], index)
        placement_pairs = placement_index.pairs(asym_a, asym_b, symmetry_1[index], symmetry_2[index])
        paired_a, paired_b = {a for a, _ in placement_pairs}, {b for _, b in placement_pairs}
        jobs = (
            *((a, b, True, True) for a, b in placement_pairs),
            *((copy, None, True, False) for copy in placement_index.by_asym[asym_a] if copy not in paired_a),
            *((None, copy, False, True) for copy in placement_index.by_asym[asym_b] if copy not in paired_b),
        )
        for endpoint_symmetry_a, endpoint_symmetry_b, placed_a, placed_b in jobs:
            resolved_sites: list[AtomSite | None] = []
            projected_endpoints: list[bool] = []
            for endpoint_number, (endpoint_symmetry, label_endpoint, auth_endpoint, required_altloc) in enumerate(
                (
                    (endpoint_symmetry_a, endpoint_1, auth_endpoint_1, alt_1[index]),
                    (endpoint_symmetry_b, endpoint_2, auth_endpoint_2, alt_2[index]),
                ),
                start=1,
            ):
                endpoint = Endpoint(label_endpoint, auth_endpoint, required_altloc)
                placed = (placed_a, placed_b)[endpoint_number - 1]
                source_match = bool(endpoint_matches[endpoint_number - 1])
                selected_match = selected_matches[endpoint_number - 1]
                projected_endpoints.append(not placed or (source_match and not selected_match))
                if not placed:
                    resolved_sites.append(None)
                    continue
                if required_altloc is not None and not selected_match:
                    resolved_sites.append(None)
                    continue
                resolved_site = site_index.unique(endpoint, symmetry=endpoint_symmetry)
                if required_altloc is not None and (
                    resolved_site is None or altloc_by_site.get(resolved_site) != required_altloc
                ):
                    resolved_site = None
                resolved_sites.append(resolved_site)
            site_a, site_b = resolved_sites
            if site_a is None or site_b is None:
                unresolved = [
                    endpoint + 1
                    for endpoint, (site, projected) in enumerate(zip(resolved_sites, projected_endpoints))
                    if site is None and not projected
                ]
                if unresolved:
                    endpoints = ", ".join(str(endpoint) for endpoint in unresolved)
                    raise ValueError(
                        f"mmCIF struct_conn row {index} cannot resolve endpoint {endpoints} "
                        "in the selected molecular declaration"
                    )
                if site_a is None and site_b is None:
                    continue
                atom_ids: list[str] = []
                component_values: list[ChemicalComponent | None] = []
                observations: list[frozenset[str]] = []
                references: list[object] = []
                for endpoint, matches, endpoint_site in zip(
                    (connection.first, connection.second), endpoint_matches, resolved_sites, strict=True
                ):
                    if endpoint_site is not None:
                        reference: object = endpoint_site
                        component_id = occurrence_components[
                            endpoint_site.instance_id, endpoint_site.component_index
                        ].component_id
                        atom_id = endpoint_site.atom_id
                        observed = observed_atom_ids_by_occurrence.get(
                            (endpoint_site.instance_id, endpoint_site.component_index), frozenset()
                        )
                    elif matches:
                        identities = {
                            (str(deposited_rows.component_id[i]), str(deposited_rows.atom_name[i]))
                            for i in matches
                            if i in model_positions
                        }
                        if len(identities) != 1:
                            raise ValueError(f"mmCIF struct_conn row {index} has an ambiguous boundary endpoint")
                        component_id, atom_id = next(iter(identities))
                        occurrences = {
                            (
                                deposited_rows.chain_id[i],
                                deposited_rows.label_seq_id[i],
                                deposited_rows.auth_residue_id[i],
                                deposited_rows.insertion_code[i],
                                deposited_rows.component_id[i],
                            )
                            for i in matches
                            if i in model_positions
                        }
                        if len(occurrences) != 1:
                            raise ValueError(f"mmCIF struct_conn row {index} has an ambiguous boundary occurrence")
                        observed = frozenset(native_observations.get(next(iter(occurrences)), ()))
                        reference = (*next(iter(occurrences)), atom_id)
                    else:
                        declared = native_sites.unique(endpoint)
                        if declared is None:
                            raise ValueError(f"mmCIF struct_conn row {index} has an unknown boundary endpoint")
                        component_id, atom_id = native_components[declared], declared.atom_id
                        reference = declared
                        observed = frozenset()
                    atom_ids.append(atom_id)
                    component_values.append(None if component_table is None else component_table.get(component_id))
                    observations.append(observed)
                    references.append(reference)
                leaving = leaving_groups(
                    index, conn_type, leaving_flag, tuple(atom_ids), tuple(component_values), tuple(observations)
                )
                frames = placement_index.frames.pair(symmetry_1[index], symmetry_2[index])
                for side, (endpoint_site, names) in enumerate(zip(resolved_sites, leaving, strict=True)):
                    if endpoint_site is not None:
                        relative_frame = np.linalg.inv(frames[side][1]) @ frames[1 - side][1]
                        oriented_direction = direction
                        if side and direction in {"a_to_b", "b_to_a"}:
                            oriented_direction = "b_to_a" if direction == "a_to_b" else "a_to_b"
                        evidence = (
                            order,
                            aromatic,
                            conn_type.startswith("metalc"),
                            oriented_direction,
                            names,
                            leaving[1 - side],
                        )
                        records = boundary_evidence.setdefault((endpoint_site, references[1 - side]), [])
                        for frame, previous in records:
                            if not np.allclose(frame, relative_frame, rtol=0, atol=1e-8):
                                continue
                            if previous[4:] != evidence[4:] or any(
                                a is not None and b is not None and a != b for a, b in zip(previous[:4], evidence[:4])
                            ):
                                raise ValueError("conflicting declarations for a projected mmCIF connection")
                        records.append((relative_frame, evidence))
                        boundary_leaving[endpoint_site] = boundary_leaving.get(endpoint_site, frozenset()) | names
                continue
            declared_branch = declared_branch_bonds.get(frozenset((site_a, site_b)))
            if direction is None:
                # The archive rarely populates the partner-role columns, so fall back to
                # chemistry: a coordination bond runs from the donor ligand atom to the
                # metal center, and the elements name which endpoint is which. Neither or
                # both being a metal leaves the direction genuinely unknown.
                metal_a = _is_metal_site(site_a, element_by_site)
                metal_b = _is_metal_site(site_b, element_by_site)
                if metal_b and not metal_a:
                    direction = "a_to_b"
                elif metal_a and not metal_b:
                    direction = "b_to_a"
                else:
                    raise ValueError(
                        f"mmCIF struct_conn row {index} declares a metal coordination "
                        "whose donor direction cannot be identified"
                    )

            if declared_branch is not None:
                declared_branch_bond, leaving_a_known, leaving_b_known = declared_branch
                declared_direction = declared_branch_bond.direction
                declared_leaving = (declared_branch_bond.leaving_a, declared_branch_bond.leaving_b)
                declared_leaving_known = (leaving_a_known, leaving_b_known)
                if declared_branch_bond.site_a == site_b and declared_branch_bond.site_b == site_a:
                    declared_leaving = declared_leaving[::-1]
                    declared_leaving_known = declared_leaving_known[::-1]
                    if declared_direction == "a_to_b":
                        declared_direction = "b_to_a"
                    elif declared_direction == "b_to_a":
                        declared_direction = "a_to_b"
                elif declared_branch_bond.site_a != site_a or declared_branch_bond.site_b != site_b:
                    raise ValueError(f"mmCIF struct_conn row {index} duplicates an ambiguous branch link")
                declared_semantics = (
                    declared_branch_bond.order,
                    declared_branch_bond.aromatic,
                    declared_branch_bond.kind,
                    declared_direction,
                )
                decoded_semantics = (
                    order,
                    aromatic,
                    "dative" if conn_type.startswith("metalc") else "covalent",
                    direction,
                )
                semantic_matches = (
                    (declared_semantics[0] is None or order is None or declared_semantics[0] == order)
                    and (declared_semantics[1] is None or aromatic is None or declared_semantics[1] == aromatic)
                    and declared_semantics[2:] == decoded_semantics[2:]
                )
                explicit_leaving = (leaving_ids_1[index], leaving_ids_2[index])
                explicit_matches = all(
                    leaving_id is None or not known or leaving_id in group
                    for leaving_id, group, known in zip(
                        explicit_leaving,
                        declared_leaving,
                        declared_leaving_known,
                        strict=True,
                    )
                )
                known_leaving_count = sum(
                    bool(group) for group, known in zip(declared_leaving, declared_leaving_known) if known
                )
                flag_matches = (
                    leaving_flag is None
                    or (all(declared_leaving_known) and leaving_flag == ("none", "one", "both")[known_leaving_count])
                    or (
                        not all(declared_leaving_known)
                        and not (known_leaving_count and leaving_flag == "none")
                        and not (known_leaving_count == 2 and leaving_flag == "one")
                    )
                )
                if not semantic_matches or not flag_matches or not explicit_matches:
                    raise ValueError(
                        f"mmCIF struct_conn row {index} conflicts with its branched-entity link: "
                        f"declared semantics {declared_semantics!r}, decoded semantics {decoded_semantics!r}, "
                        f"declared leaving {declared_leaving!r} known {declared_leaving_known!r}, "
                        f"flag {leaving_flag!r}, "
                        f"explicit leaving {explicit_leaving!r}"
                    )
                bonds.append(
                    SystemBond(
                        site_a,
                        site_b,
                        order=declared_branch_bond.order if order is None else order,
                        aromatic=declared_branch_bond.aromatic if aromatic is None else aromatic,
                        role="entity",
                        kind=declared_branch_bond.kind,
                        direction=declared_direction,
                    )
                )
                continue

            leaving_sets = leaving_groups(
                index,
                conn_type,
                leaving_flag,
                (site_a.atom_id, site_b.atom_id),
                tuple(
                    (
                        None
                        if component_table is None
                        else component_table.get(
                            occurrence_components[site.instance_id, site.component_index].component_id
                        )
                    )
                    for site in (site_a, site_b)
                ),
                tuple(
                    observed_atom_ids_by_occurrence.get((site.instance_id, site.component_index), frozenset())
                    for site in (site_a, site_b)
                ),
            )
            decoded_bond = SystemBond(
                site_a,
                site_b,
                order=order,
                aromatic=aromatic,
                kind="dative" if conn_type.startswith("metalc") else "covalent",
                direction=direction,
                role="assembly",
                leaving_a=leaving_sets[0],
                leaving_b=leaving_sets[1],
            )
            bonds.append(decoded_bond)
    return ResolvedConnections(tuple(bonds), boundary_leaving)
